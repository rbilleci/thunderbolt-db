//! General GPU executor — the `Expr` IR and its device interpreter (Charter rule 2;
//! `docs/architecture/17-general-gpu-executor.md`). This is the GENERAL path that replaces the
//! enumerated `execute_relational_*_with_resident_device_memory_probe` shape methods: a query is an
//! expression tree, lowered to a pipeline of device primitives, NOT matched against a fixed catalog
//! of shapes.
//!
//! `ResidentExpr` is the general scalar IR (it can represent any int4 arithmetic/comparison/boolean
//! tree). The interpreter's *coverage* grows node-by-node; today `lower_resident_predicate` lowers
//! the prototype shape `Compare(arith(Column, Column), Int4Literal)` to the composed buffer->buffer
//! primitive (`expr_filter_two_col_compare_from_payload`) and materializes the projected int4
//! columns by gathering the surviving rows. Fuller trees (deeper arithmetic, AND/OR over masks,
//! column-vs-column compares) land as the device bytecode VM in §2.3 of the design doc — by
//! extending this interpreter, never by adding a new shape method.
//!
//! The IR + op-code maps + `execute_resident_expr_select_with_binding` are now the production path the
//! SQL->Expr binding (`engine_sql_pg`) routes into; the GPU parity tests exercise the same lowering
//! via programmatic `ResidentExpr`s. The 2-arg `execute_resident_expr_select` convenience wrapper (run
//! a select with a programmatic predicate, binding internally) has no production caller yet — a
//! prepared-route / facade caller is the likely one — so it carries a targeted `allow(dead_code)`.

use super::*;

/// A binary operator in the resident expression IR (arithmetic, comparison, or boolean). The
/// interpreter pattern-matches these; callers (tests now, the parser/planner later) construct them.
/// Not all variants have interpreter coverage yet (the device bytecode VM, design §2.3, adds
/// AND/OR/Ne/col-vs-col).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentBinaryOp {
    Add,
    Sub,
    Mul,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    /// SQL `LIKE`: the lhs is a text column, the rhs a [`ResidentExpr::TextLiteral`] pattern (`%` =
    /// any run, `_` = any one character). Byte-wise / UTF-8-char-aware (the type matrix, doc 19).
    Like,
}

/// The general scalar-expression IR the GPU interpreter evaluates. Column fields are table column
/// indices; the interpreter resolves device byte-offsets. Grows by node (literals of other types,
/// casts, unary ops, `LIKE`, ...), never by enumerating whole query shapes.
#[derive(Debug, Clone)]
pub(crate) enum ResidentExpr {
    Column(usize),
    Int4Literal(i32),
    /// A numeric (DECIMAL) literal as its [`Decimal128`] (mantissa + scale). Compared by rescaling to
    /// the target column's scale at lowering time (the type matrix, doc 19). An integer literal
    /// compared to a numeric column arrives as `Int4Literal` and is coerced to numeric on that path.
    NumericLiteral(Decimal128),
    /// A text (`text`/`varchar`) literal -- the comparison/LIKE value for a text column. Compared
    /// byte-wise (deterministic-collation equality is byte-identity; the type matrix, doc 19).
    TextLiteral(String),
    /// A boolean literal (`true`/`false`) -- the comparison value for `flag = true` / `flag = false`
    /// (the type matrix, doc 19). A bool column is a bitmap, so the comparison lowers to the
    /// bitmap->mask kernel with the appropriate `negate`.
    BoolLiteral(bool),
    /// `col IS NULL` / `col IS NOT NULL` (`is_not_null` selects which). A unary predicate LEAF over a
    /// column's NULL validity bitmap (M3 -- doc 21): it lowers to the SAME `gpu_db_resident_bool_to_mask`
    /// kernel as a bool column, pointed at the column's validity bitmap (1 = valid/present), with
    /// `negate = !is_not_null`. A column with no validity bitmap (no NULLs) lowers to an all-constant
    /// mask. Not an arithmetic operand -- the arith/compare paths reject it.
    IsNull { col: usize, is_not_null: bool },
    Binary {
        op: ResidentBinaryOp,
        lhs: Box<ResidentExpr>,
        rhs: Box<ResidentExpr>,
    },
}

/// A column reference inside a JOIN (ON condition or projection): a bare column (`qualifier: None`) or
/// a qualified `alias.column` (`qualifier: Some(alias)`). Resolved to a specific relation + column
/// index in the join executor (against the two bound tables).
#[derive(Debug, Clone)]
pub(crate) struct JoinColRef {
    pub qualifier: Option<String>,
    pub column: String,
}

/// A SELECT-list item in a JOIN projection: a single column, or a `*` star expanded in the executor to
/// every column of the named relation (`Star(Some(alias))` = `alias.*`) or of BOTH relations, left then
/// right (`Star(None)` = bare `*`).
#[derive(Debug, Clone)]
pub(crate) enum JoinProjItem {
    Column(JoinColRef),
    Star(Option<String>),
}

/// One relation in a JOIN's FROM clause: its base name + the alias columns are qualified by (the alias,
/// or the relation name when unaliased -- mirrors the single-table builder).
#[derive(Debug, Clone)]
pub(crate) struct JoinRelationRef {
    pub table: String,
    pub alias: String,
}

/// Sentinel carried in a join's per-relation index vectors meaning "no row -> emit NULL for this
/// relation's columns" -- a LEFT OUTER join's NULL pad for an unmatched left row (M3 -- doc 21). A real
/// absolute row index can never be `u32::MAX` (residency row counts are far smaller), so it is unambiguous.
const JOIN_NULL_ROW: u32 = u32::MAX;

/// One INNER-join step in a left-deep chain. Its condition is one of: explicit ON `conjuncts`
/// (`a.k1=b.k1 [AND a.k2=b.k2]` -- in each pair one operand resolves to the newly joined relation
/// `relations[k+1]`, the other to an accumulated one); `USING(cols)`, which the parser desugars to
/// qualified `conjuncts` AND records the join column names in `coalesce` (they appear ONCE in `*`); or
/// `NATURAL` (`natural=true`, `conjuncts` empty), where the executor joins on -- and coalesces -- the
/// relations' common column names. A single join column is a plain equi-join; 2 are a composite key
/// (packed into one i64); >2 (a key wider than 64 bits) are a follow-up. USING/NATURAL are 2-relation only.
#[derive(Debug, Clone)]
pub(crate) struct JoinStep {
    pub conjuncts: Vec<(JoinColRef, JoinColRef)>,
    /// `true` for a NATURAL join (the executor derives the conjuncts + coalesce from common columns).
    pub natural: bool,
    /// USING/NATURAL join column names -- emitted ONCE in `*` and resolvable unqualified (else empty).
    pub coalesce: Vec<String>,
    /// OUTER-join flags (M3 -- doc 21), as a pair: `(outer_left, outer_right)` = (F,F) INNER, (T,F) LEFT,
    /// (F,T) RIGHT, (T,T) FULL. `outer_left` keeps every ACCUMULATED (left) row -- unmatched ones get the
    /// NEW relation NULL-padded; `outer_right` keeps every NEW (right) row -- unmatched ones get the
    /// accumulated relations NULL-padded. Currently 2-relation ON only (multi-way / OUTER NATURAL-USING /
    /// a WHERE on an outer join are follow-ups).
    pub outer_left: bool,
    pub outer_right: bool,
}

/// A LEFT-DEEP chain of INNER equi-joins parsed from the libpg_query FROM clause (M5). `relations` are in
/// left-deep order (`a JOIN b ON.. JOIN c ON..` => `[a, b, c]`); `steps[k]` is the ON that folds
/// `relations[k+1]` into the accumulated set `relations[0..=k]` (so `steps.len() == relations.len()-1`).
/// `projection` is the SELECT list (qualified/unqualified columns + `*` / `alias.*`). The executor
/// pipelines the chain: each intermediate result is materialized as a transient relation (the J5a bridge)
/// that the next step's GPU hash join probes -- no CPU relational join. A 2-relation join is the N=2 case
/// (one relation pair, one step), byte-identical to the original 2-way path.
#[derive(Debug, Clone)]
pub(crate) struct JoinPlan {
    pub relations: Vec<JoinRelationRef>,
    pub steps: Vec<JoinStep>,
    pub projection: Vec<JoinProjItem>,
    /// ORDER BY keys (plain columns only, each `(column, descending)`) applied to the join RESULT via a
    /// GPU sort. Every key must appear in `projection` (a non-projected ORDER BY key is a follow-up).
    pub order_by: Vec<(JoinColRef, bool)>,
    /// Parallel to `order_by`: the explicit NULLS FIRST/LAST override per key (M3 -- doc 21; `None` = PG
    /// default). Honored ON-DEVICE by the join-result GPU sort.
    pub order_by_nulls_first: Vec<Option<bool>>,
    /// LIMIT / OFFSET sliced off the (sorted) join result. `None` = unbounded / from row 0.
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// The device memory backing a join relation: a RESIDENT user table's published `Arc` (shared), or a
/// SYNTHESIZED catalog relation's freshly-uploaded TRANSIENT payload (owned for the query). `.mem()`
/// yields the `&CudaResidentDeviceMemory` the GPU pre-filter / key-projection / hash-join kernels run on.
enum JoinDeviceMemory {
    Resident(std::sync::Arc<gpu_db_execution::CudaResidentDeviceMemory>),
    Transient(gpu_db_execution::CudaResidentDeviceMemory),
}

impl JoinDeviceMemory {
    fn mem(&self) -> &gpu_db_execution::CudaResidentDeviceMemory {
        match self {
            JoinDeviceMemory::Resident(memory) => memory,
            JoinDeviceMemory::Transient(memory) => memory,
        }
    }
}

/// Sort `rows` ON THE GPU by `order` -- each `(result-column index, descending)` (charter: every relational
/// sort is a GPU sort, no host-side finalization). INT keys (int2/4/8/date/timestamp) feed an i64 matrix
/// (`bitonic_sort_multikey`); TEXT/NUMERIC/UUID keys feed a resident-like payload built from just those key
/// columns (`build_relational_device_payload`) that the hetero comparator reads on-device. `col_types[i]`
/// is result column `i`'s type. A no-op for <=1 row or empty `order`. `device_memory` supplies the CUDA
/// context. SHARED by the grouped result + the join result (both gather host rows then sort them on-device).
/// Map a HAVING/result-filter comparison operator to the resident-predicate IR operator.
fn having_op_to_resident(op: crate::SelectFilterOp) -> ResidentBinaryOp {
    use crate::SelectFilterOp;
    match op {
        SelectFilterOp::Eq => ResidentBinaryOp::Eq,
        SelectFilterOp::Lt => ResidentBinaryOp::Lt,
        SelectFilterOp::Lte => ResidentBinaryOp::Le,
        SelectFilterOp::Gt => ResidentBinaryOp::Gt,
        SelectFilterOp::Gte => ResidentBinaryOp::Ge,
        SelectFilterOp::LikePrefix => ResidentBinaryOp::Like,
    }
}

/// Normalize a legacy 1-aggregate grouped projection (`GroupedCount` / `GroupedSum` / `GroupedAvg` /
/// `GroupedMin` / `GroupedMax`, produced by the hand-rolled parser) to the general
/// [`SelectProjection::GroupedAggregates`] form the on-device Expr executor consumes -- the S8
/// `&Select`->general bridge. `GroupedCount { column }` groups by `column` with a COUNT(*) (no value
/// column); the others carry `(group_column, value_column)`. Returns `None` for a projection that is
/// not a legacy grouped form (already `GroupedAggregates`, or not grouped) -- the caller leaves it as-is.
fn grouped_projection_to_aggregates(projection: &SelectProjection) -> Option<SelectProjection> {
    let (group_column, aggregate) = match projection {
        SelectProjection::GroupedCount { column } => (
            column.clone(),
            GroupedAggregate {
                kind: GroupedAggKind::Count,
                value_column: None,
            },
        ),
        SelectProjection::GroupedSum {
            group_column,
            sum_column,
        } => (
            group_column.clone(),
            GroupedAggregate {
                kind: GroupedAggKind::Sum,
                value_column: Some(sum_column.clone()),
            },
        ),
        SelectProjection::GroupedAvg {
            group_column,
            avg_column,
        } => (
            group_column.clone(),
            GroupedAggregate {
                kind: GroupedAggKind::Avg,
                value_column: Some(avg_column.clone()),
            },
        ),
        SelectProjection::GroupedMin {
            group_column,
            min_column,
        } => (
            group_column.clone(),
            GroupedAggregate {
                kind: GroupedAggKind::Min,
                value_column: Some(min_column.clone()),
            },
        ),
        SelectProjection::GroupedMax {
            group_column,
            max_column,
        } => (
            group_column.clone(),
            GroupedAggregate {
                kind: GroupedAggKind::Max,
                value_column: Some(max_column.clone()),
            },
        ),
        _ => return None,
    };
    Some(SelectProjection::GroupedAggregates {
        group_column,
        aggregates: vec![aggregate],
    })
}

/// Reconstruct a faithful SQL `LIKE` pattern from the BARE literal prefix a `LikePrefix` bound filter
/// carries. The parser produces `LikePrefix` only for `LIKE '<prefix>%'` where `<prefix>` has no `%`/`_`
/// (sql/lib.rs `parse_select_like_prefix_filter`), and strips the trailing `%`, so the bound value is the
/// literal prefix bytes (a `\` is literal, not a LIKE escape, since `parse_sql_value` does not resolve LIKE
/// escapes). To match the legacy probe's literal-byte prefix match EXACTLY, escape every LIKE-special byte
/// in the prefix (`%`, `_`, `\`) and append a `%`: the resulting `LIKE '<escaped>%'` means "starts with the
/// literal prefix" — `compile_like_pattern` then yields the same byte-prefix semantics the probe had.
fn like_pattern_for_literal_prefix(prefix: &str) -> String {
    let mut pattern = Vec::with_capacity(prefix.len() + 1);
    for &byte in prefix.as_bytes() {
        if matches!(byte, b'%' | b'_' | b'\\') {
            pattern.push(b'\\');
        }
        pattern.push(byte);
    }
    pattern.push(b'%');
    // The prefix is valid UTF-8 and only ASCII escape bytes are inserted, so the result stays valid UTF-8.
    String::from_utf8(pattern).expect("ascii-escaped utf8 stays valid utf8")
}

/// Build the WHERE-predicate [`ResidentExpr`] DNF for the `&Select`->general grouped BRIDGE (S8) from a
/// bound select's resolved filters. Normalizes to the canonical DNF (OR of AND-groups) exactly as the
/// route classifiers do -- `filter_groups`, else `filters` as one AND-group, else the single `filter`,
/// else no predicate -- then turns each `(column_idx, op, value)` leaf into `Column(idx) <op> literal`
/// (AND within a group, OR across groups), mirroring the S3 HAVING DNF construction. This is the THIRD
/// predicate-construction path (vs the route classifier and `map_predicate_node`); it produces the SAME
/// `ResidentExpr` the SQL->Expr path's `map_predicate_node` builds for `WHERE col <op> 5` (an int4
/// `Column <cmp> Int4Literal`), so routing a grouped probe shape through it is behavior-preserving.
/// `None` = no WHERE (a full-table scan). The grouped routes only ever carry an int4 `Column <cmp>
/// Int4Literal` leaf (`resident_route_grouped_aggregate_shape`), but the general DNF here is robust to
/// any int4 OR-of-AND filter the bound may carry. A `LikePrefix` leaf becomes an on-device `LIKE` (see
/// [`like_pattern_for_literal_prefix`]).
fn resident_predicate_from_bound_filters(
    bound: &BoundRelationalSelect,
) -> Result<Option<ResidentExpr>, ExecuteError> {
    let groups: Vec<Vec<(usize, crate::SelectFilterOp, SqlValue)>> =
        if !bound.filter_groups.is_empty() {
            bound.filter_groups.clone()
        } else if !bound.filters.is_empty() {
            vec![bound.filters.clone()]
        } else if let Some(filter) = bound.filter.clone() {
            vec![vec![filter]]
        } else {
            return Ok(None);
        };
    let mut dnf: Option<ResidentExpr> = None;
    for group in &groups {
        let mut conj: Option<ResidentExpr> = None;
        for (idx, op, value) in group {
            // A `LikePrefix` filter carries the BARE literal prefix; reconstruct the faithful `LIKE
            // '<prefix>%'` pattern (escaped) so the bridge runs the SAME text-prefix match the retired probe
            // did — on-device via the general LIKE mask. Other ops build a `Column <cmp> literal` leaf.
            let leaf = if matches!(op, crate::SelectFilterOp::LikePrefix) {
                let SqlValue::Text(prefix) = value else {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "a LIKE-prefix filter requires a text prefix value".to_string(),
                    )));
                };
                ResidentExpr::Binary {
                    op: ResidentBinaryOp::Like,
                    lhs: Box::new(ResidentExpr::Column(*idx)),
                    rhs: Box::new(ResidentExpr::TextLiteral(like_pattern_for_literal_prefix(prefix))),
                }
            } else {
                ResidentExpr::Binary {
                    op: having_op_to_resident(*op),
                    lhs: Box::new(ResidentExpr::Column(*idx)),
                    // numeric_mode = false: a grouped-route filter constant is an int4 literal (the
                    // classifier guarantees `SqlValue::Int4`), matching `map_predicate_node`'s Int4Literal.
                    rhs: Box::new(having_value_to_resident_literal(value, false)?),
                }
            };
            conj = Some(match conj {
                None => leaf,
                Some(prev) => ResidentExpr::Binary {
                    op: ResidentBinaryOp::And,
                    lhs: Box::new(prev),
                    rhs: Box::new(leaf),
                },
            });
        }
        if let Some(c) = conj {
            dnf = Some(match dnf {
                None => c,
                Some(prev) => ResidentExpr::Binary {
                    op: ResidentBinaryOp::Or,
                    lhs: Box::new(prev),
                    rhs: Box::new(c),
                },
            });
        }
    }
    Ok(dnf)
}

/// A HAVING comparison constant (`SqlValue`) as a resident-predicate literal node, so the HAVING DNF can
/// be evaluated by the SAME device predicate VM as WHERE. The HAVING transient promotes every integer
/// column to ONE width (the VM is single-width); `numeric_mode` says which: in NUMERIC mode (the DNF
/// touches a numeric column) integer constants compare as `Numeric(scale 0)`; otherwise (all integer) they
/// ride the i32 literal (the int8 compare path sign-extends a small one). A genuine `Numeric` constant is
/// a numeric literal regardless. (A Timestamp/Uuid/Text/Bool HAVING constant is parser-rejected upstream;
/// the arm exists for safety, not reach.)
fn having_value_to_resident_literal(
    value: &SqlValue,
    numeric_mode: bool,
) -> Result<ResidentExpr, ExecuteError> {
    if numeric_mode {
        return Ok(match value {
            SqlValue::Int4(v) | SqlValue::Date(v) => {
                ResidentExpr::NumericLiteral(Decimal128::new(i128::from(*v), 0))
            }
            SqlValue::Int2(v) => ResidentExpr::NumericLiteral(Decimal128::new(i128::from(*v), 0)),
            SqlValue::Int8(v) => ResidentExpr::NumericLiteral(Decimal128::new(i128::from(*v), 0)),
            SqlValue::Numeric(d) => ResidentExpr::NumericLiteral(*d),
            other => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "HAVING comparison against a {other:?} constant is not yet on the GPU predicate path"
                ))));
            }
        });
    }
    Ok(match value {
        SqlValue::Int4(v) | SqlValue::Date(v) => ResidentExpr::Int4Literal(*v),
        SqlValue::Int2(v) => ResidentExpr::Int4Literal(i32::from(*v)),
        SqlValue::Int8(v) => match i32::try_from(*v) {
            Ok(small) => ResidentExpr::Int4Literal(small),
            Err(_) => ResidentExpr::NumericLiteral(Decimal128::new(i128::from(*v), 0)),
        },
        SqlValue::Numeric(d) => ResidentExpr::NumericLiteral(*d),
        SqlValue::Text(s) => ResidentExpr::TextLiteral(s.clone()),
        SqlValue::Bool(b) => ResidentExpr::BoolLiteral(*b),
        other => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "HAVING comparison against a {other:?} constant is not yet on the GPU predicate path"
            ))));
        }
    })
}

/// The ON-DEVICE sort PERMUTATION for `rows` by `order` ((result-column index, descending)) -- the index
/// vector `perm` such that `rows[perm[0]], rows[perm[1]], ...` is sorted. Callers reorder/window their
/// result rows by this index vector (the projection/GROUP-BY/join result order + OFFSET/LIMIT windowing)
/// AND the GROUP BY multi-pass alignment (S2.3) reorders per-pass group arrays on-device (a TOTAL key
/// order aligns every pass by index -- no host sort).
pub(crate) fn gpu_sort_permutation(
    rows: &[Vec<SqlValue>],
    order: &[(usize, bool)],
    // Parallel to `order`: the explicit NULLS FIRST/LAST override per key (None = PG default). Empty =
    // every key default. Honored ON-DEVICE for EVERY key type via the comparator's per-key validity bitmap
    // (built from the result rows into the payload) + the nulls_first bitmask -- no host NULL sentinel.
    nulls_first: &[Option<bool>],
    col_types: &[SqlType],
    device_memory: &gpu_db_execution::CudaResidentDeviceMemory,
) -> Result<Vec<u32>, ExecuteError> {
    if order.is_empty() || rows.len() <= 1 {
        return Ok((0..rows.len() as u32).collect());
    }
    let n = rows.len();
    let map_sort_err = |e: gpu_db_execution::CudaRuntimeProbeError| {
        ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
    };
    // (result-column index, kind 0=int / 1=text / 2=numeric / 3=uuid) per ORDER BY key.
    let mut classified: Vec<(usize, u8)> = Vec::with_capacity(order.len());
    for &(idx, _desc) in order {
        let kind = match col_types[idx] {
            // Bool sorts as its 0/1 ordinal in the int matrix (false < true), like the host key_cmp did.
            SqlType::Int4
            | SqlType::Int8
            | SqlType::Int2
            | SqlType::Date
            | SqlType::Timestamp
            | SqlType::Bool => 0u8,
            SqlType::Text => 1,
            SqlType::Numeric { .. } => 2,
            SqlType::Uuid => 3,
            // Exhaustive over SqlType: every result-column type is now sortable on-device. A new SqlType
            // must be classified here (compile error otherwise) rather than silently falling through.
        };
        classified.push((idx, kind));
    }
    // M3 (doc 21): per-key effective NULLS FIRST bit (explicit override, else the key's desc = PG default).
    // The NULL placement is decided ON-DEVICE by the sort comparator reading each key's validity bitmap --
    // INT keys included (their bitmap rides the payload, their value the int matrix), so there is NO host
    // NULL sentinel and int8/timestamp need no special-casing.
    let mut nulls_first_mask: u64 = 0;
    for (ki, &(_, desc)) in order.iter().enumerate() {
        if nulls_first.get(ki).copied().flatten().unwrap_or(desc) {
            nulls_first_mask |= 1u64 << ki;
        }
    }
    let num_int = classified.iter().filter(|&&(_, k)| k == 0).count();
    // INT key matrix, row-major by row position. A NULL writes a 0 PLACEHOLDER; the comparator detects the
    // NULL from the validity bitmap (built below) BEFORE the value compare, so the placeholder is never
    // compared -- the NULL placement decision is on-device, not a host sentinel value.
    let mut int_keys: Vec<i64> = Vec::with_capacity(n * num_int);
    for row in rows {
        for &(idx, kind) in &classified {
            if kind == 0 {
                int_keys.push(match row[idx] {
                    SqlValue::Int4(v) | SqlValue::Date(v) => i64::from(v),
                    SqlValue::Int2(v) => i64::from(v),
                    SqlValue::Int8(v) | SqlValue::Timestamp(v) => v,
                    SqlValue::Bool(b) => i64::from(b),
                    SqlValue::Null => 0,
                    _ => {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "ORDER BY int key encountered a non-int value".to_string(),
                        )));
                    }
                });
            }
        }
    }
    let mut desc_mask = 0u64;
    for (ki, &(_, desc)) in order.iter().enumerate() {
        if desc {
            desc_mask |= 1u64 << ki;
        }
    }
    // A NULL in ANY key column routes to the hetero comparator (it reads each key's validity bitmap from the
    // payload ON-DEVICE). An all-int, NULL-FREE sort keeps the fast int-matrix path (no bitmaps needed).
    let has_null_key = classified
        .iter()
        .any(|&(idx, _)| rows.iter().any(|r| matches!(r[idx], SqlValue::Null)));
    let non_int: Vec<(usize, u8)> = classified.iter().copied().filter(|&(_, k)| k != 0).collect();
    let perm: Vec<u32> = if non_int.is_empty() && !has_null_key {
        device_memory
            .bitonic_sort_multikey(&int_keys, n, num_int, desc_mask)
            .map_err(map_sort_err)?
    } else {
        // A resident-like payload over EVERY key column (M3 -- doc 21): build_relational_device_payload emits
        // a validity bitmap for any column holding a NULL, so an INT key's NULL placement is read ON-DEVICE
        // from its payload bitmap (the int VALUE still rides `int_keys`; the int value section in the payload
        // is unused, but its presence keeps every later section's offset correct). Non-int keys read both
        // value + validity from the payload. No host NULL sentinel on any key.
        let names: Vec<String> = (0..classified.len()).map(|i| format!("__gsk{i}")).collect();
        let types: Vec<SqlType> = classified.iter().map(|&(idx, _)| col_types[idx]).collect();
        let payload_rows: Vec<Vec<SqlValue>> = rows
            .iter()
            .map(|r| classified.iter().map(|&(idx, _)| r[idx].clone()).collect())
            .collect();
        let (payload, text_layouts, _bool, _int4, b128_layouts, null_layouts) =
            crate::engine_residency::build_relational_device_payload(&names, &types, &payload_rows)?;
        // null_offs[ki] = key ki's validity-bitmap byte offset in the payload (u64::MAX = the column has no
        // NULL ⇒ pure value compare). Found by the key's own name, so an int key gets its bitmap too.
        let null_off_of = |ki: usize| -> u64 {
            let name = format!("__gsk{ki}");
            null_layouts
                .iter()
                .find(|l| l.name == name)
                .map_or(u64::MAX, |l| l.bitmap_byte_offset)
        };
        // Walk ORDER BY order: int -> the next int-matrix slot; text/numeric/uuid -> the next section in its
        // type group (the helper lays each group out in passed-column order). The validity bitmap (all key
        // types) is read on-device via null_offs.
        let mut int_slot = 0u32;
        let mut text_idx = 0usize;
        let mut b128_idx = 0usize;
        let mut text_cols: Vec<(u64, u64)> = Vec::new();
        let mut b128_cols: Vec<u64> = Vec::new();
        let mut key_plan: Vec<u32> = Vec::with_capacity(order.len());
        let mut null_offs: Vec<u64> = Vec::with_capacity(order.len());
        for (ki, &(_, kind)) in classified.iter().enumerate() {
            match kind {
                0 => {
                    key_plan.push(int_slot);
                    int_slot += 1;
                }
                1 => {
                    let tl = &text_layouts[text_idx];
                    key_plan.push(0x4000_0000_u32 | text_cols.len() as u32);
                    text_cols.push((tl.offsets_byte_offset, tl.bytes_byte_offset));
                    text_idx += 1;
                }
                k => {
                    let off = b128_layouts[b128_idx].1;
                    let tag = if k == 2 { 0x8000_0000_u32 } else { 0xC000_0000_u32 };
                    key_plan.push(tag | b128_cols.len() as u32);
                    b128_cols.push(off);
                    b128_idx += 1;
                }
            }
            null_offs.push(null_off_of(ki));
        }
        let indices: Vec<u64> = (0..n as u64).collect();
        // NULL placement is honored ON-DEVICE for EVERY key: the comparator reads the key's validity bitmap
        // (null_offs) and places NULL per the per-key nulls_first bit (nulls_first_mask). An int key reads
        // its value from `int_keys` but its NULL-ness from the payload bitmap -- no host sentinel.
        device_memory
            .bitonic_sort_hetero_on_payload(
                &payload, &indices, &int_keys, num_int, &text_cols, &b128_cols, &key_plan, desc_mask,
                &null_offs, nulls_first_mask,
            )
            .map_err(map_sort_err)?
    };
    Ok(perm)
}

/// Narrow a grouped MIN/MAX or GROUP BY key, which the GPU kernel computes as an i64 (or, for
/// numeric, an i128 split into `lo`/`hi`), back to the column's own `SqlType`. int2/int4/date ride the
/// 4-byte read; int8/timestamp the 8-byte read; numeric reconstructs `hi:lo` at `scale`. `hi`/`scale`
/// are ignored for the non-numeric types.
fn narrow_ordered_value(ty: SqlType, lo: i64, hi: i64, scale: u8) -> SqlValue {
    match ty {
        SqlType::Numeric { .. } => {
            SqlValue::Numeric(Decimal128::new((i128::from(hi) << 64) | i128::from(lo as u64), scale))
        }
        SqlType::Int8 => SqlValue::Int8(lo),
        SqlType::Timestamp => SqlValue::Timestamp(lo),
        SqlType::Int2 => SqlValue::Int2(lo as i16),
        SqlType::Date => SqlValue::Date(lo as i32),
        // bool key / bool MIN/MAX value: the derived int4 column is 0/1 -> Bool.
        SqlType::Bool => SqlValue::Bool(lo != 0),
        _ => SqlValue::Int4(lo as i32),
    }
}

/// COUNT(DISTINCT v) over a NON-int group key: the building block of the GROUP-BY-(g,v) reduction.
/// Runs a general composite GROUP BY COUNT(*) over `members` (any mix of fixed-width + text columns --
/// the wide-key buffer for the fixed members + a text descriptor for the text ones), OPTIONALLY
/// prefixed by a DERIVED member `derived = Some((buffer_ptr, is_i64))` (the EXPRESSION group key,
/// materialized into its own per-row i32/i64 buffer -- read by wide-key build kind 4/5), over the rows
/// in `indices`, and returns each distinct tuple's REPRESENTATIVE absolute row (the b128 slot's lo).
/// The members (+ derived) must route to a rep_idx path (a text/numeric/uuid/derived member, or >2
/// columns -- never the 2-fixed i64/i128 pack, which has no representative row); the reduction
/// guarantees this by always appending the non-int value member. All grouping is on the GPU; the rep
/// rows are control-plane row indices (like the WHERE survivors) -- NO host download of key values.
fn composite_group_count_reps(
    snapshot: &RelationalResidencySnapshot,
    table: &RelationalTable,
    device_memory: &gpu_db_execution::CudaResidentDeviceMemory,
    members: &[(usize, SqlType)],
    indices: &[u32],
    row_count: u64,
    derived: Option<(u64, bool)>,
) -> Result<Vec<u32>, ExecuteError> {
    let map_err = |e: gpu_db_execution::CudaRuntimeProbeError| {
        ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
    };
    if indices.is_empty() {
        return Ok(Vec::new());
    }
    // Fixed members -> the comp_w wide-key buffer (int 8B, numeric/uuid 16B); text -> the descriptor.
    // A derived member (the expr key) is prefixed at dst 0 (kind 5=i64, kind 4=i32), read from its own
    // buffer (derived_ptr) -- so the column members start at dst 8.
    let mut descriptors: Vec<(u64, u64, u64)> = Vec::new();
    let mut dst_off: u64 = 0;
    if let Some((_, is_i64)) = derived {
        descriptors.push((if is_i64 { 5 } else { 4 }, 0, dst_off));
        dst_off += 8;
    }
    for &(idx, ty) in members {
        match ty {
            SqlType::Numeric { .. } | SqlType::Uuid => {
                descriptors.push((
                    2,
                    resident_device_numeric_column_offset(snapshot, table, idx)?,
                    dst_off,
                ));
                dst_off += 16;
            }
            SqlType::Int8 | SqlType::Timestamp => {
                descriptors.push((
                    1,
                    resident_device_int8_column_offset(snapshot, table, idx)?,
                    dst_off,
                ));
                dst_off += 8;
            }
            SqlType::Bool => {
                descriptors.push((
                    3,
                    resident_device_bool_column_offset(snapshot, table, idx)?,
                    dst_off,
                ));
                dst_off += 8;
            }
            SqlType::Text => {}
            _ => {
                descriptors.push((
                    0,
                    resident_device_int4_column_offset(snapshot, table, idx)?,
                    dst_off,
                ));
                dst_off += 8;
            }
        }
    }
    let comp_w = dst_off;
    let derived_ptr = derived.map_or(0, |(ptr, _)| ptr);
    let _kbuf;
    let key_base_override = if comp_w > 0 {
        let buf = device_memory
            // COUNT(DISTINCT) reps are over non-NULL keys (a nullable group key with COUNT(DISTINCT)
            // clean-errors upstream), so no per-member validity here.
            .build_wide_key_device(&descriptors, comp_w, row_count, derived_ptr, &[])
            .map_err(map_err)?;
        let ptr = buf.device_ptr();
        _kbuf = Some(buf);
        ptr
    } else {
        _kbuf = None;
        0
    };
    let mut flat: Vec<u64> = Vec::new();
    for &(idx, ty) in members {
        if matches!(ty, SqlType::Text) {
            let layout = resident_device_text_column_layout(snapshot, table, idx)?;
            flat.push(layout.offsets_byte_offset);
            flat.push(layout.bytes_byte_offset);
        }
    }
    let _tdesc;
    let (n_text, text_desc_ptr) = if flat.is_empty() {
        _tdesc = None;
        (0, 0)
    } else {
        let n = (flat.len() / 2) as u64;
        let buf = device_memory.upload_u64_device(&flat).map_err(map_err)?;
        let ptr = buf.device_ptr();
        _tdesc = Some(buf);
        (n, ptr)
    };
    let groups = device_memory
        .group_by_i32_count_sum_minmax_from_payload(
            0, 0, indices, false, false, false, false, false, false, 0, 0, false, 0, 0,
            key_base_override, 0, comp_w, n_text, text_desc_ptr, None, None,
        )
        .map_err(map_err)?;
    Ok(groups.iter().map(|g| g.key_i128 as u64 as u32).collect())
}

/// Device op-code for an arithmetic binary op (matches `expr_proto.ptx`: 0=add, 1=sub, 2=mul), or
/// `None` if `op` is not arithmetic.
fn arith_op_code(op: ResidentBinaryOp) -> Option<u32> {
    match op {
        ResidentBinaryOp::Add => Some(0),
        ResidentBinaryOp::Sub => Some(1),
        ResidentBinaryOp::Mul => Some(2),
        _ => None,
    }
}

/// Device comparison code for a comparison binary op (matches `expr_proto.ptx`: 0=eq, 1=lt, 2=le,
/// 3=gt, 4=ge), or `None` if `op` is not a kernel-supported comparison (`Ne` has no primitive yet).
fn compare_op_code(op: ResidentBinaryOp) -> Option<u32> {
    match op {
        ResidentBinaryOp::Eq => Some(0),
        ResidentBinaryOp::Lt => Some(1),
        ResidentBinaryOp::Le => Some(2),
        ResidentBinaryOp::Gt => Some(3),
        ResidentBinaryOp::Ge => Some(4),
        _ => None,
    }
}

/// Flip a comparison code for a swapped operand order: `k <cmp> value` == `value <flip(cmp)> k`.
/// eq stays eq; lt<->gt; le<->ge.
fn flip_comparison_code(code: u32) -> u32 {
    match code {
        1 => 3,
        2 => 4,
        3 => 1,
        4 => 2,
        other => other,
    }
}

fn is_int4_literal(expr: &ResidentExpr) -> bool {
    matches!(expr, ResidentExpr::Int4Literal(_))
}

/// The column index if `expr` is a `Column` of int8 (`SqlType::Int8`) type, else `None` (the type
/// matrix, doc 19).
fn int8_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
    match expr {
        ResidentExpr::Column(idx)
            if table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Int8) =>
        {
            Some(*idx)
        }
        _ => None,
    }
}

/// Whether any `Column` referenced anywhere in `expr` is int8 — used to reject an int8 shape the
/// simple-comparison path does not yet support (int8 arithmetic, mixed int4/int8) rather than
/// silently routing it to the int4 path.
fn expr_mentions_int8(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Int8)
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
        // IS NULL is a type-neutral validity test (it lowers to a bitmap mask), not a typed value
        // operand -- it never drives element-type routing.
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_int8(lhs, table) || expr_mentions_int8(rhs, table)
        }
    }
}

/// Whether any `Column` referenced anywhere in `expr` is int4 — used to reject a mixed int4/int8
/// expression. An `Int4Literal` is NOT an int4 column (it is coercible to int8), so it does not count.
fn expr_mentions_int4_column(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Int4)
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_int4_column(lhs, table) || expr_mentions_int4_column(rhs, table)
        }
    }
}

/// The column index if `expr` is a `Column` of numeric (`SqlType::Numeric`) type, else `None`.
fn numeric_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
    match expr {
        ResidentExpr::Column(idx)
            if matches!(
                table.columns.get(*idx).map(|column| column.ty),
                Some(SqlType::Numeric { .. })
            ) =>
        {
            Some(*idx)
        }
        _ => None,
    }
}

/// Whether `expr` mentions numeric anywhere — a numeric `Column` or a `NumericLiteral`. Marks a
/// predicate as numeric (the type matrix, doc 19); an `Int4Literal` is integer until it is coerced
/// against a numeric column.
fn expr_mentions_numeric(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => matches!(
            table.columns.get(*idx).map(|column| column.ty),
            Some(SqlType::Numeric { .. })
        ),
        ResidentExpr::NumericLiteral(_) => true,
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_numeric(lhs, table) || expr_mentions_numeric(rhs, table)
        }
    }
}

/// The column index if `expr` is a `Column` of text (`SqlType::Text`) type, else `None`.
fn text_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
    match expr {
        ResidentExpr::Column(idx)
            if table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Text) =>
        {
            Some(*idx)
        }
        _ => None,
    }
}

/// The literal bytes if `expr` is a `TextLiteral`, else `None`.
fn text_literal_value(expr: &ResidentExpr) -> Option<&str> {
    match expr {
        ResidentExpr::TextLiteral(value) => Some(value.as_str()),
        _ => None,
    }
}

/// Whether `expr` mentions text anywhere -- a text `Column` or a `TextLiteral`. Marks a predicate as
/// text (the type matrix, doc 19).
fn expr_mentions_text(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Text)
        }
        ResidentExpr::TextLiteral(_) => true,
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_text(lhs, table) || expr_mentions_text(rhs, table)
        }
    }
}

/// Whether `expr` mentions a BOOL column anywhere (a bool literal alone does not -- it only matters
/// paired with a bool column). Marks a predicate as touching the i32 bool-mask path.
fn expr_mentions_bool_column(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Bool)
        }
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_bool_column(lhs, table) || expr_mentions_bool_column(rhs, table)
        }
        _ => false,
    }
}

/// The column index if `expr` is a `Column` of `date` type, else `None`.
fn date_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
    match expr {
        ResidentExpr::Column(idx)
            if table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Date) =>
        {
            Some(*idx)
        }
        _ => None,
    }
}

/// Whether `expr` mentions a date COLUMN anywhere. A predicate is "date" when a date column is
/// involved; a bare string literal is the date VALUE, resolved against the column at lowering.
fn expr_mentions_date(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Date)
        }
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_date(lhs, table) || expr_mentions_date(rhs, table)
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
    }
}

/// The i32 day count of a date literal: a `TextLiteral` parsed as ISO `YYYY-MM-DD` (PG coerces an
/// unknown-type string literal to the column type). Anything else (an integer literal, a non-date
/// column) is a type error -- a date column compares only to a date literal or another date column.
fn date_literal_days(expr: &ResidentExpr) -> Result<i32, ExecuteError> {
    match expr {
        ResidentExpr::TextLiteral(text) => gpu_db_sql::datetime::parse_date(text).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "invalid input syntax for type date: \"{text}\""
            )))
        }),
        _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "a date column compares only to a date literal or another date column".to_string(),
        ))),
    }
}

/// The column index if `expr` is a `Column` of `timestamp` type, else `None`.
fn timestamp_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
    match expr {
        ResidentExpr::Column(idx)
            if table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Timestamp) =>
        {
            Some(*idx)
        }
        _ => None,
    }
}

/// Whether `expr` mentions a timestamp COLUMN anywhere (the predicate is "timestamp" when a timestamp
/// column is involved; a bare string literal is the timestamp VALUE, resolved at lowering).
fn expr_mentions_timestamp(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Timestamp)
        }
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_timestamp(lhs, table) || expr_mentions_timestamp(rhs, table)
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
    }
}

/// The i64 microsecond count of a timestamp literal: a `TextLiteral` parsed as `YYYY-MM-DD HH:MM:SS`.
/// Anything else is a type error -- a timestamp column compares only to a timestamp literal or column.
fn timestamp_literal_micros(expr: &ResidentExpr) -> Result<i64, ExecuteError> {
    match expr {
        ResidentExpr::TextLiteral(text) => {
            gpu_db_sql::datetime::parse_timestamp(text).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "invalid input syntax for type timestamp: \"{text}\""
                )))
            })
        }
        _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "a timestamp column compares only to a timestamp literal or another timestamp column"
                .to_string(),
        ))),
    }
}

/// The column index if `expr` is a `Column` of `uuid` type, else `None`.
fn uuid_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
    match expr {
        ResidentExpr::Column(idx)
            if table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Uuid) =>
        {
            Some(*idx)
        }
        _ => None,
    }
}

/// Whether `expr` mentions a uuid COLUMN anywhere (the predicate is "uuid" when a uuid column is
/// involved; a bare string literal is the uuid VALUE, resolved at lowering).
fn expr_mentions_uuid(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Uuid)
        }
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_uuid(lhs, table) || expr_mentions_uuid(rhs, table)
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
    }
}

/// The 16 bytes of a uuid literal: a `TextLiteral` parsed as a uuid. Anything else is a type error --
/// a uuid column compares only to a uuid literal or another uuid column.
fn uuid_literal_bytes(expr: &ResidentExpr) -> Result<[u8; 16], ExecuteError> {
    match expr {
        ResidentExpr::TextLiteral(text) => gpu_db_sql::uuid::parse_uuid(text).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "invalid input syntax for type uuid: \"{text}\""
            )))
        }),
        _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "a uuid column compares only to a uuid literal or another uuid column".to_string(),
        ))),
    }
}

/// The column index if `expr` is a `Column` of `smallint` (int2) type, else `None`.
fn int2_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
    match expr {
        ResidentExpr::Column(idx)
            if table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Int2) =>
        {
            Some(*idx)
        }
        _ => None,
    }
}

/// Whether `expr` mentions a smallint COLUMN anywhere.
fn expr_mentions_int2(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Int2)
        }
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_int2(lhs, table) || expr_mentions_int2(rhs, table)
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
    }
}

/// The i32 scalar a smallint compares against: an `Int4Literal` (PG promotes both sides to int4, so
/// an out-of-int16 literal is a valid comparison that simply matches no rows -- NOT a range error).
/// A non-integer literal is a type error -- a smallint compares only to an integer literal or column.
fn int2_literal_value(expr: &ResidentExpr) -> Result<i32, ExecuteError> {
    match expr {
        ResidentExpr::Int4Literal(value) => Ok(*value),
        _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "a smallint column compares only to an integer literal or another smallint column"
                .to_string(),
        ))),
    }
}

/// Compile a SQL `LIKE` pattern to the device kernel's u32 token array: one token per output position,
/// `(op << 8) | literal_byte`, op 0 = literal byte, 1 = any-one (`_`), 2 = any-run (`%`). The default
/// `\` escape is resolved here (`\%` / `\_` / `\\` -> a literal byte); a lone trailing `\` is an error
/// (PG: "LIKE pattern must not end with escape character"). A multi-byte UTF-8 literal character
/// becomes one literal token per byte (matched byte-by-byte against the same bytes in the text).
fn compile_like_pattern(pattern: &str) -> Result<Vec<u32>, ExecuteError> {
    let bytes = pattern.as_bytes();
    let mut tokens = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                i += 1;
                if i >= bytes.len() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "LIKE pattern must not end with escape character".to_string(),
                    )));
                }
                tokens.push(u32::from(bytes[i])); // op 0 (literal), the escaped byte
            }
            b'%' => tokens.push(2u32 << 8),
            b'_' => tokens.push(1u32 << 8),
            other => tokens.push(u32::from(other)),
        }
        i += 1;
    }
    Ok(tokens)
}

/// The numeric value of a literal operand: a `NumericLiteral` as-is, or an `Int4Literal` coerced to
/// `Decimal128` (scale 0) — PG's integer->numeric coercion. `None` for a non-literal.
fn numeric_literal_value(expr: &ResidentExpr) -> Option<Decimal128> {
    match expr {
        ResidentExpr::NumericLiteral(value) => Some(*value),
        ResidentExpr::Int4Literal(value) => Some(Decimal128::new(i128::from(*value), 0)),
        _ => None,
    }
}

/// The declared scale of a numeric column, or `None` if the column is not numeric.
fn column_numeric_scale(table: &RelationalTable, column_idx: usize) -> Option<u8> {
    match table.columns.get(column_idx).map(|column| column.ty) {
        Some(SqlType::Numeric { scale, .. }) => Some(scale),
        _ => None,
    }
}

/// Rescale a numeric literal to a column's scale and return the comparable i128 mantissa. The literal
/// rescales UP exactly (its scale <= the column scale). A literal with MORE fractional digits than the
/// column is rejected: rescaling it down would round, and PG compares numerics exactly — rounding
/// would yield wrong rows. Cross-scale comparison (rescaling the column on the GPU) is a follow-on.
fn rescale_numeric_literal(literal: Decimal128, column_scale: u8) -> Result<i128, ExecuteError> {
    // Canonicalize first (strip trailing zeros) so `10.500` is treated as `10.5` — PG ignores trailing
    // zeros (10.500 == 10.50), so it must NOT be rejected just for a wider written scale.
    let literal = literal.canonical();
    if literal.scale > column_scale {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "numeric literal has more fractional digits than the column scale (exact cross-scale \
             comparison is a follow-on)"
                .to_string(),
        )));
    }
    literal
        .rescale(column_scale)
        .map(|rescaled| rescaled.mantissa)
        .map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "numeric literal out of range after rescaling to the column scale".to_string(),
            ))
        })
}

/// 10^exp as i32 — the multiplier that rescales a numeric mantissa UP by `exp` decimal places (a
/// scale gap). Errors if it does not fit i32 (exp > 9), since the in-VM scalar is i32-bounded; a wider
/// cross-scale gap is a follow-on.
fn pow10_i32(exp: u8) -> Result<i32, ExecuteError> {
    let mut acc: i32 = 1;
    for _ in 0..exp {
        acc = acc.checked_mul(10).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "numeric cross-scale comparison spans more than 9 fractional digits (a follow-on)"
                    .to_string(),
            ))
        })?;
    }
    Ok(acc)
}

/// Push a step rescaling the top-of-stack i128 buffer from scale `from` UP to `to` (mantissa *
/// 10^(to-from), a checked multiply), or nothing if already equal. `to >= from` (the caller passes the
/// common = max scale). Used to bring the operands of a cross-scale numeric comparison to one scale.
fn push_numeric_rescale(from: u8, to: u8, program: &mut Vec<ExprStep>) -> Result<(), ExecuteError> {
    if to > from {
        program.push(ExprStep::ScalarBinary {
            op: 2, // multiply by 10^(to-from)
            scalar: pow10_i32(to - from)?,
            scalar_on_left: false,
        });
    }
    Ok(())
}

/// The result scale a numeric arithmetic expression compiles to, computed WITHOUT emitting steps. The
/// cross-scale add/sub/compare lowering needs each side's scale up front (to size the common = max
/// scale) before the operands are pushed, since the rescale step acts on the top of the stack. Columns
/// at their catalog scale; literals at their canonical scale; `+`/`-` -> max(operand scales); `*` ->
/// the sum. Mirrors `compile_numeric_arith`'s scale arithmetic exactly.
fn numeric_arith_scale(expr: &ResidentExpr, table: &RelationalTable) -> Result<u8, ExecuteError> {
    match expr {
        ResidentExpr::Column(col) => column_numeric_scale(table, *col).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "numeric arithmetic operand is not a numeric column".to_string(),
            ))
        }),
        ResidentExpr::NumericLiteral(decimal) => Ok(decimal.canonical().scale),
        ResidentExpr::Int4Literal(_) => Ok(0),
        ResidentExpr::TextLiteral(_) | ResidentExpr::BoolLiteral(_) => {
            Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a text/bool literal is not a numeric arithmetic operand".to_string(),
            )))
        }
        // IS NULL is a predicate leaf, never an arithmetic operand (the parser never produces it here).
        ResidentExpr::IsNull { .. } => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "IS NULL is not a numeric arithmetic operand".to_string(),
        ))),
        ResidentExpr::Binary { op, lhs, rhs } => {
            let lhs_scale = numeric_arith_scale(lhs, table)?;
            let rhs_scale = numeric_arith_scale(rhs, table)?;
            match arith_op_code(*op) {
                Some(2) => lhs_scale.checked_add(rhs_scale).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "numeric multiply result scale exceeds the supported range".to_string(),
                    ))
                }),
                Some(0 | 1) => Ok(lhs_scale.max(rhs_scale)),
                _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "the general GPU executor supports numeric +, -, * arithmetic only".to_string(),
                ))),
            }
        }
    }
}

/// Compile a numeric ARITHMETIC subtree to i128 VM steps, RETURNING the result's scale (computed
/// bottom-up). A `Column` loads from its device offset (scale = its catalog scale). `+`/`-` bring both
/// operands to the common = max scale (rescaling the coarser UP) and keep it; `*` ADDS the operand
/// scales (PG numeric multiply). A literal operand folds into a `ScalarBinary`: for `+`/`-` its
/// mantissa rescales to the common scale; for `*` its CANONICAL mantissa is the multiplier and its
/// canonical scale adds to the result. Every scalar must fit i32 (the `ExprStep` bound; larger literals
/// are a follow-on). Mirrors `compile_arith_program`.
fn compile_numeric_arith(
    expr: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<u8, ExecuteError> {
    let scalar_i32 = |mantissa: i128| -> Result<i32, ExecuteError> {
        i32::try_from(mantissa).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "numeric arithmetic literal is too large for the fast path yet".to_string(),
            ))
        })
    };
    let scale_overflow = || {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "numeric multiply result scale exceeds the supported range".to_string(),
        ))
    };
    let literal_only = || {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "the general GPU executor does not constant-fold literal-only numeric arithmetic"
                .to_string(),
        ))
    };
    match expr {
        ResidentExpr::Column(col) => {
            let scale = column_numeric_scale(table, *col).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "numeric arithmetic operand is not a numeric column".to_string(),
                ))
            })?;
            let byte_offset = resident_device_numeric_column_offset(snapshot, table, *col)?;
            program.push(ExprStep::LoadColumn { byte_offset });
            Ok(scale)
        }
        // IS NULL is a predicate leaf, never an arithmetic operand (the parser never produces it here).
        ResidentExpr::IsNull { .. } => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "IS NULL is not a numeric arithmetic operand".to_string(),
        ))),
        ResidentExpr::Binary { op, lhs, rhs } => {
            let op_code = arith_op_code(*op).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "the general GPU executor supports numeric +, -, * arithmetic only".to_string(),
                ))
            })?;
            let lhs_lit = numeric_literal_value(lhs);
            let rhs_lit = numeric_literal_value(rhs);
            if op_code == 2 {
                // MULTIPLY: the result scale is the SUM of the operand scales. A literal multiplier
                // contributes its canonical mantissa + canonical scale (so `* 1.5` adds scale 1).
                return match (lhs_lit, rhs_lit) {
                    (None, Some(literal)) | (Some(literal), None) => {
                        let value = if lhs_lit.is_none() { lhs.as_ref() } else { rhs.as_ref() };
                        let value_scale = compile_numeric_arith(value, table, snapshot, program)?;
                        let canonical = literal.canonical();
                        program.push(ExprStep::ScalarBinary {
                            op: 2,
                            scalar: scalar_i32(canonical.mantissa)?,
                            scalar_on_left: false, // multiply is commutative
                        });
                        value_scale.checked_add(canonical.scale).ok_or_else(scale_overflow)
                    }
                    (None, None) => {
                        let lhs_scale = compile_numeric_arith(lhs, table, snapshot, program)?;
                        let rhs_scale = compile_numeric_arith(rhs, table, snapshot, program)?;
                        program.push(ExprStep::BufferBinary { op: 2 });
                        lhs_scale.checked_add(rhs_scale).ok_or_else(scale_overflow)
                    }
                    (Some(_), Some(_)) => Err(literal_only()),
                };
            }
            // ADD / SUB: bring both operands to the common = max scale (rescale the COARSER side UP),
            // then add/sub; the result keeps the common scale.
            match (lhs_lit, rhs_lit) {
                (None, Some(literal)) => {
                    let value_scale = compile_numeric_arith(lhs, table, snapshot, program)?;
                    let common = value_scale.max(literal.canonical().scale);
                    push_numeric_rescale(value_scale, common, program)?;
                    let mantissa = rescale_numeric_literal(literal, common)?;
                    program.push(ExprStep::ScalarBinary {
                        op: op_code,
                        scalar: scalar_i32(mantissa)?,
                        scalar_on_left: false,
                    });
                    Ok(common)
                }
                (Some(literal), None) => {
                    let value_scale = compile_numeric_arith(rhs, table, snapshot, program)?;
                    let common = value_scale.max(literal.canonical().scale);
                    push_numeric_rescale(value_scale, common, program)?;
                    let mantissa = rescale_numeric_literal(literal, common)?;
                    program.push(ExprStep::ScalarBinary {
                        op: op_code,
                        scalar: scalar_i32(mantissa)?,
                        scalar_on_left: true,
                    });
                    Ok(common)
                }
                (None, None) => {
                    // The common scale must be known BEFORE compiling either side (the rescale step
                    // operates on the top of the stack), so predict each side's scale statically.
                    let common =
                        numeric_arith_scale(lhs, table)?.max(numeric_arith_scale(rhs, table)?);
                    let lhs_scale = compile_numeric_arith(lhs, table, snapshot, program)?;
                    push_numeric_rescale(lhs_scale, common, program)?;
                    let rhs_scale = compile_numeric_arith(rhs, table, snapshot, program)?;
                    push_numeric_rescale(rhs_scale, common, program)?;
                    program.push(ExprStep::BufferBinary { op: op_code });
                    Ok(common)
                }
                (Some(_), Some(_)) => Err(literal_only()),
            }
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "a bare literal cannot be a numeric arithmetic value".to_string(),
        ))),
    }
}

/// Compile a single numeric COMPARISON `lhs <cmp> rhs` to i128 VM steps that leave one MASK on the
/// stack (the buffer path: compile each side via [`compile_numeric_arith`], rescale to the common =
/// max scale, then a `CompareScalar`/`CompareBuffers` mask step). Each side may be a column, an
/// arithmetic subtree, or a literal (folded into the scalar). This is the shared comparison core for
/// both the single-predicate arithmetic arm and the numeric AND/OR mask program.
fn compile_numeric_compare(
    lhs: &ResidentExpr,
    rhs: &ResidentExpr,
    cmp: u32,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<(), ExecuteError> {
    // The numeric program runs at elem I128, so the comparison literal is a FULL i128 mantissa via
    // CompareScalarI128 -- NOT the i32 `CompareScalar` scalar (which capped a high-scale literal, e.g. a
    // scale-20 AVG result rescaled, at i32 and clean-errored a valid AND/OR HAVING/WHERE leaf).
    match (numeric_literal_value(lhs), numeric_literal_value(rhs)) {
        (Some(literal), None) => {
            // literal <cmp> arith: bring both to the common = max scale.
            let arith_scale = compile_numeric_arith(rhs, table, snapshot, program)?;
            let common = arith_scale.max(literal.canonical().scale);
            push_numeric_rescale(arith_scale, common, program)?;
            program.push(ExprStep::CompareScalarI128 {
                cmp,
                scalar: rescale_numeric_literal(literal, common)?,
                scalar_on_left: true,
            });
            // M3 (doc 21) 3VL: AND the leaf mask with the arith operand columns' validity (a NULL operand
            // is UNKNOWN ⇒ excluded). No-op when no operand column has a validity bitmap (non-null path
            // stays byte-identical). Correct for WHERE under AND/OR (no NOT) — same rule as the int4 leaf.
            push_leaf_validity_and(&[rhs], table, snapshot, program)
        }
        (None, Some(literal)) => {
            let arith_scale = compile_numeric_arith(lhs, table, snapshot, program)?;
            let common = arith_scale.max(literal.canonical().scale);
            push_numeric_rescale(arith_scale, common, program)?;
            program.push(ExprStep::CompareScalarI128 {
                cmp,
                scalar: rescale_numeric_literal(literal, common)?,
                scalar_on_left: false,
            });
            push_leaf_validity_and(&[lhs], table, snapshot, program)
        }
        (None, None) => {
            // arith <cmp> arith: bring both sides to the common = max scale.
            let common = numeric_arith_scale(lhs, table)?.max(numeric_arith_scale(rhs, table)?);
            let lhs_scale = compile_numeric_arith(lhs, table, snapshot, program)?;
            push_numeric_rescale(lhs_scale, common, program)?;
            let rhs_scale = compile_numeric_arith(rhs, table, snapshot, program)?;
            push_numeric_rescale(rhs_scale, common, program)?;
            program.push(ExprStep::CompareBuffers { cmp });
            push_leaf_validity_and(&[lhs, rhs], table, snapshot, program)
        }
        (Some(_), Some(_)) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "the general GPU executor does not evaluate literal-only numeric comparisons".to_string(),
        ))),
    }
}

/// Compile a numeric boolean predicate (comparisons combined by `AND`/`OR`) into a mask program for
/// the i128 VM, mirroring [`compile_predicate_program`] (the int path) but scale-aware: `AND`/`OR`
/// compile both operand predicates then a `MaskBinary`; a comparison leaf goes through
/// [`compile_numeric_compare`]. The VM compacts the final mask to surviving row indices.
fn compile_numeric_predicate_program(
    predicate: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<(), ExecuteError> {
    let ResidentExpr::Binary { op, lhs, rhs } = predicate else {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident numeric predicate must be a comparison or AND/OR combination".to_string(),
        )));
    };
    if let Some(bool_op) = boolean_op_code(*op) {
        compile_numeric_predicate_program(lhs, table, snapshot, program)?;
        compile_numeric_predicate_program(rhs, table, snapshot, program)?;
        program.push(ExprStep::MaskBinary { op: bool_op });
        return Ok(());
    }
    let Some(cmp) = predicate_compare_code(*op) else {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident numeric predicate node must be a comparison (eq/ne/lt/le/gt/ge) or AND/OR"
                .to_string(),
        )));
    };
    compile_numeric_compare(lhs, rhs, cmp, table, snapshot, program)
}

/// Compile an int4 arithmetic value expression into postfix [`ExprStep`] bytecode for the device
/// VM. Recurses: a `Column` loads to a buffer; a `Binary{arith}` emits its operands then a buffer x
/// buffer op, or folds an immediate literal operand into a buffer x scalar op (preserving operand
/// side). Rejects non-arithmetic ops and literal-only subtrees (host constant-folding is a later
/// step) — the predicate's top-level comparison is handled by the caller.
fn compile_arith_program(
    expr: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<(), ExecuteError> {
    match expr {
        ResidentExpr::Column(col) => {
            // Resolve int4 OR int8 offset by the column's catalog type; a program is mono-typed (the
            // engine rejects mixed int4/int8), and the VM is run with the matching element type.
            let byte_offset = resident_device_int_column_offset(snapshot, table, *col)?;
            program.push(ExprStep::LoadColumn { byte_offset });
            Ok(())
        }
        // IS NULL is a predicate leaf, never an arithmetic operand (the parser never produces it here).
        ResidentExpr::IsNull { .. } => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "IS NULL is not an arithmetic operand".to_string(),
        ))),
        ResidentExpr::Binary { op, lhs, rhs } => {
            let op_code = arith_op_code(*op).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident Expr arithmetic position requires an arithmetic op (add/sub/mul)"
                        .to_string(),
                ))
            })?;
            match (lhs.as_ref(), rhs.as_ref()) {
                (value, ResidentExpr::Int4Literal(scalar)) if !is_int4_literal(value) => {
                    compile_arith_program(value, table, snapshot, program)?;
                    program.push(ExprStep::ScalarBinary {
                        op: op_code,
                        scalar: *scalar,
                        scalar_on_left: false,
                    });
                    Ok(())
                }
                (ResidentExpr::Int4Literal(scalar), value) if !is_int4_literal(value) => {
                    compile_arith_program(value, table, snapshot, program)?;
                    program.push(ExprStep::ScalarBinary {
                        op: op_code,
                        scalar: *scalar,
                        scalar_on_left: true,
                    });
                    Ok(())
                }
                (lhs_expr, rhs_expr)
                    if !is_int4_literal(lhs_expr) && !is_int4_literal(rhs_expr) =>
                {
                    compile_arith_program(lhs_expr, table, snapshot, program)?;
                    compile_arith_program(rhs_expr, table, snapshot, program)?;
                    program.push(ExprStep::BufferBinary { op: op_code });
                    Ok(())
                }
                _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident Expr interpreter does not yet constant-fold literal-only arithmetic \
                     subtrees"
                        .to_string(),
                ))),
            }
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident Expr arithmetic value cannot be a bare literal (constant-folding pending)"
                .to_string(),
        ))),
    }
}

/// Device comparison code including `Ne` (matches the mask kernels: 0=eq, 1=lt, 2=le, 3=gt, 4=ge,
/// 5=ne), used by the mask-based predicate VM. (`compare_op_code` omits `Ne` because the fused
/// scalar/buffer compact kernels only cover 0-4.)
fn predicate_compare_code(op: ResidentBinaryOp) -> Option<u32> {
    match op {
        ResidentBinaryOp::Eq => Some(0),
        ResidentBinaryOp::Lt => Some(1),
        ResidentBinaryOp::Le => Some(2),
        ResidentBinaryOp::Gt => Some(3),
        ResidentBinaryOp::Ge => Some(4),
        ResidentBinaryOp::Ne => Some(5),
        _ => None,
    }
}

/// Device boolean-combinator code for the mask `MaskBinary` step (0=and, 1=or), or `None`.
fn boolean_op_code(op: ResidentBinaryOp) -> Option<u32> {
    match op {
        ResidentBinaryOp::And => Some(0),
        ResidentBinaryOp::Or => Some(1),
        _ => None,
    }
}

/// Collect the (distinct, first-seen order) column indices a predicate-leaf operand references. Literals
/// and `IS NULL` contribute no value-operand column (an `IS NULL` leaf is its own validity test, never an
/// arithmetic operand).
fn collect_expr_columns(expr: &ResidentExpr, out: &mut Vec<usize>) {
    match expr {
        ResidentExpr::Column(idx) => {
            if !out.contains(idx) {
                out.push(*idx);
            }
        }
        ResidentExpr::Binary { lhs, rhs, .. } => {
            collect_expr_columns(lhs, out);
            collect_expr_columns(rhs, out);
        }
        _ => {}
    }
}

/// 3VL (M3 — doc 21): AND a comparison-leaf mask (on top of the VM stack) with column `col`'s NULL
/// VALIDITY mask, so a NULL operand makes the leaf UNKNOWN ⇒ mask 0 ⇒ the row is not selected. A column
/// with NO validity bitmap (it holds no NULLs) is all-valid, so this is a no-op (skipped) and the
/// non-null predicate program stays byte-identical. Pushes `BoolMask(validity, negate=false)` then
/// `MaskBinary(AND)` (the validity bit is 1 when the row is present). Correct for WHERE under AND/OR
/// because UNKNOWN and FALSE both exclude the row; this predicate path carries no logical NOT (which
/// would distinguish them — `<>`/`!=` is a value comparison, itself UNKNOWN on a NULL operand).
fn push_column_validity_and(
    col: usize,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<(), ExecuteError> {
    if let Some(bitmap_byte_offset) = resident_device_null_column_offset(snapshot, table, col)? {
        program.push(ExprStep::BoolMask {
            bitmap_byte_offset,
            negate: false,
        });
        program.push(ExprStep::MaskBinary { op: 0 }); // 0 = AND
    }
    Ok(())
}

/// Apply [`push_column_validity_and`] for every nullable column referenced across `operands` (a
/// comparison leaf's arithmetic operand expressions), so the leaf's mask excludes any row in which ANY
/// operand column is NULL (an arithmetic result is NULL if any input is NULL).
fn push_leaf_validity_and(
    operands: &[&ResidentExpr],
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<(), ExecuteError> {
    let mut cols = Vec::new();
    for operand in operands {
        collect_expr_columns(operand, &mut cols);
    }
    for col in cols {
        push_column_validity_and(col, table, snapshot, program)?;
    }
    Ok(())
}

/// True if any VALUE-operand column the predicate references has a NULL validity bitmap (holds ≥1 NULL).
/// `IS NULL` operands are excluded (an `IS NULL` test is always defined — it needs no 3VL routing).
fn predicate_references_nullable_column(
    predicate: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
) -> Result<bool, ExecuteError> {
    let mut cols = Vec::new();
    collect_expr_columns(predicate, &mut cols);
    for col in cols {
        if resident_device_null_column_offset(snapshot, table, col)?.is_some() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The general mask-VM element type that lowers `predicate` with NULL 3VL, or `None` if the VM cannot
/// lower it (a clean-error follow-up). The VM is MONO-TYPED — one element width for the whole program —
/// and the NULL validity AND (`push_column_validity_and`) is a type-independent 1-bit mask, so the only
/// constraint is that every value-operand column shares a VM width class. `I32` = all columns int4 / text
/// / bool (text+bool are i32-compatible mask leaves). `I64` = all columns int8 (BIGINT): the i64 buffer
/// VM the non-null int8 AND/OR path already uses (`run_expr_predicate_filter(elem=I64)`), where the
/// validity BoolMask composes as an i32 mask. A MIXED-width predicate (int4+int8, or int8+text/bool), or a
/// nullable numeric/uuid/date/timestamp/int2 column, returns `None` (those compare kernels are not yet
/// validity-aware), so it must NOT route here.
fn predicate_vm_elem_type(
    predicate: &ResidentExpr,
    table: &RelationalTable,
) -> Option<ResidentElemType> {
    let mut cols = Vec::new();
    collect_expr_columns(predicate, &mut cols);
    let ty = |col: usize| table.columns.get(col).map(|column| column.ty);
    // int2 (smallint) is stored WIDENED to i32 in the int4 section, so it lowers on the I32 VM exactly
    // like int4 (its literal is an i32 that fits CompareScalar). compile_arith_program resolves its offset
    // via resident_device_int_column_offset (Int2 -> the int4 section).
    if cols.iter().all(|&col| {
        matches!(
            ty(col),
            Some(SqlType::Int4 | SqlType::Int2 | SqlType::Text | SqlType::Bool)
        )
    }) {
        Some(ResidentElemType::I32)
    } else if cols.iter().all(|&col| matches!(ty(col), Some(SqlType::Int8))) {
        Some(ResidentElemType::I64)
    } else {
        None
    }
}

/// Compile a boolean predicate expression into postfix [`ExprStep`] bytecode that leaves one MASK on
/// the VM stack. Recurses: `AND`/`OR` compile both operand predicates then a `MaskBinary`; a
/// comparison compiles its arithmetic operand(s) (via [`compile_arith_program`]) then a
/// `CompareScalar`/`CompareBuffers` mask step. The predicate VM compacts the final mask to indices.
fn compile_predicate_program(
    expr: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
    needles: &mut Vec<Vec<u8>>,
) -> Result<(), ExecuteError> {
    // A bare BOOL column used as a predicate leaf (`WHERE flag AND ...`) -> its bitmap as a mask.
    if let ResidentExpr::Column(idx) = expr {
        if table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Bool) {
            let offset = resident_device_bool_column_offset(snapshot, table, *idx)?;
            program.push(ExprStep::BoolMask {
                bitmap_byte_offset: offset,
                negate: false,
            });
            return Ok(());
        }
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "a bare column predicate leaf must be a bool column (argument of WHERE must be boolean)"
                .to_string(),
        )));
    }
    // `col IS NULL` / `col IS NOT NULL` (M3 -- doc 21): the column's NULL validity bitmap as a mask,
    // reusing the bool bitmap->mask kernel. The bitmap bit is 1 when the row is VALID (present), so
    // `IS NOT NULL` reads it as-is and `IS NULL` complements it (`negate = !is_not_null`). A column with
    // no validity bitmap holds no NULLs -> every row valid -> a constant mask (all-1 for IS NOT NULL,
    // all-0 for IS NULL), needing no kernel.
    if let ResidentExpr::IsNull { col, is_not_null } = expr {
        match resident_device_null_column_offset(snapshot, table, *col)? {
            Some(bitmap_byte_offset) => program.push(ExprStep::BoolMask {
                bitmap_byte_offset,
                negate: !is_not_null,
            }),
            None => program.push(ExprStep::ConstMask {
                value: *is_not_null,
            }),
        }
        return Ok(());
    }
    let ResidentExpr::Binary { op, lhs, rhs } = expr else {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident predicate must be a comparison or boolean (AND/OR) combination".to_string(),
        )));
    };
    if let Some(bool_op) = boolean_op_code(*op) {
        compile_predicate_program(lhs, table, snapshot, program, needles)?;
        compile_predicate_program(rhs, table, snapshot, program, needles)?;
        program.push(ExprStep::MaskBinary { op: bool_op });
        return Ok(());
    }
    // TEXT comparison leaf -> a mask step the VM combines with AND/OR (so text IN / multi-text WHERE / a
    // nullable-text or compound LIKE run on the GPU). LIKE -> TextLikeMask; `=`/`<>` -> TextEqMask. int4
    // leaves fall through to the arith VM below.
    if expr_mentions_text(lhs, table) || expr_mentions_text(rhs, table) {
        if matches!(op, ResidentBinaryOp::Like) {
            return compile_text_like_leaf(lhs, rhs, table, snapshot, program, needles);
        }
        return compile_text_eq_leaf(*op, lhs, rhs, table, snapshot, program, needles);
    }
    // BOOL comparison leaf (`boolcol =/<> true|false`, either order) -> a BoolMask step. `NOT flag` is
    // mapped to `flag = false` upstream, so this also covers it. int4 leaves fall through below.
    let is_bool_col = |e: &ResidentExpr| {
        matches!(e, ResidentExpr::Column(i)
            if table.columns.get(*i).map(|column| column.ty) == Some(SqlType::Bool))
    };
    if is_bool_col(lhs) || is_bool_col(rhs) {
        return compile_bool_leaf(*op, lhs, rhs, table, snapshot, program);
    }
    let Some(cmp) = predicate_compare_code(*op) else {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident predicate node must be a comparison (eq/ne/lt/le/gt/ge) or AND/OR".to_string(),
        )));
    };
    match (lhs.as_ref(), rhs.as_ref()) {
        (value, ResidentExpr::Int4Literal(scalar)) if !is_int4_literal(value) => {
            compile_arith_program(value, table, snapshot, program)?;
            program.push(ExprStep::CompareScalar {
                cmp,
                scalar: *scalar,
                scalar_on_left: false,
            });
            push_leaf_validity_and(&[value], table, snapshot, program)
        }
        (ResidentExpr::Int4Literal(scalar), value) if !is_int4_literal(value) => {
            compile_arith_program(value, table, snapshot, program)?;
            program.push(ExprStep::CompareScalar {
                cmp,
                scalar: *scalar,
                scalar_on_left: true,
            });
            push_leaf_validity_and(&[value], table, snapshot, program)
        }
        (lhs_expr, rhs_expr) if !is_int4_literal(lhs_expr) && !is_int4_literal(rhs_expr) => {
            compile_arith_program(lhs_expr, table, snapshot, program)?;
            compile_arith_program(rhs_expr, table, snapshot, program)?;
            program.push(ExprStep::CompareBuffers { cmp });
            push_leaf_validity_and(&[lhs_expr, rhs_expr], table, snapshot, program)
        }
        _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident predicate comparison cannot be literal-vs-literal (constant-folding pending)"
                .to_string(),
        ))),
    }
}

/// Compile a TEXT comparison leaf (`textcol = 'lit'` / `<>`) into a `TextEqMask` VM step + record its
/// needle bytes in `needles` (indexed by `needle_idx`). Mirrors the single-comparison text fast path
/// but as a mask the VM can AND/OR. `=` -> negate false, `<>` -> negate true; text inequalities (need
/// collation sort keys) and text column-vs-column are follow-ons, rejected here (never mis-answered).
fn compile_text_eq_leaf(
    op: ResidentBinaryOp,
    lhs: &ResidentExpr,
    rhs: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
    needles: &mut Vec<Vec<u8>>,
) -> Result<(), ExecuteError> {
    let negate = match op {
        ResidentBinaryOp::Eq => false,
        ResidentBinaryOp::Ne => true,
        _ => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "text inequalities need collation sort keys (a follow-on); only = and <> run on the GPU"
                    .to_string(),
            )));
        }
    };
    let (col, literal) = match (text_column_index(lhs, table), text_column_index(rhs, table)) {
        (Some(col), None) if text_literal_value(rhs).is_some() => {
            (col, text_literal_value(rhs).expect("checked"))
        }
        (None, Some(col)) if text_literal_value(lhs).is_some() => {
            (col, text_literal_value(lhs).expect("checked"))
        }
        (Some(_), Some(_)) => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "text column-vs-column comparison is a follow-on".to_string(),
            )));
        }
        _ => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "text comparison must be a column against a literal".to_string(),
            )));
        }
    };
    let layout = resident_device_text_column_layout(snapshot, table, col)?;
    let needle_idx = needles.len() as u32;
    needles.push(literal.as_bytes().to_vec());
    program.push(ExprStep::TextEqMask {
        offsets_byte_offset: layout.offsets_byte_offset,
        bytes_byte_offset: layout.bytes_byte_offset,
        needle_idx,
        negate,
    });
    // 3VL: a NULL text operand makes `=`/`<>` UNKNOWN ⇒ the row is not selected (its placeholder is an
    // empty span, which would otherwise mis-match `= ''` / mis-pass `<> 'x'`).
    push_column_validity_and(col, table, snapshot, program)
}

/// Compile a TEXT `LIKE` leaf (`textcol LIKE 'pattern'`) into a `TextLikeMask` VM step + record the
/// COMPILED pattern tokens (LE-serialized) in `needles` (indexed by `pattern_idx`). The mask-VM sibling of
/// the standalone `try_lower_text_predicate` LIKE path: the standalone path serves a non-null single LIKE
/// (it returns indices directly), but a NULLABLE-text or compound LIKE routes through the mask VM, which
/// had no LIKE step (only `Eq`/`Ne`). `LIKE` is NOT symmetric — the column is the lhs, the pattern literal
/// the rhs (PG: `x ~~ y`). The tokens are the host-compiled u32 array (`compile_like_pattern`, escapes
/// resolved), serialized to bytes for the generic `text_needles` channel; the kernel reads `ntok = len/4`.
fn compile_text_like_leaf(
    lhs: &ResidentExpr,
    rhs: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
    needles: &mut Vec<Vec<u8>>,
) -> Result<(), ExecuteError> {
    let (col, pattern) = match (text_column_index(lhs, table), text_literal_value(rhs)) {
        (Some(col), Some(pattern)) => (col, pattern),
        _ => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "LIKE must be a text column LIKE a literal pattern".to_string(),
            )));
        }
    };
    let tokens = compile_like_pattern(pattern)?;
    let layout = resident_device_text_column_layout(snapshot, table, col)?;
    let pattern_idx = needles.len() as u32;
    needles.push(tokens.iter().flat_map(|t| t.to_le_bytes()).collect());
    program.push(ExprStep::TextLikeMask {
        offsets_byte_offset: layout.offsets_byte_offset,
        bytes_byte_offset: layout.bytes_byte_offset,
        pattern_idx,
    });
    // 3VL: a NULL text operand makes `LIKE` UNKNOWN ⇒ the row is excluded. A NULL's placeholder is an
    // empty span (start==end), which already fails any non-empty pattern; but `LIKE '%'` matches the empty
    // span, so the validity AND is load-bearing (PG: `NULL LIKE '%'` is UNKNOWN, not TRUE).
    push_column_validity_and(col, table, snapshot, program)
}

/// Compile a BOOL comparison leaf (`boolcol =/<> true|false`, either operand order) into a `BoolMask` VM
/// step the mask VM combines with AND/OR. `= true` -> the set bits (negate false); `= false` (and the
/// upstream `NOT flag` -> `flag = false`) -> the clear bits (negate true); `<>` is the complement.
/// Mirrors the single-comparison bool fast path but as a mask. Ordering ops are rejected (never mis-run).
fn compile_bool_leaf(
    op: ResidentBinaryOp,
    lhs: &ResidentExpr,
    rhs: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<(), ExecuteError> {
    let is_bool_col =
        |idx: usize| table.columns.get(idx).map(|column| column.ty) == Some(SqlType::Bool);
    let (col, literal) = match (lhs, rhs) {
        (ResidentExpr::Column(col), ResidentExpr::BoolLiteral(b)) if is_bool_col(*col) => (*col, *b),
        (ResidentExpr::BoolLiteral(b), ResidentExpr::Column(col)) if is_bool_col(*col) => (*col, *b),
        _ => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a bool comparison must be a bool column against a true/false literal".to_string(),
            )));
        }
    };
    let negate = match op {
        ResidentBinaryOp::Eq => !literal,
        ResidentBinaryOp::Ne => literal,
        _ => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only = / <> against a bool literal".to_string(),
            )));
        }
    };
    let offset = resident_device_bool_column_offset(snapshot, table, col)?;
    program.push(ExprStep::BoolMask {
        bitmap_byte_offset: offset,
        negate,
    });
    // 3VL: a NULL bool operand makes `=`/`<>` UNKNOWN ⇒ not selected. The value-bitmap bit of a NULL row
    // is the 0 placeholder, so `= false` / `<> true` would otherwise wrongly select it; AND with the
    // validity mask excludes it.
    push_column_validity_and(col, table, snapshot, program)
}

/// The table payload `execute_resident_expr_select_with_binding` reads its rows from: the published
/// device buffer + the matching catalog/GPU descriptor (whose `row_count` + `resident_device_*` section
/// vectors define every column byte-offset) + the row count. Passing `None` makes the executor look these
/// up by table name in the SINGLE resident store (the whole-table buffer), byte-identical to before;
/// passing `Some(src)` INJECTS them so the SAME executor can serve one PARTITION slice (S10c) -- the
/// injected descriptor's `row_count` + section vectors describe that slice. Whole-table callers pass `None`.
pub(crate) struct ResidentExecSource {
    pub(crate) descriptor: std::sync::Arc<RelationalResidencySnapshot>,
    pub(crate) device_memory: std::sync::Arc<gpu_db_execution::CudaResidentDeviceMemory>,
    pub(crate) row_count: u64,
}

impl Engine {
    /// General GPU executor entry (Charter rule 2): run `SELECT <int4 columns> FROM <table>` filtered
    /// by a general predicate [`ResidentExpr`], evaluating the predicate on the GPU via the device
    /// interpreter and materializing the surviving rows by gathering the projected columns. The
    /// predicate is supplied as an expression tree (the unit of execution is an expression, not a
    /// recognized shape); the `Select` carries the table + projection + MVCC binding. NOT routed
    /// through `resident_route_query_shape` — this is the general path, parallel to the (frozen)
    /// enumerated probe dispatch.
    // Forward API: run a select with a PROGRAMMATIC predicate (binds the catalog internally). The GPU
    // parity tests use it; the production caller is the SQL->Expr entry, which binds once itself and
    // calls `execute_resident_expr_select_with_binding` directly, so this wrapper has no non-test
    // caller yet.
    #[allow(dead_code)]
    pub(crate) fn execute_resident_expr_select(
        &self,
        select: &Select,
        predicate: &ResidentExpr,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        self.execute_resident_expr_select_with_binding(
            select,
            &table,
            None,
            bound,
            copin_s,
            Some(predicate),
            &[],
            &[],
            None,
            &[],
        )
    }

    /// `&Select`->general-executor BRIDGE (S8 grouped int4 aggregates; S10a non-grouped projections).
    /// Runs a `Select` through the SAME on-device general executor as the SQL->Expr path, retiring the
    /// legacy resident-probe methods. For a GROUPED select it normalizes the legacy 1-aggregate
    /// projection and runs the grouped path (the int4-group + int4-value shapes the route classifier
    /// accepts); for a NON-grouped select (`group_by == None`) `grouped_projection_to_aggregates` is a
    /// no-op and the group-key list is EMPTY, so the binding executor runs the plain-projection path
    /// (WHERE VM + GPU sort + LIMIT/OFFSET window) -- this is how S10a routes the `int4_ordered_projection`
    /// shape, replacing its probe. The grouped probes' `!gpu_ordered` branches did a HOST sort / HAVING /
    /// LIMIT (a charter violation -- relational finalization on the host). The general executor does ORDER BY / HAVING / LIMIT
    /// ON-DEVICE (S2/S3/S4), so this is behavior-preserving (enumerated == general proven 0/24
    /// differential; bridge == general re-verified before the probe methods were deleted; the general
    /// grouped ORDER BY now appends a group-key tie-break, matching the legacy group-ASC tie order).
    ///
    /// Unlike `execute_resident_expr_select_sql`, this sources its inputs from the engine `&Select`
    /// (the hand-rolled parse) rather than the libpg_query parse tree, so it covers the text entry AND
    /// the `&Select` callers with no raw SQL -- CTAS and view/matview -- uniformly. The WHERE predicate
    /// is rebuilt from the bound's resolved filters (a `ResidentExpr` DNF -- the third predicate path),
    /// then the bound filters are CLEARED so the executor filters SOLELY via the predicate, exactly as
    /// the SQL->Expr path does (whose bound carries no filters -- the WHERE rides the predicate). The
    /// grouped ORDER BY keys are RESULT columns (the group column or an aggregate), resolved by the
    /// executor against the projection, so every `order_by_exprs` entry is `None` (a plain key, not an
    /// expression); NULLS FIRST/LAST is `None` (PG default) since the hand-rolled `SelectOrder` carries
    /// no explicit override -- byte-identical to the probe path it replaces.
    pub(crate) fn execute_resident_grouped_via_general(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        // Normalize the legacy 1-aggregate grouped projection (GroupedCount / GroupedSum / GroupedAvg /
        // GroupedMin / GroupedMax produced by the hand-rolled parser) to the general GroupedAggregates
        // form the Expr executor consumes -- and bind against THAT, so the binding (selected columns,
        // result schema, ORDER-BY-result-column resolution) is byte-identical to the SQL->Expr path,
        // which always produces GroupedAggregates. The WHERE / GROUP BY / ORDER BY / HAVING / LIMIT are
        // carried over unchanged in the clone.
        let mut select_owned = select.clone();
        if let Some(projection) = grouped_projection_to_aggregates(&select.projection) {
            select_owned.projection = projection;
        }
        let select = &select_owned;
        let (table, mut bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        // Rebuild the WHERE predicate from the bound filters BEFORE clearing them; then clear so the
        // executor's filter (and the access-path planner it feeds) sees an empty-filter bound, matching
        // the SQL->Expr path exactly. The predicate ResidentExpr is now the sole filter.
        let predicate = resident_predicate_from_bound_filters(&bound)?;
        bound.filter = None;
        bound.filters.clear();
        bound.filter_groups.clear();
        // Single bare-column GROUP BY (the only grouped shape the route accepts); the executor packs a
        // composite key when len()==2, but the grouped routes are single-key, so this is one column.
        let group_key_columns: Vec<String> = match &select.group_by {
            Some(column) => vec![column.clone()],
            None => Vec::new(),
        };
        // Grouped ORDER BY keys are result columns (None = plain key); no expression GROUP BY here.
        let order_by_exprs: Vec<Option<ResidentExpr>> = vec![None; select.order_by.len()];
        let order_by_nulls_first: Vec<Option<bool>> = vec![None; select.order_by.len()];
        self.execute_resident_expr_select_with_binding(
            select,
            &table,
            None,
            bound,
            copin_s,
            predicate.as_ref(),
            &order_by_exprs,
            &order_by_nulls_first,
            None,
            &group_key_columns,
        )
    }

    /// S10b: `&Select`->general BRIDGE for a single-column `SELECT DISTINCT` (the int4_[filtered_]distinct
    /// route shapes). DISTINCT over column `a` is exactly `GROUP BY a` with no aggregate; the most-audited
    /// on-device grouped path (S8) is built around >=1 aggregate, so synthesize a `COUNT(*)` grouped select,
    /// run it through `execute_resident_grouped_via_general` (one row per distinct key, ON THE DEVICE -- the
    /// retired probe deduped on a HOST `BTreeSet`), then DROP the trailing COUNT column. WHERE / ORDER BY a /
    /// LIMIT / OFFSET ride the grouped select unchanged. **PG-correct on NULLs** (a behavior change vs the
    /// retired probe, like the S10a ordered projection): GROUP BY groups a NULL key into one group (M3
    /// NULL-key slot) -> DISTINCT yields ONE `SqlValue::Null` row, whereas the NULL-blind probe read the int4
    /// column directly and surfaced a NULL as a phantom `Int4(0)`. For the FILTERED shape a NULL fails the
    /// range predicate (3VL) so it is excluded either way. **No-ORDER-BY order:** the probe returned
    /// first-seen order; the grouped path returns the deterministic default order (key ASC) -- same SET, a
    /// PG-unspecified sequence made deterministic (like the S8 tie-break).
    pub(crate) fn execute_resident_distinct_via_general(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let SelectProjection::Columns(columns) = &select.projection else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident DISTINCT bridge requires a column projection".to_string(),
            )));
        };
        let [column] = columns.as_slice() else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident DISTINCT bridge currently supports a single projected column".to_string(),
            )));
        };
        // SELECT DISTINCT a [WHERE ..] [ORDER BY a] [LIMIT ..] == SELECT a, COUNT(*) FROM t [WHERE ..]
        // GROUP BY a [ORDER BY a] [LIMIT ..], with the COUNT column dropped from the result.
        let mut grouped = select.clone();
        grouped.distinct = false;
        grouped.group_by = Some(column.clone());
        grouped.projection = SelectProjection::GroupedAggregates {
            group_column: column.clone(),
            aggregates: vec![GroupedAggregate {
                kind: GroupedAggKind::Count,
                value_column: None,
            }],
        };
        let mut result = self.execute_resident_grouped_via_general(&grouped)?;
        // Drop the trailing COUNT(*) column -> the bare distinct keys (column 0 is the group key).
        result.columns.truncate(1);
        for row in &mut result.rows {
            row.truncate(1);
        }
        Ok(result)
    }

    /// Benchmark helper (tests only): run GROUP BY on a resident table with the SINGLE-LEVEL or
    /// TWO-LEVEL kernel selected explicitly, over a full-table scan. Returns the result rows so the
    /// caller can confirm both kernels agree; the caller times repeated calls. Not on the query path.
    #[cfg(test)]
    pub(crate) fn group_by_i32_bench(
        &self,
        table_name: &str,
        key_col: &str,
        value_col: &str,
        two_level: bool,
    ) -> Result<Vec<gpu_db_execution::GroupByI32Row>, ExecuteError> {
        let table = self.relational_catalog_table(table_name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!("no table {table_name}")))
        })?;
        let snapshot = self
            .relational_residency_snapshot_ref(table_name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed("no resident snapshot".to_string()))
            })?;
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(table_name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed("no device memory".to_string()))
            })?;
        let key_idx = relational_column_index(&table, key_col)?;
        let val_idx = relational_column_index(&table, value_col)?;
        let key_off = resident_device_int4_column_offset(&snapshot, &table, key_idx)?;
        let val_off = resident_device_int4_column_offset(&snapshot, &table, val_idx)?;
        let n = u32::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed("rows exceed u32".to_string()))
        })?;
        let indices: Vec<u32> = (0..n).collect();
        device_memory
            .group_by_i32_count_sum_bench(key_off, val_off, &indices, two_level)
            .map_err(|e| ExecuteError::Engine(EngineError::ApplyFailed(e.to_string())))
    }

    /// Benchmark helper (tests only): time JUST the GROUP BY kernel (CUDA events, min of `runs`),
    /// isolating it from the alloc/H2D/D2H/host-compact overhead. Returns the min kernel milliseconds.
    #[cfg(test)]
    pub(crate) fn group_by_i32_bench_kernel_ms(
        &self,
        table_name: &str,
        key_col: &str,
        value_col: &str,
        two_level: bool,
        runs: u32,
        rows_limit: usize,
    ) -> Result<f32, ExecuteError> {
        let table = self.relational_catalog_table(table_name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!("no table {table_name}")))
        })?;
        let snapshot = self
            .relational_residency_snapshot_ref(table_name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed("no resident snapshot".to_string()))
            })?;
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(table_name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed("no device memory".to_string()))
            })?;
        let key_idx = relational_column_index(&table, key_col)?;
        let val_idx = relational_column_index(&table, value_col)?;
        let key_off = resident_device_int4_column_offset(&snapshot, &table, key_idx)?;
        let val_off = resident_device_int4_column_offset(&snapshot, &table, val_idx)?;
        let n = u32::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed("rows exceed u32".to_string()))
        })?;
        // rows_limit == 0 means the whole table; otherwise group only the first `rows_limit` rows (to
        // sweep data size at fixed cardinality from one residency).
        let m = if rows_limit == 0 {
            n
        } else {
            (rows_limit as u32).min(n)
        };
        let indices: Vec<u32> = (0..m).collect();
        device_memory
            .group_by_i32_count_sum_kernel_timed(key_off, val_off, &indices, two_level, runs)
            .map(|(_, ms)| ms)
            .map_err(|e| ExecuteError::Engine(EngineError::ApplyFailed(e.to_string())))
    }

    /// As [`Engine::execute_resident_expr_select`] but over an ALREADY-BOUND table/projection: the
    /// caller bound the catalog once and resolved the predicate's `Column` indices against this SAME
    /// `table`. The SQL->Expr entry (`engine_sql_pg`) routes through here so the predicate's column
    /// indices, the projection, and the residency snapshot all derive from one catalog generation — a
    /// concurrent shape-changing DDL cannot split the column resolution from the execution.
    /// Resolve one join relation to its (residency entry, device memory, row count): a RESIDENT user
    /// table uses its published snapshot + retained device memory; a SYNTHESIZED catalog relation
    /// (`rows = Some`, M5 J5) is uploaded as a TRANSIENT device payload + descriptor that lives only for
    /// this query. Either way the join runs the SAME GPU pre-filter + key-projection + hash-join kernels
    /// over the result -- no CPU relational join (charter).
    fn resolve_join_side(
        &self,
        name: &str,
        table: &RelationalTable,
        rows: Option<Vec<Vec<SqlValue>>>,
    ) -> Result<(RelationalResidencyEntry, JoinDeviceMemory, usize), ExecuteError> {
        match rows {
            None => {
                let entry = self.relational_residency_entry(name).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{name}\" has no resident snapshot (the join path is GPU-only)"
                    )))
                })?;
                if !entry.descriptor.is_valid() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{name}\" resident snapshot is invalid"
                    ))));
                }
                let row_count = entry.descriptor.row_count;
                let memory = self
                    .read_state
                    .residency
                    .device_memory
                    .get(name)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "relation \"{name}\" has no retained resident device memory"
                        )))
                    })?;
                Ok((entry, JoinDeviceMemory::Resident(memory), row_count))
            }
            Some(rows) => {
                let (snapshot, memory) = self.build_transient_relation_residency(table, &rows)?;
                let row_count = rows.len();
                let entry = RelationalResidencyEntry {
                    descriptor: std::sync::Arc::new(snapshot),
                    host_rows: std::sync::Arc::new(rows),
                };
                Ok((entry, JoinDeviceMemory::Transient(memory), row_count))
            }
        }
    }

    /// Whether a per-relation WHERE predicate evaluates to TRUE on an OUTER join's synthetic all-NULL pad
    /// row -- decided ON-DEVICE (S6 -- doc 22), replacing the host Kleene `predicate_truth_on_null_pad`.
    /// Build a 1-row TRANSIENT relation whose every column is NULL and run the predicate through the SAME
    /// GPU WHERE-3VL mask VM as the real rows (`lower_resident_predicate`); the pad survives iff row 0
    /// survives. On the all-NULL row a comparison/arithmetic/bare-column leaf is AND'd with the (all-zero)
    /// validity mask -> UNKNOWN -> excluded; `col IS NULL` reads the 0 validity bit -> TRUE (the anti-join);
    /// AND/OR fold via the VM's Kleene masks. So the host no longer evaluates 3VL -- the GPU does. A
    /// predicate shape the VM cannot lower over a nullable column clean-errors there (the engine-wide WHERE
    /// 3VL limitation), exactly as the real-row survivor pass would for the same shape.
    fn predicate_holds_on_null_pad(
        &self,
        predicate: &ResidentExpr,
        table: &RelationalTable,
    ) -> Result<bool, ExecuteError> {
        let null_row = vec![vec![SqlValue::Null; table.columns.len()]];
        let (snapshot, memory) = self.build_transient_relation_residency(table, &null_row)?;
        let survivors = self.lower_resident_predicate(predicate, table, &snapshot, &memory, 1)?;
        Ok(!survivors.is_empty())
    }

    /// Execute a LEFT-DEEP chain of INNER equi-joins (M5) on the GPU. `tables`/`rows`/`predicates` are
    /// parallel to `plan.relations` (one per relation, all bound at ONE catalog generation by the
    /// caller). Each step GPU-pre-filters the newly joined relation by its WHERE, projects the two
    /// int-key columns, and runs the GPU hash join (build on whichever side has a UNIQUE key; N:N is a
    /// follow-up). The pipeline carries, per relation, the ABSOLUTE row index each surviving tuple came
    /// from -- so the only growing host state is index vectors (control-plane, like survivors), each
    /// step's keys are projected on the GPU from the source payloads, and ONLY the final gather reads
    /// host_rows. Every relational op (WHERE filter, key compare, hash match) runs on the GPU; no CPU
    /// relational join (charter). A 2-relation join is the N=2 case (a single step), matching the prior
    /// 2-way path. The intermediate result is never materialized to a device payload -- the carried
    /// indices let each later step project keys straight from the original relations' payloads.
    pub(crate) fn execute_resident_expr_inner_join(
        &self,
        plan: &JoinPlan,
        tables: Vec<RelationalTable>,
        rows: Vec<Option<Vec<Vec<SqlValue>>>>,
        predicates: Vec<Option<ResidentExpr>>,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let n_rel = plan.relations.len();
        // OUTER joins (LEFT/RIGHT/FULL, M3 -- doc 21). N-way (multi-step) OUTER is supported: a prior step's
        // NULL pad carries a `JOIN_NULL_ROW` sentinel whose validity is gathered as 0, so the hash-join
        // kernel skips it as a NULL key (matches nothing; a LEFT step re-pads it), and the final gather
        // emits SqlValue::Null for it.
        //
        // A WHERE on an OUTER join is NOT filter-commutative: pushing a per-side predicate down before the
        // join would drop rows the outer join must NULL-pad, and a predicate on the padded side would
        // change which rows are padded. So for an outer join the per-side predicates are NOT pushed down
        // (the join sees ALL rows); instead each predicate's GPU-computed survivor set becomes a POST-join
        // membership filter applied to the result below: a tuple survives only if, for every predicated
        // relation, its carried row is REAL (a JOIN_NULL_ROW pad means that relation's columns are NULL ->
        // the predicate is UNKNOWN -> drop) AND that row passed the predicate on-device. This matches PG
        // (a WHERE on the inner side of a LEFT join effectively makes it inner on that condition).
        let outer_where = plan.steps.iter().any(|s| s.outer_left || s.outer_right)
            && predicates.iter().any(Option::is_some);
        // Resolve a JOIN column reference to (relation index, column index) against relations[0..=upto]:
        // a qualifier must name exactly one of them; an unqualified column must be in exactly one (PG's
        // "ambiguous" / "does not exist"). `upto` bounds an ON operand to the relations joined so far.
        let resolve = |c: &JoinColRef, upto: usize| -> Result<(usize, usize), ExecuteError> {
            match &c.qualifier {
                Some(q) => {
                    for (i, rel) in plan.relations.iter().enumerate().take(upto + 1) {
                        if rel.alias == *q {
                            return Ok((i, relational_column_index(&tables[i], &c.column)?));
                        }
                    }
                    Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "missing FROM-clause entry for table \"{q}\""
                    ))))
                }
                None => {
                    let mut found: Option<(usize, usize)> = None;
                    for (i, table) in tables.iter().take(upto + 1).enumerate() {
                        if let Ok(ci) = relational_column_index(table, &c.column) {
                            if found.is_some() {
                                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                    "column reference \"{}\" is ambiguous",
                                    c.column
                                ))));
                            }
                            found = Some((i, ci));
                        }
                    }
                    found.ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "column \"{}\" does not exist",
                            c.column
                        )))
                    })
                }
            }
        };
        // A single-conjunct key may be any int (int2/int4/date in the i32 section, int8/timestamp in the
        // i64 section -- both project to i64, so a mixed int4=int8 equi-join compares correctly). A
        // COMPOSITE (2-conjunct) key packs two members into one i64 (member0 in the high 32 bits, member1
        // in the low), so each member must be <=32 bits (int2/int4/date); an int8/timestamp composite
        // member (or >2 conjuncts) overflows 64 bits -> a follow-up. numeric/uuid/text keys are J4b/c.
        let int_key = |t: SqlType| {
            matches!(
                t,
                SqlType::Int4 | SqlType::Int2 | SqlType::Date | SqlType::Int8 | SqlType::Timestamp
            )
        };
        let narrow_key = |t: SqlType| matches!(t, SqlType::Int4 | SqlType::Int2 | SqlType::Date);
        // Pre-resolve each step's ON conjuncts + validate the key types, and resolve the SELECT
        // projection, BEFORE touching residency -- so a malformed query (unknown column, ambiguous ref,
        // non-int key) fails fast with a query error, not a "not resident" one. `step_keys[k]` = the
        // per-conjunct (accumulated relation, its key column, the new relation's key column); the newly
        // joined relation is `k+1`. Conjuncts may reference DIFFERENT accumulated relations.
        let mut step_keys: Vec<Vec<(usize, usize, usize)>> = Vec::with_capacity(plan.steps.len());
        // Per step's key kind: TEXT or NUMERIC/UUID (b128, both -> the GPU text/byte hash join) vs INT
        // (the i64 path). `step_is_text` = TEXT bytes; `step_is_b128` = a 16-byte numeric/uuid value.
        let mut step_is_text: Vec<bool> = Vec::with_capacity(plan.steps.len());
        let mut step_is_b128: Vec<bool> = Vec::with_capacity(plan.steps.len());
        // USING/NATURAL join columns, coalesced (emitted ONCE in `*`, resolvable unqualified). 2-relation
        // only (build_join_plan rejects multi-way USING/NATURAL), so this collects a single step's set.
        let mut coalesce_cols: Vec<String> = Vec::new();
        for (k, step) in plan.steps.iter().enumerate() {
            let new_rel = k + 1;
            let mut conj_keys: Vec<(usize, usize, usize)> = Vec::new();
            let step_coalesce: Vec<String> = if step.natural {
                // NATURAL (2-relation, so new_rel == 1 and acc == 0): join on the relations' COMMON column
                // names -> conjuncts (rel0.c = rel1.c) + the coalesce set.
                let mut common: Vec<String> = Vec::new();
                for (c0, col) in tables[0].columns.iter().enumerate() {
                    if let Ok(c1) = relational_column_index(&tables[new_rel], &col.name) {
                        conj_keys.push((0, c0, c1));
                        common.push(col.name.clone());
                    }
                }
                if common.is_empty() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "NATURAL JOIN has no common column name between the two relations".to_string(),
                    )));
                }
                common
            } else {
                for (on_a, on_b) in &step.conjuncts {
                    // Each conjunct must equate the newly joined relation (`new_rel`) to an already-joined
                    // one (<new_rel).
                    let ra = resolve(on_a, new_rel)?;
                    let rb = resolve(on_b, new_rel)?;
                    let ((acc_rel, acc_col), new_col) = if rb.0 == new_rel && ra.0 < new_rel {
                        ((ra.0, ra.1), rb.1)
                    } else if ra.0 == new_rel && rb.0 < new_rel {
                        ((rb.0, rb.1), ra.1)
                    } else {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "each JOIN's ON conjunct must equate the newly joined relation to an \
                             already-joined one (a.k = b.k)"
                                .to_string(),
                        )));
                    };
                    conj_keys.push((acc_rel, acc_col, new_col));
                }
                step.coalesce.clone()
            };
            // A step joins on 1 or 2 columns (ON/comma guarantee >=1; NATURAL errored above on 0). It always
            // reaches `pack_keys`, which handles only 1-2 members; reject an empty step (defensive) and >2
            // columns (a composite key wider than 64 bits -- e.g. NATURAL/USING over 3+ columns).
            if conj_keys.is_empty() {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "a join step requires at least one equality condition between the relations"
                        .to_string(),
                )));
            }
            if conj_keys.len() > 2 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "a join on more than 2 equality columns (a composite key wider than 64 bits) is a \
                     follow-up"
                        .to_string(),
                )));
            }
            coalesce_cols.extend(step_coalesce);
            // Determine the key kind + validate types. A single-conjunct key may be TEXT (-> the GPU text
            // hash join: FNV + byte-verify), NUMERIC/UUID (-> the SAME kernel over the 16-byte canonical
            // value), or INT (-> the i64 hash join). A 2-conjunct composite is INT-only (i64-packed).
            let non_int = |t: SqlType| {
                matches!(t, SqlType::Text | SqlType::Numeric { .. } | SqlType::Uuid)
            };
            let has_non_int = conj_keys.iter().any(|&(acc_rel, acc_col, new_col)| {
                non_int(tables[acc_rel].columns[acc_col].ty)
                    || non_int(tables[new_rel].columns[new_col].ty)
            });
            let mut is_text = false;
            let mut is_b128 = false;
            if has_non_int {
                // text / numeric / uuid: a single conjunct, the SAME key type on both sides (numeric also
                // the same scale, so the i128 mantissa compares value-for-value).
                let (acc_rel, acc_col, new_col) = conj_keys[0];
                let acc_ty = tables[acc_rel].columns[acc_col].ty;
                let new_ty = tables[new_rel].columns[new_col].ty;
                let same_type = conj_keys.len() == 1
                    && match (acc_ty, new_ty) {
                        (SqlType::Text, SqlType::Text) => {
                            is_text = true;
                            true
                        }
                        (SqlType::Uuid, SqlType::Uuid) => {
                            is_b128 = true;
                            true
                        }
                        (
                            SqlType::Numeric { scale: s_acc, .. },
                            SqlType::Numeric { scale: s_new, .. },
                        ) if s_acc == s_new => {
                            is_b128 = true;
                            true
                        }
                        _ => false,
                    };
                if !same_type {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "a text/numeric/uuid join key must be a single `a.k = b.k` with the SAME type \
                         on BOTH sides (numeric: the same scale); composite / mixed / different-scale \
                         keys with these types are a follow-up"
                            .to_string(),
                    )));
                }
            } else {
                // A composite member must be <=32 bits so two pack into one i64; a single key may be int8.
                let composite = conj_keys.len() == 2;
                for &(acc_rel, acc_col, new_col) in &conj_keys {
                    let ok = |t: SqlType| if composite { narrow_key(t) } else { int_key(t) };
                    if !ok(tables[acc_rel].columns[acc_col].ty)
                        || !ok(tables[new_rel].columns[new_col].ty)
                    {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "a join key must be an integer column on both sides (int2/int4/int8/date/\
                             timestamp for a single key; int2/int4/date for each member of a 2-column \
                             composite key)"
                                .to_string(),
                        )));
                    }
                }
            }
            step_keys.push(conj_keys);
            step_is_text.push(is_text);
            step_is_b128.push(is_b128);
        }
        // Resolve the SELECT list to a flat (relation index, column index) list, expanding `*` and
        // `alias.*`. With USING/NATURAL, a join column is COALESCED: it appears ONCE in bare `*` (PG order:
        // the join columns first, then the left relation's other columns, then the right's), and an
        // UNQUALIFIED reference to it resolves to the left copy (not ambiguous). `alias.*` is unchanged
        // (a relation's own columns). `coalesce_cols` is empty for ON/comma joins -> the prior behavior.
        let is_coalesced = |name: &str| coalesce_cols.iter().any(|c| c == name);
        let mut proj: Vec<(usize, usize)> = Vec::new();
        for item in &plan.projection {
            match item {
                JoinProjItem::Column(c) if c.qualifier.is_none() && is_coalesced(&c.column) => {
                    // The coalesced join column lives in relation 0 (the left side of the 2-relation join).
                    proj.push((0, relational_column_index(&tables[0], &c.column)?));
                }
                JoinProjItem::Column(c) => proj.push(resolve(c, n_rel - 1)?),
                JoinProjItem::Star(None) if !coalesce_cols.is_empty() => {
                    // USING/NATURAL (2-relation): coalesced columns first (from rel0), then each relation's
                    // remaining columns left-to-right (the right copy of a coalesced column is skipped).
                    for name in &coalesce_cols {
                        proj.push((0, relational_column_index(&tables[0], name)?));
                    }
                    for (ri, table) in tables.iter().enumerate() {
                        proj.extend(
                            table
                                .columns
                                .iter()
                                .enumerate()
                                .filter(|(_, col)| !is_coalesced(&col.name))
                                .map(|(ci, _)| (ri, ci)),
                        );
                    }
                }
                JoinProjItem::Star(None) => {
                    for (ri, table) in tables.iter().enumerate() {
                        proj.extend((0..table.columns.len()).map(|ci| (ri, ci)));
                    }
                }
                JoinProjItem::Star(Some(alias)) => {
                    let ri = plan
                        .relations
                        .iter()
                        .position(|r| r.alias == *alias)
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "missing FROM-clause entry for table \"{alias}\""
                            )))
                        })?;
                    proj.extend((0..tables[ri].columns.len()).map(|ci| (ri, ci)));
                }
            }
        }
        // Resolve every relation's device payload, at the one bound catalog generation: a RESIDENT user
        // table's published memory, or a SYNTHESIZED catalog relation's transient upload.
        let mut sides: Vec<(RelationalResidencyEntry, JoinDeviceMemory, usize)> =
            Vec::with_capacity(n_rel);
        for (row_opt, (relation, table)) in
            rows.into_iter().zip(plan.relations.iter().zip(&tables))
        {
            sides.push(self.resolve_join_side(&relation.table, table, row_opt)?);
        }
        // The `JOIN_NULL_ROW` sentinel (LEFT-pad) must be distinguishable from every real absolute row
        // index -- it is, because residency never holds anywhere near u32::MAX rows. Make it explicit.
        debug_assert!(
            sides
                .iter()
                .all(|s| s.0.host_rows.len() < JOIN_NULL_ROW as usize),
            "a join relation has too many rows to distinguish the LEFT-join NULL-pad sentinel"
        );
        let gpu_id = sides[0].0.descriptor.gpu_id;
        // Filter each relation by its per-relation WHERE on the GPU -> the surviving ABSOLUTE row indices.
        // INNER join: this is a pre-filter pushed into the join inputs (filter-commutative). OUTER join
        // (outer_where): the join must see ALL rows, so `survivors_all` stays 0..n and the GPU survivor
        // set is recorded as a per-relation post-filter keep-mask (applied to the result after the join).
        let mut survivors_all: Vec<Vec<u32>> = Vec::with_capacity(n_rel);
        // keep_masks[i] = Some(mask over 0..n_i) when relation i has a predicate AND this is an outer join
        // (else None = no post-filter for that relation). mask[row] = the row passed the predicate.
        let mut keep_masks: Vec<Option<Vec<bool>>> = vec![None; n_rel];
        // pad_survives[i] = does relation i's predicate hold on a NULL pad (a tuple where relation i is
        // NULL-padded)? `col IS NULL` (the anti-join) holds -> such tuples are KEPT; comparisons are
        // UNKNOWN -> dropped. Resolved ON-DEVICE (S6 -- doc 22) by `predicate_holds_on_null_pad`: the
        // predicate's truth on the all-NULL pad is a GPU WHERE-3VL decision, not a host Kleene fold. Only
        // consulted for outer_where relations.
        let mut pad_survives: Vec<bool> = vec![false; n_rel];
        for (i, (entry, dm, n)) in sides.iter().enumerate() {
            let surv = match &predicates[i] {
                Some(p) if *n > 0 => self.lower_resident_predicate(
                    p,
                    &tables[i],
                    &entry.descriptor,
                    dm.mem(),
                    *n as u64,
                )?,
                _ => (0..*n as u32).collect(),
            };
            if outer_where && predicates[i].is_some() {
                pad_survives[i] =
                    self.predicate_holds_on_null_pad(predicates[i].as_ref().unwrap(), &tables[i])?;
                let mut mask = vec![false; *n];
                for &r in &surv {
                    mask[r as usize] = true;
                }
                keep_masks[i] = Some(mask);
                survivors_all.push((0..*n as u32).collect());
            } else {
                survivors_all.push(surv);
            }
        }
        // Project relation `ri`'s integer key column `col_idx` to host i64 at the given ABSOLUTE rows:
        // int8/timestamp from the i64 section; int2/int4/date from the i32 section, sign-extended.
        let key_i64 = |ri: usize, col_idx: usize, abs: &[u32]| -> Result<Vec<i64>, ExecuteError> {
            if abs.is_empty() {
                return Ok(Vec::new());
            }
            let (entry, dm, row_count) = &sides[ri];
            // V1b: a carried JOIN_NULL_ROW (a prior OUTER pad) is a NULL key the kernel skips on-device, so
            // gather a 0 PLACEHOLDER for it (the value is never used -- the validity bitmap marks it invalid).
            // A 0-row relation has no payload to read; every carried row is then a pad -> all placeholders.
            if *row_count == 0 {
                return Ok(vec![0; abs.len()]);
            }
            let idxs: Vec<u64> =
                abs.iter().map(|&i| if i == JOIN_NULL_ROW { 0 } else { u64::from(i) }).collect();
            match tables[ri].columns[col_idx].ty {
                SqlType::Int8 | SqlType::Timestamp => {
                    let off =
                        resident_device_int8_column_offset(&entry.descriptor, &tables[ri], col_idx)?;
                    dm.mem().project_i64_rows_from_payload(off, &idxs).map_err(map_err)
                }
                _ => {
                    let off =
                        resident_device_int4_column_offset(&entry.descriptor, &tables[ri], col_idx)?;
                    Ok(dm
                        .mem()
                        .project_i32_rows_from_payload(off, &idxs)
                        .map_err(map_err)?
                        .into_iter()
                        .map(i64::from)
                        .collect())
                }
            }
        };
        // GPU hash join over two host i64 key arrays: build on the SMALLER side; if its key is
        // non-unique, retry building on the OTHER side (covers 1:1 + 1:N either way). Both non-unique =>
        // N:N (a follow-up). `ctx` only supplies the CUDA context (the kernel uploads the host keys); the
        // result is independent of which relation's memory hosts it. Returns (left_is_build, build, probe).
        use gpu_db_execution::HashJoinOutcome;
        // `left_valid`/`right_valid` are the optional per-side NULL-key validity bitmaps (V1b); they SWAP in
        // lockstep with the keys whenever build/probe swap, so the kernel always pairs each key array with
        // its own bitmap. The N:N path builds the chain on the LEFT (acc) side -> build_validity = left.
        let hash_join = |ctx: &gpu_db_execution::CudaResidentDeviceMemory,
                         left_keys: &[i64],
                         right_keys: &[i64],
                         left_valid: Option<&[u32]>,
                         right_valid: Option<&[u32]>|
         -> Result<(bool, Vec<u32>, Vec<u32>), ExecuteError> {
            let smaller_is_left = left_keys.len() <= right_keys.len();
            let (first_build, first_probe, first_bv, first_pv) = if smaller_is_left {
                (left_keys, right_keys, left_valid, right_valid)
            } else {
                (right_keys, left_keys, right_valid, left_valid)
            };
            match ctx.hash_join_inner_i64(first_build, first_probe, first_bv, first_pv).map_err(map_err)? {
                HashJoinOutcome::Pairs { build_idxs, probe_idxs } => {
                    Ok((smaller_is_left, build_idxs, probe_idxs))
                }
                HashJoinOutcome::DuplicateBuildKey => {
                    // Retry building the OTHER side -- swap keys AND their bitmaps.
                    match ctx.hash_join_inner_i64(first_probe, first_build, first_pv, first_bv).map_err(map_err)? {
                        HashJoinOutcome::Pairs { build_idxs, probe_idxs } => {
                            Ok((!smaller_is_left, build_idxs, probe_idxs))
                        }
                        HashJoinOutcome::DuplicateBuildKey => {
                            // N:N: both sides have duplicate keys -> the chaining many-to-many join
                            // (each key's build rows x probe rows). Build the chain on the LEFT (acc)
                            // side, so the result is (acc position, new position) directly.
                            let (build_idxs, probe_idxs) = ctx
                                .hash_join_inner_i64_nn(left_keys, right_keys, left_valid, right_valid)
                                .map_err(map_err)?;
                            Ok((true, build_idxs, probe_idxs))
                        }
                    }
                }
            }
        };
        // The TEXT analogue (M5 J4b): build on the smaller side (by row count), unique-build fallback. The
        // kernel FNV-hashes + byte-verifies, so a 64-bit hash collision between distinct texts never
        // mis-joins or spuriously reports a duplicate. Returns (left_is_build, build, probe).
        let text_hash_join = |ctx: &gpu_db_execution::CudaResidentDeviceMemory,
                              left: &[&[u8]],
                              right: &[&[u8]],
                              left_valid: Option<&[u32]>,
                              right_valid: Option<&[u32]>|
         -> Result<(bool, Vec<u32>, Vec<u32>), ExecuteError> {
            let smaller_is_left = left.len() <= right.len();
            let (first_build, first_probe, first_bv, first_pv) = if smaller_is_left {
                (left, right, left_valid, right_valid)
            } else {
                (right, left, right_valid, left_valid)
            };
            match ctx.hash_join_inner_text(first_build, first_probe, first_bv, first_pv).map_err(map_err)? {
                HashJoinOutcome::Pairs { build_idxs, probe_idxs } => {
                    Ok((smaller_is_left, build_idxs, probe_idxs))
                }
                HashJoinOutcome::DuplicateBuildKey => {
                    match ctx.hash_join_inner_text(first_probe, first_build, first_pv, first_bv).map_err(map_err)? {
                        HashJoinOutcome::Pairs { build_idxs, probe_idxs } => {
                            Ok((!smaller_is_left, build_idxs, probe_idxs))
                        }
                        HashJoinOutcome::DuplicateBuildKey => {
                            // N:N: both sides have duplicate text/numeric/uuid keys -> the chaining
                            // many-to-many text join (build the chain on the LEFT/acc side -> (acc, new)).
                            let (build_idxs, probe_idxs) = ctx
                                .hash_join_inner_text_nn(left, right, left_valid, right_valid)
                                .map_err(map_err)?;
                            Ok((true, build_idxs, probe_idxs))
                        }
                    }
                }
            }
        };
        // Gather relation `ri`'s TEXT key bytes from its DEVICE payload at the given ABSOLUTE rows (charter:
        // the keys come from device memory, not a host_rows copy -- like the i64 projection `key_i64`).
        // Returns OWNED strings (the caller borrows them into &[u8] for the GPU text hash join). The
        // precompute validated the column is TEXT. V1b: `abs` may include a carried JOIN_NULL_ROW (a prior
        // OUTER pad) or a NULL-key row -> the gather uses a 0 placeholder index (the kernel skips it via the
        // validity bitmap); a NULL cell at a real row projects as "" but is likewise skipped.
        let key_texts = |ri: usize, col_idx: usize, abs: &[u32]| -> Result<Vec<String>, ExecuteError> {
            if abs.is_empty() {
                return Ok(Vec::new());
            }
            let (entry, dm, row_count) = &sides[ri];
            if *row_count == 0 {
                return Ok(vec![String::new(); abs.len()]);
            }
            let layout = resident_device_text_column_layout(&entry.descriptor, &tables[ri], col_idx)?;
            let idxs: Vec<u64> =
                abs.iter().map(|&r| if r == JOIN_NULL_ROW { 0 } else { u64::from(r) }).collect();
            dm.mem()
                .project_text_rows_from_payload(
                    layout.offsets_byte_offset,
                    layout.bytes_byte_offset,
                    layout.bytes_len,
                    &idxs,
                )
                .map_err(map_err)
        };
        // Gather a NUMERIC/UUID key's 16-byte canonical value from its DEVICE payload (both ride the i128
        // section): the i128's LE bytes ARE the uuid raw bytes / the numeric mantissa LE (the precompute
        // requires equal scale on both sides, so the mantissa compares value-for-value). Returns OWNED
        // 16-byte values (the caller borrows them into &[u8] for the SAME text hash join: a 16-byte "text"
        // -> FNV + 16-byte verify = b128 equality). V1b: a carried JOIN_NULL_ROW / NULL-key row uses a 0
        // placeholder index (the kernel skips it via the validity bitmap).
        let key_b128 = |ri: usize, col_idx: usize, abs: &[u32]| -> Result<Vec<[u8; 16]>, ExecuteError> {
            if abs.is_empty() {
                return Ok(Vec::new());
            }
            let (entry, dm, row_count) = &sides[ri];
            if *row_count == 0 {
                return Ok(vec![[0u8; 16]; abs.len()]);
            }
            let off = resident_device_numeric_column_offset(&entry.descriptor, &tables[ri], col_idx)?;
            let idxs: Vec<u64> =
                abs.iter().map(|&r| if r == JOIN_NULL_ROW { 0 } else { u64::from(r) }).collect();
            Ok(dm
                .mem()
                .project_i128_rows_from_payload(off, &idxs)
                .map_err(map_err)?
                .into_iter()
                .map(|v| v.to_le_bytes())
                .collect())
        };
        // Gather relation `ri`'s NULL validity (`true` = the value is present) for key column `col` at the
        // carried ABSOLUTE rows, READ FROM THE DEVICE payload (charter: the NULL decision comes from device
        // memory via the same validity bitmap the result gather uses, never a host host_rows inspection). A
        // carried JOIN_NULL_ROW (a prior OUTER pad) is a NULL key -> invalid (placeholder index 0); a column
        // with no validity bitmap (no nullable values) -> every real row valid. A 0-row relation contributes
        // only pads here (carried from a prior OUTER step) -> all invalid, with no device read.
        let key_validity = |ri: usize, col: usize, abs: &[u32]| -> Result<Vec<bool>, ExecuteError> {
            let n = abs.len();
            if n == 0 {
                return Ok(Vec::new());
            }
            let (entry, dm, row_count) = &sides[ri];
            if *row_count == 0 {
                return Ok(vec![false; n]);
            }
            let is_pad: Vec<bool> = abs.iter().map(|&r| r == JOIN_NULL_ROW).collect();
            let idxs: Vec<u64> =
                abs.iter().map(|&r| if r == JOIN_NULL_ROW { 0 } else { u64::from(r) }).collect();
            let mut valid = match resident_device_null_column_offset(&entry.descriptor, &tables[ri], col)? {
                Some(off) => dm.mem().project_bool_rows_from_payload(off, &idxs).map_err(map_err)?,
                None => vec![true; n],
            };
            for (v, &pad) in valid.iter_mut().zip(is_pad.iter()) {
                if pad {
                    *v = false;
                }
            }
            Ok(valid)
        };
        // Pack a per-row validity vector (`true` = the key is non-NULL) into a dense LSB-first u32 bitmap:
        // bit `i` lives in `word[i>>5]` at `i & 31`, `1 = valid` -- exactly what the hash-join kernels read
        // (`shr 5` word, `bfe ...,1` bit). All-valid -> None -> the kernel's u64::MAX sentinel fast path,
        // keeping the common no-NULL join byte-identical to the pre-V1b host-filter path.
        let pack_validity = |valid: &[bool]| -> Option<Vec<u32>> {
            if valid.iter().all(|&v| v) {
                return None;
            }
            let mut words = vec![0u32; valid.len().div_ceil(32)];
            for (i, &v) in valid.iter().enumerate() {
                if v {
                    words[i >> 5] |= 1u32 << (i & 31);
                }
            }
            Some(words)
        };
        // Pack a step's per-conjunct host i64 key columns into one i64 per row: a single conjunct is the
        // key itself; a 2-conjunct composite puts member 0 in the HIGH 32 bits and member 1 in the LOW
        // (each member is a <=32-bit int, so this is bijective -- distinct (k0,k1) -> distinct i64; a
        // (i32::MIN, 0) pair packs to i64::MIN and is rejected by the hash launcher, as for a plain key).
        let pack_keys = |members: &[Vec<i64>]| -> Vec<i64> {
            if members.len() == 1 {
                members[0].clone()
            } else {
                (0..members[0].len())
                    .map(|i| (members[0][i] << 32) | (members[1][i] & 0xFFFF_FFFF))
                    .collect()
            }
        };
        // Left-deep pipeline. `work_idx[j]` holds, for each surviving tuple, the ABSOLUTE row index that
        // tuple took from relation j; it starts as relation 0's survivors and grows one relation per step.
        let mut work_idx: Vec<Vec<u32>> = vec![survivors_all[0].clone()];
        for (k, conjuncts) in step_keys.iter().enumerate() {
            let new_rel = k + 1;
            // V1b NULL-key skip ON-DEVICE (charter: the NULL decision lives in the kernel, NEVER a host
            // host_rows filter). In an equi-join `NULL = x` is UNKNOWN, so a row whose join key is NULL (ANY
            // conjunct member, either side) matches nothing. The FULL carried index vectors are joined (no
            // host pre-filter); each side's per-key validity is gathered from the DEVICE payload, AND'd
            // across composite members (every member must be present), with a carried JOIN_NULL_ROW pad
            // marked invalid, and passed as a dense bitmap to the hash join -- the kernel SKIPS a 0-bit key
            // (a build key never claims a slot, so it is invisible to the duplicate check; a probe key emits
            // no match). A kernel-skipped NULL key simply ends up unmatched (an OUTER step then NULL-pads
            // it). All-valid -> None -> the kernel's sentinel fast path keeps the no-NULL join byte-identical.
            let tuple_count = work_idx[0].len();
            let acc_idx: &Vec<Vec<u32>> = &work_idx;
            let new_idx: &[u32] = survivors_all[new_rel].as_slice();
            let mut acc_valid = vec![true; tuple_count];
            for &(acc_rel, acc_col, _) in conjuncts {
                for (a, v) in acc_valid.iter_mut().zip(key_validity(acc_rel, acc_col, &work_idx[acc_rel])?) {
                    *a &= v;
                }
            }
            let mut new_valid = vec![true; new_idx.len()];
            for &(_, _, new_col) in conjuncts {
                for (a, v) in new_valid.iter_mut().zip(key_validity(new_rel, new_col, new_idx)?) {
                    *a &= v;
                }
            }
            let acc_bitmap = pack_validity(&acc_valid);
            let new_bitmap = pack_validity(&new_valid);
            let (acc_is_build, build_idxs, probe_idxs) = if step_is_text[k] {
                // TEXT key (a single conjunct, validated). Gather both sides' key bytes from the DEVICE
                // payload and run the GPU text hash join (FNV-hash + byte-verify).
                let (acc_rel, acc_col, new_col) = conjuncts[0];
                let acc_texts = key_texts(acc_rel, acc_col, &acc_idx[acc_rel])?;
                let new_texts = key_texts(new_rel, new_col, new_idx)?;
                let acc_refs: Vec<&[u8]> = acc_texts.iter().map(|s| s.as_bytes()).collect();
                let new_refs: Vec<&[u8]> = new_texts.iter().map(|s| s.as_bytes()).collect();
                text_hash_join(
                    sides[new_rel].1.mem(),
                    &acc_refs,
                    &new_refs,
                    acc_bitmap.as_deref(),
                    new_bitmap.as_deref(),
                )?
            } else if step_is_b128[k] {
                // NUMERIC/UUID key: gather each side's 16-byte canonical value and run the SAME text/byte
                // hash join over it (a 16-byte "text" -> FNV + 16-byte verify = exact 128-bit equality).
                let (acc_rel, acc_col, new_col) = conjuncts[0];
                let acc_vals = key_b128(acc_rel, acc_col, &acc_idx[acc_rel])?;
                let new_vals = key_b128(new_rel, new_col, new_idx)?;
                let acc_refs: Vec<&[u8]> = acc_vals.iter().map(|v| v.as_slice()).collect();
                let new_refs: Vec<&[u8]> = new_vals.iter().map(|v| v.as_slice()).collect();
                text_hash_join(
                    sides[new_rel].1.mem(),
                    &acc_refs,
                    &new_refs,
                    acc_bitmap.as_deref(),
                    new_bitmap.as_deref(),
                )?
            } else {
                // INT key (incl. composite). Project each conjunct's accumulated-side + new-side key on
                // the GPU, then pack into one i64. Conjuncts may project from different accumulated
                // relations; all share the tuple count, so the packed columns align row-for-row.
                let acc_members: Vec<Vec<i64>> = conjuncts
                    .iter()
                    .map(|&(acc_rel, acc_col, _)| key_i64(acc_rel, acc_col, &acc_idx[acc_rel]))
                    .collect::<Result<_, _>>()?;
                let new_members: Vec<Vec<i64>> = conjuncts
                    .iter()
                    .map(|&(_, _, new_col)| key_i64(new_rel, new_col, new_idx))
                    .collect::<Result<_, _>>()?;
                let acc_keys = pack_keys(&acc_members);
                let new_keys = pack_keys(&new_members);
                hash_join(
                    sides[new_rel].1.mem(),
                    &acc_keys,
                    &new_keys,
                    acc_bitmap.as_deref(),
                    new_bitmap.as_deref(),
                )?
            };
            // build/probe positions -> (accumulated tuple position, new-relation survivor position).
            let (acc_match, new_match): (&[u32], &[u32]) = if acc_is_build {
                (&build_idxs, &probe_idxs)
            } else {
                (&probe_idxs, &build_idxs)
            };
            // Extend the carried indices: keep each matched tuple's prior rows, append the new relation's
            // matched ABSOLUTE survivor row. The kernel returns match positions over the FULL acc_idx/new_idx
            // arrays (V1b: no host pre-filter), so each position indexes directly into them.
            let mut next: Vec<Vec<u32>> = Vec::with_capacity(new_rel + 1);
            for col in acc_idx {
                next.push(acc_match.iter().map(|&p| col[p as usize]).collect());
            }
            next.push(new_match.iter().map(|&p| new_idx[p as usize]).collect());
            // LEFT OUTER (M3 -- doc 21): every accumulated (left) row must appear. An UNMATCHED left tuple
            // -- including a NULL-key left row, which the kernel skips so it matches nothing -- is appended
            // with the new relation NULL-padded via the `JOIN_NULL_ROW` sentinel (the gather emits NULL for
            // its columns). `acc_match` positions index the FULL carried array directly; the rest are padded.
            if plan.steps[k].outer_left {
                let mut matched_orig = vec![false; tuple_count];
                for &p in acc_match {
                    matched_orig[p as usize] = true;
                }
                for (o, &is_matched) in matched_orig.iter().enumerate() {
                    if !is_matched {
                        for (rel, col) in work_idx.iter().enumerate() {
                            next[rel].push(col[o]);
                        }
                        next[new_rel].push(JOIN_NULL_ROW);
                    }
                }
            }
            // RIGHT OUTER (the mirror; FULL runs both): every NEW (right) survivor must appear. An
            // UNMATCHED new row -- including a NULL-key right row, which the kernel skips so it matches
            // nothing -- is appended with the ACCUMULATED relations NULL-padded. `new_match` positions index
            // the FULL new-survivor array directly (V1b: no host pre-filter).
            if plan.steps[k].outer_right {
                let new_count = survivors_all[new_rel].len();
                let mut matched_new = vec![false; new_count];
                for &p in new_match {
                    matched_new[p as usize] = true;
                }
                for (o, &is_matched) in matched_new.iter().enumerate() {
                    if !is_matched {
                        for col in next.iter_mut().take(new_rel) {
                            col.push(JOIN_NULL_ROW);
                        }
                        next[new_rel].push(survivors_all[new_rel][o]);
                    }
                }
            }
            work_idx = next;
        }
        // OUTER + WHERE post-filter (M3 -- doc 21): apply each predicated relation's GPU-computed survivor
        // keep-mask to the joined result. A tuple survives only if, for every predicated relation: a REAL
        // carried row passed the predicate on-device (in the keep-mask); a JOIN_NULL_ROW pad survives iff
        // the predicate holds on an all-NULL row (`pad_survives` -- TRUE for `col IS NULL`, the anti-join;
        // FALSE/UNKNOWN for a comparison, which drops the pad). (INNER joins never set keep_masks; they
        // pre-filtered the inputs, so this is a no-op for them.)
        if outer_where {
            let kept: Vec<usize> = (0..work_idx[0].len())
                .filter(|&t| {
                    keep_masks.iter().enumerate().all(|(i, mask)| match mask {
                        Some(mask) => {
                            let row = work_idx[i][t];
                            if row == JOIN_NULL_ROW {
                                pad_survives[i]
                            } else {
                                mask[row as usize]
                            }
                        }
                        None => true,
                    })
                })
                .collect();
            work_idx = work_idx
                .iter()
                .map(|col| kept.iter().map(|&t| col[t]).collect())
                .collect();
        }
        // Gather: each projected column's VALUES come from its relation's DEVICE payload at the carried
        // ABSOLUTE rows (charter: result values are read back from device memory, NOT a host_rows copy --
        // host_rows is ingest-staging only). A `JOIN_NULL_ROW` sentinel (an OUTER NULL pad) emits
        // SqlValue::Null; a real row whose value is NULL emits Null via the column's device validity bitmap.
        // Mirrors the resident SELECT materialization (S1): the same per-type project_*_rows_from_payload
        // matrix, here per join side. Column-major (one gather per projected column), transposed to rows.
        let work_n = work_idx[0].len();
        // Gather ONE projected column from relation `ri`'s device payload at the carried rows. A pad row
        // (JOIN_NULL_ROW) uses a 0 placeholder index whose gathered value is overridden to NULL; a relation
        // with 0 rows contributes only pads -> all NULL (no device read at index 0 into an empty payload).
        let gather_col = |ri: usize, ci: usize, rows: &[u32]| -> Result<Vec<SqlValue>, ExecuteError> {
            let n = rows.len();
            if n == 0 {
                return Ok(Vec::new());
            }
            let (entry, dm, row_count) = &sides[ri];
            if *row_count == 0 {
                return Ok(vec![SqlValue::Null; n]);
            }
            let table = &tables[ri];
            let is_pad: Vec<bool> = rows.iter().map(|&r| r == JOIN_NULL_ROW).collect();
            let idxs: Vec<u64> = rows
                .iter()
                .map(|&r| if r == JOIN_NULL_ROW { 0 } else { u64::from(r) })
                .collect();
            // NULL validity (None = the column holds no NULLs, so every gathered value is real).
            let validity: Option<Vec<bool>> =
                match resident_device_null_column_offset(&entry.descriptor, table, ci)? {
                    Some(off) => {
                        Some(dm.mem().project_bool_rows_from_payload(off, &idxs).map_err(map_err)?)
                    }
                    None => None,
                };
            // Per-type value gather (the type matrix; mirrors the resident SELECT projection, exhaustive
            // over SqlType so a new type is a compile error rather than a silent fallthrough).
            let values: Vec<SqlValue> = match table.columns[ci].ty {
                SqlType::Int8 => {
                    let off = resident_device_int8_column_offset(&entry.descriptor, table, ci)?;
                    dm.mem()
                        .project_i64_rows_from_payload(off, &idxs)
                        .map_err(map_err)?
                        .into_iter()
                        .map(SqlValue::Int8)
                        .collect()
                }
                SqlType::Timestamp => {
                    let off = resident_device_int8_column_offset(&entry.descriptor, table, ci)?;
                    dm.mem()
                        .project_i64_rows_from_payload(off, &idxs)
                        .map_err(map_err)?
                        .into_iter()
                        .map(SqlValue::Timestamp)
                        .collect()
                }
                SqlType::Numeric { scale, .. } => {
                    let off = resident_device_numeric_column_offset(&entry.descriptor, table, ci)?;
                    dm.mem()
                        .project_i128_rows_from_payload(off, &idxs)
                        .map_err(map_err)?
                        .into_iter()
                        .map(|v| SqlValue::Numeric(Decimal128::new(v, scale)))
                        .collect()
                }
                SqlType::Uuid => {
                    let off = resident_device_numeric_column_offset(&entry.descriptor, table, ci)?;
                    dm.mem()
                        .project_i128_rows_from_payload(off, &idxs)
                        .map_err(map_err)?
                        .into_iter()
                        .map(|v| SqlValue::Uuid(v.to_le_bytes()))
                        .collect()
                }
                SqlType::Date => {
                    let off = resident_device_int4_column_offset(&entry.descriptor, table, ci)?;
                    dm.mem()
                        .project_i32_rows_from_payload(off, &idxs)
                        .map_err(map_err)?
                        .into_iter()
                        .map(SqlValue::Date)
                        .collect()
                }
                SqlType::Int2 => {
                    let off = resident_device_int4_column_offset(&entry.descriptor, table, ci)?;
                    dm.mem()
                        .project_i32_rows_from_payload(off, &idxs)
                        .map_err(map_err)?
                        .into_iter()
                        .map(|v| SqlValue::Int2(v as i16))
                        .collect()
                }
                SqlType::Bool => {
                    let off = resident_device_bool_column_offset(&entry.descriptor, table, ci)?;
                    dm.mem()
                        .project_bool_rows_from_payload(off, &idxs)
                        .map_err(map_err)?
                        .into_iter()
                        .map(SqlValue::Bool)
                        .collect()
                }
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(&entry.descriptor, table, ci)?;
                    dm.mem()
                        .project_text_rows_from_payload(
                            layout.offsets_byte_offset,
                            layout.bytes_byte_offset,
                            layout.bytes_len,
                            &idxs,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(SqlValue::Text)
                        .collect()
                }
                SqlType::Int4 => {
                    let off = resident_device_int4_column_offset(&entry.descriptor, table, ci)?;
                    dm.mem()
                        .project_i32_rows_from_payload(off, &idxs)
                        .map_err(map_err)?
                        .into_iter()
                        .map(SqlValue::Int4)
                        .collect()
                }
            };
            // A pad row (OUTER NULL pad) or a 0 validity bit -> SqlValue::Null; else the gathered value.
            Ok((0..n)
                .map(|t| {
                    if is_pad[t] || validity.as_ref().is_some_and(|v| !v[t]) {
                        SqlValue::Null
                    } else {
                        values[t].clone()
                    }
                })
                .collect())
        };
        // Column-major device gather (one per projected column), transposed into result rows.
        let proj_values: Vec<Vec<SqlValue>> = proj
            .iter()
            .map(|&(ri, ci)| gather_col(ri, ci, &work_idx[ri]))
            .collect::<Result<_, _>>()?;
        let mut result_rows: Vec<Vec<SqlValue>> = (0..work_n)
            .map(|t| proj_values.iter().map(|col| col[t].clone()).collect())
            .collect();
        // Result schema: each projected column's RelationalColumn from its owning table, attnums 1..N.
        let mut columns = Vec::with_capacity(proj.len());
        for (i, &(ri, ci)) in proj.iter().enumerate() {
            let mut col = tables[ri].columns[ci].clone();
            col.attnum = (i + 1) as i16;
            columns.push(col);
        }
        // ORDER BY on the join result is a GPU SORT (charter: every relational sort is a GPU sort), then
        // OFFSET/LIMIT WINDOWS the device sort permutation -- a control-plane index slice, never a host
        // drain/truncate on result data (S4). gpu_sort_permutation returns the index vector (identity for
        // <=1 row / empty order); we slice it to [OFFSET, OFFSET+LIMIT) and gather only that window. Each
        // key resolves to a PROJECTED result column (same resolution as the projection, incl. USING/NATURAL
        // coalescing); a non-projected ORDER BY key on the join path is a follow-up.
        let need_window = plan.offset.is_some() || plan.limit.is_some();
        if (!plan.order_by.is_empty() && result_rows.len() > 1) || need_window {
            let mut order: Vec<(usize, bool)> = Vec::with_capacity(plan.order_by.len());
            for (c, descending) in &plan.order_by {
                let resolved = if c.qualifier.is_none() && is_coalesced(&c.column) {
                    (0, relational_column_index(&tables[0], &c.column)?)
                } else {
                    resolve(c, n_rel - 1)?
                };
                let idx = proj.iter().position(|&p| p == resolved).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "ORDER BY column `{}` must appear in the SELECT list on the join path (a \
                         non-projected ORDER BY key is a follow-up)",
                        c.column
                    )))
                })?;
                order.push((idx, *descending));
            }
            let col_types: Vec<SqlType> = columns.iter().map(|col| col.ty).collect();
            // Explicit NULLS FIRST/LAST on a join ORDER BY is honored on-device (every key type, incl. int,
            // reads its NULL validity bitmap in the sort comparator -- no host sentinel). With no ORDER BY
            // (window-only), `order` is empty -> gpu_sort_permutation returns identity, so OFFSET/LIMIT
            // windows the rows in join order, matching the prior host drain/truncate.
            let perm = gpu_sort_permutation(
                &result_rows,
                &order,
                &plan.order_by_nulls_first,
                &col_types,
                sides[0].1.mem(),
            )?;
            let start = plan.offset.unwrap_or(0).min(perm.len());
            let end = plan
                .limit
                .map_or(perm.len(), |l| start.saturating_add(l).min(perm.len()));
            let windowed: Vec<Vec<SqlValue>> =
                perm[start..end].iter().map(|&p| result_rows[p as usize].clone()).collect();
            result_rows = windowed;
        }
        Ok(RelationalSelectResult {
            columns,
            rows: result_rows,
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: RelationalAccessPath::FullTableScan,
        })
    }

    #[allow(clippy::too_many_arguments)] // group_key_expr is threaded alongside the predicate/binding
    pub(crate) fn execute_resident_expr_select_with_binding(
        &self,
        select: &Select,
        table: &RelationalTable,
        // `None` = look up the table payload (descriptor + device buffer + row count) by name in the
        // SINGLE resident store (the whole-table buffer), as before. `Some(src)` INJECTS them so the
        // same executor serves one partition slice (S10c). The identity/validity guards run for both.
        src: Option<&ResidentExecSource>,
        bound: BoundRelationalSelect,
        copin_s: Index,
        // `None` = no WHERE clause: a full-table scan (every row survives).
        predicate: Option<&ResidentExpr>,
        // Parallel to `select.order_by`: `Some(expr)` = a SORT EXPRESSION key (`ORDER BY a+b`),
        // evaluated on-device into an i64 key column; `None` = a plain column key. Empty = no ORDER BY.
        order_by_exprs: &[Option<ResidentExpr>],
        // Parallel to `select.order_by`: the explicit NULLS FIRST/LAST override per key (`Some(true)` =
        // NULLS FIRST, `Some(false)` = NULLS LAST, `None` = PG default = NULLS LAST under ASC / FIRST under
        // DESC). Honored ON-DEVICE in the GPU sort comparator via the per-key nulls_first bitmask.
        order_by_nulls_first: &[Option<bool>],
        // `Some(expr)` = GROUP BY an EXPRESSION (`GROUP BY a+b`): materialized on-device into a derived
        // int key column the grouped kernel groups by (via key_base_override). `None` = a plain column
        // GROUP BY (the matrix path) or no GROUP BY.
        group_key_expr: Option<&ResidentExpr>,
        // The GROUP BY key columns. len()==2 = a COMPOSITE key (`GROUP BY a, b`): the two fixed-width
        // int columns are packed on-device into one i64 key (high|low) grouped via key_base_override,
        // and the result key unpacks back to the two columns. len()<=1 = the single-key path (unchanged).
        group_key_columns: &[String],
    ) -> Result<RelationalSelectResult, ExecuteError> {
        // Scalar aggregates (operator axis): COUNT(*) -> the surviving-row count; SUM(col) -> a GPU
        // reduction over the filtered column. They have no projected columns to materialize, so they
        // skip the projected-column checks and are computed from the filtered indices below.
        let is_aggregate = matches!(
            select.projection,
            SelectProjection::CountAll
                | SelectProjection::Sum { .. }
                | SelectProjection::Min { .. }
                | SelectProjection::Max { .. }
                | SelectProjection::Avg { .. }
                | SelectProjection::CountDistinct { .. }
        );
        // Grouped aggregates (GROUP BY) emit one row per group from a GPU hash aggregation; they are
        // handled separately below (not via the scalar-aggregate or the plain-projection paths).
        let is_grouped = matches!(
            select.projection,
            SelectProjection::GroupedCount { .. }
                | SelectProjection::GroupedSum { .. }
                | SelectProjection::GroupedAvg { .. }
                | SelectProjection::GroupedMin { .. }
                | SelectProjection::GroupedMax { .. }
                | SelectProjection::GroupedAggregates { .. }
        );
        if !is_aggregate && !is_grouped {
            if bound.selected_indexes.is_empty() {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident Expr select requires at least one projected column".to_string(),
                )));
            }
            for &col in &bound.selected_indexes {
                let ty = table.columns[col].ty;
                if ty != SqlType::Int4
                    && ty != SqlType::Int8
                    && !matches!(ty, SqlType::Numeric { .. })
                    && ty != SqlType::Date
                    && ty != SqlType::Timestamp
                    && ty != SqlType::Uuid
                    && ty != SqlType::Int2
                    && ty != SqlType::Bool
                    && ty != SqlType::Text
                {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident Expr select currently materializes int4 / int8 / numeric / date / \
                         timestamp / uuid / int2 / bool / text projection columns only"
                            .to_string(),
                    )));
                }
            }
        }

        // We discard `_query` and keep only `access_path` (metadata). Computing it for an ORDER BY /
        // LIMIT select would run a full CPU ordered table sort (relational_ordered_table_keys) whose
        // result we throw away -- a charter violation (the GPU does the sort here) + 2x work. The CPU
        // sort keys off `bound.order` (set at bind from order_by.first()), which the planner below reads
        // -- NOT `ap_select.order_by` -- so clearing ap_select alone is INEFFECTIVE. Clear bound.order:
        // that alone stops the CPU sort. bound.order is read nowhere else on this path (the GPU/grouped
        // sort uses select.order_by + order_by_exprs), so this is safe. The ap_select clear keeps the
        // synthesized path unordered/unlimited; the GPU sort + host OFFSET/LIMIT own ordering+windowing.
        let mut bound = bound;
        bound.order = None;
        let mut ap_select = select.clone();
        ap_select.order_by.clear();
        ap_select.limit = None;
        ap_select.offset = None;
        // The access-path planner resolves the projection's group COLUMN; an expression GROUP BY has
        // none (the placeholder "(expr)" is not a column), so synthesize a scan path for it. The query
        // is discarded metadata anyway; the grouped GPU aggregation/sort owns execution.
        let access_path = if group_key_expr.is_some() {
            RelationalAccessPath::FullTableScan
        } else {
            let (_query, access_path) =
                self.relational_select_mvcc_query_pinned(&ap_select, table, &bound, copin_s)?;
            access_path
        };
        // The table payload the kernels read: the published device buffer + the catalog/GPU descriptor
        // (its `row_count` + `resident_device_*` section vectors define every column byte-offset) + the
        // row count. `None` looks these up by name in the SINGLE resident store -- byte-identical to the
        // pre-S10c path: one atomic load of the descriptor, then the device-memory cell. `Some(src)`
        // INJECTS them (S10c: one partition slice). The identity + validity guards run for BOTH, so a
        // descriptor that drifted from the catalog or got invalidated is rejected either way. No
        // `host_rows` read on this path: a text GROUP BY key result is materialized ON-DEVICE.
        let (snapshot, device_memory, row_count) = match src {
            Some(src) => {
                let snapshot = src.descriptor.clone();
                if snapshot.schema != table.schema || snapshot.table != table.name {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot no longer matches catalog table identity".to_string(),
                    )));
                }
                if !snapshot.is_valid() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{}\" resident snapshot is invalid",
                        table.name
                    ))));
                }
                (snapshot, src.device_memory.clone(), src.row_count)
            }
            None => {
                let residency_entry = self
                    .relational_residency_entry(&table.name)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "relation \"{}\" has no resident snapshot",
                            table.name
                        )))
                    })?;
                let snapshot = residency_entry.descriptor.clone();
                if snapshot.schema != table.schema || snapshot.table != table.name {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot no longer matches catalog table identity".to_string(),
                    )));
                }
                if !snapshot.is_valid() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{}\" resident snapshot is invalid",
                        table.name
                    ))));
                }
                let device_memory = self
                    .read_state
                    .residency
                    .device_memory
                    .get(&table.name)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "relation \"{}\" has no retained resident device memory",
                            table.name
                        )))
                    })?;
                let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot row count exceeds retained device-memory proof range"
                            .to_string(),
                    ))
                })?;
                (snapshot, device_memory, row_count)
            }
        };

        // Evaluate the predicate on the GPU -> surviving row indices (ascending). With no WHERE clause
        // every row survives, so the indices are the full 0..row_count scan (the aggregate + projection
        // paths below are index-driven and need no other change).
        let indices = match predicate {
            Some(predicate) => self.lower_resident_predicate(
                predicate,
                table,
                &snapshot,
                &device_memory,
                row_count,
            )?,
            None => {
                let n = u32::try_from(row_count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "full-table scan row count exceeds the u32 row-index range".to_string(),
                    ))
                })?;
                (0..n).collect()
            }
        };
        let indices_u64: Vec<u64> = indices.iter().map(|&i| u64::from(i)).collect();

        // COUNT(DISTINCT v) over a (group key, value) tuple: GPU-sort (g, value) ASC + mark the first
        // row of each distinct tuple -> per-group SUM of the new-distinct flags = the per-group distinct
        // count. Shared by the grouped branch (g = the real group key) AND the scalar form below (g = a
        // constant 0 -> ONE group whose count = the total distinct). A FIXED-WIDTH value packs into an
        // i64 multikey matrix (int = k2 (g, v); numeric/uuid = k3 (g, v_hi, v_lo)); a varlen TEXT value
        // routes through the hetero sort + the text-aware mark (reads the value text on-device). The
        // sort/mark/SUM run ENTIRELY on the GPU; the host marshals only the control-plane g / index
        // arrays. `g_vals`/`idx_u64` are parallel, length n (the surviving positions). Empty -> caller
        // guards n == 0.
        let count_distinct_groups = |value_idx: usize,
                                     g_vals: &[i64],
                                     idx_u64: &[u64]|
         -> Result<Vec<gpu_db_execution::GroupByI32Row>, ExecuteError> {
            let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
            };
            let n = idx_u64.len();
            let value_ty = table.columns[value_idx].ty;
            let (g_sorted, new_distinct) = match value_ty {
                SqlType::Int2
                | SqlType::Int4
                | SqlType::Int8
                | SqlType::Date
                | SqlType::Timestamp
                | SqlType::Numeric { .. }
                | SqlType::Uuid => {
                    let (matrix, k) =
                        if matches!(value_ty, SqlType::Numeric { .. } | SqlType::Uuid) {
                            let off = resident_device_numeric_column_offset(
                                &snapshot, table, value_idx,
                            )?;
                            let v128 = device_memory
                                .project_i128_rows_from_payload(off, idx_u64)
                                .map_err(map_err)?;
                            let mut m = Vec::with_capacity(n * 3);
                            for i in 0..n {
                                m.push(g_vals[i]);
                                m.push((v128[i] >> 64) as i64);
                                m.push((v128[i] as u64) as i64);
                            }
                            (m, 3usize)
                        } else {
                            let v_vals: Vec<i64> = match value_ty {
                                SqlType::Int8 | SqlType::Timestamp => device_memory
                                    .project_i64_rows_from_payload(
                                        resident_device_int8_column_offset(
                                            &snapshot, table, value_idx,
                                        )?,
                                        idx_u64,
                                    )
                                    .map_err(map_err)?,
                                _ => device_memory
                                    .project_i32_rows_from_payload(
                                        resident_device_int4_column_offset(
                                            &snapshot, table, value_idx,
                                        )?,
                                        idx_u64,
                                    )
                                    .map_err(map_err)?
                                    .into_iter()
                                    .map(i64::from)
                                    .collect(),
                            };
                            let mut m = Vec::with_capacity(n * 2);
                            for i in 0..n {
                                m.push(g_vals[i]);
                                m.push(v_vals[i]);
                            }
                            (m, 2usize)
                        };
                    let perm = device_memory
                        .bitonic_sort_multikey(&matrix, n, k, 0)
                        .map_err(map_err)?;
                    device_memory
                        .mark_new_distinct_device(&matrix, &perm, n as u64, k)
                        .map_err(map_err)?
                }
                SqlType::Text => {
                    let layout =
                        resident_device_text_column_layout(&snapshot, table, value_idx)?;
                    let key_plan: Vec<u32> = vec![0, 0x4000_0000_u32];
                    let perm = device_memory
                        .bitonic_sort_hetero(
                            idx_u64,
                            g_vals,
                            1,
                            &[(layout.offsets_byte_offset, layout.bytes_byte_offset)],
                            &[],
                            &key_plan,
                            0,
                            // COUNT(DISTINCT) (g, text_v) reps: both keys are non-NULL by construction, so
                            // there is no NULL bitmap and the nulls_first mask is irrelevant.
                            &[],
                            0,
                        )
                        .map_err(map_err)?;
                    device_memory
                        .mark_new_distinct_text_device(
                            &perm,
                            idx_u64,
                            g_vals,
                            layout.offsets_byte_offset,
                            layout.bytes_byte_offset,
                            n as u64,
                        )
                        .map_err(map_err)?
                }
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "COUNT(DISTINCT) value column must be int/numeric/uuid/text on the \
                         GPU path"
                            .to_string(),
                    )));
                }
            };
            // SUM(new_distinct) grouped by g_sorted = the per-group distinct count. indices = 0..n (the
            // sorted positions); the kernel reads g_sorted / new_distinct (both i64) via the
            // key/value_base_override. The leases live across this synced call.
            let scan_indices: Vec<u32> = (0..n as u32).collect();
            let cd_groups = device_memory
                .group_by_i32_count_sum_minmax_from_payload(
                    0,
                    0,
                    &scan_indices,
                    true,  // value_is_int8 (new_distinct is i64)
                    true,  // key_is_int8 (g_sorted is i64)
                    false, // value not numeric
                    false, // value not uuid
                    false, // key not i128
                    false, // key not text
                    0,
                    0,
                    false, // value not text
                    0,
                    0,
                    g_sorted.device_ptr(),
                    new_distinct.device_ptr(),
                    0, // comp_w (not a wide-key composite)
                    0, // n_text (no text members)
                    0, // text_desc_ptr
                    None, // value_null_off (COUNT(DISTINCT) over the marked tuple matrix)
                    None, // key_null_off (the marked-tuple matrix key is non-null by construction)
                )
                .map_err(map_err)?;
            drop((g_sorted, new_distinct));
            Ok(cd_groups)
        };

        // Grouped aggregate (GROUP BY <int4 key>): GPU hash aggregation over the filtered rows -> one
        // row per distinct key. COUNT/SUM/AVG share the count+sum kernel; grouped MIN/MAX is a
        // follow-on. PG does not order GROUP BY without ORDER BY; sort by key for determinism.
        if is_grouped {
            let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
            };
            let SelectProjection::GroupedAggregates {
                group_column: group_name,
                aggregates,
            } = &select.projection
            else {
                unreachable!("is_grouped gates on GroupedAggregates on the Expr path");
            };
            // Resolve each aggregate's value column to an index (None for COUNT(*)), and the distinct
            // value columns in first-seen order -- one grouping pass per distinct value column, since
            // the kernel yields count+sum+min+max for one value column.
            let agg_value_indices: Vec<Option<usize>> = aggregates
                .iter()
                .map(|a| {
                    a.value_column
                        .as_deref()
                        .map(|c| relational_column_index(table, c))
                        .transpose()
                })
                .collect::<Result<Vec<_>, _>>()?;
            // The DIRECT (hash) passes -- one per distinct value column of a COUNT/SUM/AVG/MIN/MAX
            // aggregate. A COUNT(DISTINCT v) column is NOT added here: it runs a separate sort-based
            // pass (added after the direct passes), unless another DIRECT aggregate also needs it.
            let mut value_indices: Vec<usize> = Vec::new();
            for (aggregate, value_idx) in aggregates.iter().zip(&agg_value_indices) {
                if aggregate.kind == GroupedAggKind::CountDistinct {
                    continue;
                }
                if let Some(value_idx) = value_idx {
                    if !value_indices.contains(value_idx) {
                        value_indices.push(*value_idx);
                    }
                }
            }
            // M3 (doc 21) GROUP BY 3VL: a NULL group KEY forms its OWN group (NULLs group together,
            // distinct from real keys). The kernel routes a NULL key (of ANY single-column type) to a
            // dedicated reserved slot. Supported for a SINGLE COLUMN key of int2/4/8/date/timestamp / text /
            // numeric / uuid. A nullable COMPOSITE / EXPRESSION key is still a clean-error follow-up: its
            // per-member NULL semantics ((NULL,5) ≠ (NULL,6) ≠ (1,5)) need NULL encoded into the key, not
            // one reserved slot. A NULL-free key has no bitmap, so it runs unchanged either way.
            let single_key_col = if group_key_expr.is_none() && group_key_columns.len() < 2 {
                relational_column_index(table, group_name).ok()
            } else {
                None
            };
            let key_is_single_nullable_column = single_key_col.is_some_and(|idx| {
                matches!(
                    table.columns.get(idx).map(|column| column.ty),
                    Some(
                        SqlType::Int2
                            | SqlType::Int4
                            | SqlType::Int8
                            | SqlType::Date
                            | SqlType::Timestamp
                            | SqlType::Text
                            | SqlType::Numeric { .. }
                            | SqlType::Uuid
                    )
                )
            });
            // M3 (doc 21): a COMPOSITE GROUP BY key (`GROUP BY a, b`) with a nullable member groups via the
            // general wide-key path with PER-MEMBER NULL validity (gpu_db_build_wide_key writes a trailing
            // validity word; the claim hashes + memcmps it, so (NULL,5) != (0,5) != (NULL,6) are distinct).
            // A nullable TEXT member is a clean-error follow-up (text NULL in the wide key needs the text
            // descriptor's verify to be NULL-aware). Detected here so the routing + clean-error below agree.
            let composite_key_has_null = if group_key_columns.len() >= 2 {
                let mut any = false;
                for name in group_key_columns {
                    let idx = relational_column_index(table, name)?;
                    if resident_device_null_column_offset(&snapshot, table, idx)?.is_some() {
                        if matches!(table.columns[idx].ty, SqlType::Text) {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "GROUP BY a composite key with a nullable TEXT member is a follow-up on \
                                 the GPU (M3 3VL; a nullable fixed-width member is supported)"
                                    .to_string(),
                            )));
                        }
                        any = true;
                    }
                }
                any
            } else {
                false
            };
            // M3 (doc 21): an EXPRESSION group key (`a + b`) is NULL exactly when an operand is NULL. If it
            // references EXACTLY ONE nullable operand column, the expression's NULL group IS that column's
            // NULL rows, so reuse the single-column NULL-key reserved slot (pass_key_null_off = that
            // column's validity bitmap). The kernel routes by validity BEFORE reading the derived key, so a
            // NULL row's garbage derived value is never used, and the NULL group renders SqlValue::Null via
            // the existing key_is_null path -- ZERO kernel change, int4 or int8 expression alike. Two+
            // nullable operands need a DERIVED validity (the AND of the operand bitmaps): a clean-error
            // follow-up.
            let expr_key_single_null_off: Option<u64> = if let Some(expr) = group_key_expr {
                let mut cols = Vec::new();
                collect_expr_columns(expr, &mut cols);
                cols.sort_unstable();
                cols.dedup();
                let mut nullable_offs = Vec::new();
                for c in cols {
                    if let Some(off) = resident_device_null_column_offset(&snapshot, table, c)? {
                        nullable_offs.push(off);
                    }
                }
                // exactly one nullable operand -> its bitmap is the expression's validity; 0 = no NULLs
                // (unchanged); >1 -> None here, caught by the clean-error loop below.
                (nullable_offs.len() == 1).then(|| nullable_offs[0])
            } else {
                None
            };
            if !key_is_single_nullable_column
                && expr_key_single_null_off.is_none()
                && !composite_key_has_null
            {
                let mut group_by_key_check: Vec<usize> = Vec::new();
                if let Some(expr) = group_key_expr {
                    collect_expr_columns(expr, &mut group_by_key_check);
                } else if !group_key_columns.is_empty() {
                    for name in group_key_columns {
                        group_by_key_check.push(relational_column_index(table, name)?);
                    }
                } else if let Ok(idx) = relational_column_index(table, group_name) {
                    group_by_key_check.push(idx);
                }
                for col in group_by_key_check {
                    if resident_device_null_column_offset(&snapshot, table, col)?.is_some() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "GROUP BY over an EXPRESSION key with >1 nullable operand (or a bool key) is \
                             not yet supported on the GPU (M3 3VL follow-up: it needs a DERIVED NULL \
                             encoding; a single int/date/timestamp/text/numeric/uuid column key, a \
                             composite of fixed-width members, or an expression over exactly one nullable \
                             operand, is supported)"
                                .to_string(),
                        )));
                    }
                }
            }
            // M3 (doc 21): a nullable NUMERIC value runs the numeric TWO-PASS min/max kernel — pass 1
            // records each NON-NULL row's claimed slot into a POOLED `row_slots` scratch and skips NULL
            // rows (the value-skip gate at do_agg), pass 2 (gpu_db_group_by_numeric_minmax_lo) folds the
            // i128 low limb. BOTH passes now read the value validity bitmap and skip NULL rows, so a NULL
            // row's stale pooled slot is never folded (pass 2 gained `value_null_off`, matching pass 1).
            // All-NULL group → count 0 → SqlValue::Null at finalization (shared with int/int8). So a
            // nullable numeric aggregate value is supported, like int2/4/8/date/timestamp/uuid/text.
            // GROUP BY <expression> (`a+b`): the key is a DERIVED int buffer materialized on-device
            // below (grouped via key_base_override), NOT a column -- so group_idx is only a placeholder
            // for the COUNT(*) pass (which reads no value), and key_ty is the expression result type.
            // COMPOSITE GROUP BY (`GROUP BY a, b`): the on-device pack foundation
            // (gpu_db_pack_two_int4_cols / pack_two_int4_cols_device) is landed, but the N-column result
            // wiring -- build_grouped_projection accepting 2 leading group targets, bound.selected_columns
            // carrying both columns, and the result-row UNPACK of the packed i64 key back into the two
            // columns -- is a coupled change to the most-audited grouped path that belongs in its own
            // auditable slice. Until then, reject cleanly (NOT a silent single-key grouping by the first
            // column, which `select.group_by` carries).
            // COMPOSITE GROUP BY (`GROUP BY a, b[, ...]`). TWO fast paths for a 2-member key, packed
            // on-device into ONE derived key (key_base_override) + UNPACKED in the result:
            //  - both fixed-width int (int2/int4/int8/date/timestamp): one i64 `(c0<<32)|c1`
            //    (gpu_db_pack_two_int4_cols) or, when a member is int8/timestamp (> 64 bits combined),
            //    one i128 `c0:c1` (gpu_db_pack_two_cols_i128 -> the b128 key path);
            //  - one text + one fixed-int: the text-key b128 claim with the fixed member folded in.
            // The GENERAL path (composite_is_widekey) handles every OTHER composite -- >2 columns, a
            // numeric/uuid member, a tuple > 128 bits, AND any text member beyond the 2-member (fixed,
            // text) case (two-text, text in a >2 key) -- via a fixed-width WIDE KEY buffer
            // (gpu_db_build_wide_key, for the fixed members) PLUS a text-member descriptor; the claim
            // hashes + verifies both. The result reads each member from the representative row.
            let is_text = |t: SqlType| matches!(t, SqlType::Text);
            let is_fixed_int = |t: SqlType| {
                matches!(
                    t,
                    SqlType::Int4
                        | SqlType::Int2
                        | SqlType::Date
                        | SqlType::Int8
                        | SqlType::Timestamp
                )
            };
            let is_widekey_member = |t: SqlType| {
                is_fixed_int(t)
                    || matches!(t, SqlType::Numeric { .. } | SqlType::Uuid | SqlType::Bool)
            };
            let composite_members: Option<Vec<(usize, SqlType)>> = if group_key_columns.len() >= 2 {
                let mut m = Vec::with_capacity(group_key_columns.len());
                for name in group_key_columns {
                    let idx = relational_column_index(table, name)?;
                    m.push((idx, table.columns[idx].ty));
                }
                Some(m)
            } else {
                None
            };
            // 2-member fast path: both fixed-int, OR exactly one text + one fixed-int. A NULLABLE composite
            // takes the general WIDE-KEY path instead (composite_cols = None) -- the i64/i128 pack has no
            // room for a per-member validity bit, but the wide key carries a validity word.
            let composite_cols: Option<(usize, SqlType, usize, SqlType)> = match &composite_members {
                Some(m) if m.len() == 2 && !composite_key_has_null => {
                    let (c0, t0) = m[0];
                    let (c1, t1) = m[1];
                    let two_fixed = is_fixed_int(t0) && is_fixed_int(t1);
                    let fixed_text =
                        (is_text(t0) && is_fixed_int(t1)) || (is_fixed_int(t0) && is_text(t1));
                    if two_fixed || fixed_text {
                        Some((c0, t0, c1, t1))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            // General WIDE KEY path: an N>=2 composite NOT taken by the 2-member fast path. Every member
            // is fixed-width (int/numeric/uuid) OR text -- the fixed members concatenate into the
            // comp_w wide-key buffer; the text members ride a descriptor the claim folds into the hash +
            // byte-verifies (so two-text and text-in-a->2 keys are covered). bool members are a follow-up.
            let widekey_cols: Option<Vec<(usize, SqlType)>> = match &composite_members {
                Some(m) if composite_cols.is_none() => {
                    if m.iter().all(|&(_, t)| is_widekey_member(t) || is_text(t)) {
                        Some(m.clone())
                    } else {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "composite GROUP BY supports fixed-width members \
                             (int2/int4/int8/date/timestamp/numeric/uuid) and text members, any \
                             count, on the Expr path (a bool member is a follow-up)"
                                .to_string(),
                        )));
                    }
                }
                _ => None,
            };
            let composite_is_widekey = widekey_cols.is_some();
            let is_composite_key = composite_cols.is_some();
            // A composite with exactly one TEXT member groups via the text-key b128 claim with the
            // OTHER (fixed-width) member folded into the hash/verify (key_base_override). Otherwise both
            // members are fixed: i128-packed iff a member is int8/timestamp (combined width > 64 bits),
            // else i64-packed. Drives key_is_text / key_is_i128 / key_is_int8 / the pack / the result.
            let composite_is_text = composite_cols
                .is_some_and(|(_, t0, _, t1)| matches!(t0, SqlType::Text) != matches!(t1, SqlType::Text));
            let composite_is_i128 = !composite_is_text
                && composite_cols.is_some_and(|(_, t0, _, t1)| {
                    matches!(t0, SqlType::Int8 | SqlType::Timestamp)
                        || matches!(t1, SqlType::Int8 | SqlType::Timestamp)
                });
            // Composite-text + wide-key use a (rep_idx, hash) claim -> the per-pass group order is
            // rep-row (race) based, so the multi-pass alignment by a per-group key is a follow-up;
            // restrict both to a SINGLE aggregate here.
            if (composite_is_text || composite_is_widekey) && aggregates.len() > 1 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "a (fixed, text) or general all-fixed wide-key composite GROUP BY supports a \
                     single aggregate on the Expr path (multiple aggregates are a follow-up)"
                        .to_string(),
                )));
            }
            let is_expr_key = group_key_expr.is_some();
            let key_expr_is_int8 = match group_key_expr {
                Some(expr) => expr_mentions_int8(expr, table),
                None => false,
            };
            // The arith VM is mono-typed (one element width per program): a MIXED int4/int8 expression
            // key would load an int4 column with the i64 kernel (wrong stride -> reads off the section ->
            // garbage). Reject it (honest error, not a wrong answer) -- pure int4 or pure int8 work;
            // widening int4->i64 in the VM is a follow-up. Covers the COUNT(DISTINCT) reduction too (it
            // reuses this expr key buffer).
            if let Some(expr) = group_key_expr {
                if key_expr_is_int8 && expr_mentions_int4_column(expr, table) {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "GROUP BY a mixed int4/int8 arithmetic expression is not supported (the \
                         on-device arith program is mono-typed); cast the operands to one width"
                            .to_string(),
                    )));
                }
            }
            let group_idx = if is_composite_key {
                // The packed key is grouped via key_base_override; group_idx is only the COUNT(*) pass's
                // value placeholder (reads no value), so col0's index is a valid placeholder.
                composite_cols.unwrap().0
            } else if composite_is_widekey {
                // Wide-key: grouped via the wide-key buffer; group_idx is just the COUNT(*) placeholder.
                widekey_cols.as_ref().unwrap()[0].0
            } else if is_expr_key {
                0
            } else {
                relational_column_index(table, group_name)?
            };
            let key_ty = if is_composite_key || composite_is_widekey {
                // composite: the derived key (packed i64/i128, or the wide-key b128 slot) -- the result
                // reconstructs the member columns separately, so Int8 is a neutral placeholder that does
                // NOT trigger the numeric/uuid/text single-key paths.
                SqlType::Int8
            } else if is_expr_key {
                if key_expr_is_int8 {
                    SqlType::Int8
                } else {
                    SqlType::Int4
                }
            } else {
                table.columns[group_idx].ty
            };
            // GROUP BY key: int2/int4/date ride the int4 (4-byte) section; int8/timestamp ride the int8
            // (8-byte) section, which forces the single-level kernel (only it reads 64-bit keys + routes
            // the i64::MIN key, which collides with EMPTY, to its dedicated slot). An expression key is
            // int4/int8 by its result width.
            let key_is_int8 = if composite_is_widekey {
                // The wide-key path reads its key via comp_w (NOT the int8/i128/text single-key paths).
                false
            } else if is_composite_key {
                // A two-fixed composite packs into one derived key: the i64 pack reads via the int8 key
                // path; the i128 pack via the i128 path; a (fixed, text) composite via the text path
                // (key_is_int8 = false for both i128 and text).
                !composite_is_i128 && !composite_is_text
            } else if is_expr_key {
                key_expr_is_int8
            } else {
                match key_ty {
                    SqlType::Int4 | SqlType::Int2 | SqlType::Date => false,
                    SqlType::Int8 | SqlType::Timestamp => true,
                    SqlType::Numeric { .. } | SqlType::Uuid => false,
                    SqlType::Text => false,
                    // A bool key is materialized bool->int4 (0/1) into a derived buffer + grouped via
                    // key_base_override on the int4 path -- avoids a bool GROUP BY kernel + its hazard.
                    SqlType::Bool => false,
                    // EXHAUSTIVE (the prior bool lesson): a NEW SqlType is a COMPILE error here, forcing
                    // its GROUP-BY-key handling to be considered rather than silently mis-grouped.
                }
            };
            // numeric / uuid GROUP BY keys are 128-bit -- claimed via atom.cas.b128 into slot_keys_i128
            // (the single-level kernel). A WIDER COMPOSITE key (int8/timestamp member) is also packed into
            // an i128 and uses the same b128 claim. An expression key is never i128/text. key_scale
            // carries the numeric column scale onto the result key.
            let key_is_i128 = composite_is_i128
                || (!is_expr_key && matches!(key_ty, SqlType::Numeric { .. } | SqlType::Uuid));
            // A TEXT key is varlen: the kernel hashes the bytes, claims a b128 (rep_row_idx, hash) in
            // slot_keys_i128 with a full-text verify-on-lost-CAS, and the result key is read host-side
            // from the representative row. key_offsets_off/key_bytes_off locate the Arrow varlen column.
            // A (fixed, text) composite ALSO uses the text-key b128 claim, with the fixed member folded
            // into the hash/verify via key_base_override (set below).
            let key_is_text = composite_is_text || (!is_expr_key && matches!(key_ty, SqlType::Text));
            // A bool key is materialized bool->int4 (0/1) into a derived buffer + grouped via
            // key_base_override (like an expression key); int4-width, never i128/text.
            let key_is_bool = !is_expr_key && matches!(key_ty, SqlType::Bool);
            let (key_offsets_off, key_bytes_off) = if composite_is_text {
                // The text member's varlen column (the other member rides key_base_override).
                let (c0, t0, c1, _) = composite_cols.unwrap();
                let text_col = if matches!(t0, SqlType::Text) { c0 } else { c1 };
                let layout = resident_device_text_column_layout(&snapshot, table, text_col)?;
                (layout.offsets_byte_offset, layout.bytes_byte_offset)
            } else if key_is_text {
                let layout = resident_device_text_column_layout(&snapshot, table, group_idx)?;
                (layout.offsets_byte_offset, layout.bytes_byte_offset)
            } else {
                (0, 0)
            };
            let key_scale: u8 = match key_ty {
                SqlType::Numeric { scale, .. } => scale,
                _ => 0,
            };
            // The key BYTE offset, unused when key_base_override is set (the kernel then reads the
            // override base + idx*stride, not resident_base + key_offset).
            let key_offset = if is_expr_key
                || is_composite_key
                || composite_is_widekey
                || key_is_text
                || key_is_bool
            {
                0
            } else if key_is_i128 {
                resident_device_numeric_column_offset(&snapshot, table, group_idx)?
            } else if key_is_int8 {
                resident_device_int8_column_offset(&snapshot, table, group_idx)?
            } else {
                resident_device_int4_column_offset(&snapshot, table, group_idx)?
            };
            // The wide-key width (bytes/row) of the FIXED members only: each int member 8 bytes (i64),
            // each numeric/uuid 16; text members contribute 0 here (they ride the text descriptor, not
            // the fixed buffer). 0 when not a wide-key composite OR a pure all-text composite. Drives
            // comp_w on the GROUP BY launch + the build descriptors.
            let widekey_w: u64 = widekey_cols.as_ref().map_or(0, |m| {
                let fixed: u64 = m
                    .iter()
                    .map(|&(_, t)| match t {
                        SqlType::Numeric { .. } | SqlType::Uuid => 16,
                        SqlType::Text => 0,
                        _ => 8,
                    })
                    .sum();
                // M3 (doc 21): a nullable composite reserves a trailing 8-byte validity word (bit per fixed
                // member). gpu_db_build_wide_key writes it at widekey_w-8; the claim memcmps all widekey_w.
                fixed + if composite_key_has_null { 8 } else { 0 }
            });
            // GROUP BY <expression>: materialize the expr ONCE over all rows into a resident device key
            // buffer (checked overflow -> PG error) + hold it alive across EVERY pass; key_base_override
            // points the single-level kernel at it (it reads override + idx*stride, idx = the row id).
            // A column key passes 0 -> the byte-identical column path.
            let _derived_key_buf;
            let key_base_override: u64 = if let Some(expr) = group_key_expr {
                if row_count == 0 {
                    // Empty input: the on-device arith materialize rejects n=0, and there are no rows to
                    // group anyway. Skip it and group 0 rows -> 0 groups (PG returns no rows), matching
                    // the plain-column path's empty-table behavior. The override is never read (no rows).
                    _derived_key_buf = None;
                    0
                } else {
                    let mut program = Vec::new();
                    compile_arith_program(expr, table, &snapshot, &mut program)?;
                    let elem = if key_expr_is_int8 {
                        ResidentElemType::I64
                    } else {
                        ResidentElemType::I32
                    };
                    let buf = device_memory
                        .arith_value_column_device(&program, row_count, elem)
                        .map_err(|e| ExecuteError::Engine(EngineError::ApplyFailed(e.to_string())))?;
                    let ptr = buf.device_ptr();
                    _derived_key_buf = Some(buf);
                    ptr
                }
            } else if key_is_bool {
                if row_count == 0 {
                    _derived_key_buf = None;
                    0
                } else {
                    // GROUP BY a bool column: materialize bool->int4 (0/1) into a derived buffer + group
                    // via key_base_override (the audited int4 path). No bool GROUP BY kernel -> no hazard.
                    let offset = resident_device_bool_column_offset(&snapshot, table, group_idx)?;
                    let buf = device_memory
                        .bool_to_int4_column_device(offset, row_count)
                        .map_err(|e| ExecuteError::Engine(EngineError::ApplyFailed(e.to_string())))?;
                    let ptr = buf.device_ptr();
                    _derived_key_buf = Some(buf);
                    ptr
                }
            } else if is_composite_key {
                if row_count == 0 {
                    _derived_key_buf = None;
                    0
                } else {
                    // GROUP BY a, b: pack the two members into one derived key grouped via
                    // key_base_override; the result UNPACKS it. Both-int4 -> one i64 ((col0<<32)|col1);
                    // a wider member (int8/timestamp) -> one i128 (col0 high 64, col1 low 64) read by
                    // the b128 claim. Each member's width selects its section offset + the pack arg.
                    let (c0, t0, c1, t1) = composite_cols.unwrap();
                    let col_off = |idx: usize, ty: SqlType| -> Result<u64, ExecuteError> {
                        if matches!(ty, SqlType::Int8 | SqlType::Timestamp) {
                            resident_device_int8_column_offset(&snapshot, table, idx)
                        } else {
                            resident_device_int4_column_offset(&snapshot, table, idx)
                        }
                    };
                    let width = |ty: SqlType| -> u64 {
                        if matches!(ty, SqlType::Int8 | SqlType::Timestamp) {
                            8
                        } else {
                            4
                        }
                    };
                    let buf = if composite_is_text {
                        // (fixed, text): widen the FIXED member to an i64 derived buffer; the text-key
                        // claim reads the text column directly + folds this buffer in (hash + verify).
                        let (fc, ft) = if matches!(t0, SqlType::Text) {
                            (c1, t1)
                        } else {
                            (c0, t0)
                        };
                        device_memory
                            .widen_col_to_i64_device(col_off(fc, ft)?, width(ft), row_count)
                            .map_err(|e| {
                                ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
                            })?
                    } else if composite_is_i128 {
                        let off0 = col_off(c0, t0)?;
                        let off1 = col_off(c1, t1)?;
                        device_memory
                            .pack_two_cols_i128_device(off0, width(t0), off1, width(t1), row_count)
                            .map_err(|e| {
                                ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
                            })?
                    } else {
                        let off0 = col_off(c0, t0)?;
                        let off1 = col_off(c1, t1)?;
                        device_memory
                            .pack_two_int4_cols_device(off0, off1, row_count)
                            .map_err(|e| {
                                ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
                            })?
                    };
                    let ptr = buf.device_ptr();
                    _derived_key_buf = Some(buf);
                    ptr
                }
            } else if composite_is_widekey {
                if row_count == 0 || widekey_w == 0 {
                    // Empty input, OR a pure all-text composite (no fixed members) -> no fixed wide-key
                    // buffer; the text members are grouped via the text descriptor (key_base_override 0).
                    _derived_key_buf = None;
                    0
                } else {
                    // General composite: build the widekey_w-byte/row FIXED wide key (each fixed member's
                    // canonical bytes concatenated -- int 8B, numeric/uuid 16B; TEXT members are skipped
                    // here, handled by the text descriptor) -> grouped via the (rep_idx, hash) b128 claim
                    // (comp_w = widekey_w). The result reads each member from the rep row.
                    let members = widekey_cols.as_ref().expect("composite_is_widekey");
                    let mut descriptors: Vec<(u64, u64, u64)> = Vec::with_capacity(members.len());
                    // M3 (doc 21): per-FIXED-member NULL validity offsets, lockstep with `descriptors`
                    // (text members are skipped in both). Empty unless the composite is nullable.
                    let mut validity_descs: Vec<u64> = Vec::new();
                    let mut dst_off: u64 = 0;
                    for &(idx, ty) in members {
                        let (kind, src_off, w) = match ty {
                            SqlType::Numeric { .. } | SqlType::Uuid => (
                                2u64,
                                resident_device_numeric_column_offset(&snapshot, table, idx)?,
                                16u64,
                            ),
                            SqlType::Int8 | SqlType::Timestamp => (
                                1u64,
                                resident_device_int8_column_offset(&snapshot, table, idx)?,
                                8u64,
                            ),
                            // A bool member (1-byte resident) is widened 0/1 -> i64 by build kind 3.
                            SqlType::Bool => (
                                3u64,
                                resident_device_bool_column_offset(&snapshot, table, idx)?,
                                8u64,
                            ),
                            // Text members are not in the fixed buffer (they ride the text descriptor).
                            SqlType::Text => continue,
                            _ => (
                                0u64,
                                resident_device_int4_column_offset(&snapshot, table, idx)?,
                                8u64,
                            ),
                        };
                        descriptors.push((kind, src_off, dst_off));
                        if composite_key_has_null {
                            validity_descs.push(
                                resident_device_null_column_offset(&snapshot, table, idx)?
                                    .unwrap_or(u64::MAX),
                            );
                        }
                        dst_off += w;
                    }
                    let buf = device_memory
                        .build_wide_key_device(
                            &descriptors,
                            widekey_w,
                            row_count,
                            0,
                            &validity_descs,
                        )
                        .map_err(|e| ExecuteError::Engine(EngineError::ApplyFailed(e.to_string())))?;
                    let ptr = buf.device_ptr();
                    _derived_key_buf = Some(buf);
                    ptr
                }
            } else {
                _derived_key_buf = None;
                0
            };
            // General composite TEXT members: each text member's (offsets_off, bytes_off) -> a small
            // device descriptor the wide-key claim folds into the hash AND byte-verifies (row vs rep),
            // in DECLARED-relative order. Built once + held alive across the pass. No text members
            // (all-fixed wide key) -> n_text = 0, byte-identical to the prior wide-key path.
            let _widekey_text_desc;
            let (widekey_n_text, widekey_text_desc_ptr): (u64, u64) = match &widekey_cols {
                Some(members) if row_count > 0 => {
                    let mut flat: Vec<u64> = Vec::new();
                    for &(idx, ty) in members {
                        if matches!(ty, SqlType::Text) {
                            let layout =
                                resident_device_text_column_layout(&snapshot, table, idx)?;
                            flat.push(layout.offsets_byte_offset);
                            flat.push(layout.bytes_byte_offset);
                        }
                    }
                    if flat.is_empty() {
                        _widekey_text_desc = None;
                        (0, 0)
                    } else {
                        let n_text = (flat.len() / 2) as u64;
                        let buf = device_memory.upload_u64_device(&flat).map_err(map_err)?;
                        let ptr = buf.device_ptr();
                        _widekey_text_desc = Some(buf);
                        (n_text, ptr)
                    }
                }
                _ => {
                    _widekey_text_desc = None;
                    (0, 0)
                }
            };
            // One grouping PASS per distinct value column. The kernel yields count+sum+min+max for one
            // value column; each aggregate projects from its column's pass. Every pass groups the SAME
            // key column over the SAME rows, so the i-th group of every pass is the same key (single
            // level is forced when there are >=2 passes so the passes share one compaction order) ->
            // the result merges the passes by group index.
            struct Pass {
                value_idx: usize,
                groups: Vec<gpu_db_execution::GroupByI32Row>,
                value_ty: SqlType,
                value_scale: u8,
                value_is_int8: bool,
                value_is_numeric: bool,
                value_is_uuid: bool,
                value_is_text: bool,
                // A COUNT(DISTINCT v) pass (sort -> mark -> SUM), where `groups[i].sum` is the per-group
                // distinct count. Distinguished from a DIRECT pass on the same `value_idx` so the result
                // builder reads the right one (a column can have both SUM(v) and COUNT(DISTINCT v)).
                is_count_distinct: bool,
            }
            // The result group-key column for an expression GROUP BY is the DERIVED value (no source
            // column); the binding placeholdered it as column 0, so set its name + type to the
            // expression's int4/int8 result so the result schema is correct.
            if is_expr_key {
                if let Some(col) = bound.selected_columns.first_mut() {
                    "?column?".clone_into(&mut col.name);
                    col.ty = key_ty;
                }
            }
            if let Some((_, _, c1, _)) = composite_cols {
                // The binding produced [a, agg...] (GroupedAggregates carries one group_column); the
                // composite result row is [a, b, agg...]. Insert b's full column metadata (from the
                // table -- correct name/type/oid) after a, then renumber the result attnums 1..N.
                let b_col = table.columns[c1].clone();
                bound.selected_columns.insert(1, b_col);
                for (i, col) in bound.selected_columns.iter_mut().enumerate() {
                    col.attnum = (i + 1) as i16;
                }
            } else if let Some(members) = &widekey_cols {
                // Wide-key: the binding produced [member0, agg...]; insert members[1..]'s column metadata
                // after member0 (the result row is [member0..N-1, agg...]), then renumber the attnums.
                for (i, &(idx, _)) in members.iter().enumerate().skip(1) {
                    bound.selected_columns.insert(i, table.columns[idx].clone());
                }
                for (i, col) in bound.selected_columns.iter_mut().enumerate() {
                    col.attnum = (i + 1) as i16;
                }
            }
            // An expression key is read via key_base_override, which only the single-level kernel honors.
            // M3 (doc 21): a nullable VALUE column also forces single-level — only that kernel has the
            // NULL-value skip. (A column with no NULLs has no bitmap, so this never triggers for it.)
            let any_value_nullable = value_indices.iter().try_fold(false, |acc, &vidx| {
                Ok::<bool, ExecuteError>(
                    acc || resident_device_null_column_offset(&snapshot, table, vidx)?.is_some(),
                )
            })?;
            // M3 (doc 21): the group key's NULL validity offset, for a SINGLE-COLUMN key (int2/4/8/date/
            // timestamp/text/numeric/uuid) — the kernel's NULL-key check (now hoisted before the type
            // dispatch) routes a NULL key of any of these to the reserved slot, forming its own group.
            // `None` (no bitmap / a composite/expression key) leaves grouping unchanged. Forces
            // single-level (only that kernel honors the route).
            let pass_key_null_off = if key_is_single_nullable_column {
                resident_device_null_column_offset(&snapshot, table, group_idx)?
            } else {
                // An expression key over exactly one nullable operand reuses that operand's validity bitmap
                // (see expr_key_single_null_off): the kernel routes the expression's NULL rows to the NULL
                // reserved slot, forming their own group, rendered SqlValue::Null.
                expr_key_single_null_off
            };
            let force_single = value_indices.len() > 1
                || is_expr_key
                || any_value_nullable
                || pass_key_null_off.is_some();
            // M3 (doc 21): COUNT(DISTINCT v) over a NULLABLE group key is a clean-error follow-up. The
            // COUNT(DISTINCT) sub-passes (composite_group_count_reps + the step-2 GROUP BY /
            // count_distinct_groups) do NOT route the NULL key to the reserved slot, so they merge NULL-key
            // rows into the placeholder group -> fewer groups than the reference (direct) pass, whose null
            // group IS routed -> the by-index pass merge mis-aligns / panics. Reject cleanly rather than
            // panic or mis-answer. (This guard also covers the pre-existing nullable-INT-key case.) A
            // nullable COMPOSITE key has the same hole: composite_group_count_reps builds its step-1 dedup
            // wide key with NO validity (it would merge (NULL,5) with (0,5)) while step-2 re-groups with the
            // validity-bearing wide key -> a misaligned / dropped group. So clean-error it too (the
            // non-DISTINCT nullable-composite aggregates above are fully supported).
            if (pass_key_null_off.is_some() || composite_key_has_null)
                && aggregates
                    .iter()
                    .any(|a| a.kind == GroupedAggKind::CountDistinct)
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "GROUP BY a nullable key with COUNT(DISTINCT) is not yet supported on the GPU \
                     (M3 3VL follow-up: the COUNT(DISTINCT) sub-pass does not yet route the NULL key to \
                     its own group)"
                        .to_string(),
                )));
            }
            // M3 (doc 21): COUNT(DISTINCT v) over a NULLABLE VALUE column. PG counts distinct NON-NULL
            // values, but the sort-based reps pass groups by (g, v) WITHOUT value validity -- it would fold
            // NULL v into a (deterministic-or-stale) value and count it as a distinct value (an over-count),
            // and an all-NULL-v group would vanish from the pass (misaligning the by-index merge). The value
            // column is NOT in `value_indices` (it skips COUNT(DISTINCT)) so `any_value_nullable` misses it;
            // check it here. Clean-error rather than silently mis-count (excluding NULL v + keeping the
            // all-NULL-v group at 0 is the follow-up).
            for (aggregate, value_idx) in aggregates.iter().zip(&agg_value_indices) {
                if aggregate.kind == GroupedAggKind::CountDistinct {
                    if let Some(idx) = value_idx {
                        if resident_device_null_column_offset(&snapshot, table, *idx)?.is_some() {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "COUNT(DISTINCT) over a nullable column is not yet supported on the GPU \
                                 (M3 3VL follow-up: excluding NULL values from the distinct count)"
                                    .to_string(),
                            )));
                        }
                    }
                }
            }
            let run_pass =
                |value_idx_opt: Option<usize>, has_minmax: bool| -> Result<Pass, ExecuteError> {
                    // A None value column is the COUNT(*)-only pass: group over the key, read .count.
                    let value_idx = value_idx_opt.unwrap_or(group_idx);
                    let value_ty = table.columns[value_idx].ty;
                    // M3 (doc 21) 3VL: a value pass over a NULLABLE column skips NULL values ON-DEVICE so
                    // its count/sum/min/max are over only the non-NULL rows (COUNT(*) passes None and
                    // counts every row). `None` = no bitmap (no NULLs) ⇒ byte-identical. Only the
                    // single-level kernel honors it, so a nullable value forces single-level (below).
                    let pass_value_null_off = if value_idx_opt.is_some() {
                        resident_device_null_column_offset(&snapshot, table, value_idx)?
                    } else {
                        None
                    };
                    let value_scale: u8 = match value_ty {
                        SqlType::Numeric { scale, .. } => scale,
                        _ => 0,
                    };
                    // Classify by the value TYPE (the device read width). MIN/MAX accept every ordered
                    // type; SUM/AVG over an unsupported type is rejected at bind. COUNT reads no value.
                    let (value_is_int8, value_is_numeric, value_is_uuid, value_is_text) =
                        if value_idx_opt.is_none() {
                            (false, false, false, false)
                        } else {
                            match value_ty {
                                SqlType::Int4 | SqlType::Int2 | SqlType::Date => {
                                    (false, false, false, false)
                                }
                                SqlType::Int8 | SqlType::Timestamp => (true, false, false, false),
                                SqlType::Numeric { .. } => (false, true, false, false),
                                SqlType::Uuid => (false, false, true, false),
                                SqlType::Text => (false, false, false, true),
                                // bool value (MIN/MAX): materialized bool->int4, read via
                                // value_base_override on the int4 value path.
                                SqlType::Bool => (false, false, false, false),
                            }
                        };
                    let value_is_bool =
                        value_idx_opt.is_some() && matches!(value_ty, SqlType::Bool);
                    let value_offset = if value_idx_opt.is_none() || value_is_text || value_is_bool {
                        // bool: value_offset is unused (value_base_override is set); key_offset is a safe
                        // placeholder (avoids resolving an int4 offset on a bitmap bool column).
                        key_offset
                    } else if value_is_numeric || value_is_uuid {
                        resident_device_numeric_column_offset(&snapshot, table, value_idx)?
                    } else if value_is_int8 {
                        resident_device_int8_column_offset(&snapshot, table, value_idx)?
                    } else {
                        resident_device_int4_column_offset(&snapshot, table, value_idx)?
                    };
                    // MIN/MAX over a bool VALUE: materialize bool->int4 (0/1) into a derived buffer + read
                    // it via value_base_override. Held alive across the kernel call (closure-local lease).
                    let _derived_value_buf;
                    let value_base_override: u64 = if value_is_bool && row_count > 0 {
                        let offset = resident_device_bool_column_offset(&snapshot, table, value_idx)?;
                        let buf = device_memory
                            .bool_to_int4_column_device(offset, row_count)
                            .map_err(map_err)?;
                        let ptr = buf.device_ptr();
                        _derived_value_buf = Some(buf);
                        ptr
                    } else {
                        _derived_value_buf = None;
                        0
                    };
                    let (value_offsets_off, value_bytes_off) = if value_is_text {
                        let layout =
                            resident_device_text_column_layout(&snapshot, table, value_idx)?;
                        (layout.offsets_byte_offset, layout.bytes_byte_offset)
                    } else {
                        (0, 0)
                    };
                    let use_single_level = has_minmax
                        || value_is_int8
                        || value_is_numeric
                        || value_is_uuid
                        || value_is_text
                        || value_is_bool
                        || key_is_int8
                        || key_is_i128
                        || key_is_text
                        || key_is_bool
                        || composite_is_widekey
                        || force_single;
                    let groups = if use_single_level {
                        device_memory.group_by_i32_count_sum_minmax_from_payload(
                            key_offset,
                            value_offset,
                            &indices,
                            value_is_int8,
                            key_is_int8,
                            value_is_numeric,
                            value_is_uuid,
                            key_is_i128,
                            key_is_text,
                            key_offsets_off,
                            key_bytes_off,
                            value_is_text,
                            value_offsets_off,
                            value_bytes_off,
                            key_base_override,
                            value_base_override,
                            widekey_w,
                            widekey_n_text,
                            widekey_text_desc_ptr,
                            pass_value_null_off,
                            pass_key_null_off,
                        )
                    } else {
                        device_memory.group_by_i32_count_sum_from_payload(
                            key_offset,
                            value_offset,
                            &indices,
                        )
                    }
                    .map_err(map_err)?;
                    Ok(Pass {
                        value_idx,
                        groups,
                        value_ty,
                        value_scale,
                        value_is_int8,
                        value_is_numeric,
                        value_is_uuid,
                        value_is_text,
                        is_count_distinct: false,
                    })
                };
            let mut passes: Vec<Pass> = if value_indices.is_empty() {
                // COUNT(*) only: a single pass over the key column.
                vec![run_pass(None, false)?]
            } else {
                let mut passes = Vec::with_capacity(value_indices.len() + 1);
                // M3 (doc 21): a nullable value pass's `count` is the NON-NULL count, but COUNT(*) needs
                // the TOTAL (the merge reads COUNT(*) from passes[0].count). So when a value is nullable
                // AND the query has a COUNT(*), prepend a dedicated total-count pass (value_null_off=None,
                // every row counted) as the reference. All passes claim a slot for every row, so they
                // share one compaction order; the value passes only skip NULL from their count/sum/min/max.
                // (Without a COUNT(*), passes[0] = the first value pass already carries the full key set.)
                let has_count_star = aggregates.iter().any(|a| a.kind == GroupedAggKind::Count);
                if any_value_nullable && has_count_star {
                    passes.push(run_pass(None, false)?);
                }
                for &value_idx in &value_indices {
                    let has_minmax = aggregates.iter().zip(&agg_value_indices).any(|(a, vi)| {
                        *vi == Some(value_idx)
                            && matches!(a.kind, GroupedAggKind::Min | GroupedAggKind::Max)
                    });
                    passes.push(run_pass(Some(value_idx), has_minmax)?);
                }
                passes
            };
            // COUNT(DISTINCT v) passes: a separate SORT-based pass per such aggregate (the direct hash
            // pass cannot dedup). Materialize (group_key, v) as the i64 tuple matrix over the surviving
            // rows, GPU-sort by (g, v), mark first-seen (g, v) tuples, then SUM the new-distinct flags
            // grouped by g via key/value_base_override -> the per-group distinct count. Scoped to a
            // plain-column int group key (expression/bool/composite/text/numeric/uuid keys are follow-
            // ups); the value column is int2/4/8/date/timestamp (validated at bind). A pure map + the
            // GPU sort + the AUDITED int4 GROUP BY kernel (fully-drained launches) -> no new hazard.
            if aggregates
                .iter()
                .any(|a| a.kind == GroupedAggKind::CountDistinct)
            {
                let idx_u64: Vec<u64> = indices.iter().map(|&i| u64::from(i)).collect();
                let n = idx_u64.len();
                if is_composite_key
                    || composite_is_widekey
                    || key_is_text
                    || key_is_i128
                    || key_is_bool
                    || is_expr_key
                {
                    // NON-int group key: reduce COUNT(DISTINCT v) per g to counting DISTINCT (g, v)
                    // pairs per g. (1) GROUP BY (g..., v) -> one representative row per distinct (g, v)
                    // [the general composite path]; (2) GROUP BY g over those reps, COUNT(*) -> the
                    // distinct-v count per g [reusing the MAIN g config so the groups carry the real g
                    // key and align with the reference pass via materialize_key]. All on the GPU.
                    // An EXPRESSION group key has NO group COLUMNS -- it rides a DERIVED buffer
                    // (key_base_override), fed to step 1 as the wide key's derived member (build kind
                    // 4/5) and reused as the step-2 key config; column group keys pass derived = None.
                    let g_members: Vec<(usize, SqlType)> = if is_expr_key {
                        Vec::new()
                    } else if let Some(m) = &widekey_cols {
                        m.clone()
                    } else if let Some((c0, t0, c1, t1)) = composite_cols {
                        vec![(c0, t0), (c1, t1)]
                    } else {
                        vec![(group_idx, key_ty)]
                    };
                    let derived_g: Option<(u64, bool)> = if is_expr_key {
                        Some((key_base_override, key_expr_is_int8))
                    } else {
                        None
                    };
                    for (aggregate, value_idx) in aggregates.iter().zip(&agg_value_indices) {
                        if aggregate.kind != GroupedAggKind::CountDistinct {
                            continue;
                        }
                        let value_idx = value_idx.expect("COUNT(DISTINCT) has a value column");
                        let groups = if n == 0 {
                            Vec::new()
                        } else {
                            // (1) the distinct (g, v) representative rows.
                            let mut gv_members = g_members.clone();
                            gv_members.push((value_idx, table.columns[value_idx].ty));
                            let reps = composite_group_count_reps(
                                &snapshot,
                                table,
                                &device_memory,
                                &gv_members,
                                &indices,
                                row_count,
                                derived_g,
                            )?;
                            // (2) GROUP BY g over the reps (reusing the main g config) COUNT(*).
                            let mut g2 = if reps.is_empty() {
                                Vec::new()
                            } else {
                                device_memory
                                    .group_by_i32_count_sum_minmax_from_payload(
                                        key_offset,
                                        0,
                                        &reps,
                                        false,
                                        key_is_int8,
                                        false,
                                        false,
                                        key_is_i128,
                                        key_is_text,
                                        key_offsets_off,
                                        key_bytes_off,
                                        false,
                                        0,
                                        0,
                                        key_base_override,
                                        0,
                                        widekey_w,
                                        widekey_n_text,
                                        widekey_text_desc_ptr,
                                        None, // value_null_off (COUNT(DISTINCT) marked-tuple pass)
                                        None, // key_null_off (marked-tuple key non-null by construction)
                                    )
                                    .map_err(map_err)?
                            };
                            // The CountDistinct pass carries the per-group distinct count in `.sum`
                            // (the result builder reads groups[i].sum for such a pass).
                            for grp in &mut g2 {
                                grp.sum = grp.count as i64;
                            }
                            g2
                        };
                        passes.push(Pass {
                            value_idx,
                            groups,
                            value_ty: SqlType::Int8,
                            value_scale: 0,
                            value_is_int8: false,
                            value_is_numeric: false,
                            value_is_uuid: false,
                            value_is_text: false,
                            is_count_distinct: true,
                        });
                    }
                } else {
                    // Plain int group key: the direct sort -> mark -> per-group SUM pipeline.
                    let materialize_i64_col =
                        |col_idx: usize, idx_u64: &[u64]| -> Result<Vec<i64>, ExecuteError> {
                            match table.columns[col_idx].ty {
                                SqlType::Int4 | SqlType::Int2 | SqlType::Date => Ok(device_memory
                                    .project_i32_rows_from_payload(
                                        resident_device_int4_column_offset(
                                            &snapshot, table, col_idx,
                                        )?,
                                        idx_u64,
                                    )
                                    .map_err(map_err)?
                                    .into_iter()
                                    .map(i64::from)
                                    .collect()),
                                SqlType::Int8 | SqlType::Timestamp => device_memory
                                    .project_i64_rows_from_payload(
                                        resident_device_int8_column_offset(
                                            &snapshot, table, col_idx,
                                        )?,
                                        idx_u64,
                                    )
                                    .map_err(map_err),
                                _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                    "COUNT(DISTINCT) group key / value must be an i64-representable \
                                     int column"
                                        .to_string(),
                                ))),
                            }
                        };
                    // The group key column values (sorted-tuple key 0), materialized once per pass.
                    let g_vals = if n > 0 {
                        Some(materialize_i64_col(group_idx, &idx_u64)?)
                    } else {
                        None
                    };
                    for (aggregate, value_idx) in aggregates.iter().zip(&agg_value_indices) {
                        if aggregate.kind != GroupedAggKind::CountDistinct {
                            continue;
                        }
                        let value_idx = value_idx.expect("COUNT(DISTINCT) has a value column");
                        // GPU sort -> mark -> per-group SUM over (group_key, value); see
                        // `count_distinct_groups`. The group key is the real plain-int group column.
                        let groups = if n == 0 {
                            Vec::new()
                        } else {
                            let g_vals = g_vals.as_ref().expect("g_vals materialized for n > 0");
                            count_distinct_groups(value_idx, g_vals, &idx_u64)?
                        };
                        passes.push(Pass {
                            value_idx,
                            groups,
                            value_ty: SqlType::Int8,
                            value_scale: 0,
                            value_is_int8: false,
                            value_is_numeric: false,
                            value_is_uuid: false,
                            value_is_text: false,
                            is_count_distinct: true,
                        });
                    }
                }
            }
            // A TEXT key/value's string -- and a wide-key composite's member values -- live host-side: the
            // kernel stored each group's representative ABSOLUTE row index; read it from the same
            // residency_entry generation as the GPU result.
            // GROUP BY result key/value/member materialization is now fully ON-DEVICE (key_text_map S2.2a,
            // text_value_minmax S2.2b-i, member_cell S2.2b-ii) -- no host_rows clone for the grouped path.
            // S2.2a: a PLAIN text group KEY, materialized ON-DEVICE. One rep_idx -> key-string map gathered
            // from the resident payload (project_text_rows_from_payload) over EVERY pass's group rep indices
            // (materialize_key runs per-pass in #30), instead of reading host_rows. NULL-key groups are
            // excluded (they render SqlValue::Null directly, never via a rep row).
            let key_text_map: Option<std::collections::HashMap<usize, String>> =
                if key_is_text && !is_composite_key && !composite_is_widekey {
                    let mut reps: Vec<u64> = Vec::new();
                    for pass in &passes {
                        for g in &pass.groups {
                            if !g.key_is_null {
                                reps.push(g.key_i128 as u64);
                            }
                        }
                    }
                    reps.sort_unstable();
                    reps.dedup();
                    let layout = resident_device_text_column_layout(&snapshot, table, group_idx)?;
                    let texts = device_memory
                        .project_text_rows_from_payload(
                            layout.offsets_byte_offset,
                            layout.bytes_byte_offset,
                            layout.bytes_len,
                            &reps,
                        )
                        .map_err(map_err)?;
                    Some(reps.iter().map(|&r| r as usize).zip(texts).collect())
                } else {
                    None
                };
            // S2.2b-ii: composite (fixed,text) / wide-key MEMBER values, materialized ON-DEVICE into a
            // sparse (rep_row, col) -> SqlValue map -- replacing the text_host_rows host_rows clone reads.
            // Members may be ANY type, so gather each column by type (mirrors the S1 projection) with NULL
            // validity (a nullable member is SqlValue::Null), at every group's key rep index across ALL
            // passes (covers materialize_key in #30 AND the result builder, incl. NULL-key groups whose
            // members the result builder reads from the rep row).
            let materialize_col_at = |col: usize, reps: &[u64]| -> Result<Vec<SqlValue>, ExecuteError> {
                if reps.is_empty() {
                    return Ok(Vec::new());
                }
                let validity: Option<Vec<bool>> =
                    match resident_device_null_column_offset(&snapshot, table, col)? {
                        Some(off) => Some(
                            device_memory
                                .project_bool_rows_from_payload(off, reps)
                                .map_err(map_err)?,
                        ),
                        None => None,
                    };
                let base: Vec<SqlValue> = match table.columns[col].ty {
                    SqlType::Int8 => device_memory
                        .project_i64_rows_from_payload(
                            resident_device_int8_column_offset(&snapshot, table, col)?,
                            reps,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(SqlValue::Int8)
                        .collect(),
                    SqlType::Timestamp => device_memory
                        .project_i64_rows_from_payload(
                            resident_device_int8_column_offset(&snapshot, table, col)?,
                            reps,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(SqlValue::Timestamp)
                        .collect(),
                    SqlType::Numeric { scale, .. } => device_memory
                        .project_i128_rows_from_payload(
                            resident_device_numeric_column_offset(&snapshot, table, col)?,
                            reps,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(|v| SqlValue::Numeric(Decimal128::new(v, scale)))
                        .collect(),
                    SqlType::Uuid => device_memory
                        .project_i128_rows_from_payload(
                            resident_device_numeric_column_offset(&snapshot, table, col)?,
                            reps,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(|v| SqlValue::Uuid(v.to_le_bytes()))
                        .collect(),
                    SqlType::Date => device_memory
                        .project_i32_rows_from_payload(
                            resident_device_int4_column_offset(&snapshot, table, col)?,
                            reps,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(SqlValue::Date)
                        .collect(),
                    SqlType::Int2 => device_memory
                        .project_i32_rows_from_payload(
                            resident_device_int4_column_offset(&snapshot, table, col)?,
                            reps,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(|v| SqlValue::Int2(v as i16))
                        .collect(),
                    SqlType::Bool => device_memory
                        .project_bool_rows_from_payload(
                            resident_device_bool_column_offset(&snapshot, table, col)?,
                            reps,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(SqlValue::Bool)
                        .collect(),
                    SqlType::Text => {
                        let layout = resident_device_text_column_layout(&snapshot, table, col)?;
                        device_memory
                            .project_text_rows_from_payload(
                                layout.offsets_byte_offset,
                                layout.bytes_byte_offset,
                                layout.bytes_len,
                                reps,
                            )
                            .map_err(map_err)?
                            .into_iter()
                            .map(SqlValue::Text)
                            .collect()
                    }
                    SqlType::Int4 => device_memory
                        .project_i32_rows_from_payload(
                            resident_device_int4_column_offset(&snapshot, table, col)?,
                            reps,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(SqlValue::Int4)
                        .collect(),
                };
                Ok(base
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| match &validity {
                        Some(val) if !val[i] => SqlValue::Null,
                        _ => v,
                    })
                    .collect())
            };
            let member_cell: std::collections::HashMap<(usize, usize), SqlValue> = {
                let member_cols: Vec<usize> = if let Some(m) = &widekey_cols {
                    m.iter().map(|&(c, _)| c).collect()
                } else if composite_is_text {
                    let (c0, _, c1, _) =
                        composite_cols.expect("composite_is_text implies composite_cols");
                    vec![c0, c1]
                } else {
                    Vec::new()
                };
                let mut cells = std::collections::HashMap::new();
                if !member_cols.is_empty() {
                    let mut reps: Vec<u64> = passes
                        .iter()
                        .flat_map(|p| p.groups.iter().map(|g| g.key_i128 as u64))
                        .collect();
                    reps.sort_unstable();
                    reps.dedup();
                    for &col in &member_cols {
                        let vals = materialize_col_at(col, &reps)?;
                        for (r, v) in reps.iter().zip(vals) {
                            cells.insert((*r as usize, col), v);
                        }
                    }
                }
                cells
            };
            // The kernel's hash-slot / compaction order is RACE-dependent: the cas.b64 linear-probe
            // resolves bucket ownership differently per launch, so two passes do NOT share a group
            // order (proven: an int8 i64::MIN-key query misaligned only on the 6th launch). Re-sort
            // every pass by the MATERIALIZED group key so pass[k][i] is the same group across all
            // passes (and the merged output ends up key-ordered). A TEXT key must sort by the
            // materialized string -- there key_i128 is a per-pass representative row index, not the key.
            let materialize_key = |gk: &gpu_db_execution::GroupByI32Row| -> SqlValue {
                // M3 (doc 21): the NULL-KEY group (every NULL key grouped together) renders its key as SQL
                // NULL — for BOTH the pass-alignment sort (NULLs sort consistently) and the result row.
                if gk.key_is_null {
                    return SqlValue::Null;
                }
                if composite_is_widekey {
                    // Wide-key: the b128 slot's lo = the representative row index. Return the FIRST
                    // member's value for the (single-pass) alignment sort; the result rows re-sort by the
                    // FULL member tuple, so this lone value need only be consistent + non-panicking.
                    let rep_idx = gk.key_i128 as u64 as usize;
                    let m0 = widekey_cols.as_ref().expect("composite_is_widekey")[0].0;
                    member_cell[&(rep_idx, m0)].clone()
                } else if is_composite_key {
                    // The result row UNPACKS the key into the two columns; here we only need a
                    // CONSISTENT representation for pass-alignment + ordering. i64 pack -> Int8(key);
                    // i128 pack -> Numeric wrapping the packed i128; (fixed, text) -> the rep row's text
                    // value (composite-text is single-aggregate, so a one-pass sort suffices and the
                    // result rows re-sort by the full tuple). (Composite sets key_is_i128/key_is_text for
                    // the wider/text cases, so this MUST precede those arms below.)
                    if composite_is_text {
                        let rep_idx = gk.key_i128 as u64 as usize;
                        let (cc0, ct0, cc1, _) = composite_cols.unwrap();
                        let text_col = if matches!(ct0, SqlType::Text) { cc0 } else { cc1 };
                        member_cell[&(rep_idx, text_col)].clone()
                    } else if composite_is_i128 {
                        SqlValue::Numeric(Decimal128::new(gk.key_i128, 0))
                    } else {
                        SqlValue::Int8(gk.key)
                    }
                } else if key_is_text {
                    // S2.2a: plain text key, looked up from the ON-DEVICE-gathered map (no host_rows read).
                    let rep_idx = gk.key_i128 as u64 as usize;
                    SqlValue::Text(
                        key_text_map
                            .as_ref()
                            .expect("key_text_map is Some for a plain text key")[&rep_idx]
                            .clone(),
                    )
                } else if key_is_i128 {
                    match key_ty {
                        SqlType::Numeric { .. } => {
                            SqlValue::Numeric(Decimal128::new(gk.key_i128, key_scale))
                        }
                        SqlType::Uuid => SqlValue::Uuid(gk.key_i128.to_le_bytes()),
                        _ => unreachable!("key_is_i128 is only numeric/uuid"),
                    }
                } else {
                    narrow_ordered_value(key_ty, gk.key, 0, 0)
                }
            };
            // The FULL group-key tuple for a group (all key columns/members, in declared order) -- the
            // device-materialized values (key_text_map / member_cell / the packed-int unpack / the typed
            // struct). Used to align passes ON-DEVICE below.
            let full_key = |gk: &gpu_db_execution::GroupByI32Row| -> Vec<SqlValue> {
                if let Some(members) = &widekey_cols {
                    let rep_idx = gk.key_i128 as u64 as usize;
                    members
                        .iter()
                        .map(|&(idx, _)| member_cell[&(rep_idx, idx)].clone())
                        .collect()
                } else if let Some((c0, t0, c1, t1)) = composite_cols {
                    if composite_is_text {
                        let rep_idx = gk.key_i128 as u64 as usize;
                        vec![
                            member_cell[&(rep_idx, c0)].clone(),
                            member_cell[&(rep_idx, c1)].clone(),
                        ]
                    } else {
                        let (col0, col1) = if composite_is_i128 {
                            let k = gk.key_i128;
                            ((k >> 64) as i64, k as u64 as i64)
                        } else {
                            let col0 = ((((gk.key as u64) >> 32) as u32) as i32) as i64;
                            let col1 = (((gk.key as u64) as u32) as i32) as i64;
                            (col0, col1)
                        };
                        vec![
                            narrow_ordered_value(t0, col0, 0, 0),
                            narrow_ordered_value(t1, col1, 0, 0),
                        ]
                    }
                } else {
                    vec![materialize_key(gk)]
                }
            };
            // Pass-alignment (was charter debt #30, NOW ON-DEVICE -- S2.3): each aggregate's hash-agg pass
            // compacts groups in a RACE-dependent order, so MULTIPLE passes don't share a group order; the
            // merge below indexes pass[k][i] expecting the same group. Give every pass ONE shared order by
            // sorting each by the FULL group key ON THE GPU (gpu_sort_permutation). The full key is UNIQUE
            // per group, so this is a TOTAL order -> all passes align by index with no host sort and no
            // sort-stability dependence. #30 needs only CONSISTENT alignment (the FINAL result order is
            // the gpu_sort_permutation window below), so any one deterministic order works; ASC / NULL-first
            // is fine.
            // A single pass needs no alignment (its groups are internally consistent), so skip it there.
            if passes.len() > 1 {
                let key_types: Vec<SqlType> = if let Some(members) = &widekey_cols {
                    members.iter().map(|&(_, t)| t).collect()
                } else if let Some((_, t0, _, t1)) = composite_cols {
                    vec![t0, t1]
                } else {
                    vec![key_ty]
                };
                let key_order: Vec<(usize, bool)> =
                    (0..key_types.len()).map(|c| (c, false)).collect();
                let key_nulls: Vec<Option<bool>> = vec![Some(true); key_types.len()];
                for pass in passes.iter_mut() {
                    let key_rows: Vec<Vec<SqlValue>> =
                        pass.groups.iter().map(|g| full_key(g)).collect();
                    let perm = gpu_sort_permutation(
                        &key_rows,
                        &key_order,
                        &key_nulls,
                        &key_types,
                        &device_memory,
                    )?;
                    pass.groups = perm.iter().map(|&p| pass.groups[p as usize].clone()).collect();
                }
            }
            // S2.2b-i: MIN/MAX over a TEXT value, materialized ON-DEVICE. For each text value pass, gather
            // the result string at every group's g.min / g.max ROW INDEX (project_text_rows_from_payload),
            // not from host_rows. A count==0 (all-NULL) group's min/max is unused (its result is NULL), so
            // it gets a 0 placeholder rep. Passes are #30-aligned here, so [i] matches reference.groups[i].
            let text_value_minmax: std::collections::HashMap<usize, (Vec<String>, Vec<String>)> = {
                let mut m = std::collections::HashMap::new();
                for pass in &passes {
                    if pass.value_is_text && !pass.is_count_distinct {
                        let layout =
                            resident_device_text_column_layout(&snapshot, table, pass.value_idx)?;
                        let gather = |reps: &[u64]| -> Result<Vec<String>, ExecuteError> {
                            device_memory
                                .project_text_rows_from_payload(
                                    layout.offsets_byte_offset,
                                    layout.bytes_byte_offset,
                                    layout.bytes_len,
                                    reps,
                                )
                                .map_err(map_err)
                        };
                        let min_reps: Vec<u64> = pass
                            .groups
                            .iter()
                            .map(|g| if g.count == 0 { 0 } else { g.min as u64 })
                            .collect();
                        let max_reps: Vec<u64> = pass
                            .groups
                            .iter()
                            .map(|g| if g.count == 0 { 0 } else { g.max as u64 })
                            .collect();
                        let min_txt = gather(&min_reps)?;
                        let max_txt = gather(&max_reps)?;
                        m.insert(pass.value_idx, (min_txt, max_txt));
                    }
                }
                m
            };
            // Merge the (now key-aligned) passes by group index into N+1 columns [key, agg_1, .., agg_N].
            // COUNT reads the group's row count from the reference pass; each other aggregate projects
            // from its value column's pass with the per-type narrowing (SUM/AVG widen, MIN/MAX narrow).
            let reference = &passes[0];
            let mut rows: Vec<Vec<SqlValue>> = Vec::with_capacity(reference.groups.len());
            for i in 0..reference.groups.len() {
                let gk = &reference.groups[i];
                let n_group_cols = if is_composite_key {
                    2
                } else if let Some(m) = &widekey_cols {
                    m.len()
                } else {
                    1
                };
                let mut row: Vec<SqlValue> =
                    Vec::with_capacity(aggregates.len() + n_group_cols);
                if let Some(members) = &widekey_cols {
                    // Wide-key: the b128 slot's lo = the representative row index -- read EACH member from
                    // the rep row, materialized ON-DEVICE into member_cell, in DECLARED order.
                    let rep_idx = gk.key_i128 as u64 as usize;
                    for &(idx, _) in members {
                        row.push(member_cell[&(rep_idx, idx)].clone());
                    }
                } else if let Some((c0, t0, c1, t1)) = composite_cols {
                    if composite_is_text {
                        // (fixed, text): the b128 slot holds the rep row index -- read BOTH members from
                        // the rep row, materialized ON-DEVICE into member_cell, in DECLARED order.
                        let rep_idx = gk.key_i128 as u64 as usize;
                        row.push(member_cell[&(rep_idx, c0)].clone());
                        row.push(member_cell[&(rep_idx, c1)].clone());
                    } else {
                        // Two-fixed composite: UNPACK the packed key into the two columns, narrowed to
                        // their real SqlType. i64 pack -> col0 = high 32 bits, col1 = low 32 (`as u32 as
                        // i32` round-trips negatives). i128 pack -> col0 = high 64, col1 = low 64.
                        let (col0, col1) = if composite_is_i128 {
                            let k = gk.key_i128;
                            ((k >> 64) as i64, k as u64 as i64)
                        } else {
                            let col0 = ((((gk.key as u64) >> 32) as u32) as i32) as i64;
                            let col1 = (((gk.key as u64) as u32) as i32) as i64;
                            (col0, col1)
                        };
                        row.push(narrow_ordered_value(t0, col0, 0, 0));
                        row.push(narrow_ordered_value(t1, col1, 0, 0));
                    }
                } else {
                    // The GROUP BY key narrows back to its own type (same closure used for the sort above).
                    row.push(materialize_key(gk));
                }
                for (aggregate, value_idx_opt) in aggregates.iter().zip(&agg_value_indices) {
                    let value = match aggregate.kind {
                        GroupedAggKind::Count => SqlValue::Int8(reference.groups[i].count as i64),
                        GroupedAggKind::CountDistinct => {
                            let value_idx =
                                value_idx_opt.expect("COUNT(DISTINCT) has a value column");
                            // The CountDistinct (sort -> mark -> SUM) pass for THIS column; its
                            // groups[i].sum = the per-group distinct count (a small non-negative i64).
                            let pass = passes
                                .iter()
                                .find(|p| p.value_idx == value_idx && p.is_count_distinct)
                                .expect("a COUNT(DISTINCT) pass exists for its value column");
                            SqlValue::Int8(pass.groups[i].sum)
                        }
                        _ => {
                            let value_idx =
                                value_idx_opt.expect("non-count aggregate has a value column");
                            // A DIRECT pass (not the CountDistinct pass) -- a column may carry both.
                            let pass = passes
                                .iter()
                                .find(|p| p.value_idx == value_idx && !p.is_count_distinct)
                                .expect("a pass exists for every value column");
                            let g = &pass.groups[i];
                            // M3 (doc 21): a value pass's `count` is the NON-NULL count (the kernel skips
                            // NULL values), so count == 0 means EVERY value in this group is NULL -> the
                            // aggregate over zero non-NULL rows is SQL NULL (PG: SUM/AVG/MIN/MAX of no
                            // rows). A non-nullable value never yields count 0 for an existing group, so
                            // this is a no-op for the non-NULL path. COUNT(*) reads the total-count pass.
                            if g.count == 0 {
                                SqlValue::Null
                            } else {
                            // For int8/numeric the SUM is the i128 (sum_hi:sum); else sign-extend.
                            let sum_i128 = if pass.value_is_int8 || pass.value_is_numeric {
                                (i128::from(g.sum_hi) << 64) | i128::from(g.sum as u64)
                            } else {
                                i128::from(g.sum)
                            };
                            match aggregate.kind {
                                GroupedAggKind::Sum if pass.value_is_numeric => {
                                    SqlValue::Numeric(Decimal128::new(sum_i128, pass.value_scale))
                                }
                                GroupedAggKind::Sum if pass.value_is_int8 => {
                                    SqlValue::Numeric(Decimal128::new(sum_i128, 0))
                                }
                                GroupedAggKind::Sum => SqlValue::Int8(g.sum),
                                GroupedAggKind::Avg if pass.value_is_numeric => avg_numeric_sql_value(
                                    sum_i128,
                                    g.count as usize,
                                    pass.value_scale,
                                ),
                                GroupedAggKind::Avg => {
                                    average_sql_value(sum_i128, g.count as usize)
                                }
                                GroupedAggKind::Min if pass.value_is_uuid => {
                                    SqlValue::Uuid(g.min_uuid)
                                }
                                GroupedAggKind::Max if pass.value_is_uuid => {
                                    SqlValue::Uuid(g.max_uuid)
                                }
                                // S2.2b-i: device-gathered MIN/MAX text (text_value_minmax), aligned with
                                // reference.groups by index i. count==0 was handled above (returns NULL).
                                GroupedAggKind::Min if pass.value_is_text => {
                                    SqlValue::Text(text_value_minmax[&pass.value_idx].0[i].clone())
                                }
                                GroupedAggKind::Max if pass.value_is_text => {
                                    SqlValue::Text(text_value_minmax[&pass.value_idx].1[i].clone())
                                }
                                GroupedAggKind::Min => narrow_ordered_value(
                                    pass.value_ty,
                                    g.min,
                                    g.min_hi,
                                    pass.value_scale,
                                ),
                                GroupedAggKind::Max => narrow_ordered_value(
                                    pass.value_ty,
                                    g.max,
                                    g.max_hi,
                                    pass.value_scale,
                                ),
                                GroupedAggKind::Count => unreachable!("count handled above"),
                                GroupedAggKind::CountDistinct => {
                                    unreachable!("count distinct handled above")
                                }
                            }
                            }
                        }
                    };
                    row.push(value);
                }
                rows.push(row);
            }
            // The deterministic default order is applied ON THE GPU below (the gpu_sort_permutation window),
            // not by a host sort -- so the merged rows stay in raw group order here. HAVING (S3: a transient
            // device relation + the predicate VM) and LIMIT/OFFSET (S4: a control-plane window of the device
            // sort permutation) are both ON-DEVICE now; ORDER BY and the default order are GPU sorts. Each
            // clause maps its referenced result column name to an index. All are empty/None for a bare GROUP
            // BY, so only the default GPU order runs there.
            let col_index = |name: &str| -> Result<usize, ExecuteError> {
                let mut hits = bound
                    .selected_columns
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.name.eq_ignore_ascii_case(name));
                let first = hits.next().map(|(i, _)| i).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "GROUP BY ORDER BY / HAVING references unknown column \"{name}\""
                    )))
                })?;
                // PG: a name shared by two aggregates (e.g. SUM(v), SUM(w) -> both "sum") is ambiguous;
                // error rather than silently bind the first.
                if hits.next().is_some() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "GROUP BY ORDER BY / HAVING column reference \"{name}\" is ambiguous"
                    ))));
                }
                Ok(first)
            };
            if !select.having_groups.is_empty() && !rows.is_empty() {
                // HAVING on the GPU (charter: no host relational filter). Build a TRANSIENT device relation
                // from the grouped result and evaluate the HAVING DNF via the SAME device predicate VM as
                // WHERE. The VM is SINGLE-WIDTH per program, and a result column's catalog type can disagree
                // with its materialized value (a `SUM(int*)` result is DECLARED Int4 but VALUED Int8). So
                // PROMOTE every integer-family column (and value) to ONE comparable width: if the HAVING
                // touches a NUMERIC column/constant the whole predicate is NUMERIC (i128) -> promote integers
                // to Numeric(scale 0); else it is INT (i64) -> promote integers to Int8. This makes int-key +
                // int8-COUNT, and numeric-SUM + int-COUNT, a single width the VM can lower.
                let n_cols = bound.selected_columns.len();
                let numeric_mode = select.having_groups.iter().flatten().try_fold(
                    false,
                    |acc, f| -> Result<bool, ExecuteError> {
                        let idx = col_index(&f.column)?;
                        Ok(acc
                            || matches!(f.value, SqlValue::Numeric(_))
                            || matches!(bound.selected_columns[idx].ty, SqlType::Numeric { .. })
                            || rows.iter().any(|r| matches!(r[idx], SqlValue::Numeric(_))))
                    },
                )?;
                let promote_int: Vec<bool> = (0..n_cols)
                    .map(|c| {
                        rows.iter().any(|r| {
                            matches!(
                                r[c],
                                SqlValue::Int2(_)
                                    | SqlValue::Int4(_)
                                    | SqlValue::Int8(_)
                                    | SqlValue::Date(_)
                                    | SqlValue::Timestamp(_)
                            )
                        })
                    })
                    .collect();
                // Per result column, the transient column's target. An integer-family column is promoted
                // (to Numeric scale 0 in numeric_mode, else Int8). A GENUINE numeric column is normalized to
                // the MAX scale across its values: AVG yields per-GROUP scales (PG division), but
                // build_relational_device_payload stores only mantissas at ONE column scale, so every value
                // must share it -- rescaling UP to the max is exact (no rounding). `col_scale[c] = Some(s)`
                // marks a numeric-target column at scale `s`.
                let col_scale: Vec<Option<u8>> = (0..n_cols)
                    .map(|c| {
                        if promote_int[c] {
                            numeric_mode.then_some(0u8)
                        } else {
                            rows.iter()
                                .filter_map(|r| match &r[c] {
                                    SqlValue::Numeric(d) => Some(d.scale),
                                    _ => None,
                                })
                                .max()
                        }
                    })
                    .collect();
                let having_columns: Vec<RelationalColumn> = bound
                    .selected_columns
                    .iter()
                    .enumerate()
                    .map(|(c, col)| {
                        let mut col = col.clone();
                        if let Some(scale) = col_scale[c] {
                            col.ty = SqlType::Numeric {
                                precision: 38,
                                scale,
                            };
                        } else if promote_int[c] {
                            col.ty = SqlType::Int8;
                        }
                        col
                    })
                    .collect();
                let having_rows: Vec<Vec<SqlValue>> = rows
                    .iter()
                    .map(|r| -> Result<Vec<SqlValue>, ExecuteError> {
                        r.iter()
                            .enumerate()
                            .map(|(c, v)| -> Result<SqlValue, ExecuteError> {
                                let int_as_i64 = match v {
                                    SqlValue::Int2(x) => Some(i64::from(*x)),
                                    SqlValue::Int4(x) | SqlValue::Date(x) => Some(i64::from(*x)),
                                    SqlValue::Int8(x) | SqlValue::Timestamp(x) => Some(*x),
                                    _ => None,
                                };
                                Ok(match (col_scale[c], promote_int[c]) {
                                    // int-family promoted to Numeric (scale 0; int_as_i64 is exact).
                                    (Some(scale), true) => match int_as_i64 {
                                        Some(x) => {
                                            SqlValue::Numeric(Decimal128::new(i128::from(x), scale))
                                        }
                                        None => v.clone(),
                                    },
                                    // genuine numeric -> rescale UP to the column's max scale (exact).
                                    (Some(scale), false) => match v {
                                        SqlValue::Numeric(d) => {
                                            SqlValue::Numeric(d.rescale(scale).map_err(|_| {
                                                ExecuteError::Engine(EngineError::ApplyFailed(
                                                    "HAVING numeric value overflowed normalizing scale"
                                                        .to_string(),
                                                ))
                                            })?)
                                        }
                                        _ => v.clone(),
                                    },
                                    // int-family promoted to Int8 (no numeric in the predicate).
                                    (None, true) => match int_as_i64 {
                                        Some(x) => SqlValue::Int8(x),
                                        None => v.clone(),
                                    },
                                    // text/bool/uuid (or an all-NULL numeric column) -- unchanged.
                                    (None, false) => v.clone(),
                                })
                            })
                            .collect()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                // DNF -> ResidentExpr: each filter -> `Column(idx) <op> literal`; AND within a group, OR
                // across groups. col_index still validates each referenced name (unknown / ambiguous).
                let mut dnf: Option<ResidentExpr> = None;
                for group in &select.having_groups {
                    let mut conj: Option<ResidentExpr> = None;
                    for f in group {
                        let leaf = ResidentExpr::Binary {
                            op: having_op_to_resident(f.op),
                            lhs: Box::new(ResidentExpr::Column(col_index(&f.column)?)),
                            rhs: Box::new(having_value_to_resident_literal(&f.value, numeric_mode)?),
                        };
                        conj = Some(match conj {
                            None => leaf,
                            Some(prev) => ResidentExpr::Binary {
                                op: ResidentBinaryOp::And,
                                lhs: Box::new(prev),
                                rhs: Box::new(leaf),
                            },
                        });
                    }
                    if let Some(c) = conj {
                        dnf = Some(match dnf {
                            None => c,
                            Some(prev) => ResidentExpr::Binary {
                                op: ResidentBinaryOp::Or,
                                lhs: Box::new(prev),
                                rhs: Box::new(c),
                            },
                        });
                    }
                }
                if let Some(predicate) = dnf {
                    let having_table = RelationalTable {
                        schema: table.schema.clone(),
                        name: table.name.clone(),
                        oid: table.oid,
                        columns: having_columns,
                        indexes: Vec::new(),
                        check_constraints: Vec::new(),
                        foreign_keys: Vec::new(),
                        acl: std::collections::BTreeMap::new(),
                    };
                    let (h_snapshot, h_memory) =
                        self.build_transient_relation_residency(&having_table, &having_rows)?;
                    let survivors = self.lower_resident_predicate(
                        &predicate,
                        &having_table,
                        &h_snapshot,
                        &h_memory,
                        having_rows.len() as u64,
                    )?;
                    let kept: Vec<Vec<SqlValue>> =
                        survivors.iter().map(|&i| rows[i as usize].clone()).collect();
                    rows = kept;
                }
            }
            // The grouped result is ordered ON THE GPU (charter: every relational sort is a GPU sort, no
            // host-side finalization) -- both the explicit ORDER BY and, in its absence, the deterministic
            // DEFAULT order. INT/bool keys feed an i64 matrix by group position; TEXT/NUMERIC/UUID keys feed
            // a resident-like payload built (build_relational_device_payload) from just those result
            // columns, which the hetero comparator reads on-device. Single- and multi-key share the path.
            // OFFSET/LIMIT is then a control-plane WINDOW of the device-produced permutation -- never a host
            // relational drain/truncate on result data. gpu_sort_permutation returns the sort index vector
            // (identity for <=1 row / empty order); we slice it to [OFFSET, OFFSET+LIMIT) and gather ONLY
            // that window from the materialized group rows. With no LIMIT the window is the full range, so
            // this is byte-identical to the prior gpu_sort_result_rows reorder.
            if rows.len() > 1 || select.offset.is_some() || select.limit.is_some() {
                let col_types: Vec<SqlType> =
                    bound.selected_columns.iter().map(|c| c.ty).collect();
                // The GROUP-KEY result columns (emitted first by the merge): result columns 0..n_group_cols.
                let n_group_cols = if is_composite_key {
                    2
                } else if let Some(members) = &widekey_cols {
                    members.len()
                } else {
                    1
                };
                let (order, nulls_first): (Vec<(usize, bool)>, Vec<Option<bool>>) =
                    if select.order_by.is_empty() {
                        // DEFAULT order: by the full group-key tuple, ASC, with the NULL group FIRST --
                        // matching the prior host key order (PG leaves a bare GROUP BY unordered; our
                        // stable choice).
                        (
                            (0..n_group_cols).map(|c| (c, false)).collect(),
                            vec![Some(true); n_group_cols],
                        )
                    } else {
                        // Explicit ORDER BY: resolve each key to its result-column index; explicit NULLS
                        // FIRST/LAST honored ON-DEVICE (order_by_nulls_first threaded through; every key
                        // type, incl. int, reads its NULL validity bitmap in the comparator).
                        let mut order: Vec<(usize, bool)> = select
                            .order_by
                            .iter()
                            .map(|o| Ok::<_, ExecuteError>((col_index(&o.column)?, o.descending)))
                            .collect::<Result<_, _>>()?;
                        let mut nulls_first = order_by_nulls_first.to_vec();
                        // Append the GROUP-KEY columns (ASC, NULL group first) as a deterministic TIE-BREAK
                        // so rows that tie on the explicit ORDER BY key(s) -- e.g. two groups with equal
                        // SUM under `ORDER BY sum DESC` -- order by group key. This matches the legacy
                        // probe/host group-ASC tie-break (a documented cross-path contract) and keeps the
                        // result fully deterministic. The grouped result is at most one row per group, so
                        // the extra (cheap) sort key is negligible. A group column already named as an
                        // explicit key is skipped (it would be a redundant, no-op secondary key).
                        for c in 0..n_group_cols {
                            if !order.iter().any(|(idx, _)| *idx == c) {
                                order.push((c, false));
                                nulls_first.push(Some(true));
                            }
                        }
                        (order, nulls_first)
                    };
                let perm =
                    gpu_sort_permutation(&rows, &order, &nulls_first, &col_types, &device_memory)?;
                let start = select.offset.unwrap_or(0).min(perm.len());
                let end = select
                    .limit
                    .map_or(perm.len(), |l| start.saturating_add(l).min(perm.len()));
                let windowed: Vec<Vec<SqlValue>> =
                    perm[start..end].iter().map(|&p| rows[p as usize].clone()).collect();
                rows = windowed;
            }
            return Ok(RelationalSelectResult {
                columns: bound.selected_columns,
                rows,
                planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
                executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
                fallback_reason: None,
                access_path,
            });
        }

        // Scalar aggregate? Compute it from the filtered indices and return a single row.
        if is_aggregate {
            let value = match &select.projection {
                // COUNT(*): the surviving row count IS the result (PG returns bigint). The GPU filter
                // + compaction already produced the count; no per-row materialization.
                SelectProjection::CountAll => SqlValue::Int8(indices.len() as i64),
                // SUM(int4): a GPU reduction over the filtered column (gather col[indices] + reduce);
                // PG returns bigint. An EMPTY filtered set is SQL NULL, which the engine cannot
                // represent until M3 -- so it hard-errors rather than returning a wrong 0.
                SelectProjection::Sum { column } => {
                    let col_idx = relational_column_index(table, column)?;
                    if indices.is_empty() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "SUM over an empty set is NULL, which the engine cannot represent yet \
                             (NULL support is M3)"
                                .to_string(),
                        )));
                    }
                    let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    };
                    // PG: SUM(int4) -> bigint; SUM(int8) -> numeric (the sum can exceed i64, so it
                    // reduces into i128 via the two-atomic carry kernel).
                    match table.columns[col_idx].ty {
                        SqlType::Int4 => {
                            let byte_offset =
                                resident_device_int4_column_offset(&snapshot, table, col_idx)?;
                            let sum = device_memory
                                .sum_i32_at_indices_from_payload(byte_offset, &indices)
                                .map_err(map_err)?;
                            SqlValue::Int8(sum)
                        }
                        SqlType::Int8 => {
                            let byte_offset =
                                resident_device_int8_column_offset(&snapshot, table, col_idx)?;
                            let sum = device_memory
                                .sum_i64_at_indices_i128_from_payload(byte_offset, &indices)
                                .map_err(map_err)?;
                            SqlValue::Numeric(Decimal128::new(sum, 0))
                        }
                        // SUM(numeric) -> numeric at the column scale: sum the i128 mantissas (the
                        // partials kernel with CHECKED i128 overflow -> numeric field overflow).
                        SqlType::Numeric { scale, .. } => {
                            let byte_offset =
                                resident_device_numeric_column_offset(&snapshot, table, col_idx)?;
                            let mantissa = device_memory
                                .sum_i128_at_indices_from_payload(byte_offset, &indices)
                                .map_err(map_err)?;
                            SqlValue::Numeric(Decimal128::new(mantissa, scale))
                        }
                        _ => {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "SUM supports int4 / int8 / numeric columns on the Expr path"
                                    .to_string(),
                            )));
                        }
                    }
                }
                // MIN/MAX(int4): a GPU reduction; PG MIN/MAX preserve the column type (int4 -> int4).
                // Empty set is NULL -> hard error (M3), like SUM.
                SelectProjection::Min { column } | SelectProjection::Max { column } => {
                    let col_idx = relational_column_index(table, column)?;
                    if indices.is_empty() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "MIN / MAX over an empty set is NULL, which the engine cannot represent \
                             yet (NULL support is M3)"
                                .to_string(),
                        )));
                    }
                    let is_max = matches!(select.projection, SelectProjection::Max { .. });
                    let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    };
                    // MIN/MAX preserve the column type (PG): int4 -> int4 (i32 reduce), int8 -> int8
                    // (i64 reduce). Other types are follow-ons.
                    match table.columns[col_idx].ty {
                        SqlType::Int4 => {
                            let byte_offset =
                                resident_device_int4_column_offset(&snapshot, table, col_idx)?;
                            let value = if is_max {
                                device_memory
                                    .max_i32_at_indices_from_payload(byte_offset, &indices)
                            } else {
                                device_memory
                                    .min_i32_at_indices_from_payload(byte_offset, &indices)
                            }
                            .map_err(map_err)?;
                            SqlValue::Int4(value)
                        }
                        SqlType::Int8 => {
                            let byte_offset =
                                resident_device_int8_column_offset(&snapshot, table, col_idx)?;
                            let value = if is_max {
                                device_memory
                                    .max_i64_at_indices_from_payload(byte_offset, &indices)
                            } else {
                                device_memory
                                    .min_i64_at_indices_from_payload(byte_offset, &indices)
                            }
                            .map_err(map_err)?;
                            SqlValue::Int8(value)
                        }
                        // MIN/MAX(numeric) -> numeric: reduce the i128 mantissas (no native 128-bit
                        // atomic, so a partials + host-combine reduction); the result carries the
                        // column scale.
                        SqlType::Numeric { scale, .. } => {
                            let byte_offset =
                                resident_device_numeric_column_offset(&snapshot, table, col_idx)?;
                            let mantissa = if is_max {
                                device_memory
                                    .max_i128_at_indices_from_payload(byte_offset, &indices)
                            } else {
                                device_memory
                                    .min_i128_at_indices_from_payload(byte_offset, &indices)
                            }
                            .map_err(map_err)?;
                            SqlValue::Numeric(Decimal128::new(mantissa, scale))
                        }
                        _ => {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "MIN / MAX support int4 / int8 / numeric columns on the Expr path"
                                    .to_string(),
                            )));
                        }
                    }
                }
                // AVG(int4) = the GPU SUM / the count, as numeric (PG). The reduction is on the GPU;
                // the final scalar divide reuses `average_sql_value` (scale-16, matching the enumerated
                // path). Empty set is NULL -> hard error (M3), like the others.
                SelectProjection::Avg { column } => {
                    let col_idx = relational_column_index(table, column)?;
                    if indices.is_empty() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "AVG over an empty set is NULL, which the engine cannot represent yet \
                             (NULL support is M3)"
                                .to_string(),
                        )));
                    }
                    let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    };
                    // AVG -> numeric = the GPU sum / the count. int4 reduces to i64 (widened), int8 to
                    // i128 (the two-atomic carry); both divide via average_sql_value (integer sum).
                    // numeric sums the i128 mantissas and divides via avg_numeric_sql_value, which
                    // carries the column scale into PG's division-scale derivation.
                    match table.columns[col_idx].ty {
                        SqlType::Int4 => {
                            let byte_offset =
                                resident_device_int4_column_offset(&snapshot, table, col_idx)?;
                            let sum = i128::from(
                                device_memory
                                    .sum_i32_at_indices_from_payload(byte_offset, &indices)
                                    .map_err(map_err)?,
                            );
                            average_sql_value(sum, indices.len())
                        }
                        SqlType::Int8 => {
                            let byte_offset =
                                resident_device_int8_column_offset(&snapshot, table, col_idx)?;
                            let sum = device_memory
                                .sum_i64_at_indices_i128_from_payload(byte_offset, &indices)
                                .map_err(map_err)?;
                            average_sql_value(sum, indices.len())
                        }
                        SqlType::Numeric { scale, .. } => {
                            let byte_offset =
                                resident_device_numeric_column_offset(&snapshot, table, col_idx)?;
                            let mantissa = device_memory
                                .sum_i128_at_indices_from_payload(byte_offset, &indices)
                                .map_err(map_err)?;
                            avg_numeric_sql_value(mantissa, indices.len(), scale)
                        }
                        _ => {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "AVG supports int4 / int8 / numeric columns on the Expr path"
                                    .to_string(),
                            )));
                        }
                    }
                }
                // Scalar COUNT(DISTINCT v) (no GROUP BY) = ONE group: sort/mark/SUM over (g=0, v) via
                // `count_distinct_groups` with a constant group key, so the single group's count is the
                // total distinct. PG: COUNT(DISTINCT) over zero rows is 0 (NOT NULL), so an empty
                // filtered set returns 0 (the lone exception to the SUM/AVG empty-set NULL hard-error).
                SelectProjection::CountDistinct { column } => {
                    let value_idx = relational_column_index(table, column)?;
                    // M3 (doc 21): COUNT(DISTINCT v) counts distinct NON-NULL values (PG). The sort/mark/SUM
                    // pass has no value validity, so a NULL v would be counted as a distinct value (an
                    // over-count). Clean-error a nullable value rather than silently mis-count (the grouped
                    // path guards this too); excluding NULL v is the follow-up.
                    if resident_device_null_column_offset(&snapshot, table, value_idx)?.is_some() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "COUNT(DISTINCT) over a nullable column is not yet supported on the GPU \
                             (M3 3VL follow-up: excluding NULL values from the distinct count)"
                                .to_string(),
                        )));
                    }
                    if indices_u64.is_empty() {
                        SqlValue::Int8(0)
                    } else {
                        let g_vals = vec![0i64; indices_u64.len()];
                        let groups = count_distinct_groups(value_idx, &g_vals, &indices_u64)?;
                        // The constant key yields exactly one group; its SUM = the total distinct.
                        SqlValue::Int8(groups.first().map_or(0, |g| g.sum))
                    }
                }
                _ => unreachable!(
                    "is_aggregate gates on CountAll | Sum | Min | Max | Avg | CountDistinct"
                ),
            };
            return Ok(RelationalSelectResult {
                columns: bound.selected_columns,
                rows: vec![vec![value]],
                planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
                executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
                fallback_reason: None,
                access_path,
            });
        }

        // Non-grouped ORDER BY: reorder the surviving indices by the order key(s) on the GPU (bitonic
        // sort) BEFORE gathering, so the projected rows come out sorted -- a charter-native GPU sort,
        // not a host/CPU sort. The routing only sends sorts whose keys are ALL i64-sortable int columns
        // (int2/int4/int8/date/timestamp) to this path -- one key OR several (`ORDER BY a ASC, b DESC`).
        // A single TEXT-key ORDER BY takes the varlen GPU sort path (the byte-wise comparator); all
        // other routed keys are i64-sortable ints and go through the key-matrix path below.
        // Classify the ORDER BY keys: k==1 text -> the varlen text sort; k>1 with ANY text key -> the
        // heterogeneous mixed (int+text) sort; all-int -> the i64 key matrix.
        // Materialize an INT-bearing ORDER BY key's i64 values for `indices`. A sort EXPRESSION
        // (`ORDER BY a+b`) is evaluated on-device into an i64 column (checked int4 overflow -> PG error,
        // never CPU); a plain int column is projected. Both sort arms below use this; an expression key
        // always counts as an int key. `indices` is a param (the else arm below moves `indices_u64`).
        let materialize_int_key_column = |ki: usize, indices: &[u64]| -> Result<Vec<i64>, ExecuteError> {
            let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
            };
            if let Some(Some(expr)) = order_by_exprs.get(ki) {
                let mut program = Vec::new();
                compile_arith_program(expr, table, &snapshot, &mut program)?;
                let idx32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
                // int8 operands -> the i64 arith VM + i64 value width; else int4/I32. Reading an int8
                // expr as I32 would stride a BIGINT column by 4 bytes -> silently garbage sort keys.
                let elem = if expr_mentions_int8(expr, table) {
                    ResidentElemType::I64
                } else {
                    ResidentElemType::I32
                };
                // M3 (doc 21): a NULLABLE int4 expression blends the i64::MAX default-end sentinel
                // ON-DEVICE where any operand is NULL (a validity mask VM run + the on-device blend). int8
                // nullable + explicit NULLS FIRST/LAST are clean-errored at the routing loop below; a
                // non-nullable expression keeps the plain (no-validity) path.
                if elem == ResidentElemType::I32
                    && predicate_references_nullable_column(expr, table, &snapshot)?
                {
                    let mut validity_program = vec![ExprStep::ConstMask { value: true }];
                    push_leaf_validity_and(&[expr], table, &snapshot, &mut validity_program)?;
                    return device_memory
                        .arith_value_column_at_indices_nullable(
                            &program,
                            &validity_program,
                            row_count,
                            &idx32,
                        )
                        .map_err(map_err);
                }
                return device_memory
                    .arith_value_column_at_indices(&program, row_count, &idx32, elem)
                    .map_err(map_err);
            }
            let order = &select.order_by[ki];
            let order_idx = relational_column_index(table, &order.column)?;
            let keys: Vec<i64> = match table.columns[order_idx].ty {
                SqlType::Int4 | SqlType::Int2 | SqlType::Date => device_memory
                    .project_i32_rows_from_payload(
                        resident_device_int4_column_offset(&snapshot, table, order_idx)?,
                        indices,
                    )
                    .map_err(map_err)?
                    .into_iter()
                    .map(i64::from)
                    .collect(),
                SqlType::Int8 | SqlType::Timestamp => device_memory
                    .project_i64_rows_from_payload(
                        resident_device_int8_column_offset(&snapshot, table, order_idx)?,
                        indices,
                    )
                    .map_err(map_err)?,
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "ORDER BY key is not an i64-sortable int column".to_string(),
                    )))
                }
            };
            // M3 (doc 21): NULL placement is done ON-DEVICE by the sort comparator (it reads the per-key
            // validity bitmap and orders NULL keys to the PG-default end), so the key VALUES here are raw —
            // a NULL row's placeholder value is never compared. (A nullable key always routes to the
            // validity-aware hetero comparator below; this raw matrix only feeds that path or the non-null
            // pure-int path.) No host-side NULL overwrite.
            Ok(keys)
        };
        // A sort EXPRESSION is int-valued. has_text_key (TEXT only) gates the single-text fast path;
        // has_hetero_key (TEXT/NUMERIC/UUID -- the keys that can't live in the i64 matrix) routes to the
        // heterogeneous comparator.
        let mut has_text_key = false;
        let mut has_hetero_key = false;
        // M3 (doc 21): per-key NULL validity bitmap byte offset (sentinel u64::MAX = the key holds no NULL).
        // The hetero sort comparator reads this ON-DEVICE and orders NULL keys to the PG-default end (last
        // ASC / first DESC) — GPU-native, no host partition / sentinel. A nullable key of ANY type routes
        // to that comparator (below). A nullable sort EXPRESSION stays a clean-error follow-up (an
        // expression has no single column validity bitmap; a derived one is a follow-up).
        let mut key_null_offs: Vec<u64> = vec![u64::MAX; select.order_by.len()];
        for (ki, order) in select.order_by.iter().enumerate() {
            if let Some(Some(expr)) = order_by_exprs.get(ki) {
                // A NULLABLE int4 expression is supported via the on-device value-sentinel
                // (materialize_int_key_column): a NULL result becomes the i64::MAX default-end sentinel,
                // so key_null_offs stays u64::MAX (no validity bitmap -- the NULL is value-encoded). Two
                // cases still clean-error: an int8 expression (i64::MAX could collide with a real result)
                // and an explicit NULLS FIRST/LAST (the value-sentinel only realizes the PG default).
                if predicate_references_nullable_column(expr, table, &snapshot)? {
                    if expr_mentions_int8(expr, table) {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "ORDER BY a nullable int8 expression is a follow-up (the i64::MAX NULL \
                             sentinel could collide with a real bigint result)"
                                .to_string(),
                        )));
                    }
                    if order_by_nulls_first.get(ki).copied().flatten().is_some() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "explicit NULLS FIRST/LAST on a nullable ORDER BY expression is a follow-up \
                             (the default NULL placement is supported)"
                                .to_string(),
                        )));
                    }
                }
                continue;
            }
            let order_idx = relational_column_index(table, &order.column)?;
            match table.columns[order_idx].ty {
                SqlType::Text => {
                    has_text_key = true;
                    has_hetero_key = true;
                }
                SqlType::Numeric { .. } | SqlType::Uuid => has_hetero_key = true,
                _ => {}
            }
            if let Some(off) = resident_device_null_column_offset(&snapshot, table, order_idx)? {
                key_null_offs[ki] = off;
            }
        }
        // A nullable key (any type) must route to the validity-aware hetero comparator. A non-null single
        // text key keeps the dedicated text fast path; non-null int-only keys keep the int matrix/radix path.
        let any_nullable_key = key_null_offs.iter().any(|&o| o != u64::MAX);
        let single_text_key = select.order_by.len() == 1 && has_text_key && !any_nullable_key;
        let use_hetero = has_hetero_key || any_nullable_key;
        let indices_u64 = if select.order_by.is_empty() {
            indices_u64
        } else if single_text_key {
            // The varlen text key can't live in the i64 key matrix, so the kernel sorts the surviving
            // rows by reading each row's bytes from the resident text column via the indices indirection
            // (lexicographic, unsigned bytes, a prefix sorts smaller). Charter-native GPU sort.
            let order = &select.order_by[0];
            let order_idx = relational_column_index(table, &order.column)?;
            let layout = resident_device_text_column_layout(&snapshot, table, order_idx)?;
            let perm = device_memory
                .bitonic_sort_text(
                    &indices_u64,
                    layout.offsets_byte_offset,
                    layout.bytes_byte_offset,
                    order.descending,
                )
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            perm.iter().map(|&p| indices_u64[p as usize]).collect()
        } else if use_hetero {
            // A key tuple with a TEXT/NUMERIC/UUID key (`ORDER BY name /*text*/, age /*int*/`, or a
            // single numeric/uuid key): the heterogeneous GPU comparator dispatches each key to the s64
            // compare (int, from a row-major by-position matrix), the byte compare (text, in place), or
            // the 16-byte compare (numeric = signed-hi/unsigned-lo i128; uuid = big-endian unsigned),
            // reading numeric/uuid in place from the resident column. Charter-native GPU sort.
            let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
            };
            let n = indices_u64.len();
            let k = select.order_by.len();
            if k > 64 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "ORDER BY supports at most 64 sort keys on the GPU sort path".to_string(),
                )));
            }
            // Walk the keys in order: each int key takes the next int-matrix column, each text key the
            // next text-column slot; build key_plan (bit31=is_text, low bits=slot) + desc_mask.
            let mut num_int = 0usize;
            let mut text_cols: Vec<(u64, u64)> = Vec::new();
            let mut b128_cols: Vec<u64> = Vec::new();
            let mut key_plan: Vec<u32> = Vec::with_capacity(k);
            let mut desc_mask: u64 = 0;
            // Per-key effective NULLS FIRST bit: the explicit override, else the key's DESC (PG default).
            // The comparator reads it to place NULLs ON-DEVICE, decoupled from the value-compare direction.
            let mut nulls_first_mask: u64 = 0;
            let mut int_key_slots: Vec<(usize, usize)> = Vec::new();
            for (ki, order) in select.order_by.iter().enumerate() {
                if order.descending {
                    desc_mask |= 1u64 << ki;
                }
                if order_by_nulls_first
                    .get(ki)
                    .copied()
                    .flatten()
                    .unwrap_or(order.descending)
                {
                    nulls_first_mask |= 1u64 << ki;
                }
                // A sort EXPRESSION is an int-valued key -> the next int slot (materialized below).
                if order_by_exprs.get(ki).is_some_and(|e| e.is_some()) {
                    let int_slot = num_int;
                    num_int += 1;
                    key_plan.push(int_slot as u32);
                    int_key_slots.push((ki, int_slot));
                    continue;
                }
                let order_idx = relational_column_index(table, &order.column)?;
                match table.columns[order_idx].ty {
                    SqlType::Text => {
                        let layout =
                            resident_device_text_column_layout(&snapshot, table, order_idx)?;
                        let text_slot = text_cols.len() as u32;
                        text_cols.push((layout.offsets_byte_offset, layout.bytes_byte_offset));
                        // kind 1 (key_plan bits 30-31 = 01) = text.
                        key_plan.push(0x4000_0000_u32 | text_slot);
                    }
                    SqlType::Numeric { .. } => {
                        let b128_slot = b128_cols.len() as u32;
                        b128_cols.push(resident_device_numeric_column_offset(
                            &snapshot, table, order_idx,
                        )?);
                        // kind 2 (bits 30-31 = 10) = numeric (signed-hi/unsigned-lo i128).
                        key_plan.push(0x8000_0000_u32 | b128_slot);
                    }
                    SqlType::Uuid => {
                        let b128_slot = b128_cols.len() as u32;
                        b128_cols.push(resident_device_numeric_column_offset(
                            &snapshot, table, order_idx,
                        )?);
                        // kind 3 (bits 30-31 = 11) = uuid (big-endian unsigned).
                        key_plan.push(0xC000_0000_u32 | b128_slot);
                    }
                    SqlType::Int4
                    | SqlType::Int2
                    | SqlType::Date
                    | SqlType::Int8
                    | SqlType::Timestamp => {
                        let int_slot = num_int;
                        num_int += 1;
                        key_plan.push(int_slot as u32);
                        int_key_slots.push((ki, int_slot));
                    }
                    _ => {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "mixed-key ORDER BY on the Expr path supports int2 / int4 / int8 / date / \
                             timestamp / text / numeric / uuid columns (the GPU sort); other types \
                             are a follow-on"
                                .to_string(),
                        )));
                    }
                }
            }
            // Materialize the int keys into a row-major n*num_int matrix by position (matching the
            // multikey kernel's layout; the text keys read in place via the indices indirection).
            let mut int_keys = vec![0i64; n * num_int];
            for (ki, int_slot) in int_key_slots {
                let col_keys = materialize_int_key_column(ki, &indices_u64)?;
                for (i, v) in col_keys.into_iter().enumerate() {
                    int_keys[i * num_int + int_slot] = v;
                }
            }
            let perm = device_memory
                .bitonic_sort_hetero(
                    &indices_u64,
                    &int_keys,
                    num_int,
                    &text_cols,
                    &b128_cols,
                    &key_plan,
                    desc_mask,
                    &key_null_offs,
                    nulls_first_mask,
                )
                .map_err(map_err)?;
            perm.iter().map(|&p| indices_u64[p as usize]).collect()
        } else {
            let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
            };
            let n = indices_u64.len();
            let k = select.order_by.len();
            if k > 64 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "ORDER BY supports at most 64 sort keys on the GPU sort path".to_string(),
                )));
            }
            // Materialize every ORDER BY key column's i64 value for the surviving rows into a row-major
            // key matrix (`keys[row * K + key]`, key 0 most significant) and the per-key direction mask
            // (bit k set => key k is DESC). int2/int4/date sign-extend through i32; int8/timestamp are
            // the full i64. The GPU multi-key comparator breaks ties on key 0 by key 1, then key 2, ...
            let mut key_matrix = vec![0i64; n * k];
            let mut desc_mask: u64 = 0;
            for (kk, order) in select.order_by.iter().enumerate() {
                if order.descending {
                    desc_mask |= 1u64 << kk;
                }
                // Each key is a plain int column OR a sort expression (`a+b`); the helper materializes
                // the i64 value column either way (expression -> on-device eval with checked overflow).
                let col_keys = materialize_int_key_column(kk, &indices_u64)?;
                for (i, v) in col_keys.into_iter().enumerate() {
                    key_matrix[i * k + kk] = v;
                }
            }
            // Single-key dispatches by size -- bitonic for small n, radix (O(n)) for large n -- via
            // order_by_sort_i64; multi-key uses the row-major bitonic comparator.
            let perm = if k == 1 {
                device_memory
                    .order_by_sort_i64(&key_matrix, (desc_mask & 1) != 0)
                    .map_err(map_err)?
            } else {
                device_memory
                    .bitonic_sort_multikey(&key_matrix, n, k, desc_mask)
                    .map_err(map_err)?
            };
            perm.iter().map(|&p| indices_u64[p as usize]).collect()
        };
        // LIMIT/OFFSET as control-plane WINDOWING of the device-ordered index vector: slice the surviving
        // indices to the [OFFSET, OFFSET+LIMIT) window BEFORE the column gather, so only the kept rows are
        // materialized from the device (we never gather rows that would then be dropped -- the real win).
        // SQL clause order: ORDER BY (the GPU sort above) -> OFFSET -> LIMIT. Slicing an index vector is
        // control-plane; the relational ordering+windowing decision rode the device sort.
        let indices_u64 = if select.offset.is_some() || select.limit.is_some() {
            let start = select.offset.unwrap_or(0).min(indices_u64.len());
            let end = select
                .limit
                .map_or(indices_u64.len(), |l| start.saturating_add(l).min(indices_u64.len()));
            indices_u64[start..end].to_vec()
        } else {
            indices_u64
        };
        // Materialize: gather each projected column at the surviving row indices on the GPU, by type
        // (int4 -> i32 gather, int8 -> i64 gather; the type matrix, doc 19).
        enum ProjectedColumn {
            Int4(Vec<i32>),
            Int8(Vec<i64>),
            Numeric(Vec<i128>, u8),
            Date(Vec<i32>),
            Timestamp(Vec<i64>),
            // Stored as the i128 the i128 projector returns; the raw 16 uuid bytes are its LE form.
            Uuid(Vec<i128>),
            // Stored as the widened i32s the int4 projector returns; narrowed back to i16 per row.
            Int2(Vec<i32>),
            // Gathered straight from the 1-bit-per-row bool bitmap.
            Bool(Vec<bool>),
            // The text VALUE is gathered ON-DEVICE from the resident payload's offsets+bytes sections
            // (project_text_rows_from_payload) at the GPU-sorted surviving indices -- no host_rows read
            // (the host is control plane only). A NULL/empty cell returns ""; the validity override below
            // restores SqlValue::Null for a NULL row.
            Text(Vec<String>),
        }
        let mut projected_columns: Vec<ProjectedColumn> =
            Vec::with_capacity(bound.selected_indexes.len());
        for &col in &bound.selected_indexes {
            let column = match table.columns[col].ty {
                SqlType::Int8 => {
                    let byte_offset = resident_device_int8_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_i64_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Int8(values)
                }
                SqlType::Numeric { scale, .. } => {
                    let byte_offset = resident_device_numeric_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_i128_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Numeric(values, scale)
                }
                SqlType::Date => {
                    // Date rides the i32 section; project it as i32 then tag it as a date.
                    let byte_offset = resident_device_int4_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_i32_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Date(values)
                }
                SqlType::Int2 => {
                    // Smallint rides the i32 section widened; project as i32, narrow per row below.
                    let byte_offset = resident_device_int4_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_i32_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Int2(values)
                }
                SqlType::Bool => {
                    // Bool is a 1-bit-per-row bitmap; gather the selected rows' bits.
                    let byte_offset = resident_device_bool_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_bool_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Bool(values)
                }
                SqlType::Timestamp => {
                    // Timestamp rides the i64 section; project it as i64 then tag it as a timestamp.
                    let byte_offset = resident_device_int8_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_i64_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Timestamp(values)
                }
                SqlType::Uuid => {
                    // Uuid rides the i128 section; project as i128 (its LE bytes are the raw uuid).
                    let byte_offset = resident_device_numeric_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_i128_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Uuid(values)
                }
                SqlType::Text => {
                    // Gather the text VALUE ON-DEVICE from the resident payload's offsets+bytes sections at
                    // the GPU-sorted surviving indices -- no host_rows read (the host is control plane
                    // only; the GPU did the filter + sort AND now materializes the strings). A NULL/empty
                    // cell returns ""; the projected_validity override below restores SqlValue::Null.
                    let layout = resident_device_text_column_layout(&snapshot, table, col)?;
                    let values = device_memory
                        .project_text_rows_from_payload(
                            layout.offsets_byte_offset,
                            layout.bytes_byte_offset,
                            layout.bytes_len,
                            &indices_u64,
                        )
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Text(values)
                }
                _ => {
                    let byte_offset = resident_device_int4_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_i32_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Int4(values)
                }
            };
            projected_columns.push(column);
        }
        // M3 (doc 21): per projected column, the NULL validity of each surviving row (1 = present). A
        // nullable column's device value is a 0/empty PLACEHOLDER for a NULL row, so the result must emit
        // SqlValue::Null there. The validity bitmap is 1-bit-per-row like a bool column, so the bool
        // projector gathers it at the surviving indices. `None` = the column holds no NULLs (all valid),
        // so non-nullable projections are unchanged. (Text's device gather returns "" for a NULL cell, so
        // this override is what restores its SqlValue::Null.)
        let projected_validity: Vec<Option<Vec<bool>>> = bound
            .selected_indexes
            .iter()
            .map(|&col| -> Result<Option<Vec<bool>>, ExecuteError> {
                match resident_device_null_column_offset(&snapshot, table, col)? {
                    Some(off) => Ok(Some(
                        device_memory
                            .project_bool_rows_from_payload(off, &indices_u64)
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                            })?,
                    )),
                    None => Ok(None),
                }
            })
            .collect::<Result<_, _>>()?;
        let rows: Vec<Vec<SqlValue>> = (0..indices_u64.len())
            .map(|row| {
                projected_columns
                    .iter()
                    .enumerate()
                    .map(|(c, column)| {
                        // A NULL row (validity bit 0) projects as SQL NULL regardless of its placeholder.
                        if let Some(validity) = &projected_validity[c] {
                            if !validity[row] {
                                return SqlValue::Null;
                            }
                        }
                        match column {
                            ProjectedColumn::Int4(values) => SqlValue::Int4(values[row]),
                            ProjectedColumn::Int8(values) => SqlValue::Int8(values[row]),
                            ProjectedColumn::Numeric(values, scale) => {
                                SqlValue::Numeric(Decimal128::new(values[row], *scale))
                            }
                            ProjectedColumn::Date(values) => SqlValue::Date(values[row]),
                            ProjectedColumn::Timestamp(values) => SqlValue::Timestamp(values[row]),
                            ProjectedColumn::Uuid(values) => {
                                SqlValue::Uuid(values[row].to_le_bytes())
                            }
                            // The stored i32 is a widened i16, so the narrowing is exact.
                            ProjectedColumn::Int2(values) => SqlValue::Int2(values[row] as i16),
                            ProjectedColumn::Bool(values) => SqlValue::Bool(values[row]),
                            // Device-gathered String; a NULL row was already handled by the validity
                            // override above, so a bare value here is a real (possibly empty) string.
                            ProjectedColumn::Text(values) => SqlValue::Text(values[row].clone()),
                        }
                    })
                    .collect()
            })
            .collect();
        // OFFSET/LIMIT was already applied as a control-plane window of `indices_u64` above (before the
        // gather), so `rows` is the final windowed result -- no host drain/truncate on result data.
        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows,
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    /// Try to lower a SIMPLE int8 comparison to surviving row indices via the i64 compare kernels
    /// (the type matrix, doc 19). Supported shapes: `int8col <cmp> int4literal` (the int4 literal is
    /// widened to int8 — PG's int4->int8 coercion), `int4literal <cmp> int8col` (operand order
    /// preserved via `scalar_on_left`), and `int8col <cmp> int8col`.
    ///
    /// Returns `Ok(None)` for a pure-int4 predicate (the caller falls through to the int4 path) and
    /// `Err` when int8 appears in a shape not yet supported (int8 arithmetic, or a mixed int4/int8
    /// expression) — never a silent fall-through that would mis-answer.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_int8_predicate(
        &self,
        predicate: &ResidentExpr,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        let lhs_int8 = int8_column_index(lhs, table);
        let rhs_int8 = int8_column_index(rhs, table);
        // Surface the device error verbatim (e.g. int64 overflow -> "bigint out of range"), matching
        // the int4 path's mapping.
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };

        // The simple int8 comparison PEEPHOLE (resident-column compare kernels) — only for an actual
        // comparison op. `predicate_compare_code` is None for AND/OR, so boolean combinators skip this
        // and fall through to the general int8 path below (NOT the int4 path — that was the silent
        // mis-answer the audit caught: int8 AND/OR read with the i32 4-byte stride).
        if let Some(cmp) = predicate_compare_code(compare) {
            match (lhs, rhs) {
                (ResidentExpr::Int4Literal(scalar), _) if rhs_int8.is_some() => {
                    let offset = resident_device_int8_column_offset(
                        snapshot,
                        table,
                        rhs_int8.expect("checked"),
                    )?;
                    return device_memory
                        .expr_i64_compare_scalar_filter(
                            offset,
                            i64::from(*scalar),
                            true,
                            cmp,
                            row_count,
                        )
                        .map(Some)
                        .map_err(map_err);
                }
                (_, ResidentExpr::Int4Literal(scalar)) if lhs_int8.is_some() => {
                    let offset = resident_device_int8_column_offset(
                        snapshot,
                        table,
                        lhs_int8.expect("checked"),
                    )?;
                    return device_memory
                        .expr_i64_compare_scalar_filter(
                            offset,
                            i64::from(*scalar),
                            false,
                            cmp,
                            row_count,
                        )
                        .map(Some)
                        .map_err(map_err);
                }
                _ if lhs_int8.is_some() && rhs_int8.is_some() => {
                    let a_offset = resident_device_int8_column_offset(
                        snapshot,
                        table,
                        lhs_int8.expect("checked"),
                    )?;
                    let b_offset = resident_device_int8_column_offset(
                        snapshot,
                        table,
                        rhs_int8.expect("checked"),
                    )?;
                    return device_memory
                        .expr_i64_compare_columns_filter(a_offset, b_offset, cmp, row_count)
                        .map(Some)
                        .map_err(map_err);
                }
                _ => {}
            }
        }

        // General int8 path: int8 ARITHMETIC / AND-OR / deeper predicates lower via the i64 buffer VM
        // (int4 LITERALS are widened to i64). A pure-int4 predicate returns None (the int4 paths handle
        // it); a MIXED int4/int8 expression is a hard error — never a silent mis-answer.
        let mentions_int8 = expr_mentions_int8(lhs, table) || expr_mentions_int8(rhs, table);
        if !mentions_int8 {
            return Ok(None);
        }
        if expr_mentions_int4_column(lhs, table) || expr_mentions_int4_column(rhs, table) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor does not support mixed int4/int8 expressions yet"
                    .to_string(),
            )));
        }
        // Mixed int8/text or int8/bool (e.g. `bigcol > 5 AND tag = 'a'` / `AND flag`) mixes the i64 VM
        // with i32 text/bool masks -- a follow-on; reject rather than run the untested path (text/bool
        // AND/OR runs under the i32 VM, with int4). int8-only AND/OR stays on this i64 VM.
        if expr_mentions_text(lhs, table)
            || expr_mentions_text(rhs, table)
            || expr_mentions_bool_column(lhs, table)
            || expr_mentions_bool_column(rhs, table)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor does not support mixed int8/text or int8/bool expressions yet"
                    .to_string(),
            )));
        }
        let mut program = Vec::new();
        let mut needles: Vec<Vec<u8>> = Vec::new();
        compile_predicate_program(predicate, table, snapshot, &mut program, &mut needles)?;
        device_memory
            .run_expr_predicate_filter(&program, row_count, ResidentElemType::I64)
            .map(Some)
            .map_err(map_err)
    }

    /// Cross-scale `numcol <cmp> literal` (or flipped): the literal is FINER than the column, so load
    /// the column, rescale it UP to the literal's scale (mantissa * 10^k, the buffer VM path), and
    /// compare to the literal's mantissa. `scalar_on_left` is the literal's side of the comparison.
    #[allow(clippy::too_many_arguments)]
    fn numeric_cross_scale_scalar(
        &self,
        column_offset: u64,
        column_scale: u8,
        literal: Decimal128,
        scalar_on_left: bool,
        cmp: u32,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Vec<u32>, ExecuteError> {
        let needle = i32::try_from(literal.mantissa).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "numeric comparison literal is too large for the fast path yet".to_string(),
            ))
        })?;
        let mut program = vec![ExprStep::LoadColumn {
            byte_offset: column_offset,
        }];
        push_numeric_rescale(column_scale, literal.scale, &mut program)?;
        program.push(ExprStep::CompareScalar {
            cmp,
            scalar: needle,
            scalar_on_left,
        });
        device_memory
            .run_expr_predicate_filter(&program, row_count, ResidentElemType::I128)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    /// Try to lower a SIMPLE date comparison (`datecol <cmp> 'YYYY-MM-DD'`, either operand order, or
    /// `datecol <cmp> datecol`) to surviving row indices via the i32 compare path (the type matrix,
    /// doc 19): a `date` is i32 days since 2000-01-01, so it reuses the int4 residency section + the
    /// I32 VM. The string literal is coerced to a day count (`parse_date`) at lowering, like PG. Returns
    /// None for a non-date predicate. Date `AND`/`OR` / arithmetic, and a date compared to a non-date
    /// value, are hard errors -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_date_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_date(lhs, table) || expr_mentions_date(rhs, table)) {
            return Ok(None);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only simple date comparisons (date AND/OR and \
                 arithmetic are follow-ons)"
                    .to_string(),
            )));
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let run = |program: &[ExprStep]| {
            device_memory
                .run_expr_predicate_filter(program, row_count, ResidentElemType::I32)
                .map(Some)
                .map_err(map_err)
        };
        match (date_column_index(lhs, table), date_column_index(rhs, table)) {
            (Some(col), None) => {
                let days = date_literal_days(rhs)?;
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                run(&[
                    ExprStep::LoadColumn { byte_offset },
                    ExprStep::CompareScalar {
                        cmp,
                        scalar: days,
                        scalar_on_left: false,
                    },
                ])
            }
            (None, Some(col)) => {
                let days = date_literal_days(lhs)?;
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                run(&[
                    ExprStep::LoadColumn { byte_offset },
                    ExprStep::CompareScalar {
                        cmp,
                        scalar: days,
                        scalar_on_left: true,
                    },
                ])
            }
            (Some(a), Some(b)) => {
                let a_offset = resident_device_int4_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_int4_column_offset(snapshot, table, b)?;
                run(&[
                    ExprStep::LoadColumn {
                        byte_offset: a_offset,
                    },
                    ExprStep::LoadColumn {
                        byte_offset: b_offset,
                    },
                    ExprStep::CompareBuffers { cmp },
                ])
            }
            (None, None) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a date predicate must involve a date column".to_string(),
            ))),
        }
    }

    /// Try to lower a SIMPLE timestamp comparison (`ts_col <cmp> 'YYYY-MM-DD HH:MM:SS'`, either order,
    /// or `ts_col <cmp> ts_col`) to surviving row indices via the i64 compare KERNELS (the type matrix,
    /// doc 19): a `timestamp` is i64 microseconds, so it reuses the int8 residency section + the i64
    /// compare kernels. (The i64 microsecond literal exceeds the i32 `ExprStep` scalar, so this uses
    /// `expr_i64_compare_scalar_filter` directly, not the VM's i32-scalar step.) Returns None for a
    /// non-timestamp predicate. Timestamp `AND`/`OR` / arithmetic, and a timestamp compared to a
    /// non-timestamp value, are hard errors -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_timestamp_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_timestamp(lhs, table) || expr_mentions_timestamp(rhs, table)) {
            return Ok(None);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only simple timestamp comparisons (timestamp \
                 AND/OR and arithmetic are follow-ons)"
                    .to_string(),
            )));
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        match (
            timestamp_column_index(lhs, table),
            timestamp_column_index(rhs, table),
        ) {
            (Some(col), None) => {
                let micros = timestamp_literal_micros(rhs)?;
                let offset = resident_device_int8_column_offset(snapshot, table, col)?;
                device_memory
                    .expr_i64_compare_scalar_filter(offset, micros, false, cmp, row_count)
                    .map(Some)
                    .map_err(map_err)
            }
            (None, Some(col)) => {
                let micros = timestamp_literal_micros(lhs)?;
                let offset = resident_device_int8_column_offset(snapshot, table, col)?;
                device_memory
                    .expr_i64_compare_scalar_filter(offset, micros, true, cmp, row_count)
                    .map(Some)
                    .map_err(map_err)
            }
            (Some(a), Some(b)) => {
                let a_offset = resident_device_int8_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_int8_column_offset(snapshot, table, b)?;
                device_memory
                    .expr_i64_compare_columns_filter(a_offset, b_offset, cmp, row_count)
                    .map(Some)
                    .map_err(map_err)
            }
            (None, None) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a timestamp predicate must involve a timestamp column".to_string(),
            ))),
        }
    }

    /// M3 (doc 21) WHERE 3VL for a SIMPLE comparison over a nullable DATE or TIMESTAMP column (scalar,
    /// either operand order, or col-vs-col). The generic mask VM (`predicate_vm_elem_type`) cannot lower
    /// these: a date/timestamp literal is not an `Int4Literal`, and a timestamp's i64 microseconds exceed
    /// the VM's i32 `CompareScalar` scalar. So build the VM program directly — DATE: I32 + `CompareScalar`
    /// over the i32 days; TIMESTAMP: I64 + `CompareScalarI64` over the i64 micros; col-vs-col:
    /// `CompareBuffers` — then append the validity-AND (`push_column_validity_and` = `BoolMask` +
    /// `MaskBinary` AND) for each nullable operand, so a NULL operand is UNKNOWN ⇒ the row is excluded
    /// (correct at the top level / under no-NOT). Returns `None` for a non-date/timestamp predicate or a
    /// shape the peephole doesn't support (AND/OR / arithmetic), so the caller clean-errors. NO kernel
    /// change: the i64 compare-scalar / compare-buffers / bool-to-mask / mask-binary kernels already exist.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_nullable_temporal_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        // Only a simple comparison (eq/ne/lt/le/gt/ge); AND/OR yield None -> caller clean-errors.
        let Some(cmp) = predicate_compare_code(compare) else {
            return Ok(None);
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        // DATE: i32 days in the int4 section -> the I32 VM (CompareScalar fits the i32 days literal).
        if expr_mentions_date(lhs, table) || expr_mentions_date(rhs, table) {
            let mut program = Vec::new();
            match (date_column_index(lhs, table), date_column_index(rhs, table)) {
                (Some(col), None) => {
                    let days = date_literal_days(rhs)?;
                    let off = resident_device_int4_column_offset(snapshot, table, col)?;
                    program.push(ExprStep::LoadColumn { byte_offset: off });
                    program.push(ExprStep::CompareScalar { cmp, scalar: days, scalar_on_left: false });
                    push_column_validity_and(col, table, snapshot, &mut program)?;
                }
                (None, Some(col)) => {
                    let days = date_literal_days(lhs)?;
                    let off = resident_device_int4_column_offset(snapshot, table, col)?;
                    program.push(ExprStep::LoadColumn { byte_offset: off });
                    program.push(ExprStep::CompareScalar { cmp, scalar: days, scalar_on_left: true });
                    push_column_validity_and(col, table, snapshot, &mut program)?;
                }
                (Some(a), Some(b)) => {
                    let a_off = resident_device_int4_column_offset(snapshot, table, a)?;
                    let b_off = resident_device_int4_column_offset(snapshot, table, b)?;
                    program.push(ExprStep::LoadColumn { byte_offset: a_off });
                    program.push(ExprStep::LoadColumn { byte_offset: b_off });
                    program.push(ExprStep::CompareBuffers { cmp });
                    push_column_validity_and(a, table, snapshot, &mut program)?;
                    push_column_validity_and(b, table, snapshot, &mut program)?;
                }
                (None, None) => return Ok(None),
            }
            return device_memory
                .run_expr_predicate_filter(&program, row_count, ResidentElemType::I32)
                .map(Some)
                .map_err(map_err);
        }
        // TIMESTAMP: i64 micros in the int8 section -> the I64 VM with CompareScalarI64 (the micros
        // literal exceeds the i32 CompareScalar scalar).
        if expr_mentions_timestamp(lhs, table) || expr_mentions_timestamp(rhs, table) {
            let mut program = Vec::new();
            match (timestamp_column_index(lhs, table), timestamp_column_index(rhs, table)) {
                (Some(col), None) => {
                    let micros = timestamp_literal_micros(rhs)?;
                    let off = resident_device_int8_column_offset(snapshot, table, col)?;
                    program.push(ExprStep::LoadColumn { byte_offset: off });
                    program.push(ExprStep::CompareScalarI64 { cmp, scalar: micros, scalar_on_left: false });
                    push_column_validity_and(col, table, snapshot, &mut program)?;
                }
                (None, Some(col)) => {
                    let micros = timestamp_literal_micros(lhs)?;
                    let off = resident_device_int8_column_offset(snapshot, table, col)?;
                    program.push(ExprStep::LoadColumn { byte_offset: off });
                    program.push(ExprStep::CompareScalarI64 { cmp, scalar: micros, scalar_on_left: true });
                    push_column_validity_and(col, table, snapshot, &mut program)?;
                }
                (Some(a), Some(b)) => {
                    let a_off = resident_device_int8_column_offset(snapshot, table, a)?;
                    let b_off = resident_device_int8_column_offset(snapshot, table, b)?;
                    program.push(ExprStep::LoadColumn { byte_offset: a_off });
                    program.push(ExprStep::LoadColumn { byte_offset: b_off });
                    program.push(ExprStep::CompareBuffers { cmp });
                    push_column_validity_and(a, table, snapshot, &mut program)?;
                    push_column_validity_and(b, table, snapshot, &mut program)?;
                }
                (None, None) => return Ok(None),
            }
            return device_memory
                .run_expr_predicate_filter(&program, row_count, ResidentElemType::I64)
                .map(Some)
                .map_err(map_err);
        }
        Ok(None)
    }

    /// M3 (doc 21) WHERE 3VL over a nullable NUMERIC column. The generic mask VM can't lower numeric (a
    /// numeric literal is not an Int4Literal; the i128 mantissa exceeds the i32 CompareScalar). Two routes,
    /// both on the I128 VM, each AND'ing the operand columns' validity (a NULL operand is UNKNOWN, excluded
    /// — correct for WHERE under AND/OR since there is no NOT). (a) A SIMPLE same-or-coarser-scale scalar /
    /// same-scale col-vs-col uses a direct `CompareScalarI128` / `CompareBuffers` program (handles a LARGE
    /// mantissa the i32-needle path can't). (b) AND/OR, numeric arithmetic, a FINER cross-scale literal, or
    /// a different-scale col-vs-col uses the shared `compile_numeric_predicate_program` /
    /// `compile_numeric_compare` VM path, now validity-aware (`push_leaf_validity_and` per leaf; same
    /// i32-needle limit as the non-null path). Returns `None` only for a non-numeric predicate; a mixed
    /// numeric/integer predicate is a clean error. NO kernel change.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_nullable_numeric_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_numeric(lhs, table) || expr_mentions_numeric(rhs, table)) {
            return Ok(None);
        }
        // Mixed numeric/integer is a clean error (compile_numeric_arith expects numeric operands) — mirror
        // the non-null path's guard so the message is clear rather than an opaque compile failure.
        if expr_mentions_int4_column(lhs, table)
            || expr_mentions_int4_column(rhs, table)
            || expr_mentions_int8(lhs, table)
            || expr_mentions_int8(rhs, table)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor does not support mixed numeric/integer expressions yet"
                    .to_string(),
            )));
        }
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let run = |program: &[ExprStep]| {
            device_memory
                .run_expr_predicate_filter(program, row_count, ResidentElemType::I128)
                .map(Some)
                .map_err(map_err)
        };
        // numeric AND/OR -> the validity-aware compile path (each comparison leaf via compile_numeric_compare,
        // which now AND's the operand validity; MaskBinary combines).
        if let Some(bool_op) = boolean_op_code(compare) {
            let mut program = Vec::new();
            compile_numeric_predicate_program(lhs, table, snapshot, &mut program)?;
            compile_numeric_predicate_program(rhs, table, snapshot, &mut program)?;
            program.push(ExprStep::MaskBinary { op: bool_op });
            return run(&program);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            return Ok(None);
        };
        // SIMPLE same-or-coarser-scale scalar / same-scale col-vs-col -> the CompareScalarI128 fast path
        // (handles a large mantissa). `None` here means "not this fast shape" -> fall to compile_numeric_compare.
        let scalar_program = |col: usize,
                              literal: Decimal128,
                              scalar_on_left: bool|
         -> Result<Option<Vec<ExprStep>>, ExecuteError> {
            let col_scale = column_numeric_scale(table, col).expect("numeric column has a scale");
            let literal = literal.canonical();
            if literal.scale > col_scale {
                return Ok(None); // finer cross-scale literal -> the compile_numeric_compare fallback
            }
            let mantissa = rescale_numeric_literal(literal, col_scale)?;
            let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
            let mut program = vec![
                ExprStep::LoadColumn { byte_offset: offset },
                ExprStep::CompareScalarI128 { cmp, scalar: mantissa, scalar_on_left },
            ];
            push_column_validity_and(col, table, snapshot, &mut program)?;
            Ok(Some(program))
        };
        let fast: Option<Vec<ExprStep>> = match (
            numeric_column_index(lhs, table),
            numeric_column_index(rhs, table),
        ) {
            (Some(col), None) if numeric_literal_value(rhs).is_some() => {
                scalar_program(col, numeric_literal_value(rhs).expect("checked"), false)?
            }
            (None, Some(col)) if numeric_literal_value(lhs).is_some() => {
                scalar_program(col, numeric_literal_value(lhs).expect("checked"), true)?
            }
            (Some(a), Some(b))
                if column_numeric_scale(table, a) == column_numeric_scale(table, b) =>
            {
                let a_offset = resident_device_numeric_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_numeric_column_offset(snapshot, table, b)?;
                let mut program = vec![
                    ExprStep::LoadColumn { byte_offset: a_offset },
                    ExprStep::LoadColumn { byte_offset: b_offset },
                    ExprStep::CompareBuffers { cmp },
                ];
                push_column_validity_and(a, table, snapshot, &mut program)?;
                push_column_validity_and(b, table, snapshot, &mut program)?;
                Some(program)
            }
            _ => None,
        };
        if let Some(program) = fast {
            return run(&program);
        }
        // Fallback: a FINER cross-scale literal / different-scale col-vs-col / numeric arithmetic -> the
        // validity-aware shared compile path (compile_numeric_compare appends push_leaf_validity_and).
        let mut program = Vec::new();
        compile_numeric_compare(lhs, rhs, cmp, table, snapshot, &mut program)?;
        run(&program)
    }

    /// Try to lower a SIMPLE uuid comparison (`id <cmp> 'uuid-literal'`, either operand order, or
    /// `id <cmp> id2`) to surviving row indices via the byte-wise uuid compare kernel (the type matrix,
    /// doc 19): a `uuid` is 16 raw bytes in the i128 section; PG compares uuids by an unsigned
    /// big-endian memcmp. The string literal is parsed to 16 bytes at lowering. Returns None for a
    /// non-uuid predicate. Uuid `AND`/`OR`, and a uuid compared to a non-uuid value, are hard errors.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_uuid_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_uuid(lhs, table) || expr_mentions_uuid(rhs, table)) {
            return Ok(None);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only simple uuid comparisons (uuid AND/OR is a \
                 follow-on)"
                    .to_string(),
            )));
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        // M3 (doc 21): a NULLABLE uuid operand's validity bitmap offset (None when the column holds no
        // NULL); the launcher AND's it with the compare mask so a NULL operand is UNKNOWN ⇒ excluded. A
        // non-nullable column contributes nothing (the no-NULL path stays byte-identical).
        let validity = |col: usize| -> Result<Vec<u64>, ExecuteError> {
            Ok(resident_device_null_column_offset(snapshot, table, col)?
                .into_iter()
                .collect())
        };
        match (uuid_column_index(lhs, table), uuid_column_index(rhs, table)) {
            (Some(col), None) => {
                let needle = uuid_literal_bytes(rhs)?;
                let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                device_memory
                    .expr_uuid_compare_scalar_filter(offset, &needle, false, cmp, row_count, &validity(col)?)
                    .map(Some)
                    .map_err(map_err)
            }
            (None, Some(col)) => {
                let needle = uuid_literal_bytes(lhs)?;
                let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                device_memory
                    .expr_uuid_compare_scalar_filter(offset, &needle, true, cmp, row_count, &validity(col)?)
                    .map(Some)
                    .map_err(map_err)
            }
            (Some(a), Some(b)) => {
                let a_offset = resident_device_numeric_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_numeric_column_offset(snapshot, table, b)?;
                let mut validity_offsets = validity(a)?;
                validity_offsets.extend(validity(b)?);
                device_memory
                    .expr_uuid_compare_columns_filter(a_offset, b_offset, cmp, row_count, &validity_offsets)
                    .map(Some)
                    .map_err(map_err)
            }
            (None, None) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a uuid predicate must involve a uuid column".to_string(),
            ))),
        }
    }

    /// Try to lower a SIMPLE smallint comparison (`sz <cmp> 5`, either operand order, or `sz <cmp>
    /// sz2`) to surviving row indices via the i32 compare path (the type matrix, doc 19): a `smallint`
    /// is stored widened to i32 in the int4 section, so it REUSES the int4 compare VM with the literal
    /// taken as an i32 scalar. Returns None for a non-smallint predicate. Smallint `AND`/`OR` /
    /// arithmetic (int16-bounds overflow is a follow-on), and a smallint compared to a non-integer
    /// value, are hard errors -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_int2_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_int2(lhs, table) || expr_mentions_int2(rhs, table)) {
            return Ok(None);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only simple smallint comparisons (smallint \
                 AND/OR and arithmetic are follow-ons)"
                    .to_string(),
            )));
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let run = |program: &[ExprStep]| {
            device_memory
                .run_expr_predicate_filter(program, row_count, ResidentElemType::I32)
                .map(Some)
                .map_err(map_err)
        };
        match (int2_column_index(lhs, table), int2_column_index(rhs, table)) {
            (Some(col), None) => {
                let scalar = int2_literal_value(rhs)?;
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                run(&[
                    ExprStep::LoadColumn { byte_offset },
                    ExprStep::CompareScalar {
                        cmp,
                        scalar,
                        scalar_on_left: false,
                    },
                ])
            }
            (None, Some(col)) => {
                let scalar = int2_literal_value(lhs)?;
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                run(&[
                    ExprStep::LoadColumn { byte_offset },
                    ExprStep::CompareScalar {
                        cmp,
                        scalar,
                        scalar_on_left: true,
                    },
                ])
            }
            (Some(a), Some(b)) => {
                let a_offset = resident_device_int4_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_int4_column_offset(snapshot, table, b)?;
                run(&[
                    ExprStep::LoadColumn {
                        byte_offset: a_offset,
                    },
                    ExprStep::LoadColumn {
                        byte_offset: b_offset,
                    },
                    ExprStep::CompareBuffers { cmp },
                ])
            }
            (None, None) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a smallint predicate must involve a smallint column".to_string(),
            ))),
        }
    }

    /// Try to lower a SIMPLE text comparison (`textcol = 'lit'` / `textcol <> 'lit'`, either operand
    /// order) to surviving row indices via the byte-wise text-equality kernel (the type matrix, doc
    /// 19). Equality is byte identity -- PG deterministic-collation semantics. Returns None for a
    /// non-text predicate (the other type paths handle it). Text INEQUALITIES (need collation sort
    /// keys), `LIKE`, text `AND`/`OR`, text column-vs-column, and text mixed with another type are hard
    /// errors -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_text_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_text(lhs, table) || expr_mentions_text(rhs, table)) {
            return Ok(None);
        }
        // AND/OR is not a single comparison: fall through (Ok(None)) so the mask VM compiles each leaf
        // (text -> TextEqMask, int4 -> arith+compare) and combines them. A text+int4 AND/OR is VALID
        // even though mixing the two types within ONE comparison (the guard below) is not. Mixed
        // text+int8/numeric AND/OR is rejected later (the int8/numeric paths) -- the i32 mask VM only.
        if matches!(compare, ResidentBinaryOp::And | ResidentBinaryOp::Or) {
            return Ok(None);
        }
        if expr_mentions_int4_column(lhs, table)
            || expr_mentions_int4_column(rhs, table)
            || expr_mentions_int8(lhs, table)
            || expr_mentions_int8(rhs, table)
            || expr_mentions_numeric(lhs, table)
            || expr_mentions_numeric(rhs, table)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor does not support mixed text/non-text expressions"
                    .to_string(),
            )));
        }
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        // textcol LIKE 'pattern' (the pattern is on the right; LIKE is NOT symmetric). The host
        // compiles the pattern (resolving `\` escapes) to the kernel's u32 token array.
        if matches!(compare, ResidentBinaryOp::Like) {
            let (col, pattern) = match (text_column_index(lhs, table), text_literal_value(rhs)) {
                (Some(col), Some(pattern)) => (col, pattern),
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "LIKE must be a text column LIKE a literal pattern".to_string(),
                    )));
                }
            };
            let tokens = compile_like_pattern(pattern)?;
            let layout = resident_device_text_column_layout(snapshot, table, col)?;
            return device_memory
                .expr_text_like_scalar_filter(
                    layout.offsets_byte_offset,
                    layout.bytes_byte_offset,
                    &tokens,
                    row_count,
                )
                .map(Some)
                .map_err(map_err);
        }
        let negate = match compare {
            ResidentBinaryOp::Eq => false,
            ResidentBinaryOp::Ne => true,
            ResidentBinaryOp::Lt
            | ResidentBinaryOp::Le
            | ResidentBinaryOp::Gt
            | ResidentBinaryOp::Ge => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "text inequalities need collation sort keys (a follow-on); only = and <> run \
                     on the GPU"
                        .to_string(),
                )));
            }
            // AND/OR returned early (above) -> the mask VM. Any other op here is unsupported.
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "the general GPU executor supports text = and <> only (LIKE is a follow-on)"
                        .to_string(),
                )));
            }
        };
        // textcol <eq/ne> 'literal' -- equality is symmetric, so operand order does not matter.
        let (col, literal) = match (text_column_index(lhs, table), text_column_index(rhs, table)) {
            (Some(col), None) if text_literal_value(rhs).is_some() => {
                (col, text_literal_value(rhs).expect("checked"))
            }
            (None, Some(col)) if text_literal_value(lhs).is_some() => {
                (col, text_literal_value(lhs).expect("checked"))
            }
            (Some(_), Some(_)) => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "text column-vs-column comparison is a follow-on".to_string(),
                )));
            }
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "text comparison must be a column against a literal".to_string(),
                )));
            }
        };
        let layout = resident_device_text_column_layout(snapshot, table, col)?;
        device_memory
            .expr_text_eq_scalar_filter(
                layout.offsets_byte_offset,
                layout.bytes_byte_offset,
                literal.as_bytes(),
                negate,
                row_count,
            )
            .map(Some)
            .map_err(map_err)
    }

    /// Try to lower a SIMPLE numeric comparison to surviving row indices via the i128 compare kernels
    /// (the type matrix, doc 19). Supported: `numcol <cmp> literal` / `literal <cmp> numcol` and
    /// `numcol <cmp> numcol`, at ANY scale -- same-scale (and coarser-or-equal literal) use the
    /// resident compare peephole; a SCALE MISMATCH (a finer literal, or two columns of different scale)
    /// loads to i128 buffers, rescales the coarser side UP to the common = max scale (mantissa * 10^k,
    /// k <= 9; a wider gap or a rescale overflow errors), then compares. Numeric ARITHMETIC comparisons
    /// and `AND`/`OR` of numeric comparisons also lower here (each comparison -> a mask via the i128
    /// VM, MaskBinary combines). Returns None for a non-numeric predicate (the int4/int8 paths handle
    /// it). A numeric value mixed with an int4/int8 COLUMN is a hard error -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_numeric_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_numeric(lhs, table) || expr_mentions_numeric(rhs, table)) {
            return Ok(None);
        }
        if expr_mentions_int4_column(lhs, table)
            || expr_mentions_int4_column(rhs, table)
            || expr_mentions_int8(lhs, table)
            || expr_mentions_int8(rhs, table)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor does not support mixed numeric/integer expressions yet"
                    .to_string(),
            )));
        }
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        // numeric AND/OR: compile each side's comparison(s) to a MASK (the buffer path), combine with
        // MaskBinary, and compact -- the same i128 mask VM the single-comparison arithmetic path uses.
        if let Some(bool_op) = boolean_op_code(compare) {
            let mut program = Vec::new();
            compile_numeric_predicate_program(lhs, table, snapshot, &mut program)?;
            compile_numeric_predicate_program(rhs, table, snapshot, &mut program)?;
            program.push(ExprStep::MaskBinary { op: bool_op });
            return device_memory
                .run_expr_predicate_filter(&program, row_count, ResidentElemType::I128)
                .map(Some)
                .map_err(map_err);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only numeric comparisons and AND/OR".to_string(),
            )));
        };
        match (
            numeric_column_index(lhs, table),
            numeric_column_index(rhs, table),
        ) {
            (Some(col), None) if numeric_literal_value(rhs).is_some() => {
                let col_scale = column_numeric_scale(table, col).expect("numeric column has a scale");
                let literal = numeric_literal_value(rhs).expect("checked").canonical();
                let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                if literal.scale <= col_scale {
                    // literal coarser-or-equal: rescale it UP to the column scale; resident peephole.
                    let mantissa = rescale_numeric_literal(literal, col_scale)?;
                    device_memory
                        .expr_i128_compare_scalar_filter(offset, mantissa, false, cmp, row_count)
                        .map(Some)
                        .map_err(map_err)
                } else {
                    // CROSS-SCALE: the literal is FINER than the column -> rescale the column UP to the
                    // literal's scale (buffer VM path) and compare to the literal's mantissa.
                    self.numeric_cross_scale_scalar(
                        offset,
                        col_scale,
                        literal,
                        false,
                        cmp,
                        device_memory,
                        row_count,
                    )
                    .map(Some)
                }
            }
            (None, Some(col)) if numeric_literal_value(lhs).is_some() => {
                let col_scale = column_numeric_scale(table, col).expect("numeric column has a scale");
                let literal = numeric_literal_value(lhs).expect("checked").canonical();
                let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                if literal.scale <= col_scale {
                    let mantissa = rescale_numeric_literal(literal, col_scale)?;
                    device_memory
                        .expr_i128_compare_scalar_filter(offset, mantissa, true, cmp, row_count)
                        .map(Some)
                        .map_err(map_err)
                } else {
                    self.numeric_cross_scale_scalar(
                        offset, col_scale, literal, true, cmp, device_memory, row_count,
                    )
                    .map(Some)
                }
            }
            (Some(a), Some(b)) => {
                let a_scale = column_numeric_scale(table, a).expect("numeric column has a scale");
                let b_scale = column_numeric_scale(table, b).expect("numeric column has a scale");
                let a_offset = resident_device_numeric_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_numeric_column_offset(snapshot, table, b)?;
                if a_scale == b_scale {
                    device_memory
                        .expr_i128_compare_columns_filter(a_offset, b_offset, cmp, row_count)
                        .map(Some)
                        .map_err(map_err)
                } else {
                    // CROSS-SCALE: load both, rescale the coarser column UP to the common (max) scale,
                    // then compare buffers (the buffer VM path).
                    let common = a_scale.max(b_scale);
                    let mut program = vec![ExprStep::LoadColumn {
                        byte_offset: a_offset,
                    }];
                    push_numeric_rescale(a_scale, common, &mut program)?;
                    program.push(ExprStep::LoadColumn {
                        byte_offset: b_offset,
                    });
                    push_numeric_rescale(b_scale, common, &mut program)?;
                    program.push(ExprStep::CompareBuffers { cmp });
                    device_memory
                        .run_expr_predicate_filter(&program, row_count, ResidentElemType::I128)
                        .map(Some)
                        .map_err(map_err)
                }
            }
            _ => {
                // Numeric ARITHMETIC comparison (`price * tax > 100`, `100 < price - fee`, ...):
                // compile both sides to a mask via the shared comparison core (each side's scale is
                // computed bottom-up and the coarser brought up to the common scale), then run the VM.
                let mut program = Vec::new();
                compile_numeric_compare(lhs, rhs, cmp, table, snapshot, &mut program)?;
                device_memory
                    .run_expr_predicate_filter(&program, row_count, ResidentElemType::I128)
                    .map(Some)
                    .map_err(map_err)
            }
        }
    }

    /// Lower a predicate [`ResidentExpr`] to the surviving row indices, evaluated on the GPU.
    ///
    /// Coverage (grows by extending this method, per the design): the prototype shape
    /// `Compare(arith(Column a, Column b), Int4Literal k)` lowers to the composed buffer->buffer
    /// primitive `expr_filter_two_col_compare_from_payload` (elementwise `a <arith> b` into an
    /// intermediate device buffer, then compare-to-row-indices). Anything else is rejected with a
    /// pointer to the device bytecode VM that generalizes it — NOT by falling back to a shape method
    /// or to the CPU.
    /// Lower a bare bool-column predicate (`WHERE flag`) to surviving row indices (the type matrix,
    /// doc 19): the bool column's 1-bit-per-row bitmap expands directly to the row mask. A bare
    /// non-bool column is invalid SQL (PG: "argument of WHERE must be type boolean") -> hard error.
    fn lower_bool_column_predicate(
        &self,
        col: usize,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Vec<u32>, ExecuteError> {
        let column = table.columns.get(col).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident Expr predicate column is outside the catalog table".to_string(),
            ))
        })?;
        if column.ty != SqlType::Bool {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a bare column predicate is valid only for a bool column (argument of WHERE must be \
                 type boolean)"
                    .to_string(),
            )));
        }
        let offset = resident_device_bool_column_offset(snapshot, table, col)?;
        device_memory
            .expr_bool_to_mask_filter(offset, false, row_count)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    /// Lower a standalone `col IS NULL` / `IS NOT NULL` to surviving row indices (M3 -- doc 21). The
    /// column's NULL validity bitmap (1 = valid/present) feeds the same bitmap->mask kernel as a bool
    /// column: `IS NOT NULL` reads it as-is, `IS NULL` complements it (`negate = !is_not_null`). A column
    /// with NO validity bitmap holds no NULLs, so every row is valid -- IS NOT NULL is all rows, IS NULL
    /// none -- returned directly without a kernel.
    fn lower_is_null_predicate(
        &self,
        col: usize,
        is_not_null: bool,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Vec<u32>, ExecuteError> {
        match resident_device_null_column_offset(snapshot, table, col)? {
            Some(offset) => device_memory
                .expr_bool_to_mask_filter(offset, !is_not_null, row_count)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))),
            None if is_not_null => {
                let n = u32::try_from(row_count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident row count exceeds u32".to_string(),
                    ))
                })?;
                Ok((0..n).collect())
            }
            None => Ok(Vec::new()),
        }
    }

    /// Try to lower `bool_col = true` / `bool_col = false` (and `<>`, either operand order) to
    /// surviving row indices via the bitmap->mask kernel (the type matrix, doc 19): `= true` selects
    /// the set bits, `= false` (and `NOT flag`, which the mapper rewrites to `= false`) the clear bits
    /// (negate). Returns None if not a `bool_col <op> bool_literal` shape. Ordering ops (`< > ...`) and
    /// bool in AND/OR are hard errors (follow-ons) so the executor never mis-answers.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_bool_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        let is_bool_col = |idx: usize| {
            table.columns.get(idx).map(|column| column.ty) == Some(SqlType::Bool)
        };
        let (col, literal) = match (lhs, rhs) {
            (ResidentExpr::Column(col), ResidentExpr::BoolLiteral(b)) if is_bool_col(*col) => {
                (*col, *b)
            }
            (ResidentExpr::BoolLiteral(b), ResidentExpr::Column(col)) if is_bool_col(*col) => {
                (*col, *b)
            }
            _ => return Ok(None),
        };
        // `flag = true` -> set bits (negate=false); `flag = false` -> clear bits (negate=true).
        // `<>` is the complement.
        let negate = match compare {
            ResidentBinaryOp::Eq => !literal,
            ResidentBinaryOp::Ne => literal,
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "the general GPU executor supports only = / <> against a bool literal".to_string(),
                )));
            }
        };
        let offset = resident_device_bool_column_offset(snapshot, table, col)?;
        device_memory
            .expr_bool_to_mask_filter(offset, negate, row_count)
            .map(Some)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    fn lower_resident_predicate(
        &self,
        predicate: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Vec<u32>, ExecuteError> {
        // bool-predicate (type matrix, doc 19): a bare `WHERE flag` is a bool COLUMN used directly as
        // a predicate -- a bool column is a 1-bit-per-row bitmap, so it expands straight to the row
        // mask (true rows). A bare NON-bool column is invalid SQL (PG: "argument of WHERE must be type
        // boolean"), so it hard-errors rather than mis-answering.
        if let ResidentExpr::Column(col) = predicate {
            return self.lower_bool_column_predicate(*col, table, snapshot, device_memory, row_count);
        }

        // A standalone `col IS NULL` / `IS NOT NULL` (M3 -- doc 19/21): its NULL validity bitmap straight
        // to the row mask via the SAME bitmap->mask kernel as a bool column, pointed at the validity
        // bitmap. (Inside AND/OR it instead rides the general mask VM via `compile_predicate_program`.)
        if let ResidentExpr::IsNull { col, is_not_null } = predicate {
            return self.lower_is_null_predicate(
                *col,
                *is_not_null,
                table,
                snapshot,
                device_memory,
                row_count,
            );
        }

        // M3 (doc 21) WHERE 3VL: a predicate over a NULLABLE column routes through the general mask VM
        // (`compile_predicate_program`), which AND's each comparison leaf with the column's validity mask
        // so a NULL operand evaluates to UNKNOWN (the row is excluded). The typed peephole kernels below
        // read the column's placeholder bytes (0 / empty) and would mis-select NULL rows. The VM lowers an
        // int4/text/bool predicate (elem I32) or an int8 predicate (elem I64, the same i64 buffer VM the
        // non-null int8 AND/OR path uses — the validity AND is a type-independent i32 mask); a nullable
        // numeric/uuid/date/timestamp/int2 or MIXED-width predicate is a clean-error follow-up (no silent
        // mis-answer). A no-NULL table never enters here, so existing (non-nullable) plans keep their
        // peephole fast paths byte-identically.
        if predicate_references_nullable_column(predicate, table, snapshot)? {
            if let Some(elem) = predicate_vm_elem_type(predicate, table) {
                let mut program = Vec::new();
                let mut needles: Vec<Vec<u8>> = Vec::new();
                compile_predicate_program(predicate, table, snapshot, &mut program, &mut needles)?;
                return device_memory
                    .run_expr_predicate_filter_with_text(&program, &needles, row_count, elem)
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())));
            }
            // A nullable DATE/TIMESTAMP/NUMERIC simple comparison: the generic VM can't lower the literal
            // (temporal/numeric), so build the program directly (date I32 / timestamp+numeric I128/I64)
            // with the validity-AND appended.
            if let ResidentExpr::Binary { op, lhs, rhs } = predicate {
                if let Some(indices) = self.try_lower_nullable_temporal_predicate(
                    *op,
                    lhs,
                    rhs,
                    table,
                    snapshot,
                    device_memory,
                    row_count,
                )? {
                    return Ok(indices);
                }
                if let Some(indices) = self.try_lower_nullable_numeric_predicate(
                    *op,
                    lhs,
                    rhs,
                    table,
                    snapshot,
                    device_memory,
                    row_count,
                )? {
                    return Ok(indices);
                }
                // A nullable uuid SIMPLE comparison: the uuid memcmp peephole resolves the operand's
                // validity bitmap and AND's it with the compare mask (uuid has no VM step). A uuid AND/OR
                // is its own clean error from inside the helper.
                if expr_mentions_uuid(lhs, table) || expr_mentions_uuid(rhs, table) {
                    if let Some(indices) = self.try_lower_uuid_predicate(
                        *op,
                        lhs,
                        rhs,
                        table,
                        snapshot,
                        device_memory,
                        row_count,
                    )? {
                        return Ok(indices);
                    }
                }
            }
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "WHERE over a nullable column of this type/shape is not yet supported on the GPU \
                 (M3 3VL follow-up: e.g. a cross-scale or arithmetic numeric, a compound or \
                 mixed-type temporal/numeric/uuid predicate)"
                    .to_string(),
            )));
        }

        let ResidentExpr::Binary {
            op: compare,
            lhs,
            rhs,
        } = predicate
        else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident Expr predicate must be a top-level comparison or AND/OR".to_string(),
            )));
        };

        // bool path (type matrix, doc 19): `flag = true` / `flag = false` / `<>` (and `NOT flag`,
        // which the mapper rewrites to `flag = false`) expand the bool bitmap to the mask via negate.
        // None for a non-bool-literal predicate; checked first since a BoolLiteral has no other home.
        if let Some(indices) =
            self.try_lower_bool_predicate(*compare, lhs, rhs, table, snapshot, device_memory, row_count)?
        {
            return Ok(indices);
        }

        // int8 path (type matrix, doc 19): a SIMPLE int8 comparison (`int8col <cmp> literal` /
        // `literal <cmp> int8col` / `int8col <cmp> int8col`) evaluates via the i64 compare kernels,
        // before the int4 paths below. `try_lower_int8_comparison` returns None for a pure-int4
        // predicate (fall through to the int4 path) and errors for an int8 shape not yet supported
        // (int8 arithmetic / mixed int4-int8) so the executor never silently mis-answers.
        if let Some(indices) = self.try_lower_int8_predicate(
            predicate,
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // numeric path (type matrix, doc 19): a SIMPLE numeric comparison (`numcol <cmp> literal` /
        // `literal <cmp> numcol` / equal-scale `numcol <cmp> numcol`) evaluates via the i128 compare
        // kernels. Returns None for a non-numeric predicate (fall through to int4); errors on a numeric
        // shape not yet supported (numeric arithmetic / AND-OR / mixed) so it never mis-answers.
        if let Some(indices) = self.try_lower_numeric_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // date path (type matrix, doc 19): a SIMPLE date comparison (`hire_date <cmp> '2024-01-15'`)
        // evaluates via the i32 compare path (a date is i32 days). BEFORE the text path, because the
        // date literal is a string literal the text path would otherwise grab. None for a non-date
        // predicate; errors on date AND/OR / arithmetic / mixed.
        if let Some(indices) = self.try_lower_date_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // timestamp path (type matrix, doc 19): a SIMPLE timestamp comparison (i64 microseconds ->
        // the int8 compare kernels). BEFORE the text path (the literal is a string). None for a
        // non-timestamp predicate; errors on timestamp AND/OR / arithmetic / mixed.
        if let Some(indices) = self.try_lower_timestamp_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // uuid path (type matrix, doc 19): a SIMPLE uuid comparison via the byte-wise compare kernel.
        // BEFORE the text path (the literal is a string). None for a non-uuid predicate; errors on
        // uuid AND/OR / mixed.
        if let Some(indices) = self.try_lower_uuid_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // int2 path (type matrix, doc 19): a SIMPLE smallint comparison via the i32 compare VM
        // (a smallint is stored widened to i32). None for a non-smallint predicate; errors on
        // smallint AND/OR / arithmetic / a non-integer comparand.
        if let Some(indices) = self.try_lower_int2_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // text path (type matrix, doc 19): a SIMPLE text comparison (`textcol = 'lit'` / `<>`)
        // evaluates via the byte-wise text-equality kernel. Returns None for a non-text predicate
        // (fall through to int4); errors on a text shape not yet supported (LIKE / inequalities /
        // AND-OR / mixed) so it never silently mis-answers.
        if let Some(indices) = self.try_lower_text_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // Boolean combinators (AND/OR) and `Ne` lower to the general mask-based predicate VM (each
        // comparison -> a mask, MaskBinary combines, the terminal compacts). The fused 2-col / arith
        // / col-vs-col fast paths below handle the single eq/lt/le/gt/ge comparisons.
        if matches!(
            compare,
            ResidentBinaryOp::And | ResidentBinaryOp::Or | ResidentBinaryOp::Ne
        ) {
            let mut program = Vec::new();
            let mut needles: Vec<Vec<u8>> = Vec::new();
            compile_predicate_program(predicate, table, snapshot, &mut program, &mut needles)?;
            return device_memory
                .run_expr_predicate_filter_with_text(
                    &program,
                    &needles,
                    row_count,
                    ResidentElemType::I32,
                )
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())));
        }

        let Some(comparison) = compare_op_code(*compare) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident Expr predicate requires a comparison op (eq/lt/le/gt/ge/ne) or AND/OR"
                    .to_string(),
            )));
        };

        // PEEPHOLE fast-path (design: tuned fused kernels live UNDER the general executor): the exact
        // shape `Compare(arith(Column, Column), Int4Literal)` lowers to the single fused 2-col kernel
        // instead of the 3-launch VM program. Behavior-identical to the VM; just fewer launches.
        if let (
            ResidentExpr::Binary {
                op: arith,
                lhs: a,
                rhs: b,
            },
            ResidentExpr::Int4Literal(needle),
        ) = (lhs.as_ref(), rhs.as_ref())
        {
            if let (Some(op_code), ResidentExpr::Column(col_a), ResidentExpr::Column(col_b)) =
                (arith_op_code(*arith), a.as_ref(), b.as_ref())
            {
                let a_offset = resident_device_int4_column_offset(snapshot, table, *col_a)?;
                let b_offset = resident_device_int4_column_offset(snapshot, table, *col_b)?;
                return device_memory
                    .expr_filter_two_col_compare_from_payload(
                        a_offset, b_offset, op_code, row_count, *needle, comparison,
                    )
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    });
            }
        }

        // GENERAL path: compile the arithmetic side(s) to bytecode and run the device VM.
        //   - `arith_tree <cmp> literal`        -> arith VM (one value buffer vs scalar)
        //   - `literal <cmp> arith_tree`        -> arith VM, comparison flipped
        //   - `arith_tree <cmp> arith_tree`     -> compile both, col-vs-col / expr-vs-expr VM
        // (AND/OR over masks is the next slice.)
        match (lhs.as_ref(), rhs.as_ref()) {
            (value, ResidentExpr::Int4Literal(needle)) => {
                let mut program = Vec::new();
                compile_arith_program(value, table, snapshot, &mut program)?;
                device_memory
                    .run_expr_arith_filter(&program, row_count, comparison, *needle)
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
            }
            (ResidentExpr::Int4Literal(needle), value) => {
                let mut program = Vec::new();
                compile_arith_program(value, table, snapshot, &mut program)?;
                device_memory
                    .run_expr_arith_filter(
                        &program,
                        row_count,
                        flip_comparison_code(comparison),
                        *needle,
                    )
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
            }
            (lhs_expr, rhs_expr) => {
                let mut program = Vec::new();
                compile_arith_program(lhs_expr, table, snapshot, &mut program)?;
                compile_arith_program(rhs_expr, table, snapshot, &mut program)?;
                device_memory
                    .run_expr_compare_buffers_filter(&program, row_count, comparison)
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
            }
        }
    }
}
