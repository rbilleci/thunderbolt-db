//! State-free left-deep join plan IR shared by SQL binding and resident/streaming execution.

/// A column reference inside a JOIN (ON condition or projection): a bare column (`qualifier: None`) or
/// a qualified `alias.column` (`qualifier: Some(alias)`). Resolved to a specific relation + column
/// index in the join executor (against the bound relations).
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

/// One join step in a left-deep chain. Its condition is one of: explicit ON `conjuncts`
/// (`a.k1=b.k1 [AND a.k2=b.k2]` -- in each pair one operand resolves to the newly joined relation
/// `relations[k+1]`, the other to an accumulated one); `USING(cols)`, which the parser desugars to
/// qualified `conjuncts` AND records the join column names in `coalesce` (they appear ONCE in `*`); or
/// `NATURAL` (`natural=true`, `conjuncts` empty), where the executor joins on -- and coalesces -- the
/// relations' common column names. A single join column is a plain equi-join; two integer members no wider
/// than 32 bits form a composite key packed into one i64. Wider members or more than two members are a
/// follow-up. USING/NATURAL are 2-relation only.
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
    /// accumulated relations NULL-padded. Streaming N-way RIGHT/FULL completion uses bounded recursive
    /// prefix replay, so earlier complements participate in later left-deep steps without host tuples.
    pub outer_left: bool,
    pub outer_right: bool,
}

/// A LEFT-DEEP chain of equi-joins parsed from the libpg_query FROM clause (M5). `relations` are in
/// left-deep order (`a JOIN b ON.. JOIN c ON..` => `[a, b, c]`); `steps[k]` is the ON that folds
/// `relations[k+1]` into the accumulated set `relations[0..=k]` (so `steps.len() == relations.len()-1`).
/// `projection` is the SELECT list (qualified/unqualified columns + `*` / `alias.*`). The executor carries
/// device-resident relation coordinates across the left-deep steps and materializes only the final projection;
/// no CPU relational join participates. A 2-relation join is the N=2 case (one relation pair, one step).
#[derive(Debug, Clone)]
pub(crate) struct JoinPlan {
    pub relations: Vec<JoinRelationRef>,
    pub steps: Vec<JoinStep>,
    pub projection: Vec<JoinProjItem>,
    /// Output aliases parallel to SELECT-list `projection` items. Stars always carry `None`; a column
    /// alias is preserved through device materialization and may be referenced by ORDER BY.
    pub projection_aliases: Vec<Option<String>>,
    /// ORDER BY keys (plain columns only, each `(column, descending)`) applied to join coordinates on the
    /// GPU before final projection. Hidden non-projected keys resolve directly against their source relation.
    pub order_by: Vec<(JoinColRef, bool)>,
    /// Parallel to `order_by`: the explicit NULLS FIRST/LAST override per key (M3 -- doc 21; `None` = PG
    /// default). Honored ON-DEVICE by the device-coordinate GPU sort.
    pub order_by_nulls_first: Vec<Option<bool>>,
    /// LIMIT / OFFSET window the sorted device coordinates before projection. `None` = unbounded / from row 0.
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}
