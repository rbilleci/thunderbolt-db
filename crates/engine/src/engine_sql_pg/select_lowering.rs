use super::{
    a_const, catalog_range_relation_key, like_pattern_for_literal_prefix, relational_column_index,
    AExpr, AExprKind, BoolExpr, BoolExprType, CatalogSnapshot, ColumnRef, Decimal128, EngineError,
    ExecuteError, GroupedAggKind, GroupedAggregate, Node, NodeEnum, RelationalTable,
    ResidentBinaryOp, ResidentExpr, Select, SelectFilter, SelectFilterOp, SelectOrder,
    SelectProjection, SelectStmt, SetOperation, SortByDir, SortByNulls, SqlType, SqlValue,
    PG_CATALOG_NAMESPACE_OID, PG_INFORMATION_SCHEMA_NAMESPACE_OID, PG_PUBLIC_NAMESPACE_OID,
    PROJECTION_WILDCARD_SENTINEL,
};

/// Parse `sql` with libpg_query (Postgres's grammar) and lift out the single `SELECT` statement's
/// parse tree. Errors — mapped to the engine's `ApplyFailed` — on a parse failure, an empty or
/// multi-statement string, or a non-`SELECT` command: the general GPU executor binds read queries
/// only, and the bind/map layers above resolve the table, projection, and predicate from this tree.
pub(crate) fn parse_single_select(sql: &str) -> Result<SelectStmt, ExecuteError> {
    let parsed =
        pg_query::parse(sql).map_err(|err| sql_pg_error(format!("SQL parse error: {err}")))?;
    let mut stmts = parsed.protobuf.stmts;
    if stmts.len() != 1 {
        return Err(sql_pg_error(format!(
            "expected exactly one SQL statement, found {}",
            stmts.len()
        )));
    }
    let node = stmts
        .remove(0)
        .stmt
        .and_then(|boxed| boxed.node)
        .ok_or_else(|| sql_pg_error("empty SQL statement".to_string()))?;
    match node {
        NodeEnum::SelectStmt(select) => Ok(*select),
        _ => Err(sql_pg_error(
            "the general GPU executor binds SELECT statements only".to_string(),
        )),
    }
}

/// Build the hand-rolled `Select` (table + int4 projection) that the bind/execute path consumes. The
/// WHERE is NOT carried here — it is mapped to a `ResidentExpr` and evaluated by the general executor,
/// so `filter`/`filters`/`filter_groups` stay empty. Rejects the shapes the Expr path does not yet
/// model (joins / multiple FROM relations, set operations, CTEs, DISTINCT, GROUP BY / HAVING, window
/// functions, ORDER BY, LIMIT / OFFSET, row locking, SELECT INTO, VALUES).
pub(super) fn build_select_from_select_stmt(
    stmt: &SelectStmt,
) -> Result<(Select, String), ExecuteError> {
    let [from] = stmt.from_clause.as_slice() else {
        return Err(sql_pg_error(
            "the general GPU executor supports exactly one FROM relation (no joins yet)"
                .to_string(),
        ));
    };
    let NodeEnum::RangeVar(range_var) = node_enum(from)? else {
        return Err(sql_pg_error(
            "FROM must be a single base table (joins / subqueries are not on the Expr path yet)"
                .to_string(),
        ));
    };
    let table = catalog_range_relation_key(range_var)?;
    // The single qualifier a `tbl.col` reference must use: PG hides the relation name behind an alias,
    // so an aliased relation is named only by its alias; an unaliased one by its relation name. The
    // mapper validates qualified column references against this and rejects any other (PG's "missing
    // FROM-clause entry"), so a stray qualifier is never silently resolved to this relation's column.
    let qualifier = range_var
        .alias
        .as_ref()
        .map(|alias| alias.aliasname.clone())
        .unwrap_or_else(|| range_var.relname.clone());

    // HAVING is GROUPED-only. ORDER BY / LIMIT / OFFSET are supported for BOTH grouped (host-side over
    // the materialized group rows) and non-grouped (the projection sorts the surviving indices on the
    // GPU bitonic sort, then gathers + slices) -- parsed unconditionally below.
    let has_group_by = !stmt.group_clause.is_empty();
    let unsupported = [
        (!stmt.distinct_clause.is_empty(), "DISTINCT"),
        (
            stmt.having_clause.is_some() && !has_group_by,
            "HAVING without GROUP BY",
        ),
        (!stmt.window_clause.is_empty(), "window functions"),
        (
            !stmt.locking_clause.is_empty(),
            "row locking (FOR UPDATE/SHARE)",
        ),
        (stmt.with_clause.is_some(), "WITH / CTEs"),
        // A plain SELECT is SETOP_NONE (= 1; proto enums prefix Undefined = 0). UNION/INTERSECT/EXCEPT
        // put their inputs in larg/rarg with an empty top-level from_clause, so the single-FROM check
        // above already rejects them — this guard is defense-in-depth for a tree that carries a set-op
        // tag without the expected shape.
        (
            stmt.op != SetOperation::SetopNone as i32,
            "set operations (UNION/INTERSECT/EXCEPT)",
        ),
        (stmt.into_clause.is_some(), "SELECT INTO"),
        (!stmt.values_lists.is_empty(), "VALUES"),
    ];
    if let Some((_, clause)) = unsupported.iter().find(|(present, _)| *present) {
        return Err(sql_pg_error(format!(
            "{clause} is not on the general GPU executor's Expr path yet"
        )));
    }

    // GROUP BY a single column -> a grouped aggregate projection (group key + one aggregate); no
    // GROUP BY -> the scalar projection. The grouped EXECUTION is GPU hash aggregation (doc 19).
    let group_by = parse_group_by(&stmt.group_clause, &qualifier)?;
    let projection = match &group_by {
        Some(_) => build_grouped_projection(&stmt.target_list, &stmt.group_clause, &qualifier)?,
        None => build_projection(&stmt.target_list, &qualifier)?,
    };
    // ORDER BY / LIMIT / OFFSET apply to grouped (host-side) AND non-grouped (GPU-sorted projection)
    // queries. HAVING is grouped-only (HAVING-without-GROUP-BY was rejected above).
    let order_by = if let Some(default_name) = scalar_aggregate_default_name(&projection) {
        let output_name = match stmt.target_list.as_slice() {
            [target] => match node_enum(target)? {
                NodeEnum::ResTarget(target) if !target.name.is_empty() => target.name.as_str(),
                _ => default_name,
            },
            _ => default_name,
        };
        validate_scalar_aggregate_order_by(&stmt.sort_clause, output_name)?;
        Vec::new()
    } else {
        parse_order_by(&stmt.sort_clause, &qualifier)?
    };
    let limit = parse_limit(&stmt.limit_count)?;
    let offset = parse_limit(&stmt.limit_offset)?;
    let having_groups = if group_by.is_some() {
        parse_having(stmt.having_clause.as_deref(), &qualifier)?
    } else {
        Vec::new()
    };
    // Multi-key COLUMN ORDER BY on a grouped result now sorts on the GPU (the grouped-sort migration).
    // An EXPRESSION ORDER BY (empty placeholder column) on a grouped result still has no GPU path -- the
    // grouped GPU sort reads result COLUMNS, not arbitrary expressions -- so reject it cleanly.
    if group_by.is_some() && order_by.iter().any(|key| key.column.is_empty()) {
        return Err(sql_pg_error(
            "ORDER BY an expression on a grouped result is not yet supported".to_string(),
        ));
    }
    let select = Select {
        table,
        public_only: range_var.schemaname == "public",
        distinct: false,
        projection,
        group_by,
        having_groups,
        filter: None,
        filters: Vec::new(),
        filter_groups: Vec::new(),
        order_by,
        limit,
        offset,
    };
    Ok((select, qualifier))
}

/// Resolve the GROUP BY clause to a single grouped column name (the Expr path groups by one column
/// for now). Empty -> `None`. More than one grouping term is a clear follow-on error.
fn parse_group_by(group_clause: &[Node], qualifier: &str) -> Result<Option<String>, ExecuteError> {
    match group_clause {
        [] => Ok(None),
        [one] => {
            // A bare column -> the column name (the plain-column matrix path). Any other expression
            // (`a+b`, `a*2`) -> the placeholder "(expr)"; the executor materializes it on-device into a
            // derived int key column (via group_key_expr), and the SELECT projection of the SAME
            // expression reads the resulting group key.
            if let NodeEnum::ColumnRef(column_ref) = node_enum(one)? {
                Ok(Some(
                    resolve_column_name(column_ref, qualifier)?.to_string(),
                ))
            } else {
                Ok(Some("(expr)".to_string()))
            }
        }
        // COMPOSITE: two or more COLUMN terms -> select.group_by carries the FIRST (back-compat); ALL
        // names ride group_key_columns (built by the caller) which drives the on-device pack / wide key.
        // Every member must be a plain column (a non-column member -- expression/etc. -- in a composite
        // is out of scope -> a clean error). The executor validates the member TYPES.
        terms => {
            let mut first_name: Option<String> = None;
            for term in terms {
                let NodeEnum::ColumnRef(column_ref) = node_enum(term)? else {
                    return Err(sql_pg_error(
                        "composite GROUP BY supports plain columns on the Expr path (no expression \
                         member yet)"
                            .to_string(),
                    ));
                };
                let name = resolve_column_name(column_ref, qualifier)?.to_string();
                if first_name.is_none() {
                    first_name = Some(name);
                }
            }
            Ok(first_name)
        }
    }
}

/// Structural equality of two parse Nodes IGNORING location (parse positions), so a SELECT expression
/// matches the SAME GROUP BY expression (`SELECT a+b ... GROUP BY a+b`) -- their derived PartialEq would
/// differ only by parse position. Covers the arithmetic GROUP BY shapes (A_Expr over column / constant
/// operands); the operator-name, column-name, and constant leaf nodes carry no location, so comparing
/// them by value is already location-free.
fn node_struct_eq(a: &Node, b: &Node) -> bool {
    match (node_enum(a), node_enum(b)) {
        (Ok(NodeEnum::AExpr(ea)), Ok(NodeEnum::AExpr(eb))) => {
            ea.kind == eb.kind
                && ea.name == eb.name
                && opt_node_struct_eq(ea.lexpr.as_deref(), eb.lexpr.as_deref())
                && opt_node_struct_eq(ea.rexpr.as_deref(), eb.rexpr.as_deref())
        }
        (Ok(NodeEnum::ColumnRef(ca)), Ok(NodeEnum::ColumnRef(cb))) => ca.fields == cb.fields,
        (Ok(NodeEnum::AConst(ka)), Ok(NodeEnum::AConst(kb))) => {
            ka.val == kb.val && ka.isnull == kb.isnull
        }
        _ => false,
    }
}

fn opt_node_struct_eq(a: Option<&Node>, b: Option<&Node>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => node_struct_eq(x, y),
        _ => false,
    }
}

/// Map a grouped SELECT's target list to a `Grouped*` projection: exactly `<group_column>, <aggregate>`
/// (the group key first, then COUNT(*)/SUM/MIN/MAX/AVG). The projected group column must be the GROUP
/// BY column. MIN/MAX grouping is a follow-on (the count+sum kernel does not cover it yet).
fn build_grouped_projection(
    target_list: &[Node],
    group_clause: &[Node],
    qualifier: &str,
) -> Result<SelectProjection, ExecuteError> {
    let n_group = group_clause.len();
    if target_list.len() <= n_group {
        return Err(sql_pg_error(
            "a grouped SELECT must project the GROUP BY column(s) and at least one aggregate"
                .to_string(),
        ));
    }
    let (group_targets, aggregate_targets) = target_list.split_at(n_group);
    // Each leading target i must be the i-th GROUP BY term: a bare column matches by resolved name; an
    // expression GROUP BY matches the SAME expression structurally. The executor's result columns
    // 0..n_group are the group key(s) (a composite GROUP BY unpacks its packed key into them).
    for (group_target, group_node) in group_targets.iter().zip(group_clause) {
        let NodeEnum::ResTarget(group_res) = node_enum(group_target)? else {
            return Err(sql_pg_error("unexpected grouped SELECT target".to_string()));
        };
        let group_val = group_res
            .val
            .as_deref()
            .ok_or_else(|| sql_pg_error("grouped SELECT group target has no value".to_string()))?;
        match node_enum(group_val)? {
            NodeEnum::ColumnRef(group_ref) => {
                let NodeEnum::ColumnRef(want_ref) = node_enum(group_node)? else {
                    return Err(sql_pg_error(
                        "the projected column must match the GROUP BY column".to_string(),
                    ));
                };
                let want = resolve_column_name(want_ref, qualifier)?.to_string();
                if resolve_column_name(group_ref, qualifier)? != want {
                    return Err(sql_pg_error(
                        "the projected column must be a GROUP BY column".to_string(),
                    ));
                }
            }
            _ => {
                if !node_struct_eq(group_val, group_node) {
                    return Err(sql_pg_error(
                        "a leading grouped projection must be a GROUP BY column or its expression"
                            .to_string(),
                    ));
                }
            }
        }
    }
    // The GroupedAggregates carries the FIRST group column's name (a composite's later columns ride
    // group_key_columns + the binding/executor; an expression GROUP BY uses the "(expr)" placeholder).
    let group_column = match node_enum(&group_clause[0])? {
        NodeEnum::ColumnRef(first_ref) => resolve_column_name(first_ref, qualifier)?.to_string(),
        _ => "(expr)".to_string(),
    };
    // Targets 1..=N must each be a scalar aggregate; lift each to a GroupedAggregate keyed by the
    // group column. A single aggregate yields a 1-element vec (same behavior as the old 1-agg form).
    let mut aggregates = Vec::with_capacity(aggregate_targets.len());
    for aggregate_target in aggregate_targets {
        let NodeEnum::ResTarget(agg_res) = node_enum(aggregate_target)? else {
            return Err(sql_pg_error("unexpected grouped SELECT target".to_string()));
        };
        if !agg_res.name.is_empty() {
            return Err(sql_pg_error(
                "aggregate column aliases are not on the grouped Expr path yet".to_string(),
            ));
        }
        let Some(aggregate) = try_parse_scalar_aggregate(agg_res, qualifier)? else {
            return Err(sql_pg_error(
                "every grouped projection after the GROUP BY column must be an aggregate \
                 (COUNT(*) / SUM / AVG / MIN / MAX)"
                    .to_string(),
            ));
        };
        let grouped = match aggregate {
            SelectProjection::CountAll => GroupedAggregate {
                kind: GroupedAggKind::Count,
                value_column: None,
            },
            SelectProjection::Sum { column } => GroupedAggregate {
                kind: GroupedAggKind::Sum,
                value_column: Some(column),
            },
            SelectProjection::Avg { column } => GroupedAggregate {
                kind: GroupedAggKind::Avg,
                value_column: Some(column),
            },
            SelectProjection::Min { column } => GroupedAggregate {
                kind: GroupedAggKind::Min,
                value_column: Some(column),
            },
            SelectProjection::Max { column } => GroupedAggregate {
                kind: GroupedAggKind::Max,
                value_column: Some(column),
            },
            SelectProjection::CountDistinct { column } => GroupedAggregate {
                kind: GroupedAggKind::CountDistinct,
                value_column: Some(column),
            },
            _ => return Err(sql_pg_error("unsupported grouped aggregate".to_string())),
        };
        aggregates.push(grouped);
    }
    Ok(SelectProjection::GroupedAggregates {
        group_column,
        aggregates,
    })
}

/// Map the SELECT target list to a projection: a lone `*` -> `All`, otherwise a list of plain column
/// names. Rejects column aliases and projected expressions / aggregates (only int4 column projection
/// is materialized today).
fn build_projection(
    target_list: &[Node],
    qualifier: &str,
) -> Result<SelectProjection, ExecuteError> {
    if let [only] = target_list {
        if let NodeEnum::ResTarget(res_target) = node_enum(only)? {
            if let Some(val) = res_target.val.as_deref() {
                if let NodeEnum::ColumnRef(column_ref) = node_enum(val)? {
                    if column_ref.fields.len() == 1
                        && matches!(node_enum(&column_ref.fields[0])?, NodeEnum::AStar(_))
                    {
                        return Ok(SelectProjection::All);
                    }
                }
            }
            // A lone scalar aggregate (`SELECT count(*) FROM t WHERE ...`); the operator axis.
            if let Some(aggregate) = try_parse_scalar_aggregate(res_target, qualifier)? {
                return Ok(aggregate);
            }
        }
    }

    let mut columns = Vec::with_capacity(target_list.len());
    for target in target_list {
        let NodeEnum::ResTarget(res_target) = node_enum(target)? else {
            return Err(sql_pg_error("unexpected SELECT target".to_string()));
        };
        if !res_target.name.is_empty() {
            return Err(sql_pg_error(
                "column aliases are not on the Expr path yet".to_string(),
            ));
        }
        let val = res_target
            .val
            .as_deref()
            .ok_or_else(|| sql_pg_error("SELECT target has no value expression".to_string()))?;
        let NodeEnum::ColumnRef(column_ref) = node_enum(val)? else {
            return Err(sql_pg_error(
                "the general GPU executor projects plain columns only (no expressions / aggregates \
                 yet)"
                    .to_string(),
            ));
        };
        let star = match column_ref.fields.as_slice() {
            [field] => matches!(node_enum(field)?, NodeEnum::AStar(_)),
            [relation, star] => {
                matches!(node_enum(relation)?, NodeEnum::String(name) if name.sval == qualifier)
                    && matches!(node_enum(star)?, NodeEnum::AStar(_))
            }
            _ => false,
        };
        if star {
            if !res_target.name.is_empty() {
                return Err(sql_pg_error(
                    "a projected star cannot have an alias".to_string(),
                ));
            }
            columns.push(PROJECTION_WILDCARD_SENTINEL.to_string());
        } else {
            columns.push(resolve_column_name(column_ref, qualifier)?.to_string());
        }
    }
    if columns.is_empty() {
        return Err(sql_pg_error("SELECT has no projected columns".to_string()));
    }
    Ok(SelectProjection::Columns(columns))
}

/// Detect a scalar aggregate in a SELECT target: `count(*)` -> `CountAll`. Other aggregates
/// (`count(col)`, `sum`/`min`/`max`/`avg`) are recognized but rejected as follow-ons (a clear error,
/// not the generic "plain columns only" message). `None` if the target is not an aggregate call.
fn try_parse_scalar_aggregate(
    res_target: &pg_query::protobuf::ResTarget,
    qualifier: &str,
) -> Result<Option<SelectProjection>, ExecuteError> {
    let Some(val) = res_target.val.as_deref() else {
        return Ok(None);
    };
    let NodeEnum::FuncCall(func) = node_enum(val)? else {
        return Ok(None);
    };
    // The function name is the last element of `funcname` (a schema qualifier would prefix it).
    let name = match func.funcname.last().map(node_enum).transpose()? {
        Some(NodeEnum::String(string)) => string.sval.to_ascii_lowercase(),
        _ => return Ok(None),
    };
    if !matches!(name.as_str(), "count" | "sum" | "min" | "max" | "avg") {
        return Ok(None);
    }
    // FILTER / OVER / WITHIN GROUP / ordered-set modifiers live INSIDE the FuncCall, so the SELECT-
    // level guards (window_clause, ...) cannot see them. Reject them here, or `COUNT(*) FILTER (WHERE
    // p)` / `COUNT(*) OVER ()` would be silently treated as a plain COUNT(*) -- a WRONG answer (the
    // FILTER/window dropped). [audit P0] (Checked BEFORE DISTINCT so `COUNT(DISTINCT v) FILTER (..)`
    // is rejected too, not mis-parsed as a plain distinct count.)
    if func.agg_filter.is_some() {
        return Err(sql_pg_error(
            "aggregate FILTER is not on the Expr path yet".to_string(),
        ));
    }
    if func.over.is_some() {
        return Err(sql_pg_error(
            "window functions (OVER) are not on the Expr path yet".to_string(),
        ));
    }
    if !func.agg_order.is_empty() || func.agg_within_group {
        return Err(sql_pg_error(
            "ordered-set / WITHIN GROUP aggregates are not on the Expr path yet".to_string(),
        ));
    }
    // COUNT(DISTINCT v): the only DISTINCT aggregate on the Expr path. The GROUPED form runs FULLY on
    // the GPU -- sort (group, v) -> mark-new-distinct -> SUM of the per-row new-distinct flags per
    // group (no CPU). Other DISTINCT aggregates (SUM/AVG/MIN/MAX DISTINCT) are follow-ups; COUNT(DISTINCT
    // *) is invalid SQL. A bare/scalar COUNT(DISTINCT v) parses here too but is rejected at execution.
    if func.agg_distinct {
        if name == "count" && !func.agg_star {
            if let [arg] = func.args.as_slice() {
                if let NodeEnum::ColumnRef(column_ref) = node_enum(arg)? {
                    let column = resolve_column_name(column_ref, qualifier)?.to_string();
                    return Ok(Some(SelectProjection::CountDistinct { column }));
                }
            }
        }
        return Err(sql_pg_error(
            "only COUNT(DISTINCT col) is supported on the Expr path's aggregate DISTINCT"
                .to_string(),
        ));
    }
    if name == "count" && func.agg_star && func.args.is_empty() {
        return Ok(Some(SelectProjection::CountAll));
    }
    // SUM/MIN/MAX/AVG(col): reduction aggregates over a single column reference; the executor validates
    // int4 and reduces on the GPU (AVG = the GPU sum / the count). (count(col) / int8 / numeric are
    // follow-ons.)
    if matches!(name.as_str(), "sum" | "min" | "max" | "avg") {
        if let [arg] = func.args.as_slice() {
            if let NodeEnum::ColumnRef(column_ref) = node_enum(arg)? {
                let column = resolve_column_name(column_ref, qualifier)?.to_string();
                return Ok(Some(match name.as_str() {
                    "sum" => SelectProjection::Sum { column },
                    "min" => SelectProjection::Min { column },
                    "max" => SelectProjection::Max { column },
                    _ => SelectProjection::Avg { column },
                }));
            }
        }
        return Err(sql_pg_error(
            "SUM / MIN / MAX / AVG support a single column argument on the Expr path".to_string(),
        ));
    }
    Err(sql_pg_error(
        "only COUNT(*) / SUM / MIN / MAX / AVG(col) are on the general GPU executor's aggregate path \
         yet"
            .to_string(),
    ))
}

fn scalar_aggregate_default_name(projection: &SelectProjection) -> Option<&'static str> {
    match projection {
        SelectProjection::CountAll | SelectProjection::CountDistinct { .. } => Some("count"),
        SelectProjection::Sum { .. } => Some("sum"),
        SelectProjection::Avg { .. } => Some("avg"),
        SelectProjection::Min { .. } => Some("min"),
        SelectProjection::Max { .. } => Some("max"),
        _ => None,
    }
}

fn validate_scalar_aggregate_order_by(
    sort_clause: &[Node],
    output_name: &str,
) -> Result<(), ExecuteError> {
    for item in sort_clause {
        let NodeEnum::SortBy(sort_by) = node_enum(item)? else {
            return Err(sql_pg_error("malformed ORDER BY clause".to_string()));
        };
        if sort_by.sortby_dir == SortByDir::SortbyUsing as i32 || !sort_by.use_op.is_empty() {
            return Err(sql_pg_error(
                "ORDER BY USING is not supported by the GPU ordering path".to_string(),
            ));
        }
        let node = sort_by
            .node
            .as_deref()
            .ok_or_else(|| sql_pg_error("ORDER BY key has no expression".to_string()))?;
        let output_reference = match node_enum(node)? {
            NodeEnum::ColumnRef(column) => match column.fields.as_slice() {
                [field] => {
                    matches!(node_enum(field)?, NodeEnum::String(name) if name.sval == output_name)
                }
                _ => false,
            },
            NodeEnum::AConst(constant) => {
                matches!(&constant.val, Some(a_const::Val::Ival(position)) if position.ival == 1)
            }
            _ => false,
        };
        if !output_reference {
            return Err(sql_pg_error(format!(
                "scalar aggregate ORDER BY must reference output column {output_name:?} or position 1"
            )));
        }
    }
    Ok(())
}

/// Map a WHERE / predicate parse node to the general `ResidentExpr` IR, resolving column names to
/// indices against `table`. Recurses through `A_Expr` (arithmetic + comparison). Anything the IR
/// cannot represent — `BoolExpr` (`AND`/`OR`, slice 4), non-operator `A_Expr` (IN / LIKE / BETWEEN),
/// non-integer literals, function calls, subqueries — is a hard error so the routing layer never
/// silently mis-answers.
pub(super) fn map_predicate_node(
    node: &Node,
    table: &RelationalTable,
    qualifier: &str,
    catalog: &CatalogSnapshot,
) -> Result<ResidentExpr, ExecuteError> {
    match node_enum(node)? {
        NodeEnum::ColumnRef(column_ref) => Ok(ResidentExpr::Column(relational_column_index(
            table,
            resolve_column_name(column_ref, qualifier)?,
        )?)),
        NodeEnum::AConst(constant) => match &constant.val {
            Some(a_const::Val::Ival(integer)) => Ok(ResidentExpr::Int4Literal(integer.ival)),
            // A numeric literal (a decimal point / exponent) arrives as a Float whose `fval` is the
            // source text; parse it to Decimal128 at its natural scale. The type matrix (doc 19)
            // compares it by rescaling to the column scale at lowering time.
            Some(a_const::Val::Fval(float)) => Decimal128::parse(&float.fval)
                .map(ResidentExpr::NumericLiteral)
                .ok_or_else(|| sql_pg_error(format!("malformed numeric literal: {}", float.fval))),
            // A quoted string literal -> a text comparison value (the type matrix, doc 19). Byte-wise
            // (deterministic-collation equality is byte identity).
            Some(a_const::Val::Sval(string)) => Ok(ResidentExpr::TextLiteral(string.sval.clone())),
            // A `true` / `false` literal -> the comparison value for `flag = true` / `flag = false`
            // (the type matrix, doc 19).
            Some(a_const::Val::Boolval(boolean)) => Ok(ResidentExpr::BoolLiteral(boolean.boolval)),
            _ => Err(sql_pg_error(
                "the general GPU executor supports int4, numeric, text, and bool literals only"
                    .to_string(),
            )),
        },
        NodeEnum::AExpr(a_expr) => map_a_expr(a_expr, table, qualifier, catalog),
        NodeEnum::BoolExpr(bool_expr) => map_bool_expr(bool_expr, table, qualifier, catalog),
        NodeEnum::NullTest(null_test) => map_null_test(null_test, table, qualifier),
        NodeEnum::TypeCast(type_cast) => map_type_cast(type_cast, table, qualifier, catalog),
        NodeEnum::CollateClause(collate) => {
            let collation = collate
                .collname
                .iter()
                .map(|part| match node_enum(part)? {
                    NodeEnum::String(part) => Ok(part.sval.to_ascii_lowercase()),
                    _ => Err(sql_pg_error(
                        "collation name contains a non-name component".to_string(),
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?
                .join(".");
            if !is_gpu_catalog_table(table)
                || !matches!(collation.as_str(), "default" | "pg_catalog.default")
            {
                return Err(sql_pg_error(format!(
                    "collation {collation} is not supported by this GPU expression path"
                )));
            }
            let arg = collate
                .arg
                .as_deref()
                .ok_or_else(|| sql_pg_error("COLLATE is missing its argument".to_string()))?;
            map_predicate_node(arg, table, qualifier, catalog)
        }
        NodeEnum::FuncCall(function) => super::catalog_visibility::map_catalog_predicate_function(
            function, table, qualifier, catalog,
        ),
        _ => Err(sql_pg_error(
            "unsupported expression node for the general GPU executor".to_string(),
        )),
    }
}

fn map_type_cast(
    type_cast: &pg_query::protobuf::TypeCast,
    table: &RelationalTable,
    qualifier: &str,
    catalog: &CatalogSnapshot,
) -> Result<ResidentExpr, ExecuteError> {
    if !is_gpu_catalog_table(table) {
        return Err(sql_pg_error(
            "unsupported expression node for the general GPU executor".to_string(),
        ));
    }
    let arg = type_cast
        .arg
        .as_deref()
        .ok_or_else(|| sql_pg_error("type cast is missing its argument".to_string()))?;
    let type_name = type_cast
        .type_name
        .as_ref()
        .ok_or_else(|| sql_pg_error("type cast is missing its target type".to_string()))?;
    if type_name.setof
        || type_name.pct_type
        || !type_name.typmods.is_empty()
        || !type_name.array_bounds.is_empty()
    {
        return Err(sql_pg_error(
            "complex cast targets are not supported by the general GPU catalog executor"
                .to_string(),
        ));
    }
    let target = type_name
        .names
        .iter()
        .map(|name| match node_enum(name)? {
            NodeEnum::String(name) => Ok(name.sval.to_ascii_lowercase()),
            _ => Err(sql_pg_error(
                "cast target contains a non-name component".to_string(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?
        .join(".");

    if matches!(target.as_str(), "regnamespace" | "pg_catalog.regnamespace") {
        let name = cast_string_literal(arg, "regnamespace")?;
        let oid = match name.as_str() {
            "public" if catalog.relational_public_schema_exists => PG_PUBLIC_NAMESPACE_OID,
            "pg_catalog" => PG_CATALOG_NAMESPACE_OID,
            "information_schema" => PG_INFORMATION_SCHEMA_NAMESPACE_OID,
            _ => {
                return Err(sql_pg_error(format!(
                    "schema \"{name}\" does not exist for regnamespace cast"
                )))
            }
        };
        return Ok(ResidentExpr::Int4Literal(oid));
    }
    if matches!(target.as_str(), "regclass" | "pg_catalog.regclass") {
        let source = cast_string_literal(arg, "regclass")?;
        let name = source.strip_prefix("public.").unwrap_or(&source);
        let oid = catalog
            .relational_catalog
            .get(name)
            .map(|relation| relation.oid)
            .or_else(|| {
                catalog
                    .relational_views
                    .get(name)
                    .map(|relation| relation.oid)
            })
            .or_else(|| {
                catalog
                    .relational_materialized_views
                    .get(name)
                    .map(|relation| relation.oid)
            })
            .or_else(|| {
                catalog
                    .relational_sequences
                    .get(name)
                    .map(|relation| relation.oid)
            })
            .ok_or_else(|| {
                sql_pg_error(format!(
                    "relation \"{source}\" does not exist for regclass cast"
                ))
            })?;
        let oid = i32::try_from(oid)
            .map_err(|_| sql_pg_error(format!("relation OID {oid} exceeds int4")))?;
        return Ok(ResidentExpr::Int4Literal(oid));
    }

    let inner = map_predicate_node(arg, table, qualifier, catalog)?;
    let compatible = match target.as_str() {
        "int4" | "integer" | "pg_catalog.int4" => match &inner {
            ResidentExpr::Int4Literal(_) => true,
            ResidentExpr::Column(index) => matches!(table.columns[*index].ty, SqlType::Int4),
            _ => false,
        },
        "text" | "pg_catalog.text" => match &inner {
            ResidentExpr::TextLiteral(_) => true,
            ResidentExpr::Column(index) => matches!(table.columns[*index].ty, SqlType::Text),
            _ => false,
        },
        _ => false,
    };
    if compatible {
        Ok(inner)
    } else {
        Err(sql_pg_error(format!(
            "cast to {target} is not a representation-preserving GPU expression cast"
        )))
    }
}

fn cast_string_literal(node: &Node, target: &str) -> Result<String, ExecuteError> {
    let NodeEnum::AConst(constant) = node_enum(node)? else {
        return Err(sql_pg_error(format!(
            "{target} casts require a string literal on the GPU catalog path"
        )));
    };
    match &constant.val {
        Some(a_const::Val::Sval(value)) => Ok(value.sval.clone()),
        _ => Err(sql_pg_error(format!(
            "{target} casts require a string literal on the GPU catalog path"
        ))),
    }
}

/// Map a `col IS NULL` / `col IS NOT NULL` (`NullTest`) node to the general IR (M3 -- doc 21). The
/// argument must be a bare column reference (a NULL test over an expression is a follow-up); the
/// `nulltesttype` selects IS NULL (`IsNull`) vs IS NOT NULL (`IsNotNull`).
fn map_null_test(
    null_test: &pg_query::protobuf::NullTest,
    table: &RelationalTable,
    qualifier: &str,
) -> Result<ResidentExpr, ExecuteError> {
    let arg = null_test
        .arg
        .as_deref()
        .ok_or_else(|| sql_pg_error("IS NULL is missing its argument".to_string()))?;
    let NodeEnum::ColumnRef(column_ref) = node_enum(arg)? else {
        return Err(sql_pg_error(
            "IS NULL / IS NOT NULL is supported only on a column reference".to_string(),
        ));
    };
    let col = relational_column_index(table, resolve_column_name(column_ref, qualifier)?)?;
    let is_not_null = match null_test.nulltesttype() {
        pg_query::protobuf::NullTestType::IsNull => false,
        pg_query::protobuf::NullTestType::IsNotNull => true,
        pg_query::protobuf::NullTestType::Undefined => {
            return Err(sql_pg_error("malformed IS NULL test".to_string()));
        }
    };
    Ok(ResidentExpr::IsNull { col, is_not_null })
}

/// Map a `BoolExpr` (`AND` / `OR` / `NOT`) to the general IR. `AND` / `OR` left-fold their operands
/// into a binary tree of `Binary{And/Or}` — libpg_query flattens `a AND b AND c` into one BoolExpr
/// with N args, so a chain becomes `((a AND b) AND c)`. `NOT` (a unary boolean) has no IR node yet and
/// is a hard error.
fn map_bool_expr(
    bool_expr: &BoolExpr,
    table: &RelationalTable,
    qualifier: &str,
    catalog: &CatalogSnapshot,
) -> Result<ResidentExpr, ExecuteError> {
    let op = if bool_expr.boolop == BoolExprType::AndExpr as i32 {
        ResidentBinaryOp::And
    } else if bool_expr.boolop == BoolExprType::OrExpr as i32 {
        ResidentBinaryOp::Or
    } else {
        // NOT_EXPR: support `NOT flag` over a bare bool column -- it is equivalent to `flag = false`,
        // which the bool-predicate path lowers via the bitmap->mask kernel (negate). General NOT (over
        // comparisons / AND-OR, needing De Morgan) is still a follow-on.
        let mut args = bool_expr.args.iter();
        let only = args
            .next()
            .ok_or_else(|| sql_pg_error("NOT has no operand".to_string()))?;
        if args.next().is_some() {
            return Err(sql_pg_error(
                "NOT predicates are not on the Expr path yet".to_string(),
            ));
        }
        let inner = map_predicate_node(only, table, qualifier, catalog)?;
        if matches!(inner, ResidentExpr::Column(_)) {
            return Ok(ResidentExpr::Binary {
                op: ResidentBinaryOp::Eq,
                lhs: Box::new(inner),
                rhs: Box::new(ResidentExpr::BoolLiteral(false)),
            });
        }
        return Err(sql_pg_error(
            "NOT predicates are not on the Expr path yet (only NOT <bool column>)".to_string(),
        ));
    };
    let mut args = bool_expr.args.iter();
    let first = args
        .next()
        .ok_or_else(|| sql_pg_error("boolean expression has no operands".to_string()))?;
    let mut folded = map_predicate_node(first, table, qualifier, catalog)?;
    for arg in args {
        folded = ResidentExpr::Binary {
            op,
            lhs: Box::new(folded),
            rhs: Box::new(map_predicate_node(arg, table, qualifier, catalog)?),
        };
    }
    Ok(folded)
}

/// Map an `A_Expr` (a binary-operator expression) to a `ResidentExpr::Binary`. Only `AEXPR_OP` (a
/// normal operator: `+ - * = <> < <= > >=`) with both operands present is supported; other kinds
/// (`IN` / `LIKE` / `BETWEEN` / ...) and unsupported operator tokens (`/`, `%`, ...) are rejected.
fn map_a_expr(
    a_expr: &AExpr,
    table: &RelationalTable,
    qualifier: &str,
    catalog: &CatalogSnapshot,
) -> Result<ResidentExpr, ExecuteError> {
    // AEXPR_IN is `x IN (a, b, ...)` -> an OR-chain of equalities (`NOT IN` -> an AND-chain of `<>`),
    // entirely on the general GPU executor (the same Binary Eq/Ne + And/Or it already runs).
    if a_expr.kind == AExprKind::AexprIn as i32 {
        return map_in_expr(a_expr, table, qualifier, catalog);
    }
    // AEXPR_OP is a normal operator (`+ - * = <> < <= > >=`); AEXPR_LIKE is `LIKE` (operator `~~`,
    // `!~~` for NOT LIKE). Both carry the operator token in `name` and both operands; other kinds
    // (`BETWEEN` / ...) are rejected.
    if a_expr.kind != AExprKind::AexprOp as i32 && a_expr.kind != AExprKind::AexprLike as i32 {
        return Err(sql_pg_error(
            "only operator, LIKE, and IN predicates are supported (no BETWEEN yet)".to_string(),
        ));
    }
    let token = aexpr_op_token(a_expr)?;
    let lexpr = a_expr.lexpr.as_deref().ok_or_else(|| {
        sql_pg_error("operator expression is missing its left operand".to_string())
    })?;
    let rexpr = a_expr.rexpr.as_deref().ok_or_else(|| {
        sql_pg_error(
            "operator expression is missing its right operand (unary operators unsupported)"
                .to_string(),
        )
    })?;
    let mut lhs = map_predicate_node(lexpr, table, qualifier, catalog)?;
    let mut rhs = map_predicate_node(rexpr, table, qualifier, catalog)?;
    if matches!(token, "=" | "<>" | "<" | "<=" | ">" | ">=") {
        coerce_catalog_oid_literal(table, &lhs, rexpr, &mut rhs)?;
        coerce_catalog_oid_literal(table, &rhs, lexpr, &mut lhs)?;
        reject_catalog_oid_text_cast_comparison(table, &lhs, &rhs)?;
    }
    if matches!(token, "~" | "!~") {
        if !is_gpu_catalog_table(table) {
            return Err(sql_pg_error(
                "regular-expression compatibility is restricted to GPU catalog predicates"
                    .to_string(),
            ));
        }
        let ResidentExpr::TextLiteral(pattern) = rhs else {
            return Err(sql_pg_error(
                "catalog regular-expression predicates require a text literal".to_string(),
            ));
        };
        let exact_pg_toast_namespace_exclusion = token == "!~"
            && pattern == "^pg_toast"
            && table.schema == "pg_catalog"
            && table.name == "pg_namespace"
            && matches!(
                &lhs,
                ResidentExpr::Column(index)
                    if table.columns.get(*index).is_some_and(|column| column.name == "nspname")
            );
        if exact_pg_toast_namespace_exclusion {
            // The modeled namespace relation intentionally has no pg_toast row. Preserve this
            // exact psql namespace exclusion as a GPU-evaluated non-null self-comparison. The
            // pattern alone is insufficient: applying it to another catalog text column would
            // turn a real negative-regex predicate into an unconditional truth value.
            return Ok(ResidentExpr::Binary {
                op: ResidentBinaryOp::Eq,
                lhs: Box::new(lhs.clone()),
                rhs: Box::new(lhs),
            });
        }
        if token == "!~" {
            return Err(sql_pg_error(
                "negative regular-expression predicates beyond pg_toast exclusion are not supported"
                    .to_string(),
            ));
        }
        let (op, pattern) = catalog_regex_pattern(&pattern)?;
        return Ok(ResidentExpr::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(ResidentExpr::TextLiteral(pattern)),
        });
    }
    Ok(ResidentExpr::Binary {
        op: map_operator(token)?,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    })
}

fn coerce_catalog_oid_literal(
    table: &RelationalTable,
    column: &ResidentExpr,
    candidate_node: &Node,
    candidate: &mut ResidentExpr,
) -> Result<(), ExecuteError> {
    if !is_gpu_catalog_table(table) {
        return Ok(());
    }
    let ResidentExpr::Column(index) = column else {
        return Ok(());
    };
    if !matches!(table.columns[*index].ty, SqlType::Int4) {
        return Ok(());
    }
    let NodeEnum::AConst(constant) = node_enum(candidate_node)? else {
        return Ok(());
    };
    if !matches!(constant.val.as_ref(), Some(a_const::Val::Sval(_))) {
        return Ok(());
    }
    let ResidentExpr::TextLiteral(value) = candidate else {
        return Ok(());
    };
    let parsed = value
        .parse::<i32>()
        .map_err(|_| sql_pg_error(format!("invalid int4 catalog literal {value:?}")))?;
    *candidate = ResidentExpr::Int4Literal(parsed);
    Ok(())
}

fn reject_catalog_oid_text_cast_comparison(
    table: &RelationalTable,
    lhs: &ResidentExpr,
    rhs: &ResidentExpr,
) -> Result<(), ExecuteError> {
    if !is_gpu_catalog_table(table) {
        return Ok(());
    }
    let int4_column = |expr: &ResidentExpr| {
        matches!(
            expr,
            ResidentExpr::Column(index) if matches!(table.columns[*index].ty, SqlType::Int4)
        )
    };
    if (int4_column(lhs) && matches!(rhs, ResidentExpr::TextLiteral(_)))
        || (int4_column(rhs) && matches!(lhs, ResidentExpr::TextLiteral(_)))
    {
        return Err(sql_pg_error(
            "catalog oid/int4 columns cannot be compared to an explicitly typed text value"
                .to_string(),
        ));
    }
    Ok(())
}

fn is_gpu_catalog_table(table: &RelationalTable) -> bool {
    matches!(table.schema.as_str(), "pg_catalog" | "information_schema")
}

fn catalog_regex_pattern(pattern: &str) -> Result<(ResidentBinaryOp, String), ExecuteError> {
    if let Some(exact) = pattern
        .strip_prefix("^(")
        .and_then(|value| value.strip_suffix(")$"))
    {
        if let Some(prefix) = exact.strip_suffix(".*") {
            if !catalog_regex_has_metacharacter(prefix) {
                return Ok((
                    ResidentBinaryOp::Like,
                    like_pattern_for_literal_prefix(prefix),
                ));
            }
        } else if !catalog_regex_has_metacharacter(exact) {
            return Ok((ResidentBinaryOp::Eq, exact.to_string()));
        }
    }
    Err(sql_pg_error(format!(
        "catalog regular expression {pattern:?} is outside the exact/prefix GPU subset"
    )))
}

fn catalog_regex_has_metacharacter(value: &str) -> bool {
    value.chars().any(|ch| {
        matches!(
            ch,
            '.' | '^' | '$' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\'
        )
    })
}

/// Map `x IN (v1, v2, ...)` to an OR-chain of equalities (`x = v1 OR x = v2 OR ...`); `x NOT IN (...)`
/// to an AND-chain of not-equals (`x <> v1 AND x <> v2 AND ...`). libpg_query tags both as `AEXPR_IN`
/// and carries the operator in `name` (`=` for IN, `<>` for NOT IN) with the value LIST in `rexpr`. The
/// whole thing lowers to the Binary Eq/Ne + And/Or the general GPU executor already runs (no new kernel).
/// A subquery `IN (SELECT ...)` (rexpr is not a value list) is a follow-up.
fn map_in_expr(
    a_expr: &AExpr,
    table: &RelationalTable,
    qualifier: &str,
    catalog: &CatalogSnapshot,
) -> Result<ResidentExpr, ExecuteError> {
    let negated = aexpr_op_token(a_expr)? == "<>";
    let lexpr = a_expr
        .lexpr
        .as_deref()
        .ok_or_else(|| sql_pg_error("IN is missing its left operand".to_string()))?;
    let rexpr = a_expr
        .rexpr
        .as_deref()
        .ok_or_else(|| sql_pg_error("IN is missing its value list".to_string()))?;
    let NodeEnum::List(list) = node_enum(rexpr)? else {
        return Err(sql_pg_error(
            "IN requires a parenthesized value list (subquery IN is a follow-up)".to_string(),
        ));
    };
    if list.items.is_empty() {
        return Err(sql_pg_error("IN requires at least one value".to_string()));
    }
    let lhs = map_predicate_node(lexpr, table, qualifier, catalog)?;
    let (cmp, combine) = if negated {
        (ResidentBinaryOp::Ne, ResidentBinaryOp::And)
    } else {
        (ResidentBinaryOp::Eq, ResidentBinaryOp::Or)
    };
    let mut folded: Option<ResidentExpr> = None;
    for item in &list.items {
        let term = ResidentExpr::Binary {
            op: cmp,
            lhs: Box::new(lhs.clone()),
            rhs: Box::new(map_predicate_node(item, table, qualifier, catalog)?),
        };
        folded = Some(match folded {
            None => term,
            Some(prev) => ResidentExpr::Binary {
                op: combine,
                lhs: Box::new(prev),
                rhs: Box::new(term),
            },
        });
    }
    Ok(folded.expect("non-empty IN list checked above"))
}

/// Map a Postgres operator token to a `ResidentBinaryOp`. `<>` is PG's not-equal (PG normalizes `!=`
/// to `<>` in the grammar).
fn map_operator(token: &str) -> Result<ResidentBinaryOp, ExecuteError> {
    Ok(match token {
        "+" => ResidentBinaryOp::Add,
        "-" => ResidentBinaryOp::Sub,
        "*" => ResidentBinaryOp::Mul,
        "=" => ResidentBinaryOp::Eq,
        "<>" => ResidentBinaryOp::Ne,
        "<" => ResidentBinaryOp::Lt,
        "<=" => ResidentBinaryOp::Le,
        ">" => ResidentBinaryOp::Gt,
        ">=" => ResidentBinaryOp::Ge,
        // PG's `LIKE` operator (`x LIKE y` parses as `x ~~ y`). `!~~` (NOT LIKE) is a follow-on.
        "~~" => ResidentBinaryOp::Like,
        other => {
            return Err(sql_pg_error(format!(
                "operator \"{other}\" is not supported by the general GPU executor yet"
            )))
        }
    })
}

/// The populated `NodeEnum` of a parse `Node` (a node with no inner value is a parser inconsistency).
pub(super) fn node_enum(node: &Node) -> Result<&NodeEnum, ExecuteError> {
    node.node
        .as_ref()
        .ok_or_else(|| sql_pg_error("empty libpg_query parse node".to_string()))
}

/// The column name a `ColumnRef` resolves to, VALIDATING any table qualifier against the bound
/// relation's `qualifier` (its alias, else its name). `a` -> "a"; `t.a` / `alias.a` -> "a" only when
/// the qualifier names the FROM relation, else PG's "missing FROM-clause entry for table ..." (so a
/// stray `wrongtab.a` is a hard error, never silently resolved to this relation's column — load-
/// bearing once joins make two same-named columns ambiguous). Rejects `*`, qualified-star, and
/// schema/catalog-qualified (3+ part) references.
pub(super) fn resolve_column_name<'a>(
    column_ref: &'a ColumnRef,
    qualifier: &str,
) -> Result<&'a str, ExecuteError> {
    let parts = column_ref
        .fields
        .iter()
        .map(|field| match field.node.as_ref() {
            Some(NodeEnum::String(string)) => Ok(string.sval.as_str()),
            _ => Err(sql_pg_error(
                "expected a column name (got `*` or a non-name column reference)".to_string(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    match parts.as_slice() {
        [column] => Ok(column),
        [table_qualifier, column] => {
            if *table_qualifier == qualifier {
                Ok(column)
            } else {
                Err(sql_pg_error(format!(
                    "missing FROM-clause entry for table \"{table_qualifier}\""
                )))
            }
        }
        _ => Err(sql_pg_error(
            "schema-qualified or multi-part column references are not supported".to_string(),
        )),
    }
}

/// The operator token of an `A_Expr` (`name` carries it as a single `String` node).
pub(super) fn aexpr_op_token(a_expr: &AExpr) -> Result<&str, ExecuteError> {
    match a_expr.name.as_slice() {
        [name] => match name.node.as_ref() {
            Some(NodeEnum::String(string)) => Ok(&string.sval),
            _ => Err(sql_pg_error(
                "operator name is not a String node".to_string(),
            )),
        },
        [schema, name] if matches!(schema.node.as_ref(), Some(NodeEnum::String(schema)) if schema.sval == "pg_catalog") => {
            match name.node.as_ref() {
                Some(NodeEnum::String(string)) => Ok(&string.sval),
                _ => Err(sql_pg_error(
                    "operator name is not a String node".to_string(),
                )),
            }
        }
        _ => Err(sql_pg_error(
            "multi-part operators are not supported".to_string(),
        )),
    }
}

/// The RESULT column a GROUP BY ORDER BY / HAVING term references: a bare column (the group key) by its
/// name, or an aggregate by the name the binding gives its result column (count/sum/min/max/avg). Used
/// to look the value up by index in the materialized grouped row. (A projection with two of the same
/// aggregate function makes that name ambiguous -- the first match wins; uncommon.)
pub(super) fn result_column_name(node: &Node, qualifier: &str) -> Result<String, ExecuteError> {
    match node_enum(node)? {
        NodeEnum::ColumnRef(column_ref) => {
            Ok(resolve_column_name(column_ref, qualifier)?.to_string())
        }
        NodeEnum::FuncCall(func) => {
            let name = match func.funcname.last().map(node_enum).transpose()? {
                Some(NodeEnum::String(string)) => string.sval.to_ascii_lowercase(),
                _ => {
                    return Err(sql_pg_error(
                        "ORDER BY / HAVING references an unnamed function".to_string(),
                    ))
                }
            };
            if !matches!(name.as_str(), "count" | "sum" | "min" | "max" | "avg") {
                return Err(sql_pg_error(format!(
                    "ORDER BY / HAVING does not support the function \"{name}\""
                )));
            }
            Ok(name)
        }
        _ => Err(sql_pg_error(
            "ORDER BY / HAVING must reference a group column or an aggregate".to_string(),
        )),
    }
}

/// Parse a libpg_query `A_Const` literal to a `SqlValue` (HAVING right-hand side): integer -> Int4,
/// decimal/exponent -> Numeric. compare_sql_values handles the cross-type compare against the (Int8)
/// aggregate result.
fn aconst_to_sql_value(node: &Node) -> Result<SqlValue, ExecuteError> {
    match node_enum(node)? {
        NodeEnum::AConst(constant) => match &constant.val {
            Some(a_const::Val::Ival(integer)) => Ok(SqlValue::Int4(integer.ival)),
            Some(a_const::Val::Fval(float)) => Decimal128::parse(&float.fval)
                .map(SqlValue::Numeric)
                .ok_or_else(|| sql_pg_error(format!("malformed numeric literal: {}", float.fval))),
            _ => Err(sql_pg_error(
                "HAVING right-hand side must be an integer or numeric literal".to_string(),
            )),
        },
        _ => Err(sql_pg_error(
            "HAVING right-hand side must be a literal".to_string(),
        )),
    }
}

fn select_filter_op_from_token(token: &str) -> Result<SelectFilterOp, ExecuteError> {
    Ok(match token {
        "=" => SelectFilterOp::Eq,
        "<" => SelectFilterOp::Lt,
        "<=" => SelectFilterOp::Lte,
        ">" => SelectFilterOp::Gt,
        ">=" => SelectFilterOp::Gte,
        other => {
            return Err(sql_pg_error(format!(
                "HAVING operator \"{other}\" is not supported (use = < <= > >=)"
            )))
        }
    })
}

/// One HAVING comparison `<aggregate-or-key> <op> <literal>` -> a `SelectFilter` keyed by the result
/// column name (resolved by the executor against the materialized grouped columns).
fn aexpr_to_select_filter(a_expr: &AExpr, qualifier: &str) -> Result<SelectFilter, ExecuteError> {
    if a_expr.kind != AExprKind::AexprOp as i32 {
        return Err(sql_pg_error(
            "HAVING supports comparison operators only (no IN / LIKE / BETWEEN)".to_string(),
        ));
    }
    let op = select_filter_op_from_token(aexpr_op_token(a_expr)?)?;
    let lexpr = a_expr
        .lexpr
        .as_deref()
        .ok_or_else(|| sql_pg_error("HAVING comparison missing its left operand".to_string()))?;
    let rexpr = a_expr
        .rexpr
        .as_deref()
        .ok_or_else(|| sql_pg_error("HAVING comparison missing its right operand".to_string()))?;
    Ok(SelectFilter {
        column: result_column_name(lexpr, qualifier)?,
        op,
        value: aconst_to_sql_value(rexpr)?,
    })
}

/// Parse a `HAVING` clause to OR-of-ANDs `SelectFilter` groups: a bare comparison -> one group; top-
/// level `AND` -> one group of all-match filters; top-level `OR` -> one group per (comparison) operand.
/// One level deep -- nested AND/OR is rejected clearly rather than mis-parsed.
fn parse_having(
    having: Option<&Node>,
    qualifier: &str,
) -> Result<Vec<Vec<SelectFilter>>, ExecuteError> {
    let Some(node) = having else {
        return Ok(Vec::new());
    };
    match node_enum(node)? {
        NodeEnum::AExpr(a_expr) => Ok(vec![vec![aexpr_to_select_filter(a_expr, qualifier)?]]),
        NodeEnum::BoolExpr(bool_expr) if bool_expr.boolop == BoolExprType::AndExpr as i32 => {
            let filters = bool_expr
                .args
                .iter()
                .map(|arg| match node_enum(arg)? {
                    NodeEnum::AExpr(a) => aexpr_to_select_filter(a, qualifier),
                    _ => Err(sql_pg_error(
                        "HAVING AND supports flat comparisons only (no nested AND/OR)".to_string(),
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(vec![filters])
        }
        NodeEnum::BoolExpr(bool_expr) if bool_expr.boolop == BoolExprType::OrExpr as i32 => {
            let groups = bool_expr
                .args
                .iter()
                .map(|arg| match node_enum(arg)? {
                    NodeEnum::AExpr(a) => Ok(vec![aexpr_to_select_filter(a, qualifier)?]),
                    _ => Err(sql_pg_error(
                        "HAVING OR supports flat comparisons only (no nested AND/OR)".to_string(),
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(groups)
        }
        _ => Err(sql_pg_error(
            "HAVING must be a comparison, optionally combined with a single level of AND / OR"
                .to_string(),
        )),
    }
}

/// Parse a single-key `ORDER BY` (a group column or an aggregate, ASC/DESC). libpg_query `SortByDir`
/// 3 == DESC.
pub(super) fn parse_order_by(
    sort_clause: &[Node],
    qualifier: &str,
) -> Result<Vec<SelectOrder>, ExecuteError> {
    // Parse EVERY sort key, in significance order (`ORDER BY a ASC, b DESC, c`). Multi-key sorts run on
    // the GPU bitonic-sort path (the general Expr executor); the CPU/enumerated sort sites reject
    // `len() > 1`. NULLS FIRST/LAST is ignored (data is non-null until M3).
    let mut keys = Vec::with_capacity(sort_clause.len());
    for item in sort_clause {
        let NodeEnum::SortBy(sort_by) = node_enum(item)? else {
            return Err(sql_pg_error("malformed ORDER BY clause".to_string()));
        };
        if sort_by.sortby_dir == SortByDir::SortbyUsing as i32 || !sort_by.use_op.is_empty() {
            return Err(sql_pg_error(
                "ORDER BY USING is not supported by the GPU ordering path".to_string(),
            ));
        }
        let node = sort_by
            .node
            .as_deref()
            .ok_or_else(|| sql_pg_error("ORDER BY key has no expression".to_string()))?;
        let descending = sort_by.sortby_dir == SortByDir::SortbyDesc as i32;
        // A column reference OR an aggregate (`g`, `COUNT(*)`) resolves to a result-column name. An
        // arithmetic SORT EXPRESSION (`a+b`, `a*2`) does NOT -- emit a placeholder (empty column; a real
        // name is never empty) and let the general executor evaluate it (order_by_exprs, built parallel
        // to these keys in execute_resident_expr_select_sql).
        let column = result_column_name(node, qualifier).unwrap_or_default();
        keys.push(SelectOrder { column, descending });
    }
    Ok(keys)
}

pub(super) fn parse_order_by_null_placement(
    sort_clause: &[Node],
) -> Result<Vec<Option<bool>>, ExecuteError> {
    sort_clause
        .iter()
        .map(|item| {
            let NodeEnum::SortBy(sort_by) = node_enum(item)? else {
                return Err(sql_pg_error("malformed ORDER BY clause".to_string()));
            };
            Ok(
                if sort_by.sortby_nulls == SortByNulls::SortbyNullsFirst as i32 {
                    Some(true)
                } else if sort_by.sortby_nulls == SortByNulls::SortbyNullsLast as i32 {
                    Some(false)
                } else {
                    None
                },
            )
        })
        .collect()
}

/// Parse a `LIMIT` / `OFFSET` count -> a non-negative `usize`.
pub(super) fn parse_limit(limit: &Option<Box<Node>>) -> Result<Option<usize>, ExecuteError> {
    let Some(node) = limit.as_deref() else {
        return Ok(None);
    };
    match node_enum(node)? {
        NodeEnum::AConst(constant) => match &constant.val {
            Some(a_const::Val::Ival(integer)) if integer.ival >= 0 => {
                Ok(Some(integer.ival as usize))
            }
            _ => Err(sql_pg_error(
                "LIMIT / OFFSET must be a non-negative integer literal".to_string(),
            )),
        },
        _ => Err(sql_pg_error(
            "LIMIT / OFFSET must be an integer literal".to_string(),
        )),
    }
}

/// Wrap a SQL->Expr binding failure as the engine's standard `ApplyFailed` execution error (the same
/// surface the rest of the relational path uses), so callers handle it uniformly.
pub(super) fn sql_pg_error(message: String) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn libpg_projection_distinguishes_quoted_star_and_escapes_regex_prefix_for_like() {
        let stmt = parse_single_select(r#"SELECT "*", * FROM t"#).unwrap();
        let (select, _) = build_select_from_select_stmt(&stmt).unwrap();
        assert_eq!(
            select.projection,
            SelectProjection::Columns(vec![
                "*".to_string(),
                PROJECTION_WILDCARD_SENTINEL.to_string(),
            ])
        );

        assert_eq!(
            catalog_regex_pattern("^(catalog_regex_50%.*)$").unwrap(),
            (
                ResidentBinaryOp::Like,
                "catalog\\_regex\\_50\\%%".to_string(),
            )
        );
    }
}
