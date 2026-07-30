use super::select_lowering::{
    aexpr_op_token, map_predicate_node, node_enum, parse_limit, sql_pg_error,
};
use super::{
    relational_column_index, AExprKind, BoolExprType, ExecuteError, JoinColRef, JoinPlan,
    JoinProjItem, JoinRelationRef, JoinStep, JoinType, Node, NodeEnum, RelationalTable,
    ResidentBinaryOp, ResidentExpr, SelectStmt, SortByDir, SortByNulls,
};

/// True iff the FROM clause is a single explicit `JoinExpr` (an `a JOIN b ON ...`), routing to the M5
/// 2-relation path. (A comma join `FROM a, b` arrives as two from_clause entries -- not this -- and is
/// rejected by the single-table builder; explicit JOIN is required for the join path in this slice.)
pub(super) fn from_clause_is_join(stmt: &SelectStmt) -> bool {
    matches!(stmt.from_clause.as_slice(), [from] if matches!(from.node.as_ref(), Some(NodeEnum::JoinExpr(_))))
}

/// A JOIN column reference (ON operand or projection item): bare `col` or qualified `alias.col`. No
/// validation here (the executor resolves it against the two relations); rejects `*` / 3-part refs.
pub(super) fn parse_join_col_ref(node: &Node) -> Result<JoinColRef, ExecuteError> {
    let NodeEnum::ColumnRef(column_ref) = node_enum(node)? else {
        return Err(sql_pg_error(
            "a join ON/SELECT term must be a plain column reference (expressions / `*` / aggregates \
             are not on the join path yet)"
                .to_string(),
        ));
    };
    let parts = column_ref
        .fields
        .iter()
        .map(|field| match field.node.as_ref() {
            Some(NodeEnum::String(string)) => Ok(string.sval.as_str()),
            _ => Err(sql_pg_error(
                "a join column reference must be a name (got `*` or a non-name reference)"
                    .to_string(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    match parts.as_slice() {
        [column] => Ok(JoinColRef {
            qualifier: None,
            column: (*column).to_string(),
        }),
        [qualifier, column] => Ok(JoinColRef {
            qualifier: Some((*qualifier).to_string()),
            column: (*column).to_string(),
        }),
        _ => Err(sql_pg_error(
            "schema-qualified or multi-part join column references are not supported".to_string(),
        )),
    }
}

/// Parse a JOIN SELECT-list item: a plain column, or a `*` / `alias.*` star (expanded in the executor).
/// A star is an `A_Star` field in the last position: `*` = `[A_Star]`; `alias.*` = `[String, A_Star]`.
fn parse_join_proj_item(node: &Node) -> Result<JoinProjItem, ExecuteError> {
    if let NodeEnum::ColumnRef(column_ref) = node_enum(node)? {
        let last_is_star = matches!(
            column_ref
                .fields
                .last()
                .and_then(|field| field.node.as_ref()),
            Some(NodeEnum::AStar(_))
        );
        if last_is_star {
            return match column_ref.fields.as_slice() {
                [_star] => Ok(JoinProjItem::Star(None)),
                [qualifier, _star] => match qualifier.node.as_ref() {
                    Some(NodeEnum::String(string)) => {
                        Ok(JoinProjItem::Star(Some(string.sval.clone())))
                    }
                    _ => Err(sql_pg_error(
                        "a qualified `*` must be `alias.*`".to_string(),
                    )),
                },
                _ => Err(sql_pg_error(
                    "a schema-qualified `*` is not supported".to_string(),
                )),
            };
        }
    }
    parse_join_col_ref(node).map(JoinProjItem::Column)
}

/// The (relation name, qualifier) of a join side -- a base-table RangeVar (nested joins / subqueries
/// are a multi-way follow-up). The qualifier is the alias, else the relation name (mirrors the
/// single-table builder).
fn range_var_relation_name(
    range_var: &pg_query::protobuf::RangeVar,
) -> Result<String, ExecuteError> {
    if !range_var.catalogname.is_empty() {
        return Err(sql_pg_error(
            "cross-database JOIN relations are not supported".to_string(),
        ));
    }
    Ok(match range_var.schemaname.as_str() {
        "" | "public" => range_var.relname.clone(),
        schema => format!("{schema}.{}", range_var.relname),
    })
}

fn reject_join_range_column_aliases(
    range_var: &pg_query::protobuf::RangeVar,
) -> Result<(), ExecuteError> {
    if range_var
        .alias
        .as_ref()
        .is_some_and(|alias| !alias.colnames.is_empty())
    {
        return Err(sql_pg_error(
            "JOIN relation column-alias lists are not supported; use ordinary projected aliases"
                .to_string(),
        ));
    }
    Ok(())
}

fn join_side_name_alias(node: &Node) -> Result<(String, String, bool), ExecuteError> {
    let NodeEnum::RangeVar(range_var) = node_enum(node)? else {
        return Err(sql_pg_error(
            "each side of a JOIN must be a base table (nested joins / subqueries are a follow-up)"
                .to_string(),
        ));
    };
    reject_join_range_column_aliases(range_var)?;
    let table = range_var_relation_name(range_var)?;
    let alias = range_var
        .alias
        .as_ref()
        .map(|alias| alias.aliasname.clone())
        .unwrap_or_else(|| range_var.relname.clone());
    Ok((table, alias, range_var.schemaname == "public"))
}

/// Flatten ONE INNER-join node of a LEFT-DEEP chain into `relations` + `steps` (M5 J6). The left arg may
/// be a nested `JoinExpr` (recurse first, so relations end up in left-deep order) or a base table; the
/// right arg MUST be a base table (a right-nested `a JOIN (b JOIN c)` / bushy tree is a follow-up). Per
/// node: INNER only; the condition is ON (`a.k = b.k [AND ..]`), USING(cols) (desugared to qualified
/// conjuncts + recorded coalesce columns), or NATURAL (deferred to the executor). USING/NATURAL require a
/// base-table left arg (their multi-way coalescing is a follow-up). `steps[k]` (the k-th node folded in)
/// carries that node's condition; the executor figures out which operand is the newly joined relation.
fn flatten_join_chain(
    join: &pg_query::protobuf::JoinExpr,
    relations: &mut Vec<JoinRelationRef>,
    steps: &mut Vec<JoinStep>,
) -> Result<(), ExecuteError> {
    // INNER / LEFT / RIGHT / FULL (2-relation, ON only) -> the (outer_left, outer_right) flag pair.
    let (outer_left, outer_right) = if join.jointype == JoinType::JoinInner as i32 {
        (false, false)
    } else if join.jointype == JoinType::JoinLeft as i32 {
        (true, false)
    } else if join.jointype == JoinType::JoinRight as i32 {
        (false, true)
    } else if join.jointype == JoinType::JoinFull as i32 {
        (true, true)
    } else {
        return Err(sql_pg_error(
            "unsupported JOIN type (only INNER/LEFT/RIGHT/FULL are on the join path)".to_string(),
        ));
    };
    if (outer_left || outer_right) && (join.is_natural || !join.using_clause.is_empty()) {
        return Err(sql_pg_error(
            "an OUTER JOIN with NATURAL/USING is a follow-up; use OUTER JOIN ... ON".to_string(),
        ));
    }
    let larg = join
        .larg
        .as_deref()
        .ok_or_else(|| sql_pg_error("JOIN missing its left relation".to_string()))?;
    let rarg = join
        .rarg
        .as_deref()
        .ok_or_else(|| sql_pg_error("JOIN missing its right relation".to_string()))?;
    // Process the left arg (recurse if nested) and capture its alias if it is a base table -- USING needs
    // it to build qualified conjuncts, and USING/NATURAL are 2-relation only (left must be a base table).
    let left_alias: Option<String> = match node_enum(larg)? {
        NodeEnum::JoinExpr(inner) => {
            flatten_join_chain(inner, relations, steps)?;
            None
        }
        _ => {
            let (table, alias, public_only) = join_side_name_alias(larg)?;
            relations.push(JoinRelationRef {
                table,
                alias: alias.clone(),
                public_only,
            });
            Some(alias)
        }
    };
    let (right_table, right_alias, right_public_only) = join_side_name_alias(rarg)?;
    relations.push(JoinRelationRef {
        table: right_table,
        alias: right_alias.clone(),
        public_only: right_public_only,
    });
    // USING/NATURAL require a base-table left arg (their multi-way coalescing is a follow-up).
    let multi_way_using = || {
        sql_pg_error(
            "NATURAL / USING on a multi-way join is a follow-up; the left side of a NATURAL/USING join \
             must be a single base table"
                .to_string(),
        )
    };
    let step = if join.is_natural {
        // NATURAL: the executor joins on (and coalesces) the relations' common column names.
        if left_alias.is_none() {
            return Err(multi_way_using());
        }
        JoinStep {
            conjuncts: Vec::new(),
            natural: true,
            coalesce: Vec::new(),
            outer_left,
            outer_right,
        }
    } else if !join.using_clause.is_empty() {
        // USING(cols): desugar to qualified `left.c = right.c` conjuncts + record the coalesce columns.
        let left_alias = left_alias.ok_or_else(multi_way_using)?;
        let cols = parse_using_columns(&join.using_clause)?;
        let conjuncts = cols
            .iter()
            .map(|c| {
                (
                    JoinColRef {
                        qualifier: Some(left_alias.clone()),
                        column: c.clone(),
                    },
                    JoinColRef {
                        qualifier: Some(right_alias.clone()),
                        column: c.clone(),
                    },
                )
            })
            .collect();
        JoinStep {
            conjuncts,
            natural: false,
            coalesce: cols,
            outer_left,
            outer_right,
        }
    } else {
        // ON: a single `=` equi-join, or a top-level AND of `=` equi-joins (a composite key).
        let quals = join
            .quals
            .as_deref()
            .ok_or_else(|| sql_pg_error("INNER JOIN requires an ON condition".to_string()))?;
        JoinStep {
            conjuncts: parse_on_conjuncts(quals)?,
            natural: false,
            coalesce: Vec::new(),
            outer_left,
            outer_right,
        }
    };
    steps.push(step);
    Ok(())
}

/// Parse a USING column list (`USING (a, b)`) into the column names. Each entry is a String node.
fn parse_using_columns(using_clause: &[Node]) -> Result<Vec<String>, ExecuteError> {
    if using_clause.is_empty() {
        return Err(sql_pg_error(
            "USING requires at least one column".to_string(),
        ));
    }
    using_clause
        .iter()
        .map(|node| match node_enum(node)? {
            NodeEnum::String(string) => Ok(string.sval.clone()),
            _ => Err(sql_pg_error(
                "a USING column must be a plain column name".to_string(),
            )),
        })
        .collect()
}

/// Parse a single `=` equi-join ON conjunct (`a.k = b.k`) into its two column operands.
fn parse_equi_conjunct(node: &Node) -> Result<(JoinColRef, JoinColRef), ExecuteError> {
    let NodeEnum::AExpr(aexpr) = node_enum(node)? else {
        return Err(sql_pg_error(
            "a join ON condition must be `a.k = b.k` equalities (AND-combined for a composite key); \
             non-equi / function / subquery ON terms are a follow-up"
                .to_string(),
        ));
    };
    if aexpr.kind != AExprKind::AexprOp as i32 || aexpr_op_token(aexpr)? != "=" {
        return Err(sql_pg_error(
            "the join ON condition must be an equality (=) between two columns".to_string(),
        ));
    }
    let lexpr = aexpr
        .lexpr
        .as_deref()
        .ok_or_else(|| sql_pg_error("malformed join ON condition".to_string()))?;
    let rexpr = aexpr
        .rexpr
        .as_deref()
        .ok_or_else(|| sql_pg_error("malformed join ON condition".to_string()))?;
    Ok((parse_join_col_ref(lexpr)?, parse_join_col_ref(rexpr)?))
}

/// Parse a JOIN ON into its equi-join conjuncts: a single `=` (one conjunct), or a top-level `AND` chain
/// of `=` (a composite key, ≥2 conjuncts). `OR`/`NOT` in an ON is a follow-up (a clean error). The
/// executor packs a 2-conjunct key into one i64 for the hash join; >2 conjuncts (wider than 64 bits) are
/// validated/rejected there.
fn parse_on_conjuncts(quals: &Node) -> Result<Vec<(JoinColRef, JoinColRef)>, ExecuteError> {
    let mut terms = Vec::new();
    collect_and_conjuncts(quals, &mut terms)?;
    let mut equi = Vec::new();
    for term in terms {
        match parse_equi_conjunct(term) {
            Ok(conjunct) => equi.push(conjunct),
            Err(_) if on_local_filter_has_supported_syntax(term) => {}
            Err(error) => return Err(error),
        }
    }
    if equi.is_empty() {
        return Err(sql_pg_error(
            "a join ON requires at least one equality between the joined relations".to_string(),
        ));
    }
    Ok(equi)
}

fn on_local_filter_has_supported_syntax(node: &Node) -> bool {
    let Ok(NodeEnum::AExpr(expression)) = node_enum(node) else {
        return false;
    };
    if expression.kind != AExprKind::AexprOp as i32
        || !matches!(
            aexpr_op_token(expression),
            Ok("=" | "<>" | "<" | "<=" | ">" | ">=")
        )
    {
        return false;
    }
    matches!(
        (
            expression
                .lexpr
                .as_deref()
                .and_then(|node| node.node.as_ref()),
            expression
                .rexpr
                .as_deref()
                .and_then(|node| node.node.as_ref()),
        ),
        (Some(NodeEnum::ColumnRef(_)), Some(NodeEnum::AConst(_)))
            | (Some(NodeEnum::AConst(_)), Some(NodeEnum::ColumnRef(_)))
    )
}

/// Parse `SELECT <cols> FROM a JOIN b ON .. [JOIN c ON ..]` (libpg_query) into a [`JoinPlan`] (M5). Scope:
/// a LEFT-DEEP chain of INNER JOINs, each with a single `=` equi-join conjunct between two base tables,
/// plain-column / `*` projection, and NO GROUP BY/HAVING/ORDER BY/LIMIT/DISTINCT (each a clean "not on the
/// join path yet"). WHERE is supported (split per-relation in the entry). The executor resolves the ON
/// operands + projection to relations/columns, validates the key types, and pipelines the chain.
pub(super) fn build_join_plan(stmt: &SelectStmt) -> Result<JoinPlan, ExecuteError> {
    reject_unsupported_join_clauses(stmt)?;
    let (relations, steps) = build_join_structure(stmt)?;
    let (projection, projection_aliases) = parse_join_projection(stmt)?;
    let (order_by, order_by_nulls_first, limit, offset) = parse_join_order_by_limit(stmt)?;
    Ok(JoinPlan {
        relations,
        steps,
        projection,
        distinct: !stmt.distinct_clause.is_empty(),
        projection_aliases,
        order_by,
        order_by_nulls_first,
        limit,
        offset,
    })
}

pub(super) fn build_join_plan_with_projection(
    stmt: &SelectStmt,
    projection: Vec<JoinProjItem>,
    projection_aliases: Vec<Option<String>>,
    order_by: Vec<(JoinColRef, bool)>,
    order_by_nulls_first: Vec<Option<bool>>,
) -> Result<JoinPlan, ExecuteError> {
    reject_unsupported_join_clauses(stmt)?;
    let (relations, steps) = build_join_structure(stmt)?;
    let limit = parse_limit(&stmt.limit_count)?;
    let offset = parse_limit(&stmt.limit_offset)?;
    Ok(JoinPlan {
        relations,
        steps,
        projection,
        distinct: !stmt.distinct_clause.is_empty(),
        projection_aliases,
        order_by,
        order_by_nulls_first,
        limit,
        offset,
    })
}

fn build_join_structure(
    stmt: &SelectStmt,
) -> Result<(Vec<JoinRelationRef>, Vec<JoinStep>), ExecuteError> {
    let [from] = stmt.from_clause.as_slice() else {
        return Err(sql_pg_error(
            "expected a single JOIN in the FROM clause".to_string(),
        ));
    };
    let NodeEnum::JoinExpr(join) = node_enum(from)? else {
        return Err(sql_pg_error(
            "expected a JOIN in the FROM clause".to_string(),
        ));
    };
    let mut relations: Vec<JoinRelationRef> = Vec::new();
    let mut steps: Vec<JoinStep> = Vec::new();
    flatten_join_chain(join, &mut relations, &mut steps)?;
    // USING/NATURAL coalescing is supported for a single 2-relation join only (its interaction with the
    // multi-way `*` expansion is a follow-up). A USING/NATURAL node only arises with a base-table left
    // arg, so any chain longer than 2 relations that carries one is multi-way -> reject.
    if relations.len() > 2 && steps.iter().any(|s| s.natural || !s.coalesce.is_empty()) {
        return Err(sql_pg_error(
            "NATURAL / USING in a multi-way join is a follow-up; it is supported for a 2-relation join \
             only (use explicit ON in a multi-way chain)"
                .to_string(),
        ));
    }
    Ok((relations, steps))
}

/// Reject the clauses not on the join path yet (GROUP BY / HAVING / DISTINCT ON / window / WITH).
/// Plain DISTINCT is a GPU sort plus adjacent-equality compaction over the materialized projection.
/// WHERE is supported (split per-relation in the entry); ORDER BY / LIMIT / OFFSET are parsed by
/// `parse_join_order_by_limit` and applied to the result (a GPU sort then a slice). Shared by both joins.
pub(super) fn reject_unsupported_join_clauses(stmt: &SelectStmt) -> Result<(), ExecuteError> {
    let unsupported = [
        (!stmt.group_clause.is_empty(), "GROUP BY"),
        (stmt.having_clause.is_some(), "HAVING"),
        (
            !(stmt.distinct_clause.is_empty()
                || stmt.distinct_clause.len() == 1 && stmt.distinct_clause[0].node.is_none()),
            "DISTINCT ON",
        ),
        (!stmt.window_clause.is_empty(), "window functions"),
        (stmt.with_clause.is_some(), "WITH / CTEs"),
    ];
    if let Some((_, clause)) = unsupported.iter().find(|(present, _)| *present) {
        return Err(sql_pg_error(format!(
            "{clause} on a JOIN is not on the general GPU executor's join path yet"
        )));
    }
    Ok(())
}

/// Parse a join's ORDER BY / LIMIT / OFFSET. ORDER BY keys are PLAIN columns only (qualified or not -- the
/// executor resolves them to a projected result column and sorts on the GPU); an arithmetic / aggregate
/// sort expression on the join path is a follow-up. LIMIT / OFFSET are non-negative integer literals.
/// Explicit NULLS FIRST/LAST is captured per key (`nulls_first`) and honored on-device by the join-result
/// GPU sort. Shared by explicit + comma joins.
#[allow(clippy::type_complexity)] // (keys, nulls_first, LIMIT, OFFSET) -- a plain tuple, naming it adds noise
pub(super) fn parse_join_order_by_limit(
    stmt: &SelectStmt,
) -> Result<
    (
        Vec<(JoinColRef, bool)>,
        Vec<Option<bool>>,
        Option<usize>,
        Option<usize>,
    ),
    ExecuteError,
> {
    let mut order_by = Vec::with_capacity(stmt.sort_clause.len());
    let mut nulls_first = Vec::with_capacity(stmt.sort_clause.len());
    for item in &stmt.sort_clause {
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
        order_by.push((parse_join_col_ref(node)?, descending));
        nulls_first.push(
            if sort_by.sortby_nulls == SortByNulls::SortbyNullsFirst as i32 {
                Some(true)
            } else if sort_by.sortby_nulls == SortByNulls::SortbyNullsLast as i32 {
                Some(false)
            } else {
                None
            },
        );
    }
    let limit = parse_limit(&stmt.limit_count)?;
    let offset = parse_limit(&stmt.limit_offset)?;
    Ok((order_by, nulls_first, limit, offset))
}

/// Parse the SELECT target list into join projection items (plain columns / `*` / `alias.*`); non-empty.
/// Shared by explicit + comma joins.
pub(super) fn parse_join_projection(
    stmt: &SelectStmt,
) -> Result<(Vec<JoinProjItem>, Vec<Option<String>>), ExecuteError> {
    if stmt.target_list.is_empty() {
        return Err(sql_pg_error(
            "a join SELECT must project at least one column".to_string(),
        ));
    }
    let mut projection = Vec::with_capacity(stmt.target_list.len());
    let mut aliases = Vec::with_capacity(stmt.target_list.len());
    for target in &stmt.target_list {
        let NodeEnum::ResTarget(res_target) = node_enum(target)? else {
            return Err(sql_pg_error("malformed join SELECT target".to_string()));
        };
        let val = res_target
            .val
            .as_deref()
            .ok_or_else(|| sql_pg_error("join SELECT target has no expression".to_string()))?;
        let item = parse_join_proj_item(val)?;
        if matches!(item, JoinProjItem::Star(_)) && !res_target.name.is_empty() {
            return Err(sql_pg_error(
                "a join star cannot have a column alias".to_string(),
            ));
        }
        projection.push(item);
        aliases.push((!res_target.name.is_empty()).then(|| res_target.name.clone()));
    }
    Ok((projection, aliases))
}

/// Detect a comma join (`FROM a, b[, c]`) and lift out its relations. Returns `None` (not this path) for
/// a single FROM relation or any non-base-table FROM entry (an explicit `JoinExpr` -- handled by
/// `build_join_plan` -- or a subquery; a comma list MIXING those is a follow-up). Each entry's qualifier
/// is its alias, else the relation name (mirrors the explicit-JOIN builder).
pub(super) fn comma_join_relations(
    stmt: &SelectStmt,
) -> Result<Option<Vec<JoinRelationRef>>, ExecuteError> {
    if stmt.from_clause.len() < 2 {
        return Ok(None);
    }
    let mut relations = Vec::with_capacity(stmt.from_clause.len());
    for from in &stmt.from_clause {
        let Some(NodeEnum::RangeVar(range_var)) = from.node.as_ref() else {
            return Ok(None);
        };
        reject_join_range_column_aliases(range_var)?;
        let table = range_var_relation_name(range_var)?;
        let alias = range_var
            .alias
            .as_ref()
            .map(|alias| alias.aliasname.clone())
            .unwrap_or_else(|| range_var.relname.clone());
        relations.push(JoinRelationRef {
            table,
            alias,
            public_only: range_var.schemaname == "public",
        });
    }
    Ok(Some(relations))
}

/// Resolve a column reference to its relation index among `relations`/`tables` (parallel): a qualifier
/// must name exactly one relation (and the column must exist in it); an unqualified column must be in
/// exactly one (PG's "ambiguous" / "does not exist"). Used to find a comma-join WHERE equi-join's endpoints.
fn which_relation(
    c: &JoinColRef,
    relations: &[JoinRelationRef],
    tables: &[RelationalTable],
) -> Result<usize, ExecuteError> {
    match &c.qualifier {
        Some(q) => {
            let i = relations
                .iter()
                .position(|r| &r.alias == q)
                .ok_or_else(|| {
                    sql_pg_error(format!("missing FROM-clause entry for table \"{q}\""))
                })?;
            relational_column_index(&tables[i], &c.column)?;
            Ok(i)
        }
        None => {
            let mut found: Option<usize> = None;
            for (i, table) in tables.iter().enumerate() {
                if relational_column_index(table, &c.column).is_ok() {
                    if found.is_some() {
                        return Err(sql_pg_error(format!(
                            "column reference \"{}\" is ambiguous",
                            c.column
                        )));
                    }
                    found = Some(i);
                }
            }
            found.ok_or_else(|| sql_pg_error(format!("column \"{}\" does not exist", c.column)))
        }
    }
}

/// Plan a comma join (M5 J6): partition the WHERE into per-relation filters + cross-relation equi-join
/// EDGES, then derive the left-deep `steps` (in FROM order) + per-relation predicates. Each top-level AND
/// conjunct is either (a) mapped to exactly one relation -> a filter, or (b) an `=` between columns of two
/// DIFFERENT relations -> an edge folded into the LATER relation's step (so it joins to an earlier one).
/// Each non-first relation must have >=1 edge to an earlier relation (else the graph is disconnected -- a
/// cartesian/cross join, a follow-up). >2 edges into one relation (a composite key wider than 64 bits) is a
/// follow-up. This reuses the same `JoinStep`/executor as explicit JOINs (incl. composite 2-edge keys).
pub(super) fn plan_comma_join_where(
    where_clause: Option<&Node>,
    relations: &[JoinRelationRef],
    tables: &[RelationalTable],
    aliases: &[&str],
    catalog: &super::CatalogSnapshot,
) -> Result<(Vec<JoinStep>, Vec<Option<ResidentExpr>>), ExecuteError> {
    let n = relations.len();
    let mut conjuncts: Vec<&Node> = Vec::new();
    if let Some(where_node) = where_clause {
        collect_and_conjuncts(where_node, &mut conjuncts)?;
    }
    let mut filters: Vec<Vec<ResidentExpr>> = (0..n).map(|_| Vec::new()).collect();
    // edges[i] = the equi-join conjuncts that fold relation i into the accumulated set (i.e. that connect
    // relation i to some relation < i). edges[0] stays empty.
    let mut edges: Vec<Vec<(JoinColRef, JoinColRef)>> = (0..n).map(|_| Vec::new()).collect();
    for conjunct in conjuncts {
        let mapped = tables
            .iter()
            .zip(aliases)
            .enumerate()
            .filter_map(|(i, (table, alias))| {
                map_predicate_node(conjunct, table, alias, catalog)
                    .ok()
                    .map(|expr| (i, expr))
            })
            .collect::<Vec<_>>();
        match mapped.as_slice() {
            [(i, expr)] => {
                filters[*i].push(expr.clone());
                continue;
            }
            [_, _, ..] => return Err(sql_pg_error(
                "an unqualified comma-join WHERE column is ambiguous; qualify it with its relation"
                    .to_string(),
            )),
            [] => {}
        }
        // Otherwise it must be a cross-relation `colA = colB` equi-join edge.
        let (on_a, on_b) = parse_equi_conjunct(conjunct)?;
        let ra = which_relation(&on_a, relations, tables)?;
        let rb = which_relation(&on_b, relations, tables)?;
        if ra == rb {
            return Err(sql_pg_error(
                "a comma-join WHERE conjunct must reference one relation (a filter) or two different \
                 relations (an equi-join); other cross-relation predicates are a follow-up"
                    .to_string(),
            ));
        }
        edges[ra.max(rb)].push((on_a, on_b));
    }
    let mut steps = Vec::with_capacity(n - 1);
    for (i, relation_edges) in edges.into_iter().enumerate().skip(1) {
        if relation_edges.is_empty() {
            return Err(sql_pg_error(format!(
                "relation \"{}\" has no equi-join condition to an earlier relation (a comma cross-join / \
                 cartesian product is a follow-up; add `WHERE a.k = b.k` or use an explicit JOIN)",
                relations[i].alias
            )));
        }
        if relation_edges.len() > 2 {
            return Err(sql_pg_error(
                "more than 2 join conditions into one relation (a composite key wider than 64 bits) is a \
                 follow-up"
                    .to_string(),
            ));
        }
        steps.push(JoinStep {
            conjuncts: relation_edges,
            natural: false,
            coalesce: Vec::new(),
            // A comma join (`FROM a, b WHERE a.k=b.k`) is always INNER.
            outer_left: false,
            outer_right: false,
        });
    }
    let predicates = filters
        .into_iter()
        .map(|conjuncts| {
            conjuncts
                .into_iter()
                .reduce(|acc, expr| ResidentExpr::Binary {
                    op: ResidentBinaryOp::And,
                    lhs: Box::new(acc),
                    rhs: Box::new(expr),
                })
        })
        .collect();
    Ok((steps, predicates))
}

/// Split a join WHERE into per-relation predicates, one per relation in `tables`/`aliases` (M5 J3/J6).
/// Each top-level AND conjunct must reference exactly ONE relation; it is mapped (via
/// `map_predicate_node`, which handles all literal types + arithmetic/comparison/AND-OR) against the
/// only relation it resolves against, and that relation's conjuncts are AND-folded into its predicate (the executor GPU-evaluates
/// it to pre-filter before the join -- inner-join semantics are filter-commutative on per-side
/// predicates). A conjunct referencing NO single relation (a cross-relation predicate beyond the ON, e.g.
/// `a.x > b.y`) is a follow-up -> a clean error. Returns a Vec parallel to `tables`.
pub(super) fn split_join_where(
    where_node: &Node,
    tables: &[RelationalTable],
    aliases: &[&str],
    catalog: &super::CatalogSnapshot,
) -> Result<Vec<Option<ResidentExpr>>, ExecuteError> {
    let mut conjuncts: Vec<&Node> = Vec::new();
    collect_and_conjuncts(where_node, &mut conjuncts)?;
    let mut per_relation: Vec<Vec<ResidentExpr>> = (0..tables.len()).map(|_| Vec::new()).collect();
    for conjunct in conjuncts {
        let mapped = tables
            .iter()
            .zip(aliases)
            .enumerate()
            .filter_map(|(i, (table, alias))| {
                map_predicate_node(conjunct, table, alias, catalog)
                    .ok()
                    .map(|expr| (i, expr))
            })
            .collect::<Vec<_>>();
        match mapped.as_slice() {
            [(i, expr)] => per_relation[*i].push(expr.clone()),
            [] => {
                return Err(sql_pg_error(
                    "a join WHERE conjunct must reference exactly one relation (a cross-relation \
                     predicate beyond the ON is a follow-up)"
                        .to_string(),
                ))
            }
            [_, _, ..] => {
                return Err(sql_pg_error(
                    "an unqualified join WHERE column is ambiguous; qualify it with its relation"
                        .to_string(),
                ))
            }
        }
    }
    Ok(per_relation
        .into_iter()
        .map(|conjuncts| {
            conjuncts
                .into_iter()
                .reduce(|acc, expr| ResidentExpr::Binary {
                    op: ResidentBinaryOp::And,
                    lhs: Box::new(acc),
                    rhs: Box::new(expr),
                })
        })
        .collect())
}

/// Lower single-relation ON conjuncts into the same per-relation GPU predicates as WHERE. This is
/// semantics-preserving for INNER joins and for the non-preserved side of an OUTER join. Predicates
/// on a preserved side remain rejected instead of being incorrectly pushed below the join.
pub(super) fn split_join_on_local_filters(
    stmt: &SelectStmt,
    tables: &[RelationalTable],
    aliases: &[&str],
    catalog: &super::CatalogSnapshot,
) -> Result<Vec<Option<ResidentExpr>>, ExecuteError> {
    let [from] = stmt.from_clause.as_slice() else {
        return Err(sql_pg_error(
            "expected one explicit JOIN while lowering ON filters".to_string(),
        ));
    };
    let NodeEnum::JoinExpr(join) = node_enum(from)? else {
        return Err(sql_pg_error(
            "expected an explicit JOIN while lowering ON filters".to_string(),
        ));
    };
    let mut filters: Vec<Vec<ResidentExpr>> = (0..tables.len()).map(|_| Vec::new()).collect();
    collect_join_on_local_filters(join, tables, aliases, catalog, &mut filters)?;
    Ok(filters
        .into_iter()
        .map(|filters| {
            filters
                .into_iter()
                .reduce(|left, right| ResidentExpr::Binary {
                    op: ResidentBinaryOp::And,
                    lhs: Box::new(left),
                    rhs: Box::new(right),
                })
        })
        .collect())
}

fn collect_join_on_local_filters(
    join: &pg_query::protobuf::JoinExpr,
    tables: &[RelationalTable],
    aliases: &[&str],
    catalog: &super::CatalogSnapshot,
    filters: &mut [Vec<ResidentExpr>],
) -> Result<(), ExecuteError> {
    if let Some(NodeEnum::JoinExpr(inner)) =
        join.larg.as_deref().and_then(|node| node.node.as_ref())
    {
        collect_join_on_local_filters(inner, tables, aliases, catalog, filters)?;
    }
    // USING and NATURAL joins carry their equality semantics in `using_clause` / `is_natural`,
    // not `quals`. They have no independent ON-local predicates to push down.
    if join.is_natural || !join.using_clause.is_empty() {
        return Ok(());
    }
    let right = join
        .rarg
        .as_deref()
        .ok_or_else(|| sql_pg_error("JOIN missing its right relation".to_string()))?;
    let (_, right_alias, _) = join_side_name_alias(right)?;
    let right_index = aliases
        .iter()
        .position(|alias| *alias == right_alias)
        .ok_or_else(|| sql_pg_error("JOIN right relation lost its bound alias".to_string()))?;
    let quals = join
        .quals
        .as_deref()
        .ok_or_else(|| sql_pg_error("JOIN requires an ON condition".to_string()))?;
    let mut conjuncts = Vec::new();
    collect_and_conjuncts(quals, &mut conjuncts)?;
    for conjunct in conjuncts {
        if parse_equi_conjunct(conjunct).is_ok() {
            continue;
        }
        let mapped = tables
            .iter()
            .zip(aliases)
            .enumerate()
            .filter_map(|(index, (table, alias))| {
                map_predicate_node(conjunct, table, alias, catalog)
                    .ok()
                    .map(|predicate| (index, predicate))
            })
            .collect::<Vec<_>>();
        let [(index, predicate)] = mapped.as_slice() else {
            return Err(sql_pg_error(
                "a local join ON predicate must resolve against exactly one relation".to_string(),
            ));
        };
        let pushdown_safe = if join.jointype == JoinType::JoinInner as i32 {
            true
        } else if join.jointype == JoinType::JoinLeft as i32 {
            *index == right_index
        } else if join.jointype == JoinType::JoinRight as i32 {
            *index != right_index
        } else {
            false
        };
        if !pushdown_safe {
            return Err(sql_pg_error(
                "an ON predicate on the preserved side of an outer join cannot be pushed below the GPU join"
                    .to_string(),
            ));
        }
        filters[*index].push(predicate.clone());
    }
    Ok(())
}

/// Flatten a top-level chain of `AND`s into individual conjuncts; any other node is a single conjunct.
fn collect_and_conjuncts<'a>(node: &'a Node, out: &mut Vec<&'a Node>) -> Result<(), ExecuteError> {
    if let NodeEnum::BoolExpr(bool_expr) = node_enum(node)? {
        if bool_expr.boolop == BoolExprType::AndExpr as i32 {
            for arg in &bool_expr.args {
                collect_and_conjuncts(arg, out)?;
            }
            return Ok(());
        }
    }
    out.push(node);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_sql_pg::parse_single_select;

    #[test]
    fn join_relation_identity_normalizes_only_public_and_rejects_column_alias_lists() {
        let explicit = parse_single_select(
            "SELECT l.id FROM public.left_table l \
             JOIN public.right_table r ON l.id = r.id",
        )
        .unwrap();
        let plan = build_join_plan(&explicit).unwrap();
        assert_eq!(plan.relations[0].table, "left_table");
        assert_eq!(plan.relations[1].table, "right_table");
        assert!(plan.relations.iter().all(|relation| relation.public_only));

        let comma = parse_single_select(
            "SELECT l.id FROM public.left_table l, public.right_table r \
             WHERE l.id = r.id",
        )
        .unwrap();
        let relations = comma_join_relations(&comma).unwrap().unwrap();
        assert_eq!(relations[0].table, "left_table");
        assert_eq!(relations[1].table, "right_table");
        assert!(relations.iter().all(|relation| relation.public_only));

        let qualified = parse_single_select(
            "SELECT l.oid FROM evil.pg_class l \
             JOIN pg_catalog.pg_namespace n ON l.oid = n.oid",
        )
        .unwrap();
        let plan = build_join_plan(&qualified).unwrap();
        assert_eq!(plan.relations[0].table, "evil.pg_class");
        assert_eq!(plan.relations[1].table, "pg_catalog.pg_namespace");
        assert!(plan.relations.iter().all(|relation| !relation.public_only));

        for sql in [
            "SELECT l.x FROM public.left_table AS l(x) \
             JOIN public.right_table r ON l.x = r.id",
            "SELECT l.x FROM public.left_table AS l(x), public.right_table r \
             WHERE l.x = r.id",
        ] {
            let stmt = parse_single_select(sql).unwrap();
            let error = if from_clause_is_join(&stmt) {
                build_join_plan(&stmt).unwrap_err()
            } else {
                comma_join_relations(&stmt).unwrap_err()
            };
            assert!(error.to_string().contains("column-alias lists"), "{error}");
        }
    }

    #[test]
    fn using_and_natural_joins_do_not_require_an_on_predicate_for_local_filter_lowering() {
        for sql in [
            "SELECT * FROM left_table l JOIN right_table r USING (id)",
            "SELECT * FROM left_table l NATURAL JOIN right_table r",
        ] {
            let stmt = parse_single_select(sql).unwrap();
            assert!(
                split_join_on_local_filters(
                    &stmt,
                    &[],
                    &[],
                    &crate::engine_state::CatalogSnapshot::default(),
                )
                .unwrap()
                .is_empty(),
                "{sql}"
            );
        }
    }
}
