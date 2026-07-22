//! SQL -> `ResidentExpr` binding via libpg_query (Charter rule 2;
//! `docs/architecture/18-sql-to-expr-handoff.md`). The `pg_query` crate vendors libpg_query — the
//! real PostgreSQL parser — so a SQL string is parsed by Postgres's own grammar and then walked into
//! the engine's general `ResidentExpr` IR (`engine_expr_ir`, re-exported by the `engine_expr` facade) and
//! executed by the general GPU executor.
//! This is the "close the loop" path: SQL text -> general GPU execution. It deliberately does NOT
//! extend the hand-rolled `gpu_db_sql` parser and is NOT a catalog of query shapes — coverage grows by
//! node / type / operator (Charter rule 2). Today it binds a single-table `SELECT ... WHERE` over int4
//! and int8 columns with arithmetic (`+ - *`, checked overflow), comparisons (`= <> < <= > >=`),
//! column-vs-column, and `AND`/`OR`; richer types (numeric/text/bool) and operators are the next
//! slices. Anything the mapper cannot represent is a hard error — never a silent mis-answer.

use super::*;

use pg_query::protobuf::{
    a_const, AExpr, AExprKind, BoolExpr, BoolExprType, ColumnRef, JoinType, Node, SelectStmt,
    SetOperation, SortByDir, SortByNulls, SubLinkType,
};
use pg_query::NodeEnum;

use crate::engine_expr::{
    like_pattern_for_literal_prefix, JoinColRef, JoinPlan, JoinProjItem, JoinRelationRef, JoinStep,
    ResidentBinaryOp, ResidentExecSource, ResidentExpr,
};
use gpu_db_sql::{canonicalize_sql_for_exact_match, SelectFilter, SelectFilterOp, SelectOrder};

mod catalog_range_alias;
mod catalog_visibility;
mod join_lowering;
mod select_lowering;
mod statement_snapshot;
mod window_lowering;

use catalog_range_alias::{
    apply_catalog_range_column_aliases, catalog_range_relation_key,
    from_node_has_column_alias_list, from_node_is_join, rank_window_relation_binding,
};
use join_lowering::{
    build_join_plan, build_join_plan_with_projection, comma_join_relations, from_clause_is_join,
    parse_join_col_ref, parse_join_order_by_limit, parse_join_projection, plan_comma_join_where,
    reject_unsupported_join_clauses, split_join_on_local_filters, split_join_where,
};
pub(crate) use select_lowering::parse_single_select;
use select_lowering::{
    aexpr_op_token, build_select_from_select_stmt, map_predicate_node, node_enum, parse_limit,
    parse_order_by, parse_order_by_null_placement, resolve_column_name, sql_pg_error,
};
use statement_snapshot::capture_general_select_statement_snapshot;
use window_lowering::{resolve_rank_window_def, select_has_inline_window};

#[derive(Clone, Copy, PartialEq, Eq)]
enum GpuRankWindowKind {
    RowNumber,
    Rank,
    DenseRank,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GpuOffsetWindowKind {
    Lag,
    Lead,
}

enum GpuRankTarget {
    Column {
        source: String,
        output: String,
    },
    Window(GpuRankWindowKind, String),
    OffsetWindow {
        kind: GpuOffsetWindowKind,
        source: String,
        offset: u32,
        output: String,
    },
}

mod catalog_empty;
mod catalog_presentation;
pub(crate) mod pg_dump_catalog;

use catalog_presentation::{
    apply_catalog_projection_metadata, catalog_join_projection_plan,
    catalog_single_projection_plan, scalar_aggregate_alias_presentation,
    select_tree_uses_synthesized_catalog,
};

/// A resident user relation carries no host rows; a synthesized catalog relation carries the
/// transient rows that will be uploaded for the same GPU operator path.
type BoundJoinRelation = (RelationalTable, Option<Vec<Vec<SqlValue>>>);

fn bind_join_relation(
    relation: &JoinRelationRef,
    catalog: &CatalogSnapshot,
) -> Result<BoundJoinRelation, ExecuteError> {
    let name = relation.table.as_str();
    let user = |name: &str| {
        catalog
            .relational_catalog
            .get(name)
            .filter(|table| table.schema == "public")
            .cloned()
            .map(|table| (table, None))
    };
    let synthesized = |name: &str| {
        synthesize_catalog_relation(name, catalog).map(|(table, rows)| (table, Some(rows)))
    };
    let bound = if relation.public_only {
        user(name)
    } else if let Some(name) = name.strip_prefix("public.") {
        user(name)
    } else if name.starts_with("pg_catalog.") || name.starts_with("information_schema.") {
        synthesized(name)
    } else if name.contains('.') {
        None
    } else if public_relation_name_exists(catalog, name) {
        user(name)
    } else {
        synthesized(name)
    };
    bound.ok_or_else(|| sql_pg_error(format!("relation \"{name}\" does not exist")))
}

impl Engine {
    pub(crate) fn execute_resident_expr_select_sql_scoped(
        &self,
        sql: &str,
        after_snapshot: impl FnOnce(),
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let stmt = parse_single_select(sql)?;
        let statement_snapshot = capture_general_select_statement_snapshot(self, &stmt)?;
        let _statement_scope =
            statement_snapshot.map(|snapshot| self.enter_transaction_read(snapshot));
        after_snapshot();
        if let Some(result) = self.execute_pg_dump_catalog_route_if_applicable(sql)? {
            return Ok(result);
        }
        if select_has_inline_window(&stmt)? {
            return self.execute_gpu_rank_window_select(&stmt);
        }
        if let Some(result) = self.execute_empty_catalog_select_if_applicable(&stmt)? {
            return Ok(result);
        }
        // A JOIN in the FROM clause routes to the dedicated 2-relation inner-equi-join path (M5). The
        // single-table path below is byte-identical for a non-join query.
        if from_clause_is_join(&stmt) {
            // Route selection and relation binding share one exact catalog boundary. The presentation
            // extension is catalog-only; a user join keeps its original lowering error instead of
            // being reinterpreted by compatibility metadata rules.
            let s = self.read_snapshot_boundary();
            let catalog = self.read_catalog_as_of(s);
            let (plan, presentation) = match build_join_plan(&stmt) {
                Ok(plan) => (plan, None),
                Err(error) => {
                    if !select_tree_uses_synthesized_catalog(&stmt, &catalog) {
                        return Err(error);
                    }
                    let projection = catalog_join_projection_plan(&stmt)?;
                    let plan = build_join_plan_with_projection(
                        &stmt,
                        projection.projection,
                        projection.aliases,
                        projection.order_by,
                        projection.order_by_nulls_first,
                    )?;
                    (plan, Some(projection.presentation))
                }
            };
            // Bind BOTH relations at ONE catalog generation (so the WHERE mapping + the join read the
            // same generation), then map the per-relation WHERE conjuncts (each conjunct must reference
            // exactly one relation; a cross-relation conjunct is a follow-up).
            // Resolve each join relation to its catalog table + (for a synthesized catalog relation) its
            // host rows. A real user relation is resolved FIRST (so it shadows a catalog name, mirroring
            // the single-relation path) and takes the resident join path (rows = None). A
            // `pg_catalog`/`information_schema` relation has no residency snapshot, so it is SYNTHESIZED
            // here (M5 J5) and its rows are threaded to the executor, which uploads a TRANSIENT device
            // payload and runs the SAME GPU hash join -- no CPU relational join (charter).
            let bind = |relation: &JoinRelationRef| -> Result<BoundJoinRelation, ExecuteError> {
                let (mut table, mut rows) = bind_join_relation(relation, &catalog)?;
                if presentation.is_some() {
                    if let Some(rows) = rows.as_mut() {
                        add_gpu_catalog_presentation_columns(&mut table, rows, &catalog)?;
                    }
                }
                Ok((table, rows))
            };
            let mut tables: Vec<RelationalTable> = Vec::with_capacity(plan.relations.len());
            let mut rows: Vec<Option<Vec<Vec<SqlValue>>>> =
                Vec::with_capacity(plan.relations.len());
            for relation in &plan.relations {
                let (table, relation_rows) = bind(relation)?;
                tables.push(table);
                rows.push(relation_rows);
            }
            let aliases: Vec<&str> = plan.relations.iter().map(|r| r.alias.as_str()).collect();
            let mut predicates = match stmt.where_clause.as_deref() {
                Some(where_node) => split_join_where(where_node, &tables, &aliases, &catalog)?,
                None => (0..plan.relations.len()).map(|_| None).collect(),
            };
            let on_predicates = split_join_on_local_filters(&stmt, &tables, &aliases, &catalog)?;
            for (predicate, on_predicate) in predicates.iter_mut().zip(on_predicates) {
                *predicate = match (predicate.take(), on_predicate) {
                    (Some(left), Some(right)) => Some(ResidentExpr::Binary {
                        op: ResidentBinaryOp::And,
                        lhs: Box::new(left),
                        rhs: Box::new(right),
                    }),
                    (left, right) => left.or(right),
                };
            }
            if rows.iter().all(Option::is_none) {
                if let Some(result) = self.try_streaming_inner_join(&plan, &tables, &predicates, s)
                {
                    return result.and_then(|result| match &presentation {
                        Some(presentation) => {
                            apply_catalog_projection_metadata(result, presentation)
                        }
                        None => Ok(result),
                    });
                }
            }
            let result = self.execute_resident_expr_inner_join(
                &plan, tables, rows, predicates, s, None, None, false,
            )?;
            return match &presentation {
                Some(presentation) => apply_catalog_projection_metadata(result, presentation),
                None => Ok(result),
            };
        }
        // A comma join (`FROM a, b[, c] WHERE a.k = b.k ...`) is an INNER join whose conditions live in
        // the WHERE: bind the relations, then derive the left-deep steps + per-relation filters from the
        // WHERE (the same `JoinStep`/executor as an explicit JOIN, incl. composite 2-edge keys).
        if let Some(relations) = comma_join_relations(&stmt)? {
            reject_unsupported_join_clauses(&stmt)?;
            let (projection, projection_aliases) = parse_join_projection(&stmt)?;
            let s = self.read_snapshot_boundary();
            let catalog = self.read_catalog_as_of(s);
            let bind = |relation: &JoinRelationRef| -> Result<BoundJoinRelation, ExecuteError> {
                bind_join_relation(relation, &catalog)
            };
            let mut tables: Vec<RelationalTable> = Vec::with_capacity(relations.len());
            let mut rows: Vec<Option<Vec<Vec<SqlValue>>>> = Vec::with_capacity(relations.len());
            for relation in &relations {
                let (table, relation_rows) = bind(relation)?;
                tables.push(table);
                rows.push(relation_rows);
            }
            let aliases: Vec<&str> = relations.iter().map(|r| r.alias.as_str()).collect();
            let (steps, predicates) = plan_comma_join_where(
                stmt.where_clause.as_deref(),
                &relations,
                &tables,
                &aliases,
                &catalog,
            )?;
            let (order_by, order_by_nulls_first, limit, offset) = parse_join_order_by_limit(&stmt)?;
            let plan = JoinPlan {
                relations,
                steps,
                projection,
                projection_aliases,
                order_by,
                order_by_nulls_first,
                limit,
                offset,
            };
            if rows.iter().all(Option::is_none) {
                if let Some(result) = self.try_streaming_inner_join(&plan, &tables, &predicates, s)
                {
                    return result;
                }
            }
            return self.execute_resident_expr_inner_join(
                &plan, tables, rows, predicates, s, None, None, false,
            );
        }
        // The presentation fallback is classified against the same catalog generation used to bind
        // the relation. It can never reinterpret an unsupported user-table SELECT.
        let copin_s = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(copin_s);
        let (select, qualifier, presentation, presentation_requires_columns) =
            match build_select_from_select_stmt(&stmt) {
                Ok((select, qualifier)) => {
                    let presentation = scalar_aggregate_alias_presentation(&stmt, &select)?;
                    (select, qualifier, presentation, false)
                }
                Err(error) => {
                    if !select_tree_uses_synthesized_catalog(&stmt, &catalog) {
                        return Err(error);
                    }
                    let plan = catalog_single_projection_plan(&stmt)?;
                    (plan.select, plan.qualifier, Some(plan.presentation), true)
                }
            };
        // Bind once; map the predicate (if any) against that SAME bound table; execute against that
        // binding. No WHERE clause is a full-table scan (the executor takes `None` for the predicate).
        // A synthesized one-relation catalog query needs the same injected transient device source as
        // catalog joins. User relations still resolve first, preserving the established shadowing rule.
        let synthesized_catalog_allowed = stmt.from_clause.as_slice().first().is_some_and(|from| {
            matches!(
                from.node.as_ref(),
                Some(NodeEnum::RangeVar(range)) if range.schemaname != "public"
            )
        });
        let has_range_column_alias_list = stmt
            .from_clause
            .first()
            .is_some_and(from_node_has_column_alias_list);
        let (table, bound, transient_rows) = if synthesized_catalog_allowed
            && !public_relation_name_exists(&catalog, &select.table)
        {
            if let Some((mut table, mut rows)) =
                synthesize_catalog_relation(&select.table, &catalog)
            {
                let exposed_width = table.columns.len();
                if presentation_requires_columns {
                    add_gpu_catalog_presentation_columns(&mut table, &mut rows, &catalog)?;
                }
                let range = match stmt.from_clause.as_slice() {
                    [from] => match node_enum(from)? {
                        NodeEnum::RangeVar(range) => range,
                        _ => {
                            return Err(sql_pg_error(
                                "catalog single-relation binding lost its RangeVar".to_string(),
                            ))
                        }
                    },
                    _ => {
                        return Err(sql_pg_error(
                            "catalog single-relation binding requires one RangeVar".to_string(),
                        ))
                    }
                };
                apply_catalog_range_column_aliases(&mut table, range, exposed_width)?;
                let bound = bind_relational_select(&table, &select)?;
                (table, bound, Some(rows))
            } else {
                if has_range_column_alias_list {
                    return Err(sql_pg_error(
                        "relation column-alias lists are not supported for resident user relations"
                            .to_string(),
                    ));
                }
                let (table, bound, _) = self.bind_relational_select_at(&select, copin_s)?;
                (table, bound, None)
            }
        } else {
            if has_range_column_alias_list {
                return Err(sql_pg_error(
                    "relation column-alias lists are not supported for resident user relations"
                        .to_string(),
                ));
            }
            let (table, bound, _) = self.bind_relational_select_at(&select, copin_s)?;
            (table, bound, None)
        };
        let predicate = stmt
            .where_clause
            .as_deref()
            .map(|where_node| map_predicate_node(where_node, &table, &qualifier, &catalog))
            .transpose()?;
        // Build the ORDER BY sort-expression list parallel to `select.order_by`: a bare ColumnRef ->
        // None (a plain column key); any other expression (`a+b`, `a*2`) -> a `ResidentExpr` via the
        // SAME mapper the WHERE uses, which the executor evaluates on-device into an i64 key column.
        let mut order_by_exprs: Vec<Option<ResidentExpr>> =
            Vec::with_capacity(stmt.sort_clause.len());
        // Parallel to order_by_exprs: the explicit NULLS FIRST/LAST override per ORDER BY key (honored
        // ON-DEVICE in the GPU sort comparator; None = PG default placement). Carried alongside rather
        // than on SelectOrder so the legacy protocol crate's SelectOrder is untouched (charter).
        let mut order_by_nulls_first: Vec<Option<bool>> =
            Vec::with_capacity(stmt.sort_clause.len());
        for (order_index, item) in stmt
            .sort_clause
            .iter()
            .take(if select.order_by.is_empty() {
                0
            } else {
                stmt.sort_clause.len()
            })
            .enumerate()
        {
            let NodeEnum::SortBy(sort_by) = node_enum(item)? else {
                return Err(sql_pg_error("malformed ORDER BY clause".to_string()));
            };
            let node = sort_by
                .node
                .as_deref()
                .ok_or_else(|| sql_pg_error("ORDER BY key has no expression".to_string()))?;
            // Lowering has already resolved column references and positional keys such as
            // `ORDER BY 1` to a concrete projected column. Only an empty SelectOrder column marks
            // a true per-row expression that needs a derived device key.
            if select
                .order_by
                .get(order_index)
                .is_some_and(|order| !order.column.is_empty())
            {
                order_by_exprs.push(None);
            } else {
                order_by_exprs.push(Some(map_predicate_node(
                    node, &table, &qualifier, &catalog,
                )?));
            }
            order_by_nulls_first.push(
                if sort_by.sortby_nulls == SortByNulls::SortbyNullsFirst as i32 {
                    Some(true)
                } else if sort_by.sortby_nulls == SortByNulls::SortbyNullsLast as i32 {
                    Some(false)
                } else {
                    None
                },
            );
        }
        // GROUP BY <expression>: a bare ColumnRef -> None (the plain-column matrix path); any other
        // expression (`a+b`, `a*2`) -> a ResidentExpr the executor materializes on-device into a derived
        // int key column (grouped via key_base_override). Built post-bind, threaded like the predicate.
        let group_key_expr: Option<ResidentExpr> = match stmt.group_clause.first() {
            Some(node) => {
                if matches!(node_enum(node)?, NodeEnum::ColumnRef(_)) {
                    None
                } else {
                    Some(map_predicate_node(node, &table, &qualifier, &catalog)?)
                }
            }
            None => None,
        };
        // The GROUP BY key COLUMNS (a composite `GROUP BY a, b[, ...]` packs/wide-keys them on-device).
        // Every group term that is a bare ColumnRef -> its name; an expression term is skipped (handled
        // by group_key_expr). The executor validates the member count + types.
        let mut group_key_columns: Vec<String> = Vec::new();
        for node in &stmt.group_clause {
            if let NodeEnum::ColumnRef(column_ref) = node_enum(node)? {
                group_key_columns.push(resolve_column_name(column_ref, &qualifier)?.to_string());
            }
        }
        // A SHARD-resident table resolves to the unified exec source (visibility included) INSIDE
        // `execute_resident_expr_select_with_binding` — the one resolution point for every `src: None`
        // caller (THE FLIP). The whole-table single-store path carries no version columns.
        let transient_source = transient_rows
            .map(|rows| {
                let row_count = rows.len() as u64;
                let (snapshot, memory) = self.build_transient_relation_residency(&table, &rows)?;
                Ok::<ResidentExecSource, ExecuteError>(ResidentExecSource {
                    descriptor: Arc::new(snapshot),
                    device_memory: Arc::new(memory),
                    row_count,
                })
            })
            .transpose()?;
        let result = self.execute_resident_expr_select_with_binding(
            &select,
            &table,
            transient_source.as_ref(),
            bound,
            copin_s,
            predicate.as_ref(),
            None,
            &order_by_exprs,
            &order_by_nulls_first,
            group_key_expr.as_ref(),
            &group_key_columns,
        )?;
        match &presentation {
            Some(presentation) => apply_catalog_projection_metadata(result, presentation),
            None => Ok(result),
        }
    }

    fn execute_gpu_rank_window_select(
        &self,
        stmt: &SelectStmt,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_gpu_rank_window_select_with_hook(stmt, None)
    }

    #[cfg(test)]
    pub(crate) fn execute_gpu_rank_window_select_instrumented(
        &self,
        sql: &str,
        after_cold_pin: &dyn Fn(),
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let stmt = parse_single_select(sql)?;
        if !select_has_inline_window(&stmt)? {
            return Err(sql_pg_error("instrumented query has no window".to_string()));
        }
        self.execute_gpu_rank_window_select_with_hook(&stmt, Some(after_cold_pin))
    }

    fn execute_gpu_rank_window_select_with_hook(
        &self,
        stmt: &SelectStmt,
        after_cold_pin: Option<&dyn Fn()>,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let [from] = stmt.from_clause.as_slice() else {
            return Err(sql_pg_error(
                "the GPU rank-window path supports one base relation".to_string(),
            ));
        };
        let (table_name, qualifier) = rank_window_relation_binding(from)?;
        if !stmt.group_clause.is_empty()
            || stmt.having_clause.is_some()
            || !stmt.distinct_clause.is_empty()
            || stmt.with_clause.is_some()
        {
            return Err(sql_pg_error(
                "rank windows currently require a single-table input without GROUP/DISTINCT/CTE"
                    .to_string(),
            ));
        }
        let statement_copin_s = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(statement_copin_s);
        let table = catalog
            .relational_catalog
            .get(&table_name)
            .cloned()
            .ok_or_else(|| sql_pg_error(format!("relation \"{table_name}\" does not exist")))?;
        let input_predicate = stmt
            .where_clause
            .as_deref()
            .map(|node| map_predicate_node(node, &table, &qualifier, &catalog))
            .transpose()?;
        let mut targets = Vec::with_capacity(stmt.target_list.len());
        let mut projected = Vec::<String>::new();
        let mut window_order: Option<Vec<SelectOrder>> = None;
        let mut window_order_nulls: Option<Vec<Option<bool>>> = None;
        let mut window_partition: Option<Vec<String>> = None;
        for target in &stmt.target_list {
            let NodeEnum::ResTarget(target) = node_enum(target)? else {
                return Err(sql_pg_error("malformed window SELECT target".to_string()));
            };
            let value = target
                .val
                .as_deref()
                .ok_or_else(|| sql_pg_error("window SELECT target has no value".to_string()))?;
            match node_enum(value)? {
                NodeEnum::ColumnRef(column) => {
                    let name = resolve_column_name(column, &qualifier)?.to_string();
                    relational_column_index(&table, &name)?;
                    if !projected.contains(&name) {
                        projected.push(name.clone());
                    }
                    let output = if target.name.is_empty() {
                        name.clone()
                    } else {
                        target.name.clone()
                    };
                    targets.push(GpuRankTarget::Column {
                        source: name,
                        output,
                    });
                }
                NodeEnum::FuncCall(func) => {
                    let Some(raw_over) = func.over.as_deref() else {
                        return Err(sql_pg_error(
                            "only window functions may appear beside columns on the rank-window path"
                                .to_string(),
                        ));
                    };
                    let resolved_over = resolve_rank_window_def(raw_over, &stmt.window_clause, 0)?;
                    let over = &resolved_over;
                    const FRAMEOPTION_NONDEFAULT: i32 = 0x00001;
                    if over.frame_options & FRAMEOPTION_NONDEFAULT != 0
                        || over.start_offset.is_some()
                        || over.end_offset.is_some()
                    {
                        return Err(sql_pg_error(
                            "explicit window frames are not supported on the GPU window path"
                                .to_string(),
                        ));
                    }
                    if func.agg_filter.is_some()
                        || !func.agg_order.is_empty()
                        || func.agg_star
                        || func.agg_distinct
                        || func.agg_within_group
                        || func.func_variadic
                    {
                        return Err(sql_pg_error(
                            "GPU window functions received unsupported arguments or aggregate modifiers"
                                .to_string(),
                        ));
                    }
                    if func.funcname.len() != 1 {
                        return Err(sql_pg_error(
                            "GPU rank-window function names must be unqualified".to_string(),
                        ));
                    }
                    let name = match func.funcname.last().map(node_enum).transpose()? {
                        Some(NodeEnum::String(name)) => name.sval.to_ascii_lowercase(),
                        _ => {
                            return Err(sql_pg_error("malformed window function name".to_string()))
                        }
                    };
                    let output_name = if target.name.is_empty() {
                        name.clone()
                    } else {
                        target.name.clone()
                    };
                    let window_target = match name.as_str() {
                        "row_number" | "rank" | "dense_rank" => {
                            if !func.args.is_empty() {
                                return Err(sql_pg_error(
                                    "ROW_NUMBER/RANK/DENSE_RANK take no arguments or aggregate modifiers"
                                        .to_string(),
                                ));
                            }
                            let kind = match name.as_str() {
                                "row_number" => GpuRankWindowKind::RowNumber,
                                "rank" => GpuRankWindowKind::Rank,
                                "dense_rank" => GpuRankWindowKind::DenseRank,
                                _ => unreachable!(),
                            };
                            GpuRankTarget::Window(kind, output_name)
                        }
                        "lag" | "lead" => {
                            if !(1..=2).contains(&func.args.len()) {
                                return Err(sql_pg_error(
                                    "GPU LAG/LEAD require a column and optional non-negative integer offset"
                                        .to_string(),
                                ));
                            }
                            let NodeEnum::ColumnRef(column) = node_enum(&func.args[0])? else {
                                return Err(sql_pg_error(
                                    "GPU LAG/LEAD value must be a plain column".to_string(),
                                ));
                            };
                            let source = resolve_column_name(column, &qualifier)?.to_string();
                            relational_column_index(&table, &source)?;
                            if !projected.contains(&source) {
                                projected.push(source.clone());
                            }
                            let offset = if let Some(node) = func.args.get(1) {
                                match node_enum(node)? {
                                    NodeEnum::AConst(constant) => match &constant.val {
                                        Some(a_const::Val::Ival(integer)) if integer.ival >= 0 => {
                                            integer.ival as u32
                                        }
                                        _ => {
                                            return Err(sql_pg_error(
                                                "GPU LAG/LEAD offset must be a non-negative integer literal"
                                                    .to_string(),
                                            ))
                                        }
                                    },
                                    _ => {
                                        return Err(sql_pg_error(
                                            "GPU LAG/LEAD offset must be an integer literal"
                                                .to_string(),
                                        ))
                                    }
                                }
                            } else {
                                1
                            };
                            GpuRankTarget::OffsetWindow {
                                kind: if name == "lag" {
                                    GpuOffsetWindowKind::Lag
                                } else {
                                    GpuOffsetWindowKind::Lead
                                },
                                source,
                                offset,
                                output: output_name,
                            }
                        }
                        _ => {
                            return Err(sql_pg_error(format!(
                                "window function {name} is not on the GPU rank path"
                            )))
                        }
                    };
                    let order = parse_order_by(&over.order_clause, &qualifier)?;
                    let order_nulls = parse_order_by_null_placement(&over.order_clause)?;
                    for order_key in &order {
                        let order_idx = relational_column_index(&table, &order_key.column)?;
                        if !matches!(
                            table.columns[order_idx].ty,
                            SqlType::Int2
                                | SqlType::Int4
                                | SqlType::Int8
                                | SqlType::Date
                                | SqlType::Timestamp
                                | SqlType::Text
                        ) {
                            return Err(sql_pg_error(
                                "GPU rank-window ORDER BY currently requires integer/date/timestamp/text columns"
                                    .to_string(),
                            ));
                        }
                        if !projected.contains(&order_key.column) {
                            projected.push(order_key.column.clone());
                        }
                    }
                    let mut partition = Vec::with_capacity(over.partition_clause.len());
                    for node in &over.partition_clause {
                        let NodeEnum::ColumnRef(column) = node_enum(node)? else {
                            return Err(sql_pg_error(
                                "GPU rank-window PARTITION BY keys must be plain columns"
                                    .to_string(),
                            ));
                        };
                        let name = resolve_column_name(column, &qualifier)?.to_string();
                        let idx = relational_column_index(&table, &name)?;
                        if !matches!(
                            table.columns[idx].ty,
                            SqlType::Int2
                                | SqlType::Int4
                                | SqlType::Int8
                                | SqlType::Date
                                | SqlType::Timestamp
                                | SqlType::Text
                        ) {
                            return Err(sql_pg_error(
                                "GPU rank-window PARTITION BY currently requires integer/date/timestamp/text columns"
                                    .to_string(),
                            ));
                        }
                        if !projected.contains(&name) {
                            projected.push(name.clone());
                        }
                        partition.push(name);
                    }
                    if let Some(existing) = &window_partition {
                        if existing != &partition {
                            return Err(sql_pg_error(
                                "all GPU rank functions in one SELECT must share PARTITION BY"
                                    .to_string(),
                            ));
                        }
                    } else {
                        window_partition = Some(partition);
                    }
                    if let Some(existing) = &window_order {
                        if existing != &order || window_order_nulls.as_ref() != Some(&order_nulls) {
                            return Err(sql_pg_error(
                                "all GPU rank functions in one SELECT must share the same window ORDER BY"
                                    .to_string(),
                            ));
                        }
                    } else {
                        window_order = Some(order);
                        window_order_nulls = Some(order_nulls);
                    }
                    targets.push(window_target);
                }
                _ => {
                    return Err(sql_pg_error(
                        "the GPU rank-window path projects plain columns and rank functions only"
                            .to_string(),
                    ))
                }
            }
        }
        let order = window_order.ok_or_else(|| {
            sql_pg_error("window SELECT contains no supported rank function".to_string())
        })?;
        let order_nulls = window_order_nulls.unwrap_or_default();
        let partition = window_partition.unwrap_or_default();
        let mut physical_order = Vec::new();
        for partition in &partition {
            physical_order.push(SelectOrder {
                column: partition.clone(),
                descending: false,
            });
        }
        physical_order.extend(order.clone());
        let mut physical_nulls = vec![None; partition.len()];
        physical_nulls.extend(order_nulls.iter().copied());
        let top_order = parse_order_by(&stmt.sort_clause, &qualifier)?;
        let top_nulls = parse_order_by_null_placement(&stmt.sort_clause)?;
        if !top_order.is_empty() && (top_order != physical_order || top_nulls != physical_nulls) {
            return Err(sql_pg_error(
                "top-level ORDER BY must match PARTITION BY + window ORDER BY on this path"
                    .to_string(),
            ));
        }
        let limit = parse_limit(&stmt.limit_count)?;
        let offset = parse_limit(&stmt.limit_offset)?;
        // A resident input that nearly fills the configured budget leaves no room for the rank
        // mask/coordinate/sort/window allocations. Move it through the bounded STRATA repair
        // bridge before choosing the route so input plus scratch—not input alone—defines fit.
        if self.current_transaction_read_snapshot().is_none() {
            if let Some(budget) =
                self.relational_residency_budget_bytes(self.planner.default_gpu_id())
            {
                self.transition_device_table_to_streaming_repair_above(
                    &table_name,
                    (budget / 2).max(1),
                )
                .map_err(ExecuteError::Engine)?;
            }
        }
        if self.table_is_gpu_resident(&table_name) {
            return self.execute_gpu_rank_window_device_resident(
                &table_name,
                &table,
                statement_copin_s,
                input_predicate.as_ref(),
                &targets,
                &projected,
                &partition,
                &order,
                &order_nulls,
                limit,
                offset,
            );
        }
        if let Some(result) = self.try_execute_gpu_rank_window_device_streaming(
            &table_name,
            &table,
            statement_copin_s,
            input_predicate.as_ref(),
            &targets,
            &projected,
            &partition,
            &order,
            &order_nulls,
            limit,
            offset,
            after_cold_pin,
        ) {
            return result;
        }
        Err(sql_pg_error(
            "GPU rank windows require resident input or a configured device-streaming budget"
                .to_string(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_execute_gpu_rank_window_device_streaming(
        &self,
        _table_name: &str,
        table: &RelationalTable,
        statement_copin_s: Index,
        predicate: Option<&ResidentExpr>,
        targets: &[GpuRankTarget],
        projected: &[String],
        partition: &[String],
        order: &[SelectOrder],
        order_nulls: &[Option<bool>],
        limit: Option<usize>,
        offset: Option<usize>,
        after_cold_pin: Option<&dyn Fn()>,
    ) -> Option<Result<RelationalSelectResult, ExecuteError>> {
        use gpu_db_execution::{
            CudaJoinOrderKey, CudaMaterializeJoinColumn, CudaMaterializedColumnKind,
            CudaMaterializedRelation, CudaWindowRankKind,
        };
        let lead_lookahead = targets
            .iter()
            .filter_map(|target| match target {
                GpuRankTarget::OffsetWindow {
                    kind: GpuOffsetWindowKind::Lead,
                    offset,
                    ..
                } => Some(*offset as usize),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let fetch = limit
            .map(|limit| {
                limit
                    .saturating_add(offset.unwrap_or(0))
                    .saturating_add(lead_lookahead)
            })
            .unwrap_or(u32::MAX as usize);
        let fetch_u32 = match u32::try_from(fetch) {
            Ok(value) => value,
            Err(_) => return Some(Err(sql_pg_error("rank fetch exceeds u32".to_string()))),
        };
        let gpu_id = self.planner.default_gpu_id();
        let budget = self.relational_residency_budget_bytes(gpu_id)?;
        let cold =
            self.ensure_streaming_join_cold(table, statement_copin_s, gpu_id, (budget / 2).max(1))?;
        let input_peak = cold
            .chunks
            .iter()
            .map(|chunk| chunk.snapshot.resident_bytes)
            .max()
            .unwrap_or(0);
        let scratch_budget = match budget.checked_sub(input_peak) {
            Some(bytes) => bytes,
            None => {
                return Some(Err(sql_pg_error(format!(
                    "rank-window input ({input_peak}) exceeds the query budget ({budget})"
                ))))
            }
        };
        let allocation_scope = gpu_db_execution::CudaAllocationScope::with_budget(scratch_budget);
        if let Some(hook) = after_cold_pin {
            hook();
        }
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let physical_names = {
            let mut names = partition.to_vec();
            for key in order {
                if !names.contains(&key.column) {
                    names.push(key.column.clone());
                }
            }
            for name in projected {
                if !names.contains(name) {
                    names.push(name.clone());
                }
            }
            names
        };
        let projected_index = |name: &str| {
            projected
                .iter()
                .position(|column| column == name)
                .expect("rank binder added every physical key to projection")
        };
        let materialize_source = |src: &crate::engine_expr::ResidentExecSource,
                                  coordinates: &gpu_db_execution::CudaJoinCoordinatesU32|
         -> Result<CudaMaterializedRelation, ExecuteError> {
            let mut specs = Vec::with_capacity(projected.len());
            for name in projected {
                let column = relational_column_index(table, name)?;
                let validity = resident_device_null_column_offset(&src.descriptor, table, column)?;
                specs.push(match table.columns[column].ty {
                    SqlType::Text => {
                        let layout =
                            resident_device_text_column_layout(&src.descriptor, table, column)?;
                        CudaMaterializeJoinColumn::Text {
                            relation: 0,
                            payload: &src.device_memory,
                            offsets_byte_offset: layout.offsets_byte_offset,
                            bytes_byte_offset: layout.bytes_byte_offset,
                            bytes_len: layout.bytes_len,
                            validity_bitmap_offset: validity,
                        }
                    }
                    SqlType::Bool => CudaMaterializeJoinColumn::Bool {
                        relation: 0,
                        payload: &src.device_memory,
                        bitmap_byte_offset: resident_device_bool_column_offset(
                            &src.descriptor,
                            table,
                            column,
                        )?,
                        validity_bitmap_offset: validity,
                    },
                    ty => {
                        let (byte_offset, width) = match ty {
                            SqlType::Int8 | SqlType::Timestamp => (
                                resident_device_int8_column_offset(&src.descriptor, table, column)?,
                                8,
                            ),
                            SqlType::Numeric { .. } | SqlType::Uuid => (
                                resident_device_numeric_column_offset(
                                    &src.descriptor,
                                    table,
                                    column,
                                )?,
                                16,
                            ),
                            SqlType::Int2 | SqlType::Int4 | SqlType::Date => (
                                resident_device_int4_column_offset(&src.descriptor, table, column)?,
                                4,
                            ),
                            SqlType::Text | SqlType::Bool => unreachable!(),
                        };
                        CudaMaterializeJoinColumn::Fixed {
                            relation: 0,
                            payload: &src.device_memory,
                            byte_offset,
                            validity_bitmap_offset: validity,
                            width,
                        }
                    }
                });
            }
            src.device_memory
                .materialize_join_coordinates(coordinates, &specs)
                .map_err(map_err)
        };
        let materialize_run = |run: &CudaMaterializedRelation,
                               coordinates: &gpu_db_execution::CudaJoinCoordinatesU32|
         -> Result<CudaMaterializedRelation, ExecuteError> {
            let specs = run
                .columns()
                .iter()
                .map(|layout| match layout.kind {
                    CudaMaterializedColumnKind::Fixed { width } => {
                        CudaMaterializeJoinColumn::Fixed {
                            relation: 0,
                            payload: run.memory(),
                            byte_offset: layout.value_byte_offset,
                            validity_bitmap_offset: Some(layout.validity_bitmap_offset),
                            width,
                        }
                    }
                    CudaMaterializedColumnKind::Text => CudaMaterializeJoinColumn::Text {
                        relation: 0,
                        payload: run.memory(),
                        offsets_byte_offset: layout.value_byte_offset,
                        bytes_byte_offset: layout.text_bytes_byte_offset.expect("text bytes"),
                        bytes_len: layout.text_bytes_len,
                        validity_bitmap_offset: Some(layout.validity_bitmap_offset),
                    },
                })
                .collect::<Vec<_>>();
            run.memory()
                .materialize_join_coordinates(coordinates, &specs)
                .map_err(map_err)
        };
        fn run_order<'a>(
            run: &'a CudaMaterializedRelation,
            physical_names: &[String],
            projected: &[String],
            table: &RelationalTable,
            order: &[SelectOrder],
            order_nulls: &[Option<bool>],
        ) -> Result<Vec<CudaJoinOrderKey<'a>>, ExecuteError> {
            physical_names
                .iter()
                .map(|name| {
                    let index = projected
                        .iter()
                        .position(|column| column == name)
                        .expect("rank key projected");
                    let explicit = order.iter().position(|key| key.column == *name);
                    let descending = explicit.is_some_and(|idx| order[idx].descending);
                    let nulls_first = explicit
                        .and_then(|idx| order_nulls.get(idx).copied().flatten())
                        .unwrap_or(descending);
                    Ok(CudaJoinOrderKey {
                        relation: 0,
                        key: run.payload_key(index).ok_or_else(|| {
                            sql_pg_error("rank run column layout is missing".to_string())
                        })?,
                        descending,
                        nulls_first,
                        lexicographic_16: table.columns[relational_column_index(table, name)?].ty
                            == SqlType::Uuid,
                    })
                })
                .collect()
        }
        fn source_order<'a>(
            src: &'a crate::engine_expr::ResidentExecSource,
            physical_names: &[String],
            table: &RelationalTable,
            order: &[SelectOrder],
            order_nulls: &[Option<bool>],
        ) -> Result<Vec<CudaJoinOrderKey<'a>>, ExecuteError> {
            physical_names
                .iter()
                .map(|name| {
                    let column = relational_column_index(table, name)?;
                    let ty = table.columns[column].ty;
                    let validity =
                        resident_device_null_column_offset(&src.descriptor, table, column)?;
                    let key = match ty {
                        SqlType::Text => {
                            let layout =
                                resident_device_text_column_layout(&src.descriptor, table, column)?;
                            gpu_db_execution::CudaJoinPayloadKey {
                                payload: &src.device_memory,
                                byte_offset: layout.offsets_byte_offset,
                                validity_bitmap_offset: validity,
                                width: 255,
                                text_bytes_byte_offset: Some(layout.bytes_byte_offset),
                                text_bytes_len: layout.bytes_len,
                            }
                        }
                        SqlType::Numeric { .. } | SqlType::Uuid => {
                            gpu_db_execution::CudaJoinPayloadKey {
                                payload: &src.device_memory,
                                byte_offset: resident_device_numeric_column_offset(
                                    &src.descriptor,
                                    table,
                                    column,
                                )?,
                                validity_bitmap_offset: validity,
                                width: 16,
                                text_bytes_byte_offset: None,
                                text_bytes_len: 0,
                            }
                        }
                        SqlType::Int8 | SqlType::Timestamp => {
                            gpu_db_execution::CudaJoinPayloadKey {
                                payload: &src.device_memory,
                                byte_offset: resident_device_int8_column_offset(
                                    &src.descriptor,
                                    table,
                                    column,
                                )?,
                                validity_bitmap_offset: validity,
                                width: 8,
                                text_bytes_byte_offset: None,
                                text_bytes_len: 0,
                            }
                        }
                        SqlType::Int2 | SqlType::Int4 | SqlType::Date => {
                            gpu_db_execution::CudaJoinPayloadKey {
                                payload: &src.device_memory,
                                byte_offset: resident_device_int4_column_offset(
                                    &src.descriptor,
                                    table,
                                    column,
                                )?,
                                validity_bitmap_offset: validity,
                                width: 4,
                                text_bytes_byte_offset: None,
                                text_bytes_len: 0,
                            }
                        }
                        SqlType::Bool => {
                            return Err(sql_pg_error(
                                "bool rank tiebreak is not supported".to_string(),
                            ))
                        }
                    };
                    let explicit = order.iter().position(|key| key.column == *name);
                    let descending = explicit.is_some_and(|idx| order[idx].descending);
                    Ok(CudaJoinOrderKey {
                        relation: 0,
                        key,
                        descending,
                        nulls_first: explicit
                            .and_then(|idx| order_nulls.get(idx).copied().flatten())
                            .unwrap_or(descending),
                        lexicographic_16: ty == SqlType::Uuid,
                    })
                })
                .collect()
        }

        let mut accumulator: Option<CudaMaterializedRelation> = None;
        let mut peak = 0_u64;
        for chunk in cold
            .chunks
            .iter()
            .filter(|chunk| chunk.payload_copin_s <= statement_copin_s && chunk.row_count > 0)
        {
            let (src, visibility) = match self
                .stage_cold_chunk(chunk, statement_copin_s)
                .and_then(crate::engine_streaming_exec::StagedChunk::ready)
            {
                Ok(source) => source,
                Err(_) => {
                    return Some(Err(sql_pg_error(
                        "failed to stage a rank-window chunk".to_string(),
                    )))
                }
            };
            let mask = match self.resident_predicate_device_mask(
                predicate,
                table,
                &src.descriptor,
                &src.device_memory,
                src.row_count as u32,
                visibility,
            ) {
                Ok(mask) => mask,
                Err(err) => return Some(Err(err)),
            };
            let identity = match src
                .device_memory
                .identity_join_coordinates(src.row_count as u32, mask.as_ref())
            {
                Ok(value) => value,
                Err(err) => return Some(Err(map_err(err))),
            };
            let source_keys = match source_order(&src, &physical_names, table, order, order_nulls) {
                Ok(keys) => keys,
                Err(err) => return Some(Err(err)),
            };
            let sorted = match src
                .device_memory
                .sort_join_coordinates(&identity, &source_keys)
            {
                Ok(value) => value,
                Err(err) => return Some(Err(map_err(err))),
            };
            let top = match src
                .device_memory
                .window_join_coordinates(&sorted, 0, Some(fetch_u32))
            {
                Ok(value) => value,
                Err(err) => return Some(Err(map_err(err))),
            };
            let chunk_run = match materialize_source(&src, &top) {
                Ok(run) => run,
                Err(err) => return Some(Err(err)),
            };
            peak = peak.max(
                src.descriptor
                    .resident_bytes
                    .saturating_add(chunk_run.allocated_bytes())
                    .saturating_add(u64::from(identity.row_count()).saturating_mul(12)),
            );
            accumulator = match accumulator.take() {
                None => Some(chunk_run),
                Some(previous) => {
                    let previous_bytes = previous.allocated_bytes();
                    let chunk_bytes = chunk_run.allocated_bytes();
                    let combined = match previous
                        .memory()
                        .concat_materialized_relations(&previous, &chunk_run)
                    {
                        Ok(run) => run,
                        Err(err) => return Some(Err(map_err(err))),
                    };
                    peak = peak.max(
                        previous_bytes
                            .saturating_add(chunk_bytes)
                            .saturating_add(combined.allocated_bytes()),
                    );
                    drop(previous);
                    drop(chunk_run);
                    let combined_identity = match combined
                        .memory()
                        .identity_join_coordinates(combined.row_count(), None)
                    {
                        Ok(value) => value,
                        Err(err) => return Some(Err(map_err(err))),
                    };
                    let keys = match run_order(
                        &combined,
                        &physical_names,
                        projected,
                        table,
                        order,
                        order_nulls,
                    ) {
                        Ok(keys) => keys,
                        Err(err) => return Some(Err(err)),
                    };
                    let combined_sorted = match combined
                        .memory()
                        .sort_join_coordinates(&combined_identity, &keys)
                    {
                        Ok(value) => value,
                        Err(err) => return Some(Err(map_err(err))),
                    };
                    let combined_top = match combined.memory().window_join_coordinates(
                        &combined_sorted,
                        0,
                        Some(fetch_u32),
                    ) {
                        Ok(value) => value,
                        Err(err) => return Some(Err(map_err(err))),
                    };
                    let next = match materialize_run(&combined, &combined_top) {
                        Ok(run) => run,
                        Err(err) => return Some(Err(err)),
                    };
                    peak = peak.max(
                        combined
                            .allocated_bytes()
                            .saturating_add(next.allocated_bytes())
                            .saturating_add(u64::from(combined.row_count()).saturating_mul(12)),
                    );
                    Some(next)
                }
            };
            if peak > budget {
                return Some(Err(sql_pg_error(format!(
                    "rank-window allocator high-water ({peak}) exceeds the query budget ({budget})"
                ))));
            }
        }
        let run = accumulator?;
        let identity = match run
            .memory()
            .identity_join_coordinates(run.row_count(), None)
        {
            Ok(value) => value,
            Err(err) => return Some(Err(map_err(err))),
        };
        let rank_key = |name: &str| -> CudaJoinOrderKey<'_> {
            CudaJoinOrderKey {
                relation: 0,
                key: run
                    .payload_key(projected_index(name))
                    .expect("rank run key"),
                descending: false,
                nulls_first: false,
                lexicographic_16: false,
            }
        };
        let partition_keys = partition
            .iter()
            .map(|name| rank_key(name))
            .collect::<Vec<_>>();
        let order_keys = order
            .iter()
            .map(|item| rank_key(&item.column))
            .collect::<Vec<_>>();
        let ranks = match run.memory().window_ranks_from_join_coordinates(
            &identity,
            &partition_keys,
            &order_keys,
        ) {
            Ok(value) => value,
            Err(err) => return Some(Err(map_err(err))),
        };
        let offset_u32 = u32::try_from(offset.unwrap_or(0)).expect("fetch bound checked");
        let window = match run.memory().window_join_coordinates(
            &identity,
            offset_u32,
            limit.and_then(|limit| u32::try_from(limit).ok()),
        ) {
            Ok(value) => value,
            Err(err) => return Some(Err(map_err(err))),
        };
        let has_offset_window = targets
            .iter()
            .any(|target| matches!(target, GpuRankTarget::OffsetWindow { .. }));
        let final_live = run
            .allocated_bytes()
            .saturating_add(identity.allocated_bytes())
            .saturating_add(ranks.allocated_bytes())
            .saturating_add(window.allocated_bytes())
            .saturating_add(if has_offset_window {
                identity
                    .allocated_bytes()
                    .saturating_add(window.allocated_bytes())
            } else {
                0
            });
        peak = peak.max(final_live);
        if peak > budget {
            return Some(Err(sql_pg_error(format!(
                "rank-window allocator high-water ({peak}) exceeds the query budget ({budget})"
            ))));
        }
        let mut source_values = std::collections::BTreeMap::<String, Vec<SqlValue>>::new();
        for (index, name) in projected.iter().enumerate() {
            let layout = run.columns()[index];
            let ty = table.columns[relational_column_index(table, name).expect("bound")].ty;
            let values = match layout.kind {
                CudaMaterializedColumnKind::Text => run
                    .memory()
                    .project_text_from_join_coordinates(
                        &window,
                        0,
                        run.memory(),
                        layout.value_byte_offset,
                        layout.text_bytes_byte_offset.expect("text bytes"),
                        layout.text_bytes_len,
                        Some(layout.validity_bitmap_offset),
                    )
                    .map_err(map_err)
                    .map(|values| {
                        values
                            .into_iter()
                            .map(|value| value.map_or(SqlValue::Null, SqlValue::Text))
                            .collect()
                    }),
                CudaMaterializedColumnKind::Fixed { width } => run
                    .memory()
                    .project_fixed_from_join_coordinates(
                        &window,
                        0,
                        run.memory(),
                        layout.value_byte_offset,
                        Some(layout.validity_bitmap_offset),
                        width,
                    )
                    .map_err(map_err)
                    .map(|(raw, valid)| {
                        raw.chunks_exact(width as usize)
                            .zip(valid)
                            .map(|(bytes, valid)| {
                                if !valid {
                                    return SqlValue::Null;
                                }
                                match ty {
                                    SqlType::Bool => SqlValue::Bool(
                                        i32::from_le_bytes(bytes.try_into().unwrap()) != 0,
                                    ),
                                    SqlType::Int2 => SqlValue::Int2(i32::from_le_bytes(
                                        bytes.try_into().unwrap(),
                                    )
                                        as i16),
                                    SqlType::Int4 => SqlValue::Int4(i32::from_le_bytes(
                                        bytes.try_into().unwrap(),
                                    )),
                                    SqlType::Date => SqlValue::Date(i32::from_le_bytes(
                                        bytes.try_into().unwrap(),
                                    )),
                                    SqlType::Int8 => SqlValue::Int8(i64::from_le_bytes(
                                        bytes.try_into().unwrap(),
                                    )),
                                    SqlType::Timestamp => SqlValue::Timestamp(i64::from_le_bytes(
                                        bytes.try_into().unwrap(),
                                    )),
                                    SqlType::Numeric { scale, .. } => {
                                        SqlValue::Numeric(gpu_db_sql::Decimal128::new(
                                            i128::from_le_bytes(bytes.try_into().unwrap()),
                                            scale,
                                        ))
                                    }
                                    SqlType::Uuid => {
                                        SqlValue::Uuid(bytes.try_into().expect("uuid"))
                                    }
                                    SqlType::Text => unreachable!(),
                                }
                            })
                            .collect()
                    }),
            };
            match values {
                Ok(values) => {
                    source_values.insert(name.clone(), values);
                }
                Err(err) => return Some(Err(err)),
            }
        }
        let mut rank_values = std::collections::BTreeMap::new();
        for kind in [
            GpuRankWindowKind::RowNumber,
            GpuRankWindowKind::Rank,
            GpuRankWindowKind::DenseRank,
        ] {
            if targets
                .iter()
                .any(|target| matches!(target, GpuRankTarget::Window(target_kind, _) if *target_kind == kind))
            {
                let execution_kind = match kind {
                    GpuRankWindowKind::RowNumber => CudaWindowRankKind::RowNumber,
                    GpuRankWindowKind::Rank => CudaWindowRankKind::Rank,
                    GpuRankWindowKind::DenseRank => CudaWindowRankKind::DenseRank,
                };
                match ranks.readback(execution_kind, offset_u32, limit.and_then(|v|u32::try_from(v).ok())) {
                    Ok(values) => { rank_values.insert(kind as u8, values); }
                    Err(err) => return Some(Err(map_err(err))),
                }
            }
        }
        let mut offset_values = std::collections::BTreeMap::new();
        for (target_index, target) in targets.iter().enumerate() {
            let GpuRankTarget::OffsetWindow {
                kind,
                source,
                offset,
                ..
            } = target
            else {
                continue;
            };
            let delta = match kind {
                GpuOffsetWindowKind::Lag => -(*offset as i32),
                GpuOffsetWindowKind::Lead => *offset as i32,
            };
            let shifted =
                match run
                    .memory()
                    .shift_join_coordinates(&identity, &partition_keys, delta)
                {
                    Ok(value) => value,
                    Err(err) => return Some(Err(map_err(err))),
                };
            let shifted_window = match run.memory().window_join_coordinates(
                &shifted,
                offset_u32,
                limit.and_then(|value| u32::try_from(value).ok()),
            ) {
                Ok(value) => value,
                Err(err) => return Some(Err(map_err(err))),
            };
            let column = projected_index(source);
            let ty = table.columns[relational_column_index(table, source).expect("bound")].ty;
            match self.decode_materialized_column(&run, column, ty, &shifted_window) {
                Ok(values) => {
                    offset_values.insert(target_index, values);
                }
                Err(err) => return Some(Err(err)),
            }
        }
        let mut columns = Vec::with_capacity(targets.len());
        for (attnum, target) in targets.iter().enumerate() {
            let mut column = match target {
                GpuRankTarget::Column { source, output } => {
                    let mut column = table.columns
                        [relational_column_index(table, source).expect("bound")]
                    .clone();
                    column.name = output.clone();
                    column
                }
                GpuRankTarget::Window(_, name) => {
                    let mut column = table.columns[0].clone();
                    column.name = name.clone();
                    column.ty = SqlType::Int8;
                    column.domain = None;
                    column.default = None;
                    column.type_oid = 20;
                    column.type_size = 8;
                    column
                }
                GpuRankTarget::OffsetWindow { source, output, .. } => {
                    let mut column = table.columns
                        [relational_column_index(table, source).expect("bound")]
                    .clone();
                    column.name = output.clone();
                    column
                }
            };
            column.attnum = (attnum + 1) as i16;
            columns.push(column);
        }
        let rows = (0..window.row_count() as usize)
            .map(|row| {
                targets
                    .iter()
                    .enumerate()
                    .map(|(target_index, target)| match target {
                        GpuRankTarget::Column { source, .. } => source_values[source][row].clone(),
                        GpuRankTarget::Window(kind, _) => {
                            SqlValue::Int8(rank_values[&(*kind as u8)][row] as i64)
                        }
                        GpuRankTarget::OffsetWindow { .. } => {
                            offset_values[&target_index][row].clone()
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        peak = peak.max(input_peak.saturating_add(allocation_scope.peak_bytes()));
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(peak, std::sync::atomic::Ordering::Relaxed);
        self.read_state
            .residency
            .streaming_window_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(Ok(RelationalSelectResult {
            columns: Arc::new(columns),
            rows: rows.into(),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_gpu_rank_window_device_resident(
        &self,
        table_name: &str,
        table: &RelationalTable,
        statement_copin_s: Index,
        predicate: Option<&ResidentExpr>,
        targets: &[GpuRankTarget],
        projected: &[String],
        partition: &[String],
        order: &[SelectOrder],
        order_nulls: &[Option<bool>],
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        use gpu_db_execution::{CudaJoinOrderKey, CudaJoinPayloadKey, CudaWindowRankKind};
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let side = self.resolve_join_side(table_name, table, None, statement_copin_s)?;
        if side.2 > u32::MAX as usize {
            return Err(sql_pg_error(
                "rank-window input exceeds the u32 device coordinate range".to_string(),
            ));
        }
        let budget = self
            .relational_residency_budget_bytes(self.planner.default_gpu_id())
            .unwrap_or(u64::MAX / 4);
        let input_bytes = side.0.descriptor.resident_bytes;
        let scratch_budget = budget.checked_sub(input_bytes).ok_or_else(|| {
            sql_pg_error(format!(
                "rank-window resident input ({input_bytes}) exceeds the query budget ({budget})"
            ))
        })?;
        let allocation_scope = gpu_db_execution::CudaAllocationScope::with_budget(scratch_budget);
        let payload = side.1.mem();
        let mask = self.resident_predicate_device_mask(
            predicate,
            table,
            &side.0.descriptor,
            payload,
            side.2 as u32,
            side.3,
        )?;
        let identity = payload
            .identity_join_coordinates(side.2 as u32, mask.as_ref())
            .map_err(map_err)?;
        let key = |name: &str,
                   descending: bool,
                   nulls_first: bool|
         -> Result<CudaJoinOrderKey<'_>, ExecuteError> {
            let column = relational_column_index(table, name)?;
            let ty = table.columns[column].ty;
            let validity_bitmap_offset =
                resident_device_null_column_offset(&side.0.descriptor, table, column)?;
            let key = match ty {
                SqlType::Text => {
                    let layout =
                        resident_device_text_column_layout(&side.0.descriptor, table, column)?;
                    CudaJoinPayloadKey {
                        payload,
                        byte_offset: layout.offsets_byte_offset,
                        validity_bitmap_offset,
                        width: 255,
                        text_bytes_byte_offset: Some(layout.bytes_byte_offset),
                        text_bytes_len: layout.bytes_len,
                    }
                }
                SqlType::Numeric { .. } | SqlType::Uuid => CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_numeric_column_offset(
                        &side.0.descriptor,
                        table,
                        column,
                    )?,
                    validity_bitmap_offset,
                    width: 16,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                },
                SqlType::Int8 | SqlType::Timestamp => CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_int8_column_offset(
                        &side.0.descriptor,
                        table,
                        column,
                    )?,
                    validity_bitmap_offset,
                    width: 8,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                },
                SqlType::Int2 | SqlType::Int4 | SqlType::Date => CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_int4_column_offset(
                        &side.0.descriptor,
                        table,
                        column,
                    )?,
                    validity_bitmap_offset,
                    width: 4,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                },
                SqlType::Bool => {
                    return Err(sql_pg_error("bool rank keys are not supported".to_string()))
                }
            };
            Ok(CudaJoinOrderKey {
                relation: 0,
                key,
                descending,
                nulls_first,
                lexicographic_16: false,
            })
        };
        let partition_keys = partition
            .iter()
            .map(|name| key(name, false, false))
            .collect::<Result<Vec<_>, _>>()?;
        let order_keys = order
            .iter()
            .enumerate()
            .map(|(index, item)| {
                key(
                    &item.column,
                    item.descending,
                    order_nulls
                        .get(index)
                        .copied()
                        .flatten()
                        .unwrap_or(item.descending),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut physical_order = partition_keys.clone();
        physical_order.extend(order_keys.iter().copied());
        for name in projected {
            if partition.iter().any(|column| column == name)
                || order.iter().any(|column| column.column == *name)
            {
                continue;
            }
            let column = relational_column_index(table, name)?;
            if matches!(
                table.columns[column].ty,
                SqlType::Int2 | SqlType::Int4 | SqlType::Int8 | SqlType::Date | SqlType::Timestamp
            ) {
                physical_order.push(key(name, false, false)?);
            }
        }
        let sorted = payload
            .sort_join_coordinates(&identity, &physical_order)
            .map_err(map_err)?;
        let ranks = payload
            .window_ranks_from_join_coordinates(&sorted, &partition_keys, &order_keys)
            .map_err(map_err)?;
        let offset_u32 = u32::try_from(offset.unwrap_or(0))
            .map_err(|_| sql_pg_error("rank OFFSET exceeds u32".to_string()))?;
        let limit_u32 = limit
            .map(u32::try_from)
            .transpose()
            .map_err(|_| sql_pg_error("rank LIMIT exceeds u32".to_string()))?;
        let window = payload
            .window_join_coordinates(&sorted, offset_u32, limit_u32)
            .map_err(map_err)?;

        let mut estimated_live = u64::from(identity.row_count())
            .saturating_mul(
                4 + 8
                    + 24
                    + if targets
                        .iter()
                        .any(|target| matches!(target, GpuRankTarget::OffsetWindow { .. }))
                    {
                        8
                    } else {
                        0
                    },
            )
            .saturating_add((physical_order.len() as u64).saturating_mul(72));
        estimated_live =
            input_bytes.saturating_add(estimated_live.max(allocation_scope.peak_bytes()));
        if estimated_live > budget {
            return Err(sql_pg_error(format!(
                "rank-window live device bytes ({estimated_live}) exceed the query budget ({budget})"
            )));
        }
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(estimated_live, std::sync::atomic::Ordering::Relaxed);

        let mut source_values = std::collections::BTreeMap::<String, Vec<SqlValue>>::new();
        for name in projected {
            let column = relational_column_index(table, name)?;
            let validity = resident_device_null_column_offset(&side.0.descriptor, table, column)?;
            let ty = table.columns[column].ty;
            let values = match ty {
                SqlType::Text => {
                    let layout =
                        resident_device_text_column_layout(&side.0.descriptor, table, column)?;
                    payload
                        .project_text_from_join_coordinates(
                            &window,
                            0,
                            payload,
                            layout.offsets_byte_offset,
                            layout.bytes_byte_offset,
                            layout.bytes_len,
                            validity,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(|value| value.map_or(SqlValue::Null, SqlValue::Text))
                        .collect()
                }
                SqlType::Bool => {
                    let bitmap =
                        resident_device_bool_column_offset(&side.0.descriptor, table, column)?;
                    payload
                        .project_bool_from_join_coordinates(&window, 0, payload, bitmap, validity)
                        .map_err(map_err)?
                        .into_iter()
                        .map(|value| value.map_or(SqlValue::Null, SqlValue::Bool))
                        .collect()
                }
                _ => {
                    let (byte_offset, width) = match ty {
                        SqlType::Int8 | SqlType::Timestamp => (
                            resident_device_int8_column_offset(&side.0.descriptor, table, column)?,
                            8_u8,
                        ),
                        SqlType::Numeric { .. } | SqlType::Uuid => (
                            resident_device_numeric_column_offset(
                                &side.0.descriptor,
                                table,
                                column,
                            )?,
                            16_u8,
                        ),
                        SqlType::Int2 | SqlType::Int4 | SqlType::Date => (
                            resident_device_int4_column_offset(&side.0.descriptor, table, column)?,
                            4_u8,
                        ),
                        SqlType::Text | SqlType::Bool => unreachable!(),
                    };
                    let (raw, valid) = payload
                        .project_fixed_from_join_coordinates(
                            &window,
                            0,
                            payload,
                            byte_offset,
                            validity,
                            width,
                        )
                        .map_err(map_err)?;
                    raw.chunks_exact(width as usize)
                        .zip(valid)
                        .map(|(bytes, valid)| {
                            if !valid {
                                return SqlValue::Null;
                            }
                            match ty {
                                SqlType::Int2 => SqlValue::Int2(i32::from_le_bytes(
                                    bytes.try_into().unwrap(),
                                )
                                    as i16),
                                SqlType::Int4 => {
                                    SqlValue::Int4(i32::from_le_bytes(bytes.try_into().unwrap()))
                                }
                                SqlType::Date => {
                                    SqlValue::Date(i32::from_le_bytes(bytes.try_into().unwrap()))
                                }
                                SqlType::Int8 => {
                                    SqlValue::Int8(i64::from_le_bytes(bytes.try_into().unwrap()))
                                }
                                SqlType::Timestamp => SqlValue::Timestamp(i64::from_le_bytes(
                                    bytes.try_into().unwrap(),
                                )),
                                SqlType::Numeric { scale, .. } => {
                                    SqlValue::Numeric(gpu_db_sql::Decimal128::new(
                                        i128::from_le_bytes(bytes.try_into().unwrap()),
                                        scale,
                                    ))
                                }
                                SqlType::Uuid => {
                                    SqlValue::Uuid(bytes.try_into().expect("uuid width"))
                                }
                                SqlType::Text | SqlType::Bool => unreachable!(),
                            }
                        })
                        .collect()
                }
            };
            source_values.insert(name.clone(), values);
        }
        let mut rank_values = std::collections::BTreeMap::new();
        for kind in [
            GpuRankWindowKind::RowNumber,
            GpuRankWindowKind::Rank,
            GpuRankWindowKind::DenseRank,
        ] {
            if targets
                .iter()
                .any(|target| matches!(target, GpuRankTarget::Window(target_kind, _) if *target_kind == kind))
            {
                let execution_kind = match kind {
                    GpuRankWindowKind::RowNumber => CudaWindowRankKind::RowNumber,
                    GpuRankWindowKind::Rank => CudaWindowRankKind::Rank,
                    GpuRankWindowKind::DenseRank => CudaWindowRankKind::DenseRank,
                };
                rank_values.insert(
                    kind as u8,
                    ranks
                        .readback(execution_kind, offset_u32, limit_u32)
                        .map_err(map_err)?,
                );
            }
        }
        let mut offset_values = std::collections::BTreeMap::new();
        for (target_index, target) in targets.iter().enumerate() {
            let GpuRankTarget::OffsetWindow {
                kind,
                source,
                offset,
                ..
            } = target
            else {
                continue;
            };
            let delta = match kind {
                GpuOffsetWindowKind::Lag => -(*offset as i32),
                GpuOffsetWindowKind::Lead => *offset as i32,
            };
            let shifted = payload
                .shift_join_coordinates(&sorted, &partition_keys, delta)
                .map_err(map_err)?;
            let shifted_window = payload
                .window_join_coordinates(&shifted, offset_u32, limit_u32)
                .map_err(map_err)?;
            offset_values.insert(
                target_index,
                self.project_join_column_values(
                    table,
                    &side,
                    &shifted_window,
                    0,
                    relational_column_index(table, source)?,
                )?,
            );
        }
        let mut columns = Vec::with_capacity(targets.len());
        for (attnum, target) in targets.iter().enumerate() {
            let mut column = match target {
                GpuRankTarget::Column { source, output } => {
                    let mut column = table.columns[relational_column_index(table, source)?].clone();
                    column.name = output.clone();
                    column
                }
                GpuRankTarget::Window(_, name) => {
                    let mut column = table.columns[0].clone();
                    column.name = name.clone();
                    column.ty = SqlType::Int8;
                    column.domain = None;
                    column.default = None;
                    column.type_oid = 20;
                    column.type_size = 8;
                    column
                }
                GpuRankTarget::OffsetWindow { source, output, .. } => {
                    let mut column = table.columns[relational_column_index(table, source)?].clone();
                    column.name = output.clone();
                    column
                }
            };
            column.attnum = (attnum + 1) as i16;
            columns.push(column);
        }
        let rows = (0..window.row_count() as usize)
            .map(|row| {
                targets
                    .iter()
                    .enumerate()
                    .map(|(target_index, target)| match target {
                        GpuRankTarget::Column { source, .. } => source_values[source][row].clone(),
                        GpuRankTarget::Window(kind, _) => {
                            SqlValue::Int8(rank_values[&(*kind as u8)][row] as i64)
                        }
                        GpuRankTarget::OffsetWindow { .. } => {
                            offset_values[&target_index][row].clone()
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.read_state
            .residency
            .streaming_window_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(RelationalSelectResult {
            columns: Arc::new(columns),
            rows: rows.into(),
            planned_target: DeviceTarget::Gpu(self.planner.default_gpu_id()),
            executed_target: DeviceTarget::Gpu(self.planner.default_gpu_id()),
            fallback_reason: None,
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
        })
    }
}
