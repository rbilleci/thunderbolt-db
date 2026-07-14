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
    SetOperation, SortByDir, SortByNulls,
};
use pg_query::NodeEnum;

use crate::engine_expr::{
    JoinColRef, JoinPlan, JoinProjItem, JoinRelationRef, JoinStep, ResidentBinaryOp, ResidentExpr,
};
use gpu_db_sql::{SelectFilter, SelectFilterOp, SelectOrder};

mod join_lowering;

use join_lowering::{
    build_join_plan, comma_join_relations, from_clause_is_join, parse_join_order_by_limit,
    parse_join_projection, plan_comma_join_where, reject_unsupported_join_clauses,
    split_join_where,
};

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
    Column { source: String, output: String },
    Window(GpuRankWindowKind, String),
    OffsetWindow {
        kind: GpuOffsetWindowKind,
        source: String,
        offset: u32,
        output: String,
    },
}

impl Engine {
    /// Parse `sql` (libpg_query / real Postgres grammar), map it to the general `ResidentExpr` IR, and
    /// execute it on the GPU general executor (Charter rule 2). Supported shape: a single-table SELECT
    /// of int4 columns with an int4 `WHERE` predicate over arithmetic (`+ - *`), comparisons
    /// (`= <> < <= > >=`), and column-vs-column. The catalog is bound ONCE and the predicate's column
    /// indices are resolved against that SAME bound table, so the columns, projection, and residency
    /// snapshot all derive from one catalog generation (a concurrent shape-changing DDL cannot split
    /// the column resolution from the execution). Errors — not a silent fallback — on any shape the
    /// mapper cannot represent; the routing layer (slice 5) decides what to do with a rejection.
    pub fn execute_resident_expr_select_sql(
        &self,
        sql: &str,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let stmt = parse_single_select(sql)?;
        if select_has_inline_window(&stmt)? {
            return self.execute_gpu_rank_window_select(&stmt);
        }
        // A JOIN in the FROM clause routes to the dedicated 2-relation inner-equi-join path (M5). The
        // single-table path below is byte-identical for a non-join query.
        if from_clause_is_join(&stmt) {
            let plan = build_join_plan(&stmt)?;
            // Bind BOTH relations at ONE catalog generation (so the WHERE mapping + the join read the
            // same generation), then map the per-relation WHERE conjuncts (each conjunct must reference
            // exactly one relation; a cross-relation conjunct is a follow-up).
            let s = self.committed_seq();
            let catalog = self.read_state.catalog_as_of(s);
            // Resolve each join relation to its catalog table + (for a synthesized catalog relation) its
            // host rows. A real user relation is resolved FIRST (so it shadows a catalog name, mirroring
            // the single-relation path) and takes the resident join path (rows = None). A
            // `pg_catalog`/`information_schema` relation has no residency snapshot, so it is SYNTHESIZED
            // here (M5 J5) and its rows are threaded to the executor, which uploads a TRANSIENT device
            // payload and runs the SAME GPU hash join -- no CPU relational join (charter).
            #[allow(clippy::type_complexity)] // (catalog table, synthesized rows) | resident table
            let bind = |name: &str| -> Result<(RelationalTable, Option<Vec<Vec<SqlValue>>>), ExecuteError> {
                if let Some(table) = catalog.relational_catalog.get(name).cloned() {
                    Ok((table, None))
                } else if let Some((table, rows)) = synthesize_catalog_relation(name, &catalog) {
                    Ok((table, Some(rows)))
                } else {
                    Err(sql_pg_error(format!("relation \"{name}\" does not exist")))
                }
            };
            let mut tables: Vec<RelationalTable> = Vec::with_capacity(plan.relations.len());
            let mut rows: Vec<Option<Vec<Vec<SqlValue>>>> =
                Vec::with_capacity(plan.relations.len());
            for relation in &plan.relations {
                let (table, relation_rows) = bind(&relation.table)?;
                tables.push(table);
                rows.push(relation_rows);
            }
            let aliases: Vec<&str> = plan.relations.iter().map(|r| r.alias.as_str()).collect();
            let predicates = match stmt.where_clause.as_deref() {
                Some(where_node) => split_join_where(where_node, &tables, &aliases)?,
                None => (0..plan.relations.len()).map(|_| None).collect(),
            };
            if rows.iter().all(Option::is_none) {
                if let Some(result) = self.try_streaming_inner_join(&plan, &tables, &predicates, s) {
                    return result;
                }
            }
            return self.execute_resident_expr_inner_join(
                &plan, tables, rows, predicates, s, None, None, false,
            );
        }
        // A comma join (`FROM a, b[, c] WHERE a.k = b.k ...`) is an INNER join whose conditions live in
        // the WHERE: bind the relations, then derive the left-deep steps + per-relation filters from the
        // WHERE (the same `JoinStep`/executor as an explicit JOIN, incl. composite 2-edge keys).
        if let Some(relations) = comma_join_relations(&stmt) {
            reject_unsupported_join_clauses(&stmt)?;
            let (projection, projection_aliases) = parse_join_projection(&stmt)?;
            let s = self.committed_seq();
            let catalog = self.read_state.catalog_as_of(s);
            #[allow(clippy::type_complexity)] // (catalog table, synthesized rows) | resident table
            let bind = |name: &str| -> Result<(RelationalTable, Option<Vec<Vec<SqlValue>>>), ExecuteError> {
                if let Some(table) = catalog.relational_catalog.get(name).cloned() {
                    Ok((table, None))
                } else if let Some((table, rows)) = synthesize_catalog_relation(name, &catalog) {
                    Ok((table, Some(rows)))
                } else {
                    Err(sql_pg_error(format!("relation \"{name}\" does not exist")))
                }
            };
            let mut tables: Vec<RelationalTable> = Vec::with_capacity(relations.len());
            let mut rows: Vec<Option<Vec<Vec<SqlValue>>>> = Vec::with_capacity(relations.len());
            for relation in &relations {
                let (table, relation_rows) = bind(&relation.table)?;
                tables.push(table);
                rows.push(relation_rows);
            }
            let aliases: Vec<&str> = relations.iter().map(|r| r.alias.as_str()).collect();
            let (steps, predicates) =
                plan_comma_join_where(stmt.where_clause.as_deref(), &relations, &tables, &aliases)?;
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
                if let Some(result) = self.try_streaming_inner_join(&plan, &tables, &predicates, s) {
                    return result;
                }
            }
            return self.execute_resident_expr_inner_join(
                &plan, tables, rows, predicates, s, None, None, false,
            );
        }
        let (select, qualifier) = build_select_from_select_stmt(&stmt)?;
        // Bind once; map the predicate (if any) against that SAME bound table; execute against that
        // binding. No WHERE clause is a full-table scan (the executor takes `None` for the predicate).
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(&select)?;
        let predicate = stmt
            .where_clause
            .as_deref()
            .map(|where_node| map_predicate_node(where_node, &table, &qualifier))
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
        for item in &stmt.sort_clause {
            let NodeEnum::SortBy(sort_by) = node_enum(item)? else {
                return Err(sql_pg_error("malformed ORDER BY clause".to_string()));
            };
            let node = sort_by
                .node
                .as_deref()
                .ok_or_else(|| sql_pg_error("ORDER BY key has no expression".to_string()))?;
            if result_column_name(node, &qualifier).is_ok() {
                order_by_exprs.push(None);
            } else {
                order_by_exprs.push(Some(map_predicate_node(node, &table, &qualifier)?));
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
                    Some(map_predicate_node(node, &table, &qualifier)?)
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
        self.execute_resident_expr_select_with_binding(
            &select,
            &table,
            None,
            bound,
            copin_s,
            predicate.as_ref(),
            None,
            &order_by_exprs,
            &order_by_nulls_first,
            group_key_expr.as_ref(),
            &group_key_columns,
        )
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
        let NodeEnum::RangeVar(range) = node_enum(from)? else {
            return Err(sql_pg_error(
                "the GPU rank-window path supports one base relation".to_string(),
            ));
        };
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
        let table_name = range.relname.clone();
        let qualifier = range
            .alias
            .as_ref()
            .map(|alias| alias.aliasname.clone())
            .unwrap_or_else(|| table_name.clone());
        let statement_copin_s = self.committed_seq();
        let table = self
            .read_state
            .catalog_as_of(statement_copin_s)
            .relational_catalog
            .get(&table_name)
            .cloned()
            .ok_or_else(|| sql_pg_error(format!("relation \"{table_name}\" does not exist")))?;
        let input_predicate = stmt
            .where_clause
            .as_deref()
            .map(|node| map_predicate_node(node, &table, &qualifier))
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
                    let resolved_over =
                        resolve_rank_window_def(raw_over, &stmt.window_clause, 0)?;
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
                        _ => return Err(sql_pg_error("malformed window function name".to_string())),
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
                        if existing != &order
                            || window_order_nulls.as_ref() != Some(&order_nulls)
                        {
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
        if !top_order.is_empty()
            && (top_order != physical_order || top_nulls != physical_nulls)
        {
            return Err(sql_pg_error(
                "top-level ORDER BY must match PARTITION BY + window ORDER BY on this path"
                    .to_string(),
            ));
        }
        let limit = parse_limit(&stmt.limit_count)?;
        let offset = parse_limit(&stmt.limit_offset)?;
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
        let cold = self.ensure_streaming_join_cold(
            table,
            statement_copin_s,
            gpu_id,
            (budget / 2).max(1),
        )?;
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
                let validity = resident_device_null_column_offset(
                    &src.descriptor,
                    table,
                    column,
                )?;
                specs.push(match table.columns[column].ty {
                    SqlType::Text => {
                        let layout = resident_device_text_column_layout(
                            &src.descriptor,
                            table,
                            column,
                        )?;
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
                                resident_device_int8_column_offset(
                                    &src.descriptor,
                                    table,
                                    column,
                                )?,
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
                                resident_device_int4_column_offset(
                                    &src.descriptor,
                                    table,
                                    column,
                                )?,
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
                        lexicographic_16: table.columns
                            [relational_column_index(table, name)?]
                            .ty
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
                    let validity = resident_device_null_column_offset(
                        &src.descriptor,
                        table,
                        column,
                    )?;
                    let key = match ty {
                        SqlType::Text => {
                            let layout = resident_device_text_column_layout(
                                &src.descriptor,
                                table,
                                column,
                            )?;
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
        for chunk in cold.chunks.iter().filter(|chunk| {
            chunk.payload_copin_s <= statement_copin_s && chunk.row_count > 0
        }) {
            let (src, visibility) = match self.stage_cold_chunk(chunk, statement_copin_s).and_then(
                crate::engine_streaming_exec::StagedChunk::ready,
            ) {
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
            let source_keys = match source_order(
                &src,
                &physical_names,
                table,
                order,
                order_nulls,
            ) {
                Ok(keys) => keys,
                Err(err) => return Some(Err(err)),
            };
            let sorted = match src.device_memory.sort_join_coordinates(&identity, &source_keys) {
                Ok(value) => value,
                Err(err) => return Some(Err(map_err(err))),
            };
            let top = match src.device_memory.window_join_coordinates(
                &sorted,
                0,
                Some(fetch_u32),
            ) {
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
                            .saturating_add(
                                u64::from(combined.row_count()).saturating_mul(12),
                            ),
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
        let partition_keys = partition.iter().map(|name| rank_key(name)).collect::<Vec<_>>();
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
        let mut source_values =
            std::collections::BTreeMap::<String, Vec<SqlValue>>::new();
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
                                    SqlType::Int2 => SqlValue::Int2(
                                        i32::from_le_bytes(bytes.try_into().unwrap()) as i16,
                                    ),
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
                                    SqlType::Numeric { scale, .. } => SqlValue::Numeric(
                                        gpu_db_sql::Decimal128::new(
                                            i128::from_le_bytes(bytes.try_into().unwrap()),
                                            scale,
                                        ),
                                    ),
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
            let shifted = match run
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
                    let mut column = table.columns[relational_column_index(table, source).expect("bound")].clone();
                    column.name = output.clone(); column
                }
                GpuRankTarget::Window(_, name) => {
                    let mut column=table.columns[0].clone();column.name=name.clone();column.ty=SqlType::Int8;column.domain=None;column.default=None;column.type_oid=20;column.type_size=8;column
                }
                GpuRankTarget::OffsetWindow { source, output, .. } => {
                    let mut column = table.columns[relational_column_index(table, source).expect("bound")].clone();
                    column.name = output.clone();
                    column
                }
            };column.attnum=(attnum+1) as i16;columns.push(column);
        }
        let rows=(0..window.row_count() as usize).map(|row|targets.iter().enumerate().map(|(target_index,target)|match target{
            GpuRankTarget::Column{source,..}=>source_values[source][row].clone(),
            GpuRankTarget::Window(kind,_)=>SqlValue::Int8(rank_values[&(*kind as u8)][row] as i64),
            GpuRankTarget::OffsetWindow { .. } => offset_values[&target_index][row].clone(),
        }).collect::<Vec<_>>()).collect::<Vec<_>>();
        peak = peak.max(input_peak.saturating_add(allocation_scope.peak_bytes()));
        self.read_state.residency.streaming_fold_peak_chunk_bytes.fetch_max(peak,std::sync::atomic::Ordering::Relaxed);
        self.read_state.residency.streaming_window_hits.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
        Some(Ok(RelationalSelectResult{columns:Arc::new(columns),rows:rows.into(),planned_target:DeviceTarget::Gpu(gpu_id),executed_target:DeviceTarget::Gpu(gpu_id),fallback_reason:None,access_path:Arc::new(RelationalAccessPath::FullTableScan)}))
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
        use gpu_db_execution::{
            CudaJoinOrderKey, CudaJoinPayloadKey, CudaWindowRankKind,
        };
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
            let validity_bitmap_offset = resident_device_null_column_offset(
                &side.0.descriptor,
                table,
                column,
            )?;
            let key = match ty {
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(
                        &side.0.descriptor,
                        table,
                        column,
                    )?;
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
                    return Err(sql_pg_error(
                        "bool rank keys are not supported".to_string(),
                    ))
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
                SqlType::Int2
                    | SqlType::Int4
                    | SqlType::Int8
                    | SqlType::Date
                    | SqlType::Timestamp
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
                4 + 8 + 24
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
        estimated_live = input_bytes.saturating_add(
            estimated_live.max(allocation_scope.peak_bytes()),
        );
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
            let validity = resident_device_null_column_offset(
                &side.0.descriptor,
                table,
                column,
            )?;
            let ty = table.columns[column].ty;
            let values = match ty {
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(
                        &side.0.descriptor,
                        table,
                        column,
                    )?;
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
                    let bitmap = resident_device_bool_column_offset(
                        &side.0.descriptor,
                        table,
                        column,
                    )?;
                    payload
                        .project_bool_from_join_coordinates(
                            &window,
                            0,
                            payload,
                            bitmap,
                            validity,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(|value| value.map_or(SqlValue::Null, SqlValue::Bool))
                        .collect()
                }
                _ => {
                    let (byte_offset, width) = match ty {
                        SqlType::Int8 | SqlType::Timestamp => (
                            resident_device_int8_column_offset(
                                &side.0.descriptor,
                                table,
                                column,
                            )?,
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
                            resident_device_int4_column_offset(
                                &side.0.descriptor,
                                table,
                                column,
                            )?,
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
                                SqlType::Int2 => SqlValue::Int2(
                                    i32::from_le_bytes(bytes.try_into().unwrap()) as i16,
                                ),
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
                                SqlType::Numeric { scale, .. } => SqlValue::Numeric(
                                    gpu_db_sql::Decimal128::new(
                                        i128::from_le_bytes(bytes.try_into().unwrap()),
                                        scale,
                                    ),
                                ),
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

fn resolve_rank_window_def(
    over: &pg_query::protobuf::WindowDef,
    clauses: &[Node],
    depth: usize,
) -> Result<pg_query::protobuf::WindowDef, ExecuteError> {
    fn named<'a>(
        name: &str,
        clauses: &'a [Node],
    ) -> Result<&'a pg_query::protobuf::WindowDef, ExecuteError> {
        clauses
            .iter()
            .find_map(|node| match node.node.as_ref() {
                Some(NodeEnum::WindowDef(window)) if window.name == name => {
                    Some(window.as_ref())
                }
                _ => None,
            })
            .ok_or_else(|| sql_pg_error(format!("window \"{name}\" does not exist")))
    }
    fn source(
        window: &pg_query::protobuf::WindowDef,
        clauses: &[Node],
        depth: usize,
    ) -> Result<pg_query::protobuf::WindowDef, ExecuteError> {
        if depth > clauses.len().saturating_add(1) {
            return Err(sql_pg_error("cyclic named WINDOW reference".to_string()));
        }
        let mut resolved = if window.refname.is_empty() {
            window.clone()
        } else {
            let mut base = source(named(&window.refname, clauses)?, clauses, depth + 1)?;
            if !window.partition_clause.is_empty() {
                base.partition_clause = window.partition_clause.clone();
            }
            if !window.order_clause.is_empty() {
                base.order_clause = window.order_clause.clone();
            }
            if window.frame_options & 0x00001 != 0 {
                base.frame_options = window.frame_options;
                base.start_offset = window.start_offset.clone();
                base.end_offset = window.end_offset.clone();
            }
            base
        };
        resolved.name.clear();
        resolved.refname.clear();
        Ok(resolved)
    }
    if over.name.is_empty() {
        source(over, clauses, depth)
    } else {
        source(named(&over.name, clauses)?, clauses, depth + 1)
    }
}

fn select_has_inline_window(stmt: &SelectStmt) -> Result<bool, ExecuteError> {
    for target in &stmt.target_list {
        let NodeEnum::ResTarget(target) = node_enum(target)? else {
            continue;
        };
        let Some(value) = target.val.as_deref() else {
            continue;
        };
        if matches!(node_enum(value)?, NodeEnum::FuncCall(func) if func.over.is_some()) {
            return Ok(true);
        }
    }
    Ok(false)
}

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
fn build_select_from_select_stmt(stmt: &SelectStmt) -> Result<(Select, String), ExecuteError> {
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
    let table = range_var.relname.clone();
    // The single qualifier a `tbl.col` reference must use: PG hides the relation name behind an alias,
    // so an aliased relation is named only by its alias; an unaliased one by its relation name. The
    // mapper validates qualified column references against this and rejects any other (PG's "missing
    // FROM-clause entry"), so a stray qualifier is never silently resolved to this relation's column.
    let qualifier = range_var
        .alias
        .as_ref()
        .map(|alias| alias.aliasname.clone())
        .unwrap_or_else(|| table.clone());

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
    let order_by = parse_order_by(&stmt.sort_clause, &qualifier)?;
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
        columns.push(resolve_column_name(column_ref, qualifier)?.to_string());
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
    if !res_target.name.is_empty() {
        return Err(sql_pg_error(
            "aggregate column aliases are not on the Expr path yet".to_string(),
        ));
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

/// Map a WHERE / predicate parse node to the general `ResidentExpr` IR, resolving column names to
/// indices against `table`. Recurses through `A_Expr` (arithmetic + comparison). Anything the IR
/// cannot represent — `BoolExpr` (`AND`/`OR`, slice 4), non-operator `A_Expr` (IN / LIKE / BETWEEN),
/// non-integer literals, function calls, subqueries — is a hard error so the routing layer never
/// silently mis-answers.
fn map_predicate_node(
    node: &Node,
    table: &RelationalTable,
    qualifier: &str,
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
        NodeEnum::AExpr(a_expr) => map_a_expr(a_expr, table, qualifier),
        NodeEnum::BoolExpr(bool_expr) => map_bool_expr(bool_expr, table, qualifier),
        NodeEnum::NullTest(null_test) => map_null_test(null_test, table, qualifier),
        _ => Err(sql_pg_error(
            "unsupported expression node for the general GPU executor".to_string(),
        )),
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
        let inner = map_predicate_node(only, table, qualifier)?;
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
    let mut folded = map_predicate_node(first, table, qualifier)?;
    for arg in args {
        folded = ResidentExpr::Binary {
            op,
            lhs: Box::new(folded),
            rhs: Box::new(map_predicate_node(arg, table, qualifier)?),
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
) -> Result<ResidentExpr, ExecuteError> {
    // AEXPR_IN is `x IN (a, b, ...)` -> an OR-chain of equalities (`NOT IN` -> an AND-chain of `<>`),
    // entirely on the general GPU executor (the same Binary Eq/Ne + And/Or it already runs).
    if a_expr.kind == AExprKind::AexprIn as i32 {
        return map_in_expr(a_expr, table, qualifier);
    }
    // AEXPR_OP is a normal operator (`+ - * = <> < <= > >=`); AEXPR_LIKE is `LIKE` (operator `~~`,
    // `!~~` for NOT LIKE). Both carry the operator token in `name` and both operands; other kinds
    // (`BETWEEN` / ...) are rejected.
    if a_expr.kind != AExprKind::AexprOp as i32 && a_expr.kind != AExprKind::AexprLike as i32 {
        return Err(sql_pg_error(
            "only operator, LIKE, and IN predicates are supported (no BETWEEN yet)".to_string(),
        ));
    }
    let op = map_operator(aexpr_op_token(a_expr)?)?;
    let lexpr = a_expr.lexpr.as_deref().ok_or_else(|| {
        sql_pg_error("operator expression is missing its left operand".to_string())
    })?;
    let rexpr = a_expr.rexpr.as_deref().ok_or_else(|| {
        sql_pg_error(
            "operator expression is missing its right operand (unary operators unsupported)"
                .to_string(),
        )
    })?;
    Ok(ResidentExpr::Binary {
        op,
        lhs: Box::new(map_predicate_node(lexpr, table, qualifier)?),
        rhs: Box::new(map_predicate_node(rexpr, table, qualifier)?),
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
    let lhs = map_predicate_node(lexpr, table, qualifier)?;
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
            rhs: Box::new(map_predicate_node(item, table, qualifier)?),
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
fn node_enum(node: &Node) -> Result<&NodeEnum, ExecuteError> {
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
fn resolve_column_name<'a>(
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
fn aexpr_op_token(a_expr: &AExpr) -> Result<&str, ExecuteError> {
    match a_expr.name.as_slice() {
        [name] => match name.node.as_ref() {
            Some(NodeEnum::String(string)) => Ok(&string.sval),
            _ => Err(sql_pg_error(
                "operator name is not a String node".to_string(),
            )),
        },
        _ => Err(sql_pg_error(
            "schema-qualified or multi-part operators are not supported".to_string(),
        )),
    }
}

/// The RESULT column a GROUP BY ORDER BY / HAVING term references: a bare column (the group key) by its
/// name, or an aggregate by the name the binding gives its result column (count/sum/min/max/avg). Used
/// to look the value up by index in the materialized grouped row. (A projection with two of the same
/// aggregate function makes that name ambiguous -- the first match wins; uncommon.)
fn result_column_name(node: &Node, qualifier: &str) -> Result<String, ExecuteError> {
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
fn parse_order_by(sort_clause: &[Node], qualifier: &str) -> Result<Vec<SelectOrder>, ExecuteError> {
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

fn parse_order_by_null_placement(
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
fn parse_limit(limit: &Option<Box<Node>>) -> Result<Option<usize>, ExecuteError> {
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
fn sql_pg_error(message: String) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(message))
}
