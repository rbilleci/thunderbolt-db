//! General GPU executor — the `Expr` IR and its device interpreter (Charter rule 2;
//! `docs/architecture/17-general-gpu-executor.md`). This is the GENERAL path that replaces the
//! enumerated `execute_relational_*_with_resident_device_memory_probe` shape methods: a query is an
//! expression tree, lowered to a pipeline of device primitives, NOT matched against a fixed catalog
//! of shapes.
//!
//! `ResidentExpr` is the general scalar IR for the supported resident SQL types. Lowering compiles
//! arithmetic, comparison, boolean, NULL, and text predicates into typed device VM/operators; coverage
//! grows by expression node and type, never by adding another whole-query shape method.
//!
//! The IR + op-code maps + `execute_resident_expr_select_with_binding` are now the production path the
//! SQL->Expr binding (`engine_sql_pg`) routes into; the GPU parity tests exercise the same lowering
//! via programmatic `ResidentExpr`s. The 2-arg `execute_resident_expr_select` convenience wrapper (run
//! a select with a programmatic predicate, binding internally) has no production caller yet — a
//! prepared-route / facade caller is the likely one — so it carries a targeted `allow(dead_code)`.

use super::*;

pub(crate) use crate::engine_expr_ir::{ResidentBinaryOp, ResidentExpr};

pub(crate) use crate::engine_join_ir::{
    JoinColRef, JoinPlan, JoinProjItem, JoinRelationRef, JoinStep,
};

mod normalization;
use normalization::{having_op_to_resident, having_value_to_resident_literal};
pub(crate) use normalization::{
    grouped_projection_to_aggregates, like_pattern_for_literal_prefix,
    resident_predicate_from_bound_filters,
};

mod shard_pruning;
pub(crate) use shard_pruning::shard_point_lookup_int4_eq;

mod grouped_values;
use grouped_values::{composite_group_count_reps, narrow_ordered_value};
mod grouped_count_distinct;
use grouped_count_distinct::count_distinct_groups;
mod scalar_aggregate;
use scalar_aggregate::execute_scalar_aggregate;
mod projected_rows;
use projected_rows::materialize_projected_rows;

mod execution_source;
pub(crate) use execution_source::{
    ResidentExecSource, ResidentVisibility, ShardedUnifiedExecSource,
};

mod sharded_source;
mod sharded_route;
mod select_bridge;
#[cfg(test)]
mod group_bench;

mod join_source;
pub(crate) use join_source::{JoinDeviceMemory, JoinExecSide};

mod join_side;
mod join_plan;
mod join_incremental;
mod join_projection;
mod join_coordinate_filter;
mod join_coordinate_exec;

mod predicate_operands;
use predicate_operands::{expr_mentions_int4_column, expr_mentions_int8};

mod predicate_compiler;
use predicate_compiler::{
    collect_expr_columns, compile_arith_program, predicate_references_nullable_column,
    push_leaf_validity_and,
};

mod predicate_mask;
mod predicate_dispatch;
mod predicate_typed_lowering;

mod resident_dml;

pub(crate) use crate::engine_result_sort::gpu_sort_permutation;

impl Engine {
    /// As [`Engine::execute_resident_expr_select`] but over an ALREADY-BOUND table/projection: the
    /// caller bound the catalog once and resolved the predicate's `Column` indices against this SAME
    /// `table`. The SQL->Expr entry (`engine_sql_pg`) routes through here so the predicate's column
    /// indices, the projection, and the residency snapshot all derive from one catalog generation — a
    /// concurrent shape-changing DDL cannot split the column resolution from the execution.
    #[allow(clippy::too_many_arguments)] // group_key_expr is threaded alongside the predicate/binding
    pub(crate) fn execute_resident_expr_select_with_binding(
        &self,
        select: &Select,
        table: &RelationalTable,
        // `None` = look up the table payload (descriptor + device buffer + row count) by name in the
        // SINGLE resident store (the whole-table buffer), as before. `Some(src)` INJECTS them so the
        // same executor serves one shard slice (S10c). The identity/validity guards run for both.
        src: Option<&ResidentExecSource>,
        bound: BoundRelationalSelect,
        copin_s: Index,
        // `None` = no WHERE clause: a full-table scan (every row survives).
        predicate: Option<&ResidentExpr>,
        // SV3b/SV6 (MVCC visibility): `Some` = this (unified) buffer carries co-resident i64 version
        // column(s); AND `deleted_by > read_txn_id` (hide tombstoned rows) and/or `created_by <=
        // read_txn_id` (hide too-new appended versions) onto the survivors. `None` = a version-free buffer
        // (the common case) -> no visibility mask, byte-identical. Only the sharded read passes `Some`.
        visibility: Option<ResidentVisibility>,
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
        // INJECTS them (S10c: one shard slice). The identity + validity guards run for BOTH, so a
        // descriptor that drifted from the catalog or got invalidated is rejected either way. No
        // `host_rows` read on this path: a text GROUP BY key result is materialized ON-DEVICE.
        // THE FLIP: a SHARD-resident table (no single-buffer entry) resolves to the UNIFIED exec source
        // HERE — one resolution point for every `src: None` caller (the SQL->Expr PG path, the
        // parity-test wrapper, the CTAS/view bridges), so the general executor serves sharded tables
        // uniformly (zero-copy at one surviving shard; recompaction otherwise). The unified source
        // carries its own SV3b/SV6 visibility, which OVERRIDES the caller's `None`. R-ver PART 2: a
        // VERSIONED table with a reshaping clause (DISTINCT / GROUP BY / ORDER BY / HAVING) is now
        // SERVED — the visibility resolved below flows into the survivor `indices` (folded in BEFORE
        // group/sort/dedup), so no tombstoned/too-new row reaches a key. Single-buffer tables and
        // `src: Some` callers are byte-identical.
        let sharded_unified: Option<ShardedUnifiedExecSource> = if src.is_none()
            && self.relational_residency_entry(&table.name).is_none()
            && self
                .read_state
                .residency
                .shards
                .load()
                .get(&table.name)
                .is_some_and(|shards| !shards.is_empty())
        {
            Some(self.build_sharded_unified_exec_source(table, predicate, copin_s)?)
        } else {
            None
        };
        debug_assert!(
            sharded_unified.is_none() || visibility.is_none(),
            "a caller passed src: None + visibility: Some for a sharded table — the unified source's \
             visibility would silently replace it (audit P3 hardening; thread it via src instead)"
        );
        let visibility = sharded_unified
            .as_ref()
            .map_or(visibility, |unified| unified.visibility);
        let src = src.or(sharded_unified.as_ref().map(|unified| &unified.src));
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
                let residency_entry =
                    self.relational_residency_entry(&table.name)
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

        // PG aggregate semantics skip NULL inputs: SUM/AVG/MIN/MAX(col) aggregate the NON-NULL values
        // (all-NULL -> SQL NULL via the empty-set guard below) and COUNT(DISTINCT col) counts distinct
        // NON-NULL values. The scalar reductions below are RAW payload reduces (a NULL row's placeholder
        // is 0 — it would poison MIN toward 0 and inflate AVG's divisor), so when the aggregate column
        // carries a validity bitmap, AND `col IS NOT NULL` into the predicate: the survivors then
        // exclude NULL inputs BEFORE the reduce, AVG's divisor (`indices.len()`) is exactly the non-NULL
        // count, and the existing IsNull lowering + 3VL VM (+ the SV3b/SV6 visibility conjuncts, when
        // versioned) do all the work. COUNT(*) is deliberately NOT augmented (it counts NULL rows). A
        // bitmap-free column (the no-NULLs majority — bitmaps are data-driven) adds no conjunct and the
        // full-scan `(0..n)` fast arm below is untouched: byte-identical to before.
        //
        // LEDGERED trade (audit F1): the conjunct is UNCONDITIONAL. On the no-fallback text path, a
        // nullable NON-int4 aggregate column (or an int4 aggregate whose WHERE makes the augmented
        // predicate mixed-width) is no longer VM-lowerable -> CLEAN ERROR where the pre-fix code
        // returned a NULL-BLIND value (wrong for MIN/MAX/AVG, coincidentally right for SUM). A clean
        // error strictly dominates a silent wrong result; do NOT "fix" this by dropping the conjunct
        // on un-lowerable shapes — route those to the (now NULL-correct) host finalizer instead when
        // the coverage gap matters (type-coverage ledger).
        let aggregate_validity_conjunct: Option<ResidentExpr> = match &select.projection {
            SelectProjection::Sum { column }
            | SelectProjection::Avg { column }
            | SelectProjection::Min { column }
            | SelectProjection::Max { column }
            | SelectProjection::CountDistinct { column } => {
                let col_idx = relational_column_index(table, column)?;
                resident_device_null_column_offset(&snapshot, table, col_idx)?.map(|_| {
                    ResidentExpr::IsNull {
                        col: col_idx,
                        is_not_null: true,
                    }
                })
            }
            _ => None,
        };
        let augmented_predicate: Option<ResidentExpr> =
            aggregate_validity_conjunct.map(|validity| match predicate {
                Some(pred) => ResidentExpr::Binary {
                    op: ResidentBinaryOp::And,
                    lhs: Box::new(pred.clone()),
                    rhs: Box::new(validity),
                },
                None => validity,
            });
        let predicate = augmented_predicate.as_ref().or(predicate);

        // Evaluate the predicate on the GPU -> surviving row indices (ascending). With no WHERE clause
        // every row survives, so the indices are the full 0..row_count scan (the aggregate + projection
        // paths below are index-driven and need no other change).
        let indices = {
            // probe-timing (VM lever): the WHOLE predicate eval (compare kernel + compact). `compact`
            // (timed separately in compact_mask_*) vs this total localizes where the predicate cost lives.
            let _pred_scope = gpu_db_execution::Probe::scope("predicate_total");
            match predicate {
                Some(predicate) => self.lower_resident_predicate(
                    predicate,
                    table,
                    &snapshot,
                    &device_memory,
                    row_count,
                    visibility,
                )?,
                None => match visibility {
                    // SV3b/SV6: no WHERE but a versioned buffer -> the survivors are exactly the VISIBLE
                    // rows. Run a visibility-only VM program (`deleted_by > read_txn_id` and/or
                    // `created_by <= read_txn_id`) at elem=I64 -- the same mixed-width interpreter, with no
                    // i32 WHERE mask to AND against (the first conjunct IS the mask).
                    Some(vis) => {
                        let mut program: Vec<gpu_db_execution::ExprStep> = Vec::new();
                        vis.push_conjuncts(&mut program, false);
                        device_memory
                            .run_expr_predicate_filter_with_text(
                                &program,
                                &[],
                                row_count,
                                gpu_db_execution::ResidentElemType::I64,
                            )
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                            })?
                    }
                    None => {
                        let n = u32::try_from(row_count).map_err(|_| {
                            ExecuteError::Engine(EngineError::ApplyFailed(
                                "full-table scan row count exceeds the u32 row-index range"
                                    .to_string(),
                            ))
                        })?;
                        (0..n).collect()
                    }
                },
            }
        };
        let indices_u64: Vec<u64> = indices.iter().map(|&i| u64::from(i)).collect();


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
            // QUERY-AWARE AGGREGATE PRUNING (this slice): the DIRECT (hash) GROUP BY kernel computes
            // count+sum+min+max for one value column, but the executor reads only the field(s) the query
            // needs. Build a query-wide mask (`grouped_agg_mask`: 1=COUNT 2=SUM 4=MIN 8=MAX) = the OR of
            // every aggregate's field set, and pass it to EVERY direct pass (each computes a superset of
            // what its own column needs, and the executor reads only computed fields, so one shared mask
            // is correct). Bit derivation per aggregate kind:
            //   Count            -> COUNT
            //   Sum / Avg        -> SUM   (Avg also divides by count)
            //   Min              -> MIN
            //   Max              -> MAX
            //   CountDistinct    -> {} here (it runs a SEPARATE sort/mark/SUM pass; the direct kernel
            //                       reads NOTHING for it -- its passes always run their own ALL mask)
            // COUNT is FORCED ON whenever ANY value aggregate (Sum/Avg/Min/Max) is present, because the
            // result builder reads `g.count` for EVERY value pass to detect an all-NULL group (count==0 ->
            // SQL NULL); the count slot inits to 0, so a masked-out COUNT would read 0 and wrongly NULL
            // every group. (Avg already requires COUNT for its divisor.)
            //
            // COMPLETENESS vs HAVING / aggregate-ORDER-BY (the correctness-critical claim): this mask covers
            // them WITHOUT an ALL-fallback, and here is the proof. HAVING and ORDER BY on a grouped result
            // do NOT read the kernel's count/sum/min/max directly -- they consume the already-materialized
            // result `rows` (HAVING builds a transient device relation from `rows`; ORDER BY GPU-sorts result
            // COLUMNS). Both resolve their referenced column via `result_column_name` to a NAME, which the
            // executor's `col_index` maps against `bound.selected_columns` (the SELECT result columns). A
            // HAVING/ORDER-BY reference to an aggregate that is NOT a SELECT result column is a hard error
            // ("references unknown column"), so EVERY aggregate HAVING/ORDER-BY can reach is necessarily
            // already in the SELECT list -> already in `aggregates` -> already in this mask. (The grouped SQL
            // binder also builds `aggregates` only from the SELECT target list; it adds no HAVING/ORDER-BY
            // aggregate, which is exactly why such an unlisted reference errors rather than introducing a new
            // kernel field read.) Hence the OR-over-`aggregates` mask is the COMPLETE set the executor reads.
            let agg_mask: u32 = {
                use gpu_db_execution::grouped_agg_mask as gm;
                let mut m = 0u32;
                let mut any_value_agg = false;
                for a in aggregates.iter() {
                    match a.kind {
                        GroupedAggKind::Count => m |= gm::COUNT,
                        GroupedAggKind::Sum | GroupedAggKind::Avg => {
                            m |= gm::SUM;
                            any_value_agg = true;
                        }
                        GroupedAggKind::Min => {
                            m |= gm::MIN;
                            any_value_agg = true;
                        }
                        GroupedAggKind::Max => {
                            m |= gm::MAX;
                            any_value_agg = true;
                        }
                        // COUNT(DISTINCT) reads nothing from the direct kernel (separate pass).
                        GroupedAggKind::CountDistinct => {}
                    }
                }
                if any_value_agg {
                    m |= gm::COUNT; // the all-NULL-group check reads g.count on every value pass
                }
                m
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
            let composite_cols: Option<(usize, SqlType, usize, SqlType)> = match &composite_members
            {
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
            let composite_is_text = composite_cols.is_some_and(|(_, t0, _, t1)| {
                matches!(t0, SqlType::Text) != matches!(t1, SqlType::Text)
            });
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
            let key_is_text =
                composite_is_text || (!is_expr_key && matches!(key_ty, SqlType::Text));
            // A bool key is materialized bool->int4 (0/1) into a derived buffer + grouped via
            // key_base_override (like an expression key); int4-width, never i128/text.
            let key_is_bool = !is_expr_key && matches!(key_ty, SqlType::Bool);
            let (key_offsets_off, key_bytes_off, key_bytes_len) = if composite_is_text {
                // The text member's varlen column (the other member rides key_base_override).
                let (c0, t0, c1, _) = composite_cols.unwrap();
                let text_col = if matches!(t0, SqlType::Text) { c0 } else { c1 };
                let layout = resident_device_text_column_layout(&snapshot, table, text_col)?;
                (
                    layout.offsets_byte_offset,
                    layout.bytes_byte_offset,
                    layout.bytes_len,
                )
            } else if key_is_text {
                let layout = resident_device_text_column_layout(&snapshot, table, group_idx)?;
                (
                    layout.offsets_byte_offset,
                    layout.bytes_byte_offset,
                    layout.bytes_len,
                )
            } else {
                (0, 0, 0)
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
            // buffer (checked overflow -> PG error) + hold its typed view alive across EVERY pass.
            let _derived_key_buf;
            if let Some(expr) = group_key_expr {
                if row_count == 0 {
                    // Empty input: the on-device arith materialize rejects n=0, and there are no rows to
                    // group anyway. Skip it and group 0 rows -> 0 groups (PG returns no rows), matching
                    // the plain-column path's empty-table behavior.
                    _derived_key_buf = None;
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
                        .map_err(|e| {
                            ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
                        })?;
                    _derived_key_buf = Some(buf);
                }
            } else if key_is_bool {
                if row_count == 0 {
                    _derived_key_buf = None;
                } else {
                    // GROUP BY a bool column: materialize bool->int4 (0/1) into a typed derived buffer.
                    // No bool GROUP BY kernel -> no hazard.
                    let offset = resident_device_bool_column_offset(&snapshot, table, group_idx)?;
                    let buf = device_memory
                        .bool_to_int4_column_device(offset, row_count)
                        .map_err(|e| {
                            ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
                        })?;
                    _derived_key_buf = Some(buf);
                }
            } else if is_composite_key {
                if row_count == 0 {
                    _derived_key_buf = None;
                } else {
                    // GROUP BY a, b: pack the two members into one derived key; the result UNPACKS it.
                    // Both-int4 -> one i64 ((col0<<32)|col1);
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
                    _derived_key_buf = Some(buf);
                }
            } else if composite_is_widekey {
                if row_count == 0 || widekey_w == 0 {
                    // Empty input, OR a pure all-text composite (no fixed members) -> no fixed wide-key
                    // buffer; the text members are grouped via the text descriptor.
                    _derived_key_buf = None;
                } else {
                    // General composite: build the widekey_w-byte/row FIXED wide key (each fixed member's
                    // canonical bytes concatenated -- int 8B, numeric/uuid 16B; TEXT members are skipped
                    // here, handled by the text descriptor) -> grouped via the (rep_idx, hash) b128 claim
                    // (comp_w = widekey_w). The result reads each member from the rep row.
                    let members = widekey_cols.as_ref().expect("composite_is_widekey");
                    let mut descriptors: Vec<gpu_db_execution::CudaWideKeyDescriptor<'_>> =
                        Vec::with_capacity(members.len());
                    // M3 (doc 21): per-FIXED-member NULL validity offsets, lockstep with `descriptors`
                    // (text members are skipped in both). Empty unless the composite is nullable.
                    let mut validity_descs: Vec<gpu_db_execution::CudaWideKeyValidity> = Vec::new();
                    let mut dst_off: u64 = 0;
                    for &(idx, ty) in members {
                        let (source, w) = match ty {
                            SqlType::Numeric { .. } | SqlType::Uuid => (
                                gpu_db_execution::CudaWideKeySource::ResidentI128 {
                                    byte_offset: resident_device_numeric_column_offset(
                                        &snapshot, table, idx,
                                    )?,
                                },
                                16u64,
                            ),
                            SqlType::Int8 | SqlType::Timestamp => (
                                gpu_db_execution::CudaWideKeySource::ResidentI64 {
                                    byte_offset: resident_device_int8_column_offset(
                                        &snapshot, table, idx,
                                    )?,
                                },
                                8u64,
                            ),
                            // A bool member (1-byte resident) is widened 0/1 -> i64 by build kind 3.
                            SqlType::Bool => (
                                gpu_db_execution::CudaWideKeySource::ResidentBool {
                                    bitmap_byte_offset: resident_device_bool_column_offset(
                                        &snapshot, table, idx,
                                    )?,
                                },
                                8u64,
                            ),
                            // Text members are not in the fixed buffer (they ride the text descriptor).
                            SqlType::Text => continue,
                            _ => (
                                gpu_db_execution::CudaWideKeySource::ResidentI32 {
                                    byte_offset: resident_device_int4_column_offset(
                                        &snapshot, table, idx,
                                    )?,
                                },
                                8u64,
                            ),
                        };
                        descriptors.push(gpu_db_execution::CudaWideKeyDescriptor {
                            source,
                            destination_byte_offset: dst_off,
                        });
                        if composite_key_has_null {
                            validity_descs.push(
                                match resident_device_null_column_offset(&snapshot, table, idx)? {
                                    Some(byte_offset) => {
                                        gpu_db_execution::CudaWideKeyValidity::Bitmap {
                                            byte_offset,
                                        }
                                    }
                                    None => gpu_db_execution::CudaWideKeyValidity::NonNullable,
                                },
                            );
                        }
                        dst_off += w;
                    }
                    let buf = device_memory
                        .build_wide_key_device(&descriptors, widekey_w, row_count, &validity_descs)
                        .map_err(|e| {
                            ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
                        })?;
                    _derived_key_buf = Some(buf);
                }
            } else {
                _derived_key_buf = None;
            }
            // General composite TEXT members: each text member's (offsets_off, bytes_off, bytes_len) ->
            // a small device descriptor the wide-key claim bounds, hashes, and verifies (row vs rep),
            // in DECLARED-relative order. Built once + held alive across the pass. No text members
            // (all-fixed wide key) -> n_text = 0, byte-identical to the prior wide-key path.
            let _widekey_text_desc = match &widekey_cols {
                Some(members) if row_count > 0 => {
                    let mut sources = Vec::new();
                    for &(idx, ty) in members {
                        if matches!(ty, SqlType::Text) {
                            let layout = resident_device_text_column_layout(&snapshot, table, idx)?;
                            sources.push(gpu_db_execution::CudaGroupTextSource {
                                offsets_byte_offset: layout.offsets_byte_offset,
                                bytes_byte_offset: layout.bytes_byte_offset,
                                bytes_len: layout.bytes_len,
                                row_count,
                            });
                        }
                    }
                    if sources.is_empty() {
                        None
                    } else {
                        Some(
                            device_memory
                                .upload_group_text_descriptors(&sources)
                                .map_err(map_err)?,
                        )
                    }
                }
                _ => None,
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
            let run_pass = |value_idx_opt: Option<usize>,
                            has_minmax: bool|
             -> Result<Pass, ExecuteError> {
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
                let value_is_bool = value_idx_opt.is_some() && matches!(value_ty, SqlType::Bool);
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
                let _derived_value_buf = if value_is_bool && row_count > 0 {
                    let offset = resident_device_bool_column_offset(&snapshot, table, value_idx)?;
                    let buf = device_memory
                        .bool_to_int4_column_device(offset, row_count)
                        .map_err(map_err)?;
                    Some(buf)
                } else {
                    None
                };
                let (value_offsets_off, value_bytes_off, value_bytes_len) = if value_is_text {
                    let layout = resident_device_text_column_layout(&snapshot, table, value_idx)?;
                    (
                        layout.offsets_byte_offset,
                        layout.bytes_byte_offset,
                        layout.bytes_len,
                    )
                } else {
                    (0, 0, 0)
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
                    let key = if composite_is_widekey {
                        gpu_db_execution::CudaGroupKeySource::Composite {
                            fixed: _derived_key_buf.as_ref().map(|buf| {
                                gpu_db_execution::CudaGroupWideSource {
                                    buffer: buf.group_view(),
                                    row_width: widekey_w,
                                    row_count,
                                }
                            }),
                            text: _widekey_text_desc.as_ref().map(|buf| buf.descriptors()),
                            row_count,
                        }
                    } else if key_is_text {
                        gpu_db_execution::CudaGroupKeySource::Text {
                            text: gpu_db_execution::CudaGroupTextSource {
                                offsets_byte_offset: key_offsets_off,
                                bytes_byte_offset: key_bytes_off,
                                bytes_len: key_bytes_len,
                                row_count,
                            },
                            fixed_component: if composite_is_text {
                                Some(gpu_db_execution::CudaGroupFixedSource::Derived {
                                    buffer: _derived_key_buf
                                        .as_ref()
                                        .expect("fixed text component")
                                        .group_view(),
                                    width: 8,
                                    row_count,
                                })
                            } else {
                                None
                            },
                        }
                    } else {
                        let width = if key_is_i128 {
                            16
                        } else if key_is_int8 {
                            8
                        } else {
                            4
                        };
                        let source = if let Some(buf) = &_derived_key_buf {
                            gpu_db_execution::CudaGroupFixedSource::Derived {
                                buffer: buf.group_view(),
                                width,
                                row_count,
                            }
                        } else {
                            gpu_db_execution::CudaGroupFixedSource::Resident {
                                byte_offset: key_offset,
                                width,
                                row_count,
                            }
                        };
                        gpu_db_execution::CudaGroupKeySource::Fixed(source)
                    };
                    let value = if value_idx_opt.is_none() {
                        gpu_db_execution::CudaGroupValueSource::Unused { row_count }
                    } else if value_is_text {
                        gpu_db_execution::CudaGroupValueSource::Text(
                            gpu_db_execution::CudaGroupTextSource {
                                offsets_byte_offset: value_offsets_off,
                                bytes_byte_offset: value_bytes_off,
                                bytes_len: value_bytes_len,
                                row_count,
                            },
                        )
                    } else {
                        let width = if value_is_numeric || value_is_uuid {
                            16
                        } else if value_is_int8 {
                            8
                        } else {
                            4
                        };
                        let source = if let Some(buf) = &_derived_value_buf {
                            gpu_db_execution::CudaGroupFixedSource::Derived {
                                buffer: buf.group_view(),
                                width,
                                row_count,
                            }
                        } else {
                            gpu_db_execution::CudaGroupFixedSource::Resident {
                                byte_offset: value_offset,
                                width,
                                row_count,
                            }
                        };
                        if value_is_numeric {
                            gpu_db_execution::CudaGroupValueSource::Numeric(source)
                        } else if value_is_uuid {
                            gpu_db_execution::CudaGroupValueSource::Uuid(source)
                        } else {
                            gpu_db_execution::CudaGroupValueSource::Fixed(source)
                        }
                    };
                    device_memory.group_by_i32_count_sum_minmax_from_payload(
                        gpu_db_execution::CudaGroupByInput {
                            key,
                            value,
                            key_validity_bitmap_offset: pass_key_null_off,
                            value_validity_bitmap_offset: pass_value_null_off,
                        },
                        &indices,
                        // The query-wide pruning mask: this pass computes a superset of what its column
                        // needs; the executor reads only the masked-in field(s) it requested.
                        if value_idx_opt.is_none() {
                            gpu_db_execution::grouped_agg_mask::COUNT
                        } else {
                            agg_mask
                        },
                    )
                } else {
                    device_memory.group_by_i32_count_sum_from_payload(
                        gpu_db_execution::CudaGroupByInput::resident_i32(
                            key_offset,
                            value_offset,
                            row_count,
                        ),
                        &indices,
                        agg_mask,
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
                    // Non-plain-integer/composite group-key route: reduce COUNT(DISTINCT v) per g to
                    // counting DISTINCT (g, v)
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
                                if is_expr_key {
                                    Some((
                                        _derived_key_buf
                                            .as_ref()
                                            .expect("expression group key buffer")
                                            .group_view(),
                                        key_expr_is_int8,
                                    ))
                                } else {
                                    None
                                },
                            )?;
                            // (2) GROUP BY g over the reps (reusing the main g config) COUNT(*).
                            let mut g2 = if reps.is_empty() {
                                Vec::new()
                            } else {
                                device_memory
                                    .group_by_i32_count_sum_minmax_from_payload(
                                        gpu_db_execution::CudaGroupByInput {
                                            key: if composite_is_widekey {
                                                gpu_db_execution::CudaGroupKeySource::Composite {
                                                    fixed: _derived_key_buf.as_ref().map(|buf| gpu_db_execution::CudaGroupWideSource {
                                                        buffer: buf.group_view(), row_width: widekey_w, row_count,
                                                    }),
                                                    text: _widekey_text_desc.as_ref().map(|buf| buf.descriptors()),
                                                    row_count,
                                                }
                                            } else if key_is_text {
                                                gpu_db_execution::CudaGroupKeySource::Text {
                                                    text: gpu_db_execution::CudaGroupTextSource {
                                                        offsets_byte_offset: key_offsets_off,
                                                        bytes_byte_offset: key_bytes_off,
                                                        bytes_len: key_bytes_len,
                                                        row_count,
                                                    },
                                                    fixed_component: if composite_is_text {
                                                        Some(gpu_db_execution::CudaGroupFixedSource::Derived {
                                                            buffer: _derived_key_buf.as_ref().expect("fixed text component").group_view(),
                                                            width: 8,
                                                            row_count,
                                                        })
                                                    } else { None },
                                                }
                                            } else {
                                                let width = if key_is_i128 { 16 } else if key_is_int8 { 8 } else { 4 };
                                                gpu_db_execution::CudaGroupKeySource::Fixed(
                                                    if let Some(buf) = &_derived_key_buf {
                                                        gpu_db_execution::CudaGroupFixedSource::Derived {
                                                            buffer: buf.group_view(), width, row_count,
                                                        }
                                                    } else {
                                                        gpu_db_execution::CudaGroupFixedSource::Resident {
                                                            byte_offset: key_offset, width, row_count,
                                                        }
                                                    },
                                                )
                                            },
                                            value: gpu_db_execution::CudaGroupValueSource::Unused { row_count },
                                            key_validity_bitmap_offset: None,
                                            value_validity_bitmap_offset: None,
                                        },
                                        &reps,
                                        // Step-2 GROUP BY g over reps, COUNT(*): the result builder reads
                                        // `.count` (-> `.sum`). ALL is correct + behavior-preserving.
                                        gpu_db_execution::grouped_agg_mask::COUNT,
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
                            count_distinct_groups(
                                value_idx,
                                g_vals,
                                &idx_u64,
                                table,
                                &snapshot,
                                &device_memory,
                                row_count,
                            )?
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
                            row_count,
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
            let materialize_col_at =
                |col: usize, reps: &[u64]| -> Result<Vec<SqlValue>, ExecuteError> {
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
                                    row_count,
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
                        let text_col = if matches!(ct0, SqlType::Text) {
                            cc0
                        } else {
                            cc1
                        };
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
                        pass.groups.iter().map(&full_key).collect();
                    let perm = gpu_sort_permutation(
                        &key_rows,
                        &key_order,
                        &key_nulls,
                        &key_types,
                        &device_memory,
                    )?;
                    pass.groups = perm
                        .iter()
                        .map(|&p| pass.groups[p as usize])
                        .collect();
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
                                    row_count,
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
                let mut row: Vec<SqlValue> = Vec::with_capacity(aggregates.len() + n_group_cols);
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
                                        SqlValue::Numeric(Decimal128::new(
                                            sum_i128,
                                            pass.value_scale,
                                        ))
                                    }
                                    GroupedAggKind::Sum if pass.value_is_int8 => {
                                        SqlValue::Numeric(Decimal128::new(sum_i128, 0))
                                    }
                                    GroupedAggKind::Sum => SqlValue::Int8(g.sum),
                                    GroupedAggKind::Avg if pass.value_is_numeric => {
                                        avg_numeric_sql_value(
                                            sum_i128,
                                            g.count as usize,
                                            pass.value_scale,
                                        )
                                    }
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
                                    GroupedAggKind::Min if pass.value_is_text => SqlValue::Text(
                                        text_value_minmax[&pass.value_idx].0[i].clone(),
                                    ),
                                    GroupedAggKind::Max if pass.value_is_text => SqlValue::Text(
                                        text_value_minmax[&pass.value_idx].1[i].clone(),
                                    ),
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
                // WHERE. The VM is SINGLE-WIDTH per program, so PROMOTE every integer-family column (and
                // value) to ONE comparable width: if the HAVING
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
                            rhs: Box::new(having_value_to_resident_literal(
                                &f.value,
                                numeric_mode,
                            )?),
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
                        None,
                    )?;
                    let kept: Vec<Vec<SqlValue>> = survivors
                        .iter()
                        .map(|&i| rows[i as usize].clone())
                        .collect();
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
            // this is byte-identical to applying the same permutation to the materialized rows.
            if rows.len() > 1 || select.offset.is_some() || select.limit.is_some() {
                let col_types: Vec<SqlType> = bound.selected_columns.iter().map(|c| c.ty).collect();
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
                let windowed: Vec<Vec<SqlValue>> = perm[start..end]
                    .iter()
                    .map(|&p| rows[p as usize].clone())
                    .collect();
                rows = windowed;
            }
            return Ok(RelationalSelectResult {
                columns: Arc::new(bound.selected_columns),
                rows: rows.into(),
                planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
                executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
                fallback_reason: None,
                access_path: Arc::new(access_path),
            });
        }

        if is_aggregate {
            return execute_scalar_aggregate(
                select,
                table,
                bound.selected_columns,
                access_path,
                &snapshot,
                &device_memory,
                row_count,
                &indices,
                &indices_u64,
            );
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
        let materialize_int_key_column =
            |ki: usize, indices: &[u64]| -> Result<Vec<i64>, ExecuteError> {
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
        // ASC / first DESC) — GPU-native, no host shard / sentinel. A nullable key of ANY type routes
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
            let end = select.limit.map_or(indices_u64.len(), |l| {
                start.saturating_add(l).min(indices_u64.len())
            });
            indices_u64[start..end].to_vec()
        } else {
            indices_u64
        };
        materialize_projected_rows(
            table,
            bound,
            access_path,
            &snapshot,
            &device_memory,
            row_count,
            indices_u64,
        )
    }
}
