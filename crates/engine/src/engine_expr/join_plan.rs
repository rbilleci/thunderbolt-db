//! Join entry adapters and plan binding/validation. Predicate masking, projection materialization,
//! and device-coordinate execution remain with their established owners.

use super::join_source::{JoinExecSide, JOIN_NULL_ROW};
use crate::engine_expr_ir::ResidentExpr;
use crate::engine_join_ir::{JoinColRef, JoinPlan, JoinProjItem};
use crate::rel_exec_helpers::relational_column_index;
use crate::relational_model::{RelationalSelectResult, RelationalTable};
use crate::{Engine, ExecuteError};
use gpu_db_sql::{SqlType, SqlValue};
use gpu_db_types::{EngineError, Index};

impl Engine {
    /// Execute a left-deep equi-join chain on resident payloads. Predicates, MVCC masks, N:N/OUTER
    /// membership, carried coordinates, ordering, and intermediate projection remain device-resident;
    /// only the final requested result columns are framed on the host. Streaming callers may retain an
    /// opaque D2D-materialized run instead, so source chunks can be evicted without a D2H intermediate.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_resident_expr_inner_join(
        &self,
        plan: &JoinPlan,
        tables: Vec<RelationalTable>,
        rows: Vec<Option<Vec<Vec<SqlValue>>>>,
        predicates: Vec<Option<ResidentExpr>>,
        copin_s: Index,
        side_override: Option<Vec<JoinExecSide>>,
        coordinate_out: Option<&mut Option<gpu_db_execution::CudaJoinCoordinatesU32>>,
        force_outer_where: bool,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_resident_expr_join_with_device_run(
            plan,
            tables,
            rows,
            predicates,
            copin_s,
            side_override,
            coordinate_out,
            None,
            force_outer_where,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_resident_expr_join_with_device_run(
        &self,
        plan: &JoinPlan,
        tables: Vec<RelationalTable>,
        rows: Vec<Option<Vec<Vec<SqlValue>>>>,
        predicates: Vec<Option<ResidentExpr>>,
        copin_s: Index,
        side_override: Option<Vec<JoinExecSide>>,
        coordinate_out: Option<&mut Option<gpu_db_execution::CudaJoinCoordinatesU32>>,
        materialized_out: Option<&mut Option<gpu_db_execution::CudaMaterializedRelation>>,
        force_outer_where: bool,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_resident_expr_join_with_device_overrides(
            plan,
            tables,
            rows,
            predicates,
            copin_s,
            side_override,
            None,
            coordinate_out,
            materialized_out,
            force_outer_where,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_resident_expr_join_with_device_ranges(
        &self,
        plan: &JoinPlan,
        tables: Vec<RelationalTable>,
        rows: Vec<Option<Vec<Vec<SqlValue>>>>,
        predicates: Vec<Option<ResidentExpr>>,
        copin_s: Index,
        side_override: Option<Vec<JoinExecSide>>,
        row_range_override: Vec<(u32, u32)>,
        coordinate_out: Option<&mut Option<gpu_db_execution::CudaJoinCoordinatesU32>>,
        materialized_out: Option<&mut Option<gpu_db_execution::CudaMaterializedRelation>>,
        force_outer_where: bool,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_resident_expr_join_with_device_overrides(
            plan,
            tables,
            rows,
            predicates,
            copin_s,
            side_override,
            Some(row_range_override),
            coordinate_out,
            materialized_out,
            force_outer_where,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_resident_expr_join_with_device_overrides(
        &self,
        plan: &JoinPlan,
        tables: Vec<RelationalTable>,
        rows: Vec<Option<Vec<Vec<SqlValue>>>>,
        predicates: Vec<Option<ResidentExpr>>,
        // SC5 rider (ADR-013 adjunct): the STATEMENT'S bound boundary — the same `s` the caller
        // bound the catalog at, so every relation's device state resolves at ONE snapshot
        // (previously each sharded side re-read `committed_seq()`, seeding cross-relation skew).
        copin_s: Index,
        side_override: Option<Vec<JoinExecSide>>,
        row_range_override: Option<Vec<(u32, u32)>>,
        coordinate_out: Option<&mut Option<gpu_db_execution::CudaJoinCoordinatesU32>>,
        materialized_out: Option<&mut Option<gpu_db_execution::CudaMaterializedRelation>>,
        force_outer_where: bool,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let n_rel = plan.relations.len();
        if row_range_override
            .as_ref()
            .is_some_and(|ranges| ranges.len() != n_rel)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "join row-range override count does not match the plan".to_string(),
            )));
        }
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
        let outer_where = (force_outer_where
            || plan.steps.iter().any(|s| s.outer_left || s.outer_right))
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
                                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                    format!("column reference \"{}\" is ambiguous", c.column),
                                )));
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
                        "NATURAL JOIN has no common column name between the two relations"
                            .to_string(),
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
            let non_int =
                |t: SqlType| matches!(t, SqlType::Text | SqlType::Numeric { .. } | SqlType::Uuid);
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
        let sides: Vec<JoinExecSide> = match side_override {
            Some(sides) => {
                if sides.len() != n_rel {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "join side override count does not match the plan".to_string(),
                    )));
                }
                sides
            }
            None => {
                let mut sides = Vec::with_capacity(n_rel);
                for (row_opt, (relation, table)) in
                    rows.into_iter().zip(plan.relations.iter().zip(&tables))
                {
                    sides.push(self.resolve_join_side(&relation.table, table, row_opt, copin_s)?);
                }
                sides
            }
        };
        // The `JOIN_NULL_ROW` sentinel (LEFT-pad) must be distinguishable from every real absolute row
        // index -- it is, because residency never holds anywhere near u32::MAX rows. Make it explicit.
        debug_assert!(
            sides
                .iter()
                .all(|s| s.0.descriptor.row_count < JOIN_NULL_ROW as usize),
            "a join relation has too many rows to distinguish the LEFT-join NULL-pad sentinel"
        );
        let gpu_id = sides[0].0.descriptor.gpu_id;
        let projection_aliases = self.join_projection_output_aliases(plan, &tables)?;
        let mut resolved_order = Vec::with_capacity(plan.order_by.len());
        for (order_idx, (column, descending)) in plan.order_by.iter().enumerate() {
            let alias_matches = if column.qualifier.is_none() {
                projection_aliases
                    .iter()
                    .enumerate()
                    .filter_map(|(index, alias)| {
                        (alias.as_deref() == Some(&column.column)).then_some(index)
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let resolved = if let [index] = alias_matches.as_slice() {
                proj[*index]
            } else if alias_matches.len() > 1 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "ORDER BY \"{}\" is ambiguous",
                    column.column
                ))));
            } else if column.qualifier.is_none() && is_coalesced(&column.column) {
                (0, relational_column_index(&tables[0], &column.column)?)
            } else {
                resolve(column, n_rel - 1)?
            };
            resolved_order.push((
                resolved.0,
                resolved.1,
                *descending,
                plan.order_by_nulls_first.get(order_idx).copied().flatten(),
            ));
        }
        self.execute_resident_device_coordinate_join(
            plan,
            &tables,
            &sides,
            &step_keys,
            &step_is_text,
            &step_is_b128,
            &proj,
            &predicates,
            outer_where,
            row_range_override.as_deref(),
            &resolved_order,
            gpu_id,
            coordinate_out,
            materialized_out,
        )
    }
}
