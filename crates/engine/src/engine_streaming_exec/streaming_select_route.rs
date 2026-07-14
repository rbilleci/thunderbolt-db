//! Streamable SELECT classification, admission, binding, and fold dispatch.

use super::*;

/// The streaming operator class serving a SELECT: a scalar reduction fold (S-E.1), a filter/project
/// concat fold (S-E.2), or a grouped / distinct two-level fold (S-E.3).
enum StreamShape {
    Reduction(StreamAgg),
    Projection,
    /// `GROUP BY g` with Count/Sum/Min/Max aggregates — the normalized [`SelectProjection::
    /// GroupedAggregates`] form. `distinct_key_only` = the shape is a synthesized `SELECT DISTINCT col`
    /// (a GROUP BY col + COUNT(*) whose count column is dropped from the result, exactly the distinct
    /// bridge's synthesis).
    Grouped {
        normalized: SelectProjection,
        distinct_key_only: bool,
    },
    /// S-E.4: an `ORDER BY` projection whose sort keys are fully projected. `top_n` = the
    /// `[OFFSET,OFFSET+LIMIT)` window bound
    /// (`Some` = a top-N stream: each chunk's device-sorted local top-(skip+take) is its only possible
    /// contribution to the global window; `None` = unbounded — the whole survivor set must fit the
    /// budget for the final device sort, else defer).
    Ordered {
        top_n: Option<usize>,
    },
}

/// Classify a SELECT as a streamable shape, or `None` for the non-foldable classes (HAVING; grouped AVG /
/// COUNT(DISTINCT) — not associatively decomposable from per-chunk partials). A
/// SCALAR reduction (COUNT(*)/SUM/MIN/MAX, no LIMIT/OFFSET — PG applies LIMIT to the one-row aggregate
/// result, a shape not worth streaming) folds by combine; a plain `All`/`Columns` projection folds by
/// CONCAT, with LIMIT/OFFSET as cross-chunk windowing (LIMIT without ORDER BY is any-N-rows per SQL, so
/// early-exit + scan-order windowing is a valid instance); GROUP BY / DISTINCT fold TWO-LEVEL (per-chunk
/// device partials -> concat -> one final device merge pass).
fn streaming_shape(select: &Select) -> Option<StreamShape> {
    if !select.having_groups.is_empty() {
        return None;
    }
    // S-E.4 ORDER BY: ordered `All`/`Columns` projections stream when every sort key rides the
    // partials, so the FINAL device multi-key sort can re-order them. Ordered DISTINCT/grouped decline.
    if !select.order_by.is_empty() {
        if select.distinct || select.group_by.is_some() {
            return None;
        }
        if select.order_by.iter().any(|order| order.column.is_empty()) {
            return None;
        }
        let keys_projected = match &select.projection {
            SelectProjection::All => true,
            SelectProjection::Columns(columns) => select
                .order_by
                .iter()
                .all(|order| columns.contains(&order.column)),
            _ => return None,
        };
        if !keys_projected {
            return None;
        }
        // The window bound: OFFSET-without-LIMIT has no top-N bound -> treat as unbounded.
        let top_n = select
            .limit
            .map(|limit| select.offset.unwrap_or(0).saturating_add(limit));
        return Some(StreamShape::Ordered { top_n });
    }
    // S-E.3 DISTINCT: single-column `SELECT DISTINCT col` == `SELECT col, COUNT(*) GROUP BY col` with
    // the count dropped (the distinct bridge's own synthesis) — so it rides the grouped fold.
    if select.distinct {
        if select.group_by.is_some() || select.limit.is_some() || select.offset.is_some() {
            return None;
        }
        let SelectProjection::Columns(columns) = &select.projection else {
            return None;
        };
        let [column] = columns.as_slice() else {
            return None;
        };
        return Some(StreamShape::Grouped {
            normalized: SelectProjection::GroupedAggregates {
                group_column: column.clone(),
                aggregates: vec![GroupedAggregate {
                    kind: GroupedAggKind::Count,
                    value_column: None,
                }],
            },
            distinct_key_only: true,
        });
    }
    // S-E.3 GROUP BY: normalize the legacy 1-aggregate forms to GroupedAggregates (as the grouped
    // bridge does) and accept only associatively-decomposable kinds: COUNT merges as SUM(count),
    // SUM as SUM(sum), MIN as MIN(min), MAX as MAX(max). AVG needs the (sum,count) pair and
    // COUNT(DISTINCT) is not decomposable from per-chunk distinct counts — both decline to CPU.
    if let Some(group_column) = &select.group_by {
        if select.limit.is_some() || select.offset.is_some() {
            return None;
        }
        let normalized = match grouped_projection_to_aggregates(&select.projection) {
            Some(normalized) => normalized,
            None => match &select.projection {
                SelectProjection::GroupedAggregates { .. } => select.projection.clone(),
                _ => return None,
            },
        };
        let SelectProjection::GroupedAggregates {
            group_column: normalized_key,
            aggregates,
        } = &normalized
        else {
            return None;
        };
        if normalized_key != group_column {
            return None;
        }
        if !aggregates.iter().all(|aggregate| {
            matches!(
                aggregate.kind,
                GroupedAggKind::Count
                    | GroupedAggKind::Sum
                    | GroupedAggKind::Min
                    | GroupedAggKind::Max
            )
        }) {
            return None;
        }
        return Some(StreamShape::Grouped {
            normalized,
            distinct_key_only: false,
        });
    }
    match &select.projection {
        SelectProjection::All | SelectProjection::Columns(_) => Some(StreamShape::Projection),
        _ if select.limit.is_some() || select.offset.is_some() => None,
        SelectProjection::CountAll => Some(StreamShape::Reduction(StreamAgg::Count)),
        SelectProjection::Sum { .. } => Some(StreamShape::Reduction(StreamAgg::Sum)),
        SelectProjection::Min { .. } => Some(StreamShape::Reduction(StreamAgg::Min)),
        SelectProjection::Max { .. } => Some(StreamShape::Reduction(StreamAgg::Max)),
        _ => None,
    }
}

impl Engine {
    /// STRATA S-E.1/S-E.2: try to serve a SELECT OUT-OF-CORE via a streaming fold — a scalar reduction
    /// (combine partials) or a filter/project (concat + windowing). Returns `Some(result)` when the
    /// streaming path handled the read (`Ok`) or must surface a genuine SQL error (`Err`); `None` to fall
    /// through to the caller's path (the CPU pinned read). It NEVER returns a wrong answer: any shape the
    /// device cannot express defers to the authoritative CPU path.
    pub(crate) fn try_streaming_select(
        &self,
        select: &Select,
    ) -> Option<Result<RelationalSelectResult, ExecuteError>> {
        self.try_streaming_select_at(select, self.committed_seq())
    }

    pub(crate) fn try_streaming_select_at(
        &self,
        select: &Select,
        statement_copin_s: Index,
    ) -> Option<Result<RelationalSelectResult, ExecuteError>> {
        let shape = streaming_shape(select)?;
        let gpu_id = self.planner.default_gpu_id();
        // Activation gate: a per-GPU residency budget must be configured (the operator's VRAM-management
        // signal). With no budget there is no notion of "over-VRAM" -> stay on the interim host path
        // (byte-identical default behavior).
        let budget = self.relational_residency_budget_bytes(gpu_id)?;
        if budget == 0 {
            return None;
        }
        // Never stream an ELIDED table: its host MVCC store is intentionally stale (device-authoritative
        // writes), so a seq-scan would read the wrong data. Its device residency is served upstream.
        if self.table_install_elided(&select.table) {
            return None;
        }
        // Bind + lower the WHERE to a device predicate exactly as the sharded bridge does. A bind failure
        // or an un-lowerable predicate falls through to the host path (never a wrong answer).
        let (table, mut bound, copin_s) = self
            .bind_relational_select_at(select, statement_copin_s)
            .ok()?;
        // P4-2b: a CLASS table's fold must NEVER scan the (frozen) store — a cold MISS (budget
        // re-chunk, below-boundary reader, eviction) routes to the CPU-pinned path, whose guard
        // de-authoritizes first. The probe load here is the folds' own load (a hit is reused).
        if self.table_chunk_authoritative(&select.table).is_some() {
            let chunk_target = (budget / 2).max(1);
            if self
                .load_streaming_cold(&select.table, &table, chunk_target, copin_s)
                .is_none()
            {
                return Some(self.execute_relational_select_cpu_pinned(select));
            }
        }
        let predicate = resident_predicate_from_bound_filters(&bound).ok()?;
        // The executor filters SOLELY via the predicate (the SQL->Expr contract) — clear the bound filters.
        bound.filter = None;
        bound.filters.clear();
        bound.filter_groups.clear();

        Some(match shape {
            StreamShape::Reduction(agg) => self.run_streaming_reduction_fold(
                select,
                &table,
                &bound,
                predicate.as_ref(),
                copin_s,
                agg,
                gpu_id,
                budget,
            ),
            StreamShape::Projection => self.run_streaming_projection_fold(
                select,
                &table,
                &bound,
                predicate.as_ref(),
                copin_s,
                gpu_id,
                budget,
            ),
            StreamShape::Grouped {
                normalized,
                distinct_key_only,
            } => self.run_streaming_grouped_fold(
                select,
                &table,
                normalized,
                distinct_key_only,
                &bound,
                predicate.as_ref(),
                copin_s,
                gpu_id,
                budget,
            ),
            StreamShape::Ordered { top_n } => self.run_streaming_ordered_fold(
                select,
                &table,
                &bound,
                predicate.as_ref(),
                copin_s,
                top_n,
                gpu_id,
                budget,
            ),
        })
    }
}
