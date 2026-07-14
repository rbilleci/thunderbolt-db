use super::{
    resident_device_int4_column_offset, DeferredProbe, Engine, EngineError, ExecuteError, Instant,
    RelationalTable, RuntimeMetricsSnapshot,
};

impl Engine {
    /// Shared GPU-submit core for an already-validated, same-shape int4 equality-projection batch over
    /// the resident snapshot: derive the filter + projection column offsets ONCE and launch the single
    /// `equal_any` submission for all `needles`. `Ok(None)` signals the resident snapshot is present but
    /// schema-mismatched / invalidated (the caller chooses fallback vs error); a missing snapshot or
    /// device memory stays a hard error (preserving the jobs path's prior semantics).
    pub(super) fn submit_resident_int4_equal_any_payload(
        &self,
        table: &RelationalTable,
        selected_indexes: &[usize],
        filter_idx: usize,
        needles: &[i32],
    ) -> Result<Option<(u16, RuntimeMetricsSnapshot, Instant, DeferredProbe)>, ExecuteError> {
        let snapshot = self
            .relational_residency_snapshot_ref(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident snapshot",
                    table.name
                )))
            })?;
        if snapshot.schema != table.schema || snapshot.table != table.name || !snapshot.is_valid() {
            return Ok(None);
        }
        let snapshot_gpu_id = snapshot.gpu_id;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let filter_offset = resident_device_int4_column_offset(&snapshot, table, filter_idx)?;
        let projection_offsets = selected_indexes
            .iter()
            .map(|idx| resident_device_int4_column_offset(&snapshot, table, *idx))
            .collect::<Result<Vec<_>, ExecuteError>>()?;
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
        device_memory.clear_last_kernel_event_elapsed_us();
        let before_metrics = self.metrics.snapshot();
        let batch_started = Instant::now();
        // ADR-009 R1: route this resident int4 unique-key batch among two byte-identical producers (each
        // returns the SAME matched rows; only HOW they're found differs):
        //   (1) lpb GPU hash-index probe — `index_probe_enabled` ON, column unique/buildable: O(1)/needle.
        //   (2) full-scan kernel        — flag off, or column non-unique / un-buildable: O(rows).
        // Both are DEFERRED submissions drained at completion.
        // CONTRACT: the index route requires `needles` to be DISTINCT — the facade batcher's `dedup_needles`
        // guarantees this. The thread-per-needle gather emits one match per found needle vs the scan's one
        // per matched row; for a unique key + distinct needles these coincide (a bijection). Duplicate
        // needles would over-count relative to the scan, so the distinct-needle invariant is debug-asserted
        // INSIDE the index arm only — the scan arm is reachable with duplicate needles from the jobs-batch
        // caller (`submit_relational_retained_int4_projection_batch`) and must NOT be guarded.
        // launch-per-batch: GPU index probe when buildable, else the full scan -> a DeferredProbe.
        let payload = {
            match self
                .index_probe_enabled()
                .then(|| {
                    self.wave_resident_int4_index(
                        &table.name,
                        &device_memory,
                        filter_offset,
                        filter_idx,
                        row_count,
                    )
                })
                .flatten()
            {
                Some((index, table_mask, hash_shift)) => {
                    debug_assert!(
                        {
                            let mut seen = std::collections::HashSet::with_capacity(needles.len());
                            needles.iter().all(|needle| seen.insert(*needle))
                        },
                        "index route requires distinct needles (batcher dedup_needles contract)"
                    );
                    // The index route is unique (<=1 match/needle), so it can take the DENSE-emit kernel
                    // (DECISIONS "lpb read levers" #1) when the flag is on — byte-identical, no atomic/scatter.
                    if self.dense_index_probe_enabled() {
                        let dense = device_memory
                            .submit_match_project_i32_index_probe_dense_from_payload(
                                &index,
                                table_mask,
                                hash_shift,
                                needles,
                                &projection_offsets,
                                row_count,
                            )
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                            })?;
                        self.read_state
                            .residency
                            .dense_index_probe_hits
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        DeferredProbe::Dense(dense)
                    } else {
                        DeferredProbe::Atomic(
                            device_memory
                                .submit_match_project_i32_index_probe_from_payload(
                                    &index,
                                    table_mask,
                                    hash_shift,
                                    needles,
                                    &projection_offsets,
                                    row_count,
                                )
                                .map_err(|err| {
                                    ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                                })?,
                        )
                    }
                }
                // The non-unique SCAN keeps the atomic kernel (>1 match/needle needs the atomic compaction +
                // row_indices for the within-needle sort).
                None => DeferredProbe::Atomic(
                    device_memory
                        .submit_match_project_i32_equal_any_from_payload(
                            filter_offset,
                            needles,
                            &projection_offsets,
                            row_count,
                        )
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?,
                ),
            }
        };
        Ok(Some((
            snapshot_gpu_id,
            before_metrics,
            batch_started,
            payload,
        )))
    }
}
