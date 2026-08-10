//! Explicit publication proof for every catalog-named resident device index.
//!
//! PRODUCT-002 uses this boundary after seed residency and before workload warm-up. Building is a
//! device operation over the authoritative shard payload. The returned report is only published
//! after every named index covers the same immutable table generation; partial or semantically
//! declined builds fail loudly.

use super::shard_point_lookup::ShardDeviceIndexBuild;
use super::*;

/// One named catalog index proven resident across every non-empty shard in a table generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalResidentIndexPublicationEntry {
    pub index: String,
    pub key_columns: Vec<String>,
    pub unique: bool,
    pub shard_count: usize,
    pub indexed_rows: usize,
    pub allocated_bytes: u64,
}

/// Exact device-index publication report for one resident table generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalResidentIndexPublication {
    pub table: String,
    pub gpu_id: u16,
    pub snapshot_boundary: Index,
    pub shard_count: usize,
    pub indexed_rows: usize,
    /// Dedicated allocation bytes, de-duplicated by device allocation when catalog indexes share a
    /// raw single-column device key.
    pub allocated_bytes: u64,
    pub indexes: Vec<RelationalResidentIndexPublicationEntry>,
}

fn publication_error(message: impl Into<String>) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(message.into()))
}

impl Engine {
    #[cfg(test)]
    pub(crate) fn set_named_index_publication_pre_linearize_hook(
        &self,
        reached: Arc<std::sync::Barrier>,
        resume: Arc<std::sync::Barrier>,
    ) {
        *self
            .read_state
            .residency
            .named_index_publication_pre_linearize_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some((reached, resume));
    }

    #[cfg(test)]
    fn run_named_index_publication_pre_linearize_hook(&self) {
        let hook = self
            .read_state
            .residency
            .named_index_publication_pre_linearize_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some((reached, resume)) = hook {
            reached.wait();
            resume.wait();
        }
    }

    #[cfg(test)]
    pub(crate) fn set_named_index_publication_post_publish_hook(
        &self,
        reached: Arc<std::sync::Barrier>,
        resume: Arc<std::sync::Barrier>,
    ) {
        *self
            .read_state
            .residency
            .named_index_publication_post_publish_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some((reached, resume));
    }

    #[cfg(test)]
    fn run_named_index_publication_post_publish_hook(&self) {
        let hook = self
            .read_state
            .residency
            .named_index_publication_post_publish_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some((reached, resume)) = hook {
            reached.wait();
            resume.wait();
        }
    }

    pub(crate) fn relational_named_index_publication_required(
        &self,
        table: &RelationalTable,
    ) -> bool {
        // The catalog supplied by the active apply scope is authoritative. A DROP-all transaction
        // can retain the pre-commit publication marker until lifecycle maintenance retires it, but
        // that stale marker must never make a concurrent DML rollover rebuild indexes absent from
        // the transaction's final catalog (or reserve their device bytes before WAL).
        !table.indexes.is_empty()
            && self
                .read_state
                .residency
                .named_index_publications
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains_key(&table.oid)
    }

    /// Feature-gated read-only evidence for named device-index publication. Callers use this only
    /// for a fixture with exactly one named index; it never probes or builds a host index.
    #[cfg(any(test, feature = "probe-timing"))]
    pub fn relational_named_index_covered_rows(&self, table_name: &str) -> Option<usize> {
        let coverage = self
            .read_state
            .residency
            .named_index_coverage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut rows = 0_usize;
        let mut found = false;
        for ((covered_table, _, _), (_, covered_rows)) in coverage.iter() {
            if covered_table == table_name {
                found = true;
                rows = rows.checked_add(*covered_rows)?;
            }
        }
        found.then_some(rows)
    }

    /// Build/reuse every named primary and secondary index directly from one authoritative resident
    /// generation. Duplicate non-unique keys remain distinct candidate slots; typed equality and
    /// MVCC visibility are authoritative at probe time. No host row/index shadow is constructed.
    pub fn publish_relational_resident_indexes(
        &self,
        table_name: &str,
    ) -> Result<RelationalResidentIndexPublication, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if self.current_transaction_read_snapshot().is_some() {
            return Err(publication_error(
                "resident index publication cannot run inside an existing transaction snapshot",
            ));
        }

        let commit = self.commit_state();
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let snapshot = self.capture_statement_snapshot(self.committed_seq());
        drop(commit);
        let snapshot_boundary = snapshot.boundary;
        let table = snapshot
            .catalog
            .relational_catalog
            .get(table_name)
            .cloned()
            .ok_or_else(|| {
                publication_error(format!("relation \"{table_name}\" does not exist"))
            })?;
        let _scope = self.enter_transaction_read(snapshot);
        if self.read_streaming_cold_chunks().contains_key(table_name) {
            return Err(publication_error(format!(
                "relation \"{table_name}\" has cold chunks; named index publication requires zero cold accesses"
            )));
        }

        let shards = self.read_state.residency.shards.load_full();
        let table_shards = shards.get(table_name).ok_or_else(|| {
            publication_error(format!(
                "relation \"{table_name}\" has no authoritative resident shards"
            ))
        })?;
        let apply_leader =
            crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(std::cell::Cell::get);
        let _apply = if apply_leader {
            None
        } else {
            Some(
                self.read_state
                    .residency
                    .mutation_gate
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            )
        };
        self.publish_relational_resident_indexes_for_generation(
            &table,
            table_shards,
            snapshot_boundary,
            true,
            false,
            true,
        )
    }

    /// Commit-path companion for a newly published rollover shard. The caller may already hold the
    /// residency budget transaction; passing that fact avoids recursively locking it while keeping
    /// all index allocation inside the same publication transaction.
    pub(crate) fn publish_relational_resident_indexes_for_generation(
        &self,
        table: &RelationalTable,
        table_shards: &[RelationalResidentShard],
        snapshot_boundary: Index,
        apply_already_locked: bool,
        budget_already_locked: bool,
        replace_coverage: bool,
    ) -> Result<RelationalResidentIndexPublication, ExecuteError> {
        let table_name = table.name.as_str();
        if table.indexes.is_empty() {
            return Err(publication_error(format!(
                "relation \"{table_name}\" declares no indexes to publish"
            )));
        }
        if table_shards.is_empty() {
            return Err(publication_error(
                "resident index publication requires a non-empty shard set",
            ));
        }
        let write001_empty_generation =
            crate::engine_residency::write001_empty_foldable_index_enrollment(table)
                && table_shards.len() == 1
                && table_shards[0].shard_id == 0
                && table_shards[0].row_start == 0
                && table_shards[0].row_count == 0
                && table_shards[0].capacity <= 1;
        let write001_empty_shard =
            crate::engine_residency::write001_empty_index_in_place_preallocation(table)
                && write001_empty_generation
                && table_shards[0].capacity == 1;
        if !replace_coverage
            && !self
                .read_state
                .residency
                .named_index_coverage_complete
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(table_name)
                .is_some_and(|(oid, indexes)| *oid == table.oid && indexes == &table.indexes)
        {
            return Err(publication_error(
                "incremental named-index publication requires a complete prior coverage proof",
            ));
        }
        let runtime = self.router.runtime().snapshot();
        let mut gpu_id = None;
        let mut non_empty_shards = 0usize;
        let mut indexed_rows = 0usize;
        for shard in table_shards {
            if shard.schema != table.schema
                || shard.table != table.name
                || !shard.is_valid(false)
                || runtime.memory_pressured_gpu_ids.contains(&shard.gpu_id)
            {
                return Err(publication_error(format!(
                    "relation \"{table_name}\" has a torn, invalid, or pressured resident generation"
                )));
            }
            if gpu_id
                .replace(shard.gpu_id)
                .is_some_and(|prior| prior != shard.gpu_id)
            {
                return Err(publication_error(
                    "resident index publication currently requires one GPU ownership domain",
                ));
            }
            if shard.row_count != 0 {
                non_empty_shards += 1;
                indexed_rows = indexed_rows.checked_add(shard.row_count).ok_or_else(|| {
                    publication_error("resident index publication row count overflowed")
                })?;
            }
        }
        let gpu_id = gpu_id.ok_or_else(|| {
            publication_error("resident index publication requires at least one resident shard")
        })?;
        if non_empty_shards == 0 && !write001_empty_generation {
            return Err(publication_error(
                "resident index publication requires at least one indexed row",
            ));
        }
        let gc_boundary = self
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .oldest()
            .unwrap_or(snapshot_boundary);

        let mut entries = Vec::with_capacity(table.indexes.len());
        let mut allocated_bytes = 0_u64;
        let mut charged_allocations = std::collections::BTreeSet::new();
        let mut expected_indexes = Vec::new();
        for (ordinal, index) in table.indexes.iter().enumerate() {
            if !crate::engine_residency::index_all_key_columns_foldable(table, index) {
                return Err(publication_error(format!(
                    "index \"{}\" has no exact resident device-key encoding",
                    index.name
                )));
            }
            let positions = crate::engine_residency::index_key_column_positions(table, index)
                .ok_or_else(|| {
                    publication_error(format!(
                        "index \"{}\" references a missing key column",
                        index.name
                    ))
                })?;
            let key_id = crate::engine_residency::index_probe_key_id(table, index, ordinal)
                .ok_or_else(|| {
                    publication_error(format!(
                        "index \"{}\" has no resident device key id",
                        index.name
                    ))
                })?;
            let mut entry_rows = 0usize;
            let mut entry_bytes = 0_u64;
            let mut entry_shards = 0usize;
            for shard in table_shards
                .iter()
                .filter(|shard| shard.row_count != 0 || write001_empty_shard)
            {
                #[cfg(test)]
                self.read_state
                    .residency
                    .named_index_publication_shard_visits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let offsets = positions
                    .iter()
                    .map(|&position| shard_fixed_width_key_offset(shard, table, position))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        publication_error(format!(
                            "index \"{}\" has an unavailable resident key section",
                            index.name
                        ))
                    })?;
                let blob_offsets = positions
                    .iter()
                    .map(|&position| shard_key_column_blob_offset(shard, table, position))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        publication_error(format!(
                            "index \"{}\" has an unavailable resident text layout",
                            index.name
                        ))
                    })?;
                let blob_lens = positions
                    .iter()
                    .map(|&position| shard_key_column_blob_len(shard, table, position))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        publication_error(format!(
                            "index \"{}\" has an invalid resident text extent",
                            index.name
                        ))
                    })?;
                let validity_offsets = positions
                    .iter()
                    .map(|&position| shard_key_column_validity_offset(shard, table, position))
                    .collect::<Option<Vec<Option<u64>>>>()
                    .ok_or_else(|| {
                        publication_error(format!(
                            "index \"{}\" has an unavailable resident validity section",
                            index.name
                        ))
                    })?;
                let resident = shard.device_memory.clone().ok_or_else(|| {
                    publication_error(format!(
                        "index \"{}\" shard {} has no device allocation",
                        index.name, shard.shard_id
                    ))
                })?;
                if !self.shard_write_locate_cell_live(table_name, shard.shard_id, &resident) {
                    return Err(publication_error(format!(
                        "index \"{}\" shard {} allocation is no longer authoritative",
                        index.name, shard.shard_id
                    )));
                }
                let (device_index, _, _, index_row_count, _has_postings) = self
                    .ensure_shard_pk_device_index(
                        table,
                        table_name,
                        shard.shard_id,
                        &resident,
                        ShardDeviceIndexBuild {
                            key: ShardDeviceIndexKey {
                                key_id,
                                positions: &positions,
                                offsets: &offsets,
                                blob_offsets: &blob_offsets,
                                blob_lens: &blob_lens,
                                validity_offsets: &validity_offsets,
                            },
                            row_count: shard.row_count,
                            capacity_rows: shard.capacity as u64,
                            gc_boundary,
                            deleted_by: shard.deleted_by_region.clone(),
                            duplicate_tolerant: !index.unique,
                            allow_empty: write001_empty_shard && shard.row_count == 0,
                            apply_already_locked,
                            budget_already_locked,
                            defer_cache_publication: false,
                        },
                    )
                    .map_err(|error| {
                        publication_error(format!(
                            "GPU build for index \"{}\" shard {} failed: {error}",
                            index.name, shard.shard_id
                        ))
                    })?
                    .ok_or_else(|| {
                        publication_error(format!(
                            "GPU build for index \"{}\" shard {} declined",
                            index.name, shard.shard_id
                        ))
                    })?;
                if index_row_count < shard.row_count {
                    return Err(publication_error(format!(
                        "index \"{}\" shard {} covers {index_row_count} of {} rows",
                        index.name, shard.shard_id, shard.row_count
                    )));
                }
                expected_indexes.push((
                    shard.shard_id,
                    key_id,
                    resident.device_ptr(),
                    shard.row_count,
                    Arc::clone(&device_index),
                ));
                let bytes = device_index.metadata().allocated_bytes;
                entry_bytes = entry_bytes.checked_add(bytes).ok_or_else(|| {
                    publication_error("resident index allocation accounting overflowed")
                })?;
                if charged_allocations.insert(device_index.device_ptr()) {
                    allocated_bytes = allocated_bytes.checked_add(bytes).ok_or_else(|| {
                        publication_error("resident index allocation accounting overflowed")
                    })?;
                }
                entry_rows = entry_rows.checked_add(shard.row_count).ok_or_else(|| {
                    publication_error("resident index publication row count overflowed")
                })?;
                entry_shards += 1;
            }
            entries.push(RelationalResidentIndexPublicationEntry {
                index: index.name.clone(),
                key_columns: index.key_columns.clone(),
                unique: index.unique,
                shard_count: entry_shards,
                indexed_rows: entry_rows,
                allocated_bytes: entry_bytes,
            });
        }

        #[cfg(test)]
        self.run_named_index_publication_pre_linearize_hook();
        // Linearize the report, coverage manifest, and mandatory-enrollment marker with route/index
        // retirement. A report cannot race a purge and claim an index that no longer exists.
        let _route_publish = self
            .read_state
            .residency
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let current_shards = self.read_state.residency.shards.load();
        let Some(current_table_shards) = current_shards.get(table_name) else {
            return Err(publication_error(
                "resident table generation changed during named-index publication",
            ));
        };
        let captures_are_current = table_shards.iter().all(|captured| {
            let Some(captured_memory) = captured.device_memory.as_ref() else {
                return false;
            };
            current_table_shards.iter().any(|current| {
                current.shard_id == captured.shard_id
                    && current.row_count == captured.row_count
                    && current
                        .device_memory
                        .as_ref()
                        .is_some_and(|memory| Arc::ptr_eq(memory, captured_memory))
            })
        });
        let full_set_is_current = !replace_coverage
            || current_table_shards
                .iter()
                .filter(|shard| shard.row_count != 0)
                .count()
                == table_shards
                    .iter()
                    .filter(|shard| shard.row_count != 0)
                    .count();
        if !captures_are_current || !full_set_is_current {
            return Err(publication_error(
                "resident table generation changed during named-index publication",
            ));
        }
        let cache = self
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !expected_indexes.iter().all(
            |(shard_id, key_id, resident_ptr, row_count, expected_index)| {
                cache
                    .get(&(table.name.clone(), *shard_id, *key_id))
                    .is_some_and(|entry| {
                        entry.resident_device_ptr == *resident_ptr
                            && entry.row_count >= *row_count
                            && entry
                                .device_index
                                .as_ref()
                                .is_some_and(|current| Arc::ptr_eq(current, expected_index))
                    })
            },
        ) {
            return Err(publication_error(
                "resident named-index cache changed before publication could linearize",
            ));
        }
        let mut coverage = self
            .read_state
            .residency
            .named_index_coverage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if replace_coverage {
            coverage.retain(|(covered_table, _, _), _| covered_table != table_name);
        }
        for (shard_id, key_id, resident_ptr, row_count, _) in &expected_indexes {
            coverage.insert(
                (table.name.clone(), *shard_id, *key_id),
                (*resident_ptr, *row_count),
            );
        }
        self.read_state
            .residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(table.name.clone(), (table.oid, table.indexes.clone()));
        self.read_state
            .residency
            .named_index_publications
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(table.oid, table.indexes.clone());
        drop(coverage);
        drop(cache);
        drop(_route_publish);

        #[cfg(test)]
        self.run_named_index_publication_post_publish_hook();

        Ok(RelationalResidentIndexPublication {
            table: table.name.clone(),
            gpu_id,
            snapshot_boundary,
            shard_count: non_empty_shards,
            indexed_rows,
            allocated_bytes,
            indexes: entries,
        })
    }
}
