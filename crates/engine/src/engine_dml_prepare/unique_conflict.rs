//! Device-current unique-slot conflict history. Exact typed predicates identify every physical
//! version that claimed or released a key; version stamps replace the persistent CPU slot ledger.

use super::*;

const DELETED_BY_LIVE: u64 = 0x7F7F_7F7F_7F7F_7F7F;

impl Engine {
    /// Ensure an autocommit DML touching a unique-constrained table has a device generation before
    /// it captures its read snapshot. This is the GPU-native bootstrap for an empty/newly indexed
    /// table: admission runs under the commit/publication lock after all earlier classic tails and
    /// their maintenance settle. A real admission failure remains a fail-closed error; no host
    /// unique probe becomes authoritative.
    pub(crate) fn ensure_unique_history_generation(
        &self,
        command: &Command,
    ) -> Result<(), ExecuteError> {
        let table_name = match command {
            Command::Insert(insert) => insert.table.as_str(),
            Command::Update(update) => update.table.as_str(),
            Command::Delete(delete) => delete.table.as_str(),
            _ => return Ok(()),
        };
        let needs_admission = || {
            self.catalog_snapshot()
                .relational_catalog
                .get(table_name)
                .is_some_and(|table| table.indexes.iter().any(|index| index.unique))
                && !self.table_is_gpu_resident(table_name)
                && self.table_chunk_authoritative(table_name).is_none()
        };
        if !needs_admission() {
            return Ok(());
        }

        let tables: BTreeSet<String> = std::iter::once(table_name.to_string()).collect();
        loop {
            if !self.wait_wave_tail_quiescence() {
                return Err(ExecuteError::Engine(self.commit_path_unavailable_error()));
            }
            let commit = self.commit_state();
            let settled = self.commit_wave.tails_applied.load(AtomicOrdering::Acquire)
                == self
                    .commit_wave
                    .tails_finished
                    .load(AtomicOrdering::Acquire)
                && self
                    .commit_wave
                    .tail_maintenance_pending
                    .load(AtomicOrdering::Acquire)
                    == 0;
            if !settled {
                drop(commit);
                continue;
            }
            self.ensure_commit_path_available()
                .map_err(ExecuteError::Engine)?;
            self.intent_lanes_write_guard()
                .map_err(ExecuteError::Engine)?;
            if needs_admission() {
                self.auto_admit_resident_tables(&tables);
            }
            drop(commit);
            break;
        }

        if needs_admission() {
            return Err(ExecuteError::Serialization(format!(
                "device unique-history generation for relation \"{table_name}\" is unavailable after admission"
            )));
        }
        Ok(())
    }

    /// An admitted empty table owns a real device generation (the row-count header and typed
    /// layouts are published atomically with its allocation), but it has no shards or version
    /// sidecars to launch against. That exact generation is authoritative proof of an empty
    /// conflict history; absence, invalidation, pressure, or missing allocation still declines.
    pub(crate) fn zero_row_resident_generation_boundary(
        &self,
        table: &RelationalTable,
    ) -> Option<Index> {
        let entries = self.read_residency_snapshots();
        let entry = entries.get(&table.name)?;
        let descriptor = &entry.descriptor;
        (descriptor.schema == table.schema
            && descriptor.table == table.name
            && descriptor.row_count == 0
            && descriptor.is_valid()
            && !self
                .router
                .runtime()
                .snapshot()
                .memory_pressured_gpu_ids
                .contains(&descriptor.gpu_id)
            && entry.device_memory.is_some())
        .then_some(descriptor.valid_through_index)
    }

    /// Return whether any unique key touched by `delta` was claimed or released after
    /// `read_snapshot`. The off-lock snapshot remains registered while this runs, so MVCC GC cannot
    /// discard a history version needed for the verdict. `None` fails closed at the sequencer.
    pub(crate) fn device_unique_write_conflicts(
        &self,
        delta: &WriteDelta,
        read_snapshot: Index,
    ) -> Option<bool> {
        let (table_name, rows): (&str, Vec<&[SqlValue]>) = match &delta.mutation {
            PreparedMutation::Insert {
                table,
                inserted_rows,
                ..
            } => (
                table,
                inserted_rows
                    .iter()
                    .map(|(_, row)| row.as_slice())
                    .collect(),
            ),
            PreparedMutation::Update {
                table,
                installs,
                updated_old_rows,
                ..
            } => {
                let mut rows = Vec::with_capacity(installs.len() + updated_old_rows.len());
                rows.extend(updated_old_rows.iter().map(Vec::as_slice));
                rows.extend(installs.iter().map(|(_, _, row)| row.as_slice()));
                (table, rows)
            }
            PreparedMutation::Delete {
                table,
                deleted_rows,
                ..
            } => (table, deleted_rows.iter().map(Vec::as_slice).collect()),
        };
        let catalog = self.catalog_snapshot();
        let table = catalog.relational_catalog.get(table_name)?;
        self.device_unique_rows_conflict(table, &rows, read_snapshot)
    }

    /// Core exact-key history check used by both autocommit deltas and resolved explicit
    /// transaction records. `rows` contains every old released and new claimed image.
    pub(crate) fn device_unique_rows_conflict(
        &self,
        table: &RelationalTable,
        rows: &[&[SqlValue]],
        read_snapshot: Index,
    ) -> Option<bool> {
        let cold = self.table_chunk_authoritative(&table.name).is_some();
        for index in table.indexes.iter().filter(|index| index.unique) {
            let positions = crate::engine_residency::index_key_column_positions(table, index)?;
            let mut distinct = BTreeSet::<Vec<SqlValue>>::new();
            for row in rows {
                let key = positions
                    .iter()
                    .map(|position| row.get(*position).cloned())
                    .collect::<Option<Vec<_>>>()?;
                if !distinct.insert(key.clone()) {
                    continue;
                }
                let mut probe_row = vec![SqlValue::Null; table.columns.len()];
                for (&position, value) in positions.iter().zip(&key) {
                    probe_row[position] = value.clone();
                }
                let conflict = if cold {
                    self.chunk_exact_key_write_conflict(
                        table,
                        &positions,
                        &probe_row,
                        read_snapshot,
                    )?
                } else {
                    self.resident_exact_key_write_conflict(
                        table,
                        &positions,
                        &probe_row,
                        read_snapshot,
                    )?
                };
                if conflict {
                    return Some(true);
                }
            }
        }
        Some(false)
    }

    fn resident_exact_key_write_conflict(
        &self,
        table: &RelationalTable,
        positions: &[usize],
        row: &[SqlValue],
        read_snapshot: Index,
    ) -> Option<bool> {
        if let Some(history_floor) = self.zero_row_resident_generation_boundary(table) {
            if history_floor <= read_snapshot {
                self.read_state
                    .residency
                    .dml_device_validate_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(false);
            }
            return None;
        }
        let shards = self.read_residency_shards();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let runtime = self.router.runtime().snapshot();
        for shard in table_shards {
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            if !shard.is_valid(runtime.memory_pressured_gpu_ids.contains(&shard.gpu_id)) {
                return None;
            }
            let device_memory = shard.device_memory.clone()?;
            if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                return None;
            }
            let row_count = u32::try_from(shard.row_count).ok()?;
            if row_count == 0 {
                continue;
            }
            let descriptor = self.resident_snapshot_for_shard(shard, table);
            let mask = self.exact_key_device_mask(
                table,
                &descriptor,
                &device_memory,
                row_count,
                positions,
                row,
            )?;
            let verdict = device_memory
                .predicate_mask_version_conflict(
                    &mask,
                    shard.created_by_region.as_deref().map(|memory| (memory, 0)),
                    0,
                    shard.deleted_by_region.as_deref().map(|memory| (memory, 0)),
                    DELETED_BY_LIVE,
                    DELETED_BY_LIVE,
                    read_snapshot,
                )
                .ok()?;
            if verdict.readback_bytes != std::mem::size_of::<u32>() {
                return None;
            }
            if verdict.conflict {
                self.read_state
                    .residency
                    .dml_device_validate_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(true);
            }
        }
        // Every shard miss must cover the writer's full post-snapshot interval. A dense rebuild
        // may have removed a claim+release version at its floor; an older writer cannot interpret
        // that absence as no conflict.
        if table_shards
            .iter()
            .any(|shard| shard.history_floor_index > read_snapshot)
        {
            return None;
        }
        self.read_state
            .residency
            .dml_device_validate_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(false)
    }

    /// Build exact structural tuple equality as one retained device mask. Each typed/NULL key
    /// component compiles independently, then masks are ANDed on-device. Per-leaf compilation is
    /// load-bearing for mixed-width compound keys (for example NUMERIC + TEXT): the expression VM
    /// is mono-typed, while mask composition is type-agnostic.
    pub(crate) fn exact_key_device_mask(
        &self,
        table: &RelationalTable,
        descriptor: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u32,
        positions: &[usize],
        row: &[SqlValue],
    ) -> Option<gpu_db_execution::CudaPredicateMaskI32> {
        let mut combined = None;
        for &position in positions {
            let leaf = Self::class_exact_key_predicate(table, &[position], row)?;
            let mask = self
                .resident_predicate_device_mask(
                    Some(&leaf),
                    table,
                    descriptor,
                    device_memory,
                    row_count,
                    None,
                )
                .ok()??;
            combined = Some(match combined {
                None => mask,
                Some(prior) => device_memory.and_predicate_masks(&prior, &mask).ok()?,
            });
        }
        combined
    }
}
