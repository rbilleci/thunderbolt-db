use super::{
    exact_device_verdict_cardinality, Engine, EngineError, ExecuteError, Index, LaneIntent,
    LaneOpKind, RelationalTable, SqlValue,
};
use std::sync::atomic::Ordering as AtomicOrdering;

impl Engine {
    /// Probe optimistically against immutable published descriptors. If an append races that
    /// observation, repeat once behind the lane device-publication boundary instead of turning a
    /// transient torn descriptor/index pair into a spurious serialization abort.
    fn lane_visible_locate_stable(
        &self,
        table: &RelationalTable,
        filter_idx: usize,
        needles: &[i32],
        snapshots: &[u64],
    ) -> Option<crate::engine_retained_read::WaveVisibleLocate> {
        if let Some(result) = self.wave_batch_visible_locate(table, filter_idx, needles, snapshots)
        {
            return Some(result);
        }
        let apply_leader =
            crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(|flag| flag.get());
        if apply_leader {
            return self.wave_batch_visible_locate(table, filter_idx, needles, snapshots);
        }
        let lanes = self.intent_lanes.as_ref()?;
        let _device_guard = lanes
            .device_apply_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        struct DeviceGuardFlag;
        impl Drop for DeviceGuardFlag {
            fn drop(&mut self) {
                crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(|flag| flag.set(false));
            }
        }
        crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(|flag| flag.set(true));
        let _device_guard_flag = DeviceGuardFlag;
        self.wave_batch_visible_locate(table, filter_idx, needles, snapshots)
    }

    /// LEAN device validate for lane intents: needles straight from the
    /// integer slots, returning both visibility at each item's read snapshot and the highest
    /// physical-version write stamp for that key. A stamp newer than the snapshot is a retryable
    /// first-committer-wins conflict even when the key has since been deleted or moved away;
    /// otherwise an INSERT-visible match is the normal 23505 duplicate. Catalog drift (DDL between
    /// build and pump, pre-activation-window only) aborts the item retryably.
    pub(super) fn lane_validate_unique(
        &self,
        batch: &[LaneIntent],
    ) -> (
        std::collections::BTreeMap<usize, String>,
        std::collections::BTreeSet<usize>,
        Vec<Option<u64>>,
    ) {
        let mut violations = std::collections::BTreeMap::new();
        let mut conflicts = std::collections::BTreeSet::new();
        let mut target_counts = vec![None; batch.len()];
        if batch.is_empty() {
            return (violations, conflicts, target_counts);
        }
        let catalog = self.catalog_snapshot();
        // group needles per (table, filter_idx); usually exactly one group
        let mut group_keys: Vec<(&str, u32)> = Vec::new();
        let mut group_needles: Vec<Vec<i32>> = Vec::new();
        let mut group_snapshots: Vec<Vec<u64>> = Vec::new();
        let mut group_positions: Vec<Vec<usize>> = Vec::new();
        for (position, item) in batch.iter().enumerate() {
            if catalog.commit_seq != item.prepared_catalog_seq {
                violations.insert(
                    position,
                    "catalog drift between intent build and lane wave (retry)".to_string(),
                );
                continue;
            }
            let key = (&*item.table, item.filter_idx);
            let group = match group_keys.iter().position(|k| *k == key) {
                Some(index) => index,
                None => {
                    group_keys.push(key);
                    group_needles.push(Vec::new());
                    group_snapshots.push(Vec::new());
                    group_positions.push(Vec::new());
                    group_keys.len() - 1
                }
            };
            group_needles[group].push(item.slot.1);
            group_snapshots[group].push(item.read_snapshot);
            group_positions[group].push(position);
        }
        for (group, &(table_name, filter_idx)) in group_keys.iter().enumerate() {
            let Some(table) = catalog.relational_catalog.get(table_name) else {
                for &position in &group_positions[group] {
                    violations.insert(position, "table dropped".to_string());
                }
                continue;
            };
            let locate = self.lane_visible_locate_stable(
                table,
                filter_idx as usize,
                &group_needles[group],
                &group_snapshots[group],
            );
            let Some(locate) = locate else {
                // A host probe cannot prove that an after-snapshot claim was subsequently
                // released. No device history verdict therefore fails retryably before WAL.
                for &position in &group_positions[group] {
                    conflicts.insert(position);
                }
                continue;
            };
            let expected = group_positions[group].len();
            if !exact_device_verdict_cardinality(
                expected,
                &[
                    group_needles[group].len(),
                    group_snapshots[group].len(),
                    locate.counts.len(),
                    locate.shard_ids.len(),
                    locate.slots.len(),
                    locate.row_ids.len(),
                    locate.latest_write.len(),
                ],
            ) {
                // Parallel device-result vectors are one exact verdict. A short OR long component
                // must not let `zip` silently ignore a requested key.
                conflicts.extend(group_positions[group].iter().copied());
                continue;
            }
            for (((&position, &count), &latest_write), &snapshot) in group_positions[group]
                .iter()
                .zip(locate.counts.iter())
                .zip(locate.latest_write.iter())
                .zip(group_snapshots[group].iter())
            {
                if latest_write > snapshot {
                    conflicts.insert(position);
                    continue;
                }
                if count > 1 {
                    conflicts.insert(position);
                    continue;
                }
                target_counts[position] = Some(u64::from(count));
                if batch[position].op == LaneOpKind::Insert && count > 0 {
                    let index_name = table
                        .columns
                        .get(filter_idx as usize)
                        .map(|column| format!("{}_{}_key", table.name, column.name))
                        .unwrap_or_else(|| format!("{}_key", table.name));
                    violations.insert(
                        position,
                        format!("duplicate key value violates unique index \"{index_name}\""),
                    );
                }
            }
        }
        (violations, conflicts, target_counts)
    }

    /// APPLY LEADER body: merge every pending lane request per table and run
    /// ONE open-shard append pass (rows + per-row created_by stamps + row ids).
    /// The leader lock serializes appends, so the PK-index extension chain
    /// (entry.row_count == base) is preserved exactly as under the old
    /// exclusive section — just batched across lanes. The non-appended
    /// fallback mirrors flush_wave_pending_appends' rehydrate/invalidate arm
    /// using only request-carried data (no CommitWaveItem).
    pub(super) fn lane_apply_merged(&self, batch: &mut [crate::engine_intent_lanes::ApplyRequest]) {
        // P4-2b (audit L5): a CHUNK-AUTHORITATIVE table must be unreachable here — lane ingress
        // needs a covered/keyed route a keyless class table cannot build. Assert the invariant a
        // future keyless-lane path would otherwise silently break (lost writes).
        #[cfg(debug_assertions)]
        for request in batch.iter() {
            debug_assert!(
                self.table_chunk_authoritative(&request.table).is_none(),
                "a chunk-authoritative table reached the lane apply — the class write path only \
                 exists on the serialized commit"
            );
        }
        use std::collections::BTreeMap;
        // group request indexes per table (usually exactly one table)
        let mut tables: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (index, request) in batch.iter().enumerate() {
            tables.entry(request.table.clone()).or_default().push(index);
        }
        for (table, requests) in tables {
            let table = table.as_str();
            let total: usize = requests.iter().map(|&i| batch[i].rows.len()).sum();
            let mut rows: Vec<Vec<SqlValue>> = Vec::with_capacity(total);
            let mut row_ids: Vec<u64> = Vec::with_capacity(total);
            let mut stamps: Vec<Index> = Vec::with_capacity(total);
            let mut tombstones: Vec<crate::engine_intent_lanes::LaneTombstone> = Vec::new();
            let mut updates: Vec<crate::engine_intent_lanes::LaneUpdate> = Vec::new();
            for &i in &requests {
                // MOVE the row vectors (pointer moves) — the leader was cloning
                // every merged row's SqlValues, ~1900 heap allocs per wave.
                rows.append(&mut batch[i].rows);
                row_ids.extend_from_slice(&batch[i].row_ids);
                stamps.extend_from_slice(&batch[i].stamps);
                tombstones.append(&mut batch[i].tombstones);
                updates.append(&mut batch[i].updates);
            }
            // WAL-FIRST APPLY ORDER: APPEND inserts FIRST (device-visible + indexed), THEN
            // LOCATE + tombstone deletes. The locate runs at each delete's read_snapshot, and its
            // created_by<=snapshot<deleted_by filter selects EXACTLY the version the delete's
            // snapshot saw — so append-vs-locate order is immaterial: a same-batch reinsert
            // (created_by = its seq > the delete's snapshot) is filtered OUT, while a same-batch
            // insert the delete's snapshot DID see (created_by <= snapshot) is correctly targeted.
            // (Audit note: "a same-batch insert is never a target" is NOT the invariant — the
            // visibility filter is what makes every case semantically correct, not append order.)
            let appended = rows.is_empty()
                || self.try_append_resident_int4_open_shard(
                    table,
                    &rows,
                    crate::engine_residency::AppendCreatedBy::InsertPerRow(&stamps),
                    Some(&row_ids),
                );
            if appended
                && !rows.is_empty()
                && self.host_install_elision_enabled()
                && !self.table_install_elided(table)
            {
                {
                    let snapshot = self.catalog_snapshot();
                    if self.table_elision_eligible(&snapshot, table) {
                        self.set_table_install_elided(table, true);
                    }
                }
            }
            // LOCATE + tombstone deletes on the device (only if the append path is intact — a
            // failed append means the whole batch rehydrates anyway). Sets each delete's
            // rows-affected cell (0 or 1). A decline routes to the rehydrate fallback, which
            // resolves rows-affected BY KEY.
            let deletes_ok = tombstones.is_empty()
                || (appended && self.apply_lane_tombstones_device(table, &tombstones));
            // U2 WAL-FIRST: LOCATE the update-olds, tombstone them, and CONDITIONALLY append the
            // new versions (only the 1-row updates append). Sets each update's rows-affected cell
            // (0 or 1). Runs only if the insert append succeeded (a failed append rehydrates the
            // whole batch anyway); a decline routes to the rehydrate fallback (by-key resolution).
            let updates_ok =
                updates.is_empty() || (appended && self.apply_lane_updates_device(table, &updates));
            if appended && deletes_ok && updates_ok {
                continue;
            }
            // AUDIT F1 (U1, MEDIUM adopted): a delete/update decline on a NON-elided table has no
            // recovery arm below — falling through would advance the cut and ack for a mutation
            // that never applied (silent live/durable divergence until restart). Fail LOUDLY:
            // the panic rides the apply leader's catch_unwind (F2), failing the waiters and
            // poisoning the lanes; recovery replays the durable W5b records.
            if (!deletes_ok || !updates_ok) && !self.table_install_elided(table) {
                panic!(
                    "commit-path invariant violation: lane deletes/updates declined on the \
                     non-elided table \"{table}\" — refusing to ack an unapplied mutation"
                );
            }
            // Fallback (rare on the lanes path — intents gate on elided,
            // auto-admit tables): rehydrate the merged batch as upserts +
            // key-resolved removals and invalidate per txn, mirroring
            // flush_wave_pending_appends. U1: the seq window spans appends AND
            // tombstones; removals are resolved BY KEY against the pre-batch
            // gather (the tombstones' (shard, slot) targets are exactly what a
            // declined/stale device state can no longer be trusted for).
            if self.table_install_elided(table) {
                let first_seq = stamps
                    .first()
                    .copied()
                    .into_iter()
                    .chain(tombstones.first().map(|t| t.seq))
                    .chain(updates.first().map(|u| u.seq))
                    .min()
                    .unwrap_or_default();
                let last_seq = stamps
                    .last()
                    .copied()
                    .into_iter()
                    .chain(tombstones.last().map(|t| t.seq))
                    .chain(updates.last().map(|u| u.seq))
                    .max()
                    .unwrap_or_default();
                let gather_snapshot = first_seq.saturating_sub(1);
                let mut upserts: BTreeMap<u64, Vec<SqlValue>> =
                    row_ids.iter().copied().zip(rows.iter().cloned()).collect();
                let catalog_table = self
                    .relational_catalog_table(table)
                    .expect("an elided table is in the catalog");
                let (removals, matched_keys) = self
                    .resolve_elided_row_ids_by_int4_key(
                        &catalog_table,
                        gather_snapshot,
                        &tombstones
                            .iter()
                            .map(|t| (t.filter_idx as usize, t.pk))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap_or_else(|err| {
                        panic!(
                            "commit-path invariant violation: tombstone key resolution for \
                             the merged lane fallback on {table} failed: {err}"
                        )
                    });
                // WAL-FIRST: set each delete's rows-affected from the by-key resolution (the
                // device locate declined, so this host gather is authoritative). A matched key =
                // one visible row = the rehydrate removes it = rows-affected 1; else 0.
                for tombstone in &tombstones {
                    tombstone.rows_affected.store(
                        u64::from(matched_keys.contains_key(&tombstone.pk)),
                        std::sync::atomic::Ordering::Release,
                    );
                }
                // U2/R3 WAL-FIRST fallback: resolve the update-olds BY KEY too. A matched old
                // supplies the stable entity id reused by the replacement upsert; the old image
                // is overwritten by identity, not removed under one id and reinserted under a
                // fresh one. An unmatched update appends nothing. Rows-affected = matched.
                if !updates.is_empty() {
                    let (_update_removals, update_entities) = self
                        .resolve_elided_row_ids_by_int4_key(
                            &catalog_table,
                            gather_snapshot,
                            &updates
                                .iter()
                                .map(|u| (u.filter_idx as usize, u.pk))
                                .collect::<Vec<_>>(),
                        )
                        .unwrap_or_else(|err| {
                            panic!(
                                "commit-path invariant violation: update key resolution for \
                                 the merged lane fallback on {table} failed: {err}"
                            )
                        });
                    for update in &updates {
                        let entity_id = update_entities.get(&update.pk).copied();
                        if let Some(entity_id) = entity_id {
                            upserts.insert(entity_id, update.new_values.clone());
                        }
                        update.rows_affected.store(
                            u64::from(entity_id.is_some()),
                            std::sync::atomic::Ordering::Release,
                        );
                    }
                }
                self.rehydrate_elided_table(
                    &catalog_table,
                    gather_snapshot,
                    &upserts,
                    &removals,
                    last_seq,
                )
                .unwrap_or_else(|err| {
                    panic!(
                        "commit-path invariant violation: elided rehydration for the \
                         merged lane append on {table} failed: {err}"
                    )
                });
            }
            let residency: std::collections::BTreeSet<String> =
                std::iter::once(table.to_string()).collect();
            for &i in &requests {
                for (offset, txn_id) in batch[i].txn_ids.iter().enumerate() {
                    self.invalidate_relational_residency_tables_concurrent(
                        &residency,
                        *txn_id,
                        batch[i].stamps[offset],
                    );
                }
            }
        }
    }

    /// U1 WAL-FIRST: LOCATE + tombstone a merged batch's deletes on the DEVICE, at apply time
    /// (off the pump critical path). ONE visible-locate over all keys at their read snapshots
    /// resolves each to zero or one visible row; the located targets are stamped (scatter,
    /// grouped by shard with the cell-liveness recheck), and every delete's rows-affected cell
    /// is set only on FULL success. Returns `false` on ANY decline (declined locate, ambiguous
    /// multiplicity, stale cell, stamp failure) WITHOUT setting cells — the caller's rehydrate
    /// fallback then resolves rows-affected by key. Runs under the apply leader lock.
    fn apply_lane_tombstones_device(
        &self,
        table: &str,
        tombstones: &[crate::engine_intent_lanes::LaneTombstone],
    ) -> bool {
        use std::collections::BTreeMap;
        if tombstones.is_empty() {
            return true;
        }
        // Covered-delete shape: one unique pk column, so a single (table, filter) group.
        let filter_idx = tombstones[0].filter_idx;
        if tombstones.iter().any(|t| t.filter_idx != filter_idx) {
            return false; // mixed filters -> fallback (not reachable on the covered shape)
        }
        let catalog = self.catalog_snapshot();
        let Some(rel) = catalog.relational_catalog.get(table) else {
            return false;
        };
        let needles: Vec<i32> = tombstones.iter().map(|t| t.pk).collect();
        let snapshots: Vec<u64> = tombstones.iter().map(|t| t.read_snapshot).collect();
        let Some(locate) =
            self.wave_batch_visible_locate(rel, filter_idx as usize, &needles, &snapshots)
        else {
            return false; // device decline -> rehydrate fallback resolves by key
        };
        let expected = tombstones.len();
        if needles.len() != expected
            || snapshots.len() != expected
            || locate.counts.len() != expected
            || locate.shard_ids.len() != expected
            || locate.slots.len() != expected
            || locate.row_ids.len() != expected
            || locate.latest_write.len() != expected
        {
            return false;
        }
        // Resolve each delete to 0 or 1 rows; collect the 1-row targets grouped by (shard,
        // locate-region identity) for the batched scatter + cell-liveness recheck. The group
        // value = (the region Arc, the (slot, stamp) pairs for that shard).
        type TombstoneGroup = (
            std::sync::Arc<gpu_db_execution::CudaResidentDeviceMemory>,
            Vec<(u32, Index)>,
        );
        let mut counts: Vec<u64> = Vec::with_capacity(tombstones.len());
        let mut groups: BTreeMap<(u32, u64), TombstoneGroup> = BTreeMap::new();
        for (i, tombstone) in tombstones.iter().enumerate() {
            match locate.counts.get(i).copied() {
                Some(0) => counts.push(0),
                Some(1) => {
                    // AUDIT (finding 3): index the parallel output vectors defensively — a short
                    // slot/shard vector from the device declines to the fallback, never panics.
                    let (Some(&shard_id), Some(&slot)) =
                        (locate.shard_ids.get(i), locate.slots.get(i))
                    else {
                        return false;
                    };
                    let Some((_, region)) = locate.probed.iter().find(|(id, _)| *id == shard_id)
                    else {
                        return false;
                    };
                    groups
                        .entry((shard_id, region.device_ptr()))
                        .or_insert_with(|| (std::sync::Arc::clone(region), Vec::new()))
                        .1
                        .push((slot, tombstone.seq));
                    counts.push(1);
                }
                _ => return false, // ambiguous multiplicity / missing -> fallback
            }
        }
        // Stamp every 1-row target (cell-liveness recheck gates each shard group).
        let mut stamped_rows = 0u64;
        for ((shard_id, _ptr), (region, slots)) in &groups {
            if !self.shard_write_locate_cell_live(table, *shard_id, region) {
                return false;
            }
            if !self.tombstone_resident_shard_slots_stamped(table, *shard_id, slots) {
                return false;
            }
            stamped_rows += slots.len() as u64;
        }
        // FULL success — publish rows-affected (deletes ack from these cells) + counters.
        for (tombstone, count) in tombstones.iter().zip(counts.iter()) {
            tombstone
                .rows_affected
                .store(*count, std::sync::atomic::Ordering::Release);
        }
        if stamped_rows > 0 {
            self.add_tombstone_churn(table, stamped_rows);
            self.read_state
                .residency
                .lane_tombstone_applies
                .fetch_add(stamped_rows, std::sync::atomic::Ordering::Relaxed);
        }
        true
    }

    /// U2 WAL-FIRST: LOCATE the update-olds, tombstone them, and CONDITIONALLY append the new
    /// versions on the DEVICE, at apply time (off the pump critical path). ONE visible-locate over
    /// all keys at their read snapshots resolves each to zero or one visible row. A 1-row update
    /// tombstones the located old (scatter, grouped by shard with the cell-liveness recheck, EXACTLY
    /// the delete pass) AND appends its new image with the GPU-returned stable entity identity. The
    /// physical version twin still shares the pk, so the append's index CAS follows the existing
    /// duplicate/version-aware path. A 0-row update appends NOTHING; its legacy v1 WAL allocator
    /// reservation was already claimed and remains replayed for format/high-water compatibility. Every
    /// update's rows-affected cell is set only on FULL success. Returns `false` on ANY decline
    /// (declined locate, ambiguous multiplicity, stale cell, stamp/append failure) WITHOUT setting
    /// cells — the caller's rehydrate fallback then resolves by key. Runs under the apply leader lock.
    fn apply_lane_updates_device(
        &self,
        table: &str,
        updates: &[crate::engine_intent_lanes::LaneUpdate],
    ) -> bool {
        use std::collections::BTreeMap;
        if updates.is_empty() {
            return true;
        }
        // Covered-update shape: one unique pk column, so a single (table, filter) group.
        let filter_idx = updates[0].filter_idx;
        if updates.iter().any(|u| u.filter_idx != filter_idx) {
            return false; // mixed filters -> fallback (not reachable on the covered shape)
        }
        let catalog = self.catalog_snapshot();
        let Some(rel) = catalog.relational_catalog.get(table) else {
            return false;
        };
        let needles: Vec<i32> = updates.iter().map(|u| u.pk).collect();
        let snapshots: Vec<u64> = updates.iter().map(|u| u.read_snapshot).collect();
        let Some(locate) =
            self.wave_batch_visible_locate(rel, filter_idx as usize, &needles, &snapshots)
        else {
            return false; // device decline -> rehydrate fallback resolves by key
        };
        let expected = updates.len();
        if needles.len() != expected
            || snapshots.len() != expected
            || locate.counts.len() != expected
            || locate.shard_ids.len() != expected
            || locate.slots.len() != expected
            || locate.row_ids.len() != expected
            || locate.latest_write.len() != expected
        {
            return false;
        }
        // Resolve each update to 0 or 1 rows; collect the 1-row olds grouped by (shard, region) for
        // the batched tombstone scatter (identical to the delete pass) AND, in the SAME order, the
        // 1-row updates' new-version append inputs (new image, birth seq, stable entity id).
        type TombstoneGroup = (
            std::sync::Arc<gpu_db_execution::CudaResidentDeviceMemory>,
            Vec<(u32, Index)>,
        );
        let mut counts: Vec<u64> = Vec::with_capacity(updates.len());
        let mut groups: BTreeMap<(u32, u64), TombstoneGroup> = BTreeMap::new();
        let mut append_rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut append_stamps: Vec<Index> = Vec::new();
        let mut append_row_ids: Vec<u64> = Vec::new();
        for (i, update) in updates.iter().enumerate() {
            match locate.counts.get(i).copied() {
                Some(0) => counts.push(0),
                Some(1) => {
                    // AUDIT (finding 3, delete-parity): index the parallel output vectors
                    // defensively — a short slot/shard vector from the device declines, never panics.
                    let (Some(&shard_id), Some(&slot)) =
                        (locate.shard_ids.get(i), locate.slots.get(i))
                    else {
                        return false;
                    };
                    let Some((_, region)) = locate.probed.iter().find(|(id, _)| *id == shard_id)
                    else {
                        return false;
                    };
                    groups
                        .entry((shard_id, region.device_ptr()))
                        .or_insert_with(|| (std::sync::Arc::clone(region), Vec::new()))
                        .1
                        .push((slot, update.seq));
                    let Some(&entity_id) = locate.row_ids.get(i) else {
                        return false;
                    };
                    if entity_id == u64::MAX {
                        return false; // identity-unknown lineage: never invent a replacement identity
                    }
                    append_rows.push(update.new_values.clone());
                    append_stamps.push(update.seq);
                    append_row_ids.push(entity_id);
                    counts.push(1);
                }
                _ => return false, // ambiguous multiplicity / missing -> fallback
            }
        }
        // TOMBSTONE every 1-row old FIRST (fresh-from-locate regions, before any append can
        // roll the open shard), cell-liveness recheck gating each shard group — the delete pass.
        let mut stamped_rows = 0u64;
        for ((shard_id, _ptr), (region, slots)) in &groups {
            if !self.shard_write_locate_cell_live(table, *shard_id, region) {
                return false;
            }
            if !self.tombstone_resident_shard_slots_stamped(table, *shard_id, slots) {
                return false;
            }
            stamped_rows += slots.len() as u64;
        }
        // CONDITIONAL new-version append: only the 1-row updates append (0-row updates appended
        // nothing above). The new versions carry created_by = their own seq, so no reader below
        // the (not-yet-advanced) cut sees them; a decline here leaves the olds tombstoned but the
        // news unappended — the rehydrate fallback rebuilds the table correctly (gather sees the
        // old live at first_seq-1, delta removes it + upserts the new version).
        if !append_rows.is_empty()
            && !self.try_append_resident_int4_open_shard(
                table,
                &append_rows,
                crate::engine_residency::AppendCreatedBy::InsertPerRow(&append_stamps),
                Some(&append_row_ids),
            )
        {
            return false; // append decline (rollover / null / not-int4-resident) -> fallback
        }
        // FULL success — publish rows-affected (updates ack from these cells) + churn counters.
        for (update, count) in updates.iter().zip(counts.iter()) {
            update
                .rows_affected
                .store(*count, std::sync::atomic::Ordering::Release);
        }
        if stamped_rows > 0 {
            self.add_tombstone_churn(table, stamped_rows);
            self.read_state
                .residency
                .lane_tombstone_applies
                .fetch_add(stamped_rows, std::sync::atomic::Ordering::Relaxed);
        }
        true
    }

    /// Settle every lane wave whose end seq the visible cut covers (durable AND
    /// applied), publishing `committed_seq` to the global cut first so a polled
    /// Ok is never observable before the commit is readable.
    pub(super) fn settle_intent_lane(
        &self,
        lanes: &std::sync::Arc<crate::engine_intent_lanes::IntentLaneState>,
        lane: usize,
    ) -> bool {
        // AUDIT F3 (+ apply poison): an ASYNC durability failure (fence pool
        // poison after the append returned Ok) OR a failed merged apply
        // permanently stalls the cut; without this check the stalled waves
        // would hang their clients forever instead of wedging loudly like the
        // classic path. The probes are lock-free flags; the mutex-walking
        // reason fetch (N poison locks) is paid only on an actual wedge.
        let visible_seq = lanes.visible_inclusive_seq();
        if visible_seq.is_err() {
            lanes
                .apply_poisoned
                .store(true, std::sync::atomic::Ordering::Release);
        }
        let wal_poisoned = lanes.wal_peek().is_some_and(|wal| wal.is_poisoned());
        if wal_poisoned
            || lanes
                .apply_poisoned
                .load(std::sync::atomic::Ordering::Acquire)
        {
            let reason = if let Err(err) = &visible_seq {
                format!("intent lane visibility boundary invalid: {err}")
            } else if wal_poisoned {
                let inner = lanes
                    .wal_peek()
                    .and_then(|wal| wal.poison_reason())
                    .unwrap_or_else(|| "lane wedged (reason pending)".to_string());
                format!("intent lane WAL poisoned: {inner}")
            } else {
                "intent lane apply leader failed; cut permanently holed".to_string()
            };
            self.wedge_commit_path();
            let mut queue = lanes.settle[lane]
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut settled = false;
            while let Some(entry) = queue.pop_front() {
                for item in entry.winners.into_iter().chain(entry.async_winners) {
                    item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                        reason.clone(),
                    ))));
                }
                settled = true;
            }
            return settled;
        }
        let local_cut = lanes.visible_local_cut();
        if let Some(visible_seq) = visible_seq.expect("visibility error drained above") {
            self.publish_committed_seq(visible_seq);
        }
        let mut settled = false;
        let mut queue = lanes.settle[lane]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Compatibility slot only: ADR-014 production formation leaves `async_winners` empty, so
        // no SQL success can settle at the applied-only cut ahead of durable publication.
        let applied_cut = lanes
            .applied_mirror
            .load(std::sync::atomic::Ordering::Acquire);
        for entry in queue.iter_mut() {
            if entry.end_seq > applied_cut {
                break;
            }
            if !entry.async_settled {
                entry.async_settled = true;
                for item in entry.async_winners.drain(..) {
                    let rows = item.resolved_rows_affected();
                    item.set_outcome(Ok(rows));
                }
                settled = true;
            }
        }
        while queue
            .front()
            .is_some_and(|entry| entry.end_seq <= local_cut)
        {
            let entry = queue.pop_front().expect("front checked");
            debug_assert!(
                entry
                    .apply_slot
                    .done
                    .load(std::sync::atomic::Ordering::Acquire),
                "cut covered a wave whose apply slot is not done"
            );
            lanes.stat_acklag_ns.fetch_add(
                entry.published_at.elapsed().as_nanos() as u64,
                AtomicOrdering::Relaxed,
            );
            lanes
                .stat_settled_waves
                .fetch_add(1, AtomicOrdering::Relaxed);
            for item in entry.winners.into_iter().chain(entry.async_winners) {
                let rows = item.resolved_rows_affected();
                item.set_outcome(Ok(rows));
            }
            settled = true;
        }
        settled
    }
}
