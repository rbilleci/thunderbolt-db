use super::{
    Engine, EngineError, ExecuteError, Index, LaneIntent, LaneOpKind, RelationalTable, SqlValue,
};
use std::sync::atomic::Ordering as AtomicOrdering;

impl Engine {
    /// LEAN device validate for lane intents: needles straight from the
    /// integer slots, locate through the cross-lane coalescer, count>0 hits
    /// re-checked authoritatively at the item's read snapshot (same semantics
    /// as wave_batch_validate_unique's covered-insert arm). Returns 23505
    /// messages by batch position. Catalog drift (DDL between build and pump,
    /// pre-activation-window only) aborts the item retryably.
    pub(super) fn lane_validate_unique(
        &self,
        batch: &[LaneIntent],
    ) -> std::collections::BTreeMap<usize, String> {
        let mut violations = std::collections::BTreeMap::new();
        if batch.is_empty() {
            return violations;
        }
        let catalog = self.catalog_snapshot();
        // group needles per (table, filter_idx); usually exactly one group
        let mut group_keys: Vec<(&str, u32)> = Vec::new();
        let mut group_needles: Vec<Vec<i32>> = Vec::new();
        let mut group_positions: Vec<Vec<usize>> = Vec::new();
        for (position, item) in batch.iter().enumerate() {
            if catalog.commit_seq != item.prepared_catalog_seq {
                violations.insert(
                    position,
                    "catalog drift between intent build and lane wave (retry)".to_string(),
                );
                continue;
            }
            // U1/U2: DELETE and UPDATE items resolve via the apply-time visible-locate; the
            // insert-dup validate has nothing to check for them (their pk is EXPECTED to exist —
            // a dup verdict would wrongly reject the very row they mutate).
            if item.op != LaneOpKind::Insert {
                continue;
            }
            let key = (&*item.table, item.filter_idx);
            let group = match group_keys.iter().position(|k| *k == key) {
                Some(index) => index,
                None => {
                    group_keys.push(key);
                    group_needles.push(Vec::new());
                    group_positions.push(Vec::new());
                    group_keys.len() - 1
                }
            };
            group_needles[group].push(item.slot.1);
            group_positions[group].push(position);
        }
        for (group, &(table_name, filter_idx)) in group_keys.iter().enumerate() {
            let Some(table) = catalog.relational_catalog.get(table_name) else {
                for &position in &group_positions[group] {
                    violations.insert(position, "table dropped".to_string());
                }
                continue;
            };
            let locate = self.wave_batch_locate_hit_counts(
                table,
                filter_idx as usize,
                &group_needles[group],
            );
            let Some(counts) = locate else {
                // decline -> authoritative per-needle recheck (rare)
                for (&position, &needle) in group_positions[group]
                    .iter()
                    .zip(group_needles[group].iter())
                {
                    self.lane_authoritative_dup_check(
                        table,
                        filter_idx as usize,
                        needle,
                        batch[position].read_snapshot,
                        position,
                        &mut violations,
                    );
                }
                continue;
            };
            for ((&position, &needle), &count) in group_positions[group]
                .iter()
                .zip(group_needles[group].iter())
                .zip(counts.iter())
            {
                if count == 0 {
                    continue;
                }
                self.lane_authoritative_dup_check(
                    table,
                    filter_idx as usize,
                    needle,
                    batch[position].read_snapshot,
                    position,
                    &mut violations,
                );
            }
        }
        violations
    }

    fn lane_authoritative_dup_check(
        &self,
        table: &RelationalTable,
        filter_idx: usize,
        needle: i32,
        read_snapshot: Index,
        position: usize,
        violations: &mut std::collections::BTreeMap<usize, String>,
    ) {
        let visibility = crate::StorageVisibility {
            read_txn_id: read_snapshot,
        };
        let value = SqlValue::Int4(needle);
        match self.visible_row_with_value(table, visibility, filter_idx, &value, None) {
            Ok(true) => {
                let index_name = table
                    .columns
                    .get(filter_idx)
                    .map(|column| format!("{}_{}_key", table.name, column.name))
                    .unwrap_or_else(|| format!("{}_key", table.name));
                violations.insert(
                    position,
                    format!("duplicate key value violates unique index \"{index_name}\""),
                );
            }
            Ok(false) => {}
            Err(err) => {
                violations.insert(position, format!("unique validation failed: {err}"));
            }
        }
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
                || (self.auto_admit_on_commit_enabled()
                    && self.try_append_resident_int4_open_shard(
                        table,
                        &rows,
                        crate::engine_residency::AppendCreatedBy::InsertPerRow(&stamps),
                        Some(&row_ids),
                    ));
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
                let (mut removals, matched_keys) = self
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
                        u64::from(matched_keys.contains(&tombstone.pk)),
                        std::sync::atomic::Ordering::Release,
                    );
                }
                // U2 WAL-FIRST fallback: resolve the update-olds BY KEY too. A matched old =
                // remove it (its resolved row id joins `removals`) + upsert the new version at its
                // claimed `new_row_id` (the CONDITIONAL append, done here by hand); an unmatched
                // (0-row) update removes nothing and appends nothing. Rows-affected = matched.
                if !updates.is_empty() {
                    let (update_removals, update_matched) = self
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
                    removals.extend(update_removals);
                    for update in &updates {
                        let matched = update_matched.contains(&update.pk);
                        if matched {
                            upserts.insert(update.new_row_id, update.new_values.clone());
                        }
                        update
                            .rows_affected
                            .store(u64::from(matched), std::sync::atomic::Ordering::Release);
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
    /// the delete pass) AND appends its new image at the claimed `new_row_id` — a dead twin sharing
    /// the pk, so the append's index CAS collides with the still-indexed old and DROPS the pk-index
    /// cache (the next locate rebuilds it visibility-aware, skipping dead-below-GC twins; this is the
    /// F3/U4 dead-twin cost). A 0-row update appends NOTHING (the CONDITIONAL append) but its
    /// `new_row_id` was already claimed + WAL-durable, so replay stays in allocator lock-step. Every
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
        // Resolve each update to 0 or 1 rows; collect the 1-row olds grouped by (shard, region) for
        // the batched tombstone scatter (identical to the delete pass) AND, in the SAME order, the
        // 1-row updates' new-version append inputs (new image, birth seq = its own seq, new row id).
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
                    append_rows.push(update.new_values.clone());
                    append_stamps.push(update.seq);
                    append_row_ids.push(update.new_row_id);
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
        let wal_poisoned = lanes.wal_peek().is_some_and(|wal| wal.is_poisoned());
        if wal_poisoned
            || lanes
                .apply_poisoned
                .load(std::sync::atomic::Ordering::Acquire)
        {
            let reason = if wal_poisoned {
                let inner = lanes
                    .wal_peek()
                    .and_then(|wal| wal.poison_reason())
                    .unwrap_or_else(|| "lane wedged (reason pending)".to_string());
                format!("intent lane WAL poisoned: {inner}")
            } else {
                "intent lane apply leader failed; cut permanently holed".to_string()
            };
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
        let global_cut = lanes.visible_global_cut();
        if global_cut > 0 {
            self.publish_committed_seq(global_cut);
        }
        let mut settled = false;
        let mut queue = lanes.settle[lane]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // ASYNC COMMIT tier (pg `synchronous_commit = off`): winners that
        // opted out of the durability wait ack as soon as the APPLIED cut
        // covers their wave — the WAL fence keeps running behind them.
        // Visibility (`publish_committed_seq` above) stays gated on the
        // STRICT cut, so readers never observe a row a power failure could
        // revoke; the async writer's own read-back lags by <= ~one fence
        // (documented deviation from pg, which exposes async commits
        // immediately).
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
