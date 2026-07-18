//! Current resident DML transition orchestration: physical old-version locate, generation gating,
//! tombstoning, and created-by-stamped UPDATE append. This is the exact live path; `R3-001` in
//! `docs/PLAN.md` remains the sole owner of future version-storage, index, and concurrency design.

use super::shard_pruning::{mandatory_int4_equalities, shard_zone_map_excludes};
use crate::engine_expr_ir::ResidentExpr;
use crate::relational_model::RelationalTable;
#[cfg(test)]
use crate::DdlCatalogState;
use crate::Engine;
use gpu_db_sql::SqlValue;
use gpu_db_types::Index;

enum DetailedLocateAttempt {
    Complete(Vec<crate::engine_retained_read::ShardPkHit>),
    Declined,
    Superseded,
}

impl Engine {
    /// SV4 (GPU-native DELETE -- LOCATE phase): find the resident `(shard_id, LOCAL slot)` positions of
    /// every row matching `predicate` (an int4-equality point-lookup shape), via zone-map-pruned per-shard
    /// `lower_resident_predicate`. PER-SHARD (NOT the recompacted unified buffer of the read path), so the
    /// returned slots are LOCAL to each shard's own device buffer -- exactly what the per-shard,
    /// local-slot-indexed `deleted_by` region needs. Returns `None` if the table is not shard-resident / a
    /// shard is invalid or missing device memory / the predicate cannot lower on a shard (the commit fails
    /// closed). Visibility = `None`: locate addresses PHYSICAL positions (a DELETE
    /// stamps a row by WHERE IT SITS, independent of read-time visibility; the raw buffer's rows are present),
    /// and only reads that already-committed shard buffer. WIRED by SV4b (the DELETE commit path routes
    /// through `try_tombstone_resident_delete_table`).
    #[cfg(test)]
    pub(crate) fn locate_resident_delete_slots(
        &self,
        table: &RelationalTable,
        predicate: &ResidentExpr,
    ) -> Option<Vec<(u32, Vec<u32>)>> {
        let shards = self.read_residency_shards();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        // Zone-map prune inputs: the mandatory (top-level-AND) int4 equalities the predicate requires.
        let mut constraints: Vec<(usize, i32)> = Vec::new();
        mandatory_int4_equalities(predicate, &mut constraints);
        let column_names: Vec<&str> = table.columns.iter().map(|c| c.name.as_str()).collect();
        let runtime_snapshot = self.router.runtime().snapshot();
        let mut out: Vec<(u32, Vec<u32>)> = Vec::new();
        for shard in table_shards.iter() {
            // S-d3 zone-map prune: skip a shard whose min/max provably excludes every mandatory needle (it
            // cannot hold a matching row). Same soundness as the sharded read: prune ONLY on a stat-carrying
            // shard that provably excludes; a no-stat shard is always kept.
            if !constraints.is_empty()
                && constraints.iter().any(|(col, needle)| {
                    shard_zone_map_excludes(
                        &column_names,
                        &shard.resident_device_int4_column_stats,
                        *col,
                        *needle,
                    )
                })
            {
                continue;
            }
            // Per-shard identity/validity precheck (mirror `execute_resident_sharded_via_general::source_for`).
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            let memory_pressure_active = runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            if !shard.is_valid(memory_pressure_active) {
                return None;
            }
            // D4: the buffer rides the loaded descriptor (generation-consistent by construction).
            let device_memory = shard.device_memory.clone()?;
            // The per-shard descriptor is capacity-strided + row_count-sized to THIS shard's buffer
            // (`resident_snapshot_for_shard`), so the predicate reads the shard's int4 columns at the right
            // offsets and returns slots LOCAL to `[0, row_count)`.
            let descriptor = self.resident_snapshot_for_shard(shard, table);
            let slots = self
                .lower_resident_predicate(
                    predicate,
                    table,
                    &descriptor,
                    &device_memory,
                    shard.row_count as u64,
                    None,
                )
                .ok()?;
            if !slots.is_empty() {
                out.push((shard.shard_id, slots));
            }
        }
        Some(out)
    }

    /// CPU-ENGINE RETIREMENT (ADR-006): the GENERATION-CONSISTENT twin of [`Self::locate_resident_delete_slots`]
    /// for the elided-DML predicate RESOLVE (prepare_delete / prepare_update, which run OFF the commit lock).
    /// For every resident slot matching `predicate`, returns a [`ShardPkHit`] whose device buffer + all three
    /// version regions (`deleted_by` / `created_by` / `row_id`) + descriptor are CAPTURED FROM THE SAME
    /// `shards.load()` snapshot the slot was computed against, gated by the W0 `shard_write_locate_cell_live`
    /// liveness check. This closes the concurrent TOCTOU the slot-only variant would expose to a lock-free
    /// caller: a reordering re-admit (VACUUM / SV3a recompaction) between two independent `shards.load()`s
    /// could otherwise apply one generation's slots to another generation's compacted buffer = a wrong row.
    /// Mirrors `locate_resident_pk_via_shard_index_detailed`'s single-snapshot discipline. `None`
    /// means no complete device verdict; a DML caller rejects before WAL or fails stop after the
    /// durable cut rather than reconstructing host state.
    pub(crate) fn locate_resident_delete_slots_detailed(
        &self,
        table: &RelationalTable,
        predicate: &ResidentExpr,
    ) -> Option<Vec<crate::engine_retained_read::ShardPkHit>> {
        self.locate_resident_slots_detailed(table, Some(predicate), None)
    }

    /// Compound-key counterpart: evaluate every independently typed equality/IS-NULL leaf to a
    /// retained device mask, AND the masks on-device, then compact once to local slots. Keeping the
    /// leaves independent lets mixed i32/i64/i128/text/uuid/bool keys share one exact device verdict
    /// without forcing unlike value buffers through a mono-typed expression VM.
    pub(crate) fn locate_resident_conjunct_slots_detailed(
        &self,
        table: &RelationalTable,
        conjuncts: &[ResidentExpr],
    ) -> Option<Vec<crate::engine_retained_read::ShardPkHit>> {
        self.locate_resident_slots_detailed(table, None, Some(conjuncts))
    }

    /// Predicate-free counterpart used by full-table DML. The identity range is generated and
    /// compacted on the device for each shard; the host receives only the approved local slots.
    pub(crate) fn locate_resident_all_slots_detailed(
        &self,
        table: &RelationalTable,
    ) -> Option<Vec<crate::engine_retained_read::ShardPkHit>> {
        self.locate_resident_slots_detailed(table, None, None)
    }

    fn locate_resident_slots_detailed(
        &self,
        table: &RelationalTable,
        predicate: Option<&ResidentExpr>,
        conjuncts: Option<&[ResidentExpr]>,
    ) -> Option<Vec<crate::engine_retained_read::ShardPkHit>> {
        // Off-lock prepare can race a generation publication after loading the shard map but
        // before the W0 liveness check. Retry only that benign supersession; unsupported shapes,
        // pressure, missing resources, and kernel failures remain authoritative declines.
        const GENERATION_RETRIES: usize = 64;
        for _ in 0..GENERATION_RETRIES {
            match self.locate_resident_slots_detailed_once(table, predicate, conjuncts) {
                DetailedLocateAttempt::Complete(hits) => return Some(hits),
                DetailedLocateAttempt::Declined => return None,
                DetailedLocateAttempt::Superseded => std::thread::yield_now(),
            }
        }
        None
    }

    fn locate_resident_slots_detailed_once(
        &self,
        table: &RelationalTable,
        predicate: Option<&ResidentExpr>,
        conjuncts: Option<&[ResidentExpr]>,
    ) -> DetailedLocateAttempt {
        debug_assert!(predicate.is_none() || conjuncts.is_none());
        let shards = self.read_residency_shards();
        let Some(table_shards) = shards.get(&table.name).filter(|shards| !shards.is_empty()) else {
            return self.locate_resident_single_slots_detailed_once(table, predicate, conjuncts);
        };
        let mut constraints: Vec<(usize, i32)> = Vec::new();
        if let Some(predicate) = predicate {
            mandatory_int4_equalities(predicate, &mut constraints);
        }
        if let Some(conjuncts) = conjuncts {
            for predicate in conjuncts {
                mandatory_int4_equalities(predicate, &mut constraints);
            }
        }
        let column_names: Vec<&str> = table.columns.iter().map(|c| c.name.as_str()).collect();
        let runtime_snapshot = self.router.runtime().snapshot();
        let mut out: Vec<crate::engine_retained_read::ShardPkHit> = Vec::new();
        for shard in table_shards.iter() {
            if shard.schema != table.schema || shard.table != table.name {
                return DetailedLocateAttempt::Declined;
            }
            let memory_pressure_active = runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            if !shard.is_valid(memory_pressure_active) {
                return DetailedLocateAttempt::Declined;
            }
            let Some(device_memory) = shard.device_memory.clone() else {
                return DetailedLocateAttempt::Declined;
            };
            // W0: the descriptor flags don't see concurrent invalidations — require the authoritative cell to
            // still publish THIS buffer, else decline (the located slot would address a superseded generation).
            if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                return DetailedLocateAttempt::Superseded;
            }
            // S-d3 zone-map prune only AFTER the W0 liveness proof. A stale generation's zone map
            // may exclude a value inserted after invalidation; pruning first would skip the
            // liveness check and turn an invalid generation into an authoritative empty result.
            if !constraints.is_empty()
                && constraints.iter().any(|(col, needle)| {
                    shard_zone_map_excludes(
                        &column_names,
                        &shard.resident_device_int4_column_stats,
                        *col,
                        *needle,
                    )
                })
            {
                continue;
            }
            let descriptor = self.resident_snapshot_for_shard(shard, table);
            let Some(slots) = self.resident_dml_slots(
                table,
                &descriptor,
                &device_memory,
                shard.row_count,
                predicate,
                conjuncts,
            ) else {
                return DetailedLocateAttempt::Declined;
            };
            // Every hit captures the SAME generation's buffer + regions + descriptor as its slot.
            for slot in slots {
                out.push(crate::engine_retained_read::ShardPkHit {
                    shard_id: shard.shard_id,
                    slot,
                    descriptor: descriptor.clone(),
                    device_memory: device_memory.clone(),
                    deleted_by: shard.deleted_by_region.clone(),
                    created_by: shard.created_by_region.clone(),
                    row_id: shard.row_id_region.clone(),
                });
            }
        }
        DetailedLocateAttempt::Complete(out)
    }

    fn locate_resident_single_slots_detailed_once(
        &self,
        table: &RelationalTable,
        predicate: Option<&ResidentExpr>,
        conjuncts: Option<&[ResidentExpr]>,
    ) -> DetailedLocateAttempt {
        let Some(entry) = self.relational_residency_entry(&table.name) else {
            return DetailedLocateAttempt::Declined;
        };
        let descriptor = entry.descriptor;
        if descriptor.schema != table.schema || descriptor.table != table.name {
            return DetailedLocateAttempt::Declined;
        }
        let pressured = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&descriptor.gpu_id);
        if !descriptor.is_valid() || pressured {
            return DetailedLocateAttempt::Declined;
        }
        let Some(device_memory) = self.read_resident_device_memory(&table.name) else {
            return DetailedLocateAttempt::Declined;
        };
        let row_id = self
            .read_state
            .residency
            .shard_row_id_memory
            .get(&(table.name.clone(), 0));
        if descriptor.row_count != 0 && row_id.is_none() {
            return DetailedLocateAttempt::Declined;
        }
        let Some(slots) = self.resident_dml_slots(
            table,
            &descriptor,
            &device_memory,
            descriptor.row_count,
            predicate,
            conjuncts,
        ) else {
            return DetailedLocateAttempt::Declined;
        };
        let deleted_by = self
            .read_state
            .residency
            .shard_deleted_by_memory
            .get(&(table.name.clone(), 0));
        let created_by = self
            .read_state
            .residency
            .shard_created_by_memory
            .get(&(table.name.clone(), 0));
        DetailedLocateAttempt::Complete(
            slots
                .into_iter()
                .map(|slot| crate::engine_retained_read::ShardPkHit {
                    shard_id: 0,
                    slot,
                    descriptor: descriptor.as_ref().clone(),
                    device_memory: device_memory.clone(),
                    deleted_by: deleted_by.clone(),
                    created_by: created_by.clone(),
                    row_id: row_id.clone(),
                })
                .collect(),
        )
    }

    fn resident_dml_slots(
        &self,
        table: &RelationalTable,
        descriptor: &crate::relational_model::RelationalResidencySnapshot,
        device_memory: &gpu_db_execution::CudaResidentDeviceMemory,
        row_count: usize,
        predicate: Option<&ResidentExpr>,
        conjuncts: Option<&[ResidentExpr]>,
    ) -> Option<Vec<u32>> {
        let row_count_u32 = u32::try_from(row_count).ok()?;
        match (predicate, conjuncts) {
            (Some(predicate), None) => self
                .lower_resident_predicate(
                    predicate,
                    table,
                    descriptor,
                    device_memory,
                    row_count as u64,
                    None,
                )
                .ok(),
            (None, Some(_)) if row_count == 0 => Some(Vec::new()),
            (None, Some(conjuncts)) => {
                let mut combined = None;
                for predicate in conjuncts {
                    let next = self
                        .resident_predicate_device_mask(
                            Some(predicate),
                            table,
                            descriptor,
                            device_memory,
                            row_count_u32,
                            None,
                        )
                        .ok()??;
                    combined = Some(match combined {
                        None => next,
                        Some(previous) => {
                            device_memory.and_predicate_masks(&previous, &next).ok()?
                        }
                    });
                }
                match combined {
                    Some(mask) => device_memory.predicate_mask_indices_u32(&mask).ok(),
                    None => device_memory
                        .row_range_indices_u32(row_count_u32, 0, row_count_u32)
                        .ok(),
                }
            }
            (None, None) => device_memory
                .row_range_indices_u32(row_count_u32, 0, row_count_u32)
                .ok(),
            (Some(_), Some(_)) => None,
        }
    }

    /// SV4 (GPU-native DELETE): LOCATE the resident slots matching `predicate` + stamp `deleted_by =
    /// commit_seq` on them IN PLACE (O(rows touched)), instead of the O(table) invalidate + re-admit. Returns
    /// `Some(n)` = n slots tombstoned (n may be 0: the predicate matched no resident row -- still a success);
    /// `None` = the test helper could not complete the device mutation. MUST run under the commit lock so the
    /// per-shard `deleted_by` get-or-allocate is atomic (SV2 prereq #2). WIRED by SV4b.
    #[cfg(test)]
    pub(crate) fn try_tombstone_resident_delete(
        &self,
        table: &RelationalTable,
        predicate: &ResidentExpr,
        commit_seq: Index,
    ) -> Option<usize> {
        let located = self.locate_resident_delete_slots(table, predicate)?;
        // Ledger #16 (SI-fix audit follow-up) — the DEFENSIVE consumer gate: the locate is
        // visibility-blind (physical int4-image match), so a caller-supplied STALE old image can
        // physically match an ALREADY-DEAD slot; re-stamping it would leave the truly-current
        // version live (the fixed SV6 double-read class). Read each located slot's deleted_by
        // FIRST and treat any already-tombstoned slot as NO MATCH (drop it) — the caller's
        // exact-count gate then mismatches and falls to the always-correct re-admit. One i64
        // read per located slot on an O(rows-touched) path; a false "already dead" can only
        // trigger a correct rebuild, never a wrong result.
        let shards = self.read_state.residency.shards.load();
        let table_shards = shards.get(&table.name)?;
        let mut total = 0usize;
        for (shard_id, slots) in &located {
            let live_slots: Vec<u32> = match table_shards
                .iter()
                .find(|shard| shard.shard_id == *shard_id)
                .and_then(|shard| shard.deleted_by_region.clone())
            {
                None => slots.clone(), // no region = delete-free shard: every located slot is live
                Some(region) => slots
                    .iter()
                    .copied()
                    .filter(|slot| {
                        region
                            .read_resident_i32_column(u64::from(*slot) * 8, 2)
                            .ok()
                            .map(|halves| {
                                let deleted =
                                    (halves[0] as u32 as u64) | ((halves[1] as u32 as u64) << 32);
                                // The delete-free fill (0x7F...) reads far above any real seq.
                                deleted > commit_seq
                            })
                            // A device-read failure keeps the slot: the stamp attempt below
                            // fails loudly -> None -> re-admit (never a silent drop).
                            .unwrap_or(true)
                    })
                    .collect(),
            };
            if live_slots.is_empty() {
                continue;
            }
            if !self.tombstone_resident_shard_slots(&table.name, *shard_id, &live_slots, commit_seq)
            {
                return None;
            }
            total = total.saturating_add(live_slots.len());
        }
        Some(total)
    }

    /// SV5 (commit path): for a single-entry UPDATE commit, TOMBSTONE the OLD version's resident slot + APPEND
    /// the NEW image to the open shard IN PLACE. ORDER matters:
    /// tombstone-OLD FIRST so `locate` runs on the buffer BEFORE the new row is appended -- an UPDATE that
    /// leaves the int4 columns UNCHANGED still locates EXACTLY the old row (count 1) rather than matching both
    /// the old + the just-appended new slot. Returns `true` only if BOTH steps succeed. Any failure
    /// returns `false`; the enclosing durable apply then wedges before acknowledgement and WAL replay
    /// reconstructs the device generation. Runs under commit_mutex + catalog latch.
    ///
    /// **SI (SV6 — the audit-flagged P2 flip-gate, FIXED):** the append + `row_count` bump happen BEFORE
    /// `publish_committed_seq`, and lock-free reads bind `read_txn_id = committed_seq()` then load `shards`
    /// separately -- so a concurrent reader that observes `committed_seq = C-1` can observe the shards with
    /// the appended row already present. The appended NEW version is therefore STAMPED
    /// `created_by = commit_seq` (a per-shard on-demand i64 region mirroring `deleted_by`, written while the
    /// slots are still invisible headroom — see the append path's ORDER comment), and EVERY sharded read
    /// path ANDs the device-side `created_by <= read_txn_id` lower bound (scan/VM conjunct, 3b route
    /// per-hit gate, batched gather gate; the un-gated GPU dense kernel DECLINES stamped shards). So the
    /// C-1 reader sees the key EXACTLY ONCE (the OLD version: `deleted_by = C > C-1` visible, new hidden)
    /// and a reader at C sees exactly the NEW one. Gated by the
    /// `sv6_created_by_gate_reader_at_prior_snapshot_never_sees_updated_key_twice` torn-window differential
    /// and the concurrent hammer test. (DELETE/SV4b needs no lower bound -- no new row; a plain INSERT
    /// append stays unstamped/born-visible, the milder as-if-later read of a decided commit.)
    #[cfg(test)]
    pub(crate) fn try_update_resident_commit(
        &self,
        cat: &DdlCatalogState,
        table_name: &str,
        old_rows: &[Vec<SqlValue>],
        new_rows: &[Vec<SqlValue>],
        commit_seq: Index,
        row_ids: Option<&[u64]>,
    ) -> bool {
        let Some(table) = cat.relational_catalog.get(table_name) else {
            return false;
        };
        self.try_update_resident_table(table, old_rows, new_rows, commit_seq, row_ids)
    }

    /// Device UPDATE maintenance using a table from the caller's pinned catalog generation.
    pub(crate) fn try_update_resident_table(
        &self,
        table: &RelationalTable,
        old_rows: &[Vec<SqlValue>],
        new_rows: &[Vec<SqlValue>],
        commit_seq: Index,
        row_ids: Option<&[u64]>,
    ) -> bool {
        // ADR-006: a ZERO-ROW UPDATE (WHERE matched nothing) is a data NO-OP — `old_rows`/`new_rows` are
        // both empty (parallel), nothing to tombstone or append, the elided table is byte-unchanged.
        // Report it handled so the commit path retains the unchanged device generation. Authority entry
        // is separately gated on a non-empty applied set, so the no-op cannot establish authority.
        if old_rows.is_empty() {
            return new_rows.is_empty();
        }
        // RETIREMENT A4b: MULTI-ROW — old/new/row_ids must be parallel and identity-complete;
        // ALL tombstones land before ANY append (the locate must run on the pre-append buffer).
        if old_rows.len() != new_rows.len() || row_ids.is_none_or(|ids| ids.len() != new_rows.len())
        {
            return false;
        }
        let Some(row_ids) = row_ids else {
            return false;
        };
        // 1. Tombstone every OLD version by its stable device identity (locates run on the buffer
        // BEFORE the appends). The identity+full-row verification covers nullable and mixed-width
        // images that the retired all-int4 structural predicate could not name.
        if !self.try_tombstone_rows_by_identity(table, old_rows, row_ids, commit_seq) {
            return false;
        }
        // 2. Append the NEW image to the open shard, stamped `created_by = commit_seq` (SV6 — the P2
        //    flip-gate fix): the append + row_count bump land BEFORE `publish_committed_seq`, so a
        //    concurrent reader bound to `committed_seq = C-1` can observe the appended slots; the stamp +
        //    the read path's `created_by <= read_txn_id` device conjunct hide the new version from that
        //    reader (it sees exactly the OLD version, still live at its snapshot). If this fails after
        //    the tombstone, the enclosing commit wedges before acknowledgement and recovery replays WAL.
        if !self.try_append_resident_int4_open_shard(
            &table.name,
            new_rows,
            crate::engine_residency::AppendCreatedBy::UpdateNewVersion(commit_seq),
            Some(row_ids),
        ) {
            return false;
        }
        true
    }
}
