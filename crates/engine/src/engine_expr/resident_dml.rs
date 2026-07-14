//! Current resident DML transition orchestration: physical old-version locate, generation gating,
//! tombstoning, and created-by-stamped UPDATE append. This is the exact live path; `R3-001` in
//! `docs/PLAN.md` remains the sole owner of future version-storage, index, and concurrency design.

use super::shard_pruning::{mandatory_int4_equalities, shard_zone_map_excludes};
use crate::engine_expr_ir::{ResidentBinaryOp, ResidentExpr};
use crate::relational_model::RelationalTable;
use crate::{DdlCatalogState, Engine};
use gpu_db_sql::{SqlType, SqlValue};
use gpu_db_types::Index;

impl Engine {
    /// SV4 (GPU-native DELETE -- LOCATE phase): find the resident `(shard_id, LOCAL slot)` positions of
    /// every row matching `predicate` (an int4-equality point-lookup shape), via zone-map-pruned per-shard
    /// `lower_resident_predicate`. PER-SHARD (NOT the recompacted unified buffer of the read path), so the
    /// returned slots are LOCAL to each shard's own device buffer -- exactly what the per-shard,
    /// local-slot-indexed `deleted_by` region needs. Returns `None` if the table is not shard-resident / a
    /// shard is invalid or missing device memory / the predicate cannot lower on a shard (caller falls back to
    /// the O(table) invalidate + re-admit). Visibility = `None`: locate addresses PHYSICAL positions (a DELETE
    /// stamps a row by WHERE IT SITS, independent of read-time visibility; the raw buffer's rows are present),
    /// and only reads that already-committed shard buffer. WIRED by SV4b (the DELETE commit path routes
    /// through `try_tombstone_resident_delete_commit`).
    pub(crate) fn locate_resident_delete_slots(
        &self,
        table: &RelationalTable,
        predicate: &ResidentExpr,
    ) -> Option<Vec<(u32, Vec<u32>)>> {
        let shards = self.read_state.residency.shards.load();
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
    /// Mirrors `locate_resident_pk_via_shard_index_detailed`'s single-snapshot discipline. `None` = decline
    /// (the caller rehydrates): not shard-resident, a shard invalid / memory-pressured / catalog-mismatched /
    /// superseded (W0) / missing its device memory, or a predicate that could not lower on a shard.
    pub(crate) fn locate_resident_delete_slots_detailed(
        &self,
        table: &RelationalTable,
        predicate: &ResidentExpr,
    ) -> Option<Vec<crate::engine_retained_read::ShardPkHit>> {
        let shards = self.read_state.residency.shards.load();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let mut constraints: Vec<(usize, i32)> = Vec::new();
        mandatory_int4_equalities(predicate, &mut constraints);
        let column_names: Vec<&str> = table.columns.iter().map(|c| c.name.as_str()).collect();
        let runtime_snapshot = self.router.runtime().snapshot();
        let mut out: Vec<crate::engine_retained_read::ShardPkHit> = Vec::new();
        for shard in table_shards.iter() {
            // S-d3 zone-map prune (same soundness as the slot-only variant: prune ONLY a stat-carrying shard
            // that provably excludes every mandatory needle; a no-stat shard is always kept).
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
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            let memory_pressure_active = runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            if !shard.is_valid(memory_pressure_active) {
                return None;
            }
            let device_memory = shard.device_memory.clone()?;
            // W0: the descriptor flags don't see concurrent invalidations — require the authoritative cell to
            // still publish THIS buffer, else decline (the located slot would address a superseded generation).
            if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                return None;
            }
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
        Some(out)
    }

    /// SV4 (GPU-native DELETE): LOCATE the resident slots matching `predicate` + stamp `deleted_by =
    /// commit_seq` on them IN PLACE (O(rows touched)), instead of the O(table) invalidate + re-admit. Returns
    /// `Some(n)` = n slots tombstoned (n may be 0: the predicate matched no resident row -- still a success,
    /// nothing to re-admit); `None` = the caller must fall back to invalidate + re-admit (not shard-resident /
    /// locate could not run / a tombstone write failed). A PARTIAL stamp before a `None` is harmless: the
    /// fallback re-admit rebuilds every shard all-live from the host store (which already applied the DELETE)
    /// AND SV4-prereq-#1 releases any partial `deleted_by` region. MUST run under the commit lock so the
    /// per-shard `deleted_by` get-or-allocate is atomic (SV2 prereq #2). WIRED by SV4b.
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

    /// COMPOUND KEYS (wider types, Stage 2b): locate + tombstone the deleted `row`'s resident slot via the
    /// compound FINGERPRINT index, for tables the int4-column predicate can't uniquely locate (an i64 key
    /// column is not in the predicate). Folds the row's key tuple to its fingerprint, probes the compound
    /// index, and — because the probe is fingerprint-based (a collision could point at a DIFFERENT tuple) —
    /// TUPLE-VERIFIES each hit by materializing the slot on-device and comparing the full key columns. Only
    /// the verified, snapshot-LIVE slot is tombstoned. Returns `Some(1)` on the unique match, else `None`
    /// (ambiguous / device decline / can't-materialize) -> the caller re-admits (always correct). `ord` is
    /// `index`'s position in `table.indexes`.
    fn try_tombstone_resident_delete_via_fingerprint(
        &self,
        table: &RelationalTable,
        index: &crate::relational_model::RelationalIndex,
        ord: usize,
        row: &[SqlValue],
        commit_seq: Index,
    ) -> Option<usize> {
        let fingerprint =
            crate::engine_residency::compound_index_row_fingerprint(table, index, row)?;
        let key_id = crate::engine_residency::index_probe_key_id(table, index, ord)?;
        let key_positions = crate::engine_residency::index_key_column_positions(table, index)?;
        let hits = self.locate_resident_pk_via_shard_index_detailed(table, key_id, fingerprint)?;
        // TUPLE-VERIFY every fingerprint hit: materialize the slot at the commit boundary (created_by <=
        // seq && deleted_by > seq = snapshot-live, so an already-tombstoned slot yields None and is
        // dropped — the SI-fix already-dead discipline), then confirm the full key tuple matches. A device
        // decline (materialize None) re-admits.
        let mut matched: Vec<(u32, u32)> = Vec::new();
        for hit in &hits {
            let materialized = self.materialize_resident_row_via_hit(table, hit, commit_seq);
            let mrow = match materialized {
                Some(Some(mrow)) => mrow,
                Some(None) => continue, // not snapshot-live (already dead / future): not our slot
                None => return None, // can't materialize (device err / wider value column) -> re-admit
            };
            if key_positions.iter().all(|&p| mrow.get(p) == row.get(p)) {
                matched.push((hit.shard_id, hit.slot));
            }
        }
        // EXACT-1: a unique key identifies exactly one live slot. Anything else (0 = the resolved row
        // moved/vanished; >1 = a fingerprint collision that both tuple-matched, impossible for a unique
        // key but guarded) declines to the re-admit.
        if matched.len() != 1 {
            return None;
        }
        let (shard_id, slot) = matched[0];
        if !self.tombstone_resident_shard_slots(&table.name, shard_id, &[slot], commit_seq) {
            return None;
        }
        Some(1)
    }

    /// COMPOUND KEYS (wider types): the first compound unique index whose slot the int4-column predicate
    /// CANNOT uniquely locate — i.e. it has a key column outside the i32 section (an i64 key). Such a
    /// table's DELETE/UPDATE must locate via the fingerprint index (`try_tombstone_resident_delete_via_
    /// fingerprint`); an all-i32-section table keeps the proven int4-predicate locate. Returns `(ord, index)`.
    fn compound_index_needing_fingerprint_locate(
        table: &RelationalTable,
    ) -> Option<(usize, &crate::relational_model::RelationalIndex)> {
        table.indexes.iter().enumerate().find(|(_, index)| {
            index.unique
                && crate::engine_residency::index_is_compound(index)
                && index.key_columns.iter().any(|name| {
                    table
                        .columns
                        .iter()
                        .find(|c| &c.name == name)
                        .is_some_and(|c| {
                            !matches!(c.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2)
                        })
                })
        })
    }

    /// SV4b (commit path): for a single-entry DELETE commit, LOCATE + tombstone the deleted rows' resident
    /// slots IN PLACE instead of the O(table) invalidate + re-admit. Builds an int4-equality predicate that
    /// matches the deleted row's resident int4 columns and stamps the located slots. Returns `true` (the
    /// caller SKIPS re-admit) ONLY when the located+tombstoned count EXACTLY equals the deleted-row count;
    /// ANY ambiguity or unsupported shape returns `false` -> the caller invalidates + re-admits (rebuild
    /// all-live from the host store = always correct, so a false here is only a missed optimization, never a
    /// wrong result). Conservative FIRST-SLICE scope: exactly one deleted row, all int4 columns non-NULL
    /// plain `Int4`. `cat` is the catalog the commit already holds (NO latch re-entry). Runs under
    /// commit_mutex + catalog latch, so the per-shard `deleted_by` get-or-allocate is atomic (SV2 prereq #2).
    /// (Audit note: `cat` is the WORKING catalog while `prepare_delete` decoded the row against the published
    /// snapshot; for a single-entry non-DDL commit under the held latch these are the same shape, and any
    /// mismatch is caught by the `row.len() != table.columns.len()` guard below -> fallback.)
    pub(crate) fn try_tombstone_resident_delete_commit(
        &self,
        cat: &DdlCatalogState,
        table_name: &str,
        deleted_rows: &[Vec<SqlValue>],
        commit_seq: Index,
    ) -> bool {
        // RETIREMENT A4b: MULTI-ROW — per-row locate + tombstone, each gated EXACT count == 1. Any
        // ambiguity on ANY row (dup int4 values across the statement's rows, a locate miss, an
        // int4-identical already-tombstoned slot, NULL/non-int4) returns false -> the caller
        // invalidates + re-admits, which SUPERSEDES any tombstones already stamped this commit
        // (they are pre-publish; the re-admit rebuilds all-live and releases the regions — the
        // same partial-failure argument SV5 documented for tombstone-without-append).
        if deleted_rows.is_empty() {
            // ADR-006: a ZERO-ROW DELETE (WHERE matched nothing) is a data NO-OP — nothing to tombstone,
            // the elided table is byte-unchanged. Report it HANDLED (`true`) so the commit path keeps the
            // table ELIDED instead of treating the no-op as unhandled and REHYDRATING (de-eliding) it — a
            // pure de-elide trigger on the common `DELETE ... WHERE <no match>` OLTP shape (confirmed via
            // backtrace: apply_and_publish_committed_inner's `!handled && elided -> rehydrate` arm).
            // `deleted_rows` is the APPLIED removed set (resolved at apply), so empty == genuinely zero
            // matches, never a resolution failure. Elision-ENTER is separately gated on a non-empty applied
            // set in the caller, so this no-op never drives a table INTO elision.
            return true;
        }
        let Some(table) = cat.relational_catalog.get(table_name) else {
            return false;
        };
        // COMPOUND KEYS (wider types): a compound key with an i64 column can't be located by the
        // int4-column predicate -> use the fingerprint index + tuple-verify. All-i32-section tables keep
        // the proven int4-predicate locate.
        let fp_index = Self::compound_index_needing_fingerprint_locate(table);
        for row in deleted_rows {
            if row.len() != table.columns.len() {
                return false;
            }
            if let Some((ord, index)) = fp_index {
                if !matches!(
                    self.try_tombstone_resident_delete_via_fingerprint(
                        table, index, ord, row, commit_seq
                    ),
                    Some(1)
                ) {
                    return false;
                }
                continue;
            }
            let Some(predicate) = Self::resident_int4_row_predicate(table, row) else {
                return false;
            };
            // EXACT-1 per row. NOTE: a slot tombstoned by an EARLIER row of this same statement
            // may still be visible to this locate (stamped at commit_seq, read below it) — that
            // can only happen when two deleted rows are int4-identical, and then the FIRST row's
            // locate already saw count 2 and bailed. The per-row gate is the wrong-results net.
            if !matches!(
                self.try_tombstone_resident_delete(table, &predicate, commit_seq),
                Some(1)
            ) {
                return false;
            }
        }
        // VACUUM #5: every stamped tombstone is a DEAD SLOT until a rebuild — feed the churn
        // signal the auto-trigger reads (serialized path; the counter resets on any re-admit).
        self.add_tombstone_churn(table_name, deleted_rows.len() as u64);
        true
    }

    /// The AND-of-int4-equalities predicate locating exactly one physical row image: `(col_i =
    /// v_i)` over the table's plain-`Int4` columns. `None` when any int4 column holds NULL or a
    /// non-`Int4` value, or the table has zero int4 columns (cannot safely locate) -> re-admit.
    fn resident_int4_row_predicate(
        table: &RelationalTable,
        row: &[SqlValue],
    ) -> Option<ResidentExpr> {
        let mut predicate: Option<ResidentExpr> = None;
        for (idx, column) in table.columns.iter().enumerate() {
            if column.ty != SqlType::Int4 {
                continue;
            }
            let value = match &row[idx] {
                SqlValue::Int4(v) => *v,
                _ => return None,
            };
            let eq = ResidentExpr::Binary {
                op: ResidentBinaryOp::Eq,
                lhs: Box::new(ResidentExpr::Column(idx)),
                rhs: Box::new(ResidentExpr::Int4Literal(value)),
            };
            predicate = Some(match predicate {
                None => eq,
                Some(prev) => ResidentExpr::Binary {
                    op: ResidentBinaryOp::And,
                    lhs: Box::new(prev),
                    rhs: Box::new(eq),
                },
            });
        }
        predicate
    }

    /// SV5 (commit path): for a single-entry UPDATE commit, TOMBSTONE the OLD version's resident slot + APPEND
    /// the NEW image to the open shard IN PLACE, instead of the O(table) invalidate + re-admit. ORDER matters:
    /// tombstone-OLD FIRST so `locate` runs on the buffer BEFORE the new row is appended -- an UPDATE that
    /// leaves the int4 columns UNCHANGED still locates EXACTLY the old row (count 1) rather than matching both
    /// the old + the just-appended new slot. Returns `true` (caller SKIPS re-admit) only if BOTH steps
    /// succeed; ANY failure (multi-row / NULL / non-int4 / dup-ambiguous / non-resident / no append headroom)
    /// returns `false` -> the caller invalidates + re-admits, which rebuilds the table all-live from the host
    /// store (old hidden + new present, the version rewrite already applied) = always correct. So a partial
    /// tombstone-without-append (append failed after the tombstone) is harmless -- the re-admit supersedes it
    /// and SV4-prereq-#1 releases the partial region. Runs under commit_mutex + catalog latch.
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
    pub(crate) fn try_update_resident_commit(
        &self,
        cat: &DdlCatalogState,
        table_name: &str,
        old_rows: &[Vec<SqlValue>],
        new_rows: &[Vec<SqlValue>],
        commit_seq: Index,
        row_ids: Option<&[u64]>,
    ) -> bool {
        // ADR-006: a ZERO-ROW UPDATE (WHERE matched nothing) is a data NO-OP — `old_rows`/`new_rows` are
        // both empty (parallel), nothing to tombstone or append, the elided table is byte-unchanged.
        // Report it HANDLED (`true`) so the commit path keeps the table ELIDED instead of REHYDRATING it
        // (the `DELETE/UPDATE ... WHERE <no match>` de-elide trigger). Elision-ENTER is separately gated on
        // a non-empty applied set in the caller, so this no-op never drives a table INTO elision.
        if old_rows.is_empty() {
            return new_rows.is_empty();
        }
        // RETIREMENT A4b: MULTI-ROW — old/new/row_ids must be parallel and identity-complete;
        // ALL tombstones land before ANY append (the locate must run on the pre-append buffer).
        if old_rows.len() != new_rows.len() || row_ids.is_none_or(|ids| ids.len() != new_rows.len())
        {
            return false;
        }
        // 1. Tombstone every OLD version's slot (locates run on the buffer BEFORE the appends).
        if !self.try_tombstone_resident_delete_commit(cat, table_name, old_rows, commit_seq) {
            return false;
        }
        // 2. Append the NEW image to the open shard, stamped `created_by = commit_seq` (SV6 — the P2
        //    flip-gate fix): the append + row_count bump land BEFORE `publish_committed_seq`, so a
        //    concurrent reader bound to `committed_seq = C-1` can observe the appended slots; the stamp +
        //    the read path's `created_by <= read_txn_id` device conjunct hide the new version from that
        //    reader (it sees exactly the OLD version, still live at its snapshot). If this fails AFTER the
        //    tombstone, the caller's re-admit rebuilds all-live from the host store (which already applied
        //    the version rewrite), superseding.
        if !self.try_append_resident_int4_open_shard(
            table_name,
            new_rows,
            crate::engine_residency::AppendCreatedBy::UpdateNewVersion(commit_seq),
            row_ids,
        ) {
            return false;
        }
        true
    }
}
