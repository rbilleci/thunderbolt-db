//! Residency feature policy, elision eligibility, and telemetry ownership.

use super::*;

impl Engine {
    /// STRATA S-F: enable/disable automatic GPU-residency admission on commit. Default ON: committed
    /// tables become GPU-resident so subsequent reads take the GPU-native route. Turning it off is an
    /// explicit parity-oracle/operator kill switch. `&self` (an interior-mutable flag the commit path reads).
    pub fn set_auto_admit_on_commit(&self, on: bool) {
        self.auto_admit_on_commit
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn auto_admit_on_commit_enabled(&self) -> bool {
        self.auto_admit_on_commit
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// ADR-009 R1: enable/disable the GPU index-probe point-lookup route (default OFF). When on, a
    /// resident int4 unique-key equality batch probes a cached GPU hash index instead of full-scanning;
    /// non-unique columns / un-buildable indexes transparently fall back to the scan. `&self` (an
    /// interior-mutable flag the read path reads).
    pub fn set_index_probe_enabled(&self, on: bool) {
        self.index_probe_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn index_probe_enabled(&self) -> bool {
        self.index_probe_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// DECISIONS "lpb read levers" #1: route the lpb unique index probe through the DENSE-emit kernel. Default
    /// off; byte-identical to the atomic kernel. `&self` (interior-mutable flag the read path reads).
    pub fn set_dense_index_probe_enabled(&self, on: bool) {
        self.dense_index_probe_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn dense_index_probe_enabled(&self) -> bool {
        self.dense_index_probe_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Billions-of-rows segmented layout (S-d1): enable/disable admitting a table as a SEGMENTED shard list
    /// (routed through the sharded resident read path) instead of one capacity-padded unified buffer.
    /// DEFAULT OFF — the A/B lever to validate the shard path before flipping the default. Interior-mutable.
    pub fn set_shard_residency_enabled(&self, on: bool) {
        self.shard_residency_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn shard_residency_enabled(&self) -> bool {
        self.shard_residency_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// SV4b: enable GPU-native incremental DELETE — a single-entry DELETE commit locates + tombstones the
    /// deleted rows' resident slots IN PLACE (O(rows)) instead of the O(table) invalidate + re-admit. DEFAULT
    /// OFF (nested under the shard path); OFF => a DELETE re-admits exactly as before (byte-identical). The
    /// A/B lever for the incremental-DELETE win. Interior-mutable (the commit path reads it).
    pub fn set_resident_delete_tombstone_enabled(&self, on: bool) {
        self.resident_delete_tombstone_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn resident_delete_tombstone_enabled(&self) -> bool {
        self.resident_delete_tombstone_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// SV5: enable GPU-native incremental UPDATE — a single-entry UPDATE commit tombstones the old version's
    /// resident slot + appends the new image to the open shard IN PLACE (O(rows)) instead of the O(table)
    /// invalidate + re-admit. DEFAULT OFF (nested under the shard path); OFF => an UPDATE re-admits exactly as
    /// before (byte-identical). The A/B lever for the incremental-UPDATE win. Interior-mutable.
    ///
    /// **The audit-P2 `created_by` flip-gate is FIXED (SV6):** the appended new version is stamped
    /// `created_by = commit_seq` and every sharded read path ANDs the device-side
    /// `created_by <= read_txn_id` lower bound, so a concurrent reader at `committed_seq = C-1`
    /// (pre-publish torn read) sees the updated key exactly once (the OLD version). Gated by the SV6
    /// torn-window + concurrent-reader differentials. See `Engine::try_update_resident_commit`'s SI note.
    /// (The default stays OFF pending the remaining shards-default gates — sharded predicate NULL 3VL.)
    pub fn set_resident_update_tombstone_enabled(&self, on: bool) {
        self.resident_update_tombstone_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn resident_update_tombstone_enabled(&self) -> bool {
        self.resident_update_tombstone_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Sub-slice 3b: enable the CROSS-SHARD PK-INDEX point-lookup route on the sharded read path — a
    /// shard-resident int4 UNIQUE-key equality point lookup uses the cached hash+bloom `locate` to gather
    /// ONLY the located shard(s) instead of every zone-map-non-excluded shard. DEFAULT OFF (nested under the
    /// shard path); OFF => the sharded read scans + recompacts exactly as before (byte-identical). The A/B
    /// lever for the membership-pruning win under UPDATE key-scatter. Interior-mutable (the read path reads it).
    pub fn set_shard_index_probe_enabled(&self, on: bool) {
        self.shard_index_probe_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn shard_index_probe_enabled(&self) -> bool {
        self.shard_index_probe_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// lpb-for-shards wiring: enable serving a shard-resident int4 point-lookup BATCH (from the facade
    /// batcher) via the batched cross-shard gather instead of per-query single-flight. DEFAULT OFF; OFF =>
    /// `submit_sharded_point_lookups_batched` returns `None` (byte-identical). The A/B lever that LANDS the
    /// ~310x batched throughput on real workloads. Public (the facade toggles + the batched entry reads it).
    pub fn set_shard_batched_point_read_enabled(&self, on: bool) {
        self.shard_batched_point_read_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn shard_batched_point_read_enabled(&self) -> bool {
        self.shard_batched_point_read_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Sub-slice 3b: count of sharded point-lookup reads served by the CROSS-SHARD PK INDEX route (the cached
    /// `locate` restricted the gathered shard set). The non-vacuity signal that the index route actually fired
    /// — output equality can't prove it (the index route and the full scan return byte-identical rows by
    /// construction; only the SET of shards gathered differs, which `sharded_shards_gathered` reflects).
    pub fn shard_index_route_hits(&self) -> u64 {
        self.read_state
            .residency
            .shard_index_route_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Step 1 (lpb-for-shards): count of BATCHES served by the batched cross-shard point-lookup gather
    /// (`gather_sharded_int4_point_lookups_batched`). Non-vacuity signal that the batched path fired.
    pub fn sharded_point_batch_hits(&self) -> u64 {
        self.read_state
            .residency
            .sharded_point_batch_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Sub-slice 8 (GPU-native probe): count of batches served by the FULLY-GPU dense-emit path
    /// (`gather_sharded_int4_point_lookups_batched_gpu`). Non-vacuity signal that the GPU-native probe (vs the
    /// host-probe fallback) served the batch — output equality can't prove which path ran.
    pub fn sharded_point_gpu_probe_hits(&self) -> u64 {
        self.read_state
            .residency
            .sharded_point_gpu_probe_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Sub-slice 8 v3 (O(1) routing): count of GPU-native batches where the multi-shard kernel took the
    /// BINARY-SEARCH path (host-proven ascending-disjoint shards -> each needle routes to its one shard in
    /// O(log shards)). Non-vacuity signal that binary routing (vs the linear fallback) fired.
    pub fn sharded_point_binary_route_hits(&self) -> u64 {
        self.read_state
            .residency
            .sharded_point_binary_route_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// RETIREMENT A2: count of DML statements served by the DEVICE resolve (non-vacuity signal).
    pub fn dml_device_resolve_hits(&self) -> u64 {
        self.read_state
            .residency
            .dml_device_resolve_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// CPU-ENGINE RETIREMENT (ADR-006): count of SELECTs the specialized route declined but the GENERAL
    /// GPU Expr executor served on-device (instead of de-eliding to the CPU pinned path). Non-vacuity
    /// signal for the read fallback — proves a wider-type/non-enumerated shape stayed on the GPU.
    pub fn general_read_fallback_hits(&self) -> u64 {
        self.read_state
            .residency
            .general_read_fallback_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// PHASE C slice 1: enable/disable the VALUE-INDEX resolve for DELETE/UPDATE prepare (default
    /// ON). OFF = the O(table) seq_scan (the oracle path) — the A/B lever the differentials use.
    pub fn set_dml_value_index_resolve_enabled(&self, on: bool) {
        self.dml_value_index_resolve_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn dml_value_index_resolve_enabled(&self) -> bool {
        self.dml_value_index_resolve_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// RETIREMENT A2: enable/disable the DEVICE DML resolve (default ON). OFF -> the value-index
    /// resolve (slice 1), then the scan — the differential ladder.
    pub fn set_dml_device_resolve_enabled(&self, on: bool) {
        self.dml_device_resolve_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn dml_device_resolve_enabled(&self) -> bool {
        self.dml_device_resolve_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// RETIREMENT A3: count of constraint probes ANSWERED by the device index (non-vacuity signal;
    /// both true and false answers count — the FALSE answer is the load-bearing one).
    pub fn dml_device_validate_hits(&self) -> u64 {
        self.read_state
            .residency
            .dml_device_validate_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// RETIREMENT A3: enable/disable the DEVICE constraint-probe validators (default ON). OFF ->
    /// the value-index probes (slice 1b), then the scan validators — the differential ladder.
    pub fn set_dml_device_validate_enabled(&self, on: bool) {
        self.dml_device_validate_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn dml_device_validate_enabled(&self) -> bool {
        self.dml_device_validate_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// RETIREMENT A4e: enable/disable the HOST-INSTALL ELISION (default OFF — the A/B lever; the
    /// flip is gated on the SLO measurement + the ADR-013 stamps/publication gates + audits).
    /// W5a kill switch: covered inserts log binary WAL records (see `wal_binary`). NOTE for the
    /// flip checklist (audit 21eddaa7, MEDIUM): once BINWAL records exist in a segment, binaries
    /// OLDER than 21eddaa7 silently DROP them at replay (their from_utf8 skip arm) — the WAL is
    /// non-downgradeable past this commit once enabled.
    pub fn set_binary_wal_records_enabled(&self, on: bool) {
        self.binary_wal_records_enabled
            .store(on, std::sync::atomic::Ordering::Release);
    }

    pub(crate) fn binary_wal_records_enabled(&self) -> bool {
        self.binary_wal_records_enabled
            .load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn set_host_install_elision_enabled(&self, on: bool) {
        self.host_install_elision_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn host_install_elision_enabled(&self) -> bool {
        self.host_install_elision_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// TYPE-COVERAGE track 1: enable/disable elision for UNIQUE-INDEXED (PK'd) i32-section
    /// tables (Int4/Date/Int2). DEFAULT ON since the 2026-07-03 flip; OFF = the kill switch
    /// (stops NEW elisions only — already-elided tables keep rehydrating through the seams).
    pub fn set_constrained_elision_enabled(&self, on: bool) {
        self.constrained_elision_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn constrained_elision_enabled(&self) -> bool {
        self.constrained_elision_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// TYPE-COVERAGE track 2 slice 2: enable/disable i64-SECTION (Int8/Timestamp) columns in
    /// sharded admission (default OFF — flips after the read/append/elision stages + SLO + audit).
    pub fn set_shard_int8_section_enabled(&self, on: bool) {
        self.shard_int8_section_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn shard_int8_section_enabled(&self) -> bool {
        self.shard_int8_section_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// M1 (charter-pure): enable/disable the DEVICE write-locate (host PK-hash probe replacement).
    pub fn set_device_write_locate_enabled(&self, on: bool) {
        self.device_write_locate_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn device_write_locate_enabled(&self) -> bool {
        self.device_write_locate_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// E2.5c 2M+ push (b): enable/disable the FUSED merged-apply device pass.
    pub fn set_fused_apply_enabled(&self, on: bool) {
        self.fused_apply_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn fused_apply_enabled(&self) -> bool {
        self.fused_apply_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Fused merged-apply passes that actually ran (non-vacuity telemetry — output equality
    /// cannot prove the fused kernel produced the state).
    pub fn fused_apply_hits(&self) -> u64 {
        self.read_state
            .residency
            .fused_apply_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// M1 design B: enable/disable WAVE-TIME batched PK-unique validation.
    pub fn set_device_write_locate_wave_batch_enabled(&self, on: bool) {
        self.device_write_locate_wave_batch_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn device_write_locate_wave_batch_enabled(&self) -> bool {
        self.device_write_locate_wave_batch_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// M1 design B: is this INSERT's PK-unique check DEFERRABLE to the wave-time batched locate?
    /// The eligibility is SHARED by the off-lock skip (`prepare_insert`) and the wave-time
    /// validate (the sequencer), so they can never diverge into a constraint bypass. Requires:
    /// the wave-batch + device-locate flags; the table ELIDED (device-authoritative — the locate
    /// is the source of truth); every unique index has a canonical raw/fingerprint device key;
    /// NO CHECK / outbound-FK / inbound-FK (those aren't device-batch-validated
    /// here — they keep the off-lock path). Same-wave dups are caught by the unique-slot conflict
    /// ledger (#18); the wave-time locate catches ALREADY-COMMITTED dups.
    pub(crate) fn insert_unique_wave_batchable(
        &self,
        catalog: &CatalogSnapshot,
        table: &RelationalTable,
    ) -> bool {
        if !self.device_write_locate_wave_batch_enabled() || !self.device_write_locate_enabled() {
            return false;
        }
        if !self.table_install_elided(&table.name) {
            return false;
        }
        if !table.check_constraints.is_empty() || !table.foreign_keys.is_empty() {
            return false;
        }
        // No OTHER table references this one (inbound FK -> off-lock path).
        if catalog.relational_catalog.values().any(|other| {
            other
                .foreign_keys
                .iter()
                .any(|fk| fk.referenced_table == table.name)
        }) {
            return false;
        }
        // At least one unique index, and EVERY unique index key column has a canonical resident fold.
        // Raw single i32-section keys and flagged compound/single-wide fingerprints ride the same
        // batched device write-locate.
        let mut has_unique = false;
        for index in table.indexes.iter().filter(|index| index.unique) {
            has_unique = true;
            if !index_all_key_columns_foldable(table, index) {
                return false;
            }
        }
        has_unique
    }

    /// M1: PK locates served by the DEVICE kernel (non-vacuity telemetry).
    pub fn device_write_locate_hits(&self) -> u64 {
        self.read_state
            .residency
            .device_write_locate_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// U1: coalesced device VISIBLE-LOCATE launches (lane DELETE target resolution).
    pub fn device_visible_locate_hits(&self) -> u64 {
        self.read_state
            .residency
            .device_visible_locate_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// U1: lane DELETE tombstones stamped IN PLACE by the apply coalescer.
    pub fn lane_tombstone_applies(&self) -> u64 {
        self.read_state
            .residency
            .lane_tombstone_applies
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// RETIREMENT A4e (audit B1) + TYPE-COVERAGE track 1: may `table` ENTER elision?
    /// Strictly-Int4, no checks, no outbound FKs, NO OTHER TABLE REFERENCES IT — and UNIQUE
    /// indexes (PK'd tables, the core-banking shape) allowed ONLY under
    /// `constrained_elision_enabled` with BOTH validator-ladder flags live. The original B1
    /// hazard (constraint validation reading the elided host store's stale prefix = silent
    /// bypass) is closed at both ends: every hot-path validator probe now runs through the
    /// index-driven ladder (`validate_dml_constraints_via_index` -> `visible_row_with_value`,
    /// device-first, rehydrate-on-decline, self-pinned views — including `prepare_insert`,
    /// this slice) and the residual scan arm's source (`visible_relational_rows`) rehydrates
    /// elided tables itself. CHECK/FK exclusions stay: CHECKs ride the scan arm when the
    /// resolve flag is off, and FK elision is cross-table interplay (the ledgered next step).
    pub(crate) fn table_elision_eligible(
        &self,
        catalog: &CatalogSnapshot,
        table_name: &str,
    ) -> bool {
        let Some(table) = catalog.relational_catalog.get(table_name) else {
            return false;
        };
        let unique_ok = !table.indexes.iter().any(|index| index.unique)
            || (self.constrained_elision_enabled()
                && self.dml_value_index_resolve_enabled()
                && self.dml_device_validate_enabled());
        table.columns.iter().all(|column| {
            // TYPE-COVERAGE track 2 (stages 1 + iii): every FIXED-WIDTH-section type is
            // device-authoritative-capable (A4a/A4c type from the catalog; appends ride the
            // section-aware encoder). The gather requires the shard layout the flag admits,
            // so i64 columns only ever appear here when `shard_int8_section_enabled` built
            // them — eligibility composes with admission by construction.
            matches!(
                column.ty,
                gpu_db_sql::SqlType::Int4
                    | gpu_db_sql::SqlType::Date
                    | gpu_db_sql::SqlType::Int2
                    | gpu_db_sql::SqlType::Int8
                    | gpu_db_sql::SqlType::Timestamp
                    // TYPE-COVERAGE #14 (numeric): b128 (Numeric/Uuid) VALUE columns ride the
                    // fixed-width section + section-aware append. (The unique-index gate below still
                    // requires an i32-section key — device locate probes i32 only — so numeric stays
                    // a value column, not a PK/unique key.)
                    | gpu_db_sql::SqlType::Numeric { .. }
                    | gpu_db_sql::SqlType::Uuid
                    // TYPE-COVERAGE #14 (bool slice 1): bool VALUE columns are device-authoritative
                    // (the payload builder emits a 1-bit/row bitmap; the general executor reads it).
                    // NON-appendable for now (like int8 stage i): a bool-bearing open shard declines
                    // in-place append and RE-ADMITS (rebuild rebuilds the bitmap), so bool tables stay
                    // a single dense shard until slice 2 adds the incremental bitmap append. Bool is a
                    // value column only — the unique-index gate below keeps keys on the i32 section.
                    | gpu_db_sql::SqlType::Bool
                    // TYPE-COVERAGE #14 (text): TEXT value columns are device-authoritative (an
                    // 8-aligned offsets section + a bytes blob). Variable-length can't use capacity
                    // headroom, so a text-bearing open shard NEVER appends in place -> it ROLLS OVER
                    // (each commit seals a fresh DENSE text shard the payload builder emits); reads span
                    // the shards via the blob-concat + offset-rebase gather. Text stays a value column.
                    | gpu_db_sql::SqlType::Text
            )
        }) && table.indexes.iter().all(|index| {
            // Every unique key column must have a canonical device fold. Raw single i32-section
            // indexes keep their existing layout; compound and single wider/text keys use the
            // fingerprint index plus exact typed recheck.
            !index.unique || index_all_key_columns_foldable(table, index)
        }) && unique_ok
            // CHECK constraints DO NOT block elision (ADR-006): CHECK validation is ROW-LOCAL —
            // `validate_check_constraints_for_rows` evaluates the NEW values only (host-held
            // control-plane literals / device-materialized update images), never the tuple store; and
            // ALTER ADD CHECK's existing-row validation scans via the elision-safe-by-construction
            // DDL row-validator (which rehydrates first). The device-history re-resolve proof
            // already treats CHECK as deterministic-on-values.
            //
            // OUTBOUND FKs no longer block (ADR-006 FK elision, child side) when the table is
            // not SELF-REFERENCING (the prepare ladders' self-FK arm keeps the scan-validator
            // semantics — "a new row may provide for another new row" — which wants the host
            // path). The inbound child-reference check (`does any child row carry fk_col =
            // departed_parent_key?`) stays device-native for EVERY fk column type the column
            // gate above admits: the ELIDED scan fallback in `device_visible_row_with_value`
            // serves it via the Eq scan-locate (`device_eq_scan_literal` has one canonical arm
            // per type; i32-section columns may answer from the hash index first). The child's
            // OWN writes never need its host rows (item 3 probes the PARENT; new images are
            // host-held/device-materialized).
            && !table
                .foreign_keys
                .iter()
                .any(|fk| fk.referenced_table == table_name)
            // INBOUND FKs no longer block (ADR-006 FK elision, parent side): a table REFERENCED by
            // other tables may elide when EVERY inbound FK's referenced column (on THIS table) is a
            // single-column foldable PK/UNIQUE — exactly the shape `device_visible_row_with_value`
            // answers ON THE DEVICE (`locate_resident_pk_via_shard_index_detailed` + the elided
            // materialize), so a child INSERT's parent-exists probe and a parent DELETE's
            // surviving-provider probe stay device-native (a decline rehydrates — the existing
            // safety net, never a wrong answer). The unique-index requirement means `unique_ok`
            // above already demanded the constrained-elision device flags for such a table.
            // A parent DELETE/UPDATE's own inbound-FK validation reads the CHILDREN (non-elided —
            // outbound FKs still block) via the host, and its own rows are the host-held candidates.
            && catalog.relational_catalog.values().all(|other| {
                other.foreign_keys.iter().all(|fk| {
                    fk.referenced_table != table_name
                        || table.indexes.iter().any(|index| {
                            index.unique
                                && index.key_columns.len() == 1
                                && index.column == fk.referenced_column
                                && index_all_key_columns_foldable(table, index)
                        })
                })
            })
    }

    /// RETIREMENT A4e: is `table` device-authoritative (commits skip the host install)?
    /// `pub` for bench/telemetry (read-only; the A/B arms assert steady-state elided-ness).
    pub fn table_install_elided(&self, table: &str) -> bool {
        if let Some(snapshot) = self.current_transaction_read_snapshot() {
            return snapshot.elided_tables.contains(table);
        }
        self.read_state
            .residency
            .elided_tables
            .load()
            .contains(table)
    }

    /// RETIREMENT A4e: commits that skipped the host install (non-vacuity telemetry).
    pub fn host_install_elisions(&self) -> u64 {
        self.read_state
            .residency
            .host_install_elisions
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// RETIREMENT A4e: COW-add/remove a table from the elided set (serialized commit path only).
    /// Testing/probe seam (W5a recovery probe): elision normally engages automatically at the
    /// wave append flush; forcing it marks the table device-authoritative WITHOUT device
    /// backing, so use only in WAL/replay experiments that never read pre-restart state.
    #[doc(hidden)]
    pub fn set_table_install_elided(&self, table: &str, elided: bool) {
        let cur = self.read_state.residency.elided_tables.load();
        if cur.contains(table) == elided {
            return;
        }
        let mut next = (**cur).clone();
        if elided {
            next.insert(table.to_string());
        } else {
            next.remove(table);
        }
        self.read_state
            .residency
            .elided_tables
            .store(std::sync::Arc::new(next));
    }

    /// S-d2c: set the target row count per shard (the rollover/seal threshold). Settable small in tests.
    pub fn set_shard_size_target(&self, rows: usize) {
        self.shard_size_target
            .store(rows.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn shard_size_target(&self) -> usize {
        self.shard_size_target
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// DECISIONS "lpb read levers" #1: count of batches served by the DENSE-emit index probe. The test signal
    /// that the dense route actually ran (dense and atomic are byte-identical, so output equality can't prove
    /// which kernel produced the rows).
    pub fn dense_index_probe_hits(&self) -> u64 {
        self.read_state
            .residency
            .dense_index_probe_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Slice 1b-ii-c: count of commits served by the IN-PLACE open-shard append (vs a whole-table
    /// re-admit). The test/telemetry signal that the append actually fired — output equality and even
    /// device-ptr stability can't prove it (a same-size re-admit reuses the freed address).
    pub fn open_shard_append_hits(&self) -> u64 {
        self.read_state
            .residency
            .open_shard_append_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// S-d3: count of shards actually GATHERED (recompacted) by the sharded read after zone-map pruning.
    /// The non-vacuity signal that pruning fired — output equality can't prove a shard was skipped, since
    /// a pruned shard holds no matching rows and the result is identical either way.
    pub fn sharded_shards_gathered(&self) -> u64 {
        self.read_state
            .residency
            .sharded_shards_gathered
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The number of resident shards a table currently holds (0 if not shard-resident). Read-only
    /// telemetry for benchmarks/tests that assert a table really grew into N bounded shards (else a
    /// "flat latency vs shard count" claim could be vacuously true on a silently single-shard table).
    pub fn resident_shard_count(&self, table: &str) -> usize {
        self.read_state
            .residency
            .shards
            .load()
            .get(table)
            .map_or(0, |shards| shards.len())
    }

    pub fn rollover_budget_declines(&self) -> u64 {
        self.read_state
            .residency
            .rollover_budget_declines
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}
