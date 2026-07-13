//! GPU residency management + resident-route planning (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for populating/admitting
//! resident snapshots (incl. on-GPU), the benchmark chunk/shard installs,
//! resident device-memory + bytes accounting, retained-read snapshot handles,
//! warmup/maintenance policy execution, and the resident-route planners
//! (plan_relational_resident_route + sharded variant) + residency status.

use super::*;

/// Snapshot construction, admission, and publication ownership.
mod admission;
/// Typed payload/key encoding and open-shard append construction.
mod payload;

pub(crate) use payload::{
    AppendCreatedBy, COMPOUND_KEY_ID_FLAG, CREATED_BY_VISIBLE_FILL_BYTE,
    DELETED_BY_LIVE_FILL_BYTE, ROW_ID_UNSTAMPED_FILL_BYTE, UnifiedResidentSnapshotParts,
    build_relational_device_payload, build_relational_device_payload_with_capacity,
    compound_index_row_fingerprint, compound_key_fingerprint, compound_key_type_supported,
    compound_unique_slot_id, compute_open_shard_int4_append_chunks,
    i32_section_needle, index_all_key_columns_foldable, index_is_compound,
    index_key_column_positions, index_probe_key_id, key_column_width_words,
    parse_relational_row_id, probe_key_id_positions, sql_value_as_int4,
    sql_value_from_i32_section, sql_value_from_i64_section, sql_value_key_words,
};

#[cfg(test)]
// STRUCT-001 keeps this parent-positioned include as a source-reconstruction boundary. Moving the
// 6,819-line test owner after production items would obscure exact extraction history for no runtime gain.
#[allow(clippy::items_after_test_module)]
mod capacity_payload_tests {
    use super::*;

    fn int4_cols() -> (Vec<String>, Vec<SqlType>) {
        (
            vec!["id".to_string(), "balance".to_string()],
            vec![SqlType::Int4, SqlType::Int4],
        )
    }
    fn int4_rows(n: i32) -> Vec<Vec<SqlValue>> {
        (0..n)
            .map(|i| vec![SqlValue::Int4(i), SqlValue::Int4(i * 10)])
            .collect()
    }

    include!("tests/residency_payload.rs");

    include!("tests/residency_shard_baseline.rs");

    /// Read a shard's ON-DEMAND `deleted_by` region back from device (DtoH), first `count` slots. Returns
    /// `None` when the shard has NO region (delete-free). u64 reconstructed from i32 LE (lo, hi) pairs.
    fn read_shard_deleted_by_region(
        e: &Engine,
        table: &str,
        shard_id: u32,
        count: usize,
    ) -> Option<Vec<u64>> {
        let region = e
            .read_state
            .residency
            .shard_deleted_by_memory
            .get(&(table.to_string(), shard_id))?;
        let halves = region
            .read_resident_i32_column(0, count * 2)
            .expect("read deleted_by region");
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let lo = halves[2 * i] as u32 as u64;
            let hi = halves[2 * i + 1] as u32 as u64;
            out.push((hi << 32) | lo);
        }
        Some(out)
    }

    /// Test helper: does ANY shard of `table` currently hold a LIVE `deleted_by` region (cell present AND
    /// `Some`)? False after either `invalidate_table` (publishes `None`, device buffer freed, cell kept) or
    /// `remove_table` (cell dropped). Use to prove a region was RELEASED. Reads the published cell map
    /// directly (in-crate).
    fn table_has_any_deleted_by_cell(e: &Engine, table: &str) -> bool {
        e.read_state
            .residency
            .shard_deleted_by_memory
            .cells
            .load()
            .iter()
            .any(|((cell_table, _), cell)| cell_table == table && cell.load().get().is_some())
    }

    /// Test helper: does ANY cell KEY for `table` still exist (regardless of `Some`/`None`)? Distinguishes
    /// `invalidate_table` (key KEPT as a `None` tombstone) from `remove_table` (key ERASED). Use to prove
    /// DROP fully removes the entry -- invalidate alone would leak a dangling `None` key per dropped table.
    fn table_has_any_deleted_by_key(e: &Engine, table: &str) -> bool {
        e.read_state
            .residency
            .shard_deleted_by_memory
            .cells
            .load()
            .keys()
            .any(|(cell_table, _)| cell_table == table)
    }

    /// SV6 test helper: does ANY shard of `table` hold a LIVE `created_by` region? Mirrors
    /// `table_has_any_deleted_by_cell` — the presence proof that the UPDATE-append STAMP path ran (a
    /// re-admit fallback rebuilds all-live with NO region), and the release proof for the lifecycle gates.
    fn table_has_any_created_by_cell(e: &Engine, table: &str) -> bool {
        e.read_state
            .residency
            .shard_created_by_memory
            .cells
            .load()
            .iter()
            .any(|((cell_table, _), cell)| cell_table == table && cell.load().get().is_some())
    }

    /// SV6 test helper: does ANY `created_by` cell KEY for `table` still exist? Mirrors
    /// `table_has_any_deleted_by_key` (DROP must erase keys, not just publish `None`).
    fn table_has_any_created_by_key(e: &Engine, table: &str) -> bool {
        e.read_state
            .residency
            .shard_created_by_memory
            .cells
            .load()
            .keys()
            .any(|(cell_table, _)| cell_table == table)
    }

    include!("tests/residency_sparse_visibility.rs");

    include!("tests/residency_update_visibility.rs");

    include!("tests/residency_pk_index.rs");

    include!("tests/residency_route_parity.rs");

    include!("tests/residency_elision_core.rs");

    include!("tests/residency_type_coverage.rs");

    include!("tests/residency_device_locate.rs");

    include!("tests/residency_wide_type_reads.rs");

    include!("tests/residency_elision_waves.rs");

    include!("tests/residency_maintenance_materialization.rs");

    include!("tests/residency_identity_validation.rs");

    include!("tests/residency_sharded_point_reads.rs");

    include!("tests/residency_capacity_budget.rs");
}

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
    /// is the source of truth); every unique index on a strictly-i32 column (the device locate
    /// probes i32 keys); NO CHECK / outbound-FK / inbound-FK (those aren't device-batch-validated
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
        // At least one unique index, and EVERY unique index's key column(s) are strictly-i32-section.
        // COMPOUND KEYS: a compound key over i32-section columns folds to a fingerprint surrogate and
        // rides the same batched device write-locate.
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
            // The A2/A3 device locate probes i32-SECTION keys only: a unique index on an
            // i64 column could not be validated device-side, so such a table must not
            // elide (its probes would decline -> rehydrate thrash at best). COMPOUND KEYS
            // (TYPE-COVERAGE #14 Track 3): EVERY key column must be i32-section — a compound
            // key over i32-section columns folds to a fingerprint surrogate that rides the
            // same i32 device index (`compound_key_fingerprint`).
            !index.unique || index_all_key_columns_foldable(table, index)
        }) && unique_ok
            // CHECK constraints DO NOT block elision (ADR-006): CHECK validation is ROW-LOCAL —
            // `validate_check_constraints_for_rows` evaluates the NEW values only (host-held
            // control-plane literals / device-materialized update images), never the tuple store; and
            // ALTER ADD CHECK's existing-row validation scans via the elision-safe-by-construction
            // DDL row-validator (which rehydrates first). The ledger-#18 re-resolve coverage proof
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
            // single-column i32-section PK/UNIQUE — exactly the shape `device_visible_row_with_value`
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
                                && table
                                    .columns
                                    .iter()
                                    .find(|c| c.name == fk.referenced_column)
                                    .is_some_and(|c| {
                                        matches!(
                                            c.ty,
                                            gpu_db_sql::SqlType::Int4
                                                | gpu_db_sql::SqlType::Date
                                                | gpu_db_sql::SqlType::Int2
                                        )
                                    })
                        })
                })
            })
    }

    /// RETIREMENT A4e: is `table` device-authoritative (commits skip the host install)?
    /// `pub` for bench/telemetry (read-only; the A/B arms assert steady-state elided-ness).
    pub fn table_install_elided(&self, table: &str) -> bool {
        self.read_state
            .residency
            .elided_tables
            .load()
            .contains(table)
    }

    /// TYPE-COVERAGE track 1 diagnostics: shard PK-index cache convergence counters
    /// (writer-side flush extensions / prober-side tail-DtoH extensions / full O(shard) rebuilds).
    pub fn pk_index_maintenance_stats(&self) -> (u64, u64, u64) {
        (
            self.read_state
                .residency
                .pk_index_writer_extends
                .load(std::sync::atomic::Ordering::Relaxed),
            self.read_state
                .residency
                .pk_index_prober_extends
                .load(std::sync::atomic::Ordering::Relaxed),
            self.read_state
                .residency
                .pk_index_rebuilds
                .load(std::sync::atomic::Ordering::Relaxed),
        )
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

    /// Slice 1b-ii: append an INSERT's APPLIED rows IN PLACE into the table's resident OPEN shard's
    /// capacity headroom, instead of a full re-admit. `new_rows` MUST be the post-coercion/post-default
    /// applied images (the WriteDelta's `PreparedMutation::Insert.inserted_rows`), in catalog order, so
    /// the appended bytes match what a full rebuild would store. Returns `true` iff it appended +
    /// republished; `false` (the caller MUST fall back to invalidate + re-admit) when the table is not
    /// purely-int4-resident, is invalidated, lacks headroom, or the device append fails. MUST run BEFORE
    /// `committed_seq` is published, so a reader at the new commit observes the advanced `row_count`
    /// (visibility ordering — same placement as the invalidation it replaces).
    pub(crate) fn try_append_resident_int4_open_shard(
        &self,
        table: &str,
        new_rows: &[Vec<SqlValue>],
        // RETIREMENT A1: `row_ids` = the appended rows' host identities (parsed from the commit's
        // write-set keys; an UPDATE append passes the ORIGINAL row's id). `None` = unknown (the
        // benchmark/synthetic paths): existing regions stay sentinel at those slots and no region
        // is created on rollover — identity-unknown, the device resolve declines.
        // D3 (ADR-013 pre1, STAMP-ALL-APPENDS): every append carries its birth commit seq(s) — the
        // sharded (default) layout stamps `created_by` for INSERT and UPDATE alike, so a reader
        // pinned at `s < commit_seq` no longer sees a decided-but-unpublished append (the former
        // "born-visible" premature-insert anomaly). See [`AppendCreatedBy`] for the variants
        // (uniform / per-row / update-new-version) and the single-buffer kill-switch scoping.
        created_by: AppendCreatedBy<'_>,
        row_ids: Option<&[u64]>,
    ) -> bool {
        if new_rows.is_empty() {
            return false;
        }
        debug_assert!(
            row_ids.is_none_or(|ids| ids.len() == new_rows.len()),
            "row_ids must parallel new_rows"
        );
        // S-d2b: a SHARD-resident table (the segmented layout, default-OFF flag) appends to its OPEN shard's
        // headroom instead of the single buffer. (Admission publishes a table to shards XOR snapshots, so
        // the two paths never overlap for one table.) ADR-006 (NULL coverage): the shard path now handles a
        // NULL in an appended row by rolling a DENSE shard whose validity bitmaps the payload builder
        // constructs (like TEXT), so a NULL insert STAYS ELIDED instead of de-eliding via re-admit.
        if self
            .read_state
            .residency
            .shards
            .load()
            .get(table)
            .is_some_and(|shards| !shards.is_empty())
        {
            return self.try_append_to_resident_open_shard(table, new_rows, created_by, row_ids);
        }
        // SINGLE-BUFFER path only: a NULL in an appended row would need a validity bitmap, but the single
        // buffer this path appends into is bitmap-free by construction (the `purely_int4` eligibility below
        // requires `resident_device_null_columns.is_empty()`) and this append writes a NULL int4 as a
        // placeholder 0 WITHOUT a bitmap. The device aggregate / DISTINCT / GROUP BY routes derive NULL-ness
        // solely from the bitmap, so an appended NULL would read as a phantom 0. Decline -> the caller
        // re-admits, which BUILDS the correct bitmap. (The SHARD path above now maintains bitmaps on the
        // rollover, so it no longer declines here; only the single-buffer kill-switch layout does.)
        if new_rows
            .iter()
            .any(|row| row.iter().any(|v| matches!(v, SqlValue::Null)))
        {
            return false;
        }
        // SV6 defensive: an UPDATE-appended NEW VERSION must be stamped + hidden from older readers,
        // and the single unified buffer carries no per-row version regions — decline and let the
        // caller re-admit (always correct). Unreachable today: the SV5 UPDATE route requires shard
        // residency. An INSERT append proceeds UNSTAMPED here: the single-buffer layout is the
        // kill-switch configuration outside the ADR-013/A5 gate (no region machinery); its
        // born-visible INSERT semantics are documented pre-D3 behavior.
        if matches!(created_by, AppendCreatedBy::UpdateNewVersion(_)) {
            return false;
        }
        let (capacity, row_start, column_count) = {
            let snapshots = self.read_state.residency.snapshots.load();
            let Some(entry) = snapshots.get(table) else {
                return false;
            };
            let s = &entry.descriptor;
            // Purely int4-resident: every column rides the i32 section (no other typed sections), so the
            // int4 append op covers the whole row. (Date/Int2 ride i32 too — handled by the op.)
            let purely_int4 = s.resident_device_int8_columns.is_empty()
                && s.resident_device_numeric_columns.is_empty()
                && s.resident_device_bool_columns.is_empty()
                && s.resident_device_text_columns.is_empty()
                && s.resident_device_null_columns.is_empty()
                && s.column_count == s.resident_device_int4_columns.len();
            if !s.is_valid() || !purely_int4 {
                return false;
            }
            match s.row_count.checked_add(new_rows.len()) {
                Some(end) if end <= s.capacity => (s.capacity, s.row_count, s.column_count),
                _ => return false, // no headroom (or overflow) -> caller re-admits (with fresh headroom)
            }
        };
        let Some(device_memory) = self.read_state.residency.device_memory.get(table) else {
            return false;
        };
        // The append op reads each value's SqlValue variant (Int4/Date/Int2) for encoding; the column
        // TYPES only gate eligibility + count, and a purely-int4 table is all-i32-section by definition.
        let column_types = vec![SqlType::Int4; column_count];
        let chunks = match compute_open_shard_int4_append_chunks(
            &column_types,
            capacity,
            row_start,
            new_rows,
        ) {
            Ok(chunks) => chunks,
            Err(_) => return false,
        };
        if device_memory.append_owned_chunks(chunks).is_err() {
            // A partial/failed append leaves bytes only in the (still-invisible) headroom beyond
            // row_count; returning false makes the caller invalidate + re-admit, discarding them.
            return false;
        }
        let k = new_rows.len();
        let appended_bytes = (k * column_count * std::mem::size_of::<i32>()) as u64;
        self.read_state.residency.with_snapshots_mut(|snapshots| {
            if let Some(entry) = snapshots.get_mut(table) {
                let desc = std::sync::Arc::make_mut(&mut entry.descriptor);
                desc.generation = desc.generation.saturating_add(1);
                desc.row_count += k;
                desc.resident_bytes = desc.resident_bytes.saturating_add(appended_bytes);
            }
        });
        // Slice 1b-ii (audit Finding A): the wave/lpb GPU index cache (engine_retained_read.rs) validates a
        // cached entry by (column, resident_device_ptr) ONLY — it is BLIND to generation/row_count. An
        // in-place append keeps the SAME device_ptr, so a cached index built over [0, old_row_count) would
        // be a stale HIT that reports the just-appended keys as not-found (a lost-from-reads committed
        // INSERT). Drop the table's entry so the next probe rebuilds over the new row_count. This runs
        // before the caller's publish_committed_seq, so a reader that observes the new committed_seq can
        // never bind the stale index. (In-flight probes pinned the prior index's own Arc; removing the
        // map entry only prevents NEW binds — the buffer frees once no submission holds it.)
        self.read_state
            .residency
            .wave_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(table);
        self.read_state
            .residency
            .open_shard_append_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// S-d2b: append a committed INSERT's applied rows IN PLACE into the resident table's OPEN shard's
    /// headroom (the last shard in `residency.shards`) — the shard-path analog of the single-buffer append.
    /// Empty + NULL-bearing rows are already rejected by the caller. Returns false (caller invalidates +
    /// re-admits) when the open shard isn't int4-appendable, is invalid, or has no headroom (seal + a fresh
    /// open shard on overflow is S-d2c), or the device append fails. Shard tables read via device
    /// recompaction and have no single-buffer `wave_index` to drop.
    /// (The sub-slice-3a `shard_pk_index` per-shard cache IS ptr-keyed but ALSO row_count-validated, so an
    /// in-place append grows row_count -> next probe misses -> rebuild; no explicit invalidation needed here.)
    fn try_append_to_resident_open_shard(
        &self,
        table: &str,
        new_rows: &[Vec<SqlValue>],
        created_by: AppendCreatedBy<'_>,
        row_ids: Option<&[u64]>,
    ) -> bool {
        // D3: materialize one birth stamp per appended row (validated len) — the in-place branch
        // stamps them into the open shard's created_by region and the rollover branch bakes them
        // into the new shard's region; both bump the descriptor's max_created_by high-water.
        let Some(stamps) = created_by.stamps_for(new_rows.len()) else {
            return false;
        };
        let stamps_max = stamps.iter().copied().max().unwrap_or(0);
        let pressured_gpus = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .clone();
        let k = new_rows.len();
        // ADR-006 (NULL coverage): does this appended BATCH carry any NULL? A null-bearing batch cannot
        // append in place (the in-place chunk encoder has no validity-bitmap channel); it rolls a DENSE
        // shard whose bitmaps the payload builder constructs — exactly like a text column.
        let batch_has_null = new_rows
            .iter()
            .any(|row| row.iter().any(|v| matches!(v, SqlValue::Null)));
        // Read the OPEN (last) shard's state once.
        let (
            shard_id,
            capacity,
            row_count,
            row_start,
            shard_int4_names,
            shard_int8_names,
            shard_numeric_names,
            shard_bool_layouts,
            shard_text_layouts,
            shard_null_layouts,
            gpu_id,
            schema,
            max_shard_id,
        ) = {
            let shards = self.read_state.residency.shards.load();
            let Some(table_shards) = shards.get(table) else {
                return false;
            };
            let Some(open) = table_shards.last() else {
                return false;
            };
            if !open.int4_appendable || !open.is_valid(pressured_gpus.contains(&open.gpu_id)) {
                return false;
            }
            (
                open.shard_id,
                open.capacity,
                open.row_count,
                open.row_start,
                open.resident_device_int4_columns.clone(),
                open.resident_device_int8_columns.clone(),
                open.resident_device_numeric_columns.clone(),
                open.resident_device_bool_columns.clone(),
                open.resident_device_text_columns.clone(),
                open.resident_device_null_columns.clone(),
                open.gpu_id,
                open.schema.clone(),
                table_shards.iter().map(|s| s.shard_id).max().unwrap_or(0),
            )
        };
        // TYPE-COVERAGE track 2 slice 2 stage (ii): CATALOG-ordered names/types drive the
        // section-aware chunk encoder + the rollover payload (mixed i32/i64 sections —
        // catalog order != section ordinal). Defensive arity guard: the shard's section
        // lists must cover the catalog exactly, else decline to the re-admit oracle.
        let Some(catalog_table) = self
            .catalog_snapshot()
            .relational_catalog
            .get(table)
            .cloned()
        else {
            return false;
        };
        let column_names: Vec<String> = catalog_table
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_types: Vec<SqlType> = catalog_table
            .columns
            .iter()
            .map(|column| column.ty)
            .collect();
        let column_count = column_types.len();
        if shard_int4_names.len()
            + shard_int8_names.len()
            + shard_numeric_names.len()
            + shard_bool_layouts.len()
            + shard_text_layouts.len()
            != column_count
        {
            return false;
        }
        // TYPE-COVERAGE #14 (text): a text-bearing shard is DENSE (no headroom) -> it never appends in
        // place; force the ROLLOVER branch (the in-place path's chunk encoder rejects text anyway).
        let has_text = !shard_text_layouts.is_empty();
        // ADR-006 (NULL coverage): a NULL-bearing shard is likewise DENSE (rolled with validity bitmaps), so
        // it never appends in place; and a null-carrying BATCH must roll a fresh bitmap-bearing dense shard
        // (the in-place chunk encoder has no validity-bitmap channel — it would write a NULL as a phantom 0).
        let has_null_shard = !shard_null_layouts.is_empty();
        let num_i32_cols = shard_int4_names.len();
        let num_i64_cols = shard_int8_names.len();
        let num_numeric_cols = shard_numeric_names.len();
        let num_bool_cols = shard_bool_layouts.len();

        // FITS the open shard's headroom -> append IN PLACE (1b-ii on the shard path). Text / null-bearing
        // shards never qualify (dense: row_count == capacity), and a null-carrying batch is excluded so it
        // takes the bitmap-building rollover; gate explicitly so the intent is clear.
        if !has_text
            && !has_null_shard
            && !batch_has_null
            && row_count.checked_add(k).is_some_and(|end| end <= capacity)
        {
            let Some(shard_device_memory) = self
                .read_state
                .residency
                .shard_device_memory
                .get(&(table.to_string(), shard_id))
            else {
                return false;
            };
            // The append position within THIS shard's buffer is its LOCAL row_count (rows [0, row_count)
            // are live; the new rows go at [row_count, row_count+k)), NOT the shard's global `row_start`.
            let chunks = match compute_open_shard_int4_append_chunks(
                &column_types,
                capacity,
                row_count,
                new_rows,
            ) {
                Ok(chunks) => chunks,
                Err(_) => return false,
            };
            // Column values in catalog order (used by the fused pass, the host PK-index cache
            // extension, and the device index maintenance below). NULL-free by the appendable
            // guard, so `sql_value_as_int4` yields exactly the bytes the chunks encode for the
            // i32 columns.
            let column_values: Vec<Vec<i32>> = (0..column_count)
                .map(|c| {
                    new_rows
                        .iter()
                        .map(|row| sql_value_as_int4(&row[c]))
                        .collect()
                })
                .collect();
            // E2.5c 2M+ push (b): the FUSED merged-apply device pass — column scatter +
            // created_by/row-id stamps + PK index insert in ONE staging HtoD + ONE launch
            // (replacing the ~8 driver calls of the unfused chain below). Int4-only shards
            // (the covered-INSERT shape); ineligible falls through to the unfused sequence,
            // byte-identical to before the flag.
            let fused = if self.fused_apply_enabled()
                && num_i64_cols == 0
                && num_numeric_cols == 0
                && num_bool_cols == 0
            {
                let append_started =
                    crate::engine_dml_concurrent::wave_device_phase_timing_enabled()
                        .then(std::time::Instant::now);
                // Per-COLUMN offsets only: the encoder's FINAL chunk is the device
                // row-count header (offset 0), which the fused submit publishes after fencing the kernel.
                let chunk_offsets: Vec<u64> = chunks
                    .iter()
                    .take(column_count)
                    .map(|chunk| chunk.byte_offset)
                    .collect();
                let outcome = self.try_fused_apply_in_place(
                    table,
                    shard_id,
                    &shard_device_memory,
                    &chunk_offsets,
                    &column_values,
                    row_count,
                    capacity,
                    gpu_id,
                    &stamps,
                    row_ids,
                );
                if let Some(started) = append_started {
                    crate::engine_dml_concurrent::WAVE_DEVICE_STATS[1].fetch_add(
                        started.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                match outcome {
                    Some(true) => true,
                    // Device failure mid-pass: bytes live only in invisible headroom beyond
                    // row_count; the caller invalidates + re-admits (same contract as the
                    // unfused arm's partial-failure rule).
                    Some(false) => return false,
                    None => false, // not eligible -> unfused sequence
                }
            } else {
                false
            };
            if !fused {
                // `deleted_by` needs no write on append — the headroom was pre-filled with the live sentinel at
                // admission, so appended rows are born live. SV6: an UPDATE-appended NEW VERSION additionally
                // stamps `created_by = commit_seq` (below); a plain INSERT append stays unstamped (born-visible).
                let append_started =
                    crate::engine_dml_concurrent::wave_device_phase_timing_enabled()
                        .then(std::time::Instant::now);
                let append_result = shard_device_memory.append_owned_chunks(chunks);
                if let Some(started) = append_started {
                    crate::engine_dml_concurrent::WAVE_DEVICE_STATS[1].fetch_add(
                        started.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                if append_result.is_err() {
                    // Partial/failed append leaves bytes only in invisible headroom beyond row_count;
                    // returning false makes the caller invalidate + re-admit, discarding them.
                    return false;
                }
                // TYPE-COVERAGE #14 (bool): the chunk encoder emits NO bytes for bool columns (a bitmap
                // is not a capacity-strided fixed-width chunk), so set the k appended rows' value bits
                // here via the device atomicOr op — writing into the pre-zeroed bitmap headroom at the
                // shard's LOCAL row_count. Same before-the-`row_count`-bump ordering as the version
                // stamps: the slots are still invisible headroom, so a torn (bits written, count not
                // bumped) state is unreadable, and a failure -> false -> re-admit (rebuild is truthful).
                for layout in &shard_bool_layouts {
                    let Some(col_idx) = column_names.iter().position(|n| n == &layout.name) else {
                        return false; // shard/catalog bool label mismatch -> decline to the oracle
                    };
                    let values: Vec<u8> = new_rows
                        .iter()
                        .map(|row| match row[col_idx] {
                            SqlValue::Bool(true) => 1u8,
                            // false / NULL leave the bit 0 (NULL-free by the appendable guard anyway;
                            // the validity bitmap, absent here, would decide a real NULL).
                            _ => 0u8,
                        })
                        .collect();
                    if shard_device_memory
                        .set_bool_bitmap_range(layout.bitmap_byte_offset, row_count as u32, &values)
                        .is_err()
                    {
                        return false;
                    }
                }
                // SV6 ORDER (load-bearing): stamp created_by BEFORE the `row_count` bump below publishes the
                // appended slots. The slots are still invisible headroom here, so a torn state (values + stamps
                // written, count not bumped) is unreadable; stamping AFTER the bump would let a reader bound to
                // an older snapshot observe the new version born-visible (created_by = fill 0) — exactly the
                // SV5 P2 double-read window this gate closes. A stamp failure -> false -> the caller re-admits
                // (the re-admit purge releases any partial region; rebuild-all-live is always correct).
                if !self.stamp_created_by_resident_shard_slots(
                    table, shard_id, row_count, capacity, gpu_id, &stamps,
                ) {
                    return false;
                }
                // RETIREMENT A1: stamp the appended slots' host identities (get-or-skip: a region-less
                // benchmark lineage skips; an identity-bearing shard gets exact stamps). Same
                // before-the-bump ordering as the version stamps.
                if let Some(ids) = row_ids {
                    if !self.stamp_row_id_resident_shard_slots(table, shard_id, row_count, ids) {
                        return false;
                    }
                }
            }
            let appended_bytes = (k
                * (num_i32_cols * std::mem::size_of::<i32>()
                    + num_i64_cols * std::mem::size_of::<i64>()
                    + num_numeric_cols * 16)) as u64;
            // S-d3: extend the open shard's zone map (min/max per int4 column) to cover the appended
            // rows. The stats vector is INT4-ORDINAL-aligned, so iterate only the i32-section catalog
            // columns, in order (stage ii: i64 columns carry no zone map — they simply never prune).
            let new_min_max: Vec<(i32, i32)> = (0..column_count)
                .filter(|&c| {
                    matches!(
                        column_types[c],
                        SqlType::Int4 | SqlType::Date | SqlType::Int2
                    )
                })
                .map(|c| {
                    new_rows.iter().fold((i32::MAX, i32::MIN), |(lo, hi), row| {
                        let v = sql_value_as_int4(&row[c]);
                        (lo.min(v), hi.max(v))
                    })
                })
                .collect();
            // TYPE-COVERAGE track 1 (ledger #3): writer-side PK-index cache maintenance — the
            // appended values are in hand, so cached (table, shard, col) entries extend O(k)
            // with no device read. ORDER (measured): extend BEFORE the row_count publish below.
            // Post-publish extension opened a per-flush window where preparers pinned to the
            // FRESH count found a stale entry and raced into tail-DtoH reads against this very
            // extension (run-to-run TPS swung 51-84k @32w); pre-publish, probers at the old
            // count read the AHEAD entry via the slot-bound rule and probers at the new count
            // find the cache already current. Stage (ii): entries are keyed by CATALOG col_idx;
            // i64 columns produce inert placeholder vecs (their probes decline pre-cache, so no
            // entry can exist to extend). NULL-free by the guard above, so `sql_value_as_int4`
            // yields exactly the bytes the chunks wrote for the i32 columns.
            self.extend_shard_pk_index_cache_on_append(
                table,
                shard_id,
                shard_device_memory.device_ptr(),
                row_count,
                &column_values,
            );
            // M1 (ledger #24): incrementally maintain the DEVICE PK index too (the index_insert
            // kernel), so the wave-batched device locate never triggers the O(rows) rebuild.
            // Only fires when a device index is cached (device_write_locate on); no-op otherwise.
            if !fused && self.device_write_locate_enabled() {
                let idx_started = crate::engine_dml_concurrent::wave_device_phase_timing_enabled()
                    .then(std::time::Instant::now);
                self.extend_shard_pk_device_index_on_append(
                    table,
                    shard_id,
                    shard_device_memory.device_ptr(),
                    row_count,
                    &column_values,
                    new_rows,
                );
                if let Some(started) = idx_started {
                    crate::engine_dml_concurrent::WAVE_DEVICE_STATS[2].fetch_add(
                        started.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
            }
            self.read_state.residency.with_shards_mut(|shards| {
                if let Some(table_shards) = shards.get_mut(table) {
                    if let Some(open) = table_shards.last_mut() {
                        open.row_count += k;
                        // D3: the high-water publishes WITH the row_count that exposes the slots —
                        // a reader at s >= hwm treats the shard as effectively version-free.
                        open.max_created_by = open.max_created_by.max(stamps_max);
                        open.resident_bytes = open.resident_bytes.saturating_add(appended_bytes);
                        for (stat, (lo, hi)) in open
                            .resident_device_int4_column_stats
                            .iter_mut()
                            .zip(new_min_max.iter())
                        {
                            stat.min = stat.min.min(*lo);
                            stat.max = stat.max.max(*hi);
                        }
                    }
                }
            });
            self.read_state
                .residency
                .open_shard_append_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return true;
        }

        // S-d2c ROLLOVER: the open shard is full -> SEAL it (leave it in place, immutable) and build + install
        // a NEW open shard holding the k rows (capacity = the target, so it grows to the target before the
        // next rollover). O(rows appended), NOT the O(table) re-admit -> this is what removes the 536M cap.
        // TYPE-COVERAGE #14 (text): a text column has NO capacity-strided headroom (the builder rejects
        // capacity > row_count for text), so a text-bearing rollover shard is DENSE (capacity == k) — it
        // never appends in place; the NEXT commit rolls another dense shard. Fixed-width/bool rollovers
        // keep the growth headroom.
        let has_text = column_types.iter().any(|ty| matches!(ty, SqlType::Text));
        // ADR-006 (NULL coverage): a null-carrying batch rolls a DENSE shard (capacity == k, like text) so
        // the validity bitmaps are exact for the live rows and the shard never appends in place afterward
        // (which would need in-place bitmap maintenance). A fixed-width null-FREE batch keeps growth headroom.
        let new_capacity = if has_text || batch_has_null {
            k
        } else {
            self.shard_size_target()
                .max(k.saturating_mul(2).next_power_of_two())
        };
        let (device_payload, text_layouts, bool_layouts, int4_stats, null_layouts) =
            match build_relational_device_payload_with_capacity(
                &column_names,
                &column_types,
                new_rows,
                new_capacity,
            ) {
                // Keep the columnar payload + the int4 zone-map stats (min/max over the k rows) for pruning
                // (S-d3). TYPE-COVERAGE #14: the bool bitmap AND text (offsets+blob) layouts (offsets into
                // THIS payload) carry. ADR-006 (NULL coverage): the VALIDITY BITMAP layouts (`null_cols`,
                // offsets into THIS payload) now ALSO carry, so a null-carrying rollover STAYS ELIDED with
                // correct NULL-ness (was discarded, forcing a de-elide via re-admit). Null-free batch -> empty.
                Ok((payload, text_cols, bool_cols, stats, _b128, null_cols)) => {
                    (payload, text_cols, bool_cols, stats, null_cols)
                }
                Err(_) => return false,
            };
        // R-1 / S-F: rollover is an admission event too. Account the new payload plus its
        // mandatory created_by region and optional row-identity region before allocating any of
        // them. If it would cross the configured/default GPU budget, decline atomically; the
        // caller invalidates this table and the normal admission path may evict an older table or
        // leave this relation to the bounded streaming executor. The commit remains durable.
        let rollover_bytes = (device_payload.len() as u64)
            .saturating_add((new_capacity as u64).saturating_mul(8))
            .saturating_add(if row_ids.is_some() {
                (new_capacity as u64).saturating_mul(8)
            } else {
                0
            });
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self
            .relational_residency_budget_bytes(gpu_id)
            .is_some_and(|budget| {
                self.relational_resident_bytes_for_gpu(gpu_id)
                    .saturating_add(rollover_bytes)
                    > budget
            })
        {
            self.read_state
                .residency
                .rollover_budget_declines
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        // SV1/SV2: the rolled shard carries NO version metadata in its payload — `created_by` is gone and
        // `deleted_by` is on-demand (allocated in `shard_deleted_by_memory` on the shard's first delete).
        let Some(new_device_memory) = self
            .relational_residency_device_memory(gpu_id, &device_payload)
            .map(Arc::new)
        else {
            return false;
        };
        let new_shard_id = max_shard_id.saturating_add(1);
        let pressured = pressured_gpus.contains(&gpu_id);
        // D4 (ADR-013 pre2): build the regions BEFORE the descriptor literal so their Arcs ride the
        // published descriptor — the map inserts below keep the same Arcs for write-side bookkeeping.
        // SV6: a version-stamped (UPDATE-appended) rollover's created_by region must be observable
        // with the shard itself; carrying it IN the descriptor makes that atomic by construction.
        // D3: every rolled shard's first `k` slots carry their birth stamps; the headroom keeps the
        // born-visible fill (0) and later appends stamp into it. The region costs 8B/slot on the
        // OPEN shard lineage only (bulk-admitted shards stay region-free, hwm 0); reclaiming sealed
        // shards' regions once hwm falls below every active reader is VACUUM's job (ledger #5).
        let rolled_created_by_region = {
            let mut created_payload =
                vec![CREATED_BY_VISIBLE_FILL_BYTE; new_capacity * std::mem::size_of::<u64>()];
            for (slot, stamp) in stamps.iter().enumerate() {
                created_payload[slot * 8..slot * 8 + 8].copy_from_slice(&stamp.to_le_bytes());
            }
            let Some(created_region) =
                self.relational_residency_device_memory(gpu_id, &created_payload)
            else {
                return false;
            };
            Some(Arc::new(created_region))
        };
        let rolled_row_id_region = if let Some(ids) = &row_ids {
            let mut payload =
                vec![ROW_ID_UNSTAMPED_FILL_BYTE; new_capacity * std::mem::size_of::<u64>()];
            for (slot, row_id) in ids.iter().enumerate() {
                payload[slot * 8..slot * 8 + 8].copy_from_slice(&row_id.to_le_bytes());
            }
            self.relational_residency_device_memory(gpu_id, &payload)
                .map(Arc::new)
        } else {
            None
        };
        if row_ids.is_some() && rolled_row_id_region.is_none() {
            return false;
        }
        let rollover_allocated_bytes = new_device_memory
            .metadata()
            .allocated_bytes
            .saturating_add(
                rolled_created_by_region
                    .as_ref()
                    .map_or(0, |region| region.metadata().allocated_bytes),
            )
            .saturating_add(
                rolled_row_id_region
                    .as_ref()
                    .map_or(0, |region| region.metadata().allocated_bytes),
            );
        if self
            .relational_residency_budget_bytes(gpu_id)
            .is_some_and(|budget| {
                self.relational_resident_bytes_for_gpu(gpu_id)
                    .saturating_add(rollover_allocated_bytes)
                    > budget
            })
        {
            self.read_state
                .residency
                .rollover_budget_declines
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        let new_shard = RelationalResidentShard {
            shard_id: new_shard_id,
            row_start: row_start.saturating_add(row_count),
            row_count: k,
            capacity: new_capacity,
            int4_appendable: true,
            resident_device_int4_column_stats: int4_stats,
            // Audit NOTE adopted: count i64 columns at 8 bytes + b128 (Numeric/Uuid) columns at
            // 16 bytes (was a telemetry undercount vs the admit path; allocated_bytes was always
            // correct).
            resident_bytes: (8
                + k * (num_i32_cols * std::mem::size_of::<i32>()
                    + num_i64_cols * std::mem::size_of::<i64>()
                    + num_numeric_cols * 16)
                + bool_layouts.len() * k.div_ceil(32) * 4
                // ADR-006 (NULL coverage): each validity bitmap is ceil(k/32) u32 words.
                + null_layouts.len() * k.div_ceil(32) * 4
                // TYPE-COVERAGE #14 (text): the offsets section ((k+1)*8) + the bytes blob per column.
                + text_layouts
                    .iter()
                    .map(|t| (k + 1) * 8 + t.bytes_len as usize)
                    .sum::<usize>()) as u64,
            allocated_bytes: new_device_memory.metadata().allocated_bytes,
            count_header_byte_offset: 0,
            resident_device_int4_columns: shard_int4_names.clone(),
            resident_device_int8_columns: shard_int8_names.clone(),
            // TYPE-COVERAGE #14 (numeric): the b128 (Numeric/Uuid) section rides the rollover payload.
            resident_device_numeric_columns: shard_numeric_names.clone(),
            // TYPE-COVERAGE #14 (bool): the bool bitmaps ride the rollover payload (offsets from the builder).
            resident_device_bool_columns: bool_layouts,
            // TYPE-COVERAGE #14 (text): the DENSE text (offsets+blob) layouts ride the rollover payload.
            resident_device_text_columns: text_layouts,
            // ADR-006 (NULL coverage): the validity-bitmap layouts (offsets into THIS payload) ride the
            // rollover, so a null-carrying rolled shard reads NULL-correctly on-device + rehydrate. Empty
            // for a null-free batch (the common case) -> byte-identical to before.
            resident_device_null_columns: null_layouts,
            gpu_id,
            schema,
            table: table.to_string(),
            device_memory_proof: Some(new_device_memory.metadata().clone()),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: pressured,
            memory_pressure_active: pressured,
            // D4: the descriptor IS the one-load snapshot — buffer + regions ride it. First `k`
            // created_by slots = `commit_seq`; the headroom keeps the born-visible fill for now
            // (D3 stamps later appends into it via stamp_created_by_resident_shard_slots).
            device_memory: Some(Arc::clone(&new_device_memory)),
            deleted_by_region: None,
            created_by_region: rolled_created_by_region.clone(),
            row_id_region: rolled_row_id_region.clone(),
            max_created_by: stamps_max,
        };
        // Write-side bookkeeping mirrors of the SAME Arcs (alloc/stamp/purge choreography unchanged);
        // readers take them from the published descriptor above.
        if let Some(created_region) = rolled_created_by_region {
            self.read_state
                .residency
                .shard_created_by_memory
                .insert_shard(table, new_shard_id, created_region);
        }
        if let Some(region) = rolled_row_id_region {
            self.read_state
                .residency
                .shard_row_id_memory
                .insert_shard(table, new_shard_id, region);
        }
        // Publish the new shard's device memory BEFORE its metadata, so a reader that observes the new shard
        // in the shards list always finds its device memory (the recompaction loads the list then the memory).
        self.read_state.residency.shard_device_memory.insert_shard(
            table,
            new_shard_id,
            new_device_memory,
        );
        self.read_state.residency.with_shards_mut(|shards| {
            if let Some(table_shards) = shards.get_mut(table) {
                table_shards.push(new_shard);
            }
        });
        self.read_state
            .residency
            .open_shard_append_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// Slice A2 (incremental DELETE): stamp `deleted_by[slot] = commit_seq` for each of `slots` in a
    /// resident shard's tombstone section, via ONE targeted device write (`append_owned_chunks`, which
    /// bounds-checks each chunk against the allocation). This is an OUT-OF-LINE tombstone: it writes only
    /// the `deleted_by` metadata word, NEVER the row's column bytes — so a lock-free, predicate-free reader
    /// can never observe a torn row (review Finding 3), and the change is a single aligned u64 store.
    ///
    /// Returns `false` (caller must fall back to invalidate + re-admit) if the shard is missing / any slot is
    /// out of `[0, row_count)` / the region allocation or device write fails. `slots` are LOCAL indices.
    ///
    /// WIRED into the DELETE commit path by SV4b (slot-finding via the pruned-shard predicate).
    /// **SV4 PREREQUISITES (audit-flagged):**
    ///  1. **Lifecycle/leak — DONE (SV4 prereq #1):** `shard_deleted_by_memory` cleanup is now wired at every
    ///     site the resident buffer it annotates is retired. Two categories, distinguished by whether an
    ///     invalidate precedes the retire:
    ///       - `invalidate_table` (device buffer freed, cell kept) on the three invalidate paths (serialized
    ///         `invalidate_relational_residency_table` + concurrent-commit + memory-pressure variants).
    ///       - full `remove_table` (keys erased) on the paths that retire a buffer WITHOUT a preceding
    ///         invalidate: the single-buffer AND sharded re-admit branches in `populate_..._snapshot_inner`
    ///         (a warmup/refresh has no invalidate), the BUDGET-EVICTION path (`RelationalResidentCache::
    ///         remove_table`, evicting a different table during admission), and `apply_drop_table` (a DROPped
    ///         table is gone for good — stronger than `shard_device_memory`, which leaves `None` cells on DROP).
    ///         Gates (all sabotage-verified non-vacuous): `shard_deleted_by_region_released_on_invalidate_and_drop`
    ///         (invalidate + DROP), `shard_deleted_by_region_released_on_warmup_readmit` (sharded re-admit with no
    ///         preceding invalidate), and `resident_cache_remove_table_releases_deleted_by_region` (the eviction-
    ///         cleanup method contract, currently defensive). This keeps a re-admit (rebuilt all-live from the host
    ///         store) from inheriting a stale tombstone region and stops evicted/dropped tables leaking regions.
    ///  2. **Concurrency:** hold the COMMIT LOCK across the get-or-allocate below, else two concurrent
    ///     first-deletes to the same shard both allocate + the losing region's `Arc` leaks (writes still land
    ///     safely; only the buffer leaks). SV4 runs this under the serialized commit lock, which is the fix.
    pub(crate) fn tombstone_resident_shard_slots(
        &self,
        table: &str,
        shard_id: u32,
        slots: &[u32],
        commit_seq: Index,
    ) -> bool {
        let stamped: Vec<(u32, Index)> = slots.iter().map(|&slot| (slot, commit_seq)).collect();
        self.tombstone_resident_shard_slots_stamped(table, shard_id, &stamped)
    }

    /// U1: the per-slot-stamp generalization of [`Self::tombstone_resident_shard_slots`] — a
    /// merged lane apply batch spans commit seqs, so each tombstone carries its OWN stamp
    /// (exactly like the append path's `InsertPerRow` birth stamps). Same on-demand region
    /// allocation, same bounds/decline contract.
    pub(crate) fn tombstone_resident_shard_slots_stamped(
        &self,
        table: &str,
        shard_id: u32,
        slots: &[(u32, Index)],
    ) -> bool {
        if slots.is_empty() {
            return true;
        }
        // Read the shard's shape once. The commit lock (when wired) makes this + the region allocation atomic;
        // even without it, `append_owned_chunks` re-bounds-checks every chunk vs the region's allocated_bytes,
        // so a torn read can only produce a rejected write (-> `false` -> re-admit), never an OOB.
        let (capacity, row_count, gpu_id) = {
            let shards = self.read_state.residency.shards.load();
            let Some(table_shards) = shards.get(table) else {
                return false;
            };
            let Some(shard) = table_shards.iter().find(|s| s.shard_id == shard_id) else {
                return false;
            };
            (shard.capacity, shard.row_count, shard.gpu_id)
        };
        // Bounds: every slot must be a live row of THIS shard (never headroom / out of range).
        if slots.iter().any(|&(slot, _)| (slot as usize) >= row_count) {
            return false;
        }
        // SV2: get-or-allocate the shard's ON-DEMAND `deleted_by` region (a separate `capacity`-sized u64
        // device buffer born all-live). A delete-free shard has NO entry -> the FIRST delete allocates it, so
        // the un-versioned majority pays zero. The region is `capacity` (not `row_count`) u64s so later
        // in-place appends into the open shard's headroom are already live without extending it.
        let width = std::mem::size_of::<u64>() as u64;
        let region = match self
            .read_state
            .residency
            .shard_deleted_by_memory
            .get(&(table.to_string(), shard_id))
        {
            Some(region) => region,
            None => {
                let _budget_allocation = self
                    .read_state
                    .residency
                    .budget_allocation_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                // Re-check after joining the allocation transaction: another first-delete may
                // have published the region while this caller waited.
                if let Some(region) = self
                    .read_state
                    .residency
                    .shard_deleted_by_memory
                    .get(&(table.to_string(), shard_id))
                {
                    region
                } else {
                    // Born all-live: every u64 is the large positive signed sentinel produced by
                    // byte-fill 0x7F. The signed visibility comparison therefore keeps it live.
                    let live_payload =
                        vec![DELETED_BY_LIVE_FILL_BYTE; capacity * std::mem::size_of::<u64>()];
                    if self
                        .relational_residency_budget_bytes(gpu_id)
                        .is_some_and(|budget| {
                            self.relational_resident_bytes_for_gpu(gpu_id)
                                .saturating_add(live_payload.len() as u64)
                                > budget
                        })
                    {
                        return false;
                    }
                    let Some(region) =
                        self.relational_residency_device_memory(gpu_id, &live_payload)
                    else {
                        return false;
                    };
                    let region = Arc::new(region);
                    if self
                        .relational_residency_budget_bytes(gpu_id)
                        .is_some_and(|budget| {
                            self.relational_resident_bytes_for_gpu(gpu_id)
                                .saturating_add(region.metadata().allocated_bytes)
                                > budget
                        })
                    {
                        return false;
                    }
                    self.read_state
                        .residency
                        .shard_deleted_by_memory
                        .insert_shard(table, shard_id, Arc::clone(&region));
                    // D4: republish the descriptor with the same region before releasing the
                    // allocation transaction; readers obtain resources from one shard snapshot.
                    self.read_state.residency.with_shards_mut(|shards| {
                        if let Some(table_shards) = shards.get_mut(table) {
                            if let Some(shard) =
                                table_shards.iter_mut().find(|s| s.shard_id == shard_id)
                            {
                                shard.deleted_by_region = Some(Arc::clone(&region));
                            }
                        }
                    });
                    region
                }
            }
        };
        // U1 perf lever B: ONE scatter launch (2 HtoDs + 1 kernel) instead of N per-slot HtoD
        // chunks — the measured device-apply cost (~468us/wave at ~75 tombstones). The region is
        // JUST deleted_by (0-based u64s): slot `s`'s stamp is at byte `s*8`, which the scatter
        // kernel computes from the slot index directly. `width` is unused on this path now.
        let _ = width;
        let slot_ids: Vec<u32> = slots.iter().map(|&(slot, _)| slot).collect();
        let stamps: Vec<u64> = slots.iter().map(|&(_, stamp)| stamp).collect();
        region.scatter_u64_slots(&slot_ids, &stamps).is_ok()
    }

    /// RETIREMENT A1: stamp the row-identity region for `k` just-appended contiguous slots
    /// `[first_slot, first_slot+k)` with the rows' `row_id`s. GET-OR-SKIP (not get-or-allocate):
    /// a shard WITHOUT a region (benchmark/synthetic install — no host identity exists) skips
    /// silently, keeping the absent-region = identity-unknown contract; a shard WITH one (admission
    /// or rollover created it, sentinel-filled headroom) gets exact stamps. Runs BEFORE the
    /// row_count bump (the slots are invisible headroom), same ordering as the version stamps.
    /// Get-or-allocate a shard's ON-DEMAND `created_by` region (factored from the stamp path
    /// so the FUSED apply pass shares the exact allocate + descriptor-republish semantics; see
    /// `stamp_created_by_resident_shard_slots` for the SV6/D4 contract).
    fn get_or_alloc_created_by_region(
        &self,
        table: &str,
        shard_id: u32,
        capacity: usize,
        gpu_id: u16,
    ) -> Option<Arc<CudaResidentDeviceMemory>> {
        if let Some(region) = self
            .read_state
            .residency
            .shard_created_by_memory
            .get(&(table.to_string(), shard_id))
        {
            return Some(region);
        }
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(region) = self
            .read_state
            .residency
            .shard_created_by_memory
            .get(&(table.to_string(), shard_id))
        {
            return Some(region);
        }
        let payload = vec![CREATED_BY_VISIBLE_FILL_BYTE; capacity * std::mem::size_of::<u64>()];
        if self
            .relational_residency_budget_bytes(gpu_id)
            .is_some_and(|budget| {
                self.relational_resident_bytes_for_gpu(gpu_id)
                    .saturating_add(payload.len() as u64)
                    > budget
            })
        {
            return None;
        }
        let region = Arc::new(self.relational_residency_device_memory(gpu_id, &payload)?);
        if self
            .relational_residency_budget_bytes(gpu_id)
            .is_some_and(|budget| {
                self.relational_resident_bytes_for_gpu(gpu_id)
                    .saturating_add(region.metadata().allocated_bytes)
                    > budget
            })
        {
            return None;
        }
        self.read_state
            .residency
            .shard_created_by_memory
            .insert_shard(table, shard_id, Arc::clone(&region));
        // D4 (ADR-013 pre2): REPUBLISH the descriptor with the new region (see the
        // deleted_by twin above). Born all-visible (fill 0), so a reader observing the
        // republished descriptor mid-commit is unchanged until the stamps + row_count land.
        self.read_state.residency.with_shards_mut(|shards| {
            if let Some(table_shards) = shards.get_mut(table) {
                if let Some(shard) = table_shards.iter_mut().find(|s| s.shard_id == shard_id) {
                    shard.created_by_region = Some(Arc::clone(&region));
                }
            }
        });
        Some(region)
    }

    /// E2.5c 2M+ push (b): the FUSED merged-apply device pass — column scatter + created_by /
    /// row-id stamps + PK hash-index insert in one staging HtoD + one launch + one decline DtoH,
    /// followed by the ordered device-header HtoD. The decline read completes every kernel block
    /// before that header publishes, preserving the SV6 stamp-before-publish order. Returns:
    /// - `None`  -> not eligible; the caller runs the unfused sequence (byte-identical);
    /// - `Some(true)`  -> the pass covered append + stamps + index maintenance;
    /// - `Some(false)` -> device failure mid-pass; bytes live only in invisible headroom, the
    ///   caller must NOT publish and must invalidate + re-admit (the unfused contract).
    #[allow(clippy::too_many_arguments)]
    fn try_fused_apply_in_place(
        &self,
        table: &str,
        shard_id: u32,
        shard_device_memory: &Arc<CudaResidentDeviceMemory>,
        chunk_offsets: &[u64],
        column_values: &[Vec<i32>],
        row_count: usize,
        capacity: usize,
        gpu_id: u16,
        stamps: &[Index],
        row_ids: Option<&[u64]>,
    ) -> Option<bool> {
        let k = stamps.len();
        if k == 0
            || chunk_offsets.len() != column_values.len()
            || column_values.iter().any(|col| col.len() != k)
            || row_count.saturating_add(k) > capacity
        {
            return None;
        }
        let base_row_u32 = u32::try_from(row_count).ok()?;
        // created_by region (get-or-allocate — same semantics as the unfused stamp path).
        let created_by_region =
            self.get_or_alloc_created_by_region(table, shard_id, capacity, gpu_id)?;
        // Row-id region: get-or-skip, exactly like `stamp_row_id_resident_shard_slots` (a
        // region-less lineage stamps nothing).
        let row_ids_arg = row_ids.and_then(|ids| {
            if ids.len() != k {
                return None;
            }
            self.read_state
                .residency
                .shard_row_id_memory
                .get(&(table.to_string(), shard_id))
                .map(|region| {
                    (
                        ids,
                        gpu_db_execution::CudaWriteDestination {
                            memory: Arc::clone(&region),
                            byte_offset: (row_count as u64)
                                * std::mem::size_of::<u64>() as u64,
                        },
                    )
                })
        });
        if row_ids.is_some() && row_ids_arg.is_none() {
            // ids provided but no region (or arity drift): only the no-region case is a
            // legitimate skip; arity drift declines to the unfused path's own guards.
            if row_ids.is_some_and(|ids| ids.len() != k) {
                return None;
            }
        }
        // PK device-index snapshot: mirror `extend_shard_pk_device_index_on_append`'s basis
        // validation + load rule for the (at most one) column with a LIVE cached device index.
        // More than one live entry -> not eligible (the fused kernel inserts into one index).
        let mut index_arg: Option<gpu_db_execution::CudaWriteIndex> = None;
        let mut index_col: Option<usize> = None;
        if self.device_write_locate_enabled() {
            let new_count = row_count + k;
            let device_ptr = shard_device_memory.device_ptr();
            let mut live: Vec<(usize, Arc<CudaResidentDeviceMemory>, u32, u32)> = Vec::new();
            {
                let cache = self
                    .read_state
                    .residency
                    .shard_pk_device_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                for col_idx in 0..column_values.len() {
                    let key = (table.to_string(), shard_id, col_idx);
                    let Some(entry) = cache.get(&key) else {
                        continue;
                    };
                    if entry.resident_device_ptr != device_ptr || entry.row_count != row_count {
                        continue; // stale basis -> the prober path converges it (unfused rule)
                    }
                    let Some(index) = entry.device_index.clone() else {
                        continue; // DECLINED is monotone under appends
                    };
                    live.push((col_idx, index, entry.table_mask, entry.hash_shift));
                }
            }
            match live.len() {
                0 => {}
                1 => {
                    let (col_idx, index, table_mask, hash_shift) = live.pop().expect("len 1");
                    let table_size = (table_mask as u64) + 1;
                    if (new_count as u64).saturating_mul(2) > table_size {
                        // Past the builder's load rule -> drop so the next probe rebuilds at
                        // the grown size (unfused rule), then run WITHOUT index maintenance.
                        self.read_state
                            .residency
                            .shard_pk_device_index
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .remove(&(table.to_string(), shard_id, col_idx));
                    } else {
                        index_arg = Some(gpu_db_execution::CudaWriteIndex {
                            memory: index,
                            table_mask,
                            hash_shift,
                            key_column: col_idx as u32,
                        });
                        index_col = Some(col_idx);
                    }
                }
                _ => return None, // multi-index shard: the unfused per-column loop handles it
            }
        }
        // Flatten col-major values + owned per-column destinations.
        let values: Vec<i32> = column_values.iter().flatten().copied().collect();
        let columns: Vec<gpu_db_execution::CudaWriteDestination> = chunk_offsets
            .iter()
            .map(|&byte_offset| gpu_db_execution::CudaWriteDestination {
                memory: Arc::clone(shard_device_memory),
                byte_offset,
            })
            .collect();
        let request = gpu_db_execution::FusedApplyRequest {
            columns: &columns,
            values: &values,
            stamps,
            created_by: gpu_db_execution::CudaWriteDestination {
                memory: created_by_region,
                byte_offset: (row_count as u64) * std::mem::size_of::<u64>() as u64,
            },
            row_ids: row_ids_arg,
            index: index_arg,
            base_row: base_row_u32,
            // The device row-count header word (the unfused path's FINAL append chunk).
            header: gpu_db_execution::CudaWriteDestination {
                memory: Arc::clone(shard_device_memory),
                byte_offset: 0,
            },
        };
        match shard_device_memory.submit_i32_fused_apply(&request) {
            Ok(dup) => {
                if let Some(col_idx) = index_col {
                    // Post-launch entry update, mirroring the unfused path: advance the basis,
                    // or DECLINE monotonically on a dup/overflow verdict.
                    let mut cache = self
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if let Some(entry) = cache.get_mut(&(table.to_string(), shard_id, col_idx)) {
                        if entry.resident_device_ptr == shard_device_memory.device_ptr()
                            && entry.row_count == row_count
                        {
                            if dup {
                                entry.device_index = None;
                            }
                            entry.row_count = row_count + k;
                        }
                    }
                }
                self.read_state
                    .residency
                    .fused_apply_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(true)
            }
            Err(_) => {
                if let Some(col_idx) = index_col {
                    // A failed launch may have partially mutated the index -> drop the entry
                    // (rebuild on next probe); never a wrong index. Same as the unfused path.
                    self.read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .remove(&(table.to_string(), shard_id, col_idx));
                }
                Some(false)
            }
        }
    }

    fn stamp_row_id_resident_shard_slots(
        &self,
        table: &str,
        shard_id: u32,
        first_slot: usize,
        row_ids: &[u64],
    ) -> bool {
        if row_ids.is_empty() {
            return true;
        }
        let Some(region) = self
            .read_state
            .residency
            .shard_row_id_memory
            .get(&(table.to_string(), shard_id))
        else {
            return true; // no identity region on this shard lineage: nothing to keep consistent
        };
        let mut bytes = Vec::with_capacity(row_ids.len() * 8);
        for row_id in row_ids {
            bytes.extend_from_slice(&row_id.to_le_bytes());
        }
        region
            .append_owned_chunks(vec![CudaOwnedDeviceMemoryChunk {
                byte_offset: (first_slot as u64) * 8,
                bytes,
            }])
            .is_ok()
    }

    /// SV6 (the SV5 `created_by` flip-gate): stamp `created_by[slot] = commit_seq` for the `k` just-appended
    /// CONTIGUOUS slots `[first_slot, first_slot + k)` of a resident shard, get-or-allocating the shard's
    /// ON-DEMAND `created_by` region — a `capacity`-sized i64 device buffer born all-visible
    /// ([`CREATED_BY_VISIBLE_FILL_BYTE`] = 0x00: `0 <= read_txn_id` for every snapshot) — on its first
    /// stamped append, so un-versioned shards pay zero (the same sparse-versioning property as
    /// `deleted_by`). The caller MUST invoke this BEFORE the shard's `row_count` bump publishes the slots
    /// (they are invisible headroom here — see the append path's ORDER comment) and runs under the commit
    /// lock, making the get-or-allocate atomic (SV2 prereq #2). Returns `false` (caller falls back to
    /// invalidate + re-admit; the re-admit purge releases any partial region) on any allocation or device
    /// write failure. ONE contiguous chunk write (`k * 8` bytes at `first_slot * 8`), bounds-checked by
    /// `append_owned_chunks` against the region's allocation.
    #[allow(clippy::too_many_arguments)] // mirrors the shard-shape tuple its caller already destructured
    fn stamp_created_by_resident_shard_slots(
        &self,
        table: &str,
        shard_id: u32,
        first_slot: usize,
        capacity: usize,
        gpu_id: u16,
        // D3: one birth stamp per appended slot (the wave-batched flush spans commit seqs).
        stamps: &[Index],
    ) -> bool {
        let k = stamps.len();
        if k == 0 {
            return true;
        }
        // Bounds: the stamped slots must lie inside the region (capacity slots). `append_owned_chunks`
        // re-checks against the real allocation, so a torn shape read can only reject, never write OOB.
        if first_slot.saturating_add(k) > capacity {
            return false;
        }
        let Some(region) = self.get_or_alloc_created_by_region(table, shard_id, capacity, gpu_id)
        else {
            return false;
        };
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(stamps));
        for stamp in stamps {
            bytes.extend_from_slice(&stamp.to_le_bytes());
        }
        region
            .append_owned_chunks(vec![CudaOwnedDeviceMemoryChunk {
                byte_offset: (first_slot as u64) * std::mem::size_of::<u64>() as u64,
                bytes,
            }])
            .is_ok()
    }

    /// STRATA S-B: commit-triggered, best-effort GPU-residency admission for the tables a commit
    /// mutated. Runs AFTER `publish_committed_seq` (so it snapshots the new generation) while the
    /// commit_mutex is held; it can NEVER fail the commit — over-budget / memory-pressure / GPU-absent /
    /// dropped-table simply leaves the table non-resident (reads fall back to the host path). N=1
    /// unified buffer per table (single-GPU); shard/spill is S-C/S-E.
    pub(crate) fn auto_admit_resident_tables(&self, tables: &std::collections::BTreeSet<String>) {
        // P4-2b (audit M4): NEVER admit a CHUNK-AUTHORITATIVE table — its store is FROZEN
        // (post-freeze writes live only in the chunks), so an admission (e.g. after a budget
        // raise) would publish a STALE resident snapshot that the resident route serves BEFORE
        // the streaming dispatch, with no de-auth guard in between.
        let class_map = self.read_state.residency.chunk_authoritative_tables.load();
        let tables: std::collections::BTreeSet<String> = tables
            .iter()
            .filter(|t| !class_map.contains_key(*t))
            .cloned()
            .collect();
        let tables = &tables;
        if tables.is_empty() {
            return;
        }
        // VACUUM #5: any rebuild resets the churn signal (the new generation is dense all-live).
        for table in tables {
            self.reset_tombstone_churn(table);
        }
        if tables.is_empty() {
            return;
        }
        let gpu_id = self.planner.default_gpu_id();
        let mut guard = self.ddl_catalog();
        let cat = &mut *guard;
        for table in tables {
            let _ = self.populate_relational_residency_snapshot_inner(cat, table, gpu_id);
        }
    }

    fn relational_residency_device_memory(
        &self,
        gpu_id: u16,
        payload: &[u8],
    ) -> Option<CudaResidentDeviceMemory> {
        let runtime = self.cuda_driver_probe_runtime();
        runtime.retain_device_memory_copy(gpu_id, payload).ok()
    }

    /// Build a TRANSIENT resident-like relation from already-materialized host `rows` -- a `RelationalTable`
    /// descriptor + an uploaded device payload that the GPU join path consumes EXACTLY like a published
    /// resident table (`lower_resident_predicate`, `project_*_rows_from_payload`, `hash_join_inner_i64`),
    /// but WITHOUT publishing/admitting/evicting anything (the descriptor + device memory live only for the
    /// caller's query). This is the M5 J5 bridge for a SYNTHESIZED `pg_catalog`/`information_schema`
    /// relation, which has no residency snapshot: synthesize its rows -> this helper -> the existing int4
    /// inner join over the transient payload. Charter: the catalog join runs on the SAME GPU kernels as a
    /// user-table join (no CPU relational join; only the host-rows gather crosses to the host, as for a
    /// resident table). `&self`: the upload only needs `cuda_driver_probe_runtime` (also `&self`).
    ///
    /// Mirrors `populate_relational_residency_snapshot_on_gpu`'s payload + descriptor build (the column
    /// lists feed `build_relational_device_payload`, whose offsets the descriptor's resident-column lists
    /// index), but SKIPS the MVCC tuple tail (the join reads columnar sections + host rows, never the tail)
    /// and the admission machinery. A 0-row relation is fine: the payload is still a non-empty 8-byte
    /// row-count header (the upload's empty-payload guard never trips), and the inner join then yields an
    /// empty result via the empty-survivor / empty-key short-circuits (an empty side is the join's identity).
    pub(crate) fn build_transient_relation_residency(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
    ) -> Result<(RelationalResidencySnapshot, CudaResidentDeviceMemory), ExecuteError> {
        let gpu_id = self.planner.default_gpu_id();
        let column_names: Vec<String> = table
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        // The resident-column lists name (in catalog order) the columns living in each type-grouped
        // payload section; the descriptor's offset helpers index `build_relational_device_payload`'s
        // sections via these lists, so they MUST use the SAME type filters as the resident builder.
        let resident_device_int4_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_int8_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_numeric_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let (
            device_payload,
            resident_device_text_columns,
            resident_device_bool_columns,
            resident_device_int4_column_stats,
            _resident_device_b128_columns,
            resident_device_null_columns,
        ) = build_relational_device_payload(&column_names, &column_types, rows)?;
        let runtime = self.cuda_driver_probe_runtime();
        let device_memory = runtime
            .retain_device_memory_copy(gpu_id, &device_payload)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: table.schema.clone(),
            table: table.name.clone(),
            generation: 1,
            row_count: rows.len(),
            capacity: rows.len(),
            column_count: table.columns.len(),
            resident_bytes: device_payload.len() as u64,
            resident_device_int4_columns,
            resident_device_int4_column_stats,
            resident_device_int8_columns,
            resident_device_numeric_columns,
            resident_device_bool_columns,
            resident_device_text_columns,
            resident_device_null_columns,
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: false,
            memory_pressure_active: false,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: Vec::new(),
            device_memory_proof: Some(device_memory.metadata().clone()),
        };
        Ok((snapshot, device_memory))
    }

    /// STRATA S-E.5: [`Engine::build_transient_relation_residency`] with the HtoD upload enqueued
    /// ASYNCHRONOUSLY (pinned staging + a private pooled copy stream), so the DMA overlaps the caller's
    /// host staging of the NEXT chunk and the previous chunk's kernels. The descriptor is complete
    /// immediately (proof metadata is allocation-time); the allocation must not be kernel-read until
    /// the returned pending copy's `wait()`. Falls back to the synchronous copy transparently when
    /// async staging is unavailable (`wait()` is then a no-op).
    /// 6c-3 (adopting the 6c-1 audit's F4): the payload + descriptor WITHOUT any upload — the cold
    /// tier's rebuild/eager-maintenance path constructs chunks for LATER replay (stage_cold_chunk
    /// stamps a fresh proof per upload), so building here must not spend a throwaway DMA — least of
    /// all on the COMMIT path. `device_memory_proof` is None until a replay stamps it.
    pub(crate) fn build_transient_relation_payload_only(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
    ) -> Result<(RelationalResidencySnapshot, Vec<u8>), ExecuteError> {
        let (snapshot, payload) = self.build_transient_parts(table, rows)?;
        Ok((snapshot, payload))
    }

    /// The shared payload+descriptor construction (no device work).
    fn build_transient_parts(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
    ) -> Result<(RelationalResidencySnapshot, Vec<u8>), ExecuteError> {
        let gpu_id = self.planner.default_gpu_id();
        let column_names: Vec<String> = table
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        let resident_device_int4_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_int8_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_numeric_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let (
            device_payload,
            resident_device_text_columns,
            resident_device_bool_columns,
            resident_device_int4_column_stats,
            _resident_device_b128_columns,
            resident_device_null_columns,
        ) = build_relational_device_payload(&column_names, &column_types, rows)?;
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: table.schema.clone(),
            table: table.name.clone(),
            generation: 1,
            row_count: rows.len(),
            capacity: rows.len(),
            column_count: table.columns.len(),
            resident_bytes: device_payload.len() as u64,
            resident_device_int4_columns,
            resident_device_int4_column_stats,
            resident_device_int8_columns,
            resident_device_numeric_columns,
            resident_device_bool_columns,
            resident_device_text_columns,
            resident_device_null_columns,
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: false,
            memory_pressure_active: false,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: Vec::new(),
            device_memory_proof: None,
        };
        Ok((snapshot, device_payload))
    }

    pub(crate) fn build_transient_relation_residency_async(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
        gpu_id: u16,
    ) -> Result<
        (
            RelationalResidencySnapshot,
            gpu_db_execution::PendingCudaResidentDeviceCopy,
            // The built device payload BYTES, handed back so the streaming cold tier (S-E.6) can
            // cache them for byte-replay (the upload staged them into pinned memory already).
            Vec<u8>,
        ),
        ExecuteError,
    > {
        let column_names: Vec<String> = table
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        let resident_device_int4_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_int8_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_numeric_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let (
            device_payload,
            resident_device_text_columns,
            resident_device_bool_columns,
            resident_device_int4_column_stats,
            _resident_device_b128_columns,
            resident_device_null_columns,
        ) = build_relational_device_payload(&column_names, &column_types, rows)?;
        let runtime = self.cuda_driver_probe_runtime();
        let pending = runtime
            .retain_device_memory_copy_async(gpu_id, &device_payload)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: table.schema.clone(),
            table: table.name.clone(),
            generation: 1,
            row_count: rows.len(),
            capacity: rows.len(),
            column_count: table.columns.len(),
            resident_bytes: device_payload.len() as u64,
            resident_device_int4_columns,
            resident_device_int4_column_stats,
            resident_device_int8_columns,
            resident_device_numeric_columns,
            resident_device_bool_columns,
            resident_device_text_columns,
            resident_device_null_columns,
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: false,
            memory_pressure_active: false,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: Vec::new(),
            device_memory_proof: Some(pending.metadata().clone()),
        };
        Ok((snapshot, pending, device_payload))
    }

    pub fn install_benchmark_relational_residency_chunks(
        &mut self,
        install: BenchmarkRelationalResidencyChunkInstall<'_>,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
        let table = install.table;
        let gpu_id = install.gpu_id;
        let row_count = install.row_count;
        let resident_bytes = install.resident_bytes;
        let allocated_bytes = install.allocated_bytes;
        let chunks = install.chunks;
        if row_count == 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission requires at least one generated row"
                    .to_string(),
            )));
        }
        if chunks.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission requires at least one retained chunk"
                    .to_string(),
            )));
        }
        let catalog_table = self
            .ddl_catalog_mut()
            .relational_catalog
            .get(table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table}\" does not exist"
                )))
            })?
            .clone();

        let visible_rows = self.visible_relational_row_count(table)?;
        if visible_rows != 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk admission requires relation \"{table}\" to have no SQL-visible rows; found {visible_rows}"
            ))));
        }
        Self::validate_benchmark_resident_chunk_columns(
            &catalog_table,
            &install.resident_device_int4_columns,
            &install.resident_device_int4_column_stats,
            &install.resident_device_text_columns,
        )?;
        let copied_bytes = chunks
            .iter()
            .try_fold(0_u64, |total, chunk| {
                let len = u64::try_from(chunk.bytes.len()).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident chunk length exceeds u64".to_string(),
                    ))
                })?;
                let end = chunk.byte_offset.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident chunk offset overflowed".to_string(),
                    ))
                })?;
                if end > allocated_bytes {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "benchmark resident chunk ending at byte {end} exceeds allocation {allocated_bytes}"
                    ))));
                }
                total.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident copied byte count overflowed".to_string(),
                    ))
                })
            })?;
        if copied_bytes == 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission copied no bytes".to_string(),
            )));
        }

        let previous_snapshot = self
            .read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| entry.descriptor.clone());
        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let runtime = self.cuda_driver_probe_runtime();
        let device_memory = runtime
            .retain_device_memory_chunks(gpu_id, allocated_bytes, chunks)
            .map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident chunk admission failed CUDA retained upload: {err}"
                )))
            })?;
        let allocated_bytes = device_memory.metadata().allocated_bytes;
        let admission_budget_bytes = self.relational_residency_budget_bytes(gpu_id);
        let (evicted_tables_on_admission, resident_bytes_after_admission) =
            self.admit_relational_residency_snapshot(table, gpu_id, allocated_bytes)?;
        let device_memory_proof = Some(device_memory.metadata().clone());
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: catalog_table.schema,
            table: catalog_table.name.clone(),
            generation: RelationalResidencySnapshot::next_generation(previous_snapshot.as_deref()),
            row_count,
            capacity: row_count,
            column_count: catalog_table.columns.len(),
            resident_bytes,
            resident_device_int4_columns: install.resident_device_int4_columns,
            resident_device_int4_column_stats: install.resident_device_int4_column_stats,
            // Benchmark install path: int8 device retention is not wired here yet (doc 19 — the
            // general executor reads int8 only from the standard populate path).
            resident_device_int8_columns: Vec::new(),
            resident_device_numeric_columns: Vec::new(),
            resident_device_bool_columns: Vec::new(),
            resident_device_text_columns: install.resident_device_text_columns,
            resident_device_null_columns: Vec::new(),
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: memory_pressure_active,
            memory_pressure_active,
            last_refresh_cost: previous_snapshot.as_ref().map(|previous| {
                RelationalResidencyRefreshCost {
                    previous_row_count: previous.row_count,
                    refreshed_row_count: row_count,
                    row_delta: row_count as i128 - previous.row_count as i128,
                    previous_resident_bytes: previous.resident_bytes,
                    refreshed_resident_bytes: resident_bytes,
                    resident_byte_delta: resident_bytes as i128 - previous.resident_bytes as i128,
                    refreshed_from_index: previous.valid_through_index,
                    refreshed_through_index: self.committed_seq(),
                    invalidated_by_txn_id: previous.invalidated_by_txn_id,
                    invalidated_at_index: previous.invalidated_at_index,
                    invalidated_by_memory_pressure: previous.invalidated_by_memory_pressure,
                }
            }),
            admission_budget_bytes,
            resident_bytes_after_admission,
            evicted_tables_on_admission,
            device_memory_proof,
        };
        let read_state = Arc::clone(&self.read_state);
        self.ddl_catalog()
            .relational_resident_cache
            .install_snapshot(
                catalog_table.name,
                snapshot.clone(),
                Some(device_memory),
                &read_state.residency,
            );
        Ok(snapshot)
    }

    pub fn install_benchmark_relational_residency_owned_chunks<I>(
        &mut self,
        install: BenchmarkRelationalResidencyOwnedChunkInstall<'_, I>,
    ) -> Result<RelationalResidencySnapshot, ExecuteError>
    where
        I: IntoIterator<Item = CudaOwnedDeviceMemoryChunk>,
    {
        let table = install.table;
        let gpu_id = install.gpu_id;
        let row_count = install.row_count;
        let resident_bytes = install.resident_bytes;
        let allocated_bytes = install.allocated_bytes;
        if row_count == 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission requires at least one generated row"
                    .to_string(),
            )));
        }
        let catalog_table = self
            .ddl_catalog_mut()
            .relational_catalog
            .get(table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table}\" does not exist"
                )))
            })?
            .clone();

        let visible_rows = self.visible_relational_row_count(table)?;
        if visible_rows != 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk admission requires relation \"{table}\" to have no SQL-visible rows; found {visible_rows}"
            ))));
        }
        Self::validate_benchmark_resident_chunk_columns(
            &catalog_table,
            &install.resident_device_int4_columns,
            &install.resident_device_int4_column_stats,
            &install.resident_device_text_columns,
        )?;

        let previous_snapshot = self
            .read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| entry.descriptor.clone());
        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let runtime = self.cuda_driver_probe_runtime();
        let device_memory = runtime
            .retain_device_memory_owned_chunks(gpu_id, allocated_bytes, install.chunks)
            .map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident chunk admission failed CUDA retained upload: {err}"
                )))
            })?;
        let allocated_bytes = device_memory.metadata().allocated_bytes;
        let admission_budget_bytes = self.relational_residency_budget_bytes(gpu_id);
        let (evicted_tables_on_admission, resident_bytes_after_admission) =
            self.admit_relational_residency_snapshot(table, gpu_id, allocated_bytes)?;
        let device_memory_proof = Some(device_memory.metadata().clone());
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: catalog_table.schema,
            table: catalog_table.name.clone(),
            generation: RelationalResidencySnapshot::next_generation(previous_snapshot.as_deref()),
            row_count,
            capacity: row_count,
            column_count: catalog_table.columns.len(),
            resident_bytes,
            resident_device_int4_columns: install.resident_device_int4_columns,
            resident_device_int4_column_stats: install.resident_device_int4_column_stats,
            // Benchmark install path: int8 device retention is not wired here yet (doc 19 — the
            // general executor reads int8 only from the standard populate path).
            resident_device_int8_columns: Vec::new(),
            resident_device_numeric_columns: Vec::new(),
            resident_device_bool_columns: Vec::new(),
            resident_device_text_columns: install.resident_device_text_columns,
            resident_device_null_columns: Vec::new(),
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: memory_pressure_active,
            memory_pressure_active,
            last_refresh_cost: previous_snapshot.as_ref().map(|previous| {
                RelationalResidencyRefreshCost {
                    previous_row_count: previous.row_count,
                    refreshed_row_count: row_count,
                    row_delta: row_count as i128 - previous.row_count as i128,
                    previous_resident_bytes: previous.resident_bytes,
                    refreshed_resident_bytes: resident_bytes,
                    resident_byte_delta: resident_bytes as i128 - previous.resident_bytes as i128,
                    refreshed_from_index: previous.valid_through_index,
                    refreshed_through_index: self.committed_seq(),
                    invalidated_by_txn_id: previous.invalidated_by_txn_id,
                    invalidated_at_index: previous.invalidated_at_index,
                    invalidated_by_memory_pressure: previous.invalidated_by_memory_pressure,
                }
            }),
            admission_budget_bytes,
            resident_bytes_after_admission,
            evicted_tables_on_admission,
            device_memory_proof,
        };
        let read_state = Arc::clone(&self.read_state);
        self.ddl_catalog()
            .relational_resident_cache
            .install_snapshot(
                catalog_table.name,
                snapshot.clone(),
                Some(device_memory),
                &read_state.residency,
            );
        Ok(snapshot)
    }

    pub fn install_benchmark_relational_residency_owned_shards(
        &mut self,
        install: BenchmarkRelationalResidencyOwnedShardInstall<'_>,
    ) -> Result<(), ExecuteError> {
        let table = install.table;
        if install.shards.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident shard admission requires at least one shard".to_string(),
            )));
        }
        let catalog_table = self
            .ddl_catalog_mut()
            .relational_catalog
            .get(table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table}\" does not exist"
                )))
            })?
            .clone();
        let visible_rows = self.visible_relational_row_count(table)?;
        if visible_rows != 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident shard admission requires relation \"{table}\" to have no SQL-visible rows; found {visible_rows}"
            ))));
        }

        let _total_resident_bytes = install.shards.iter().try_fold(0_u64, |total, shard| {
            if shard.row_count == 0 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident shard {} has no rows",
                    shard.shard_id
                ))));
            }
            if shard.chunks.is_empty() {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident shard {} has no retained chunks",
                    shard.shard_id
                ))));
            }
            total.checked_add(shard.resident_bytes).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "benchmark resident shard byte count overflowed".to_string(),
                ))
            })
        })?;
        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&install.gpu_id);
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let runtime = self.cuda_driver_probe_runtime();
        let mut shards = Vec::new();
        let mut device_memory = BTreeMap::new();
        let mut total_allocated_bytes = 0_u64;
        for shard in install.shards {
            let copied_bytes = shard.chunks.iter().try_fold(0_u64, |total, chunk| {
                let len = u64::try_from(chunk.bytes.len()).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident shard chunk length exceeds u64".to_string(),
                    ))
                })?;
                let end = chunk.byte_offset.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident shard chunk offset overflowed".to_string(),
                    ))
                })?;
                if end > shard.allocated_bytes {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "benchmark resident shard {} chunk ending at byte {end} exceeds allocation {}",
                        shard.shard_id, shard.allocated_bytes
                    ))));
                }
                total.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident shard copied byte count overflowed".to_string(),
                    ))
                })
            })?;
            if copied_bytes == 0 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident shard {} copied no bytes",
                    shard.shard_id
                ))));
            }
            let retained = runtime
                .retain_device_memory_owned_chunks(
                    install.gpu_id,
                    shard.allocated_bytes,
                    shard.chunks,
                )
                .map_err(|err| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "benchmark resident shard admission failed CUDA retained upload: {err}"
                    )))
                })?;
            let retained_allocated_bytes = retained.metadata().allocated_bytes;
            total_allocated_bytes = total_allocated_bytes
                .checked_add(retained_allocated_bytes)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident shard allocated byte count overflowed".to_string(),
                    ))
                })?;
            let device_memory_proof = Some(retained.metadata().clone());
            shards.push(RelationalResidentShard {
                shard_id: shard.shard_id,
                row_start: shard.row_start,
                row_count: shard.row_count,
                // Benchmark shards are read DENSE (explicit chunk layouts sized by row_count) and are not
                // append targets.
                capacity: shard.row_count,
                int4_appendable: false,
                // S-d3: benchmark shards carry no zone map -> never pruned (always gathered).
                resident_device_int4_column_stats: Vec::new(),
                // A1/SV2: benchmark shards carry no version metadata (no on-demand deleted_by region) -> the
                // visibility mask is skipped (they read as all-live, correct for latest-snapshot benchmarks).
                resident_bytes: shard.resident_bytes,
                allocated_bytes: retained_allocated_bytes,
                count_header_byte_offset: 0,
                resident_device_int4_columns: shard.resident_device_int4_columns,
                resident_device_int8_columns: Vec::new(), // benchmark chunks are int4-only
                resident_device_numeric_columns: Vec::new(),
                resident_device_bool_columns: Vec::new(),
                resident_device_text_columns: shard.resident_device_text_columns,
                // Benchmark installs carry no NULL metadata (dense, read-only, NULL-free chunks).
                resident_device_null_columns: Vec::new(),
                gpu_id: install.gpu_id,
                schema: catalog_table.schema.clone(),
                table: catalog_table.name.clone(),
                device_memory_proof,
                invalidated_by_txn_id: None,
                invalidated_at_index: None,
                invalidated_by_memory_pressure: memory_pressure_active,
                memory_pressure_active,
                // D4: `install_shards` attaches `device_memory` from the map (the one enforcement
                // point); benchmark shards carry no version/identity regions (all-live, read-only).
                device_memory: None,
                deleted_by_region: None,
                created_by_region: None,
                row_id_region: None,
                max_created_by: 0,
            });
            device_memory.insert(shard.shard_id, Arc::new(retained));
        }
        let (_evicted_tables_on_admission, _resident_bytes_after_admission) = self
            .admit_relational_residency_snapshot(
                table,
                install.gpu_id,
                total_allocated_bytes,
            )?;
        shards.sort_by_key(|shard| (shard.row_start, shard.shard_id));
        let read_state = Arc::clone(&self.read_state);
        self.ddl_catalog()
            .relational_resident_cache
            .install_shards(
                catalog_table.name,
                shards,
                device_memory,
                &read_state.residency,
            );
        Ok(())
    }

    /// RETIREMENT A4e (audit B3): rehydrate an elided table FROM AN OFF-COMMIT-LOCK context
    /// (the CPU-shape read seam, the execute_text DDL entry). `rehydrate_elided_table` mutates the
    /// host store via COW `with_table_mut` — safe ONLY under the commit lock (writers + other
    /// rehydrators serialize there; a lost-update would leave the table DE-ELIDED WITH A STALE
    /// STORE = permanent wrong reads). Mid-commit internal reads (matview refresh) already HOLD
    /// the lock — detected via the same thread-local that suppresses their leader check — so they
    /// rehydrate directly (a second acquisition would self-deadlock). The elided-ness RE-CHECK
    /// under the lock closes the race with a rehydrator that won the lock first.
    /// VACUUM #5 (A5 gate): REBUILD a churned table's residency DENSE + ALL-LIVE — reclaims
    /// tombstoned slots and stale duplicate physical keys (an SV5/A4b update-append leaves the
    /// old version's slot holding the key, which dup-declines the per-shard PK index until a
    /// rebuild changes the buffer ptr — the monotone decline clears BY DESIGN on a new
    /// generation). Composition of audited pieces: an ELIDED table first REHYDRATES (the A4c
    /// device gather is the truth; the host store is a stale prefix), then the standard
    /// invalidate + re-admit rebuilds dense from the now-complete store; a non-elided table's
    /// store is already complete, so it skips straight to the rebuild. The table RE-ENTERS
    /// elision on its next handled commit (the normal entry path) — vacuum does not special-case
    /// it. Runs under the COMMIT LOCK (the same discipline as `rehydrate_elided_serialized`; the
    /// mid-commit-read detection makes an auto-trigger from inside a commit safe). The churn
    /// counter resets so the auto-trigger re-arms.
    ///
    /// V2 (ledgered): KEY-CLUSTERED rebuild (feed the builder rows sorted by PK so zone maps
    /// tighten under update scatter) — needs the slot-order-decoupled builder.
    pub fn vacuum_table(&self, table_name: &str) -> Result<(), EngineError> {
        if self.mvcc_read_skips_leader_check() {
            return self.vacuum_table_locked(table_name);
        }
        let _commit_guard = self.commit_state();
        self.vacuum_table_locked(table_name)
    }

    /// The vacuum core for callers ALREADY under the commit lock (the auto-trigger fires inside
    /// the serialized commit arm; a second acquisition would self-deadlock).
    pub(crate) fn vacuum_table_locked(&self, table_name: &str) -> Result<(), EngineError> {
        let run = |engine: &Self| -> Result<(), EngineError> {
            let Some(table) = engine.relational_catalog_table(table_name) else {
                return Ok(()); // no such table: vacuum is a no-op, not an error
            };
            if engine.table_install_elided(table_name) {
                let seq = engine.committed_seq();
                engine.rehydrate_elided_table(
                    &table,
                    seq,
                    &Default::default(),
                    &Default::default(),
                    seq,
                )?;
            }
            let seq = engine.committed_seq();
            let tables: std::collections::BTreeSet<String> =
                std::iter::once(table_name.to_string()).collect();
            engine.invalidate_relational_residency_tables_concurrent(&tables, seq, seq);
            if engine.auto_admit_on_commit_enabled() {
                engine.auto_admit_resident_tables(&tables);
            }
            engine.reset_tombstone_churn(table_name);
            Ok(())
        };
        run(self)
    }

    /// VACUUM #5: enable/disable the churn-triggered AUTO vacuum (default OFF — the A/B lever;
    /// `vacuum_table` stays callable either way).
    pub fn set_auto_vacuum_enabled(&self, on: bool) {
        self.auto_vacuum_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// VACUUM #5 (audit F1): deferred auto-vacuums that failed (telemetry; the trigger re-arms).
    pub fn auto_vacuum_failures(&self) -> u64 {
        self.read_state
            .residency
            .auto_vacuum_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn auto_vacuum_enabled(&self) -> bool {
        self.auto_vacuum_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// VACUUM #5: bump the per-table churn counter by `stamps` tombstones (serialized path only)
    /// and return the new value.
    pub(crate) fn add_tombstone_churn(&self, table: &str, stamps: u64) -> u64 {
        let cur = self.read_state.residency.resident_tombstone_churn.load();
        let mut next = (**cur).clone();
        let counter = next.entry(table.to_string()).or_insert(0);
        *counter += stamps;
        let value = *counter;
        self.read_state
            .residency
            .resident_tombstone_churn
            .store(std::sync::Arc::new(next));
        value
    }

    pub(crate) fn reset_tombstone_churn(&self, table: &str) {
        let cur = self.read_state.residency.resident_tombstone_churn.load();
        if !cur.contains_key(table) {
            return;
        }
        let mut next = (**cur).clone();
        next.remove(table);
        self.read_state
            .residency
            .resident_tombstone_churn
            .store(std::sync::Arc::new(next));
    }

    /// VACUUM #5: the churn threshold — max(1024, table's resident rows / 8). Above it the
    /// auto-trigger rebuilds (dead slots ≥ ~12% bloat scans and keep the PK index dup-declined).
    pub(crate) fn tombstone_churn(&self, table: &str) -> u64 {
        self.read_state
            .residency
            .resident_tombstone_churn
            .load()
            .get(table)
            .copied()
            .unwrap_or(0)
    }

    /// Test lever: force the auto-vacuum threshold (0 = the size-derived default).
    pub fn set_tombstone_churn_threshold_override(&self, threshold: u64) {
        self.tombstone_churn_threshold_override
            .store(threshold, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn tombstone_churn_threshold(&self, table: &str) -> u64 {
        let forced = self
            .tombstone_churn_threshold_override
            .load(std::sync::atomic::Ordering::Relaxed);
        if forced != 0 {
            return forced;
        }
        let rows: usize = self
            .read_state
            .residency
            .shards
            .load()
            .get(table)
            .map(|shards| shards.iter().map(|shard| shard.row_count).sum())
            .unwrap_or(0);
        (rows as u64 / 8).max(1024)
    }

    pub(crate) fn rehydrate_elided_serialized(&self, table_name: &str) -> Result<(), EngineError> {
        let rehydrate = |engine: &Self| -> Result<(), EngineError> {
            if !engine.table_install_elided(table_name) {
                return Ok(()); // another rehydrator won the race
            }
            // PUBLISHED-SNAPSHOT catalog read, NEVER `relational_catalog_table` (audit f80f2350
            // FINDING A, second cycle): that accessor takes the CATALOG LATCH, and this seam is
            // reachable from the DDL apply loop which already HOLDS it (the internal-read flag
            // wrap) — the re-acquire self-deadlocked (gdb-verified: apply_and_publish held
            // commit_mutex + latch, this closure blocked in ddl_catalog()). The published
            // snapshot is layout-correct here: any column-shape-changing DDL rehydrates via the
            // pre-commit execute_text sweep, so the mid-apply seam only fires on the re-elision
            // race, where the layout is unchanged (the in-vacuum catalog-latch lesson, again).
            let Some(table) = engine
                .catalog_snapshot()
                .relational_catalog
                .get(table_name)
                .cloned()
            else {
                return Ok(());
            };
            let seq = engine.committed_seq();
            engine.rehydrate_elided_table(
                &table,
                seq,
                &Default::default(),
                &Default::default(),
                seq,
            )
        };
        if self.mvcc_read_skips_leader_check() {
            // Mid-commit internal read: the commit lock is already held by THIS thread.
            return rehydrate(self);
        }
        let _commit_guard = self.commit_state();
        rehydrate(self)
    }

    /// RETIREMENT A4e: REHYDRATE an elided table — the STICKY DE-ELISION transition. The A4c
    /// gather (at `read_txn`, the last seq whose state the device fully holds) repopulates the
    /// host tuple store + value indexes THROUGH the normal install path (clearing the stale
    /// pre-elision prefix first), then the table LEAVES the elided set. Callers: a DML
    /// prepare/probe whose device resolve declines on an elided table (then the host path
    /// proceeds, always correct), and the commit arm's !handled fallback (then the re-admit
    /// rebuilds from the now-complete store). `extra_rows` carries an in-flight commit's rows
    /// (the mutation the device could NOT absorb — e.g. a NULL append) that the gather at
    /// `read_txn = C-1` cannot see. Returns Err when the gather declines — for an elided table
    /// that is a broken invariant (elision eligibility ⊆ gather eligibility), and failing LOUDLY
    /// beats a silently incomplete store.
    /// U1: resolve elided rows' identities BY int4 KEY against the device gather at `read_txn`
    /// — the rare lane-delete fallback's removal set (the tombstones' (shard, slot) targets are
    /// exactly what a declined/stale device generation can no longer be trusted for; the KEY is
    /// generation-independent). `keys` are `(column_index, value)`; a key with no visible match
    /// at `read_txn` resolves to nothing (its delete was against a row this gather cannot see —
    /// impossible for a wave-located 1-row target, but the resolve is total rather than lossy).
    /// Returns `(row-id removals for the rehydrate, the KEY VALUES that matched a visible row)`
    /// — the matched-key set lets the WAL-first delete fallback set each delete's rows-affected
    /// (1 if its key matched, else 0).
    pub(crate) fn resolve_elided_row_ids_by_int4_key(
        &self,
        table: &RelationalTable,
        read_txn: u64,
        keys: &[(usize, i32)],
    ) -> Result<
        (
            std::collections::BTreeSet<u64>,
            std::collections::HashSet<i32>,
        ),
        EngineError,
    > {
        if keys.is_empty() {
            return Ok(Default::default());
        }
        let gathered = self
            .gather_resident_table_rows_from_device(table, read_txn)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "tombstone key-resolution gather declined for elided table \"{}\"",
                    table.name
                ))
            })?;
        let mut removals = std::collections::BTreeSet::new();
        let mut matched_keys = std::collections::HashSet::new();
        for (row_id, values) in &gathered {
            for &(column, key) in keys {
                if values.get(column) == Some(&SqlValue::Int4(key)) {
                    removals.insert(*row_id);
                    matched_keys.insert(key);
                }
            }
        }
        Ok((removals, matched_keys))
    }

    pub(crate) fn rehydrate_elided_table(
        &self,
        table: &RelationalTable,
        read_txn: u64,
        upserts: &std::collections::BTreeMap<u64, Vec<SqlValue>>,
        removals: &std::collections::BTreeSet<u64>,
        commit_seq: u64,
    ) -> Result<(), EngineError> {
        let gathered = self
            .gather_resident_table_rows_from_device(table, read_txn)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "rehydration gather declined for elided table \"{}\" — device-authoritative \
                     invariant broken (WAL replay is the recovery path)",
                    table.name
                ))
            })?;
        let prefix = relational_key_prefix(&table.name);
        // The gather is the device truth at read_txn; the in-flight commit's delta (the mutation
        // the device could NOT absorb) applies ON TOP: upserts overwrite by identity (an UPDATE
        // keeps its row_id — the new image wins), removals drop (a DELETE the apply skipped).
        let mut merged: std::collections::BTreeMap<u64, Vec<SqlValue>> =
            gathered.into_iter().collect();
        for (row_id, row) in upserts {
            merged.insert(*row_id, row.clone());
        }
        for row_id in removals {
            merged.remove(row_id);
        }
        let install: Vec<(u64, Vec<SqlValue>)> = merged.into_iter().collect();
        let visibility = crate::StorageVisibility {
            read_txn_id: read_txn,
        };
        self.read_state.mvcc.with_table_mut(&table.name, |data| {
            // RECONCILE, not clear+reinsert: the stale pre-elision prefix rows update in place
            // (same key -> tuple_update), gathered-only keys insert, host-only keys (deleted
            // during the elided era) tombstone. The per-table value_index rebuilds wholesale.
            let mut stale: std::collections::BTreeMap<String, u64> = Default::default();
            {
                let mut cursor = data
                    .rows
                    .seq_scan_open(visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                while let Some(tuple) = cursor.next() {
                    if tuple.key.starts_with(&prefix) {
                        stale.insert(tuple.key.clone(), tuple.tuple_id);
                    }
                }
            }
            for (row_id, row) in &install {
                let key = relational_row_key(&table.name, *row_id);
                if let Some(tuple_id) = stale.remove(&key) {
                    data.rows
                        .tuple_update(tuple_id, encode_relational_row(row), commit_seq)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                } else {
                    let tuple_id = self.read_state.mvcc.reserve_tuple_id();
                    data.rows
                        .tuple_insert_reserved_key_with_id(
                            tuple_id,
                            gpu_db_storage::NewTuple {
                                key: key.clone(),
                                value: encode_relational_row(row),
                            },
                            commit_seq,
                        )
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                }
            }
            for (_key, tuple_id) in stale {
                data.rows
                    .tuple_delete(tuple_id, commit_seq)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            }
            data.value_index.clear();
            for (row_id, row) in &install {
                let key = relational_row_key(&table.name, *row_id);
                for (idx, column) in table.columns.iter().enumerate() {
                    let slot_key = crate::resident_storage::ColumnValueKey {
                        column: column.name.clone(),
                        value: relational_index_value(&row[idx]),
                    };
                    let mut slot = data.value_index.get(&slot_key).cloned().unwrap_or_default();
                    slot.push_back(key.clone());
                    data.value_index.insert(slot_key, slot);
                }
            }
            Ok::<(), EngineError>(())
        })?;
        self.set_table_install_elided(&table.name, false);
        Ok(())
    }

    /// RETIREMENT A4c: gather a shard-resident table's VISIBLE rows + identities ENTIRELY FROM
    /// THE DEVICE — the rebuild source that replaces the host store for re-admits and for the
    /// eligibility de-elision transition once A4e stops installing host rows. Per shard: one bulk
    /// DtoH per int4 column + the row_id/deleted_by/created_by regions, then the host-side
    /// SV3b/SV6 visibility filter (`created_by <= read_txn < deleted_by`) — an amortized-once
    /// control-plane readback (the DATA SOURCE is the device generation, not host tuples); the
    /// device-to-device recompaction that avoids the round-trip is the ledgered follow-up.
    /// Returns rows in (shard, slot) order with their identities. `None` = DECLINE (caller must
    /// use the host store): invalid/mismatched shard, null-bearing shard (raw i32 would alias
    /// NULL as 0), non-strictly-Int4 table (Date/Int2 would mistype — the A4a F1 discipline), a
    /// missing identity region, an UNSTAMPED live slot (identity hole), or a device-read failure.
    /// Same born-visible contract as A4a, PLUS snapshot freshness (audit A4c F2): callers must
    /// run on the SERIALIZED commit path with `read_txn` >= every INSERT-appended slot's commit
    /// AND the loaded shard snapshot already reflecting every commit <= `read_txn` (re-admit
    /// callers pass the invalidating commit's seq or newer). Completeness rests on the pinned
    /// `row_count` bounding born-visible slots and on seq monotonicity making any concurrent
    /// commit's mutations (seq > read_txn) correctly invisible to the sequential region reads.
    pub(crate) fn gather_resident_table_rows_from_device(
        &self,
        table: &RelationalTable,
        read_txn: u64,
    ) -> Option<Vec<(u64, Vec<SqlValue>)>> {
        // TYPE-COVERAGE track 2 (stage iii): every FIXED-WIDTH-section type gathers with its
        // catalog-derived variant (i32 via one u32/slot; i64 via two — the 4-mod-8 discipline).
        // TYPE-COVERAGE #14 (bool/numeric/uuid): these also gather here — this is the DEVICE->HOST
        // rehydration a read shape the on-device routes can't serve falls back to (a filtered bool/
        // numeric projection, an ORDER BY on a bool key). Without it an elided table with such a column
        // would hard-error on those shapes. Bool = 1 bit/row bitmap; Numeric/Uuid = the 16-byte b128
        // section (numeric = i128 mantissa LE at the catalog scale; uuid = the raw 16 bytes); Text = the
        // offsets section + bytes blob. Every elision-eligible type now rehydrates (nothing declined by type).
        if table.columns.iter().any(|column| {
            !matches!(
                column.ty,
                gpu_db_sql::SqlType::Int4
                    | gpu_db_sql::SqlType::Date
                    | gpu_db_sql::SqlType::Int2
                    | gpu_db_sql::SqlType::Int8
                    | gpu_db_sql::SqlType::Timestamp
                    | gpu_db_sql::SqlType::Bool
                    | gpu_db_sql::SqlType::Numeric { .. }
                    | gpu_db_sql::SqlType::Uuid
                    | gpu_db_sql::SqlType::Text
            )
        }) {
            return None;
        }
        let shards = self.read_state.residency.shards.load();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let runtime_snapshot = self.router.runtime().snapshot();
        let mut out: Vec<(u64, Vec<SqlValue>)> = Vec::new();
        for shard in table_shards.iter() {
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            let memory_pressure_active = runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            if !shard.is_valid(memory_pressure_active) {
                return None;
            }
            if shard.row_count == 0 {
                continue;
            }
            let descriptor = self.resident_snapshot_for_shard(shard, table);
            // D4 (ADR-013 pre2): buffer + identity + version regions all ride the loaded descriptor
            // — the gather's freshness seam (audit A4c F2) now holds by construction instead of by
            // four separate map loads racing a republish.
            let device_memory = shard.device_memory.clone()?;
            let row_id_region = shard.row_id_region.clone()?;
            let rows = shard.row_count;
            // ADR-006 (NULL coverage): per CATALOG column, the shard's validity bitmap words (ceil(rows/32)
            // u32, read as i32; bit `slot` = 1 => valid, 0 => NULL) if the column carries one, else `None`
            // (no NULLs => all-valid). A sparse map: only null-bearing columns have a layout. This lets the
            // gather (the rehydrate source) materialize a null-bearing shard instead of declining +
            // hard-erroring, so a null-carrying elided table's rehydrate is CORRECT (was: unrecoverable).
            let null_bitmaps: Vec<Option<Vec<i32>>> = {
                let mut per_col = Vec::with_capacity(table.columns.len());
                for column in &table.columns {
                    match shard
                        .resident_device_null_columns
                        .iter()
                        .find(|layout| layout.name == column.name)
                    {
                        Some(layout) => per_col.push(Some(
                            device_memory
                                .read_resident_i32_column(
                                    layout.bitmap_byte_offset,
                                    rows.div_ceil(32),
                                )
                                .ok()?,
                        )),
                        None => per_col.push(None),
                    }
                }
                per_col
            };
            // Bulk DtoH: identities (2 i32 halves LE per slot), then each column's live prefix.
            let id_halves = row_id_region.read_resident_i32_column(0, rows * 2).ok()?;
            let deleted = match &shard.deleted_by_region {
                Some(region) => Some(region.read_resident_i32_column(0, rows * 2).ok()?),
                None => None,
            };
            let created = match &shard.created_by_region {
                Some(region) => Some(region.read_resident_i32_column(0, rows * 2).ok()?),
                None => None,
            };
            // Per-column raw reads: i32 sections one u32/slot, i64 sections two u32/slot (the
            // halves pair below). The enum keeps slot addressing uniform for the typing zip.
            enum GatheredColumn {
                I32(Vec<i32>),
                I64(Vec<i32>),
                // TYPE-COVERAGE #14 (bool): the raw bitmap words (ceil(rows/32) u32, read as i32); bit
                // `slot` = word[slot/32] >> (slot%32) & 1.
                Bool(Vec<i32>),
                // TYPE-COVERAGE #14 (numeric/uuid): the b128 section as FOUR i32 words/slot (16 LE
                // bytes/row). Reassembled to i128 per row: numeric = the mantissa (at the catalog
                // scale); uuid = the raw 16 bytes.
                B128(Vec<i32>),
                // TYPE-COVERAGE #14 (text): the offsets (row+1 u64) + the bytes blob; row `slot` =
                // blob[offsets[slot]..offsets[slot+1]].
                Text(Vec<u64>, Vec<u8>),
            }
            let mut columns: Vec<GatheredColumn> = Vec::with_capacity(table.columns.len());
            for idx in 0..table.columns.len() {
                match table.columns[idx].ty {
                    gpu_db_sql::SqlType::Int8 | gpu_db_sql::SqlType::Timestamp => {
                        let base = crate::relational_model::resident_device_int8_column_offset(
                            &descriptor,
                            table,
                            idx,
                        )
                        .ok()?;
                        columns.push(GatheredColumn::I64(
                            device_memory
                                .read_resident_i32_column(base, rows * 2)
                                .ok()?,
                        ));
                    }
                    gpu_db_sql::SqlType::Bool => {
                        let base = crate::relational_model::resident_device_bool_column_offset(
                            &descriptor,
                            table,
                            idx,
                        )
                        .ok()?;
                        // ceil(rows/32) words cover the live prefix (the shard bitmap is
                        // capacity-strided; the words past `rows` map to headroom -> unread).
                        columns.push(GatheredColumn::Bool(
                            device_memory
                                .read_resident_i32_column(base, rows.div_ceil(32))
                                .ok()?,
                        ));
                    }
                    gpu_db_sql::SqlType::Numeric { .. } | gpu_db_sql::SqlType::Uuid => {
                        let base = crate::relational_model::resident_device_numeric_column_offset(
                            &descriptor,
                            table,
                            idx,
                        )
                        .ok()?;
                        // FOUR i32 words per row (16 bytes), read as the live prefix.
                        columns.push(GatheredColumn::B128(
                            device_memory
                                .read_resident_i32_column(base, rows * 4)
                                .ok()?,
                        ));
                    }
                    gpu_db_sql::SqlType::Text => {
                        let layout = crate::relational_model::resident_device_text_column_layout(
                            &descriptor,
                            table,
                            idx,
                        )
                        .ok()?;
                        // The offsets section is (rows+1) u64; the blob is `bytes_len` raw bytes.
                        let offsets = device_memory
                            .read_resident_u64_column(layout.offsets_byte_offset, rows + 1)
                            .ok()?;
                        let blob = device_memory
                            .read_resident_bytes(
                                layout.bytes_byte_offset,
                                layout.bytes_len as usize,
                            )
                            .ok()?;
                        columns.push(GatheredColumn::Text(offsets, blob));
                    }
                    _ => {
                        let base = crate::relational_model::resident_device_int4_column_offset(
                            &descriptor,
                            table,
                            idx,
                        )
                        .ok()?;
                        columns.push(GatheredColumn::I32(
                            device_memory.read_resident_i32_column(base, rows).ok()?,
                        ));
                    }
                }
            }
            let u64_at = |halves: &[i32], slot: usize| -> u64 {
                (halves[slot * 2] as u32 as u64) | ((halves[slot * 2 + 1] as u32 as u64) << 32)
            };
            for slot in 0..rows {
                let deleted_by = deleted.as_ref().map_or(u64::MAX, |h| u64_at(h, slot));
                let created_by = created.as_ref().map_or(0, |h| u64_at(h, slot));
                if !(created_by <= read_txn && read_txn < deleted_by) {
                    continue; // not visible at this snapshot (tombstoned / future version)
                }
                let row_id = u64_at(&id_halves, slot);
                if row_id == u64::MAX {
                    return None; // an UNSTAMPED live slot: identity hole -> host source
                }
                let row: Vec<SqlValue> = columns
                    .iter()
                    .zip(table.columns.iter())
                    .enumerate()
                    .map(|(col_idx, (column, catalog_column))| {
                        // ADR-006 (NULL coverage): a column with a validity bitmap whose bit for this slot is
                        // 0 is NULL — short-circuit the decode (the raw section holds a don't-care placeholder
                        // 0 / empty, exactly what the payload builder wrote). Columns without a bitmap are
                        // all-valid. Mirrors the on-device M3 read path's NULL semantics.
                        if let Some(words) = &null_bitmaps[col_idx] {
                            if (words[slot / 32] as u32 >> (slot % 32)) & 1 == 0 {
                                return Some(SqlValue::Null);
                            }
                        }
                        match column {
                            GatheredColumn::I32(vals) => {
                                sql_value_from_i32_section(catalog_column.ty, vals[slot])
                            }
                            GatheredColumn::I64(halves) => {
                                let lo = halves[slot * 2] as u32 as u64;
                                let hi = halves[slot * 2 + 1] as u32 as u64;
                                sql_value_from_i64_section(
                                    catalog_column.ty,
                                    (lo | (hi << 32)) as i64,
                                )
                            }
                            GatheredColumn::Bool(words) => {
                                let bit = (words[slot / 32] as u32 >> (slot % 32)) & 1;
                                Some(SqlValue::Bool(bit == 1))
                            }
                            GatheredColumn::B128(words) => {
                                // Reassemble the 16 LE bytes (4 u32 words) for this slot.
                                let mut bytes = [0u8; 16];
                                for w in 0..4 {
                                    bytes[w * 4..w * 4 + 4]
                                        .copy_from_slice(&words[slot * 4 + w].to_le_bytes());
                                }
                                match catalog_column.ty {
                                    gpu_db_sql::SqlType::Numeric { scale, .. } => {
                                        // numeric = i128 mantissa LE, at the column's declared scale
                                        // (byte-identical to the on-device projection's Decimal128::new).
                                        Some(SqlValue::Numeric(gpu_db_sql::Decimal128::new(
                                            i128::from_le_bytes(bytes),
                                            scale,
                                        )))
                                    }
                                    // uuid = the raw 16 bytes (storage wrote them verbatim).
                                    gpu_db_sql::SqlType::Uuid => Some(SqlValue::Uuid(bytes)),
                                    _ => None,
                                }
                            }
                            GatheredColumn::Text(offsets, blob) => {
                                // row `slot` = blob[offsets[slot]..offsets[slot+1]] as UTF-8.
                                let start = offsets[slot] as usize;
                                let end = offsets[slot + 1] as usize;
                                blob.get(start..end)
                                    .and_then(|b| std::str::from_utf8(b).ok())
                                    .map(|s| SqlValue::Text(s.to_string()))
                            }
                        }
                    })
                    .collect::<Option<Vec<SqlValue>>>()?;
                out.push((row_id, row));
            }
        }
        Some(out)
    }

    /// Synthesize a single-store-shaped [`RelationalResidencySnapshot`] DESCRIPTOR for ONE shard
    /// (S10c slice 1). A shard's SoA payload is self-contained (`count_header_byte_offset == 0`,
    /// sized by `shard.row_count`), so a descriptor whose `row_count == shard.row_count` plus the
    /// shard's `resident_device_{int4,text}_columns` makes the SINGLE-store offset helpers address the
    /// shard buffer BYTE-IDENTICALLY — letting the general resident-Expr executor serve one shard
    /// slice when handed it via `ResidentExecSource`. Mirrors the benchmark snapshot constructor (the
    /// per-table install path) field-for-field; the fields the offset helpers DON'T read (generation,
    /// stats, int8/numeric/bool/null columns, refresh cost, admission accounting) take inert defaults.
    /// The identity guard (`schema`/`table` == catalog) and `is_valid()` are satisfied for a valid
    /// shard, so the executor's per-source identity/validity prechecks pass.
    pub(crate) fn resident_snapshot_for_shard(
        &self,
        shard: &RelationalResidentShard,
        table: &RelationalTable,
    ) -> RelationalResidencySnapshot {
        RelationalResidencySnapshot {
            gpu_id: shard.gpu_id,
            schema: shard.schema.clone(),
            table: shard.table.clone(),
            generation: 0,
            // CRITICAL: the shard's own row count sizes the SoA the single-store offset helpers
            // read, so they address THIS shard's buffer (not the whole table). S-d2: an OPEN shard is
            // capacity-padded (headroom for appends), so the column STRIDE is `shard.capacity` while the
            // live row count is `shard.row_count` — exactly the single buffer's capacity/row_count split.
            row_count: shard.row_count,
            capacity: shard.capacity,
            column_count: table.columns.len(),
            resident_bytes: shard.resident_bytes,
            resident_device_int4_columns: shard.resident_device_int4_columns.clone(),
            resident_device_int4_column_stats: Vec::new(),
            // TYPE-COVERAGE track 2 slice 2: the shard's i64 section labels ride the synthesized
            // descriptor so the shared offset helpers address it (layout == single-buffer).
            resident_device_int8_columns: shard.resident_device_int8_columns.clone(),
            // TYPE-COVERAGE #14 (numeric): the shard's b128 (Numeric/Uuid) section labels ride the
            // descriptor (layout == single-buffer, so the shared 16-byte offset helper addresses it).
            resident_device_numeric_columns: shard.resident_device_numeric_columns.clone(),
            // TYPE-COVERAGE #14 (bool): the shard's per-column bool bitmaps (offsets relative to the
            // shard's buffer, which this descriptor addresses) so the executor reads bool on-device.
            resident_device_bool_columns: shard.resident_device_bool_columns.clone(),
            resident_device_text_columns: shard.resident_device_text_columns.clone(),
            // M3-for-shards: carry the shard's own per-column NULL bitmaps (offsets are relative to the
            // shard's buffer, which this descriptor addresses). Empty for the NULL-free majority.
            resident_device_null_columns: shard.resident_device_null_columns.clone(),
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: shard.invalidated_by_txn_id,
            invalidated_at_index: shard.invalidated_at_index,
            invalidated_by_memory_pressure: shard.invalidated_by_memory_pressure,
            memory_pressure_active: shard.memory_pressure_active,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: Vec::new(),
            device_memory_proof: shard.device_memory_proof.clone(),
        }
    }

    /// S10c slice 2a: synthesize the single-store-shaped DESCRIPTOR for the ONE UNIFIED int4-only
    /// buffer recompacted from all of a table's shards. Like [`Self::resident_snapshot_for_shard`]
    /// but sized by the WHOLE table (`row_count == total_row_count`) so the single-store offset helpers
    /// address the unified SoA byte-identically. `int4_columns` is the shards' OWN (uniform)
    /// `resident_device_int4_columns` -- i.e. the list the unified buffer was physically recompacted from,
    /// NOT a catalog re-derivation. Labelling the descriptor with the actual buffer layout keeps the
    /// offset helper's per-read name-check load-bearing (a read of a column whose name does not sit at the
    /// labelled int4 ordinal errors instead of silently returning another column's bytes) -- audit F1.
    /// Text is deferred in this slice, so `resident_device_text_columns` is empty. The
    /// `device_memory_proof` is the unified buffer's freshly-built proof.
    pub(crate) fn resident_snapshot_for_unified(
        &self,
        table: &RelationalTable,
        parts: UnifiedResidentSnapshotParts,
    ) -> RelationalResidencySnapshot {
        let UnifiedResidentSnapshotParts {
            total_row_count,
            gpu_id,
            resident_bytes,
            proof,
            int4_columns,
            int8_columns,
            numeric_columns,
            bool_columns,
            text_columns,
            null_columns,
        } = parts;
        let resident_device_int4_columns = int4_columns;
        RelationalResidencySnapshot {
            gpu_id,
            schema: table.schema.clone(),
            table: table.name.clone(),
            generation: 0,
            row_count: total_row_count,
            capacity: total_row_count,
            column_count: table.columns.len(),
            resident_bytes,
            resident_device_int4_columns,
            resident_device_int4_column_stats: Vec::new(),
            resident_device_int8_columns: int8_columns,
            resident_device_numeric_columns: numeric_columns,
            resident_device_bool_columns: bool_columns,
            resident_device_text_columns: text_columns,
            resident_device_null_columns: null_columns,
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: false,
            memory_pressure_active: false,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: Vec::new(),
            device_memory_proof: Some(proof),
        }
    }

    fn validate_benchmark_resident_chunk_columns(
        table: &RelationalTable,
        int4_columns: &[String],
        int4_stats: &[ResidentDeviceInt4ColumnStats],
        text_columns: &[ResidentDeviceTextColumnLayout],
    ) -> Result<(), ExecuteError> {
        let expected_int4 = table
            .columns
            .iter()
            .filter(|column| column.ty == SqlType::Int4)
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        if int4_columns != expected_int4.as_slice() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk int4 column layout {:?} does not match catalog int4 columns {:?}",
                int4_columns, expected_int4
            ))));
        }
        let actual_int4_stats = int4_stats
            .iter()
            .map(|stats| stats.name.clone())
            .collect::<Vec<_>>();
        if actual_int4_stats != expected_int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk int4 stats layout {:?} does not match catalog int4 columns {:?}",
                actual_int4_stats, expected_int4
            ))));
        }
        if let Some(stats) = int4_stats.iter().find(|stats| stats.min > stats.max) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk int4 stats for column \"{}\" have min greater than max",
                stats.name
            ))));
        }
        let expected_text = table
            .columns
            .iter()
            .filter(|column| column.ty == SqlType::Text)
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let actual_text = text_columns
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        if actual_text != expected_text {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk text column layout {:?} does not match catalog text columns {:?}",
                actual_text, expected_text
            ))));
        }
        Ok(())
    }

    fn visible_relational_row_count(&self, table: &str) -> Result<usize, ExecuteError> {
        let visibility = StorageVisibility {
            read_txn_id: self.committed_seq(),
        };
        let prefix = relational_key_prefix(table);
        let table_rows = self.read_state.mvcc.table_rows(table);
        let mut cursor = table_rows.store().seq_scan_open(visibility)?;
        let mut row_count = 0usize;
        while let Some(tuple) = cursor.next() {
            if tuple.key.starts_with(&prefix) {
                row_count = row_count.checked_add(1).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "visible relational row count overflowed".to_string(),
                    ))
                })?;
            }
        }
        Ok(row_count)
    }

    fn relational_resident_bytes_for_gpu_excluding(&self, gpu_id: u16, table: &str) -> u64 {
        let snapshot_bytes: u64 = self
            .read_state
            .residency
            .snapshots
            .load()
            .iter()
            .filter(|(name, entry)| name.as_str() != table && entry.descriptor.gpu_id == gpu_id)
            .map(|(_name, entry)| {
                entry
                    .descriptor
                    .device_memory_proof
                    .as_ref()
                    .map_or(0, |proof| proof.allocated_bytes)
            })
            .sum();
        let shard_bytes: u64 = self
            .read_state
            .residency
            .shards
            .load()
            .iter()
            .filter(|(name, _shards)| name.as_str() != table)
            .flat_map(|(_name, shards)| shards)
            .filter(|shard| shard.gpu_id == gpu_id)
            .map(|shard| {
                let regions = [
                    shard.deleted_by_region.as_ref(),
                    shard.created_by_region.as_ref(),
                    shard.row_id_region.as_ref(),
                ]
                .into_iter()
                .flatten()
                .map(|region| region.metadata().allocated_bytes)
                .sum::<u64>();
                shard.allocated_bytes.saturating_add(regions)
            })
            .sum();
        let single_indexes = self
            .read_state
            .residency
            .wave_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|(name, _)| name.as_str() != table)
            .filter_map(|(_, index)| index.index_memory.as_ref())
            .filter(|memory| memory.metadata().gpu_id == gpu_id)
            .map(|memory| memory.metadata().allocated_bytes)
            .sum::<u64>();
        let shard_indexes = self
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|((name, _, _), _)| name.as_str() != table)
            .filter_map(|(_, index)| index.device_index.as_ref())
            .filter(|memory| memory.metadata().gpu_id == gpu_id)
            .map(|memory| memory.metadata().allocated_bytes)
            .sum::<u64>();
        snapshot_bytes
            .saturating_add(shard_bytes)
            .saturating_add(single_indexes)
            .saturating_add(shard_indexes)
    }

    /// Actual retained allocation bytes attributable to one table on one GPU. This is the exact
    /// inverse unit used by two-phase admission: candidate selection subtracts these bytes from the
    /// same payload/region/index categories counted by `relational_resident_bytes_for_gpu_excluding`,
    /// so the chosen prefix is known to fit before any descriptor is retired.
    fn relational_resident_table_bytes_for_gpu(&self, table: &str, gpu_id: u16) -> u64 {
        let snapshot_bytes = self
            .read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .filter(|entry| entry.descriptor.gpu_id == gpu_id)
            .and_then(|entry| entry.descriptor.device_memory_proof.as_ref())
            .map_or(0, |proof| proof.allocated_bytes);
        let shard_bytes = self
            .read_state
            .residency
            .shards
            .load()
            .get(table)
            .into_iter()
            .flatten()
            .filter(|shard| shard.gpu_id == gpu_id)
            .map(|shard| {
                let regions = [
                    shard.deleted_by_region.as_ref(),
                    shard.created_by_region.as_ref(),
                    shard.row_id_region.as_ref(),
                ]
                .into_iter()
                .flatten()
                .map(|region| region.metadata().allocated_bytes)
                .sum::<u64>();
                shard.allocated_bytes.saturating_add(regions)
            })
            .sum::<u64>();
        let single_index_bytes = self
            .read_state
            .residency
            .wave_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(table)
            .and_then(|index| index.index_memory.as_ref())
            .filter(|memory| memory.metadata().gpu_id == gpu_id)
            .map_or(0, |memory| memory.metadata().allocated_bytes);
        let shard_index_bytes = self
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|((name, _, _), _)| name == table)
            .filter_map(|(_, index)| index.device_index.as_ref())
            .filter(|memory| memory.metadata().gpu_id == gpu_id)
            .map(|memory| memory.metadata().allocated_bytes)
            .sum::<u64>();
        snapshot_bytes
            .saturating_add(shard_bytes)
            .saturating_add(single_index_bytes)
            .saturating_add(shard_index_bytes)
    }

    pub fn relational_residency_snapshot(
        &self,
        table: &str,
    ) -> Option<RelationalResidencySnapshot> {
        self.read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| {
                let mut snapshot = (*entry.descriptor).clone();
                snapshot.memory_pressure_active = self
                    .router
                    .runtime()
                    .snapshot()
                    .memory_pressured_gpu_ids
                    .contains(&snapshot.gpu_id);
                snapshot
            })
    }

    pub fn relational_retained_snapshot_handle(
        &self,
        table: &str,
    ) -> Option<RelationalRetainedSnapshotHandle> {
        self.read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| {
                let snapshot = &entry.descriptor;
                let memory_pressure_active = self
                    .router
                    .runtime()
                    .snapshot()
                    .memory_pressured_gpu_ids
                    .contains(&snapshot.gpu_id);
                RelationalRetainedSnapshotHandle {
                    schema: snapshot.schema.clone(),
                    table: snapshot.table.clone(),
                    gpu_id: snapshot.gpu_id,
                    generation: snapshot.generation,
                    row_count: snapshot.row_count,
                    column_count: snapshot.column_count,
                    resident_bytes: snapshot.resident_bytes,
                    valid_through_index: snapshot.valid_through_index,
                    valid: snapshot.invalidated_by_txn_id.is_none()
                        && snapshot.invalidated_at_index.is_none()
                        && !snapshot.invalidated_by_memory_pressure
                        && !memory_pressure_active,
                    has_retained_device_memory: self
                        .read_state
                        .residency
                        .device_memory
                        .contains_key(table),
                    resident_device_int4_columns: snapshot.resident_device_int4_columns.clone(),
                    resident_device_text_columns: snapshot.resident_device_text_columns.clone(),
                    resident_device_null_columns: snapshot.resident_device_null_columns.clone(),
                }
            })
    }

    pub fn relational_retained_device_read_view(
        &self,
        table: &str,
    ) -> Option<CudaResidentDeviceMemoryReadView> {
        let handle = self.relational_retained_snapshot_handle(table)?;
        if !handle.valid || !handle.has_retained_device_memory {
            return None;
        }
        self.read_state
            .residency
            .device_memory
            .get(table)
            .map(|device_memory| device_memory.read_view())
    }

    /// Pin the resident snapshot metadata for `table` as an OWNED clone (Stage 3 — blocker #2). The
    /// resident-route consumers used to hold a `&` borrow of the snapshot map across the kernel launch;
    /// now the map is published behind `ArcSwap`, so this loads the published generation and clones the
    /// table's entry out. The clone is owned (no map/guard borrow held across the submission), and the
    /// consumers only read scalar fields + column layouts off it before submitting — so an owned clone
    /// is a drop-in for the former borrow with no lifetime entanglement. Cloning a single snapshot's
    /// metadata once per resident-route statement is negligible against the GPU kernel it precedes.
    /// The lightweight, Arc-shared GPU/catalog descriptor for a resident table. Readers clone the
    /// `Arc` -- a refcount bump, never row data.
    pub(crate) fn relational_residency_snapshot_ref(
        &self,
        table: &str,
    ) -> Option<Arc<RelationalResidencySnapshot>> {
        self.read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| entry.descriptor.clone())
    }

    /// The residency entry from one atomic descriptor-map load.
    pub(crate) fn relational_residency_entry(
        &self,
        table: &str,
    ) -> Option<RelationalResidencyEntry> {
        self.read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .cloned()
    }

    pub fn warm_relational_residency_with_policy(
        &mut self,
        policy: RelationalResidencyWarmupPolicy,
    ) -> RelationalResidencyWarmupReport {
        let policy_sets_gpu = policy.gpu_id.is_some();
        let gpu_id = policy
            .gpu_id
            .unwrap_or_else(|| self.planner.default_gpu_id());
        let policy_sets_budget = policy.budget_bytes.is_some();
        if let Some(budget_bytes) = policy.budget_bytes {
            self.set_relational_residency_budget_bytes(gpu_id, budget_bytes);
        }
        let budget_bytes = self.relational_residency_budget_bytes(gpu_id);
        let requested_tables = if policy.tables.is_empty() {
            self.ddl_catalog_mut()
                .relational_catalog
                .keys()
                .cloned()
                .collect::<Vec<_>>()
        } else {
            policy.tables.clone()
        };
        let mut selected_tables = requested_tables.clone();
        selected_tables.sort();
        selected_tables.dedup();
        if let Some(max_table_count) = policy.max_table_count {
            selected_tables.truncate(max_table_count);
        }

        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let mut entries = Vec::new();
        for table in selected_tables {
            if memory_pressure_active {
                entries.push(RelationalResidencyWarmupEntry {
                    table,
                    action: RelationalResidencyWarmupAction::Skipped,
                    reason: format!("GPU {gpu_id} is memory pressured"),
                    resident_bytes: 0,
                    evicted_tables: Vec::new(),
                    route_decision: None,
                });
                continue;
            }
            if !self
                .ddl_catalog_mut()
                .relational_catalog
                .contains_key(&table)
            {
                entries.push(RelationalResidencyWarmupEntry {
                    table: table.clone(),
                    action: RelationalResidencyWarmupAction::Skipped,
                    reason: "only supported public base tables can be warmed".to_string(),
                    resident_bytes: 0,
                    evicted_tables: Vec::new(),
                    route_decision: None,
                });
                continue;
            }

            let existing = self.relational_residency_snapshot(&table);
            let existing_valid = existing
                .as_ref()
                .is_some_and(|snapshot| snapshot.is_valid());
            let existing_retained = self.read_state.residency.device_memory.contains_key(&table);
            if existing_valid && existing_retained && !policy_sets_budget && !policy_sets_gpu {
                let route_decision = self.warmup_route_readiness_decision(&table);
                entries.push(RelationalResidencyWarmupEntry {
                    table: table.clone(),
                    action: RelationalResidencyWarmupAction::AlreadyResident,
                    reason: "resident snapshot is already valid and retained".to_string(),
                    resident_bytes: existing
                        .as_ref()
                        .map(|snapshot| snapshot.resident_bytes)
                        .unwrap_or(0),
                    evicted_tables: Vec::new(),
                    route_decision,
                });
                continue;
            }
            if existing.is_some() && !policy.refresh_invalidated && !existing_valid {
                entries.push(RelationalResidencyWarmupEntry {
                    table: table.clone(),
                    action: RelationalResidencyWarmupAction::Skipped,
                    reason: "resident snapshot is invalidated and refresh is disabled".to_string(),
                    resident_bytes: existing
                        .as_ref()
                        .map(|snapshot| snapshot.resident_bytes)
                        .unwrap_or(0),
                    evicted_tables: Vec::new(),
                    route_decision: self.warmup_route_readiness_decision(&table),
                });
                continue;
            }

            let refreshing = existing.is_some();
            match self.populate_relational_residency_snapshot_on_gpu(&table, gpu_id) {
                Ok(snapshot) => {
                    let route_decision = self.warmup_route_readiness_decision(&table);
                    entries.push(RelationalResidencyWarmupEntry {
                        table: table.clone(),
                        action: if refreshing {
                            RelationalResidencyWarmupAction::Refreshed
                        } else {
                            RelationalResidencyWarmupAction::Warmed
                        },
                        reason: self
                            .ddl_catalog_mut()
                            .relational_resident_cache
                            .last_decision(&table)
                            .map(|decision| decision.reason.clone())
                            .unwrap_or_else(|| "resident snapshot warmed".to_string()),
                        resident_bytes: snapshot.resident_bytes,
                        evicted_tables: snapshot.evicted_tables_on_admission,
                        route_decision,
                    });
                }
                Err(err) => {
                    entries.push(RelationalResidencyWarmupEntry {
                        table: table.clone(),
                        action: RelationalResidencyWarmupAction::Error,
                        reason: err.to_string(),
                        resident_bytes: 0,
                        evicted_tables: Vec::new(),
                        route_decision: self.warmup_route_readiness_decision(&table),
                    });
                }
            }
        }

        RelationalResidencyWarmupReport {
            gpu_id,
            budget_bytes,
            requested_tables,
            entries,
        }
    }

    pub fn maintain_relational_residency_with_policy(
        &mut self,
        policy: RelationalResidencyMaintenancePolicy,
    ) -> RelationalResidencyMaintenanceReport {
        let warmup = self.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
            gpu_id: policy.gpu_id,
            tables: policy.tables,
            max_table_count: policy.max_table_count,
            budget_bytes: policy.budget_bytes,
            refresh_invalidated: policy.refresh_invalidated,
        });
        let mut warmed_count = 0;
        let mut refreshed_count = 0;
        let mut already_resident_count = 0;
        let mut skipped_count = 0;
        let mut error_count = 0;
        let mut route_ready_tables = Vec::new();
        let mut route_blockers = Vec::new();

        for entry in &warmup.entries {
            match entry.action {
                RelationalResidencyWarmupAction::Warmed => warmed_count += 1,
                RelationalResidencyWarmupAction::Refreshed => refreshed_count += 1,
                RelationalResidencyWarmupAction::AlreadyResident => already_resident_count += 1,
                RelationalResidencyWarmupAction::Skipped => skipped_count += 1,
                RelationalResidencyWarmupAction::Error => error_count += 1,
            }

            match entry.route_decision.as_ref() {
                Some(route) if route.accepted => route_ready_tables.push(entry.table.clone()),
                Some(route) => route_blockers.push(RelationalResidencyMaintenanceBlocker {
                    table: entry.table.clone(),
                    reason: if matches!(
                        entry.action,
                        RelationalResidencyWarmupAction::Skipped
                            | RelationalResidencyWarmupAction::Error
                    ) {
                        entry.reason.clone()
                    } else {
                        route.reason.clone()
                    },
                }),
                None => route_blockers.push(RelationalResidencyMaintenanceBlocker {
                    table: entry.table.clone(),
                    reason: entry.reason.clone(),
                }),
            }
        }

        let entry_count = warmup.entries.len();
        RelationalResidencyMaintenanceReport {
            gpu_id: warmup.gpu_id,
            budget_bytes: warmup.budget_bytes,
            requested_tables: warmup.requested_tables,
            entry_count,
            warmed_count,
            refreshed_count,
            already_resident_count,
            skipped_count,
            error_count,
            route_ready_count: route_ready_tables.len(),
            route_blocked_count: route_blockers.len(),
            route_ready_tables,
            route_blockers,
            entries: warmup.entries,
        }
    }

    fn warmup_route_readiness_decision(
        &mut self,
        table: &str,
    ) -> Option<RelationalResidentRouteDecisionStatus> {
        let sql = format!("SELECT COUNT(*) FROM {table}");
        let Ok(Command::Select(select)) = parse_command(&sql) else {
            return None;
        };
        Some(self.plan_relational_resident_route(&select))
    }

    fn relational_snapshot_cache_state(
        snapshot: &RelationalResidencySnapshot,
        memory_pressure_active: bool,
    ) -> &'static str {
        if memory_pressure_active || snapshot.invalidated_by_memory_pressure {
            "InvalidatedByMemoryPressure"
        } else if snapshot.invalidated_by_txn_id.is_some()
            || snapshot.invalidated_at_index.is_some()
        {
            "Invalidated"
        } else {
            "Valid"
        }
    }

    fn resident_route_reject(
        table: &str,
        reason: impl Into<String>,
        query_shape: impl Into<String>,
    ) -> RelationalResidentRouteDecisionStatus {
        RelationalResidentRouteDecisionStatus {
            table: table.to_string(),
            gpu_id: None,
            snapshot_generation: None,
            shard_count: 0,
            accepted: false,
            reason: reason.into(),
            query_shape: query_shape.into(),
            cache_state: "Absent".to_string(),
            valid: false,
            has_retained_device_memory: false,
            estimated_rows: 0,
            resident_bytes: 0,
            budget_bytes: None,
            refresh_resident_bytes: None,
            h2d_bytes_if_resident: 0,
            h2d_bytes_if_cold: 0,
            d2h_bytes_estimate: 0,
            d2h_rows_estimate: 0,
            last_execution_h2d_bytes: None,
            last_execution_d2h_bytes: None,
            last_execution_kernel_samples: None,
            last_execution_kernel_ms: None,
            last_execution_kernel_event_elapsed_us: None,
            last_execution_rows: None,
            last_execution_wall_micros: None,
            last_execution_device_lookup_micros: None,
            last_execution_match_index_micros: None,
            last_execution_selected_projection_micros: None,
            last_execution_result_materialization_micros: None,
            last_execution_matched_rows: None,
        }
    }

    pub fn plan_relational_resident_route(
        &self,
        select: &Select,
    ) -> RelationalResidentRouteDecisionStatus {
        let decision = self.plan_relational_resident_route_inner(select);
        self.read_state
            .route_telemetry
            .record_route_decision(decision.clone());
        decision
    }

    fn plan_relational_resident_route_inner(
        &self,
        select: &Select,
    ) -> RelationalResidentRouteDecisionStatus {
        // Lock-free read path (Stage 2 — blocker #1): pin the catalog snapshot for the relation-kind
        // check (the subsequent table bind pins its own; both are immutable published snapshots).
        let catalog = self.catalog_snapshot();
        if catalog.relational_views.contains_key(&select.table)
            || catalog
                .relational_materialized_views
                .contains_key(&select.table)
        {
            return Self::resident_route_reject(
                &select.table,
                "resident routing currently supports only public base tables",
                "unsupported_relation_kind",
            );
        }

        let (table, bound, _copin_s) = match self.bind_relational_select_for_execution(select) {
            Ok(bound) => bound,
            Err(err) => {
                return Self::resident_route_reject(
                    &select.table,
                    format!("unsupported select shape: {err}"),
                    "unsupported_select",
                );
            }
        };

        // Stage 3 — blocker #2: pin the published resident shard + snapshot maps for the rest of
        // the planning decision (the shard slice is passed by reference into the sharded-route
        // planner, and the snapshot is read field-by-field below — both must outlive those uses, so the
        // guards are bound here and held to the end of the function).
        let shards_guard = self.read_state.residency.shards.load();
        let snapshots_guard = self.read_state.residency.snapshots.load();

        let query_shape = match resident_route_query_shape(select, &table, &bound) {
            Some(shape) => shape,
            None => {
                if let Some(shards) = shards_guard.get(&table.name) {
                    if let Some(shape) = sharded_resident_route_query_shape(select, &table, &bound)
                    {
                        return self
                            .plan_relational_sharded_resident_route(select, &table, shape, shards);
                    }
                }
                return Self::resident_route_reject(
                    &table.name,
                    "resident routing has no retained-kernel proof for this SELECT shape",
                    "unsupported_select",
                );
            }
        };

        if let Some(shards) = shards_guard.get(&table.name) {
            return self.plan_relational_sharded_resident_route(
                select,
                &table,
                query_shape,
                shards,
            );
        }

        let Some(entry) = snapshots_guard.get(&table.name) else {
            return Self::resident_route_reject(
                &table.name,
                "relation has no resident snapshot",
                query_shape,
            );
        };
        let snapshot = &entry.descriptor;
        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&snapshot.gpu_id);
        let cache_state = Self::relational_snapshot_cache_state(snapshot, memory_pressure_active);
        let valid = snapshot.invalidated_by_txn_id.is_none()
            && snapshot.invalidated_at_index.is_none()
            && !snapshot.invalidated_by_memory_pressure
            && !memory_pressure_active;
        let has_retained_device_memory = self
            .read_state
            .residency
            .device_memory
            .contains_key(&table.name);
        let d2h_bytes_estimate = resident_route_d2h_bytes_estimate(select, &query_shape, snapshot);
        let mut decision = RelationalResidentRouteDecisionStatus {
            table: table.name.clone(),
            gpu_id: Some(snapshot.gpu_id),
            snapshot_generation: Some(snapshot.generation),
            shard_count: 1,
            accepted: false,
            reason: String::new(),
            query_shape,
            cache_state: cache_state.to_string(),
            valid,
            has_retained_device_memory,
            estimated_rows: snapshot.row_count,
            resident_bytes: snapshot.resident_bytes,
            budget_bytes: snapshot.admission_budget_bytes,
            refresh_resident_bytes: snapshot
                .last_refresh_cost
                .as_ref()
                .map(|cost| cost.refreshed_resident_bytes),
            h2d_bytes_if_resident: 0,
            h2d_bytes_if_cold: snapshot.resident_bytes,
            d2h_bytes_estimate,
            d2h_rows_estimate: resident_route_d2h_rows_estimate(select, snapshot.row_count),
            last_execution_h2d_bytes: None,
            last_execution_d2h_bytes: None,
            last_execution_kernel_samples: None,
            last_execution_kernel_ms: None,
            last_execution_kernel_event_elapsed_us: None,
            last_execution_rows: None,
            last_execution_wall_micros: None,
            last_execution_device_lookup_micros: None,
            last_execution_match_index_micros: None,
            last_execution_selected_projection_micros: None,
            last_execution_result_materialization_micros: None,
            last_execution_matched_rows: None,
        };

        if snapshot.schema != table.schema || snapshot.table != table.name {
            decision.reason =
                "resident snapshot no longer matches catalog table identity".to_string();
        } else if !valid {
            decision.reason = format!("resident snapshot is {cache_state}");
        } else if !has_retained_device_memory {
            decision.reason = "resident snapshot has no retained device memory".to_string();
        } else {
            decision.accepted = true;
            decision.reason = "resident route accepted".to_string();
        }
        decision
    }

    fn plan_relational_sharded_resident_route(
        &self,
        select: &Select,
        table: &RelationalTable,
        query_shape: String,
        shards: &[RelationalResidentShard],
    ) -> RelationalResidentRouteDecisionStatus {
        let total_rows = shards.iter().map(|shard| shard.row_count).sum::<usize>();
        let total_resident_bytes = shards.iter().map(|shard| shard.resident_bytes).sum::<u64>();
        let gpu_id = shards.first().map(|shard| shard.gpu_id);
        let sharded_query_shape = if query_shape == "count_all" {
            "sharded_count_all".to_string()
        } else if query_shape == "int4_equality_projection" {
            "sharded_int4_equality_projection".to_string()
        } else if query_shape == "int4_equality_multi_column_projection" {
            "sharded_int4_equality_multi_column_projection".to_string()
        } else if matches!(
            query_shape.as_str(),
            "sharded_int4_equality_sum"
                | "sharded_int4_between_avg"
                | "sharded_int4_filtered_avg"
                | "sharded_int4_filtered_min"
                | "sharded_int4_filtered_max"
        ) {
            query_shape
        } else if matches!(
            query_shape.as_str(),
            "int4_equality_count"
                | "int4_range_count"
                | "int4_between_scalar_aggregate"
                | "int4_projection"
                | "int4_projection_all"
                | "int4_composite_equality_multi_column_projection"
        ) {
            // THE FLIP audit F1: these filtered/range int4 shapes had NO sharded mapping, so the
            // now-default sharded layout demoted them to the CPU host scan (GPU-served pre-flip).
            // The `sharded_` prefix routes them to the sharded BRIDGE (their unprefixed names
            // dispatch to the single-buffer enumerated kernels), whose general executor evaluates
            // the predicate + projection/aggregate on-device over the unified (or zero-copy
            // single-shard) source.
            format!("sharded_{query_shape}")
        } else if query_shape == "int4_filter_group_count" {
            // THE FLIP (burn-in): an OR-of-int4-equalities COUNT fell to the CPU engine on a sharded
            // table (no sharded mapping — the SUM cliff's sibling). The shape keeps its single-buffer
            // name: the dispatch arm routes it to the grouped bridge, whose `src: None` now resolves
            // the sharded unified source inside `execute_resident_expr_select_with_binding`.
            query_shape
        } else if query_shape == "int4_scalar_aggregate" {
            // FLIP slice (measured): an UNFILTERED scalar aggregate (bare SUM/AVG/MIN/MAX) had NO sharded
            // mapping, so it fell through the dispatch to the CPU engine's host scan — MEASURED p50
            // 496,554us vs the bridge-served sharded COUNT's 460us at 524k rows (~1000x, a charter
            // violation in the hot path). The bridge's COUNT-precheck + general run computes scalar
            // aggregates on the unified device buffer, so route it there.
            "sharded_int4_scalar_aggregate".to_string()
        } else if query_shape == "int4_filtered_scalar_aggregate"
            && matches!(select.projection, SelectProjection::Avg { .. })
        {
            "sharded_int4_filtered_avg".to_string()
        } else if query_shape == "int4_filtered_scalar_aggregate"
            && matches!(select.projection, SelectProjection::Min { .. })
        {
            "sharded_int4_filtered_min".to_string()
        } else if query_shape == "int4_filtered_scalar_aggregate"
            && matches!(select.projection, SelectProjection::Max { .. })
        {
            "sharded_int4_filtered_max".to_string()
        } else if query_shape == "int4_filtered_scalar_aggregate" {
            // THE FLIP audit F1 (residue): the filtered aggregates NOT covered by the tuned
            // avg/min/max mappings above (a filtered SUM) route to the sharded bridge's general
            // executor instead of falling to the CPU host scan.
            "sharded_int4_filtered_scalar_aggregate".to_string()
        } else if query_shape == "int4_distinct_projection" {
            "sharded_int4_distinct_projection".to_string()
        } else if query_shape == "int4_filtered_distinct_projection" {
            "sharded_int4_filtered_distinct_projection".to_string()
        } else if query_shape == "int4_grouped_aggregate" {
            "sharded_int4_grouped_aggregate".to_string()
        } else if query_shape == "int4_filtered_grouped_aggregate" {
            "sharded_int4_filtered_grouped_aggregate".to_string()
        } else if query_shape == "int4_ordered_projection" {
            "sharded_int4_ordered_projection".to_string()
        } else {
            query_shape
        };
        let d2h_bytes_estimate = if matches!(
            sharded_query_shape.as_str(),
            "sharded_count_all"
                | "sharded_int4_equality_projection"
                | "sharded_int4_equality_sum"
                | "sharded_int4_between_avg"
                | "sharded_int4_filtered_avg"
                | "sharded_int4_filtered_min"
                | "sharded_int4_filtered_max"
        ) {
            shards
                .len()
                .checked_mul(std::mem::size_of::<u64>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .unwrap_or(u64::MAX)
        } else if sharded_query_shape == "sharded_int4_equality_multi_column_projection" {
            let SelectProjection::Columns(columns) = &select.projection else {
                return Self::resident_route_reject(
                    &table.name,
                    "sharded resident routing has no retained-kernel proof for this SELECT shape",
                    sharded_query_shape,
                );
            };
            u64::try_from(total_rows)
                .unwrap_or(u64::MAX)
                .saturating_mul(
                    u64::try_from(columns.len())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(std::mem::size_of::<i32>() as u64)
                        .saturating_add(std::mem::size_of::<u64>() as u64),
                )
                .saturating_add(
                    u64::try_from(shards.len())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(std::mem::size_of::<u64>() as u64),
                )
        } else {
            0
        };
        let mut decision = RelationalResidentRouteDecisionStatus {
            table: table.name.clone(),
            gpu_id,
            snapshot_generation: None,
            shard_count: shards.len(),
            accepted: false,
            reason: String::new(),
            query_shape: sharded_query_shape,
            cache_state: "Valid".to_string(),
            valid: true,
            has_retained_device_memory: false,
            estimated_rows: total_rows,
            resident_bytes: total_resident_bytes,
            budget_bytes: gpu_id.and_then(|gpu_id| self.relational_residency_budget_bytes(gpu_id)),
            refresh_resident_bytes: None,
            h2d_bytes_if_resident: 0,
            h2d_bytes_if_cold: total_resident_bytes,
            d2h_bytes_estimate,
            d2h_rows_estimate: resident_route_d2h_rows_estimate(select, total_rows),
            last_execution_h2d_bytes: None,
            last_execution_d2h_bytes: None,
            last_execution_kernel_samples: None,
            last_execution_kernel_ms: None,
            last_execution_kernel_event_elapsed_us: None,
            last_execution_rows: None,
            last_execution_wall_micros: None,
            last_execution_device_lookup_micros: None,
            last_execution_match_index_micros: None,
            last_execution_selected_projection_micros: None,
            last_execution_result_materialization_micros: None,
            last_execution_matched_rows: None,
        };

        if !matches!(
            decision.query_shape.as_str(),
            "sharded_count_all"
                | "sharded_int4_scalar_aggregate"
                | "int4_filter_group_count"
                | "sharded_int4_equality_count"
                | "sharded_int4_range_count"
                | "sharded_int4_filtered_scalar_aggregate"
                | "sharded_int4_between_scalar_aggregate"
                | "sharded_int4_projection"
                | "sharded_int4_projection_all"
                | "sharded_int4_composite_equality_multi_column_projection"
                | "sharded_int4_equality_projection"
                | "sharded_int4_equality_multi_column_projection"
                | "sharded_int4_equality_sum"
                | "sharded_int4_between_avg"
                | "sharded_int4_filtered_avg"
                | "sharded_int4_filtered_min"
                | "sharded_int4_filtered_max"
                | "sharded_int4_distinct_projection"
                | "sharded_int4_filtered_distinct_projection"
                | "sharded_int4_grouped_aggregate"
                | "sharded_int4_filtered_grouped_aggregate"
                | "sharded_int4_ordered_projection"
        ) {
            decision.cache_state = "Absent".to_string();
            decision.valid = false;
            decision.reason =
                "sharded resident routing currently supports only unfiltered COUNT(*), same-column int4 equality projection, int4 equality multi-column projection, int4 equality SUM, int4 BETWEEN AVG, int4 filtered AVG, int4 filtered MIN, int4 filtered MAX, int4 [filtered] DISTINCT projection, int4 [filtered] grouped aggregate, and int4 ordered projection"
                    .to_string();
            return decision;
        }
        let mut required_int4_columns = BTreeSet::new();
        // R-ver: the UNFILTERED projection (`sharded_int4_projection_all`) needs exactly its
        // projected columns resident — no filter columns (there is no WHERE by shape definition).
        // It shares the multi-column projection's extraction (the filter loops are no-ops here).
        if decision.query_shape == "sharded_int4_equality_multi_column_projection"
            || decision.query_shape == "sharded_int4_projection_all"
        {
            match &select.projection {
                SelectProjection::Columns(columns) => {
                    for column in columns {
                        required_int4_columns.insert(column.clone());
                    }
                }
                // R-ver: `SELECT * FROM t` needs EVERY column resident (the classifier already
                // proved all columns are int4 for the `sharded_int4_projection_all` shape).
                SelectProjection::All => {
                    for column in &table.columns {
                        required_int4_columns.insert(column.name.clone());
                    }
                }
                _ => {
                    decision.cache_state = "Absent".to_string();
                    decision.valid = false;
                    decision.reason =
                        "sharded resident routing requires projected columns".to_string();
                    return decision;
                }
            }
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if decision.query_shape == "sharded_int4_scalar_aggregate" {
            // FLIP slice: an UNFILTERED scalar aggregate (bare SUM/AVG/MIN/MAX over an int4 column —
            // the single-buffer `int4_scalar_aggregate` shape, remapped). Only the aggregate column is
            // required; there are no filters by shape definition.
            let (SelectProjection::Sum { column }
            | SelectProjection::Avg { column }
            | SelectProjection::Min { column }
            | SelectProjection::Max { column }) = &select.projection
            else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason =
                    "sharded resident routing requires SUM/AVG/MIN/MAX(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
        } else if decision.query_shape == "sharded_int4_equality_sum" {
            let SelectProjection::Sum { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason = "sharded resident routing requires SUM(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if matches!(
            decision.query_shape.as_str(),
            "sharded_int4_between_avg" | "sharded_int4_filtered_avg"
        ) {
            let SelectProjection::Avg { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason = "sharded resident routing requires AVG(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if decision.query_shape == "sharded_int4_filtered_min" {
            let SelectProjection::Min { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason = "sharded resident routing requires MIN(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if decision.query_shape == "sharded_int4_filtered_max" {
            let SelectProjection::Max { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason = "sharded resident routing requires MAX(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if matches!(
            decision.query_shape.as_str(),
            "sharded_int4_grouped_aggregate" | "sharded_int4_filtered_grouped_aggregate"
        ) {
            // Grouped int4 aggregate (S10c slice 2b): the per-shard layout check must cover both the
            // GROUP BY key column AND the aggregated value column, plus every filter column. Mirror the
            // route classifier's group/value extraction (resident_route.rs `resident_route_query_shape`):
            // GroupedCount groups by `column` and counts it; the other grouped projections carry an
            // explicit group/value column pair.
            match &select.projection {
                SelectProjection::GroupedCount { column } => {
                    required_int4_columns.insert(column.clone());
                }
                SelectProjection::GroupedSum {
                    group_column,
                    sum_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(sum_column.clone());
                }
                SelectProjection::GroupedAvg {
                    group_column,
                    avg_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(avg_column.clone());
                }
                SelectProjection::GroupedMin {
                    group_column,
                    min_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(min_column.clone());
                }
                SelectProjection::GroupedMax {
                    group_column,
                    max_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(max_column.clone());
                }
                _ => {
                    decision.cache_state = "Absent".to_string();
                    decision.valid = false;
                    decision.reason =
                        "sharded resident routing requires a grouped int4 aggregate projection"
                            .to_string();
                    return decision;
                }
            }
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if matches!(
            decision.query_shape.as_str(),
            "sharded_int4_distinct_projection"
                | "sharded_int4_filtered_distinct_projection"
                | "sharded_int4_ordered_projection"
        ) {
            // Single-column DISTINCT / ordered int4 projection (S10c slice 2b): the per-shard layout
            // check must cover the single projected/distinct column plus every filter column. The route
            // classifier accepts only a single projected column for these shapes.
            let SelectProjection::Columns(columns) = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason =
                    "sharded resident routing requires a single projected int4 column".to_string();
                return decision;
            };
            for column in columns {
                required_int4_columns.insert(column.clone());
            }
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        }
        if shards.is_empty() {
            decision.cache_state = "Absent".to_string();
            decision.valid = false;
            decision.reason = "relation has no resident shards".to_string();
            return decision;
        }

        let mut has_all_device_memory = true;
        for shard in shards {
            let memory_pressure_active = self
                .router
                .runtime()
                .snapshot()
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            let valid = shard.is_valid(memory_pressure_active);
            decision.valid &= valid;
            if memory_pressure_active || shard.invalidated_by_memory_pressure {
                decision.cache_state = "InvalidatedByMemoryPressure".to_string();
            } else if shard.invalidated_by_txn_id.is_some() || shard.invalidated_at_index.is_some()
            {
                decision.cache_state = "Invalidated".to_string();
            }
            if shard.schema != table.schema || shard.table != table.name {
                decision.reason =
                    "resident shard no longer matches catalog table identity".to_string();
                return decision;
            }
            // D4: the planner's device check reads the loaded descriptor (advisory — execution
            // re-validates from its own snapshot).
            if shard.device_memory.is_none() {
                has_all_device_memory = false;
            }
            // TYPE-COVERAGE #14: a required column must sit in SOME device section the general
            // executor + recompaction gather serve — int4 / int8 / b128 (Numeric/Uuid) / bool bitmap.
            // (Filter columns are still int4 by shape definition; only PROJECTED columns can be
            // i64/b128/bool.) Text required columns are never classified here.
            if !required_int4_columns.is_empty()
                && required_int4_columns.iter().any(|column| {
                    !shard.resident_device_int4_columns.contains(column)
                        && !shard.resident_device_int8_columns.contains(column)
                        && !shard.resident_device_numeric_columns.contains(column)
                        && !shard
                            .resident_device_bool_columns
                            .iter()
                            .any(|b| &b.name == column)
                        && !shard
                            .resident_device_text_columns
                            .iter()
                            .any(|t| &t.name == column)
                })
            {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                // (Message kept as "int4 projection layout" for the p8 route tests; the check now
                // also accepts i64/b128 sections — a truly-missing column still rejects here.)
                decision.reason = format!(
                    "resident shard {} lacks required int4 projection layout",
                    shard.shard_id
                );
                return decision;
            }
        }
        decision.has_retained_device_memory = has_all_device_memory;
        if !decision.valid {
            decision.reason = format!("resident shard set is {}", decision.cache_state);
        } else if !has_all_device_memory {
            decision.reason = "resident shard set has missing retained device memory".to_string();
        } else {
            decision.accepted = true;
            decision.reason = "sharded resident route accepted".to_string();
        }
        decision
    }

    pub(crate) fn relational_residency_status(&self) -> RelationalResidencyStatus {
        // Stage 3 — blocker #2: iterate a pinned snapshot generation (the per-table `last_decision` it
        // joins to still lives on the resident cache and is read via `&self` inside the closure).
        let snapshots_guard = self.read_state.residency.snapshots.load();
        let mut tables = snapshots_guard
            .values()
            .map(|entry| {
                let snapshot = &entry.descriptor;
                let memory_pressure_active = self
                    .router
                    .runtime()
                    .snapshot()
                    .memory_pressured_gpu_ids
                    .contains(&snapshot.gpu_id);
                let last_decision = self
                    .ddl_catalog()
                    .relational_resident_cache
                    .last_decision(&snapshot.table)
                    .cloned();
                let last_decision = last_decision.as_ref();
                let cache_state =
                    Self::relational_snapshot_cache_state(snapshot, memory_pressure_active);
                RelationalResidencyTableStatus {
                    schema: snapshot.schema.clone(),
                    table: snapshot.table.clone(),
                    gpu_id: snapshot.gpu_id,
                    snapshot_generation: snapshot.generation,
                    cache_state: cache_state.to_string(),
                    row_count: snapshot.row_count,
                    column_count: snapshot.column_count,
                    resident_bytes: snapshot.resident_bytes,
                    valid_through_index: snapshot.valid_through_index,
                    valid: snapshot.invalidated_by_txn_id.is_none()
                        && snapshot.invalidated_at_index.is_none()
                        && !snapshot.invalidated_by_memory_pressure
                        && !memory_pressure_active,
                    invalidated_by_txn_id: snapshot.invalidated_by_txn_id,
                    invalidated_at_index: snapshot.invalidated_at_index,
                    invalidated_by_memory_pressure: snapshot.invalidated_by_memory_pressure,
                    memory_pressure_active,
                    admission_budget_bytes: snapshot.admission_budget_bytes,
                    resident_bytes_after_admission: snapshot.resident_bytes_after_admission,
                    evicted_tables_on_admission: snapshot.evicted_tables_on_admission.clone(),
                    last_decision_accepted: last_decision.map(|decision| decision.accepted),
                    last_decision_reason: last_decision.map(|decision| decision.reason.clone()),
                    last_decision_current_bytes_before: last_decision
                        .map(|decision| decision.current_bytes_before),
                    last_decision_current_bytes_after: last_decision
                        .map(|decision| decision.current_bytes_after),
                    device_memory_proof: snapshot.device_memory_proof.clone(),
                }
            })
            .collect::<Vec<_>>();
        tables.sort_by(|left, right| {
            left.gpu_id
                .cmp(&right.gpu_id)
                .then_with(|| left.schema.cmp(&right.schema))
                .then_with(|| left.table.cmp(&right.table))
        });

        let mut resident_bytes_by_gpu = BTreeMap::new();
        for table in &tables {
            *resident_bytes_by_gpu.entry(table.gpu_id).or_insert(0) += table.resident_bytes;
        }

        RelationalResidencyStatus {
            tables,
            latest_route_decisions: self
                .read_state
                .route_telemetry
                .route_decisions()
                .values()
                .cloned()
                .collect(),
            resident_bytes_by_gpu,
            budget_bytes_by_gpu: self
                .ddl_catalog()
                .relational_resident_cache
                .budget_bytes_by_gpu
                .clone(),
        }
    }
}
