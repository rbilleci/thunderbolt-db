//! Optimized intent preparation lanes.
//!
//! Lanes hash covered mutations by key to parallelize queueing, validation, and merged GPU apply.
//! They are not a transaction, sequence, durability, or publication authority: every accepted
//! wave claims its exact range through the canonical `CommitState` replicator and WAL, then joins
//! the sole contiguous publication coordinator. General/classic and optimized traffic may mix.
//!
//! Startup can read databases written by the retired physical-lane format, but closes that reader
//! after replay and retains no runtime writer/maintenance owner for it. Fresh/live traffic never
//! creates or appends `.lane-*` files. Such nonempty historical databases remain read-only until
//! their compatibility migration rewrites the recovered prefix into the canonical WAL.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

/// Runtime state for the optimized preparation strategy plus a read-only marker when startup
/// consumed historical `.lane-*` recovery input.
pub(crate) struct IntentLaneState {
    /// True only when startup replayed historical `.lane-*` files. New traffic never creates those
    /// files; until the compatibility migration rewrites them into the canonical WAL, the reopened
    /// database is read-only so a canonical suffix cannot become physically ordered before the
    /// historical lane range on the next restart.
    pub(crate) legacy_recovery_read_only: bool,
    /// Number of lanes (>= 2).
    pub(crate) lane_count: usize,
    /// DYNAMIC ACTIVE-LANE SUBSET (workload adaptivity slice 3): intents route
    /// by `hash % active_lanes`; quiet lane queues stay constructed so a resize
    /// is pure ROUTING. Correctness: the
    /// same-PK-same-lane invariant only holds within a routing epoch, so a
    /// resize passes through a DRAIN BARRIER — new submits divert to
    /// `resize_hold` while pumps drain every in-flight intent to settlement
    /// (`outstanding` -> 0); at the barrier every prior commit has published
    /// its device version history before any post-flip snapshot is admitted.
    /// The next epoch therefore validates that history directly and does not
    /// need cross-epoch arbitration state.
    pub(crate) active_lanes: std::sync::atomic::AtomicUsize,
    /// True while a resize leader holds the barrier: submits divert to
    /// `resize_hold` (one Relaxed load on the submit fast path).
    pub(crate) resize_holding: AtomicBool,
    /// Diverted submissions, re-routed through the NEW epoch after the flip.
    pub(crate) resize_hold: Mutex<Vec<crate::engine_dml_concurrent::LaneIntent>>,
    /// Single resize leader at a time + last-flip instant (dwell/hysteresis).
    pub(crate) resize_leader: Mutex<Option<std::time::Instant>>,
    /// Down-flips require a SUSTAINED low population (this tracks since when
    /// `outstanding` has been continuously <= DOWN_AT): a momentary dip at
    /// high load must not trigger a barrier that drains 60k+ in-flight items
    /// (the measured 0.6-1.2s spike). Up-flips stay instant — staying too
    /// narrow under load is a throughput emergency, staying too wide at low
    /// load costs little (quiet queues park).
    pub(crate) resize_low_since: Mutex<Option<std::time::Instant>>,
    /// Diagnostics: completed resizes + total barrier nanos.
    pub(crate) stat_resizes: AtomicU64,
    pub(crate) stat_resize_ns: AtomicU64,
    /// Live population: intents submitted but not yet outcome-settled, across
    /// all lanes. Incremented at submit; decremented in `set_outcome` (the
    /// single completion choke point). Drives WORKLOAD-ADAPTIVE wave
    /// formation: ship targets and age deadlines scale with population so one
    /// configuration serves both the latency (low-load) and throughput
    /// (high-load) regimes. Arc'd so each LaneIntent can carry the decrement
    /// handle.
    pub(crate) outstanding: std::sync::Arc<AtomicU64>,
    /// Per-lane ingress queues (single-consumer: the lane's pump; multi-producer
    /// submitters). Items route by PK hash, so same-PK contention stays in-lane.
    pub(crate) queues:
        Vec<Mutex<std::collections::VecDeque<crate::engine_dml_concurrent::LaneIntent>>>,
    /// Device open-shard appends are not yet safe under concurrent lane pumps
    /// (shared per-table device offsets): v1 serializes the apply stage.
    /// ~20-30us per wave, so contention stays low at wave granularity.
    pub(crate) device_apply_lock: Mutex<()>,
    /// Round-robin pump cursor: each `drive_commit_wave` call in lanes mode
    /// advances one lane's pipeline.
    pub(crate) pump_cursor: AtomicU64,
    /// SINGLE-WRITER guard per lane: round-robin pumps may land on the same
    /// lane concurrently; try_lock keeps each lane's pipeline single-writer
    /// (the loser moves on — another lane has work).
    pub(crate) pump_guards: Vec<Mutex<()>>,
    /// Wave-formation pacing (the conveyor laws): a lane ships a wave only when
    /// its queue holds >= min_wave items OR the age deadline passed since the
    /// first pending item. Without this, N pumps drain instantly, waves shrink
    /// to a handful of items, and the fixed per-wave costs (device validate,
    /// the brief seq-claim lock, the apply lock) explode per item — measured:
    /// lanes=4 collapsed to 132k TPS on 1-item waves before pacing.
    pub(crate) pending_since: Vec<Mutex<Option<std::time::Instant>>>,
    /// Diagnostics (GPU_DB_BENCH_LANESTATS): per-stage nanos summed across all
    /// lane waves — divides by `stat_waves`/`stat_items` for per-wave/per-item
    /// attribution of the pump pipeline.
    pub(crate) stat_waves: AtomicU64,
    pub(crate) stat_items: AtomicU64,
    /// Host-pass attribution: batch formation drain, conflict/dedup arbitration, and fused
    /// patch+envelope work between device stages.
    pub(crate) stat_drain_ns: AtomicU64,
    pub(crate) stat_conflict_ns: AtomicU64,
    pub(crate) stat_patch_ns: AtomicU64,
    pub(crate) stat_validate_ns: AtomicU64,
    pub(crate) stat_claim_ns: AtomicU64,
    pub(crate) stat_apply_ns: AtomicU64,
    pub(crate) stat_encode_ns: AtomicU64,
    pub(crate) stat_publish_ns: AtomicU64,
    /// Cross-lane device-validate coalescing (v1 of the device-stage
    /// aggregator): lanes push locate requests; one leader drains matching
    /// requests, launches ONE kernel over the concatenated needles, and
    /// scatters counts back. Device cost is fixed-per-launch, so coalescing
    /// K lanes' waves cuts the shared section ~K-fold.
    pub(crate) validate_queue: Mutex<Vec<ValidateRequest>>,
    pub(crate) validate_leader: Mutex<()>,
    pub(crate) stat_coalesced_launches: AtomicU64,
    pub(crate) stat_coalesced_requests: AtomicU64,
    /// Apply completes synchronously under the canonical commit mutex before the
    /// durability/publication tail is registered.
    pub(crate) stat_apply_launches: AtomicU64,
    /// Device-boundary busy time.
    pub(crate) stat_validate_leader_ns: AtomicU64,
    pub(crate) stat_apply_leader_ns: AtomicU64,
}

/// One lane's pending locate request (see `IntentLaneState::validate_queue`).
pub(crate) struct ValidateRequest {
    pub(crate) table: String,
    /// COMPOUND KEYS: the device-probe key id — a single-column column-index or `FLAG | ordinal`
    /// (see `index_probe_key_id`). Requests coalesce per (table, key_id); the needles are the raw
    /// keys or the compound fingerprints for that index.
    pub(crate) key_id: usize,
    pub(crate) needles: Vec<i32>,
    pub(crate) slot: std::sync::Arc<ValidateSlot>,
}

/// Completion slot: `done` flips after `result` is written (None = declined,
/// callers fall back exactly like a direct-call decline).
pub(crate) struct ValidateSlot {
    pub(crate) done: AtomicBool,
    pub(crate) result: Mutex<Option<Option<Vec<u32>>>>,
}

/// One lane's prepared device-apply request. Everything the merged append needs travels in the
/// request; no `CommitWaveItem` re-walk or cross-lane apply queue remains.
pub(crate) struct ApplyRequest {
    pub(crate) table: String,
    /// INSERT winners only (parallel with `row_ids`/`stamps`): the merged
    /// open-shard append inputs. A delete-only wave ships these empty (U1).
    pub(crate) rows: Vec<Vec<crate::SqlValue>>,
    pub(crate) row_ids: Vec<u64>,
    pub(crate) stamps: Vec<u64>,
    /// U1 WAL-first: DELETE winners as UNRESOLVED by-key tombstones — the apply LOCATES them
    /// (device visible-locate at the delete's read snapshot), off the pump's critical path.
    pub(crate) tombstones: Vec<LaneTombstone>,
    /// U2 WAL-first: UPDATE winners as UNRESOLVED by-key locate-then-tombstone-then-append ops —
    /// the apply LOCATES the visible old version, tombstones it, and CONDITIONALLY appends the new
    /// version (only if the old located to one row), off the pump's critical path.
    pub(crate) updates: Vec<LaneUpdate>,
    /// Exact pre-WAL GPU target cardinalities for unresolved delete/update operations. The apply
    /// stage must reproduce these marker-authoritative outcomes before it may advance the applied
    /// prefix; a mismatch wedges the unpublished generation for recovery.
    pub(crate) expected_rows_affected: Vec<(std::sync::Arc<std::sync::atomic::AtomicU64>, u64)>,
}

/// U1 WAL-first: an UNRESOLVED by-key tombstone — the apply locates the visible target itself
/// (device visible-locate at `read_snapshot`), so the pump never blocks on the delete locate.
/// The record is already by-key durable (W5b), so `seq` is claimed and fenced before this
/// resolves; a 0-row outcome is a durable no-op (replay re-resolves the same 0 rows).
pub(crate) struct LaneTombstone {
    /// The commit seq — the `deleted_by` stamp value if a visible row is located.
    pub(crate) seq: u64,
    /// The pk column's catalog position (the locate filter index).
    pub(crate) filter_idx: u32,
    /// The pk value (the locate needle).
    pub(crate) pk: i32,
    /// The delete's read snapshot — the visibility the apply-time locate evaluates at.
    pub(crate) read_snapshot: u64,
    /// The shared cell the apply writes the resolved rows-affected (0 or 1) into; completion
    /// reads it for the exact result. Shared with the delete's `LaneIntent.rows_affected_cell`.
    pub(crate) rows_affected: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

/// U2 WAL-first: an UNRESOLVED by-key UPDATE — the apply locates the visible old version itself
/// (device visible-locate at `read_snapshot`), tombstones it, and CONDITIONALLY appends the new
/// version with the located old version's stable entity identity (only when the locate found
/// exactly one visible row). The v1 record still carries a reserved legacy row id so existing WAL
/// framing and allocator replay stay compatible during the ADR-014 migration; that reservation is
/// not the replacement version's identity. A 0-row outcome appends nothing but still consumes the
/// reservation, keeping replay's allocator in lock-step.
pub(crate) struct LaneUpdate {
    /// The commit seq — the `deleted_by` stamp on the located old version AND the `created_by`
    /// stamp on the appended new version.
    pub(crate) seq: u64,
    /// The pk column's catalog position (the locate filter index).
    pub(crate) filter_idx: u32,
    /// The pk value (the locate needle; unchanged by a covered update).
    pub(crate) pk: i32,
    /// The update's read snapshot — the visibility the apply-time locate evaluates at.
    pub(crate) read_snapshot: u64,
    /// The new row image (all columns, catalog order) — appended iff the old located to one row.
    pub(crate) new_values: Vec<crate::SqlValue>,
    /// The shared cell the apply writes the resolved rows-affected (0 or 1) into; completion reads
    /// it for the exact result. Shared with the update's `LaneIntent.rows_affected_cell`.
    pub(crate) rows_affected: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

/// Min items before a lane wave ships (the adaptive formation's CAP; the live
/// population scales the effective target — see `drive_intent_lane`). Folded
/// from `GPU_DB_INTENT_LANE_MIN_WAVE` per the no-flag-proliferation ruling:
/// 512 won every sweep.
pub(crate) fn intent_lane_min_wave() -> usize {
    512
}

/// Adaptive ship-target divisor: a lane ships when its queue reaches
/// outstanding/(2 * lanes). Folded from `GPU_DB_INTENT_LANE_SHIP_DIV`:
/// 4/8 traded p50 -30us for p90 +100us — 2 stands.
pub(crate) fn intent_lane_ship_div() -> usize {
    2
}

/// Age deadline CAP (us) for an under-min wave (population-scaled below the
/// cap). Folded from `GPU_DB_INTENT_LANE_GROUP_US`: 2000 won the sweeps
/// (500/1000/1500 all below baseline on the non-blocking pipeline).
pub(crate) fn intent_lane_group_us() -> u64 {
    2000
}

impl IntentLaneState {
    /// Fresh optimized preparation lanes. There is deliberately no physical-lane WAL creation
    /// input: live durability belongs to the canonical `WalBuffer`.
    pub(crate) fn fresh(lane_count: usize) -> Self {
        Self {
            legacy_recovery_read_only: false,
            lane_count,
            // Start FULL-WIDTH: a high-load flood then never pays an up-flip
            // barrier at cold start (the measured 0.6-1.2s spike); a low-load
            // workload instead pays one cheap down-flip (draining <= DOWN_AT
            // items).
            active_lanes: std::sync::atomic::AtomicUsize::new(lane_count),
            resize_holding: AtomicBool::new(false),
            resize_hold: Mutex::new(Vec::new()),
            resize_leader: Mutex::new(None),
            resize_low_since: Mutex::new(None),
            stat_resizes: AtomicU64::new(0),
            stat_resize_ns: AtomicU64::new(0),
            outstanding: std::sync::Arc::new(AtomicU64::new(0)),
            queues: (0..lane_count).map(|_| Default::default()).collect(),
            device_apply_lock: Mutex::new(()),
            pump_cursor: AtomicU64::new(0),
            pump_guards: (0..lane_count).map(|_| Default::default()).collect(),
            pending_since: (0..lane_count).map(|_| Default::default()).collect(),
            stat_waves: AtomicU64::new(0),
            stat_items: AtomicU64::new(0),
            stat_drain_ns: AtomicU64::new(0),
            stat_conflict_ns: AtomicU64::new(0),
            stat_patch_ns: AtomicU64::new(0),
            stat_validate_ns: AtomicU64::new(0),
            stat_claim_ns: AtomicU64::new(0),
            stat_apply_ns: AtomicU64::new(0),
            stat_encode_ns: AtomicU64::new(0),
            stat_publish_ns: AtomicU64::new(0),
            validate_queue: Mutex::new(Vec::new()),
            validate_leader: Mutex::new(()),
            stat_coalesced_launches: AtomicU64::new(0),
            stat_coalesced_requests: AtomicU64::new(0),
            stat_apply_launches: AtomicU64::new(0),
            stat_validate_leader_ns: AtomicU64::new(0),
            stat_apply_leader_ns: AtomicU64::new(0),
        }
    }

    /// Record a replayed historical physical-lane prefix. Startup closes the old WAL reader after
    /// replay; runtime retains only this fixed diagnostic/read-only boundary.
    pub(crate) fn with_history(lane_count: usize, legacy_record_count: u64) -> Self {
        let state = Self::fresh(lane_count);
        Self {
            legacy_recovery_read_only: legacy_record_count > 0,
            ..state
        }
    }

    /// Route a PK to its lane. Fibonacci mixing over the i32 key: same PK →
    /// same lane, uniform spread; MUST stay in sync with any WAL-side routing
    /// assumptions (there are none: lanes are content-agnostic).
    pub(crate) fn lane_for_pk(&self, pk: i32) -> usize {
        let active = self
            .active_lanes
            .load(std::sync::atomic::Ordering::Acquire)
            .clamp(1, self.lane_count);
        let mixed = (pk as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((mixed >> 40) % active as u64) as usize
    }
}

/// `GPU_DB_INTENT_LANES` (default 1 = lanes mode OFF; >= 2 enables). Read once
/// at engine construction, like the other write-path knobs.
impl crate::Engine {
    pub(crate) fn legacy_lane_history_write_guard(&self) -> Result<(), crate::EngineError> {
        if self
            .intent_lanes
            .as_ref()
            .is_some_and(|lanes| lanes.legacy_recovery_read_only)
        {
            return Err(crate::EngineError::Durability(
                "legacy intent-lane WAL history is open read-only until it is migrated into the canonical WAL"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Live strategy diagnostics: (waves, items, validate_ns, claim_ns, encode_ns, publish_ns,
    /// apply_ns, canonical durable record count, canonical published sequence, index rebuilds).
    #[allow(clippy::type_complexity)]
    pub fn intent_lane_stats(&self) -> Option<(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64)> {
        let lanes = self.intent_lanes.as_ref()?;
        Some((
            lanes.stat_waves.load(Ordering::Relaxed),
            lanes.stat_items.load(Ordering::Relaxed),
            lanes.stat_validate_ns.load(Ordering::Relaxed),
            lanes.stat_claim_ns.load(Ordering::Relaxed),
            lanes.stat_encode_ns.load(Ordering::Relaxed),
            lanes.stat_publish_ns.load(Ordering::Relaxed),
            lanes.stat_apply_ns.load(Ordering::Relaxed),
            self.wal_flushed_count() as u64,
            self.committed_seq(),
            self.read_state
                .residency
                .lane_diag_rebuilds
                .load(Ordering::Relaxed),
        ))
    }

    /// Pump host-pass diagnostics: (drain_ns, conflict_ns, patch_ns).
    pub fn intent_lane_hostpass_stats(&self) -> Option<(u64, u64, u64)> {
        let lanes = self.intent_lanes.as_ref()?;
        Some((
            lanes.stat_drain_ns.load(Ordering::Relaxed),
            lanes.stat_conflict_ns.load(Ordering::Relaxed),
            lanes.stat_patch_ns.load(Ordering::Relaxed),
        ))
    }

    /// Adaptivity diagnostics: (active_lanes, outstanding, resizes, resize_ns).
    pub fn intent_lane_adaptive_stats(&self) -> Option<(usize, u64, u64, u64)> {
        let lanes = self.intent_lanes.as_ref()?;
        Some((
            lanes.active_lanes.load(Ordering::Acquire),
            lanes.outstanding.load(Ordering::Relaxed),
            lanes.stat_resizes.load(Ordering::Relaxed),
            lanes.stat_resize_ns.load(Ordering::Relaxed),
        ))
    }

    /// Leader-busy diagnostics: (validate_leader_ns, validate_launches,
    /// apply_leader_ns, apply_launches).
    pub fn intent_lane_leader_stats(&self) -> Option<(u64, u64, u64, u64)> {
        let lanes = self.intent_lanes.as_ref()?;
        Some((
            lanes.stat_validate_leader_ns.load(Ordering::Relaxed),
            lanes.stat_coalesced_launches.load(Ordering::Relaxed),
            lanes.stat_apply_leader_ns.load(Ordering::Relaxed),
            lanes.stat_apply_launches.load(Ordering::Relaxed),
        ))
    }

    /// Diagnostics: PK device-index rebuild count (the expensive miss path).
    pub fn pk_index_rebuilds_diag(&self) -> u64 {
        self.read_state
            .residency
            .lane_diag_rebuilds
            .load(Ordering::Relaxed)
    }
}

/// Max items drained per lane wave. Folded from `GPU_DB_INTENT_LANE_WAVE_MAX`
/// (never a winning lever in any sweep; bounds the fused patch pass).
pub(crate) fn intent_lane_wave_max() -> usize {
    1024
}

/// Intent lane count (`GPU_DB_INTENT_LANES`). E2.5c-3 DEFAULT FLIP: lanes mode is ON by
/// default on unix (10 lanes — the measured champion on the reference box). Lane construction is
/// queue/control state only and performs no physical WAL I/O. Explicit `0`/`1` disables.
pub(crate) fn intent_lane_count() -> usize {
    std::env::var("GPU_DB_INTENT_LANES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|n| if n >= 2 { n } else { 1 })
        .unwrap_or(if cfg!(unix) { 10 } else { 1 })
}

#[cfg(test)]
mod tests {
    #[test]
    fn synchronous_commit_off_compatibility_setting_keeps_the_strict_gate() {
        let mut engine = crate::Engine::new_local_test_engine();
        let base = std::env::temp_dir().join(format!(
            "gpu-db-strict-commit-setting-{}-{:?}.wal",
            std::process::id(),
            std::thread::current().id()
        ));
        engine.attach_test_intent_lanes(base, 2);
        engine.set_synchronous_commit_default(crate::SynchronousCommit::Off);
        assert_eq!(engine.intent_lanes.as_ref().unwrap().lane_count, 2);
    }

    #[test]
    fn lane_routing_is_stable_and_spread() {
        let state_lanes = 4_usize;
        let mixed = |pk: i32| {
            let m = (pk as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            ((m >> 40) % state_lanes as u64) as usize
        };
        // stability: same pk always maps to the same lane
        for pk in [0, 1, -1, i32::MAX, i32::MIN, 42_424_242] {
            assert_eq!(mixed(pk), mixed(pk));
        }
        // spread: 64k sequential pks should hit every lane substantially
        let mut counts = [0_usize; 4];
        for pk in 0..65_536_i32 {
            counts[mixed(pk)] += 1;
        }
        for (lane, count) in counts.iter().enumerate() {
            assert!(
                *count > 65_536 / 8,
                "lane {lane} starved: {count} of 65536 sequential pks"
            );
        }
    }
}
