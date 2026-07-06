//! E2.5b-2 — N-lane intent commit pipeline (the disruptor-mandate stage graph).
//!
//! The measured wall since E2.3 is the single serial ordered cut (~1.7us/item ≈
//! 625k TPS). This module parallelizes the CUT ITSELF: covered-INSERT intents
//! hash by PK to one of N lanes; each lane is a SINGLE-WRITER pipeline (own
//! ingress queue, own private integer conflict ledger, own `FuaWalLaneSet`
//! WAL lane with its own fence pool), and the only shared-state touch is ONE
//! brief `CommitState` lock per WAVE (global commit-seq block claim via
//! `propose_batch` + timestamp merge). Visibility publishes exclusively at the
//! CROSS-LANE CONTIGUOUS CUT: `committed_seq` advances to S only when every
//! global seq < S is durable in its WAL lane (the lane set's cut) AND applied
//! (this module's `SeqCut`), so a reader can never observe seq N ahead of any
//! seq below N — the fence-pool law, lifted to the engine.
//!
//! V1 scoping (honest, enforced): lanes mode is INTENT-ONLY once the first
//! lane seq is claimed. Classic/DDL writes before lane activation (schema DDL,
//! elision warm-up) run on the classic path and land in the serial WAL;
//! recovery replays the serial log first, then the lane merge (disjoint,
//! contiguous seq ranges). A classic write AFTER activation fails loudly with
//! a clear error rather than risking a mixed-order log. Same-PK intents land
//! in the same lane by construction (hash routing), preserving single-winner
//! 23505 without cross-lane coordination.

// Stage 2: constructed by `engine_lifecycle` behind GPU_DB_INTENT_LANES; the
// pump-facing methods (visible_cut/record_applied/lane_for_pk) go live with the
// stage-3 lane pump loop.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

/// Cross-lane APPLIED cut: lanes report disjoint contiguous global-seq blocks
/// `[start, end)` as their waves finish device apply; `advance` returns the
/// largest S such that every seq < S is applied. Mirrors the interval-merge
/// law of the WAL lane set's durable cut (property-tested there); this is the
/// engine-side twin for the apply stage.
#[derive(Debug, Default)]
pub(crate) struct SeqCut {
    /// The contiguous applied prefix `[base, cut)` is implicit; `pending`
    /// holds applied blocks stranded behind a gap, keyed by start.
    cut: u64,
    pending: BTreeMap<u64, u64>,
}

impl SeqCut {
    pub(crate) fn with_base(base: u64) -> Self {
        Self {
            cut: base,
            pending: BTreeMap::new(),
        }
    }

    /// Record `[start, end)` as applied and advance the contiguous cut as far
    /// as possible. Blocks may arrive in any order; overlaps are a caller bug
    /// (each global seq is claimed by exactly one lane wave) and are rejected
    /// fail-closed via debug_assert + skip.
    pub(crate) fn record(&mut self, start: u64, end: u64) -> u64 {
        debug_assert!(start <= end, "SeqCut block must be a forward range");
        debug_assert!(
            start >= self.cut,
            "SeqCut block below the cut: seq {start} was already applied (double apply?)"
        );
        if start > end || start < self.cut {
            return self.cut;
        }
        self.pending.insert(start, end);
        while let Some((&start, &end)) = self.pending.first_key_value() {
            if start != self.cut {
                break;
            }
            self.pending.remove(&start);
            self.cut = end;
        }
        self.cut
    }

    pub(crate) fn cut(&self) -> u64 {
        self.cut
    }
}

/// Runtime state for lanes mode. Constructed only when
/// `GPU_DB_INTENT_LANES=N>=2` at engine build; `None` keeps every existing
/// path byte-identical.
pub(crate) struct IntentLaneState {
    /// Number of lanes (>= 2).
    pub(crate) lane_count: usize,
    /// Fence lanes per WAL lane (the pool depth) — the auto-subframe signal
    /// compares free slots against this.
    pub(crate) fence_lanes: usize,
    /// DYNAMIC ACTIVE-LANE SUBSET (workload adaptivity slice 3): intents route
    /// by `hash % active_lanes`; quiet lanes stay constructed (their WAL files
    /// and fence pools idle) so a resize is pure ROUTING. Correctness: the
    /// same-PK-same-lane invariant only holds within a routing epoch, so a
    /// resize passes through a DRAIN BARRIER — new submits divert to
    /// `resize_hold` while pumps drain every in-flight intent to settlement
    /// (`outstanding` -> 0); at the barrier every prior commit is covered by
    /// any post-flip snapshot, so the device validate alone catches old
    /// duplicates and lane-ledger continuity across the flip is not needed.
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
    /// load costs little (fence lanes park).
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
    /// The multi-lane durable WAL (per-lane fence pools + cross-lane durable cut).
    pub(crate) wal_lanes: gpu_db_wal::FuaWalLaneSet,
    /// Cross-lane applied cut (device apply completion), advanced under the mutex;
    /// mirrored lock-free for pollers.
    pub(crate) applied: Mutex<SeqCut>,
    pub(crate) applied_mirror: AtomicU64,
    /// Set once the first lane seq block is claimed; classic writes then fail
    /// loudly (v1 intent-only contract — see module docs).
    pub(crate) activated: AtomicBool,
    /// First global seq owned by the lanes (everything below it lives in the
    /// serial WAL from the pre-activation warm-up; recovery replays serial
    /// then lanes over disjoint ranges).
    pub(crate) base_seq: AtomicU64,
    /// Per-lane ingress queues (single-consumer: the lane's pump; multi-producer
    /// submitters). Items route by PK hash, so same-PK contention stays in-lane.
    pub(crate) queues:
        Vec<Mutex<std::collections::VecDeque<crate::engine_dml_concurrent::LaneIntent>>>,
    /// Per-lane PRIVATE conflict ledgers (integer slots only in lanes mode; the
    /// intent-only contract means no classic write can race them).
    pub(crate) ledgers: Vec<Mutex<crate::write_path::RecentCommitsLedger>>,
    /// Per-lane settlement queues: waves whose outcomes are set once the
    /// visible cut covers their end seq (ack = durable ∧ applied ∧ published).
    pub(crate) settle: Vec<Mutex<std::collections::VecDeque<LaneSettle>>>,
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
    /// Host-pass attribution (E2.5b-2 round 2): batch formation drain, ledger
    /// conflict/dedup, fused patch+envelope, and settle-pass time — the
    /// previously invisible ~3.5ms/lane-cycle between the measured stages.
    pub(crate) stat_drain_ns: AtomicU64,
    pub(crate) stat_conflict_ns: AtomicU64,
    pub(crate) stat_patch_ns: AtomicU64,
    pub(crate) stat_settle_ns: AtomicU64,
    /// Sum of publish->settle lag over settled waves (see `LaneSettle::published_at`).
    pub(crate) stat_acklag_ns: AtomicU64,
    pub(crate) stat_settled_waves: AtomicU64,
    pub(crate) stat_validate_ns: AtomicU64,
    pub(crate) stat_claim_ns: AtomicU64,
    pub(crate) stat_append_ns: AtomicU64,
    pub(crate) stat_apply_ns: AtomicU64,
    pub(crate) stat_encode_ns: AtomicU64,
    pub(crate) stat_publish_ns: AtomicU64,
    /// Lock-free commit-timestamp reservation: the highest micros reserved by
    /// any lane wave. Waves CAS-reserve [base, base+k) OUTSIDE the commit lock
    /// (the per-winner map insert under that lock was the measured 8-lane
    /// convoy: ~2us/item of lock hold). Seeded from the classic path's
    /// max_commit_timestamp under the FIRST wave's lock (classic writes are
    /// guarded off after activation, so the two never interleave afterwards).
    pub(crate) ts_reservation: AtomicU64,
    /// Cross-lane device-validate coalescing (v1 of the device-stage
    /// aggregator): lanes push locate requests; one leader drains matching
    /// requests, launches ONE kernel over the concatenated needles, and
    /// scatters counts back. Device cost is fixed-per-launch, so coalescing
    /// K lanes' waves cuts the shared section ~K-fold.
    pub(crate) validate_queue: Mutex<Vec<ValidateRequest>>,
    pub(crate) validate_leader: Mutex<()>,
    pub(crate) stat_coalesced_launches: AtomicU64,
    pub(crate) stat_coalesced_requests: AtomicU64,
    /// LANES-MODE SEQ ORACLE: after activation, waves claim commit-seq blocks
    /// with a plain fetch_add — no commit lock on the pump at all. Seeded ONCE
    /// (under the commit lock) from repl.peek_next_index at activation; safe
    /// because the v1 contract guards classic writes off after activation and
    /// refuses lanes reopen (single-node: the repl log intentionally does not
    /// carry lane payloads — recovery reads the lane logs' explicit seqs; Raft
    /// integration is an E2.5c+ concern, documented). The measured win: the
    /// per-wave commit-lock claim was 77% lock-wait at 8 lanes (2.2ms/wave).
    pub(crate) seq_oracle: AtomicU64,
    /// Apply-side coalescing queue (leader = whoever wins `device_apply_lock`).
    pub(crate) apply_queue: Mutex<Vec<ApplyRequest>>,
    /// NO-REAP APPLY PIPELINE (disruptor staging): the pump pushes its wave to
    /// the apply coalescer, queues its settlement entry immediately, and never
    /// waits — the device apply runs under opportunistic non-blocking leader
    /// passes (`drive_apply_queue_once`) and the applied cut (advanced BY the
    /// leader at completion) gates settlement. Same-slot safety does not
    /// depend on apply completion: the lane LEDGER records winners at claim
    /// time and PK-hash routing pins a PK to one lane, so the next wave's
    /// conflict pass sees the previous wave's slots regardless of device-index
    /// freshness.
    ///
    /// Fail-closed: an apply-leader failure PERMANENTLY holes the applied cut
    /// (its seqs never apply), so it must poison the lanes — later waves would
    /// otherwise wait forever behind the hole. Settle drains everything loudly
    /// when this is set (same shape as the WAL poison drain).
    pub(crate) apply_poisoned: std::sync::atomic::AtomicBool,
    pub(crate) stat_apply_launches: AtomicU64,
    pub(crate) stat_apply_requests: AtomicU64,
    /// Leader BUSY time (drain+merge+launch+scatter, excluding waiter spin) —
    /// the serial-resource test for the coalesced device stages.
    pub(crate) stat_validate_leader_ns: AtomicU64,
    pub(crate) stat_apply_leader_ns: AtomicU64,
    /// E2.5c-2: single-flight guard for the lanes checkpoint (concurrent checkpoints would
    /// interleave sidecar writes and double-truncate).
    pub(crate) checkpoint_lock: Mutex<()>,
}

/// One lane's pending locate request (see `IntentLaneState::validate_queue`).
pub(crate) struct ValidateRequest {
    pub(crate) table: String,
    pub(crate) filter_idx: usize,
    pub(crate) needles: Vec<i32>,
    pub(crate) slot: std::sync::Arc<ValidateSlot>,
}

/// Completion slot: `done` flips after `result` is written (None = declined,
/// callers fall back exactly like a direct-call decline).
pub(crate) struct ValidateSlot {
    pub(crate) done: AtomicBool,
    pub(crate) result: Mutex<Option<Option<Vec<u32>>>>,
}

/// One lane's prepared device-apply request (the apply-side coalescer; same
/// leader pattern as validate). Everything the merged append needs travels in
/// the request — no `CommitWaveItem` re-walk on the apply path.
pub(crate) struct ApplyRequest {
    pub(crate) table: String,
    pub(crate) rows: Vec<Vec<crate::SqlValue>>,
    pub(crate) row_ids: Vec<u64>,
    pub(crate) stamps: Vec<u64>,
    pub(crate) txn_ids: Vec<u64>,
    pub(crate) slot: std::sync::Arc<ApplySlot>,
}

/// Apply completion: `done` flips after the merged append (or its fallback)
/// covered this request's rows.
pub(crate) struct ApplySlot {
    pub(crate) done: AtomicBool,
    /// AUDIT F2: set when the apply leader panicked/poisoned before covering
    /// this request — the waiter must fail its winners, not settle them.
    pub(crate) failed: AtomicBool,
}

/// Min items before a lane wave ships (`GPU_DB_INTENT_LANE_MIN_WAVE`, default 192).
pub(crate) fn intent_lane_min_wave() -> usize {
    std::env::var("GPU_DB_INTENT_LANE_MIN_WAVE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(512)
}

/// Sub-frames per wave publish (`GPU_DB_INTENT_LANE_SUBFRAMES`; 0 = AUTO,
/// the default). Splitting a wave's WAL payload into N contiguous-seq frames
/// multiplies the FUA fence queue depth generated by the SAME traffic — this
/// drive's FUA latency is bimodal (~1.6ms/fence below ~qd8, 0.68ms at qd16+),
/// so the low-depth regime splits to push the drive into fast mode (the ack
/// then waits on N pipelined fast fences instead of one slow one). AUTO
/// splits when the lane's fence pool is mostly idle (low-depth regime) and
/// publishes single frames when the pool is busy.
pub(crate) fn intent_lane_subframes() -> usize {
    std::env::var("GPU_DB_INTENT_LANE_SUBFRAMES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Adaptive ship-target divisor (`GPU_DB_INTENT_LANE_SHIP_DIV`, default 2):
/// a lane ships when its queue reaches outstanding/(div * lanes). Larger
/// divisors ship SMALLER waves sooner — lower formation wait (the oldest
/// item pays the full fill time) and more WAL frames in flight (deeper FUA
/// pipeline), at more per-wave fixed cost.
pub(crate) fn intent_lane_ship_div() -> usize {
    std::env::var("GPU_DB_INTENT_LANE_SHIP_DIV")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(2)
}

/// Age deadline for an under-min wave (`GPU_DB_INTENT_LANE_GROUP_US`, default 200).
pub(crate) fn intent_lane_group_us() -> u64 {
    std::env::var("GPU_DB_INTENT_LANE_GROUP_US")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(300)
}

/// One lane wave awaiting the visible cut: `[first_seq, end_seq)` plus the
/// winner items whose outcome slots settle (Ok) when the cut covers end_seq.
/// NO-REAP PIPELINE: the entry is queued the moment the wave's ApplyRequest is
/// pushed (before the apply completes) — a settled-Ok is still correct because
/// the applied cut only advances when the apply LEADER completes the wave, and
/// the cut gates settlement. `apply_slot.failed` covers the failure path.
pub(crate) struct LaneSettle {
    pub(crate) end_seq: u64,
    pub(crate) winners: Vec<crate::engine_dml_concurrent::LaneIntent>,
    pub(crate) apply_slot: std::sync::Arc<ApplySlot>,
    /// When the wave's WAL frames were published — settle accumulates
    /// publish→settle lag into `stat_acklag_ns` (the fence+cut+settle share
    /// of client ack latency, the low-load SLO's dominant term).
    pub(crate) published_at: std::time::Instant,
}

impl IntentLaneState {
    /// A fresh (never-activated) lane state over `wal_lanes`. Shared by the durable
    /// constructor and the reopen path (which then pre-seeds the activation latch,
    /// `base_seq`, the seq oracle, and the applied cut from the recovered history
    /// before the state is installed — single-threaded, so plain stores suffice).
    pub(crate) fn fresh(
        lane_count: usize,
        fence_lanes: usize,
        wal_lanes: gpu_db_wal::FuaWalLaneSet,
    ) -> Self {
        Self {
            lane_count,
            fence_lanes,
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
            wal_lanes,
            applied: Mutex::new(SeqCut::default()),
            applied_mirror: AtomicU64::new(0),
            activated: AtomicBool::new(false),
            base_seq: AtomicU64::new(0),
            queues: (0..lane_count).map(|_| Default::default()).collect(),
            ledgers: (0..lane_count).map(|_| Default::default()).collect(),
            settle: (0..lane_count).map(|_| Default::default()).collect(),
            device_apply_lock: Mutex::new(()),
            pump_cursor: AtomicU64::new(0),
            pump_guards: (0..lane_count).map(|_| Default::default()).collect(),
            pending_since: (0..lane_count).map(|_| Default::default()).collect(),
            stat_waves: AtomicU64::new(0),
            stat_items: AtomicU64::new(0),
            stat_drain_ns: AtomicU64::new(0),
            stat_conflict_ns: AtomicU64::new(0),
            stat_patch_ns: AtomicU64::new(0),
            stat_settle_ns: AtomicU64::new(0),
            stat_acklag_ns: AtomicU64::new(0),
            stat_settled_waves: AtomicU64::new(0),
            stat_validate_ns: AtomicU64::new(0),
            stat_claim_ns: AtomicU64::new(0),
            stat_append_ns: AtomicU64::new(0),
            stat_apply_ns: AtomicU64::new(0),
            stat_encode_ns: AtomicU64::new(0),
            stat_publish_ns: AtomicU64::new(0),
            ts_reservation: AtomicU64::new(0),
            validate_queue: Mutex::new(Vec::new()),
            validate_leader: Mutex::new(()),
            stat_coalesced_launches: AtomicU64::new(0),
            stat_coalesced_requests: AtomicU64::new(0),
            seq_oracle: AtomicU64::new(0),
            apply_queue: Mutex::new(Vec::new()),
            apply_poisoned: AtomicBool::new(false),
            stat_apply_launches: AtomicU64::new(0),
            stat_apply_requests: AtomicU64::new(0),
            stat_validate_leader_ns: AtomicU64::new(0),
            stat_apply_leader_ns: AtomicU64::new(0),
            checkpoint_lock: Mutex::new(()),
        }
    }

    /// The lanes' visibility frontier in LANE-LOCAL seq space (local seq =
    /// global seq - base_seq; the lane logs tile [0, N) exactly, per the
    /// FuaWalLaneSet contract — the pre-activation range lives in the serial
    /// log and never touches the lanes). Every local seq below this is durable
    /// in its WAL lane AND device-applied.
    pub(crate) fn visible_local_cut(&self) -> u64 {
        self.wal_lanes
            .durable_cut()
            .min(self.applied_mirror.load(Ordering::Acquire))
    }

    /// The GLOBAL visibility frontier the engine may publish for lane-claimed
    /// seqs: base + local cut (0 pre-activation: nothing to publish).
    pub(crate) fn visible_global_cut(&self) -> u64 {
        if !self.activated.load(Ordering::Acquire) {
            return 0;
        }
        self.base_seq.load(Ordering::Acquire) + self.visible_local_cut()
    }

    /// Record a lane wave's applied block and refresh the lock-free mirror.
    pub(crate) fn record_applied(&self, start: u64, end: u64) -> u64 {
        let cut = self
            .applied
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record(start, end);
        self.applied_mirror.fetch_max(cut, Ordering::AcqRel);
        cut
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
    /// Lane diagnostics for benches: (waves, items, validate_ns, claim_ns,
    /// append_ns, apply_ns, durable_cut, applied_cut). None when lanes are off.
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
            lanes.wal_lanes.durable_cut(),
            lanes.applied_mirror.load(Ordering::Acquire),
            self.read_state
                .residency
                .lane_diag_rebuilds
                .load(Ordering::Relaxed),
        ))
    }

    /// Pump host-pass diagnostics: (drain_ns, conflict_ns, patch_ns, settle_ns)
    /// — the formation/ledger/patch+envelope/settle passes between the staged
    /// stats above. None when lanes are off.
    pub fn intent_lane_hostpass_stats(&self) -> Option<(u64, u64, u64, u64)> {
        let lanes = self.intent_lanes.as_ref()?;
        Some((
            lanes.stat_drain_ns.load(Ordering::Relaxed),
            lanes.stat_conflict_ns.load(Ordering::Relaxed),
            lanes.stat_patch_ns.load(Ordering::Relaxed),
            lanes.stat_settle_ns.load(Ordering::Relaxed),
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

    /// WAL fence latency (publish->fence-done): (total ns, fenced frames).
    pub fn intent_lane_fence_stats(&self) -> Option<(u64, u64)> {
        let lanes = self.intent_lanes.as_ref()?;
        Some(lanes.wal_lanes.fence_latency_stats())
    }

    /// Publish->settle lag: (total ns, settled waves). The fence+cut+settle
    /// share of client ack latency.
    pub fn intent_lane_acklag_stats(&self) -> Option<(u64, u64)> {
        let lanes = self.intent_lanes.as_ref()?;
        Some((
            lanes.stat_acklag_ns.load(Ordering::Relaxed),
            lanes.stat_settled_waves.load(Ordering::Relaxed),
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

    /// V1 intent-only contract: once the first lane seq block is claimed, classic
    /// DML/DDL writes are refused fail-loud — a classic record appended to the
    /// serial WAL AFTER lane activation would interleave two ordered logs with
    /// no merge rule (full serial+lanes merge replay is the E2.5c slice).
    /// Pre-activation traffic (schema DDL, elision warm-up) is unaffected.
    pub(crate) fn intent_lanes_write_guard(&self) -> Result<(), crate::EngineError> {
        if let Some(lanes) = &self.intent_lanes {
            if lanes.activated.load(Ordering::Acquire) {
                return Err(crate::EngineError::Durability(
                    "intent lanes are ACTIVE: the engine is intent-only (v1 lanes contract); \
                     classic DML/DDL writes are refused after the first lane commit"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Max items drained per lane wave (`GPU_DB_INTENT_LANE_WAVE_MAX`, default 1024).
pub(crate) fn intent_lane_wave_max() -> usize {
    std::env::var("GPU_DB_INTENT_LANE_WAVE_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1024)
}

pub(crate) fn intent_lane_count() -> usize {
    std::env::var("GPU_DB_INTENT_LANES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 2)
        .unwrap_or(1)
}

/// Fence lanes per WAL lane (`GPU_DB_INTENT_LANE_FENCES`, default 16) — the per-lane FUA
/// fence-pool depth.
pub(crate) fn intent_lane_fences() -> usize {
    std::env::var("GPU_DB_INTENT_LANE_FENCES")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(16)
}

/// Per-lane WAL segment capacity (`GPU_DB_INTENT_LANE_SEGMENT_BYTES`).
///
/// Default stays SMALL (64MiB): every lanes engine prewrites segment_bytes x2 (active +
/// pre-staged) PER LANE at open — a big default quota-bombs test tempdirs. M-TPS deployments
/// should set GPU_DB_INTENT_LANE_SEGMENT_BYTES to 512MiB+: rolls (drain + swap + the
/// pre-stager's prewrite-fsync FLUSH) are the lane tail's dominant stall, and 64MiB rolls
/// every ~15s/lane at 1.6M TPS (same total log bytes either way — only roll cadence changes;
/// measured p99 191ms -> 90.5ms at 512MiB).
pub(crate) fn intent_lane_segment_bytes() -> usize {
    std::env::var("GPU_DB_INTENT_LANE_SEGMENT_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(64 << 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seq_cut_advances_in_order_and_holds_at_gaps() {
        let mut cut = SeqCut::with_base(10);
        assert_eq!(cut.cut(), 10);
        // out-of-order blocks hold behind the gap at 10
        assert_eq!(cut.record(20, 30), 10);
        assert_eq!(cut.record(14, 20), 10);
        // the gap-filling block releases everything contiguous
        assert_eq!(cut.record(10, 14), 30);
        assert_eq!(cut.cut(), 30);
        // empty block at the frontier is a no-op that still reports the cut
        assert_eq!(cut.record(30, 30), 30);
        assert_eq!(cut.record(31, 40), 30);
        assert_eq!(cut.record(30, 31), 40);
    }

    #[test]
    fn seq_cut_random_tilings_match_brute_force() {
        // deterministic pseudo-random tiling: split [0, 4096) into blocks,
        // apply in shuffled order, assert the cut equals the brute-force
        // contiguous frontier after every step.
        let mut seed = 0xDEAD_BEEF_u64;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..50 {
            let mut blocks = Vec::new();
            let mut at = 0_u64;
            while at < 4096 {
                let len = 1 + (rand() % 96);
                let end = (at + len).min(4096);
                blocks.push((at, end));
                at = end;
            }
            // shuffle
            for i in (1..blocks.len()).rev() {
                let j = (rand() % (i as u64 + 1)) as usize;
                blocks.swap(i, j);
            }
            let mut cut = SeqCut::with_base(0);
            let mut applied: Vec<(u64, u64)> = Vec::new();
            for &(start, end) in &blocks {
                applied.push((start, end));
                let got = cut.record(start, end);
                // brute force: sort applied, walk contiguous from 0
                let mut sorted = applied.clone();
                sorted.sort_unstable();
                let mut expect = 0_u64;
                for &(s, e) in &sorted {
                    if s == expect {
                        expect = e;
                    } else if s < expect {
                        unreachable!("tiling produced overlap");
                    } else {
                        break;
                    }
                }
                assert_eq!(got, expect, "cut diverged from brute force");
            }
            assert_eq!(cut.cut(), 4096);
        }
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
