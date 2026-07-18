//! E2.5b-2 — N-lane intent commit pipeline (the disruptor-mandate stage graph).
//!
//! The measured wall since E2.3 is the single serial ordered cut (~1.7us/item ≈
//! 625k TPS). This module parallelizes the CUT ITSELF: covered-INSERT intents
//! hash by PK to one of N lanes; each lane is a SINGLE-WRITER pipeline (own
//! ingress queue, bounded same-wave/unpublished-slot arbitration, own `FuaWalLaneSet`
//! WAL lane with its own fence pool), and the only shared-state touch is ONE
//! brief `CommitState` lock per WAVE (global commit-seq block claim via
//! `propose_batch` + timestamp merge). Visibility publishes exclusively at the
//! CROSS-LANE CONTIGUOUS CUT: when the exclusive next-slot frontier is S,
//! `committed_seq` advances to inclusive sequence S-1 only when every global
//! sequence below S is durable in its WAL lane (the lane set's cut) AND
//! applied (this module's `SeqCut`). A reader can therefore never observe
//! sequence N ahead of any sequence below N — the fence-pool law, lifted to
//! the engine.
//!
//! V1 scoping (honest, enforced): lanes mode is INTENT-ONLY once the first
//! nonempty lane wave activates. Classic/DDL writes before lane activation (schema DDL,
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

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

fn inclusive_global_boundary(base: u64, exclusive_local_cut: u64) -> u64 {
    base.saturating_add(exclusive_local_cut).saturating_sub(1)
}

#[cfg(test)]
type ClassicPrelockHook = (
    usize,
    std::sync::Arc<std::sync::Barrier>,
    std::sync::Arc<std::sync::Barrier>,
);

#[cfg(test)]
fn classic_prelock_hook() -> &'static Mutex<Option<ClassicPrelockHook>> {
    static HOOK: std::sync::OnceLock<Mutex<Option<ClassicPrelockHook>>> =
        std::sync::OnceLock::new();
    HOOK.get_or_init(|| Mutex::new(None))
}

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

/// Convert a lane-local EXCLUSIVE next-slot prefix to the GLOBAL INCLUSIVE
/// sequence consumed by the current MVCC watermark. `u64::MAX` is reserved as
/// the live `deleted_by` infinity sentinel: it may be the exclusive next value
/// covering the last valid commit (`u64::MAX - 1`), but is never itself a
/// commit sequence.
fn inclusive_seq_from_exclusive_prefix(
    base_seq: u64,
    local_next: u64,
) -> Result<Option<u64>, &'static str> {
    if local_next == 0 {
        return Ok(None);
    }
    let visible_next = base_seq
        .checked_add(local_next)
        .ok_or("intent-lane visible-next frontier overflow")?;
    Ok(Some(visible_next - 1))
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
    /// (`outstanding` -> 0); at the barrier every prior commit has published
    /// its device version history before any post-flip snapshot is admitted.
    /// The next epoch therefore validates that history directly and does not
    /// need an unpublished-slot bridge carried across the flip.
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
    /// Engine-wide default commit mode (the PostgreSQL `synchronous_commit`
    /// server default): true = strict durable acks. Seeded from
    /// `GPU_DB_SYNCHRONOUS_COMMIT`; per-statement override via
    /// `Engine::submit_covered_insert_intent_with_commit`.
    pub(crate) synchronous_commit_default: AtomicBool,
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
    /// The multi-lane durable WAL (per-lane fence pools + cross-lane durable cut). LAZY for a
    /// fresh engine (E2.5c-3 default flip: lanes-ON-by-default must not prewrite N x 2 lane
    /// segments for every durable engine that never takes the intent path — the backing is
    /// created on the first lane wave via [`Self::wal`]); PRE-POPULATED by the reopen path
    /// (a lanes database's files already exist and the recovered cuts must install).
    wal_lanes: std::sync::OnceLock<gpu_db_wal::FuaWalLaneSet>,
    /// Serializes the one-time lazy creation (OnceLock has no fallible get_or_init on stable).
    wal_lanes_init: Mutex<()>,
    /// Lane WAL base path + per-lane segment capacity for the lazy create.
    lane_base: std::path::PathBuf,
    lane_segment_bytes: usize,
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
    /// Canonical database lineage captured at the serialized activation fence. Lane pumps run
    /// off-lock and copy this immutable identity into every durable envelope.
    pub(crate) canonical_identity: OnceLock<gpu_db_wal::CanonicalIdentity>,
    /// Catalog binding frozen at the serial-to-lane activation fence. Activated lanes are
    /// intent-only, so no catalog transition may change this boundary afterward.
    pub(crate) canonical_catalog_epoch: AtomicU64,
    pub(crate) canonical_catalog_digest: OnceLock<gpu_db_wal::CanonicalDigest>,
    /// Stable transaction-id claims for the live lane path. Pending claims deduplicate concurrent
    /// submissions; terminal entries are rebuilt from canonical WAL on reopen and retained for
    /// exact same-id/same-digest retry resolution.
    pub(crate) transaction_claims: std::sync::Arc<Mutex<HashMap<u64, LaneTransactionClaim>>>,
    /// Per-lane ingress queues (single-consumer: the lane's pump; multi-producer
    /// submitters). Items route by PK hash, so same-PK contention stays in-lane.
    pub(crate) queues:
        Vec<Mutex<std::collections::VecDeque<crate::engine_dml_concurrent::LaneIntent>>>,
    /// Keys whose WAL block is claimed but whose device apply has not completed. This is bounded
    /// by the apply pipeline, not retained commit history: same-PK routing makes it the exact
    /// arbitration bridge until the resident version stamps become authoritative.
    pub(crate) inflight_slots:
        Vec<Mutex<std::collections::HashSet<crate::write_path::IntUniqueSlotKey>>>,
    /// Per-lane settlement queues: waves whose outcomes are set once the
    /// visible cut covers their end seq (ack = durable ∧ applied ∧ published).
    pub(crate) settle: Vec<Mutex<std::collections::VecDeque<LaneSettle>>>,
    /// Device open-shard appends are not yet safe under concurrent lane pumps
    /// (shared per-table device offsets): v1 serializes the apply stage.
    /// ~20-30us per wave, so contention stays low at wave granularity.
    pub(crate) device_apply_lock: Mutex<()>,
    /// Seqlock-style witness for lane device publication. The apply leader increments this after a
    /// merged apply has completed (or unwound after touching device state) and before removing any
    /// in-flight slot. A validator samples it before its device probe and again while holding the
    /// target lane's `inflight_slots` lock; a changed value makes the whole verdict retry. This
    /// closes probe-vs-bridge-removal TOCTOU without serializing GPU validation behind apply.
    pub(crate) device_publication_epoch: AtomicU64,
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
    /// Host-pass attribution (E2.5b-2 round 2): batch formation drain,
    /// conflict/dedup arbitration, fused patch+envelope, and settle-pass time — the
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
    /// leader at completion) gates settlement. Same-slot safety does not depend on apply
    /// completion: the bounded in-flight bridge records selected winners before WAL claim, and the
    /// publication epoch makes bridge removal atomic with the next device verdict. PK-hash routing
    /// pins a PK to one lane.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneTransactionClaimState {
    Pending,
    Terminal { commit_seq: u64, affected_rows: u64 },
    AbortedDiscardedOrphan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LaneTransactionClaim {
    pub(crate) request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) state: LaneTransactionClaimState,
}

pub(crate) enum LaneClaimResolution {
    New,
    Pending,
    Terminal(u64),
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

/// One lane's prepared device-apply request (the apply-side coalescer; same
/// leader pattern as validate). Everything the merged append needs travels in
/// the request — no `CommitWaveItem` re-walk on the apply path.
pub(crate) struct ApplyRequest {
    /// Owning lane for removal from its bounded in-flight arbitration set after device apply.
    pub(crate) lane: usize,
    pub(crate) table: String,
    /// INSERT winners only (parallel with `row_ids`/`stamps`/`txn_ids`): the merged
    /// open-shard append inputs. A delete-only wave ships these empty (U1).
    pub(crate) rows: Vec<Vec<crate::SqlValue>>,
    pub(crate) row_ids: Vec<u64>,
    pub(crate) stamps: Vec<u64>,
    pub(crate) txn_ids: Vec<u64>,
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
    /// The wave's WHOLE claimed seq block `[seq_first, seq_first + seq_len)` — the applied-cut
    /// advance covers every claimed seq regardless of the insert/delete mix (U1: `stamps` is
    /// insert-only and can no longer stand in for the block).
    pub(crate) seq_first: u64,
    pub(crate) seq_len: u64,
    /// Every winner's unique slot. These remain in `inflight_slots[lane]` only until this request
    /// completes its merged device apply.
    pub(crate) unique_slots: Vec<crate::write_path::IntUniqueSlotKey>,
    pub(crate) slot: std::sync::Arc<ApplySlot>,
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
    /// The shared cell the apply writes the resolved rows-affected (0 or 1) into; the settle
    /// reads it to ack. Shared with the delete's `LaneIntent.rows_affected_cell`.
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
    /// The shared cell the apply writes the resolved rows-affected (0 or 1) into; the settle reads
    /// it to ack. Shared with the update's `LaneIntent.rows_affected_cell`.
    pub(crate) rows_affected: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

/// Apply completion: `done` flips after the merged append (or its fallback)
/// covered this request's rows.
pub(crate) struct ApplySlot {
    pub(crate) done: AtomicBool,
    /// AUDIT F2: set when the apply leader panicked/poisoned before covering
    /// this request — the waiter must fail its winners, not settle them.
    pub(crate) failed: AtomicBool,
}

/// Min items before a lane wave ships (the adaptive formation's CAP; the live
/// population scales the effective target — see `drive_intent_lane`). Folded
/// from `GPU_DB_INTENT_LANE_MIN_WAVE` per the no-flag-proliferation ruling:
/// 512 won every sweep.
pub(crate) fn intent_lane_min_wave() -> usize {
    512
}

/// Sub-frames per wave publish: ALWAYS AUTO (split x2 when the lane's fence
/// pool is mostly idle — the low-depth bimodal-FUA regime; single frames when
/// busy). Folded from `GPU_DB_INTENT_LANE_SUBFRAMES`: AUTO won; fixed >=3
/// regressed past the drive's depth knee.
pub(crate) fn intent_lane_subframes() -> usize {
    0
}

/// ADR-014 standalone durability is strict: the compatibility environment setting is parsed by
/// callers but cannot lower the acknowledgement gate until a separate async-visibility design is
/// accepted. Returning `true` makes both configured modes wait for durable + applied publication.
pub(crate) fn synchronous_commit_default_from_env() -> bool {
    true
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

/// One lane wave awaiting the visible cut: `[first_seq, end_seq)` plus the
/// winner items whose outcome slots settle (Ok) when the cut covers end_seq.
/// NO-REAP PIPELINE: the entry is queued the moment the wave's ApplyRequest is
/// pushed (before the apply completes) — a settled-Ok is still correct because
/// the applied cut only advances when the apply LEADER completes the wave, and
/// the cut gates settlement. `apply_slot.failed` covers the failure path.
pub(crate) struct LaneSettle {
    pub(crate) end_seq: u64,
    /// Winners acking at the STRICT gate (visible cut = durable AND applied).
    pub(crate) winners: Vec<crate::engine_dml_concurrent::LaneIntent>,
    /// Retained layout slot from the pre-ADR-014 async experiment. Production formation leaves
    /// this empty: `SynchronousCommit::Off` is strict until a separate design is accepted.
    pub(crate) async_winners: Vec<crate::engine_dml_concurrent::LaneIntent>,
    /// True once `async_winners` were settled (the entry then waits only for
    /// the strict gate to settle `winners` and pop).
    pub(crate) async_settled: bool,
    pub(crate) apply_slot: std::sync::Arc<ApplySlot>,
    /// When the wave's WAL frames were published — settle accumulates
    /// publish→settle lag into `stat_acklag_ns` (the fence+cut+settle share
    /// of client ack latency, the low-load SLO's dominant term).
    pub(crate) published_at: std::time::Instant,
}

impl IntentLaneState {
    /// A fresh (never-activated) lane state with LAZY WAL backing at `lane_base` (created on
    /// the first lane wave — the E2.5c-3 default flip makes lanes-ON the durable default, and
    /// an engine that never takes the intent path must not pay N x 2 segment prewrites).
    pub(crate) fn fresh(
        lane_count: usize,
        fence_lanes: usize,
        lane_base: std::path::PathBuf,
        lane_segment_bytes: usize,
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
            synchronous_commit_default: AtomicBool::new(synchronous_commit_default_from_env()),
            stat_resizes: AtomicU64::new(0),
            stat_resize_ns: AtomicU64::new(0),
            outstanding: std::sync::Arc::new(AtomicU64::new(0)),
            wal_lanes: std::sync::OnceLock::new(),
            wal_lanes_init: Mutex::new(()),
            lane_base,
            lane_segment_bytes,
            applied: Mutex::new(SeqCut::default()),
            applied_mirror: AtomicU64::new(0),
            activated: AtomicBool::new(false),
            base_seq: AtomicU64::new(0),
            canonical_identity: OnceLock::new(),
            canonical_catalog_epoch: AtomicU64::new(0),
            canonical_catalog_digest: OnceLock::new(),
            transaction_claims: std::sync::Arc::new(Mutex::new(HashMap::new())),
            queues: (0..lane_count).map(|_| Default::default()).collect(),
            inflight_slots: (0..lane_count).map(|_| Default::default()).collect(),
            settle: (0..lane_count).map(|_| Default::default()).collect(),
            device_apply_lock: Mutex::new(()),
            device_publication_epoch: AtomicU64::new(0),
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

    /// A lane state over an ALREADY-OPEN WAL backing (the reopen path: the lane files exist
    /// and the recovered cuts must install; the caller then pre-seeds the activation latch,
    /// `base_seq`, the seq oracle, and the applied cut — single-threaded, plain stores).
    pub(crate) fn with_backing(
        lane_count: usize,
        fence_lanes: usize,
        wal_lanes: gpu_db_wal::FuaWalLaneSet,
        lane_segment_bytes: usize,
    ) -> Self {
        let lane_base = wal_lanes.base_path().to_path_buf();
        let state = Self::fresh(lane_count, fence_lanes, lane_base, lane_segment_bytes);
        let _ = state.wal_lanes.set(wal_lanes);
        state
    }

    /// The lane WAL, creating the on-disk backing on FIRST use (see the field docs). Failure
    /// (ENOSPC/EDQUOT during the per-lane prewrite) surfaces to the caller — the wave that
    /// triggered creation fails loudly; the engine itself stays up.
    pub(crate) fn wal(&self) -> Result<&gpu_db_wal::FuaWalLaneSet, crate::EngineError> {
        if let Some(set) = self.wal_lanes.get() {
            return Ok(set);
        }
        let _init = self
            .wal_lanes_init
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(set) = self.wal_lanes.get() {
            return Ok(set);
        }
        let set = gpu_db_wal::FuaWalLaneSet::create(
            &self.lane_base,
            self.lane_count,
            self.fence_lanes,
            self.lane_segment_bytes,
        )?;
        let _ = self.wal_lanes.set(set);
        Ok(self
            .wal_lanes
            .get()
            .expect("just installed under the init lock"))
    }

    pub(crate) fn claim_transaction(
        &self,
        txn_id: u64,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<LaneClaimResolution, crate::EngineError> {
        let mut claims = self
            .transaction_claims
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match claims.get(&txn_id).copied() {
            Some(claim) if claim.request_digest != request_digest => {
                Err(crate::EngineError::Durability(format!(
                    "transaction id {txn_id} is already durably claimed by a different request"
                )))
            }
            Some(LaneTransactionClaim {
                state: LaneTransactionClaimState::Pending,
                ..
            }) => Ok(LaneClaimResolution::Pending),
            Some(LaneTransactionClaim {
                state: LaneTransactionClaimState::Terminal { affected_rows, .. },
                ..
            }) => Ok(LaneClaimResolution::Terminal(affected_rows)),
            Some(LaneTransactionClaim {
                state: LaneTransactionClaimState::AbortedDiscardedOrphan,
                ..
            }) => Err(crate::EngineError::Durability(format!(
                "transaction id {txn_id} was durably aborted during crash recovery"
            ))),
            None => {
                claims.insert(
                    txn_id,
                    LaneTransactionClaim {
                        request_digest,
                        state: LaneTransactionClaimState::Pending,
                    },
                );
                Ok(LaneClaimResolution::New)
            }
        }
    }

    pub(crate) fn install_recovered_transaction(
        &self,
        txn_id: u64,
        request_digest: gpu_db_wal::CanonicalDigest,
        commit_seq: u64,
        affected_rows: u64,
    ) -> Result<(), crate::EngineError> {
        let mut claims = self
            .transaction_claims
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let recovered = LaneTransactionClaim {
            request_digest,
            state: LaneTransactionClaimState::Terminal {
                commit_seq,
                affected_rows,
            },
        };
        match claims.get(&txn_id).copied() {
            Some(existing) if existing != recovered => {
                return Err(crate::EngineError::Durability(format!(
                    "recovered transaction claim {txn_id} conflicts with an existing lane claim"
                )));
            }
            Some(_) => {}
            None => {
                claims.insert(txn_id, recovered);
            }
        }
        Ok(())
    }

    pub(crate) fn install_recovered_aborted_transaction(
        &self,
        txn_id: u64,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<(), crate::EngineError> {
        let mut claims = self
            .transaction_claims
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let recovered = LaneTransactionClaim {
            request_digest,
            state: LaneTransactionClaimState::AbortedDiscardedOrphan,
        };
        match claims.get(&txn_id).copied() {
            Some(existing) if existing != recovered => {
                Err(crate::EngineError::Durability(format!(
                    "reconciled transaction claim {txn_id} conflicts with an existing lane claim"
                )))
            }
            Some(_) => Ok(()),
            None => {
                claims.insert(txn_id, recovered);
                Ok(())
            }
        }
    }

    /// Non-creating peek for pollers (stats, settle, visibility): `None` means the intent
    /// path has never run — nothing durable, nothing poisoned, cut 0.
    pub(crate) fn wal_peek(&self) -> Option<&gpu_db_wal::FuaWalLaneSet> {
        self.wal_lanes.get()
    }

    /// The lanes' visibility frontier in LANE-LOCAL seq space (local seq =
    /// global seq - base_seq; the lane logs tile [0, N) exactly, per the
    /// FuaWalLaneSet contract — the pre-activation range lives in the serial
    /// log and never touches the lanes). Every local seq below this is durable
    /// in its WAL lane AND device-applied.
    pub(crate) fn visible_local_cut(&self) -> u64 {
        self.wal_peek()
            .map(|wal| wal.durable_cut())
            .unwrap_or(0)
            .min(self.applied_mirror.load(Ordering::Acquire))
    }

    /// The GLOBAL INCLUSIVE commit sequence the engine may publish for
    /// lane-claimed records. `visible_local_cut` is an EXCLUSIVE local
    /// next-slot frontier, so `[0, cut)` maps to global last sequence
    /// `base_seq + cut - 1`. `None` means no lane record is yet both durable
    /// and applied (including pre-activation); publishing `base_seq + cut`
    /// would expose the next, uncovered slot.
    pub(crate) fn visible_inclusive_seq(&self) -> Result<Option<u64>, crate::EngineError> {
        self.visible_boundary().map(|(_, inclusive)| inclusive)
    }

    /// Capture the lane-local cut and its global inclusive publication boundary from one
    /// observation. Settlement must use this pair: two independent `durable_cut()` reads can
    /// straddle another pump's cut advance, otherwise acknowledging against the newer local cut
    /// while publishing the older global boundary.
    pub(crate) fn visible_boundary(&self) -> Result<(u64, Option<u64>), crate::EngineError> {
        if !self.activated.load(Ordering::Acquire) {
            return Ok((0, None));
        }
        let local_cut = self.visible_local_cut();
        let inclusive =
            inclusive_seq_from_exclusive_prefix(self.base_seq.load(Ordering::Acquire), local_cut)
                .map_err(|message| crate::EngineError::Durability(message.to_string()))?;
        Ok((local_cut, inclusive))
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

#[cfg(test)]
mod boundary_tests {
    #[test]
    fn exclusive_lane_cut_maps_to_inclusive_commit_boundary() {
        assert_eq!(super::inclusive_global_boundary(4, 0), 3);
        assert_eq!(super::inclusive_global_boundary(4, 1), 4);
        assert_eq!(super::inclusive_global_boundary(4, 7), 10);
    }
}

/// `GPU_DB_INTENT_LANES` (default 1 = lanes mode OFF; >= 2 enables). Read once
/// at engine construction, like the other write-path knobs.
impl crate::Engine {
    #[cfg(test)]
    pub(crate) fn set_intent_lanes_classic_prelock_hook(
        &self,
        reached: std::sync::Arc<std::sync::Barrier>,
        resume: std::sync::Arc<std::sync::Barrier>,
    ) {
        *classic_prelock_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some((self as *const Self as usize, reached, resume));
    }

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
            lanes.wal_peek().map(|wal| wal.durable_cut()).unwrap_or(0),
            lanes.applied_mirror.load(Ordering::Acquire),
            self.read_state
                .residency
                .lane_diag_rebuilds
                .load(Ordering::Relaxed),
        ))
    }

    /// Pump host-pass diagnostics: (drain_ns, conflict_ns, patch_ns, settle_ns)
    /// — the formation/arbitration/patch+envelope/settle passes between the staged
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
        Some(
            lanes
                .wal_peek()
                .map(|wal| wal.fence_latency_stats())
                .unwrap_or((0, 0)),
        )
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

    /// V1 intent-only contract: once the first nonempty lane wave activates, classic
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
        #[cfg(test)]
        {
            let prelock_hook = {
                let mut hook = classic_prelock_hook()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                hook.as_ref()
                    .is_some_and(|(owner, _, _)| *owner == self as *const Self as usize)
                    .then(|| hook.take())
                    .flatten()
            };
            if let Some((_, reached, resume)) = prelock_hook {
                reached.wait();
                resume.wait();
            }
        }
        Ok(())
    }
}

/// Max items drained per lane wave. Folded from `GPU_DB_INTENT_LANE_WAVE_MAX`
/// (never a winning lever in any sweep; bounds the fused patch pass).
pub(crate) fn intent_lane_wave_max() -> usize {
    1024
}

/// Intent lane count (`GPU_DB_INTENT_LANES`). E2.5c-3 DEFAULT FLIP: lanes mode is ON by
/// default on unix (10 lanes — the measured champion on the reference box; the WAL backing is
/// LAZY, so engines that never take the intent path pay nothing). Explicit `0`/`1` disables.
pub(crate) fn intent_lane_count() -> usize {
    std::env::var("GPU_DB_INTENT_LANES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|n| if n >= 2 { n } else { 1 })
        .unwrap_or(if cfg!(unix) { 10 } else { 1 })
}

/// Fence lanes per WAL lane (the per-lane FUA pool depth). Folded from
/// `GPU_DB_INTENT_LANE_FENCES`: 16 = the drive's fast-mode knee; 24/32
/// regressed (thread contention pre-park, no gain post-park).
pub(crate) fn intent_lane_fences() -> usize {
    16
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

    #[cfg(unix)]
    #[test]
    fn visible_lane_prefix_publishes_inclusive_last_commit_not_exclusive_next() {
        let base = std::env::temp_dir().join(format!(
            "gpu-db-r3-006-prefix-{}-{:?}.wal",
            std::process::id(),
            std::thread::current().id()
        ));
        gpu_db_wal::remove_stale_lane_files(&base).expect("clear stale lane files");
        let wal = gpu_db_wal::FuaWalLaneSet::create(&base, 2, 1, 1 << 20)
            .expect("create bounded test lane set");
        wal.append(
            0,
            0,
            &[gpu_db_wal::WalRecord {
                txn_id: 7,
                payload: b"r3-006".to_vec().into(),
            }],
        )
        .expect("append first local record");
        wal.wait_durable(1).expect("fence first local record");

        let state = IntentLaneState::with_backing(2, 1, wal, 1 << 20);
        state.base_seq.store(41, Ordering::Release);
        state.activated.store(true, Ordering::Release);
        state.record_applied(0, 1);

        assert_eq!(state.visible_local_cut(), 1);
        assert_eq!(
            state.visible_inclusive_seq().expect("valid prefix"),
            Some(41),
            "[0, 1) covers global commit 41; 42 is the exclusive next slot and must stay hidden"
        );

        drop(state);
        gpu_db_wal::remove_stale_lane_files(&base).expect("remove bounded test lane files");
    }

    #[cfg(unix)]
    #[test]
    fn visible_lane_prefix_holds_when_apply_leads_durability() {
        let base = std::env::temp_dir().join(format!(
            "gpu-db-r3-006-apply-leads-{}-{:?}.wal",
            std::process::id(),
            std::thread::current().id()
        ));
        gpu_db_wal::remove_stale_lane_files(&base).expect("clear stale lane files");
        let wal = gpu_db_wal::FuaWalLaneSet::create(&base, 2, 1, 1 << 20)
            .expect("create bounded test lane set");
        let state = IntentLaneState::with_backing(2, 1, wal, 1 << 20);
        state.base_seq.store(41, Ordering::Release);
        state.activated.store(true, Ordering::Release);

        state.record_applied(0, 2);
        assert_eq!(
            state.visible_inclusive_seq().expect("valid empty prefix"),
            None,
            "applied work is hidden while the durable prefix is empty"
        );
        let wal = state.wal_peek().expect("installed WAL");
        for local in 0..2 {
            wal.append(
                local as usize,
                local,
                &[gpu_db_wal::WalRecord {
                    txn_id: 10 + local,
                    payload: format!("apply-leads-{local}").into_bytes().into(),
                }],
            )
            .expect("append local record");
            wal.wait_durable(local + 1).expect("fence local record");
            assert_eq!(
                state.visible_inclusive_seq().expect("valid joined prefix"),
                Some(41 + local),
                "only the newly durable member of the applied prefix may publish"
            );
        }

        drop(state);
        gpu_db_wal::remove_stale_lane_files(&base).expect("remove bounded test lane files");
    }

    #[cfg(unix)]
    #[test]
    fn visible_lane_prefix_holds_when_durability_leads_apply() {
        let base = std::env::temp_dir().join(format!(
            "gpu-db-r3-006-durable-leads-{}-{:?}.wal",
            std::process::id(),
            std::thread::current().id()
        ));
        gpu_db_wal::remove_stale_lane_files(&base).expect("clear stale lane files");
        let wal = gpu_db_wal::FuaWalLaneSet::create(&base, 2, 1, 1 << 20)
            .expect("create bounded test lane set");
        for local in 0..2 {
            wal.append(
                local as usize,
                local,
                &[gpu_db_wal::WalRecord {
                    txn_id: 20 + local,
                    payload: format!("durable-leads-{local}").into_bytes().into(),
                }],
            )
            .expect("append local record");
        }
        wal.wait_durable(2).expect("fence both local records");
        let state = IntentLaneState::with_backing(2, 1, wal, 1 << 20);
        state.base_seq.store(41, Ordering::Release);
        state.activated.store(true, Ordering::Release);

        assert_eq!(
            state.visible_inclusive_seq().expect("valid empty prefix"),
            None,
            "durable work is hidden while the applied prefix is empty"
        );
        state.record_applied(0, 1);
        assert_eq!(
            state.visible_inclusive_seq().expect("valid first prefix"),
            Some(41),
            "the durable second slot stays hidden behind apply lag"
        );
        state.record_applied(1, 2);
        assert_eq!(
            state.visible_inclusive_seq().expect("valid second prefix"),
            Some(42)
        );

        drop(state);
        gpu_db_wal::remove_stale_lane_files(&base).expect("remove bounded test lane files");
    }

    #[test]
    fn exclusive_prefix_conversion_covers_genesis_normal_and_exhaustion_boundaries() {
        assert_eq!(
            inclusive_seq_from_exclusive_prefix(1, 0).expect("empty prefix"),
            None
        );
        assert_eq!(
            inclusive_seq_from_exclusive_prefix(1, 1).expect("first commit"),
            Some(1)
        );
        assert_eq!(
            inclusive_seq_from_exclusive_prefix(41, 9).expect("normal prefix"),
            Some(49)
        );
        assert_eq!(
            inclusive_seq_from_exclusive_prefix(u64::MAX - 1, 1)
                .expect("last representable commit"),
            Some(u64::MAX - 1)
        );
        assert!(
            inclusive_seq_from_exclusive_prefix(u64::MAX - 1, 2).is_err(),
            "the infinity sentinel must never become an inclusive commit sequence"
        );
    }

    #[test]
    fn synchronous_commit_off_compatibility_setting_keeps_the_strict_gate() {
        let mut engine = crate::Engine::new_local_cpu_oracle();
        let base = std::env::temp_dir().join(format!(
            "gpu-db-strict-commit-setting-{}-{:?}.wal",
            std::process::id(),
            std::thread::current().id()
        ));
        engine.attach_test_intent_lanes(base, 2);
        engine.set_synchronous_commit_default(crate::SynchronousCommit::Off);
        assert!(
            engine
                .intent_lanes
                .as_ref()
                .unwrap()
                .synchronous_commit_default
                .load(Ordering::Relaxed),
            "Off must behave synchronously until an async durability design is accepted"
        );
        assert!(synchronous_commit_default_from_env());
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
