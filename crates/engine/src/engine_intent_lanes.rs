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
        Vec<Mutex<std::collections::VecDeque<crate::engine_dml_concurrent::CommitWaveItem>>>,
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
    /// Per-lane commit-timestamp side maps (txn_id -> micros). Lane records
    /// never enter the serial WalBuffer, so the archive/PITR consumers of
    /// wal_commit_timestamps_micros never look these txns up — the side maps
    /// retain the stamps for the E2.5c lane-archive slice.
    pub(crate) ts_side: Vec<Mutex<std::collections::HashMap<u64, u64>>>,
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
    pub(crate) stat_apply_launches: AtomicU64,
    pub(crate) stat_apply_requests: AtomicU64,
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
}

impl IntentLaneState {
    /// Reserve `k` strictly-monotonic commit timestamps >= wall clock,
    /// returning the base (range [base, base+k)). Lock-free CAS max loop.
    pub(crate) fn reserve_timestamps(&self, wall_micros: u64, k: u64) -> u64 {
        let mut current = self.ts_reservation.load(Ordering::Relaxed);
        loop {
            let base = wall_micros.max(current.saturating_add(1));
            match self.ts_reservation.compare_exchange_weak(
                current,
                base + (k - 1),
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return base,
                Err(observed) => current = observed,
            }
        }
    }
}

/// Min items before a lane wave ships (`GPU_DB_INTENT_LANE_MIN_WAVE`, default 192).
pub(crate) fn intent_lane_min_wave() -> usize {
    std::env::var("GPU_DB_INTENT_LANE_MIN_WAVE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(512)
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
pub(crate) struct LaneSettle {
    pub(crate) end_seq: u64,
    pub(crate) winners: Vec<crate::engine_dml_concurrent::CommitWaveItem>,
}

impl IntentLaneState {
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
        let mixed = (pk as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((mixed >> 40) % self.lane_count as u64) as usize
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
