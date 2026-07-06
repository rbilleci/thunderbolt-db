//! N INDEPENDENT ORDERED WAL LANES with explicit global commit seqs (E2.5a — Variant 2).
//!
//! The single-log FUA path ([`crate::fua`]) parallelizes DURABILITY (a fence pool) but keeps the
//! ordered append/commit CUT serial: every group's `first_seq` is a lane-local record count, so
//! recovery infers order positionally within one log. E2.4a measured that serial cut to be the
//! wall (~1.3µs/item). Variant 2 splits the cut across `N` lanes:
//!
//! * **Each lane is its own [`FuaWalBackend`] segment chain** (own appender, own fence pool, own
//!   rolling/poison). Appends to different lanes share NOTHING — no mutex on the cross-lane append
//!   path (a lane holds only its own backend lock + its own interval queue lock).
//! * **Records carry EXPLICIT GLOBAL seqs.** A lane stamps the caller-assigned global seq into the
//!   EXISTING frame-header `first_seq` (the field is already opaque to the frame log). No frame
//!   format change, and no `write_conveyor` change: a lane's frame covers the contiguous global-seq
//!   block `[first_seq, first_seq + seq_count)` it was handed, and successive frames in one lane
//!   have strictly increasing `first_seq` (block claims are a monotonic global allocation), so the
//!   per-lane contiguous-durable cut math is byte-identical to the single-log backend.
//! * **Recovery semantics change from LANE-LOCAL to GLOBAL, gated by a SEPARATE entry point.** The
//!   single-log [`recover_fua_wal_records`] (which enforces `first_seq` contiguity WITHIN one log)
//!   is untouched; lane-sets use their OWN files (`<base>.lane-<L>.fua.<id>`) and their OWN
//!   [`recover_lanes`] merge. Existing single-lane logs stay readable exactly as before.
//!
//! The correctness heart is the CROSS-LANE CONTIGUOUS DURABLE CUT ([`CutState`]): the largest `S`
//! such that every global seq `< S` is durable in its owning lane. Lanes report per-frame durable
//! intervals; the merger holds the cut at the FIRST gap (a lagging lane, or an unclaimed seq),
//! exactly the fence pool's contiguous-cut law one level up.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use gpu_db_types::EngineError;
use gpu_db_write_conveyor::recover_frame_log_by_scan;

use crate::fua::{fua_segment_paths_sorted, FuaWalBackend};
use crate::{decode_wal_record_run, encode_record_into, WalRecord};

/// Busy-spins before falling back to `yield_now` in [`FuaWalLaneSet::wait_durable`] (same law as
/// the single-log backend: no per-commit futex wakeups).
const SPIN_BEFORE_YIELD: u32 = 256;

/// One lane's on-disk base path: `<base>.lane-<lane_id>` (its segment files are then
/// `<base>.lane-<lane_id>.fua.<segment_id>`, owned exclusively by that lane's backend).
fn lane_base_path(base: &Path, lane_id: usize) -> PathBuf {
    let stem = base
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("wal.segment");
    base.with_file_name(format!("{stem}.lane-{lane_id}"))
}

/// Parse the lane id out of a `<stem>.lane-<L>.fua.<segment_id>` file name (segment id must be a
/// valid u64 so we do not mistake an unrelated `<stem>.lane-<L>.foo` for a lane segment).
fn parse_lane_id(name: &str, stem: &str) -> Option<usize> {
    let prefix = format!("{stem}.lane-");
    let rest = name.strip_prefix(&prefix)?;
    let (lane_str, segment_str) = rest.split_once(".fua.")?;
    segment_str.parse::<u64>().ok()?;
    lane_str.parse::<usize>().ok()
}

/// Distinct lane ids that have at least one segment file beside `base`.
fn discover_lane_ids(base: &Path) -> BTreeSet<usize> {
    let mut ids = BTreeSet::new();
    let Some(parent) = base.parent().filter(|p| !p.as_os_str().is_empty()) else {
        return ids;
    };
    let Some(stem) = base.file_name().and_then(|n| n.to_str()) else {
        return ids;
    };
    if let Ok(entries) = std::fs::read_dir(parent) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if let Some(id) = parse_lane_id(name, stem) {
                    ids.insert(id);
                }
            }
        }
    }
    ids
}

/// The cross-lane contiguous durable cut merger.
///
/// `cut` = the current `global_durable_seq`; `[0, cut)` is contiguously durable across every lane.
/// `pending` holds durable intervals whose start is above `cut` (blocked behind a gap); each is
/// absorbed once the cut reaches its start. This is order-INSENSITIVE: intervals may be ingested
/// in any interleaving and the greedy advance always yields the same cut (the largest contiguous
/// prefix), so it holds correctly at a gap regardless of which lane got ahead.
#[derive(Debug)]
struct CutState {
    cut: u64,
    pending: BTreeMap<u64, u64>,
}

impl CutState {
    fn new(cut: u64) -> Self {
        Self {
            cut,
            pending: BTreeMap::new(),
        }
    }

    /// Record a durable interval `[start, end)`. Idempotent-ish: a duplicate start keeps the larger
    /// end (defensive against a re-reported frame).
    fn ingest(&mut self, start: u64, end: u64) {
        let slot = self.pending.entry(start).or_insert(end);
        if end > *slot {
            *slot = end;
        }
    }

    /// Advance the cut over every pending interval that is now contiguous from `cut`. Returns the
    /// new cut. An interval with `start > cut` is a GAP and stops the advance (held until its
    /// predecessor becomes durable).
    fn advance(&mut self) -> u64 {
        while let Some((&start, &end)) = self.pending.first_key_value() {
            if start > self.cut {
                break;
            }
            self.pending.remove(&start);
            if end > self.cut {
                self.cut = end;
            }
        }
        self.cut
    }
}

/// One lane: its own segment-chain backend and the queue of global-seq intervals it has appended
/// but the cut has not yet absorbed. The interval queue is drained (front-to-back, in append
/// order) by [`FuaWalLaneSet::durable_cut`] as the lane's own contiguous-durable end advances.
struct Lane {
    backend: Arc<FuaWalBackend>,
    intervals: Mutex<VecDeque<(u64, u64)>>,
}

/// A set of `N` independent ordered WAL lanes with a cross-lane contiguous durable cut and merge
/// recovery. See the module docs for the design and invariants.
pub struct FuaWalLaneSet {
    base: PathBuf,
    lanes: Vec<Lane>,
    /// The cross-lane merger. Touched only by cut READERS ([`Self::durable_cut`]) — NEVER on the
    /// append path — so appends to different lanes do not contend on it.
    cut: Mutex<CutState>,
}

impl std::fmt::Debug for FuaWalLaneSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuaWalLaneSet")
            .field("base", &self.base)
            .field("lanes", &self.lanes.len())
            .finish_non_exhaustive()
    }
}

impl FuaWalLaneSet {
    /// Create a FRESH `lane_count`-lane WAL set at `base`. Each lane gets its own fence pool of
    /// `fence_lanes` and `segment_bytes` segments; any stale `<base>.lane-*.fua.*` files are
    /// clobbered per lane (the same fresh-create semantic as the single-log backend).
    pub fn create(
        base: impl Into<PathBuf>,
        lane_count: usize,
        fence_lanes: usize,
        segment_bytes: usize,
    ) -> Result<Self, EngineError> {
        let base = base.into();
        if lane_count == 0 {
            return Err(EngineError::Durability(
                "FUA WAL lane set requires at least one lane".to_string(),
            ));
        }
        let mut lanes = Vec::with_capacity(lane_count);
        for lane_id in 0..lane_count {
            let backend =
                FuaWalBackend::create(lane_base_path(&base, lane_id), fence_lanes, segment_bytes)?;
            lanes.push(Lane {
                backend: Arc::new(backend),
                intervals: Mutex::new(VecDeque::new()),
            });
        }
        Ok(Self {
            base,
            lanes,
            cut: Mutex::new(CutState::new(0)),
        })
    }

    /// REOPEN an existing `lane_count`-lane WAL set and continue appending ABOVE the recovered
    /// global history. The caller has already replayed [`recover_lanes`]. Reopen requires a CLEAN
    /// state (no torn tail / no gap): if any lane holds durable frames ABOVE the global contiguous
    /// cut, reopen FAILS CLOSED — those orphaned frames would otherwise collide (duplicate global
    /// seqs) with the new segment's appends. Recover such a database offline first. A mismatched
    /// `lane_count` is a clear error.
    pub fn reopen(
        base: impl Into<PathBuf>,
        lane_count: usize,
        fence_lanes: usize,
        segment_bytes: usize,
    ) -> Result<Self, EngineError> {
        let base = base.into();
        let recovery = recover_lanes_detailed(&base, lane_count)?;
        if recovery.records.len() != recovery.total_scanned {
            return Err(EngineError::Durability(format!(
                "cannot reopen FUA WAL lane set at {}: {} durable record(s) lie ABOVE the global \
                 contiguous cut of {} (a torn tail / gap in one lane); recover offline before \
                 reopening so no acked seq is lost",
                base.display(),
                recovery.total_scanned - recovery.records.len(),
                recovery.records.len(),
            )));
        }
        let mut lanes = Vec::with_capacity(lane_count);
        for lane_id in 0..lane_count {
            let backend = FuaWalBackend::reopen(
                lane_base_path(&base, lane_id),
                fence_lanes,
                segment_bytes,
                recovery.lane_ends[lane_id] as usize,
            )?;
            lanes.push(Lane {
                backend: Arc::new(backend),
                intervals: Mutex::new(VecDeque::new()),
            });
        }
        Ok(Self {
            base,
            lanes,
            cut: Mutex::new(CutState::new(recovery.next_seq)),
        })
    }

    pub fn lane_count(&self) -> usize {
        self.lanes.len()
    }

    pub fn base_path(&self) -> &Path {
        &self.base
    }

    /// Free fence slots in a lane's pool — the per-lane PACING signal (append only while a slot is
    /// free, exactly as the single-log backend paces).
    pub fn free_fence_slots(&self, lane: usize) -> Option<usize> {
        self.lanes.get(lane).map(|l| l.backend.free_fence_slots())
    }

    /// Append ONE contiguous global-seq block `[first_seq, first_seq + records.len())` to `lane` as
    /// a single frame and return WITHOUT blocking on durability (the fence pool pipelines it; poll
    /// [`Self::durable_cut`] for the cross-lane watermark). CONTRACT: `first_seq` must be strictly
    /// increasing per lane (monotonic block claims) and the union of all lanes' blocks must tile
    /// `[0, N)` with no permanent gap for the cut to advance; a single writer drives each lane.
    pub fn append(
        &self,
        lane: usize,
        first_seq: u64,
        records: &[WalRecord],
    ) -> Result<(), EngineError> {
        let Some(lane_ref) = self.lanes.get(lane) else {
            return Err(EngineError::Durability(format!(
                "FUA WAL lane set append to lane {lane} but only {} lane(s) exist",
                self.lanes.len()
            )));
        };
        if records.is_empty() {
            return Ok(());
        }
        let seq_count = u32::try_from(records.len()).map_err(|_| {
            EngineError::Durability("FUA WAL lane frame record count exceeds u32".to_string())
        })?;
        let payload = encode_lane_frame_payload(records)?;
        let end = first_seq + records.len() as u64;
        self.append_encoded_inner(lane_ref, &payload, first_seq, end, seq_count)
    }

    /// Two-phase append for instrumented callers: publish an already-encoded
    /// payload (from [`encode_lane_frame_payload`]) covering [first_seq, end).
    pub fn append_encoded(
        &self,
        lane: usize,
        first_seq: u64,
        end: u64,
        seq_count: u32,
        payload: &[u8],
    ) -> Result<(), EngineError> {
        let Some(lane_ref) = self.lanes.get(lane) else {
            return Err(EngineError::Durability(format!(
                "FUA WAL lane set append to lane {lane} but only {} lane(s) exist",
                self.lanes.len()
            )));
        };
        self.append_encoded_inner(lane_ref, payload, first_seq, end, seq_count)
    }

    fn append_encoded_inner(
        &self,
        lane_ref: &Lane,
        payload: &[u8],
        first_seq: u64,
        end: u64,
        seq_count: u32,
    ) -> Result<(), EngineError> {
        // Record the interval BEFORE publishing so the cut can never observe a durable frame whose
        // interval it has not yet seen. If the publish then fails, the interval sits un-absorbable
        // in the queue (its frame never becomes durable) and the cut holds — fail-closed.
        lane_ref
            .intervals
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push_back((first_seq, end));
        lane_ref
            .backend
            .append_frame(&payload, first_seq, seq_count)?;
        Ok(())
    }

    /// The cross-lane CONTIGUOUS durable cut: the largest `S` such that every global seq `< S` is
    /// durable in its owning lane. Non-blocking — it reads each lane's cheap durable-end atomic,
    /// absorbs any newly-durable intervals, and advances the merged watermark. Holds at the first
    /// gap (a lagging lane).
    pub fn durable_cut(&self) -> u64 {
        let mut cut = self.cut.lock().unwrap_or_else(|p| p.into_inner());
        for lane in &self.lanes {
            let durable_end = lane.backend.durable_records() as u64;
            let mut queue = lane.intervals.lock().unwrap_or_else(|p| p.into_inner());
            while let Some(&(start, end)) = queue.front() {
                if end <= durable_end {
                    queue.pop_front();
                    cut.ingest(start, end);
                } else {
                    break;
                }
            }
        }
        cut.advance()
    }

    /// First lane durability failure, if any (surfaces the wedge to cut waiters / the engine seam).
    pub fn poison_reason(&self) -> Option<String> {
        for (lane_id, lane) in self.lanes.iter().enumerate() {
            if let Some(reason) = lane.backend.poison_reason() {
                return Some(format!(
                    "FUA WAL lane {lane_id} durability failure: {reason}"
                ));
            }
        }
        None
    }

    /// Wait (spin-then-yield, no per-commit wakeups) until the cross-lane cut covers `seq` — the
    /// ack path. Fails closed if any lane wedges.
    pub fn wait_durable(&self, seq: u64) -> Result<(), EngineError> {
        let mut spins = 0u32;
        loop {
            if self.durable_cut() >= seq {
                return Ok(());
            }
            if let Some(reason) = self.poison_reason() {
                return Err(EngineError::Durability(reason));
            }
            if spins < SPIN_BEFORE_YIELD {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// Rich result of a lane-set merge recovery (the reopen path needs the torn-tail detection and the
/// per-lane continuation baselines; [`recover_lanes`] exposes only `records`).
struct LaneRecovery {
    /// Records in ascending GLOBAL seq, up to (exclusive) the first missing seq.
    records: Vec<WalRecord>,
    /// Total records scanned across all lanes (>= `records.len()`; a surplus means orphans above
    /// the cut — a torn tail).
    total_scanned: usize,
    /// The next global seq to assign on reopen (== `records.len()`).
    next_seq: u64,
    /// Per-lane highest durable global end (the lane's reopen continuation baseline).
    lane_ends: Vec<u64>,
}

fn recover_lanes_detailed(base: &Path, lane_count: usize) -> Result<LaneRecovery, EngineError> {
    if lane_count == 0 {
        return Err(EngineError::Durability(
            "FUA WAL lane set recovery requires at least one lane".to_string(),
        ));
    }
    let discovered = discover_lane_ids(base);
    let mut lane_ends = vec![0u64; lane_count];
    if discovered.is_empty() {
        // No lane segments at all: a fresh (never-flushed) database.
        return Ok(LaneRecovery {
            records: Vec::new(),
            total_scanned: 0,
            next_seq: 0,
            lane_ends,
        });
    }
    let expected: BTreeSet<usize> = (0..lane_count).collect();
    if discovered != expected {
        return Err(EngineError::Durability(format!(
            "FUA WAL lane set at {} has lane ids {:?} but {} lane(s) were requested (0..{}); \
             the lane count must match the original database",
            base.display(),
            discovered,
            lane_count,
            lane_count
        )));
    }

    // Merge every lane's durable frames by GLOBAL seq. The whole space is partitioned across lanes,
    // so a seq appears in exactly one lane; a duplicate is corruption.
    let mut by_seq: BTreeMap<u64, WalRecord> = BTreeMap::new();
    for (lane_id, lane_end) in lane_ends.iter_mut().enumerate() {
        let lane_base = lane_base_path(base, lane_id);
        for segment_path in fua_segment_paths_sorted(&lane_base)? {
            let frames = recover_frame_log_by_scan(&segment_path).map_err(|err| {
                EngineError::Durability(format!(
                    "failed to scan-recover FUA WAL lane {lane_id} segment {}: {err}",
                    segment_path.display()
                ))
            })?;
            for frame in frames {
                let decoded = decode_wal_record_run(&frame.payload)?;
                if decoded.len() as u64 != frame.seq_count as u64 {
                    return Err(EngineError::Durability(format!(
                        "FUA WAL lane {lane_id} segment {} frame {} declares {} records but its \
                         payload decodes to {}",
                        segment_path.display(),
                        frame.frame_id,
                        frame.seq_count,
                        decoded.len()
                    )));
                }
                for (offset, record) in decoded.into_iter().enumerate() {
                    let seq = frame.first_seq + offset as u64;
                    if by_seq.insert(seq, record).is_some() {
                        return Err(EngineError::Durability(format!(
                            "FUA WAL lane set at {} has duplicate global seq {seq} (lane {lane_id} \
                             overlaps another lane's claim)",
                            base.display()
                        )));
                    }
                }
                let end = frame.first_seq + frame.seq_count as u64;
                *lane_end = (*lane_end).max(end);
            }
        }
    }

    let total_scanned = by_seq.len();
    // Contiguous global prefix from 0; the first missing seq (a torn tail in some lane, or an
    // unclaimed seq) truncates the global history there — fail-closed.
    let mut records = Vec::new();
    for (expected_seq, (seq, record)) in by_seq.into_iter().enumerate() {
        if seq != expected_seq as u64 {
            break;
        }
        records.push(record);
    }
    let next_seq = records.len() as u64;
    Ok(LaneRecovery {
        records,
        total_scanned,
        next_seq,
        lane_ends,
    })
}

/// Recover the totally-ordered `WalRecord` history from an `n`-lane FUA WAL set at `base_path`, in
/// ASCENDING global seq. Each lane is scan-recovered to its contiguous durable frame prefix and the
/// lanes are merged by the global seq embedded in each frame's `first_seq`. A torn tail in ANY lane
/// (or any unclaimed seq) truncates the global history at the FIRST missing seq — fail-closed.
/// Because the cross-lane cut only advances when a seq is durable in its lane, every ACKED seq is
/// strictly below the cut and is always recovered: nothing acked is lost. A `lane_count` that does
/// not match the on-disk database is a clear error.
/// Encode a lane frame payload (the record run) for [`FuaWalLaneSet::append_encoded`].
pub fn encode_lane_frame_payload(records: &[WalRecord]) -> Result<Vec<u8>, EngineError> {
    let mut payload = Vec::with_capacity(records.iter().map(|r| r.payload.len() + 32).sum());
    for record in records {
        encode_record_into(&mut payload, record)?;
    }
    Ok(payload)
}

pub fn recover_lanes(
    base_path: impl AsRef<Path>,
    lane_count: usize,
) -> Result<Vec<WalRecord>, EngineError> {
    Ok(recover_lanes_detailed(base_path.as_ref(), lane_count)?.records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_db_types::TxnId;

    fn test_base(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("fua-lane-set-tests");
        std::fs::create_dir_all(&dir).expect("create test dir");
        let base = dir.join(format!("{name}-{}.wal", std::process::id()));
        // Clear any stray lane segments from a prior run of this exact name.
        for lane in 0..16 {
            for path in fua_segment_paths_sorted(&lane_base_path(&base, lane)).unwrap_or_default() {
                let _ = std::fs::remove_file(path);
            }
        }
        base
    }

    fn record(seq: u64) -> WalRecord {
        // Payload encodes the global seq so recovery order is checkable; kept < 4032B so one record
        // is exactly one 4KiB frame (deterministic on-disk layout for the torn-tail test).
        WalRecord {
            txn_id: seq as TxnId,
            payload: format!("seq-{seq}").into_bytes().into(),
        }
    }

    fn cleanup(base: &Path, lane_count: usize) {
        for lane in 0..lane_count {
            for path in fua_segment_paths_sorted(&lane_base_path(base, lane)).unwrap_or_default() {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    const SEGMENT_BYTES: usize = 8 << 20; // 8 MiB — all test frames fit one segment (no rolling).

    /// Round-robin one-record-per-frame append of global seqs `0..total` across `lane_count` lanes.
    fn append_round_robin(set: &FuaWalLaneSet, total: u64) {
        for seq in 0..total {
            let lane = (seq % set.lane_count() as u64) as usize;
            set.append(lane, seq, &[record(seq)]).expect("append");
        }
    }

    #[test]
    fn interleaved_multi_lane_appends_recover_in_ascending_global_order() {
        let base = test_base("interleaved");
        let total = 200u64;
        {
            let set = FuaWalLaneSet::create(&base, 4, 4, SEGMENT_BYTES).expect("create");
            append_round_robin(&set, total);
            set.wait_durable(total).expect("all durable");
            assert_eq!(set.durable_cut(), total);
            // Drop drains every lane's fence pool → fully durable on disk.
        }
        let records = recover_lanes(&base, 4).expect("recover");
        assert_eq!(records.len(), total as usize);
        for (seq, rec) in records.iter().enumerate() {
            assert_eq!(
                rec,
                &record(seq as u64),
                "record {seq} out of order or wrong"
            );
        }
        cleanup(&base, 4);
    }

    #[test]
    fn single_lane_degenerates_to_ordered_log() {
        let base = test_base("single-lane");
        let total = 64u64;
        {
            let set = FuaWalLaneSet::create(&base, 1, 4, SEGMENT_BYTES).expect("create");
            for seq in 0..total {
                set.append(0, seq, &[record(seq)]).expect("append");
            }
            set.wait_durable(total).expect("durable");
            assert_eq!(set.durable_cut(), total);
        }
        let records = recover_lanes(&base, 1).expect("recover");
        assert_eq!(records.len(), total as usize);
        for (seq, rec) in records.iter().enumerate() {
            assert_eq!(rec, &record(seq as u64));
        }
        cleanup(&base, 1);
    }

    #[test]
    fn reopen_continues_global_seqs() {
        let base = test_base("reopen");
        {
            let set = FuaWalLaneSet::create(&base, 3, 2, SEGMENT_BYTES).expect("create");
            append_round_robin(&set, 30);
            set.wait_durable(30).expect("durable");
        }
        {
            let set = FuaWalLaneSet::reopen(&base, 3, 2, SEGMENT_BYTES).expect("reopen");
            assert_eq!(
                set.durable_cut(),
                30,
                "reopened cut continues from the recovered global history"
            );
            for seq in 30..48 {
                let lane = (seq % 3) as usize;
                set.append(lane, seq, &[record(seq)])
                    .expect("append after reopen");
            }
            set.wait_durable(48).expect("durable after reopen");
            assert_eq!(set.durable_cut(), 48);
        }
        let records = recover_lanes(&base, 3).expect("recover");
        assert_eq!(records.len(), 48);
        for (seq, rec) in records.iter().enumerate() {
            assert_eq!(rec, &record(seq as u64));
        }
        cleanup(&base, 3);
    }

    #[test]
    fn mismatched_lane_count_is_a_clear_error() {
        let base = test_base("lane-mismatch");
        {
            let set = FuaWalLaneSet::create(&base, 4, 2, SEGMENT_BYTES).expect("create");
            append_round_robin(&set, 40);
            set.wait_durable(40).expect("durable");
        }
        let err = recover_lanes(&base, 3).expect_err("lane count mismatch must error");
        assert!(
            matches!(&err, EngineError::Durability(msg) if msg.contains("lane count must match")),
            "unexpected error: {err:?}"
        );
        assert!(FuaWalLaneSet::reopen(&base, 2, 2, SEGMENT_BYTES).is_err());
        // Correct lane count still recovers.
        assert_eq!(recover_lanes(&base, 4).expect("recover 4").len(), 40);
        cleanup(&base, 4);
    }

    /// Corrupt the payload of the frame holding global seq `torn_seq` (breaking its CRC) so scan
    /// recovery of that lane stops there — simulating a torn tail. Layout: 4KiB file header then
    /// one 4KiB frame per record; within lane `L = torn_seq % n` the frame index is `torn_seq / n`.
    fn corrupt_frame(base: &Path, lane_count: usize, torn_seq: u64) {
        use std::io::{Seek, SeekFrom, Write};
        let lane = (torn_seq % lane_count as u64) as usize;
        let frame_index = torn_seq / lane_count as u64;
        let segments = fua_segment_paths_sorted(&lane_base_path(base, lane)).expect("segments");
        let path = segments.first().expect("lane segment file");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("open for corruption");
        // 4096 file header + frame_index * 4096 (frame) + 64 (frame header) = first payload byte.
        let offset = 4096 + frame_index * 4096 + 64;
        file.seek(SeekFrom::Start(offset)).expect("seek");
        file.write_all(&[0xFF]).expect("corrupt byte");
        file.sync_all().expect("sync");
    }

    #[test]
    fn torn_tail_in_one_lane_truncates_global_history_at_the_cut() {
        let base = test_base("torn-tail");
        let lane_count = 3usize;
        let total = 12u64;
        {
            let set = FuaWalLaneSet::create(&base, lane_count, 2, SEGMENT_BYTES).expect("create");
            append_round_robin(&set, total);
            set.wait_durable(total).expect("durable");
        }
        // Tear lane 1 at global seq 7 (7 % 3 == 1, frame index 2). Seqs 7 and 10 (its later frames)
        // vanish; the first MISSING global seq is exactly 7.
        corrupt_frame(&base, lane_count, 7);
        let records = recover_lanes(&base, lane_count).expect("recover");
        assert_eq!(
            records.len(),
            7,
            "global history truncates at the first torn seq"
        );
        for (seq, rec) in records.iter().enumerate() {
            assert_eq!(rec, &record(seq as u64));
        }
        // Reopen must FAIL CLOSED: lanes 0/2 hold durable seqs (8, 9, 11) above the cut of 7.
        assert!(
            FuaWalLaneSet::reopen(&base, lane_count, 2, SEGMENT_BYTES).is_err(),
            "reopen over a torn tail with orphans above the cut must fail closed"
        );
        cleanup(&base, lane_count);
    }

    // ---- CutState (the cross-lane merger) unit + property tests ----

    /// Feed a lane's first `durable_count` intervals into the merger (a contiguous durable prefix,
    /// mirroring per-lane frame-contiguous durability) and return nothing — the caller advances.
    fn feed_lane(cut: &mut CutState, intervals: &[(u64, u64)], durable_count: usize) {
        for &(start, end) in &intervals[..durable_count] {
            cut.ingest(start, end);
        }
    }

    #[test]
    fn cut_holds_at_gap_when_a_lane_lags() {
        // Lane 0 owns even seqs, lane 1 owns odd seqs (blocks of 1). Lane 0 is durable through 4;
        // lane 1 is durable through only seq 1 (seq 3 not yet durable) → durable {0,1,2,4}, so the
        // contiguous cut holds at 3 (seq 3 is the first gap).
        let lane0 = vec![(0, 1), (2, 3), (4, 5)];
        let lane1 = vec![(1, 2), (3, 4), (5, 6)];
        let mut cut = CutState::new(0);
        feed_lane(&mut cut, &lane0, 3); // seqs 0,2,4 durable
        feed_lane(&mut cut, &lane1, 1); // only seq 1 durable
        assert_eq!(
            cut.advance(),
            3,
            "held at the gap left by the missing seq 3"
        );
        // Lane 1 catches up one frame (seq 3) → durable {0,1,2,3,4}, cut jumps to 5 (gap at 5).
        feed_lane(&mut cut, &lane1[1..], 1);
        assert_eq!(cut.advance(), 5);
        // Lane 1 fully durable (seq 5) → cut reaches 6.
        feed_lane(&mut cut, &lane1[2..], 1);
        assert_eq!(cut.advance(), 6);
    }

    #[test]
    fn cut_is_order_insensitive() {
        // Ingesting an ahead-of-cut interval first must not wrongly advance the cut.
        let mut cut = CutState::new(0);
        cut.ingest(4, 5);
        cut.ingest(2, 3);
        assert_eq!(cut.advance(), 0, "no interval starts at the cut yet");
        cut.ingest(0, 1);
        assert_eq!(cut.advance(), 1, "0..1 absorbed; 2.. still gapped at seq 1");
        cut.ingest(1, 2);
        assert_eq!(cut.advance(), 3, "1,2 now contiguous; 4.. gapped at seq 3");
        cut.ingest(3, 4);
        assert_eq!(cut.advance(), 5);
    }

    #[test]
    fn property_random_block_claims_match_brute_force_cut() {
        // Deterministic LCG so failures reproduce.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _trial in 0..400 {
            let lane_count = 1 + (next() % 4) as usize;
            let total = 1 + (next() % 60);
            // Tile [0, total) into random blocks, each assigned to a random lane. Per-lane, blocks
            // appear in increasing start (we scan left to right) — the monotonic-claim invariant.
            let mut lane_intervals: Vec<Vec<(u64, u64)>> = vec![Vec::new(); lane_count];
            let mut seq = 0u64;
            while seq < total {
                let remaining = total - seq;
                let block = 1 + next() % remaining.min(4);
                let lane = (next() % lane_count as u64) as usize;
                lane_intervals[lane].push((seq, seq + block));
                seq += block;
            }
            // Random contiguous durable prefix per lane.
            let durable_counts: Vec<usize> = lane_intervals
                .iter()
                .map(|iv| {
                    if iv.is_empty() {
                        0
                    } else {
                        (next() as usize) % (iv.len() + 1)
                    }
                })
                .collect();

            // Merger cut.
            let mut cut = CutState::new(0);
            for (lane, intervals) in lane_intervals.iter().enumerate() {
                feed_lane(&mut cut, intervals, durable_counts[lane]);
            }
            let merged = cut.advance();

            // Brute force: a seq is durable iff it lies in one of its lane's first-k durable blocks.
            let mut durable = vec![false; total as usize];
            for (lane, intervals) in lane_intervals.iter().enumerate() {
                for &(start, end) in &intervals[..durable_counts[lane]] {
                    for s in start..end {
                        durable[s as usize] = true;
                    }
                }
            }
            let brute = durable.iter().take_while(|&&d| d).count() as u64;
            assert_eq!(
                merged, brute,
                "merger cut {merged} != brute {brute}; lanes={lane_intervals:?} durable={durable_counts:?}"
            );
        }
    }
}
