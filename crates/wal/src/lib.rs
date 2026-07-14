use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};

use gpu_db_types::{EngineError, TxnId};

// E1 step 1: the optional FUA fence-pool durability backend (unix-only — it drives a
// `gpu_db_write_conveyor::FuaFrameLog`, an `O_DIRECT|O_DSYNC` construct). Default is OFF; the
// serial `WalDurableCore` path above is unchanged.
#[cfg(unix)]
mod fua;
#[cfg(unix)]
pub use fua::{fua_wal_segments_exist, recover_fua_wal_records};

// E2.5a (Variant 2): N independent ordered WAL lanes whose records carry EXPLICIT global commit
// seqs, with a cross-lane contiguous durable cut and merge recovery. Self-contained here; the
// engine consumer is a later slice. Unix-only (drives the FUA fence-pool lanes).
#[cfg(unix)]
mod fua_lanes;
#[cfg(unix)]
pub use fua_lanes::{
    discover_lane_count, encode_lane_frame_payload, lane_segment_capacity_bytes, recover_lanes,
    recover_lanes_from, remove_stale_lane_files, repair_lane_orphans, repair_lane_orphans_from,
    FuaWalLaneSet,
};

mod archive_timeline;
pub use archive_timeline::{
    apply_wal_archive_timeline_prune, fork_wal_archive_timeline_to_timestamp_micros,
    fork_wal_archive_timeline_to_txn, plan_wal_archive_timeline_prune, read_wal_archive_timeline,
    read_wal_archive_timeline_registry, register_wal_archive_timeline, select_wal_archive_timeline,
    write_wal_archive_timeline, write_wal_archive_timeline_registry, WalArchiveTimeline,
    WalArchiveTimelineBranch, WalArchiveTimelinePrunePlan, WalArchiveTimelineRegistry,
    WalArchiveTimelineRegistryEntry, WalArchiveTimelineSelection,
};

mod buffer;
#[cfg(test)]
use buffer::wal_prealloc_chunk_bytes;
pub use buffer::{WalBuffer, WalDurability, WalGroupFlushBegin, WalGroupFlushJob};

const WAL_SEGMENT_MAGIC: &[u8; 10] = b"GPUDBWAL1\n";
const WAL_CONTROL_MAGIC: &str = "GPUDBWALCONTROL1";
const WAL_ARCHIVE_MANIFEST_MAGIC: &str = "GPUDBWALARCHIVE1";
const WAL_ARCHIVE_OBJECT_BACKUP_MAGIC: &str = "GPUDBWALOBJECTBACKUP1";
const WAL_RECORD_HEADER_LEN: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalRecord {
    pub txn_id: TxnId,
    /// W1a: shared with the replication log entry + the commit-wave item (one allocation per
    /// statement, refcounted; was a fresh `Vec` copy per record on the commit hot path).
    pub payload: std::sync::Arc<[u8]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalCheckpointMeta {
    pub durable_record_count: usize,
    pub last_durable_txn_id: Option<TxnId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalControlFile {
    pub segment_path: PathBuf,
    pub checkpoint: WalCheckpointMeta,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveSegment {
    pub segment_path: PathBuf,
    pub record_count: usize,
    pub first_txn_id: Option<TxnId>,
    pub last_txn_id: Option<TxnId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveManifest {
    pub segments: Vec<WalArchiveSegment>,
    pub checkpoint: WalCheckpointMeta,
    pub record_timestamps: Vec<WalArchiveRecordTimestamp>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalArchiveRecoveryTarget {
    pub target_txn_id: TxnId,
    pub recovered_record_count: usize,
    pub last_recovered_txn_id: TxnId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalArchiveRecordTimestamp {
    pub txn_id: TxnId,
    pub timestamp_micros: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalArchiveTimestampRecoveryTarget {
    pub target_timestamp_micros: u64,
    pub target_txn_id: TxnId,
    pub recovered_record_count: usize,
    pub last_recovered_txn_id: TxnId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveRetentionPlan {
    pub target_txn_id: TxnId,
    pub retained_record_count: usize,
    pub removed_record_count: usize,
    pub retained_manifest: WalArchiveManifest,
    pub removed_segments: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveObject {
    pub source_path: PathBuf,
    pub object_path: PathBuf,
    pub byte_len: u64,
    pub checksum: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveObjectBackup {
    pub archive_manifest: WalArchiveManifest,
    pub objects: Vec<WalArchiveObject>,
}

/// The result of tolerantly reading a live (append-only) WAL segment at recovery time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalSegmentRecovery {
    /// Every record whose bytes were fully written and CRC-valid, in log order.
    pub records: Vec<WalRecord>,
    /// Byte length of the valid prefix (magic + `records`); the file is truncated to this on
    /// recovery install.
    pub valid_bytes: u64,
    /// Bytes discarded beyond the valid prefix (a torn tail from a crash mid-append). Zero on a
    /// clean segment.
    pub discarded_torn_bytes: u64,
}

impl WalSegmentRecovery {
    /// A missing / empty / never-created segment: a fresh durable database.
    pub fn empty() -> Self {
        Self {
            records: Vec::new(),
            valid_bytes: 0,
            discarded_torn_bytes: 0,
        }
    }
}

/// Group-commit accounting for a [`WalBuffer`].
///
/// Each `flush_all` that performs a real fsync batches **all** currently-unflushed records into a
/// single segment write / single fsync — that batch is one *group*. The serialized commit path
/// flushes one record at a time (size-1 groups); the engine's concurrent DML path elects a
/// designated flusher whose [`WalGroupFlushJob`] IO runs lock-free, so committers that append
/// while a group's fsync is in flight coalesce into the NEXT group (`mean_group_size` grows with
/// write concurrency). These counters expose that batching ratio
/// (`durable_records / flush_groups`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WalGroupCommitStats {
    /// Number of `flush_all` calls that performed a real durable fsync (one group each).
    pub flush_groups: u64,
    /// Total records made durable across all groups.
    pub durable_records: u64,
    /// Largest single group (records fsynced by one `flush_all`).
    pub max_group_size: usize,
}

impl WalGroupCommitStats {
    /// Mean records-per-fsync (the group-commit amortization ratio). `0.0` before any flush.
    pub fn mean_group_size(&self) -> f64 {
        if self.flush_groups == 0 {
            0.0
        } else {
            self.durable_records as f64 / self.flush_groups as f64
        }
    }
}

/// Fsync the parent directory of `segment_path` so a freshly-`rename`d segment file's directory
/// entry is durable across a crash (POSIX: an fsync of the file does not guarantee the containing
/// directory entry is persisted). A best-effort no-op on platforms / filesystems that refuse to
/// open a directory for fsync is intentionally NOT done — a hard error here means the existence of
/// the just-written WAL could be lost on crash, which would violate durability, so it propagates.
/// W1b audit fix 1: crash-durability for the checkpoint/control RENAMES — a rename is not
/// durable until the parent directory is fsynced; the rotation must do this BEFORE truncating
/// the live segment, or a strict-POSIX crash can lose the control file (and with it the entire
/// checkpointed prefix, silently) after the truncation survived.
pub fn sync_wal_parent_dir(path: &Path) -> Result<(), EngineError> {
    sync_segment_parent_dir(path)
}

fn sync_segment_parent_dir(segment_path: &Path) -> Result<(), EngineError> {
    let parent = segment_path.parent().filter(|p| !p.as_os_str().is_empty());
    let Some(parent) = parent else {
        // No parent component (e.g. a bare relative file name) — the current working directory is
        // the container; there is nothing portable to fsync, so treat as durable.
        return Ok(());
    };
    let dir = File::open(parent).map_err(|err| {
        EngineError::Durability(format!(
            "failed to open WAL segment directory for fsync {}: {err}",
            parent.display()
        ))
    })?;
    dir.sync_all().map_err(|err| {
        EngineError::Durability(format!(
            "failed to fsync WAL segment directory {}: {err}",
            parent.display()
        ))
    })
}

pub fn write_wal_segment(path: impl AsRef<Path>, records: &[WalRecord]) -> Result<(), EngineError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL segment directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let tmp_path = temporary_segment_path(path);
    let write_result = (|| {
        let mut file = File::create(&tmp_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL segment {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.write_all(WAL_SEGMENT_MAGIC).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write WAL segment header {}: {err}",
                tmp_path.display()
            ))
        })?;
        for record in records {
            write_record(&mut file, record)?;
        }
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync WAL segment {}: {err}",
                tmp_path.display()
            ))
        })?;
        Ok::<_, EngineError>(())
    })();

    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }

    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        EngineError::Durability(format!(
            "failed to install WAL segment {}: {err}",
            path.display()
        ))
    })
}

pub fn read_wal_segment(path: impl AsRef<Path>) -> Result<Vec<WalRecord>, EngineError> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to open WAL segment {}: {err}",
            path.display()
        ))
    })?;
    let mut magic = [0_u8; WAL_SEGMENT_MAGIC.len()];
    file.read_exact(&mut magic).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL segment header {}: {err}",
            path.display()
        ))
    })?;
    if &magic != WAL_SEGMENT_MAGIC {
        return Err(EngineError::Durability(format!(
            "invalid WAL segment header {}",
            path.display()
        )));
    }

    let mut records = Vec::new();
    loop {
        let mut header = [0_u8; WAL_RECORD_HEADER_LEN];
        match file.read(&mut header[..1]) {
            Ok(0) => break,
            Ok(1) => {
                file.read_exact(&mut header[1..]).map_err(|err| {
                    EngineError::Durability(format!(
                        "failed to read WAL segment record header {}: {err}",
                        path.display()
                    ))
                })?;
                let txn_id = u64::from_le_bytes(header[0..8].try_into().expect("txn id bytes"));
                let payload_len =
                    u64::from_le_bytes(header[8..16].try_into().expect("payload len bytes"));
                let expected_checksum =
                    u64::from_le_bytes(header[16..24].try_into().expect("checksum bytes"));
                // W4a: an ALL-ZERO header is the preallocated zero tail — clean end-of-log. It
                // can never be a real record: the FNV checksum of a zero header is nonzero
                // (test-asserted), so (0, 0, 0) is unrepresentable by any valid record.
                if txn_id == 0 && payload_len == 0 && expected_checksum == 0 {
                    break;
                }
                let payload_len = usize::try_from(payload_len).map_err(|_| {
                    EngineError::Durability(format!(
                        "WAL segment {} record payload length is too large",
                        path.display()
                    ))
                })?;
                let mut payload = vec![0_u8; payload_len];
                file.read_exact(&mut payload).map_err(|err| {
                    EngineError::Durability(format!(
                        "failed to read WAL segment payload {}: {err}",
                        path.display()
                    ))
                })?;
                let actual_checksum = wal_record_checksum(txn_id, payload_len as u64, &payload);
                if actual_checksum != expected_checksum {
                    return Err(EngineError::Durability(format!(
                        "WAL segment {} record checksum mismatch for txn {}",
                        path.display(),
                        txn_id
                    )));
                }
                records.push(WalRecord {
                    txn_id,
                    payload: payload.into(),
                });
            }
            Ok(_) => unreachable!("one-byte read returned more than one byte"),
            Err(err) => {
                return Err(EngineError::Durability(format!(
                    "failed to read WAL segment record {}: {err}",
                    path.display()
                )));
            }
        }
    }
    Ok(records)
}

/// W1b — the AUTO-CHECKPOINT path convention: for a live segment `P`, the rolling checkpoint
/// control file is `P.control` and the checkpoint segment is `P.checkpoint`. The engine's
/// auto-rotation writes with these paths and the checkpoint-aware open detects `P.control` to
/// recover checkpoint-then-suffix; a database that never checkpointed has no control file and
/// opens exactly as before.
pub fn wal_checkpoint_control_path(segment_path: &Path) -> PathBuf {
    let file_name = segment_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.segment");
    segment_path.with_file_name(format!("{file_name}.control"))
}

/// See [`wal_checkpoint_control_path`].
pub fn wal_checkpoint_segment_path(segment_path: &Path) -> PathBuf {
    let file_name = segment_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.segment");
    segment_path.with_file_name(format!("{file_name}.checkpoint"))
}

/// E2.5c-2 — the LANES CHECKPOINT (`<base>.lanes-checkpoint` + `<base>.lanes-checkpoint.seg.<cut>`):
/// the database's history up to the checkpoint boundary lives in a GENERATION-pathed checkpoint
/// segment (serial prefix ++ lane records `[0, lane_cut)` in global-merge order); lane logs are
/// then only required to be contiguous FROM `lane_cut` (segments below it may be pruned /
/// recycled). The SIDECAR IS THE SINGLE COMMIT POINT (audit finding: a control-file/sidecar
/// split let a crash between two commit artifacts strand a repeat checkpoint unopenable): it
/// names the segment file it commits to, it is written atomically (temp + rename + parent-dir
/// fsync) only AFTER that segment is durable, each generation writes a NEW segment path (the
/// prior checkpoint is never overwritten), and pruning runs only after the sidecar commit. A
/// crash before the sidecar rename leaves the OLD checkpoint fully intact; after it, the NEW
/// one — there is no intermediate state.
pub fn lanes_checkpoint_sidecar_path(segment_path: &Path) -> PathBuf {
    let file_name = segment_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.segment");
    segment_path.with_file_name(format!("{file_name}.lanes-checkpoint"))
}

/// The generation-pathed checkpoint segment for baseline `lane_cut` (monotonic per database, so
/// successive checkpoints never overwrite each other). See [`lanes_checkpoint_sidecar_path`].
pub fn lanes_checkpoint_segment_path(segment_path: &Path, lane_cut: u64) -> PathBuf {
    let file_name = segment_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.segment");
    segment_path.with_file_name(format!("{file_name}.lanes-checkpoint.seg.{lane_cut}"))
}

const LANES_CHECKPOINT_MAGIC: &str = "gpu-db-lanes-checkpoint v1";

/// A committed lanes checkpoint: the frozen serial prefix length, the lane baseline, and the
/// full merged record history `serial ++ lanes[0, lane_cut)`.
pub struct LanesCheckpoint {
    pub serial_records: u64,
    pub lane_cut: u64,
    pub records: Vec<WalRecord>,
}

/// Commit a lanes checkpoint: write `records` (`== serial ++ lanes[0, lane_cut)`) to the NEW
/// generation segment, fsync it durable, then atomically commit the sidecar naming it, then
/// best-effort remove older generations. See [`lanes_checkpoint_sidecar_path`] for the crash
/// contract. The CALLER prunes lane segments only after this returns.
pub fn write_lanes_checkpoint(
    segment_path: &Path,
    serial_records: u64,
    lane_cut: u64,
    records: &[WalRecord],
) -> Result<(), EngineError> {
    if records.len() as u64 != serial_records + lane_cut {
        return Err(EngineError::Durability(format!(
            "lanes checkpoint of {} holds {} record(s) but declares serial {serial_records} + \
             lane cut {lane_cut}",
            segment_path.display(),
            records.len()
        )));
    }
    let seg_path = lanes_checkpoint_segment_path(segment_path, lane_cut);
    write_wal_segment(&seg_path, records)?; // atomic temp + rename
    sync_wal_parent_dir(&seg_path)?;
    let seg_name = seg_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            EngineError::Durability(format!(
                "lanes checkpoint segment path {} has no file name",
                seg_path.display()
            ))
        })?
        .to_string();
    // THE COMMIT POINT: the sidecar rename. Before it, recovery reads the old checkpoint (or
    // none); after it, the new one.
    let path = lanes_checkpoint_sidecar_path(segment_path);
    let temp = path.with_extension("lanes-checkpoint.tmp");
    let body = format!("{LANES_CHECKPOINT_MAGIC} {serial_records} {lane_cut} {seg_name}\n");
    (|| -> std::io::Result<()> {
        {
            let mut file = fs::File::create(&temp)?;
            std::io::Write::write_all(&mut file, body.as_bytes())?;
            file.sync_all()?;
        }
        fs::rename(&temp, &path)?;
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::File::open(parent)?.sync_all()?;
        }
        Ok(())
    })()
    .map_err(|err| {
        EngineError::Durability(format!(
            "failed to commit lanes checkpoint sidecar {}: {err}",
            path.display()
        ))
    })?;
    // Retire older generations (best-effort; a leftover is re-collected next checkpoint).
    if let (Some(parent), Some(stem)) = (
        segment_path.parent().filter(|p| !p.as_os_str().is_empty()),
        segment_path.file_name().and_then(|n| n.to_str()),
    ) {
        let prefix = format!("{stem}.lanes-checkpoint.seg.");
        if let Ok(entries) = fs::read_dir(parent) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(gen) = name
                        .strip_prefix(&prefix)
                        .and_then(|g| g.parse::<u64>().ok())
                    {
                        if gen < lane_cut {
                            let _ = fs::remove_file(entry.path());
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Read the committed lanes checkpoint: `Ok(None)` when absent (never checkpointed), the full
/// [`LanesCheckpoint`] when present, and a loud error on any malformed/inconsistent state (a
/// committed sidecar whose segment is missing or record-count-inconsistent must never silently
/// re-derive a wrong baseline).
pub fn read_lanes_checkpoint(segment_path: &Path) -> Result<Option<LanesCheckpoint>, EngineError> {
    let path = lanes_checkpoint_sidecar_path(segment_path);
    let content = match fs::read_to_string(&path) {
        Ok(content) => content,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(EngineError::Durability(format!(
                "failed to read lanes checkpoint sidecar {}: {err}",
                path.display()
            )));
        }
    };
    let malformed = || {
        EngineError::Durability(format!(
            "malformed lanes checkpoint sidecar {} (content {content:?})",
            path.display()
        ))
    };
    let rest = content
        .strip_prefix(LANES_CHECKPOINT_MAGIC)
        .ok_or_else(malformed)?;
    let mut fields = rest.split_whitespace();
    let serial_records = fields
        .next()
        .and_then(|f| f.parse::<u64>().ok())
        .ok_or_else(malformed)?;
    let lane_cut = fields
        .next()
        .and_then(|f| f.parse::<u64>().ok())
        .ok_or_else(malformed)?;
    let seg_name = fields.next().ok_or_else(malformed)?.to_string();
    if fields.next().is_some() {
        return Err(malformed());
    }
    let seg_path = path.with_file_name(&seg_name);
    let records = read_wal_segment(&seg_path)?;
    if records.len() as u64 != serial_records + lane_cut {
        return Err(EngineError::Durability(format!(
            "lanes checkpoint segment {} holds {} record(s) but its sidecar commits to serial \
             {serial_records} + lane cut {lane_cut}; refusing an inconsistent checkpoint",
            seg_path.display(),
            records.len()
        )));
    }
    Ok(Some(LanesCheckpoint {
        serial_records,
        lane_cut,
        records,
    }))
}

const WAL_TAIL_MAGIC: &str = "GPUDBWALTAIL1";

/// Path of the durable tail-offset sidecar for a live segment (`<segment>.tail`).
pub fn wal_tail_offset_path(segment_path: &Path) -> PathBuf {
    let file_name = segment_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.segment");
    segment_path.with_file_name(format!("{file_name}.tail"))
}

/// Atomically record the segment's known-durable byte length. Advisory (a LOWER bound for
/// recovery's torn-tail tolerance window), so it is not fsynced — losing it merely widens the
/// window; it never loses data.
fn write_wal_tail_offset(segment_path: &Path, durable_bytes: u64) -> Result<(), EngineError> {
    let path = wal_tail_offset_path(segment_path);
    let tmp_path = temporary_control_path(&path);
    let body = format!("{WAL_TAIL_MAGIC}\n{durable_bytes}\n");
    fs::write(&tmp_path, body).map_err(|err| {
        EngineError::Durability(format!(
            "failed to write WAL tail-offset file {}: {err}",
            tmp_path.display()
        ))
    })?;
    fs::rename(&tmp_path, &path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        EngineError::Durability(format!(
            "failed to install WAL tail-offset file {}: {err}",
            path.display()
        ))
    })
}

/// The recorded durable tail offset, or 0 when the sidecar is missing or unreadable (pure
/// torn-tail-tolerant recovery — the safe direction for an advisory lower bound).
fn read_wal_tail_offset(segment_path: &Path) -> u64 {
    let path = wal_tail_offset_path(segment_path);
    let Ok(body) = fs::read_to_string(&path) else {
        return 0;
    };
    let mut lines = body.lines();
    if lines.next() != Some(WAL_TAIL_MAGIC) {
        return 0;
    }
    lines.next().and_then(|raw| raw.parse().ok()).unwrap_or(0)
}

/// Tolerantly read a LIVE (append-only) segment at recovery time.
///
/// Unlike the strict [`read_wal_segment`] (for checkpoint/archive segments, which are written
/// atomically and must be intact end-to-end), a live segment can legitimately end in a torn
/// record: a crash between an append's `write_all` and its fsync acknowledgment. Such a record
/// was never acknowledged as committed, so it is safe — and required — to truncate it away.
///
/// The durable tail-offset sidecar bounds how far that tolerance reaches: an invalid region
/// starting AT or BEYOND the recorded offset is a torn tail (recovered records so far are
/// returned, with `valid_bytes` marking the truncation boundary); an invalid record starting
/// BELOW it means acknowledged-durable data is damaged (bit rot, external truncation), which
/// fails loudly with the same error the strict reader would raise.
pub fn recover_wal_segment(path: impl AsRef<Path>) -> Result<WalSegmentRecovery, EngineError> {
    let path = path.as_ref();
    let recorded_tail = read_wal_tail_offset(path);
    // Every non-loud return must reach at least the recorded durable tail: a segment that ends
    // CLEANLY short of it (external truncation, a lost/foreign file next to a live sidecar) has
    // lost acknowledged-durable records and must fail loudly, exactly like below-tail corruption.
    let ends_short = |valid_bytes: u64| {
        EngineError::Durability(format!(
            "WAL segment {} ends at byte {valid_bytes}, before the recorded durable tail offset \
             {recorded_tail}",
            path.display()
        ))
    };
    if !path.exists() {
        if recorded_tail > 0 {
            return Err(ends_short(0));
        }
        return Ok(WalSegmentRecovery::empty());
    }
    let bytes = fs::read(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL segment {}: {err}",
            path.display()
        ))
    })?;
    if bytes.len() < WAL_SEGMENT_MAGIC.len() {
        // A crash during segment creation (mid-magic write) leaves a short prefix of the magic;
        // treat it as a fresh database. Anything else short is a foreign file — refuse to clobber.
        if WAL_SEGMENT_MAGIC.starts_with(bytes.as_slice()) {
            if recorded_tail > 0 {
                return Err(ends_short(0));
            }
            return Ok(WalSegmentRecovery {
                records: Vec::new(),
                valid_bytes: 0,
                discarded_torn_bytes: bytes.len() as u64,
            });
        }
        return Err(EngineError::Durability(format!(
            "invalid WAL segment header {}",
            path.display()
        )));
    }
    if &bytes[..WAL_SEGMENT_MAGIC.len()] != WAL_SEGMENT_MAGIC {
        return Err(EngineError::Durability(format!(
            "invalid WAL segment header {}",
            path.display()
        )));
    }
    let mut records = Vec::new();
    let mut offset = WAL_SEGMENT_MAGIC.len();
    while offset < bytes.len() {
        let record_start = offset;
        let torn = |records: Vec<WalRecord>| {
            Ok(WalSegmentRecovery {
                records,
                valid_bytes: record_start as u64,
                discarded_torn_bytes: (bytes.len() - record_start) as u64,
            })
        };
        let below_recorded_tail = (record_start as u64) < recorded_tail;
        if bytes.len() - record_start < WAL_RECORD_HEADER_LEN {
            if below_recorded_tail {
                return Err(EngineError::Durability(format!(
                    "failed to read WAL segment record header {}: acknowledged-durable record is \
                     truncated at byte {record_start}",
                    path.display()
                )));
            }
            return torn(records);
        }
        let header = &bytes[record_start..record_start + WAL_RECORD_HEADER_LEN];
        let txn_id = u64::from_le_bytes(header[0..8].try_into().expect("txn id bytes"));
        let payload_len = u64::from_le_bytes(header[8..16].try_into().expect("payload len bytes"));
        let expected_checksum =
            u64::from_le_bytes(header[16..24].try_into().expect("checksum bytes"));
        // W4a: an all-zero header starts the preallocated zero tail. If the ENTIRE remainder is
        // zeros this is a CLEAN end-of-log (discarded_torn_bytes = 0); any non-zero byte in the
        // remainder is a genuine torn tail and takes the torn path below. An all-zero header can
        // never be a real record (the FNV checksum of a zero header is nonzero, test-asserted),
        // and acknowledged-durable records live below `recorded_tail`, which the zero region
        // never reaches (`below_recorded_tail` would fail loudly first).
        if txn_id == 0
            && payload_len == 0
            && expected_checksum == 0
            && !below_recorded_tail
            && bytes[record_start..].iter().all(|&b| b == 0)
        {
            return Ok(WalSegmentRecovery {
                records,
                valid_bytes: record_start as u64,
                discarded_torn_bytes: 0,
            });
        }
        let payload_start = record_start + WAL_RECORD_HEADER_LEN;
        let payload_end = usize::try_from(payload_len)
            .ok()
            .and_then(|len| payload_start.checked_add(len));
        let Some(payload_end) = payload_end.filter(|end| *end <= bytes.len()) else {
            if below_recorded_tail {
                return Err(EngineError::Durability(format!(
                    "failed to read WAL segment payload {}: acknowledged-durable record is \
                     truncated at byte {record_start}",
                    path.display()
                )));
            }
            return torn(records);
        };
        let payload = &bytes[payload_start..payload_end];
        let actual_checksum = wal_record_checksum(txn_id, payload_len, payload);
        if actual_checksum != expected_checksum {
            if below_recorded_tail {
                return Err(EngineError::Durability(format!(
                    "WAL segment {} record checksum mismatch for txn {}",
                    path.display(),
                    txn_id
                )));
            }
            return torn(records);
        }
        records.push(WalRecord {
            txn_id,
            payload: payload.to_vec().into(),
        });
        offset = payload_end;
    }
    if (offset as u64) < recorded_tail {
        return Err(ends_short(offset as u64));
    }
    Ok(WalSegmentRecovery {
        records,
        valid_bytes: offset as u64,
        discarded_torn_bytes: 0,
    })
}

pub fn write_wal_control_file(
    path: impl AsRef<Path>,
    control: &WalControlFile,
) -> Result<(), EngineError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL control directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let tmp_path = temporary_control_path(path);
    let last_txn = control
        .checkpoint
        .last_durable_txn_id
        .map(|txn_id| txn_id.to_string())
        .unwrap_or_else(|| "none".to_string());
    let body = format!(
        "{WAL_CONTROL_MAGIC}\nsegment={}\ndurable_record_count={}\nlast_durable_txn_id={last_txn}\n",
        control.segment_path.display(),
        control.checkpoint.durable_record_count,
    );

    let write_result = (|| {
        let mut file = File::create(&tmp_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL control file {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.write_all(body.as_bytes()).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write WAL control file {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync WAL control file {}: {err}",
                tmp_path.display()
            ))
        })?;
        Ok::<_, EngineError>(())
    })();

    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }

    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        EngineError::Durability(format!(
            "failed to install WAL control file {}: {err}",
            path.display()
        ))
    })
}

pub fn read_wal_control_file(path: impl AsRef<Path>) -> Result<WalControlFile, EngineError> {
    let path = path.as_ref();
    let body = fs::read_to_string(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL control file {}: {err}",
            path.display()
        ))
    })?;
    let mut lines = body.lines();
    if lines.next() != Some(WAL_CONTROL_MAGIC) {
        return Err(EngineError::Durability(format!(
            "invalid WAL control header {}",
            path.display()
        )));
    }

    let segment_path = parse_control_value(lines.next(), "segment", path).map(PathBuf::from)?;
    let durable_record_count = parse_control_value(lines.next(), "durable_record_count", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL control durable_record_count {}: {err}",
                path.display()
            ))
        })?;
    let last_durable_txn_id = match parse_control_value(lines.next(), "last_durable_txn_id", path)?
    {
        "none" => None,
        raw => Some(raw.parse().map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL control last_durable_txn_id {}: {err}",
                path.display()
            ))
        })?),
    };

    Ok(WalControlFile {
        segment_path,
        checkpoint: WalCheckpointMeta {
            durable_record_count,
            last_durable_txn_id,
        },
    })
}

pub fn read_wal_checkpoint(
    control_path: impl AsRef<Path>,
) -> Result<(WalControlFile, Vec<WalRecord>), EngineError> {
    let control_path = control_path.as_ref();
    let control = read_wal_control_file(control_path)?;
    let segment_path = if control.segment_path.is_absolute() {
        control.segment_path.clone()
    } else {
        control_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&control.segment_path)
    };
    let mut records = read_wal_segment(&segment_path)?;
    // W1b audit fix 2: the rotation renames the checkpoint segment BEFORE the control file; a
    // crash between them leaves a NEWER (longer) checkpoint paired with the previous control.
    // The CONTROL FILE is the commit point — the checkpoint's extra tail records were never
    // committed as a checkpoint, but every one of them is still covered by the (untruncated)
    // live segment, so truncating the LIST to the control's count recovers exactly the
    // committed state. A SHORTER checkpoint than the control commits to remains a loud error
    // (acknowledged checkpoint data is missing).
    if records.len() > control.checkpoint.durable_record_count {
        records.truncate(control.checkpoint.durable_record_count);
    }
    validate_checkpoint_control(control_path, &control, &records)?;
    Ok((control, records))
}

pub fn write_wal_archive(
    manifest_path: impl AsRef<Path>,
    segment_dir: impl AsRef<Path>,
    records: &[WalRecord],
    records_per_segment: usize,
) -> Result<WalArchiveManifest, EngineError> {
    write_wal_archive_with_timestamps(
        manifest_path,
        segment_dir,
        records,
        records_per_segment,
        &[],
    )
}

pub fn write_wal_archive_with_timestamps(
    manifest_path: impl AsRef<Path>,
    segment_dir: impl AsRef<Path>,
    records: &[WalRecord],
    records_per_segment: usize,
    record_timestamps: &[WalArchiveRecordTimestamp],
) -> Result<WalArchiveManifest, EngineError> {
    if records_per_segment == 0 {
        return Err(EngineError::Durability(
            "WAL archive records_per_segment must be non-zero".to_string(),
        ));
    }

    let manifest_path = manifest_path.as_ref();
    let segment_dir = segment_dir.as_ref();
    validate_timestamp_metadata(manifest_path, records, record_timestamps)?;
    fs::create_dir_all(segment_dir).map_err(|err| {
        EngineError::Durability(format!(
            "failed to create WAL archive segment directory {}: {err}",
            segment_dir.display()
        ))
    })?;

    let mut segments = Vec::new();
    for (index, chunk) in records.chunks(records_per_segment).enumerate() {
        let file_name = format!("segment-{:04}.wal", index + 1);
        let segment_path = segment_dir.join(&file_name);
        write_wal_segment(&segment_path, chunk)?;
        let manifest_segment_path = segment_path
            .strip_prefix(manifest_path.parent().unwrap_or_else(|| Path::new(".")))
            .unwrap_or(&segment_path)
            .to_path_buf();
        segments.push(WalArchiveSegment {
            segment_path: manifest_segment_path,
            record_count: chunk.len(),
            first_txn_id: chunk.first().map(|record| record.txn_id),
            last_txn_id: chunk.last().map(|record| record.txn_id),
        });
    }

    let manifest = WalArchiveManifest {
        segments,
        checkpoint: WalCheckpointMeta {
            durable_record_count: records.len(),
            last_durable_txn_id: records.last().map(|record| record.txn_id),
        },
        record_timestamps: record_timestamps.to_vec(),
    };
    write_wal_archive_manifest(manifest_path, &manifest)?;
    Ok(manifest)
}

pub fn append_wal_archive_segment(
    manifest_path: impl AsRef<Path>,
    segment_path: impl AsRef<Path>,
) -> Result<WalArchiveManifest, EngineError> {
    append_wal_archive_segment_with_timestamps(manifest_path, segment_path, &[])
}

pub fn append_wal_archive_segment_with_timestamps(
    manifest_path: impl AsRef<Path>,
    segment_path: impl AsRef<Path>,
    record_timestamps: &[WalArchiveRecordTimestamp],
) -> Result<WalArchiveManifest, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let segment_path = segment_path.as_ref();
    let (manifest, mut records) = read_wal_archive(manifest_path)?;
    let segment_records = read_wal_segment(segment_path)?;
    if segment_records.is_empty() {
        return Err(EngineError::Durability(format!(
            "WAL archive {} cannot ingest empty segment {}",
            manifest_path.display(),
            segment_path.display()
        )));
    }

    validate_archive_ingest_timestamps(
        manifest_path,
        &manifest,
        &segment_records,
        record_timestamps,
    )?;
    validate_archive_ingest_continuity(manifest_path, &manifest, &segment_records)?;

    let appended_record_count = segment_records.len();
    let appended_first_txn_id = segment_records.first().map(|record| record.txn_id);
    let appended_last_txn_id = segment_records.last().map(|record| record.txn_id);
    records.extend(segment_records);
    let mut combined_timestamps = manifest.record_timestamps.clone();
    combined_timestamps.extend_from_slice(record_timestamps);
    validate_timestamp_metadata(manifest_path, &records, &combined_timestamps)?;

    let manifest_segment_path = segment_path
        .strip_prefix(manifest_path.parent().unwrap_or_else(|| Path::new(".")))
        .unwrap_or(segment_path)
        .to_path_buf();
    let mut segments = manifest.segments;
    segments.push(WalArchiveSegment {
        segment_path: manifest_segment_path,
        record_count: appended_record_count,
        first_txn_id: appended_first_txn_id,
        last_txn_id: appended_last_txn_id,
    });

    let appended = WalArchiveManifest {
        segments,
        checkpoint: WalCheckpointMeta {
            durable_record_count: records.len(),
            last_durable_txn_id: records.last().map(|record| record.txn_id),
        },
        record_timestamps: combined_timestamps,
    };
    validate_archive_manifest_shape(manifest_path, &appended)?;
    validate_archive_records(manifest_path, &appended, &records)?;
    validate_archive_timestamps(manifest_path, &appended, &records)?;
    write_wal_archive_manifest(manifest_path, &appended)?;
    Ok(appended)
}

pub fn export_wal_archive_object_backup(
    manifest_path: impl AsRef<Path>,
    backup_manifest_path: impl AsRef<Path>,
    object_dir: impl AsRef<Path>,
) -> Result<WalArchiveObjectBackup, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let backup_manifest_path = backup_manifest_path.as_ref();
    let object_dir = object_dir.as_ref();
    let (archive_manifest, _records) = read_wal_archive(manifest_path)?;

    fs::create_dir_all(object_dir).map_err(|err| {
        EngineError::Durability(format!(
            "failed to create WAL archive object directory {}: {err}",
            object_dir.display()
        ))
    })?;

    let mut objects = Vec::with_capacity(archive_manifest.segments.len() + 1);
    objects.push(write_wal_archive_backup_object(
        backup_manifest_path,
        Path::new("MANIFEST"),
        manifest_path,
        &object_dir.join("archive-manifest.object"),
    )?);
    for (idx, segment) in archive_manifest.segments.iter().enumerate() {
        let segment_source_path = resolve_manifest_path(manifest_path, &segment.segment_path);
        let object_path = object_dir.join(format!("segment-{:04}.wal.object", idx + 1));
        objects.push(write_wal_archive_backup_object(
            backup_manifest_path,
            &segment.segment_path,
            &segment_source_path,
            &object_path,
        )?);
    }

    let backup = WalArchiveObjectBackup {
        archive_manifest,
        objects,
    };
    write_wal_archive_object_backup_manifest(backup_manifest_path, &backup)?;
    Ok(backup)
}

pub fn restore_wal_archive_object_backup(
    backup_manifest_path: impl AsRef<Path>,
    restored_manifest_path: impl AsRef<Path>,
    restored_segment_dir: impl AsRef<Path>,
) -> Result<WalArchiveManifest, EngineError> {
    let backup_manifest_path = backup_manifest_path.as_ref();
    let restored_manifest_path = restored_manifest_path.as_ref();
    let restored_segment_dir = restored_segment_dir.as_ref();
    let backup = read_wal_archive_object_backup_manifest(backup_manifest_path)?;
    validate_archive_manifest_shape(backup_manifest_path, &backup.archive_manifest)?;

    let manifest_object = backup
        .objects
        .iter()
        .find(|object| object.source_path == Path::new("MANIFEST"))
        .ok_or_else(|| {
            EngineError::Durability(format!(
                "WAL archive object backup {} has no manifest object",
                backup_manifest_path.display()
            ))
        })?;
    let manifest_bytes =
        read_verified_wal_archive_backup_object(backup_manifest_path, manifest_object)?;
    let expected_manifest_bytes =
        render_wal_archive_manifest_body(&backup.archive_manifest)?.into_bytes();
    if manifest_bytes != expected_manifest_bytes {
        return Err(EngineError::Durability(format!(
            "WAL archive object backup {} manifest object does not match backup manifest metadata",
            backup_manifest_path.display()
        )));
    }

    if restored_segment_dir.exists() {
        return Err(EngineError::Durability(format!(
            "restored WAL archive segment directory {} already exists",
            restored_segment_dir.display()
        )));
    }
    let restored_segment_parent = restored_segment_dir
        .parent()
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(restored_segment_parent).map_err(|err| {
        EngineError::Durability(format!(
            "failed to create restored WAL archive segment parent {}: {err}",
            restored_segment_parent.display()
        ))
    })?;
    let restored_segment_name = restored_segment_dir
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "segments".to_string());
    let staging_segment_dir = restored_segment_parent.join(format!(
        ".{restored_segment_name}.restore-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&staging_segment_dir);
    fs::create_dir_all(&staging_segment_dir).map_err(|err| {
        EngineError::Durability(format!(
            "failed to create staging WAL archive segment directory {}: {err}",
            staging_segment_dir.display()
        ))
    })?;

    let restore_result = (|| {
        let mut restored_segments = Vec::with_capacity(backup.archive_manifest.segments.len());
        for (idx, segment) in backup.archive_manifest.segments.iter().enumerate() {
            let object = backup
                .objects
                .iter()
                .find(|object| object.source_path == segment.segment_path)
                .ok_or_else(|| {
                    EngineError::Durability(format!(
                        "WAL archive object backup {} missing segment object {}",
                        backup_manifest_path.display(),
                        segment.segment_path.display()
                    ))
                })?;
            let bytes = read_verified_wal_archive_backup_object(backup_manifest_path, object)?;
            let final_segment_path =
                restored_segment_dir.join(format!("segment-{:04}.wal", idx + 1));
            let staging_segment_path =
                staging_segment_dir.join(format!("segment-{:04}.wal", idx + 1));
            write_verified_backup_bytes(&staging_segment_path, &bytes)?;
            let manifest_segment_path = final_segment_path
                .strip_prefix(
                    restored_manifest_path
                        .parent()
                        .unwrap_or_else(|| Path::new(".")),
                )
                .unwrap_or(&final_segment_path)
                .to_path_buf();
            restored_segments.push(WalArchiveSegment {
                segment_path: manifest_segment_path,
                record_count: segment.record_count,
                first_txn_id: segment.first_txn_id,
                last_txn_id: segment.last_txn_id,
            });
        }

        fs::rename(&staging_segment_dir, restored_segment_dir).map_err(|err| {
            EngineError::Durability(format!(
                "failed to install restored WAL archive segment directory {}: {err}",
                restored_segment_dir.display()
            ))
        })?;
        Ok::<_, EngineError>(restored_segments)
    })();
    let restored_segments = match restore_result {
        Ok(restored_segments) => restored_segments,
        Err(err) => {
            let _ = fs::remove_dir_all(&staging_segment_dir);
            return Err(err);
        }
    };

    let restored_manifest = WalArchiveManifest {
        segments: restored_segments,
        checkpoint: backup.archive_manifest.checkpoint,
        record_timestamps: backup.archive_manifest.record_timestamps,
    };
    write_wal_archive_manifest(restored_manifest_path, &restored_manifest)?;
    let (validated_manifest, _records) = read_wal_archive(restored_manifest_path)?;
    Ok(validated_manifest)
}

pub fn write_wal_archive_object_backup_manifest(
    path: impl AsRef<Path>,
    backup: &WalArchiveObjectBackup,
) -> Result<(), EngineError> {
    let path = path.as_ref();
    validate_archive_manifest_shape(path, &backup.archive_manifest)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive object backup directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let manifest = &backup.archive_manifest;
    let mut body = format!(
        "{WAL_ARCHIVE_OBJECT_BACKUP_MAGIC}\ndurable_record_count={}\nlast_durable_txn_id={}\nsegments={}\n",
        manifest.checkpoint.durable_record_count,
        format_optional_txn(manifest.checkpoint.last_durable_txn_id),
        manifest.segments.len()
    );
    for segment in &manifest.segments {
        validate_backup_path(path, "segment", &segment.segment_path)?;
        body.push_str(&format!(
            "segment={}|{}|{}|{}\n",
            segment.segment_path.display(),
            segment.record_count,
            format_optional_txn(segment.first_txn_id),
            format_optional_txn(segment.last_txn_id)
        ));
    }
    body.push_str(&format!(
        "record_timestamps={}\n",
        manifest.record_timestamps.len()
    ));
    for timestamp in &manifest.record_timestamps {
        body.push_str(&format!(
            "record_timestamp={}|{}\n",
            timestamp.txn_id, timestamp.timestamp_micros
        ));
    }
    body.push_str(&format!("objects={}\n", backup.objects.len()));
    for object in &backup.objects {
        validate_backup_path(path, "object source", &object.source_path)?;
        validate_backup_path(path, "object path", &object.object_path)?;
        body.push_str(&format!(
            "object={}|{}|{}|{}\n",
            object.source_path.display(),
            object.object_path.display(),
            object.byte_len,
            object.checksum
        ));
    }

    let tmp_path = temporary_control_path(path);
    let write_result = (|| {
        let mut file = File::create(&tmp_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive object backup {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.write_all(body.as_bytes()).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write WAL archive object backup {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync WAL archive object backup {}: {err}",
                tmp_path.display()
            ))
        })?;
        Ok::<_, EngineError>(())
    })();

    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }

    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        EngineError::Durability(format!(
            "failed to install WAL archive object backup {}: {err}",
            path.display()
        ))
    })
}

pub fn read_wal_archive_object_backup_manifest(
    path: impl AsRef<Path>,
) -> Result<WalArchiveObjectBackup, EngineError> {
    let path = path.as_ref();
    let body = fs::read_to_string(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL archive object backup {}: {err}",
            path.display()
        ))
    })?;
    let mut lines = body.lines();
    if lines.next() != Some(WAL_ARCHIVE_OBJECT_BACKUP_MAGIC) {
        return Err(EngineError::Durability(format!(
            "invalid WAL archive object backup header {}",
            path.display()
        )));
    }

    let durable_record_count = parse_control_value(lines.next(), "durable_record_count", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive object backup durable_record_count {}: {err}",
                path.display()
            ))
        })?;
    let last_durable_txn_id = parse_optional_txn(
        parse_control_value(lines.next(), "last_durable_txn_id", path)?,
        "last_durable_txn_id",
        path,
    )?;
    let segment_count: usize = parse_control_value(lines.next(), "segments", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive object backup segments {}: {err}",
                path.display()
            ))
        })?;
    let mut segments = Vec::with_capacity(segment_count);
    for _ in 0..segment_count {
        let raw = parse_control_value(lines.next(), "segment", path)?;
        let mut parts = raw.split('|');
        let segment_path = PathBuf::from(parts.next().ok_or_else(|| {
            EngineError::Durability(format!(
                "missing WAL archive object backup segment path in {}",
                path.display()
            ))
        })?);
        let record_count = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive object backup segment count in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive object backup segment count {}: {err}",
                    path.display()
                ))
            })?;
        let first_txn_id = parse_optional_txn(
            parts.next().ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive object backup segment first txn in {}",
                    path.display()
                ))
            })?,
            "segment first txn",
            path,
        )?;
        let last_txn_id = parse_optional_txn(
            parts.next().ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive object backup segment last txn in {}",
                    path.display()
                ))
            })?,
            "segment last txn",
            path,
        )?;
        if parts.next().is_some() {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive object backup segment field count in {}",
                path.display()
            )));
        }
        segments.push(WalArchiveSegment {
            segment_path,
            record_count,
            first_txn_id,
            last_txn_id,
        });
    }

    let timestamp_count: usize = parse_control_value(lines.next(), "record_timestamps", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive object backup timestamp count {}: {err}",
                path.display()
            ))
        })?;
    let mut record_timestamps = Vec::with_capacity(timestamp_count);
    for _ in 0..timestamp_count {
        let raw = parse_control_value(lines.next(), "record_timestamp", path)?;
        let mut parts = raw.split('|');
        let txn_id = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive object backup timestamp txn in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive object backup timestamp txn {}: {err}",
                    path.display()
                ))
            })?;
        let timestamp_micros = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive object backup timestamp value in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive object backup timestamp value {}: {err}",
                    path.display()
                ))
            })?;
        if parts.next().is_some() {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive object backup timestamp field count in {}",
                path.display()
            )));
        }
        record_timestamps.push(WalArchiveRecordTimestamp {
            txn_id,
            timestamp_micros,
        });
    }

    let object_count: usize = parse_control_value(lines.next(), "objects", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive object backup object count {}: {err}",
                path.display()
            ))
        })?;
    let mut objects = Vec::with_capacity(object_count);
    for _ in 0..object_count {
        let raw = parse_control_value(lines.next(), "object", path)?;
        let mut parts = raw.split('|');
        let source_path = PathBuf::from(parts.next().ok_or_else(|| {
            EngineError::Durability(format!(
                "missing WAL archive object backup source path in {}",
                path.display()
            ))
        })?);
        let object_path = PathBuf::from(parts.next().ok_or_else(|| {
            EngineError::Durability(format!(
                "missing WAL archive object backup object path in {}",
                path.display()
            ))
        })?);
        let byte_len = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive object backup byte length in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive object backup byte length {}: {err}",
                    path.display()
                ))
            })?;
        let checksum = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive object backup checksum in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive object backup checksum {}: {err}",
                    path.display()
                ))
            })?;
        if parts.next().is_some() {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive object backup object field count in {}",
                path.display()
            )));
        }
        objects.push(WalArchiveObject {
            source_path,
            object_path,
            byte_len,
            checksum,
        });
    }
    if lines.next().is_some() {
        return Err(EngineError::Durability(format!(
            "unexpected trailing WAL archive object backup data in {}",
            path.display()
        )));
    }

    let archive_manifest = WalArchiveManifest {
        segments,
        checkpoint: WalCheckpointMeta {
            durable_record_count,
            last_durable_txn_id,
        },
        record_timestamps,
    };
    validate_archive_manifest_shape(path, &archive_manifest)?;
    Ok(WalArchiveObjectBackup {
        archive_manifest,
        objects,
    })
}

pub fn write_wal_archive_manifest(
    path: impl AsRef<Path>,
    manifest: &WalArchiveManifest,
) -> Result<(), EngineError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive manifest directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let body = render_wal_archive_manifest_body(manifest)?;

    let tmp_path = temporary_control_path(path);
    let write_result = (|| {
        let mut file = File::create(&tmp_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive manifest {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.write_all(body.as_bytes()).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write WAL archive manifest {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync WAL archive manifest {}: {err}",
                tmp_path.display()
            ))
        })?;
        Ok::<_, EngineError>(())
    })();

    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }

    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        EngineError::Durability(format!(
            "failed to install WAL archive manifest {}: {err}",
            path.display()
        ))
    })
}

fn render_wal_archive_manifest_body(manifest: &WalArchiveManifest) -> Result<String, EngineError> {
    let mut body = format!(
        "{WAL_ARCHIVE_MANIFEST_MAGIC}\ndurable_record_count={}\nlast_durable_txn_id={}\nsegments={}\n",
        manifest.checkpoint.durable_record_count,
        format_optional_txn(manifest.checkpoint.last_durable_txn_id),
        manifest.segments.len()
    );
    for segment in &manifest.segments {
        if segment.segment_path.to_string_lossy().contains('|') {
            return Err(EngineError::Durability(format!(
                "WAL archive segment path contains unsupported delimiter: {}",
                segment.segment_path.display()
            )));
        }
        body.push_str(&format!(
            "segment={}|{}|{}|{}\n",
            segment.segment_path.display(),
            segment.record_count,
            format_optional_txn(segment.first_txn_id),
            format_optional_txn(segment.last_txn_id)
        ));
    }
    for timestamp in &manifest.record_timestamps {
        body.push_str(&format!(
            "record_timestamp={}|{}\n",
            timestamp.txn_id, timestamp.timestamp_micros
        ));
    }
    Ok(body)
}

pub fn read_wal_archive_manifest(
    path: impl AsRef<Path>,
) -> Result<WalArchiveManifest, EngineError> {
    let path = path.as_ref();
    let body = fs::read_to_string(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL archive manifest {}: {err}",
            path.display()
        ))
    })?;
    let mut lines = body.lines();
    if lines.next() != Some(WAL_ARCHIVE_MANIFEST_MAGIC) {
        return Err(EngineError::Durability(format!(
            "invalid WAL archive manifest header {}",
            path.display()
        )));
    }

    let durable_record_count = parse_control_value(lines.next(), "durable_record_count", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive durable_record_count {}: {err}",
                path.display()
            ))
        })?;
    let last_durable_txn_id = parse_optional_txn(
        parse_control_value(lines.next(), "last_durable_txn_id", path)?,
        "last_durable_txn_id",
        path,
    )?;
    let segment_count: usize = parse_control_value(lines.next(), "segments", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive segments {}: {err}",
                path.display()
            ))
        })?;

    let mut segments = Vec::with_capacity(segment_count);
    for _ in 0..segment_count {
        let raw = parse_control_value(lines.next(), "segment", path)?;
        let mut parts = raw.split('|');
        let segment_path = parts.next().ok_or_else(|| {
            EngineError::Durability(format!("invalid WAL archive segment in {}", path.display()))
        })?;
        let record_count = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive segment record count in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive segment record count {}: {err}",
                    path.display()
                ))
            })?;
        let first_txn_id = parse_optional_txn(
            parts.next().ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive segment first txn in {}",
                    path.display()
                ))
            })?,
            "segment first txn",
            path,
        )?;
        let last_txn_id = parse_optional_txn(
            parts.next().ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive segment last txn in {}",
                    path.display()
                ))
            })?,
            "segment last txn",
            path,
        )?;
        if parts.next().is_some() {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive segment field count in {}",
                path.display()
            )));
        }
        segments.push(WalArchiveSegment {
            segment_path: PathBuf::from(segment_path),
            record_count,
            first_txn_id,
            last_txn_id,
        });
    }
    let mut record_timestamps = Vec::new();
    for line in lines {
        let raw = parse_control_value(Some(line), "record_timestamp", path)?;
        let mut parts = raw.split('|');
        let txn_id = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive record timestamp txn in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive record timestamp txn {}: {err}",
                    path.display()
                ))
            })?;
        let timestamp_micros = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive record timestamp value in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive record timestamp value {}: {err}",
                    path.display()
                ))
            })?;
        if parts.next().is_some() {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive record timestamp field count in {}",
                path.display()
            )));
        }
        record_timestamps.push(WalArchiveRecordTimestamp {
            txn_id,
            timestamp_micros,
        });
    }

    let manifest = WalArchiveManifest {
        segments,
        checkpoint: WalCheckpointMeta {
            durable_record_count,
            last_durable_txn_id,
        },
        record_timestamps,
    };
    validate_archive_manifest_shape(path, &manifest)?;
    Ok(manifest)
}

pub fn read_wal_archive(
    manifest_path: impl AsRef<Path>,
) -> Result<(WalArchiveManifest, Vec<WalRecord>), EngineError> {
    let manifest_path = manifest_path.as_ref();
    let manifest = read_wal_archive_manifest(manifest_path)?;
    let mut records = Vec::new();
    for segment in &manifest.segments {
        let segment_path = resolve_manifest_path(manifest_path, &segment.segment_path);
        let segment_records = read_wal_segment(&segment_path)?;
        validate_archive_segment(manifest_path, segment, &segment_records)?;
        records.extend(segment_records);
    }
    validate_archive_records(manifest_path, &manifest, &records)?;
    validate_archive_timestamps(manifest_path, &manifest, &records)?;
    Ok((manifest, records))
}

pub fn read_wal_archive_to_txn(
    manifest_path: impl AsRef<Path>,
    target_txn_id: TxnId,
) -> Result<(WalArchiveManifest, WalArchiveRecoveryTarget, Vec<WalRecord>), EngineError> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, records) = read_wal_archive(manifest_path)?;
    let Some(first_txn_id) = records.first().map(|record| record.txn_id) else {
        return Err(EngineError::Durability(format!(
            "WAL archive {} has no records for target transaction {}",
            manifest_path.display(),
            target_txn_id
        )));
    };
    if target_txn_id < first_txn_id {
        return Err(EngineError::Durability(format!(
            "WAL archive {} target transaction {} is before first archived transaction {}",
            manifest_path.display(),
            target_txn_id,
            first_txn_id
        )));
    }
    if target_txn_id > manifest.checkpoint.last_durable_txn_id.unwrap_or(0) {
        return Err(EngineError::Durability(format!(
            "WAL archive {} target transaction {} is beyond last durable transaction {:?}",
            manifest_path.display(),
            target_txn_id,
            manifest.checkpoint.last_durable_txn_id
        )));
    }

    let recovered_record_count = records
        .iter()
        .take_while(|record| record.txn_id <= target_txn_id)
        .count();
    let last_recovered_txn_id = records
        .get(recovered_record_count.saturating_sub(1))
        .map(|record| record.txn_id);
    if last_recovered_txn_id != Some(target_txn_id) {
        return Err(EngineError::Durability(format!(
            "WAL archive {} does not contain target transaction {}",
            manifest_path.display(),
            target_txn_id
        )));
    }

    let target = WalArchiveRecoveryTarget {
        target_txn_id,
        recovered_record_count,
        last_recovered_txn_id: target_txn_id,
    };
    Ok((
        manifest,
        target,
        records.into_iter().take(recovered_record_count).collect(),
    ))
}

pub fn read_wal_archive_to_timestamp_micros(
    manifest_path: impl AsRef<Path>,
    target_timestamp_micros: u64,
) -> Result<
    (
        WalArchiveManifest,
        WalArchiveTimestampRecoveryTarget,
        Vec<WalRecord>,
    ),
    EngineError,
> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, records) = read_wal_archive(manifest_path)?;
    if records.is_empty() {
        return Err(EngineError::Durability(format!(
            "WAL archive {} has no records for target timestamp {}",
            manifest_path.display(),
            target_timestamp_micros
        )));
    }
    if manifest.record_timestamps.is_empty() {
        return Err(EngineError::Durability(format!(
            "WAL archive {} has no timestamp metadata for target timestamp {}",
            manifest_path.display(),
            target_timestamp_micros
        )));
    }

    let first_timestamp = manifest
        .record_timestamps
        .first()
        .expect("non-empty timestamp metadata")
        .timestamp_micros;
    let last_timestamp = manifest
        .record_timestamps
        .last()
        .expect("non-empty timestamp metadata")
        .timestamp_micros;
    if target_timestamp_micros < first_timestamp {
        return Err(EngineError::Durability(format!(
            "WAL archive {} target timestamp {} is before first archived timestamp {}",
            manifest_path.display(),
            target_timestamp_micros,
            first_timestamp
        )));
    }
    if target_timestamp_micros > last_timestamp {
        return Err(EngineError::Durability(format!(
            "WAL archive {} target timestamp {} is beyond last durable timestamp {}",
            manifest_path.display(),
            target_timestamp_micros,
            last_timestamp
        )));
    }

    let matching_indexes = manifest
        .record_timestamps
        .iter()
        .enumerate()
        .filter_map(|(idx, timestamp)| {
            (timestamp.timestamp_micros == target_timestamp_micros).then_some(idx)
        })
        .collect::<Vec<_>>();
    match matching_indexes.as_slice() {
        [] => Err(EngineError::Durability(format!(
            "WAL archive {} target timestamp {} falls between archived transaction boundaries",
            manifest_path.display(),
            target_timestamp_micros
        ))),
        [_first, _second, ..] => Err(EngineError::Durability(format!(
            "WAL archive {} target timestamp {} is ambiguous across multiple transaction boundaries",
            manifest_path.display(),
            target_timestamp_micros
        ))),
        [idx] => {
            let recovered_record_count = idx + 1;
            let target_txn_id = records[*idx].txn_id;
            let target = WalArchiveTimestampRecoveryTarget {
                target_timestamp_micros,
                target_txn_id,
                recovered_record_count,
                last_recovered_txn_id: target_txn_id,
            };
            Ok((
                manifest,
                target,
                records.into_iter().take(recovered_record_count).collect(),
            ))
        }
    }
}

pub fn plan_wal_archive_retention_to_txn(
    manifest_path: impl AsRef<Path>,
    target_txn_id: TxnId,
) -> Result<WalArchiveRetentionPlan, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, target, retained_records) =
        read_wal_archive_to_txn(manifest_path, target_txn_id)?;
    let records_per_segment = archive_records_per_segment(manifest_path, &manifest)?;
    let segment_dir = archive_segment_dir(manifest_path, &manifest)?;
    let retained_manifest = build_wal_archive_manifest(
        manifest_path,
        &segment_dir,
        &retained_records,
        records_per_segment,
        &manifest.record_timestamps[..target
            .recovered_record_count
            .min(manifest.record_timestamps.len())],
    );
    let retained_paths: HashSet<PathBuf> = retained_manifest
        .segments
        .iter()
        .map(|segment| resolve_manifest_path(manifest_path, &segment.segment_path))
        .collect();
    let removed_segments = manifest
        .segments
        .iter()
        .map(|segment| resolve_manifest_path(manifest_path, &segment.segment_path))
        .filter(|path| !retained_paths.contains(path))
        .collect();

    Ok(WalArchiveRetentionPlan {
        target_txn_id,
        retained_record_count: target.recovered_record_count,
        removed_record_count: manifest
            .checkpoint
            .durable_record_count
            .saturating_sub(target.recovered_record_count),
        retained_manifest,
        removed_segments,
    })
}

pub fn plan_wal_archive_retention_to_timestamp_micros(
    manifest_path: impl AsRef<Path>,
    target_timestamp_micros: u64,
) -> Result<WalArchiveRetentionPlan, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, target, retained_records) =
        read_wal_archive_to_timestamp_micros(manifest_path, target_timestamp_micros)?;
    let records_per_segment = archive_records_per_segment(manifest_path, &manifest)?;
    let segment_dir = archive_segment_dir(manifest_path, &manifest)?;
    let retained_manifest = build_wal_archive_manifest(
        manifest_path,
        &segment_dir,
        &retained_records,
        records_per_segment,
        &manifest.record_timestamps[..target
            .recovered_record_count
            .min(manifest.record_timestamps.len())],
    );
    let retained_paths: HashSet<PathBuf> = retained_manifest
        .segments
        .iter()
        .map(|segment| resolve_manifest_path(manifest_path, &segment.segment_path))
        .collect();
    let removed_segments = manifest
        .segments
        .iter()
        .map(|segment| resolve_manifest_path(manifest_path, &segment.segment_path))
        .filter(|path| !retained_paths.contains(path))
        .collect();

    Ok(WalArchiveRetentionPlan {
        target_txn_id: target.target_txn_id,
        retained_record_count: target.recovered_record_count,
        removed_record_count: manifest
            .checkpoint
            .durable_record_count
            .saturating_sub(target.recovered_record_count),
        retained_manifest,
        removed_segments,
    })
}

pub fn plan_wal_archive_retention_from_txn(
    manifest_path: impl AsRef<Path>,
    base_txn_id: TxnId,
) -> Result<WalArchiveRetentionPlan, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, records) = read_wal_archive(manifest_path)?;
    let first_txn = records.first().map(|record| record.txn_id).ok_or_else(|| {
        EngineError::Durability(format!(
            "WAL archive {} has no records for base transaction {}",
            manifest_path.display(),
            base_txn_id
        ))
    })?;
    if base_txn_id < first_txn {
        return Err(EngineError::Durability(format!(
            "WAL archive {} base transaction {} is before first archived transaction {}",
            manifest_path.display(),
            base_txn_id,
            first_txn
        )));
    }
    let last_txn = manifest.checkpoint.last_durable_txn_id.ok_or_else(|| {
        EngineError::Durability(format!(
            "WAL archive {} has no durable transaction for base transaction {}",
            manifest_path.display(),
            base_txn_id
        ))
    })?;
    if base_txn_id > last_txn {
        return Err(EngineError::Durability(format!(
            "WAL archive {} base transaction {} is beyond last durable transaction {}",
            manifest_path.display(),
            base_txn_id,
            last_txn
        )));
    }
    let start_index = records
        .iter()
        .position(|record| record.txn_id == base_txn_id)
        .ok_or_else(|| {
            EngineError::Durability(format!(
                "WAL archive {} does not contain base backup transaction boundary {}",
                manifest_path.display(),
                base_txn_id
            ))
        })?;
    let retained_records = records[start_index..].to_vec();
    let records_per_segment = archive_records_per_segment(manifest_path, &manifest)?;
    let segment_dir = archive_segment_dir(manifest_path, &manifest)?;
    let retained_timestamps = if manifest.record_timestamps.is_empty() {
        &[][..]
    } else {
        &manifest.record_timestamps[start_index..]
    };
    let retained_manifest = build_wal_archive_manifest(
        manifest_path,
        &segment_dir,
        &retained_records,
        records_per_segment,
        retained_timestamps,
    );
    let retained_paths: HashSet<PathBuf> = retained_manifest
        .segments
        .iter()
        .map(|segment| resolve_manifest_path(manifest_path, &segment.segment_path))
        .collect();
    let removed_segments = manifest
        .segments
        .iter()
        .map(|segment| resolve_manifest_path(manifest_path, &segment.segment_path))
        .filter(|path| !retained_paths.contains(path))
        .collect();

    Ok(WalArchiveRetentionPlan {
        target_txn_id: base_txn_id,
        retained_record_count: retained_records.len(),
        removed_record_count: start_index,
        retained_manifest,
        removed_segments,
    })
}

pub fn apply_wal_archive_retention_from_txn(
    manifest_path: impl AsRef<Path>,
    base_txn_id: TxnId,
) -> Result<WalArchiveRetentionPlan, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, records) = read_wal_archive(manifest_path)?;
    let plan = plan_wal_archive_retention_from_txn(manifest_path, base_txn_id)?;
    let start_index = records
        .iter()
        .position(|record| record.txn_id == base_txn_id)
        .ok_or_else(|| {
            EngineError::Durability(format!(
                "WAL archive {} does not contain base backup transaction boundary {}",
                manifest_path.display(),
                base_txn_id
            ))
        })?;
    let records_per_segment = archive_records_per_segment(manifest_path, &manifest)?;
    let segment_dir = archive_segment_dir(manifest_path, &manifest)?;
    let retained_records = &records[start_index..];
    let retained_timestamps = if manifest.record_timestamps.is_empty() {
        &[][..]
    } else {
        &manifest.record_timestamps[start_index..]
    };
    let retained_manifest = write_wal_archive_with_timestamps(
        manifest_path,
        &segment_dir,
        retained_records,
        records_per_segment,
        retained_timestamps,
    )?;
    for removed_segment in &plan.removed_segments {
        match fs::remove_file(removed_segment) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(EngineError::Durability(format!(
                    "failed to remove obsolete WAL archive segment {}: {err}",
                    removed_segment.display()
                )));
            }
        }
    }

    Ok(WalArchiveRetentionPlan {
        retained_manifest,
        ..plan
    })
}

pub fn apply_wal_archive_retention_to_timestamp_micros(
    manifest_path: impl AsRef<Path>,
    target_timestamp_micros: u64,
) -> Result<WalArchiveRetentionPlan, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, _target, retained_records) =
        read_wal_archive_to_timestamp_micros(manifest_path, target_timestamp_micros)?;
    let records_per_segment = archive_records_per_segment(manifest_path, &manifest)?;
    let segment_dir = archive_segment_dir(manifest_path, &manifest)?;
    let plan =
        plan_wal_archive_retention_to_timestamp_micros(manifest_path, target_timestamp_micros)?;

    let retained_timestamps = &manifest.record_timestamps[..plan
        .retained_record_count
        .min(manifest.record_timestamps.len())];
    let retained_manifest = write_wal_archive_with_timestamps(
        manifest_path,
        &segment_dir,
        &retained_records,
        records_per_segment,
        retained_timestamps,
    )?;
    for removed_segment in &plan.removed_segments {
        match fs::remove_file(removed_segment) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(EngineError::Durability(format!(
                    "failed to remove obsolete WAL archive segment {}: {err}",
                    removed_segment.display()
                )));
            }
        }
    }

    Ok(WalArchiveRetentionPlan {
        retained_manifest,
        ..plan
    })
}

pub fn apply_wal_archive_retention_to_txn(
    manifest_path: impl AsRef<Path>,
    target_txn_id: TxnId,
) -> Result<WalArchiveRetentionPlan, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, _target, retained_records) =
        read_wal_archive_to_txn(manifest_path, target_txn_id)?;
    let records_per_segment = archive_records_per_segment(manifest_path, &manifest)?;
    let segment_dir = archive_segment_dir(manifest_path, &manifest)?;
    let plan = plan_wal_archive_retention_to_txn(manifest_path, target_txn_id)?;

    let retained_timestamps = &manifest.record_timestamps[..plan
        .retained_record_count
        .min(manifest.record_timestamps.len())];
    let retained_manifest = write_wal_archive_with_timestamps(
        manifest_path,
        &segment_dir,
        &retained_records,
        records_per_segment,
        retained_timestamps,
    )?;
    for removed_segment in &plan.removed_segments {
        match fs::remove_file(removed_segment) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(EngineError::Durability(format!(
                    "failed to remove obsolete WAL archive segment {}: {err}",
                    removed_segment.display()
                )));
            }
        }
    }

    Ok(WalArchiveRetentionPlan {
        retained_manifest,
        ..plan
    })
}

fn validate_checkpoint_control(
    control_path: &Path,
    control: &WalControlFile,
    records: &[WalRecord],
) -> Result<(), EngineError> {
    if records.len() != control.checkpoint.durable_record_count {
        return Err(EngineError::Durability(format!(
            "WAL control {} expected {} durable records but segment contains {}",
            control_path.display(),
            control.checkpoint.durable_record_count,
            records.len()
        )));
    }
    let actual_last_txn = records.last().map(|record| record.txn_id);
    if actual_last_txn != control.checkpoint.last_durable_txn_id {
        return Err(EngineError::Durability(format!(
            "WAL control {} expected last durable txn {:?} but segment contains {:?}",
            control_path.display(),
            control.checkpoint.last_durable_txn_id,
            actual_last_txn
        )));
    }
    Ok(())
}

fn build_wal_archive_manifest(
    manifest_path: &Path,
    segment_dir: &Path,
    records: &[WalRecord],
    records_per_segment: usize,
    record_timestamps: &[WalArchiveRecordTimestamp],
) -> WalArchiveManifest {
    let mut segments = Vec::new();
    for (index, chunk) in records.chunks(records_per_segment).enumerate() {
        let file_name = format!("segment-{:04}.wal", index + 1);
        let segment_path = segment_dir.join(&file_name);
        let manifest_segment_path = segment_path
            .strip_prefix(manifest_path.parent().unwrap_or_else(|| Path::new(".")))
            .unwrap_or(&segment_path)
            .to_path_buf();
        segments.push(WalArchiveSegment {
            segment_path: manifest_segment_path,
            record_count: chunk.len(),
            first_txn_id: chunk.first().map(|record| record.txn_id),
            last_txn_id: chunk.last().map(|record| record.txn_id),
        });
    }
    WalArchiveManifest {
        segments,
        checkpoint: WalCheckpointMeta {
            durable_record_count: records.len(),
            last_durable_txn_id: records.last().map(|record| record.txn_id),
        },
        record_timestamps: record_timestamps.to_vec(),
    }
}

fn retained_timestamps(
    manifest: &WalArchiveManifest,
    retained_record_count: usize,
) -> &[WalArchiveRecordTimestamp] {
    &manifest.record_timestamps[..retained_record_count.min(manifest.record_timestamps.len())]
}

fn write_wal_archive_backup_object(
    backup_manifest_path: &Path,
    source_path: &Path,
    source_file_path: &Path,
    object_path: &Path,
) -> Result<WalArchiveObject, EngineError> {
    validate_backup_path(backup_manifest_path, "object source", source_path)?;
    validate_backup_path(backup_manifest_path, "object path", object_path)?;
    let bytes = fs::read(source_file_path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL archive backup source {}: {err}",
            source_file_path.display()
        ))
    })?;
    write_verified_backup_bytes(object_path, &bytes)?;
    let object_path = object_path
        .strip_prefix(
            backup_manifest_path
                .parent()
                .unwrap_or_else(|| Path::new(".")),
        )
        .unwrap_or(object_path)
        .to_path_buf();
    Ok(WalArchiveObject {
        source_path: source_path.to_path_buf(),
        object_path,
        byte_len: bytes.len() as u64,
        checksum: wal_object_checksum(&bytes),
    })
}

fn read_verified_wal_archive_backup_object(
    backup_manifest_path: &Path,
    object: &WalArchiveObject,
) -> Result<Vec<u8>, EngineError> {
    let object_path = resolve_manifest_path(backup_manifest_path, &object.object_path);
    let bytes = fs::read(&object_path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL archive backup object {}: {err}",
            object_path.display()
        ))
    })?;
    if bytes.len() as u64 != object.byte_len {
        return Err(EngineError::Durability(format!(
            "WAL archive backup object {} expected {} bytes but read {}",
            object_path.display(),
            object.byte_len,
            bytes.len()
        )));
    }
    let actual_checksum = wal_object_checksum(&bytes);
    if actual_checksum != object.checksum {
        return Err(EngineError::Durability(format!(
            "WAL archive backup object {} checksum mismatch",
            object_path.display()
        )));
    }
    Ok(bytes)
}

fn write_verified_backup_bytes(path: &Path, bytes: &[u8]) -> Result<(), EngineError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive backup object directory {}: {err}",
                parent.display()
            ))
        })?;
    }
    let tmp_path = temporary_control_path(path);
    let write_result = (|| {
        let mut file = File::create(&tmp_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive backup object {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.write_all(bytes).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write WAL archive backup object {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync WAL archive backup object {}: {err}",
                tmp_path.display()
            ))
        })?;
        Ok::<_, EngineError>(())
    })();
    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        EngineError::Durability(format!(
            "failed to install WAL archive backup object {}: {err}",
            path.display()
        ))
    })
}

fn validate_backup_path(path: &Path, field: &str, value: &Path) -> Result<(), EngineError> {
    let rendered = value.to_string_lossy();
    if rendered.is_empty()
        || rendered.contains('|')
        || rendered.contains('\n')
        || rendered.contains('\r')
    {
        return Err(EngineError::Durability(format!(
            "WAL archive object backup {field} contains unsupported path in {}",
            path.display()
        )));
    }
    Ok(())
}

fn archive_records_per_segment(
    manifest_path: &Path,
    manifest: &WalArchiveManifest,
) -> Result<usize, EngineError> {
    manifest
        .segments
        .first()
        .map(|segment| segment.record_count)
        .filter(|record_count| *record_count > 0)
        .ok_or_else(|| {
            EngineError::Durability(format!(
                "WAL archive {} has no segment sizing for retention",
                manifest_path.display()
            ))
        })
}

fn archive_segment_dir(
    manifest_path: &Path,
    manifest: &WalArchiveManifest,
) -> Result<PathBuf, EngineError> {
    let first_segment = manifest.segments.first().ok_or_else(|| {
        EngineError::Durability(format!(
            "WAL archive {} has no segment directory for retention",
            manifest_path.display()
        ))
    })?;
    let first_segment_path = resolve_manifest_path(manifest_path, &first_segment.segment_path);
    Ok(first_segment_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf())
}

fn validate_archive_manifest_shape(
    manifest_path: &Path,
    manifest: &WalArchiveManifest,
) -> Result<(), EngineError> {
    let segment_records: usize = manifest
        .segments
        .iter()
        .map(|segment| segment.record_count)
        .sum();
    if segment_records != manifest.checkpoint.durable_record_count {
        return Err(EngineError::Durability(format!(
            "WAL archive {} expected {} durable records but manifest segments describe {}",
            manifest_path.display(),
            manifest.checkpoint.durable_record_count,
            segment_records
        )));
    }
    if manifest.segments.is_empty() && manifest.checkpoint.last_durable_txn_id.is_some() {
        return Err(EngineError::Durability(format!(
            "WAL archive {} has no segments but records a last durable transaction",
            manifest_path.display()
        )));
    }
    if !manifest.record_timestamps.is_empty()
        && manifest.record_timestamps.len() != manifest.checkpoint.durable_record_count
    {
        return Err(EngineError::Durability(format!(
            "WAL archive {} expected {} timestamp records but manifest contains {}",
            manifest_path.display(),
            manifest.checkpoint.durable_record_count,
            manifest.record_timestamps.len()
        )));
    }
    Ok(())
}

fn validate_archive_ingest_continuity(
    manifest_path: &Path,
    manifest: &WalArchiveManifest,
    segment_records: &[WalRecord],
) -> Result<(), EngineError> {
    let Some(first_appended_txn) = segment_records.first().map(|record| record.txn_id) else {
        return Ok(());
    };
    if let Some(last_durable_txn) = manifest.checkpoint.last_durable_txn_id {
        if first_appended_txn <= last_durable_txn {
            return Err(EngineError::Durability(format!(
                "WAL archive {} ingest segment starts at transaction {} not after durable transaction {}",
                manifest_path.display(),
                first_appended_txn,
                last_durable_txn
            )));
        }
    }
    for window in segment_records.windows(2) {
        if window[0].txn_id >= window[1].txn_id {
            return Err(EngineError::Durability(format!(
                "WAL archive {} ingest segment has non-increasing transaction order at {} then {}",
                manifest_path.display(),
                window[0].txn_id,
                window[1].txn_id
            )));
        }
    }
    Ok(())
}

fn validate_archive_ingest_timestamps(
    manifest_path: &Path,
    manifest: &WalArchiveManifest,
    segment_records: &[WalRecord],
    record_timestamps: &[WalArchiveRecordTimestamp],
) -> Result<(), EngineError> {
    match (
        manifest.record_timestamps.is_empty(),
        manifest.checkpoint.durable_record_count,
        record_timestamps.is_empty(),
    ) {
        (true, 0, _) => {}
        (true, _, true) => {}
        (true, _, false) => {
            return Err(EngineError::Durability(format!(
                "WAL archive {} cannot add timestamp metadata to an existing archive without timestamps",
                manifest_path.display()
            )));
        }
        (false, _, true) => {
            return Err(EngineError::Durability(format!(
                "WAL archive {} requires timestamp metadata for ingested segment",
                manifest_path.display()
            )));
        }
        (false, _, false) => {}
    }
    validate_timestamp_metadata(manifest_path, segment_records, record_timestamps)?;
    if let (Some(existing_last), Some(appended_first)) = (
        manifest
            .record_timestamps
            .last()
            .map(|timestamp| timestamp.timestamp_micros),
        record_timestamps
            .first()
            .map(|timestamp| timestamp.timestamp_micros),
    ) {
        if appended_first < existing_last {
            return Err(EngineError::Durability(format!(
                "WAL archive {} ingest segment timestamp {} is before last archived timestamp {}",
                manifest_path.display(),
                appended_first,
                existing_last
            )));
        }
    }
    Ok(())
}

fn validate_archive_segment(
    manifest_path: &Path,
    segment: &WalArchiveSegment,
    records: &[WalRecord],
) -> Result<(), EngineError> {
    if records.len() != segment.record_count {
        return Err(EngineError::Durability(format!(
            "WAL archive {} segment {} expected {} records but contains {}",
            manifest_path.display(),
            segment.segment_path.display(),
            segment.record_count,
            records.len()
        )));
    }
    let actual_first = records.first().map(|record| record.txn_id);
    let actual_last = records.last().map(|record| record.txn_id);
    if actual_first != segment.first_txn_id || actual_last != segment.last_txn_id {
        return Err(EngineError::Durability(format!(
            "WAL archive {} segment {} expected txn range {:?}..{:?} but contains {:?}..{:?}",
            manifest_path.display(),
            segment.segment_path.display(),
            segment.first_txn_id,
            segment.last_txn_id,
            actual_first,
            actual_last
        )));
    }
    Ok(())
}

fn validate_archive_records(
    manifest_path: &Path,
    manifest: &WalArchiveManifest,
    records: &[WalRecord],
) -> Result<(), EngineError> {
    if records.len() != manifest.checkpoint.durable_record_count {
        return Err(EngineError::Durability(format!(
            "WAL archive {} expected {} durable records but read {}",
            manifest_path.display(),
            manifest.checkpoint.durable_record_count,
            records.len()
        )));
    }
    let actual_last = records.last().map(|record| record.txn_id);
    if actual_last != manifest.checkpoint.last_durable_txn_id {
        return Err(EngineError::Durability(format!(
            "WAL archive {} expected last durable txn {:?} but read {:?}",
            manifest_path.display(),
            manifest.checkpoint.last_durable_txn_id,
            actual_last
        )));
    }
    for window in records.windows(2) {
        if window[0].txn_id >= window[1].txn_id {
            return Err(EngineError::Durability(format!(
                "WAL archive {} has non-increasing transaction order at {} then {}",
                manifest_path.display(),
                window[0].txn_id,
                window[1].txn_id
            )));
        }
    }
    Ok(())
}

fn validate_timestamp_metadata(
    manifest_path: &Path,
    records: &[WalRecord],
    record_timestamps: &[WalArchiveRecordTimestamp],
) -> Result<(), EngineError> {
    if record_timestamps.is_empty() {
        return Ok(());
    }
    if record_timestamps.len() != records.len() {
        return Err(EngineError::Durability(format!(
            "WAL archive {} expected {} timestamp records but received {}",
            manifest_path.display(),
            records.len(),
            record_timestamps.len()
        )));
    }
    for (record, timestamp) in records.iter().zip(record_timestamps) {
        if record.txn_id != timestamp.txn_id {
            return Err(EngineError::Durability(format!(
                "WAL archive {} timestamp metadata transaction {} does not match record transaction {}",
                manifest_path.display(),
                timestamp.txn_id,
                record.txn_id
            )));
        }
    }
    Ok(())
}

fn validate_archive_timestamps(
    manifest_path: &Path,
    manifest: &WalArchiveManifest,
    records: &[WalRecord],
) -> Result<(), EngineError> {
    validate_timestamp_metadata(manifest_path, records, &manifest.record_timestamps)?;
    for window in manifest.record_timestamps.windows(2) {
        if window[0].timestamp_micros > window[1].timestamp_micros {
            return Err(EngineError::Durability(format!(
                "WAL archive {} has decreasing timestamp order at {} then {}",
                manifest_path.display(),
                window[0].timestamp_micros,
                window[1].timestamp_micros
            )));
        }
    }
    Ok(())
}

fn resolve_manifest_path(manifest_path: &Path, data_path: &Path) -> PathBuf {
    if data_path.is_absolute() {
        data_path.to_path_buf()
    } else {
        manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(data_path)
    }
}

fn format_optional_txn(txn_id: Option<TxnId>) -> String {
    txn_id
        .map(|txn_id| txn_id.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn format_optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn parse_optional_txn(raw: &str, field: &str, path: &Path) -> Result<Option<TxnId>, EngineError> {
    match raw {
        "none" => Ok(None),
        raw => raw.parse().map(Some).map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive {field} {}: {err}",
                path.display()
            ))
        }),
    }
}

fn parse_optional_u64(raw: &str, field: &str, path: &Path) -> Result<Option<u64>, EngineError> {
    match raw {
        "none" => Ok(None),
        raw => raw.parse().map(Some).map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive {field} {}: {err}",
                path.display()
            ))
        }),
    }
}

fn write_record(file: &mut File, record: &WalRecord) -> Result<(), EngineError> {
    let mut bytes = Vec::with_capacity(WAL_RECORD_HEADER_LEN + record.payload.len());
    encode_record_into(&mut bytes, record)?;
    file.write_all(&bytes)
        .map_err(|err| EngineError::Durability(format!("failed to write WAL record: {err}")))
}

/// Serialize one record (header + payload) onto `buf` in the on-disk segment format.
/// Encode one record's envelope+payload into `buf` from raw parts — the fused
/// single-pass form for lane pumps (payload bytes are warm from the row-id
/// patch that immediately precedes this in the caller's loop).
pub fn encode_wal_record_parts_into(buf: &mut Vec<u8>, txn_id: TxnId, payload: &[u8]) {
    let payload_len = payload.len() as u64;
    let checksum = wal_record_checksum(txn_id, payload_len, payload);
    buf.extend_from_slice(&txn_id.to_le_bytes());
    buf.extend_from_slice(&payload_len.to_le_bytes());
    buf.extend_from_slice(&checksum.to_le_bytes());
    buf.extend_from_slice(payload);
}

fn encode_record_into(buf: &mut Vec<u8>, record: &WalRecord) -> Result<(), EngineError> {
    let payload_len = u64::try_from(record.payload.len()).map_err(|_| {
        EngineError::Durability("WAL record payload length exceeds u64".to_string())
    })?;
    let checksum = wal_record_checksum(record.txn_id, payload_len, &record.payload);
    buf.extend_from_slice(&record.txn_id.to_le_bytes());
    buf.extend_from_slice(&payload_len.to_le_bytes());
    buf.extend_from_slice(&checksum.to_le_bytes());
    buf.extend_from_slice(&record.payload);
    Ok(())
}

/// On-disk byte length of one serialized record.
fn encoded_record_len(record: &WalRecord) -> u64 {
    WAL_RECORD_HEADER_LEN as u64 + record.payload.len() as u64
}

/// Strictly decode a run of [`encode_record_into`]-encoded records that must consume EXACTLY
/// `bytes` (no torn tail — the caller has already validated the container's integrity, e.g. a FUA
/// frame's payload CRC). This is the FUA backend's replay decode: a frame payload is the byte-for-
/// byte record run the serial group flush would have written, so it decodes to the identical
/// `WalRecord`s. Each record's own checksum is re-verified as defense in depth, and any leftover
/// or truncated bytes are a hard error (an intact frame can never contain a partial record).
#[cfg(unix)]
pub(crate) fn decode_wal_record_run(bytes: &[u8]) -> Result<Vec<WalRecord>, EngineError> {
    let mut records = Vec::new();
    let mut offset = 0usize;
    while offset < bytes.len() {
        if bytes.len() - offset < WAL_RECORD_HEADER_LEN {
            return Err(EngineError::Durability(
                "FUA WAL frame payload ends with a truncated record header".to_string(),
            ));
        }
        let header = &bytes[offset..offset + WAL_RECORD_HEADER_LEN];
        let txn_id = u64::from_le_bytes(header[0..8].try_into().expect("txn id bytes"));
        let payload_len = u64::from_le_bytes(header[8..16].try_into().expect("payload len bytes"));
        let expected_checksum =
            u64::from_le_bytes(header[16..24].try_into().expect("checksum bytes"));
        let payload_start = offset + WAL_RECORD_HEADER_LEN;
        let payload_end = usize::try_from(payload_len)
            .ok()
            .and_then(|len| payload_start.checked_add(len))
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| {
                EngineError::Durability(
                    "FUA WAL frame payload record length overruns the frame".to_string(),
                )
            })?;
        let payload = &bytes[payload_start..payload_end];
        if wal_record_checksum(txn_id, payload_len, payload) != expected_checksum {
            return Err(EngineError::Durability(format!(
                "FUA WAL frame payload record checksum mismatch for txn {txn_id}"
            )));
        }
        records.push(WalRecord {
            txn_id,
            payload: payload.to_vec().into(),
        });
        offset = payload_end;
    }
    Ok(records)
}

fn temporary_segment_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.segment");
    path.with_file_name(format!(".{file_name}.tmp.{}", std::process::id()))
}

fn temporary_control_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.control");
    path.with_file_name(format!(".{file_name}.tmp.{}", std::process::id()))
}

fn parse_control_value<'a>(
    line: Option<&'a str>,
    key: &str,
    path: &Path,
) -> Result<&'a str, EngineError> {
    let line = line.ok_or_else(|| {
        EngineError::Durability(format!(
            "missing WAL control field {key} in {}",
            path.display()
        ))
    })?;
    line.strip_prefix(&format!("{key}=")).ok_or_else(|| {
        EngineError::Durability(format!(
            "invalid WAL control field {key} in {}",
            path.display()
        ))
    })
}

fn wal_record_checksum(txn_id: TxnId, payload_len: u64, payload: &[u8]) -> u64 {
    // FNV-1a, BYTE-IDENTICAL to the original chained-iterator form (same
    // format, same values) but in tight slice loops: the chained iterator
    // defeated optimization and was measured at ~1.5us per ~100B record on
    // the lane pump's encode stage (1.5ms of a 1000-record wave).
    #[inline]
    fn fnv_step(mut hash: u64, bytes: &[u8]) -> u64 {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    hash = fnv_step(hash, &txn_id.to_le_bytes());
    hash = fnv_step(hash, &payload_len.to_le_bytes());
    fnv_step(hash, payload)
}

fn wal_object_checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_PATH_ID: AtomicU64 = AtomicU64::new(1);

    fn test_wal_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "gpu-db-wal-{name}-{}-{}.segment",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    include!("tests/buffer_segment_checkpoint.rs");
    include!("tests/archive.rs");
    include!("tests/fua_backend.rs");
}
