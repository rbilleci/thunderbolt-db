use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use gpu_db_types::{EngineError, TxnId};

const WAL_SEGMENT_MAGIC: &[u8; 10] = b"GPUDBWAL1\n";
const WAL_CONTROL_MAGIC: &str = "GPUDBWALCONTROL1";
const WAL_ARCHIVE_MANIFEST_MAGIC: &str = "GPUDBWALARCHIVE1";
const WAL_ARCHIVE_TIMELINE_MAGIC: &str = "GPUDBWALTIMELINE1";
const WAL_ARCHIVE_TIMELINE_REGISTRY_MAGIC: &str = "GPUDBWALTIMELINEREGISTRY1";
const WAL_ARCHIVE_OBJECT_BACKUP_MAGIC: &str = "GPUDBWALOBJECTBACKUP1";
const WAL_RECORD_HEADER_LEN: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalRecord {
    pub txn_id: TxnId,
    pub payload: Vec<u8>,
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
pub struct WalArchiveTimeline {
    pub timeline_id: String,
    pub parent_timeline_id: Option<String>,
    pub fork_txn_id: TxnId,
    pub fork_timestamp_micros: Option<u64>,
    pub source_manifest_path: PathBuf,
    pub branch_manifest_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveTimelineBranch {
    pub timeline: WalArchiveTimeline,
    pub manifest: WalArchiveManifest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveTimelineRegistryEntry {
    pub timeline_id: String,
    pub parent_timeline_id: Option<String>,
    pub fork_txn_id: TxnId,
    pub fork_timestamp_micros: Option<u64>,
    pub timeline_path: PathBuf,
    pub branch_manifest_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveTimelineRegistry {
    pub timelines: Vec<WalArchiveTimelineRegistryEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveTimelineSelection {
    pub entry: WalArchiveTimelineRegistryEntry,
    pub timeline: WalArchiveTimeline,
    pub manifest: WalArchiveManifest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveTimelinePrunePlan {
    pub retained_timeline_id: String,
    pub retained_timeline_ids: Vec<String>,
    pub removed_timeline_ids: Vec<String>,
    pub retained_registry: WalArchiveTimelineRegistry,
    pub removed_timeline_paths: Vec<PathBuf>,
    pub removed_branch_manifest_paths: Vec<PathBuf>,
    pub removed_segment_paths: Vec<PathBuf>,
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

/// The durable backing for a [`WalBuffer`].
///
/// When present, every `flush_all` rewrites the buffer's full record prefix to a single segment
/// file via [`write_wal_segment`] (atomic temp-write + `sync_all` + rename) and then fsyncs the
/// segment's **parent directory** so the rename — i.e. the segment file's *existence* — is itself
/// crash-durable, not just the file's bytes. The parent directory is fsynced once, the first time
/// the segment is installed (the directory entry never changes afterward — the segment keeps the
/// same path and is replaced in place by atomic rename), so steady-state commits pay a single
/// file fsync.
#[derive(Debug, Clone)]
struct WalDurableSegment {
    segment_path: PathBuf,
}

/// Group-commit accounting for a [`WalBuffer`].
///
/// Each `flush_all` that performs a real fsync batches **all** currently-unflushed records into a
/// single segment write / single fsync — that batch is one *group*. While the writer is serialized
/// (Stage 1), commits arrive one at a time, so the common case is a size-1 group; the same code
/// path amortizes automatically once Stage 4 lets multiple committers enqueue records before a
/// designated flusher drives one `flush_all`. These counters let a microbenchmark observe the
/// fsync-per-commit cost now and the batching ratio (`durable_records / flush_groups`) later.
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

#[derive(Debug, Default)]
pub struct WalBuffer {
    records: Vec<WalRecord>,
    flushed: usize,
    fail_next_flush: bool,
    durable: Option<WalDurableSegment>,
    group_commit: WalGroupCommitStats,
}

impl WalBuffer {
    /// An in-memory WAL buffer with no durable backing (the default — `flush_all` only advances the
    /// in-memory durable watermark). Used by ephemeral engines and the bulk of the test suite.
    pub fn new() -> Self {
        Self::default()
    }

    /// A WAL buffer backed by a real, fsync-durable segment file at `segment_path`.
    ///
    /// `flush_all` persists the buffer's record prefix to that file and fsyncs both the file and
    /// its parent directory before advancing the durable watermark. Recovery
    /// reads the segment back with [`read_wal_segment`].
    pub fn with_durable_segment(segment_path: impl Into<PathBuf>) -> Self {
        Self {
            durable: Some(WalDurableSegment {
                segment_path: segment_path.into(),
            }),
            ..Self::default()
        }
    }

    /// The durable segment path, if this buffer is backed by one.
    pub fn durable_segment_path(&self) -> Option<&Path> {
        self.durable.as_ref().map(|d| d.segment_path.as_path())
    }

    /// Whether `flush_all` performs a real fsync (vs. in-memory watermark advance only).
    pub fn is_durable(&self) -> bool {
        self.durable.is_some()
    }

    pub fn append(&mut self, rec: WalRecord) {
        self.records.push(rec);
    }

    /// Seed the buffer with records already known to be durable (e.g. recovered from a segment),
    /// marking them as the flushed prefix WITHOUT performing any I/O. Used right after a durable
    /// segment is installed on a recovered engine so the next real `flush_all` rewrites a segment
    /// that still contains the recovered history rather than only the newly-appended tail. Must be
    /// called on an otherwise-empty buffer.
    pub fn reinstate_durable_records(&mut self, records: Vec<WalRecord>) {
        debug_assert!(
            self.records.is_empty(),
            "reinstate_durable_records on a non-empty WAL buffer"
        );
        self.flushed = records.len();
        self.records = records;
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn truncate(&mut self, len: usize) {
        self.records.truncate(len);
        if self.flushed > self.records.len() {
            self.flushed = self.records.len();
        }
    }

    /// Make every appended record durable.
    ///
    /// In-memory mode: advances the durable watermark to the full record count.
    ///
    /// Durable mode: this is the **commit fsync** and the group-commit point. It writes the
    /// buffer's entire record prefix to the segment as one atomic, fsynced unit (and fsyncs the
    /// parent directory the first time the segment is installed), batching all currently-unflushed
    /// records into a single fsync. Only after the fsync succeeds is the in-memory durable
    /// watermark advanced — so a caller that gates visibility on `flushed_count` can never publish
    /// a record whose WAL bytes are not yet on disk (the WAL-before-visibility invariant). On any
    /// I/O error the watermark is left untouched and the error is returned, so the caller can roll
    /// back the in-flight commit before it becomes visible.
    pub fn flush_all(&mut self) -> Result<(), EngineError> {
        if self.fail_next_flush {
            self.fail_next_flush = false;
            return Err(EngineError::Durability(
                "simulated wal flush failure".to_string(),
            ));
        }
        let target = self.records.len();
        let group_size = target.saturating_sub(self.flushed);
        if let Some(durable) = self.durable.as_ref() {
            if group_size > 0 {
                let segment_path = durable.segment_path.clone();
                // Persist the FULL durable prefix (the segment is rewritten in place by atomic
                // rename), fsyncing the segment file's bytes (`write_wal_segment` -> `sync_all`).
                write_wal_segment(&segment_path, &self.records[..target])?;
                // Then fsync the parent directory so the rename (the dentry->inode mapping) is durable
                // before the watermark advances. Done after EVERY rename, not just the first install:
                // each commit's rename-over-existing mutates the dentry, and the WAL-before-visibility
                // invariant must not depend on the filesystem journal-ordering that metadata vs the data.
                sync_segment_parent_dir(&segment_path)?;
                self.group_commit.flush_groups += 1;
                self.group_commit.durable_records += group_size as u64;
                self.group_commit.max_group_size = self.group_commit.max_group_size.max(group_size);
            }
        }
        // Watermark advances only after the fsync has succeeded (or in in-memory mode).
        self.flushed = target;
        Ok(())
    }

    pub fn flushed_count(&self) -> usize {
        self.flushed
    }

    pub fn flushed_records(&self) -> &[WalRecord] {
        &self.records[..self.flushed]
    }

    pub fn unflushed_count(&self) -> usize {
        self.records.len().saturating_sub(self.flushed)
    }

    /// Group-commit accounting (fsync groups, durable records, largest group). See
    /// [`WalGroupCommitStats`].
    pub fn group_commit_stats(&self) -> WalGroupCommitStats {
        self.group_commit
    }

    pub fn checkpoint_meta(&self) -> WalCheckpointMeta {
        WalCheckpointMeta {
            durable_record_count: self.flushed,
            last_durable_txn_id: self.flushed_records().last().map(|record| record.txn_id),
        }
    }

    pub fn fail_next_flush(&mut self) {
        self.fail_next_flush = true;
    }
}

/// Fsync the parent directory of `segment_path` so a freshly-`rename`d segment file's directory
/// entry is durable across a crash (POSIX: an fsync of the file does not guarantee the containing
/// directory entry is persisted). A best-effort no-op on platforms / filesystems that refuse to
/// open a directory for fsync is intentionally NOT done — a hard error here means the existence of
/// the just-written WAL could be lost on crash, which would violate durability, so it propagates.
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
                records.push(WalRecord { txn_id, payload });
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
    let records = read_wal_segment(&segment_path)?;
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

pub fn fork_wal_archive_timeline_to_txn(
    source_manifest_path: impl AsRef<Path>,
    branch_manifest_path: impl AsRef<Path>,
    branch_segment_dir: impl AsRef<Path>,
    timeline_path: impl AsRef<Path>,
    timeline_id: impl AsRef<str>,
    parent_timeline_id: Option<&str>,
    target_txn_id: TxnId,
) -> Result<WalArchiveTimelineBranch, EngineError> {
    let source_manifest_path = source_manifest_path.as_ref();
    let branch_manifest_path = branch_manifest_path.as_ref();
    let branch_segment_dir = branch_segment_dir.as_ref();
    let timeline_path = timeline_path.as_ref();
    let timeline = validate_timeline_identity(
        timeline_path,
        timeline_id.as_ref(),
        parent_timeline_id,
        source_manifest_path,
        branch_manifest_path,
        target_txn_id,
        None,
    )?;
    let (source_manifest, target, records) =
        read_wal_archive_to_txn(source_manifest_path, target_txn_id)?;
    let record_timestamps = retained_timestamps(&source_manifest, target.recovered_record_count);
    let records_per_segment = archive_records_per_segment(source_manifest_path, &source_manifest)?;
    let manifest = write_wal_archive_with_timestamps(
        branch_manifest_path,
        branch_segment_dir,
        &records,
        records_per_segment,
        record_timestamps,
    )?;
    write_wal_archive_timeline(timeline_path, &timeline)?;
    Ok(WalArchiveTimelineBranch { timeline, manifest })
}

pub fn fork_wal_archive_timeline_to_timestamp_micros(
    source_manifest_path: impl AsRef<Path>,
    branch_manifest_path: impl AsRef<Path>,
    branch_segment_dir: impl AsRef<Path>,
    timeline_path: impl AsRef<Path>,
    timeline_id: impl AsRef<str>,
    parent_timeline_id: Option<&str>,
    target_timestamp_micros: u64,
) -> Result<WalArchiveTimelineBranch, EngineError> {
    let source_manifest_path = source_manifest_path.as_ref();
    let branch_manifest_path = branch_manifest_path.as_ref();
    let branch_segment_dir = branch_segment_dir.as_ref();
    let timeline_path = timeline_path.as_ref();
    let (source_manifest, target, records) =
        read_wal_archive_to_timestamp_micros(source_manifest_path, target_timestamp_micros)?;
    let timeline = validate_timeline_identity(
        timeline_path,
        timeline_id.as_ref(),
        parent_timeline_id,
        source_manifest_path,
        branch_manifest_path,
        target.target_txn_id,
        Some(target_timestamp_micros),
    )?;
    let record_timestamps = retained_timestamps(&source_manifest, target.recovered_record_count);
    let records_per_segment = archive_records_per_segment(source_manifest_path, &source_manifest)?;
    let manifest = write_wal_archive_with_timestamps(
        branch_manifest_path,
        branch_segment_dir,
        &records,
        records_per_segment,
        record_timestamps,
    )?;
    write_wal_archive_timeline(timeline_path, &timeline)?;
    Ok(WalArchiveTimelineBranch { timeline, manifest })
}

pub fn write_wal_archive_timeline(
    path: impl AsRef<Path>,
    timeline: &WalArchiveTimeline,
) -> Result<(), EngineError> {
    let path = path.as_ref();
    validate_timeline_value(path, "timeline_id", &timeline.timeline_id)?;
    if let Some(parent) = timeline.parent_timeline_id.as_ref() {
        validate_timeline_value(path, "parent_timeline_id", parent)?;
        if parent == &timeline.timeline_id {
            return Err(EngineError::Durability(format!(
                "WAL archive timeline {} cannot be its own parent",
                timeline.timeline_id
            )));
        }
    }
    validate_timeline_path(path, "source_manifest_path", &timeline.source_manifest_path)?;
    validate_timeline_path(path, "branch_manifest_path", &timeline.branch_manifest_path)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL timeline directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let body = format!(
        "{WAL_ARCHIVE_TIMELINE_MAGIC}\ntimeline_id={}\nparent_timeline_id={}\nfork_txn_id={}\nfork_timestamp_micros={}\nsource_manifest_path={}\nbranch_manifest_path={}\n",
        timeline.timeline_id,
        timeline
            .parent_timeline_id
            .as_deref()
            .unwrap_or("none"),
        timeline.fork_txn_id,
        format_optional_u64(timeline.fork_timestamp_micros),
        timeline.source_manifest_path.display(),
        timeline.branch_manifest_path.display()
    );

    let tmp_path = temporary_control_path(path);
    let write_result = (|| {
        let mut file = File::create(&tmp_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive timeline {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.write_all(body.as_bytes()).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write WAL archive timeline {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync WAL archive timeline {}: {err}",
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
            "failed to install WAL archive timeline {}: {err}",
            path.display()
        ))
    })
}

pub fn read_wal_archive_timeline(
    path: impl AsRef<Path>,
) -> Result<WalArchiveTimeline, EngineError> {
    let path = path.as_ref();
    let body = fs::read_to_string(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL archive timeline {}: {err}",
            path.display()
        ))
    })?;
    let mut lines = body.lines();
    if lines.next() != Some(WAL_ARCHIVE_TIMELINE_MAGIC) {
        return Err(EngineError::Durability(format!(
            "invalid WAL archive timeline header {}",
            path.display()
        )));
    }

    let timeline_id = parse_control_value(lines.next(), "timeline_id", path)?.to_string();
    let parent_timeline_id = match parse_control_value(lines.next(), "parent_timeline_id", path)? {
        "none" => None,
        parent => Some(parent.to_string()),
    };
    let fork_txn_id = parse_control_value(lines.next(), "fork_txn_id", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive timeline fork transaction {}: {err}",
                path.display()
            ))
        })?;
    let fork_timestamp_micros = parse_optional_u64(
        parse_control_value(lines.next(), "fork_timestamp_micros", path)?,
        "fork timestamp",
        path,
    )?;
    let source_manifest_path = PathBuf::from(parse_control_value(
        lines.next(),
        "source_manifest_path",
        path,
    )?);
    let branch_manifest_path = PathBuf::from(parse_control_value(
        lines.next(),
        "branch_manifest_path",
        path,
    )?);
    let timeline = WalArchiveTimeline {
        timeline_id,
        parent_timeline_id,
        fork_txn_id,
        fork_timestamp_micros,
        source_manifest_path,
        branch_manifest_path,
    };
    validate_timeline_value(path, "timeline_id", &timeline.timeline_id)?;
    if let Some(parent) = timeline.parent_timeline_id.as_ref() {
        validate_timeline_value(path, "parent_timeline_id", parent)?;
        if parent == &timeline.timeline_id {
            return Err(EngineError::Durability(format!(
                "WAL archive timeline {} cannot be its own parent",
                timeline.timeline_id
            )));
        }
    }
    validate_timeline_path(path, "source_manifest_path", &timeline.source_manifest_path)?;
    validate_timeline_path(path, "branch_manifest_path", &timeline.branch_manifest_path)?;
    Ok(timeline)
}

pub fn register_wal_archive_timeline(
    registry_path: impl AsRef<Path>,
    timeline_path: impl AsRef<Path>,
) -> Result<WalArchiveTimelineRegistry, EngineError> {
    let registry_path = registry_path.as_ref();
    let timeline_path = timeline_path.as_ref();
    let timeline = read_wal_archive_timeline(timeline_path)?;
    let mut registry = if registry_path.exists() {
        read_wal_archive_timeline_registry(registry_path)?
    } else {
        WalArchiveTimelineRegistry {
            timelines: Vec::new(),
        }
    };
    if registry
        .timelines
        .iter()
        .any(|entry| entry.timeline_id == timeline.timeline_id)
    {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline registry {} already contains timeline {}",
            registry_path.display(),
            timeline.timeline_id
        )));
    }
    if let Some(parent) = timeline.parent_timeline_id.as_ref() {
        if !registry
            .timelines
            .iter()
            .any(|entry| &entry.timeline_id == parent)
        {
            return Err(EngineError::Durability(format!(
                "WAL archive timeline registry {} is missing parent timeline {} for child {}",
                registry_path.display(),
                parent,
                timeline.timeline_id
            )));
        }
    }

    let (_manifest, _records) = read_wal_archive(&timeline.branch_manifest_path)?;
    registry.timelines.push(WalArchiveTimelineRegistryEntry {
        timeline_id: timeline.timeline_id,
        parent_timeline_id: timeline.parent_timeline_id,
        fork_txn_id: timeline.fork_txn_id,
        fork_timestamp_micros: timeline.fork_timestamp_micros,
        timeline_path: timeline_path.to_path_buf(),
        branch_manifest_path: timeline.branch_manifest_path,
    });
    write_wal_archive_timeline_registry(registry_path, &registry)?;
    read_wal_archive_timeline_registry(registry_path)
}

pub fn select_wal_archive_timeline(
    registry_path: impl AsRef<Path>,
    timeline_id: impl AsRef<str>,
) -> Result<WalArchiveTimelineSelection, EngineError> {
    let registry_path = registry_path.as_ref();
    let timeline_id = timeline_id.as_ref();
    validate_timeline_value(registry_path, "timeline_id", timeline_id)?;
    let registry = read_wal_archive_timeline_registry(registry_path)?;
    let entry = registry
        .timelines
        .iter()
        .find(|entry| entry.timeline_id == timeline_id)
        .cloned()
        .ok_or_else(|| {
            EngineError::Durability(format!(
                "WAL archive timeline registry {} has no timeline {}",
                registry_path.display(),
                timeline_id
            ))
        })?;
    let timeline = read_wal_archive_timeline(&entry.timeline_path)?;
    if timeline.timeline_id != entry.timeline_id
        || timeline.parent_timeline_id != entry.parent_timeline_id
        || timeline.fork_txn_id != entry.fork_txn_id
        || timeline.fork_timestamp_micros != entry.fork_timestamp_micros
        || timeline.branch_manifest_path != entry.branch_manifest_path
    {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline registry {} entry {} does not match sidecar {}",
            registry_path.display(),
            entry.timeline_id,
            entry.timeline_path.display()
        )));
    }
    let (manifest, _records) = read_wal_archive(&timeline.branch_manifest_path)?;
    Ok(WalArchiveTimelineSelection {
        entry,
        timeline,
        manifest,
    })
}

pub fn plan_wal_archive_timeline_prune(
    registry_path: impl AsRef<Path>,
    retained_timeline_id: impl AsRef<str>,
) -> Result<WalArchiveTimelinePrunePlan, EngineError> {
    let registry_path = registry_path.as_ref();
    let retained_timeline_id = retained_timeline_id.as_ref();
    validate_timeline_value(registry_path, "timeline_id", retained_timeline_id)?;
    let registry = read_wal_archive_timeline_registry(registry_path)?;
    let mut retained = HashSet::new();
    let mut next_timeline_id = Some(retained_timeline_id.to_string());
    while let Some(timeline_id) = next_timeline_id {
        let entry = registry
            .timelines
            .iter()
            .find(|entry| entry.timeline_id == timeline_id)
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "WAL archive timeline registry {} has no timeline {}",
                    registry_path.display(),
                    timeline_id
                ))
            })?;
        retained.insert(entry.timeline_id.clone());
        next_timeline_id = entry.parent_timeline_id.clone();
    }

    for entry in &registry.timelines {
        validate_registered_timeline_entry(registry_path, entry)?;
    }

    let retained_registry = WalArchiveTimelineRegistry {
        timelines: registry
            .timelines
            .iter()
            .filter(|entry| retained.contains(&entry.timeline_id))
            .cloned()
            .collect(),
    };
    validate_timeline_registry_shape(registry_path, &retained_registry)?;
    let retained_timeline_ids = retained_registry
        .timelines
        .iter()
        .map(|entry| entry.timeline_id.clone())
        .collect();
    let removed_timeline_ids = registry
        .timelines
        .iter()
        .filter(|entry| !retained.contains(&entry.timeline_id))
        .map(|entry| entry.timeline_id.clone())
        .collect();

    let retained_timeline_paths: HashSet<PathBuf> = retained_registry
        .timelines
        .iter()
        .map(|entry| entry.timeline_path.clone())
        .collect();
    let retained_manifest_paths: HashSet<PathBuf> = retained_registry
        .timelines
        .iter()
        .map(|entry| entry.branch_manifest_path.clone())
        .collect();
    let mut retained_segment_paths = HashSet::new();
    for entry in &retained_registry.timelines {
        let (manifest, _records) = read_wal_archive(&entry.branch_manifest_path)?;
        for segment in &manifest.segments {
            retained_segment_paths.insert(resolve_manifest_path(
                &entry.branch_manifest_path,
                &segment.segment_path,
            ));
        }
    }

    let mut removed_timeline_paths = Vec::new();
    let mut removed_branch_manifest_paths = Vec::new();
    let mut removed_segment_paths = Vec::new();
    let mut seen_removed_timeline_paths = HashSet::new();
    let mut seen_removed_manifest_paths = HashSet::new();
    let mut seen_removed_segment_paths = HashSet::new();
    for entry in registry
        .timelines
        .iter()
        .filter(|entry| !retained.contains(&entry.timeline_id))
    {
        if !retained_timeline_paths.contains(&entry.timeline_path)
            && seen_removed_timeline_paths.insert(entry.timeline_path.clone())
        {
            removed_timeline_paths.push(entry.timeline_path.clone());
        }
        if !retained_manifest_paths.contains(&entry.branch_manifest_path)
            && seen_removed_manifest_paths.insert(entry.branch_manifest_path.clone())
        {
            removed_branch_manifest_paths.push(entry.branch_manifest_path.clone());
        }

        let (manifest, _records) = read_wal_archive(&entry.branch_manifest_path)?;
        for segment in &manifest.segments {
            let segment_path =
                resolve_manifest_path(&entry.branch_manifest_path, &segment.segment_path);
            if !retained_segment_paths.contains(&segment_path)
                && seen_removed_segment_paths.insert(segment_path.clone())
            {
                removed_segment_paths.push(segment_path);
            }
        }
    }

    Ok(WalArchiveTimelinePrunePlan {
        retained_timeline_id: retained_timeline_id.to_string(),
        retained_timeline_ids,
        removed_timeline_ids,
        retained_registry,
        removed_timeline_paths,
        removed_branch_manifest_paths,
        removed_segment_paths,
    })
}

pub fn apply_wal_archive_timeline_prune(
    registry_path: impl AsRef<Path>,
    retained_timeline_id: impl AsRef<str>,
) -> Result<WalArchiveTimelinePrunePlan, EngineError> {
    let registry_path = registry_path.as_ref();
    let plan = plan_wal_archive_timeline_prune(registry_path, retained_timeline_id)?;
    write_wal_archive_timeline_registry(registry_path, &plan.retained_registry)?;
    for segment_path in &plan.removed_segment_paths {
        remove_wal_archive_timeline_artifact("segment", segment_path)?;
    }
    for manifest_path in &plan.removed_branch_manifest_paths {
        remove_wal_archive_timeline_artifact("manifest", manifest_path)?;
    }
    for timeline_path in &plan.removed_timeline_paths {
        remove_wal_archive_timeline_artifact("sidecar", timeline_path)?;
    }
    Ok(plan)
}

pub fn write_wal_archive_timeline_registry(
    path: impl AsRef<Path>,
    registry: &WalArchiveTimelineRegistry,
) -> Result<(), EngineError> {
    let path = path.as_ref();
    validate_timeline_registry_shape(path, registry)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive timeline registry directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let mut body = format!(
        "{WAL_ARCHIVE_TIMELINE_REGISTRY_MAGIC}\ntimeline_count={}\n",
        registry.timelines.len()
    );
    for entry in &registry.timelines {
        body.push_str(&format!(
            "timeline={}|{}|{}|{}|{}|{}\n",
            entry.timeline_id,
            entry.parent_timeline_id.as_deref().unwrap_or("none"),
            entry.fork_txn_id,
            format_optional_u64(entry.fork_timestamp_micros),
            entry.timeline_path.display(),
            entry.branch_manifest_path.display()
        ));
    }

    let tmp_path = temporary_control_path(path);
    let write_result = (|| {
        let mut file = File::create(&tmp_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive timeline registry {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.write_all(body.as_bytes()).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write WAL archive timeline registry {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync WAL archive timeline registry {}: {err}",
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
            "failed to install WAL archive timeline registry {}: {err}",
            path.display()
        ))
    })
}

pub fn read_wal_archive_timeline_registry(
    path: impl AsRef<Path>,
) -> Result<WalArchiveTimelineRegistry, EngineError> {
    let path = path.as_ref();
    let body = fs::read_to_string(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL archive timeline registry {}: {err}",
            path.display()
        ))
    })?;
    let mut lines = body.lines();
    if lines.next() != Some(WAL_ARCHIVE_TIMELINE_REGISTRY_MAGIC) {
        return Err(EngineError::Durability(format!(
            "invalid WAL archive timeline registry header {}",
            path.display()
        )));
    }
    let expected_count: usize = parse_control_value(lines.next(), "timeline_count", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive timeline registry count {}: {err}",
                path.display()
            ))
        })?;
    let mut timelines = Vec::new();
    for line in lines {
        let raw = line.strip_prefix("timeline=").ok_or_else(|| {
            EngineError::Durability(format!(
                "invalid WAL archive timeline registry entry in {}",
                path.display()
            ))
        })?;
        let parts: Vec<&str> = raw.split('|').collect();
        if parts.len() != 6 {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive timeline registry entry in {}",
                path.display()
            )));
        }
        let parent_timeline_id = match parts[1] {
            "none" => None,
            parent => Some(parent.to_string()),
        };
        let entry = WalArchiveTimelineRegistryEntry {
            timeline_id: parts[0].to_string(),
            parent_timeline_id,
            fork_txn_id: parts[2].parse().map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive timeline registry fork transaction {}: {err}",
                    path.display()
                ))
            })?,
            fork_timestamp_micros: parse_optional_u64(
                parts[3],
                "timeline registry fork timestamp",
                path,
            )?,
            timeline_path: PathBuf::from(parts[4]),
            branch_manifest_path: PathBuf::from(parts[5]),
        };
        timelines.push(entry);
    }
    let registry = WalArchiveTimelineRegistry { timelines };
    if registry.timelines.len() != expected_count {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline registry {} expected {expected_count} timelines but found {}",
            path.display(),
            registry.timelines.len()
        )));
    }
    validate_timeline_registry_shape(path, &registry)?;
    Ok(registry)
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

fn validate_timeline_identity(
    timeline_path: &Path,
    timeline_id: &str,
    parent_timeline_id: Option<&str>,
    source_manifest_path: &Path,
    branch_manifest_path: &Path,
    fork_txn_id: TxnId,
    fork_timestamp_micros: Option<u64>,
) -> Result<WalArchiveTimeline, EngineError> {
    validate_timeline_value(timeline_path, "timeline_id", timeline_id)?;
    if let Some(parent) = parent_timeline_id {
        validate_timeline_value(timeline_path, "parent_timeline_id", parent)?;
        if parent == timeline_id {
            return Err(EngineError::Durability(format!(
                "WAL archive timeline {timeline_id} cannot be its own parent"
            )));
        }
    }
    if source_manifest_path == branch_manifest_path {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline {timeline_id} cannot fork into the source manifest {}",
            source_manifest_path.display()
        )));
    }
    validate_timeline_path(timeline_path, "source_manifest_path", source_manifest_path)?;
    validate_timeline_path(timeline_path, "branch_manifest_path", branch_manifest_path)?;
    Ok(WalArchiveTimeline {
        timeline_id: timeline_id.to_string(),
        parent_timeline_id: parent_timeline_id.map(ToOwned::to_owned),
        fork_txn_id,
        fork_timestamp_micros,
        source_manifest_path: source_manifest_path.to_path_buf(),
        branch_manifest_path: branch_manifest_path.to_path_buf(),
    })
}

fn validate_timeline_value(path: &Path, field: &str, value: &str) -> Result<(), EngineError> {
    if value.is_empty() {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline {field} must not be empty in {}",
            path.display()
        )));
    }
    if value == "none" || value.contains('\n') || value.contains('\r') {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline {field} contains unsupported value in {}",
            path.display()
        )));
    }
    Ok(())
}

fn validate_timeline_path(path: &Path, field: &str, value: &Path) -> Result<(), EngineError> {
    let rendered = value.to_string_lossy();
    if rendered.is_empty()
        || rendered.contains('\n')
        || rendered.contains('\r')
        || rendered.contains('|')
    {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline {field} contains unsupported path in {}",
            path.display()
        )));
    }
    Ok(())
}

fn validate_timeline_registry_shape(
    path: &Path,
    registry: &WalArchiveTimelineRegistry,
) -> Result<(), EngineError> {
    let mut seen = HashSet::new();
    for entry in &registry.timelines {
        validate_timeline_value(path, "timeline_id", &entry.timeline_id)?;
        if !seen.insert(entry.timeline_id.clone()) {
            return Err(EngineError::Durability(format!(
                "WAL archive timeline registry {} contains duplicate timeline {}",
                path.display(),
                entry.timeline_id
            )));
        }
        if let Some(parent) = entry.parent_timeline_id.as_ref() {
            validate_timeline_value(path, "parent_timeline_id", parent)?;
            if parent == &entry.timeline_id {
                return Err(EngineError::Durability(format!(
                    "WAL archive timeline {} cannot be its own parent",
                    entry.timeline_id
                )));
            }
            if !seen.contains(parent) {
                return Err(EngineError::Durability(format!(
                    "WAL archive timeline registry {} lists child {} before parent {}",
                    path.display(),
                    entry.timeline_id,
                    parent
                )));
            }
        }
        validate_timeline_path(path, "timeline_path", &entry.timeline_path)?;
        validate_timeline_path(path, "branch_manifest_path", &entry.branch_manifest_path)?;
    }
    Ok(())
}

fn validate_registered_timeline_entry(
    registry_path: &Path,
    entry: &WalArchiveTimelineRegistryEntry,
) -> Result<(), EngineError> {
    let timeline = read_wal_archive_timeline(&entry.timeline_path)?;
    if timeline.timeline_id != entry.timeline_id
        || timeline.parent_timeline_id != entry.parent_timeline_id
        || timeline.fork_txn_id != entry.fork_txn_id
        || timeline.fork_timestamp_micros != entry.fork_timestamp_micros
        || timeline.branch_manifest_path != entry.branch_manifest_path
    {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline registry {} entry {} does not match sidecar {}",
            registry_path.display(),
            entry.timeline_id,
            entry.timeline_path.display()
        )));
    }
    let (_manifest, _records) = read_wal_archive(&entry.branch_manifest_path)?;
    Ok(())
}

fn remove_wal_archive_timeline_artifact(kind: &str, path: &Path) -> Result<(), EngineError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(EngineError::Durability(format!(
            "failed to remove obsolete WAL archive timeline {kind} {}: {err}",
            path.display()
        ))),
    }
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
    let payload_len = u64::try_from(record.payload.len()).map_err(|_| {
        EngineError::Durability("WAL record payload length exceeds u64".to_string())
    })?;
    let checksum = wal_record_checksum(record.txn_id, payload_len, &record.payload);
    file.write_all(&record.txn_id.to_le_bytes())
        .and_then(|_| file.write_all(&payload_len.to_le_bytes()))
        .and_then(|_| file.write_all(&checksum.to_le_bytes()))
        .and_then(|_| file.write_all(&record.payload))
        .map_err(|err| EngineError::Durability(format!("failed to write WAL record: {err}")))
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
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in txn_id
        .to_le_bytes()
        .into_iter()
        .chain(payload_len.to_le_bytes())
        .chain(payload.iter().copied())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
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

    #[test]
    fn flush_commits_all_appended_records() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec(),
        });

        wal.flush_all().unwrap();

        assert_eq!(wal.flushed_count(), 2);
    }

    #[test]
    fn fail_next_flush_is_one_shot() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        });

        wal.fail_next_flush();
        let err = wal.flush_all().unwrap_err();
        assert!(matches!(err, EngineError::Durability(_)));
        assert_eq!(wal.flushed_count(), 0);

        wal.flush_all().unwrap();
        assert_eq!(wal.flushed_count(), 1);
    }

    #[test]
    fn truncate_shrinks_records_and_adjusts_flushed_count() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec(),
        });

        wal.flush_all().unwrap();
        assert_eq!(wal.flushed_count(), 2);

        wal.truncate(1);
        assert_eq!(wal.len(), 1);
        assert_eq!(wal.flushed_count(), 1);
    }

    #[test]
    fn unflushed_count_tracks_unpersisted_tail() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec(),
        });

        assert_eq!(wal.unflushed_count(), 2);

        wal.flush_all().unwrap();
        assert_eq!(wal.unflushed_count(), 0);

        wal.append(WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec(),
        });
        assert_eq!(wal.unflushed_count(), 1);
    }

    #[test]
    fn flushed_records_expose_only_durable_prefix() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec(),
        });

        assert!(wal.flushed_records().is_empty());

        wal.flush_all().unwrap();
        wal.append(WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec(),
        });

        let durable = wal.flushed_records();
        assert_eq!(durable.len(), 2);
        assert_eq!(durable[0].txn_id, 1);
        assert_eq!(durable[1].txn_id, 2);
    }

    #[test]
    fn flush_failure_does_not_advance_flushed_records() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        });

        wal.fail_next_flush();
        let _ = wal.flush_all();

        assert!(wal.flushed_records().is_empty());
        assert_eq!(wal.unflushed_count(), 1);
    }

    #[test]
    fn checkpoint_meta_tracks_durable_prefix_and_last_txn_id() {
        let mut wal = WalBuffer::default();
        assert_eq!(
            wal.checkpoint_meta(),
            WalCheckpointMeta {
                durable_record_count: 0,
                last_durable_txn_id: None,
            }
        );

        wal.append(WalRecord {
            txn_id: 7,
            payload: b"SET a=1".to_vec(),
        });
        wal.append(WalRecord {
            txn_id: 8,
            payload: b"SET b=2".to_vec(),
        });

        assert_eq!(
            wal.checkpoint_meta(),
            WalCheckpointMeta {
                durable_record_count: 0,
                last_durable_txn_id: None,
            }
        );

        wal.flush_all().unwrap();
        assert_eq!(
            wal.checkpoint_meta(),
            WalCheckpointMeta {
                durable_record_count: 2,
                last_durable_txn_id: Some(8),
            }
        );

        wal.append(WalRecord {
            txn_id: 9,
            payload: b"SET c=3".to_vec(),
        });
        assert_eq!(
            wal.checkpoint_meta(),
            WalCheckpointMeta {
                durable_record_count: 2,
                last_durable_txn_id: Some(8),
            }
        );
    }

    #[test]
    fn checkpoint_meta_does_not_advance_on_failed_flush() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 11,
            payload: b"SET a=1".to_vec(),
        });
        wal.flush_all().unwrap();

        wal.append(WalRecord {
            txn_id: 12,
            payload: b"SET b=2".to_vec(),
        });
        wal.fail_next_flush();
        assert!(wal.flush_all().is_err());

        assert_eq!(
            wal.checkpoint_meta(),
            WalCheckpointMeta {
                durable_record_count: 1,
                last_durable_txn_id: Some(11),
            }
        );
    }

    #[test]
    fn durable_flush_persists_records_to_real_segment() {
        let path = test_wal_path("durable-flush");
        let mut wal = WalBuffer::with_durable_segment(&path);
        assert!(wal.is_durable());
        assert_eq!(wal.durable_segment_path(), Some(path.as_path()));

        wal.append(WalRecord {
            txn_id: 1,
            payload: b"CREATE TABLE t (id INT)".to_vec(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"INSERT INTO t (id) VALUES (1)".to_vec(),
        });
        // Nothing on disk until the flush.
        assert!(!path.exists());

        wal.flush_all().unwrap();
        assert_eq!(wal.flushed_count(), 2);

        // The flushed records are now a real, CRC-checked, fsynced segment.
        let recovered = read_wal_segment(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].txn_id, 1);
        assert_eq!(
            recovered[1].payload,
            b"INSERT INTO t (id) VALUES (1)".to_vec()
        );
    }

    #[test]
    fn durable_flush_rewrites_full_prefix_so_history_is_preserved() {
        let path = test_wal_path("durable-history");
        let mut wal = WalBuffer::with_durable_segment(&path);

        wal.append(WalRecord {
            txn_id: 1,
            payload: b"one".to_vec(),
        });
        wal.flush_all().unwrap();
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"two".to_vec(),
        });
        wal.flush_all().unwrap();

        // The second flush must rewrite the segment with BOTH records, not just the new tail.
        let recovered = read_wal_segment(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].txn_id, 1);
        assert_eq!(recovered[1].txn_id, 2);
    }

    #[test]
    fn durable_group_commit_stats_count_one_group_per_fsync() {
        let path = test_wal_path("durable-groups");
        let mut wal = WalBuffer::with_durable_segment(&path);

        // Two records, then ONE flush => a single group of size 2.
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"a".to_vec(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"b".to_vec(),
        });
        wal.flush_all().unwrap();
        // One more record, separate flush => a second group of size 1.
        wal.append(WalRecord {
            txn_id: 3,
            payload: b"c".to_vec(),
        });
        wal.flush_all().unwrap();
        // A flush with nothing new must NOT count as a group (no fsync performed).
        wal.flush_all().unwrap();

        let _ = fs::remove_file(&path);
        let stats = wal.group_commit_stats();
        assert_eq!(stats.flush_groups, 2);
        assert_eq!(stats.durable_records, 3);
        assert_eq!(stats.max_group_size, 2);
        assert!((stats.mean_group_size() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn durable_flush_failure_leaves_no_durable_advance() {
        let path = test_wal_path("durable-fail");
        let mut wal = WalBuffer::with_durable_segment(&path);
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"a".to_vec(),
        });

        wal.fail_next_flush();
        let err = wal.flush_all().unwrap_err();
        assert!(matches!(err, EngineError::Durability(_)));
        // The watermark did not advance and (because the simulated failure short-circuits before
        // any I/O) the segment was never created — nothing partially durable.
        assert_eq!(wal.flushed_count(), 0);
        assert!(!path.exists());

        // A subsequent successful flush makes the record durable.
        wal.flush_all().unwrap();
        let recovered = read_wal_segment(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(recovered.len(), 1);
        assert_eq!(wal.flushed_count(), 1);
    }

    #[test]
    fn in_memory_flush_writes_no_segment() {
        // The default buffer is in-memory only: flush advances the watermark but touches no disk.
        let mut wal = WalBuffer::new();
        assert!(!wal.is_durable());
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"a".to_vec(),
        });
        wal.flush_all().unwrap();
        assert_eq!(wal.flushed_count(), 1);
        assert_eq!(wal.group_commit_stats(), WalGroupCommitStats::default());
    }

    #[test]
    fn reinstate_durable_records_seeds_flushed_prefix() {
        let mut wal = WalBuffer::new();
        wal.reinstate_durable_records(vec![
            WalRecord {
                txn_id: 1,
                payload: b"a".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"b".to_vec(),
            },
        ]);
        assert_eq!(wal.len(), 2);
        assert_eq!(wal.flushed_count(), 2);
        assert_eq!(wal.unflushed_count(), 0);
    }

    #[test]
    fn wal_segment_round_trips_durable_records() {
        let path = test_wal_path("roundtrip");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')".to_vec(),
            },
        ];

        write_wal_segment(&path, &records).unwrap();
        let recovered = read_wal_segment(&path).unwrap();
        let _ = fs::remove_file(path);

        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].txn_id, 1);
        assert_eq!(recovered[0].payload, records[0].payload);
        assert_eq!(recovered[1].txn_id, 2);
        assert_eq!(recovered[1].payload, records[1].payload);
    }

    #[test]
    fn wal_segment_rejects_checksum_mismatch() {
        let path = test_wal_path("checksum");
        let records = vec![WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        }];

        write_wal_segment(&path, &records).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.last_mut().unwrap();
        *last ^= 0x01;
        fs::write(&path, bytes).unwrap();

        let err = read_wal_segment(&path).unwrap_err();
        let _ = fs::remove_file(path);

        assert!(err.to_string().contains("checksum mismatch"));
    }

    #[test]
    fn wal_segment_rejects_truncated_record_header() {
        let path = test_wal_path("truncated");
        fs::write(
            &path,
            [WAL_SEGMENT_MAGIC.as_slice(), &[1_u8, 2, 3]].concat(),
        )
        .unwrap();

        let err = read_wal_segment(&path).unwrap_err();
        let _ = fs::remove_file(path);

        assert!(err.to_string().contains("record header"));
    }

    #[test]
    fn wal_control_file_round_trips_checkpoint_metadata() {
        let control_path = test_wal_path("control").with_extension("control");
        let control = WalControlFile {
            segment_path: PathBuf::from("segment-0001.wal"),
            checkpoint: WalCheckpointMeta {
                durable_record_count: 2,
                last_durable_txn_id: Some(42),
            },
        };

        write_wal_control_file(&control_path, &control).unwrap();
        let recovered = read_wal_control_file(&control_path).unwrap();
        let _ = fs::remove_file(control_path);

        assert_eq!(recovered, control);
    }

    #[test]
    fn wal_checkpoint_reads_segment_named_by_control_file() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-checkpoint-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let control_path = dir.join("CONTROL");
        let segment_path = dir.join("segment-0001.wal");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
        ];
        let control = WalControlFile {
            segment_path: PathBuf::from("segment-0001.wal"),
            checkpoint: WalCheckpointMeta {
                durable_record_count: 2,
                last_durable_txn_id: Some(2),
            },
        };

        write_wal_segment(&segment_path, &records).unwrap();
        write_wal_control_file(&control_path, &control).unwrap();
        let (recovered_control, recovered_records) = read_wal_checkpoint(&control_path).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(recovered_control, control);
        assert_eq!(recovered_records.len(), 2);
        assert_eq!(recovered_records[1].payload, b"SET b=2");
    }

    #[test]
    fn wal_checkpoint_rejects_control_record_count_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-checkpoint-mismatch-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let control_path = dir.join("CONTROL");
        let segment_path = dir.join("segment-0001.wal");
        let records = vec![WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        }];
        let control = WalControlFile {
            segment_path: PathBuf::from("segment-0001.wal"),
            checkpoint: WalCheckpointMeta {
                durable_record_count: 2,
                last_durable_txn_id: Some(1),
            },
        };

        write_wal_segment(&segment_path, &records).unwrap();
        write_wal_control_file(&control_path, &control).unwrap();
        let err = read_wal_checkpoint(&control_path).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("expected 2 durable records"));
    }

    #[test]
    fn wal_archive_round_trips_ordered_segments() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec(),
            },
        ];

        let manifest = write_wal_archive(&manifest_path, &segment_dir, &records, 2).unwrap();
        let (recovered_manifest, recovered_records) = read_wal_archive(&manifest_path).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(recovered_manifest, manifest);
        assert_eq!(recovered_manifest.segments.len(), 2);
        assert_eq!(
            recovered_manifest.checkpoint,
            WalCheckpointMeta {
                durable_record_count: 3,
                last_durable_txn_id: Some(3),
            }
        );
        assert_eq!(recovered_records.len(), 3);
        assert_eq!(recovered_records[2].payload, b"SET c=3");
    }

    #[test]
    fn wal_archive_object_backup_exports_and_restores_archive() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-object-backup-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("source").join("MANIFEST");
        let segment_dir = dir.join("source").join("segments");
        let backup_path = dir.join("backup").join("BACKUP");
        let object_dir = dir.join("backup").join("objects");
        let restored_manifest_path = dir.join("restored").join("MANIFEST");
        let restored_segment_dir = dir.join("restored").join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 3,
                timestamp_micros: 3_000,
            },
        ];

        let source_manifest = write_wal_archive_with_timestamps(
            &manifest_path,
            &segment_dir,
            &records,
            2,
            &timestamps,
        )
        .unwrap();
        let backup =
            export_wal_archive_object_backup(&manifest_path, &backup_path, &object_dir).unwrap();
        let restored_manifest = restore_wal_archive_object_backup(
            &backup_path,
            &restored_manifest_path,
            &restored_segment_dir,
        )
        .unwrap();
        let (_validated_manifest, restored_records) =
            read_wal_archive(&restored_manifest_path).unwrap();
        let (_timestamp_manifest, target, timestamp_records) =
            read_wal_archive_to_timestamp_micros(&restored_manifest_path, 2_000).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(backup.archive_manifest, source_manifest);
        assert_eq!(backup.objects.len(), 3);
        assert_eq!(restored_manifest.checkpoint, source_manifest.checkpoint);
        assert_eq!(restored_manifest.record_timestamps, timestamps);
        assert_eq!(restored_records, records);
        assert_eq!(target.target_txn_id, 2);
        assert_eq!(timestamp_records.len(), 2);
    }

    #[test]
    fn wal_archive_object_backup_rejects_corrupt_object_before_manifest_install() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-object-backup-corrupt-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("source").join("MANIFEST");
        let segment_dir = dir.join("source").join("segments");
        let backup_path = dir.join("backup").join("BACKUP");
        let object_dir = dir.join("backup").join("objects");
        let restored_manifest_path = dir.join("restored").join("MANIFEST");
        let restored_segment_dir = dir.join("restored").join("segments");
        let records = vec![WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        }];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let backup =
            export_wal_archive_object_backup(&manifest_path, &backup_path, &object_dir).unwrap();
        let segment_object = backup
            .objects
            .iter()
            .find(|object| object.source_path != Path::new("MANIFEST"))
            .unwrap();
        let object_path = resolve_manifest_path(&backup_path, &segment_object.object_path);
        fs::write(&object_path, b"corrupt wal object").unwrap();

        let err = restore_wal_archive_object_backup(
            &backup_path,
            &restored_manifest_path,
            &restored_segment_dir,
        )
        .unwrap_err();
        let manifest_installed = restored_manifest_path.exists();
        let _ = fs::remove_dir_all(dir);

        let err = err.to_string();
        assert!(
            err.contains("checksum mismatch")
                || (err.contains("expected") && err.contains("bytes"))
        );
        assert!(!manifest_installed);
    }

    #[test]
    fn wal_archive_object_backup_rejects_late_corrupt_object_before_segment_install() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-object-backup-late-corrupt-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("source").join("MANIFEST");
        let segment_dir = dir.join("source").join("segments");
        let backup_path = dir.join("backup").join("BACKUP");
        let object_dir = dir.join("backup").join("objects");
        let restored_manifest_path = dir.join("restored").join("MANIFEST");
        let restored_segment_dir = dir.join("restored").join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
        ];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let backup =
            export_wal_archive_object_backup(&manifest_path, &backup_path, &object_dir).unwrap();
        let second_segment_object = backup
            .objects
            .iter()
            .filter(|object| object.source_path != Path::new("MANIFEST"))
            .nth(1)
            .unwrap();
        let object_path = resolve_manifest_path(&backup_path, &second_segment_object.object_path);
        fs::write(&object_path, b"late corrupt wal object").unwrap();

        let err = restore_wal_archive_object_backup(
            &backup_path,
            &restored_manifest_path,
            &restored_segment_dir,
        )
        .unwrap_err();
        let manifest_installed = restored_manifest_path.exists();
        let segment_dir_installed = restored_segment_dir.exists();
        let staging_segment_dir_installed = restored_segment_dir
            .parent()
            .unwrap()
            .join(format!(
                ".{}.restore-{}",
                restored_segment_dir.file_name().unwrap().to_string_lossy(),
                std::process::id()
            ))
            .exists();
        let _ = fs::remove_dir_all(dir);

        let err = err.to_string();
        assert!(
            err.contains("checksum mismatch")
                || (err.contains("expected") && err.contains("bytes"))
        );
        assert!(!manifest_installed);
        assert!(!segment_dir_installed);
        assert!(!staging_segment_dir_installed);
    }

    #[test]
    fn wal_archive_object_backup_rejects_manifest_metadata_drift_before_install() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-object-backup-manifest-drift-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("source").join("MANIFEST");
        let segment_dir = dir.join("source").join("segments");
        let backup_path = dir.join("backup").join("BACKUP");
        let object_dir = dir.join("backup").join("objects");
        let restored_manifest_path = dir.join("restored").join("MANIFEST");
        let restored_segment_dir = dir.join("restored").join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
            .unwrap();
        export_wal_archive_object_backup(&manifest_path, &backup_path, &object_dir).unwrap();
        let backup_body = fs::read_to_string(&backup_path).unwrap();
        fs::write(
            &backup_path,
            backup_body.replace("record_timestamp=2|2000", "record_timestamp=2|2500"),
        )
        .unwrap();

        let err = restore_wal_archive_object_backup(
            &backup_path,
            &restored_manifest_path,
            &restored_segment_dir,
        )
        .unwrap_err();
        let manifest_installed = restored_manifest_path.exists();
        let segment_dir_installed = restored_segment_dir.exists();
        let _ = fs::remove_dir_all(dir);

        let err = err.to_string();
        assert!(err.contains("manifest object does not match backup manifest metadata"));
        assert!(!manifest_installed);
        assert!(!segment_dir_installed);
    }

    #[test]
    fn wal_archive_reads_prefix_to_transaction_target() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-target-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec(),
            },
        ];

        write_wal_archive(&manifest_path, &segment_dir, &records, 2).unwrap();
        let (_manifest, target, recovered_records) =
            read_wal_archive_to_txn(&manifest_path, 2).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(
            target,
            WalArchiveRecoveryTarget {
                target_txn_id: 2,
                recovered_record_count: 2,
                last_recovered_txn_id: 2,
            }
        );
        assert_eq!(recovered_records.len(), 2);
        assert_eq!(recovered_records[1].payload, b"SET b=2");
    }

    #[test]
    fn wal_archive_target_rejects_before_first_transaction() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-target-before-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![WalRecord {
            txn_id: 10,
            payload: b"SET a=1".to_vec(),
        }];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let err = read_wal_archive_to_txn(&manifest_path, 9).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err
            .to_string()
            .contains("before first archived transaction"));
    }

    #[test]
    fn wal_archive_target_rejects_beyond_durable_archive() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-target-beyond-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        }];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let err = read_wal_archive_to_txn(&manifest_path, 2).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("beyond last durable transaction"));
    }

    #[test]
    fn wal_archive_target_rejects_missing_transaction_boundary() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-target-missing-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec(),
            },
        ];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let err = read_wal_archive_to_txn(&manifest_path, 2).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err
            .to_string()
            .contains("does not contain target transaction"));
    }

    #[test]
    fn wal_archive_reads_prefix_to_timestamp_target() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timestamp-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 3,
                timestamp_micros: 3_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 2, &timestamps)
            .unwrap();
        let (_manifest, target, recovered_records) =
            read_wal_archive_to_timestamp_micros(&manifest_path, 2_000).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(
            target,
            WalArchiveTimestampRecoveryTarget {
                target_timestamp_micros: 2_000,
                target_txn_id: 2,
                recovered_record_count: 2,
                last_recovered_txn_id: 2,
            }
        );
        assert_eq!(recovered_records.len(), 2);
        assert_eq!(recovered_records[1].payload, b"SET b=2");
    }

    #[test]
    fn wal_archive_timestamp_target_rejects_missing_metadata() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timestamp-missing-meta-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        }];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let err = read_wal_archive_to_timestamp_micros(&manifest_path, 1_000).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("no timestamp metadata"));
    }

    #[test]
    fn wal_archive_timestamp_target_rejects_unavailable_boundaries() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timestamp-unavailable-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 10,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 20,
                payload: b"SET b=2".to_vec(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 10,
                timestamp_micros: 10_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 20,
                timestamp_micros: 20_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
            .unwrap();
        let before = read_wal_archive_to_timestamp_micros(&manifest_path, 9_999).unwrap_err();
        let between = read_wal_archive_to_timestamp_micros(&manifest_path, 15_000).unwrap_err();
        let beyond = read_wal_archive_to_timestamp_micros(&manifest_path, 20_001).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(before
            .to_string()
            .contains("before first archived timestamp"));
        assert!(between
            .to_string()
            .contains("falls between archived transaction boundaries"));
        assert!(beyond.to_string().contains("beyond last durable timestamp"));
    }

    #[test]
    fn wal_archive_timestamp_target_rejects_ambiguous_boundary() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timestamp-ambiguous-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 1_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
            .unwrap();
        let err = read_wal_archive_to_timestamp_micros(&manifest_path, 1_000).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("ambiguous"));
    }

    #[test]
    fn wal_archive_ingests_next_segment_and_preserves_timestamps() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-ingest-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let ingest_segment = segment_dir.join("segment-0002.wal");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
        ];
        let ingest_records = vec![
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec(),
            },
            WalRecord {
                txn_id: 4,
                payload: b"SET d=4".to_vec(),
            },
        ];
        let ingest_timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 3,
                timestamp_micros: 3_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 4,
                timestamp_micros: 4_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 2, &timestamps)
            .unwrap();
        write_wal_segment(&ingest_segment, &ingest_records).unwrap();
        let manifest = append_wal_archive_segment_with_timestamps(
            &manifest_path,
            &ingest_segment,
            &ingest_timestamps,
        )
        .unwrap();
        let (_read_manifest, read_records) = read_wal_archive(&manifest_path).unwrap();
        let (_manifest, target, target_records) =
            read_wal_archive_to_timestamp_micros(&manifest_path, 4_000).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(manifest.segments.len(), 2);
        assert_eq!(manifest.checkpoint.durable_record_count, 4);
        assert_eq!(manifest.checkpoint.last_durable_txn_id, Some(4));
        assert_eq!(manifest.record_timestamps.len(), 4);
        assert_eq!(read_records.len(), 4);
        assert_eq!(read_records[3].payload, b"SET d=4");
        assert_eq!(target.target_txn_id, 4);
        assert_eq!(target_records.len(), 4);
    }

    #[test]
    fn wal_archive_ingest_rejects_non_increasing_segment_without_manifest_mutation() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-ingest-reject-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let ingest_segment = segment_dir.join("segment-0002.wal");
        let records = vec![WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec(),
        }];
        let ingest_records = vec![WalRecord {
            txn_id: 2,
            payload: b"SET duplicate=2".to_vec(),
        }];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let before_manifest = fs::read_to_string(&manifest_path).unwrap();
        write_wal_segment(&ingest_segment, &ingest_records).unwrap();
        let err = append_wal_archive_segment(&manifest_path, &ingest_segment).unwrap_err();
        let after_manifest = fs::read_to_string(&manifest_path).unwrap();
        let (_manifest, read_records) = read_wal_archive(&manifest_path).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("not after durable transaction 2"));
        assert_eq!(after_manifest, before_manifest);
        assert_eq!(read_records.len(), 1);
        assert_eq!(read_records[0].payload, b"SET b=2");
    }

    #[test]
    fn wal_archive_ingest_requires_timestamp_metadata_when_archive_has_timestamps() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-ingest-timestamp-required-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let ingest_segment = segment_dir.join("segment-0002.wal");
        let records = vec![WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        }];
        let timestamps = vec![WalArchiveRecordTimestamp {
            txn_id: 1,
            timestamp_micros: 1_000,
        }];
        let ingest_records = vec![WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec(),
        }];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
            .unwrap();
        let before_manifest = fs::read_to_string(&manifest_path).unwrap();
        write_wal_segment(&ingest_segment, &ingest_records).unwrap();
        let err = append_wal_archive_segment(&manifest_path, &ingest_segment).unwrap_err();
        let after_manifest = fs::read_to_string(&manifest_path).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert!(err
            .to_string()
            .contains("requires timestamp metadata for ingested segment"));
        assert_eq!(after_manifest, before_manifest);
    }

    #[test]
    fn wal_archive_forks_transaction_timeline_with_ancestry() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timeline-txn-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let source_manifest = dir.join("source").join("MANIFEST");
        let source_segments = dir.join("source").join("segments");
        let branch_manifest = dir.join("branch").join("MANIFEST");
        let branch_segments = dir.join("branch").join("segments");
        let timeline_path = dir.join("branch").join("TIMELINE");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')".to_vec(),
            },
        ];
        write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();

        let branch = fork_wal_archive_timeline_to_txn(
            &source_manifest,
            &branch_manifest,
            &branch_segments,
            &timeline_path,
            "timeline-0002",
            Some("timeline-0001"),
            2,
        )
        .unwrap();
        let timeline = read_wal_archive_timeline(&timeline_path).unwrap();
        let (_manifest, branch_records) = read_wal_archive(&branch_manifest).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(branch.timeline, timeline);
        assert_eq!(timeline.timeline_id, "timeline-0002");
        assert_eq!(
            timeline.parent_timeline_id.as_deref(),
            Some("timeline-0001")
        );
        assert_eq!(timeline.fork_txn_id, 2);
        assert_eq!(timeline.fork_timestamp_micros, None);
        assert_eq!(branch.manifest.checkpoint.durable_record_count, 2);
        assert_eq!(branch.manifest.checkpoint.last_durable_txn_id, Some(2));
        assert_eq!(branch_records.len(), 2);
        assert_eq!(branch_records[1].txn_id, 2);
    }

    #[test]
    fn wal_archive_forks_timestamp_timeline_and_rejects_self_parent_without_mutation() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timeline-timestamp-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let source_manifest = dir.join("source").join("MANIFEST");
        let source_segments = dir.join("source").join("segments");
        let branch_manifest = dir.join("branch").join("MANIFEST");
        let branch_segments = dir.join("branch").join("segments");
        let timeline_path = dir.join("branch").join("TIMELINE");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')".to_vec(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 3,
                timestamp_micros: 3_000,
            },
        ];
        write_wal_archive_with_timestamps(
            &source_manifest,
            &source_segments,
            &records,
            2,
            &timestamps,
        )
        .unwrap();

        let self_parent_err = fork_wal_archive_timeline_to_timestamp_micros(
            &source_manifest,
            &branch_manifest,
            &branch_segments,
            &timeline_path,
            "timeline-0002",
            Some("timeline-0002"),
            2_000,
        )
        .unwrap_err();
        assert!(!branch_manifest.exists());
        assert!(self_parent_err
            .to_string()
            .contains("cannot be its own parent"));

        let branch = fork_wal_archive_timeline_to_timestamp_micros(
            &source_manifest,
            &branch_manifest,
            &branch_segments,
            &timeline_path,
            "timeline-0002",
            Some("timeline-0001"),
            2_000,
        )
        .unwrap();
        let timeline = read_wal_archive_timeline(&timeline_path).unwrap();
        let (_manifest, branch_records) = read_wal_archive(&branch_manifest).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(timeline.fork_txn_id, 2);
        assert_eq!(timeline.fork_timestamp_micros, Some(2_000));
        assert_eq!(branch.manifest.record_timestamps.len(), 2);
        assert_eq!(branch_records.len(), 2);
        assert_eq!(branch.timeline, timeline);
    }

    #[test]
    fn wal_archive_timeline_registry_requires_parent_before_child_and_unique_ids() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timeline-registry-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let source_manifest = dir.join("source").join("MANIFEST");
        let source_segments = dir.join("source").join("segments");
        let root_timeline_path = dir.join("source").join("TIMELINE");
        let branch_manifest = dir.join("branch").join("MANIFEST");
        let branch_segments = dir.join("branch").join("segments");
        let branch_timeline_path = dir.join("branch").join("TIMELINE");
        let missing_parent_branch_manifest = dir.join("missing-parent").join("MANIFEST");
        let missing_parent_branch_segments = dir.join("missing-parent").join("segments");
        let missing_parent_timeline_path = dir.join("missing-parent").join("TIMELINE");
        let registry_path = dir.join("TIMELINE_REGISTRY");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')".to_vec(),
            },
        ];
        write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();
        write_wal_archive_timeline(
            &root_timeline_path,
            &WalArchiveTimeline {
                timeline_id: "timeline-0001".to_string(),
                parent_timeline_id: None,
                fork_txn_id: 0,
                fork_timestamp_micros: None,
                source_manifest_path: source_manifest.clone(),
                branch_manifest_path: source_manifest.clone(),
            },
        )
        .unwrap();
        fork_wal_archive_timeline_to_txn(
            &source_manifest,
            &branch_manifest,
            &branch_segments,
            &branch_timeline_path,
            "timeline-0002",
            Some("timeline-0001"),
            2,
        )
        .unwrap();
        fork_wal_archive_timeline_to_txn(
            &source_manifest,
            &missing_parent_branch_manifest,
            &missing_parent_branch_segments,
            &missing_parent_timeline_path,
            "timeline-0003",
            Some("timeline-missing"),
            2,
        )
        .unwrap();

        let missing_parent_err =
            register_wal_archive_timeline(&registry_path, &missing_parent_timeline_path)
                .unwrap_err();
        assert!(missing_parent_err
            .to_string()
            .contains("missing parent timeline timeline-missing"));
        assert!(!registry_path.exists());

        let root_registry =
            register_wal_archive_timeline(&registry_path, &root_timeline_path).unwrap();
        assert_eq!(root_registry.timelines.len(), 1);
        assert_eq!(root_registry.timelines[0].timeline_id, "timeline-0001");

        let registry =
            register_wal_archive_timeline(&registry_path, &branch_timeline_path).unwrap();
        assert_eq!(registry.timelines.len(), 2);
        assert_eq!(registry.timelines[1].timeline_id, "timeline-0002");
        assert_eq!(
            registry.timelines[1].parent_timeline_id.as_deref(),
            Some("timeline-0001")
        );
        assert_eq!(registry.timelines[1].fork_txn_id, 2);

        let before_registry = fs::read_to_string(&registry_path).unwrap();
        let duplicate_err =
            register_wal_archive_timeline(&registry_path, &branch_timeline_path).unwrap_err();
        let after_registry = fs::read_to_string(&registry_path).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert!(duplicate_err
            .to_string()
            .contains("already contains timeline timeline-0002"));
        assert_eq!(after_registry, before_registry);
    }

    #[test]
    fn wal_archive_timeline_registry_selects_validated_branch_and_rejects_stale_sidecar() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timeline-select-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let source_manifest = dir.join("source").join("MANIFEST");
        let source_segments = dir.join("source").join("segments");
        let root_timeline_path = dir.join("source").join("TIMELINE");
        let branch_manifest = dir.join("branch").join("MANIFEST");
        let branch_segments = dir.join("branch").join("segments");
        let branch_timeline_path = dir.join("branch").join("TIMELINE");
        let registry_path = dir.join("TIMELINE_REGISTRY");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')".to_vec(),
            },
        ];
        write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();
        write_wal_archive_timeline(
            &root_timeline_path,
            &WalArchiveTimeline {
                timeline_id: "timeline-0001".to_string(),
                parent_timeline_id: None,
                fork_txn_id: 0,
                fork_timestamp_micros: None,
                source_manifest_path: source_manifest.clone(),
                branch_manifest_path: source_manifest.clone(),
            },
        )
        .unwrap();
        fork_wal_archive_timeline_to_txn(
            &source_manifest,
            &branch_manifest,
            &branch_segments,
            &branch_timeline_path,
            "timeline-0002",
            Some("timeline-0001"),
            2,
        )
        .unwrap();
        register_wal_archive_timeline(&registry_path, &root_timeline_path).unwrap();
        register_wal_archive_timeline(&registry_path, &branch_timeline_path).unwrap();

        let selection = select_wal_archive_timeline(&registry_path, "timeline-0002").unwrap();
        assert_eq!(selection.entry.timeline_id, "timeline-0002");
        assert_eq!(selection.timeline.fork_txn_id, 2);
        assert_eq!(selection.manifest.checkpoint.durable_record_count, 2);

        let missing_err =
            select_wal_archive_timeline(&registry_path, "timeline-missing").unwrap_err();
        assert!(missing_err
            .to_string()
            .contains("has no timeline timeline-missing"));

        write_wal_archive_timeline(
            &branch_timeline_path,
            &WalArchiveTimeline {
                timeline_id: "timeline-0002".to_string(),
                parent_timeline_id: Some("timeline-0001".to_string()),
                fork_txn_id: 3,
                fork_timestamp_micros: None,
                source_manifest_path: source_manifest.clone(),
                branch_manifest_path: branch_manifest.clone(),
            },
        )
        .unwrap();
        let stale_err = select_wal_archive_timeline(&registry_path, "timeline-0002").unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(stale_err.to_string().contains("does not match sidecar"));
    }

    #[test]
    fn wal_archive_timeline_prune_keeps_target_ancestry_and_removes_unreferenced_artifacts() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timeline-prune-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let source_manifest = dir.join("source").join("MANIFEST");
        let source_segments = dir.join("source").join("segments");
        let source_timeline_path = dir.join("source").join("TIMELINE");
        let keep_manifest = dir.join("keep").join("MANIFEST");
        let keep_segments = dir.join("keep").join("segments");
        let keep_timeline_path = dir.join("keep").join("TIMELINE");
        let prune_manifest = dir.join("prune").join("MANIFEST");
        let prune_segments = dir.join("prune").join("segments");
        let prune_timeline_path = dir.join("prune").join("TIMELINE");
        let registry_path = dir.join("TIMELINE_REGISTRY");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')".to_vec(),
            },
        ];
        write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();
        write_wal_archive_timeline(
            &source_timeline_path,
            &WalArchiveTimeline {
                timeline_id: "timeline-main-0001".to_string(),
                parent_timeline_id: None,
                fork_txn_id: 0,
                fork_timestamp_micros: None,
                source_manifest_path: source_manifest.clone(),
                branch_manifest_path: source_manifest.clone(),
            },
        )
        .unwrap();
        fork_wal_archive_timeline_to_txn(
            &source_manifest,
            &keep_manifest,
            &keep_segments,
            &keep_timeline_path,
            "timeline-keep-0002",
            Some("timeline-main-0001"),
            3,
        )
        .unwrap();
        fork_wal_archive_timeline_to_txn(
            &source_manifest,
            &prune_manifest,
            &prune_segments,
            &prune_timeline_path,
            "timeline-prune-0003",
            Some("timeline-main-0001"),
            2,
        )
        .unwrap();
        register_wal_archive_timeline(&registry_path, &source_timeline_path).unwrap();
        register_wal_archive_timeline(&registry_path, &keep_timeline_path).unwrap();
        register_wal_archive_timeline(&registry_path, &prune_timeline_path).unwrap();

        let plan = plan_wal_archive_timeline_prune(&registry_path, "timeline-keep-0002").unwrap();
        let applied =
            apply_wal_archive_timeline_prune(&registry_path, "timeline-keep-0002").unwrap();
        let registry = read_wal_archive_timeline_registry(&registry_path).unwrap();
        let selection = select_wal_archive_timeline(&registry_path, "timeline-keep-0002").unwrap();
        let pruned_err =
            select_wal_archive_timeline(&registry_path, "timeline-prune-0003").unwrap_err();

        assert_eq!(plan.retained_timeline_id, "timeline-keep-0002");
        assert_eq!(
            plan.retained_timeline_ids,
            vec![
                "timeline-main-0001".to_string(),
                "timeline-keep-0002".to_string()
            ]
        );
        assert_eq!(
            plan.removed_timeline_ids,
            vec!["timeline-prune-0003".to_string()]
        );
        assert_eq!(applied, plan);
        assert_eq!(registry.timelines.len(), 2);
        assert_eq!(selection.manifest.checkpoint.durable_record_count, 3);
        assert!(!prune_timeline_path.exists());
        assert!(!prune_manifest.exists());
        assert!(!prune_segments.join("segment-0001.wal").exists());
        assert!(keep_timeline_path.exists());
        assert!(keep_manifest.exists());
        assert!(keep_segments.join("segment-0001.wal").exists());
        assert!(pruned_err
            .to_string()
            .contains("has no timeline timeline-prune-0003"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn wal_archive_timeline_prune_rejects_stale_sidecar_without_registry_mutation() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timeline-prune-stale-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let source_manifest = dir.join("source").join("MANIFEST");
        let source_segments = dir.join("source").join("segments");
        let source_timeline_path = dir.join("source").join("TIMELINE");
        let branch_manifest = dir.join("branch").join("MANIFEST");
        let branch_segments = dir.join("branch").join("segments");
        let branch_timeline_path = dir.join("branch").join("TIMELINE");
        let registry_path = dir.join("TIMELINE_REGISTRY");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')".to_vec(),
            },
        ];
        write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();
        write_wal_archive_timeline(
            &source_timeline_path,
            &WalArchiveTimeline {
                timeline_id: "timeline-main-0001".to_string(),
                parent_timeline_id: None,
                fork_txn_id: 0,
                fork_timestamp_micros: None,
                source_manifest_path: source_manifest.clone(),
                branch_manifest_path: source_manifest.clone(),
            },
        )
        .unwrap();
        fork_wal_archive_timeline_to_txn(
            &source_manifest,
            &branch_manifest,
            &branch_segments,
            &branch_timeline_path,
            "timeline-branch-0002",
            Some("timeline-main-0001"),
            2,
        )
        .unwrap();
        register_wal_archive_timeline(&registry_path, &source_timeline_path).unwrap();
        register_wal_archive_timeline(&registry_path, &branch_timeline_path).unwrap();
        let before_registry = fs::read_to_string(&registry_path).unwrap();
        write_wal_archive_timeline(
            &branch_timeline_path,
            &WalArchiveTimeline {
                timeline_id: "timeline-branch-0002".to_string(),
                parent_timeline_id: Some("timeline-main-0001".to_string()),
                fork_txn_id: 1,
                fork_timestamp_micros: None,
                source_manifest_path: source_manifest.clone(),
                branch_manifest_path: branch_manifest.clone(),
            },
        )
        .unwrap();

        let err =
            apply_wal_archive_timeline_prune(&registry_path, "timeline-branch-0002").unwrap_err();
        let after_registry = fs::read_to_string(&registry_path).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("does not match sidecar"));
        assert_eq!(after_registry, before_registry);
    }

    #[test]
    fn wal_archive_retention_plan_keeps_exact_transaction_prefix() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-retention-plan-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec(),
            },
            WalRecord {
                txn_id: 4,
                payload: b"SET d=4".to_vec(),
            },
            WalRecord {
                txn_id: 5,
                payload: b"SET e=5".to_vec(),
            },
        ];

        write_wal_archive(&manifest_path, &segment_dir, &records, 2).unwrap();
        let plan = plan_wal_archive_retention_to_txn(&manifest_path, 3).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(plan.target_txn_id, 3);
        assert_eq!(plan.retained_record_count, 3);
        assert_eq!(plan.removed_record_count, 2);
        assert_eq!(plan.retained_manifest.segments.len(), 2);
        assert_eq!(
            plan.retained_manifest.checkpoint,
            WalCheckpointMeta {
                durable_record_count: 3,
                last_durable_txn_id: Some(3),
            }
        );
        assert_eq!(plan.retained_manifest.segments[1].record_count, 1);
        assert_eq!(plan.retained_manifest.segments[1].last_txn_id, Some(3));
        assert_eq!(
            plan.removed_segments,
            vec![segment_dir.join("segment-0003.wal")]
        );
    }

    #[test]
    fn wal_archive_retention_apply_rewrites_manifest_and_removes_tail_segments() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-retention-apply-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec(),
            },
            WalRecord {
                txn_id: 4,
                payload: b"SET d=4".to_vec(),
            },
            WalRecord {
                txn_id: 5,
                payload: b"SET e=5".to_vec(),
            },
        ];

        write_wal_archive(&manifest_path, &segment_dir, &records, 2).unwrap();
        let removed_tail = segment_dir.join("segment-0003.wal");
        assert!(removed_tail.exists());

        let plan = apply_wal_archive_retention_to_txn(&manifest_path, 3).unwrap();
        let (retained_manifest, retained_records) = read_wal_archive(&manifest_path).unwrap();
        let target_err = read_wal_archive_to_txn(&manifest_path, 4).unwrap_err();

        assert_eq!(plan.retained_record_count, 3);
        assert_eq!(plan.removed_record_count, 2);
        assert!(!removed_tail.exists());
        assert_eq!(retained_manifest.checkpoint.durable_record_count, 3);
        assert_eq!(retained_manifest.checkpoint.last_durable_txn_id, Some(3));
        assert_eq!(retained_records.len(), 3);
        assert_eq!(retained_records[2].payload, b"SET c=3");
        assert!(target_err
            .to_string()
            .contains("beyond last durable transaction"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn wal_archive_timestamp_retention_rewrites_manifest_and_preserves_timestamp_prefix() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timestamp-retention-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec(),
            },
            WalRecord {
                txn_id: 4,
                payload: b"SET d=4".to_vec(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 3,
                timestamp_micros: 3_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 4,
                timestamp_micros: 4_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
            .unwrap();
        let removed_tail = segment_dir.join("segment-0004.wal");
        assert!(removed_tail.exists());

        let plan = apply_wal_archive_retention_to_timestamp_micros(&manifest_path, 3_000).unwrap();
        let (retained_manifest, retained_records) = read_wal_archive(&manifest_path).unwrap();
        let target_err = read_wal_archive_to_timestamp_micros(&manifest_path, 4_000).unwrap_err();

        assert_eq!(plan.target_txn_id, 3);
        assert_eq!(plan.retained_record_count, 3);
        assert_eq!(plan.removed_record_count, 1);
        assert!(!removed_tail.exists());
        assert_eq!(retained_manifest.checkpoint.durable_record_count, 3);
        assert_eq!(retained_manifest.checkpoint.last_durable_txn_id, Some(3));
        assert_eq!(
            retained_manifest.record_timestamps,
            vec![
                WalArchiveRecordTimestamp {
                    txn_id: 1,
                    timestamp_micros: 1_000,
                },
                WalArchiveRecordTimestamp {
                    txn_id: 2,
                    timestamp_micros: 2_000,
                },
                WalArchiveRecordTimestamp {
                    txn_id: 3,
                    timestamp_micros: 3_000,
                },
            ]
        );
        assert_eq!(retained_records.len(), 3);
        assert_eq!(retained_records[2].payload, b"SET c=3");
        assert!(target_err
            .to_string()
            .contains("beyond last durable timestamp"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn wal_archive_timestamp_retention_rejects_between_boundary_without_mutation() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timestamp-retention-missing-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 10,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 20,
                payload: b"SET b=2".to_vec(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 10,
                timestamp_micros: 10_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 20,
                timestamp_micros: 20_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
            .unwrap();
        let before_manifest = fs::read_to_string(&manifest_path).unwrap();
        let err =
            apply_wal_archive_retention_to_timestamp_micros(&manifest_path, 15_000).unwrap_err();
        let after_manifest = fs::read_to_string(&manifest_path).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert!(err
            .to_string()
            .contains("falls between archived transaction boundaries"));
        assert_eq!(after_manifest, before_manifest);
    }

    #[test]
    fn wal_archive_base_retention_plan_keeps_base_boundary_suffix() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-base-retention-plan-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec(),
            },
            WalRecord {
                txn_id: 4,
                payload: b"SET d=4".to_vec(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 3,
                timestamp_micros: 3_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 4,
                timestamp_micros: 4_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
            .unwrap();
        let plan = plan_wal_archive_retention_from_txn(&manifest_path, 2).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(plan.target_txn_id, 2);
        assert_eq!(plan.retained_record_count, 3);
        assert_eq!(plan.removed_record_count, 1);
        assert_eq!(plan.retained_manifest.checkpoint.durable_record_count, 3);
        assert_eq!(
            plan.retained_manifest.checkpoint.last_durable_txn_id,
            Some(4)
        );
        assert_eq!(plan.retained_manifest.segments[0].first_txn_id, Some(2));
        assert_eq!(plan.retained_manifest.record_timestamps[0].txn_id, 2);
        assert_eq!(
            plan.retained_manifest.record_timestamps[0].timestamp_micros,
            2_000
        );
    }

    #[test]
    fn wal_archive_base_retention_apply_rewrites_to_base_suffix() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-base-retention-apply-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec(),
            },
            WalRecord {
                txn_id: 4,
                payload: b"SET d=4".to_vec(),
            },
        ];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let plan = apply_wal_archive_retention_from_txn(&manifest_path, 2).unwrap();
        let (retained_manifest, retained_records) = read_wal_archive(&manifest_path).unwrap();
        let target_err = read_wal_archive_to_txn(&manifest_path, 1).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(plan.retained_record_count, 3);
        assert_eq!(plan.removed_record_count, 1);
        assert_eq!(retained_manifest.checkpoint.durable_record_count, 3);
        assert_eq!(retained_manifest.checkpoint.last_durable_txn_id, Some(4));
        assert_eq!(
            retained_records
                .iter()
                .map(|record| record.txn_id)
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
        assert!(target_err
            .to_string()
            .contains("before first archived transaction"));
    }

    #[test]
    fn wal_archive_retention_rejects_malformed_archive_before_cleanup() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-retention-malformed-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            },
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            },
        ];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let err = apply_wal_archive_retention_to_txn(&manifest_path, 1).unwrap_err();
        assert!(segment_dir.join("segment-0002.wal").exists());
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("non-increasing transaction order"));
    }

    #[test]
    fn wal_archive_rejects_missing_segment() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-missing-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        }];

        let manifest = write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        fs::remove_file(resolve_manifest_path(
            &manifest_path,
            &manifest.segments[0].segment_path,
        ))
        .unwrap();
        let err = read_wal_archive(&manifest_path).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("failed to open WAL segment"));
    }

    #[test]
    fn wal_archive_rejects_manifest_record_count_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-count-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_path = dir.join("segment-0001.wal");
        write_wal_segment(
            &segment_path,
            &[WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            }],
        )
        .unwrap();
        let manifest = WalArchiveManifest {
            segments: vec![WalArchiveSegment {
                segment_path: PathBuf::from("segment-0001.wal"),
                record_count: 2,
                first_txn_id: Some(1),
                last_txn_id: Some(1),
            }],
            checkpoint: WalCheckpointMeta {
                durable_record_count: 2,
                last_durable_txn_id: Some(1),
            },
            record_timestamps: Vec::new(),
        };
        write_wal_archive_manifest(&manifest_path, &manifest).unwrap();

        let err = read_wal_archive(&manifest_path).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("expected 2 records"));
    }

    #[test]
    fn wal_archive_rejects_non_increasing_transaction_order() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-order-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_a = dir.join("segment-0001.wal");
        let segment_b = dir.join("segment-0002.wal");
        write_wal_segment(
            &segment_a,
            &[WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec(),
            }],
        )
        .unwrap();
        write_wal_segment(
            &segment_b,
            &[WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec(),
            }],
        )
        .unwrap();
        let manifest = WalArchiveManifest {
            segments: vec![
                WalArchiveSegment {
                    segment_path: PathBuf::from("segment-0001.wal"),
                    record_count: 1,
                    first_txn_id: Some(2),
                    last_txn_id: Some(2),
                },
                WalArchiveSegment {
                    segment_path: PathBuf::from("segment-0002.wal"),
                    record_count: 1,
                    first_txn_id: Some(1),
                    last_txn_id: Some(1),
                },
            ],
            checkpoint: WalCheckpointMeta {
                durable_record_count: 2,
                last_durable_txn_id: Some(1),
            },
            record_timestamps: Vec::new(),
        };
        write_wal_archive_manifest(&manifest_path, &manifest).unwrap();

        let err = read_wal_archive(&manifest_path).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("non-increasing transaction order"));
    }
}
