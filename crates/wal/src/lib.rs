use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use gpu_db_types::{EngineError, TxnId};

const WAL_SEGMENT_MAGIC: &[u8; 10] = b"GPUDBWAL1\n";
const WAL_CONTROL_MAGIC: &str = "GPUDBWALCONTROL1";
const WAL_ARCHIVE_MANIFEST_MAGIC: &str = "GPUDBWALARCHIVE1";
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

#[derive(Debug, Default)]
pub struct WalBuffer {
    records: Vec<WalRecord>,
    flushed: usize,
    fail_next_flush: bool,
}

impl WalBuffer {
    pub fn append(&mut self, rec: WalRecord) {
        self.records.push(rec);
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

    pub fn flush_all(&mut self) -> Result<(), EngineError> {
        if self.fail_next_flush {
            self.fail_next_flush = false;
            return Err(EngineError::Durability(
                "simulated wal flush failure".to_string(),
            ));
        }
        self.flushed = self.records.len();
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
