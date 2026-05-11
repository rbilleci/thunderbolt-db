use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use gpu_db_types::{EngineError, TxnId};

const WAL_SEGMENT_MAGIC: &[u8; 10] = b"GPUDBWAL1\n";
const WAL_CONTROL_MAGIC: &str = "GPUDBWALCONTROL1";
const WAL_RECORD_HEADER_LEN: usize = 24;

#[derive(Debug, Clone)]
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
}
