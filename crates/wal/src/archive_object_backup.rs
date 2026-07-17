//! WAL archive object export, verified storage, and restore.

use super::*;
use sha2::{Digest, Sha256};

const WAL_ARCHIVE_OBJECT_BACKUP_MAGIC_V1: &str = "GPUDBWALOBJECTBACKUP1";
const WAL_ARCHIVE_OBJECT_BACKUP_MAGIC: &str = "GPUDBWALOBJECTBACKUP2";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveObject {
    pub source_path: PathBuf,
    pub object_path: PathBuf,
    pub byte_len: u64,
    /// Legacy v1 compatibility checksum.
    pub checksum: u64,
    /// Collision-resistant v2 object authority.
    pub sha256: Option<CanonicalDigest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveObjectBackup {
    pub archive_manifest: WalArchiveManifest,
    pub objects: Vec<WalArchiveObject>,
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

    create_wal_dir_all(object_dir).map_err(|err| {
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
        append_sha256_trailer(render_wal_archive_manifest_body(&backup.archive_manifest)?)
            .into_bytes();
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
    create_wal_dir_all(restored_segment_parent).map_err(|err| {
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
    create_wal_dir_all(&staging_segment_dir).map_err(|err| {
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
        sync_wal_parent_dir(restored_segment_dir)?;
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
        create_wal_dir_all(parent).map_err(|err| {
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
        let sha256 = object.sha256.ok_or_else(|| {
            EngineError::Durability(format!(
                "WAL archive object {} lacks its v2 SHA-256 authority",
                object.object_path.display()
            ))
        })?;
        body.push_str(&format!(
            "object={}|{}|{}|{}|{}\n",
            object.source_path.display(),
            object.object_path.display(),
            object.byte_len,
            object.checksum,
            format_sha256(sha256)
        ));
    }
    let body = append_sha256_trailer(body);

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
    })?;
    sync_wal_parent_dir(path)
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
    let is_v2 = body.lines().next() == Some(WAL_ARCHIVE_OBJECT_BACKUP_MAGIC);
    let body = match body.lines().next() {
        Some(WAL_ARCHIVE_OBJECT_BACKUP_MAGIC) => verify_sha256_trailer(&body, path)?,
        Some(WAL_ARCHIVE_OBJECT_BACKUP_MAGIC_V1) => body,
        _ => {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive object backup header {}",
                path.display()
            )))
        }
    };
    let mut lines = body.lines();
    let _magic = lines.next();

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
        let sha256 = if is_v2 {
            Some(parse_sha256(
                parts.next().ok_or_else(|| {
                    EngineError::Durability(format!(
                        "missing WAL archive object backup SHA-256 in {}",
                        path.display()
                    ))
                })?,
                path,
            )?)
        } else {
            None
        };
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
            sha256,
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
        sha256: Some(wal_object_sha256(&bytes)),
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
    let checksum_matches = match object.sha256 {
        Some(expected) => wal_object_sha256(&bytes) == expected,
        None => wal_object_checksum(&bytes) == object.checksum,
    };
    if !checksum_matches {
        return Err(EngineError::Durability(format!(
            "WAL archive backup object {} checksum mismatch",
            object_path.display()
        )));
    }
    Ok(bytes)
}

fn write_verified_backup_bytes(path: &Path, bytes: &[u8]) -> Result<(), EngineError> {
    if let Some(parent) = path.parent() {
        create_wal_dir_all(parent).map_err(|err| {
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
    })?;
    sync_wal_parent_dir(path)
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

fn wal_object_checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn wal_object_sha256(bytes: &[u8]) -> CanonicalDigest {
    Sha256::digest(bytes).into()
}

fn format_sha256(digest: CanonicalDigest) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn parse_sha256(raw: &str, path: &Path) -> Result<CanonicalDigest, EngineError> {
    if raw.len() != 64 || !raw.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(EngineError::Durability(format!(
            "invalid WAL archive object SHA-256 in {}",
            path.display()
        )));
    }
    let mut digest = [0_u8; 32];
    for (index, slot) in digest.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&raw[index * 2..index * 2 + 2], 16).map_err(|_| {
            EngineError::Durability(format!(
                "invalid WAL archive object SHA-256 in {}",
                path.display()
            ))
        })?;
    }
    Ok(digest)
}
