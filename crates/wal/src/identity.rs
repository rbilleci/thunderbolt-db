//! Durable database/cluster/timeline lineage anchor.

use super::*;

const IDENTITY_MAGIC: &str = "GPUDBIDENTITY2";

pub fn durable_identity_path(base: &Path) -> PathBuf {
    let name = base
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.segment");
    base.with_file_name(format!("{name}.identity"))
}

pub fn write_durable_identity(base: &Path, identity: CanonicalIdentity) -> Result<(), EngineError> {
    let path = durable_identity_path(base);
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        create_wal_dir_all(parent)?;
    }
    let body = append_sha256_trailer(format!(
        "{IDENTITY_MAGIC}\ndatabase_id={}\ncluster_id={}\ntimeline_id={}\nformat_epoch={}\n",
        encode_id(identity.database_id),
        encode_id(identity.cluster_id),
        encode_id(identity.timeline_id),
        identity.format_epoch
    ));
    let temp = temporary_control_path(&path);
    let write = (|| -> Result<(), EngineError> {
        let mut file = File::create(&temp).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create durable identity {}: {err}",
                temp.display()
            ))
        })?;
        file.write_all(body.as_bytes()).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write durable identity {}: {err}",
                temp.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync durable identity {}: {err}",
                temp.display()
            ))
        })?;
        fs::rename(&temp, &path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to install durable identity {}: {err}",
                path.display()
            ))
        })?;
        sync_wal_parent_dir(&path)
    })();
    if write.is_err() {
        let _ = fs::remove_file(&temp);
    }
    write
}

pub fn read_durable_identity(base: &Path) -> Result<Option<CanonicalIdentity>, EngineError> {
    let path = durable_identity_path(base);
    let body = match fs::read_to_string(&path) {
        Ok(body) => body,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(EngineError::Durability(format!(
                "failed to read durable identity {}: {err}",
                path.display()
            )))
        }
    };
    if body.lines().next() != Some(IDENTITY_MAGIC) {
        return Err(EngineError::Durability(format!(
            "invalid durable identity header {}",
            path.display()
        )));
    }
    let verified = verify_sha256_trailer(&body, &path)?;
    let mut lines = verified.lines();
    let _magic = lines.next();
    let identity = CanonicalIdentity {
        database_id: decode_id(
            parse_control_value(lines.next(), "database_id", &path)?,
            &path,
        )?,
        cluster_id: decode_id(
            parse_control_value(lines.next(), "cluster_id", &path)?,
            &path,
        )?,
        timeline_id: decode_id(
            parse_control_value(lines.next(), "timeline_id", &path)?,
            &path,
        )?,
        format_epoch: parse_control_value(lines.next(), "format_epoch", &path)?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid durable identity format epoch {}: {err}",
                    path.display()
                ))
            })?,
    };
    if lines.next().is_some()
        || identity.database_id == [0; 16]
        || identity.cluster_id == [0; 16]
        || identity.timeline_id == [0; 16]
        || identity.format_epoch == 0
    {
        return Err(EngineError::Durability(format!(
            "invalid durable identity fields {}",
            path.display()
        )));
    }
    Ok(Some(identity))
}

/// Bind a durable artifact path to the one canonical lineage present in `records`. Writers call
/// this before making a copied/streamed WAL authority durable, so a segment can never exist as a
/// canonical recovery source without its checksummed identity anchor. Legacy-only record sets do
/// not create an anchor. An existing foreign anchor is corruption/substitution and is never
/// replaced.
pub fn bind_or_install_durable_identity(
    base: &Path,
    records: &[crate::WalRecord],
) -> Result<(), EngineError> {
    let mut identity = None;
    for record in records {
        let Some(envelope) = crate::decode_canonical_record_payload(&record.payload)? else {
            continue;
        };
        match identity {
            None => identity = Some(envelope.header.identity),
            Some(expected) if expected == envelope.header.identity => {}
            Some(_) => {
                return Err(EngineError::Durability(
                    "canonical WAL records span multiple durable identities".to_string(),
                ));
            }
        }
    }
    let Some(identity) = identity else {
        return Ok(());
    };
    match read_durable_identity(base)? {
        Some(existing) if existing == identity => Ok(()),
        Some(_) => Err(EngineError::Durability(format!(
            "durable identity anchor beside {} belongs to another database/timeline",
            base.display()
        ))),
        None => write_durable_identity(base, identity),
    }
}

fn encode_id(id: [u8; 16]) -> String {
    let mut encoded = String::with_capacity(32);
    for byte in id {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn decode_id(raw: &str, path: &Path) -> Result<[u8; 16], EngineError> {
    if raw.len() != 32 || !raw.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(EngineError::Durability(format!(
            "invalid durable identity in {}",
            path.display()
        )));
    }
    let mut id = [0_u8; 16];
    for (index, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&raw[index * 2..index * 2 + 2], 16).map_err(|_| {
            EngineError::Durability(format!("invalid durable identity in {}", path.display()))
        })?;
    }
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durable_identity_round_trips_and_tampering_fails_closed() {
        let base = std::env::temp_dir().join(format!(
            "gpu-db-identity-{}-{:?}.wal",
            std::process::id(),
            std::thread::current().id()
        ));
        let identity = CanonicalIdentity {
            database_id: [1; 16],
            cluster_id: [2; 16],
            timeline_id: [3; 16],
            format_epoch: 4,
        };
        write_durable_identity(&base, identity).unwrap();
        assert_eq!(read_durable_identity(&base).unwrap(), Some(identity));
        let path = durable_identity_path(&base);
        let body = fs::read_to_string(&path).unwrap();
        fs::write(&path, body.replace("format_epoch=4", "format_epoch=5")).unwrap();
        assert!(read_durable_identity(&base).is_err());
        let _ = fs::remove_file(path);
    }
}
