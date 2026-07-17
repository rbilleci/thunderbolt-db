//! Durable reconciliation authority for transaction ranges discarded above a lane gap.

use std::collections::BTreeMap;

use super::*;

const RECONCILED_STATUS_MAGIC: &str = "GPUDBRECONCILEDSTATUS1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconciledTransactionStatus {
    pub identity: CanonicalIdentity,
    pub txn_id: TxnId,
    pub request_digest: CanonicalDigest,
    pub proposed_commit_seq: u64,
    pub physical: CanonicalPhysicalRange,
    pub frame_count: u32,
}

pub fn reconciled_status_path(base: &Path) -> PathBuf {
    let name = base
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.segment");
    base.with_file_name(format!("{name}.reconciled-status"))
}

pub fn read_reconciled_transaction_statuses(
    base: &Path,
) -> Result<Vec<ReconciledTransactionStatus>, EngineError> {
    let path = reconciled_status_path(base);
    let body = match fs::read_to_string(&path) {
        Ok(body) => body,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => {
            return Err(EngineError::Durability(format!(
                "failed to read reconciled transaction status {}: {err}",
                path.display()
            )))
        }
    };
    let verified = verify_sha256_trailer(&body, &path)?;
    let mut lines = verified.lines();
    if lines.next() != Some(RECONCILED_STATUS_MAGIC) {
        return Err(EngineError::Durability(format!(
            "invalid reconciled transaction status header {}",
            path.display()
        )));
    }
    let identity = CanonicalIdentity {
        database_id: decode_fixed_hex(
            parse_control_value(lines.next(), "database_id", &path)?,
            &path,
        )?,
        cluster_id: decode_fixed_hex(
            parse_control_value(lines.next(), "cluster_id", &path)?,
            &path,
        )?,
        timeline_id: decode_fixed_hex(
            parse_control_value(lines.next(), "timeline_id", &path)?,
            &path,
        )?,
        format_epoch: parse_u64(lines.next(), "format_epoch", &path)?,
    };
    if identity.database_id == [0; 16]
        || identity.cluster_id == [0; 16]
        || identity.timeline_id == [0; 16]
        || identity.format_epoch == 0
    {
        return Err(EngineError::Durability(format!(
            "invalid reconciled transaction lineage {}",
            path.display()
        )));
    }
    let entry_count = parse_u64(lines.next(), "entry_count", &path)?;
    let entry_count = usize::try_from(entry_count).map_err(|_| {
        EngineError::Durability(format!(
            "reconciled transaction status entry count overflows usize in {}",
            path.display()
        ))
    })?;
    let mut entries = Vec::with_capacity(entry_count);
    let mut prior_txn = None;
    for _ in 0..entry_count {
        let line = lines.next().ok_or_else(|| malformed(&path))?;
        let mut fields = line.split(',');
        let txn_id = parse_entry_u64(fields.next(), "txn", &path)?;
        let request_digest = decode_fixed_hex(
            parse_entry_value(fields.next(), "request_digest", &path)?,
            &path,
        )?;
        let proposed_commit_seq = parse_entry_u64(fields.next(), "proposed_commit_seq", &path)?;
        let log_epoch = parse_entry_u64(fields.next(), "log_epoch", &path)?;
        let lane_id = u32::try_from(parse_entry_u64(fields.next(), "lane_id", &path)?)
            .map_err(|_| malformed(&path))?;
        let segment_id = parse_entry_u64(fields.next(), "segment_id", &path)?;
        let first_frame_ordinal = parse_entry_u64(fields.next(), "first_frame_ordinal", &path)?;
        let frame_count = u32::try_from(parse_entry_u64(fields.next(), "frame_count", &path)?)
            .map_err(|_| malformed(&path))?;
        if fields.next() != Some("disposition=aborted-discarded-orphan")
            || fields.next() != Some("retention_deadline=18446744073709551615")
            || fields.next().is_some()
            || txn_id == 0
            || proposed_commit_seq == 0
            || proposed_commit_seq == u64::MAX
            || log_epoch == 0
            || segment_id == 0
            || frame_count < 2
            || prior_txn.is_some_and(|prior| txn_id <= prior)
        {
            return Err(malformed(&path));
        }
        prior_txn = Some(txn_id);
        entries.push(ReconciledTransactionStatus {
            identity,
            txn_id,
            request_digest,
            proposed_commit_seq,
            physical: CanonicalPhysicalRange {
                log_epoch,
                lane_id,
                segment_id,
                first_frame_ordinal,
            },
            frame_count,
        });
    }
    if lines.next().is_some() {
        return Err(malformed(&path));
    }
    Ok(entries)
}

/// Persist every complete canonical transaction proven to be above the first lane gap. This
/// function is called before destructive frame invalidation. Repeating it after a crash is
/// idempotent; conflicting stable-ID evidence fails closed.
pub(crate) fn persist_reconciled_discarded_transactions(
    base: &Path,
    records: &[WalRecord],
) -> Result<(), EngineError> {
    let mut entries: BTreeMap<TxnId, ReconciledTransactionStatus> =
        read_reconciled_transaction_statuses(base)?
            .into_iter()
            .map(|entry| (entry.txn_id, entry))
            .collect();
    let mut added = false;
    for record in records {
        let Some(envelope) = decode_canonical_record_payload(&record.payload)? else {
            continue;
        };
        let frame_count = u32::try_from(envelope.fragments.len() + 1).map_err(|_| {
            EngineError::Durability(
                "canonical discarded transaction frame count exceeds u32".to_string(),
            )
        })?;
        let entry = ReconciledTransactionStatus {
            identity: envelope.header.identity,
            txn_id: record.txn_id,
            request_digest: envelope.header.request_digest,
            proposed_commit_seq: envelope.header.commit_seq,
            physical: envelope.physical,
            frame_count,
        };
        match entries.get(&record.txn_id) {
            Some(existing) if existing == &entry => {}
            Some(_) => {
                return Err(EngineError::Durability(format!(
                    "discarded transaction {} conflicts with retained reconciliation authority",
                    record.txn_id
                )))
            }
            None => {
                entries.insert(record.txn_id, entry);
                added = true;
            }
        }
    }
    if !added {
        return Ok(());
    }
    let identity = entries
        .values()
        .next()
        .expect("an added status makes the map nonempty")
        .identity;
    if entries.values().any(|entry| entry.identity != identity) {
        return Err(EngineError::Durability(
            "reconciled transaction statuses span multiple database lineages".to_string(),
        ));
    }
    match read_durable_identity(base)? {
        Some(anchor) if anchor == identity => {}
        Some(_) => {
            return Err(EngineError::Durability(format!(
                "reconciled transaction status beside {} belongs to another database/timeline",
                base.display()
            )))
        }
        None => {
            return Err(EngineError::Durability(format!(
                "cannot reconcile discarded canonical transactions beside {} without a durable identity anchor",
                base.display()
            )))
        }
    }
    write_reconciled_transaction_statuses(base, identity, entries.values().copied())?;
    let verified = read_reconciled_transaction_statuses(base)?;
    if verified.len() != entries.len()
        || verified
            .iter()
            .zip(entries.values())
            .any(|(actual, expected)| actual != expected)
    {
        return Err(EngineError::Durability(format!(
            "reconciled transaction status read-back mismatch beside {}",
            base.display()
        )));
    }
    Ok(())
}

fn write_reconciled_transaction_statuses(
    base: &Path,
    identity: CanonicalIdentity,
    entries: impl ExactSizeIterator<Item = ReconciledTransactionStatus>,
) -> Result<(), EngineError> {
    let path = reconciled_status_path(base);
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        create_wal_dir_all(parent)?;
    }
    let mut body = format!(
        "{RECONCILED_STATUS_MAGIC}\ndatabase_id={}\ncluster_id={}\ntimeline_id={}\nformat_epoch={}\nentry_count={}\n",
        encode_hex(identity.database_id),
        encode_hex(identity.cluster_id),
        encode_hex(identity.timeline_id),
        identity.format_epoch,
        entries.len()
    );
    for entry in entries {
        use std::fmt::Write as _;
        writeln!(
            &mut body,
            "txn={},request_digest={},proposed_commit_seq={},log_epoch={},lane_id={},segment_id={},first_frame_ordinal={},frame_count={},disposition=aborted-discarded-orphan,retention_deadline={}",
            entry.txn_id,
            encode_hex(entry.request_digest),
            entry.proposed_commit_seq,
            entry.physical.log_epoch,
            entry.physical.lane_id,
            entry.physical.segment_id,
            entry.physical.first_frame_ordinal,
            entry.frame_count,
            u64::MAX
        )
        .expect("writing to String cannot fail");
    }
    let body = append_sha256_trailer(body);
    let temp = temporary_control_path(&path);
    let result = (|| -> Result<(), EngineError> {
        let mut file = File::create(&temp).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create reconciled transaction status {}: {err}",
                temp.display()
            ))
        })?;
        file.write_all(body.as_bytes()).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write reconciled transaction status {}: {err}",
                temp.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync reconciled transaction status {}: {err}",
                temp.display()
            ))
        })?;
        fs::rename(&temp, &path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to install reconciled transaction status {}: {err}",
                path.display()
            ))
        })?;
        sync_wal_parent_dir(&path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}

fn parse_u64(line: Option<&str>, key: &str, path: &Path) -> Result<u64, EngineError> {
    parse_control_value(line, key, path)?
        .parse()
        .map_err(|_| malformed(path))
}

fn parse_entry_value<'a>(
    field: Option<&'a str>,
    key: &str,
    path: &Path,
) -> Result<&'a str, EngineError> {
    field
        .and_then(|field| field.strip_prefix(key))
        .and_then(|value| value.strip_prefix('='))
        .ok_or_else(|| malformed(path))
}

fn parse_entry_u64(field: Option<&str>, key: &str, path: &Path) -> Result<u64, EngineError> {
    parse_entry_value(field, key, path)?
        .parse()
        .map_err(|_| malformed(path))
}

fn encode_hex<const N: usize>(bytes: [u8; N]) -> String {
    let mut encoded = String::with_capacity(N * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn decode_fixed_hex<const N: usize>(raw: &str, path: &Path) -> Result<[u8; N], EngineError> {
    if raw.len() != N * 2 || !raw.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(malformed(path));
    }
    let mut decoded = [0_u8; N];
    for (index, slot) in decoded.iter_mut().enumerate() {
        *slot =
            u8::from_str_radix(&raw[index * 2..index * 2 + 2], 16).map_err(|_| malformed(path))?;
    }
    Ok(decoded)
}

fn malformed(path: &Path) -> EngineError {
    EngineError::Durability(format!(
        "malformed reconciled transaction status {}",
        path.display()
    ))
}
