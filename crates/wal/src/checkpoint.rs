//! WAL checkpoint and control-file ownership.

use super::*;

const WAL_CONTROL_MAGIC_V1: &str = "GPUDBWALCONTROL1";
const WAL_CONTROL_MAGIC: &str = "GPUDBWALCONTROL2";

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

const LANES_CHECKPOINT_MAGIC_V1: &str = "gpu-db-lanes-checkpoint v1";
const LANES_CHECKPOINT_MAGIC: &str = "GPUDBLANESCHECKPOINT2";

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
    write_wal_segment(&seg_path, records)?; // atomic temp + rename + parent-directory sync
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
    let body = append_sha256_trailer(format!(
        "{LANES_CHECKPOINT_MAGIC}\nserial_records={serial_records}\nlane_cut={lane_cut}\nsegment={seg_name}\n"
    ));
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
    // Retire older generations only after the new sidecar is authoritative. A failure is a safe
    // leak, but it is reported loudly so an operator never mistakes incomplete pruning for a
    // completed checkpoint-maintenance cycle. Persist every successful unlink in the directory.
    if let (Some(parent), Some(stem)) = (
        segment_path.parent().filter(|p| !p.as_os_str().is_empty()),
        segment_path.file_name().and_then(|n| n.to_str()),
    ) {
        let prefix = format!("{stem}.lanes-checkpoint.seg.");
        let entries = fs::read_dir(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to enumerate old lanes checkpoint generations in {}: {err}",
                parent.display()
            ))
        })?;
        let mut removed = false;
        for entry in entries {
            let entry = entry.map_err(|err| {
                EngineError::Durability(format!(
                    "failed to enumerate an old lanes checkpoint generation in {}: {err}",
                    parent.display()
                ))
            })?;
            if let Some(name) = entry.file_name().to_str() {
                if let Some(gen) = name
                    .strip_prefix(&prefix)
                    .and_then(|g| g.parse::<u64>().ok())
                {
                    if gen < lane_cut {
                        fs::remove_file(entry.path()).map_err(|err| {
                            EngineError::Durability(format!(
                                "failed to remove old lanes checkpoint generation {}: {err}",
                                entry.path().display()
                            ))
                        })?;
                        removed = true;
                    }
                }
            }
        }
        if removed {
            sync_wal_parent_dir(&path)?;
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
    let (serial_records, lane_cut, seg_name) =
        if content.lines().next() == Some(LANES_CHECKPOINT_MAGIC) {
            let verified = verify_sha256_trailer(&content, &path)?;
            let mut lines = verified.lines();
            if lines.next() != Some(LANES_CHECKPOINT_MAGIC) {
                return Err(malformed());
            }
            let serial_records = parse_control_value(lines.next(), "serial_records", &path)?
                .parse::<u64>()
                .map_err(|_| malformed())?;
            let lane_cut = parse_control_value(lines.next(), "lane_cut", &path)?
                .parse::<u64>()
                .map_err(|_| malformed())?;
            let seg_name = parse_control_value(lines.next(), "segment", &path)?.to_string();
            if lines.next().is_some() {
                return Err(malformed());
            }
            (serial_records, lane_cut, seg_name)
        } else {
            let rest = content
                .strip_prefix(LANES_CHECKPOINT_MAGIC_V1)
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
            (serial_records, lane_cut, seg_name)
        };
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

pub fn write_wal_control_file(
    path: impl AsRef<Path>,
    control: &WalControlFile,
) -> Result<(), EngineError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        create_wal_dir_all(parent).map_err(|err| {
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
    let body = append_sha256_trailer(format!(
        "{WAL_CONTROL_MAGIC}\nsegment={}\ndurable_record_count={}\nlast_durable_txn_id={last_txn}\n",
        control.segment_path.display(),
        control.checkpoint.durable_record_count,
    ));

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
    })?;
    sync_wal_parent_dir(path)
}

pub fn read_wal_control_file(path: impl AsRef<Path>) -> Result<WalControlFile, EngineError> {
    let path = path.as_ref();
    let body = fs::read_to_string(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL control file {}: {err}",
            path.display()
        ))
    })?;
    let body = match body.lines().next() {
        Some(WAL_CONTROL_MAGIC) => verify_sha256_trailer(&body, path)?,
        Some(WAL_CONTROL_MAGIC_V1) => body,
        _ => {
            return Err(EngineError::Durability(format!(
                "invalid WAL control header {}",
                path.display()
            )))
        }
    };
    let mut lines = body.lines();
    let _magic = lines.next();

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
    if lines.next().is_some() {
        return Err(EngineError::Durability(format!(
            "unexpected trailing WAL control fields in {}",
            path.display()
        )));
    }

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
