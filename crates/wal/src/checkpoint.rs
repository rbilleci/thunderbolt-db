//! WAL checkpoint and control-file ownership.

use super::*;

const WAL_CONTROL_MAGIC_V1: &str = "GPUDBWALCONTROL1";
const WAL_CONTROL_MAGIC: &str = "GPUDBWALCONTROL2";
const WAL_CONTROL_MAGIC_V3: &str = "GPUDBWALCONTROL3";
const SEALED_INT4_REBUILD_ROOT_FORMAT_V1: u16 = 1;
const SEALED_INT4_REBUILD_GRAMMAR_V1: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalCheckpointMeta {
    pub durable_record_count: usize,
    pub last_durable_txn_id: Option<TxnId>,
}

/// Semantic, GPU-produced commitments for the deliberately narrow WRITE-001 recovery spine.
///
/// This is checkpoint metadata, not a second WAL or an input to a host hash.  The engine writes
/// the table/database roots only after a quiesced GPU rebuild. Recovery recomputes those roots
/// independently from replayed resident data and compares them through the sealed GPU comparator.
/// It never persists the operator's intermediate proof slots. Keeping these durable expectations
/// in the checkpoint control record makes the control-file rename the one durable cut for both
/// WAL history and this sealed generation.
#[derive(Clone, PartialEq, Eq)]
pub struct SealedInt4RebuildManifestV1 {
    database_id: [u8; 16],
    cluster_id: [u8; 16],
    timeline_id: [u8; 16],
    lineage_format_epoch: u64,
    checkpoint_durable_record_count: usize,
    checkpoint_last_durable_txn_id: Option<TxnId>,
    migration_complete: bool,
    table_oid: u32,
    stable_table_id: u64,
    stable_table_high_water: u64,
    index_high_water: u64,
    current_index_count: u32,
    data_generation: u64,
    logical_row_count: u64,
    column_owner_table_oid: u32,
    legacy_column_id: u32,
    stable_column_high_water: u64,
    stable_column_id: u64,
    attnum: i16,
    catalog_epoch: u64,
    catalog_digest: [u8; 32],
    covered_through: u64,
    visible_next: u64,
    table_root: [u8; 32],
    database_root: [u8; 32],
}

impl SealedInt4RebuildManifestV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        identity: CanonicalIdentity,
        checkpoint: WalCheckpointMeta,
        table_oid: u32,
        stable_table_id: u64,
        stable_table_high_water: u64,
        index_high_water: u64,
        data_generation: u64,
        logical_row_count: u64,
        column_owner_table_oid: u32,
        legacy_column_id: u32,
        stable_column_id: u64,
        stable_column_high_water: u64,
        attnum: i16,
        catalog_epoch: u64,
        catalog_digest: [u8; 32],
        covered_through: u64,
        table_root: [u8; 32],
        database_root: [u8; 32],
    ) -> Result<Self, EngineError> {
        let visible_next = covered_through.checked_add(1).ok_or_else(|| {
            EngineError::Durability("sealed nullable-int4 visible-next overflow".to_string())
        })?;
        let manifest = Self {
            database_id: identity.database_id,
            cluster_id: identity.cluster_id,
            timeline_id: identity.timeline_id,
            lineage_format_epoch: identity.format_epoch,
            checkpoint_durable_record_count: checkpoint.durable_record_count,
            checkpoint_last_durable_txn_id: checkpoint.last_durable_txn_id,
            migration_complete: true,
            table_oid,
            stable_table_id,
            stable_table_high_water,
            index_high_water,
            current_index_count: 0,
            data_generation,
            logical_row_count,
            column_owner_table_oid,
            legacy_column_id,
            stable_column_high_water,
            stable_column_id,
            attnum,
            catalog_epoch,
            catalog_digest,
            covered_through,
            visible_next,
            table_root,
            database_root,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), EngineError> {
        if self.database_id == [0; 16]
            || self.cluster_id == [0; 16]
            || self.timeline_id == [0; 16]
            || self.lineage_format_epoch == 0
        {
            return Err(EngineError::Durability(
                "invalid sealed nullable-int4 rebuild manifest lineage".to_string(),
            ));
        }
        if !self.migration_complete
            || self.table_oid == 0
            || self.stable_table_id == 0
            || self.stable_table_high_water < self.stable_table_id
            || self.current_index_count != 0
            || self.column_owner_table_oid != self.table_oid
            || self.legacy_column_id == 0
            || self.stable_column_high_water < self.stable_column_id
            || self.stable_column_id == 0
            || self.attnum <= 0
        {
            return Err(EngineError::Durability(
                "invalid sealed nullable-int4 rebuild manifest migration".to_string(),
            ));
        }
        if self.data_generation == 0 || self.logical_row_count == 0 {
            return Err(EngineError::Durability(
                "invalid sealed nullable-int4 rebuild manifest data generation".to_string(),
            ));
        }
        if self.covered_through == 0
            || self.visible_next != self.covered_through.checked_add(1).unwrap_or(0)
        {
            return Err(EngineError::Durability(
                "invalid sealed nullable-int4 rebuild manifest checkpoint cut".to_string(),
            ));
        }
        if self.catalog_digest == [0; 32]
            || self.table_root == [0; 32]
            || self.database_root == [0; 32]
        {
            return Err(EngineError::Durability(
                "invalid sealed nullable-int4 rebuild manifest commitment".to_string(),
            ));
        }
        Ok(())
    }

    pub fn matches_checkpoint(&self, checkpoint: WalCheckpointMeta) -> bool {
        self.checkpoint_durable_record_count == checkpoint.durable_record_count
            && self.checkpoint_last_durable_txn_id == checkpoint.last_durable_txn_id
    }

    /// Bind the sealed visibility cut to the actual terminal canonical commit in the checkpoint
    /// prefix. Control metadata identifies the record count and transaction ID, while the
    /// canonical record supplies the publication sequence that must equal `covered_through`.
    pub fn validate_checkpoint_cut(
        &self,
        checkpoint: WalCheckpointMeta,
        terminal_commit_seq: u64,
    ) -> Result<(), EngineError> {
        self.validate()?;
        if !self.matches_checkpoint(checkpoint)
            || checkpoint.durable_record_count == 0
            || checkpoint.last_durable_txn_id.is_none()
            || terminal_commit_seq != self.covered_through
        {
            return Err(EngineError::Durability(
                "sealed nullable-int4 rebuild manifest checkpoint cut does not match its canonical WAL prefix"
                    .to_string(),
            ));
        }
        Ok(())
    }

    pub fn matches_lineage(&self, identity: CanonicalIdentity) -> bool {
        self.database_id == identity.database_id
            && self.cluster_id == identity.cluster_id
            && self.timeline_id == identity.timeline_id
            && self.lineage_format_epoch == identity.format_epoch
    }

    pub fn database_id(&self) -> [u8; 16] {
        self.database_id
    }
    pub fn table_oid(&self) -> u32 {
        self.table_oid
    }
    pub fn stable_table_id(&self) -> u64 {
        self.stable_table_id
    }
    pub fn stable_table_high_water(&self) -> u64 {
        self.stable_table_high_water
    }
    pub fn index_high_water(&self) -> u64 {
        self.index_high_water
    }
    pub fn data_generation(&self) -> u64 {
        self.data_generation
    }
    pub fn logical_row_count(&self) -> u64 {
        self.logical_row_count
    }
    pub fn column_owner_table_oid(&self) -> u32 {
        self.column_owner_table_oid
    }
    pub fn legacy_column_id(&self) -> u32 {
        self.legacy_column_id
    }
    pub fn stable_column_id(&self) -> u64 {
        self.stable_column_id
    }
    pub fn stable_column_high_water(&self) -> u64 {
        self.stable_column_high_water
    }
    pub fn attnum(&self) -> i16 {
        self.attnum
    }
    pub fn catalog_epoch(&self) -> u64 {
        self.catalog_epoch
    }
    pub fn catalog_digest(&self) -> [u8; 32] {
        self.catalog_digest
    }
    pub fn covered_through(&self) -> u64 {
        self.covered_through
    }
    pub fn visible_next(&self) -> u64 {
        self.visible_next
    }

    /// Root bytes cross this closure only into the engine's sealed GPU comparator; callers have
    /// no field, formatter, serializer, or generic root accessor.
    pub fn with_expected_roots<R>(&self, use_roots: impl FnOnce(&[u8; 32], &[u8; 32]) -> R) -> R {
        use_roots(&self.table_root, &self.database_root)
    }
}

impl std::fmt::Debug for SealedInt4RebuildManifestV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SealedInt4RebuildManifestV1")
            .field("table_oid", &self.table_oid)
            .field("stable_table_id", &self.stable_table_id)
            .field("data_generation", &self.data_generation)
            .field("logical_row_count", &self.logical_row_count)
            .field("covered_through", &self.covered_through)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalControlFile {
    pub segment_path: PathBuf,
    pub checkpoint: WalCheckpointMeta,
    /// Present only for the one sealed nullable-INT4 recovery generation.  Other checkpoint
    /// callers retain their historical behavior and install no WRITE-001 reader generation.
    pub sealed_int4_rebuild: Option<SealedInt4RebuildManifestV1>,
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
    if let Some(manifest) = &control.sealed_int4_rebuild {
        manifest.validate()?;
        if manifest.checkpoint_durable_record_count != control.checkpoint.durable_record_count
            || manifest.checkpoint_last_durable_txn_id != control.checkpoint.last_durable_txn_id
        {
            return Err(EngineError::Durability(
                "sealed nullable-int4 rebuild manifest is not bound to this checkpoint".to_string(),
            ));
        }
    }
    let sealed_manifest = control
        .sealed_int4_rebuild
        .as_ref()
        .map(encode_sealed_int4_rebuild_manifest)
        .transpose()?;
    let sealed_manifest_line = sealed_manifest
        .as_deref()
        .map(|encoded| format!("sealed_int4_rebuild={encoded}\n"))
        .unwrap_or_default();
    let body = append_sha256_trailer(format!(
        "{WAL_CONTROL_MAGIC_V3}\nsegment={}\ndurable_record_count={}\nlast_durable_txn_id={last_txn}\n{sealed_manifest_line}",
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
    let magic = body.lines().next().map(str::to_owned);
    let body = match magic.as_deref() {
        Some(WAL_CONTROL_MAGIC_V3) | Some(WAL_CONTROL_MAGIC) => verify_sha256_trailer(&body, path)?,
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
    let sealed_int4_rebuild = if magic.as_deref() == Some(WAL_CONTROL_MAGIC_V3) {
        match lines.next() {
            Some(line) => Some(decode_sealed_int4_rebuild_manifest(
                parse_control_value(Some(line), "sealed_int4_rebuild", path)?,
                path,
            )?),
            None => None,
        }
    } else {
        None
    };
    if let Some(manifest) = &sealed_int4_rebuild {
        if manifest.checkpoint_durable_record_count != durable_record_count
            || manifest.checkpoint_last_durable_txn_id != last_durable_txn_id
        {
            return Err(EngineError::Durability(format!(
                "sealed nullable-int4 rebuild manifest is not bound to checkpoint metadata in {}",
                path.display()
            )));
        }
    }
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
        sealed_int4_rebuild,
    })
}

fn encode_sealed_int4_rebuild_manifest(
    manifest: &SealedInt4RebuildManifestV1,
) -> Result<String, EngineError> {
    manifest.validate()?;
    let mut bytes = Vec::with_capacity(270);
    bytes.extend_from_slice(&SEALED_INT4_REBUILD_ROOT_FORMAT_V1.to_le_bytes());
    bytes.push(SEALED_INT4_REBUILD_GRAMMAR_V1);
    bytes.extend_from_slice(&manifest.database_id);
    bytes.extend_from_slice(&manifest.cluster_id);
    bytes.extend_from_slice(&manifest.timeline_id);
    bytes.extend_from_slice(&manifest.lineage_format_epoch.to_le_bytes());
    bytes.extend_from_slice(&(manifest.checkpoint_durable_record_count as u64).to_le_bytes());
    bytes.extend_from_slice(
        &manifest
            .checkpoint_last_durable_txn_id
            .unwrap_or(0)
            .to_le_bytes(),
    );
    bytes.push(u8::from(manifest.migration_complete));
    bytes.extend_from_slice(&manifest.table_oid.to_le_bytes());
    bytes.extend_from_slice(&manifest.stable_table_id.to_le_bytes());
    bytes.extend_from_slice(&manifest.stable_table_high_water.to_le_bytes());
    bytes.extend_from_slice(&manifest.index_high_water.to_le_bytes());
    bytes.extend_from_slice(&manifest.current_index_count.to_le_bytes());
    bytes.extend_from_slice(&manifest.data_generation.to_le_bytes());
    bytes.extend_from_slice(&manifest.logical_row_count.to_le_bytes());
    bytes.extend_from_slice(&manifest.column_owner_table_oid.to_le_bytes());
    bytes.extend_from_slice(&manifest.legacy_column_id.to_le_bytes());
    bytes.extend_from_slice(&manifest.stable_column_high_water.to_le_bytes());
    bytes.extend_from_slice(&manifest.stable_column_id.to_le_bytes());
    bytes.extend_from_slice(&manifest.attnum.to_le_bytes());
    bytes.extend_from_slice(&manifest.catalog_epoch.to_le_bytes());
    bytes.extend_from_slice(&manifest.catalog_digest);
    bytes.extend_from_slice(&manifest.covered_through.to_le_bytes());
    bytes.extend_from_slice(&manifest.visible_next.to_le_bytes());
    bytes.extend_from_slice(&manifest.table_root);
    bytes.extend_from_slice(&manifest.database_root);
    Ok(bytes_to_lower_hex(&bytes))
}

fn decode_sealed_int4_rebuild_manifest(
    encoded: &str,
    path: &Path,
) -> Result<SealedInt4RebuildManifestV1, EngineError> {
    const HEADER_BYTES: usize = 206;
    let bytes = lower_hex_to_bytes(encoded, path)?;
    let expected = HEADER_BYTES + 64;
    if bytes.len() != expected {
        return Err(EngineError::Durability(format!(
            "invalid sealed nullable-int4 rebuild manifest length in {}",
            path.display()
        )));
    }
    let mut at = 0;
    let take = |count: usize, at: &mut usize| {
        let start = *at;
        *at += count;
        &bytes[start..start + count]
    };
    let root_format = u16::from_le_bytes(take(2, &mut at).try_into().expect("fixed slice"));
    if root_format != SEALED_INT4_REBUILD_ROOT_FORMAT_V1 {
        return Err(EngineError::Durability(format!(
            "unsupported sealed nullable-int4 rebuild root format {root_format} in {}",
            path.display()
        )));
    }
    let grammar = take(1, &mut at)[0];
    if grammar != SEALED_INT4_REBUILD_GRAMMAR_V1 {
        return Err(EngineError::Durability(format!(
            "unsupported sealed nullable-int4 rebuild grammar {grammar} in {}",
            path.display()
        )));
    }
    let database_id = take(16, &mut at).try_into().expect("fixed slice");
    let cluster_id = take(16, &mut at).try_into().expect("fixed slice");
    let timeline_id = take(16, &mut at).try_into().expect("fixed slice");
    let lineage_format_epoch =
        u64::from_le_bytes(take(8, &mut at).try_into().expect("fixed slice"));
    let checkpoint_durable_record_count = usize::try_from(u64::from_le_bytes(
        take(8, &mut at).try_into().expect("fixed slice"),
    ))
    .map_err(|_| {
        EngineError::Durability(format!(
            "sealed nullable-int4 checkpoint count exceeds usize in {}",
            path.display()
        ))
    })?;
    let checkpoint_last_durable_txn_id =
        match u64::from_le_bytes(take(8, &mut at).try_into().expect("fixed slice")) {
            0 => None,
            txn_id => Some(txn_id),
        };
    let migration_complete = match take(1, &mut at) {
        [0] => false,
        [1] => true,
        _ => {
            return Err(EngineError::Durability(format!(
                "invalid sealed nullable-int4 migration flag in {}",
                path.display()
            )))
        }
    };
    let table_oid = u32::from_le_bytes(take(4, &mut at).try_into().expect("fixed slice"));
    let stable_table_id = u64::from_le_bytes(take(8, &mut at).try_into().expect("fixed slice"));
    let stable_table_high_water =
        u64::from_le_bytes(take(8, &mut at).try_into().expect("fixed slice"));
    let index_high_water = u64::from_le_bytes(take(8, &mut at).try_into().expect("fixed slice"));
    let current_index_count = u32::from_le_bytes(take(4, &mut at).try_into().expect("fixed slice"));
    let data_generation = u64::from_le_bytes(take(8, &mut at).try_into().expect("fixed slice"));
    let logical_row_count = u64::from_le_bytes(take(8, &mut at).try_into().expect("fixed slice"));
    let column_owner_table_oid =
        u32::from_le_bytes(take(4, &mut at).try_into().expect("fixed slice"));
    let legacy_column_id = u32::from_le_bytes(take(4, &mut at).try_into().expect("fixed slice"));
    let stable_column_high_water =
        u64::from_le_bytes(take(8, &mut at).try_into().expect("fixed slice"));
    let stable_column_id = u64::from_le_bytes(take(8, &mut at).try_into().expect("fixed slice"));
    let attnum = i16::from_le_bytes(take(2, &mut at).try_into().expect("fixed slice"));
    let catalog_epoch = u64::from_le_bytes(take(8, &mut at).try_into().expect("fixed slice"));
    let catalog_digest = take(32, &mut at).try_into().expect("fixed catalog digest");
    let covered_through = u64::from_le_bytes(take(8, &mut at).try_into().expect("fixed slice"));
    let visible_next = u64::from_le_bytes(take(8, &mut at).try_into().expect("fixed slice"));
    let table_root = take(32, &mut at).try_into().expect("fixed table root");
    let database_root = take(32, &mut at).try_into().expect("fixed database root");
    let manifest = SealedInt4RebuildManifestV1 {
        database_id,
        cluster_id,
        timeline_id,
        lineage_format_epoch,
        checkpoint_durable_record_count,
        checkpoint_last_durable_txn_id,
        migration_complete,
        table_oid,
        stable_table_id,
        stable_table_high_water,
        index_high_water,
        current_index_count,
        data_generation,
        logical_row_count,
        column_owner_table_oid,
        legacy_column_id,
        stable_column_high_water,
        stable_column_id,
        attnum,
        catalog_epoch,
        catalog_digest,
        covered_through,
        visible_next,
        table_root,
        database_root,
    };
    manifest.validate()?;
    Ok(manifest)
}

fn bytes_to_lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

fn lower_hex_to_bytes(encoded: &str, path: &Path) -> Result<Vec<u8>, EngineError> {
    if !encoded.len().is_multiple_of(2) {
        return Err(EngineError::Durability(format!(
            "invalid sealed nullable-int4 rebuild manifest encoding in {}",
            path.display()
        )));
    }
    let nibble = |value: u8| match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    };
    encoded
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| match (nibble(pair[0]), nibble(pair[1])) {
            (Some(high), Some(low)) => Ok((high << 4) | low),
            _ => Err(EngineError::Durability(format!(
                "invalid sealed nullable-int4 rebuild manifest encoding in {}",
                path.display()
            ))),
        })
        .collect()
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
