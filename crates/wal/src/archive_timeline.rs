//! WAL archive timeline creation, registry, selection, and pruning.

use super::*;

const WAL_ARCHIVE_TIMELINE_MAGIC_V1: &str = "GPUDBWALTIMELINE1";
const WAL_ARCHIVE_TIMELINE_MAGIC: &str = "GPUDBWALTIMELINE2";
const WAL_ARCHIVE_TIMELINE_REGISTRY_MAGIC_V1: &str = "GPUDBWALTIMELINEREGISTRY1";
const WAL_ARCHIVE_TIMELINE_REGISTRY_MAGIC: &str = "GPUDBWALTIMELINEREGISTRY2";

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
        create_wal_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL timeline directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let body = append_sha256_trailer(format!(
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
    ));

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
    })?;
    sync_wal_parent_dir(path)
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
    let body = match body.lines().next() {
        Some(WAL_ARCHIVE_TIMELINE_MAGIC) => verify_sha256_trailer(&body, path)?,
        Some(WAL_ARCHIVE_TIMELINE_MAGIC_V1) => body,
        _ => {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive timeline header {}",
                path.display()
            )))
        }
    };
    let mut lines = body.lines();
    let _magic = lines.next();

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
        create_wal_dir_all(parent).map_err(|err| {
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
    let body = append_sha256_trailer(body);

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
    })?;
    sync_wal_parent_dir(path)
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
    let body = match body.lines().next() {
        Some(WAL_ARCHIVE_TIMELINE_REGISTRY_MAGIC) => verify_sha256_trailer(&body, path)?,
        Some(WAL_ARCHIVE_TIMELINE_REGISTRY_MAGIC_V1) => body,
        _ => {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive timeline registry header {}",
                path.display()
            )))
        }
    };
    let mut lines = body.lines();
    let _magic = lines.next();
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
    if value == "none" || value.contains('\n') || value.contains('\r') || value.contains('|') {
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
        Ok(()) => sync_wal_parent_dir(path),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(EngineError::Durability(format!(
            "failed to remove obsolete WAL archive timeline {kind} {}: {err}",
            path.display()
        ))),
    }
}
