use super::*;

pub(crate) fn assert_gpu_mvcc_execution_required(engine: &Engine, error: ExecuteError) {
    assert!(
        error
            .to_string()
            .contains("generic KV MVCC queries require a device-resident result pipeline"),
        "unexpected error: {error}"
    );
    assert_eq!(engine.metrics().snapshot().fallback_total, 0);
}

pub(crate) fn assert_gpu_relational_execution_required(
    engine: &Engine,
    error: ExecuteError,
    table: &str,
    fallback_before: u64,
) {
    assert!(
        error.to_string().contains(&format!(
            "GPU execution is required for SELECT on relation \"{table}\""
        )),
        "unexpected error: {error}"
    );
    assert_eq!(
        engine.metrics().snapshot().fallback_total,
        fallback_before,
        "a loud GPU decline must not manufacture fallback telemetry"
    );
}

/// Recovery now bulk-admits after WAL replay (STRATA S-F). A successful relational read must be GPU
/// executed without fallback; the resident route may report either its device scan framing or the
/// independently expected index metadata.
pub(crate) fn assert_recovered_relational_access_path(
    result: &RelationalSelectResult,
    expected: RelationalAccessPath,
) {
    assert!(matches!(result.planned_target, DeviceTarget::Gpu(_)));
    assert!(matches!(result.executed_target, DeviceTarget::Gpu(_)));
    assert!(
        matches!(
            result.access_path.as_ref(),
            RelationalAccessPath::FullTableScan
        ) || *result.access_path == expected
    );
    assert_eq!(result.fallback_reason, None);
}

/// An exact, test-owned durable-WAL basename.
///
/// A durable WAL has a family of on-disk artifacts rather than one file: the serial tail and
/// identity sidecars, FUA frame segments, historical lane files, checkpoints, and recovery
/// status sidecars. Returning a bare `PathBuf` made every fixture responsible for knowing that
/// evolving family, which left fixed-size FUA segments behind until a full suite exhausted its
/// test volume. This test-only owner removes only files whose name begins with its globally
/// unique basename when the fixture scope ends.
pub(crate) struct TestWalPath {
    path: std::path::PathBuf,
}

impl TestWalPath {
    /// Use only for test-only APIs that deliberately discard the filesystem base.
    pub(crate) fn into_path_buf(self) -> std::path::PathBuf {
        self.path.clone()
    }
}

impl AsRef<std::path::Path> for TestWalPath {
    fn as_ref(&self) -> &std::path::Path {
        &self.path
    }
}

impl From<&TestWalPath> for std::path::PathBuf {
    fn from(path: &TestWalPath) -> Self {
        path.path.clone()
    }
}

impl std::ops::Deref for TestWalPath {
    type Target = std::path::Path;

    fn deref(&self) -> &Self::Target {
        &self.path
    }
}

impl Drop for TestWalPath {
    fn drop(&mut self) {
        cleanup_test_wal_artifacts(&self.path);
    }
}

/// Best-effort cleanup for the complete artifact family rooted at one uniquely allocated test
/// WAL basename. The `test_wal_path` nonce makes the prefix exact enough that parallel tests
/// cannot remove one another's files.
pub(crate) fn cleanup_test_wal_artifacts(path: &std::path::Path) {
    let (Some(parent), Some(stem)) = (path.parent(), path.file_name()) else {
        return;
    };
    // Checkpoint/control companions replace `.segment` rather than append to it, while FUA,
    // identity, status, and tail companions append. Both forms retain this nonce-bearing root.
    let root = stem
        .to_str()
        .and_then(|name| name.strip_suffix(".segment"))
        .unwrap_or_else(|| stem.to_str().unwrap_or("wal.segment"));
    let prefix = format!("{root}.");
    if let Ok(entries) = std::fs::read_dir(parent) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name == stem || name.to_string_lossy().starts_with(&prefix) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

pub(crate) fn test_wal_path(name: &str) -> TestWalPath {
    TestWalPath {
        path: std::env::temp_dir().join(format!(
            "gpu-db-engine-{name}-{}-{}.segment",
            std::process::id(),
            NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
        )),
    }
}

#[test]
fn test_wal_path_drop_cleans_its_complete_artifact_family_only() {
    let owned = test_wal_path("artifact-cleanup");
    let base = std::path::PathBuf::from(&owned);
    let stem = base
        .file_name()
        .expect("test WAL basename")
        .to_string_lossy();
    let fua = base.with_file_name(format!("{stem}.fua.1"));
    let checkpoint_identity = base.with_extension("checkpoint.identity");
    let neighbor = base.with_file_name(format!("unrelated-{stem}"));
    for artifact in [&base, &fua, &checkpoint_identity, &neighbor] {
        std::fs::write(artifact, b"test artifact").expect("write test artifact");
    }

    drop(owned);

    assert!(!base.exists());
    assert!(!fua.exists());
    assert!(!checkpoint_identity.exists());
    assert!(
        neighbor.exists(),
        "cleanup must not widen beyond its exact root"
    );
    let _ = std::fs::remove_file(neighbor);
}

/// Remove a mandatory DML generation when a residency-control test needs to construct a synthetic
/// cold/absent starting state. This is test-state setup only; no relational statement executes
/// against the absent generation.
pub(crate) fn forget_test_relational_residency(engine: &Engine, table: &str) {
    // R3-004: dropping a mandatory generation without first preserving its rows would destroy the
    // sole relational data copy. Residency-control tests that need an artificial cold state cross
    // the explicit RETIRE-002 repair boundary first; production has no corresponding cold fallback.
    repair_test_relational_host_copy(engine, table);
    let catalog = engine.ddl_catalog();
    catalog.relational_resident_cache.remove_table(
        table,
        &engine.read_state.residency,
        &engine.read_state.route_telemetry,
    );
}

pub(crate) fn repair_test_relational_host_copy(engine: &Engine, table: &str) {
    // A non-authoritative test generation already has a complete host tuple-store image, so
    // there is nothing to reverse-gather. This also covers the legacy single-buffer layouts used
    // by read-kernel tests: the production repair gather intentionally accepts shard generations
    // only.
    if !engine.table_device_authoritative(table) {
        return;
    }
    let catalog_table = engine
        .relational_catalog_table(table)
        .expect("test repair table must exist");
    let boundary = engine.committed_seq();
    engine
        .rehydrate_elided_table(
            &catalog_table,
            boundary,
            &Default::default(),
            &Default::default(),
            boundary,
        )
        .expect("test repair must gather the live device generation");
}

/// Rebuild a device-current table into the legacy single-buffer layout for tests whose subject is
/// that read layout. Normal DML remains shard-authoritative; this explicitly crosses the existing
/// test/repair boundary and must not be used to model a production write path.
pub(crate) fn install_test_single_buffer_residency(
    engine: &mut Engine,
    table: &str,
) -> RelationalResidencySnapshot {
    repair_test_relational_host_copy(engine, table);
    invalidate_test_relational_residency(engine, table);
    engine.set_shard_residency_enabled(false);
    engine
        .populate_relational_residency_snapshot(table)
        .expect("single-buffer test admission must succeed")
}

/// Publish an explicit invalid descriptor for route-planning tests. Production DML now maintains
/// its device generation, so invalidation scenarios must be injected rather than inferred from a
/// successful mutation.
pub(crate) fn invalidate_test_relational_residency(engine: &Engine, table: &str) {
    let current = engine.committed_seq();
    engine.invalidate_relational_residency_tables_concurrent(
        &BTreeSet::from([table.to_string()]),
        current,
        current,
    );
}
