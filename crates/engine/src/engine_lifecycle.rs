//! Engine construction, recovery & configuration (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for the constructors
//! (new_local, with_*planner/batching/durable-WAL-segment config), the
//! durable-WAL / checkpoint / archive recovery entry points (recover_from_*),
//! the commit-state / catalog / read-state / batcher accessors, the
//! leader-check internal-read shim, and the GPU-availability / memory-pressure /
//! residency-budget / runtime-saturation control flags.

use super::*;

thread_local! {
    /// GPU bytes already admitted by an explicit transaction and retained as an account credit
    /// while canonical apply replaces private allocations with globally published ones. Other
    /// threads see the charge normally; only the publication owner subtracts it from budget reads.
    static TRANSACTION_COMMIT_GPU_CREDIT:
        std::cell::RefCell<Vec<BTreeMap<u16, u64>>> = const { std::cell::RefCell::new(Vec::new()) };
}

struct TransactionCommitGpuCreditGuard;

impl Drop for TransactionCommitGpuCreditGuard {
    fn drop(&mut self) {
        TRANSACTION_COMMIT_GPU_CREDIT.with(|credits| {
            let popped = credits.borrow_mut().pop();
            debug_assert!(
                popped.is_some(),
                "transaction commit GPU credit must balance"
            );
        });
    }
}

#[cfg(test)]
thread_local! {
    static RECOVERY_CONTEXT_LOSS_INJECTIONS: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    static RECOVERY_ATTEMPT_COUNT: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

impl Engine {
    pub(crate) fn with_transaction_commit_gpu_credit<T>(
        &self,
        credit: &BTreeMap<u16, u64>,
        apply: impl FnOnce() -> T,
    ) -> T {
        TRANSACTION_COMMIT_GPU_CREDIT.with(|credits| credits.borrow_mut().push(credit.clone()));
        let _guard = TransactionCommitGpuCreditGuard;
        apply()
    }

    pub(crate) fn release_transaction_commit_gpu_credit(&self, credit: &BTreeMap<u16, u64>) {
        let _budget_guard = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut account = self
            .transaction_private_gpu_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (gpu_id, bytes) in credit {
            let slot = account.entry(*gpu_id).or_default();
            *slot = slot.saturating_sub(*bytes);
        }
        account.retain(|_, bytes| *bytes != 0);
    }

    pub(super) fn active_transaction_commit_gpu_credit(gpu_id: u16) -> u64 {
        TRANSACTION_COMMIT_GPU_CREDIT.with(|credits| {
            credits
                .borrow()
                .iter()
                .filter_map(|credit| credit.get(&gpu_id))
                .copied()
                .sum()
        })
    }

    pub fn new_local() -> Self {
        Self::with_planner_config(PlannerConfig::default())
    }

    /// Construct a recovery engine with the CUDA domain fixed before replay can allocate a
    /// resident source.  `Some` is used only for the one context-loss retry and is deliberately
    /// never installed into a normal engine's process-primary runtime cache.
    fn new_recovery_engine(
        planner_cfg: PlannerConfig,
        recovery_runtime: Option<CudaDriverRuntime>,
    ) -> Self {
        let engine = Self::with_planner_config(planner_cfg);
        if let Some(runtime) = recovery_runtime {
            engine
                .cached_cuda_probe_runtime
                .set(runtime)
                .expect("fresh recovery engine CUDA runtime cache must be empty");
        }
        engine
    }

    /// Unit-test configuration with automatic residency admission disabled.
    ///
    /// This constructor does not select an execution backend. Each test explicitly chooses a
    /// rows-only specification API or an actual device route while retaining deterministic setup.
    #[cfg(test)]
    pub(crate) fn new_local_test_engine() -> Self {
        let engine = Self::new_local();
        engine.set_auto_admit_on_commit(false);
        engine
    }

    /// Install optimized preparation lanes without constructing a durable production engine.
    #[cfg(test)]
    pub(crate) fn attach_test_intent_lanes(
        &mut self,
        _lane_base: std::path::PathBuf,
        lane_count: usize,
    ) {
        assert!(lane_count >= 2, "tests need at least two intent lanes");
        assert!(self.intent_lanes.is_none(), "test lanes already attached");
        self.intent_lanes = Some(std::sync::Arc::new(
            crate::engine_intent_lanes::IntentLaneState::fresh(lane_count),
        ));
    }

    pub(crate) fn begin_recovery_replay(&self) {
        self.set_auto_admit_on_commit(false);
    }

    pub(crate) fn finish_recovery_replay(&self) -> Result<(), EngineError> {
        #[cfg(test)]
        RECOVERY_CONTEXT_LOSS_INJECTIONS.with(|remaining| {
            if remaining.get() > 0 {
                remaining.set(remaining.get() - 1);
                return Err(EngineError::ApplyFailed(
                    "CUDA kernel launch failed: 719".to_string(),
                ));
            }
            Ok(())
        })?;
        let tables: Vec<String> = self
            .catalog_snapshot()
            .relational_catalog
            .keys()
            .cloned()
            .collect();
        for table in tables {
            let sealed_capture_target = self
                .catalog_snapshot()
                .relational_catalog
                .get(&table)
                .is_some_and(|catalog_table| {
                    self.sealed_int4_recovery_capture_is_armed_for(catalog_table.oid)
                });
            if sealed_capture_target {
                // The sealed builder is still in quiescent recovery and has only the frozen
                // canonical WAL prefix as input.  A CREATE initially admits an empty bootstrap
                // shard while automatic post-replay admission is off, so materialize exactly one
                // final device image here. Admission captures those owners before publication;
                // the later served route uses that capture, never this recovery map or a host
                // shadow.
                self.populate_relational_residency_snapshot_shared(&table)
                    .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
                // A replayed append may have replaced the CREATE-time empty shard after that
                // admission hook ran. Take the one final map value while recovery is still
                // quiescent, then retain its Arc owners; no served reader performs this lookup.
                let table_oid = self
                    .catalog_snapshot()
                    .relational_catalog
                    .get(&table)
                    .map(|catalog_table| catalog_table.oid)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(
                            "sealed nullable-int4 recovery table disappeared".to_string(),
                        )
                    })?;
                let shards = self
                    .read_residency_shards()
                    .get(&table)
                    .cloned()
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(
                            "sealed nullable-int4 recovery has no final shard generation"
                                .to_string(),
                        )
                    })?;
                self.capture_sealed_int4_recovery_shard(table_oid, shards)
                    .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
                continue;
            }
            if self.table_has_live_dml_generation(&table) {
                continue;
            }
            self.populate_relational_residency_snapshot_shared(&table)
                .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
        }
        // The sealed V1 object is intentionally constructed only after the ordinary canonical
        // replay/admission path has completed.  It consumes the shard owners captured directly
        // by admission, recomputes the GPU commitments, and asks the coordinator (the sole
        // publication authority) to install the optional read-only generation before service.
        if let Some(manifest) = self.take_staged_sealed_int4_recovery_manifest() {
            let recovered_through = self.committed_seq();
            if recovered_through == manifest.covered_through() {
                let identity = self.commit_state().canonical_identity;
                if !manifest.matches_lineage(identity) {
                    return Err(EngineError::Durability(
                        "sealed nullable-int4 checkpoint lineage does not match recovery WAL"
                            .to_string(),
                    ));
                }
                let (catalog_epoch, catalog_digest) = Self::canonical_catalog_boundary(
                    identity,
                    self.commit_state().wal.canonical_catalog_tail()?,
                )?;
                if catalog_epoch != manifest.catalog_epoch()
                    || catalog_digest != manifest.catalog_digest()
                {
                    return Err(EngineError::Durability(
                        "sealed nullable-int4 checkpoint catalog boundary does not match recovery WAL"
                            .to_string(),
                    ));
                }
                let catalog = self.catalog_snapshot();
                let table_name = catalog
                    .relational_catalog
                    .iter()
                    .find_map(|(name, table)| {
                        (table.oid == manifest.table_oid()).then(|| name.clone())
                    })
                    .ok_or_else(|| {
                        EngineError::Durability(
                            "sealed nullable-int4 checkpoint table is absent after recovery"
                                .to_string(),
                        )
                    })?;
                let shards = self
                    .take_sealed_int4_recovery_capture()
                    .map_err(|error| EngineError::Durability(error.to_string()))?;
                let generation = self.import_recovered_sealed_int4_generation(
                    manifest, catalog, table_name, shards,
                )?;
                self.install_recovered_sealed_int4_generation(generation)?;
            } else if recovered_through > manifest.covered_through() {
                // A valid checkpoint may have a live suffix. It is safe to recover and serve the
                // newer normal generation, but this sealed base no longer covers the reader cut.
                // Consume/clear the private capture rather than ever mixing it with the suffix.
                let _ = self.take_sealed_int4_recovery_capture();
            } else {
                return Err(EngineError::Durability(
                    "sealed nullable-int4 recovery did not reach the checkpoint visibility cut"
                        .to_string(),
                ));
            }
        }
        self.set_auto_admit_on_commit(true);
        Ok(())
    }

    fn is_cuda_context_loss(error: &EngineError) -> bool {
        let message = error.to_string().to_ascii_lowercase();
        if message.contains("cuda_error_invalid_context")
            || message.contains("cuda_error_context_is_destroyed")
            || message.contains("cuda context is destroyed")
            || message.contains("cuda illegal address")
            || message.contains("cuda misaligned address")
        {
            return true;
        }
        let Some(code) = message
            .split("cuda kernel launch failed:")
            .nth(1)
            .and_then(|tail| tail.split_whitespace().next())
            .and_then(|code| code.parse::<i32>().ok())
        else {
            return false;
        };
        matches!(code, 201 | 700 | 702 | 709 | 710 | 716 | 717 | 718 | 719)
    }

    fn validate_sealed_int4_checkpoint_cut(
        manifest: &gpu_db_wal::SealedInt4RebuildManifestV1,
        checkpoint: gpu_db_wal::WalCheckpointMeta,
        checkpoint_records: &[gpu_db_wal::WalRecord],
    ) -> Result<(), EngineError> {
        if checkpoint_records.len() != checkpoint.durable_record_count
            || checkpoint_records.last().map(|record| record.txn_id)
                != checkpoint.last_durable_txn_id
        {
            return Err(EngineError::Durability(
                "sealed nullable-int4 checkpoint metadata does not match its WAL prefix"
                    .to_string(),
            ));
        }
        let terminal = checkpoint_records.last().ok_or_else(|| {
            EngineError::Durability(
                "sealed nullable-int4 checkpoint has no terminal canonical WAL record".to_string(),
            )
        })?;
        let envelope =
            gpu_db_wal::decode_canonical_record_payload(&terminal.payload)?.ok_or_else(|| {
                EngineError::Durability(
                    "sealed nullable-int4 checkpoint terminal WAL record is not canonical"
                        .to_string(),
                )
            })?;
        manifest.validate_checkpoint_cut(checkpoint, envelope.header.commit_seq)
    }

    fn recover_with_fresh_context_retry<F>(mut attempt: F) -> Result<Self, EngineError>
    where
        F: FnMut(Option<CudaDriverRuntime>) -> Result<Self, EngineError>,
    {
        #[cfg(test)]
        RECOVERY_ATTEMPT_COUNT.with(|count| count.set(count.get().saturating_add(1)));
        match attempt(None) {
            Err(error) if Self::is_cuda_context_loss(&error) => {
                // Never reset, evict, or re-retain the process-wide primary context here: live
                // readers or the unknown submission may still own allocations in it.  The retry
                // instead receives a new non-registry driver context before replay's first
                // allocation.  The failed context remains parked by its unknown owner.
                let runtime = CudaDriverRuntime::probe_dedicated_recovery().map_err(|failure| {
                    EngineError::ApplyFailed(format!(
                        "sealed recovery dedicated CUDA context: {failure}"
                    ))
                })?;
                #[cfg(test)]
                RECOVERY_ATTEMPT_COUNT.with(|count| count.set(count.get().saturating_add(1)));
                attempt(Some(runtime))
            }
            result => result,
        }
    }

    #[cfg(test)]
    pub(crate) fn inject_one_recovery_context_loss() {
        Self::inject_recovery_context_losses(1);
    }

    #[cfg(test)]
    pub(crate) fn inject_recovery_context_losses(count: u8) {
        RECOVERY_CONTEXT_LOSS_INJECTIONS.with(|remaining| remaining.set(count));
        RECOVERY_ATTEMPT_COUNT.with(|count| count.set(0));
    }

    #[cfg(test)]
    pub(crate) fn recovery_attempt_count() -> u8 {
        RECOVERY_ATTEMPT_COUNT.with(std::cell::Cell::get)
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn uses_dedicated_recovery_cuda_contexts_for_test(&self) -> bool {
        self.cached_cuda_probe_runtime
            .get()
            .is_some_and(CudaDriverRuntime::uses_dedicated_recovery_contexts_for_test)
    }

    pub fn recover_from_durable_wal(records: &[WalRecord]) -> Result<Self, EngineError> {
        Self::recover_with_fresh_context_retry(|runtime| {
            Self::recover_from_durable_wal_once(records, runtime)
        })
    }

    fn recover_from_durable_wal_once(
        records: &[WalRecord],
        recovery_runtime: Option<CudaDriverRuntime>,
    ) -> Result<Self, EngineError> {
        let engine = Self::new_recovery_engine(PlannerConfig::default(), recovery_runtime);
        engine.prepare_legacy_index_oid_recovery(records)?;
        // Recovery reconstructs the durable host/store image first. Per-record admission would
        // repeatedly upload partial generations and can enter device-authoritative elision while
        // later WAL records still need the host image. Admit once, after the complete replay.
        engine.begin_recovery_replay();
        engine.replay_durable_records(records)?;
        engine.finish_recovery_replay()?;
        Ok(engine)
    }

    pub fn recover_from_durable_wal_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, EngineError> {
        let path = path.as_ref();
        // E2.5c-3 default-flip compatibility (merge audit): the DEFAULT engine
        // writes the FUA frame-log layout (`<base>.fua.*`) and, once intents
        // ran, the lane layout (`<base>.lane-N.fua.*`) — there is no plain
        // serial segment file for the read below, and pre-flip callers of
        // this API got "No such file or directory" after a crash. Dispatch on
        // the on-disk shape exactly like `open_durable_wal_segment` (which
        // also repairs crash-stranded lane orphans). Genuine serial segments
        // keep the original byte-identical replay path.
        #[cfg(unix)]
        if !path.exists()
            && (Self::intent_lane_files_exist(path) || gpu_db_wal::fua_wal_segments_exist(path))
        {
            return Self::open_durable_wal_segment(path);
        }
        let records = read_wal_segment(path)?;
        Self::recover_with_fresh_context_retry(|runtime| {
            let engine = Self::new_recovery_engine(PlannerConfig::default(), runtime);
            engine.bind_durable_identity_for_recovery(path, &records)?;
            engine.begin_recovery_replay();
            engine.replay_durable_records(&records)?;
            engine.finish_recovery_replay()?;
            Ok(engine)
        })
    }

    pub fn recover_from_durable_wal_checkpoint(
        control_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, EngineError> {
        let (_control, records) = read_wal_checkpoint(control_path)?;
        Self::recover_from_durable_wal(&records)
    }

    pub fn recover_from_durable_wal_archive(
        manifest_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, EngineError> {
        let (_manifest, records) = read_wal_archive(manifest_path)?;
        Self::recover_from_durable_wal(&records)
    }

    pub fn recover_from_durable_wal_archive_to_txn(
        manifest_path: impl AsRef<std::path::Path>,
        target_txn_id: TxnId,
    ) -> Result<Self, EngineError> {
        let (_manifest, _target, records) = read_wal_archive_to_txn(manifest_path, target_txn_id)?;
        Self::recover_from_durable_wal(&records)
    }

    pub fn recover_from_durable_wal_archive_to_timestamp_micros(
        manifest_path: impl AsRef<std::path::Path>,
        target_timestamp_micros: u64,
    ) -> Result<Self, EngineError> {
        let (_manifest, _target, records) =
            read_wal_archive_to_timestamp_micros(manifest_path, target_timestamp_micros)?;
        Self::recover_from_durable_wal(&records)
    }

    pub fn recover_from_durable_wal_checkpoint_and_archive_to_txn(
        control_path: impl AsRef<std::path::Path>,
        manifest_path: impl AsRef<std::path::Path>,
        target_txn_id: TxnId,
    ) -> Result<Self, EngineError> {
        let (control, base_records) = read_wal_checkpoint(control_path)?;
        let (_manifest, _target, archive_records) =
            read_wal_archive_to_txn(manifest_path, target_txn_id)?;
        Self::recover_from_checkpoint_and_archive_records(&control, &base_records, &archive_records)
    }

    pub fn recover_from_durable_wal_checkpoint_and_archive_to_timestamp_micros(
        control_path: impl AsRef<std::path::Path>,
        manifest_path: impl AsRef<std::path::Path>,
        target_timestamp_micros: u64,
    ) -> Result<Self, EngineError> {
        let (control, base_records) = read_wal_checkpoint(control_path)?;
        let (_manifest, _target, archive_records) =
            read_wal_archive_to_timestamp_micros(manifest_path, target_timestamp_micros)?;
        Self::recover_from_checkpoint_and_archive_records(&control, &base_records, &archive_records)
    }

    fn recover_from_checkpoint_and_archive_records(
        control: &WalControlFile,
        base_records: &[WalRecord],
        archive_records: &[WalRecord],
    ) -> Result<Self, EngineError> {
        let boundary_index =
            Self::validate_checkpoint_archive_overlap(control, base_records, archive_records)?;

        let mut recovered_records = base_records.to_vec();
        recovered_records.extend_from_slice(&archive_records[boundary_index + 1..]);
        Self::recover_from_durable_wal(&recovered_records)
    }

    pub(crate) fn validate_checkpoint_archive_overlap(
        control: &WalControlFile,
        base_records: &[WalRecord],
        archive_records: &[WalRecord],
    ) -> Result<usize, EngineError> {
        let base_last_txn_id = control.checkpoint.last_durable_txn_id.ok_or_else(|| {
            EngineError::Durability(
                "base backup checkpoint has no durable transaction boundary".to_string(),
            )
        })?;
        let boundary_index = archive_records
            .iter()
            .position(|record| record.txn_id == base_last_txn_id)
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "WAL archive does not overlap base backup transaction boundary {}",
                    base_last_txn_id
                ))
            })?;
        let archive_prefix = &archive_records[..=boundary_index];
        if archive_prefix == base_records {
            return Ok(boundary_index);
        }
        if boundary_index == 0 && archive_records.first() == base_records.last() {
            return Ok(boundary_index);
        }
        Err(EngineError::Durability(format!(
            "WAL archive prefix does not match base backup checkpoint boundary {}",
            base_last_txn_id
        )))
    }

    pub fn with_planner_config(planner_cfg: PlannerConfig) -> Self {
        Self {
            #[cfg(feature = "probe-timing")]
            insert_probe: Default::default(),
            commit: Mutex::new(CommitState {
                control_plane_reservation_owner_id: next_commit_state_control_plane_owner_id(),
                canonical_identity: Self::fresh_canonical_identity(),
                canonical_lineage_bound: false,
                canonical_replay_seen: false,
                transaction_status: HashMap::new(),
                transaction_status_reservation_generation: 0,
                last_applied_outcome: None,
                repl: LocalReplicator::leader(),
                wal: WalBuffer::default(),
                wal_commit_timestamps_micros: HashMap::new(),
                wal_commit_timestamp_reservation_generation: 0,
                max_commit_timestamp_micros: 0,
                ledger: RecentCommitsLedger::default(),
                sm: KvStateMachine::default(),
                txn_manager: TxnManager::default(),
            }),
            commit_publication: Default::default(),
            pending_transaction_claims: Arc::new(Mutex::new(HashMap::new())),
            transaction_id_allocator: Arc::new(AtomicU64::new(1)),
            sequence_value_outcomes: Mutex::new(HashMap::new()),
            commit_path_wedged: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(test)]
            fail_next_transaction_post_durable_apply: AtomicBool::new(false),
            #[cfg(test)]
            fail_next_fixed_insert_post_wal_apply: AtomicBool::new(false),
            #[cfg(test)]
            transaction_post_durable_hook: Mutex::new(None),
            active_snapshots: std::sync::Arc::new(Mutex::new(ActiveSnapshots::default())),
            table_access: Arc::new(TableAccessRegistry::default()),
            transaction_private_gpu_bytes: Arc::new(Mutex::new(BTreeMap::new())),
            transaction_retained_gpu_allocations: Arc::new(Mutex::new(BTreeMap::new())),
            group_flush: GroupFlushState::default(),
            intent_lanes: None,
            // Mirrors LocalReplicator::leader() below.
            repl_role_mirror: std::sync::atomic::AtomicU8::new(0),
            commit_wave: engine_dml_concurrent::CommitWaveState::default(),
            read_state: Arc::new(ReadState::new()),
            catalog_latch: Mutex::new(DdlCatalogState {
                relational_catalog: BTreeMap::new(),
                relational_views: BTreeMap::new(),
                relational_materialized_views: BTreeMap::new(),
                relational_functions: BTreeMap::new(),
                relational_sequences: BTreeMap::new(),
                relational_domains: BTreeMap::new(),
                relational_publications: BTreeMap::new(),
                relational_subscriptions: BTreeMap::new(),
                relational_roles: BTreeMap::new(),
                relational_databases: BTreeMap::new(),
                relational_tablespaces: BTreeMap::new(),
                relational_public_schema_exists: true,
                relational_public_schema_implicit: true,
                relational_schema_acl: BTreeMap::new(),
                relational_default_table_acl: BTreeMap::new(),
                relational_comments: BTreeMap::new(),
                relational_resident_cache: RelationalResidentCache::default(),
                relational_next_oid: FIRST_USER_RELATION_OID,
                legacy_recovery_next_index_oid: FIRST_LEGACY_RECOVERY_INDEX_OID,
                legacy_recovery_index_oids_assigned: false,
                legacy_recovery_floor_prepared: false,
                index_oid_epoch_current: true,
                relational_next_column_id: FIRST_USER_COLUMN_ID,
            }),
            metrics: RuntimeMetrics::default(),
            prepared_transaction_class_admissions: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            prepared_transaction_service:
                crate::engine_prepared_transaction::PreparedTransactionServiceController::default(),
            batcher: Mutex::new(DualTriggerBatcher::new(64, Duration::from_millis(1))),
            planner: Planner::new(planner_cfg),
            router: DeviceRouter::new(MockGpuRuntime::default()),
            cached_cuda_probe_runtime: OnceLock::new(),
            // STRATA S-F (2026-07-12): residency is the production default. Deterministic
            // shard-aware admission eviction, bounded streaming for over-budget relations,
            // incremental append/update/delete, and the production mixed read/write gate have
            // closed the former default-OFF gates. The setter remains the explicit parity-oracle
            // and operator kill switch; it is not the product direction.
            auto_admit_on_commit: std::sync::atomic::AtomicBool::new(true),
            // DEFAULT ON (user 2026-06-29: lpb chosen over the wave engine): the R1 unique-key index probe is
            // the production read path (O(1)/needle vs the O(rows) scan it replaces; byte-identical). The
            // persistent wave has been RETIRED.
            index_probe_enabled: std::sync::atomic::AtomicBool::new(true),
            // DEFAULT ON (user 2026-06-29): the dense kernel is the default for the lpb unique-index route — a
            // strict win at >=b4096, neutral at b256, byte-identical + audit SHIP. (The index route itself is
            // `index_probe_enabled`, also default ON.)
            dense_index_probe_enabled: std::sync::atomic::AtomicBool::new(true),
            // THE FLIP (2026-07-02, autonomous-completion mandate): the GPU-native sharded data plane is
            // the DEFAULT. Both correctness flip-gates are closed (SV6 created_by SI + SLICE B predicate
            // NULL 3VL); point reads are index-routed (3b/batched); scans are zero-copy at one surviving
            // shard, metadata-served for version-free COUNT(*), and recompaction-served otherwise (the
            // multi-shard aggregate kernel is the ledgered #4 endgame). Incremental DELETE/UPDATE
            // (identity locate + tombstone + stamped append, O(rows touched)) are mandatory.
            shard_residency_enabled: std::sync::atomic::AtomicBool::new(true),
            shard_index_probe_enabled: std::sync::atomic::AtomicBool::new(true),
            shard_batched_point_read_enabled: std::sync::atomic::AtomicBool::new(true),
            // A5 THE FLIP (user-authorized 2026-07-03): device-authoritative commits are the
            // DEFAULT wherever auto-admit runs; the paired auto-vacuum reclaims elided-write
            // churn. The burn-in SI bug this was once held on (stale fallback view after a
            // rehydrating decline -> a dead slot re-tombstoned, the live version leaked) is
            // FIXED by re-pinning the view at every post-rehydration fallback — pinned by the
            // SV6 concurrent hammer, which now runs elided BY DEFAULT.
            // INSERT-001: resolved binary WAL is product-default for every construction path
            // through `with_planner_config` (local, durable, and recovery/reopen). The setter is
            // retained solely as an explicit compatibility/parity kill switch.
            binary_wal_records_enabled: std::sync::atomic::AtomicBool::new(true),
            // THE CONSTRAINED-ELISION FLIP (user-authorized 2026-07-03): unique/PK'd
            auto_vacuum_enabled: std::sync::atomic::AtomicBool::new(true),
            // THE i64-SECTION FLIP (user-authorized 2026-07-03): Int8/Timestamp columns ride
            // sharded admission BY DEFAULT — the int8-payload core-banking shape runs 83-91k
            // elided vs 825 single-buffer. Stack-audited (PUSH: byte-level offset verification
            // at the 4-mod-8 case; tombstone weak-predicate soundness; i64-unique never elides).
            // Kill switch retained; default-ON makes the suite the continuing burn-in.
            shard_int8_section_enabled: std::sync::atomic::AtomicBool::new(true),
            // Default ON (measured: best-of-3 sustained 1.65M vs 1.41M unfused, p50 21.6ms
            // vs 26.3ms on the champion shape; full GPU parity incl. the reopen/checkpoint
            // arcs). ALWAYS ON (flag folded per the no-flag-proliferation
            // ruling): the unfused sequence remains ONLY as the
            // ineligible-shape fallback (i64 sections / no live index /
            // in-pass decline), an eligibility dispatch — not configuration.
            fused_apply_enabled: std::sync::atomic::AtomicBool::new(true),
            device_write_locate_wave_batch_enabled: std::sync::atomic::AtomicBool::new(false),
            tombstone_churn_threshold_override: std::sync::atomic::AtomicU64::new(0),
            // S-d2c: ~4M rows/shard (seals ~3ms, ~250 shards/1B); settable small in tests.
            shard_size_target: std::sync::atomic::AtomicUsize::new(4_000_000),
        }
    }

    /// Lock the commit_mutex, recovering from poison. Continuing past a poisoned commit lock is the
    /// engine's existing wedge-don't-recover policy's counterpart here: the façade re-homes the
    /// poison-on-panic decision to the commit path (a committer that panics mid-section poisons this
    /// lock, and the façade refuses to serve), so this `into_inner` recovery is the path the façade
    /// then trips on — it never silently serves torn state.
    pub(crate) fn commit_state(&self) -> std::sync::MutexGuard<'_, CommitState> {
        self.commit
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// `&mut`-access the commit substate WITHOUT locking — sound because `&mut self` already proves
    /// exclusive access (no other thread can hold `&self`). Used by the serialized DDL apply,
    /// recovery, and checkpoint/snapshot admin paths, which all run under the façade's exclusive
    /// (catalog-latch) write lock.
    pub(crate) fn commit_state_mut(&mut self) -> &mut CommitState {
        self.commit
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Lock the **catalog latch**, recovering from poison (same wedge-don't-recover policy as
    /// [`Engine::commit_state`]: a DDL that panics mid-apply poisons this latch, and the façade then
    /// refuses to serve rather than expose a torn working catalog). Lock order is fixed: a caller that
    /// also needs the commit_mutex must take `commit_state()` FIRST, then this. Lock-free readers and
    /// the concurrent-DML path never call this.
    pub(crate) fn ddl_catalog(&self) -> std::sync::MutexGuard<'_, DdlCatalogState> {
        self.catalog_latch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// `&mut`-access the DDL catalog state WITHOUT locking — sound because `&mut self` already proves
    /// exclusive access. Used by construction / serialized DDL-via-`&mut self` / recovery / residency
    /// admin paths (mirrors [`Engine::commit_state_mut`]).
    pub(crate) fn ddl_catalog_mut(&mut self) -> &mut DdlCatalogState {
        self.catalog_latch
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Whether the catalog latch is currently poisoned (a DDL panicked mid-apply). The façade checks
    /// this on the DDL branch so a DDL that panicked wedges service rather than serving a torn catalog.
    pub fn is_catalog_latch_poisoned(&self) -> bool {
        self.catalog_latch.is_poisoned()
    }

    /// Whether the commit_mutex is currently poisoned (a committer panicked mid-section). The façade
    /// checks this to re-home its poison-on-panic policy onto the commit path (write-half Stage 4).
    pub fn is_commit_path_poisoned(&self) -> bool {
        self.commit.is_poisoned()
            || self
                .commit_path_wedged
                .load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn commit_path_unavailable_error(&self) -> EngineError {
        if let Some(fault) = self.group_flush.fixed_poison.snapshot() {
            return EngineError::DurabilityFault(fault);
        }
        EngineError::Durability(
            "commit path is wedged; restart recovery required before reads or writes resume"
                .to_string(),
        )
    }

    pub(crate) fn ensure_commit_path_available(&self) -> Result<(), EngineError> {
        if self.is_commit_path_poisoned() {
            Err(self.commit_path_unavailable_error())
        } else {
            Ok(())
        }
    }

    pub(crate) fn wedge_commit_path(&self) {
        self.commit_path_wedged
            .store(true, std::sync::atomic::Ordering::Release);
        self.fail_all_pending_commit_work(
            "a durable commit could not be applied or published; restart recovery is required",
        );
    }

    /// A value-`Arc` clone of the lock-free read-path state. The concurrent-dispatch façade pins this
    /// once (e.g. at `SharedEngine` construction) so reads + the concurrent-DML path reach `mvcc`,
    /// `committed_seq`, resident device memory, and route telemetry WITHOUT taking the engine
    /// `RwLock` (lock-free read path, write-half MVCC). Cheap (one refcount bump); the returned
    /// handle shares the *same* interior-mutable state the engine mutates under the catalog latch.
    pub fn read_state(&self) -> Arc<ReadState> {
        Arc::clone(&self.read_state)
    }

    // `&self` read-only shims over the commit substate, for the scattered leader-checks and admin
    // queries that used to read `self.repl`/`self.wal` directly. Each takes the commit lock only
    // briefly (a cheap field read) — never across a read body.
    pub(crate) fn repl_role(&self) -> Role {
        // Lock-free (ledger #6): reading through the commit_mutex made every statement's leader
        // check convoy behind the wave sequencer's mutex hold. The mirror is updated by the rare
        // `become_*` transitions (which require `&mut Engine`, so no statement races them).
        match self
            .repl_role_mirror
            .load(std::sync::atomic::Ordering::Acquire)
        {
            0 => Role::Leader,
            1 => Role::Follower,
            _ => Role::Candidate,
        }
    }

    /// Lock the pending-mutation batcher, recovering from poison. Held only briefly (a field read or a
    /// drain), never across a `commit_mutation` (see the `batcher` field doc).
    pub(crate) fn batcher(&self) -> std::sync::MutexGuard<'_, DualTriggerBatcher<PendingMutation>> {
        self.batcher
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Whether the current thread is executing a relational read from INSIDE the commit critical
    /// section (a materialized-view create/refresh applying a committed entry), where the commit_mutex
    /// is already held and the deep read executor's leader gate would self-deadlock if it re-locked it.
    /// Thread-local + RAII-scoped ([`Engine::skip_leader_check_during_internal_read`]); false for every
    /// client read, so their leader gate is unchanged.
    pub(crate) fn mvcc_read_skips_leader_check(&self) -> bool {
        MVCC_READ_SKIPS_LEADER_CHECK.with(|flag| flag.get())
    }

    /// Run `f` (an internal mid-commit relational read) with the deep read executor's leader re-check
    /// suppressed on this thread, restoring the prior value on the way out (RAII, panic-safe).
    pub(crate) fn skip_leader_check_during_internal_read<R>(
        &self,
        f: impl FnOnce(&Self) -> R,
    ) -> R {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                MVCC_READ_SKIPS_LEADER_CHECK.with(|flag| flag.set(self.0));
            }
        }
        let _restore = MVCC_READ_SKIPS_LEADER_CHECK.with(|flag| {
            let prev = flag.get();
            flag.set(true);
            Restore(prev)
        });
        f(self)
    }

    pub fn with_batching(max_items: usize, max_wait: Duration) -> Self {
        Self::with_batching_and_planner_config(max_items, max_wait, PlannerConfig::default())
    }

    pub fn with_batching_and_planner_config(
        max_items: usize,
        max_wait: Duration,
        planner_cfg: PlannerConfig,
    ) -> Self {
        let mut s = Self::with_planner_config(planner_cfg);
        s.batcher = Mutex::new(DualTriggerBatcher::new(max_items, max_wait));
        s
    }

    /// A fresh engine whose commit path is **crash-durable**: every committed mutation's WAL record
    /// is fsynced to `segment_path` (file bytes + parent-directory entry) before the commit becomes
    /// visible (`visible_up_to` is bumped). The default [`Engine::new_local`] keeps the WAL purely
    /// in-memory; this is the constructor to use when durability is required.
    pub fn with_durable_wal_segment(segment_path: impl Into<std::path::PathBuf>) -> Self {
        Self::with_durable_wal_segment_and_planner_config(segment_path, PlannerConfig::default())
    }

    pub fn with_durable_wal_segment_and_planner_config(
        segment_path: impl Into<std::path::PathBuf>,
        planner_cfg: PlannerConfig,
    ) -> Self {
        let mut engine = Self::with_planner_config(planner_cfg);
        // E1 step 2 — ONE authority for the durability backend: default SerialFdatasync, opt into
        // the FUA fence pool via `GPU_DB_WAL_DURABILITY=fua` (+ `GPU_DB_WAL_FUA_LANES` /
        // `GPU_DB_WAL_FUA_SEGMENT_BYTES`). Exact FUA owns its physical frame/fence lifecycle;
        // the engine always admits exactly one logical durability group through its coordinator.
        let segment_path = segment_path.into();
        engine
            .install_fresh_durable_identity(&segment_path)
            .expect("failed to install durable database identity");
        let durable_identity = engine.commit_state().canonical_identity;
        #[cfg(unix)]
        let lane_base_path = segment_path.clone();
        let wal = match WalDurability::from_env() {
            #[cfg(unix)]
            WalDurability::FuaFencePool {
                lanes,
                segment_bytes,
            } => WalBuffer::with_fua_durable_segment_bound_to_identity(
                segment_path,
                lanes,
                segment_bytes,
                durable_identity,
            )
            .expect("failed to create FUA durable WAL segment (GPU_DB_WAL_DURABILITY=fua)"),
            // Non-unix has no FUA backend; from_env can still name it, so fall back to serial.
            #[allow(unreachable_patterns)]
            _ => WalBuffer::with_durable_segment_bound_to_identity(segment_path, durable_identity)
                .expect("failed to create serial durable WAL segment"),
        };
        engine.commit_state_mut().wal = wal;
        // Construct optional optimized preparation lanes. They share the canonical WalBuffer
        // above and never create a physical `.lane-*` log.
        #[cfg(unix)]
        engine
            .attach_fresh_intent_lanes(&lane_base_path, true)
            .expect("failed to attach optimized intent lanes (GPU_DB_INTENT_LANES)");
        engine
    }

    /// Construct fresh optimized preparation lanes when enabled. `lane_base_path` is used only to
    /// remove stale files from the retired format or reject a stranded historical checkpoint;
    /// the live strategy has no separate WAL backing.
    ///
    /// `fresh` = fresh-database semantics: STALE lane files from a previous database life at
    /// this path are clobbered NOW (leaving them would make the next reopen misread this
    /// database as a lanes DB holding the prior life's records). On a REOPEN (`fresh=false`,
    /// reached only when no lane segment files exist) a LANES CHECKPOINT sidecar with no lane
    /// files is corruption — it embeds committed lane records — so refuse loudly instead of
    /// silently deleting it (audit E2.5c-3 F3).
    #[cfg(unix)]
    fn attach_fresh_intent_lanes(
        &mut self,
        lane_base_path: &std::path::Path,
        fresh: bool,
    ) -> Result<(), EngineError> {
        if fresh {
            gpu_db_wal::remove_stale_lane_files(lane_base_path)?;
        } else if gpu_db_wal::lanes_checkpoint_sidecar_path(lane_base_path).exists() {
            return Err(EngineError::Durability(format!(
                "a lanes checkpoint exists beside {} but no lane segment files do; the \
                 checkpoint embeds committed lane records and the lane logs appear to have been \
                 removed — refusing to reopen over possible data loss",
                lane_base_path.display()
            )));
        }
        let lane_count = engine_intent_lanes::intent_lane_count();
        if lane_count < 2 {
            return Ok(());
        }
        self.intent_lanes = Some(std::sync::Arc::new(
            engine_intent_lanes::IntentLaneState::fresh(lane_count),
        ));
        Ok(())
    }

    /// Open (recover) a durable database from an existing WAL `segment_path` and keep writing to it.
    ///
    /// On crash recovery this is the realistic entry point: it replays every record that was fsync-
    /// durable in the segment — reconstructing exactly the committed state, since each record is
    /// re-applied through [`Engine::commit_mutation`], which re-derives the MVCC stamp from the
    /// commit `Index` (Stage 0 stamp/boundary unification) — and then continues to APPEND durably to
    /// the same segment. The append-only writer means a crash mid-append can leave a torn trailing
    /// record; [`recover_wal_segment`] truncates such a tail at the last valid record boundary
    /// (that commit was never acknowledged — WAL-before-visibility means it was never visible
    /// either), while corruption BELOW the segment's recorded durable tail offset — damage to
    /// acknowledged-durable records — still fails loudly rather than silently dropping data. If
    /// the segment does not exist yet, this behaves like [`Engine::with_durable_wal_segment`] (a
    /// fresh durable database).
    /// True if `<base>.lane-<L>.fua.*` intent-lane files exist beside the serial segment.
    #[cfg(unix)]
    fn intent_lane_files_exist(segment_path: &std::path::Path) -> bool {
        let Some(parent) = segment_path.parent() else {
            return false;
        };
        let Some(stem) = segment_path.file_name().and_then(|n| n.to_str()) else {
            return false;
        };
        let prefix = format!("{stem}.lane-");
        std::fs::read_dir(parent)
            .map(|entries| {
                entries.flatten().any(|entry| {
                    entry
                        .file_name()
                        .to_str()
                        .is_some_and(|name| name.starts_with(&prefix))
                })
            })
            .unwrap_or(false)
    }

    pub fn open_durable_wal_segment(
        segment_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, EngineError> {
        Self::open_durable_wal_segment_with_planner_config(segment_path, PlannerConfig::default())
    }

    pub fn open_durable_wal_segment_with_planner_config(
        segment_path: impl AsRef<std::path::Path>,
        planner_cfg: PlannerConfig,
    ) -> Result<Self, EngineError> {
        let segment_path = segment_path.as_ref().to_path_buf();
        Self::recover_with_fresh_context_retry(|runtime| {
            Self::open_durable_wal_segment_with_planner_config_once(
                &segment_path,
                planner_cfg,
                runtime,
            )
        })
    }

    fn open_durable_wal_segment_with_planner_config_once(
        segment_path: &std::path::Path,
        planner_cfg: PlannerConfig,
        recovery_runtime: Option<CudaDriverRuntime>,
    ) -> Result<Self, EngineError> {
        // E2.5c-1: intent-lane files beside the base identify a LANES-MODE database. Its history
        // is the serial log (pre-activation DDL/warm-up) followed by the lane merge (explicit
        // global seqs above `base_seq`); reopen replays serial-then-lanes and continues appending
        // to the SAME lane set (disk-authoritative — the on-disk lane count wins over env).
        #[cfg(unix)]
        if Self::intent_lane_files_exist(segment_path) {
            return Self::open_lanes_durable_wal_segment(
                segment_path,
                planner_cfg,
                recovery_runtime,
            );
        }
        // E1 step 3 — reopen is DISK-AUTHORITATIVE (not env-authoritative): a FUA log lives in
        // `<segment_path>.fua.*` frame-log segments (NO plain serial segment file), a serial log in
        // the plain `<segment_path>` file, so the on-disk shape identifies the backend
        // unambiguously. Recover whichever is present; refuse a genuine MIXED log (both present —
        // ambiguous ordering), which the pure-fua / pure-serial test suite never produces. This is
        // the task's "recover the non-empty one, refuse a real mix" rule, made robust to a caller
        // whose env disagrees with what a prior run wrote.
        #[cfg(unix)]
        {
            let fua_present = gpu_db_wal::fua_wal_segments_exist(segment_path);
            let serial_present = segment_path.exists();
            if fua_present && serial_present {
                return Err(EngineError::Durability(format!(
                    "both serial and FUA WAL segments exist beside {}; refusing an ambiguous \
                     mixed-backend reopen (remove one backend's segments to disambiguate)",
                    segment_path.display()
                )));
            }
            let env_is_fua = matches!(
                WalDurability::from_env(),
                WalDurability::FuaFencePool { .. }
            );
            // FUA reopen when a FUA log is on disk (honor it regardless of env — a serial-env
            // reopen must NOT shadow it), or when the env selects FUA for a fresh db. A serial log
            // on disk always recovers serially below (the on-disk format wins over the env).
            if fua_present || (!serial_present && env_is_fua) {
                let (lanes, segment_bytes) = match WalDurability::from_env() {
                    WalDurability::FuaFencePool {
                        lanes,
                        segment_bytes,
                    } => (lanes, segment_bytes),
                    _ => (
                        WalDurability::DEFAULT_FUA_LANES,
                        WalDurability::DEFAULT_FUA_SEGMENT_BYTES,
                    ),
                };
                let records = gpu_db_wal::recover_fua_wal_records(segment_path)?;
                let mut engine = Self::new_recovery_engine(planner_cfg, recovery_runtime);
                engine.bind_durable_identity_for_recovery(segment_path, &records)?;
                let durable_identity = engine.commit_state().canonical_identity;
                engine.begin_recovery_replay();
                // Replay the durable prefix WITHOUT a durable backing (no segment I/O), then install
                // a reopened FUA backend that appends above the recovered history in a fresh segment.
                engine.replay_durable_records(&records)?;
                let wal = WalBuffer::with_recovered_fua_durable_segment_bound_to_identity(
                    segment_path,
                    records,
                    lanes,
                    segment_bytes,
                    durable_identity,
                )?;
                engine.commit_state_mut().wal = wal;
                // E2.5c-1: a reopened database accepts lane intents like a fresh one (no lane
                // files existed here, so the set is created fresh; activation seeds base_seq
                // from the recovered commit index).
                engine.attach_fresh_intent_lanes(segment_path, false)?;
                engine.finish_recovery_replay()?;
                return Ok(engine);
            }
        }
        let recovery = recover_wal_segment(segment_path)?;
        let mut engine = Self::new_recovery_engine(planner_cfg, recovery_runtime);
        engine.bind_durable_identity_for_recovery(segment_path, &recovery.records)?;
        let durable_identity = engine.commit_state().canonical_identity;
        engine.begin_recovery_replay();
        // Replay the durable prefix WITHOUT a durable backing so the replay does no segment I/O;
        // then install the recovered segment so post-recovery commits keep appending to the same
        // file (the torn tail, if any, is durably truncated at install time).
        engine.replay_durable_records(&recovery.records)?;
        let records = recovery.records.clone();
        engine.commit_state_mut().wal =
            WalBuffer::with_recovered_durable_segment_bound_to_identity(
                segment_path,
                records,
                &recovery,
                durable_identity,
            )?;
        #[cfg(unix)]
        engine.attach_fresh_intent_lanes(segment_path, false)?;
        engine.finish_recovery_replay()?;
        Ok(engine)
    }

    /// Reopen the retired physical-lane format: replay its frozen serial prefix plus lane merge,
    /// repair unacknowledged orphan frames, and install a read-only compatibility backing. Empty
    /// old lane files do not trigger read-only mode. Nonempty histories must be migrated into the
    /// canonical WAL before new writes are admitted.
    ///
    /// DISK-AUTHORITATIVE: the lane count and per-lane segment capacity come from the on-disk
    /// set (env must not silently reshape an existing database); `GPU_DB_INTENT_LANE_SEGMENT_BYTES`
    /// still overrides the capacity of NEW segments when explicitly set.
    #[cfg(unix)]
    fn open_lanes_durable_wal_segment(
        segment_path: &std::path::Path,
        planner_cfg: PlannerConfig,
        recovery_runtime: Option<CudaDriverRuntime>,
    ) -> Result<Self, EngineError> {
        let lane_count = gpu_db_wal::discover_lane_count(segment_path)?.ok_or_else(|| {
            EngineError::Durability(format!(
                "lanes reopen of {} found no lane files (raced a cleanup?)",
                segment_path.display()
            ))
        })?;
        // The serial (pre-activation) prefix: disk-authoritative backend detection, exactly as
        // the non-lanes reopen below — a FUA log lives in `<base>.fua.*`, a serial log in the
        // plain `<base>` file; both present is ambiguous and refused.
        let fua_present = gpu_db_wal::fua_wal_segments_exist(segment_path);
        let serial_present = segment_path.exists();
        if fua_present && serial_present {
            return Err(EngineError::Durability(format!(
                "both serial and FUA WAL segments exist beside {}; refusing an ambiguous \
                 mixed-backend reopen (remove one backend's segments to disambiguate)",
                segment_path.display()
            )));
        }
        // E2.5c-2: a lanes CHECKPOINT shifts the lane merge to its baseline — records below it
        // live in the checkpoint segment (which embeds the serial prefix too; the atomic
        // sidecar is the single commit point) and the lane logs may be pruned below it.
        let lanes_checkpoint = gpu_db_wal::read_lanes_checkpoint(segment_path)?;
        let baseline = lanes_checkpoint
            .as_ref()
            .map(|checkpoint| checkpoint.lane_cut)
            .unwrap_or(0);
        // Lane repair BEFORE recovery: durably discard orphan frames above the cross-lane cut
        // (never acknowledged — acks gate on cut coverage — so nothing a client saw is lost;
        // see `gpu_db_wal::repair_lane_orphans`).
        let repaired = gpu_db_wal::repair_lane_orphans_from(segment_path, lane_count, baseline)?;
        if repaired > 0 {
            eprintln!(
                "[gpu-db] lanes reopen of {}: discarded {repaired} never-acknowledged orphan \
                 record(s) stranded above the durable cut by a crash mid-wave",
                segment_path.display()
            );
        }
        let lane_records = gpu_db_wal::recover_lanes_from(segment_path, lane_count, baseline)?;

        let mut engine = Self::new_recovery_engine(planner_cfg, recovery_runtime);
        engine.begin_recovery_replay();
        // Replay (checkpoint | serial)-then-lanes WITHOUT durable backing (no segment I/O),
        // then install the continuation backends. The serial prefix's record count IS
        // `base_seq`: activation seeded the oracle from `repl.peek_next_index()` while the
        // serial log was the only log, and the v1 guard refuses classic writes afterwards, so
        // the serial log froze at exactly that index.
        let serial_records: Vec<gpu_db_wal::WalRecord>;
        let serial_recovery: Option<gpu_db_wal::WalSegmentRecovery>;
        if fua_present {
            serial_records = gpu_db_wal::recover_fua_wal_records(segment_path)?;
            serial_recovery = None;
        } else if serial_present {
            let recovery = recover_wal_segment(segment_path)?;
            serial_records = recovery.records.clone();
            serial_recovery = Some(recovery);
        } else {
            serial_records = Vec::new();
            serial_recovery = None;
        }
        let mut identity_records = lanes_checkpoint
            .as_ref()
            .map(|checkpoint| checkpoint.records.clone())
            .unwrap_or_else(|| serial_records.clone());
        identity_records.extend_from_slice(&lane_records);
        engine.bind_durable_identity_for_recovery(segment_path, &identity_records)?;
        // The retained serial continuation can sit outside a lanes checkpoint's replay source.
        // Its checked recovered-WAL constructor independently verifies every supplied canonical
        // record against this anchor before it seeds the incremental identity cursor below.
        let durable_identity = engine.commit_state().canonical_identity;
        engine.install_reconciled_transaction_statuses(segment_path)?;
        let initial_index = engine.commit_state().repl.peek_next_index();
        let serial_count = if let Some(checkpoint) = &lanes_checkpoint {
            // CHECKPOINT REPLAY: the checkpoint segment holds serial ++ lanes[0, lane_cut)
            // (count-verified against its commit sidecar by read_lanes_checkpoint); the
            // on-disk serial log is its (frozen) head and is NOT replayed again — it is only
            // installed below as the WalBuffer continuation.
            if !serial_records.is_empty()
                && serial_records.len() as u64 != checkpoint.serial_records
            {
                return Err(EngineError::Durability(format!(
                    "lanes reopen of {}: the serial log holds {} record(s) but the checkpoint \
                     froze it at {}; the frozen-serial invariant is violated — refusing",
                    segment_path.display(),
                    serial_records.len(),
                    checkpoint.serial_records
                )));
            }
            engine.replay_durable_records(&checkpoint.records)?;
            checkpoint.serial_records
        } else {
            engine.replay_durable_records(&serial_records)?;
            serial_records.len() as u64
        };
        let base_seq = initial_index.checked_add(serial_count).ok_or_else(|| {
            EngineError::Durability("lane recovery base sequence overflow".to_string())
        })?;
        let replayed_prefix = engine.commit_state().repl.peek_next_index();
        let expected_replayed_prefix = base_seq.checked_add(baseline).ok_or_else(|| {
            EngineError::Durability("lane recovery checkpoint prefix overflow".to_string())
        })?;
        if replayed_prefix != expected_replayed_prefix {
            return Err(EngineError::Durability(format!(
                "lanes reopen of {}: prefix replay advanced the commit index to \
                 {replayed_prefix} (started at {initial_index}) but expected {} (serial \
                 {serial_count} + lane baseline {baseline}); the lane base seq would be wrong — \
                 refusing",
                segment_path.display(),
                expected_replayed_prefix
            )));
        }
        // P1 (sealed-shards-primary): restore the durable COLD TIER at the SEAM — the store now
        // holds exactly the checkpoint's records (the artifact's boundary state; strict equality
        // `boundary == committed_seq()` is verified inside, any mismatch a benign skip), so the
        // restored entries pin the CURRENT generation Arc and the lane suffix below IS the delta
        // stream: each replayed record patches them forward through the 6c-1 patcher via the
        // 6c-3 commit hooks. First streaming reads after reopen replay bytes instead of scanning.
        if lanes_checkpoint.is_some() {
            engine.restore_streaming_cold_checkpoint(segment_path, baseline);
        }
        engine.replay_durable_records(&lane_records)?;
        // Lane-local history length: checkpointed lane records + the recovered suffix.
        let recovered_lane_count = u64::try_from(lane_records.len()).map_err(|_| {
            EngineError::Durability("lane recovery record count exceeds u64".to_string())
        })?;
        let lane_record_count = baseline.checked_add(recovered_lane_count).ok_or_else(|| {
            EngineError::Durability("lane recovery local prefix overflow".to_string())
        })?;
        let next_seq = engine.commit_state().repl.peek_next_index();
        let expected_next_seq = base_seq.checked_add(lane_record_count).ok_or_else(|| {
            EngineError::Durability("lane recovery global prefix overflow".to_string())
        })?;
        if next_seq != expected_next_seq {
            return Err(EngineError::Durability(format!(
                "lanes reopen of {}: lane replay advanced the commit index to {next_seq}, \
                 expected {} — refusing an inconsistent seq space",
                segment_path.display(),
                expected_next_seq
            )));
        }

        // Install the serial WAL continuation (fua/serial/fresh per what is on disk). Classic
        // writes are refused after activation, so an ACTIVATED database never appends here —
        // but a lanes-mode database that never activated continues its classic warm-up exactly
        // like a non-lanes reopen.
        let wal = if fua_present {
            let (lanes, segment_bytes) = match WalDurability::from_env() {
                WalDurability::FuaFencePool {
                    lanes,
                    segment_bytes,
                } => (lanes, segment_bytes),
                _ => (
                    WalDurability::DEFAULT_FUA_LANES,
                    WalDurability::DEFAULT_FUA_SEGMENT_BYTES,
                ),
            };
            WalBuffer::with_recovered_fua_durable_segment_bound_to_identity(
                segment_path,
                serial_records,
                lanes,
                segment_bytes,
                durable_identity,
            )?
        } else if let Some(recovery) = &serial_recovery {
            WalBuffer::with_recovered_durable_segment_bound_to_identity(
                segment_path,
                serial_records,
                recovery,
                durable_identity,
            )?
        } else {
            // No serial log on disk (a lanes database whose serial WAL never flushed): create a
            // fresh durable buffer per env, exactly like the durable constructor.
            match WalDurability::from_env() {
                WalDurability::FuaFencePool {
                    lanes,
                    segment_bytes,
                } => WalBuffer::with_fua_durable_segment_bound_to_identity(
                    segment_path,
                    lanes,
                    segment_bytes,
                    durable_identity,
                )?,
                #[allow(unreachable_patterns)]
                _ => WalBuffer::with_durable_segment_bound_to_identity(
                    segment_path,
                    durable_identity,
                )?,
            }
        };
        engine.commit_state_mut().wal = wal;

        // Startup closes the retired physical reader after replay. Empty remnants do not impose
        // read-only mode; a nonempty prefix retains only a fixed diagnostic/write guard until an
        // offline migration rewrites it into canonical WAL.
        let state =
            engine_intent_lanes::IntentLaneState::with_history(lane_count, lane_record_count);
        engine.intent_lanes = Some(std::sync::Arc::new(state));
        engine.finish_recovery_replay()?;
        Ok(engine)
    }

    /// Open (recover) a durable database from a checkpoint (control file + checkpoint segment)
    /// PLUS the live segment's post-checkpoint suffix — the recovery pairing for
    /// [`Engine::checkpoint_and_truncate_durable_wal`]. Replays the checkpoint's records, then the
    /// live segment's, and keeps appending durably to the live segment.
    pub fn open_durable_wal_segment_with_checkpoint(
        control_path: impl AsRef<std::path::Path>,
        segment_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, EngineError> {
        let control_path = control_path.as_ref().to_path_buf();
        let segment_path = segment_path.as_ref().to_path_buf();
        Self::recover_with_fresh_context_retry(|runtime| {
            Self::open_durable_wal_segment_with_checkpoint_once(
                &control_path,
                &segment_path,
                runtime,
            )
        })
    }

    fn open_durable_wal_segment_with_checkpoint_once(
        control_path: &std::path::Path,
        segment_path: &std::path::Path,
        recovery_runtime: Option<CudaDriverRuntime>,
    ) -> Result<Self, EngineError> {
        // Checkpoint rotation is a SERIAL-log mechanism; a lanes-mode database's history spans
        // the serial log AND the lane logs, and cross-lane checkpoint/truncation is the E2.5c-2
        // slice. Refuse loudly rather than replay a checkpoint that silently drops lane commits.
        #[cfg(unix)]
        if Self::intent_lane_files_exist(segment_path) {
            return Err(EngineError::Durability(format!(
                "intent-lane WAL files exist beside {}: checkpoint-based reopen does not cover \
                 lane logs yet (E2.5c-2); reopen via open_durable_wal_segment instead",
                segment_path.display()
            )));
        }
        let (control, checkpoint_records) = read_wal_checkpoint(control_path)?;
        let mut recovery = recover_wal_segment(segment_path)?;
        // W1b — the checkpoint/truncation crash window: `checkpoint_and_truncate_durable_wal`
        // writes the checkpoint segment (the FULL flushed history), then the control file, then
        // truncates the live segment's prefix. A crash between the control-file write and the
        // truncation leaves overlapping records in BOTH files; blindly chaining them would
        // replay the overlap TWICE (duplicate rows). The overlap shape is: the LIVE segment's
        // HEAD equals the CHECKPOINT'S TAIL —
        //   - first-rotation crash: live = the full history = exactly the checkpoint (the tail
        //     match is the whole checkpoint);
        //   - Nth-rotation crash: live = the (N-1)th rotation's suffix, which is the tail of
        //     the new full-history checkpoint. (Head-to-head prefix matching would miss this
        //     and double-replay the suffix.)
        // Find the largest k where live[..k] == checkpoint[C-k..] (txn_id + payload) and skip
        // it. A properly-truncated segment shares no such region (its first record is
        // post-checkpoint), so k=0 on the normal path.
        let c = checkpoint_records.len();
        let max_k = c.min(recovery.records.len());
        let overlap = (0..=max_k)
            .rev()
            .find(|&k| {
                checkpoint_records[c - k..]
                    .iter()
                    .zip(recovery.records[..k].iter())
                    .all(|(checkpointed, live)| {
                        checkpointed.txn_id == live.txn_id && checkpointed.payload == live.payload
                    })
            })
            .unwrap_or(0);
        if overlap > 0 {
            recovery.records.drain(..overlap);
        }
        let mut engine = Self::new_recovery_engine(PlannerConfig::default(), recovery_runtime);
        let mut identity_records = checkpoint_records.clone();
        identity_records.extend_from_slice(&recovery.records);
        engine.bind_durable_identity_for_recovery(segment_path, &identity_records)?;
        let durable_identity = engine.commit_state().canonical_identity;
        if let Some(manifest) = control.sealed_int4_rebuild.map(Arc::new) {
            if !manifest.matches_lineage(durable_identity) {
                return Err(EngineError::Durability(
                    "sealed nullable-int4 checkpoint lineage does not match control WAL"
                        .to_string(),
                ));
            }
            Self::validate_sealed_int4_checkpoint_cut(
                &manifest,
                control.checkpoint,
                &checkpoint_records,
            )?;
            engine
                .stage_sealed_int4_recovery_manifest(manifest)
                .map_err(|error| EngineError::Durability(error.to_string()))?;
        }
        engine.begin_recovery_replay();
        let checkpoint_count = checkpoint_records.len();
        let mut records = checkpoint_records;
        records.extend_from_slice(&recovery.records);
        engine.replay_durable_records(&records)?;
        engine.commit_state_mut().wal =
            WalBuffer::with_recovered_durable_segment_bound_to_identity(
                segment_path,
                records,
                &recovery,
                durable_identity,
            )?;
        if overlap > 0 {
            // Repair: complete the crashed rotation's truncation so the live segment converges
            // to the suffix-only layout (the overlap-skip above makes the pre-repair state
            // readable; this makes it go away). Failure here is non-fatal for serving — the
            // next successful rotation or reopen repairs again.
            let _ = engine
                .commit_state_mut()
                .wal
                .truncate_durable_segment_prefix(checkpoint_count);
        }
        engine.finish_recovery_replay()?;
        Ok(engine)
    }

    /// W1b — the CHECKPOINT-AWARE standard open: recovers checkpoint-then-suffix when the
    /// convention control file (`<segment>.control`) exists, else exactly the plain open. This
    /// is the entry point the facade uses, so an auto-rotated database restarts with its FULL
    /// history — wiring auto-checkpointing through the plain open would silently drop the
    /// checkpointed prefix on restart.
    pub fn open_durable_wal_segment_auto(
        segment_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, EngineError> {
        let segment_path = segment_path.as_ref();
        // E2.5c-2: a LANES database routes through the plain open, which itself detects the
        // lanes checkpoint sidecar and replays checkpoint-then-lane-suffix (the serial
        // checkpoint-open below cannot cover lane logs).
        #[cfg(unix)]
        if Self::intent_lane_files_exist(segment_path) {
            return Self::open_durable_wal_segment(segment_path);
        }
        let control_path = gpu_db_wal::wal_checkpoint_control_path(segment_path);
        if control_path.exists() {
            Self::open_durable_wal_segment_with_checkpoint(control_path, segment_path)
        } else {
            Self::open_durable_wal_segment(segment_path)
        }
    }

    pub fn simulate_next_wal_flush_failure(&mut self) {
        self.commit_state_mut().wal.fail_next_flush();
    }

    /// Group-commit accounting for the live WAL (fsync groups, durable records, largest group).
    pub fn wal_group_commit_stats(&self) -> WalGroupCommitStats {
        self.commit_state().wal.group_commit_stats()
    }

    /// Whether the engine's commit path is crash-durable (WAL fsynced before visibility).
    pub fn wal_is_durable(&self) -> bool {
        self.commit_state().wal.is_durable()
    }

    pub fn mark_gpu_unavailable(&mut self, gpu_id: u16) {
        self.router.runtime_mut().mark_unavailable(gpu_id);
    }

    pub fn clear_gpu_unavailable(&mut self, gpu_id: u16) {
        self.router.runtime_mut().clear_unavailable(gpu_id);
    }

    pub fn mark_gpu_memory_pressured(&mut self, gpu_id: u16) {
        self.router.runtime_mut().mark_memory_pressured(gpu_id);
        self.invalidate_relational_residency_for_memory_pressure(gpu_id);
    }

    pub fn clear_gpu_memory_pressured(&mut self, gpu_id: u16) {
        self.router.runtime_mut().clear_memory_pressured(gpu_id);
    }

    pub fn set_relational_residency_budget_bytes(&mut self, gpu_id: u16, budget_bytes: u64) {
        self.ddl_catalog_mut()
            .relational_resident_cache
            .budget_bytes_by_gpu
            .insert(gpu_id, budget_bytes);
        self.mirror_admission_budgets();
    }

    pub fn clear_relational_residency_budget_bytes(&mut self, gpu_id: u16) {
        self.ddl_catalog_mut()
            .relational_resident_cache
            .budget_bytes_by_gpu
            .remove(&gpu_id);
        self.mirror_admission_budgets();
    }

    /// Republish the LOCK-FREE budget mirror from the (authoritative) catalog copy. Called by the
    /// `&mut self` budget setters — exclusive access, so the mirror can never lag a concurrent reader.
    fn mirror_admission_budgets(&mut self) {
        let budgets = self
            .ddl_catalog()
            .relational_resident_cache
            .budget_bytes_by_gpu
            .clone();
        self.read_state
            .residency
            .admission_budget_bytes_by_gpu
            .store(std::sync::Arc::new(budgets));
    }

    /// Reads the LOCK-FREE mirror, NOT the latched catalog — safe from inside the commit critical
    /// section (the resident-route planner runs there for a materialized-view create/refresh internal
    /// read; the latched read self-deadlocked — THE FLIP burn-in caught it).
    pub fn relational_residency_budget_bytes(&self, gpu_id: u16) -> Option<u64> {
        let explicit = self
            .read_state
            .residency
            .admission_budget_bytes_by_gpu
            .load()
            .get(&gpu_id)
            .copied();
        #[cfg(test)]
        return explicit;
        #[cfg(not(test))]
        explicit.or_else(|| {
            self.cuda_driver_probe_runtime()
                .snapshot()
                .devices
                .into_iter()
                .find(|device| device.id == gpu_id)
                .map(|device| device.total_memory_bytes.saturating_mul(4) / 5)
                .filter(|budget| *budget > 0)
        })
    }

    pub fn relational_resident_bytes_for_gpu(&self, gpu_id: u16) -> u64 {
        self.relational_resident_bytes_and_entries_for_gpu(gpu_id).0
    }

    /// Exact GPU-residency accounting plus the number of resident-accounting map entries it
    /// actually walked. The entry count is diagnostic only; the byte total remains the sole
    /// admission authority. Keeping both in one traversal prevents a probe from reporting a
    /// synthetic capacity-search count as if it were retained-allocation accounting work.
    pub(crate) fn relational_resident_bytes_and_entries_for_gpu(&self, gpu_id: u16) -> (u64, u64) {
        // Lifetime registry first: explicit snapshot capture uses registry -> descriptor/cache.
        // Holding it across the current-map scan makes replacement/purge and capture/accounting
        // linearizable even though publishers themselves never need this read-side registry lock.
        let retained_gpu = self
            .transaction_retained_gpu_allocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut current_allocation_identities = BTreeSet::new();
        let snapshots = self.read_state.residency.snapshots.load();
        let mut accounting_entries = (snapshots.len() as u64).saturating_mul(2);
        current_allocation_identities.extend(snapshots.values().filter_map(|entry| {
            entry
                .device_memory
                .as_ref()
                .map(|memory| (memory.metadata().gpu_id, memory.device_ptr()))
        }));
        let snapshot_bytes: u64 = snapshots
            .values()
            .filter(|entry| entry.descriptor.gpu_id == gpu_id)
            .map(|entry| {
                entry
                    .descriptor
                    .device_memory_proof
                    .as_ref()
                    .map_or(0, |proof| proof.allocated_bytes)
            })
            .sum();
        let snapshot_sidecar_bytes = [
            &self.read_state.residency.shard_deleted_by_memory,
            &self.read_state.residency.shard_created_by_memory,
            &self.read_state.residency.shard_row_id_memory,
        ]
        .into_iter()
        .map(|sidecars| {
            let (bytes, entries) = sidecars.retained_bytes_and_entries_matching(gpu_id, |table| {
                snapshots
                    .get(table)
                    .is_some_and(|entry| entry.descriptor.gpu_id == gpu_id)
            });
            accounting_entries = accounting_entries.saturating_add(entries);
            bytes
        })
        .sum::<u64>();
        let shards = self.read_state.residency.shards.load();
        let shard_entries = shards.values().map(Vec::len).sum::<usize>() as u64;
        accounting_entries = accounting_entries.saturating_add(shard_entries.saturating_mul(2));
        current_allocation_identities.extend(shards.values().flatten().flat_map(|shard| {
            [
                shard.device_memory.as_ref(),
                shard.deleted_by_region.as_ref(),
                shard.created_by_region.as_ref(),
                shard.row_id_region.as_ref(),
            ]
            .into_iter()
            .flatten()
            .map(|memory| (memory.metadata().gpu_id, memory.device_ptr()))
        }));
        let shard_bytes: u64 = shards
            .values()
            .flatten()
            .filter(|shard| shard.gpu_id == gpu_id)
            .map(|shard| {
                let regions = [
                    shard.deleted_by_region.as_ref(),
                    shard.created_by_region.as_ref(),
                    shard.row_id_region.as_ref(),
                ]
                .into_iter()
                .flatten()
                .map(|region| region.metadata().allocated_bytes)
                .sum::<u64>();
                shard.allocated_bytes.saturating_add(regions)
            })
            .sum();
        let mut device_index_allocations = BTreeMap::new();
        {
            let cache = self
                .read_state
                .residency
                .wave_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            accounting_entries = accounting_entries.saturating_add(cache.len() as u64);
            for memory in cache
                .values()
                .filter_map(|index| index.index_memory.as_ref())
            {
                device_index_allocations.insert(
                    (memory.metadata().gpu_id, memory.device_ptr()),
                    memory.metadata().allocated_bytes,
                );
                current_allocation_identities
                    .insert((memory.metadata().gpu_id, memory.device_ptr()));
            }
        }
        {
            let cache = self
                .read_state
                .residency
                .shard_pk_device_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            accounting_entries = accounting_entries.saturating_add(cache.len() as u64);
            for memory in cache
                .values()
                .filter_map(|index| index.device_index.as_ref())
            {
                device_index_allocations.insert(
                    (memory.metadata().gpu_id, memory.device_ptr()),
                    memory.metadata().allocated_bytes,
                );
                current_allocation_identities
                    .insert((memory.metadata().gpu_id, memory.device_ptr()));
            }
        }
        {
            let cache = self
                .read_state
                .residency
                .chunk_key_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            accounting_entries = accounting_entries.saturating_add(cache.len() as u64);
            for index in cache.values() {
                device_index_allocations.insert(
                    (index.device.metadata().gpu_id, index.device.device_ptr()),
                    index.device.metadata().allocated_bytes,
                );
                current_allocation_identities
                    .insert((index.device.metadata().gpu_id, index.device.device_ptr()));
            }
        }
        {
            let cache = self
                .read_state
                .residency
                .chunk_key_bloom
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            accounting_entries = accounting_entries.saturating_add(cache.len() as u64);
            for bloom in cache.values() {
                device_index_allocations.insert(
                    (bloom.device.metadata().gpu_id, bloom.device.device_ptr()),
                    bloom.device.metadata().allocated_bytes,
                );
                current_allocation_identities
                    .insert((bloom.device.metadata().gpu_id, bloom.device.device_ptr()));
            }
        }
        let device_index_bytes = device_index_allocations
            .into_iter()
            .filter(|((device, _), _)| *device == gpu_id)
            .map(|(_, bytes)| bytes)
            .sum::<u64>();
        accounting_entries = accounting_entries.saturating_add(retained_gpu.len() as u64);
        let retained_orphan_bytes = retained_gpu
            .iter()
            .filter(|((device, ptr), _)| {
                *device == gpu_id && !current_allocation_identities.contains(&(*device, *ptr))
            })
            .map(|(_, (bytes, _owners))| *bytes)
            .sum::<u64>();
        drop(retained_gpu);
        accounting_entries = accounting_entries
            .saturating_add(self.read_state.residency.sharded_point_route_count() as u64);
        let route_descriptors = self
            .read_state
            .residency
            .sharded_point_route_descriptor_bytes_for_gpu(gpu_id);
        let (live_compound_routes, compound_route_entries) =
            self.live_compound_point_route_bytes_and_entries_for_gpu(gpu_id);
        accounting_entries = accounting_entries.saturating_add(compound_route_entries);
        let private_bytes = self
            .transaction_private_gpu_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&gpu_id)
            .copied()
            .unwrap_or(0);
        let bytes = snapshot_bytes
            .saturating_add(snapshot_sidecar_bytes)
            .saturating_add(shard_bytes)
            .saturating_add(device_index_bytes)
            .saturating_add(retained_orphan_bytes)
            .saturating_add(route_descriptors)
            .saturating_add(live_compound_routes)
            .saturating_add(private_bytes)
            .saturating_sub(Self::active_transaction_commit_gpu_credit(gpu_id));
        (bytes, accounting_entries)
    }

    pub fn set_gpu_runtime_saturated(&mut self, saturated: bool) {
        self.router.runtime_mut().set_saturated(saturated);
    }
}

#[cfg(test)]
mod sealed_int4_retry_tests {
    use super::*;

    #[test]
    fn sealed_rebuild_unknown_quiescence_preserves_context_loss_for_the_bounded_retry() {
        let error = EngineError::ApplyFailed(
            "sealed nullable-int4 rebuild unknown quiescence: CUDA kernel launch failed: 719"
                .to_string(),
        );
        assert!(Engine::is_cuda_context_loss(&error));
    }
}
