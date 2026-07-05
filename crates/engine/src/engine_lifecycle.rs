//! Engine construction, recovery & configuration (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for the constructors
//! (new_local, with_*planner/batching/durable-WAL-segment config), the
//! durable-WAL / checkpoint / archive recovery entry points (recover_from_*),
//! the commit-state / catalog / read-state / batcher accessors, the
//! leader-check internal-read shim, and the GPU-availability / memory-pressure /
//! residency-budget / runtime-saturation control flags.

use super::*;

impl Engine {
    pub fn new_local() -> Self {
        Self::with_planner_config(PlannerConfig::default())
    }

    pub fn recover_from_durable_wal(records: &[WalRecord]) -> Result<Self, EngineError> {
        let engine = Self::new_local();
        for record in records {
            engine.commit_mutation(record.txn_id, record.payload.clone())?;
        }
        Ok(engine)
    }

    pub fn recover_from_durable_wal_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, EngineError> {
        let records = read_wal_segment(path)?;
        Self::recover_from_durable_wal(&records)
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
            commit: Mutex::new(CommitState {
                repl: LocalReplicator::leader(),
                wal: WalBuffer::default(),
                wal_commit_timestamps_micros: BTreeMap::new(),
                max_commit_timestamp_micros: 0,
                ledger: RecentCommitsLedger::default(),
                sm: KvStateMachine::default(),
                txn_manager: TxnManager::default(),
            }),
            active_snapshots: Mutex::new(ActiveSnapshots::default()),
            group_flush: GroupFlushState::default(),
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
                relational_next_column_id: FIRST_USER_COLUMN_ID,
            }),
            metrics: RuntimeMetrics::default(),
            batcher: Mutex::new(DualTriggerBatcher::new(64, Duration::from_millis(1))),
            planner: Planner::new(planner_cfg),
            router: DeviceRouter::new(MockGpuRuntime::default()),
            cached_cuda_probe_runtime: OnceLock::new(),
            // AUTO-ADMIT stays default OFF behind TWO NAMED GATES (user-ratified 2026-07-03,
            // with the constrained-elision flip): (1) R-1 admission budgeting / shard-aware
            // EVICTION (a default that admits every eligible table has no principled memory
            // policy); (2) the ~200ms FIRST-ROLLOVER STALL off the commit critical path
            // (ledger #19 — a p-max landmine on the first write after bulk admission). Flip
            // when both land: the charter makes GPU residency the substrate, not an opt-in.
            auto_admit_on_commit: std::sync::atomic::AtomicBool::new(false),
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
            // (tombstone + stamped append, O(rows touched)) replace the O(table) re-admit. Each flag
            // remains individually settable — the A/B levers and kill switches are unchanged.
            shard_residency_enabled: std::sync::atomic::AtomicBool::new(true),
            resident_delete_tombstone_enabled: std::sync::atomic::AtomicBool::new(true),
            resident_update_tombstone_enabled: std::sync::atomic::AtomicBool::new(true),
            shard_index_probe_enabled: std::sync::atomic::AtomicBool::new(true),
            shard_batched_point_read_enabled: std::sync::atomic::AtomicBool::new(true),
            // PHASE C slice 1 (ledger #1): DELETE/UPDATE resolve matches via the per-table equality
            // VALUE INDEX (O(matches)) instead of the O(table) prepare seq_scan. DEFAULT ON; the
            // kill switch reverts to the scan (the oracle path) — the A/B lever the differentials use.
            dml_value_index_resolve_enabled: std::sync::atomic::AtomicBool::new(true),
            // RETIREMENT A2: the device resolve is DEFAULT ON (measured: see the A2 bench line);
            // the fallback chain (value-index resolve -> scan) remains complete behind it.
            dml_device_resolve_enabled: std::sync::atomic::AtomicBool::new(true),
            dml_device_validate_enabled: std::sync::atomic::AtomicBool::new(true),
            // A5 THE FLIP (user-authorized 2026-07-03): device-authoritative commits are the
            // DEFAULT wherever auto-admit runs; the paired auto-vacuum reclaims elided-write
            // churn. The burn-in SI bug this was once held on (stale fallback view after a
            // rehydrating decline -> a dead slot re-tombstoned, the live version leaked) is
            // FIXED by re-pinning the view at every post-rehydration fallback — pinned by the
            // SV6 concurrent hammer, which now runs elided BY DEFAULT.
            host_install_elision_enabled: std::sync::atomic::AtomicBool::new(true),
            binary_wal_records_enabled: std::sync::atomic::AtomicBool::new(false),
            // THE CONSTRAINED-ELISION FLIP (user-authorized 2026-07-03): unique/PK'd
            // i32-section tables are device-authoritative BY DEFAULT — the core-banking shape
            // runs 90-94k @32w vs 16.5k host-installed. Evidence at the flip: six GPU
            // differentials + three deterministic CPU races (all sabotage-verified), three
            // adversarial audits adopted to zero open findings; default-ON makes the whole
            // suite the continuing burn-in (the A5-flip lesson). Kill switch retained.
            constrained_elision_enabled: std::sync::atomic::AtomicBool::new(true),
            auto_vacuum_enabled: std::sync::atomic::AtomicBool::new(true),
            // THE i64-SECTION FLIP (user-authorized 2026-07-03): Int8/Timestamp columns ride
            // sharded admission BY DEFAULT — the int8-payload core-banking shape runs 83-91k
            // elided vs 825 single-buffer. Stack-audited (PUSH: byte-level offset verification
            // at the 4-mod-8 case; tombstone weak-predicate soundness; i64-unique never elides).
            // Kill switch retained; default-ON makes the suite the continuing burn-in.
            shard_int8_section_enabled: std::sync::atomic::AtomicBool::new(true),
            // M1 (charter-pure device locate): default OFF (the A/B lever vs the host-probe
            // oracle); flip after the SLO gate (wave-prefetch batching) + audit.
            device_write_locate_enabled: std::sync::atomic::AtomicBool::new(false),
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
        // `GPU_DB_WAL_FUA_SEGMENT_BYTES`). The FUA backend admits MULTIPLE durable jobs in flight;
        // the concurrent-flush seam in `wait_group_durable` keys off `durability_is_concurrent()`.
        let segment_path = segment_path.into();
        let wal = match WalDurability::from_env() {
            #[cfg(unix)]
            WalDurability::FuaFencePool {
                lanes,
                segment_bytes,
            } => WalBuffer::with_fua_durable_segment(segment_path, lanes, segment_bytes)
                .expect("failed to create FUA durable WAL segment (GPU_DB_WAL_DURABILITY=fua)"),
            // Non-unix has no FUA backend; from_env can still name it, so fall back to serial.
            #[allow(unreachable_patterns)]
            _ => WalBuffer::with_durable_segment(segment_path),
        };
        engine.group_flush.concurrent_durability = wal.durability_is_concurrent();
        engine.commit_state_mut().wal = wal;
        engine
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
    pub fn open_durable_wal_segment(
        segment_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, EngineError> {
        Self::open_durable_wal_segment_with_planner_config(segment_path, PlannerConfig::default())
    }

    pub fn open_durable_wal_segment_with_planner_config(
        segment_path: impl AsRef<std::path::Path>,
        planner_cfg: PlannerConfig,
    ) -> Result<Self, EngineError> {
        let segment_path = segment_path.as_ref();
        let recovery = recover_wal_segment(segment_path)?;
        let mut engine = Self::with_planner_config(planner_cfg);
        // Replay the durable prefix WITHOUT a durable backing so the replay does no segment I/O;
        // then install the recovered segment so post-recovery commits keep appending to the same
        // file (the torn tail, if any, is durably truncated at install time).
        for record in &recovery.records {
            engine.commit_mutation(record.txn_id, record.payload.clone())?;
        }
        let records = recovery.records.clone();
        engine.commit_state_mut().wal =
            WalBuffer::with_recovered_durable_segment(segment_path, records, &recovery)?;
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
        let segment_path = segment_path.as_ref();
        let (_control, checkpoint_records) = read_wal_checkpoint(control_path)?;
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
        let mut engine = Self::new_local();
        for record in checkpoint_records.iter().chain(recovery.records.iter()) {
            engine.commit_mutation(record.txn_id, record.payload.clone())?;
        }
        let checkpoint_count = checkpoint_records.len();
        let mut records = checkpoint_records;
        records.extend_from_slice(&recovery.records);
        engine.commit_state_mut().wal =
            WalBuffer::with_recovered_durable_segment(segment_path, records, &recovery)?;
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
        self.read_state
            .residency
            .admission_budget_bytes_by_gpu
            .load()
            .get(&gpu_id)
            .copied()
    }

    pub fn relational_resident_bytes_for_gpu(&self, gpu_id: u16) -> u64 {
        let snapshot_bytes: u64 = self
            .read_state
            .residency
            .snapshots
            .load()
            .values()
            .filter(|entry| entry.descriptor.gpu_id == gpu_id)
            .map(|entry| entry.descriptor.resident_bytes)
            .sum();
        let shard_bytes: u64 = self
            .read_state
            .residency
            .shards
            .load()
            .values()
            .flatten()
            .filter(|shard| shard.gpu_id == gpu_id)
            .map(|shard| shard.resident_bytes)
            .sum();
        snapshot_bytes.saturating_add(shard_bytes)
    }

    pub fn set_gpu_runtime_saturated(&mut self, saturated: bool) {
        self.router.runtime_mut().set_saturated(saturated);
    }
}
