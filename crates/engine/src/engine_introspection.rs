//! Engine introspection, telemetry & maintenance accessors (P0 §9.6
//! decomposition, behavior-preserving): a focused `impl Engine` block for the
//! catalog/commit-seq accessors (visible_up_to, committed_seq, catalog_snapshot,
//! publish_*), MVCC checkpoint vacuum + active-snapshot/txn accessors, snapshot
//! meta install/export, metrics/telemetry/status snapshots, the pending-batch
//! accessors, and route_command. Small read-only / maintenance entry points.

use super::*;

thread_local! {
    /// Catalog view for one entry in a serialized multi-entry apply. The catalog latch stays held
    /// across the whole apply loop, so publishing between entries would expose a partial commit.
    /// Nested preflight/DML helpers nevertheless need to observe the entries already applied to the
    /// working catalog rather than the last globally published generation.
    static APPLY_WORKING_CATALOG:
        std::cell::RefCell<Vec<(*const Engine, Arc<CatalogSnapshot>)>> =
            const { std::cell::RefCell::new(Vec::new()) };
}

struct ApplyWorkingCatalogGuard;

impl Drop for ApplyWorkingCatalogGuard {
    fn drop(&mut self) {
        APPLY_WORKING_CATALOG.with(|catalogs| {
            let popped = catalogs.borrow_mut().pop();
            debug_assert!(popped.is_some(), "working catalog scope must be balanced");
        });
    }
}

impl Engine {
    pub fn visible_up_to(&self) -> Index {
        self.committed_seq()
    }

    /// First transaction identity not already claimed by this engine's durable status index.
    /// Wrappers that adopt a pre-seeded or recovered engine must start allocation here rather than
    /// at one, or a new request can alias an earlier canonical WAL identity.
    pub fn next_durable_transaction_id_floor(&self) -> TxnId {
        self.commit_state()
            .transaction_status
            .keys()
            .copied()
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .unwrap_or(TxnId::MAX)
    }

    /// Acquire-load the MVCC visibility/publish boundary (the highest committed `commit_seq`). The
    /// commit critical section release-stores it LAST, so an acquire-load here observes a fully
    /// published commit's data (rows + value-index generation) — the reader-side half of the
    /// publish-on-commit memory ordering (write-half Stage 4).
    pub(crate) fn committed_seq(&self) -> Index {
        self.read_state.committed_seq.load(AtomicOrdering::Acquire)
    }

    /// Pin the LATEST published catalog generation as an owned `Arc` (Stage 2 — blocker #1). This is
    /// the un-pinned "newest" view; the co-pinned read path uses [`ReadState::catalog_as_of`] with a
    /// boundary instead (PART B), but the off-latch DML preflight and a handful of admin reads that are
    /// not boundary-pinned use this. A statement that does pin loads its boundary once and threads it.
    pub(crate) fn catalog_snapshot(&self) -> Arc<CatalogSnapshot> {
        let working_catalog = APPLY_WORKING_CATALOG.with(|catalogs| {
            catalogs
                .borrow()
                .iter()
                .rev()
                .find(|(engine, _)| std::ptr::eq(*engine, self))
                .map(|(_, catalog)| Arc::clone(catalog))
        });
        if let Some(catalog) = working_catalog {
            return catalog;
        }
        self.current_transaction_read_snapshot().map_or_else(
            || self.read_state.latest_catalog(),
            |snapshot| snapshot.transaction_catalog(),
        )
    }

    /// Run one serialized apply entry with every nested catalog lookup bound to the immutable view
    /// of the transaction's current working catalog. The thread-local scope is intentional: helper
    /// APIs are deep and read-only, apply is serialized under `commit_mutex -> catalog_latch`, and
    /// other threads must continue to see only the last atomically published generation.
    pub(crate) fn with_apply_catalog<T>(
        &self,
        catalog: Option<Arc<CatalogSnapshot>>,
        apply: impl FnOnce() -> T,
    ) -> T {
        let Some(catalog) = catalog else {
            return apply();
        };
        APPLY_WORKING_CATALOG.with(|catalogs| {
            catalogs.borrow_mut().push((self, catalog));
        });
        let _guard = ApplyWorkingCatalogGuard;
        apply()
    }

    pub(crate) fn catalog_snapshot_from_working(
        cat: &DdlCatalogState,
        commit_seq: Index,
    ) -> Arc<CatalogSnapshot> {
        Arc::new(CatalogSnapshot {
            commit_seq,
            relational_catalog: cat.relational_catalog.clone(),
            relational_views: cat.relational_views.clone(),
            relational_materialized_views: cat.relational_materialized_views.clone(),
            relational_functions: cat.relational_functions.clone(),
            relational_sequences: cat.relational_sequences.clone(),
            relational_domains: cat.relational_domains.clone(),
            relational_publications: cat.relational_publications.clone(),
            relational_subscriptions: cat.relational_subscriptions.clone(),
            relational_roles: cat.relational_roles.clone(),
            relational_databases: cat.relational_databases.clone(),
            relational_tablespaces: cat.relational_tablespaces.clone(),
            relational_public_schema_exists: cat.relational_public_schema_exists,
            relational_public_schema_implicit: cat.relational_public_schema_implicit,
            relational_schema_acl: cat.relational_schema_acl.clone(),
            relational_default_table_acl: cat.relational_default_table_acl.clone(),
            relational_comments: cat.relational_comments.clone(),
        })
    }

    /// Whether applying `entry` can change catalog metadata used by a later entry in the same
    /// unpublished group. Pure DML/KV batches stay on their existing zero-catalog-clone path;
    /// sequence-default INSERTs and every catalog command bind following entries to `cat`.
    pub(crate) fn entry_mutates_working_catalog(entry: &LogEntry, cat: &DdlCatalogState) -> bool {
        if is_binary_wal_record(&entry.payload) {
            return matches!(
                decode_binary_record(&entry.payload),
                Ok(crate::wal_binary::BinaryWalRecord::Transaction(record))
                    if !record.catalog_commands.is_empty()
                        || !record.sequence_advances.is_empty()
            );
        }
        let Ok(Some(command)) = Self::decode_engine_command(&entry.payload) else {
            return true;
        };
        match command {
            Command::SetKv { .. }
            | Command::DeleteKv { .. }
            | Command::Update(_)
            | Command::Delete(_) => false,
            Command::Insert(insert) => {
                cat.relational_catalog
                    .get(&insert.table)
                    .is_some_and(|table| {
                        table.columns.iter().any(|column| {
                            matches!(
                                column.default.as_ref(),
                                Some(ColumnDefault::SequenceNextVal { .. })
                            )
                        })
                    })
            }
            _ => true,
        }
    }

    /// Build a fresh immutable catalog generation from `cat` (the DDL working maps, held under the
    /// catalog latch) stamped at `commit_seq`, push it onto the catalog ring (pruning generations the
    /// oldest active read snapshot can no longer need), and return it (Stage 2 — blocker #1; PART B
    /// co-pinning). Called by the catalog-latch apply path AFTER it has mutated the working maps and
    /// BEFORE it release-stores `committed_seq` — the publish ordering is now: data gen → residency
    /// tombstones → **catalog ring push (this)** → publication-coordinator join LAST. Because the join
    /// happens before `committed_seq` is bumped, a reader that loads `committed_seq = commit_seq` and
    /// selects `catalog_as_of(commit_seq)` is guaranteed to find this generation (catalog visible no
    /// later than `committed_seq`). DDL is the only publisher and runs under the exclusive latch.
    pub(crate) fn publish_catalog_snapshot(
        &self,
        cat: &DdlCatalogState,
        commit_seq: Index,
        prune_below: Index,
    ) {
        let generation = Self::catalog_snapshot_from_working(cat, commit_seq);
        let history = self.read_state.catalog_history.load();
        self.read_state
            .catalog_history
            .store(Arc::new(history.pushed(generation, prune_below)));
    }

    /// The oldest active read snapshot's prune boundary for the catalog ring: generations strictly
    /// older than the one a reader pinned at this boundary would select can be dropped. `None` (no
    /// in-flight reader) ⇒ prune everything redundant up to `up_to` (the just-committed seq).
    pub(crate) fn catalog_prune_boundary(&self, up_to: Index) -> Index {
        self.active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .oldest()
            .map(|oldest| oldest.saturating_sub(1))
            .unwrap_or(up_to)
    }

    pub fn applied_len(&self) -> usize {
        self.commit_state().sm.applied.len()
    }

    /// Vacuum MVCC versions whose `deleted_by` (a **`commit_seq`** = commit `Index`) is `<=`
    /// `safe_commit_seq`. Stage 0 introduced this prune by-`commit_seq` while its guards still reasoned
    /// in façade-`txn_id` space; that mismatch is data-corrupting once `txn_id ≠ Index` (it could
    /// over-prune below an active reader). Stage 4 fixes it to reason ENTIRELY in `commit_seq`/`Index`
    /// space (write-half MVCC, design Risk #6 / Stage-0 GC-boundary debt):
    ///
    /// - The active-snapshot guard is the oldest active READ SNAPSHOT (a `commit_seq`, from
    ///   [`ActiveSnapshots`]) — pruning at/after it could remove a version an in-flight reader still
    ///   needs. (The old guard used `txn_manager.oldest_active_txn_id()`, a different id space.)
    /// - The durability guard is `committed_seq` — the published boundary, which the commit path
    ///   bumps strictly AFTER the WAL fsync, so any version at/below it is already durable. (The old
    ///   guard compared against `wal.last_durable_txn_id`, again the wrong id space.)
    pub fn checkpoint_vacuum_mvcc_versions(
        &self,
        safe_commit_seq: Index,
    ) -> Result<PruneStats, EngineError> {
        if safe_commit_seq == 0 {
            return Err(EngineError::Durability(
                "checkpoint vacuum safe commit_seq must be non-zero".to_string(),
            ));
        }
        if let Some(oldest_active) = self.active_snapshots_oldest() {
            if safe_commit_seq >= oldest_active {
                return Err(EngineError::Durability(format!(
                    "checkpoint vacuum safe commit_seq {safe_commit_seq} crosses active read snapshot {oldest_active}"
                )));
            }
        }
        let durable_boundary = self.committed_seq();
        if durable_boundary == 0 {
            return Err(EngineError::Durability(
                "checkpoint vacuum requires a durable commit boundary".to_string(),
            ));
        }
        if safe_commit_seq > durable_boundary {
            return Err(EngineError::Durability(format!(
                "checkpoint vacuum safe commit_seq {safe_commit_seq} is newer than the durable commit boundary {durable_boundary}"
            )));
        }
        let stats = self
            .read_state
            .mvcc
            .prune_versions_deleted_at_or_before(safe_commit_seq);
        self.prune_relational_value_indexes_to_retained_versions()?;
        Ok(stats)
    }

    /// Rebuild each append-only relational value index from the versions that survived the same
    /// checkpoint GC fence. A row key remains in every slot needed by any retained snapshot; stale
    /// update values and fully deleted identities disappear together with their versions.
    fn prune_relational_value_indexes_to_retained_versions(&self) -> Result<usize, EngineError> {
        let catalog = self.catalog_snapshot();
        let mut removed = 0usize;
        for table in catalog.relational_catalog.values() {
            removed += self.read_state.mvcc.with_table_mut(&table.name, |data| {
                let before = data.value_index.values().map(imbl::Vector::len).sum::<usize>();
                let mut retained = BTreeMap::<ColumnValueKey, BTreeSet<String>>::new();
                for version in data.rows.all_versions() {
                    let row = decode_relational_row(&version.value, &table.columns).map_err(
                        |err| {
                            EngineError::Durability(format!(
                                "value-index GC could not decode retained row in relation \"{}\": {err}",
                                table.name
                            ))
                        },
                    )?;
                    for (column, value) in table.columns.iter().zip(row.iter()) {
                        retained
                            .entry(ColumnValueKey {
                                column: column.name.clone(),
                                value: relational_index_value(value),
                            })
                            .or_default()
                            .insert(version.key.clone());
                    }
                }
                let mut rebuilt = imbl::OrdMap::new();
                for (slot, row_keys) in retained {
                    rebuilt.insert(slot, row_keys.into_iter().collect());
                }
                let after = rebuilt.values().map(imbl::Vector::len).sum::<usize>();
                data.value_index = rebuilt;
                Ok::<usize, EngineError>(before.saturating_sub(after))
            })?;
        }
        Ok(removed)
    }

    /// The oldest active read snapshot (`commit_seq`), or `None` when no transaction is in flight —
    /// the safe MVCC GC / ledger-prune boundary (write-half Stage 4).
    pub(crate) fn active_snapshots_oldest(&self) -> Option<Index> {
        self.active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .oldest()
    }

    pub fn get(&self, key: &str) -> Option<String> {
        if self.ensure_commit_path_available().is_err() {
            return None;
        }
        // KV now lives under the commit_mutex (A.2): a `MutexGuard` cannot lend a borrow that
        // outlives it, so this returns an owned `String` (the historical `Option<&str>`). The KV read
        // path is not perf-critical; the relational read path is the lock-free one.
        self.commit_state().sm.kv.get(key).map(|s| s.to_string())
    }

    pub fn visible_state_fingerprint(&self) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x00000100000001B3;

        fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
            for b in bytes {
                hash ^= *b as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            hash
        }

        let mut hash = FNV_OFFSET_BASIS;
        let commit = self.commit_state();
        for (k, v) in &commit.sm.kv {
            hash = hash_bytes(hash, k.as_bytes());
            hash = hash_bytes(hash, &[0xFF]);
            hash = hash_bytes(hash, v.as_bytes());
            hash = hash_bytes(hash, &[0x00]);
        }
        hash
    }

    pub fn active_txn_count(&self) -> usize {
        self.commit_state().txn_manager.active_count()
    }

    pub fn replication_watermarks(&self) -> ReplicationWatermarks {
        let now = Instant::now();
        let pending_batch_len = self.batcher().len();
        let pending_batch_cap = self.batcher().max_items();
        // Read all the commit-substate watermarks under ONE commit_mutex acquisition (a snapshot of
        // the replicator + WAL counters), then release before the rest of the computation.
        let (
            wal_unflushed_count,
            role,
            commit_index,
            applied_index,
            term,
            snapshot_id,
            wal_checkpoint,
            wal_buffered_count,
            active_txn_count,
            oldest_active_txn_id,
            newest_active_txn_id,
        ) = {
            let commit = self.commit_state();
            (
                commit.wal.unflushed_count(),
                commit.repl.role(),
                commit.repl.commit_index(),
                commit.repl.applied_index(),
                commit.repl.current_term(),
                commit.repl.snapshot_meta().snapshot_id,
                commit.wal.checkpoint_meta(),
                commit.wal.len(),
                commit.txn_manager.active_count(),
                commit.txn_manager.oldest_active_txn_id(),
                commit.txn_manager.newest_active_txn_id(),
            )
        };
        let pending_batch_remaining_capacity = pending_batch_cap.saturating_sub(pending_batch_len);
        let pending_batch_utilization_permyriad = if pending_batch_cap == 0 {
            0
        } else {
            let utilization =
                (pending_batch_len as u128).saturating_mul(10_000) / (pending_batch_cap as u128);
            utilization.min(10_000) as u16
        };
        let pending_batch_remaining_capacity_permyriad =
            10_000u16.saturating_sub(pending_batch_utilization_permyriad);

        let visible_index = self.committed_seq();

        let commit_apply_gap = commit_index.saturating_sub(applied_index);
        let apply_visible_gap = applied_index.saturating_sub(visible_index);

        let has_wal_backlog = wal_unflushed_count > 0;
        let has_pending_batch_backlog = pending_batch_len > 0;
        let has_active_txn_backlog = active_txn_count > 0;
        let has_commit_apply_gap = commit_apply_gap > 0;
        let has_apply_visible_gap = apply_visible_gap > 0;
        let backlog_blocker_mask = (u8::from(has_wal_backlog)
            * ReplicationWatermarks::BACKLOG_BLOCKER_WAL)
            | (u8::from(has_pending_batch_backlog)
                * ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH)
            | (u8::from(has_active_txn_backlog)
                * ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN)
            | (u8::from(has_commit_apply_gap)
                * ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP)
            | (u8::from(has_apply_visible_gap)
                * ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP);
        let backlog_blocker_count =
            ReplicationWatermarks::backlog_blocker_count_from_mask(backlog_blocker_mask);
        let has_backlog_blockers =
            ReplicationWatermarks::has_backlog_blockers_in_mask(backlog_blocker_mask);

        ReplicationWatermarks {
            role,
            term,
            commit_index,
            applied_index,
            visible_index,
            commit_apply_gap,
            apply_visible_gap,
            snapshot_id,
            wal_flushed_count: wal_checkpoint.durable_record_count,
            wal_last_durable_txn_id: wal_checkpoint.last_durable_txn_id,
            wal_buffered_count,
            wal_unflushed_count,
            pending_batch_len,
            pending_batch_cap,
            pending_batch_remaining_capacity,
            pending_batch_utilization_permyriad,
            pending_batch_remaining_capacity_permyriad,
            pending_batch_oldest_age_ms: self
                .pending_batch_oldest_age(now)
                .map(|age| age.as_millis() as u64),
            pending_batch_time_until_deadline_ms: self
                .pending_batch_time_until_deadline(now)
                .map(|remaining| remaining.as_millis() as u64),
            active_txn_count,
            oldest_active_txn_id,
            newest_active_txn_id,
            has_wal_backlog,
            has_pending_batch_backlog,
            has_active_txn_backlog,
            has_commit_apply_gap,
            has_apply_visible_gap,
            has_backlog_blockers,
            backlog_blocker_count,
            backlog_blocker_mask,
            mutation_admission_saturated: pending_batch_len >= pending_batch_cap,
            quiescent_for_failover: role == Role::Leader
                && !has_wal_backlog
                && !has_pending_batch_backlog
                && !has_active_txn_backlog,
            follower_promotion_ready: role == Role::Follower
                && !has_commit_apply_gap
                && !has_apply_visible_gap
                && !has_wal_backlog
                && !has_pending_batch_backlog
                && !has_active_txn_backlog,
        }
    }

    pub fn export_snapshot_meta(&mut self) -> SnapshotMeta {
        self.commit_state_mut().repl.export_snapshot_meta()
    }

    pub fn install_snapshot(&mut self, meta: SnapshotMeta) {
        let last_included_index = {
            let commit = self.commit_state_mut();
            commit.repl.install_snapshot(meta);
            commit.repl.snapshot_meta().last_included_index
        };
        self.install_publication_snapshot(last_included_index);
    }

    pub fn snapshot_meta(&self) -> SnapshotMeta {
        self.commit_state().repl.snapshot_meta()
    }

    pub fn metrics(&self) -> &RuntimeMetrics {
        &self.metrics
    }

    pub fn telemetry_snapshot(&self) -> EngineTelemetrySnapshot {
        let marks = self.replication_watermarks();
        let relational_residency = self.relational_residency_status();
        EngineTelemetrySnapshot {
            role: marks.role,
            replication_lag: ReplicationLagSnapshot {
                commit_index: marks.commit_index,
                applied_index: marks.applied_index,
                visible_index: marks.visible_index,
                commit_apply_gap: marks.commit_apply_gap,
                apply_visible_gap: marks.apply_visible_gap,
            },
            runtime_metrics: self.metrics.snapshot(),
            snapshot_id: marks.snapshot_id,
            wal_flushed_count: marks.wal_flushed_count,
            wal_last_durable_txn_id: marks.wal_last_durable_txn_id,
            wal_buffered_count: marks.wal_buffered_count,
            wal_unflushed_count: marks.wal_unflushed_count,
            pending_batch_len: marks.pending_batch_len,
            pending_batch_cap: marks.pending_batch_cap,
            active_txn_count: marks.active_txn_count,
            backlog_blocker_count: marks.backlog_blocker_count,
            backlog_blocker_mask: marks.backlog_blocker_mask,
            mutation_admission_saturated: marks.mutation_admission_saturated,
            quiescent_for_failover: marks.quiescent_for_failover,
            follower_promotion_ready: marks.follower_promotion_ready,
            gpu_parity_fallbacks: self.metrics.fallback_counts_by_gpu_parity_issue(),
            gpu_runtime: self.router.runtime().snapshot(),
            relational_residency,
        }
    }

    pub fn status_snapshot(&self) -> EngineStatusSnapshot {
        let marks = self.replication_watermarks();
        let snapshot_meta = self.snapshot_meta();
        let runtime_metrics = self.metrics.snapshot();
        let gpu_runtime = self.router.runtime().snapshot();

        let mut active_reasons = Vec::new();
        if !gpu_runtime.unavailable_gpu_ids.is_empty() {
            active_reasons.push(ActiveFallbackReason::GpuUnavailable {
                gpu_ids: gpu_runtime.unavailable_gpu_ids.clone(),
            });
        }
        if !gpu_runtime.memory_pressured_gpu_ids.is_empty() {
            active_reasons.push(ActiveFallbackReason::GpuMemoryPressure {
                gpu_ids: gpu_runtime.memory_pressured_gpu_ids.clone(),
            });
        }
        if gpu_runtime.saturated {
            active_reasons.push(ActiveFallbackReason::GpuQueueSaturated);
        }

        let mut status = EngineStatusSnapshot::new(
            marks.role,
            marks.term,
            SnapshotStatus {
                snapshot_id: snapshot_meta.snapshot_id,
                last_included_index: snapshot_meta.last_included_index,
                last_included_term: snapshot_meta.last_included_term,
                visible_index: marks.visible_index,
            },
            ReplicationLagSnapshot {
                commit_index: marks.commit_index,
                applied_index: marks.applied_index,
                visible_index: marks.visible_index,
                commit_apply_gap: marks.commit_apply_gap,
                apply_visible_gap: marks.apply_visible_gap,
            },
            ReadinessStatus {
                pending_batch_len: marks.pending_batch_len,
                pending_batch_cap: marks.pending_batch_cap,
                active_txn_count: marks.active_txn_count,
                wal_unflushed_count: marks.wal_unflushed_count,
                backlog_blocker_count: marks.backlog_blocker_count,
                backlog_blocker_mask: marks.backlog_blocker_mask,
                mutation_admission_saturated: marks.mutation_admission_saturated,
                quiescent_for_failover: marks.quiescent_for_failover,
                follower_promotion_ready: marks.follower_promotion_ready,
            },
            FallbackStatus {
                last_reason: runtime_metrics.last_fallback_reason,
                gpu_parity_fallbacks: self.metrics.fallback_counts_by_gpu_parity_issue(),
                active_reasons,
                gpu_runtime,
            },
            runtime_metrics,
        )
        .expect("engine status snapshot invariants should hold");
        status.relational_residency = self.relational_residency_status();
        status
    }

    pub fn publish_telemetry<S: TelemetrySink>(&self, sink: &mut S) {
        sink.publish(&self.telemetry_snapshot());
    }

    pub fn pending_batch_len(&self) -> usize {
        self.batcher().len()
    }

    pub fn has_pending_batch(&self) -> bool {
        !self.batcher().is_empty()
    }

    pub fn pending_batch_oldest_age(&self, now: Instant) -> Option<Duration> {
        self.batcher()
            .first_enqueued_at()
            .map(|head| now.saturating_duration_since(head))
    }

    pub fn pending_batch_time_until_deadline(&self, now: Instant) -> Option<Duration> {
        self.batcher().time_until_flush_deadline(now)
    }

    pub fn batching_config(&self) -> (usize, Duration) {
        // ONE batcher lock: two `self.batcher()` in a tuple would keep the first guard alive while
        // taking the second — a self-deadlock on the non-reentrant mutex.
        let batcher = self.batcher();
        (batcher.max_items(), batcher.max_wait())
    }

    pub(crate) fn route_command(&self, cmd: &Command) -> RouteDecision {
        let plan = self.planner.plan_command(cmd);
        let Some(node) = plan.nodes().first() else {
            return RouteDecision::Cpu;
        };
        self.router.route(&node.op)
    }

    pub(crate) fn simulate_kernel_occupancy_permyriad(payload_len: usize) -> u16 {
        // Bootstrap heuristic for no-GPU mode: scale occupancy with payload size
        // while capping at 100% to keep telemetry realistic.
        let permyriad = 2_500u64.saturating_add((payload_len as u64).saturating_mul(100));
        permyriad.min(10_000) as u16
    }
}
