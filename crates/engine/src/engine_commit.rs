//! Commit / replication-apply path (P0 §9.6 decomposition, behavior-preserving):
//! a focused `impl Engine` block for the Raft role transitions (become_follower/
//! leader/candidate), the commit oracle (commit_mutation, commit_mutation_at,
//! next_commit_timestamp_micros, commit_mutation_at_with_current_apply), the
//! resident-memory invalidation appliers (invalidate_relational_residency and its
//! table/concurrent/for-commit/for-memory-pressure variants + scope), and the
//! committed MVCC log-entry applier (apply_mvcc_entry).

use super::*;

impl Engine {
    pub fn become_follower(&mut self, term: Term) {
        self.commit_state_mut().repl.become_follower(term);
    }

    pub fn become_leader(&mut self, term: Term) {
        self.commit_state_mut().repl.become_leader(term);
    }

    pub fn become_candidate(&mut self, term: Term) {
        self.commit_state_mut().repl.become_candidate(term);
    }

    pub fn commit_mutation(
        &self,
        txn_id: u64,
        payload: Vec<u8>,
    ) -> Result<CommitToken, EngineError> {
        let timestamp_micros = self.next_commit_timestamp_micros();
        self.commit_mutation_at(txn_id, payload, timestamp_micros)
    }

    pub(crate) fn next_commit_timestamp_micros(&self) -> u64 {
        let wall_clock = current_timestamp_micros();
        self.commit_state()
            .wal_commit_timestamps_micros
            .values()
            .copied()
            .max()
            .map(|last| wall_clock.max(last.saturating_add(1)))
            .unwrap_or(wall_clock)
    }

    pub fn commit_mutation_at(
        &self,
        txn_id: u64,
        payload: Vec<u8>,
        timestamp_micros: u64,
    ) -> Result<CommitToken, EngineError> {
        if self.repl_role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        // Durable-commit critical section under the **commit_mutex** (A.4 unification — this method is
        // now `&self`, the SERIALIZED commit path for DDL / KV / sequence-default INSERT / replay,
        // reached through `&Engine`; the concurrent autocommit-DML path is `commit_dml_concurrent`,
        // which runs the SAME WAL-before-publish sequence under this same lock). Lock order is fixed:
        // `commit_state()` (commit_mutex) FIRST, then `ddl_catalog()` (catalog latch) acquired INSIDE.
        // `next_commit_timestamp_micros` (which itself locks the commit_mutex) is computed by the
        // caller BEFORE this, so we never re-enter the non-reentrant commit_mutex.
        let mut commit = self.commit_state();
        let token = {
            let wal_len_before = commit.wal.len();
            commit.wal.append(WalRecord {
                txn_id,
                payload: payload.clone(),
            });
            let token = match commit.repl.propose(payload) {
                Ok(token) => token,
                Err(err) => {
                    commit.wal.truncate(wal_len_before);
                    return Err(err);
                }
            };
            if let Err(err) = commit.wal.flush_all() {
                commit.repl.rollback_unapplied_from(token.index);
                commit.wal.truncate(wal_len_before);
                return Err(err);
            }
            commit
                .repl
                .wait_committed(token, Duration::from_millis(0))?;
            // `txn_id` (the façade `next_txn_id`) is the durable transaction *identity* — recorded in
            // the WAL record and keyed here for PITR lookups. Intentionally DECOUPLED from the MVCC
            // version stamp, which uses the commit `Index` (see `apply_mvcc_entry`).
            commit
                .wal_commit_timestamps_micros
                .insert(txn_id, timestamp_micros);
            token
        };

        let to_apply: Vec<LogEntry> = commit
            .repl
            .drain_committed_from(commit.repl.applied_index())
            .cloned()
            .collect();

        // Hold the catalog latch across the WHOLE apply loop AND the catalog publish (PART B), so a
        // DDL's working-map mutation + the published-snapshot push are atomic w.r.t. another DDL. Lock
        // order is fixed: commit_mutex (held in `commit`) FIRST, then this latch.
        {
            let mut catalog_guard = self.ddl_catalog();
            let cat = &mut *catalog_guard;
            for e in &to_apply {
                commit.sm.apply(e)?;
                self.apply_mvcc_entry(e, cat)?;
                commit.repl.mark_applied(e.index);
            }

            // Publish ordering (Stage 2 — blocker #1; PART B catalog↔data co-pinning). The apply loop
            // published this commit's data generation(s) and mutated the working catalog maps (for any
            // DDL entries). Order the rest so a lock-free reader gets a consistent (catalog, data) pair:
            //   1. residency tombstones, 2. catalog ring push (stamped at `token.index`), then LAST
            //   3. `committed_seq` release-store.
            // The catalog is pushed BEFORE `committed_seq` (the FLIP from the old order) so a reader
            // that loads `committed_seq = token.index` and selects `catalog_as_of(token.index)` is
            // guaranteed to find this generation — the catalog is visible no later than `committed_seq`.
            // That, with the per-boundary self-consistency of the data (MVCC versions stamp old/new
            // part-counts at the DDL's commit_seq), rules out a reader straddling a shape-changing DDL.
            self.invalidate_relational_residency_for_commit(&to_apply, txn_id, token.index);
            let prune_below = self.catalog_prune_boundary(token.index);
            self.publish_catalog_snapshot(cat, token.index, prune_below);
        }
        self.publish_committed_seq(token.index);
        // STRATA S-B: best-effort GPU-residency admission for the committed mutation's tables (flag-gated,
        // after the publish so it snapshots the new generation; never fails the already-durable commit).
        if self.auto_admit_on_commit_enabled() {
            if let Some(tables) = Self::residency_invalidation_scope(&to_apply) {
                self.auto_admit_resident_tables(&tables);
            }
        }
        self.metrics.inc_commit();
        drop(commit);

        Ok(token)
    }

    pub(crate) fn commit_mutation_at_with_current_apply<F>(
        &self,
        txn_id: u64,
        payload: Vec<u8>,
        timestamp_micros: u64,
        mut apply_current: F,
    ) -> Result<(CommitToken, u128), EngineError>
    where
        // `apply_current` receives the commit sequence (the replicator-assigned commit `Index`) so
        // the directly-applied current entry stamps versions with the SAME commit-seq that
        // `apply_mvcc_entry` derives from `entry.index` on replay — keeping the live COPY hot path
        // byte-identical to a WAL replay of the same record (Stage 0 stamp/boundary unification).
        F: FnMut(&Self, &mut DdlCatalogState, Index) -> Result<(), EngineError>,
    {
        if self.repl_role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        // A.4 unification: `&self`, the whole critical section under the commit_mutex (held in
        // `commit`); the catalog latch is acquired INSIDE (fixed lock order).
        let mut commit = self.commit_state();
        let token = {
            let wal_len_before = commit.wal.len();
            commit.wal.append(WalRecord {
                txn_id,
                payload: payload.clone(),
            });
            let token = match commit.repl.propose(payload) {
                Ok(token) => token,
                Err(err) => {
                    commit.wal.truncate(wal_len_before);
                    return Err(err);
                }
            };
            if let Err(err) = commit.wal.flush_all() {
                commit.repl.rollback_unapplied_from(token.index);
                commit.wal.truncate(wal_len_before);
                return Err(err);
            }
            commit
                .repl
                .wait_committed(token, Duration::from_millis(0))?;
            // `txn_id` is the durable transaction identity (decoupled from the MVCC `commit_seq`).
            commit
                .wal_commit_timestamps_micros
                .insert(txn_id, timestamp_micros);
            token
        };

        let to_apply: Vec<LogEntry> = commit
            .repl
            .drain_committed_from(commit.repl.applied_index())
            .cloned()
            .collect();

        // Hold the catalog latch across the apply loop AND the catalog publish (PART B; lock order:
        // commit_mutex FIRST, then this latch).
        let residency_invalidation_micros;
        {
            let mut catalog_guard = self.ddl_catalog();
            let cat = &mut *catalog_guard;
            for e in &to_apply {
                if e.index == token.index {
                    // The caller applies the current entry directly through Engine state. Avoid
                    // cloning and reparsing the large SQL payload through the generic KV state
                    // machine on the COPY hot path while preserving WAL/replay records. Pass the
                    // commit sequence (`e.index`) so the stamp matches `apply_mvcc_entry`'s replay
                    // stamp, plus the held catalog latch for any working-map mutation.
                    apply_current(self, cat, e.index)?;
                } else {
                    commit.sm.apply(e)?;
                    self.apply_mvcc_entry(e, cat)?;
                }
                commit.repl.mark_applied(e.index);
            }

            // Publish ordering (PART B): residency → catalog ring push → `committed_seq` LAST (mirrors
            // `commit_mutation_at`). The current-apply closure already published data + mutated the maps.
            let residency_invalidation_started = Instant::now();
            self.invalidate_relational_residency_for_commit(&to_apply, txn_id, token.index);
            residency_invalidation_micros = residency_invalidation_started.elapsed().as_micros();
            let prune_below = self.catalog_prune_boundary(token.index);
            self.publish_catalog_snapshot(cat, token.index, prune_below);
        }
        self.publish_committed_seq(token.index);
        // STRATA S-B: best-effort GPU-residency admission for the committed mutation's tables (flag-gated,
        // after the publish so it snapshots the new generation; never fails the already-durable commit).
        if self.auto_admit_on_commit_enabled() {
            if let Some(tables) = Self::residency_invalidation_scope(&to_apply) {
                self.auto_admit_resident_tables(&tables);
            }
        }
        self.metrics.inc_commit();

        Ok((token, residency_invalidation_micros))
    }

    /// Global (stop-the-world) residency invalidation: invalidate every resident
    /// table. Retained as the **conservative fallback** for commit batches whose
    /// mutated tables cannot be determined precisely ([`Engine::residency_invalidation_scope`]
    /// returns `None`). Equivalent to invalidating each resident table individually.
    fn invalidate_relational_residency(&self, txn_id: TxnId, index: Index) {
        let tables: BTreeSet<String> = self
            .read_state
            .residency
            .snapshots
            .load()
            .keys()
            .cloned()
            .chain(self.read_state.residency.shards.load().keys().cloned())
            .collect();
        for table in &tables {
            self.invalidate_relational_residency_table(table, txn_id, index);
        }
    }

    /// Invalidate the residency of a **single** table (its snapshot, device memory,
    /// and shards). This is the per-table unit the commit path uses so a write to
    /// one table no longer evicts every other table's residency — the former
    /// stop-the-world behavior. Summing this over all resident tables reproduces the
    /// previous global invalidation; the device-memory/shard removals here are
    /// unconditional, so in degenerate cache states it may clear a stray cross-map
    /// entry the old two-loop form left — strictly-safe extra cleanup, never stale.
    fn invalidate_relational_residency_table(&self, table: &str, txn_id: TxnId, index: Index) {
        // Stage 3 — blocker #2: the snapshot/shard flag maps are now published behind `ArcSwap`,
        // so flag the invalidation copy-on-write under the serialized catalog latch (this never runs
        // on the concurrent commit path — that uses `invalidate_relational_residency_tables_concurrent`
        // which only tombstones the device-memory cells via `&self`).
        self.read_state.residency.with_snapshots_mut(|snapshots| {
            if let Some(entry) = snapshots.get_mut(table) {
                // make_mut COWs the shared descriptor into a fresh version (host_rows stays shared).
                let snapshot = std::sync::Arc::make_mut(&mut entry.descriptor);
                if snapshot.invalidated_by_txn_id.is_none() {
                    snapshot.invalidated_by_txn_id = Some(txn_id);
                    snapshot.invalidated_at_index = Some(index);
                }
                if let Some(proof) = snapshot.device_memory_proof.as_mut() {
                    proof.retained = false;
                }
            }
        });
        self.read_state.residency.device_memory.invalidate(table);
        self.read_state.residency.with_shards_mut(|shards| {
            if let Some(shards) = shards.get_mut(table) {
                for shard in shards.iter_mut() {
                    if shard.invalidated_by_txn_id.is_none() {
                        shard.invalidated_by_txn_id = Some(txn_id);
                        shard.invalidated_at_index = Some(index);
                    }
                    if let Some(proof) = shard.device_memory_proof.as_mut() {
                        proof.retained = false;
                    }
                }
            }
        });
        self.read_state
            .residency
            .shard_device_memory
            .invalidate_table(table);
    }

    /// Invalidate the GPU residency of the `tables` a CONCURRENT commit mutated, via `&self`
    /// (write-half MVCC, Stage 4). Publishes a `None` tombstone on each table's resident
    /// device-memory cell(s) — the authoritative gate the read-path's `plan_relational_resident_route`
    /// checks (`has_retained_device_memory`), so after this a reader takes the CPU route on the new
    /// committed data rather than a stale GPU snapshot (residency↔data consistency, design Risk #3).
    /// Called INSIDE the commit critical section, before `committed_seq` is bumped, so a reader that
    /// observes the new `committed_seq` also observes the residency tombstone.
    ///
    /// It deliberately does NOT mutate the `snapshots`/`shards` flag maps (those are not
    /// interior-mutable and are read lock-free by `&self` readers): the cell tombstone alone forces
    /// the CPU route. The flag-based telemetry / the explicit resident-snapshot-probe API are kept
    /// current only on the serialized invalidation path (`invalidate_relational_residency_table`),
    /// which runs under the exclusive catalog latch.
    pub(crate) fn invalidate_relational_residency_tables_concurrent(
        &self,
        tables: &BTreeSet<String>,
        _txn_id: TxnId,
        _index: Index,
    ) {
        for table in tables {
            self.read_state.residency.device_memory.invalidate(table);
            self.read_state
                .residency
                .shard_device_memory
                .invalidate_table(table);
        }
    }

    /// The set of tables a committed batch invalidates, or `None` to fall back to a
    /// global invalidation. **Conservative by construction:** it narrows only for
    /// commands whose mutated table(s) are unambiguous (single-table DML, TRUNCATE,
    /// DROP TABLE) and treats CREATE TABLE as touching no existing residency. Any other
    /// command — or a payload that fails to decode or parse — returns `None`, so
    /// residency is never left stale. Over-invalidation is merely a performance cost;
    /// under-invalidation would serve wrong rows, so this must never narrow when unsure.
    pub(crate) fn residency_invalidation_scope(entries: &[LogEntry]) -> Option<BTreeSet<String>> {
        let mut tables = BTreeSet::new();
        for entry in entries {
            let text = std::str::from_utf8(&entry.payload).ok()?;
            let command = parse_command(text).ok()?;
            match command {
                Command::Insert(insert) => {
                    tables.insert(insert.table);
                }
                Command::Update(update) => {
                    tables.insert(update.table);
                }
                Command::Delete(delete) => {
                    tables.insert(delete.table);
                }
                Command::TruncateTable(truncate) => {
                    tables.insert(truncate.name);
                }
                Command::DropTable(drop) => {
                    tables.extend(drop.names);
                }
                // A brand-new table has no prior residency to invalidate.
                Command::CreateTable(_) => {}
                // Any other command (other DDL, ACL, KV, schema/db/role/...) is not yet
                // precisely scoped; invalidate everything rather than risk staleness.
                _ => return None,
            }
        }
        Some(tables)
    }

    /// Invalidate residency for a committed batch: per-table when the mutated tables
    /// can be determined, else a conservative global invalidation.
    fn invalidate_relational_residency_for_commit(
        &self,
        entries: &[LogEntry],
        txn_id: TxnId,
        index: Index,
    ) {
        match Self::residency_invalidation_scope(entries) {
            Some(tables) => {
                for table in &tables {
                    self.invalidate_relational_residency_table(table, txn_id, index);
                }
            }
            None => self.invalidate_relational_residency(txn_id, index),
        }
    }

    pub(crate) fn invalidate_relational_residency_for_memory_pressure(&self, gpu_id: u16) {
        // Stage 3 — blocker #2: COW the snapshot/shard flag maps under the catalog latch, then
        // tombstone the device-memory cells of every table that was pressured (the cell `invalidate`
        // is `&self`, done outside the COW closure on the collected tables).
        let pressured_snapshot_tables = self.read_state.residency.with_snapshots_mut(|snapshots| {
            let mut tables = Vec::new();
            for (table, entry) in snapshots.iter_mut() {
                if entry.descriptor.gpu_id == gpu_id {
                    // COW only the pressured tables' descriptors (host_rows stays shared).
                    let snapshot = std::sync::Arc::make_mut(&mut entry.descriptor);
                    snapshot.invalidated_by_memory_pressure = true;
                    snapshot.memory_pressure_active = true;
                    if let Some(proof) = snapshot.device_memory_proof.as_mut() {
                        proof.retained = false;
                    }
                    tables.push(table.clone());
                }
            }
            tables
        });
        for table in &pressured_snapshot_tables {
            self.read_state.residency.device_memory.invalidate(table);
        }
        let pressured_shard_tables =
            self.read_state.residency.with_shards_mut(|shards| {
                let mut tables = Vec::new();
                for (table, table_shards) in shards.iter_mut() {
                    let mut table_pressured = false;
                    for shard in table_shards {
                        if shard.gpu_id == gpu_id {
                            shard.invalidated_by_memory_pressure = true;
                            shard.memory_pressure_active = true;
                            table_pressured = true;
                            if let Some(proof) = shard.device_memory_proof.as_mut() {
                                proof.retained = false;
                            }
                        }
                    }
                    if table_pressured {
                        tables.push(table.clone());
                    }
                }
                tables
            });
        for table in &pressured_shard_tables {
            self.read_state
                .residency
                .shard_device_memory
                .invalidate_table(table);
        }
    }

    /// Apply one committed log entry to the engine state (`&self`). The caller holds the **catalog
    /// latch** and passes `&mut DdlCatalogState` so a DDL entry's working-map mutation can be made
    /// atomic with the subsequent catalog-snapshot publish (the caller holds the SAME guard across both
    /// — PART B). Lock order is fixed: the caller already holds the commit_mutex, then the catalog
    /// latch. The `apply_*` methods (now `&self`) freely call `self.read_state.*` and the `&self`
    /// `preflight_*` tree (which reads the published catalog snapshot, never this latch — no reentry).
    fn apply_mvcc_entry(
        &self,
        entry: &LogEntry,
        cat: &mut DdlCatalogState,
    ) -> Result<(), EngineError> {
        let Ok(text) = std::str::from_utf8(&entry.payload) else {
            return Ok(());
        };
        let Ok(cmd) = parse_command(text) else {
            return Ok(());
        };

        // Stage 0 (write-half MVCC): the version stamp is the commit sequence, which is the
        // replicator-assigned commit `Index` (== WAL append order == read boundary `visible_up_to`).
        // Deliberately NOT the façade `next_txn_id`: under the future off-lock prepare, txn_id
        // allocation order diverges from commit order, but `entry.index` is always in commit order.
        // Because recovery re-proposes WAL records in log order, each entry gets the identical
        // monotonic `entry.index` on replay, so this re-derives byte-identical `created_by`/
        // `deleted_by` stamps from log position alone — independent of the recorded façade txn_id.
        let commit_seq: TxnId = entry.index;
        let visibility = StorageVisibility {
            read_txn_id: commit_seq,
        };

        match cmd {
            Command::SetKv { key, value } => {
                // KV lives in its own shard. Read the current version under the loaded
                // generation, then publish a new KV generation with the update/insert applied.
                let existing = self
                    .read_state
                    .mvcc
                    .load_kv()
                    .get()
                    .tuple_fetch_by_key(&key, visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                let tuple_id = if existing.is_none() {
                    Some(self.read_state.mvcc.reserve_tuple_id())
                } else {
                    None
                };
                self.read_state.mvcc.with_kv_mut(|store| {
                    if let Some(tuple) = existing {
                        store
                            .tuple_update(tuple.tuple_id, value, commit_seq)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))
                    } else {
                        store
                            .tuple_insert_with_id(
                                tuple_id.expect("fresh id reserved for new key"),
                                NewTuple { key, value },
                                commit_seq,
                            )
                            .map(|_| ())
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))
                    }
                })?;
            }
            Command::DeleteKv { key } => {
                let existing = self
                    .read_state
                    .mvcc
                    .load_kv()
                    .get()
                    .tuple_fetch_by_key(&key, visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                if let Some(tuple) = existing {
                    self.read_state.mvcc.with_kv_mut(|store| {
                        store
                            .tuple_delete(tuple.tuple_id, commit_seq)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))
                    })?;
                }
            }
            Command::CreateSchema(create) => self.apply_create_schema(cat, create)?,
            Command::DropSchema(drop) => self.apply_drop_schema(cat, drop)?,
            Command::CreateDatabase(create) => self.apply_create_database(cat, create)?,
            Command::DropDatabase(drop) => self.apply_drop_database(cat, drop)?,
            Command::RenameDatabase(rename) => self.apply_rename_database(cat, rename)?,
            Command::CreateTablespace(create) => self.apply_create_tablespace(cat, create)?,
            Command::DropTablespace(drop) => self.apply_drop_tablespace(cat, drop)?,
            Command::RenameTablespace(rename) => self.apply_rename_tablespace(cat, rename)?,
            Command::CreateTable(create) => self.apply_create_table(cat, create)?,
            Command::AddPrimaryKey(add) => self.apply_add_primary_key(cat, add)?,
            Command::AddUniqueConstraint(add) => self.apply_add_unique_constraint(cat, add)?,
            Command::AddCheckConstraint(add) => self.apply_add_check_constraint(cat, add)?,
            Command::AddForeignKey(add) => self.apply_add_foreign_key(cat, add, commit_seq)?,
            Command::AddColumn(add) => self.apply_add_column(cat, add, commit_seq)?,
            Command::RenameTable(rename) => self.apply_rename_table(cat, rename, commit_seq)?,
            Command::RenameColumn(rename) => self.apply_rename_column(cat, rename)?,
            Command::RenameConstraint(rename) => self.apply_rename_constraint(cat, rename)?,
            Command::DropColumn(drop) => self.apply_drop_column(cat, drop, commit_seq)?,
            Command::DropConstraint(drop) => self.apply_drop_constraint(cat, drop)?,
            Command::CreateIndex(create) => self.apply_create_index(cat, create)?,
            Command::RenameIndex(rename) => self.apply_rename_index(cat, rename)?,
            Command::CreateView(create) => self.apply_create_view(cat, create)?,
            Command::RenameView(rename) => self.apply_rename_view(cat, rename)?,
            Command::CreateMaterializedView(create) => {
                self.apply_create_materialized_view(cat, create)?
            }
            Command::RefreshMaterializedView(refresh) => {
                self.apply_refresh_materialized_view(cat, refresh)?
            }
            Command::RenameMaterializedView(rename) => {
                self.apply_rename_materialized_view(cat, rename)?
            }
            Command::CreateFunction(create) => self.apply_create_function(cat, create)?,
            Command::RenameFunction(rename) => self.apply_rename_function(cat, rename)?,
            Command::DropFunction(drop) => self.apply_drop_function(cat, drop)?,
            Command::SelectFunction(_) => {}
            Command::CreateSequence(create) => self.apply_create_sequence(cat, create)?,
            Command::CreateDomain(create) => self.apply_create_domain(cat, create)?,
            Command::SequenceNextVal(nextval) => {
                self.apply_sequence_nextval(cat, nextval)?;
            }
            Command::SequenceSetVal(setval) => {
                self.apply_sequence_setval(cat, setval)?;
            }
            Command::RenameSequence(rename) => self.apply_rename_sequence(cat, rename)?,
            Command::DropTable(drop) => self.apply_drop_table(cat, drop, commit_seq)?,
            Command::TruncateTable(truncate) => {
                self.apply_truncate_table(cat, truncate, commit_seq)?
            }
            Command::DropIndex(drop) => self.apply_drop_index(cat, drop)?,
            Command::DropView(drop) => self.apply_drop_view(cat, drop)?,
            Command::DropMaterializedView(drop) => self.apply_drop_materialized_view(cat, drop)?,
            Command::DropSequence(drop) => self.apply_drop_sequence(cat, drop)?,
            Command::DropDomain(drop) => self.apply_drop_domain(cat, drop)?,
            Command::CreatePublication(create) => self.apply_create_publication(cat, create)?,
            Command::DropPublication(drop) => self.apply_drop_publication(cat, drop)?,
            Command::CreateSubscription(create) => self.apply_create_subscription(cat, create)?,
            Command::DropSubscription(drop) => self.apply_drop_subscription(cat, drop)?,
            Command::CreateRole(create) => self.apply_create_role(cat, create)?,
            Command::DropRole(drop) => self.apply_drop_role(cat, drop)?,
            Command::RenameRole(rename) => self.apply_rename_role(cat, rename)?,
            Command::GrantTable(grant) => self.apply_grant_acl(
                cat,
                &grant.relation,
                grant.kind,
                &grant.grantee,
                &grant.privileges,
            )?,
            Command::RevokeTable(revoke) => self.apply_revoke_acl(
                cat,
                &revoke.relation,
                revoke.kind,
                &revoke.grantee,
                &revoke.privileges,
            )?,
            Command::GrantSchema(grant) => {
                self.apply_grant_schema_acl(cat, &grant.schema, &grant.grantee, &grant.privileges)?
            }
            Command::RevokeSchema(revoke) => self.apply_revoke_schema_acl(
                cat,
                &revoke.schema,
                &revoke.grantee,
                &revoke.privileges,
            )?,
            Command::GrantDatabase(grant) => self.apply_grant_database_acl(
                cat,
                &grant.database,
                &grant.grantee,
                &grant.privileges,
            )?,
            Command::RevokeDatabase(revoke) => self.apply_revoke_database_acl(
                cat,
                &revoke.database,
                &revoke.grantee,
                &revoke.privileges,
            )?,
            Command::GrantTablespace(grant) => self.apply_grant_tablespace_acl(
                cat,
                &grant.tablespace,
                &grant.grantee,
                &grant.privileges,
            )?,
            Command::RevokeTablespace(revoke) => self.apply_revoke_tablespace_acl(
                cat,
                &revoke.tablespace,
                &revoke.grantee,
                &revoke.privileges,
            )?,
            Command::GrantFunction(grant) => self.apply_grant_function_acl(
                cat,
                &grant.function,
                &grant.grantee,
                &grant.privileges,
            )?,
            Command::RevokeFunction(revoke) => self.apply_revoke_function_acl(
                cat,
                &revoke.function,
                &revoke.grantee,
                &revoke.privileges,
            )?,
            Command::GrantDefaultTablePrivileges(grant) => {
                self.apply_grant_default_table_privileges(cat, &grant.grantee, &grant.privileges)?
            }
            Command::RevokeDefaultTablePrivileges(revoke) => self
                .apply_revoke_default_table_privileges(cat, &revoke.grantee, &revoke.privileges)?,
            Command::AlterColumnDefault(alter) => self.apply_alter_column_default(cat, alter)?,
            Command::CommentOn(comment) => self.apply_comment_on(cat, comment)?,
            Command::Insert(insert) => self.apply_insert(cat, insert, commit_seq)?,
            Command::Delete(delete) => self.apply_delete(cat, delete, commit_seq)?,
            Command::Update(update) => self.apply_update(cat, update, commit_seq)?,
            _ => {}
        }

        Ok(())
    }
}
