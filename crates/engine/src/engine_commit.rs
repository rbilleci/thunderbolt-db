//! Commit / replication-apply path (P0 §9.6 decomposition, behavior-preserving):
//! a focused `impl Engine` block for the Raft role transitions (become_follower/
//! leader/candidate), the commit oracle (commit_mutation, commit_mutation_at,
//! next_commit_timestamp_micros, commit_mutation_at_with_current_apply), the
//! resident-memory invalidation appliers (invalidate_relational_residency and its
//! table/concurrent/for-commit/for-memory-pressure variants + scope), and the
//! committed MVCC log-entry applier (apply_mvcc_entry).

use super::*;

/// Failure from [`Engine::commit_mutation_batch`]. `rolled_back` distinguishes a clean pre-durable
/// abort (WAL truncated + proposals rolled back — every item may be requeued and retried) from a
/// post-fsync failure (the batch's records are durable; retrying would append duplicates).
pub(crate) struct BatchCommitFailure {
    pub(crate) rolled_back: bool,
    pub(crate) error: EngineError,
}

impl Engine {
    pub fn become_follower(&mut self, term: Term) {
        self.commit_state_mut().repl.become_follower(term);
        self.publish_repl_role_mirror();
    }

    pub fn become_leader(&mut self, term: Term) {
        self.commit_state_mut().repl.become_leader(term);
        self.publish_repl_role_mirror();
    }

    pub fn become_candidate(&mut self, term: Term) {
        self.commit_state_mut().repl.become_candidate(term);
        self.publish_repl_role_mirror();
    }

    /// Refresh the lock-free role mirror from the replicator after a role transition (the only
    /// mutation points route through the three `become_*` wrappers above).
    fn publish_repl_role_mirror(&mut self) {
        let role = self.commit_state_mut().repl.role();
        self.repl_role_mirror.store(
            match role {
                Role::Leader => 0,
                Role::Follower => 1,
                Role::Candidate => 2,
            },
            AtomicOrdering::Release,
        );
    }

    pub fn commit_mutation(
        &self,
        txn_id: u64,
        payload: std::sync::Arc<[u8]>,
    ) -> Result<CommitToken, EngineError> {
        let timestamp_micros = self.next_commit_timestamp_micros();
        self.commit_mutation_at(txn_id, payload, timestamp_micros)
    }

    pub(crate) fn next_commit_timestamp_micros(&self) -> u64 {
        let wall_clock = current_timestamp_micros();
        // O(1): read the running max instead of scanning the never-pruned timestamp map. Identical
        // to the old `wal_commit_timestamps_micros.values().max().map(|last| wall.max(last+1))`:
        // before any commit the max is 0, so `wall.max(0+1) == wall` reproduces the empty-map arm
        // (wall-clock micros dwarf 1); after commits it is `wall.max(prior_max + 1)`, guaranteeing a
        // strictly-monotonic timestamp >= wall clock.
        let prior_max = self.commit_state().max_commit_timestamp_micros;
        wall_clock.max(prior_max.saturating_add(1))
    }

    pub fn commit_mutation_at(
        &self,
        txn_id: u64,
        payload: std::sync::Arc<[u8]>,
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
            commit.record_commit_timestamp(txn_id, timestamp_micros);
            token
        };

        self.apply_and_publish_committed(&mut commit, txn_id, token.index)?;
        self.metrics.inc_commit();
        drop(commit);

        Ok(token)
    }

    /// Commit a BATCH of mutations as ONE WAL flush group (write-path assessment D3 / scalability
    /// ledger #7): every record is appended + proposed under the commit_mutex, the whole tail is
    /// made durable by a SINGLE `flush_all` (one fsync), and only then are the entries applied in
    /// proposal order and `committed_seq` published once at the batch's last commit seq. Compared
    /// to committing each item through [`Engine::commit_mutation`], a k-item batch pays 1 fsync
    /// instead of k. WAL-before-visibility is unchanged: nothing is applied or published until
    /// the group fsync has succeeded.
    ///
    /// Failure semantics:
    /// - `rolled_back: true` — the failure happened BEFORE anything was durable (propose or the
    ///   group fsync): the WAL tail is truncated and the proposals rolled back; the caller may
    ///   safely requeue and retry every item (none of them committed).
    /// - `rolled_back: false` — the failure happened AFTER the group fsync: the batch's records
    ///   are already durable and MUST NOT be retried (a retry would append duplicate records; a
    ///   restart replays the durable log as the source of truth).
    pub(crate) fn commit_mutation_batch(
        &self,
        items: &[(TxnId, std::sync::Arc<[u8]>)],
    ) -> Result<(), BatchCommitFailure> {
        let Some((last_txn_id, _)) = items.last() else {
            return Ok(());
        };
        let last_txn_id = *last_txn_id;
        if self.repl_role() != Role::Leader {
            return Err(BatchCommitFailure {
                rolled_back: true,
                error: EngineError::NotLeader,
            });
        }
        let wall_clock = current_timestamp_micros();

        let mut commit = self.commit_state();
        let last_index = {
            let wal_len_before = commit.wal.len();
            let mut first_index: Option<Index> = None;
            let mut last_index = 0;
            for (txn_id, payload) in items {
                commit.wal.append(WalRecord {
                    txn_id: *txn_id,
                    payload: payload.clone(),
                });
                match commit.repl.propose(payload.clone()) {
                    Ok(token) => {
                        first_index.get_or_insert(token.index);
                        last_index = token.index;
                    }
                    Err(error) => {
                        if let Some(first) = first_index {
                            commit.repl.rollback_unapplied_from(first);
                        }
                        commit.wal.truncate(wal_len_before);
                        return Err(BatchCommitFailure {
                            rolled_back: true,
                            error,
                        });
                    }
                }
            }
            // THE group-commit point: one fsync covers every record appended above.
            if let Err(error) = commit.wal.flush_all() {
                commit
                    .repl
                    .rollback_unapplied_from(first_index.expect("non-empty batch proposed"));
                commit.wal.truncate(wal_len_before);
                return Err(BatchCommitFailure {
                    rolled_back: true,
                    error,
                });
            }
            if let Err(error) = commit
                .repl
                .wait_committed(CommitToken { index: last_index }, Duration::from_millis(0))
            {
                // The records are already fsync-durable; surface the replication failure without
                // pretending the batch can be cleanly retried.
                return Err(BatchCommitFailure {
                    rolled_back: false,
                    error,
                });
            }
            // Per-item strictly-monotonic commit timestamps (same formula as
            // `next_commit_timestamp_micros`, inlined because the commit_mutex is already held).
            for (txn_id, _) in items {
                let timestamp_micros =
                    wall_clock.max(commit.max_commit_timestamp_micros.saturating_add(1));
                commit.record_commit_timestamp(*txn_id, timestamp_micros);
            }
            last_index
        };

        self.apply_and_publish_committed(&mut commit, last_txn_id, last_index)
            .map_err(|error| BatchCommitFailure {
                rolled_back: false,
                error,
            })?;
        for _ in items {
            self.metrics.inc_commit();
        }
        Ok(())
    }

    /// Shared post-durability tail of the serialized commit paths: drain the committed entries,
    /// apply them under the catalog latch, maintain/invalidate residency, publish the catalog
    /// snapshot, then publish `committed_seq` at `publish_index` — callers run this only AFTER
    /// the corresponding WAL records are fsync-durable (WAL-before-visibility).
    ///
    /// AUDIT f80f2350 FINDING A: the whole apply runs inside the commit critical section, so
    /// every internal read it performs (the DDL row-validators via `visible_relational_rows`,
    /// mat-view reads, the elision rehydrate seams) is flagged via
    /// `skip_leader_check_during_internal_read` — a lock-aware seam
    /// (`rehydrate_elided_serialized`) then takes its DIRECT branch instead of self-deadlocking
    /// on the commit_mutex re-lock. The repro this closes: a concurrent INSERT wave RE-ELIDED a
    /// table during a single-entry `CREATE UNIQUE INDEX` commit's fsync window (the off-lock
    /// execute_text sweep had de-elided it earlier); the apply-time unique validator then hit
    /// the rehydrate seam under the held lock and wedged the commit path permanently.
    fn apply_and_publish_committed(
        &self,
        commit: &mut CommitState,
        txn_id: TxnId,
        publish_index: Index,
    ) -> Result<(), EngineError> {
        self.skip_leader_check_during_internal_read(|engine| {
            engine.apply_and_publish_committed_inner(commit, txn_id, publish_index)
        })
    }

    fn apply_and_publish_committed_inner(
        &self,
        commit: &mut CommitState,
        txn_id: TxnId,
        publish_index: Index,
    ) -> Result<(), EngineError> {
        let to_apply: Vec<LogEntry> = commit
            .repl
            .drain_committed_from(commit.repl.applied_index())
            .cloned()
            .collect();
        // RETIREMENT A4e (audit B2): a MULTI-ENTRY commit bypasses the single-mutation elision
        // hooks below (they rehydrate exactly ONE mutation's delta) — every entry's apply would
        // skip the host install and the final invalidate+re-admit would rebuild every touched
        // table from its STALE store (elided-era rows lost). Rehydrate every elided table in the
        // batch's scope FIRST (under this commit lock; state through committed_seq is fully on
        // device) — the tables de-elide, the applies install normally, the re-admit is truthful.
        if to_apply.len() > 1 && self.host_install_elision_enabled() {
            if let Some(scope) = Self::residency_invalidation_scope(&to_apply) {
                for table_name in &scope {
                    if self.table_install_elided(table_name) {
                        let Some(table) = self.relational_catalog_table(table_name) else {
                            continue;
                        };
                        let seq = self.committed_seq();
                        self.rehydrate_elided_table(
                            &table,
                            seq,
                            &Default::default(),
                            &Default::default(),
                            seq,
                        )?;
                    }
                }
            }
        }

        // Hold the catalog latch across the WHOLE apply loop AND the catalog publish (PART B), so a
        // DDL's working-map mutation + the published-snapshot push are atomic w.r.t. another DDL. Lock
        // order is fixed: commit_mutex (held by the caller) FIRST, then this latch.
        let handled = {
            let mut catalog_guard = self.ddl_catalog();
            let cat = &mut *catalog_guard;
            let mut applied: Option<AppliedRowMutation> = None;
            let mut recorded_write_set = false;
            for e in &to_apply {
                commit.sm.apply(e)?;
                if let Some(m) = self.apply_mvcc_entry(e, cat)? {
                    // C2 (write-path assessment): record the SERIALIZED path's write-set into the
                    // SI recent-commits ledger, exactly as the concurrent path records its own —
                    // so a concurrent committer whose read snapshot predates this commit sees the
                    // conflict (first-committer-wins) instead of silently overwriting it (a lost
                    // update). Before this, the ledger was populated only by the concurrent path
                    // and safety rested on the emergent fact that serialized DML is INSERT-shaped
                    // in production routing.
                    commit.ledger.record(m.write_set(), e.index);
                    recorded_write_set = true;
                    applied = Some(m);
                }
                commit.repl.mark_applied(e.index);
            }
            if recorded_write_set {
                // Same bounding policy as the concurrent path's (3e): prune below the oldest
                // active read snapshot (or everything up to this commit when none are active —
                // a future snapshot can never conflict against entries at or below its own seq).
                let prune_boundary = self
                    .active_snapshots
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .oldest()
                    .map(|oldest| oldest.saturating_sub(1))
                    .unwrap_or(publish_index);
                commit.ledger.prune_below(prune_boundary);
            }

            // Publish ordering (Stage 2 — blocker #1; PART B catalog↔data co-pinning). The apply loop
            // published this commit's data generation(s) and mutated the working catalog maps (for any
            // DDL entries). Order the rest so a lock-free reader gets a consistent (catalog, data) pair:
            //   1. residency tombstones, 2. catalog ring push (stamped at `publish_index`), then LAST
            //   3. `committed_seq` release-store.
            // The catalog is pushed BEFORE `committed_seq` (the FLIP from the old order) so a reader
            // that loads `committed_seq = token.index` and selects `catalog_as_of(token.index)` is
            // guaranteed to find this generation — the catalog is visible no later than `committed_seq`.
            // That, with the per-boundary self-consistency of the data (MVCC versions stamp old/new
            // part-counts at the DDL's commit_seq), rules out a reader straddling a shape-changing DDL.
            // Slice 1b-ii-c / SV4b: an INSERT or DELETE commit of EXACTLY ONE applied log entry maintains the
            // resident shard INCREMENTALLY (O(rows touched)) instead of invalidating + re-admitting the whole
            // table (the O(table) dual-store tax): INSERT appends its rows into the open shard's headroom;
            // DELETE locates the deleted rows' resident slots (zone-map-pruned) and stamps `deleted_by` in
            // place. The single-entry guard keeps the residency scope == {that one table} — a multi-entry
            // batch / DDL / update falls back to the conservative invalidate below. Runs BEFORE
            // publish_committed_seq, so a reader that observes the new committed_seq sees the change; append
            // also drops the table's stale GPU index (audit Finding A). On ANY failure (not-int4-resident / no
            // headroom / non-single-row / NULL-or-dup-ambiguous locate / device err) the helper returns false
            // and we invalidate + re-admit (which rebuilds all-live from the host store = always correct).
            // DELETE-tombstoning is gated behind `resident_delete_tombstone_enabled` (default OFF, nested under
            // the shard path) so its A/B lever is independent; OFF => a DELETE re-admits exactly as before.
            let handled = self.auto_admit_on_commit_enabled()
                && to_apply.len() == 1
                && match applied.as_ref() {
                    Some(AppliedRowMutation::Insert {
                        table,
                        rows,
                        row_ids,
                        ..
                    }) => {
                        // D3 (ADR-013 pre1): INSERT appends are STAMPED `created_by = commit_seq`
                        // like every other append — a reader pinned at an older snapshot no longer
                        // sees a decided-but-unpublished insert. RETIREMENT A1: identities ride the
                        // mutation (parsed from the delta's installed keys).
                        self.try_append_resident_int4_open_shard(
                            table,
                            rows,
                            crate::engine_residency::AppendCreatedBy::InsertUniform(publish_index),
                            Some(row_ids),
                        )
                    }
                    Some(AppliedRowMutation::Delete { table, rows, .. })
                        if self.resident_delete_tombstone_enabled() =>
                    {
                        self.try_tombstone_resident_delete_commit(cat, table, rows, publish_index)
                    }
                    Some(AppliedRowMutation::Update {
                        table,
                        old_rows,
                        new_rows,
                        row_ids,
                        ..
                    }) if self.resident_update_tombstone_enabled() => {
                        // RETIREMENT A1/A4b: the appended new versions keep the ORIGINAL rows'
                        // identities — parsed from the installs' keys (exact parallel to
                        // old_rows/new_rows), surfaced on the mutation.
                        self.try_update_resident_commit(
                            cat,
                            table,
                            old_rows,
                            new_rows,
                            publish_index,
                            row_ids.as_deref(),
                        )
                    }
                    _ => false,
                };
            // RETIREMENT A4e: the ELIDED lifecycle. A handled incremental commit on an eligible
            // strictly-Int4 table ENTERS elision (subsequent applies skip the host install); an
            // UNHANDLED commit on an elided table REHYDRATES FIRST (device gather @ C-1 + this
            // commit's delta -> the host store is complete again, sticky de-elision) so the
            // invalidate+re-admit below rebuilds from a truthful store.
            if let Some(applied_ref) = applied.as_ref() {
                let table_name = match applied_ref {
                    AppliedRowMutation::Insert { table, .. }
                    | AppliedRowMutation::Delete { table, .. }
                    | AppliedRowMutation::Update { table, .. } => table.as_str(),
                };
                if handled
                    && self.host_install_elision_enabled()
                    && !self.table_install_elided(table_name)
                {
                    // Audit B1: eligibility = strictly-Int4 AND constraint-free both directions
                    // (the published snapshot is the same catalog `cat` mirrors here).
                    let snapshot = self.catalog_snapshot();
                    if self.table_elision_eligible(&snapshot, table_name) {
                        self.set_table_install_elided(table_name, true);
                    }
                } else if !handled && self.table_install_elided(table_name) {
                    let (upserts, removals) = Self::elided_commit_delta(applied_ref);
                    if let Some(table) = cat.relational_catalog.get(table_name) {
                        self.rehydrate_elided_table(
                            table,
                            publish_index.saturating_sub(1),
                            &upserts,
                            &removals,
                            publish_index,
                        )?;
                    }
                }
            }
            // VACUUM #5 auto-trigger: a handled incremental commit that pushed the table's
            // tombstone churn past the threshold rebuilds it NOW, inside the held commit lock
            // (dead slots bloat every scan and keep the PK index dup-declined; the rebuild is
            // the same invalidate+re-admit a declined commit would do — pre-publish, so its
            // all-live born-visible semantics match the existing re-admit class).
            if handled && self.auto_vacuum_enabled() {
                if let Some(applied_ref) = applied.as_ref() {
                    let table_name = match applied_ref {
                        AppliedRowMutation::Insert { table, .. }
                        | AppliedRowMutation::Delete { table, .. }
                        | AppliedRowMutation::Update { table, .. } => table.as_str(),
                    };
                    let churn = self.tombstone_churn(table_name);
                    if churn >= self.tombstone_churn_threshold(table_name) {
                        // DEFER (deadlock discipline): this arm holds the commit lock AND the
                        // catalog latch; the vacuum's re-admit needs the latch. Park the table —
                        // the execute_text tail vacuums right after both release.
                        *self
                            .read_state
                            .residency
                            .pending_auto_vacuum
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                            Some(table_name.to_string());
                    }
                }
            }
            if !handled {
                self.invalidate_relational_residency_for_commit(&to_apply, txn_id, publish_index);
            }
            let prune_below = self.catalog_prune_boundary(publish_index);
            self.publish_catalog_snapshot(cat, publish_index, prune_below);
            handled
        };
        self.publish_committed_seq(publish_index);
        // STRATA S-B: best-effort GPU-residency admission for the committed mutation's tables (flag-gated,
        // after the publish so it snapshots the new generation; never fails the already-durable commit).
        // Skipped when we maintained residency in place above — that table is already resident + current.
        if self.auto_admit_on_commit_enabled() && !handled {
            if let Some(tables) = Self::residency_invalidation_scope(&to_apply) {
                self.auto_admit_resident_tables(&tables);
            }
        }
        Ok(())
    }

    /// RETIREMENT A4e: an unhandled commit's (upserts, removals) by row identity — the delta the
    /// device could not absorb, applied on top of the rehydration gather. Identities: INSERT/
    /// UPDATE carry `row_ids` on the mutation; DELETE parses the write-set's row keys.
    fn elided_commit_delta(
        applied: &AppliedRowMutation,
    ) -> (
        std::collections::BTreeMap<u64, Vec<SqlValue>>,
        std::collections::BTreeSet<u64>,
    ) {
        let mut upserts = std::collections::BTreeMap::new();
        let mut removals = std::collections::BTreeSet::new();
        match applied {
            AppliedRowMutation::Insert {
                table,
                rows,
                row_ids,
                ..
            } => {
                let _ = table;
                for (row_id, row) in row_ids.iter().zip(rows.iter()) {
                    upserts.insert(*row_id, row.clone());
                }
            }
            AppliedRowMutation::Update {
                new_rows, row_ids, ..
            } => {
                if let Some(ids) = row_ids {
                    for (row_id, row) in ids.iter().zip(new_rows.iter()) {
                        upserts.insert(*row_id, row.clone());
                    }
                }
            }
            AppliedRowMutation::Delete {
                table, write_set, ..
            } => {
                let prefix = relational_key_prefix(table);
                for row in &write_set.rows {
                    if let Some(row_id) =
                        crate::engine_residency::parse_relational_row_id(&row.row_key, &prefix)
                    {
                        removals.insert(row_id);
                    }
                }
            }
        }
        (upserts, removals)
    }

    pub(crate) fn commit_mutation_at_with_current_apply<F>(
        &self,
        txn_id: u64,
        payload: std::sync::Arc<[u8]>,
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
            commit.record_commit_timestamp(txn_id, timestamp_micros);
            token
        };

        let to_apply: Vec<LogEntry> = commit
            .repl
            .drain_committed_from(commit.repl.applied_index())
            .cloned()
            .collect();
        // RETIREMENT A4e (audit B2): a MULTI-ENTRY commit bypasses the single-mutation elision
        // hooks below (they rehydrate exactly ONE mutation's delta) — every entry's apply would
        // skip the host install and the final invalidate+re-admit would rebuild every touched
        // table from its STALE store (elided-era rows lost). Rehydrate every elided table in the
        // batch's scope FIRST (under this commit lock; state through committed_seq is fully on
        // device) — the tables de-elide, the applies install normally, the re-admit is truthful.
        if to_apply.len() > 1 && self.host_install_elision_enabled() {
            if let Some(scope) = Self::residency_invalidation_scope(&to_apply) {
                for table_name in &scope {
                    if self.table_install_elided(table_name) {
                        let Some(table) = self.relational_catalog_table(table_name) else {
                            continue;
                        };
                        let seq = self.committed_seq();
                        self.rehydrate_elided_table(
                            &table,
                            seq,
                            &Default::default(),
                            &Default::default(),
                            seq,
                        )?;
                    }
                }
            }
        }

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
        // Stage 3 — blocker #2: the snapshot/shard flag maps are published behind `ArcSwap`;
        // W0 extracted the copy-on-write flagging into a helper SHARED with the concurrent
        // invalidation (publishers serialize on `descriptor_publish_lock`).
        self.flag_residency_descriptors_invalidated(table, txn_id, index);
        self.read_state.residency.device_memory.invalidate(table);
        self.read_state
            .residency
            .shard_device_memory
            .invalidate_table(table);
        // SV4 prereq #1 (lifecycle): release the shard's on-demand `deleted_by` region alongside the
        // resident buffer it annotates. The re-admit that follows an invalidating commit rebuilds the
        // shard ALL-LIVE from the host store, so a surviving tombstone region would wrongly hide rows
        // (and leak device memory). Mirrors `shard_device_memory` exactly. INERT until SV4 (no region
        // exists in production today), so this leaves the OFF path byte-identical.
        self.read_state
            .residency
            .shard_deleted_by_memory
            .invalidate_table(table);
        // SV6: the `created_by` region lives and dies with the buffer it annotates, exactly like
        // `deleted_by` (a stale region surviving a re-admit would wrongly HIDE rebuilt all-live rows).
        self.read_state
            .residency
            .shard_created_by_memory
            .invalidate_table(table);
        // RETIREMENT A1: the row-identity region follows the buffer it annotates.
        self.read_state
            .residency
            .shard_row_id_memory
            .invalidate_table(table);
        // Sub-slice 3b: drop the table's cached per-shard PK indexes (they pin stale buffers).
        self.read_state
            .residency
            .purge_shard_pk_index_for_table(table);
    }

    /// W0: flag a table's residency DESCRIPTORS (table snapshot + every shard) invalidated at
    /// `(txn_id, index)` — the copy-on-write mutation both invalidation paths share. This is what
    /// the descriptor-trusting consumers observe: the sharded read-route planner + executor
    /// (`plan_relational_sharded_resident_route` reads `shard.is_valid()` and the descriptor's
    /// riding `device_memory`) and the write-locates. Before W0 only the SERIALIZED path set these
    /// flags; the concurrent path tombstoned the CELLS only, which the D4 descriptor-riding
    /// consumers never consult — so a concurrent commit on a shard-resident table left readers
    /// serving STALE device bytes (a read-your-writes violation) and let a duplicate key
    /// FALSE-PASS the write-locate's unique probe. Publishers of the descriptor maps serialize on
    /// `descriptor_publish_lock`; runs BEFORE `committed_seq` publishes, so a reader that pins the
    /// new boundary and THEN loads the maps observes the flags (pin-seq-before-load ordering).
    /// Repro + regression: `w0_concurrent_invalidation_must_not_leave_write_locate_trusting_stale_shards`.
    pub(crate) fn flag_residency_descriptors_invalidated(
        &self,
        table: &str,
        txn_id: TxnId,
        index: Index,
    ) {
        self.read_state
            .residency
            .flag_table_descriptors_invalidated(table, txn_id, index);
    }

    /// Invalidate the GPU residency of the `tables` a CONCURRENT commit mutated, via `&self`
    /// (write-half MVCC, Stage 4). Publishes a `None` tombstone on each table's resident
    /// device-memory cell(s) — the authoritative gate the read-path's `plan_relational_resident_route`
    /// checks (`has_retained_device_memory`), so after this a reader takes the CPU route on the new
    /// committed data rather than a stale GPU snapshot (residency↔data consistency, design Risk #3).
    /// Called INSIDE the commit critical section, before `committed_seq` is bumped, so a reader that
    /// observes the new `committed_seq` also observes the residency tombstone.
    ///
    /// W0: it ALSO flags the `snapshots`/`shards` DESCRIPTOR maps (the pre-W0 form tombstoned only
    /// the cells, believing "the cell tombstone alone forces the CPU route" — true for the
    /// TABLE-level route, but the D4 SHARDED planner/executor and the write-locates read the
    /// descriptor's `is_valid()` + its riding `device_memory` Arc and never consult the cells, so
    /// a concurrent commit left them serving/probing STALE device bytes). The maps' publishers
    /// serialize on `descriptor_publish_lock`, so this is safe from the commit critical section
    /// without the catalog latch.
    pub(crate) fn invalidate_relational_residency_tables_concurrent(
        &self,
        tables: &BTreeSet<String>,
        txn_id: TxnId,
        index: Index,
    ) {
        for table in tables {
            self.flag_residency_descriptors_invalidated(table, txn_id, index);
            self.read_state.residency.device_memory.invalidate(table);
            self.read_state
                .residency
                .shard_device_memory
                .invalidate_table(table);
            // SV4 prereq #1: mirror the deleted_by cleanup on the concurrent commit path (INERT today).
            self.read_state
                .residency
                .shard_deleted_by_memory
                .invalidate_table(table);
            // SV6: mirror the created_by cleanup (same lifecycle contract).
            self.read_state
                .residency
                .shard_created_by_memory
                .invalidate_table(table);
            // RETIREMENT A1: the row-identity region follows the buffer it annotates.
            self.read_state
                .residency
                .shard_row_id_memory
                .invalidate_table(table);
            // Sub-slice 3b: drop the table's cached per-shard PK indexes.
            self.read_state
                .residency
                .purge_shard_pk_index_for_table(table);
        }
        // ADR-009 R2.2b: deliberately does NOT evict the persistent wave read engine here. Dropping it tears
        // a live kernel down (petter join ~watchdog window + stream sync) — far too costly to do inside the
        // commit critical section. It is unnecessary for correctness: this concurrent path tombstones the
        // `device_memory` cell (above; and since W0 also flags the descriptor maps), so the gate that fires is
        // the `device_memory.get(...) -> None -> Err` ("relation has no retained resident device memory") at the
        // TOP of `submit_resident_int4_equal_any_payload`, which returns BEFORE the wave route is reached
        // (NOT the `!snapshot.is_valid()` check — `is_valid()` reads the untouched descriptor). Either way a
        // stale engine is never used; a re-admission allocates a new resident buffer (new ptr) whose first
        // read misses the ptr-keyed cache and rebuilds, dropping the stale engine on that read thread. Net
        // cost of skipping eviction here: a stale kernel holds its SM until the table is re-admitted + read
        // (only when the wave route is enabled — default OFF).
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
        let pressured_shard_tables = self.read_state.residency.with_shards_mut(|shards| {
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
            // SV4 prereq #1: a pressured shard's deleted_by region is released with its buffer (INERT today).
            self.read_state
                .residency
                .shard_deleted_by_memory
                .invalidate_table(table);
            // SV6: a pressured shard's created_by region is released with its buffer too.
            self.read_state
                .residency
                .shard_created_by_memory
                .invalidate_table(table);
            // RETIREMENT A1: the row-identity region follows the buffer it annotates.
            self.read_state
                .residency
                .shard_row_id_memory
                .invalidate_table(table);
            // Sub-slice 3b: drop the pressured table's cached per-shard PK indexes.
            self.read_state
                .residency
                .purge_shard_pk_index_for_table(table);
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
    ) -> Result<Option<AppliedRowMutation>, EngineError> {
        // Returns the APPLIED row mutation for a single INSERT or DELETE entry (the caller maintains GPU
        // residency incrementally for a single-entry commit: Insert -> append in place, Delete -> tombstone
        // in place; Slice 1b-ii-c / SV4b); `None` for every other command (and for a non-UTF-8 /
        // unparseable payload — a defensive no-op as before).
        let Ok(text) = std::str::from_utf8(&entry.payload) else {
            return Ok(None);
        };
        let Ok(cmd) = parse_command(text) else {
            return Ok(None);
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

        let mut applied: Option<AppliedRowMutation> = None;
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
            Command::Insert(insert) => {
                applied = self.apply_insert(cat, insert, commit_seq)?.map(
                    |(table, rows, write_set, row_ids)| AppliedRowMutation::Insert {
                        table,
                        rows,
                        write_set,
                        row_ids,
                    },
                );
            }
            Command::Delete(delete) => {
                applied =
                    self.apply_delete(cat, delete, commit_seq)?
                        .map(|(table, rows, write_set)| AppliedRowMutation::Delete {
                            table,
                            rows,
                            write_set,
                        });
            }
            Command::Update(update) => {
                applied = self.apply_update(cat, update, commit_seq)?.map(
                    |(table, old_rows, new_rows, row_ids, write_set)| AppliedRowMutation::Update {
                        table,
                        old_rows,
                        new_rows,
                        row_ids,
                        write_set,
                    },
                );
            }
            _ => {}
        }

        Ok(applied)
    }
}

#[cfg(test)]
mod commit_timestamp_tests {
    use crate::Engine;

    fn engine_with_commits(n: u64) -> Engine {
        let engine = Engine::new_local();
        engine
            .execute_text(1, "CREATE TABLE t (id INT)")
            .expect("create table");
        for i in 0..n {
            engine
                .execute_text(i + 2, &format!("INSERT INTO t (id) VALUES ({i})"))
                .expect("insert");
        }
        engine
    }

    /// NON-VACUOUS DIFFERENTIAL: the O(1) `max_commit_timestamp_micros` must equal the O(n)
    /// `wal_commit_timestamps_micros.values().max()` it replaced, after a real commit sequence.
    /// This is byte-identity by construction — if `record_commit_timestamp` ever fails to bump the
    /// running max, the two diverge and this fails. The length assert proves commits actually ran
    /// (so the equality is not vacuously over an empty map).
    #[test]
    fn running_max_equals_full_scan_of_map() {
        let engine = engine_with_commits(64);
        let commit = engine.commit_state();
        let scan_max = commit
            .wal_commit_timestamps_micros
            .values()
            .copied()
            .max()
            .unwrap_or(0);
        assert!(
            commit.wal_commit_timestamps_micros.len() >= 64,
            "expected the commit-timestamp map to be populated (got {})",
            commit.wal_commit_timestamps_micros.len()
        );
        assert_ne!(
            scan_max, 0,
            "non-vacuity: the scanned max must be a real timestamp"
        );
        assert_eq!(
            commit.max_commit_timestamp_micros, scan_max,
            "O(1) running max diverged from the O(n) scan it replaced"
        );
    }

    /// The assignment property the O(n) scan guaranteed is preserved: commit timestamps are strictly
    /// increasing. (txn_ids are assigned monotonically here, so `values()` is in commit order.)
    #[test]
    fn assigned_timestamps_are_strictly_monotonic() {
        let engine = engine_with_commits(32);
        let commit = engine.commit_state();
        let stamps: Vec<u64> = commit
            .wal_commit_timestamps_micros
            .values()
            .copied()
            .collect();
        assert!(stamps.len() >= 32, "expected commits to be recorded");
        for pair in stamps.windows(2) {
            assert!(
                pair[1] > pair[0],
                "commit timestamps must be strictly increasing: {} !> {}",
                pair[1],
                pair[0]
            );
        }
    }

    /// Fresh engine (empty map): the running max is 0 and `next_commit_timestamp_micros` returns the
    /// wall clock — reproducing the old `unwrap_or(wall_clock)` arm (wall micros >> 1).
    #[test]
    fn fresh_engine_next_timestamp_is_wall_clock() {
        let engine = Engine::new_local();
        {
            let commit = engine.commit_state();
            assert_eq!(commit.max_commit_timestamp_micros, 0);
            assert!(commit.wal_commit_timestamps_micros.is_empty());
        }
        let ts = engine.next_commit_timestamp_micros();
        assert!(
            ts > 1,
            "fresh-engine timestamp should be the wall clock, got {ts}"
        );
    }
}
