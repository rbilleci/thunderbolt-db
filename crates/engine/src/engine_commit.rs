//! Commit / replication-apply path (P0 §9.6 decomposition, behavior-preserving):
//! a focused `impl Engine` block for the Raft role transitions (become_follower/
//! leader/candidate), the commit oracle (commit_mutation, commit_mutation_at,
//! next_commit_timestamp_micros), the
//! resident-memory invalidation appliers (invalidate_relational_residency and its
//! table/concurrent/for-commit/for-memory-pressure variants + scope), and the
//! committed MVCC log-entry applier (apply_mvcc_entry).

use super::*;

#[path = "engine_commit/empty_root_publication.rs"]
mod empty_root_publication;
#[path = "engine_commit/root_publication.rs"]
mod root_publication;

pub(crate) use root_publication::LiveTypedGenerationRootPublication;
#[path = "engine_commit/index_enrollment.rs"]
mod index_enrollment;
#[path = "engine_commit/reset_root.rs"]
mod reset_root;

/// Failure from [`Engine::commit_mutation_batch`]. `requeue` is true only for a clean transient
/// pre-durable abort; semantic claim failures and post-fsync uncertainty must not poison the queue
/// with an item that can never validly retry.
pub(crate) struct BatchCommitFailure {
    pub(crate) requeue: bool,
    pub(crate) error: EngineError,
}

/// One already-validated typed transaction whose device bytes were applied after WAL/status
/// durability. The shared publisher consumes this only when it proves the exact committed entry
/// identity. Both the live terminal and the strict codec-5 recovery owner use this same final
/// device/publication cut; neither path decodes a host row matrix after generation.
pub(crate) struct LiveTypedTransactionApply {
    pub(crate) txn_id: TxnId,
    pub(crate) expected_index: Index,
    pub(crate) payload_authority: LiveTypedPayloadAuthority,
    pub(crate) payload_len: usize,
    pub(crate) tables: Box<[String]>,
    pub(crate) allocator_high_water: u64,
    /// The generic one-record codec-5 route consumes its exact proposed range immediately after
    /// device apply. Historical claim/lease replay retains the legacy monotonic apply behavior.
    pub(crate) allocator_already_consumed: bool,
    pub(crate) affected_rows: u64,
    pub(crate) write_set: WriteSet,
    /// Final transaction-private sequence states closed by codec-5 S2/S5. These publications
    /// share the row-generation visibility cut. When the authenticated stable OID is currently
    /// bound under a different name, the same publication cut applies the S2-witnessed rename
    /// through the existing catalog mutation owner before installing the scalar state.
    pub(crate) private_sequence_publications: Box<[LiveTypedPrivateSequencePublication]>,
    /// Existing ordered catalog envelope decoded from / written into codec-5 S3. It carries no
    /// row mutation or allocator state and is consumed by the same catalog applier as legacy
    /// ordered transactions at this transaction's single visibility cut.
    pub(crate) catalog_composition: Option<BinaryTransactionRecord>,
    pub(crate) parent_authority: Option<LiveTypedParentAuthority>,
}

pub(crate) struct LiveTypedPrivateSequencePublication {
    pub(crate) name: Box<str>,
    pub(crate) sequence_oid: u32,
    pub(crate) last_value: i64,
    pub(crate) is_called: bool,
}

impl Engine {
    pub(crate) fn apply_codec5_catalog_composition(
        &self,
        cat: &mut DdlCatalogState,
        commit_seq: Index,
        record: &BinaryTransactionRecord,
    ) -> Result<(), EngineError> {
        if record.catalog_commands.is_empty()
            || !record.mutations.is_empty()
            || record.allocator_high_water != 0
            || !record.table_resets.is_empty()
        {
            return Err(EngineError::ApplyFailed(
                "codec-5 S3 catalog composition acquired row, reset, or allocator authority"
                    .to_string(),
            ));
        }
        self.apply_transaction_catalog_envelope(
            cat,
            commit_seq.saturating_sub(1),
            record.catalog_epoch,
            commit_seq,
            &record.catalog_commands,
            crate::engine_transaction_catalog::TransactionCatalogEnvelopeSlices {
                view_operations: &record.view_operations,
                view_lifecycle_operations: &record.view_lifecycle_operations,
                index_lifecycle_operations: &record.index_lifecycle_operations,
                sequence_lifecycle_operations: &record.sequence_lifecycle_operations,
                sequence_reset_operations: &record.sequence_reset_operations,
                operation_order: &record.operation_order,
                catalog_output: record.catalog_output.as_ref(),
                sequence_input_oids: &record.sequence_input_oids,
                sequence_value_references: &record.sequence_value_references,
            },
            true,
        )
    }

    /// Apply codec-5's one optional S3 operation through the existing transaction applier. S2/
    /// S4/S7 remain the sole INSERT carrier; this admits only pre-existing-row UPDATE/DELETE
    /// facts that cannot be folded into those private images. It is deliberately called inside
    /// the same common publication owner as the typed device plan, never as a second terminal.
    pub(crate) fn apply_codec5_operation_composition(
        &self,
        entry: &LogEntry,
        cat: &mut DdlCatalogState,
        record: &BinaryTransactionRecord,
    ) -> Result<Vec<AppliedRowMutation>, EngineError> {
        if record.allocator_high_water != 0 || !record.table_resets.is_empty() {
            return Err(EngineError::ApplyFailed(
                "codec-5 S3 operation composition acquired reset or allocator authority"
                    .to_string(),
            ));
        }
        if record.catalog_commands.is_empty() {
            if record.mutations.is_empty()
                || record.catalog_output.is_some()
                || !record.operation_order.is_empty()
                || !record.statement_digests.is_empty()
                || !record.sequence_input_oids.is_empty()
                || !record.sequence_value_references.is_empty()
            {
                return Err(EngineError::ApplyFailed(
                    "codec-5 row-only S3 composition is not an exact resolved UPDATE/DELETE record"
                        .to_string(),
                ));
            }
        } else if record.catalog_commands.iter().any(|command| {
            !crate::wal_binary::command_is_codec5_catalog_composition(&command.command)
        }) {
            return Err(EngineError::ApplyFailed(
                "codec-5 S3 catalog composition contains an unsupported command".to_string(),
            ));
        }
        if !record.catalog_commands.is_empty() {
            // Preserve the established S3 catalog owner for all historical/current DDL and
            // sequence closure shapes. Only the co-resident pre-existing-row mutations are
            // projected into the existing row-only transaction applier afterwards.
            let mut catalog_only = record.clone();
            catalog_only.mutations.clear();
            self.apply_codec5_catalog_composition(cat, entry.index, &catalog_only)?;
            if record.mutations.is_empty() {
                return Ok(Vec::new());
            }
        }
        let mut row_only = record.clone();
        let mutation_tables = row_only
            .mutations
            .iter()
            .map(|mutation| match mutation {
                BinaryTransactionMutation::Insert { table, .. }
                | BinaryTransactionMutation::Update { table, .. }
                | BinaryTransactionMutation::Delete { table, .. } => table,
            })
            .cloned()
            .collect::<BTreeSet<_>>();
        row_only.catalog_epoch = crate::wal_binary::BinaryTransactionCatalogEpoch::Legacy;
        row_only.catalog_commands.clear();
        row_only.created_table_identities.clear();
        row_only.created_table_index_identities.clear();
        row_only.catalog_output = None;
        row_only.view_operations.clear();
        row_only.view_lifecycle_operations.clear();
        row_only.index_lifecycle_operations.clear();
        row_only.sequence_lifecycle_operations.clear();
        row_only.sequence_reset_operations.clear();
        row_only.sequence_advances_by_oid.clear();
        row_only.operation_order.clear();
        row_only.statement_digests.clear();
        row_only.sequence_input_oids.clear();
        row_only.sequence_value_references.clear();
        row_only.sequence_advances.clear();
        row_only
            .table_identities
            .retain(|table, _| mutation_tables.contains(table));
        self.apply_binary_transaction_record(entry, cat, row_only)
    }

    /// Apply the catalog-name portion of one codec-5 private-sequence publication. Live commit
    /// and fresh recovery preflight share this exact stable-OID binding: recovery applies it only
    /// to a cloned catalog postimage, while the live publication cut applies it to the working
    /// catalog before installing the scalar sequence state.
    pub(crate) fn apply_codec5_private_sequence_name_binding(
        &self,
        cat: &mut DdlCatalogState,
        publication: &LiveTypedPrivateSequencePublication,
    ) -> Result<(), EngineError> {
        if publication.sequence_oid == 0 {
            return Err(EngineError::ApplyFailed(
                "codec-5 private sequence publication has zero stable identity".to_string(),
            ));
        }
        let current_name = cat
            .relational_sequences
            .iter()
            .find_map(|(name, sequence)| {
                (sequence.oid == publication.sequence_oid).then(|| name.clone())
            })
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "codec-5 private sequence publication targets missing stable identity {}",
                    publication.sequence_oid
                ))
            })?;
        if current_name != publication.name.as_ref() {
            self.apply_rename_sequence(
                cat,
                RenameSequence {
                    old_name: current_name,
                    new_name: publication.name.to_string(),
                },
            )?;
        }
        let sequence = cat
            .relational_sequences
            .get(publication.name.as_ref())
            .expect("stable-OID rename established the final sequence name");
        if sequence.oid != publication.sequence_oid {
            return Err(EngineError::ApplyFailed(format!(
                "codec-5 private sequence publication changed stable identity for {:?}",
                publication.name
            )));
        }
        Ok(())
    }
}

pub(crate) enum LiveTypedPayloadAuthority {
    /// The typed exact control plane preserves this Arc through replication and WAL append. Live
    /// apply therefore proves identity without hashing the multi-fragment payload before and after
    /// proposal.
    Exact(std::sync::Arc<[u8]>),
    /// Compatibility and fresh-recovery callers may not share the live allocation and retain the
    /// content digest proof.
    Digest(gpu_db_wal::CanonicalDigest),
}

impl LiveTypedPayloadAuthority {
    fn matches(&self, payload: &std::sync::Arc<[u8]>) -> bool {
        match self {
            Self::Exact(expected) => std::sync::Arc::ptr_eq(expected, payload),
            Self::Digest(expected) => gpu_db_wal::canonical_request_digest(payload) == *expected,
        }
    }
}

pub(crate) struct LiveTypedParentAuthority {
    pub(crate) request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) autocommit: bool,
    pub(crate) typed_statement_digests: Box<[gpu_db_wal::CanonicalDigest]>,
}

/// The ordinary validated witness remains on the legacy/default state-machine path. Only the
/// strict codec-5 semantics-v2 owner may select the prevalidated path, after either live
/// S1--S8 closure or its exact durable replay closure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum LiveTypedApplySemantics {
    #[default]
    Legacy,
    SemanticsV2Codec5,
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
        self.commit_mutation_with_table_access(txn_id, payload, None)
    }

    pub(crate) fn commit_mutation_with_table_access(
        &self,
        txn_id: u64,
        payload: std::sync::Arc<[u8]>,
        table_access: Option<&Arc<TableAccessLease>>,
    ) -> Result<CommitToken, EngineError> {
        if self.transaction_snapshot_handle(txn_id).is_some() {
            return Err(EngineError::ApplyFailed(format!(
                "transaction id {txn_id} is active and cannot be claimed by an autocommit write"
            )));
        }
        Self::reject_discarded_returning_payload(&payload)?;
        self.ensure_commit_path_available()?;
        self.legacy_lane_history_write_guard()?;
        #[cfg(test)]
        self.run_commit_prelock_hook();
        let timestamp_micros = self.next_commit_timestamp_micros();
        self.commit_mutation_at_with_catalog_inner(
            txn_id,
            payload,
            timestamp_micros,
            None,
            table_access,
        )
    }

    fn reject_discarded_returning_payload(payload: &[u8]) -> Result<(), EngineError> {
        if matches!(
            Self::decode_engine_command(payload),
            Ok(Some(command))
                if crate::engine_dml_concurrent::command_has_returning(&command)
        ) {
            return Err(crate::engine_dml_concurrent::discarded_returning_engine_error());
        }
        Ok(())
    }

    /// Raw serialized commits remain the compatibility authority for non-INSERT commands and
    /// historical replay. A newly admitted textual INSERT must never use that generic record
    /// shape: it would skip the move-only typed batch, transaction overlay, device plan, and
    /// codec-5 terminal. Recovery intentionally keeps decoding prior records in
    /// `apply_mvcc_entry`; this guard applies only to new live claims.
    fn reject_live_serialized_insert_payload(payload: &[u8]) -> Result<(), EngineError> {
        let text_insert = matches!(
            Self::decode_engine_command(payload),
            Ok(Some(Command::Insert(_)))
        );
        // The binary INSERT and transaction frames remain decode-only so an old durable prefix
        // can recover.  A live caller may not smuggle either frame through the generic commit
        // API: doing so would recreate the resolved-row encoder, raw apply, and publication
        // authority that codec-5 replaced.
        let binary_insert = if is_binary_wal_record(payload) {
            match decode_binary_record(payload)? {
                BinaryWalRecord::Insert(_) => true,
                BinaryWalRecord::Transaction(record) => record
                    .mutations
                    .iter()
                    .any(|mutation| matches!(mutation, BinaryTransactionMutation::Insert { .. })),
                BinaryWalRecord::DeleteByKey(_)
                | BinaryWalRecord::UpdateByKey(_)
                | BinaryWalRecord::SequenceValueTransition(_) => false,
            }
        } else {
            false
        };
        if text_insert || binary_insert {
            return Err(EngineError::ApplyFailed(
                "raw serialized INSERT must enter typed transaction admission".to_string(),
            ));
        }
        Ok(())
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
        self.commit_mutation_at_with_catalog(txn_id, payload, timestamp_micros, None)
    }

    pub(crate) fn commit_mutation_at_with_catalog(
        &self,
        txn_id: u64,
        payload: std::sync::Arc<[u8]>,
        timestamp_micros: u64,
        expected_catalog_version: Option<Index>,
    ) -> Result<CommitToken, EngineError> {
        self.commit_mutation_at_with_catalog_inner(
            txn_id,
            payload,
            timestamp_micros,
            expected_catalog_version,
            None,
        )
    }

    pub(crate) fn commit_mutation_at_with_catalog_table_access(
        &self,
        txn_id: u64,
        payload: std::sync::Arc<[u8]>,
        timestamp_micros: u64,
        expected_catalog_version: Option<Index>,
        table_access: Option<&Arc<TableAccessLease>>,
    ) -> Result<CommitToken, EngineError> {
        self.commit_mutation_at_with_catalog_inner(
            txn_id,
            payload,
            timestamp_micros,
            expected_catalog_version,
            table_access,
        )
    }

    fn commit_mutation_at_with_catalog_inner(
        &self,
        txn_id: u64,
        payload: std::sync::Arc<[u8]>,
        timestamp_micros: u64,
        expected_catalog_version: Option<Index>,
        table_access: Option<&Arc<TableAccessLease>>,
    ) -> Result<CommitToken, EngineError> {
        if self.transaction_snapshot_handle(txn_id).is_some() {
            return Err(EngineError::ApplyFailed(format!(
                "transaction id {txn_id} is active and cannot be claimed by an autocommit write"
            )));
        }
        Self::reject_discarded_returning_payload(&payload)?;
        Self::reject_live_serialized_insert_payload(&payload)?;
        self.ensure_commit_path_available()?;
        if self.repl_role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        // Durable-commit critical section under the **commit_mutex** (A.4 unification — this method is
        // now `&self`, the SERIALIZED commit path for DDL / KV / replay,
        // reached through `&Engine`; the concurrent autocommit-DML path is `commit_dml_concurrent`,
        // which runs the SAME WAL-before-publish sequence under this same lock). Lock order is fixed:
        // `commit_state()` (commit_mutex) FIRST, then `ddl_catalog()` (catalog latch) acquired INSIDE.
        // `next_commit_timestamp_micros` (which itself locks the commit_mutex) is computed by the
        // caller BEFORE this, so we never re-enter the non-reentrant commit_mutex.
        // Serialized relational DML can run beside classic commit waves (for example an UPDATE
        // falling back from an elided device table). Its exact canonical terminal outcome and
        // subsequent apply must bind after every earlier wave's durability publication and
        // residency maintenance. Wait off-lock, then prove the same condition again while owning
        // the publication lock so a newly applied wave cannot occupy the acquisition gap.
        let mut commit = self.commit_state_after_wave_quiescence()?;
        self.ensure_commit_path_available()?;
        self.legacy_lane_history_write_guard()?;
        if let Some(token) = commit.resolve_transaction_retry(txn_id, &payload)? {
            return Ok(token);
        }
        // Raw compatibility callers can still reach this low-level claimant without the typed
        // admission facade. Retain a shared stable-OID guard through apply/publication so a
        // transactional table reset cannot replace the same root concurrently. This raw API
        // exposes only `EngineError`; product-facing paths acquire the guard earlier and preserve
        // the retryable `ExecuteError::Serialization`/40001 classification.
        let _raw_table_access = table_access
            .is_none()
            .then(|| self.acquire_raw_mutation_table_access(&payload))
            .transpose()?;
        if let Some(expected) = expected_catalog_version {
            let actual = self.catalog_snapshot().commit_seq;
            if expected != actual {
                return Err(EngineError::ApplyFailed(format!(
                    "prepared command catalog changed before execution (expected generation {expected}, current generation {actual}); re-Parse is required"
                )));
            }
        }
        if self.resolve_pending_transaction_claim(
            txn_id,
            gpu_db_wal::canonical_request_digest(&payload),
        )? {
            return Err(EngineError::Durability(format!(
                "transaction id {txn_id} is pending in canonical mutation admission"
            )));
        }
        if let Some(state) = commit.txn_manager.state(txn_id) {
            return Err(EngineError::ApplyFailed(format!(
                "transaction id {txn_id} is already owned by transaction state {state:?}"
            )));
        }
        self.preflight_serialized_dml_under_commit_lock(&payload)?;
        let token = {
            let wal_len_before = commit.wal.len();
            let commit_seq = commit.repl.peek_next_index();
            // The exact terminal outcome re-prepares relational DML at the proposed commit
            // boundary. We already own the commit mutex here, so mark that preparation as an
            // internal under-lock read: an elided-table device decline may rehydrate, and the
            // lock-aware rehydration seam must not try to acquire this non-reentrant mutex again.
            let (outcome_kind, affected_rows) =
                self.skip_leader_check_during_internal_read(|engine| {
                    engine.canonical_serialized_outcome(&payload, commit_seq)
                })?;
            let token = match commit.repl.propose(payload.clone()) {
                Ok(token) => token,
                Err(err) => {
                    return Err(err);
                }
            };
            let record = match Self::canonical_wal_record_with_commit_outcome(
                &mut commit,
                txn_id,
                token.index,
                0,
                &payload,
                gpu_db_wal::canonical_request_digest(&payload),
                outcome_kind,
                affected_rows,
            ) {
                Ok(record) => record,
                Err(err) => {
                    commit.repl.rollback_unapplied_from(token.index);
                    return Err(err);
                }
            };
            commit.wal.append_canonical(record);
            if let Err(err) = commit.wal.flush_all() {
                commit.repl.rollback_unapplied_from(token.index);
                commit.wal.truncate(wal_len_before);
                return Err(err);
            }
            if let Err(error) = commit.repl.wait_committed(token, Duration::from_millis(0)) {
                self.wedge_commit_path();
                return Err(EngineError::Durability(format!(
                    "transaction {txn_id} WAL is durable but replication confirmation failed: {error}; outcome is indeterminate until restart recovery"
                )));
            }
            if let Err(error) = commit.record_transaction_status_digest_outcome(
                txn_id,
                gpu_db_wal::canonical_request_digest(&payload),
                token.index,
                affected_rows,
            ) {
                self.wedge_commit_path();
                return Err(EngineError::Durability(format!(
                    "transaction {txn_id} WAL is durable but terminal status installation failed: {error}; outcome is indeterminate until restart recovery"
                )));
            }
            // `txn_id` (the façade `next_txn_id`) is the durable transaction *identity* — recorded in
            // the WAL record and keyed here for PITR lookups. Intentionally DECOUPLED from the MVCC
            // version stamp, which uses the commit `Index` (see `apply_mvcc_entry`).
            commit.record_commit_timestamp(txn_id, timestamp_micros);
            token
        };

        if let Err(error) = self.apply_and_publish_committed(&mut commit, txn_id, token.index) {
            self.wedge_commit_path();
            return Err(error);
        }
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
    /// Failure semantics: `requeue` is set only when the whole group cleanly rolled back before
    /// durability and the failure is transient. Semantic transaction-identity rejection is clean
    /// but non-retryable; post-fsync failures are indeterminate and fail-stop service.
    pub(crate) fn commit_mutation_batch(
        &self,
        items: &[(TxnId, std::sync::Arc<[u8]>)],
    ) -> Result<(), BatchCommitFailure> {
        if items.is_empty() {
            return Ok(());
        }
        for (_, payload) in items {
            Self::reject_discarded_returning_payload(payload).map_err(|error| {
                BatchCommitFailure {
                    requeue: false,
                    error,
                }
            })?;
            Self::reject_live_serialized_insert_payload(payload).map_err(|error| {
                BatchCommitFailure {
                    requeue: false,
                    error,
                }
            })?;
        }
        if let Err(error) = self.ensure_commit_path_available() {
            return Err(BatchCommitFailure {
                requeue: true,
                error,
            });
        }
        if let Err(error) = self.legacy_lane_history_write_guard() {
            return Err(BatchCommitFailure {
                requeue: false,
                error,
            });
        }
        if self.repl_role() != Role::Leader {
            return Err(BatchCommitFailure {
                requeue: true,
                error: EngineError::NotLeader,
            });
        }
        let wall_clock = current_timestamp_micros();

        let mut commit = match self.commit_state_after_wave_quiescence() {
            Ok(commit) => commit,
            Err(error) => {
                return Err(BatchCommitFailure {
                    requeue: true,
                    error,
                });
            }
        };
        if let Err(error) = self.ensure_commit_path_available() {
            return Err(BatchCommitFailure {
                requeue: true,
                error,
            });
        }
        if let Err(error) = self.legacy_lane_history_write_guard() {
            return Err(BatchCommitFailure {
                requeue: false,
                error,
            });
        }
        // Resolve the global transaction authority before any proposal. Exact retries (whether
        // from an earlier commit or duplicated inside this flush group) are already complete and
        // are omitted; a stable-id mismatch rejects the whole still-clean group.
        let mut batch_claims = std::collections::HashMap::new();
        let mut admitted = Vec::with_capacity(items.len());
        for (txn_id, payload) in items {
            let request_digest = gpu_db_wal::canonical_request_digest(payload);
            match commit.resolve_transaction_retry_digest_outcome(*txn_id, request_digest) {
                Ok(Some((token, _))) if self.committed_seq() >= token.index => {
                    self.release_pending_transaction_claim(*txn_id, request_digest);
                    continue;
                }
                Ok(Some((token, _))) => {
                    return Err(BatchCommitFailure {
                        requeue: false,
                        error: EngineError::Durability(format!(
                            "transaction id {txn_id} has canonical commit sequence {} but publication has not reached it",
                            token.index
                        )),
                    });
                }
                Err(error) => {
                    return Err(BatchCommitFailure {
                        requeue: false,
                        error,
                    });
                }
                Ok(None) => {
                    match self.resolve_pending_transaction_claim(*txn_id, request_digest) {
                        Ok(_) => {}
                        Err(error) => {
                            return Err(BatchCommitFailure {
                                requeue: false,
                                error,
                            });
                        }
                    }
                    if let Some(state) = commit.txn_manager.state(*txn_id) {
                        return Err(BatchCommitFailure {
                            requeue: false,
                            error: EngineError::ApplyFailed(format!(
                                "transaction id {txn_id} is already owned by transaction state {state:?}"
                            )),
                        });
                    }
                    match batch_claims.entry(*txn_id) {
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(request_digest);
                            admitted.push((*txn_id, std::sync::Arc::clone(payload)));
                        }
                        std::collections::hash_map::Entry::Occupied(entry)
                            if *entry.get() == request_digest => {}
                        std::collections::hash_map::Entry::Occupied(_) => {
                            return Err(BatchCommitFailure {
                                requeue: false,
                                error: EngineError::Durability(format!(
                                    "transaction id {txn_id} appears more than once in one batch with different requests"
                                )),
                            });
                        }
                    }
                }
            }
        }
        let Some((last_txn_id, _)) = admitted.last() else {
            return Ok(());
        };
        let last_txn_id = *last_txn_id;
        // Queue items already own these guards, but the crate-private batch claimant also has
        // direct recovery/compatibility tests. Re-derive the complete set from admitted payloads
        // so no caller can bypass the table-reset boundary; exact canonical retries were removed
        // above and need no new lease.
        let _table_access = admitted
            .iter()
            .map(|(_, payload)| self.acquire_raw_mutation_table_access(payload))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| BatchCommitFailure {
                requeue: false,
                error,
            })?;
        let exact_outcomes: Vec<u64> = admitted
            .iter()
            .map(|(_, payload)| Self::canonical_affected_rows(payload))
            .collect::<Result<_, _>>()
            .map_err(|error| BatchCommitFailure {
                requeue: false,
                error,
            })?;
        let last_index = {
            let wal_len_before = commit.wal.len();
            let mut first_index: Option<Index> = None;
            let mut last_index = 0;
            for (txn_id, payload) in &admitted {
                match commit.repl.propose(payload.clone()) {
                    Ok(token) => {
                        first_index.get_or_insert(token.index);
                        last_index = token.index;
                        match Self::canonical_wal_record(
                            &mut commit,
                            *txn_id,
                            token.index,
                            0,
                            payload,
                        ) {
                            Ok(record) => commit.wal.append_canonical(record),
                            Err(error) => {
                                commit.repl.rollback_unapplied_from(
                                    first_index.expect("the current proposal established it"),
                                );
                                commit.wal.truncate(wal_len_before);
                                return Err(BatchCommitFailure {
                                    requeue: true,
                                    error,
                                });
                            }
                        }
                    }
                    Err(error) => {
                        if let Some(first) = first_index {
                            commit.repl.rollback_unapplied_from(first);
                        }
                        commit.wal.truncate(wal_len_before);
                        return Err(BatchCommitFailure {
                            requeue: true,
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
                    requeue: true,
                    error,
                });
            }
            if let Err(error) = commit
                .repl
                .wait_committed(CommitToken { index: last_index }, Duration::from_millis(0))
            {
                // The records are already fsync-durable; surface the replication failure without
                // pretending the batch can be cleanly retried.
                self.wedge_commit_path();
                return Err(BatchCommitFailure {
                    requeue: false,
                    error,
                });
            }
            // Per-item strictly-monotonic commit timestamps (same formula as
            // `next_commit_timestamp_micros`, inlined because the commit_mutex is already held).
            let batch_first_index = first_index.expect("non-empty batch proposed");
            for (offset, ((txn_id, payload), affected_rows)) in
                admitted.iter().zip(exact_outcomes).enumerate()
            {
                let timestamp_micros =
                    wall_clock.max(commit.max_commit_timestamp_micros.saturating_add(1));
                commit.record_commit_timestamp(*txn_id, timestamp_micros);
                if let Err(error) = commit.record_transaction_status_digest_outcome(
                    *txn_id,
                    gpu_db_wal::canonical_request_digest(payload),
                    batch_first_index + offset as u64,
                    affected_rows,
                ) {
                    self.wedge_commit_path();
                    return Err(BatchCommitFailure {
                        requeue: false,
                        error,
                    });
                }
                self.release_pending_transaction_claim(
                    *txn_id,
                    gpu_db_wal::canonical_request_digest(payload),
                );
            }
            last_index
        };

        if let Err(error) = self.apply_and_publish_committed(&mut commit, last_txn_id, last_index) {
            self.wedge_commit_path();
            return Err(BatchCommitFailure {
                requeue: false,
                error,
            });
        }
        for _ in &admitted {
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
    pub(crate) fn apply_and_publish_committed(
        &self,
        commit: &mut CommitState,
        txn_id: TxnId,
        publish_index: Index,
    ) -> Result<(), EngineError> {
        self.skip_leader_check_during_internal_read(|engine| {
            engine.apply_and_publish_committed_inner(
                commit,
                txn_id,
                publish_index,
                None,
                LiveTypedApplySemantics::default(),
                None,
            )
        })
    }

    /// Apply one already-durable WRITE-001 claim/allocator control record. These records are
    /// canonical authorities, but are deliberately not relational operations and therefore never
    /// enter the legacy `GPUDBOP1` decoder or transaction-status terminal map.
    pub(crate) fn apply_and_publish_write_authority_control(
        &self,
        commit: &mut CommitState,
        expected_txn_id: TxnId,
        publish_index: Index,
    ) -> Result<(), EngineError> {
        let mut to_apply = commit
            .repl
            .drain_committed_from(commit.repl.applied_index());
        let entry = to_apply.next().cloned().ok_or_else(|| {
            EngineError::ApplyFailed(
                "write-authority control apply found no committed entry".to_string(),
            )
        })?;
        if to_apply.next().is_some()
            || entry.index != publish_index
            || entry.index == 0
            || expected_txn_id == 0
        {
            return Err(EngineError::ApplyFailed(
                "write-authority control apply does not own one exact committed entry".to_string(),
            ));
        }
        drop(to_apply);
        let envelope =
            gpu_db_wal::decode_canonical_record_payload(&entry.payload)?.ok_or_else(|| {
                EngineError::ApplyFailed(
                    "write-authority control entry is not canonical WAL".to_string(),
                )
            })?;
        if envelope.header.stable_transaction_id != expected_txn_id
            || envelope.header.commit_seq != publish_index
        {
            return Err(EngineError::ApplyFailed(
                "write-authority control entry identity differs from its apply witness".to_string(),
            ));
        }
        let authority = crate::engine_write_authority::decode_write_authority_envelope(&envelope)?
            .ok_or_else(|| {
                EngineError::ApplyFailed(
                    "canonical entry is not a strict WRITE-001 control record".to_string(),
                )
            })?;
        self.apply_and_publish_decoded_write_authority_control(commit, publish_index, authority)
    }

    fn apply_and_publish_decoded_write_authority_control(
        &self,
        commit: &mut CommitState,
        publish_index: Index,
        authority: crate::engine_write_authority::DecodedWriteAuthority,
    ) -> Result<(), EngineError> {
        let allocator_high_water = commit.write_authority.apply(
            publish_index,
            authority,
            self.read_state.mvcc.current_row_id(),
        )?;
        if let Some(high_water) = allocator_high_water {
            self.read_state.mvcc.advance_row_id_to_at_least(high_water);
        }
        commit.last_applied_outcome = Some((publish_index, 0));
        commit.repl.mark_applied(publish_index);
        let visible = self.publish_ready_indices(std::iter::once(publish_index))?;
        self.require_publication_coverage(visible, publish_index)?;
        commit.ledger.mark_published_through(publish_index);
        Ok(())
    }

    /// Consume the already-prevalidated codec-5 semantics-v2 entry after its device generation
    /// is complete. The strict retained decoder is the only recovery authority allowed to reach
    /// the corresponding recovered entry point below.
    pub(crate) fn apply_and_publish_committed_with_live_semantics_v2_codec5_transaction(
        &self,
        commit: &mut CommitState,
        txn_id: TxnId,
        publish_index: Index,
        live: LiveTypedTransactionApply,
        root_publication: Option<LiveTypedGenerationRootPublication>,
    ) -> Result<(), EngineError> {
        self.skip_leader_check_during_internal_read(|engine| {
            engine.apply_and_publish_committed_inner(
                commit,
                txn_id,
                publish_index,
                Some(live),
                LiveTypedApplySemantics::SemanticsV2Codec5,
                root_publication,
            )
        })
    }

    /// Publish a codec-5 entry whose exact durable aggregate was strictly closed, rebound to
    /// the recovered catalog/allocator authorities, and applied by the same generic device plan
    /// used by the live terminal. This accepts no raw WAL body and cannot be selected by legacy
    /// recovery, so it is not a decoder-free compatibility escape hatch.
    pub(crate) fn apply_and_publish_committed_with_recovered_semantics_v2_codec5_transaction(
        &self,
        commit: &mut CommitState,
        txn_id: TxnId,
        publish_index: Index,
        recovered: LiveTypedTransactionApply,
        root_publication: Option<LiveTypedGenerationRootPublication>,
    ) -> Result<(), EngineError> {
        self.skip_leader_check_during_internal_read(|engine| {
            engine.apply_and_publish_committed_inner(
                commit,
                txn_id,
                publish_index,
                Some(recovered),
                LiveTypedApplySemantics::SemanticsV2Codec5,
                root_publication,
            )
        })
    }

    fn apply_and_publish_committed_inner(
        &self,
        commit: &mut CommitState,
        txn_id: TxnId,
        publish_index: Index,
        live_typed: Option<LiveTypedTransactionApply>,
        live_typed_semantics: LiveTypedApplySemantics,
        mut root_publication: Option<LiveTypedGenerationRootPublication>,
    ) -> Result<(), EngineError> {
        if live_typed_semantics == LiveTypedApplySemantics::SemanticsV2Codec5
            && live_typed.is_none()
        {
            return Err(EngineError::ApplyFailed(
                "semantics-v2 codec-5 live apply requires an exact live witness".to_string(),
            ));
        }
        if root_publication.is_some()
            && live_typed.is_some()
            && live_typed_semantics != LiveTypedApplySemantics::SemanticsV2Codec5
        {
            return Err(EngineError::ApplyFailed(
                "typed generation root publication requires semantics-v2 codec-5 live apply"
                    .to_string(),
            ));
        }
        let to_apply: Vec<LogEntry> = commit
            .repl
            .drain_committed_from(commit.repl.applied_index())
            .cloned()
            .collect();
        if let Some(live) = live_typed.as_ref() {
            let matches_exact_entry = to_apply.len() == 1
                && live.txn_id == txn_id
                && live.expected_index == publish_index
                && to_apply[0].index == live.expected_index
                && to_apply[0].payload.len() == live.payload_len
                && live.payload_authority.matches(&to_apply[0].payload);
            if !matches_exact_entry {
                return Err(EngineError::ApplyFailed(
                    "live typed transaction witness does not match the durable canonical entry"
                        .to_string(),
                ));
            }
        }
        // Coalesce an insert-only durable batch into one device append per table. Every other DML
        // entry is published at its exact durable boundary below so a following repair-consuming
        // DDL observes it. No batch is de-authorized to a host tuple store.
        let keep_device_authoritative: BTreeSet<String> = if to_apply.len() > 1 {
            self.batch_insert_only_device_authoritative_tables(&to_apply)
                .into_iter()
                .filter(|table| self.table_device_authoritative(table))
                .collect()
        } else {
            BTreeSet::new()
        };
        // Hold the catalog latch across the WHOLE apply loop AND the catalog publish (PART B), so a
        // DDL's working-map mutation + the published-snapshot push are atomic w.r.t. another DDL. Lock
        // order is fixed: commit_mutex (held by the caller) FIRST, then this latch.
        // ADR-006 (multi-statement elision): a keep-elided table's INSERTs accumulate here across ALL
        // entries of the batch — `rows` in seq-scan order, `row_ids` their host identities, `seqs`
        // each row's birth commit index (entries span multiple indices, so the append stamps PER ROW
        // via `AppendCreatedBy::InsertPerRow`, exactly like the concurrent wave flush). One incremental
        // device append per table after the loop keeps it elided.
        #[derive(Default)]
        struct InsertAccum {
            rows: Vec<Vec<SqlValue>>,
            row_ids: Vec<u64>,
            seqs: Vec<Index>,
        }
        let (handled, maintained) = {
            let mut catalog_guard = self.ddl_catalog();
            let cat = &mut *catalog_guard;
            let atomic_transaction = live_typed.is_some()
                || crate::engine_transaction_reset::is_single_binary_transaction(&to_apply);
            let mut applied: Vec<AppliedRowMutation> = Vec::new();
            let mut recorded_write_set = false;
            let mut live_typed_consumed = false;
            let mut insert_batch: BTreeMap<String, InsertAccum> = BTreeMap::new();
            let mut maintained: BTreeSet<String> = BTreeSet::new();
            // A codec-5 CREATE-with-rowset plan has already installed its first resident shard
            // before this common catalog cut.  Remember only tables that the exact captured root
            // predecessor proves absent, so the existing named-index enrollment can publish the
            // corresponding device directory after S3 makes the final catalog identity visible.
            let mut typed_created_index_lifecycle_tables = BTreeSet::new();
            // The root holder remains the sole GPU root/publication authority.  This is only the
            // exact physical-residency enrollment it requires before the first indexed append.
            let mut write001_empty_index_lifecycle_table: Option<String> = None;
            let mut working_catalog_changed = false;
            for e in &to_apply {
                // Codec-5 retains any admitted ordered catalog envelope as S3 and applies it
                // through the existing catalog owner at the same visibility cut as its rows.
                // Live commit never decodes its just-written bytes; recovery supplies the same
                // move-only decoded record after strict aggregate closure.
                let mutates_working_catalog = if live_typed.is_none() {
                    Self::entry_mutates_working_catalog(e, cat)
                } else {
                    false
                };
                let prior_table_identities =
                    mutates_working_catalog.then(|| Self::table_root_identities(cat));
                if live_typed.is_none() {
                    crate::engine_transaction_reset::validate_binary_table_reset_source_roots_payload(
                        &e.payload,
                        &commit.ledger,
                    )?;
                }
                // RETIRE-002 repair boundary: DML no longer maintains a host tuple-store shadow.
                // A rare DDL/recovery operator that still consumes that repair representation must
                // reconstruct it explicitly from the current device generation immediately before
                // the DDL, at the preceding durable boundary. This is not write-path authority.
                if live_typed.is_none() && Self::entry_requires_relational_repair(e) {
                    let boundary = e.index.saturating_sub(1);
                    let class_tables = self
                        .read_state
                        .residency
                        .chunk_authoritative_tables
                        .load()
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>();
                    for table_name in class_tables {
                        self.deauthoritize_chunk_table(&table_name, true)?;
                    }
                    let elided = self
                        .read_state
                        .residency
                        .device_authoritative_tables
                        .load()
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>();
                    for table_name in elided {
                        let Some(table) = cat.relational_catalog.get(&table_name) else {
                            continue;
                        };
                        self.rehydrate_elided_table(
                            table,
                            boundary,
                            &Default::default(),
                            &Default::default(),
                            boundary,
                        )?;
                    }
                }
                let working_catalog = working_catalog_changed
                    .then(|| Self::catalog_snapshot_from_working(cat, e.index.saturating_sub(1)));
                #[cfg(feature = "probe-timing")]
                let transaction_record_apply_started =
                    atomic_transaction.then(std::time::Instant::now);
                let applied_entry = if let Some(live) = live_typed.as_ref() {
                    if !live_typed_consumed {
                        match live_typed_semantics {
                            LiveTypedApplySemantics::Legacy => commit.sm.apply(e)?,
                            LiveTypedApplySemantics::SemanticsV2Codec5 => {
                                // Historical bit-30 codec-5 replay carries the retired
                                // claim/lease parent witness. The generic one-record arm has no
                                // such parent by construction and must not recreate one here.
                                if let Some(parent) = live.parent_authority.as_ref() {
                                    commit.write_authority.mark_parent_terminal(
                                        txn_id,
                                        parent.request_digest,
                                        e.index,
                                        parent.autocommit,
                                        &parent.typed_statement_digests,
                                    )?;
                                }
                                commit.sm.record_prevalidated_semantics_v2_codec5(e)?
                            }
                        }
                        let mut prior_sequence_oid = None;
                        let composed_mutations =
                            if let Some(catalog_composition) = live.catalog_composition.as_ref() {
                                let mutations = self.apply_codec5_operation_composition(
                                    e,
                                    cat,
                                    catalog_composition,
                                )?;
                                working_catalog_changed |=
                                    !catalog_composition.catalog_commands.is_empty();
                                mutations
                            } else {
                                Vec::new()
                            };
                        for publication in live.private_sequence_publications.iter() {
                            if publication.sequence_oid == 0
                                || prior_sequence_oid
                                    .is_some_and(|prior| prior >= publication.sequence_oid)
                            {
                                return Err(EngineError::ApplyFailed(
                                    "codec-5 private sequence publications are not in unique stable-OID order"
                                        .to_string(),
                                ));
                            }
                            self.apply_codec5_private_sequence_name_binding(cat, publication)?;
                            let sequence = cat
                                .relational_sequences
                                .get_mut(publication.name.as_ref())
                                .expect("stable-OID rename established the final sequence name");
                            if sequence.oid != publication.sequence_oid {
                                return Err(EngineError::ApplyFailed(format!(
                                    "codec-5 private sequence publication changed stable identity for {:?}",
                                    publication.name
                                )));
                            }
                            sequence.last_value = publication.last_value;
                            sequence.is_called = publication.is_called;
                            prior_sequence_oid = Some(publication.sequence_oid);
                            working_catalog_changed = true;
                        }
                        if !live.allocator_already_consumed {
                            self.read_state
                                .mvcc
                                .advance_row_id_to_at_least(live.allocator_high_water);
                        }
                        // The device plan has already appended the exact committed row images
                        // under its post-WAL permit. Establish the same authority telemetry and
                        // table flag as the typed-wave terminal before the visibility cut.
                        for table in live.tables.iter() {
                            if self.table_chunk_authoritative(table).is_some() {
                                self.read_state
                                    .residency
                                    .chunk_class_device_commits
                                    .fetch_add(1, AtomicOrdering::Relaxed);
                            } else {
                                if !self.table_device_authoritative(table) {
                                    // S3 may have just introduced this relation in `cat`, before
                                    // the catalog-ring publication below.  Eligibility must see
                                    // that exact working postimage so the already-applied first
                                    // device shard becomes the read authority at the same cut.
                                    let snapshot =
                                        Self::catalog_snapshot_from_working(cat, e.index);
                                    if self.table_device_authority_eligible(&snapshot, table) {
                                        self.set_table_device_authoritative(table, true);
                                    }
                                }
                                self.read_state
                                    .residency
                                    .device_authoritative_commits
                                    .fetch_add(1, AtomicOrdering::Relaxed);
                            }
                            let catalog_table = cat.relational_catalog.get(table).ok_or_else(|| {
                                EngineError::ApplyFailed(format!(
                                    "codec-5 live typed table \"{table}\" is absent after catalog composition"
                                ))
                            })?;
                            if !catalog_table.indexes.is_empty()
                                && root_publication.as_ref().is_some_and(|publication| {
                                    publication.introduced_table_from_absent_predecessor(
                                        catalog_table.stable_table_id,
                                    )
                                })
                            {
                                typed_created_index_lifecycle_tables.insert(table.clone());
                            }
                        }
                        live_typed_consumed = true;
                        composed_mutations
                    } else {
                        return Err(EngineError::ApplyFailed(
                            "live typed transaction witness attempted to consume multiple entries"
                                .to_string(),
                        ));
                    }
                } else {
                    self.with_apply_catalog(working_catalog, || {
                        commit.sm.apply(e)?;
                        self.apply_mvcc_entry(e, cat)
                    })?
                };
                if live_typed.is_none() && to_apply.len() == 1 && root_publication.is_none() {
                    root_publication =
                        self.prepare_empty_typed_create_root_publication(commit, e, cat)?;
                    if root_publication.is_none() {
                        root_publication = self
                            .prepare_empty_typed_index_enrollment_root_publication(
                                commit, e, cat,
                            )?;
                    }
                    if root_publication.is_none() {
                        root_publication =
                            self.prepare_empty_typed_reset_root_publication(commit, e, cat)?;
                    }
                    if root_publication.is_none() {
                        root_publication = self
                            .prepare_populated_typed_catalog_rewrite_root_publication(
                                commit, e, cat,
                            )?;
                    }
                }
                if live_typed.is_none() {
                    match Self::decode_engine_command(&e.payload)? {
                        Some(Command::CreateIndex(create))
                            if cat.relational_catalog.get(&create.table).is_some_and(
                                crate::engine_residency::write001_empty_foldable_index_enrollment,
                            ) =>
                        {
                            write001_empty_index_lifecycle_table = Some(create.table);
                        }
                        Some(Command::AddPrimaryKey(add))
                            if cat.relational_catalog.get(&add.table).is_some_and(
                                crate::engine_residency::write001_empty_foldable_index_enrollment,
                            ) =>
                        {
                            write001_empty_index_lifecycle_table = Some(add.table);
                        }
                        Some(Command::AddUniqueConstraint(add))
                            if cat.relational_catalog.get(&add.table).is_some_and(
                                crate::engine_residency::write001_empty_foldable_index_enrollment,
                            ) =>
                        {
                            write001_empty_index_lifecycle_table = Some(add.table);
                        }
                        Some(Command::CreateTable(create))
                            if cat.relational_catalog.get(&create.table).is_some_and(|table| {
                                !table.indexes.is_empty()
                                    && crate::engine_residency::write001_empty_foldable_index_enrollment(table)
                            }) =>
                        {
                            // Inline indexed CREATE has no earlier unindexed commit from which to
                            // inherit the real zero-row resident generation. Materialize that exact
                            // working-catalog generation now; the common lifecycle owner below then
                            // replaces it with the one-slot indexed predecessor used by first append.
                            let working = Self::catalog_snapshot_from_working(cat, e.index);
                            self.with_apply_catalog(Some(working), || {
                                self.populate_relational_residency_snapshot_inner_with_boundary(
                                    cat,
                                    &create.table,
                                    self.planner.default_gpu_id(),
                                    Some(e.index),
                                )
                                .map(|_| ())
                                .map_err(|error| {
                                    EngineError::ApplyFailed(format!(
                                        "inline indexed CREATE could not publish its zero-row GPU generation: {error}"
                                    ))
                                })
                            })?;
                            write001_empty_index_lifecycle_table = Some(create.table);
                        }
                        _ => {}
                    }
                }
                #[cfg(feature = "probe-timing")]
                if let Some(started) = transaction_record_apply_started {
                    self.record_insert_probe_transaction_canonical_record_apply_nanos(
                        started.elapsed().as_nanos() as u64,
                    );
                }
                if live_typed.is_none() && Self::entry_requires_relational_repair(e) {
                    self.rebuild_device_generations_from_repair(cat, e.index)?;
                }
                // A mixed DDL/DML durable batch must publish each DML entry to the device before
                // the following repair-consuming DDL reverse-gathers. Pure insert batches retain
                // their coalesced append below; every other multi-entry DML is maintained now.
                if to_apply.len() > 1
                    && !atomic_transaction
                    && !applied_entry.is_empty()
                    && applied_entry.iter().all(|mutation| {
                        !keep_device_authoritative.contains(&Self::applied_mutation_table(mutation))
                    })
                {
                    let touched = applied_entry
                        .iter()
                        .map(Self::applied_mutation_table)
                        .collect::<BTreeSet<_>>();
                    for table in &touched {
                        maintained.remove(table);
                    }
                    let working = Self::catalog_snapshot_from_working(cat, e.index);
                    let entry_maintained = self.with_apply_catalog(Some(working), || {
                        self.try_maintain_transaction_residency(
                            cat,
                            &applied_entry,
                            e.index,
                            &BTreeSet::new(),
                        )
                    })?;
                    if let Some(table) = touched.difference(&entry_maintained).next() {
                        self.wedge_commit_path();
                        return Err(EngineError::ApplyFailed(format!(
                            "durable DML for relation \"{table}\" could not publish its device generation at {}",
                            e.index
                        )));
                    }
                    maintained.extend(touched);
                }
                let affected_rows = if let Some(live) = live_typed.as_ref() {
                    live.affected_rows
                } else if atomic_transaction {
                    // Pre-coalescing v1 transaction WAL counted statement-order mutation records
                    // in its durable outcome marker. Canonical apply normalizes those records to
                    // final entity images, but retry/recovery status must remain byte-compatible
                    // with the marker carried by the original payload.
                    Self::canonical_affected_rows(&e.payload)?
                } else {
                    applied_entry
                        .iter()
                        .map(AppliedRowMutation::rows_affected)
                        .sum()
                };
                commit.last_applied_outcome = Some((e.index, affected_rows));
                working_catalog_changed |= mutates_working_catalog;
                if let Some(live) = live_typed.as_ref() {
                    commit.ledger.record(&live.write_set, e.index);
                    recorded_write_set = true;
                }
                for m in applied_entry {
                    // C2 (write-path assessment): record the SERIALIZED path's write-set into the
                    // SI recent-commits ledger, exactly as the concurrent path records its own —
                    // so a concurrent committer whose read snapshot predates this commit sees the
                    // conflict (first-committer-wins) instead of silently overwriting it (a lost
                    // update). Before this, the ledger was populated only by the concurrent path
                    // and safety rested on the emergent fact that serialized DML is INSERT-shaped
                    // in production routing.
                    commit.ledger.record(m.write_set(), e.index);
                    recorded_write_set = true;
                    if let AppliedRowMutation::Insert {
                        table,
                        rows,
                        row_ids,
                        ..
                    } = &m
                    {
                        if keep_device_authoritative.contains(table) {
                            let acc = insert_batch.entry(table.clone()).or_default();
                            acc.rows.extend(rows.iter().cloned());
                            acc.row_ids.extend(row_ids.iter().copied());
                            acc.seqs.extend(std::iter::repeat_n(e.index, rows.len()));
                        }
                    }
                    applied.push(m);
                }
                if let Some(prior_table_identities) = prior_table_identities.as_ref() {
                    // Catalog/DML groups are replayed one durable record at a time. Reconcile at
                    // this exact transition as well, so create/rename/drop/recreate batches build
                    // the same stable-OID root ledger regardless of apply grouping.
                    self.reconcile_table_root_ledger(
                        &mut commit.ledger,
                        prior_table_identities,
                        cat,
                    );
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
            // R3-004: an ordinary relational commit maintains its resident generation before the
            // visibility cut. INSERT appends into open-shard headroom; UPDATE/DELETE stamps exact stable
            // identities and UPDATE appends its new version. A device decline is handled below by wedging
            // before acknowledgement; only DDL/recovery repair reaches conservative invalidation.
            let handled = if live_typed.is_some() {
                // The exact witness proved a typed INSERT transaction whose catalog effect, if
                // present, is only the S2/S5-closed private sequence final state applied above.
                // Its move-only device
                // plan has already published the sole resident append, so decoding the just
                // durable transaction only to rediscover empty lifecycle sets would rebuild
                // host rows and duplicate physicalization. Recovery never has this witness and
                // therefore continues through the generic decoder below.
                if !applied.is_empty() {
                    let final_catalog = Self::catalog_snapshot_from_working(cat, publish_index);
                    let row_maintained = self.with_apply_catalog(Some(final_catalog), || {
                        self.try_maintain_transaction_residency(
                            cat,
                            &applied,
                            publish_index,
                            &BTreeSet::new(),
                        )
                    })?;
                    let touched = applied
                        .iter()
                        .map(Self::applied_mutation_table)
                        .collect::<BTreeSet<_>>();
                    if row_maintained != touched {
                        return Err(EngineError::ApplyFailed(
                            "codec-5 S3 pre-existing-row mutations did not publish every GPU generation"
                                .to_string(),
                        ));
                    }
                    maintained.extend(row_maintained);
                }
                if !typed_created_index_lifecycle_tables.is_empty() {
                    let enrolled = self.maintain_transaction_index_lifecycle_residency(
                        cat,
                        &typed_created_index_lifecycle_tables,
                        publish_index,
                    )?;
                    if enrolled != typed_created_index_lifecycle_tables {
                        return Err(EngineError::ApplyFailed(
                            "codec-5 CREATE-with-rowset did not enroll every initial named-index generation"
                                .to_string(),
                        ));
                    }
                    maintained.extend(enrolled);
                }
                true
            } else if !atomic_transaction
                && applied.is_empty()
                && to_apply.len() == 1
                && Self::entry_is_relational_dml(&to_apply[0])
            {
                // A zero-row DELETE/UPDATE (including its typed WAL replay form) changes no
                // relational bytes. Preserve the current device generation: invalidating it would
                // manufacture a missing-authority gap before the next replayed/device-prepared DML.
                // Allocator-only effects, such as a zero-row UPDATE's burned reservation, were
                // already applied above and do not require a generation rewrite.
                true
            } else if atomic_transaction {
                let (transaction_created_tables, index_lifecycle_tables) =
                    match decode_binary_record(&to_apply[0].payload) {
                        Ok(crate::wal_binary::BinaryWalRecord::Transaction(record)) => {
                            let created = record
                                .catalog_commands
                                .into_iter()
                                .filter_map(|operation| match operation.command {
                                    Command::CreateTable(create) => Some(create.table),
                                    _ => None,
                                })
                                .collect();
                            let index_tables = record
                                .index_lifecycle_operations
                                .into_iter()
                                .flat_map(|operation| operation.targets)
                                .filter_map(|target| target.owner_name)
                                .collect();
                            (created, index_tables)
                        }
                        _ => (BTreeSet::new(), BTreeSet::new()),
                    };
                let final_catalog = Self::catalog_snapshot_from_working(cat, publish_index);
                #[cfg(feature = "probe-timing")]
                let transaction_residency_publish_started = std::time::Instant::now();
                let (row_maintained, index_maintained) =
                    self.with_apply_catalog(Some(final_catalog), || {
                        let row_maintained = self.try_maintain_transaction_residency(
                            cat,
                            &applied,
                            publish_index,
                            &transaction_created_tables,
                        )?;
                        let index_maintained = self
                            .maintain_transaction_index_lifecycle_residency(
                                cat,
                                &index_lifecycle_tables,
                                publish_index,
                            )?;
                        Ok::<_, EngineError>((row_maintained, index_maintained))
                    })?;
                #[cfg(feature = "probe-timing")]
                self.record_insert_probe_transaction_canonical_residency_publish_nanos(
                    transaction_residency_publish_started.elapsed().as_nanos() as u64,
                );
                maintained = row_maintained;
                maintained.extend(index_maintained);
                let touched = applied
                    .iter()
                    .map(Self::applied_mutation_table)
                    .chain(transaction_created_tables)
                    .chain(index_lifecycle_tables)
                    .collect::<BTreeSet<_>>();
                maintained == touched
            } else {
                to_apply.len() == 1
                    && match applied.last() {
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
                                crate::engine_residency::AppendCreatedBy::InsertUniform(
                                    publish_index,
                                ),
                                Some(row_ids),
                            )
                        }
                        Some(AppliedRowMutation::Delete {
                            table,
                            rows,
                            write_set,
                            ..
                        }) => {
                            let prefix = relational_key_prefix(table);
                            let row_ids = write_set
                                .rows
                                .iter()
                                .filter_map(|key| {
                                    crate::engine_residency::parse_relational_row_id(
                                        &key.row_key,
                                        &prefix,
                                    )
                                })
                                .collect::<Vec<_>>();
                            self.try_tombstone_transaction_rows_by_identity(
                                cat,
                                table,
                                rows,
                                &row_ids,
                                publish_index,
                            )
                        }
                        Some(AppliedRowMutation::Update {
                            table,
                            old_rows,
                            new_rows,
                            row_ids: Some(row_ids),
                            ..
                        }) => {
                            if old_rows.is_empty() {
                                new_rows.is_empty()
                            } else {
                                self.try_tombstone_transaction_rows_by_identity(
                                    cat,
                                    table,
                                    old_rows,
                                    row_ids,
                                    publish_index,
                                ) && self.try_append_resident_int4_open_shard(
                                    table,
                                    new_rows,
                                    crate::engine_residency::AppendCreatedBy::UpdateNewVersion(
                                        publish_index,
                                    ),
                                    Some(row_ids),
                                )
                            }
                        }
                        _ => false,
                    }
            };
            // A handled incremental commit confirms and advances the authoritative device
            // generation. A DML decline below is fatal to this live apply and is never dispatched
            // to a host tuple-store repair path.
            if !atomic_transaction {
                if let Some(applied_ref) = applied.last() {
                    let table_name = match applied_ref {
                        AppliedRowMutation::TableReset { reset, .. } => reset.table.as_str(),
                        AppliedRowMutation::Insert { table, .. }
                        | AppliedRowMutation::Delete { table, .. }
                        | AppliedRowMutation::Update { table, .. } => table.as_str(),
                    };
                    // A zero-row UPDATE/DELETE is byte-unchanged. It reports handled so an already
                    // authoritative generation stays current, but cannot establish authority for a table
                    // whose residency was not confirmed by an actual append or tombstone.
                    let applied_changed_rows = match applied_ref {
                        AppliedRowMutation::TableReset { .. } => true,
                        AppliedRowMutation::Insert { rows, .. } => !rows.is_empty(),
                        AppliedRowMutation::Delete { rows, .. } => !rows.is_empty(),
                        AppliedRowMutation::Update { old_rows, .. } => !old_rows.is_empty(),
                    };
                    if handled
                        && applied_changed_rows
                        && !self.table_device_authoritative(table_name)
                    {
                        // Audit B1: eligibility = device-authoritative types AND FK-free both directions
                        // (CHECK is row-local and no longer blocks — ADR-006; the published snapshot is
                        // the same catalog `cat` mirrors here).
                        let snapshot = self.catalog_snapshot();
                        if self.table_device_authority_eligible(&snapshot, table_name) {
                            self.set_table_device_authoritative(table_name, true);
                        }
                    } else if !handled
                        && self.table_device_authoritative(table_name)
                        && !keep_device_authoritative.contains(table_name)
                        && !maintained.contains(table_name)
                    {
                        self.wedge_commit_path();
                        return Err(EngineError::ApplyFailed(format!(
                            "durable DML for relation \"{table_name}\" could not publish its device generation at {publish_index}"
                        )));
                    }
                    // P4-2b (S-E.P4): the CHUNK-AUTHORITATIVE lifecycle — strictly the elision arm's
                    // ELSE (mutual exclusion, design review H1). A class INSERT materializes as a
                    // TAIL APPEND from the statement's own rows (the store install was skipped; the
                    // WAL is durability, this is the representation). A failed append — or any
                    // DELETE/UPDATE that somehow reached apply with the flag still set (the prepare
                    // guard de-authoritizes first; this is the backstop) — exits the class LOUDLY.
                    // Class ENTRY happens after the cold maintenance below (the entry must be fresh).
                    if !self.table_device_authoritative(table_name)
                        && self.table_chunk_authoritative(table_name).is_some()
                        && !maintained.contains(table_name)
                    {
                        match applied_ref {
                            AppliedRowMutation::TableReset { .. } => {
                                unreachable!("table resets are atomic-transaction-only")
                            }
                            AppliedRowMutation::Insert { rows, row_ids, .. }
                                if !rows.is_empty() =>
                            {
                                let appended = cat
                                    .relational_catalog
                                    .get(table_name)
                                    .map(|table| {
                                        self.append_streaming_cold_tail(
                                            table,
                                            rows,
                                            row_ids,
                                            publish_index,
                                        )
                                    })
                                    .unwrap_or(false);
                                if !appended {
                                    self.deauthoritize_chunk_table(table_name, true)?;
                                }
                            }
                            AppliedRowMutation::Insert { .. } => {}
                            // P4-2b-ii: class DELETE — stamp the packed coordinates iff the
                            // installed entry still carries the prepare-time epoch (the P4-2a
                            // coordinate token); any mismatch/failure exits the class LOUDLY.
                            AppliedRowMutation::Delete {
                                rows, class_stamp, ..
                            } => match class_stamp {
                                Some((coords, epoch)) if !rows.is_empty() => {
                                    if !self.stamp_class_coordinates(
                                        table_name,
                                        coords,
                                        *epoch,
                                        publish_index,
                                    ) {
                                        // ⚠️ AUDIT CONTRACT (MEDIUM, latent): this fallback DROPS
                                        // the in-flight delete from the LIVE store (the apply
                                        // skipped it; the de-auth replay lacks its stamps — WAL
                                        // replay recovers it). UNREACHABLE today: single-entry
                                        // class DML runs prepare→apply→hook under ONE held commit
                                        // mutex, so the epoch cannot drift. The OFF-LOCK-PREPARE
                                        // future (see the write-path plan) MUST replace this arm
                                        // with re-resolve+stamp under the lock — never a drop.
                                        eprintln!(
                                            "[gpu-db] class stamp REFUSED for \"{table_name}\" \
                                         (epoch drift) — de-authoritizing; the live store \
                                         DROPS this delete until WAL replay (latent-unreachable \
                                         path, see the P4-2b-ii audit contract)"
                                        );
                                        self.deauthoritize_chunk_table(table_name, true)?;
                                    }
                                }
                                Some(_) => {} // a 0-row class delete: nothing to stamp
                                // Resolved via a non-class arm while classed (a de-auth raced the
                                // prepare): the store now holds the truth — exit.
                                None => {
                                    self.deauthoritize_chunk_table(table_name, true)?;
                                }
                            },
                            // P4-2b-ii: class UPDATE = stamp the OLD coordinates + tail-append the
                            // NEW images at this commit (the U2 tombstone-old/append-new shape).
                            AppliedRowMutation::Update {
                                new_rows,
                                row_ids,
                                class_stamp,
                                ..
                            } => match (class_stamp, row_ids) {
                                (Some((coords, epoch)), Some(entity_ids))
                                    if !new_rows.is_empty() =>
                                {
                                    let stamped = self.stamp_class_coordinates(
                                        table_name,
                                        coords,
                                        *epoch,
                                        publish_index,
                                    );
                                    let appended = stamped
                                        && cat
                                            .relational_catalog
                                            .get(table_name)
                                            .map(|table| {
                                                self.append_streaming_cold_tail(
                                                    table,
                                                    new_rows,
                                                    entity_ids,
                                                    publish_index,
                                                )
                                            })
                                            .unwrap_or(false);
                                    if !appended {
                                        self.deauthoritize_chunk_table(table_name, true)?;
                                    }
                                }
                                (Some(_), _) => {} // 0-row class update: nothing to do
                                _ => {
                                    self.deauthoritize_chunk_table(table_name, true)?;
                                }
                            },
                        }
                    }
                }
            }
            // Drain accumulated multi-statement INSERTs with one stamped device append per table.
            // A decline wedges before acknowledgement; no host reconstruction or re-admission path
            // is eligible for these durable rows.
            for (table_name, acc) in insert_batch {
                if acc.rows.is_empty() {
                    continue;
                }
                debug_assert!(
                    self.table_device_authoritative(&table_name),
                    "device-authoritative insert batches retain authority through the apply loop"
                );
                let appended = self.try_append_resident_int4_open_shard(
                    &table_name,
                    &acc.rows,
                    crate::engine_residency::AppendCreatedBy::InsertPerRow(&acc.seqs),
                    Some(&acc.row_ids),
                );
                if appended {
                    maintained.insert(table_name);
                } else {
                    self.wedge_commit_path();
                    return Err(EngineError::ApplyFailed(format!(
                        "durable insert batch for relation \"{table_name}\" could not publish its device generation at {publish_index}"
                    )));
                }
            }
            // VACUUM #5 auto-trigger: a handled incremental commit that pushed tombstone churn
            // past the threshold schedules the explicit RETIRE-002 dense-repair boundary.
            if handled && self.auto_vacuum_enabled() {
                if let Some(applied_ref) = applied.last() {
                    let table_name = match applied_ref {
                        AppliedRowMutation::TableReset { reset, .. } => reset.table.as_str(),
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
            if let Some(table_name) = write001_empty_index_lifecycle_table.take() {
                let enrolled = self.maintain_transaction_index_lifecycle_residency(
                    cat,
                    &BTreeSet::from([table_name.clone()]),
                    publish_index,
                )?;
                if !enrolled.contains(&table_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "typed indexed DDL did not enroll its WRITE-001 empty physical predecessor for relation \"{table_name}\""
                    )));
                }
                maintained.extend(enrolled);
            }
            if !handled {
                // Only representation-changing DDL/repair entries may remain unhandled here. Never
                // invalidate a table whose device generation was maintained by this batch.
                self.invalidate_relational_residency_for_commit_except(
                    &to_apply,
                    &maintained,
                    txn_id,
                    publish_index,
                );
            }
            let prune_below = self.catalog_prune_boundary(publish_index);
            self.publish_catalog_snapshot(cat, publish_index, prune_below);
            (handled, maintained)
        };
        // The device plan is already complete and the exact entry/witness was validated above.
        // Install its type-neutral root successor before the contiguous visibility join; any
        // predecessor drift is post-durable and therefore fails closed for restart recovery.
        if let Some(root_publication) = root_publication {
            root_publication.install_if_current(&self.read_state.typed_generation_roots)?;
        }
        let visible = self.publish_ready_indices(to_apply.iter().map(|entry| entry.index))?;
        self.require_publication_coverage(visible, publish_index)?;
        commit.ledger.mark_published_through(publish_index);
        // STRATA read-cache policy remains independently configurable. R3-004 establishes mandatory
        // device write generations in DML preflight; this post-publish refresh is only the broader
        // read-residency policy for unhandled DDL/global invalidation.
        if !handled {
            let precise_scope = Self::residency_invalidation_scope(&to_apply);
            let mandatory_refresh = precise_scope.clone().unwrap_or_else(|| {
                let mut resident: BTreeSet<String> = self
                    .read_state
                    .residency
                    .snapshots
                    .load()
                    .keys()
                    .cloned()
                    .collect();
                resident.extend(self.read_state.residency.shards.load().keys().cloned());
                resident
            });
            let tables = if self.auto_admit_on_commit_enabled() {
                precise_scope.unwrap_or_else(|| {
                    self.catalog_snapshot()
                        .relational_catalog
                        .keys()
                        .cloned()
                        .collect()
                })
            } else {
                mandatory_refresh
            };
            // Device-maintained tables are already current. Admission here is restricted to the
            // non-authoritative bootstrap/repair scope.
            let admit: BTreeSet<String> = tables.difference(&maintained).cloned().collect();
            self.auto_admit_resident_tables(&admit);
        }
        // R3-004: no commit-time cold-store patch or class entry. Normal relational DML publishes
        // only its device generation; RETIRE-002 owns any future device-native cold repair/import.
        Ok(())
    }

    /// Run an effect-free validation at the same serialized current-state boundary used by a
    /// direct-current commit, without claiming a sequence, replication slot, WAL record, or
    /// publication. Empty COPY uses this to make its exact-target decision linearizable with DDL.
    pub(crate) fn validate_effect_free_at_current_commit_boundary<V>(
        &self,
        txn_id: u64,
        mut validate_current: V,
    ) -> Result<(), EngineError>
    where
        V: FnMut(&Self, Index) -> Result<(), EngineError>,
    {
        if self.transaction_snapshot_handle(txn_id).is_some() {
            return Err(EngineError::ApplyFailed(format!(
                "transaction id {txn_id} is active and cannot use the autocommit validation boundary"
            )));
        }
        self.legacy_lane_history_write_guard()?;
        self.ensure_commit_path_available()?;
        if self.repl_role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        let commit = self.commit_state_after_wave_quiescence()?;
        self.legacy_lane_history_write_guard()?;
        self.ensure_commit_path_available()?;
        if let Some(state) = commit.txn_manager.state(txn_id) {
            return Err(EngineError::ApplyFailed(format!(
                "transaction id {txn_id} is already owned by transaction state {state:?}"
            )));
        }
        let prospective_commit_seq = commit.repl.peek_next_index();
        self.skip_leader_check_during_internal_read(|engine| {
            validate_current(engine, prospective_commit_seq)
        })
    }

    /// Global (stop-the-world) residency invalidation: invalidate every resident
    /// table. Retained as the **conservative fallback** for commit batches whose
    /// mutated tables cannot be determined precisely ([`Engine::residency_invalidation_scope`]
    /// returns `None`). Equivalent to invalidating each resident table individually.
    pub(crate) fn invalidate_relational_residency(&self, txn_id: TxnId, index: Index) {
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
    pub(crate) fn invalidate_relational_residency_table(
        &self,
        table: &str,
        txn_id: TxnId,
        index: Index,
    ) {
        // Stage 3 — blocker #2: the snapshot/shard flag maps are published behind `ArcSwap`;
        // W0 extracted the copy-on-write flagging into a helper SHARED with the concurrent
        // invalidation (publishers serialize on `descriptor_publish_lock`).
        self.flag_residency_descriptors_invalidated(table, txn_id, index);
        self.read_state.residency.device_memory.invalidate(table);
        self.read_state
            .residency
            .shard_device_memory
            .invalidate_table(table);
        // Release the shard's on-demand `deleted_by` region alongside the resident buffer it
        // annotates. The explicit repair rebuild publishes fresh version sidecars; retaining an old
        // region would hide unrelated rows and leak device memory.
        self.read_state
            .residency
            .shard_deleted_by_memory
            .invalidate_table(table);
        // The `created_by` region lives and dies with the buffer it annotates.
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

    /// Publish invalid descriptors for an explicit repair/vacuum/test transition. Ordinary DML
    /// maintains device authority in place and must never call this as a fallback. Callers hold the
    /// commit boundary until a replacement generation is rebuilt, so no acknowledged relational
    /// state is exposed through a missing-device window.
    ///
    /// W0: it ALSO flags the `snapshots`/`shards` DESCRIPTOR maps (the pre-W0 form tombstoned only
    /// the cells, but the D4 SHARDED planner/executor and the write-locates read the
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

    /// Apply one committed log entry to the engine state (`&self`). The caller holds the **catalog
    /// latch** and passes `&mut DdlCatalogState` so a DDL entry's working-map mutation can be made
    /// atomic with the subsequent catalog-snapshot publish (the caller holds the SAME guard across both
    /// — PART B). Lock order is fixed: the caller already holds the commit_mutex, then the catalog
    /// latch. The `apply_*` methods (now `&self`) freely call `self.read_state.*` and the `&self`
    /// `preflight_*` tree (which reads the published catalog snapshot, never this latch — no reentry).
    /// W5a — apply a BINARY insert record (replay + serialized apply): decode → reconstruct the
    /// [`WriteDelta`] → install via [`Engine::apply_delta`], the SAME installer the runtime wave
    /// used. No parse, no coercion, no validation re-run: the record exists only because the
    /// original commit validated it, and it carries the ORIGINAL row ids (replay does not
    /// re-derive them from the allocator; the allocator advances by `rows_consumed` to stay in
    /// lock-step for interleaved text records).
    fn apply_binary_wal_entry(
        &self,
        entry: &LogEntry,
        cat: &mut DdlCatalogState,
    ) -> Result<Option<AppliedRowMutation>, EngineError> {
        let record = match decode_binary_record(&entry.payload)? {
            crate::wal_binary::BinaryWalRecord::Insert(record) => record,
            crate::wal_binary::BinaryWalRecord::DeleteByKey(record) => {
                // U1 (W5b): replay a by-key DELETE through the EXACT text-arm apply path — a
                // synthesized `Command::Delete` with the single pk-equality filter. Replay
                // re-resolves the key against the replayed state; determinism holds because all
                // ops on a key were lane-serialized in this same seq order live.
                let delete = Delete {
                    table: record.table.clone(),
                    filter: None,
                    // The binder consumes `filters`/`filter_groups` (`filter` is legacy).
                    filters: vec![gpu_db_sql::SelectFilter {
                        column: record.pk_column.clone(),
                        op: gpu_db_sql::SelectFilterOp::Eq,
                        value: SqlValue::Int4(record.pk_value),
                    }],
                    filter_groups: Vec::new(),
                    returning: Vec::new(),
                };
                let applied = self.apply_delete(cat, delete, entry.index)?;
                // WAL-FIRST: a W5b record is claimed + fenced BEFORE the visible target is
                // located (the locate moved to apply), so a 0-row delete DOES reach the WAL. At
                // replay it re-resolves to the SAME 0 rows deterministically (all ops on a key
                // are lane-serialized in seq order) — a legal no-op, not corruption. `None` =
                // 0 rows applied; the delete simply affected nothing.
                let Some((table, rows, write_set, _class_stamp)) = applied else {
                    return Ok(None);
                };
                return Ok(Some(AppliedRowMutation::Delete {
                    table,
                    rows,
                    write_set,
                    class_stamp: None,
                }));
            }
            crate::wal_binary::BinaryWalRecord::UpdateByKey(record) => {
                // U2/R3 replay — re-resolve the key against the replayed state (deterministic: all
                // ops on a key are lane-serialized in this same seq order), tombstone the visible
                // old, and install the replacement with that old version's stable entity identity.
                // The v1 record's `new_row_id` is retained as an allocator reservation for backward
                // WAL/high-water compatibility only. A 0-row update still consumes it, so replay
                // advances the allocator by one in both branches.
                let table = cat
                    .relational_catalog
                    .get(&record.table)
                    .ok_or_else(|| {
                        EngineError::Durability(format!(
                            "binary WAL update record targets unknown relation \"{}\"",
                            record.table
                        ))
                    })?
                    .clone();
                let new_values = decode_relational_row(&record.new_row_encoded, &table.columns)
                    .map_err(|err| {
                        EngineError::Durability(format!(
                            "binary WAL update record image decode failed for \"{}\": {err}",
                            record.table
                        ))
                    })?;
                if record.new_row_id == 0 {
                    return Err(EngineError::Durability(format!(
                        "binary WAL update record for \"{}\" carries an unpatched allocator reservation",
                        record.table
                    )));
                }
                // Do not assert `current_row_id() == new_row_id`: merged concurrent preparation
                // may reserve row identities in a different order than canonical commit ranges.
                // Replay walks records in commit order, so its allocator position need not equal
                // a particular record's pre-reserved id.
                // Correctness does NOT need it: the update reuses its resolved stable entity id and
                // advances the legacy allocator by exactly 1 (both branches), so the final high-water
                // remains base + (row-consuming records) regardless of order. An assert here would
                // manufacture a spurious, unrecoverable `Durability` failure under ordinary
                // concurrent update/insert traffic.
                // Tombstone the visible old version by key (the delete arm's re-resolve, verbatim).
                let delete = Delete {
                    table: record.table.clone(),
                    filter: None,
                    filters: vec![gpu_db_sql::SelectFilter {
                        column: record.pk_column.clone(),
                        op: gpu_db_sql::SelectFilterOp::Eq,
                        value: SqlValue::Int4(record.pk_value),
                    }],
                    filter_groups: Vec::new(),
                    returning: Vec::new(),
                };
                // Capture the old version's stable entity identity from the delete's write-set.
                // Replay reuses it for the replacement version, migrating pre-ADR-014 lane WAL
                // away from fresh per-update identities without changing the v1 record framing.
                let (old_rows, entity_ids) = match self.apply_delete(cat, delete, entry.index)? {
                    Some((_, rows, del_write_set, _class_stamp)) => {
                        let prefix = relational_key_prefix(&record.table);
                        let ids: Vec<u64> = del_write_set
                            .rows
                            .iter()
                            .filter_map(|r| {
                                crate::engine_residency::parse_relational_row_id(
                                    &r.row_key, &prefix,
                                )
                            })
                            .collect();
                        (rows, ids)
                    }
                    None => (Vec::new(), Vec::new()),
                };
                if old_rows.is_empty() {
                    // 0-row update: a durable no-op that still burned the claimed `new_row_id`.
                    // Advance the allocator by 1 to keep replay in lock-step with the live pump.
                    self.read_state.mvcc.advance_row_id(1)?;
                    return Ok(None);
                }
                if old_rows.len() != 1 || entity_ids.len() != 1 {
                    return Err(EngineError::Durability(format!(
                        "binary WAL update for \"{}\" resolved one row but not one stable entity identity",
                        record.table
                    )));
                }
                // The encoded `new_row_id` remains a consumed v1 allocator reservation only. The
                // replacement version keeps the old row's stable entity identity.
                let entity_id = entity_ids[0];
                let row_key = relational_row_key(&record.table, entity_id);
                let mut write_set = WriteSet::default();
                write_set.add_table(&table);
                write_set.add_row(&table, entity_id, row_key.clone());
                write_set.add_unique_slots(&table, &new_values);
                let delta = WriteDelta {
                    write_set: write_set.clone(),
                    read_snapshot: entry.index,
                    catalog_dependencies: BTreeMap::from([(record.table.clone(), table.clone())]),
                    foreign_key_dependencies: BTreeSet::new(),
                    rows_consumed: 1,
                    mutation: PreparedMutation::Insert {
                        table: record.table.clone(),
                        inserted_rows: vec![(row_key, new_values.clone())],
                        seq_advances: BTreeMap::new(),
                    },
                };
                self.apply_delta(delta, entry.index, None)?;
                return Ok(Some(AppliedRowMutation::Update {
                    class_stamp: None, // replay: never a class table (process-local flag)
                    table: record.table,
                    old_rows,
                    new_rows: vec![new_values],
                    row_ids: Some(vec![entity_id]),
                    write_set,
                }));
            }
            crate::wal_binary::BinaryWalRecord::Transaction(_) => {
                return Err(EngineError::Durability(
                    "transaction WAL record reached the single-mutation applier".to_string(),
                ));
            }
            crate::wal_binary::BinaryWalRecord::SequenceValueTransition(_) => {
                return Err(EngineError::Durability(
                    "sequence transition reached the single-row mutation applier".to_string(),
                ));
            }
        };
        let table = cat
            .relational_catalog
            .get(&record.table)
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "binary WAL record targets unknown relation \"{}\"",
                    record.table
                ))
            })?
            .clone();
        let commit_seq = entry.index;
        let mut write_set = WriteSet::default();
        write_set.add_table(&table);
        let mut inserted_rows = Vec::with_capacity(record.rows.len());
        let mut rows = Vec::with_capacity(record.rows.len());
        let mut row_ids = Vec::with_capacity(record.rows.len());
        for (row_id, encoded) in &record.rows {
            let values = decode_relational_row(encoded, &table.columns).map_err(|err| {
                EngineError::Durability(format!(
                    "binary WAL record row decode failed for \"{}\": {err}",
                    record.table
                ))
            })?;
            let row_key = relational_row_key(&record.table, *row_id);
            // Parity with prepare_insert (audit 21eddaa7 B): freshly-inserted rows are
            // DELIBERATELY not conflict points (a predicted key would falsely conflict), so the
            // replayed write_set matches text replay exactly — unique slots only.
            write_set.add_unique_slots(&table, &values);
            inserted_rows.push((row_key, values.clone()));
            rows.push(values);
            row_ids.push(*row_id);
        }
        let delta = WriteDelta {
            write_set: write_set.clone(),
            read_snapshot: commit_seq,
            catalog_dependencies: BTreeMap::from([(record.table.clone(), table.clone())]),
            foreign_key_dependencies: BTreeSet::new(),
            rows_consumed: record.rows.len() as u64,
            mutation: PreparedMutation::Insert {
                table: record.table.clone(),
                inserted_rows,
                seq_advances: BTreeMap::new(),
            },
        };
        self.apply_delta(delta, commit_seq, None)?;
        Ok(Some(AppliedRowMutation::Insert {
            table: record.table,
            rows,
            write_set,
            row_ids,
        }))
    }

    fn apply_mvcc_entry(
        &self,
        entry: &LogEntry,
        cat: &mut DdlCatalogState,
    ) -> Result<Vec<AppliedRowMutation>, EngineError> {
        // Returns the APPLIED row mutation for a single INSERT or DELETE entry (the caller maintains GPU
        // residency incrementally for a single-entry commit: Insert -> append in place, Delete -> tombstone
        // in place; Slice 1b-ii-c / SV4b); `None` for every other command (and for a non-UTF-8 /
        // unparseable payload — a defensive no-op as before).
        //
        // W5a: BINARY records dispatch FIRST — their 0xFF tag deliberately fails the UTF-8 check
        // below, and the defensive no-op arm would otherwise SILENTLY SKIP an acknowledged
        // insert's apply (data loss at replay). A tagged record that fails to decode is loud.
        if is_binary_wal_record(&entry.payload) {
            return match decode_binary_record(&entry.payload)? {
                BinaryWalRecord::Transaction(record) => {
                    self.apply_binary_transaction_record(entry, cat, record)
                }
                BinaryWalRecord::SequenceValueTransition(record) => {
                    self.apply_sequence_value_transition_record(entry, cat, record)?;
                    Ok(Vec::new())
                }
                _ => self
                    .apply_binary_wal_entry(entry, cat)
                    .map(|mutation| mutation.into_iter().collect()),
            };
        }
        let Some(cmd) = Self::decode_engine_command(&entry.payload)? else {
            return Ok(Vec::new());
        };
        let command_uses_current_semantics =
            Self::engine_command_uses_current_index_semantics(&entry.payload);
        let current_index_semantics = command_uses_current_semantics || cat.index_oid_epoch_current;
        let historical_command_replay = !command_uses_current_semantics;
        if current_index_semantics {
            // The codec/typed-command epoch is the durable one-way migration boundary, not the
            // command family. A current CREATE VIEW/SEQUENCE/etc. must lift the recovery-only
            // index range before it can consume the next shared `pg_class` OID. Once lifted,
            // byte-stable index-neutral legacy records use the current shared namespace too.
            cat.finalize_legacy_index_oid_migration()?;
        }
        let _index_semantics = self.enter_apply_index_semantics(current_index_semantics);
        // Recovery and direct committed-entry apply bypass the user-facing preflight. Establish
        // the exact target/related device generations while the caller's commit+catalog locks are
        // already held, before any DML resolver or constraint probe runs.
        self.ensure_dml_device_generation_with_catalog(&cmd, cat)?;

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
            Command::CreateTable(create) => {
                if current_index_semantics {
                    self.apply_create_table(cat, create)?;
                } else {
                    self.apply_create_table_legacy_replay(cat, create, entry.index, 0)?;
                }
            }
            Command::AddPrimaryKey(add) => {
                if current_index_semantics {
                    self.apply_add_primary_key(cat, add)?;
                } else {
                    self.apply_add_primary_key_legacy_replay(cat, add, entry.index)?;
                }
            }
            Command::AddUniqueConstraint(add) => {
                if current_index_semantics {
                    self.apply_add_unique_constraint(cat, add)?;
                } else {
                    self.apply_add_unique_constraint_legacy_replay(cat, add, entry.index)?;
                }
            }
            Command::AddCheckConstraint(add) => self.apply_add_check_constraint(cat, add)?,
            Command::AddForeignKey(add) => self.apply_add_foreign_key(cat, add, commit_seq)?,
            Command::AddColumn(add) => self.apply_add_column(cat, add, commit_seq)?,
            Command::RenameTable(rename) => {
                self.apply_rename_table(cat, rename, commit_seq, current_index_semantics)?
            }
            Command::RenameColumn(rename) => self.apply_rename_column(cat, rename)?,
            Command::RenameConstraint(rename) => self.apply_rename_constraint(cat, rename)?,
            Command::DropColumn(drop) => self.apply_drop_column(cat, drop, commit_seq)?,
            Command::DropConstraint(drop) => self.apply_drop_constraint(cat, drop)?,
            Command::CreateIndex(create) => {
                if current_index_semantics {
                    self.apply_create_index(cat, create)?;
                } else {
                    self.apply_create_index_legacy_replay(cat, create, entry.index, 0, true)?;
                }
            }
            Command::RenameIndex(rename) => {
                if current_index_semantics {
                    self.apply_rename_index(cat, rename)?;
                } else {
                    self.apply_rename_index_legacy_replay(cat, rename)?;
                }
            }
            Command::CreateView(create) => {
                if current_index_semantics {
                    self.apply_create_view(cat, create)?;
                } else {
                    self.apply_create_view_legacy_replay(cat, create)?;
                }
            }
            Command::RenameView(rename) => {
                if current_index_semantics {
                    self.apply_rename_view(cat, rename)?;
                } else {
                    self.apply_rename_view_legacy_replay(cat, rename)?;
                }
            }
            Command::CreateMaterializedView(create) => {
                if current_index_semantics {
                    self.apply_create_materialized_view(cat, create)?;
                } else {
                    self.apply_create_materialized_view_legacy_replay(cat, create)?;
                }
            }
            Command::RefreshMaterializedView(refresh) => {
                if current_index_semantics {
                    self.apply_refresh_materialized_view(cat, refresh)?;
                } else {
                    self.apply_refresh_materialized_view_legacy_replay(cat, refresh)?;
                }
            }
            Command::RenameMaterializedView(rename) => {
                if current_index_semantics {
                    self.apply_rename_materialized_view(cat, rename)?;
                } else {
                    self.apply_rename_materialized_view_legacy_replay(cat, rename)?;
                }
            }
            Command::CreateFunction(create) => self.apply_create_function(cat, create)?,
            Command::RenameFunction(rename) => self.apply_rename_function(cat, rename)?,
            Command::DropFunction(drop) => self.apply_drop_function(cat, drop)?,
            Command::SelectFunction(_) | Command::SelectLiteral(_) => {}
            Command::CreateSequence(create) => {
                if current_index_semantics {
                    self.apply_create_sequence(cat, create)?;
                } else {
                    self.apply_create_sequence_legacy_replay(cat, create)?;
                }
            }
            Command::CreateDomain(create) => self.apply_create_domain(cat, create)?,
            Command::SequenceNextVal(nextval) => {
                if current_index_semantics {
                    self.apply_sequence_nextval(cat, nextval)?;
                } else {
                    self.apply_sequence_nextval_legacy_replay(cat, nextval)?;
                }
            }
            Command::SequenceSetVal(setval) => {
                if current_index_semantics {
                    self.apply_sequence_setval(cat, setval)?;
                } else {
                    self.apply_sequence_setval_legacy_replay(cat, setval)?;
                }
            }
            Command::SequenceRestart(restart) => {
                self.apply_restart_sequence(cat, restart)?;
            }
            Command::RenameSequence(rename) => {
                if historical_command_replay {
                    self.apply_rename_sequence_historical_replay(
                        cat,
                        rename,
                        !current_index_semantics,
                    )?;
                } else if current_index_semantics {
                    self.apply_rename_sequence(cat, rename)?;
                } else {
                    self.apply_rename_sequence_legacy_replay(cat, rename)?;
                }
            }
            Command::DropTable(drop) => self.apply_drop_table(cat, drop, commit_seq)?,
            Command::TruncateTable(truncate) => {
                self.apply_truncate_table(cat, truncate, commit_seq)?
            }
            Command::DropIndex(drop) => {
                if current_index_semantics {
                    self.apply_drop_index(cat, drop)?;
                } else {
                    self.apply_drop_index_legacy_replay(cat, drop)?;
                }
            }
            Command::DropView(drop) => {
                if current_index_semantics {
                    self.apply_drop_view(cat, drop)?;
                } else {
                    self.apply_drop_view_legacy_replay(cat, drop)?;
                }
            }
            Command::DropMaterializedView(drop) => {
                if current_index_semantics {
                    self.apply_drop_materialized_view(cat, drop)?;
                } else {
                    self.apply_drop_materialized_view_legacy_replay(cat, drop)?;
                }
            }
            Command::DropSequence(drop) => {
                if historical_command_replay {
                    self.apply_drop_sequence_historical_replay(
                        cat,
                        drop,
                        !current_index_semantics,
                    )?;
                } else if current_index_semantics {
                    self.apply_drop_sequence(cat, drop)?;
                } else {
                    self.apply_drop_sequence_legacy_replay(cat, drop)?;
                }
            }
            Command::DropDomain(drop) => self.apply_drop_domain(cat, drop)?,
            Command::CreatePublication(create) => self.apply_create_publication(cat, create)?,
            Command::DropPublication(drop) => self.apply_drop_publication(cat, drop)?,
            Command::CreateSubscription(create) => self.apply_create_subscription(cat, create)?,
            Command::DropSubscription(drop) => self.apply_drop_subscription(cat, drop)?,
            Command::CreateRole(create) => self.apply_create_role(cat, create)?,
            Command::DropRole(drop) => self.apply_drop_role(cat, drop)?,
            Command::RenameRole(rename) => self.apply_rename_role(cat, rename)?,
            Command::AlterRoleLogin(alter) => self.apply_alter_role_login(cat, alter)?,
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
                applied = self.apply_delete(cat, delete, commit_seq)?.map(
                    |(table, rows, write_set, class_stamp)| AppliedRowMutation::Delete {
                        table,
                        rows,
                        write_set,
                        class_stamp,
                    },
                );
            }
            Command::Update(update) => {
                applied = self.apply_update(cat, update, commit_seq)?.map(
                    |(table, old_rows, new_rows, row_ids, write_set, class_stamp)| {
                        AppliedRowMutation::Update {
                            table,
                            old_rows,
                            new_rows,
                            row_ids,
                            class_stamp,
                            write_set,
                        }
                    },
                );
            }
            _ => {}
        }

        Ok(applied.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_db_replication::LogReplicator;

    fn opaque_codec5_payload() -> Arc<[u8]> {
        Arc::from(&b"GPUDBCANREC1\0\0\0\0prevalidated-codec5"[..])
    }

    fn live_witness(txn_id: TxnId, index: Index, payload: &Arc<[u8]>) -> LiveTypedTransactionApply {
        LiveTypedTransactionApply {
            txn_id,
            expected_index: index,
            payload_authority: LiveTypedPayloadAuthority::Exact(Arc::clone(payload)),
            payload_len: payload.len(),
            // These unit witnesses prove the common publication/root cut only; they do not
            // synthesize a catalog or a typed table contribution. Production witnesses name
            // their tables after S3 has established the catalog postimage.
            tables: Box::new([]),
            allocator_high_water: 17,
            allocator_already_consumed: false,
            affected_rows: 1,
            write_set: WriteSet::default(),
            private_sequence_publications: Box::new([]),
            catalog_composition: None,
            parent_authority: None,
        }
    }

    fn test_column_root(seed: u8) -> TypedColumnGenerationRoot {
        TypedColumnGenerationRoot {
            catalog_column_ordinal: 0,
            stable_column_id: u32::from(seed) + 1,
            attnum: 1,
            column_shape_root: [seed; 32],
            column_root: [seed.wrapping_add(1); 32],
        }
    }

    #[test]
    fn exact_live_payload_authority_rejects_an_equal_valued_distinct_allocation() {
        let engine = Engine::new_local();
        let proposed = opaque_codec5_payload();
        let equal_copy: Arc<[u8]> = Arc::from(proposed.as_ref());
        assert!(!Arc::ptr_eq(&proposed, &equal_copy));
        let mut commit = engine.commit_state();
        let token = commit
            .repl
            .propose(Arc::clone(&proposed))
            .expect("local proposal succeeds");

        let error = engine
            .apply_and_publish_committed_with_live_semantics_v2_codec5_transaction(
                &mut commit,
                9_000,
                token.index,
                live_witness(9_000, token.index, &equal_copy),
                None,
            )
            .expect_err("equal bytes under another allocation are not live payload authority");

        assert!(error
            .to_string()
            .contains("live typed transaction witness does not match"));
        assert_eq!(engine.visible_up_to(), 0);
    }

    #[test]
    fn semantics_v2_codec5_live_apply_records_prevalidated_entry_and_root_before_visibility() {
        let engine = Engine::new_local();
        let payload = opaque_codec5_payload();
        let predecessor = engine.read_state.typed_generation_roots.load_full();
        let successor = TypedTableGenerationRoot {
            data_generation: 1,
            table_root: [0x41; 32],
            logical_row_count: 0,
        };
        let completion = TypedTableMapGpuCompletion::synthetic_first_create_for_test(41);
        let root_publication =
            LiveTypedGenerationRootPublication::from_exact_gpu_completed_table_map_predecessor(
                predecessor,
                41,
                None,
                false,
                None,
                successor,
                &[test_column_root(0x41)],
                &[],
                &[],
                [0x73; 32],
                &completion,
            )
            .expect("initial typed root can produce its first successor");
        let mut commit = engine.commit_state();
        let token = commit
            .repl
            .propose(Arc::clone(&payload))
            .expect("local proposal succeeds");

        engine
            .apply_and_publish_committed_with_live_semantics_v2_codec5_transaction(
                &mut commit,
                9_001,
                token.index,
                live_witness(9_001, token.index, &payload),
                Some(root_publication),
            )
            .expect("prevalidated codec-5 live entry publishes");

        assert_eq!(commit.sm.applied, vec![payload.to_vec()]);
        assert!(commit.sm.kv.is_empty());
        assert_eq!(commit.last_applied_outcome, Some((token.index, 1)));
        let roots = engine.read_state.typed_generation_roots.load_full();
        assert_eq!(roots.database_root, Some([0x73; 32]));
        assert_eq!(roots.table(41), Some(successor));
        assert_eq!(engine.visible_up_to(), token.index);
    }

    #[test]
    fn semantics_v2_codec5_live_apply_fails_closed_on_root_predecessor_drift() {
        let engine = Engine::new_local();
        let payload = opaque_codec5_payload();
        // This equal-valued but independently allocated snapshot proves installation requires the
        // exact captured publication object, not just an equal root collection.
        let stale_predecessor = Arc::new(TypedGenerationRootSnapshot::default());
        let completion = TypedTableMapGpuCompletion::synthetic_first_create_for_test(42);
        let root_publication =
            LiveTypedGenerationRootPublication::from_exact_gpu_completed_table_map_predecessor(
                stale_predecessor,
                42,
                None,
                false,
                None,
                TypedTableGenerationRoot {
                    data_generation: 1,
                    table_root: [0x42; 32],
                    logical_row_count: 0,
                },
                &[test_column_root(0x42)],
                &[],
                &[],
                [0x74; 32],
                &completion,
            )
            .expect("stale root can still form a self-consistent candidate");
        let mut commit = engine.commit_state();
        let token = commit
            .repl
            .propose(Arc::clone(&payload))
            .expect("local proposal succeeds");

        let error = engine
            .apply_and_publish_committed_with_live_semantics_v2_codec5_transaction(
                &mut commit,
                9_002,
                token.index,
                live_witness(9_002, token.index, &payload),
                Some(root_publication),
            )
            .expect_err("root predecessor drift must stop publication");

        assert!(error
            .to_string()
            .contains("typed generation root publication predecessor drifted"));
        assert!(engine
            .read_state
            .typed_generation_roots
            .load_full()
            .database_root
            .is_none());
        assert_eq!(engine.visible_up_to(), 0);
    }
}
