//! Ordinary published-sequence value transitions.
//!
//! Each transition is a resolved binary operation committed through the engine's sole
//! commit/WAL/status/publication authority.  The enclosing user transaction may later commit or
//! roll back, but it never owns this state change. Transaction-private CREATE/RESTART identities
//! are deliberately refused here and remain children of the existing private catalog envelope.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceValueOutcome {
    pub transition_txn_id: TxnId,
    pub sequence_oid: u32,
    pub value: i64,
    pub currval_updated: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct AppliedSequenceValueTransition {
    pub(crate) commit_seq: Index,
    pub(crate) record: BinarySequenceValueTransitionRecord,
}

#[derive(Debug, Clone)]
struct SequenceValueTarget {
    sequence_oid: u32,
    source_name: String,
    effective_name: String,
    published_name: Option<String>,
    base_catalog_generation: Index,
    private_descriptor_digest: Option<gpu_db_wal::CanonicalDigest>,
}

pub(crate) struct SequenceDefaultStatementIdentity {
    pub(crate) parent_txn_id: TxnId,
    pub(crate) parent_autocommit: bool,
    pub(crate) statement_ordinal: u32,
    pub(crate) expression_ordinal_base: u32,
    pub(crate) parent_request_digest: gpu_db_wal::CanonicalDigest,
}

impl Engine {
    /// Share the canonical transaction-id allocator with the protocol-neutral facade. Engine-owned
    /// sequence transitions and facade-owned user envelopes consequently cannot alias.
    #[doc(hidden)]
    pub fn shared_transaction_id_allocator(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.transaction_id_allocator)
    }

    pub(crate) fn observe_transaction_id(&self, txn_id: TxnId) {
        self.transaction_id_allocator
            .fetch_max(txn_id.saturating_add(1), AtomicOrdering::Relaxed);
    }

    pub(crate) fn allocate_transaction_id(&self) -> Result<TxnId, TxnError> {
        self.transaction_id_allocator
            .fetch_update(AtomicOrdering::Relaxed, AtomicOrdering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map_err(|_| TxnError::IdExhausted)
    }

    /// Classify an autocommit INSERT against one immutable catalog generation. The returned
    /// generation is carried through either route, including the negative case, so a concurrent
    /// ADD/DROP DEFAULT cannot switch an already-classified statement between ordinary
    /// transition semantics and the legacy pure DML evaluator.
    pub(crate) fn insert_sequence_default_route(&self, command: &Command) -> (bool, Option<Index>) {
        let Command::Insert(insert) = command else {
            return (false, None);
        };
        let catalog = self.catalog_snapshot();
        if insert.columns.is_empty() {
            return (false, Some(catalog.commit_seq));
        }
        let omitted = catalog
            .relational_catalog
            .get(&insert.table)
            .is_some_and(|table| {
                table.columns.iter().any(|column| {
                    !insert.columns.contains(&column.name)
                        && matches!(column.default, Some(ColumnDefault::SequenceNextVal { .. }))
                })
            });
        (omitted, Some(catalog.commit_seq))
    }

    pub(crate) fn execute_sequence_default_autocommit(
        &self,
        txn_id: TxnId,
        command: Command,
        catalog_expectation: Option<crate::engine_mutation_admission::CatalogVersionExpectation>,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        let request_digest = transaction_statement_digest(&command)?;
        self.validate_sequence_default_parent_request(txn_id, request_digest)?;
        if let Some(rows_affected) =
            self.resolve_stable_retry_before_table_access(txn_id, request_digest)?
        {
            return Ok(DmlExecutionResult {
                rows_affected,
                returning: None,
            });
        }
        match self.reserve_pending_transaction_claim(txn_id, request_digest) {
            Ok(true) => {}
            Ok(false) => {
                return Err(ExecuteError::Indeterminate(format!(
                    "sequence-default transaction {txn_id} is pending in canonical mutation admission"
                )));
            }
            Err(error) => return Err(ExecuteError::Engine(error)),
        }
        if let Err(error) = self.begin_claimed_transaction_context(
            txn_id,
            TransactionCharacteristics::READ_COMMITTED_READ_WRITE,
            request_digest,
        ) {
            self.release_pending_transaction_claim(txn_id, request_digest);
            return match self.resolve_stable_retry_before_table_access(txn_id, request_digest) {
                Ok(Some(rows_affected)) => Ok(DmlExecutionResult {
                    rows_affected,
                    returning: None,
                }),
                Ok(None) => Err(error),
                Err(retry_error) => Err(retry_error),
            };
        }
        let result = match self.execute_prepared_dml_in_transaction_with_result(
            txn_id,
            command,
            catalog_expectation,
            true,
        ) {
            Ok(result) => result,
            Err(error) => {
                let _ = self.cancel_internal_transaction_context(txn_id);
                self.release_pending_transaction_claim(txn_id, request_digest);
                return Err(error);
            }
        };
        match self.commit_claimed_transaction_delta(
            txn_id,
            request_digest,
            current_timestamp_micros(),
        ) {
            Ok(()) => Ok(result),
            Err(error) => {
                if !self.is_commit_path_poisoned()
                    && self.transaction_snapshot_handle(txn_id).is_some()
                {
                    let _ = self.cancel_internal_transaction_context(txn_id);
                    self.release_pending_transaction_claim(txn_id, request_digest);
                }
                Err(error)
            }
        }
    }

    pub(crate) fn execute_sequence_value_command(
        &self,
        parent_txn_id: TxnId,
        command: &Command,
        principal: AuthorizationPrincipal,
    ) -> Result<SequenceValueOutcome, ExecuteError> {
        let (source_name, operation, set_value) = match command {
            Command::SequenceNextVal(nextval) => (
                nextval.name.as_str(),
                BinarySequenceValueOperation::NextVal,
                None,
            ),
            Command::SequenceSetVal(setval) => (
                setval.name.as_str(),
                BinarySequenceValueOperation::SetVal {
                    is_called: setval.is_called,
                },
                Some(setval.value),
            ),
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "sequence-value admission accepts nextval or setval".to_string(),
                )));
            }
        };
        let parent_request_digest = transaction_statement_digest(command)?;
        if let Some(snapshot) = self.transaction_snapshot_handle(parent_txn_id) {
            self.ensure_transaction_not_program_owned(parent_txn_id, &snapshot)?;
            let statement_lock = Arc::clone(&snapshot.statement_lock);
            let _statement = statement_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            self.ensure_transaction_snapshot_current(parent_txn_id, &snapshot)?;
            let snapshot =
                self.refresh_transaction_snapshot_for_statement(parent_txn_id, &snapshot)?;
            if snapshot.characteristics.access == TransactionAccessMode::ReadOnly {
                return Err(ExecuteError::Unsupported(
                    "cannot change a sequence in a READ ONLY transaction".to_string(),
                ));
            }
            let catalog = snapshot.transaction_catalog();
            self.authorize_command_at(&catalog, principal, command)?;
            let target =
                self.sequence_value_target(source_name, &catalog, Some(snapshot.as_ref()))?;
            if self.transaction_sequence_value_is_private(&snapshot, target.sequence_oid) {
                return Err(ExecuteError::Unsupported(
                    "nextval/setval on a transaction-private CREATE or RESTART sequence remains a private sequence child and is not an ordinary value transition"
                        .to_string(),
                ));
            }
            snapshot
                .table_access
                .acquire_shared([target.sequence_oid])?;
            let (statement_ordinal, expression_ordinal) = {
                let delta = snapshot
                    .delta
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let statement_ordinal = u32::try_from(delta.operations.len()).map_err(|_| {
                    ExecuteError::Unsupported(
                        "transaction operation count exceeds typed WAL framing".to_string(),
                    )
                })?;
                let expression_ordinal = u32::try_from(
                    delta
                        .sequence_value_references
                        .iter()
                        .filter(|reference| reference.statement_ordinal == statement_ordinal)
                        .count(),
                )
                .map_err(|_| {
                    ExecuteError::Unsupported(
                        "transaction sequence expression count exceeds typed WAL framing"
                            .to_string(),
                    )
                })?;
                (statement_ordinal, expression_ordinal)
            };
            let input_digest = sequence_value_input_digest(SequenceValueInput {
                parent_txn_id,
                parent_autocommit: false,
                statement_ordinal,
                expression_ordinal,
                parent_request_digest,
                source_name,
                operation,
                set_value,
            });
            let transition_txn_id = self.allocate_transaction_id()?;
            let outcome = self.commit_sequence_value_transition(
                transition_txn_id,
                input_digest,
                target,
                parent_txn_id,
                false,
                statement_ordinal,
                expression_ordinal,
                parent_request_digest,
                operation,
                set_value,
            )?;
            snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .sequence_value_references
                .push(BinarySequenceValueReference {
                    transition_txn_id: outcome.transition_txn_id,
                    parent_txn_id,
                    statement_ordinal,
                    expression_ordinal,
                    sequence_oid: outcome.sequence_oid,
                    returned_value: outcome.value,
                    input_digest,
                    table_oid: 0,
                    column_id: 0,
                    staging_row_ordinal: 0,
                    row_id: 0,
                    final_value_overwritten: false,
                    default_expression: false,
                });
            Ok(outcome)
        } else {
            let statement_ordinal = 0;
            let expression_ordinal = 0;
            let input_digest = sequence_value_input_digest(SequenceValueInput {
                parent_txn_id,
                parent_autocommit: true,
                statement_ordinal,
                expression_ordinal,
                parent_request_digest,
                source_name,
                operation,
                set_value,
            });
            if let Some(outcome) =
                self.resolve_sequence_value_transition_retry(parent_txn_id, input_digest)?
            {
                return Ok(outcome);
            }
            let catalog = self.catalog_snapshot();
            self.authorize_command_at(&catalog, principal, command)?;
            let target = self.sequence_value_target(source_name, &catalog, None)?;
            let lease = self.table_access.lease();
            lease.acquire_shared([target.sequence_oid])?;
            self.commit_sequence_value_transition(
                parent_txn_id,
                input_digest,
                target,
                parent_txn_id,
                true,
                statement_ordinal,
                expression_ordinal,
                parent_request_digest,
                operation,
                set_value,
            )
        }
    }

    fn resolve_sequence_value_transition_retry(
        &self,
        transition_txn_id: TxnId,
        input_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<Option<SequenceValueOutcome>, ExecuteError> {
        let Some(applied) = self
            .sequence_value_outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&transition_txn_id)
            .cloned()
        else {
            return Ok(None);
        };
        if applied.record.input_digest != input_digest {
            return Err(ExecuteError::Engine(EngineError::Durability(format!(
                "sequence transition id {transition_txn_id} is already claimed by a different input"
            ))));
        }
        let payload = encode_sequence_value_transition(&applied.record).ok_or_else(|| {
            ExecuteError::Engine(EngineError::Durability(
                "memoized sequence transition is not canonical".to_string(),
            ))
        })?;
        let commit = self.commit_state();
        if commit
            .resolve_transaction_retry(transition_txn_id, &payload)
            .map_err(ExecuteError::Engine)?
            .is_none()
        {
            return Err(ExecuteError::Engine(EngineError::Durability(format!(
                "memoized sequence transition {transition_txn_id} has no terminal claim"
            ))));
        }
        Ok(Some(sequence_value_outcome(
            transition_txn_id,
            &applied.record,
        )))
    }

    pub(crate) fn validate_sequence_default_parent_request(
        &self,
        parent_txn_id: TxnId,
        parent_request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<(), ExecuteError> {
        let outcomes = self
            .sequence_value_outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if outcomes.values().any(|applied| {
            applied.record.parent_autocommit
                && applied.record.parent_txn_id == parent_txn_id
                && applied.record.parent_request_digest != parent_request_digest
        }) {
            return Err(ExecuteError::Engine(EngineError::Durability(format!(
                "transaction id {parent_txn_id} already has a durable sequence outcome for a different request"
            ))));
        }
        Ok(())
    }

    pub(crate) fn validate_sequence_autocommit_statement_parent(
        &self,
        parent_txn_id: TxnId,
        command: &Command,
    ) -> Result<(), ExecuteError> {
        let claimed = self
            .sequence_value_outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .any(|applied| {
                applied.record.parent_autocommit && applied.record.parent_txn_id == parent_txn_id
            });
        if !claimed {
            return Ok(());
        }
        self.validate_sequence_default_parent_request(
            parent_txn_id,
            transaction_statement_digest(command)?,
        )
    }

    pub(crate) fn reject_nonstatement_sequence_autocommit_parent(
        &self,
        parent_txn_id: TxnId,
    ) -> Result<(), ExecuteError> {
        let claimed = self
            .sequence_value_outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .any(|applied| {
                applied.record.parent_autocommit && applied.record.parent_txn_id == parent_txn_id
            });
        if claimed {
            return Err(ExecuteError::Engine(EngineError::Durability(format!(
                "transaction id {parent_txn_id} already has a durable sequence outcome for a different request"
            ))));
        }
        Ok(())
    }

    pub fn sequence_oid_for_session(
        &self,
        txn_id: Option<TxnId>,
        name: &str,
    ) -> Result<u32, ExecuteError> {
        if let Some(txn_id) = txn_id {
            let snapshot = self
                .transaction_snapshot_handle(txn_id)
                .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
            self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
            let statement_lock = Arc::clone(&snapshot.statement_lock);
            let _statement = statement_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
            let snapshot = self.refresh_transaction_snapshot_for_statement(txn_id, &snapshot)?;
            return self
                .sequence_value_target(name, &snapshot.transaction_catalog(), Some(&snapshot))
                .map(|target| target.sequence_oid);
        }
        self.sequence_value_target(name, &self.catalog_snapshot(), None)
            .map(|target| target.sequence_oid)
    }

    pub fn sequence_currval_effects(&self, parent_txn_id: TxnId) -> Vec<(u32, i64)> {
        let mut effects = self
            .sequence_value_outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .filter(|applied| {
                applied.record.parent_txn_id == parent_txn_id
                    && match applied.record.operation {
                        BinarySequenceValueOperation::NextVal
                        | BinarySequenceValueOperation::Default => true,
                        BinarySequenceValueOperation::SetVal { is_called } => is_called,
                    }
            })
            .map(|applied| {
                (
                    applied.commit_seq,
                    applied.record.expression_ordinal,
                    applied.record.sequence_oid,
                    applied.record.returned_value,
                )
            })
            .collect::<Vec<_>>();
        effects.sort_unstable();
        effects
            .into_iter()
            .map(|(_, _, oid, value)| (oid, value))
            .collect()
    }

    /// Replace omitted defaults backed by unchanged published sequences with their already-durable
    /// values. Private CREATE/RESTART identities remain omitted so the existing pure default
    /// evaluator keeps them inside the user transaction's catalog/value overlay.
    pub(crate) fn materialize_published_sequence_defaults(
        &self,
        snapshot: &TransactionSnapshot,
        transaction_catalog: &CatalogSnapshot,
        table: &RelationalTable,
        command: &mut Command,
        identity: SequenceDefaultStatementIdentity,
    ) -> Result<Vec<BinarySequenceValueReference>, ExecuteError> {
        let Command::Insert(insert) = command else {
            return Ok(Vec::new());
        };
        if insert.columns.is_empty() {
            return Ok(Vec::new());
        }

        let mut provided = BTreeSet::new();
        for column in &insert.columns {
            if !provided.insert(column.as_str()) {
                return Err(ExecuteError::Engine(EngineError::DuplicateColumn(
                    column.clone(),
                )));
            }
            if !table
                .columns
                .iter()
                .any(|candidate| candidate.name == *column)
            {
                return Err(ExecuteError::Engine(EngineError::UndefinedColumn(
                    column.clone(),
                )));
            }
        }
        if insert
            .rows
            .iter()
            .any(|row| row.len() != insert.columns.len())
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "INSERT value count must match target columns".to_string(),
            )));
        }

        let mut omitted = Vec::new();
        for column in &table.columns {
            if provided.contains(column.name.as_str()) {
                continue;
            }
            let Some(ColumnDefault::SequenceNextVal { sequence, .. }) = &column.default else {
                continue;
            };
            let target =
                self.sequence_value_target(sequence, transaction_catalog, Some(snapshot))?;
            let private = self.transaction_sequence_value_is_private(snapshot, target.sequence_oid);
            omitted.push((
                column.name.clone(),
                column.id,
                sequence.clone(),
                target,
                private,
            ));
        }
        if omitted.is_empty() {
            return Ok(Vec::new());
        }

        let expression_count = insert
            .rows
            .len()
            .checked_mul(omitted.len())
            .ok_or_else(|| {
                ExecuteError::Unsupported(
                    "INSERT sequence-default expression count overflow".to_string(),
                )
            })?;
        u32::try_from(expression_count).map_err(|_| {
            ExecuteError::Unsupported(
                "INSERT sequence-default expression count exceeds typed WAL framing".to_string(),
            )
        })?;

        let mut references = Vec::new();
        for (row_index, row) in insert.rows.iter_mut().enumerate() {
            for (column_index, (_, column_id, sequence, target, private)) in
                omitted.iter().enumerate()
            {
                if *private {
                    continue;
                }
                let expression_ordinal = identity
                    .expression_ordinal_base
                    .checked_add(
                        u32::try_from(row_index * omitted.len() + column_index)
                            .expect("bounded above"),
                    )
                    .ok_or_else(|| {
                        ExecuteError::Unsupported(
                            "INSERT sequence-default expression ordinal overflow".to_string(),
                        )
                    })?;
                let operation = BinarySequenceValueOperation::Default;
                let input_digest = sequence_value_input_digest(SequenceValueInput {
                    parent_txn_id: identity.parent_txn_id,
                    parent_autocommit: identity.parent_autocommit,
                    statement_ordinal: identity.statement_ordinal,
                    expression_ordinal,
                    parent_request_digest: identity.parent_request_digest,
                    source_name: sequence,
                    operation,
                    set_value: None,
                });
                let transition_txn_id = self.allocate_transaction_id()?;
                let outcome = self.commit_sequence_value_transition(
                    transition_txn_id,
                    input_digest,
                    target.clone(),
                    identity.parent_txn_id,
                    identity.parent_autocommit,
                    identity.statement_ordinal,
                    expression_ordinal,
                    identity.parent_request_digest,
                    operation,
                    None,
                )?;
                let value = i32::try_from(outcome.value)
                    .map(SqlValue::Int4)
                    .map_err(|_| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "sequence value is out of range for int4 default".to_string(),
                        ))
                    })?;
                row.push(value);
                references.push(BinarySequenceValueReference {
                    transition_txn_id: outcome.transition_txn_id,
                    parent_txn_id: identity.parent_txn_id,
                    statement_ordinal: identity.statement_ordinal,
                    expression_ordinal,
                    sequence_oid: outcome.sequence_oid,
                    returned_value: outcome.value,
                    input_digest,
                    table_oid: table.oid,
                    column_id: *column_id,
                    staging_row_ordinal: u32::try_from(row_index)
                        .expect("expression count bounded above"),
                    row_id: 0,
                    final_value_overwritten: false,
                    default_expression: true,
                });
            }
        }
        insert.columns.extend(
            omitted
                .iter()
                .filter(|(_, _, _, _, private)| !private)
                .map(|(column, _, _, _, _)| column.clone()),
        );
        Ok(references)
    }

    /// Bind every materialized default to the exact transaction-private INSERT entity before the
    /// statement delta can be published. The returned transition is already durable at this
    /// point, so any mismatch must fail closed rather than leave an unverified reference for
    /// commit-time WAL construction.
    pub(crate) fn bind_sequence_default_insert_rows(
        &self,
        table: &RelationalTable,
        prepared: &WriteDelta,
        references: &mut [BinarySequenceValueReference],
    ) -> Result<(), ExecuteError> {
        if references.is_empty() {
            return Ok(());
        }
        let PreparedMutation::Insert {
            table: prepared_table,
            inserted_rows,
            ..
        } = &prepared.mutation
        else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "materialized sequence defaults did not produce an INSERT delta".to_string(),
            )));
        };
        if prepared_table != &table.name {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "materialized sequence defaults changed their INSERT target".to_string(),
            )));
        }
        let prefix = relational_key_prefix(prepared_table);
        let mut bindings = BTreeSet::new();
        for reference in references {
            if !reference.default_expression || reference.table_oid != table.oid {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "materialized sequence default lost its stable table identity".to_string(),
                )));
            }
            let row_index = usize::try_from(reference.staging_row_ordinal).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "materialized sequence default row ordinal exceeds usize".to_string(),
                ))
            })?;
            let (row_key, row) = inserted_rows.get(row_index).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "materialized sequence default row ordinal left its INSERT".to_string(),
                ))
            })?;
            let column_index = table
                .columns
                .iter()
                .position(|column| column.id == reference.column_id)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "materialized sequence default lost its stable column identity".to_string(),
                    ))
                })?;
            let expected = i32::try_from(reference.returned_value).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "materialized sequence default is outside its int4 column domain".to_string(),
                ))
            })?;
            if row.get(column_index) != Some(&SqlValue::Int4(expected)) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "materialized sequence default value left its prepared INSERT row".to_string(),
                )));
            }
            let row_id = crate::engine_residency::parse_relational_row_id(row_key, &prefix)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "materialized sequence default INSERT lost entity identity".to_string(),
                    ))
                })?;
            if !bindings.insert((row_id, reference.column_id)) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "materialized sequence default repeats one row-expression binding".to_string(),
                )));
            }
            reference.row_id = row_id;
            reference.staging_row_ordinal = 0;
        }
        Ok(())
    }

    /// Record whether the final coalesced user mutation still carries each sequence-produced
    /// value. A later update/delete may legitimately replace it, but that disposition is itself a
    /// typed WAL claim and is checked again during canonical apply and recovery.
    pub(crate) fn bind_sequence_reference_final_dispositions(
        record: &mut BinaryTransactionRecord,
        catalog: &CatalogSnapshot,
    ) -> Result<(), ExecuteError> {
        for reference in record
            .sequence_value_references
            .iter_mut()
            .filter(|reference| reference.default_expression)
        {
            let table = catalog
                .relational_catalog
                .values()
                .find(|table| table.oid == reference.table_oid)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "sequence default reference {} lost table identity before WAL binding",
                        reference.transition_txn_id
                    )))
                })?;
            let column_index = table
                .columns
                .iter()
                .position(|column| column.id == reference.column_id)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "sequence default reference {} lost column identity before WAL binding",
                        reference.transition_txn_id
                    )))
                })?;
            let expected = i32::try_from(reference.returned_value).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "sequence default reference {} is outside its int4 column domain",
                    reference.transition_txn_id
                )))
            })?;
            let mut matching_rows = record
                .mutations
                .iter()
                .filter_map(|mutation| match mutation {
                    BinaryTransactionMutation::Insert {
                        table: mutation_table,
                        row_id,
                        row_encoded,
                    } if mutation_table == &table.name && *row_id == reference.row_id => {
                        Some(row_encoded)
                    }
                    BinaryTransactionMutation::Update {
                        table: mutation_table,
                        row_id,
                        new_row_encoded,
                        ..
                    } if mutation_table == &table.name && *row_id == reference.row_id => {
                        Some(new_row_encoded)
                    }
                    BinaryTransactionMutation::Insert { .. }
                    | BinaryTransactionMutation::Update { .. }
                    | BinaryTransactionMutation::Delete { .. } => None,
                });
            let value_retained = match (matching_rows.next(), matching_rows.next()) {
                (Some(encoded), None) => {
                    let row = decode_relational_row(encoded, &table.columns)?;
                    row.get(column_index) == Some(&SqlValue::Int4(expected))
                }
                (None, None) => false,
                (Some(_), Some(_)) | (None, Some(_)) => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "sequence default reference {} has ambiguous final entity identity",
                        reference.transition_txn_id
                    ))));
                }
            };
            reference.final_value_overwritten = !value_retained;
        }
        Ok(())
    }

    fn sequence_value_target(
        &self,
        source_name: &str,
        catalog: &CatalogSnapshot,
        transaction_snapshot: Option<&TransactionSnapshot>,
    ) -> Result<SequenceValueTarget, ExecuteError> {
        match catalog
            .pg_class_relation_kind(source_name)
            .map_err(ExecuteError::Engine)?
        {
            Some(PgClassRelationKind::Sequence) => {}
            Some(_) => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{source_name}\" is not a sequence"
                ))));
            }
            None => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "sequence \"{source_name}\" does not exist"
                ))));
            }
        }
        let sequence = catalog
            .relational_sequences
            .get(source_name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(format!(
                    "catalog sequence binding {source_name:?} disappeared"
                )))
            })?;
        let published_name = self
            .catalog_snapshot()
            .relational_sequences
            .iter()
            .find_map(|(name, candidate)| (candidate.oid == sequence.oid).then_some(name.clone()));
        let private_descriptor_digest = transaction_snapshot
            .is_some()
            .then_some(published_name.as_deref())
            .flatten()
            .filter(|published| *published != source_name)
            .map(|_| sequence_descriptor_digest(sequence.oid, source_name));
        Ok(SequenceValueTarget {
            sequence_oid: sequence.oid,
            source_name: source_name.to_string(),
            effective_name: source_name.to_string(),
            published_name,
            base_catalog_generation: transaction_snapshot
                .map_or(catalog.commit_seq, |snapshot| snapshot.catalog.commit_seq),
            private_descriptor_digest,
        })
    }

    fn transaction_sequence_value_is_private(
        &self,
        snapshot: &TransactionSnapshot,
        sequence_oid: u32,
    ) -> bool {
        if !snapshot
            .catalog
            .relational_sequences
            .values()
            .any(|sequence| sequence.oid == sequence_oid)
        {
            return true;
        }
        let delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        delta.operations.iter().any(|operation| match operation {
            TransactionOperation::Catalog(staged) => {
                matches!(staged.command, Command::SequenceRestart(_))
                    && staged
                        .sequence_identity
                        .as_ref()
                        .into_iter()
                        .flat_map(|identity| &identity.targets)
                        .any(|target| {
                            target
                                .target_before
                                .as_ref()
                                .or(target.target_after.as_ref())
                                .is_some_and(|target| target.oid == sequence_oid)
                        })
            }
            TransactionOperation::TableReset(reset) => reset
                .sequence_reset_identity
                .as_ref()
                .into_iter()
                .flat_map(|identity| &identity.targets)
                .any(|target| {
                    target
                        .target_before
                        .as_ref()
                        .or(target.target_after.as_ref())
                        .is_some_and(|target| target.oid == sequence_oid)
                }),
            TransactionOperation::Row(_) => false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_sequence_value_transition(
        &self,
        transition_txn_id: TxnId,
        input_digest: gpu_db_wal::CanonicalDigest,
        target: SequenceValueTarget,
        parent_txn_id: TxnId,
        parent_autocommit: bool,
        statement_ordinal: u32,
        expression_ordinal: u32,
        parent_request_digest: gpu_db_wal::CanonicalDigest,
        operation: BinarySequenceValueOperation,
        set_value: Option<i64>,
    ) -> Result<SequenceValueOutcome, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        let mut commit = self
            .commit_state_after_wave_quiescence()
            .map_err(ExecuteError::Engine)?;
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;

        let applied = {
            let outcomes = self
                .sequence_value_outcomes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            outcomes.get(&transition_txn_id).cloned().or_else(|| {
                outcomes
                    .values()
                    .find(|applied| {
                        applied.record.parent_txn_id == parent_txn_id
                            && applied.record.parent_autocommit == parent_autocommit
                            && applied.record.statement_ordinal == statement_ordinal
                            && applied.record.expression_ordinal == expression_ordinal
                            && applied.record.parent_request_digest == parent_request_digest
                            && applied.record.input_digest == input_digest
                    })
                    .cloned()
            })
        };
        if let Some(applied) = applied {
            if applied.record.input_digest != input_digest
                || applied.record.parent_txn_id != parent_txn_id
                || applied.record.parent_autocommit != parent_autocommit
                || applied.record.statement_ordinal != statement_ordinal
                || applied.record.expression_ordinal != expression_ordinal
                || applied.record.parent_request_digest != parent_request_digest
            {
                return Err(ExecuteError::Engine(EngineError::Durability(format!(
                    "sequence transition id {transition_txn_id} is already claimed by a different input"
                ))));
            }
            let payload = encode_sequence_value_transition(&applied.record).ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(
                    "memoized sequence transition is not canonical".to_string(),
                ))
            })?;
            if commit
                .resolve_transaction_retry(applied.record.transition_txn_id, &payload)
                .map_err(ExecuteError::Engine)?
                .is_none()
            {
                return Err(ExecuteError::Engine(EngineError::Durability(format!(
                    "memoized sequence transition {} has no terminal claim",
                    applied.record.transition_txn_id
                ))));
            }
            return Ok(sequence_value_outcome(
                applied.record.transition_txn_id,
                &applied.record,
            ));
        }
        if commit.transaction_status.contains_key(&transition_txn_id)
            || self
                .pending_transaction_claims
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains_key(&transition_txn_id)
            || commit.txn_manager.state(transition_txn_id).is_some()
        {
            return Err(ExecuteError::Engine(EngineError::Durability(format!(
                "sequence transition id {transition_txn_id} is already claimed"
            ))));
        }

        let record = {
            let catalog = self.ddl_catalog();
            let (published_name, sequence) = catalog
                .relational_sequences
                .iter()
                .find(|(_, sequence)| sequence.oid == target.sequence_oid)
                .ok_or_else(|| {
                    ExecuteError::Serialization(format!(
                        "sequence stable identity {} left the published catalog",
                        target.sequence_oid
                    ))
                })?;
            if target.published_name.as_deref() != Some(published_name.as_str()) {
                return Err(ExecuteError::Serialization(format!(
                    "sequence stable identity {} changed its published binding after value resolution",
                    target.sequence_oid
                )));
            }
            let (new_last_value, new_is_called, returned_value) = match operation {
                BinarySequenceValueOperation::NextVal | BinarySequenceValueOperation::Default => {
                    let value = if sequence.is_called {
                        sequence.last_value.checked_add(1).ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(
                                "sequence value overflow".to_string(),
                            ))
                        })?
                    } else {
                        sequence.last_value
                    };
                    (value, true, value)
                }
                BinarySequenceValueOperation::SetVal { is_called } => {
                    let value = set_value.expect("setval transition carries requested value");
                    (value, is_called, value)
                }
            };
            BinarySequenceValueTransitionRecord {
                transition_txn_id,
                parent_txn_id,
                parent_autocommit,
                statement_ordinal,
                expression_ordinal,
                parent_request_digest,
                input_digest,
                sequence_oid: target.sequence_oid,
                source_name: target.source_name,
                effective_name: target.effective_name,
                published_name: published_name.clone(),
                base_catalog_generation: target.base_catalog_generation,
                prior_last_value: sequence.last_value,
                prior_is_called: sequence.is_called,
                new_last_value,
                new_is_called,
                returned_value,
                private_descriptor_digest: target.private_descriptor_digest,
                operation,
            }
        };
        let payload: Arc<[u8]> =
            Arc::from(encode_sequence_value_transition(&record).ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(
                    "resolved sequence transition exceeds binary framing limits".to_string(),
                ))
            })?);
        let timestamp_micros = current_timestamp_micros();
        let wal_len_before = commit.wal.len();
        let token = commit
            .repl
            .propose(Arc::clone(&payload))
            .map_err(ExecuteError::Engine)?;
        let wal_record = match Self::canonical_wal_record(
            &mut commit,
            transition_txn_id,
            token.index,
            0,
            &payload,
        ) {
            Ok(record) => record,
            Err(error) => {
                commit.repl.rollback_unapplied_from(token.index);
                return Err(ExecuteError::Engine(error));
            }
        };
        commit.wal.append_canonical(wal_record);
        if let Err(error) = commit.wal.flush_all() {
            commit.repl.rollback_unapplied_from(token.index);
            commit.wal.truncate(wal_len_before);
            return Err(ExecuteError::Engine(error));
        }
        if let Err(error) = commit.repl.wait_committed(token, Duration::from_millis(0)) {
            self.wedge_commit_path();
            return Err(ExecuteError::Indeterminate(format!(
                "sequence transition {transition_txn_id} is durable but replication confirmation failed: {error}; restart recovery must resolve it"
            )));
        }
        if let Err(error) = commit.record_transaction_status_digest_outcome(
            transition_txn_id,
            gpu_db_wal::canonical_request_digest(&payload),
            token.index,
            0,
        ) {
            self.wedge_commit_path();
            return Err(ExecuteError::Indeterminate(format!(
                "sequence transition {transition_txn_id} is durable but terminal status installation failed: {error}; restart recovery must resolve it"
            )));
        }
        commit.record_commit_timestamp(transition_txn_id, timestamp_micros);
        if let Err(error) =
            self.apply_and_publish_committed(&mut commit, transition_txn_id, token.index)
        {
            self.wedge_commit_path();
            return Err(ExecuteError::Indeterminate(format!(
                "sequence transition {transition_txn_id} is durable but could not be installed: {error}; restart recovery required"
            )));
        }
        self.metrics.inc_commit();
        drop(commit);
        self.maybe_auto_checkpoint_wal();
        Ok(sequence_value_outcome(transition_txn_id, &record))
    }

    pub(crate) fn apply_sequence_value_transition_record(
        &self,
        entry: &LogEntry,
        catalog: &mut DdlCatalogState,
        record: BinarySequenceValueTransitionRecord,
    ) -> Result<(), EngineError> {
        if !valid_sequence_value_transition(&record) {
            return Err(EngineError::Durability(
                "sequence transition WAL is not canonical".to_string(),
            ));
        }
        let mut outcomes = self
            .sequence_value_outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if outcomes.contains_key(&record.transition_txn_id) {
            return Err(EngineError::Durability(format!(
                "sequence transition WAL repeats identity {}",
                record.transition_txn_id
            )));
        }
        let (published_name, sequence) = catalog
            .relational_sequences
            .iter_mut()
            .find(|(_, sequence)| sequence.oid == record.sequence_oid)
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "sequence transition targets unknown stable identity {}",
                    record.sequence_oid
                ))
            })?;
        if published_name != &record.published_name {
            return Err(EngineError::Durability(format!(
                "sequence transition {} published binding does not match stable identity {}",
                record.transition_txn_id, record.sequence_oid
            )));
        }
        if (sequence.last_value, sequence.is_called)
            != (record.prior_last_value, record.prior_is_called)
        {
            return Err(EngineError::Durability(format!(
                "sequence transition {} prior state does not match stable identity {}",
                record.transition_txn_id, record.sequence_oid
            )));
        }
        sequence.last_value = record.new_last_value;
        sequence.is_called = record.new_is_called;
        outcomes.insert(
            record.transition_txn_id,
            AppliedSequenceValueTransition {
                commit_seq: entry.index,
                record,
            },
        );
        Ok(())
    }

    pub(crate) fn validate_sequence_value_references(
        &self,
        references: &[BinarySequenceValueReference],
    ) -> Result<(), EngineError> {
        self.sequence_value_reference_records(references)
            .map(|_| ())
    }

    pub(crate) fn sequence_value_reference_records(
        &self,
        references: &[BinarySequenceValueReference],
    ) -> Result<Vec<BinarySequenceValueTransitionRecord>, EngineError> {
        let outcomes = self
            .sequence_value_outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut records = Vec::with_capacity(references.len());
        for reference in references {
            let applied = outcomes.get(&reference.transition_txn_id).ok_or_else(|| {
                EngineError::Durability(format!(
                    "user transaction references missing sequence transition {}",
                    reference.transition_txn_id
                ))
            })?;
            if applied.record.sequence_oid != reference.sequence_oid
                || applied.record.parent_txn_id != reference.parent_txn_id
                || applied.record.returned_value != reference.returned_value
                || applied.record.input_digest != reference.input_digest
                || applied.record.statement_ordinal != reference.statement_ordinal
                || applied.record.expression_ordinal != reference.expression_ordinal
                || (reference.default_expression
                    != matches!(
                        applied.record.operation,
                        BinarySequenceValueOperation::Default
                    ))
            {
                return Err(EngineError::Durability(format!(
                    "user transaction sequence reference {} does not match its durable transition",
                    reference.transition_txn_id
                )));
            }
            records.push(applied.record.clone());
        }
        Ok(records)
    }

    pub(crate) fn prepare_sequence_lifecycle_reference_replay(
        &self,
        working: &mut DdlCatalogState,
        references: &[BinarySequenceValueReference],
        lifecycle_oids: &BTreeSet<u32>,
    ) -> Result<Vec<BinarySequenceValueTransitionRecord>, EngineError> {
        let records = self
            .sequence_value_reference_records(references)?
            .into_iter()
            .filter(|record| lifecycle_oids.contains(&record.sequence_oid))
            .collect::<Vec<_>>();
        for sequence_oid in lifecycle_oids {
            let relevant = records
                .iter()
                .filter(|record| record.sequence_oid == *sequence_oid)
                .collect::<Vec<_>>();
            let Some(first) = relevant.first() else {
                continue;
            };
            for pair in relevant.windows(2) {
                if (pair[0].new_last_value, pair[0].new_is_called)
                    != (pair[1].prior_last_value, pair[1].prior_is_called)
                {
                    return Err(EngineError::Durability(format!(
                        "sequence transition references for stable identity {sequence_oid} do not form one ordered state chain"
                    )));
                }
            }
            let last = relevant.last().expect("first proved non-empty");
            let sequence = working
                .relational_sequences
                .values_mut()
                .find(|sequence| sequence.oid == *sequence_oid)
                .ok_or_else(|| {
                    EngineError::Durability(format!(
                        "sequence lifecycle reference targets unknown stable identity {sequence_oid}"
                    ))
                })?;
            if (sequence.last_value, sequence.is_called)
                != (last.new_last_value, last.new_is_called)
            {
                return Err(EngineError::Durability(format!(
                    "published sequence state for stable identity {sequence_oid} does not end at its referenced transition"
                )));
            }
            sequence.last_value = first.prior_last_value;
            sequence.is_called = first.prior_is_called;
        }
        Ok(records)
    }

    pub(crate) fn apply_sequence_lifecycle_references_through(
        records: &[BinarySequenceValueTransitionRecord],
        next_record: &mut usize,
        through_statement_ordinal: u32,
        working: &mut DdlCatalogState,
    ) -> Result<(), EngineError> {
        while records
            .get(*next_record)
            .is_some_and(|record| record.statement_ordinal <= through_statement_ordinal)
        {
            let record = &records[*next_record];
            let sequence = working
                .relational_sequences
                .values_mut()
                .find(|sequence| sequence.oid == record.sequence_oid)
                .ok_or_else(|| {
                    EngineError::Durability(format!(
                        "sequence transition {} cannot be interleaved after its stable identity was removed",
                        record.transition_txn_id
                    ))
                })?;
            if (sequence.last_value, sequence.is_called)
                != (record.prior_last_value, record.prior_is_called)
            {
                return Err(EngineError::Durability(format!(
                    "sequence transition {} does not follow its lifecycle-visible prior state",
                    record.transition_txn_id
                )));
            }
            sequence.last_value = record.new_last_value;
            sequence.is_called = record.new_is_called;
            *next_record += 1;
        }
        Ok(())
    }

    pub(crate) fn catalogs_same_ignoring_referenced_sequence_values(
        left: &CatalogSnapshot,
        right: &CatalogSnapshot,
        references: &[BinarySequenceValueReference],
    ) -> bool {
        if references.is_empty() {
            return left.same_contents(right);
        }
        let oids = references
            .iter()
            .map(|reference| reference.sequence_oid)
            .collect::<BTreeSet<_>>();
        let mut left = left.clone();
        let mut right = right.clone();
        for oid in oids {
            let left_sequence = left
                .relational_sequences
                .values_mut()
                .find(|sequence| sequence.oid == oid);
            let right_sequence = right
                .relational_sequences
                .values_mut()
                .find(|sequence| sequence.oid == oid);
            match (left_sequence, right_sequence) {
                (Some(left), Some(right)) => {
                    left.last_value = 0;
                    left.is_called = false;
                    right.last_value = 0;
                    right.is_called = false;
                }
                (None, None) => {}
                (Some(_), None) | (None, Some(_)) => return false,
            }
        }
        left.same_contents(&right)
    }
}

fn sequence_value_outcome(
    transition_txn_id: TxnId,
    record: &BinarySequenceValueTransitionRecord,
) -> SequenceValueOutcome {
    SequenceValueOutcome {
        transition_txn_id,
        sequence_oid: record.sequence_oid,
        value: record.returned_value,
        currval_updated: match record.operation {
            BinarySequenceValueOperation::NextVal | BinarySequenceValueOperation::Default => true,
            BinarySequenceValueOperation::SetVal { is_called } => is_called,
        },
    }
}
