//! Transaction-private catalog staging.
//!
//! Database-local `CREATE TABLE` operations share the transaction's statement-ordered operation
//! envelope with DML and typed table resets. Each statement rebuilds the private catalog from the
//! exact published base plus its typed predecessors, then atomically replaces the private overlay.

use super::*;
use crate::engine_mutation_admission::validate_prepared_catalog_version;

mod index_identity;
mod index_owner_generation;
mod reset_rebind;
mod sequence_identity;
mod view_identity;

#[derive(Clone, Copy)]
pub(crate) struct TransactionCatalogEnvelopeSlices<'a> {
    pub(crate) view_operations: &'a [BinaryTransactionViewOperationIdentity],
    pub(crate) view_lifecycle_operations: &'a [BinaryTransactionViewLifecycleOperationIdentity],
    pub(crate) index_lifecycle_operations: &'a [BinaryTransactionIndexLifecycleOperationIdentity],
    pub(crate) sequence_lifecycle_operations:
        &'a [BinaryTransactionSequenceLifecycleOperationIdentity],
    pub(crate) sequence_reset_operations: &'a [BinaryTransactionSequenceResetOperationIdentity],
    pub(crate) operation_order: &'a [BinaryTransactionOperationIdentity],
    pub(crate) catalog_output: Option<&'a BinaryTransactionCatalogOutput>,
    pub(crate) sequence_input_oids: &'a BTreeMap<(u32, String), u32>,
    pub(crate) sequence_value_references: &'a [BinarySequenceValueReference],
}

impl Engine {
    pub(crate) fn execute_catalog_in_transaction(
        &self,
        txn_id: TxnId,
        command: Command,
        source: Arc<str>,
        expected_catalog_version: Option<Index>,
    ) -> Result<(), ExecuteError> {
        self.execute_catalog_in_transaction_with_hook(
            txn_id,
            command,
            source,
            expected_catalog_version,
            || {},
        )
    }

    #[cfg(test)]
    fn execute_catalog_in_transaction_instrumented(
        &self,
        txn_id: TxnId,
        command: Command,
        source: Arc<str>,
        expected_catalog_version: Option<Index>,
        on_catalog_latched: impl FnOnce(),
    ) -> Result<(), ExecuteError> {
        self.execute_catalog_in_transaction_with_hook(
            txn_id,
            command,
            source,
            expected_catalog_version,
            on_catalog_latched,
        )
    }

    fn execute_catalog_in_transaction_with_hook(
        &self,
        txn_id: TxnId,
        command: Command,
        source: Arc<str>,
        expected_catalog_version: Option<Index>,
        on_catalog_latched: impl FnOnce(),
    ) -> Result<(), ExecuteError> {
        if !Self::transaction_catalog_command_is_supported(&command) {
            return Err(ExecuteError::Unsupported(
                "transactional catalog staging currently supports CREATE TABLE, stored-view lifecycle, index lifecycle, and sequence lifecycle commands only"
                    .to_string(),
            ));
        }
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
        self.execute_catalog_in_transaction_statement_locked(
            txn_id,
            command,
            source,
            expected_catalog_version,
            &snapshot,
            on_catalog_latched,
        )
    }

    fn execute_catalog_in_transaction_statement_locked(
        &self,
        txn_id: TxnId,
        command: Command,
        _source: Arc<str>,
        expected_catalog_version: Option<Index>,
        snapshot: &Arc<TransactionSnapshot>,
        on_catalog_latched: impl FnOnce(),
    ) -> Result<(), ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        self.ensure_transaction_snapshot_current(txn_id, snapshot)?;
        if snapshot.characteristics.access == TransactionAccessMode::ReadOnly {
            return Err(ExecuteError::Unsupported(
                "cannot execute DDL in a READ ONLY transaction".to_string(),
            ));
        }
        if let Some(expected) = expected_catalog_version {
            validate_prepared_catalog_version(expected, snapshot.catalog.commit_seq)?;
        }

        let (
            generation,
            prior_operations,
            prior_commands,
            prior_overlay,
            prior_base,
            sequence_value_references,
        ) = {
            let delta = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                delta.generation,
                delta.operations.clone(),
                delta
                    .operations
                    .iter()
                    .filter_map(|operation| match operation {
                        TransactionOperation::Catalog(staged) => Some(staged.as_ref().clone()),
                        TransactionOperation::Row(_) | TransactionOperation::TableReset(_) => None,
                    })
                    .collect::<Vec<_>>(),
                delta.catalog_overlay.clone(),
                delta.catalog_base.clone(),
                delta.sequence_value_references.clone(),
            )
        };
        let prior_has_sequence_reset = prior_operations.iter().any(|operation| {
            matches!(
                operation,
                TransactionOperation::TableReset(reset)
                    if reset.sequence_reset_identity.is_some()
            )
        });
        if (!prior_commands.is_empty() || prior_has_sequence_reset)
            && (prior_overlay.is_none() || prior_base.is_none())
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "transactional catalog operation envelope lost its base or private overlay"
                    .to_string(),
            )));
        }
        if prior_base.as_deref().is_some_and(|base| {
            !Self::catalogs_same_ignoring_referenced_sequence_values(
                base,
                snapshot.catalog.as_ref(),
                &sequence_value_references,
            )
        }) {
            return Err(ExecuteError::Serialization(
                "transactional catalog base changed before the next catalog statement".to_string(),
            ));
        }
        if command_is_index_lifecycle(&command) {
            let access_catalog = prior_overlay
                .as_deref()
                .unwrap_or(snapshot.catalog.as_ref());
            let identities = Self::transaction_index_table_identities(access_catalog, &command)?;
            #[cfg(test)]
            if matches!(&command, Command::CreateIndex(create) if create.unique) {
                self.run_index_owner_pre_acquire_hook();
            }
            self.acquire_transaction_write_table_access_identities(snapshot, &identities)?;
            snapshot
                .table_access
                .acquire_exclusive(identities.values().copied())?;
            if let Command::CreateIndex(create) = &command {
                self.validate_transaction_unique_index_owner_generation(snapshot, create)?;
            }
        }
        if command_is_sequence_lifecycle(&command) {
            let access_catalog = prior_overlay
                .as_deref()
                .unwrap_or(snapshot.catalog.as_ref());
            let names = match &command {
                Command::CreateSequence(_) => Vec::new(),
                Command::SequenceRestart(restart) => vec![restart.name.as_str()],
                Command::RenameSequence(rename) => vec![rename.old_name.as_str()],
                Command::DropSequence(drop) => drop.names.iter().map(String::as_str).collect(),
                _ => unreachable!("sequence lifecycle classifier checked above"),
            };
            let identities = names
                .into_iter()
                .filter_map(|name| {
                    access_catalog
                        .relational_sequences
                        .get(name)
                        .map(|sequence| sequence.oid)
                })
                .collect::<BTreeSet<_>>();
            snapshot.table_access.acquire_exclusive(identities)?;
        }

        // Validation works on a private clone. The exact published base must still match the
        // statement snapshot; otherwise retryable serialization wins before any private state is
        // installed. No allocator or global catalog field changes here, so rollback is complete.
        let catalog_guard = self.ddl_catalog();
        let current =
            Self::catalog_snapshot_from_working(&catalog_guard, snapshot.catalog.commit_seq);
        if current.as_ref() != snapshot.catalog.as_ref() {
            return Err(ExecuteError::Serialization(
                "catalog changed before transactional DDL staging".to_string(),
            ));
        }
        on_catalog_latched();
        let mut working = catalog_guard.clone();
        let sequence_lifecycle_oids = prior_operations
            .iter()
            .flat_map(|operation| match operation {
                TransactionOperation::Catalog(staged) => staged
                    .sequence_identity
                    .as_ref()
                    .into_iter()
                    .flat_map(|identity| &identity.targets)
                    .filter_map(|target| {
                        target
                            .target_before
                            .as_ref()
                            .or(target.target_after.as_ref())
                            .map(|target| target.oid)
                    })
                    .collect::<Vec<_>>(),
                TransactionOperation::TableReset(reset) => reset
                    .sequence_reset_identity
                    .as_ref()
                    .into_iter()
                    .flat_map(|identity| &identity.targets)
                    .filter_map(|target| {
                        target
                            .target_before
                            .as_ref()
                            .or(target.target_after.as_ref())
                            .map(|target| target.oid)
                    })
                    .collect::<Vec<_>>(),
                TransactionOperation::Row(_) => Vec::new(),
            })
            .collect::<BTreeSet<_>>();
        let sequence_reference_records = self
            .prepare_sequence_lifecycle_reference_replay(
                &mut working,
                &sequence_value_references,
                &sequence_lifecycle_oids,
            )
            .map_err(ExecuteError::Engine)?;
        let mut next_sequence_reference = 0usize;
        for (ordinal, operation) in prior_operations.iter().enumerate() {
            Self::apply_sequence_lifecycle_references_through(
                &sequence_reference_records,
                &mut next_sequence_reference,
                u32::try_from(ordinal).map_err(|_| {
                    ExecuteError::Unsupported(
                        "transaction operation ordinal exceeds typed catalog framing".to_string(),
                    )
                })?,
                &mut working,
            )
            .map_err(ExecuteError::Engine)?;
            let TransactionOperation::Catalog(staged) = operation else {
                if let TransactionOperation::TableReset(reset) = operation {
                    if let Some(identity) = &reset.sequence_reset_identity {
                        Self::apply_transaction_sequence_reset_identity(
                            &mut working,
                            snapshot.catalog.commit_seq,
                            identity,
                        )
                        .map_err(ExecuteError::Engine)?;
                    }
                }
                continue;
            };
            let scoped = Self::catalog_snapshot_from_working(&working, snapshot.catalog.commit_seq);
            if let Some(identity) = &staged.view_identity {
                Self::validate_transaction_view_before(&scoped, &staged.command, identity)
                    .map_err(ExecuteError::Engine)?;
            }
            if let Some(identity) = &staged.index_identity {
                Self::validate_transaction_index_before(&scoped, &staged.command, identity)
                    .map_err(ExecuteError::Engine)?;
            }
            if let Some(identity) = &staged.sequence_identity {
                Self::validate_transaction_sequence_before(&scoped, &staged.command, identity)
                    .map_err(ExecuteError::Engine)?;
            }
            self.with_apply_catalog(Some(Arc::clone(&scoped)), || {
                self.apply_transaction_catalog_command(&mut working, staged.command.clone())
            })
            .map_err(ExecuteError::Engine)?;
            if let Some(identity) = &staged.view_identity {
                let applied =
                    Self::catalog_snapshot_from_working(&working, snapshot.catalog.commit_seq);
                Self::validate_transaction_view_after(&applied, &staged.command, identity)
                    .map_err(ExecuteError::Engine)?;
            } else if command_is_view_lifecycle(&staged.command) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "transactional stored-view operation lost its typed identity closure"
                        .to_string(),
                )));
            }
            if let Some(identity) = &staged.index_identity {
                let applied =
                    Self::catalog_snapshot_from_working(&working, snapshot.catalog.commit_seq);
                Self::validate_transaction_index_after(&applied, &staged.command, identity)
                    .map_err(ExecuteError::Engine)?;
            } else if command_is_index_lifecycle(&staged.command) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "transactional index operation lost its typed identity closure".to_string(),
                )));
            }
            if let Some(identity) = &staged.sequence_identity {
                let applied =
                    Self::catalog_snapshot_from_working(&working, snapshot.catalog.commit_seq);
                Self::validate_transaction_sequence_after(&applied, &staged.command, identity)
                    .map_err(ExecuteError::Engine)?;
            } else if command_is_sequence_lifecycle(&staged.command) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "transactional sequence operation lost its typed identity closure".to_string(),
                )));
            }
        }
        Self::apply_sequence_lifecycle_references_through(
            &sequence_reference_records,
            &mut next_sequence_reference,
            u32::MAX,
            &mut working,
        )
        .map_err(ExecuteError::Engine)?;
        if let Some(expected) = prior_overlay.as_deref() {
            let reconstructed =
                Self::catalog_snapshot_from_working(&working, snapshot.catalog.commit_seq);
            if reconstructed.as_ref() != expected {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "typed transactional catalog operations no longer reconstruct their private overlay"
                        .to_string(),
                )));
            }
        }
        let scoped = Self::catalog_snapshot_from_working(&working, snapshot.catalog.commit_seq);
        let index_epoch_before = working.index_oid_epoch_current;
        self.with_apply_catalog(Some(Arc::clone(&scoped)), || {
            self.apply_transaction_catalog_command(&mut working, command.clone())
        })
        .map_err(ExecuteError::Engine)?;
        let index_epoch_transition = !index_epoch_before && working.index_oid_epoch_current;
        // CREATE validation helpers resolve domains and sequence/default dependencies through the
        // scoped working generation. Retain the catalog latch through reconstruction and the new
        // private apply so no global catalog generation can cross the checked base.
        drop(catalog_guard);
        let overlay = Self::catalog_snapshot_from_working(&working, snapshot.catalog.commit_seq);
        if let Command::CreateIndex(create) = &command {
            self.validate_transaction_create_index_device(snapshot, &overlay, create)?;
        }
        let rebound_resets = self.rebind_transaction_table_reset_outputs(
            &prior_operations,
            &scoped,
            &overlay,
            &command,
        )?;

        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if delta.generation != generation {
            return Err(ExecuteError::Serialization(
                "transaction catalog generation changed during statement staging".to_string(),
            ));
        }
        let ordinal = u32::try_from(delta.operations.len()).map_err(|_| {
            ExecuteError::Unsupported(
                "transaction operation ordinal exceeds typed catalog framing".to_string(),
            )
        })?;
        let command_index = u32::try_from(prior_commands.len()).map_err(|_| {
            ExecuteError::Unsupported(
                "transaction catalog operation count exceeds typed identity framing".to_string(),
            )
        })?;
        let statement_digest = transaction_statement_digest(&command)?;
        let view_identity = if command_is_view_lifecycle(&command) {
            Some(Self::transaction_view_operation_identity(
                command_index,
                ordinal,
                &scoped,
                &overlay,
                &command,
            )?)
        } else {
            None
        };
        let index_identity = if command_is_index_lifecycle(&command) {
            Some(Self::transaction_index_operation_identity(
                command_index,
                ordinal,
                &scoped,
                &overlay,
                &command,
            )?)
        } else {
            None
        };
        let sequence_identity = if command_is_sequence_lifecycle(&command) {
            Some(Self::transaction_sequence_operation_identity(
                command_index,
                ordinal,
                &scoped,
                &overlay,
                &command,
            )?)
        } else {
            None
        };
        let sequence_input_oids = match &command {
            Command::CreateTable(create) => create
                .columns
                .iter()
                .filter_map(|column| match &column.default {
                    Some(ColumnDefault::SequenceNextVal { sequence, .. }) => Some(sequence),
                    _ => None,
                })
                .map(|name| {
                    overlay
                        .relational_sequences
                        .get(name)
                        .map(|sequence| (name.clone(), sequence.oid))
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                                "transaction catalog sequence input \"{name}\" left its command postimage"
                            ))
                        })
                })
                .collect::<Result<BTreeMap<_, _>, _>>()?,
            _ => BTreeMap::new(),
        };
        if let Command::CreateTable(create) = &command {
            let mut resident_shards = delta.resident_shards.as_ref().clone();
            resident_shards.entry(create.table.clone()).or_default();
            delta.publish_resident_shards(Arc::new(resident_shards));
            for (name, oid) in &sequence_input_oids {
                if create.columns.iter().any(|column| {
                    matches!(
                        &column.default,
                        Some(ColumnDefault::SequenceNextVal {
                            sequence,
                            create_if_missing: true,
                        }) if sequence == name
                    )
                }) {
                    delta.sequence_state.insert(name.clone(), (1, false));
                    delta.sequence_state_by_oid.insert(*oid, (1, false));
                }
            }
        }
        match &command {
            Command::CreateSequence(create) => {
                let sequence = overlay
                    .relational_sequences
                    .get(&create.name)
                    .expect("CREATE SEQUENCE postimage contains its target");
                delta.sequence_state.insert(
                    create.name.clone(),
                    (sequence.last_value, sequence.is_called),
                );
                delta
                    .sequence_state_by_oid
                    .insert(sequence.oid, (sequence.last_value, sequence.is_called));
            }
            Command::SequenceRestart(restart) => {
                let sequence = overlay
                    .relational_sequences
                    .get(&restart.name)
                    .expect("SEQUENCE RESTART postimage contains its target");
                delta.sequence_state.insert(
                    restart.name.clone(),
                    (sequence.last_value, sequence.is_called),
                );
                delta
                    .sequence_state_by_oid
                    .insert(sequence.oid, (sequence.last_value, sequence.is_called));
            }
            Command::RenameSequence(rename) => {
                if let Some(state) = delta.sequence_state.remove(&rename.old_name) {
                    delta.sequence_state.insert(rename.new_name.clone(), state);
                }
            }
            Command::DropSequence(_) => {
                if let Some(identity) = &sequence_identity {
                    for target in &identity.targets {
                        delta.sequence_state.remove(&target.before_name);
                        if let Some(after_name) = &target.after_name {
                            delta.sequence_state.remove(after_name);
                        }
                        if let Some(target_before) = &target.target_before {
                            delta.sequence_state_by_oid.remove(&target_before.oid);
                        }
                    }
                }
            }
            _ => {}
        }
        let rebind_owners = delta
            .operations
            .iter()
            .filter(|operation| {
                matches!(
                    operation,
                    TransactionOperation::TableReset(reset)
                        if rebound_resets.contains_key(&reset.ordinal)
                )
            })
            .count();
        if rebind_owners != rebound_resets.len() {
            return Err(ExecuteError::Serialization(
                "transaction reset output rebind lost its statement owner".to_string(),
            ));
        }
        for operation in &mut delta.operations {
            let TransactionOperation::TableReset(reset) = operation else {
                continue;
            };
            if let Some(rebound) = rebound_resets.get(&reset.ordinal) {
                *reset = Arc::new(rebound.clone());
            }
        }
        if delta.catalog_base.is_none() {
            delta.catalog_base = Some(Arc::clone(&snapshot.catalog));
        }
        delta.catalog_overlay = Some(overlay);
        delta
            .operations
            .push(TransactionOperation::Catalog(Arc::new(
                StagedCatalogCommand {
                    ordinal,
                    statement_digest,
                    command,
                    index_epoch_transition,
                    view_identity,
                    index_identity,
                    sequence_identity,
                    sequence_input_oids,
                },
            )));
        delta.generation = delta.generation.saturating_add(1);
        Ok(())
    }
}

#[cfg(test)]
#[path = "engine_transaction_catalog/ordered_tests.rs"]
mod ordered_tests;

#[cfg(test)]
#[path = "engine_transaction_catalog/view_tests.rs"]
mod view_tests;

#[cfg(test)]
#[path = "engine_transaction_catalog/view_lifecycle_tests.rs"]
mod view_lifecycle_tests;

#[cfg(test)]
#[path = "engine_transaction_catalog/index_lifecycle_tests.rs"]
mod index_lifecycle_tests;

#[cfg(test)]
#[path = "engine_transaction_catalog/sequence_lifecycle_tests.rs"]
mod sequence_lifecycle_tests;

#[cfg(test)]
#[path = "engine_transaction_catalog/index_owner_generation_tests.rs"]
mod index_owner_generation_tests;

#[cfg(test)]
#[path = "engine_transaction_catalog/core_tests.rs"]
mod tests;
