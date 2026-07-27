//! Canonical binding of the transaction-private operation stream to one ordered WAL record.

use super::*;

impl Engine {
    pub(super) fn bind_transaction_record_catalog_envelope(
        record: &mut BinaryTransactionRecord,
        operations: &[TransactionOperation],
        catalog_commands: &[StagedCatalogCommand],
        table_resets: &[StagedTableReset],
        transaction_catalog: &CatalogSnapshot,
    ) -> Result<(), ExecuteError> {
        let sequence_reset_record = operations.iter().any(|operation| {
            matches!(
                operation,
                TransactionOperation::TableReset(reset)
                    if reset.sequence_reset_identity.is_some()
            )
        });
        let index_catalog_record = catalog_commands.iter().any(|staged| {
            staged.index_epoch_transition
                || command_requires_index_catalog_opcode(&staged.command)
                || command_is_sequence_lifecycle(&staged.command)
        }) || sequence_reset_record;
        record.catalog_epoch = if index_catalog_record {
            BinaryTransactionCatalogEpoch::IndexIdentityV1
        } else {
            BinaryTransactionCatalogEpoch::Legacy
        };
        record.catalog_commands.clear();
        record.created_table_identities.clear();
        record.created_table_index_identities.clear();
        record.view_operations.clear();
        record.view_lifecycle_operations.clear();
        record.index_lifecycle_operations.clear();
        record.sequence_lifecycle_operations.clear();
        record.sequence_reset_operations.clear();
        record.sequence_advances_by_oid.clear();
        let view_lifecycle_record = catalog_commands.iter().any(|staged| {
            matches!(
                &staged.command,
                Command::RenameView(_) | Command::DropView(_)
            )
        });
        for staged in catalog_commands {
            record
                .catalog_commands
                .push(BinaryTransactionCatalogCommand {
                    ordinal: staged.ordinal,
                    command: staged.command.clone(),
                });
            match &staged.command {
                Command::CreateTable(create) => {
                    let table = transaction_catalog
                        .relational_catalog
                        .get(&create.table)
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                                "transaction-created relation \"{}\" left its private catalog before WAL binding",
                                create.table
                            ))
                        })?;
                    record.created_table_identities.insert(
                        create.table.clone(),
                        BinaryTransactionTableIdentity {
                            table_oid: table.oid,
                            schema_digest: table_schema_digest(table)?,
                        },
                    );
                    if index_catalog_record {
                        record.created_table_index_identities.insert(
                            create.table.clone(),
                            Self::transaction_created_table_implicit_index_identities(
                                transaction_catalog,
                                &create.table,
                            )?,
                        );
                    }
                    if staged.view_identity.is_some()
                        || staged.index_identity.is_some()
                        || staged.sequence_identity.is_some()
                    {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "transactional CREATE TABLE carries a lifecycle identity".to_string(),
                        )));
                    }
                }
                command if command_is_view_lifecycle(command) => {
                    let identity = staged.view_identity.clone().ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "transactional stored-view operation lost its typed identity closure"
                                .to_string(),
                        ))
                    })?;
                    if staged.index_identity.is_some() || staged.sequence_identity.is_some() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "transactional stored-view operation carries another catalog-family identity"
                                .to_string(),
                        )));
                    }
                    if index_catalog_record || view_lifecycle_record {
                        record.view_lifecycle_operations.push(identity);
                    } else {
                        let Command::CreateView(create) = command else {
                            unreachable!("lifecycle record classification checked above");
                        };
                        record
                            .view_operations
                            .push(legacy_from_lifecycle_create(create, &identity).ok_or_else(
                                || {
                                    ExecuteError::Engine(EngineError::ApplyFailed(
                                        "transactional CREATE VIEW identity cannot use the accepted legacy layout"
                                            .to_string(),
                                    ))
                                },
                            )?);
                    }
                }
                command if command_is_index_lifecycle(command) => {
                    if staged.view_identity.is_some() || staged.sequence_identity.is_some() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "transactional index operation carries another catalog-family identity"
                                .to_string(),
                        )));
                    }
                    let identity = staged.index_identity.clone().ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "transactional index operation lost its typed identity closure"
                                .to_string(),
                        ))
                    })?;
                    if !valid_index_lifecycle_operation_identity(command, &identity) {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "transactional index operation carries a noncanonical identity"
                                .to_string(),
                        )));
                    }
                    record.index_lifecycle_operations.push(identity);
                }
                command if command_is_sequence_lifecycle(command) => {
                    if staged.view_identity.is_some() || staged.index_identity.is_some() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "transactional sequence operation carries another catalog-family identity"
                                .to_string(),
                        )));
                    }
                    let identity = staged.sequence_identity.clone().ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "transactional sequence operation lost its typed identity closure"
                                .to_string(),
                        ))
                    })?;
                    if !valid_sequence_lifecycle_operation_identity(command, &identity) {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "transactional sequence operation carries a noncanonical identity"
                                .to_string(),
                        )));
                    }
                    record.sequence_lifecycle_operations.push(identity);
                }
                _ => unreachable!("transactional catalog staging admitted an unsupported family"),
            }
        }

        record
            .sequence_reset_operations
            .extend(operations.iter().filter_map(|operation| match operation {
                TransactionOperation::TableReset(reset) => reset.sequence_reset_identity.clone(),
                TransactionOperation::Catalog(_) | TransactionOperation::Row(_) => None,
            }));
        if !catalog_commands.is_empty() || sequence_reset_record {
            let mut created_sequence_oids = BTreeMap::new();
            for staged in catalog_commands {
                let Command::CreateTable(create) = &staged.command else {
                    continue;
                };
                for column in &create.columns {
                    let Some(ColumnDefault::SequenceNextVal {
                        sequence,
                        create_if_missing: true,
                    }) = &column.default
                    else {
                        continue;
                    };
                    let sequence_oid = staged
                        .sequence_input_oids
                        .get(sequence)
                        .copied()
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "transaction-created sequence \"{sequence}\" lost its statement-local stable identity"
                            )))
                        })?;
                    if !transaction_catalog
                        .relational_sequences
                        .values()
                        .any(|sequence| sequence.oid == sequence_oid)
                    {
                        return Err(ExecuteError::Serialization(format!(
                            "transaction-created sequence \"{sequence}\" stable identity left its private catalog before WAL binding"
                        )));
                    }
                    created_sequence_oids.insert(sequence.clone(), sequence_oid);
                }
            }
            record.catalog_output = Some(BinaryTransactionCatalogOutput {
                relational_next_oid: transaction_catalog.relational_next_oid,
                relational_next_column_id: transaction_catalog.relational_next_column_id,
                created_sequence_oids,
            });

            record.statement_digests.clear();
            record.sequence_input_oids.clear();
            record.operation_order.clear();
            let mut catalog_index = 0u32;
            for (ordinal, operation) in operations.iter().enumerate() {
                let ordinal = u32::try_from(ordinal).map_err(|_| {
                    ExecuteError::Unsupported(
                        "transaction operation count exceeds typed WAL framing".to_string(),
                    )
                })?;
                let statement_digest = match operation {
                    TransactionOperation::Catalog(staged) => staged.statement_digest,
                    TransactionOperation::Row(staged) => staged.statement_digest,
                    TransactionOperation::TableReset(reset) => reset.statement_digest,
                };
                record.statement_digests.push(statement_digest);

                let sequence_inputs = match operation {
                    TransactionOperation::Catalog(staged) => {
                        staged.sequence_input_oids.iter().collect::<Vec<_>>()
                    }
                    TransactionOperation::Row(staged) => {
                        staged.sequence_input_oids.iter().collect::<Vec<_>>()
                    }
                    TransactionOperation::TableReset(_) => Vec::new(),
                };
                for (sequence, oid) in sequence_inputs {
                    record
                        .sequence_input_oids
                        .insert((ordinal, sequence.clone()), *oid);
                }
                let sequence_record = sequence_reset_record
                    || catalog_commands
                        .iter()
                        .any(|staged| command_is_sequence_lifecycle(&staged.command));
                if sequence_record {
                    match operation {
                        TransactionOperation::Row(staged) => {
                            if let PreparedMutation::Insert { seq_advances, .. } = &staged.mutation
                            {
                                for (sequence, state) in seq_advances {
                                    let oid =
                                        staged.sequence_input_oids.get(sequence).ok_or_else(|| {
                                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                                "transaction INSERT sequence input \"{sequence}\" lost its stable identity"
                                            )))
                                        })?;
                                    record.sequence_advances_by_oid.insert(*oid, *state);
                                }
                            }
                        }
                        TransactionOperation::Catalog(staged) => {
                            if let Command::SequenceRestart(restart) = &staged.command {
                                let identity =
                                    staged.sequence_identity.as_ref().ok_or_else(|| {
                                        ExecuteError::Engine(EngineError::ApplyFailed(
                                            "sequence lifecycle state barrier lost its identity"
                                                .to_string(),
                                        ))
                                    })?;
                                for target in &identity.targets {
                                    let stable = target
                                        .target_after
                                        .as_ref()
                                        .or(target.target_before.as_ref())
                                        .ok_or_else(|| {
                                            ExecuteError::Engine(EngineError::ApplyFailed(
                                                "sequence restart state barrier lost its stable target"
                                                    .to_string(),
                                            ))
                                        })?;
                                    if let Some(state) =
                                        record.sequence_advances_by_oid.get_mut(&stable.oid)
                                    {
                                        *state = (restart.value, false);
                                    }
                                }
                            }
                        }
                        TransactionOperation::TableReset(reset) => {
                            if let Some(identity) = &reset.sequence_reset_identity {
                                for target in &identity.targets {
                                    let stable = target
                                        .target_after
                                        .as_ref()
                                        .or(target.target_before.as_ref())
                                        .ok_or_else(|| {
                                            ExecuteError::Engine(EngineError::ApplyFailed(
                                                "sequence reset state barrier lost its stable target"
                                                    .to_string(),
                                            ))
                                        })?;
                                    if let Some(state) =
                                        record.sequence_advances_by_oid.get_mut(&stable.oid)
                                    {
                                        *state = (1, false);
                                    }
                                }
                            }
                        }
                    }
                }

                let identity = match operation {
                    TransactionOperation::Catalog(_) => {
                        let identity = BinaryTransactionOperationIdentity::Catalog {
                            command_index: catalog_index,
                        };
                        catalog_index = catalog_index.checked_add(1).ok_or_else(|| {
                            ExecuteError::Unsupported(
                                "transaction catalog operation count exceeds typed WAL framing"
                                    .to_string(),
                            )
                        })?;
                        identity
                    }
                    TransactionOperation::Row(delta) => match &delta.mutation {
                        PreparedMutation::Insert { table, .. } => {
                            BinaryTransactionOperationIdentity::Insert {
                                table: table.clone(),
                            }
                        }
                        PreparedMutation::Update { table, .. } => {
                            BinaryTransactionOperationIdentity::Update {
                                table: table.clone(),
                            }
                        }
                        PreparedMutation::Delete { table, .. } => {
                            BinaryTransactionOperationIdentity::Delete {
                                table: table.clone(),
                            }
                        }
                    },
                    TransactionOperation::TableReset(reset) => {
                        BinaryTransactionOperationIdentity::TableReset {
                            table: reset.table.clone(),
                        }
                    }
                };
                record.operation_order.push(identity);
            }
            if !record.sequence_advances_by_oid.is_empty()
                || sequence_reset_record
                || catalog_commands
                    .iter()
                    .any(|staged| command_is_sequence_lifecycle(&staged.command))
            {
                record.sequence_advances.clear();
            }
        }

        record.table_resets.clear();
        record
            .table_resets
            .extend(table_resets.iter().map(StagedTableReset::to_binary));
        let mut identity_table_names = BTreeSet::new();
        if catalog_commands.is_empty() {
            for mutation in &record.mutations {
                let table = match mutation {
                    BinaryTransactionMutation::Insert { table, .. }
                    | BinaryTransactionMutation::Update { table, .. }
                    | BinaryTransactionMutation::Delete { table, .. } => table,
                };
                identity_table_names.insert(table.clone());
            }
        } else {
            for operation in &record.operation_order {
                let table = match operation {
                    BinaryTransactionOperationIdentity::Insert { table }
                    | BinaryTransactionOperationIdentity::Update { table }
                    | BinaryTransactionOperationIdentity::Delete { table } => table,
                    BinaryTransactionOperationIdentity::Catalog { .. }
                    | BinaryTransactionOperationIdentity::TableReset { .. } => continue,
                };
                if !record.created_table_identities.contains_key(table) {
                    identity_table_names.insert(table.clone());
                }
            }
        }

        record.table_identities.clear();
        for table_name in identity_table_names {
            let table = transaction_catalog
                .relational_catalog
                .get(&table_name)
                .ok_or_else(|| {
                    ExecuteError::Serialization(format!(
                        "transaction mutation relation \"{table_name}\" left its catalog before WAL binding"
                    ))
                })?;
            record.table_identities.insert(
                table_name,
                BinaryTransactionTableIdentity {
                    table_oid: table.oid,
                    schema_digest: table_schema_digest(table)?,
                },
            );
        }
        Ok(())
    }
}
