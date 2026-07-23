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
        record.catalog_commands.clear();
        record.created_table_identities.clear();
        for staged in catalog_commands {
            record
                .catalog_commands
                .push(BinaryTransactionCatalogCommand {
                    ordinal: staged.ordinal,
                    command: staged.command.clone(),
                });
            let create = match &staged.command {
                Command::CreateTable(create) => create,
                _ => unreachable!("transactional catalog staging supports CREATE TABLE"),
            };
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
        }

        if !catalog_commands.is_empty() {
            let mut created_sequence_oids = BTreeMap::new();
            for staged in catalog_commands {
                let create = match &staged.command {
                    Command::CreateTable(create) => create,
                    _ => unreachable!("transactional catalog staging supports CREATE TABLE"),
                };
                for column in &create.columns {
                    let Some(ColumnDefault::SequenceNextVal {
                        sequence,
                        create_if_missing: true,
                    }) = &column.default
                    else {
                        continue;
                    };
                    let sequence_oid = transaction_catalog
                        .relational_sequences
                        .get(sequence)
                        .map(|sequence| sequence.oid)
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                                "transaction-created sequence \"{sequence}\" left its private catalog before WAL binding"
                            ))
                        })?;
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

                let mut sequence_names = BTreeSet::new();
                match operation {
                    TransactionOperation::Catalog(staged) => {
                        let create = match &staged.command {
                            Command::CreateTable(create) => create,
                            _ => {
                                unreachable!("transactional catalog staging supports CREATE TABLE")
                            }
                        };
                        for column in &create.columns {
                            if let Some(ColumnDefault::SequenceNextVal { sequence, .. }) =
                                &column.default
                            {
                                sequence_names.insert(sequence.as_str());
                            }
                        }
                    }
                    TransactionOperation::Row(staged) => {
                        if let PreparedMutation::Insert { seq_advances, .. } = &staged.mutation {
                            sequence_names.extend(seq_advances.keys().map(String::as_str));
                        }
                    }
                    TransactionOperation::TableReset(_) => {}
                }
                for sequence in sequence_names {
                    let oid = transaction_catalog
                        .relational_sequences
                        .get(sequence)
                        .map(|sequence| sequence.oid)
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                                "transaction sequence input \"{sequence}\" left its private catalog before WAL binding"
                            ))
                        })?;
                    record
                        .sequence_input_oids
                        .insert((ordinal, sequence.to_string()), oid);
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
