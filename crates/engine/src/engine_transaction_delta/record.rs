//! Resolved row-transaction identity allocation and durable record construction.

use super::*;

impl Engine {
    pub(crate) fn transaction_insert_identities(
        deltas: &[WriteDelta],
    ) -> Result<BTreeSet<(String, u64)>, ExecuteError> {
        let mut identities = BTreeSet::new();
        for delta in deltas {
            let PreparedMutation::Insert {
                table,
                inserted_rows,
                ..
            } = &delta.mutation
            else {
                continue;
            };
            let prefix = relational_key_prefix(table);
            for (key, _) in inserted_rows {
                let row_id = crate::engine_residency::parse_relational_row_id(key, &prefix)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "transaction INSERT lost provisional entity identity".to_string(),
                        ))
                    })?;
                if !identities.insert((table.clone(), row_id)) {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction reused a provisional entity identity".to_string(),
                    )));
                }
            }
        }
        Ok(identities)
    }

    pub(crate) fn resolved_transaction_record(
        all_deltas: &[WriteDelta],
        final_deltas: &[WriteDelta],
        provisional_inserts: &BTreeSet<(String, u64)>,
        final_base: u64,
        allocator_high_water: u64,
        sequence_value_references: &[BinarySequenceValueReference],
    ) -> Result<BinaryTransactionRecord, ExecuteError> {
        let final_ids = provisional_inserts
            .iter()
            .cloned()
            .zip(final_base..)
            .collect::<BTreeMap<_, _>>();
        let mut final_ids_by_provisional = BTreeMap::new();
        for ((_, provisional), final_id) in &final_ids {
            if final_ids_by_provisional
                .insert(*provisional, *final_id)
                .is_some()
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "transaction reused one provisional entity identity across relations"
                        .to_string(),
                )));
            }
        }
        let mut sequence_value_references = sequence_value_references.to_vec();
        for reference in sequence_value_references
            .iter_mut()
            .filter(|reference| reference.default_expression)
        {
            reference.row_id = final_ids_by_provisional
                .get(&reference.row_id)
                .copied()
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "sequence default reference {} lost its provisional INSERT identity",
                        reference.transition_txn_id
                    )))
                })?;
        }
        let mut mutations = Vec::new();
        let mut sequence_advances = BTreeMap::new();
        for delta in all_deltas {
            if let PreparedMutation::Insert { seq_advances, .. } = &delta.mutation {
                sequence_advances.extend(
                    seq_advances
                        .iter()
                        .map(|(sequence, state)| (sequence.clone(), *state)),
                );
            }
        }
        for delta in final_deltas {
            match &delta.mutation {
                PreparedMutation::Insert {
                    table,
                    inserted_rows,
                    ..
                } => {
                    let prefix = relational_key_prefix(table);
                    for (key, row) in inserted_rows {
                        let provisional =
                            crate::engine_residency::parse_relational_row_id(key, &prefix)
                                .ok_or_else(|| {
                                    ExecuteError::Engine(EngineError::ApplyFailed(
                                        "transaction INSERT lost entity identity".to_string(),
                                    ))
                                })?;
                        let row_id = *final_ids
                            .get(&(table.clone(), provisional))
                            .expect("insert identity map covers every inserted row");
                        mutations.push(BinaryTransactionMutation::Insert {
                            table: table.clone(),
                            row_id,
                            row_encoded: encode_relational_row(row),
                        });
                    }
                }
                PreparedMutation::Update {
                    table,
                    installs,
                    updated_old_rows,
                    ..
                } => {
                    if installs.len() != updated_old_rows.len() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "transaction UPDATE is not a resolved resident mutation".to_string(),
                        )));
                    }
                    let prefix = relational_key_prefix(table);
                    for ((_, key, new_row), old_row) in installs.iter().zip(updated_old_rows) {
                        let provisional =
                            crate::engine_residency::parse_relational_row_id(key, &prefix)
                                .ok_or_else(|| {
                                    ExecuteError::Engine(EngineError::ApplyFailed(
                                        "transaction UPDATE lost entity identity".to_string(),
                                    ))
                                })?;
                        let row_id = final_ids
                            .get(&(table.clone(), provisional))
                            .copied()
                            .unwrap_or(provisional);
                        mutations.push(BinaryTransactionMutation::Update {
                            table: table.clone(),
                            row_id,
                            old_row_encoded: encode_relational_row(old_row),
                            new_row_encoded: encode_relational_row(new_row),
                        });
                    }
                }
                PreparedMutation::Delete {
                    table,
                    deleted_rows,
                    ..
                } => {
                    if delta.write_set.rows.len() != deleted_rows.len() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "transaction DELETE is not a resolved resident mutation".to_string(),
                        )));
                    }
                    let prefix = relational_key_prefix(table);
                    for (key, old_row) in delta.write_set.rows.iter().zip(deleted_rows) {
                        let provisional =
                            crate::engine_residency::parse_relational_row_id(&key.row_key, &prefix)
                                .ok_or_else(|| {
                                    ExecuteError::Engine(EngineError::ApplyFailed(
                                        "transaction DELETE lost entity identity".to_string(),
                                    ))
                                })?;
                        let row_id = final_ids
                            .get(&(table.clone(), provisional))
                            .copied()
                            .unwrap_or(provisional);
                        mutations.push(BinaryTransactionMutation::Delete {
                            table: table.clone(),
                            row_id,
                            old_row_encoded: encode_relational_row(old_row),
                        });
                    }
                }
            }
        }
        let mutations = Self::coalesce_transaction_mutations(mutations)?;
        Ok(BinaryTransactionRecord {
            // Row-only opcodes predate the catalog epoch. Ordered WAL binding upgrades this to
            // `IndexIdentityV1` iff the transaction actually carries catalog commands.
            catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
            allocator_high_water,
            catalog_commands: Vec::new(),
            created_table_identities: BTreeMap::new(),
            created_table_index_identities: BTreeMap::new(),
            catalog_output: None,
            view_operations: Vec::new(),
            view_lifecycle_operations: Vec::new(),
            index_lifecycle_operations: Vec::new(),
            sequence_lifecycle_operations: Vec::new(),
            sequence_reset_operations: Vec::new(),
            sequence_advances_by_oid: BTreeMap::new(),
            operation_order: Vec::new(),
            statement_digests: Vec::new(),
            sequence_input_oids: BTreeMap::new(),
            sequence_value_references,
            table_resets: Vec::new(),
            sequence_advances,
            table_identities: BTreeMap::new(),
            mutations,
        })
    }
}
