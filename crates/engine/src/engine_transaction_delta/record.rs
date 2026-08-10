//! Resolved row-transaction identity allocation and durable record construction.

use super::*;

impl Engine {
    pub(crate) fn resolved_insert_identities(
        provisional_inserts: &BTreeSet<(String, u64)>,
        final_base: u64,
    ) -> Result<BTreeMap<(String, u64), u64>, ExecuteError> {
        provisional_inserts
            .iter()
            .cloned()
            .enumerate()
            .map(|(ordinal, identity)| {
                let ordinal = u64::try_from(ordinal).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction INSERT identity count exceeds u64".to_string(),
                    ))
                })?;
                let final_id = final_base.checked_add(ordinal).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction row identity space exhausted".to_string(),
                    ))
                })?;
                Ok((identity, final_id))
            })
            .collect()
    }

    /// Bind the existing sequence-reference facts to the final allocator range without first
    /// constructing a legacy binary transaction record. Codec-5 carries these same references in
    /// S2/S7; legacy record construction calls this helper as well so the binding remains single.
    pub(crate) fn resolved_sequence_value_references(
        provisional_inserts: &BTreeSet<(String, u64)>,
        final_base: u64,
        sequence_value_references: &[BinarySequenceValueReference],
    ) -> Result<Vec<BinarySequenceValueReference>, ExecuteError> {
        let final_ids = Self::resolved_insert_identities(provisional_inserts, final_base)?;
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
        let mut resolved = sequence_value_references.to_vec();
        for reference in resolved
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
        Ok(resolved)
    }

    /// Collect provisional INSERT identities from the ordered transaction stream.  Typed INSERT
    /// contributes its already-bound ids directly and never reconstructs a legacy row key.  The
    /// generic row-operation stream is UPDATE/DELETE-only after the WRITE-001 cutover.
    pub(crate) fn transaction_operation_insert_identities(
        operations: &[TransactionOperation],
    ) -> Result<BTreeSet<(String, u64)>, ExecuteError> {
        if operations.iter().any(|operation| {
            matches!(
                operation,
                TransactionOperation::Row(staged)
                    if matches!(staged.delta.mutation, PreparedMutation::Insert { .. })
            )
        }) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "legacy INSERT reached the UPDATE/DELETE transaction operation stream after codec-5 cutover"
                    .to_string(),
            )));
        }
        let mut identities = BTreeSet::new();
        for staged in operations.iter().filter_map(|operation| match operation {
            TransactionOperation::TypedInsert(staged) => Some(staged),
            TransactionOperation::Catalog(_)
            | TransactionOperation::Row(_)
            | TransactionOperation::TableReset(_) => None,
        }) {
            for row_id in staged.provisional_row_ids.iter().copied() {
                if !identities.insert((staged.table.clone(), row_id)) {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction reused a provisional entity identity".to_string(),
                    )));
                }
            }
        }
        Ok(identities)
    }

    pub(crate) fn resolved_transaction_record(
        final_rows: &[crate::engine_transaction_reset::FinalTransactionRowOperation],
        provisional_inserts: &BTreeSet<(String, u64)>,
        final_base: u64,
        allocator_high_water: u64,
        sequence_value_references: &[BinarySequenceValueReference],
    ) -> Result<BinaryTransactionRecord, ExecuteError> {
        let final_ids = Self::resolved_insert_identities(provisional_inserts, final_base)?;
        let sequence_value_references = Self::resolved_sequence_value_references(
            provisional_inserts,
            final_base,
            sequence_value_references,
        )?;
        let mut mutations = Vec::new();
        for staged in final_rows {
            match staged {
                // Codec-5 owns every typed INSERT image in S2/S4/S7. This helper now resolves
                // only the remaining pre-existing-row UPDATE/DELETE facts for its one S3
                // composition; the non-codec terminal still calls it only when no typed insert
                // exists. Never reconstruct a typed INSERT as a binary mutation here.
                crate::engine_transaction_reset::FinalTransactionRowOperation::TypedInsert(_) => {}
                crate::engine_transaction_reset::FinalTransactionRowOperation::Legacy(staged) => {
                    let delta = &staged.delta;
                    match &delta.mutation {
                        PreparedMutation::Insert { .. } => {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "legacy INSERT reached the resolved binary transaction encoder after codec-5 cutover"
                                    .to_string(),
                            )));
                        }
                        PreparedMutation::Update {
                            table,
                            installs,
                            updated_old_rows,
                            ..
                        } => {
                            if installs.len() != updated_old_rows.len() {
                                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                    "transaction UPDATE is not a resolved resident mutation"
                                        .to_string(),
                                )));
                            }
                            let prefix = relational_key_prefix(table);
                            for ((_, key, new_row), old_row) in
                                installs.iter().zip(updated_old_rows)
                            {
                                let provisional =
                                    crate::engine_residency::parse_relational_row_id(key, &prefix)
                                        .ok_or_else(|| {
                                            ExecuteError::Engine(EngineError::ApplyFailed(
                                                "transaction UPDATE lost entity identity"
                                                    .to_string(),
                                            ))
                                        })?;
                                if provisional_inserts.contains(&(table.clone(), provisional)) {
                                    // A later UPDATE of a transaction-private INSERT is already
                                    // folded into the codec-5 final image. Emitting it here would
                                    // create a second row authority after the allocator rebinding.
                                    continue;
                                }
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
                                    "transaction DELETE is not a resolved resident mutation"
                                        .to_string(),
                                )));
                            }
                            let prefix = relational_key_prefix(table);
                            for (key, old_row) in delta.write_set.rows.iter().zip(deleted_rows) {
                                let provisional = crate::engine_residency::parse_relational_row_id(
                                    &key.row_key,
                                    &prefix,
                                )
                                .ok_or_else(|| {
                                    ExecuteError::Engine(EngineError::ApplyFailed(
                                        "transaction DELETE lost entity identity".to_string(),
                                    ))
                                })?;
                                if provisional_inserts.contains(&(table.clone(), provisional)) {
                                    // INSERT -> DELETE vanishes from the S2 final image and must
                                    // not reappear as a binary DELETE after codec-5 publication.
                                    continue;
                                }
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
            sequence_advances: BTreeMap::new(),
            table_identities: BTreeMap::new(),
            mutations,
        })
    }
}
