//! W5a — BINARY row-op WAL records (the resolved-change-record format, assessment D5/R3).
//!
//! The WAL was a logical SQL-text log: replay re-parsed and re-executed every statement
//! (~30µs/record), the sequencer's covered INSERTs re-serialized their statement text, and every
//! checkpoint carried the full SQL. A binary record carries the RESOLVED mutation instead — the
//! table, the row ids, and the row images in the tuple store's canonical cell encoding — so
//! replay is decode + install (no parse, no re-resolve, no validation re-run: the record exists
//! only because the original commit validated it).
//!
//! FRAMING: the record rides the existing WAL framing (txn_id / len / FNV checksum) as an opaque
//! payload. `payload[0] == 0xFF` marks a binary record — 0xFF is an invalid UTF-8 leading byte,
//! so every SQL-text consumer's defensive `from_utf8 -> skip` arm (KvStateMachine, telemetry)
//! ignores binary records WITHOUT modification; the consumers that must apply them
//! (`apply_mvcc_entry`, `residency_invalidation_scope`) dispatch on the tag explicitly BEFORE
//! their UTF-8 checks. SQL text can never collide (it starts with printable ASCII).
//!
//! v1 (W5a) began with the covered single-table INSERT class; W5b added UPDATE/DELETE, and R3-003
//! added one resolved explicit-transaction operation carrying ordered row mutations plus atomic
//! sequence post-state. PRODUCT-001 extends that same operation with typed transaction-owned
//! catalog mutations; old row-only records remain byte-for-byte v1 compatible. Unsupported
//! autocommit shapes remain SQL text.

use super::*;

mod index_identity_codec;
mod operation_identity;
mod record_decode;
mod row_codec;
mod sequence_identity_codec;
mod transaction_types;
mod view_identity_codec;
use index_identity_codec::*;
pub(crate) use index_identity_codec::{
    command_is_index_lifecycle, command_requires_index_catalog_opcode,
    valid_index_lifecycle_operation_identity, BinaryCatalogIndexIdentity,
    BinaryTransactionIndexLifecycleOperationIdentity,
    BinaryTransactionIndexLifecycleTargetIdentity,
};
pub(crate) use operation_identity::BinaryTransactionOperationIdentity;
pub(crate) use record_decode::decode_binary_record;
pub(crate) use row_codec::*;
use sequence_identity_codec::*;
pub(crate) use sequence_identity_codec::{
    command_is_sequence_lifecycle, generated_sequence_names, generated_sequence_output_from_inputs,
    generated_sequence_output_matches_inputs, valid_sequence_lifecycle_operation_identity,
    valid_sequence_reset_operation_identity, BinarySequenceColumnDependencyIdentity,
    BinaryTransactionSequenceLifecycleOperationIdentity,
    BinaryTransactionSequenceLifecycleTargetIdentity,
    BinaryTransactionSequenceResetOperationIdentity,
};
pub(crate) use transaction_types::{
    BinaryTransactionCatalogCommand, BinaryTransactionCatalogEpoch, BinaryTransactionCatalogOutput,
    BinaryTransactionMutation, BinaryTransactionRecord, BinaryTransactionTableIdentity,
    BinaryTransactionTableReset, BinaryWalRecord, WAL_BINARY_TAG,
};
use transaction_types::{
    OP_COMPOSITE_TRANSACTION, OP_DELETE_BY_KEY, OP_IDENTITY_COMPOSITE_TRANSACTION,
    OP_IDENTITY_ORDERED_CATALOG_TRANSACTION,
    OP_IDENTITY_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION,
    OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION, OP_IDENTITY_TABLE_RESET_TRANSACTION,
    OP_IDENTITY_TRANSACTION, OP_INSERT, OP_ORDERED_CATALOG_TRANSACTION,
    OP_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION, OP_ORDERED_CATALOG_VIEW_TRANSACTION,
    OP_TABLE_RESET_TRANSACTION, OP_TRANSACTION, OP_UPDATE_BY_KEY, TXN_DELETE, TXN_INSERT,
    TXN_OPERATION_CATALOG, TXN_OPERATION_DELETE, TXN_OPERATION_INSERT, TXN_OPERATION_TABLE_RESET,
    TXN_OPERATION_UPDATE, TXN_UPDATE, WAL_BINARY_VERSION,
};
use view_identity_codec::*;
pub(crate) use view_identity_codec::{
    command_is_view_lifecycle, legacy_from_lifecycle_create, lifecycle_from_legacy_create,
    valid_view_lifecycle_operation_identity, BinaryCatalogRelationIdentity,
    BinaryCatalogRelationKind, BinaryTransactionViewLifecycleOperationIdentity,
    BinaryTransactionViewLifecycleTargetIdentity, BinaryTransactionViewOperationIdentity,
};

/// Encode one resolved explicit transaction as ONE WAL payload. Width overflow is reported as
/// `None`; callers must fail the transaction rather than fall back to statement SQL records, which
/// would lose atomicity and predicate-resolution identity.
pub(crate) fn try_encode_binary_transaction(record: &BinaryTransactionRecord) -> Option<Vec<u8>> {
    let reset_names = record
        .table_resets
        .iter()
        .map(|reset| reset.table.as_str())
        .collect::<BTreeSet<_>>();
    let mutation_names = record
        .mutations
        .iter()
        .map(|mutation| match mutation {
            BinaryTransactionMutation::Insert { table, .. }
            | BinaryTransactionMutation::Update { table, .. }
            | BinaryTransactionMutation::Delete { table, .. } => table.as_str(),
        })
        .collect::<BTreeSet<_>>();
    let identity_names = record
        .table_identities
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let identity_bound = !record.table_identities.is_empty();
    let mut catalog_names = BTreeMap::<&str, u32>::new();
    let mut view_commands = Vec::new();
    let mut index_commands = Vec::new();
    let mut sequence_commands = Vec::new();
    let mut requires_view_lifecycle_opcode = false;
    let mut created_sequence_names = BTreeSet::new();
    let mut prior_catalog_ordinal = None;
    for (command_index, operation) in record.catalog_commands.iter().enumerate() {
        if prior_catalog_ordinal.is_some_and(|prior| operation.ordinal <= prior) {
            return None;
        }
        prior_catalog_ordinal = Some(operation.ordinal);
        match &operation.command {
            Command::CreateTable(create) => {
                if catalog_names
                    .insert(create.table.as_str(), operation.ordinal)
                    .is_some()
                {
                    return None;
                }
                for sequence in create
                    .columns
                    .iter()
                    .filter_map(|column| match &column.default {
                        Some(ColumnDefault::SequenceNextVal {
                            sequence,
                            create_if_missing: true,
                        }) => Some(sequence.as_str()),
                        _ => None,
                    })
                {
                    if !created_sequence_names.insert(sequence) {
                        return None;
                    }
                }
            }
            command if command_is_view_lifecycle(command) => {
                requires_view_lifecycle_opcode |= command_requires_view_lifecycle_opcode(command);
                view_commands.push((
                    u32::try_from(command_index).ok()?,
                    operation.ordinal,
                    command,
                ));
            }
            command if command_is_index_lifecycle(command) => {
                index_commands.push((
                    u32::try_from(command_index).ok()?,
                    operation.ordinal,
                    command,
                ));
            }
            command if command_is_sequence_lifecycle(command) => {
                sequence_commands.push((
                    u32::try_from(command_index).ok()?,
                    operation.ordinal,
                    command,
                ));
            }
            _ => return None,
        }
    }
    let created_identity_names = record
        .created_table_identities
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let created_index_identity_names = record
        .created_table_index_identities
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let ordered_catalog = !record.sequence_reset_operations.is_empty()
        || (!record.catalog_commands.is_empty()
            && (!record.operation_order.is_empty()
                || record.catalog_commands.len() != 1
                || record.catalog_commands[0].ordinal != 0
                || !record.table_resets.is_empty()
                || !record.created_table_identities.is_empty()
                || record.catalog_output.is_some()
                || !record.view_operations.is_empty()
                || !record.view_lifecycle_operations.is_empty()
                || !record.index_lifecycle_operations.is_empty()
                || !record.sequence_lifecycle_operations.is_empty()));
    let view_catalog = !view_commands.is_empty();
    let legacy_view_catalog = !record.view_operations.is_empty();
    let view_lifecycle_catalog = !record.view_lifecycle_operations.is_empty();
    let index_command_catalog = !index_commands.is_empty();
    let index_semantics_required = record
        .catalog_commands
        .iter()
        .any(|operation| command_requires_index_catalog_opcode(&operation.command));
    let index_catalog =
        ordered_catalog && record.catalog_epoch == BinaryTransactionCatalogEpoch::IndexIdentityV1;
    let index_identity_catalog = !record.index_lifecycle_operations.is_empty();
    let sequence_command_catalog = !sequence_commands.is_empty();
    let sequence_identity_catalog = !record.sequence_lifecycle_operations.is_empty()
        || !record.sequence_reset_operations.is_empty();
    let mut created_class_oids = BTreeSet::new();
    let created_index_closure_valid = if index_catalog {
        created_index_identity_names == catalog_names.keys().copied().collect::<BTreeSet<_>>()
            && record.catalog_commands.iter().all(|operation| {
                let Command::CreateTable(create) = &operation.command else {
                    return true;
                };
                let Some(table_identity) = record.created_table_identities.get(&create.table)
                else {
                    return false;
                };
                let Some(index_identities) =
                    record.created_table_index_identities.get(&create.table)
                else {
                    return false;
                };
                let expected_count =
                    usize::from(create.primary_key.is_some()) + create.unique_constraints.len();
                table_identity.table_oid != 0
                    && table_identity.table_oid <= i32::MAX as u32
                    && created_class_oids.insert(table_identity.table_oid)
                    && index_identities.len() == expected_count
                    && index_identities.iter().all(|identity| {
                        identity.oid <= i32::MAX as u32
                            && identity.table_oid == table_identity.table_oid
                            && identity.digest != [0; 32]
                            && created_class_oids.insert(identity.oid)
                    })
            })
    } else {
        record.created_table_index_identities.is_empty()
    };
    if legacy_view_catalog && view_lifecycle_catalog {
        return None;
    }
    if !created_index_closure_valid
        || index_command_catalog != index_identity_catalog
        || sequence_command_catalog != !record.sequence_lifecycle_operations.is_empty()
        || (!record.sequence_advances_by_oid.is_empty() && !sequence_identity_catalog)
        || (sequence_identity_catalog && !index_catalog)
        || (index_semantics_required && !index_catalog)
        || (index_catalog && legacy_view_catalog)
        || (view_catalog
            != if index_catalog || requires_view_lifecycle_opcode {
                view_lifecycle_catalog
            } else {
                legacy_view_catalog
            })
    {
        return None;
    }
    if legacy_view_catalog
        && (record.view_operations.len() != view_commands.len()
            || record.view_operations.iter().zip(&view_commands).any(
                |(identity, (command_index, ordinal, command))| {
                    identity.command_index != *command_index
                        || identity.ordinal != *ordinal
                        || !matches!(command, Command::CreateView(_))
                        || !valid_legacy_view_operation_identity(identity)
                },
            ))
    {
        return None;
    }
    if view_lifecycle_catalog
        && ((!index_catalog && !requires_view_lifecycle_opcode)
            || record.view_lifecycle_operations.len() != view_commands.len()
            || record
                .view_lifecycle_operations
                .iter()
                .zip(&view_commands)
                .any(|(identity, (command_index, ordinal, command))| {
                    identity.command_index != *command_index
                        || identity.ordinal != *ordinal
                        || !valid_view_lifecycle_operation_identity(command, identity)
                }))
    {
        return None;
    }
    if index_command_catalog
        && (record.index_lifecycle_operations.len() != index_commands.len()
            || record
                .index_lifecycle_operations
                .iter()
                .zip(&index_commands)
                .any(|(identity, (command_index, ordinal, command))| {
                    identity.command_index != *command_index
                        || identity.ordinal != *ordinal
                        || !valid_index_lifecycle_operation_identity(command, identity)
                }))
    {
        return None;
    }
    if sequence_command_catalog
        && (record.sequence_lifecycle_operations.len() != sequence_commands.len()
            || record
                .sequence_lifecycle_operations
                .iter()
                .zip(&sequence_commands)
                .any(|(identity, (command_index, ordinal, command))| {
                    identity.command_index != *command_index
                        || identity.ordinal != *ordinal
                        || !valid_sequence_lifecycle_operation_identity(command, identity)
                }))
    {
        return None;
    }
    let mut sequence_resets_by_ordinal = BTreeMap::new();
    for identity in &record.sequence_reset_operations {
        if !valid_sequence_reset_operation_identity(identity)
            || sequence_resets_by_ordinal
                .insert(identity.ordinal, identity)
                .is_some()
        {
            return None;
        }
    }
    let mut sequence_names_by_ordinal = BTreeMap::<u32, BTreeSet<&str>>::new();
    for ((ordinal, sequence), oid) in &record.sequence_input_oids {
        if sequence.len() > u16::MAX as usize || *oid == 0 {
            return None;
        }
        sequence_names_by_ordinal
            .entry(*ordinal)
            .or_default()
            .insert(sequence);
    }
    let mut ordered_row_names = BTreeSet::new();
    let mut ordered_reset_names = BTreeSet::new();
    if ordered_catalog {
        if !generated_sequence_output_matches_inputs(
            &record.catalog_commands,
            &record.sequence_input_oids,
            record.catalog_output.as_ref()?,
        ) {
            return None;
        }
        if record.operation_order.is_empty()
            || record.statement_digests.len() != record.operation_order.len()
            || record.statement_digests.contains(&[0; 32])
        {
            return None;
        }
        let mut next_catalog_index = 0usize;
        for (ordinal, operation) in record.operation_order.iter().enumerate() {
            let ordinal_u32 = u32::try_from(ordinal).ok()?;
            let statement_digest = record.statement_digests.get(ordinal)?;
            match operation {
                BinaryTransactionOperationIdentity::Catalog { command_index } => {
                    if usize::try_from(*command_index).ok() != Some(next_catalog_index)
                        || record
                            .catalog_commands
                            .get(next_catalog_index)
                            .is_none_or(|command| {
                                usize::try_from(command.ordinal).ok() != Some(ordinal)
                            })
                    {
                        return None;
                    }
                    let command = &record.catalog_commands.get(next_catalog_index)?.command;
                    if transaction_statement_digest(command).ok().as_ref() != Some(statement_digest)
                    {
                        return None;
                    }
                    let expected_sequences = match command {
                        Command::CreateTable(create) => create
                            .columns
                            .iter()
                            .filter_map(|column| match &column.default {
                                Some(ColumnDefault::SequenceNextVal { sequence, .. }) => {
                                    Some(sequence.as_str())
                                }
                                _ => None,
                            })
                            .collect::<BTreeSet<_>>(),
                        Command::CreateView(_) | Command::RenameView(_) | Command::DropView(_) => {
                            BTreeSet::new()
                        }
                        Command::CreateIndex(_)
                        | Command::RenameIndex(_)
                        | Command::DropIndex(_) => BTreeSet::new(),
                        Command::CreateSequence(_)
                        | Command::SequenceRestart(_)
                        | Command::RenameSequence(_)
                        | Command::DropSequence(_) => BTreeSet::new(),
                        _ => return None,
                    };
                    if sequence_names_by_ordinal
                        .get(&ordinal_u32)
                        .cloned()
                        .unwrap_or_default()
                        != expected_sequences
                    {
                        return None;
                    }
                    next_catalog_index += 1;
                }
                BinaryTransactionOperationIdentity::Insert { table }
                | BinaryTransactionOperationIdentity::Update { table }
                | BinaryTransactionOperationIdentity::Delete { table } => {
                    if table.len() > u16::MAX as usize
                        || catalog_names
                            .get(table.as_str())
                            .is_some_and(|create_ordinal| {
                                usize::try_from(*create_ordinal)
                                    .ok()
                                    .is_none_or(|created| created >= ordinal)
                            })
                    {
                        return None;
                    }
                    ordered_row_names.insert(table.as_str());
                }
                BinaryTransactionOperationIdentity::TableReset { table } => {
                    if table.len() > u16::MAX as usize
                        || catalog_names
                            .get(table.as_str())
                            .is_some_and(|create_ordinal| {
                                usize::try_from(*create_ordinal)
                                    .ok()
                                    .is_none_or(|created| created >= ordinal)
                            })
                    {
                        return None;
                    }
                    let command = Command::TruncateTable(TruncateTable {
                        name: table.clone(),
                        restart_identity: sequence_resets_by_ordinal
                            .get(&ordinal_u32)
                            .is_some_and(|identity| identity.table == *table),
                    });
                    if transaction_statement_digest(&command).ok().as_ref()
                        != Some(statement_digest)
                    {
                        return None;
                    }
                    ordered_reset_names.insert(table.as_str());
                }
            }
            if !matches!(
                operation,
                BinaryTransactionOperationIdentity::Catalog { .. }
                    | BinaryTransactionOperationIdentity::Insert { .. }
            ) && sequence_names_by_ordinal.contains_key(&ordinal_u32)
            {
                return None;
            }
        }
        if next_catalog_index != record.catalog_commands.len() {
            return None;
        }
        if sequence_names_by_ordinal.keys().any(|ordinal| {
            usize::try_from(*ordinal)
                .ok()
                .is_none_or(|ordinal| ordinal >= record.operation_order.len())
        }) {
            return None;
        }
        let insert_sequence_names = record
            .sequence_input_oids
            .keys()
            .filter_map(|(ordinal, sequence)| {
                usize::try_from(*ordinal)
                    .ok()
                    .and_then(|ordinal| record.operation_order.get(ordinal))
                    .and_then(|operation| {
                        matches!(operation, BinaryTransactionOperationIdentity::Insert { .. })
                            .then_some(sequence.as_str())
                    })
            })
            .collect::<BTreeSet<_>>();
        if sequence_identity_catalog {
            let insert_sequence_oids = record
                .sequence_input_oids
                .iter()
                .filter_map(|((ordinal, _), oid)| {
                    usize::try_from(*ordinal)
                        .ok()
                        .and_then(|ordinal| record.operation_order.get(ordinal))
                        .and_then(|operation| {
                            matches!(operation, BinaryTransactionOperationIdentity::Insert { .. })
                                .then_some(*oid)
                        })
                })
                .collect::<BTreeSet<_>>();
            if !record.sequence_advances.is_empty()
                || insert_sequence_oids
                    != record
                        .sequence_advances_by_oid
                        .keys()
                        .copied()
                        .collect::<BTreeSet<_>>()
            {
                return None;
            }
        } else if insert_sequence_names
            != record
                .sequence_advances
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
            || !record.sequence_advances_by_oid.is_empty()
        {
            return None;
        }
    } else if !record.statement_digests.is_empty()
        || !record.sequence_input_oids.is_empty()
        || !record.sequence_advances_by_oid.is_empty()
    {
        return None;
    }
    let ordered_existing_row_names = ordered_row_names
        .difference(&catalog_names.keys().copied().collect::<BTreeSet<_>>())
        .copied()
        .collect::<BTreeSet<_>>();
    if record.catalog_commands.len() > u32::MAX as usize
        || record.created_table_identities.len() > u32::MAX as usize
        || record.created_table_index_identities.len() > u32::MAX as usize
        || record
            .created_table_index_identities
            .values()
            .any(|identities| identities.len() > u32::MAX as usize)
        || record.catalog_output.as_ref().is_some_and(|output| {
            output.created_sequence_oids.len() > u32::MAX as usize
                || output
                    .created_sequence_oids
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
                    != created_sequence_names
        })
        || record.operation_order.len() > u32::MAX as usize
        || record.statement_digests.len() > u32::MAX as usize
        || record.sequence_input_oids.len() > u32::MAX as usize
        || record.table_resets.len() > u32::MAX as usize
        || record.view_operations.len() > u32::MAX as usize
        || record.view_lifecycle_operations.len() > u32::MAX as usize
        || record.index_lifecycle_operations.len() > u32::MAX as usize
        || record.sequence_lifecycle_operations.len() > u32::MAX as usize
        || record.sequence_reset_operations.len() > u32::MAX as usize
        || record.sequence_advances_by_oid.len() > u32::MAX as usize
        || record.sequence_advances.len() > u32::MAX as usize
        || record.mutations.len() > u32::MAX as usize
        || (record.catalog_commands.is_empty() && !record.created_table_identities.is_empty())
        || (record.catalog_commands.is_empty() && !record.created_table_index_identities.is_empty())
        || (record.catalog_commands.is_empty()
            && record.sequence_reset_operations.is_empty()
            && record.catalog_output.is_some())
        || (record.catalog_commands.is_empty() && !record.view_operations.is_empty())
        || (record.catalog_commands.is_empty() && !record.view_lifecycle_operations.is_empty())
        || (record.catalog_commands.is_empty() && !record.index_lifecycle_operations.is_empty())
        || (record.catalog_commands.is_empty() && !record.sequence_lifecycle_operations.is_empty())
        || (ordered_catalog
            && created_identity_names != catalog_names.keys().copied().collect::<BTreeSet<_>>())
        || (!ordered_catalog && !record.created_table_identities.is_empty())
        || (!index_catalog && !record.created_table_index_identities.is_empty())
        || (ordered_catalog != record.catalog_output.is_some())
        || (!ordered_catalog && !record.operation_order.is_empty())
        || reset_names.len() != record.table_resets.len()
        || (ordered_catalog
            && (ordered_reset_names != reset_names
                || ordered_existing_row_names != identity_names
                || !mutation_names.is_subset(&ordered_row_names)))
        || (!ordered_catalog && identity_bound && identity_names != mutation_names)
        || record.mutations.iter().any(|mutation| match mutation {
            BinaryTransactionMutation::Insert { .. } => false,
            BinaryTransactionMutation::Update { table, .. }
            | BinaryTransactionMutation::Delete { table, .. } => {
                reset_names.contains(table.as_str())
            }
        })
    {
        return None;
    }
    if ordered_catalog {
        if record.sequence_reset_operations.iter().any(|identity| {
            usize::try_from(identity.ordinal)
                .ok()
                .and_then(|ordinal| record.operation_order.get(ordinal))
                .is_none_or(|operation| {
                    !matches!(
                        operation,
                        BinaryTransactionOperationIdentity::TableReset { table }
                            if table == &identity.table
                    )
                })
        }) {
            return None;
        }
        for reset in &record.table_resets {
            let operation = usize::try_from(reset.ordinal)
                .ok()
                .and_then(|ordinal| record.operation_order.get(ordinal))?;
            if !matches!(operation, BinaryTransactionOperationIdentity::TableReset { table } if table == &reset.table)
            {
                return None;
            }
        }
        for mutation in &record.mutations {
            let table = match mutation {
                BinaryTransactionMutation::Insert { table, .. }
                | BinaryTransactionMutation::Update { table, .. }
                | BinaryTransactionMutation::Delete { table, .. } => table,
            };
            let last_reset = record
                .operation_order
                .iter()
                .rposition(|operation| matches!(operation, BinaryTransactionOperationIdentity::TableReset { table: reset_table } if reset_table == table));
            if !record
                .operation_order
                .iter()
                .enumerate()
                .any(|(ordinal, operation)| {
                    operation.matches_mutation(mutation)
                        && last_reset.is_none_or(|reset_ordinal| ordinal > reset_ordinal)
                })
            {
                return None;
            }
        }
    }
    let mut out = Vec::with_capacity(32 + record.mutations.len() * 96);
    out.push(WAL_BINARY_TAG);
    out.push(WAL_BINARY_VERSION);
    let op = if ordered_catalog {
        if sequence_identity_catalog {
            if identity_bound {
                OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
            } else {
                OP_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
            }
        } else if index_catalog {
            if identity_bound {
                OP_IDENTITY_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            } else {
                OP_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            }
        } else if view_lifecycle_catalog {
            if identity_bound {
                OP_IDENTITY_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            } else {
                OP_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            }
        } else if view_catalog {
            if identity_bound {
                OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
            } else {
                OP_ORDERED_CATALOG_VIEW_TRANSACTION
            }
        } else if identity_bound {
            OP_IDENTITY_ORDERED_CATALOG_TRANSACTION
        } else {
            OP_ORDERED_CATALOG_TRANSACTION
        }
    } else {
        match (
            identity_bound,
            !record.table_resets.is_empty(),
            record.catalog_commands.is_empty(),
        ) {
            (true, true, _) => OP_IDENTITY_TABLE_RESET_TRANSACTION,
            (true, false, true) => OP_IDENTITY_TRANSACTION,
            (true, false, false) => OP_IDENTITY_COMPOSITE_TRANSACTION,
            (false, true, _) => OP_TABLE_RESET_TRANSACTION,
            (false, false, true) => OP_TRANSACTION,
            (false, false, false) => OP_COMPOSITE_TRANSACTION,
        }
    };
    out.push(op);
    if matches!(
        op,
        OP_COMPOSITE_TRANSACTION | OP_IDENTITY_COMPOSITE_TRANSACTION
    ) {
        out.extend_from_slice(&(record.catalog_commands.len() as u32).to_le_bytes());
        for operation in &record.catalog_commands {
            let encoded = serde_json::to_vec(&operation.command).ok()?;
            if encoded.len() > u32::MAX as usize {
                return None;
            }
            out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            out.extend_from_slice(&encoded);
        }
    }
    if matches!(
        op,
        OP_ORDERED_CATALOG_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_TRANSACTION
            | OP_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            | OP_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
    ) {
        out.extend_from_slice(&(record.catalog_commands.len() as u32).to_le_bytes());
        for operation in &record.catalog_commands {
            let encoded = serde_json::to_vec(&operation.command).ok()?;
            if encoded.len() > u32::MAX as usize {
                return None;
            }
            out.extend_from_slice(&operation.ordinal.to_le_bytes());
            out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            out.extend_from_slice(&encoded);
        }
        out.extend_from_slice(&(record.created_table_identities.len() as u32).to_le_bytes());
        for (table, identity) in &record.created_table_identities {
            if table.len() > u16::MAX as usize {
                return None;
            }
            out.extend_from_slice(&(table.len() as u16).to_le_bytes());
            out.extend_from_slice(table.as_bytes());
            out.extend_from_slice(&identity.table_oid.to_le_bytes());
            out.extend_from_slice(&identity.schema_digest);
        }
        if index_catalog {
            out.extend_from_slice(
                &(record.created_table_index_identities.len() as u32).to_le_bytes(),
            );
            for (table, identities) in &record.created_table_index_identities {
                if table.len() > u16::MAX as usize {
                    return None;
                }
                out.extend_from_slice(&(table.len() as u16).to_le_bytes());
                out.extend_from_slice(table.as_bytes());
                out.extend_from_slice(&(identities.len() as u32).to_le_bytes());
                for identity in identities {
                    out.extend_from_slice(&identity.oid.to_le_bytes());
                    out.extend_from_slice(&identity.table_oid.to_le_bytes());
                    out.extend_from_slice(&identity.digest);
                }
            }
        }
        let catalog_output = record.catalog_output.as_ref()?;
        out.extend_from_slice(&catalog_output.relational_next_oid.to_le_bytes());
        out.extend_from_slice(&catalog_output.relational_next_column_id.to_le_bytes());
        out.extend_from_slice(&(catalog_output.created_sequence_oids.len() as u32).to_le_bytes());
        for (sequence, oid) in &catalog_output.created_sequence_oids {
            if sequence.len() > u16::MAX as usize {
                return None;
            }
            out.extend_from_slice(&(sequence.len() as u16).to_le_bytes());
            out.extend_from_slice(sequence.as_bytes());
            out.extend_from_slice(&oid.to_le_bytes());
        }
        if matches!(
            op,
            OP_ORDERED_CATALOG_VIEW_TRANSACTION | OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
        ) {
            encode_legacy_view_operations(&mut out, &record.view_operations)?;
        } else if matches!(
            op,
            OP_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
                | OP_IDENTITY_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
        ) {
            encode_view_lifecycle_operations(&mut out, &record.view_lifecycle_operations)?;
        } else if matches!(
            op,
            OP_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
                | OP_IDENTITY_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
                | OP_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
                | OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
        ) {
            encode_view_lifecycle_operations(&mut out, &record.view_lifecycle_operations)?;
            encode_index_lifecycle_operations(&mut out, &record.index_lifecycle_operations)?;
            if matches!(
                op,
                OP_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
                    | OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
            ) {
                encode_sequence_lifecycle_operations(
                    &mut out,
                    &record.sequence_lifecycle_operations,
                )?;
                encode_sequence_reset_operations(&mut out, &record.sequence_reset_operations)?;
                out.extend_from_slice(
                    &(record.sequence_advances_by_oid.len() as u32).to_le_bytes(),
                );
                for (oid, (last_value, is_called)) in &record.sequence_advances_by_oid {
                    if *oid == 0 {
                        return None;
                    }
                    out.extend_from_slice(&oid.to_le_bytes());
                    out.extend_from_slice(&last_value.to_le_bytes());
                    out.push(u8::from(*is_called));
                }
            }
        }
        out.extend_from_slice(&(record.operation_order.len() as u32).to_le_bytes());
        for operation in &record.operation_order {
            let (kind, table) = match operation {
                BinaryTransactionOperationIdentity::Catalog { command_index } => {
                    out.push(TXN_OPERATION_CATALOG);
                    out.extend_from_slice(&command_index.to_le_bytes());
                    continue;
                }
                BinaryTransactionOperationIdentity::Insert { table } => {
                    (TXN_OPERATION_INSERT, table)
                }
                BinaryTransactionOperationIdentity::Update { table } => {
                    (TXN_OPERATION_UPDATE, table)
                }
                BinaryTransactionOperationIdentity::Delete { table } => {
                    (TXN_OPERATION_DELETE, table)
                }
                BinaryTransactionOperationIdentity::TableReset { table } => {
                    (TXN_OPERATION_TABLE_RESET, table)
                }
            };
            out.push(kind);
            out.extend_from_slice(&(table.len() as u16).to_le_bytes());
            out.extend_from_slice(table.as_bytes());
        }
        out.extend_from_slice(&(record.statement_digests.len() as u32).to_le_bytes());
        for digest in &record.statement_digests {
            out.extend_from_slice(digest);
        }
        out.extend_from_slice(&(record.sequence_input_oids.len() as u32).to_le_bytes());
        for ((ordinal, sequence), oid) in &record.sequence_input_oids {
            out.extend_from_slice(&ordinal.to_le_bytes());
            out.extend_from_slice(&(sequence.len() as u16).to_le_bytes());
            out.extend_from_slice(sequence.as_bytes());
            out.extend_from_slice(&oid.to_le_bytes());
        }
    }
    if matches!(
        op,
        OP_TABLE_RESET_TRANSACTION
            | OP_IDENTITY_TABLE_RESET_TRANSACTION
            | OP_ORDERED_CATALOG_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_TRANSACTION
            | OP_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            | OP_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
    ) {
        out.extend_from_slice(&(record.table_resets.len() as u32).to_le_bytes());
        let mut prior_ordinal = None;
        let mut tables = BTreeSet::new();
        for reset in &record.table_resets {
            if prior_ordinal.is_some_and(|prior| reset.ordinal <= prior)
                || !tables.insert(reset.table_oid)
                || reset.table.len() > u16::MAX as usize
                || reset.dependency_identities.len() > u32::MAX as usize
                || reset.dependency_identities.get(&reset.table) != Some(&reset.table_oid)
            {
                return None;
            }
            if catalog_names
                .get(reset.table.as_str())
                .is_some_and(|create_ordinal| *create_ordinal >= reset.ordinal)
            {
                return None;
            }
            prior_ordinal = Some(reset.ordinal);
            out.extend_from_slice(&reset.ordinal.to_le_bytes());
            out.extend_from_slice(&(reset.table.len() as u16).to_le_bytes());
            out.extend_from_slice(reset.table.as_bytes());
            out.extend_from_slice(&reset.table_oid.to_le_bytes());
            out.extend_from_slice(&reset.schema_digest);
            out.extend_from_slice(&reset.source_commit_seq.to_le_bytes());
            out.extend_from_slice(&reset.before_digest);
            out.extend_from_slice(&reset.expected_rows.to_le_bytes());
            out.extend_from_slice(&reset.after_empty_digest);
            out.extend_from_slice(&(reset.dependency_identities.len() as u32).to_le_bytes());
            for (name, oid) in &reset.dependency_identities {
                if name.len() > u16::MAX as usize {
                    return None;
                }
                out.extend_from_slice(&(name.len() as u16).to_le_bytes());
                out.extend_from_slice(name.as_bytes());
                out.extend_from_slice(&oid.to_le_bytes());
            }
        }
    }
    if identity_bound {
        out.extend_from_slice(&(record.table_identities.len() as u32).to_le_bytes());
        for (table, identity) in &record.table_identities {
            if table.len() > u16::MAX as usize {
                return None;
            }
            out.extend_from_slice(&(table.len() as u16).to_le_bytes());
            out.extend_from_slice(table.as_bytes());
            out.extend_from_slice(&identity.table_oid.to_le_bytes());
            out.extend_from_slice(&identity.schema_digest);
        }
    }
    out.extend_from_slice(&record.allocator_high_water.to_le_bytes());
    out.extend_from_slice(&(record.sequence_advances.len() as u32).to_le_bytes());
    for (sequence, (last_value, is_called)) in &record.sequence_advances {
        if sequence.len() > u16::MAX as usize {
            return None;
        }
        out.extend_from_slice(&(sequence.len() as u16).to_le_bytes());
        out.extend_from_slice(sequence.as_bytes());
        out.extend_from_slice(&last_value.to_le_bytes());
        out.push(u8::from(*is_called));
    }
    out.extend_from_slice(&(record.mutations.len() as u32).to_le_bytes());
    for mutation in &record.mutations {
        let (kind, table, row_id) = match mutation {
            BinaryTransactionMutation::Insert { table, row_id, .. } => (TXN_INSERT, table, *row_id),
            BinaryTransactionMutation::Update { table, row_id, .. } => (TXN_UPDATE, table, *row_id),
            BinaryTransactionMutation::Delete { table, row_id, .. } => (TXN_DELETE, table, *row_id),
        };
        if table.len() > u16::MAX as usize {
            return None;
        }
        out.push(kind);
        out.extend_from_slice(&(table.len() as u16).to_le_bytes());
        out.extend_from_slice(table.as_bytes());
        out.extend_from_slice(&row_id.to_le_bytes());
        let mut push_row = |encoded: &str| -> Option<()> {
            if encoded.len() > u32::MAX as usize {
                return None;
            }
            out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            out.extend_from_slice(encoded.as_bytes());
            Some(())
        };
        match mutation {
            BinaryTransactionMutation::Insert { row_encoded, .. } => push_row(row_encoded)?,
            BinaryTransactionMutation::Update {
                old_row_encoded,
                new_row_encoded,
                ..
            } => {
                push_row(old_row_encoded)?;
                push_row(new_row_encoded)?;
            }
            BinaryTransactionMutation::Delete {
                old_row_encoded, ..
            } => push_row(old_row_encoded)?,
        }
    }
    out.shrink_to_fit();
    Some(out)
}

fn decode_binary_transaction(payload: &[u8]) -> Result<BinaryTransactionRecord, EngineError> {
    let fail = |what: &str| EngineError::Durability(format!("malformed binary WAL record: {what}"));
    let mut at = 0usize;
    let mut take = |n: usize| -> Result<&[u8], EngineError> {
        let end = at.checked_add(n).ok_or_else(|| fail("length overflow"))?;
        let slice = payload.get(at..end).ok_or_else(|| fail("truncated"))?;
        at = end;
        Ok(slice)
    };
    if take(1)?[0] != WAL_BINARY_TAG {
        return Err(fail("missing tag"));
    }
    if take(1)?[0] != WAL_BINARY_VERSION {
        return Err(fail("unsupported version"));
    }
    let op = take(1)?[0];
    if !matches!(
        op,
        OP_TRANSACTION
            | OP_COMPOSITE_TRANSACTION
            | OP_TABLE_RESET_TRANSACTION
            | OP_IDENTITY_TRANSACTION
            | OP_IDENTITY_COMPOSITE_TRANSACTION
            | OP_IDENTITY_TABLE_RESET_TRANSACTION
            | OP_ORDERED_CATALOG_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_TRANSACTION
            | OP_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            | OP_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
    ) {
        return Err(fail("op dispatch mismatch"));
    }
    let mut catalog_commands = Vec::new();
    let ordered_catalog = matches!(
        op,
        OP_ORDERED_CATALOG_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_TRANSACTION
            | OP_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            | OP_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
    );
    let index_catalog = matches!(
        op,
        OP_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
    );
    let sequence_catalog = matches!(
        op,
        OP_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
    );
    let view_catalog = matches!(
        op,
        OP_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            | OP_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
    );
    let view_lifecycle_catalog = matches!(
        op,
        OP_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            | OP_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
    );
    if ordered_catalog
        || matches!(
            op,
            OP_COMPOSITE_TRANSACTION | OP_IDENTITY_COMPOSITE_TRANSACTION
        )
    {
        let command_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        if (!ordered_catalog && command_count != 1)
            || (ordered_catalog && command_count == 0 && !sequence_catalog)
        {
            return Err(fail(
                "catalog transaction contains a non-canonical operation count",
            ));
        }
        catalog_commands.reserve(command_count.min(1024));
        let mut prior_ordinal = None;
        for _ in 0..command_count {
            let ordinal = if ordered_catalog {
                u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"))
            } else {
                0
            };
            if prior_ordinal.is_some_and(|prior| ordinal <= prior) {
                return Err(fail(
                    "catalog operations are not in canonical ordinal order",
                ));
            }
            prior_ordinal = Some(ordinal);
            let len = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            let bytes = take(len)?;
            let command: Command = serde_json::from_slice(bytes)
                .map_err(|error| fail(&format!("typed catalog command decode failed: {error}")))?;
            let supported_view = if view_lifecycle_catalog {
                command_is_view_lifecycle(&command)
            } else {
                view_catalog && matches!(command, Command::CreateView(_))
            };
            let supported_index = index_catalog && command_is_index_lifecycle(&command);
            let supported_sequence = sequence_catalog && command_is_sequence_lifecycle(&command);
            if !matches!(command, Command::CreateTable(_))
                && !supported_view
                && !supported_index
                && !supported_sequence
            {
                return Err(fail("unsupported composite catalog command"));
            }
            if serde_json::to_vec(&command).ok().as_deref() != Some(bytes) {
                return Err(fail("non-canonical typed catalog command"));
            }
            catalog_commands.push(BinaryTransactionCatalogCommand { ordinal, command });
        }
    }
    let mut created_table_identities = BTreeMap::new();
    let mut created_table_index_identities = BTreeMap::new();
    let mut catalog_output = None;
    let mut view_operations = Vec::new();
    let mut view_lifecycle_operations = Vec::new();
    let mut index_lifecycle_operations = Vec::new();
    let mut sequence_lifecycle_operations = Vec::new();
    let mut sequence_reset_operations = Vec::new();
    let mut sequence_advances_by_oid = BTreeMap::new();
    let mut operation_order = Vec::new();
    let mut statement_digests = Vec::new();
    let mut sequence_input_oids = BTreeMap::new();
    if ordered_catalog {
        let identity_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        for _ in 0..identity_count {
            let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let table = std::str::from_utf8(take(name_len)?)
                .map_err(|_| fail("non-utf8 created-table identity name"))?
                .to_string();
            let table_oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            let schema_digest = take(32)?.try_into().expect("32 bytes");
            if table_oid == 0 || table_oid > i32::MAX as u32 || schema_digest == [0; 32] {
                return Err(fail("invalid created-table stable identity"));
            }
            if created_table_identities
                .insert(
                    table,
                    BinaryTransactionTableIdentity {
                        table_oid,
                        schema_digest,
                    },
                )
                .is_some()
            {
                return Err(fail("duplicate created-table identity"));
            }
        }
        let command_names = catalog_commands
            .iter()
            .filter_map(|operation| match &operation.command {
                Command::CreateTable(create) => Some(create.table.as_str()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let created_table_command_count = catalog_commands
            .iter()
            .filter(|operation| matches!(&operation.command, Command::CreateTable(_)))
            .count();
        let identity_names = created_table_identities
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if command_names.len() != created_table_command_count || command_names != identity_names {
            return Err(fail(
                "ordered catalog identities do not cover the exact created-table set",
            ));
        }
        if index_catalog {
            let table_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            if table_count != command_names.len() {
                return Err(fail(
                    "created-table index identities do not cover the exact table set",
                ));
            }
            let mut prior_name = None;
            let mut class_oids = created_table_identities
                .values()
                .map(|identity| identity.table_oid)
                .collect::<BTreeSet<_>>();
            if class_oids.len() != created_table_identities.len() {
                return Err(fail("created tables reuse a pg_class OID"));
            }
            for _ in 0..table_count {
                let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
                let table = std::str::from_utf8(take(name_len)?)
                    .map_err(|_| fail("non-utf8 created-table index identity name"))?
                    .to_string();
                if prior_name.as_ref().is_some_and(|prior| prior >= &table) {
                    return Err(fail(
                        "created-table index identities are not in canonical name order",
                    ));
                }
                prior_name = Some(table.clone());
                let expected_count = catalog_commands
                    .iter()
                    .find_map(|operation| match &operation.command {
                        Command::CreateTable(create) if create.table == table => Some(
                            usize::from(create.primary_key.is_some())
                                + create.unique_constraints.len(),
                        ),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        fail("created-table index identity has no CREATE TABLE command")
                    })?;
                let index_count =
                    u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
                if index_count != expected_count {
                    return Err(fail(
                        "created-table index identity count changed from CREATE TABLE",
                    ));
                }
                let table_oid = created_table_identities
                    .get(&table)
                    .map(|identity| identity.table_oid)
                    .ok_or_else(|| fail("created-table index identity lost its table"))?;
                let mut identities = Vec::with_capacity(index_count.min(1024));
                for _ in 0..index_count {
                    let identity = BinaryCatalogIndexIdentity {
                        oid: u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")),
                        table_oid: u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")),
                        digest: take(32)?.try_into().expect("32 bytes"),
                    };
                    if identity.oid == 0
                        || identity.oid > i32::MAX as u32
                        || identity.table_oid != table_oid
                        || identity.digest == [0; 32]
                        || !class_oids.insert(identity.oid)
                    {
                        return Err(fail(
                            "created-table index identity is invalid or reuses a pg_class OID",
                        ));
                    }
                    identities.push(identity);
                }
                if created_table_index_identities
                    .insert(table, identities)
                    .is_some()
                {
                    return Err(fail("duplicate created-table index identity"));
                }
            }
            if created_table_index_identities
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                != command_names
            {
                return Err(fail(
                    "created-table index identities do not cover the exact table set",
                ));
            }
        }
        let relational_next_oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
        let relational_next_column_id = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
        let sequence_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        let mut created_sequence_oids = BTreeMap::new();
        for _ in 0..sequence_count {
            let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let sequence = std::str::from_utf8(take(name_len)?)
                .map_err(|_| fail("non-utf8 created-sequence identity name"))?
                .to_string();
            let oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            if created_sequence_oids.insert(sequence, oid).is_some() {
                return Err(fail("duplicate created-sequence identity"));
            }
        }
        if generated_sequence_names(&catalog_commands).is_none_or(|expected| {
            expected
                != created_sequence_oids
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
        }) {
            return Err(fail(
                "ordered catalog sequence identities do not cover the exact generated set",
            ));
        }
        catalog_output = Some(BinaryTransactionCatalogOutput {
            relational_next_oid,
            relational_next_column_id,
            created_sequence_oids,
        });
        if view_catalog {
            let view_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            let expected_views = catalog_commands
                .iter()
                .enumerate()
                .filter_map(|(command_index, operation)| {
                    command_is_view_lifecycle(&operation.command).then_some((
                        command_index,
                        operation.ordinal,
                        &operation.command,
                    ))
                })
                .collect::<Vec<_>>();
            if view_count != expected_views.len()
                || (!index_catalog
                    && (view_count == 0
                        || (view_lifecycle_catalog
                            != expected_views.iter().any(|(_, _, command)| {
                                command_requires_view_lifecycle_opcode(command)
                            }))))
            {
                return Err(fail(
                    "view identities do not cover the exact stored-view command set",
                ));
            }
            if view_lifecycle_catalog {
                view_lifecycle_operations.reserve(view_count.min(1024));
            } else {
                view_operations.reserve(view_count.min(1024));
            }
            for (expected_command_index, expected_ordinal, command) in expected_views {
                let command_index = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
                let ordinal = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
                if usize::try_from(command_index).ok() != Some(expected_command_index)
                    || ordinal != expected_ordinal
                {
                    return Err(fail(
                        "view identity does not match its catalog command position",
                    ));
                }
                if view_lifecycle_catalog {
                    let identity =
                        decode_view_lifecycle_operation(command_index, ordinal, &mut take, &fail)?;
                    if !valid_view_lifecycle_operation_identity(command, &identity) {
                        return Err(fail("invalid stored-view lifecycle identity closure"));
                    }
                    view_lifecycle_operations.push(identity);
                } else {
                    if !matches!(command, Command::CreateView(_)) {
                        return Err(fail("legacy view opcode carries a non-CREATE command"));
                    }
                    view_operations.push(decode_legacy_view_operation(
                        command_index,
                        ordinal,
                        &mut take,
                        &fail,
                    )?);
                }
            }
        }
        if index_catalog {
            let index_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            let expected_indexes = catalog_commands
                .iter()
                .enumerate()
                .filter_map(|(command_index, operation)| {
                    command_is_index_lifecycle(&operation.command).then_some((
                        command_index,
                        operation.ordinal,
                        &operation.command,
                    ))
                })
                .collect::<Vec<_>>();
            if index_count != expected_indexes.len() {
                return Err(fail(
                    "index identities do not cover the exact index command set",
                ));
            }
            index_lifecycle_operations.reserve(index_count.min(1024));
            for (expected_command_index, expected_ordinal, command) in expected_indexes {
                let command_index = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
                let ordinal = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
                if usize::try_from(command_index).ok() != Some(expected_command_index)
                    || ordinal != expected_ordinal
                {
                    return Err(fail(
                        "index identity does not match its catalog command position",
                    ));
                }
                let identity =
                    decode_index_lifecycle_operation(command_index, ordinal, &mut take, &fail)?;
                if !valid_index_lifecycle_operation_identity(command, &identity) {
                    return Err(fail("invalid index lifecycle identity closure"));
                }
                index_lifecycle_operations.push(identity);
            }
        }
        if sequence_catalog {
            let sequence_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            let expected_sequences = catalog_commands
                .iter()
                .enumerate()
                .filter_map(|(command_index, operation)| {
                    command_is_sequence_lifecycle(&operation.command).then_some((
                        command_index,
                        operation.ordinal,
                        &operation.command,
                    ))
                })
                .collect::<Vec<_>>();
            if sequence_count != expected_sequences.len() {
                return Err(fail(
                    "sequence identities do not cover the exact sequence command set",
                ));
            }
            sequence_lifecycle_operations.reserve(sequence_count.min(1024));
            for (expected_command_index, expected_ordinal, command) in expected_sequences {
                let command_index = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
                let ordinal = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
                if usize::try_from(command_index).ok() != Some(expected_command_index)
                    || ordinal != expected_ordinal
                {
                    return Err(fail(
                        "sequence identity does not match its catalog command position",
                    ));
                }
                let identity =
                    decode_sequence_lifecycle_operation(command_index, ordinal, &mut take, &fail)?;
                if !valid_sequence_lifecycle_operation_identity(command, &identity) {
                    return Err(fail("invalid sequence lifecycle identity closure"));
                }
                sequence_lifecycle_operations.push(identity);
            }
            sequence_reset_operations = decode_sequence_reset_operations(&mut take, &fail)?;
            let advance_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            for _ in 0..advance_count {
                let oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
                let last_value = i64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
                let is_called = match take(1)?[0] {
                    0 => false,
                    1 => true,
                    _ => return Err(fail("invalid stable-ID sequence called flag")),
                };
                if oid == 0
                    || sequence_advances_by_oid
                        .insert(oid, (last_value, is_called))
                        .is_some()
                {
                    return Err(fail("duplicate or invalid stable-ID sequence advancement"));
                }
            }
            if sequence_lifecycle_operations.is_empty() && sequence_reset_operations.is_empty() {
                return Err(fail(
                    "sequence lifecycle opcode requires a lifecycle or reset identity",
                ));
            }
        }
        let operation_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        if operation_count == 0 {
            return Err(fail(
                "ordered catalog transaction has an empty operation order",
            ));
        }
        operation_order.reserve(operation_count.min(64 * 1024));
        for _ in 0..operation_count {
            let kind = take(1)?[0];
            let operation = if kind == TXN_OPERATION_CATALOG {
                BinaryTransactionOperationIdentity::Catalog {
                    command_index: u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")),
                }
            } else {
                let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
                let table = std::str::from_utf8(take(name_len)?)
                    .map_err(|_| fail("non-utf8 ordered operation table"))?
                    .to_string();
                match kind {
                    TXN_OPERATION_INSERT => BinaryTransactionOperationIdentity::Insert { table },
                    TXN_OPERATION_UPDATE => BinaryTransactionOperationIdentity::Update { table },
                    TXN_OPERATION_DELETE => BinaryTransactionOperationIdentity::Delete { table },
                    TXN_OPERATION_TABLE_RESET => {
                        BinaryTransactionOperationIdentity::TableReset { table }
                    }
                    other => {
                        return Err(fail(&format!(
                            "unsupported ordered transaction operation {other}"
                        )))
                    }
                }
            };
            operation_order.push(operation);
        }
        let digest_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        if digest_count != operation_count {
            return Err(fail(
                "ordered statement digests do not cover the operation order",
            ));
        }
        statement_digests.reserve(digest_count.min(64 * 1024));
        for _ in 0..digest_count {
            let digest: gpu_db_wal::CanonicalDigest = take(32)?.try_into().expect("32 bytes");
            if digest == [0; 32] {
                return Err(fail("ordered statement digest is empty"));
            }
            statement_digests.push(digest);
        }
        let sequence_input_count =
            u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        let mut prior_sequence_key = None;
        for _ in 0..sequence_input_count {
            let ordinal = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let sequence = std::str::from_utf8(take(name_len)?)
                .map_err(|_| fail("non-utf8 ordered sequence input name"))?
                .to_string();
            let oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            let key = (ordinal, sequence);
            if oid == 0
                || prior_sequence_key
                    .as_ref()
                    .is_some_and(|prior| prior >= &key)
                || sequence_input_oids.insert(key.clone(), oid).is_some()
            {
                return Err(fail("non-canonical ordered sequence input identity"));
            }
            prior_sequence_key = Some(key);
        }
        if catalog_output.as_ref().is_none_or(|output| {
            !generated_sequence_output_matches_inputs(
                &catalog_commands,
                &sequence_input_oids,
                output,
            )
        }) {
            return Err(fail(
                "generated-sequence output contradicts its CREATE statement input",
            ));
        }
    }
    let mut table_resets = Vec::new();
    if ordered_catalog
        || matches!(
            op,
            OP_TABLE_RESET_TRANSACTION | OP_IDENTITY_TABLE_RESET_TRANSACTION
        )
    {
        let reset_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        if !ordered_catalog && reset_count == 0 {
            return Err(fail("table-reset transaction requires at least one reset"));
        }
        table_resets.reserve(reset_count.min(1024));
        let mut prior_ordinal = None;
        let mut tables = BTreeSet::new();
        for _ in 0..reset_count {
            let ordinal = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            if prior_ordinal.is_some_and(|prior| ordinal <= prior) {
                return Err(fail("table resets are not in canonical ordinal order"));
            }
            prior_ordinal = Some(ordinal);
            let table_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let table = std::str::from_utf8(take(table_len)?)
                .map_err(|_| fail("non-utf8 reset table name"))?
                .to_string();
            let table_oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            if !tables.insert(table_oid) {
                return Err(fail("duplicate table reset identity"));
            }
            let schema_digest = take(32)?.try_into().expect("32 bytes");
            let source_commit_seq = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
            let before_digest = take(32)?.try_into().expect("32 bytes");
            let expected_rows = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
            let after_empty_digest = take(32)?.try_into().expect("32 bytes");
            let dependency_count =
                u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            let mut dependency_identities = BTreeMap::new();
            for _ in 0..dependency_count {
                let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
                let name = std::str::from_utf8(take(name_len)?)
                    .map_err(|_| fail("non-utf8 reset dependency name"))?
                    .to_string();
                let oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
                if dependency_identities.insert(name, oid).is_some() {
                    return Err(fail("duplicate table reset dependency"));
                }
            }
            if dependency_identities.get(&table) != Some(&table_oid) {
                return Err(fail(
                    "table reset dependency closure omits its target identity",
                ));
            }
            table_resets.push(BinaryTransactionTableReset {
                ordinal,
                table,
                table_oid,
                schema_digest,
                source_commit_seq,
                before_digest,
                expected_rows,
                after_empty_digest,
                dependency_identities,
            });
        }
    }
    let identity_bound = matches!(
        op,
        OP_IDENTITY_TRANSACTION
            | OP_IDENTITY_COMPOSITE_TRANSACTION
            | OP_IDENTITY_TABLE_RESET_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION
    );
    let mut table_identities = BTreeMap::new();
    if identity_bound {
        let identity_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        for _ in 0..identity_count {
            let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let table = std::str::from_utf8(take(name_len)?)
                .map_err(|_| fail("non-utf8 transaction identity table"))?
                .to_string();
            let table_oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            let schema_digest = take(32)?.try_into().expect("32 bytes");
            if table_identities
                .insert(
                    table,
                    BinaryTransactionTableIdentity {
                        table_oid,
                        schema_digest,
                    },
                )
                .is_some()
            {
                return Err(fail("duplicate transaction table identity"));
            }
        }
    }
    let allocator_high_water = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
    let sequence_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
    let mut sequence_advances = BTreeMap::new();
    for _ in 0..sequence_count {
        let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
        let name = std::str::from_utf8(take(name_len)?)
            .map_err(|_| fail("non-utf8 sequence name"))?
            .to_string();
        let last_value = i64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
        let is_called = match take(1)?[0] {
            0 => false,
            1 => true,
            _ => return Err(fail("invalid sequence called flag")),
        };
        if sequence_advances
            .insert(name, (last_value, is_called))
            .is_some()
        {
            return Err(fail("duplicate sequence advancement"));
        }
    }
    let mutation_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
    let mut mutations = Vec::with_capacity(mutation_count.min(64 * 1024));
    for _ in 0..mutation_count {
        let kind = take(1)?[0];
        let table_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
        let table = std::str::from_utf8(take(table_len)?)
            .map_err(|_| fail("non-utf8 table name"))?
            .to_string();
        let row_id = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
        let mut take_row = || -> Result<String, EngineError> {
            let len = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            std::str::from_utf8(take(len)?)
                .map(str::to_string)
                .map_err(|_| fail("non-utf8 row encoding"))
        };
        let mutation = match kind {
            TXN_INSERT => BinaryTransactionMutation::Insert {
                table,
                row_id,
                row_encoded: take_row()?,
            },
            TXN_UPDATE => BinaryTransactionMutation::Update {
                table,
                row_id,
                old_row_encoded: take_row()?,
                new_row_encoded: take_row()?,
            },
            TXN_DELETE => BinaryTransactionMutation::Delete {
                table,
                row_id,
                old_row_encoded: take_row()?,
            },
            other => return Err(fail(&format!("unsupported transaction mutation {other}"))),
        };
        mutations.push(mutation);
    }
    if at != payload.len() {
        return Err(fail("trailing bytes"));
    }
    let reset_names = table_resets
        .iter()
        .map(|reset| reset.table.as_str())
        .collect::<BTreeSet<_>>();
    let mutation_names = mutations
        .iter()
        .map(|mutation| match mutation {
            BinaryTransactionMutation::Insert { table, .. }
            | BinaryTransactionMutation::Update { table, .. }
            | BinaryTransactionMutation::Delete { table, .. } => table.as_str(),
        })
        .collect::<BTreeSet<_>>();
    let identity_names = table_identities
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let created_ordinals = catalog_commands
        .iter()
        .filter_map(|operation| match &operation.command {
            Command::CreateTable(create) => Some((create.table.as_str(), operation.ordinal)),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    let sequence_resets_by_ordinal = sequence_reset_operations
        .iter()
        .map(|identity| (identity.ordinal, identity))
        .collect::<BTreeMap<_, _>>();
    if sequence_resets_by_ordinal.len() != sequence_reset_operations.len() {
        return Err(fail("duplicate sequence reset operation ordinal"));
    }
    let mut ordered_row_names = BTreeSet::new();
    let mut ordered_reset_names = BTreeSet::new();
    let mut next_catalog_index = 0usize;
    if ordered_catalog {
        for (ordinal, operation) in operation_order.iter().enumerate() {
            let ordinal_u32 = u32::try_from(ordinal)
                .map_err(|_| fail("ordered operation ordinal exceeds u32"))?;
            let statement_digest = statement_digests
                .get(ordinal)
                .ok_or_else(|| fail("ordered operation has no statement digest"))?;
            let input_sequence_names = sequence_input_oids
                .keys()
                .filter_map(|(input_ordinal, sequence)| {
                    (*input_ordinal == ordinal_u32).then_some(sequence.as_str())
                })
                .collect::<BTreeSet<_>>();
            match operation {
                BinaryTransactionOperationIdentity::Catalog { command_index } => {
                    if usize::try_from(*command_index).ok() != Some(next_catalog_index)
                        || catalog_commands
                            .get(next_catalog_index)
                            .is_none_or(|command| {
                                usize::try_from(command.ordinal).ok() != Some(ordinal)
                            })
                    {
                        return Err(fail(
                            "ordered catalog command does not match its operation position",
                        ));
                    }
                    let command = &catalog_commands[next_catalog_index].command;
                    if transaction_statement_digest(command)
                        .map_err(|error| fail(&error.to_string()))?
                        != *statement_digest
                    {
                        return Err(fail("ordered catalog statement digest mismatch"));
                    }
                    let expected_sequences = match command {
                        Command::CreateTable(create) => create
                            .columns
                            .iter()
                            .filter_map(|column| match &column.default {
                                Some(ColumnDefault::SequenceNextVal { sequence, .. }) => {
                                    Some(sequence.as_str())
                                }
                                _ => None,
                            })
                            .collect::<BTreeSet<_>>(),
                        Command::CreateView(_) | Command::RenameView(_) | Command::DropView(_) => {
                            BTreeSet::new()
                        }
                        Command::CreateIndex(_)
                        | Command::RenameIndex(_)
                        | Command::DropIndex(_) => BTreeSet::new(),
                        Command::CreateSequence(_)
                        | Command::SequenceRestart(_)
                        | Command::RenameSequence(_)
                        | Command::DropSequence(_) => BTreeSet::new(),
                        _ => return Err(fail("unsupported ordered catalog command")),
                    };
                    if input_sequence_names != expected_sequences {
                        return Err(fail(
                            "ordered catalog sequence inputs do not cover its exact defaults",
                        ));
                    }
                    next_catalog_index += 1;
                }
                BinaryTransactionOperationIdentity::Insert { table }
                | BinaryTransactionOperationIdentity::Update { table }
                | BinaryTransactionOperationIdentity::Delete { table } => {
                    if created_ordinals.get(table.as_str()).is_some_and(|created| {
                        usize::try_from(*created)
                            .ok()
                            .is_none_or(|created| created >= ordinal)
                    }) {
                        return Err(fail(
                            "ordered row operation precedes its transaction-private relation",
                        ));
                    }
                    ordered_row_names.insert(table.as_str());
                }
                BinaryTransactionOperationIdentity::TableReset { table } => {
                    if created_ordinals.get(table.as_str()).is_some_and(|created| {
                        usize::try_from(*created)
                            .ok()
                            .is_none_or(|created| created >= ordinal)
                    }) {
                        return Err(fail(
                            "ordered table reset precedes its transaction-private relation",
                        ));
                    }
                    let command = Command::TruncateTable(TruncateTable {
                        name: table.clone(),
                        restart_identity: sequence_resets_by_ordinal
                            .get(&ordinal_u32)
                            .is_some_and(|identity| identity.table == *table),
                    });
                    if transaction_statement_digest(&command)
                        .map_err(|error| fail(&error.to_string()))?
                        != *statement_digest
                    {
                        return Err(fail("ordered table-reset statement digest mismatch"));
                    }
                    ordered_reset_names.insert(table.as_str());
                }
            }
            if !matches!(
                operation,
                BinaryTransactionOperationIdentity::Catalog { .. }
                    | BinaryTransactionOperationIdentity::Insert { .. }
            ) && !input_sequence_names.is_empty()
            {
                return Err(fail(
                    "non-INSERT ordered operation carries a sequence input",
                ));
            }
        }
        if next_catalog_index != catalog_commands.len() {
            return Err(fail("ordered operation vector omits a catalog command"));
        }
        if sequence_input_oids.keys().any(|(ordinal, _)| {
            usize::try_from(*ordinal)
                .ok()
                .is_none_or(|ordinal| ordinal >= operation_order.len())
        }) {
            return Err(fail(
                "ordered sequence input ordinal exceeds operation order",
            ));
        }
        let insert_sequence_names = sequence_input_oids
            .keys()
            .filter_map(|(ordinal, sequence)| {
                usize::try_from(*ordinal)
                    .ok()
                    .and_then(|ordinal| operation_order.get(ordinal))
                    .and_then(|operation| {
                        matches!(operation, BinaryTransactionOperationIdentity::Insert { .. })
                            .then_some(sequence.as_str())
                    })
            })
            .collect::<BTreeSet<_>>();
        if sequence_catalog {
            let insert_sequence_oids = sequence_input_oids
                .iter()
                .filter_map(|((ordinal, _), oid)| {
                    usize::try_from(*ordinal)
                        .ok()
                        .and_then(|ordinal| operation_order.get(ordinal))
                        .and_then(|operation| {
                            matches!(operation, BinaryTransactionOperationIdentity::Insert { .. })
                                .then_some(*oid)
                        })
                })
                .collect::<BTreeSet<_>>();
            if !sequence_advances.is_empty()
                || insert_sequence_oids
                    != sequence_advances_by_oid
                        .keys()
                        .copied()
                        .collect::<BTreeSet<_>>()
            {
                return Err(fail(
                    "ordered INSERT stable sequence identities do not close over sequence advances",
                ));
            }
        } else if insert_sequence_names
            != sequence_advances
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
            || !sequence_advances_by_oid.is_empty()
        {
            return Err(fail(
                "ordered INSERT sequence identities do not close over sequence advances",
            ));
        }
    }
    let created_names = created_ordinals.keys().copied().collect::<BTreeSet<_>>();
    let ordered_existing_row_names = ordered_row_names
        .difference(&created_names)
        .copied()
        .collect::<BTreeSet<_>>();
    if reset_names.len() != table_resets.len()
        || (ordered_catalog
            && (ordered_reset_names != reset_names
                || ordered_existing_row_names != identity_names
                || !mutation_names.is_subset(&ordered_row_names)))
        || (!ordered_catalog && identity_bound && identity_names != mutation_names)
        || mutations.iter().any(|mutation| match mutation {
            BinaryTransactionMutation::Insert { .. } => false,
            BinaryTransactionMutation::Update { table, .. }
            | BinaryTransactionMutation::Delete { table, .. } => {
                reset_names.contains(table.as_str())
            }
        })
    {
        return Err(fail("table reset has non-canonical post-reset mutations"));
    }
    if ordered_catalog {
        if sequence_reset_operations.iter().any(|identity| {
            usize::try_from(identity.ordinal)
                .ok()
                .and_then(|ordinal| operation_order.get(ordinal))
                .is_none_or(|operation| {
                    !matches!(
                        operation,
                        BinaryTransactionOperationIdentity::TableReset { table }
                            if table == &identity.table
                    )
                })
        }) {
            return Err(fail(
                "sequence reset identity does not match an ordered table reset",
            ));
        }
        for reset in &table_resets {
            let Some(operation) = usize::try_from(reset.ordinal)
                .ok()
                .and_then(|ordinal| operation_order.get(ordinal))
            else {
                return Err(fail("table reset ordinal exceeds the operation envelope"));
            };
            if !matches!(operation, BinaryTransactionOperationIdentity::TableReset { table } if table == &reset.table)
            {
                return Err(fail(
                    "table reset output does not match its ordered operation identity",
                ));
            }
        }
        for mutation in &mutations {
            let table = match mutation {
                BinaryTransactionMutation::Insert { table, .. }
                | BinaryTransactionMutation::Update { table, .. }
                | BinaryTransactionMutation::Delete { table, .. } => table,
            };
            let last_reset = operation_order.iter().rposition(
                |operation| matches!(operation, BinaryTransactionOperationIdentity::TableReset { table: reset_table } if reset_table == table),
            );
            if !operation_order
                .iter()
                .enumerate()
                .any(|(ordinal, operation)| {
                    operation.matches_mutation(mutation)
                        && last_reset.is_none_or(|reset_ordinal| ordinal > reset_ordinal)
                })
            {
                return Err(fail(
                    "resolved mutation has no ordered row operation after its last reset",
                ));
            }
        }
    }
    Ok(BinaryTransactionRecord {
        catalog_epoch: if index_catalog {
            BinaryTransactionCatalogEpoch::IndexIdentityV1
        } else {
            BinaryTransactionCatalogEpoch::Legacy
        },
        allocator_high_water,
        catalog_commands,
        created_table_identities,
        created_table_index_identities,
        catalog_output,
        view_operations,
        view_lifecycle_operations,
        index_lifecycle_operations,
        sequence_lifecycle_operations,
        sequence_reset_operations,
        sequence_advances_by_oid,
        operation_order,
        statement_digests,
        sequence_input_oids,
        table_resets,
        sequence_advances,
        table_identities,
        mutations,
    })
}

#[cfg(test)]
#[path = "wal_binary/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "wal_binary/w5b_tests.rs"]
mod w5b_tests;
