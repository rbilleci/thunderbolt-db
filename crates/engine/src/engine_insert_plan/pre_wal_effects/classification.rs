//! Stable-OID classification of inert INSERT sequence effects.
//!
//! This module consumes one statement-locked transaction capture and returns compact planned
//! effects only. It neither publishes a transition nor materializes a typed INSERT value.

use super::{
    baseline_drift, ExplicitCapture, InsertEffectParentIdentity, PreparedMutation,
    TransactionOperation,
};
use crate::typed_insert_batch::{PreparedTypedInsert, SequenceDefaultRequestEffectShape};
use crate::{
    command_is_sequence_lifecycle, sequence_descriptor_digest, sequence_value_input_digest,
    valid_sequence_lifecycle_operation_identity, valid_sequence_reset_operation_identity,
    BinaryCatalogRelationKind, BinarySequenceValueOperation, CatalogSnapshot, Command,
    ExecuteError, SequenceValueInput,
};
use std::collections::BTreeMap;

#[cfg(test)]
use super::receipt::{
    PrivateOwnerEvidence, PrivatePredecessorEvidence, PublishedReceiptInput,
    SequenceOutputEvidence, SequenceReceiptBundle, SequenceSealBindingBundle,
    SequenceTransitionEvidence,
};
#[cfg(test)]
use crate::typed_insert_batch::sequence_defaults::effects::{
    PrivateSequencePlanningEvidence, SequenceDefaultBinding, SequenceDefaultBindings,
    SequenceDefaultParentContext,
};
#[cfg(test)]
use std::collections::BTreeSet;

/// Compact, move-only planned effects. These digests are planning witnesses only; no field is a
/// WAL record digest or a durable receipt.
#[allow(dead_code)] // The terminal effect owner is deliberately deferred to the next PLAN slice.
pub(super) struct PlannedSequenceEffects {
    parent: PlannedSequenceParent,
    effects: Box<[PlannedSequenceEffect]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlannedSequenceParent {
    txn_id: crate::TxnId,
    autocommit: bool,
    request_digest: gpu_db_wal::CanonicalDigest,
    statement_ordinal: crate::insert_semantic_ir::InsertStatementOrdinal,
    expression_ordinal_base: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedSequenceTarget {
    target_table_oid: u32,
    row_ordinal: u32,
    catalog_column_ordinal: u32,
    column_id: u32,
    sequence_oid: u32,
    source_name: Box<str>,
    effective_name: Box<str>,
    statement_ordinal: crate::insert_semantic_ir::InsertStatementOrdinal,
    local_expression_ordinal: u32,
    absolute_expression_ordinal: u32,
    input_digest: gpu_db_wal::CanonicalDigest,
    descriptor_digest: gpu_db_wal::CanonicalDigest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequenceLifetimeOrigin {
    Published,
    Private,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrivateValueOwner {
    Create,
    Restart,
    TruncateRestart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrivateValueOwnerIdentity {
    kind: PrivateValueOwner,
    statement_ordinal: u32,
    statement_digest: gpu_db_wal::CanonicalDigest,
    creator_catalog_column_ordinal: Option<u32>,
}

#[derive(Debug)]
#[allow(dead_code)] // Production stores this move-only plan before the terminal owner is added.
enum PlannedSequenceEffect {
    /// A published identity has no private value to speculate over in this inert slice.
    Published { target: PlannedSequenceTarget },
    /// A private state exists in the captured explicit transaction and can be planned exactly.
    Private(PlannedPrivateSequenceEffect),
}

#[derive(Debug)]
#[allow(dead_code)] // Production stores this move-only plan before the terminal owner is added.
struct PlannedPrivateSequenceEffect {
    target: PlannedSequenceTarget,
    lifetime_origin: SequenceLifetimeOrigin,
    prior_state: (i64, bool),
    next_state: (i64, bool),
    output_i32: i32,
    prior_owner: PrivateValueOwnerIdentity,
    predecessor: PlannedPrivatePredecessor,
    /// Domain `GPUDBPRIVATESEQCHILD1`; planning-only, never appended to WAL.
    planned_child_digest: gpu_db_wal::CanonicalDigest,
    /// Domain `GPUDBPRIVATESEQOUTCOME1`; planning-only, never appended to WAL.
    planned_outcome_digest: gpu_db_wal::CanonicalDigest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlannedPrivatePredecessor {
    Lifecycle(PrivateValueOwnerIdentity),
    PlannedOutcome(gpu_db_wal::CanonicalDigest),
}

#[derive(Debug, Clone, Copy)]
struct PlannedPrivateState {
    state: (i64, bool),
    owner: PrivateValueOwnerIdentity,
    predecessor: PlannedPrivatePredecessor,
}

#[derive(Debug, Clone)]
struct FoldedSequence {
    effective_name: String,
    lifetime_origin: SequenceLifetimeOrigin,
    latest_private: Option<((i64, bool), PrivateValueOwnerIdentity)>,
}

pub(super) fn classify_autocommit(
    prepared: &PreparedTypedInsert,
    parent: &InsertEffectParentIdentity,
    catalog: &CatalogSnapshot,
) -> Result<PlannedSequenceEffects, ExecuteError> {
    let parent = planned_parent(parent);
    let mut effects = Vec::new();
    for request in prepared.effect_sequence_requests() {
        let target = planned_target(request, parent)?;
        let sequence = catalog
            .relational_sequences
            .get(target.effective_name.as_ref())
            .ok_or_else(|| baseline_drift("autocommit sequence target left its catalog cut"))?;
        if sequence.oid != target.sequence_oid {
            return Err(baseline_drift(
                "autocommit sequence target stable identity changed in its catalog cut",
            ));
        }
        effects.push(PlannedSequenceEffect::Published { target });
    }
    Ok(PlannedSequenceEffects {
        parent,
        effects: effects.into(),
    })
}

pub(super) fn classify_explicit(
    prepared: &PreparedTypedInsert,
    parent: &InsertEffectParentIdentity,
    captured: &ExplicitCapture,
) -> Result<PlannedSequenceEffects, ExecuteError> {
    let parent = planned_parent(parent);
    let mut folded = published_sequences(&captured.snapshot_catalog)?;
    let mut names = folded
        .iter()
        .map(|(oid, sequence)| (sequence.effective_name.clone(), *oid))
        .collect::<BTreeMap<_, _>>();
    fold_operations(captured, &mut folded, &mut names)?;
    validate_final_private_states(captured, &folded)?;

    let mut planned_states = folded
        .iter()
        .filter_map(|(oid, sequence)| {
            sequence.latest_private.map(|(state, owner)| {
                (
                    *oid,
                    PlannedPrivateState {
                        state,
                        owner,
                        predecessor: PlannedPrivatePredecessor::Lifecycle(owner),
                    },
                )
            })
        })
        .collect::<BTreeMap<_, _>>();
    let mut effects = Vec::new();
    for request in prepared.effect_sequence_requests() {
        let target = planned_target(request, parent)?;
        let folded_sequence = folded.get(&target.sequence_oid).ok_or_else(|| {
            baseline_drift("prepared sequence request names no live stable sequence identity")
        })?;
        if folded_sequence.effective_name != target.effective_name.as_ref()
            || names.get(target.effective_name.as_ref()) != Some(&target.sequence_oid)
        {
            return Err(baseline_drift(
                "prepared sequence request effective name does not match stable-OID fold",
            ));
        }
        let final_sequence = captured
            .transaction_catalog
            .relational_sequences
            .get(target.effective_name.as_ref())
            .ok_or_else(|| baseline_drift("folded sequence left the transaction catalog"))?;
        if final_sequence.oid != target.sequence_oid {
            return Err(baseline_drift(
                "folded sequence stable identity differs from transaction catalog",
            ));
        }
        let Some(prior) = planned_states.get(&target.sequence_oid).copied() else {
            effects.push(PlannedSequenceEffect::Published { target });
            continue;
        };
        let output = if prior.state.1 {
            prior.state.0.checked_add(1).ok_or_else(|| {
                baseline_drift("private sequence request overflows i64 while advancing")
            })?
        } else {
            prior.state.0
        };
        let output_i32 = i32::try_from(output).map_err(|_| {
            baseline_drift("private sequence request result cannot be represented as i32")
        })?;
        let next_state = (output, true);
        let planned_child_digest = planned_private_child_digest(
            &parent,
            &target,
            folded_sequence.lifetime_origin,
            prior.state,
            prior.predecessor,
        );
        let planned_outcome_digest =
            planned_private_outcome_digest(planned_child_digest, prior.owner, output, next_state);
        planned_states.insert(
            target.sequence_oid,
            PlannedPrivateState {
                state: next_state,
                owner: prior.owner,
                predecessor: PlannedPrivatePredecessor::PlannedOutcome(planned_outcome_digest),
            },
        );
        effects.push(PlannedSequenceEffect::Private(
            PlannedPrivateSequenceEffect {
                target,
                lifetime_origin: folded_sequence.lifetime_origin,
                prior_state: prior.state,
                next_state,
                output_i32,
                prior_owner: prior.owner,
                predecessor: prior.predecessor,
                planned_child_digest,
                planned_outcome_digest,
            },
        ));
    }
    Ok(PlannedSequenceEffects {
        parent,
        effects: effects.into(),
    })
}

fn planned_parent(parent: &InsertEffectParentIdentity) -> PlannedSequenceParent {
    PlannedSequenceParent {
        txn_id: parent.txn_id,
        autocommit: parent.autocommit,
        request_digest: parent.request_digest,
        statement_ordinal: parent.statement_ordinal,
        expression_ordinal_base: parent.expression_ordinal_base,
    }
}

fn planned_target(
    request: SequenceDefaultRequestEffectShape<'_>,
    parent: PlannedSequenceParent,
) -> Result<PlannedSequenceTarget, ExecuteError> {
    if parent.txn_id == 0
        || parent.request_digest == [0; 32]
        || request.statement_ordinal() != parent.statement_ordinal
    {
        return Err(baseline_drift(
            "planned sequence request lost its exact parent identity",
        ));
    }
    let absolute_expression_ordinal = parent
        .expression_ordinal_base
        .checked_add(request.expression_ordinal())
        .ok_or_else(|| baseline_drift("sequence expression ordinal overflows parent framing"))?;
    let source_name = request.sequence_source_name();
    let effective_name = request.sequence_effective_name();
    let input_digest = sequence_value_input_digest(SequenceValueInput {
        parent_txn_id: parent.txn_id,
        parent_autocommit: parent.autocommit,
        statement_ordinal: parent.statement_ordinal.as_u32(),
        expression_ordinal: absolute_expression_ordinal,
        parent_request_digest: parent.request_digest,
        source_name,
        operation: BinarySequenceValueOperation::Default,
        set_value: None,
    });
    Ok(PlannedSequenceTarget {
        target_table_oid: request.target_table_oid(),
        row_ordinal: request.row_ordinal(),
        catalog_column_ordinal: request.catalog_column_ordinal(),
        column_id: request.column_id(),
        sequence_oid: request.sequence_oid(),
        source_name: source_name.into(),
        effective_name: effective_name.into(),
        statement_ordinal: request.statement_ordinal(),
        local_expression_ordinal: request.expression_ordinal(),
        absolute_expression_ordinal,
        input_digest,
        descriptor_digest: sequence_descriptor_digest(request.sequence_oid(), effective_name),
    })
}

fn published_sequences(
    catalog: &CatalogSnapshot,
) -> Result<BTreeMap<u32, FoldedSequence>, ExecuteError> {
    let mut sequences = BTreeMap::new();
    for (name, sequence) in &catalog.relational_sequences {
        if sequence.oid == 0
            || sequences
                .insert(
                    sequence.oid,
                    FoldedSequence {
                        effective_name: name.clone(),
                        lifetime_origin: SequenceLifetimeOrigin::Published,
                        latest_private: None,
                    },
                )
                .is_some()
        {
            return Err(baseline_drift(
                "snapshot catalog has non-unique or zero sequence stable identity",
            ));
        }
    }
    Ok(sequences)
}

fn fold_operations(
    captured: &ExplicitCapture,
    folded: &mut BTreeMap<u32, FoldedSequence>,
    names: &mut BTreeMap<String, u32>,
) -> Result<(), ExecuteError> {
    let mut catalog_index = 0u32;
    let published_oids = captured
        .snapshot_catalog
        .relational_sequences
        .values()
        .map(|sequence| sequence.oid)
        .collect::<std::collections::BTreeSet<_>>();
    for (operation_ordinal, operation) in captured.operations.iter().enumerate() {
        let operation_ordinal = u32::try_from(operation_ordinal)
            .map_err(|_| baseline_drift("operation ordinal exceeds stable-OID fold framing"))?;
        match operation {
            TransactionOperation::Catalog(staged) => {
                if staged.ordinal != operation_ordinal {
                    return Err(baseline_drift("catalog operation lost its global ordinal"));
                }
                fold_catalog_operation(staged, catalog_index, &published_oids, folded, names)?;
                catalog_index = catalog_index.checked_add(1).ok_or_else(|| {
                    baseline_drift("catalog command index exceeds stable-OID fold framing")
                })?;
            }
            TransactionOperation::TableReset(reset) => {
                if reset.ordinal != operation_ordinal {
                    return Err(baseline_drift("table reset lost its global ordinal"));
                }
                fold_table_reset(reset, folded, names)?;
            }
            TransactionOperation::Row(staged) => {
                fold_row_advances(staged, folded, names)?;
            }
        }
    }
    Ok(())
}

fn fold_catalog_operation(
    staged: &crate::StagedCatalogCommand,
    catalog_index: u32,
    published_oids: &std::collections::BTreeSet<u32>,
    folded: &mut BTreeMap<u32, FoldedSequence>,
    names: &mut BTreeMap<String, u32>,
) -> Result<(), ExecuteError> {
    if command_is_sequence_lifecycle(&staged.command) {
        let identity = staged
            .sequence_identity
            .as_ref()
            .ok_or_else(|| baseline_drift("sequence lifecycle command lost its stable identity"))?;
        if identity.command_index != catalog_index
            || identity.ordinal != staged.ordinal
            || !valid_sequence_lifecycle_operation_identity(&staged.command, identity)
        {
            return Err(baseline_drift(
                "sequence lifecycle command has invalid ordinal or stable identity",
            ));
        }
        match &staged.command {
            Command::CreateSequence(create) => {
                let [target] = identity.targets.as_slice() else {
                    return Err(baseline_drift(
                        "CREATE SEQUENCE identity has unexpected targets",
                    ));
                };
                let (name, oid) = target_after(target)?;
                if name != create.name || target.target_before.is_some() {
                    return Err(baseline_drift(
                        "CREATE SEQUENCE identity does not match command",
                    ));
                }
                insert_private_sequence(
                    folded,
                    names,
                    oid,
                    name,
                    (1, false),
                    catalog_owner(staged, PrivateValueOwner::Create, None),
                )?;
            }
            Command::RenameSequence(rename) => {
                let [target] = identity.targets.as_slice() else {
                    return Err(baseline_drift(
                        "RENAME SEQUENCE identity has unexpected targets",
                    ));
                };
                let (before_name, before_oid) = target_before(target)?;
                let (after_name, after_oid) = target_after(target)?;
                if before_name != rename.old_name
                    || after_name != rename.new_name
                    || before_oid != after_oid
                {
                    return Err(baseline_drift(
                        "RENAME SEQUENCE identity does not match command",
                    ));
                }
                rename_sequence(folded, names, before_oid, before_name, after_name)?;
            }
            Command::SequenceRestart(restart) => {
                let [target] = identity.targets.as_slice() else {
                    return Err(baseline_drift(
                        "RESTART SEQUENCE identity has unexpected targets",
                    ));
                };
                let (before_name, before_oid) = target_before(target)?;
                let (after_name, after_oid) = target_after(target)?;
                if before_name != restart.name
                    || after_name != restart.name
                    || before_oid != after_oid
                {
                    return Err(baseline_drift(
                        "RESTART SEQUENCE identity does not match command",
                    ));
                }
                set_private_state(
                    folded,
                    names,
                    before_oid,
                    before_name,
                    (restart.value, false),
                    catalog_owner(staged, PrivateValueOwner::Restart, None),
                )?;
            }
            Command::DropSequence(drop) => {
                if identity.targets.len() != drop.names.len() {
                    return Err(baseline_drift(
                        "DROP SEQUENCE identity target count changed",
                    ));
                }
                for (target, name) in identity.targets.iter().zip(&drop.names) {
                    if target.target_before.is_none() {
                        if target.before_name != *name
                            || target.after_name.is_some()
                            || target.target_after.is_some()
                        {
                            return Err(baseline_drift(
                                "DROP SEQUENCE IF EXISTS absence identity does not match command",
                            ));
                        }
                        continue;
                    }
                    let (before_name, oid) = target_before(target)?;
                    if before_name != *name
                        || target.after_name.is_some()
                        || target.target_after.is_some()
                    {
                        return Err(baseline_drift(
                            "DROP SEQUENCE identity does not match command",
                        ));
                    }
                    remove_sequence(folded, names, oid, before_name)?;
                }
            }
            _ => unreachable!("sequence lifecycle classifier admitted a non-sequence command"),
        }
        return Ok(());
    }
    if staged.sequence_identity.is_some() {
        return Err(baseline_drift(
            "non-sequence catalog command carries a sequence lifecycle identity",
        ));
    }
    if let Command::CreateTable(create) = &staged.command {
        let mut expected: BTreeMap<String, (bool, u32)> = BTreeMap::new();
        for (column_ordinal, column) in create.columns.iter().enumerate() {
            let Some(crate::ColumnDefault::SequenceNextVal {
                sequence,
                create_if_missing,
            }) = &column.default
            else {
                continue;
            };
            let column_ordinal = u32::try_from(column_ordinal).map_err(|_| {
                baseline_drift("CREATE TABLE creator column ordinal exceeds stable-OID framing")
            })?;
            expected
                .entry(sequence.clone())
                .and_modify(|(prior_create_if_missing, creator_column_ordinal)| {
                    if *create_if_missing && !*prior_create_if_missing {
                        *prior_create_if_missing = true;
                        *creator_column_ordinal = column_ordinal;
                    }
                })
                .or_insert((*create_if_missing, column_ordinal));
        }
        if staged.sequence_input_oids.len() != expected.len() {
            return Err(baseline_drift(
                "CREATE TABLE sequence inputs changed stable name geometry",
            ));
        }
        for (sequence_name, (create_if_missing, creator_column_ordinal)) in expected {
            let oid = staged
                .sequence_input_oids
                .get(&sequence_name)
                .copied()
                .ok_or_else(|| {
                    baseline_drift("CREATE TABLE sequence default lost its stable identity")
                })?;
            if create_if_missing && !published_oids.contains(&oid) {
                insert_private_sequence(
                    folded,
                    names,
                    oid,
                    sequence_name,
                    (1, false),
                    catalog_owner(
                        staged,
                        PrivateValueOwner::Create,
                        Some(creator_column_ordinal),
                    ),
                )?;
            } else if names.get(&sequence_name) != Some(&oid) {
                return Err(baseline_drift(
                    "CREATE TABLE resolved published sequence identity under another effective name",
                ));
            }
        }
    } else if !staged.sequence_input_oids.is_empty() {
        return Err(baseline_drift(
            "non-CREATE TABLE catalog command carries sequence input stable identities",
        ));
    }
    Ok(())
}

fn fold_table_reset(
    reset: &crate::StagedTableReset,
    folded: &mut BTreeMap<u32, FoldedSequence>,
    names: &mut BTreeMap<String, u32>,
) -> Result<(), ExecuteError> {
    let Some(identity) = reset.sequence_reset_identity.as_ref() else {
        return Ok(());
    };
    if identity.ordinal != reset.ordinal
        || identity.table != reset.table
        || !valid_sequence_reset_operation_identity(identity)
    {
        return Err(baseline_drift(
            "TRUNCATE RESTART IDENTITY lost its ordinal or table stable identity",
        ));
    }
    for target in &identity.targets {
        let (name, oid) = target_before(target)?;
        let (_, after_oid) = target_after(target)?;
        if oid != after_oid {
            return Err(baseline_drift(
                "TRUNCATE RESTART IDENTITY target changed stable sequence identity",
            ));
        }
        set_private_state(
            folded,
            names,
            oid,
            name,
            (1, false),
            reset_owner(reset, PrivateValueOwner::TruncateRestart),
        )?;
    }
    Ok(())
}

fn fold_row_advances(
    staged: &crate::StagedRowOperation,
    folded: &mut BTreeMap<u32, FoldedSequence>,
    names: &BTreeMap<String, u32>,
) -> Result<(), ExecuteError> {
    let PreparedMutation::Insert { seq_advances, .. } = &staged.mutation else {
        if !staged.sequence_input_oids.is_empty() {
            return Err(baseline_drift(
                "non-INSERT row operation carries sequence input stable identities",
            ));
        }
        return Ok(());
    };
    if staged.sequence_input_oids.len() != seq_advances.len() {
        return Err(baseline_drift(
            "row sequence advances and stable inputs have different geometry",
        ));
    }
    for (name, state) in seq_advances {
        let oid = staged
            .sequence_input_oids
            .get(name)
            .copied()
            .ok_or_else(|| baseline_drift("row sequence advance lost its stable input identity"))?;
        if names.get(name) != Some(&oid) {
            return Err(baseline_drift(
                "row sequence advance does not match its effective stable-OID binding",
            ));
        }
        let sequence = folded.get_mut(&oid).ok_or_else(|| {
            baseline_drift("row sequence advance names a dropped or unknown stable identity")
        })?;
        let (prior_state, owner) = sequence.latest_private.ok_or_else(|| {
            baseline_drift("row sequence advance cannot speculate over a published sequence")
        })?;
        if !state.1
            || if prior_state.1 {
                state.0 <= prior_state.0
            } else {
                state.0 < prior_state.0
            }
        {
            return Err(baseline_drift(
                "row sequence advance is not reachable from its prior private state",
            ));
        }
        sequence.latest_private = Some((*state, owner));
    }
    Ok(())
}

fn insert_private_sequence(
    folded: &mut BTreeMap<u32, FoldedSequence>,
    names: &mut BTreeMap<String, u32>,
    oid: u32,
    effective_name: String,
    state: (i64, bool),
    owner: PrivateValueOwnerIdentity,
) -> Result<(), ExecuteError> {
    if oid == 0 || folded.contains_key(&oid) || names.insert(effective_name.clone(), oid).is_some()
    {
        return Err(baseline_drift(
            "private sequence creation collided with an existing stable identity",
        ));
    }
    folded.insert(
        oid,
        FoldedSequence {
            effective_name,
            lifetime_origin: SequenceLifetimeOrigin::Private,
            latest_private: Some((state, owner)),
        },
    );
    Ok(())
}

fn rename_sequence(
    folded: &mut BTreeMap<u32, FoldedSequence>,
    names: &mut BTreeMap<String, u32>,
    oid: u32,
    before_name: String,
    after_name: String,
) -> Result<(), ExecuteError> {
    if names.remove(&before_name) != Some(oid) || names.contains_key(&after_name) {
        return Err(baseline_drift(
            "sequence rename does not preserve the folded stable identity",
        ));
    }
    let sequence = folded
        .get_mut(&oid)
        .ok_or_else(|| baseline_drift("sequence rename names an unknown stable identity"))?;
    if sequence.effective_name != before_name {
        return Err(baseline_drift(
            "sequence rename effective name differs from stable-OID fold",
        ));
    }
    sequence.effective_name = after_name.clone();
    names.insert(after_name, oid);
    Ok(())
}

fn set_private_state(
    folded: &mut BTreeMap<u32, FoldedSequence>,
    names: &BTreeMap<String, u32>,
    oid: u32,
    effective_name: String,
    state: (i64, bool),
    owner: PrivateValueOwnerIdentity,
) -> Result<(), ExecuteError> {
    if names.get(&effective_name) != Some(&oid) {
        return Err(baseline_drift(
            "private sequence state owner does not match effective stable-OID binding",
        ));
    }
    let sequence = folded
        .get_mut(&oid)
        .ok_or_else(|| baseline_drift("private sequence state names an unknown stable identity"))?;
    sequence.latest_private = Some((state, owner));
    Ok(())
}

fn remove_sequence(
    folded: &mut BTreeMap<u32, FoldedSequence>,
    names: &mut BTreeMap<String, u32>,
    oid: u32,
    effective_name: String,
) -> Result<(), ExecuteError> {
    if names.remove(&effective_name) != Some(oid)
        || folded
            .get(&oid)
            .is_none_or(|sequence| sequence.effective_name != effective_name)
    {
        return Err(baseline_drift(
            "sequence drop does not match the folded stable identity",
        ));
    }
    folded.remove(&oid);
    Ok(())
}

fn catalog_owner(
    staged: &crate::StagedCatalogCommand,
    kind: PrivateValueOwner,
    creator_catalog_column_ordinal: Option<u32>,
) -> PrivateValueOwnerIdentity {
    PrivateValueOwnerIdentity {
        kind,
        statement_ordinal: staged.ordinal,
        statement_digest: staged.statement_digest,
        creator_catalog_column_ordinal,
    }
}

fn reset_owner(
    reset: &crate::StagedTableReset,
    kind: PrivateValueOwner,
) -> PrivateValueOwnerIdentity {
    PrivateValueOwnerIdentity {
        kind,
        statement_ordinal: reset.ordinal,
        statement_digest: reset.statement_digest,
        creator_catalog_column_ordinal: None,
    }
}

fn target_before(
    target: &crate::BinaryTransactionSequenceLifecycleTargetIdentity,
) -> Result<(String, u32), ExecuteError> {
    let identity = target.target_before.as_ref().ok_or_else(|| {
        baseline_drift("sequence lifecycle target lost its stable preimage identity")
    })?;
    if identity.kind != BinaryCatalogRelationKind::Sequence || identity.oid == 0 {
        return Err(baseline_drift(
            "sequence lifecycle target preimage is not a stable sequence identity",
        ));
    }
    Ok((target.before_name.clone(), identity.oid))
}

fn target_after(
    target: &crate::BinaryTransactionSequenceLifecycleTargetIdentity,
) -> Result<(String, u32), ExecuteError> {
    let name = target.after_name.clone().ok_or_else(|| {
        baseline_drift("sequence lifecycle target lost its effective postimage name")
    })?;
    let identity = target.target_after.as_ref().ok_or_else(|| {
        baseline_drift("sequence lifecycle target lost its stable postimage identity")
    })?;
    if identity.kind != BinaryCatalogRelationKind::Sequence || identity.oid == 0 {
        return Err(baseline_drift(
            "sequence lifecycle target postimage is not a stable sequence identity",
        ));
    }
    Ok((name, identity.oid))
}

fn validate_final_private_states(
    captured: &ExplicitCapture,
    folded: &BTreeMap<u32, FoldedSequence>,
) -> Result<(), ExecuteError> {
    let expected_live = folded
        .iter()
        .map(|(oid, sequence)| (sequence.effective_name.clone(), *oid))
        .collect::<BTreeMap<_, _>>();
    let captured_live = captured
        .transaction_catalog
        .relational_sequences
        .iter()
        .map(|(name, sequence)| (name.clone(), sequence.oid))
        .collect::<BTreeMap<_, _>>();
    if expected_live != captured_live {
        return Err(baseline_drift(
            "stable-OID sequence fold disagrees with the captured transaction catalog",
        ));
    }
    let expected_by_oid = folded
        .iter()
        .filter_map(|(oid, sequence)| sequence.latest_private.map(|(state, _)| (*oid, state)))
        .collect::<BTreeMap<_, _>>();
    let expected_by_name = folded
        .values()
        .filter_map(|sequence| {
            sequence
                .latest_private
                .map(|(state, _)| (sequence.effective_name.clone(), state))
        })
        .collect::<BTreeMap<_, _>>();
    if expected_by_oid != captured.sequence_state_by_oid
        || expected_by_name != captured.sequence_state
    {
        return Err(baseline_drift(
            "stable-OID sequence fold disagrees with captured private state maps",
        ));
    }
    Ok(())
}

fn planned_private_child_digest(
    parent: &PlannedSequenceParent,
    target: &PlannedSequenceTarget,
    lifetime_origin: SequenceLifetimeOrigin,
    prior_state: (i64, bool),
    predecessor: PlannedPrivatePredecessor,
) -> gpu_db_wal::CanonicalDigest {
    let mut body = Vec::with_capacity(176 + target.source_name.len() + target.effective_name.len());
    body.extend_from_slice(b"GPUDBPRIVATESEQCHILD1");
    append_parent(&mut body, parent);
    append_target(&mut body, target);
    body.push(match lifetime_origin {
        SequenceLifetimeOrigin::Published => 1,
        SequenceLifetimeOrigin::Private => 2,
    });
    body.extend_from_slice(&prior_state.0.to_le_bytes());
    body.push(u8::from(prior_state.1));
    append_private_predecessor(&mut body, predecessor);
    gpu_db_wal::canonical_request_digest(&body)
}

fn planned_private_outcome_digest(
    child_digest: gpu_db_wal::CanonicalDigest,
    owner: PrivateValueOwnerIdentity,
    output: i64,
    next_state: (i64, bool),
) -> gpu_db_wal::CanonicalDigest {
    let mut body = Vec::with_capacity(80);
    body.extend_from_slice(b"GPUDBPRIVATESEQOUTCOME1");
    body.extend_from_slice(&child_digest);
    append_private_owner(&mut body, owner);
    body.extend_from_slice(&output.to_le_bytes());
    body.extend_from_slice(&next_state.0.to_le_bytes());
    body.push(u8::from(next_state.1));
    gpu_db_wal::canonical_request_digest(&body)
}

fn append_parent(body: &mut Vec<u8>, parent: &PlannedSequenceParent) {
    body.extend_from_slice(&parent.txn_id.to_le_bytes());
    body.push(u8::from(parent.autocommit));
    body.extend_from_slice(&parent.request_digest);
    body.extend_from_slice(&parent.statement_ordinal.as_u32().to_le_bytes());
    body.extend_from_slice(&parent.expression_ordinal_base.to_le_bytes());
}

fn append_target(body: &mut Vec<u8>, target: &PlannedSequenceTarget) {
    body.extend_from_slice(&target.target_table_oid.to_le_bytes());
    body.extend_from_slice(&target.row_ordinal.to_le_bytes());
    body.extend_from_slice(&target.catalog_column_ordinal.to_le_bytes());
    body.extend_from_slice(&target.column_id.to_le_bytes());
    body.extend_from_slice(&target.sequence_oid.to_le_bytes());
    append_string(body, &target.source_name);
    append_string(body, &target.effective_name);
    body.extend_from_slice(&target.statement_ordinal.as_u32().to_le_bytes());
    body.extend_from_slice(&target.local_expression_ordinal.to_le_bytes());
    body.extend_from_slice(&target.absolute_expression_ordinal.to_le_bytes());
    body.extend_from_slice(&target.input_digest);
    body.extend_from_slice(&target.descriptor_digest);
}

fn append_private_owner(body: &mut Vec<u8>, owner: PrivateValueOwnerIdentity) {
    body.push(match owner.kind {
        PrivateValueOwner::Create => 1,
        PrivateValueOwner::Restart => 2,
        PrivateValueOwner::TruncateRestart => 3,
    });
    body.extend_from_slice(&owner.statement_ordinal.to_le_bytes());
    body.extend_from_slice(&owner.statement_digest);
    match owner.creator_catalog_column_ordinal {
        Some(ordinal) => {
            body.push(1);
            body.extend_from_slice(&ordinal.to_le_bytes());
        }
        None => body.push(0),
    }
}

fn append_private_predecessor(body: &mut Vec<u8>, predecessor: PlannedPrivatePredecessor) {
    match predecessor {
        PlannedPrivatePredecessor::Lifecycle(owner) => {
            body.push(1);
            append_private_owner(body, owner);
        }
        PlannedPrivatePredecessor::PlannedOutcome(digest) => {
            body.push(2);
            body.extend_from_slice(&digest);
        }
    }
}

fn append_string(body: &mut Vec<u8>, value: &str) {
    body.extend_from_slice(&(value.len() as u64).to_le_bytes());
    body.extend_from_slice(value.as_bytes());
}

/// Turn the classifier plan into exact seal bindings only in test builds. This is intentionally
/// the sole WRITE-001 construction point: published inputs are caller supplied, while every
/// private scalar and digest comes from the stable-OID plan and is independently re-proved here.
#[cfg(test)]
pub(super) fn build_seal_bindings_for_test(
    prepared: &PreparedTypedInsert,
    parent: &InsertEffectParentIdentity,
    planned: &PlannedSequenceEffects,
    receipts: SequenceReceiptBundle,
) -> Result<SequenceSealBindingBundle, ExecuteError> {
    if planned.parent != planned_parent(parent) {
        return Err(baseline_drift(
            "sequence seal parent differs from its classifier plan",
        ));
    }
    let requests = prepared.sequence_requests();
    if planned.effects.len() != requests.len() {
        return Err(baseline_drift(
            "sequence seal effect count differs from typed request geometry",
        ));
    }
    let binding_parent = SequenceDefaultParentContext::for_test(
        parent.txn_id,
        parent.autocommit,
        parent.request_digest,
        parent.statement_ordinal,
        parent.expression_ordinal_base,
    );
    let mut published = receipts.published.into_vec().into_iter();
    let mut transition_ids = BTreeSet::new();
    let mut target_slots = BTreeSet::new();
    let mut private_chains = BTreeMap::new();
    let mut bindings = Vec::with_capacity(requests.len());
    let mut outputs = Vec::with_capacity(requests.len());
    for ((request, request_shape), effect) in requests
        .iter()
        .zip(prepared.effect_sequence_requests())
        .zip(planned.effects.iter())
    {
        let expected_target = planned_target(request_shape, planned.parent)?;
        let target = match effect {
            PlannedSequenceEffect::Published { target } => target,
            PlannedSequenceEffect::Private(effect) => &effect.target,
        };
        if *target != expected_target
            || !target_slots.insert((target.sequence_oid, target.absolute_expression_ordinal))
        {
            return Err(baseline_drift(
                "sequence seal target or absolute expression order drifted",
            ));
        }
        match effect {
            PlannedSequenceEffect::Published { target } => {
                let PublishedReceiptInput {
                    transition_txn_id,
                    returned_value,
                    input_digest,
                } = published.next().ok_or_else(|| {
                    baseline_drift("published sequence seal receipt count is incomplete")
                })?;
                if transition_txn_id == 0
                    || !transition_ids.insert(transition_txn_id)
                    || input_digest != target.input_digest
                    || i32::try_from(returned_value).is_err()
                {
                    return Err(baseline_drift(
                        "published sequence seal receipt identity, input, or value drifted",
                    ));
                }
                bindings.push(SequenceDefaultBinding::published_exact_for_test(
                    request.clone(),
                    binding_parent.clone(),
                    returned_value,
                    transition_txn_id,
                    input_digest,
                ));
                outputs.push(SequenceOutputEvidence {
                    sequence_oid: target.sequence_oid,
                    row_ordinal: target.row_ordinal,
                    catalog_column_ordinal: target.catalog_column_ordinal,
                    column_id: target.column_id,
                    local_expression_ordinal: target.local_expression_ordinal,
                    absolute_expression_ordinal: target.absolute_expression_ordinal,
                    input_digest: target.input_digest,
                    descriptor_digest: target.descriptor_digest,
                    returned_value: i32::try_from(returned_value)
                        .expect("validated published sequence value fits i32"),
                    transition: SequenceTransitionEvidence::Published { transition_txn_id },
                });
            }
            PlannedSequenceEffect::Private(effect) => {
                validate_private_effect_for_terminal(effect, planned.parent, &mut private_chains)?;
                let owner = private_owner_evidence(effect.prior_owner);
                let predecessor = private_predecessor_evidence(effect.predecessor);
                let planning = PrivateSequencePlanningEvidence::exact_for_test(
                    lifetime_origin_tag(effect.lifetime_origin),
                    owner.kind,
                    owner.statement_ordinal,
                    owner.statement_digest,
                    owner.creator_catalog_column_ordinal,
                    predecessor_tag(effect.predecessor),
                    predecessor_digest(effect.predecessor),
                    effect.target.input_digest,
                    effect.target.descriptor_digest,
                    effect.planned_child_digest,
                    effect.planned_outcome_digest,
                );
                bindings.push(SequenceDefaultBinding::private_exact_for_test(
                    request.clone(),
                    binding_parent.clone(),
                    i64::from(effect.output_i32),
                    effect.prior_state,
                    effect.next_state,
                    planning,
                ));
                outputs.push(SequenceOutputEvidence {
                    sequence_oid: effect.target.sequence_oid,
                    row_ordinal: effect.target.row_ordinal,
                    catalog_column_ordinal: effect.target.catalog_column_ordinal,
                    column_id: effect.target.column_id,
                    local_expression_ordinal: effect.target.local_expression_ordinal,
                    absolute_expression_ordinal: effect.target.absolute_expression_ordinal,
                    input_digest: effect.target.input_digest,
                    descriptor_digest: effect.target.descriptor_digest,
                    returned_value: effect.output_i32,
                    transition: SequenceTransitionEvidence::Private {
                        lifetime_origin: lifetime_origin_tag(effect.lifetime_origin),
                        prior_state: effect.prior_state,
                        next_state: effect.next_state,
                        owner,
                        predecessor,
                        child_digest: effect.planned_child_digest,
                        outcome_digest: effect.planned_outcome_digest,
                    },
                });
            }
        }
    }
    if published.next().is_some() {
        return Err(baseline_drift(
            "published sequence seal receipt count has trailing entries",
        ));
    }
    let bindings = if bindings.is_empty() {
        SequenceDefaultBindings::empty()
    } else {
        SequenceDefaultBindings::from_bindings(binding_parent, bindings)
    };
    prepared.validate_sequence_bindings_for_terminal(&bindings)?;
    Ok(SequenceSealBindingBundle {
        bindings,
        outputs: outputs.into(),
    })
}

#[cfg(test)]
fn validate_private_effect_for_terminal(
    effect: &PlannedPrivateSequenceEffect,
    parent: PlannedSequenceParent,
    chains: &mut BTreeMap<
        u32,
        (
            (i64, bool),
            PrivateValueOwnerIdentity,
            gpu_db_wal::CanonicalDigest,
        ),
    >,
) -> Result<(), ExecuteError> {
    if !private_lifetime_owner_is_compatible(effect.lifetime_origin, effect.prior_owner) {
        return Err(baseline_drift(
            "private sequence seal lifetime origin and owner provenance are incompatible",
        ));
    }
    let output = if effect.prior_state.1 {
        effect
            .prior_state
            .0
            .checked_add(1)
            .ok_or_else(|| baseline_drift("private sequence seal advance overflows i64"))?
    } else {
        effect.prior_state.0
    };
    if i32::try_from(output) != Ok(effect.output_i32)
        || effect.next_state != (output, true)
        || effect.target.descriptor_digest
            != sequence_descriptor_digest(effect.target.sequence_oid, &effect.target.effective_name)
        || effect.planned_child_digest
            != planned_private_child_digest(
                &parent,
                &effect.target,
                effect.lifetime_origin,
                effect.prior_state,
                effect.predecessor,
            )
        || effect.planned_outcome_digest
            != planned_private_outcome_digest(
                effect.planned_child_digest,
                effect.prior_owner,
                output,
                effect.next_state,
            )
    {
        return Err(baseline_drift(
            "private sequence seal state, descriptor, or child witness drifted",
        ));
    }
    if let Some((prior_state, owner, outcome_digest)) = chains.get(&effect.target.sequence_oid) {
        if effect.prior_state != *prior_state
            || effect.prior_owner != *owner
            || effect.predecessor != PlannedPrivatePredecessor::PlannedOutcome(*outcome_digest)
        {
            return Err(baseline_drift(
                "private sequence seal chain predecessor or owner drifted",
            ));
        }
    } else if effect.predecessor != PlannedPrivatePredecessor::Lifecycle(effect.prior_owner) {
        return Err(baseline_drift(
            "private sequence seal initial predecessor is not its lifecycle owner",
        ));
    }
    chains.insert(
        effect.target.sequence_oid,
        (
            effect.next_state,
            effect.prior_owner,
            effect.planned_outcome_digest,
        ),
    );
    Ok(())
}

#[cfg(test)]
fn private_lifetime_owner_is_compatible(
    lifetime_origin: SequenceLifetimeOrigin,
    owner: PrivateValueOwnerIdentity,
) -> bool {
    match (
        lifetime_origin,
        owner.kind,
        owner.creator_catalog_column_ordinal,
    ) {
        // A pre-existing catalog sequence can be restarted or truncate-restarted in an explicit
        // transaction, but can never be the creation-owned private lifetime.
        (SequenceLifetimeOrigin::Published, PrivateValueOwner::Create, _) => false,
        // `creator_catalog_column_ordinal` identifies the implicit serial-like CREATE TABLE
        // owner. Restart owners and truncate-restart owners cannot retain that provenance.
        (_, PrivateValueOwner::Restart | PrivateValueOwner::TruncateRestart, Some(_)) => false,
        // A private lifetime is created, restarted, or truncate-restarted in the transaction;
        // only its Create form may carry the optional creator-column provenance.
        _ => true,
    }
}

#[cfg(test)]
fn lifetime_origin_tag(origin: SequenceLifetimeOrigin) -> u8 {
    match origin {
        SequenceLifetimeOrigin::Published => 1,
        SequenceLifetimeOrigin::Private => 2,
    }
}

#[cfg(test)]
fn private_owner_evidence(owner: PrivateValueOwnerIdentity) -> PrivateOwnerEvidence {
    PrivateOwnerEvidence {
        kind: match owner.kind {
            PrivateValueOwner::Create => 1,
            PrivateValueOwner::Restart => 2,
            PrivateValueOwner::TruncateRestart => 3,
        },
        statement_ordinal: owner.statement_ordinal,
        statement_digest: owner.statement_digest,
        creator_catalog_column_ordinal: owner.creator_catalog_column_ordinal,
    }
}

#[cfg(test)]
fn private_predecessor_evidence(
    predecessor: PlannedPrivatePredecessor,
) -> PrivatePredecessorEvidence {
    match predecessor {
        PlannedPrivatePredecessor::Lifecycle(owner) => {
            PrivatePredecessorEvidence::Lifecycle(private_owner_evidence(owner))
        }
        PlannedPrivatePredecessor::PlannedOutcome(digest) => {
            PrivatePredecessorEvidence::PlannedOutcome(digest)
        }
    }
}

#[cfg(test)]
fn predecessor_tag(predecessor: PlannedPrivatePredecessor) -> u8 {
    match predecessor {
        PlannedPrivatePredecessor::Lifecycle(_) => 1,
        PlannedPrivatePredecessor::PlannedOutcome(_) => 2,
    }
}

#[cfg(test)]
fn predecessor_digest(predecessor: PlannedPrivatePredecessor) -> gpu_db_wal::CanonicalDigest {
    match predecessor {
        PlannedPrivatePredecessor::Lifecycle(owner) => owner.statement_digest,
        PlannedPrivatePredecessor::PlannedOutcome(digest) => digest,
    }
}

#[cfg(test)]
pub(super) fn published_input_digests_for_test(
    planned: &PlannedSequenceEffects,
) -> Vec<gpu_db_wal::CanonicalDigest> {
    planned
        .effects
        .iter()
        .filter_map(|effect| match effect {
            PlannedSequenceEffect::Published { target } => Some(target.input_digest),
            PlannedSequenceEffect::Private(_) => None,
        })
        .collect()
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(super) enum TerminalPlanSabotage {
    EffectCount,
    EffectOrder,
    Target,
    Descriptor,
    Child,
    Outcome,
    Owner,
    Predecessor,
    PrivateValue,
    LifetimeOrigin,
    PrivateInput,
    RehashedPublishedCreateOwner,
    RehashedNonCreateOwnerOrdinal,
}

#[cfg(test)]
pub(super) fn sabotage_for_terminal_test(
    planned: &mut PlannedSequenceEffects,
    sabotage: TerminalPlanSabotage,
) {
    match sabotage {
        TerminalPlanSabotage::EffectCount => {
            let mut effects = std::mem::take(&mut planned.effects).into_vec();
            effects.pop();
            planned.effects = effects.into();
        }
        TerminalPlanSabotage::EffectOrder => {
            let effects = planned.effects.as_mut();
            assert!(
                effects.len() >= 2,
                "effect-order sabotage needs two effects"
            );
            effects.swap(0, 1);
        }
        TerminalPlanSabotage::Target => {
            let target = planned
                .effects
                .iter_mut()
                .map(|effect| match effect {
                    PlannedSequenceEffect::Published { target } => Some(target),
                    PlannedSequenceEffect::Private(effect) => Some(&mut effect.target),
                })
                .next()
                .flatten()
                .expect("terminal sabotage fixture has one sequence effect");
            target.sequence_oid = target.sequence_oid.saturating_add(1);
        }
        TerminalPlanSabotage::Descriptor => {
            let effect = first_private_effect_mut(planned);
            effect.target.descriptor_digest[0] ^= 0x5a;
        }
        TerminalPlanSabotage::Child => {
            first_private_effect_mut(planned).planned_child_digest[0] ^= 0x5a;
        }
        TerminalPlanSabotage::Outcome => {
            first_private_effect_mut(planned).planned_outcome_digest[0] ^= 0x5a;
        }
        TerminalPlanSabotage::Owner => {
            let effect = first_private_effect_mut(planned);
            effect.prior_owner.statement_ordinal =
                effect.prior_owner.statement_ordinal.saturating_add(1);
        }
        TerminalPlanSabotage::Predecessor => {
            first_private_effect_mut(planned).predecessor =
                PlannedPrivatePredecessor::PlannedOutcome([9; 32]);
        }
        TerminalPlanSabotage::PrivateValue => {
            first_private_effect_mut(planned).output_i32 = 99;
        }
        TerminalPlanSabotage::LifetimeOrigin => {
            let effect = first_private_effect_mut(planned);
            effect.lifetime_origin = match effect.lifetime_origin {
                SequenceLifetimeOrigin::Published => SequenceLifetimeOrigin::Private,
                SequenceLifetimeOrigin::Private => SequenceLifetimeOrigin::Published,
            };
        }
        TerminalPlanSabotage::PrivateInput => {
            first_private_effect_mut(planned).target.input_digest[0] ^= 0x5a;
        }
        TerminalPlanSabotage::RehashedPublishedCreateOwner => {
            let parent = planned.parent;
            let effect = first_private_effect_mut(planned);
            effect.lifetime_origin = SequenceLifetimeOrigin::Published;
            effect.prior_owner.kind = PrivateValueOwner::Create;
            effect.prior_owner.creator_catalog_column_ordinal = Some(0);
            effect.predecessor = PlannedPrivatePredecessor::Lifecycle(effect.prior_owner);
            rehash_private_effect_for_terminal(effect, parent);
        }
        TerminalPlanSabotage::RehashedNonCreateOwnerOrdinal => {
            let parent = planned.parent;
            let effect = first_private_effect_mut(planned);
            effect.lifetime_origin = SequenceLifetimeOrigin::Private;
            effect.prior_owner.kind = PrivateValueOwner::Restart;
            effect.prior_owner.creator_catalog_column_ordinal = Some(0);
            effect.predecessor = PlannedPrivatePredecessor::Lifecycle(effect.prior_owner);
            rehash_private_effect_for_terminal(effect, parent);
        }
    }
}

#[cfg(test)]
fn rehash_private_effect_for_terminal(
    effect: &mut PlannedPrivateSequenceEffect,
    parent: PlannedSequenceParent,
) {
    effect.planned_child_digest = planned_private_child_digest(
        &parent,
        &effect.target,
        effect.lifetime_origin,
        effect.prior_state,
        effect.predecessor,
    );
    effect.planned_outcome_digest = planned_private_outcome_digest(
        effect.planned_child_digest,
        effect.prior_owner,
        i64::from(effect.output_i32),
        effect.next_state,
    );
}

#[cfg(test)]
fn first_private_effect_mut(
    planned: &mut PlannedSequenceEffects,
) -> &mut PlannedPrivateSequenceEffect {
    planned
        .effects
        .iter_mut()
        .find_map(|effect| match effect {
            PlannedSequenceEffect::Published { .. } => None,
            PlannedSequenceEffect::Private(effect) => Some(effect),
        })
        .expect("terminal sabotage fixture has a private sequence effect")
}

#[cfg(test)]
#[path = "classification_tests.rs"]
mod tests;
