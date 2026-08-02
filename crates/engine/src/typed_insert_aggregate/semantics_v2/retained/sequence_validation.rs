//! Durable published-sequence outcome proof between catalog/allocator closure and generation.
//!
//! The retained graph already proves the S5-only row tuple against S2/S4/S7.  This module does
//! not recreate that tuple from the Engine outcome index.  It borrows the one immutable outcome
//! index once, authenticates its complete checkpoint-pinned view, and binds each retained S5
//! reference to the already-published `Default` transition that precedes the enclosing parent.

use super::{graph::ReservedSemanticsV2Graph, SemanticsV2BoundIdentity};
use crate::typed_insert_batch::DecodedSequenceEffectKindFacts;
use sha2::{Digest, Sha256};

/// Concrete borrowed authority over the Engine's recovered `sequence_value_outcomes` index.
///
/// The future live bridge must form this only while holding the Engine's sole recovered outcome
/// authority.  This inert checkpoint deliberately provides no production constructor, clone, or
/// decoded-body substitute.
pub(in crate::typed_insert_aggregate::semantics_v2) struct SemanticsV2DurableSequenceOutcomeIndexProof<
    'a,
> {
    snapshot: &'a SemanticsV2ImmutableDurableSequenceOutcomeIndex<'a>,
    checkpoint_pin: &'a SemanticsV2PinnedSequenceOutcomeIndexGeneration,
}

/// Authenticated immutable view of every recovered ordinary sequence transition in one lineage.
/// Complete/durable/published frontiers describe the index as a whole; individual rows do not
/// self-attest lifecycle state.
#[allow(dead_code)]
pub(super) struct SemanticsV2ImmutableDurableSequenceOutcomeIndex<'a> {
    database_id: [u8; 16],
    timeline_id: [u8; 16],
    recovery_prefix: u64,
    index_generation: u64,
    index_root: [u8; 32],
    complete_next_commit_sequence: u64,
    durable_next_commit_sequence: u64,
    published_next_commit_sequence: u64,
    outcomes: &'a [SemanticsV2DurableSequenceOutcome<'a>],
}

/// Opaque active checkpoint ownership for the recovered outcome-index generation.  The proof
/// borrows this non-`Copy` guard, so a later phase cannot retire, replace, or forge retention.
#[allow(dead_code)]
pub(super) struct SemanticsV2PinnedSequenceOutcomeIndexGeneration {
    database_id: [u8; 16],
    cluster_id: [u8; 16],
    timeline_id: [u8; 16],
    format_epoch: u64,
    leader_epoch: u64,
    index_generation: u64,
    index_root: [u8; 32],
    retained_through_commit_sequence: u64,
}

/// One recovered outcome.  `lookup_transition_txn_id` is the immutable map key while the stored
/// transition id remains independently bound, so a corrupt key/value pairing cannot pose as a
/// durable reference.  It intentionally has no S5 row/table/column/disposition fields.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SemanticsV2DurableSequenceOutcome<'a> {
    lookup_transition_txn_id: u64,
    transition_txn_id: u64,
    database_id: [u8; 16],
    timeline_id: [u8; 16],
    applied_commit_sequence: u64,
    sequence_oid: u32,
    parent_txn_id: u64,
    parent_autocommit: bool,
    parent_request_digest: gpu_db_wal::CanonicalDigest,
    statement_ordinal: u32,
    expression_ordinal: u32,
    returned_value: i64,
    input_digest: gpu_db_wal::CanonicalDigest,
    source_name: &'a str,
    operation: crate::BinarySequenceValueOperation,
}

/// Validate the full checkpoint-pinned index before walking any S5 reference.  This makes a
/// missing, duplicate, pruned, stale, cross-lineage, or unauthenticated index a durability error
/// even when the transaction happens to have no sequence effects.
pub(super) fn validate(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    proof: &SemanticsV2DurableSequenceOutcomeIndexProof<'_>,
) -> Result<(), crate::EngineError> {
    validate_complete_index(identity, proof)?;
    for effect in &graph.sequence_effects {
        validate_effect(identity, graph, proof, effect)?;
    }
    Ok(())
}

fn validate_complete_index(
    identity: SemanticsV2BoundIdentity,
    proof: &SemanticsV2DurableSequenceOutcomeIndexProof<'_>,
) -> Result<(), crate::EngineError> {
    let snapshot = proof.snapshot;
    let pin = proof.checkpoint_pin;
    require(
        snapshot.database_id == identity.database_id
            && snapshot.timeline_id == identity.timeline_id
            && snapshot.recovery_prefix >= identity.commit_sequence
            && snapshot.index_generation != 0
            && snapshot.index_generation != u64::MAX
            && snapshot.index_root != [0; 32]
            && snapshot.complete_next_commit_sequence > identity.commit_sequence
            && snapshot.durable_next_commit_sequence > identity.commit_sequence
            && snapshot.published_next_commit_sequence > identity.commit_sequence
            && pin.database_id == identity.database_id
            && pin.cluster_id == identity.cluster_id
            && pin.timeline_id == identity.timeline_id
            && pin.format_epoch == identity.format_epoch
            && pin.leader_epoch == identity.leader_epoch
            && pin.leader_epoch != 0
            && pin.index_generation == snapshot.index_generation
            && pin.index_root == snapshot.index_root
            && pin.retained_through_commit_sequence != 0
            && immutable_index_root(snapshot) == snapshot.index_root,
        "durable sequence outcome index lacks one authenticated pinned lineage/frontier",
    )?;
    for (ordinal, outcome) in snapshot.outcomes.iter().enumerate() {
        validate_outcome_shape(snapshot, outcome)?;
        for earlier in &snapshot.outcomes[..ordinal] {
            require(
                earlier.lookup_transition_txn_id != outcome.lookup_transition_txn_id,
                "durable sequence outcome index repeats a transition lookup identity",
            )?;
        }
    }
    Ok(())
}

fn validate_outcome_shape(
    snapshot: &SemanticsV2ImmutableDurableSequenceOutcomeIndex<'_>,
    outcome: &SemanticsV2DurableSequenceOutcome<'_>,
) -> Result<(), crate::EngineError> {
    require(
        outcome.lookup_transition_txn_id != 0
            && outcome.lookup_transition_txn_id != u64::MAX
            && outcome.transition_txn_id != 0
            && outcome.transition_txn_id != u64::MAX
            && outcome.lookup_transition_txn_id == outcome.transition_txn_id
            && outcome.database_id == snapshot.database_id
            && outcome.timeline_id == snapshot.timeline_id
            && outcome.applied_commit_sequence != 0
            && outcome.applied_commit_sequence != u64::MAX
            && outcome.applied_commit_sequence <= snapshot.recovery_prefix
            && outcome.applied_commit_sequence < snapshot.complete_next_commit_sequence
            && outcome.applied_commit_sequence < snapshot.durable_next_commit_sequence
            && outcome.applied_commit_sequence < snapshot.published_next_commit_sequence
            && outcome.sequence_oid != 0
            && outcome.parent_txn_id != 0
            && outcome.parent_txn_id != u64::MAX
            && outcome.parent_request_digest != [0; 32]
            && outcome.input_digest != [0; 32]
            && !outcome.source_name.is_empty(),
        "complete durable sequence outcome index has an invalid immutable row",
    )
}

fn validate_effect(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    proof: &SemanticsV2DurableSequenceOutcomeIndexProof<'_>,
    effect: &super::graph::RetainedSequenceEffect,
) -> Result<(), crate::EngineError> {
    let record = one_record_for_statement(graph, effect.statement_ordinal)?;
    let statement = one_statement_for_ordinal(graph, effect.statement_ordinal)?;
    let source = one_sequence_source(record, effect.effect_ordinal)?;
    let binding = one_sequence_binding(record, effect.effect_ordinal)?;
    let parent = record
        .sequence_parent()
        .ok_or_else(|| sequence_error("S5 published sequence effect has no retained S2 parent"))?;
    let disposition = graph
        .dispositions
        .get(usize::try_from(effect.disposition_ref).map_err(|_| {
            sequence_error("S5 sequence disposition reference exceeds host addressability")
        })?)
        .ok_or_else(|| sequence_error("S5 sequence disposition is absent"))?;
    let resolution = graph
        .resolutions
        .get(
            usize::try_from(effect.statement_ordinal)
                .map_err(|_| sequence_error("S5 statement ordinal exceeds host addressability"))?,
        )
        .ok_or_else(|| sequence_error("S5 statement has no retained S7 resolution"))?;
    let DecodedSequenceEffectKindFacts::Published {
        transition_txn_id,
        input_digest,
        returned_value,
    } = source.kind
    else {
        return Err(sequence_error(
            "S5 durable-sequence proof encountered a non-published S2 source",
        ));
    };
    let expected_overwritten = disposition.disposition != 1;
    require(
        effect.reference.transition_txn_id == transition_txn_id
            && effect.reference.parent_txn_id == identity.stable_transaction_id
            && effect.reference.parent_txn_id == parent.txn_id
            && effect.reference.statement_ordinal == effect.statement_ordinal
            && effect.reference.statement_ordinal == parent.statement_ordinal.as_u32()
            && effect.reference.expression_ordinal == source.request.absolute_expression_ordinal
            && effect.reference.sequence_oid == source.request.sequence_oid
            && effect.reference.returned_value == returned_value
            && effect.reference.input_digest == input_digest
            && effect.reference.default_expression
            && effect.reference.table_oid == source.request.target_table_oid
            && effect.reference.column_id == source.request.column_id
            && effect.reference.row_id == disposition.stable_row_id
            && effect.reference.final_value_overwritten == expected_overwritten
            && disposition.statement_ordinal == effect.statement_ordinal
            && disposition.source_row_ordinal == source.request.row_ordinal
            && resolution.table_ref == disposition.table_ref
            && binding.request == source.request
            && parent.txn_id == identity.stable_transaction_id
            && parent.autocommit == identity.autocommit
            && parent.request_digest == statement.request_digest
            && parent.statement_ordinal.as_u32() == effect.statement_ordinal,
        "S5 published sequence effect does not retain its exact S1/S2/S4/S7 identity",
    )?;

    let outcome = lookup_outcome(proof.snapshot, effect.reference.transition_txn_id)?;
    let recomputed_input = matches!(
        outcome.operation,
        crate::BinarySequenceValueOperation::Default
    )
    .then(|| {
        recompute_default_input_digest(
            parent.txn_id,
            parent.autocommit,
            parent.statement_ordinal.as_u32(),
            source.request.absolute_expression_ordinal,
            parent.request_digest,
            outcome.source_name,
        )
    });
    require(
        outcome.lookup_transition_txn_id == effect.reference.transition_txn_id
            && outcome.transition_txn_id == effect.reference.transition_txn_id
            && outcome.database_id == identity.database_id
            && outcome.timeline_id == identity.timeline_id
            && outcome.applied_commit_sequence < identity.commit_sequence
            && outcome.applied_commit_sequence < proof.snapshot.complete_next_commit_sequence
            && outcome.applied_commit_sequence < proof.snapshot.durable_next_commit_sequence
            && outcome.applied_commit_sequence < proof.snapshot.published_next_commit_sequence
            && outcome.applied_commit_sequence
                <= proof.checkpoint_pin.retained_through_commit_sequence
            && outcome.sequence_oid == effect.reference.sequence_oid
            && outcome.parent_txn_id == parent.txn_id
            && outcome.parent_autocommit == parent.autocommit
            && outcome.parent_request_digest == parent.request_digest
            && outcome.statement_ordinal == parent.statement_ordinal.as_u32()
            && outcome.expression_ordinal == source.request.absolute_expression_ordinal
            && outcome.returned_value == effect.reference.returned_value
            && outcome.input_digest == effect.reference.input_digest
            && outcome.source_name == binding.source_name
            && matches!(
                outcome.operation,
                crate::BinarySequenceValueOperation::Default
            )
            && recomputed_input == Some(outcome.input_digest),
        "durable sequence outcome is missing, forward, unretained, cross-lineage, or mismatched",
    )
}

fn one_record_for_statement(
    graph: &ReservedSemanticsV2Graph,
    statement_ordinal: u32,
) -> Result<&crate::typed_insert_batch::DecodedTypedInsertRecord, crate::EngineError> {
    let mut matches = graph
        .records
        .iter()
        .filter(|record| record.facts().statement_ordinal.as_u32() == statement_ordinal);
    let record = matches
        .next()
        .ok_or_else(|| sequence_error("S5 sequence effect statement has no retained S2 record"))?;
    require(
        matches.next().is_none(),
        "S5 sequence effect statement has duplicate retained S2 records",
    )?;
    Ok(record)
}

fn one_statement_for_ordinal(
    graph: &ReservedSemanticsV2Graph,
    statement_ordinal: u32,
) -> Result<&super::graph::RetainedStatement, crate::EngineError> {
    let mut matches = graph
        .statements
        .iter()
        .filter(|statement| statement.statement_ordinal == statement_ordinal);
    let statement = matches
        .next()
        .ok_or_else(|| sequence_error("S5 sequence effect statement has no retained S1 row"))?;
    require(
        matches.next().is_none(),
        "S5 sequence effect statement has duplicate retained S1 rows",
    )?;
    Ok(statement)
}

fn one_sequence_source(
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    effect_ordinal: u32,
) -> Result<crate::typed_insert_batch::DecodedSequenceEffectFacts, crate::EngineError> {
    let mut matches = record
        .sequence_effects()
        .filter(|effect| effect.request.effect_ordinal == effect_ordinal);
    let source = matches
        .next()
        .ok_or_else(|| sequence_error("S5 effect has no retained S2 sequence source"))?;
    require(
        matches.next().is_none(),
        "S5 effect has duplicate retained S2 sequence sources",
    )?;
    Ok(source)
}

fn one_sequence_binding<'a>(
    record: &'a crate::typed_insert_batch::DecodedTypedInsertRecord,
    effect_ordinal: u32,
) -> Result<crate::typed_insert_batch::DecodedSequenceBindingFacts<'a>, crate::EngineError> {
    let mut matches = record
        .sequence_bindings()
        .filter(|binding| binding.effect_ordinal == effect_ordinal);
    let binding = matches
        .next()
        .ok_or_else(|| sequence_error("S5 effect has no retained S2 sequence binding"))?;
    require(
        matches.next().is_none(),
        "S5 effect has duplicate retained S2 sequence bindings",
    )?;
    Ok(binding)
}

fn lookup_outcome<'a>(
    snapshot: &'a SemanticsV2ImmutableDurableSequenceOutcomeIndex<'a>,
    transition_txn_id: u64,
) -> Result<&'a SemanticsV2DurableSequenceOutcome<'a>, crate::EngineError> {
    let mut matched = None;
    for outcome in snapshot.outcomes {
        if outcome.lookup_transition_txn_id == transition_txn_id
            && matched.replace(outcome).is_some()
        {
            return Err(sequence_error(
                "durable sequence outcome index has duplicate matching transitions",
            ));
        }
    }
    matched
        .ok_or_else(|| sequence_error("durable sequence outcome is absent from the Engine index"))
}

/// Recompute the exact codec-5 `Default` input digest without staging its encoded body.
///
/// The writer-side digest helper intentionally builds a durable record body for framing.  This
/// validator owns only a borrowed immutable index, so it streams the same canonical framing
/// directly into SHA-256 and never allocates per S5 effect.
fn recompute_default_input_digest(
    parent_txn_id: u64,
    parent_autocommit: bool,
    statement_ordinal: u32,
    expression_ordinal: u32,
    parent_request_digest: gpu_db_wal::CanonicalDigest,
    source_name: &str,
) -> gpu_db_wal::CanonicalDigest {
    const REQUEST_DIGEST_DOMAIN: &[u8] = b"gpu-db/adr014/request/v1";
    const SEQUENCE_VALUE_INPUT_DOMAIN: &[u8] = b"GPUDBSEQVALUEINPUT1";
    const DEFAULT_OPERATION_TAG: u8 = 2;

    let parent_txn_id = parent_txn_id.to_le_bytes();
    let parent_autocommit = [u8::from(parent_autocommit)];
    let statement_ordinal = statement_ordinal.to_le_bytes();
    let expression_ordinal = expression_ordinal.to_le_bytes();
    let source_name_length = u64::try_from(source_name.len())
        .expect("durable sequence source-name length fits u64")
        .to_le_bytes();
    let input_length = [
        SEQUENCE_VALUE_INPUT_DOMAIN.len(),
        parent_txn_id.len(),
        parent_autocommit.len(),
        statement_ordinal.len(),
        expression_ordinal.len(),
        parent_request_digest.len(),
        source_name_length.len(),
        source_name.len(),
        1,
    ]
    .into_iter()
    .try_fold(0usize, |total, length| total.checked_add(length))
    .expect("durable sequence input digest byte length fits usize");

    let mut digest = Sha256::new();
    digest.update(
        u64::try_from(REQUEST_DIGEST_DOMAIN.len())
            .expect("request digest domain length fits u64")
            .to_le_bytes(),
    );
    digest.update(REQUEST_DIGEST_DOMAIN);
    digest.update(
        u64::try_from(input_length)
            .expect("durable sequence input digest byte length fits u64")
            .to_le_bytes(),
    );
    digest.update(SEQUENCE_VALUE_INPUT_DOMAIN);
    digest.update(parent_txn_id);
    digest.update(parent_autocommit);
    digest.update(statement_ordinal);
    digest.update(expression_ordinal);
    digest.update(parent_request_digest);
    digest.update(source_name_length);
    digest.update(source_name.as_bytes());
    digest.update([DEFAULT_OPERATION_TAG]);
    digest.finalize().into()
}

fn immutable_index_root(
    snapshot: &SemanticsV2ImmutableDurableSequenceOutcomeIndex<'_>,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"gpu-db/write001/durable-sequence-outcomes/v1");
    digest.update(snapshot.database_id);
    digest.update(snapshot.timeline_id);
    digest.update(snapshot.recovery_prefix.to_le_bytes());
    digest.update(snapshot.index_generation.to_le_bytes());
    digest.update(snapshot.complete_next_commit_sequence.to_le_bytes());
    digest.update(snapshot.durable_next_commit_sequence.to_le_bytes());
    digest.update(snapshot.published_next_commit_sequence.to_le_bytes());
    digest.update(
        u64::try_from(snapshot.outcomes.len())
            .expect("durable sequence outcome count fits u64")
            .to_le_bytes(),
    );
    for outcome in snapshot.outcomes {
        digest.update(outcome.lookup_transition_txn_id.to_le_bytes());
        digest.update(outcome.transition_txn_id.to_le_bytes());
        digest.update(outcome.database_id);
        digest.update(outcome.timeline_id);
        digest.update(outcome.applied_commit_sequence.to_le_bytes());
        digest.update(outcome.sequence_oid.to_le_bytes());
        digest.update(outcome.parent_txn_id.to_le_bytes());
        digest.update([u8::from(outcome.parent_autocommit)]);
        digest.update(outcome.parent_request_digest);
        digest.update(outcome.statement_ordinal.to_le_bytes());
        digest.update(outcome.expression_ordinal.to_le_bytes());
        digest.update(outcome.returned_value.to_le_bytes());
        digest.update(outcome.input_digest);
        digest.update(
            u64::try_from(outcome.source_name.len())
                .expect("durable sequence source-name length fits u64")
                .to_le_bytes(),
        );
        digest.update(outcome.source_name.as_bytes());
        match outcome.operation {
            crate::BinarySequenceValueOperation::NextVal => digest.update([1]),
            crate::BinarySequenceValueOperation::Default => digest.update([2]),
            crate::BinarySequenceValueOperation::SetVal { is_called } => {
                digest.update([3, u8::from(is_called)]);
            }
        }
    }
    digest.finalize().into()
}

fn sequence_error(message: &str) -> crate::EngineError {
    crate::EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v2 durable sequence: {message}"
    ))
}

fn require(condition: bool, message: &str) -> Result<(), crate::EngineError> {
    if condition {
        Ok(())
    } else {
        Err(sequence_error(message))
    }
}

#[cfg(test)]
#[test]
fn streamed_default_input_digest_matches_the_durable_codec() {
    let parent_request_digest = [0xA5; 32];
    let expected = crate::sequence_value_input_digest(crate::SequenceValueInput {
        parent_txn_id: 17,
        parent_autocommit: true,
        statement_ordinal: 3,
        expression_ordinal: 11,
        parent_request_digest,
        source_name: "public.orders_id_seq",
        operation: crate::BinarySequenceValueOperation::Default,
        set_value: None,
    });

    assert_eq!(
        recompute_default_input_digest(
            17,
            true,
            3,
            11,
            parent_request_digest,
            "public.orders_id_seq",
        ),
        expected,
    );
}

/// Exact test fixture row for an already-published sequence transition.  The callback adapter
/// below owns all index/pin backing, so this narrow input cannot escape as an authority.
#[cfg(test)]
#[derive(Clone, Copy)]
pub(in crate::typed_insert_aggregate::semantics_v2) struct SequenceOutcomeSpecForTest<'a> {
    pub(in crate::typed_insert_aggregate::semantics_v2) transition_txn_id: u64,
    pub(in crate::typed_insert_aggregate::semantics_v2) applied_commit_sequence: u64,
    pub(in crate::typed_insert_aggregate::semantics_v2) sequence_oid: u32,
    pub(in crate::typed_insert_aggregate::semantics_v2) parent_txn_id: u64,
    pub(in crate::typed_insert_aggregate::semantics_v2) parent_autocommit: bool,
    pub(in crate::typed_insert_aggregate::semantics_v2) parent_request_digest:
        gpu_db_wal::CanonicalDigest,
    pub(in crate::typed_insert_aggregate::semantics_v2) statement_ordinal: u32,
    pub(in crate::typed_insert_aggregate::semantics_v2) expression_ordinal: u32,
    pub(in crate::typed_insert_aggregate::semantics_v2) returned_value: i64,
    pub(in crate::typed_insert_aggregate::semantics_v2) input_digest: gpu_db_wal::CanonicalDigest,
    pub(in crate::typed_insert_aggregate::semantics_v2) source_name: &'a str,
    pub(in crate::typed_insert_aggregate::semantics_v2) operation:
        crate::BinarySequenceValueOperation,
}

/// Closed hostile proof variants used only by focused durable-index sabotage tests.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::typed_insert_aggregate::semantics_v2) enum SequenceOutcomeProofSabotageForTest {
    Missing,
    Duplicate,
    IndexRoot,
    Incomplete,
    Nondurable,
    Unpublished,
    RecoveryPrefix,
    CheckpointLineage,
    CheckpointRetention,
    ForwardCommit,
    CrossLineage,
    TransitionId,
    HiddenLookupTransitionMismatch,
    SequenceOid,
    ParentTxn,
    ParentAutocommit,
    ParentRequestDigest,
    StatementOrdinal,
    ExpressionOrdinal,
    ReturnedValue,
    InputDigest,
    SourceName,
    NonDefault,
}

/// Build an immutable Engine-index-shaped proof only for the callback extent.  Production has no
/// construction seam; the future bridge must borrow the actual Engine index once instead.
#[cfg(test)]
pub(super) fn with_sequence_outcome_proof_for_test<T>(
    identity: SemanticsV2BoundIdentity,
    specs: &[SequenceOutcomeSpecForTest<'_>],
    sabotage: Option<SequenceOutcomeProofSabotageForTest>,
    operation: impl FnOnce(SemanticsV2DurableSequenceOutcomeIndexProof<'_>) -> T,
) -> T {
    let mut outcomes: Vec<_> = specs
        .iter()
        .map(|spec| SemanticsV2DurableSequenceOutcome {
            lookup_transition_txn_id: spec.transition_txn_id,
            transition_txn_id: spec.transition_txn_id,
            database_id: identity.database_id,
            timeline_id: identity.timeline_id,
            applied_commit_sequence: spec.applied_commit_sequence,
            sequence_oid: spec.sequence_oid,
            parent_txn_id: spec.parent_txn_id,
            parent_autocommit: spec.parent_autocommit,
            parent_request_digest: spec.parent_request_digest,
            statement_ordinal: spec.statement_ordinal,
            expression_ordinal: spec.expression_ordinal,
            returned_value: spec.returned_value,
            input_digest: spec.input_digest,
            source_name: spec.source_name,
            operation: spec.operation,
        })
        .collect();
    let mut pin = SemanticsV2PinnedSequenceOutcomeIndexGeneration {
        database_id: identity.database_id,
        cluster_id: identity.cluster_id,
        timeline_id: identity.timeline_id,
        format_epoch: identity.format_epoch,
        leader_epoch: identity.leader_epoch,
        index_generation: 1,
        index_root: [0; 32],
        retained_through_commit_sequence: identity.commit_sequence,
    };
    if let Some(sabotage) = sabotage {
        apply_sabotage(&mut outcomes, &mut pin, identity.commit_sequence, sabotage);
    }
    let recovery_prefix = if sabotage == Some(SequenceOutcomeProofSabotageForTest::RecoveryPrefix) {
        identity.commit_sequence.saturating_sub(1)
    } else {
        identity.commit_sequence
    };
    let complete_next_commit_sequence =
        if sabotage == Some(SequenceOutcomeProofSabotageForTest::Incomplete) {
            identity.commit_sequence
        } else {
            identity.commit_sequence + 1
        };
    let durable_next_commit_sequence =
        if sabotage == Some(SequenceOutcomeProofSabotageForTest::Nondurable) {
            identity.commit_sequence
        } else {
            identity.commit_sequence + 1
        };
    let published_next_commit_sequence =
        if sabotage == Some(SequenceOutcomeProofSabotageForTest::Unpublished) {
            identity.commit_sequence
        } else {
            identity.commit_sequence + 1
        };
    let mut snapshot = SemanticsV2ImmutableDurableSequenceOutcomeIndex {
        database_id: identity.database_id,
        timeline_id: identity.timeline_id,
        recovery_prefix,
        index_generation: 1,
        index_root: [0; 32],
        complete_next_commit_sequence,
        durable_next_commit_sequence,
        published_next_commit_sequence,
        outcomes: &outcomes,
    };
    snapshot.index_root = immutable_index_root(&snapshot);
    pin.index_root = snapshot.index_root;
    if sabotage == Some(SequenceOutcomeProofSabotageForTest::IndexRoot) {
        pin.index_root[0] ^= 1;
    }
    operation(SemanticsV2DurableSequenceOutcomeIndexProof {
        snapshot: &snapshot,
        checkpoint_pin: &pin,
    })
}

#[cfg(test)]
fn apply_sabotage(
    outcomes: &mut Vec<SemanticsV2DurableSequenceOutcome<'_>>,
    pin: &mut SemanticsV2PinnedSequenceOutcomeIndexGeneration,
    parent_commit_sequence: u64,
    sabotage: SequenceOutcomeProofSabotageForTest,
) {
    match sabotage {
        SequenceOutcomeProofSabotageForTest::Missing => outcomes.clear(),
        SequenceOutcomeProofSabotageForTest::Duplicate => {
            if let Some(outcome) = outcomes.first().copied() {
                outcomes.push(outcome);
            }
        }
        SequenceOutcomeProofSabotageForTest::IndexRoot => {}
        SequenceOutcomeProofSabotageForTest::Incomplete
        | SequenceOutcomeProofSabotageForTest::Nondurable
        | SequenceOutcomeProofSabotageForTest::Unpublished
        | SequenceOutcomeProofSabotageForTest::RecoveryPrefix => {}
        SequenceOutcomeProofSabotageForTest::CheckpointLineage => {
            pin.timeline_id[0] ^= 1;
        }
        SequenceOutcomeProofSabotageForTest::CheckpointRetention => {
            pin.retained_through_commit_sequence = 1;
        }
        SequenceOutcomeProofSabotageForTest::ForwardCommit => {
            if let Some(outcome) = outcomes.first_mut() {
                outcome.applied_commit_sequence = parent_commit_sequence;
            }
        }
        SequenceOutcomeProofSabotageForTest::CrossLineage => {
            if let Some(outcome) = outcomes.first_mut() {
                outcome.timeline_id[0] ^= 1;
            }
        }
        SequenceOutcomeProofSabotageForTest::TransitionId => {
            if let Some(outcome) = outcomes.first_mut() {
                outcome.transition_txn_id += 1;
            }
        }
        SequenceOutcomeProofSabotageForTest::HiddenLookupTransitionMismatch => {
            if let Some(outcome) = outcomes.first().copied() {
                let mut hidden = outcome;
                hidden.lookup_transition_txn_id =
                    if outcome.transition_txn_id == 1 { 2 } else { 1 };
                outcomes.push(hidden);
            }
        }
        SequenceOutcomeProofSabotageForTest::SequenceOid => {
            if let Some(outcome) = outcomes.first_mut() {
                outcome.sequence_oid += 1;
            }
        }
        SequenceOutcomeProofSabotageForTest::ParentTxn => {
            if let Some(outcome) = outcomes.first_mut() {
                outcome.parent_txn_id += 1;
            }
        }
        SequenceOutcomeProofSabotageForTest::ParentAutocommit => {
            if let Some(outcome) = outcomes.first_mut() {
                outcome.parent_autocommit = !outcome.parent_autocommit;
            }
        }
        SequenceOutcomeProofSabotageForTest::ParentRequestDigest => {
            if let Some(outcome) = outcomes.first_mut() {
                outcome.parent_request_digest[0] ^= 1;
            }
        }
        SequenceOutcomeProofSabotageForTest::StatementOrdinal => {
            if let Some(outcome) = outcomes.first_mut() {
                outcome.statement_ordinal += 1;
            }
        }
        SequenceOutcomeProofSabotageForTest::ExpressionOrdinal => {
            if let Some(outcome) = outcomes.first_mut() {
                outcome.expression_ordinal += 1;
            }
        }
        SequenceOutcomeProofSabotageForTest::ReturnedValue => {
            if let Some(outcome) = outcomes.first_mut() {
                outcome.returned_value += 1;
            }
        }
        SequenceOutcomeProofSabotageForTest::InputDigest => {
            if let Some(outcome) = outcomes.first_mut() {
                outcome.input_digest[0] ^= 1;
            }
        }
        SequenceOutcomeProofSabotageForTest::SourceName => {
            if let Some(outcome) = outcomes.first_mut() {
                outcome.source_name = "corrupt_sequence_source";
            }
        }
        SequenceOutcomeProofSabotageForTest::NonDefault => {
            if let Some(outcome) = outcomes.first_mut() {
                outcome.operation = crate::BinarySequenceValueOperation::SetVal { is_called: true };
            }
        }
    }
}
