//! Stateful sequence evidence coverage for the otherwise inert codec.

use super::*;

pub(super) fn private_chain_batch() -> TypedInsertBatch {
    private_chain_batch_for_test(InsertStatementOrdinal::from_u32(4), 81)
}

/// Exact private-chain fixture reused by aggregate S5 tests. It remains compiled only in test
/// builds and returns a batch that must still pass the one canonical encoder.
pub(crate) fn private_chain_batch_for_test(
    parent_statement: InsertStatementOrdinal,
    parent_txn_id: TxnId,
) -> TypedInsertBatch {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(45, "CREATE TABLE codec_private (id serial, payload int4)")
        .expect("private sequence fixture creates");
    let crate::Command::Insert(insert) = crate::parse_command(
        "INSERT INTO codec_private (id, payload) VALUES (DEFAULT, 7), (DEFAULT, 8)",
    )
    .expect("private sequence fixture parses") else {
        panic!("fixture is INSERT");
    };
    let catalog = engine.catalog_snapshot();
    let prepared = prepare_typed_insert_semantics_at(
        &insert,
        &catalog,
        catalog.commit_seq,
        None,
        parent_statement,
    )
    .expect("private sequence semantic preparation succeeds")
    .expect("private sequence fixture prepares");
    let parent = sequence_defaults::effects::SequenceDefaultParentContext::for_test(
        parent_txn_id,
        false,
        prepared.typed_statement_digest(),
        parent_statement,
        11,
    );
    let parent_view = CanonicalSequenceParentView {
        txn_id: parent_txn_id,
        autocommit: false,
        request_digest: prepared.typed_statement_digest(),
        statement_ordinal: parent_statement,
        expression_ordinal_base: 11,
    };
    let owner = sequence::PrivateOwner {
        kind: 1,
        statement_ordinal: parent_statement
            .as_u32()
            .checked_sub(1)
            .expect("private-chain fixture parent must follow its owner"),
        statement_digest: [0x93; 32],
        creator_catalog_column_ordinal: None,
    };
    let mut prior = (40_i64, true);
    let mut predecessor = sequence::PrivatePredecessor::Lifecycle(owner);
    let mut predecessor_tag = 1_u8;
    let mut predecessor_digest = owner.statement_digest;
    let mut bindings = Vec::new();
    for request in prepared.sequence_requests().iter().cloned() {
        let value = prior.0 + 1;
        let temporary = sequence_defaults::SequenceDefaultBinding::published(
            request.clone(),
            parent.clone(),
            value,
        );
        let request_view = temporary.canonical_view().request;
        let absolute = parent_view
            .expression_ordinal_base
            .checked_add(request_view.expression_ordinal)
            .expect("fixture expression ordinal fits");
        let input_digest = sequence::sequence_input_digest(parent_view, request_view, absolute);
        let descriptor_digest = crate::sequence_descriptor_digest(
            request_view.sequence_oid,
            request_view.sequence_effective_name,
        );
        let child_digest = sequence::private_child_digest(
            parent_view,
            request_view,
            absolute,
            input_digest,
            descriptor_digest,
            2,
            prior,
            predecessor,
        )
        .expect("private child digest computes");
        let next = (value, true);
        let outcome_digest = sequence::private_outcome_digest(child_digest, owner, value, next)
            .expect("private outcome digest computes");
        let planning = sequence_defaults::effects::PrivateSequencePlanningEvidence::exact_for_test(
            2,
            owner.kind,
            owner.statement_ordinal,
            owner.statement_digest,
            owner.creator_catalog_column_ordinal,
            predecessor_tag,
            predecessor_digest,
            input_digest,
            descriptor_digest,
            child_digest,
            outcome_digest,
        );
        bindings.push(
            sequence_defaults::SequenceDefaultBinding::private_exact_for_test(
                request,
                parent.clone(),
                value,
                prior,
                next,
                planning,
            ),
        );
        prior = next;
        predecessor = sequence::PrivatePredecessor::Outcome(outcome_digest);
        predecessor_tag = 2;
        predecessor_digest = outcome_digest;
    }
    prepared
        .seal(sequence_defaults::SequenceDefaultBindings::from_bindings(
            parent, bindings,
        ))
        .expect("private sequence fixture seals")
}

#[test]
fn canonical_codec_round_trips_private_multi_effect_predecessor_chain() {
    let batch = private_chain_batch();
    let bytes = encode(&batch).expect("private sequence record encodes");
    assert_eq!(
        decode(&bytes)
            .expect("private sequence record decodes")
            .reencode(),
        bytes
    );
}

#[test]
fn decoded_record_exposes_private_sequence_facts_without_exposing_state() {
    let decoded = decode(&encode(&private_chain_batch()).expect("private record encodes"))
        .expect("private record decodes");
    let parent = decoded
        .sequence_parent()
        .expect("private effects retain their shared parent");
    assert_eq!(parent.txn_id, 81);
    assert!(!parent.autocommit);
    assert_eq!(
        parent.statement_ordinal,
        InsertStatementOrdinal::from_u32(4)
    );
    assert_eq!(parent.expression_ordinal_base, 11);

    let effects = decoded.sequence_effects().collect::<Vec<_>>();
    assert_eq!(effects.len(), 2);
    for (ordinal, effect) in effects.iter().enumerate() {
        assert_eq!(effect.request.effect_ordinal, ordinal as u32);
        assert_eq!(effect.request.row_ordinal, ordinal as u32);
        assert_eq!(effect.request.expression_ordinal, ordinal as u32);
        assert_eq!(
            effect.request.absolute_expression_ordinal,
            parent.expression_ordinal_base + ordinal as u32
        );
        assert_eq!(effect.resolved_value, 41 + ordinal as i64);
        let DecodedSequenceEffectKindFacts::Private { input_digest } = effect.kind else {
            panic!("private fixture must expose private scalar witnesses")
        };
        assert_eq!(
            input_digest,
            crate::sequence_value_input_digest(crate::SequenceValueInput {
                parent_txn_id: parent.txn_id,
                parent_autocommit: parent.autocommit,
                statement_ordinal: parent.statement_ordinal.as_u32(),
                expression_ordinal: effect.request.absolute_expression_ordinal,
                parent_request_digest: parent.request_digest,
                source_name: "codec_private_id_seq",
                operation: crate::BinarySequenceValueOperation::Default,
                set_value: None,
            })
        );
        assert_ne!(input_digest, [0; 32]);
    }
}
