use super::*;

const COUNTS: [u32; AGGREGATE_SECTION_COUNT] = [1, 1, 0, 1, 0, 1, 1, 0];

fn view<'a>(payloads: &'a [Vec<u8>; AGGREGATE_SECTION_COUNT]) -> TypedInsertAggregateView<'a> {
    TypedInsertAggregateView {
        flags: AGGREGATE_FLAG_AUTOCOMMIT,
        outer_flags: OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
        stable_transaction_id: 41,
        statement_count: 1,
        insert_statement_count: 1,
        original_inserted_row_count: 1,
        final_row_transition_count: 1,
        allocator_before: 99,
        allocator_high_water: 100,
        table_block_count: 1,
        sections: std::array::from_fn(|index| TypedInsertAggregateSectionView {
            entry_count: COUNTS[index],
            payload: &payloads[index],
        }),
    }
}

fn encoded_bodies() -> EncodedTypedInsertAggregateBodies {
    let payloads: [Vec<u8>; AGGREGATE_SECTION_COUNT] =
        std::array::from_fn(|index| vec![index as u8 + 1; index * 7 + 1]);
    let view = view(&payloads);
    encoded_bodies_for(&view)
}

fn encoded_bodies_for(view: &TypedInsertAggregateView<'_>) -> EncodedTypedInsertAggregateBodies {
    let layout = view.measure().unwrap();
    let roots = typed_insert_aggregate_status_roots(view, &layout).unwrap();
    let status = TypedInsertStatusV2 {
        database_id: [1; 16],
        timeline_id: [3; 16],
        txn_id: view.stable_transaction_id,
        request_digest: [4; 32],
        isolation: 1,
        flags: 0,
        retention_deadline: 0,
        statement_count: view.statement_count,
        response_artifact_count: 0,
        statement_outcome_root: roots.statement_outcome_root,
        response_root: roots.response_root,
        aggregate_root: roots.aggregate_root,
    };
    let reserved = reserve_typed_insert_aggregate_bodies(layout).unwrap();
    encode_typed_insert_aggregate_bodies(view, &status, reserved).unwrap()
}

fn physical() -> gpu_db_wal::CanonicalPhysicalRange {
    gpu_db_wal::CanonicalPhysicalRange {
        log_epoch: 11,
        lane_id: 0,
        segment_id: 17,
        first_frame_ordinal: 23,
    }
}

fn header(bodies: &EncodedTypedInsertAggregateBodies) -> gpu_db_wal::CanonicalPreApplyHeader {
    let measure = bodies.layout().measure;
    gpu_db_wal::CanonicalPreApplyHeader {
        identity: gpu_db_wal::CanonicalIdentity {
            database_id: bodies.status().database_id,
            cluster_id: [2; 16],
            timeline_id: bodies.status().timeline_id,
            format_epoch: 1,
        },
        leader_epoch: 11,
        commit_seq: 37,
        stable_transaction_id: measure.stable_transaction_id,
        request_digest: bodies.status().request_digest,
        isolation: gpu_db_wal::CanonicalIsolation::ReadCommitted,
        flags: measure.outer_flags,
        catalog_before_epoch: 5,
        catalog_after_epoch: 5,
        catalog_before_digest: [6; 32],
        catalog_after_digest: [6; 32],
        operation_count: bodies.layout().fragment_count,
        table_block_count: measure.table_block_count,
        allocator_high_water: measure.allocator_high_water,
    }
}

fn outcome(bodies: &EncodedTypedInsertAggregateBodies) -> gpu_db_wal::CanonicalOutcome {
    gpu_db_wal::CanonicalOutcome {
        kind: gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
        affected_rows: 1,
        sqlstate: None,
        constraint_id: 0,
        target_digest: bodies.aggregate_root(),
        returning_digest: bodies.status().response_root,
    }
}

#[test]
fn exact_outer_buffers_are_reserved_then_encoded_without_owned_fragment_copies() {
    let bodies = encoded_bodies();
    let expected = bodies.layout().wal;
    let header = header(&bodies);
    let outcome = outcome(&bodies);
    let reserved = reserve_typed_insert_canonical_envelope(
        bodies,
        physical(),
        header.clone(),
        outcome.clone(),
    )
    .unwrap();
    assert_eq!(
        reserved.packed_payload_len() as u64,
        expected.packed_record_bytes
    );
    assert_eq!(
        reserved.serialized_record_len() as u64,
        expected.serialized_record_bytes
    );

    let encoded = encode_reserved_typed_insert_canonical_envelope(reserved).unwrap();
    assert_eq!(encoded.encoding().footprint, expected);
    assert_eq!(encoded.physical(), physical());
    assert_eq!(encoded.header(), &header);
    assert_eq!(encoded.outcome(), &outcome);
    let outer_header_len =
        usize::try_from(expected.serialized_record_bytes - expected.packed_record_bytes).unwrap();
    assert_eq!(
        &encoded.serialized_record()[outer_header_len..],
        encoded.packed_payload()
    );
    assert_eq!(
        u64::from_le_bytes(encoded.serialized_record()[0..8].try_into().unwrap()),
        header.stable_transaction_id
    );
    assert_eq!(
        u64::from_le_bytes(encoded.serialized_record()[8..16].try_into().unwrap()),
        encoded.packed_payload().len() as u64
    );
    let decoded = gpu_db_wal::decode_canonical_record_payload(encoded.packed_payload())
        .unwrap()
        .expect("exact payload is canonical");
    assert_eq!(decoded.header, header);
    assert_eq!(decoded.outcome, outcome);
    assert_eq!(
        decoded.ordered_fragment_root,
        encoded.encoding().ordered_fragment_root
    );
    assert_eq!(decoded.final_digest, encoded.encoding().final_digest);
    assert_eq!(
        decoded.fragments.len(),
        encoded.bodies().layout().fragment_count as usize
    );
    assert!(decoded.fragments[..decoded.fragments.len() - 1]
        .iter()
        .all(|fragment| fragment.kind == gpu_db_wal::CanonicalFragmentKind::RowMutation));
    assert_eq!(
        decoded.fragments.last().unwrap().kind,
        gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus
    );
}

#[test]
fn every_outer_identity_dimension_fails_before_reservation_authority() {
    for sabotage in 0..11 {
        let bodies = encoded_bodies();
        let mut header = header(&bodies);
        let mut outcome = outcome(&bodies);
        match sabotage {
            0 => header.stable_transaction_id += 1,
            1 => header.identity.database_id[0] ^= 1,
            2 => header.identity.timeline_id[0] ^= 1,
            3 => header.request_digest[0] ^= 1,
            4 => header.isolation = gpu_db_wal::CanonicalIsolation::RepeatableRead,
            5 => header.flags ^= OUTER_CONTENT_CATALOG,
            6 => header.operation_count += 1,
            7 => header.table_block_count += 1,
            8 => header.allocator_high_water += 1,
            9 => outcome.target_digest[0] ^= 1,
            10 => outcome.affected_rows += 1,
            _ => unreachable!(),
        }
        assert!(
            reserve_typed_insert_canonical_envelope(bodies, physical(), header, outcome).is_err(),
            "accepted outer binding sabotage {sabotage}"
        );
    }
}

#[test]
fn autocommit_success_binds_the_single_statement_inserted_rows_not_statement_count() {
    let payloads: [Vec<u8>; AGGREGATE_SECTION_COUNT] =
        std::array::from_fn(|index| vec![index as u8 + 1; index * 7 + 1]);
    let mut view = view(&payloads);
    view.original_inserted_row_count = 3;
    view.final_row_transition_count = 3;
    view.allocator_high_water = 102;
    view.sections[3].entry_count = 3;
    let bodies = encoded_bodies_for(&view);
    let header = header(&bodies);

    let mut actual_inserted_rows = outcome(&bodies);
    actual_inserted_rows.affected_rows = 3;
    assert!(reserve_typed_insert_canonical_envelope(
        bodies,
        physical(),
        header.clone(),
        actual_inserted_rows,
    )
    .is_ok());

    let bodies = encoded_bodies_for(&view);
    let mut statement_count = outcome(&bodies);
    statement_count.affected_rows = u64::from(view.statement_count);
    assert!(
        reserve_typed_insert_canonical_envelope(bodies, physical(), header, statement_count)
            .is_err()
    );
}

#[test]
fn explicit_success_binds_the_final_transition_count_not_statement_or_insert_count() {
    let payloads: [Vec<u8>; AGGREGATE_SECTION_COUNT] =
        std::array::from_fn(|index| vec![index as u8 + 1; index * 7 + 1]);
    let mut view = view(&payloads);
    view.flags = AGGREGATE_FLAG_EXPLICIT;
    view.statement_count = 2;
    view.insert_statement_count = 2;
    view.original_inserted_row_count = 3;
    view.final_row_transition_count = 1;
    view.allocator_high_water = 102;
    view.sections[0].entry_count = 2;
    view.sections[1].entry_count = 2;
    view.sections[3].entry_count = 3;
    view.sections[5].entry_count = 2;
    let bodies = encoded_bodies_for(&view);
    let header = header(&bodies);

    let mut final_transition_count = outcome(&bodies);
    final_transition_count.affected_rows = 1;
    assert!(reserve_typed_insert_canonical_envelope(
        bodies,
        physical(),
        header.clone(),
        final_transition_count,
    )
    .is_ok());

    let bodies = encoded_bodies_for(&view);
    let mut statement_count = outcome(&bodies);
    statement_count.affected_rows = u64::from(view.statement_count);
    assert!(
        reserve_typed_insert_canonical_envelope(bodies, physical(), header, statement_count)
            .is_err()
    );
}

#[test]
fn no_op_and_abort_keep_their_zero_affected_row_contracts() {
    let payloads: [Vec<u8>; AGGREGATE_SECTION_COUNT] =
        std::array::from_fn(|index| vec![index as u8 + 1; index * 7 + 1]);
    let mut no_op_view = view(&payloads);
    no_op_view.original_inserted_row_count = 3;
    no_op_view.final_row_transition_count = 0;
    no_op_view.allocator_high_water = 102;
    no_op_view.sections[3].entry_count = 3;
    let bodies = encoded_bodies_for(&no_op_view);
    let no_op_header = header(&bodies);
    let mut no_op = outcome(&bodies);
    no_op.kind = gpu_db_wal::CanonicalOutcomeKind::CommitNoOp;
    no_op.affected_rows = 0;
    assert!(
        reserve_typed_insert_canonical_envelope(bodies, physical(), no_op_header, no_op,).is_ok()
    );

    let bodies = encoded_bodies();
    let header = header(&bodies);
    let mut aborted = outcome(&bodies);
    aborted.kind = gpu_db_wal::CanonicalOutcomeKind::AbortError;
    aborted.affected_rows = 0;
    aborted.sqlstate = Some(*b"23505");
    assert!(reserve_typed_insert_canonical_envelope(bodies, physical(), header, aborted).is_ok());
}

#[test]
fn outer_envelope_source_is_borrowed_and_collection_free() {
    let source = include_str!("envelope.rs");
    assert!(source.contains("canonical_fragment_refs"));
    assert!(source.contains("encode_canonical_record_exact_from_borrowed"));
    for forbidden in ["CanonicalFragment {", "Vec<", "Vec::", "collect("] {
        assert!(
            !source.contains(forbidden),
            "outer exact owner reintroduced {forbidden}"
        );
    }
}
