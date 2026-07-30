//! Local S5/S6 sabotage coverage for the inert source-materialization seam.

use super::*;

fn s5_entry_ranges(bytes: &[u8]) -> Vec<Range<usize>> {
    let mut cursor = 0_usize;
    let mut ranges = Vec::new();
    while cursor < bytes.len() {
        let start = cursor;
        let body_len =
            u32::from_le_bytes(bytes[cursor + 16..cursor + 20].try_into().unwrap()) as usize;
        cursor += 52 + body_len;
        ranges.push(start..cursor);
    }
    ranges
}

fn suppress_s4_row(payload: &mut [u8], row: usize) {
    let base = row * ROW_DISPOSITION_ENTRY_BYTES as usize;
    payload[base + 16] = DISPOSITION_SUPPRESSED_AT_STATEMENT;
    payload[base + 20..base + 24].copy_from_slice(&u32::MAX.to_le_bytes());
    payload[base + 24..base + 28].copy_from_slice(&u32::MAX.to_le_bytes());
}

fn preserve_s4_row_as_survivor(payload: &mut [u8], row: usize) {
    let base = row * ROW_DISPOSITION_ENTRY_BYTES as usize;
    payload[base + 16] = DISPOSITION_SURVIVES;
    payload[base + 20..base + 24].copy_from_slice(&(row as u32).to_le_bytes());
    payload[base + 24..base + 28].copy_from_slice(&(row as u32).to_le_bytes());
}

#[test]
fn s5_s6_sabotage_rejects_order_width_digest_overwrite_and_outcome_drift() {
    let sequence_payloads = source_payloads(
        serial_typed_record(0, 41, false),
        private_serial_typed_record(2, 41),
        current_set_body("s5-sabotage"),
    );
    let ranges = s5_entry_ranges(&sequence_payloads[4]);

    let mut swapped = sequence_payloads.clone();
    let first = swapped[4][ranges[0].clone()].to_vec();
    let second = swapped[4][ranges[1].clone()].to_vec();
    let rest = swapped[4][ranges[1].end..].to_vec();
    swapped[4].clear();
    swapped[4].extend_from_slice(&second);
    swapped[4].extend_from_slice(&first);
    swapped[4].extend_from_slice(&rest);
    assert_rejected(&swapped, "S5 statement/effect order");

    let mut duplicate = sequence_payloads.clone();
    let first = duplicate[4][ranges[0].clone()].to_vec();
    duplicate[4].splice(ranges[1].clone(), first);
    assert_rejected(&duplicate, "S5 statement/effect order");

    let mut omitted = sequence_payloads.clone();
    omitted[4].truncate(ranges[0].end);
    assert_rejected(&omitted, "S5 physical entry count");

    let mut surplus = sequence_payloads.clone();
    surplus[4].extend_from_slice(&sequence_payloads[4][ranges[0].clone()]);
    assert_rejected(&surplus, "S5 physical entry count");

    let mut published_width = sequence_payloads.clone();
    published_width[4][ranges[0].start + 16..ranges[0].start + 20]
        .copy_from_slice(&(crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES as u32 - 1).to_le_bytes());
    published_width[4].remove(ranges[0].end - 1);
    assert_rejected(&published_width, "S5 published body width is not exact");

    let mut reserved = sequence_payloads.clone();
    reserved[4][ranges[0].start + 10] = 1;
    assert_rejected(&reserved, "S5 flags or reserved");

    let mut digest = sequence_payloads.clone();
    digest[4][ranges[0].start + 20] ^= 1;
    assert_rejected(&digest, "S5 published body digest");

    let mut private_body = sequence_payloads.clone();
    private_body[4][ranges[1].start + 16..ranges[1].start + 20]
        .copy_from_slice(&1_u32.to_le_bytes());
    private_body[4].insert(ranges[1].end, 0);
    assert_rejected(&private_body, "S5 private entry must have an empty");

    let mut private_overwrite = sequence_payloads.clone();
    private_overwrite[4][ranges[1].start + 9] &= !2;
    assert_rejected(&private_overwrite, "S5 private default flags");

    let mut overwrite = source_payloads(
        serial_typed_record(0, 41, false),
        typed_record(2, &[8]),
        current_set_body("s4-overwrite"),
    );
    suppress_s4_row(&mut overwrite[3], 0);
    assert_rejected(&overwrite, "S5 published default reference");

    let mut copy_class = mixed_payloads();
    copy_class[5][8..10].copy_from_slice(&2_u16.to_le_bytes());
    assert_rejected(&copy_class, "S6 CopyInsert");

    let mut returning_flag = mixed_payloads();
    returning_flag[5][10] &= !1;
    assert_rejected(&returning_flag, "S6 typed INSERT RETURNING flag");

    let mut target = mixed_payloads();
    target[5][44 + 28] ^= 1;
    assert_rejected(&target, "S6 statement identity, class, target");

    let mut statement_digest = mixed_payloads();
    statement_digest[5][12] ^= 1;
    assert_rejected(&statement_digest, "S6 statement identity, class, target");

    let mut unknown_flag = mixed_payloads();
    unknown_flag[5][10] |= 4;
    assert_rejected(&unknown_flag, "S6 outcome flags");

    let mut swapped_s6 = mixed_payloads();
    let first = swapped_s6[5][..136].to_vec();
    let second = swapped_s6[5][136..272].to_vec();
    swapped_s6[5][..136].copy_from_slice(&second);
    swapped_s6[5][136..272].copy_from_slice(&first);
    assert_rejected(&swapped_s6, "S6 statement identity");

    let mut duplicate_s6 = mixed_payloads();
    let first = duplicate_s6[5][..136].to_vec();
    duplicate_s6[5][136..272].copy_from_slice(&first);
    assert_rejected(&duplicate_s6, "S6 statement identity");

    let mut one_byte_short_s6 = mixed_payloads();
    one_byte_short_s6[5].pop();
    assert_rejected(&one_byte_short_s6, "S6 payload length");

    let mut omitted_s6 = mixed_payloads();
    omitted_s6[5].truncate(2 * 136);
    assert_rejected(&omitted_s6, "S6 payload length");

    let mut surplus_s6 = mixed_payloads();
    surplus_s6[5].extend_from_slice(&mixed_payloads()[5][..136]);
    assert_rejected(&surplus_s6, "S6 payload length");

    let mut noop = mixed_payloads();
    noop[5][44] = gpu_db_wal::CanonicalOutcomeKind::CommitNoOp as u8;
    noop[5][48..56].fill(0);
    assert_rejected(&noop, "S6 CommitNoOp must have zero affected rows");

    let mut zero_row_returning_noop = mixed_payloads();
    suppress_s4_row(&mut zero_row_returning_noop[3], 0);
    suppress_s4_row(&mut zero_row_returning_noop[3], 1);
    replace_s6_outcome(
        &mut zero_row_returning_noop[5],
        0,
        gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::CommitNoOp,
            affected_rows: 0,
            sqlstate: None,
            constraint_id: 0,
            target_digest: [2; 32],
            returning_digest: [0x44; 32],
        },
    );
    assert!(materialize(&zero_row_returning_noop, 41).is_ok());

    let mut retained_abort = mixed_payloads();
    suppress_s4_row(&mut retained_abort[3], 0);
    suppress_s4_row(&mut retained_abort[3], 1);
    retained_abort[5][10] |= 2;
    retained_abort[7].push(8);
    replace_s6_outcome(
        &mut retained_abort[5],
        0,
        gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::AbortError,
            affected_rows: 0,
            sqlstate: Some(*b"22003"),
            constraint_id: 0,
            target_digest: [2; 32],
            returning_digest: [0; 32],
        },
    );
    assert_rejected(
        &retained_abort,
        "S6 statement identity, class, target, or retention",
    );
}

#[test]
fn s6_strict_s3_returning_sabotage_rejects_flag_digest_retention_and_closure_drift() {
    let returning_body =
        current_command_body("UPDATE source_s6 SET id = 8 WHERE id = 7 RETURNING id");
    let returning = source_payloads_with_s3(
        typed_record(0, &[7]),
        typed_record(2, &[8]),
        returning_body,
        gpu_db_wal::CanonicalFragmentKind::RowMutation,
        41,
    );
    let s3_entry = 136;

    let mut missing_flag = returning.clone();
    missing_flag[5][s3_entry + 10] &= !1;
    assert_rejected(&missing_flag, "S6 current S3 RETURNING flag does not match");

    let mut missing_digest = returning.clone();
    replace_s6_outcome(
        &mut missing_digest[5],
        1,
        gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            affected_rows: 0,
            sqlstate: None,
            constraint_id: 0,
            target_digest: [3; 32],
            returning_digest: [0; 32],
        },
    );
    assert_rejected(
        &missing_digest,
        "S6 current S3 success RETURNING digest presence",
    );

    let mut extra_flag = mixed_payloads();
    extra_flag[5][s3_entry + 10] |= 1;
    assert_rejected(&extra_flag, "S6 current S3 RETURNING flag does not match");

    let mut extra_digest = mixed_payloads();
    replace_s6_outcome(
        &mut extra_digest[5],
        1,
        gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            affected_rows: 0,
            sqlstate: None,
            constraint_id: 0,
            target_digest: [3; 32],
            returning_digest: [0x55; 32],
        },
    );
    assert_rejected(
        &extra_digest,
        "S6 current S3 success RETURNING digest presence",
    );

    let mut retained_abort = returning.clone();
    retained_abort[5][s3_entry + 10] |= 2;
    retained_abort[7].push(8);
    replace_s6_outcome(
        &mut retained_abort[5],
        1,
        gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::AbortError,
            affected_rows: 0,
            sqlstate: Some(*b"22003"),
            constraint_id: 0,
            target_digest: [3; 32],
            returning_digest: [0; 32],
        },
    );
    assert_rejected(
        &retained_abort,
        "S6 statement identity, class, target, or retention",
    );

    let mut missing_s8 = returning.clone();
    missing_s8[5][s3_entry + 10] |= 2;
    assert!(materialize(&missing_s8, 41).is_err());

    let mut retained = returning;
    retained[5][s3_entry + 10] |= 2;
    retained[7].push(8);
    assert!(materialize(&retained, 41).is_ok());
    assert!(materialize_with_view_mutation(&retained, 41, |view| {
        view.flags &= !crate::typed_insert_aggregate::AGGREGATE_FLAG_RETURNING;
    })
    .is_err());
    assert!(materialize_with_view_mutation(&retained, 41, |view| {
        view.outer_flags &= !crate::typed_insert_aggregate::OUTER_CONTENT_RETURNING;
    })
    .is_err());
    assert!(materialize_with_view_mutation(&retained, 41, |view| {
        view.sections[7].entry_count = 0;
    })
    .is_err());
}

#[test]
fn s5_private_survivor_with_overwrite_is_retained_for_later_s7_closure() {
    let mut payloads = source_payloads(
        serial_typed_record(0, 41, false),
        private_serial_typed_record(2, 41),
        current_set_body("private-survivor"),
    );
    // The first private effect belongs to global S4 row 1. It was applied/canceled in the base
    // fixture; preserve it instead while retaining S5's final-value-overwritten witness. S7,
    // not this inert S1--S6 seam, will bind that surviving final image.
    preserve_s4_row_as_survivor(&mut payloads[3], 1);
    let draft = materialize(&payloads, 41).unwrap();
    let CompactSequenceEffectKind::Private {
        final_value_overwritten,
        ..
    } = &draft.sequence_effects[1].kind
    else {
        panic!("private survivor must retain its S5 private witness");
    };
    assert!(*final_value_overwritten);
}

#[test]
fn s5_s3_parent_digest_and_s6_abort_constraint_rules_are_exact() {
    let mut request_not_statement = source_payloads_with_s3(
        typed_record(0, &[7]),
        typed_record(2, &[8]),
        current_command_body("SELECT nextval('source_s5_parent_digest')"),
        gpu_db_wal::CanonicalFragmentKind::SequenceValueTransition,
        41,
    );
    // S1 entry 1's request digest is its codec-4 body digest; the S3 sequence parent must
    // instead use its statement digest, so rehash this otherwise well-formed S5 body.
    let request_digest: [u8; 32] = request_not_statement[0][160..192].try_into().unwrap();
    request_not_statement[4][52 + 36..52 + 68].copy_from_slice(&request_digest);
    let body_digest = gpu_db_wal::canonical_request_digest(&request_not_statement[4][52..138]);
    request_not_statement[4][20..52].copy_from_slice(&body_digest);
    assert_rejected(
        &request_not_statement,
        "S5 explicit sequence reference does not close",
    );

    let base = source_payloads(
        typed_record_without_returning(0, 7),
        typed_record_without_returning(2, 8),
        current_set_body("abort-constraint"),
    );
    for (state, constraint_id) in [
        (*b"23502", 91_u64),
        (*b"23503", 92_u64),
        (*b"23505", 93_u64),
        (*b"23514", 94_u64),
        (*b"22003", 0_u64),
    ] {
        let mut accepted = base.clone();
        suppress_s4_row(&mut accepted[3], 0);
        replace_s6_outcome(
            &mut accepted[5],
            0,
            gpu_db_wal::CanonicalOutcome {
                kind: gpu_db_wal::CanonicalOutcomeKind::AbortError,
                affected_rows: 0,
                sqlstate: Some(state),
                constraint_id,
                target_digest: [2; 32],
                returning_digest: [0; 32],
            },
        );
        assert!(materialize(&accepted, 41).is_ok());
    }
    for (state, constraint_id) in [(*b"23505", 0_u64), (*b"22003", 91_u64)] {
        let mut rejected = base.clone();
        suppress_s4_row(&mut rejected[3], 0);
        replace_s6_outcome(
            &mut rejected[5],
            0,
            gpu_db_wal::CanonicalOutcome {
                kind: gpu_db_wal::CanonicalOutcomeKind::AbortError,
                affected_rows: 0,
                sqlstate: Some(state),
                constraint_id,
                target_digest: [2; 32],
                returning_digest: [0; 32],
            },
        );
        assert_rejected(&rejected, "S6 abort constraint id");
    }
}
