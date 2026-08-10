//! Coherently reframed hostile S8 fixtures.
//!
//! The fixtures refresh the S8 payload digest plus aggregate and STATUS2 roots after mutation,
//! so rejection is attributable to the production S8 semantic proof rather than stale framing.

use super::{
    measure_canonical_semantics_v2,
    s8_vectors::{reframe_s8_after_mutation_for_sabotage, selective_a_b_a_fixture},
};

#[test]
fn coherently_rehashed_s8_selection_substitution_is_rejected_by_the_raw_s8_proof() {
    let fixture = reframe_s8_after_mutation_for_sabotage(selective_a_b_a_fixture(), |s8| {
        let selection_offset = u64::from_le_bytes(
            s8[72..80]
                .try_into()
                .expect("fixed S8 selection offset width"),
        ) as usize;
        s8[selection_offset + 16..selection_offset + 20].copy_from_slice(&u32::MAX.to_le_bytes());
    });
    assert_raw_s8_rejection(
        fixture,
        "S8 row selection does not biject a selected S4 row",
    );
}

#[test]
fn coherently_rehashed_s8_header_root_substitution_is_rejected_by_the_raw_s8_proof() {
    let fixture = reframe_s8_after_mutation_for_sabotage(selective_a_b_a_fixture(), |s8| {
        s8[144] ^= 1;
    });
    assert_raw_s8_rejection(fixture, "S8 fixed header identity is invalid");
}

#[test]
fn clear_bit_generic_request_identity_rejects_outer_status_and_s8_disagreement() {
    let mut fixture = selective_a_b_a_fixture();
    assert_eq!(
        fixture.outer.flags
            & crate::typed_insert_aggregate::OUTER_FLAG_FIRST_TYPED_INSERT_WRITER_EPOCH,
        0,
        "the generic closure test must exercise the clear-bit protocol"
    );
    // Keep the authenticated aggregate bytes, STATUS2, and present S8 header mutually coherent,
    // then change only the enclosing generic request authority. This proves the clear-bit rule
    // cannot split retry/status identity from the outer canonical request.
    fixture.outer.request_digest[0] ^= 1;
    assert_ne!(
        &fixture.status[52..84],
        fixture.outer.request_digest.as_slice()
    );
    assert_ne!(
        &fixture.sections[7][112..144],
        fixture.outer.request_digest.as_slice()
    );
    assert_raw_s8_rejection(
        fixture,
        "generic v2 request identity does not close over outer/STATUS2",
    );
}

#[test]
fn coherently_rehashed_s8_role_two_image_substitution_is_rejected_by_the_raw_s8_proof() {
    let fixture = reframe_s8_after_mutation_for_sabotage(selective_a_b_a_fixture(), |s8| {
        let image_offset =
            u64::from_le_bytes(s8[88..96].try_into().expect("fixed S8 image offset width"))
                as usize;
        s8[image_offset + 20..image_offset + 24].copy_from_slice(&1_u32.to_le_bytes());
    });
    assert_raw_s8_rejection(fixture, "S8 nested shared image fails strict measurement");
}

fn assert_raw_s8_rejection(fixture: super::q1_vectors::Q1Fixture, expected: &str) {
    let fragments = [
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::RowMutation,
            body: &fixture.fragment_body,
        },
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
            body: &fixture.status,
        },
    ];
    let error = measure_canonical_semantics_v2(&fixture.outer, &fixture.outcome, &fragments)
        .expect_err("coherently reframed S8 wire substitution cannot survive the raw proof");
    assert!(
        error.to_string().contains(expected),
        "S8 raw boundary rejects the fully reframed hostile wire at {expected}: {error}"
    );
}
