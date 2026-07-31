//! Q2 end-to-end evidence: closed codec owner, independent authority, then exact re-encoding.

use super::super::codec_closed_canonical_semantics_v2_for_test;
use super::{
    q1_sabotage, q1_vectors, q2_witnesses, MinimalAbortFixture, MINIMAL_ABORT_SECTION_HEX,
};
use crate::typed_insert_aggregate::semantics_v2::retained::{
    reencode_q2_golden_from_closed_for_test, validate_q2_golden_sabotage_from_closed_for_test,
    Q2GoldenCase, Q2GoldenSabotage,
};

// Checked-in after independent Q2 generation-input observation.  This is evidence only: the
// builder computes its digest from sealed neutral input before any test literal is consulted.
const SUCCESS_GENERATION_INPUT_DIGEST: [u8; 32] = [
    0x93, 0xb5, 0xb2, 0xc1, 0xa5, 0x1b, 0xbe, 0xf2, 0x86, 0x7b, 0xd6, 0x44, 0xe8, 0x30, 0x0b, 0x93,
    0x0d, 0x05, 0xb6, 0x3e, 0xe5, 0x75, 0xaa, 0x5a, 0x26, 0xb1, 0x1e, 0xcb, 0x29, 0x2a, 0x2b, 0x7f,
];

#[test]
fn q2_minimal_abort_reencodes_from_fully_validated_owner_to_frozen_literals() {
    let fixture = super::minimal_abort_fixture();
    let expected = std::array::from_fn(|ordinal| {
        super::expected_minimal_abort_section(ordinal, MINIMAL_ABORT_SECTION_HEX[ordinal])
    });
    assert_reencodes_to_frozen_literals(
        &fixture,
        q2_witnesses::Q2WitnessCase::MinimalAbort,
        Q2GoldenCase::MinimalAbort,
        expected,
        super::digest_literal(super::MINIMAL_ABORT_SECTION_ROOT_HEX[6]),
        None,
    );
}

#[test]
fn q2_explicit_abort_reencodes_from_fully_validated_owner_to_frozen_literals() {
    let fixture = q1_vectors::explicit_abort_fixture();
    assert_reencodes_to_frozen_literals(
        &fixture,
        q2_witnesses::Q2WitnessCase::ExplicitAbort,
        Q2GoldenCase::ExplicitAbort,
        q1_vectors::explicit_abort_sections_literal_for_q2(),
        q1_vectors::explicit_abort_s7_section_root_literal_for_q2(),
        None,
    );
}

#[test]
fn q2_successful_interleaved_reencodes_from_fully_validated_owner_to_frozen_literals() {
    let fixture = q1_vectors::successful_a_b_a_fixture();
    assert_reencodes_to_frozen_literals(
        &fixture,
        q2_witnesses::Q2WitnessCase::SuccessfulInterleaved,
        Q2GoldenCase::SuccessfulInterleaved,
        q1_vectors::successful_sections_literal_for_q2(),
        q1_vectors::successful_s7_section_root_literal_for_q2(),
        Some(SUCCESS_GENERATION_INPUT_DIGEST),
    );
}

#[test]
fn q2_repaired_root_substitution_reaches_builder_and_is_rejected_unchanged() {
    let fixture = q1_sabotage::q2_repaired_root_substitution_fixture();
    let closed = codec_closed(&fixture);
    q2_witnesses::with_witness(
        q2_witnesses::Q2WitnessCase::SuccessfulInterleaved,
        |catalog, leases| {
            let error = reencode_q2_golden_from_closed_for_test(
                closed,
                catalog,
                leases,
                Q2GoldenCase::SuccessfulInterleaved,
            )
            .expect_err("unchanged Q2 builder rejects the coherently substituted S7 root");
            assert!(
                error
                    .to_string()
                    .contains("generation output header does not match retained envelope")
                    || error
                        .to_string()
                        .contains("generation table output identity/order differs from S7"),
                "root substitution reaches independent builder validation: {error}"
            );
        },
    );
}

#[test]
fn q2_explicit_missing_first_ordinary_guard_reaches_catalog_and_is_rejected() {
    let fixture = q1_sabotage::q2_explicit_missing_first_ordinary_guard_fixture();
    let closed = codec_closed(&fixture);
    q2_witnesses::with_witness(
        q2_witnesses::Q2WitnessCase::ExplicitAbort,
        |catalog, leases| {
            let error = reencode_q2_golden_from_closed_for_test(
                closed,
                catalog,
                leases,
                Q2GoldenCase::ExplicitAbort,
            )
            .expect_err("catalog closure rejects the missing first ordinary guard");
            assert!(
                error
                    .to_string()
                    .contains("catalog guard does not have exactly one statement use"),
                "the old missing-ordinary shape reaches catalog guard closure: {error}"
            );
        },
    );
}

#[test]
fn q2_builder_final_root_sabotage_is_rejected_before_reencoding() {
    let fixture = q1_vectors::successful_a_b_a_fixture();
    let closed = codec_closed(&fixture);
    q2_witnesses::with_witness(
        q2_witnesses::Q2WitnessCase::SuccessfulInterleaved,
        |catalog, leases| {
            let error = validate_q2_golden_sabotage_from_closed_for_test(
                closed,
                catalog,
                leases,
                Q2GoldenCase::SuccessfulInterleaved,
                Q2GoldenSabotage::FinalRoot,
            )
            .expect_err("builder final-root sabotage must fail retained equality");
            assert!(
                error
                    .to_string()
                    .contains("generation output header does not match retained envelope"),
                "builder sabotage reaches final-root equality: {error}"
            );
        },
    );
}

fn assert_reencodes_to_frozen_literals(
    fixture: &impl Q2Fixture,
    witness_case: q2_witnesses::Q2WitnessCase,
    golden_case: Q2GoldenCase,
    expected: [Vec<u8>; 7],
    expected_s7_section_root: [u8; 32],
    expected_generation_input_digest: Option<[u8; 32]>,
) {
    let closed = codec_closed(fixture);
    q2_witnesses::with_witness(witness_case, |catalog, leases| {
        let (actual, builder_generation_input_digest) =
            reencode_q2_golden_from_closed_for_test(closed, catalog, leases, golden_case)
                .expect("strict fill, codec close, pinned witnesses, and Q2 builder validate");
        if let Some(expected_generation_input_digest) = expected_generation_input_digest {
            assert_eq!(
                builder_generation_input_digest, expected_generation_input_digest,
                "builder observes the independent sealed neutral generation-input digest"
            );
        }
        for (ordinal, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let mismatch = actual
                .iter()
                .zip(expected)
                .position(|(actual, expected)| actual != expected);
            assert!(
                actual.len() == expected.len() && mismatch.is_none(),
                "Q2 reencoder S{} literal mismatch: actual_len={}, expected_len={}, first_difference={mismatch:?}",
                ordinal + 1,
                actual.len(),
                expected.len(),
            );
        }
        assert_s7_evidence(&actual[6], &expected[6], expected_s7_section_root);
    });
}

fn assert_s7_evidence(actual: &[u8], expected: &[u8], expected_section_root: [u8; 32]) {
    assert_eq!(actual.len(), expected.len(), "S7 total length");
    assert_eq!(&actual[104..328], &expected[104..328], "S7 fixed directory");
    assert_eq!(
        &actual[536..568],
        &expected[536..568],
        "S7 root-descriptor evidence"
    );
    assert_eq!(
        &actual[568..600],
        &expected[568..600],
        "S7 payload-digest evidence"
    );
    let mut section_header = [0_u8; 16];
    section_header[..2].copy_from_slice(&7_u16.to_le_bytes());
    section_header[4..8].copy_from_slice(&1_u32.to_le_bytes());
    section_header[8..16].copy_from_slice(&(actual.len() as u64).to_le_bytes());
    assert_eq!(
        super::v1_digest(
            b"gpu-db/write001/aggregate-section/v1",
            &[&section_header, actual],
        ),
        expected_section_root,
        "S7 canonical section-root evidence"
    );
}

fn codec_closed(
    fixture: &impl Q2Fixture,
) -> crate::typed_insert_aggregate::semantics_v2::retained::Q2CodecClosedSemanticsV2 {
    let fragments = [
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::RowMutation,
            body: fixture.fragment_body(),
        },
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
            body: fixture.status(),
        },
    ];
    codec_closed_canonical_semantics_v2_for_test(fixture.outer(), fixture.outcome(), &fragments)
        .expect("actual strict fill and codec closure accept the frozen fixture")
}

trait Q2Fixture {
    fn outer(&self) -> &gpu_db_wal::CanonicalPreApplyHeader;
    fn outcome(&self) -> &gpu_db_wal::CanonicalOutcome;
    fn fragment_body(&self) -> &[u8];
    fn status(&self) -> &[u8];
}

impl Q2Fixture for MinimalAbortFixture {
    fn outer(&self) -> &gpu_db_wal::CanonicalPreApplyHeader {
        &self.outer
    }

    fn outcome(&self) -> &gpu_db_wal::CanonicalOutcome {
        &self.outcome
    }

    fn fragment_body(&self) -> &[u8] {
        &self.fragment_body
    }

    fn status(&self) -> &[u8] {
        &self.status
    }
}

impl Q2Fixture for q1_vectors::Q1Fixture {
    fn outer(&self) -> &gpu_db_wal::CanonicalPreApplyHeader {
        &self.outer
    }

    fn outcome(&self) -> &gpu_db_wal::CanonicalOutcome {
        &self.outcome
    }

    fn fragment_body(&self) -> &[u8] {
        &self.fragment_body
    }

    fn status(&self) -> &[u8] {
        &self.status
    }
}
