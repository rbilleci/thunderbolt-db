//! Q3 durable-sequence proof evidence over checked-in codec-5 vectors.

use super::super::fill_canonical_semantics_v2_for_test;
use super::{q1_vectors, q2_witnesses, MinimalAbortFixture};
use crate::typed_insert_aggregate::semantics_v2::retained::{
    with_historical_sequence_pending_for_test, with_live_sequence_pending_for_test,
    AllocatorAssignmentProofSabotageForTest, SequenceOutcomeProofSabotageForTest,
    SequenceOutcomeSpecForTest,
};

#[test]
fn durable_sequence_index_accepts_the_checked_in_explicit_abort_child() {
    let fixture = q1_vectors::explicit_abort_fixture();
    let outcomes = [q1_vectors::explicit_abort_sequence_outcome_for_test()];
    q2_witnesses::with_witness(
        q2_witnesses::Q2WitnessCase::ExplicitAbort,
        |catalog, leases| {
            with_live_sequence_pending_for_test(
                retention_pending(&fixture),
                catalog,
                leases,
                &outcomes,
                None,
                None,
                |generation_pending| {
                    drop(generation_pending);
                    Ok(())
                },
            )
            .expect("published child remains durable when its parent aborts");
        },
    );
}

#[test]
fn durable_sequence_index_rejects_missing_lifecycle_and_every_exact_record_mismatch() {
    let outcomes = [q1_vectors::explicit_abort_sequence_outcome_for_test()];
    for sabotage in [
        SequenceOutcomeProofSabotageForTest::Missing,
        SequenceOutcomeProofSabotageForTest::Duplicate,
        SequenceOutcomeProofSabotageForTest::IndexRoot,
        SequenceOutcomeProofSabotageForTest::Incomplete,
        SequenceOutcomeProofSabotageForTest::Nondurable,
        SequenceOutcomeProofSabotageForTest::Unpublished,
        SequenceOutcomeProofSabotageForTest::RecoveryPrefix,
        SequenceOutcomeProofSabotageForTest::CheckpointLineage,
        SequenceOutcomeProofSabotageForTest::CheckpointRetention,
        SequenceOutcomeProofSabotageForTest::ForwardCommit,
        SequenceOutcomeProofSabotageForTest::CrossLineage,
        SequenceOutcomeProofSabotageForTest::TransitionId,
        SequenceOutcomeProofSabotageForTest::SequenceOid,
        SequenceOutcomeProofSabotageForTest::ParentTxn,
        SequenceOutcomeProofSabotageForTest::ParentAutocommit,
        SequenceOutcomeProofSabotageForTest::ParentRequestDigest,
        SequenceOutcomeProofSabotageForTest::StatementOrdinal,
        SequenceOutcomeProofSabotageForTest::ExpressionOrdinal,
        SequenceOutcomeProofSabotageForTest::ReturnedValue,
        SequenceOutcomeProofSabotageForTest::InputDigest,
        SequenceOutcomeProofSabotageForTest::SourceName,
        SequenceOutcomeProofSabotageForTest::NonDefault,
    ] {
        let fixture = q1_vectors::explicit_abort_fixture();
        q2_witnesses::with_witness(
            q2_witnesses::Q2WitnessCase::ExplicitAbort,
            |catalog, leases| {
                let error = with_live_sequence_pending_for_test(
                    retention_pending(&fixture),
                    catalog,
                    leases,
                    &outcomes,
                    Some(sabotage),
                    None,
                    |generation_pending| {
                        drop(generation_pending);
                        Ok(())
                    },
                )
                .expect_err("durable-sequence sabotage must quarantine before generation");
                assert!(
                    error.to_string().contains("durable sequence"),
                    "{sabotage:?} must fail only through durable-sequence validation: {error}"
                );
            },
        );
    }
}

#[test]
fn durable_sequence_index_rejects_hidden_unreferenced_lookup_key_value_mismatch() {
    let fixture = q1_vectors::explicit_abort_fixture();
    let outcomes = [q1_vectors::explicit_abort_sequence_outcome_for_test()];
    q2_witnesses::with_witness(
        q2_witnesses::Q2WitnessCase::ExplicitAbort,
        |catalog, leases| {
            let error = with_live_sequence_pending_for_test(
                retention_pending(&fixture),
                catalog,
                leases,
                &outcomes,
                Some(SequenceOutcomeProofSabotageForTest::HiddenLookupTransitionMismatch),
                None,
                |generation_pending| {
                    drop(generation_pending);
                    Ok(())
                },
            )
            .expect_err("a complete index rejects an unreferenced corrupt lookup/value pair");
            assert!(
                error.to_string().contains("durable sequence"),
                "the hidden mismatch must fail while authenticating the complete durable index: {error}"
            );
        },
    );
}

#[test]
fn empty_s5_authenticates_the_complete_index_for_live_and_historical_retention() {
    let fixture = super::minimal_abort_fixture();
    let outcomes: [SequenceOutcomeSpecForTest<'_>; 0] = [];
    q2_witnesses::with_witness(
        q2_witnesses::Q2WitnessCase::MinimalAbort,
        |catalog, leases| {
            with_live_sequence_pending_for_test(
                retention_pending(&fixture),
                catalog,
                leases,
                &outcomes,
                None,
                None,
                |generation_pending| {
                    drop(generation_pending);
                    Ok(())
                },
            )
            .expect("empty S5 advances after live index authentication");
        },
    );
    q2_witnesses::with_witness(
        q2_witnesses::Q2WitnessCase::MinimalAbort,
        |catalog, leases| {
            with_historical_sequence_pending_for_test(
                retention_pending(&fixture),
                catalog,
                leases,
                &outcomes,
                None,
                None,
                |generation_pending| {
                    drop(generation_pending);
                    Ok(())
                },
            )
            .expect("empty S5 advances after historical index authentication");
        },
    );
}

#[test]
fn allocator_assignment_closes_minimal_and_explicit_source_rows_before_generation() {
    let minimal = super::minimal_abort_fixture();
    let empty: [SequenceOutcomeSpecForTest<'_>; 0] = [];
    q2_witnesses::with_witness(
        q2_witnesses::Q2WitnessCase::MinimalAbort,
        |catalog, leases| {
            with_historical_sequence_pending_for_test(
                retention_pending(&minimal),
                catalog,
                leases,
                &empty,
                None,
                None,
                |generation_pending| {
                    drop(generation_pending);
                    Ok(())
                },
            )
            .expect("minimal suppressed S1/S2 row retains its exact durable assignment");
        },
    );

    let explicit = q1_vectors::explicit_abort_fixture();
    let outcomes = [q1_vectors::explicit_abort_sequence_outcome_for_test()];
    q2_witnesses::with_witness(
        q2_witnesses::Q2WitnessCase::ExplicitAbort,
        |catalog, leases| {
            with_live_sequence_pending_for_test(
                retention_pending(&explicit),
                catalog,
                leases,
                &outcomes,
                None,
                None,
                |generation_pending| {
                    drop(generation_pending);
                    Ok(())
                },
            )
            .expect("explicit canceled and suppressed S1/S2 rows retain exact durable assignments");
        },
    );
}

#[test]
fn allocator_assignment_rejects_every_parent_bound_exact_mapping_sabotage() {
    let outcomes = [q1_vectors::explicit_abort_sequence_outcome_for_test()];
    for sabotage in [
        AllocatorAssignmentProofSabotageForTest::Absent,
        AllocatorAssignmentProofSabotageForTest::Duplicate,
        AllocatorAssignmentProofSabotageForTest::WrongParent,
        AllocatorAssignmentProofSabotageForTest::HiddenConflictingParent,
        AllocatorAssignmentProofSabotageForTest::HiddenDifferentParentOverlap,
        AllocatorAssignmentProofSabotageForTest::StatementChain,
        AllocatorAssignmentProofSabotageForTest::Table,
        AllocatorAssignmentProofSabotageForTest::Order,
        AllocatorAssignmentProofSabotageForTest::Range,
        AllocatorAssignmentProofSabotageForTest::Lineage,
        AllocatorAssignmentProofSabotageForTest::Lifecycle,
        AllocatorAssignmentProofSabotageForTest::MarkerIdentity,
    ] {
        let fixture = q1_vectors::explicit_abort_fixture();
        q2_witnesses::with_witness(
            q2_witnesses::Q2WitnessCase::ExplicitAbort,
            |catalog, leases| {
                let error = with_live_sequence_pending_for_test(
                    retention_pending(&fixture),
                    catalog,
                    leases,
                    &outcomes,
                    None,
                    Some(sabotage),
                    |generation_pending| {
                        drop(generation_pending);
                        Ok(())
                    },
                )
                .expect_err("allocator assignment sabotage must reject before generation");
                assert!(
                    error.to_string().contains("allocator assignment"),
                    "{sabotage:?} must fail through allocator assignment validation before generation: {error}"
                );
                if sabotage == AllocatorAssignmentProofSabotageForTest::HiddenConflictingParent {
                    assert!(
                        error
                            .to_string()
                            .contains("stable-parent root allocator assignment"),
                        "{sabotage:?} must fail stable-parent completeness before selected-witness validation: {error}"
                    );
                }
                if sabotage == AllocatorAssignmentProofSabotageForTest::HiddenDifferentParentOverlap
                {
                    assert!(
                        error
                            .to_string()
                            .contains("overlapping root allocator assignment stable row IDs"),
                        "{sabotage:?} must reject the root-authenticated foreign-parent row overlap before closure: {error}"
                    );
                }
                if matches!(
                    sabotage,
                    AllocatorAssignmentProofSabotageForTest::Lifecycle
                        | AllocatorAssignmentProofSabotageForTest::MarkerIdentity
                ) {
                    assert!(
                        error.to_string().contains("lease marker identity"),
                        "{sabotage:?} must reject a reused or mismatched durable lease marker: {error}"
                    );
                }
            },
        );
    }
}

fn retention_pending(
    fixture: &impl Q3Fixture,
) -> crate::typed_insert_aggregate::semantics_v2::retained::AggregateReplayTxn<
    crate::typed_insert_aggregate::semantics_v2::retained::RetentionAuthorityPending,
> {
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
    fill_canonical_semantics_v2_for_test(fixture.outer(), fixture.outcome(), &fragments)
        .expect("actual strict fill accepts the frozen fixture")
        .close_codec()
        .expect("actual codec closure accepts the frozen fixture")
}

trait Q3Fixture {
    fn outer(&self) -> &gpu_db_wal::CanonicalPreApplyHeader;
    fn outcome(&self) -> &gpu_db_wal::CanonicalOutcome;
    fn fragment_body(&self) -> &[u8];
    fn status(&self) -> &[u8];
}

impl Q3Fixture for MinimalAbortFixture {
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

impl Q3Fixture for q1_vectors::Q1Fixture {
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
