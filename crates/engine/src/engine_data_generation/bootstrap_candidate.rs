//! Move-only, phase-only owner for one prepared bootstrap publication build.
//!
//! This is deliberately not a materialized generation. It retains the root-free GPU compiler
//! capability, comparator-only expectations, and the complete replay/checkpoint carry together
//! so a later phase cannot reload or substitute any of those facts from live state.

use gpu_db_execution::{
    RuntimeGenerationRebuildAttempt, RuntimeGenerationRebuildError, RuntimeGenerationRebuildTarget,
};

use super::{
    bootstrap_publication::BootstrapPublicationCarry,
    bootstrap_rebuild::{
        BootstrapRebuildCompilerInput, BootstrapRebuildExpectations,
        BootstrapRebuildPreparationToken,
    },
    bootstrap_rebuild_gpu::{
        prepare_v1_single_table_int4_rebuild, BootstrapRuntimeGenerationRebuildCompletion,
        BootstrapRuntimeGenerationRebuildPrepareFailure, BootstrapRuntimeGenerationRebuildProof,
        BootstrapRuntimeGenerationRebuildSubmission,
        BootstrapRuntimeGenerationRebuildUnknownQuiescence,
        PreparedBootstrapRuntimeGenerationRebuild,
    },
};

/// One fully prepared, phase-only pre-GPU bootstrap publication build. The private fields
/// intentionally give this phase no read, installation, CUDA-success, WAL, or publication
/// conversion surface.
/// It has neither `Clone` nor `Debug`, and can only be formed by bootstrap rebuild preparation.
#[must_use = "a prepared bootstrap publication build must be retained for its next private phase or intentionally dropped"]
pub(super) struct PreparedBootstrapPublicationBuild {
    compiler: BootstrapRebuildCompilerInput,
    expectations: BootstrapRebuildExpectations,
    publication_carry: BootstrapPublicationCarry,
}

/// Private ownership relay around the sealed V1 rebuild adapter. This phase may submit and
/// complete device work, but it cannot compare roots, decode a proof, or install a generation.
#[must_use = "a prepared bootstrap publication attempt must be enqueued or intentionally dropped"]
pub(super) struct PreparedBootstrapPublicationAttempt {
    lower: PreparedBootstrapRuntimeGenerationRebuild,
    expectations: BootstrapRebuildExpectations,
    publication_carry: BootstrapPublicationCarry,
}

/// One submitted bootstrap publication attempt. The same comparator-only and publication facts
/// remain move-only while the lower execution submission owns device completion state.
#[must_use = "a submitted bootstrap publication attempt must be completed or intentionally dropped"]
pub(super) struct BootstrapPublicationAttemptSubmission {
    lower: BootstrapRuntimeGenerationRebuildSubmission,
    expectations: BootstrapRebuildExpectations,
    publication_carry: BootstrapPublicationCarry,
}

/// Terminal result of the private attempt wrapper. The opaque success proof remains phase-only;
/// failures retain the exact expectation and publication facts consumed by the attempt.
#[must_use = "a bootstrap publication attempt completion must be retained for its next private phase or intentionally dropped"]
#[allow(
    clippy::large_enum_variant,
    reason = "boxing unknown quiescence would allocate after the lower attempt has been enqueued"
)]
pub(super) enum BootstrapPublicationAttemptCompletion {
    Quiesced(Result<BootstrapPublicationPhaseProof, BootstrapPublicationAttemptFailure>),
    UnknownQuiescence(BootstrapPublicationAttemptUnknownQuiescence),
}

/// A fail-closed unknown-quiescence wrapper. It can only retry the same lower submission and
/// carries no root, generation, or recovery authority.
#[must_use = "an unknown-quiescence bootstrap publication attempt must be retried or intentionally dropped"]
pub(super) struct BootstrapPublicationAttemptUnknownQuiescence {
    lower: BootstrapRuntimeGenerationRebuildUnknownQuiescence,
    expectations: BootstrapRebuildExpectations,
    publication_carry: BootstrapPublicationCarry,
}

/// Move-only failure owner for either preparation or terminal lower execution failure.
#[must_use = "a bootstrap publication attempt failure retains consumed facts and must not be discarded accidentally"]
pub(super) struct BootstrapPublicationAttemptFailure {
    lower: BootstrapPublicationAttemptFailureKind,
    expectations: BootstrapRebuildExpectations,
    publication_carry: BootstrapPublicationCarry,
}

enum BootstrapPublicationAttemptFailureKind {
    Prepare(Box<BootstrapRuntimeGenerationRebuildPrepareFailure>),
    Terminal(RuntimeGenerationRebuildError),
}

/// Opaque successful GPU proof bundled with the exact expectation and publication carry that
/// entered the attempt. This bundle exposes neither root comparison nor a generation conversion.
#[must_use = "a bootstrap publication phase proof must be retained for its next private phase or intentionally dropped"]
pub(super) struct BootstrapPublicationPhaseProof {
    lower: BootstrapRuntimeGenerationRebuildProof,
    expectations: BootstrapRebuildExpectations,
    publication_carry: BootstrapPublicationCarry,
}

impl PreparedBootstrapPublicationBuild {
    /// The unforgeable token is minted only by `bootstrap_rebuild`: its tuple field is private to
    /// that sibling, so another `engine_data_generation` module cannot create this build even
    /// though the candidate type must remain visible to the preparation owner.
    pub(super) fn from_prepared_parts(
        _preparation: BootstrapRebuildPreparationToken,
        compiler: BootstrapRebuildCompilerInput,
        expectations: BootstrapRebuildExpectations,
        publication_carry: BootstrapPublicationCarry,
    ) -> Self {
        Self {
            compiler,
            expectations,
            publication_carry,
        }
    }

    /// Consume the prepared compiler capability at the sealed V1 adapter boundary. Comparator
    /// expectations and the complete publication carry are relayed unchanged to every outcome.
    pub(super) fn prepare_v1_single_table_int4_attempt(
        self,
        target: RuntimeGenerationRebuildTarget,
        attempt: RuntimeGenerationRebuildAttempt,
    ) -> Result<PreparedBootstrapPublicationAttempt, Box<BootstrapPublicationAttemptFailure>> {
        let Self {
            compiler,
            expectations,
            publication_carry,
        } = self;
        match prepare_v1_single_table_int4_rebuild(compiler, target, attempt) {
            Ok(lower) => Ok(PreparedBootstrapPublicationAttempt {
                lower,
                expectations,
                publication_carry,
            }),
            Err(lower) => Err(prepare_failure_with_facts(
                lower,
                expectations,
                publication_carry,
            )),
        }
    }
}

impl PreparedBootstrapPublicationAttempt {
    pub(super) fn enqueue(self) -> BootstrapPublicationAttemptSubmission {
        BootstrapPublicationAttemptSubmission {
            lower: self.lower.enqueue(),
            expectations: self.expectations,
            publication_carry: self.publication_carry,
        }
    }
}

impl BootstrapPublicationAttemptSubmission {
    pub(super) fn complete(self) -> BootstrapPublicationAttemptCompletion {
        complete_attempt_with_facts(
            self.lower.complete(),
            self.expectations,
            self.publication_carry,
        )
    }
}

impl BootstrapPublicationAttemptUnknownQuiescence {
    pub(super) fn retry_complete(self) -> BootstrapPublicationAttemptCompletion {
        complete_attempt_with_facts(
            self.lower.retry_complete(),
            self.expectations,
            self.publication_carry,
        )
    }
}

fn prepare_failure_with_facts(
    lower: Box<BootstrapRuntimeGenerationRebuildPrepareFailure>,
    expectations: BootstrapRebuildExpectations,
    publication_carry: BootstrapPublicationCarry,
) -> Box<BootstrapPublicationAttemptFailure> {
    Box::new(BootstrapPublicationAttemptFailure {
        lower: BootstrapPublicationAttemptFailureKind::Prepare(lower),
        expectations,
        publication_carry,
    })
}

fn complete_attempt_with_facts(
    completion: BootstrapRuntimeGenerationRebuildCompletion,
    expectations: BootstrapRebuildExpectations,
    publication_carry: BootstrapPublicationCarry,
) -> BootstrapPublicationAttemptCompletion {
    match completion {
        BootstrapRuntimeGenerationRebuildCompletion::Quiesced(Ok(lower)) => {
            BootstrapPublicationAttemptCompletion::Quiesced(Ok(BootstrapPublicationPhaseProof {
                lower,
                expectations,
                publication_carry,
            }))
        }
        BootstrapRuntimeGenerationRebuildCompletion::Quiesced(Err(error)) => {
            BootstrapPublicationAttemptCompletion::Quiesced(Err(
                BootstrapPublicationAttemptFailure {
                    lower: BootstrapPublicationAttemptFailureKind::Terminal(error),
                    expectations,
                    publication_carry,
                },
            ))
        }
        BootstrapRuntimeGenerationRebuildCompletion::UnknownQuiescence(lower) => {
            BootstrapPublicationAttemptCompletion::UnknownQuiescence(
                BootstrapPublicationAttemptUnknownQuiescence {
                    lower,
                    expectations,
                    publication_carry,
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use gpu_db_execution::CudaDriverRuntime;

    use super::*;
    use crate::engine_data_generation::{
        bootstrap_rebuild::prepare_bootstrap_rebuild, resources, DataGenerationError,
    };

    #[test]
    fn wrapper_types_are_move_only_and_have_no_live_or_generation_surface() {
        macro_rules! assert_not_clone {
            ($type:ty) => {{
                trait AmbiguousIfClone<Marker> {
                    fn marker() {}
                }
                impl<T: ?Sized> AmbiguousIfClone<()> for T {}
                impl<T: Clone> AmbiguousIfClone<u8> for T {}
                let _ = <$type as AmbiguousIfClone<_>>::marker;
            }};
        }
        macro_rules! assert_not_debug {
            ($type:ty) => {{
                trait AmbiguousIfDebug<Marker> {
                    fn marker() {}
                }
                impl<T: ?Sized> AmbiguousIfDebug<()> for T {}
                impl<T: ?Sized + std::fmt::Debug> AmbiguousIfDebug<u8> for T {}
                let _ = <$type as AmbiguousIfDebug<_>>::marker;
            }};
        }
        assert_not_clone!(PreparedBootstrapPublicationBuild);
        assert_not_clone!(PreparedBootstrapPublicationAttempt);
        assert_not_clone!(BootstrapPublicationAttemptSubmission);
        assert_not_clone!(BootstrapPublicationAttemptCompletion);
        assert_not_clone!(BootstrapPublicationAttemptUnknownQuiescence);
        assert_not_clone!(BootstrapPublicationAttemptFailure);
        assert_not_clone!(BootstrapPublicationPhaseProof);
        assert_not_debug!(PreparedBootstrapPublicationBuild);
        assert_not_debug!(PreparedBootstrapPublicationAttempt);
        assert_not_debug!(BootstrapPublicationAttemptSubmission);
        assert_not_debug!(BootstrapPublicationAttemptCompletion);
        assert_not_debug!(BootstrapPublicationAttemptUnknownQuiescence);
        assert_not_debug!(BootstrapPublicationAttemptFailure);
        assert_not_debug!(BootstrapPublicationPhaseProof);

        let source = include_str!("bootstrap_candidate.rs");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("candidate production section");
        for required in [
            "compiler: BootstrapRebuildCompilerInput",
            "expectations: BootstrapRebuildExpectations",
            "publication_carry: BootstrapPublicationCarry",
            "_preparation: BootstrapRebuildPreparationToken",
            "lower: PreparedBootstrapRuntimeGenerationRebuild",
            "lower: BootstrapRuntimeGenerationRebuildSubmission",
            "lower: BootstrapRuntimeGenerationRebuildUnknownQuiescence",
            "lower: BootstrapRuntimeGenerationRebuildProof",
            "BootstrapPublicationAttemptFailureKind::Prepare(lower)",
            "BootstrapPublicationAttemptFailureKind::Terminal(error)",
        ] {
            assert!(production.contains(required), "candidate lost {required}");
        }
        for forbidden in [
            ["Arc", "Swap"].concat(),
            ["Read", "State"].concat(),
            ["fn ", "into_gpu"].concat(),
            ["fn ", "into_compiler"].concat(),
            ["fn ", "into_generation"].concat(),
            ["fn ", "into_live"].concat(),
            ["fn ", "into_reader"].concat(),
            ["fn ", "compare"].concat(),
            ["fn ", "recover"].concat(),
            ["fn ", "install"].concat(),
            ["fn ", "publish"].concat(),
            ["fn ", "wal"].concat(),
        ] {
            assert!(
                !production.contains(&forbidden),
                "candidate wrapper unexpectedly exposes {forbidden}"
            );
        }

        let source = include_str!("bootstrap_publication.rs");
        for required in [
            "_durable_cut: BootstrapDurableCut",
            "_root_format: RootFormatVersion",
            "_database_id: DatabaseId",
            "_catalog_identity: BootstrapCatalogPair",
            "_catalog_snapshot: Arc<crate::engine_state::CatalogSnapshot>",
            "_stable_ids: BootstrapStableIdState",
            "_terminal_status: ReplayedTerminalStatusWitness",
            "_current_table_expectations: Vec<BootstrapCurrentTable>",
            "_canonical_resource_ledger: Box<[BootstrapResourceLedgerEntry]>",
        ] {
            assert!(
                source.contains(required),
                "publication carry lost {required}"
            );
        }
        assert!(
            !source.contains(&["Uninstalled", "PublicationGeneration"].concat()),
            "the pre-GPU replay wrapper must not be promoted as a generation"
        );

        let rebuild = include_str!("bootstrap_rebuild.rs");
        assert!(
            rebuild.contains("pub(super) struct BootstrapRebuildPreparationToken(());"),
            "only bootstrap rebuild may mint the candidate preparation token"
        );
    }

    #[test]
    fn lower_prepare_failure_retains_every_consumed_candidate_fact_until_dropped() {
        let (attached, owner) =
            resources::attached_resources_with_owner_witness_for_bootstrap_rebuild_test();
        let build = prepare_bootstrap_rebuild(attached)
            .unwrap_or_else(|_| panic!("prepared bootstrap publication build"));
        let PreparedBootstrapPublicationBuild {
            compiler,
            expectations,
            publication_carry,
        } = build;
        let failure = prepare_failure_with_facts(
            Box::new(BootstrapRuntimeGenerationRebuildPrepareFailure::Compiler {
                error: DataGenerationError::Invalid("test lower preparation failure"),
                compiler,
            }),
            expectations,
            publication_carry,
        );
        assert!(
            owner.upgrade().is_some(),
            "the failure owner must retain the lower compiler capability"
        );

        let BootstrapPublicationAttemptFailure {
            lower,
            expectations,
            publication_carry,
        } = *failure;
        let BootstrapPublicationAttemptFailureKind::Prepare(lower) = lower else {
            panic!("test constructed a lower preparation failure");
        };
        assert!(matches!(
            &*lower,
            BootstrapRuntimeGenerationRebuildPrepareFailure::Compiler { .. }
        ));
        assert!(
            owner.upgrade().is_some(),
            "moving the exact lower failure out must retain its compiler capability"
        );
        drop(expectations);
        drop(publication_carry);
        drop(lower);
        assert!(
            owner.upgrade().is_none(),
            "releasing the lower failure releases the retained attached owner"
        );
    }

    #[test]
    fn completion_relays_success_bundle_and_unknown_retry_without_new_authority() {
        let source = include_str!("bootstrap_candidate.rs");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("candidate production section");
        for required in [
            "BootstrapRuntimeGenerationRebuildCompletion::Quiesced(Ok(lower))",
            "BootstrapPublicationPhaseProof {",
            "BootstrapRuntimeGenerationRebuildCompletion::UnknownQuiescence(lower)",
            "self.lower.retry_complete()",
            "complete_attempt_with_facts(",
        ] {
            assert!(
                production.contains(required),
                "completion relay lost {required}"
            );
        }
        for forbidden in [
            "root_format()",
            "slot_count()",
            "into_durable",
            "into_durable_v1_commitments",
            "into_generation",
            "compare_",
            "compare_durable_v1_commitments_with_table_map",
            "RuntimeGenerationRebuildV1TableMapCompletion",
            "fn extract",
        ] {
            assert!(
                !production.contains(forbidden),
                "candidate completion must not inspect or convert opaque proof via {forbidden}"
            );
        }
    }

    #[test]
    fn actual_gpu_all_cold_wrapper_accepts_nullable_and_all_valid_tail_sources() {
        let Ok(runtime) = CudaDriverRuntime::probe() else {
            return;
        };
        if runtime.snapshot().device_count == 0 {
            return;
        }
        let Ok(target) = runtime.runtime_generation_rebuild_target(0) else {
            return;
        };

        for (all_valid_tail, attempt) in [(false, 701_u64), (true, 702_u64)] {
            let attached = resources::attached_v1_single_table_int4_resources_for_gpu_wrapper_test(
                &target,
                all_valid_tail,
            );
            let build = prepare_bootstrap_rebuild(attached)
                .unwrap_or_else(|_| panic!("prepared V1 cold GPU-wrapper build"));
            let prepared = build
                .prepare_v1_single_table_int4_attempt(
                    target.clone(),
                    RuntimeGenerationRebuildAttempt::new(attempt)
                        .expect("nonzero GPU-wrapper attempt"),
                )
                .unwrap_or_else(|_| panic!("prepared V1 cold GPU-wrapper attempt"));
            match prepared.enqueue().complete() {
                BootstrapPublicationAttemptCompletion::Quiesced(Ok(proof)) => {
                    let BootstrapPublicationPhaseProof {
                        lower,
                        expectations,
                        publication_carry,
                    } = proof;
                    assert_eq!(lower.attempt(), attempt);
                    assert_eq!(lower.slot_count(), 200);
                    drop(lower);
                    drop(expectations);
                    drop(publication_carry);
                }
                BootstrapPublicationAttemptCompletion::Quiesced(Err(_)) => {
                    panic!("GPU wrapper rejected valid all-cold V1 source")
                }
                BootstrapPublicationAttemptCompletion::UnknownQuiescence(_) => {
                    panic!("GPU wrapper left valid all-cold V1 source at unknown quiescence")
                }
            }
        }
    }
}
