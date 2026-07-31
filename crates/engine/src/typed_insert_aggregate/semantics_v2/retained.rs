//! Move-only post-pass-zero ownership states for codec-5 semantics v2.
//!
//! This module deliberately has no WAL, recovery, apply, device, result, or publication caller.
//! It defines the only retained-state boundary: a raw-proof owner is quarantined, exact catalog
//! plus allocator witnesses can make it generation-pending, and only an independently produced
//! generation witness may make it fully witness-validated.  A later strict S2 source pass fills
//! the private retained graph before any transition constructor is exposed.

#[path = "retained/catalog_validation.rs"]
mod catalog_validation;
#[path = "retained/codec_closure.rs"]
mod codec_closure;
#[path = "retained/fill.rs"]
mod fill;
#[path = "retained/generation_validation.rs"]
mod generation_validation;
#[path = "retained/graph.rs"]
mod graph;
#[cfg(test)]
#[path = "retained/reencode.rs"]
mod reencode;
#[path = "retained/reservation.rs"]
mod reservation;

/// Shared identity captured from the canonical outer/S7 framing before a model can leave raw
/// proof. It owns no catalog or publication authority.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SemanticsV2BoundIdentity {
    pub(super) database_id: [u8; 16],
    pub(super) cluster_id: [u8; 16],
    pub(super) timeline_id: [u8; 16],
    pub(super) format_epoch: u64,
    pub(super) leader_epoch: u64,
    pub(super) catalog_epoch: u64,
    pub(super) catalog_digest: [u8; 32],
    pub(super) stable_transaction_id: u64,
    pub(super) autocommit: bool,
    pub(super) commit_sequence: u64,
    pub(super) initial_database_root: [u8; 32],
}

/// Borrowed catalog and allocator validation material. It is intentionally outside codec bytes
/// and has no fallback catalog lookup behavior. The full retained graph performs the row-by-row
/// joins before this input can advance the first typestate.
///
/// The allocator input is a sealed proof from the durable allocator index, not a codec-owned
/// slice of Boolean claims.  Its construction remains unavailable to the raw decoder.
#[allow(dead_code)]
pub(super) struct SemanticsV2CatalogAllocatorWitness<'a> {
    pub(super) catalog: SemanticsV2CatalogWitness<'a>,
    allocator_index: catalog_validation::SemanticsV2DurableAllocatorIndexProof<'a>,
}

/// Literal test-fixture input for the closure-scoped durable allocator adapter below.  Keeping
/// this wrapper at the retained boundary lets sibling golden fixtures provide only exact table
/// intervals, while the authenticated index, active pin, selected records, and proof remain
/// wholly inside `catalog_validation`.
#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) struct AllocatorLeaseSpecForTest {
    pub(super) stable_allocator_id: u64,
    pub(super) lease_start: u64,
    pub(super) lease_end: u64,
}

/// Closed independent generation evidence cases accepted by the retained Q2 fixture facade.
/// This is intentionally distinct from the private builder's enum so sibling golden tests never
/// receive a builder, candidate, work owner, or module path.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Q2GoldenCase {
    MinimalAbort,
    ExplicitAbort,
    SuccessfulInterleaved,
}

/// Closed hostile-output variants for the Q2 independent generation facade.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Q2GoldenSabotage {
    FinalRoot,
}

#[allow(dead_code)]
pub(super) struct SemanticsV2CatalogWitness<'a> {
    pub(super) database_id: [u8; 16],
    pub(super) catalog_epoch: u64,
    pub(super) catalog_digest: [u8; 32],
    pub(super) tables: &'a [SemanticsV2CatalogTableWitness<'a>],
    pub(super) indexes: &'a [SemanticsV2CatalogIndexWitness<'a>],
    pub(super) domains: &'a [SemanticsV2CatalogDomainWitness<'a>],
    pub(super) guards: &'a [SemanticsV2CatalogGuardWitness<'a>],
    pub(super) sequences: &'a [SemanticsV2CatalogSequenceWitness<'a>],
}

#[allow(dead_code)]
pub(super) struct SemanticsV2CatalogTableWitness<'a> {
    pub(super) stable_table_id: u64,
    pub(super) display_oid: u32,
    pub(super) schema: &'a str,
    pub(super) name: &'a str,
    pub(super) schema_digest: [u8; 32],
    pub(super) data_generation: u64,
    pub(super) data_root: [u8; 32],
    pub(super) catalog_columns: &'a [SemanticsV2CatalogColumnWitness<'a>],
    pub(super) not_null_guards: &'a [SemanticsV2CatalogGuardWitness<'a>],
    pub(super) check_guards: &'a [SemanticsV2CatalogGuardWitness<'a>],
    pub(super) foreign_keys: &'a [SemanticsV2CatalogForeignKeyWitness<'a>],
}

#[allow(dead_code)]
pub(super) struct SemanticsV2CatalogColumnWitness<'a> {
    pub(super) catalog_column_ordinal: u32,
    pub(super) stable_column_id: u32,
    pub(super) attnum: i16,
    pub(super) name: &'a str,
    pub(super) storage: [u8; 4],
    pub(super) declared_type_oid: u32,
    pub(super) signed_type_size: i16,
    pub(super) column_shape_digest: [u8; 32],
    pub(super) column_root: [u8; 32],
}

#[allow(dead_code)]
pub(super) struct SemanticsV2CatalogIndexWitness<'a> {
    pub(super) stable_index_id: u64,
    pub(super) display_oid: u32,
    pub(super) owner_stable_table_id: u64,
    pub(super) owner_display_oid: u32,
    pub(super) schema: &'a str,
    pub(super) name: &'a str,
    pub(super) owner_schema: &'a str,
    pub(super) owner_name: &'a str,
    pub(super) constraint_stable_id: u64,
    pub(super) constraint_display_oid: u32,
    pub(super) constraint_schema: &'a str,
    pub(super) constraint_name: &'a str,
    pub(super) schema_digest: [u8; 32],
    pub(super) index_flags: u32,
    pub(super) null_equality_policy: u8,
    pub(super) raw_catalog_ordinal: u32,
    pub(super) key_columns: &'a [SemanticsV2CatalogIndexKeyWitness<'a>],
    pub(super) catalog_epoch: u64,
    pub(super) base_generation: u64,
    pub(super) base_root: [u8; 32],
}

#[allow(dead_code)]
pub(super) struct SemanticsV2CatalogIndexKeyWitness<'a> {
    pub(super) key_ordinal: u32,
    pub(super) owner_catalog_column_ordinal: u32,
    pub(super) stable_column_id: u32,
    pub(super) attnum: i16,
    pub(super) name: &'a str,
    pub(super) storage: [u8; 4],
    pub(super) declared_type_oid: u32,
    pub(super) signed_type_size: i16,
    pub(super) column_name_digest: [u8; 32],
}

#[allow(dead_code)]
pub(super) struct SemanticsV2CatalogDomainWitness<'a> {
    pub(super) stable_domain_id: u64,
    pub(super) display_oid: u32,
    pub(super) schema: &'a str,
    pub(super) name: &'a str,
    pub(super) storage: [u8; 4],
    pub(super) declared_type_oid: u32,
    pub(super) signed_type_size: i16,
    pub(super) storage_shape_digest: [u8; 32],
    pub(super) catalog_generation: u64,
    pub(super) constraints: &'a [SemanticsV2CatalogGuardWitness<'a>],
}

#[allow(dead_code)]
pub(super) struct SemanticsV2CatalogGuardWitness<'a> {
    pub(super) kind: u8,
    pub(super) stable_guard_id: u64,
    pub(super) display_oid: u32,
    pub(super) schema: &'a str,
    pub(super) name: &'a str,
    pub(super) synthesized_not_null: bool,
    pub(super) owner_kind: u8,
    pub(super) owner_stable_id: u64,
    pub(super) owner_display_oid: u32,
    pub(super) owner_catalog_column_ordinal: u32,
    pub(super) domain_ordinal: u32,
    pub(super) raw_constraint_ordinal: u32,
    pub(super) source_ordinal: u32,
    pub(super) shape_digest: [u8; 32],
    pub(super) program_or_descriptor_root: [u8; 32],
    pub(super) catalog_generation: u64,
}

#[allow(dead_code)]
pub(super) struct SemanticsV2CatalogForeignKeyWitness<'a> {
    pub(super) stable_constraint_id: u64,
    pub(super) display_oid: u32,
    pub(super) raw_foreign_key_ordinal: u32,
    pub(super) schema: &'a str,
    pub(super) name: &'a str,
    pub(super) child_catalog_column_ordinal: u32,
    pub(super) child_stable_column_id: u32,
    pub(super) parent_stable_table_id: u64,
    pub(super) parent_display_oid: u32,
    pub(super) parent_catalog_column_ordinal: u32,
    pub(super) parent_stable_column_id: u32,
    pub(super) supporting_stable_index_id: u64,
}

#[allow(dead_code)]
pub(super) struct SemanticsV2CatalogSequenceWitness<'a> {
    pub(super) stable_sequence_id: u64,
    pub(super) display_oid: u32,
    pub(super) schema: &'a str,
    pub(super) name: &'a str,
    pub(super) catalog_generation: u64,
    pub(super) descriptor_digest: [u8; 32],
}

/// Private retained graph placeholder. It is intentionally unconstructable outside the complete
/// post-reservation source/image decoder, which is the sole owner allowed to add decoded S1--S7
/// fields. Keeping it private prevents a test helper from manufacturing a phase transition.
#[allow(dead_code)]
struct RetainedSemanticsV2Graph {
    identity: SemanticsV2BoundIdentity,
    graph: graph::ReservedSemanticsV2Graph,
}

/// Context-free structural decode. This state has only raw/fixed-directory proof plus strict
/// source ownership. It is deliberately unable to consult catalog or allocator evidence: Q1
/// must first consume it through the codec-only S1--S7 closure.
#[allow(dead_code)]
pub(super) struct QuarantinedSemanticsV2 {
    graph: RetainedSemanticsV2Graph,
}

/// Context-free codec closure has recomputed every witness-free S1--S7 fact from the retained
/// typed graph. Only this move-only state may consume catalog/allocator evidence. It remains
/// inert: no generation builder, reencoder, WAL, recovery, apply, device, result, or
/// publication path is reachable here.
#[allow(dead_code)]
pub(super) struct CodecClosedSemanticsV2 {
    graph: RetainedSemanticsV2Graph,
}

/// Catalog and durable allocator closure has succeeded. Only this state may be handed to the
/// external reserved generation builder; it is still not publication eligible.
#[allow(dead_code)]
pub(super) struct GenerationPendingSemanticsV2<'a> {
    graph: RetainedSemanticsV2Graph,
    catalog_and_allocator: CatalogAndAllocatorValidated<'a>,
}

/// All catalog, lease, and generation output equality checks have succeeded.  The catalog and
/// allocator borrows end at this transition: a future reencoder can only receive the retained
/// graph plus the builder's opaque, fully-owned generation result.
#[allow(dead_code)]
pub(super) struct FullyWitnessValidatedSemanticsV2<C> {
    graph: RetainedSemanticsV2Graph,
    generation: generation_validation::OwnedGenerationResult<C>,
}

#[cfg(test)]
impl<C> FullyWitnessValidatedSemanticsV2<C> {
    /// Rebuild the canonical S1--S7 payloads only from a fully witness-validated owner.
    ///
    /// This is a private golden-evidence seam: it has no raw aggregate input, graph getter,
    /// generation-result extractor, production compilation, or live write/recovery route.
    pub(super) fn reencode_s1_s7_for_test(&self) -> Result<[Vec<u8>; 7], crate::EngineError> {
        reencode::reencode_s1_s7_for_test(self)
    }
}

struct CatalogAndAllocatorValidated<'a> {
    catalog: SemanticsV2CatalogWitness<'a>,
    allocator_index: catalog_validation::SemanticsV2DurableAllocatorIndexProof<'a>,
}

/// Seal the complete, exact post-reservation graph into the context-free quarantine state.
///
/// The strict S1--S7 fill is the sole caller: it consumes `into_exact_graph()` immediately
/// before this transition, so a partially filled owner cannot become witness-checkable.
fn quarantine_after_strict_fill(
    identity: SemanticsV2BoundIdentity,
    graph: graph::ReservedSemanticsV2Graph,
) -> QuarantinedSemanticsV2 {
    QuarantinedSemanticsV2 {
        graph: RetainedSemanticsV2Graph { identity, graph },
    }
}

/// Private retained-boundary entry for a successful allocation-free proof.  The facade may pass
/// it a bounded framing, but only this module's child can consume the exact graph owner or seal
/// a quarantine state.
pub(super) fn fill_after_pass_zero(
    framing: &super::super::codec::DecodedAggregateFraming<'_>,
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    pass_zero: super::pass_zero::SemanticsV2PassZero,
) -> Result<QuarantinedSemanticsV2, crate::EngineError> {
    fill::fill_after_pass_zero(framing, outer, pass_zero)
}

#[cfg(test)]
pub(super) fn fail_source_copy_at_for_test<T>(attempt: u64, operation: impl FnOnce() -> T) -> T {
    fill::fail_copy_at_for_test(attempt, operation)
}

impl QuarantinedSemanticsV2 {
    /// Consume structural ownership only after every witness-free S1--S7 relation has been
    /// recomputed from strict S2/image owners. A failed close drops the raw quarantine, so no
    /// caller can retain a partially closed or externally advanceable graph.
    pub(super) fn close_codec(self) -> Result<CodecClosedSemanticsV2, crate::EngineError> {
        codec_closure::validate(self.graph.identity, &self.graph.graph)?;
        Ok(CodecClosedSemanticsV2 { graph: self.graph })
    }

    /// Test-only continuation from an actual strict-fill state. It has no raw constructor and
    /// still must traverse the same consuming codec closure as production scaffolding.
    #[cfg(test)]
    pub(super) fn close_codec_for_test(self) -> Result<CodecClosedSemanticsV2, crate::EngineError> {
        self.close_codec()
    }

    /// Test-only guard continuation.  It takes the sole existing codec-close bridge before the
    /// catalog leaf, keeping the facade from gaining a second close transition spelling.
    #[cfg(test)]
    pub(super) fn validate_guards_after_codec_for_test(
        self,
        catalog: &SemanticsV2CatalogWitness<'_>,
    ) -> Result<(), crate::EngineError> {
        self.close_codec_for_test()?
            .validate_guards_for_test(catalog)
    }
}

impl CodecClosedSemanticsV2 {
    /// Consume the sole codec-closed owner after exact pinned-catalog and durable allocator
    /// closure. No generation result is accepted, retained, or constructed at this boundary.
    pub(super) fn validate_catalog_and_allocator<'a>(
        self,
        witness: SemanticsV2CatalogAllocatorWitness<'a>,
    ) -> Result<GenerationPendingSemanticsV2<'a>, crate::EngineError> {
        catalog_validation::validate(self.graph.identity, &self.graph.graph, &witness)?;
        let SemanticsV2CatalogAllocatorWitness {
            catalog,
            allocator_index,
        } = witness;
        Ok(GenerationPendingSemanticsV2 {
            graph: self.graph,
            catalog_and_allocator: CatalogAndAllocatorValidated {
                catalog,
                allocator_index,
            },
        })
    }

    /// Test-only continuation from an actual codec-closed state. It cannot construct a raw/S7
    /// graph and does not expose the generation witness constructor or any live path.
    #[cfg(test)]
    pub(super) fn validate_catalog_and_allocator_for_test<'a>(
        self,
        witness: SemanticsV2CatalogAllocatorWitness<'a>,
    ) -> Result<GenerationPendingSemanticsV2<'a>, crate::EngineError> {
        self.validate_catalog_and_allocator(witness)
    }

    /// Consume a codec-closed graph through only the catalog-guard witness leaf.  This is a
    /// test seam for hostile evidence; it neither exposes retained state nor constructs a
    /// generation-pending owner.
    #[cfg(test)]
    pub(super) fn validate_guards_for_test(
        self,
        catalog: &SemanticsV2CatalogWitness<'_>,
    ) -> Result<(), crate::EngineError> {
        catalog_validation::validate_guards_for_test(
            self.graph.identity,
            &self.graph.graph,
            catalog,
        )
    }
}

/// Consume an actual codec-closed owner through the one catalog/allocator transition while the
/// test fixture's durable proof is still scoped to this call.  The callback receives only the
/// resulting pending typestate; it cannot retain or construct a proof, inspect the graph, or
/// spell another codec-close transition.
#[cfg(test)]
pub(super) fn with_catalog_allocator_pending_for_test<T>(
    closed: CodecClosedSemanticsV2,
    catalog: SemanticsV2CatalogWitness<'_>,
    leases: &[AllocatorLeaseSpecForTest],
    operation: impl FnOnce(GenerationPendingSemanticsV2<'_>) -> Result<T, crate::EngineError>,
) -> Result<T, crate::EngineError> {
    let identity = closed.graph.identity;
    let authority_leases: Vec<_> = leases
        .iter()
        .map(|lease| catalog_validation::AllocatorLeaseSpecForTest {
            stable_allocator_id: lease.stable_allocator_id,
            lease_start: lease.lease_start,
            lease_end: lease.lease_end,
        })
        .collect();
    catalog_validation::with_allocator_proof_for_test(
        identity,
        &authority_leases,
        |allocator_index| {
            operation(closed.validate_catalog_and_allocator_for_test(
                SemanticsV2CatalogAllocatorWitness {
                    catalog,
                    allocator_index,
                },
            )?)
        },
    )
}

/// Complete the Q2 test-only catalog, allocator, generation, and logical-reencoding chain.
/// The fixture contributes only a real codec-closed owner, pinned catalog facts, exact allocator
/// intervals, and a closed case tag.  The private builder/candidate/work types and the final
/// validated owner never cross this retained facade.
#[cfg(test)]
pub(super) fn reencode_q2_golden_from_closed_for_test(
    closed: CodecClosedSemanticsV2,
    catalog: SemanticsV2CatalogWitness<'_>,
    leases: &[AllocatorLeaseSpecForTest],
    case: Q2GoldenCase,
) -> Result<([Vec<u8>; 7], gpu_db_wal::CanonicalDigest), crate::EngineError> {
    with_catalog_allocator_pending_for_test(closed, catalog, leases, |pending| {
        generation_validation::golden_builder::validate_and_reencode_case(
            pending,
            golden_builder_case(case),
        )
    })
}

/// Drive the same closed Q2 fixture chain with an independently configured hostile generation
/// output.  It returns no owner or reconstructed bytes, so a failed builder validation cannot
/// become a side channel around the fully validated typestate.
#[cfg(test)]
pub(super) fn validate_q2_golden_sabotage_from_closed_for_test(
    closed: CodecClosedSemanticsV2,
    catalog: SemanticsV2CatalogWitness<'_>,
    leases: &[AllocatorLeaseSpecForTest],
    case: Q2GoldenCase,
    sabotage: Q2GoldenSabotage,
) -> Result<(), crate::EngineError> {
    with_catalog_allocator_pending_for_test(closed, catalog, leases, |pending| {
        generation_validation::golden_builder::validate_case_sabotage_result(
            pending,
            golden_builder_case(case),
            golden_builder_sabotage(sabotage),
        )
    })
}

#[cfg(test)]
fn golden_builder_case(case: Q2GoldenCase) -> generation_validation::golden_builder::GoldenCase {
    match case {
        Q2GoldenCase::MinimalAbort => {
            generation_validation::golden_builder::GoldenCase::MinimalAbort
        }
        Q2GoldenCase::ExplicitAbort => {
            generation_validation::golden_builder::GoldenCase::ExplicitAbort
        }
        Q2GoldenCase::SuccessfulInterleaved => {
            generation_validation::golden_builder::GoldenCase::SuccessfulInterleaved
        }
    }
}

#[cfg(test)]
fn golden_builder_sabotage(
    sabotage: Q2GoldenSabotage,
) -> generation_validation::golden_builder::GoldenBuilderSabotage {
    match sabotage {
        Q2GoldenSabotage::FinalRoot => {
            generation_validation::golden_builder::GoldenBuilderSabotage::FinalRoot
        }
    }
}

impl<'a> GenerationPendingSemanticsV2<'a> {
    /// Consume this sole post-catalog owner through the sealed, already-reserved generation
    /// builder.  The builder result is compared against the retained graph before this returns
    /// the final witness-validated typestate; no wire-derived generation witness is accepted.
    fn validate_generation_with_reserved_builder<B>(
        self,
        builder: B,
        quarantine_registry: &generation_validation::GenerationQuarantineRegistry<
            B::Candidate,
            B::Work,
        >,
    ) -> Result<FullyWitnessValidatedSemanticsV2<B::Candidate>, crate::EngineError>
    where
        B: generation_validation::SemanticsV2ReservedGenerationBuilder,
    {
        generation_validation::validate_reserved_builder_result(self, builder, quarantine_registry)
    }

    #[cfg(test)]
    fn mutate_final_table_generation_before_neutral_seal_for_test(&mut self) {
        self.graph.graph.tables[0].data_generation_after = self.graph.graph.tables[0]
            .data_generation_after
            .checked_add(1)
            .expect("test final generation remains representable");
    }
}

/// Validate only the catalog/allocator identities which do not depend on retained S1--S7 joins.
/// This is the first phase only: it deliberately cannot inspect, store, or accept a generation
/// result. The complete graph validator consumes this proof before it constructs the pending
/// owner and hands that owner to the sealed generation-builder interface.
#[allow(dead_code)]
pub(super) fn validate_catalog_allocator_witness_identity(
    identity: SemanticsV2BoundIdentity,
    witness: &SemanticsV2CatalogAllocatorWitness<'_>,
) -> Result<(), crate::EngineError> {
    let catalog = &witness.catalog;
    if identity.database_id == [0; 16]
        || identity.catalog_epoch == 0
        || identity.catalog_digest == [0; 32]
        || identity.cluster_id == [0; 16]
        || identity.timeline_id == [0; 16]
        || identity.format_epoch == 0
        || identity.leader_epoch == 0
        || identity.stable_transaction_id == 0
        || identity.commit_sequence == 0
        || identity.initial_database_root == [0; 32]
        || catalog.database_id != identity.database_id
        || catalog.catalog_epoch != identity.catalog_epoch
        || catalog.catalog_digest != identity.catalog_digest
    {
        return Err(retained_error(
            "witness group identity does not match the sealed v2 envelope",
        ));
    }
    catalog_validation::validate_durable_allocator_index_identity(
        identity,
        &witness.allocator_index,
    )
}

fn retained_error(message: &str) -> crate::EngineError {
    crate::EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v2 retained: {message}"
    ))
}

#[cfg(test)]
mod generation_lifecycle_tests {
    use super::*;
    use crate::typed_insert_aggregate::{
        encode_status_v2, TypedInsertStatusV2, AGGREGATE_CHUNK_FLAG_FIRST,
        AGGREGATE_CHUNK_FLAG_LAST, AGGREGATE_CHUNK_HEADER_BYTES, AGGREGATE_CHUNK_MAGIC,
        AGGREGATE_FORMAT_VERSION, AGGREGATE_SECTION_COUNT, AGGREGATE_STATUS_V2_BYTES,
        AGGREGATE_STREAM_MAGIC, ENGINE_OPERATION_CODEC_TYPED_INSERT_AGGREGATE, OUTER_CONTENT_ROW,
        OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1,
    };
    use sha2::{Digest, Sha256};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };

    const TABLE_ID: u64 = 101;
    const TABLE_GENERATION: u64 = 11;
    const TERMINAL_CONSTRAINT_ID: u64 = 301;
    const STABLE_TRANSACTION_ID: u64 = 77;
    const COMMIT_SEQUENCE: u64 = 17;
    const CATALOG_EPOCH: u64 = 7;
    const ABSENT_U32: u32 = u32::MAX;

    #[test]
    fn checked_in_abort_fixture_crosses_fill_catalog_pending_and_owned_lifecycle() {
        let (result, calls, drops) =
            run_abort_lifecycle(generation_validation::DeterministicAbortSabotage::None);
        result.expect("checked-in fixture reaches fully owned generation result");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn checked_in_abort_fixture_rejects_prelaunch_and_output_sabotage() {
        use generation_validation::DeterministicAbortSabotage as Sabotage;

        let (result, calls, drops) = run_abort_lifecycle(Sabotage::CandidateReservationFailure);
        assert!(
            result.is_err(),
            "candidate reservation failure rejects before launch"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "prelaunch failure performs no work"
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "no candidate exists on reserve failure"
        );

        for (name, sabotage) in [
            ("missing", Sabotage::MissingOutput),
            ("duplicate", Sabotage::DuplicateOutput),
            ("extra", Sabotage::ExtraOutput),
            ("header identity", Sabotage::HeaderIdentity),
            ("header input digest", Sabotage::HeaderInputDigest),
            ("header final root", Sabotage::HeaderFinalRoot),
            ("table identity", Sabotage::TableIdentity),
            ("table generation", Sabotage::TableGeneration),
            ("table root", Sabotage::TableRoot),
            ("table count", Sabotage::TableCount),
            ("table index range", Sabotage::TableIndexRange),
            ("neutral input", Sabotage::NeutralInput),
        ] {
            let (result, calls, drops) = run_abort_lifecycle(sabotage);
            assert!(
                result.is_err(),
                "{name} sabotage must fail independent validation"
            );
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1,
                "sabotage still drains once"
            );
            assert_eq!(
                drops.load(Ordering::SeqCst),
                1,
                "quiesced sabotage releases candidate"
            );
        }
    }

    #[test]
    fn checked_in_abort_fixture_drains_immediate_and_partial_launch_failures() {
        use generation_validation::DeterministicAbortSabotage as Sabotage;

        let (result, calls, drops, parked) =
            run_abort_lifecycle_with_observer(Sabotage::ImmediateLaunchFailure, None);
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(parked, 0, "immediate failure is proven quiesced");

        let (result, calls, drops, parked) =
            run_abort_lifecycle_with_observer(Sabotage::PartialLaunchFailure, None);
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            parked, 1,
            "partial failure retains the real fixture backing"
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "only the authorized reaper releases partial fixture backing",
        );
    }

    fn run_abort_lifecycle(
        sabotage: generation_validation::DeterministicAbortSabotage,
    ) -> (
        Result<(), crate::EngineError>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let (result, calls, drops, _) = run_abort_lifecycle_with_observer(sabotage, None);
        (result, calls, drops)
    }

    #[test]
    fn neutral_input_digest_ignores_final_output_sabotage() {
        use generation_validation::DeterministicAbortSabotage as Sabotage;

        let normal = Arc::new(Mutex::new(None));
        let root = Arc::new(Mutex::new(None));
        let generation = Arc::new(Mutex::new(None));
        assert!(
            run_abort_lifecycle_with_observer(Sabotage::None, Some(Arc::clone(&normal)))
                .0
                .is_ok()
        );
        assert!(run_abort_lifecycle_with_observer(
            Sabotage::HeaderFinalRoot,
            Some(Arc::clone(&root))
        )
        .0
        .is_err());
        assert!(run_abort_lifecycle_with_observer(
            Sabotage::TableGeneration,
            Some(Arc::clone(&generation))
        )
        .0
        .is_err());
        let normal = normal
            .lock()
            .expect("normal observation lock")
            .expect("normal digest");
        assert_eq!(
            normal,
            root.lock()
                .expect("root observation lock")
                .expect("root digest"),
            "final database root is not a neutral builder input",
        );
        assert_eq!(
            normal,
            generation
                .lock()
                .expect("generation observation lock")
                .expect("generation digest"),
            "final generation is not a neutral builder input",
        );
    }

    #[test]
    fn preseal_final_generation_mutation_keeps_neutral_digest_but_rejects_output() {
        use generation_validation::DeterministicAbortSabotage as Sabotage;

        let baseline = Arc::new(Mutex::new(None));
        let mutated = Arc::new(Mutex::new(None));
        assert!(
            run_abort_lifecycle_with_observer(Sabotage::None, Some(Arc::clone(&baseline)))
                .0
                .is_ok()
        );
        let (result, calls, drops, parked) =
            run_abort_lifecycle_inner(Sabotage::None, Some(Arc::clone(&mutated)), true);
        assert!(
            result.is_err(),
            "S7 final generation output must still be validated"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(parked, 0);
        let baseline_digest = *baseline.lock().expect("baseline input-digest lock");
        let mutated_digest = *mutated.lock().expect("mutated input-digest lock");
        assert_eq!(
            baseline_digest, mutated_digest,
            "pre-seal S7 final generation is outside the neutral builder digest",
        );
    }

    fn run_abort_lifecycle_with_observer(
        sabotage: generation_validation::DeterministicAbortSabotage,
        observed_input_digest: Option<Arc<Mutex<Option<[u8; 32]>>>>,
    ) -> (
        Result<(), crate::EngineError>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
        usize,
    ) {
        run_abort_lifecycle_inner(sabotage, observed_input_digest, false)
    }

    fn run_abort_lifecycle_inner(
        sabotage: generation_validation::DeterministicAbortSabotage,
        observed_input_digest: Option<Arc<Mutex<Option<[u8; 32]>>>>,
        mutate_final_generation_before_neutral_seal: bool,
    ) -> (
        Result<(), crate::EngineError>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
        usize,
    ) {
        let (outer, outcome, fragment, status) = checked_in_minimal_abort_fixture();
        let fragments = [
            gpu_db_wal::CanonicalFragmentRef {
                kind: gpu_db_wal::CanonicalFragmentKind::RowMutation,
                body: &fragment,
            },
            gpu_db_wal::CanonicalFragmentRef {
                kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
                body: &status,
            },
        ];
        let closed =
            super::super::fill_canonical_semantics_v2_for_test(&outer, &outcome, &fragments)
                .expect("checked-in minimal abort fixture fills retained owners")
                .close_codec_for_test()
                .expect("checked-in minimal abort fixture closes codec witnesses");

        let record = crate::typed_insert_batch::decode_canonical_typed_insert_record(&frozen_hex(
            "MINIMAL_ABORT_NULL_S2_TYPED_INSERT_HEX",
        ))
        .expect("checked-in fixture S2 record decodes");
        let facts = record.facts();
        let source_column = record
            .catalog_columns()
            .next()
            .expect("checked-in fixture has one target column");
        assert!(record.catalog_columns().nth(1).is_none());
        let columns = [SemanticsV2CatalogColumnWitness {
            catalog_column_ordinal: source_column.catalog_column_ordinal,
            stable_column_id: source_column.column_id,
            attnum: source_column.attnum,
            name: source_column.name,
            storage: storage(source_column.ty),
            declared_type_oid: source_column.type_oid,
            signed_type_size: source_column.type_size,
            column_shape_digest: [0x23; 32],
            column_root: [0x24; 32],
        }];
        let guards = [not_null_guard(facts.target.oid)];
        let tables = [SemanticsV2CatalogTableWitness {
            stable_table_id: TABLE_ID,
            display_oid: facts.target.oid,
            schema: "public",
            name: "codec_golden",
            schema_digest: facts.target.schema_digest,
            data_generation: TABLE_GENERATION,
            data_root: [0x22; 32],
            catalog_columns: &columns,
            not_null_guards: &guards,
            check_guards: &[],
            foreign_keys: &[],
        }];
        let catalog = SemanticsV2CatalogWitness {
            database_id: [0xa1; 16],
            catalog_epoch: CATALOG_EPOCH,
            catalog_digest: [0x33; 32],
            tables: &tables,
            indexes: &[],
            domains: &[],
            guards: &guards,
            sequences: &[],
        };
        let identity = SemanticsV2BoundIdentity {
            database_id: [0xa1; 16],
            cluster_id: [0xa3; 16],
            timeline_id: [0xa2; 16],
            format_epoch: 5,
            leader_epoch: 6,
            catalog_epoch: CATALOG_EPOCH,
            catalog_digest: [0x33; 32],
            stable_transaction_id: STABLE_TRANSACTION_ID,
            autocommit: true,
            commit_sequence: COMMIT_SEQUENCE,
            initial_database_root: [0x44; 32],
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let quarantine_registry = generation_validation::GenerationQuarantineRegistry::<
            generation_validation::AbortCandidate,
            generation_validation::AbortWork,
        >::new();
        let result = catalog_validation::with_allocator_proof_for_test(
            identity,
            &[catalog_validation::AllocatorLeaseSpecForTest {
                stable_allocator_id: TABLE_ID,
                lease_start: 1,
                lease_end: 1_000,
            }],
            |allocator_index| {
                let mut pending = closed
                    .validate_catalog_and_allocator_for_test(SemanticsV2CatalogAllocatorWitness {
                        catalog,
                        allocator_index,
                    })
                    .expect("checked-in fixture closes pinned catalog and allocator");
                if mutate_final_generation_before_neutral_seal {
                    pending.mutate_final_table_generation_before_neutral_seal_for_test();
                }
                pending.validate_generation_with_reserved_builder(match observed_input_digest {
                        Some(observed) => generation_validation::DeterministicAbortBuilder::with_observed_input_digest(
                            Arc::clone(&calls),
                            Arc::clone(&drops),
                            sabotage,
                            observed,
                        ),
                        None => generation_validation::DeterministicAbortBuilder::with_sabotage(
                            Arc::clone(&calls),
                            Arc::clone(&drops),
                            sabotage,
                        ),
                    }, &quarantine_registry)
            },
        );
        let quarantine_reaper = quarantine_registry.authorized_reaper_for_test();
        let parked = quarantine_reaper.occupancy().occupied;
        if parked != 0 {
            quarantine_reaper.reap_after_external_quiescence();
        }
        // Successful output has no catalog/allocator lifetime and can move past every borrowed
        // witness; unknown failure is retained until this separately authorized test reaper.
        let result = result.map(drop);
        (result, calls, drops, parked)
    }

    fn not_null_guard(table_oid: u32) -> SemanticsV2CatalogGuardWitness<'static> {
        SemanticsV2CatalogGuardWitness {
            kind: 9,
            stable_guard_id: TERMINAL_CONSTRAINT_ID,
            display_oid: 0,
            schema: "",
            name: "",
            synthesized_not_null: true,
            owner_kind: 1,
            owner_stable_id: TABLE_ID,
            owner_display_oid: table_oid,
            owner_catalog_column_ordinal: 0,
            domain_ordinal: ABSENT_U32,
            raw_constraint_ordinal: 0,
            source_ordinal: 0,
            shape_digest: [0x23; 32],
            program_or_descriptor_root: [0x24; 32],
            catalog_generation: 12,
        }
    }

    fn storage(ty: crate::SqlType) -> [u8; 4] {
        match ty {
            crate::SqlType::Int2 => [1, 0, 0, 0],
            crate::SqlType::Int4 => [2, 0, 0, 0],
            crate::SqlType::Int8 => [3, 0, 0, 0],
            crate::SqlType::Numeric { precision, scale } => [4, precision, scale, 0],
            crate::SqlType::Bool => [5, 0, 0, 0],
            crate::SqlType::Text => [6, 0, 0, 0],
            crate::SqlType::Date => [7, 0, 0, 0],
            crate::SqlType::Timestamp => [8, 0, 0, 0],
            crate::SqlType::Uuid => [9, 0, 0, 0],
        }
    }

    fn checked_in_minimal_abort_fixture() -> (
        gpu_db_wal::CanonicalPreApplyHeader,
        gpu_db_wal::CanonicalOutcome,
        Vec<u8>,
        Vec<u8>,
    ) {
        let s1 = frozen_hex("MINIMAL_ABORT_S1_HEX");
        let s2 = frozen_hex("MINIMAL_ABORT_NULL_S2_TYPED_INSERT_HEX");
        let s7 = frozen_hex("MINIMAL_ABORT_S7_HEX");
        let typed_digest: [u8; 32] = s1[16..48].try_into().expect("S1 digest width");
        let overlay_after: [u8; 32] = s1[112..144].try_into().expect("S1 overlay width");
        let mut s4 = vec![0; 64];
        s4[8..16].copy_from_slice(&100_u64.to_le_bytes());
        s4[16] = 3;
        s4[20..24].copy_from_slice(&0_u32.to_le_bytes());
        s4[24..28].copy_from_slice(&ABSENT_U32.to_le_bytes());
        s4[32..64].copy_from_slice(&typed_digest);
        let statement_outcome = gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::AbortError,
            affected_rows: 0,
            sqlstate: Some(*b"23502"),
            constraint_id: TERMINAL_CONSTRAINT_ID,
            target_digest: overlay_after,
            returning_digest: [0; 32],
        };
        let mut statement_outcome_bytes = [0; gpu_db_wal::CANONICAL_OUTCOME_BYTES];
        gpu_db_wal::encode_canonical_outcome_into_exact(
            &statement_outcome,
            &mut statement_outcome_bytes,
        )
        .expect("test abort outcome encodes");
        let mut s6 = vec![0; 136];
        s6[8..10].copy_from_slice(&1_u16.to_le_bytes());
        s6[12..44].copy_from_slice(&typed_digest);
        s6[44..136].copy_from_slice(&statement_outcome_bytes);
        let mut s2_section = (s2.len() as u32).to_le_bytes().to_vec();
        s2_section.extend_from_slice(&s2);
        let sections = [
            s1,
            s2_section,
            Vec::new(),
            s4,
            Vec::new(),
            s6,
            s7,
            Vec::new(),
        ];
        let entries = [1_u32, 1, 0, 1, 0, 1, 1, 0];
        let headers: [[u8; 16]; AGGREGATE_SECTION_COUNT] = std::array::from_fn(|index| {
            let mut header = [0; 16];
            header[..2].copy_from_slice(&(index as u16 + 1).to_le_bytes());
            header[4..8].copy_from_slice(&entries[index].to_le_bytes());
            header[8..16].copy_from_slice(&(sections[index].len() as u64).to_le_bytes());
            header
        });
        let section_bytes = headers
            .iter()
            .zip(sections.iter())
            .map(|(_, section)| 16_u64 + section.len() as u64)
            .sum::<u64>();
        let mut aggregate_header = [0; 96];
        aggregate_header[..16].copy_from_slice(AGGREGATE_STREAM_MAGIC);
        aggregate_header[16..18].copy_from_slice(&AGGREGATE_FORMAT_VERSION.to_le_bytes());
        aggregate_header[18..20].copy_from_slice(&2_u16.to_le_bytes());
        aggregate_header[20..22].copy_from_slice(&AGGREGATE_FORMAT_VERSION.to_le_bytes());
        aggregate_header[22..24].copy_from_slice(&AGGREGATE_FORMAT_VERSION.to_le_bytes());
        aggregate_header[24..28].copy_from_slice(&1_u32.to_le_bytes());
        aggregate_header[28..30].copy_from_slice(&(AGGREGATE_SECTION_COUNT as u16).to_le_bytes());
        aggregate_header[32..40].copy_from_slice(&section_bytes.to_le_bytes());
        aggregate_header[40..48].copy_from_slice(&STABLE_TRANSACTION_ID.to_le_bytes());
        aggregate_header[48..52].copy_from_slice(&1_u32.to_le_bytes());
        aggregate_header[52..56].copy_from_slice(&1_u32.to_le_bytes());
        aggregate_header[56..64].copy_from_slice(&1_u64.to_le_bytes());
        aggregate_header[88..92].copy_from_slice(&1_u32.to_le_bytes());
        let section_roots: [[u8; 32]; AGGREGATE_SECTION_COUNT] = std::array::from_fn(|index| {
            v1_digest(
                b"gpu-db/write001/aggregate-section/v1",
                &[&headers[index], &sections[index]],
            )
        });
        let aggregate_root = v1_digest(
            b"gpu-db/write001/aggregate-root/v1",
            &[
                &aggregate_header,
                &section_roots[0],
                &section_roots[1],
                &section_roots[2],
                &section_roots[3],
                &section_roots[4],
                &section_roots[5],
                &section_roots[6],
                &section_roots[7],
            ],
        );
        let mut stream = aggregate_header.to_vec();
        for (header, section) in headers.iter().zip(sections.iter()) {
            stream.extend_from_slice(header);
            stream.extend_from_slice(section);
        }
        stream.extend_from_slice(&aggregate_root);
        let mut fragment = vec![0; AGGREGATE_CHUNK_HEADER_BYTES as usize];
        fragment[..8].copy_from_slice(AGGREGATE_CHUNK_MAGIC);
        fragment[8] = ENGINE_OPERATION_CODEC_TYPED_INSERT_AGGREGATE;
        fragment[9] = AGGREGATE_FORMAT_VERSION as u8;
        fragment[10..12].copy_from_slice(
            &(AGGREGATE_CHUNK_FLAG_FIRST | AGGREGATE_CHUNK_FLAG_LAST).to_le_bytes(),
        );
        fragment[12..20].copy_from_slice(&(stream.len() as u64).to_le_bytes());
        fragment[24..28].copy_from_slice(&1_u32.to_le_bytes());
        fragment[36..40].copy_from_slice(&(stream.len() as u32).to_le_bytes());
        fragment[44..76].copy_from_slice(&aggregate_root);
        fragment.extend_from_slice(&stream);
        let request_digest = v2_digest(
            b"gpu-db/write001/aggregate-request/v2",
            &[
                &[1],
                &1_u32.to_le_bytes(),
                &0_u32.to_le_bytes(),
                &typed_digest,
                &0_u32.to_le_bytes(),
            ],
        );
        let status = TypedInsertStatusV2 {
            database_id: [0xa1; 16],
            timeline_id: [0xa2; 16],
            txn_id: STABLE_TRANSACTION_ID,
            request_digest,
            isolation: 1,
            flags: 0,
            retention_deadline: 0,
            statement_count: 1,
            response_artifact_count: 0,
            statement_outcome_root: section_roots[5],
            response_root: [0; 32],
            aggregate_root,
        };
        let mut status_bytes = vec![0; AGGREGATE_STATUS_V2_BYTES as usize];
        encode_status_v2(&status, &mut status_bytes).expect("test STATUS2 encodes");
        let outer = gpu_db_wal::CanonicalPreApplyHeader {
            identity: gpu_db_wal::CanonicalIdentity {
                database_id: [0xa1; 16],
                cluster_id: [0xa3; 16],
                timeline_id: [0xa2; 16],
                format_epoch: 5,
            },
            leader_epoch: 6,
            commit_seq: COMMIT_SEQUENCE,
            stable_transaction_id: STABLE_TRANSACTION_ID,
            request_digest,
            isolation: gpu_db_wal::CanonicalIsolation::ReadCommitted,
            flags: OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
            catalog_before_epoch: CATALOG_EPOCH,
            catalog_after_epoch: CATALOG_EPOCH,
            catalog_before_digest: [0x33; 32],
            catalog_after_digest: [0x33; 32],
            operation_count: 2,
            table_block_count: 1,
            allocator_high_water: 0,
        };
        let outcome = gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::AbortError,
            affected_rows: 0,
            sqlstate: Some(*b"23502"),
            constraint_id: TERMINAL_CONSTRAINT_ID,
            target_digest: aggregate_root,
            returning_digest: [0; 32],
        };
        (outer, outcome, fragment, status_bytes)
    }

    fn frozen_hex(name: &str) -> Vec<u8> {
        let source = include_str!("goldens.rs");
        let prefix = format!("const {name}: &str =");
        let declaration = source
            .find(&prefix)
            .expect("checked-in fixture constant exists")
            + prefix.len();
        let start = source[declaration..]
            .find('"')
            .expect("checked-in fixture constant opens")
            + declaration
            + 1;
        let end = source[start..]
            .find("\";")
            .expect("checked-in fixture constant terminates")
            + start;
        source.as_bytes()[start..end]
            .chunks_exact(2)
            .map(|pair| {
                u8::from_str_radix(std::str::from_utf8(pair).expect("fixture ASCII"), 16)
                    .expect("fixture hex")
            })
            .collect()
    }

    fn v1_digest(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update((domain.len() as u64).to_le_bytes());
        digest.update(domain);
        for field in fields {
            digest.update((field.len() as u64).to_le_bytes());
            digest.update(field);
        }
        digest.finalize().into()
    }

    fn v2_digest(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update((domain.len() as u64).to_le_bytes());
        digest.update(domain);
        for field in fields {
            digest.update(field);
        }
        digest.finalize().into()
    }
}
