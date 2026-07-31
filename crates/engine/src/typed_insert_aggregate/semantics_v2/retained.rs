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
#[path = "retained/reservation.rs"]
mod reservation;

/// Shared identity captured from the canonical outer/S7 framing before a model can leave raw
/// proof. It owns no catalog or publication authority.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SemanticsV2BoundIdentity {
    pub(super) database_id: [u8; 16],
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
    allocator_index: SemanticsV2DurableAllocatorIndexProof<'a>,
}

/// Borrowed evidence returned by the durable allocator index after it has established marker
/// durability, publication lineage, epoch non-overlap, and checkpoint retention.  The proof is
/// a private callback, not a codec-constructable wrapper around claimed Boolean fields.
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(super) struct SemanticsV2DurableAllocatorIndexProof<'a> {
    index: &'a dyn DurableAllocatorIndexLeaseProof,
}

impl<'a> SemanticsV2DurableAllocatorIndexProof<'a> {
    fn leases(self) -> &'a [SemanticsV2RowAllocatorLeaseWitness] {
        self.index.proven_table_row_leases()
    }
}

/// Implemented only by the retained durable-index adapter.  There is deliberately no public
/// constructor or codec-visible trait: calling this method asserts that the index has already
/// proved the complete/durable/published/same-lineage/non-overlap/checkpoint predicates.
trait DurableAllocatorIndexLeaseProof {
    fn proven_table_row_leases(&self) -> &[SemanticsV2RowAllocatorLeaseWitness];
}

/// The only production constructor will be placed with the durable allocator-index adapter,
/// which is a child of this retained boundary and can prove the trait's preconditions.  Keeping
/// it private leaves the current inert codec with no way to manufacture lease evidence.
fn durable_allocator_index_proof_from_proven_index(
    index: &dyn DurableAllocatorIndexLeaseProof,
) -> SemanticsV2DurableAllocatorIndexProof<'_> {
    SemanticsV2DurableAllocatorIndexProof { index }
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

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SemanticsV2RowAllocatorLeaseWitness {
    pub(super) database_id: [u8; 16],
    pub(super) allocator_kind: u8,
    pub(super) stable_allocator_id: u64,
    pub(super) lease_epoch: u64,
    pub(super) lease_start: u64,
    pub(super) lease_end: u64,
    pub(super) prior_high_water: u64,
    pub(super) new_high_water: u64,
    pub(super) marker_system_transaction_id: u64,
    pub(super) marker_commit_sequence: u64,
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

/// All catalog, lease, and generation output equality checks have succeeded. Test-only byte
/// reencoding will be implemented exclusively on this state after the retained graph is full.
#[allow(dead_code)]
pub(super) struct FullyWitnessValidatedSemanticsV2<'a> {
    graph: RetainedSemanticsV2Graph,
    witnesses: FullyValidatedWitnesses<'a>,
}

struct CatalogAndAllocatorValidated<'a> {
    catalog: &'a SemanticsV2CatalogWitness<'a>,
    allocator_index: SemanticsV2DurableAllocatorIndexProof<'a>,
}

struct FullyValidatedWitnesses<'a> {
    catalog: &'a SemanticsV2CatalogWitness<'a>,
    allocator_index: SemanticsV2DurableAllocatorIndexProof<'a>,
    generation: SemanticsV2GenerationWitness<'a>,
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
}

impl CodecClosedSemanticsV2 {
    /// Consume the sole codec-closed owner after exact pinned-catalog and durable allocator
    /// closure. No generation result is accepted, retained, or constructed at this boundary.
    pub(super) fn validate_catalog_and_allocator<'a>(
        self,
        witness: &'a SemanticsV2CatalogAllocatorWitness<'a>,
    ) -> Result<GenerationPendingSemanticsV2<'a>, crate::EngineError> {
        catalog_validation::validate(self.graph.identity, &self.graph.graph, witness)?;
        Ok(GenerationPendingSemanticsV2 {
            graph: self.graph,
            catalog_and_allocator: CatalogAndAllocatorValidated {
                catalog: &witness.catalog,
                allocator_index: witness.allocator_index,
            },
        })
    }

    /// Test-only continuation from an actual codec-closed state. It cannot construct a raw/S7
    /// graph and does not expose the generation witness constructor or any live path.
    #[cfg(test)]
    pub(super) fn validate_catalog_and_allocator_for_test<'a>(
        self,
        witness: &'a SemanticsV2CatalogAllocatorWitness<'a>,
    ) -> Result<GenerationPendingSemanticsV2<'a>, crate::EngineError> {
        self.validate_catalog_and_allocator(witness)
    }
}

impl<'a> GenerationPendingSemanticsV2<'a> {
    /// Consume this sole post-catalog owner through the sealed, already-reserved generation
    /// builder.  The builder result is compared against the retained graph before this returns
    /// the final witness-validated typestate; no wire-derived generation witness is accepted.
    fn validate_generation_with_reserved_builder<B>(
        self,
        builder: B,
    ) -> Result<FullyWitnessValidatedSemanticsV2<'a>, crate::EngineError>
    where
        B: generation_validation::SemanticsV2ReservedGenerationBuilder<'a>,
    {
        generation_validation::validate_reserved_builder_result(self, builder)
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
    for lease in witness.allocator_index.leases() {
        if lease.database_id != identity.database_id
            || lease.allocator_kind != 1
            || lease.stable_allocator_id == 0
            || lease.stable_allocator_id == u64::MAX
            || lease.lease_epoch == 0
            || lease.lease_epoch == u64::MAX
            || lease.lease_start == 0
            || lease.lease_start == u64::MAX
            || lease.lease_end == 0
            || lease.lease_end == u64::MAX
            || lease.prior_high_water > lease.lease_start
            || lease.lease_start >= lease.lease_end
            || lease.new_high_water != lease.lease_end
            || lease.marker_system_transaction_id == 0
            || lease.marker_system_transaction_id == u64::MAX
            || lease.marker_commit_sequence == 0
            || lease.marker_commit_sequence == u64::MAX
        {
            return Err(retained_error(
                "allocator lease witness has an invalid durable shape",
            ));
        }
    }
    Ok(())
}

/// Generation output is intentionally an opaque builder-owned capability. The wire reader may
/// compare a supplied value only after `GenerationPendingSemanticsV2` has been consumed by the
/// reserved builder; no sibling module can assemble one from S7 roots.
#[allow(dead_code)]
pub(super) struct SemanticsV2GenerationWitness<'a> {
    root_descriptor_version: u16,
    database_id: [u8; 16],
    catalog_epoch: u64,
    catalog_digest: [u8; 32],
    stable_transaction_id: u64,
    commit_sequence: u64,
    initial_database_root: [u8; 32],
    generation_input_digest: [u8; 32],
    final_database_root: [u8; 32],
    tables: &'a [SemanticsV2GenerationTableWitness<'a>],
}

#[allow(dead_code)]
pub(super) struct SemanticsV2GenerationTableWitness<'a> {
    stable_table_id: u64,
    final_data_generation: u64,
    final_table_root: [u8; 32],
    final_logical_row_count: u64,
    indexes: &'a [SemanticsV2GenerationIndexWitness],
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SemanticsV2GenerationIndexWitness {
    stable_index_id: u64,
    final_index_generation: u64,
    final_index_root: [u8; 32],
}

fn retained_error(message: &str) -> crate::EngineError {
    crate::EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v2 retained: {message}"
    ))
}
