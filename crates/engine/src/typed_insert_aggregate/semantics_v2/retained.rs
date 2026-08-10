//! Move-only post-pass-zero ownership states for codec-5 semantics v2.
//!
//! This module deliberately has no WAL, recovery, apply, device, result, or publication caller.
//! Production ends at the move-only `DurableSequencePending -> GenerationPending` transition
//! after strict S1--S8 fill, local codec closure, retention authority, pinned catalog/allocator,
//! and the durable Engine sequence-index proof. Historical Q2 generation evidence is compiled
//! only for tests and cannot become a production successor.

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
#[path = "retained/retention_authority.rs"]
mod retention_authority;
#[path = "retained/sequence_validation.rs"]
pub(super) mod sequence_validation;

/// The exact three-owner handoff from a closed single-table S7 artifact into replay planning.
/// It carries the existing resident source, private sequence publication and S3 catalog record;
/// no new recovery carrier or authority is introduced here.
pub(crate) type SemanticsV2RecoverySource = (
    Option<crate::typed_insert_batch::PreparedResidentAppendSource>,
    Box<[crate::engine_commit::LiveTypedPrivateSequencePublication]>,
    Option<crate::wal_binary::BinaryTransactionRecord>,
);

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
    pub(super) request_digest: [u8; 32],
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

/// Exact test fixture row for a durable published sequence outcome. The real production bridge
/// remains responsible for borrowing the sole Engine outcome index; this narrow shape exists only
/// to hold checked-in hostile-proof backing for one callback extent.
#[cfg(test)]
pub(super) use sequence_validation::{
    SequenceOutcomeProofSabotageForTest, SequenceOutcomeSpecForTest,
};

/// Test-only hostile input for the callback-scoped durable allocator assignment adapter.
#[cfg(test)]
pub(super) use catalog_validation::AllocatorAssignmentProofSabotageForTest;

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
/// post-reservation source/image decoder, which is the sole owner allowed to add decoded S1--S8
/// fields. Keeping it private prevents a test helper from manufacturing a phase transition.
#[allow(dead_code)]
struct RetainedSemanticsV2Graph {
    identity: SemanticsV2BoundIdentity,
    graph: graph::ReservedSemanticsV2Graph,
}

/// Unique retained aggregate owner.  Its phase is private, move-only state rather than a
/// capability bag: no phase exposes raw aggregate bytes, a graph getter, or a live successor.
#[allow(dead_code)]
pub(super) struct AggregateReplayTxn<Phase> {
    graph: RetainedSemanticsV2Graph,
    phase: Phase,
}

/// Scalar identity proven by the retained v2 closure and consumed by the one recovery route.
/// It intentionally contains no aggregate body, typed vectors, row matrix, or mutable plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SemanticsV2ReplayMetadata {
    pub(crate) canonical_identity: gpu_db_wal::CanonicalIdentity,
    pub(crate) leader_epoch: u64,
    pub(crate) stable_transaction_id: u64,
    pub(crate) request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) autocommit: bool,
    pub(crate) commit_sequence: u64,
    pub(crate) catalog_epoch: u64,
    pub(crate) catalog_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) statement_count: u32,
    /// Historical one-statement codec-5 records bind this scalar into the generation ABI.
    /// Composed generic transactions bind statement identity per transition instead and use
    /// zero here so recovery cannot accidentally collapse them into one statement identity.
    pub(crate) typed_statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) stable_table_id: u64,
    pub(crate) display_oid: u32,
    pub(crate) table_schema_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) data_generation_before: u64,
    pub(crate) data_generation_after: u64,
    pub(crate) row_allocator_before: u64,
    pub(crate) row_allocator_high_water: u64,
    pub(crate) initial_logical_row_count: u64,
    pub(crate) final_logical_row_count: u64,
    pub(crate) resets_existing_rows: bool,
    /// This S7-authenticated flag distinguishes the sole first table generation from an
    /// ordinary append with a missing predecessor during fresh recovery.
    pub(crate) initial_table_absent: bool,
    pub(crate) initial_table_root: gpu_db_wal::CanonicalDigest,
    pub(crate) final_table_root: gpu_db_wal::CanonicalDigest,
    pub(crate) initial_database_root: gpu_db_wal::CanonicalDigest,
    pub(crate) final_database_root: gpu_db_wal::CanonicalDigest,
    pub(crate) image_layout_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) image_content_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) affected_rows: u64,
}

/// One durable named-index generation carried from the strictly closed S7 graph into replay.
/// These are catalog bindings and exact GPU predecessor/successor commitments only; they carry
/// no host index contents, lookup structure, apply implementation, or publication capability.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SemanticsV2ReplayIndex {
    pub(crate) stable_index_id: u64,
    pub(crate) raw_catalog_ordinal: u32,
    pub(crate) display_oid: u32,
    pub(crate) name: Box<str>,
    pub(crate) flags: u32,
    pub(crate) null_equality_policy: u8,
    pub(crate) base_generation: u64,
    pub(crate) base_root: gpu_db_wal::CanonicalDigest,
    pub(crate) final_generation: u64,
    pub(crate) final_root: gpu_db_wal::CanonicalDigest,
    pub(crate) key_columns: Box<[SemanticsV2ReplayIndexKey]>,
    /// The typed S2 statements predate this index, so its durable catalog identity can only be
    /// supplied by the already-reserved S3 CREATE INDEX target.  This is metadata for the one
    /// replay artifact, not a second catalog or replay authority.
    s3_created_on_existing_table: bool,
}

/// Ordered catalog key binding for [`SemanticsV2ReplayIndex`]. Values remain solely in the
/// retained final typed image and are resolved by the shared runtime-generation source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SemanticsV2ReplayIndexKey {
    pub(crate) key_ordinal: u32,
    pub(crate) catalog_column_ordinal: u32,
    pub(crate) stable_column_id: u32,
    pub(crate) attnum: i16,
    pub(crate) storage: [u8; 4],
    pub(crate) declared_type_oid: u32,
    pub(crate) signed_type_size: i16,
    pub(crate) column_name_digest: gpu_db_wal::CanonicalDigest,
}

/// The one move-only v2 transaction result accepted by recovery after strict physical and
/// semantic closure. Each table image can only be consumed into the existing resident append
/// source, keeping recovery on the exact same type-neutral CUDA/apply representation as live
/// INSERT while the outer owner preserves one transaction/WAL/publication boundary.
pub(crate) struct SemanticsV2ReplayArtifact {
    metadata: SemanticsV2ReplayMetadata,
    tables: Box<[SemanticsV2ReplayTableArtifact]>,
    private_sequence_publications: Box<[crate::engine_commit::LiveTypedPrivateSequencePublication]>,
    /// A private sequence may be referenced on both sides of one transaction-local rename, or
    /// have its final S2 effect before the final S3 rename. The already-decoded S3 envelope
    /// proves the same stable-OID transition in either case; it is not a second sequence or
    /// recovery authority.
    private_sequence_name_changes: Box<[PrivateSequenceNameChange]>,
    catalog_composition: Option<crate::wal_binary::BinaryTransactionRecord>,
}

struct PrivateSequenceNameChange {
    sequence_oid: u32,
    before_name: Box<str>,
    after_name: Box<str>,
}

struct PrivateSequenceReplayPublications {
    publications: Box<[crate::engine_commit::LiveTypedPrivateSequencePublication]>,
    name_changes: Box<[PrivateSequenceNameChange]>,
}

pub(crate) struct SemanticsV2ReplayTableArtifact {
    metadata: SemanticsV2ReplayMetadata,
    table_ref: u32,
    target_schema: Box<str>,
    target_name: Box<str>,
    indexes: Box<[SemanticsV2ReplayIndex]>,
    row_sources: Box<[crate::engine_transaction_delta::TypedInsertRuntimeGenerationRowSource]>,
    final_image: crate::typed_insert_batch::DecodedTypedImage,
}

impl SemanticsV2ReplayArtifact {
    pub(super) fn with_catalog_composition(
        mut self,
        catalog_composition: Option<crate::wal_binary::BinaryTransactionRecord>,
    ) -> Result<Self, crate::EngineError> {
        // S2 normally carries the final effective sequence name. A terminal S3 RENAME has no
        // following default-bearing S2 record, so recover that presentation binding from its
        // exact stable-OID S3 identity before the common name-change proof below. This only
        // retitles an existing S2-derived private publication; S3 still owns the catalog state,
        // and there is no separate sequence/recovery/application path.
        let mut name_changes = self.private_sequence_name_changes.into_vec();
        if let Some(record) = catalog_composition.as_ref() {
            for publication in self.private_sequence_publications.iter_mut() {
                let matching_targets = record
                    .sequence_lifecycle_operations
                    .iter()
                    .flat_map(|operation| operation.targets.iter())
                    .filter(|target| {
                        target.before_name == publication.name.as_ref()
                            && target.after_name.as_deref().is_some_and(|after| {
                                after != target.before_name && !after.is_empty()
                            })
                            && target
                                .target_before
                                .as_ref()
                                .is_some_and(|before| before.oid == publication.sequence_oid)
                            && target
                                .target_after
                                .as_ref()
                                .is_some_and(|after| after.oid == publication.sequence_oid)
                    })
                    .collect::<Vec<_>>();
                let Some(target) = matching_targets.first() else {
                    continue;
                };
                if matching_targets.len() != 1 {
                    return Err(replay_error(
                        "private sequence publication has ambiguous S3 stable-OID rename",
                    ));
                }
                let after_name = target
                    .after_name
                    .as_deref()
                    .expect("filter requires an after name");
                match name_changes
                    .iter()
                    .find(|change| change.sequence_oid == publication.sequence_oid)
                {
                    Some(change)
                        if change.before_name.as_ref() != publication.name.as_ref()
                            || change.after_name.as_ref() != after_name =>
                    {
                        return Err(replay_error(
                            "private sequence publication has conflicting S2 and S3 rename bindings",
                        ));
                    }
                    Some(_) => {}
                    None => name_changes.push(PrivateSequenceNameChange {
                        sequence_oid: publication.sequence_oid,
                        before_name: publication.name.clone(),
                        after_name: after_name.into(),
                    }),
                }
                publication.name = after_name.into();
            }
        }
        self.private_sequence_name_changes = name_changes.into_boxed_slice();
        for change in self.private_sequence_name_changes.iter() {
            let proven = catalog_composition.as_ref().is_some_and(|record| {
                record
                    .sequence_lifecycle_operations
                    .iter()
                    .any(|operation| {
                        operation.targets.iter().any(|target| {
                            target.before_name == change.before_name.as_ref()
                                && target.after_name.as_deref() == Some(change.after_name.as_ref())
                                && target
                                    .target_before
                                    .as_ref()
                                    .is_some_and(|before| before.oid == change.sequence_oid)
                                && target
                                    .target_after
                                    .as_ref()
                                    .is_some_and(|after| after.oid == change.sequence_oid)
                        })
                    })
            });
            if !proven {
                return Err(replay_error(
                    "private sequence binding rename is not closed by the S3 stable-OID transition",
                ));
            }
        }
        for table in self.tables.iter_mut() {
            if table.metadata.initial_table_absent {
                continue;
            }
            for index in table
                .indexes
                .iter_mut()
                .filter(|index| index.base_generation == 0 && index.base_root == [0; 32])
            {
                let matching_targets = catalog_composition
                    .as_ref()
                    .into_iter()
                    .flat_map(|record| record.index_lifecycle_operations.iter())
                    .flat_map(|operation| operation.targets.iter())
                    .filter(|target| {
                        target.index_before.is_none()
                            && target.table_before.as_ref().is_some_and(|table_before| {
                                table_before.oid == table.metadata.display_oid
                            })
                            && target.table_after.as_ref().is_some_and(|table_after| {
                                table_after.oid == table.metadata.display_oid
                            })
                            && target.index_after.as_ref().is_some_and(|index_after| {
                                u64::from(index_after.oid) == index.stable_index_id
                                    && index_after.table_oid == table.metadata.display_oid
                            })
                    })
                    .collect::<Vec<_>>();
                let [target] = matching_targets.as_slice() else {
                    return Err(replay_error(
                        "zero-based existing-table index is not closed by the S3 CREATE INDEX identity",
                    ));
                };
                let Some(after_name) = target.after_name.as_deref() else {
                    return Err(replay_error(
                        "zero-based existing-table index S3 identity has no final name",
                    ));
                };
                if index.s3_created_on_existing_table {
                    if !index.name.is_empty() {
                        return Err(replay_error(
                            "S3-created existing-table index retained an unexpected S2 name",
                        ));
                    }
                    index.name = after_name.into();
                } else if after_name != index.name.as_ref() {
                    return Err(replay_error(
                        "zero-based existing-table index S3 identity differs from its retained S2 name",
                    ));
                }
            }
        }
        self.catalog_composition = catalog_composition;
        Ok(self)
    }

    pub(crate) fn metadata(&self) -> SemanticsV2ReplayMetadata {
        self.metadata
    }

    pub(crate) fn table_count(&self) -> usize {
        self.tables.len()
    }

    pub(crate) fn target_table_name(&self) -> &str {
        self.tables
            .first()
            .expect("closed replay artifact has at least one table")
            .target_table_name()
    }

    pub(crate) fn indexes(&self) -> &[SemanticsV2ReplayIndex] {
        self.tables
            .first()
            .expect("closed replay artifact has at least one table")
            .indexes()
    }

    pub(crate) fn row_sources(
        &self,
    ) -> &[crate::engine_transaction_delta::TypedInsertRuntimeGenerationRowSource] {
        self.tables
            .first()
            .expect("closed replay artifact has at least one table")
            .row_sources()
    }

    pub(crate) fn private_sequence_publications(
        &self,
    ) -> &[crate::engine_commit::LiveTypedPrivateSequencePublication] {
        &self.private_sequence_publications
    }

    pub(crate) fn catalog_composition(
        &self,
    ) -> Option<&crate::wal_binary::BinaryTransactionRecord> {
        self.catalog_composition.as_ref()
    }

    pub(crate) fn into_recovery_source(
        self,
        physical_table: &crate::RelationalTable,
        final_table: &crate::RelationalTable,
    ) -> Result<SemanticsV2RecoverySource, crate::EngineError> {
        if self.tables.len() != 1 {
            return Err(replay_error(
                "plural replay artifact cannot be collapsed into one recovery source",
            ));
        }
        let source = self
            .tables
            .into_vec()
            .pop()
            .expect("one checked replay table")
            .into_recovery_source(physical_table, final_table)?;
        Ok((
            source,
            self.private_sequence_publications,
            self.catalog_composition,
        ))
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        Box<[SemanticsV2ReplayTableArtifact]>,
        Box<[crate::engine_commit::LiveTypedPrivateSequencePublication]>,
        Option<crate::wal_binary::BinaryTransactionRecord>,
    ) {
        (
            self.tables,
            self.private_sequence_publications,
            self.catalog_composition,
        )
    }
}

impl SemanticsV2ReplayTableArtifact {
    pub(crate) fn metadata(&self) -> SemanticsV2ReplayMetadata {
        self.metadata
    }

    pub(crate) fn table_ref(&self) -> u32 {
        self.table_ref
    }

    pub(crate) fn target_table_name(&self) -> &str {
        self.target_name.as_ref()
    }

    pub(crate) fn indexes(&self) -> &[SemanticsV2ReplayIndex] {
        &self.indexes
    }

    pub(crate) fn row_sources(
        &self,
    ) -> &[crate::engine_transaction_delta::TypedInsertRuntimeGenerationRowSource] {
        &self.row_sources
    }

    /// Bind the closed image to the current catalog table before moving its sole vector owners
    /// into the shared resident append source.  This is deliberately not a row-value conversion
    /// and cannot manufacture a second replay representation.
    pub(crate) fn into_recovery_source(
        self,
        physical_table: &crate::RelationalTable,
        final_table: &crate::RelationalTable,
    ) -> Result<Option<crate::typed_insert_batch::PreparedResidentAppendSource>, crate::EngineError>
    {
        let metadata = self.metadata;
        let image_facts = self.final_image.facts();
        let surviving_rows = u64::try_from(self.row_sources.len())
            .map_err(|_| replay_error("recovery survivor count exceeds u64"))?;
        let final_schema_digest = crate::engine_transaction_reset::table_schema_digest(final_table)
            .map_err(|error| {
                replay_error(&format!("catalog table schema digest failed: {error}"))
            })?;
        let physical_schema_digest = crate::engine_transaction_reset::table_schema_digest(
            physical_table,
        )
        .map_err(|error| replay_error(&format!("physical table schema digest failed: {error}")))?;
        let expected_final_logical_row_count = if metadata.resets_existing_rows {
            surviving_rows
        } else {
            metadata
                .initial_logical_row_count
                .checked_add(surviving_rows)
                .ok_or_else(|| replay_error("recovery logical row count overflows"))?
        };
        if final_table.schema.as_str() != self.target_schema.as_ref()
            || final_table.name.as_str() != self.target_name.as_ref()
            || final_table.stable_table_id != metadata.stable_table_id
            || final_table.oid != metadata.display_oid
            || final_schema_digest != metadata.table_schema_digest
            || physical_table.schema != final_table.schema
            || physical_table.name != final_table.name
            || physical_table.stable_table_id != final_table.stable_table_id
            || physical_table.oid != final_table.oid
            || physical_table.columns.len() != final_table.columns.len()
            || (!metadata.initial_table_absent && metadata.data_generation_before == 0)
            || (metadata.initial_table_absent
                && (metadata.data_generation_before != 0
                    || metadata.initial_logical_row_count != 0
                    || metadata.resets_existing_rows))
            || metadata.row_allocator_before == 0
            || metadata.row_allocator_high_water
                != metadata
                    .row_allocator_before
                    .checked_add(metadata.affected_rows)
                    .ok_or_else(|| replay_error("row allocator range overflows"))?
            || metadata.final_logical_row_count != expected_final_logical_row_count
            || (!metadata.initial_table_absent && metadata.initial_table_root == [0; 32])
            || (metadata.initial_table_absent && metadata.initial_table_root != [0; 32])
            || metadata.final_table_root == [0; 32]
            || metadata.initial_database_root == [0; 32]
            || metadata.final_database_root == [0; 32]
            || surviving_rows > metadata.affected_rows
            || (surviving_rows != 0
                && (metadata.data_generation_after != metadata.commit_sequence
                    || metadata.data_generation_after <= metadata.data_generation_before
                    || metadata.initial_table_root == metadata.final_table_root
                    || metadata.initial_database_root == metadata.final_database_root))
            || (surviving_rows == 0
                && (metadata.data_generation_after != metadata.data_generation_before
                    || metadata.initial_table_root != metadata.final_table_root))
            || image_facts.role != crate::typed_insert_batch::TypedImageRole::FinalTableImage
            || u64::from(image_facts.rows) != surviving_rows
            || usize::try_from(image_facts.columns).ok() != Some(final_table.columns.len())
            || image_facts.layout_digest != metadata.image_layout_digest
        {
            return Err(replay_error(
                "closed v2 aggregate does not bind the current recovery catalog/table generation",
            ));
        }
        if surviving_rows == 0 {
            if !self.indexes.is_empty() {
                return Err(replay_error(
                    "neutral closed v2 table retained an index publication authority",
                ));
            }
            return Ok(None);
        }
        if self.indexes.len() != final_table.indexes.len()
            || self.indexes.iter().any(|retained| {
                let Some(catalog) = final_table
                    .indexes
                    .get(retained.raw_catalog_ordinal as usize)
                else {
                    return true;
                };
                catalog.oid == 0
                    || u64::from(catalog.oid) != retained.stable_index_id
                    || catalog.oid != retained.display_oid
                    || catalog.name.as_str() != retained.name.as_ref()
                    || retained.flags
                        != (u32::from(catalog.unique)
                            | (u32::from(catalog.primary_key) << 1)
                            | (u32::from(catalog.unique_constraint) << 2)
                            | (1 << 3))
                    || retained.null_equality_policy != 1
                    || retained.final_generation != metadata.data_generation_after
                    || catalog.key_columns.len() != retained.key_columns.len()
                    || retained
                        .key_columns
                        .iter()
                        .enumerate()
                        .any(|(ordinal, key)| {
                            let Some(column) =
                                final_table.columns.get(key.catalog_column_ordinal as usize)
                            else {
                                return true;
                            };
                            key.key_ordinal as usize != ordinal
                                || catalog.key_columns.get(ordinal).map(String::as_str)
                                    != Some(column.name.as_str())
                                || key.stable_column_id != column.id
                                || key.attnum != column.attnum
                                || key.storage
                                    != crate::typed_insert_batch::typed_image_sql_storage(column.ty)
                                || key.declared_type_oid != column.type_oid
                                || key.signed_type_size != column.type_size
                                || crate::typed_insert_aggregate::write001_identifier_digest(
                                    &column.name,
                                )
                                .ok()
                                    != Some(key.column_name_digest)
                        })
            })
        {
            return Err(replay_error(
                "closed v2 aggregate index bindings differ from the recovery catalog",
            ));
        }
        crate::typed_insert_batch::PreparedResidentAppendSource::from_decoded_final_table_image(
            self.final_image,
            physical_table,
            physical_schema_digest,
            metadata.data_generation_before,
        )
        .map(Some)
    }
}

/// Strict structural decode and typed fill have completed, but witness-free codec closure has
/// not yet recomputed the retained relations.
#[allow(dead_code)]
pub(super) struct CodecQuarantined(PrivateSeal);

/// The codec-only retained closure has completed.  The next sealed phase authenticates the one
/// durable retention claim (or the sealed historical no-retention proof) before catalog or
/// allocator validation becomes available.
#[allow(dead_code)]
pub(super) struct RetentionAuthorityPending(PrivateSeal);

/// The sole retention authority has been authenticated and is carried opaquely to the later
/// catalog/allocator phase.  This state intentionally has no catalog, allocator, sequence, GPU,
/// WAL, recovery, apply, result, or publication operation yet.
#[allow(dead_code)]
pub(super) struct CatalogAllocatorPending<'a> {
    authority: retention_authority::ValidatedRetentionAuthority<'a>,
    seal: PrivateSeal,
}

/// The retained graph has exactly matched the one pinned catalog plus concrete durable
/// allocator-lease authority.  The retained retention authority remains opaque and accompanies
/// these borrowed witnesses into the later durable-sequence proof.  No sequence, GPU, WAL,
/// recovery, apply, result, or publication successor is available in this checkpoint.
#[allow(dead_code)]
pub(super) struct DurableSequencePending<'retention, 'catalog> {
    authority: retention_authority::ValidatedRetentionAuthority<'retention>,
    catalog: SemanticsV2CatalogWitness<'catalog>,
    allocator_index: catalog_validation::SemanticsV2DurableAllocatorIndexProof<'catalog>,
    allocator_assignment: catalog_validation::ValidatedAllocatorAssignment<'catalog>,
    seal: PrivateSeal,
}

/// Every retained published S5 effect has been bound to its exact durable, complete, published,
/// retained, same-lineage `Default` transition strictly before the parent.  This owner carries
/// the retention, catalog, allocator, and sequence proofs opaquely but offers no compiler, GPU,
/// WAL, recovery, apply, result, or publication successor in this checkpoint.
#[allow(dead_code)]
pub(super) struct GenerationPending<'retention, 'catalog, 'sequence> {
    authority: retention_authority::ValidatedRetentionAuthority<'retention>,
    catalog: SemanticsV2CatalogWitness<'catalog>,
    allocator_index: catalog_validation::SemanticsV2DurableAllocatorIndexProof<'catalog>,
    allocator_assignment: catalog_validation::ValidatedAllocatorAssignment<'catalog>,
    sequence_index: sequence_validation::SemanticsV2DurableSequenceOutcomeIndexProof<'sequence>,
    seal: PrivateSeal,
}

/// The generic generation builder has independently reproduced every S7 generation output.
/// The borrowed retention/catalog/allocator/sequence authorities remain pinned until the caller
/// consumes this owner into the single apply/publication handoff; no codec or host digest can
/// manufacture this phase.
#[allow(dead_code)]
pub(super) struct GenerationValidated<'retention, 'catalog, 'sequence, C> {
    authority: retention_authority::ValidatedRetentionAuthority<'retention>,
    catalog: SemanticsV2CatalogWitness<'catalog>,
    allocator_index: catalog_validation::SemanticsV2DurableAllocatorIndexProof<'catalog>,
    allocator_assignment: catalog_validation::ValidatedAllocatorAssignment<'catalog>,
    sequence_index: sequence_validation::SemanticsV2DurableSequenceOutcomeIndexProof<'sequence>,
    generation: generation_validation::OwnedGenerationResult<C>,
    seal: PrivateSeal,
}

/// Only this module can construct a phase marker, keeping the move-only graph transition sealed.
struct PrivateSeal;

/// Q2's historical empty-S8 evidence bridge.  It is test-only and can be formed only by
/// consuming an actual `RetentionAuthorityPending` owner after proving canonical empty S8.
#[cfg(test)]
pub(super) struct Q2CodecClosedSemanticsV2 {
    graph: RetainedSemanticsV2Graph,
}

/// Catalog and durable allocator closure has succeeded. Only this state may be handed to the
/// external reserved generation builder; it is still not publication eligible.
#[cfg(test)]
#[allow(dead_code)]
pub(super) struct GenerationPendingSemanticsV2<'a> {
    graph: RetainedSemanticsV2Graph,
    catalog_and_allocator: CatalogAndAllocatorValidated<'a>,
}

/// All catalog, lease, and generation output equality checks have succeeded.  The catalog and
/// allocator borrows end at this transition: a future reencoder can only receive the retained
/// graph plus the builder's opaque, fully-owned generation result.
#[cfg(test)]
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

#[cfg(test)]
struct CatalogAndAllocatorValidated<'a> {
    catalog: SemanticsV2CatalogWitness<'a>,
    allocator_index: catalog_validation::SemanticsV2DurableAllocatorIndexProof<'a>,
    allocator_assignment: catalog_validation::ValidatedAllocatorAssignment<'a>,
}

/// Seal the complete, exact post-reservation graph into the context-free quarantine state.
///
/// The strict S1--S8 fill is the sole caller: it consumes `into_exact_graph()` immediately
/// before this transition, so a partially filled owner cannot become witness-checkable.
fn quarantine_after_strict_fill(
    identity: SemanticsV2BoundIdentity,
    graph: graph::ReservedSemanticsV2Graph,
) -> AggregateReplayTxn<CodecQuarantined> {
    AggregateReplayTxn {
        graph: RetainedSemanticsV2Graph { identity, graph },
        phase: CodecQuarantined(PrivateSeal),
    }
}

/// Private retained-boundary entry for a successful allocation-free proof.  The facade may pass
/// it a bounded framing, but only this module's child can consume the exact graph owner or seal
/// a quarantine state.
pub(super) fn fill_after_pass_zero(
    framing: &super::super::codec::DecodedAggregateFraming<'_>,
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    pass_zero: super::pass_zero::SemanticsV2PassZero,
) -> Result<AggregateReplayTxn<CodecQuarantined>, crate::EngineError> {
    fill::fill_after_pass_zero(framing, outer, pass_zero)
}

#[cfg(test)]
pub(super) fn fail_source_copy_at_for_test<T>(attempt: u64, operation: impl FnOnce() -> T) -> T {
    fill::fail_copy_at_for_test(attempt, operation)
}

#[cfg(test)]
pub(super) fn observe_source_copy_attempts_for_test<T>(operation: impl FnOnce() -> T) -> (T, u64) {
    fill::observe_copy_attempts_for_test(operation)
}

impl AggregateReplayTxn<CodecQuarantined> {
    /// Consume structural ownership only after every witness-free S1--S8 relation has been
    /// recomputed from strict S2/image owners. A failed close drops the raw quarantine, so no
    /// caller can retain a partially closed or externally advanceable graph.
    pub(super) fn close_codec(
        self,
        catalog_composition: Option<&crate::wal_binary::BinaryTransactionRecord>,
    ) -> Result<AggregateReplayTxn<RetentionAuthorityPending>, crate::EngineError> {
        codec_closure::validate(self.graph.identity, &self.graph.graph, catalog_composition)?;
        Ok(AggregateReplayTxn {
            graph: self.graph,
            phase: RetentionAuthorityPending(PrivateSeal),
        })
    }

    /// Q2's test-only adapter consumes the sole production shell and accepts only canonical
    /// empty S8.  Nonempty retained responses cannot enter the historical Q2 chain.
    #[cfg(test)]
    pub(super) fn close_codec_for_test(
        self,
    ) -> Result<Q2CodecClosedSemanticsV2, crate::EngineError> {
        let pending = self.close_codec(None)?;
        if !matches!(
            &pending.graph.graph.response,
            graph::RetainedResponseEnvelope::Empty(_)
        ) {
            return Err(retained_error(
                "Q2 codec-closed adapter requires canonical empty S8",
            ));
        }
        Ok(Q2CodecClosedSemanticsV2 {
            graph: pending.graph,
        })
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

impl AggregateReplayTxn<RetentionAuthorityPending> {
    /// Consume the codec-closed owner only after the authenticated claim/status index proves the
    /// immutable retention intent, or a sealed historical proof establishes no retention.  This
    /// phase is deliberately before catalog, allocator, durable sequence, GPU replay, WAL, apply,
    /// and publication work.
    pub(super) fn validate_retention_authority<'a>(
        self,
        input: retention_authority::RetentionAuthorityInput<'a>,
    ) -> Result<AggregateReplayTxn<CatalogAllocatorPending<'a>>, crate::EngineError> {
        let authority =
            retention_authority::validate(self.graph.identity, &self.graph.graph, input)?;
        Ok(AggregateReplayTxn {
            graph: self.graph,
            phase: CatalogAllocatorPending {
                authority,
                seal: PrivateSeal,
            },
        })
    }

    /// Consume the closed retained graph into the first production recovery artifact.  The
    /// current vertical is one table generation composed from one or more INSERT statements;
    /// indexed, published-sequence, and logical-RETURNING breadth remains one statement until
    /// those device closures are composed too. Published sequence receipts and logical
    /// RETURNING projections have already been closed against S2/S4/S5/S6/S7; neither needs a
    /// second replay action because sequence transitions are independently ordered durable
    /// records and RETURNING is not table state. Transaction mode is already authenticated by
    /// the aggregate flag and may be claimed autocommit or explicit COMMIT. Unsupported v2
    /// breadth stays rejected rather than falling back through the legacy engine-operation
    /// decoder.
    pub(super) fn into_replay_artifact(
        self,
        catalog_composition: Option<&crate::wal_binary::BinaryTransactionRecord>,
    ) -> Result<SemanticsV2ReplayArtifact, crate::EngineError> {
        let AggregateReplayTxn { graph, phase: _ } = self;
        let RetainedSemanticsV2Graph { identity, graph } = graph;
        if graph.tables.len() > 1 {
            return into_plural_replay_artifact(identity, graph);
        }
        let private_sequence_publications = private_sequence_publications_from_graph(&graph)?;
        let statement_count = graph.statements.len();
        if statement_count == 0
            || graph.records.len() != statement_count
            || graph.outcomes.len() != statement_count
            || graph.tables.len() != 1
            || graph.images.len() != 1
            || graph.resolutions.len() != statement_count
            || !matches!(graph.response, graph::RetainedResponseEnvelope::Empty(_))
        {
            return Err(replay_error(
                "v2 aggregate is outside the one-table generic INSERT recovery vertical",
            ));
        }
        let record = graph
            .records
            .first()
            .expect("nonempty checked retained records");
        let record_facts = record.facts();
        let target = record.target_identity();
        let target_schema: Box<str> = target.schema.into();
        let target_name: Box<str> = target.name.into();
        let table = graph.tables.first().expect("one checked retained table");
        let total_rows = graph.records.iter().try_fold(0_u32, |total, record| {
            total
                .checked_add(record.facts().row_count)
                .ok_or_else(|| replay_error("v2 aggregate row count overflows"))
        })?;
        let surviving_rows = u64::from(table.transition_count);
        let expected_final_logical_row_count = if table.resets_existing_rows {
            surviving_rows
        } else {
            table
                .initial_logical_row_count
                .checked_add(surviving_rows)
                .ok_or_else(|| replay_error("v2 aggregate logical row count overflows"))?
        };
        let statements_are_exact = graph
            .records
            .iter()
            .zip(&graph.statements)
            .zip(&graph.resolutions)
            .zip(&graph.outcomes)
            .enumerate()
            .all(|(ordinal, (((record, statement), resolution), outcome))| {
                let facts = record.facts();
                let target = record.target_identity();
                facts.statement_ordinal.as_u32() as usize == ordinal
                    && statement.statement_ordinal as usize == ordinal
                    && resolution.statement_ordinal as usize == ordinal
                    && outcome.statement_ordinal as usize == ordinal
                    && facts.typed_statement_digest == statement.typed_statement_digest
                    && facts.typed_statement_digest == resolution.typed_statement_digest
                    && facts.typed_statement_digest == outcome.typed_statement_digest
                    && facts.row_count != 0
                    && facts.column_count == record_facts.column_count
                    && target.schema == target_schema.as_ref()
                    && target.name == target_name.as_ref()
                    && facts.target.oid == table.display_oid
                    && resolution.table_ref == 0
                    && resolution.affected_row_count == u64::from(facts.row_count)
                    && outcome.outcome.affected_rows == u64::from(facts.row_count)
                    && outcome.outcome.kind == gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
            });
        // Codec closure has already proved the complete target-index and FK supporting-index
        // descriptor/effect graph. Replay publishes only target-owned generations; immutable
        // parent descriptors remain validation evidence and must not become apply authorities.
        let target_indexes = graph
            .indexes
            .iter()
            .filter(|index| index.owner_table_ref == 0)
            .collect::<Vec<_>>();
        let indexed_shape = target_indexes
            .iter()
            .all(|index| (1..=32).contains(&index.key_count));
        if !statements_are_exact
            || total_rows == 0
            || record_facts.column_count == 0
            || record_facts.target.oid != table.display_oid
            || graph.records.last().is_none_or(|record| {
                record.facts().target.schema_digest != table.schema_digest
                    && !codec_closure::terminal_s3_sequence_rename_closes_table_schema(
                        table,
                        Some(record),
                        catalog_composition,
                    )
            })
            || table.disposition_count != total_rows
            || table.transition_count > total_rows
            || (table.transition_count != 0
                && (table.data_generation_after != identity.commit_sequence
                    || table.data_generation_after <= table.data_generation_before
                    || table.final_table_root == table.initial_table_root
                    || graph.header.final_database_root == identity.initial_database_root))
            || (table.transition_count == 0
                && (table.data_generation_after != table.data_generation_before
                    || table.final_table_root != table.initial_table_root
                    || graph.header.final_database_root != identity.initial_database_root))
            || table.catalog_epoch != identity.catalog_epoch
            || table.final_logical_row_count != expected_final_logical_row_count
            || graph.header.final_database_root == [0; 32]
            || !indexed_shape
        {
            return Err(replay_error(
                "closed v2 aggregate retained facts do not form one generic INSERT replay artifact",
            ));
        }
        let row_sources = graph
            .transitions
            .iter()
            .enumerate()
            .map(|(row, transition)| {
                let row = u32::try_from(row)
                    .map_err(|_| replay_error("v2 transition ordinal exceeds u32"))?;
                let source_record = graph
                    .records
                    .get(transition.source_statement_ordinal as usize)
                    .ok_or_else(|| replay_error("v2 transition source statement is absent"))?;
                let source_disposition = graph
                    .dispositions
                    .get(transition.source_disposition_ref as usize)
                    .ok_or_else(|| replay_error("v2 transition source disposition is absent"))?;
                if transition.transition_ref != row
                    || transition.table_ref != 0
                    || transition.stable_row_id
                        != table
                            .row_allocator_before
                            .checked_add(u64::from(row))
                            .ok_or_else(|| replay_error("v2 transition row identity overflows"))?
                    || transition.image_ref != 0
                    || transition.image_row_ordinal != row
                    || source_disposition.disposition != 1
                    || source_disposition.transition_ref != transition.transition_ref
                    || source_disposition.table_ref != transition.table_ref
                    || source_disposition.stable_row_id != transition.stable_row_id
                    || source_disposition.statement_ordinal != transition.source_statement_ordinal
                    || source_disposition.source_row_ordinal != transition.source_row_ordinal
                    || source_disposition.typed_statement_digest
                        != transition.typed_statement_digest
                    || (transition.final_writer_statement_digest == [0; 32]
                        && transition.final_writer_statement_ordinal
                            != transition.source_statement_ordinal)
                    || (transition.final_writer_statement_digest != [0; 32]
                        && transition.final_writer_statement_ordinal
                            <= transition.source_statement_ordinal)
                    || transition.source_row_ordinal >= source_record.facts().row_count
                    || transition.typed_statement_digest
                        != source_record.facts().typed_statement_digest
                {
                    return Err(replay_error(
                        "v2 transition does not retain an exact composed statement-row source",
                    ));
                }
                Ok(
                    crate::engine_transaction_delta::TypedInsertRuntimeGenerationRowSource {
                        stable_row_id: transition.stable_row_id,
                        statement_ordinal: transition.source_statement_ordinal,
                        source_row_ordinal: transition.source_row_ordinal,
                    },
                )
            })
            .collect::<Result<Vec<_>, crate::EngineError>>()?
            .into_boxed_slice();
        if row_sources.len() != table.transition_count as usize {
            return Err(replay_error(
                "v2 transition sources do not exhaust the combined final image",
            ));
        }
        let indexes = replay_indexes_for_table(&identity, &graph, table, 0, record)?;
        let final_image = graph
            .images
            .into_iter()
            .next()
            .expect("one checked retained image");
        let metadata = SemanticsV2ReplayMetadata {
            canonical_identity: gpu_db_wal::CanonicalIdentity {
                database_id: identity.database_id,
                cluster_id: identity.cluster_id,
                timeline_id: identity.timeline_id,
                format_epoch: identity.format_epoch,
            },
            leader_epoch: identity.leader_epoch,
            stable_transaction_id: identity.stable_transaction_id,
            request_digest: identity.request_digest,
            autocommit: identity.autocommit,
            commit_sequence: identity.commit_sequence,
            catalog_epoch: identity.catalog_epoch,
            catalog_digest: identity.catalog_digest,
            statement_count: u32::try_from(statement_count)
                .map_err(|_| replay_error("v2 statement count exceeds u32"))?,
            typed_statement_digest: if statement_count == 1 {
                record_facts.typed_statement_digest
            } else {
                [0; 32]
            },
            stable_table_id: table.stable_table_id,
            display_oid: table.display_oid,
            table_schema_digest: table.schema_digest,
            data_generation_before: table.data_generation_before,
            data_generation_after: table.data_generation_after,
            row_allocator_before: table.row_allocator_before,
            row_allocator_high_water: table.row_allocator_high_water,
            initial_logical_row_count: table.initial_logical_row_count,
            final_logical_row_count: table.final_logical_row_count,
            resets_existing_rows: table.resets_existing_rows,
            initial_table_absent: table.initial_table_absent,
            initial_table_root: table.initial_table_root,
            final_table_root: table.final_table_root,
            initial_database_root: identity.initial_database_root,
            final_database_root: graph.header.final_database_root,
            image_layout_digest: table.image_layout_digest,
            image_content_digest: table.image_content_digest,
            affected_rows: u64::from(total_rows),
        };
        Ok(SemanticsV2ReplayArtifact {
            metadata,
            tables: vec![SemanticsV2ReplayTableArtifact {
                metadata,
                table_ref: 0,
                target_schema,
                target_name,
                indexes,
                row_sources,
                final_image,
            }]
            .into_boxed_slice(),
            private_sequence_publications: private_sequence_publications.publications,
            private_sequence_name_changes: private_sequence_publications.name_changes,
            catalog_composition: None,
        })
    }
}

fn replay_indexes_for_table(
    identity: &SemanticsV2BoundIdentity,
    graph: &graph::ReservedSemanticsV2Graph,
    table: &graph::RetainedTable,
    table_ref: u32,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
) -> Result<Box<[SemanticsV2ReplayIndex]>, crate::EngineError> {
    let record_facts = record.facts();
    let target_indexes = graph
        .indexes
        .iter()
        .filter(|index| index.owner_table_ref == table_ref)
        .collect::<Vec<_>>();
    // The S7 table block is the existing authority for first-generation absence. A named
    // index created with that table therefore has the paired zero predecessor consumed by the
    // same replay action; published-table indexes still require an authenticated nonzero base.
    let index_predecessor_is_absent = table.initial_table_absent;
    target_indexes
        .into_iter()
        .enumerate()
        .map(|(raw_catalog_ordinal, index)| {
            let source = record
                .indexes()
                .find(|source| source.raw_ordinal == index.raw_catalog_ordinal);
            // Rows inserted before a transaction-local CREATE INDEX have no S2 binding for
            // that future index. S7 proves its paired-zero generation and S3 later supplies
            // the exact catalog identity; all other index descriptors remain S2-backed.
            let s3_created_on_existing_table = source.is_none()
                && !index_predecessor_is_absent
                && index.base_index_generation == 0
                && index.base_index_root == [0; 32];
            if source.is_none() && !s3_created_on_existing_table {
                #[cfg(feature = "probe-timing")]
                eprintln!(
                    "[probe] codec5_replay_index_source_missing oid={} raw={} record_indexes={:?}",
                    index.display_oid,
                    index.raw_catalog_ordinal,
                    record
                        .indexes()
                        .map(|source| (source.oid, source.raw_ordinal))
                        .collect::<Vec<_>>(),
                );
                return Err(replay_error(
                    "closed v2 aggregate index descriptor has no sealed S2 source",
                ));
            }
            let keys = graph
                .index_key_columns
                .iter()
                .filter(|key| key.index_ref == index.index_ref)
                .map(|key| SemanticsV2ReplayIndexKey {
                    key_ordinal: key.key_ordinal,
                    catalog_column_ordinal: key.owner_catalog_column_ordinal,
                    stable_column_id: key.stable_column_id,
                    attnum: key.attnum,
                    storage: key.storage,
                    declared_type_oid: key.declared_type_oid,
                    signed_type_size: key.signed_type_size,
                    column_name_digest: key.column_name_digest,
                })
                .collect::<Vec<_>>();
            let expected_flags = source.map_or(index.flags, |source| {
                u32::from(source.unique)
                    | (u32::from(source.primary_key) << 1)
                    | (u32::from(source.unique_constraint) << 2)
                    | (1 << 3)
            });
            let constraint_backed = source.map_or_else(
                || index.flags & ((1 << 1) | (1 << 2)) != 0,
                |source| source.primary_key || source.unique_constraint,
            );
            if index.owner_table_ref != table_ref
                || usize::try_from(index.raw_catalog_ordinal).ok() != Some(raw_catalog_ordinal)
                || index.stable_index_id == 0
                || index.display_oid == 0
                || index.stable_index_id != u64::from(index.display_oid)
                || index.stable_constraint_id
                    != if constraint_backed {
                        index.stable_index_id
                    } else {
                        u64::MAX
                    }
                || index.constraint_display_oid
                    != if constraint_backed {
                        index.display_oid
                    } else {
                        0
                    }
                || index.constraint_name_digest
                    != if constraint_backed {
                        index.index_name_digest
                    } else {
                        [0; 32]
                    }
                || index.flags != expected_flags
                || source.is_none()
                    && (index.flags & !0b1111 != 0
                        || index.flags & (1 << 3) == 0
                        || index.flags & ((1 << 1) | (1 << 2)) != 0 && index.flags & 1 == 0)
                || index.null_equality_policy != 1
                || index.key_count == 0
                || index.key_count > 32
                || index.catalog_epoch != identity.catalog_epoch
                || index.owner_stable_table_id != table.stable_table_id
                || index.owner_display_oid != table.display_oid
                || index.owner_schema_digest != table.schema_digest
                || index.owner_table_base_root != table.initial_table_root
                || (index.base_index_root == [0; 32]
                    && !index_predecessor_is_absent
                    && index.base_index_generation != 0)
                || index.final_index_root == [0; 32]
                || index.base_index_root == index.final_index_root
                || index.owner_data_generation != table.data_generation_before
                || (index.base_index_generation == 0
                    && !index_predecessor_is_absent
                    && index.base_index_root != [0; 32])
                || index.base_index_generation > table.data_generation_before
                || index.final_index_generation != table.data_generation_after
                || keys.len() != index.key_count as usize
                || keys.iter().enumerate().any(|(ordinal, key)| {
                    key.key_ordinal as usize != ordinal
                        || key.catalog_column_ordinal as usize >= record_facts.column_count as usize
                })
            {
                return Err(replay_error(
                    "closed v2 aggregate named-index generation exceeds the recovery vertical",
                ));
            }
            Ok(SemanticsV2ReplayIndex {
                stable_index_id: index.stable_index_id,
                raw_catalog_ordinal: index.raw_catalog_ordinal,
                display_oid: index.display_oid,
                name: source.map_or_else(|| Box::from(""), |source| source.name.into()),
                flags: index.flags,
                null_equality_policy: index.null_equality_policy,
                base_generation: index.base_index_generation,
                base_root: index.base_index_root,
                final_generation: index.final_index_generation,
                final_root: index.final_index_root,
                key_columns: keys.into_boxed_slice(),
                s3_created_on_existing_table,
            })
        })
        .collect::<Result<Vec<_>, crate::EngineError>>()
        .map(Vec::into_boxed_slice)
}

fn into_plural_replay_artifact(
    identity: SemanticsV2BoundIdentity,
    graph: graph::ReservedSemanticsV2Graph,
) -> Result<SemanticsV2ReplayArtifact, crate::EngineError> {
    let private_sequence_publications = private_sequence_publications_from_graph(&graph)?;
    let statement_count = graph.statements.len();
    if statement_count == 0
        || graph.tables.len() < 2
        || graph.images.len() != graph.tables.len()
        || graph.records.len() != statement_count
        || graph.outcomes.len() != statement_count
        || graph.resolutions.len() != statement_count
        || !matches!(graph.response, graph::RetainedResponseEnvelope::Empty(_))
    {
        return Err(replay_error(
            "plural v2 aggregate is outside the generic INSERT recovery vertical",
        ));
    }
    let total_rows = graph.records.iter().try_fold(0_u64, |total, record| {
        total
            .checked_add(u64::from(record.facts().row_count))
            .ok_or_else(|| replay_error("plural v2 aggregate row count overflows"))
    })?;
    if total_rows == 0 || graph.header.final_database_root == [0; 32] {
        return Err(replay_error(
            "plural v2 aggregate has no rows or final database root",
        ));
    }
    for (ordinal, (((record, statement), resolution), outcome)) in graph
        .records
        .iter()
        .zip(&graph.statements)
        .zip(&graph.resolutions)
        .zip(&graph.outcomes)
        .enumerate()
    {
        let facts = record.facts();
        let table = graph
            .tables
            .get(resolution.table_ref as usize)
            .ok_or_else(|| replay_error("plural statement target table is absent"))?;
        if facts.statement_ordinal.as_u32() as usize != ordinal
            || statement.statement_ordinal as usize != ordinal
            || resolution.statement_ordinal as usize != ordinal
            || outcome.statement_ordinal as usize != ordinal
            || resolution.record_ref as usize != ordinal
            || resolution.outcome_ref as usize != ordinal
            || facts.typed_statement_digest != statement.typed_statement_digest
            || facts.typed_statement_digest != resolution.typed_statement_digest
            || facts.typed_statement_digest != outcome.typed_statement_digest
            || facts.row_count == 0
            || facts.column_count == 0
            || facts.target.oid != table.display_oid
            || resolution.affected_row_count != u64::from(facts.row_count)
            || outcome.outcome.affected_rows != u64::from(facts.row_count)
            || outcome.outcome.kind != gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
        {
            return Err(replay_error(
                "plural v2 statement directory is not an exact committed INSERT mapping",
            ));
        }
    }

    let canonical_identity = gpu_db_wal::CanonicalIdentity {
        database_id: identity.database_id,
        cluster_id: identity.cluster_id,
        timeline_id: identity.timeline_id,
        format_epoch: identity.format_epoch,
    };
    let mut pending_tables = Vec::with_capacity(graph.tables.len());
    let mut allocator_intervals = Vec::with_capacity(graph.tables.len());
    for (table_ref, table) in graph.tables.iter().enumerate() {
        let table_ref = u32::try_from(table_ref)
            .map_err(|_| replay_error("plural table reference exceeds u32"))?;
        if table.table_ref != table_ref
            || table.image_ref != table_ref
            || table.catalog_epoch != identity.catalog_epoch
            || (table.initial_table_root == [0; 32] && !table.initial_table_absent)
            || table.final_table_root == [0; 32]
        {
            return Err(replay_error(
                "plural v2 table block exceeds the generic recovery shape",
            ));
        }
        let table_rows = graph
            .resolutions
            .iter()
            .filter(|resolution| resolution.table_ref == table_ref)
            .try_fold(0_u64, |rows, resolution| {
                rows.checked_add(resolution.affected_row_count)
                    .ok_or_else(|| replay_error("plural table row count overflows"))
            })?;
        let surviving_rows = u64::from(table.transition_count);
        let expected_final_rows = if table.resets_existing_rows {
            surviving_rows
        } else {
            table
                .initial_logical_row_count
                .checked_add(surviving_rows)
                .ok_or_else(|| replay_error("plural table logical row count overflows"))?
        };
        let expected_allocator_high_water = table
            .row_allocator_before
            .checked_add(table_rows)
            .ok_or_else(|| replay_error("plural table allocator interval overflows"))?;
        if u64::from(table.disposition_count) != table_rows
            || surviving_rows > table_rows
            || table.final_logical_row_count != expected_final_rows
            || table.row_allocator_high_water != expected_allocator_high_water
            || (surviving_rows != 0
                && (table.data_generation_after != identity.commit_sequence
                    || table.data_generation_after <= table.data_generation_before
                    || table.initial_table_root == table.final_table_root))
            || (surviving_rows == 0
                && (table.data_generation_after != table.data_generation_before
                    || table.initial_table_root != table.final_table_root))
        {
            return Err(replay_error(
                "plural table rows do not close their allocator and logical intervals",
            ));
        }
        allocator_intervals.push((table.row_allocator_before, table.row_allocator_high_water));
        let first_resolution = graph
            .resolutions
            .iter()
            .find(|resolution| resolution.table_ref == table_ref)
            .ok_or_else(|| replay_error("plural table has no source statement"))?;
        let first_record = graph
            .records
            .get(first_resolution.record_ref as usize)
            .ok_or_else(|| replay_error("plural table source record is absent"))?;
        let first_target = first_record.target_identity();
        if graph
            .resolutions
            .iter()
            .filter(|resolution| resolution.table_ref == table_ref)
            .any(|resolution| {
                graph
                    .records
                    .get(resolution.record_ref as usize)
                    .is_none_or(|record| {
                        let target = record.target_identity();
                        target.schema != first_target.schema
                            || target.name != first_target.name
                            || target.oid != table.display_oid
                            || record.facts().column_count != table.catalog_column_count
                    })
            })
        {
            return Err(replay_error(
                "plural table statements do not share one exact target shape",
            ));
        }
        let indexes = if surviving_rows == 0 {
            if graph
                .indexes
                .iter()
                .any(|index| index.owner_table_ref == table_ref)
            {
                return Err(replay_error(
                    "neutral plural table retained an index publication authority",
                ));
            }
            Box::new([])
        } else {
            replay_indexes_for_table(&identity, &graph, table, table_ref, first_record)?
        };
        let transition_start = table.transition_start as usize;
        let transition_end = transition_start
            .checked_add(table.transition_count as usize)
            .ok_or_else(|| replay_error("plural transition range overflows"))?;
        let transitions = graph
            .transitions
            .get(transition_start..transition_end)
            .ok_or_else(|| replay_error("plural table transition range is absent"))?;
        let row_sources = transitions
            .iter()
            .enumerate()
            .map(|(local_row, transition)| {
                let local_row = u32::try_from(local_row)
                    .map_err(|_| replay_error("plural table row ordinal exceeds u32"))?;
                let transition_ref = table
                    .transition_start
                    .checked_add(local_row)
                    .ok_or_else(|| replay_error("plural transition reference overflows"))?;
                let stable_row_id = table
                    .row_allocator_before
                    .checked_add(u64::from(local_row))
                    .ok_or_else(|| replay_error("plural stable row identity overflows"))?;
                let source_record = graph
                    .records
                    .get(transition.source_statement_ordinal as usize)
                    .ok_or_else(|| replay_error("plural transition source statement is absent"))?;
                let source_resolution = graph
                    .resolutions
                    .get(transition.source_statement_ordinal as usize)
                    .ok_or_else(|| replay_error("plural transition source resolution is absent"))?;
                if transition.transition_ref != transition_ref
                    || transition.table_ref != table_ref
                    || transition.stable_row_id != stable_row_id
                    || transition.image_ref != table_ref
                    || transition.image_row_ordinal != local_row
                    || (transition.final_writer_statement_digest == [0; 32]
                        && transition.final_writer_statement_ordinal
                            != transition.source_statement_ordinal)
                    || (transition.final_writer_statement_digest != [0; 32]
                        && transition.final_writer_statement_ordinal
                            <= transition.source_statement_ordinal)
                    || source_resolution.table_ref != table_ref
                    || transition.source_row_ordinal >= source_record.facts().row_count
                    || transition.typed_statement_digest
                        != source_record.facts().typed_statement_digest
                {
                    return Err(replay_error(
                        "plural transition does not retain its exact table-local row source",
                    ));
                }
                Ok(
                    crate::engine_transaction_delta::TypedInsertRuntimeGenerationRowSource {
                        stable_row_id: transition.stable_row_id,
                        statement_ordinal: transition.source_statement_ordinal,
                        source_row_ordinal: transition.source_row_ordinal,
                    },
                )
            })
            .collect::<Result<Vec<_>, crate::EngineError>>()?
            .into_boxed_slice();
        let metadata = SemanticsV2ReplayMetadata {
            canonical_identity,
            leader_epoch: identity.leader_epoch,
            stable_transaction_id: identity.stable_transaction_id,
            request_digest: identity.request_digest,
            autocommit: identity.autocommit,
            commit_sequence: identity.commit_sequence,
            catalog_epoch: identity.catalog_epoch,
            catalog_digest: identity.catalog_digest,
            statement_count: u32::try_from(statement_count)
                .map_err(|_| replay_error("plural statement count exceeds u32"))?,
            typed_statement_digest: [0; 32],
            stable_table_id: table.stable_table_id,
            display_oid: table.display_oid,
            table_schema_digest: table.schema_digest,
            data_generation_before: table.data_generation_before,
            data_generation_after: table.data_generation_after,
            row_allocator_before: table.row_allocator_before,
            row_allocator_high_water: table.row_allocator_high_water,
            initial_logical_row_count: table.initial_logical_row_count,
            final_logical_row_count: table.final_logical_row_count,
            resets_existing_rows: table.resets_existing_rows,
            initial_table_absent: table.initial_table_absent,
            initial_table_root: table.initial_table_root,
            final_table_root: table.final_table_root,
            initial_database_root: identity.initial_database_root,
            final_database_root: graph.header.final_database_root,
            image_layout_digest: table.image_layout_digest,
            image_content_digest: table.image_content_digest,
            affected_rows: table_rows,
        };
        pending_tables.push((
            metadata,
            table_ref,
            Box::<str>::from(first_target.schema),
            Box::<str>::from(first_target.name),
            indexes,
            row_sources,
        ));
    }
    allocator_intervals.sort_unstable_by_key(|interval| interval.0);
    let global_allocator_before = allocator_intervals
        .first()
        .map(|interval| interval.0)
        .ok_or_else(|| replay_error("plural allocator interval is absent"))?;
    let mut global_allocator_high_water = global_allocator_before;
    for (start, end) in allocator_intervals {
        if start != global_allocator_high_water || end <= start {
            return Err(replay_error(
                "plural table allocator intervals do not form one exact transaction range",
            ));
        }
        global_allocator_high_water = end;
    }
    if global_allocator_high_water
        != global_allocator_before
            .checked_add(total_rows)
            .ok_or_else(|| replay_error("plural transaction allocator range overflows"))?
    {
        return Err(replay_error(
            "plural allocator range does not equal the transaction row count",
        ));
    }
    let mut tables = Vec::with_capacity(pending_tables.len());
    for ((metadata, table_ref, target_schema, target_name, indexes, row_sources), final_image) in
        pending_tables.into_iter().zip(graph.images)
    {
        tables.push(SemanticsV2ReplayTableArtifact {
            metadata,
            table_ref,
            target_schema,
            target_name,
            indexes,
            row_sources,
            final_image,
        });
    }
    let mut metadata = tables
        .first()
        .ok_or_else(|| replay_error("plural replay table artifact is absent"))?
        .metadata;
    metadata.row_allocator_before = global_allocator_before;
    metadata.row_allocator_high_water = global_allocator_high_water;
    metadata.affected_rows = total_rows;
    Ok(SemanticsV2ReplayArtifact {
        metadata,
        tables: tables.into_boxed_slice(),
        private_sequence_publications: private_sequence_publications.publications,
        private_sequence_name_changes: private_sequence_publications.name_changes,
        catalog_composition: None,
    })
}

fn private_sequence_publications_from_graph(
    graph: &graph::ReservedSemanticsV2Graph,
) -> Result<PrivateSequenceReplayPublications, crate::EngineError> {
    let mut final_states = std::collections::BTreeMap::<u32, (Box<str>, i64, bool)>::new();
    let mut name_changes = std::collections::BTreeMap::<u32, (Box<str>, Box<str>)>::new();
    for record in &graph.records {
        for source in record.sequence_effects() {
            if !matches!(
                source.kind,
                crate::typed_insert_batch::DecodedSequenceEffectKindFacts::Private { .. }
            ) {
                continue;
            }
            let mut bindings = record
                .sequence_bindings()
                .filter(|binding| binding.effect_ordinal == source.request.effect_ordinal);
            let binding = bindings.next().ok_or_else(|| {
                replay_error("private sequence replay effect has no strict S2 binding")
            })?;
            if bindings.next().is_some() || binding.request != source.request {
                return Err(replay_error(
                    "private sequence replay effect has a noncanonical S2 binding",
                ));
            }
            record_private_sequence_final_state(
                &mut final_states,
                &mut name_changes,
                source.request.sequence_oid,
                binding.effective_name,
                source.resolved_value,
                true,
            )?;
        }
    }
    for effect in &graph.sequence_effects {
        let Some(restart) = effect.terminal_restart else {
            continue;
        };
        let record = graph
            .records
            .get(effect.statement_ordinal as usize)
            .ok_or_else(|| replay_error("terminal sequence restart has no S2 record"))?;
        let source = record
            .sequence_effects()
            .find(|source| source.request.effect_ordinal == effect.effect_ordinal)
            .ok_or_else(|| replay_error("terminal sequence restart has no S2 source"))?;
        let binding = record
            .sequence_bindings()
            .find(|binding| binding.effect_ordinal == effect.effect_ordinal)
            .ok_or_else(|| replay_error("terminal sequence restart has no S2 binding"))?;
        if restart.sequence_oid != source.request.sequence_oid
            || restart.descriptor_digest != binding.descriptor_digest
        {
            return Err(replay_error(
                "terminal sequence restart changed its stable S2 identity",
            ));
        }
        record_private_sequence_final_state(
            &mut final_states,
            &mut name_changes,
            restart.sequence_oid,
            binding.effective_name,
            restart.last_value,
            false,
        )?;
    }
    Ok(PrivateSequenceReplayPublications {
        publications: final_states
            .into_iter()
            .map(|(sequence_oid, (name, last_value, is_called))| {
                crate::engine_commit::LiveTypedPrivateSequencePublication {
                    name,
                    sequence_oid,
                    last_value,
                    is_called,
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        name_changes: name_changes
            .into_iter()
            .map(
                |(sequence_oid, (before_name, after_name))| PrivateSequenceNameChange {
                    sequence_oid,
                    before_name,
                    after_name,
                },
            )
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    })
}

fn record_private_sequence_final_state(
    final_states: &mut std::collections::BTreeMap<u32, (Box<str>, i64, bool)>,
    name_changes: &mut std::collections::BTreeMap<u32, (Box<str>, Box<str>)>,
    sequence_oid: u32,
    name: &str,
    last_value: i64,
    is_called: bool,
) -> Result<(), crate::EngineError> {
    let next_name: Box<str> = name.into();
    if let Some((current_name, _, _)) = final_states.get(&sequence_oid) {
        if current_name.as_ref() != next_name.as_ref() {
            match name_changes.entry(sequence_oid) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((current_name.clone(), next_name.clone()));
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if entry.get().1.as_ref() == next_name.as_ref() => {}
                std::collections::btree_map::Entry::Occupied(_) => {
                    return Err(replay_error(
                        "private sequence replay effect has more than one stable-OID rename",
                    ));
                }
            }
        }
    }
    final_states.insert(sequence_oid, (next_name, last_value, is_called));
    Ok(())
}

fn replay_error(message: &str) -> crate::EngineError {
    crate::EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v2 replay: {message}"
    ))
}

impl<'retention> AggregateReplayTxn<CatalogAllocatorPending<'retention>> {
    /// Consume the authenticated retention authority only after the one pinned catalog and the
    /// concrete ADR-014 allocator-lease index close the complete retained graph.  The resulting
    /// owner deliberately stops before durable sequence validation, GPU capacity/replay, WAL,
    /// recovery, apply, result, or publication work.
    pub(super) fn validate_catalog_and_allocator<'catalog>(
        self,
        witness: SemanticsV2CatalogAllocatorWitness<'catalog>,
    ) -> Result<AggregateReplayTxn<DurableSequencePending<'retention, 'catalog>>, crate::EngineError>
    {
        let allocator_assignment =
            catalog_validation::validate(self.graph.identity, &self.graph.graph, &witness)?;
        let SemanticsV2CatalogAllocatorWitness {
            catalog,
            allocator_index,
        } = witness;
        Ok(AggregateReplayTxn {
            graph: self.graph,
            phase: DurableSequencePending {
                authority: self.phase.authority,
                catalog,
                allocator_index,
                allocator_assignment,
                seal: PrivateSeal,
            },
        })
    }
}

impl<'retention, 'catalog> AggregateReplayTxn<DurableSequencePending<'retention, 'catalog>> {
    /// Consume the pinned catalog/allocator owner only after the sole Engine
    /// `sequence_value_outcomes` index proves every retained S5 transition.  This performs no
    /// sequence inference or re-evaluation and stops before GPU capacity/replay, compiler, WAL,
    /// recovery, apply, result, and publication work.
    pub(super) fn validate_durable_sequence_outcomes<'sequence>(
        self,
        sequence_index: sequence_validation::SemanticsV2DurableSequenceOutcomeIndexProof<'sequence>,
    ) -> Result<
        AggregateReplayTxn<GenerationPending<'retention, 'catalog, 'sequence>>,
        crate::EngineError,
    > {
        sequence_validation::validate(self.graph.identity, &self.graph.graph, &sequence_index)?;
        Ok(AggregateReplayTxn {
            graph: self.graph,
            phase: GenerationPending {
                authority: self.phase.authority,
                catalog: self.phase.catalog,
                allocator_index: self.phase.allocator_index,
                allocator_assignment: self.phase.allocator_assignment,
                sequence_index,
                seal: PrivateSeal,
            },
        })
    }
}

impl<'retention, 'catalog, 'sequence>
    AggregateReplayTxn<GenerationPending<'retention, 'catalog, 'sequence>>
{
    /// Consume the fully witness-bound replay owner through the sole neutral generation builder.
    /// Builder reservation is the final fallible operation before launch; every launched path
    /// subsequently proves quiescence or parks all backing in the pre-reserved quarantine.
    #[allow(dead_code)]
    pub(self) fn validate_generation<B>(
        self,
        builder: B,
        quarantine_registry: &generation_validation::GenerationQuarantineRegistry<
            B::Candidate,
            B::Work,
        >,
    ) -> Result<
        AggregateReplayTxn<GenerationValidated<'retention, 'catalog, 'sequence, B::Candidate>>,
        crate::EngineError,
    >
    where
        B: generation_validation::SemanticsV2ReservedGenerationBuilder,
    {
        let AggregateReplayTxn { graph, phase } = self;
        let generation = generation_validation::validate_reserved_graph(
            &graph,
            &phase.catalog,
            builder,
            quarantine_registry,
        )?;
        Ok(AggregateReplayTxn {
            graph,
            phase: GenerationValidated {
                authority: phase.authority,
                catalog: phase.catalog,
                allocator_index: phase.allocator_index,
                allocator_assignment: phase.allocator_assignment,
                sequence_index: phase.sequence_index,
                generation,
                seal: PrivateSeal,
            },
        })
    }
}

#[cfg(test)]
impl Q2CodecClosedSemanticsV2 {
    /// Consume the sole codec-closed owner after exact pinned-catalog and durable allocator
    /// closure. No generation result is accepted, retained, or constructed at this boundary.
    pub(super) fn validate_catalog_and_allocator<'a>(
        self,
        witness: SemanticsV2CatalogAllocatorWitness<'a>,
    ) -> Result<GenerationPendingSemanticsV2<'a>, crate::EngineError> {
        let allocator_assignment =
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
                allocator_assignment,
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

/// Build test-only allocator backing while this move-only owner is disassembled, then restore the
/// owner only inside the same proof callback.  Assignment rows are synthesized from the real
/// retained graph but can neither escape this extent nor be constructed beside a second proof.
#[cfg(test)]
fn with_scoped_allocator_proof_for_test<T, Phase>(
    pending: AggregateReplayTxn<Phase>,
    leases: &[AllocatorLeaseSpecForTest],
    sabotage: Option<AllocatorAssignmentProofSabotageForTest>,
    operation: impl FnOnce(
        AggregateReplayTxn<Phase>,
        catalog_validation::SemanticsV2DurableAllocatorIndexProof<'_>,
    ) -> T,
) -> T {
    let AggregateReplayTxn {
        graph: RetainedSemanticsV2Graph { identity, graph },
        phase,
    } = pending;
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
        graph,
        &authority_leases,
        sabotage,
        |allocator_index, graph| {
            operation(
                AggregateReplayTxn {
                    graph: RetainedSemanticsV2Graph { identity, graph },
                    phase,
                },
                allocator_index,
            )
        },
    )
}

#[cfg(test)]
fn with_q2_scoped_allocator_proof_for_test<T>(
    closed: Q2CodecClosedSemanticsV2,
    leases: &[AllocatorLeaseSpecForTest],
    operation: impl FnOnce(
        Q2CodecClosedSemanticsV2,
        catalog_validation::SemanticsV2DurableAllocatorIndexProof<'_>,
    ) -> T,
) -> T {
    with_scoped_allocator_proof_for_test(
        AggregateReplayTxn {
            graph: closed.graph,
            phase: PrivateSeal,
        },
        leases,
        None,
        |closed, allocator_index| {
            operation(
                Q2CodecClosedSemanticsV2 {
                    graph: closed.graph,
                },
                allocator_index,
            )
        },
    )
}

/// Consume an actual codec-closed owner through the one catalog/allocator transition while the
/// test fixture's durable proof is still scoped to this call.  The callback receives only the
/// resulting pending typestate; it cannot retain or construct a proof, inspect the graph, or
/// spell another codec-close transition.
#[cfg(test)]
pub(super) fn with_catalog_allocator_pending_for_test<T>(
    closed: Q2CodecClosedSemanticsV2,
    catalog: SemanticsV2CatalogWitness<'_>,
    leases: &[AllocatorLeaseSpecForTest],
    operation: impl FnOnce(GenerationPendingSemanticsV2<'_>) -> Result<T, crate::EngineError>,
) -> Result<T, crate::EngineError> {
    with_q2_scoped_allocator_proof_for_test(closed, leases, |closed, allocator_index| {
        operation(closed.validate_catalog_and_allocator_for_test(
            SemanticsV2CatalogAllocatorWitness {
                catalog,
                allocator_index,
            },
        )?)
    })
}

/// Consume a real production-shaped catalog/allocator owner through the one durable sequence
/// transition while both the durable allocator and Engine-index-shaped proof remain scoped to the
/// callback. The caller cannot retain, manufacture, or inspect any carried authority.
#[cfg(test)]
fn with_catalog_allocator_sequence_pending_for_test<T>(
    catalog_pending: AggregateReplayTxn<CatalogAllocatorPending<'_>>,
    catalog: SemanticsV2CatalogWitness<'_>,
    leases: &[AllocatorLeaseSpecForTest],
    outcomes: &[SequenceOutcomeSpecForTest<'_>],
    sabotage: Option<SequenceOutcomeProofSabotageForTest>,
    assignment_sabotage: Option<AllocatorAssignmentProofSabotageForTest>,
    operation: impl FnOnce(
        AggregateReplayTxn<GenerationPending<'_, '_, '_>>,
    ) -> Result<T, crate::EngineError>,
) -> Result<T, crate::EngineError> {
    with_scoped_allocator_proof_for_test(
        catalog_pending,
        leases,
        assignment_sabotage,
        |catalog_pending, allocator_index| {
            let identity = catalog_pending.graph.identity;
            let sequence_pending = catalog_pending.validate_catalog_and_allocator(
                SemanticsV2CatalogAllocatorWitness {
                    catalog,
                    allocator_index,
                },
            )?;
            sequence_validation::with_sequence_outcome_proof_for_test(
                identity,
                outcomes,
                sabotage,
                |sequence_index| {
                    operation(sequence_pending.validate_durable_sequence_outcomes(sequence_index)?)
                },
            )
        },
    )
}

/// Test-only live-retention continuation over the same production phase arrows. It is callback
/// scoped so the authenticated claim, allocator pin, and sequence-index proof cannot escape.
#[cfg(test)]
pub(super) fn with_live_sequence_pending_for_test<T>(
    pending: AggregateReplayTxn<RetentionAuthorityPending>,
    catalog: SemanticsV2CatalogWitness<'_>,
    leases: &[AllocatorLeaseSpecForTest],
    outcomes: &[SequenceOutcomeSpecForTest<'_>],
    sabotage: Option<SequenceOutcomeProofSabotageForTest>,
    assignment_sabotage: Option<AllocatorAssignmentProofSabotageForTest>,
    operation: impl FnOnce(
        AggregateReplayTxn<GenerationPending<'_, '_, '_>>,
    ) -> Result<T, crate::EngineError>,
) -> Result<T, crate::EngineError> {
    retention_authority::with_live_claim_pending_for_test(pending, |catalog_pending| {
        with_catalog_allocator_sequence_pending_for_test(
            catalog_pending,
            catalog,
            leases,
            outcomes,
            sabotage,
            assignment_sabotage,
            operation,
        )
    })
}

/// Historical no-retention follows the same catalog/allocator/sequence boundary. In particular,
/// an empty S5 vector must still authenticate the complete borrowed index before it advances.
#[cfg(test)]
pub(super) fn with_historical_sequence_pending_for_test<T>(
    pending: AggregateReplayTxn<RetentionAuthorityPending>,
    catalog: SemanticsV2CatalogWitness<'_>,
    leases: &[AllocatorLeaseSpecForTest],
    outcomes: &[SequenceOutcomeSpecForTest<'_>],
    sabotage: Option<SequenceOutcomeProofSabotageForTest>,
    assignment_sabotage: Option<AllocatorAssignmentProofSabotageForTest>,
    operation: impl FnOnce(
        AggregateReplayTxn<GenerationPending<'_, '_, '_>>,
    ) -> Result<T, crate::EngineError>,
) -> Result<T, crate::EngineError> {
    retention_authority::with_historical_no_retention_pending_for_test(pending, |catalog_pending| {
        with_catalog_allocator_sequence_pending_for_test(
            catalog_pending,
            catalog,
            leases,
            outcomes,
            sabotage,
            assignment_sabotage,
            operation,
        )
    })
}

/// Complete the Q2 test-only catalog, allocator, generation, and logical-reencoding chain.
/// The fixture contributes only a real codec-closed owner, pinned catalog facts, exact allocator
/// intervals, and a closed case tag.  The private builder/candidate/work types and the final
/// validated owner never cross this retained facade.
#[cfg(test)]
pub(super) fn reencode_q2_golden_from_closed_for_test(
    closed: Q2CodecClosedSemanticsV2,
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
    closed: Q2CodecClosedSemanticsV2,
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

#[cfg(test)]
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
fn validate_catalog_allocator_witness_identity(
    identity: SemanticsV2BoundIdentity,
    graph: &graph::ReservedSemanticsV2Graph,
    witness: &SemanticsV2CatalogAllocatorWitness<'_>,
) -> Result<(), crate::EngineError> {
    let catalog = &witness.catalog;
    // The sole fresh-catalog exception is already authenticated by S7: every target must be the
    // transaction's initially absent table.  Any append/reset/table mix still requires a
    // nonzero catalog predecessor, so this does not create a second bootstrap or recovery path.
    let genesis_first_table = identity.catalog_epoch == 0
        && !graph.tables.is_empty()
        && graph.tables.iter().all(|table| table.initial_table_absent);
    if identity.database_id == [0; 16]
        || (identity.catalog_epoch == 0 && !genesis_first_table)
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
mod replay_shell_source_guards {
    #[test]
    fn retained_shell_has_one_sealed_durable_sequence_transition_and_stops_before_gpu_replay() {
        let source = include_str!("retained.rs");
        let facade = include_str!("../semantics_v2.rs");
        let fill = include_str!("retained/fill.rs");
        let retention = include_str!("retained/retention_authority.rs");
        let sequence = include_str!("retained/sequence_validation.rs");
        let sealed = ["struct Private", "Seal;"].concat();
        let quarantined = ["struct Codec", "Quarantined(PrivateSeal);"].concat();
        let pending = ["struct RetentionAuthority", "Pending(PrivateSeal);"].concat();
        let quarantined_impl = ["impl AggregateReplayTxn<", "CodecQuarantined", ">"].concat();
        let pending_transition = [
            "Result<AggregateReplayTxn<",
            "RetentionAuthorityPending",
            ">",
        ]
        .concat();
        let strict_fill = ["fn quarantine_", "after_strict_fill("].concat();
        assert!(source.contains(&sealed));
        assert!(source.contains(&quarantined));
        assert!(source.contains(&pending));
        assert!(source.contains(&quarantined_impl));
        assert!(source.contains(&pending_transition));
        assert_eq!(
            source.matches(&strict_fill).count(),
            1,
            "strict fill has exactly one move-only aggregate constructor"
        );
        assert_eq!(
            source.matches(&pending_transition).count(),
            1,
            "codec quarantine has exactly one production phase transition"
        );
        let pending_impl = ["impl AggregateReplayTxn<", "RetentionAuthorityPending", ">"].concat();
        assert_eq!(
            source.matches(&pending_impl).count(),
            1,
            "retention authority has one sealed aggregate transition"
        );
        let catalog_pending = ["struct CatalogAllocator", "Pending<'a>"].concat();
        let catalog_pending_impl = [
            "impl<'retention> AggregateReplayTxn<",
            "CatalogAllocatorPending<'retention>>",
        ]
        .concat();
        assert!(source.contains(&catalog_pending));
        assert_eq!(
            source.matches(&catalog_pending_impl).count(),
            1,
            "the catalog/allocator phase has exactly one sealed aggregate transition"
        );
        let sequence_pending = ["struct DurableSequence", "Pending<'"].concat();
        let sequence_pending_impl = [
            "impl<'retention, 'catalog> AggregateReplayTxn<",
            "DurableSequencePending<'retention, 'catalog>>",
        ]
        .concat();
        assert!(source.contains(&sequence_pending));
        assert_eq!(
            source.matches(&sequence_pending_impl).count(),
            1,
            "the durable-sequence phase has exactly one sealed aggregate transition"
        );
        let generation_pending = ["struct Generation", "Pending<'"].concat();
        let generation_pending_impl = [
            "impl<'retention, 'catalog, 'sequence> AggregateReplayTxn<",
            "GenerationPending<'retention, 'catalog, 'sequence>>",
        ]
        .concat();
        assert!(source.contains(&generation_pending));
        assert!(
            !source.contains(&generation_pending_impl),
            "this checkpoint must stop before the generation/GPU replay successor"
        );
        let production_shell = source
            .split("#[cfg(test)]\nmod replay_shell_source_guards")
            .next()
            .expect("retained source has a production boundary");
        for forbidden in [
            "ReplayBaseGenerationPin",
            "CudaI32InsertReplayPreparation",
            "CudaInsertReplaySubmission",
            "engine_data_generation",
        ] {
            assert!(
                !production_shell.contains(forbidden),
                "the retained prerequisite must not consume or couple to later replay authority: {forbidden}"
            );
        }
        let transaction_decl = ["pub(super) struct AggregateReplayTxn", "<Phase>"].concat();
        let declaration_offset = source
            .find(&transaction_decl)
            .expect("move-only aggregate owner is declared");
        let derive_window = &source[declaration_offset.saturating_sub(128)..declaration_offset];
        assert!(
            !derive_window.contains("#[derive"),
            "AggregateReplayTxn cannot derive Clone or Copy"
        );
        for forbidden in [
            ["impl Clone for ", "AggregateReplayTxn"].concat(),
            ["impl Copy for ", "AggregateReplayTxn"].concat(),
            ["impl<Phase> Clone for ", "AggregateReplayTxn"].concat(),
            ["impl<Phase> Copy for ", "AggregateReplayTxn"].concat(),
            ["impl<Phase> std::ops::Deref for ", "AggregateReplayTxn"].concat(),
            ["impl<Phase> AsRef", "<"].concat(),
            ["fn raw_", "aggregate"].concat(),
            ["fn aggregate_", "body"].concat(),
            ["fn into_", "inner"].concat(),
            ["fn reencode_", "s8"].concat(),
        ] {
            assert!(
                !source.contains(&forbidden),
                "replay shell must not expose {forbidden}"
            );
        }
        let catalog_module = ["#[path = \"", "retained/catalog_validation.rs\"]"].concat();
        let generation_module = ["#[path = \"", "retained/generation_validation.rs\"]"].concat();
        let q2_struct = [
            "#[cfg(test)]\npub(super) struct ",
            "Q2CodecClosedSemanticsV2",
        ]
        .concat();
        let q2_impl = ["#[cfg(test)]\nimpl ", "Q2CodecClosedSemanticsV2"].concat();
        let q2_empty = ["Q2 codec-closed adapter ", "requires canonical empty S8"].concat();
        assert!(source.contains(&catalog_module));
        assert!(source.contains(&generation_module));
        assert!(source.contains(&q2_struct));
        assert!(source.contains(&q2_impl));
        assert!(source.contains(&q2_empty));
        assert!(facade.contains("#[cfg(test)]\nfn fill_canonical_semantics_v2_for_test"));
        assert!(facade.contains("#[cfg(test)]\nfn codec_closed_canonical_semantics_v2_for_test"));
        assert!(!facade.contains("pub(super) fn fill_canonical_semantics_v2_for_test"));
        assert!(fill.contains(
            "This has no transition to WAL, recovery, execution, GPU, result, or publication state."
        ));
        assert!(!fill.contains("fn publish"));
        assert!(!fill.contains("fn replay"));
        assert!(retention.contains("struct AuthenticatedClaimStatusIndex"));
        assert!(retention.contains("struct ValidatedRetentionAuthority"));
        assert!(retention.contains("HistoricalNoRetention"));
        let production_retention = retention
            .split("#[cfg(test)]")
            .next()
            .expect("retention authority has a production boundary");
        for forbidden in ["HashMap", "Vec<", "Box<", ".clone()", ".collect()"] {
            assert!(
                !production_retention.contains(forbidden),
                "retention authority must borrow without clone or allocation: {forbidden}"
            );
        }
        let live_claim_validator = retention
            .split("fn validate_live_claim")
            .nth(1)
            .and_then(|source| source.split("fn validate_historical_no_retention").next())
            .expect("retention authority has one live-claim validator");
        for sealed_terminal_field in [
            "s6_retained_statement_bits",
            "s7_retained_statement_bits",
            "s8_artifact_statement_ordinals",
            "s8_artifact_count",
            "aggregate_retained_response",
            "status_artifact_count",
            "status_deadline",
        ] {
            assert!(
                !live_claim_validator.contains(sealed_terminal_field),
                "retention authority must carry, not inspect, terminal field {sealed_terminal_field}"
            );
        }
        for forbidden in [
            "SemanticsV2Catalog",
            "allocator_index",
            "GenerationPending",
            "WalBuffer",
            "DeviceInsertPlan",
            "fn publish",
            "fn replay",
            "fn apply",
        ] {
            assert!(
                !retention.contains(forbidden),
                "retention authority must not gain later authority {forbidden}"
            );
        }
        let catalog_allocator = source
            .split(&catalog_pending_impl)
            .nth(1)
            .and_then(|source| source.split(&sequence_pending_impl).next())
            .expect("catalog/allocator transition is bounded before durable sequence validation");
        for forbidden in [
            "sequence_value_outcomes",
            "DeviceInsertPlan",
            "WalBuffer",
            "fn replay",
            "fn apply",
            "fn publish",
        ] {
            assert!(
                !catalog_allocator.contains(forbidden),
                "catalog/allocator transition must not gain later authority {forbidden}"
            );
        }
        let durable_sequence = source
            .split(&sequence_pending_impl)
            .nth(1)
            .and_then(|source| {
                source
                    .split("#[cfg(test)]\nimpl Q2CodecClosedSemanticsV2")
                    .next()
            })
            .expect("durable-sequence transition is bounded before the test-only Q2 bridge");
        for forbidden in [
            "SemanticsV2CatalogAllocatorWitness",
            "RetentionAuthorityInput",
            "catalog_validation::validate",
            "DeviceInsertPlan",
            "WalBuffer",
            "fn replay",
            "fn apply",
            "fn publish",
        ] {
            assert!(
                !durable_sequence.contains(forbidden),
                "durable-sequence transition must not gain a second authority or later work: {forbidden}"
            );
        }
        let production_sequence = sequence
            .split("#[cfg(test)]")
            .next()
            .expect("durable sequence has a production boundary");
        for forbidden in [
            "HashMap",
            "Vec<",
            "Box<",
            ".clone()",
            ".collect()",
            "sequence_value_input_digest",
            "canonical_request_digest",
        ] {
            assert!(
                !production_sequence.contains(forbidden),
                "durable sequence proof must borrow without clone or allocation: {forbidden}"
            );
        }
        for forbidden in [
            "SemanticsV2Catalog",
            "ValidatedRetentionAuthority",
            "DeviceInsertPlan",
            "WalBuffer",
            "fn replay",
            "fn apply",
            "fn publish",
        ] {
            assert!(
                !production_sequence.contains(forbidden),
                "durable sequence proof must not gain later authority {forbidden}"
            );
        }
        for (name, q2_source) in [
            ("Q2 witnesses", include_str!("goldens/q2_witnesses.rs")),
            (
                "Q2 golden reencoder",
                include_str!("goldens/q2_reencode.rs"),
            ),
            (
                "Q2 generation builder",
                include_str!("retained/generation_validation/golden_builder.rs"),
            ),
            (
                "Q2 retained reencoder",
                include_str!("retained/reencode.rs"),
            ),
        ] {
            for forbidden in ["S8", "AggregateReplayTxn"] {
                assert!(
                    !q2_source.contains(forbidden),
                    "{name} must remain isolated from {forbidden}"
                );
            }
        }
    }
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
    fn checked_in_abort_fixture_crosses_retention_catalog_allocator_and_owned_lifecycle() {
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
        let retention_pending =
            super::super::fill_canonical_semantics_v2_for_test(&outer, &outcome, &fragments)
                .expect("checked-in minimal abort fixture fills retained owners")
                .close_codec(None)
                .expect("checked-in minimal abort fixture closes codec witnesses");
        let live_retention_pending =
            super::super::fill_canonical_semantics_v2_for_test(&outer, &outcome, &fragments)
                .expect("checked-in minimal abort fixture fills retained owners")
                .close_codec(None)
                .expect("checked-in minimal abort fixture closes codec witnesses");
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
        let catalog_for_retention = SemanticsV2CatalogWitness {
            database_id: [0xa1; 16],
            catalog_epoch: CATALOG_EPOCH,
            catalog_digest: [0x33; 32],
            tables: &tables,
            indexes: &[],
            domains: &[],
            guards: &guards,
            sequences: &[],
        };
        let catalog_for_live_retention = SemanticsV2CatalogWitness {
            database_id: [0xa1; 16],
            catalog_epoch: CATALOG_EPOCH,
            catalog_digest: [0x33; 32],
            tables: &tables,
            indexes: &[],
            domains: &[],
            guards: &guards,
            sequences: &[],
        };
        let _identity = SemanticsV2BoundIdentity {
            database_id: [0xa1; 16],
            cluster_id: [0xa3; 16],
            timeline_id: [0xa2; 16],
            format_epoch: 5,
            leader_epoch: 6,
            catalog_epoch: CATALOG_EPOCH,
            catalog_digest: [0x33; 32],
            stable_transaction_id: STABLE_TRANSACTION_ID,
            request_digest: [0x55; 32],
            autocommit: true,
            commit_sequence: COMMIT_SEQUENCE,
            initial_database_root: [0x44; 32],
        };
        let retention_transition =
            retention_authority::with_historical_no_retention_pending_for_test(
                retention_pending,
                |catalog_pending| {
                    with_scoped_allocator_proof_for_test(
                        catalog_pending,
                        &[AllocatorLeaseSpecForTest {
                            stable_allocator_id: TABLE_ID,
                            lease_start: 1,
                            lease_end: 1_000,
                        }],
                        None,
                        |catalog_pending, allocator_index| {
                            catalog_pending
                                .validate_catalog_and_allocator(
                                    SemanticsV2CatalogAllocatorWitness {
                                        catalog: catalog_for_retention,
                                        allocator_index,
                                    },
                                )
                                .map(drop)
                        },
                    )
                },
            );
        retention_transition.expect(
            "checked-in empty-S8 fixture crosses retention authority and pinned catalog/allocator",
        );
        let live_retention_transition = retention_authority::with_live_claim_pending_for_test(
            live_retention_pending,
            |catalog_pending| {
                with_scoped_allocator_proof_for_test(
                    catalog_pending,
                    &[AllocatorLeaseSpecForTest {
                        stable_allocator_id: TABLE_ID,
                        lease_start: 1,
                        lease_end: 1_000,
                    }],
                    None,
                    |catalog_pending, allocator_index| {
                        catalog_pending
                            .validate_catalog_and_allocator(SemanticsV2CatalogAllocatorWitness {
                                catalog: catalog_for_live_retention,
                                allocator_index,
                            })
                            .map(drop)
                    },
                )
            },
        );
        live_retention_transition.expect(
            "checked-in empty-S8 fixture carries a live claim through catalog/allocator validation",
        );
        let calls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let quarantine_registry = generation_validation::GenerationQuarantineRegistry::<
            generation_validation::AbortCandidate,
            generation_validation::AbortWork,
        >::new();
        let result = with_scoped_allocator_proof_for_test(
            AggregateReplayTxn {
                graph: closed.graph,
                phase: PrivateSeal,
            },
            &[AllocatorLeaseSpecForTest {
                stable_allocator_id: TABLE_ID,
                lease_start: 1,
                lease_end: 1_000,
            }],
            None,
            |closed, allocator_index| {
                let mut pending = Q2CodecClosedSemanticsV2 {
                    graph: closed.graph,
                }
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
