//! Private typed S1--S7 ownership graph reserved before semantic joins.
//!
//! No entry has a raw aggregate-body field. The later decoder fills these typed owners only after
//! every corresponding fallible reservation succeeds, then closes cross-directory and witness
//! invariants before moving the graph into `QuarantinedSemanticsV2`.

#![allow(dead_code)] // The inert retained graph has no production caller until its strict decoder lands.

use crate::typed_insert_batch::{DecodedTypedImage, DecodedTypedInsertRecord};

type Digest = gpu_db_wal::CanonicalDigest;

pub(super) struct ReservedSemanticsV2Graph {
    /// Fixed S7 root identity, carried as typed fields after pass zero rather than a header-byte
    /// duplicate. All variable header facts needed for a future test-only logical reencoder live
    /// here; fixed version/flag constants remain derived.
    pub(super) header: super::super::pass_zero::SemanticsV2S7HeaderIdentity,
    pub(super) statements: Vec<RetainedStatement>,
    pub(super) records: Vec<DecodedTypedInsertRecord>,
    pub(super) dispositions: Vec<RetainedDisposition>,
    pub(super) sequence_effects: Vec<RetainedSequenceEffect>,
    pub(super) outcomes: Vec<RetainedStatementOutcome>,
    pub(super) tables: Vec<RetainedTable>,
    pub(super) table_dispositions: Vec<RetainedTableDisposition>,
    pub(super) resolutions: Vec<RetainedStatementResolution>,
    pub(super) dependencies: Vec<RetainedDependencyToken>,
    pub(super) dependency_uses: Vec<RetainedStatementDependencyUse>,
    pub(super) indexes: Vec<RetainedIndexDescriptor>,
    pub(super) index_key_columns: Vec<RetainedIndexKeyColumn>,
    pub(super) transitions: Vec<RetainedTransition>,
    pub(super) key_effects: Vec<RetainedKeyEffect>,
    pub(super) key_components: Vec<RetainedKeyComponent>,
    pub(super) projections: Vec<RetainedProjectionBinding>,
    pub(super) images: Vec<DecodedTypedImage>,
}

pub(super) struct RetainedStatement {
    pub(super) statement_ordinal: u32,
    pub(super) family_ordinal: u32,
    pub(super) input_row_count: u32,
    pub(super) request_digest: Digest,
    pub(super) typed_statement_digest: Digest,
    pub(super) overlay_before: Digest,
    pub(super) overlay_after: Digest,
    /// S2 has exactly one move-only decoded record per statement.  These compact source facts
    /// let the later S7 resolution bind its record digest/length after the raw copy is dropped.
    pub(super) record_bytes: u32,
    pub(super) record_digest: Digest,
}

pub(super) struct RetainedDisposition {
    pub(super) statement_ordinal: u32,
    pub(super) source_row_ordinal: u32,
    pub(super) stable_row_id: u64,
    pub(super) disposition: u8,
    pub(super) table_ref: u32,
    pub(super) transition_ref: u32,
    pub(super) typed_statement_digest: Digest,
}

pub(super) struct RetainedSequenceEffect {
    pub(super) statement_ordinal: u32,
    pub(super) effect_ordinal: u32,
    pub(super) disposition_ref: u32,
    pub(super) flags: u8,
    pub(super) body_digest: Digest,
    pub(super) reference: crate::BinarySequenceValueReference,
}

pub(super) struct RetainedStatementOutcome {
    pub(super) statement_ordinal: u32,
    pub(super) family_ordinal: u32,
    pub(super) semantic_class: u16,
    pub(super) flags: u16,
    pub(super) typed_statement_digest: Digest,
    /// Exact S6 entry digest, recomputed at strict fill while the fixed source is borrowed.
    /// Keeping only the digest lets Q1 bind the resolution without retaining S6 bytes.
    pub(super) outcome_digest: Digest,
    pub(super) outcome: gpu_db_wal::CanonicalOutcome,
}

pub(super) struct RetainedTable {
    pub(super) table_ref: u32,
    pub(super) stable_table_id: u64,
    pub(super) display_oid: u32,
    pub(super) target_dependency_ref: u32,
    pub(super) catalog_epoch: u64,
    pub(super) data_generation_before: u64,
    pub(super) data_generation_after: u64,
    pub(super) row_allocator_before: u64,
    pub(super) row_allocator_high_water: u64,
    pub(super) initial_logical_row_count: u64,
    pub(super) final_logical_row_count: u64,
    pub(super) disposition_start: u32,
    pub(super) disposition_count: u32,
    pub(super) transition_start: u32,
    pub(super) transition_count: u32,
    pub(super) key_effect_start: u32,
    pub(super) key_effect_count: u32,
    pub(super) owned_index_start: u32,
    pub(super) owned_index_count: u32,
    pub(super) image_ref: u32,
    pub(super) catalog_column_count: u32,
    pub(super) schema_digest: Digest,
    pub(super) initial_table_root: Digest,
    pub(super) final_table_root: Digest,
    pub(super) transition_root: Digest,
    pub(super) index_effect_root: Digest,
    pub(super) image_layout_digest: Digest,
    pub(super) image_content_digest: Digest,
    /// Compact image-descriptor facts needed by the table-manifest closure. The decoded image
    /// remains the only value/vector owner; these fields never retain image-arena bytes.
    pub(super) image_arena_offset: u64,
    pub(super) image_encoded_bytes: u64,
    pub(super) image_descriptor_digest: Digest,
    pub(super) manifest_digest: Digest,
}

pub(super) struct RetainedTableDisposition {
    pub(super) table_ref: u32,
    pub(super) disposition_ref: u32,
    pub(super) stable_row_id: u64,
    pub(super) statement_ordinal: u32,
    pub(super) source_row_ordinal: u32,
    pub(super) disposition: u8,
}

pub(super) struct RetainedStatementResolution {
    pub(super) statement_ordinal: u32,
    pub(super) record_ref: u32,
    pub(super) outcome_ref: u32,
    pub(super) table_ref: u32,
    pub(super) flags: u32,
    pub(super) s4_start: u32,
    pub(super) s4_count: u32,
    pub(super) s5_start: u32,
    pub(super) s5_count: u32,
    pub(super) dependency_use_start: u32,
    pub(super) dependency_use_count: u32,
    pub(super) projection_start: u32,
    pub(super) projection_count: u32,
    pub(super) input_row_count: u32,
    pub(super) surviving_row_count: u32,
    pub(super) affected_row_count: u64,
    pub(super) dependency_validation_floor: u64,
    pub(super) record_bytes: u32,
    pub(super) terminal_dependency_ref: u32,
    pub(super) terminal_row_ordinal: u32,
    pub(super) terminal_source_ordinal: u32,
    pub(super) request_digest: Digest,
    pub(super) typed_statement_digest: Digest,
    pub(super) record_digest: Digest,
    pub(super) returning_digest: Digest,
    pub(super) overlay_before: Digest,
    pub(super) overlay_after: Digest,
    pub(super) outcome_digest: Digest,
}

pub(super) struct RetainedDependencyToken {
    pub(super) dependency_ref: u32,
    pub(super) kind: u8,
    pub(super) access: u8,
    pub(super) flags: u16,
    pub(super) stable_object_id: u64,
    pub(super) display_oid: u32,
    pub(super) target_table_ref: u32,
    pub(super) base_generation: u64,
    pub(super) snapshot_floor: u64,
    pub(super) key_effect_ref: u32,
    pub(super) descriptor_ref: u32,
    pub(super) catalog_epoch: u64,
    pub(super) schema_digest: Digest,
    pub(super) base_root: Digest,
    pub(super) name_digest: Digest,
    pub(super) identity_digest: Digest,
    /// Exact token digest. The dependency-root closure needs this digest but not token bytes.
    pub(super) token_digest: Digest,
}

pub(super) struct RetainedStatementDependencyUse {
    pub(super) statement_ordinal: u32,
    pub(super) dependency_ref: u32,
    pub(super) role: u16,
    pub(super) source_ordinal: u32,
    pub(super) transition_ref: u32,
    pub(super) key_effect_ref: u32,
}

pub(super) struct RetainedIndexDescriptor {
    pub(super) index_ref: u32,
    pub(super) owner_table_ref: u32,
    pub(super) raw_catalog_ordinal: u32,
    pub(super) stable_index_id: u64,
    pub(super) display_oid: u32,
    pub(super) stable_constraint_id: u64,
    pub(super) constraint_display_oid: u32,
    pub(super) flags: u32,
    pub(super) null_equality_policy: u8,
    pub(super) key_start: u32,
    pub(super) key_count: u32,
    /// The one S7 descriptor catalog epoch (wire offset 72).  It is the pinned owner catalog
    /// epoch, not a second independently supplied epoch.
    pub(super) catalog_epoch: u64,
    pub(super) owner_stable_table_id: u64,
    pub(super) owner_display_oid: u32,
    pub(super) owner_schema_digest: Digest,
    pub(super) owner_name_digest: Digest,
    pub(super) index_name_digest: Digest,
    pub(super) constraint_name_digest: Digest,
    pub(super) owner_table_base_root: Digest,
    pub(super) base_index_root: Digest,
    pub(super) final_index_root: Digest,
    pub(super) descriptor_digest: Digest,
    pub(super) owner_data_generation: u64,
    pub(super) base_index_generation: u64,
    pub(super) final_index_generation: u64,
}

pub(super) struct RetainedIndexKeyColumn {
    pub(super) key_column_ref: u32,
    pub(super) index_ref: u32,
    pub(super) key_ordinal: u32,
    pub(super) owner_catalog_column_ordinal: u32,
    pub(super) stable_column_id: u32,
    pub(super) owner_display_table_oid: u32,
    pub(super) attnum: i16,
    pub(super) storage: [u8; 4],
    pub(super) declared_type_oid: u32,
    pub(super) signed_type_size: i16,
    pub(super) column_name_digest: Digest,
    pub(super) key_digest: Digest,
}

pub(super) struct RetainedTransition {
    pub(super) transition_ref: u32,
    pub(super) table_ref: u32,
    pub(super) stable_row_id: u64,
    pub(super) source_disposition_ref: u32,
    pub(super) source_statement_ordinal: u32,
    pub(super) source_row_ordinal: u32,
    pub(super) image_ref: u32,
    pub(super) image_row_ordinal: u32,
    pub(super) key_effect_start: u32,
    pub(super) key_effect_count: u32,
    pub(super) final_writer_statement_ordinal: u32,
    pub(super) typed_statement_digest: Digest,
    pub(super) final_row_digest: Digest,
    pub(super) transition_digest: Digest,
}

pub(super) struct RetainedKeyEffect {
    pub(super) effect_ref: u32,
    pub(super) role: u8,
    pub(super) action: u8,
    pub(super) transition_ref: u32,
    pub(super) index_ref: u32,
    pub(super) dependency_ref: u32,
    pub(super) new_component_start: u32,
    pub(super) new_component_count: u32,
    pub(super) key_arity: u32,
    pub(super) participates: bool,
    pub(super) contains_null: bool,
    pub(super) source_catalog_ordinal: u32,
    pub(super) typed_key_digest: Digest,
    pub(super) effect_digest: Digest,
}

pub(super) struct RetainedKeyComponent {
    pub(super) component_ref: u32,
    pub(super) effect_ref: u32,
    pub(super) side: u8,
    pub(super) validity: u8,
    pub(super) component_ordinal: u32,
    pub(super) key_column_ref: u32,
    pub(super) source_catalog_ordinal: u32,
    pub(super) value_arena_offset: u64,
    pub(super) value_bytes: u32,
    pub(super) storage: [u8; 4],
    pub(super) declared_type_oid: u32,
    pub(super) signed_type_size: i16,
    /// The value authority is the selected move-only final image cell.  Its transition carries
    /// the S4 statement/row source; retaining those redundant references here would manufacture
    /// a second scalar path rather than a compact graph edge.
    pub(super) typed_value_digest: Digest,
    pub(super) component_digest: Digest,
}

pub(super) struct RetainedProjectionBinding {
    pub(super) projection_ref: u32,
    pub(super) statement_ordinal: u32,
    pub(super) projection_ordinal: u32,
    pub(super) source_catalog_ordinal: u32,
    pub(super) stable_column_id: u32,
    pub(super) table_ref: u32,
    pub(super) attnum: i16,
    pub(super) storage: [u8; 4],
    pub(super) declared_type_oid: u32,
    pub(super) signed_type_size: i16,
    /// The statement resolution selects the move-only S2 record; this is intentionally a
    /// compact reference to its catalog column instead of a duplicated source name/type shape.
    pub(super) record_ref: u32,
    /// S7 result format is not carried by the S2 RETURNING source and therefore remains this
    /// binding's own variable wire fact.
    pub(super) result_format: u16,
    /// S2's projection ordinal is an independently encoded wire fact. It need not be inferred
    /// from SQL projection order until the strict S2 closure has proved their equality.
    pub(super) s2_projection_ordinal: u32,
    pub(super) name_digest: Digest,
    pub(super) projection_digest: Digest,
}
