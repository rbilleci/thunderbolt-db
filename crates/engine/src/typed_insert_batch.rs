//! Immutable typed INSERT batches prepared off the commit sequencer.
//!
//! The batch is the one private semantic carrier for INSERT input.  It has no raw constructor,
//! no `Clone`, no predicted row identities, and no `WriteDelta`: physical plans and WAL templates
//! may only consume its already catalog-ordered vectors.

use super::*;
use crate::engine_insert_plan::host_retention::{HostRetentionGeometry, HostRetentionReport};
#[cfg(test)]
use crate::insert_semantic_ir::InsertSourceOrdinal;
use crate::insert_semantic_ir::{
    InsertStatementOrdinal, ResolvedInsertInput, ResolvedInsertSemantics,
};
use crate::rel_exec_helpers::{append_relational_cell, coerce_insert_value, RelationalCellRef};

mod builder;
mod canonical_codec;
// Shared, inert typed-vector/image authority for future codec-5 final-table images and
// retained responses.  It deliberately owns no WAL, replay, result, or publication path.
#[allow(dead_code)]
mod typed_image_codec;
mod typed_image_codec_value_contract;
#[allow(unused_imports)] // Narrow future codec-5 S2/S5 read views; no live replay caller yet.
pub(crate) use canonical_codec::{
    CanonicalTypedInsertRecordPrefix, DecodedReturningLayoutFacts, DecodedSequenceEffectFacts,
    DecodedSequenceEffectKindFacts, DecodedSequenceParentFacts, DecodedSequenceRequestFacts,
    DecodedTypedInsertRecord, DecodedTypedInsertRecordFacts, DecodedTypedInsertTargetFacts,
    CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES,
};
mod constraint_source;
mod defaults;
mod resident_source;
mod resident_source_retention;
mod returning;
mod semantics;
pub(crate) mod sequence_defaults;
pub(crate) use constraint_source::TypedInsertConstraintDeviceSource;
#[cfg(test)]
pub(crate) use constraint_source::{
    constraint_source_upload_count, reset_constraint_source_upload_count,
};
pub(crate) use resident_source::{
    PreparedResidentAppendColumn, PreparedResidentAppendSource, PreparedResidentDensePayload,
    PreparedResidentFixedBoolUpload, PreparedResidentFixedChunk, PreparedResidentFixedChunkOwners,
};
#[allow(unused_imports)] // Re-exported for the next production-compiled effect-plan handoff.
pub(crate) use returning::{InsertReturningEffectShape, ReturningProjectionEffectIdentity};
#[allow(unused_imports)]
pub(crate) use semantics::{
    prepare_typed_insert_semantics, prepare_typed_insert_semantics_at, PreparedTypedInsert,
    PreparedTypedInsertEffectTarget,
};
#[allow(unused_imports)] // Re-exported for the production-compiled sequence-effect classifier.
pub(crate) use sequence_defaults::SequenceDefaultRequestEffectShape;

/// Scalar-only geometry for new fixed append owners.  It contains no backing identity and is
/// intentionally computed without building a temporary chunk vector, so capacity sizing cannot
/// perturb the host peak it measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FixedAppendHostOwnerGeometry {
    pub(crate) chunk_count: usize,
    pub(crate) chunk_payload_bytes: u64,
    pub(crate) int4_count: usize,
    pub(crate) bool_count: usize,
}

/// Shared logical RETURNING digest for inert pre-WAL evidence. This is metadata-only and has no
/// projection buffer, result route, or physical offset authority.
#[allow(dead_code)] // The inert test terminal is the only current caller.
pub(crate) fn canonical_returning_layout_digest(
    prepared: &PreparedTypedInsert,
) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
    prepared.canonical_returning_layout_digest()
}

/// Exact byte length of the inert typed-INSERT v1 logical record. The codec itself owns the
/// traversal and validation; this narrow facade exposes only its bounded count for pre-WAL
/// resource planning, never a raw encoder or a second logical format.
#[allow(dead_code)] // Adopted by the inert pre-WAL reservation footprint owner.
pub(crate) fn canonical_typed_insert_encoded_len(
    batch: &TypedInsertBatch,
) -> Result<usize, EngineError> {
    canonical_codec::encoded_len(batch)
}

/// Validate the bounded fixed prefix of one canonical typed-INSERT record.  Aggregate framing
/// uses this single authority for the format, length, and logical-result digests before it ever
/// asks the full strict decoder to retain a record.
#[allow(dead_code)] // The aggregate owner consumes this inert codec primitive in a later slice.
pub(crate) fn parse_canonical_typed_insert_record_prefix(
    header: &[u8; CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES],
    record_bytes: usize,
) -> Result<CanonicalTypedInsertRecordPrefix, EngineError> {
    canonical_codec::parse_canonical_typed_insert_record_prefix(header, record_bytes)
}

/// Strict, model-owning typed-INSERT v1 decoder for the future aggregate S2/S5 closure. Its
/// result exposes only scalar, borrowed facts and cannot be converted into a live write carrier.
#[allow(dead_code)] // Reserved for the inert codec-5 S2/S5 closure, never a live write path.
pub(crate) fn decode_canonical_typed_insert_record(
    bytes: &[u8],
) -> Result<DecodedTypedInsertRecord, EngineError> {
    canonical_codec::decode(bytes)
}

/// Test-only fixture bridge for aggregate S2 materialization.  Production code has no raw
/// canonical-record encoder facade; the codec-5 source-owner tests use this solely to construct
/// a strict record that the same model-owning decoder then consumes.
#[cfg(test)]
pub(crate) fn encode_canonical_typed_insert_record_for_test(
    batch: &TypedInsertBatch,
) -> Result<Vec<u8>, EngineError> {
    canonical_codec::encode(batch)
}

/// Test-only bridge to the canonical private sequence-chain fixture. The aggregate S5 tests use
/// this exact existing codec fixture rather than inventing a second private-effect encoder.
#[cfg(test)]
pub(crate) fn private_sequence_chain_record_for_test(
    statement_ordinal: InsertStatementOrdinal,
    parent_txn_id: TxnId,
) -> Result<Vec<u8>, EngineError> {
    let batch = canonical_codec::sequence_tests::private_chain_batch_for_test(
        statement_ordinal,
        parent_txn_id,
    );
    canonical_codec::encode(&batch)
}

/// Bind legacy INSERT RETURNING before default lowering without constructing a result frame.
///
/// The resident-append adapter declines RETURNING before semantic lowering, but the legacy
/// execution path must retain semantic diagnostic order: supplied-value coercion, then RETURNING
/// binding, then scalar default evaluation.
pub(crate) fn validate_legacy_insert_returning(
    table: &RelationalTable,
    returning: &[String],
    row_count: usize,
) -> Result<(), EngineError> {
    let row_count = u32::try_from(row_count)
        .map_err(|_| EngineError::Durability("typed INSERT row count exceeds u32".to_string()))?;
    returning::bind(table, returning, row_count).map(|_| ())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TypedInsertBatchTable {
    schema: Arc<str>,
    name: Arc<str>,
    oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
    prepared_catalog_seq: Index,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TypedInsertDependencyBinding {
    schema: Arc<str>,
    name: Arc<str>,
    oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
}

/// One catalog-order column. `column_id` and `attnum`, rather than a parsed column-list position,
/// are the stable identity carried toward later plan compilation.
struct TypedInsertColumn {
    name: Arc<str>,
    column_id: u32,
    attnum: i16,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
    /// `None` represents an omitted target column. When present, the complete source identity
    /// is derived from this ordinal plus `TypedInsertBatch::statement_ordinal` and the row index;
    /// it must never be repeated in every row's semantic metadata.
    source_column_ordinal: Option<u32>,
    /// First-occurrence ordinal into `TypedInsertBatch::domain_dependencies`.  This binds a
    /// column's catalog domain without a later catalog scan or string-based rediscovery.
    domain_dependency_ordinal: Option<u32>,
    validity: TypedInsertColumnValidity,
    presence: TypedInsertColumnPresence,
    input_states: Box<[TypedInsertInputState]>,
    input_provenance: Box<[TypedInsertInputProvenance]>,
    /// Physical output records whether an original omitted/DEFAULT cell was deterministically
    /// resolved. The original state/provenance above remain the semantic audit authority.
    default_resolution: TypedInsertDefaultResolution,
    values: TypedInsertColumnValues,
    /// Full-vector validation is a seal-time boundary, never row-template work. The test-only
    /// counter keeps that O(columns) ownership boundary from regressing into an O(rows^2) path.
    #[cfg(test)]
    full_invariant_scans: std::sync::atomic::AtomicUsize,
}

/// The sealed pre-default meaning of every catalog-order input cell. `presence` and `validity`
/// remain compact physical compatibility views derived during lowering; this state map is the
/// semantic authority that prevents DEFAULT/omission/NULL collapse before later plan operators.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TypedInsertInputState {
    Provided,
    ProvidedNull,
    Omitted,
    ExplicitDefault,
}

/// Compact per-cell source authority. Position is deliberately absent: statement/row/source
/// column identity is reconstructed from the batch, vector index, and column metadata.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TypedInsertInputProvenance {
    Omitted,
    Literal,
    BoundParameter { index: u32 },
    ProgrammaticValue,
    SqlDefault,
    ProgrammaticDefault,
}

/// Device-ready value vectors. An invalid entry always retains its zero placeholder; validity is
/// the only authority for whether that placeholder denotes SQL NULL.
enum TypedInsertColumnValues {
    /// `int2`, `int4`, and `date` share their existing i32 storage domain.
    I32(Box<[i32]>),
    /// `int8` and `timestamp` share their existing i64 storage domain.
    I64(Box<[i64]>),
    /// Numeric stores the already coerced fixed-point mantissa; scale lives in `SqlType`.
    I128(Box<[i128]>),
    /// UUID is its original 16 wire-order bytes, not a host-endian integer interpretation.
    Bytes16(Box<[[u8; 16]]>),
    /// Bool is a one-bit-per-row vector, LSB-first, with an exact zero-tailed word count.
    BoolBits(Box<[u32]>),
    /// Text is one contiguous UTF-8 byte vector and N+1 byte offsets. Invalid rows repeat the
    /// preceding offset, so an empty string remains distinct from NULL through validity.
    Text {
        offsets: Box<[u64]>,
        bytes: Box<[u8]>,
    },
}

/// SQL validity: bit `row % 32` in word `row / 32` is one for a non-NULL input. Bitmap tails are
/// always zero, including the final word of a 33-row vector.
enum TypedInsertColumnValidity {
    AllValid,
    Bitmap(Box<[u32]>),
}

/// Physical output presence is independent from validity: supplied SQL NULL is present+invalid.
/// The original source presence remains in `input_states`; the default resolver changes this
/// compatibility view to all-provided only after it has materialized every omitted/DEFAULT cell.
enum TypedInsertColumnPresence {
    AllProvided,
    Bitmap(Box<[u32]>),
}

/// Compact proof that every original default-bearing cell reached a deterministic physical
/// output. `AllDirect` stores no row bitmap for an all-supplied column.
enum TypedInsertDefaultResolution {
    AllDirect,
    Bitmap(Box<[u32]>),
}

/// One stable metadata-only domain dependency. The prepared device plan takes ownership of
/// these bindings with the sealed batch and revalidates them before it binds row identities.
pub(crate) struct TypedInsertDomainBinding {
    pub(super) schema: Arc<str>,
    pub(super) name: Arc<str>,
    pub(super) oid: u32,
    pub(super) base_type: SqlType,
}

/// Immutable raw-catalog closure retained solely for canonical typed-INSERT evidence.  It is
/// captured during semantic preparation from the same admitted snapshot as the vectors; no
/// codec path may discover indexes or foreign keys by consulting a later catalog.
struct TypedInsertCanonicalCatalog {
    indexes: Box<[TypedInsertCanonicalIndexBinding]>,
    foreign_keys: Box<[TypedInsertCanonicalForeignKeyBinding]>,
}

#[derive(Clone)]
struct TypedInsertCanonicalColumnBinding {
    dependency_ordinal: u32,
    catalog_column_ordinal: u32,
    column_id: u32,
    attnum: i16,
    name: Arc<str>,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
}

struct TypedInsertCanonicalIndexBinding {
    owner_dependency_ordinal: u32,
    raw_ordinal: u32,
    oid: u32,
    name: Arc<str>,
    table_name: Arc<str>,
    first_column_name: Arc<str>,
    key_columns: Box<[TypedInsertCanonicalColumnBinding]>,
    unique: bool,
    primary_key: bool,
    unique_constraint: bool,
}

struct TypedInsertCanonicalForeignKeyBinding {
    raw_ordinal: u32,
    name: Arc<str>,
    child_column_name: Arc<str>,
    referenced_table_name: Arc<str>,
    referenced_column_name: Arc<str>,
    child_column: TypedInsertCanonicalColumnBinding,
    parent_dependency_ordinal: u32,
    parent_column: TypedInsertCanonicalColumnBinding,
    supporting_index: TypedInsertCanonicalIndexBinding,
}

impl TypedInsertBatchTable {
    fn append_host_retention(&self, report: &mut HostRetentionReport) -> Result<(), EngineError> {
        report.retain_arc_str(&self.schema)?;
        report.retain_arc_str(&self.name)
    }
}

impl TypedInsertDependencyBinding {
    fn append_host_retention(&self, report: &mut HostRetentionReport) -> Result<(), EngineError> {
        report.retain_arc_str(&self.schema)?;
        report.retain_arc_str(&self.name)
    }
}

#[allow(dead_code)] // Read by the inert host-retention owner before its reservation caller exists.
impl TypedInsertDomainBinding {
    fn append_host_retention(&self, report: &mut HostRetentionReport) -> Result<(), EngineError> {
        report.retain_arc_str(&self.schema)?;
        report.retain_arc_str(&self.name)
    }
}

#[allow(dead_code)] // Read by the inert host-retention owner before its reservation caller exists.
impl TypedInsertColumn {
    fn append_host_retention(&self, report: &mut HostRetentionReport) -> Result<(), EngineError> {
        report.retain_arc_str(&self.name)?;
        report.retain_boxed_slice(&self.input_states)?;
        report.retain_boxed_slice(&self.input_provenance)?;
        self.validity.append_host_retention(report)?;
        self.presence.append_host_retention(report)?;
        self.default_resolution.append_host_retention(report)?;
        self.values.append_host_retention(report)
    }

    fn append_resident_source_host_retention(
        &self,
        report: &mut HostRetentionReport,
    ) -> Result<(), EngineError> {
        self.validity.append_host_retention(report)?;
        self.values.append_host_retention(report)
    }
}

impl TypedInsertColumnValues {
    fn append_host_retention(&self, report: &mut HostRetentionReport) -> Result<(), EngineError> {
        match self {
            Self::I32(values) => report.retain_boxed_slice(values),
            Self::I64(values) => report.retain_boxed_slice(values),
            Self::I128(values) => report.retain_boxed_slice(values),
            Self::Bytes16(values) => report.retain_boxed_slice(values),
            Self::BoolBits(values) => report.retain_boxed_slice(values),
            Self::Text { offsets, bytes } => {
                report.retain_boxed_slice(offsets)?;
                report.retain_boxed_slice(bytes)
            }
        }
    }
}

impl TypedInsertColumnValidity {
    fn append_host_retention(&self, report: &mut HostRetentionReport) -> Result<(), EngineError> {
        match self {
            Self::AllValid => Ok(()),
            Self::Bitmap(words) => report.retain_boxed_slice(words),
        }
    }
}

#[allow(dead_code)] // Read by the inert host-retention owner before its reservation caller exists.
impl TypedInsertColumnPresence {
    fn append_host_retention(&self, report: &mut HostRetentionReport) -> Result<(), EngineError> {
        match self {
            Self::AllProvided => Ok(()),
            Self::Bitmap(words) => report.retain_boxed_slice(words),
        }
    }
}

#[allow(dead_code)] // Read by the inert host-retention owner before its reservation caller exists.
impl TypedInsertDefaultResolution {
    fn append_host_retention(&self, report: &mut HostRetentionReport) -> Result<(), EngineError> {
        match self {
            Self::AllDirect => Ok(()),
            Self::Bitmap(words) => report.retain_boxed_slice(words),
        }
    }
}

#[allow(dead_code)] // Read by the inert host-retention owner before its reservation caller exists.
impl TypedInsertCanonicalColumnBinding {
    fn append_host_retention(&self, report: &mut HostRetentionReport) -> Result<(), EngineError> {
        report.retain_arc_str(&self.name)
    }
}

#[allow(dead_code)] // Read by the inert host-retention owner before its reservation caller exists.
impl TypedInsertCanonicalIndexBinding {
    fn append_host_retention(&self, report: &mut HostRetentionReport) -> Result<(), EngineError> {
        report.retain_arc_str(&self.name)?;
        report.retain_arc_str(&self.table_name)?;
        report.retain_arc_str(&self.first_column_name)?;
        report.retain_boxed_slice(&self.key_columns)?;
        for column in self.key_columns.iter() {
            column.append_host_retention(report)?;
        }
        Ok(())
    }
}

#[allow(dead_code)] // Read by the inert host-retention owner before its reservation caller exists.
impl TypedInsertCanonicalForeignKeyBinding {
    fn append_host_retention(&self, report: &mut HostRetentionReport) -> Result<(), EngineError> {
        report.retain_arc_str(&self.name)?;
        report.retain_arc_str(&self.child_column_name)?;
        report.retain_arc_str(&self.referenced_table_name)?;
        report.retain_arc_str(&self.referenced_column_name)?;
        self.child_column.append_host_retention(report)?;
        self.parent_column.append_host_retention(report)?;
        self.supporting_index.append_host_retention(report)
    }
}

#[allow(dead_code)] // Read by the inert host-retention owner before its reservation caller exists.
impl TypedInsertCanonicalCatalog {
    fn append_host_retention(&self, report: &mut HostRetentionReport) -> Result<(), EngineError> {
        report.retain_boxed_slice(&self.indexes)?;
        for index in self.indexes.iter() {
            index.append_host_retention(report)?;
        }
        report.retain_boxed_slice(&self.foreign_keys)?;
        for foreign_key in self.foreign_keys.iter() {
            foreign_key.append_host_retention(report)?;
        }
        Ok(())
    }
}

/// A private builder may decline a shape before any batch, WAL, or device plan exists.  This keeps
/// future feature work explicit rather than silently reinterpreting an unsupported input.
#[derive(Debug, PartialEq, Eq)]
enum TypedInsertDeferred {
    CatalogGeneration,
    Constraints,
    /// Semantic lowering succeeded, but the current resident-append compiler deliberately has
    /// no physical owner for this shape yet.  In particular, RETURNING and stateful sequence
    /// defaults are kept as prepared semantic data rather than reclassified as parse failures.
    PhysicalUnsupported,
}

#[allow(clippy::large_enum_variant)]
enum TypedInsertBuildResult {
    Ready(TypedInsertBatch),
    Deferred(TypedInsertDeferred),
}

/// The general builder has one final-shaped representation. The direct capability exists only to
/// preserve the device compiler's fixed-width eligibility; it is not another semantic
/// builder or a second SQL interpretation authority.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TypedInsertBuildCapability {
    ResidentAppend,
    #[cfg(test)]
    SemanticOnly,
    /// Exercises the real pre-WAL composite proof for indexed tables without widening the
    /// production resident-append eligibility gate.
    #[cfg(test)]
    ProofOnly,
    /// Exercises the inert current-generation FK proof without widening either live or indexed
    /// proof-only eligibility.
    #[cfg(test)]
    ForeignKeyProofOnly,
}

impl TypedInsertColumnValues {
    fn zeroed(ty: SqlType, rows: usize) -> Result<Self, EngineError> {
        let words = bitmap_words(rows)?;
        Ok(match ty {
            SqlType::Int2 | SqlType::Int4 | SqlType::Date => Self::I32(vec![0; rows].into()),
            SqlType::Int8 | SqlType::Timestamp => Self::I64(vec![0; rows].into()),
            SqlType::Numeric { .. } => Self::I128(vec![0; rows].into()),
            SqlType::Uuid => Self::Bytes16(vec![[0; 16]; rows].into()),
            SqlType::Bool => Self::BoolBits(vec![0; words].into()),
            SqlType::Text => Self::Text {
                offsets: vec![
                    0;
                    rows.checked_add(1).ok_or_else(|| {
                        EngineError::Durability(
                            "typed INSERT text offset count overflows".to_string(),
                        )
                    })?
                ]
                .into(),
                bytes: Box::default(),
            },
        })
    }

    fn len(&self) -> usize {
        match self {
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::I128(values) => values.len(),
            Self::Bytes16(values) => values.len(),
            Self::BoolBits(_) => 0,
            Self::Text { offsets, .. } => offsets.len().saturating_sub(1),
        }
    }

    fn rows_match(&self, rows: usize) -> bool {
        match self {
            Self::BoolBits(words) => bitmap_shape_is_exact(words, rows),
            _ => self.len() == rows,
        }
    }

    #[cfg(test)]
    fn as_i32(&self) -> Option<&[i32]> {
        match self {
            Self::I32(values) => Some(values),
            _ => None,
        }
    }

    #[cfg(test)]
    fn byte_len(&self) -> usize {
        match self {
            Self::I32(values) => values.len() * std::mem::size_of::<i32>(),
            Self::I64(values) => values.len() * std::mem::size_of::<i64>(),
            Self::I128(values) => values.len() * std::mem::size_of::<i128>(),
            Self::Bytes16(values) => values.len() * std::mem::size_of::<[u8; 16]>(),
            Self::BoolBits(words) => words.len() * std::mem::size_of::<u32>(),
            Self::Text { offsets, bytes } => {
                offsets.len() * std::mem::size_of::<u64>() + bytes.len()
            }
        }
    }

    fn text_invariants_hold(&self, rows: usize) -> bool {
        let Self::Text { offsets, bytes } = self else {
            return true;
        };
        offsets.len() == rows.saturating_add(1)
            && offsets.first() == Some(&0)
            && offsets
                .windows(2)
                .all(|window| window[0] <= window[1] && usize::try_from(window[1]).is_ok())
            && offsets
                .last()
                .and_then(|offset| usize::try_from(*offset).ok())
                == Some(bytes.len())
            && std::str::from_utf8(bytes).is_ok()
    }

    /// Bounded metadata check for a borrowed physical upload.  Full offset monotonicity and UTF-8
    /// validation are seal-time invariants; the pre-WAL CHECK source must not rescan text values
    /// on the host before giving the GPU the operator input.
    fn text_extent_matches(&self, rows: usize) -> bool {
        let Self::Text { offsets, bytes } = self else {
            return true;
        };
        offsets.len() == rows.saturating_add(1)
            && offsets.first() == Some(&0)
            && offsets
                .last()
                .and_then(|offset| usize::try_from(*offset).ok())
                == Some(bytes.len())
    }
}

impl TypedInsertColumnValidity {
    fn is_valid(&self, row: usize) -> bool {
        match self {
            Self::AllValid => true,
            Self::Bitmap(words) => bit_is_set(words, row),
        }
    }

    fn shape_is_exact(&self, rows: usize) -> bool {
        match self {
            Self::AllValid => true,
            Self::Bitmap(words) => bitmap_shape_is_exact(words, rows),
        }
    }

    fn is_all_valid(&self) -> bool {
        matches!(self, Self::AllValid)
    }

    fn bitmap_words(&self) -> Option<&[u32]> {
        match self {
            Self::AllValid => None,
            Self::Bitmap(words) => Some(words),
        }
    }
}

impl TypedInsertColumnPresence {
    fn is_provided(&self, row: usize) -> bool {
        match self {
            Self::AllProvided => true,
            Self::Bitmap(words) => bit_is_set(words, row),
        }
    }

    #[cfg(test)]
    fn all_provided(&self, rows: usize) -> bool {
        match self {
            Self::AllProvided => true,
            Self::Bitmap(words) => bitmap_is_all_set(words, rows),
        }
    }

    fn shape_is_exact(&self, rows: usize) -> bool {
        match self {
            Self::AllProvided => true,
            Self::Bitmap(words) => bitmap_shape_is_exact(words, rows),
        }
    }
}

impl TypedInsertDefaultResolution {
    fn was_defaulted(&self, row: usize) -> bool {
        match self {
            Self::AllDirect => false,
            Self::Bitmap(words) => bit_is_set(words, row),
        }
    }

    fn shape_is_exact(&self, rows: usize) -> bool {
        match self {
            Self::AllDirect => true,
            Self::Bitmap(words) => bitmap_shape_is_exact(words, rows),
        }
    }
}

impl TypedInsertColumn {
    #[cfg(test)]
    fn is_all_valid_i32(&self, row_count: usize) -> bool {
        matches!(self.validity, TypedInsertColumnValidity::AllValid)
            && matches!(self.presence, TypedInsertColumnPresence::AllProvided)
            && self.all_inputs_are_provided(row_count)
            && self.ty == SqlType::Int4
            && self.values.rows_match(row_count)
            && self.values.as_i32().is_some()
    }

    fn is_resident_append_vector(&self, row_count: usize) -> bool {
        matches!(self.presence, TypedInsertColumnPresence::AllProvided)
            && self.all_inputs_are_resolved(row_count)
            && is_live_resident_append_type(self.ty)
            && self.values.rows_match(row_count)
    }

    #[cfg(test)]
    fn all_inputs_are_provided(&self, rows: usize) -> bool {
        self.input_states.len() == rows
            && self.input_provenance.len() == rows
            && self
                .input_states
                .iter()
                .all(|state| *state == TypedInsertInputState::Provided)
    }

    fn all_inputs_are_resolved(&self, rows: usize) -> bool {
        self.input_states.len() == rows
            && self.input_provenance.len() == rows
            && self.default_resolution.shape_is_exact(rows)
            && self
                .input_states
                .iter()
                .enumerate()
                .all(|(row, state)| match state {
                    TypedInsertInputState::Provided | TypedInsertInputState::ProvidedNull => {
                        !self.default_resolution.was_defaulted(row)
                    }
                    TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault => {
                        self.default_resolution.was_defaulted(row)
                    }
                })
    }

    #[cfg(test)]
    fn source_ordinal(
        &self,
        statement: InsertStatementOrdinal,
        row: u32,
    ) -> Option<InsertSourceOrdinal> {
        self.source_column_ordinal
            .map(|column| InsertSourceOrdinal {
                statement,
                row,
                column,
            })
    }

    /// Complete vector validation belongs to batch sealing. It may scan every semantic cell and
    /// text offset exactly once because the private, immutable batch has no later mutator.
    fn full_invariants_hold(&self, rows: usize) -> bool {
        #[cfg(test)]
        self.full_invariant_scans
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.values.rows_match(rows)
            && self.values.text_invariants_hold(rows)
            && self.validity.shape_is_exact(rows)
            && self.presence.shape_is_exact(rows)
            && self.default_resolution.shape_is_exact(rows)
            && self.input_states.len() == rows
            && self.input_provenance.len() == rows
            && self
                .input_states
                .iter()
                .zip(&self.input_provenance)
                .enumerate()
                .all(|(row, (state, provenance))| match state {
                    TypedInsertInputState::Provided | TypedInsertInputState::ProvidedNull => {
                        matches!(
                            provenance,
                            TypedInsertInputProvenance::Literal
                                | TypedInsertInputProvenance::BoundParameter { .. }
                                | TypedInsertInputProvenance::ProgrammaticValue
                        ) && self.source_column_ordinal.is_some()
                            && self.presence.is_provided(row)
                            && !self.default_resolution.was_defaulted(row)
                            && (self.validity.is_valid(row)
                                == (*state == TypedInsertInputState::Provided))
                    }
                    TypedInsertInputState::Omitted => {
                        *provenance == TypedInsertInputProvenance::Omitted
                            && self.source_column_ordinal.is_none()
                            && self.presence.is_provided(row)
                            && self.default_resolution.was_defaulted(row)
                    }
                    TypedInsertInputState::ExplicitDefault => {
                        matches!(
                            provenance,
                            TypedInsertInputProvenance::SqlDefault
                                | TypedInsertInputProvenance::ProgrammaticDefault
                        ) && self.source_column_ordinal.is_some()
                            && self.presence.is_provided(row)
                            && self.default_resolution.was_defaulted(row)
                    }
                })
    }

    /// Constant-time local safety check for one WAL-template cell. The full vector relationship
    /// was established at seal time; re-walking all rows here would make row encoding quadratic.
    fn row_invariants_hold(&self, row: usize, rows: usize) -> bool {
        self.values.rows_match(rows)
            && self.validity.shape_is_exact(rows)
            && self.presence.shape_is_exact(rows)
            && self.default_resolution.shape_is_exact(rows)
            && self.input_states.len() == rows
            && self.input_provenance.len() == rows
            && row < rows
            && self
                .input_states
                .get(row)
                .zip(self.input_provenance.get(row))
                .is_some_and(|(state, provenance)| match state {
                    TypedInsertInputState::Provided | TypedInsertInputState::ProvidedNull => {
                        matches!(
                            provenance,
                            TypedInsertInputProvenance::Literal
                                | TypedInsertInputProvenance::BoundParameter { .. }
                                | TypedInsertInputProvenance::ProgrammaticValue
                        ) && self.source_column_ordinal.is_some()
                            && self.presence.is_provided(row)
                            && !self.default_resolution.was_defaulted(row)
                            && (self.validity.is_valid(row)
                                == (*state == TypedInsertInputState::Provided))
                    }
                    TypedInsertInputState::Omitted => {
                        *provenance == TypedInsertInputProvenance::Omitted
                            && self.source_column_ordinal.is_none()
                            && self.presence.is_provided(row)
                            && self.default_resolution.was_defaulted(row)
                    }
                    TypedInsertInputState::ExplicitDefault => {
                        matches!(
                            provenance,
                            TypedInsertInputProvenance::SqlDefault
                                | TypedInsertInputProvenance::ProgrammaticDefault
                        ) && self.source_column_ordinal.is_some()
                            && self.presence.is_provided(row)
                            && self.default_resolution.was_defaulted(row)
                    }
                })
    }

    #[cfg(test)]
    fn full_invariant_scan_count(&self) -> usize {
        self.full_invariant_scans
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn append_relational_cell(
        &self,
        row: usize,
        rows: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), EngineError> {
        if !self.row_invariants_hold(row, rows) {
            return Err(EngineError::Durability(
                "sealed typed INSERT column invariants are invalid".to_string(),
            ));
        }
        match self.input_states[row] {
            TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault
                if !self.default_resolution.was_defaulted(row) =>
            {
                return Err(EngineError::Durability(
                    "unresolved INSERT default state cannot enter a scalar WAL template"
                        .to_string(),
                ));
            }
            TypedInsertInputState::Provided
            | TypedInsertInputState::ProvidedNull
            | TypedInsertInputState::Omitted
            | TypedInsertInputState::ExplicitDefault => {}
        }
        if !self.validity.is_valid(row) {
            append_relational_cell(out, RelationalCellRef::Null);
            return Ok(());
        }
        let cell = match (&self.values, self.ty) {
            (TypedInsertColumnValues::I32(values), SqlType::Int2) => {
                RelationalCellRef::Int2(i16::try_from(values[row]).map_err(|_| {
                    EngineError::Durability("typed int2 vector is out of range".to_string())
                })?)
            }
            (TypedInsertColumnValues::I32(values), SqlType::Int4) => {
                RelationalCellRef::Int4(values[row])
            }
            (TypedInsertColumnValues::I32(values), SqlType::Date) => {
                RelationalCellRef::Date(values[row])
            }
            (TypedInsertColumnValues::I64(values), SqlType::Int8) => {
                RelationalCellRef::Int8(values[row])
            }
            (TypedInsertColumnValues::I64(values), SqlType::Timestamp) => {
                RelationalCellRef::Timestamp(values[row])
            }
            (TypedInsertColumnValues::I128(values), SqlType::Numeric { scale, .. }) => {
                RelationalCellRef::Numeric {
                    mantissa: values[row],
                    scale,
                }
            }
            (TypedInsertColumnValues::Bytes16(values), SqlType::Uuid) => {
                RelationalCellRef::Uuid(&values[row])
            }
            (TypedInsertColumnValues::BoolBits(words), SqlType::Bool) => {
                RelationalCellRef::Bool(bit_is_set(words, row))
            }
            (TypedInsertColumnValues::Text { offsets, bytes }, SqlType::Text) => {
                let start = usize::try_from(offsets[row]).map_err(|_| {
                    EngineError::Durability("typed INSERT text start offset overflows".to_string())
                })?;
                let end = usize::try_from(offsets[row + 1]).map_err(|_| {
                    EngineError::Durability("typed INSERT text end offset overflows".to_string())
                })?;
                let value = std::str::from_utf8(bytes.get(start..end).ok_or_else(|| {
                    EngineError::Durability("typed INSERT text offsets are invalid".to_string())
                })?)
                .map_err(|_| {
                    EngineError::Durability("typed INSERT text is not UTF-8".to_string())
                })?;
                RelationalCellRef::Text(value)
            }
            _ => {
                return Err(EngineError::Durability(
                    "typed INSERT column type and vector arm disagree".to_string(),
                ));
            }
        };
        append_relational_cell(out, cell);
        Ok(())
    }
}

/// The one sealed semantic authority for an already-authoritative off-lock INSERT preparation.
pub(crate) struct TypedInsertBatch {
    table: TypedInsertBatchTable,
    statement_ordinal: InsertStatementOrdinal,
    row_count: u32,
    columns: Box<[TypedInsertColumn]>,
    dependencies: Box<[TypedInsertDependencyBinding]>,
    domain_dependencies: Box<[TypedInsertDomainBinding]>,
    /// Complete raw-order index/FK closure from semantic preparation, retained as inert logical
    /// metadata only. It has no physical planner, resident handle, or apply authority.
    canonical_catalog: TypedInsertCanonicalCatalog,
    /// Domain `GPUDBTYPEDINSERTSTATEMENT1`, over pre-effect logical intent only.
    typed_statement_digest: gpu_db_wal::CanonicalDigest,
    /// Bound output identities and exact result geometry. This remains metadata-only until a
    /// typed DML result route owns projection; resident append deliberately rejects it today.
    returning: returning::BoundInsertReturning,
    /// Move-only sequence effects consumed at semantic seal time.  The current live adapter
    /// declines sequence-default batches before this carrier is produced, but retaining the
    /// exact bindings here keeps future typed WAL ownership explicit rather than reconstructing
    /// them from materialized scalar vectors.
    sequence_bindings: Box<[sequence_defaults::SequenceDefaultBinding]>,
    /// The indexed resident-key path remains a test-only proof seam.  Keeping the marker on the
    /// sealed carrier prevents the live ResidentAppend builder from accidentally taking an
    /// indexed pre-WAL branch merely because a later catalog acquired an index.
    #[cfg(test)]
    proof_only_indexed_constraints: bool,
    #[cfg(test)]
    foreign_key_proof_only: bool,
}

impl TypedInsertBatch {
    #[cfg(test)]
    pub(crate) fn is_proof_only_indexed_constraints(&self) -> bool {
        self.proof_only_indexed_constraints
    }
    #[cfg(test)]
    pub(crate) fn is_foreign_key_proof_only(&self) -> bool {
        self.foreign_key_proof_only
    }
    #[cfg(test)]
    fn value_bytes(&self) -> usize {
        self.columns
            .iter()
            .map(|column| column.values.byte_len())
            .sum()
    }

    /// Every backing allocation retained by this sealed semantic owner right now.
    ///
    /// Sequence effects and bound terminal response metadata deliberately remain outside the
    /// host-plan domain. They receive their own reservation domains once an owner for their
    /// stateful/result lifetimes exists.
    #[allow(dead_code)] // Adopted by the inert reservation carrier next.
    pub(crate) fn host_retention_report(&self) -> Result<HostRetentionReport, EngineError> {
        let mut report = HostRetentionReport::default();
        self.table.append_host_retention(&mut report)?;
        report.retain_boxed_slice(&self.columns)?;
        for column in self.columns.iter() {
            column.append_host_retention(&mut report)?;
        }
        report.retain_boxed_slice(&self.dependencies)?;
        for dependency in self.dependencies.iter() {
            dependency.append_host_retention(&mut report)?;
        }
        report.retain_boxed_slice(&self.domain_dependencies)?;
        for dependency in self.domain_dependencies.iter() {
            dependency.append_host_retention(&mut report)?;
        }
        self.canonical_catalog.append_host_retention(&mut report)?;
        Ok(report)
    }

    /// Predict the exact host retention geometry of the resident source before its move-only
    /// materialization. The source's new outer column box is included as one future backing
    /// allocation; semantic-only vectors deliberately are not.
    #[allow(dead_code)] // Compared by the materialized reservation owner next.
    pub(crate) fn resident_append_source_host_retention_prediction(
        &self,
    ) -> Result<HostRetentionGeometry, EngineError> {
        resident_source_retention::prediction(self)
    }

    pub(super) fn binary_insert_template(
        &self,
    ) -> Result<crate::wal_binary::PreparedBinaryInsertTemplate, EngineError> {
        crate::wal_binary::PreparedBinaryInsertTemplate::from_sealed_batch(self)
    }

    pub(crate) fn binary_insert_template_table_name(&self) -> &str {
        &self.table.name
    }

    /// Stable catalog identity carried into the sole live typed INSERT plan.
    pub(crate) fn table_oid(&self) -> u32 {
        self.table.oid
    }

    pub(crate) fn binary_insert_template_row_count(&self) -> u32 {
        self.row_count
    }

    /// Borrow the sealed catalog-order vectors as one short-lived physical device relation for a
    /// row-local operator.  The derivative is deliberately not retained by the batch or plan:
    /// vector ownership remains this batch's semantic and WAL authority until commit binding.
    pub(crate) fn row_local_constraint_device_source(
        &self,
        engine: &Engine,
        table: &RelationalTable,
    ) -> Result<TypedInsertConstraintDeviceSource, ExecuteError> {
        constraint_source::build(self, engine, table)
    }

    /// Exact raw source bytes for the short-lived row-local operator.  Scratch consumers add
    /// their pooled-mask geometry before opening an allocation scope, so the whole peak is
    /// rejected before the first device allocation.
    pub(crate) fn row_local_constraint_device_payload_bytes(&self) -> Result<u64, EngineError> {
        constraint_source::payload_bytes(self)
    }

    /// Report only the sealed validity representation for one bound physical column.  This is
    /// geometry metadata: it neither reads a bitmap word nor derives a row-level verdict.
    pub(crate) fn row_local_constraint_column_has_validity_bitmap(
        &self,
        column_id: u32,
    ) -> Result<bool, EngineError> {
        self.columns
            .iter()
            .find(|column| column.column_id == column_id)
            .map(|column| matches!(column.validity, TypedInsertColumnValidity::Bitmap(_)))
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "sealed typed INSERT constraint column id {column_id} is absent"
                ))
            })
    }

    /// Exact target binding copied into an unforgeable pre-WAL proof.  Keeping this accessor
    /// narrow prevents physical consumers from discovering or reusing logical values.
    pub(crate) fn row_local_constraint_target(&self) -> (u32, gpu_db_wal::CanonicalDigest, Index) {
        (
            self.table.oid,
            self.table.schema_digest,
            self.table.prepared_catalog_seq,
        )
    }

    /// Domain bindings are semantic metadata, not resident physical dependencies. Move them to
    /// the prepared device plan with the vectors so the plan can revalidate them at the commit
    /// gate without exposing a host-side domain constraint path.
    pub(crate) fn take_domain_dependencies(&mut self) -> Box<[TypedInsertDomainBinding]> {
        std::mem::take(&mut self.domain_dependencies)
    }

    /// Revalidate metadata-only domain type bindings while the prepared plan still owns the
    /// sealed target proof. Domain predicates remain unsupported; this proves only that the
    /// domain name/OID/base type and every participating table column still mean what preparation
    /// established.
    pub(crate) fn domain_dependencies_match(
        &self,
        domain_dependencies: &[TypedInsertDomainBinding],
        catalog: &CatalogSnapshot,
    ) -> bool {
        let Some(table) = catalog.relational_catalog.get(self.table.name.as_ref()) else {
            return false;
        };
        domain_dependencies.iter().all(|binding| {
            catalog
                .relational_domains
                .get(binding.name.as_ref())
                .is_some_and(|domain| {
                    domain.oid == binding.oid && domain.base_type == binding.base_type
                })
                && {
                    let mut matching = table
                        .columns
                        .iter()
                        .filter(|column| column.domain.as_deref() == Some(binding.name.as_ref()));
                    matching.clone().next().is_some()
                        && matching.all(|column| {
                            column.type_oid == binding.oid && column.ty == binding.base_type
                        })
                }
        })
    }

    pub(crate) fn matches_resident_append_insert(
        &self,
        expected_write_set: &WriteSet,
        catalog: &CatalogSnapshot,
        prepared_catalog_seq: Index,
    ) -> bool {
        self.statement_ordinal == InsertStatementOrdinal::FIRST
            && self.table.prepared_catalog_seq == prepared_catalog_seq
            && catalog.commit_seq == prepared_catalog_seq
            && self.row_count != 0
            && !self.columns.is_empty()
            && expected_write_set.tables.len() == 1
            && expected_write_set.tables.contains(self.table.name.as_ref())
            && expected_write_set.rows.is_empty()
            && expected_write_set.unique_slots.is_empty()
            && expected_write_set.unique_slots_i32.is_empty()
            && self.exact_dependencies_match(catalog)
            && catalog
                .relational_catalog
                .get(self.table.name.as_ref())
                .is_some_and(|table| {
                    table.oid == self.table.oid
                        && crate::engine_transaction_reset::table_schema_digest(table).ok()
                            == Some(self.table.schema_digest)
                        && table.columns.len() == self.columns.len()
                        && self
                            .columns
                            .iter()
                            .zip(&table.columns)
                            .all(|(column, live)| {
                                column.column_id == live.id
                                    && column.attnum == live.attnum
                                    && column.is_resident_append_vector(self.row_count as usize)
                                    && column.ty == live.ty
                                    && column.type_oid == live.type_oid
                                    && column.type_size == live.type_size
                            })
                })
    }

    fn exact_dependencies_match(&self, catalog: &CatalogSnapshot) -> bool {
        self.dependencies.len() == 1
            && self.dependencies[0].name == self.table.name
            && self.dependencies[0].oid == self.table.oid
            && self.dependencies[0].schema_digest == self.table.schema_digest
            && catalog
                .relational_catalog
                .get(self.dependencies[0].name.as_ref())
                .is_some_and(|table| {
                    table.oid == self.dependencies[0].oid
                        && crate::engine_transaction_reset::table_schema_digest(table).ok()
                            == Some(self.dependencies[0].schema_digest)
                })
    }

    /// Append one canonical v1 row directly from typed vectors. This does not reconstruct a
    /// row-major `SqlValue` matrix; the sealed column vectors remain the only values authority.
    pub(crate) fn append_binary_insert_template_row(
        &self,
        row: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), EngineError> {
        if self.columns.is_empty() || row >= self.row_count as usize {
            return Err(EngineError::Durability(
                "sealed typed INSERT batch row is out of bounds".to_string(),
            ));
        }
        for (position, column) in self.columns.iter().enumerate() {
            if position != 0 {
                out.push(b'|');
            }
            column.append_relational_cell(row, self.row_count as usize, out)?;
        }
        Ok(())
    }

    /// Consume the semantic batch into the only physical compiler input. The retained source
    /// preserves NULL validity and text offset/blob vectors; physical plan selection decides
    /// whether those vectors require a dense rollover or can use the fixed-width in-place arm.
    pub(crate) fn into_resident_append_source(self) -> Option<PreparedResidentAppendSource> {
        // Stateful sequence effects have no live resident-append/WAL owner.  Semantic lowering
        // may seal them for a future typed route, but this existing compiler must stay closed.
        if !self.sequence_bindings.is_empty() || !self.returning.is_empty() {
            return None;
        }
        let rows = self.row_count as usize;
        let requires_dense_rollover = self
            .columns
            .iter()
            .any(|column| column.ty == SqlType::Text || !column.validity.is_all_valid());
        let columns = self
            .columns
            .into_vec()
            .into_iter()
            .map(|column| {
                (matches!(column.presence, TypedInsertColumnPresence::AllProvided)
                    && column.all_inputs_are_resolved(rows)
                    && is_live_resident_append_type(column.ty)
                    && column.values.rows_match(rows))
                .then_some(PreparedResidentAppendColumn {
                    column_id: column.column_id,
                    attnum: column.attnum,
                    ty: column.ty,
                    type_oid: column.type_oid,
                    type_size: column.type_size,
                    validity: Some(column.validity),
                    values: Some(column.values),
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(PreparedResidentAppendSource {
            table: self.table,
            row_count: self.row_count,
            columns: columns.into(),
            dependencies: self.dependencies,
            requires_dense_rollover,
        })
    }
}

fn is_live_resident_append_type(ty: SqlType) -> bool {
    matches!(
        ty,
        SqlType::Int2
            | SqlType::Int4
            | SqlType::Date
            | SqlType::Int8
            | SqlType::Timestamp
            | SqlType::Numeric { .. }
            | SqlType::Uuid
            | SqlType::Bool
            | SqlType::Text
    )
}

fn checked_chunk_capacity(rows: usize, width: usize) -> Result<usize, ExecuteError> {
    rows.checked_mul(width).ok_or_else(|| {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "sealed typed fixed-width chunk length overflows".to_string(),
        ))
    })
}

/// Scalar-only fixed-width layout.  Unlike the legacy row encoder's geometry this intentionally
/// retains no catalog-ordered offset vector: the typed encoder emits its final boxed chunk and
/// fused descriptor owners directly.
struct TypedFixedAppendLayout {
    end: usize,
    i32_base: usize,
    i64_base: usize,
    b128_base: usize,
    non_bool_columns: usize,
}

fn fixed_append_error(message: &'static str) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(message.to_string()))
}

fn checked_typed_fixed_append_layout(
    columns: &[PreparedResidentAppendColumn],
    capacity: usize,
    row_start: usize,
    rows: usize,
) -> Result<TypedFixedAppendLayout, ExecuteError> {
    let end = row_start
        .checked_add(rows)
        .ok_or_else(|| fixed_append_error("sealed typed fixed-width row index overflowed"))?;
    if end > capacity {
        return Err(fixed_append_error(
            "sealed typed fixed-width append exceeds shard capacity",
        ));
    }
    if capacity > (1_usize << 31) {
        return Err(fixed_append_error(
            "sealed typed fixed-width capacity is implausibly large",
        ));
    }
    let mut i32_count = 0_usize;
    let mut i64_count = 0_usize;
    let mut b128_count = 0_usize;
    let mut non_bool_columns = 0_usize;
    for column in columns {
        match column.ty {
            SqlType::Int2 | SqlType::Int4 | SqlType::Date => {
                i32_count = i32_count.checked_add(1).ok_or_else(|| {
                    fixed_append_error("sealed typed fixed-width i32 column count overflowed")
                })?;
                non_bool_columns = non_bool_columns.checked_add(1).ok_or_else(|| {
                    fixed_append_error("sealed typed fixed-width column count overflowed")
                })?;
            }
            SqlType::Int8 | SqlType::Timestamp => {
                i64_count = i64_count.checked_add(1).ok_or_else(|| {
                    fixed_append_error("sealed typed fixed-width i64 column count overflowed")
                })?;
                non_bool_columns = non_bool_columns.checked_add(1).ok_or_else(|| {
                    fixed_append_error("sealed typed fixed-width column count overflowed")
                })?;
            }
            SqlType::Numeric { .. } | SqlType::Uuid => {
                b128_count = b128_count.checked_add(1).ok_or_else(|| {
                    fixed_append_error("sealed typed fixed-width b128 column count overflowed")
                })?;
                non_bool_columns = non_bool_columns.checked_add(1).ok_or_else(|| {
                    fixed_append_error("sealed typed fixed-width column count overflowed")
                })?;
            }
            SqlType::Bool => {}
            SqlType::Text => {
                return Err(fixed_append_error(
                    "sealed typed fixed-width source contains a text column",
                ));
            }
        }
    }
    let section_bytes = |columns: usize, width: usize| {
        columns
            .checked_mul(capacity)
            .and_then(|bytes| bytes.checked_mul(width))
            .ok_or_else(|| fixed_append_error("sealed typed fixed-width section overflowed"))
    };
    let i32_base = std::mem::size_of::<u64>();
    let i64_base = i32_base
        .checked_add(section_bytes(i32_count, std::mem::size_of::<i32>())?)
        .ok_or_else(|| fixed_append_error("sealed typed fixed-width section overflowed"))?;
    let b128_base = i64_base
        .checked_add(section_bytes(i64_count, std::mem::size_of::<i64>())?)
        .ok_or_else(|| fixed_append_error("sealed typed fixed-width section overflowed"))?;
    b128_base
        .checked_add(section_bytes(b128_count, 16)?)
        .ok_or_else(|| fixed_append_error("sealed typed fixed-width section overflowed"))?;
    Ok(TypedFixedAppendLayout {
        end,
        i32_base,
        i64_base,
        b128_base,
        non_bool_columns,
    })
}

fn typed_fixed_section_offset(
    base: usize,
    ordinal: usize,
    capacity: usize,
    row_start: usize,
    width: usize,
) -> Result<u64, ExecuteError> {
    let offset = ordinal
        .checked_mul(capacity)
        .and_then(|bytes| bytes.checked_mul(width))
        .and_then(|bytes| {
            row_start
                .checked_mul(width)
                .and_then(|start| bytes.checked_add(start))
        })
        .and_then(|bytes| base.checked_add(bytes))
        .ok_or_else(|| fixed_append_error("sealed typed fixed-width column offset overflowed"))?;
    u64::try_from(offset)
        .map_err(|_| fixed_append_error("sealed typed fixed-width column offset overflowed"))
}

fn dense_payload_shape_error() -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(
        "sealed typed dense resident payload lost its typed vector arm".to_string(),
    ))
}

fn bitmap_words(rows: usize) -> Result<usize, EngineError> {
    rows.checked_add(31)
        .map(|value| value / 32)
        .ok_or_else(|| EngineError::Durability("typed INSERT bitmap length overflows".to_string()))
}

fn bit_is_set(words: &[u32], row: usize) -> bool {
    words
        .get(row / 32)
        .is_some_and(|word| word & (1_u32 << (row % 32)) != 0)
}

fn set_bit(words: &mut [u32], row: usize) -> Result<(), EngineError> {
    let word = words.get_mut(row / 32).ok_or_else(|| {
        EngineError::Durability("typed INSERT bitmap index is out of bounds".to_string())
    })?;
    *word |= 1_u32 << (row % 32);
    Ok(())
}

fn bitmap_shape_is_exact(words: &[u32], rows: usize) -> bool {
    let Ok(expected_words) = bitmap_words(rows) else {
        return false;
    };
    if words.len() != expected_words {
        return false;
    }
    if rows.is_multiple_of(32) {
        return true;
    }
    let tail_mask = (1_u32 << (rows % 32)) - 1;
    words.last().is_some_and(|word| word & !tail_mask == 0)
}

fn bitmap_is_all_set(words: &[u32], rows: usize) -> bool {
    if !bitmap_shape_is_exact(words, rows) {
        return false;
    }
    let full_words = rows / 32;
    words[..full_words].iter().all(|word| *word == u32::MAX)
        && match rows % 32 {
            0 => true,
            tail => words.last() == Some(&((1_u32 << tail) - 1)),
        }
}

impl PreparedResidentAppendSource {
    pub(crate) fn table_name(&self) -> &str {
        &self.table.name
    }

    pub(crate) fn table_oid(&self) -> u32 {
        self.table.oid
    }

    pub(crate) fn schema_digest(&self) -> gpu_db_wal::CanonicalDigest {
        self.table.schema_digest
    }

    pub(crate) fn prepared_catalog_seq(&self) -> Index {
        self.table.prepared_catalog_seq
    }

    pub(crate) fn row_count(&self) -> usize {
        self.row_count as usize
    }

    /// Every host backing allocation held by the materialized resident source. This is separate
    /// from the semantic batch report because the move drops provenance/default/catalog vectors
    /// and creates one new outer physical-column box.
    #[allow(dead_code)] // Adopted by the inert reservation carrier next.
    pub(crate) fn host_retention_report(&self) -> Result<HostRetentionReport, EngineError> {
        let mut report = HostRetentionReport::default();
        self.table.append_host_retention(&mut report)?;
        report.retain_boxed_slice(&self.columns)?;
        for column in self.columns.iter() {
            if let Some(validity) = column.validity.as_ref() {
                validity.append_host_retention(&mut report)?;
            }
            if let Some(values) = column.values.as_ref() {
                values.append_host_retention(&mut report)?;
            }
        }
        report.retain_boxed_slice(&self.dependencies)?;
        for dependency in self.dependencies.iter() {
            dependency.append_host_retention(&mut report)?;
        }
        Ok(report)
    }

    /// Allocation-free host geometry for pre-lease source sizing. The identity-aware report is
    /// intentionally not reachable here: its map/set construction is diagnostic-only after an
    /// owner already exists.
    pub(crate) fn host_retention_geometry(&self) -> Result<HostRetentionGeometry, EngineError> {
        resident_source_retention::materialized_geometry(self)
    }

    pub(crate) fn columns(&self) -> &[PreparedResidentAppendColumn] {
        &self.columns
    }

    pub(crate) fn exact_single_table_dependency(&self) -> bool {
        self.dependencies.len() == 1
            && self.dependencies[0].name == self.table.name
            && self.dependencies[0].oid == self.table.oid
            && self.dependencies[0].schema_digest == self.table.schema_digest
    }

    /// Borrowed descriptor check for post-WAL apply and pre-WAL rollover planning.  Materializing
    /// a second `Vec<SqlType>` here would make a validation-only path part of the retained peak.
    pub(crate) fn column_types_match(&self, column_types: &[SqlType]) -> bool {
        self.columns.len() == column_types.len()
            && self
                .columns
                .iter()
                .zip(column_types)
                .all(|(column, ty)| column.ty == *ty)
    }

    pub(crate) fn requires_dense_rollover(&self) -> bool {
        self.requires_dense_rollover
    }

    #[cfg(test)]
    pub(crate) fn dense_vectors_are_consumed(&self) -> bool {
        self.requires_dense_rollover
            && self
                .columns
                .iter()
                .all(|column| column.values.is_none() && column.validity.is_none())
    }

    /// Delegate physical dense encoding to the resident-source leaf; semantic binding retains no
    /// layout authority beyond this stable facade.
    pub(crate) fn checked_dense_payload(
        &mut self,
        table: &RelationalTable,
    ) -> Result<PreparedResidentDensePayload, ExecuteError> {
        resident_source::checked_dense_payload(self, table)
    }

    /// Checked, column-major fixed-width encoding.  The sealed source owns the only logical
    /// values; this produces transfer chunks directly without a row-major `SqlValue` rebuild.
    pub(crate) fn checked_append_chunks(
        &self,
        capacity: usize,
        row_start: usize,
    ) -> Result<PreparedResidentFixedChunkOwners, ExecuteError> {
        let rows = self.row_count();
        if rows == 0 || self.columns.is_empty() || !self.exact_fixed_width_shape() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "sealed typed fixed-width source lost its parallel geometry".to_string(),
            )));
        }
        let layout = checked_typed_fixed_append_layout(&self.columns, capacity, row_start, rows)?;
        let mut chunks = Vec::with_capacity(layout.non_bool_columns + 1);
        let mut offsets = Vec::with_capacity(layout.non_bool_columns + 1);
        let mut i32_ordinal = 0_usize;
        let mut i64_ordinal = 0_usize;
        let mut b128_ordinal = 0_usize;
        for column in self.columns.iter() {
            let (byte_offset, width) = match column.ty {
                SqlType::Int2 | SqlType::Int4 | SqlType::Date => {
                    let offset = typed_fixed_section_offset(
                        layout.i32_base,
                        i32_ordinal,
                        capacity,
                        row_start,
                        std::mem::size_of::<i32>(),
                    )?;
                    i32_ordinal = i32_ordinal.checked_add(1).ok_or_else(|| {
                        fixed_append_error("sealed typed fixed-width i32 ordinal overflowed")
                    })?;
                    (offset, std::mem::size_of::<i32>())
                }
                SqlType::Int8 | SqlType::Timestamp => {
                    let offset = typed_fixed_section_offset(
                        layout.i64_base,
                        i64_ordinal,
                        capacity,
                        row_start,
                        std::mem::size_of::<i64>(),
                    )?;
                    i64_ordinal = i64_ordinal.checked_add(1).ok_or_else(|| {
                        fixed_append_error("sealed typed fixed-width i64 ordinal overflowed")
                    })?;
                    (offset, std::mem::size_of::<i64>())
                }
                SqlType::Numeric { .. } | SqlType::Uuid => {
                    let offset = typed_fixed_section_offset(
                        layout.b128_base,
                        b128_ordinal,
                        capacity,
                        row_start,
                        16,
                    )?;
                    b128_ordinal = b128_ordinal.checked_add(1).ok_or_else(|| {
                        fixed_append_error("sealed typed fixed-width b128 ordinal overflowed")
                    })?;
                    (offset, 16)
                }
                SqlType::Bool => continue,
                SqlType::Text => {
                    return Err(fixed_append_error(
                        "sealed typed fixed-width source contains a text column",
                    ));
                }
            };
            let mut bytes = Vec::with_capacity(checked_chunk_capacity(rows, width)?);
            match (column.values.as_ref(), column.ty) {
                (Some(TypedInsertColumnValues::I32(values)), SqlType::Int2)
                | (Some(TypedInsertColumnValues::I32(values)), SqlType::Int4)
                | (Some(TypedInsertColumnValues::I32(values)), SqlType::Date) => {
                    for value in values.iter() {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
                (Some(TypedInsertColumnValues::I64(values)), SqlType::Int8)
                | (Some(TypedInsertColumnValues::I64(values)), SqlType::Timestamp) => {
                    for value in values.iter() {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
                (Some(TypedInsertColumnValues::I128(values)), SqlType::Numeric { .. }) => {
                    for value in values.iter() {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
                (Some(TypedInsertColumnValues::Bytes16(values)), SqlType::Uuid) => {
                    for value in values.iter() {
                        bytes.extend_from_slice(value);
                    }
                }
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "sealed typed fixed-width source lost its value arm".to_string(),
                    )));
                }
            }
            if bytes.len() != checked_chunk_capacity(rows, width)? {
                return Err(fixed_append_error(
                    "sealed typed fixed-width payload length drifted",
                ));
            }
            chunks.push(PreparedResidentFixedChunk {
                byte_offset,
                bytes: bytes.into_boxed_slice(),
            });
            offsets.push(byte_offset);
        }
        let header: Box<[u8]> = (layout.end as u64).to_le_bytes().into();
        chunks.push(PreparedResidentFixedChunk {
            byte_offset: 0,
            bytes: header,
        });
        offsets.push(0);
        Ok(PreparedResidentFixedChunkOwners {
            chunks: chunks.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
        })
    }

    /// Predict the retained fixed append encodings without allocating or encoding their bytes.
    /// The final plan owns one exact payload box per non-BOOL column plus the count-header box.
    pub(crate) fn fixed_append_host_owner_geometry(
        &self,
        capacity: usize,
        row_start: usize,
    ) -> Option<FixedAppendHostOwnerGeometry> {
        let rows = self.row_count();
        if rows == 0
            || self.columns.is_empty()
            || !self.exact_fixed_width_shape()
            || row_start.checked_add(rows)? > capacity
        {
            return None;
        }
        let mut chunk_count = 1_usize; // sealed row-count header
        let mut chunk_payload_bytes = u64::try_from(std::mem::size_of::<u64>()).ok()?;
        let mut int4_count = 0_usize;
        let mut bool_count = 0_usize;
        for column in self.columns.iter() {
            let width = match column.ty {
                SqlType::Int2 | SqlType::Int4 | SqlType::Date => {
                    int4_count = int4_count.checked_add(1)?;
                    Some(std::mem::size_of::<i32>())
                }
                SqlType::Int8 | SqlType::Timestamp => Some(std::mem::size_of::<i64>()),
                SqlType::Numeric { .. } | SqlType::Uuid => Some(16),
                SqlType::Bool => {
                    bool_count = bool_count.checked_add(1)?;
                    None
                }
                SqlType::Text => return None,
            };
            if let Some(width) = width {
                chunk_count = chunk_count.checked_add(1)?;
                let bytes = u64::try_from(rows)
                    .ok()?
                    .checked_mul(u64::try_from(width).ok()?)?;
                chunk_payload_bytes = chunk_payload_bytes.checked_add(bytes)?;
            }
        }
        Some(FixedAppendHostOwnerGeometry {
            chunk_count,
            chunk_payload_bytes,
            int4_count,
            bool_count,
        })
    }

    pub(crate) fn int4_min_max(&self) -> Option<Box<[(i32, i32)]>> {
        self.columns
            .iter()
            .filter_map(|column| match (column.values.as_ref(), column.ty) {
                (Some(TypedInsertColumnValues::I32(values)), SqlType::Int2)
                | (Some(TypedInsertColumnValues::I32(values)), SqlType::Int4)
                | (Some(TypedInsertColumnValues::I32(values)), SqlType::Date) => {
                    Some((values.len() == self.row_count()).then(|| {
                        values
                            .iter()
                            .copied()
                            .fold((i32::MAX, i32::MIN), |(min, max), value| {
                                (min.min(value), max.max(value))
                            })
                    }))
                }
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .map(Vec::into_boxed_slice)
    }

    /// Materialize the exact fixed-rollover zone-map descriptors once, before WAL.  The boxed
    /// descriptor array is transferred into the pending shard rather than built from a temporary
    /// min/max vector during reservation.
    pub(crate) fn fixed_int4_stats(
        &self,
        table: &RelationalTable,
    ) -> Option<Box<[ResidentDeviceInt4ColumnStats]>> {
        if table.columns.len() != self.columns.len()
            || !table
                .columns
                .iter()
                .zip(&self.columns)
                .all(|(live, column)| live.id == column.column_id && live.ty == column.ty)
        {
            return None;
        }
        let expected = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date))
            .count();
        let mut stats = Vec::with_capacity(expected);
        for (column, live) in self.columns.iter().zip(&table.columns) {
            if !matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date) {
                continue;
            }
            let Some(TypedInsertColumnValues::I32(values)) = column.values.as_ref() else {
                return None;
            };
            if values.len() != self.row_count() {
                return None;
            }
            let (min, max) = values
                .iter()
                .copied()
                .fold((i32::MAX, i32::MIN), |(min, max), value| {
                    (min.min(value), max.max(value))
                });
            stats.push(ResidentDeviceInt4ColumnStats {
                name: String::from(Box::<str>::from(live.name.as_str())),
                min,
                max,
            });
        }
        (stats.len() == expected).then(|| stats.into_boxed_slice())
    }

    /// Materialize the final fixed-plan BOOL uploads directly from sealed bit vectors.  Names
    /// are copied into exact boxes once; no intermediate ID-only upload vector crosses this
    /// boundary.
    pub(crate) fn fixed_bool_uploads(
        &self,
        table: &RelationalTable,
    ) -> Option<Box<[PreparedResidentFixedBoolUpload]>> {
        if table.columns.len() != self.columns.len()
            || !table
                .columns
                .iter()
                .zip(&self.columns)
                .all(|(live, column)| live.id == column.column_id && live.ty == column.ty)
        {
            return None;
        }
        let bool_count = table
            .columns
            .iter()
            .filter(|column| column.ty == SqlType::Bool)
            .count();
        let mut uploads = Vec::with_capacity(bool_count);
        for (column, live) in self.columns.iter().zip(&table.columns) {
            if column.ty != SqlType::Bool {
                continue;
            }
            let Some(TypedInsertColumnValues::BoolBits(words)) = column.values.as_ref() else {
                return None;
            };
            if !bitmap_shape_is_exact(words, self.row_count()) {
                return None;
            }
            let mut values = Vec::with_capacity(self.row_count());
            values.extend((0..self.row_count()).map(|row| u8::from(bit_is_set(words, row))));
            uploads.push(PreparedResidentFixedBoolUpload {
                name: Box::from(live.name.as_str()),
                values: values.into_boxed_slice(),
            });
        }
        (uploads.len() == bool_count).then(|| uploads.into_boxed_slice())
    }

    fn exact_fixed_width_shape(&self) -> bool {
        self.columns.iter().all(|column| {
            matches!(column.validity, Some(TypedInsertColumnValidity::AllValid))
                && is_live_resident_append_type(column.ty)
                && column
                    .values
                    .as_ref()
                    .is_some_and(|values| values.rows_match(self.row_count()))
                && match (column.values.as_ref(), column.ty) {
                    (Some(TypedInsertColumnValues::I32(values)), SqlType::Int2)
                    | (Some(TypedInsertColumnValues::I32(values)), SqlType::Int4)
                    | (Some(TypedInsertColumnValues::I32(values)), SqlType::Date) => {
                        values.len() == self.row_count()
                    }
                    (Some(TypedInsertColumnValues::I64(values)), SqlType::Int8)
                    | (Some(TypedInsertColumnValues::I64(values)), SqlType::Timestamp) => {
                        values.len() == self.row_count()
                    }
                    (Some(TypedInsertColumnValues::I128(values)), SqlType::Numeric { .. }) => {
                        values.len() == self.row_count()
                    }
                    (Some(TypedInsertColumnValues::Bytes16(values)), SqlType::Uuid) => {
                        values.len() == self.row_count()
                    }
                    (Some(TypedInsertColumnValues::BoolBits(words)), SqlType::Bool) => {
                        bitmap_shape_is_exact(words, self.row_count())
                    }
                    _ => false,
                }
        })
    }
}

impl PreparedResidentAppendColumn {
    pub(crate) fn column_id(&self) -> u32 {
        self.column_id
    }

    pub(crate) fn attnum(&self) -> i16 {
        self.attnum
    }

    pub(crate) fn ty(&self) -> SqlType {
        self.ty
    }

    pub(crate) fn type_oid(&self) -> u32 {
        self.type_oid
    }

    pub(crate) fn type_size(&self) -> i16 {
        self.type_size
    }

    pub(crate) fn i32_values(&self) -> Option<&[i32]> {
        match (self.values.as_ref(), self.ty) {
            (Some(TypedInsertColumnValues::I32(values)), SqlType::Int2)
            | (Some(TypedInsertColumnValues::I32(values)), SqlType::Int4)
            | (Some(TypedInsertColumnValues::I32(values)), SqlType::Date) => Some(values),
            _ => None,
        }
    }
}

/// The live route consumes the one shared semantic builder, then compiles fixed-width vectors or
/// a sealed dense nullable/text rollover. Reordered full column lists have already been bound
/// into catalog order here; neither physical branch creates another publisher.
pub(super) fn try_prepare_typed_insert_batch(
    command: &Command,
    catalog: &CatalogSnapshot,
    prepared_catalog_seq: Index,
    expected_catalog_version: Option<crate::engine_mutation_admission::CatalogVersionExpectation>,
) -> Result<Option<TypedInsertBatch>, ExecuteError> {
    let Command::Insert(insert) = command else {
        return Ok(None);
    };
    match builder::build(
        insert,
        catalog,
        prepared_catalog_seq,
        expected_catalog_version,
        TypedInsertBuildCapability::ResidentAppend,
    )? {
        TypedInsertBuildResult::Ready(batch) => Ok(Some(batch)),
        TypedInsertBuildResult::Deferred(reason) => {
            let _ = reason;
            Ok(None)
        }
    }
}

#[cfg(test)]
pub(super) fn try_prepare_typed_insert_batch_proof_only(
    command: &Command,
    catalog: &CatalogSnapshot,
    prepared_catalog_seq: Index,
) -> Result<Option<TypedInsertBatch>, ExecuteError> {
    let Command::Insert(insert) = command else {
        return Ok(None);
    };
    match builder::build(
        insert,
        catalog,
        prepared_catalog_seq,
        None,
        TypedInsertBuildCapability::ProofOnly,
    )? {
        TypedInsertBuildResult::Ready(batch) => Ok(Some(batch)),
        TypedInsertBuildResult::Deferred(reason) => {
            let _ = reason;
            Ok(None)
        }
    }
}

#[cfg(test)]
pub(super) fn try_prepare_typed_insert_batch_foreign_key_proof_only(
    command: &Command,
    catalog: &CatalogSnapshot,
    prepared_catalog_seq: Index,
) -> Result<Option<TypedInsertBatch>, ExecuteError> {
    let Command::Insert(insert) = command else {
        return Ok(None);
    };
    match builder::build(
        insert,
        catalog,
        prepared_catalog_seq,
        None,
        TypedInsertBuildCapability::ForeignKeyProofOnly,
    )? {
        TypedInsertBuildResult::Ready(batch) => Ok(Some(batch)),
        TypedInsertBuildResult::Deferred(reason) => {
            let _ = reason;
            Ok(None)
        }
    }
}

#[cfg(test)]
#[path = "typed_insert_batch/defaults_tests.rs"]
mod defaults_tests;
#[cfg(test)]
#[path = "typed_insert_batch/semantics_tests.rs"]
mod semantics_tests;
#[cfg(test)]
#[path = "typed_insert_batch_tests.rs"]
mod tests;
