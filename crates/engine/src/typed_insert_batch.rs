//! Immutable typed INSERT batches prepared off the commit sequencer.
//!
//! The batch is the one private semantic carrier for INSERT input.  It has no raw constructor,
//! no `Clone`, no predicted row identities, and no `WriteDelta`: physical plans and WAL templates
//! may only consume its already catalog-ordered vectors.

use super::*;
#[cfg(test)]
use crate::insert_semantic_ir::InsertSourceOrdinal;
use crate::insert_semantic_ir::{
    InsertStatementOrdinal, ResolvedInsertInput, ResolvedInsertSemantics,
};
use crate::rel_exec_helpers::{append_relational_cell, coerce_insert_value, RelationalCellRef};

mod builder;
mod constraint_source;
mod defaults;
mod resident_source;
pub(crate) use constraint_source::TypedInsertConstraintDeviceSource;
#[cfg(test)]
pub(crate) use constraint_source::{
    constraint_source_upload_count, reset_constraint_source_upload_count,
};
pub(crate) use resident_source::{
    PreparedResidentAppendColumn, PreparedResidentAppendSource, PreparedResidentBoolUpload,
    PreparedResidentDensePayload,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct TypedInsertBatchTable {
    name: Arc<str>,
    oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
    prepared_catalog_seq: Index,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TypedInsertDependencyBinding {
    name: Arc<str>,
    oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
}

/// One catalog-order column. `column_id` and `attnum`, rather than a parsed column-list position,
/// are the stable identity carried toward later plan compilation.
struct TypedInsertColumn {
    column_id: u32,
    attnum: i16,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
    /// `None` represents an omitted target column. When present, the complete source identity
    /// is derived from this ordinal plus `TypedInsertBatch::statement_ordinal` and the row index;
    /// it must never be repeated in every row's semantic metadata.
    source_column_ordinal: Option<u32>,
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
    pub(super) name: Arc<str>,
    pub(super) oid: u32,
    pub(super) base_type: SqlType,
}

/// A private builder may decline a shape before any batch, WAL, or device plan exists.  This keeps
/// future feature work explicit rather than silently reinterpreting an unsupported input.
#[derive(Debug, PartialEq, Eq)]
enum TypedInsertDeferred {
    CatalogGeneration,
    Returning,
    Constraints,
    Default { column_id: u32 },
}

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
    /// The indexed resident-key path remains a test-only proof seam.  Keeping the marker on the
    /// sealed carrier prevents the live ResidentAppend builder from accidentally taking an
    /// indexed pre-WAL branch merely because a later catalog acquired an index.
    #[cfg(test)]
    proof_only_indexed_constraints: bool,
}

impl TypedInsertBatch {
    #[cfg(test)]
    pub(crate) fn is_proof_only_indexed_constraints(&self) -> bool {
        self.proof_only_indexed_constraints
    }
    #[cfg(test)]
    fn value_bytes(&self) -> usize {
        self.columns
            .iter()
            .map(|column| column.values.byte_len())
            .sum()
    }

    pub(super) fn binary_insert_template(
        &self,
    ) -> Result<crate::wal_binary::PreparedBinaryInsertTemplate, EngineError> {
        crate::wal_binary::PreparedBinaryInsertTemplate::from_sealed_batch(self)
    }

    pub(crate) fn binary_insert_template_table_name(&self) -> &str {
        &self.table.name
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

    pub(crate) fn columns(&self) -> &[PreparedResidentAppendColumn] {
        &self.columns
    }

    pub(crate) fn exact_single_table_dependency(&self) -> bool {
        self.dependencies.len() == 1
            && self.dependencies[0].name == self.table.name
            && self.dependencies[0].oid == self.table.oid
            && self.dependencies[0].schema_digest == self.table.schema_digest
    }

    pub(crate) fn column_types(&self) -> Vec<SqlType> {
        self.columns.iter().map(|column| column.ty).collect()
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
    ) -> Result<Vec<gpu_db_execution::CudaOwnedDeviceMemoryChunk>, ExecuteError> {
        let rows = self.row_count();
        if rows == 0 || self.columns.is_empty() || !self.exact_fixed_width_shape() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "sealed typed fixed-width source lost its parallel geometry".to_string(),
            )));
        }
        let types = self.column_types();
        let geometry = crate::engine_residency::checked_fixed_width_append_geometry(
            &types, capacity, row_start, rows,
        )?;
        let mut chunks = Vec::with_capacity(self.columns.len() + 1);
        for (ordinal, column) in self.columns.iter().enumerate() {
            let byte_offset = match geometry.column_offset(ordinal) {
                Some(offset) => offset,
                None if column.ty == SqlType::Bool => continue,
                None => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "sealed typed fixed-width source lost its section offset".to_string(),
                    )));
                }
            };
            let mut bytes = Vec::new();
            match (column.values.as_ref(), column.ty) {
                (Some(TypedInsertColumnValues::I32(values)), SqlType::Int2)
                | (Some(TypedInsertColumnValues::I32(values)), SqlType::Int4)
                | (Some(TypedInsertColumnValues::I32(values)), SqlType::Date) => {
                    bytes.reserve(checked_chunk_capacity(
                        values.len(),
                        std::mem::size_of::<i32>(),
                    )?);
                    for value in values.iter() {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
                (Some(TypedInsertColumnValues::I64(values)), SqlType::Int8)
                | (Some(TypedInsertColumnValues::I64(values)), SqlType::Timestamp) => {
                    bytes.reserve(checked_chunk_capacity(
                        values.len(),
                        std::mem::size_of::<i64>(),
                    )?);
                    for value in values.iter() {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
                (Some(TypedInsertColumnValues::I128(values)), SqlType::Numeric { .. }) => {
                    bytes.reserve(checked_chunk_capacity(values.len(), 16)?);
                    for value in values.iter() {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
                (Some(TypedInsertColumnValues::Bytes16(values)), SqlType::Uuid) => {
                    bytes.reserve(checked_chunk_capacity(values.len(), 16)?);
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
            chunks.push(gpu_db_execution::CudaOwnedDeviceMemoryChunk { byte_offset, bytes });
        }
        chunks.push(gpu_db_execution::CudaOwnedDeviceMemoryChunk {
            byte_offset: 0,
            bytes: (geometry.end() as u64).to_le_bytes().to_vec(),
        });
        Ok(chunks)
    }

    pub(crate) fn i32_column_slices(&self) -> Option<Vec<&[i32]>> {
        self.columns
            .iter()
            .map(|column| match (column.values.as_ref(), column.ty) {
                (Some(TypedInsertColumnValues::I32(values)), SqlType::Int2)
                | (Some(TypedInsertColumnValues::I32(values)), SqlType::Int4)
                | (Some(TypedInsertColumnValues::I32(values)), SqlType::Date)
                    if values.len() == self.row_count() =>
                {
                    Some(values.as_ref())
                }
                _ => None,
            })
            .collect()
    }

    pub(crate) fn int4_min_max(&self) -> Option<Vec<(i32, i32)>> {
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
            .collect()
    }

    pub(crate) fn bool_uploads(&self) -> Option<Vec<PreparedResidentBoolUpload>> {
        self.columns
            .iter()
            .filter_map(|column| match (column.values.as_ref(), column.ty) {
                (Some(TypedInsertColumnValues::BoolBits(words)), SqlType::Bool) => {
                    Some(bitmap_shape_is_exact(words, self.row_count()).then(|| {
                        PreparedResidentBoolUpload {
                            column_id: column.column_id,
                            values: (0..self.row_count())
                                .map(|row| u8::from(bit_is_set(words, row)))
                                .collect(),
                        }
                    }))
                }
                _ => None,
            })
            .collect()
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
#[path = "typed_insert_batch/defaults_tests.rs"]
mod defaults_tests;
#[cfg(test)]
#[path = "typed_insert_batch_tests.rs"]
mod tests;
