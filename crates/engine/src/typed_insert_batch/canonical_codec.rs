//! Inert canonical v1 typed-INSERT logical record.
//!
//! This leaf owns only stable logical bytes and their strict decoder. It has no WAL opcode,
//! transaction/status, row-id, physical-plan, residency, recovery, or apply authority.

use super::*;
use crate::typed_insert_batch::sequence_defaults::effects::CanonicalSequenceParentView;
use std::collections::{BTreeMap, BTreeSet};

#[path = "canonical_codec_decode.rs"]
mod decode;
#[path = "canonical_codec_sequence.rs"]
mod sequence;
#[path = "canonical_codec_validation.rs"]
mod validation;

#[cfg(test)]
#[path = "canonical_codec_sabotage_tests.rs"]
mod sabotage_tests;
#[cfg(test)]
#[path = "canonical_codec_semantics_tests.rs"]
mod semantics_tests;
#[cfg(test)]
#[path = "canonical_codec_sequence_tests.rs"]
pub(crate) mod sequence_tests;
#[cfg(test)]
#[path = "canonical_codec_tests.rs"]
mod tests;

pub(crate) const CANONICAL_TYPED_INSERT_RECORD_MAX_BYTES: usize = 16 * 1024 * 1024;
pub(crate) const CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES: usize = 100;
pub(super) const MAX_RECORD_BYTES: usize = CANONICAL_TYPED_INSERT_RECORD_MAX_BYTES;
const HEADER_LEN: usize = CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES;
const SECTION_COUNT: u16 = 8;
const MAGIC: [u8; 16] = *b"GPUDBTYPEDINS1\0\0";
const FORMAT_VERSION: u16 = 1;
const SEMANTICS_VERSION: u16 = 1;
const MAX_IDENTIFIER_BYTES: usize = u16::MAX as usize;
const SECTION_TARGET: u16 = 1;
const SECTION_COLUMNS: u16 = 2;
const SECTION_DEPENDENCIES: u16 = 3;
const SECTION_DOMAINS: u16 = 4;
const SECTION_INDEXES: u16 = 5;
const SECTION_FOREIGN_KEYS: u16 = 6;
const SECTION_RETURNING: u16 = 7;
const SECTION_SEQUENCE_EFFECTS: u16 = 8;

pub(crate) use decode::{
    DecodedReturningLayoutFacts, DecodedSequenceEffectFacts, DecodedSequenceEffectKindFacts,
    DecodedSequenceParentFacts, DecodedSequenceRequestFacts, DecodedTypedInsertRecordFacts,
    DecodedTypedInsertTargetFacts,
};

/// Crate-private, non-cloneable decoded evidence. It intentionally owns the validated parsed
/// model, not an input-byte duplicate, and offers no conversion into a typed batch, physical
/// plan, or DML carrier.
#[allow(dead_code)] // Future inert codec-5 S2/S5 reader; no live caller exists yet.
pub(crate) struct DecodedTypedInsertRecord {
    model: decode::DecodedModel,
}

/// Shared fixed prefix used by both the canonical typed-record decoder and codec-5 aggregate
/// replay framing.  Keeping this authority here prevents a future replay reader from silently
/// drifting in magic/version/limit/header-digest interpretation.
#[derive(Clone, Copy)]
pub(crate) struct CanonicalTypedInsertRecordPrefix {
    pub(crate) typed_statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) returning_digest: gpu_db_wal::CanonicalDigest,
}

pub(crate) fn parse_canonical_typed_insert_record_prefix(
    header: &[u8; CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES],
    record_bytes: usize,
) -> Result<CanonicalTypedInsertRecordPrefix, EngineError> {
    if !(CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES..=CANONICAL_TYPED_INSERT_RECORD_MAX_BYTES)
        .contains(&record_bytes)
        || header[0..16] != MAGIC
        || u16::from_le_bytes(header[16..18].try_into().expect("fixed typed record field"))
            != FORMAT_VERSION
        || u16::from_le_bytes(header[18..20].try_into().expect("fixed typed record field"))
            != SEMANTICS_VERSION
        || u16::from_le_bytes(header[20..22].try_into().expect("fixed typed record field"))
            != FORMAT_VERSION
        || u16::from_le_bytes(header[22..24].try_into().expect("fixed typed record field"))
            != FORMAT_VERSION
        || u32::from_le_bytes(header[24..28].try_into().expect("fixed typed record field")) != 0
        || u16::from_le_bytes(header[28..30].try_into().expect("fixed typed record field"))
            != SECTION_COUNT
        || header[30..32].iter().any(|byte| *byte != 0)
        || usize::try_from(u32::from_le_bytes(
            header[32..36].try_into().expect("fixed typed record field"),
        ))
        .ok()
        .is_none_or(|body| body != record_bytes - CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES)
    {
        return Err(codec_error(
            "record prefix is not canonical typed-INSERT v1",
        ));
    }
    Ok(CanonicalTypedInsertRecordPrefix {
        typed_statement_digest: header[36..68]
            .try_into()
            .expect("fixed typed statement digest"),
        returning_digest: header[68..100].try_into().expect("fixed RETURNING digest"),
    })
}

#[allow(dead_code)] // The strict views become live only with the future codec-5 replay owner.
impl DecodedTypedInsertRecord {
    /// Rebuild canonical bytes only from the validated retained model. This is test-only so no
    /// production caller can mistake the decoder for a second raw-byte authority.
    #[cfg(test)]
    pub(super) fn reencode(&self) -> Vec<u8> {
        decode::reencode_decoded(&self.model)
            .expect("validated canonical typed-INSERT model must reencode exactly")
    }

    /// Scalar record facts needed to bind one decoded S2 typed INSERT to its directory and
    /// outcome sections. This view has no vector, plan, apply, or mutation authority.
    pub(crate) fn facts(&self) -> DecodedTypedInsertRecordFacts {
        decode::record_facts(&self.model)
    }

    /// Typed statement identity from the retained model.
    pub(crate) fn typed_statement_digest(&self) -> gpu_db_wal::CanonicalDigest {
        self.facts().typed_statement_digest
    }

    /// RETURNING-layout identity from the retained model.
    pub(crate) fn returning_digest(&self) -> gpu_db_wal::CanonicalDigest {
        self.facts().returning.digest
    }

    /// The optional, admitted sequence parent identity. Sequence state remains exclusively
    /// owned by the retained decoded model.
    pub(crate) fn sequence_parent(&self) -> Option<DecodedSequenceParentFacts> {
        decode::sequence_parent_facts(&self.model)
    }

    /// Canonically ordered, scalar-only sequence-effect facts. The iterator borrows this record
    /// and cannot yield its owned names, vectors, batch, or sequence state.
    pub(crate) fn sequence_effects(
        &self,
    ) -> impl ExactSizeIterator<Item = DecodedSequenceEffectFacts> + '_ {
        decode::sequence_effect_facts(&self.model)
    }
}

#[allow(dead_code)]
pub(super) fn encode(batch: &TypedInsertBatch) -> Result<Vec<u8>, EngineError> {
    let mut record = Writer::new(MAX_RECORD_BYTES);
    encode_into(batch, &mut record)?;
    Ok(record.finish())
}

/// Count the exact v1 record bytes by executing the same validation and section traversal as
/// [`encode`].  The counting writer intentionally retains no output buffer, so this remains a
/// side-effect-free pre-WAL sizing probe rather than a second codec formula.
pub(super) fn encoded_len(batch: &TypedInsertBatch) -> Result<usize, EngineError> {
    let mut record = Writer::counting(MAX_RECORD_BYTES);
    encode_into(batch, &mut record)?;
    Ok(record.len())
}

fn encode_into(batch: &TypedInsertBatch, record: &mut Writer) -> Result<(), EngineError> {
    validate_batch(batch)?;
    let typed_statement_digest = typed_statement_digest_for_batch(batch)?;
    if typed_statement_digest != batch.typed_statement_digest {
        return Err(codec_error(
            "typed statement digest drifted after semantic preparation",
        ));
    }
    let returning_digest = returning_layout_digest(&batch.returning)?;
    record.bytes(&MAGIC)?;
    record.u16(FORMAT_VERSION)?;
    record.u16(SEMANTICS_VERSION)?;
    record.u16(FORMAT_VERSION)?; // minimum reader
    record.u16(FORMAT_VERSION)?; // maximum reader
    record.u32(0)?; // flags
    record.u16(SECTION_COUNT)?;
    record.u16(0)?; // reserved
    record.u32(0)?; // body length, filled once all sections are present
    record.digest(&typed_statement_digest)?;
    record.digest(&returning_digest)?;
    debug_assert_eq!(record.len(), HEADER_LEN);

    append_section(record, SECTION_TARGET, |section| {
        encode_target(batch, section)
    })?;
    append_section(record, SECTION_COLUMNS, |section| {
        encode_columns(
            &batch.columns,
            batch.row_count,
            &batch.domain_dependencies,
            section,
        )
    })?;
    append_section(record, SECTION_DEPENDENCIES, |section| {
        encode_dependencies(&batch.dependencies, section)
    })?;
    append_section(record, SECTION_DOMAINS, |section| {
        encode_domains(&batch.domain_dependencies, section)
    })?;
    append_section(record, SECTION_INDEXES, |section| {
        encode_indexes(&batch.canonical_catalog.indexes, section)
    })?;
    append_section(record, SECTION_FOREIGN_KEYS, |section| {
        encode_foreign_keys(&batch.canonical_catalog.foreign_keys, section)
    })?;
    append_section(record, SECTION_RETURNING, |section| {
        encode_returning(&batch.returning, returning_digest, section)
    })?;
    append_section(record, SECTION_SEQUENCE_EFFECTS, |section| {
        sequence::encode_sequence_effects(
            &batch.sequence_bindings,
            typed_statement_digest,
            batch.statement_ordinal,
            section,
        )
    })?;
    let body_len = record
        .len()
        .checked_sub(HEADER_LEN)
        .ok_or_else(|| codec_error("canonical header length underflow"))?;
    record.patch_u32(
        32,
        u32::try_from(body_len).map_err(|_| codec_error("record body exceeds u32"))?,
    );
    Ok(())
}

/// Strictly decode one canonical typed-INSERT v1 record into private validated model evidence.
/// The returned move-only carrier has only the scalar read views above; it cannot be promoted
/// into a live batch, plan, WAL writer, recovery state, or apply operation.
#[allow(dead_code)] // Exposed through the inert typed-batch facade for a later codec-5 slice.
pub(crate) fn decode(bytes: &[u8]) -> Result<DecodedTypedInsertRecord, EngineError> {
    decode::decode(bytes)
}

/// The one digest used by both semantic preparation and the inert effect terminal's RETURNING
/// evidence. It owns no result-route state or physical layout.
pub(crate) fn returning_layout_digest(
    returning: &returning::BoundInsertReturning,
) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
    let shape = returning.effect_shape();
    let mut body = Writer::new(MAX_RECORD_BYTES);
    body.bytes(b"GPUDBTYPEDINSERTRETURNING1")?;
    body.u16(FORMAT_VERSION)?;
    body.u16(SEMANTICS_VERSION)?;
    body.u32(shape.row_count())?;
    body.u32(shape.column_count())?;
    body.u64(shape.cell_count())?;
    body.u32(shape.column_count())?;
    for projection in returning.effect_projection_identities() {
        append_projection(
            &mut body,
            projection.catalog_column_ordinal(),
            projection.column_id(),
            projection.attnum(),
            projection.name(),
            projection.ty(),
            projection.type_oid(),
            projection.type_size(),
        )?;
    }
    Ok(gpu_db_wal::canonical_request_digest(&body.finish()))
}

/// Compute the pre-effect statement intent during semantic preparation, before receipt/outcome
/// authority exists. In particular, no prepared catalog sequence or sequence result is present.
pub(super) struct TypedStatementDigestInput<'a> {
    pub(super) table: &'a RelationalTable,
    pub(super) statement_ordinal: InsertStatementOrdinal,
    pub(super) row_count: u32,
    pub(super) columns: &'a [TypedInsertColumn],
    pub(super) dependencies: &'a [TypedInsertDependencyBinding],
    pub(super) domains: &'a [TypedInsertDomainBinding],
    pub(super) canonical_catalog: &'a TypedInsertCanonicalCatalog,
    pub(super) returning: &'a returning::BoundInsertReturning,
    pub(super) sequence_requests: &'a sequence_defaults::SequenceDefaultRequests,
}

pub(super) fn typed_statement_digest_for_prepared(
    input: TypedStatementDigestInput<'_>,
) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
    let sequence_cells = input
        .sequence_requests
        .effect_shapes()
        .map(|request| (request.catalog_column_ordinal(), request.row_ordinal()))
        .collect::<BTreeSet<_>>();
    let mut body = Writer::new(MAX_RECORD_BYTES);
    append_statement_preamble(
        &mut body,
        input.table.schema.as_str(),
        input.table.name.as_str(),
        input.table.oid,
        crate::engine_transaction_reset::table_schema_digest(input.table)
            .map_err(|error| codec_error(&error.to_string()))?,
        input.statement_ordinal,
        input.row_count,
        input.columns,
        input.dependencies,
        input.domains,
        input.canonical_catalog,
        returning_layout_digest(input.returning)?,
        &sequence_cells,
    )?;
    body.u32(checked_u32(
        input.sequence_requests.effect_shapes().len(),
        "sequence request count",
    )?)?;
    for request in input.sequence_requests.effect_shapes() {
        append_sequence_request(
            &mut body,
            request.target_table_oid(),
            request.row_ordinal(),
            request.catalog_column_ordinal(),
            request.column_id(),
            request.sequence_oid(),
            request.sequence_source_name(),
            request.sequence_effective_name(),
            request.statement_ordinal(),
            request.expression_ordinal(),
        )?;
    }
    Ok(gpu_db_wal::canonical_request_digest(&body.finish()))
}

fn typed_statement_digest_for_batch(
    batch: &TypedInsertBatch,
) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
    let effects = batch
        .sequence_bindings
        .iter()
        .map(|binding| binding.canonical_view())
        .collect::<Vec<_>>();
    let sequence_cells = effects
        .iter()
        .map(|effect| {
            (
                effect.request.catalog_column_ordinal,
                effect.request.row_ordinal,
            )
        })
        .collect::<BTreeSet<_>>();
    let mut body = Writer::new(MAX_RECORD_BYTES);
    append_statement_preamble(
        &mut body,
        batch.table.schema.as_ref(),
        batch.table.name.as_ref(),
        batch.table.oid,
        batch.table.schema_digest,
        batch.statement_ordinal,
        batch.row_count,
        &batch.columns,
        &batch.dependencies,
        &batch.domain_dependencies,
        &batch.canonical_catalog,
        returning_layout_digest(&batch.returning)?,
        &sequence_cells,
    )?;
    body.u32(checked_u32(effects.len(), "sequence effect count")?)?;
    for effect in effects {
        let request = effect.request;
        append_sequence_request(
            &mut body,
            request.target_table_oid,
            request.row_ordinal,
            request.catalog_column_ordinal,
            request.column_id,
            request.sequence_oid,
            request.sequence_source_name,
            request.sequence_effective_name,
            request.statement_ordinal,
            request.expression_ordinal,
        )?;
    }
    Ok(gpu_db_wal::canonical_request_digest(&body.finish()))
}

#[allow(clippy::too_many_arguments)]
fn append_statement_preamble(
    body: &mut Writer,
    schema: &str,
    name: &str,
    oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
    statement_ordinal: InsertStatementOrdinal,
    rows: u32,
    columns: &[TypedInsertColumn],
    dependencies: &[TypedInsertDependencyBinding],
    domains: &[TypedInsertDomainBinding],
    canonical_catalog: &TypedInsertCanonicalCatalog,
    returning_digest: gpu_db_wal::CanonicalDigest,
    sequence_cells: &BTreeSet<(u32, u32)>,
) -> Result<(), EngineError> {
    body.bytes(b"GPUDBTYPEDINSERTSTATEMENT1")?;
    body.u16(FORMAT_VERSION)?;
    body.u16(SEMANTICS_VERSION)?;
    body.identifier(schema)?;
    body.identifier(name)?;
    body.u32(oid)?;
    body.digest(&schema_digest)?;
    body.u32(statement_ordinal.as_u32())?;
    body.u32(rows)?;
    append_intent_columns(body, columns, rows, sequence_cells)?;
    append_dependencies(body, dependencies)?;
    append_domains(body, domains)?;
    append_indexes(body, &canonical_catalog.indexes)?;
    append_foreign_keys(body, &canonical_catalog.foreign_keys)?;
    body.digest(&returning_digest)
}

fn append_section(
    record: &mut Writer,
    tag: u16,
    encode: impl FnOnce(&mut Writer) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    record.u16(tag)?;
    record.u16(0)?;
    let length_offset = record.len();
    record.u32(0)?;
    let payload_start = record.len();
    encode(record)?;
    let length = record
        .len()
        .checked_sub(payload_start)
        .ok_or_else(|| codec_error("section payload length underflow"))?;
    record.patch_u32(length_offset, checked_u32(length, "section length")?);
    Ok(())
}

fn encode_target(batch: &TypedInsertBatch, out: &mut Writer) -> Result<(), EngineError> {
    out.identifier(batch.table.schema.as_ref())?;
    out.identifier(batch.table.name.as_ref())?;
    out.u32(batch.table.oid)?;
    out.digest(&batch.table.schema_digest)?;
    out.u32(batch.statement_ordinal.as_u32())?;
    out.u32(batch.row_count)?;
    out.u32(checked_u32(batch.columns.len(), "target column count")?)
}

fn encode_columns(
    columns: &[TypedInsertColumn],
    rows: u32,
    domains: &[TypedInsertDomainBinding],
    out: &mut Writer,
) -> Result<(), EngineError> {
    out.u32(checked_u32(columns.len(), "column count")?)?;
    for (ordinal, column) in columns.iter().enumerate() {
        encode_column(
            out,
            column,
            u32::try_from(ordinal).map_err(|_| codec_error("column ordinal"))?,
            rows,
            domains,
        )?;
    }
    Ok(())
}

fn encode_column(
    out: &mut Writer,
    column: &TypedInsertColumn,
    ordinal: u32,
    rows: u32,
    domains: &[TypedInsertDomainBinding],
) -> Result<(), EngineError> {
    let row_count = usize::try_from(rows).map_err(|_| codec_error("row count addressability"))?;
    if !column.full_invariants_hold(row_count) {
        return Err(codec_error(
            "sealed column invariant drifted before canonical encoding",
        ));
    }
    validate_column_domain(column, domains)?;
    out.u32(ordinal)?;
    out.identifier(column.name.as_ref())?;
    out.u32(column.column_id)?;
    out.i16(column.attnum)?;
    append_sql_type(out, column.ty)?;
    out.u32(column.type_oid)?;
    out.i16(column.type_size)?;
    out.option_u32(column.source_column_ordinal)?;
    out.option_u32(column.domain_dependency_ordinal)?;
    append_bitmap_form(out, &column.validity, rows, BitmapRole::Validity)?;
    append_presence_form(out, &column.presence, rows)?;
    append_default_form(out, &column.default_resolution, rows)?;
    out.u32(rows)?;
    for (state, provenance) in column.input_states.iter().zip(&column.input_provenance) {
        append_input_state(out, *state)?;
        append_input_provenance(out, *provenance)?;
    }
    validate_invalid_placeholders(column, row_count)?;
    append_values(out, &column.values, column.ty, rows)
}

fn encode_dependencies(
    dependencies: &[TypedInsertDependencyBinding],
    out: &mut Writer,
) -> Result<(), EngineError> {
    append_dependencies(out, dependencies)
}

fn append_dependencies(
    out: &mut Writer,
    dependencies: &[TypedInsertDependencyBinding],
) -> Result<(), EngineError> {
    out.u32(checked_u32(dependencies.len(), "dependency count")?)?;
    for (ordinal, dependency) in dependencies.iter().enumerate() {
        out.u32(checked_u32(ordinal, "dependency ordinal")?)?;
        out.u8(if ordinal == 0 { 1 } else { 2 })?;
        out.identifier(dependency.schema.as_ref())?;
        out.identifier(dependency.name.as_ref())?;
        out.u32(dependency.oid)?;
        out.digest(&dependency.schema_digest)?;
    }
    Ok(())
}

fn encode_domains(
    domains: &[TypedInsertDomainBinding],
    out: &mut Writer,
) -> Result<(), EngineError> {
    append_domains(out, domains)
}

fn append_domains(
    out: &mut Writer,
    domains: &[TypedInsertDomainBinding],
) -> Result<(), EngineError> {
    out.u32(checked_u32(domains.len(), "domain count")?)?;
    for (ordinal, domain) in domains.iter().enumerate() {
        out.u32(checked_u32(ordinal, "domain ordinal")?)?;
        out.identifier(domain.schema.as_ref())?;
        out.identifier(domain.name.as_ref())?;
        out.u32(domain.oid)?;
        append_sql_type(out, domain.base_type)?;
    }
    Ok(())
}

fn encode_indexes(
    indexes: &[TypedInsertCanonicalIndexBinding],
    out: &mut Writer,
) -> Result<(), EngineError> {
    append_indexes(out, indexes)
}

fn append_indexes(
    out: &mut Writer,
    indexes: &[TypedInsertCanonicalIndexBinding],
) -> Result<(), EngineError> {
    out.u32(checked_u32(indexes.len(), "index count")?)?;
    for index in indexes {
        append_index(out, index)?;
    }
    Ok(())
}

fn append_index(
    out: &mut Writer,
    index: &TypedInsertCanonicalIndexBinding,
) -> Result<(), EngineError> {
    out.u32(index.owner_dependency_ordinal)?;
    out.u32(index.raw_ordinal)?;
    out.u32(index.oid)?;
    out.identifier(index.name.as_ref())?;
    out.identifier(index.table_name.as_ref())?;
    out.identifier(index.first_column_name.as_ref())?;
    out.bool(index.unique)?;
    out.bool(index.primary_key)?;
    out.bool(index.unique_constraint)?;
    out.u32(checked_u32(index.key_columns.len(), "index key count")?)?;
    for column in &index.key_columns {
        append_catalog_column_binding(out, column)?;
    }
    Ok(())
}

fn encode_foreign_keys(
    foreign_keys: &[TypedInsertCanonicalForeignKeyBinding],
    out: &mut Writer,
) -> Result<(), EngineError> {
    append_foreign_keys(out, foreign_keys)
}

fn append_foreign_keys(
    out: &mut Writer,
    foreign_keys: &[TypedInsertCanonicalForeignKeyBinding],
) -> Result<(), EngineError> {
    out.u32(checked_u32(foreign_keys.len(), "foreign-key count")?)?;
    for foreign_key in foreign_keys {
        out.u32(foreign_key.raw_ordinal)?;
        out.identifier(foreign_key.name.as_ref())?;
        out.identifier(foreign_key.child_column_name.as_ref())?;
        out.identifier(foreign_key.referenced_table_name.as_ref())?;
        out.identifier(foreign_key.referenced_column_name.as_ref())?;
        append_catalog_column_binding(out, &foreign_key.child_column)?;
        out.u32(foreign_key.parent_dependency_ordinal)?;
        append_catalog_column_binding(out, &foreign_key.parent_column)?;
        append_index(out, &foreign_key.supporting_index)?;
    }
    Ok(())
}

fn append_catalog_column_binding(
    out: &mut Writer,
    column: &TypedInsertCanonicalColumnBinding,
) -> Result<(), EngineError> {
    out.u32(column.dependency_ordinal)?;
    out.u32(column.catalog_column_ordinal)?;
    out.u32(column.column_id)?;
    out.i16(column.attnum)?;
    out.identifier(column.name.as_ref())?;
    append_sql_type(out, column.ty)?;
    out.u32(column.type_oid)?;
    out.i16(column.type_size)
}

fn encode_returning(
    returning: &returning::BoundInsertReturning,
    digest: gpu_db_wal::CanonicalDigest,
    out: &mut Writer,
) -> Result<(), EngineError> {
    let shape = returning.effect_shape();
    out.u32(shape.row_count())?;
    out.u32(shape.column_count())?;
    out.u64(shape.cell_count())?;
    out.digest(&digest)?;
    out.u32(shape.column_count())?;
    for projection in returning.effect_projection_identities() {
        append_projection(
            out,
            projection.catalog_column_ordinal(),
            projection.column_id(),
            projection.attnum(),
            projection.name(),
            projection.ty(),
            projection.type_oid(),
            projection.type_size(),
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_projection(
    out: &mut Writer,
    catalog_column_ordinal: u32,
    column_id: u32,
    attnum: i16,
    name: &str,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
) -> Result<(), EngineError> {
    out.u32(catalog_column_ordinal)?;
    out.u32(column_id)?;
    out.i16(attnum)?;
    out.identifier(name)?;
    append_sql_type(out, ty)?;
    out.u32(type_oid)?;
    out.i16(type_size)
}

fn append_intent_columns(
    out: &mut Writer,
    columns: &[TypedInsertColumn],
    rows: u32,
    sequence_cells: &BTreeSet<(u32, u32)>,
) -> Result<(), EngineError> {
    out.u32(checked_u32(columns.len(), "intent column count")?)?;
    let rows_usize = usize::try_from(rows).map_err(|_| codec_error("row count addressability"))?;
    for (ordinal, column) in columns.iter().enumerate() {
        out.u32(u32::try_from(ordinal).map_err(|_| codec_error("intent column ordinal"))?)?;
        out.identifier(column.name.as_ref())?;
        out.u32(column.column_id)?;
        out.i16(column.attnum)?;
        append_sql_type(out, column.ty)?;
        out.u32(column.type_oid)?;
        out.i16(column.type_size)?;
        out.option_u32(column.source_column_ordinal)?;
        out.option_u32(column.domain_dependency_ordinal)?;
        out.u32(rows)?;
        for row in 0..rows_usize {
            let state = column.input_states[row];
            append_input_state(out, state)?;
            append_input_provenance(out, column.input_provenance[row])?;
            let is_sequence = sequence_cells.contains(&(ordinal as u32, row as u32));
            match state {
                TypedInsertInputState::Provided => {
                    out.u8(2)?;
                    append_value_at(out, &column.values, column.ty, row)?;
                }
                TypedInsertInputState::ProvidedNull => out.u8(1)?,
                TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault
                    if is_sequence =>
                {
                    out.u8(0)?;
                }
                TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault => {
                    if !column.default_resolution.was_defaulted(row) {
                        return Err(codec_error("deterministic default was not resolved"));
                    }
                    if column.validity.is_valid(row) {
                        out.u8(2)?;
                        append_value_at(out, &column.values, column.ty, row)?;
                    } else {
                        out.u8(1)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn append_sequence_parent(
    out: &mut Writer,
    parent: CanonicalSequenceParentView,
) -> Result<(), EngineError> {
    out.u64(parent.txn_id)?;
    out.bool(parent.autocommit)?;
    out.digest(&parent.request_digest)?;
    out.u32(parent.statement_ordinal.as_u32())?;
    out.u32(parent.expression_ordinal_base)
}

#[allow(clippy::too_many_arguments)]
fn append_sequence_request(
    out: &mut Writer,
    target_table_oid: u32,
    row_ordinal: u32,
    catalog_column_ordinal: u32,
    column_id: u32,
    sequence_oid: u32,
    source_name: &str,
    effective_name: &str,
    statement_ordinal: InsertStatementOrdinal,
    expression_ordinal: u32,
) -> Result<(), EngineError> {
    out.u32(target_table_oid)?;
    out.u32(row_ordinal)?;
    out.u32(catalog_column_ordinal)?;
    out.u32(column_id)?;
    out.u32(sequence_oid)?;
    out.identifier(source_name)?;
    out.identifier(effective_name)?;
    out.u32(statement_ordinal.as_u32())?;
    out.u32(expression_ordinal)
}

fn append_sql_type(out: &mut Writer, ty: SqlType) -> Result<(), EngineError> {
    let (tag, precision, scale) = match ty {
        SqlType::Int2 => (1, 0, 0),
        SqlType::Int4 => (2, 0, 0),
        SqlType::Int8 => (3, 0, 0),
        SqlType::Numeric { precision, scale } => (4, precision, scale),
        SqlType::Bool => (5, 0, 0),
        SqlType::Text => (6, 0, 0),
        SqlType::Date => (7, 0, 0),
        SqlType::Timestamp => (8, 0, 0),
        SqlType::Uuid => (9, 0, 0),
    };
    if tag == 4 && (precision == 0 || precision > 38 || scale > precision) {
        return Err(codec_error(
            "numeric typmod is outside the canonical v1 domain",
        ));
    }
    out.u8(tag)?;
    out.u8(precision)?;
    out.u8(scale)?;
    out.u8(0)
}

fn append_input_state(out: &mut Writer, state: TypedInsertInputState) -> Result<(), EngineError> {
    out.u8(match state {
        TypedInsertInputState::Provided => 1,
        TypedInsertInputState::ProvidedNull => 2,
        TypedInsertInputState::Omitted => 3,
        TypedInsertInputState::ExplicitDefault => 4,
    })
}

fn append_input_provenance(
    out: &mut Writer,
    provenance: TypedInsertInputProvenance,
) -> Result<(), EngineError> {
    match provenance {
        TypedInsertInputProvenance::Omitted => {
            out.u8(1)?;
            out.u32(0)
        }
        TypedInsertInputProvenance::Literal => {
            out.u8(2)?;
            out.u32(0)
        }
        TypedInsertInputProvenance::BoundParameter { index } if index != 0 => {
            out.u8(3)?;
            out.u32(index)
        }
        TypedInsertInputProvenance::BoundParameter { .. } => {
            Err(codec_error("bound parameter index is zero"))
        }
        TypedInsertInputProvenance::ProgrammaticValue => {
            out.u8(4)?;
            out.u32(0)
        }
        TypedInsertInputProvenance::SqlDefault => {
            out.u8(5)?;
            out.u32(0)
        }
        TypedInsertInputProvenance::ProgrammaticDefault => {
            out.u8(6)?;
            out.u32(0)
        }
    }
}

fn append_bitmap_form(
    out: &mut Writer,
    validity: &TypedInsertColumnValidity,
    rows: u32,
    _role: BitmapRole,
) -> Result<(), EngineError> {
    super::typed_image_codec::append_typed_validity(out, validity, rows)
}

fn append_presence_form(
    out: &mut Writer,
    presence: &TypedInsertColumnPresence,
    rows: u32,
) -> Result<(), EngineError> {
    let _ = rows;
    match presence {
        TypedInsertColumnPresence::AllProvided => out.u8(0),
        TypedInsertColumnPresence::Bitmap(_) => Err(codec_error(
            "sealed canonical record requires AllProvided presence",
        )),
    }
}

fn append_default_form(
    out: &mut Writer,
    resolution: &TypedInsertDefaultResolution,
    rows: u32,
) -> Result<(), EngineError> {
    match resolution {
        TypedInsertDefaultResolution::AllDirect => out.u8(0),
        TypedInsertDefaultResolution::Bitmap(words) => {
            if !bitmap_shape_is_exact(words, rows as usize) || words.iter().all(|word| *word == 0) {
                return Err(codec_error("default-resolution bitmap is not canonical"));
            }
            out.u8(1)?;
            append_bitmap_words(out, words)
        }
    }
}

fn append_bitmap_words(out: &mut Writer, words: &[u32]) -> Result<(), EngineError> {
    out.u32(checked_u32(words.len(), "bitmap word count")?)?;
    for word in words {
        out.u32(*word)?;
    }
    Ok(())
}

fn append_values(
    out: &mut Writer,
    values: &TypedInsertColumnValues,
    ty: SqlType,
    rows: u32,
) -> Result<(), EngineError> {
    super::typed_image_codec::append_typed_values(out, values, ty, rows)
}

// Test-only decoded-record reencoding intentionally permits selected malformed fixture vectors;
// it therefore needs only the fixed vector header, while canonical value authority stays in the
// shared typed-image codec above.
fn append_vector_header(
    out: &mut Writer,
    shape: u8,
    logical_elements: u32,
    payload_len: u32,
) -> Result<(), EngineError> {
    out.u8(shape)?;
    out.u32(logical_elements)?;
    out.u32(payload_len)
}

fn append_value_at(
    out: &mut Writer,
    values: &TypedInsertColumnValues,
    ty: SqlType,
    row: usize,
) -> Result<(), EngineError> {
    match (values, ty) {
        (TypedInsertColumnValues::I32(values), SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
            let value = *values
                .get(row)
                .ok_or_else(|| codec_error("i32 value row absent"))?;
            if ty == SqlType::Int2 && i16::try_from(value).is_err() {
                return Err(codec_error("int2 value out of range"));
            }
            out.i32(value)
        }
        (TypedInsertColumnValues::I64(values), SqlType::Int8 | SqlType::Timestamp) => out.i64(
            *values
                .get(row)
                .ok_or_else(|| codec_error("i64 value row absent"))?,
        ),
        (TypedInsertColumnValues::I128(values), SqlType::Numeric { .. }) => {
            let value = *values
                .get(row)
                .ok_or_else(|| codec_error("numeric value row absent"))?;
            validate_numeric_mantissa(value, ty)?;
            out.i128(value)
        }
        (TypedInsertColumnValues::Bytes16(values), SqlType::Uuid) => out.bytes(
            values
                .get(row)
                .ok_or_else(|| codec_error("uuid value row absent"))?,
        ),
        (TypedInsertColumnValues::BoolBits(words), SqlType::Bool) => {
            out.bool(bit_is_set(words, row))
        }
        (TypedInsertColumnValues::Text { offsets, bytes }, SqlType::Text) => {
            let start = usize::try_from(
                *offsets
                    .get(row)
                    .ok_or_else(|| codec_error("text start absent"))?,
            )
            .map_err(|_| codec_error("text start overflow"))?;
            let end = usize::try_from(
                *offsets
                    .get(row + 1)
                    .ok_or_else(|| codec_error("text end absent"))?,
            )
            .map_err(|_| codec_error("text end overflow"))?;
            out.blob(
                bytes
                    .get(start..end)
                    .ok_or_else(|| codec_error("text offset drifted"))?,
            )
        }
        _ => Err(codec_error("typed value arm disagrees with SQL type")),
    }
}

fn validate_batch(batch: &TypedInsertBatch) -> Result<(), EngineError> {
    if batch.table.schema.is_empty()
        || batch.table.name.is_empty()
        || batch.table.oid == 0
        || zero_digest(batch.table.schema_digest)
        || batch.columns.is_empty()
        || batch.dependencies.is_empty()
    {
        return Err(codec_error("typed batch target identity is incomplete"));
    }
    if batch.dependencies[0].schema != batch.table.schema
        || batch.dependencies[0].name != batch.table.name
        || batch.dependencies[0].oid != batch.table.oid
        || batch.dependencies[0].schema_digest != batch.table.schema_digest
    {
        return Err(codec_error(
            "target dependency is not canonical dependency zero",
        ));
    }
    let mut dependency_oids = BTreeSet::new();
    for dependency in &batch.dependencies {
        if dependency.schema.is_empty()
            || dependency.name.is_empty()
            || dependency.oid == 0
            || zero_digest(dependency.schema_digest)
            || !dependency_oids.insert(dependency.oid)
        {
            return Err(codec_error("dependency identity is not unique and stable"));
        }
    }
    validate_domains_and_columns(&batch.columns, &batch.domain_dependencies, batch.row_count)?;
    validate_batch_target_catalog_references(
        &batch.columns,
        &batch.canonical_catalog.indexes,
        &batch.canonical_catalog.foreign_keys,
    )?;
    validate_sequence_binding_targets(
        &batch.table,
        batch.row_count,
        &batch.columns,
        &batch.sequence_bindings,
    )?;
    validate_batch_dependency_closure_order(
        &batch.dependencies,
        &batch.canonical_catalog.foreign_keys,
    )?;
    validation::validate_catalog_closure(
        &batch.dependencies,
        &batch.canonical_catalog.indexes,
        &batch.canonical_catalog.foreign_keys,
    )?;
    let target_columns = batch
        .columns
        .iter()
        .map(|column| validation::GlobalTargetColumnIdentity {
            name: &column.name,
            column_id: column.column_id,
            attnum: column.attnum,
            ty: column.ty,
            type_oid: column.type_oid,
            type_size: column.type_size,
        })
        .collect::<Vec<_>>();
    let sequences = batch
        .sequence_bindings
        .iter()
        .map(|binding| {
            let request = binding.canonical_view().request;
            validation::GlobalSequenceDescriptor {
                oid: request.sequence_oid,
                effective_name: request.sequence_effective_name,
            }
        })
        .collect::<Vec<_>>();
    validation::validate_global_identity_registry(
        validation::GlobalTargetIdentity {
            schema: &batch.table.schema,
            name: &batch.table.name,
            oid: batch.table.oid,
            schema_digest: batch.table.schema_digest,
        },
        &target_columns,
        &batch.dependencies,
        &batch.domain_dependencies,
        &batch.canonical_catalog.indexes,
        &batch.canonical_catalog.foreign_keys,
        &sequences,
    )
}

fn validate_sequence_binding_targets(
    target: &TypedInsertBatchTable,
    rows: u32,
    columns: &[TypedInsertColumn],
    bindings: &[sequence_defaults::SequenceDefaultBinding],
) -> Result<(), EngineError> {
    for binding in bindings {
        let view = binding.canonical_view();
        let column = columns
            .get(view.request.catalog_column_ordinal as usize)
            .ok_or_else(|| codec_error("sequence binding column is absent"))?;
        let row = usize::try_from(view.request.row_ordinal)
            .map_err(|_| codec_error("sequence binding row overflows"))?;
        let resolved_value = match &column.values {
            TypedInsertColumnValues::I32(values) => values.get(row).copied(),
            _ => None,
        };
        validation::validate_sequence_target_cell(
            target.oid,
            rows,
            view.request,
            column.column_id,
            column.ty,
            column.input_states.get(row).copied(),
            column.default_resolution.was_defaulted(row),
            column.validity.is_valid(row),
            resolved_value,
            view.value,
        )?;
    }
    Ok(())
}

fn validate_batch_target_catalog_references(
    columns: &[TypedInsertColumn],
    indexes: &[TypedInsertCanonicalIndexBinding],
    foreign_keys: &[TypedInsertCanonicalForeignKeyBinding],
) -> Result<(), EngineError> {
    for index in indexes {
        if index.owner_dependency_ordinal != 0 {
            return Err(codec_error("target index owner dependency is not zero"));
        }
        for key in &index.key_columns {
            let column = columns
                .get(key.catalog_column_ordinal as usize)
                .ok_or_else(|| codec_error("target index key column is absent"))?;
            if key.dependency_ordinal != 0
                || key.column_id != column.column_id
                || key.attnum != column.attnum
                || key.name.as_ref() != column.name.as_ref()
                || key.ty != column.ty
                || key.type_oid != column.type_oid
                || key.type_size != column.type_size
            {
                return Err(codec_error("target index key identity is fabricated"));
            }
        }
    }
    for foreign_key in foreign_keys {
        let column = columns
            .get(foreign_key.child_column.catalog_column_ordinal as usize)
            .ok_or_else(|| codec_error("foreign-key child column is absent"))?;
        if foreign_key.child_column.dependency_ordinal != 0
            || foreign_key.child_column.column_id != column.column_id
            || foreign_key.child_column.attnum != column.attnum
            || foreign_key.child_column.name.as_ref() != column.name.as_ref()
            || foreign_key.child_column.ty != column.ty
            || foreign_key.child_column.type_oid != column.type_oid
            || foreign_key.child_column.type_size != column.type_size
        {
            return Err(codec_error(
                "foreign-key child column identity is fabricated",
            ));
        }
    }
    Ok(())
}

fn validate_batch_dependency_closure_order(
    dependencies: &[TypedInsertDependencyBinding],
    foreign_keys: &[TypedInsertCanonicalForeignKeyBinding],
) -> Result<(), EngineError> {
    let mut by_oid = BTreeMap::from([(dependencies[0].oid, 0_u32)]);
    let mut next = 1_u32;
    for foreign_key in foreign_keys {
        let dependency = dependencies
            .get(foreign_key.parent_dependency_ordinal as usize)
            .ok_or_else(|| codec_error("foreign-key parent dependency is absent"))?;
        if dependency.name.as_ref() != foreign_key.referenced_table_name.as_ref() {
            return Err(codec_error("foreign-key parent dependency name drifted"));
        }
        match by_oid.get(&dependency.oid) {
            Some(ordinal) if *ordinal == foreign_key.parent_dependency_ordinal => {}
            Some(_) => {
                return Err(codec_error(
                    "foreign-key parent has ambiguous dependency ordinal",
                ))
            }
            None if foreign_key.parent_dependency_ordinal == next => {
                by_oid.insert(dependency.oid, next);
                next = next
                    .checked_add(1)
                    .ok_or_else(|| codec_error("dependency ordinal overflows"))?;
            }
            None => {
                return Err(codec_error(
                    "FK parent dependency is not first-occurrence ordered",
                ))
            }
        }
    }
    if dependencies.len() != next as usize {
        return Err(codec_error("dependency closure has an unused extra table"));
    }
    Ok(())
}

fn validate_domains_and_columns(
    columns: &[TypedInsertColumn],
    domains: &[TypedInsertDomainBinding],
    rows: u32,
) -> Result<(), EngineError> {
    let mut ids = BTreeSet::new();
    let mut sources = BTreeSet::new();
    let mut first_domains = Vec::new();
    for column in columns {
        if column.name.is_empty()
            || column.column_id == 0
            || column.type_oid == 0
            || column.type_size != column.ty.type_size()
            || !ids.insert(column.column_id)
        {
            return Err(codec_error("catalog-order column identity is invalid"));
        }
        if let Some(source) = column.source_column_ordinal {
            if !sources.insert(source) {
                return Err(codec_error("source column ordinal is duplicated"));
            }
        }
        if let Some(domain) = column.domain_dependency_ordinal {
            let domain = domains
                .get(domain as usize)
                .ok_or_else(|| codec_error("column domain ordinal is absent"))?;
            if column.type_oid != domain.oid || column.ty != domain.base_type {
                return Err(codec_error("column domain binding drifted"));
            }
            if !first_domains
                .iter()
                .any(|prior: &&TypedInsertDomainBinding| prior.oid == domain.oid)
            {
                first_domains.push(domain);
            }
        } else if column.type_oid != column.ty.postgres_oid() {
            return Err(codec_error("base-type column has noncanonical type OID"));
        }
        if !column.full_invariants_hold(rows as usize) {
            return Err(codec_error("column semantic vectors drifted"));
        }
    }
    if sources
        .iter()
        .copied()
        .enumerate()
        .any(|(expected, actual)| u32::try_from(expected).ok() != Some(actual))
    {
        return Err(codec_error("source column ordinals are not contiguous"));
    }
    if first_domains.len() != domains.len()
        || !first_domains
            .iter()
            .zip(domains)
            .all(|(left, right)| std::ptr::eq(*left, right))
    {
        return Err(codec_error(
            "domain dependencies are not in first catalog-column order",
        ));
    }
    let mut domain_ids = BTreeSet::new();
    for domain in domains {
        if domain.schema.is_empty()
            || domain.name.is_empty()
            || domain.oid == 0
            || !domain_ids.insert(domain.oid)
        {
            return Err(codec_error("domain identity is invalid"));
        }
    }
    Ok(())
}

fn validate_column_domain(
    column: &TypedInsertColumn,
    domains: &[TypedInsertDomainBinding],
) -> Result<(), EngineError> {
    if let Some(ordinal) = column.domain_dependency_ordinal {
        let domain = domains
            .get(ordinal as usize)
            .ok_or_else(|| codec_error("column domain ordinal absent"))?;
        if column.type_oid != domain.oid || column.ty != domain.base_type {
            return Err(codec_error("column domain metadata drifted"));
        }
    } else if column.type_oid != column.ty.postgres_oid() {
        return Err(codec_error("base-type column OID drifted"));
    }
    Ok(())
}

fn validate_invalid_placeholders(
    column: &TypedInsertColumn,
    rows: usize,
) -> Result<(), EngineError> {
    for row in 0..rows {
        if column.validity.is_valid(row) {
            continue;
        }
        let zero = match &column.values {
            TypedInsertColumnValues::I32(values) => values.get(row) == Some(&0),
            TypedInsertColumnValues::I64(values) => values.get(row) == Some(&0),
            TypedInsertColumnValues::I128(values) => values.get(row) == Some(&0),
            TypedInsertColumnValues::Bytes16(values) => values.get(row) == Some(&[0; 16]),
            TypedInsertColumnValues::BoolBits(words) => !bit_is_set(words, row),
            TypedInsertColumnValues::Text { offsets, .. } => {
                offsets.get(row) == offsets.get(row + 1)
            }
        };
        if !zero {
            return Err(codec_error(
                "invalid typed value placeholder is not canonical zero",
            ));
        }
    }
    Ok(())
}

fn validate_numeric_mantissa(value: i128, ty: SqlType) -> Result<(), EngineError> {
    let SqlType::Numeric { precision, .. } = ty else {
        return Err(codec_error("numeric mantissa has a nonnumeric SQL type"));
    };
    if crate::numeric_exceeds_precision(value, precision) {
        return Err(codec_error("numeric mantissa exceeds declared precision"));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum BitmapRole {
    Validity,
}

fn zero_digest(digest: gpu_db_wal::CanonicalDigest) -> bool {
    digest == [0; 32]
}
fn checked_u32(value: usize, what: &str) -> Result<u32, EngineError> {
    u32::try_from(value).map_err(|_| codec_error(&format!("{what} exceeds u32")))
}
fn codec_error(message: &str) -> EngineError {
    EngineError::Durability(format!("typed INSERT canonical codec: {message}"))
}

struct Writer {
    bytes: Vec<u8>,
    len: usize,
    limit: usize,
    materialize: bool,
}
impl Writer {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            len: 0,
            limit,
            materialize: true,
        }
    }
    fn counting(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            len: 0,
            limit,
            materialize: false,
        }
    }
    fn len(&self) -> usize {
        self.len
    }
    fn finish(self) -> Vec<u8> {
        debug_assert!(
            self.materialize,
            "counting codec writer cannot materialize bytes"
        );
        debug_assert_eq!(self.bytes.len(), self.len);
        self.bytes
    }
    fn reserve(&mut self, count: usize) -> Result<(), EngineError> {
        let len = self
            .len
            .checked_add(count)
            .ok_or_else(|| codec_error("record length overflows"))?;
        if len > self.limit {
            return Err(codec_error(
                "record exceeds the 16 MiB one-fragment ceiling",
            ));
        }
        self.len = len;
        if self.materialize {
            self.bytes.reserve(count);
        }
        Ok(())
    }
    fn bytes(&mut self, bytes: &[u8]) -> Result<(), EngineError> {
        self.reserve(bytes.len())?;
        if self.materialize {
            self.bytes.extend_from_slice(bytes);
        }
        Ok(())
    }
    fn u8(&mut self, value: u8) -> Result<(), EngineError> {
        self.bytes(&[value])
    }
    fn bool(&mut self, value: bool) -> Result<(), EngineError> {
        self.u8(u8::from(value))
    }
    fn u16(&mut self, value: u16) -> Result<(), EngineError> {
        self.bytes(&value.to_le_bytes())
    }
    fn i16(&mut self, value: i16) -> Result<(), EngineError> {
        self.bytes(&value.to_le_bytes())
    }
    fn u32(&mut self, value: u32) -> Result<(), EngineError> {
        self.bytes(&value.to_le_bytes())
    }
    fn i32(&mut self, value: i32) -> Result<(), EngineError> {
        self.bytes(&value.to_le_bytes())
    }
    fn u64(&mut self, value: u64) -> Result<(), EngineError> {
        self.bytes(&value.to_le_bytes())
    }
    fn i64(&mut self, value: i64) -> Result<(), EngineError> {
        self.bytes(&value.to_le_bytes())
    }
    fn i128(&mut self, value: i128) -> Result<(), EngineError> {
        self.bytes(&value.to_le_bytes())
    }
    fn digest(&mut self, value: &gpu_db_wal::CanonicalDigest) -> Result<(), EngineError> {
        self.bytes(value)
    }
    fn identifier(&mut self, value: &str) -> Result<(), EngineError> {
        if value.is_empty() || value.len() > MAX_IDENTIFIER_BYTES || value.as_bytes().contains(&0) {
            return Err(codec_error(
                "identifier is empty, too long, or contains NUL",
            ));
        }
        self.blob(value.as_bytes())
    }
    fn blob(&mut self, bytes: &[u8]) -> Result<(), EngineError> {
        self.u32(checked_u32(bytes.len(), "blob length")?)?;
        self.bytes(bytes)
    }
    fn option_u32(&mut self, value: Option<u32>) -> Result<(), EngineError> {
        match value {
            None => self.u8(0),
            Some(value) => {
                self.u8(1)?;
                self.u32(value)
            }
        }
    }
    fn patch_u32(&mut self, start: usize, value: u32) {
        debug_assert!(
            start.checked_add(4).is_some_and(|end| end <= self.len),
            "canonical patch must remain inside the encoded record"
        );
        if self.materialize {
            self.bytes[start..start + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
}

impl super::typed_image_codec::TypedVectorSink for Writer {
    fn vector_bytes(&mut self, bytes: &[u8]) -> Result<(), EngineError> {
        self.bytes(bytes)
    }
}
