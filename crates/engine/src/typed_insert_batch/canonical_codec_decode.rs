//! Strict decoder for the inert canonical typed-INSERT v1 record.

use super::sequence::{
    append_private_owner, append_private_predecessor, private_child_digest, private_outcome_digest,
    sequence_input_digest, validate_private_effect_state, validate_sequence_parent,
    validate_sequence_request, PrivateChain, PrivateEffectEvidence, PrivateOwner,
    PrivatePredecessor,
};
use super::*;
use crate::typed_insert_batch::sequence_defaults::effects::{
    CanonicalSequenceEffectKindView, CanonicalSequenceEffectView, CanonicalSequenceParentView,
    CanonicalSequenceRequestView,
};

#[cfg(test)]
#[path = "canonical_codec_decode_forge.rs"]
mod forge;
#[path = "canonical_codec_decode/read_at.rs"]
#[allow(dead_code)] // Inert S7 recovery contract; exercised directly by codec tests.
mod read_at;
#[path = "canonical_codec_decode_reencode.rs"]
mod reencode;
#[path = "canonical_codec_decode_reservation.rs"]
mod reservation;
#[path = "canonical_codec_decode_sequence.rs"]
mod sequence_section;
#[path = "canonical_codec_decode_views.rs"]
mod views;
#[cfg(test)]
pub(super) use forge::{rehashed_forgery_for_test, RehashedForgery};
pub(crate) use read_at::CanonicalTypedInsertReadAt;
#[cfg(test)]
pub(crate) use reservation::note_typed_vector_owner_for_test;
pub(crate) use reservation::CanonicalTypedInsertDecodeMeasure;
#[cfg(test)]
pub(super) use reservation::{fail_at_for_test, fail_with_limits_for_test, observe_stats_for_test};
use reservation::{into_exact_boxed_slice, require_exact_vec, reserve_exact, reserve_string};
pub(crate) use views::{
    DecodedCatalogBindingFacts, DecodedCatalogColumnFacts, DecodedDependencyFacts,
    DecodedDomainFacts, DecodedForeignKeyFacts, DecodedIndexFacts, DecodedReturningLayoutFacts,
    DecodedReturningProjectionFacts, DecodedSequenceBindingFacts, DecodedSequenceEffectFacts,
    DecodedSequenceEffectKindFacts, DecodedSequenceParentFacts, DecodedSequenceRequestFacts,
    DecodedTypedInsertRecordFacts, DecodedTypedInsertTargetFacts,
    DecodedTypedInsertTargetIdentityFacts, DecodedTypedValueFacts,
};

fn parse_valid_model(bytes: &[u8]) -> Result<DecodedModel, EngineError> {
    if bytes.len() > MAX_RECORD_BYTES || bytes.len() < HEADER_LEN {
        return Err(codec_error("record length is outside canonical bounds"));
    }
    let header: &[u8; HEADER_LEN] = bytes[..HEADER_LEN]
        .try_into()
        .map_err(|_| codec_error("record prefix is truncated"))?;
    let prefix = parse_canonical_typed_insert_record_prefix(header, bytes.len())?;
    let mut reader = Reader::new(bytes);
    reader.take(HEADER_LEN)?;
    let recorded_statement_digest = prefix.typed_statement_digest;
    let recorded_returning_digest = prefix.returning_digest;
    let mut section_bytes: [&[u8]; SECTION_COUNT as usize] = [&[]; SECTION_COUNT as usize];
    for expected in 1..=SECTION_COUNT {
        let tag = reader.u16()?;
        if tag != expected || reader.u16()? != 0 {
            return Err(codec_error("section tag/order/flags are not canonical"));
        }
        let len =
            usize::try_from(reader.u32()?).map_err(|_| codec_error("section length overflows"))?;
        section_bytes[usize::from(expected - 1)] = reader.take(len)?;
    }
    if !reader.done() {
        return Err(codec_error(
            "record has trailing bytes after eight sections",
        ));
    }

    let mut sections = section_bytes.map(Reader::new);
    let target = read_target(&mut sections[0])?;
    let columns = read_columns(&mut sections[1], target.rows)?;
    let dependencies = read_dependencies(&mut sections[2])?;
    let domains = read_domains(&mut sections[3])?;
    let indexes = read_indexes(&mut sections[4])?;
    let foreign_keys = read_foreign_keys(&mut sections[5])?;
    let returning = read_returning(&mut sections[6])?;
    let effects = read_sequence_effects(
        &mut sections[7],
        &target,
        &columns,
        recorded_statement_digest,
    )?;
    if sections.iter().any(|section| !section.done()) {
        return Err(codec_error("length-delimited section has trailing bytes"));
    }
    validate_decoded_metadata(
        &target,
        &columns,
        &dependencies,
        &domains,
        &indexes,
        &foreign_keys,
        &returning,
        &effects,
    )?;
    if returning.digest != recorded_returning_digest
        || returning.digest != decoded_returning_digest(&returning)?
    {
        return Err(codec_error(
            "recorded RETURNING logical-layout digest drifted",
        ));
    }
    let recomputed_statement = decoded_statement_digest(
        &target,
        &columns,
        &dependencies,
        &domains,
        &indexes,
        &foreign_keys,
        returning.digest,
        &effects,
    )?;
    if recomputed_statement != recorded_statement_digest {
        return Err(codec_error("recorded typed statement digest drifted"));
    }
    Ok(DecodedModel {
        target,
        columns,
        dependencies,
        domains,
        indexes,
        foreign_keys,
        returning,
        effects,
        typed_statement_digest: recorded_statement_digest,
        returning_digest: recorded_returning_digest,
    })
}

struct Target {
    schema: String,
    name: String,
    oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
    statement_ordinal: InsertStatementOrdinal,
    rows: u32,
    column_count: u32,
}

struct DecodedColumn {
    ordinal: u32,
    name: String,
    column_id: u32,
    attnum: i16,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
    source_ordinal: Option<u32>,
    domain_ordinal: Option<u32>,
    validity: TypedInsertColumnValidity,
    presence: TypedInsertColumnPresence,
    defaults: TypedInsertDefaultResolution,
    states: Box<[TypedInsertInputState]>,
    provenance: Box<[TypedInsertInputProvenance]>,
    values: TypedInsertColumnValues,
}

/// Decoder-private catalog owners deliberately retain exact `String`/`Vec` storage rather than
/// converting untrusted text to the batch builder's shared `Arc` graph.  The decoded record is
/// inert; reencoding and S7 use borrowed views only.
struct DecodedDependency {
    schema: String,
    name: String,
    oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
}

struct DecodedDomain {
    schema: String,
    name: String,
    oid: u32,
    base_type: SqlType,
}

struct DecodedCatalogColumn {
    dependency_ordinal: u32,
    catalog_column_ordinal: u32,
    column_id: u32,
    attnum: i16,
    name: String,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
}

struct DecodedIndex {
    owner_dependency_ordinal: u32,
    raw_ordinal: u32,
    oid: u32,
    name: String,
    table_name: String,
    first_column_name: String,
    key_columns: Vec<DecodedCatalogColumn>,
    unique: bool,
    primary_key: bool,
    unique_constraint: bool,
}

struct DecodedForeignKey {
    raw_ordinal: u32,
    name: String,
    child_column_name: String,
    referenced_table_name: String,
    referenced_column_name: String,
    child_column: DecodedCatalogColumn,
    parent_dependency_ordinal: u32,
    parent_column: DecodedCatalogColumn,
    supporting_index: DecodedIndex,
}

struct DecodedProjection {
    catalog_column_ordinal: u32,
    column_id: u32,
    attnum: i16,
    name: String,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
}

struct DecodedReturning {
    rows: u32,
    columns: u32,
    cells: u64,
    digest: gpu_db_wal::CanonicalDigest,
    projections: Vec<DecodedProjection>,
}

#[derive(Clone, Copy)]
struct DecodedParent {
    txn_id: TxnId,
    autocommit: bool,
    request_digest: gpu_db_wal::CanonicalDigest,
    statement_ordinal: InsertStatementOrdinal,
    expression_base: u32,
}

#[derive(Clone)]
struct DecodedRequest {
    ordinal: u32,
    target_table_oid: u32,
    row_ordinal: u32,
    catalog_column_ordinal: u32,
    column_id: u32,
    sequence_oid: u32,
    source_name: String,
    effective_name: String,
    statement_ordinal: InsertStatementOrdinal,
    expression_ordinal: u32,
    absolute_expression_ordinal: u32,
    descriptor_digest: gpu_db_wal::CanonicalDigest,
    value: i64,
}

#[derive(Clone)]
enum DecodedEffectKind {
    Published {
        transition_txn_id: TxnId,
        input_digest: gpu_db_wal::CanonicalDigest,
        returned_value: i64,
    },
    Private {
        prior_last_value: i64,
        prior_is_called: bool,
        next_last_value: i64,
        next_is_called: bool,
        lifetime_origin: u8,
        owner: PrivateOwner,
        predecessor: PrivatePredecessor,
        input_digest: gpu_db_wal::CanonicalDigest,
        chain: PrivateChain,
    },
}

#[derive(Clone)]
struct DecodedEffect {
    request: DecodedRequest,
    kind: DecodedEffectKind,
}

struct DecodedEffects {
    parent: Option<DecodedParent>,
    effects: Vec<DecodedEffect>,
}

pub(super) struct DecodedModel {
    target: Target,
    columns: Vec<DecodedColumn>,
    dependencies: Vec<DecodedDependency>,
    domains: Vec<DecodedDomain>,
    indexes: Vec<DecodedIndex>,
    foreign_keys: Vec<DecodedForeignKey>,
    returning: DecodedReturning,
    effects: DecodedEffects,
    typed_statement_digest: gpu_db_wal::CanonicalDigest,
    returning_digest: gpu_db_wal::CanonicalDigest,
}

pub(super) fn decode(bytes: &[u8]) -> Result<DecodedTypedInsertRecord, EngineError> {
    let model = parse_valid_model(bytes)?;
    let canonical = reencode::reencode_decoded(&model)?;
    if canonical.as_slice() != bytes {
        return Err(codec_error(
            "decoded logical model does not reencode to the input bytes",
        ));
    }
    // The temporary reencode proves exact canonical identity, but raw input/reencoded bytes are
    // not retained as a second authority. The private model is the sole decoded evidence.
    drop(canonical);
    Ok(DecodedTypedInsertRecord { model })
}

#[allow(dead_code)]
pub(crate) fn measure_decoded_canonical_typed_insert_from_source<
    S: CanonicalTypedInsertReadAt + ?Sized,
>(
    source: &S,
) -> Result<CanonicalTypedInsertDecodeMeasure, EngineError> {
    read_at::measure_from_source(source)
}

#[allow(dead_code)]
pub(crate) fn measure_decoded_canonical_typed_insert_published_only_from_source<
    S: CanonicalTypedInsertReadAt + ?Sized,
>(
    source: &S,
) -> Result<CanonicalTypedInsertDecodeMeasure, EngineError> {
    read_at::measure_published_only_from_source(source)
}

#[allow(dead_code)]
pub(crate) fn copy_decoded_canonical_typed_insert_after_measure<
    S: CanonicalTypedInsertReadAt + ?Sized,
>(
    source: &S,
    measure: CanonicalTypedInsertDecodeMeasure,
    destination: &mut [u8],
) -> Result<(), EngineError> {
    read_at::copy_after_measure(source, measure, destination)
}

#[allow(dead_code)]
pub(crate) fn copy_decoded_canonical_typed_insert_published_only_after_measure<
    S: CanonicalTypedInsertReadAt + ?Sized,
>(
    source: &S,
    measure: CanonicalTypedInsertDecodeMeasure,
    destination: &mut [u8],
) -> Result<(), EngineError> {
    read_at::copy_published_only_after_measure(source, measure, destination)
}

#[allow(dead_code)]
pub(crate) fn decode_decoded_canonical_typed_insert_after_measure(
    bytes: &[u8],
    measure: CanonicalTypedInsertDecodeMeasure,
) -> Result<DecodedTypedInsertRecord, EngineError> {
    read_at::decode_after_measure(bytes, measure)
}

#[allow(dead_code)]
pub(crate) fn decode_decoded_canonical_typed_insert_published_only_after_measure(
    bytes: &[u8],
    measure: CanonicalTypedInsertDecodeMeasure,
) -> Result<DecodedTypedInsertRecord, EngineError> {
    read_at::decode_published_only_after_measure(bytes, measure)
}

pub(super) fn record_facts(model: &DecodedModel) -> DecodedTypedInsertRecordFacts {
    views::record_facts(model)
}

pub(super) fn target_identity(model: &DecodedModel) -> DecodedTypedInsertTargetIdentityFacts<'_> {
    views::target_identity(model)
}

pub(super) fn column_value_at(
    model: &DecodedModel,
    catalog_column_ordinal: u32,
    row_ordinal: u32,
) -> Result<(bool, DecodedTypedValueFacts<'_>), EngineError> {
    views::column_value_at(model, catalog_column_ordinal, row_ordinal)
}

pub(super) fn sequence_parent_facts(model: &DecodedModel) -> Option<DecodedSequenceParentFacts> {
    views::sequence_parent_facts(model)
}

pub(super) fn sequence_effect_facts(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedSequenceEffectFacts> + '_ {
    views::sequence_effect_facts(model)
}

pub(super) fn catalog_columns(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedCatalogColumnFacts<'_>> {
    views::catalog_columns(model)
}

pub(super) fn dependencies(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedDependencyFacts<'_>> {
    views::dependencies(model)
}

pub(super) fn domains(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedDomainFacts<'_>> {
    views::domains(model)
}

pub(super) fn indexes(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedIndexFacts<'_>> {
    views::indexes(model)
}

pub(super) fn index_key_columns(
    model: &DecodedModel,
    index_ordinal: u32,
) -> Result<impl ExactSizeIterator<Item = DecodedCatalogBindingFacts<'_>>, EngineError> {
    views::index_key_columns(model, index_ordinal)
}

pub(super) fn foreign_keys(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedForeignKeyFacts<'_>> {
    views::foreign_keys(model)
}

pub(super) fn foreign_key_supporting_index_keys(
    model: &DecodedModel,
    foreign_key_ordinal: u32,
) -> Result<impl ExactSizeIterator<Item = DecodedCatalogBindingFacts<'_>>, EngineError> {
    views::foreign_key_supporting_index_keys(model, foreign_key_ordinal)
}

pub(super) fn sequence_bindings(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedSequenceBindingFacts<'_>> {
    views::sequence_bindings(model)
}

pub(super) fn returning_projections(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedReturningProjectionFacts<'_>> {
    views::returning_projections(model)
}

#[cfg(test)]
pub(super) fn reencode_decoded(model: &DecodedModel) -> Result<Vec<u8>, EngineError> {
    reencode::reencode_decoded(model)
}

fn read_target(reader: &mut Reader<'_>) -> Result<Target, EngineError> {
    let target = Target {
        schema: reader.identifier()?,
        name: reader.identifier()?,
        oid: reader.u32()?,
        schema_digest: reader.digest()?,
        statement_ordinal: InsertStatementOrdinal::from_u32(reader.u32()?),
        rows: reader.u32()?,
        column_count: reader.u32()?,
    };
    if target.oid == 0 || zero_digest(target.schema_digest) {
        return Err(codec_error("target identity is invalid"));
    }
    Ok(target)
}

fn read_columns(reader: &mut Reader<'_>, rows: u32) -> Result<Vec<DecodedColumn>, EngineError> {
    let count = reader.count_bounded("column", 38)?;
    let mut columns = Vec::new();
    reserve_exact(&mut columns, count, "decoded column directory")?;
    for expected in 0..count {
        let ordinal = reader.u32()?;
        if ordinal != expected as u32 {
            return Err(codec_error("catalog column order is not canonical"));
        }
        let name = reader.identifier()?;
        let column_id = reader.u32()?;
        let attnum = reader.i16()?;
        let ty = reader.sql_type()?;
        let type_oid = reader.u32()?;
        let type_size = reader.i16()?;
        let source_ordinal = reader.option_u32()?;
        let domain_ordinal = reader.option_u32()?;
        let validity = reader.validity(rows)?;
        let presence = reader.presence(rows)?;
        let defaults = reader.defaults(rows)?;
        if reader.u32()? != rows {
            return Err(codec_error("column state vector row count drifted"));
        }
        if usize::try_from(rows)
            .ok()
            .is_none_or(|rows| rows > reader.remaining() / 6)
        {
            return Err(codec_error(
                "column row count cannot fit its remaining state bytes",
            ));
        }
        let row_count = usize::try_from(rows).map_err(|_| codec_error("row count overflows"))?;
        let mut states = Vec::new();
        reserve_exact(&mut states, row_count, "decoded input-state vector")?;
        let mut provenance = Vec::new();
        reserve_exact(&mut provenance, row_count, "decoded provenance vector")?;
        for _ in 0..rows {
            states.push(reader.input_state()?);
            provenance.push(reader.input_provenance()?);
        }
        let values = reader.values(ty, rows)?;
        columns.push(DecodedColumn {
            ordinal,
            name,
            column_id,
            attnum,
            ty,
            type_oid,
            type_size,
            source_ordinal,
            domain_ordinal,
            validity,
            presence,
            defaults,
            states: into_exact_boxed_slice(states)?,
            provenance: into_exact_boxed_slice(provenance)?,
            values,
        });
    }
    require_exact_vec(&columns)?;
    Ok(columns)
}

fn read_dependencies(reader: &mut Reader<'_>) -> Result<Vec<DecodedDependency>, EngineError> {
    let count = reader.count_bounded("dependency", 51)?;
    let mut dependencies = Vec::new();
    reserve_exact(&mut dependencies, count, "decoded dependency directory")?;
    for ordinal in 0..count {
        if reader.u32()? != ordinal as u32 || reader.u8()? != if ordinal == 0 { 1 } else { 2 } {
            return Err(codec_error("dependency ordinal or role is noncanonical"));
        }
        dependencies.push(DecodedDependency {
            schema: reader.identifier()?,
            name: reader.identifier()?,
            oid: reader.u32()?,
            schema_digest: reader.digest()?,
        });
    }
    require_exact_vec(&dependencies)?;
    Ok(dependencies)
}

fn read_domains(reader: &mut Reader<'_>) -> Result<Vec<DecodedDomain>, EngineError> {
    let count = reader.count_bounded("domain", 18)?;
    let mut domains = Vec::new();
    reserve_exact(&mut domains, count, "decoded domain directory")?;
    for ordinal in 0..count {
        if reader.u32()? != ordinal as u32 {
            return Err(codec_error("domain ordinal is noncanonical"));
        }
        domains.push(DecodedDomain {
            schema: reader.identifier()?,
            name: reader.identifier()?,
            oid: reader.u32()?,
            base_type: reader.sql_type()?,
        });
    }
    require_exact_vec(&domains)?;
    Ok(domains)
}

fn read_indexes(reader: &mut Reader<'_>) -> Result<Vec<DecodedIndex>, EngineError> {
    let count = reader.count_bounded("index", 36)?;
    let mut indexes = Vec::new();
    reserve_exact(&mut indexes, count, "decoded index directory")?;
    for _ in 0..count {
        indexes.push(read_index(reader)?);
    }
    require_exact_vec(&indexes)?;
    Ok(indexes)
}

fn read_index(reader: &mut Reader<'_>) -> Result<DecodedIndex, EngineError> {
    let owner_dependency_ordinal = reader.u32()?;
    let raw_ordinal = reader.u32()?;
    let oid = reader.u32()?;
    let name = reader.identifier()?;
    let table_name = reader.identifier()?;
    let first_column_name = reader.identifier()?;
    let unique = reader.bool()?;
    let primary_key = reader.bool()?;
    let unique_constraint = reader.bool()?;
    let count = reader.count_bounded("index key", 25)?;
    let mut key_columns = Vec::new();
    reserve_exact(&mut key_columns, count, "decoded index key directory")?;
    for _ in 0..count {
        key_columns.push(read_catalog_column_binding(reader)?);
    }
    require_exact_vec(&key_columns)?;
    Ok(DecodedIndex {
        owner_dependency_ordinal,
        raw_ordinal,
        oid,
        name,
        table_name,
        first_column_name,
        key_columns,
        unique,
        primary_key,
        unique_constraint,
    })
}

fn read_catalog_column_binding(
    reader: &mut Reader<'_>,
) -> Result<DecodedCatalogColumn, EngineError> {
    Ok(DecodedCatalogColumn {
        dependency_ordinal: reader.u32()?,
        catalog_column_ordinal: reader.u32()?,
        column_id: reader.u32()?,
        attnum: reader.i16()?,
        name: reader.identifier()?,
        ty: reader.sql_type()?,
        type_oid: reader.u32()?,
        type_size: reader.i16()?,
    })
}

fn read_foreign_keys(reader: &mut Reader<'_>) -> Result<Vec<DecodedForeignKey>, EngineError> {
    let count = reader.count_bounded("foreign-key", 64)?;
    let mut foreign_keys = Vec::new();
    reserve_exact(&mut foreign_keys, count, "decoded foreign-key directory")?;
    for _ in 0..count {
        foreign_keys.push(DecodedForeignKey {
            raw_ordinal: reader.u32()?,
            name: reader.identifier()?,
            child_column_name: reader.identifier()?,
            referenced_table_name: reader.identifier()?,
            referenced_column_name: reader.identifier()?,
            child_column: read_catalog_column_binding(reader)?,
            parent_dependency_ordinal: reader.u32()?,
            parent_column: read_catalog_column_binding(reader)?,
            supporting_index: read_index(reader)?,
        });
    }
    require_exact_vec(&foreign_keys)?;
    Ok(foreign_keys)
}

fn read_returning(reader: &mut Reader<'_>) -> Result<DecodedReturning, EngineError> {
    let rows = reader.u32()?;
    let columns = reader.u32()?;
    let cells = reader.u64()?;
    let digest = reader.digest()?;
    if reader.u32()? != columns || u64::from(rows).checked_mul(u64::from(columns)) != Some(cells) {
        return Err(codec_error("RETURNING geometry is inconsistent"));
    }
    if usize::try_from(columns)
        .ok()
        .is_none_or(|count| count > reader.remaining() / 25)
    {
        return Err(codec_error(
            "RETURNING projection count cannot fit remaining bytes",
        ));
    }
    let projection_count =
        usize::try_from(columns).map_err(|_| codec_error("projection count overflows"))?;
    let mut projections = Vec::new();
    reserve_exact(
        &mut projections,
        projection_count,
        "decoded RETURNING projection directory",
    )?;
    for _ in 0..columns {
        projections.push(DecodedProjection {
            catalog_column_ordinal: reader.u32()?,
            column_id: reader.u32()?,
            attnum: reader.i16()?,
            name: reader.identifier()?,
            ty: reader.sql_type()?,
            type_oid: reader.u32()?,
            type_size: reader.i16()?,
        });
    }
    require_exact_vec(&projections)?;
    Ok(DecodedReturning {
        rows,
        columns,
        cells,
        digest,
        projections,
    })
}

fn read_sequence_effects(
    reader: &mut Reader<'_>,
    target: &Target,
    columns: &[DecodedColumn],
    expected_request_digest: gpu_db_wal::CanonicalDigest,
) -> Result<DecodedEffects, EngineError> {
    let parent = reader.bool()?.then(|| read_parent(reader)).transpose()?;
    let count = reader.count_bounded("sequence effect", 58)?;
    if parent.is_none() != (count == 0) {
        return Err(codec_error(
            "sequence parent optional form is not canonical",
        ));
    }
    if let Some(parent) = parent {
        validate_sequence_parent(
            parent.view(),
            expected_request_digest,
            target.statement_ordinal,
        )?;
    }
    let mut effects: Vec<DecodedEffect> = Vec::new();
    reserve_exact(&mut effects, count, "decoded sequence-effect directory")?;
    let mut previous_key = None;
    for expected in 0..count {
        let ordinal = reader.u32()?;
        if ordinal != expected as u32 {
            return Err(codec_error("sequence binding ordinal is not canonical"));
        }
        let mut request = read_request(reader, ordinal)?;
        request.absolute_expression_ordinal = reader.u32()?;
        request.descriptor_digest = reader.digest()?;
        request.value = reader.i64()?;
        let parent_value = parent.expect("nonempty canonical effects have a parent");
        let request_view = request.view();
        validate_sequence_request(request_view, parent_value.view(), None)?;
        let current_key = (
            request.row_ordinal,
            request.catalog_column_ordinal,
            request.expression_ordinal,
        );
        if previous_key.is_some_and(|previous| previous >= current_key) {
            return Err(codec_error(
                "sequence requests are not row-major/catalog ordered",
            ));
        }
        if request.absolute_expression_ordinal
            != parent_value
                .expression_base
                .checked_add(request.expression_ordinal)
                .ok_or_else(|| codec_error("sequence expression ordinal overflow"))?
            || request.descriptor_digest
                != crate::sequence_descriptor_digest(request.sequence_oid, &request.effective_name)
            || effects.iter().any(|prior| {
                prior.request.row_ordinal == request.row_ordinal
                    && prior.request.catalog_column_ordinal == request.catalog_column_ordinal
            })
        {
            return Err(codec_error(
                "sequence request descriptor/order/target drifted",
            ));
        }
        let kind = match reader.u8()? {
            1 => {
                let transition_txn_id = reader.u64()?;
                let input_digest = reader.digest()?;
                let returned_value = reader.i64()?;
                if transition_txn_id == 0
                    || effects.iter().any(|prior| {
                        matches!(prior.kind, DecodedEffectKind::Published { transition_txn_id: prior_id, .. } if prior_id == transition_txn_id)
                    })
                    || returned_value != request.value
                    || input_digest
                        != sequence_input_digest(
                            parent_value.view(),
                            request_view,
                            request.absolute_expression_ordinal,
                        )
                {
                    return Err(codec_error("published sequence receipt is invalid"));
                }
                DecodedEffectKind::Published {
                    transition_txn_id,
                    input_digest,
                    returned_value,
                }
            }
            2 => {
                let prior_last_value = reader.i64()?;
                let prior_is_called = reader.bool()?;
                let next_last_value = reader.i64()?;
                let next_is_called = reader.bool()?;
                let lifetime_origin = reader.u8()?;
                let owner = read_private_owner(reader)?;
                let predecessor = read_private_predecessor(reader, owner)?;
                let input_digest = reader.digest()?;
                let descriptor = request.descriptor_digest;
                let child = private_child_digest(
                    parent_value.view(),
                    request_view,
                    request.absolute_expression_ordinal,
                    input_digest,
                    descriptor,
                    lifetime_origin,
                    (prior_last_value, prior_is_called),
                    predecessor,
                )?;
                let outcome = private_outcome_digest(
                    child,
                    owner,
                    request.value,
                    (next_last_value, next_is_called),
                )?;
                let previous = effects.iter().rev().find_map(|prior| {
                    (prior.request.sequence_oid == request.sequence_oid)
                        .then_some(match prior.kind {
                            DecodedEffectKind::Private { chain, .. } => Some(chain),
                            DecodedEffectKind::Published { .. } => None,
                        })
                        .flatten()
                });
                let chain = validate_private_effect_state(
                    CanonicalSequenceEffectView {
                        request: request_view,
                        value: request.value,
                        kind: CanonicalSequenceEffectKindView::Private {
                            parent: parent_value.view(),
                            prior_last_value,
                            prior_is_called,
                            next_last_value,
                            next_is_called,
                            lifetime_origin,
                            owner_kind: owner.kind,
                            owner_statement_ordinal: owner.statement_ordinal,
                            owner_statement_digest: owner.statement_digest,
                            owner_creator_catalog_column_ordinal: owner
                                .creator_catalog_column_ordinal,
                            predecessor_tag: match predecessor {
                                PrivatePredecessor::Lifecycle(_) => 1,
                                PrivatePredecessor::Outcome(_) => 2,
                            },
                            predecessor_digest: match predecessor {
                                PrivatePredecessor::Lifecycle(owner) => owner.statement_digest,
                                PrivatePredecessor::Outcome(digest) => digest,
                            },
                            input_digest,
                            descriptor_digest: descriptor,
                            child_digest: child,
                            outcome_digest: outcome,
                        },
                    },
                    parent_value.view(),
                    request.absolute_expression_ordinal,
                    descriptor,
                    PrivateEffectEvidence {
                        prior_last_value,
                        prior_is_called,
                        next_last_value,
                        next_is_called,
                        lifetime_origin,
                        owner,
                        predecessor,
                        input_digest,
                        descriptor_digest: descriptor,
                        child_digest: child,
                        outcome_digest: outcome,
                    },
                    previous,
                )?;
                DecodedEffectKind::Private {
                    prior_last_value,
                    prior_is_called,
                    next_last_value,
                    next_is_called,
                    lifetime_origin,
                    owner,
                    predecessor,
                    input_digest,
                    chain,
                }
            }
            _ => return Err(codec_error("sequence effect tag is unknown")),
        };
        sequence_section::validate_sequence_vector_target(&request, target, columns)?;
        previous_key = Some(current_key);
        effects.push(DecodedEffect { request, kind });
    }
    if let Some(parent_value) = parent {
        sequence_section::validate_decoded_sequence_section(parent_value, &effects)?;
    }
    require_exact_vec(&effects)?;
    Ok(DecodedEffects { parent, effects })
}

fn read_parent(reader: &mut Reader<'_>) -> Result<DecodedParent, EngineError> {
    let parent = DecodedParent {
        txn_id: reader.u64()?,
        autocommit: reader.bool()?,
        request_digest: reader.digest()?,
        statement_ordinal: InsertStatementOrdinal::from_u32(reader.u32()?),
        expression_base: reader.u32()?,
    };
    if parent.txn_id == 0 || zero_digest(parent.request_digest) {
        return Err(codec_error("sequence parent identity is invalid"));
    }
    Ok(parent)
}

fn read_request(reader: &mut Reader<'_>, ordinal: u32) -> Result<DecodedRequest, EngineError> {
    Ok(DecodedRequest {
        ordinal,
        target_table_oid: reader.u32()?,
        row_ordinal: reader.u32()?,
        catalog_column_ordinal: reader.u32()?,
        column_id: reader.u32()?,
        sequence_oid: reader.u32()?,
        source_name: reader.identifier()?,
        effective_name: reader.identifier()?,
        statement_ordinal: InsertStatementOrdinal::from_u32(reader.u32()?),
        expression_ordinal: reader.u32()?,
        absolute_expression_ordinal: 0,
        descriptor_digest: [0; 32],
        value: 0,
    })
}

fn read_private_owner(reader: &mut Reader<'_>) -> Result<PrivateOwner, EngineError> {
    let owner = PrivateOwner {
        kind: reader.u8()?,
        statement_ordinal: reader.u32()?,
        statement_digest: reader.digest()?,
        creator_catalog_column_ordinal: reader.option_u32()?,
    };
    if !matches!(owner.kind, 1..=3) || zero_digest(owner.statement_digest) {
        return Err(codec_error("private owner tag or identity is invalid"));
    }
    Ok(owner)
}

fn read_private_predecessor(
    reader: &mut Reader<'_>,
    owner: PrivateOwner,
) -> Result<PrivatePredecessor, EngineError> {
    match reader.u8()? {
        1 => {
            let predecessor_owner = read_private_owner(reader)?;
            if predecessor_owner != owner {
                return Err(codec_error(
                    "lifecycle predecessor differs from private owner",
                ));
            }
            Ok(PrivatePredecessor::Lifecycle(owner))
        }
        2 => {
            let digest = reader.digest()?;
            if zero_digest(digest) {
                Err(codec_error("private outcome predecessor is zero"))
            } else {
                Ok(PrivatePredecessor::Outcome(digest))
            }
        }
        _ => Err(codec_error("private predecessor tag is unknown")),
    }
}

impl DecodedParent {
    fn view(self) -> CanonicalSequenceParentView {
        CanonicalSequenceParentView {
            txn_id: self.txn_id,
            autocommit: self.autocommit,
            request_digest: self.request_digest,
            statement_ordinal: self.statement_ordinal,
            expression_ordinal_base: self.expression_base,
        }
    }
}
impl DecodedRequest {
    fn view(&self) -> CanonicalSequenceRequestView<'_> {
        CanonicalSequenceRequestView {
            target_table_oid: self.target_table_oid,
            row_ordinal: self.row_ordinal,
            catalog_column_ordinal: self.catalog_column_ordinal,
            column_id: self.column_id,
            sequence_oid: self.sequence_oid,
            sequence_source_name: &self.source_name,
            sequence_effective_name: &self.effective_name,
            statement_ordinal: self.statement_ordinal,
            expression_ordinal: self.expression_ordinal,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_decoded_metadata(
    target: &Target,
    columns: &[DecodedColumn],
    dependencies: &[DecodedDependency],
    domains: &[DecodedDomain],
    indexes: &[DecodedIndex],
    foreign_keys: &[DecodedForeignKey],
    returning: &DecodedReturning,
    effects: &DecodedEffects,
) -> Result<(), EngineError> {
    if columns.len() != target.column_count as usize
        || columns.is_empty()
        || dependencies.is_empty()
        || dependencies[0].schema.as_ref() != target.schema
        || dependencies[0].name.as_ref() != target.name
        || dependencies[0].oid != target.oid
        || dependencies[0].schema_digest != target.schema_digest
    {
        return Err(codec_error("target dependency zero drifted"));
    }
    for dependency in dependencies {
        if dependency.schema.is_empty()
            || dependency.name.is_empty()
            || dependency.oid == 0
            || zero_digest(dependency.schema_digest)
            || dependencies
                .iter()
                .take_while(|prior| !std::ptr::eq(*prior, dependency))
                .any(|prior| prior.oid == dependency.oid)
        {
            return Err(codec_error("dependency identity is invalid"));
        }
    }
    validate_decoded_columns(columns, domains, target.rows)?;
    if returning.rows != target.rows {
        return Err(codec_error("RETURNING row geometry differs from target"));
    }
    for projection in &returning.projections {
        let column = columns
            .get(projection.catalog_column_ordinal as usize)
            .ok_or_else(|| codec_error("RETURNING projection catalog column is absent"))?;
        if projection.column_id != column.column_id
            || projection.attnum != column.attnum
            || projection.name != column.name
            || projection.ty != column.ty
            || projection.type_oid != column.type_oid
            || projection.type_size != column.type_size
        {
            return Err(codec_error("RETURNING projection identity is fabricated"));
        }
    }
    validate_target_catalog_references(columns, indexes, foreign_keys)?;
    validate_dependency_closure_order(dependencies, foreign_keys)?;
    validate_decoded_catalog_closure(dependencies, indexes, foreign_keys)?;
    validate_decoded_global_identity_registry(
        target,
        columns,
        dependencies,
        domains,
        indexes,
        foreign_keys,
        effects,
    )
}

/// Decoder counterpart to the producer's catalog closure registry.  The model has already
/// reserved each retained directory, so these checks deliberately rescan it instead of creating
/// attacker-sized maps/sets.  Every duplicate key below must repeat its complete identity.
fn validate_decoded_catalog_closure(
    dependencies: &[DecodedDependency],
    indexes: &[DecodedIndex],
    foreign_keys: &[DecodedForeignKey],
) -> Result<(), EngineError> {
    for (ordinal, index) in indexes.iter().enumerate() {
        if index.owner_dependency_ordinal != 0
            || index.raw_ordinal != ordinal as u32
            || indexes[..ordinal]
                .iter()
                .any(|prior| prior.oid == index.oid || prior.name == index.name)
        {
            return Err(codec_error("target index order or identity drifted"));
        }
        validate_decoded_index(index, dependencies)?;
        for prior in &indexes[..ordinal] {
            validate_same_primary_index(prior, index)?;
            validate_decoded_catalog_column_sets(&prior.key_columns, &index.key_columns)?;
        }
    }
    for (ordinal, foreign_key) in foreign_keys.iter().enumerate() {
        if foreign_key.raw_ordinal != ordinal as u32
            || foreign_keys[..ordinal]
                .iter()
                .any(|prior| prior.name == foreign_key.name)
            || foreign_key.child_column.dependency_ordinal != 0
            || foreign_key.parent_column.dependency_ordinal != foreign_key.parent_dependency_ordinal
        {
            return Err(codec_error(
                "foreign-key order or dependency identity drifted",
            ));
        }
        validate_decoded_catalog_column(&foreign_key.child_column, dependencies)?;
        validate_decoded_catalog_column(&foreign_key.parent_column, dependencies)?;
        if foreign_key.child_column.name != foreign_key.child_column_name
            || foreign_key.parent_column.name != foreign_key.referenced_column_name
            || dependencies
                .get(foreign_key.parent_dependency_ordinal as usize)
                .map(|dependency| dependency.name.as_str())
                != Some(foreign_key.referenced_table_name.as_str())
            || foreign_key.child_column.ty != foreign_key.parent_column.ty
        {
            return Err(codec_error("foreign-key column/type binding drifted"));
        }
        validate_decoded_index(&foreign_key.supporting_index, dependencies)?;
        if foreign_key.supporting_index.owner_dependency_ordinal
            != foreign_key.parent_dependency_ordinal
            || !foreign_key.supporting_index.unique
            || !(foreign_key.supporting_index.primary_key
                || foreign_key.supporting_index.unique_constraint)
            || foreign_key.supporting_index.key_columns.len() != 1
            || !same_decoded_catalog_column(
                &foreign_key.supporting_index.key_columns[0],
                &foreign_key.parent_column,
            )
        {
            return Err(codec_error("foreign-key supporting unique index drifted"));
        }
        for index in indexes {
            validate_same_primary_index(index, &foreign_key.supporting_index)?;
            validate_decoded_catalog_column_sets(
                &index.key_columns,
                &foreign_key.supporting_index.key_columns,
            )?;
            if index.oid == foreign_key.supporting_index.oid
                && !same_decoded_index(index, &foreign_key.supporting_index)
            {
                return Err(codec_error("catalog index OID identity drifted"));
            }
        }
        for prior in &foreign_keys[..ordinal] {
            validate_same_primary_index(&prior.supporting_index, &foreign_key.supporting_index)?;
            validate_decoded_catalog_column_pair(&prior.child_column, &foreign_key.child_column)?;
            validate_decoded_catalog_column_pair(&prior.parent_column, &foreign_key.parent_column)?;
            validate_decoded_catalog_column_sets(
                &prior.supporting_index.key_columns,
                &foreign_key.supporting_index.key_columns,
            )?;
            if prior.supporting_index.oid == foreign_key.supporting_index.oid
                && !same_decoded_index(&prior.supporting_index, &foreign_key.supporting_index)
            {
                return Err(codec_error("catalog index OID identity drifted"));
            }
            if foreign_key.parent_dependency_ordinal != 0
                && prior.parent_dependency_ordinal == foreign_key.parent_dependency_ordinal
                && (prior.supporting_index.raw_ordinal == foreign_key.supporting_index.raw_ordinal
                    || prior.supporting_index.name == foreign_key.supporting_index.name)
                && !same_decoded_index(&prior.supporting_index, &foreign_key.supporting_index)
            {
                return Err(codec_error(
                    "external foreign-key supporting-index copy drifted",
                ));
            }
        }
        if foreign_key.parent_dependency_ordinal == 0 {
            let self_index = indexes
                .get(foreign_key.supporting_index.raw_ordinal as usize)
                .filter(|index| same_decoded_index(index, &foreign_key.supporting_index));
            if self_index.is_none() {
                return Err(codec_error(
                    "self-referencing foreign-key supporting index drifted",
                ));
            }
        }
    }
    Ok(())
}

fn validate_decoded_index(
    index: &DecodedIndex,
    dependencies: &[DecodedDependency],
) -> Result<(), EngineError> {
    let owner = dependencies
        .get(index.owner_dependency_ordinal as usize)
        .ok_or_else(|| codec_error("index owner dependency absent"))?;
    if !valid_decoded_oid(index.oid)
        || index.name.is_empty()
        || index.table_name != owner.name
        || index.first_column_name.is_empty()
        || !(1..=32).contains(&index.key_columns.len())
        || index.key_columns[0].name != index.first_column_name
        || ((index.primary_key || index.unique_constraint) && !index.unique)
        || (index.primary_key && index.unique_constraint)
    {
        return Err(codec_error("index metadata is invalid"));
    }
    for (ordinal, column) in index.key_columns.iter().enumerate() {
        if column.dependency_ordinal != index.owner_dependency_ordinal
            || index.key_columns[..ordinal]
                .iter()
                .any(|prior| prior.column_id == column.column_id)
        {
            return Err(codec_error("index repeats a key-column identity"));
        }
        validate_decoded_catalog_column(column, dependencies)?;
    }
    Ok(())
}

fn validate_decoded_catalog_column(
    column: &DecodedCatalogColumn,
    dependencies: &[DecodedDependency],
) -> Result<(), EngineError> {
    if column.column_id == 0
        || column.name.is_empty()
        || column.type_oid == 0
        || column.type_size != column.ty.type_size()
        || dependencies
            .get(column.dependency_ordinal as usize)
            .is_none()
    {
        return Err(codec_error("resolved catalog column identity is invalid"));
    }
    Ok(())
}

fn validate_decoded_catalog_column_sets(
    left: &[DecodedCatalogColumn],
    right: &[DecodedCatalogColumn],
) -> Result<(), EngineError> {
    for column in left {
        for candidate in right {
            validate_decoded_catalog_column_pair(column, candidate)?;
        }
    }
    Ok(())
}

fn validate_decoded_catalog_column_pair(
    left: &DecodedCatalogColumn,
    right: &DecodedCatalogColumn,
) -> Result<(), EngineError> {
    let conflicts = left.column_id == right.column_id
        || (left.dependency_ordinal == right.dependency_ordinal
            && (left.catalog_column_ordinal == right.catalog_column_ordinal
                || left.attnum == right.attnum
                || left.name == right.name));
    if conflicts && !same_decoded_catalog_column(left, right) {
        return Err(codec_error("catalog column identity drifted"));
    }
    Ok(())
}

fn same_decoded_catalog_column(left: &DecodedCatalogColumn, right: &DecodedCatalogColumn) -> bool {
    left.dependency_ordinal == right.dependency_ordinal
        && left.catalog_column_ordinal == right.catalog_column_ordinal
        && left.column_id == right.column_id
        && left.attnum == right.attnum
        && left.name == right.name
        && left.ty == right.ty
        && left.type_oid == right.type_oid
        && left.type_size == right.type_size
}

fn same_decoded_index(left: &DecodedIndex, right: &DecodedIndex) -> bool {
    left.owner_dependency_ordinal == right.owner_dependency_ordinal
        && left.raw_ordinal == right.raw_ordinal
        && left.oid == right.oid
        && left.name == right.name
        && left.table_name == right.table_name
        && left.first_column_name == right.first_column_name
        && left.unique == right.unique
        && left.primary_key == right.primary_key
        && left.unique_constraint == right.unique_constraint
        && left.key_columns.len() == right.key_columns.len()
        && left
            .key_columns
            .iter()
            .zip(&right.key_columns)
            .all(|(left, right)| same_decoded_catalog_column(left, right))
}

fn validate_same_primary_index(
    left: &DecodedIndex,
    right: &DecodedIndex,
) -> Result<(), EngineError> {
    if left.primary_key
        && right.primary_key
        && left.owner_dependency_ordinal == right.owner_dependency_ordinal
        && !same_decoded_index(left, right)
    {
        return Err(codec_error("relation has conflicting primary-key indexes"));
    }
    Ok(())
}

/// Allocation-free equivalent of the producer's global catalog namespace registry. The producer
/// owns BTreeMap-backed identities; this inert recovery boundary must not retain attacker keys,
/// so it rescans every completed registry dimension and compares each pair exactly.
#[allow(clippy::too_many_arguments)]
fn validate_decoded_global_identity_registry(
    target: &Target,
    columns: &[DecodedColumn],
    dependencies: &[DecodedDependency],
    domains: &[DecodedDomain],
    indexes: &[DecodedIndex],
    foreign_keys: &[DecodedForeignKey],
    effects: &DecodedEffects,
) -> Result<(), EngineError> {
    if !valid_decoded_oid(target.oid)
        || target.schema.is_empty()
        || target.name.is_empty()
        || zero_digest(target.schema_digest)
    {
        return Err(codec_error("relation identity is invalid"));
    }
    for dependency in dependencies {
        if !valid_decoded_oid(dependency.oid)
            || dependency.schema.is_empty()
            || dependency.name.is_empty()
            || zero_digest(dependency.schema_digest)
        {
            return Err(codec_error("relation identity is invalid"));
        }
    }
    for domain in domains {
        if !valid_decoded_oid(domain.oid) || domain.schema.is_empty() || domain.name.is_empty() {
            return Err(codec_error("domain identity is invalid"));
        }
    }
    let mut prior_attnum = None;
    for (ordinal, column) in columns.iter().enumerate() {
        if column.name.is_empty()
            || column.column_id == 0
            || column.attnum <= 0
            || u32::try_from(ordinal)
                .ok()
                .is_none_or(|ordinal| ordinal >= column.attnum as u32)
            || column.type_oid == 0
            || column.type_size != column.ty.type_size()
            || prior_attnum.is_some_and(|prior| prior >= column.attnum)
        {
            return Err(codec_error("target catalog column identity is invalid"));
        }
        prior_attnum = Some(column.attnum);
    }
    for effect in &effects.effects {
        if !valid_decoded_oid(effect.request.sequence_oid)
            || effect.request.effective_name.is_empty()
        {
            return Err(codec_error("sequence descriptor identity is invalid"));
        }
    }
    validate_decoded_domain_names(domains)?;
    validate_decoded_global_object_registry(
        target,
        columns,
        dependencies,
        domains,
        indexes,
        foreign_keys,
        effects,
    )?;
    validate_decoded_global_column_registry(target, columns, dependencies, indexes, foreign_keys)?;
    validate_decoded_sequence_effective_names(effects)?;
    validate_decoded_class_names(target, dependencies, indexes, foreign_keys, effects)
}

#[derive(Clone, Copy)]
enum DecodedGlobalObject<'a> {
    Relation {
        oid: u32,
        schema: &'a str,
        name: &'a str,
        schema_digest: gpu_db_wal::CanonicalDigest,
    },
    Domain {
        oid: u32,
        base_type: SqlType,
        named: Option<(&'a str, &'a str)>,
    },
    Index(&'a DecodedIndex),
    Sequence {
        oid: u32,
        effective_name: &'a str,
    },
}

struct DecodedGlobalObjectSources<'a> {
    target: &'a Target,
    columns: &'a [DecodedColumn],
    dependencies: &'a [DecodedDependency],
    domains: &'a [DecodedDomain],
    indexes: &'a [DecodedIndex],
    foreign_keys: &'a [DecodedForeignKey],
    effects: &'a DecodedEffects,
}

impl DecodedGlobalObject<'_> {
    fn oid(self) -> u32 {
        match self {
            Self::Relation { oid, .. } | Self::Domain { oid, .. } | Self::Sequence { oid, .. } => {
                oid
            }
            Self::Index(index) => index.oid,
        }
    }
}

fn validate_decoded_global_object_registry(
    target: &Target,
    columns: &[DecodedColumn],
    dependencies: &[DecodedDependency],
    domains: &[DecodedDomain],
    indexes: &[DecodedIndex],
    foreign_keys: &[DecodedForeignKey],
    effects: &DecodedEffects,
) -> Result<(), EngineError> {
    let sources = DecodedGlobalObjectSources {
        target,
        columns,
        dependencies,
        domains,
        indexes,
        foreign_keys,
        effects,
    };
    let mut ordinal = 0_usize;
    visit_decoded_global_objects(&sources, |candidate| {
        let candidate_ordinal = ordinal;
        let mut prior_ordinal = 0_usize;
        visit_decoded_global_objects(&sources, |prior| {
            if prior_ordinal < candidate_ordinal
                && prior.oid() == candidate.oid()
                && !same_decoded_global_object(prior, candidate)
            {
                return Err(codec_error("global catalog OID identity drifted"));
            }
            prior_ordinal += 1;
            Ok(())
        })?;
        ordinal += 1;
        Ok(())
    })
}

fn visit_decoded_global_objects(
    sources: &DecodedGlobalObjectSources<'_>,
    mut visit: impl FnMut(DecodedGlobalObject<'_>) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    visit(DecodedGlobalObject::Relation {
        oid: sources.target.oid,
        schema: &sources.target.schema,
        name: &sources.target.name,
        schema_digest: sources.target.schema_digest,
    })?;
    for dependency in sources.dependencies {
        visit(DecodedGlobalObject::Relation {
            oid: dependency.oid,
            schema: &dependency.schema,
            name: &dependency.name,
            schema_digest: dependency.schema_digest,
        })?;
    }
    for domain in sources.domains {
        visit(DecodedGlobalObject::Domain {
            oid: domain.oid,
            base_type: domain.base_type,
            named: Some((&domain.schema, &domain.name)),
        })?;
    }
    for column in sources.columns {
        visit_decoded_implicit_domain(column.ty, column.type_oid, &mut visit)?;
    }
    for index in sources.indexes {
        for column in &index.key_columns {
            visit_decoded_implicit_domain(column.ty, column.type_oid, &mut visit)?;
        }
        visit(DecodedGlobalObject::Index(index))?;
    }
    for foreign_key in sources.foreign_keys {
        visit_decoded_implicit_domain(
            foreign_key.child_column.ty,
            foreign_key.child_column.type_oid,
            &mut visit,
        )?;
        visit_decoded_implicit_domain(
            foreign_key.parent_column.ty,
            foreign_key.parent_column.type_oid,
            &mut visit,
        )?;
        for column in &foreign_key.supporting_index.key_columns {
            visit_decoded_implicit_domain(column.ty, column.type_oid, &mut visit)?;
        }
        visit(DecodedGlobalObject::Index(&foreign_key.supporting_index))?;
    }
    for effect in &sources.effects.effects {
        visit(DecodedGlobalObject::Sequence {
            oid: effect.request.sequence_oid,
            effective_name: &effect.request.effective_name,
        })?;
    }
    Ok(())
}

fn validate_decoded_domain_names(domains: &[DecodedDomain]) -> Result<(), EngineError> {
    for (ordinal, domain) in domains.iter().enumerate() {
        if domains[..ordinal].iter().any(|prior| {
            prior.schema == domain.schema && prior.name == domain.name && prior.oid != domain.oid
        }) {
            return Err(codec_error("domain qualified-name identity drifted"));
        }
    }
    Ok(())
}

fn visit_decoded_implicit_domain(
    ty: SqlType,
    type_oid: u32,
    visit: &mut impl FnMut(DecodedGlobalObject<'_>) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    if type_oid != ty.postgres_oid()
        && !gpu_db_sql::SUPPORTED_SQL_TYPES
            .iter()
            .any(|builtin| type_oid == builtin.postgres_oid())
    {
        visit(DecodedGlobalObject::Domain {
            oid: type_oid,
            base_type: ty,
            named: None,
        })?;
    }
    Ok(())
}

fn same_decoded_global_object(
    left: DecodedGlobalObject<'_>,
    right: DecodedGlobalObject<'_>,
) -> bool {
    match (left, right) {
        (
            DecodedGlobalObject::Relation {
                schema: left_schema,
                name: left_name,
                schema_digest: left_digest,
                ..
            },
            DecodedGlobalObject::Relation {
                schema: right_schema,
                name: right_name,
                schema_digest: right_digest,
                ..
            },
        ) => left_schema == right_schema && left_name == right_name && left_digest == right_digest,
        (
            DecodedGlobalObject::Domain {
                base_type: left_type,
                named: left_name,
                ..
            },
            DecodedGlobalObject::Domain {
                base_type: right_type,
                named: right_name,
                ..
            },
        ) => {
            left_type == right_type
                && match (left_name, right_name) {
                    (Some(left), Some(right)) => left == right,
                    _ => true,
                }
        }
        (DecodedGlobalObject::Index(left), DecodedGlobalObject::Index(right)) => {
            same_decoded_index(left, right)
        }
        (
            DecodedGlobalObject::Sequence {
                effective_name: left_name,
                ..
            },
            DecodedGlobalObject::Sequence {
                effective_name: right_name,
                ..
            },
        ) => left_name == right_name,
        _ => false,
    }
}

#[derive(Clone, Copy)]
struct DecodedGlobalColumnIdentity<'a> {
    relation_oid: u32,
    catalog_column_ordinal: u32,
    column_id: u32,
    attnum: i16,
    name: &'a str,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
}

#[allow(clippy::too_many_arguments)]
fn validate_decoded_global_column_registry(
    target: &Target,
    columns: &[DecodedColumn],
    dependencies: &[DecodedDependency],
    indexes: &[DecodedIndex],
    foreign_keys: &[DecodedForeignKey],
) -> Result<(), EngineError> {
    for (ordinal, column) in columns.iter().enumerate() {
        validate_decoded_column_type(column.ty, column.type_oid)?;
        let catalog_column_ordinal =
            u32::try_from(ordinal).map_err(|_| codec_error("target column ordinal overflows"))?;
        let candidate = decoded_target_column_identity(target, catalog_column_ordinal, column);
        for (prior_ordinal, prior) in columns[..ordinal].iter().enumerate() {
            let prior_ordinal = u32::try_from(prior_ordinal)
                .map_err(|_| codec_error("target column ordinal overflows"))?;
            validate_decoded_global_column_pair(
                decoded_target_column_identity(target, prior_ordinal, prior),
                candidate,
            )?;
        }
        visit_decoded_catalog_columns(indexes, foreign_keys, |catalog_column| {
            validate_decoded_global_column_pair(
                candidate,
                decoded_catalog_column_identity(catalog_column, dependencies, target, columns)?,
            )
        })?;
    }
    let mut ordinal = 0_usize;
    visit_decoded_catalog_columns(indexes, foreign_keys, |catalog_column| {
        let candidate =
            decoded_catalog_column_identity(catalog_column, dependencies, target, columns)?;
        let candidate_ordinal = ordinal;
        let mut prior_ordinal = 0_usize;
        visit_decoded_catalog_columns(indexes, foreign_keys, |prior| {
            if prior_ordinal < candidate_ordinal {
                validate_decoded_global_column_pair(
                    decoded_catalog_column_identity(prior, dependencies, target, columns)?,
                    candidate,
                )?;
            }
            prior_ordinal += 1;
            Ok(())
        })?;
        ordinal += 1;
        Ok(())
    })
}

fn visit_decoded_catalog_columns(
    indexes: &[DecodedIndex],
    foreign_keys: &[DecodedForeignKey],
    mut visit: impl FnMut(&DecodedCatalogColumn) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    for index in indexes {
        for column in &index.key_columns {
            visit(column)?;
        }
    }
    for foreign_key in foreign_keys {
        visit(&foreign_key.child_column)?;
        visit(&foreign_key.parent_column)?;
        for column in &foreign_key.supporting_index.key_columns {
            visit(column)?;
        }
    }
    Ok(())
}

fn decoded_target_column_identity<'a>(
    target: &'a Target,
    ordinal: u32,
    column: &'a DecodedColumn,
) -> DecodedGlobalColumnIdentity<'a> {
    DecodedGlobalColumnIdentity {
        relation_oid: target.oid,
        catalog_column_ordinal: ordinal,
        column_id: column.column_id,
        attnum: column.attnum,
        name: &column.name,
        ty: column.ty,
        type_oid: column.type_oid,
        type_size: column.type_size,
    }
}

fn decoded_catalog_column_identity<'a>(
    column: &'a DecodedCatalogColumn,
    dependencies: &'a [DecodedDependency],
    target: &'a Target,
    columns: &'a [DecodedColumn],
) -> Result<DecodedGlobalColumnIdentity<'a>, EngineError> {
    let relation_oid = dependencies
        .get(column.dependency_ordinal as usize)
        .map(|dependency| dependency.oid)
        .ok_or_else(|| codec_error("catalog column dependency is absent"))?;
    validate_decoded_column_type(column.ty, column.type_oid)?;
    if column.column_id == 0
        || column.attnum <= 0
        || column.catalog_column_ordinal >= column.attnum as u32
        || column.name.is_empty()
        || column.type_size != column.ty.type_size()
    {
        return Err(codec_error("catalog column identity is invalid"));
    }
    let candidate = DecodedGlobalColumnIdentity {
        relation_oid,
        catalog_column_ordinal: column.catalog_column_ordinal,
        column_id: column.column_id,
        attnum: column.attnum,
        name: &column.name,
        ty: column.ty,
        type_oid: column.type_oid,
        type_size: column.type_size,
    };
    if relation_oid == target.oid {
        let target_column = columns
            .get(column.catalog_column_ordinal as usize)
            .ok_or_else(|| codec_error("target catalog column identity is invalid"))?;
        let expected =
            decoded_target_column_identity(target, column.catalog_column_ordinal, target_column);
        if !same_decoded_global_column_identity(candidate, expected) {
            return Err(codec_error("global catalog column identity drifted"));
        }
    }
    Ok(candidate)
}

fn validate_decoded_global_column_pair(
    left: DecodedGlobalColumnIdentity<'_>,
    right: DecodedGlobalColumnIdentity<'_>,
) -> Result<(), EngineError> {
    let conflicts = left.column_id == right.column_id
        || (left.relation_oid == right.relation_oid
            && (left.catalog_column_ordinal == right.catalog_column_ordinal
                || left.attnum == right.attnum
                || left.name == right.name));
    if conflicts && !same_decoded_global_column_identity(left, right) {
        return Err(codec_error("global catalog column identity drifted"));
    }
    Ok(())
}

fn same_decoded_global_column_identity(
    left: DecodedGlobalColumnIdentity<'_>,
    right: DecodedGlobalColumnIdentity<'_>,
) -> bool {
    left.relation_oid == right.relation_oid
        && left.catalog_column_ordinal == right.catalog_column_ordinal
        && left.column_id == right.column_id
        && left.attnum == right.attnum
        && left.name == right.name
        && left.ty == right.ty
        && left.type_oid == right.type_oid
        && left.type_size == right.type_size
}

fn validate_decoded_column_type(ty: SqlType, type_oid: u32) -> Result<(), EngineError> {
    if type_oid == ty.postgres_oid() {
        return Ok(());
    }
    if gpu_db_sql::SUPPORTED_SQL_TYPES
        .iter()
        .any(|builtin| type_oid == builtin.postgres_oid())
    {
        return Err(codec_error("builtin type OID has the wrong SQL type"));
    }
    if !valid_decoded_oid(type_oid) {
        return Err(codec_error("domain type OID is invalid"));
    }
    Ok(())
}

fn validate_decoded_sequence_effective_names(effects: &DecodedEffects) -> Result<(), EngineError> {
    for (ordinal, effect) in effects.effects.iter().enumerate() {
        for prior in &effects.effects[..ordinal] {
            if prior.request.effective_name == effect.request.effective_name
                && prior.request.sequence_oid != effect.request.sequence_oid
            {
                return Err(codec_error("sequence effective-name identity drifted"));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct DecodedClassIdentity<'a> {
    schema: &'a str,
    name: &'a str,
    oid: u32,
}

fn validate_decoded_class_names(
    target: &Target,
    dependencies: &[DecodedDependency],
    indexes: &[DecodedIndex],
    foreign_keys: &[DecodedForeignKey],
    effects: &DecodedEffects,
) -> Result<(), EngineError> {
    let mut ordinal = 0_usize;
    visit_decoded_class_identities(
        target,
        dependencies,
        indexes,
        foreign_keys,
        effects,
        |candidate| {
            let candidate_ordinal = ordinal;
            let mut prior_ordinal = 0_usize;
            visit_decoded_class_identities(
                target,
                dependencies,
                indexes,
                foreign_keys,
                effects,
                |prior| {
                    if prior_ordinal < candidate_ordinal
                        && prior.schema == candidate.schema
                        && prior.name == candidate.name
                        && prior.oid != candidate.oid
                    {
                        return Err(codec_error("qualified class-name identity drifted"));
                    }
                    prior_ordinal += 1;
                    Ok(())
                },
            )?;
            ordinal += 1;
            Ok(())
        },
    )
}

fn visit_decoded_class_identities(
    target: &Target,
    dependencies: &[DecodedDependency],
    indexes: &[DecodedIndex],
    foreign_keys: &[DecodedForeignKey],
    effects: &DecodedEffects,
    mut visit: impl FnMut(DecodedClassIdentity<'_>) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    visit(DecodedClassIdentity {
        schema: &target.schema,
        name: &target.name,
        oid: target.oid,
    })?;
    for dependency in dependencies {
        visit(DecodedClassIdentity {
            schema: &dependency.schema,
            name: &dependency.name,
            oid: dependency.oid,
        })?;
    }
    for index in indexes {
        visit_decoded_index_class_identity(index, dependencies, &mut visit)?;
    }
    for foreign_key in foreign_keys {
        visit_decoded_index_class_identity(
            &foreign_key.supporting_index,
            dependencies,
            &mut visit,
        )?;
    }
    for effect in &effects.effects {
        visit(DecodedClassIdentity {
            schema: "public",
            name: &effect.request.effective_name,
            oid: effect.request.sequence_oid,
        })?;
    }
    Ok(())
}

fn visit_decoded_index_class_identity(
    index: &DecodedIndex,
    dependencies: &[DecodedDependency],
    visit: &mut impl FnMut(DecodedClassIdentity<'_>) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    let owner = dependencies
        .get(index.owner_dependency_ordinal as usize)
        .ok_or_else(|| codec_error("index owner dependency absent"))?;
    visit(DecodedClassIdentity {
        schema: &owner.schema,
        name: &index.name,
        oid: index.oid,
    })
}

fn valid_decoded_oid(oid: u32) -> bool {
    (1..=i32::MAX as u32).contains(&oid)
}

fn validate_target_catalog_references(
    columns: &[DecodedColumn],
    indexes: &[DecodedIndex],
    foreign_keys: &[DecodedForeignKey],
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
                || key.name != column.name
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
            || foreign_key.child_column.name != column.name
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

fn validate_dependency_closure_order(
    dependencies: &[DecodedDependency],
    foreign_keys: &[DecodedForeignKey],
) -> Result<(), EngineError> {
    let mut next = 1_u32;
    for foreign_key in foreign_keys {
        let dependency = dependencies
            .get(foreign_key.parent_dependency_ordinal as usize)
            .ok_or_else(|| codec_error("foreign-key parent dependency is absent"))?;
        if dependency.name != foreign_key.referenced_table_name {
            return Err(codec_error("foreign-key parent dependency name drifted"));
        }
        if foreign_key.parent_dependency_ordinal == 0 {
            continue;
        }
        let prior = foreign_keys
            .iter()
            .take_while(|prior| !std::ptr::eq(*prior, foreign_key))
            .find(|prior| {
                dependencies
                    .get(prior.parent_dependency_ordinal as usize)
                    .is_some_and(|candidate| candidate.oid == dependency.oid)
            });
        if let Some(prior) = prior {
            if prior.parent_dependency_ordinal != foreign_key.parent_dependency_ordinal {
                return Err(codec_error(
                    "foreign-key parent has ambiguous dependency ordinal",
                ));
            }
        } else if foreign_key.parent_dependency_ordinal == next {
            next = next
                .checked_add(1)
                .ok_or_else(|| codec_error("dependency ordinal overflows"))?;
        } else {
            return Err(codec_error(
                "FK parent dependency is not first-occurrence ordered",
            ));
        }
    }
    if dependencies.len() != next as usize {
        return Err(codec_error("dependency closure has an unused extra table"));
    }
    Ok(())
}

fn validate_decoded_columns(
    columns: &[DecodedColumn],
    domains: &[DecodedDomain],
    rows: u32,
) -> Result<(), EngineError> {
    let mut next_domain_ordinal = 0_u32;
    for (expected, column) in columns.iter().enumerate() {
        if column.ordinal != expected as u32
            || column.name.is_empty()
            || column.column_id == 0
            || column.type_oid == 0
            || column.type_size != column.ty.type_size()
            || columns
                .iter()
                .take(expected)
                .any(|prior| prior.column_id == column.column_id)
        {
            return Err(codec_error("decoded column identity is invalid"));
        }
        if let Some(source_ordinal) = column.source_ordinal {
            if columns
                .iter()
                .take(expected)
                .any(|prior| prior.source_ordinal == Some(source_ordinal))
            {
                return Err(codec_error("decoded source ordinal is duplicated"));
            }
        }
        if let Some(ordinal) = column.domain_ordinal {
            let domain = domains
                .get(ordinal as usize)
                .ok_or_else(|| codec_error("decoded domain ordinal absent"))?;
            if column.type_oid != domain.oid || column.ty != domain.base_type {
                return Err(codec_error("decoded domain binding drifted"));
            }
            if !columns[..expected]
                .iter()
                .any(|prior| prior.domain_ordinal == Some(ordinal))
            {
                if ordinal != next_domain_ordinal {
                    return Err(codec_error(
                        "decoded domains are not first-occurrence ordered",
                    ));
                }
                next_domain_ordinal = next_domain_ordinal
                    .checked_add(1)
                    .ok_or_else(|| codec_error("decoded domain ordinal overflows"))?;
            }
        } else if column.type_oid != column.ty.postgres_oid() {
            return Err(codec_error("decoded base type OID drifted"));
        }
        validate_decoded_column_vectors(column, rows)?;
    }
    let source_count = columns
        .iter()
        .filter(|column| column.source_ordinal.is_some())
        .count();
    if (0..source_count).any(|expected| {
        columns
            .iter()
            .filter(|column| column.source_ordinal == u32::try_from(expected).ok())
            .count()
            != 1
    }) {
        return Err(codec_error("decoded source ordinals are not contiguous"));
    }
    if usize::try_from(next_domain_ordinal).ok() != Some(domains.len()) {
        return Err(codec_error(
            "decoded domains are not first-occurrence ordered",
        ));
    }
    for (ordinal, domain) in domains.iter().enumerate() {
        if domain.schema.is_empty()
            || domain.name.is_empty()
            || domain.oid == 0
            || domains
                .iter()
                .take(ordinal)
                .any(|prior| prior.oid == domain.oid)
        {
            return Err(codec_error("decoded domain identity is invalid"));
        }
    }
    Ok(())
}

fn validate_decoded_column_vectors(column: &DecodedColumn, rows: u32) -> Result<(), EngineError> {
    let rows_usize = rows as usize;
    if !column.validity.shape_is_exact(rows_usize)
        || !column.presence.shape_is_exact(rows_usize)
        || !column.defaults.shape_is_exact(rows_usize)
        || column.states.len() != rows_usize
        || column.provenance.len() != rows_usize
        || !values_shape(&column.values, column.ty, rows_usize)
    {
        return Err(codec_error("decoded column vector shape drifted"));
    }
    for row in 0..rows_usize {
        let valid = column.validity.is_valid(row);
        let provided = column.presence.is_provided(row);
        let defaulted = column.defaults.was_defaulted(row);
        let ok = match (column.states[row], column.provenance[row]) {
            (
                TypedInsertInputState::Provided,
                TypedInsertInputProvenance::Literal
                | TypedInsertInputProvenance::BoundParameter { .. }
                | TypedInsertInputProvenance::ProgrammaticValue,
            ) => column.source_ordinal.is_some() && valid && provided && !defaulted,
            (
                TypedInsertInputState::ProvidedNull,
                TypedInsertInputProvenance::Literal
                | TypedInsertInputProvenance::BoundParameter { .. }
                | TypedInsertInputProvenance::ProgrammaticValue,
            ) => column.source_ordinal.is_some() && !valid && provided && !defaulted,
            (TypedInsertInputState::Omitted, TypedInsertInputProvenance::Omitted) => {
                column.source_ordinal.is_none() && provided && defaulted
            }
            (
                TypedInsertInputState::ExplicitDefault,
                TypedInsertInputProvenance::SqlDefault
                | TypedInsertInputProvenance::ProgrammaticDefault,
            ) => column.source_ordinal.is_some() && provided && defaulted,
            _ => false,
        };
        if !ok || (!valid && !decoded_placeholder_zero(&column.values, row)) {
            return Err(codec_error(
                "decoded input state/provenance or placeholder drifted",
            ));
        }
    }
    Ok(())
}

fn values_shape(values: &TypedInsertColumnValues, ty: SqlType, rows: usize) -> bool {
    super::super::typed_image_codec::typed_values_shape_is_valid(values, ty, rows)
}

fn decoded_placeholder_zero(values: &TypedInsertColumnValues, row: usize) -> bool {
    match values {
        TypedInsertColumnValues::I32(values) => values.get(row) == Some(&0),
        TypedInsertColumnValues::I64(values) => values.get(row) == Some(&0),
        TypedInsertColumnValues::I128(values) => values.get(row) == Some(&0),
        TypedInsertColumnValues::Bytes16(values) => values.get(row) == Some(&[0; 16]),
        TypedInsertColumnValues::BoolBits(words) => !bit_is_set(words, row),
        TypedInsertColumnValues::Text { offsets, .. } => offsets.get(row) == offsets.get(row + 1),
    }
}

fn decoded_returning_digest(
    returning: &DecodedReturning,
) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
    let mut counting = Writer::counting(MAX_RECORD_BYTES);
    append_decoded_returning_digest_body(&mut counting, returning)?;
    let mut body = Writer::hashing_request_body(MAX_RECORD_BYTES, counting.len())?;
    append_decoded_returning_digest_body(&mut body, returning)?;
    body.finish_request_digest()
}

fn append_decoded_returning_digest_body(
    body: &mut Writer,
    returning: &DecodedReturning,
) -> Result<(), EngineError> {
    body.bytes(b"GPUDBTYPEDINSERTRETURNING1")?;
    body.u16(FORMAT_VERSION)?;
    body.u16(SEMANTICS_VERSION)?;
    body.u32(returning.rows)?;
    body.u32(returning.columns)?;
    body.u64(returning.cells)?;
    body.u32(checked_u32(
        returning.projections.len(),
        "RETURNING projection count",
    )?)?;
    for projection in &returning.projections {
        append_projection(
            body,
            projection.catalog_column_ordinal,
            projection.column_id,
            projection.attnum,
            &projection.name,
            projection.ty,
            projection.type_oid,
            projection.type_size,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decoded_statement_digest(
    target: &Target,
    columns: &[DecodedColumn],
    dependencies: &[DecodedDependency],
    domains: &[DecodedDomain],
    indexes: &[DecodedIndex],
    foreign_keys: &[DecodedForeignKey],
    returning_digest: gpu_db_wal::CanonicalDigest,
    effects: &DecodedEffects,
) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
    let mut counting = Writer::counting(MAX_RECORD_BYTES);
    append_decoded_statement_digest_body(
        &mut counting,
        target,
        columns,
        dependencies,
        domains,
        indexes,
        foreign_keys,
        returning_digest,
        effects,
    )?;
    let mut body = Writer::hashing_request_body(MAX_RECORD_BYTES, counting.len())?;
    append_decoded_statement_digest_body(
        &mut body,
        target,
        columns,
        dependencies,
        domains,
        indexes,
        foreign_keys,
        returning_digest,
        effects,
    )?;
    body.finish_request_digest()
}

#[allow(clippy::too_many_arguments)]
fn append_decoded_statement_digest_body(
    body: &mut Writer,
    target: &Target,
    columns: &[DecodedColumn],
    dependencies: &[DecodedDependency],
    domains: &[DecodedDomain],
    indexes: &[DecodedIndex],
    foreign_keys: &[DecodedForeignKey],
    returning_digest: gpu_db_wal::CanonicalDigest,
    effects: &DecodedEffects,
) -> Result<(), EngineError> {
    body.bytes(b"GPUDBTYPEDINSERTSTATEMENT1")?;
    body.u16(FORMAT_VERSION)?;
    body.u16(SEMANTICS_VERSION)?;
    body.identifier(&target.schema)?;
    body.identifier(&target.name)?;
    body.u32(target.oid)?;
    body.digest(&target.schema_digest)?;
    body.u32(target.statement_ordinal.as_u32())?;
    body.u32(target.rows)?;
    reencode::append_decoded_intent_columns(body, columns, target.rows, &effects.effects)?;
    reencode::append_decoded_dependencies(body, dependencies)?;
    reencode::append_decoded_domains(body, domains)?;
    reencode::append_decoded_indexes(body, indexes)?;
    reencode::append_decoded_foreign_keys(body, foreign_keys)?;
    body.digest(&returning_digest)?;
    body.u32(checked_u32(
        effects.effects.len(),
        "decoded sequence effect count",
    )?)?;
    for effect in &effects.effects {
        let request = &effect.request;
        append_sequence_request(
            body,
            request.target_table_oid,
            request.row_ordinal,
            request.catalog_column_ordinal,
            request.column_id,
            request.sequence_oid,
            &request.source_name,
            &request.effective_name,
            request.statement_ordinal,
            request.expression_ordinal,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }
    fn done(&self) -> bool {
        self.cursor == self.bytes.len()
    }
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.cursor)
    }
    fn take(&mut self, count: usize) -> Result<&'a [u8], EngineError> {
        let end = self
            .cursor
            .checked_add(count)
            .ok_or_else(|| codec_error("decode cursor overflow"))?;
        let bytes = self
            .bytes
            .get(self.cursor..end)
            .ok_or_else(|| codec_error("record is truncated"))?;
        self.cursor = end;
        Ok(bytes)
    }
    fn exact<const N: usize>(&mut self) -> Result<[u8; N], EngineError> {
        self.take(N)?
            .try_into()
            .map_err(|_| codec_error("fixed-width decode failed"))
    }
    fn u8(&mut self) -> Result<u8, EngineError> {
        Ok(self.exact::<1>()?[0])
    }
    fn bool(&mut self) -> Result<bool, EngineError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(codec_error("boolean encoding is noncanonical")),
        }
    }
    fn u16(&mut self) -> Result<u16, EngineError> {
        Ok(u16::from_le_bytes(self.exact()?))
    }
    fn i16(&mut self) -> Result<i16, EngineError> {
        Ok(i16::from_le_bytes(self.exact()?))
    }
    fn u32(&mut self) -> Result<u32, EngineError> {
        Ok(u32::from_le_bytes(self.exact()?))
    }
    fn u64(&mut self) -> Result<u64, EngineError> {
        Ok(u64::from_le_bytes(self.exact()?))
    }
    fn i64(&mut self) -> Result<i64, EngineError> {
        Ok(i64::from_le_bytes(self.exact()?))
    }
    fn digest(&mut self) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
        self.exact()
    }
    fn count_bounded(
        &mut self,
        kind: &str,
        minimum_item_bytes: usize,
    ) -> Result<usize, EngineError> {
        let count = usize::try_from(self.u32()?).map_err(|_| codec_error("count overflows"))?;
        if minimum_item_bytes == 0 || count > self.remaining() / minimum_item_bytes {
            Err(codec_error(&format!(
                "{kind} count exceeds its remaining-byte bound"
            )))
        } else {
            Ok(count)
        }
    }
    fn identifier(&mut self) -> Result<String, EngineError> {
        let count =
            usize::try_from(self.u32()?).map_err(|_| codec_error("blob length overflows"))?;
        if count > self.remaining() || count > MAX_RECORD_BYTES {
            return Err(codec_error("blob length is outside section bounds"));
        }
        let bytes = self.take(count)?;
        if bytes.is_empty() || bytes.len() > MAX_IDENTIFIER_BYTES || bytes.contains(&0) {
            return Err(codec_error("identifier encoding is noncanonical"));
        }
        let text =
            std::str::from_utf8(bytes).map_err(|_| codec_error("identifier is not UTF-8"))?;
        let mut owned = String::new();
        reserve_string(&mut owned, text.len(), "decoded identifier")?;
        owned.push_str(text);
        if owned.len() != owned.capacity() {
            return Err(codec_error(
                "decoded identifier capacity is not exact before retention",
            ));
        }
        Ok(owned)
    }
    fn option_u32(&mut self) -> Result<Option<u32>, EngineError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.u32()?)),
            _ => Err(codec_error("option tag is noncanonical")),
        }
    }
    fn sql_type(&mut self) -> Result<SqlType, EngineError> {
        let tag = self.u8()?;
        let precision = self.u8()?;
        let scale = self.u8()?;
        if self.u8()? != 0 {
            return Err(codec_error("SQL type reserved byte is nonzero"));
        }
        match tag {
            1 if precision == 0 && scale == 0 => Ok(SqlType::Int2),
            2 if precision == 0 && scale == 0 => Ok(SqlType::Int4),
            3 if precision == 0 && scale == 0 => Ok(SqlType::Int8),
            4 if (1..=38).contains(&precision) && scale <= precision => {
                Ok(SqlType::Numeric { precision, scale })
            }
            5 if precision == 0 && scale == 0 => Ok(SqlType::Bool),
            6 if precision == 0 && scale == 0 => Ok(SqlType::Text),
            7 if precision == 0 && scale == 0 => Ok(SqlType::Date),
            8 if precision == 0 && scale == 0 => Ok(SqlType::Timestamp),
            9 if precision == 0 && scale == 0 => Ok(SqlType::Uuid),
            _ => Err(codec_error("SQL type tag/typmod is noncanonical")),
        }
    }
    fn input_state(&mut self) -> Result<TypedInsertInputState, EngineError> {
        match self.u8()? {
            1 => Ok(TypedInsertInputState::Provided),
            2 => Ok(TypedInsertInputState::ProvidedNull),
            3 => Ok(TypedInsertInputState::Omitted),
            4 => Ok(TypedInsertInputState::ExplicitDefault),
            _ => Err(codec_error("input state tag is unknown")),
        }
    }
    fn input_provenance(&mut self) -> Result<TypedInsertInputProvenance, EngineError> {
        let tag = self.u8()?;
        let index = self.u32()?;
        match (tag, index) {
            (1, 0) => Ok(TypedInsertInputProvenance::Omitted),
            (2, 0) => Ok(TypedInsertInputProvenance::Literal),
            (3, index) if index != 0 => Ok(TypedInsertInputProvenance::BoundParameter { index }),
            (4, 0) => Ok(TypedInsertInputProvenance::ProgrammaticValue),
            (5, 0) => Ok(TypedInsertInputProvenance::SqlDefault),
            (6, 0) => Ok(TypedInsertInputProvenance::ProgrammaticDefault),
            _ => Err(codec_error("input provenance tag/index is noncanonical")),
        }
    }
    fn bitmap_words(&mut self, rows: u32) -> Result<Box<[u32]>, EngineError> {
        let count = self.count_bounded("bitmap word", 4)?;
        let expected = bitmap_words(rows as usize)?;
        if count != expected {
            return Err(codec_error("bitmap word count is not exact"));
        }
        let mut words = Vec::new();
        reserve_exact(&mut words, count, "decoded bitmap owner")?;
        for _ in 0..count {
            words.push(self.u32()?);
        }
        if !bitmap_shape_is_exact(&words, rows as usize) {
            return Err(codec_error("bitmap tail bits are noncanonical"));
        }
        into_exact_boxed_slice(words)
    }
    fn validity(&mut self, rows: u32) -> Result<TypedInsertColumnValidity, EngineError> {
        let (validity, consumed) = super::super::typed_image_codec::decode_typed_validity(
            self.bytes
                .get(self.cursor..)
                .ok_or_else(|| codec_error("validity cursor is outside record"))?,
            rows,
        )?;
        self.take(consumed)?;
        Ok(validity)
    }
    fn presence(&mut self, _rows: u32) -> Result<TypedInsertColumnPresence, EngineError> {
        if self.u8()? != 0 {
            return Err(codec_error(
                "sealed canonical record requires AllProvided presence",
            ));
        }
        Ok(TypedInsertColumnPresence::AllProvided)
    }
    fn defaults(&mut self, rows: u32) -> Result<TypedInsertDefaultResolution, EngineError> {
        match self.u8()? {
            0 => Ok(TypedInsertDefaultResolution::AllDirect),
            1 => {
                let words = self.bitmap_words(rows)?;
                if words.iter().all(|word| *word == 0) {
                    Err(codec_error("empty default bitmap must use AllDirect form"))
                } else {
                    Ok(TypedInsertDefaultResolution::Bitmap(words))
                }
            }
            _ => Err(codec_error("default-resolution form tag is unknown")),
        }
    }
    fn values(&mut self, ty: SqlType, rows: u32) -> Result<TypedInsertColumnValues, EngineError> {
        let (values, consumed) = super::super::typed_image_codec::decode_typed_values(
            self.bytes
                .get(self.cursor..)
                .ok_or_else(|| codec_error("values cursor is outside record"))?,
            ty,
            rows,
        )?;
        self.take(consumed)?;
        Ok(values)
    }
}
