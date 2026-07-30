//! Model-to-bytes canonical rebuilding for the strict decoder.

use super::super::*;
use super::*;

pub(super) fn reencode_decoded(model: &DecodedModel) -> Result<Vec<u8>, EngineError> {
    reencode_decoded_with(model, false)
}

#[cfg(test)]
pub(super) fn reencode_decoded_unchecked(model: &DecodedModel) -> Result<Vec<u8>, EngineError> {
    reencode_decoded_with(model, true)
}

fn reencode_decoded_with(
    model: &DecodedModel,
    unchecked_values: bool,
) -> Result<Vec<u8>, EngineError> {
    let mut record = Writer::new(MAX_RECORD_BYTES);
    record.bytes(&MAGIC)?;
    record.u16(FORMAT_VERSION)?;
    record.u16(SEMANTICS_VERSION)?;
    record.u16(FORMAT_VERSION)?;
    record.u16(FORMAT_VERSION)?;
    record.u32(0)?;
    record.u16(SECTION_COUNT)?;
    record.u16(0)?;
    record.u32(0)?;
    record.digest(&model.typed_statement_digest)?;
    record.digest(&model.returning_digest)?;
    append_section(&mut record, SECTION_TARGET, |out| {
        out.identifier(&model.target.schema)?;
        out.identifier(&model.target.name)?;
        out.u32(model.target.oid)?;
        out.digest(&model.target.schema_digest)?;
        out.u32(model.target.statement_ordinal.as_u32())?;
        out.u32(model.target.rows)?;
        out.u32(model.target.column_count)
    })?;
    append_section(&mut record, SECTION_COLUMNS, |out| {
        encode_decoded_columns(out, &model.columns, model.target.rows, unchecked_values)
    })?;
    append_section(&mut record, SECTION_DEPENDENCIES, |out| {
        append_dependencies(out, &model.dependencies)
    })?;
    append_section(&mut record, SECTION_DOMAINS, |out| {
        append_domains(out, &model.domains)
    })?;
    append_section(&mut record, SECTION_INDEXES, |out| {
        append_indexes(out, &model.indexes)
    })?;
    append_section(&mut record, SECTION_FOREIGN_KEYS, |out| {
        append_foreign_keys(out, &model.foreign_keys)
    })?;
    append_section(&mut record, SECTION_RETURNING, |out| {
        encode_decoded_returning(out, &model.returning)
    })?;
    append_section(&mut record, SECTION_SEQUENCE_EFFECTS, |out| {
        encode_decoded_effects(out, &model.effects)
    })?;
    let body_len = record
        .len()
        .checked_sub(HEADER_LEN)
        .ok_or_else(|| codec_error("decoded record header length underflow"))?;
    record.patch_u32(32, checked_u32(body_len, "decoded record body length")?);
    Ok(record.finish())
}

fn encode_decoded_columns(
    out: &mut Writer,
    columns: &[DecodedColumn],
    rows: u32,
    unchecked_values: bool,
) -> Result<(), EngineError> {
    out.u32(checked_u32(columns.len(), "decoded column count")?)?;
    for column in columns {
        out.u32(column.ordinal)?;
        out.identifier(&column.name)?;
        out.u32(column.column_id)?;
        out.i16(column.attnum)?;
        append_sql_type(out, column.ty)?;
        out.u32(column.type_oid)?;
        out.i16(column.type_size)?;
        out.option_u32(column.source_ordinal)?;
        out.option_u32(column.domain_ordinal)?;
        append_bitmap_form(out, &column.validity, rows, BitmapRole::Validity)?;
        append_presence_form(out, &column.presence, rows)?;
        append_default_form(out, &column.defaults, rows)?;
        out.u32(rows)?;
        for (state, provenance) in column.states.iter().zip(&column.provenance) {
            append_input_state(out, *state)?;
            append_input_provenance(out, *provenance)?;
        }
        if unchecked_values {
            append_values_unchecked(out, &column.values, column.ty, rows)?;
        } else {
            append_values(out, &column.values, column.ty, rows)?;
        }
    }
    Ok(())
}

fn append_values_unchecked(
    out: &mut Writer,
    values: &TypedInsertColumnValues,
    ty: SqlType,
    rows: u32,
) -> Result<(), EngineError> {
    match (values, ty) {
        (TypedInsertColumnValues::I32(values), SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
            let payload = checked_u32(
                values
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| codec_error("unchecked i32 payload overflows"))?,
                "unchecked i32 payload",
            )?;
            append_vector_header(out, 1, rows, payload)?;
            for value in values {
                out.i32(*value)?;
            }
            Ok(())
        }
        (TypedInsertColumnValues::I64(values), SqlType::Int8 | SqlType::Timestamp) => {
            let payload = checked_u32(
                values
                    .len()
                    .checked_mul(8)
                    .ok_or_else(|| codec_error("unchecked i64 payload overflows"))?,
                "unchecked i64 payload",
            )?;
            append_vector_header(out, 2, rows, payload)?;
            for value in values {
                out.i64(*value)?;
            }
            Ok(())
        }
        _ => append_values(out, values, ty, rows),
    }
}

fn encode_decoded_returning(
    out: &mut Writer,
    returning: &DecodedReturning,
) -> Result<(), EngineError> {
    out.u32(returning.rows)?;
    out.u32(returning.columns)?;
    out.u64(returning.cells)?;
    out.digest(&returning.digest)?;
    out.u32(checked_u32(
        returning.projections.len(),
        "decoded RETURNING count",
    )?)?;
    for projection in &returning.projections {
        append_projection(
            out,
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

fn encode_decoded_effects(out: &mut Writer, effects: &DecodedEffects) -> Result<(), EngineError> {
    out.bool(effects.parent.is_some())?;
    if let Some(parent) = effects.parent {
        append_sequence_parent(out, parent.view())?;
    }
    out.u32(checked_u32(
        effects.effects.len(),
        "decoded sequence effect count",
    )?)?;
    for effect in &effects.effects {
        out.u32(effect.request.ordinal)?;
        let request = &effect.request;
        append_sequence_request(
            out,
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
        out.u32(request.absolute_expression_ordinal)?;
        out.digest(&request.descriptor_digest)?;
        out.i64(request.value)?;
        match effect.kind {
            DecodedEffectKind::Published {
                transition_txn_id,
                input_digest,
                returned_value,
            } => {
                out.u8(1)?;
                out.u64(transition_txn_id)?;
                out.digest(&input_digest)?;
                out.i64(returned_value)?;
            }
            DecodedEffectKind::Private {
                prior_last_value,
                prior_is_called,
                next_last_value,
                next_is_called,
                lifetime_origin,
                owner,
                predecessor,
                input_digest,
            } => {
                out.u8(2)?;
                out.i64(prior_last_value)?;
                out.bool(prior_is_called)?;
                out.i64(next_last_value)?;
                out.bool(next_is_called)?;
                out.u8(lifetime_origin)?;
                append_private_owner(out, owner)?;
                append_private_predecessor(out, predecessor)?;
                out.digest(&input_digest)?;
            }
        }
    }
    Ok(())
}

pub(super) fn append_decoded_intent_columns(
    out: &mut Writer,
    columns: &[DecodedColumn],
    rows: u32,
    sequence_cells: &BTreeSet<(u32, u32)>,
) -> Result<(), EngineError> {
    out.u32(checked_u32(columns.len(), "decoded intent column count")?)?;
    for column in columns {
        out.u32(column.ordinal)?;
        out.identifier(&column.name)?;
        out.u32(column.column_id)?;
        out.i16(column.attnum)?;
        append_sql_type(out, column.ty)?;
        out.u32(column.type_oid)?;
        out.i16(column.type_size)?;
        out.option_u32(column.source_ordinal)?;
        out.option_u32(column.domain_ordinal)?;
        out.u32(rows)?;
        for row in 0..rows as usize {
            let state = column.states[row];
            append_input_state(out, state)?;
            append_input_provenance(out, column.provenance[row])?;
            let sequence = sequence_cells.contains(&(column.ordinal, row as u32));
            match state {
                TypedInsertInputState::Provided => {
                    out.u8(2)?;
                    append_value_at(out, &column.values, column.ty, row)?;
                }
                TypedInsertInputState::ProvidedNull => out.u8(1)?,
                TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault
                    if sequence =>
                {
                    out.u8(0)?
                }
                TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault
                    if column.validity.is_valid(row) =>
                {
                    out.u8(2)?;
                    append_value_at(out, &column.values, column.ty, row)?;
                }
                TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault => {
                    out.u8(1)?
                }
            }
        }
    }
    Ok(())
}
