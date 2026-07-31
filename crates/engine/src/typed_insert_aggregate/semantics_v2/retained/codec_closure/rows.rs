//! S4/table/transition/final-image closure.

use super::error;
use super::statements::{begin, range, storage};
use crate::typed_insert_aggregate::semantics_v2::retained::graph::{
    ReservedSemanticsV2Graph, RetainedTransition,
};
use crate::typed_insert_batch::{
    DecodedTypedImage, DecodedTypedValueFacts, TypedInsertColumnValidity, TypedInsertColumnValues,
};
use crate::{EngineError, SqlType};
use sha2::{Digest, Sha256};

const ABSENT_U32: u32 = u32::MAX;
const SURVIVES: u8 = 1;

pub(super) fn validate(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    if graph.tables.len() != graph.images.len() {
        return Err(error("table/image cardinalities diverge"));
    }
    let mut expected_disposition = 0_u32;
    let mut expected_transition = 0_u32;
    let mut previous_stable_table_id = None;
    for (table_ordinal, table) in graph.tables.iter().enumerate() {
        let table_ref = table_ordinal as u32;
        let dispositions = range(
            &graph.table_dispositions,
            table.disposition_start,
            table.disposition_count,
            "table disposition",
        )?;
        let transitions = range(
            &graph.transitions,
            table.transition_start,
            table.transition_count,
            "table transition",
        )?;
        let image = graph
            .images
            .get(table.image_ref as usize)
            .ok_or_else(|| error("table image is absent"))?;
        let facts = image.facts();
        if table.table_ref != table_ref
            || previous_stable_table_id.is_some_and(|previous| previous >= table.stable_table_id)
            || table.image_ref != table_ref
            || table.disposition_start != expected_disposition
            || table.transition_start != expected_transition
            || table
                .row_allocator_high_water
                .checked_sub(table.row_allocator_before)
                != Some(table.disposition_count as u64)
            || table.final_logical_row_count
                != table
                    .initial_logical_row_count
                    .checked_add(table.transition_count as u64)
                    .ok_or_else(|| error("final row count overflows"))?
            || (table.transition_count == 0
                && (table.data_generation_after != table.data_generation_before
                    || table.final_table_root != table.initial_table_root))
            || (table.transition_count != 0
                && table.data_generation_after <= table.data_generation_before)
            || facts.role != crate::typed_insert_batch::TypedImageRole::FinalTableImage
            || facts.rows != table.transition_count
            || facts.columns != table.catalog_column_count
            || facts.layout_digest != table.image_layout_digest
            || table.image_descriptor_digest
                != image_descriptor_digest(table, facts.rows, facts.columns)
        {
            return Err(error("table range/image/allocator closure is invalid"));
        }
        validate_final_image_schema_for_each_target_record(graph, table_ref, image)?;
        let mut previous_row = None;
        for (offset, reorder) in dispositions.iter().enumerate() {
            let disposition = graph
                .dispositions
                .get(reorder.disposition_ref as usize)
                .ok_or_else(|| error("table disposition S4 entry is absent"))?;
            if reorder.table_ref != table_ref
                || reorder.stable_row_id != disposition.stable_row_id
                || reorder.statement_ordinal != disposition.statement_ordinal
                || reorder.source_row_ordinal != disposition.source_row_ordinal
                || reorder.disposition != disposition.disposition
                || disposition.table_ref != table_ref
                || previous_row.is_some_and(|previous| previous >= reorder.stable_row_id)
                || reorder.stable_row_id != table.row_allocator_before + offset as u64
            {
                return Err(error("S4/table disposition bijection is invalid"));
            }
            previous_row = Some(reorder.stable_row_id);
        }
        for (disposition_ref, disposition) in graph.dispositions.iter().enumerate() {
            if disposition.table_ref != table_ref {
                continue;
            }
            let table_ordinal = disposition
                .stable_row_id
                .checked_sub(table.row_allocator_before)
                .filter(|ordinal| *ordinal < u64::from(table.disposition_count))
                .ok_or_else(|| error("S4 table row lies outside its exact allocator range"))?;
            let table_ordinal = usize::try_from(table_ordinal)
                .map_err(|_| error("S4 table disposition ordinal is not addressable"))?;
            let expected_ref = u32::try_from(disposition_ref)
                .map_err(|_| error("S4 disposition ref exceeds u32"))?;
            if dispositions
                .get(table_ordinal)
                .is_none_or(|reorder| reorder.disposition_ref != expected_ref)
            {
                return Err(error("S4/table directory does not biject every table row"));
            }
        }
        for (offset, transition) in transitions.iter().enumerate() {
            validate_transition(
                graph,
                table_ref,
                table.stable_table_id,
                image,
                transition,
                expected_transition + offset as u32,
            )?;
        }
        if transitions.len()
            != graph
                .dispositions
                .iter()
                .filter(|entry| entry.table_ref == table_ref && entry.disposition == SURVIVES)
                .count()
        {
            return Err(error("table transitions do not biject S4 survivors"));
        }
        expected_disposition = expected_disposition
            .checked_add(table.disposition_count)
            .ok_or_else(|| error("table disposition count overflows"))?;
        expected_transition = expected_transition
            .checked_add(table.transition_count)
            .ok_or_else(|| error("table transition count overflows"))?;
        previous_stable_table_id = Some(table.stable_table_id);
    }
    if expected_disposition as usize != graph.table_dispositions.len()
        || expected_disposition as usize != graph.dispositions.len()
        || expected_transition as usize != graph.transitions.len()
    {
        return Err(error(
            "table ranges do not exhaust S4/transition directories",
        ));
    }
    Ok(())
}

fn validate_final_image_schema_for_each_target_record(
    graph: &ReservedSemanticsV2Graph,
    table_ref: u32,
    image: &DecodedTypedImage,
) -> Result<(), EngineError> {
    let mut record_count = 0_u32;
    for resolution in graph
        .resolutions
        .iter()
        .filter(|resolution| resolution.table_ref == table_ref)
    {
        let record = graph
            .records
            .get(resolution.record_ref as usize)
            .ok_or_else(|| error("final-image target S2 record is absent"))?;
        let mut image_columns = image.columns();
        let mut source_columns = record.catalog_columns();
        loop {
            match (image_columns.next(), source_columns.next()) {
                (None, None) => break,
                (Some(column), Some(source))
                    if column.catalog_column_ordinal == source.catalog_column_ordinal
                        && column.stable_column_id == source.column_id
                        && column.attnum == source.attnum
                        && column.ty == source.ty
                        && column.type_oid == source.type_oid
                        && column.type_size == source.type_size
                        && column.table_ref == table_ref
                        && column.name.is_empty() => {}
                _ => {
                    return Err(error(
                        "final-image catalog descriptors do not exhaustively equal a target S2 record",
                    ));
                }
            }
        }
        record_count = record_count
            .checked_add(1)
            .ok_or_else(|| error("final-image target S2 record count overflows"))?;
    }
    if record_count == 0 {
        return Err(error("final image has no target S2 record"));
    }
    Ok(())
}

fn image_descriptor_digest(
    table: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedTable,
    rows: u32,
    columns: u32,
) -> [u8; 32] {
    let mut raw = [0_u8; 128];
    raw[..4].copy_from_slice(&table.image_ref.to_le_bytes());
    raw[4..8].copy_from_slice(&table.table_ref.to_le_bytes());
    raw[8..12].copy_from_slice(&1_u32.to_le_bytes());
    raw[16..20].copy_from_slice(&rows.to_le_bytes());
    raw[20..24].copy_from_slice(&columns.to_le_bytes());
    raw[24..32].copy_from_slice(&table.image_arena_offset.to_le_bytes());
    raw[32..40].copy_from_slice(&table.image_encoded_bytes.to_le_bytes());
    raw[64..96].copy_from_slice(&table.image_layout_digest);
    raw[96..128].copy_from_slice(&table.image_content_digest);
    let mut digest = begin(b"gpu-db/write001/s7-image-descriptor/v2");
    digest.update(raw);
    digest.update([0; 32]);
    digest.finalize().into()
}

fn validate_transition(
    graph: &ReservedSemanticsV2Graph,
    table_ref: u32,
    stable_table_id: u64,
    image: &DecodedTypedImage,
    transition: &RetainedTransition,
    expected_ref: u32,
) -> Result<(), EngineError> {
    let disposition = graph
        .dispositions
        .get(transition.source_disposition_ref as usize)
        .ok_or_else(|| error("transition S4 source is absent"))?;
    let record = graph
        .records
        .get(transition.source_statement_ordinal as usize)
        .ok_or_else(|| error("transition S2 source is absent"))?;
    if transition.transition_ref != expected_ref
        || transition.table_ref != table_ref
        || transition.image_ref != table_ref
        || transition.image_row_ordinal
            != expected_ref - graph.tables[table_ref as usize].transition_start
        || transition.final_writer_statement_ordinal != transition.source_statement_ordinal
        || disposition.disposition != SURVIVES
        || disposition.transition_ref != transition.transition_ref
        || disposition.table_ref != table_ref
        || disposition.stable_row_id != transition.stable_row_id
        || disposition.statement_ordinal != transition.source_statement_ordinal
        || disposition.source_row_ordinal != transition.source_row_ordinal
        || disposition.typed_statement_digest != transition.typed_statement_digest
        || transition.final_row_digest
            != final_row_digest(graph, stable_table_id, image, transition)?
    {
        return Err(error(
            "transition does not close its S4/image/final-row facts",
        ));
    }
    for (column, source) in image.columns().zip(record.catalog_columns()) {
        if column.catalog_column_ordinal != source.catalog_column_ordinal
            || column.stable_column_id != source.column_id
            || column.attnum != source.attnum
            || column.ty != source.ty
            || column.type_oid != source.type_oid
            || column.type_size != source.type_size
            || column.table_ref != transition.table_ref
            || !column.name.is_empty()
        {
            return Err(error(
                "final image column differs from strict S2 catalog order",
            ));
        }
        let (valid, value) =
            record.column_value_at(column.catalog_column_ordinal, transition.source_row_ordinal)?;
        if !record_image_cell_equal(
            valid,
            value,
            column.validity,
            column.values,
            column.ty,
            image.facts().rows,
            transition.image_row_ordinal,
        )? {
            return Err(error("strict S2 cell differs from final-image cell"));
        }
    }
    Ok(())
}

fn record_image_cell_equal(
    source_valid: bool,
    source: DecodedTypedValueFacts<'_>,
    validity: &TypedInsertColumnValidity,
    values: &TypedInsertColumnValues,
    ty: SqlType,
    rows: u32,
    row: u32,
) -> Result<bool, EngineError> {
    let row = usize::try_from(row).map_err(|_| error("image row overflows"))?;
    if row >= rows as usize {
        return Err(error("image row is absent"));
    }
    let image_valid = match validity {
        TypedInsertColumnValidity::AllValid => true,
        TypedInsertColumnValidity::Bitmap(words) => words
            .get(row / 32)
            .is_some_and(|word| word & (1 << (row % 32)) != 0),
    };
    if source_valid != image_valid {
        return Ok(false);
    }
    if !source_valid {
        return Ok(true);
    }
    Ok(match (source, values, ty) {
        (
            DecodedTypedValueFacts::I32(source),
            TypedInsertColumnValues::I32(values),
            SqlType::Int2 | SqlType::Int4 | SqlType::Date,
        ) => values.get(row) == Some(&source),
        (
            DecodedTypedValueFacts::I64(source),
            TypedInsertColumnValues::I64(values),
            SqlType::Int8 | SqlType::Timestamp,
        ) => values.get(row) == Some(&source),
        (
            DecodedTypedValueFacts::I128(source),
            TypedInsertColumnValues::I128(values),
            SqlType::Numeric { .. },
        ) => values.get(row) == Some(&source),
        (
            DecodedTypedValueFacts::Uuid(source),
            TypedInsertColumnValues::Bytes16(values),
            SqlType::Uuid,
        ) => values.get(row) == Some(&source),
        (
            DecodedTypedValueFacts::Bool(source),
            TypedInsertColumnValues::BoolBits(words),
            SqlType::Bool,
        ) => words
            .get(row / 32)
            .is_some_and(|word| (word & (1 << (row % 32)) != 0) == source),
        (
            DecodedTypedValueFacts::Text(source),
            TypedInsertColumnValues::Text { offsets, bytes },
            SqlType::Text,
        ) => {
            let Some((&start, &end)) = offsets.get(row).zip(offsets.get(row + 1)) else {
                return Ok(false);
            };
            bytes.get(start as usize..end as usize) == Some(source.as_bytes())
        }
        _ => false,
    })
}

pub(super) fn final_row_digest(
    graph: &ReservedSemanticsV2Graph,
    stable_table_id: u64,
    image: &DecodedTypedImage,
    transition: &RetainedTransition,
) -> Result<[u8; 32], EngineError> {
    let table = &graph.tables[transition.table_ref as usize];
    let mut digest = begin(b"gpu-db/write001/s7-final-row/v2");
    digest.update(stable_table_id.to_le_bytes());
    digest.update(transition.stable_row_id.to_le_bytes());
    digest.update(transition.image_ref.to_le_bytes());
    digest.update(transition.image_row_ordinal.to_le_bytes());
    digest.update(table.catalog_column_count.to_le_bytes());
    for column in image.columns() {
        digest.update(column.catalog_column_ordinal.to_le_bytes());
        digest.update(column.stable_column_id.to_le_bytes());
        digest.update(column.attnum.to_le_bytes());
        digest.update(storage(column.ty));
        digest.update(column.type_oid.to_le_bytes());
        digest.update(column.type_size.to_le_bytes());
        append_image_value(
            &mut digest,
            column.validity,
            column.values,
            column.ty,
            image.facts().rows,
            transition.image_row_ordinal,
        )?;
    }
    Ok(digest.finalize().into())
}

pub(super) fn typed_value_digest_from_image(
    image: &DecodedTypedImage,
    row: u32,
    catalog_column: u32,
    storage_type: [u8; 4],
    type_oid: u32,
    type_size: i16,
) -> Result<[u8; 32], EngineError> {
    Ok(typed_value_facts_from_image(
        image,
        row,
        catalog_column,
        storage_type,
        type_oid,
        type_size,
    )?
    .2)
}

/// The component directory retains the validity bit and logical byte length separately from its
/// value digest.  Recompute all three from the final image so an attacker cannot make a NULL
/// cell look participating while keeping a coherent typed-value digest.
pub(super) fn typed_value_facts_from_image(
    image: &DecodedTypedImage,
    row: u32,
    catalog_column: u32,
    storage_type: [u8; 4],
    type_oid: u32,
    type_size: i16,
) -> Result<(u8, u32, [u8; 32]), EngineError> {
    let column = image
        .columns()
        .find(|column| column.catalog_column_ordinal == catalog_column)
        .ok_or_else(|| error("key image column is absent"))?;
    if storage(column.ty) != storage_type
        || column.type_oid != type_oid
        || column.type_size != type_size
    {
        return Err(error("key image metadata differs from component"));
    }
    let validity = image_cell_validity(column.validity, image.facts().rows, row)?;
    let value_bytes = image_value_bytes(
        column.validity,
        column.values,
        column.ty,
        image.facts().rows,
        row,
    )?;
    let mut digest = begin(b"gpu-db/write001/s7-typed-key-value/v2");
    digest.update(storage_type);
    digest.update(type_oid.to_le_bytes());
    digest.update(type_size.to_le_bytes());
    append_image_value(
        &mut digest,
        column.validity,
        column.values,
        column.ty,
        image.facts().rows,
        row,
    )?;
    Ok((validity, value_bytes, digest.finalize().into()))
}

pub(super) fn append_image_value(
    digest: &mut Sha256,
    validity: &TypedInsertColumnValidity,
    values: &TypedInsertColumnValues,
    ty: SqlType,
    rows: u32,
    row: u32,
) -> Result<(), EngineError> {
    let row = usize::try_from(row).map_err(|_| error("image row overflows"))?;
    if row >= rows as usize {
        return Err(error("image row is absent"));
    }
    let valid = image_cell_validity(validity, rows, row as u32)? == 0;
    digest.update([u8::from(!valid)]);
    if !valid {
        digest.update(0_u32.to_le_bytes());
        return Ok(());
    }
    match (values, ty) {
        (TypedInsertColumnValues::I32(values), SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
            let value = values
                .get(row)
                .ok_or_else(|| error("i32 image vector is short"))?;
            digest.update(4_u32.to_le_bytes());
            digest.update(value.to_le_bytes());
        }
        (TypedInsertColumnValues::I64(values), SqlType::Int8 | SqlType::Timestamp) => {
            let value = values
                .get(row)
                .ok_or_else(|| error("i64 image vector is short"))?;
            digest.update(8_u32.to_le_bytes());
            digest.update(value.to_le_bytes());
        }
        (TypedInsertColumnValues::I128(values), SqlType::Numeric { .. }) => {
            let value = values
                .get(row)
                .ok_or_else(|| error("numeric image vector is short"))?;
            digest.update(16_u32.to_le_bytes());
            digest.update(value.to_le_bytes());
        }
        (TypedInsertColumnValues::Bytes16(values), SqlType::Uuid) => {
            let value = values
                .get(row)
                .ok_or_else(|| error("UUID image vector is short"))?;
            digest.update(16_u32.to_le_bytes());
            digest.update(value);
        }
        (TypedInsertColumnValues::BoolBits(words), SqlType::Bool) => {
            let word = words
                .get(row / 32)
                .ok_or_else(|| error("bool image vector is short"))?;
            digest.update(1_u32.to_le_bytes());
            digest.update([u8::from(word & (1 << (row % 32)) != 0)]);
        }
        (TypedInsertColumnValues::Text { offsets, bytes }, SqlType::Text) => {
            let start = *offsets
                .get(row)
                .ok_or_else(|| error("text image offsets are short"))?
                as usize;
            let end = *offsets
                .get(row + 1)
                .ok_or_else(|| error("text image offsets are short"))?
                as usize;
            let value = bytes
                .get(start..end)
                .ok_or_else(|| error("text image range is invalid"))?;
            digest.update((value.len() as u32).to_le_bytes());
            digest.update(value);
        }
        _ => return Err(error("image vector arm does not match SQL type")),
    }
    Ok(())
}

fn image_cell_validity(
    validity: &TypedInsertColumnValidity,
    rows: u32,
    row: u32,
) -> Result<u8, EngineError> {
    let row = usize::try_from(row).map_err(|_| error("image row overflows"))?;
    if row >= rows as usize {
        return Err(error("image row is absent"));
    }
    let valid = match validity {
        TypedInsertColumnValidity::AllValid => true,
        TypedInsertColumnValidity::Bitmap(words) => words
            .get(row / 32)
            .is_some_and(|word| word & (1 << (row % 32)) != 0),
    };
    Ok(u8::from(!valid))
}

fn image_value_bytes(
    validity: &TypedInsertColumnValidity,
    values: &TypedInsertColumnValues,
    ty: SqlType,
    rows: u32,
    row: u32,
) -> Result<u32, EngineError> {
    let row_index = usize::try_from(row).map_err(|_| error("image row overflows"))?;
    if image_cell_validity(validity, rows, row)? != 0 {
        return Ok(0);
    }
    match (values, ty) {
        (TypedInsertColumnValues::I32(values), SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
            values
                .get(row_index)
                .map(|_| 4)
                .ok_or_else(|| error("i32 image vector is short"))
        }
        (TypedInsertColumnValues::I64(values), SqlType::Int8 | SqlType::Timestamp) => values
            .get(row_index)
            .map(|_| 8)
            .ok_or_else(|| error("i64 image vector is short")),
        (TypedInsertColumnValues::I128(values), SqlType::Numeric { .. }) => values
            .get(row_index)
            .map(|_| 16)
            .ok_or_else(|| error("numeric image vector is short")),
        (TypedInsertColumnValues::Bytes16(values), SqlType::Uuid) => values
            .get(row_index)
            .map(|_| 16)
            .ok_or_else(|| error("UUID image vector is short")),
        (TypedInsertColumnValues::BoolBits(words), SqlType::Bool) => words
            .get(row_index / 32)
            .map(|_| 1)
            .ok_or_else(|| error("bool image vector is short")),
        (TypedInsertColumnValues::Text { offsets, bytes }, SqlType::Text) => {
            let start = *offsets
                .get(row_index)
                .ok_or_else(|| error("text image offsets are short"))?
                as usize;
            let end = *offsets
                .get(row_index + 1)
                .ok_or_else(|| error("text image offsets are short"))?
                as usize;
            let value = bytes
                .get(start..end)
                .ok_or_else(|| error("text image range is invalid"))?;
            u32::try_from(value.len()).map_err(|_| error("text image value length exceeds u32"))
        }
        _ => Err(error("image vector arm does not match SQL type")),
    }
}
