//! Inert canonical typed-vector and typed-image v2 authority.
//!
//! This module is deliberately below `typed_insert_batch`: it shares the sealed INSERT vector
//! grammar without becoming a second INSERT carrier.  The only callers today are the v1
//! canonical record codec and its tests.  In particular, it owns no WAL opcode, replay/apply,
//! device upload, result publication, or live execution capability.

use super::typed_image_codec_value_contract::{
    decoded_column_allocation_slots, decoded_column_owned_bytes, i32_values_are_valid,
    i64_values_are_valid, numeric_values_are_valid, sql_type_bytes, sql_type_from_bytes,
    text_shape_is_exact,
};
use super::*;
use sha2::{Digest, Sha256};

#[path = "typed_image_codec/decode_reservation.rs"]
mod decode_reservation;
#[path = "typed_image_codec/read_at.rs"]
mod read_at;
#[cfg(test)]
#[path = "typed_image_codec_tests.rs"]
mod tests;
pub(crate) use decode_reservation::TypedImageDecodeMeasure;
#[cfg(test)]
use decode_reservation::{
    fail_decode_reservation_for_test, observe_decode_reservation_stats_for_test,
    observe_decode_reservations_for_test,
};
use decode_reservation::{
    into_exact_boxed_slice, into_exact_boxed_str, reserve_decode_exact, reserve_decode_string,
};
pub(crate) use read_at::TypedImageReadAt;

pub(crate) const TYPED_IMAGE_HEADER_BYTES: u64 = 112;
pub(crate) const TYPED_IMAGE_DESCRIPTOR_BYTES: u64 = 96;
pub(crate) const TYPED_IMAGE_LAYOUT_DIGEST_DOMAIN: &[u8] = b"gpu-db/write001/image-layout/v2";
pub(crate) const TYPED_IMAGE_CONTENT_FINGERPRINT_DOMAIN: &[u8] =
    b"gpu-db/write001/image-content-fingerprint/v2";
pub(crate) const TYPED_VECTOR_DIGEST_DOMAIN: &[u8] = b"gpu-db/write001/typed-vector/v2";

const IMAGE_MAGIC: [u8; 16] = *b"GPUDBTYPEDIMAGE2";
const IMAGE_VERSION: u16 = 2;
const FINAL_TABLE_IMAGE: u32 = 1;
const RETAINED_RESPONSE: u32 = 2;
const DERIVED_U32: u32 = u32::MAX;
const DERIVED_ATNUM: i16 = i16::MIN;

/// The only two v2 image roles.  The bits are intentionally not combinable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TypedImageRole {
    FinalTableImage,
    RetainedResponse,
}

impl TypedImageRole {
    fn bits(self) -> u32 {
        match self {
            Self::FinalTableImage => FINAL_TABLE_IMAGE,
            Self::RetainedResponse => RETAINED_RESPONSE,
        }
    }

    fn decode(bits: u32) -> Result<Self, EngineError> {
        match bits {
            FINAL_TABLE_IMAGE => Ok(Self::FinalTableImage),
            RETAINED_RESPONSE => Ok(Self::RetainedResponse),
            _ => Err(image_error("image role flags are not exact")),
        }
    }
}

/// Borrowed catalog/projection metadata plus the sealed vector values.  This is a codec view,
/// not an execution plan: it intentionally has no row-id, source SQL, or replay authority.
pub(crate) struct TypedImageColumnView<'a> {
    pub(crate) catalog_column_ordinal: u32,
    pub(crate) stable_column_id: u32,
    pub(crate) table_ref: u32,
    pub(crate) attnum: i16,
    pub(crate) ty: SqlType,
    pub(crate) type_oid: u32,
    pub(crate) type_size: i16,
    pub(crate) result_format: u16,
    pub(crate) name: &'a str,
    pub(crate) validity: &'a TypedInsertColumnValidity,
    pub(crate) values: &'a TypedInsertColumnValues,
}

/// A borrowed image input.  Its columns must already be catalog/projection order.
pub(crate) struct TypedImageView<'a> {
    pub(crate) role: TypedImageRole,
    pub(crate) rows: u32,
    pub(crate) columns: &'a [TypedImageColumnView<'a>],
}

/// Exact capacity evidence retained for a later recovery-peak proof. `decoded_owned_bytes`
/// includes final retained vectors and heap metadata. The only concurrent scratch is the one
/// fallibly reserved descriptor directory; raw bodies are borrowed and never become a second
/// byte-producing owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TypedImageMeasure {
    encoded_bytes: u64,
    descriptor_bytes: u64,
    name_bytes: u64,
    vector_bytes: u64,
    decoded_owned_bytes: u64,
    encode_maximum_scratch_bytes: u64,
    decode_maximum_scratch_bytes: u64,
    encoded_allocation_slots: u64,
    decoded_persistent_allocation_slots: u64,
    encode_maximum_scratch_allocation_slots: u64,
    decode_maximum_scratch_allocation_slots: u64,
}

impl TypedImageMeasure {
    pub(crate) fn encoded_bytes(self) -> u64 {
        self.encoded_bytes
    }

    pub(crate) fn descriptor_bytes(self) -> u64 {
        self.descriptor_bytes
    }

    pub(crate) fn name_bytes(self) -> u64 {
        self.name_bytes
    }

    pub(crate) fn vector_bytes(self) -> u64 {
        self.vector_bytes
    }

    pub(crate) fn decoded_owned_bytes(self) -> u64 {
        self.decoded_owned_bytes
    }

    pub(crate) fn encode_maximum_scratch_bytes(self) -> u64 {
        self.encode_maximum_scratch_bytes
    }

    pub(crate) fn decode_maximum_scratch_bytes(self) -> u64 {
        self.decode_maximum_scratch_bytes
    }

    pub(crate) fn encoded_allocation_slots(self) -> u64 {
        self.encoded_allocation_slots
    }

    pub(crate) fn decoded_persistent_allocation_slots(self) -> u64 {
        self.decoded_persistent_allocation_slots
    }

    pub(crate) fn encode_maximum_scratch_allocation_slots(self) -> u64 {
        self.encode_maximum_scratch_allocation_slots
    }

    pub(crate) fn decode_maximum_scratch_allocation_slots(self) -> u64 {
        self.decode_maximum_scratch_allocation_slots
    }
}

/// Scalar image facts which can be borrowed by a future final replay owner without exposing raw
/// image bytes or an alternate byte-producing route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedTypedImageFacts {
    pub(crate) role: TypedImageRole,
    pub(crate) rows: u32,
    pub(crate) columns: u32,
    pub(crate) layout_digest: gpu_db_wal::CanonicalDigest,
}

/// Move-only decoded image owner.  It provides typed ownership for a future replay path but no
/// raw byte accessor, `Clone`, extracting conversion, or canonical byte-production route.
#[allow(dead_code)] // Reserved for inert codec-5 S7/S8 construction.
pub(crate) struct DecodedTypedImage {
    facts: DecodedTypedImageFacts,
    columns: Box<[DecodedTypedImageColumn]>,
}

struct DecodedTypedImageColumn {
    catalog_column_ordinal: u32,
    stable_column_id: u32,
    table_ref: u32,
    attnum: i16,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
    result_format: u16,
    name: Box<str>,
    validity: TypedInsertColumnValidity,
    values: TypedInsertColumnValues,
    vector_digest: gpu_db_wal::CanonicalDigest,
}

#[allow(dead_code)] // Future inert codec-5 final owner reads scalar/vector facts only.
impl DecodedTypedImage {
    pub(crate) fn facts(&self) -> DecodedTypedImageFacts {
        self.facts
    }

    pub(crate) fn columns(
        &self,
    ) -> impl ExactSizeIterator<Item = DecodedTypedImageColumnFacts<'_>> {
        self.columns
            .iter()
            .map(|column| DecodedTypedImageColumnFacts {
                catalog_column_ordinal: column.catalog_column_ordinal,
                stable_column_id: column.stable_column_id,
                table_ref: column.table_ref,
                attnum: column.attnum,
                ty: column.ty,
                type_oid: column.type_oid,
                type_size: column.type_size,
                result_format: column.result_format,
                name: &column.name,
                validity: &column.validity,
                values: &column.values,
                vector_digest: column.vector_digest,
            })
    }

    /// Rebind one final image to its stable-table-ordered S7 reference before runtime generation.
    /// Re-encoding is required because the table reference participates in the authenticated
    /// layout digest; callers receive the exact decoded/encoded pair used by both CUDA and WAL.
    pub(super) fn rebind_final_table_ref(
        mut self,
        table_ref: u32,
    ) -> Result<(Self, Arc<[u8]>), EngineError> {
        if self.facts.role != TypedImageRole::FinalTableImage
            || self.facts.rows == 0
            || table_ref == DERIVED_U32
        {
            return Err(image_error("final image table reference is invalid"));
        }
        for column in &mut self.columns {
            column.table_ref = table_ref;
        }
        let views = self
            .columns
            .iter()
            .map(|column| TypedImageColumnView {
                catalog_column_ordinal: column.catalog_column_ordinal,
                stable_column_id: column.stable_column_id,
                table_ref: column.table_ref,
                attnum: column.attnum,
                ty: column.ty,
                type_oid: column.type_oid,
                type_size: column.type_size,
                result_format: column.result_format,
                name: &column.name,
                validity: &column.validity,
                values: &column.values,
            })
            .collect::<Vec<_>>();
        let encoded: Arc<[u8]> = encode_typed_image(&TypedImageView {
            role: TypedImageRole::FinalTableImage,
            rows: self.facts.rows,
            columns: &views,
        })?
        .into();
        Ok((decode_typed_image(&encoded)?, encoded))
    }

    /// Concatenate same-table strict images directly at their S7 table reference. Each statement
    /// keeps its own S2 record; this produces the sole catalog-order physical image consumed by
    /// the transaction's CUDA generation and device plan.  Do not first materialize an encoded
    /// reference-neutral combined image only to encode/decode it again for its final S7 reference.
    pub(super) fn concatenate_final_table_images_for_table_ref(
        images: Vec<Self>,
        table_ref: u32,
    ) -> Result<(Self, Arc<[u8]>), EngineError> {
        if table_ref == DERIVED_U32 {
            return Err(image_error("final image table reference is invalid"));
        }
        let first = images
            .first()
            .ok_or_else(|| image_error("final image concatenation is empty"))?;
        if first.facts.role != TypedImageRole::FinalTableImage || first.facts.rows == 0 {
            return Err(image_error(
                "final image concatenation requires nonempty table images",
            ));
        }
        let column_count = first.columns.len();
        let mut total_rows = 0_u32;
        for image in &images {
            if image.facts.role != TypedImageRole::FinalTableImage
                || image.facts.rows == 0
                || image.columns.len() != column_count
                || image
                    .columns
                    .iter()
                    .zip(first.columns.iter())
                    .any(|(column, expected)| {
                        column.catalog_column_ordinal != expected.catalog_column_ordinal
                            || column.stable_column_id != expected.stable_column_id
                            || column.table_ref != expected.table_ref
                            || column.attnum != expected.attnum
                            || column.ty != expected.ty
                            || column.type_oid != expected.type_oid
                            || column.type_size != expected.type_size
                            || column.result_format != expected.result_format
                            || column.name != expected.name
                    })
            {
                return Err(image_error(
                    "final image concatenation schema or identity differs",
                ));
            }
            total_rows = total_rows
                .checked_add(image.facts.rows)
                .ok_or_else(|| image_error("final image concatenation row count overflows"))?;
        }

        let mut groups = (0..column_count)
            .map(|_| Vec::with_capacity(images.len()))
            .collect::<Vec<_>>();
        for image in images {
            for (ordinal, column) in image.columns.into_vec().into_iter().enumerate() {
                groups[ordinal].push((image.facts.rows, column));
            }
        }
        let mut columns = Vec::with_capacity(column_count);
        for group in groups {
            columns.push(concatenate_final_image_column(group, total_rows)?);
        }
        let mut candidate = Self {
            facts: DecodedTypedImageFacts {
                role: TypedImageRole::FinalTableImage,
                rows: total_rows,
                columns: u32::try_from(column_count)
                    .map_err(|_| image_error("final image column count exceeds u32"))?,
                layout_digest: [0; 32],
            },
            columns: columns.into_boxed_slice(),
        };
        for column in &mut candidate.columns {
            column.table_ref = table_ref;
        }
        let views = candidate
            .columns
            .iter()
            .map(|column| TypedImageColumnView {
                catalog_column_ordinal: column.catalog_column_ordinal,
                stable_column_id: column.stable_column_id,
                table_ref: column.table_ref,
                attnum: column.attnum,
                ty: column.ty,
                type_oid: column.type_oid,
                type_size: column.type_size,
                result_format: column.result_format,
                name: &column.name,
                validity: &column.validity,
                values: &column.values,
            })
            .collect::<Vec<_>>();
        let encoded: Arc<[u8]> = encode_typed_image(&TypedImageView {
            role: TypedImageRole::FinalTableImage,
            rows: total_rows,
            columns: &views,
        })?
        .into();
        // Strictly decode the exact S7-bound combined bytes once, then move that one vector owner
        // directly into the resident source. This retains byte-level validation without
        // revalidating an intermediate reference-neutral image that no authority can observe.
        let decoded = decode_typed_image(&encoded)?;
        Ok((decoded, encoded))
    }

    /// Consume one strict final-table image into the sole columnar append carrier. The image
    /// grammar has already sealed every vector body; this boundary only
    /// binds those owners to one exact live catalog table before moving them.  It deliberately
    /// cannot recover SQL text, construct row values, or surface an image column for mutation.
    pub(super) fn into_resident_append_source(
        self,
        table: &RelationalTable,
        table_schema_digest: gpu_db_wal::CanonicalDigest,
        prepared_catalog_seq: Index,
    ) -> Result<PreparedResidentAppendSource, EngineError> {
        let DecodedTypedImage { facts, columns } = self;
        let column_count = usize::try_from(facts.columns).map_err(|_| {
            resident_source_error("final image column count exceeds addressability")
        })?;
        if facts.role != TypedImageRole::FinalTableImage
            || column_count != columns.len()
            || column_count != table.columns.len()
            || table.stable_table_id == 0
            || table.stable_table_id == u64::MAX
            || table.oid == 0
            || crate::engine_transaction_reset::table_schema_digest(table)
                .map_err(|_| resident_source_error("supplied table schema cannot be digested"))?
                != table_schema_digest
        {
            return Err(resident_source_error(
                "final image/table schema or identity does not match",
            ));
        }

        for (ordinal, (decoded, catalog)) in columns.iter().zip(&table.columns).enumerate() {
            let catalog_ordinal = u32::try_from(ordinal)
                .map_err(|_| resident_source_error("catalog column ordinal exceeds u32"))?;
            if decoded.catalog_column_ordinal != catalog_ordinal
                || decoded.stable_column_id != catalog.id
                || decoded.attnum != catalog.attnum
                || decoded.ty != catalog.ty
                || decoded.type_oid != catalog.type_oid
                || decoded.type_size != catalog.type_size
                || catalog.table_oid != table.oid
                || decoded.result_format != 0
                || !decoded.name.is_empty()
                || !is_live_resident_append_type(decoded.ty)
            {
                return Err(resident_source_error(
                    "final image catalog-order column identity does not match",
                ));
            }
        }

        let requires_dense_rollover = columns
            .iter()
            .any(|column| column.ty == SqlType::Text || !column.validity.is_all_valid());
        #[cfg(feature = "probe-timing")]
        {
            let non_all_valid = columns
                .iter()
                .filter(|column| !column.validity.is_all_valid())
                .count();
            eprintln!(
                "[probe] codec5_final_image_source table={} rows={} columns={} non_all_valid={} dense={}",
                table.name,
                facts.rows,
                columns.len(),
                non_all_valid,
                requires_dense_rollover,
            );
        }
        let schema: Arc<str> = Arc::from(table.schema.as_str());
        let name: Arc<str> = Arc::from(table.name.as_str());
        let mut prepared_columns = Vec::new();
        prepared_columns
            .try_reserve_exact(columns.len())
            .map_err(|_| resident_source_error("resident source column reservation failed"))?;
        for column in columns.into_vec() {
            prepared_columns.push(PreparedResidentAppendColumn {
                column_id: column.stable_column_id,
                attnum: column.attnum,
                ty: column.ty,
                type_oid: column.type_oid,
                type_size: column.type_size,
                validity: Some(column.validity),
                values: Some(column.values),
            });
        }
        let rows = usize::try_from(facts.rows)
            .map_err(|_| resident_source_error("final image rows exceed addressability"))?;
        let runtime_value_bytes =
            resident_source::runtime_generation_value_bytes(rows, &prepared_columns)?;
        Ok(PreparedResidentAppendSource {
            table: TypedInsertBatchTable {
                schema: Arc::clone(&schema),
                name: Arc::clone(&name),
                stable_table_id: table.stable_table_id,
                oid: table.oid,
                schema_digest: table_schema_digest,
                prepared_catalog_seq,
            },
            row_count: facts.rows,
            columns: into_exact_boxed_slice(
                prepared_columns,
                "recovery resident source column directory",
            )?,
            runtime_value_bytes,
            dependencies: Box::new([TypedInsertDependencyBinding {
                schema,
                name,
                oid: table.oid,
                schema_digest: table_schema_digest,
            }]),
            requires_dense_rollover,
        })
    }
}

/// Encode the transaction overlay's already-resolved final rows into the same catalog-order
/// image consumed by runtime generation, the device plan, and recovery.  This is deliberately an
/// image transformation, not another INSERT semantic carrier: the original S2 records remain the
/// statement authority, while these values are the final private-overlay outcome of later DML.
pub(crate) fn encode_final_table_image_from_resolved_rows(
    table_ref: u32,
    table: &RelationalTable,
    rows: &[Vec<SqlValue>],
) -> Result<(DecodedTypedImage, Arc<[u8]>), EngineError> {
    if table_ref == DERIVED_U32
        || table.columns.is_empty()
        || rows.iter().any(|row| row.len() != table.columns.len())
    {
        return Err(image_error(
            "resolved final rows do not match their catalog table geometry",
        ));
    }
    let row_count = rows.len();
    let mut validities = Vec::with_capacity(table.columns.len());
    let mut values = Vec::with_capacity(table.columns.len());
    for (column_ordinal, column) in table.columns.iter().enumerate() {
        let mut validity = vec![0_u32; bitmap_words(row_count)?];
        let column_values = match column.ty {
            SqlType::Int2 | SqlType::Int4 | SqlType::Date => {
                let mut output = vec![0_i32; row_count];
                for (row_ordinal, row) in rows.iter().enumerate() {
                    match (&row[column_ordinal], column.ty) {
                        (SqlValue::Null, _) => continue,
                        (SqlValue::Int2(value), SqlType::Int2) => {
                            output[row_ordinal] = i32::from(*value)
                        }
                        (SqlValue::Int4(value), SqlType::Int4)
                        | (SqlValue::Date(value), SqlType::Date) => output[row_ordinal] = *value,
                        _ => {
                            return Err(image_error(
                                "resolved final i32-section value changed SQL type",
                            ));
                        }
                    }
                    set_bit(&mut validity, row_ordinal)?;
                }
                TypedInsertColumnValues::I32(output.into_boxed_slice())
            }
            SqlType::Int8 | SqlType::Timestamp => {
                let mut output = vec![0_i64; row_count];
                for (row_ordinal, row) in rows.iter().enumerate() {
                    match (&row[column_ordinal], column.ty) {
                        (SqlValue::Null, _) => continue,
                        (SqlValue::Int8(value), SqlType::Int8)
                        | (SqlValue::Timestamp(value), SqlType::Timestamp) => {
                            output[row_ordinal] = *value
                        }
                        _ => {
                            return Err(image_error(
                                "resolved final i64-section value changed SQL type",
                            ));
                        }
                    }
                    set_bit(&mut validity, row_ordinal)?;
                }
                TypedInsertColumnValues::I64(output.into_boxed_slice())
            }
            SqlType::Numeric { scale, .. } => {
                let mut output = vec![0_i128; row_count];
                for (row_ordinal, row) in rows.iter().enumerate() {
                    match &row[column_ordinal] {
                        SqlValue::Null => continue,
                        SqlValue::Numeric(value) if value.scale == scale => {
                            output[row_ordinal] = value.mantissa
                        }
                        _ => {
                            return Err(image_error(
                                "resolved final NUMERIC value changed SQL type or scale",
                            ));
                        }
                    }
                    set_bit(&mut validity, row_ordinal)?;
                }
                TypedInsertColumnValues::I128(output.into_boxed_slice())
            }
            SqlType::Uuid => {
                let mut output = vec![[0_u8; 16]; row_count];
                for (row_ordinal, row) in rows.iter().enumerate() {
                    match &row[column_ordinal] {
                        SqlValue::Null => continue,
                        SqlValue::Uuid(value) => output[row_ordinal] = *value,
                        _ => {
                            return Err(image_error("resolved final UUID value changed SQL type"));
                        }
                    }
                    set_bit(&mut validity, row_ordinal)?;
                }
                TypedInsertColumnValues::Bytes16(output.into_boxed_slice())
            }
            SqlType::Bool => {
                let mut output = vec![0_u32; bitmap_words(row_count)?];
                for (row_ordinal, row) in rows.iter().enumerate() {
                    match &row[column_ordinal] {
                        SqlValue::Null => continue,
                        SqlValue::Bool(value) => {
                            if *value {
                                set_bit(&mut output, row_ordinal)?;
                            }
                        }
                        _ => {
                            return Err(image_error("resolved final BOOL value changed SQL type"));
                        }
                    }
                    set_bit(&mut validity, row_ordinal)?;
                }
                TypedInsertColumnValues::BoolBits(output.into_boxed_slice())
            }
            SqlType::Text => {
                let mut offsets = Vec::with_capacity(row_count + 1);
                let mut output = Vec::new();
                offsets.push(0_u64);
                for (row_ordinal, row) in rows.iter().enumerate() {
                    match &row[column_ordinal] {
                        SqlValue::Null => {}
                        SqlValue::Text(value) => {
                            output.try_reserve(value.len()).map_err(|_| {
                                image_error("resolved final TEXT reservation failed")
                            })?;
                            output.extend_from_slice(value.as_bytes());
                            set_bit(&mut validity, row_ordinal)?;
                        }
                        _ => {
                            return Err(image_error("resolved final TEXT value changed SQL type"));
                        }
                    }
                    offsets.push(
                        u64::try_from(output.len())
                            .map_err(|_| image_error("resolved final TEXT bytes exceed u64"))?,
                    );
                }
                TypedInsertColumnValues::Text {
                    offsets: offsets.into_boxed_slice(),
                    bytes: output.into_boxed_slice(),
                }
            }
        };
        validities.push(if bitmap_is_all_set(&validity, row_count) {
            TypedInsertColumnValidity::AllValid
        } else {
            TypedInsertColumnValidity::Bitmap(validity.into_boxed_slice())
        });
        values.push(column_values);
    }
    let views = table
        .columns
        .iter()
        .enumerate()
        .map(|(ordinal, column)| TypedImageColumnView {
            catalog_column_ordinal: u32::try_from(ordinal)
                .expect("catalog column count is already bounded"),
            stable_column_id: column.id,
            table_ref,
            attnum: column.attnum,
            ty: column.ty,
            type_oid: column.type_oid,
            type_size: column.type_size,
            result_format: 0,
            name: "",
            validity: &validities[ordinal],
            values: &values[ordinal],
        })
        .collect::<Vec<_>>();
    let encoded: Arc<[u8]> = encode_typed_image(&TypedImageView {
        role: TypedImageRole::FinalTableImage,
        rows: u32::try_from(row_count)
            .map_err(|_| image_error("resolved final row count exceeds u32"))?,
        columns: &views,
    })?
    .into();
    Ok((decode_typed_image(&encoded)?, encoded))
}

fn concatenate_final_image_column(
    group: Vec<(u32, DecodedTypedImageColumn)>,
    total_rows: u32,
) -> Result<DecodedTypedImageColumn, EngineError> {
    let total_rows = usize::try_from(total_rows)
        .map_err(|_| image_error("final image row count exceeds addressability"))?;
    let first = group
        .first()
        .ok_or_else(|| image_error("final image column group is empty"))?;
    let mut validity_words = vec![0_u32; bitmap_words(total_rows)?];
    let mut row_base = 0_usize;
    for (rows, column) in &group {
        let rows = usize::try_from(*rows)
            .map_err(|_| image_error("final image row count exceeds addressability"))?;
        if !column.validity.shape_is_exact(rows) || !column.values.rows_match(rows) {
            return Err(image_error(
                "final image column vector geometry differs before concatenation",
            ));
        }
        for row in 0..rows {
            if column.validity.is_valid(row) {
                set_bit(&mut validity_words, row_base + row)?;
            }
        }
        row_base = row_base
            .checked_add(rows)
            .ok_or_else(|| image_error("final image row offset overflows"))?;
    }
    if row_base != total_rows {
        return Err(image_error(
            "final image column rows do not exhaust the combined image",
        ));
    }
    let validity = if bitmap_is_all_set(&validity_words, total_rows) {
        TypedInsertColumnValidity::AllValid
    } else {
        TypedInsertColumnValidity::Bitmap(validity_words.into_boxed_slice())
    };
    let ty = first.1.ty;
    let values = match ty {
        SqlType::Int2 | SqlType::Int4 | SqlType::Date => TypedInsertColumnValues::I32(
            group
                .iter()
                .flat_map(|(_, column)| match &column.values {
                    TypedInsertColumnValues::I32(values) => values.iter().copied(),
                    _ => [].iter().copied(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
        SqlType::Int8 | SqlType::Timestamp => TypedInsertColumnValues::I64(
            group
                .iter()
                .flat_map(|(_, column)| match &column.values {
                    TypedInsertColumnValues::I64(values) => values.iter().copied(),
                    _ => [].iter().copied(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
        SqlType::Numeric { .. } => TypedInsertColumnValues::I128(
            group
                .iter()
                .flat_map(|(_, column)| match &column.values {
                    TypedInsertColumnValues::I128(values) => values.iter().copied(),
                    _ => [].iter().copied(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
        SqlType::Uuid => TypedInsertColumnValues::Bytes16(
            group
                .iter()
                .flat_map(|(_, column)| match &column.values {
                    TypedInsertColumnValues::Bytes16(values) => values.iter().copied(),
                    _ => [].iter().copied(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
        SqlType::Bool => {
            let mut words = vec![0_u32; bitmap_words(total_rows)?];
            let mut row_base = 0_usize;
            for (rows, column) in &group {
                let rows = *rows as usize;
                let TypedInsertColumnValues::BoolBits(source) = &column.values else {
                    return Err(image_error("final image BOOL storage arm differs"));
                };
                for row in 0..rows {
                    if bit_is_set(source, row) {
                        set_bit(&mut words, row_base + row)?;
                    }
                }
                row_base += rows;
            }
            TypedInsertColumnValues::BoolBits(words.into_boxed_slice())
        }
        SqlType::Text => {
            let mut offsets = Vec::with_capacity(total_rows + 1);
            let mut bytes = Vec::new();
            offsets.push(0_u64);
            for (_, column) in &group {
                let TypedInsertColumnValues::Text {
                    offsets: source_offsets,
                    bytes: source_bytes,
                } = &column.values
                else {
                    return Err(image_error("final image TEXT storage arm differs"));
                };
                let base = u64::try_from(bytes.len())
                    .map_err(|_| image_error("final image TEXT bytes exceed u64"))?;
                bytes.extend_from_slice(source_bytes);
                offsets.extend(source_offsets.iter().skip(1).map(|offset| base + *offset));
            }
            TypedInsertColumnValues::Text {
                offsets: offsets.into_boxed_slice(),
                bytes: bytes.into_boxed_slice(),
            }
        }
    };
    if !values.rows_match(total_rows) {
        return Err(image_error(
            "final image concatenated storage arm lost row geometry",
        ));
    }
    let (_, first) = group
        .into_iter()
        .next()
        .expect("validated nonempty final image column group");
    Ok(DecodedTypedImageColumn {
        catalog_column_ordinal: first.catalog_column_ordinal,
        stable_column_id: first.stable_column_id,
        table_ref: first.table_ref,
        attnum: first.attnum,
        ty: first.ty,
        type_oid: first.type_oid,
        type_size: first.type_size,
        result_format: first.result_format,
        name: first.name,
        validity,
        values,
        // The candidate is immediately encoded and strictly decoded above; this placeholder is
        // never exposed as a retained image owner.
        vector_digest: [0; 32],
    })
}

impl DecodedTypedImageColumnFacts<'_> {
    pub(crate) fn logical_value_len_at(&self, row: usize) -> Result<usize, EngineError> {
        self.with_logical_cell_at(row, |_is_null, value| value.len())
    }

    /// Visit one canonical logical cell without manufacturing a row matrix. Fixed-width values
    /// use a stack-local little-endian buffer for the duration of the callback; TEXT borrows its
    /// existing arena and NULL always visits an empty slice.
    pub(crate) fn with_logical_cell_at<R>(
        &self,
        row: usize,
        consume: impl FnOnce(bool, &[u8]) -> R,
    ) -> Result<R, EngineError> {
        let is_valid = self.validity.is_valid(row);
        if !is_valid {
            return Ok(consume(true, &[]));
        }
        match (self.values, self.ty) {
            (
                TypedInsertColumnValues::I32(values),
                SqlType::Int2 | SqlType::Int4 | SqlType::Date,
            ) => {
                let value = values
                    .get(row)
                    .ok_or_else(|| image_error("logical i32 cell row is out of range"))?
                    .to_le_bytes();
                Ok(consume(false, &value))
            }
            (TypedInsertColumnValues::I64(values), SqlType::Int8 | SqlType::Timestamp) => {
                let value = values
                    .get(row)
                    .ok_or_else(|| image_error("logical i64 cell row is out of range"))?
                    .to_le_bytes();
                Ok(consume(false, &value))
            }
            (TypedInsertColumnValues::I128(values), SqlType::Numeric { .. }) => {
                let value = values
                    .get(row)
                    .ok_or_else(|| image_error("logical i128 cell row is out of range"))?
                    .to_le_bytes();
                Ok(consume(false, &value))
            }
            (TypedInsertColumnValues::Bytes16(values), SqlType::Uuid) => values
                .get(row)
                .map(|value| consume(false, value))
                .ok_or_else(|| image_error("logical UUID cell row is out of range")),
            (TypedInsertColumnValues::BoolBits(words), SqlType::Bool) => {
                if row / 32 >= words.len() {
                    return Err(image_error("logical BOOL cell row is out of range"));
                }
                let value = [u8::from(words[row / 32] & (1_u32 << (row % 32)) != 0)];
                Ok(consume(false, &value))
            }
            (TypedInsertColumnValues::Text { offsets, bytes }, SqlType::Text) => {
                let start = offsets
                    .get(row)
                    .and_then(|value| usize::try_from(*value).ok())
                    .ok_or_else(|| image_error("logical TEXT start offset is invalid"))?;
                let end = offsets
                    .get(row + 1)
                    .and_then(|value| usize::try_from(*value).ok())
                    .ok_or_else(|| image_error("logical TEXT end offset is invalid"))?;
                let value = bytes
                    .get(start..end)
                    .ok_or_else(|| image_error("logical TEXT cell range is invalid"))?;
                Ok(consume(false, value))
            }
            _ => Err(image_error(
                "logical cell vector does not match its SQL type",
            )),
        }
    }
}

pub(crate) const fn typed_image_sql_storage(ty: SqlType) -> [u8; 4] {
    match ty {
        SqlType::Int2 => [1, 0, 0, 0],
        SqlType::Int4 => [2, 0, 0, 0],
        SqlType::Int8 => [3, 0, 0, 0],
        SqlType::Numeric { precision, scale } => [4, precision, scale, 0],
        SqlType::Bool => [5, 0, 0, 0],
        SqlType::Text => [6, 0, 0, 0],
        SqlType::Date => [7, 0, 0, 0],
        SqlType::Timestamp => [8, 0, 0, 0],
        SqlType::Uuid => [9, 0, 0, 0],
    }
}

/// Borrowed typed values from the move-only image owner.  The slices cannot outlive it and no
/// byte representation is exposed.
pub(crate) struct DecodedTypedImageColumnFacts<'a> {
    pub(crate) catalog_column_ordinal: u32,
    pub(crate) stable_column_id: u32,
    pub(crate) table_ref: u32,
    pub(crate) attnum: i16,
    pub(crate) ty: SqlType,
    pub(crate) type_oid: u32,
    pub(crate) type_size: i16,
    pub(crate) result_format: u16,
    pub(crate) name: &'a str,
    pub(crate) validity: &'a TypedInsertColumnValidity,
    pub(crate) values: &'a TypedInsertColumnValues,
    pub(crate) vector_digest: gpu_db_wal::CanonicalDigest,
}

/// The small adapter used by the v1 canonical record writer.  It keeps vector grammar in this
/// file while v1 retains its existing outer framing and bytes.
pub(super) trait TypedVectorSink {
    fn vector_bytes(&mut self, bytes: &[u8]) -> Result<(), EngineError>;
}

/// Append the exact validity grammar shared by v1 and v2.
pub(super) fn append_typed_validity<S: TypedVectorSink>(
    out: &mut S,
    validity: &TypedInsertColumnValidity,
    rows: u32,
) -> Result<(), EngineError> {
    match validity {
        TypedInsertColumnValidity::AllValid => out.vector_bytes(&[0]),
        TypedInsertColumnValidity::Bitmap(words) => {
            let rows =
                usize::try_from(rows).map_err(|_| image_error("row count addressability"))?;
            if !bitmap_shape_is_exact(words, rows) || bitmap_is_all_set(words, rows) {
                return Err(image_error("validity bitmap is not canonical"));
            }
            append_u8(out, 1)?;
            append_bitmap_words(out, words)
        }
    }
}

/// Append the exact value grammar shared by v1 and v2.  The validity form is deliberately
/// separate because v1 interleaves other sealed metadata between validity and values.
pub(super) fn append_typed_values<S: TypedVectorSink>(
    out: &mut S,
    values: &TypedInsertColumnValues,
    ty: SqlType,
    rows: u32,
) -> Result<(), EngineError> {
    let expected = usize::try_from(rows).map_err(|_| image_error("row count addressability"))?;
    match (values, ty) {
        (TypedInsertColumnValues::I32(values), SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
            let payload = checked_u32_len(
                values
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| image_error("i32 payload overflows"))?,
                "i32 payload",
            )?;
            if values.len() != expected || !i32_values_are_valid(values, ty) {
                return Err(image_error("i32 vector shape or bounds drifted"));
            }
            append_vector_header(out, 1, rows, payload)?;
            for value in values.iter() {
                append_i32(out, *value)?;
            }
        }
        (TypedInsertColumnValues::I64(values), SqlType::Int8 | SqlType::Timestamp) => {
            let payload = checked_u32_len(
                values
                    .len()
                    .checked_mul(8)
                    .ok_or_else(|| image_error("i64 payload overflows"))?,
                "i64 payload",
            )?;
            if values.len() != expected || !i64_values_are_valid(values, ty) {
                return Err(image_error("i64 vector shape or bounds drifted"));
            }
            append_vector_header(out, 2, rows, payload)?;
            for value in values.iter() {
                append_i64(out, *value)?;
            }
        }
        (TypedInsertColumnValues::I128(values), ty @ SqlType::Numeric { .. }) => {
            let payload = checked_u32_len(
                values
                    .len()
                    .checked_mul(16)
                    .ok_or_else(|| image_error("numeric payload overflows"))?,
                "numeric payload",
            )?;
            if values.len() != expected || !numeric_values_are_valid(values, ty) {
                return Err(image_error("numeric vector shape or bounds drifted"));
            }
            append_vector_header(out, 3, rows, payload)?;
            for value in values.iter() {
                append_i128(out, *value)?;
            }
        }
        (TypedInsertColumnValues::Bytes16(values), SqlType::Uuid) => {
            let payload = checked_u32_len(
                values
                    .len()
                    .checked_mul(16)
                    .ok_or_else(|| image_error("uuid payload overflows"))?,
                "uuid payload",
            )?;
            if values.len() != expected {
                return Err(image_error("uuid vector shape drifted"));
            }
            append_vector_header(out, 4, rows, payload)?;
            for value in values.iter() {
                out.vector_bytes(value)?;
            }
        }
        (TypedInsertColumnValues::BoolBits(words), SqlType::Bool) => {
            if !bitmap_shape_is_exact(words, expected) {
                return Err(image_error("bool bitmap shape drifted"));
            }
            let payload = checked_u32_len(
                words
                    .len()
                    .checked_mul(4)
                    .and_then(|bytes| bytes.checked_add(4))
                    .ok_or_else(|| image_error("bool payload overflows"))?,
                "bool payload",
            )?;
            append_vector_header(out, 5, rows, payload)?;
            append_bitmap_words(out, words)?;
        }
        (TypedInsertColumnValues::Text { offsets, bytes }, SqlType::Text) => {
            if !text_shape_is_exact(offsets, bytes, expected) {
                return Err(image_error("text vector shape drifted"));
            }
            let payload = offsets
                .len()
                .checked_mul(8)
                .and_then(|offset_bytes| offset_bytes.checked_add(8))
                .and_then(|prefix_bytes| prefix_bytes.checked_add(bytes.len()))
                .ok_or_else(|| image_error("text payload overflows"))?;
            append_vector_header(out, 6, rows, checked_u32_len(payload, "text payload")?)?;
            append_u32(out, checked_u32_len(offsets.len(), "text offset count")?)?;
            for offset in offsets.iter() {
                append_u64(out, *offset)?;
            }
            append_u32(out, checked_u32_len(bytes.len(), "text byte length")?)?;
            out.vector_bytes(bytes)?;
        }
        _ => return Err(image_error("typed vector arm disagrees with SQL type")),
    }
    Ok(())
}

pub(super) fn typed_values_shape_is_valid(
    values: &TypedInsertColumnValues,
    ty: SqlType,
    rows: usize,
) -> bool {
    match (values, ty) {
        (TypedInsertColumnValues::I32(values), SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
            values.len() == rows && i32_values_are_valid(values, ty)
        }
        (TypedInsertColumnValues::I64(values), SqlType::Int8 | SqlType::Timestamp) => {
            values.len() == rows && i64_values_are_valid(values, ty)
        }
        (TypedInsertColumnValues::I128(values), ty @ SqlType::Numeric { .. }) => {
            values.len() == rows && numeric_values_are_valid(values, ty)
        }
        (TypedInsertColumnValues::Bytes16(values), SqlType::Uuid) => values.len() == rows,
        (TypedInsertColumnValues::BoolBits(words), SqlType::Bool) => {
            bitmap_shape_is_exact(words, rows)
        }
        (TypedInsertColumnValues::Text { offsets, bytes }, SqlType::Text) => {
            text_shape_is_exact(offsets, bytes, rows)
        }
        _ => false,
    }
}

/// Allocation-free exact measure for a full v2 image.  It uses the same append traversal as
/// encoding, but the counting sink stores no output bytes.
pub(crate) fn measure_typed_image(
    view: &TypedImageView<'_>,
) -> Result<TypedImageMeasure, EngineError> {
    let columns = checked_u32_len(view.columns.len(), "image column count")?;
    let _cells = u64::from(view.rows)
        .checked_mul(u64::from(columns))
        .ok_or_else(|| image_error("image cell count overflows"))?;
    let descriptor_bytes = u64::from(columns)
        .checked_mul(TYPED_IMAGE_DESCRIPTOR_BYTES)
        .ok_or_else(|| image_error("descriptor bytes overflow"))?;
    let mut name_bytes = 0_u64;
    let mut vector_bytes = 0_u64;
    let mut decoded_persistent_allocation_slots: u64 = if columns == 0 { 0 } else { 1 };
    let column_owner_bytes = u64::try_from(view.columns.len())
        .map_err(|_| image_error("decoded column count addressability"))?
        .checked_mul(
            u64::try_from(std::mem::size_of::<DecodedTypedImageColumn>())
                .map_err(|_| image_error("decoded column size addressability"))?,
        )
        .ok_or_else(|| image_error("decoded column owner bytes overflow"))?;
    let mut decoded_owned_bytes = column_owner_bytes;
    for (ordinal, column) in view.columns.iter().enumerate() {
        validate_column_view(view.role, ordinal, column, view.rows)?;
        let names = name_len(view.role, column)?;
        name_bytes = name_bytes
            .checked_add(names)
            .ok_or_else(|| image_error("name bytes overflow"))?;
        let vector = measure_typed_vector(column.validity, column.values, column.ty, view.rows)?;
        vector_bytes = vector_bytes
            .checked_add(vector)
            .ok_or_else(|| image_error("vector bytes overflow"))?;
        decoded_owned_bytes = decoded_owned_bytes
            .checked_add(names)
            .ok_or_else(|| image_error("decoded name owner bytes overflow"))?
            .checked_add(decoded_column_owned_bytes(column.validity, column.values)?)
            .ok_or_else(|| image_error("decoded image owner bytes overflow"))?;
        decoded_persistent_allocation_slots = decoded_persistent_allocation_slots
            .checked_add(if names == 0 { 0 } else { 1 })
            .and_then(|slots| {
                decoded_column_allocation_slots(column.validity, column.values)
                    .and_then(|column_slots| slots.checked_add(column_slots))
            })
            .ok_or_else(|| image_error("decoded allocation slot count overflows"))?;
    }
    let encoded_bytes = TYPED_IMAGE_HEADER_BYTES
        .checked_add(descriptor_bytes)
        .and_then(|bytes| bytes.checked_add(name_bytes))
        .and_then(|bytes| bytes.checked_add(vector_bytes))
        .ok_or_else(|| image_error("image bytes overflow"))?;
    Ok(TypedImageMeasure {
        encoded_bytes,
        descriptor_bytes,
        name_bytes,
        vector_bytes,
        decoded_owned_bytes,
        encode_maximum_scratch_bytes: u64::from(columns)
            .checked_mul(
                u64::try_from(std::mem::size_of::<ImageDescriptor>())
                    .map_err(|_| image_error("descriptor owner size addressability"))?,
            )
            .ok_or_else(|| image_error("encode descriptor scratch bytes overflow"))?,
        decode_maximum_scratch_bytes: u64::from(columns)
            .checked_mul(
                u64::try_from(std::mem::size_of::<ImageDescriptor>())
                    .map_err(|_| image_error("descriptor owner size addressability"))?,
            )
            .ok_or_else(|| image_error("decode descriptor scratch bytes overflow"))?,
        encoded_allocation_slots: 1,
        decoded_persistent_allocation_slots,
        encode_maximum_scratch_allocation_slots: if columns == 0 { 0 } else { 1 },
        decode_maximum_scratch_allocation_slots: if columns == 0 { 0 } else { 1 },
    })
}

/// Encode a prevalidated v2 image into one exactly reserved buffer.  The output is an inert
/// codec artifact; no caller can route it to WAL or a result path yet.
pub(crate) fn encode_typed_image(view: &TypedImageView<'_>) -> Result<Vec<u8>, EngineError> {
    let measure = measure_typed_image(view)?;
    let capacity = usize::try_from(measure.encoded_bytes)
        .map_err(|_| image_error("image bytes exceed addressability"))?;
    let columns = checked_u32_len(view.columns.len(), "image column count")?;
    let descriptors = build_descriptors(view, measure, columns)?;
    let layout_digest = layout_digest(view.role, view.rows, columns, &descriptors, view.columns)?;
    let mut out = Vec::new();
    out.try_reserve_exact(capacity)
        .map_err(|_| image_error("image output reservation failed"))?;
    append_image_header(
        &mut out,
        view.role,
        view.rows,
        columns,
        measure,
        layout_digest,
    )?;
    for descriptor in &descriptors {
        append_descriptor(&mut out, descriptor)?;
    }
    for column in view.columns {
        if view.role == TypedImageRole::RetainedResponse {
            out.extend_from_slice(column.name.as_bytes());
        }
    }
    for (column, descriptor) in view.columns.iter().zip(&descriptors) {
        let before = out.len();
        append_typed_vector(
            &mut VecSink(&mut out),
            column.validity,
            column.values,
            column.ty,
            view.rows,
        )?;
        let written = out
            .len()
            .checked_sub(before)
            .ok_or_else(|| image_error("vector length underflow"))?;
        let expected = usize::try_from(descriptor.vector_len)
            .map_err(|_| image_error("vector length exceeds addressability"))?;
        if written != expected {
            return Err(image_error("vector output length drifted from measure"));
        }
    }
    if out.len() != capacity {
        return Err(image_error("image output length drifted from measure"));
    }
    Ok(out)
}

/// Allocation-free raw-pass sizing for a strict typed-image decode.  The named child owns the
/// reservation/injection mechanics; this facade keeps the grammar authority local.
#[allow(dead_code)] // Consumed by the inert semantics-v2 S7 owner, never a live result path.
pub(crate) fn measure_decoded_typed_image(
    bytes: &[u8],
) -> Result<TypedImageDecodeMeasure, EngineError> {
    read_at::measure_decoded_typed_image_from_source(&read_at::SliceImageSource::new(bytes))
}

/// Allocation-free raw-pass sizing through a borrowed random-access image source.  This is the
/// sole strict image grammar: the contiguous-byte entry point above is only its slice adapter.
/// In particular, callers with chunked records must complete this pass before reserving or
/// copying a full image body.
#[allow(dead_code)] // Consumed by the inert semantics-v2 S7 owner, never a live result path.
pub(crate) fn measure_decoded_typed_image_from_source<S: TypedImageReadAt + ?Sized>(
    source: &S,
) -> Result<TypedImageDecodeMeasure, EngineError> {
    read_at::measure_decoded_typed_image_from_source(source)
}

/// Copy an image only after a successful source-backed raw pass.  The caller owns the exact
/// reservation (and any aggregate scratch accounting); this helper neither grows nor allocates.
#[allow(dead_code)] // Narrow S7 recovery helper.
pub(crate) fn copy_typed_image_after_measure<S: TypedImageReadAt + ?Sized>(
    source: &S,
    measure: TypedImageDecodeMeasure,
    destination: &mut [u8],
) -> Result<(), EngineError> {
    read_at::copy_typed_image_after_measure(source, measure, destination)
}

/// Strict v2 decoder.  It first completes [`measure_decoded_typed_image`], then makes only the
/// measured, exact fallible reservations.  Any failure drops the complete partial graph before
/// this function returns, so the same bytes may be retried immediately.
#[allow(dead_code)] // Future inert codec-5 S7/S8 reader.
pub(crate) fn decode_typed_image(bytes: &[u8]) -> Result<DecodedTypedImage, EngineError> {
    let measure = measure_decoded_typed_image(bytes)?;
    decode_typed_image_after_measure(bytes, measure)
}

/// Decode after a caller has completed the allocation-free pass.  The opaque measure is checked
/// again against the bytes so a stale measure cannot authorize a different raw image.
#[allow(dead_code)] // Future inert codec-5 S7 reader supplies the raw-pass measure.
pub(crate) fn decode_typed_image_after_measure(
    bytes: &[u8],
    measure: TypedImageDecodeMeasure,
) -> Result<DecodedTypedImage, EngineError> {
    decode_reservation::decode_typed_image_after_measure(bytes, measure)
}

pub(super) fn decode_typed_validity(
    bytes: &[u8],
    rows: u32,
) -> Result<(TypedInsertColumnValidity, usize), EngineError> {
    let rows_usize = usize::try_from(rows).map_err(|_| image_error("row count addressability"))?;
    let form = *bytes
        .first()
        .ok_or_else(|| image_error("validity form is truncated"))?;
    match form {
        0 => Ok((TypedInsertColumnValidity::AllValid, 1)),
        1 => {
            let used = measure_typed_validity(bytes, rows)?;
            let expected = bitmap_words(rows_usize)?;
            let raw = bytes
                .get(5..used)
                .ok_or_else(|| image_error("bitmap is truncated before allocation"))?;
            let mut words = Vec::new();
            reserve_decode_exact(
                &mut words,
                expected,
                sized_owner_bytes::<u32>(expected)?,
                0,
                "decoded validity bitmap",
            )?;
            for chunk in raw.chunks_exact(4) {
                words.push(u32::from_le_bytes(
                    chunk.try_into().expect("exact bitmap word"),
                ));
            }
            Ok((
                TypedInsertColumnValidity::Bitmap(into_exact_boxed_slice(
                    words,
                    "decoded validity bitmap",
                )?),
                used,
            ))
        }
        _ => Err(image_error("validity form tag is unknown")),
    }
}

pub(super) fn decode_typed_values(
    bytes: &[u8],
    ty: SqlType,
    rows: u32,
) -> Result<(TypedInsertColumnValues, usize), EngineError> {
    let rows_usize = usize::try_from(rows).map_err(|_| image_error("row count addressability"))?;
    let shape = *bytes
        .first()
        .ok_or_else(|| image_error("vector header is truncated"))?;
    if read_u32(bytes, 1)? != rows {
        return Err(image_error("vector logical count is not exact"));
    }
    let payload = usize::try_from(read_u32(bytes, 5)?)
        .map_err(|_| image_error("vector payload exceeds addressability"))?;
    let total = measure_typed_values(bytes, ty, rows)?;
    let body = bytes
        .get(9..total)
        .ok_or_else(|| image_error("vector payload is truncated before allocation"))?;
    let values = match (shape, ty) {
        (1, SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
            exact_fixed_payload(payload, rows_usize, 4, "i32")?;
            validate_i32_body(body, ty)?;
            let mut values = Vec::new();
            reserve_decode_exact(
                &mut values,
                rows_usize,
                sized_owner_bytes::<i32>(rows_usize)?,
                0,
                "decoded i32 values",
            )?;
            for chunk in body.chunks_exact(4) {
                values.push(i32::from_le_bytes(chunk.try_into().expect("exact i32")));
            }
            TypedInsertColumnValues::I32(into_exact_boxed_slice(values, "decoded i32 values")?)
        }
        (2, SqlType::Int8 | SqlType::Timestamp) => {
            exact_fixed_payload(payload, rows_usize, 8, "i64")?;
            validate_i64_body(body, ty)?;
            let mut values = Vec::new();
            reserve_decode_exact(
                &mut values,
                rows_usize,
                sized_owner_bytes::<i64>(rows_usize)?,
                0,
                "decoded i64 values",
            )?;
            for chunk in body.chunks_exact(8) {
                values.push(i64::from_le_bytes(chunk.try_into().expect("exact i64")));
            }
            TypedInsertColumnValues::I64(into_exact_boxed_slice(values, "decoded i64 values")?)
        }
        (3, ty @ SqlType::Numeric { .. }) => {
            exact_fixed_payload(payload, rows_usize, 16, "numeric")?;
            validate_i128_body(body, ty)?;
            let mut values = Vec::new();
            reserve_decode_exact(
                &mut values,
                rows_usize,
                sized_owner_bytes::<i128>(rows_usize)?,
                0,
                "decoded numeric values",
            )?;
            for chunk in body.chunks_exact(16) {
                values.push(i128::from_le_bytes(chunk.try_into().expect("exact i128")));
            }
            TypedInsertColumnValues::I128(into_exact_boxed_slice(values, "decoded numeric values")?)
        }
        (4, SqlType::Uuid) => {
            exact_fixed_payload(payload, rows_usize, 16, "uuid")?;
            let mut values = Vec::new();
            reserve_decode_exact(
                &mut values,
                rows_usize,
                sized_owner_bytes::<[u8; 16]>(rows_usize)?,
                0,
                "decoded UUID values",
            )?;
            for chunk in body.chunks_exact(16) {
                values.push(chunk.try_into().expect("exact UUID"));
            }
            TypedInsertColumnValues::Bytes16(into_exact_boxed_slice(values, "decoded UUID values")?)
        }
        (5, SqlType::Bool) => {
            let expected_words = bitmap_words(rows_usize)?;
            let expected = 4_usize
                .checked_add(
                    expected_words
                        .checked_mul(4)
                        .ok_or_else(|| image_error("bool bytes overflow"))?,
                )
                .ok_or_else(|| image_error("bool payload overflow"))?;
            if payload != expected
                || read_u32(body, 0)?
                    != u32::try_from(expected_words)
                        .map_err(|_| image_error("bool word count overflow"))?
            {
                return Err(image_error("bool payload/word count is not exact"));
            }
            let raw = body
                .get(4..)
                .ok_or_else(|| image_error("bool bitmap is truncated"))?;
            if !bitmap_raw_is_canonical(raw, rows_usize, false)? {
                return Err(image_error("bool bitmap tail is noncanonical"));
            }
            let mut words = Vec::new();
            reserve_decode_exact(
                &mut words,
                expected_words,
                sized_owner_bytes::<u32>(expected_words)?,
                0,
                "decoded bool bitmap",
            )?;
            for chunk in raw.chunks_exact(4) {
                words.push(u32::from_le_bytes(
                    chunk.try_into().expect("exact bool word"),
                ));
            }
            TypedInsertColumnValues::BoolBits(into_exact_boxed_slice(words, "decoded bool bitmap")?)
        }
        (6, SqlType::Text) => decode_text_values(body, payload, rows_usize)?,
        _ => return Err(image_error("vector shape tag does not match SQL type")),
    };
    Ok((values, total))
}

/// Allocation-free vector grammar validation used for directory validation before the decoder
/// allocates any column owner.  It deliberately receives only borrowed bytes and returns an
/// exact consumed length.
fn measure_typed_validity(bytes: &[u8], rows: u32) -> Result<usize, EngineError> {
    let rows = usize::try_from(rows).map_err(|_| image_error("row count addressability"))?;
    match *bytes
        .first()
        .ok_or_else(|| image_error("validity form is truncated"))?
    {
        0 => Ok(1),
        1 => {
            let expected = bitmap_words(rows)?;
            if read_u32(bytes, 1)?
                != u32::try_from(expected).map_err(|_| image_error("bitmap word count overflow"))?
            {
                return Err(image_error("bitmap word count is not exact"));
            }
            let used = 5_usize
                .checked_add(
                    expected
                        .checked_mul(4)
                        .ok_or_else(|| image_error("bitmap bytes overflow"))?,
                )
                .ok_or_else(|| image_error("bitmap length overflow"))?;
            let raw = bytes
                .get(5..used)
                .ok_or_else(|| image_error("bitmap is truncated before allocation"))?;
            if !bitmap_raw_is_canonical(raw, rows, true)? {
                return Err(image_error("validity bitmap is noncanonical"));
            }
            Ok(used)
        }
        _ => Err(image_error("validity form tag is unknown")),
    }
}

fn measure_typed_values(bytes: &[u8], ty: SqlType, rows: u32) -> Result<usize, EngineError> {
    let rows = usize::try_from(rows).map_err(|_| image_error("row count addressability"))?;
    let shape = *bytes
        .first()
        .ok_or_else(|| image_error("vector header is truncated"))?;
    if read_u32(bytes, 1)?
        != u32::try_from(rows).map_err(|_| image_error("row count exceeds u32"))?
    {
        return Err(image_error("vector logical count is not exact"));
    }
    let payload = usize::try_from(read_u32(bytes, 5)?)
        .map_err(|_| image_error("vector payload exceeds addressability"))?;
    let total = 9_usize
        .checked_add(payload)
        .ok_or_else(|| image_error("vector body length overflow"))?;
    let body = bytes
        .get(9..total)
        .ok_or_else(|| image_error("vector payload is truncated before allocation"))?;
    match (shape, ty) {
        (1, SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
            exact_fixed_payload(payload, rows, 4, "i32")?;
            validate_i32_body(body, ty)?;
        }
        (2, SqlType::Int8 | SqlType::Timestamp) => {
            exact_fixed_payload(payload, rows, 8, "i64")?;
            validate_i64_body(body, ty)?;
        }
        (3, ty @ SqlType::Numeric { .. }) => {
            exact_fixed_payload(payload, rows, 16, "numeric")?;
            validate_i128_body(body, ty)?;
        }
        (4, SqlType::Uuid) => exact_fixed_payload(payload, rows, 16, "uuid")?,
        (5, SqlType::Bool) => {
            let words = bitmap_words(rows)?;
            let expected = 4_usize
                .checked_add(
                    words
                        .checked_mul(4)
                        .ok_or_else(|| image_error("bool bytes overflow"))?,
                )
                .ok_or_else(|| image_error("bool payload overflow"))?;
            if payload != expected
                || read_u32(body, 0)?
                    != u32::try_from(words).map_err(|_| image_error("bool word count overflow"))?
                || !bitmap_raw_is_canonical(
                    body.get(4..)
                        .ok_or_else(|| image_error("bool bitmap is truncated"))?,
                    rows,
                    false,
                )?
            {
                return Err(image_error("bool payload/bitmap is noncanonical"));
            }
        }
        (6, SqlType::Text) => {
            let offsets = usize::try_from(read_u32(body, 0)?)
                .map_err(|_| image_error("text offset count exceeds addressability"))?;
            if offsets
                != rows
                    .checked_add(1)
                    .ok_or_else(|| image_error("text offset count overflow"))?
            {
                return Err(image_error("text offset count is not exact"));
            }
            let offset_bytes = offsets
                .checked_mul(8)
                .ok_or_else(|| image_error("text offsets bytes overflow"))?;
            let byte_len_at = 4_usize
                .checked_add(offset_bytes)
                .ok_or_else(|| image_error("text header overflow"))?;
            let text_at = byte_len_at
                .checked_add(4)
                .ok_or_else(|| image_error("text data offset overflow"))?;
            let text_len = usize::try_from(read_u32(body, byte_len_at)?)
                .map_err(|_| image_error("text bytes exceed addressability"))?;
            if payload
                != text_at
                    .checked_add(text_len)
                    .ok_or_else(|| image_error("text payload overflow"))?
            {
                return Err(image_error("text payload length is not exact"));
            }
            validate_text_raw(
                body.get(4..byte_len_at)
                    .ok_or_else(|| image_error("text offsets are truncated"))?,
                body.get(text_at..)
                    .ok_or_else(|| image_error("text bytes are truncated"))?,
                rows,
            )?;
        }
        _ => return Err(image_error("vector shape tag does not match SQL type")),
    }
    Ok(total)
}

pub(super) fn sized_owner_bytes<T>(count: usize) -> Result<u64, EngineError> {
    u64::try_from(count)
        .map_err(|_| image_error("decoded owner count addressability"))?
        .checked_mul(
            u64::try_from(std::mem::size_of::<T>())
                .map_err(|_| image_error("decoded owner element size addressability"))?,
        )
        .ok_or_else(|| image_error("decoded owner bytes overflow"))
}

fn append_typed_vector<S: TypedVectorSink>(
    out: &mut S,
    validity: &TypedInsertColumnValidity,
    values: &TypedInsertColumnValues,
    ty: SqlType,
    rows: u32,
) -> Result<(), EngineError> {
    append_typed_validity(out, validity, rows)?;
    append_typed_values(out, values, ty, rows)
}

fn measure_typed_vector(
    validity: &TypedInsertColumnValidity,
    values: &TypedInsertColumnValues,
    ty: SqlType,
    rows: u32,
) -> Result<u64, EngineError> {
    let mut count = CountingSink::default();
    append_typed_vector(&mut count, validity, values, ty, rows)?;
    Ok(count.len)
}

fn typed_vector_digest(
    validity: &TypedInsertColumnValidity,
    values: &TypedInsertColumnValues,
    ty: SqlType,
    rows: u32,
) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
    let mut digest = DomainDigest::new(TYPED_VECTOR_DIGEST_DOMAIN);
    digest.bytes(&sql_type_bytes(ty)?);
    digest.bytes(&rows.to_le_bytes());
    append_typed_vector(&mut digest, validity, values, ty, rows)?;
    Ok(digest.finish())
}

fn build_descriptors(
    view: &TypedImageView<'_>,
    measure: TypedImageMeasure,
    columns: u32,
) -> Result<Vec<ImageDescriptor>, EngineError> {
    let count =
        usize::try_from(columns).map_err(|_| image_error("image column count addressability"))?;
    let mut descriptors = Vec::new();
    descriptors
        .try_reserve_exact(count)
        .map_err(|_| image_error("descriptor reservation failed"))?;
    let mut name_offset = TYPED_IMAGE_HEADER_BYTES
        .checked_add(measure.descriptor_bytes)
        .ok_or_else(|| image_error("name offset overflow"))?;
    let mut vector_offset = name_offset
        .checked_add(measure.name_bytes)
        .ok_or_else(|| image_error("vector offset overflow"))?;
    for (ordinal, column) in view.columns.iter().enumerate() {
        let ordinal = u32::try_from(ordinal).map_err(|_| image_error("column ordinal overflow"))?;
        let name_len = name_len(view.role, column)?;
        let vector_len =
            measure_typed_vector(column.validity, column.values, column.ty, view.rows)?;
        let descriptor = ImageDescriptor {
            ordinal,
            catalog_column_ordinal: column.catalog_column_ordinal,
            stable_column_id: column.stable_column_id,
            table_ref: column.table_ref,
            attnum: column.attnum,
            ty: column.ty,
            type_oid: column.type_oid,
            type_size: column.type_size,
            result_format: column.result_format,
            name_offset: if name_len == 0 { 0 } else { name_offset },
            name_len,
            vector_offset,
            vector_len,
            vector_digest: typed_vector_digest(
                column.validity,
                column.values,
                column.ty,
                view.rows,
            )?,
        };
        name_offset = name_offset
            .checked_add(name_len)
            .ok_or_else(|| image_error("name offset overflow"))?;
        vector_offset = vector_offset
            .checked_add(vector_len)
            .ok_or_else(|| image_error("vector offset overflow"))?;
        descriptors.push(descriptor);
    }
    Ok(descriptors)
}

fn append_image_header(
    out: &mut Vec<u8>,
    role: TypedImageRole,
    rows: u32,
    columns: u32,
    measure: TypedImageMeasure,
    layout_digest: gpu_db_wal::CanonicalDigest,
) -> Result<(), EngineError> {
    out.extend_from_slice(&IMAGE_MAGIC);
    out.extend_from_slice(&IMAGE_VERSION.to_le_bytes());
    out.extend_from_slice(&(TYPED_IMAGE_HEADER_BYTES as u16).to_le_bytes());
    out.extend_from_slice(&role.bits().to_le_bytes());
    out.extend_from_slice(&rows.to_le_bytes());
    out.extend_from_slice(&columns.to_le_bytes());
    out.extend_from_slice(
        &u64::from(rows)
            .checked_mul(u64::from(columns))
            .ok_or_else(|| image_error("image cell count overflows"))?
            .to_le_bytes(),
    );
    out.extend_from_slice(&measure.descriptor_bytes.to_le_bytes());
    out.extend_from_slice(&measure.name_bytes.to_le_bytes());
    out.extend_from_slice(&measure.vector_bytes.to_le_bytes());
    out.extend_from_slice(&layout_digest);
    out.extend_from_slice(&[0; 16]);
    Ok(())
}

#[derive(Clone)]
struct ImageDescriptor {
    ordinal: u32,
    catalog_column_ordinal: u32,
    stable_column_id: u32,
    table_ref: u32,
    attnum: i16,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
    result_format: u16,
    name_offset: u64,
    name_len: u64,
    vector_offset: u64,
    vector_len: u64,
    vector_digest: gpu_db_wal::CanonicalDigest,
}

fn append_descriptor(out: &mut Vec<u8>, descriptor: &ImageDescriptor) -> Result<(), EngineError> {
    let before = out.len();
    out.extend_from_slice(&descriptor.ordinal.to_le_bytes());
    out.extend_from_slice(&descriptor.catalog_column_ordinal.to_le_bytes());
    out.extend_from_slice(&descriptor.stable_column_id.to_le_bytes());
    out.extend_from_slice(&descriptor.table_ref.to_le_bytes());
    out.extend_from_slice(&descriptor.attnum.to_le_bytes());
    out.extend_from_slice(&0_u16.to_le_bytes());
    out.extend_from_slice(&sql_type_bytes(descriptor.ty)?);
    out.extend_from_slice(&descriptor.type_oid.to_le_bytes());
    out.extend_from_slice(&descriptor.type_size.to_le_bytes());
    out.extend_from_slice(&descriptor.result_format.to_le_bytes());
    out.extend_from_slice(&descriptor.name_offset.to_le_bytes());
    out.extend_from_slice(&descriptor.name_len.to_le_bytes());
    out.extend_from_slice(&descriptor.vector_offset.to_le_bytes());
    out.extend_from_slice(&descriptor.vector_len.to_le_bytes());
    out.extend_from_slice(&descriptor.vector_digest);
    if out.len().checked_sub(before) != Some(TYPED_IMAGE_DESCRIPTOR_BYTES as usize) {
        return Err(image_error("descriptor width drifted"));
    }
    Ok(())
}

fn layout_digest(
    role: TypedImageRole,
    rows: u32,
    columns: u32,
    descriptors: &[ImageDescriptor],
    views: &[TypedImageColumnView<'_>],
) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
    let mut digest = DomainDigest::new(TYPED_IMAGE_LAYOUT_DIGEST_DOMAIN);
    digest.bytes(&rows.to_le_bytes());
    digest.bytes(&columns.to_le_bytes());
    digest.bytes(
        &u64::from(rows)
            .checked_mul(u64::from(columns))
            .ok_or_else(|| image_error("image cell count overflows"))?
            .to_le_bytes(),
    );
    for descriptor in descriptors {
        append_normalized_descriptor(&mut digest, descriptor)?;
    }
    if role == TypedImageRole::RetainedResponse {
        for view in views {
            digest.bytes(view.name.as_bytes());
        }
    }
    Ok(digest.finish())
}

fn append_normalized_descriptor<S: TypedVectorSink>(
    out: &mut S,
    descriptor: &ImageDescriptor,
) -> Result<(), EngineError> {
    append_u32(out, descriptor.ordinal)?;
    append_u32(out, descriptor.catalog_column_ordinal)?;
    append_u32(out, descriptor.stable_column_id)?;
    append_u32(out, descriptor.table_ref)?;
    append_i16(out, descriptor.attnum)?;
    append_u16(out, 0)?;
    out.vector_bytes(&sql_type_bytes(descriptor.ty)?)?;
    append_u32(out, descriptor.type_oid)?;
    append_i16(out, descriptor.type_size)?;
    append_u16(out, descriptor.result_format)?;
    append_u64(out, descriptor.name_offset)?;
    append_u64(out, descriptor.name_len)?;
    append_u64(out, 0)?;
    append_u64(out, 0)?;
    out.vector_bytes(&[0; 32])
}

fn parse_header(bytes: &[u8]) -> Result<ImageHeader, EngineError> {
    let header = bytes
        .get(..TYPED_IMAGE_HEADER_BYTES as usize)
        .ok_or_else(|| image_error("image header is truncated"))?;
    if header[..16] != IMAGE_MAGIC
        || read_u16(header, 16)? != IMAGE_VERSION
        || read_u16(header, 18)? != TYPED_IMAGE_HEADER_BYTES as u16
        || header[96..112].iter().any(|byte| *byte != 0)
    {
        return Err(image_error(
            "image magic/version/header/reserved is noncanonical",
        ));
    }
    let role = TypedImageRole::decode(read_u32(header, 20)?)?;
    let rows = read_u32(header, 24)?;
    let columns = read_u32(header, 28)?;
    let cells = read_u64(header, 32)?;
    if cells
        != u64::from(rows)
            .checked_mul(u64::from(columns))
            .ok_or_else(|| image_error("image cell count overflows"))?
    {
        return Err(image_error("image cell count is not exact"));
    }
    let descriptor_bytes = read_u64(header, 40)?;
    if descriptor_bytes
        != u64::from(columns)
            .checked_mul(TYPED_IMAGE_DESCRIPTOR_BYTES)
            .ok_or_else(|| image_error("descriptor bytes overflow"))?
    {
        return Err(image_error("descriptor bytes are not exact"));
    }
    Ok(ImageHeader {
        role,
        rows,
        columns,
        descriptor_bytes,
        name_bytes: read_u64(header, 48)?,
        vector_bytes: read_u64(header, 56)?,
        layout_digest: header[64..96].try_into().expect("fixed layout digest"),
    })
}

#[derive(Clone, Copy)]
struct ImageHeader {
    role: TypedImageRole,
    rows: u32,
    columns: u32,
    descriptor_bytes: u64,
    name_bytes: u64,
    vector_bytes: u64,
    layout_digest: gpu_db_wal::CanonicalDigest,
}

fn checked_total_len(columns: u32, name_bytes: u64, vector_bytes: u64) -> Result<u64, EngineError> {
    TYPED_IMAGE_HEADER_BYTES
        .checked_add(
            u64::from(columns)
                .checked_mul(TYPED_IMAGE_DESCRIPTOR_BYTES)
                .ok_or_else(|| image_error("descriptor bytes overflow"))?,
        )
        .and_then(|bytes| bytes.checked_add(name_bytes))
        .and_then(|bytes| bytes.checked_add(vector_bytes))
        .ok_or_else(|| image_error("image total length overflow"))
}

struct ImageDecodeLayout<'a> {
    region: &'a [u8],
    role: TypedImageRole,
    rows: u32,
    columns: usize,
    name_start: usize,
    name_len: usize,
    vector_start: usize,
    vector_len: u64,
    layout_digest: gpu_db_wal::CanonicalDigest,
    image: &'a [u8],
}

fn decoded_image_layout(bytes: &[u8]) -> Result<(ImageHeader, ImageDecodeLayout<'_>), EngineError> {
    let header = parse_header(bytes)?;
    let total = checked_total_len(header.columns, header.name_bytes, header.vector_bytes)?;
    if total != bytes.len() as u64 {
        return Err(image_error("image length is not exact"));
    }
    let columns = usize::try_from(header.columns)
        .map_err(|_| image_error("image column count exceeds addressability"))?;
    let descriptors_start = usize::try_from(TYPED_IMAGE_HEADER_BYTES)
        .map_err(|_| image_error("header addressability"))?;
    let descriptors_len = usize::try_from(header.descriptor_bytes)
        .map_err(|_| image_error("descriptor bytes exceed addressability"))?;
    let descriptor_end = descriptors_start
        .checked_add(descriptors_len)
        .ok_or_else(|| image_error("descriptor boundary overflow"))?;
    let descriptor_region = bytes
        .get(descriptors_start..descriptor_end)
        .ok_or_else(|| image_error("descriptor region is truncated"))?;
    let name_start = descriptor_end;
    let name_len = usize::try_from(header.name_bytes)
        .map_err(|_| image_error("name bytes exceed addressability"))?;
    let vector_start = name_start
        .checked_add(name_len)
        .ok_or_else(|| image_error("vector boundary overflow"))?;
    if vector_start > bytes.len() {
        return Err(image_error("name region is truncated"));
    }
    Ok((
        header,
        ImageDecodeLayout {
            region: descriptor_region,
            role: header.role,
            rows: header.rows,
            columns,
            name_start,
            name_len,
            vector_start,
            vector_len: header.vector_bytes,
            layout_digest: header.layout_digest,
            image: bytes,
        },
    ))
}

/// The raw pass has already verified directory geometry and every nested body.  This second pass
/// reserves exactly the measured transient descriptor directory and materializes its fixed
/// entries for the owned decode; it performs no attacker-sized implicit reservation.
fn parse_descriptors_after_measure(
    layout: &ImageDecodeLayout<'_>,
) -> Result<Vec<ImageDescriptor>, EngineError> {
    if layout.region.len()
        != layout
            .columns
            .checked_mul(TYPED_IMAGE_DESCRIPTOR_BYTES as usize)
            .ok_or_else(|| image_error("descriptor region overflow"))?
    {
        return Err(image_error("descriptor region is not exact"));
    }
    let mut result = Vec::new();
    reserve_decode_exact(
        &mut result,
        layout.columns,
        0,
        sized_owner_bytes::<ImageDescriptor>(layout.columns)?,
        "decoded descriptor scratch",
    )?;
    for ordinal in 0..layout.columns {
        result.push(raw_descriptor(layout.region, ordinal)?);
    }
    Ok(result)
}

fn raw_descriptor(region: &[u8], ordinal: usize) -> Result<ImageDescriptor, EngineError> {
    let base = ordinal
        .checked_mul(TYPED_IMAGE_DESCRIPTOR_BYTES as usize)
        .ok_or_else(|| image_error("descriptor offset overflow"))?;
    let raw = region
        .get(base..base + TYPED_IMAGE_DESCRIPTOR_BYTES as usize)
        .ok_or_else(|| image_error("descriptor is truncated"))?;
    parse_descriptor_entry(raw, ordinal)
}

/// Parse one fixed descriptor entry after its caller has isolated the exact 96-byte range.  Both
/// the contiguous adapter and the chunked `read_at` grammar use this one entry authority.
fn parse_descriptor_entry(raw: &[u8], ordinal: usize) -> Result<ImageDescriptor, EngineError> {
    if raw.len() != TYPED_IMAGE_DESCRIPTOR_BYTES as usize {
        return Err(image_error("descriptor is not exact"));
    }
    let expected =
        u32::try_from(ordinal).map_err(|_| image_error("descriptor ordinal overflow"))?;
    if read_u32(raw, 0)? != expected || read_u16(raw, 18)? != 0 {
        return Err(image_error("descriptor ordinal/flags are noncanonical"));
    }
    Ok(ImageDescriptor {
        ordinal: expected,
        catalog_column_ordinal: read_u32(raw, 4)?,
        stable_column_id: read_u32(raw, 8)?,
        table_ref: read_u32(raw, 12)?,
        attnum: read_i16(raw, 16)?,
        ty: sql_type_from_bytes(
            raw.get(20..24)
                .ok_or_else(|| image_error("descriptor SQL type is truncated"))?,
        )?,
        type_oid: read_u32(raw, 24)?,
        type_size: read_i16(raw, 28)?,
        result_format: read_u16(raw, 30)?,
        name_offset: read_u64(raw, 32)?,
        name_len: read_u64(raw, 40)?,
        vector_offset: read_u64(raw, 48)?,
        vector_len: read_u64(raw, 56)?,
        vector_digest: raw[64..96].try_into().expect("fixed vector digest"),
    })
}

fn validate_column_view(
    role: TypedImageRole,
    ordinal: usize,
    column: &TypedImageColumnView<'_>,
    rows: u32,
) -> Result<(), EngineError> {
    if role == TypedImageRole::RetainedResponse && column.name.as_bytes().contains(&0) {
        return Err(image_error("response projection name contains NUL"));
    }
    let identity = ImageDescriptor {
        ordinal: u32::try_from(ordinal).map_err(|_| image_error("column ordinal overflow"))?,
        catalog_column_ordinal: column.catalog_column_ordinal,
        stable_column_id: column.stable_column_id,
        table_ref: column.table_ref,
        attnum: column.attnum,
        ty: column.ty,
        type_oid: column.type_oid,
        type_size: column.type_size,
        result_format: column.result_format,
        name_offset: 0,
        name_len: u64::try_from(column.name.len())
            .map_err(|_| image_error("name length addressability"))?,
        vector_offset: 0,
        vector_len: 0,
        vector_digest: [0; 32],
    };
    validate_column_identity(role, &identity)?;
    validate_invalid_placeholders(column.validity, column.values, rows)?;
    Ok(())
}

fn validate_decoded_descriptor(
    role: TypedImageRole,
    descriptor: &ImageDescriptor,
) -> Result<(), EngineError> {
    validate_column_identity(role, descriptor)
}

fn validate_column_identity(
    role: TypedImageRole,
    descriptor: &ImageDescriptor,
) -> Result<(), EngineError> {
    if descriptor.type_oid == 0
        || descriptor.type_size != descriptor.ty.type_size()
        || descriptor.result_format > 1
    {
        return Err(image_error("descriptor type/result format is invalid"));
    }
    let is_derived = descriptor.catalog_column_ordinal == DERIVED_U32
        || descriptor.stable_column_id == 0
        || descriptor.table_ref == DERIVED_U32
        || descriptor.attnum == DERIVED_ATNUM;
    match role {
        TypedImageRole::FinalTableImage => {
            if is_derived
                || descriptor.result_format != 0
                || descriptor.name_len != 0
                || descriptor.catalog_column_ordinal != descriptor.ordinal
            {
                return Err(image_error(
                    "final-table descriptor identity/name/format is invalid",
                ));
            }
        }
        TypedImageRole::RetainedResponse => {
            if descriptor.name_len == 0 {
                return Err(image_error("response descriptor name is empty"));
            }
            if is_derived
                && !(descriptor.catalog_column_ordinal == DERIVED_U32
                    && descriptor.stable_column_id == 0
                    && descriptor.table_ref == DERIVED_U32
                    && descriptor.attnum == DERIVED_ATNUM)
            {
                return Err(image_error(
                    "derived response descriptor must use all sentinels",
                ));
            }
        }
    }
    Ok(())
}

fn name_len(role: TypedImageRole, column: &TypedImageColumnView<'_>) -> Result<u64, EngineError> {
    match role {
        TypedImageRole::FinalTableImage => Ok(0),
        TypedImageRole::RetainedResponse => {
            u64::try_from(column.name.len()).map_err(|_| image_error("name length addressability"))
        }
    }
}

fn descriptor_name(bytes: &[u8], descriptor: &ImageDescriptor) -> Result<Box<str>, EngineError> {
    let raw = descriptor_name_bytes(bytes, descriptor)?;
    if raw.is_empty() {
        return Ok(Box::default());
    }
    let name = std::str::from_utf8(raw).map_err(|_| image_error("name is not UTF-8"))?;
    let mut owned = String::new();
    reserve_decode_string(
        &mut owned,
        name.len(),
        u64::try_from(name.len()).map_err(|_| image_error("name allocation addressability"))?,
        "decoded name",
    )?;
    owned.push_str(name);
    into_exact_boxed_str(owned, "decoded name")
}

fn descriptor_name_bytes<'a>(
    bytes: &'a [u8],
    descriptor: &ImageDescriptor,
) -> Result<&'a [u8], EngineError> {
    if descriptor.name_len == 0 {
        return Ok(&[]);
    }
    let start = usize::try_from(descriptor.name_offset)
        .map_err(|_| image_error("name offset exceeds addressability"))?;
    let len = usize::try_from(descriptor.name_len)
        .map_err(|_| image_error("name length exceeds addressability"))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| image_error("name range overflow"))?;
    let raw = bytes
        .get(start..end)
        .ok_or_else(|| image_error("name is truncated"))?;
    if raw.contains(&0) || std::str::from_utf8(raw).is_err() {
        return Err(image_error("name is not canonical UTF-8"));
    }
    Ok(raw)
}

fn vector_bytes<'a>(
    bytes: &'a [u8],
    descriptor: &ImageDescriptor,
) -> Result<&'a [u8], EngineError> {
    let start = usize::try_from(descriptor.vector_offset)
        .map_err(|_| image_error("vector offset exceeds addressability"))?;
    let len = usize::try_from(descriptor.vector_len)
        .map_err(|_| image_error("vector length exceeds addressability"))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| image_error("vector range overflow"))?;
    bytes
        .get(start..end)
        .ok_or_else(|| image_error("vector is truncated"))
}

fn decode_text_values(
    body: &[u8],
    payload: usize,
    rows: usize,
) -> Result<TypedInsertColumnValues, EngineError> {
    let offset_count = usize::try_from(read_u32(body, 0)?)
        .map_err(|_| image_error("text offset count exceeds addressability"))?;
    let expected_count = rows
        .checked_add(1)
        .ok_or_else(|| image_error("text offset count overflow"))?;
    if offset_count != expected_count {
        return Err(image_error("text offset count is not exact"));
    }
    let offsets_bytes = offset_count
        .checked_mul(8)
        .ok_or_else(|| image_error("text offsets bytes overflow"))?;
    let byte_len_offset = 4_usize
        .checked_add(offsets_bytes)
        .ok_or_else(|| image_error("text header overflow"))?;
    let data_offset = byte_len_offset
        .checked_add(4)
        .ok_or_else(|| image_error("text data offset overflow"))?;
    let byte_len = usize::try_from(read_u32(body, byte_len_offset)?)
        .map_err(|_| image_error("text byte length exceeds addressability"))?;
    let expected_payload = data_offset
        .checked_add(byte_len)
        .ok_or_else(|| image_error("text payload overflow"))?;
    if payload != expected_payload || body.len() != expected_payload {
        return Err(image_error("text payload length is not exact"));
    }
    let offsets_raw = body
        .get(4..byte_len_offset)
        .ok_or_else(|| image_error("text offsets are truncated before allocation"))?;
    let text = body
        .get(data_offset..)
        .ok_or_else(|| image_error("text bytes are truncated before allocation"))?;
    validate_text_raw(offsets_raw, text, rows)?;
    let mut offsets = Vec::new();
    reserve_decode_exact(
        &mut offsets,
        offset_count,
        sized_owner_bytes::<u64>(offset_count)?,
        0,
        "decoded text offsets",
    )?;
    for chunk in offsets_raw.chunks_exact(8) {
        offsets.push(u64::from_le_bytes(
            chunk.try_into().expect("exact text offset"),
        ));
    }
    let mut owned_bytes = Vec::new();
    reserve_decode_exact(
        &mut owned_bytes,
        text.len(),
        u64::try_from(text.len())
            .map_err(|_| image_error("text byte allocation addressability"))?,
        0,
        "decoded text bytes",
    )?;
    owned_bytes.extend_from_slice(text);
    Ok(TypedInsertColumnValues::Text {
        offsets: into_exact_boxed_slice(offsets, "decoded text offsets")?,
        bytes: into_exact_boxed_slice(owned_bytes, "decoded text bytes")?,
    })
}

fn validate_text_raw(offsets_raw: &[u8], text: &[u8], rows: usize) -> Result<(), EngineError> {
    if offsets_raw.len()
        != rows
            .checked_add(1)
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| image_error("text offsets length overflow"))?
    {
        return Err(image_error("text offsets length is not exact"));
    }
    let text = std::str::from_utf8(text).map_err(|_| image_error("text bytes are not UTF-8"))?;
    let mut previous = 0_u64;
    for (ordinal, chunk) in offsets_raw.chunks_exact(8).enumerate() {
        let offset = u64::from_le_bytes(chunk.try_into().expect("exact text offset"));
        let addressable = usize::try_from(offset)
            .map_err(|_| image_error("text offset exceeds addressability"))?;
        if (ordinal == 0 && offset != 0)
            || offset < previous
            || addressable > text.len()
            || !text.is_char_boundary(addressable)
        {
            return Err(image_error("text offsets are noncanonical"));
        }
        previous = offset;
    }
    if previous
        != u64::try_from(text.len()).map_err(|_| image_error("text length addressability"))?
    {
        return Err(image_error("text final offset is not exact"));
    }
    Ok(())
}

fn validate_i32_body(body: &[u8], ty: SqlType) -> Result<(), EngineError> {
    for chunk in body.chunks_exact(4) {
        let value = i32::from_le_bytes(chunk.try_into().expect("exact i32"));
        if (ty == SqlType::Int2 && i16::try_from(value).is_err())
            || (ty == SqlType::Date && gpu_db_sql::datetime::validate_date_carrier(value).is_err())
        {
            return Err(image_error("i32 value is outside SQL type bounds"));
        }
    }
    Ok(())
}

fn validate_i64_body(body: &[u8], ty: SqlType) -> Result<(), EngineError> {
    if ty == SqlType::Timestamp {
        for chunk in body.chunks_exact(8) {
            if gpu_db_sql::datetime::validate_timestamp_carrier(i64::from_le_bytes(
                chunk.try_into().expect("exact i64"),
            ))
            .is_err()
            {
                return Err(image_error("timestamp is outside SQL type bounds"));
            }
        }
    }
    Ok(())
}

fn validate_i128_body(body: &[u8], ty: SqlType) -> Result<(), EngineError> {
    let SqlType::Numeric { precision, .. } = ty else {
        return Err(image_error("numeric vector has nonnumeric SQL type"));
    };
    for chunk in body.chunks_exact(16) {
        if crate::numeric_exceeds_precision(
            i128::from_le_bytes(chunk.try_into().expect("exact i128")),
            precision,
        ) {
            return Err(image_error("numeric mantissa exceeds declared precision"));
        }
    }
    Ok(())
}

fn exact_fixed_payload(
    payload: usize,
    rows: usize,
    width: usize,
    kind: &str,
) -> Result<(), EngineError> {
    if payload
        != rows
            .checked_mul(width)
            .ok_or_else(|| image_error(&format!("{kind} payload overflow")))?
    {
        return Err(image_error(&format!(
            "{kind} payload length is noncanonical"
        )));
    }
    Ok(())
}

fn bitmap_raw_is_canonical(
    raw: &[u8],
    rows: usize,
    reject_all_set: bool,
) -> Result<bool, EngineError> {
    let expected_words = bitmap_words(rows)?;
    if raw.len()
        != expected_words
            .checked_mul(4)
            .ok_or_else(|| image_error("bitmap bytes overflow"))?
    {
        return Ok(false);
    }
    let mut all_set = true;
    for (ordinal, chunk) in raw.chunks_exact(4).enumerate() {
        let word = u32::from_le_bytes(chunk.try_into().expect("exact bitmap word"));
        let expected = if ordinal < rows / 32 {
            u32::MAX
        } else {
            (1_u32 << (rows % 32)) - 1
        };
        if ordinal + 1 == expected_words && !rows.is_multiple_of(32) && word & !expected != 0 {
            return Ok(false);
        }
        all_set &= word == expected;
    }
    Ok(!reject_all_set || !all_set)
}

fn validate_invalid_placeholders(
    validity: &TypedInsertColumnValidity,
    values: &TypedInsertColumnValues,
    rows: u32,
) -> Result<(), EngineError> {
    let rows = usize::try_from(rows).map_err(|_| image_error("row count addressability"))?;
    if !validity.shape_is_exact(rows) || !values.rows_match(rows) {
        return Err(image_error("validity/value rows are not exact"));
    }
    for row in 0..rows {
        if validity.is_valid(row) {
            continue;
        }
        let zero = match values {
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
            return Err(image_error("invalid typed value placeholder is nonzero"));
        }
    }
    Ok(())
}

fn append_vector_header<S: TypedVectorSink>(
    out: &mut S,
    shape: u8,
    rows: u32,
    payload: u32,
) -> Result<(), EngineError> {
    append_u8(out, shape)?;
    append_u32(out, rows)?;
    append_u32(out, payload)
}

fn append_bitmap_words<S: TypedVectorSink>(out: &mut S, words: &[u32]) -> Result<(), EngineError> {
    append_u32(out, checked_u32_len(words.len(), "bitmap word count")?)?;
    for word in words {
        append_u32(out, *word)?;
    }
    Ok(())
}

fn append_u8<S: TypedVectorSink>(out: &mut S, value: u8) -> Result<(), EngineError> {
    out.vector_bytes(&[value])
}
fn append_u16<S: TypedVectorSink>(out: &mut S, value: u16) -> Result<(), EngineError> {
    out.vector_bytes(&value.to_le_bytes())
}
fn append_i16<S: TypedVectorSink>(out: &mut S, value: i16) -> Result<(), EngineError> {
    out.vector_bytes(&value.to_le_bytes())
}
fn append_u32<S: TypedVectorSink>(out: &mut S, value: u32) -> Result<(), EngineError> {
    out.vector_bytes(&value.to_le_bytes())
}
fn append_i32<S: TypedVectorSink>(out: &mut S, value: i32) -> Result<(), EngineError> {
    out.vector_bytes(&value.to_le_bytes())
}
fn append_u64<S: TypedVectorSink>(out: &mut S, value: u64) -> Result<(), EngineError> {
    out.vector_bytes(&value.to_le_bytes())
}
fn append_i64<S: TypedVectorSink>(out: &mut S, value: i64) -> Result<(), EngineError> {
    out.vector_bytes(&value.to_le_bytes())
}
fn append_i128<S: TypedVectorSink>(out: &mut S, value: i128) -> Result<(), EngineError> {
    out.vector_bytes(&value.to_le_bytes())
}

struct VecSink<'a>(&'a mut Vec<u8>);
impl TypedVectorSink for VecSink<'_> {
    fn vector_bytes(&mut self, bytes: &[u8]) -> Result<(), EngineError> {
        self.0.extend_from_slice(bytes);
        Ok(())
    }
}

#[derive(Default)]
struct CountingSink {
    len: u64,
}
impl TypedVectorSink for CountingSink {
    fn vector_bytes(&mut self, bytes: &[u8]) -> Result<(), EngineError> {
        self.len = self
            .len
            .checked_add(
                u64::try_from(bytes.len())
                    .map_err(|_| image_error("counting length addressability"))?,
            )
            .ok_or_else(|| image_error("counting length overflow"))?;
        Ok(())
    }
}

struct DomainDigest(Sha256);
impl DomainDigest {
    fn new(domain: &[u8]) -> Self {
        let mut hash = Sha256::new();
        hash.update(
            u64::try_from(domain.len())
                .expect("static digest domain length")
                .to_le_bytes(),
        );
        hash.update(domain);
        Self(hash)
    }
    fn bytes(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
    fn finish(self) -> gpu_db_wal::CanonicalDigest {
        self.0.finalize().into()
    }
}
impl TypedVectorSink for DomainDigest {
    fn vector_bytes(&mut self, bytes: &[u8]) -> Result<(), EngineError> {
        self.bytes(bytes);
        Ok(())
    }
}

fn checked_u32_len(value: usize, what: &str) -> Result<u32, EngineError> {
    u32::try_from(value).map_err(|_| image_error(&format!("{what} exceeds u32")))
}
fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, EngineError> {
    Ok(u16::from_le_bytes(
        bytes
            .get(
                offset
                    ..offset
                        .checked_add(2)
                        .ok_or_else(|| image_error("u16 offset overflow"))?,
            )
            .ok_or_else(|| image_error("u16 is truncated"))?
            .try_into()
            .expect("exact u16"),
    ))
}
fn read_i16(bytes: &[u8], offset: usize) -> Result<i16, EngineError> {
    Ok(i16::from_le_bytes(
        bytes
            .get(
                offset
                    ..offset
                        .checked_add(2)
                        .ok_or_else(|| image_error("i16 offset overflow"))?,
            )
            .ok_or_else(|| image_error("i16 is truncated"))?
            .try_into()
            .expect("exact i16"),
    ))
}
fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, EngineError> {
    Ok(u32::from_le_bytes(
        bytes
            .get(
                offset
                    ..offset
                        .checked_add(4)
                        .ok_or_else(|| image_error("u32 offset overflow"))?,
            )
            .ok_or_else(|| image_error("u32 is truncated"))?
            .try_into()
            .expect("exact u32"),
    ))
}
fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, EngineError> {
    Ok(u64::from_le_bytes(
        bytes
            .get(
                offset
                    ..offset
                        .checked_add(8)
                        .ok_or_else(|| image_error("u64 offset overflow"))?,
            )
            .ok_or_else(|| image_error("u64 is truncated"))?
            .try_into()
            .expect("exact u64"),
    ))
}
pub(super) fn image_error(message: &str) -> EngineError {
    EngineError::Durability(format!("typed image/vector canonical codec: {message}"))
}

fn resident_source_error(message: &str) -> EngineError {
    EngineError::Durability(format!("typed image resident source: {message}"))
}
