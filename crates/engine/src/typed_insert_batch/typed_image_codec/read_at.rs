//! Allocation-free strict typed-image grammar over a borrowed random-access source.
//!
//! The S7 reader receives images embedded in chunked canonical records.  It must prove image
//! geometry and both digest layers before it reserves the one post-pass copy, so this leaf never
//! asks a source to materialize a range.  The contiguous decoder reaches the same grammar only
//! through `SliceImageSource`.

use super::*;

/// Borrowed random-access bytes for an inert typed image.  Implementors may span record chunks;
/// `read_at` must fill `out` exactly or return an error.
pub(crate) trait TypedImageReadAt {
    fn len(&self) -> u64;
    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), EngineError>;
}

pub(super) struct SliceImageSource<'a> {
    bytes: &'a [u8],
}

impl<'a> SliceImageSource<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }
}

impl TypedImageReadAt for SliceImageSource<'_> {
    fn len(&self) -> u64 {
        u64::try_from(self.bytes.len()).expect("slice length fits u64")
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), EngineError> {
        let start = usize::try_from(offset)
            .map_err(|_| image_error("image source offset exceeds addressability"))?;
        let end = start
            .checked_add(out.len())
            .ok_or_else(|| image_error("image source range overflows"))?;
        let source = self
            .bytes
            .get(start..end)
            .ok_or_else(|| image_error("image source range is truncated"))?;
        out.copy_from_slice(source);
        Ok(())
    }
}

pub(super) fn measure_decoded_typed_image_from_source<S: TypedImageReadAt + ?Sized>(
    source: &S,
) -> Result<TypedImageDecodeMeasure, EngineError> {
    let layout = source_layout(source)?;
    let mut digest = DomainDigest::new(TYPED_IMAGE_LAYOUT_DIGEST_DOMAIN);
    digest.bytes(&layout.header.rows.to_le_bytes());
    digest.bytes(&layout.header.columns.to_le_bytes());
    digest.bytes(
        &u64::from(layout.header.rows)
            .checked_mul(u64::from(layout.header.columns))
            .ok_or_else(|| image_error("image cell count overflows"))?
            .to_le_bytes(),
    );

    let mut persistent_bytes = sized_owner_bytes::<DecodedTypedImageColumn>(layout.columns)?;
    let mut persistent_slots = u64::from(layout.columns != 0);
    let mut expected_name = layout.name_start;
    let mut expected_vector = layout.vector_start;

    for ordinal in 0..layout.columns {
        let descriptor = source_descriptor(source, ordinal)?;
        validate_decoded_descriptor(layout.header.role, &descriptor)?;
        append_normalized_descriptor(&mut digest, &descriptor)?;
        if descriptor.name_len == 0 {
            if descriptor.name_offset != 0 {
                return Err(image_error("empty descriptor name has an offset"));
            }
        } else if descriptor.name_offset != expected_name {
            return Err(image_error("descriptor names have a gap or overlap"));
        }
        expected_name = expected_name
            .checked_add(descriptor.name_len)
            .ok_or_else(|| image_error("name range overflow"))?;
        if descriptor.vector_offset != expected_vector {
            return Err(image_error("descriptor vectors have a gap or overlap"));
        }
        expected_vector = expected_vector
            .checked_add(descriptor.vector_len)
            .ok_or_else(|| image_error("vector range overflow"))?;
    }

    if expected_name != layout.name_end {
        return Err(image_error("descriptor names do not fill the name region"));
    }
    if expected_vector != layout.vector_end {
        return Err(image_error(
            "descriptor vectors do not fill the vector region",
        ));
    }

    // The layout domain puts the complete fixed directory before the response-name stream.
    // Preserve that byte order even though each descriptor/name/vector can arrive in different
    // record chunks.
    for ordinal in 0..layout.columns {
        let descriptor = source_descriptor(source, ordinal)?;
        validate_name_source(
            source,
            descriptor.name_offset,
            descriptor.name_len,
            (layout.header.role == TypedImageRole::RetainedResponse).then_some(&mut digest),
        )?;
        if descriptor.name_len != 0 {
            persistent_bytes = persistent_bytes
                .checked_add(descriptor.name_len)
                .ok_or_else(|| image_error("decoded name owner bytes overflow"))?;
            persistent_slots = persistent_slots
                .checked_add(1)
                .ok_or_else(|| image_error("decoded name allocation slots overflow"))?;
        }
    }

    for ordinal in 0..layout.columns {
        let descriptor = source_descriptor(source, ordinal)?;
        let allocation = validate_vector_source(source, &layout.header, &descriptor)?;
        persistent_bytes = persistent_bytes
            .checked_add(allocation.validity_bytes)
            .and_then(|value| value.checked_add(allocation.value_bytes))
            .ok_or_else(|| image_error("decoded vector owner bytes overflow"))?;
        persistent_slots = persistent_slots
            .checked_add(allocation.validity_slots)
            .and_then(|value| value.checked_add(allocation.value_slots))
            .ok_or_else(|| image_error("decoded vector allocation slots overflow"))?;
    }

    if digest.finish() != layout.header.layout_digest {
        return Err(image_error("image layout digest drifted"));
    }

    Ok(TypedImageDecodeMeasure::from_raw_parts(
        source.len(),
        source_content_fingerprint(source)?,
        persistent_bytes,
        persistent_slots,
        sized_owner_bytes::<ImageDescriptor>(layout.columns)?,
        u64::from(layout.columns != 0),
    ))
}

pub(super) fn copy_typed_image_after_measure<S: TypedImageReadAt + ?Sized>(
    source: &S,
    measure: TypedImageDecodeMeasure,
    destination: &mut [u8],
) -> Result<(), EngineError> {
    let actual = measure_decoded_typed_image_from_source(source)?;
    if actual != measure || actual.content_fingerprint() != measure.content_fingerprint() {
        return Err(image_error("image source drifted after its raw measure"));
    }
    let expected = usize::try_from(measure.image_bytes())
        .map_err(|_| image_error("measured image length exceeds addressability"))?;
    if destination.len() != expected || source.len() != measure.image_bytes() {
        return Err(image_error("measured image source/copy length drifted"));
    }
    source.read_at(0, destination)?;
    let copied = measure_decoded_typed_image_from_source(&SliceImageSource::new(destination))?;
    if copied != measure || copied.content_fingerprint() != measure.content_fingerprint() {
        return Err(image_error("copied image drifted from its raw measure"));
    }
    Ok(())
}

/// Hash the exact borrowed image stream without materializing it.  The layout digest proves
/// canonical grammar; this independent whole-source fingerprint makes a same-length source
/// mutation visible at the raw-pass-to-copy handoff.
fn source_content_fingerprint<S: TypedImageReadAt + ?Sized>(
    source: &S,
) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
    let length = source.len();
    let mut digest = DomainDigest::new(TYPED_IMAGE_CONTENT_FINGERPRINT_DOMAIN);
    digest.bytes(&length.to_le_bytes());
    let mut offset = 0_u64;
    let mut chunk = [0_u8; 4096];
    while offset < length {
        let remaining = length
            .checked_sub(offset)
            .ok_or_else(|| image_error("image fingerprint offset underflows"))?;
        let count = usize::try_from(remaining.min(chunk.len() as u64))
            .map_err(|_| image_error("image fingerprint chunk exceeds addressability"))?;
        source_read(source, offset, &mut chunk[..count])?;
        digest.bytes(&chunk[..count]);
        offset = offset
            .checked_add(u64::try_from(count).expect("fingerprint chunk fits u64"))
            .ok_or_else(|| image_error("image fingerprint offset overflows"))?;
    }
    Ok(digest.finish())
}

struct SourceLayout {
    header: ImageHeader,
    columns: usize,
    name_start: u64,
    name_end: u64,
    vector_start: u64,
    vector_end: u64,
}

fn source_layout<S: TypedImageReadAt + ?Sized>(source: &S) -> Result<SourceLayout, EngineError> {
    let mut raw = [0_u8; TYPED_IMAGE_HEADER_BYTES as usize];
    source_read(source, 0, &mut raw)?;
    let header = parse_header(&raw)?;
    let total = checked_total_len(header.columns, header.name_bytes, header.vector_bytes)?;
    if total != source.len() {
        return Err(image_error("image length is not exact"));
    }
    let columns = usize::try_from(header.columns)
        .map_err(|_| image_error("image column count exceeds addressability"))?;
    let name_start = TYPED_IMAGE_HEADER_BYTES
        .checked_add(header.descriptor_bytes)
        .ok_or_else(|| image_error("name boundary overflows"))?;
    let name_end = name_start
        .checked_add(header.name_bytes)
        .ok_or_else(|| image_error("name boundary overflows"))?;
    let vector_end = name_end
        .checked_add(header.vector_bytes)
        .ok_or_else(|| image_error("vector boundary overflows"))?;
    Ok(SourceLayout {
        header,
        columns,
        name_start,
        name_end,
        vector_start: name_end,
        vector_end,
    })
}

fn source_descriptor<S: TypedImageReadAt + ?Sized>(
    source: &S,
    ordinal: usize,
) -> Result<ImageDescriptor, EngineError> {
    let offset = u64::try_from(ordinal)
        .map_err(|_| image_error("descriptor ordinal addressability"))?
        .checked_mul(TYPED_IMAGE_DESCRIPTOR_BYTES)
        .and_then(|value| value.checked_add(TYPED_IMAGE_HEADER_BYTES))
        .ok_or_else(|| image_error("descriptor offset overflows"))?;
    let mut raw = [0_u8; TYPED_IMAGE_DESCRIPTOR_BYTES as usize];
    source_read(source, offset, &mut raw)?;
    parse_descriptor_entry(&raw, ordinal)
}

struct VectorAllocation {
    validity_len: u64,
    validity_bytes: u64,
    validity_slots: u64,
    value_bytes: u64,
    value_slots: u64,
}

fn validate_vector_source<S: TypedImageReadAt + ?Sized>(
    source: &S,
    header: &ImageHeader,
    descriptor: &ImageDescriptor,
) -> Result<VectorAllocation, EngineError> {
    let validity = source_validity(source, descriptor.vector_offset, header.rows)?;
    let value_start = descriptor
        .vector_offset
        .checked_add(validity.validity_len)
        .ok_or_else(|| image_error("value offset overflows"))?;
    let values = source_values(source, value_start, descriptor.ty, header.rows)?;
    if validity
        .validity_len
        .checked_add(values.total_len)
        .filter(|total| *total == descriptor.vector_len)
        .is_none()
    {
        return Err(image_error("descriptor vector has trailing bytes"));
    }
    validate_source_placeholders(
        source,
        descriptor.vector_offset,
        validity.validity_len,
        values.shape,
        descriptor.ty,
        header.rows,
        values.body_offset,
    )?;
    let mut digest = DomainDigest::new(TYPED_VECTOR_DIGEST_DOMAIN);
    digest.bytes(&sql_type_bytes(descriptor.ty)?);
    digest.bytes(&header.rows.to_le_bytes());
    hash_source_range(
        source,
        descriptor.vector_offset,
        descriptor.vector_len,
        &mut digest,
    )?;
    if digest.finish() != descriptor.vector_digest {
        return Err(image_error("vector digest drifted"));
    }
    Ok(VectorAllocation {
        validity_len: validity.validity_len,
        validity_bytes: validity.bytes,
        validity_slots: validity.slots,
        value_bytes: values.bytes,
        value_slots: values.slots,
    })
}

struct ValidityAllocation {
    validity_len: u64,
    bytes: u64,
    slots: u64,
}

fn source_validity<S: TypedImageReadAt + ?Sized>(
    source: &S,
    start: u64,
    rows: u32,
) -> Result<ValidityAllocation, EngineError> {
    match source_u8(source, start)? {
        0 => Ok(ValidityAllocation {
            validity_len: 1,
            bytes: 0,
            slots: 0,
        }),
        1 => {
            let rows =
                usize::try_from(rows).map_err(|_| image_error("row count addressability"))?;
            let words = bitmap_words(rows)?;
            if source_u32(source, add(start, 1, "bitmap count offset")?)?
                != u32::try_from(words).map_err(|_| image_error("bitmap word count overflow"))?
                || !source_bitmap_is_canonical(
                    source,
                    add(start, 5, "bitmap body offset")?,
                    words,
                    rows,
                    true,
                )?
            {
                return Err(image_error("validity bitmap is noncanonical"));
            }
            let body = u64::try_from(words)
                .map_err(|_| image_error("bitmap word count addressability"))?
                .checked_mul(4)
                .ok_or_else(|| image_error("bitmap bytes overflow"))?;
            Ok(ValidityAllocation {
                validity_len: add(5, body, "validity length")?,
                bytes: sized_owner_bytes::<u32>(words)?,
                slots: u64::from(words != 0),
            })
        }
        _ => Err(image_error("validity form tag is unknown")),
    }
}

struct ValuesAllocation {
    total_len: u64,
    shape: u8,
    body_offset: u64,
    bytes: u64,
    slots: u64,
}

fn source_values<S: TypedImageReadAt + ?Sized>(
    source: &S,
    start: u64,
    ty: SqlType,
    rows: u32,
) -> Result<ValuesAllocation, EngineError> {
    let rows_usize = usize::try_from(rows).map_err(|_| image_error("row count addressability"))?;
    let shape = source_u8(source, start)?;
    if source_u32(source, add(start, 1, "vector count offset")?)? != rows {
        return Err(image_error("vector logical count is not exact"));
    }
    let payload = u64::from(source_u32(source, add(start, 5, "vector payload offset")?)?);
    let body_offset = add(start, 9, "vector body offset")?;
    let total_len = add(9, payload, "vector body length")?;
    let (bytes, slots) = match (shape, ty) {
        (1, SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
            source_fixed_values(source, body_offset, payload, rows_usize, 4, ty)?;
            (
                sized_owner_bytes::<i32>(rows_usize)?,
                u64::from(rows_usize != 0),
            )
        }
        (2, SqlType::Int8 | SqlType::Timestamp) => {
            source_fixed_values(source, body_offset, payload, rows_usize, 8, ty)?;
            (
                sized_owner_bytes::<i64>(rows_usize)?,
                u64::from(rows_usize != 0),
            )
        }
        (3, ty @ SqlType::Numeric { .. }) => {
            source_fixed_values(source, body_offset, payload, rows_usize, 16, ty)?;
            (
                sized_owner_bytes::<i128>(rows_usize)?,
                u64::from(rows_usize != 0),
            )
        }
        (4, SqlType::Uuid) => {
            source_fixed_values(source, body_offset, payload, rows_usize, 16, ty)?;
            (
                sized_owner_bytes::<[u8; 16]>(rows_usize)?,
                u64::from(rows_usize != 0),
            )
        }
        (5, SqlType::Bool) => {
            let words = bitmap_words(rows_usize)?;
            let expected = 4_u64
                .checked_add(
                    u64::try_from(words)
                        .map_err(|_| image_error("bool word count addressability"))?
                        .checked_mul(4)
                        .ok_or_else(|| image_error("bool bytes overflow"))?,
                )
                .ok_or_else(|| image_error("bool payload overflow"))?;
            if payload != expected
                || source_u32(source, body_offset)?
                    != u32::try_from(words).map_err(|_| image_error("bool word count overflow"))?
                || !source_bitmap_is_canonical(
                    source,
                    add(body_offset, 4, "bool bitmap offset")?,
                    words,
                    rows_usize,
                    false,
                )?
            {
                return Err(image_error("bool payload/bitmap is noncanonical"));
            }
            (sized_owner_bytes::<u32>(words)?, u64::from(words != 0))
        }
        (6, SqlType::Text) => source_text_values(source, body_offset, payload, rows_usize)?,
        _ => return Err(image_error("vector shape tag does not match SQL type")),
    };
    Ok(ValuesAllocation {
        total_len,
        shape,
        body_offset,
        bytes,
        slots,
    })
}

fn source_fixed_values<S: TypedImageReadAt + ?Sized>(
    source: &S,
    start: u64,
    payload: u64,
    rows: usize,
    width: usize,
    ty: SqlType,
) -> Result<(), EngineError> {
    let expected = u64::try_from(rows)
        .map_err(|_| image_error("row count addressability"))?
        .checked_mul(u64::try_from(width).expect("fixed width fits u64"))
        .ok_or_else(|| image_error("fixed payload overflows"))?;
    if payload != expected {
        return Err(image_error("fixed vector payload length is noncanonical"));
    }
    let mut raw = [0_u8; 16];
    for ordinal in 0..rows {
        let offset = add(
            start,
            u64::try_from(ordinal)
                .expect("ordinal fits u64")
                .checked_mul(u64::try_from(width).expect("fixed width fits u64"))
                .ok_or_else(|| image_error("fixed value offset overflows"))?,
            "fixed value offset",
        )?;
        source_read(source, offset, &mut raw[..width])?;
        match ty {
            SqlType::Int2
                if i16::try_from(i32::from_le_bytes(raw[..4].try_into().expect("i32")))
                    .is_err() =>
            {
                return Err(image_error("i32 value is outside SQL type bounds"));
            }
            SqlType::Date
                if gpu_db_sql::datetime::validate_date_carrier(i32::from_le_bytes(
                    raw[..4].try_into().expect("i32"),
                ))
                .is_err() =>
            {
                return Err(image_error("i32 value is outside SQL type bounds"));
            }
            SqlType::Timestamp
                if gpu_db_sql::datetime::validate_timestamp_carrier(i64::from_le_bytes(
                    raw[..8].try_into().expect("i64"),
                ))
                .is_err() =>
            {
                return Err(image_error("timestamp is outside SQL type bounds"));
            }
            SqlType::Numeric { precision, .. }
                if crate::numeric_exceeds_precision(
                    i128::from_le_bytes(raw[..16].try_into().expect("i128")),
                    precision,
                ) =>
            {
                return Err(image_error("numeric mantissa exceeds declared precision"));
            }
            _ => {}
        }
    }
    Ok(())
}

fn source_text_values<S: TypedImageReadAt + ?Sized>(
    source: &S,
    body: u64,
    payload: u64,
    rows: usize,
) -> Result<(u64, u64), EngineError> {
    let offset_count = usize::try_from(source_u32(source, body)?)
        .map_err(|_| image_error("text offset count exceeds addressability"))?;
    let expected_count = rows
        .checked_add(1)
        .ok_or_else(|| image_error("text offset count overflow"))?;
    if offset_count != expected_count {
        return Err(image_error("text offset count is not exact"));
    }
    let offset_bytes = u64::try_from(offset_count)
        .map_err(|_| image_error("text offset count addressability"))?
        .checked_mul(8)
        .ok_or_else(|| image_error("text offsets bytes overflow"))?;
    let byte_len_at = add(
        body,
        add(4, offset_bytes, "text offset end")?,
        "text byte length offset",
    )?;
    let text_start = add(byte_len_at, 4, "text data offset")?;
    let text_len = u64::from(source_u32(source, byte_len_at)?);
    if payload
        != text_start
            .checked_sub(body)
            .expect("text offsets after body")
            .checked_add(text_len)
            .ok_or_else(|| image_error("text payload overflow"))?
    {
        return Err(image_error("text payload length is not exact"));
    }
    validate_text_source(
        source,
        add(body, 4, "text offsets start")?,
        offset_count,
        text_start,
        text_len,
    )?;
    let bytes = sized_owner_bytes::<u64>(offset_count)?
        .checked_add(text_len)
        .ok_or_else(|| image_error("text allocation bytes overflow"))?;
    Ok((
        bytes,
        u64::from(offset_count != 0)
            .checked_add(u64::from(text_len != 0))
            .ok_or_else(|| image_error("text allocation slots overflow"))?,
    ))
}

fn validate_text_source<S: TypedImageReadAt + ?Sized>(
    source: &S,
    offsets: u64,
    offset_count: usize,
    text_start: u64,
    text_len: u64,
) -> Result<(), EngineError> {
    let mut offset_ordinal = 0_usize;
    let mut next = source_u64(source, offsets)?;
    if next != 0 {
        return Err(image_error("text offsets are noncanonical"));
    }
    let mut previous = 0_u64;
    let mut position = 0_u64;
    loop {
        while offset_ordinal < offset_count && next == position {
            previous = next;
            offset_ordinal += 1;
            if offset_ordinal < offset_count {
                next = source_u64(
                    source,
                    add(
                        offsets,
                        u64::try_from(offset_ordinal)
                            .expect("offset ordinal fits u64")
                            .checked_mul(8)
                            .ok_or_else(|| image_error("text offset position overflows"))?,
                        "text offset position",
                    )?,
                )?;
                if next < previous || next > text_len {
                    return Err(image_error("text offsets are noncanonical"));
                }
            }
        }
        if offset_ordinal < offset_count && next < position {
            return Err(image_error("text offsets are noncanonical"));
        }
        if position == text_len {
            break;
        }
        let width =
            source_utf8_codepoint_width(source, add(text_start, position, "text byte offset")?)?;
        position = position
            .checked_add(width)
            .filter(|end| *end <= text_len)
            .ok_or_else(|| image_error("text bytes are not UTF-8"))?;
    }
    if offset_ordinal != offset_count || previous != text_len {
        return Err(image_error("text final offset is not exact"));
    }
    Ok(())
}

fn validate_source_placeholders<S: TypedImageReadAt + ?Sized>(
    source: &S,
    vector_start: u64,
    validity_len: u64,
    shape: u8,
    ty: SqlType,
    rows: u32,
    body: u64,
) -> Result<(), EngineError> {
    if source_u8(source, vector_start)? == 0 {
        return Ok(());
    }
    let rows = usize::try_from(rows).map_err(|_| image_error("row count addressability"))?;
    let values = add(vector_start, validity_len, "placeholder value offset")?;
    let payload = add(values, 9, "placeholder payload offset")?;
    for row in 0..rows {
        let word = source_u32(
            source,
            add(
                add(vector_start, 5, "placeholder validity offset")?,
                u64::try_from(row / 32)
                    .expect("row fits u64")
                    .checked_mul(4)
                    .ok_or_else(|| image_error("placeholder word offset overflows"))?,
                "placeholder word offset",
            )?,
        )?;
        if word & (1_u32 << (row % 32)) != 0 {
            continue;
        }
        let zero = match (shape, ty) {
            (1, SqlType::Int2 | SqlType::Int4 | SqlType::Date) => source_zeroes(
                source,
                add(
                    payload,
                    u64::try_from(row).expect("row fits u64") * 4,
                    "i32 placeholder",
                )?,
                4,
            )?,
            (2, SqlType::Int8 | SqlType::Timestamp) => source_zeroes(
                source,
                add(
                    payload,
                    u64::try_from(row).expect("row fits u64") * 8,
                    "i64 placeholder",
                )?,
                8,
            )?,
            (3, SqlType::Numeric { .. }) | (4, SqlType::Uuid) => source_zeroes(
                source,
                add(
                    payload,
                    u64::try_from(row).expect("row fits u64") * 16,
                    "wide placeholder",
                )?,
                16,
            )?,
            (5, SqlType::Bool) => {
                source_u32(
                    source,
                    add(
                        payload,
                        4 + u64::try_from(row / 32).expect("row fits u64") * 4,
                        "bool placeholder",
                    )?,
                )? & (1_u32 << (row % 32))
                    == 0
            }
            (6, SqlType::Text) => {
                let offsets = add(payload, 4, "text placeholder offsets")?;
                source_u64(
                    source,
                    add(
                        offsets,
                        u64::try_from(row).expect("row fits u64") * 8,
                        "text placeholder start",
                    )?,
                )? == source_u64(
                    source,
                    add(
                        offsets,
                        u64::try_from(row + 1).expect("row fits u64") * 8,
                        "text placeholder end",
                    )?,
                )?
            }
            _ => {
                return Err(image_error(
                    "placeholder value shape does not match SQL type",
                ));
            }
        };
        if !zero {
            return Err(image_error("invalid typed value placeholder is nonzero"));
        }
    }
    let _ = body;
    Ok(())
}

fn validate_name_source<S: TypedImageReadAt + ?Sized>(
    source: &S,
    start: u64,
    len: u64,
    mut digest: Option<&mut DomainDigest>,
) -> Result<(), EngineError> {
    if len == 0 {
        return Ok(());
    }
    let mut position = 0_u64;
    while position < len {
        let offset = add(start, position, "name byte offset")?;
        let (width, raw) = source_utf8_codepoint(source, offset)?;
        if position
            .checked_add(width)
            .filter(|end| *end <= len)
            .is_none()
            || raw[..usize::try_from(width).expect("UTF-8 width fits usize")].contains(&0)
        {
            return Err(image_error("name is not canonical UTF-8"));
        }
        if let Some(digest) = digest.as_deref_mut() {
            digest.bytes(&raw[..usize::try_from(width).expect("UTF-8 width fits usize")]);
        }
        position += width;
    }
    Ok(())
}

fn source_utf8_codepoint_width<S: TypedImageReadAt + ?Sized>(
    source: &S,
    start: u64,
) -> Result<u64, EngineError> {
    Ok(source_utf8_codepoint(source, start)?.0)
}

fn source_utf8_codepoint<S: TypedImageReadAt + ?Sized>(
    source: &S,
    start: u64,
) -> Result<(u64, [u8; 4]), EngineError> {
    let mut raw = [0_u8; 4];
    raw[0] = source_u8(source, start)?;
    let width = match raw[0] {
        0x00..=0x7f => 1,
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => return Err(image_error("text bytes are not UTF-8")),
    };
    for (ordinal, byte) in raw.iter_mut().enumerate().take(width).skip(1) {
        *byte = source_u8(
            source,
            add(
                start,
                u64::try_from(ordinal).expect("UTF-8 ordinal fits u64"),
                "UTF-8 continuation",
            )?,
        )?;
    }
    if std::str::from_utf8(&raw[..width]).is_err() {
        return Err(image_error("text bytes are not UTF-8"));
    }
    Ok((u64::try_from(width).expect("UTF-8 width fits u64"), raw))
}

fn source_bitmap_is_canonical<S: TypedImageReadAt + ?Sized>(
    source: &S,
    start: u64,
    words: usize,
    rows: usize,
    reject_all_set: bool,
) -> Result<bool, EngineError> {
    let mut all_set = true;
    for ordinal in 0..words {
        let word = source_u32(
            source,
            add(
                start,
                u64::try_from(ordinal)
                    .expect("bitmap ordinal fits u64")
                    .checked_mul(4)
                    .ok_or_else(|| image_error("bitmap word offset overflows"))?,
                "bitmap word offset",
            )?,
        )?;
        let expected = if ordinal < rows / 32 {
            u32::MAX
        } else {
            (1_u32 << (rows % 32)) - 1
        };
        if ordinal + 1 == words && !rows.is_multiple_of(32) && word & !expected != 0 {
            return Ok(false);
        }
        all_set &= word == expected;
    }
    Ok(!reject_all_set || !all_set)
}

fn hash_source_range<S: TypedImageReadAt + ?Sized>(
    source: &S,
    start: u64,
    len: u64,
    digest: &mut DomainDigest,
) -> Result<(), EngineError> {
    let mut buffer = [0_u8; 1024];
    let mut position = 0_u64;
    while position < len {
        let remaining = len - position;
        let count = usize::try_from(remaining.min(buffer.len() as u64))
            .expect("bounded digest chunk fits usize");
        source_read(
            source,
            add(start, position, "digest range offset")?,
            &mut buffer[..count],
        )?;
        digest.bytes(&buffer[..count]);
        position += u64::try_from(count).expect("chunk fits u64");
    }
    Ok(())
}

fn source_zeroes<S: TypedImageReadAt + ?Sized>(
    source: &S,
    start: u64,
    len: usize,
) -> Result<bool, EngineError> {
    let mut raw = [0_u8; 16];
    source_read(source, start, &mut raw[..len])?;
    Ok(raw[..len].iter().all(|byte| *byte == 0))
}

fn source_u8<S: TypedImageReadAt + ?Sized>(source: &S, offset: u64) -> Result<u8, EngineError> {
    let mut raw = [0_u8; 1];
    source_read(source, offset, &mut raw)?;
    Ok(raw[0])
}

fn source_u32<S: TypedImageReadAt + ?Sized>(source: &S, offset: u64) -> Result<u32, EngineError> {
    let mut raw = [0_u8; 4];
    source_read(source, offset, &mut raw)?;
    Ok(u32::from_le_bytes(raw))
}

fn source_u64<S: TypedImageReadAt + ?Sized>(source: &S, offset: u64) -> Result<u64, EngineError> {
    let mut raw = [0_u8; 8];
    source_read(source, offset, &mut raw)?;
    Ok(u64::from_le_bytes(raw))
}

fn source_read<S: TypedImageReadAt + ?Sized>(
    source: &S,
    offset: u64,
    out: &mut [u8],
) -> Result<(), EngineError> {
    let end = offset
        .checked_add(
            u64::try_from(out.len())
                .map_err(|_| image_error("source read length addressability"))?,
        )
        .ok_or_else(|| image_error("image source range overflows"))?;
    if end > source.len() {
        return Err(image_error("image source range is truncated"));
    }
    source.read_at(offset, out)
}

fn add(left: u64, right: u64, what: &str) -> Result<u64, EngineError> {
    left.checked_add(right)
        .ok_or_else(|| image_error(&format!("{what} overflows")))
}
