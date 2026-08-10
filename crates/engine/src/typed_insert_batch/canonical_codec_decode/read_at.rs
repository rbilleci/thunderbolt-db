//! Strict S2 raw grammar over a borrowed random-access source.
//!
//! This leaf owns only pass-zero parsing/measurement.  It never constructs a decoded model or a
//! contiguous record; later recovery reserves its one raw-copy scratch and calls the existing
//! strict model decoder against that copy.

use super::*;
use sha2::Digest;

pub(crate) trait CanonicalTypedInsertReadAt {
    fn len(&self) -> u64;
    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), EngineError>;
}

pub(super) struct SliceCanonicalSource<'a>(&'a [u8]);
impl<'a> SliceCanonicalSource<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self(bytes)
    }
}
impl CanonicalTypedInsertReadAt for SliceCanonicalSource<'_> {
    fn len(&self) -> u64 {
        u64::try_from(self.0.len()).expect("slice len fits u64")
    }
    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), EngineError> {
        let start =
            usize::try_from(offset).map_err(|_| codec_error("S2 source offset overflows"))?;
        let end = start
            .checked_add(out.len())
            .ok_or_else(|| codec_error("S2 source range overflows"))?;
        out.copy_from_slice(
            self.0
                .get(start..end)
                .ok_or_else(|| codec_error("S2 source is truncated"))?,
        );
        Ok(())
    }
}

pub(super) fn measure_from_source<S: CanonicalTypedInsertReadAt + ?Sized>(
    source: &S,
) -> Result<CanonicalTypedInsertDecodeMeasure, EngineError> {
    measure_from_source_with_profile(source, EffectProfile::CanonicalV1)
}

/// S7 semantics 2 permits only published sequence evidence.  Keep that policy in the raw pass
/// so private v1 forms cannot reach an allocating retained decoder.
pub(super) fn measure_published_only_from_source<S: CanonicalTypedInsertReadAt + ?Sized>(
    source: &S,
) -> Result<CanonicalTypedInsertDecodeMeasure, EngineError> {
    measure_from_source_with_profile(source, EffectProfile::PublishedOnly)
}

fn measure_from_source_with_profile<S: CanonicalTypedInsertReadAt + ?Sized>(
    source: &S,
    profile: EffectProfile,
) -> Result<CanonicalTypedInsertDecodeMeasure, EngineError> {
    let record_bytes = source.len();
    let record_len = usize::try_from(record_bytes)
        .map_err(|_| codec_error("S2 source length exceeds addressability"))?;
    if !(HEADER_LEN..=MAX_RECORD_BYTES).contains(&record_len) {
        return Err(codec_error("record length is outside canonical bounds"));
    }
    let mut header = [0_u8; HEADER_LEN];
    read(source, 0, &mut header)?;
    parse_canonical_typed_insert_record_prefix(&header, record_len)?;
    let mut outer = SourceReader::new(source, 0, record_bytes);
    outer.take(HEADER_LEN)?;
    let mut sections = [(0_u64, 0_u64); SECTION_COUNT as usize];
    for expected in 1..=SECTION_COUNT {
        if outer.u16()? != expected || outer.u16()? != 0 {
            return Err(codec_error("section tag/order/flags are not canonical"));
        }
        let len = u64::from(outer.u32()?);
        let start = outer.cursor;
        outer.take_u64(len)?;
        sections[usize::from(expected - 1)] = (start, len);
    }
    if !outer.done() {
        return Err(codec_error(
            "record has trailing bytes after eight sections",
        ));
    }

    let mut measure = Owners::default();
    let target = scan_target(SourceReader::section(source, sections[0])?, &mut measure)?;
    if scan_columns(
        SourceReader::section(source, sections[1])?,
        target.rows,
        &mut measure,
    )? != target.column_count
    {
        return Err(codec_error("target catalog column count drifted"));
    }
    scan_dependencies(SourceReader::section(source, sections[2])?, &mut measure)?;
    scan_domains(SourceReader::section(source, sections[3])?, &mut measure)?;
    scan_indexes(SourceReader::section(source, sections[4])?, &mut measure)?;
    scan_foreign_keys(SourceReader::section(source, sections[5])?, &mut measure)?;
    if scan_returning(SourceReader::section(source, sections[6])?, &mut measure)? != target.rows {
        return Err(codec_error("RETURNING row geometry differs from target"));
    }
    scan_effects(
        SourceReader::section(source, sections[7])?,
        &mut measure,
        profile,
    )?;

    // A successful strict decode performs three sequential exact comparison buffers.  The full
    // record reencode dominates both logical digest bodies, so it is the internal maximum.
    Ok(CanonicalTypedInsertDecodeMeasure::new(
        record_bytes,
        source_fingerprint(source)?,
        measure.bytes,
        measure.slots,
        record_bytes,
        u64::from(record_bytes != 0),
    ))
}

pub(super) fn copy_after_measure<S: CanonicalTypedInsertReadAt + ?Sized>(
    source: &S,
    measure: CanonicalTypedInsertDecodeMeasure,
    destination: &mut [u8],
) -> Result<(), EngineError> {
    copy_after_measure_with_profile(source, measure, destination, EffectProfile::CanonicalV1)
}

pub(super) fn copy_published_only_after_measure<S: CanonicalTypedInsertReadAt + ?Sized>(
    source: &S,
    measure: CanonicalTypedInsertDecodeMeasure,
    destination: &mut [u8],
) -> Result<(), EngineError> {
    copy_after_measure_with_profile(source, measure, destination, EffectProfile::PublishedOnly)
}

fn copy_after_measure_with_profile<S: CanonicalTypedInsertReadAt + ?Sized>(
    source: &S,
    measure: CanonicalTypedInsertDecodeMeasure,
    destination: &mut [u8],
    profile: EffectProfile,
) -> Result<(), EngineError> {
    let actual = measure_from_source_with_profile(source, profile)?;
    if actual != measure {
        return Err(codec_error("S2 source drifted after its raw measure"));
    }
    let expected = usize::try_from(measure.record_bytes())
        .map_err(|_| codec_error("S2 measured copy length exceeds addressability"))?;
    if destination.len() != expected || source.len() != measure.record_bytes() {
        return Err(codec_error("S2 measured copy length drifted"));
    }
    source.read_at(0, destination)?;
    let copied =
        measure_from_source_with_profile(&SliceCanonicalSource::new(destination), profile)?;
    if copied != measure || copied.content_fingerprint() != measure.content_fingerprint() {
        return Err(codec_error(
            "S2 copied bytes drifted from their raw measure",
        ));
    }
    Ok(())
}

/// Revalidate opaque source-pass evidence against the exact copied bytes before the allocating
/// model decoder runs.  This is the only supported handoff from a borrowed S2 source to the
/// existing retained-model decoder, and rejects stale same-length measurements.
pub(super) fn decode_after_measure(
    bytes: &[u8],
    measure: CanonicalTypedInsertDecodeMeasure,
) -> Result<DecodedTypedInsertRecord, EngineError> {
    decode_after_measure_with_profile(bytes, measure, EffectProfile::CanonicalV1)
}

pub(super) fn decode_published_only_after_measure(
    bytes: &[u8],
    measure: CanonicalTypedInsertDecodeMeasure,
) -> Result<DecodedTypedInsertRecord, EngineError> {
    decode_after_measure_with_profile(bytes, measure, EffectProfile::PublishedOnly)
}

fn decode_after_measure_with_profile(
    bytes: &[u8],
    measure: CanonicalTypedInsertDecodeMeasure,
    profile: EffectProfile,
) -> Result<DecodedTypedInsertRecord, EngineError> {
    let actual = measure_from_source_with_profile(&SliceCanonicalSource::new(bytes), profile)?;
    if actual != measure || actual.content_fingerprint() != measure.content_fingerprint() {
        return Err(codec_error(
            "S2 copied bytes do not match their raw measure",
        ));
    }
    super::decode(bytes)
}

#[derive(Clone, Copy)]
enum EffectProfile {
    CanonicalV1,
    PublishedOnly,
}

#[derive(Clone, Copy)]
struct TargetRaw {
    rows: u32,
    column_count: usize,
}

/// A raw bitmap stays in the source during pass zero.  Keeping only its bounded offset avoids
/// an attacker-directed staging allocation while still letting the values pass enforce every
/// invalid-cell placeholder and the sealed state/default law.
#[derive(Clone, Copy)]
struct BitmapRaw {
    words_start: Option<u64>,
    implicit: bool,
}

impl BitmapRaw {
    fn bit<S: CanonicalTypedInsertReadAt + ?Sized>(
        self,
        source: &S,
        row: usize,
    ) -> Result<bool, EngineError> {
        let Some(words_start) = self.words_start else {
            return Ok(self.implicit);
        };
        let word = row / 32;
        let bit = row % 32;
        let byte_offset = u64::try_from(word)
            .map_err(|_| codec_error("bitmap ordinal overflows"))?
            .checked_mul(4)
            .ok_or_else(|| codec_error("bitmap byte offset overflows"))?;
        let at = words_start
            .checked_add(byte_offset)
            .ok_or_else(|| codec_error("bitmap source offset overflows"))?;
        let mut raw = [0_u8; 4];
        read(source, at, &mut raw)?;
        Ok((u32::from_le_bytes(raw) & (1_u32 << bit)) != 0)
    }
}

#[derive(Default)]
struct Owners {
    bytes: u64,
    slots: u64,
}
impl Owners {
    fn vec<T>(&mut self, count: usize) -> Result<(), EngineError> {
        let bytes = u64::try_from(count)
            .map_err(|_| codec_error("S2 owner count overflows"))?
            .checked_mul(u64::try_from(std::mem::size_of::<T>()).expect("type size fits u64"))
            .ok_or_else(|| codec_error("S2 owner bytes overflow"))?;
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| codec_error("S2 owner bytes overflow"))?;
        self.slots = self
            .slots
            .checked_add(u64::from(count != 0))
            .ok_or_else(|| codec_error("S2 owner slots overflow"))?;
        Ok(())
    }
    fn string(&mut self, bytes: usize) -> Result<(), EngineError> {
        self.bytes = self
            .bytes
            .checked_add(u64::try_from(bytes).map_err(|_| codec_error("S2 string bytes overflow"))?)
            .ok_or_else(|| codec_error("S2 owner bytes overflow"))?;
        self.slots = self
            .slots
            .checked_add(1)
            .ok_or_else(|| codec_error("S2 owner slots overflow"))?;
        Ok(())
    }
}

struct SourceReader<'a, S: ?Sized> {
    source: &'a S,
    cursor: u64,
    end: u64,
}
impl<'a, S: CanonicalTypedInsertReadAt + ?Sized> SourceReader<'a, S> {
    fn new(source: &'a S, start: u64, end: u64) -> Self {
        Self {
            source,
            cursor: start,
            end,
        }
    }
    fn section(source: &'a S, range: (u64, u64)) -> Result<Self, EngineError> {
        let end = range
            .0
            .checked_add(range.1)
            .ok_or_else(|| codec_error("section range overflows"))?;
        Ok(Self::new(source, range.0, end))
    }
    fn done(&self) -> bool {
        self.cursor == self.end
    }
    fn remaining(&self) -> u64 {
        self.end - self.cursor
    }
    fn take(&mut self, count: usize) -> Result<u64, EngineError> {
        self.take_u64(u64::try_from(count).expect("usize fits u64"))
    }
    fn take_u64(&mut self, count: u64) -> Result<u64, EngineError> {
        let start = self.cursor;
        self.cursor = self
            .cursor
            .checked_add(count)
            .filter(|end| *end <= self.end)
            .ok_or_else(|| codec_error("record is truncated"))?;
        Ok(start)
    }
    fn exact<const N: usize>(&mut self) -> Result<[u8; N], EngineError> {
        let at = self.take(N)?;
        let mut raw = [0; N];
        read(self.source, at, &mut raw)?;
        Ok(raw)
    }
    fn u8(&mut self) -> Result<u8, EngineError> {
        Ok(self.exact::<1>()?[0])
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
    fn i128(&mut self) -> Result<i128, EngineError> {
        Ok(i128::from_le_bytes(self.exact()?))
    }
    fn bool(&mut self) -> Result<bool, EngineError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(codec_error("boolean encoding is noncanonical")),
        }
    }
    fn digest(&mut self) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
        self.exact()
    }
    fn identifier(&mut self, owners: &mut Owners) -> Result<(), EngineError> {
        let len =
            usize::try_from(self.u32()?).map_err(|_| codec_error("identifier length overflows"))?;
        if len == 0
            || len > MAX_IDENTIFIER_BYTES
            || u64::try_from(len).expect("len fits") > self.remaining()
        {
            return Err(codec_error("identifier encoding is noncanonical"));
        }
        let start = self.take(len)?;
        validate_utf8_no_nul(self.source, start, len)?;
        owners.string(len)
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
        let p = self.u8()?;
        let s = self.u8()?;
        if self.u8()? != 0 {
            return Err(codec_error("SQL type reserved byte is nonzero"));
        }
        match tag {
            1 if p == 0 && s == 0 => Ok(SqlType::Int2),
            2 if p == 0 && s == 0 => Ok(SqlType::Int4),
            3 if p == 0 && s == 0 => Ok(SqlType::Int8),
            4 if (1..=38).contains(&p) && s <= p => Ok(SqlType::Numeric {
                precision: p,
                scale: s,
            }),
            5 if p == 0 && s == 0 => Ok(SqlType::Bool),
            6 if p == 0 && s == 0 => Ok(SqlType::Text),
            7 if p == 0 && s == 0 => Ok(SqlType::Date),
            8 if p == 0 && s == 0 => Ok(SqlType::Timestamp),
            9 if p == 0 && s == 0 => Ok(SqlType::Uuid),
            _ => Err(codec_error("SQL type tag/typmod is noncanonical")),
        }
    }
}

fn scan_target<S: CanonicalTypedInsertReadAt + ?Sized>(
    mut r: SourceReader<'_, S>,
    o: &mut Owners,
) -> Result<TargetRaw, EngineError> {
    r.identifier(o)?;
    r.identifier(o)?;
    if !valid_oid(r.u32()?) || zero_digest(r.digest()?) {
        return Err(codec_error("target identity is invalid"));
    }
    let _statement_ordinal = r.u32()?;
    let rows = r.u32()?;
    let column_count = usize::try_from(r.u32()?)
        .map_err(|_| codec_error("target column count exceeds addressability"))?;
    if column_count == 0 {
        return Err(codec_error("target has no catalog columns"));
    }
    if !r.done() {
        return Err(codec_error("target section has trailing bytes"));
    }
    Ok(TargetRaw { rows, column_count })
}

fn count<S: CanonicalTypedInsertReadAt + ?Sized>(
    r: &mut SourceReader<'_, S>,
    min: u64,
    what: &str,
) -> Result<usize, EngineError> {
    let count = usize::try_from(r.u32()?).map_err(|_| codec_error("count overflows"))?;
    if min == 0
        || u64::try_from(count).map_err(|_| codec_error("count exceeds addressability"))?
            > r.remaining() / min
    {
        return Err(codec_error(what));
    }
    Ok(count)
}

fn scan_columns<S: CanonicalTypedInsertReadAt + ?Sized>(
    mut r: SourceReader<'_, S>,
    rows: u32,
    o: &mut Owners,
) -> Result<usize, EngineError> {
    let count = count(&mut r, 38, "column count exceeds remaining-byte bound")?;
    let rows = rows_usize(rows)?;
    o.vec::<DecodedColumn>(count)?;
    for ordinal in 0..count {
        if r.u32()?
            != u32::try_from(ordinal).map_err(|_| codec_error("column ordinal overflows"))?
        {
            return Err(codec_error("catalog column order is not canonical"));
        }
        r.identifier(o)?;
        let column_id = r.u32()?;
        let attnum = r.i16()?;
        let ty = r.sql_type()?;
        let type_oid = r.u32()?;
        let type_size = r.i16()?;
        let source_ordinal = r.option_u32()?;
        let _domain_ordinal = r.option_u32()?;
        if column_id == 0 || attnum <= 0 || !valid_oid(type_oid) || type_size != ty.type_size() {
            return Err(codec_error("decoded column identity is invalid"));
        }
        let validity = scan_validity(&mut r, rows, o)?;
        if r.u8()? != 0 {
            return Err(codec_error(
                "sealed canonical record requires AllProvided presence",
            ));
        }
        let defaults = scan_defaults(&mut r, rows, o)?;
        if usize::try_from(r.u32()?).map_err(|_| codec_error("column row count overflows"))? != rows
        {
            return Err(codec_error("column state vector row count drifted"));
        }
        let state_bytes = rows
            .checked_mul(6)
            .ok_or_else(|| codec_error("column state bytes overflow"))?;
        if u64::try_from(state_bytes).map_err(|_| codec_error("column state bytes overflow"))?
            > r.remaining()
        {
            return Err(codec_error(
                "column row count cannot fit remaining state bytes",
            ));
        }
        o.vec::<TypedInsertInputState>(rows)?;
        o.vec::<TypedInsertInputProvenance>(rows)?;
        for row in 0..rows {
            let state = scan_state(&mut r)?;
            let provenance = scan_provenance(&mut r)?;
            validate_cell_law(
                source_ordinal.is_some(),
                validity.bit(r.source, row)?,
                defaults.bit(r.source, row)?,
                state,
                provenance,
            )?;
        }
        scan_values(&mut r, ty, rows, validity, o)?;
    }
    if !r.done() {
        return Err(codec_error("column section has trailing bytes"));
    }
    Ok(count)
}

fn scan_state<S: CanonicalTypedInsertReadAt + ?Sized>(
    r: &mut SourceReader<'_, S>,
) -> Result<u8, EngineError> {
    let state = r.u8()?;
    if !matches!(state, 1..=4) {
        return Err(codec_error("input state tag is unknown"));
    }
    Ok(state)
}

fn scan_provenance<S: CanonicalTypedInsertReadAt + ?Sized>(
    r: &mut SourceReader<'_, S>,
) -> Result<(u8, u32), EngineError> {
    let tag = r.u8()?;
    let index = r.u32()?;
    if !matches!(
        (tag, index),
        (1, 0) | (2, 0) | (3, 1..) | (4, 0) | (5, 0) | (6, 0)
    ) {
        return Err(codec_error("input provenance tag/index is noncanonical"));
    }
    Ok((tag, index))
}

fn validate_cell_law(
    has_source: bool,
    valid: bool,
    defaulted: bool,
    state: u8,
    provenance: (u8, u32),
) -> Result<(), EngineError> {
    let input_is_value = matches!(provenance.0, 2..=4);
    let canonical = match state {
        1 => has_source && valid && !defaulted && input_is_value,
        2 => has_source && !valid && !defaulted && input_is_value,
        3 => !has_source && defaulted && provenance.0 == 1,
        4 => has_source && defaulted && matches!(provenance.0, 5 | 6),
        _ => false,
    };
    if canonical {
        Ok(())
    } else {
        Err(codec_error(
            "decoded input state/provenance/default law drifted",
        ))
    }
}

fn rows_usize(rows: u32) -> Result<usize, EngineError> {
    usize::try_from(rows).map_err(|_| codec_error("row count exceeds addressability"))
}

fn bitmap_words(rows: usize) -> Result<usize, EngineError> {
    rows.checked_add(31)
        .map(|rounded| rounded / 32)
        .ok_or_else(|| codec_error("bitmap word count overflows"))
}

fn scan_validity<S: CanonicalTypedInsertReadAt + ?Sized>(
    r: &mut SourceReader<'_, S>,
    rows: usize,
    o: &mut Owners,
) -> Result<BitmapRaw, EngineError> {
    scan_bitmap_form(r, rows, o, true, "validity")
}

fn scan_defaults<S: CanonicalTypedInsertReadAt + ?Sized>(
    r: &mut SourceReader<'_, S>,
    rows: usize,
    o: &mut Owners,
) -> Result<BitmapRaw, EngineError> {
    scan_bitmap_form(r, rows, o, false, "default-resolution")
}

fn scan_bitmap_form<S: CanonicalTypedInsertReadAt + ?Sized>(
    r: &mut SourceReader<'_, S>,
    rows: usize,
    o: &mut Owners,
    validity: bool,
    role: &str,
) -> Result<BitmapRaw, EngineError> {
    match r.u8()? {
        0 => Ok(BitmapRaw {
            words_start: None,
            implicit: validity,
        }),
        1 => {
            let words = count(r, 4, "bitmap count exceeds remaining-byte bound")?;
            if words != bitmap_words(rows)? {
                return Err(codec_error("bitmap word count is not exact"));
            }
            let words_start = r.cursor;
            let mut all_set = true;
            let mut any_set = false;
            for ordinal in 0..words {
                let word = r.u32()?;
                let expected = bitmap_word_mask(rows, ordinal, words)?;
                if word & !expected != 0 {
                    return Err(codec_error("bitmap tail bits are noncanonical"));
                }
                all_set &= word == expected;
                any_set |= word != 0;
            }
            if (validity && all_set) || (!validity && !any_set) {
                return Err(codec_error(if validity {
                    "validity bitmap is noncanonical"
                } else {
                    "empty default bitmap must use AllDirect form"
                }));
            }
            o.vec::<u32>(words)?;
            let _ = role;
            Ok(BitmapRaw {
                words_start: Some(words_start),
                implicit: validity,
            })
        }
        _ => Err(codec_error("bitmap form tag is unknown")),
    }
}

fn bitmap_word_mask(rows: usize, ordinal: usize, words: usize) -> Result<u32, EngineError> {
    if ordinal >= words {
        return Err(codec_error("bitmap ordinal exceeds exact word count"));
    }
    let remainder = rows % 32;
    if ordinal + 1 == words && remainder != 0 {
        Ok((1_u32 << remainder) - 1)
    } else {
        Ok(u32::MAX)
    }
}

fn scan_values<S: CanonicalTypedInsertReadAt + ?Sized>(
    r: &mut SourceReader<'_, S>,
    ty: SqlType,
    rows: usize,
    validity: BitmapRaw,
    o: &mut Owners,
) -> Result<(), EngineError> {
    let shape = r.u8()?;
    if usize::try_from(r.u32()?).map_err(|_| codec_error("vector count exceeds addressability"))?
        != rows
    {
        return Err(codec_error("vector logical count is not exact"));
    }
    let payload = usize::try_from(r.u32()?)
        .map_err(|_| codec_error("vector payload exceeds addressability"))?;
    if u64::try_from(payload).map_err(|_| codec_error("vector payload exceeds addressability"))?
        > r.remaining()
    {
        return Err(codec_error("vector payload is truncated"));
    }
    let body_start = r.cursor;
    match (shape, ty) {
        (1, SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
            fixed(payload, rows, 4)?;
            for row in 0..rows {
                let value = i32::from_le_bytes(r.exact()?);
                if (ty == SqlType::Int2 && i16::try_from(value).is_err())
                    || (ty == SqlType::Date
                        && gpu_db_sql::datetime::validate_date_carrier(value).is_err())
                    || (!validity.bit(r.source, row)? && value != 0)
                {
                    return Err(codec_error(
                        "i32 value or invalid placeholder is noncanonical",
                    ));
                }
            }
            o.vec::<i32>(rows)?;
        }
        (2, SqlType::Int8 | SqlType::Timestamp) => {
            fixed(payload, rows, 8)?;
            for row in 0..rows {
                let value = r.i64()?;
                if (ty == SqlType::Timestamp
                    && gpu_db_sql::datetime::validate_timestamp_carrier(value).is_err())
                    || (!validity.bit(r.source, row)? && value != 0)
                {
                    return Err(codec_error(
                        "i64 value or invalid placeholder is noncanonical",
                    ));
                }
            }
            o.vec::<i64>(rows)?;
        }
        (3, SqlType::Numeric { precision, .. }) => {
            fixed(payload, rows, 16)?;
            for row in 0..rows {
                let value = r.i128()?;
                if crate::numeric_exceeds_precision(value, precision)
                    || (!validity.bit(r.source, row)? && value != 0)
                {
                    return Err(codec_error(
                        "numeric value or invalid placeholder is noncanonical",
                    ));
                }
            }
            o.vec::<i128>(rows)?;
        }
        (4, SqlType::Uuid) => {
            fixed(payload, rows, 16)?;
            for row in 0..rows {
                let value = r.exact::<16>()?;
                if !validity.bit(r.source, row)? && value != [0; 16] {
                    return Err(codec_error("UUID invalid placeholder is noncanonical"));
                }
            }
            o.vec::<[u8; 16]>(rows)?;
        }
        (5, SqlType::Bool) => scan_bool_values(r, payload, rows, validity, o)?,
        (6, SqlType::Text) => scan_text(r, payload, rows, validity, o)?,
        _ => return Err(codec_error("vector shape tag does not match SQL type")),
    }
    let expected_end = body_start
        .checked_add(
            u64::try_from(payload)
                .map_err(|_| codec_error("vector payload exceeds addressability"))?,
        )
        .ok_or_else(|| codec_error("vector payload end overflows"))?;
    if r.cursor != expected_end {
        return Err(codec_error("vector payload length is not exact"));
    }
    Ok(())
}

fn fixed(payload: usize, rows: usize, width: usize) -> Result<(), EngineError> {
    if payload
        != rows
            .checked_mul(width)
            .ok_or_else(|| codec_error("vector payload overflows"))?
    {
        return Err(codec_error("vector payload length is noncanonical"));
    }
    Ok(())
}

fn scan_bool_values<S: CanonicalTypedInsertReadAt + ?Sized>(
    r: &mut SourceReader<'_, S>,
    payload: usize,
    rows: usize,
    validity: BitmapRaw,
    o: &mut Owners,
) -> Result<(), EngineError> {
    let words = bitmap_words(rows)?;
    let expected_payload = words
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or_else(|| codec_error("bool payload length overflows"))?;
    if payload != expected_payload
        || usize::try_from(r.u32()?).map_err(|_| codec_error("bool word count overflows"))? != words
    {
        return Err(codec_error("bool payload/bitmap is noncanonical"));
    }
    for ordinal in 0..words {
        let word = r.u32()?;
        if word & !bitmap_word_mask(rows, ordinal, words)? != 0 {
            return Err(codec_error("bool payload/bitmap is noncanonical"));
        }
        let first = ordinal
            .checked_mul(32)
            .ok_or_else(|| codec_error("bool row offset overflows"))?;
        let last = rows.min(
            first
                .checked_add(32)
                .ok_or_else(|| codec_error("bool row end overflows"))?,
        );
        for row in first..last {
            if !validity.bit(r.source, row)? && word & (1_u32 << (row % 32)) != 0 {
                return Err(codec_error("bool invalid placeholder is noncanonical"));
            }
        }
    }
    o.vec::<u32>(words)
}

fn scan_text<S: CanonicalTypedInsertReadAt + ?Sized>(
    r: &mut SourceReader<'_, S>,
    payload: usize,
    rows: usize,
    validity: BitmapRaw,
    o: &mut Owners,
) -> Result<(), EngineError> {
    let offsets =
        usize::try_from(r.u32()?).map_err(|_| codec_error("text offset count overflows"))?;
    let expected_offsets = rows
        .checked_add(1)
        .ok_or_else(|| codec_error("text offset count overflows"))?;
    if offsets != expected_offsets {
        return Err(codec_error("text offset count is not exact"));
    }
    let offsets_start = r.cursor;
    let offset_bytes = offsets
        .checked_mul(8)
        .ok_or_else(|| codec_error("text offsets bytes overflow"))?;
    r.take(offset_bytes)?;
    let text_len = usize::try_from(r.u32()?).map_err(|_| codec_error("text bytes overflow"))?;
    let expected_payload = 4_usize
        .checked_add(offset_bytes)
        .and_then(|bytes| bytes.checked_add(4))
        .and_then(|bytes| bytes.checked_add(text_len))
        .ok_or_else(|| codec_error("text payload length overflows"))?;
    if payload != expected_payload {
        return Err(codec_error("text payload length is not exact"));
    }
    let text_start = r.take(text_len)?;
    validate_text_offsets(r.source, offsets_start, offsets, text_start, text_len)?;
    for row in 0..rows {
        if !validity.bit(r.source, row)?
            && text_offset_at(r.source, offsets_start, row)?
                != text_offset_at(r.source, offsets_start, row + 1)?
        {
            return Err(codec_error("text invalid placeholder is noncanonical"));
        }
    }
    o.vec::<u64>(offsets)?;
    o.vec::<u8>(text_len)
}

fn scan_dependencies<S: CanonicalTypedInsertReadAt + ?Sized>(
    mut r: SourceReader<'_, S>,
    o: &mut Owners,
) -> Result<(), EngineError> {
    let count = count(&mut r, 51, "dependency count exceeds remaining-byte bound")?;
    if count == 0 {
        return Err(codec_error("target dependency zero is absent"));
    }
    o.vec::<DecodedDependency>(count)?;
    for ordinal in 0..count {
        if r.u32()?
            != u32::try_from(ordinal).map_err(|_| codec_error("dependency ordinal overflows"))?
            || r.u8()? != if ordinal == 0 { 1 } else { 2 }
        {
            return Err(codec_error("dependency ordinal or role is noncanonical"));
        }
        r.identifier(o)?;
        r.identifier(o)?;
        if !valid_oid(r.u32()?) || zero_digest(r.digest()?) {
            return Err(codec_error("dependency identity is invalid"));
        }
    }
    if !r.done() {
        return Err(codec_error("dependency section has trailing bytes"));
    }
    Ok(())
}

fn scan_domains<S: CanonicalTypedInsertReadAt + ?Sized>(
    mut r: SourceReader<'_, S>,
    o: &mut Owners,
) -> Result<(), EngineError> {
    let count = count(&mut r, 18, "domain count exceeds remaining-byte bound")?;
    o.vec::<DecodedDomain>(count)?;
    for ordinal in 0..count {
        if r.u32()?
            != u32::try_from(ordinal).map_err(|_| codec_error("domain ordinal overflows"))?
        {
            return Err(codec_error("domain ordinal is noncanonical"));
        }
        r.identifier(o)?;
        r.identifier(o)?;
        if !valid_oid(r.u32()?) {
            return Err(codec_error("domain identity is invalid"));
        }
        let _base_type = r.sql_type()?;
    }
    if !r.done() {
        return Err(codec_error("domain section has trailing bytes"));
    }
    Ok(())
}

fn scan_indexes<S: CanonicalTypedInsertReadAt + ?Sized>(
    mut r: SourceReader<'_, S>,
    o: &mut Owners,
) -> Result<(), EngineError> {
    let count = count(&mut r, 36, "index count exceeds remaining-byte bound")?;
    o.vec::<DecodedIndex>(count)?;
    for _ in 0..count {
        scan_index(&mut r, o)?;
    }
    if !r.done() {
        return Err(codec_error("index section has trailing bytes"));
    }
    Ok(())
}

fn scan_index<S: CanonicalTypedInsertReadAt + ?Sized>(
    r: &mut SourceReader<'_, S>,
    o: &mut Owners,
) -> Result<(), EngineError> {
    let _dependency_ordinal = r.u32()?;
    let _raw_ordinal = r.u32()?;
    if !valid_oid(r.u32()?) {
        return Err(codec_error("index identity is invalid"));
    }
    r.identifier(o)?;
    r.identifier(o)?;
    r.identifier(o)?;
    let unique = r.bool()?;
    let primary_key = r.bool()?;
    let unique_constraint = r.bool()?;
    if (primary_key || unique_constraint) && !unique || primary_key && unique_constraint {
        return Err(codec_error("index uniqueness flags are noncanonical"));
    }
    let keys = count(r, 25, "index key count exceeds remaining-byte bound")?;
    if !(1..=32).contains(&keys) {
        return Err(codec_error("index key count is outside canonical bounds"));
    }
    o.vec::<DecodedCatalogColumn>(keys)?;
    for _ in 0..keys {
        scan_catalog_column(r, o)?;
    }
    Ok(())
}

fn scan_catalog_column<S: CanonicalTypedInsertReadAt + ?Sized>(
    r: &mut SourceReader<'_, S>,
    o: &mut Owners,
) -> Result<(), EngineError> {
    let _dependency_ordinal = r.u32()?;
    let _catalog_column_ordinal = r.u32()?;
    let column_id = r.u32()?;
    let attnum = r.i16()?;
    r.identifier(o)?;
    let ty = r.sql_type()?;
    let type_oid = r.u32()?;
    let type_size = r.i16()?;
    if column_id == 0 || attnum <= 0 || !valid_oid(type_oid) || type_size != ty.type_size() {
        return Err(codec_error("resolved catalog column identity is invalid"));
    }
    Ok(())
}

fn scan_foreign_keys<S: CanonicalTypedInsertReadAt + ?Sized>(
    mut r: SourceReader<'_, S>,
    o: &mut Owners,
) -> Result<(), EngineError> {
    let count = count(&mut r, 64, "foreign-key count exceeds remaining-byte bound")?;
    o.vec::<DecodedForeignKey>(count)?;
    for _ in 0..count {
        let _raw_ordinal = r.u32()?;
        r.identifier(o)?;
        r.identifier(o)?;
        r.identifier(o)?;
        r.identifier(o)?;
        scan_catalog_column(&mut r, o)?;
        let _parent_dependency_ordinal = r.u32()?;
        scan_catalog_column(&mut r, o)?;
        scan_index(&mut r, o)?;
    }
    if !r.done() {
        return Err(codec_error("foreign-key section has trailing bytes"));
    }
    Ok(())
}

fn scan_returning<S: CanonicalTypedInsertReadAt + ?Sized>(
    mut r: SourceReader<'_, S>,
    o: &mut Owners,
) -> Result<u32, EngineError> {
    let rows = r.u32()?;
    let count = usize::try_from(r.u32()?).map_err(|_| codec_error("RETURNING count overflows"))?;
    let cells = r.u64()?;
    if zero_digest(r.digest()?)
        || cells
            != u64::from(rows)
                .checked_mul(
                    u64::try_from(count).map_err(|_| codec_error("RETURNING count overflows"))?,
                )
                .ok_or_else(|| codec_error("RETURNING cells overflow"))?
        || usize::try_from(r.u32()?).map_err(|_| codec_error("RETURNING count overflows"))? != count
    {
        return Err(codec_error("RETURNING geometry is inconsistent"));
    }
    o.vec::<DecodedProjection>(count)?;
    for _ in 0..count {
        let _catalog_column_ordinal = r.u32()?;
        let column_id = r.u32()?;
        let attnum = r.i16()?;
        r.identifier(o)?;
        let ty = r.sql_type()?;
        let type_oid = r.u32()?;
        let type_size = r.i16()?;
        if column_id == 0 || attnum <= 0 || !valid_oid(type_oid) || type_size != ty.type_size() {
            return Err(codec_error("RETURNING projection identity is invalid"));
        }
    }
    if !r.done() {
        return Err(codec_error("RETURNING section has trailing bytes"));
    }
    Ok(rows)
}
fn scan_effects<S: CanonicalTypedInsertReadAt + ?Sized>(
    mut r: SourceReader<'_, S>,
    o: &mut Owners,
    profile: EffectProfile,
) -> Result<(), EngineError> {
    let has_parent = r.bool()?;
    if has_parent {
        let txn_id = r.u64()?;
        let _autocommit = r.bool()?;
        if txn_id == 0 || zero_digest(r.digest()?) {
            return Err(codec_error("sequence parent identity is invalid"));
        }
        let _statement_ordinal = r.u32()?;
        let _expression_ordinal_base = r.u32()?;
    }
    let count = count(
        &mut r,
        58,
        "sequence effect count exceeds remaining-byte bound",
    )?;
    if has_parent != (count != 0) {
        return Err(codec_error(
            "sequence parent optional form is not canonical",
        ));
    }
    o.vec::<DecodedEffect>(count)?;
    for ordinal in 0..count {
        if r.u32()?
            != u32::try_from(ordinal).map_err(|_| codec_error("sequence ordinal overflows"))?
        {
            return Err(codec_error("sequence binding ordinal is not canonical"));
        }
        scan_sequence_request(&mut r, o)?;
        let _absolute_expression_ordinal = r.u32()?;
        if zero_digest(r.digest()?) {
            return Err(codec_error("sequence descriptor digest is zero"));
        }
        let _value = r.i64()?;
        match r.u8()? {
            1 => {
                if r.u64()? == 0 || zero_digest(r.digest()?) {
                    return Err(codec_error("published sequence receipt is invalid"));
                }
                let _returned_value = r.i64()?;
            }
            2 if matches!(profile, EffectProfile::CanonicalV1) => scan_private_effect(&mut r)?,
            2 => {
                return Err(codec_error(
                    "private sequence effects are forbidden by this S2 profile",
                ));
            }
            _ => return Err(codec_error("sequence effect tag is unknown")),
        }
    }
    if !r.done() {
        return Err(codec_error("sequence section has trailing bytes"));
    }
    Ok(())
}

fn scan_sequence_request<S: CanonicalTypedInsertReadAt + ?Sized>(
    r: &mut SourceReader<'_, S>,
    o: &mut Owners,
) -> Result<(), EngineError> {
    if !valid_oid(r.u32()?) {
        return Err(codec_error("sequence target table OID is invalid"));
    }
    let _row_ordinal = r.u32()?;
    let _catalog_column_ordinal = r.u32()?;
    if r.u32()? == 0 || !valid_oid(r.u32()?) {
        return Err(codec_error("sequence request identity is invalid"));
    }
    r.identifier(o)?;
    r.identifier(o)?;
    let _statement_ordinal = r.u32()?;
    let _expression_ordinal = r.u32()?;
    Ok(())
}

fn scan_private_effect<S: CanonicalTypedInsertReadAt + ?Sized>(
    r: &mut SourceReader<'_, S>,
) -> Result<(), EngineError> {
    let _prior_last_value = r.i64()?;
    let _prior_is_called = r.bool()?;
    let _next_last_value = r.i64()?;
    let _next_is_called = r.bool()?;
    if !matches!(r.u8()?, 1 | 2) {
        return Err(codec_error("private sequence lifetime origin is invalid"));
    }
    let owner = scan_private_owner(r)?;
    match r.u8()? {
        1 => {
            let predecessor = scan_private_owner(r)?;
            if predecessor != owner {
                return Err(codec_error(
                    "lifecycle predecessor differs from private owner",
                ));
            }
        }
        2 => {
            if zero_digest(r.digest()?) {
                return Err(codec_error("private outcome predecessor is zero"));
            }
        }
        _ => return Err(codec_error("private predecessor tag is unknown")),
    }
    if zero_digest(r.digest()?) {
        return Err(codec_error("private sequence input digest is zero"));
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct PrivateOwnerRaw {
    kind: u8,
    statement_ordinal: u32,
    statement_digest: gpu_db_wal::CanonicalDigest,
    creator_catalog_column_ordinal: Option<u32>,
}

fn scan_private_owner<S: CanonicalTypedInsertReadAt + ?Sized>(
    r: &mut SourceReader<'_, S>,
) -> Result<PrivateOwnerRaw, EngineError> {
    let owner = PrivateOwnerRaw {
        kind: r.u8()?,
        statement_ordinal: r.u32()?,
        statement_digest: r.digest()?,
        creator_catalog_column_ordinal: r.option_u32()?,
    };
    if !matches!(owner.kind, 1..=3) || zero_digest(owner.statement_digest) {
        return Err(codec_error("private owner tag or identity is invalid"));
    }
    Ok(owner)
}

fn read<S: CanonicalTypedInsertReadAt + ?Sized>(
    s: &S,
    at: u64,
    out: &mut [u8],
) -> Result<(), EngineError> {
    let end = at
        .checked_add(u64::try_from(out.len()).expect("len"))
        .filter(|end| *end <= s.len())
        .ok_or_else(|| codec_error("S2 source range is truncated"))?;
    let _ = end;
    s.read_at(at, out)
}

fn source_fingerprint<S: CanonicalTypedInsertReadAt + ?Sized>(
    source: &S,
) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
    let mut digest = sha2::Sha256::new();
    digest.update(b"gpu-db/typed-insert-s2-source/v1");
    digest.update(source.len().to_le_bytes());
    let mut offset = 0_u64;
    let mut scratch = [0_u8; 4096];
    while offset < source.len() {
        let remaining = source
            .len()
            .checked_sub(offset)
            .ok_or_else(|| codec_error("S2 source length drifted while hashing"))?;
        let take = usize::try_from(
            remaining.min(u64::try_from(scratch.len()).expect("scratch length fits")),
        )
        .map_err(|_| codec_error("S2 source hash chunk exceeds addressability"))?;
        read(source, offset, &mut scratch[..take])?;
        digest.update(&scratch[..take]);
        offset = offset
            .checked_add(u64::try_from(take).expect("hash chunk fits u64"))
            .ok_or_else(|| codec_error("S2 source hash offset overflows"))?;
    }
    Ok(digest.finalize().into())
}
fn valid_oid(oid: u32) -> bool {
    (1..=i32::MAX as u32).contains(&oid)
}

fn validate_utf8_no_nul<S: CanonicalTypedInsertReadAt + ?Sized>(
    source: &S,
    start: u64,
    len: usize,
) -> Result<(), EngineError> {
    let mut position = 0_usize;
    let mut raw = [0_u8; 4];
    while position < len {
        let at = source_offset(start, position, "identifier byte offset")?;
        read(source, at, &mut raw[..1])?;
        let width = utf8_width(raw[0], "identifier")?;
        let end = position
            .checked_add(width)
            .filter(|end| *end <= len)
            .ok_or_else(|| codec_error("identifier is not UTF-8"))?;
        read(source, at, &mut raw[..width])?;
        if raw[..width].contains(&0) || std::str::from_utf8(&raw[..width]).is_err() {
            return Err(codec_error("identifier encoding is noncanonical"));
        }
        position = end;
    }
    Ok(())
}

fn validate_text_offsets<S: CanonicalTypedInsertReadAt + ?Sized>(
    source: &S,
    offsets: u64,
    count: usize,
    text_start: u64,
    text_len: usize,
) -> Result<(), EngineError> {
    let text_len = u64::try_from(text_len).map_err(|_| codec_error("text length overflows"))?;
    let mut ordinal = 0_usize;
    let mut next = text_offset_at(source, offsets, ordinal)?;
    if next != 0 {
        return Err(codec_error("text offsets are noncanonical"));
    }
    let mut previous = 0_u64;
    let mut position = 0_u64;
    loop {
        while ordinal < count && next == position {
            previous = next;
            ordinal = ordinal
                .checked_add(1)
                .ok_or_else(|| codec_error("text offset ordinal overflows"))?;
            if ordinal < count {
                next = text_offset_at(source, offsets, ordinal)?;
                if next < previous || next > text_len {
                    return Err(codec_error("text offsets are noncanonical"));
                }
            }
        }
        if ordinal < count && next < position {
            return Err(codec_error("text offsets are noncanonical"));
        }
        if position == text_len {
            break;
        }
        let at = text_start
            .checked_add(position)
            .ok_or_else(|| codec_error("text byte offset overflows"))?;
        let mut raw = [0_u8; 4];
        read(source, at, &mut raw[..1])?;
        let width = u64::try_from(utf8_width(raw[0], "text")?)
            .map_err(|_| codec_error("UTF-8 width overflows"))?;
        let end = position
            .checked_add(width)
            .filter(|end| *end <= text_len)
            .ok_or_else(|| codec_error("text bytes are not UTF-8"))?;
        read(
            source,
            at,
            &mut raw[..usize::try_from(width).map_err(|_| codec_error("UTF-8 width overflows"))?],
        )?;
        if std::str::from_utf8(
            &raw[..usize::try_from(width).map_err(|_| codec_error("UTF-8 width overflows"))?],
        )
        .is_err()
        {
            return Err(codec_error("text bytes are not UTF-8"));
        }
        if ordinal < count && next < end {
            return Err(codec_error("text offset is not a UTF-8 character boundary"));
        }
        position = end;
    }
    if ordinal != count || previous != text_len {
        return Err(codec_error("text final offset is not exact"));
    }
    Ok(())
}

fn text_offset_at<S: CanonicalTypedInsertReadAt + ?Sized>(
    source: &S,
    offsets: u64,
    ordinal: usize,
) -> Result<u64, EngineError> {
    let offset = u64::try_from(ordinal)
        .map_err(|_| codec_error("text offset ordinal overflows"))?
        .checked_mul(8)
        .ok_or_else(|| codec_error("text offset byte position overflows"))?;
    let at = offsets
        .checked_add(offset)
        .ok_or_else(|| codec_error("text offset source position overflows"))?;
    let mut raw = [0_u8; 8];
    read(source, at, &mut raw)?;
    Ok(u64::from_le_bytes(raw))
}

fn source_offset(start: u64, offset: usize, what: &str) -> Result<u64, EngineError> {
    start
        .checked_add(u64::try_from(offset).map_err(|_| codec_error(what))?)
        .ok_or_else(|| codec_error(what))
}

fn utf8_width(first: u8, kind: &str) -> Result<usize, EngineError> {
    match first {
        0..=0x7f => Ok(1),
        0xc2..=0xdf => Ok(2),
        0xe0..=0xef => Ok(3),
        0xf0..=0xf4 => Ok(4),
        _ => Err(codec_error(if kind == "text" {
            "text bytes are not UTF-8"
        } else {
            "identifier is not UTF-8"
        })),
    }
}
