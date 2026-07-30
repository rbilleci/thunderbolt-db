//! Bounded random access over physical fixed-width codec-5 sections and S7 directories.
//!
//! Section readers skip chunked bytes without assembling an attacker-controlled aggregate. Fixed
//! references are therefore O(1) in reader setup; only the caller's explicitly owned range scan
//! is linear.

use super::{
    error, read_u32, read_u64, DecodedAggregateFraming, EngineError, S4_BYTES, S7_HEADER_BYTES,
};

pub(super) fn s7_fixed_at<const N: usize>(
    framing: &DecodedAggregateFraming<'_>,
    directory: usize,
    ordinal: u32,
    count: u32,
    width: u64,
) -> Result<[u8; N], EngineError> {
    if ordinal >= count || width != N as u64 || directory >= 12 {
        return Err(error("S7 fixed directory reference or width is invalid"));
    }
    framing.with_section_reader(6, |reader| {
        let header = reader.exact::<640>()?;
        let offset = read_u64(&header, 104 + directory * 16);
        reader.skip(
            offset
                .checked_add(
                    u64::from(ordinal)
                        .checked_mul(width)
                        .ok_or_else(|| error("S7 fixed directory reference offset overflows"))?,
                )
                .ok_or_else(|| error("S7 fixed directory reference end overflows"))?
                .checked_sub(S7_HEADER_BYTES)
                .ok_or_else(|| error("S7 fixed directory reference starts before payload"))?,
        )?;
        let raw = reader.exact::<N>()?;
        reader.skip(reader.remaining())?;
        Ok(raw)
    })
}

pub(super) fn s1_at(
    framing: &DecodedAggregateFraming<'_>,
    ordinal: u32,
    count: u32,
) -> Result<[u8; 144], EngineError> {
    fixed_section_entry_at(framing, 0, ordinal, count, 144, "S1")
}

pub(super) fn s6_at(
    framing: &DecodedAggregateFraming<'_>,
    ordinal: u32,
    count: u32,
) -> Result<[u8; 136], EngineError> {
    fixed_section_entry_at(framing, 5, ordinal, count, 136, "S6")
}

fn fixed_section_entry_at<const N: usize>(
    framing: &DecodedAggregateFraming<'_>,
    section: usize,
    ordinal: u32,
    count: u32,
    width: u64,
    label: &'static str,
) -> Result<[u8; N], EngineError> {
    if ordinal >= count || width != N as u64 {
        return Err(error("fixed section ordinal or width is invalid"));
    }
    framing
        .with_section_reader(section, |reader| {
            reader.skip(
                u64::from(ordinal)
                    .checked_mul(width)
                    .ok_or_else(|| error("fixed section entry offset overflows"))?,
            )?;
            let raw = reader.exact::<N>()?;
            reader.skip(reader.remaining())?;
            Ok(raw)
        })
        .map_err(|err| {
            EngineError::Durability(format!(
                "typed INSERT aggregate semantics-v2 pass zero {label} reference: {err}"
            ))
        })
}

pub(super) fn s4_range_counts(
    framing: &DecodedAggregateFraming<'_>,
    start: u32,
    count: u32,
    total: u32,
) -> Result<(u32, u64), EngineError> {
    let end = start
        .checked_add(count)
        .filter(|end| *end <= total)
        .ok_or_else(|| error("S7 resolution S4 range is out of bounds"))?;
    let mut survivors = 0_u32;
    let mut affected = 0_u64;
    for ordinal in start..end {
        match s4_at(framing, ordinal, total)?[16] {
            1 => {
                survivors = survivors
                    .checked_add(1)
                    .ok_or_else(|| error("S7 resolution survivor count overflows"))?;
                affected = affected
                    .checked_add(1)
                    .ok_or_else(|| error("S7 resolution affected-row count overflows"))?;
            }
            2 => {
                affected = affected
                    .checked_add(1)
                    .ok_or_else(|| error("S7 resolution affected-row count overflows"))?;
            }
            3 => {}
            _ => return Err(error("S7 resolution S4 disposition kind is invalid")),
        }
    }
    Ok((survivors, affected))
}

pub(super) fn s4_at(
    framing: &DecodedAggregateFraming<'_>,
    ordinal: u32,
    count: u32,
) -> Result<[u8; 64], EngineError> {
    if ordinal >= count {
        return Err(error("S4 absolute disposition reference is out of range"));
    }
    framing.with_section_reader(3, |reader| {
        reader.skip(
            u64::from(ordinal)
                .checked_mul(S4_BYTES)
                .ok_or_else(|| error("S4 disposition offset overflows"))?,
        )?;
        let raw = reader.exact::<64>()?;
        reader.skip(reader.remaining())?;
        Ok(raw)
    })
}

pub(super) fn s7_table_at(
    framing: &DecodedAggregateFraming<'_>,
    table_ref: u32,
) -> Result<[u8; 384], EngineError> {
    framing.with_section_reader(6, |reader| {
        let header = reader.exact::<640>()?;
        let table_count = read_u32(&header, 40);
        if table_ref >= table_count {
            return Err(error("S7 table reference is out of range"));
        }
        reader.skip(
            u64::from(table_ref)
                .checked_mul(384)
                .ok_or_else(|| error("S7 table-block offset overflows"))?,
        )?;
        let raw = reader.exact::<384>()?;
        reader.skip(reader.remaining())?;
        Ok(raw)
    })
}
