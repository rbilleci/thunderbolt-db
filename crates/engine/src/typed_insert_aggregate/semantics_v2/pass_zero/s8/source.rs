//! Bounded borrowed sources for S8 pass zero.

use crate::typed_insert_aggregate::codec::DecodedAggregateFraming;
use crate::typed_insert_batch::TypedImageReadAt;
use crate::EngineError;

pub(super) fn fixed_at<const N: usize>(
    framing: &DecodedAggregateFraming<'_>,
    section: usize,
    offset: u64,
    label: &str,
) -> Result<[u8; N], EngineError> {
    framing
        .with_section_reader(section, |reader| {
            reader.skip(offset)?;
            let raw = reader.exact::<N>()?;
            reader.skip(reader.remaining())?;
            Ok(raw)
        })
        .map_err(|err| {
            EngineError::Durability(format!(
                "typed INSERT aggregate semantics-v2 S8 {label}: {err}"
            ))
        })
}

pub(super) fn s6_at(
    framing: &DecodedAggregateFraming<'_>,
    ordinal: u32,
    count: u32,
) -> Result<[u8; 136], EngineError> {
    if ordinal >= count {
        return Err(error("S6 ordinal is outside the statement range"));
    }
    let offset = u64::from(ordinal)
        .checked_mul(136)
        .ok_or_else(|| error("S6 reference offset overflows"))?;
    fixed_at(framing, 5, offset, "S6 reference")
}

pub(super) fn s4_at(
    framing: &DecodedAggregateFraming<'_>,
    ordinal: u32,
    count: u64,
) -> Result<[u8; 64], EngineError> {
    if u64::from(ordinal) >= count {
        return Err(error("S4 ordinal is outside the disposition range"));
    }
    let offset = u64::from(ordinal)
        .checked_mul(64)
        .ok_or_else(|| error("S4 reference offset overflows"))?;
    fixed_at(framing, 3, offset, "S4 reference")
}

pub(super) fn s7_fixed_at<const N: usize>(
    framing: &DecodedAggregateFraming<'_>,
    directory: usize,
    ordinal: u32,
    count: u32,
    width: u64,
) -> Result<[u8; N], EngineError> {
    if directory >= 12 || ordinal >= count || width != N as u64 {
        return Err(error("S7 fixed reference is invalid"));
    }
    let header = fixed_at::<640>(framing, 6, 0, "S7 header reference")?;
    let start = u64_at(&header, 104 + directory * 16)
        .checked_add(
            u64::from(ordinal)
                .checked_mul(width)
                .ok_or_else(|| error("S7 reference multiply overflows"))?,
        )
        .ok_or_else(|| error("S7 reference offset overflows"))?;
    fixed_at(framing, 6, start, "S7 fixed reference")
}

pub(super) struct S8ImageSource<'a> {
    framing: &'a DecodedAggregateFraming<'a>,
    start: u64,
    bytes: u64,
}

impl<'a> S8ImageSource<'a> {
    pub(super) fn new(framing: &'a DecodedAggregateFraming<'a>, start: u64, bytes: u64) -> Self {
        Self {
            framing,
            start,
            bytes,
        }
    }
}

impl TypedImageReadAt for S8ImageSource<'_> {
    fn len(&self) -> u64 {
        self.bytes
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), EngineError> {
        let bytes =
            u64::try_from(out.len()).map_err(|_| error("S8 image read length overflows"))?;
        offset
            .checked_add(bytes)
            .filter(|end| *end <= self.bytes)
            .ok_or_else(|| error("S8 image read leaves its measured range"))?;
        self.framing.with_section_reader(7, |reader| {
            reader.skip(
                self.start
                    .checked_add(offset)
                    .ok_or_else(|| error("S8 image offset overflows"))?,
            )?;
            reader.copy_exact(out)?;
            reader.skip(reader.remaining())?;
            Ok(())
        })
    }
}

pub(super) fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("fixed u32"))
}

pub(super) fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("fixed u64"))
}

pub(super) fn digest_at(bytes: &[u8], offset: usize) -> [u8; 32] {
    bytes[offset..offset + 32].try_into().expect("fixed digest")
}

pub(super) fn error(message: &str) -> EngineError {
    EngineError::Durability(format!("typed INSERT aggregate semantics-v2 S8: {message}"))
}
