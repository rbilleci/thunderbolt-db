//! Canonical WRITE-001 typed-transaction aggregate geometry.
//!
//! This module owns the additive engine codec-5 layout.  The outer ADR-014 envelope remains
//! version 1; one aggregate stream is split into one to four canonical row-mutation chunks and
//! followed by exactly one `GPUDBSTATUS2` fragment.  Measurement is allocation-free and carries
//! no WAL, replay, physical-plan, or publication capability.

#![allow(dead_code)] // Inert codec foundation; live writer adoption follows its acceptance gate.

use gpu_db_wal::CanonicalWalFootprint;

#[path = "typed_insert_aggregate/codec.rs"]
mod codec;
#[path = "typed_insert_aggregate/envelope.rs"]
mod envelope;
#[path = "typed_insert_aggregate/status.rs"]
mod status;

// Semantics two is the sole live codec-5 writer and strict replay authority. Historical codec
// readers remain in their compatibility owners; no provisional S3/mixed-operation semantics is
// compiled beside this INSERT-only profile.
#[path = "typed_insert_aggregate/semantics_v2.rs"]
mod semantics_v2;

#[cfg(test)]
pub(crate) use codec::reencode_legacy_catalog_marker_for_test;
#[allow(unused_imports)] // Terminal pre-WAL adoption consumes this complete inert API next.
pub(crate) use codec::{
    decode_typed_insert_aggregate_bodies, encode_typed_insert_aggregate_bodies,
    measure_and_prepare_typed_insert_aggregate_encoding, prepare_typed_insert_aggregate_encoding,
    reserve_typed_insert_aggregate_bodies, typed_insert_aggregate_root,
    typed_insert_aggregate_status_roots, DecodedTypedInsertAggregate,
    EncodedTypedInsertAggregateBodies, PreparedTypedInsertAggregateEncoding,
    ReservedTypedInsertAggregateBodyBuffers, TypedInsertAggregateFragmentRefs,
    TypedInsertAggregateSectionView, TypedInsertAggregateStatusRoots, TypedInsertAggregateView,
};
#[allow(unused_imports)]
// Consumed by ReservedInsertPreWalPlan after the inert ownership gate.
pub(crate) use envelope::{
    encode_reserved_typed_insert_canonical_envelope, reserve_typed_insert_canonical_envelope,
    EncodedTypedInsertCanonicalEnvelope, ReservedTypedInsertCanonicalEnvelope,
};
#[cfg(test)]
pub(crate) use semantics_v2::decode_catalog_composition_for_test;
#[allow(unused_imports)]
// Consumed by the canonical transaction terminal during WRITE-001 cutover.
pub(crate) use semantics_v2::{
    decode_closed_semantics_v2_replay, encode_live_typed_insert,
    encode_live_typed_insert_transaction, live_autocommit_request_digest,
    live_explicit_request_digest, live_explicit_request_digest_with_catalog,
    write001_identifier_digest, LiveFinalRowDigestSource, LiveTypedInsertCreatedIndex,
    LiveTypedInsertFinalWriter, LiveTypedInsertForeignIndexGeneration, LiveTypedInsertIdentity,
    LiveTypedInsertIndexGeneration, LiveTypedInsertMode, LiveTypedInsertSequenceRestart,
    LiveTypedInsertStatementView, LiveTypedInsertTableGeneration, LiveTypedInsertView,
    SemanticsV2ReplayArtifact, SemanticsV2ReplayMetadata,
};
pub(crate) use status::{decode_status_v2, encode_status_v2, TypedInsertStatusV2};

#[cfg(test)]
#[path = "typed_insert_aggregate/codec_tests.rs"]
mod codec_tests;
#[cfg(test)]
#[path = "typed_insert_aggregate/envelope_tests.rs"]
mod envelope_tests;

pub(crate) const ENGINE_OPERATION_CODEC_TYPED_INSERT_AGGREGATE: u8 = 5;
pub(crate) const AGGREGATE_FORMAT_VERSION: u16 = 1;
/// The only semantics emitted by the current codec-5 writer.
pub(crate) const AGGREGATE_SEMANTICS_V1: u16 = 1;
/// Reserved for the inert, reader-only S4/S7 checkpoint.  No production writer can emit it.
pub(crate) const AGGREGATE_SEMANTICS_V2: u16 = 2;
pub(crate) const AGGREGATE_SECTION_COUNT: usize = 8;
pub(crate) const AGGREGATE_HEADER_BYTES: u64 = 96;
pub(crate) const AGGREGATE_SECTION_HEADER_BYTES: u64 = 16;
pub(crate) const AGGREGATE_ROOT_TRAILER_BYTES: u64 = 32;
pub(crate) const AGGREGATE_CHUNK_HEADER_BYTES: u64 = 76;
pub(crate) const AGGREGATE_STATUS_V2_BYTES: u64 = 204;
pub(crate) const AGGREGATE_MAX_CHUNKS: usize = 4;

/// Typed aggregate semantics selected before measurement. Keeping this as a closed type prevents
/// a live writer from smuggling an arbitrary wire version through otherwise-valid v1 geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TypedInsertAggregateSemantics {
    V1,
    V2,
}

impl TypedInsertAggregateSemantics {
    pub(crate) const fn wire_version(self) -> u16 {
        match self {
            Self::V1 => AGGREGATE_SEMANTICS_V1,
            Self::V2 => AGGREGATE_SEMANTICS_V2,
        }
    }
}

/// The existing canonical fragment-body ceiling minus the codec-5 chunk header.
pub(crate) const AGGREGATE_CHUNK_PAYLOAD_BYTES: u64 =
    gpu_db_wal::canonical_fragment_body_limit() - AGGREGATE_CHUNK_HEADER_BYTES;

/// The outer v1 64-MiB pre-apply ceiling after its 240-byte header, four chunk headers, and
/// STATUS2.  A fifth chunk can therefore never fit.
pub(crate) const AGGREGATE_MAX_STREAM_BYTES: u64 = 64 * 1024 * 1024
    - 240
    - AGGREGATE_MAX_CHUNKS as u64 * AGGREGATE_CHUNK_HEADER_BYTES
    - AGGREGATE_STATUS_V2_BYTES;

pub(crate) const AGGREGATE_FLAG_AUTOCOMMIT: u32 = 1 << 0;
pub(crate) const AGGREGATE_FLAG_EXPLICIT: u32 = 1 << 1;
pub(crate) const AGGREGATE_FLAG_CATALOG: u32 = 1 << 2;
pub(crate) const AGGREGATE_FLAG_RESET: u32 = 1 << 3;
pub(crate) const AGGREGATE_FLAG_PUBLISHED_SEQUENCE: u32 = 1 << 4;
pub(crate) const AGGREGATE_FLAG_PRIVATE_SEQUENCE: u32 = 1 << 5;
pub(crate) const AGGREGATE_FLAG_RETURNING: u32 = 1 << 6;
pub(crate) const AGGREGATE_FLAG_RETAINED_RESPONSE: u32 = 1 << 7;
/// S3 carries one existing ordered transaction operation beside typed INSERT rows. `CATALOG`
/// remains narrower: it says that the S3 operation changes the outer catalog boundary.
pub(crate) const AGGREGATE_FLAG_OPERATION_COMPOSITION: u32 = 1 << 8;
const AGGREGATE_KNOWN_FLAGS: u32 = (1 << 9) - 1;

pub(crate) const OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1: u32 = 1 << 31;
pub(crate) const OUTER_FLAG_FIRST_TYPED_INSERT_WRITER_EPOCH: u32 = 1 << 30;
pub(crate) const OUTER_CONTENT_ROW: u32 = 1 << 0;
pub(crate) const OUTER_CONTENT_CATALOG: u32 = 1 << 1;
pub(crate) const OUTER_CONTENT_RESET: u32 = 1 << 2;
pub(crate) const OUTER_CONTENT_REWRITE: u32 = 1 << 3;
pub(crate) const OUTER_CONTENT_PUBLISHED_SEQUENCE: u32 = 1 << 4;
pub(crate) const OUTER_CONTENT_PRIVATE_SEQUENCE: u32 = 1 << 5;
pub(crate) const OUTER_CONTENT_RETURNING: u32 = 1 << 6;
pub(crate) const OUTER_CONTENT_OPERATION_COMPOSITION: u32 = 1 << 7;
const OUTER_KNOWN_CONTENT: u32 = (1 << 8) - 1;

pub(crate) const AGGREGATE_CHUNK_MAGIC: &[u8; 8] = b"GPUDBOP1";
pub(crate) const AGGREGATE_STREAM_MAGIC: &[u8; 16] = b"GPUDBTXNAGG1\0\0\0\0";
pub(crate) const AGGREGATE_STATUS_MAGIC: &[u8; 12] = b"GPUDBSTATUS2";
pub(crate) const AGGREGATE_CHUNK_FLAG_FIRST: u16 = 1 << 0;
pub(crate) const AGGREGATE_CHUNK_FLAG_LAST: u16 = 1 << 1;
pub(crate) const AGGREGATE_CHUNK_KNOWN_FLAGS: u16 =
    AGGREGATE_CHUNK_FLAG_FIRST | AGGREGATE_CHUNK_FLAG_LAST;

/// Scalar geometry for one ordered aggregate section.  Tags are fixed at one through eight and
/// all v1 section flags are zero, so neither is caller-selectable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AggregateSectionMeasure {
    pub(crate) entry_count: u32,
    pub(crate) payload_bytes: u64,
}

/// Allocation-free input to codec-5 measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TypedInsertAggregateMeasure {
    pub(crate) semantics: TypedInsertAggregateSemantics,
    pub(crate) flags: u32,
    pub(crate) outer_flags: u32,
    pub(crate) stable_transaction_id: u64,
    pub(crate) statement_count: u32,
    pub(crate) insert_statement_count: u32,
    pub(crate) original_inserted_row_count: u64,
    pub(crate) final_row_transition_count: u64,
    pub(crate) allocator_before: u64,
    pub(crate) allocator_high_water: u64,
    pub(crate) table_block_count: u32,
    pub(crate) sections: [AggregateSectionMeasure; AGGREGATE_SECTION_COUNT],
}

/// Exact section offset/size inside the aggregate stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct AggregateSectionLayout {
    pub(crate) tag: u16,
    pub(crate) entry_count: u32,
    pub(crate) header_offset: u64,
    pub(crate) payload_offset: u64,
    pub(crate) payload_bytes: u64,
}

/// Exact chunk offset/size.  `body_bytes` includes the 76-byte codec header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct AggregateChunkLayout {
    pub(crate) ordinal: u32,
    pub(crate) stream_offset: u64,
    pub(crate) payload_bytes: u32,
    pub(crate) body_bytes: u64,
    pub(crate) first: bool,
    pub(crate) last: bool,
}

/// Checked scalar authority used by later exact-buffer allocation and capacity admission.
///
/// The arrays are fixed-size, so measurement itself performs no heap allocation.  Only the first
/// `chunk_count` chunk entries and first `fragment_count` fragment-body lengths are live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TypedInsertAggregateLayout {
    pub(crate) measure: TypedInsertAggregateMeasure,
    pub(crate) sections: [AggregateSectionLayout; AGGREGATE_SECTION_COUNT],
    pub(crate) section_region_bytes: u64,
    pub(crate) stream_bytes: u64,
    pub(crate) chunks: [AggregateChunkLayout; AGGREGATE_MAX_CHUNKS],
    pub(crate) chunk_count: u32,
    pub(crate) fragment_body_bytes: [u64; AGGREGATE_MAX_CHUNKS + 1],
    pub(crate) fragment_count: u32,
    pub(crate) wal: CanonicalWalFootprint,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TypedInsertAggregateLayoutError {
    ZeroTransaction,
    InvalidModeFlags,
    UnknownAggregateFlags(u32),
    UnknownOuterFlags(u32),
    MissingRowContent,
    InvalidStatementCounts,
    InvalidAllocatorRange,
    EmptyTableSet,
    InconsistentContentFlags,
    SectionCountMismatch {
        section: u16,
        expected: u64,
        actual: u64,
    },
    Overflow(&'static str),
    StreamTooLarge(u64),
    CanonicalWal,
}

impl std::fmt::Display for TypedInsertAggregateLayoutError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroTransaction => formatter.write_str("aggregate transaction id is zero"),
            Self::InvalidModeFlags => {
                formatter.write_str("aggregate must select exactly one transaction mode")
            }
            Self::UnknownAggregateFlags(flags) => {
                write!(formatter, "aggregate has unknown flags {flags:#x}")
            }
            Self::UnknownOuterFlags(flags) => {
                write!(formatter, "aggregate has unknown outer flags {flags:#x}")
            }
            Self::MissingRowContent => {
                formatter.write_str("typed INSERT aggregate lacks row content")
            }
            Self::InvalidStatementCounts => {
                formatter.write_str("aggregate statement counts are inconsistent")
            }
            Self::InvalidAllocatorRange => {
                formatter.write_str("aggregate allocator range is inconsistent")
            }
            Self::EmptyTableSet => formatter.write_str("aggregate table-block set is empty"),
            Self::InconsistentContentFlags => {
                formatter.write_str("aggregate and outer content flags are inconsistent")
            }
            Self::SectionCountMismatch {
                section,
                expected,
                actual,
            } => write!(
                formatter,
                "aggregate section {section} count mismatch: expected {expected}, got {actual}"
            ),
            Self::Overflow(domain) => write!(formatter, "aggregate layout overflows {domain}"),
            Self::StreamTooLarge(bytes) => {
                write!(formatter, "aggregate stream {bytes} exceeds codec-5 bound")
            }
            Self::CanonicalWal => {
                formatter.write_str("aggregate layout does not fit canonical WAL v1")
            }
        }
    }
}

impl std::error::Error for TypedInsertAggregateLayoutError {}

impl TypedInsertAggregateMeasure {
    /// Measure the exact stream, canonical chunks, STATUS2 fragment, and outer WAL footprint.
    ///
    /// Counts which can be cross-checked without parsing section payloads are rejected here.
    /// The later exact encoder/decoder owns entry-level semantic closure.
    pub(crate) fn measure(
        self,
    ) -> Result<TypedInsertAggregateLayout, TypedInsertAggregateLayoutError> {
        self.validate_scalar_contract()?;
        let mut sections = [AggregateSectionLayout::default(); AGGREGATE_SECTION_COUNT];
        let mut cursor = AGGREGATE_HEADER_BYTES;
        for (index, measure) in self.sections.iter().copied().enumerate() {
            let tag = u16::try_from(index + 1)
                .map_err(|_| TypedInsertAggregateLayoutError::Overflow("section tag"))?;
            let header_offset = cursor;
            cursor = checked_add(cursor, AGGREGATE_SECTION_HEADER_BYTES, "section headers")?;
            let payload_offset = cursor;
            cursor = checked_add(cursor, measure.payload_bytes, "section payloads")?;
            sections[index] = AggregateSectionLayout {
                tag,
                entry_count: measure.entry_count,
                header_offset,
                payload_offset,
                payload_bytes: measure.payload_bytes,
            };
        }
        let section_region_bytes = cursor
            .checked_sub(AGGREGATE_HEADER_BYTES)
            .ok_or(TypedInsertAggregateLayoutError::Overflow("section region"))?;
        let stream_bytes = checked_add(
            cursor,
            AGGREGATE_ROOT_TRAILER_BYTES,
            "aggregate root trailer",
        )?;
        if stream_bytes > AGGREGATE_MAX_STREAM_BYTES {
            return Err(TypedInsertAggregateLayoutError::StreamTooLarge(
                stream_bytes,
            ));
        }
        let chunk_count_u64 = stream_bytes
            .checked_add(AGGREGATE_CHUNK_PAYLOAD_BYTES - 1)
            .ok_or(TypedInsertAggregateLayoutError::Overflow("chunk count"))?
            / AGGREGATE_CHUNK_PAYLOAD_BYTES;
        let chunk_count = u32::try_from(chunk_count_u64)
            .map_err(|_| TypedInsertAggregateLayoutError::Overflow("chunk count"))?;
        if chunk_count == 0 || chunk_count as usize > AGGREGATE_MAX_CHUNKS {
            return Err(TypedInsertAggregateLayoutError::StreamTooLarge(
                stream_bytes,
            ));
        }

        let mut chunks = [AggregateChunkLayout::default(); AGGREGATE_MAX_CHUNKS];
        let mut fragment_body_bytes = [0_u64; AGGREGATE_MAX_CHUNKS + 1];
        let mut remaining = stream_bytes;
        let mut stream_offset = 0_u64;
        for ordinal in 0..chunk_count as usize {
            let payload = remaining.min(AGGREGATE_CHUNK_PAYLOAD_BYTES);
            if ordinal + 1 != chunk_count as usize && payload != AGGREGATE_CHUNK_PAYLOAD_BYTES {
                return Err(TypedInsertAggregateLayoutError::Overflow(
                    "canonical chunk boundary",
                ));
            }
            let body_bytes = checked_add(AGGREGATE_CHUNK_HEADER_BYTES, payload, "chunk body")?;
            let payload_u32 = u32::try_from(payload)
                .map_err(|_| TypedInsertAggregateLayoutError::Overflow("chunk payload"))?;
            chunks[ordinal] = AggregateChunkLayout {
                ordinal: ordinal as u32,
                stream_offset,
                payload_bytes: payload_u32,
                body_bytes,
                first: ordinal == 0,
                last: ordinal + 1 == chunk_count as usize,
            };
            fragment_body_bytes[ordinal] = body_bytes;
            stream_offset = checked_add(stream_offset, payload, "chunk stream offset")?;
            remaining -= payload;
        }
        if remaining != 0 || stream_offset != stream_bytes {
            return Err(TypedInsertAggregateLayoutError::Overflow("chunk coverage"));
        }
        let status_index = chunk_count as usize;
        fragment_body_bytes[status_index] = AGGREGATE_STATUS_V2_BYTES;
        let fragment_count = chunk_count
            .checked_add(1)
            .ok_or(TypedInsertAggregateLayoutError::Overflow("fragment count"))?;
        let wal =
            gpu_db_wal::canonical_wal_footprint(&fragment_body_bytes[..fragment_count as usize])
                .map_err(|_| TypedInsertAggregateLayoutError::CanonicalWal)?;
        if wal.fragment_count != fragment_count || wal.frame_count != fragment_count + 1 {
            return Err(TypedInsertAggregateLayoutError::CanonicalWal);
        }
        Ok(TypedInsertAggregateLayout {
            measure: self,
            sections,
            section_region_bytes,
            stream_bytes,
            chunks,
            chunk_count,
            fragment_body_bytes,
            fragment_count,
            wal,
        })
    }

    fn validate_scalar_contract(&self) -> Result<(), TypedInsertAggregateLayoutError> {
        if self.stable_transaction_id == 0 {
            return Err(TypedInsertAggregateLayoutError::ZeroTransaction);
        }
        if self.flags & !AGGREGATE_KNOWN_FLAGS != 0 {
            return Err(TypedInsertAggregateLayoutError::UnknownAggregateFlags(
                self.flags,
            ));
        }
        let mode = self.flags & (AGGREGATE_FLAG_AUTOCOMMIT | AGGREGATE_FLAG_EXPLICIT);
        if mode != AGGREGATE_FLAG_AUTOCOMMIT && mode != AGGREGATE_FLAG_EXPLICIT {
            return Err(TypedInsertAggregateLayoutError::InvalidModeFlags);
        }
        let allowed_outer = OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1
            | OUTER_FLAG_FIRST_TYPED_INSERT_WRITER_EPOCH
            | OUTER_KNOWN_CONTENT;
        if self.outer_flags & !allowed_outer != 0
            || self.outer_flags & OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 == 0
        {
            return Err(TypedInsertAggregateLayoutError::UnknownOuterFlags(
                self.outer_flags,
            ));
        }
        if self.outer_flags & OUTER_CONTENT_ROW == 0 {
            return Err(TypedInsertAggregateLayoutError::MissingRowContent);
        }
        // S3 operation composition is additive semantics-v2 state. Retain historical v1
        // decoder strictness for the formerly unallocated flag/content bits rather than letting
        // a new v2 feature silently widen an old aggregate profile.
        if self.semantics == TypedInsertAggregateSemantics::V1
            && (self.flags & AGGREGATE_FLAG_OPERATION_COMPOSITION != 0
                || self.outer_flags & OUTER_CONTENT_OPERATION_COMPOSITION != 0)
        {
            return Err(TypedInsertAggregateLayoutError::InconsistentContentFlags);
        }
        if self.statement_count == 0
            || self.insert_statement_count == 0
            || self.insert_statement_count > self.statement_count
            || (mode == AGGREGATE_FLAG_AUTOCOMMIT
                && (self.statement_count != 1 || self.insert_statement_count != 1))
        {
            return Err(TypedInsertAggregateLayoutError::InvalidStatementCounts);
        }
        let allocator_is_valid = match self.semantics {
            TypedInsertAggregateSemantics::V1 => {
                self.allocator_before != 0
                    && self.allocator_high_water > self.allocator_before
                    && self.allocator_high_water - self.allocator_before
                        == self.original_inserted_row_count
            }
            // Semantics v2 moves allocator ranges into per-table S7 witnesses. The aggregate
            // header sentinels must stay zero so no global row allocator can be reintroduced.
            TypedInsertAggregateSemantics::V2 => {
                self.allocator_before == 0 && self.allocator_high_water == 0
            }
        };
        if self.original_inserted_row_count == 0 || !allocator_is_valid {
            return Err(TypedInsertAggregateLayoutError::InvalidAllocatorRange);
        }
        if self.table_block_count == 0 {
            return Err(TypedInsertAggregateLayoutError::EmptyTableSet);
        }
        let paired_content = [
            (AGGREGATE_FLAG_CATALOG, OUTER_CONTENT_CATALOG),
            (
                AGGREGATE_FLAG_OPERATION_COMPOSITION,
                OUTER_CONTENT_OPERATION_COMPOSITION,
            ),
            (AGGREGATE_FLAG_RESET, OUTER_CONTENT_RESET),
            (
                AGGREGATE_FLAG_PUBLISHED_SEQUENCE,
                OUTER_CONTENT_PUBLISHED_SEQUENCE,
            ),
            (
                AGGREGATE_FLAG_PRIVATE_SEQUENCE,
                OUTER_CONTENT_PRIVATE_SEQUENCE,
            ),
            (AGGREGATE_FLAG_RETURNING, OUTER_CONTENT_RETURNING),
        ];
        if paired_content.into_iter().any(|(aggregate, outer)| {
            (self.flags & aggregate != 0) != (self.outer_flags & outer != 0)
        }) || (self.flags & AGGREGATE_FLAG_RETAINED_RESPONSE != 0)
            != (self.sections[7].entry_count != 0)
            || (self.flags & AGGREGATE_FLAG_RETAINED_RESPONSE != 0)
                && self.flags & AGGREGATE_FLAG_RETURNING == 0
        {
            return Err(TypedInsertAggregateLayoutError::InconsistentContentFlags);
        }
        // Before semantics-v2 allocated OPERATION_COMPOSITION, a catalog-bearing S3 was the
        // only composition shape. Preserve that decoder contract: historical CATALOG implies
        // the legacy S3 slot, while new writers also set the explicit operation bit.
        let expected_counts = [
            u64::from(self.statement_count),
            u64::from(self.insert_statement_count),
            u64::from(
                self.flags & (AGGREGATE_FLAG_OPERATION_COMPOSITION | AGGREGATE_FLAG_CATALOG) != 0,
            ),
            self.original_inserted_row_count,
            0,
            u64::from(self.statement_count),
            1,
            0,
        ];
        for (index, expected) in expected_counts.into_iter().enumerate() {
            if (index == 2 || expected != 0)
                && u64::from(self.sections[index].entry_count) != expected
            {
                return Err(TypedInsertAggregateLayoutError::SectionCountMismatch {
                    section: (index + 1) as u16,
                    expected,
                    actual: u64::from(self.sections[index].entry_count),
                });
            }
        }
        Ok(())
    }
}

impl TypedInsertAggregateLayout {
    pub(crate) fn live_chunks(&self) -> &[AggregateChunkLayout] {
        &self.chunks[..self.chunk_count as usize]
    }

    pub(crate) fn live_fragment_body_bytes(&self) -> &[u64] {
        &self.fragment_body_bytes[..self.fragment_count as usize]
    }
}

fn checked_add(
    left: u64,
    right: u64,
    domain: &'static str,
) -> Result<u64, TypedInsertAggregateLayoutError> {
    left.checked_add(right)
        .ok_or(TypedInsertAggregateLayoutError::Overflow(domain))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measure_with_stream_payload(section_payload_bytes: u64) -> TypedInsertAggregateMeasure {
        let mut sections = [AggregateSectionMeasure {
            entry_count: 0,
            payload_bytes: 0,
        }; AGGREGATE_SECTION_COUNT];
        sections[0] = AggregateSectionMeasure {
            entry_count: 1,
            payload_bytes: section_payload_bytes,
        };
        sections[1] = AggregateSectionMeasure {
            entry_count: 1,
            payload_bytes: 48,
        };
        sections[3] = AggregateSectionMeasure {
            entry_count: 1,
            payload_bytes: 64,
        };
        sections[5] = AggregateSectionMeasure {
            entry_count: 1,
            payload_bytes: 136,
        };
        sections[6] = AggregateSectionMeasure {
            entry_count: 1,
            payload_bytes: 160,
        };
        TypedInsertAggregateMeasure {
            semantics: TypedInsertAggregateSemantics::V1,
            flags: AGGREGATE_FLAG_AUTOCOMMIT,
            outer_flags: OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
            stable_transaction_id: 41,
            statement_count: 1,
            insert_statement_count: 1,
            original_inserted_row_count: 1,
            final_row_transition_count: 1,
            allocator_before: 99,
            allocator_high_water: 100,
            table_block_count: 1,
            sections,
        }
    }

    #[test]
    fn one_to_four_chunk_boundaries_are_canonical_and_exact() {
        let fixed_stream_bytes = AGGREGATE_HEADER_BYTES
            + AGGREGATE_SECTION_HEADER_BYTES * AGGREGATE_SECTION_COUNT as u64
            + 48
            + 64
            + 136
            + 160
            + AGGREGATE_ROOT_TRAILER_BYTES;
        for chunks in 1..=AGGREGATE_MAX_CHUNKS {
            let desired_stream = if chunks == 1 {
                fixed_stream_bytes
            } else {
                AGGREGATE_CHUNK_PAYLOAD_BYTES * (chunks as u64 - 1) + 1
            };
            let payload = desired_stream - fixed_stream_bytes;
            let layout = measure_with_stream_payload(payload).measure().unwrap();
            assert_eq!(layout.chunk_count as usize, chunks);
            assert_eq!(layout.stream_bytes, desired_stream);
            assert_eq!(
                layout
                    .live_chunks()
                    .iter()
                    .map(|chunk| u64::from(chunk.payload_bytes))
                    .sum::<u64>(),
                desired_stream
            );
            assert!(layout.live_chunks()[0].first);
            assert!(layout.live_chunks().last().unwrap().last);
            assert!(layout.live_chunks()[..chunks - 1]
                .iter()
                .all(|chunk| u64::from(chunk.payload_bytes) == AGGREGATE_CHUNK_PAYLOAD_BYTES));
            assert_eq!(layout.fragment_count as usize, chunks + 1);
            assert_eq!(
                layout.wal.fragment_count as usize,
                layout.live_fragment_body_bytes().len()
            );
        }
    }

    #[test]
    fn exact_maximum_stream_fits_and_one_byte_more_fails() {
        let fixed_stream_bytes = AGGREGATE_HEADER_BYTES
            + AGGREGATE_SECTION_HEADER_BYTES * AGGREGATE_SECTION_COUNT as u64
            + 48
            + 64
            + 136
            + 160
            + AGGREGATE_ROOT_TRAILER_BYTES;
        let exact = measure_with_stream_payload(AGGREGATE_MAX_STREAM_BYTES - fixed_stream_bytes)
            .measure()
            .unwrap();
        assert_eq!(exact.stream_bytes, AGGREGATE_MAX_STREAM_BYTES);
        assert_eq!(exact.chunk_count as usize, AGGREGATE_MAX_CHUNKS);
        assert_eq!(exact.wal.preapply_bytes, 64_u64 * 1024 * 1024);
        assert!(matches!(
            measure_with_stream_payload(
                AGGREGATE_MAX_STREAM_BYTES - fixed_stream_bytes + 1
            )
            .measure(),
            Err(TypedInsertAggregateLayoutError::StreamTooLarge(bytes))
                if bytes == AGGREGATE_MAX_STREAM_BYTES + 1
        ));
    }

    #[test]
    fn scalar_identity_count_and_reserved_flag_sabotage_fail_closed() {
        let valid = measure_with_stream_payload(1);
        let mut cases = Vec::new();
        let mut changed = valid;
        changed.stable_transaction_id = 0;
        cases.push(changed);
        let mut changed = valid;
        changed.flags |= 1 << 8;
        cases.push(changed);
        let mut changed = valid;
        changed.flags |= AGGREGATE_FLAG_EXPLICIT;
        cases.push(changed);
        let mut changed = valid;
        changed.outer_flags |= 1 << 29;
        cases.push(changed);
        let mut changed = valid;
        changed.outer_flags &= !OUTER_CONTENT_ROW;
        cases.push(changed);
        let mut changed = valid;
        changed.statement_count = 2;
        cases.push(changed);
        let mut changed = valid;
        changed.sections[3].entry_count = 0;
        cases.push(changed);
        let mut changed = valid;
        changed.allocator_high_water += 1;
        cases.push(changed);
        for case in cases {
            assert!(case.measure().is_err());
        }
    }

    #[test]
    fn measurement_source_has_no_heap_collection_path() {
        let source = include_str!("typed_insert_aggregate.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("production aggregate layout precedes tests");
        for forbidden in ["Vec<", "Vec::", "BTreeMap", "BTreeSet", "Box<", "collect("] {
            assert!(
                !source.contains(forbidden),
                "aggregate measurement must remain allocation-free: {forbidden}"
            );
        }
        assert!(source.contains("[AggregateSectionLayout; AGGREGATE_SECTION_COUNT]"));
        assert!(source.contains("[AggregateChunkLayout; AGGREGATE_MAX_CHUNKS]"));
        assert!(source.contains("[u64; AGGREGATE_MAX_CHUNKS + 1]"));
    }
}
