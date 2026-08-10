//! Allocation-free semantics-v2 aggregate and S7 fixed-directory proof.
//!
//! This pass consumes only canonical chunk-backed section readers.  It never turns a hostile
//! section into a contiguous `Vec`, and it records just the checked counts/byte maxima required
//! to reserve the later move-only model exactly.

mod access;
mod aggregate;
mod framing;

use super::super::super::codec::DecodedAggregateFraming;
use super::SemanticsV2S7HeaderIdentity;
use crate::typed_insert_aggregate::{
    AGGREGATE_FLAG_AUTOCOMMIT, AGGREGATE_FLAG_EXPLICIT, AGGREGATE_FLAG_PRIVATE_SEQUENCE,
    AGGREGATE_FLAG_PUBLISHED_SEQUENCE, AGGREGATE_FLAG_RETAINED_RESPONSE,
    OUTER_FLAG_FIRST_TYPED_INSERT_WRITER_EPOCH,
};
use crate::typed_insert_batch::{measure_decoded_typed_image_from_source, TypedImageReadAt};
use crate::EngineError;
use access::{s1_at, s4_at, s4_range_counts, s6_at, s7_fixed_at, s7_table_at};
use aggregate::{
    measure_s1_s4_s6, measure_s2, measure_s5, scan_s6_terminal, validate_scalar_and_outer_flags,
};
use framing::{validate_s7_header_roots, validate_s7_payload_digest};
use sha2::{Digest, Sha256};

const S4_BYTES: u64 = 64;
const ABSENT_U32: u32 = u32::MAX;
const S7_HEADER_BYTES: u64 = 640;
const S7_DIRECTORY_COUNT: usize = 14;
/// Header count slot for each physical directory.  S7 deliberately places statement resolutions
/// in header slot 1 but after table dispositions in the physical payload.
const S7_DIRECTORY_COUNT_INDEX: [usize; S7_DIRECTORY_COUNT] =
    [0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0];
const S7_FIXED_WIDTHS: [u64; 12] = [384, 32, 320, 224, 32, 384, 112, 192, 192, 128, 128, 160];

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SemanticsV2PassZero {
    pub(super) statement_count: u32,
    pub(super) original_row_count: u64,
    pub(super) table_count: u32,
    pub(super) transition_count: u32,
    pub(super) image_count: u32,
    /// Exact logical-directory counts.  Retained construction consumes these measured values;
    /// it must not reread a hostile directory count to size an owner.
    pub(super) s1_statement_count: u32,
    pub(super) s2_record_count: u32,
    pub(super) s4_disposition_count: u64,
    pub(super) s5_effect_count: u32,
    pub(super) s6_outcome_count: u32,
    pub(super) s7_directory_counts: [u32; 12],
    pub(super) s7_header: SemanticsV2S7HeaderIdentity,
    pub(super) s8: super::s8::S8PassZeroMeasure,
    pub(super) s2_wire_bytes: u64,
    pub(super) s5_wire_bytes: u64,
    pub(super) s7_value_arena_bytes: u64,
    pub(super) s7_image_arena_bytes: u64,
    pub(super) key_text_bytes: u64,
    pub(super) key_text_slots: u64,
    pub(super) largest_image_bytes: u64,
    pub(super) s2_largest_record_bytes: u64,
    /// Exact persistent terms owned by the already-strict subdecoders. The retained graph adds
    /// its own typed directory owners later; it must not mistake wire widths for that ABI.
    pub(super) s2_decoded_persistent_bytes: u64,
    pub(super) s2_decoded_persistent_slots: u64,
    pub(super) image_decoded_persistent_bytes: u64,
    pub(super) image_decoded_persistent_slots: u64,
    /// Maximum concurrent raw-copy/decode scratch established without a retained graph.
    pub(super) raw_maximum_scratch_bytes: u64,
    pub(super) raw_maximum_scratch_slots: u64,
    pub(crate) terminal_kind: gpu_db_wal::CanonicalOutcomeKind,
    pub(crate) terminal_sqlstate: Option<[u8; 5]>,
    pub(crate) terminal_constraint_id: u64,
}

pub(super) fn measure(
    framing: &DecodedAggregateFraming<'_>,
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
) -> Result<SemanticsV2PassZero, EngineError> {
    if framing.semantics_version() != 2 {
        return Err(error("pass zero received a non-v2 framing"));
    }
    let scalar = framing.header_scalars();
    let sections = framing.sections();
    validate_scalar_and_outer_flags(framing)?;
    if scalar.allocator_before != 0
        || scalar.allocator_high_water != 0
        || scalar.statement_count == 0
        || scalar.insert_statement_count != scalar.statement_count
        || scalar.original_inserted_row_count == 0
        || scalar.table_block_count == 0
        || sections[0].entry_count != scalar.statement_count
        || sections[1].entry_count != scalar.statement_count
        || if scalar.flags
            & (crate::typed_insert_aggregate::AGGREGATE_FLAG_OPERATION_COMPOSITION
                | crate::typed_insert_aggregate::AGGREGATE_FLAG_CATALOG)
            != 0
        {
            sections[2].entry_count != 1
                || !(20..=16 * 1024 * 1024).contains(&sections[2].payload_bytes)
        } else {
            sections[2].entry_count != 0 || sections[2].payload_bytes != 0
        }
        || u64::from(sections[3].entry_count) != scalar.original_inserted_row_count
        || sections[5].entry_count != scalar.statement_count
        || sections[6].entry_count != 1
    {
        return Err(error("v2 aggregate scalar/section profile is invalid"));
    }
    let s2 = measure_s2(framing)?;
    let s5 = measure_s5(
        framing,
        scalar.statement_count,
        scalar.original_inserted_row_count,
    )?;
    if (scalar.flags & AGGREGATE_FLAG_PUBLISHED_SEQUENCE != 0) != (s5.published_count != 0)
        || (scalar.flags & AGGREGATE_FLAG_PRIVATE_SEQUENCE != 0) != (s5.private_count != 0)
    {
        return Err(error(
            "S5 entry presence does not close over the aggregate sequence flag",
        ));
    }
    let terminal = scan_s6_terminal(framing, scalar.statement_count)?;
    let survivors = measure_s1_s4_s6(
        framing,
        scalar.statement_count,
        scalar.original_inserted_row_count,
        terminal.abort_at,
    )?;
    if scalar.final_row_transition_count != survivors {
        return Err(error(
            "aggregate final-transition count does not equal surviving S4 rows",
        ));
    }
    let s7 = measure_s7(
        framing,
        outer,
        scalar.table_block_count,
        scalar.statement_count,
        scalar.original_inserted_row_count,
        terminal.abort_at.is_some(),
    )?;
    let s8 = super::s8::measure(
        framing,
        outer,
        &s7,
        terminal.abort_at,
        scalar.flags & AGGREGATE_FLAG_RETAINED_RESPONSE != 0,
    )?;
    Ok(SemanticsV2PassZero {
        statement_count: scalar.statement_count,
        original_row_count: scalar.original_inserted_row_count,
        table_count: s7.table_count,
        transition_count: s7.transition_count,
        image_count: s7.image_count,
        s1_statement_count: scalar.statement_count,
        s2_record_count: scalar.statement_count,
        s4_disposition_count: scalar.original_inserted_row_count,
        s5_effect_count: s5.effect_count,
        s6_outcome_count: scalar.statement_count,
        s7_directory_counts: s7.directory_counts,
        s7_header: s7.header,
        s8,
        s2_wire_bytes: sections[1].payload_bytes,
        s5_wire_bytes: sections[4].payload_bytes,
        s7_value_arena_bytes: s7.value_arena_bytes,
        s7_image_arena_bytes: s7.image_arena_bytes,
        key_text_bytes: s7.key_text_bytes,
        key_text_slots: s7.key_text_slots,
        largest_image_bytes: s7.largest_image_bytes,
        s2_largest_record_bytes: s2.largest_record_bytes,
        s2_decoded_persistent_bytes: s2.persistent_bytes,
        s2_decoded_persistent_slots: s2.persistent_slots,
        image_decoded_persistent_bytes: s7.image_persistent_bytes,
        image_decoded_persistent_slots: s7.image_persistent_slots,
        // S7 fixed headers/directory entries and streaming digests stay in stack buffers. Only
        // a later fallible S2/image source copy plus its strict decoder owns heap scratch.
        raw_maximum_scratch_bytes: s2.maximum_scratch_bytes.max(s7.image_maximum_scratch_bytes),
        raw_maximum_scratch_slots: s2.maximum_scratch_slots.max(s7.image_maximum_scratch_slots),
        terminal_kind: terminal.kind,
        terminal_sqlstate: terminal.sqlstate,
        terminal_constraint_id: terminal.constraint_id,
    })
}

#[derive(Clone, Copy)]
pub(super) struct S7Measure {
    table_count: u32,
    transition_count: u32,
    image_count: u32,
    largest_image_bytes: u64,
    image_persistent_bytes: u64,
    image_persistent_slots: u64,
    image_maximum_scratch_bytes: u64,
    image_maximum_scratch_slots: u64,
    pub(super) directory_counts: [u32; 12],
    header: SemanticsV2S7HeaderIdentity,
    value_arena_bytes: u64,
    image_arena_bytes: u64,
    key_text_bytes: u64,
    key_text_slots: u64,
}

fn measure_s7(
    framing: &DecodedAggregateFraming<'_>,
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    expected_tables: u32,
    expected_statements: u32,
    expected_rows: u64,
    aborted: bool,
) -> Result<S7Measure, EngineError> {
    framing.with_section_reader(6, |reader| {
        let header = reader.exact::<640>()?;
        if &header[..16] != b"GPUDBS7OVERLAY2\0"
            || u16::from_le_bytes(header[16..18].try_into().expect("fixed S7 version")) != 1
            || u16::from_le_bytes(header[18..20].try_into().expect("fixed S7 semantics")) != 2
            || u32::from_le_bytes(header[20..24].try_into().expect("fixed S7 header bytes")) != 640
            || u32::from_le_bytes(header[24..28].try_into().expect("fixed S7 flags")) != 0
            || u16::from_le_bytes(header[28..30].try_into().expect("fixed S7 directories"))
                != S7_DIRECTORY_COUNT as u16
            || u16::from_le_bytes(header[30..32].try_into().expect("fixed S7 root version")) != 1
            || header[600..640].iter().any(|byte| *byte != 0)
        {
            return Err(error("S7 fixed header is invalid"));
        }
        let total = u64::from_le_bytes(header[32..40].try_into().expect("fixed S7 total"));
        if total != 640 + reader.remaining() {
            return Err(error("S7 total bytes do not match section framing"));
        }
        let counts = [
            read_u32(&header, 40),
            read_u32(&header, 44),
            read_u32(&header, 48),
            read_u32(&header, 52),
            read_u32(&header, 56),
            read_u32(&header, 60),
            read_u32(&header, 64),
            read_u32(&header, 68),
            read_u32(&header, 72),
            read_u32(&header, 76),
            read_u32(&header, 80),
            read_u32(&header, 84),
        ];
        if counts[0] != expected_tables
            || counts[0] == 0
            || counts[1] != expected_statements
            || u64::from(counts[2]) != expected_rows
            || counts[11] != counts[0]
        {
            return Err(error("S7 required directory counts are invalid"));
        }
        let mut cursor = S7_HEADER_BYTES;
        let mut directory_bytes = [0_u64; S7_DIRECTORY_COUNT];
        for index in 0..S7_DIRECTORY_COUNT {
            let offset = read_u64(&header, 104 + index * 16);
            let bytes = read_u64(&header, 112 + index * 16);
            let expected = if index < 12 {
                checked_mul(
                    u64::from(counts[S7_DIRECTORY_COUNT_INDEX[index]]),
                    S7_FIXED_WIDTHS[index],
                )?
            } else if index == 12 {
                read_u64(&header, 88)
            } else {
                read_u64(&header, 96)
            };
            if offset != cursor || bytes != expected {
                return Err(error("S7 directory offset/length adjacency is invalid"));
            }
            directory_bytes[index] = bytes;
            cursor = cursor
                .checked_add(bytes)
                .ok_or_else(|| error("S7 directory end overflows"))?;
        }
        if cursor != total {
            return Err(error("S7 directory coverage does not reach total bytes"));
        }
        validate_s7_header_roots(framing, outer, &header)?;
        validate_s7_payload_digest(framing, &header)?;
        validate_table_blocks_and_root(reader, &header, &counts)?;
        validate_table_dispositions(framing, reader, &counts)?;
        validate_statement_resolutions(framing, reader, &counts, outer.commit_seq)?;
        validate_dependency_tokens(framing, reader, &counts, read_u64(&header, 328), aborted)?;
        validate_statement_dependency_uses(framing, reader, &counts)?;
        validate_index_descriptors(framing, reader, &counts)?;
        validate_index_key_columns(framing, reader, &counts)?;
        validate_transitions(framing, reader, &counts)?;
        validate_key_effects(framing, reader, &counts)?;
        let key_values =
            validate_typed_key_components(framing, reader, &counts, directory_bytes[12])?;
        validate_projection_bindings(framing, reader, &counts)?;
        let images = validate_images(
            framing,
            reader,
            &counts,
            read_u64(&header, 104 + 13 * 16),
            directory_bytes[13],
        )?;
        reader.skip(directory_bytes[12])?;
        reader.skip(directory_bytes[13])?;
        let historical_request_identity =
            framing.outer_flags() & OUTER_FLAG_FIRST_TYPED_INSERT_WRITER_EPOCH != 0;
        if historical_request_identity {
            // Historical bit30-set aggregates never carried the caller's canonical request
            // identity.  Preserve their frozen aggregate-derived closure exactly so a clear-bit
            // generic record cannot relabel an old authority.
            let request_digest = aggregate_request_digest(framing, &counts)?;
            if request_digest != outer.request_digest
                || request_digest != framing.status().request_digest
            {
                return Err(error(
                    "historical v2 aggregate request identity does not close over outer/STATUS2",
                ));
            }
        } else if outer.request_digest == [0; 32]
            || outer.request_digest != framing.status().request_digest
        {
            // Generic bit30-clear codec-5 records bind the normal CanonicalRequest text digest
            // through the outer header and STATUS2.  S8, when present, independently echoes the
            // outer digest in its fixed header; the typed statement/projection roots above still
            // close the INSERT body itself.
            return Err(error(
                "generic v2 request identity does not close over outer/STATUS2",
            ));
        }
        Ok(S7Measure {
            table_count: counts[0],
            transition_count: counts[7],
            image_count: counts[11],
            largest_image_bytes: images.largest_bytes,
            image_persistent_bytes: images.persistent_bytes,
            image_persistent_slots: images.persistent_slots,
            image_maximum_scratch_bytes: images.maximum_scratch_bytes,
            image_maximum_scratch_slots: images.maximum_scratch_slots,
            directory_counts: counts,
            header: SemanticsV2S7HeaderIdentity {
                total_bytes: total,
                root_descriptor_version: u16::from_le_bytes(
                    header[30..32].try_into().expect("fixed S7 root version"),
                ),
                catalog_before_epoch: read_u64(&header, 328),
                catalog_after_epoch: read_u64(&header, 336),
                catalog_before_digest: header[344..376]
                    .try_into()
                    .expect("fixed S7 catalog-before digest"),
                catalog_after_digest: header[376..408]
                    .try_into()
                    .expect("fixed S7 catalog-after digest"),
                initial_database_root: header[408..440]
                    .try_into()
                    .expect("fixed S7 initial database root"),
                final_database_root: header[440..472]
                    .try_into()
                    .expect("fixed S7 final database root"),
                initial_overlay_root: header[472..504]
                    .try_into()
                    .expect("fixed S7 initial overlay root"),
                final_overlay_root: header[504..536]
                    .try_into()
                    .expect("fixed S7 final overlay root"),
                root_descriptor: header[536..568]
                    .try_into()
                    .expect("fixed S7 root descriptor"),
                payload_digest: header[568..600]
                    .try_into()
                    .expect("fixed S7 payload digest"),
            },
            value_arena_bytes: directory_bytes[12],
            image_arena_bytes: directory_bytes[13],
            key_text_bytes: key_values.text_bytes,
            key_text_slots: key_values.text_slots,
        })
    })
}

fn aggregate_request_digest(
    framing: &DecodedAggregateFraming<'_>,
    counts: &[u32; 12],
) -> Result<[u8; 32], EngineError> {
    let scalar = framing.header_scalars();
    let mode = match scalar.flags & (AGGREGATE_FLAG_AUTOCOMMIT | AGGREGATE_FLAG_EXPLICIT) {
        AGGREGATE_FLAG_AUTOCOMMIT => 1_u8,
        AGGREGATE_FLAG_EXPLICIT => 2_u8,
        _ => {
            return Err(error(
                "v2 aggregate request has an invalid transaction mode",
            ));
        }
    };
    let mut digest = begin_v2_digest(b"gpu-db/write001/aggregate-request/v2");
    digest.update([mode]);
    digest.update(scalar.statement_count.to_le_bytes());
    for ordinal in 0..counts[1] {
        let s1 = s1_at(framing, ordinal, counts[1])?;
        let resolution = s7_fixed_at::<320>(framing, 2, ordinal, counts[1], 320)?;
        let projection_start = read_u32(&resolution, 48);
        let projection_count = read_u32(&resolution, 52);
        digest.update(ordinal.to_le_bytes());
        digest.update(&s1[16..48]);
        digest.update(projection_count.to_le_bytes());
        for projection in projection_start
            ..projection_start
                .checked_add(projection_count)
                .filter(|end| *end <= counts[10])
                .ok_or_else(|| error("v2 aggregate request projection range is invalid"))?
        {
            let binding = s7_fixed_at::<128>(framing, 10, projection, counts[10], 128)?;
            digest.update(&binding[38..40]);
        }
    }
    let mut writer_count = 0_u32;
    for disposition_ordinal in 0..counts[2] {
        if final_writer_binding(framing, disposition_ordinal, counts)?.is_some() {
            writer_count = writer_count
                .checked_add(1)
                .ok_or_else(|| error("v2 aggregate final-writer count overflows"))?;
        }
    }
    if writer_count != 0 {
        if mode != 2 {
            return Err(error(
                "v2 autocommit aggregate carries a later final writer",
            ));
        }
        digest.update(b"FINALWRITERS1");
        digest.update(writer_count.to_le_bytes());
        for disposition_ordinal in 0..counts[2] {
            if let Some((ordinal, statement_digest)) =
                final_writer_binding(framing, disposition_ordinal, counts)?
            {
                digest.update(ordinal.to_le_bytes());
                digest.update(statement_digest);
            }
        }
    }
    if scalar.flags
        & (crate::typed_insert_aggregate::AGGREGATE_FLAG_OPERATION_COMPOSITION
            | crate::typed_insert_aggregate::AGGREGATE_FLAG_CATALOG)
        != 0
    {
        let catalog_bytes = framing.sections()[2].payload_bytes;
        digest.update(b"CATALOG1");
        digest.update(catalog_bytes.to_le_bytes());
        framing.with_section_reader(2, |reader| {
            let mut scratch = [0_u8; 4096];
            while reader.remaining() != 0 {
                let take = usize::try_from(reader.remaining().min(scratch.len() as u64))
                    .map_err(|_| error("S3 catalog operation length is not addressable"))?;
                reader.copy_exact(&mut scratch[..take])?;
                digest.update(&scratch[..take]);
            }
            Ok(())
        })?;
    }
    Ok(digest.finalize().into())
}

fn final_writer_binding(
    framing: &DecodedAggregateFraming<'_>,
    disposition_ordinal: u32,
    counts: &[u32; 12],
) -> Result<Option<(u32, [u8; 32])>, EngineError> {
    let disposition = s4_at(framing, disposition_ordinal, counts[2])?;
    if disposition[16] == 2 && disposition[17] == 1 {
        let digest = disposition[32..64]
            .try_into()
            .expect("fixed canceled-writer digest");
        if digest == [0; 32] {
            return Err(error("v2 aggregate canceled row has a zero writer digest"));
        }
        return Ok(Some((read_u32(&disposition, 28), digest)));
    }
    if disposition[16] != 1 {
        return Ok(None);
    }
    let transition_ref = read_u32(&disposition, 24);
    let transition = s7_fixed_at::<192>(framing, 7, transition_ref, counts[7], 192)?;
    if transition[17] != 1 {
        return Ok(None);
    }
    if read_u32(&transition, 20) != disposition_ordinal || transition[160..192] == [0; 32] {
        return Err(error(
            "v2 aggregate final writer does not bind its S4 disposition",
        ));
    }
    Ok(Some((
        read_u32(&transition, 48),
        transition[160..192]
            .try_into()
            .expect("fixed final-writer digest"),
    )))
}

#[derive(Clone, Copy)]
struct ImagePassZeroMeasure {
    largest_bytes: u64,
    persistent_bytes: u64,
    persistent_slots: u64,
    maximum_scratch_bytes: u64,
    maximum_scratch_slots: u64,
}

fn validate_images(
    framing: &DecodedAggregateFraming<'_>,
    reader: &mut super::super::super::codec::DecodedAggregateSectionReader<'_>,
    counts: &[u32; 12],
    image_arena_start: u64,
    image_arena_bytes: u64,
) -> Result<ImagePassZeroMeasure, EngineError> {
    let mut image_cursor = 0_u64;
    let mut largest = 0_u64;
    let mut persistent_bytes = 0_u64;
    let mut persistent_slots = 0_u64;
    let mut maximum_scratch_bytes = 0_u64;
    let mut maximum_scratch_slots = 0_u64;
    for ordinal in 0..counts[11] {
        let raw = reader.exact::<160>()?;
        let table_ref = read_u32(&raw, 4);
        let offset = read_u64(&raw, 24);
        let bytes = read_u64(&raw, 32);
        let table = s7_table_at(framing, table_ref)?;
        if read_u32(&raw, 0) != ordinal
            || table_ref != ordinal
            || read_u32(&raw, 8) != 1
            || read_u32(&raw, 12) != 0
            || read_u32(&raw, 16) != read_u32(&table, 92)
            || read_u32(&raw, 20) != read_u32(&table, 116)
            || offset != image_cursor
            || bytes == 0
            || raw[40..64].iter().any(|byte| *byte != 0)
            || raw[64..96] != table[224..256]
            || raw[96..128] != table[256..288]
            || raw[64..128]
                .chunks_exact(32)
                .any(|digest| digest == [0; 32])
            || raw[128..160]
                != exact_v2_digest(
                    b"gpu-db/write001/s7-image-descriptor/v2",
                    &[&raw[..128], &[0; 32]],
                )
        {
            return Err(error(
                "S7 image-descriptor scalar/table/digest form is invalid",
            ));
        }
        let source = S7ImageSource {
            framing,
            image_start: image_arena_start
                .checked_add(offset)
                .ok_or_else(|| error("S7 image arena offset overflows"))?,
            bytes,
        };
        let image_measure = measure_decoded_typed_image_from_source(&source).map_err(|_| {
            error("S7 nested shared image fails allocation-free strict measurement")
        })?;
        if image_measure.image_bytes() != bytes {
            return Err(error("S7 nested shared image measured length drifted"));
        }
        persistent_bytes = persistent_bytes
            .checked_add(image_measure.persistent_bytes())
            .ok_or_else(|| error("S7 retained shared-image bytes overflow"))?;
        persistent_slots = persistent_slots
            .checked_add(image_measure.persistent_allocation_slots())
            .ok_or_else(|| error("S7 retained shared-image slots overflow"))?;
        let image_scratch = image_measure
            .maximum_with_image_copy_bytes()
            .map_err(|_| error("S7 image copy and decoder scratch overflow"))?;
        maximum_scratch_bytes = maximum_scratch_bytes.max(image_scratch);
        maximum_scratch_slots = maximum_scratch_slots.max(
            image_measure
                .maximum_with_image_copy_allocation_slots()
                .map_err(|_| error("S7 image copy and decoder scratch slots overflow"))?,
        );
        validate_final_image_header_and_columns(
            &source,
            table_ref,
            read_u32(&raw, 16),
            read_u32(&raw, 20),
            &raw[64..96],
        )?;
        if image_content_digest(&source)? != raw[96..128] {
            return Err(error("S7 nested shared image content digest is invalid"));
        }
        image_cursor = image_cursor
            .checked_add(bytes)
            .ok_or_else(|| error("S7 image descriptor end overflows"))?;
        largest = largest.max(bytes);
    }
    if image_cursor != image_arena_bytes {
        return Err(error(
            "S7 image descriptors do not exactly cover the image arena",
        ));
    }
    Ok(ImagePassZeroMeasure {
        largest_bytes: largest,
        persistent_bytes,
        persistent_slots,
        maximum_scratch_bytes,
        maximum_scratch_slots,
    })
}

struct S7ImageSource<'a> {
    framing: &'a DecodedAggregateFraming<'a>,
    image_start: u64,
    bytes: u64,
}

impl TypedImageReadAt for S7ImageSource<'_> {
    fn len(&self) -> u64 {
        self.bytes
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), EngineError> {
        let bytes =
            u64::try_from(out.len()).map_err(|_| error("S7 image read length overflows"))?;
        let end = offset
            .checked_add(bytes)
            .filter(|end| *end <= self.bytes)
            .ok_or_else(|| error("S7 image read is outside the measured image"))?;
        self.framing.with_section_reader(6, |reader| {
            reader.skip(
                self.image_start
                    .checked_add(offset)
                    .ok_or_else(|| error("S7 image absolute read offset overflows"))?,
            )?;
            reader.copy_exact(out)?;
            reader.skip(reader.remaining())?;
            Ok(())
        })?;
        debug_assert_eq!(end, offset + bytes);
        Ok(())
    }
}

fn validate_final_image_header_and_columns(
    source: &impl TypedImageReadAt,
    table_ref: u32,
    rows: u32,
    columns: u32,
    layout_digest: &[u8],
) -> Result<(), EngineError> {
    let mut header = [0_u8; 112];
    source.read_at(0, &mut header)?;
    if &header[..16] != b"GPUDBTYPEDIMAGE2"
        || u16::from_le_bytes(header[16..18].try_into().expect("fixed image version")) != 2
        || u16::from_le_bytes(header[18..20].try_into().expect("fixed image header bytes")) != 112
        || read_u32(&header, 20) != 1
        || read_u32(&header, 24) != rows
        || read_u32(&header, 28) != columns
        || header[64..96] != *layout_digest
        || header[96..112].iter().any(|byte| *byte != 0)
    {
        return Err(error(
            "S7 final shared image header does not bind its descriptor",
        ));
    }
    for ordinal in 0..columns {
        let mut descriptor = [0_u8; 96];
        source.read_at(112 + u64::from(ordinal) * 96, &mut descriptor)?;
        if read_u32(&descriptor, 0) != ordinal
            || read_u32(&descriptor, 4) != ordinal
            || read_u32(&descriptor, 8) == 0
            || read_u32(&descriptor, 12) != table_ref
            || i16::from_le_bytes(descriptor[16..18].try_into().expect("fixed image attnum")) <= 0
        {
            return Err(error(
                "S7 final shared image catalog-order identity is invalid",
            ));
        }
    }
    Ok(())
}

fn image_content_digest(source: &impl TypedImageReadAt) -> Result<[u8; 32], EngineError> {
    let mut digest = begin_v2_digest(b"gpu-db/write001/s7-image-content/v2");
    digest.update(source.len().to_le_bytes());
    let mut offset = 0_u64;
    let mut scratch = [0_u8; 4096];
    while offset != source.len() {
        let take = usize::try_from((source.len() - offset).min(scratch.len() as u64))
            .map_err(|_| error("S7 image content digest scratch is not addressable"))?;
        source.read_at(offset, &mut scratch[..take])?;
        digest.update(&scratch[..take]);
        offset = offset
            .checked_add(take as u64)
            .ok_or_else(|| error("S7 image content digest offset overflows"))?;
    }
    Ok(digest.finalize().into())
}

fn validate_projection_bindings(
    framing: &DecodedAggregateFraming<'_>,
    reader: &mut super::super::super::codec::DecodedAggregateSectionReader<'_>,
    counts: &[u32; 12],
) -> Result<(), EngineError> {
    let mut previous = None;
    for ordinal in 0..counts[10] {
        let raw = reader.exact::<128>()?;
        let statement = read_u32(&raw, 4);
        let projection_ordinal = read_u32(&raw, 8);
        let table_ref = read_u32(&raw, 20);
        let attnum = i16::from_le_bytes(raw[24..26].try_into().expect("fixed projection attnum"));
        let storage: [u8; 4] = raw[28..32]
            .try_into()
            .expect("fixed projection storage type");
        let declared_oid = read_u32(&raw, 32);
        let type_size = i16::from_le_bytes(raw[36..38].try_into().expect("fixed projection size"));
        let resolution = s7_fixed_at::<320>(framing, 2, statement, counts[1], 320)?;
        if read_u32(&raw, 0) != ordinal
            || statement >= counts[1]
            || projection_ordinal >= read_u32(&resolution, 52)
            || read_u32(&resolution, 48)
                .checked_add(projection_ordinal)
                .filter(|expected| *expected == ordinal)
                .is_none()
            || read_u32(&raw, 12) == ABSENT_U32
            || read_u32(&raw, 16) == 0
            || table_ref >= counts[0]
            || attnum <= 0
            || raw[26..28].iter().any(|byte| *byte != 0)
            || !valid_sql_storage(storage, declared_oid, type_size)
            || !matches!(
                u16::from_le_bytes(raw[38..40].try_into().expect("fixed result format")),
                0 | 1
            )
            || read_u32(&raw, 40) != projection_ordinal
            || raw[44..64].iter().any(|byte| *byte != 0)
            || raw[64..96] == [0; 32]
            || raw[96..128]
                != exact_v2_digest(b"gpu-db/write001/s7-projection/v2", &[&raw[..96], &[0; 32]])
            || previous.is_some_and(|prior: (u32, u32)| prior >= (statement, projection_ordinal))
        {
            return Err(error(
                "S7 projection-binding scalar/order/digest form is invalid",
            ));
        }
        previous = Some((statement, projection_ordinal));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct KeyValueMeasure {
    text_bytes: u64,
    text_slots: u64,
}

fn validate_typed_key_components(
    framing: &DecodedAggregateFraming<'_>,
    reader: &mut super::super::super::codec::DecodedAggregateSectionReader<'_>,
    counts: &[u32; 12],
    value_arena_bytes: u64,
) -> Result<KeyValueMeasure, EngineError> {
    let mut previous = None;
    let mut next_value = 0_u64;
    let mut text_bytes = 0_u64;
    let mut text_slots = 0_u64;
    for ordinal in 0..counts[9] {
        let raw = reader.exact::<128>()?;
        let effect_ref = read_u32(&raw, 4);
        let side = raw[8];
        let validity = raw[9];
        let component_ordinal = read_u32(&raw, 12);
        let key_column_ref = read_u32(&raw, 16);
        let source_column = read_u32(&raw, 20);
        let value_offset = read_u64(&raw, 24);
        let value_bytes = read_u32(&raw, 32);
        let storage: [u8; 4] = raw[36..40]
            .try_into()
            .expect("fixed component storage type");
        let declared_oid = read_u32(&raw, 40);
        let type_size = i16::from_le_bytes(raw[44..46].try_into().expect("fixed component size"));
        let effect = s7_fixed_at::<192>(framing, 8, effect_ref, counts[8], 192)?;
        let key = s7_fixed_at::<112>(framing, 6, key_column_ref, counts[6], 112)?;
        let end = value_offset
            .checked_add(u64::from(value_bytes))
            .filter(|end| *end <= value_arena_bytes)
            .ok_or_else(|| error("S7 typed-key value range is out of bounds"))?;
        if validity == 0 {
            validate_typed_key_value_domain(
                framing,
                storage,
                value_offset,
                value_bytes,
                value_arena_bytes,
            )?;
        }
        if read_u32(&raw, 0) != ordinal
            || effect_ref >= counts[8]
            || side != 2
            || !matches!(validity, 0 | 1)
            || raw[10..12].iter().any(|byte| *byte != 0)
            || component_ordinal >= read_u32(&effect, 32)
            || read_u32(&effect, 28)
                .checked_add(component_ordinal)
                .filter(|expected| *expected == ordinal)
                .is_none()
            || key_column_ref >= counts[6]
            || read_u32(&key, 4) != read_u32(&effect, 12)
            || storage != key[28..32]
            || type_size
                != i16::from_le_bytes(key[36..38].try_into().expect("fixed key-column size"))
            || source_column == ABSENT_U32
            || value_offset != next_value
            || (validity == 1 && value_bytes != 0)
            || (validity == 0 && !valid_value_length(storage[0], value_bytes))
            || !valid_sql_storage(storage, declared_oid, type_size)
            || raw[46..48].iter().any(|byte| *byte != 0)
            || raw[48..80]
                != typed_value_digest(
                    framing,
                    storage,
                    declared_oid,
                    type_size,
                    validity,
                    value_offset,
                    value_bytes,
                    value_arena_bytes,
                )?
            || raw[80..112]
                != exact_v2_digest(
                    b"gpu-db/write001/s7-typed-key-component/v2",
                    &[&raw[..80], &[0; 32], &raw[112..128], &key[72..104]],
                )
            || raw[112..128].iter().any(|byte| *byte != 0)
            || previous
                .is_some_and(|prior: (u32, u8, u32)| prior >= (effect_ref, side, component_ordinal))
        {
            return Err(error(
                "S7 typed-key-component scalar/order/value/digest form is invalid",
            ));
        }
        if storage[0] == 6 && validity == 0 {
            text_bytes = text_bytes
                .checked_add(u64::from(value_bytes))
                .ok_or_else(|| error("S7 typed-key text bytes overflow"))?;
            text_slots = text_slots
                .checked_add(u64::from(value_bytes != 0))
                .ok_or_else(|| error("S7 typed-key text slots overflow"))?;
        }
        next_value = end;
        previous = Some((effect_ref, side, component_ordinal));
    }
    if next_value != value_arena_bytes {
        return Err(error("S7 typed-key values do not exactly fill their arena"));
    }
    Ok(KeyValueMeasure {
        text_bytes,
        text_slots,
    })
}

fn valid_value_length(storage_tag: u8, bytes: u32) -> bool {
    match storage_tag {
        1 | 2 | 7 => bytes == 4,
        3 | 8 => bytes == 8,
        4 | 9 => bytes == 16,
        5 => bytes == 1,
        6 => true,
        _ => false,
    }
}

/// Validate the component's logical scalar bytes directly from the arena before a retained
/// owner exists.  In particular, a component digest alone is not evidence that a forged
/// int2/numeric/date/timestamp/bool/text carrier has the shared typed-vector value domain.
fn validate_typed_key_value_domain(
    framing: &DecodedAggregateFraming<'_>,
    storage: [u8; 4],
    offset: u64,
    bytes: u32,
    arena_bytes: u64,
) -> Result<(), EngineError> {
    if !valid_value_length(storage[0], bytes) {
        return Err(error("S7 typed-key value has a noncanonical scalar length"));
    }
    let end = offset
        .checked_add(u64::from(bytes))
        .filter(|end| *end <= arena_bytes)
        .ok_or_else(|| error("S7 typed-key value domain range is out of bounds"))?;
    framing.with_section_reader(6, |reader| {
        let header = reader.exact::<640>()?;
        let value_start = read_u64(&header, 104 + 12 * 16);
        reader.skip(
            value_start
                .checked_add(offset)
                .and_then(|start| start.checked_sub(S7_HEADER_BYTES))
                .ok_or_else(|| error("S7 typed-key value domain source offset is invalid"))?,
        )?;
        match storage[0] {
            1 | 2 | 7 => {
                let raw = reader.exact::<4>()?;
                let value = i32::from_le_bytes(raw);
                if (storage[0] == 1 && i16::try_from(value).is_err())
                    || (storage[0] == 7
                        && gpu_db_sql::datetime::validate_date_carrier(value).is_err())
                {
                    return Err(error("S7 typed-key i32 scalar is outside its SQL domain"));
                }
            }
            3 | 8 => {
                let raw = reader.exact::<8>()?;
                let value = i64::from_le_bytes(raw);
                if storage[0] == 8
                    && gpu_db_sql::datetime::validate_timestamp_carrier(value).is_err()
                {
                    return Err(error("S7 typed-key timestamp is outside its SQL domain"));
                }
            }
            4 => {
                let raw = reader.exact::<16>()?;
                let value = i128::from_le_bytes(raw);
                if crate::numeric_exceeds_precision(value, storage[1]) {
                    return Err(error("S7 typed-key numeric exceeds its declared precision"));
                }
            }
            5 => {
                if reader.u8()? > 1 {
                    return Err(error("S7 typed-key bool is not canonical"));
                }
            }
            6 => validate_typed_key_text_utf8(reader, u64::from(bytes))?,
            9 => {
                reader.skip(16)?;
            }
            _ => return Err(error("S7 typed-key storage tag is invalid")),
        }
        reader.skip(reader.remaining())?;
        Ok(())
    })?;
    debug_assert_eq!(end, offset + u64::from(bytes));
    Ok(())
}

fn validate_typed_key_text_utf8(
    reader: &mut super::super::super::codec::DecodedAggregateSectionReader<'_>,
    mut remaining: u64,
) -> Result<(), EngineError> {
    let mut pending = [0_u8; 3];
    let mut pending_len = 0_usize;
    let mut chunk = [0_u8; 4096];
    let mut joined = [0_u8; 4099];
    while remaining != 0 {
        let take = usize::try_from(remaining.min(chunk.len() as u64))
            .map_err(|_| error("S7 typed-key text chunk is not addressable"))?;
        reader.copy_exact(&mut chunk[..take])?;
        joined[..pending_len].copy_from_slice(&pending[..pending_len]);
        joined[pending_len..pending_len + take].copy_from_slice(&chunk[..take]);
        let input = &joined[..pending_len + take];
        match std::str::from_utf8(input) {
            Ok(_) => pending_len = 0,
            Err(utf8) if utf8.error_len().is_none() => {
                let suffix = &input[utf8.valid_up_to()..];
                if suffix.len() > pending.len() {
                    return Err(error("S7 typed-key text has an invalid UTF-8 prefix"));
                }
                pending[..suffix.len()].copy_from_slice(suffix);
                pending_len = suffix.len();
            }
            Err(_) => return Err(error("S7 typed-key text is not UTF-8")),
        }
        remaining -= take as u64;
    }
    if pending_len != 0 {
        return Err(error(
            "S7 typed-key text ends in an incomplete UTF-8 sequence",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn typed_value_digest(
    framing: &DecodedAggregateFraming<'_>,
    storage: [u8; 4],
    declared_oid: u32,
    type_size: i16,
    validity: u8,
    offset: u64,
    bytes: u32,
    arena_bytes: u64,
) -> Result<[u8; 32], EngineError> {
    let end = offset
        .checked_add(u64::from(bytes))
        .filter(|end| *end <= arena_bytes)
        .ok_or_else(|| error("S7 typed-value digest range is out of bounds"))?;
    let mut digest = begin_v2_digest(b"gpu-db/write001/s7-typed-key-value/v2");
    digest.update(storage);
    digest.update(declared_oid.to_le_bytes());
    digest.update(type_size.to_le_bytes());
    digest.update([validity]);
    digest.update(bytes.to_le_bytes());
    framing.with_section_reader(6, |reader| {
        let header = reader.exact::<640>()?;
        let value_start = read_u64(&header, 104 + 12 * 16);
        reader.skip(
            value_start
                .checked_add(offset)
                .and_then(|start| start.checked_sub(S7_HEADER_BYTES))
                .ok_or_else(|| error("S7 typed-value absolute source offset is invalid"))?,
        )?;
        let mut remaining = end - offset;
        let mut scratch = [0_u8; 4096];
        while remaining != 0 {
            let take = usize::try_from(remaining.min(scratch.len() as u64))
                .map_err(|_| error("S7 typed-value digest scratch is not addressable"))?;
            reader.copy_exact(&mut scratch[..take])?;
            digest.update(&scratch[..take]);
            remaining -= take as u64;
        }
        reader.skip(reader.remaining())?;
        Ok(())
    })?;
    Ok(digest.finalize().into())
}

fn validate_key_effects(
    framing: &DecodedAggregateFraming<'_>,
    reader: &mut super::super::super::codec::DecodedAggregateSectionReader<'_>,
    counts: &[u32; 12],
) -> Result<(), EngineError> {
    let mut previous = None;
    let mut next_component = 0_u32;
    for ordinal in 0..counts[8] {
        let raw = reader.exact::<192>()?;
        let role = raw[4];
        let action = raw[5];
        let transition = read_u32(&raw, 8);
        let index_ref = read_u32(&raw, 12);
        let dependency = read_u32(&raw, 16);
        let new_start = read_u32(&raw, 28);
        let new_count = read_u32(&raw, 32);
        let arity = read_u32(&raw, 36);
        let participates = raw[43];
        let contains_null = raw[44];
        let source_ordinal = read_u32(&raw, 48);
        let transition_record = s7_fixed_at::<192>(framing, 7, transition, counts[7], 192)?;
        let index = s7_fixed_at::<384>(framing, 5, index_ref, counts[5], 384)?;
        let expected_action = role;
        if read_u32(&raw, 0) != ordinal
            || !(1..=3).contains(&role)
            || action != expected_action
            || raw[6..8].iter().any(|byte| *byte != 0)
            || transition >= counts[7]
            || index_ref >= counts[5]
            || read_u32(&transition_record, 40)
                .checked_add(read_u32(&transition_record, 44))
                .filter(|end| ordinal >= read_u32(&transition_record, 40) && ordinal < *end)
                .is_none()
            || read_u32(&raw, 20) != ABSENT_U32
            || read_u32(&raw, 24) != 0
            || new_start != next_component
            || new_count == 0
            || new_count != arity
            || arity != read_u32(&index, 44)
            || raw[40] != 0
            || raw[41] != 1
            || raw[42] != 1
            || !matches!(participates, 0 | 1)
            || !matches!(contains_null, 0 | 1)
            || raw[45..48].iter().any(|byte| *byte != 0)
            || raw[52..64].iter().any(|byte| *byte != 0)
            || raw[64..96] != [0; 32]
            || raw[96..128] != typed_key_digest(framing, ordinal, new_start, new_count, counts[9])?
            || raw[128..160]
                != key_effect_digest(
                    framing, &raw, index_ref, counts[5], new_start, new_count, counts[9],
                )?
            || raw[160..192].iter().any(|byte| *byte != 0)
            || !valid_effect_dependency(
                framing,
                role,
                participates,
                contains_null,
                dependency,
                ordinal,
                counts[3],
            )?
            || previous.is_some_and(|prior: (u32, u8, u32, u32)| {
                prior >= (transition, role, index_ref, source_ordinal)
            })
        {
            return Err(error(
                "S7 key-effect scalar/order/component/digest form is invalid",
            ));
        }
        next_component = next_component
            .checked_add(new_count)
            .ok_or_else(|| error("S7 key-effect component range overflows"))?;
        previous = Some((transition, role, index_ref, source_ordinal));
    }
    if next_component != counts[9] {
        return Err(error(
            "S7 key-effect component ranges do not exhaust components",
        ));
    }
    Ok(())
}

fn valid_effect_dependency(
    framing: &DecodedAggregateFraming<'_>,
    role: u8,
    participates: u8,
    contains_null: u8,
    dependency: u32,
    effect_ordinal: u32,
    dependency_count: u32,
) -> Result<bool, EngineError> {
    let expected_kind = match role {
        1 => 3,
        2 => 4,
        3 => 6,
        _ => return Ok(false),
    };
    let must_participate = role == 1 || contains_null == 0;
    if participates != u8::from(must_participate) {
        return Ok(false);
    }
    if !must_participate {
        return Ok(dependency == ABSENT_U32);
    }
    if dependency >= dependency_count {
        return Ok(false);
    }
    let token = s7_fixed_at::<224>(framing, 3, dependency, dependency_count, 224)?;
    Ok(token[4] == expected_kind
        && if role == 1 {
            read_u32(&token, 40) == ABSENT_U32
        } else {
            read_u32(&token, 40) == effect_ordinal
        })
}

fn typed_key_digest(
    framing: &DecodedAggregateFraming<'_>,
    effect: u32,
    start: u32,
    count: u32,
    total: u32,
) -> Result<[u8; 32], EngineError> {
    let end = start
        .checked_add(count)
        .filter(|end| *end <= total)
        .ok_or_else(|| error("S7 typed-key component range is out of bounds"))?;
    let mut digest = begin_v2_digest(b"gpu-db/write001/s7-typed-key/v2");
    digest.update(effect.to_le_bytes());
    digest.update([2]);
    digest.update(count.to_le_bytes());
    for ordinal in start..end {
        let component = s7_fixed_at::<128>(framing, 9, ordinal, total, 128)?;
        digest.update(&component[80..112]);
    }
    Ok(digest.finalize().into())
}

fn key_effect_digest(
    framing: &DecodedAggregateFraming<'_>,
    raw: &[u8; 192],
    index_ref: u32,
    total_indexes: u32,
    start: u32,
    count: u32,
    total_components: u32,
) -> Result<[u8; 32], EngineError> {
    let end = start
        .checked_add(count)
        .filter(|end| *end <= total_components)
        .ok_or_else(|| error("S7 key-effect component range is out of bounds"))?;
    let index = s7_fixed_at::<384>(framing, 5, index_ref, total_indexes, 384)?;
    let mut digest = begin_v2_digest(b"gpu-db/write001/s7-key-effect/v2");
    digest.update(&raw[..16]);
    digest.update(ABSENT_U32.to_le_bytes());
    digest.update(&raw[20..128]);
    digest.update([0; 32]);
    digest.update(&raw[160..192]);
    digest.update(&index[304..336]);
    for ordinal in start..end {
        let component = s7_fixed_at::<128>(framing, 9, ordinal, total_components, 128)?;
        digest.update(&component[80..112]);
    }
    Ok(digest.finalize().into())
}

fn validate_transitions(
    framing: &DecodedAggregateFraming<'_>,
    reader: &mut super::super::super::codec::DecodedAggregateSectionReader<'_>,
    counts: &[u32; 12],
) -> Result<(), EngineError> {
    let mut previous = None;
    let mut next_effect = 0_u32;
    for ordinal in 0..counts[7] {
        let raw = reader.exact::<192>()?;
        let table_ref = read_u32(&raw, 4);
        let stable_row_id = read_u64(&raw, 8);
        let source_s4 = read_u32(&raw, 20);
        let source_statement = read_u32(&raw, 24);
        let source_row = read_u32(&raw, 28);
        let image_ref = read_u32(&raw, 32);
        let image_row = read_u32(&raw, 36);
        let effect_start = read_u32(&raw, 40);
        let effect_count = read_u32(&raw, 44);
        let final_writer_statement = read_u32(&raw, 48);
        let rewritten = raw[17] == 1;
        let table = s7_table_at(framing, table_ref)?;
        let s4 = s4_at(framing, source_s4, counts[2])?;
        if read_u32(&raw, 0) != ordinal
            || table_ref >= counts[0]
            || stable_row_id == 0
            || stable_row_id == u64::MAX
            || raw[16] != 1
            || !matches!(raw[17], 0 | 1)
            || raw[18..20].iter().any(|byte| *byte != 0)
            || source_s4 >= counts[2]
            || source_statement >= counts[1]
            || image_ref != table_ref
            || image_row
                != ordinal
                    .checked_sub(read_u32(&table, 88))
                    .ok_or_else(|| error("S7 transition ordinal precedes its table range"))?
            || effect_start != next_effect
            || (!rewritten
                && (final_writer_statement != source_statement || raw[160..192] != [0; 32]))
            || (rewritten
                && (final_writer_statement <= source_statement || raw[160..192] == [0; 32]))
            || raw[52..64].iter().any(|byte| *byte != 0)
            || raw[64..96] != s4[32..64]
            || raw[96..128] == [0; 32]
            || raw[128..160]
                != transition_digest(framing, &raw, effect_start, effect_count, counts[8])?
            || read_u32(&s4, 0) != source_statement
            || read_u32(&s4, 4) != source_row
            || read_u64(&s4, 8) != stable_row_id
            || s4[16] != 1
            || read_u32(&s4, 20) != table_ref
            || read_u32(&s4, 24) != ordinal
            || previous.is_some_and(|prior: (u32, u64)| prior >= (table_ref, stable_row_id))
        {
            return Err(error(
                "S7 transition scalar/order/S4/image/digest closure is invalid",
            ));
        }
        next_effect = next_effect
            .checked_add(effect_count)
            .ok_or_else(|| error("S7 transition key-effect range overflows"))?;
        previous = Some((table_ref, stable_row_id));
    }
    if next_effect != counts[8] {
        return Err(error(
            "S7 transition effect ranges do not exhaust key effects",
        ));
    }
    Ok(())
}

fn transition_digest(
    framing: &DecodedAggregateFraming<'_>,
    raw: &[u8; 192],
    effect_start: u32,
    effect_count: u32,
    total_effects: u32,
) -> Result<[u8; 32], EngineError> {
    let end = effect_start
        .checked_add(effect_count)
        .filter(|end| *end <= total_effects)
        .ok_or_else(|| error("S7 transition key-effect range is out of bounds"))?;
    let mut digest = begin_v2_digest(b"gpu-db/write001/s7-transition/v2");
    digest.update(&raw[..128]);
    digest.update([0; 32]);
    digest.update(&raw[160..192]);
    for ordinal in effect_start..end {
        let effect = s7_fixed_at::<192>(framing, 8, ordinal, total_effects, 192)?;
        digest.update(&effect[128..160]);
    }
    Ok(digest.finalize().into())
}

fn validate_index_key_columns(
    framing: &DecodedAggregateFraming<'_>,
    reader: &mut super::super::super::codec::DecodedAggregateSectionReader<'_>,
    counts: &[u32; 12],
) -> Result<(), EngineError> {
    let mut prior_index_key = None;
    for ordinal in 0..counts[6] {
        let raw = reader.exact::<112>()?;
        let index_ref = read_u32(&raw, 4);
        let key_ordinal = read_u32(&raw, 8);
        let index = s7_fixed_at::<384>(framing, 5, index_ref, counts[5], 384)?;
        let attnum = i16::from_le_bytes(raw[24..26].try_into().expect("fixed key attnum"));
        let storage: [u8; 4] = raw[28..32].try_into().expect("fixed key storage type");
        let declared_oid = read_u32(&raw, 32);
        let type_size = i16::from_le_bytes(raw[36..38].try_into().expect("fixed key type size"));
        if read_u32(&raw, 0) != ordinal
            || index_ref >= counts[5]
            || key_ordinal >= read_u32(&index, 44)
            || read_u32(&index, 40)
                .checked_add(key_ordinal)
                .filter(|expected| *expected == ordinal)
                .is_none()
            || read_u32(&raw, 12) == ABSENT_U32
            || read_u32(&raw, 16) == 0
            || read_u32(&raw, 20) == 0
            || read_u32(&raw, 20) > 0x7fff_ffff
            || attnum <= 0
            || raw[26..28].iter().any(|byte| *byte != 0)
            || !valid_sql_storage(storage, declared_oid, type_size)
            || raw[38..40].iter().any(|byte| *byte != 0)
            || raw[40..72] == [0; 32]
            || raw[72..104]
                != exact_v2_digest(
                    b"gpu-db/write001/s7-index-key-column/v2",
                    &[&raw[..72], &[0; 32], &raw[104..112]],
                )
            || raw[104..112].iter().any(|byte| *byte != 0)
            || prior_index_key.is_some_and(|prior: (u32, u32)| prior >= (index_ref, key_ordinal))
        {
            return Err(error(
                "S7 index-key-column scalar/order/digest form is invalid",
            ));
        }
        prior_index_key = Some((index_ref, key_ordinal));
    }
    Ok(())
}

fn valid_sql_storage(storage: [u8; 4], declared_oid: u32, type_size: i16) -> bool {
    if declared_oid == 0 || storage[3] != 0 {
        return false;
    }
    match storage[0] {
        1 => storage[1] == 0 && storage[2] == 0 && type_size == 2,
        2 | 7 => storage[1] == 0 && storage[2] == 0 && type_size == 4,
        3 | 8 => storage[1] == 0 && storage[2] == 0 && type_size == 8,
        4 => (1..=38).contains(&storage[1]) && storage[2] <= storage[1] && type_size == -1,
        5 => storage[1] == 0 && storage[2] == 0 && type_size == 1,
        6 => storage[1] == 0 && storage[2] == 0 && type_size == -1,
        9 => storage[1] == 0 && storage[2] == 0 && type_size == 16,
        _ => false,
    }
}

fn validate_index_descriptors(
    framing: &DecodedAggregateFraming<'_>,
    reader: &mut super::super::super::codec::DecodedAggregateSectionReader<'_>,
    counts: &[u32; 12],
) -> Result<(), EngineError> {
    let mut previous = None;
    let mut next_key = 0_u32;
    for ordinal in 0..counts[5] {
        let raw = reader.exact::<384>()?;
        let flags = read_u32(&raw, 4);
        let owner_ref = read_u32(&raw, 8);
        let stable_index_id = read_u64(&raw, 16);
        let display_oid = read_u32(&raw, 24);
        let constraint_oid = read_u32(&raw, 28);
        let stable_constraint_id = read_u64(&raw, 32);
        let key_start = read_u32(&raw, 40);
        let key_count = read_u32(&raw, 44);
        let constraint_backed = flags & ((1 << 1) | (1 << 2)) != 0;
        let unique = flags & 1 != 0;
        let primary = flags & (1 << 1) != 0;
        let unique_constraint = flags & (1 << 2) != 0;
        let maintained = flags & (1 << 3) != 0;
        let owner_is_target = owner_ref != ABSENT_U32;
        let owner_is_initially_absent = owner_is_target
            && owner_ref < counts[0]
            && (read_u32(&s7_table_at(framing, owner_ref)?, 4) & 2 != 0);
        // The only other zero index predecessor is an existing table's S3-proven CREATE INDEX.
        // Pass-zero preserves the paired absence form; the retained codec closure binds it to
        // the ordered S3 lifecycle identity before any replay artifact can escape.
        let index_predecessor_is_absent =
            owner_is_initially_absent || (read_u64(&raw, 352) == 0 && raw[240..272] == [0; 32]);
        if read_u32(&raw, 0) != ordinal
            || flags & !0xf != 0
            || (primary && !unique)
            || (unique_constraint && !unique)
            || (maintained != owner_is_target)
            || (owner_is_target && owner_ref >= counts[0])
            || stable_index_id == 0
            || stable_index_id == u64::MAX
            || display_oid == 0
            || display_oid > 0x7fff_ffff
            || (constraint_backed
                && (constraint_oid == 0
                    || constraint_oid > 0x7fff_ffff
                    || stable_constraint_id == 0
                    || stable_constraint_id == u64::MAX
                    || raw[176..208] == [0; 32]))
            || (!constraint_backed
                && (constraint_oid != 0
                    || stable_constraint_id != u64::MAX
                    || raw[176..208] != [0; 32]))
            || key_start != next_key
            || key_count == 0
            || raw[48] != 1
            || raw[49..64].iter().any(|byte| *byte != 0)
            // The first table in a new database inherits catalog epoch zero. Its S7 table
            // block already proves the exact absent predecessor; an owned first-generation
            // descriptor must carry that same epoch rather than fabricating a catalog cut.
            || (read_u64(&raw, 72) == 0 && !owner_is_initially_absent)
            || raw[80..176] == [0; 96]
            || (raw[208..240] == [0; 32] && !owner_is_initially_absent)
            || (raw[240..272] == [0; 32] && !index_predecessor_is_absent)
            || raw[272..336]
                .chunks_exact(32)
                .any(|digest| digest == [0; 32])
            || raw[340..344].iter().any(|byte| *byte != 0)
            || raw[368..384].iter().any(|byte| *byte != 0)
            || (read_u64(&raw, 344) == 0 && !owner_is_initially_absent)
            || read_u64(&raw, 344) == u64::MAX
            || (read_u64(&raw, 352) == 0 && !index_predecessor_is_absent)
            || read_u64(&raw, 352) == u64::MAX
            || read_u64(&raw, 360) == 0
            || read_u64(&raw, 360) == u64::MAX
            || previous.as_ref().is_some_and(|prior: &[u8; 384]| {
                let left = (read_u64(prior, 64), read_u64(prior, 16));
                let right = (read_u64(&raw, 64), stable_index_id);
                left >= right
            })
        {
            return Err(error(
                "S7 index-descriptor scalar/order/reference form is invalid",
            ));
        }
        if owner_is_target {
            let table = s7_table_at(framing, owner_ref)?;
            if read_u64(&raw, 64) != read_u64(&table, 8)
                || read_u32(&raw, 336) != read_u32(&table, 16)
                || read_u64(&raw, 72) != read_u64(&table, 24)
                || raw[80..112] != table[128..160]
                || raw[112..144] == [0; 32]
                || raw[208..240] != table[160..192]
                || read_u64(&raw, 344) != read_u64(&table, 32)
            {
                return Err(error("S7 target index does not bind its table block"));
            }
        }
        let mut descriptor_digest = begin_v2_digest(b"gpu-db/write001/s7-index-descriptor/v2");
        descriptor_digest.update(&raw[..304]);
        descriptor_digest.update([0; 32]);
        descriptor_digest.update(&raw[336..384]);
        for key_ref in key_start
            ..key_start
                .checked_add(key_count)
                .ok_or_else(|| error("S7 index key range overflows"))?
        {
            let key = s7_fixed_at::<112>(framing, 6, key_ref, counts[6], 112)?;
            descriptor_digest.update(&key[72..104]);
        }
        if raw[304..336] != <[u8; 32]>::from(descriptor_digest.finalize()) {
            return Err(error("S7 index-descriptor digest is invalid"));
        }
        next_key = next_key
            .checked_add(key_count)
            .ok_or_else(|| error("S7 index key range overflows"))?;
        previous = Some(raw);
    }
    if next_key != counts[6] {
        return Err(error(
            "S7 index key ranges do not exhaust the key-column directory",
        ));
    }
    Ok(())
}

fn validate_statement_dependency_uses(
    framing: &DecodedAggregateFraming<'_>,
    reader: &mut super::super::super::codec::DecodedAggregateSectionReader<'_>,
    counts: &[u32; 12],
) -> Result<(), EngineError> {
    let mut previous = None;
    for _ in 0..counts[4] {
        let raw = reader.exact::<32>()?;
        let statement = read_u32(&raw, 0);
        let dependency = read_u32(&raw, 4);
        let role = u16::from_le_bytes(raw[8..10].try_into().expect("fixed use role"));
        let source = read_u32(&raw, 12);
        let transition = read_u32(&raw, 16);
        let key_effect = read_u32(&raw, 20);
        let token = s7_fixed_at::<224>(framing, 3, dependency, counts[3], 224)?;
        let token_kind = token[4];
        let expected_token_kind = match role {
            1 => 1,
            2 => 3,
            3 => 4,
            4 => 2,
            5..=11 => role as u8,
            _ => return Err(error("S7 statement-dependency-use role is invalid")),
        };
        let static_role = matches!(role, 1 | 2 | 4 | 5 | 7 | 9 | 10 | 11);
        let equality_role = matches!(role, 3 | 6);
        if statement >= counts[1]
            || dependency >= counts[3]
            || !(1..=11).contains(&role)
            || token_kind != expected_token_kind
            || raw[10..12].iter().any(|byte| *byte != 0)
            || raw[24..32].iter().any(|byte| *byte != 0)
            || (static_role && (transition != ABSENT_U32 || key_effect != ABSENT_U32))
            || (role == 8 && key_effect != ABSENT_U32)
            || (equality_role && ((transition == ABSENT_U32) != (key_effect == ABSENT_U32)))
            || (transition != ABSENT_U32 && transition >= counts[7])
            || (key_effect != ABSENT_U32 && key_effect >= counts[8])
            || previous.as_ref().is_some_and(|prior: &[u8; 32]| {
                use_order_cmp(prior, &raw) != std::cmp::Ordering::Less
            })
        {
            return Err(error(
                "S7 statement-dependency-use scalar/order/reference form is invalid",
            ));
        }
        let _ = source;
        previous = Some(raw);
    }
    Ok(())
}

fn use_order_cmp(left: &[u8; 32], right: &[u8; 32]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let values = [
        (read_u32(left, 0), read_u32(right, 0)),
        (
            u32::from(u16::from_le_bytes(
                left[8..10].try_into().expect("fixed use role"),
            )),
            u32::from(u16::from_le_bytes(
                right[8..10].try_into().expect("fixed use role"),
            )),
        ),
        (read_u32(left, 12), read_u32(right, 12)),
        (read_u32(left, 16), read_u32(right, 16)),
        (read_u32(left, 20), read_u32(right, 20)),
        (read_u32(left, 4), read_u32(right, 4)),
    ];
    for (left, right) in values {
        match left.cmp(&right) {
            Ordering::Equal => {}
            order => return order,
        }
    }
    Ordering::Equal
}

fn validate_dependency_tokens(
    framing: &DecodedAggregateFraming<'_>,
    reader: &mut super::super::super::codec::DecodedAggregateSectionReader<'_>,
    counts: &[u32; 12],
    catalog_epoch: u64,
    aborted: bool,
) -> Result<(), EngineError> {
    if counts[3] < counts[0] {
        return Err(error("S7 has fewer dependency tokens than target tables"));
    }
    let mut previous = None;
    let mut terminal_count = 0_u32;
    for ordinal in 0..counts[3] {
        let raw = reader.exact::<224>()?;
        let kind = raw[4];
        let access = raw[5];
        let flags = u16::from_le_bytes(raw[6..8].try_into().expect("fixed dependency flags"));
        let stable_object_id = read_u64(&raw, 8);
        let display_oid = read_u32(&raw, 16);
        let table_ref = read_u32(&raw, 20);
        let base_generation = read_u64(&raw, 24);
        let floor = read_u64(&raw, 32);
        let key_effect = read_u32(&raw, 40);
        let descriptor = read_u32(&raw, 44);
        // A transaction-created table has no public predecessor generation. Its target token
        // and any maintained/unique descriptor it owns witness that same zero predecessor.
        // Read the authoritative S7 table block rather than creating a second catalog lookup
        // convention here; FK supporting indexes remain bound to their published parents.
        let target_is_initially_absent = kind == 1
            && table_ref < counts[0]
            && (read_u32(&s7_table_at(framing, table_ref)?, 4) & 2 != 0);
        let index_is_initially_absent = matches!(kind, 3 | 4) && descriptor < counts[5] && {
            let index = s7_fixed_at::<384>(framing, 5, descriptor, counts[5], 384)?;
            let owner_ref = read_u32(&index, 8);
            owner_ref == table_ref
                && (owner_ref < counts[0]
                    && (read_u32(&s7_table_at(framing, owner_ref)?, 4) & 2 != 0)
                    // An existing table's S3-authorized CREATE INDEX has the same paired-zero
                    // predecessor form. The retained closure binds that exception to its exact
                    // lifecycle identity before replay may consume this grammar.
                    || (read_u64(&index, 352) == 0 && index[240..272] == [0; 32]))
        };
        // A domain introduced by the same ordered S3 catalog composition as an initially absent
        // target table has no published predecessor either. Its catalog mutation remains the
        // authoritative creation proof; this only admits its zero snapshot generation in the
        // already-authenticated S7 dependency grammar.
        let domain_is_initially_absent = kind == 7
            && table_ref < counts[0]
            && (read_u32(&s7_table_at(framing, table_ref)?, 4) & 2 != 0);
        let absent_predecessor =
            target_is_initially_absent || index_is_initially_absent || domain_is_initially_absent;
        let expected_access = match kind {
            1 | 3 => 3,
            2 | 4 | 5 | 6 | 9 | 10 | 11 => 2,
            7 => 1,
            8 => 4,
            _ => return Err(error("S7 dependency token kind is invalid")),
        };
        let live_effect = flags & 1 != 0;
        let terminal = flags & 2 != 0;
        let descriptor_required = (3..=6).contains(&kind);
        let base_root_zero = kind == 7 || absent_predecessor;
        let schema_zero = kind == 8;
        if read_u32(&raw, 0) != ordinal
            || access != expected_access
            || flags & !3 != 0
            || live_effect && terminal
            || stable_object_id == 0
            || stable_object_id == u64::MAX
            // A zero display OID is reserved for synthesized NOT NULL dependencies only.
            // Check guards always name their catalog object, even when they are terminal.
            || (display_oid == 0 && !matches!(kind, 9 | 11))
            || display_oid > 0x7fff_ffff
            || table_ref >= counts[0]
            || (base_generation == 0 && !absent_predecessor)
            || base_generation == u64::MAX
            || (kind == 8 && floor != 0)
            || (kind != 8 && !absent_predecessor && floor == 0)
            || read_u64(&raw, 48) != catalog_epoch
            || (live_effect
                && (!matches!(kind, 4 | 6) || key_effect >= counts[8]))
            || (!live_effect && key_effect != ABSENT_U32)
            || (matches!(kind, 4 | 6) && !live_effect && !terminal)
            || (terminal && !matches!(kind, 4 | 6 | 9 | 10 | 11))
            || (descriptor_required && descriptor >= counts[5])
            || (!descriptor_required && descriptor != ABSENT_U32)
            || raw[56..64].iter().any(|byte| *byte != 0)
            || (base_root_zero && raw[96..128] != [0; 32])
            || (!base_root_zero && raw[96..128] == [0; 32])
            || (schema_zero && raw[64..96] != [0; 32])
            || (!schema_zero && raw[64..96] == [0; 32])
            || raw[128..160] == [0; 32]
            || raw[160..192] == [0; 32]
            || raw[192..224] == [0; 32]
            || previous
                .as_ref()
                .is_some_and(|prior: &[u8; 224]| dependency_identity_cmp(prior, &raw) != std::cmp::Ordering::Less)
        {
            return Err(error(
                "S7 dependency-token scalar/order/reference form is invalid",
            ));
        }
        validate_dependency_token_digests(framing, &raw, counts)?;
        terminal_count = terminal_count
            .checked_add(u32::from(terminal))
            .ok_or_else(|| error("S7 terminal token count overflows"))?;
        previous = Some(raw);
    }
    if (terminal_count != 0) != aborted || terminal_count > 1 {
        return Err(error(
            "S7 terminal-error dependency-token cardinality is invalid",
        ));
    }
    Ok(())
}

/// Recompute the complete dependency identity from fixed bounded readers.  Token bytes name
/// directory references but never get to self-authorize their opaque digests: every referenced
/// descriptor/effect digest is reread and the published-sequence form rereads its exact S5 body.
fn validate_dependency_token_digests(
    framing: &DecodedAggregateFraming<'_>,
    raw: &[u8; 224],
    counts: &[u32; 12],
) -> Result<(), EngineError> {
    let kind = raw[4];
    let descriptor_ref = read_u32(raw, 44);
    let key_effect_ref = read_u32(raw, 40);
    let descriptor_digest = if descriptor_ref == ABSENT_U32 {
        [0; 32]
    } else {
        s7_fixed_at::<384>(framing, 5, descriptor_ref, counts[5], 384)?[304..336]
            .try_into()
            .expect("fixed index descriptor digest")
    };
    let key_effect_digest = if key_effect_ref == ABSENT_U32 {
        [0; 32]
    } else {
        s7_fixed_at::<192>(framing, 8, key_effect_ref, counts[8], 192)?[128..160]
            .try_into()
            .expect("fixed key-effect digest")
    };
    let identity = match kind {
        1 | 2 => table_dependency_identity(raw),
        3..=6 => index_dependency_identity(raw, descriptor_digest, key_effect_digest),
        7 => domain_dependency_identity(raw),
        8 => published_sequence_dependency_identity(framing, raw)?,
        9..=11 => constraint_dependency_identity(framing, raw)?,
        _ => return Err(error("S7 dependency-token digest has an invalid kind")),
    };
    let token = exact_v2_digest(
        b"gpu-db/write001/s7-dependency-token/v2",
        &[
            &raw[..192],
            &[0; 32],
            &descriptor_digest,
            &key_effect_digest,
        ],
    );
    if raw[160..192] != identity || raw[192..224] != token {
        return Err(error(
            "S7 dependency-token identity or digest does not match its exact preimage",
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn validate_dependency_token_digests_for_test(
    framing: &DecodedAggregateFraming<'_>,
    raw: &[u8; 224],
) -> Result<(), EngineError> {
    let counts = framing.with_section_reader(6, |reader| {
        let header = reader.exact::<640>()?;
        reader.skip(reader.remaining())?;
        Ok([
            read_u32(&header, 40),
            read_u32(&header, 44),
            read_u32(&header, 48),
            read_u32(&header, 52),
            read_u32(&header, 56),
            read_u32(&header, 60),
            read_u32(&header, 64),
            read_u32(&header, 68),
            read_u32(&header, 72),
            read_u32(&header, 76),
            read_u32(&header, 80),
            read_u32(&header, 84),
        ])
    })?;
    validate_dependency_token_digests(framing, raw, &counts)
}

fn table_dependency_identity(raw: &[u8; 224]) -> [u8; 32] {
    let kind = [raw[4]];
    let stable_object_id = read_u64(raw, 8).to_le_bytes();
    let display_oid = read_u32(raw, 16).to_le_bytes();
    let catalog_epoch = read_u64(raw, 48).to_le_bytes();
    let base_generation = read_u64(raw, 24).to_le_bytes();
    exact_v2_digest(
        b"gpu-db/write001/s7-table-object/v2",
        &[
            &kind,
            &stable_object_id,
            &display_oid,
            &catalog_epoch,
            &base_generation,
            &raw[64..96],
            &raw[96..128],
            &raw[128..160],
        ],
    )
}

fn index_dependency_identity(
    raw: &[u8; 224],
    descriptor_digest: [u8; 32],
    key_effect_digest: [u8; 32],
) -> [u8; 32] {
    let kind = [raw[4]];
    let stable_object_id = read_u64(raw, 8).to_le_bytes();
    let display_oid = read_u32(raw, 16).to_le_bytes();
    let catalog_epoch = read_u64(raw, 48).to_le_bytes();
    let base_generation = read_u64(raw, 24).to_le_bytes();
    exact_v2_digest(
        b"gpu-db/write001/s7-index-object/v2",
        &[
            &kind,
            &stable_object_id,
            &display_oid,
            &catalog_epoch,
            &base_generation,
            &raw[64..96],
            &raw[96..128],
            &raw[128..160],
            &descriptor_digest,
            &key_effect_digest,
        ],
    )
}

fn domain_dependency_identity(raw: &[u8; 224]) -> [u8; 32] {
    let stable_object_id = read_u64(raw, 8).to_le_bytes();
    let display_oid = read_u32(raw, 16).to_le_bytes();
    let catalog_epoch = read_u64(raw, 48).to_le_bytes();
    let base_generation = read_u64(raw, 24).to_le_bytes();
    exact_v2_digest(
        b"gpu-db/write001/s7-domain-object/v2",
        &[
            &stable_object_id,
            &display_oid,
            &catalog_epoch,
            &base_generation,
            &raw[64..96],
            &raw[128..160],
        ],
    )
}

fn constraint_dependency_identity(
    framing: &DecodedAggregateFraming<'_>,
    raw: &[u8; 224],
) -> Result<[u8; 32], EngineError> {
    let table = s7_table_at(framing, read_u32(raw, 20))?;
    let kind = [raw[4]];
    let stable_object_id = read_u64(raw, 8).to_le_bytes();
    let display_oid = read_u32(raw, 16).to_le_bytes();
    let target_stable_table_id = read_u64(&table, 8).to_le_bytes();
    let catalog_epoch = read_u64(raw, 48).to_le_bytes();
    let base_generation = read_u64(raw, 24).to_le_bytes();
    Ok(exact_v2_digest(
        b"gpu-db/write001/s7-constraint-object/v2",
        &[
            &kind,
            &stable_object_id,
            &display_oid,
            &target_stable_table_id,
            &catalog_epoch,
            &base_generation,
            &raw[64..96],
            &raw[96..128],
            &raw[128..160],
        ],
    ))
}

fn published_sequence_dependency_identity(
    framing: &DecodedAggregateFraming<'_>,
    raw: &[u8; 224],
) -> Result<[u8; 32], EngineError> {
    let expected_transition = read_u64(raw, 24);
    let expected_sequence_oid = read_u32(raw, 16);
    let expected_body_digest: [u8; 32] =
        raw[96..128].try_into().expect("fixed sequence body digest");
    let mut matching_body = None;
    let mut matches = 0_u32;
    framing.with_section_reader(4, |reader| {
        while !reader.done() {
            let prefix = reader.exact::<52>()?;
            let kind = prefix[8];
            let body_bytes = read_u32(&prefix, 16);
            if kind != 1 {
                reader.skip(u64::from(body_bytes))?;
                continue;
            }
            let expected_reference_bytes = crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES as u32;
            let expected_with_restart = expected_reference_bytes
                + crate::typed_insert_aggregate::semantics_v2::sequence_terminal::SEQUENCE_RESTART_TAIL_BYTES as u32;
            if body_bytes != expected_reference_bytes && body_bytes != expected_with_restart {
                return Err(error("S5 published sequence body length drifted"));
            }
            let body = reader.exact::<{ crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES }>()?;
            if body_bytes == expected_with_restart {
                let tail = reader.exact::<{
                    crate::typed_insert_aggregate::semantics_v2::sequence_terminal::SEQUENCE_RESTART_TAIL_BYTES
                }>()?;
                crate::typed_insert_aggregate::semantics_v2::sequence_terminal::decode(&tail)
                    .map_err(|_| error("S5 terminal restart fails exact reread"))?;
            }
            let reference = crate::decode_sequence_value_reference_exact(&body)
                .map_err(|_| error("S5 published sequence body fails exact reread"))?;
            let reference_body_digest = gpu_db_wal::canonical_request_digest(&body);
            if reference.transition_txn_id == expected_transition
                && reference.sequence_oid == expected_sequence_oid
                && reference_body_digest == expected_body_digest
            {
                matches = matches
                    .checked_add(1)
                    .ok_or_else(|| error("S5 published sequence match count overflows"))?;
                matching_body = Some(body);
            }
        }
        Ok(())
    })?;
    let body = matching_body
        .filter(|_| matches == 1)
        .ok_or_else(|| error("S7 published-sequence token does not name one exact S5 reference"))?;
    let stable_object_id = read_u64(raw, 8).to_le_bytes();
    let display_oid = expected_sequence_oid.to_le_bytes();
    let catalog_epoch = read_u64(raw, 48).to_le_bytes();
    let transition = expected_transition.to_le_bytes();
    Ok(exact_v2_digest(
        b"gpu-db/write001/s7-published-sequence/v2",
        &[
            &stable_object_id,
            &display_oid,
            &catalog_epoch,
            &transition,
            &raw[128..160],
            &expected_body_digest,
            &body,
        ],
    ))
}

fn dependency_identity_cmp(left: &[u8; 224], right: &[u8; 224]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let terminal_left =
        u16::from_le_bytes(left[6..8].try_into().expect("fixed token flags")) & 2 != 0;
    let terminal_right =
        u16::from_le_bytes(right[6..8].try_into().expect("fixed token flags")) & 2 != 0;
    let scalars = [
        (u64::from(left[4]), u64::from(right[4])),
        (u64::from(terminal_left), u64::from(terminal_right)),
        (read_u64(left, 8), read_u64(right, 8)),
        (
            u64::from(read_u32(left, 16)),
            u64::from(read_u32(right, 16)),
        ),
        (
            u64::from(read_u32(left, 20)),
            u64::from(read_u32(right, 20)),
        ),
        (read_u64(left, 48), read_u64(right, 48)),
        (read_u64(left, 24), read_u64(right, 24)),
        (
            u64::from(read_u32(left, 40)),
            u64::from(read_u32(right, 40)),
        ),
        (
            u64::from(read_u32(left, 44)),
            u64::from(read_u32(right, 44)),
        ),
    ];
    for (left, right) in scalars {
        match left.cmp(&right) {
            Ordering::Equal => {}
            order => return order,
        }
    }
    for (start, end) in [(64, 96), (96, 128), (128, 160), (160, 192)] {
        match left[start..end].cmp(&right[start..end]) {
            Ordering::Equal => {}
            order => return order,
        }
    }
    Ordering::Equal
}

fn validate_statement_resolutions(
    framing: &DecodedAggregateFraming<'_>,
    reader: &mut super::super::super::codec::DecodedAggregateSectionReader<'_>,
    counts: &[u32; 12],
    commit_sequence: u64,
) -> Result<(), EngineError> {
    framing.with_section_reader(1, |s2_reader| {
        let mut next_s4 = 0_u32;
        let mut next_s5 = 0_u32;
        let mut next_use = 0_u32;
        let mut next_projection = 0_u32;
        for ordinal in 0..counts[1] {
            let raw = reader.exact::<320>()?;
            let s1 = s1_at(framing, ordinal, counts[1])?;
            let s6 = s6_at(framing, ordinal, counts[1])?;
            let input_rows = read_u32(&s1, 12);
            let s4_start = read_u32(&raw, 24);
            let s4_count = read_u32(&raw, 28);
            let s5_start = read_u32(&raw, 32);
            let s5_count = read_u32(&raw, 36);
            let use_start = read_u32(&raw, 40);
            let use_count = read_u32(&raw, 44);
            let projection_start = read_u32(&raw, 48);
            let projection_count = read_u32(&raw, 52);
            let flags = read_u32(&raw, 20);
            let target_is_initially_absent =
                read_u32(&s7_table_at(framing, read_u32(&raw, 16))?, 4) & 2 != 0;
            let s6_flags = u16::from_le_bytes(s6[10..12].try_into().expect("fixed S6 flags"));
            let outcome = gpu_db_wal::decode_canonical_outcome_exact(&s6[44..136])
                .map_err(|_| error("S7 resolution S6 outcome is invalid"))?;
            let s2 = next_s2_record_digest(s2_reader)?;
            let s6_digest = exact_v2_digest(b"gpu-db/write001/s7-s6-entry/v2", &[&s6]);
            let (surviving_rows, affected_rows) =
                s4_range_counts(framing, s4_start, s4_count, counts[2])?;
            if read_u32(&raw, 0) != ordinal
                || read_u32(&raw, 4) != ordinal
                || read_u32(&raw, 8) != ordinal
                || read_u32(&raw, 12) != ordinal
                || read_u32(&raw, 16) >= counts[0]
                || flags & !3 != 0
                || (flags & 2 != 0 && flags & 1 == 0)
                || flags != u32::from(s6_flags)
                || s4_start != next_s4
                || s4_count != input_rows
                || s5_start != next_s5
                || use_start != next_use
                || projection_start != next_projection
                || read_u32(&raw, 56) != input_rows
                || read_u32(&raw, 60) != surviving_rows
                || read_u64(&raw, 64) != affected_rows
                || read_u64(&raw, 64) != outcome.affected_rows
                || (!target_is_initially_absent
                    && (read_u64(&raw, 72) == 0 || read_u64(&raw, 72) >= commit_sequence))
                || (target_is_initially_absent && read_u64(&raw, 72) > commit_sequence)
                || read_u32(&raw, 80) != s2.bytes
                || raw[96..128] != s1[16..48]
                || raw[128..160] != s1[48..80]
                || raw[160..192] != s2.digest
                || raw[224..256] != s1[80..112]
                || raw[256..288] != s1[112..144]
                || raw[288..320] != s6_digest
            {
                return Err(error(
                    "S7 statement-resolution scalar/source/digest closure is invalid",
                ));
            }
            let has_returning = flags & 1 != 0;
            // This is the canonical S2 *layout* digest, not an S8 result digest.  A zero-column
            // RETURNING layout can still have a nonzero digest, while its S7 projection range
            // and response-retained flag remain absent.
            if (projection_count != 0) != has_returning
                || (has_returning && raw[192..224] == [0; 32])
            {
                return Err(error("S7 resolution RETURNING geometry is invalid"));
            }
            let terminal_token = read_u32(&raw, 84);
            let terminal_row = read_u32(&raw, 88);
            let terminal_source = read_u32(&raw, 92);
            match outcome.kind {
                gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
                    if terminal_token == ABSENT_U32
                        && terminal_row == ABSENT_U32
                        && terminal_source == ABSENT_U32 => {}
                gpu_db_wal::CanonicalOutcomeKind::AbortError
                    if terminal_token < counts[3]
                        && terminal_row < input_rows
                        && terminal_source != ABSENT_U32 => {}
                _ => return Err(error("S7 resolution terminal-error form is invalid")),
            }
            next_s4 = next_s4
                .checked_add(s4_count)
                .ok_or_else(|| error("S7 resolution S4 range overflows"))?;
            next_s5 = next_s5
                .checked_add(s5_count)
                .ok_or_else(|| error("S7 resolution S5 range overflows"))?;
            next_use = next_use
                .checked_add(use_count)
                .ok_or_else(|| error("S7 resolution dependency-use range overflows"))?;
            next_projection = next_projection
                .checked_add(projection_count)
                .ok_or_else(|| error("S7 resolution projection range overflows"))?;
        }
        if s2_reader.remaining() != 0
            || next_s4 != counts[2]
            || next_s5 != framing.sections()[4].entry_count
            || next_use != counts[4]
            || next_projection != counts[10]
        {
            return Err(error("S7 statement-resolution range coverage is invalid"));
        }
        Ok(())
    })
}

#[derive(Clone, Copy)]
struct S2RecordDigest {
    bytes: u32,
    digest: [u8; 32],
}

fn next_s2_record_digest(
    reader: &mut super::super::super::codec::DecodedAggregateSectionReader<'_>,
) -> Result<S2RecordDigest, EngineError> {
    let bytes = reader.u32()?;
    if bytes == 0 || u64::from(bytes) > reader.remaining() {
        return Err(error("S2 exact-record range is invalid"));
    }
    let mut digest = begin_v2_digest(b"gpu-db/write001/s7-s2-record/v2");
    digest.update(bytes.to_le_bytes());
    let mut scratch = [0_u8; 4096];
    let mut remaining = u64::from(bytes);
    while remaining != 0 {
        let take = usize::try_from(remaining.min(scratch.len() as u64))
            .map_err(|_| error("S2 record digest scratch is not addressable"))?;
        reader.copy_exact(&mut scratch[..take])?;
        digest.update(&scratch[..take]);
        remaining -= take as u64;
    }
    Ok(S2RecordDigest {
        bytes,
        digest: digest.finalize().into(),
    })
}

/// The sorted table-disposition directory is the allocation-free duplicate/missing proof for S4:
/// each table names every row ID in its exact allocator interval once, and each entry must echo
/// the one S4 row at its absolute reference.  No attacker-sized seen bitmap is needed.
fn validate_table_dispositions(
    framing: &DecodedAggregateFraming<'_>,
    reader: &mut super::super::super::codec::DecodedAggregateSectionReader<'_>,
    counts: &[u32; 12],
) -> Result<(), EngineError> {
    for table_ref in 0..counts[0] {
        let table = s7_table_at(framing, table_ref)?;
        let allocator_before = read_u64(&table, 48);
        let disposition_count = read_u32(&table, 84);
        for within_table in 0..disposition_count {
            let raw = reader.exact::<32>()?;
            let s4_ref = read_u32(&raw, 4);
            let s4 = s4_at(framing, s4_ref, counts[2])?;
            let expected_row_id = allocator_before
                .checked_add(u64::from(within_table))
                .ok_or_else(|| error("S7 table-local disposition row ID overflows"))?;
            if read_u32(&raw, 0) != table_ref
                || s4_ref >= counts[2]
                || raw[25..32].iter().any(|byte| *byte != 0)
                || read_u64(&raw, 8) != expected_row_id
                || read_u32(&raw, 16) != read_u32(&s4, 0)
                || read_u32(&raw, 20) != read_u32(&s4, 4)
                || raw[24] != s4[16]
                || read_u64(&s4, 8) != expected_row_id
                || read_u32(&s4, 20) != table_ref
            {
                return Err(error(
                    "S7 table-disposition/S4 table-local bijection is invalid",
                ));
            }
        }
    }
    Ok(())
}

/// Table blocks are the sole table-local wire range description and defensive cross-check in
/// v2. They never authorize or advance allocation: the independently durable published
/// allocator lease is the sole allocator authority. This pass retains no table array; contiguous
/// directory ownership and the root-descriptor preimage are checked in stable-table order.
fn validate_table_blocks_and_root(
    reader: &mut super::super::super::codec::DecodedAggregateSectionReader<'_>,
    header: &[u8; S7_HEADER_BYTES as usize],
    counts: &[u32; 12],
) -> Result<(), EngineError> {
    let catalog_before_epoch = read_u64(header, 328);
    let catalog_after_epoch = read_u64(header, 336);
    if catalog_after_epoch == 0 || header[344..376] == [0; 32] || header[376..408] == [0; 32] {
        return Err(error("S7 catalog before/after echoes are invalid"));
    }
    let mut root = begin_v2_digest(b"gpu-db/write001/s7-root-descriptor/v2");
    root.update(&header[30..32]);
    root.update(&header[328..536]);
    root.update(&header[40..44]);

    let mut prior_table_id = None;
    let mut next_disposition = 0_u32;
    let mut next_transition = 0_u32;
    let mut next_key_effect = 0_u32;
    let mut all_tables_are_initially_absent = true;
    for ordinal in 0..counts[0] {
        let raw = reader.exact::<384>()?;
        let table_ref = read_u32(&raw, 0);
        let table_flags = read_u32(&raw, 4);
        let resets_existing_rows = table_flags & 1 != 0;
        let initial_table_absent = table_flags & 2 != 0;
        all_tables_are_initially_absent &= initial_table_absent;
        let stable_table_id = read_u64(&raw, 8);
        let display_oid = read_u32(&raw, 16);
        let target_dependency = read_u32(&raw, 20);
        let data_generation_before = read_u64(&raw, 32);
        let data_generation_after = read_u64(&raw, 40);
        let allocator_before = read_u64(&raw, 48);
        let allocator_high_water = read_u64(&raw, 56);
        let initial_rows = read_u64(&raw, 64);
        let final_rows = read_u64(&raw, 72);
        let disposition_start = read_u32(&raw, 80);
        let disposition_count = read_u32(&raw, 84);
        let transition_start = read_u32(&raw, 88);
        let transition_count = read_u32(&raw, 92);
        let key_effect_start = read_u32(&raw, 104);
        let key_effect_count = read_u32(&raw, 108);
        if table_ref != ordinal
            || table_flags & !3 != 0
            || (initial_table_absent && resets_existing_rows)
            || stable_table_id == 0
            || stable_table_id == u64::MAX
            || prior_table_id.is_some_and(|prior| prior >= stable_table_id)
            || display_oid == 0
            || display_oid > 0x7fff_ffff
            || target_dependency >= counts[3]
            || read_u64(&raw, 24) != catalog_before_epoch
            || (!initial_table_absent && data_generation_before == 0)
            || (initial_table_absent && data_generation_before != 0)
            || data_generation_before == u64::MAX
            || data_generation_after == 0
            || data_generation_after == u64::MAX
            || allocator_before == 0
            || allocator_before == u64::MAX
            || allocator_high_water <= allocator_before
            || disposition_start != next_disposition
            || disposition_count == 0
            || transition_start != next_transition
            || key_effect_start != next_key_effect
            || read_u32(&raw, 112) != table_ref
            || read_u32(&raw, 116) == 0
            || raw[120..128].iter().any(|byte| *byte != 0)
            || raw[128..160] == [0; 32]
            || (!initial_table_absent && raw[160..192] == [0; 32])
            || (initial_table_absent && raw[160..192] != [0; 32])
            || raw[192..384]
                .chunks_exact(32)
                .any(|digest| digest == [0; 32])
        {
            return Err(error("S7 table-block scalar/order/root fields are invalid"));
        }
        let allocator_count = allocator_high_water
            .checked_sub(allocator_before)
            .ok_or_else(|| error("S7 table allocator range underflows"))?;
        if allocator_count != u64::from(disposition_count)
            || (initial_table_absent && initial_rows != 0)
            || final_rows
                != if resets_existing_rows {
                    u64::from(transition_count)
                } else {
                    initial_rows
                        .checked_add(u64::from(transition_count))
                        .ok_or_else(|| error("S7 table final-row count overflows"))?
                }
            || (transition_count == 0
                && (data_generation_after != data_generation_before
                    || raw[160..192] != raw[192..224]))
            || (transition_count != 0
                && (data_generation_after <= data_generation_before
                    || raw[160..192] == raw[192..224]))
        {
            return Err(error(
                "S7 table generation/allocator/row-count closure is invalid",
            ));
        }
        next_disposition = next_disposition
            .checked_add(disposition_count)
            .ok_or_else(|| error("S7 table disposition range overflows"))?;
        next_transition = next_transition
            .checked_add(transition_count)
            .ok_or_else(|| error("S7 table transition range overflows"))?;
        next_key_effect = next_key_effect
            .checked_add(key_effect_count)
            .ok_or_else(|| error("S7 table key-effect range overflows"))?;
        root.update(&raw[0..4]);
        root.update(&raw[4..8]);
        root.update(&raw[8..16]);
        root.update(&raw[32..48]);
        root.update(&raw[160..224]);
        root.update(&raw[352..384]);
        prior_table_id = Some(stable_table_id);
    }
    if next_disposition != counts[2]
        || next_transition != counts[7]
        || next_key_effect != counts[8]
        || (catalog_before_epoch == 0 && !all_tables_are_initially_absent)
        || <[u8; 32]>::from(root.finalize()) != header[536..568]
    {
        return Err(error(
            "S7 table ranges or root-descriptor digest do not close",
        ));
    }
    Ok(())
}

fn begin_v2_digest(domain: &[u8]) -> Sha256 {
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest
}

fn exact_v2_digest(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut digest = begin_v2_digest(domain);
    for field in fields {
        digest.update(field);
    }
    digest.finalize().into()
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("fixed u32"))
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("fixed u64"))
}

fn checked_mul(left: u64, right: u64) -> Result<u64, EngineError> {
    left.checked_mul(right)
        .ok_or_else(|| error("pass-zero multiplication overflows"))
}

fn error(message: &str) -> EngineError {
    EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v2 pass zero: {message}"
    ))
}
