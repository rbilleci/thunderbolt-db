//! Exact S8 directories, digests, and local S1--S7/STATUS closure.

use super::source::{
    digest_at, error, fixed_at, s4_at, s6_at, s7_fixed_at, u32_at, u64_at, S8ImageSource,
};
use crate::typed_insert_aggregate::{AGGREGATE_FLAG_RETAINED_RESPONSE, AGGREGATE_FLAG_RETURNING};
use crate::typed_insert_batch::{measure_decoded_typed_image_from_source, TypedImageReadAt};
use crate::EngineError;
use sha2::{Digest, Sha256};

const S8_HEADER_BYTES: u64 = 256;
const S8_ARTIFACT_BYTES: u64 = 288;
const S8_SELECTION_BYTES: u64 = 32;
const S6_FLAG_HAS_RETURNING: u16 = 1;
const S6_FLAG_RESPONSE_RETAINED: u16 = 2;
const S7_FLAG_HAS_RETURNING: u32 = 1;
const S7_FLAG_RESPONSE_RETAINED: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::typed_insert_aggregate::semantics_v2) struct S8ResponseIdentity {
    pub(in crate::typed_insert_aggregate::semantics_v2) present: bool,
    pub(in crate::typed_insert_aggregate::semantics_v2) aggregate_flags: u32,
    pub(in crate::typed_insert_aggregate::semantics_v2) stable_transaction_id: u64,
    pub(in crate::typed_insert_aggregate::semantics_v2) request_digest: [u8; 32],
    pub(in crate::typed_insert_aggregate::semantics_v2) s6_section_root: [u8; 32],
    pub(in crate::typed_insert_aggregate::semantics_v2) s7_section_root: [u8; 32],
    pub(in crate::typed_insert_aggregate::semantics_v2) s8_section_root: [u8; 32],
    pub(in crate::typed_insert_aggregate::semantics_v2) response_root: [u8; 32],
    pub(in crate::typed_insert_aggregate::semantics_v2) status_artifact_count: u32,
    pub(in crate::typed_insert_aggregate::semantics_v2) retention_deadline: u64,
    pub(in crate::typed_insert_aggregate::semantics_v2) total_bytes: u64,
    pub(in crate::typed_insert_aggregate::semantics_v2) artifact_count: u32,
    pub(in crate::typed_insert_aggregate::semantics_v2) selection_count: u32,
    pub(in crate::typed_insert_aggregate::semantics_v2) image_arena_bytes: u64,
    pub(in crate::typed_insert_aggregate::semantics_v2) payload_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::typed_insert_aggregate::semantics_v2) struct S8PassZeroMeasure {
    pub(in crate::typed_insert_aggregate::semantics_v2) identity: S8ResponseIdentity,
    pub(in crate::typed_insert_aggregate::semantics_v2) image_persistent_bytes: u64,
    pub(in crate::typed_insert_aggregate::semantics_v2) image_persistent_slots: u64,
    pub(in crate::typed_insert_aggregate::semantics_v2) maximum_scratch_bytes: u64,
    pub(in crate::typed_insert_aggregate::semantics_v2) maximum_scratch_slots: u64,
}

#[derive(Clone, Copy)]
struct Header {
    total_bytes: u64,
    artifact_count: u32,
    selection_count: u32,
    image_arena_bytes: u64,
    artifact_offset: u64,
    selection_offset: u64,
    image_offset: u64,
    payload_digest: [u8; 32],
}

#[derive(Clone, Copy)]
struct ArtifactCoordinates {
    artifact_ref: u32,
    statement: u32,
    selection_start: u32,
    selection_count: u32,
    projection_start: u32,
    projection_count: u32,
    s4_start: u32,
    s4_count: u32,
}

pub(super) fn measure(
    framing: &crate::typed_insert_aggregate::codec::DecodedAggregateFraming<'_>,
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    s7_counts: &[u32; 12],
    _terminal_abort: Option<u32>,
    aggregate_retained_response: bool,
) -> Result<S8PassZeroMeasure, EngineError> {
    let section = &framing.sections()[7];
    if section.entry_count == 0 || section.payload_bytes == 0 {
        if section.entry_count != 0 || section.payload_bytes != 0 || aggregate_retained_response {
            return Err(error(
                "S8 absent form or aggregate retained-response flag is invalid",
            ));
        }
        validate_statement_and_status_closure(framing, s7_counts, None, 0, false)?;
        return Ok(S8PassZeroMeasure {
            identity: S8ResponseIdentity {
                present: false,
                aggregate_flags: framing.header_scalars().flags,
                stable_transaction_id: outer.stable_transaction_id,
                request_digest: outer.request_digest,
                s6_section_root: framing.section_roots()[5],
                s7_section_root: framing.section_roots()[6],
                s8_section_root: framing.section_roots()[7],
                response_root: framing.status().response_root,
                status_artifact_count: framing.status().response_artifact_count,
                retention_deadline: framing.status().retention_deadline,
                total_bytes: 0,
                artifact_count: 0,
                selection_count: 0,
                image_arena_bytes: 0,
                payload_digest: [0; 32],
            },
            image_persistent_bytes: 0,
            image_persistent_slots: 0,
            maximum_scratch_bytes: 0,
            maximum_scratch_slots: 0,
        });
    }
    if !aggregate_retained_response {
        return Err(error(
            "nonempty S8 requires the aggregate retained-response flag",
        ));
    }
    let raw = fixed_at::<256>(framing, 7, 0, "header")?;
    let header = validate_header(framing, outer, s7_counts, &raw)?;
    if section.entry_count != header.artifact_count || header.artifact_count == 0 {
        return Err(error(
            "S8 section entry count is not its nonempty artifact count",
        ));
    }
    validate_payload_digest(framing, &raw, header)?;

    let mut image_cursor = 0_u64;
    let mut selection_cursor = 0_u32;
    let mut image_persistent_bytes = 0_u64;
    let mut image_persistent_slots = 0_u64;
    let mut maximum_scratch_bytes = 0_u64;
    let mut maximum_scratch_slots = 0_u64;
    let mut previous_statement = None;
    for artifact_ref in 0..header.artifact_count {
        let offset = header
            .artifact_offset
            .checked_add(
                u64::from(artifact_ref)
                    .checked_mul(S8_ARTIFACT_BYTES)
                    .ok_or_else(|| error("S8 artifact offset multiplication overflows"))?,
            )
            .ok_or_else(|| error("S8 artifact offset overflows"))?;
        let artifact = fixed_at::<288>(framing, 7, offset, "artifact")?;
        let statement = u32_at(&artifact, 4);
        if previous_statement.is_some_and(|prior| prior >= statement) {
            return Err(error(
                "S8 artifact statement order is not strictly ascending",
            ));
        }
        previous_statement = Some(statement);
        let selection_start = u32_at(&artifact, 16);
        let selection_count = u32_at(&artifact, 20);
        let projection_start = u32_at(&artifact, 24);
        let projection_count = u32_at(&artifact, 28);
        let image_offset = u64_at(&artifact, 40);
        let image_bytes = u64_at(&artifact, 48);
        if u32_at(&artifact, 0) != artifact_ref
            || u32_at(&artifact, 8) != statement
            || u32_at(&artifact, 12) != statement
            || selection_start != selection_cursor
            || image_offset != image_cursor
            || u16::from_le_bytes(artifact[56..58].try_into().expect("fixed artifact kind")) != 1
            || u16::from_le_bytes(artifact[58..60].try_into().expect("fixed artifact flags")) != 0
            || u32_at(&artifact, 60) != 2
            || image_bytes == 0
            || artifact[64..288]
                .chunks_exact(32)
                .any(|value| value == [0; 32])
        {
            return Err(error(
                "S8 artifact scalar, role, or nonzero digest form is invalid",
            ));
        }
        let end_selection = selection_start
            .checked_add(selection_count)
            .filter(|end| *end <= header.selection_count)
            .ok_or_else(|| error("S8 artifact selection range is outside the directory"))?;
        let end_image = image_offset
            .checked_add(image_bytes)
            .filter(|end| *end <= header.image_arena_bytes)
            .ok_or_else(|| error("S8 artifact image range is outside the arena"))?;
        let s6 = s6_at(framing, statement, s7_counts[1])?;
        let resolution = s7_fixed_at::<320>(framing, 2, statement, s7_counts[1], 320)?;
        let coordinates = ArtifactCoordinates {
            artifact_ref,
            statement,
            selection_start,
            selection_count,
            projection_start,
            projection_count,
            s4_start: u32_at(&resolution, 24),
            s4_count: u32_at(&resolution, 28),
        };
        let s6_flags = u16::from_le_bytes(s6[10..12].try_into().expect("fixed S6 flags"));
        let resolution_flags = u32_at(&resolution, 20);
        let outcome = gpu_db_wal::decode_canonical_outcome_exact(&s6[44..136])
            .map_err(|_| error("S8 artifact references an invalid S6 outcome"))?;
        if s6_flags & (S6_FLAG_HAS_RETURNING | S6_FLAG_RESPONSE_RETAINED)
            != S6_FLAG_HAS_RETURNING | S6_FLAG_RESPONSE_RETAINED
            || resolution_flags & (S7_FLAG_HAS_RETURNING | S7_FLAG_RESPONSE_RETAINED)
                != S7_FLAG_HAS_RETURNING | S7_FLAG_RESPONSE_RETAINED
            || outcome.kind != gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
            || artifact[64..96] != s6[12..44]
            || artifact[96..128] != outcome.returning_digest
            || outcome.returning_digest == [0; 32]
            || u32_at(&resolution, 48) != projection_start
            || u32_at(&resolution, 52) != projection_count
            || u32_at(&artifact, 32) != selection_count
            || u32_at(&artifact, 36) != projection_count
        {
            return Err(error(
                "S8 artifact does not bind a successful retained S6/S7 statement",
            ));
        }
        let projection_root = projection_root(
            framing,
            s7_counts[10],
            statement,
            projection_start,
            projection_count,
        )?;
        if artifact[128..160] != projection_root {
            return Err(error("S8 artifact projection-root echo is invalid"));
        }
        let selection_root = selection_root(
            framing,
            header.selection_offset,
            coordinates,
            s7_counts[2] as u64,
        )?;
        if artifact[160..192] != selection_root {
            return Err(error("S8 artifact row-selection root is invalid"));
        }
        let source = S8ImageSource::new(
            framing,
            header
                .image_offset
                .checked_add(image_offset)
                .ok_or_else(|| error("S8 image source offset overflows"))?,
            image_bytes,
        );
        let image_measure = measure_decoded_typed_image_from_source(&source)
            .map_err(|_| error("S8 nested shared image fails strict measurement"))?;
        if image_measure.image_bytes() != image_bytes {
            return Err(error("S8 nested image measure length drifted"));
        }
        validate_image_header_and_projections(
            &source,
            framing,
            s7_counts[10],
            projection_start,
            projection_count,
            selection_count,
            &artifact,
        )?;
        if image_content_digest(&source)? != digest_at(&artifact, 224) {
            return Err(error("S8 image-content digest is invalid"));
        }
        let artifact_digest = artifact_digest(
            framing,
            &artifact,
            &s6,
            &resolution,
            s7_counts[10],
            header.selection_offset,
            coordinates,
        )?;
        if artifact_digest != digest_at(&artifact, 256) {
            return Err(error("S8 artifact digest is invalid"));
        }
        image_persistent_bytes = image_persistent_bytes
            .checked_add(image_measure.persistent_bytes())
            .ok_or_else(|| error("S8 image persistent bytes overflow"))?;
        image_persistent_slots = image_persistent_slots
            .checked_add(image_measure.persistent_allocation_slots())
            .ok_or_else(|| error("S8 image persistent slots overflow"))?;
        maximum_scratch_bytes = maximum_scratch_bytes.max(
            image_measure
                .maximum_with_image_copy_bytes()
                .map_err(|_| error("S8 image scratch bytes overflow"))?,
        );
        maximum_scratch_slots = maximum_scratch_slots.max(
            image_measure
                .maximum_with_image_copy_allocation_slots()
                .map_err(|_| error("S8 image scratch slots overflow"))?,
        );
        selection_cursor = end_selection;
        image_cursor = end_image;
    }
    if selection_cursor != header.selection_count || image_cursor != header.image_arena_bytes {
        return Err(error(
            "S8 directories are not exhaustively covered by artifacts",
        ));
    }
    validate_statement_and_status_closure(
        framing,
        s7_counts,
        Some((header.artifact_offset, header.artifact_count)),
        header.artifact_count,
        true,
    )?;
    Ok(S8PassZeroMeasure {
        identity: S8ResponseIdentity {
            present: true,
            aggregate_flags: framing.header_scalars().flags,
            stable_transaction_id: outer.stable_transaction_id,
            request_digest: outer.request_digest,
            s6_section_root: framing.section_roots()[5],
            s7_section_root: framing.section_roots()[6],
            s8_section_root: framing.section_roots()[7],
            response_root: framing.status().response_root,
            status_artifact_count: framing.status().response_artifact_count,
            retention_deadline: framing.status().retention_deadline,
            total_bytes: header.total_bytes,
            artifact_count: header.artifact_count,
            selection_count: header.selection_count,
            image_arena_bytes: header.image_arena_bytes,
            payload_digest: header.payload_digest,
        },
        image_persistent_bytes,
        image_persistent_slots,
        maximum_scratch_bytes,
        maximum_scratch_slots,
    })
}

fn validate_header(
    framing: &crate::typed_insert_aggregate::codec::DecodedAggregateFraming<'_>,
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    _s7_counts: &[u32; 12],
    raw: &[u8; 256],
) -> Result<Header, EngineError> {
    if &raw[..16] != b"GPUDBS8RESPONSE2"
        || u16::from_le_bytes(raw[16..18].try_into().expect("fixed S8 version")) != 1
        || u16::from_le_bytes(raw[18..20].try_into().expect("fixed S8 semantics")) != 2
        || u32_at(raw, 20) != 256
        || u32_at(raw, 24) != 0
        || u16::from_le_bytes(raw[28..30].try_into().expect("fixed S8 directory count")) != 3
        || u16::from_le_bytes(raw[30..32].try_into().expect("fixed S8 image version")) != 2
        || u64_at(raw, 104) != outer.stable_transaction_id
        || raw[112..144] != outer.request_digest
        || raw[144..176] != framing.section_roots()[5]
        || raw[176..208] != framing.section_roots()[6]
        || raw[240..256].iter().any(|byte| *byte != 0)
    {
        return Err(error("S8 fixed header identity is invalid"));
    }
    let total_bytes = u64_at(raw, 32);
    let artifact_count = u32_at(raw, 40);
    let selection_count = u32_at(raw, 44);
    let image_arena_bytes = u64_at(raw, 48);
    let artifact_offset = u64_at(raw, 56);
    let artifact_bytes = u64_at(raw, 64);
    let selection_offset = u64_at(raw, 72);
    let selection_bytes = u64_at(raw, 80);
    let image_offset = u64_at(raw, 88);
    let image_bytes = u64_at(raw, 96);
    let expected_artifacts = u64::from(artifact_count)
        .checked_mul(S8_ARTIFACT_BYTES)
        .ok_or_else(|| error("S8 artifact directory bytes overflow"))?;
    let expected_selections = u64::from(selection_count)
        .checked_mul(S8_SELECTION_BYTES)
        .ok_or_else(|| error("S8 selection directory bytes overflow"))?;
    let selection_end = artifact_offset
        .checked_add(artifact_bytes)
        .ok_or_else(|| error("S8 artifact directory end overflows"))?;
    let image_end = selection_offset
        .checked_add(selection_bytes)
        .ok_or_else(|| error("S8 selection directory end overflows"))?;
    let total_end = image_offset
        .checked_add(image_bytes)
        .ok_or_else(|| error("S8 image directory end overflows"))?;
    if total_bytes != framing.sections()[7].payload_bytes
        || artifact_count == 0
        || artifact_offset != S8_HEADER_BYTES
        || artifact_bytes != expected_artifacts
        || selection_offset != selection_end
        || selection_bytes != expected_selections
        || image_offset != image_end
        || image_bytes != image_arena_bytes
        || total_end != total_bytes
    {
        return Err(error("S8 directory geometry is not exact and adjacent"));
    }
    Ok(Header {
        total_bytes,
        artifact_count,
        selection_count,
        image_arena_bytes,
        artifact_offset,
        selection_offset,
        image_offset,
        payload_digest: digest_at(raw, 208),
    })
}

fn validate_payload_digest(
    framing: &crate::typed_insert_aggregate::codec::DecodedAggregateFraming<'_>,
    header: &[u8; 256],
    shape: Header,
) -> Result<(), EngineError> {
    let mut digest = begin(b"gpu-db/write001/s8-payload/v2");
    digest.update(shape.total_bytes.to_le_bytes());
    digest.update(&header[..208]);
    digest.update([0; 32]);
    digest.update(&header[240..]);
    let mut offset = S8_HEADER_BYTES;
    let mut scratch = [0_u8; 4096];
    while offset != shape.total_bytes {
        let take = usize::try_from((shape.total_bytes - offset).min(scratch.len() as u64))
            .map_err(|_| error("S8 payload scratch is not addressable"))?;
        framing.with_section_reader(7, |reader| {
            reader.skip(offset)?;
            reader.copy_exact(&mut scratch[..take])?;
            reader.skip(reader.remaining())
        })?;
        digest.update(&scratch[..take]);
        offset = offset
            .checked_add(take as u64)
            .ok_or_else(|| error("S8 payload offset overflows"))?;
    }
    if digest.finalize().as_slice() != shape.payload_digest {
        return Err(error("S8 payload digest is invalid"));
    }
    Ok(())
}

fn selection_root(
    framing: &crate::typed_insert_aggregate::codec::DecodedAggregateFraming<'_>,
    selection_offset: u64,
    coordinates: ArtifactCoordinates,
    s4_count: u64,
) -> Result<[u8; 32], EngineError> {
    let mut digest = begin(b"gpu-db/write001/s8-row-selection-root/v2");
    digest.update(coordinates.artifact_ref.to_le_bytes());
    digest.update(coordinates.statement.to_le_bytes());
    digest.update(coordinates.selection_count.to_le_bytes());
    let s4_end = coordinates
        .s4_start
        .checked_add(coordinates.s4_count)
        .filter(|end| u64::from(*end) <= s4_count)
        .ok_or_else(|| error("S8 artifact S4 range is outside the disposition directory"))?;
    let mut selected = 0_u32;
    for disposition in coordinates.s4_start..s4_end {
        let s4 = s4_at(framing, disposition, s4_count)?;
        if !matches!(s4[16], 1 | 2) {
            continue;
        }
        let index = coordinates
            .selection_start
            .checked_add(selected)
            .ok_or_else(|| error("S8 selection ordinal overflows"))?;
        let offset = selection_offset
            .checked_add(
                u64::from(index)
                    .checked_mul(S8_SELECTION_BYTES)
                    .ok_or_else(|| error("S8 selection offset multiplication overflows"))?,
            )
            .ok_or_else(|| error("S8 selection offset overflows"))?;
        let raw = fixed_at::<32>(framing, 7, offset, "selection")?;
        if u32_at(&raw, 0) != coordinates.artifact_ref
            || u32_at(&raw, 4) != selected
            || u32_at(&raw, 8) != disposition
            || u32_at(&raw, 12) != coordinates.statement
            || u32_at(&raw, 12) != u32_at(&s4, 0)
            || u32_at(&raw, 16) != u32_at(&s4, 4)
            || u32_at(&raw, 20) != u32_at(&s4, 20)
            || u64_at(&raw, 24) != u64_at(&s4, 8)
        {
            return Err(error("S8 row selection does not biject a selected S4 row"));
        }
        digest.update(raw);
        selected = selected
            .checked_add(1)
            .ok_or_else(|| error("S8 qualifying selection count overflows"))?;
    }
    if selected != coordinates.selection_count {
        return Err(error(
            "S8 selection count does not exhaust qualifying S4 rows",
        ));
    }
    Ok(digest.finalize().into())
}

fn projection_root(
    framing: &crate::typed_insert_aggregate::codec::DecodedAggregateFraming<'_>,
    projection_total: u32,
    statement: u32,
    start: u32,
    count: u32,
) -> Result<[u8; 32], EngineError> {
    let end = start
        .checked_add(count)
        .filter(|end| *end <= projection_total)
        .ok_or_else(|| error("S8 projection range is outside S7"))?;
    let mut digest = begin(b"gpu-db/write001/s7-statement-projections/v2");
    digest.update(statement.to_le_bytes());
    digest.update(count.to_le_bytes());
    for ordinal in start..end {
        let projection = s7_fixed_at::<128>(framing, 10, ordinal, projection_total, 128)?;
        if u32_at(&projection, 4) != statement || u32_at(&projection, 8) != ordinal - start {
            return Err(error("S8 projection range is not one S7 SQL-order range"));
        }
        digest.update(&projection[96..128]);
    }
    Ok(digest.finalize().into())
}

fn validate_image_header_and_projections(
    source: &S8ImageSource<'_>,
    framing: &crate::typed_insert_aggregate::codec::DecodedAggregateFraming<'_>,
    projection_total: u32,
    projection_start: u32,
    projection_count: u32,
    selection_count: u32,
    artifact: &[u8; 288],
) -> Result<(), EngineError> {
    let mut header = [0_u8; 112];
    source.read_at(0, &mut header)?;
    if &header[..16] != b"GPUDBTYPEDIMAGE2"
        || u16::from_le_bytes(header[16..18].try_into().expect("fixed image version")) != 2
        || u16::from_le_bytes(header[18..20].try_into().expect("fixed image header bytes")) != 112
        || u32_at(&header, 20) != 2
        || u32_at(&header, 24) != selection_count
        || u32_at(&header, 28) != projection_count
        || header[64..96] != artifact[192..224]
        || header[96..112].iter().any(|byte| *byte != 0)
    {
        return Err(error("S8 role-2 image header does not bind its artifact"));
    }
    let end = projection_start
        .checked_add(projection_count)
        .filter(|end| *end <= projection_total)
        .ok_or_else(|| error("S8 image projection range overflows"))?;
    for (image_ordinal, projection_ordinal) in (projection_start..end).enumerate() {
        let mut descriptor = [0_u8; 96];
        let descriptor_offset = 112_u64
            .checked_add(
                u64::try_from(image_ordinal)
                    .map_err(|_| error("S8 image descriptor ordinal exceeds u64"))?
                    .checked_mul(96)
                    .ok_or_else(|| error("S8 image descriptor offset overflows"))?,
            )
            .ok_or_else(|| error("S8 image descriptor offset overflows"))?;
        source.read_at(descriptor_offset, &mut descriptor)?;
        let projection =
            s7_fixed_at::<128>(framing, 10, projection_ordinal, projection_total, 128)?;
        if u32_at(&descriptor, 0) != image_ordinal as u32
            || descriptor[4..8] != projection[12..16]
            || descriptor[8..12] != projection[16..20]
            || descriptor[12..16] != projection[20..24]
            || descriptor[16..18] != projection[24..26]
            || descriptor[18..20].iter().any(|byte| *byte != 0)
            || descriptor[20..24] != projection[28..32]
            || descriptor[24..28] != projection[32..36]
            || descriptor[28..30] != projection[36..38]
            || descriptor[30..32] != projection[38..40]
            || identifier_digest(source, u64_at(&descriptor, 32), u64_at(&descriptor, 40))?
                != digest_at(&projection, 64)
        {
            return Err(error(
                "S8 role-2 image descriptor does not equal its S7 projection",
            ));
        }
    }
    Ok(())
}

fn identifier_digest(
    source: &S8ImageSource<'_>,
    offset: u64,
    bytes: u64,
) -> Result<[u8; 32], EngineError> {
    let bytes_u32 = u32::try_from(bytes)
        .map_err(|_| error("S8 image name length exceeds S7 identifier grammar"))?;
    let mut digest = begin(b"gpu-db/write001/s7-identifier/v2");
    digest.update(bytes_u32.to_le_bytes());
    let mut remaining = bytes;
    let mut cursor = offset;
    let mut scratch = [0_u8; 1024];
    while remaining != 0 {
        let take = usize::try_from(remaining.min(scratch.len() as u64))
            .map_err(|_| error("S8 name scratch is not addressable"))?;
        source.read_at(cursor, &mut scratch[..take])?;
        digest.update(&scratch[..take]);
        cursor = cursor
            .checked_add(take as u64)
            .ok_or_else(|| error("S8 name offset overflows"))?;
        remaining -= take as u64;
    }
    Ok(digest.finalize().into())
}

fn image_content_digest(source: &S8ImageSource<'_>) -> Result<[u8; 32], EngineError> {
    let mut digest = begin(b"gpu-db/write001/s8-image-content/v2");
    digest.update(source.len().to_le_bytes());
    let mut offset = 0_u64;
    let mut scratch = [0_u8; 4096];
    while offset != source.len() {
        let take = usize::try_from((source.len() - offset).min(scratch.len() as u64))
            .map_err(|_| error("S8 image digest scratch is not addressable"))?;
        source.read_at(offset, &mut scratch[..take])?;
        digest.update(&scratch[..take]);
        offset = offset
            .checked_add(take as u64)
            .ok_or_else(|| error("S8 image digest offset overflows"))?;
    }
    Ok(digest.finalize().into())
}

fn artifact_digest(
    framing: &crate::typed_insert_aggregate::codec::DecodedAggregateFraming<'_>,
    artifact: &[u8; 288],
    s6: &[u8; 136],
    resolution: &[u8; 320],
    projection_total: u32,
    selection_offset: u64,
    coordinates: ArtifactCoordinates,
) -> Result<[u8; 32], EngineError> {
    let mut digest = begin(b"gpu-db/write001/s8-artifact/v2");
    digest.update(&artifact[..256]);
    digest.update([0; 32]);
    digest.update(s6_entry_digest(s6));
    digest.update(resolution);
    let projection_end = coordinates
        .projection_start
        .checked_add(coordinates.projection_count)
        .filter(|end| *end <= projection_total)
        .ok_or_else(|| error("S8 artifact projection digest range overflows"))?;
    for ordinal in coordinates.projection_start..projection_end {
        digest.update(
            s7_fixed_at::<128>(framing, 10, ordinal, projection_total, 128)?[96..128].as_ref(),
        );
    }
    for ordinal in 0..coordinates.selection_count {
        let index = coordinates
            .selection_start
            .checked_add(ordinal)
            .ok_or_else(|| error("S8 artifact selection digest ordinal overflows"))?;
        let offset = selection_offset
            .checked_add(
                u64::from(index)
                    .checked_mul(S8_SELECTION_BYTES)
                    .ok_or_else(|| {
                        error("S8 artifact selection digest multiplication overflows")
                    })?,
            )
            .ok_or_else(|| error("S8 artifact selection digest offset overflows"))?;
        digest.update(fixed_at::<32>(framing, 7, offset, "artifact selection")?);
    }
    Ok(digest.finalize().into())
}

fn validate_statement_and_status_closure(
    framing: &crate::typed_insert_aggregate::codec::DecodedAggregateFraming<'_>,
    s7_counts: &[u32; 12],
    artifacts: Option<(u64, u32)>,
    artifact_count: u32,
    present: bool,
) -> Result<(), EngineError> {
    let mut artifact_cursor = 0_u32;
    let mut has_returning = false;
    for statement in 0..s7_counts[1] {
        let s6 = s6_at(framing, statement, s7_counts[1])?;
        let resolution = s7_fixed_at::<320>(framing, 2, statement, s7_counts[1], 320)?;
        let s6_flags = u16::from_le_bytes(s6[10..12].try_into().expect("fixed S6 flags"));
        let resolution_flags = u32_at(&resolution, 20);
        let outcome = gpu_db_wal::decode_canonical_outcome_exact(&s6[44..136])
            .map_err(|_| error("S8 closure S6 outcome is invalid"))?;
        let has = s6_flags & S6_FLAG_HAS_RETURNING != 0;
        let retained = s6_flags & S6_FLAG_RESPONSE_RETAINED != 0;
        if s6_flags & !3 != 0
            || retained && !has
            || resolution_flags & !3 != 0
            || resolution_flags != u32::from(s6_flags)
            || has != (u32_at(&resolution, 52) != 0)
            || (outcome.kind == gpu_db_wal::CanonicalOutcomeKind::AbortError
                && (retained || outcome.returning_digest != [0; 32]))
            || (outcome.kind == gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
                && (has != (outcome.returning_digest != [0; 32])))
        {
            return Err(error("S6/S7 RETURNING and retention closure is invalid"));
        }
        has_returning |= has;
        let named = if let Some((artifact_offset, count)) = artifacts {
            if artifact_cursor < count {
                let artifact = fixed_at::<288>(
                    framing,
                    7,
                    artifact_offset
                        .checked_add(
                            u64::from(artifact_cursor)
                                .checked_mul(S8_ARTIFACT_BYTES)
                                .ok_or_else(|| {
                                    error("S8 closure artifact offset multiplication overflows")
                                })?,
                        )
                        .ok_or_else(|| error("S8 closure artifact offset overflows"))?,
                    "closure artifact",
                )?;
                if u32_at(&artifact, 4) == statement {
                    artifact_cursor += 1;
                    true
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };
        if retained != named {
            return Err(error("S8 membership does not close S6/S7 retained bits"));
        }
    }
    let scalar = framing.header_scalars();
    let status = framing.status();
    if artifact_cursor != artifact_count
        || (scalar.flags & AGGREGATE_FLAG_RETURNING != 0) != has_returning
        || (scalar.flags & AGGREGATE_FLAG_RETAINED_RESPONSE != 0) != present
        || status.response_artifact_count != artifact_count
        || (artifact_count == 0 && status.retention_deadline != 0)
        || (artifact_count != 0 && !(1..u64::MAX).contains(&status.retention_deadline))
    {
        return Err(error(
            "S8 aggregate flag, STATUS2 count, or deadline closure is invalid",
        ));
    }
    Ok(())
}

fn begin(domain: &[u8]) -> Sha256 {
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest
}

fn s6_entry_digest(s6: &[u8; 136]) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/write001/s7-s6-entry/v2");
    digest.update(s6);
    digest.finalize().into()
}
