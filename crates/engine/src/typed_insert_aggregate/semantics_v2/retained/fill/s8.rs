//! Strict post-reservation fill for S8 role-2 retained responses.

use super::fixed::{digest_at, push_exact, u32_at, u64_at};
use super::source::{exact_copy_scratch, fill_error, AggregateRegionSource};
use super::ObservedSourceMeasure;
use crate::typed_insert_aggregate::codec::DecodedAggregateFraming;
use crate::typed_insert_aggregate::semantics_v2::retained::graph::{
    ReservedSemanticsV2Graph, RetainedResponseArtifact, RetainedResponseEnvelope,
    RetainedResponseSelection,
};
use crate::typed_insert_batch::{
    copy_typed_image_after_measure, decode_typed_image_after_measure,
    measure_decoded_typed_image_from_source, TypedImageRole,
};
use crate::EngineError;
use sha2::{Digest, Sha256};

pub(super) fn fill_s8(
    framing: &DecodedAggregateFraming<'_>,
    graph: &mut ReservedSemanticsV2Graph,
    observed: &mut ObservedSourceMeasure,
) -> Result<(), EngineError> {
    let RetainedResponseEnvelope::Present(response) = &mut graph.response else {
        if framing.sections()[7].entry_count != 0 || framing.sections()[7].payload_bytes != 0 {
            return Err(fill_error("S8 presence changed after pass-zero"));
        }
        return Ok(());
    };
    let stable_transaction_id = response.identity.stable_transaction_id;
    framing.with_section_reader(7, |reader| {
        let header = reader.exact::<256>()?;
        if &header[..16] != b"GPUDBS8RESPONSE2"
            || u64_at(&header, 32) != response.identity.total_bytes
            || u32_at(&header, 40) != response.identity.artifact_count
            || u32_at(&header, 44) != response.identity.selection_count
            || u64_at(&header, 48) != response.identity.image_arena_bytes
            || u64_at(&header, 104) != stable_transaction_id
            || header[112..144] != response.identity.request_digest
            || header[144..176] != response.identity.s6_section_root
            || header[176..208] != response.identity.s7_section_root
            || digest_at(&header, 208) != response.identity.payload_digest
        {
            return Err(fill_error(
                "S8 header changed after pass-zero envelope proof",
            ));
        }
        for _ in 0..response.artifacts.capacity() {
            let raw = reader.exact::<288>()?;
            push_exact(
                &mut response.artifacts,
                RetainedResponseArtifact {
                    artifact_ref: u32_at(&raw, 0),
                    statement_ref: u32_at(&raw, 4),
                    selection_start: u32_at(&raw, 16),
                    selection_count: u32_at(&raw, 20),
                    projection_start: u32_at(&raw, 24),
                    projection_count: u32_at(&raw, 28),
                    image_rows: u32_at(&raw, 32),
                    image_columns: u32_at(&raw, 36),
                    image_arena_offset: u64_at(&raw, 40),
                    image_bytes: u64_at(&raw, 48),
                    typed_statement_digest: digest_at(&raw, 64),
                    logical_result_digest: digest_at(&raw, 96),
                    projection_root: digest_at(&raw, 128),
                    selection_root: digest_at(&raw, 160),
                    image_layout_digest: digest_at(&raw, 192),
                    image_content_digest: digest_at(&raw, 224),
                    artifact_digest: digest_at(&raw, 256),
                },
                "S8 response-artifact directory",
            )?;
        }
        for _ in 0..response.selections.capacity() {
            let raw = reader.exact::<32>()?;
            push_exact(
                &mut response.selections,
                RetainedResponseSelection {
                    artifact_ref: u32_at(&raw, 0),
                    image_row_ordinal: u32_at(&raw, 4),
                    disposition_ref: u32_at(&raw, 8),
                    statement_ref: u32_at(&raw, 12),
                    source_row_ordinal: u32_at(&raw, 16),
                    table_ref: u32_at(&raw, 20),
                    stable_row_id: u64_at(&raw, 24),
                },
                "S8 response-selection directory",
            )?;
        }
        // `with_section_reader` owns the whole S8 section.  The image arena is decoded below
        // through bounded random-access sources, but this cursor must still account for its
        // exact measured extent before the section boundary is accepted.
        reader.skip(response.identity.image_arena_bytes)?;
        Ok(())
    })?;

    let image_start = 256_u64
        .checked_add(
            u64::try_from(response.artifacts.len())
                .map_err(|_| fill_error("S8 artifact count exceeds addressability"))?
                .checked_mul(288)
                .ok_or_else(|| fill_error("S8 artifact bytes overflow"))?,
        )
        .and_then(|offset| {
            offset.checked_add(
                u64::try_from(response.selections.len())
                    .ok()?
                    .checked_mul(32)?,
            )
        })
        .ok_or_else(|| fill_error("S8 image arena start overflows"))?;
    for artifact in &response.artifacts {
        let source = AggregateRegionSource::new(
            framing,
            7,
            image_start
                .checked_add(artifact.image_arena_offset)
                .ok_or_else(|| fill_error("S8 image source offset overflows"))?,
            artifact.image_bytes,
        );
        let measure = measure_decoded_typed_image_from_source(&source)
            .map_err(|_| fill_error("S8 response image source measurement fails strict decode"))?;
        if measure.image_bytes() != artifact.image_bytes {
            return Err(fill_error("S8 response image measure length drifted"));
        }
        observed.checked_add_response_image(
            measure.persistent_bytes(),
            measure.persistent_allocation_slots(),
            measure
                .maximum_with_image_copy_bytes()
                .map_err(|_| fill_error("S8 image copy scratch measure overflows"))?,
            measure
                .maximum_with_image_copy_allocation_slots()
                .map_err(|_| fill_error("S8 image copy scratch slot measure overflows"))?,
        )?;
        let mut scratch = exact_copy_scratch(artifact.image_bytes, "S8 exact image source copy")?;
        copy_typed_image_after_measure(&source, measure, &mut scratch)
            .map_err(|_| fill_error("S8 response image source changed after measurement"))?;
        let content = image_content_digest(&scratch);
        let decoded = decode_typed_image_after_measure(&scratch, measure)
            .map_err(|_| fill_error("S8 exact image copy fails strict decode"))?;
        drop(scratch);
        let facts = decoded.facts();
        if facts.role != TypedImageRole::RetainedResponse
            || facts.rows != artifact.image_rows
            || facts.columns != artifact.image_columns
            || facts.layout_digest != artifact.image_layout_digest
            || content != artifact.image_content_digest
        {
            return Err(fill_error(
                "S8 decoded image does not match its retained artifact",
            ));
        }
        push_exact(
            &mut response.images,
            decoded,
            "S8 decoded response-image directory",
        )?;
    }
    Ok(())
}

fn image_content_digest(bytes: &[u8]) -> [u8; 32] {
    let domain = b"gpu-db/write001/s8-image-content/v2";
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
    digest.finalize().into()
}
