//! Local S8 closure over already move-only S2/S4/S6/S7 and role-2 image owners.
//!
//! This deliberately proves only local codec consistency.  It has no RetentionIntent input and
//! therefore cannot turn a coherent omitted eligible artifact into a locally disproven claim.

use super::{error, statements};
use crate::typed_insert_aggregate::semantics_v2::retained::graph::{
    ReservedSemanticsV2Graph, RetainedProjectionBinding, RetainedResponseArtifact,
    RetainedResponseEnvelope, RetainedResponseEnvelopeIdentity, RetainedResponseSelection,
    RetainedStatementResolution,
};
use crate::typed_insert_aggregate::semantics_v2::retained::SemanticsV2BoundIdentity;
use crate::typed_insert_aggregate::{AGGREGATE_FLAG_RETAINED_RESPONSE, AGGREGATE_FLAG_RETURNING};
use crate::typed_insert_batch::{
    DecodedTypedImageColumnFacts, DecodedTypedValueFacts, TypedImageRole,
    TypedInsertColumnValidity, TypedInsertColumnValues,
};
use crate::EngineError;
use sha2::{Digest, Sha256};

pub(super) fn validate(
    bound_identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    let (present, identity, artifacts, selections, images) = match &graph.response {
        RetainedResponseEnvelope::Empty(identity) => (false, identity, &[][..], &[][..], &[][..]),
        RetainedResponseEnvelope::Present(response) => (
            true,
            &response.identity,
            response.artifacts.as_slice(),
            response.selections.as_slice(),
            response.images.as_slice(),
        ),
    };
    validate_envelope_identity(bound_identity, present, identity, artifacts.len())?;
    if artifacts.len() != images.len()
        || artifacts.len() != identity.artifact_count as usize
        || selections.len() != identity.selection_count as usize
    {
        return Err(error(
            "S8 retained owner cardinalities do not match the envelope",
        ));
    }
    let mut next_selection = 0_usize;
    let mut next_statement = 0_u32;
    for (artifact_ordinal, (artifact, image)) in artifacts.iter().zip(images).enumerate() {
        if artifact.artifact_ref
            != u32::try_from(artifact_ordinal)
                .map_err(|_| error("S8 artifact ordinal exceeds u32"))?
            || artifact.statement_ref < next_statement
        {
            return Err(error("S8 retained artifact order is not canonical"));
        }
        next_statement = artifact
            .statement_ref
            .checked_add(1)
            .ok_or_else(|| error("S8 statement ordinal overflows"))?;
        let resolution = graph
            .resolutions
            .get(artifact.statement_ref as usize)
            .ok_or_else(|| error("S8 artifact resolution is absent"))?;
        let outcome = graph
            .outcomes
            .get(artifact.statement_ref as usize)
            .ok_or_else(|| error("S8 artifact S6 outcome is absent"))?;
        if outcome.outcome.kind != gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
            || outcome.flags & 3 != 3
            || resolution.flags & 3 != 3
            || artifact.typed_statement_digest != resolution.typed_statement_digest
            || artifact.logical_result_digest != outcome.outcome.returning_digest
            || artifact.logical_result_digest == [0; 32]
            || artifact.projection_start != resolution.projection_start
            || artifact.projection_count != resolution.projection_count
            || artifact.image_rows != artifact.selection_count
            || artifact.image_columns != artifact.projection_count
        {
            return Err(error(
                "S8 artifact does not close a retained successful statement",
            ));
        }
        if artifact.selection_start
            != u32::try_from(next_selection)
                .map_err(|_| error("S8 running selection ordinal exceeds u32"))?
        {
            return Err(error(
                "S8 artifact selection start is not the next exact selection",
            ));
        }
        let selection_count = usize::try_from(artifact.selection_count)
            .map_err(|_| error("S8 selection count exceeds host addressability"))?;
        let selection_end = next_selection
            .checked_add(selection_count)
            .filter(|end| *end <= selections.len())
            .ok_or_else(|| error("S8 artifact selection range is absent"))?;
        let selected = &selections[next_selection..selection_end];
        validate_selections(graph, artifact, selected)?;
        if selection_root(artifact, selected) != artifact.selection_root {
            return Err(error("S8 retained row-selection root is invalid"));
        }
        let projections = projection_range(graph, resolution)?;
        if projection_root(resolution.statement_ordinal, projections)? != artifact.projection_root {
            return Err(error("S8 retained projection-root echo is invalid"));
        }
        validate_image_and_values(graph, artifact, image, projections, selected)?;
        if artifact_digest(
            artifact,
            outcome.outcome_digest,
            resolution,
            projections,
            selected,
        )? != artifact.artifact_digest
        {
            return Err(error("S8 retained artifact digest is invalid"));
        }
        next_selection = selection_end;
    }
    if next_selection != selections.len() {
        return Err(error("S8 retained selections are not exhaustively owned"));
    }
    validate_retained_membership(graph, artifacts)?;
    Ok(())
}

fn validate_envelope_identity(
    bound_identity: SemanticsV2BoundIdentity,
    present: bool,
    identity: &RetainedResponseEnvelopeIdentity,
    artifact_len: usize,
) -> Result<(), EngineError> {
    if identity.stable_transaction_id != bound_identity.stable_transaction_id
        || identity.request_digest != bound_identity.request_digest
        || identity.present != present
    {
        return Err(error(
            "S8 retained request digest is not the aggregate request identity",
        ));
    }
    let expected_response = if identity.aggregate_flags & AGGREGATE_FLAG_RETURNING != 0 {
        response_root(identity.s6_section_root, identity.s8_section_root)
    } else {
        [0; 32]
    };
    if identity.s6_section_root == [0; 32]
        || identity.s7_section_root == [0; 32]
        || identity.s8_section_root == [0; 32]
        || identity.response_root != expected_response
        || (identity.aggregate_flags & AGGREGATE_FLAG_RETAINED_RESPONSE != 0) != (artifact_len != 0)
        || present != (artifact_len != 0)
        || identity.status_artifact_count
            != u32::try_from(artifact_len)
                .map_err(|_| error("S8 retained artifact count exceeds u32"))?
        || (artifact_len == 0 && identity.retention_deadline != 0)
        || (artifact_len != 0 && !(1..u64::MAX).contains(&identity.retention_deadline))
    {
        return Err(error(
            "S8 retained aggregate/STATUS envelope closure is invalid",
        ));
    }
    Ok(())
}

fn validate_retained_membership(
    graph: &ReservedSemanticsV2Graph,
    artifacts: &[RetainedResponseArtifact],
) -> Result<(), EngineError> {
    let mut cursor = 0_usize;
    let mut has_returning = false;
    for (statement, (outcome, resolution)) in
        graph.outcomes.iter().zip(&graph.resolutions).enumerate()
    {
        let has = outcome.flags & 1 != 0;
        let retained = outcome.flags & 2 != 0;
        if outcome.flags & !3 != 0
            || outcome.flags as u32 != resolution.flags
            || retained && !has
            || has != (resolution.projection_count != 0)
        {
            return Err(error("S8 retained S6/S7 flag closure is invalid"));
        }
        has_returning |= has;
        let named = artifacts
            .get(cursor)
            .is_some_and(|artifact| artifact.statement_ref == statement as u32);
        if named {
            cursor += 1;
        }
        if named != retained {
            return Err(error("S8 retained membership differs from S6/S7 bit 1"));
        }
    }
    let identity = match &graph.response {
        RetainedResponseEnvelope::Empty(identity) => identity,
        RetainedResponseEnvelope::Present(response) => &response.identity,
    };
    if cursor != artifacts.len()
        || (identity.aggregate_flags & AGGREGATE_FLAG_RETURNING != 0) != has_returning
    {
        return Err(error("S8 retained aggregate RETURNING closure is invalid"));
    }
    Ok(())
}

fn validate_selections(
    graph: &ReservedSemanticsV2Graph,
    artifact: &RetainedResponseArtifact,
    selections: &[RetainedResponseSelection],
) -> Result<(), EngineError> {
    let resolution = graph
        .resolutions
        .get(artifact.statement_ref as usize)
        .ok_or_else(|| error("S8 selection resolution is absent"))?;
    let dispositions = disposition_range(graph, resolution)?;
    let mut selected = selections.iter();
    let mut selected_count = 0_usize;
    for (source, disposition) in dispositions.iter().enumerate() {
        if !matches!(disposition.disposition, 1 | 2) {
            continue;
        }
        let selection = selected
            .next()
            .ok_or_else(|| error("S8 selection cardinality does not exhaust qualifying S4 rows"))?;
        if selection.artifact_ref != artifact.artifact_ref
            || selection.image_row_ordinal
                != u32::try_from(selected_count)
                    .map_err(|_| error("S8 selection ordinal exceeds u32"))?
            || selection.disposition_ref
                != resolution
                    .s4_start
                    .checked_add(
                        u32::try_from(source)
                            .map_err(|_| error("S8 disposition ordinal exceeds u32"))?,
                    )
                    .ok_or_else(|| error("S8 disposition reference overflows"))?
            || selection.statement_ref != artifact.statement_ref
            || selection.source_row_ordinal != disposition.source_row_ordinal
            || selection.table_ref != disposition.table_ref
            || selection.stable_row_id != disposition.stable_row_id
        {
            return Err(error(
                "S8 selection is not the exact qualifying S4 source order",
            ));
        }
        selected_count = selected_count
            .checked_add(1)
            .ok_or_else(|| error("S8 selected-row count overflows"))?;
    }
    if selected.next().is_some() {
        return Err(error(
            "S8 selection cardinality does not exhaust qualifying S4 rows",
        ));
    }
    Ok(())
}

fn validate_image_and_values(
    graph: &ReservedSemanticsV2Graph,
    artifact: &RetainedResponseArtifact,
    image: &crate::typed_insert_batch::DecodedTypedImage,
    projections: &[RetainedProjectionBinding],
    selections: &[RetainedResponseSelection],
) -> Result<(), EngineError> {
    let facts = image.facts();
    if facts.role != TypedImageRole::RetainedResponse
        || facts.rows != artifact.image_rows
        || facts.columns != artifact.image_columns
        || facts.layout_digest != artifact.image_layout_digest
    {
        return Err(error("S8 role-2 image facts do not match the artifact"));
    }
    let mut columns = image.columns();
    let record = graph
        .records
        .get(artifact.statement_ref as usize)
        .ok_or_else(|| error("S8 source S2 record is absent"))?;
    for (ordinal, projection) in projections.iter().enumerate() {
        let column = columns
            .next()
            .ok_or_else(|| error("S8 image column count differs from S7 projections"))?;
        if column.catalog_column_ordinal != projection.source_catalog_ordinal
            || column.stable_column_id != projection.stable_column_id
            || column.table_ref != projection.table_ref
            || column.attnum != projection.attnum
            || storage(column.ty) != projection.storage
            || column.type_oid != projection.declared_type_oid
            || column.type_size != projection.signed_type_size
            || column.result_format != projection.result_format
            || identifier_digest(column.name) != projection.name_digest
        {
            return Err(error("S8 image descriptor differs from its S7 projection"));
        }
        for (row, selection) in selections.iter().enumerate() {
            let (valid, value) = record.column_value_at(
                projection.source_catalog_ordinal,
                selection.source_row_ordinal,
            )?;
            if !image_cell_matches(&column, row, valid, value)? {
                return Err(error(
                    "S8 retained image cell differs from its selected S2 value",
                ));
            }
        }
        if u32::try_from(ordinal).map_err(|_| error("S8 projection ordinal exceeds u32"))?
            != projection.projection_ordinal
        {
            return Err(error("S8 image projection order is not SQL order"));
        }
    }
    if columns.next().is_some() {
        return Err(error("S8 image column count differs from S7 projections"));
    }
    Ok(())
}

fn image_cell_matches(
    column: &DecodedTypedImageColumnFacts<'_>,
    row: usize,
    source_valid: bool,
    source: DecodedTypedValueFacts<'_>,
) -> Result<bool, EngineError> {
    let valid = match column.validity {
        TypedInsertColumnValidity::AllValid => true,
        TypedInsertColumnValidity::Bitmap(words) => words
            .get(row / 32)
            .is_some_and(|word| word & (1 << (row % 32)) != 0),
    };
    if valid != source_valid {
        return Ok(false);
    }
    if !valid {
        return Ok(true);
    }
    Ok(match (column.values, source) {
        (TypedInsertColumnValues::I32(values), DecodedTypedValueFacts::I32(value)) => {
            values.get(row) == Some(&value)
        }
        (TypedInsertColumnValues::I64(values), DecodedTypedValueFacts::I64(value)) => {
            values.get(row) == Some(&value)
        }
        (TypedInsertColumnValues::I128(values), DecodedTypedValueFacts::I128(value)) => {
            values.get(row) == Some(&value)
        }
        (TypedInsertColumnValues::Bytes16(values), DecodedTypedValueFacts::Uuid(value)) => {
            values.get(row) == Some(&value)
        }
        (TypedInsertColumnValues::BoolBits(words), DecodedTypedValueFacts::Bool(value)) => words
            .get(row / 32)
            .is_some_and(|word| (word & (1 << (row % 32)) != 0) == value),
        (TypedInsertColumnValues::Text { offsets, bytes }, DecodedTypedValueFacts::Text(value)) => {
            let Some((&start, &end)) = offsets.get(row).zip(offsets.get(row + 1)) else {
                return Ok(false);
            };
            usize::try_from(start)
                .ok()
                .zip(usize::try_from(end).ok())
                .and_then(|(start, end)| bytes.get(start..end))
                == Some(value.as_bytes())
        }
        _ => false,
    })
}

fn projection_range<'a>(
    graph: &'a ReservedSemanticsV2Graph,
    resolution: &RetainedStatementResolution,
) -> Result<&'a [RetainedProjectionBinding], EngineError> {
    let start = resolution.projection_start as usize;
    let end = start
        .checked_add(resolution.projection_count as usize)
        .ok_or_else(|| error("S8 projection range overflows"))?;
    graph
        .projections
        .get(start..end)
        .ok_or_else(|| error("S8 projection range is absent"))
}

fn disposition_range<'a>(
    graph: &'a ReservedSemanticsV2Graph,
    resolution: &RetainedStatementResolution,
) -> Result<
    &'a [crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedDisposition],
    EngineError,
> {
    let start = resolution.s4_start as usize;
    let end = start
        .checked_add(resolution.s4_count as usize)
        .ok_or_else(|| error("S8 disposition range overflows"))?;
    graph
        .dispositions
        .get(start..end)
        .ok_or_else(|| error("S8 disposition range is absent"))
}

fn selection_root(
    artifact: &RetainedResponseArtifact,
    selections: &[RetainedResponseSelection],
) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/write001/s8-row-selection-root/v2");
    digest.update(artifact.artifact_ref.to_le_bytes());
    digest.update(artifact.statement_ref.to_le_bytes());
    digest.update(artifact.selection_count.to_le_bytes());
    for selection in selections {
        digest.update(selection_bytes(selection));
    }
    digest.finalize().into()
}

fn projection_root(
    statement: u32,
    projections: &[RetainedProjectionBinding],
) -> Result<[u8; 32], EngineError> {
    let mut digest = begin(b"gpu-db/write001/s7-statement-projections/v2");
    digest.update(statement.to_le_bytes());
    digest.update(
        u32::try_from(projections.len())
            .map_err(|_| error("S8 projection count overflows"))?
            .to_le_bytes(),
    );
    for projection in projections {
        digest.update(projection.projection_digest);
    }
    Ok(digest.finalize().into())
}

fn artifact_digest(
    artifact: &RetainedResponseArtifact,
    s6_digest: [u8; 32],
    resolution: &RetainedStatementResolution,
    projections: &[RetainedProjectionBinding],
    selections: &[RetainedResponseSelection],
) -> Result<[u8; 32], EngineError> {
    let mut digest = begin(b"gpu-db/write001/s8-artifact/v2");
    digest.update(artifact_prefix(artifact));
    digest.update([0; 32]);
    digest.update(s6_digest);
    digest.update(resolution_bytes(resolution));
    for projection in projections {
        digest.update(projection.projection_digest);
    }
    for selection in selections {
        digest.update(selection_bytes(selection));
    }
    Ok(digest.finalize().into())
}

fn response_root(s6: [u8; 32], s8: [u8; 32]) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/write001/response-root/v1");
    digest.update(32_u64.to_le_bytes());
    digest.update(s6);
    digest.update(32_u64.to_le_bytes());
    digest.update(s8);
    digest.finalize().into()
}

fn artifact_prefix(artifact: &RetainedResponseArtifact) -> [u8; 256] {
    let mut raw = [0_u8; 256];
    raw[..4].copy_from_slice(&artifact.artifact_ref.to_le_bytes());
    raw[4..8].copy_from_slice(&artifact.statement_ref.to_le_bytes());
    raw[8..12].copy_from_slice(&artifact.statement_ref.to_le_bytes());
    raw[12..16].copy_from_slice(&artifact.statement_ref.to_le_bytes());
    raw[16..20].copy_from_slice(&artifact.selection_start.to_le_bytes());
    raw[20..24].copy_from_slice(&artifact.selection_count.to_le_bytes());
    raw[24..28].copy_from_slice(&artifact.projection_start.to_le_bytes());
    raw[28..32].copy_from_slice(&artifact.projection_count.to_le_bytes());
    raw[32..36].copy_from_slice(&artifact.image_rows.to_le_bytes());
    raw[36..40].copy_from_slice(&artifact.image_columns.to_le_bytes());
    raw[40..48].copy_from_slice(&artifact.image_arena_offset.to_le_bytes());
    raw[48..56].copy_from_slice(&artifact.image_bytes.to_le_bytes());
    raw[56..58].copy_from_slice(&1_u16.to_le_bytes());
    raw[60..64].copy_from_slice(&2_u32.to_le_bytes());
    raw[64..96].copy_from_slice(&artifact.typed_statement_digest);
    raw[96..128].copy_from_slice(&artifact.logical_result_digest);
    raw[128..160].copy_from_slice(&artifact.projection_root);
    raw[160..192].copy_from_slice(&artifact.selection_root);
    raw[192..224].copy_from_slice(&artifact.image_layout_digest);
    raw[224..256].copy_from_slice(&artifact.image_content_digest);
    raw
}

fn selection_bytes(selection: &RetainedResponseSelection) -> [u8; 32] {
    let mut raw = [0_u8; 32];
    raw[..4].copy_from_slice(&selection.artifact_ref.to_le_bytes());
    raw[4..8].copy_from_slice(&selection.image_row_ordinal.to_le_bytes());
    raw[8..12].copy_from_slice(&selection.disposition_ref.to_le_bytes());
    raw[12..16].copy_from_slice(&selection.statement_ref.to_le_bytes());
    raw[16..20].copy_from_slice(&selection.source_row_ordinal.to_le_bytes());
    raw[20..24].copy_from_slice(&selection.table_ref.to_le_bytes());
    raw[24..32].copy_from_slice(&selection.stable_row_id.to_le_bytes());
    raw
}

fn resolution_bytes(resolution: &RetainedStatementResolution) -> [u8; 320] {
    let mut raw = [0_u8; 320];
    raw[..4].copy_from_slice(&resolution.statement_ordinal.to_le_bytes());
    // S7 redundantly carries the statement ordinal at offset four.  The retained graph has no
    // separate field because pass-zero proves it equals `statement_ordinal`, but exact S8
    // artifact replay must reconstruct the physical transcript byte-for-byte.
    raw[4..8].copy_from_slice(&resolution.statement_ordinal.to_le_bytes());
    raw[8..12].copy_from_slice(&resolution.record_ref.to_le_bytes());
    raw[12..16].copy_from_slice(&resolution.outcome_ref.to_le_bytes());
    raw[16..20].copy_from_slice(&resolution.table_ref.to_le_bytes());
    raw[20..24].copy_from_slice(&resolution.flags.to_le_bytes());
    raw[24..28].copy_from_slice(&resolution.s4_start.to_le_bytes());
    raw[28..32].copy_from_slice(&resolution.s4_count.to_le_bytes());
    raw[32..36].copy_from_slice(&resolution.s5_start.to_le_bytes());
    raw[36..40].copy_from_slice(&resolution.s5_count.to_le_bytes());
    raw[40..44].copy_from_slice(&resolution.dependency_use_start.to_le_bytes());
    raw[44..48].copy_from_slice(&resolution.dependency_use_count.to_le_bytes());
    raw[48..52].copy_from_slice(&resolution.projection_start.to_le_bytes());
    raw[52..56].copy_from_slice(&resolution.projection_count.to_le_bytes());
    raw[56..60].copy_from_slice(&resolution.input_row_count.to_le_bytes());
    raw[60..64].copy_from_slice(&resolution.surviving_row_count.to_le_bytes());
    raw[64..72].copy_from_slice(&resolution.affected_row_count.to_le_bytes());
    raw[72..80].copy_from_slice(&resolution.dependency_validation_floor.to_le_bytes());
    raw[80..84].copy_from_slice(&resolution.record_bytes.to_le_bytes());
    raw[84..88].copy_from_slice(&resolution.terminal_dependency_ref.to_le_bytes());
    raw[88..92].copy_from_slice(&resolution.terminal_row_ordinal.to_le_bytes());
    raw[92..96].copy_from_slice(&resolution.terminal_source_ordinal.to_le_bytes());
    raw[96..128].copy_from_slice(&resolution.request_digest);
    raw[128..160].copy_from_slice(&resolution.typed_statement_digest);
    raw[160..192].copy_from_slice(&resolution.record_digest);
    raw[192..224].copy_from_slice(&resolution.returning_digest);
    raw[224..256].copy_from_slice(&resolution.overlay_before);
    raw[256..288].copy_from_slice(&resolution.overlay_after);
    raw[288..320].copy_from_slice(&resolution.outcome_digest);
    raw
}

fn identifier_digest(value: &str) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/write001/s7-identifier/v2");
    digest.update((value.len() as u32).to_le_bytes());
    digest.update(value.as_bytes());
    digest.finalize().into()
}
fn storage(ty: crate::SqlType) -> [u8; 4] {
    statements::storage(ty)
}
fn begin(domain: &[u8]) -> Sha256 {
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest
}
