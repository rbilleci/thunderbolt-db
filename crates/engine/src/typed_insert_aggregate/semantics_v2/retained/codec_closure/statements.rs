//! S1/S2/S4/S5/S6 resolution and logical-`RETURNING` closure.

use super::error;
use crate::typed_insert_aggregate::semantics_v2::retained::graph::{
    ReservedSemanticsV2Graph, RetainedProjectionBinding,
};
use crate::typed_insert_batch::DecodedTypedValueFacts;
use crate::{EngineError, SqlType};
use sha2::{Digest, Sha256};

const ABSENT_U32: u32 = u32::MAX;
const SURVIVES: u8 = 1;
const APPLIED_THEN_CANCELED: u8 = 2;
const SUPPRESSED_AT_STATEMENT: u8 = 3;

pub(super) fn validate(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    if graph.statements.len() != graph.records.len()
        || graph.statements.len() != graph.outcomes.len()
        || graph.statements.len() != graph.resolutions.len()
        || graph.statements.is_empty()
    {
        return Err(error("S1/S2/S6/resolution cardinalities diverge"));
    }
    let mut next_s4 = 0_u32;
    let mut next_s5 = 0_u32;
    let mut next_dependency_use = 0_u32;
    let mut next_projection = 0_u32;
    for (ordinal, resolution) in graph.resolutions.iter().enumerate() {
        let ordinal = u32::try_from(ordinal).map_err(|_| error("statement ordinal overflows"))?;
        let statement = graph
            .statements
            .get(ordinal as usize)
            .ok_or_else(|| error("resolution S1 reference is absent"))?;
        let record = graph
            .records
            .get(usize::try_from(resolution.record_ref).map_err(|_| error("S2 ref overflows"))?)
            .ok_or_else(|| error("resolution S2 reference is absent"))?;
        let outcome = graph
            .outcomes
            .get(usize::try_from(resolution.outcome_ref).map_err(|_| error("S6 ref overflows"))?)
            .ok_or_else(|| error("resolution S6 reference is absent"))?;
        let table = graph
            .tables
            .get(usize::try_from(resolution.table_ref).map_err(|_| error("table ref overflows"))?)
            .ok_or_else(|| error("resolution target table is absent"))?;
        let facts = record.facts();
        if resolution.statement_ordinal != ordinal
            || resolution.record_ref != ordinal
            || resolution.outcome_ref != ordinal
            || statement.statement_ordinal != ordinal
            || outcome.statement_ordinal != ordinal
            || outcome.family_ordinal != statement.family_ordinal
            || outcome.semantic_class != 1
            || outcome.typed_statement_digest != statement.typed_statement_digest
            || resolution.outcome_digest != outcome.outcome_digest
            || resolution.request_digest != statement.request_digest
            || resolution.typed_statement_digest != statement.typed_statement_digest
            || resolution.record_bytes != statement.record_bytes
            || resolution.record_digest != statement.record_digest
            || resolution.overlay_before != statement.overlay_before
            || resolution.overlay_after != statement.overlay_after
            || outcome.outcome.target_digest != resolution.overlay_after
            || facts.statement_ordinal.as_u32() != ordinal
            || facts.typed_statement_digest != statement.typed_statement_digest
            || facts.row_count != statement.input_row_count
            || facts.row_count != resolution.input_row_count
            || facts.returning.digest != resolution.returning_digest
            || facts.target.oid != table.display_oid
            || resolution.s4_start != next_s4
            || resolution.s5_start != next_s5
            || resolution.dependency_use_start != next_dependency_use
            || resolution.projection_start != next_projection
        {
            return Err(error(
                "statement resolution does not close its retained S1/S2/S6 facts",
            ));
        }
        next_s4 = next_s4
            .checked_add(resolution.s4_count)
            .ok_or_else(|| error("S4 range count overflows"))?;
        next_s5 = next_s5
            .checked_add(resolution.s5_count)
            .ok_or_else(|| error("S5 range count overflows"))?;
        next_dependency_use = next_dependency_use
            .checked_add(resolution.dependency_use_count)
            .ok_or_else(|| error("dependency-use range count overflows"))?;
        next_projection = next_projection
            .checked_add(resolution.projection_count)
            .ok_or_else(|| error("projection range count overflows"))?;
        let dispositions = range(
            &graph.dispositions,
            resolution.s4_start,
            resolution.s4_count,
            "resolution S4",
        )?;
        if dispositions.len() != facts.row_count as usize
            || !dispositions.iter().enumerate().all(|(row, entry)| {
                entry.statement_ordinal == ordinal
                    && entry.source_row_ordinal == row as u32
                    && entry.typed_statement_digest == statement.typed_statement_digest
                    && entry.table_ref == resolution.table_ref
            })
        {
            return Err(error(
                "resolution S4 range is not its exact source-row range",
            ));
        }
        let surviving = dispositions
            .iter()
            .filter(|entry| entry.disposition == SURVIVES)
            .count();
        let affected = dispositions
            .iter()
            .filter(|entry| matches!(entry.disposition, SURVIVES | APPLIED_THEN_CANCELED))
            .count();
        if resolution.surviving_row_count != surviving as u32 {
            return Err(error("resolution survivor count differs from S4"));
        }
        let sequences = range(
            &graph.sequence_effects,
            resolution.s5_start,
            resolution.s5_count,
            "resolution S5",
        )?;
        if !sequences.iter().enumerate().all(|(effect, entry)| {
            entry.statement_ordinal == ordinal && entry.effect_ordinal == effect as u32
        }) || sequences.len() != facts.sequence_effect_count as usize
        {
            return Err(error(
                "resolution S5 range does not biject the decoded S2 effects",
            ));
        }
        let uses = range(
            &graph.dependency_uses,
            resolution.dependency_use_start,
            resolution.dependency_use_count,
            "resolution dependency use",
        )?;
        if uses.iter().any(|usage| usage.statement_ordinal != ordinal) {
            return Err(error(
                "resolution dependency-use range leaks another statement",
            ));
        }
        let projections = range(
            &graph.projections,
            resolution.projection_start,
            resolution.projection_count,
            "resolution projection",
        )?;
        let has_returning = resolution.projection_count != 0;
        if (resolution.flags & !3) != 0
            || (resolution.flags & 1 != 0) != has_returning
            || resolution.flags & 2 != 0
            || has_returning != (facts.returning.column_count != 0)
            || projections.len() != facts.returning.column_count as usize
        {
            return Err(error("statement RETURNING/retention flags are not exact"));
        }
        match outcome.outcome.kind {
            gpu_db_wal::CanonicalOutcomeKind::CommitSuccess => {
                if outcome.flags != u16::from(has_returning)
                    || outcome.outcome.affected_rows != affected as u64
                    || resolution.affected_row_count != affected as u64
                    || resolution.terminal_dependency_ref != ABSENT_U32
                    || resolution.terminal_row_ordinal != ABSENT_U32
                    || resolution.terminal_source_ordinal != ABSENT_U32
                {
                    return Err(error("successful statement outcome does not close S4/S7"));
                }
            }
            gpu_db_wal::CanonicalOutcomeKind::AbortError => {
                if outcome.flags != 0
                    || outcome.outcome.affected_rows != 0
                    || resolution.affected_row_count != 0
                    || resolution.terminal_dependency_ref == ABSENT_U32
                    || resolution.terminal_row_ordinal >= facts.row_count
                    || resolution.terminal_source_ordinal == ABSENT_U32
                    || outcome.outcome.returning_digest != [0; 32]
                {
                    return Err(error("terminal abort outcome does not close S4/S7"));
                }
            }
            gpu_db_wal::CanonicalOutcomeKind::CommitNoOp => {
                return Err(error("semantics-v2 permits no no-op statement outcome"));
            }
        }
    }
    if next_s4 as usize != graph.dispositions.len()
        || next_s5 as usize != graph.sequence_effects.len()
        || next_dependency_use as usize != graph.dependency_uses.len()
        || next_projection as usize != graph.projections.len()
    {
        return Err(error(
            "statement resolution ranges do not exhaust their retained directories",
        ));
    }
    let abort = graph
        .outcomes
        .iter()
        .position(|outcome| outcome.outcome.kind == gpu_db_wal::CanonicalOutcomeKind::AbortError);
    match abort {
        None => {
            if graph.outcomes.iter().any(|outcome| {
                outcome.outcome.kind != gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
            }) || graph
                .dispositions
                .iter()
                .any(|entry| entry.disposition != SURVIVES)
            {
                return Err(error(
                    "successful aggregate does not use the all-survives matrix",
                ));
            }
        }
        Some(last) => {
            if last + 1 != graph.outcomes.len()
                || graph.outcomes[..last].iter().any(|outcome| {
                    outcome.outcome.kind != gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
                })
                || graph.dispositions.iter().any(|entry| {
                    if (entry.statement_ordinal as usize) < last {
                        entry.disposition != APPLIED_THEN_CANCELED
                    } else {
                        entry.disposition != SUPPRESSED_AT_STATEMENT
                    }
                })
                || !graph.transitions.is_empty()
                || !graph.key_effects.is_empty()
                || graph.tables.iter().any(|table| {
                    table.data_generation_after != table.data_generation_before
                        || table.final_table_root != table.initial_table_root
                        || table.final_logical_row_count != table.initial_logical_row_count
                })
                || graph.indexes.iter().any(|index| {
                    index.final_index_generation != index.base_index_generation
                        || index.final_index_root != index.base_index_root
                })
                || graph.header.final_database_root != graph.header.initial_database_root
            {
                return Err(error(
                    "aborted aggregate does not use the final-abort matrix",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn validate_returning(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    for resolution in &graph.resolutions {
        let record = &graph.records[resolution.record_ref as usize];
        let outcome = &graph.outcomes[resolution.outcome_ref as usize];
        let projections = range(
            &graph.projections,
            resolution.projection_start,
            resolution.projection_count,
            "logical RETURNING projection",
        )?;
        validate_projection_bindings(
            record,
            resolution.statement_ordinal,
            resolution.table_ref,
            resolution.record_ref,
            resolution.projection_start,
            projections,
        )?;
        let expected = match outcome.outcome.kind {
            gpu_db_wal::CanonicalOutcomeKind::AbortError => [0; 32],
            gpu_db_wal::CanonicalOutcomeKind::CommitSuccess if projections.is_empty() => [0; 32],
            gpu_db_wal::CanonicalOutcomeKind::CommitSuccess => logical_returning_digest(
                graph,
                resolution.statement_ordinal,
                resolution.s4_start,
                resolution.s4_count,
                record,
                projections,
            )?,
            gpu_db_wal::CanonicalOutcomeKind::CommitNoOp => {
                return Err(error("no-op RETURNING is invalid"))
            }
        };
        if outcome.outcome.returning_digest != expected {
            return Err(error(
                "S6 logical RETURNING digest differs from strict S2 values",
            ));
        }
    }
    Ok(())
}

fn validate_projection_bindings(
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    statement: u32,
    table: u32,
    record_ref: u32,
    projection_start: u32,
    bindings: &[RetainedProjectionBinding],
) -> Result<(), EngineError> {
    let mut sources = record.returning_projections();
    for (ordinal, binding) in bindings.iter().enumerate() {
        let source = sources
            .next()
            .ok_or_else(|| error("S2 projection is absent"))?;
        if binding.projection_ref != projection_start + ordinal as u32
            || binding.statement_ordinal != statement
            || binding.projection_ordinal != ordinal as u32
            || binding.table_ref != table
            || binding.record_ref != record_ref
            || binding.source_catalog_ordinal != source.catalog_column_ordinal
            || binding.stable_column_id != source.column_id
            || binding.attnum != source.attnum
            || binding.storage != storage(source.ty)
            || binding.declared_type_oid != source.type_oid
            || binding.signed_type_size != source.type_size
            || binding.s2_projection_ordinal != ordinal as u32
            || binding.name_digest != identifier_digest(source.name)
            || binding.projection_digest != projection_digest(binding)
            || binding.result_format > 1
        {
            return Err(error(
                "S7 projection does not equal its strict S2 projection",
            ));
        }
    }
    if sources.next().is_some() {
        return Err(error("S7 projection range omits a strict S2 projection"));
    }
    Ok(())
}

fn logical_returning_digest(
    graph: &ReservedSemanticsV2Graph,
    statement: u32,
    start: u32,
    count: u32,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    projections: &[RetainedProjectionBinding],
) -> Result<[u8; 32], EngineError> {
    let dispositions = range(&graph.dispositions, start, count, "logical RETURNING S4")?;
    let selected_count = dispositions
        .iter()
        .filter(|entry| matches!(entry.disposition, SURVIVES | APPLIED_THEN_CANCELED))
        .count();
    let mut digest = begin(b"gpu-db/write001/s7-statement-returning-result/v2");
    digest.update(statement.to_le_bytes());
    digest.update(
        u32::try_from(selected_count)
            .map_err(|_| error("RETURNING row count overflows"))?
            .to_le_bytes(),
    );
    digest.update(
        u32::try_from(projections.len())
            .map_err(|_| error("RETURNING projection count overflows"))?
            .to_le_bytes(),
    );
    for projection in projections {
        digest.update(projection.projection_digest);
    }
    for disposition in dispositions
        .iter()
        .filter(|entry| matches!(entry.disposition, SURVIVES | APPLIED_THEN_CANCELED))
    {
        digest.update(disposition.source_row_ordinal.to_le_bytes());
        for projection in projections {
            digest.update(projection.projection_ordinal.to_le_bytes());
            digest.update(projection.source_catalog_ordinal.to_le_bytes());
            digest.update(projection.stable_column_id.to_le_bytes());
            digest.update(projection.attnum.to_le_bytes());
            digest.update(projection.storage);
            digest.update(projection.declared_type_oid.to_le_bytes());
            digest.update(projection.signed_type_size.to_le_bytes());
            digest.update(projection.result_format.to_le_bytes());
            let (valid, value) = record.column_value_at(
                projection.source_catalog_ordinal,
                disposition.source_row_ordinal,
            )?;
            append_record_value(&mut digest, valid, value);
        }
    }
    Ok(digest.finalize().into())
}

pub(super) fn append_record_value(
    digest: &mut Sha256,
    valid: bool,
    value: DecodedTypedValueFacts<'_>,
) {
    digest.update([u8::from(!valid)]);
    if !valid {
        digest.update(0_u32.to_le_bytes());
        return;
    }
    match value {
        DecodedTypedValueFacts::I32(value) => {
            digest.update(4_u32.to_le_bytes());
            digest.update(value.to_le_bytes());
        }
        DecodedTypedValueFacts::I64(value) => {
            digest.update(8_u32.to_le_bytes());
            digest.update(value.to_le_bytes());
        }
        DecodedTypedValueFacts::I128(value) => {
            digest.update(16_u32.to_le_bytes());
            digest.update(value.to_le_bytes());
        }
        DecodedTypedValueFacts::Uuid(value) => {
            digest.update(16_u32.to_le_bytes());
            digest.update(value);
        }
        DecodedTypedValueFacts::Bool(value) => {
            digest.update(1_u32.to_le_bytes());
            digest.update([u8::from(value)]);
        }
        DecodedTypedValueFacts::Text(value) => {
            digest.update((value.len() as u32).to_le_bytes());
            digest.update(value.as_bytes());
        }
    }
}

pub(super) fn storage(ty: SqlType) -> [u8; 4] {
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

fn projection_digest(binding: &RetainedProjectionBinding) -> [u8; 32] {
    let mut raw = [0_u8; 96];
    raw[..4].copy_from_slice(&binding.projection_ref.to_le_bytes());
    raw[4..8].copy_from_slice(&binding.statement_ordinal.to_le_bytes());
    raw[8..12].copy_from_slice(&binding.projection_ordinal.to_le_bytes());
    raw[12..16].copy_from_slice(&binding.source_catalog_ordinal.to_le_bytes());
    raw[16..20].copy_from_slice(&binding.stable_column_id.to_le_bytes());
    raw[20..24].copy_from_slice(&binding.table_ref.to_le_bytes());
    raw[24..26].copy_from_slice(&binding.attnum.to_le_bytes());
    raw[28..32].copy_from_slice(&binding.storage);
    raw[32..36].copy_from_slice(&binding.declared_type_oid.to_le_bytes());
    raw[36..38].copy_from_slice(&binding.signed_type_size.to_le_bytes());
    raw[38..40].copy_from_slice(&binding.result_format.to_le_bytes());
    raw[40..44].copy_from_slice(&binding.s2_projection_ordinal.to_le_bytes());
    raw[64..96].copy_from_slice(&binding.name_digest);
    exact(b"gpu-db/write001/s7-projection/v2", &[&raw, &[0; 32]])
}

fn identifier_digest(value: &str) -> [u8; 32] {
    exact(
        b"gpu-db/write001/s7-identifier/v2",
        &[&(value.len() as u32).to_le_bytes(), value.as_bytes()],
    )
}
pub(super) fn begin(domain: &[u8]) -> Sha256 {
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest
}
pub(super) fn exact(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut digest = begin(domain);
    for field in fields {
        digest.update(field);
    }
    digest.finalize().into()
}

pub(super) fn range<'a, T>(
    values: &'a [T],
    start: u32,
    count: u32,
    owner: &str,
) -> Result<&'a [T], EngineError> {
    let start = usize::try_from(start).map_err(|_| error(format!("{owner} start overflows")))?;
    let end = start
        .checked_add(usize::try_from(count).map_err(|_| error(format!("{owner} count overflows")))?)
        .ok_or_else(|| error(format!("{owner} range overflows")))?;
    values
        .get(start..end)
        .ok_or_else(|| error(format!("{owner} range is absent")))
}
