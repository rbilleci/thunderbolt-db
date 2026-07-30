//! Exact fixed-width S1/S4/S5/S6 retained owners.
//!
//! Pass zero has already proved the nonlocal grammar. This pass copies every variable wire fact
//! into the graph's pre-reserved typed owners and repeats local bounds before a push, so no
//! malformed or overfull directory can turn into a hidden allocation.

use super::source::fill_error;
use crate::typed_insert_aggregate::codec::DecodedAggregateFraming;
use crate::typed_insert_aggregate::semantics_v2::retained::graph::{
    ReservedSemanticsV2Graph, RetainedDisposition, RetainedSequenceEffect, RetainedStatement,
    RetainedStatementOutcome,
};
use crate::EngineError;

const S1_BYTES: u64 = 144;
const S4_BYTES: u64 = 64;
const S6_BYTES: u64 = 136;
const S5_PREFIX_BYTES: u64 = 52;

pub(super) fn fill_fixed_sections(
    framing: &DecodedAggregateFraming<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    fill_s1(framing, graph)?;
    fill_s4(framing, graph)?;
    fill_s5(framing, graph)?;
    fill_s6(framing, graph)
}

fn fill_s1(
    framing: &DecodedAggregateFraming<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    framing.with_section_reader(0, |reader| {
        while !reader.done() {
            let raw = reader.exact::<144>()?;
            push_exact(
                &mut graph.statements,
                RetainedStatement {
                    statement_ordinal: u32_at(&raw, 0),
                    family_ordinal: u32_at(&raw, 4),
                    input_row_count: u32_at(&raw, 12),
                    request_digest: digest_at(&raw, 16),
                    typed_statement_digest: digest_at(&raw, 48),
                    overlay_before: digest_at(&raw, 80),
                    overlay_after: digest_at(&raw, 112),
                    record_bytes: 0,
                    record_digest: [0; 32],
                },
                "S1 statement directory",
            )?;
        }
        Ok(())
    })?;
    if u64::try_from(graph.statements.len())
        .ok()
        .and_then(|count| count.checked_mul(S1_BYTES))
        != Some(framing.sections()[0].payload_bytes)
    {
        return Err(fill_error(
            "S1 retained count does not cover its fixed bytes",
        ));
    }
    Ok(())
}

fn fill_s4(
    framing: &DecodedAggregateFraming<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    framing.with_section_reader(3, |reader| {
        while !reader.done() {
            let raw = reader.exact::<64>()?;
            push_exact(
                &mut graph.dispositions,
                RetainedDisposition {
                    statement_ordinal: u32_at(&raw, 0),
                    source_row_ordinal: u32_at(&raw, 4),
                    stable_row_id: u64_at(&raw, 8),
                    disposition: raw[16],
                    table_ref: u32_at(&raw, 20),
                    transition_ref: u32_at(&raw, 24),
                    typed_statement_digest: digest_at(&raw, 32),
                },
                "S4 disposition directory",
            )?;
        }
        Ok(())
    })?;
    if u64::try_from(graph.dispositions.len())
        .ok()
        .and_then(|count| count.checked_mul(S4_BYTES))
        != Some(framing.sections()[3].payload_bytes)
    {
        return Err(fill_error(
            "S4 retained count does not cover its fixed bytes",
        ));
    }
    Ok(())
}

fn fill_s5(
    framing: &DecodedAggregateFraming<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    framing.with_section_reader(4, |reader| {
        while !reader.done() {
            if reader.remaining() < S5_PREFIX_BYTES {
                return Err(fill_error("S5 retained effect prefix is truncated"));
            }
            let statement_ordinal = reader.u32()?;
            let effect_ordinal = reader.u32()?;
            let kind = reader.u8()?;
            let flags = reader.u8()?;
            let reserved = reader.u16()?;
            let disposition_ref = reader.u32()?;
            let body_bytes = reader.u32()?;
            let body_digest = reader.digest()?;
            if kind != 1
                || reserved != 0
                || body_bytes != crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES as u32
                || u64::from(body_bytes) > reader.remaining()
            {
                return Err(fill_error("S5 retained published effect shape is invalid"));
            }
            let mut body = [0_u8; crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES];
            reader.copy_exact(&mut body)?;
            if gpu_db_wal::canonical_request_digest(&body) != body_digest {
                return Err(fill_error("S5 retained sequence body digest drifted"));
            }
            let reference = crate::decode_sequence_value_reference_exact(&body)
                .map_err(|_| fill_error("S5 retained sequence reference fails strict decode"))?;
            push_exact(
                &mut graph.sequence_effects,
                RetainedSequenceEffect {
                    statement_ordinal,
                    effect_ordinal,
                    disposition_ref,
                    flags,
                    body_digest,
                    reference,
                },
                "S5 sequence-effect directory",
            )?;
        }
        Ok(())
    })
}

fn fill_s6(
    framing: &DecodedAggregateFraming<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    framing.with_section_reader(5, |reader| {
        while !reader.done() {
            let raw = reader.exact::<136>()?;
            let outcome = gpu_db_wal::decode_canonical_outcome_exact(&raw[44..136])
                .map_err(|_| fill_error("S6 retained canonical outcome fails strict decode"))?;
            push_exact(
                &mut graph.outcomes,
                RetainedStatementOutcome {
                    statement_ordinal: u32_at(&raw, 0),
                    semantic_class: u16_at(&raw, 8),
                    flags: u16_at(&raw, 10),
                    typed_statement_digest: digest_at(&raw, 12),
                    outcome,
                },
                "S6 outcome directory",
            )?;
        }
        Ok(())
    })?;
    if u64::try_from(graph.outcomes.len())
        .ok()
        .and_then(|count| count.checked_mul(S6_BYTES))
        != Some(framing.sections()[5].payload_bytes)
    {
        return Err(fill_error(
            "S6 retained count does not cover its fixed bytes",
        ));
    }
    Ok(())
}

pub(super) fn push_exact<T>(values: &mut Vec<T>, value: T, owner: &str) -> Result<(), EngineError> {
    if values.len() == values.capacity() {
        return Err(fill_error(&format!(
            "{owner} exceeds its pass-zero exact reservation"
        )));
    }
    values.push(value);
    Ok(())
}

pub(super) fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("fixed u16 field"),
    )
}

pub(super) fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("fixed u32 field"),
    )
}

pub(super) fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("fixed u64 field"),
    )
}

pub(super) fn i16_at(bytes: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("fixed i16 field"),
    )
}

pub(super) fn digest_at(bytes: &[u8], offset: usize) -> [u8; 32] {
    bytes[offset..offset + 32]
        .try_into()
        .expect("fixed digest field")
}
