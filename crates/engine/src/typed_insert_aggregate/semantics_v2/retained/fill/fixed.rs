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
use sha2::{Digest, Sha256};

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
            let has_final_writer = raw[17] == 1;
            push_exact(
                &mut graph.dispositions,
                RetainedDisposition {
                    statement_ordinal: u32_at(&raw, 0),
                    source_row_ordinal: u32_at(&raw, 4),
                    stable_row_id: u64_at(&raw, 8),
                    disposition: raw[16],
                    table_ref: u32_at(&raw, 20),
                    transition_ref: u32_at(&raw, 24),
                    typed_statement_digest: if has_final_writer {
                        [0; 32]
                    } else {
                        digest_at(&raw, 32)
                    },
                    final_writer_statement_ordinal: if has_final_writer {
                        u32_at(&raw, 28)
                    } else {
                        u32_at(&raw, 0)
                    },
                    final_writer_statement_digest: if has_final_writer {
                        digest_at(&raw, 32)
                    } else {
                        [0; 32]
                    },
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
            if reserved != 0 || u64::from(body_bytes) > reader.remaining() {
                return Err(fill_error("S5 retained effect shape is invalid"));
            }
            let mut reference_body_digest = [0; 32];
            let mut terminal_restart = None;
            let reference = match kind {
                1 => {
                    let base = crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES as u32;
                    let with_restart = base
                        + crate::typed_insert_aggregate::semantics_v2::sequence_terminal::SEQUENCE_RESTART_TAIL_BYTES
                            as u32;
                    if !matches!(body_bytes, value if value == base || value == with_restart) {
                        return Err(fill_error(
                            "S5 retained published effect body width is invalid",
                        ));
                    }
                    let mut body = [0_u8; crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES];
                    reader.copy_exact(&mut body)?;
                    reference_body_digest = gpu_db_wal::canonical_request_digest(&body);
                    let mut full = [0_u8;
                        crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES
                            + crate::typed_insert_aggregate::semantics_v2::sequence_terminal::SEQUENCE_RESTART_TAIL_BYTES];
                    full[..crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES]
                        .copy_from_slice(&body);
                    if body_bytes == with_restart {
                        let tail = &mut full[crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES..];
                        reader.copy_exact(tail)?;
                        terminal_restart = Some(
                            crate::typed_insert_aggregate::semantics_v2::sequence_terminal::decode(
                                tail,
                            )
                            .map_err(|_| fill_error("S5 terminal restart fails strict decode"))?,
                        );
                    }
                    if gpu_db_wal::canonical_request_digest(&full[..body_bytes as usize])
                        != body_digest
                    {
                        return Err(fill_error("S5 retained sequence body digest drifted"));
                    }
                    Some(
                        crate::decode_sequence_value_reference_exact(&body).map_err(|_| {
                            fill_error("S5 retained sequence reference fails strict decode")
                        })?,
                    )
                }
                2 => {
                    let tail_bytes = crate::typed_insert_aggregate::semantics_v2::sequence_terminal::SEQUENCE_RESTART_TAIL_BYTES as u32;
                    if !(body_bytes == 0 || body_bytes == tail_bytes)
                        || (body_bytes == 0) != (body_digest == [0; 32])
                    {
                        return Err(fill_error(
                            "S5 retained private effect body is not canonical",
                        ));
                    }
                    if body_bytes == tail_bytes {
                        let mut tail = [0_u8;
                            crate::typed_insert_aggregate::semantics_v2::sequence_terminal::SEQUENCE_RESTART_TAIL_BYTES];
                        reader.copy_exact(&mut tail)?;
                        if gpu_db_wal::canonical_request_digest(&tail) != body_digest {
                            return Err(fill_error("S5 private terminal body digest drifted"));
                        }
                        terminal_restart = Some(
                            crate::typed_insert_aggregate::semantics_v2::sequence_terminal::decode(
                                &tail,
                            )
                            .map_err(|_| fill_error("S5 terminal restart fails strict decode"))?,
                        );
                    }
                    None
                }
                _ => return Err(fill_error("S5 retained sequence effect kind is invalid")),
            };
            push_exact(
                &mut graph.sequence_effects,
                RetainedSequenceEffect {
                    statement_ordinal,
                    effect_ordinal,
                    disposition_ref,
                    flags,
                    body_digest,
                    reference_body_digest,
                    reference,
                    terminal_restart,
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
                    family_ordinal: u32_at(&raw, 4),
                    semantic_class: u16_at(&raw, 8),
                    flags: u16_at(&raw, 10),
                    typed_statement_digest: digest_at(&raw, 12),
                    outcome_digest: s6_entry_digest(&raw),
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

fn s6_entry_digest(raw: &[u8; S6_BYTES as usize]) -> [u8; 32] {
    let domain = b"gpu-db/write001/s7-s6-entry/v2";
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(raw);
    digest.finalize().into()
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
