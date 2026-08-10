//! Allocation-free aggregate/S1--S6 bindings for codec-5 semantics v2.
//!
//! These checks deliberately stop at the section boundary.  S7 owns the table-local durable
//! graph; this module establishes the fixed aggregate profile and the S1/S2/S4/S5/S6 matrix
//! before that graph is traversed.

use super::super::super::super::codec::DecodedAggregateFraming;
use super::{checked_mul, error, read_u32, ABSENT_U32, S4_BYTES};
use crate::typed_insert_aggregate::{
    AGGREGATE_FLAG_AUTOCOMMIT, AGGREGATE_FLAG_CATALOG, AGGREGATE_FLAG_EXPLICIT,
    AGGREGATE_FLAG_OPERATION_COMPOSITION, AGGREGATE_FLAG_PRIVATE_SEQUENCE,
    AGGREGATE_FLAG_PUBLISHED_SEQUENCE, AGGREGATE_FLAG_RETAINED_RESPONSE, AGGREGATE_FLAG_RETURNING,
    OUTER_CONTENT_CATALOG, OUTER_CONTENT_OPERATION_COMPOSITION, OUTER_CONTENT_PRIVATE_SEQUENCE,
    OUTER_CONTENT_PUBLISHED_SEQUENCE, OUTER_CONTENT_RETURNING, OUTER_CONTENT_ROW,
    OUTER_FLAG_FIRST_TYPED_INSERT_WRITER_EPOCH, OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1,
};
use crate::typed_insert_batch::{
    measure_decoded_canonical_typed_insert_from_source, parse_canonical_typed_insert_record_prefix,
    CanonicalTypedInsertReadAt, CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES,
};
use crate::EngineError;

const S5_PREFIX_BYTES: u64 = 52;
const S5_PUBLISHED: u8 = 1;
const S5_PRIVATE: u8 = 2;
const S5_DEFAULT_EXPRESSION: u8 = 1;
const S5_FINAL_OVERWRITTEN: u8 = 1 << 1;

pub(super) fn validate_scalar_and_outer_flags(
    framing: &DecodedAggregateFraming<'_>,
) -> Result<(), EngineError> {
    let scalar = framing.header_scalars();
    let aggregate_allowed = AGGREGATE_FLAG_AUTOCOMMIT
        | AGGREGATE_FLAG_EXPLICIT
        | AGGREGATE_FLAG_CATALOG
        | AGGREGATE_FLAG_OPERATION_COMPOSITION
        | AGGREGATE_FLAG_PRIVATE_SEQUENCE
        | AGGREGATE_FLAG_PUBLISHED_SEQUENCE
        | AGGREGATE_FLAG_RETURNING
        | AGGREGATE_FLAG_RETAINED_RESPONSE;
    let mode = scalar.flags & (AGGREGATE_FLAG_AUTOCOMMIT | AGGREGATE_FLAG_EXPLICIT);
    if scalar.stable_transaction_id == 0
        || scalar.flags & !aggregate_allowed != 0
        || !matches!(mode, AGGREGATE_FLAG_AUTOCOMMIT | AGGREGATE_FLAG_EXPLICIT)
        || (mode == AGGREGATE_FLAG_AUTOCOMMIT && scalar.statement_count != 1)
    {
        return Err(error(
            "v2 aggregate flags, mode, or transaction identity are invalid",
        ));
    }
    let outer = framing.outer_flags();
    let outer_allowed = OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1
        | OUTER_FLAG_FIRST_TYPED_INSERT_WRITER_EPOCH
        | OUTER_CONTENT_ROW
        | OUTER_CONTENT_CATALOG
        | OUTER_CONTENT_OPERATION_COMPOSITION
        | OUTER_CONTENT_PRIVATE_SEQUENCE
        | OUTER_CONTENT_PUBLISHED_SEQUENCE
        | OUTER_CONTENT_RETURNING;
    if outer & !outer_allowed != 0
        || outer & OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 == 0
        || outer & OUTER_CONTENT_ROW == 0
        || (scalar.flags & AGGREGATE_FLAG_CATALOG != 0) != (outer & OUTER_CONTENT_CATALOG != 0)
        || (scalar.flags & AGGREGATE_FLAG_OPERATION_COMPOSITION != 0)
            != (outer & OUTER_CONTENT_OPERATION_COMPOSITION != 0)
        || (scalar.flags & AGGREGATE_FLAG_PUBLISHED_SEQUENCE != 0)
            != (outer & OUTER_CONTENT_PUBLISHED_SEQUENCE != 0)
        || (scalar.flags & AGGREGATE_FLAG_PRIVATE_SEQUENCE != 0)
            != (outer & OUTER_CONTENT_PRIVATE_SEQUENCE != 0)
        || (scalar.flags & AGGREGATE_FLAG_RETURNING != 0) != (outer & OUTER_CONTENT_RETURNING != 0)
    {
        return Err(error(
            "v2 outer content flags are not the closed typed-INSERT profile",
        ));
    }
    let status = framing.status();
    if status.txn_id != scalar.stable_transaction_id
        || status.statement_count != scalar.statement_count
    {
        return Err(error(
            "STATUS2 does not match the v2 aggregate identity profile",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct S2PassZeroMeasure {
    pub(super) largest_record_bytes: u64,
    pub(super) persistent_bytes: u64,
    pub(super) persistent_slots: u64,
    pub(super) maximum_scratch_bytes: u64,
    pub(super) maximum_scratch_slots: u64,
}

pub(super) fn measure_s2(
    framing: &DecodedAggregateFraming<'_>,
) -> Result<S2PassZeroMeasure, EngineError> {
    framing.with_section_reader(0, |s1| {
        framing.with_section_reader(1, |reader| {
            let mut largest = 0_u64;
            let mut persistent_bytes = 0_u64;
            let mut persistent_slots = 0_u64;
            let mut maximum_scratch_bytes = 0_u64;
            let mut maximum_scratch_slots = 0_u64;
            let mut statement = 0_u32;
            while reader.remaining() != 0 {
                let record_start = framing.sections()[1]
                    .payload_bytes
                    .checked_sub(reader.remaining())
                    .ok_or_else(|| error("S2 record start is outside its section"))?;
                let bytes = u64::from(reader.u32()?);
                if !(100..=16 * 1024 * 1024).contains(&bytes) || bytes > reader.remaining() {
                    return Err(error("S2 record length is not bounded by its section"));
                }
                let record_bytes =
                    usize::try_from(bytes).map_err(|_| error("S2 record length addressability"))?;
                let prefix = reader.exact::<CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES>()?;
                let prefix = parse_canonical_typed_insert_record_prefix(&prefix, record_bytes)
                    .map_err(|_| error("S2 canonical typed-record prefix is invalid"))?;
                let s1 = s1.exact::<144>()?;
                if read_u32(&s1, 0) != statement
                    || s1[8] != 1
                    || prefix.typed_statement_digest != s1[48..80]
                {
                    return Err(error("S1/S2 typed-record digest/order closure is invalid"));
                }
                let source = S2RecordSource {
                    framing,
                    record_start: record_start
                        .checked_add(4)
                        .ok_or_else(|| error("S2 record source offset overflows"))?,
                    bytes,
                };
                let measured = measure_decoded_canonical_typed_insert_from_source(&source)
                    .map_err(|_| error("S2 source measure rejects the canonical record"))?;
                if measured.record_bytes() != bytes {
                    return Err(error("S2 strict source measure length drifted"));
                }
                reader.skip(bytes - CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES as u64)?;
                largest = largest.max(bytes);
                persistent_bytes = persistent_bytes
                    .checked_add(measured.persistent_bytes())
                    .ok_or_else(|| error("S2 persistent owner bytes overflow"))?;
                persistent_slots = persistent_slots
                    .checked_add(measured.persistent_allocation_slots())
                    .ok_or_else(|| error("S2 persistent owner slots overflow"))?;
                maximum_scratch_bytes = maximum_scratch_bytes.max(
                    measured
                        .maximum_with_record_copy_bytes()
                        .map_err(|_| error("S2 copy plus decoder scratch overflows"))?,
                );
                maximum_scratch_slots = maximum_scratch_slots.max(
                    measured
                        .maximum_with_record_copy_allocation_slots()
                        .map_err(|_| error("S2 copy plus decoder scratch slots overflow"))?,
                );
                statement = statement
                    .checked_add(1)
                    .ok_or_else(|| error("S2 statement count overflows"))?;
            }
            if s1.remaining() != 0 {
                return Err(error(
                    "S2 did not consume one typed record per S1 statement",
                ));
            }
            Ok(S2PassZeroMeasure {
                largest_record_bytes: largest,
                persistent_bytes,
                persistent_slots,
                maximum_scratch_bytes,
                maximum_scratch_slots,
            })
        })
    })
}

/// One S2 record embedded in the chunk-backed aggregate section. The source is used only for
/// strict raw measurement and an eventual exact post-reservation copy; it never exposes a slice
/// of the aggregate or creates an owned record during pass zero.
struct S2RecordSource<'a> {
    framing: &'a DecodedAggregateFraming<'a>,
    record_start: u64,
    bytes: u64,
}

impl CanonicalTypedInsertReadAt for S2RecordSource<'_> {
    fn len(&self) -> u64 {
        self.bytes
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), EngineError> {
        let requested =
            u64::try_from(out.len()).map_err(|_| error("S2 source read length overflows"))?;
        let end = offset
            .checked_add(requested)
            .filter(|end| *end <= self.bytes)
            .ok_or_else(|| error("S2 source read is outside its measured record"))?;
        self.framing.with_section_reader(1, |reader| {
            reader.skip(
                self.record_start
                    .checked_add(offset)
                    .ok_or_else(|| error("S2 source absolute read offset overflows"))?,
            )?;
            reader.copy_exact(out)?;
            reader.skip(reader.remaining())?;
            Ok(())
        })?;
        debug_assert_eq!(end, offset + requested);
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(super) struct S5PassZeroMeasure {
    pub(super) effect_count: u32,
    pub(super) published_count: u32,
    pub(super) private_count: u32,
}

/// S5 admits the existing two canonical sequence forms. A later terminal RESTART may suffix the
/// existing effect body without changing the S2/S5 entry bijection. Semantic closure belongs to
/// the retained-model pass.
pub(super) fn measure_s5(
    framing: &DecodedAggregateFraming<'_>,
    statement_count: u32,
    original_rows: u64,
) -> Result<S5PassZeroMeasure, EngineError> {
    let expected_body = crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES as u32;
    framing.with_section_reader(4, |reader| {
        let mut count = 0_u32;
        let mut published_count = 0_u32;
        let mut private_count = 0_u32;
        let mut prior_transition = None;
        while reader.remaining() != 0 {
            if reader.remaining() < S5_PREFIX_BYTES {
                return Err(error("S5 effect prefix is truncated"));
            }
            let statement = reader.u32()?;
            let effect = reader.u32()?;
            let kind = reader.u8()?;
            let flags = reader.u8()?;
            if reader.u16()? != 0 {
                return Err(error("S5 effect reserved bytes are nonzero"));
            }
            let disposition = reader.u32()?;
            let body_bytes = reader.u32()?;
            let digest = reader.digest()?;
            if statement >= statement_count
                || effect == ABSENT_U32
                || !matches!(kind, S5_PUBLISHED | S5_PRIVATE)
                || flags & !(S5_DEFAULT_EXPRESSION | S5_FINAL_OVERWRITTEN) != 0
                || flags & S5_DEFAULT_EXPRESSION == 0
                || disposition == ABSENT_U32
                || u64::from(disposition) >= original_rows
                || u64::from(body_bytes) > reader.remaining()
            {
                return Err(error("S5 sequence-effect shell is invalid"));
            }
            match kind {
                S5_PUBLISHED => {
                    let with_restart = expected_body
                        + crate::typed_insert_aggregate::semantics_v2::sequence_terminal::SEQUENCE_RESTART_TAIL_BYTES as u32;
                    if !matches!(body_bytes, value if value == expected_body || value == with_restart)
                        || digest == [0; 32]
                    {
                        return Err(error("S5 published sequence-effect shell is invalid"));
                    }
                    let mut full = [0_u8;
                        crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES
                            + crate::typed_insert_aggregate::semantics_v2::sequence_terminal::SEQUENCE_RESTART_TAIL_BYTES];
                    reader.copy_exact(&mut full[..crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES])?;
                    if body_bytes == with_restart {
                        let tail = &mut full[crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES..];
                        reader.copy_exact(tail)?;
                        crate::typed_insert_aggregate::semantics_v2::sequence_terminal::decode(tail)
                            .map_err(|_| error("S5 terminal restart fails strict decode"))?;
                    }
                    if gpu_db_wal::canonical_request_digest(&full[..body_bytes as usize]) != digest {
                        return Err(error("S5 published sequence body digest is invalid"));
                    }
                    let reference = crate::decode_sequence_value_reference_exact(
                        &full[..crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES]
                    )
                        .map_err(|_| error("S5 published sequence body fails strict decode"))?;
                    if !reference.default_expression
                        || prior_transition
                            .is_some_and(|prior| prior >= reference.transition_txn_id)
                    {
                        return Err(error("S5 published sequence transition order is invalid"));
                    }
                    prior_transition = Some(reference.transition_txn_id);
                    published_count = published_count
                        .checked_add(1)
                        .ok_or_else(|| error("S5 published count overflows"))?;
                }
                S5_PRIVATE => {
                    let tail_bytes = crate::typed_insert_aggregate::semantics_v2::sequence_terminal::SEQUENCE_RESTART_TAIL_BYTES as u32;
                    if !(body_bytes == 0 || body_bytes == tail_bytes)
                        || (body_bytes == 0) != (digest == [0; 32])
                    {
                        return Err(error("S5 private sequence body is not canonical"));
                    }
                    if body_bytes == tail_bytes {
                        let mut tail = [0_u8;
                            crate::typed_insert_aggregate::semantics_v2::sequence_terminal::SEQUENCE_RESTART_TAIL_BYTES];
                        reader.copy_exact(&mut tail)?;
                        if gpu_db_wal::canonical_request_digest(&tail) != digest {
                            return Err(error("S5 private terminal body digest is invalid"));
                        }
                        crate::typed_insert_aggregate::semantics_v2::sequence_terminal::decode(&tail)
                            .map_err(|_| error("S5 terminal restart fails strict decode"))?;
                    }
                    private_count = private_count
                        .checked_add(1)
                        .ok_or_else(|| error("S5 private count overflows"))?;
                }
                _ => unreachable!("kind checked above"),
            }
            count = count
                .checked_add(1)
                .ok_or_else(|| error("S5 entry count overflows"))?;
        }
        if framing.sections()[4].entry_count != count {
            return Err(error(
                "S5 entry count does not equal its canonical byte traversal",
            ));
        }
        Ok(S5PassZeroMeasure {
            effect_count: count,
            published_count,
            private_count,
        })
    })
}

#[derive(Clone, Copy)]
pub(super) struct TerminalS6 {
    pub(super) abort_at: Option<u32>,
    pub(super) kind: gpu_db_wal::CanonicalOutcomeKind,
    pub(super) sqlstate: Option<[u8; 5]>,
    pub(super) constraint_id: u64,
}

pub(super) fn scan_s6_terminal(
    framing: &DecodedAggregateFraming<'_>,
    statement_count: u32,
) -> Result<TerminalS6, EngineError> {
    framing.with_section_reader(5, |s6| {
        if s6.remaining() != checked_mul(u64::from(statement_count), 136)? {
            return Err(error("S6 fixed outcome bytes are not exact"));
        }
        let mut abort_at = None;
        let mut final_outcome = None;
        for ordinal in 0..statement_count {
            let raw = s6.exact::<136>()?;
            let outcome = gpu_db_wal::decode_canonical_outcome_exact(&raw[44..136])
                .map_err(|_| error("S6 canonical outcome is invalid"))?;
            match outcome.kind {
                gpu_db_wal::CanonicalOutcomeKind::CommitSuccess => {
                    if ordinal + 1 == statement_count {
                        final_outcome = Some(outcome);
                    }
                }
                gpu_db_wal::CanonicalOutcomeKind::AbortError if ordinal + 1 == statement_count => {
                    if abort_at.replace(ordinal).is_some() {
                        return Err(error("S6 has multiple terminal abort outcomes"));
                    }
                    final_outcome = Some(outcome);
                }
                gpu_db_wal::CanonicalOutcomeKind::AbortError => {
                    return Err(error("S6 abort is not the final statement"));
                }
                gpu_db_wal::CanonicalOutcomeKind::CommitNoOp => {
                    return Err(error("S6 CommitNoOp is invalid for semantics v2"));
                }
            }
        }
        let outcome = final_outcome.ok_or_else(|| error("S6 has no terminal outcome"))?;
        Ok(TerminalS6 {
            abort_at,
            kind: outcome.kind,
            sqlstate: outcome.sqlstate,
            constraint_id: outcome.constraint_id,
        })
    })
}

pub(super) fn measure_s1_s4_s6(
    framing: &DecodedAggregateFraming<'_>,
    statement_count: u32,
    original_rows: u64,
    abort_at: Option<u32>,
) -> Result<u64, EngineError> {
    framing.with_section_reader(0, |s1| {
        framing.with_section_reader(3, |s4| {
            framing.with_section_reader(5, |s6| {
                const S1_BYTES: u64 = 144;
                if s1.remaining() != checked_mul(u64::from(statement_count), S1_BYTES)?
                    || s4.remaining() != checked_mul(original_rows, S4_BYTES)?
                {
                    return Err(error("S1/S4 fixed entry bytes are not exact"));
                }
                let mut prior_statement = 0_u32;
                let mut prior_source = 0_u32;
                let mut total_rows = 0_u64;
                let mut survivors = 0_u64;
                let mut prior_overlay_after = None;
                for expected_statement in 0..statement_count {
                    let raw = s1.exact::<144>()?;
                    let statement = read_u32(&raw, 0);
                    let family_ordinal = read_u32(&raw, 4);
                    let family = raw[8];
                    let input_rows = read_u32(&raw, 12);
                    let request_digest: [u8; 32] =
                        raw[16..48].try_into().expect("fixed S1 request");
                    let statement_digest: [u8; 32] =
                        raw[48..80].try_into().expect("fixed S1 digest");
                    let overlay_before: [u8; 32] =
                        raw[80..112].try_into().expect("fixed S1 before");
                    let overlay_after: [u8; 32] = raw[112..144].try_into().expect("fixed S1 after");
                    if statement != expected_statement
                        || family_ordinal != expected_statement
                        || family != 1
                        || raw[9] != 0
                        || raw[10..12].iter().any(|byte| *byte != 0)
                        || input_rows == 0
                        || request_digest == [0; 32]
                        || request_digest != statement_digest
                        || overlay_before == [0; 32]
                        || overlay_after == [0; 32]
                        || prior_overlay_after.is_some_and(|prior| prior != overlay_before)
                    {
                        return Err(error(
                            "S1 typed-INSERT/ordinal/digest/overlay closure is invalid",
                        ));
                    }
                    prior_overlay_after = Some(overlay_after);
                    let outcome_entry = s6.exact::<136>()?;
                    let flags = u16::from_le_bytes(
                        outcome_entry[10..12].try_into().expect("fixed S6 flags"),
                    );
                    let outcome =
                        gpu_db_wal::decode_canonical_outcome_exact(&outcome_entry[44..136])
                            .map_err(|_| error("S6 canonical outcome is invalid"))?;
                    if read_u32(&outcome_entry, 0) != expected_statement
                        || read_u32(&outcome_entry, 4) != expected_statement
                        || u16::from_le_bytes(
                            outcome_entry[8..10].try_into().expect("fixed S6 class"),
                        ) != 1
                        || flags & !3 != 0
                        || (flags & 2 != 0 && flags & 1 == 0)
                        || outcome_entry[12..44] != statement_digest
                        || outcome.target_digest != overlay_after
                    {
                        return Err(error(
                            "S6/S1 typed-INSERT identity/overlay closure is invalid",
                        ));
                    }
                    match (abort_at, outcome.kind) {
                        (None, gpu_db_wal::CanonicalOutcomeKind::CommitSuccess)
                            if outcome.affected_rows == u64::from(input_rows)
                                && outcome.sqlstate.is_none()
                                && outcome.constraint_id == 0
                                && ((flags & 1 != 0) == (outcome.returning_digest != [0; 32])) => {}
                        (Some(final_abort), gpu_db_wal::CanonicalOutcomeKind::CommitSuccess)
                            if expected_statement < final_abort
                                && outcome.affected_rows == u64::from(input_rows)
                                && outcome.sqlstate.is_none()
                                && outcome.constraint_id == 0
                                && ((flags & 1 != 0) == (outcome.returning_digest != [0; 32])) => {}
                        (Some(final_abort), gpu_db_wal::CanonicalOutcomeKind::AbortError)
                            if expected_statement == final_abort
                                && outcome.affected_rows == 0
                                && outcome.returning_digest == [0; 32]
                                && outcome.constraint_id != 0
                                && outcome.sqlstate.is_some_and(is_constraint_sqlstate)
                                && flags & 2 == 0 => {}
                        _ => return Err(error("S6 outcome does not match the v2 terminal matrix")),
                    }
                    for expected_source in 0..input_rows {
                        let statement = s4.u32()?;
                        let source = s4.u32()?;
                        let row_id = s4.u64()?;
                        let kind = s4.u8()?;
                        let writer_flag = s4.u8()?;
                        let reserved = s4.u16()?;
                        let table = s4.u32()?;
                        let transition = s4.u32()?;
                        let final_writer_statement = s4.u32()?;
                        let digest = s4.digest()?;
                        let ordinary = kind == expected_s4_kind(abort_at, expected_statement)
                            && writer_flag == 0
                            && final_writer_statement == 0
                            && digest == statement_digest
                            && ((kind == 1 && transition != u32::MAX)
                                || (kind != 1 && transition == u32::MAX));
                        let committed_cancellation = abort_at.is_none()
                            && kind == 2
                            && writer_flag == 1
                            && transition == u32::MAX
                            && final_writer_statement > expected_statement
                            && digest != [0; 32];
                        if statement != expected_statement
                            || source != expected_source
                            || row_id == 0
                            || row_id == u64::MAX
                            || table == u32::MAX
                            || reserved != 0
                            || !(ordinary || committed_cancellation)
                            || (total_rows != 0
                                && (statement < prior_statement
                                    || (statement == prior_statement
                                        && source
                                            != prior_source.checked_add(1).ok_or_else(|| {
                                                error("S4 source ordinal overflow")
                                            })?)))
                        {
                            return Err(error(
                                "S4/S1 statement/source/digest/reference closure is invalid",
                            ));
                        }
                        survivors = survivors
                            .checked_add(u64::from(kind == 1))
                            .ok_or_else(|| error("S4 survivor count overflows"))?;
                        prior_statement = statement;
                        prior_source = source;
                        total_rows = total_rows
                            .checked_add(1)
                            .ok_or_else(|| error("S4 row count overflows"))?;
                    }
                }
                if total_rows != original_rows {
                    return Err(error("S1 input rows do not equal aggregate S4 rows"));
                }
                Ok(survivors)
            })
        })
    })
}

fn expected_s4_kind(abort_at: Option<u32>, statement: u32) -> u8 {
    match abort_at {
        None => 1,
        Some(final_abort) if statement < final_abort => 2,
        Some(_) => 3,
    }
}

fn is_constraint_sqlstate(state: [u8; 5]) -> bool {
    matches!(
        state,
        [b'2', b'3', b'5', b'0', b'2']
            | [b'2', b'3', b'5', b'0', b'3']
            | [b'2', b'3', b'5', b'0', b'5']
            | [b'2', b'3', b'5', b'1', b'4']
    )
}
