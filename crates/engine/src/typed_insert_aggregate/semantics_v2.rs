//! Inert codec-5 semantics-v2 S4/S7 ownership facade.
//!
//! The only entry point consumes canonical aggregate chunk/status framing after semantic-version
//! dispatch.  It is production-compiled but unreachable from WAL, recovery, apply, and
//! publication.  Test-only byte reencoding is added only after the complete retained model has
//! passed catalog/lease and generation-witness validation.

#[cfg(test)]
#[path = "semantics_v2/goldens.rs"]
mod goldens;
#[path = "semantics_v2/pass_zero.rs"]
mod pass_zero;
#[path = "semantics_v2/retained.rs"]
mod retained;

use super::codec::decode_aggregate_framing;
use crate::typed_insert_aggregate::{
    AGGREGATE_FLAG_AUTOCOMMIT, AGGREGATE_FLAG_EXPLICIT, AGGREGATE_MAX_CHUNKS,
};
use crate::EngineError;

/// Run the canonical physical proof and the semantics-v2 allocation-free pass zero directly over
/// the original codec-5 chunk bodies.  The returned measurement has no decoded raw aggregate,
/// S2, image, catalog, lease, generation, replay, or publication capability.
///
/// This is deliberately crate-private and has no production caller while the remaining retained
/// S1--S7 owner is built.  It exists to make version dispatch explicit and to ensure v2 never
/// takes the historical semantics-v1 global-allocator path.
pub(super) fn measure_canonical_semantics_v2<'a>(
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    outcome: &gpu_db_wal::CanonicalOutcome,
    fragments: &[gpu_db_wal::CanonicalFragmentRef<'a>],
) -> Result<pass_zero::SemanticsV2PassZero, EngineError> {
    if !(2..=AGGREGATE_MAX_CHUNKS + 1).contains(&fragments.len()) {
        return Err(v2_error(
            "canonical fragment count is outside codec-5 bounds",
        ));
    }
    let chunk_count = fragments.len() - 1;
    if fragments[..chunk_count]
        .iter()
        .any(|fragment| fragment.kind != gpu_db_wal::CanonicalFragmentKind::RowMutation)
        || fragments[chunk_count].kind != gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus
    {
        return Err(v2_error(
            "v2 aggregate requires row chunks followed by one STATUS2 fragment",
        ));
    }
    let mut bodies = [&[][..]; AGGREGATE_MAX_CHUNKS + 1];
    for (target, fragment) in bodies.iter_mut().zip(fragments.iter()) {
        *target = fragment.body;
    }
    let framing = decode_aggregate_framing(outer.flags, &bodies[..fragments.len()])?;
    if framing.semantics_version() != crate::typed_insert_aggregate::AGGREGATE_SEMANTICS_V2 {
        return Err(v2_error("reader received a non-v2 aggregate"));
    }
    let measured = pass_zero::measure(&framing, outer)?;
    let scalar = framing.header_scalars();
    let status = framing.status();
    if outer.stable_transaction_id != scalar.stable_transaction_id
        || outer.identity.database_id != status.database_id
        || outer.identity.timeline_id != status.timeline_id
        || outer.request_digest != status.request_digest
        || outer.isolation as u8 != status.isolation
        || outer.flags != framing.outer_flags()
        || outer.operation_count != fragments.len() as u32
        || outer.table_block_count != scalar.table_block_count
        || outer.allocator_high_water != 0
        || outcome.target_digest != framing.aggregate_root()
        || outcome.returning_digest != status.response_root
        || matches!(outcome.kind, gpu_db_wal::CanonicalOutcomeKind::CommitNoOp)
    {
        return Err(v2_error(
            "outer/header/status/outcome v2 identity closure is invalid",
        ));
    }
    match outcome.kind {
        gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
            if outcome.sqlstate.is_none()
                && outcome.constraint_id == 0
                && measured.terminal_kind == gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
                && measured.terminal_sqlstate.is_none()
                && measured.terminal_constraint_id == 0
                && outcome.affected_rows
                    == if scalar.flags & AGGREGATE_FLAG_AUTOCOMMIT != 0 {
                        scalar.original_inserted_row_count
                    } else if scalar.flags & AGGREGATE_FLAG_EXPLICIT != 0 {
                        scalar.final_row_transition_count
                    } else {
                        return Err(v2_error("aggregate mode vanished after pass zero"));
                    } => {}
        gpu_db_wal::CanonicalOutcomeKind::AbortError
            if outcome.affected_rows == 0
                && measured.terminal_kind == gpu_db_wal::CanonicalOutcomeKind::AbortError
                && outcome.sqlstate == measured.terminal_sqlstate
                && outcome.constraint_id == measured.terminal_constraint_id => {}
        _ => {
            return Err(v2_error(
                "outer outcome does not follow the v2 terminal matrix",
            ));
        }
    }
    Ok(measured)
}

/// Test-only end-to-end retained construction.  Production has no caller and no exported
/// conversion from its quarantine owner; this bridge exists solely to prove that the frozen
/// golden bytes can complete post-reservation source/image filling and immediately drain on a
/// retryable failure.
#[cfg(test)]
fn fill_canonical_semantics_v2_for_test<'a>(
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    outcome: &gpu_db_wal::CanonicalOutcome,
    fragments: &[gpu_db_wal::CanonicalFragmentRef<'a>],
) -> Result<retained::QuarantinedSemanticsV2, EngineError> {
    let measured = measure_canonical_semantics_v2(outer, outcome, fragments)?;
    let mut bodies = [&[][..]; AGGREGATE_MAX_CHUNKS + 1];
    for (target, fragment) in bodies.iter_mut().zip(fragments.iter()) {
        *target = fragment.body;
    }
    let framing = decode_aggregate_framing(outer.flags, &bodies[..fragments.len()])?;
    retained::fill_after_pass_zero(&framing, outer, measured)
}

#[cfg(test)]
fn close_canonical_semantics_v2_for_test<'a>(
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    outcome: &gpu_db_wal::CanonicalOutcome,
    fragments: &[gpu_db_wal::CanonicalFragmentRef<'a>],
) -> Result<(), EngineError> {
    fill_canonical_semantics_v2_for_test(outer, outcome, fragments)?.close_codec_for_test()?;
    Ok(())
}

#[cfg(test)]
fn fail_retained_source_copy_at_for_test<T>(attempt: u64, operation: impl FnOnce() -> T) -> T {
    retained::fail_source_copy_at_for_test(attempt, operation)
}

#[cfg(test)]
fn validate_dependency_token_digest_for_test<'a>(
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    fragments: &[gpu_db_wal::CanonicalFragmentRef<'a>],
    raw: &[u8; 224],
) -> Result<(), EngineError> {
    let mut bodies = [&[][..]; AGGREGATE_MAX_CHUNKS + 1];
    for (target, fragment) in bodies.iter_mut().zip(fragments.iter()) {
        *target = fragment.body;
    }
    let framing = decode_aggregate_framing(outer.flags, &bodies[..fragments.len()])?;
    pass_zero::validate_dependency_token_digests_for_test(&framing, raw)
}

fn v2_error(message: &str) -> EngineError {
    EngineError::Durability(format!("typed INSERT aggregate semantics-v2: {message}"))
}
