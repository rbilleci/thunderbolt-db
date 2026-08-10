//! Codec-5 semantics-v2 S4/S7 ownership facade.
//!
//! The strict close entry point consumes canonical aggregate chunk/status framing after semantic-
//! version dispatch.  It proves the complete retained S1--S8 graph and its witness-free codec
//! closure without exposing retained state or creating a WAL, recovery, apply, or publication
//! authority. Test-only byte reencoding remains isolated behind `cfg(test)`.

#[cfg(test)]
#[path = "semantics_v2/goldens.rs"]
mod goldens;
#[path = "semantics_v2/pass_zero.rs"]
mod pass_zero;
#[path = "semantics_v2/retained.rs"]
mod retained;
#[path = "semantics_v2/sequence_terminal.rs"]
mod sequence_terminal;
#[path = "semantics_v2/writer.rs"]
mod writer;

pub(crate) use writer::{
    encode_live_typed_insert, encode_live_typed_insert_transaction, live_autocommit_request_digest,
    live_explicit_request_digest, live_explicit_request_digest_with_catalog,
    write001_identifier_digest, LiveFinalRowDigestSource, LiveTypedInsertCreatedIndex,
    LiveTypedInsertFinalWriter, LiveTypedInsertForeignIndexGeneration, LiveTypedInsertIdentity,
    LiveTypedInsertIndexGeneration, LiveTypedInsertMode, LiveTypedInsertSequenceRestart,
    LiveTypedInsertStatementView, LiveTypedInsertTableGeneration, LiveTypedInsertView,
};

pub(crate) use retained::{SemanticsV2ReplayArtifact, SemanticsV2ReplayMetadata};

use super::codec::decode_aggregate_framing;
use crate::typed_insert_aggregate::{
    AGGREGATE_FLAG_AUTOCOMMIT, AGGREGATE_FLAG_CATALOG, AGGREGATE_FLAG_EXPLICIT,
    AGGREGATE_FLAG_OPERATION_COMPOSITION, AGGREGATE_MAX_CHUNKS,
};
use crate::EngineError;

/// Run the canonical physical proof and the semantics-v2 allocation-free pass zero directly over
/// the original codec-5 chunk bodies.  The returned measurement has no decoded raw aggregate,
/// S2, image, catalog, lease, generation, replay, or publication capability.
///
/// This is deliberately crate-private and has no production caller while the full retained
/// S1--S8 owner remains inert.  It exists to make version dispatch explicit and to ensure v2 never
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
    measure_canonical_semantics_v2_framing(outer, outcome, &framing)
}

/// Complete the version-two physical/header/status matrix over one already-decoded immutable
/// framing.  Live close and fresh recovery both retain the one framing proof rather than
/// reparsing and rehashing the same canonical bytes before retained closure.
fn measure_canonical_semantics_v2_framing(
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    outcome: &gpu_db_wal::CanonicalOutcome,
    framing: &super::codec::DecodedAggregateFraming<'_>,
) -> Result<pass_zero::SemanticsV2PassZero, EngineError> {
    if framing.semantics_version() != crate::typed_insert_aggregate::AGGREGATE_SEMANTICS_V2 {
        return Err(v2_error("reader received a non-v2 aggregate"));
    }
    let measured = pass_zero::measure(framing, outer)?;
    let scalar = framing.header_scalars();
    let status = framing.status();
    if outer.stable_transaction_id != scalar.stable_transaction_id
        || outer.identity.database_id != status.database_id
        || outer.identity.timeline_id != status.timeline_id
        || outer.request_digest != status.request_digest
        || outer.isolation as u8 != status.isolation
        || outer.flags != framing.outer_flags()
        || outer.operation_count != framing.fragment_count() as u32
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
                && (scalar.flags & (AGGREGATE_FLAG_AUTOCOMMIT | AGGREGATE_FLAG_EXPLICIT) != 0)
                && outcome.affected_rows == scalar.original_inserted_row_count => {}
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

/// Strictly close one canonical semantics-v2 aggregate using the same retained production
/// machinery as replay. This is deliberately a validation seam: it returns no retained graph or
/// follow-on authority, so the caller cannot bypass the later retention, catalog, allocator,
/// sequence, generation, and publication gates.
pub(crate) fn close_canonical_semantics_v2<'a>(
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    outcome: &gpu_db_wal::CanonicalOutcome,
    fragments: &[gpu_db_wal::CanonicalFragmentRef<'a>],
) -> Result<(), EngineError> {
    let mut bodies = [&[][..]; AGGREGATE_MAX_CHUNKS + 1];
    for (target, fragment) in bodies.iter_mut().zip(fragments.iter()) {
        *target = fragment.body;
    }
    let framing = decode_aggregate_framing(outer.flags, &bodies[..fragments.len()])?;
    let measured = measure_canonical_semantics_v2_framing(outer, outcome, &framing)?;
    let catalog_composition = decode_catalog_composition(&framing, outer)?;
    retained::fill_after_pass_zero(&framing, outer, measured)?
        .close_codec(catalog_composition.as_ref())?;
    Ok(())
}

/// Decode and close the one production-reachable codec-5 aggregate into its move-only recovery
/// artifact.  This stays on the v2 retained decoder: callers receive neither aggregate bytes nor
/// a generic typed-record decoder that could be routed around the S1--S8 closure.
///
/// `Ok(None)` means the physical codec-5 framing selected another semantic version; that is
/// deliberately distinct from a malformed v2 aggregate, which is a durability failure.
pub(crate) fn decode_closed_semantics_v2_replay<'a>(
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    outcome: &gpu_db_wal::CanonicalOutcome,
    fragments: &[gpu_db_wal::CanonicalFragmentRef<'a>],
) -> Result<Option<SemanticsV2ReplayArtifact>, EngineError> {
    let mut bodies = [&[][..]; AGGREGATE_MAX_CHUNKS + 1];
    if fragments.len() > bodies.len() {
        return Err(v2_error("canonical fragment count exceeds codec-5 bounds"));
    }
    for (target, fragment) in bodies.iter_mut().zip(fragments.iter()) {
        *target = fragment.body;
    }
    let framing = decode_aggregate_framing(outer.flags, &bodies[..fragments.len()])?;
    if framing.semantics_version() != crate::typed_insert_aggregate::AGGREGATE_SEMANTICS_V2 {
        return Ok(None);
    }
    let measured = measure_canonical_semantics_v2_framing(outer, outcome, &framing)?;
    let catalog_composition = decode_catalog_composition(&framing, outer)?;
    let closed = retained::fill_after_pass_zero(&framing, outer, measured)?
        .close_codec(catalog_composition.as_ref())?;
    Ok(Some(
        closed
            .into_replay_artifact(catalog_composition.as_ref())?
            .with_catalog_composition(catalog_composition)?,
    ))
}

/// Decode the already-reserved S3 slot as the existing ordered transaction operation envelope.
/// This is not another recovery carrier: typed INSERT rows remain S2/S4/S7-owned, while any
/// pre-existing-row UPDATE/DELETE reaches the same common transaction applier as live commit.
fn decode_catalog_composition(
    framing: &super::codec::DecodedAggregateFraming<'_>,
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
) -> Result<Option<crate::wal_binary::BinaryTransactionRecord>, EngineError> {
    let flags = framing.header_scalars().flags;
    // Older codec-5 v2 records used CATALOG as the sole marker for their S3 operation. New
    // records use OPERATION_COMPOSITION for any S3 operation and reserve CATALOG for the outer
    // catalog boundary. Accept the former without widening the new writer contract.
    let has_operation_composition =
        flags & (AGGREGATE_FLAG_OPERATION_COMPOSITION | AGGREGATE_FLAG_CATALOG) != 0;
    if !has_operation_composition {
        return Ok(None);
    }
    let changes_catalog = flags & AGGREGATE_FLAG_CATALOG != 0;
    let bytes = usize::try_from(framing.sections()[2].payload_bytes)
        .map_err(|_| v2_error("S3 catalog operation length is not addressable"))?;
    let mut body = Vec::new();
    body.try_reserve_exact(bytes)
        .map_err(|_| v2_error("S3 catalog operation reservation failed"))?;
    body.resize(bytes, 0);
    framing.with_section_reader(2, |reader| reader.copy_exact(&mut body))?;
    let (expected_after_epoch, expected_after_digest) = if changes_catalog {
        (
            outer
                .catalog_before_epoch
                .checked_add(1)
                .ok_or_else(|| v2_error("S3 catalog epoch overflows"))?,
            crate::Engine::canonical_catalog_transition(
                outer.catalog_before_digest,
                gpu_db_wal::CanonicalFragmentKind::CatalogMutation,
                &body,
            ),
        )
    } else {
        (outer.catalog_before_epoch, outer.catalog_before_digest)
    };
    if outer.catalog_after_epoch != expected_after_epoch
        || outer.catalog_after_digest != expected_after_digest
    {
        return Err(v2_error(
            "S3 operation does not close the outer catalog boundary",
        ));
    }
    let payload = crate::Engine::decode_engine_operation(&body)?;
    let crate::wal_binary::BinaryWalRecord::Transaction(record) =
        crate::wal_binary::decode_binary_record(&payload)?
    else {
        return Err(v2_error("S3 is not an ordered catalog transaction"));
    };
    let catalog_shape = changes_catalog
        && !record.catalog_commands.is_empty()
        && record.catalog_commands.iter().all(|command| {
            crate::wal_binary::command_is_codec5_catalog_composition(&command.command)
        });
    let row_only_shape = !changes_catalog
        && record.catalog_commands.is_empty()
        && !record.mutations.is_empty()
        && record.catalog_output.is_none()
        && record.operation_order.is_empty()
        && record.statement_digests.is_empty()
        && record.sequence_input_oids.is_empty()
        && record.sequence_value_references.is_empty();
    if !(catalog_shape || row_only_shape)
        || record.allocator_high_water != 0
        || !record.table_resets.is_empty()
    {
        return Err(v2_error(
            "S3 operation transaction does not match its catalog or pre-existing-row authority",
        ));
    }
    Ok(Some(record))
}

#[cfg(test)]
pub(crate) fn decode_catalog_composition_for_test(
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    fragments: &[gpu_db_wal::CanonicalFragment],
) -> Result<Option<crate::wal_binary::BinaryTransactionRecord>, EngineError> {
    let refs = fragments
        .iter()
        .map(|fragment| gpu_db_wal::CanonicalFragmentRef {
            kind: fragment.kind,
            body: &fragment.body,
        })
        .collect::<Vec<_>>();
    let mut bodies = [&[][..]; AGGREGATE_MAX_CHUNKS + 1];
    for (target, fragment) in bodies.iter_mut().zip(refs.iter()) {
        *target = fragment.body;
    }
    let framing = decode_aggregate_framing(outer.flags, &bodies[..refs.len()])?;
    decode_catalog_composition(&framing, outer)
}

/// Test-only S8 measurement seam.  It decodes two complete canonical aggregate framings and
/// reuses the production S7 directory measure from the unretained reference before measuring
/// the candidate's S8 through the production artifact proof.  It does not define a second S8
/// grammar or construct an S8 payload in isolation.
#[cfg(test)]
fn measure_s8_against_reference_for_test<'reference, 'candidate>(
    reference_outer: &gpu_db_wal::CanonicalPreApplyHeader,
    reference_fragments: &[gpu_db_wal::CanonicalFragmentRef<'reference>],
    candidate_outer: &gpu_db_wal::CanonicalPreApplyHeader,
    candidate_fragments: &[gpu_db_wal::CanonicalFragmentRef<'candidate>],
) -> Result<pass_zero::S8PassZeroMeasure, EngineError> {
    let mut reference_bodies = [&[][..]; AGGREGATE_MAX_CHUNKS + 1];
    for (target, fragment) in reference_bodies.iter_mut().zip(reference_fragments.iter()) {
        *target = fragment.body;
    }
    let reference = decode_aggregate_framing(
        reference_outer.flags,
        &reference_bodies[..reference_fragments.len()],
    )?;
    let mut candidate_bodies = [&[][..]; AGGREGATE_MAX_CHUNKS + 1];
    for (target, fragment) in candidate_bodies.iter_mut().zip(candidate_fragments.iter()) {
        *target = fragment.body;
    }
    let candidate = decode_aggregate_framing(
        candidate_outer.flags,
        &candidate_bodies[..candidate_fragments.len()],
    )?;
    pass_zero::measure_s8_against_reference_for_test(
        &reference,
        reference_outer,
        &candidate,
        candidate_outer,
    )
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
) -> Result<retained::AggregateReplayTxn<retained::CodecQuarantined>, EngineError> {
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
    close_canonical_semantics_v2(outer, outcome, fragments)
}

/// Test-only handoff of an actual strict-filled, codec-closed owner to the retained Q2 golden
/// facade.  The returned typestate has no public graph accessor or builder method; only the
/// retained facade can consume it with a pinned catalog and allocator lease proof.
#[cfg(test)]
fn codec_closed_canonical_semantics_v2_for_test<'a>(
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    outcome: &gpu_db_wal::CanonicalOutcome,
    fragments: &[gpu_db_wal::CanonicalFragmentRef<'a>],
) -> Result<retained::Q2CodecClosedSemanticsV2, EngineError> {
    fill_canonical_semantics_v2_for_test(outer, outcome, fragments)?.close_codec_for_test()
}

/// Test-only catalog-guard continuation.  It consumes the actual codec-closed owner through the
/// one guard-validation leaf and returns no retained state or generation-pending capability.
#[cfg(test)]
fn validate_canonical_semantics_v2_guards_for_test<'a>(
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    outcome: &gpu_db_wal::CanonicalOutcome,
    fragments: &[gpu_db_wal::CanonicalFragmentRef<'a>],
    catalog: &retained::SemanticsV2CatalogWitness<'_>,
) -> Result<(), EngineError> {
    fill_canonical_semantics_v2_for_test(outer, outcome, fragments)?
        .validate_guards_after_codec_for_test(catalog)
}

#[cfg(test)]
fn fail_retained_source_copy_at_for_test<T>(attempt: u64, operation: impl FnOnce() -> T) -> T {
    retained::fail_source_copy_at_for_test(attempt, operation)
}

#[cfg(test)]
fn observe_retained_source_copy_attempts_for_test<T>(operation: impl FnOnce() -> T) -> (T, u64) {
    retained::observe_source_copy_attempts_for_test(operation)
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
