//! Sealed retention-authority proof between local codec closure and catalog validation.
//!
//! This module authenticates only the immutable `RetentionIntent` carried by the one durable
//! claim/status authority.  It deliberately does not consult a catalog or allocator, validate
//! durable sequences, reserve capacity, compile or launch GPU work, or expose a WAL, recovery,
//! apply, result, or publication successor.

use super::{graph::ReservedSemanticsV2Graph, SemanticsV2BoundIdentity};
use crate::EngineError;

type Digest = gpu_db_wal::CanonicalDigest;

/// Borrowed once from the sole authenticated Engine claim/status index.  There is no production
/// constructor in this inert checkpoint: the future owner must bind the existing authority rather
/// than materialising a parallel claim cache or a decoded-body substitute.
pub(super) struct AuthenticatedClaimStatusIndex<'a> {
    lineage: ClaimStatusIndexLineage,
    claims: &'a [AuthenticatedTransactionClaimStatus<'a>],
}

/// The durable identity and recovery frontier covered by one authenticated status index.
#[derive(Clone, Copy)]
pub(super) struct ClaimStatusIndexLineage {
    database_id: [u8; 16],
    timeline_id: [u8; 16],
    recovery_prefix: u64,
}

/// One immutable durable claim head.  The terminal branch is intentionally opaque to this phase;
/// only the post-replay comparator may inspect terminal outcome fields in a later checkpoint.
pub(super) struct AuthenticatedTransactionClaimStatus<'a> {
    database_id: [u8; 16],
    timeline_id: [u8; 16],
    stable_transaction_id: u64,
    request_digest: Digest,
    statement_digests: &'a [Digest],
    intent: RetentionIntent<'a>,
    head: ClaimHead<'a>,
    completion: ClaimCompletion,
}

/// The only retention-intent source for a live aggregate.  The raw bitset is borrowed and stays
/// sealed in the validated authority; it is not reconstructed from S6/S7/S8 facts at this phase.
pub(super) struct RetentionIntent<'a> {
    eligible_statement_bits: &'a [u8],
    candidate_deadline: u64,
}

/// A pending claim is valid input, as is a terminal claim whose immutable intent bytes survived
/// the transition unchanged.  `Terminal` carries no inspectable terminal verdict here.
#[derive(Clone, Copy)]
pub(super) enum ClaimHead<'a> {
    Pending,
    Terminal(SealedTerminalClaim<'a>),
}

/// Opaque terminal-head marker.  The predecessor intent proves an update did not mutate the
/// immutable pre-effect choice.  The complete terminal retention facts stay sealed in this
/// proof for the post-GPU comparator; this phase may neither read them nor use them to choose
/// execution or a verdict.
#[derive(Clone, Copy)]
pub(super) struct SealedTerminalClaim<'a> {
    predecessor_eligible_statement_bits: &'a [u8],
    predecessor_candidate_deadline: u64,
    terminal: SealedTerminalRetentionFacts<'a>,
    _private: (),
}

/// The terminal claim's final-retention evidence.  The later comparator, not this authority
/// checkpoint, compares these sealed bytes and scalars with the independently observed S6/S7,
/// S8, aggregate, and STATUS2 closure.  Keeping both S6 and S7 bitsets prevents a terminal
/// record from collapsing their independently checked equality into one inferred value.
#[derive(Clone, Copy)]
struct SealedTerminalRetentionFacts<'a> {
    s6_retained_statement_bits: &'a [u8],
    s7_retained_statement_bits: &'a [u8],
    s8_artifact_statement_ordinals: &'a [u32],
    s8_artifact_count: u32,
    aggregate_retained_response: bool,
    status_artifact_count: u32,
    status_deadline: u64,
}

/// A claim must be complete and durable before this phase can advance.  An incomplete durable
/// prefix is not an empty intent and never permits allocator, sequence, or parent work.
#[derive(Clone, Copy)]
pub(super) enum ClaimCompletion {
    Incomplete,
    Complete(CompleteClaimDurability),
}

/// Durable placement and sequencing evidence for a complete claim.  Child positions may be
/// absent when no corresponding child exists; whenever present the claim must precede them.
#[derive(Clone, Copy)]
pub(super) struct CompleteClaimDurability {
    claim_sequence: u64,
    recovery_prefix: u64,
    location: AuthenticatedClaimLocation,
    allocator_child_sequence: Option<u64>,
    sequence_child_sequence: Option<u64>,
    parent_effect_sequence: Option<u64>,
}

/// A complete claim is authenticated either in the pinned checkpoint status section or in the
/// canonical WAL suffix retained beyond that checkpoint.  Both forms remain in the same lineage.
#[derive(Clone, Copy)]
pub(super) enum AuthenticatedClaimLocation {
    CheckpointStatus {
        checkpoint_sequence: u64,
    },
    CanonicalWalSuffix {
        first_sequence: u64,
        last_sequence: u64,
    },
}

/// Historical records cannot manufacture an empty claim.  A separate authenticated translator
/// may instead hand over this sealed proof when its literal (codec, opcode) allowlist made
/// response retention impossible.
pub(super) struct HistoricalNoRetentionProof<'a> {
    database_id: [u8; 16],
    timeline_id: [u8; 16],
    stable_transaction_id: u64,
    request_digest: Digest,
    statement_digests: &'a [Digest],
    recovery_prefix: u64,
    source_is_allowlisted: bool,
    no_retention_is_proven: bool,
}

/// The two and only two authority inputs.  The claim arm borrows the single Engine index; the
/// historical arm is a sealed no-retention proof and never substitutes an empty status record.
pub(in crate::typed_insert_aggregate::semantics_v2) struct RetentionAuthorityInput<'a>(
    RetentionAuthoritySource<'a>,
);

enum RetentionAuthoritySource<'a> {
    Claim(&'a AuthenticatedClaimStatusIndex<'a>),
    HistoricalNoRetention(HistoricalNoRetentionProof<'a>),
}

/// Opaque validated sum carried into later catalog, sequence, and replay phases.  No accessor is
/// exposed before the final comparator consumes it, so neither compiler nor execution can use a
/// retention decision as an input or verdict selector.
pub(super) struct ValidatedRetentionAuthority<'a>(ValidatedRetentionAuthorityState<'a>);

enum ValidatedRetentionAuthorityState<'a> {
    Claim(ValidatedRetentionIntent<'a>),
    HistoricalNoRetention(ValidatedHistoricalNoRetention<'a>),
}

struct ValidatedRetentionIntent<'a> {
    claim: &'a AuthenticatedTransactionClaimStatus<'a>,
}

struct ValidatedHistoricalNoRetention<'a> {
    proof: HistoricalNoRetentionProof<'a>,
}

/// Validate exactly one authority source before catalog or allocator work is reachable.
pub(super) fn validate<'a>(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    input: RetentionAuthorityInput<'a>,
) -> Result<ValidatedRetentionAuthority<'a>, EngineError> {
    match input.0 {
        RetentionAuthoritySource::Claim(index) => {
            let claim = lookup_claim(index, identity)?;
            validate_live_claim(identity, graph, index.lineage, claim)?;
            Ok(ValidatedRetentionAuthority(
                ValidatedRetentionAuthorityState::Claim(ValidatedRetentionIntent { claim }),
            ))
        }
        RetentionAuthoritySource::HistoricalNoRetention(proof) => {
            validate_historical_no_retention(identity, graph, &proof)?;
            Ok(ValidatedRetentionAuthority(
                ValidatedRetentionAuthorityState::HistoricalNoRetention(
                    ValidatedHistoricalNoRetention { proof },
                ),
            ))
        }
    }
}

fn lookup_claim<'a>(
    index: &'a AuthenticatedClaimStatusIndex<'a>,
    identity: SemanticsV2BoundIdentity,
) -> Result<&'a AuthenticatedTransactionClaimStatus<'a>, EngineError> {
    let mut matched = None;
    for claim in index.claims {
        if claim.database_id == identity.database_id
            && claim.timeline_id == identity.timeline_id
            && claim.stable_transaction_id == identity.stable_transaction_id
            && matched.replace(claim).is_some()
        {
            return Err(authority_error(
                "claim/status index has duplicate transaction claim heads",
            ));
        }
    }
    matched.ok_or_else(|| authority_error("claim/status index has no matching transaction claim"))
}

fn validate_live_claim(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    lineage: ClaimStatusIndexLineage,
    claim: &AuthenticatedTransactionClaimStatus<'_>,
) -> Result<(), EngineError> {
    if lineage.database_id != identity.database_id
        || lineage.timeline_id != identity.timeline_id
        || lineage.recovery_prefix == 0
        || claim.database_id != identity.database_id
        || claim.timeline_id != identity.timeline_id
        || claim.stable_transaction_id != identity.stable_transaction_id
        || claim.request_digest != identity.request_digest
    {
        return Err(authority_error(
            "claim/status lineage or request identity differs from the aggregate",
        ));
    }
    validate_statement_digest_chain(graph, claim.statement_digests)?;
    if let ClaimHead::Terminal(terminal) = claim.head {
        if terminal.predecessor_eligible_statement_bits != claim.intent.eligible_statement_bits
            || terminal.predecessor_candidate_deadline != claim.intent.candidate_deadline
        {
            return Err(authority_error(
                "terminal claim mutated immutable retention intent bytes",
            ));
        }
    }
    let ClaimCompletion::Complete(durability) = claim.completion else {
        return Err(authority_error(
            "claim/status record is incomplete in the durable recovery prefix",
        ));
    };
    validate_claim_durability(identity, lineage, durability)?;
    validate_retention_intent(graph, &claim.intent)
}

fn validate_historical_no_retention(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    proof: &HistoricalNoRetentionProof<'_>,
) -> Result<(), EngineError> {
    if proof.database_id != identity.database_id
        || proof.timeline_id != identity.timeline_id
        || proof.stable_transaction_id != identity.stable_transaction_id
        || proof.request_digest != identity.request_digest
        || proof.recovery_prefix == 0
        || proof.recovery_prefix > identity.commit_sequence
        || !proof.source_is_allowlisted
        || !proof.no_retention_is_proven
    {
        return Err(authority_error(
            "historical no-retention proof is not authenticated for this aggregate",
        ));
    }
    validate_statement_digest_chain(graph, proof.statement_digests)
}

fn validate_statement_digest_chain(
    graph: &ReservedSemanticsV2Graph,
    expected: &[Digest],
) -> Result<(), EngineError> {
    if graph.statements.is_empty()
        || graph.resolutions.len() != graph.statements.len()
        || expected.len() != graph.statements.len()
    {
        return Err(authority_error(
            "claim statement-digest chain has the wrong cardinality",
        ));
    }
    for (ordinal, (statement, digest)) in graph.statements.iter().zip(expected).enumerate() {
        let resolution = graph
            .resolutions
            .get(ordinal)
            .ok_or_else(|| authority_error("claim statement lacks an S7 resolution"))?;
        if statement.statement_ordinal != ordinal as u32
            || resolution.statement_ordinal != ordinal as u32
            || statement.typed_statement_digest != *digest
            || resolution.typed_statement_digest != *digest
        {
            return Err(authority_error(
                "claim statement-digest chain differs from the decoded aggregate",
            ));
        }
    }
    Ok(())
}

fn validate_claim_durability(
    identity: SemanticsV2BoundIdentity,
    lineage: ClaimStatusIndexLineage,
    durability: CompleteClaimDurability,
) -> Result<(), EngineError> {
    if durability.claim_sequence == 0
        || durability.recovery_prefix != lineage.recovery_prefix
        || durability.recovery_prefix > identity.commit_sequence
        || durability.claim_sequence > durability.recovery_prefix
        || durability.claim_sequence >= identity.commit_sequence
    {
        return Err(authority_error(
            "claim is incomplete or outside the durable recovery prefix",
        ));
    }
    match durability.location {
        AuthenticatedClaimLocation::CheckpointStatus {
            checkpoint_sequence,
        } if checkpoint_sequence != 0
            && durability.claim_sequence <= checkpoint_sequence
            && checkpoint_sequence <= durability.recovery_prefix => {}
        AuthenticatedClaimLocation::CanonicalWalSuffix {
            first_sequence,
            last_sequence,
        } if first_sequence != 0
            && first_sequence <= durability.claim_sequence
            && durability.claim_sequence <= last_sequence
            && last_sequence <= durability.recovery_prefix => {}
        _ => {
            return Err(authority_error(
                "claim is absent from its authenticated checkpoint status or WAL suffix",
            ));
        }
    }
    for child in [
        durability.allocator_child_sequence,
        durability.sequence_child_sequence,
        durability.parent_effect_sequence,
    ]
    .into_iter()
    .flatten()
    {
        if child == 0 || child > durability.recovery_prefix || durability.claim_sequence >= child {
            return Err(authority_error(
                "claim does not precede a durable allocator, sequence, or parent effect",
            ));
        }
    }
    Ok(())
}

fn validate_retention_intent(
    graph: &ReservedSemanticsV2Graph,
    intent: &RetentionIntent<'_>,
) -> Result<(), EngineError> {
    let expected_bytes = graph
        .statements
        .len()
        .checked_add(7)
        .ok_or_else(|| authority_error("statement-count bitset length overflows"))?
        / 8;
    if intent.eligible_statement_bits.len() != expected_bytes {
        return Err(authority_error(
            "retention intent bitset is not canonical for the statement count",
        ));
    }
    let mut eligible_count = 0_usize;
    for (ordinal, resolution) in graph.resolutions.iter().enumerate() {
        if bit_is_set(intent.eligible_statement_bits, ordinal) {
            if resolution.projection_count == 0 || resolution.flags & 1 == 0 {
                return Err(authority_error(
                    "retention intent selects a statement without RETURNING",
                ));
            }
            eligible_count = eligible_count
                .checked_add(1)
                .ok_or_else(|| authority_error("retention intent count overflows"))?;
        }
    }
    let used_bits = graph.statements.len() % 8;
    if used_bits != 0 {
        let unused_mask = !((1_u8 << used_bits) - 1);
        if intent
            .eligible_statement_bits
            .last()
            .copied()
            .unwrap_or_default()
            & unused_mask
            != 0
        {
            return Err(authority_error(
                "retention intent has noncanonical high statement bits",
            ));
        }
    }
    let deadline_is_valid = (1..u64::MAX).contains(&intent.candidate_deadline);
    if (eligible_count == 0 && intent.candidate_deadline != 0)
        || (eligible_count != 0 && !deadline_is_valid)
    {
        return Err(authority_error(
            "retention intent deadline does not match its eligible statement set",
        ));
    }
    Ok(())
}

fn bit_is_set(bits: &[u8], ordinal: usize) -> bool {
    bits[ordinal / 8] & (1_u8 << (ordinal % 8)) != 0
}

fn authority_error(message: &str) -> EngineError {
    EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v2 retention authority: {message}"
    ))
}

/// Test-only adapter for the checked-in historical empty-S8 vectors.  It owns the only
/// allowlisted historical-no-retention proof long enough to drive the real catalog/allocator
/// successor, while keeping both the proof and resulting typestate from escaping the callback.
#[cfg(test)]
pub(super) fn with_historical_no_retention_pending_for_test<T>(
    pending: super::AggregateReplayTxn<super::RetentionAuthorityPending>,
    operation: impl FnOnce(super::AggregateReplayTxn<super::CatalogAllocatorPending<'_>>) -> T,
) -> T {
    let identity = pending.graph.identity;
    assert!(
        matches!(
            &pending.graph.graph.response,
            super::graph::RetainedResponseEnvelope::Empty(_)
        ),
        "historical no-retention test adapter accepts only canonical empty S8"
    );
    let statement_digests: Vec<_> = pending
        .graph
        .graph
        .statements
        .iter()
        .map(|statement| statement.typed_statement_digest)
        .collect();
    let catalog_pending = pending
        .validate_retention_authority(RetentionAuthorityInput(
            RetentionAuthoritySource::HistoricalNoRetention(HistoricalNoRetentionProof {
                database_id: identity.database_id,
                timeline_id: identity.timeline_id,
                stable_transaction_id: identity.stable_transaction_id,
                request_digest: identity.request_digest,
                statement_digests: &statement_digests,
                recovery_prefix: identity.commit_sequence,
                source_is_allowlisted: true,
                no_retention_is_proven: true,
            }),
        ))
        .expect("canonical empty-S8 fixture has a sealed historical no-retention proof");
    operation(catalog_pending)
}

/// Test-only live-claim adapter for checked-in vectors.  Like the production boundary it borrows
/// one authenticated index and carries the resulting opaque authority through the callback; the
/// local claim/index/bitset backing cannot escape it.
#[cfg(test)]
pub(super) fn with_live_claim_pending_for_test<T>(
    pending: super::AggregateReplayTxn<super::RetentionAuthorityPending>,
    operation: impl FnOnce(super::AggregateReplayTxn<super::CatalogAllocatorPending<'_>>) -> T,
) -> T {
    let identity = pending.graph.identity;
    let statement_digests: Vec<_> = pending
        .graph
        .graph
        .statements
        .iter()
        .map(|statement| statement.typed_statement_digest)
        .collect();
    let eligible_statement_bits = vec![0; statement_digests.len().div_ceil(8)];
    let claim = AuthenticatedTransactionClaimStatus {
        database_id: identity.database_id,
        timeline_id: identity.timeline_id,
        stable_transaction_id: identity.stable_transaction_id,
        request_digest: identity.request_digest,
        statement_digests: &statement_digests,
        intent: RetentionIntent {
            eligible_statement_bits: &eligible_statement_bits,
            candidate_deadline: 0,
        },
        head: ClaimHead::Pending,
        completion: ClaimCompletion::Complete(CompleteClaimDurability {
            claim_sequence: 1,
            recovery_prefix: identity.commit_sequence,
            location: AuthenticatedClaimLocation::CheckpointStatus {
                checkpoint_sequence: 1,
            },
            allocator_child_sequence: Some(2),
            sequence_child_sequence: None,
            parent_effect_sequence: Some(identity.commit_sequence),
        }),
    };
    let claims = [claim];
    let index = AuthenticatedClaimStatusIndex {
        lineage: ClaimStatusIndexLineage {
            database_id: identity.database_id,
            timeline_id: identity.timeline_id,
            recovery_prefix: identity.commit_sequence,
        },
        claims: &claims,
    };
    let catalog_pending = pending
        .validate_retention_authority(RetentionAuthorityInput(RetentionAuthoritySource::Claim(
            &index,
        )))
        .expect("checked-in fixture has an exact authenticated live retention claim");
    operation(catalog_pending)
}

#[cfg(test)]
mod tests {
    use super::super::{
        AggregateReplayTxn, PrivateSeal, RetainedSemanticsV2Graph, RetentionAuthorityPending,
    };
    use super::*;
    use crate::typed_insert_aggregate::semantics_v2::retained::graph::{
        empty_response_for_test, RetainedStatement, RetainedStatementResolution,
    };

    const DATABASE: [u8; 16] = [1; 16];
    const TIMELINE: [u8; 16] = [2; 16];
    const REQUEST: Digest = [3; 32];
    const STATEMENT: Digest = [4; 32];

    #[test]
    fn exact_live_claim_consumes_the_pending_owner_before_catalog_validation() {
        let identity = identity();
        let digests = [STATEMENT];
        let bits = [0_u8];
        let claim = valid_claim(&digests, &bits);
        let claim_index = index(&claim);

        let validated = pending(identity)
            .validate_retention_authority(RetentionAuthorityInput(RetentionAuthoritySource::Claim(
                &claim_index,
            )))
            .expect("the exact durable claim is the sole valid live retention authority");
        match validated.phase.authority.0 {
            ValidatedRetentionAuthorityState::Claim(_) => {}
            ValidatedRetentionAuthorityState::HistoricalNoRetention(_) => {
                panic!("live claim cannot become historical authority")
            }
        }
    }

    #[test]
    fn claim_authority_rejects_rehashed_same_id_request_and_chain_mismatches() {
        let identity = identity();
        let graph = graph();
        let digests = [STATEMENT];
        let bits = [0_u8];
        let mut claim = valid_claim(&digests, &bits);
        claim.request_digest[0] ^= 1;
        let claim_index = index(&claim);
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&claim_index)),
        ));
        assert!(error.to_string().contains("request identity"));

        let wrong_digests = [[5; 32]];
        let claim = valid_claim(&wrong_digests, &bits);
        let claim_index = index(&claim);
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&claim_index)),
        ));
        assert!(error.to_string().contains("statement-digest chain"));
    }

    #[test]
    fn claim_authority_rejects_duplicate_ineligible_or_incomplete_intent_sources() {
        let identity = identity();
        let graph = graph();
        let digests = [STATEMENT];
        let bits = [0_u8];
        let claim = valid_claim(&digests, &bits);
        let duplicate = valid_claim(&digests, &bits);
        let duplicate_index = AuthenticatedClaimStatusIndex {
            lineage: lineage(),
            claims: &[claim, duplicate],
        };
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&duplicate_index)),
        ));
        assert!(error.to_string().contains("duplicate"));

        let ineligible_bits = [1_u8];
        let claim = valid_claim(&digests, &ineligible_bits);
        let claim_index = index(&claim);
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&claim_index)),
        ));
        assert!(error.to_string().contains("without RETURNING"));

        let high_bits = [0x80_u8];
        let claim = valid_claim(&digests, &high_bits);
        let claim_index = index(&claim);
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&claim_index)),
        ));
        assert!(error.to_string().contains("high statement bits"));

        let mut claim = valid_claim(&digests, &bits);
        claim.intent.candidate_deadline = 99;
        let claim_index = index(&claim);
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&claim_index)),
        ));
        assert!(error.to_string().contains("deadline"));

        let mut claim = valid_claim(&digests, &bits);
        claim.completion = ClaimCompletion::Incomplete;
        let claim_index = index(&claim);
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&claim_index)),
        ));
        assert!(error.to_string().contains("incomplete"));

        let mut claim = valid_claim(&digests, &bits);
        let ClaimCompletion::Complete(durability) = &mut claim.completion else {
            panic!("valid test claim must be complete")
        };
        durability.sequence_child_sequence = Some(1);
        let claim_index = index(&claim);
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&claim_index)),
        ));
        assert!(error.to_string().contains("does not precede"));
    }

    #[test]
    fn claim_authority_rejects_absent_cross_lineage_and_invalid_durable_locations() {
        let identity = identity();
        let graph = graph();
        let digests = [STATEMENT];
        let bits = [0_u8];

        let empty_index = AuthenticatedClaimStatusIndex {
            lineage: lineage(),
            claims: &[],
        };
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&empty_index)),
        ));
        assert!(error.to_string().contains("no matching"));

        let claim = valid_claim(&digests, &bits);
        let cross_lineage = AuthenticatedClaimStatusIndex {
            lineage: ClaimStatusIndexLineage {
                timeline_id: [99; 16],
                ..lineage()
            },
            claims: std::slice::from_ref(&claim),
        };
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&cross_lineage)),
        ));
        assert!(error.to_string().contains("lineage"));

        let mut claim = valid_claim(&digests, &bits);
        let ClaimCompletion::Complete(durability) = &mut claim.completion else {
            panic!("valid test claim must be complete")
        };
        durability.claim_sequence = 0;
        let claim_index = index(&claim);
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&claim_index)),
        ));
        assert!(error
            .to_string()
            .contains("outside the durable recovery prefix"));

        let mut claim = valid_claim(&digests, &bits);
        let ClaimCompletion::Complete(durability) = &mut claim.completion else {
            panic!("valid test claim must be complete")
        };
        durability.claim_sequence = identity.commit_sequence;
        durability.location = AuthenticatedClaimLocation::CanonicalWalSuffix {
            first_sequence: identity.commit_sequence,
            last_sequence: identity.commit_sequence,
        };
        let claim_index = index(&claim);
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&claim_index)),
        ));
        assert!(error
            .to_string()
            .contains("outside the durable recovery prefix"));

        let mut claim = valid_claim(&digests, &bits);
        let ClaimCompletion::Complete(durability) = &mut claim.completion else {
            panic!("valid test claim must be complete")
        };
        durability.location = AuthenticatedClaimLocation::CanonicalWalSuffix {
            first_sequence: 2,
            last_sequence: 1,
        };
        let claim_index = index(&claim);
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&claim_index)),
        ));
        assert!(error
            .to_string()
            .contains("checkpoint status or WAL suffix"));
    }

    #[test]
    fn nonempty_intent_requires_returning_and_a_finite_deadline() {
        let identity = identity();
        let digests = [STATEMENT];
        let bits = [1_u8];
        let mut returning_graph = graph();
        returning_graph.resolutions[0].flags = 1;
        returning_graph.resolutions[0].projection_count = 1;
        let mut claim = valid_claim(&digests, &bits);
        claim.intent.candidate_deadline = 1;
        let claim_index = index(&claim);
        assert!(matches!(
            validate(
                identity,
                &returning_graph,
                RetentionAuthorityInput(RetentionAuthoritySource::Claim(&claim_index)),
            ),
            Ok(ValidatedRetentionAuthority(
                ValidatedRetentionAuthorityState::Claim(_)
            ))
        ));

        let mut claim = valid_claim(&digests, &bits);
        claim.intent.candidate_deadline = u64::MAX;
        let claim_index = index(&claim);
        let error = rejected(validate(
            identity,
            &returning_graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&claim_index)),
        ));
        assert!(error.to_string().contains("deadline"));
    }

    #[test]
    fn terminal_claim_in_a_pinned_canonical_wal_suffix_preserves_its_intent() {
        let identity = identity();
        let graph = graph();
        let digests = [STATEMENT];
        let bits = [0_u8];
        let mut claim = valid_claim(&digests, &bits);
        claim.head = ClaimHead::Terminal(SealedTerminalClaim {
            predecessor_eligible_statement_bits: &bits,
            predecessor_candidate_deadline: 0,
            terminal: terminal_retention_facts(),
            _private: (),
        });
        claim.completion = ClaimCompletion::Complete(CompleteClaimDurability {
            claim_sequence: 1,
            recovery_prefix: 3,
            location: AuthenticatedClaimLocation::CanonicalWalSuffix {
                first_sequence: 1,
                last_sequence: 2,
            },
            allocator_child_sequence: Some(2),
            sequence_child_sequence: None,
            parent_effect_sequence: Some(3),
        });
        let claim_index = index(&claim);
        let validated = validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&claim_index)),
        )
        .expect("the terminal claim preserves its opaque final-retention evidence");
        let ValidatedRetentionAuthority(ValidatedRetentionAuthorityState::Claim(intent)) =
            validated
        else {
            panic!("terminal claim cannot become historical authority")
        };
        let ClaimHead::Terminal(terminal) = intent.claim.head else {
            panic!("terminal test claim must remain terminal")
        };
        let expected = terminal_retention_facts();
        assert_eq!(
            terminal.terminal.s6_retained_statement_bits,
            expected.s6_retained_statement_bits
        );
        assert_eq!(
            terminal.terminal.s7_retained_statement_bits,
            expected.s7_retained_statement_bits
        );
        assert_eq!(
            terminal.terminal.s8_artifact_statement_ordinals,
            expected.s8_artifact_statement_ordinals
        );
        assert_eq!(
            terminal.terminal.s8_artifact_count,
            expected.s8_artifact_count
        );
        assert_eq!(
            terminal.terminal.aggregate_retained_response,
            expected.aggregate_retained_response
        );
        assert_eq!(
            terminal.terminal.status_artifact_count,
            expected.status_artifact_count
        );
        assert_eq!(terminal.terminal.status_deadline, expected.status_deadline);
    }

    #[test]
    fn terminal_claim_cannot_mutate_the_authenticated_pending_intent() {
        let identity = identity();
        let graph = graph();
        let digests = [STATEMENT];
        let bits = [0_u8];
        let predecessor_bits = [1_u8];
        let mut claim = valid_claim(&digests, &bits);
        claim.head = ClaimHead::Terminal(SealedTerminalClaim {
            predecessor_eligible_statement_bits: &predecessor_bits,
            predecessor_candidate_deadline: 0,
            terminal: terminal_retention_facts(),
            _private: (),
        });
        let claim_index = index(&claim);
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::Claim(&claim_index)),
        ));
        assert!(error
            .to_string()
            .contains("mutated immutable retention intent"));
    }

    #[test]
    fn authenticated_historical_no_retention_is_the_only_nonclaim_source() {
        let identity = identity();
        let graph = graph();
        let digests = [STATEMENT];
        let proof = HistoricalNoRetentionProof {
            database_id: DATABASE,
            timeline_id: TIMELINE,
            stable_transaction_id: 7,
            request_digest: REQUEST,
            statement_digests: &digests,
            recovery_prefix: 3,
            source_is_allowlisted: true,
            no_retention_is_proven: true,
        };
        assert!(matches!(
            validate(
                identity,
                &graph,
                RetentionAuthorityInput(RetentionAuthoritySource::HistoricalNoRetention(proof)),
            ),
            Ok(ValidatedRetentionAuthority(
                ValidatedRetentionAuthorityState::HistoricalNoRetention(_)
            ))
        ));
    }

    #[test]
    fn historical_no_retention_rejects_unpinned_or_mismatched_proofs() {
        let identity = identity();
        let graph = graph();
        let digests = [STATEMENT];
        let mut proof = HistoricalNoRetentionProof {
            database_id: DATABASE,
            timeline_id: TIMELINE,
            stable_transaction_id: 7,
            request_digest: REQUEST,
            statement_digests: &digests,
            recovery_prefix: 3,
            source_is_allowlisted: true,
            no_retention_is_proven: true,
        };
        proof.source_is_allowlisted = false;
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::HistoricalNoRetention(proof)),
        ));
        assert!(error.to_string().contains("historical no-retention"));

        let mut proof = HistoricalNoRetentionProof {
            database_id: DATABASE,
            timeline_id: TIMELINE,
            stable_transaction_id: 7,
            request_digest: REQUEST,
            statement_digests: &digests,
            recovery_prefix: 3,
            source_is_allowlisted: true,
            no_retention_is_proven: true,
        };
        proof.recovery_prefix = identity.commit_sequence.saturating_add(1);
        let error = rejected(validate(
            identity,
            &graph,
            RetentionAuthorityInput(RetentionAuthoritySource::HistoricalNoRetention(proof)),
        ));
        assert!(error.to_string().contains("historical no-retention"));
    }

    fn rejected(result: Result<ValidatedRetentionAuthority<'_>, EngineError>) -> EngineError {
        match result {
            Ok(_) => panic!("retention authority unexpectedly validated"),
            Err(error) => error,
        }
    }

    fn identity() -> SemanticsV2BoundIdentity {
        SemanticsV2BoundIdentity {
            database_id: DATABASE,
            cluster_id: [5; 16],
            timeline_id: TIMELINE,
            format_epoch: 1,
            leader_epoch: 1,
            catalog_epoch: 1,
            catalog_digest: [6; 32],
            stable_transaction_id: 7,
            request_digest: REQUEST,
            autocommit: true,
            commit_sequence: 3,
            initial_database_root: [7; 32],
        }
    }

    fn pending(
        identity: SemanticsV2BoundIdentity,
    ) -> AggregateReplayTxn<RetentionAuthorityPending> {
        AggregateReplayTxn {
            graph: RetainedSemanticsV2Graph {
                identity,
                graph: graph(),
            },
            phase: RetentionAuthorityPending(PrivateSeal),
        }
    }

    fn graph() -> ReservedSemanticsV2Graph {
        ReservedSemanticsV2Graph {
            header: crate::typed_insert_aggregate::semantics_v2::pass_zero::SemanticsV2S7HeaderIdentity {
                total_bytes: 1,
                root_descriptor_version: 1,
                catalog_before_epoch: 1,
                catalog_after_epoch: 1,
                catalog_before_digest: [6; 32],
                catalog_after_digest: [6; 32],
                initial_database_root: [7; 32],
                final_database_root: [1; 32],
                initial_overlay_root: [1; 32],
                final_overlay_root: [1; 32],
                root_descriptor: [1; 32],
                payload_digest: [1; 32],
            },
            statements: vec![RetainedStatement {
                statement_ordinal: 0,
                family_ordinal: 0,
                input_row_count: 1,
                request_digest: REQUEST,
                typed_statement_digest: STATEMENT,
                overlay_before: [0; 32],
                overlay_after: [0; 32],
                record_bytes: 0,
                record_digest: [0; 32],
            }],
            records: Vec::new(),
            dispositions: Vec::new(),
            sequence_effects: Vec::new(),
            outcomes: Vec::new(),
            tables: Vec::new(),
            table_dispositions: Vec::new(),
            resolutions: vec![RetainedStatementResolution {
                statement_ordinal: 0,
                record_ref: 0,
                outcome_ref: 0,
                table_ref: 0,
                flags: 0,
                s4_start: 0,
                s4_count: 0,
                s5_start: 0,
                s5_count: 0,
                dependency_use_start: 0,
                dependency_use_count: 0,
                projection_start: 0,
                projection_count: 0,
                input_row_count: 1,
                surviving_row_count: 0,
                affected_row_count: 0,
                dependency_validation_floor: 0,
                record_bytes: 0,
                terminal_dependency_ref: u32::MAX,
                terminal_row_ordinal: u32::MAX,
                terminal_source_ordinal: u32::MAX,
                request_digest: REQUEST,
                typed_statement_digest: STATEMENT,
                record_digest: [0; 32],
                returning_digest: [0; 32],
                overlay_before: [0; 32],
                overlay_after: [0; 32],
                outcome_digest: [0; 32],
            }],
            dependencies: Vec::new(),
            dependency_uses: Vec::new(),
            indexes: Vec::new(),
            index_key_columns: Vec::new(),
            transitions: Vec::new(),
            key_effects: Vec::new(),
            key_components: Vec::new(),
            projections: Vec::new(),
            images: Vec::new(),
            response: empty_response_for_test(),
        }
    }

    fn lineage() -> ClaimStatusIndexLineage {
        ClaimStatusIndexLineage {
            database_id: DATABASE,
            timeline_id: TIMELINE,
            recovery_prefix: 3,
        }
    }

    fn valid_claim<'a>(
        statement_digests: &'a [Digest],
        eligible_statement_bits: &'a [u8],
    ) -> AuthenticatedTransactionClaimStatus<'a> {
        AuthenticatedTransactionClaimStatus {
            database_id: DATABASE,
            timeline_id: TIMELINE,
            stable_transaction_id: 7,
            request_digest: REQUEST,
            statement_digests,
            intent: RetentionIntent {
                eligible_statement_bits,
                candidate_deadline: 0,
            },
            head: ClaimHead::Pending,
            completion: ClaimCompletion::Complete(CompleteClaimDurability {
                claim_sequence: 1,
                recovery_prefix: 3,
                location: AuthenticatedClaimLocation::CheckpointStatus {
                    checkpoint_sequence: 1,
                },
                allocator_child_sequence: Some(2),
                sequence_child_sequence: None,
                parent_effect_sequence: Some(3),
            }),
        }
    }

    fn index<'a>(
        claim: &'a AuthenticatedTransactionClaimStatus<'a>,
    ) -> AuthenticatedClaimStatusIndex<'a> {
        AuthenticatedClaimStatusIndex {
            lineage: lineage(),
            claims: std::slice::from_ref(claim),
        }
    }

    fn terminal_retention_facts<'a>() -> SealedTerminalRetentionFacts<'a> {
        const RETAINED_BITS: &[u8] = &[0b0000_0101];
        const ARTIFACT_STATEMENTS: &[u32] = &[0, 2];
        SealedTerminalRetentionFacts {
            s6_retained_statement_bits: RETAINED_BITS,
            s7_retained_statement_bits: RETAINED_BITS,
            s8_artifact_statement_ordinals: ARTIFACT_STATEMENTS,
            s8_artifact_count: 2,
            aggregate_retained_response: true,
            status_artifact_count: 2,
            status_deadline: 99,
        }
    }
}
