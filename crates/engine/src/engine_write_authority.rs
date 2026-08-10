//! Strict canonical control bodies for WRITE-001 pre-parent authorities.
//!
//! These bodies use the already-assigned canonical fragment kinds
//! `TransactionClaimStatus` and `AllocatorLease`. They are not relational engine operations and
//! must never pass through the legacy `GPUDBOP1` decoder. The enclosing canonical envelope owns
//! physical framing, lineage, ordering, and its terminal marker; this module owns only the exact
//! control-body grammar and its intrinsic cross-field closure.

use crate::{EngineError, Index, TxnId};
use std::collections::HashMap;

const CLAIM_MAGIC: &[u8; 16] = b"GPUDBCLAIM2\0\0\0\0\0";
const CLAIM_VERSION: u16 = 1;
const CLAIM_HEADER_BYTES: usize = 184;

const ALLOCATOR_MAGIC: &[u8; 16] = b"GPUDBALLOC1\0\0\0\0\0";
const ALLOCATOR_VERSION: u16 = 1;
const ALLOCATOR_HEADER_BYTES: usize = 272;
const ALLOCATOR_ASSIGNMENT_BYTES: usize = 104;

pub(crate) const TABLE_ROW_ALLOCATOR_KIND: u8 = 1;
pub(crate) const EXACT_ROW_ASSIGNMENT_MAPPING_V1: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CanonicalRetentionClaim {
    pub(crate) identity: gpu_db_wal::CanonicalIdentity,
    pub(crate) leader_epoch: u64,
    pub(crate) parent_stable_transaction_id: u64,
    pub(crate) parent_request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) parent_autocommit: bool,
    pub(crate) statement_digests: Box<[gpu_db_wal::CanonicalDigest]>,
    pub(crate) eligible_statement_bits: Box<[u8]>,
    pub(crate) candidate_deadline: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct CanonicalRetentionClaimView<'a> {
    pub(crate) identity: gpu_db_wal::CanonicalIdentity,
    pub(crate) leader_epoch: u64,
    pub(crate) parent_stable_transaction_id: u64,
    pub(crate) parent_request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) parent_autocommit: bool,
    pub(crate) statement_digests: &'a [gpu_db_wal::CanonicalDigest],
    pub(crate) eligible_statement_bits: &'a [u8],
    pub(crate) candidate_deadline: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CanonicalAllocatorAssignment {
    pub(crate) parent_statement_ordinal: u32,
    pub(crate) source_row_ordinal: u32,
    pub(crate) parent_statement_request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) parent_typed_statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) source_order: u64,
    pub(crate) assignment_start: u64,
    pub(crate) assignment_end: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CanonicalAllocatorLease {
    pub(crate) identity: gpu_db_wal::CanonicalIdentity,
    pub(crate) leader_epoch: u64,
    pub(crate) allocator_kind: u8,
    pub(crate) mapping_version: u32,
    pub(crate) stable_allocator_id: u64,
    pub(crate) lease_epoch: u64,
    pub(crate) lease_start: u64,
    pub(crate) lease_end: u64,
    pub(crate) prior_high_water: u64,
    pub(crate) new_high_water: u64,
    pub(crate) marker_system_transaction_id: u64,
    pub(crate) marker_commit_sequence: u64,
    pub(crate) parent_stable_transaction_id: u64,
    pub(crate) parent_request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) parent_commit_sequence: u64,
    pub(crate) parent_autocommit: bool,
    pub(crate) assignments: Box<[CanonicalAllocatorAssignment]>,
}

#[derive(Clone, Copy)]
pub(crate) struct CanonicalAllocatorLeaseView<'a> {
    pub(crate) identity: gpu_db_wal::CanonicalIdentity,
    pub(crate) leader_epoch: u64,
    pub(crate) allocator_kind: u8,
    pub(crate) mapping_version: u32,
    pub(crate) stable_allocator_id: u64,
    pub(crate) lease_epoch: u64,
    pub(crate) lease_start: u64,
    pub(crate) lease_end: u64,
    pub(crate) prior_high_water: u64,
    pub(crate) new_high_water: u64,
    pub(crate) marker_system_transaction_id: u64,
    pub(crate) marker_commit_sequence: u64,
    pub(crate) parent_stable_transaction_id: u64,
    pub(crate) parent_request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) parent_commit_sequence: u64,
    #[cfg(test)]
    pub(crate) parent_autocommit: bool,
    pub(crate) assignments: &'a [CanonicalAllocatorAssignment],
}

pub(crate) enum DecodedWriteAuthority {
    RetentionClaim(CanonicalRetentionClaim),
    AllocatorLease(CanonicalAllocatorLease),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DurableRetentionClaimHead {
    pub(crate) claim_sequence: Index,
    pub(crate) terminal_sequence: Option<Index>,
    pub(crate) claim: CanonicalRetentionClaim,
}

/// The sole live/recovered lookup projection of the canonical claim and allocator records.
/// Canonical WAL remains authority; these maps exist only to make retry, parent binding, and
/// allocator non-overlap checks bounded rather than rescanning retained history.
#[derive(Clone, Default)]
pub(crate) struct DurableWriteAuthorityIndex {
    retention_claims: HashMap<TxnId, DurableRetentionClaimHead>,
    allocator_leases_by_marker: HashMap<TxnId, CanonicalAllocatorLease>,
    allocator_markers_by_parent_allocator: HashMap<(TxnId, u64), TxnId>,
    allocator_high_water_by_epoch: HashMap<(u64, u64), u64>,
}

impl DurableWriteAuthorityIndex {
    /// A clear-bit one-record codec-5 terminal has no retired claim or allocator control record.
    /// Recovery must reject any retained marker for that parent rather than interpreting absence
    /// of a partial historical chain as permission to downgrade it.
    pub(crate) fn has_any_parent_marker(&self, parent_txn_id: TxnId) -> bool {
        self.retention_claims.contains_key(&parent_txn_id)
            || self
                .allocator_leases_by_marker
                .values()
                .any(|lease| lease.parent_stable_transaction_id == parent_txn_id)
            || self
                .allocator_markers_by_parent_allocator
                .keys()
                .any(|(parent, _)| *parent == parent_txn_id)
    }

    /// Verify the retired bit-30 writer's complete pre-parent control chain without changing its
    /// terminal state. The later shared publisher remains the one authority that marks it
    /// terminal after exact device application succeeds.
    #[allow(clippy::too_many_arguments)] // retained historical control record fields are validated independently
    pub(crate) fn validate_historical_codec5_parent(
        &self,
        parent_txn_id: TxnId,
        request_digest: gpu_db_wal::CanonicalDigest,
        commit_sequence: Index,
        typed_statement_digest: gpu_db_wal::CanonicalDigest,
        row_allocator_before: u64,
        row_allocator_high_water: u64,
        affected_rows: u64,
    ) -> Result<(), EngineError> {
        let head = self.retention_claims.get(&parent_txn_id).ok_or_else(|| {
            authority_error("historical codec-5 parent has no retained transaction claim")
        })?;
        if head.claim.parent_request_digest != request_digest
            || !head.claim.parent_autocommit
            || head.claim.statement_digests.as_ref() != [typed_statement_digest]
            || head.claim_sequence >= commit_sequence
            || head.terminal_sequence.is_some()
        {
            return Err(authority_error(
                "historical codec-5 parent differs from its retained claim",
            ));
        }
        let mut matching = self.allocator_leases_by_marker.values().filter(|lease| {
            lease.parent_stable_transaction_id == parent_txn_id
                && lease.parent_commit_sequence == commit_sequence
        });
        let lease = matching.next().ok_or_else(|| {
            authority_error("historical codec-5 parent has no retained allocator lease")
        })?;
        if matching.next().is_some()
            || lease.parent_request_digest != request_digest
            || !lease.parent_autocommit
            || lease.marker_commit_sequence <= head.claim_sequence
            || lease.marker_commit_sequence >= commit_sequence
            || lease.prior_high_water != row_allocator_before
            || lease.lease_start != row_allocator_before
            || lease.lease_end != row_allocator_high_water
            || lease.new_high_water != row_allocator_high_water
            || lease.assignments.len() as u64 != affected_rows
            || lease.assignments.iter().any(|assignment| {
                assignment.parent_statement_ordinal != 0
                    || assignment.parent_statement_request_digest != request_digest
                    || assignment.parent_typed_statement_digest != typed_statement_digest
            })
        {
            return Err(authority_error(
                "historical codec-5 parent differs from its retained allocator lease",
            ));
        }
        Ok(())
    }

    pub(crate) fn apply(
        &mut self,
        commit_sequence: Index,
        authority: DecodedWriteAuthority,
        observed_row_allocator_high_water: u64,
    ) -> Result<Option<u64>, EngineError> {
        match authority {
            DecodedWriteAuthority::RetentionClaim(claim) => {
                let parent_txn_id = claim.parent_stable_transaction_id;
                if self.retention_claims.contains_key(&parent_txn_id) {
                    return Err(authority_error(
                        "retention claim repeats an already-authenticated parent identity",
                    ));
                }
                self.retention_claims.insert(
                    parent_txn_id,
                    DurableRetentionClaimHead {
                        claim_sequence: commit_sequence,
                        terminal_sequence: None,
                        claim,
                    },
                );
                Ok(None)
            }
            DecodedWriteAuthority::AllocatorLease(lease) => {
                self.validate_allocator_against_claim_and_index(
                    commit_sequence,
                    &lease,
                    observed_row_allocator_high_water,
                )?;
                let marker = lease.marker_system_transaction_id;
                let parent_allocator = (
                    lease.parent_stable_transaction_id,
                    lease.stable_allocator_id,
                );
                let allocator_epoch = (lease.stable_allocator_id, lease.lease_epoch);
                let new_high_water = lease.new_high_water;
                self.allocator_leases_by_marker.insert(marker, lease);
                self.allocator_markers_by_parent_allocator
                    .insert(parent_allocator, marker);
                self.allocator_high_water_by_epoch
                    .insert(allocator_epoch, new_high_water);
                Ok(Some(new_high_water))
            }
        }
    }

    pub(crate) fn mark_parent_terminal(
        &mut self,
        parent_txn_id: TxnId,
        request_digest: gpu_db_wal::CanonicalDigest,
        commit_sequence: Index,
        autocommit: bool,
        typed_statement_digests: &[gpu_db_wal::CanonicalDigest],
    ) -> Result<(), EngineError> {
        let head = self
            .retention_claims
            .get_mut(&parent_txn_id)
            .ok_or_else(|| {
                authority_error("typed parent has no retained pre-parent transaction claim")
            })?;
        if head.claim.parent_request_digest != request_digest
            || head.claim.parent_autocommit != autocommit
            || head.claim.statement_digests.as_ref() != typed_statement_digests
            || head.claim_sequence >= commit_sequence
            || head.terminal_sequence.is_some()
        {
            return Err(authority_error(
                "typed parent differs from its retained transaction claim",
            ));
        }
        let mut saw_allocator = false;
        for lease in self.allocator_leases_by_marker.values().filter(|lease| {
            lease.parent_stable_transaction_id == parent_txn_id
                && lease.parent_commit_sequence == commit_sequence
        }) {
            saw_allocator = true;
            if lease.parent_request_digest != request_digest
                || lease.parent_autocommit != autocommit
                || lease.marker_commit_sequence <= head.claim_sequence
                || lease.marker_commit_sequence >= commit_sequence
            {
                return Err(authority_error(
                    "typed parent differs from its retained allocator lease",
                ));
            }
        }
        if !saw_allocator {
            return Err(authority_error(
                "typed parent has no retained allocator lease",
            ));
        }
        head.terminal_sequence = Some(commit_sequence);
        Ok(())
    }

    fn validate_allocator_against_claim_and_index(
        &self,
        commit_sequence: Index,
        lease: &CanonicalAllocatorLease,
        observed_row_allocator_high_water: u64,
    ) -> Result<(), EngineError> {
        if lease.marker_commit_sequence != commit_sequence
            || self
                .allocator_leases_by_marker
                .contains_key(&lease.marker_system_transaction_id)
            || self.allocator_markers_by_parent_allocator.contains_key(&(
                lease.parent_stable_transaction_id,
                lease.stable_allocator_id,
            ))
        {
            return Err(authority_error(
                "allocator marker or parent allocator identity is repeated",
            ));
        }
        let claim = self
            .retention_claims
            .get(&lease.parent_stable_transaction_id)
            .ok_or_else(|| authority_error("allocator lease has no retained parent claim"))?;
        if claim.terminal_sequence.is_some()
            || claim.claim.parent_request_digest != lease.parent_request_digest
            || claim.claim.parent_autocommit != lease.parent_autocommit
            || claim.claim_sequence >= commit_sequence
            || commit_sequence >= lease.parent_commit_sequence
        {
            return Err(authority_error(
                "allocator lease differs from its retained parent claim",
            ));
        }
        for assignment in lease.assignments.iter() {
            let statement = usize::try_from(assignment.parent_statement_ordinal)
                .ok()
                .and_then(|ordinal| claim.claim.statement_digests.get(ordinal));
            if statement != Some(&assignment.parent_typed_statement_digest) {
                return Err(authority_error(
                    "allocator assignment is not bound to its claimed typed statement",
                ));
            }
        }
        let allocator_epoch = (lease.stable_allocator_id, lease.lease_epoch);
        let expected_prior = self
            .allocator_high_water_by_epoch
            .get(&allocator_epoch)
            .copied()
            .unwrap_or(observed_row_allocator_high_water);
        if lease.prior_high_water != expected_prior || lease.lease_start != expected_prior {
            return Err(authority_error(
                "allocator lease is not contiguous with the authenticated high-water",
            ));
        }
        Ok(())
    }
}

pub(crate) fn decode_write_authority_envelope(
    envelope: &gpu_db_wal::CanonicalEnvelope,
) -> Result<Option<DecodedWriteAuthority>, EngineError> {
    let [fragment] = envelope.fragments.as_slice() else {
        return Ok(None);
    };
    let decoded = match fragment.kind {
        gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus => {
            let claim = decode_retention_claim(&fragment.body)?;
            if claim.identity != envelope.header.identity
                || claim.leader_epoch != envelope.header.leader_epoch
                || claim.parent_stable_transaction_id != envelope.header.stable_transaction_id
                || claim.parent_request_digest != envelope.header.request_digest
                || envelope.header.allocator_high_water != 0
            {
                return Err(authority_error(
                    "retention claim body differs from its canonical envelope",
                ));
            }
            DecodedWriteAuthority::RetentionClaim(claim)
        }
        gpu_db_wal::CanonicalFragmentKind::AllocatorLease => {
            let lease = decode_allocator_lease(&fragment.body)?;
            if lease.identity != envelope.header.identity
                || lease.leader_epoch != envelope.header.leader_epoch
                || lease.marker_system_transaction_id != envelope.header.stable_transaction_id
                || lease.marker_commit_sequence != envelope.header.commit_seq
                || gpu_db_wal::canonical_request_digest(&fragment.body)
                    != envelope.header.request_digest
                || lease.new_high_water != envelope.header.allocator_high_water
            {
                return Err(authority_error(
                    "allocator lease body differs from its canonical envelope",
                ));
            }
            DecodedWriteAuthority::AllocatorLease(lease)
        }
        _ => return Ok(None),
    };
    if envelope.header.isolation != gpu_db_wal::CanonicalIsolation::System
        || envelope.header.flags != u32::from(fragment.kind as u16)
        || envelope.header.catalog_before_epoch != envelope.header.catalog_after_epoch
        || envelope.header.catalog_before_digest != envelope.header.catalog_after_digest
        || envelope.header.operation_count != 1
        || envelope.header.table_block_count != 0
        || envelope.outcome.kind != gpu_db_wal::CanonicalOutcomeKind::CommitNoOp
        || envelope.outcome.affected_rows != 0
        || envelope.outcome.sqlstate.is_some()
        || envelope.outcome.constraint_id != 0
        || envelope.outcome.returning_digest != [0; 32]
        || envelope.outcome.target_digest != gpu_db_wal::canonical_request_digest(&fragment.body)
    {
        return Err(authority_error(
            "canonical control envelope metadata is noncanonical",
        ));
    }
    Ok(Some(decoded))
}

#[cfg(test)]
fn encode_historical_retention_claim_fixture(
    claim: CanonicalRetentionClaimView<'_>,
) -> Result<Vec<u8>, EngineError> {
    validate_retention_claim_view(&claim)?;
    let statement_bytes = claim
        .statement_digests
        .len()
        .checked_mul(32)
        .ok_or_else(|| authority_error("claim statement arena overflows"))?;
    let bitset_offset = CLAIM_HEADER_BYTES
        .checked_add(statement_bytes)
        .ok_or_else(|| authority_error("claim statement arena overflows"))?;
    let total_bytes = bitset_offset
        .checked_add(claim.eligible_statement_bits.len())
        .ok_or_else(|| authority_error("claim body length overflows"))?;
    let total_u64 = u64::try_from(total_bytes)
        .map_err(|_| authority_error("claim body exceeds u64 framing"))?;
    let mut body = vec![0_u8; total_bytes];
    body[..16].copy_from_slice(CLAIM_MAGIC);
    put_u16(&mut body, 16, CLAIM_VERSION);
    put_u16(&mut body, 18, CLAIM_HEADER_BYTES as u16);
    put_u64(&mut body, 24, total_u64);
    body[32..48].copy_from_slice(&claim.identity.database_id);
    body[48..64].copy_from_slice(&claim.identity.cluster_id);
    body[64..80].copy_from_slice(&claim.identity.timeline_id);
    put_u64(&mut body, 80, claim.identity.format_epoch);
    put_u64(&mut body, 88, claim.leader_epoch);
    put_u64(&mut body, 96, claim.parent_stable_transaction_id);
    body[104..136].copy_from_slice(&claim.parent_request_digest);
    body[136] = u8::from(claim.parent_autocommit);
    put_u32(
        &mut body,
        144,
        u32::try_from(claim.statement_digests.len())
            .map_err(|_| authority_error("claim statement count exceeds u32"))?,
    );
    put_u32(
        &mut body,
        148,
        u32::try_from(claim.eligible_statement_bits.len())
            .map_err(|_| authority_error("claim eligibility bitset exceeds u32"))?,
    );
    put_u64(&mut body, 152, claim.candidate_deadline);
    put_u64(&mut body, 160, CLAIM_HEADER_BYTES as u64);
    put_u64(
        &mut body,
        168,
        u64::try_from(bitset_offset)
            .map_err(|_| authority_error("claim bitset offset exceeds u64"))?,
    );
    let mut at = CLAIM_HEADER_BYTES;
    for digest in claim.statement_digests {
        body[at..at + 32].copy_from_slice(digest);
        at += 32;
    }
    body[at..].copy_from_slice(claim.eligible_statement_bits);
    Ok(body)
}

pub(crate) fn decode_retention_claim(body: &[u8]) -> Result<CanonicalRetentionClaim, EngineError> {
    if body.len() < CLAIM_HEADER_BYTES || body.get(..16) != Some(CLAIM_MAGIC) {
        return Err(authority_error("retention claim header is malformed"));
    }
    if read_u16(body, 16)? != CLAIM_VERSION
        || usize::from(read_u16(body, 18)?) != CLAIM_HEADER_BYTES
        || read_u32(body, 20)? != 0
        || read_u64(body, 24)?
            != u64::try_from(body.len())
                .map_err(|_| authority_error("claim body exceeds u64 framing"))?
        || body[137..144] != [0; 7]
        || read_u64(body, 176)? != 0
    {
        return Err(authority_error(
            "retention claim fixed fields are noncanonical",
        ));
    }
    let statement_count = usize::try_from(read_u32(body, 144)?)
        .map_err(|_| authority_error("claim statement count is unaddressable"))?;
    let bitset_bytes = usize::try_from(read_u32(body, 148)?)
        .map_err(|_| authority_error("claim bitset length is unaddressable"))?;
    let statement_offset = usize::try_from(read_u64(body, 160)?)
        .map_err(|_| authority_error("claim statement offset is unaddressable"))?;
    let bitset_offset = usize::try_from(read_u64(body, 168)?)
        .map_err(|_| authority_error("claim bitset offset is unaddressable"))?;
    let expected_bitset_offset = statement_offset
        .checked_add(
            statement_count
                .checked_mul(32)
                .ok_or_else(|| authority_error("claim statement arena overflows"))?,
        )
        .ok_or_else(|| authority_error("claim statement arena overflows"))?;
    let expected_end = bitset_offset
        .checked_add(bitset_bytes)
        .ok_or_else(|| authority_error("claim bitset end overflows"))?;
    if statement_offset != CLAIM_HEADER_BYTES
        || bitset_offset != expected_bitset_offset
        || expected_end != body.len()
    {
        return Err(authority_error("retention claim arenas are noncanonical"));
    }
    let mut statement_digests = Vec::new();
    statement_digests
        .try_reserve_exact(statement_count)
        .map_err(|_| authority_error("claim statement owner reservation failed"))?;
    for chunk in body[statement_offset..bitset_offset].chunks_exact(32) {
        statement_digests.push(chunk.try_into().expect("claim digest chunk is exact"));
    }
    let claim = CanonicalRetentionClaim {
        identity: gpu_db_wal::CanonicalIdentity {
            database_id: read_array(body, 32)?,
            cluster_id: read_array(body, 48)?,
            timeline_id: read_array(body, 64)?,
            format_epoch: read_u64(body, 80)?,
        },
        leader_epoch: read_u64(body, 88)?,
        parent_stable_transaction_id: read_u64(body, 96)?,
        parent_request_digest: read_array(body, 104)?,
        parent_autocommit: decode_bool(body[136], "claim autocommit")?,
        statement_digests: statement_digests.into_boxed_slice(),
        eligible_statement_bits: body[bitset_offset..].to_vec().into_boxed_slice(),
        candidate_deadline: read_u64(body, 152)?,
    };
    validate_retention_claim(&claim)?;
    Ok(claim)
}

#[cfg(test)]
fn encode_historical_allocator_lease_fixture(
    lease: CanonicalAllocatorLeaseView<'_>,
) -> Result<Vec<u8>, EngineError> {
    validate_allocator_lease_view(&lease)?;
    let assignment_bytes = lease
        .assignments
        .len()
        .checked_mul(ALLOCATOR_ASSIGNMENT_BYTES)
        .ok_or_else(|| authority_error("allocator assignment arena overflows"))?;
    let total_bytes = ALLOCATOR_HEADER_BYTES
        .checked_add(assignment_bytes)
        .ok_or_else(|| authority_error("allocator body length overflows"))?;
    let mut body = vec![0_u8; total_bytes];
    body[..16].copy_from_slice(ALLOCATOR_MAGIC);
    put_u16(&mut body, 16, ALLOCATOR_VERSION);
    put_u16(&mut body, 18, ALLOCATOR_HEADER_BYTES as u16);
    put_u64(
        &mut body,
        24,
        u64::try_from(total_bytes)
            .map_err(|_| authority_error("allocator body exceeds u64 framing"))?,
    );
    body[32..48].copy_from_slice(&lease.identity.database_id);
    body[48..64].copy_from_slice(&lease.identity.cluster_id);
    body[64..80].copy_from_slice(&lease.identity.timeline_id);
    put_u64(&mut body, 80, lease.identity.format_epoch);
    put_u64(&mut body, 88, lease.leader_epoch);
    body[96] = lease.allocator_kind;
    body[97] = u8::from(lease.parent_autocommit);
    put_u32(&mut body, 100, lease.mapping_version);
    put_u64(&mut body, 104, lease.stable_allocator_id);
    put_u64(&mut body, 112, lease.lease_epoch);
    put_u64(&mut body, 120, lease.lease_start);
    put_u64(&mut body, 128, lease.lease_end);
    put_u64(&mut body, 136, lease.prior_high_water);
    put_u64(&mut body, 144, lease.new_high_water);
    put_u64(&mut body, 152, lease.marker_system_transaction_id);
    put_u64(&mut body, 160, lease.marker_commit_sequence);
    put_u64(&mut body, 168, lease.parent_stable_transaction_id);
    body[176..208].copy_from_slice(&lease.parent_request_digest);
    put_u64(&mut body, 240, lease.parent_commit_sequence);
    put_u32(
        &mut body,
        248,
        u32::try_from(lease.assignments.len())
            .map_err(|_| authority_error("allocator assignment count exceeds u32"))?,
    );
    put_u64(&mut body, 256, ALLOCATOR_HEADER_BYTES as u64);
    let mut at = ALLOCATOR_HEADER_BYTES;
    for assignment in lease.assignments {
        put_u32(&mut body, at, assignment.parent_statement_ordinal);
        put_u32(&mut body, at + 4, assignment.source_row_ordinal);
        // This body contains exactly one lease, so every assignment selects lease ordinal zero.
        put_u32(&mut body, at + 8, 0);
        body[at + 16..at + 48].copy_from_slice(&assignment.parent_statement_request_digest);
        body[at + 48..at + 80].copy_from_slice(&assignment.parent_typed_statement_digest);
        put_u64(&mut body, at + 80, assignment.source_order);
        put_u64(&mut body, at + 88, assignment.assignment_start);
        put_u64(&mut body, at + 96, assignment.assignment_end);
        at += ALLOCATOR_ASSIGNMENT_BYTES;
    }
    Ok(body)
}

pub(crate) fn decode_allocator_lease(body: &[u8]) -> Result<CanonicalAllocatorLease, EngineError> {
    if body.len() < ALLOCATOR_HEADER_BYTES || body.get(..16) != Some(ALLOCATOR_MAGIC) {
        return Err(authority_error("allocator lease header is malformed"));
    }
    if read_u16(body, 16)? != ALLOCATOR_VERSION
        || usize::from(read_u16(body, 18)?) != ALLOCATOR_HEADER_BYTES
        || read_u32(body, 20)? != 0
        || read_u64(body, 24)?
            != u64::try_from(body.len())
                .map_err(|_| authority_error("allocator body exceeds u64 framing"))?
        || body[98..100] != [0; 2]
        || read_u32(body, 252)? != 0
        || read_u64(body, 264)? != 0
    {
        return Err(authority_error(
            "allocator lease fixed fields are noncanonical",
        ));
    }
    let assignment_count = usize::try_from(read_u32(body, 248)?)
        .map_err(|_| authority_error("allocator assignment count is unaddressable"))?;
    let assignments_offset = usize::try_from(read_u64(body, 256)?)
        .map_err(|_| authority_error("allocator assignment offset is unaddressable"))?;
    let expected_end = assignments_offset
        .checked_add(
            assignment_count
                .checked_mul(ALLOCATOR_ASSIGNMENT_BYTES)
                .ok_or_else(|| authority_error("allocator assignment arena overflows"))?,
        )
        .ok_or_else(|| authority_error("allocator assignment arena overflows"))?;
    if assignments_offset != ALLOCATOR_HEADER_BYTES || expected_end != body.len() {
        return Err(authority_error(
            "allocator assignment arena is noncanonical",
        ));
    }
    let mut assignments = Vec::new();
    assignments
        .try_reserve_exact(assignment_count)
        .map_err(|_| authority_error("allocator assignment owner reservation failed"))?;
    for ordinal in 0..assignment_count {
        let at = assignments_offset + ordinal * ALLOCATOR_ASSIGNMENT_BYTES;
        if read_u32(body, at + 8)? != 0 || read_u32(body, at + 12)? != 0 {
            return Err(authority_error(
                "allocator assignment lease reference or reserved field is noncanonical",
            ));
        }
        assignments.push(CanonicalAllocatorAssignment {
            parent_statement_ordinal: read_u32(body, at)?,
            source_row_ordinal: read_u32(body, at + 4)?,
            parent_statement_request_digest: read_array(body, at + 16)?,
            parent_typed_statement_digest: read_array(body, at + 48)?,
            source_order: read_u64(body, at + 80)?,
            assignment_start: read_u64(body, at + 88)?,
            assignment_end: read_u64(body, at + 96)?,
        });
    }
    let lease = CanonicalAllocatorLease {
        identity: gpu_db_wal::CanonicalIdentity {
            database_id: read_array(body, 32)?,
            cluster_id: read_array(body, 48)?,
            timeline_id: read_array(body, 64)?,
            format_epoch: read_u64(body, 80)?,
        },
        leader_epoch: read_u64(body, 88)?,
        allocator_kind: body[96],
        mapping_version: read_u32(body, 100)?,
        stable_allocator_id: read_u64(body, 104)?,
        lease_epoch: read_u64(body, 112)?,
        lease_start: read_u64(body, 120)?,
        lease_end: read_u64(body, 128)?,
        prior_high_water: read_u64(body, 136)?,
        new_high_water: read_u64(body, 144)?,
        marker_system_transaction_id: read_u64(body, 152)?,
        marker_commit_sequence: read_u64(body, 160)?,
        parent_stable_transaction_id: read_u64(body, 168)?,
        parent_request_digest: read_array(body, 176)?,
        parent_commit_sequence: read_u64(body, 240)?,
        parent_autocommit: decode_bool(body[97], "allocator parent autocommit")?,
        assignments: assignments.into_boxed_slice(),
    };
    validate_allocator_lease(&lease)?;
    Ok(lease)
}

fn validate_retention_claim(claim: &CanonicalRetentionClaim) -> Result<(), EngineError> {
    validate_retention_claim_view(&CanonicalRetentionClaimView {
        identity: claim.identity,
        leader_epoch: claim.leader_epoch,
        parent_stable_transaction_id: claim.parent_stable_transaction_id,
        parent_request_digest: claim.parent_request_digest,
        parent_autocommit: claim.parent_autocommit,
        statement_digests: &claim.statement_digests,
        eligible_statement_bits: &claim.eligible_statement_bits,
        candidate_deadline: claim.candidate_deadline,
    })
}

fn validate_retention_claim_view(
    claim: &CanonicalRetentionClaimView<'_>,
) -> Result<(), EngineError> {
    validate_lineage(claim.identity, claim.leader_epoch)?;
    if !ordinary_identity(claim.parent_stable_transaction_id)
        || zero_digest(&claim.parent_request_digest)
        || claim.statement_digests.is_empty()
        || claim.statement_digests.iter().any(zero_digest)
        || claim.parent_autocommit && claim.statement_digests.len() != 1
    {
        return Err(authority_error(
            "retention claim identity or statement chain is invalid",
        ));
    }
    let expected_bits = claim.statement_digests.len().div_ceil(8);
    if claim.eligible_statement_bits.len() != expected_bits {
        return Err(authority_error(
            "retention claim eligibility bitset length is invalid",
        ));
    }
    let high_bits = expected_bits * 8 - claim.statement_digests.len();
    if high_bits != 0
        && claim
            .eligible_statement_bits
            .last()
            .is_some_and(|last| last & (!0_u8 << (8 - high_bits)) != 0)
    {
        return Err(authority_error(
            "retention claim eligibility high bits are nonzero",
        ));
    }
    let any_eligible = claim.eligible_statement_bits.iter().any(|byte| *byte != 0);
    if any_eligible != (claim.candidate_deadline != 0) || claim.candidate_deadline == u64::MAX {
        return Err(authority_error("retention claim deadline form is invalid"));
    }
    Ok(())
}

fn validate_allocator_lease(lease: &CanonicalAllocatorLease) -> Result<(), EngineError> {
    validate_allocator_lease_view(&CanonicalAllocatorLeaseView {
        identity: lease.identity,
        leader_epoch: lease.leader_epoch,
        allocator_kind: lease.allocator_kind,
        mapping_version: lease.mapping_version,
        stable_allocator_id: lease.stable_allocator_id,
        lease_epoch: lease.lease_epoch,
        lease_start: lease.lease_start,
        lease_end: lease.lease_end,
        prior_high_water: lease.prior_high_water,
        new_high_water: lease.new_high_water,
        marker_system_transaction_id: lease.marker_system_transaction_id,
        marker_commit_sequence: lease.marker_commit_sequence,
        parent_stable_transaction_id: lease.parent_stable_transaction_id,
        parent_request_digest: lease.parent_request_digest,
        parent_commit_sequence: lease.parent_commit_sequence,
        #[cfg(test)]
        parent_autocommit: lease.parent_autocommit,
        assignments: &lease.assignments,
    })
}

fn validate_allocator_lease_view(
    lease: &CanonicalAllocatorLeaseView<'_>,
) -> Result<(), EngineError> {
    validate_lineage(lease.identity, lease.leader_epoch)?;
    if lease.allocator_kind != TABLE_ROW_ALLOCATOR_KIND
        || lease.mapping_version != EXACT_ROW_ASSIGNMENT_MAPPING_V1
        || !ordinary_identity(lease.stable_allocator_id)
        || !ordinary_identity(lease.lease_epoch)
        || lease.prior_high_water > lease.lease_start
        || lease.lease_start >= lease.lease_end
        || lease.lease_end != lease.new_high_water
        || !ordinary_identity(lease.marker_system_transaction_id)
        || !ordinary_identity(lease.marker_commit_sequence)
        || !ordinary_identity(lease.parent_stable_transaction_id)
        || !ordinary_identity(lease.parent_commit_sequence)
        || lease.marker_system_transaction_id == lease.parent_stable_transaction_id
        || lease.marker_commit_sequence >= lease.parent_commit_sequence
        || zero_digest(&lease.parent_request_digest)
        || lease.assignments.is_empty()
    {
        return Err(authority_error(
            "allocator lease identity or interval is invalid",
        ));
    }
    let lease_len = lease
        .lease_end
        .checked_sub(lease.lease_start)
        .ok_or_else(|| authority_error("allocator lease interval underflows"))?;
    if usize::try_from(lease_len).ok() != Some(lease.assignments.len()) {
        return Err(authority_error(
            "allocator lease and assignment counts differ",
        ));
    }
    for (ordinal, assignment) in lease.assignments.iter().enumerate() {
        let expected_order = u64::try_from(ordinal)
            .map_err(|_| authority_error("allocator source order exceeds u64"))?;
        let expected_start = lease
            .lease_start
            .checked_add(expected_order)
            .ok_or_else(|| authority_error("allocator assignment identity overflows"))?;
        if assignment.source_order != expected_order
            || assignment.assignment_start != expected_start
            || assignment.assignment_end
                != expected_start.checked_add(1).ok_or_else(|| {
                    authority_error("allocator assignment exclusive end overflows")
                })?
            || zero_digest(&assignment.parent_statement_request_digest)
            || zero_digest(&assignment.parent_typed_statement_digest)
        {
            return Err(authority_error(
                "allocator exact assignment order is invalid",
            ));
        }
    }
    Ok(())
}

fn validate_lineage(
    identity: gpu_db_wal::CanonicalIdentity,
    leader_epoch: u64,
) -> Result<(), EngineError> {
    if identity.database_id == [0; 16]
        || identity.cluster_id == [0; 16]
        || identity.timeline_id == [0; 16]
        || identity.format_epoch == 0
        || leader_epoch == 0
    {
        return Err(authority_error("control-body lineage is invalid"));
    }
    Ok(())
}

fn ordinary_identity(value: u64) -> bool {
    value != 0 && value != u64::MAX
}

fn zero_digest(value: &gpu_db_wal::CanonicalDigest) -> bool {
    *value == [0; 32]
}

fn decode_bool(value: u8, label: &str) -> Result<bool, EngineError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(authority_error(label)),
    }
}

fn read_array<const N: usize>(bytes: &[u8], at: usize) -> Result<[u8; N], EngineError> {
    bytes
        .get(
            at..at
                .checked_add(N)
                .ok_or_else(|| authority_error("offset overflows"))?,
        )
        .and_then(|value| value.try_into().ok())
        .ok_or_else(|| authority_error("control body is truncated"))
}

fn read_u16(bytes: &[u8], at: usize) -> Result<u16, EngineError> {
    Ok(u16::from_le_bytes(read_array(bytes, at)?))
}

fn read_u32(bytes: &[u8], at: usize) -> Result<u32, EngineError> {
    Ok(u32::from_le_bytes(read_array(bytes, at)?))
}

fn read_u64(bytes: &[u8], at: usize) -> Result<u64, EngineError> {
    Ok(u64::from_le_bytes(read_array(bytes, at)?))
}

#[cfg(test)]
fn put_u16(bytes: &mut [u8], at: usize, value: u16) {
    bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
fn put_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
fn put_u64(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

fn authority_error(message: &str) -> EngineError {
    EngineError::Durability(format!("canonical WRITE-001 authority: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> gpu_db_wal::CanonicalIdentity {
        gpu_db_wal::CanonicalIdentity {
            database_id: [1; 16],
            cluster_id: [2; 16],
            timeline_id: [3; 16],
            format_epoch: 4,
        }
    }

    #[test]
    fn claim_round_trips_and_rejects_noncanonical_bitsets() {
        let statements = [[5; 32], [6; 32]];
        let encoded = encode_historical_retention_claim_fixture(CanonicalRetentionClaimView {
            identity: identity(),
            leader_epoch: 7,
            parent_stable_transaction_id: 8,
            parent_request_digest: [9; 32],
            parent_autocommit: false,
            statement_digests: &statements,
            eligible_statement_bits: &[1],
            candidate_deadline: 10,
        })
        .unwrap();
        let decoded = decode_retention_claim(&encoded).unwrap();
        assert_eq!(decoded.statement_digests.as_ref(), statements);
        assert_eq!(decoded.eligible_statement_bits.as_ref(), [1]);
        assert_eq!(decoded.candidate_deadline, 10);

        let mut high_bit = encoded.clone();
        *high_bit.last_mut().unwrap() = 0x80;
        assert!(decode_retention_claim(&high_bit).is_err());
        let mut gap = encoded;
        gap.insert(CLAIM_HEADER_BYTES, 0);
        let gap_len = gap.len() as u64;
        put_u64(&mut gap, 24, gap_len);
        assert!(decode_retention_claim(&gap).is_err());
    }

    #[test]
    fn allocator_round_trips_exact_assignments_and_rejects_reconstruction() {
        let assignments = [
            CanonicalAllocatorAssignment {
                parent_statement_ordinal: 0,
                source_row_ordinal: 0,
                parent_statement_request_digest: [10; 32],
                parent_typed_statement_digest: [11; 32],
                source_order: 0,
                assignment_start: 20,
                assignment_end: 21,
            },
            CanonicalAllocatorAssignment {
                parent_statement_ordinal: 0,
                source_row_ordinal: 1,
                parent_statement_request_digest: [10; 32],
                parent_typed_statement_digest: [11; 32],
                source_order: 1,
                assignment_start: 21,
                assignment_end: 22,
            },
        ];
        let encoded = encode_historical_allocator_lease_fixture(CanonicalAllocatorLeaseView {
            identity: identity(),
            leader_epoch: 7,
            allocator_kind: TABLE_ROW_ALLOCATOR_KIND,
            mapping_version: EXACT_ROW_ASSIGNMENT_MAPPING_V1,
            stable_allocator_id: 12,
            lease_epoch: 13,
            lease_start: 20,
            lease_end: 22,
            prior_high_water: 20,
            new_high_water: 22,
            marker_system_transaction_id: 14,
            marker_commit_sequence: 15,
            parent_stable_transaction_id: 16,
            parent_request_digest: [17; 32],
            parent_commit_sequence: 16,
            parent_autocommit: true,
            assignments: &assignments,
        })
        .unwrap();
        let decoded = decode_allocator_lease(&encoded).unwrap();
        assert_eq!(decoded.assignments.as_ref(), assignments);

        let mut forged = encoded;
        put_u64(
            &mut forged,
            ALLOCATOR_HEADER_BYTES + ALLOCATOR_ASSIGNMENT_BYTES + 88,
            20,
        );
        assert!(decode_allocator_lease(&forged).is_err());
    }

    #[test]
    fn allocator_exact_order_is_generic_across_statement_row_ordinals() {
        let assignments = [
            CanonicalAllocatorAssignment {
                parent_statement_ordinal: 2,
                source_row_ordinal: 4,
                parent_statement_request_digest: [10; 32],
                parent_typed_statement_digest: [11; 32],
                source_order: 0,
                assignment_start: 20,
                assignment_end: 21,
            },
            CanonicalAllocatorAssignment {
                parent_statement_ordinal: 7,
                source_row_ordinal: 0,
                parent_statement_request_digest: [12; 32],
                parent_typed_statement_digest: [13; 32],
                source_order: 1,
                assignment_start: 21,
                assignment_end: 22,
            },
        ];
        let encoded = encode_historical_allocator_lease_fixture(CanonicalAllocatorLeaseView {
            identity: identity(),
            leader_epoch: 7,
            allocator_kind: TABLE_ROW_ALLOCATOR_KIND,
            mapping_version: EXACT_ROW_ASSIGNMENT_MAPPING_V1,
            stable_allocator_id: 12,
            lease_epoch: 13,
            lease_start: 20,
            lease_end: 22,
            prior_high_water: 20,
            new_high_water: 22,
            marker_system_transaction_id: 14,
            marker_commit_sequence: 15,
            parent_stable_transaction_id: 16,
            parent_request_digest: [17; 32],
            parent_commit_sequence: 16,
            parent_autocommit: false,
            assignments: &assignments,
        })
        .unwrap();
        assert_eq!(
            decode_allocator_lease(&encoded)
                .unwrap()
                .assignments
                .as_ref(),
            assignments
        );
    }
}
