//! Canonical ADR-014 WAL transaction envelope.
//!
//! The storage backends remain responsible for physical frame CRCs and durable prefixes. This
//! layer supplies the logical authority inside those frames: one immutable outcome-free header,
//! non-circular fragment leaves/root, one terminal commit/no-op/abort marker, and an explicit
//! lane-local physical to global logical mapping. A verifier accepts only the complete ordered
//! frame range; no fragment can independently authorize apply or publication.

use gpu_db_types::EngineError;
use sha2::{Digest, Sha256};
use std::sync::Arc;

mod exact;
mod generation_terminal;
pub use exact::{
    encode_canonical_record_exact_from_borrowed, encode_canonical_record_exact_into,
    measure_canonical_exact_buffers, measure_canonical_exact_buffers_from_fragments,
    CanonicalFragmentRef, ExactCanonicalRecordEncoding,
};
use generation_terminal::{
    canonical_terminal_marker_digest_from_encoded, decode_canonical_terminal_marker,
    encode_canonical_terminal_marker_into, measure_canonical_terminal_marker,
    CanonicalTerminalMarker,
};
#[cfg(test)]
mod exact_tests;

pub type CanonicalDigest = [u8; 32];

const HEADER_MAGIC: &[u8; 16] = b"GPUDBCANHDR1\0\0\0\0";
const FRAME_MAGIC: &[u8; 16] = b"GPUDBCANWAL1\0\0\0\0";
const RECORD_MAGIC: &[u8; 16] = b"GPUDBCANREC1\0\0\0\0";
const FORMAT_VERSION: u16 = 1;
const SEMANTICS_VERSION: u16 = 1;
const MIN_READER_VERSION: u16 = 1;
const MAX_READER_VERSION: u16 = 1;
const FRAME_FRAGMENT: u8 = 1;
const FRAME_MARKER: u8 = 2;
const FRAME_FIXED_BYTES: usize = 244;
const FRAME_DIGEST_BYTES: usize = 32;
const PREAPPLY_HEADER_BYTES: usize = 240;
/// Exact byte width of a canonical terminal outcome.
///
/// This is an independently usable, fixed-width subcodec of the canonical envelope.  Callers
/// that reserve a complete envelope can use [`encode_canonical_outcome_into_exact`] without an
/// intermediate allocation, while readers must use [`decode_canonical_outcome_exact`] to retain
/// the layout's no-truncation/no-surplus rule.
pub const CANONICAL_OUTCOME_BYTES: usize = 92;
const PACKED_RECORD_PREFIX_BYTES: usize = RECORD_MAGIC.len() + std::mem::size_of::<u32>();
const PACKED_FRAME_LENGTH_BYTES: usize = std::mem::size_of::<u32>();
const MAX_FRAGMENT_BYTES: usize = 16 * 1024 * 1024;
const MAX_ENVELOPE_BYTES: usize = 64 * 1024 * 1024;

/// Maximum raw canonical-fragment body accepted by the real envelope encoder.
pub const fn canonical_fragment_body_limit() -> u64 {
    MAX_FRAGMENT_BYTES as u64
}

const PREAPPLY_DOMAIN: &[u8] = b"gpu-db/adr014/preapply/v1";
const FRAGMENT_LEAF_DOMAIN: &[u8] = b"gpu-db/adr014/fragment-leaf/v1";
const FRAGMENT_ROOT_DOMAIN: &[u8] = b"gpu-db/adr014/fragment-root/v1";
const FINAL_DOMAIN: &[u8] = b"gpu-db/adr014/final-outcome/v1";
const FRAME_DOMAIN: &[u8] = b"gpu-db/adr014/physical-frame/v1";

fn durability(message: impl Into<String>) -> EngineError {
    EngineError::Durability(format!("canonical WAL: {}", message.into()))
}

/// Exact storage framing for canonical fragments before any physical range/header values exist.
///
/// This is an accounting-only projection of the canonical encoder and packer. It carries no WAL
/// record, physical coordinates, or append capability, and every bound matches the real
/// `encode_canonical_envelope`/`pack_canonical_record_payload` path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalWalFootprint {
    pub fragment_count: u32,
    /// Canonical fragment frames plus the one required terminal outcome-marker frame.
    pub frame_count: u32,
    /// Sum of the raw canonical fragment bodies, excluding fragment kinds and framing.
    pub fragment_body_bytes: u64,
    /// One pre-apply header plus the raw fragment bodies, the encoder's envelope-limit input.
    pub preapply_bytes: u64,
    /// All fragment frames including their digest tails, but excluding packed u32 lengths.
    pub fragment_frame_bytes: u64,
    /// The terminal outcome-marker frame including its digest tail, but excluding packed length.
    pub marker_frame_bytes: u64,
    /// Packed fragment slots, including each u32 frame length.
    pub fragment_packed_bytes: u64,
    /// Packed terminal marker slot, including its u32 frame length.
    pub marker_packed_bytes: u64,
    /// Exact `WalRecord::payload` bytes after canonical frame packing.
    pub packed_record_bytes: u64,
    /// Exact serialized record bytes including the immutable 24-byte outer WAL header.
    pub serialized_record_bytes: u64,
}

/// Compute canonical framing bytes from already-bounded fragment-body lengths without encoding
/// payload buffers. The caller supplies only body lengths: kind, header, digest, marker, packed
/// slot, and outer-record costs remain owned here beside the real encoder.
pub fn canonical_wal_footprint(
    fragment_body_lengths: &[u64],
) -> Result<CanonicalWalFootprint, EngineError> {
    canonical_wal_footprint_by_index(fragment_body_lengths.len(), |index| {
        Ok(fragment_body_lengths[index])
    })
}

/// Shared framing arithmetic for the allocating v1 encoder, the public count-only projection,
/// and the exact caller-buffer encoder. `body_len` is intentionally index-addressed so the exact
/// path can traverse existing fragments without building a length collection.
pub(super) fn canonical_wal_footprint_by_index(
    fragment_len: usize,
    body_len: impl FnMut(usize) -> Result<u64, EngineError>,
) -> Result<CanonicalWalFootprint, EngineError> {
    canonical_wal_footprint_by_index_with_marker(fragment_len, body_len, CANONICAL_OUTCOME_BYTES)
}

fn canonical_wal_footprint_by_index_with_marker(
    fragment_len: usize,
    mut body_len: impl FnMut(usize) -> Result<u64, EngineError>,
    terminal_marker_bytes: usize,
) -> Result<CanonicalWalFootprint, EngineError> {
    let fragment_count = u32::try_from(fragment_len)
        .map_err(|_| durability("fragment count exceeds u32 framing"))?;
    if fragment_count == 0 || fragment_count == u32::MAX {
        return Err(durability("fragment count must be in 1..u32::MAX"));
    }
    let frame_count = fragment_count
        .checked_add(1)
        .ok_or_else(|| durability("frame count overflows"))?;

    let max_fragment = u64::try_from(MAX_FRAGMENT_BYTES)
        .map_err(|_| durability("fragment byte bound exceeds u64"))?;
    let max_envelope = u64::try_from(MAX_ENVELOPE_BYTES)
        .map_err(|_| durability("envelope byte bound exceeds u64"))?;
    let preapply_header = u64::try_from(PREAPPLY_HEADER_BYTES)
        .map_err(|_| durability("pre-apply header bytes exceed u64"))?;
    let frame_fixed =
        u64::try_from(FRAME_FIXED_BYTES).map_err(|_| durability("frame fixed bytes exceed u64"))?;
    let frame_digest = u64::try_from(FRAME_DIGEST_BYTES)
        .map_err(|_| durability("frame digest bytes exceed u64"))?;
    let packed_prefix = u64::try_from(PACKED_RECORD_PREFIX_BYTES)
        .map_err(|_| durability("packed record prefix exceeds u64"))?;
    let packed_length = u64::try_from(PACKED_FRAME_LENGTH_BYTES)
        .map_err(|_| durability("packed frame length exceeds u64"))?;
    let terminal_marker_bytes = u64::try_from(terminal_marker_bytes)
        .map_err(|_| durability("terminal marker bytes exceed u64"))?;
    let outer_record = u64::try_from(crate::WAL_RECORD_HEADER_LEN)
        .map_err(|_| durability("outer WAL header exceeds u64"))?;

    let mut fragment_body_bytes = 0_u64;
    let mut fragment_frame_bytes = 0_u64;
    for index in 0..fragment_len {
        let body = body_len(index)?;
        if body > max_fragment {
            return Err(durability(format!("fragment {index} exceeds byte bound")));
        }
        fragment_body_bytes = fragment_body_bytes
            .checked_add(body)
            .ok_or_else(|| durability("fragment body byte sum overflows"))?;
        let header = if index == 0 { preapply_header } else { 0 };
        let frame = frame_fixed
            .checked_add(header)
            .and_then(|bytes| bytes.checked_add(body))
            .and_then(|bytes| bytes.checked_add(frame_digest))
            .ok_or_else(|| durability("fragment frame byte length overflows"))?;
        fragment_frame_bytes = fragment_frame_bytes
            .checked_add(frame)
            .ok_or_else(|| durability("fragment frame byte sum overflows"))?;
    }
    let preapply_bytes = preapply_header
        .checked_add(fragment_body_bytes)
        .ok_or_else(|| durability("pre-apply envelope byte length overflows"))?;
    if preapply_bytes > max_envelope {
        return Err(durability("envelope exceeds byte bound"));
    }

    let marker_frame_bytes = frame_fixed
        .checked_add(terminal_marker_bytes)
        .and_then(|bytes| bytes.checked_add(frame_digest))
        .ok_or_else(|| durability("marker frame byte length overflows"))?;
    // The packed-record budget is deliberately looser than one canonical frame.  A variable
    // root-format-v1 terminal marker must still fit the decoder's per-frame ceiling, otherwise
    // the encoder could persist a record the canonical reader is required to reject.
    if marker_frame_bytes > max_envelope {
        return Err(durability("terminal marker frame exceeds byte bound"));
    }
    let fragment_packed_bytes = fragment_frame_bytes
        .checked_add(
            packed_length
                .checked_mul(u64::from(fragment_count))
                .ok_or_else(|| durability("fragment packed length count overflows"))?,
        )
        .ok_or_else(|| durability("fragment packed byte sum overflows"))?;
    let marker_packed_bytes = packed_length
        .checked_add(marker_frame_bytes)
        .ok_or_else(|| durability("marker packed byte length overflows"))?;
    let packed_record_bytes = packed_prefix
        .checked_add(fragment_packed_bytes)
        .and_then(|bytes| bytes.checked_add(marker_packed_bytes))
        .ok_or_else(|| durability("canonical record byte length overflow"))?;
    let max_packed = max_envelope
        .checked_add(1024 * 1024)
        .ok_or_else(|| durability("canonical packed byte bound overflows"))?;
    if packed_record_bytes > max_packed {
        return Err(durability("canonical record exceeds byte bound"));
    }
    let serialized_record_bytes = packed_record_bytes
        .checked_add(outer_record)
        .ok_or_else(|| durability("serialized record byte length overflow"))?;

    Ok(CanonicalWalFootprint {
        fragment_count,
        frame_count,
        fragment_body_bytes,
        preapply_bytes,
        fragment_frame_bytes,
        marker_frame_bytes,
        fragment_packed_bytes,
        marker_packed_bytes,
        packed_record_bytes,
        serialized_record_bytes,
    })
}

fn digest_parts(domain: &[u8], parts: &[&[u8]]) -> CanonicalDigest {
    let mut hash = Sha256::new();
    hash.update((domain.len() as u64).to_le_bytes());
    hash.update(domain);
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    hash.finalize().into()
}

/// Stable request identity used by transaction claims and same-id retry resolution.
pub fn canonical_request_digest(request: &[u8]) -> CanonicalDigest {
    digest_parts(b"gpu-db/adr014/request/v1", &[request])
}

fn nonzero_identity(value: &[u8; 16], field: &str) -> Result<(), EngineError> {
    if value.iter().all(|byte| *byte == 0) {
        return Err(durability(format!("{field} must not be all zero")));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalIdentity {
    pub database_id: [u8; 16],
    pub cluster_id: [u8; 16],
    pub timeline_id: [u8; 16],
    pub format_epoch: u64,
}

impl CanonicalIdentity {
    fn validate(&self) -> Result<(), EngineError> {
        nonzero_identity(&self.database_id, "database id")?;
        nonzero_identity(&self.cluster_id, "cluster id")?;
        nonzero_identity(&self.timeline_id, "timeline id")?;
        if self.format_epoch == 0 {
            return Err(durability("format epoch zero is reserved"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CanonicalIsolation {
    ReadCommitted = 1,
    RepeatableRead = 2,
    System = 3,
}

impl CanonicalIsolation {
    fn decode(value: u8) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::ReadCommitted),
            2 => Ok(Self::RepeatableRead),
            3 => Ok(Self::System),
            other => Err(durability(format!("unsupported isolation code {other}"))),
        }
    }
}

/// Outcome-free immutable transaction header. No fragment root or terminal result is present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalPreApplyHeader {
    pub identity: CanonicalIdentity,
    pub leader_epoch: u64,
    pub commit_seq: u64,
    pub stable_transaction_id: u64,
    pub request_digest: CanonicalDigest,
    pub isolation: CanonicalIsolation,
    pub flags: u32,
    pub catalog_before_epoch: u64,
    pub catalog_after_epoch: u64,
    pub catalog_before_digest: CanonicalDigest,
    pub catalog_after_digest: CanonicalDigest,
    pub operation_count: u32,
    pub table_block_count: u32,
    pub allocator_high_water: u64,
}

impl CanonicalPreApplyHeader {
    fn validate(&self) -> Result<(), EngineError> {
        self.identity.validate()?;
        if self.leader_epoch == 0 {
            return Err(durability("leader/log epoch zero is reserved"));
        }
        if self.commit_seq == 0 || self.commit_seq == u64::MAX {
            return Err(durability("commit sequence must be in 1..=u64::MAX-1"));
        }
        if self.stable_transaction_id == 0 {
            return Err(durability("stable transaction id zero is reserved"));
        }
        if self.operation_count == 0 {
            return Err(durability("operation count must be non-zero"));
        }
        if self.catalog_after_epoch < self.catalog_before_epoch {
            return Err(durability("catalog epoch regressed"));
        }
        Ok(())
    }

    fn encode(&self) -> Result<Vec<u8>, EngineError> {
        let mut out = vec![0; PREAPPLY_HEADER_BYTES];
        self.encode_into(&mut out)?;
        Ok(out)
    }

    /// Encode the immutable header into its one fixed v1 layout. The exact-buffer path shares
    /// this field order with the legacy allocating encoder.
    pub(super) fn encode_into(&self, out: &mut [u8]) -> Result<(), EngineError> {
        self.validate()?;
        let mut writer = FixedEncoder::new(out);
        writer.bytes(HEADER_MAGIC)?;
        writer.u16(FORMAT_VERSION)?;
        writer.u16(SEMANTICS_VERSION)?;
        writer.u16(MIN_READER_VERSION)?;
        writer.u16(MAX_READER_VERSION)?;
        writer.bytes(&self.identity.database_id)?;
        writer.bytes(&self.identity.cluster_id)?;
        writer.bytes(&self.identity.timeline_id)?;
        writer.u64(self.identity.format_epoch)?;
        writer.u64(self.leader_epoch)?;
        writer.u64(self.commit_seq)?;
        writer.u64(self.stable_transaction_id)?;
        writer.bytes(&self.request_digest)?;
        writer.u8(self.isolation as u8)?;
        writer.bytes(&[0; 3])?;
        writer.u32(self.flags)?;
        writer.u64(self.catalog_before_epoch)?;
        writer.u64(self.catalog_after_epoch)?;
        writer.bytes(&self.catalog_before_digest)?;
        writer.bytes(&self.catalog_after_digest)?;
        writer.u32(self.operation_count)?;
        writer.u32(self.table_block_count)?;
        writer.u64(self.allocator_high_water)?;
        writer.finish()
    }

    fn decode(bytes: &[u8]) -> Result<Self, EngineError> {
        let mut cursor = Cursor::new(bytes);
        if cursor.take(HEADER_MAGIC.len())? != HEADER_MAGIC {
            return Err(durability("invalid pre-apply header magic"));
        }
        let format = cursor.u16()?;
        let semantics = cursor.u16()?;
        let min_reader = cursor.u16()?;
        let max_reader = cursor.u16()?;
        if format != FORMAT_VERSION
            || semantics != SEMANTICS_VERSION
            || min_reader > FORMAT_VERSION
            || max_reader < FORMAT_VERSION
        {
            return Err(durability(format!(
                "unsupported header versions format={format} semantics={semantics} reader={min_reader}..={max_reader}"
            )));
        }
        let identity = CanonicalIdentity {
            database_id: cursor.array()?,
            cluster_id: cursor.array()?,
            timeline_id: cursor.array()?,
            format_epoch: cursor.u64()?,
        };
        let leader_epoch = cursor.u64()?;
        let commit_seq = cursor.u64()?;
        let stable_transaction_id = cursor.u64()?;
        let request_digest = cursor.array()?;
        let isolation = CanonicalIsolation::decode(cursor.u8()?)?;
        if cursor.take(3)? != [0; 3] {
            return Err(durability("non-zero reserved header bytes"));
        }
        let flags = cursor.u32()?;
        let catalog_before_epoch = cursor.u64()?;
        let catalog_after_epoch = cursor.u64()?;
        let catalog_before_digest = cursor.array()?;
        let catalog_after_digest = cursor.array()?;
        let operation_count = cursor.u32()?;
        let table_block_count = cursor.u32()?;
        let allocator_high_water = cursor.u64()?;
        cursor.finish()?;
        let header = Self {
            identity,
            leader_epoch,
            commit_seq,
            stable_transaction_id,
            request_digest,
            isolation,
            flags,
            catalog_before_epoch,
            catalog_after_epoch,
            catalog_before_digest,
            catalog_after_digest,
            operation_count,
            table_block_count,
            allocator_high_water,
        };
        header.validate()?;
        Ok(header)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum CanonicalFragmentKind {
    RowMutation = 1,
    CatalogMutation = 2,
    TableReset = 3,
    TableRewrite = 4,
    SequenceValueTransition = 5,
    PrivateSequenceChild = 6,
    AllocatorLease = 7,
    TransactionClaimStatus = 8,
}

impl CanonicalFragmentKind {
    /// Decode the exact v1 wire value for a canonical fragment kind.
    ///
    /// This is the sole numeric-kind authority for current canonical-envelope readers.
    /// Callers must reject values not represented by this enum rather than assigning a
    /// compatibility fallback.
    pub fn decode(value: u16) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::RowMutation),
            2 => Ok(Self::CatalogMutation),
            3 => Ok(Self::TableReset),
            4 => Ok(Self::TableRewrite),
            5 => Ok(Self::SequenceValueTransition),
            6 => Ok(Self::PrivateSequenceChild),
            7 => Ok(Self::AllocatorLease),
            8 => Ok(Self::TransactionClaimStatus),
            other => Err(durability(format!("unsupported fragment kind {other}"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalFragment {
    pub kind: CanonicalFragmentKind,
    /// Canonical typed body owned by the engine format layer. SQL text is not a valid body.
    pub body: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CanonicalOutcomeKind {
    CommitSuccess = 1,
    CommitNoOp = 2,
    AbortError = 3,
}

impl CanonicalOutcomeKind {
    fn decode(value: u8) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::CommitSuccess),
            2 => Ok(Self::CommitNoOp),
            3 => Ok(Self::AbortError),
            other => Err(durability(format!("unsupported outcome kind {other}"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalOutcome {
    pub kind: CanonicalOutcomeKind,
    pub affected_rows: u64,
    /// Five-byte PostgreSQL SQLSTATE for `AbortError`; absent for success/no-op.
    pub sqlstate: Option<[u8; 5]>,
    /// Stable constraint identity, or zero when the outcome is not constraint-specific.
    pub constraint_id: u64,
    pub target_digest: CanonicalDigest,
    pub returning_digest: CanonicalDigest,
}

impl CanonicalOutcome {
    fn validate(&self) -> Result<(), EngineError> {
        match (self.kind, self.sqlstate) {
            (CanonicalOutcomeKind::AbortError, Some(state))
                if state.iter().all(u8::is_ascii_alphanumeric) => {}
            (CanonicalOutcomeKind::AbortError, _) => {
                return Err(durability("AbortError requires an ASCII SQLSTATE"));
            }
            (_, None) => {}
            (_, Some(_)) => return Err(durability("successful outcome must not carry SQLSTATE")),
        }
        Ok(())
    }

    fn encode(&self) -> Result<Vec<u8>, EngineError> {
        let mut out = vec![0; CANONICAL_OUTCOME_BYTES];
        self.encode_into(&mut out)?;
        Ok(out)
    }

    /// Encode the fixed terminal-outcome layout used by both v1 encoder forms.
    pub(super) fn encode_into(&self, out: &mut [u8]) -> Result<(), EngineError> {
        self.validate()?;
        let mut writer = FixedEncoder::new(out);
        writer.u8(self.kind as u8)?;
        writer.u8(u8::from(self.sqlstate.is_some()))?;
        writer.bytes(&[0; 2])?;
        writer.u64(self.affected_rows)?;
        writer.u64(self.constraint_id)?;
        writer.bytes(&self.sqlstate.unwrap_or([0; 5]))?;
        writer.bytes(&[0; 3])?;
        writer.bytes(&self.target_digest)?;
        writer.bytes(&self.returning_digest)?;
        writer.finish()
    }

    fn decode(bytes: &[u8]) -> Result<Self, EngineError> {
        let mut cursor = Cursor::new(bytes);
        let kind = CanonicalOutcomeKind::decode(cursor.u8()?)?;
        let has_sqlstate = match cursor.u8()? {
            0 => false,
            1 => true,
            other => return Err(durability(format!("invalid SQLSTATE flag {other}"))),
        };
        if cursor.take(2)? != [0; 2] {
            return Err(durability("non-zero outcome reserved bytes"));
        }
        let affected_rows = cursor.u64()?;
        let constraint_id = cursor.u64()?;
        let state: [u8; 5] = cursor.array()?;
        if !has_sqlstate && state != [0; 5] {
            return Err(durability("non-zero absent outcome SQLSTATE"));
        }
        if cursor.take(3)? != [0; 3] {
            return Err(durability("non-zero outcome padding"));
        }
        let target_digest = cursor.array()?;
        let returning_digest = cursor.array()?;
        cursor.finish()?;
        let outcome = Self {
            kind,
            affected_rows,
            sqlstate: has_sqlstate.then_some(state),
            constraint_id,
            target_digest,
            returning_digest,
        };
        outcome.validate()?;
        Ok(outcome)
    }
}

/// Encode one canonical terminal outcome into the exact fixed-width caller buffer.
///
/// The output remains untouched when validation fails.  This owns no envelope framing and makes
/// no allocation; it is deliberately the same private layout writer used by v1 canonical
/// envelopes.
pub fn encode_canonical_outcome_into_exact(
    outcome: &CanonicalOutcome,
    out: &mut [u8; CANONICAL_OUTCOME_BYTES],
) -> Result<(), EngineError> {
    outcome.encode_into(out)
}

/// Decode one canonical terminal outcome only when `bytes` is the exact fixed width.
///
/// In addition to rejecting truncation and surplus, this enforces every reserved byte and the
/// canonical SQLSTATE flag/value relation before yielding an outcome.
pub fn decode_canonical_outcome_exact(bytes: &[u8]) -> Result<CanonicalOutcome, EngineError> {
    if bytes.len() != CANONICAL_OUTCOME_BYTES {
        return Err(durability(format!(
            "canonical outcome length {} does not match exact width {CANONICAL_OUTCOME_BYTES}",
            bytes.len()
        )));
    }
    CanonicalOutcome::decode(bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalPhysicalRange {
    pub log_epoch: u64,
    pub lane_id: u32,
    pub segment_id: u64,
    pub first_frame_ordinal: u64,
}

impl CanonicalPhysicalRange {
    fn validate(&self, frame_count: u32) -> Result<(), EngineError> {
        if self.log_epoch == 0 || self.segment_id == 0 {
            return Err(durability("log epoch and segment id must be non-zero"));
        }
        self.first_frame_ordinal
            .checked_add(u64::from(frame_count))
            .ok_or_else(|| durability("physical frame ordinal overflow"))?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalEnvelope {
    pub physical: CanonicalPhysicalRange,
    pub header: CanonicalPreApplyHeader,
    pub fragments: Vec<CanonicalFragment>,
    /// The exact terminal marker body. `outcome` is retained as a compatibility projection for
    /// existing callers that need only PostgreSQL result fields.
    pub(crate) terminal_marker: CanonicalTerminalMarker,
    pub outcome: CanonicalOutcome,
    pub ordered_fragment_root: CanonicalDigest,
    pub final_digest: CanonicalDigest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedCanonicalEnvelope {
    /// Fragment frames followed by exactly one terminal outcome-marker frame.
    frames: Vec<Vec<u8>>,
    ordered_fragment_root: CanonicalDigest,
    final_digest: CanonicalDigest,
    // Kept private so only `encode_canonical_envelope` can bind the prepared record's immutable
    // outer bytes and catalog tail to the same validated pre-apply header.
    header: CanonicalPreApplyHeader,
}

/// The catalog boundary carried by the last canonical logical record.
///
/// This is intentionally small and copyable: `WalBuffer` owns the cache, while the engine owns
/// genesis derivation when no canonical record has yet been appended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalCatalogTail {
    pub identity: CanonicalIdentity,
    pub catalog_after_epoch: u64,
    pub catalog_after_digest: CanonicalDigest,
}

/// An immutable canonical WAL record prepared from one successfully encoded envelope.
///
/// Fields are private on purpose. The only construction path is
/// [`EncodedCanonicalEnvelope::into_prepared_record`], which seals the exact record bytes and
/// catalog-after tail from the same validated header. Live owners hand this value directly to
/// [`crate::WalBuffer::append_canonical`].
#[derive(Debug, PartialEq, Eq)]
pub struct PreparedCanonicalWalRecord {
    record: crate::WalRecord,
    tail: CanonicalCatalogTail,
    /// Typed INSERT's pre-WAL owner retains the exact outer bytes and the immutable inputs that
    /// produced them.  Generic/legacy canonical callers deliberately leave this absent and keep
    /// the allocating compatibility encoder until they migrate to the typed lifecycle.
    exact: Option<ExactCanonicalRecordAuthority>,
}

/// Immutable authority for one exact-buffer canonical record.
///
/// The packed bytes remain in the paired [`crate::WalRecord`] because replication owns that
/// payload.  This value owns only the independently framed outer record used by `WalBuffer`,
/// together with enough immutable evidence to reject a mismatched owner before it mutates a
/// logical frontier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExactCanonicalRecordAuthority {
    serialized_record: Arc<[u8]>,
    encoding: ExactCanonicalRecordEncoding,
    physical: CanonicalPhysicalRange,
    header: CanonicalPreApplyHeader,
}

impl ExactCanonicalRecordAuthority {
    /// Exact v1 outer WAL bytes, including the 24-byte storage header and checksum.
    pub(crate) fn serialized_record(&self) -> &Arc<[u8]> {
        &self.serialized_record
    }

    /// Immutable geometry/digest proof from the exact encoder.
    pub(crate) fn encoding(&self) -> ExactCanonicalRecordEncoding {
        self.encoding
    }

    /// The identity/catalog/commit binding validated before proposal.
    pub(crate) fn header(&self) -> &CanonicalPreApplyHeader {
        &self.header
    }
}

impl PreparedCanonicalWalRecord {
    /// Borrow the sealed outer record for recovery/archive evidence. Normal live append owners
    /// consume this value through [`crate::WalBuffer::append_canonical`] instead.
    pub fn as_wal_record(&self) -> &crate::WalRecord {
        &self.record
    }

    /// Consume the sealed wrapper when a caller is deliberately materializing an offline/archive
    /// record rather than appending it to a live [`crate::WalBuffer`].
    pub fn into_wal_record(self) -> crate::WalRecord {
        self.record
    }

    /// Typed callers use this to prove that replication receives the precise packed bytes which
    /// remain paired with the prebuilt outer record.  It intentionally borrows the existing Arc;
    /// no payload materialization or re-encoding is permitted after proposal.
    pub(crate) fn exact_packed_payload(&self) -> Option<&Arc<[u8]>> {
        self.exact.as_ref().map(|_| &self.record.payload)
    }

    /// Typed callers use this to hand the prebuilt outer bytes to `WalBuffer` unchanged.
    pub(crate) fn exact_authority(&self) -> Option<&ExactCanonicalRecordAuthority> {
        self.exact.as_ref()
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        crate::WalRecord,
        CanonicalCatalogTail,
        Option<ExactCanonicalRecordAuthority>,
    ) {
        (self.record, self.tail, self.exact)
    }

    pub(crate) fn from_parts(
        record: crate::WalRecord,
        tail: CanonicalCatalogTail,
        exact: Option<ExactCanonicalRecordAuthority>,
    ) -> Self {
        Self {
            record,
            tail,
            exact,
        }
    }
}

impl EncodedCanonicalEnvelope {
    /// Seal the immutable outer record and the catalog tail produced by this exact envelope.
    pub fn into_prepared_record(
        self,
        txn_id: gpu_db_types::TxnId,
    ) -> Result<PreparedCanonicalWalRecord, EngineError> {
        let tail = CanonicalCatalogTail {
            identity: self.header.identity,
            catalog_after_epoch: self.header.catalog_after_epoch,
            catalog_after_digest: self.header.catalog_after_digest,
        };
        let payload = pack_canonical_record_payload(&self)?;
        Ok(PreparedCanonicalWalRecord {
            record: crate::WalRecord {
                txn_id,
                payload: payload.into(),
            },
            tail,
            exact: None,
        })
    }
}

/// Build typed INSERT's one exact canonical record before replication proposal.
///
/// This is intentionally the sole constructor that couples the packed `WalRecord` payload with
/// an Arc-backed outer serialized record.  Both buffers are exactly measured and filled before
/// this function returns; after proposal the caller can only move this immutable owner through
/// the reservation lifecycle.
pub fn prepare_exact_canonical_wal_record(
    txn_id: gpu_db_types::TxnId,
    physical: CanonicalPhysicalRange,
    header: CanonicalPreApplyHeader,
    fragments: &[CanonicalFragment],
    outcome: CanonicalOutcome,
) -> Result<PreparedCanonicalWalRecord, EngineError> {
    let footprint =
        measure_canonical_exact_buffers_from_fragments(physical, &header, fragments, &outcome)?;
    let packed_len = usize::try_from(footprint.packed_record_bytes)
        .map_err(|_| durability("packed canonical record exceeds addressable memory"))?;
    let serialized_len = usize::try_from(footprint.serialized_record_bytes)
        .map_err(|_| durability("serialized canonical record exceeds addressable memory"))?;
    // These are the persistent authority buffers.  They are allocated and Arc-backed before the
    // proposal boundary; later lifecycle phases only clone/move their Arc handles.
    let mut packed = exact_authority_buffer(packed_len, "packed canonical record")?;
    let mut serialized = exact_authority_buffer(serialized_len, "serialized canonical record")?;
    let encoding = encode_canonical_record_exact_into(
        txn_id,
        physical,
        &header,
        fragments,
        &outcome,
        &mut packed,
        &mut serialized,
    )?;
    debug_assert_eq!(encoding.footprint, footprint);
    let packed: Arc<[u8]> = packed.into();
    let serialized: Arc<[u8]> = serialized.into();
    let payload_start = crate::WAL_RECORD_HEADER_LEN;
    if serialized.get(payload_start..) != Some(packed.as_ref()) {
        return Err(durability(
            "exact serialized record payload diverged from packed replication payload",
        ));
    }
    let tail = CanonicalCatalogTail {
        identity: header.identity,
        catalog_after_epoch: header.catalog_after_epoch,
        catalog_after_digest: header.catalog_after_digest,
    };
    Ok(PreparedCanonicalWalRecord {
        record: crate::WalRecord {
            txn_id,
            payload: packed,
        },
        tail,
        exact: Some(ExactCanonicalRecordAuthority {
            serialized_record: serialized,
            encoding,
            physical,
            header,
        }),
    })
}

fn exact_authority_buffer(len: usize, label: &str) -> Result<Vec<u8>, EngineError> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(|_| {
        durability(format!(
            "unable to reserve exact {label} authority before replication proposal"
        ))
    })?;
    // The reservation above fixes capacity; resize cannot allocate and simply initializes the
    // exact caller-owned output range required by the encoder.
    bytes.resize(len, 0);
    Ok(bytes)
}

/// Exact manifest-accounted logical intent/outcome bytes.
///
/// This intentionally counts each typed fragment's two-byte kind plus body and the canonical
/// terminal outcome body. It excludes the immutable pre-apply identity/header, Merkle/digest
/// material, physical coordinates, frame headers/digests, record packing, and storage padding.
/// Those bytes remain durability authority and are reported separately as physical WAL bytes; they
/// are not request-amplified logical intent/result payload.
pub fn canonical_logical_intent_outcome_bytes(
    fragments: &[CanonicalFragment],
    outcome: &CanonicalOutcome,
) -> Result<u64, EngineError> {
    if fragments.is_empty() {
        return Err(durability("logical byte accounting requires a fragment"));
    }
    let mut total = 0u64;
    for (index, fragment) in fragments.iter().enumerate() {
        if fragment.body.len() > MAX_FRAGMENT_BYTES {
            return Err(durability(format!("fragment {index} exceeds byte bound")));
        }
        let body = u64::try_from(fragment.body.len())
            .map_err(|_| durability("logical fragment length exceeds u64 framing"))?;
        total = total
            .checked_add(2)
            .and_then(|value| value.checked_add(body))
            .ok_or_else(|| durability("logical intent byte length overflow"))?;
    }
    let outcome = outcome.encode()?;
    total
        .checked_add(
            u64::try_from(outcome.len())
                .map_err(|_| durability("logical outcome length exceeds u64 framing"))?,
        )
        .ok_or_else(|| durability("logical intent/outcome byte length overflow"))
}

/// Pack the physical fragment frames into the payload of one existing [`crate::WalRecord`].
/// The outer record remains the storage/replication indexing unit, while this inner container is
/// the canonical transaction authority.
pub fn pack_canonical_record_payload(
    encoded: &EncodedCanonicalEnvelope,
) -> Result<Vec<u8>, EngineError> {
    let frame_count = u32::try_from(encoded.frames.len())
        .map_err(|_| durability("canonical record frame count overflow"))?;
    let mut total = RECORD_MAGIC.len() + 4;
    for frame in &encoded.frames {
        total = total
            .checked_add(4)
            .and_then(|value| value.checked_add(frame.len()))
            .ok_or_else(|| durability("canonical record byte length overflow"))?;
    }
    if total > MAX_ENVELOPE_BYTES + 1024 * 1024 {
        return Err(durability("canonical record exceeds byte bound"));
    }
    let mut payload = Vec::with_capacity(total);
    payload.extend_from_slice(RECORD_MAGIC);
    put_u32(&mut payload, frame_count);
    for frame in &encoded.frames {
        put_u32(
            &mut payload,
            u32::try_from(frame.len())
                .map_err(|_| durability("canonical physical frame length overflow"))?,
        );
        payload.extend_from_slice(frame);
    }
    Ok(payload)
}

/// Decode a canonical transaction payload. `Ok(None)` identifies a legacy pre-ADR-014 payload;
/// once the magic is present every malformed/truncated byte is a loud durability error.
pub fn decode_canonical_record_payload(
    payload: &[u8],
) -> Result<Option<CanonicalEnvelope>, EngineError> {
    if !payload.starts_with(RECORD_MAGIC) {
        return Ok(None);
    }
    let mut cursor = Cursor::new(&payload[RECORD_MAGIC.len()..]);
    let frame_count = cursor.u32()? as usize;
    if frame_count < 2 {
        return Err(durability(
            "canonical record omits fragment or terminal marker",
        ));
    }
    let mut frames = Vec::with_capacity(frame_count);
    for _ in 0..frame_count {
        let len = cursor.u32()? as usize;
        if len > MAX_ENVELOPE_BYTES {
            return Err(durability("canonical record frame exceeds byte bound"));
        }
        frames.push(cursor.take(len)?.to_vec());
    }
    cursor.finish()?;
    decode_canonical_envelope(&frames).map(Some)
}

pub fn encode_canonical_envelope(
    physical: CanonicalPhysicalRange,
    header: &CanonicalPreApplyHeader,
    fragments: &[CanonicalFragment],
    outcome: &CanonicalOutcome,
) -> Result<EncodedCanonicalEnvelope, EngineError> {
    encode_canonical_envelope_with_terminal_marker(
        physical,
        header,
        fragments,
        &CanonicalTerminalMarker::Legacy(outcome.clone()),
    )
}

/// Encode a canonical envelope with either the historical 92-byte terminal marker or the
/// root-format-v1 terminal descriptor extension.  Current live writers call only the legacy
/// wrapper above; this generic entry point is intentionally the later authority seam for
/// root-format-v1 resolved/WAL-first operations.
pub(crate) fn encode_canonical_envelope_with_terminal_marker(
    physical: CanonicalPhysicalRange,
    header: &CanonicalPreApplyHeader,
    fragments: &[CanonicalFragment],
    marker: &CanonicalTerminalMarker,
) -> Result<EncodedCanonicalEnvelope, EngineError> {
    if fragments.is_empty() || fragments.len() > u32::MAX as usize - 1 {
        return Err(durability("fragment count must be in 1..u32::MAX"));
    }
    marker.validate_for_header(header)?;
    if header.operation_count as usize != fragments.len() {
        return Err(durability(format!(
            "header operation count {} does not match {} fragments",
            header.operation_count,
            fragments.len()
        )));
    }
    let marker_measure = measure_canonical_terminal_marker(marker)?;
    // Reject marker/frame growth before this allocating compatibility encoder materializes any
    // canonical body, leaf, frame, or packed record.  Root-format-v1 callers use the same
    // measure before their later exact reservation path exists.
    canonical_wal_footprint_by_index_with_marker(
        fragments.len(),
        |index| {
            u64::try_from(fragments[index].body.len())
                .map_err(|_| durability("fragment byte length exceeds u64 framing"))
        },
        marker_measure.marker_bytes,
    )?;
    let frame_count = fragments.len() as u32 + 1;
    physical.validate(frame_count)?;
    let header_bytes = header.encode()?;
    let header_digest = digest_parts(PREAPPLY_DOMAIN, &[&header_bytes]);
    let mut total = header_bytes.len();
    let mut canonical_bodies = Vec::with_capacity(fragments.len());
    let mut leaves = Vec::with_capacity(fragments.len());
    for (index, fragment) in fragments.iter().enumerate() {
        if fragment.body.len() > MAX_FRAGMENT_BYTES {
            return Err(durability(format!("fragment {index} exceeds byte bound")));
        }
        total = total
            .checked_add(fragment.body.len())
            .ok_or_else(|| durability("envelope byte overflow"))?;
        if total > MAX_ENVELOPE_BYTES {
            return Err(durability("envelope exceeds byte bound"));
        }
        let mut canonical = Vec::with_capacity(2 + fragment.body.len());
        put_u16(&mut canonical, fragment.kind as u16);
        canonical.extend_from_slice(&fragment.body);
        let index_bytes = (index as u32).to_le_bytes();
        let length_bytes = (canonical.len() as u64).to_le_bytes();
        let leaf = digest_parts(
            FRAGMENT_LEAF_DOMAIN,
            &[&header_digest, &index_bytes, &length_bytes, &canonical],
        );
        canonical_bodies.push(canonical);
        leaves.push(leaf);
    }
    let mut root_material = Vec::with_capacity(4 + leaves.len() * 44);
    put_u32(&mut root_material, fragments.len() as u32);
    for (index, (canonical, leaf)) in canonical_bodies.iter().zip(&leaves).enumerate() {
        put_u32(&mut root_material, index as u32);
        put_u64(&mut root_material, canonical.len() as u64);
        root_material.extend_from_slice(leaf);
    }
    let root = digest_parts(FRAGMENT_ROOT_DOMAIN, &[&root_material]);
    let mut marker_bytes = vec![0; marker_measure.marker_bytes];
    encode_canonical_terminal_marker_into(marker, &mut marker_bytes)?;
    let final_digest =
        canonical_terminal_marker_digest_from_encoded(&header_bytes, root, marker, &marker_bytes)?;
    let mut frames = Vec::with_capacity(frame_count as usize);
    for (index, (fragment, leaf)) in fragments.iter().zip(&leaves).enumerate() {
        frames.push(encode_frame(
            FRAME_FRAGMENT,
            physical,
            header,
            index as u32,
            fragments.len() as u32,
            fragment.kind as u16,
            if index == 0 {
                header_bytes.as_slice()
            } else {
                &[]
            },
            &fragment.body,
            header_digest,
            root,
            *leaf,
        )?);
    }
    frames.push(encode_frame(
        FRAME_MARKER,
        physical,
        header,
        fragments.len() as u32,
        fragments.len() as u32,
        0,
        &[],
        &marker_bytes,
        header_digest,
        root,
        final_digest,
    )?);
    Ok(EncodedCanonicalEnvelope {
        frames,
        ordered_fragment_root: root,
        final_digest,
        header: header.clone(),
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_frame(
    frame_type: u8,
    physical: CanonicalPhysicalRange,
    header: &CanonicalPreApplyHeader,
    frame_index: u32,
    fragment_count: u32,
    kind: u16,
    header_bytes: &[u8],
    body: &[u8],
    header_digest: CanonicalDigest,
    root: CanonicalDigest,
    leaf_or_final: CanonicalDigest,
) -> Result<Vec<u8>, EngineError> {
    let frame_len = canonical_frame_encoded_len(header_bytes.len(), body.len())?;
    let mut frame = vec![0; frame_len];
    encode_frame_into(
        &mut frame,
        frame_type,
        physical,
        header,
        frame_index,
        fragment_count,
        kind,
        header_bytes,
        body,
        header_digest,
        root,
        leaf_or_final,
    )?;
    Ok(frame)
}

pub(super) fn canonical_frame_encoded_len(
    header_len: usize,
    body_len: usize,
) -> Result<usize, EngineError> {
    FRAME_FIXED_BYTES
        .checked_add(header_len)
        .and_then(|bytes| bytes.checked_add(body_len))
        .and_then(|bytes| bytes.checked_add(FRAME_DIGEST_BYTES))
        .ok_or_else(|| durability("physical frame byte length overflow"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_frame_into(
    frame: &mut [u8],
    frame_type: u8,
    physical: CanonicalPhysicalRange,
    header: &CanonicalPreApplyHeader,
    frame_index: u32,
    fragment_count: u32,
    kind: u16,
    header_bytes: &[u8],
    body: &[u8],
    header_digest: CanonicalDigest,
    root: CanonicalDigest,
    leaf_or_final: CanonicalDigest,
) -> Result<(), EngineError> {
    let header_len = u32::try_from(header_bytes.len())
        .map_err(|_| durability("pre-apply header length overflow"))?;
    let body_len = u32::try_from(body.len()).map_err(|_| durability("frame body overflow"))?;
    let expected = canonical_frame_encoded_len(header_bytes.len(), body.len())?;
    if frame.len() != expected {
        return Err(durability("exact physical frame buffer length mismatch"));
    }
    let without_digest = expected - FRAME_DIGEST_BYTES;
    {
        let mut writer = FixedEncoder::new(&mut frame[..without_digest]);
        writer.bytes(FRAME_MAGIC)?;
        writer.u16(FORMAT_VERSION)?;
        writer.u16(SEMANTICS_VERSION)?;
        writer.u8(frame_type)?;
        writer.bytes(&[0; 3])?;
        writer.bytes(&header.identity.database_id)?;
        writer.bytes(&header.identity.cluster_id)?;
        writer.bytes(&header.identity.timeline_id)?;
        writer.u64(header.identity.format_epoch)?;
        writer.u64(physical.log_epoch)?;
        writer.u32(physical.lane_id)?;
        writer.u32(0)?;
        writer.u64(physical.segment_id)?;
        writer.u64(physical.first_frame_ordinal)?;
        writer.u64(header.stable_transaction_id)?;
        writer.u64(header.commit_seq)?;
        writer.u32(frame_index)?;
        writer.u32(fragment_count)?;
        writer.u16(kind)?;
        writer.u16(0)?;
        writer.u32(header_len)?;
        writer.u32(body_len)?;
        writer.bytes(&header_digest)?;
        writer.bytes(&root)?;
        writer.bytes(&leaf_or_final)?;
        debug_assert_eq!(writer.position(), FRAME_FIXED_BYTES);
        writer.bytes(header_bytes)?;
        writer.bytes(body)?;
        writer.finish()?;
    }
    let frame_digest = digest_parts(FRAME_DOMAIN, &[&frame[..without_digest]]);
    frame[without_digest..].copy_from_slice(&frame_digest);
    Ok(())
}

/// Decode the legacy canonical envelope form accepted by all current engine recovery/apply
/// callers. A root-format-v1 marker fails here until a later semantic decoder proves that its
/// fragments and root descriptor belong to the one immutable-generation authority.
pub fn decode_canonical_envelope(frames: &[Vec<u8>]) -> Result<CanonicalEnvelope, EngineError> {
    let envelope = decode_canonical_envelope_with_terminal_marker(frames)?;
    if !matches!(envelope.terminal_marker, CanonicalTerminalMarker::Legacy(_)) {
        return Err(durability(
            "root-format-v1 terminal descriptor requires its authenticated generation decoder",
        ));
    }
    Ok(envelope)
}

/// Private extended-marker decoder. A later root-format-v1 semantic/recovery owner may call this
/// only after it has authenticated the operation class and fragment grammar; it must not be
/// surfaced through the legacy generic replay path above.
pub(crate) fn decode_canonical_envelope_with_terminal_marker(
    frames: &[Vec<u8>],
) -> Result<CanonicalEnvelope, EngineError> {
    if frames.len() < 2 || frames.len() > u32::MAX as usize {
        return Err(durability("envelope requires fragments plus one marker"));
    }
    let decoded: Vec<DecodedFrame<'_>> = frames
        .iter()
        .map(|frame| decode_frame(frame))
        .collect::<Result<_, _>>()?;
    let first = &decoded[0];
    let fragment_count = first.fragment_count as usize;
    if fragment_count == 0 || frames.len() != fragment_count + 1 {
        return Err(durability(
            "physical frame count does not match fragment count",
        ));
    }
    if first.frame_type != FRAME_FRAGMENT || first.frame_index != 0 || first.header_bytes.is_empty()
    {
        return Err(durability("ordinal-zero mapping/header frame is absent"));
    }
    let header = CanonicalPreApplyHeader::decode(first.header_bytes)?;
    let physical = first.physical;
    physical.validate(frames.len() as u32)?;
    if physical.log_epoch != header.leader_epoch {
        return Err(durability(
            "physical log epoch does not match the pre-apply leader epoch",
        ));
    }
    let header_bytes = header.encode()?;
    let header_digest = digest_parts(PREAPPLY_DOMAIN, &[&header_bytes]);
    if header_digest != first.header_digest {
        return Err(durability("pre-apply header digest mismatch"));
    }
    let mut fragments = Vec::with_capacity(fragment_count);
    let mut root_material = Vec::with_capacity(4 + fragment_count * 44);
    put_u32(&mut root_material, fragment_count as u32);
    for (index, frame) in decoded[..fragment_count].iter().enumerate() {
        verify_common_frame(frame, first)?;
        if frame.frame_type != FRAME_FRAGMENT || frame.frame_index != index as u32 {
            return Err(durability(format!("fragment frame {index} is reordered")));
        }
        if index != 0 && !frame.header_bytes.is_empty() {
            return Err(durability("pre-apply header repeated outside ordinal zero"));
        }
        let kind = CanonicalFragmentKind::decode(frame.kind)?;
        let mut canonical = Vec::with_capacity(2 + frame.body.len());
        put_u16(&mut canonical, kind as u16);
        canonical.extend_from_slice(frame.body);
        let index_bytes = (index as u32).to_le_bytes();
        let length_bytes = (canonical.len() as u64).to_le_bytes();
        let expected_leaf = digest_parts(
            FRAGMENT_LEAF_DOMAIN,
            &[&header_digest, &index_bytes, &length_bytes, &canonical],
        );
        if expected_leaf != frame.leaf_or_final {
            return Err(durability(format!("fragment {index} leaf digest mismatch")));
        }
        put_u32(&mut root_material, index as u32);
        put_u64(&mut root_material, canonical.len() as u64);
        root_material.extend_from_slice(&expected_leaf);
        fragments.push(CanonicalFragment {
            kind,
            body: frame.body.to_vec(),
        });
    }
    let expected_root = digest_parts(FRAGMENT_ROOT_DOMAIN, &[&root_material]);
    if expected_root != first.root {
        return Err(durability("ordered fragment root mismatch"));
    }
    let marker = &decoded[fragment_count];
    verify_common_frame(marker, first)?;
    if marker.frame_type != FRAME_MARKER
        || marker.frame_index != fragment_count as u32
        || marker.kind != 0
        || !marker.header_bytes.is_empty()
    {
        return Err(durability("terminal marker is absent or malformed"));
    }
    let terminal_marker = decode_canonical_terminal_marker(marker.body)?;
    terminal_marker.validate_for_header(&header)?;
    let final_digest = canonical_terminal_marker_digest_from_encoded(
        &header_bytes,
        expected_root,
        &terminal_marker,
        marker.body,
    )?;
    if marker.leaf_or_final != final_digest {
        return Err(durability("terminal outcome digest mismatch"));
    }
    Ok(CanonicalEnvelope {
        physical,
        header,
        fragments,
        outcome: terminal_marker.outcome().clone(),
        terminal_marker,
        ordered_fragment_root: expected_root,
        final_digest,
    })
}

fn verify_common_frame(
    frame: &DecodedFrame<'_>,
    first: &DecodedFrame<'_>,
) -> Result<(), EngineError> {
    if frame.physical != first.physical
        || frame.identity != first.identity
        || frame.stable_transaction_id != first.stable_transaction_id
        || frame.commit_seq != first.commit_seq
        || frame.fragment_count != first.fragment_count
        || frame.header_digest != first.header_digest
        || frame.root != first.root
    {
        return Err(durability("frame mapping/header metadata mismatch"));
    }
    Ok(())
}

struct DecodedFrame<'a> {
    frame_type: u8,
    identity: CanonicalIdentity,
    physical: CanonicalPhysicalRange,
    stable_transaction_id: u64,
    commit_seq: u64,
    frame_index: u32,
    fragment_count: u32,
    kind: u16,
    header_digest: CanonicalDigest,
    root: CanonicalDigest,
    leaf_or_final: CanonicalDigest,
    header_bytes: &'a [u8],
    body: &'a [u8],
}

fn decode_frame(frame: &[u8]) -> Result<DecodedFrame<'_>, EngineError> {
    if frame.len() < FRAME_FIXED_BYTES + FRAME_DIGEST_BYTES {
        return Err(durability("truncated physical frame"));
    }
    let (without_digest, digest_bytes) = frame.split_at(frame.len() - FRAME_DIGEST_BYTES);
    let expected = digest_parts(FRAME_DOMAIN, &[without_digest]);
    if digest_bytes != expected {
        return Err(durability("physical frame digest mismatch"));
    }
    let mut cursor = Cursor::new(without_digest);
    if cursor.take(FRAME_MAGIC.len())? != FRAME_MAGIC {
        return Err(durability("invalid physical frame magic"));
    }
    let format = cursor.u16()?;
    let semantics = cursor.u16()?;
    if format != FORMAT_VERSION || semantics != SEMANTICS_VERSION {
        return Err(durability(format!(
            "unsupported frame versions format={format} semantics={semantics}"
        )));
    }
    let frame_type = cursor.u8()?;
    if !matches!(frame_type, FRAME_FRAGMENT | FRAME_MARKER) {
        return Err(durability(format!(
            "unsupported physical frame type {frame_type}"
        )));
    }
    if cursor.take(3)? != [0; 3] {
        return Err(durability("non-zero frame reserved bytes"));
    }
    let identity = CanonicalIdentity {
        database_id: cursor.array()?,
        cluster_id: cursor.array()?,
        timeline_id: cursor.array()?,
        format_epoch: cursor.u64()?,
    };
    identity.validate()?;
    let physical = CanonicalPhysicalRange {
        log_epoch: cursor.u64()?,
        lane_id: cursor.u32()?,
        segment_id: {
            if cursor.u32()? != 0 {
                return Err(durability("non-zero lane padding"));
            }
            cursor.u64()?
        },
        first_frame_ordinal: cursor.u64()?,
    };
    let stable_transaction_id = cursor.u64()?;
    let commit_seq = cursor.u64()?;
    let frame_index = cursor.u32()?;
    let fragment_count = cursor.u32()?;
    let kind = cursor.u16()?;
    if cursor.u16()? != 0 {
        return Err(durability("non-zero frame kind padding"));
    }
    let header_len = cursor.u32()? as usize;
    let body_len = cursor.u32()? as usize;
    let header_digest = cursor.array()?;
    let root = cursor.array()?;
    let leaf_or_final = cursor.array()?;
    debug_assert_eq!(cursor.position(), FRAME_FIXED_BYTES);
    let header_bytes = cursor.take(header_len)?;
    let body = cursor.take(body_len)?;
    cursor.finish()?;
    Ok(DecodedFrame {
        frame_type,
        identity,
        physical,
        stable_transaction_id,
        commit_seq,
        frame_index,
        fragment_count,
        kind,
        header_digest,
        root,
        leaf_or_final,
        header_bytes,
        body,
    })
}

/// A bounded fixed-slice writer shared by v1's allocating encoder and its allocation-free exact
/// sibling. Every field writer refuses a mismatched geometry instead of extending storage.
struct FixedEncoder<'a> {
    bytes: &'a mut [u8],
    position: usize,
}

impl<'a> FixedEncoder<'a> {
    fn new(bytes: &'a mut [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn bytes(&mut self, value: &[u8]) -> Result<(), EngineError> {
        let end = self
            .position
            .checked_add(value.len())
            .ok_or_else(|| durability("fixed encoder byte length overflow"))?;
        let target = self
            .bytes
            .get_mut(self.position..end)
            .ok_or_else(|| durability("fixed encoder buffer length mismatch"))?;
        target.copy_from_slice(value);
        self.position = end;
        Ok(())
    }

    fn u8(&mut self, value: u8) -> Result<(), EngineError> {
        self.bytes(&[value])
    }

    fn u16(&mut self, value: u16) -> Result<(), EngineError> {
        self.bytes(&value.to_le_bytes())
    }

    fn u32(&mut self, value: u32) -> Result<(), EngineError> {
        self.bytes(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> Result<(), EngineError> {
        self.bytes(&value.to_le_bytes())
    }

    fn finish(self) -> Result<(), EngineError> {
        if self.position != self.bytes.len() {
            return Err(durability("fixed encoder leaves unwritten bytes"));
        }
        Ok(())
    }
}

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], EngineError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| durability("decode length overflow"))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| durability("truncated encoding"))?;
        self.position = end;
        Ok(value)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], EngineError> {
        self.take(N)?
            .try_into()
            .map_err(|_| durability("invalid fixed-width field"))
    }

    fn u8(&mut self) -> Result<u8, EngineError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, EngineError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, EngineError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, EngineError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn finish(&self) -> Result<(), EngineError> {
        if self.position != self.bytes.len() {
            return Err(durability("trailing bytes"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> CanonicalDigest {
        [byte; 32]
    }

    #[test]
    fn canonical_fragment_kind_decodes_every_wire_value_and_rejects_unknown_values() {
        for (wire, expected) in [
            (1, CanonicalFragmentKind::RowMutation),
            (2, CanonicalFragmentKind::CatalogMutation),
            (3, CanonicalFragmentKind::TableReset),
            (4, CanonicalFragmentKind::TableRewrite),
            (5, CanonicalFragmentKind::SequenceValueTransition),
            (6, CanonicalFragmentKind::PrivateSequenceChild),
            (7, CanonicalFragmentKind::AllocatorLease),
            (8, CanonicalFragmentKind::TransactionClaimStatus),
        ] {
            assert_eq!(CanonicalFragmentKind::decode(wire).unwrap(), expected);
        }

        for wire in [0, 9, u16::MAX] {
            assert!(CanonicalFragmentKind::decode(wire).is_err(), "wire={wire}");
        }
    }

    fn identity() -> CanonicalIdentity {
        CanonicalIdentity {
            database_id: [1; 16],
            cluster_id: [2; 16],
            timeline_id: [3; 16],
            format_epoch: 7,
        }
    }

    fn header() -> CanonicalPreApplyHeader {
        CanonicalPreApplyHeader {
            identity: identity(),
            leader_epoch: 11,
            commit_seq: 19,
            stable_transaction_id: 23,
            request_digest: digest(4),
            isolation: CanonicalIsolation::ReadCommitted,
            flags: 5,
            catalog_before_epoch: 29,
            catalog_after_epoch: 30,
            catalog_before_digest: digest(6),
            catalog_after_digest: digest(7),
            operation_count: 3,
            table_block_count: 1,
            allocator_high_water: 31,
        }
    }

    fn physical() -> CanonicalPhysicalRange {
        CanonicalPhysicalRange {
            log_epoch: 11,
            lane_id: 2,
            segment_id: 41,
            first_frame_ordinal: 43,
        }
    }

    fn fragments() -> Vec<CanonicalFragment> {
        vec![
            CanonicalFragment {
                kind: CanonicalFragmentKind::CatalogMutation,
                body: b"typed-create-table".to_vec(),
            },
            CanonicalFragment {
                kind: CanonicalFragmentKind::RowMutation,
                body: b"typed-insert-row".to_vec(),
            },
            CanonicalFragment {
                kind: CanonicalFragmentKind::SequenceValueTransition,
                body: b"typed-sequence-state".to_vec(),
            },
        ]
    }

    fn outcome() -> CanonicalOutcome {
        CanonicalOutcome {
            kind: CanonicalOutcomeKind::CommitSuccess,
            affected_rows: 1,
            sqlstate: None,
            constraint_id: 0,
            target_digest: digest(8),
            returning_digest: digest(9),
        }
    }

    #[test]
    fn standalone_outcome_codec_is_exact_strict_and_byte_identical() {
        assert_eq!(CANONICAL_OUTCOME_BYTES, 92);
        let outcome = CanonicalOutcome {
            kind: CanonicalOutcomeKind::AbortError,
            affected_rows: 0x0102_0304_0506_0708,
            sqlstate: Some(*b"23505"),
            constraint_id: 0x1112_1314_1516_1718,
            target_digest: digest(0x29),
            returning_digest: digest(0x3a),
        };
        let mut bytes = [0xff; CANONICAL_OUTCOME_BYTES];
        encode_canonical_outcome_into_exact(&outcome, &mut bytes).unwrap();

        let mut expected = [0; CANONICAL_OUTCOME_BYTES];
        expected[0] = CanonicalOutcomeKind::AbortError as u8;
        expected[1] = 1;
        expected[4..12].copy_from_slice(&outcome.affected_rows.to_le_bytes());
        expected[12..20].copy_from_slice(&outcome.constraint_id.to_le_bytes());
        expected[20..25].copy_from_slice(b"23505");
        expected[28..60].copy_from_slice(&outcome.target_digest);
        expected[60..92].copy_from_slice(&outcome.returning_digest);
        assert_eq!(bytes, expected, "every outcome field owns its fixed bytes");

        let decoded = decode_canonical_outcome_exact(&bytes).unwrap();
        assert_eq!(decoded, outcome);
        let mut reencoded = [0; CANONICAL_OUTCOME_BYTES];
        encode_canonical_outcome_into_exact(&decoded, &mut reencoded).unwrap();
        assert_eq!(
            reencoded, bytes,
            "decode/reencode must preserve canonical bytes"
        );
        assert_eq!(outcome.encode().unwrap(), bytes);

        for retained in 0..CANONICAL_OUTCOME_BYTES {
            assert!(decode_canonical_outcome_exact(&bytes[..retained]).is_err());
        }
        let mut surplus = bytes.to_vec();
        surplus.push(0);
        assert!(decode_canonical_outcome_exact(&surplus).is_err());

        for reserved_offset in [2, 3, 25, 26, 27] {
            let mut sabotaged = bytes;
            sabotaged[reserved_offset] = 1;
            assert!(decode_canonical_outcome_exact(&sabotaged).is_err());
        }
        let mut invalid_kind = bytes;
        invalid_kind[0] = 0;
        assert!(decode_canonical_outcome_exact(&invalid_kind).is_err());
        let mut invalid_flag = bytes;
        invalid_flag[1] = 2;
        assert!(decode_canonical_outcome_exact(&invalid_flag).is_err());

        let mut absent_state = [0; CANONICAL_OUTCOME_BYTES];
        let mut successful_outcome = outcome.clone();
        successful_outcome.kind = CanonicalOutcomeKind::CommitSuccess;
        successful_outcome.sqlstate = None;
        encode_canonical_outcome_into_exact(&successful_outcome, &mut absent_state).unwrap();
        absent_state[20] = b'2';
        assert!(decode_canonical_outcome_exact(&absent_state).is_err());

        let mut malformed_abort = outcome.clone();
        malformed_abort.sqlstate = Some(*b"23-05");
        assert!(encode_canonical_outcome_into_exact(&malformed_abort, &mut bytes).is_err());
        let mut malformed_success = outcome;
        malformed_success.kind = CanonicalOutcomeKind::CommitSuccess;
        assert!(encode_canonical_outcome_into_exact(&malformed_success, &mut bytes).is_err());
    }

    #[test]
    fn canonical_fragmented_envelope_round_trips_with_explicit_mapping() {
        let encoded =
            encode_canonical_envelope(physical(), &header(), &fragments(), &outcome()).unwrap();
        assert_eq!(encoded.frames.len(), 4);
        let decoded = decode_canonical_envelope(&encoded.frames).unwrap();
        assert_eq!(decoded.physical, physical());
        assert_eq!(decoded.header, header());
        assert_eq!(decoded.fragments, fragments());
        assert_eq!(decoded.outcome, outcome());
        assert_eq!(decoded.ordered_fragment_root, encoded.ordered_fragment_root);
        assert_eq!(decoded.final_digest, encoded.final_digest);
    }

    #[test]
    fn logical_intent_outcome_bytes_exclude_physical_frames_and_record_packing() {
        let fragments = fragments();
        let outcome = outcome();
        let logical = canonical_logical_intent_outcome_bytes(&fragments, &outcome).unwrap();
        let expected = fragments
            .iter()
            .map(|fragment| 2u64 + fragment.body.len() as u64)
            .sum::<u64>()
            + outcome.encode().unwrap().len() as u64;
        assert_eq!(logical, expected);

        let first = encode_canonical_envelope(physical(), &header(), &fragments, &outcome).unwrap();
        let mut moved = physical();
        moved.lane_id = 99;
        moved.segment_id = 999;
        let second = encode_canonical_envelope(moved, &header(), &fragments, &outcome).unwrap();
        assert_eq!(
            canonical_logical_intent_outcome_bytes(&fragments, &outcome).unwrap(),
            logical
        );
        assert!(pack_canonical_record_payload(&first).unwrap().len() as u64 > logical);
        assert!(pack_canonical_record_payload(&second).unwrap().len() as u64 > logical);
    }

    fn assert_footprint_matches_real_framing(body_lengths: &[usize]) {
        let fragments = body_lengths
            .iter()
            .enumerate()
            .map(|(index, length)| CanonicalFragment {
                kind: CanonicalFragmentKind::RowMutation,
                body: vec![u8::try_from(index).expect("small fixture ordinal"); *length],
            })
            .collect::<Vec<_>>();
        let mut header = header();
        header.operation_count = u32::try_from(fragments.len()).expect("fixture count fits");
        let footprint = canonical_wal_footprint(
            &body_lengths
                .iter()
                .copied()
                .map(|length| u64::try_from(length).expect("fixture length fits"))
                .collect::<Vec<_>>(),
        )
        .expect("footprint accepts real encoder fixture");
        let encoded = encode_canonical_envelope(physical(), &header, &fragments, &outcome())
            .expect("real canonical envelope encodes");
        let fragment_frame_bytes = encoded.frames[..fragments.len()]
            .iter()
            .map(|frame| u64::try_from(frame.len()).expect("frame length fits"))
            .sum::<u64>();
        let marker_frame_bytes =
            u64::try_from(encoded.frames.last().expect("terminal marker frame").len())
                .expect("marker length fits");
        let packed = pack_canonical_record_payload(&encoded).expect("real record packs");
        let prepared = encoded
            .into_prepared_record(header.stable_transaction_id)
            .expect("real envelope seals its WAL record");
        let record = prepared.as_wal_record();
        assert_eq!(record.payload.as_ref(), packed.as_slice());
        let mut serialized = Vec::new();
        crate::encode_record_into(&mut serialized, record).expect("outer record serializes");

        assert_eq!(footprint.fragment_count as usize, fragments.len());
        assert_eq!(footprint.frame_count as usize, fragments.len() + 1);
        assert_eq!(
            footprint.fragment_body_bytes,
            body_lengths.iter().map(|length| *length as u64).sum()
        );
        assert_eq!(footprint.fragment_frame_bytes, fragment_frame_bytes);
        assert_eq!(footprint.marker_frame_bytes, marker_frame_bytes);
        assert_eq!(footprint.packed_record_bytes as usize, packed.len());
        assert_eq!(footprint.serialized_record_bytes as usize, serialized.len());
        assert_eq!(footprint.marker_packed_bytes, marker_frame_bytes + 4);
    }

    #[test]
    fn canonical_wal_footprint_matches_one_two_many_and_max_real_envelopes() {
        assert_footprint_matches_real_framing(&[17]);
        assert_footprint_matches_real_framing(&[17, 100]);
        assert_footprint_matches_real_framing(&[3, 19, 257, 4_096, 71]);
        assert_footprint_matches_real_framing(&[MAX_FRAGMENT_BYTES]);
    }

    #[test]
    fn canonical_wal_footprint_rejects_real_encoder_bound_violations() {
        assert!(canonical_wal_footprint(&[]).is_err());
        assert!(canonical_wal_footprint(&[MAX_FRAGMENT_BYTES as u64 + 1]).is_err());
        assert!(canonical_wal_footprint(&[MAX_FRAGMENT_BYTES as u64; 4]).is_err());
        // A zero-body fragment fanout can stay below the raw-body bound while overflowing the
        // packed record's per-frame framing budget.
        assert!(canonical_wal_footprint(&vec![0; 243_421]).is_err());
    }

    #[test]
    fn canonical_wal_footprint_keeps_the_preapply_envelope_limit_inclusive() {
        let body_limit = canonical_fragment_body_limit();
        let exact = [body_limit, body_limit, body_limit, body_limit - 240];
        let footprint = canonical_wal_footprint(&exact).expect("64 MiB pre-apply fits exactly");
        assert_eq!(footprint.preapply_bytes, 64 * 1024 * 1024);

        let one_byte_over = [body_limit, body_limit, body_limit, body_limit - 239];
        assert!(canonical_wal_footprint(&one_byte_over).is_err());
    }

    #[test]
    fn canonical_wal_footprint_keeps_the_terminal_marker_frame_limit_inclusive() {
        let exact_marker_body = MAX_ENVELOPE_BYTES - FRAME_FIXED_BYTES - FRAME_DIGEST_BYTES;
        let exact = canonical_wal_footprint_by_index_with_marker(1, |_| Ok(0), exact_marker_body)
            .expect("terminal marker frame fits exactly");
        assert_eq!(exact.marker_frame_bytes, MAX_ENVELOPE_BYTES as u64);

        assert!(
            canonical_wal_footprint_by_index_with_marker(1, |_| Ok(0), exact_marker_body + 1,)
                .is_err()
        );
    }

    #[test]
    fn canonical_record_container_round_trips_and_rejects_every_truncation() {
        let encoded =
            encode_canonical_envelope(physical(), &header(), &fragments(), &outcome()).unwrap();
        let payload = pack_canonical_record_payload(&encoded).unwrap();
        let decoded = decode_canonical_record_payload(&payload)
            .unwrap()
            .expect("canonical payload");
        assert_eq!(decoded.header, header());
        assert_eq!(decoded.fragments, fragments());
        assert!(decode_canonical_record_payload(b"legacy SQL")
            .unwrap()
            .is_none());
        for retained in RECORD_MAGIC.len()..payload.len() {
            assert!(decode_canonical_record_payload(&payload[..retained]).is_err());
        }
    }

    #[test]
    fn every_fragment_and_marker_corruption_fails_closed() {
        let encoded =
            encode_canonical_envelope(physical(), &header(), &fragments(), &outcome()).unwrap();
        for frame_index in 0..encoded.frames.len() {
            let mut corrupted = encoded.frames.clone();
            let byte_index = FRAME_FIXED_BYTES + usize::from(frame_index == 0);
            corrupted[frame_index][byte_index] ^= 0x5a;
            assert!(decode_canonical_envelope(&corrupted).is_err());
        }
        for retained in 0..encoded.frames.len() {
            assert!(decode_canonical_envelope(&encoded.frames[..retained]).is_err());
        }
    }

    #[test]
    fn duplicate_reorder_foreign_lane_and_terminal_tamper_fail_closed() {
        let encoded =
            encode_canonical_envelope(physical(), &header(), &fragments(), &outcome()).unwrap();
        let mut reordered = encoded.frames.clone();
        reordered.swap(1, 2);
        assert!(decode_canonical_envelope(&reordered).is_err());
        let mut duplicate = encoded.frames.clone();
        duplicate[2] = duplicate[1].clone();
        assert!(decode_canonical_envelope(&duplicate).is_err());

        let other_physical = CanonicalPhysicalRange {
            lane_id: 9,
            ..physical()
        };
        let foreign =
            encode_canonical_envelope(other_physical, &header(), &fragments(), &outcome()).unwrap();
        let mut mixed = encoded.frames.clone();
        mixed[1] = foreign.frames[1].clone();
        assert!(decode_canonical_envelope(&mixed).is_err());

        let mut tampered = encoded.frames;
        let marker = tampered.last_mut().unwrap();
        let outcome_offset = FRAME_FIXED_BYTES;
        marker[outcome_offset] = CanonicalOutcomeKind::CommitNoOp as u8;
        let digest_offset = marker.len() - FRAME_DIGEST_BYTES;
        let frame_digest = digest_parts(FRAME_DOMAIN, &[&marker[..digest_offset]]);
        marker[digest_offset..].copy_from_slice(&frame_digest);
        assert!(decode_canonical_envelope(&tampered).is_err());
    }

    #[test]
    fn genesis_infinity_and_physical_overflow_are_rejected_before_encoding() {
        for invalid_seq in [0, u64::MAX] {
            let mut invalid = header();
            invalid.commit_seq = invalid_seq;
            assert!(
                encode_canonical_envelope(physical(), &invalid, &fragments(), &outcome()).is_err()
            );
        }
        let mut last = header();
        last.commit_seq = u64::MAX - 1;
        assert!(encode_canonical_envelope(physical(), &last, &fragments(), &outcome()).is_ok());
        let overflow = CanonicalPhysicalRange {
            first_frame_ordinal: u64::MAX - 1,
            ..physical()
        };
        assert!(encode_canonical_envelope(overflow, &header(), &fragments(), &outcome()).is_err());
    }

    #[test]
    fn abort_requires_sqlstate_and_success_refuses_one() {
        let mut abort = outcome();
        abort.kind = CanonicalOutcomeKind::AbortError;
        assert!(encode_canonical_envelope(physical(), &header(), &fragments(), &abort).is_err());
        abort.sqlstate = Some(*b"23505");
        assert!(encode_canonical_envelope(physical(), &header(), &fragments(), &abort).is_ok());
        abort.kind = CanonicalOutcomeKind::CommitNoOp;
        assert!(encode_canonical_envelope(physical(), &header(), &fragments(), &abort).is_err());
    }
}
