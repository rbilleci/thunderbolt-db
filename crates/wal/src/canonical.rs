//! Canonical ADR-014 WAL transaction envelope.
//!
//! The storage backends remain responsible for physical frame CRCs and durable prefixes. This
//! layer supplies the logical authority inside those frames: one immutable outcome-free header,
//! non-circular fragment leaves/root, one terminal commit/no-op/abort marker, and an explicit
//! lane-local physical to global logical mapping. A verifier accepts only the complete ordered
//! frame range; no fragment can independently authorize apply or publication.

use gpu_db_types::EngineError;
use sha2::{Digest, Sha256};

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
const MAX_FRAGMENT_BYTES: usize = 16 * 1024 * 1024;
const MAX_ENVELOPE_BYTES: usize = 64 * 1024 * 1024;

const PREAPPLY_DOMAIN: &[u8] = b"gpu-db/adr014/preapply/v1";
const FRAGMENT_LEAF_DOMAIN: &[u8] = b"gpu-db/adr014/fragment-leaf/v1";
const FRAGMENT_ROOT_DOMAIN: &[u8] = b"gpu-db/adr014/fragment-root/v1";
const FINAL_DOMAIN: &[u8] = b"gpu-db/adr014/final-outcome/v1";
const FRAME_DOMAIN: &[u8] = b"gpu-db/adr014/physical-frame/v1";

fn durability(message: impl Into<String>) -> EngineError {
    EngineError::Durability(format!("canonical WAL: {}", message.into()))
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
        self.validate()?;
        let mut out = Vec::with_capacity(236);
        out.extend_from_slice(HEADER_MAGIC);
        put_u16(&mut out, FORMAT_VERSION);
        put_u16(&mut out, SEMANTICS_VERSION);
        put_u16(&mut out, MIN_READER_VERSION);
        put_u16(&mut out, MAX_READER_VERSION);
        out.extend_from_slice(&self.identity.database_id);
        out.extend_from_slice(&self.identity.cluster_id);
        out.extend_from_slice(&self.identity.timeline_id);
        put_u64(&mut out, self.identity.format_epoch);
        put_u64(&mut out, self.leader_epoch);
        put_u64(&mut out, self.commit_seq);
        put_u64(&mut out, self.stable_transaction_id);
        out.extend_from_slice(&self.request_digest);
        out.push(self.isolation as u8);
        out.extend_from_slice(&[0; 3]);
        put_u32(&mut out, self.flags);
        put_u64(&mut out, self.catalog_before_epoch);
        put_u64(&mut out, self.catalog_after_epoch);
        out.extend_from_slice(&self.catalog_before_digest);
        out.extend_from_slice(&self.catalog_after_digest);
        put_u32(&mut out, self.operation_count);
        put_u32(&mut out, self.table_block_count);
        put_u64(&mut out, self.allocator_high_water);
        Ok(out)
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
    fn decode(value: u16) -> Result<Self, EngineError> {
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
    fn encode(&self) -> Result<Vec<u8>, EngineError> {
        match (self.kind, self.sqlstate) {
            (CanonicalOutcomeKind::AbortError, Some(state))
                if state.iter().all(u8::is_ascii_alphanumeric) => {}
            (CanonicalOutcomeKind::AbortError, _) => {
                return Err(durability("AbortError requires an ASCII SQLSTATE"));
            }
            (_, None) => {}
            (_, Some(_)) => return Err(durability("successful outcome must not carry SQLSTATE")),
        }
        let mut out = Vec::with_capacity(88);
        out.push(self.kind as u8);
        out.push(u8::from(self.sqlstate.is_some()));
        out.extend_from_slice(&[0; 2]);
        put_u64(&mut out, self.affected_rows);
        put_u64(&mut out, self.constraint_id);
        out.extend_from_slice(&self.sqlstate.unwrap_or([0; 5]));
        out.extend_from_slice(&[0; 3]);
        out.extend_from_slice(&self.target_digest);
        out.extend_from_slice(&self.returning_digest);
        Ok(out)
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
        outcome.encode()?;
        Ok(outcome)
    }
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

    pub(crate) fn into_parts(self) -> (crate::WalRecord, CanonicalCatalogTail) {
        (self.record, self.tail)
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
        })
    }
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
    if fragments.is_empty() || fragments.len() > u32::MAX as usize - 1 {
        return Err(durability("fragment count must be in 1..u32::MAX"));
    }
    if header.operation_count as usize != fragments.len() {
        return Err(durability(format!(
            "header operation count {} does not match {} fragments",
            header.operation_count,
            fragments.len()
        )));
    }
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
    let outcome_bytes = outcome.encode()?;
    let final_digest = digest_parts(FINAL_DOMAIN, &[&header_bytes, &root, &outcome_bytes]);
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
        &outcome_bytes,
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
    let header_len = u32::try_from(header_bytes.len())
        .map_err(|_| durability("pre-apply header length overflow"))?;
    let body_len = u32::try_from(body.len()).map_err(|_| durability("frame body overflow"))?;
    let mut frame = Vec::with_capacity(
        FRAME_FIXED_BYTES + header_bytes.len() + body.len() + FRAME_DIGEST_BYTES,
    );
    frame.extend_from_slice(FRAME_MAGIC);
    put_u16(&mut frame, FORMAT_VERSION);
    put_u16(&mut frame, SEMANTICS_VERSION);
    frame.push(frame_type);
    frame.extend_from_slice(&[0; 3]);
    frame.extend_from_slice(&header.identity.database_id);
    frame.extend_from_slice(&header.identity.cluster_id);
    frame.extend_from_slice(&header.identity.timeline_id);
    put_u64(&mut frame, header.identity.format_epoch);
    put_u64(&mut frame, physical.log_epoch);
    put_u32(&mut frame, physical.lane_id);
    put_u32(&mut frame, 0);
    put_u64(&mut frame, physical.segment_id);
    put_u64(&mut frame, physical.first_frame_ordinal);
    put_u64(&mut frame, header.stable_transaction_id);
    put_u64(&mut frame, header.commit_seq);
    put_u32(&mut frame, frame_index);
    put_u32(&mut frame, fragment_count);
    put_u16(&mut frame, kind);
    put_u16(&mut frame, 0);
    put_u32(&mut frame, header_len);
    put_u32(&mut frame, body_len);
    frame.extend_from_slice(&header_digest);
    frame.extend_from_slice(&root);
    frame.extend_from_slice(&leaf_or_final);
    debug_assert_eq!(frame.len(), FRAME_FIXED_BYTES);
    frame.extend_from_slice(header_bytes);
    frame.extend_from_slice(body);
    let frame_digest = digest_parts(FRAME_DOMAIN, &[&frame]);
    frame.extend_from_slice(&frame_digest);
    Ok(frame)
}

pub fn decode_canonical_envelope(frames: &[Vec<u8>]) -> Result<CanonicalEnvelope, EngineError> {
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
    let outcome = CanonicalOutcome::decode(marker.body)?;
    let outcome_bytes = outcome.encode()?;
    let final_digest = digest_parts(
        FINAL_DOMAIN,
        &[&header_bytes, &expected_root, &outcome_bytes],
    );
    if marker.leaf_or_final != final_digest {
        return Err(durability("terminal outcome digest mismatch"));
    }
    Ok(CanonicalEnvelope {
        physical,
        header,
        fragments,
        outcome,
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
