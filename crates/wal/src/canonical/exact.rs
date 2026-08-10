//! Allocation-free exact-buffer canonical WAL encoding.
//!
//! The v1 allocating encoder remains the compatibility reference. This leaf shares its header,
//! outcome, frame, digest, and outer-record layout, but walks the fragment slice repeatedly
//! instead of retaining canonical bodies, leaves, frames, or a packed record during preparation.

use super::*;
use sha2::{Digest, Sha256};

/// Codec-5 emits at most four aggregate chunks followed by STATUS2. Keeping those leaf digests
/// inline lets the borrowed exact path reuse its ordered-root traversal during frame emission
/// without adding pre-WAL scratch allocation. Larger compatibility envelopes retain the generic
/// allocation-free repeated traversal.
const INLINE_FRAGMENT_LEAVES: usize = 5;

/// The immutable evidence produced while writing caller-owned exact canonical buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactCanonicalRecordEncoding {
    /// Exact packed-payload and outer-record geometry verified before either output is written.
    pub footprint: CanonicalWalFootprint,
    pub ordered_fragment_root: CanonicalDigest,
    pub final_digest: CanonicalDigest,
    /// Build-only wall attribution for digest closure, packed-frame fill, storage checksum,
    /// serialized payload copy, and final immutable seal respectively.
    #[cfg(feature = "probe-timing")]
    pub probe_timing_nanos: [u64; 5],
}

/// Allocation-free fragment view for exact-buffer callers that already own canonical bodies.
///
/// Unlike [`CanonicalFragment`], this carries no owned body buffer and cannot duplicate the
/// engine format owner's bytes merely to enter the outer WAL encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalFragmentRef<'a> {
    pub kind: CanonicalFragmentKind,
    pub body: &'a [u8],
}

/// Measure the exact v1 geometry from fragment-body lengths and validate the immutable envelope
/// metadata. This performs no output allocation and is suitable for reserving caller-owned
/// payload and serialized-record slices before fragment bodies are materialized into a record.
pub fn measure_canonical_exact_buffers(
    physical: CanonicalPhysicalRange,
    header: &CanonicalPreApplyHeader,
    fragment_body_lengths: &[u64],
    outcome: &CanonicalOutcome,
) -> Result<CanonicalWalFootprint, EngineError> {
    measure_exact_with_lengths(
        physical,
        header,
        fragment_body_lengths.len(),
        |index| Ok(fragment_body_lengths[index]),
        outcome,
    )
}

/// Measure exact v1 geometry directly from already-owned canonical fragments without allocating
/// a length sidecar.  Typed pre-WAL preparation uses this so the only persistent allocations are
/// its packed and serialized authority buffers.
pub fn measure_canonical_exact_buffers_from_fragments(
    physical: CanonicalPhysicalRange,
    header: &CanonicalPreApplyHeader,
    fragments: &[CanonicalFragment],
    outcome: &CanonicalOutcome,
) -> Result<CanonicalWalFootprint, EngineError> {
    measure_exact_with_lengths(
        physical,
        header,
        fragments.len(),
        |index| {
            u64::try_from(fragments[index].body.len())
                .map_err(|_| durability("fragment byte length exceeds u64 framing"))
        },
        outcome,
    )
}

/// Encode a canonical envelope directly into caller-owned exact output slices.
///
/// `packed_payload` must be exactly [`CanonicalWalFootprint::packed_record_bytes`] and
/// `serialized_record` exactly [`CanonicalWalFootprint::serialized_record_bytes`] from the same
/// immutable inputs. Both lengths are checked before any output byte is written. On success the
/// packed slice is the v1 `WalRecord::payload`; the serialized slice is the v1 outer WAL record
/// for `txn_id` and that payload.
pub fn encode_canonical_record_exact_into(
    txn_id: gpu_db_types::TxnId,
    physical: CanonicalPhysicalRange,
    header: &CanonicalPreApplyHeader,
    fragments: &[CanonicalFragment],
    outcome: &CanonicalOutcome,
    packed_payload: &mut [u8],
    serialized_record: &mut [u8],
) -> Result<ExactCanonicalRecordEncoding, EngineError> {
    encode_canonical_record_exact_from_source(
        txn_id,
        physical,
        header,
        fragments,
        outcome,
        packed_payload,
        serialized_record,
    )
}

/// Encode directly from borrowed engine-owned fragment bodies without constructing legacy
/// owned-body carriers.
pub fn encode_canonical_record_exact_from_borrowed(
    txn_id: gpu_db_types::TxnId,
    physical: CanonicalPhysicalRange,
    header: &CanonicalPreApplyHeader,
    fragments: &[CanonicalFragmentRef<'_>],
    outcome: &CanonicalOutcome,
    packed_payload: &mut [u8],
    serialized_record: &mut [u8],
) -> Result<ExactCanonicalRecordEncoding, EngineError> {
    encode_canonical_record_exact_from_source(
        txn_id,
        physical,
        header,
        fragments,
        outcome,
        packed_payload,
        serialized_record,
    )
}

fn encode_canonical_record_exact_from_source<F: CanonicalFragmentSource>(
    txn_id: gpu_db_types::TxnId,
    physical: CanonicalPhysicalRange,
    header: &CanonicalPreApplyHeader,
    fragments: &[F],
    outcome: &CanonicalOutcome,
    packed_payload: &mut [u8],
    serialized_record: &mut [u8],
) -> Result<ExactCanonicalRecordEncoding, EngineError> {
    let footprint = measure_exact_with_lengths(
        physical,
        header,
        fragments.len(),
        |index| {
            u64::try_from(fragments[index].body().len())
                .map_err(|_| durability("fragment byte length exceeds u64 framing"))
        },
        outcome,
    )?;
    require_exact_buffer(
        packed_payload.len(),
        footprint.packed_record_bytes,
        "packed canonical record",
    )?;
    require_exact_buffer(
        serialized_record.len(),
        footprint.serialized_record_bytes,
        "serialized canonical record",
    )?;

    // `measure_exact_with_lengths` fully validated the immutable inputs and both output sizes.
    // The following field writes therefore have no failure path after caller-owned buffers are
    // touched; internal assertions defend the shared layout law against future drift.
    #[cfg(feature = "probe-timing")]
    let probe_digest_started = std::time::Instant::now();
    let mut header_bytes = [0_u8; PREAPPLY_HEADER_BYTES];
    header
        .encode_into(&mut header_bytes)
        .expect("validated exact header encodes into its fixed v1 layout");
    let header_digest = digest_parts(PREAPPLY_DOMAIN, &[&header_bytes]);
    let mut inline_leaves = [[0_u8; 32]; INLINE_FRAGMENT_LEAVES];
    let cached_leaves = if fragments.len() <= inline_leaves.len() {
        for (index, fragment) in fragments.iter().enumerate() {
            inline_leaves[index] = fragment_leaf(header_digest, index, fragment);
        }
        Some(&inline_leaves[..fragments.len()])
    } else {
        None
    };
    let root = match cached_leaves {
        Some(leaves) => ordered_fragment_root_from_leaves(fragments, leaves),
        None => ordered_fragment_root(header_digest, fragments),
    };
    let mut outcome_bytes = [0_u8; CANONICAL_OUTCOME_BYTES];
    encode_canonical_outcome_into_exact(outcome, &mut outcome_bytes)
        .expect("validated exact outcome encodes into its fixed v1 layout");
    let final_digest = digest_parts(FINAL_DOMAIN, &[&header_bytes, &root, &outcome_bytes]);
    #[cfg(feature = "probe-timing")]
    let probe_digest_nanos = probe_digest_started.elapsed().as_nanos() as u64;

    #[cfg(feature = "probe-timing")]
    let probe_packed_started = std::time::Instant::now();
    write_packed_payload(
        packed_payload,
        physical,
        header,
        fragments,
        &header_bytes,
        &outcome_bytes,
        header_digest,
        root,
        final_digest,
        cached_leaves,
    );
    #[cfg(feature = "probe-timing")]
    let probe_packed_nanos = probe_packed_started.elapsed().as_nanos() as u64;
    let _serialized_probe = write_serialized_record(serialized_record, txn_id, packed_payload);
    Ok(ExactCanonicalRecordEncoding {
        footprint,
        ordered_fragment_root: root,
        final_digest,
        #[cfg(feature = "probe-timing")]
        probe_timing_nanos: [
            probe_digest_nanos,
            probe_packed_nanos,
            _serialized_probe[0],
            _serialized_probe[1],
            0,
        ],
    })
}

fn measure_exact_with_lengths(
    physical: CanonicalPhysicalRange,
    header: &CanonicalPreApplyHeader,
    fragment_count: usize,
    body_len: impl FnMut(usize) -> Result<u64, EngineError>,
    outcome: &CanonicalOutcome,
) -> Result<CanonicalWalFootprint, EngineError> {
    if header.operation_count as usize != fragment_count {
        return Err(durability(format!(
            "header operation count {} does not match {fragment_count} fragments",
            header.operation_count,
        )));
    }
    let footprint = canonical_wal_footprint_by_index(fragment_count, body_len)?;
    physical.validate(footprint.frame_count)?;
    header.validate()?;
    outcome.validate()?;
    Ok(footprint)
}

fn require_exact_buffer(actual: usize, expected: u64, label: &str) -> Result<(), EngineError> {
    let expected = usize::try_from(expected)
        .map_err(|_| durability(format!("{label} length exceeds addressable memory")))?;
    if actual != expected {
        return Err(durability(format!(
            "{label} buffer must be exactly {expected} bytes, got {actual}"
        )));
    }
    Ok(())
}

fn ordered_fragment_root<F: CanonicalFragmentSource>(
    header_digest: CanonicalDigest,
    fragments: &[F],
) -> CanonicalDigest {
    let root_material_len = fragments
        .len()
        .checked_mul(44)
        .and_then(|bytes| bytes.checked_add(4))
        .expect("validated canonical frame count bounds root material");
    let mut hash = begin_digest(FRAGMENT_ROOT_DOMAIN, root_material_len);
    hash.update(
        u32::try_from(fragments.len())
            .expect("validated canonical frame count fits u32")
            .to_le_bytes(),
    );
    for (index, fragment) in fragments.iter().enumerate() {
        let leaf = fragment_leaf(header_digest, index, fragment);
        let canonical_len = fragment
            .body()
            .len()
            .checked_add(std::mem::size_of::<u16>())
            .expect("validated fragment body length leaves kind headroom");
        hash.update(
            u32::try_from(index)
                .expect("validated canonical frame count fits u32")
                .to_le_bytes(),
        );
        hash.update(
            u64::try_from(canonical_len)
                .expect("validated canonical length fits u64")
                .to_le_bytes(),
        );
        hash.update(leaf);
    }
    hash.finalize().into()
}

fn ordered_fragment_root_from_leaves<F: CanonicalFragmentSource>(
    fragments: &[F],
    leaves: &[CanonicalDigest],
) -> CanonicalDigest {
    debug_assert_eq!(fragments.len(), leaves.len());
    let root_material_len = leaves
        .len()
        .checked_mul(44)
        .and_then(|bytes| bytes.checked_add(4))
        .expect("validated canonical frame count bounds root material");
    let mut hash = begin_digest(FRAGMENT_ROOT_DOMAIN, root_material_len);
    hash.update(
        u32::try_from(leaves.len())
            .expect("validated canonical frame count fits u32")
            .to_le_bytes(),
    );
    for (index, (fragment, leaf)) in fragments.iter().zip(leaves).enumerate() {
        let canonical_len = fragment
            .body()
            .len()
            .checked_add(std::mem::size_of::<u16>())
            .expect("validated fragment body length leaves kind headroom");
        hash.update(
            u32::try_from(index)
                .expect("validated canonical frame count fits u32")
                .to_le_bytes(),
        );
        hash.update(
            u64::try_from(canonical_len)
                .expect("validated canonical length fits u64")
                .to_le_bytes(),
        );
        hash.update(leaf);
    }
    hash.finalize().into()
}

fn fragment_leaf(
    header_digest: CanonicalDigest,
    index: usize,
    fragment: &impl CanonicalFragmentSource,
) -> CanonicalDigest {
    let canonical_len = fragment
        .body()
        .len()
        .checked_add(std::mem::size_of::<u16>())
        .expect("validated fragment body length leaves kind headroom");
    let mut hash = begin_digest(FRAGMENT_LEAF_DOMAIN, 0);
    append_part(&mut hash, &header_digest);
    append_part(
        &mut hash,
        &u32::try_from(index)
            .expect("validated canonical frame count fits u32")
            .to_le_bytes(),
    );
    append_part(
        &mut hash,
        &u64::try_from(canonical_len)
            .expect("validated canonical length fits u64")
            .to_le_bytes(),
    );
    hash.update(
        u64::try_from(canonical_len)
            .expect("validated canonical length fits u64")
            .to_le_bytes(),
    );
    hash.update((fragment.kind() as u16).to_le_bytes());
    hash.update(fragment.body());
    hash.finalize().into()
}

fn begin_digest(domain: &[u8], first_part_len: usize) -> Sha256 {
    let mut hash = Sha256::new();
    hash.update(
        u64::try_from(domain.len())
            .expect("static canonical digest domain fits u64")
            .to_le_bytes(),
    );
    hash.update(domain);
    if first_part_len != 0 {
        hash.update(
            u64::try_from(first_part_len)
                .expect("validated canonical digest material fits u64")
                .to_le_bytes(),
        );
    }
    hash
}

fn append_part(hash: &mut Sha256, part: &[u8]) {
    hash.update(
        u64::try_from(part.len())
            .expect("validated canonical digest part fits u64")
            .to_le_bytes(),
    );
    hash.update(part);
}

#[allow(clippy::too_many_arguments)]
fn write_packed_payload(
    packed_payload: &mut [u8],
    physical: CanonicalPhysicalRange,
    header: &CanonicalPreApplyHeader,
    fragments: &[impl CanonicalFragmentSource],
    header_bytes: &[u8; PREAPPLY_HEADER_BYTES],
    outcome_bytes: &[u8; CANONICAL_OUTCOME_BYTES],
    header_digest: CanonicalDigest,
    root: CanonicalDigest,
    final_digest: CanonicalDigest,
    cached_leaves: Option<&[CanonicalDigest]>,
) {
    packed_payload[..RECORD_MAGIC.len()].copy_from_slice(RECORD_MAGIC);
    let mut offset = RECORD_MAGIC.len();
    put_u32_slice(
        &mut packed_payload[offset..offset + PACKED_FRAME_LENGTH_BYTES],
        u32::try_from(fragments.len() + 1).expect("validated canonical frame count fits u32"),
    );
    offset += PACKED_FRAME_LENGTH_BYTES;
    for (index, fragment) in fragments.iter().enumerate() {
        let frame_len = canonical_frame_encoded_len(
            usize::from(index == 0) * PREAPPLY_HEADER_BYTES,
            fragment.body().len(),
        )
        .expect("validated exact fragment frame length");
        put_u32_slice(
            &mut packed_payload[offset..offset + PACKED_FRAME_LENGTH_BYTES],
            u32::try_from(frame_len).expect("validated exact frame length fits u32"),
        );
        offset += PACKED_FRAME_LENGTH_BYTES;
        let frame = &mut packed_payload[offset..offset + frame_len];
        encode_frame_into(
            frame,
            FRAME_FRAGMENT,
            physical,
            header,
            u32::try_from(index).expect("validated canonical frame count fits u32"),
            u32::try_from(fragments.len()).expect("validated canonical frame count fits u32"),
            fragment.kind() as u16,
            if index == 0 { header_bytes } else { &[] },
            fragment.body(),
            header_digest,
            root,
            cached_leaves.map_or_else(
                || fragment_leaf(header_digest, index, fragment),
                |leaves| leaves[index],
            ),
        )
        .expect("validated exact fragment frame encodes");
        offset += frame_len;
    }
    let marker_len = canonical_frame_encoded_len(0, CANONICAL_OUTCOME_BYTES)
        .expect("validated exact terminal frame length");
    put_u32_slice(
        &mut packed_payload[offset..offset + PACKED_FRAME_LENGTH_BYTES],
        u32::try_from(marker_len).expect("validated terminal frame length fits u32"),
    );
    offset += PACKED_FRAME_LENGTH_BYTES;
    let marker = &mut packed_payload[offset..offset + marker_len];
    encode_frame_into(
        marker,
        FRAME_MARKER,
        physical,
        header,
        u32::try_from(fragments.len()).expect("validated canonical frame count fits u32"),
        u32::try_from(fragments.len()).expect("validated canonical frame count fits u32"),
        0,
        &[],
        outcome_bytes,
        header_digest,
        root,
        final_digest,
    )
    .expect("validated exact terminal frame encodes");
    offset += marker_len;
    assert_eq!(
        offset,
        packed_payload.len(),
        "measured packed geometry must consume its exact caller buffer"
    );
}

trait CanonicalFragmentSource {
    fn kind(&self) -> CanonicalFragmentKind;
    fn body(&self) -> &[u8];
}

impl CanonicalFragmentSource for CanonicalFragment {
    fn kind(&self) -> CanonicalFragmentKind {
        self.kind
    }

    fn body(&self) -> &[u8] {
        &self.body
    }
}

impl CanonicalFragmentSource for CanonicalFragmentRef<'_> {
    fn kind(&self) -> CanonicalFragmentKind {
        self.kind
    }

    fn body(&self) -> &[u8] {
        self.body
    }
}

#[cfg(feature = "probe-timing")]
fn write_serialized_record(
    serialized_record: &mut [u8],
    txn_id: gpu_db_types::TxnId,
    packed_payload: &[u8],
) -> [u64; 2] {
    let payload_len =
        u64::try_from(packed_payload.len()).expect("measured canonical payload fits u64 framing");
    let checksum_started = std::time::Instant::now();
    let checksum = crate::wal_record_checksum(txn_id, payload_len, packed_payload);
    let checksum_nanos = checksum_started.elapsed().as_nanos() as u64;
    let copy_started = std::time::Instant::now();
    serialized_record[..8].copy_from_slice(&txn_id.to_le_bytes());
    serialized_record[8..16].copy_from_slice(&payload_len.to_le_bytes());
    serialized_record[16..crate::WAL_RECORD_HEADER_LEN].copy_from_slice(&checksum.to_le_bytes());
    serialized_record[crate::WAL_RECORD_HEADER_LEN..].copy_from_slice(packed_payload);
    [checksum_nanos, copy_started.elapsed().as_nanos() as u64]
}

#[cfg(not(feature = "probe-timing"))]
fn write_serialized_record(
    serialized_record: &mut [u8],
    txn_id: gpu_db_types::TxnId,
    packed_payload: &[u8],
) {
    let payload_len =
        u64::try_from(packed_payload.len()).expect("measured canonical payload fits u64 framing");
    let checksum = crate::wal_record_checksum(txn_id, payload_len, packed_payload);
    serialized_record[..8].copy_from_slice(&txn_id.to_le_bytes());
    serialized_record[8..16].copy_from_slice(&payload_len.to_le_bytes());
    serialized_record[16..crate::WAL_RECORD_HEADER_LEN].copy_from_slice(&checksum.to_le_bytes());
    serialized_record[crate::WAL_RECORD_HEADER_LEN..].copy_from_slice(packed_payload);
}

fn put_u32_slice(out: &mut [u8], value: u32) {
    assert_eq!(out.len(), PACKED_FRAME_LENGTH_BYTES);
    out.copy_from_slice(&value.to_le_bytes());
}
