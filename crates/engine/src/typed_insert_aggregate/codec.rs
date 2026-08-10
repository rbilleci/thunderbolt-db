//! Exact-buffer aggregate stream and canonical chunk encoding.

use super::*;
use crate::EngineError;
use sha2::{Digest, Sha256};

const SECTION_DOMAIN: &[u8] = b"gpu-db/write001/aggregate-section/v1";
const ROOT_DOMAIN: &[u8] = b"gpu-db/write001/aggregate-root/v1";
const RESPONSE_ROOT_DOMAIN: &[u8] = b"gpu-db/write001/response-root/v1";

/// One already-canonical section payload. Entry semantics stay with the section-specific
/// encoders; this owner fixes ordering, count, and exact bytes for the aggregate traversal.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TypedInsertAggregateSectionView<'a> {
    pub(crate) entry_count: u32,
    pub(crate) payload: &'a [u8],
}

/// Borrowed aggregate semantics used by the allocation-free measure/encode passes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TypedInsertAggregateView<'a> {
    pub(crate) semantics: TypedInsertAggregateSemantics,
    pub(crate) flags: u32,
    pub(crate) outer_flags: u32,
    pub(crate) stable_transaction_id: u64,
    pub(crate) statement_count: u32,
    pub(crate) insert_statement_count: u32,
    pub(crate) original_inserted_row_count: u64,
    pub(crate) final_row_transition_count: u64,
    pub(crate) allocator_before: u64,
    pub(crate) allocator_high_water: u64,
    pub(crate) table_block_count: u32,
    pub(crate) sections: [TypedInsertAggregateSectionView<'a>; AGGREGATE_SECTION_COUNT],
}

impl TypedInsertAggregateView<'_> {
    pub(crate) fn measure(
        &self,
    ) -> Result<TypedInsertAggregateLayout, TypedInsertAggregateLayoutError> {
        let mut sections = [AggregateSectionMeasure {
            entry_count: 0,
            payload_bytes: 0,
        }; AGGREGATE_SECTION_COUNT];
        for (target, source) in sections.iter_mut().zip(self.sections.iter()) {
            *target = AggregateSectionMeasure {
                entry_count: source.entry_count,
                payload_bytes: u64::try_from(source.payload.len())
                    .map_err(|_| TypedInsertAggregateLayoutError::Overflow("section payload"))?,
            };
        }
        TypedInsertAggregateMeasure {
            semantics: self.semantics,
            flags: self.flags,
            outer_flags: self.outer_flags,
            stable_transaction_id: self.stable_transaction_id,
            statement_count: self.statement_count,
            insert_statement_count: self.insert_statement_count,
            original_inserted_row_count: self.original_inserted_row_count,
            final_row_transition_count: self.final_row_transition_count,
            allocator_before: self.allocator_before,
            allocator_high_water: self.allocator_high_water,
            table_block_count: self.table_block_count,
            sections,
        }
        .measure()
    }
}

/// Exact fragment-body owners. The fixed array itself is inline; every live entry is one charged
/// boxed allocation, and unused entries must remain `None`.
pub(crate) struct ReservedTypedInsertAggregateBodyBuffers {
    layout: TypedInsertAggregateLayout,
    bodies: [Option<Box<[u8]>>; AGGREGATE_MAX_CHUNKS + 1],
}

impl ReservedTypedInsertAggregateBodyBuffers {
    pub(crate) fn layout(&self) -> &TypedInsertAggregateLayout {
        &self.layout
    }
}

/// Fully encoded aggregate chunk and STATUS2 owners. This type cannot be constructed until every
/// exact body has been populated and the repeated aggregate-root closure has been verified.
pub(crate) struct EncodedTypedInsertAggregateBodies {
    layout: TypedInsertAggregateLayout,
    aggregate_root: gpu_db_wal::CanonicalDigest,
    section_roots: [gpu_db_wal::CanonicalDigest; AGGREGATE_SECTION_COUNT],
    status: TypedInsertStatusV2,
    bodies: [Option<Box<[u8]>>; AGGREGATE_MAX_CHUNKS + 1],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TypedInsertAggregateStatusRoots {
    pub(crate) aggregate_root: gpu_db_wal::CanonicalDigest,
    pub(crate) statement_outcome_root: gpu_db_wal::CanonicalDigest,
    pub(crate) response_root: gpu_db_wal::CanonicalDigest,
}

/// Move-only result of the generic, allocation-free aggregate preparation pass.
///
/// The prepared view carries the one measured layout, encoded framing, and section-root
/// traversal that both STATUS2 and the final body encoder require.  Keeping it tied to the
/// exact borrowed view prevents the live writer from remeasuring or rehashing the same typed
/// payload merely to cross the pre-WAL status/body boundary.
pub(crate) struct PreparedTypedInsertAggregateEncoding<'a> {
    view: TypedInsertAggregateView<'a>,
    layout: TypedInsertAggregateLayout,
    header: [u8; AGGREGATE_HEADER_BYTES as usize],
    section_headers: [[u8; AGGREGATE_SECTION_HEADER_BYTES as usize]; AGGREGATE_SECTION_COUNT],
    roots: TypedInsertAggregateStatusRoots,
    section_roots: [gpu_db_wal::CanonicalDigest; AGGREGATE_SECTION_COUNT],
}

/// Stack-only borrowed projection into the exact outer canonical WAL encoder.
pub(crate) struct TypedInsertAggregateFragmentRefs<'a> {
    fragments: [gpu_db_wal::CanonicalFragmentRef<'a>; AGGREGATE_MAX_CHUNKS + 1],
    count: usize,
}

impl<'a> TypedInsertAggregateFragmentRefs<'a> {
    pub(crate) fn as_slice(&self) -> &[gpu_db_wal::CanonicalFragmentRef<'a>] {
        &self.fragments[..self.count]
    }
}

impl EncodedTypedInsertAggregateBodies {
    pub(crate) fn layout(&self) -> &TypedInsertAggregateLayout {
        &self.layout
    }

    pub(crate) fn aggregate_root(&self) -> gpu_db_wal::CanonicalDigest {
        self.aggregate_root
    }

    pub(crate) fn section_roots(&self) -> &[gpu_db_wal::CanonicalDigest; AGGREGATE_SECTION_COUNT] {
        &self.section_roots
    }

    pub(crate) fn status(&self) -> &TypedInsertStatusV2 {
        &self.status
    }

    pub(crate) fn fragment_count(&self) -> usize {
        self.layout.fragment_count as usize
    }

    pub(crate) fn fragment_body(&self, index: usize) -> Option<&[u8]> {
        self.bodies.get(index)?.as_deref()
    }

    pub(crate) fn fragment_bodies(&self) -> impl ExactSizeIterator<Item = &[u8]> {
        self.bodies[..self.fragment_count()]
            .iter()
            .map(|body| body.as_deref().expect("live aggregate body is present"))
    }

    pub(crate) fn canonical_fragment_refs(&self) -> TypedInsertAggregateFragmentRefs<'_> {
        let empty = gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::RowMutation,
            body: &[],
        };
        let mut fragments = [empty; AGGREGATE_MAX_CHUNKS + 1];
        for (index, body) in self
            .fragment_bodies()
            .take(self.layout.chunk_count as usize)
            .enumerate()
        {
            fragments[index] = gpu_db_wal::CanonicalFragmentRef {
                kind: gpu_db_wal::CanonicalFragmentKind::RowMutation,
                body,
            };
        }
        let status_index = self.layout.chunk_count as usize;
        fragments[status_index] = gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
            body: self
                .fragment_body(status_index)
                .expect("encoded STATUS2 fragment body exists"),
        };
        TypedInsertAggregateFragmentRefs {
            fragments,
            count: self.fragment_count(),
        }
    }
}

/// Allocate exactly the fragment-body owners measured by `layout`. Callers invoke this only while
/// holding the global pre-WAL lease; no semantic encoding or authority is created yet.
pub(crate) fn reserve_typed_insert_aggregate_bodies(
    layout: TypedInsertAggregateLayout,
) -> Result<ReservedTypedInsertAggregateBodyBuffers, EngineError> {
    let mut bodies: [Option<Box<[u8]>>; AGGREGATE_MAX_CHUNKS + 1] = std::array::from_fn(|_| None);
    for (index, bytes) in layout
        .live_fragment_body_bytes()
        .iter()
        .copied()
        .enumerate()
    {
        let len = usize::try_from(bytes)
            .map_err(|_| error("fragment body exceeds addressable host memory"))?;
        bodies[index] = Some(zeroed_box(len));
    }
    Ok(ReservedTypedInsertAggregateBodyBuffers { layout, bodies })
}

/// Compute the aggregate root from a measured view without allocating or mutating output.
pub(crate) fn typed_insert_aggregate_root(
    view: &TypedInsertAggregateView<'_>,
    layout: &TypedInsertAggregateLayout,
) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
    Ok(typed_insert_aggregate_status_roots(view, layout)?.aggregate_root)
}

/// Compute the aggregate, statement-outcome, and RETURNING roots from the same exact traversal.
pub(crate) fn typed_insert_aggregate_status_roots(
    view: &TypedInsertAggregateView<'_>,
    layout: &TypedInsertAggregateLayout,
) -> Result<TypedInsertAggregateStatusRoots, EngineError> {
    Ok(prepare_typed_insert_aggregate_encoding(*view, *layout)?.roots())
}

/// Measure and hash the exact generic aggregate once before a caller constructs STATUS2.
///
/// This is deliberately data-driven: every supported aggregate semantic/type shape uses the
/// same view, framing, and root proof as the compatibility entry points.
pub(crate) fn prepare_typed_insert_aggregate_encoding<'a>(
    view: TypedInsertAggregateView<'a>,
    layout: TypedInsertAggregateLayout,
) -> Result<PreparedTypedInsertAggregateEncoding<'a>, EngineError> {
    let measured = view
        .measure()
        .map_err(|layout_error| error(&format!("measurement failed: {layout_error}")))?;
    if measured != layout {
        return Err(error("measured layout does not match aggregate view"));
    }
    prepare_typed_insert_aggregate_encoding_from_measured(view, layout)
}

/// Measure one immutable aggregate view and prepare the exact encoding from that measurement.
///
/// The live codec-5 writer owns this short, synchronous handoff.  It must not remeasure the
/// unchanged aggregate after it has just established the layout; callers with separately held
/// layouts continue to use [`prepare_typed_insert_aggregate_encoding`], which verifies them.
pub(crate) fn measure_and_prepare_typed_insert_aggregate_encoding<'a>(
    view: TypedInsertAggregateView<'a>,
) -> Result<
    (
        TypedInsertAggregateLayout,
        PreparedTypedInsertAggregateEncoding<'a>,
    ),
    EngineError,
> {
    let layout = view
        .measure()
        .map_err(|layout_error| error(&format!("measurement failed: {layout_error}")))?;
    let prepared = prepare_typed_insert_aggregate_encoding_from_measured(view, layout)?;
    Ok((layout, prepared))
}

/// Finish preparation from the measurement made in this call path.
///
/// Keeping this private ensures no independent caller can skip the layout/view equivalence
/// check exposed by `prepare_typed_insert_aggregate_encoding`.
fn prepare_typed_insert_aggregate_encoding_from_measured<'a>(
    view: TypedInsertAggregateView<'a>,
    layout: TypedInsertAggregateLayout,
) -> Result<PreparedTypedInsertAggregateEncoding<'a>, EngineError> {
    let (header, section_headers) = encoded_headers(&view, &layout)?;
    let section_roots = aggregate_section_roots(&view, &section_headers);
    let roots = status_roots_from_sections(view.flags, &header, &section_roots);
    Ok(PreparedTypedInsertAggregateEncoding {
        view,
        layout,
        header,
        section_headers,
        roots,
        section_roots,
    })
}

/// Fill every reserved chunk and STATUS2 body. Complete geometry, status identity, and all output
/// lengths are checked before the first output byte is changed. Success consumes the unencoded
/// buffers into the only type that can expose canonical fragment bodies.
pub(crate) fn encode_typed_insert_aggregate_bodies(
    view: &TypedInsertAggregateView<'_>,
    status: &TypedInsertStatusV2,
    reserved: ReservedTypedInsertAggregateBodyBuffers,
) -> Result<EncodedTypedInsertAggregateBodies, EngineError> {
    prepare_typed_insert_aggregate_encoding(*view, *reserved.layout())?.encode(status, reserved)
}

impl<'a> PreparedTypedInsertAggregateEncoding<'a> {
    pub(crate) fn roots(&self) -> TypedInsertAggregateStatusRoots {
        self.roots
    }

    /// Fill the buffers whose exact geometry this preparation pass authenticated.  STATUS2
    /// closure and all output lengths are checked before the first output byte is changed.
    pub(crate) fn encode(
        self,
        status: &TypedInsertStatusV2,
        mut reserved: ReservedTypedInsertAggregateBodyBuffers,
    ) -> Result<EncodedTypedInsertAggregateBodies, EngineError> {
        if reserved.layout != self.layout {
            return Err(error(
                "reserved layout does not match prepared aggregate view",
            ));
        }
        validate_body_buffers(&self.layout, &reserved.bodies)?;
        validate_status_identity(&self.view, &self.roots, status)?;
        // Validate STATUS2 against a stack buffer before touching any reserved body.
        let mut status_bytes = [0_u8; AGGREGATE_STATUS_V2_BYTES as usize];
        encode_status_v2(status, &mut status_bytes)?;
        for (index, chunk) in self.layout.live_chunks().iter().copied().enumerate() {
            let body = reserved.bodies[index]
                .as_deref_mut()
                .expect("validated live aggregate chunk body exists");
            encode_chunk_header(body, &self.layout, chunk, self.roots.aggregate_root);
        }
        let mut stream = ChunkPayloadWriter::new(&self.layout, &mut reserved.bodies);
        stream.bytes(&self.header)?;
        for (index, section) in self.view.sections.iter().enumerate() {
            stream.bytes(&self.section_headers[index])?;
            stream.bytes(section.payload)?;
        }
        stream.bytes(&self.roots.aggregate_root)?;
        stream.finish()?;
        let status_index = self.layout.chunk_count as usize;
        reserved.bodies[status_index]
            .as_deref_mut()
            .expect("validated STATUS2 aggregate body exists")
            .copy_from_slice(&status_bytes);
        Ok(EncodedTypedInsertAggregateBodies {
            layout: self.layout,
            aggregate_root: self.roots.aggregate_root,
            section_roots: self.section_roots,
            status: *status,
            bodies: reserved.bodies,
        })
    }
}

fn validate_status_identity(
    view: &TypedInsertAggregateView<'_>,
    roots: &TypedInsertAggregateStatusRoots,
    status: &TypedInsertStatusV2,
) -> Result<(), EngineError> {
    if status.txn_id != view.stable_transaction_id
        || status.statement_count != view.statement_count
        || status.response_artifact_count != view.sections[7].entry_count
        || status.aggregate_root != roots.aggregate_root
        || status.statement_outcome_root != roots.statement_outcome_root
        || status.response_root != roots.response_root
    {
        return Err(error("STATUS2 identity or aggregate root drifted"));
    }
    Ok(())
}

/// Strictly decode chunk bodies plus STATUS2 without assembling an aggregate stream allocation.
pub(crate) fn decode_typed_insert_aggregate_bodies<'a>(
    outer_flags: u32,
    bodies: &[&'a [u8]],
) -> Result<DecodedTypedInsertAggregate<'a>, EngineError> {
    decode_aggregate_framing(outer_flags, bodies)?.into_semantics_v1()
}

/// Decode only the canonical chunk/status/header/section framing.  Semantic dispatch happens
/// after this shared physical proof and before either version's scalar rules.  The type is kept
/// inside the aggregate codec so semantics-v2 cannot receive an alternate raw-stream carrier.
pub(super) fn decode_aggregate_framing<'a>(
    outer_flags: u32,
    bodies: &[&'a [u8]],
) -> Result<DecodedAggregateFraming<'a>, EngineError> {
    if !(2..=AGGREGATE_MAX_CHUNKS + 1).contains(&bodies.len()) {
        return Err(error("fragment body count is outside codec-5 bounds"));
    }
    let chunk_count = bodies.len() - 1;
    let mut payloads: [Option<&'a [u8]>; AGGREGATE_MAX_CHUNKS] = std::array::from_fn(|_| None);
    let mut root = None;
    let mut stream_bytes = None;
    let mut stream_offset = 0_u64;
    for (index, body) in bodies[..chunk_count].iter().copied().enumerate() {
        let decoded = decode_chunk_header(body)?;
        if decoded.ordinal as usize != index
            || decoded.chunk_count as usize != chunk_count
            || decoded.stream_offset != stream_offset
            || decoded.first != (index == 0)
            || decoded.last != (index + 1 == chunk_count)
            || (index + 1 != chunk_count
                && u64::from(decoded.payload_bytes) != AGGREGATE_CHUNK_PAYLOAD_BYTES)
        {
            return Err(error(
                "chunk order, flags, offset, or canonical size drifted",
            ));
        }
        if root
            .replace(decoded.root)
            .is_some_and(|prior| prior != decoded.root)
            || stream_bytes
                .replace(decoded.stream_bytes)
                .is_some_and(|prior| prior != decoded.stream_bytes)
        {
            return Err(error("chunk stream length or repeated root drifted"));
        }
        let payload = &body[AGGREGATE_CHUNK_HEADER_BYTES as usize..];
        payloads[index] = Some(payload);
        stream_offset = stream_offset
            .checked_add(payload.len() as u64)
            .ok_or_else(|| error("chunk payload coverage overflows"))?;
    }
    let root = root.ok_or_else(|| error("aggregate has no chunk root"))?;
    if root == [0; 32] || stream_bytes != Some(stream_offset) {
        return Err(error("aggregate chunk coverage or root is invalid"));
    }
    let status = decode_status_v2(bodies[chunk_count])?;
    if status.aggregate_root != root {
        return Err(error("STATUS2 aggregate root differs from chunk root"));
    }

    let mut reader = ChunkPayloadReader::new(payloads, chunk_count, stream_offset)?;
    let header = reader.exact::<{ AGGREGATE_HEADER_BYTES as usize }>()?;
    let decoded_header = decode_header(&header, outer_flags)?;
    let mut section_headers =
        [[0_u8; AGGREGATE_SECTION_HEADER_BYTES as usize]; AGGREGATE_SECTION_COUNT];
    let mut sections = [DecodedAggregateSection::default(); AGGREGATE_SECTION_COUNT];
    let mut section_roots = [[0_u8; 32]; AGGREGATE_SECTION_COUNT];
    for index in 0..AGGREGATE_SECTION_COUNT {
        let header_bytes = reader.exact::<{ AGGREGATE_SECTION_HEADER_BYTES as usize }>()?;
        section_headers[index] = header_bytes;
        let section = decode_section_header(&header_bytes, index)?;
        let payload_offset = reader.position();
        section_roots[index] =
            reader.digest_part(SECTION_DOMAIN, &header_bytes, section.payload_bytes)?;
        sections[index] = DecodedAggregateSection {
            tag: section.tag,
            entry_count: section.entry_count,
            payload_offset,
            payload_bytes: section.payload_bytes,
        };
    }
    let trailer = reader.exact::<32>()?;
    reader.finish()?;
    let expected_roots = status_roots_from_sections(decoded_header.flags, &header, &section_roots);
    if trailer != root
        || expected_roots.aggregate_root != root
        || status.statement_outcome_root != expected_roots.statement_outcome_root
        || status.response_root != expected_roots.response_root
    {
        return Err(error(
            "aggregate trailer, section-root, or STATUS2 root closure drifted",
        ));
    }
    let section_region_bytes = sections.iter().try_fold(0_u64, |total, section| {
        total
            .checked_add(AGGREGATE_SECTION_HEADER_BYTES)
            .and_then(|value| value.checked_add(section.payload_bytes))
            .ok_or_else(|| error("decoded section region overflows"))
    })?;
    if decoded_header.section_region_bytes != section_region_bytes {
        return Err(error("aggregate header section-region length drifted"));
    }
    Ok(DecodedAggregateFraming {
        outer_flags,
        header: decoded_header,
        header_bytes: header,
        chunk_count,
        stream_bytes: stream_offset,
        aggregate_root: root,
        section_roots,
        status,
        sections,
        payloads,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct DecodedAggregateSection {
    pub(crate) tag: u16,
    pub(crate) entry_count: u32,
    pub(crate) payload_offset: u64,
    pub(crate) payload_bytes: u64,
}

/// Shared, allocation-free physical aggregate proof.  It owns no version-specific scalar
/// interpretation: semantics-v1 consumes it through `into_semantics_v1`, while the inert S4/S7
/// owner receives the same chunk/status/section evidence through crate-private access.
pub(super) struct DecodedAggregateFraming<'a> {
    outer_flags: u32,
    header: DecodedHeader,
    header_bytes: [u8; AGGREGATE_HEADER_BYTES as usize],
    chunk_count: usize,
    stream_bytes: u64,
    aggregate_root: gpu_db_wal::CanonicalDigest,
    section_roots: [gpu_db_wal::CanonicalDigest; AGGREGATE_SECTION_COUNT],
    status: TypedInsertStatusV2,
    sections: [DecodedAggregateSection; AGGREGATE_SECTION_COUNT],
    payloads: [Option<&'a [u8]>; AGGREGATE_MAX_CHUNKS],
}

impl<'a> DecodedAggregateFraming<'a> {
    fn into_semantics_v1(self) -> Result<DecodedTypedInsertAggregate<'a>, EngineError> {
        // Dispatch precedes every v1 scalar/layout rule. Semantics two is routed only through
        // the inert S4/S7 owner; v1 must never reinterpret its zero allocator sentinels.
        if self.header.semantics_version != AGGREGATE_SEMANTICS_V1 {
            return Err(error(
                "semantics-v2 aggregate requires the inert S4/S7 decoder",
            ));
        }
        let measure = TypedInsertAggregateMeasure {
            semantics: TypedInsertAggregateSemantics::V1,
            flags: self.header.flags,
            outer_flags: self.outer_flags,
            stable_transaction_id: self.header.stable_transaction_id,
            statement_count: self.header.statement_count,
            insert_statement_count: self.header.insert_statement_count,
            original_inserted_row_count: self.header.original_inserted_row_count,
            final_row_transition_count: self.header.final_row_transition_count,
            allocator_before: self.header.allocator_before,
            allocator_high_water: self.header.allocator_high_water,
            table_block_count: self.header.table_block_count,
            sections: std::array::from_fn(|index| AggregateSectionMeasure {
                entry_count: self.sections[index].entry_count,
                payload_bytes: self.sections[index].payload_bytes,
            }),
        };
        let layout = measure
            .measure()
            .map_err(|layout| error(&format!("decoded layout is invalid: {layout}")))?;
        if layout.stream_bytes != self.stream_bytes
            || layout.chunk_count as usize != self.chunk_count
            || layout
                .live_chunks()
                .iter()
                .zip(self.payloads.iter().flatten())
                .any(|(chunk, payload)| chunk.payload_bytes as usize != payload.len())
            || self.status.txn_id != measure.stable_transaction_id
            || self.status.statement_count != measure.statement_count
            || self.status.response_artifact_count != self.sections[7].entry_count
        {
            return Err(error("decoded chunk/layout/status geometry drifted"));
        }
        Ok(DecodedTypedInsertAggregate {
            layout,
            framing: self,
        })
    }

    pub(super) fn semantics_version(&self) -> u16 {
        self.header.semantics_version
    }

    /// The exact canonical row-chunk count plus the trailing STATUS2 fragment.
    /// Version-specific closure shares this already-proved framing rather than rebuilding it.
    pub(super) fn fragment_count(&self) -> usize {
        self.chunk_count + 1
    }

    pub(super) fn outer_flags(&self) -> u32 {
        self.outer_flags
    }

    pub(super) fn status(&self) -> &TypedInsertStatusV2 {
        &self.status
    }

    pub(super) fn sections(&self) -> &[DecodedAggregateSection; AGGREGATE_SECTION_COUNT] {
        &self.sections
    }

    pub(super) fn header_bytes(&self) -> &[u8; AGGREGATE_HEADER_BYTES as usize] {
        &self.header_bytes
    }

    pub(super) fn header_scalars(&self) -> DecodedAggregateHeaderScalars {
        DecodedAggregateHeaderScalars {
            flags: self.header.flags,
            stable_transaction_id: self.header.stable_transaction_id,
            statement_count: self.header.statement_count,
            insert_statement_count: self.header.insert_statement_count,
            original_inserted_row_count: self.header.original_inserted_row_count,
            final_row_transition_count: self.header.final_row_transition_count,
            allocator_before: self.header.allocator_before,
            allocator_high_water: self.header.allocator_high_water,
            table_block_count: self.header.table_block_count,
        }
    }

    pub(super) fn aggregate_root(&self) -> gpu_db_wal::CanonicalDigest {
        self.aggregate_root
    }

    pub(super) fn section_roots(&self) -> &[gpu_db_wal::CanonicalDigest; AGGREGATE_SECTION_COUNT] {
        &self.section_roots
    }

    pub(super) fn with_section_reader<T>(
        &self,
        section: usize,
        visitor: impl FnOnce(&mut DecodedAggregateSectionReader<'_>) -> Result<T, EngineError>,
    ) -> Result<T, EngineError> {
        let section = self
            .sections
            .get(section)
            .ok_or_else(|| error("section ordinal is out of range"))?;
        let mut reader = DecodedAggregateSectionReader::new(
            self.payloads,
            self.chunk_count,
            section.payload_offset,
            section.payload_bytes,
        )?;
        let value = visitor(&mut reader)?;
        reader.finish()?;
        Ok(value)
    }
}

/// Re-encode a valid catalog-bearing semantics-v2 aggregate as the historical profile that
/// predated the additive `OPERATION_COMPOSITION` marker. This exists solely to keep the old
/// CATALOG-only S3 decoder/replay contract covered without making the current writer emit a
/// retired profile.
#[cfg(test)]
pub(crate) fn reencode_legacy_catalog_marker_for_test(
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    outcome: &gpu_db_wal::CanonicalOutcome,
    fragments: &[gpu_db_wal::CanonicalFragment],
) -> Result<
    (
        gpu_db_wal::CanonicalPreApplyHeader,
        gpu_db_wal::CanonicalOutcome,
        Vec<gpu_db_wal::CanonicalFragment>,
    ),
    EngineError,
> {
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
    let scalar = framing.header_scalars();
    if framing.semantics_version() != AGGREGATE_SEMANTICS_V2
        || scalar.flags & AGGREGATE_FLAG_CATALOG == 0
        || scalar.flags & AGGREGATE_FLAG_OPERATION_COMPOSITION == 0
        || outer.flags & OUTER_CONTENT_CATALOG == 0
        || outer.flags & OUTER_CONTENT_OPERATION_COMPOSITION == 0
    {
        return Err(error(
            "legacy catalog-marker test source is not an explicit current catalog S3 aggregate",
        ));
    }

    let mut payloads: [Vec<u8>; AGGREGATE_SECTION_COUNT] = std::array::from_fn(|_| Vec::new());
    for (index, payload) in payloads.iter_mut().enumerate() {
        let bytes = usize::try_from(framing.sections()[index].payload_bytes)
            .map_err(|_| error("legacy catalog-marker section length is not addressable"))?;
        payload
            .try_reserve_exact(bytes)
            .map_err(|_| error("legacy catalog-marker section reservation failed"))?;
        payload.resize(bytes, 0);
        framing.with_section_reader(index, |reader| reader.copy_exact(payload))?;
    }
    let sections = std::array::from_fn(|index| TypedInsertAggregateSectionView {
        entry_count: framing.sections()[index].entry_count,
        payload: payloads[index].as_slice(),
    });
    let view = TypedInsertAggregateView {
        semantics: TypedInsertAggregateSemantics::V2,
        flags: scalar.flags & !AGGREGATE_FLAG_OPERATION_COMPOSITION,
        outer_flags: outer.flags & !OUTER_CONTENT_OPERATION_COMPOSITION,
        stable_transaction_id: scalar.stable_transaction_id,
        statement_count: scalar.statement_count,
        insert_statement_count: scalar.insert_statement_count,
        original_inserted_row_count: scalar.original_inserted_row_count,
        final_row_transition_count: scalar.final_row_transition_count,
        allocator_before: scalar.allocator_before,
        allocator_high_water: scalar.allocator_high_water,
        table_block_count: scalar.table_block_count,
        sections,
    };
    let (layout, prepared) = measure_and_prepare_typed_insert_aggregate_encoding(view)?;
    let roots = prepared.roots();
    let mut status = *framing.status();
    status.statement_outcome_root = roots.statement_outcome_root;
    status.response_root = roots.response_root;
    status.aggregate_root = roots.aggregate_root;
    let reencoded = prepared.encode(&status, reserve_typed_insert_aggregate_bodies(layout)?)?;

    let mut historical_outer = outer.clone();
    historical_outer.flags &= !OUTER_CONTENT_OPERATION_COMPOSITION;
    let mut historical_outcome = outcome.clone();
    historical_outcome.target_digest = roots.aggregate_root;
    historical_outcome.returning_digest = roots.response_root;
    let historical_fragments = reencoded
        .canonical_fragment_refs()
        .as_slice()
        .iter()
        .map(|fragment| gpu_db_wal::CanonicalFragment {
            kind: fragment.kind,
            body: fragment.body.to_vec(),
        })
        .collect();
    Ok((historical_outer, historical_outcome, historical_fragments))
}

#[derive(Clone, Copy)]
pub(super) struct DecodedAggregateHeaderScalars {
    pub(super) flags: u32,
    pub(super) stable_transaction_id: u64,
    pub(super) statement_count: u32,
    pub(super) insert_statement_count: u32,
    pub(super) original_inserted_row_count: u64,
    pub(super) final_row_transition_count: u64,
    pub(super) allocator_before: u64,
    pub(super) allocator_high_water: u64,
    pub(super) table_block_count: u32,
}

pub(crate) struct DecodedTypedInsertAggregate<'a> {
    layout: TypedInsertAggregateLayout,
    framing: DecodedAggregateFraming<'a>,
}

impl DecodedTypedInsertAggregate<'_> {
    pub(crate) fn layout(&self) -> &TypedInsertAggregateLayout {
        &self.layout
    }

    pub(crate) fn aggregate_root(&self) -> gpu_db_wal::CanonicalDigest {
        self.framing.aggregate_root
    }

    pub(crate) fn section_roots(&self) -> &[gpu_db_wal::CanonicalDigest; AGGREGATE_SECTION_COUNT] {
        &self.framing.section_roots
    }

    pub(crate) fn status(&self) -> &TypedInsertStatusV2 {
        &self.framing.status
    }

    pub(crate) fn sections(&self) -> &[DecodedAggregateSection; AGGREGATE_SECTION_COUNT] {
        &self.framing.sections
    }

    pub(crate) fn copy_section_payload(
        &self,
        section: usize,
        out: &mut [u8],
    ) -> Result<(), EngineError> {
        let section = self
            .framing
            .sections
            .get(section)
            .ok_or_else(|| error("section ordinal is out of range"))?;
        if out.len() as u64 != section.payload_bytes {
            return Err(error("section output length is not exact"));
        }
        copy_payload_range(
            &self.framing.payloads,
            self.framing.chunk_count,
            section.payload_offset,
            out,
        )
    }

    /// Visit one length-delimited aggregate section without first assembling it into a contiguous
    /// host buffer.  The reader is bounded to the selected section and keeps the original chunk
    /// ownership borrowed, so callers cannot accidentally turn a hostile four-chunk aggregate
    /// into an unaccounted full-section allocation.
    pub(super) fn with_section_reader<T>(
        &self,
        section: usize,
        visitor: impl FnOnce(&mut DecodedAggregateSectionReader<'_>) -> Result<T, EngineError>,
    ) -> Result<T, EngineError> {
        self.framing.with_section_reader(section, visitor)
    }
}

/// Bounded cross-chunk reader for one decoded codec-5 section.  It intentionally exposes only
/// fixed-width scalar reads, exact copies into caller-owned storage, and checked skipping.  In
/// particular it never returns a raw aggregate slice and never owns a `Vec`, which makes a
/// validation pass allocation-free even when an entry straddles a fragment boundary.
pub(super) struct DecodedAggregateSectionReader<'a> {
    payloads: [Option<&'a [u8]>; AGGREGATE_MAX_CHUNKS],
    chunk_count: usize,
    position: u64,
    end: u64,
}

impl<'a> DecodedAggregateSectionReader<'a> {
    fn new(
        payloads: [Option<&'a [u8]>; AGGREGATE_MAX_CHUNKS],
        chunk_count: usize,
        offset: u64,
        bytes: u64,
    ) -> Result<Self, EngineError> {
        if payloads[..chunk_count].iter().any(Option::is_none)
            || payloads[chunk_count..].iter().any(Option::is_some)
        {
            return Err(error("decoded section payload set is not exact"));
        }
        let stream_bytes =
            payloads[..chunk_count]
                .iter()
                .flatten()
                .try_fold(0_u64, |total, payload| {
                    total
                        .checked_add(payload.len() as u64)
                        .ok_or_else(|| error("decoded section stream length overflows"))
                })?;
        let end = offset
            .checked_add(bytes)
            .filter(|end| *end <= stream_bytes)
            .ok_or_else(|| error("decoded section range exceeds aggregate stream"))?;
        Ok(Self {
            payloads,
            chunk_count,
            position: offset,
            end,
        })
    }

    pub(super) fn remaining(&self) -> u64 {
        self.end.saturating_sub(self.position)
    }

    pub(super) fn done(&self) -> bool {
        self.position == self.end
    }

    pub(super) fn u8(&mut self) -> Result<u8, EngineError> {
        Ok(self.exact::<1>()?[0])
    }

    pub(super) fn u16(&mut self) -> Result<u16, EngineError> {
        Ok(u16::from_le_bytes(self.exact()?))
    }

    pub(super) fn u32(&mut self) -> Result<u32, EngineError> {
        Ok(u32::from_le_bytes(self.exact()?))
    }

    pub(super) fn u64(&mut self) -> Result<u64, EngineError> {
        Ok(u64::from_le_bytes(self.exact()?))
    }

    pub(super) fn i64(&mut self) -> Result<i64, EngineError> {
        Ok(i64::from_le_bytes(self.exact()?))
    }

    pub(super) fn digest(&mut self) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
        self.exact()
    }

    pub(super) fn exact<const N: usize>(&mut self) -> Result<[u8; N], EngineError> {
        let mut out = [0_u8; N];
        self.copy_exact(&mut out)?;
        Ok(out)
    }

    pub(super) fn copy_exact(&mut self, mut out: &mut [u8]) -> Result<(), EngineError> {
        let bytes = u64::try_from(out.len()).map_err(|_| error("section copy length overflows"))?;
        let requested_end = self
            .position
            .checked_add(bytes)
            .filter(|end| *end <= self.end)
            .ok_or_else(|| error("section reader is truncated"))?;
        while !out.is_empty() {
            let (chunk, within) = self.current_chunk()?;
            let take = (chunk.len() - within).min(out.len());
            out[..take].copy_from_slice(&chunk[within..within + take]);
            self.position = self
                .position
                .checked_add(take as u64)
                .ok_or_else(|| error("section reader cursor overflows"))?;
            out = &mut out[take..];
        }
        if self.position != requested_end {
            return Err(error("section reader did not consume the requested range"));
        }
        Ok(())
    }

    pub(super) fn skip(&mut self, bytes: u64) -> Result<(), EngineError> {
        self.position = self
            .position
            .checked_add(bytes)
            .filter(|end| *end <= self.end)
            .ok_or_else(|| error("section skip exceeds its exact range"))?;
        Ok(())
    }

    fn current_chunk(&self) -> Result<(&'a [u8], usize), EngineError> {
        if self.position >= self.end {
            return Err(error("section reader is truncated"));
        }
        let mut start = 0_u64;
        for payload in self.payloads[..self.chunk_count].iter().flatten() {
            let end = start
                .checked_add(payload.len() as u64)
                .ok_or_else(|| error("section chunk coverage overflows"))?;
            if self.position < end {
                return Ok((
                    payload,
                    usize::try_from(self.position - start)
                        .map_err(|_| error("section chunk-relative offset overflows"))?,
                ));
            }
            start = end;
        }
        Err(error("section reader has no current chunk"))
    }

    fn finish(&self) -> Result<(), EngineError> {
        if self.done() {
            Ok(())
        } else {
            Err(error("section reader leaves trailing bytes"))
        }
    }
}

fn encoded_headers(
    view: &TypedInsertAggregateView<'_>,
    layout: &TypedInsertAggregateLayout,
) -> Result<
    (
        [u8; AGGREGATE_HEADER_BYTES as usize],
        [[u8; AGGREGATE_SECTION_HEADER_BYTES as usize]; AGGREGATE_SECTION_COUNT],
    ),
    EngineError,
> {
    let mut header = [0_u8; AGGREGATE_HEADER_BYTES as usize];
    let mut writer = FixedWriter::new(&mut header);
    writer.bytes(AGGREGATE_STREAM_MAGIC)?;
    writer.u16(AGGREGATE_FORMAT_VERSION)?;
    writer.u16(view.semantics.wire_version())?;
    writer.u16(AGGREGATE_FORMAT_VERSION)?;
    writer.u16(AGGREGATE_FORMAT_VERSION)?;
    writer.u32(view.flags)?;
    writer.u16(AGGREGATE_SECTION_COUNT as u16)?;
    writer.u16(0)?;
    writer.u64(layout.section_region_bytes)?;
    writer.u64(view.stable_transaction_id)?;
    writer.u32(view.statement_count)?;
    writer.u32(view.insert_statement_count)?;
    writer.u64(view.original_inserted_row_count)?;
    writer.u64(view.final_row_transition_count)?;
    writer.u64(view.allocator_before)?;
    writer.u64(view.allocator_high_water)?;
    writer.u32(view.table_block_count)?;
    writer.u32(0)?;
    writer.finish()?;
    let section_headers = std::array::from_fn(|index| {
        let section = &view.sections[index];
        let mut bytes = [0_u8; AGGREGATE_SECTION_HEADER_BYTES as usize];
        bytes[0..2].copy_from_slice(&((index + 1) as u16).to_le_bytes());
        bytes[4..8].copy_from_slice(&section.entry_count.to_le_bytes());
        bytes[8..16].copy_from_slice(&layout.sections[index].payload_bytes.to_le_bytes());
        bytes
    });
    Ok((header, section_headers))
}

fn aggregate_status_roots(
    view: &TypedInsertAggregateView<'_>,
    header: &[u8; AGGREGATE_HEADER_BYTES as usize],
    section_headers: &[[u8; AGGREGATE_SECTION_HEADER_BYTES as usize]; AGGREGATE_SECTION_COUNT],
) -> TypedInsertAggregateStatusRoots {
    let section_roots = aggregate_section_roots(view, section_headers);
    status_roots_from_sections(view.flags, header, &section_roots)
}

fn aggregate_section_roots(
    view: &TypedInsertAggregateView<'_>,
    section_headers: &[[u8; AGGREGATE_SECTION_HEADER_BYTES as usize]; AGGREGATE_SECTION_COUNT],
) -> [gpu_db_wal::CanonicalDigest; AGGREGATE_SECTION_COUNT] {
    std::array::from_fn(|index| {
        digest_parts(
            SECTION_DOMAIN,
            &[&section_headers[index], view.sections[index].payload],
        )
    })
}

fn status_roots_from_sections(
    flags: u32,
    header: &[u8; AGGREGATE_HEADER_BYTES as usize],
    section_roots: &[[u8; 32]; AGGREGATE_SECTION_COUNT],
) -> TypedInsertAggregateStatusRoots {
    let aggregate_root = root_from_section_roots(header, section_roots);
    let statement_outcome_root = section_roots[5];
    let response_root = if flags & AGGREGATE_FLAG_RETURNING != 0 {
        digest_parts(
            RESPONSE_ROOT_DOMAIN,
            &[&section_roots[5], &section_roots[7]],
        )
    } else {
        [0; 32]
    };
    TypedInsertAggregateStatusRoots {
        aggregate_root,
        statement_outcome_root,
        response_root,
    }
}

fn root_from_section_roots(
    header: &[u8; AGGREGATE_HEADER_BYTES as usize],
    section_roots: &[[u8; 32]; AGGREGATE_SECTION_COUNT],
) -> gpu_db_wal::CanonicalDigest {
    let mut hash = begin_digest(ROOT_DOMAIN);
    append_part(&mut hash, header);
    for root in section_roots {
        append_part(&mut hash, root);
    }
    hash.finalize().into()
}

fn digest_parts(domain: &[u8], parts: &[&[u8]]) -> gpu_db_wal::CanonicalDigest {
    let mut hash = begin_digest(domain);
    for part in parts {
        append_part(&mut hash, part);
    }
    hash.finalize().into()
}

fn begin_digest(domain: &[u8]) -> Sha256 {
    let mut hash = Sha256::new();
    hash.update((domain.len() as u64).to_le_bytes());
    hash.update(domain);
    hash
}

fn append_part(hash: &mut Sha256, part: &[u8]) {
    hash.update((part.len() as u64).to_le_bytes());
    hash.update(part);
}

fn validate_body_buffers(
    layout: &TypedInsertAggregateLayout,
    bodies: &[Option<Box<[u8]>>; AGGREGATE_MAX_CHUNKS + 1],
) -> Result<(), EngineError> {
    for (index, expected) in layout
        .live_fragment_body_bytes()
        .iter()
        .copied()
        .enumerate()
    {
        if bodies[index].as_deref().map(|body| body.len() as u64) != Some(expected) {
            return Err(error("reserved fragment body length is not exact"));
        }
    }
    if bodies[layout.fragment_count as usize..]
        .iter()
        .any(Option::is_some)
    {
        return Err(error("unused reserved fragment body is populated"));
    }
    Ok(())
}

fn encode_chunk_header(
    body: &mut [u8],
    layout: &TypedInsertAggregateLayout,
    chunk: AggregateChunkLayout,
    root: gpu_db_wal::CanonicalDigest,
) {
    let header = &mut body[..AGGREGATE_CHUNK_HEADER_BYTES as usize];
    header[0..8].copy_from_slice(AGGREGATE_CHUNK_MAGIC);
    header[8] = ENGINE_OPERATION_CODEC_TYPED_INSERT_AGGREGATE;
    header[9] = AGGREGATE_FORMAT_VERSION as u8;
    let flags = (u16::from(chunk.first) * AGGREGATE_CHUNK_FLAG_FIRST)
        | (u16::from(chunk.last) * AGGREGATE_CHUNK_FLAG_LAST);
    header[10..12].copy_from_slice(&flags.to_le_bytes());
    header[12..20].copy_from_slice(&layout.stream_bytes.to_le_bytes());
    header[20..24].copy_from_slice(&chunk.ordinal.to_le_bytes());
    header[24..28].copy_from_slice(&layout.chunk_count.to_le_bytes());
    header[28..36].copy_from_slice(&chunk.stream_offset.to_le_bytes());
    header[36..40].copy_from_slice(&chunk.payload_bytes.to_le_bytes());
    header[40..44].fill(0);
    header[44..76].copy_from_slice(&root);
}

struct DecodedChunkHeader {
    stream_bytes: u64,
    ordinal: u32,
    chunk_count: u32,
    stream_offset: u64,
    payload_bytes: u32,
    root: gpu_db_wal::CanonicalDigest,
    first: bool,
    last: bool,
}

fn decode_chunk_header(body: &[u8]) -> Result<DecodedChunkHeader, EngineError> {
    if body.len() < AGGREGATE_CHUNK_HEADER_BYTES as usize
        || &body[0..8] != AGGREGATE_CHUNK_MAGIC
        || body[8] != ENGINE_OPERATION_CODEC_TYPED_INSERT_AGGREGATE
        || body[9] != AGGREGATE_FORMAT_VERSION as u8
    {
        return Err(error("chunk header magic, codec, or format is invalid"));
    }
    let flags = u16::from_le_bytes(body[10..12].try_into().expect("fixed chunk field"));
    if flags & !AGGREGATE_CHUNK_KNOWN_FLAGS != 0 || body[40..44].iter().any(|byte| *byte != 0) {
        return Err(error("chunk flags or reserved field is invalid"));
    }
    let payload_bytes = u32::from_le_bytes(body[36..40].try_into().expect("fixed chunk field"));
    if payload_bytes as usize != body.len() - AGGREGATE_CHUNK_HEADER_BYTES as usize {
        return Err(error("chunk payload length differs from exact body"));
    }
    Ok(DecodedChunkHeader {
        stream_bytes: u64::from_le_bytes(body[12..20].try_into().expect("fixed chunk field")),
        ordinal: u32::from_le_bytes(body[20..24].try_into().expect("fixed chunk field")),
        chunk_count: u32::from_le_bytes(body[24..28].try_into().expect("fixed chunk field")),
        stream_offset: u64::from_le_bytes(body[28..36].try_into().expect("fixed chunk field")),
        payload_bytes,
        root: body[44..76].try_into().expect("fixed chunk root"),
        first: flags & AGGREGATE_CHUNK_FLAG_FIRST != 0,
        last: flags & AGGREGATE_CHUNK_FLAG_LAST != 0,
    })
}

struct DecodedHeader {
    semantics_version: u16,
    flags: u32,
    section_region_bytes: u64,
    stable_transaction_id: u64,
    statement_count: u32,
    insert_statement_count: u32,
    original_inserted_row_count: u64,
    final_row_transition_count: u64,
    allocator_before: u64,
    allocator_high_water: u64,
    table_block_count: u32,
}

fn decode_header(
    bytes: &[u8; AGGREGATE_HEADER_BYTES as usize],
    _outer_flags: u32,
) -> Result<DecodedHeader, EngineError> {
    if &bytes[0..16] != AGGREGATE_STREAM_MAGIC
        || u16::from_le_bytes(bytes[16..18].try_into().expect("fixed field"))
            != AGGREGATE_FORMAT_VERSION
        || !matches!(
            u16::from_le_bytes(bytes[18..20].try_into().expect("fixed field")),
            AGGREGATE_SEMANTICS_V1 | AGGREGATE_SEMANTICS_V2
        )
        || u16::from_le_bytes(bytes[20..22].try_into().expect("fixed field"))
            != AGGREGATE_FORMAT_VERSION
        || u16::from_le_bytes(bytes[22..24].try_into().expect("fixed field"))
            != AGGREGATE_FORMAT_VERSION
        || u16::from_le_bytes(bytes[28..30].try_into().expect("fixed field"))
            != AGGREGATE_SECTION_COUNT as u16
        || bytes[30..32].iter().any(|byte| *byte != 0)
        || bytes[92..96].iter().any(|byte| *byte != 0)
    {
        return Err(error(
            "aggregate header version, count, or reserved field drifted",
        ));
    }
    Ok(DecodedHeader {
        semantics_version: u16::from_le_bytes(bytes[18..20].try_into().expect("fixed field")),
        flags: u32::from_le_bytes(bytes[24..28].try_into().expect("fixed field")),
        section_region_bytes: u64::from_le_bytes(bytes[32..40].try_into().expect("fixed field")),
        stable_transaction_id: u64::from_le_bytes(bytes[40..48].try_into().expect("fixed field")),
        statement_count: u32::from_le_bytes(bytes[48..52].try_into().expect("fixed field")),
        insert_statement_count: u32::from_le_bytes(bytes[52..56].try_into().expect("fixed field")),
        original_inserted_row_count: u64::from_le_bytes(
            bytes[56..64].try_into().expect("fixed field"),
        ),
        final_row_transition_count: u64::from_le_bytes(
            bytes[64..72].try_into().expect("fixed field"),
        ),
        allocator_before: u64::from_le_bytes(bytes[72..80].try_into().expect("fixed field")),
        allocator_high_water: u64::from_le_bytes(bytes[80..88].try_into().expect("fixed field")),
        table_block_count: u32::from_le_bytes(bytes[88..92].try_into().expect("fixed field")),
    })
}

struct SectionHeader {
    tag: u16,
    entry_count: u32,
    payload_bytes: u64,
}

fn decode_section_header(
    bytes: &[u8; AGGREGATE_SECTION_HEADER_BYTES as usize],
    index: usize,
) -> Result<SectionHeader, EngineError> {
    let tag = u16::from_le_bytes(bytes[0..2].try_into().expect("fixed field"));
    if tag != (index + 1) as u16 || bytes[2..4].iter().any(|byte| *byte != 0) {
        return Err(error("section tag, order, or flags are invalid"));
    }
    Ok(SectionHeader {
        tag,
        entry_count: u32::from_le_bytes(bytes[4..8].try_into().expect("fixed field")),
        payload_bytes: u64::from_le_bytes(bytes[8..16].try_into().expect("fixed field")),
    })
}

struct ChunkPayloadWriter<'a> {
    layout: &'a TypedInsertAggregateLayout,
    bodies: &'a mut [Option<Box<[u8]>>; AGGREGATE_MAX_CHUNKS + 1],
    position: u64,
}

impl<'a> ChunkPayloadWriter<'a> {
    fn new(
        layout: &'a TypedInsertAggregateLayout,
        bodies: &'a mut [Option<Box<[u8]>>; AGGREGATE_MAX_CHUNKS + 1],
    ) -> Self {
        Self {
            layout,
            bodies,
            position: 0,
        }
    }

    fn bytes(&mut self, mut value: &[u8]) -> Result<(), EngineError> {
        while !value.is_empty() {
            let chunk_index = usize::try_from(self.position / AGGREGATE_CHUNK_PAYLOAD_BYTES)
                .map_err(|_| error("aggregate stream position overflows"))?;
            let chunk = self
                .layout
                .live_chunks()
                .get(chunk_index)
                .ok_or_else(|| error("aggregate stream exceeds reserved chunks"))?;
            let within = usize::try_from(self.position - chunk.stream_offset)
                .map_err(|_| error("aggregate chunk-relative position overflows"))?;
            let available = chunk.payload_bytes as usize - within;
            let take = available.min(value.len());
            let start = AGGREGATE_CHUNK_HEADER_BYTES as usize + within;
            let end = start + take;
            self.bodies[chunk_index]
                .as_deref_mut()
                .expect("validated chunk body exists")[start..end]
                .copy_from_slice(&value[..take]);
            self.position += take as u64;
            value = &value[take..];
        }
        Ok(())
    }

    fn finish(self) -> Result<(), EngineError> {
        if self.position != self.layout.stream_bytes {
            return Err(error(
                "aggregate stream leaves reserved payload bytes unwritten",
            ));
        }
        Ok(())
    }
}

struct ChunkPayloadReader<'a> {
    payloads: [Option<&'a [u8]>; AGGREGATE_MAX_CHUNKS],
    chunk_count: usize,
    stream_bytes: u64,
    position: u64,
}

impl<'a> ChunkPayloadReader<'a> {
    fn new(
        payloads: [Option<&'a [u8]>; AGGREGATE_MAX_CHUNKS],
        chunk_count: usize,
        stream_bytes: u64,
    ) -> Result<Self, EngineError> {
        if payloads[..chunk_count].iter().any(Option::is_none)
            || payloads[chunk_count..].iter().any(Option::is_some)
        {
            return Err(error("decoded chunk payload set is not exact"));
        }
        Ok(Self {
            payloads,
            chunk_count,
            stream_bytes,
            position: 0,
        })
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn exact<const N: usize>(&mut self) -> Result<[u8; N], EngineError> {
        let mut out = [0_u8; N];
        self.copy_exact(&mut out)?;
        Ok(out)
    }

    fn copy_exact(&mut self, mut out: &mut [u8]) -> Result<(), EngineError> {
        while !out.is_empty() {
            let (chunk, within) = self.current_chunk()?;
            let take = (chunk.len() - within).min(out.len());
            out[..take].copy_from_slice(&chunk[within..within + take]);
            self.position += take as u64;
            out = &mut out[take..];
        }
        Ok(())
    }

    fn digest_part(
        &mut self,
        domain: &[u8],
        header: &[u8],
        payload_bytes: u64,
    ) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
        let mut hash = begin_digest(domain);
        append_part(&mut hash, header);
        hash.update(payload_bytes.to_le_bytes());
        let end = self
            .position
            .checked_add(payload_bytes)
            .filter(|end| *end <= self.stream_bytes)
            .ok_or_else(|| error("section payload exceeds aggregate stream"))?;
        while self.position < end {
            let (chunk, within) = self.current_chunk()?;
            let remaining = usize::try_from(end - self.position)
                .map_err(|_| error("section payload range overflows"))?;
            let take = (chunk.len() - within).min(remaining);
            hash.update(&chunk[within..within + take]);
            self.position += take as u64;
        }
        Ok(hash.finalize().into())
    }

    fn current_chunk(&self) -> Result<(&'a [u8], usize), EngineError> {
        if self.position >= self.stream_bytes {
            return Err(error("aggregate stream is truncated"));
        }
        let mut start = 0_u64;
        for payload in self.payloads[..self.chunk_count].iter().flatten() {
            let end = start
                .checked_add(payload.len() as u64)
                .ok_or_else(|| error("decoded chunk coverage overflows"))?;
            if self.position < end {
                return Ok((
                    payload,
                    usize::try_from(self.position - start)
                        .map_err(|_| error("decoded chunk offset overflows"))?,
                ));
            }
            start = end;
        }
        Err(error("aggregate stream has no current chunk"))
    }

    fn finish(self) -> Result<(), EngineError> {
        if self.position != self.stream_bytes {
            return Err(error("aggregate stream has trailing bytes"));
        }
        Ok(())
    }
}

fn copy_payload_range(
    payloads: &[Option<&[u8]>; AGGREGATE_MAX_CHUNKS],
    chunk_count: usize,
    offset: u64,
    mut out: &mut [u8],
) -> Result<(), EngineError> {
    let mut stream_start = 0_u64;
    let mut wanted = offset;
    for payload in payloads[..chunk_count].iter().flatten() {
        let stream_end = stream_start
            .checked_add(payload.len() as u64)
            .ok_or_else(|| error("section copy stream range overflows"))?;
        if wanted < stream_end {
            let within = usize::try_from(wanted - stream_start)
                .map_err(|_| error("section copy offset overflows"))?;
            let take = (payload.len() - within).min(out.len());
            out[..take].copy_from_slice(&payload[within..within + take]);
            out = &mut out[take..];
            wanted += take as u64;
            if out.is_empty() {
                return Ok(());
            }
        }
        stream_start = stream_end;
    }
    Err(error("section copy exceeds decoded aggregate payload"))
}

struct FixedWriter<'a> {
    bytes: &'a mut [u8],
    position: usize,
}

impl<'a> FixedWriter<'a> {
    fn new(bytes: &'a mut [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn bytes(&mut self, value: &[u8]) -> Result<(), EngineError> {
        let end = self
            .position
            .checked_add(value.len())
            .ok_or_else(|| error("fixed aggregate writer overflows"))?;
        self.bytes
            .get_mut(self.position..end)
            .ok_or_else(|| error("fixed aggregate output is short"))?
            .copy_from_slice(value);
        self.position = end;
        Ok(())
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
            return Err(error("fixed aggregate output has surplus bytes"));
        }
        Ok(())
    }
}

fn zeroed_box(len: usize) -> Box<[u8]> {
    let mut bytes = Box::<[u8]>::new_uninit_slice(len);
    for byte in bytes.iter_mut() {
        byte.write(0);
    }
    // SAFETY: every u8 slot is initialized above.
    unsafe { bytes.assume_init() }
}

fn error(message: &str) -> EngineError {
    EngineError::Durability(format!("typed INSERT aggregate codec: {message}"))
}

#[cfg(test)]
mod internal_tests {
    use super::*;

    const COUNTS: [u32; AGGREGATE_SECTION_COUNT] = [1, 1, 0, 1, 0, 1, 1, 0];

    fn view<'a>(payloads: &'a [Vec<u8>; AGGREGATE_SECTION_COUNT]) -> TypedInsertAggregateView<'a> {
        TypedInsertAggregateView {
            semantics: TypedInsertAggregateSemantics::V1,
            flags: AGGREGATE_FLAG_AUTOCOMMIT,
            outer_flags: OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
            stable_transaction_id: 7,
            statement_count: 1,
            insert_statement_count: 1,
            original_inserted_row_count: 1,
            final_row_transition_count: 1,
            allocator_before: 70,
            allocator_high_water: 71,
            table_block_count: 1,
            sections: std::array::from_fn(|index| TypedInsertAggregateSectionView {
                entry_count: COUNTS[index],
                payload: &payloads[index],
            }),
        }
    }

    fn status(roots: TypedInsertAggregateStatusRoots) -> TypedInsertStatusV2 {
        TypedInsertStatusV2 {
            database_id: [1; 16],
            timeline_id: [2; 16],
            txn_id: 7,
            request_digest: [3; 32],
            isolation: 1,
            flags: 0,
            retention_deadline: 0,
            statement_count: 1,
            response_artifact_count: 0,
            statement_outcome_root: roots.statement_outcome_root,
            response_root: roots.response_root,
            aggregate_root: roots.aggregate_root,
        }
    }

    #[test]
    fn short_or_surplus_reserved_body_fails_before_any_canary_mutation() {
        let payloads: [Vec<u8>; AGGREGATE_SECTION_COUNT] =
            std::array::from_fn(|index| vec![index as u8 + 1; index + 1]);
        let view = view(&payloads);
        let layout = view.measure().unwrap();

        for (body_index, adjustment) in [(0_usize, -1_isize), (0, 1), (1, -1), (1, 1)] {
            let mut reserved = reserve_typed_insert_aggregate_bodies(layout).unwrap();
            for body in reserved.bodies.iter_mut().flatten() {
                body.fill(0xa5);
            }
            let original_len = reserved.bodies[body_index].as_ref().unwrap().len();
            let changed_len = original_len.checked_add_signed(adjustment).unwrap();
            reserved.bodies[body_index] = Some(vec![0xa5; changed_len].into_boxed_slice());
            let before: Vec<Vec<u8>> = reserved
                .bodies
                .iter()
                .flatten()
                .map(|body| body.to_vec())
                .collect();
            // The prepared encoder validates exact reserved geometry before it can write a
            // chunk header, so this is the no-mutation rejection boundary.
            assert!(validate_body_buffers(&layout, &reserved.bodies).is_err());
            let after: Vec<Vec<u8>> = reserved
                .bodies
                .iter()
                .flatten()
                .map(|body| body.to_vec())
                .collect();
            assert_eq!(after, before);
            assert!(after.iter().flatten().all(|byte| *byte == 0xa5));
        }
    }

    #[test]
    fn section_reader_crosses_chunk_boundaries_without_assembling_a_section() {
        let first = b"abc".as_slice();
        let second = b"def".as_slice();
        let mut payloads = [None; AGGREGATE_MAX_CHUNKS];
        payloads[0] = Some(first);
        payloads[1] = Some(second);
        let mut reader = DecodedAggregateSectionReader::new(payloads, 2, 1, 4).unwrap();
        assert_eq!(reader.exact::<4>().unwrap(), *b"bcde");
        assert!(reader.done());
        reader.finish().unwrap();
    }

    #[test]
    fn section_reader_enforces_exact_boundary_truncation_and_trailing_contracts() {
        let mut payloads = [None; AGGREGATE_MAX_CHUNKS];
        payloads[0] = Some(b"abc".as_slice());

        let mut truncated = DecodedAggregateSectionReader::new(payloads, 1, 0, 3).unwrap();
        assert!(truncated.u32().is_err());

        let mut trailing = DecodedAggregateSectionReader::new(payloads, 1, 0, 3).unwrap();
        assert_eq!(trailing.u8().unwrap(), b'a');
        assert!(trailing.finish().is_err());

        let mut bounded = DecodedAggregateSectionReader::new(payloads, 1, 1, 2).unwrap();
        assert!(bounded.skip(3).is_err());
        bounded.skip(2).unwrap();
        bounded.finish().unwrap();
    }
}
