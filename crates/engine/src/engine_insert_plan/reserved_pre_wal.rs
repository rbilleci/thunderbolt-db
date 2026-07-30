//! Terminal, inert owner for one completely admitted typed INSERT before canonical WAL.
//!
//! This module deliberately has no WAL append, apply, mutation, publication, status insertion, or
//! acknowledgement method.  It proves the ownership boundary first: exact codec/WAL buffers,
//! terminal coordinator payloads, and the complete private indexed physical reservation all live
//! beneath one capacity lease.  The lease is declared last so every retained resource drains
//! before its capacity is returned.

#![allow(dead_code)] // Production promotion remains closed until this checkpoint is accepted.

use super::pre_wal_capacity::{
    PreWalAggregateCapacityAcquireError, PreWalCapacityLease, PreWalCapacityPool,
};
use super::pre_wal_footprint::{
    AggregateTypedInsertShape, IndexedPreWalPhysicalForecast, PreWalOverlayLink,
    PreWalTransactionOwnerIdentity,
};
use super::IndexedPhysicalMaterializationPermit;
use crate::typed_insert_aggregate::{
    encode_reserved_typed_insert_canonical_envelope, encode_typed_insert_aggregate_bodies,
    reserve_typed_insert_aggregate_bodies, reserve_typed_insert_canonical_envelope,
    typed_insert_aggregate_status_roots, EncodedTypedInsertCanonicalEnvelope,
    TypedInsertAggregateView, TypedInsertStatusV2,
};
use crate::{EngineError, ExecuteError};

#[derive(Debug)]
#[allow(clippy::large_enum_variant)] // Capacity rejection must retain the exact move-only request without allocation.
pub(super) enum ReserveInsertPreWalPlanError {
    Capacity(PreWalAggregateCapacityAcquireError),
    Identity(&'static str),
    Encoding(EngineError),
    Physical(ExecuteError),
}

impl std::fmt::Display for ReserveInsertPreWalPlanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capacity(error) => write!(formatter, "pre-WAL capacity: {}", error.cause()),
            Self::Identity(message) => write!(formatter, "pre-WAL identity: {message}"),
            Self::Encoding(error) => write!(formatter, "pre-WAL encoding: {error}"),
            Self::Physical(error) => write!(formatter, "pre-WAL physical preparation: {error}"),
        }
    }
}

impl std::error::Error for ReserveInsertPreWalPlanError {}

/// One exact, non-authoritative coordinator payload reservation.
///
/// Bytes and logical slots are deliberately inseparable even when a zero-byte domain has zero
/// slots. These buffers are not live map/queue entries and expose no insertion capability.
struct ReservedTerminalDomain {
    bytes: Box<[u8]>,
    slots: u64,
}

impl ReservedTerminalDomain {
    fn exact(bytes: u64, slots: u64) -> Result<Self, EngineError> {
        if (bytes == 0) != (slots == 0) {
            return Err(EngineError::Durability(
                "typed INSERT terminal byte/slot geometry is inconsistent".to_string(),
            ));
        }
        let len = usize::try_from(bytes).map_err(|_| {
            EngineError::Durability(
                "typed INSERT terminal payload exceeds addressable host memory".to_string(),
            )
        })?;
        Ok(Self {
            bytes: zeroed_box(len),
            slots,
        })
    }
}

/// The sole inert pre-WAL owner. `P` is the opaque physical reservation returned only after the
/// terminal module issues its unforgeable materialization permit.
#[must_use]
pub(crate) struct ReservedInsertPreWalPlan<P> {
    // Drop order is load-bearing. The physical reservation may own CUDA launches, allocations,
    // mutation/lifecycle locks, and generation pins; it must drain before any accounting credit.
    physical: P,
    envelope: EncodedTypedInsertCanonicalEnvelope,
    status_index: ReservedTerminalDomain,
    response: ReservedTerminalDomain,
    publication: ReservedTerminalDomain,
    publication_join: ReservedTerminalDomain,
    completion: ReservedTerminalDomain,
    owner: PreWalTransactionOwnerIdentity,
    final_overlay: PreWalOverlayLink,
    first_statement_ordinal: u32,
    last_statement_ordinal: u32,
    // Keep this last: Rust drops struct fields in declaration order.
    lease: PreWalCapacityLease,
}

/// Build the full private candidate under one aggregate capacity lease.
///
/// `materialize` is called exactly once and only after every exact host/WAL/terminal allocation
/// succeeds. Its only argument is the capability residency requires before allocating or
/// launching indexed physical work.
#[allow(clippy::result_large_err)] // Exact capacity retry remains inline and move-only.
#[allow(clippy::too_many_arguments)] // One terminal call binds every independently typed authority.
pub(crate) fn reserve_indexed_insert_pre_wal_plan<P>(
    pool: &PreWalCapacityPool,
    aggregate: AggregateTypedInsertShape,
    view: &TypedInsertAggregateView<'_>,
    status: &TypedInsertStatusV2,
    physical_range: gpu_db_wal::CanonicalPhysicalRange,
    header: gpu_db_wal::CanonicalPreApplyHeader,
    outcome: gpu_db_wal::CanonicalOutcome,
    physical_forecast: &IndexedPreWalPhysicalForecast,
    materialize: impl FnOnce(IndexedPhysicalMaterializationPermit) -> Result<P, ExecuteError>,
) -> Result<ReservedInsertPreWalPlan<P>, ReserveInsertPreWalPlanError> {
    reserve_indexed_insert_pre_wal_plan_inner(
        pool,
        aggregate,
        view,
        status,
        physical_range,
        header,
        outcome,
        physical_forecast,
        ReservationFault::None,
        materialize,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReservationFault {
    None,
    AfterLease,
    AfterBodies,
    AfterEnvelope,
    BeforePhysical,
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)] // Exact capacity retry remains inline and move-only.
fn reserve_indexed_insert_pre_wal_plan_inner<P>(
    pool: &PreWalCapacityPool,
    aggregate: AggregateTypedInsertShape,
    view: &TypedInsertAggregateView<'_>,
    status: &TypedInsertStatusV2,
    physical_range: gpu_db_wal::CanonicalPhysicalRange,
    header: gpu_db_wal::CanonicalPreApplyHeader,
    outcome: gpu_db_wal::CanonicalOutcome,
    physical_forecast: &IndexedPreWalPhysicalForecast,
    fault: ReservationFault,
    materialize: impl FnOnce(IndexedPhysicalMaterializationPermit) -> Result<P, ExecuteError>,
) -> Result<ReservedInsertPreWalPlan<P>, ReserveInsertPreWalPlanError> {
    let measured = view
        .measure()
        .map_err(|_| ReserveInsertPreWalPlanError::Identity("aggregate view does not measure"))?;
    if aggregate.layout() != &measured
        || aggregate
            .admission_binding()
            .indexed_physical()
            .is_none_or(|admitted| admitted != physical_forecast)
    {
        return Err(ReserveInsertPreWalPlanError::Identity(
            "aggregate layout or indexed forecast drifted before admission",
        ));
    }
    // Root/status closure is allocation-free and must reject drift before capacity is acquired.
    let roots = typed_insert_aggregate_status_roots(view, &measured)
        .map_err(ReserveInsertPreWalPlanError::Encoding)?;
    if status.aggregate_root != roots.aggregate_root
        || status.statement_outcome_root != roots.statement_outcome_root
        || status.response_root != roots.response_root
    {
        return Err(ReserveInsertPreWalPlanError::Identity(
            "aggregate STATUS2 roots drifted before admission",
        ));
    }

    let admitted = pool
        .try_acquire_aggregate(aggregate)
        .map_err(ReserveInsertPreWalPlanError::Capacity)?;
    // This probe is declared before every envelope/terminal owner below. On a materialization
    // error or unwind its Drop runs after those owners but while `admitted` still owns the lease.
    #[cfg(test)]
    let _resource_drop_window_probe = take_resource_drop_window_probe();
    injected(fault, ReservationFault::AfterLease)?;

    let reserved_bodies = reserve_typed_insert_aggregate_bodies(measured)
        .map_err(ReserveInsertPreWalPlanError::Encoding)?;
    let bodies = encode_typed_insert_aggregate_bodies(view, status, reserved_bodies)
        .map_err(ReserveInsertPreWalPlanError::Encoding)?;
    injected(fault, ReservationFault::AfterBodies)?;

    let reserved_envelope =
        reserve_typed_insert_canonical_envelope(bodies, physical_range, header, outcome)
            .map_err(ReserveInsertPreWalPlanError::Encoding)?;
    let envelope = encode_reserved_typed_insert_canonical_envelope(reserved_envelope)
        .map_err(ReserveInsertPreWalPlanError::Encoding)?;
    injected(fault, ReservationFault::AfterEnvelope)?;

    let footprint = admitted.footprint();
    let status_index =
        ReservedTerminalDomain::exact(footprint.status_index_bytes, footprint.status_index_slots)
            .map_err(ReserveInsertPreWalPlanError::Encoding)?;
    let response = ReservedTerminalDomain::exact(
        footprint.terminal_response_bytes,
        footprint.terminal_response_slots,
    )
    .map_err(ReserveInsertPreWalPlanError::Encoding)?;
    let publication = ReservedTerminalDomain::exact(
        footprint.publication_bytes,
        footprint.publication_target_slots,
    )
    .map_err(ReserveInsertPreWalPlanError::Encoding)?;
    let publication_join = ReservedTerminalDomain::exact(
        footprint.publication_join_entry_bytes,
        footprint.publication_join_entry_slots,
    )
    .map_err(ReserveInsertPreWalPlanError::Encoding)?;
    let completion =
        ReservedTerminalDomain::exact(footprint.completion_bytes, footprint.completion_slots)
            .map_err(ReserveInsertPreWalPlanError::Encoding)?;
    injected(fault, ReservationFault::BeforePhysical)?;

    let physical = materialize(IndexedPhysicalMaterializationPermit {
        _only_engine_insert_plan_may_issue: (),
    })
    .map_err(ReserveInsertPreWalPlanError::Physical)?;
    // Do not split the aggregate owner until physical materialization has succeeded. Until this
    // point an Err or unwind drops the complete aggregate after every envelope/terminal owner,
    // so the capacity lease cannot be released into a live resource-drop window.
    let (lease, binding) = admitted.into_parts();
    let (first_statement_ordinal, last_statement_ordinal) = binding.statement_ordinals();
    Ok(ReservedInsertPreWalPlan {
        physical,
        envelope,
        status_index,
        response,
        publication,
        publication_join,
        completion,
        owner: binding.owner().clone(),
        final_overlay: binding.overlay_span().clone(),
        first_statement_ordinal,
        last_statement_ordinal,
        lease,
    })
}

#[allow(clippy::result_large_err)] // Shares the terminal error type for test-only fault injection.
fn injected(
    #[cfg_attr(not(test), allow(unused_variables))] actual: ReservationFault,
    #[cfg_attr(not(test), allow(unused_variables))] expected: ReservationFault,
) -> Result<(), ReserveInsertPreWalPlanError> {
    #[cfg(test)]
    if actual == expected {
        return Err(ReserveInsertPreWalPlanError::Identity(
            "injected inert pre-WAL reservation failure",
        ));
    }
    Ok(())
}

fn zeroed_box(len: usize) -> Box<[u8]> {
    let mut bytes = Box::<[u8]>::new_uninit_slice(len);
    for byte in bytes.iter_mut() {
        byte.write(0);
    }
    // SAFETY: every u8 slot is initialized above.
    unsafe { bytes.assume_init() }
}

#[cfg(test)]
impl<P> ReservedInsertPreWalPlan<P> {
    fn inspect<R>(
        self,
        inspect: impl FnOnce(P, ReservedInsertPreWalPlanReport) -> Result<R, ExecuteError>,
    ) -> Result<R, ExecuteError> {
        let Self {
            physical,
            envelope,
            status_index,
            response,
            publication,
            publication_join,
            completion,
            owner,
            final_overlay,
            first_statement_ordinal,
            last_statement_ordinal,
            lease,
        } = self;
        let report = ReservedInsertPreWalPlanReport {
            packed_wal_bytes: envelope.packed_payload().len() as u64,
            serialized_wal_bytes: envelope.serialized_record().len() as u64,
            status_index_bytes: status_index.bytes.len() as u64,
            status_index_slots: status_index.slots,
            response_bytes: response.bytes.len() as u64,
            response_slots: response.slots,
            publication_bytes: publication.bytes.len() as u64,
            publication_slots: publication.slots,
            publication_join_bytes: publication_join.bytes.len() as u64,
            publication_join_slots: publication_join.slots,
            completion_bytes: completion.bytes.len() as u64,
            completion_slots: completion.slots,
            owner,
            final_overlay,
            first_statement_ordinal,
            last_statement_ordinal,
        };
        // The lease and every encoded/terminal owner remain alive while physical inspection
        // consumes the private reservation.
        let result = inspect(physical, report);
        drop(envelope);
        drop(status_index);
        drop(response);
        drop(publication);
        drop(publication_join);
        drop(completion);
        drop(lease);
        result
    }
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
struct ReservedInsertPreWalPlanReport {
    packed_wal_bytes: u64,
    serialized_wal_bytes: u64,
    status_index_bytes: u64,
    status_index_slots: u64,
    response_bytes: u64,
    response_slots: u64,
    publication_bytes: u64,
    publication_slots: u64,
    publication_join_bytes: u64,
    publication_join_slots: u64,
    completion_bytes: u64,
    completion_slots: u64,
    owner: PreWalTransactionOwnerIdentity,
    final_overlay: PreWalOverlayLink,
    first_statement_ordinal: u32,
    last_statement_ordinal: u32,
}

#[cfg(test)]
struct ResourceDropWindowProbe {
    resources_dropped: std::sync::mpsc::Sender<()>,
    release: std::sync::mpsc::Receiver<()>,
}

#[cfg(test)]
impl Drop for ResourceDropWindowProbe {
    fn drop(&mut self) {
        // This local is declared before the envelope and terminal-domain locals. Rust drops
        // locals in reverse declaration order, so their owned buffers have drained before this
        // test-only handoff, while the earlier `admitted` aggregate still retains its lease.
        let _ = self.resources_dropped.send(());
        let _ = self.release.recv();
    }
}

#[cfg(test)]
thread_local! {
    static RESOURCE_DROP_WINDOW_PROBE: std::cell::RefCell<Option<ResourceDropWindowProbe>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn arm_resource_drop_window_probe(
    resources_dropped: std::sync::mpsc::Sender<()>,
    release: std::sync::mpsc::Receiver<()>,
) {
    RESOURCE_DROP_WINDOW_PROBE.with(|probe| {
        assert!(
            probe
                .replace(Some(ResourceDropWindowProbe {
                    resources_dropped,
                    release,
                }))
                .is_none(),
            "resource-drop window probe was already armed"
        );
    });
}

#[cfg(test)]
fn take_resource_drop_window_probe() -> Option<ResourceDropWindowProbe> {
    RESOURCE_DROP_WINDOW_PROBE.with(|probe| probe.replace(None))
}

/// Real indexed GPU proof adapter. It supplies only a minimal one-statement logical aggregate;
/// the physical forecast, exact capacity, codec/WAL buffers, terminal slots, and permit ordering
/// are the production-compiled owners above. No live engine route can call this test capability.
#[cfg(test)]
pub(super) fn inspect_test_indexed_autocommit<P, R>(
    physical_forecast: IndexedPreWalPhysicalForecast,
    allocator_before: u64,
    allocator_high_water: u64,
    inserted_rows: u32,
    materialize: impl FnOnce(IndexedPhysicalMaterializationPermit) -> Result<P, ExecuteError>,
    inspect: impl FnOnce(P) -> Result<R, ExecuteError>,
) -> Result<R, ExecuteError> {
    use super::pre_wal_footprint::{
        LogicalStatementContribution, PreWalLogicalFootprintInput, PreWalPublicationGeometry,
        PreWalStatementIdentity,
    };
    use crate::typed_insert_aggregate::{
        TypedInsertAggregateSectionView, AGGREGATE_FLAG_AUTOCOMMIT, OUTER_CONTENT_ROW,
        OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1,
    };

    if inserted_rows == 0
        || allocator_before == 0
        || allocator_high_water.checked_sub(allocator_before) != Some(u64::from(inserted_rows))
    {
        return Err(ExecuteError::Serialization(
            "indexed pre-WAL proof allocator geometry drifted".to_string(),
        ));
    }
    let before_generation = physical_forecast.generation.predecessor_boundary;
    let after_generation = before_generation.checked_add(1).ok_or_else(|| {
        ExecuteError::Serialization(
            "indexed pre-WAL proof overlay generation overflows".to_string(),
        )
    })?;
    let mut before_seed = [0_u8; 48];
    before_seed[..32].copy_from_slice(&physical_forecast.target.schema_digest);
    before_seed[32..40].copy_from_slice(&before_generation.to_le_bytes());
    before_seed[40..48].copy_from_slice(b"BEFOREV1");
    let before_root = gpu_db_wal::canonical_request_digest(&before_seed);
    let mut after_seed = [0_u8; 56];
    after_seed[..32].copy_from_slice(&physical_forecast.target.schema_digest);
    after_seed[32..40].copy_from_slice(&after_generation.to_le_bytes());
    after_seed[40..48].copy_from_slice(&allocator_high_water.to_le_bytes());
    after_seed[48..56].copy_from_slice(b"AFTER_V1");
    let after_root = gpu_db_wal::canonical_request_digest(&after_seed);
    let row_id_bytes = u64::from(inserted_rows).checked_mul(8).ok_or_else(|| {
        ExecuteError::Serialization("indexed pre-WAL proof row-id bytes overflow".to_string())
    })?;
    let logical = LogicalStatementContribution::new(
        PreWalLogicalFootprintInput {
            host_plan_retained_bytes: 0,
            host_allocation_slots: 0,
            typed_shadow_retained_bytes: 0,
            host_generation_pin_slots: 0,
            host_scratch_peak_bytes: 0,
            statement_identity: PreWalStatementIdentity {
                owner: PreWalTransactionOwnerIdentity {
                    txn_id: allocator_before,
                    registration_nonce: after_generation,
                },
                overlay_link: PreWalOverlayLink {
                    before_generation,
                    after_generation,
                    before_root_digest: before_root,
                    after_root_digest: after_root,
                },
                statement_ordinal: 0,
            },
            row_id_bytes,
            row_id_slots: u64::from(inserted_rows),
            sequence_effect_bytes: 0,
            sequence_effect_slots: 0,
            terminal_response_bytes: 0,
            terminal_response_slots: 0,
        },
        1,
    )
    .map_err(|error| {
        ExecuteError::Serialization(format!("indexed pre-WAL proof logical footprint: {error}"))
    })?;

    let empty = &[][..];
    let counts = [1, 1, 0, inserted_rows, 0, 1, 1, 0];
    let view = TypedInsertAggregateView {
        flags: AGGREGATE_FLAG_AUTOCOMMIT,
        outer_flags: OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
        stable_transaction_id: allocator_before,
        statement_count: 1,
        insert_statement_count: 1,
        original_inserted_row_count: u64::from(inserted_rows),
        final_row_transition_count: u64::from(inserted_rows),
        allocator_before,
        allocator_high_water,
        table_block_count: 1,
        sections: std::array::from_fn(|index| TypedInsertAggregateSectionView {
            entry_count: counts[index],
            payload: empty,
        }),
    };
    let layout = view.measure().map_err(|error| {
        ExecuteError::Serialization(format!("indexed pre-WAL proof aggregate layout: {error}"))
    })?;
    let shape = logical
        .bind_single_indexed_final_overlay(
            physical_forecast.clone(),
            PreWalPublicationGeometry {
                bytes: 32,
                slots: 1,
            },
        )
        .and_then(|decision| decision.bind_typed_insert_aggregate_layout(layout))
        .map_err(|error| {
            ExecuteError::Serialization(format!("indexed pre-WAL proof final footprint: {error}"))
        })?;
    let pool = exact_test_capacity_pool(&shape)?;
    let roots = typed_insert_aggregate_status_roots(&view, &layout)?;
    let request_digest = physical_forecast.target.schema_digest;
    let status = TypedInsertStatusV2 {
        database_id: [1; 16],
        timeline_id: [2; 16],
        txn_id: allocator_before,
        request_digest,
        isolation: 1,
        flags: 0,
        retention_deadline: 0,
        statement_count: 1,
        response_artifact_count: 0,
        statement_outcome_root: roots.statement_outcome_root,
        response_root: roots.response_root,
        aggregate_root: roots.aggregate_root,
    };
    let physical_range = gpu_db_wal::CanonicalPhysicalRange {
        log_epoch: physical_forecast.generation.catalog_seq.max(1),
        lane_id: u32::from(physical_forecast.target.gpu_id),
        segment_id: u64::from(physical_forecast.generation.open_shard_id.max(1)),
        first_frame_ordinal: 1,
    };
    let header = gpu_db_wal::CanonicalPreApplyHeader {
        identity: gpu_db_wal::CanonicalIdentity {
            database_id: status.database_id,
            cluster_id: [3; 16],
            timeline_id: status.timeline_id,
            format_epoch: 1,
        },
        leader_epoch: physical_forecast.generation.predecessor_boundary.max(1),
        commit_seq: physical_forecast.generation.predecessor_boundary,
        stable_transaction_id: allocator_before,
        request_digest,
        isolation: gpu_db_wal::CanonicalIsolation::ReadCommitted,
        flags: view.outer_flags,
        catalog_before_epoch: physical_forecast.generation.catalog_seq,
        catalog_after_epoch: physical_forecast.generation.catalog_seq,
        catalog_before_digest: physical_forecast.target.schema_digest,
        catalog_after_digest: physical_forecast.target.schema_digest,
        operation_count: layout.fragment_count,
        table_block_count: 1,
        allocator_high_water,
    };
    let outcome = gpu_db_wal::CanonicalOutcome {
        kind: gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
        // The one autocommit statement's terminal outcome reports its inserted rows.
        affected_rows: u64::from(inserted_rows),
        sqlstate: None,
        constraint_id: 0,
        target_digest: roots.aggregate_root,
        returning_digest: roots.response_root,
    };
    let plan = reserve_indexed_insert_pre_wal_plan(
        &pool,
        shape,
        &view,
        &status,
        physical_range,
        header,
        outcome,
        &physical_forecast,
        materialize,
    )
    .map_err(|error| {
        ExecuteError::Serialization(format!("indexed pre-WAL terminal reservation: {error}"))
    })?;
    plan.inspect(|physical, _| inspect(physical))
}

#[cfg(test)]
fn exact_test_capacity_pool(
    shape: &AggregateTypedInsertShape,
) -> Result<PreWalCapacityPool, ExecuteError> {
    use super::pre_wal_capacity::{
        PreWalAuthoritativeGpuResidency, PreWalCapacityLimits, PreWalGpuCapacityLimit,
    };
    let footprint = shape.footprint();
    let mut cursor = footprint.gpu_pool_cursor();
    let gpu = cursor
        .try_next()
        .map_err(|error| ExecuteError::Serialization(error.to_string()))?
        .ok_or_else(|| {
            ExecuteError::Serialization("indexed pre-WAL proof has no GPU pool".to_string())
        })?;
    if cursor
        .try_next()
        .map_err(|error| ExecuteError::Serialization(error.to_string()))?
        .is_some()
    {
        return Err(ExecuteError::Serialization(
            "indexed pre-WAL proof unexpectedly spans multiple GPU pools".to_string(),
        ));
    }
    let generation_envelope = gpu
        .old_generation_pinned_bytes
        .checked_add(gpu.new_persistent_bytes)
        .ok_or_else(|| {
            ExecuteError::Serialization(
                "indexed pre-WAL proof generation envelope overflows".to_string(),
            )
        })?;
    let mut limits = PreWalCapacityLimits {
        host_retained_bytes: footprint
            .global_host_retained_bytes()
            .map_err(|error| ExecuteError::Serialization(error.to_string()))?,
        host_allocation_slots: footprint.host_allocation_slots,
        host_scratch_peak_bytes: footprint.host_scratch_peak_bytes,
        host_generation_pin_slots: footprint.host_generation_pin_slots,
        wal_packed_record_bytes: footprint.wal_packed_record_bytes,
        wal_serialized_record_bytes: footprint.wal_serialized_record_bytes,
        wal_record_slots: footprint.wal_record_slots,
        wal_frame_slots: footprint.wal_frame_slots,
        row_id_slots: footprint.row_id_slots,
        sequence_effect_slots: footprint.sequence_effect_slots,
        status_index_bytes: footprint.status_index_bytes,
        status_index_slots: footprint.status_index_slots,
        terminal_response_bytes: footprint.terminal_response_bytes,
        terminal_response_slots: footprint.terminal_response_slots,
        completion_bytes: footprint.completion_bytes,
        completion_slots: footprint.completion_slots,
        publication_bytes: footprint.publication_bytes,
        publication_target_slots: footprint.publication_target_slots,
        publication_join_entry_bytes: footprint.publication_join_entry_bytes,
        publication_join_entry_slots: footprint.publication_join_entry_slots,
        gpus: vec![PreWalGpuCapacityLimit {
            gpu_id: gpu.gpu_id,
            hard_device_bytes: gpu
                .old_generation_pinned_bytes
                .checked_add(
                    gpu.incremental_device_peak_bytes()
                        .map_err(|error| ExecuteError::Serialization(error.to_string()))?,
                )
                .ok_or_else(|| {
                    ExecuteError::Serialization(
                        "indexed pre-WAL proof hard device limit overflows".to_string(),
                    )
                })?,
            plan_retained_transient_bytes: gpu.plan_retained_transient_bytes,
            result_retained_device_bytes: gpu.result_retained_device_bytes,
            scratch_peak_bytes: gpu.scratch_peak_bytes,
            allocation_slots: gpu.allocation_slots,
            generation_pin_slots: gpu.generation_pin_slots,
            generation_envelope_bytes: Some(generation_envelope),
        }]
        .into(),
    };
    TEST_NEXT_HOST_SCRATCH_LIMIT.with(|slot| {
        if let Some(limit) = slot.replace(None) {
            limits.host_scratch_peak_bytes = limit;
        }
    });
    PreWalCapacityPool::new(
        limits,
        vec![PreWalAuthoritativeGpuResidency {
            gpu_id: gpu.gpu_id,
            bytes: gpu.old_generation_pinned_bytes,
        }]
        .into(),
    )
    .map_err(|error| ExecuteError::Serialization(error.to_string()))
}

#[cfg(test)]
thread_local! {
    static TEST_NEXT_HOST_SCRATCH_LIMIT: std::cell::Cell<Option<u64>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn limit_next_test_indexed_host_scratch_to(bytes: u64) {
    TEST_NEXT_HOST_SCRATCH_LIMIT.with(|slot| {
        assert!(
            slot.replace(Some(bytes)).is_none(),
            "indexed host-scratch sabotage was already armed"
        );
    });
}

#[cfg(test)]
#[path = "reserved_pre_wal_tests.rs"]
mod tests;
