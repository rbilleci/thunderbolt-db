//! Inert, all-or-nothing capacity admission for one current pre-WAL footprint.
//!
//! This leaf owns no GPU allocation and has no engine caller. It only reserves scalar demand
//! against a fixed authoritative-residency baseline; a future live owner must share the residency
//! allocation lock before it can move this credit into an authoritative generation.

#![allow(dead_code)] // Inert PLAN foundation; no live database field or caller owns this pool yet.

use super::pre_wal_footprint::{
    AggregateTypedInsertAdmissionBinding, AggregateTypedInsertShape, CurrentTwoFragmentShape,
    PreWalFootprint, PreWalGpuPoolFootprint,
};
use std::sync::{Arc, Mutex};

/// Immutable per-GPU limits. `generation_envelope_bytes` is a per-request old-plus-new evidence
/// ceiling, never an additive pool resource: old authoritative residency is not double charged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PreWalGpuCapacityLimit {
    pub(super) gpu_id: u16,
    pub(super) hard_device_bytes: u64,
    pub(super) plan_retained_transient_bytes: u64,
    pub(super) result_retained_device_bytes: u64,
    pub(super) scratch_peak_bytes: u64,
    pub(super) allocation_slots: u64,
    pub(super) generation_pin_slots: u64,
    pub(super) generation_envelope_bytes: Option<u64>,
}

/// The scalar budget for every capacity domain owned by one pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreWalCapacityLimits {
    pub(super) host_retained_bytes: u64,
    pub(super) host_allocation_slots: u64,
    pub(super) host_scratch_peak_bytes: u64,
    pub(super) host_generation_pin_slots: u64,
    pub(super) wal_packed_record_bytes: u64,
    pub(super) wal_serialized_record_bytes: u64,
    pub(super) wal_record_slots: u64,
    pub(super) wal_frame_slots: u64,
    pub(super) row_id_slots: u64,
    pub(super) sequence_effect_slots: u64,
    pub(super) status_index_bytes: u64,
    pub(super) status_index_slots: u64,
    pub(super) terminal_response_bytes: u64,
    pub(super) terminal_response_slots: u64,
    pub(super) completion_bytes: u64,
    pub(super) completion_slots: u64,
    pub(super) publication_bytes: u64,
    pub(super) publication_target_slots: u64,
    pub(super) publication_join_entry_bytes: u64,
    pub(super) publication_join_entry_slots: u64,
    pub(super) gpus: Box<[PreWalGpuCapacityLimit]>,
}

/// Fixed, externally sampled authoritative device residency for the inert pool's lifetime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreWalAuthoritativeGpuResidency {
    pub(super) gpu_id: u16,
    pub(super) bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PreWalCapacityError {
    EmptyGpuLimits,
    GpuLimitOrderDrift(u16),
    DuplicateGpuLimit(u16),
    AuthoritativeResidencyOrderDrift(u16),
    DuplicateAuthoritativeResidency(u16),
    MissingAuthoritativeResidency(u16),
    UnexpectedAuthoritativeResidency(u16),
    AuthoritativeResidencyExceedsHard { gpu_id: u16, bytes: u64, hard: u64 },
    UnknownGpuDemand(u16),
    GenerationEnvelopeExceeded { gpu_id: u16, bytes: u64, limit: u64 },
    ResourceExhausted(&'static str),
    Overflow(&'static str),
}

impl std::fmt::Display for PreWalCapacityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyGpuLimits => formatter.write_str("capacity pool has no GPU limits"),
            Self::GpuLimitOrderDrift(gpu_id) => {
                write!(
                    formatter,
                    "GPU capacity limits are not ordered at GPU {gpu_id}"
                )
            }
            Self::DuplicateGpuLimit(gpu_id) => {
                write!(
                    formatter,
                    "GPU capacity limit appears twice for GPU {gpu_id}"
                )
            }
            Self::AuthoritativeResidencyOrderDrift(gpu_id) => write!(
                formatter,
                "authoritative residency snapshot is not ordered at GPU {gpu_id}"
            ),
            Self::DuplicateAuthoritativeResidency(gpu_id) => write!(
                formatter,
                "authoritative residency snapshot appears twice for GPU {gpu_id}"
            ),
            Self::MissingAuthoritativeResidency(gpu_id) => {
                write!(
                    formatter,
                    "GPU {gpu_id} has no authoritative residency snapshot"
                )
            }
            Self::UnexpectedAuthoritativeResidency(gpu_id) => write!(
                formatter,
                "authoritative residency snapshot names unconfigured GPU {gpu_id}"
            ),
            Self::AuthoritativeResidencyExceedsHard {
                gpu_id,
                bytes,
                hard,
            } => write!(
                formatter,
                "authoritative GPU {gpu_id} residency {bytes} exceeds hard limit {hard}"
            ),
            Self::UnknownGpuDemand(gpu_id) => {
                write!(
                    formatter,
                    "current footprint names unconfigured GPU {gpu_id}"
                )
            }
            Self::GenerationEnvelopeExceeded {
                gpu_id,
                bytes,
                limit,
            } => write!(
                formatter,
                "GPU {gpu_id} generation evidence {bytes} exceeds envelope {limit}"
            ),
            Self::ResourceExhausted(domain) => {
                write!(formatter, "pre-WAL capacity exhausted {domain}")
            }
            Self::Overflow(domain) => write!(formatter, "pre-WAL capacity overflows {domain}"),
        }
    }
}

impl std::error::Error for PreWalCapacityError {}

/// A rejected admission retains the exact move-only request, so the caller may retry it.
#[derive(Debug)]
#[must_use]
pub(super) struct PreWalCapacityAcquireError {
    cause: PreWalCapacityError,
    request: PreWalCapacityRequest,
}

/// Aggregate rejection retains both the exact scalar request and the immutable owner/layout/
/// physical binding. Retrying therefore cannot remeasure, rebuild, or substitute a forecast.
#[derive(Debug)]
#[must_use]
pub(super) struct PreWalAggregateCapacityAcquireError {
    cause: PreWalCapacityError,
    request: PreWalCapacityRequest,
    binding: AggregateTypedInsertAdmissionBinding,
}

/// A rejected explicit-overlay replacement preserves both move-only inputs: the incumbent lease
/// still owns its exact admitted demand and the candidate request may be repaired or retried.
#[derive(Debug)]
#[must_use]
pub(super) struct PreWalCapacityReplaceError {
    cause: PreWalCapacityError,
    lease: PreWalCapacityLease,
    candidate: PreWalCapacityRequest,
}

impl PreWalCapacityReplaceError {
    pub(super) fn cause(&self) -> &PreWalCapacityError {
        &self.cause
    }

    pub(super) fn into_parts(self) -> (PreWalCapacityLease, PreWalCapacityRequest) {
        (self.lease, self.candidate)
    }
}

impl PreWalCapacityAcquireError {
    pub(super) fn cause(&self) -> &PreWalCapacityError {
        &self.cause
    }

    pub(super) fn into_request(self) -> PreWalCapacityRequest {
        self.request
    }
}

impl PreWalAggregateCapacityAcquireError {
    pub(super) fn cause(&self) -> &PreWalCapacityError {
        &self.cause
    }

    /// Retry the exact rejected aggregate without reconstructing its codec or physical witness.
    #[allow(clippy::result_large_err)] // Exact retry retains the sole move-only admission inputs.
    pub(super) fn retry(
        self,
        pool: &PreWalCapacityPool,
    ) -> Result<AdmittedAggregateTypedInsert, Self> {
        pool.try_acquire_aggregate_parts(self.request, self.binding)
    }
}

/// The only current-statement capacity handoff. It owns the complete footprint and derived
/// scalar demand until the pool consumes it into a Drop-only lease.
#[derive(Debug)]
#[must_use]
pub(super) struct PreWalCapacityRequest {
    footprint: PreWalFootprint,
    demand: PreWalCapacityDemand,
}

/// A successful admission. It deliberately exposes only a borrow of its sole footprint.
#[derive(Debug)]
#[must_use]
pub(super) struct PreWalCapacityLease {
    state: Arc<Mutex<PreWalCapacityState>>,
    request: Option<PreWalCapacityRequest>,
}

/// Successful aggregate admission. Unlike the generic scalar lease, this owner retains the
/// codec-5 layout, transaction/overlay identity, and indexed physical forecast that were admitted.
#[derive(Debug)]
#[must_use]
pub(super) struct AdmittedAggregateTypedInsert {
    lease: PreWalCapacityLease,
    binding: AggregateTypedInsertAdmissionBinding,
}

impl AdmittedAggregateTypedInsert {
    pub(super) fn footprint(&self) -> &PreWalFootprint {
        self.lease.footprint()
    }

    pub(super) fn binding(&self) -> &AggregateTypedInsertAdmissionBinding {
        &self.binding
    }

    pub(super) fn into_parts(self) -> (PreWalCapacityLease, AggregateTypedInsertAdmissionBinding) {
        (self.lease, self.binding)
    }
}

/// Two-phase explicit-overlay capacity replacement. While armed, the ledger owns the incumbent
/// demand plus every positive candidate delta, so candidate resources may be materialized and
/// revalidated without releasing capacity to another transaction.
#[derive(Debug)]
#[must_use]
pub(super) struct PreWalCapacityReplacementGuard {
    incumbent: Option<PreWalCapacityLease>,
    candidate: Option<PreWalCapacityRequest>,
    headroom_armed: bool,
}

impl PreWalCapacityLease {
    pub(super) fn footprint(&self) -> &PreWalFootprint {
        &self
            .request
            .as_ref()
            .expect("live capacity lease always owns its request")
            .footprint
    }

    /// Reserve every positive candidate delta while preserving the incumbent aggregate. The
    /// returned guard must stay alive through candidate materialization and final-overlay
    /// currentness validation.
    #[allow(clippy::result_large_err)] // Failure must preserve both move-only owners for retry.
    pub(super) fn begin_replace(
        self,
        candidate: PreWalCapacityRequest,
    ) -> Result<PreWalCapacityReplacementGuard, PreWalCapacityReplaceError> {
        let mut state = lock_recover(&self.state);
        let incumbent = self
            .request
            .as_ref()
            .expect("live capacity lease always owns its request");
        if let Err(cause) = state.validate_replacement_headroom(incumbent, &candidate) {
            drop(state);
            return Err(PreWalCapacityReplaceError {
                cause,
                lease: self,
                candidate,
            });
        }
        state.apply_replacement_headroom(incumbent, &candidate);
        drop(state);
        Ok(PreWalCapacityReplacementGuard {
            incumbent: Some(self),
            candidate: Some(candidate),
            headroom_armed: true,
        })
    }

    /// Convenience for an already-materialized candidate. New construction code must use
    /// `begin_replace`, retain the guard while building, and call `commit` only after revalidation.
    #[allow(clippy::result_large_err)] // Failure must preserve both move-only owners for retry.
    pub(super) fn try_replace(
        self,
        candidate: PreWalCapacityRequest,
    ) -> Result<Self, PreWalCapacityReplaceError> {
        Ok(self.begin_replace(candidate)?.commit())
    }
}

impl PreWalCapacityReplacementGuard {
    /// Install the candidate accounting after its complete owner and overlay have been proven.
    /// This is infallible: `begin_replace` already admitted the simultaneous peak.
    pub(super) fn commit(mut self) -> PreWalCapacityLease {
        let mut incumbent = self
            .incumbent
            .take()
            .expect("armed replacement owns its incumbent lease");
        let candidate = self
            .candidate
            .take()
            .expect("armed replacement owns its candidate request");
        let old = incumbent
            .request
            .as_ref()
            .expect("replacement incumbent owns its admitted request");
        let mut state = lock_recover(&incumbent.state);
        state.commit_replacement(old, &candidate);
        drop(state);
        incumbent.request = Some(candidate);
        self.headroom_armed = false;
        incumbent
    }

    /// Cancel normal candidate construction without disturbing the incumbent aggregate.
    pub(super) fn abort(mut self) -> (PreWalCapacityLease, PreWalCapacityRequest) {
        let incumbent = self
            .incumbent
            .take()
            .expect("armed replacement owns its incumbent lease");
        let candidate = self
            .candidate
            .take()
            .expect("armed replacement owns its candidate request");
        let old = incumbent
            .request
            .as_ref()
            .expect("replacement incumbent owns its admitted request");
        let mut state = lock_recover(&incumbent.state);
        state.release_replacement_headroom(old, &candidate);
        drop(state);
        self.headroom_armed = false;
        (incumbent, candidate)
    }
}

impl Drop for PreWalCapacityReplacementGuard {
    fn drop(&mut self) {
        if !self.headroom_armed {
            return;
        }
        let incumbent = self
            .incumbent
            .as_ref()
            .expect("armed replacement owns its incumbent lease");
        let candidate = self
            .candidate
            .as_ref()
            .expect("armed replacement owns its candidate request");
        let old = incumbent
            .request
            .as_ref()
            .expect("replacement incumbent owns its admitted request");
        let mut state = lock_recover(&incumbent.state);
        state.release_replacement_headroom(old, candidate);
        self.headroom_armed = false;
    }
}

impl Drop for PreWalCapacityLease {
    fn drop(&mut self) {
        let Some(request) = self.request.take() else {
            return;
        };
        let mut state = lock_recover(&self.state);
        state.release(&request);
    }
}

/// One-mutex pool. Admission never holds that mutex across GPU residency work because this leaf
/// has no allocation or publication authority.
#[derive(Clone)]
pub(super) struct PreWalCapacityPool {
    state: Arc<Mutex<PreWalCapacityState>>,
}

impl PreWalCapacityPool {
    pub(super) fn new(
        limits: PreWalCapacityLimits,
        authoritative_residency: Box<[PreWalAuthoritativeGpuResidency]>,
    ) -> Result<Self, PreWalCapacityError> {
        let gpus = validate_gpu_configuration(&limits, authoritative_residency)?;
        Ok(Self {
            state: Arc::new(Mutex::new(PreWalCapacityState {
                global: PreWalGlobalUsed::default(),
                gpus,
                limits,
            })),
        })
    }

    /// Consumes the exclusive current-shape route; explicit transactions consume their shape into
    /// logical aggregation instead and therefore cannot enter this current-format pool.
    #[allow(clippy::result_large_err)] // Failure returns the sole move-only request without a retry allocation.
    pub(super) fn try_acquire(
        &self,
        current: CurrentTwoFragmentShape,
    ) -> Result<PreWalCapacityLease, PreWalCapacityAcquireError> {
        self.try_acquire_request(current.into_capacity_request())
    }

    /// The distinct codec-5 aggregate entry point. An unresolved final overlay cannot construct
    /// `AggregateTypedInsertShape`, so it cannot reserve capacity using current-format geometry.
    #[allow(clippy::result_large_err)] // Failure returns the sole move-only request for retry.
    pub(super) fn try_acquire_aggregate(
        &self,
        aggregate: AggregateTypedInsertShape,
    ) -> Result<AdmittedAggregateTypedInsert, PreWalAggregateCapacityAcquireError> {
        let binding = aggregate.admission_binding();
        self.try_acquire_aggregate_parts(aggregate.into_capacity_request(), binding)
    }

    #[allow(clippy::result_large_err)] // Exact retry retains the sole move-only admission inputs.
    fn try_acquire_aggregate_parts(
        &self,
        request: PreWalCapacityRequest,
        binding: AggregateTypedInsertAdmissionBinding,
    ) -> Result<AdmittedAggregateTypedInsert, PreWalAggregateCapacityAcquireError> {
        let mut state = lock_recover(&self.state);
        if let Err(cause) = state.validate_reservation(&request) {
            return Err(PreWalAggregateCapacityAcquireError {
                cause,
                request,
                binding,
            });
        }
        state.apply_reservation(&request);
        drop(state);
        Ok(AdmittedAggregateTypedInsert {
            lease: PreWalCapacityLease {
                state: Arc::clone(&self.state),
                request: Some(request),
            },
            binding,
        })
    }

    #[cfg(test)]
    pub(super) fn is_empty_for_test(&self) -> bool {
        let state = lock_recover(&self.state);
        state.global == PreWalGlobalUsed::default()
            && state
                .gpus
                .iter()
                .all(|gpu| gpu.used == PreWalGpuUsed::default())
    }

    /// Retry-only seam for a request returned by `PreWalCapacityAcquireError`; no footprint can
    /// enter here by borrow or copy.
    #[allow(clippy::result_large_err)] // Failure preserves the exact request for allocation-free retry.
    pub(super) fn try_acquire_request(
        &self,
        request: PreWalCapacityRequest,
    ) -> Result<PreWalCapacityLease, PreWalCapacityAcquireError> {
        let mut state = lock_recover(&self.state);
        if let Err(cause) = state.validate_reservation(&request) {
            return Err(PreWalCapacityAcquireError { cause, request });
        }
        state.apply_reservation(&request);
        drop(state);
        Ok(PreWalCapacityLease {
            state: Arc::clone(&self.state),
            request: Some(request),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct PreWalGlobalUsed {
    host_retained_bytes: u64,
    host_allocation_slots: u64,
    host_scratch_peak_bytes: u64,
    host_generation_pin_slots: u64,
    wal_packed_record_bytes: u64,
    wal_serialized_record_bytes: u64,
    wal_record_slots: u64,
    wal_frame_slots: u64,
    row_id_slots: u64,
    sequence_effect_slots: u64,
    status_index_bytes: u64,
    status_index_slots: u64,
    terminal_response_bytes: u64,
    terminal_response_slots: u64,
    completion_bytes: u64,
    completion_slots: u64,
    publication_bytes: u64,
    publication_target_slots: u64,
    publication_join_entry_bytes: u64,
    publication_join_entry_slots: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreWalGpuState {
    limit: PreWalGpuCapacityLimit,
    authoritative_resident_bytes: u64,
    used: PreWalGpuUsed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct PreWalGpuUsed {
    incremental_device_peak_bytes: u64,
    plan_retained_transient_bytes: u64,
    result_retained_device_bytes: u64,
    scratch_peak_bytes: u64,
    allocation_slots: u64,
    generation_pin_slots: u64,
}

#[derive(Debug)]
struct PreWalCapacityState {
    global: PreWalGlobalUsed,
    gpus: Box<[PreWalGpuState]>,
    limits: PreWalCapacityLimits,
}

#[derive(Debug)]
struct PreWalCapacityDemand {
    global: PreWalGlobalUsed,
}

impl PreWalCapacityRequest {
    pub(super) fn from_validated_footprint(footprint: PreWalFootprint) -> Self {
        let host_retained_bytes = footprint
            .global_host_retained_bytes()
            .expect("current shape validated global host geometry");
        Self {
            demand: PreWalCapacityDemand {
                global: PreWalGlobalUsed {
                    host_retained_bytes,
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
                },
            },
            footprint,
        }
    }
}

impl PreWalCapacityState {
    /// Pass one is mutation-free.  All totals come from canonical footprint slices, so no map,
    /// boxed clone, or request-side demand copy can perturb the peak being admitted.
    fn validate_reservation(
        &self,
        request: &PreWalCapacityRequest,
    ) -> Result<(), PreWalCapacityError> {
        let mut global = self.global;
        reserve_global(&mut global, self.global_limits(), &request.demand.global)?;
        let mut pools = request.footprint.gpu_pool_cursor();
        while let Some(pool) = pools
            .try_next()
            .expect("current footprint validated GPU pool geometry")
        {
            let Some(gpu) = self.gpu_state(pool.gpu_id) else {
                return Err(PreWalCapacityError::UnknownGpuDemand(pool.gpu_id));
            };
            let mut used = gpu.used;
            reserve_gpu(
                &mut used,
                &gpu.limit,
                gpu.authoritative_resident_bytes,
                &pool,
            )?;
        }
        Ok(())
    }

    /// Validate the simultaneous incumbent/candidate peak by adding only positive candidate
    /// deltas. Negative deltas remain charged until commit.
    fn validate_replacement_headroom(
        &self,
        incumbent: &PreWalCapacityRequest,
        candidate: &PreWalCapacityRequest,
    ) -> Result<(), PreWalCapacityError> {
        let mut global = self.global;
        reserve_global_headroom(
            &mut global,
            self.global_limits(),
            &incumbent.demand.global,
            &candidate.demand.global,
        )?;

        let mut candidate_pools = candidate.footprint.gpu_pool_cursor();
        while let Some(pool) = candidate_pools
            .try_next()
            .expect("candidate footprint validated GPU pool geometry")
        {
            if self.gpu_index(pool.gpu_id).is_none() {
                return Err(PreWalCapacityError::UnknownGpuDemand(pool.gpu_id));
            }
        }
        for gpu in self.gpus.iter() {
            let mut used = gpu.used;
            let old = gpu_pool_for(&incumbent.footprint, gpu.limit.gpu_id);
            let next = gpu_pool_for(&candidate.footprint, gpu.limit.gpu_id);
            reserve_gpu_headroom(
                &mut used,
                &gpu.limit,
                gpu.authoritative_resident_bytes,
                old.as_ref(),
                next.as_ref(),
            )?;
        }
        Ok(())
    }

    /// Pass two runs only after the complete scalar validation succeeds under this mutex.  The
    /// same checked arithmetic is therefore infallible; any failure would be a broken invariant,
    /// not a partially-installed admission.
    fn apply_reservation(&mut self, request: &PreWalCapacityRequest) {
        reserve_global(&mut self.global, &self.limits, &request.demand.global)
            .expect("validated pre-WAL global reservation");
        let mut pools = request.footprint.gpu_pool_cursor();
        while let Some(pool) = pools
            .try_next()
            .expect("current footprint validated GPU pool geometry")
        {
            let index = self
                .gpu_index(pool.gpu_id)
                .expect("validated pre-WAL GPU reservation");
            let (limit, authoritative_resident_bytes) = {
                let gpu = &self.gpus[index];
                (gpu.limit, gpu.authoritative_resident_bytes)
            };
            reserve_gpu(
                &mut self.gpus[index].used,
                &limit,
                authoritative_resident_bytes,
                &pool,
            )
            .expect("validated pre-WAL GPU reservation");
        }
    }

    fn apply_replacement_headroom(
        &mut self,
        incumbent: &PreWalCapacityRequest,
        candidate: &PreWalCapacityRequest,
    ) {
        reserve_global_headroom(
            &mut self.global,
            &self.limits,
            &incumbent.demand.global,
            &candidate.demand.global,
        )
        .expect("validated pre-WAL replacement global headroom");
        for index in 0..self.gpus.len() {
            let gpu_id = self.gpus[index].limit.gpu_id;
            let old = gpu_pool_for(&incumbent.footprint, gpu_id);
            let next = gpu_pool_for(&candidate.footprint, gpu_id);
            let (limit, authoritative_resident_bytes) = {
                let gpu = &self.gpus[index];
                (gpu.limit, gpu.authoritative_resident_bytes)
            };
            reserve_gpu_headroom(
                &mut self.gpus[index].used,
                &limit,
                authoritative_resident_bytes,
                old.as_ref(),
                next.as_ref(),
            )
            .expect("validated pre-WAL replacement GPU headroom");
        }
    }

    fn release_replacement_headroom(
        &mut self,
        incumbent: &PreWalCapacityRequest,
        candidate: &PreWalCapacityRequest,
    ) {
        release_global_headroom(
            &mut self.global,
            &incumbent.demand.global,
            &candidate.demand.global,
        );
        for index in 0..self.gpus.len() {
            let gpu_id = self.gpus[index].limit.gpu_id;
            let old = gpu_pool_for(&incumbent.footprint, gpu_id);
            let next = gpu_pool_for(&candidate.footprint, gpu_id);
            release_gpu_headroom(&mut self.gpus[index].used, old.as_ref(), next.as_ref());
        }
    }

    fn commit_replacement(
        &mut self,
        incumbent: &PreWalCapacityRequest,
        candidate: &PreWalCapacityRequest,
    ) {
        self.release_replacement_headroom(incumbent, candidate);
        self.release(incumbent);
        self.apply_reservation(candidate);
    }

    fn global_limits(&self) -> &PreWalCapacityLimits {
        &self.limits
    }

    fn release(&mut self, request: &PreWalCapacityRequest) {
        release_global(&mut self.global, &request.demand.global);
        let mut pools = request.footprint.gpu_pool_cursor();
        while let Some(pool) = pools
            .try_next()
            .expect("admitted footprint preserves GPU pool geometry")
        {
            let gpu = self
                .gpu_state_mut(pool.gpu_id)
                .expect("lease footprint was admitted against a configured GPU");
            release_gpu(&mut gpu.used, &pool);
        }
    }

    fn gpu_state(&self, gpu_id: u16) -> Option<&PreWalGpuState> {
        self.gpu_index(gpu_id).map(|index| &self.gpus[index])
    }

    fn gpu_state_mut(&mut self, gpu_id: u16) -> Option<&mut PreWalGpuState> {
        let index = self.gpu_index(gpu_id)?;
        self.gpus.get_mut(index)
    }

    fn gpu_index(&self, gpu_id: u16) -> Option<usize> {
        self.gpus
            .binary_search_by_key(&gpu_id, |gpu| gpu.limit.gpu_id)
            .ok()
    }
}

fn gpu_pool_for(footprint: &PreWalFootprint, gpu_id: u16) -> Option<PreWalGpuPoolFootprint> {
    let mut pools = footprint.gpu_pool_cursor();
    while let Some(pool) = pools
        .try_next()
        .expect("validated footprint preserves GPU pool geometry")
    {
        match pool.gpu_id.cmp(&gpu_id) {
            std::cmp::Ordering::Less => {}
            std::cmp::Ordering::Equal => return Some(pool),
            std::cmp::Ordering::Greater => return None,
        }
    }
    None
}

fn validate_gpu_configuration(
    limits: &PreWalCapacityLimits,
    authoritative_residency: Box<[PreWalAuthoritativeGpuResidency]>,
) -> Result<Box<[PreWalGpuState]>, PreWalCapacityError> {
    if limits.gpus.is_empty() {
        return Err(PreWalCapacityError::EmptyGpuLimits);
    }
    validate_gpu_limit_order(&limits.gpus)?;
    validate_authoritative_order(&authoritative_residency)?;
    let mut gpus = Vec::with_capacity(limits.gpus.len());
    let mut authoritative = IntoIterator::into_iter(authoritative_residency).peekable();
    for limit in limits.gpus.iter() {
        let Some(current) = authoritative.peek() else {
            return Err(PreWalCapacityError::MissingAuthoritativeResidency(
                limit.gpu_id,
            ));
        };
        if current.gpu_id != limit.gpu_id {
            if current.gpu_id < limit.gpu_id {
                return Err(PreWalCapacityError::UnexpectedAuthoritativeResidency(
                    current.gpu_id,
                ));
            }
            return Err(PreWalCapacityError::MissingAuthoritativeResidency(
                limit.gpu_id,
            ));
        }
        let current = authoritative.next().expect("peeked residency exists");
        if current.bytes > limit.hard_device_bytes {
            return Err(PreWalCapacityError::AuthoritativeResidencyExceedsHard {
                gpu_id: limit.gpu_id,
                bytes: current.bytes,
                hard: limit.hard_device_bytes,
            });
        }
        gpus.push(PreWalGpuState {
            limit: *limit,
            authoritative_resident_bytes: current.bytes,
            used: PreWalGpuUsed::default(),
        });
    }
    if let Some(extra) = authoritative.next() {
        return Err(PreWalCapacityError::UnexpectedAuthoritativeResidency(
            extra.gpu_id,
        ));
    }
    Ok(gpus.into())
}

fn validate_gpu_limit_order(limits: &[PreWalGpuCapacityLimit]) -> Result<(), PreWalCapacityError> {
    for pair in limits.windows(2) {
        match pair[0].gpu_id.cmp(&pair[1].gpu_id) {
            std::cmp::Ordering::Less => {}
            std::cmp::Ordering::Equal => {
                return Err(PreWalCapacityError::DuplicateGpuLimit(pair[1].gpu_id));
            }
            std::cmp::Ordering::Greater => {
                return Err(PreWalCapacityError::GpuLimitOrderDrift(pair[1].gpu_id));
            }
        }
    }
    Ok(())
}

fn validate_authoritative_order(
    residency: &[PreWalAuthoritativeGpuResidency],
) -> Result<(), PreWalCapacityError> {
    for pair in residency.windows(2) {
        match pair[0].gpu_id.cmp(&pair[1].gpu_id) {
            std::cmp::Ordering::Less => {}
            std::cmp::Ordering::Equal => {
                return Err(PreWalCapacityError::DuplicateAuthoritativeResidency(
                    pair[1].gpu_id,
                ));
            }
            std::cmp::Ordering::Greater => {
                return Err(PreWalCapacityError::AuthoritativeResidencyOrderDrift(
                    pair[1].gpu_id,
                ));
            }
        }
    }
    Ok(())
}

fn reserve_global(
    used: &mut PreWalGlobalUsed,
    limit: &PreWalCapacityLimits,
    demand: &PreWalGlobalUsed,
) -> Result<(), PreWalCapacityError> {
    reserve(
        &mut used.host_retained_bytes,
        demand.host_retained_bytes,
        limit.host_retained_bytes,
        "host retained bytes",
    )?;
    reserve(
        &mut used.host_allocation_slots,
        demand.host_allocation_slots,
        limit.host_allocation_slots,
        "host allocation slots",
    )?;
    reserve(
        &mut used.host_scratch_peak_bytes,
        demand.host_scratch_peak_bytes,
        limit.host_scratch_peak_bytes,
        "host scratch peak bytes",
    )?;
    reserve(
        &mut used.host_generation_pin_slots,
        demand.host_generation_pin_slots,
        limit.host_generation_pin_slots,
        "host generation-pin slots",
    )?;
    reserve(
        &mut used.wal_packed_record_bytes,
        demand.wal_packed_record_bytes,
        limit.wal_packed_record_bytes,
        "WAL packed-record bytes",
    )?;
    reserve(
        &mut used.wal_serialized_record_bytes,
        demand.wal_serialized_record_bytes,
        limit.wal_serialized_record_bytes,
        "WAL serialized-record bytes",
    )?;
    reserve(
        &mut used.wal_record_slots,
        demand.wal_record_slots,
        limit.wal_record_slots,
        "WAL record slots",
    )?;
    reserve(
        &mut used.wal_frame_slots,
        demand.wal_frame_slots,
        limit.wal_frame_slots,
        "WAL frame slots",
    )?;
    reserve(
        &mut used.row_id_slots,
        demand.row_id_slots,
        limit.row_id_slots,
        "row-id slots",
    )?;
    reserve(
        &mut used.sequence_effect_slots,
        demand.sequence_effect_slots,
        limit.sequence_effect_slots,
        "sequence-effect slots",
    )?;
    reserve(
        &mut used.status_index_bytes,
        demand.status_index_bytes,
        limit.status_index_bytes,
        "status-index bytes",
    )?;
    reserve(
        &mut used.status_index_slots,
        demand.status_index_slots,
        limit.status_index_slots,
        "status-index slots",
    )?;
    reserve(
        &mut used.terminal_response_bytes,
        demand.terminal_response_bytes,
        limit.terminal_response_bytes,
        "terminal-response bytes",
    )?;
    reserve(
        &mut used.terminal_response_slots,
        demand.terminal_response_slots,
        limit.terminal_response_slots,
        "terminal-response slots",
    )?;
    reserve(
        &mut used.completion_bytes,
        demand.completion_bytes,
        limit.completion_bytes,
        "completion bytes",
    )?;
    reserve(
        &mut used.completion_slots,
        demand.completion_slots,
        limit.completion_slots,
        "completion slots",
    )?;
    reserve(
        &mut used.publication_bytes,
        demand.publication_bytes,
        limit.publication_bytes,
        "publication bytes",
    )?;
    reserve(
        &mut used.publication_target_slots,
        demand.publication_target_slots,
        limit.publication_target_slots,
        "publication target slots",
    )?;
    reserve(
        &mut used.publication_join_entry_bytes,
        demand.publication_join_entry_bytes,
        limit.publication_join_entry_bytes,
        "publication-join entry bytes",
    )?;
    reserve(
        &mut used.publication_join_entry_slots,
        demand.publication_join_entry_slots,
        limit.publication_join_entry_slots,
        "publication-join entry slots",
    )
}

fn reserve_global_headroom(
    used: &mut PreWalGlobalUsed,
    limit: &PreWalCapacityLimits,
    old: &PreWalGlobalUsed,
    next: &PreWalGlobalUsed,
) -> Result<(), PreWalCapacityError> {
    macro_rules! reserve_field {
        ($field:ident, $domain:literal) => {
            reserve_positive_delta(
                &mut used.$field,
                old.$field,
                next.$field,
                limit.$field,
                $domain,
            )?
        };
    }
    reserve_field!(host_retained_bytes, "host retained bytes");
    reserve_field!(host_allocation_slots, "host allocation slots");
    reserve_field!(host_scratch_peak_bytes, "host scratch peak bytes");
    reserve_field!(host_generation_pin_slots, "host generation-pin slots");
    reserve_field!(wal_packed_record_bytes, "WAL packed-record bytes");
    reserve_field!(wal_serialized_record_bytes, "WAL serialized-record bytes");
    reserve_field!(wal_record_slots, "WAL record slots");
    reserve_field!(wal_frame_slots, "WAL frame slots");
    reserve_field!(row_id_slots, "row-id slots");
    reserve_field!(sequence_effect_slots, "sequence-effect slots");
    reserve_field!(status_index_bytes, "status-index bytes");
    reserve_field!(status_index_slots, "status-index slots");
    reserve_field!(terminal_response_bytes, "terminal-response bytes");
    reserve_field!(terminal_response_slots, "terminal-response slots");
    reserve_field!(completion_bytes, "completion bytes");
    reserve_field!(completion_slots, "completion slots");
    reserve_field!(publication_bytes, "publication bytes");
    reserve_field!(publication_target_slots, "publication target slots");
    reserve_field!(publication_join_entry_bytes, "publication-join entry bytes");
    reserve_field!(publication_join_entry_slots, "publication-join entry slots");
    Ok(())
}

fn release_global_headroom(
    used: &mut PreWalGlobalUsed,
    old: &PreWalGlobalUsed,
    next: &PreWalGlobalUsed,
) {
    macro_rules! release_field {
        ($field:ident) => {
            release_positive_delta(&mut used.$field, old.$field, next.$field)
        };
    }
    release_field!(host_retained_bytes);
    release_field!(host_allocation_slots);
    release_field!(host_scratch_peak_bytes);
    release_field!(host_generation_pin_slots);
    release_field!(wal_packed_record_bytes);
    release_field!(wal_serialized_record_bytes);
    release_field!(wal_record_slots);
    release_field!(wal_frame_slots);
    release_field!(row_id_slots);
    release_field!(sequence_effect_slots);
    release_field!(status_index_bytes);
    release_field!(status_index_slots);
    release_field!(terminal_response_bytes);
    release_field!(terminal_response_slots);
    release_field!(completion_bytes);
    release_field!(completion_slots);
    release_field!(publication_bytes);
    release_field!(publication_target_slots);
    release_field!(publication_join_entry_bytes);
    release_field!(publication_join_entry_slots);
}

fn reserve_gpu(
    used: &mut PreWalGpuUsed,
    limit: &PreWalGpuCapacityLimit,
    authoritative_resident_bytes: u64,
    demand: &PreWalGpuPoolFootprint,
) -> Result<(), PreWalCapacityError> {
    if let Some(envelope_limit) = limit.generation_envelope_bytes {
        let bytes = demand
            .old_generation_pinned_bytes
            .checked_add(demand.new_persistent_bytes)
            .ok_or(PreWalCapacityError::Overflow(
                "GPU generation evidence bytes",
            ))?;
        if bytes > envelope_limit {
            return Err(PreWalCapacityError::GenerationEnvelopeExceeded {
                gpu_id: demand.gpu_id,
                bytes,
                limit: envelope_limit,
            });
        }
    }
    reserve_hard_gpu(used, limit, authoritative_resident_bytes, demand)?;
    reserve(
        &mut used.plan_retained_transient_bytes,
        demand.plan_retained_transient_bytes,
        limit.plan_retained_transient_bytes,
        "GPU plan-retained transient bytes",
    )?;
    reserve(
        &mut used.result_retained_device_bytes,
        demand.result_retained_device_bytes,
        limit.result_retained_device_bytes,
        "GPU result-retained device bytes",
    )?;
    reserve(
        &mut used.scratch_peak_bytes,
        demand.scratch_peak_bytes,
        limit.scratch_peak_bytes,
        "GPU scratch peak bytes",
    )?;
    reserve(
        &mut used.allocation_slots,
        demand.allocation_slots,
        limit.allocation_slots,
        "GPU allocation slots",
    )?;
    reserve(
        &mut used.generation_pin_slots,
        demand.generation_pin_slots,
        limit.generation_pin_slots,
        "GPU generation-pin slots",
    )
}

fn reserve_gpu_headroom(
    used: &mut PreWalGpuUsed,
    limit: &PreWalGpuCapacityLimit,
    authoritative_resident_bytes: u64,
    old: Option<&PreWalGpuPoolFootprint>,
    next: Option<&PreWalGpuPoolFootprint>,
) -> Result<(), PreWalCapacityError> {
    if let (Some(envelope_limit), Some(next)) = (limit.generation_envelope_bytes, next) {
        let bytes = next
            .old_generation_pinned_bytes
            .checked_add(next.new_persistent_bytes)
            .ok_or(PreWalCapacityError::Overflow(
                "GPU generation evidence bytes",
            ))?;
        if bytes > envelope_limit {
            return Err(PreWalCapacityError::GenerationEnvelopeExceeded {
                gpu_id: next.gpu_id,
                bytes,
                limit: envelope_limit,
            });
        }
    }
    let old = old.map(gpu_used_from_pool).transpose()?.unwrap_or_default();
    let next = next
        .map(gpu_used_from_pool)
        .transpose()?
        .unwrap_or_default();
    let incremental_limit = limit
        .hard_device_bytes
        .checked_sub(authoritative_resident_bytes)
        .ok_or(PreWalCapacityError::Overflow("GPU hard device bytes"))?;
    reserve_positive_delta(
        &mut used.incremental_device_peak_bytes,
        old.incremental_device_peak_bytes,
        next.incremental_device_peak_bytes,
        incremental_limit,
        "GPU hard device bytes",
    )?;
    reserve_positive_delta(
        &mut used.plan_retained_transient_bytes,
        old.plan_retained_transient_bytes,
        next.plan_retained_transient_bytes,
        limit.plan_retained_transient_bytes,
        "GPU plan-retained transient bytes",
    )?;
    reserve_positive_delta(
        &mut used.result_retained_device_bytes,
        old.result_retained_device_bytes,
        next.result_retained_device_bytes,
        limit.result_retained_device_bytes,
        "GPU result-retained device bytes",
    )?;
    reserve_positive_delta(
        &mut used.scratch_peak_bytes,
        old.scratch_peak_bytes,
        next.scratch_peak_bytes,
        limit.scratch_peak_bytes,
        "GPU scratch peak bytes",
    )?;
    reserve_positive_delta(
        &mut used.allocation_slots,
        old.allocation_slots,
        next.allocation_slots,
        limit.allocation_slots,
        "GPU allocation slots",
    )?;
    reserve_positive_delta(
        &mut used.generation_pin_slots,
        old.generation_pin_slots,
        next.generation_pin_slots,
        limit.generation_pin_slots,
        "GPU generation-pin slots",
    )
}

fn release_gpu_headroom(
    used: &mut PreWalGpuUsed,
    old: Option<&PreWalGpuPoolFootprint>,
    next: Option<&PreWalGpuPoolFootprint>,
) {
    let old = old
        .map(gpu_used_from_pool)
        .transpose()
        .expect("admitted incumbent GPU geometry remains checked")
        .unwrap_or_default();
    let next = next
        .map(gpu_used_from_pool)
        .transpose()
        .expect("admitted candidate GPU geometry remains checked")
        .unwrap_or_default();
    release_positive_delta(
        &mut used.incremental_device_peak_bytes,
        old.incremental_device_peak_bytes,
        next.incremental_device_peak_bytes,
    );
    release_positive_delta(
        &mut used.plan_retained_transient_bytes,
        old.plan_retained_transient_bytes,
        next.plan_retained_transient_bytes,
    );
    release_positive_delta(
        &mut used.result_retained_device_bytes,
        old.result_retained_device_bytes,
        next.result_retained_device_bytes,
    );
    release_positive_delta(
        &mut used.scratch_peak_bytes,
        old.scratch_peak_bytes,
        next.scratch_peak_bytes,
    );
    release_positive_delta(
        &mut used.allocation_slots,
        old.allocation_slots,
        next.allocation_slots,
    );
    release_positive_delta(
        &mut used.generation_pin_slots,
        old.generation_pin_slots,
        next.generation_pin_slots,
    );
}

fn gpu_used_from_pool(pool: &PreWalGpuPoolFootprint) -> Result<PreWalGpuUsed, PreWalCapacityError> {
    Ok(PreWalGpuUsed {
        incremental_device_peak_bytes: pool
            .incremental_device_peak_bytes()
            .map_err(|_| PreWalCapacityError::Overflow("GPU hard device bytes"))?,
        plan_retained_transient_bytes: pool.plan_retained_transient_bytes,
        result_retained_device_bytes: pool.result_retained_device_bytes,
        scratch_peak_bytes: pool.scratch_peak_bytes,
        allocation_slots: pool.allocation_slots,
        generation_pin_slots: pool.generation_pin_slots,
    })
}

fn reserve_hard_gpu(
    used: &mut PreWalGpuUsed,
    limit: &PreWalGpuCapacityLimit,
    authoritative_resident_bytes: u64,
    demand: &PreWalGpuPoolFootprint,
) -> Result<(), PreWalCapacityError> {
    let incremental_device_peak_bytes = demand
        .incremental_device_peak_bytes()
        .map_err(|_| PreWalCapacityError::Overflow("GPU hard device bytes"))?;
    let outstanding = used
        .incremental_device_peak_bytes
        .checked_add(incremental_device_peak_bytes)
        .ok_or(PreWalCapacityError::Overflow("GPU hard device bytes"))?;
    let total = authoritative_resident_bytes
        .checked_add(outstanding)
        .ok_or(PreWalCapacityError::Overflow("GPU hard device bytes"))?;
    if total > limit.hard_device_bytes {
        return Err(PreWalCapacityError::ResourceExhausted(
            "GPU hard device bytes",
        ));
    }
    used.incremental_device_peak_bytes = outstanding;
    Ok(())
}

fn reserve(
    used: &mut u64,
    demand: u64,
    limit: u64,
    domain: &'static str,
) -> Result<(), PreWalCapacityError> {
    let candidate = used
        .checked_add(demand)
        .ok_or(PreWalCapacityError::Overflow(domain))?;
    if candidate > limit {
        return Err(PreWalCapacityError::ResourceExhausted(domain));
    }
    *used = candidate;
    Ok(())
}

fn reserve_positive_delta(
    used: &mut u64,
    old: u64,
    next: u64,
    limit: u64,
    domain: &'static str,
) -> Result<(), PreWalCapacityError> {
    reserve(used, next.saturating_sub(old), limit, domain)
}

fn release_global(used: &mut PreWalGlobalUsed, demand: &PreWalGlobalUsed) {
    release(&mut used.host_retained_bytes, demand.host_retained_bytes);
    release(
        &mut used.host_allocation_slots,
        demand.host_allocation_slots,
    );
    release(
        &mut used.host_scratch_peak_bytes,
        demand.host_scratch_peak_bytes,
    );
    release(
        &mut used.host_generation_pin_slots,
        demand.host_generation_pin_slots,
    );
    release(
        &mut used.wal_packed_record_bytes,
        demand.wal_packed_record_bytes,
    );
    release(
        &mut used.wal_serialized_record_bytes,
        demand.wal_serialized_record_bytes,
    );
    release(&mut used.wal_record_slots, demand.wal_record_slots);
    release(&mut used.wal_frame_slots, demand.wal_frame_slots);
    release(&mut used.row_id_slots, demand.row_id_slots);
    release(
        &mut used.sequence_effect_slots,
        demand.sequence_effect_slots,
    );
    release(&mut used.status_index_bytes, demand.status_index_bytes);
    release(&mut used.status_index_slots, demand.status_index_slots);
    release(
        &mut used.terminal_response_bytes,
        demand.terminal_response_bytes,
    );
    release(
        &mut used.terminal_response_slots,
        demand.terminal_response_slots,
    );
    release(&mut used.completion_bytes, demand.completion_bytes);
    release(&mut used.completion_slots, demand.completion_slots);
    release(&mut used.publication_bytes, demand.publication_bytes);
    release(
        &mut used.publication_target_slots,
        demand.publication_target_slots,
    );
    release(
        &mut used.publication_join_entry_bytes,
        demand.publication_join_entry_bytes,
    );
    release(
        &mut used.publication_join_entry_slots,
        demand.publication_join_entry_slots,
    );
}

fn release_gpu(used: &mut PreWalGpuUsed, demand: &PreWalGpuPoolFootprint) {
    let incremental_device_peak_bytes = demand
        .incremental_device_peak_bytes()
        .expect("admitted GPU pool has checked peak bytes");
    release(
        &mut used.incremental_device_peak_bytes,
        incremental_device_peak_bytes,
    );
    release(
        &mut used.plan_retained_transient_bytes,
        demand.plan_retained_transient_bytes,
    );
    release(
        &mut used.result_retained_device_bytes,
        demand.result_retained_device_bytes,
    );
    release(&mut used.scratch_peak_bytes, demand.scratch_peak_bytes);
    release(&mut used.allocation_slots, demand.allocation_slots);
    release(&mut used.generation_pin_slots, demand.generation_pin_slots);
}

fn release(used: &mut u64, demand: u64) {
    *used = used
        .checked_sub(demand)
        .expect("Drop-only capacity lease releases exactly its admitted demand");
}

fn release_positive_delta(used: &mut u64, old: u64, next: u64) {
    release(used, next.saturating_sub(old));
}

fn lock_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
#[path = "pre_wal_capacity_tests.rs"]
mod tests;
