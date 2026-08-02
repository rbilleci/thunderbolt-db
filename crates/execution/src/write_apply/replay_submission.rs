//! Owned asynchronous execution for one already-typed i32 INSERT replay segment.
//!
//! This is deliberately below engine durability/apply/publication authority.  It owns a real fused
//! device append launch, the private CUDA stream that carries it, every device and pinned-host
//! backing the stream can still touch, and the only bounded completion readback.  The caller gets
//! a quiesced kernel result or an owner representing *unknown* quiescence; it never gets an
//! apparently reusable buffer after an unproved stream drain.

use super::prepared_fused::{
    validate_fused_apply_preparation, visit_preparation_owner_memories, FusedApplyValidatedGeometry,
};
use super::{FusedApplyPreparation, FUSED_APPLY_PTX};
use crate::{
    CudaResidentDeviceMemory, CudaResidentIndexStatus, CudaResidentReadSource,
    CudaRuntimeProbeError, PinnedHostBufferOwned, PooledDeviceBufferOwned, PooledStreamOwned,
};
use std::{
    ffi::c_void,
    sync::{Arc, Mutex},
};

#[allow(clippy::type_complexity)]
type CuLaunchKernel = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;

/// Exact resources held by this asynchronous replay route after preparation and before enqueue.
///
/// This deliberately does not reuse the synchronous token's footprint: its boxed staging image
/// and stack status word do not exist here. Every capacity below is read from the concrete owner
/// that this route actually retains, apart from the exact temporary NUL-terminated PTX image used
/// solely while caching the already-resolved kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaI32InsertReplayResourceGeometry {
    /// Deduplicated source/destination/header/index CUDA allocations pinned through drain.
    pub pinned_cuda_allocation_count: u64,
    /// Exact boxed `Arc<CudaResidentDeviceAllocation>` backing that holds those pins.
    pub owner_array_backing_bytes: u64,
    pub owner_array_allocation_slots: u64,
    /// Page-locked HtoD staging capacity from the pinned-host pool's actual bucket.
    pub pinned_htod_staging_bytes: u64,
    pub pinned_htod_staging_allocation_slots: u64,
    /// Page-locked four-byte DtoH verdict capacity from the pinned-host pool's actual bucket.
    pub pinned_status_bytes: u64,
    pub pinned_status_allocation_slots: u64,
    /// Device staging capacity from the pooled-device buffer's actual bucket.
    pub pooled_device_staging_bytes: u64,
    pub pooled_device_staging_allocation_slots: u64,
    /// One owned private stream, including the stream pool's fixed device scratch allocation.
    pub private_stream_count: u64,
    pub private_stream_scratch_bytes: u64,
    pub private_stream_scratch_allocation_slots: u64,
    /// `GpuPrimaryContext::acquire_pooled_stream` creates either both timing-event handles or
    /// neither, so this is exactly `0` or `2`. CUDA owns their opaque storage; this route holds no
    /// separate Rust host backing for those handles.
    pub private_stream_timing_event_count: u64,
    pub private_stream_timing_event_host_backing_bytes: u64,
    pub private_stream_timing_event_host_backing_slots: u64,
    /// The sole transient host allocation during preparation: FUSED_APPLY_PTX plus its NUL.
    pub transient_ptx_staging_bytes: u64,
    pub transient_ptx_staging_allocation_slots: u64,
}

/// A pre-enqueue failure for the narrow asynchronous typed INSERT replay primitive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaInsertReplayPrepareError {
    Runtime(CudaRuntimeProbeError),
    /// The driver lacks one of the stream-ordered HtoD/DtoH operations required to return while
    /// a real write kernel remains in flight.  This route intentionally does not downgrade to a
    /// blocking/default-stream apply.
    AsyncTransportUnavailable,
    /// Page-locked host backing is mandatory for this owned async contract; accepting pageable
    /// input would make the driver free to synchronously stage it inside `enqueue`.
    PinnedHostStagingUnavailable,
}

impl From<CudaRuntimeProbeError> for CudaInsertReplayPrepareError {
    fn from(value: CudaRuntimeProbeError) -> Self {
        Self::Runtime(value)
    }
}

/// The only successful output of a drained replay launch.  It intentionally contains no header
/// publication or engine mutation authority: visibility remains owned by the later retained
/// generation boundary.  These booleans are the fused index kernel's bounded device verdict,
/// not transaction/status/publication authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaI32InsertReplayResult {
    pub index_declined: bool,
    pub created_posting: bool,
}

/// Preparation owns every host/device/stream resource before the first enqueue.  It is move-only
/// so a sealed typed segment has one submit attempt; dropping it before [`Self::enqueue`] simply
/// returns unsubmitted leases to their pools.
#[must_use = "an owned typed INSERT replay preparation must be enqueued or intentionally dropped"]
pub struct CudaI32InsertReplayPreparation {
    primary: Arc<crate::GpuPrimaryContext>,
    resources: ReplayResources,
    function: *mut c_void,
    launch: CuLaunchKernel,
    blocks: u32,
    geometry: CudaI32InsertReplayResourceGeometry,
}

/// One real asynchronous typed i32 INSERT replay launch.  Every pointer reachable by its HtoD,
/// kernel, or queued DtoH is owned until [`Self::complete`] proves the private stream quiescent.
#[must_use = "an in-flight CUDA replay submission must be completed or safely dropped"]
pub struct CudaInsertReplaySubmission {
    primary: Arc<crate::GpuPrimaryContext>,
    resources: Option<ReplayResources>,
    function: *mut c_void,
    launch: CuLaunchKernel,
    blocks: u32,
    phase: ReplayPhase,
    /// The first execution/terminal error after enqueue.  Later fence failures can leave
    /// quiescence unknown, but may not overwrite this fail-closed outcome when a retry drains.
    first_terminal_error: Option<CudaRuntimeProbeError>,
    /// Any genuine CUDA API result after replay submission quarantines the private stream and all
    /// context-bound pooled backing even when a later fence proves the stream idle.
    driver_error_observed: bool,
}

/// Completion separates a known-idle outcome from an unproved fence.  `Quiesced(Err(_))` means
/// the launch/readback failed *and* the stream was still successfully drained; callers may safely
/// abandon it.  `UnknownQuiescence` retains the exact resources and offers a retry drain instead.
#[must_use = "unknown CUDA quiescence must be retained or retried; dropping it parks resources"]
#[allow(
    clippy::large_enum_variant,
    reason = "boxing unknown quiescence would allocate after a CUDA command has been enqueued"
)]
pub enum CudaInsertReplayCompletion {
    Quiesced(Result<CudaI32InsertReplayResult, CudaRuntimeProbeError>),
    UnknownQuiescence(CudaInsertReplayUnknownQuiescence),
}

/// The fail-closed owner returned when a stream fence cannot prove that the queued replay work is
/// idle.  It deliberately has no success-like result accessor.  Retrying completion may establish
/// quiescence later; dropping it attempts one final drain and otherwise parks the owned
/// allocations rather than returning in-flight memory or a stream to a shared pool.
#[must_use = "unknown CUDA quiescence retains in-flight resources until retried or parked"]
pub struct CudaInsertReplayUnknownQuiescence {
    error: CudaRuntimeProbeError,
    submission: Option<CudaInsertReplaySubmission>,
}

struct ReplayResources {
    // Keep the wrapper as well as the exact deduplicated allocation guards.  The wrapper carries
    // source metadata/primary identity; the boxed guards pin every source/destination/index
    // allocation the fused device ABI may dereference.
    _source: Arc<CudaResidentDeviceMemory>,
    _owners: Box<[Arc<crate::resident_memory::CudaResidentDeviceAllocation>]>,
    staging_host: PinnedHostBufferOwned,
    status_host: PinnedHostBufferOwned,
    staging_device: PooledDeviceBufferOwned,
    stream: PooledStreamOwned,
    staging_len: usize,
}

// The parking lot is deliberately fixed-size: Drop may run after the first DMA enqueue and must
// not allocate while trying to keep unproved resources out of shared pools.  Ordinary callers
// should retain `CudaInsertReplayUnknownQuiescence` and retry; this is only the fail-closed final
// Drop path.  Exhaustion remains safe by intentionally retaining the resources without reuse.
const PARKED_REPLAY_CAPACITY: usize = 8;

struct ReplayParkingLot {
    slots: [Option<ReplayResources>; PARKED_REPLAY_CAPACITY],
}

static PARKED_REPLAY_RESOURCES: Mutex<ReplayParkingLot> = Mutex::new(ReplayParkingLot {
    slots: [const { None }; PARKED_REPLAY_CAPACITY],
});

// A successfully drained stream is not automatically healthy after a CUDA API error.  Keep its
// exact pool leases out of reuse permanently; callers receive the first error and must use a
// fresh canonical recovery owner rather than redeeming this replay attempt.
const QUARANTINED_REPLAY_CAPACITY: usize = 8;

struct ReplayQuarantineLot {
    slots: [Option<ReplayResources>; QUARANTINED_REPLAY_CAPACITY],
}

static QUARANTINED_REPLAY_RESOURCES: Mutex<ReplayQuarantineLot> = Mutex::new(ReplayQuarantineLot {
    slots: [const { None }; QUARANTINED_REPLAY_CAPACITY],
});

#[cfg(test)]
thread_local! {
    static REPLAY_SUCCESSFUL_ENQUEUES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[derive(Debug, Clone)]
enum ReplayPhase {
    /// HtoD and kernel launch completed successfully; no DtoH status readback is queued yet.
    InFlight,
    /// The bounded status DtoH follows the kernel on the private stream.
    StatusReadbackQueued,
    /// Submission/completion itself failed.  `drain_required` distinguishes a pre-enqueue error
    /// from an error after a command might already have touched the private stream.
    TerminalFailure {
        drain_required: bool,
    },
    Quiesced,
}

impl ReplayPhase {
    fn drain_required(&self) -> bool {
        matches!(
            self,
            Self::InFlight
                | Self::StatusReadbackQueued
                | Self::TerminalFailure {
                    drain_required: true
                }
        )
    }
}

// SAFETY: this move-only owner pins all device and host backings, owns one private stream, and
// re-binds its primary context before enqueue/drain on a completion thread.  No raw pointer is
// exposed from the public API and the submission is deliberately not Sync.
unsafe impl Send for CudaI32InsertReplayPreparation {}
unsafe impl Send for CudaInsertReplaySubmission {}
unsafe impl Send for CudaInsertReplayUnknownQuiescence {}
// SAFETY: a parked entry has the same unique ownership and context-rebind discipline as the
// submission that contained it; the fixed parking lot never exposes its raw pointers.
unsafe impl Send for ReplayResources {}

impl CudaI32InsertReplayPreparation {
    /// Reserve all CUDA and host resources and materialize the immutable typed launch image.
    /// This performs no device enqueue; [`Self::enqueue`] is the sole submit edge.
    pub fn prepare(
        source: Arc<CudaResidentDeviceMemory>,
        preparation: &FusedApplyPreparation<'_>,
        stamps: &[u64],
    ) -> Result<Self, CudaInsertReplayPrepareError> {
        let validated = validate_replay_inputs(source.as_ref(), preparation, stamps)?;
        let primary = source.primary_arc();
        if primary.cu_memcpy_htod_async.is_none() || primary.cu_memcpy_dtoh_async.is_none() {
            return Err(CudaInsertReplayPrepareError::AsyncTransportUnavailable);
        }
        primary.set_current()?;

        // The exact owners are materialized once before any enqueue.  Boxed slots match the
        // allocation-free footprint rather than retaining caller slices or a capacity-bearing Vec.
        let owners = materialize_owners(source.as_ref(), preparation, &validated)?;
        let mut staging_host = primary
            .lease_pinned_host_buffer_owned(validated.total_staging_bytes)
            .ok_or(CudaInsertReplayPrepareError::PinnedHostStagingUnavailable)?;
        let mut status_host = primary
            .lease_pinned_host_buffer_owned(std::mem::size_of::<u32>())
            .ok_or(CudaInsertReplayPrepareError::PinnedHostStagingUnavailable)?;
        let staging_device = primary.lease_device_buffer_owned(validated.total_staging_bytes)?;
        let stream = PooledStreamOwned {
            primary: Arc::clone(&primary),
            pooled: Some(primary.acquire_pooled_stream()?),
        };

        let ptx_len = FUSED_APPLY_PTX
            .len()
            .checked_add(1)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let geometry = replay_resource_geometry(
            &owners,
            &staging_host,
            &status_host,
            &staging_device,
            &stream,
            ptx_len,
        )?;

        write_staging_image(
            staging_host.as_mut_bytes(validated.total_staging_bytes)?,
            preparation,
            stamps,
            &validated,
        )?;
        status_host
            .as_mut_bytes(std::mem::size_of::<u32>())?
            .fill(0);

        // Module/cache work is preparation work.  The NUL-terminated PTX backing dies before
        // `enqueue`, and the cached function handle is safe for concurrent private streams.
        let mut ptx = Box::<[u8]>::new_uninit_slice(ptx_len);
        for (slot, byte) in ptx.iter_mut().zip(FUSED_APPLY_PTX.iter().copied()) {
            slot.write(byte);
        }
        ptx[FUSED_APPLY_PTX.len()].write(0);
        // SAFETY: every byte above, including the required trailing NUL, was initialized.
        let ptx = unsafe { ptx.assume_init() };
        let function = primary.cached_function(c"gpu_db_resident_i32_fused_apply", &ptx)?;
        let launch = unsafe {
            *primary
                .lib()
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };

        Ok(Self {
            primary,
            resources: ReplayResources {
                _source: source,
                _owners: owners,
                staging_host,
                status_host,
                staging_device,
                stream,
                staging_len: validated.total_staging_bytes,
            },
            function,
            launch,
            blocks: validated.k_u32.div_ceil(128),
            geometry,
        })
    }

    /// The exact pre-reserved geometry, available without exposing staging bytes or mutable CUDA
    /// pointers to the caller.
    pub fn resource_geometry_report(&self) -> CudaI32InsertReplayResourceGeometry {
        self.geometry
    }

    /// Enqueue the one pinned HtoD, real fused write kernel, and retain ownership while the work
    /// remains in flight.  Driver failures are captured in the returned owner so completion can
    /// still prove—or refuse to claim—stream quiescence.
    pub fn enqueue(self) -> CudaInsertReplaySubmission {
        let mut submission = CudaInsertReplaySubmission {
            primary: self.primary,
            resources: Some(self.resources),
            function: self.function,
            launch: self.launch,
            blocks: self.blocks,
            phase: ReplayPhase::InFlight,
            first_terminal_error: None,
            driver_error_observed: false,
        };
        submission.enqueue_inner();
        submission
    }
}

impl CudaInsertReplaySubmission {
    fn resources(&self) -> &ReplayResources {
        self.resources
            .as_ref()
            .expect("replay submission resources remain owned until quiescence")
    }

    fn stream(&self) -> *mut c_void {
        self.resources()
            .stream
            .pooled
            .as_ref()
            .expect("private replay stream remains owned until quiescence")
            .stream
    }

    fn enqueue_inner(&mut self) {
        if let Err(error) = self.primary.set_current() {
            self.retain_driver_error(&error);
            self.phase = ReplayPhase::TerminalFailure {
                drain_required: false,
            };
            return;
        }
        let stream = self.stream();
        let resources = self.resources();
        // Once the driver has seen the first async command we fail closed and drain even when it
        // reports an error: a sticky earlier error must never make pooled memory look reusable.
        if let Err(error) = self.primary.enqueue_owned_stream_htod(
            resources.staging_device.ptr,
            resources.staging_host.ptr.cast_const(),
            resources.staging_len,
            stream,
        ) {
            self.retain_driver_error(&error);
            self.phase = ReplayPhase::TerminalFailure {
                drain_required: true,
            };
            return;
        }
        let mut staging_arg = resources.staging_device.ptr;
        let mut args = [(&mut staging_arg as *mut u64).cast::<c_void>()];
        let result = self.primary.check_owned_stream_launch_result(unsafe {
            (self.launch)(
                self.function,
                self.blocks,
                1,
                1,
                128,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        });
        if let Err(error) = result {
            self.retain_driver_error(&error);
            self.phase = ReplayPhase::TerminalFailure {
                drain_required: true,
            };
        } else {
            self.primary.after_owned_stream_enqueue();
            #[cfg(test)]
            REPLAY_SUCCESSFUL_ENQUEUES.with(|value| value.set(value.get() + 1));
        }
    }

    fn queue_status_readback(&mut self) -> Result<(), CudaRuntimeProbeError> {
        if !matches!(self.phase, ReplayPhase::InFlight) {
            return Ok(());
        }
        self.primary
            .bind_owned_stream_for_replay()
            .map_err(replay_fence_failure_error)?;
        let stream = self.stream();
        let resources = self.resources();
        self.primary.enqueue_owned_stream_dtoh(
            resources.status_host.ptr,
            resources.staging_device.ptr + 28,
            std::mem::size_of::<u32>(),
            stream,
        )?;
        self.phase = ReplayPhase::StatusReadbackQueued;
        Ok(())
    }

    fn drain_stream(&self) -> Result<(), crate::cuda_context::ReplayOwnedStreamFenceFailure> {
        self.primary
            .synchronize_owned_stream_for_replay(self.stream())
    }

    fn completion_error(&self) -> Option<CudaRuntimeProbeError> {
        self.first_terminal_error.clone()
    }

    fn retain_first_terminal_error(&mut self, error: &CudaRuntimeProbeError) {
        if self.first_terminal_error.is_none() {
            self.first_terminal_error = Some(error.clone());
        }
    }

    fn retain_driver_error(&mut self, error: &CudaRuntimeProbeError) {
        self.retain_first_terminal_error(error);
        self.driver_error_observed = true;
    }

    fn unknown(self, error: CudaRuntimeProbeError) -> CudaInsertReplayCompletion {
        CudaInsertReplayCompletion::UnknownQuiescence(CudaInsertReplayUnknownQuiescence {
            error,
            submission: Some(self),
        })
    }

    /// Complete the kernel's bounded status readback and synchronize the owned private stream.
    /// Success proves all DMA/kernel accesses have stopped before any buffer or stream returns to
    /// its pool.  An unprovable synchronize result returns [`CudaInsertReplayUnknownQuiescence`]
    /// instead of a potentially misleading ordinary CUDA error.
    pub fn complete(mut self) -> CudaInsertReplayCompletion {
        if self.phase.drain_required() && matches!(self.phase, ReplayPhase::InFlight) {
            if let Err(error) = self.queue_status_readback() {
                self.retain_driver_error(&error);
                self.phase = ReplayPhase::TerminalFailure {
                    drain_required: true,
                };
            }
        }
        if self.phase.drain_required() {
            if let Err(failure) = self.drain_stream() {
                let error = replay_fence_failure_error(failure.clone());
                if failure.cuda_error().is_some() {
                    self.retain_driver_error(&error);
                }
                return self.unknown(error);
            }
        }

        let result = if let Some(error) = self.completion_error() {
            Err(error)
        } else if matches!(self.phase, ReplayPhase::StatusReadbackQueued) {
            // The covering private-stream sync above establishes host visibility for this exact
            // four-byte pinned DtoH result.  No public raw pointer escapes this owner.
            let status = unsafe {
                u32::from_le_bytes(
                    std::slice::from_raw_parts(
                        self.resources().status_host.ptr.cast::<u8>(),
                        std::mem::size_of::<u32>(),
                    )
                    .try_into()
                    .expect("four-byte replay status backing"),
                )
            };
            let index_verdict = CudaResidentIndexStatus::from_bits(status);
            Ok(CudaI32InsertReplayResult {
                index_declined: index_verdict.declined,
                created_posting: index_verdict.created_posting,
            })
        } else {
            Err(CudaRuntimeProbeError::KernelLaunchFailed(-1))
        };
        self.phase = ReplayPhase::Quiesced;
        self.quarantine_if_driver_error();
        CudaInsertReplayCompletion::Quiesced(result)
    }

    fn quarantine_if_driver_error(&mut self) {
        if self.driver_error_observed {
            if let Some(resources) = self.resources.take() {
                quarantine_replay_resources(resources);
            }
        }
    }
}

impl Drop for CudaInsertReplaySubmission {
    fn drop(&mut self) {
        if !self.phase.drain_required() {
            self.quarantine_if_driver_error();
            return;
        }
        // Cancellation/early-error/panic must not return the private stream, pinned DMA backing,
        // or device staging buffer to a concurrent pool user while the kernel might still touch
        // it.  A failed fence is conservatively parked below.
        match self.drain_stream() {
            Ok(()) => {
                self.phase = ReplayPhase::Quiesced;
                self.quarantine_if_driver_error();
            }
            Err(failure) => {
                if let Some(error) = failure.cuda_error() {
                    self.retain_driver_error(error);
                }
                if let Some(resources) = self.resources.take() {
                    if self.driver_error_observed {
                        quarantine_replay_resources(resources);
                    } else {
                        park_replay_resources(resources);
                    }
                }
            }
        }
    }
}

fn park_replay_resources(resources: ReplayResources) {
    let Ok(mut parking) = PARKED_REPLAY_RESOURCES.lock() else {
        // The only sound fallback when CUDA cannot prove quiescence is to retain every object
        // that could still be referenced.  This deliberately sacrifices bounded memory over
        // returning an in-flight allocation/stream to a shared pool.
        std::mem::forget(resources);
        return;
    };
    if let Some(slot) = parking.slots.iter_mut().find(|slot| slot.is_none()) {
        *slot = Some(resources);
    } else {
        // Fixed capacity keeps Drop allocation-free.  Full parking is exceptional and remains
        // fail-closed rather than releasing a resource whose private stream was not proven idle.
        std::mem::forget(resources);
    }
}

fn quarantine_replay_resources(resources: ReplayResources) {
    let Ok(mut quarantine) = QUARANTINED_REPLAY_RESOURCES.lock() else {
        std::mem::forget(resources);
        return;
    };
    if let Some(slot) = quarantine.slots.iter_mut().find(|slot| slot.is_none()) {
        *slot = Some(resources);
    } else {
        // Quarantine must never return a potentially poisoned stream/buffer to a shared pool.
        std::mem::forget(resources);
    }
}

#[cfg(test)]
pub(crate) fn parked_replay_resource_count_for_test() -> usize {
    PARKED_REPLAY_RESOURCES
        .lock()
        .map(|parking| parking.slots.iter().filter(|slot| slot.is_some()).count())
        .unwrap_or(0)
}

#[cfg(test)]
pub(crate) fn quarantined_replay_resource_count_for_test() -> usize {
    QUARANTINED_REPLAY_RESOURCES
        .lock()
        .map(|quarantine| {
            quarantine
                .slots
                .iter()
                .filter(|slot| slot.is_some())
                .count()
        })
        .unwrap_or(0)
}

#[cfg(test)]
pub(crate) fn replay_successful_enqueue_count_for_test() -> u64 {
    REPLAY_SUCCESSFUL_ENQUEUES.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn drain_parked_replay_resources_for_test() -> Result<usize, CudaRuntimeProbeError> {
    let mut parking = PARKED_REPLAY_RESOURCES
        .lock()
        .map_err(|_| CudaRuntimeProbeError::KernelLaunchFailed(-9_992))?;
    let mut drained = 0_usize;
    for slot in &mut parking.slots {
        let Some(resources) = slot.take() else {
            continue;
        };
        let stream = resources
            .stream
            .pooled
            .as_ref()
            .expect("parked replay retains its private stream")
            .stream;
        if let Err(error) = resources.stream.primary.synchronize_owned_stream(stream) {
            *slot = Some(resources);
            return Err(error);
        }
        drop(resources);
        drained += 1;
    }
    Ok(drained)
}

impl CudaInsertReplayUnknownQuiescence {
    /// The most recent failure that prevented a proof of stream quiescence.
    pub fn error(&self) -> &CudaRuntimeProbeError {
        &self.error
    }

    /// Retry the bounded completion fence while retaining exactly the original device, stream,
    /// and pinned-host owners.  This never replays the HtoD or kernel launch.
    pub fn retry_complete(mut self) -> CudaInsertReplayCompletion {
        self.submission
            .take()
            .expect("unknown quiescence retains one submission")
            .complete()
    }
}

fn replay_fence_failure_error(
    failure: crate::cuda_context::ReplayOwnedStreamFenceFailure,
) -> CudaRuntimeProbeError {
    match failure {
        #[cfg(test)]
        crate::cuda_context::ReplayOwnedStreamFenceFailure::SyntheticFenceNotAttempted => {
            CudaRuntimeProbeError::KernelLaunchFailed(-9_991)
        }
        crate::cuda_context::ReplayOwnedStreamFenceFailure::CudaApiReported(error) => error,
    }
}

fn validate_replay_inputs(
    source: &CudaResidentDeviceMemory,
    preparation: &FusedApplyPreparation<'_>,
    stamps: &[u64],
) -> Result<FusedApplyValidatedGeometry, CudaInsertReplayPrepareError> {
    let validated = validate_fused_apply_preparation(source, preparation)?;
    if stamps.len() != validated.k {
        return Err(CudaRuntimeProbeError::InvalidInputLength(stamps.len()).into());
    }
    Ok(validated)
}

fn replay_resource_geometry(
    owners: &[Arc<crate::resident_memory::CudaResidentDeviceAllocation>],
    staging_host: &PinnedHostBufferOwned,
    status_host: &PinnedHostBufferOwned,
    staging_device: &PooledDeviceBufferOwned,
    stream: &PooledStreamOwned,
    transient_ptx_staging_bytes: usize,
) -> Result<CudaI32InsertReplayResourceGeometry, CudaRuntimeProbeError> {
    let owner_array_backing_bytes = owners
        .len()
        .checked_mul(std::mem::size_of::<
            Arc<crate::resident_memory::CudaResidentDeviceAllocation>,
        >())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let pooled = stream
        .pooled
        .as_ref()
        .expect("replay preparation retains its pooled stream");
    let timing_event_count = u64::from(!pooled.start_event.is_null())
        .checked_add(u64::from(!pooled.stop_event.is_null()))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if timing_event_count != 0 && timing_event_count != 2 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(timing_event_count).unwrap_or(usize::MAX),
        ));
    }
    Ok(CudaI32InsertReplayResourceGeometry {
        pinned_cuda_allocation_count: u64::try_from(owners.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        owner_array_backing_bytes,
        owner_array_allocation_slots: u64::from(owner_array_backing_bytes != 0),
        pinned_htod_staging_bytes: u64::try_from(staging_host.capacity)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        pinned_htod_staging_allocation_slots: 1,
        pinned_status_bytes: u64::try_from(status_host.capacity)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        pinned_status_allocation_slots: 1,
        pooled_device_staging_bytes: u64::try_from(staging_device.capacity)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        pooled_device_staging_allocation_slots: 1,
        private_stream_count: 1,
        private_stream_scratch_bytes: u64::try_from(crate::POOLED_STREAM_SCRATCH_BYTES)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        private_stream_scratch_allocation_slots: 1,
        private_stream_timing_event_count: timing_event_count,
        private_stream_timing_event_host_backing_bytes: 0,
        private_stream_timing_event_host_backing_slots: 0,
        transient_ptx_staging_bytes: u64::try_from(transient_ptx_staging_bytes)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        transient_ptx_staging_allocation_slots: 1,
    })
}

fn materialize_owners(
    source: &CudaResidentDeviceMemory,
    preparation: &FusedApplyPreparation<'_>,
    validated: &FusedApplyValidatedGeometry,
) -> Result<Box<[Arc<crate::resident_memory::CudaResidentDeviceAllocation>]>, CudaRuntimeProbeError>
{
    let owner_count = usize::try_from(validated.footprint.pinned_cuda_allocation_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut owner_slots =
        Box::<[Arc<crate::resident_memory::CudaResidentDeviceAllocation>]>::new_uninit_slice(
            owner_count,
        );
    let mut written = 0_usize;
    visit_preparation_owner_memories(source, preparation, |position, memory| {
        let already_seen = {
            let target_identity = memory.allocation_identity();
            let mut seen = false;
            visit_preparation_owner_memories(source, preparation, |prior, candidate| {
                if prior < position && candidate.allocation_identity() == target_identity {
                    seen = true;
                }
            });
            seen
        };
        if !already_seen {
            owner_slots[written].write(memory.allocation_arc());
            written += 1;
        }
    });
    if written != owner_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(written));
    }
    // SAFETY: full validation counted this exact deduplicated visitor sequence; every slot above
    // is initialized exactly once before the boxed owner becomes retained submission state.
    Ok(unsafe { owner_slots.assume_init() })
}

fn write_staging_image(
    staging: &mut [u8],
    preparation: &FusedApplyPreparation<'_>,
    stamps: &[u64],
    validated: &FusedApplyValidatedGeometry,
) -> Result<(), CudaRuntimeProbeError> {
    if staging.len() != validated.total_staging_bytes || stamps.len() != validated.k {
        return Err(CudaRuntimeProbeError::InvalidInputLength(staging.len()));
    }
    staging.fill(0);
    staging[0..4].copy_from_slice(&validated.k_u32.to_le_bytes());
    staging[4..8].copy_from_slice(&validated.cols_u32.to_le_bytes());
    staging[8..12].copy_from_slice(&validated.pk_col.to_le_bytes());
    staging[12..16].copy_from_slice(&preparation.base_row.to_le_bytes());
    staging[16..20].copy_from_slice(&validated.index_mask.to_le_bytes());
    staging[20..24].copy_from_slice(&validated.index_shift.to_le_bytes());
    staging[24..28].copy_from_slice(&u32::from(preparation.row_ids.is_some()).to_le_bytes());
    staging[32..40].copy_from_slice(&validated.index_ptr.to_le_bytes());
    staging[40..48].copy_from_slice(&validated.created_by_dest.to_le_bytes());
    staging[48..56].copy_from_slice(&validated.row_id_dest.to_le_bytes());
    for (column, destination) in preparation.columns.iter().enumerate() {
        let pointer = destination
            .memory
            .device_ptr()
            .checked_add(destination.byte_offset)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let offset = 80 + column * std::mem::size_of::<u64>();
        staging[offset..offset + 8].copy_from_slice(&pointer.to_le_bytes());
    }
    let values_offset = validated.header_bytes;
    for (position, value) in preparation.values.iter().copied().enumerate() {
        let offset = values_offset
            .checked_add(
                position
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(position))?,
            )
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(position))?;
        staging[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    let stamps_offset = values_offset + validated.values_padded;
    for (position, stamp) in stamps.iter().copied().enumerate() {
        let offset = stamps_offset
            .checked_add(
                position
                    .checked_mul(std::mem::size_of::<u64>())
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(position))?,
            )
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(position))?;
        staging[offset..offset + 8].copy_from_slice(&stamp.to_le_bytes());
    }
    if let Some((row_ids, _)) = preparation.row_ids.as_ref() {
        let row_ids_offset = stamps_offset
            .checked_add(validated.stamp_bytes)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        for (position, row_id) in row_ids.iter().copied().enumerate() {
            let offset = row_ids_offset
                .checked_add(
                    position
                        .checked_mul(std::mem::size_of::<u64>())
                        .ok_or(CudaRuntimeProbeError::InvalidInputLength(position))?,
                )
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(position))?;
            staging[offset..offset + 8].copy_from_slice(&row_id.to_le_bytes());
        }
    }
    Ok(())
}
