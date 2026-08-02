//! Sealed asynchronous DtoH completion for one bounded batch of GPU SHA-256 roots.
//!
//! This is execution transport only.  It retains every source allocation, descriptor staging
//! buffer, unique digest output, pinned result buffer, primary context, and private stream until
//! the covering fence proves quiescence.  It never exposes digest bytes to its caller.

use std::{
    ffi::c_void,
    fmt,
    sync::{Arc, Mutex},
};

use crate::{
    cuda_context::{PinnedHostBufferOwned, PooledDeviceBufferOwned, PooledStreamOwned},
    sha256::{
        checked_completion_descriptor_bytes, checked_completion_span,
        launch_runtime_generation_v1_genesis_kernel, launch_sha256_kernel,
        resolve_runtime_generation_v1_genesis_launch, resolve_sha256_launch, CuLaunchKernel,
    },
    CudaResidentReadSource, CudaRuntimeProbeError, CudaSha256DeviceBuffer, GpuPrimaryContext,
    CUDA_SHA256_DIGEST_BYTES, CUDA_SHA256_MAX_INPUT_BYTES,
};

/// Fixed output slots for the closed runtime-generation-v1 genesis program: table-map empty
/// roots at depths 0..64, status-view empty roots at depths 0..64, then the database root.
pub const CUDA_RUNTIME_GENERATION_V1_GENESIS_ROOT_SLOTS: usize = 131;

const CUDA_RUNTIME_GENERATION_V1_DATABASE_ID_BYTES: u64 = 16;
const CUDA_RUNTIME_GENERATION_V1_ROOT_FORMAT: u64 = 1;

/// A caller-chosen, nonzero identity for one whole SHA completion batch.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CudaSha256BatchId(u64);

impl CudaSha256BatchId {
    pub fn new(value: u64) -> Result<Self, CudaRuntimeProbeError> {
        if value == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
        Ok(Self(value))
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for CudaSha256BatchId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("CudaSha256BatchId")
            .field(&self.0)
            .finish()
    }
}

/// Immutable transport identity for exactly one batch slot.  Semantic root-domain descriptors
/// remain owned by the engine's sealed layout; this execution descriptor prevents a caller from
/// sorting, duplicating, or splicing DtoH slots across batches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaSha256CompletionDescriptor {
    batch_id: CudaSha256BatchId,
    slot_ordinal: u32,
}

impl CudaSha256CompletionDescriptor {
    pub fn new(batch_id: CudaSha256BatchId, slot_ordinal: u32) -> Self {
        Self {
            batch_id,
            slot_ordinal,
        }
    }

    pub fn batch_id(self) -> CudaSha256BatchId {
        self.batch_id
    }

    pub fn slot_ordinal(self) -> u32 {
        self.slot_ordinal
    }
}

/// One resident SHA preimage range paired with its immutable output-slot descriptor.
#[derive(Clone, Copy, Debug)]
pub struct CudaSha256CompletionInput<'a> {
    pub input: CudaSha256DeviceBuffer<'a>,
    pub descriptor: CudaSha256CompletionDescriptor,
}

impl<'a> CudaSha256CompletionInput<'a> {
    pub fn new(
        input: CudaSha256DeviceBuffer<'a>,
        descriptor: CudaSha256CompletionDescriptor,
    ) -> Self {
        Self { input, descriptor }
    }
}

/// One opaque 32-byte GPU completion digest.  Its representation intentionally has no byte,
/// slice, pointer, formatter, hex, serde, or hash accessor.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct OpaqueCudaSha256Digest([u8; 32]);

impl fmt::Debug for OpaqueCudaSha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OpaqueCudaSha256Digest(<opaque>)")
    }
}

impl OpaqueCudaSha256Digest {
    /// Internal bridge for a quiesced fixed-slot GPU operator.  It deliberately remains
    /// crate-private so only execution transports can re-label an already-produced device
    /// digest; callers still have no constructor or byte accessor.
    pub(crate) fn from_runtime_generation_rebuild_slot(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

/// A digest token can only be obtained from a fully quiesced whole batch.  It retains immutable
/// batch/slot identity together with the opaque digest and is intentionally not Clone or Copy.
pub struct OpaqueCudaSha256Token {
    descriptor: CudaSha256CompletionDescriptor,
    digest: OpaqueCudaSha256Digest,
}

impl OpaqueCudaSha256Token {
    pub fn descriptor(&self) -> CudaSha256CompletionDescriptor {
        self.descriptor
    }

    /// This remains opaque: consumers can carry it only into a sealed domain-specific materializer.
    pub fn opaque_digest(&self) -> OpaqueCudaSha256Digest {
        self.digest
    }
}

impl fmt::Debug for OpaqueCudaSha256Token {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OpaqueCudaSha256Token(<opaque>)")
    }
}

/// All output tokens for one GPU completion.  It exposes no per-slot extraction API: consuming
/// the batch gives a sealed caller one borrowed view of every slot, then destroys the batch.
pub struct OpaqueCudaSha256Batch {
    batch_id: CudaSha256BatchId,
    tokens: Box<[OpaqueCudaSha256Token]>,
}

impl OpaqueCudaSha256Batch {
    pub fn batch_id(&self) -> CudaSha256BatchId {
        self.batch_id
    }

    pub fn consume<R>(
        self,
        consume: impl FnOnce(CudaSha256BatchId, &[OpaqueCudaSha256Token]) -> R,
    ) -> R {
        consume(self.batch_id, &self.tokens)
    }
}

impl fmt::Debug for OpaqueCudaSha256Batch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OpaqueCudaSha256Batch(<opaque>)")
    }
}

/// A failure before the first enqueue.  All resource reservations are either absent or unwind
/// locally; no stream can still reference them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaSha256CompletionPrepareError {
    Runtime(CudaRuntimeProbeError),
    AsyncTransportUnavailable,
    PinnedHostStagingUnavailable,
}

impl From<CudaRuntimeProbeError> for CudaSha256CompletionPrepareError {
    fn from(value: CudaRuntimeProbeError) -> Self {
        Self::Runtime(value)
    }
}

/// Prepared, fully owned SHA completion.  It has one enqueue attempt; dropping it before
/// [`Self::enqueue`] releases only unsubmitted reservations.
#[must_use = "a prepared CUDA SHA completion must be enqueued or intentionally dropped"]
pub struct PreparedCudaSha256Completion {
    primary: Arc<GpuPrimaryContext>,
    resources: Sha256CompletionResources,
    descriptors: Box<[CudaSha256CompletionDescriptor]>,
    batch_id: CudaSha256BatchId,
    function: *mut c_void,
    launch: CuLaunchKernel,
    work: Sha256CompletionWork,
}

/// One in-flight SHA launch with all host/device resources retained until a fence succeeds.
#[must_use = "an in-flight CUDA SHA completion must be completed or safely dropped"]
pub struct CudaSha256Submission {
    primary: Arc<GpuPrimaryContext>,
    resources: Option<Sha256CompletionResources>,
    descriptors: Box<[CudaSha256CompletionDescriptor]>,
    batch_id: CudaSha256BatchId,
    function: *mut c_void,
    launch: CuLaunchKernel,
    work: Sha256CompletionWork,
    phase: Sha256CompletionPhase,
    /// The first terminal CUDA result after submission.  A later fence can leave quiescence
    /// unknown, but it may not redeem this attempt when a retry eventually drains the stream.
    first_terminal_error: Option<CudaRuntimeProbeError>,
    /// Any real CUDA API result after submission keeps these exact private-stream pool leases out
    /// of reuse permanently, even after a later fence proves the stream is idle.
    driver_error_observed: bool,
}

/// A quiesced outcome proves every HtoD/kernel/DtoH access has stopped.  An unknown fence retains
/// the exact submission and may retry *only* the fence; it never launches or copies again.
#[must_use = "unknown CUDA quiescence must be retained or retried; dropping it parks resources"]
#[allow(
    clippy::large_enum_variant,
    reason = "boxing after a CUDA enqueue would allocate on the exceptional safety path"
)]
pub enum CudaSha256Completion {
    Quiesced(Result<OpaqueCudaSha256Batch, CudaRuntimeProbeError>),
    UnknownQuiescence(CudaSha256UnknownQuiescence),
}

/// The fail-closed owner for a submission whose private stream could not be proven idle.
#[must_use = "unknown CUDA quiescence retains in-flight resources until retried or parked"]
pub struct CudaSha256UnknownQuiescence {
    error: CudaRuntimeProbeError,
    submission: Option<CudaSha256Submission>,
}

struct Sha256CompletionResources {
    // The device launch record contains raw device pointers. These guards retain every allocation
    // those pointers can address even after the caller's source wrappers are dropped.
    _source_owners: Box<[Arc<crate::resident_memory::CudaResidentDeviceAllocation>]>,
    descriptor_host: PinnedHostBufferOwned,
    descriptor_device: PooledDeviceBufferOwned,
    output_host: PinnedHostBufferOwned,
    output_device: PooledDeviceBufferOwned,
    stream: PooledStreamOwned,
    descriptor_bytes: usize,
    output_bytes: usize,
}

#[derive(Clone, Copy)]
enum Sha256CompletionWork {
    DescriptorBatch { input_count: u32 },
    RuntimeGenerationV1Genesis,
}

const PARKED_SHA256_COMPLETION_CAPACITY: usize = 8;

struct Sha256CompletionParkingLot {
    slots: [Option<Sha256CompletionResources>; PARKED_SHA256_COMPLETION_CAPACITY],
}

static PARKED_SHA256_COMPLETIONS: Mutex<Sha256CompletionParkingLot> =
    Mutex::new(Sha256CompletionParkingLot {
        slots: [const { None }; PARKED_SHA256_COMPLETION_CAPACITY],
    });

// A successfully drained stream is not automatically reusable after a CUDA API result.  Keep the
// exact stream, DMA backings, and source guards here so another completion cannot inherit a
// context-bound poisoned lease.
const QUARANTINED_SHA256_COMPLETION_CAPACITY: usize = 8;

struct Sha256CompletionQuarantineLot {
    slots: [Option<Sha256CompletionResources>; QUARANTINED_SHA256_COMPLETION_CAPACITY],
}

static QUARANTINED_SHA256_COMPLETIONS: Mutex<Sha256CompletionQuarantineLot> =
    Mutex::new(Sha256CompletionQuarantineLot {
        slots: [const { None }; QUARANTINED_SHA256_COMPLETION_CAPACITY],
    });

#[cfg(test)]
thread_local! {
    static SHA256_SUCCESSFUL_ENQUEUES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[derive(Debug, Clone)]
enum Sha256CompletionPhase {
    InFlight,
    ReadbackQueued,
    TerminalFailure { drain_required: bool },
    Quiesced,
}

impl Sha256CompletionPhase {
    fn drain_required(&self) -> bool {
        matches!(
            self,
            Self::InFlight
                | Self::ReadbackQueued
                | Self::TerminalFailure {
                    drain_required: true,
                    ..
                }
        )
    }
}

// SAFETY: the owner retains every allocation/host buffer/stream the queued work can touch, and
// re-binds the primary context on completion.  No public raw pointer exits this state machine.
unsafe impl Send for PreparedCudaSha256Completion {}
unsafe impl Send for CudaSha256Submission {}
unsafe impl Send for CudaSha256UnknownQuiescence {}
unsafe impl Send for Sha256CompletionResources {}

impl PreparedCudaSha256Completion {
    /// Validate the whole ordered batch and reserve every backing before its one enqueue edge.
    pub fn prepare(
        inputs: &[CudaSha256CompletionInput<'_>],
    ) -> Result<Self, CudaSha256CompletionPrepareError> {
        let validated = validate_completion_inputs(inputs)?;
        let input_count = u32::try_from(inputs.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(inputs.len()))?;
        Self::prepare_validated(
            validated,
            Sha256CompletionWork::DescriptorBatch { input_count },
        )
    }

    /// Prepare the one closed runtime-generation-v1 genesis root program.  The program admits
    /// only an exact 16-byte, device-resident database identity and a caller-chosen batch ID; it
    /// fixes root format v1, all five canonical ASCII domains, zero status subtree counts, the
    /// 64-to-0 child chains, and all 131 semantic output slots inside the CUDA kernel.
    ///
    /// This is deliberately not a generic root-construction API.  It exists so the engine's
    /// sealed layout can consume one opaque whole batch without ever staging child digests or
    /// domain text through host memory.
    pub fn prepare_runtime_generation_v1_genesis(
        database_id: CudaSha256DeviceBuffer<'_>,
        batch_id: CudaSha256BatchId,
    ) -> Result<Self, CudaSha256CompletionPrepareError> {
        Self::prepare_validated(
            validate_runtime_generation_v1_genesis_input(database_id, batch_id)?,
            Sha256CompletionWork::RuntimeGenerationV1Genesis,
        )
    }

    fn prepare_validated(
        validated: ValidatedCompletionInputs,
        work: Sha256CompletionWork,
    ) -> Result<Self, CudaSha256CompletionPrepareError> {
        let primary = validated.primary;
        if primary.cu_memcpy_htod_async.is_none() || primary.cu_memcpy_dtoh_async.is_none() {
            return Err(CudaSha256CompletionPrepareError::AsyncTransportUnavailable);
        }
        primary.set_current()?;

        let mut descriptor_host = primary
            .lease_pinned_host_buffer_owned(validated.descriptor_bytes)
            .ok_or(CudaSha256CompletionPrepareError::PinnedHostStagingUnavailable)?;
        encode_device_descriptors(
            descriptor_host.as_mut_bytes(validated.descriptor_bytes)?,
            &validated.device_descriptors,
        )?;
        let descriptor_device = primary.lease_device_buffer_owned(validated.descriptor_bytes)?;
        let output_device = primary.lease_device_buffer_owned(validated.output_bytes)?;
        if descriptor_device.ptr == output_device.ptr {
            return Err(CudaRuntimeProbeError::InvalidInputLength(validated.output_bytes).into());
        }
        if validated
            .source_spans
            .iter()
            .any(|(source_ptr, source_len)| {
                device_spans_overlap(
                    *source_ptr,
                    *source_len,
                    output_device.ptr,
                    validated.output_bytes as u64,
                )
            })
        {
            return Err(CudaRuntimeProbeError::InvalidInputLength(validated.output_bytes).into());
        }
        let output_host = primary
            .lease_pinned_host_buffer_owned(validated.output_bytes)
            .ok_or(CudaSha256CompletionPrepareError::PinnedHostStagingUnavailable)?;
        let stream = PooledStreamOwned {
            primary: Arc::clone(&primary),
            pooled: Some(primary.acquire_pooled_stream()?),
        };
        let (function, launch) = match work {
            Sha256CompletionWork::DescriptorBatch { .. } => resolve_sha256_launch(&primary)?,
            Sha256CompletionWork::RuntimeGenerationV1Genesis => {
                resolve_runtime_generation_v1_genesis_launch(&primary)?
            }
        };
        Ok(Self {
            primary,
            resources: Sha256CompletionResources {
                _source_owners: validated.source_owners,
                descriptor_host,
                descriptor_device,
                output_host,
                output_device,
                stream,
                descriptor_bytes: validated.descriptor_bytes,
                output_bytes: validated.output_bytes,
            },
            descriptors: validated.descriptors,
            batch_id: validated.batch_id,
            function,
            launch,
            work,
        })
    }

    /// The sole asynchronous submit edge.  Every post-enqueue error remains inside the returned
    /// owner so completion can prove quiescence or retain unknown resources.
    pub fn enqueue(self) -> CudaSha256Submission {
        let mut submission = CudaSha256Submission {
            primary: self.primary,
            resources: Some(self.resources),
            descriptors: self.descriptors,
            batch_id: self.batch_id,
            function: self.function,
            launch: self.launch,
            work: self.work,
            phase: Sha256CompletionPhase::InFlight,
            first_terminal_error: None,
            driver_error_observed: false,
        };
        submission.enqueue_inner();
        submission
    }
}

impl CudaSha256Submission {
    fn resources(&self) -> &Sha256CompletionResources {
        self.resources
            .as_ref()
            .expect("SHA completion resources remain owned until quiescence")
    }

    fn stream(&self) -> *mut c_void {
        self.resources()
            .stream
            .pooled
            .as_ref()
            .expect("SHA completion stream remains owned until quiescence")
            .stream
    }

    fn enqueue_inner(&mut self) {
        if let Err(error) = self.primary.bind_owned_stream_before_submission() {
            self.retain_first_terminal_error(&error);
            self.phase = Sha256CompletionPhase::TerminalFailure {
                drain_required: false,
            };
            return;
        }
        let stream = self.stream();
        let resources = self.resources();
        if let Err(error) = self.primary.enqueue_owned_stream_htod(
            resources.descriptor_device.ptr,
            resources.descriptor_host.ptr.cast_const(),
            resources.descriptor_bytes,
            stream,
        ) {
            self.retain_driver_error(&error);
            self.phase = Sha256CompletionPhase::TerminalFailure {
                drain_required: true,
            };
            return;
        }
        let launch_status = match self.work {
            Sha256CompletionWork::DescriptorBatch { input_count } => unsafe {
                launch_sha256_kernel(
                    self.launch,
                    self.function,
                    input_count,
                    resources.descriptor_device.ptr,
                    resources.output_device.ptr,
                    stream,
                )
            },
            Sha256CompletionWork::RuntimeGenerationV1Genesis => unsafe {
                launch_runtime_generation_v1_genesis_kernel(
                    self.launch,
                    self.function,
                    resources.descriptor_device.ptr,
                    resources.output_device.ptr,
                    stream,
                )
            },
        };
        let launch = self.primary.check_owned_stream_launch_result(launch_status);
        if let Err(error) = launch {
            self.retain_driver_error(&error);
            self.phase = Sha256CompletionPhase::TerminalFailure {
                drain_required: true,
            };
        } else {
            self.primary.after_owned_stream_enqueue();
            #[cfg(test)]
            SHA256_SUCCESSFUL_ENQUEUES.with(|value| value.set(value.get() + 1));
        }
    }

    fn queue_readback(&mut self) -> Result<(), CudaRuntimeProbeError> {
        if !matches!(self.phase, Sha256CompletionPhase::InFlight) {
            return Ok(());
        }
        self.primary
            .bind_owned_stream_for_completion()
            .map_err(|failure| failure.error())?;
        let stream = self.stream();
        let resources = self.resources();
        self.primary.enqueue_owned_stream_dtoh(
            resources.output_host.ptr,
            resources.output_device.ptr,
            resources.output_bytes,
            stream,
        )?;
        self.phase = Sha256CompletionPhase::ReadbackQueued;
        Ok(())
    }

    fn drain_stream(&self) -> Result<(), crate::cuda_context::CompletionOwnedStreamFenceFailure> {
        self.primary
            .synchronize_owned_stream_for_completion(self.stream())
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

    fn unknown(self, error: CudaRuntimeProbeError) -> CudaSha256Completion {
        CudaSha256Completion::UnknownQuiescence(CudaSha256UnknownQuiescence {
            error,
            submission: Some(self),
        })
    }

    /// Fence exactly the already-enqueued launch and its bounded DtoH.  Retrying an unknown
    /// outcome re-enters only this method's fence phase; no HtoD or kernel is launched twice.
    pub fn complete(mut self) -> CudaSha256Completion {
        if self.phase.drain_required() && matches!(self.phase, Sha256CompletionPhase::InFlight) {
            if let Err(error) = self.queue_readback() {
                self.retain_driver_error(&error);
                self.phase = Sha256CompletionPhase::TerminalFailure {
                    drain_required: true,
                };
            }
        }
        if self.phase.drain_required() {
            if let Err(failure) = self.drain_stream() {
                let error = failure.error();
                if let Some(cuda_error) = failure.cuda_error() {
                    self.retain_driver_error(cuda_error);
                }
                return self.unknown(error);
            }
        }
        let result = if let Some(error) = self.completion_error() {
            Err(error)
        } else if matches!(self.phase, Sha256CompletionPhase::ReadbackQueued) {
            self.tokens_after_quiescence()
        } else {
            Err(CudaRuntimeProbeError::KernelLaunchFailed(-1))
        };
        self.phase = Sha256CompletionPhase::Quiesced;
        self.quarantine_if_driver_error();
        CudaSha256Completion::Quiesced(result)
    }

    fn tokens_after_quiescence(&self) -> Result<OpaqueCudaSha256Batch, CudaRuntimeProbeError> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.resources().output_host.ptr.cast::<u8>(),
                self.resources().output_bytes,
            )
        };
        opaque_batch_from_quiesced_bytes(self.batch_id, &self.descriptors, bytes)
    }

    fn quarantine_if_driver_error(&mut self) {
        if self.driver_error_observed {
            if let Some(resources) = self.resources.take() {
                quarantine_sha256_completion_resources(resources);
            }
        }
    }
}

impl Drop for CudaSha256Submission {
    fn drop(&mut self) {
        if !self.phase.drain_required() {
            self.quarantine_if_driver_error();
            return;
        }
        match self.drain_stream() {
            Ok(()) => {
                self.phase = Sha256CompletionPhase::Quiesced;
                self.quarantine_if_driver_error();
            }
            Err(failure) => {
                if let Some(cuda_error) = failure.cuda_error() {
                    self.retain_driver_error(cuda_error);
                }
                if let Some(resources) = self.resources.take() {
                    if self.driver_error_observed {
                        quarantine_sha256_completion_resources(resources);
                    } else {
                        park_sha256_completion_resources(resources);
                    }
                }
            }
        }
    }
}

impl CudaSha256UnknownQuiescence {
    pub fn error(&self) -> &CudaRuntimeProbeError {
        &self.error
    }

    pub fn retry_complete(mut self) -> CudaSha256Completion {
        self.submission
            .take()
            .expect("unknown SHA quiescence retains one submission")
            .complete()
    }
}

fn park_sha256_completion_resources(resources: Sha256CompletionResources) {
    let Ok(mut parking) = PARKED_SHA256_COMPLETIONS.lock() else {
        std::mem::forget(resources);
        return;
    };
    if let Some(slot) = parking.slots.iter_mut().find(|slot| slot.is_none()) {
        *slot = Some(resources);
    } else {
        std::mem::forget(resources);
    }
}

fn quarantine_sha256_completion_resources(resources: Sha256CompletionResources) {
    let Ok(mut quarantine) = QUARANTINED_SHA256_COMPLETIONS.lock() else {
        std::mem::forget(resources);
        return;
    };
    if let Some(slot) = quarantine.slots.iter_mut().find(|slot| slot.is_none()) {
        *slot = Some(resources);
    } else {
        // Quarantine exhaustion must fail closed: these leases may no longer re-enter a pool.
        std::mem::forget(resources);
    }
}

struct ValidatedCompletionInputs {
    primary: Arc<GpuPrimaryContext>,
    batch_id: CudaSha256BatchId,
    descriptors: Box<[CudaSha256CompletionDescriptor]>,
    device_descriptors: Box<[u64]>,
    source_owners: Box<[Arc<crate::resident_memory::CudaResidentDeviceAllocation>]>,
    source_spans: Box<[(u64, u64)]>,
    descriptor_bytes: usize,
    output_bytes: usize,
}

fn validate_completion_inputs(
    inputs: &[CudaSha256CompletionInput<'_>],
) -> Result<ValidatedCompletionInputs, CudaSha256CompletionPrepareError> {
    let descriptor_bytes = checked_completion_descriptor_bytes(inputs.len())?;
    let first = inputs
        .first()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    let primary = first.input.memory.primary_arc();
    let gpu_id = first.input.memory.metadata().gpu_id;
    let batch_id = first.descriptor.batch_id();
    let output_bytes_u64 = u64::try_from(inputs.len())
        .ok()
        .and_then(|count| count.checked_mul(CUDA_SHA256_DIGEST_BYTES))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_bytes = usize::try_from(output_bytes_u64)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut device_descriptors = Vec::with_capacity(inputs.len() * 2);
    let mut descriptors = Vec::with_capacity(inputs.len());
    let mut source_owners = Vec::new();
    let mut source_spans = Vec::with_capacity(inputs.len());
    for (position, completion_input) in inputs.iter().enumerate() {
        let expected_ordinal = u32::try_from(position)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(position))?;
        if completion_input.descriptor.batch_id() != batch_id
            || completion_input.descriptor.slot_ordinal() != expected_ordinal
            || completion_input.input.byte_len > CUDA_SHA256_MAX_INPUT_BYTES
            || completion_input.input.memory.metadata().gpu_id != gpu_id
            || !Arc::ptr_eq(&completion_input.input.memory.primary_arc(), &primary)
        {
            return Err(CudaRuntimeProbeError::InvalidInputLength(position).into());
        }
        let device_ptr = checked_completion_span(
            completion_input.input.memory,
            completion_input.input.byte_offset,
            completion_input.input.byte_len,
        )?;
        let owner = completion_input.input.memory.allocation_arc();
        if !source_owners
            .iter()
            .any(|existing| Arc::ptr_eq(existing, &owner))
        {
            source_owners.push(owner);
        }
        device_descriptors.extend_from_slice(&[device_ptr, completion_input.input.byte_len]);
        source_spans.push((device_ptr, completion_input.input.byte_len));
        descriptors.push(completion_input.descriptor);
    }
    let expected_words = descriptor_bytes / std::mem::size_of::<u64>();
    if device_descriptors.len() != expected_words {
        return Err(CudaRuntimeProbeError::InvalidInputLength(device_descriptors.len()).into());
    }
    Ok(ValidatedCompletionInputs {
        primary,
        batch_id,
        descriptors: descriptors.into_boxed_slice(),
        device_descriptors: device_descriptors.into_boxed_slice(),
        source_owners: source_owners.into_boxed_slice(),
        source_spans: source_spans.into_boxed_slice(),
        descriptor_bytes,
        output_bytes,
    })
}

fn validate_runtime_generation_v1_genesis_input(
    database_id: CudaSha256DeviceBuffer<'_>,
    batch_id: CudaSha256BatchId,
) -> Result<ValidatedCompletionInputs, CudaSha256CompletionPrepareError> {
    if database_id.byte_len != CUDA_RUNTIME_GENERATION_V1_DATABASE_ID_BYTES {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(database_id.byte_len).unwrap_or(usize::MAX),
        )
        .into());
    }
    let primary = database_id.memory.primary_arc();
    let database_ptr = checked_completion_span(
        database_id.memory,
        database_id.byte_offset,
        database_id.byte_len,
    )?;
    let output_bytes = CUDA_RUNTIME_GENERATION_V1_GENESIS_ROOT_SLOTS
        .checked_mul(CUDA_SHA256_DIGEST_BYTES as usize)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let descriptors = (0..CUDA_RUNTIME_GENERATION_V1_GENESIS_ROOT_SLOTS)
        .map(|ordinal| {
            Ok(CudaSha256CompletionDescriptor::new(
                batch_id,
                u32::try_from(ordinal)
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(ordinal))?,
            ))
        })
        .collect::<Result<Vec<_>, CudaRuntimeProbeError>>()?
        .into_boxed_slice();
    // The fixed 16-byte device configuration is `[database_id_device_ptr, root_format_v1]`.
    // It is still staged through the owned pinned/device descriptor buffers and has no room for
    // caller-supplied domain text, root kind, ordinal, child digest, or status count.
    let device_descriptors = Box::new([database_ptr, CUDA_RUNTIME_GENERATION_V1_ROOT_FORMAT]);
    let descriptor_bytes = std::mem::size_of_val(device_descriptors.as_ref());
    debug_assert_eq!(
        descriptor_bytes,
        checked_completion_descriptor_bytes(1).expect("one configuration record")
    );
    Ok(ValidatedCompletionInputs {
        primary,
        batch_id,
        descriptors,
        device_descriptors,
        source_owners: Box::new([database_id.memory.allocation_arc()]),
        source_spans: Box::new([(database_ptr, database_id.byte_len)]),
        descriptor_bytes,
        output_bytes,
    })
}

fn device_spans_overlap(left_ptr: u64, left_len: u64, right_ptr: u64, right_len: u64) -> bool {
    if left_len == 0 || right_len == 0 {
        return false;
    }
    let Some(left_end) = left_ptr.checked_add(left_len) else {
        return true;
    };
    let Some(right_end) = right_ptr.checked_add(right_len) else {
        return true;
    };
    left_ptr < right_end && right_ptr < left_end
}

fn encode_device_descriptors(
    destination: &mut [u8],
    descriptors: &[u64],
) -> Result<(), CudaRuntimeProbeError> {
    if destination.len() != std::mem::size_of_val(descriptors) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(destination.len()));
    }
    for (slot, descriptor) in destination
        .chunks_exact_mut(std::mem::size_of::<u64>())
        .zip(descriptors)
    {
        slot.copy_from_slice(&descriptor.to_ne_bytes());
    }
    Ok(())
}

fn opaque_batch_from_quiesced_bytes(
    batch_id: CudaSha256BatchId,
    descriptors: &[CudaSha256CompletionDescriptor],
    bytes: &[u8],
) -> Result<OpaqueCudaSha256Batch, CudaRuntimeProbeError> {
    if bytes.len()
        != descriptors
            .len()
            .checked_mul(CUDA_SHA256_DIGEST_BYTES as usize)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes.len()));
    }
    // Do not expose a partial batch: a zero digest is not a logical root commitment, so the
    // entire quiesced handoff fails before constructing even the first opaque token.
    if bytes
        .chunks_exact(CUDA_SHA256_DIGEST_BYTES as usize)
        .any(|digest| digest.iter().all(|byte| *byte == 0))
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let tokens = descriptors
        .iter()
        .copied()
        .zip(bytes.chunks_exact(CUDA_SHA256_DIGEST_BYTES as usize))
        .map(|(descriptor, bytes)| {
            let mut digest = [0_u8; CUDA_SHA256_DIGEST_BYTES as usize];
            digest.copy_from_slice(bytes);
            OpaqueCudaSha256Token {
                descriptor,
                digest: OpaqueCudaSha256Digest(digest),
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    Ok(OpaqueCudaSha256Batch { batch_id, tokens })
}

#[cfg(test)]
pub(crate) fn parked_sha256_completion_count_for_test() -> usize {
    PARKED_SHA256_COMPLETIONS
        .lock()
        .map(|parking| parking.slots.iter().filter(|slot| slot.is_some()).count())
        .unwrap_or(0)
}

#[cfg(test)]
pub(crate) fn parked_sha256_completion_contains_source_owner_for_test(identity: usize) -> bool {
    PARKED_SHA256_COMPLETIONS
        .lock()
        .map(|parking| {
            parking.slots.iter().flatten().any(|resources| {
                resources
                    ._source_owners
                    .iter()
                    .any(|owner| Arc::as_ptr(owner) as usize == identity)
            })
        })
        .unwrap_or(false)
}

#[cfg(test)]
pub(crate) fn quarantined_sha256_completion_count_for_test() -> usize {
    QUARANTINED_SHA256_COMPLETIONS
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
pub(crate) fn quarantined_sha256_completion_contains_source_owner_for_test(
    identity: usize,
) -> bool {
    QUARANTINED_SHA256_COMPLETIONS
        .lock()
        .map(|quarantine| {
            quarantine.slots.iter().flatten().any(|resources| {
                resources
                    ._source_owners
                    .iter()
                    .any(|owner| Arc::as_ptr(owner) as usize == identity)
            })
        })
        .unwrap_or(false)
}

#[cfg(test)]
fn sha256_successful_enqueue_count_for_test() -> u64 {
    SHA256_SUCCESSFUL_ENQUEUES.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn drain_parked_sha256_completions_for_test() -> Result<usize, CudaRuntimeProbeError> {
    let mut parking = PARKED_SHA256_COMPLETIONS
        .lock()
        .map_err(|_| CudaRuntimeProbeError::KernelLaunchFailed(-9_995))?;
    let mut drained = 0_usize;
    for slot in &mut parking.slots {
        let Some(resources) = slot.take() else {
            continue;
        };
        let stream = resources
            .stream
            .pooled
            .as_ref()
            .expect("parked SHA completion retains its private stream")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cuda_context::{
            distinct_gpu_primary_context_for_test, fail_next_owned_stream_dtoh_enqueue_for_test,
            fail_next_owned_stream_htod_enqueue_after_dispatch_for_test,
            fail_next_owned_stream_launch_after_dispatch_for_test,
            fail_next_owned_stream_post_submit_bind_raw_cuda_result_for_test,
            fail_next_owned_stream_pre_submit_bind_raw_cuda_result_for_test,
            fail_next_owned_stream_synthetic_fence_not_attempted_for_test,
            fail_owned_stream_syncs_for_test,
        },
        CudaDriverRuntime,
    };
    use sha2::{Digest, Sha256};

    fn runtime() -> Option<CudaDriverRuntime> {
        let runtime = CudaDriverRuntime::probe().ok()?;
        let snapshot = runtime.snapshot();
        (snapshot.driver_available && snapshot.device_count > 0).then_some(runtime)
    }

    fn one_input<'a>(
        source: &'a crate::CudaResidentDeviceMemory,
        batch_id: CudaSha256BatchId,
    ) -> CudaSha256CompletionInput<'a> {
        CudaSha256CompletionInput::new(
            CudaSha256DeviceBuffer::new(source, 0, source.metadata().allocated_bytes),
            CudaSha256CompletionDescriptor::new(batch_id, 0),
        )
    }

    fn genesis_submission(
        source: &crate::CudaResidentDeviceMemory,
        batch_id: CudaSha256BatchId,
    ) -> CudaSha256Submission {
        PreparedCudaSha256Completion::prepare_runtime_generation_v1_genesis(
            CudaSha256DeviceBuffer::new(source, 0, 16),
            batch_id,
        )
        .expect("genesis submission preparation")
        .enqueue()
    }

    fn runtime_generation_v1_digest(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update((domain.len() as u64).to_le_bytes());
        hasher.update(domain);
        for field in fields {
            hasher.update(field);
        }
        hasher.finalize().into()
    }

    fn runtime_generation_v1_genesis_oracle(database_id: [u8; 16]) -> [[u8; 32]; 131] {
        const MAP_EMPTY_LEAF: &[u8] = b"gpu-db/runtime-generation/map-empty-leaf/v1";
        const MAP_EMPTY_NODE: &[u8] = b"gpu-db/runtime-generation/map-empty-node/v1";
        const STATUS_EMPTY_LEAF: &[u8] = b"gpu-db/runtime-generation/status-empty-leaf/v1";
        const STATUS_EMPTY_NODE: &[u8] = b"gpu-db/runtime-generation/status-empty-node/v1";
        const DATABASE_ROOT: &[u8] = b"gpu-db/runtime-generation/database-root/v1";

        let root_format = 1_u16.to_le_bytes();
        let zero_count = 0_u64.to_le_bytes();
        let mut roots = [[0_u8; 32]; CUDA_RUNTIME_GENERATION_V1_GENESIS_ROOT_SLOTS];
        roots[64] = runtime_generation_v1_digest(MAP_EMPTY_LEAF, &[&root_format, &database_id]);
        for depth in (0_u8..64).rev() {
            let child = roots[usize::from(depth) + 1];
            roots[usize::from(depth)] = runtime_generation_v1_digest(
                MAP_EMPTY_NODE,
                &[&root_format, &database_id, &[depth], &child, &child],
            );
        }
        roots[65 + 64] = runtime_generation_v1_digest(
            STATUS_EMPTY_LEAF,
            &[&root_format, &database_id, &zero_count],
        );
        for depth in (0_u8..64).rev() {
            let child = roots[65 + usize::from(depth) + 1];
            roots[65 + usize::from(depth)] = runtime_generation_v1_digest(
                STATUS_EMPTY_NODE,
                &[
                    &root_format,
                    &database_id,
                    &[depth],
                    &zero_count,
                    &child,
                    &child,
                ],
            );
        }
        roots[130] =
            runtime_generation_v1_digest(DATABASE_ROOT, &[&root_format, &database_id, &roots[0]]);
        roots
    }

    #[test]
    fn opaque_batch_rejects_zero_digest_before_exposing_any_slot() {
        let batch_id = CudaSha256BatchId::new(7).expect("batch id");
        let descriptors = [CudaSha256CompletionDescriptor::new(batch_id, 0)];
        assert!(matches!(
            opaque_batch_from_quiesced_bytes(batch_id, &descriptors, &[0; 32]),
            Err(CudaRuntimeProbeError::InvalidInputLength(0))
        ));
    }

    #[test]
    fn completion_tokens_are_opaque_and_whole_batch_consumed() {
        let batch_id = CudaSha256BatchId::new(8).expect("batch id");
        let descriptors = [CudaSha256CompletionDescriptor::new(batch_id, 0)];
        let bytes: [u8; 32] = Sha256::digest(b"opaque completion test").into();
        let batch = opaque_batch_from_quiesced_bytes(batch_id, &descriptors, &bytes)
            .expect("nonzero opaque batch");
        assert_eq!(format!("{batch:?}"), "OpaqueCudaSha256Batch(<opaque>)");
        batch.consume(|returned_batch, tokens| {
            assert_eq!(returned_batch, batch_id);
            assert_eq!(tokens.len(), 1);
            assert_eq!(tokens[0].descriptor(), descriptors[0]);
            assert_eq!(
                format!("{:?}", tokens[0]),
                "OpaqueCudaSha256Token(<opaque>)"
            );
            assert_eq!(tokens[0].digest.0, bytes);
        });
        let source = include_str!("sha256_completion.rs");
        let production = source
            .split("#[cfg(test)]\nmod tests")
            .next()
            .expect("production completion source");
        assert!(!production.contains("impl std::fmt::Display for OpaqueCudaSha256"));
        assert!(!production.contains("Serialize for OpaqueCudaSha256"));
        assert!(!production.contains("Hash for OpaqueCudaSha256"));
    }

    #[test]
    fn completion_rejects_empty_duplicate_and_out_of_order_slot_reservations() {
        assert!(matches!(
            PreparedCudaSha256Completion::prepare(&[]),
            Err(CudaSha256CompletionPrepareError::Runtime(
                CudaRuntimeProbeError::InvalidInputLength(0)
            ))
        ));
        let Some(runtime) = runtime() else {
            return;
        };
        let source = runtime
            .retain_device_memory_copy(0, b"slot ordering")
            .expect("resident source");
        let batch_id = CudaSha256BatchId::new(9).expect("batch id");
        let first = one_input(&source, batch_id);
        let duplicate = CudaSha256CompletionInput::new(
            first.input,
            CudaSha256CompletionDescriptor::new(batch_id, 0),
        );
        assert!(matches!(
            PreparedCudaSha256Completion::prepare(&[first, duplicate]),
            Err(CudaSha256CompletionPrepareError::Runtime(
                CudaRuntimeProbeError::InvalidInputLength(1)
            ))
        ));
        let out_of_order = CudaSha256CompletionInput::new(
            first.input,
            CudaSha256CompletionDescriptor::new(batch_id, 1),
        );
        assert!(matches!(
            PreparedCudaSha256Completion::prepare(&[out_of_order]),
            Err(CudaSha256CompletionPrepareError::Runtime(
                CudaRuntimeProbeError::InvalidInputLength(0)
            ))
        ));
    }

    #[test]
    fn completion_validates_context_and_owns_distinct_output_and_descriptor_buffers() {
        let Some(runtime) = runtime() else {
            return;
        };
        let source = runtime
            .retain_device_memory_copy(0, b"completion context")
            .expect("resident source");
        let wrong_primary = distinct_gpu_primary_context_for_test(0).expect("distinct primary");
        let wrong_context = source.clone_with_primary_for_test(wrong_primary);
        let batch_id = CudaSha256BatchId::new(10).expect("batch id");
        let good = one_input(&source, batch_id);
        let wrong = CudaSha256CompletionInput::new(
            CudaSha256DeviceBuffer::new(&wrong_context, 0, 1),
            CudaSha256CompletionDescriptor::new(batch_id, 1),
        );
        assert!(matches!(
            PreparedCudaSha256Completion::prepare(&[good, wrong]),
            Err(CudaSha256CompletionPrepareError::Runtime(
                CudaRuntimeProbeError::InvalidInputLength(1)
            ))
        ));
        let prepared = PreparedCudaSha256Completion::prepare(&[good]).expect("prepared completion");
        assert_ne!(
            prepared.resources.descriptor_device.ptr, prepared.resources.output_device.ptr,
            "completion output must be uniquely reserved rather than aliasing descriptor staging"
        );
    }

    #[test]
    fn completion_hashes_a_null_bearing_typed_byte_range_without_raw_public_output() {
        let Some(runtime) = runtime() else {
            return;
        };
        let payload = b"prefix\0typed\xffvalue\0suffix";
        let source = runtime
            .retain_device_memory_copy(0, payload)
            .expect("resident null-bearing source");
        let batch_id = CudaSha256BatchId::new(12).expect("batch id");
        let input = CudaSha256CompletionInput::new(
            CudaSha256DeviceBuffer::new(&source, 7, 13),
            CudaSha256CompletionDescriptor::new(batch_id, 0),
        );
        let expected: [u8; 32] = Sha256::digest(&payload[7..20]).into();
        let batch = match PreparedCudaSha256Completion::prepare(&[input])
            .expect("null-bearing preparation")
            .enqueue()
            .complete()
        {
            CudaSha256Completion::Quiesced(Ok(batch)) => batch,
            CudaSha256Completion::Quiesced(Err(error)) => panic!("CUDA completion failed: {error}"),
            CudaSha256Completion::UnknownQuiescence(_) => panic!("CUDA completion fence unknown"),
        };
        batch.consume(|_, tokens| assert_eq!(tokens[0].digest.0, expected));
    }

    #[test]
    fn runtime_generation_v1_genesis_chain_matches_all_canonical_roots_on_device() {
        let Some(runtime) = runtime() else {
            return;
        };
        let database_id = [0x5a_u8; 16];
        let source = runtime
            .retain_device_memory_copy(0, &database_id)
            .expect("resident database identity");
        let batch_id = CudaSha256BatchId::new(13).expect("batch id");
        let expected = runtime_generation_v1_genesis_oracle(database_id);
        let batch = match PreparedCudaSha256Completion::prepare_runtime_generation_v1_genesis(
            CudaSha256DeviceBuffer::new(&source, 0, database_id.len() as u64),
            batch_id,
        )
        .expect("genesis-chain preparation")
        .enqueue()
        .complete()
        {
            CudaSha256Completion::Quiesced(Ok(batch)) => batch,
            CudaSha256Completion::Quiesced(Err(error)) => panic!("genesis chain failed: {error}"),
            CudaSha256Completion::UnknownQuiescence(_) => {
                panic!("genesis-chain fence unknown")
            }
        };
        batch.consume(|returned_batch, tokens| {
            assert_eq!(returned_batch, batch_id);
            assert_eq!(tokens.len(), CUDA_RUNTIME_GENERATION_V1_GENESIS_ROOT_SLOTS);
            for (ordinal, token) in tokens.iter().enumerate() {
                assert_eq!(token.descriptor().slot_ordinal(), ordinal as u32);
                assert_eq!(token.digest.0, expected[ordinal], "root slot {ordinal}");
            }
            assert_ne!(
                tokens[0].digest.0, tokens[1].digest.0,
                "map parent depends on child"
            );
            assert_ne!(
                tokens[65].digest.0, tokens[66].digest.0,
                "status parent depends on child"
            );
            assert_eq!(
                tokens[130].digest.0,
                runtime_generation_v1_digest(
                    b"gpu-db/runtime-generation/database-root/v1",
                    &[&1_u16.to_le_bytes(), &database_id, &tokens[0].digest.0],
                ),
                "database root binds the table-map root only"
            );
            assert_ne!(
                tokens[130].digest.0,
                runtime_generation_v1_digest(
                    b"gpu-db/runtime-generation/database-root/v1",
                    &[&1_u16.to_le_bytes(), &database_id, &tokens[65].digest.0],
                ),
                "database root must not bind the independent status-view root"
            );
        });
    }

    #[test]
    fn completion_post_submission_cuda_errors_quarantine_and_synthetic_fences_can_release() {
        let Some(runtime) = runtime() else {
            return;
        };
        let _ = drain_parked_sha256_completions_for_test();
        let quarantined_before = quarantined_sha256_completion_count_for_test();

        let pre_submit_source = runtime
            .retain_device_memory_copy(0, &[0x10_u8; 16])
            .expect("pre-submit source");
        let pre_submit_weak = pre_submit_source.allocation_weak_for_test();
        fail_next_owned_stream_pre_submit_bind_raw_cuda_result_for_test(-9_989);
        assert!(matches!(
            genesis_submission(
                &pre_submit_source,
                CudaSha256BatchId::new(10_001).expect("pre-submit batch"),
            )
            .complete(),
            CudaSha256Completion::Quiesced(Err(CudaRuntimeProbeError::KernelLaunchFailed(-9_989)))
        ));
        drop(pre_submit_source);
        assert!(
            !pre_submit_weak.is_alive(),
            "a pre-submit bind failure releases never-enqueued pool leases"
        );
        assert_eq!(
            quarantined_sha256_completion_count_for_test(),
            quarantined_before,
            "the pre-submit terminal error is not a poisoned stream result"
        );

        let source = runtime
            .retain_device_memory_copy(0, &[0x11_u8; 16])
            .expect("resident database identity");
        let source_identity = source.allocation_identity();
        let source_weak = source.allocation_weak_for_test();
        let batch_id = CudaSha256BatchId::new(10_002).expect("batch id");

        fail_next_owned_stream_htod_enqueue_after_dispatch_for_test();
        assert!(matches!(
            genesis_submission(&source, batch_id).complete(),
            CudaSha256Completion::Quiesced(Err(CudaRuntimeProbeError::KernelLaunchFailed(-9_992)))
        ));

        fail_next_owned_stream_launch_after_dispatch_for_test();
        let launch_then_fence = genesis_submission(&source, batch_id);
        fail_owned_stream_syncs_for_test(1);
        let launch_then_fence = match launch_then_fence.complete() {
            CudaSha256Completion::UnknownQuiescence(unknown) => unknown,
            CudaSha256Completion::Quiesced(result) => {
                panic!("launch plus raw fence result must retain unknown quiescence: {result:?}")
            }
        };
        assert!(matches!(
            launch_then_fence.retry_complete(),
            CudaSha256Completion::Quiesced(Err(CudaRuntimeProbeError::KernelLaunchFailed(-9_993)))
        ));

        let dtoh_submission = genesis_submission(&source, batch_id);
        fail_next_owned_stream_dtoh_enqueue_for_test();
        assert!(matches!(
            dtoh_submission.complete(),
            CudaSha256Completion::Quiesced(Err(CudaRuntimeProbeError::KernelLaunchFailed(-9_994)))
        ));

        let bind_submission = genesis_submission(&source, batch_id);
        fail_next_owned_stream_post_submit_bind_raw_cuda_result_for_test(-9_995);
        assert!(matches!(
            bind_submission.complete(),
            CudaSha256Completion::Quiesced(Err(CudaRuntimeProbeError::KernelLaunchFailed(-9_995)))
        ));

        let enqueues_before_fence_retry = sha256_successful_enqueue_count_for_test();
        let sync_submission = genesis_submission(&source, batch_id);
        fail_owned_stream_syncs_for_test(1);
        let unknown = match sync_submission.complete() {
            CudaSha256Completion::UnknownQuiescence(unknown) => unknown,
            CudaSha256Completion::Quiesced(result) => {
                panic!("raw CUDA fence result must retain unknown quiescence: {result:?}")
            }
        };
        assert!(matches!(
            unknown.retry_complete(),
            CudaSha256Completion::Quiesced(Err(CudaRuntimeProbeError::KernelLaunchFailed(-9_991)))
        ));
        assert_eq!(
            sha256_successful_enqueue_count_for_test(),
            enqueues_before_fence_retry + 1,
            "retrying unknown quiescence only fences the original SHA submission"
        );

        fail_next_owned_stream_launch_after_dispatch_for_test();
        drop(genesis_submission(&source, batch_id));
        assert_eq!(
            quarantined_sha256_completion_count_for_test(),
            quarantined_before + 6,
            "every post-submit CUDA result, including a Drop-drained owner, quarantines"
        );
        drop(source);
        assert!(source_weak.is_alive());
        assert!(quarantined_sha256_completion_contains_source_owner_for_test(source_identity));

        let clean_source = runtime
            .retain_device_memory_copy(0, &[0x12_u8; 16])
            .expect("synthetic-fence source");
        let clean_weak = clean_source.allocation_weak_for_test();
        let clean_submission = genesis_submission(
            &clean_source,
            CudaSha256BatchId::new(10_003).expect("synthetic-fence batch"),
        );
        drop(clean_source);
        fail_next_owned_stream_synthetic_fence_not_attempted_for_test();
        let clean_unknown = match clean_submission.complete() {
            CudaSha256Completion::UnknownQuiescence(unknown) => unknown,
            CudaSha256Completion::Quiesced(result) => {
                panic!("synthetic interruption must leave quiescence unknown: {result:?}")
            }
        };
        match clean_unknown.retry_complete() {
            CudaSha256Completion::Quiesced(Ok(batch)) => batch.consume(|_, _| {}),
            CudaSha256Completion::Quiesced(Err(error)) => {
                panic!("synthetic fence retry must remain clean: {error}")
            }
            CudaSha256Completion::UnknownQuiescence(_) => {
                panic!("synthetic fence retry remained unknown")
            }
        }
        assert!(!clean_weak.is_alive());
        assert_eq!(
            quarantined_sha256_completion_count_for_test(),
            quarantined_before + 6,
            "the synthetic unattempted fence is the sole retry-to-success path"
        );

        let parking_before = parked_sha256_completion_count_for_test();
        let parked_source = runtime
            .retain_device_memory_copy(0, &[0x13_u8; 16])
            .expect("parking source");
        let parked_identity = parked_source.allocation_identity();
        let parked_weak = parked_source.allocation_weak_for_test();
        let parked_submission = genesis_submission(
            &parked_source,
            CudaSha256BatchId::new(10_004).expect("parking batch"),
        );
        drop(parked_source);
        fail_next_owned_stream_synthetic_fence_not_attempted_for_test();
        let parked = match parked_submission.complete() {
            CudaSha256Completion::UnknownQuiescence(unknown) => unknown,
            CudaSha256Completion::Quiesced(result) => {
                panic!("synthetic interruption must leave parking owner unknown: {result:?}")
            }
        };
        fail_next_owned_stream_synthetic_fence_not_attempted_for_test();
        drop(parked);
        assert_eq!(
            parked_sha256_completion_count_for_test(),
            parking_before + 1
        );
        assert!(parked_sha256_completion_contains_source_owner_for_test(
            parked_identity
        ));
        assert!(parked_weak.is_alive());
        assert_eq!(
            drain_parked_sha256_completions_for_test().expect("parked drain"),
            1
        );
        assert!(!parked_weak.is_alive());
    }

    #[test]
    fn sha256_synthetic_fence_retry_is_safe_for_three_serial_and_two_concurrent_submissions() {
        let Some(runtime) = runtime() else {
            return;
        };
        let source = std::sync::Arc::new(
            runtime
                .retain_device_memory_copy(0, &[0x14_u8; 16])
                .expect("serial and concurrent SHA source"),
        );
        for raw_batch in 10_100..10_103 {
            let submission = genesis_submission(
                &source,
                CudaSha256BatchId::new(raw_batch).expect("serial batch"),
            );
            fail_next_owned_stream_synthetic_fence_not_attempted_for_test();
            let unknown = match submission.complete() {
                CudaSha256Completion::UnknownQuiescence(unknown) => unknown,
                CudaSha256Completion::Quiesced(result) => {
                    panic!("serial synthetic fence must retain unknown quiescence: {result:?}")
                }
            };
            assert!(matches!(
                unknown.retry_complete(),
                CudaSha256Completion::Quiesced(Ok(_))
            ));
        }
        let joins = (0..2)
            .map(|offset| {
                let source = std::sync::Arc::clone(&source);
                std::thread::spawn(move || {
                    let submission = genesis_submission(
                        &source,
                        CudaSha256BatchId::new(10_200 + offset).expect("concurrent batch"),
                    );
                    fail_next_owned_stream_synthetic_fence_not_attempted_for_test();
                    match submission.complete() {
                        CudaSha256Completion::UnknownQuiescence(unknown) => matches!(
                            unknown.retry_complete(),
                            CudaSha256Completion::Quiesced(Ok(_))
                        ),
                        CudaSha256Completion::Quiesced(_) => false,
                    }
                })
            })
            .collect::<Vec<_>>();
        for join in joins {
            assert!(join.join().expect("concurrent SHA completion thread"));
        }
    }
}
