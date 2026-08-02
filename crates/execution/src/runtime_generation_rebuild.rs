//! GPU-only V1 runtime-generation rebuild for one nullable INT4 table.
//!
//! This transport deliberately owns neither catalog installation nor comparison against an
//! expected root. It accepts exact, closed physical owners, stages cold bytes once, computes the
//! canonical logical roots on the device, and returns only opaque proof digests after one fence.

mod abi;

use std::{
    ffi::c_void,
    fmt,
    num::NonZeroU64,
    sync::{Arc, Mutex},
};

use crate::{
    cuda_context::{PinnedHostBufferOwned, PooledDeviceBufferOwned, PooledStreamOwned},
    sha256::CuLaunchKernel,
    CudaResidentDeviceMemory, CudaResidentReadSource, CudaRuntimeProbeError, GpuPrimaryContext,
    OpaqueCudaSha256Digest,
};

const RUNTIME_GENERATION_REBUILD_PTX: &[u8] =
    include_bytes!("runtime_generation_rebuild/kernel.ptx");
const ROOT_FORMAT_V1: u16 = 1;
const INT4_OID: u32 = 23;
const INT4_SIGNED_SIZE: i16 = 4;
const PARKED_REBUILD_COMPLETIONS: usize = 8;
/// Exact durable V1 encoding of the table root followed by the database root.
pub const RUNTIME_GENERATION_REBUILD_V1_DURABLE_COMMITMENT_BYTES: usize = 64;

/// A target is a retained primary CUDA context, never a raw context or device pointer.
#[derive(Clone)]
pub struct RuntimeGenerationRebuildTarget {
    primary: Arc<GpuPrimaryContext>,
    device_ordinal: u16,
    context_identity: usize,
}

impl RuntimeGenerationRebuildTarget {
    pub fn device_ordinal(&self) -> u16 {
        self.device_ordinal
    }

    /// Pointer-free opaque identity used only to reject cross-primary resident owners.
    pub fn context_identity(&self) -> usize {
        self.context_identity
    }

    pub(crate) fn from_primary(device_ordinal: u16, primary: Arc<GpuPrimaryContext>) -> Self {
        Self {
            context_identity: primary.context() as usize,
            primary,
            device_ordinal,
        }
    }
}

impl fmt::Debug for RuntimeGenerationRebuildTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeGenerationRebuildTarget")
            .field("device_ordinal", &self.device_ordinal)
            .finish_non_exhaustive()
    }
}

/// A caller-chosen nonzero identity for one rebuild attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeGenerationRebuildAttempt(NonZeroU64);

impl RuntimeGenerationRebuildAttempt {
    pub fn new(value: u64) -> Result<Self, RuntimeGenerationRebuildPrepareError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(RuntimeGenerationRebuildPrepareError::InvalidInput(
                "zero rebuild attempt",
            ))
    }

    pub fn get(self) -> u64 {
        self.0.get()
    }
}

/// One exact source owner. It has no raw-pointer accessor and can only be consumed by this
/// operator's closed descriptor encoder.
pub struct RuntimeGenerationRebuildSource {
    backing: RuntimeGenerationRebuildSourceBacking,
}

enum RuntimeGenerationRebuildSourceBacking {
    Resident {
        memory: Arc<CudaResidentDeviceMemory>,
        byte_offset: u64,
        byte_len: u64,
    },
    ColdRam {
        bytes: Arc<[u8]>,
        byte_offset: u64,
        byte_len: u64,
    },
}

impl RuntimeGenerationRebuildSource {
    /// Retain one exact resident source span. Bounds and target identity are verified in
    /// [`PreparedRuntimeGenerationRebuild::prepare`] before any CUDA enqueue.
    pub fn resident(
        memory: Arc<CudaResidentDeviceMemory>,
        byte_offset: u64,
        byte_len: u64,
    ) -> Self {
        Self {
            backing: RuntimeGenerationRebuildSourceBacking::Resident {
                memory,
                byte_offset,
                byte_len,
            },
        }
    }

    /// Retain one exact immutable cold-RAM source span. It is copied once into the single pinned
    /// descriptor/cold arena during preparation and then rehydrated on the device.
    pub fn cold_ram(bytes: Arc<[u8]>, byte_offset: u64, byte_len: u64) -> Self {
        Self {
            backing: RuntimeGenerationRebuildSourceBacking::ColdRam {
                bytes,
                byte_offset,
                byte_len,
            },
        }
    }

    fn byte_len(&self) -> u64 {
        match &self.backing {
            RuntimeGenerationRebuildSourceBacking::Resident { byte_len, .. }
            | RuntimeGenerationRebuildSourceBacking::ColdRam { byte_len, .. } => *byte_len,
        }
    }

    /// The one-source shard convenience may fan one moved source into its five fixed physical
    /// roles, but callers cannot clone it into another submission.
    fn clone_for_shard_role(&self) -> Self {
        let backing = match &self.backing {
            RuntimeGenerationRebuildSourceBacking::Resident {
                memory,
                byte_offset,
                byte_len,
            } => RuntimeGenerationRebuildSourceBacking::Resident {
                memory: Arc::clone(memory),
                byte_offset: *byte_offset,
                byte_len: *byte_len,
            },
            RuntimeGenerationRebuildSourceBacking::ColdRam {
                bytes,
                byte_offset,
                byte_len,
            } => RuntimeGenerationRebuildSourceBacking::ColdRam {
                bytes: Arc::clone(bytes),
                byte_offset: *byte_offset,
                byte_len: *byte_len,
            },
        };
        Self { backing }
    }

    /// Equality is intentionally restricted to exact physical transport ownership, not byte
    /// equality.  It lets one convenience source retain one HtoD cold-arena copy even though
    /// each role owns its own move-only attachment.
    fn is_same_transport_source(&self, other: &Self) -> bool {
        match (&self.backing, &other.backing) {
            (
                RuntimeGenerationRebuildSourceBacking::Resident {
                    memory: left_memory,
                    byte_offset: left_offset,
                    byte_len: left_len,
                },
                RuntimeGenerationRebuildSourceBacking::Resident {
                    memory: right_memory,
                    byte_offset: right_offset,
                    byte_len: right_len,
                },
            ) => {
                Arc::ptr_eq(left_memory, right_memory)
                    && left_offset == right_offset
                    && left_len == right_len
            }
            (
                RuntimeGenerationRebuildSourceBacking::ColdRam {
                    bytes: left_bytes,
                    byte_offset: left_offset,
                    byte_len: left_len,
                },
                RuntimeGenerationRebuildSourceBacking::ColdRam {
                    bytes: right_bytes,
                    byte_offset: right_offset,
                    byte_len: right_len,
                },
            ) => {
                Arc::ptr_eq(left_bytes, right_bytes)
                    && left_offset == right_offset
                    && left_len == right_len
            }
            _ => false,
        }
    }
}

impl fmt::Debug for RuntimeGenerationRebuildSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RuntimeGenerationRebuildSource(<opaque>)")
    }
}

/// A byte span relative to an exact source owner. It is a closed physical descriptor coordinate,
/// not a pointer and not a logical-root field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeGenerationRebuildRoleSpan {
    pub byte_offset: u64,
    pub byte_len: u64,
}

/// Five device physical vectors necessary for the supported table grammar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeGenerationRebuildShardRoles {
    pub stable_row_ids: RuntimeGenerationRebuildRoleSpan,
    pub validity: RuntimeGenerationRebuildRoleSpan,
    pub values: RuntimeGenerationRebuildRoleSpan,
    pub created_by: RuntimeGenerationRebuildRoleSpan,
    pub deleted_by: RuntimeGenerationRebuildRoleSpan,
}

/// One physical rebuild role paired with the exact source allocation that owns it.  All five
/// roles remain mandatory: V1 has no device descriptor sentinel for an omitted MVCC vector, so
/// treating a missing source as a host-side default would weaken the device proof.
pub struct RuntimeGenerationRebuildShardRoleSource {
    source: RuntimeGenerationRebuildSource,
    span: RuntimeGenerationRebuildRoleSpan,
}

impl RuntimeGenerationRebuildShardRoleSource {
    pub fn new(
        source: RuntimeGenerationRebuildSource,
        span: RuntimeGenerationRebuildRoleSpan,
    ) -> Self {
        Self { source, span }
    }
}

impl fmt::Debug for RuntimeGenerationRebuildShardRoleSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RuntimeGenerationRebuildShardRoleSource(<opaque>)")
    }
}

/// Exact independently retained physical owners for one canonical shard.  This is the direct
/// five-source form; roles can be resident or cold independently without exposing pointers.
pub struct RuntimeGenerationRebuildShardRoleSources {
    pub stable_row_ids: RuntimeGenerationRebuildShardRoleSource,
    pub validity: RuntimeGenerationRebuildShardRoleSource,
    pub values: RuntimeGenerationRebuildShardRoleSource,
    pub created_by: RuntimeGenerationRebuildShardRoleSource,
    pub deleted_by: RuntimeGenerationRebuildShardRoleSource,
}

/// One canonical table shard. A source is consumed exactly once, so a caller cannot reuse a
/// physical owner in two independently submitted rebuilds without an explicit reattachment.
pub struct RuntimeGenerationRebuildShard {
    row_start: u64,
    row_count: u64,
    roles: RuntimeGenerationRebuildShardRoleSources,
}

impl RuntimeGenerationRebuildShard {
    pub fn new(
        source: RuntimeGenerationRebuildSource,
        row_start: u64,
        row_count: u64,
        roles: RuntimeGenerationRebuildShardRoles,
    ) -> Self {
        Self::from_role_sources(
            row_start,
            row_count,
            RuntimeGenerationRebuildShardRoleSources {
                stable_row_ids: RuntimeGenerationRebuildShardRoleSource::new(
                    source.clone_for_shard_role(),
                    roles.stable_row_ids,
                ),
                validity: RuntimeGenerationRebuildShardRoleSource::new(
                    source.clone_for_shard_role(),
                    roles.validity,
                ),
                values: RuntimeGenerationRebuildShardRoleSource::new(
                    source.clone_for_shard_role(),
                    roles.values,
                ),
                created_by: RuntimeGenerationRebuildShardRoleSource::new(
                    source.clone_for_shard_role(),
                    roles.created_by,
                ),
                deleted_by: RuntimeGenerationRebuildShardRoleSource::new(source, roles.deleted_by),
            },
        )
    }

    /// Consume five independently retained role sources for one canonical V1 shard.  This is
    /// deliberately separate from [`Self::new`], which preserves the one-source convenience
    /// without making the source itself cloneable across submissions.
    pub fn from_role_sources(
        row_start: u64,
        row_count: u64,
        roles: RuntimeGenerationRebuildShardRoleSources,
    ) -> Self {
        Self {
            row_start,
            row_count,
            roles,
        }
    }

    fn role_sources(&self) -> [&RuntimeGenerationRebuildShardRoleSource; 5] {
        [
            &self.roles.stable_row_ids,
            &self.roles.validity,
            &self.roles.values,
            &self.roles.created_by,
            &self.roles.deleted_by,
        ]
    }
}

/// Closed metadata for the only accepted semantic shape: V1, one table, one nullable INT4
/// column, no indexes or predicates. Roots, raw pointers, cache callbacks, and domain strings do
/// not enter this input type.
pub struct RuntimeGenerationRebuildInput {
    target: RuntimeGenerationRebuildTarget,
    attempt: RuntimeGenerationRebuildAttempt,
    database_id: [u8; 16],
    table_id: u64,
    data_generation: u64,
    visibility_cut: u64,
    logical_row_count: u64,
    column_id: u64,
    attnum: i16,
    declared_type_oid: u32,
    signed_type_size: i16,
    shards: Box<[RuntimeGenerationRebuildShard]>,
}

impl RuntimeGenerationRebuildInput {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target: RuntimeGenerationRebuildTarget,
        attempt: RuntimeGenerationRebuildAttempt,
        database_id: [u8; 16],
        table_id: u64,
        data_generation: u64,
        visibility_cut: u64,
        logical_row_count: u64,
        column_id: u64,
        attnum: i16,
        declared_type_oid: u32,
        signed_type_size: i16,
        shards: Box<[RuntimeGenerationRebuildShard]>,
    ) -> Self {
        Self {
            target,
            attempt,
            database_id,
            table_id,
            data_generation,
            visibility_cut,
            logical_row_count,
            column_id,
            attnum,
            declared_type_oid,
            signed_type_size,
            shards,
        }
    }
}

/// Preparation errors occur before the first HtoD enqueue. They deliberately distinguish the
/// unsupported V1 grammar from CUDA transport failures without manufacturing a CPU fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeGenerationRebuildPrepareError {
    Runtime(CudaRuntimeProbeError),
    AsyncTransportUnavailable,
    PinnedHostStagingUnavailable,
    InvalidInput(&'static str),
}

impl From<CudaRuntimeProbeError> for RuntimeGenerationRebuildPrepareError {
    fn from(value: CudaRuntimeProbeError) -> Self {
        Self::Runtime(value)
    }
}

/// A pre-enqueue failure retains the exact move-only input for a caller that can repair an
/// admission error. No stream or device copy can still observe it.
pub struct RuntimeGenerationRebuildPrepareFailure {
    error: RuntimeGenerationRebuildPrepareError,
    input: RuntimeGenerationRebuildInput,
}

impl RuntimeGenerationRebuildPrepareFailure {
    pub fn error(&self) -> &RuntimeGenerationRebuildPrepareError {
        &self.error
    }

    pub fn into_input(self) -> RuntimeGenerationRebuildInput {
        self.input
    }
}

impl fmt::Debug for RuntimeGenerationRebuildPrepareFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeGenerationRebuildPrepareFailure")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

/// The only durable V1 proof commitments: the final table root and final database root, in that
/// exact order.  The intermediate proof slots never leave this transport.  Its byte encoding is
/// intentionally a fixed serializer/deserializer rather than a generic digest accessor, so WAL
/// code can persist the two sealed commitments without gaining a host-hash input surface.
pub struct RuntimeGenerationRebuildV1DurableCommitments {
    table_root: [u8; 32],
    database_root: [u8; 32],
}

impl RuntimeGenerationRebuildV1DurableCommitments {
    /// Serialize the exact V1 table/database-root pair into caller-owned durable storage.
    pub fn encode_durable_into(
        &self,
        destination: &mut [u8],
    ) -> Result<(), RuntimeGenerationRebuildV1CommitmentBytesError> {
        if destination.len() != RUNTIME_GENERATION_REBUILD_V1_DURABLE_COMMITMENT_BYTES {
            return Err(
                RuntimeGenerationRebuildV1CommitmentBytesError::InvalidLength(destination.len()),
            );
        }
        destination[..32].copy_from_slice(&self.table_root);
        destination[32..].copy_from_slice(&self.database_root);
        Ok(())
    }

    /// Reconstruct the exact sealed V1 table/database-root pair from durable storage.  This
    /// validates only the closed transport geometry; a later GPU proof comparison authenticates
    /// the bytes without recomputing or hashing them on the host.
    pub fn decode_durable(
        source: &[u8],
    ) -> Result<Self, RuntimeGenerationRebuildV1CommitmentBytesError> {
        if source.len() != RUNTIME_GENERATION_REBUILD_V1_DURABLE_COMMITMENT_BYTES {
            return Err(
                RuntimeGenerationRebuildV1CommitmentBytesError::InvalidLength(source.len()),
            );
        }
        let mut table_root = [0_u8; 32];
        let mut database_root = [0_u8; 32];
        table_root.copy_from_slice(&source[..32]);
        database_root.copy_from_slice(&source[32..]);
        Ok(Self {
            table_root,
            database_root,
        })
    }
}

impl fmt::Debug for RuntimeGenerationRebuildV1DurableCommitments {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RuntimeGenerationRebuildV1DurableCommitments(<opaque>)")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeGenerationRebuildV1CommitmentBytesError {
    InvalidLength(usize),
}

/// The two persisted semantic commitments did not match the later quiesced V1 proof.  The
/// mismatch is intentionally not attributed to one root, avoiding an extra commitment oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeGenerationRebuildV1CommitmentMismatch;

/// Whole opaque proof arena for one drained attempt. Slot meanings stay in the engine's sealed
/// proof layout; this transport reveals only attempt/version/count structure.
pub struct OpaqueRuntimeGenerationRebuildProof {
    attempt: RuntimeGenerationRebuildAttempt,
    root_format: u16,
    slots: Box<[[u8; 32]]>,
}

/// The closed one-entry table-map completion emitted by the V1 rebuild kernel.  It is available
/// only after the durable table/database commitments have matched.  All roots remain opaque:
/// the engine may consume this fixed leaf/path/empty-root shape to import one persistent map,
/// but cannot manufacture, reorder, or serialize a generic map completion.
pub struct RuntimeGenerationRebuildV1TableMapCompletion {
    empty_roots: Box<[OpaqueCudaSha256Digest]>,
    leaf_root: OpaqueCudaSha256Digest,
    path_roots: Box<[OpaqueCudaSha256Digest]>,
}

impl RuntimeGenerationRebuildV1TableMapCompletion {
    /// Consume the exact closed completion. `empty_roots` are ordered depth 0 through 64 and
    /// `path_roots` are ordered depth 0 through 63.  The callback has no access to raw digest
    /// bytes, and there is no constructor outside the quiesced Rebuild proof.
    pub fn consume<R>(
        self,
        consume: impl FnOnce(
            &[OpaqueCudaSha256Digest],
            OpaqueCudaSha256Digest,
            &[OpaqueCudaSha256Digest],
        ) -> R,
    ) -> R {
        consume(&self.empty_roots, self.leaf_root, &self.path_roots)
    }
}

impl fmt::Debug for RuntimeGenerationRebuildV1TableMapCompletion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RuntimeGenerationRebuildV1TableMapCompletion(<opaque>)")
    }
}

impl OpaqueRuntimeGenerationRebuildProof {
    pub fn attempt(&self) -> RuntimeGenerationRebuildAttempt {
        self.attempt
    }

    pub fn root_format(&self) -> u16 {
        self.root_format
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Consume a quiesced proof into the exact two V1 semantic commitments that may cross the
    /// durable engine boundary.  This only copies device-produced bytes; it never host-hashes.
    pub fn into_durable_v1_commitments(self) -> RuntimeGenerationRebuildV1DurableCommitments {
        debug_assert_eq!(self.root_format, ROOT_FORMAT_V1);
        debug_assert_eq!(self.slots.len(), abi::PROOF_DIGEST_SLOTS);
        RuntimeGenerationRebuildV1DurableCommitments {
            table_root: self.slots[abi::SLOT_TABLE_ROOT],
            database_root: self.slots[abi::SLOT_DATABASE_ROOT],
        }
    }

    /// Compare the only durable V1 commitments and, on success only, hand the sealed caller the
    /// GPU-emitted one-table persistent-map path.  This does not widen the durable format: the
    /// table-map root is already bound by the compared database root and is recomputed on every
    /// recovery from the same device-owned table image.
    pub fn compare_durable_v1_commitments_with_table_map<R>(
        self,
        expected: &RuntimeGenerationRebuildV1DurableCommitments,
        consume: impl FnOnce(
            RuntimeGenerationRebuildAttempt,
            OpaqueCudaSha256Digest,
            RuntimeGenerationRebuildV1TableMapCompletion,
            OpaqueCudaSha256Digest,
        ) -> R,
    ) -> Result<R, RuntimeGenerationRebuildV1CommitmentMismatch> {
        debug_assert_eq!(self.root_format, ROOT_FORMAT_V1);
        debug_assert_eq!(self.slots.len(), abi::PROOF_DIGEST_SLOTS);
        let table_matches =
            constant_time_equal(&self.slots[abi::SLOT_TABLE_ROOT], &expected.table_root);
        let database_matches = constant_time_equal(
            &self.slots[abi::SLOT_DATABASE_ROOT],
            &expected.database_root,
        );
        if !(table_matches && database_matches) {
            return Err(RuntimeGenerationRebuildV1CommitmentMismatch);
        }
        let table_map = RuntimeGenerationRebuildV1TableMapCompletion {
            empty_roots: self.slots[abi::SLOT_TABLE_MAP_EMPTY..=abi::SLOT_TABLE_MAP_EMPTY + 64]
                .iter()
                .copied()
                .map(OpaqueCudaSha256Digest::from_runtime_generation_rebuild_slot)
                .collect(),
            leaf_root: OpaqueCudaSha256Digest::from_runtime_generation_rebuild_slot(
                self.slots[abi::SLOT_TABLE_MAP_LEAF],
            ),
            path_roots: self.slots[abi::SLOT_TABLE_MAP_PATH..=abi::SLOT_TABLE_MAP_PATH + 63]
                .iter()
                .copied()
                .map(OpaqueCudaSha256Digest::from_runtime_generation_rebuild_slot)
                .collect(),
        };
        Ok(consume(
            self.attempt,
            OpaqueCudaSha256Digest::from_runtime_generation_rebuild_slot(
                self.slots[abi::SLOT_TABLE_ROOT],
            ),
            table_map,
            OpaqueCudaSha256Digest::from_runtime_generation_rebuild_slot(
                self.slots[abi::SLOT_DATABASE_ROOT],
            ),
        ))
    }
}

impl fmt::Debug for OpaqueRuntimeGenerationRebuildProof {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpaqueRuntimeGenerationRebuildProof")
            .field("attempt", &self.attempt)
            .field("root_format", &self.root_format)
            .field("slot_count", &self.slots.len())
            .finish()
    }
}

/// Fully reserved pre-enqueue resources. Dropping it releases only never-submitted resources.
#[must_use = "a prepared runtime-generation rebuild must be enqueued or intentionally dropped"]
pub struct PreparedRuntimeGenerationRebuild {
    primary: Arc<GpuPrimaryContext>,
    resources: RuntimeGenerationRebuildResources,
    attempt: RuntimeGenerationRebuildAttempt,
    function: *mut c_void,
    launch: CuLaunchKernel,
}

/// An in-flight rebuild retaining all source guards, staging, workspace, output, and stream.
#[must_use = "an in-flight runtime-generation rebuild must be completed or safely dropped"]
pub struct RuntimeGenerationRebuildSubmission {
    primary: Arc<GpuPrimaryContext>,
    resources: Option<RuntimeGenerationRebuildResources>,
    attempt: RuntimeGenerationRebuildAttempt,
    function: *mut c_void,
    launch: CuLaunchKernel,
    phase: RuntimeGenerationRebuildPhase,
}

/// Known quiescence is separated from an unproved fence. The latter has only a fence retry, never
/// an opportunity to relaunch the HtoD/kernel/DtoH sequence.
#[must_use = "unknown runtime-generation rebuild quiescence must be retained or retried"]
#[allow(
    clippy::large_enum_variant,
    reason = "boxing after a CUDA enqueue would allocate on the exceptional safety path"
)]
pub enum RuntimeGenerationRebuildCompletion {
    Quiesced(Result<OpaqueRuntimeGenerationRebuildProof, RuntimeGenerationRebuildError>),
    UnknownQuiescence(RuntimeGenerationRebuildUnknownQuiescence),
}

/// Fail-closed owner of every resource that could still be observed by a private stream.
#[must_use = "unknown quiescence retains in-flight rebuild resources until retried or parked"]
pub struct RuntimeGenerationRebuildUnknownQuiescence {
    error: RuntimeGenerationRebuildError,
    submission: Option<RuntimeGenerationRebuildSubmission>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeGenerationRebuildError {
    Runtime(CudaRuntimeProbeError),
    DeviceRejected(u32),
}

impl From<CudaRuntimeProbeError> for RuntimeGenerationRebuildError {
    fn from(value: CudaRuntimeProbeError) -> Self {
        Self::Runtime(value)
    }
}

struct RuntimeGenerationRebuildResources {
    // These strong guards prove that every direct resident pointer in the staged descriptor
    // remains allocated until the covering fence. Cold owners are retained too, even though the
    // stream addresses only the copied pinned arena after enqueue.
    _resident_owners: Box<[Arc<crate::resident_memory::CudaResidentDeviceAllocation>]>,
    _cold_owners: Box<[Arc<[u8]>]>,
    input_host: PinnedHostBufferOwned,
    input_device: PooledDeviceBufferOwned,
    _workspace: PooledDeviceBufferOwned,
    output_host: PinnedHostBufferOwned,
    output_device: PooledDeviceBufferOwned,
    stream: PooledStreamOwned,
    input_bytes: usize,
    proof_slots: Box<[[u8; 32]]>,
}

enum RuntimeGenerationRebuildPhase {
    InFlight,
    ReadbackQueued,
    TerminalFailure {
        error: CudaRuntimeProbeError,
        drain_required: bool,
    },
    Quiesced,
}

impl RuntimeGenerationRebuildPhase {
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

unsafe impl Send for PreparedRuntimeGenerationRebuild {}
unsafe impl Send for RuntimeGenerationRebuildSubmission {}
unsafe impl Send for RuntimeGenerationRebuildUnknownQuiescence {}
unsafe impl Send for RuntimeGenerationRebuildResources {}

fn constant_time_equal(left: &[u8; 32], right: &[u8; 32]) -> bool {
    left.iter()
        .zip(right)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

impl PreparedRuntimeGenerationRebuild {
    /// Validate, reserve, deduplicate source guards, and materialize the one pinned HtoD
    /// descriptor+cold arena before the sole enqueue edge. An error returns its original input.
    pub fn prepare(
        input: RuntimeGenerationRebuildInput,
    ) -> Result<Self, Box<RuntimeGenerationRebuildPrepareFailure>> {
        match prepare_inner(&input) {
            Ok(prepared) => Ok(prepared),
            Err(error) => Err(Box::new(RuntimeGenerationRebuildPrepareFailure {
                error,
                input,
            })),
        }
    }

    /// The one submit edge. Every failure after HtoD remains represented by the returned owner.
    pub fn enqueue(self) -> RuntimeGenerationRebuildSubmission {
        let mut submission = RuntimeGenerationRebuildSubmission {
            primary: self.primary,
            resources: Some(self.resources),
            attempt: self.attempt,
            function: self.function,
            launch: self.launch,
            phase: RuntimeGenerationRebuildPhase::InFlight,
        };
        submission.enqueue_inner();
        submission
    }
}

impl RuntimeGenerationRebuildSubmission {
    fn resources(&self) -> &RuntimeGenerationRebuildResources {
        self.resources
            .as_ref()
            .expect("rebuild resources remain owned until quiescence")
    }

    fn resources_mut(&mut self) -> &mut RuntimeGenerationRebuildResources {
        self.resources
            .as_mut()
            .expect("rebuild resources remain owned until quiescence")
    }

    fn stream(&self) -> *mut c_void {
        self.resources()
            .stream
            .pooled
            .as_ref()
            .expect("rebuild stream remains owned until quiescence")
            .stream
    }

    fn enqueue_inner(&mut self) {
        if let Err(error) = self.primary.set_current() {
            self.phase = RuntimeGenerationRebuildPhase::TerminalFailure {
                error,
                drain_required: false,
            };
            return;
        }
        let stream = self.stream();
        let resources = self.resources();
        if let Err(error) = self.primary.enqueue_owned_stream_htod(
            resources.input_device.ptr,
            resources.input_host.ptr.cast_const(),
            resources.input_bytes,
            stream,
        ) {
            self.phase = RuntimeGenerationRebuildPhase::TerminalFailure {
                error,
                drain_required: true,
            };
            return;
        }
        let launch_status = unsafe {
            launch_runtime_generation_rebuild_kernel(
                self.launch,
                self.function,
                resources.input_device.ptr,
                resources.output_device.ptr,
                stream,
            )
        };
        match self.primary.check_owned_stream_launch_result(launch_status) {
            Ok(()) => self.primary.after_owned_stream_enqueue(),
            Err(error) => {
                self.phase = RuntimeGenerationRebuildPhase::TerminalFailure {
                    error,
                    drain_required: true,
                };
            }
        }
    }

    fn queue_readback(&mut self) -> Result<(), CudaRuntimeProbeError> {
        if !matches!(self.phase, RuntimeGenerationRebuildPhase::InFlight) {
            return Ok(());
        }
        self.primary.set_current()?;
        let stream = self.stream();
        let resources = self.resources();
        self.primary.enqueue_owned_stream_dtoh(
            resources.output_host.ptr,
            resources.output_device.ptr,
            abi::OUTPUT_BYTES,
            stream,
        )?;
        self.phase = RuntimeGenerationRebuildPhase::ReadbackQueued;
        Ok(())
    }

    fn drain_stream(&self) -> Result<(), CudaRuntimeProbeError> {
        self.primary.synchronize_owned_stream(self.stream())
    }

    fn terminal_error(&self) -> Option<CudaRuntimeProbeError> {
        match &self.phase {
            RuntimeGenerationRebuildPhase::TerminalFailure { error, .. } => Some(error.clone()),
            _ => None,
        }
    }

    fn unknown(self, error: RuntimeGenerationRebuildError) -> RuntimeGenerationRebuildCompletion {
        RuntimeGenerationRebuildCompletion::UnknownQuiescence(
            RuntimeGenerationRebuildUnknownQuiescence {
                error,
                submission: Some(self),
            },
        )
    }

    /// Fence only the previously enqueued stream. It cannot enqueue, allocate, or launch again.
    pub fn complete(mut self) -> RuntimeGenerationRebuildCompletion {
        if self.phase.drain_required()
            && matches!(self.phase, RuntimeGenerationRebuildPhase::InFlight)
        {
            if let Err(error) = self.queue_readback() {
                self.phase = RuntimeGenerationRebuildPhase::TerminalFailure {
                    error,
                    drain_required: true,
                };
            }
        }
        if self.phase.drain_required() {
            if let Err(error) = self.drain_stream() {
                return self.unknown(error.into());
            }
        }
        let result = if let Some(error) = self.terminal_error() {
            Err(error.into())
        } else if matches!(self.phase, RuntimeGenerationRebuildPhase::ReadbackQueued) {
            self.proof_after_quiescence()
        } else {
            Err(CudaRuntimeProbeError::KernelLaunchFailed(-1).into())
        };
        self.phase = RuntimeGenerationRebuildPhase::Quiesced;
        RuntimeGenerationRebuildCompletion::Quiesced(result)
    }

    fn proof_after_quiescence(
        &mut self,
    ) -> Result<OpaqueRuntimeGenerationRebuildProof, RuntimeGenerationRebuildError> {
        let resources = self.resources_mut();
        let bytes = unsafe {
            std::slice::from_raw_parts(resources.output_host.ptr.cast::<u8>(), abi::OUTPUT_BYTES)
        };
        let status = abi::get_u32(bytes, 0);
        if status != 0 {
            return Err(RuntimeGenerationRebuildError::DeviceRejected(status));
        }
        for (index, slot) in resources.proof_slots.iter_mut().enumerate() {
            let begin = abi::OUTPUT_STATUS_BYTES + index * 32;
            slot.copy_from_slice(&bytes[begin..begin + 32]);
        }
        let slots = std::mem::take(&mut resources.proof_slots);
        Ok(OpaqueRuntimeGenerationRebuildProof {
            attempt: self.attempt,
            root_format: ROOT_FORMAT_V1,
            slots,
        })
    }
}

impl Drop for RuntimeGenerationRebuildSubmission {
    fn drop(&mut self) {
        if !self.phase.drain_required() {
            return;
        }
        if self.drain_stream().is_ok() {
            self.phase = RuntimeGenerationRebuildPhase::Quiesced;
            return;
        }
        if let Some(resources) = self.resources.take() {
            park_runtime_generation_rebuild_resources(resources);
        }
    }
}

impl RuntimeGenerationRebuildUnknownQuiescence {
    pub fn error(&self) -> &RuntimeGenerationRebuildError {
        &self.error
    }

    pub fn retry_complete(mut self) -> RuntimeGenerationRebuildCompletion {
        self.submission
            .take()
            .expect("unknown rebuild quiescence retains one submission")
            .complete()
    }
}

static PARKED_REBUILDS: Mutex<
    [Option<RuntimeGenerationRebuildResources>; PARKED_REBUILD_COMPLETIONS],
> = Mutex::new([const { None }; PARKED_REBUILD_COMPLETIONS]);

fn park_runtime_generation_rebuild_resources(resources: RuntimeGenerationRebuildResources) {
    let Ok(mut parked) = PARKED_REBUILDS.lock() else {
        std::mem::forget(resources);
        return;
    };
    if let Some(slot) = parked.iter_mut().find(|slot| slot.is_none()) {
        *slot = Some(resources);
    } else {
        std::mem::forget(resources);
    }
}

fn prepare_inner(
    input: &RuntimeGenerationRebuildInput,
) -> Result<PreparedRuntimeGenerationRebuild, RuntimeGenerationRebuildPrepareError> {
    validate_input(input)?;
    let primary = Arc::clone(&input.target.primary);
    if primary.cu_memcpy_htod_async.is_none() || primary.cu_memcpy_dtoh_async.is_none() {
        return Err(RuntimeGenerationRebuildPrepareError::AsyncTransportUnavailable);
    }
    primary.set_current()?;
    let descriptor_bytes = abi::descriptor_bytes(input.shards.len()).ok_or(
        RuntimeGenerationRebuildPrepareError::InvalidInput("descriptor capacity"),
    )?;
    let cold_bytes = checked_cold_bytes(input)?;
    let input_bytes = descriptor_bytes.checked_add(cold_bytes).ok_or(
        RuntimeGenerationRebuildPrepareError::InvalidInput("cold arena capacity"),
    )?;
    let workspace_bytes = abi::workspace_bytes(input.logical_row_count).ok_or(
        RuntimeGenerationRebuildPrepareError::InvalidInput("workspace capacity"),
    )?;
    let input_device = primary.lease_device_buffer_owned(input_bytes)?;
    let workspace = primary.lease_device_buffer_owned(workspace_bytes)?;
    let output_device = primary.lease_device_buffer_owned(abi::OUTPUT_BYTES)?;
    if input_device.ptr == workspace.ptr
        || input_device.ptr == output_device.ptr
        || workspace.ptr == output_device.ptr
    {
        return Err(RuntimeGenerationRebuildPrepareError::InvalidInput(
            "distinct rebuild device buffers",
        ));
    }
    let mut input_host = primary
        .lease_pinned_host_buffer_owned(input_bytes)
        .ok_or(RuntimeGenerationRebuildPrepareError::PinnedHostStagingUnavailable)?;
    let output_host = primary
        .lease_pinned_host_buffer_owned(abi::OUTPUT_BYTES)
        .ok_or(RuntimeGenerationRebuildPrepareError::PinnedHostStagingUnavailable)?;
    let (resident_owners, cold_owners) = encode_input_arena(
        input,
        input_host.as_mut_bytes(input_bytes)?,
        descriptor_bytes,
        input_device.ptr,
        workspace.ptr,
    )?;
    let stream = PooledStreamOwned {
        primary: Arc::clone(&primary),
        pooled: Some(primary.acquire_pooled_stream()?),
    };
    let (function, launch) = resolve_runtime_generation_rebuild_launch(&primary)?;
    Ok(PreparedRuntimeGenerationRebuild {
        primary,
        resources: RuntimeGenerationRebuildResources {
            _resident_owners: resident_owners,
            _cold_owners: cold_owners,
            input_host,
            input_device,
            _workspace: workspace,
            output_host,
            output_device,
            stream,
            input_bytes,
            proof_slots: vec![[0; 32]; abi::PROOF_DIGEST_SLOTS].into_boxed_slice(),
        },
        attempt: input.attempt,
        function,
        launch,
    })
}

fn validate_input(
    input: &RuntimeGenerationRebuildInput,
) -> Result<(), RuntimeGenerationRebuildPrepareError> {
    if input.database_id == [0; 16]
        || input.table_id == 0
        || input.data_generation == 0
        || input.visibility_cut == 0
        || input.logical_row_count == 0
        || input.column_id == 0
        || input.attnum <= 0
        || input.declared_type_oid != INT4_OID
        || input.signed_type_size != INT4_SIGNED_SIZE
        || input.shards.is_empty()
        || input.shards.len() > u32::MAX as usize
    {
        return Err(RuntimeGenerationRebuildPrepareError::InvalidInput(
            "unsupported runtime-generation V1 single-table INT4 shape",
        ));
    }
    validate_proof_vector_geometry(input.logical_row_count)?;
    let mut covered = 0_u64;
    for shard in input.shards.iter() {
        if shard.row_count == 0 || shard.row_start != covered {
            return Err(RuntimeGenerationRebuildPrepareError::InvalidInput(
                "noncanonical rebuild shard coverage",
            ));
        }
        validate_role_sources(input, shard)?;
        covered = covered.checked_add(shard.row_count).ok_or(
            RuntimeGenerationRebuildPrepareError::InvalidInput("rebuild row count overflow"),
        )?;
    }
    if covered != input.logical_row_count {
        return Err(RuntimeGenerationRebuildPrepareError::InvalidInput(
            "rebuild logical row coverage",
        ));
    }
    Ok(())
}

fn validate_proof_vector_geometry(rows: u64) -> Result<(), RuntimeGenerationRebuildPrepareError> {
    let frames = abi::proof_vector_preimage_bytes(rows).ok_or(
        RuntimeGenerationRebuildPrepareError::InvalidInput("rebuild proof-vector capacity"),
    )?;
    if frames.iter().any(|bytes| *bytes > u64::from(u32::MAX)) {
        return Err(RuntimeGenerationRebuildPrepareError::InvalidInput(
            "rebuild proof-vector capacity",
        ));
    }
    let workspace = abi::workspace_layout(rows).ok_or(
        RuntimeGenerationRebuildPrepareError::InvalidInput("workspace capacity"),
    )?;
    if frames.iter().any(|bytes| *bytes > workspace.scratch_bytes) {
        return Err(RuntimeGenerationRebuildPrepareError::InvalidInput(
            "rebuild proof-vector capacity",
        ));
    }
    Ok(())
}

fn validate_role_sources(
    input: &RuntimeGenerationRebuildInput,
    shard: &RuntimeGenerationRebuildShard,
) -> Result<(), RuntimeGenerationRebuildPrepareError> {
    let ids = checked_bytes(shard.row_count, 8)?;
    let values = checked_bytes(shard.row_count, 4)?;
    let validity = shard
        .row_count
        .checked_add(31)
        .and_then(|rows| rows.checked_div(32))
        .and_then(|words| words.checked_mul(4))
        .ok_or(RuntimeGenerationRebuildPrepareError::InvalidInput(
            "validity capacity",
        ))?;
    for (role, required) in shard
        .role_sources()
        .into_iter()
        .zip([ids, validity, values, ids, ids])
    {
        if role.span.byte_len != required || span_end(role.span)? > role.source.byte_len() {
            return Err(RuntimeGenerationRebuildPrepareError::InvalidInput(
                "rebuild physical role geometry",
            ));
        }
        validate_source(input, &role.source)?;
    }
    Ok(())
}

fn validate_source(
    input: &RuntimeGenerationRebuildInput,
    source: &RuntimeGenerationRebuildSource,
) -> Result<(), RuntimeGenerationRebuildPrepareError> {
    match &source.backing {
        RuntimeGenerationRebuildSourceBacking::Resident {
            memory,
            byte_offset,
            byte_len,
        } => {
            let end = byte_offset.checked_add(*byte_len).ok_or(
                RuntimeGenerationRebuildPrepareError::InvalidInput("resident source bounds"),
            )?;
            if *byte_len == 0
                || end > memory.metadata().allocated_bytes
                || memory.metadata().gpu_id != input.target.device_ordinal
                || !Arc::ptr_eq(&memory.primary_arc(), &input.target.primary)
            {
                return Err(RuntimeGenerationRebuildPrepareError::InvalidInput(
                    "resident rebuild target",
                ));
            }
        }
        RuntimeGenerationRebuildSourceBacking::ColdRam {
            bytes,
            byte_offset,
            byte_len,
        } => {
            let end = byte_offset.checked_add(*byte_len).ok_or(
                RuntimeGenerationRebuildPrepareError::InvalidInput("cold source bounds"),
            )?;
            if *byte_len == 0 || end > bytes.len() as u64 {
                return Err(RuntimeGenerationRebuildPrepareError::InvalidInput(
                    "cold rebuild source",
                ));
            }
        }
    }
    Ok(())
}

fn checked_cold_bytes(
    input: &RuntimeGenerationRebuildInput,
) -> Result<usize, RuntimeGenerationRebuildPrepareError> {
    unique_rebuild_sources(input)
        .into_iter()
        .try_fold(0_usize, |total, source| {
            let bytes = match &source.backing {
                RuntimeGenerationRebuildSourceBacking::Resident { .. } => 0,
                RuntimeGenerationRebuildSourceBacking::ColdRam { byte_len, .. } => {
                    usize::try_from(*byte_len).map_err(|_| {
                        RuntimeGenerationRebuildPrepareError::InvalidInput(
                            "cold arena addressability",
                        )
                    })?
                }
            };
            total
                .checked_add(bytes)
                .ok_or(RuntimeGenerationRebuildPrepareError::InvalidInput(
                    "cold arena capacity",
                ))
        })
}

fn unique_rebuild_sources(
    input: &RuntimeGenerationRebuildInput,
) -> Vec<&RuntimeGenerationRebuildSource> {
    let mut unique: Vec<&RuntimeGenerationRebuildSource> = Vec::new();
    for source in input
        .shards
        .iter()
        .flat_map(RuntimeGenerationRebuildShard::role_sources)
        .map(|role| &role.source)
    {
        if !unique
            .iter()
            .any(|existing| existing.is_same_transport_source(source))
        {
            unique.push(source);
        }
    }
    unique
}

type RuntimeGenerationRebuildResidentOwners =
    Box<[Arc<crate::resident_memory::CudaResidentDeviceAllocation>]>;
type RuntimeGenerationRebuildColdOwners = Box<[Arc<[u8]>]>;

struct EncodedRuntimeGenerationRebuildSource<'a> {
    source: &'a RuntimeGenerationRebuildSource,
    device_pointer: u64,
}

fn encode_input_arena(
    input: &RuntimeGenerationRebuildInput,
    bytes: &mut [u8],
    descriptor_bytes: usize,
    input_device_pointer: u64,
    workspace_pointer: u64,
) -> Result<
    (
        RuntimeGenerationRebuildResidentOwners,
        RuntimeGenerationRebuildColdOwners,
    ),
    RuntimeGenerationRebuildPrepareError,
> {
    bytes.fill(0);
    bytes[abi::DATABASE_ID_OFFSET..abi::DATABASE_ID_OFFSET + 16]
        .copy_from_slice(&input.database_id);
    abi::put_u64(bytes, abi::TABLE_ID_OFFSET, input.table_id);
    abi::put_u64(bytes, abi::DATA_GENERATION_OFFSET, input.data_generation);
    abi::put_u64(bytes, abi::ROW_COUNT_OFFSET, input.logical_row_count);
    abi::put_u64(bytes, abi::VISIBILITY_CUT_OFFSET, input.visibility_cut);
    abi::put_u64(bytes, abi::COLUMN_ID_OFFSET, input.column_id);
    abi::put_u16(bytes, abi::ATTNUM_OFFSET, input.attnum as u16);
    abi::put_u32(bytes, abi::DECLARED_OID_OFFSET, input.declared_type_oid);
    abi::put_u16(
        bytes,
        abi::SIGNED_SIZE_OFFSET,
        input.signed_type_size as u16,
    );
    abi::put_u16(bytes, abi::ROOT_FORMAT_OFFSET, ROOT_FORMAT_V1);
    abi::put_u32(
        bytes,
        abi::SHARD_COUNT_OFFSET,
        u32::try_from(input.shards.len()).map_err(|_| {
            RuntimeGenerationRebuildPrepareError::InvalidInput("rebuild shard count")
        })?,
    );
    abi::put_u64(bytes, abi::WORKSPACE_POINTER_OFFSET, workspace_pointer);

    let mut cold_cursor = descriptor_bytes;
    let mut resident_owners = Vec::new();
    let mut cold_owners = Vec::new();
    let mut encoded_sources = Vec::new();
    for source in unique_rebuild_sources(input) {
        let device_pointer = match &source.backing {
            RuntimeGenerationRebuildSourceBacking::Resident {
                memory,
                byte_offset,
                ..
            } => {
                let owner = memory.allocation_arc();
                if !resident_owners
                    .iter()
                    .any(|existing| Arc::ptr_eq(existing, &owner))
                {
                    resident_owners.push(owner);
                }
                memory.device_ptr().checked_add(*byte_offset).ok_or(
                    RuntimeGenerationRebuildPrepareError::InvalidInput("resident source pointer"),
                )?
            }
            RuntimeGenerationRebuildSourceBacking::ColdRam {
                bytes: cold,
                byte_offset,
                byte_len,
            } => {
                let copy_len = usize::try_from(*byte_len).map_err(|_| {
                    RuntimeGenerationRebuildPrepareError::InvalidInput("cold source addressability")
                })?;
                let copy_start = usize::try_from(*byte_offset).map_err(|_| {
                    RuntimeGenerationRebuildPrepareError::InvalidInput("cold source addressability")
                })?;
                let copy_end = copy_start.checked_add(copy_len).ok_or(
                    RuntimeGenerationRebuildPrepareError::InvalidInput("cold source bounds"),
                )?;
                let arena_end = cold_cursor.checked_add(copy_len).ok_or(
                    RuntimeGenerationRebuildPrepareError::InvalidInput("cold arena capacity"),
                )?;
                bytes[cold_cursor..arena_end].copy_from_slice(&cold[copy_start..copy_end]);
                let pointer = input_device_pointer
                    .checked_add(u64::try_from(cold_cursor).map_err(|_| {
                        RuntimeGenerationRebuildPrepareError::InvalidInput("cold device pointer")
                    })?)
                    .ok_or(RuntimeGenerationRebuildPrepareError::InvalidInput(
                        "cold device pointer",
                    ))?;
                cold_cursor = arena_end;
                if !cold_owners
                    .iter()
                    .any(|existing| Arc::ptr_eq(existing, cold))
                {
                    cold_owners.push(Arc::clone(cold));
                }
                pointer
            }
        };
        encoded_sources.push(EncodedRuntimeGenerationRebuildSource {
            source,
            device_pointer,
        });
    }
    for (ordinal, shard) in input.shards.iter().enumerate() {
        let base = abi::DESCRIPTOR_HEADER_BYTES + ordinal * abi::SHARD_DESCRIPTOR_BYTES;
        abi::put_u64(bytes, base + abi::SHARD_ROW_START_OFFSET, shard.row_start);
        abi::put_u64(bytes, base + abi::SHARD_ROW_COUNT_OFFSET, shard.row_count);
        for (pointer_offset, role) in [
            (
                abi::SHARD_ROW_ID_POINTER_OFFSET,
                &shard.roles.stable_row_ids,
            ),
            (abi::SHARD_VALIDITY_POINTER_OFFSET, &shard.roles.validity),
            (abi::SHARD_VALUE_POINTER_OFFSET, &shard.roles.values),
            (abi::SHARD_CREATED_POINTER_OFFSET, &shard.roles.created_by),
            (abi::SHARD_DELETED_POINTER_OFFSET, &shard.roles.deleted_by),
        ] {
            let source_pointer = encoded_sources
                .iter()
                .find(|encoded| encoded.source.is_same_transport_source(&role.source))
                .map(|encoded| encoded.device_pointer)
                .ok_or(RuntimeGenerationRebuildPrepareError::InvalidInput(
                    "rebuild role source",
                ))?;
            encode_shard_role_pointer(bytes, base + pointer_offset, source_pointer, role.span)?;
        }
    }
    if cold_cursor != bytes.len() {
        return Err(RuntimeGenerationRebuildPrepareError::InvalidInput(
            "rebuild cold arena accounting",
        ));
    }
    Ok((
        resident_owners.into_boxed_slice(),
        cold_owners.into_boxed_slice(),
    ))
}

fn encode_shard_role_pointer(
    bytes: &mut [u8],
    offset: usize,
    source_pointer: u64,
    role: RuntimeGenerationRebuildRoleSpan,
) -> Result<(), RuntimeGenerationRebuildPrepareError> {
    let pointer = source_pointer.checked_add(role.byte_offset).ok_or(
        RuntimeGenerationRebuildPrepareError::InvalidInput("rebuild role pointer"),
    )?;
    abi::put_u64(bytes, offset, pointer);
    Ok(())
}

fn span_end(
    span: RuntimeGenerationRebuildRoleSpan,
) -> Result<u64, RuntimeGenerationRebuildPrepareError> {
    span.byte_offset.checked_add(span.byte_len).ok_or(
        RuntimeGenerationRebuildPrepareError::InvalidInput("rebuild role span"),
    )
}

fn checked_bytes(rows: u64, stride: u64) -> Result<u64, RuntimeGenerationRebuildPrepareError> {
    rows.checked_mul(stride)
        .ok_or(RuntimeGenerationRebuildPrepareError::InvalidInput(
            "rebuild physical capacity",
        ))
}

fn resolve_runtime_generation_rebuild_launch(
    primary: &GpuPrimaryContext,
) -> Result<(*mut c_void, CuLaunchKernel), CudaRuntimeProbeError> {
    let mut ptx = Vec::with_capacity(RUNTIME_GENERATION_REBUILD_PTX.len() + 1);
    ptx.extend_from_slice(RUNTIME_GENERATION_REBUILD_PTX);
    ptx.push(0);
    let function = primary.cached_function(
        c"gpu_db_runtime_generation_v1_single_table_int4_rebuild",
        &ptx,
    )?;
    let launch = unsafe {
        *primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    Ok((function, launch))
}

unsafe fn launch_runtime_generation_rebuild_kernel(
    launch: CuLaunchKernel,
    function: *mut c_void,
    descriptor_device_pointer: u64,
    output_device_pointer: u64,
    stream: *mut c_void,
) -> i32 {
    let mut descriptor = descriptor_device_pointer;
    let mut output = output_device_pointer;
    let mut args = [
        (&mut descriptor as *mut u64).cast::<c_void>(),
        (&mut output as *mut u64).cast::<c_void>(),
    ];
    unsafe {
        launch(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::CudaDriverRuntime;
    use sha2::{Digest, Sha256};

    fn cuda_runtime() -> Option<CudaDriverRuntime> {
        let runtime = CudaDriverRuntime::probe().ok()?;
        let snapshot = runtime.snapshot();
        (snapshot.driver_available && snapshot.device_count > 0).then_some(runtime)
    }

    fn payload(
        rows: &[(u64, Option<i32>)],
        created: u64,
        deleted: u64,
        tail_noise: bool,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        for (id, _) in rows {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        let validity_offset = bytes.len();
        let mut validity = vec![0_u8; rows.len().div_ceil(32) * 4];
        for (row, (_, value)) in rows.iter().enumerate() {
            if value.is_some() {
                validity[row / 8] |= 1 << (row % 8);
            }
        }
        if tail_noise {
            validity[0] |= 0x80;
        }
        bytes.extend_from_slice(&validity);
        for (_, value) in rows {
            bytes.extend_from_slice(&value.unwrap_or_default().to_le_bytes());
        }
        for _ in rows {
            bytes.extend_from_slice(&created.to_le_bytes());
        }
        for _ in rows {
            bytes.extend_from_slice(&deleted.to_le_bytes());
        }
        assert_eq!(validity_offset, rows.len() * 8);
        bytes
    }

    fn roles_for_rows(rows: u64) -> RuntimeGenerationRebuildShardRoles {
        let stable_bytes = rows.checked_mul(8).expect("test rows");
        let validity_bytes = rows
            .checked_add(31)
            .and_then(|rows| rows.checked_div(32))
            .and_then(|words| words.checked_mul(4))
            .expect("test rows");
        let value_bytes = rows.checked_mul(4).expect("test rows");
        RuntimeGenerationRebuildShardRoles {
            stable_row_ids: RuntimeGenerationRebuildRoleSpan {
                byte_offset: 0,
                byte_len: stable_bytes,
            },
            validity: RuntimeGenerationRebuildRoleSpan {
                byte_offset: stable_bytes,
                byte_len: validity_bytes,
            },
            values: RuntimeGenerationRebuildRoleSpan {
                byte_offset: stable_bytes + validity_bytes,
                byte_len: value_bytes,
            },
            created_by: RuntimeGenerationRebuildRoleSpan {
                byte_offset: stable_bytes + validity_bytes + value_bytes,
                byte_len: stable_bytes,
            },
            deleted_by: RuntimeGenerationRebuildRoleSpan {
                byte_offset: stable_bytes + validity_bytes + value_bytes + stable_bytes,
                byte_len: stable_bytes,
            },
        }
    }

    fn roles(rows: usize) -> RuntimeGenerationRebuildShardRoles {
        roles_for_rows(u64::try_from(rows).expect("test rows"))
    }

    fn input(
        target: RuntimeGenerationRebuildTarget,
        attempt: u64,
        sources: Vec<(RuntimeGenerationRebuildSource, usize)>,
    ) -> RuntimeGenerationRebuildInput {
        let mut row_start = 0_u64;
        let mut shards = Vec::new();
        for (source, rows) in sources {
            shards.push(RuntimeGenerationRebuildShard::new(
                source,
                row_start,
                rows as u64,
                roles(rows),
            ));
            row_start += rows as u64;
        }
        RuntimeGenerationRebuildInput::new(
            target,
            RuntimeGenerationRebuildAttempt::new(attempt).expect("nonzero attempt"),
            [0x5a; 16],
            41,
            9,
            10,
            row_start,
            7,
            1,
            INT4_OID,
            INT4_SIGNED_SIZE,
            shards.into_boxed_slice(),
        )
    }

    fn complete(input: RuntimeGenerationRebuildInput) -> OpaqueRuntimeGenerationRebuildProof {
        match PreparedRuntimeGenerationRebuild::prepare(input)
            .expect("preparation")
            .enqueue()
            .complete()
        {
            RuntimeGenerationRebuildCompletion::Quiesced(Ok(proof)) => proof,
            RuntimeGenerationRebuildCompletion::Quiesced(Err(error)) => {
                panic!("GPU rebuild rejected valid source: {error:?}")
            }
            RuntimeGenerationRebuildCompletion::UnknownQuiescence(unknown) => {
                panic!(
                    "GPU rebuild fence unexpectedly unknown: {:?}",
                    unknown.error()
                )
            }
        }
    }

    fn proof_for_commitment_test(
        attempt: u64,
        table_root: [u8; 32],
        database_root: [u8; 32],
    ) -> OpaqueRuntimeGenerationRebuildProof {
        let mut slots = vec![[0_u8; 32]; abi::PROOF_DIGEST_SLOTS].into_boxed_slice();
        slots[abi::SLOT_TABLE_ROOT] = table_root;
        slots[abi::SLOT_DATABASE_ROOT] = database_root;
        OpaqueRuntimeGenerationRebuildProof {
            attempt: RuntimeGenerationRebuildAttempt::new(attempt).expect("nonzero attempt"),
            root_format: ROOT_FORMAT_V1,
            slots,
        }
    }

    #[test]
    fn durable_commitments_are_exactly_two_roots_and_gate_the_opaque_handoff() {
        let expected =
            proof_for_commitment_test(1, [0x41; 32], [0x42; 32]).into_durable_v1_commitments();
        let mut durable = [0_u8; RUNTIME_GENERATION_REBUILD_V1_DURABLE_COMMITMENT_BYTES];
        expected
            .encode_durable_into(&mut durable)
            .expect("fixed durable commitment encoding");
        assert_eq!(&durable[..32], &[0x41; 32]);
        assert_eq!(&durable[32..], &[0x42; 32]);
        assert!(matches!(
            RuntimeGenerationRebuildV1DurableCommitments::decode_durable(&durable[..63]),
            Err(RuntimeGenerationRebuildV1CommitmentBytesError::InvalidLength(63))
        ));
        let expected = RuntimeGenerationRebuildV1DurableCommitments::decode_durable(&durable)
            .expect("fixed durable commitment decoding");
        let table_map = proof_for_commitment_test(2, [0x41; 32], [0x42; 32])
            .compare_durable_v1_commitments_with_table_map(
                &expected,
                |attempt, table_root, table_map, database_root| {
                    table_map.consume(|empty_roots, leaf_root, path_roots| {
                        (
                            attempt.get(),
                            format!("{table_root:?}"),
                            empty_roots.len(),
                            format!("{leaf_root:?}"),
                            path_roots.len(),
                            format!("{database_root:?}"),
                        )
                    })
                },
            )
            .expect("matching commitments preserve the sealed table-map completion");
        assert_eq!(table_map.0, 2);
        assert_eq!(table_map.1, "OpaqueCudaSha256Digest(<opaque>)");
        assert_eq!(table_map.2, 65);
        assert_eq!(table_map.3, "OpaqueCudaSha256Digest(<opaque>)");
        assert_eq!(table_map.4, 64);
        assert_eq!(table_map.5, "OpaqueCudaSha256Digest(<opaque>)");

        let mut map_consumed = false;
        assert_eq!(
            proof_for_commitment_test(3, [0x43; 32], [0x42; 32])
                .compare_durable_v1_commitments_with_table_map(&expected, |_, _, _, _| {
                    map_consumed = true;
                }),
            Err(RuntimeGenerationRebuildV1CommitmentMismatch)
        );
        assert!(
            !map_consumed,
            "a mismatched proof must not reveal the table-map completion"
        );
    }

    #[test]
    fn one_source_shard_convenience_retains_one_exact_source_for_all_roles() {
        let source =
            RuntimeGenerationRebuildSource::cold_ram(Arc::<[u8]>::from(vec![0_u8; 32]), 0, 32);
        let shard = RuntimeGenerationRebuildShard::new(source, 0, 1, roles(1));
        let roles = shard.role_sources();
        for role in roles.iter().skip(1) {
            assert!(roles[0].source.is_same_transport_source(&role.source));
        }
    }

    fn independent_role_sources(
        runtime: &CudaDriverRuntime,
        bytes: &[u8],
        rows: usize,
    ) -> RuntimeGenerationRebuildShardRoleSources {
        let roles = roles(rows);
        let cold_source = |span: RuntimeGenerationRebuildRoleSpan| {
            let begin = usize::try_from(span.byte_offset).expect("test role offset");
            let end = begin
                .checked_add(usize::try_from(span.byte_len).expect("test role length"))
                .expect("test role range");
            RuntimeGenerationRebuildShardRoleSource::new(
                RuntimeGenerationRebuildSource::cold_ram(
                    Arc::<[u8]>::from(bytes[begin..end].to_vec()),
                    0,
                    span.byte_len,
                ),
                RuntimeGenerationRebuildRoleSpan {
                    byte_offset: 0,
                    byte_len: span.byte_len,
                },
            )
        };
        let stable_begin =
            usize::try_from(roles.stable_row_ids.byte_offset).expect("test role offset");
        let stable_end = stable_begin
            .checked_add(usize::try_from(roles.stable_row_ids.byte_len).expect("test role length"))
            .expect("test role range");
        let stable_row_ids = RuntimeGenerationRebuildShardRoleSource::new(
            RuntimeGenerationRebuildSource::resident(
                Arc::new(
                    runtime
                        .retain_device_memory_copy(0, &bytes[stable_begin..stable_end])
                        .expect("resident stable IDs"),
                ),
                0,
                roles.stable_row_ids.byte_len,
            ),
            RuntimeGenerationRebuildRoleSpan {
                byte_offset: 0,
                byte_len: roles.stable_row_ids.byte_len,
            },
        );
        RuntimeGenerationRebuildShardRoleSources {
            stable_row_ids,
            validity: cold_source(roles.validity),
            values: cold_source(roles.values),
            created_by: cold_source(roles.created_by),
            deleted_by: cold_source(roles.deleted_by),
        }
    }

    #[test]
    fn gpu_rebuild_accepts_independently_retained_five_role_sources() {
        let Some(runtime) = cuda_runtime() else {
            return;
        };
        let target = runtime
            .runtime_generation_rebuild_target(0)
            .expect("primary target");
        let bytes = payload(&[(2, Some(11)), (5, None)], 3, 100, false);
        let expected = complete(input(
            target.clone(),
            70,
            vec![(
                RuntimeGenerationRebuildSource::cold_ram(
                    Arc::<[u8]>::from(bytes.clone()),
                    0,
                    bytes.len() as u64,
                ),
                2,
            )],
        ));
        let actual = complete(RuntimeGenerationRebuildInput::new(
            target,
            RuntimeGenerationRebuildAttempt::new(71).expect("nonzero attempt"),
            [0x5a; 16],
            41,
            9,
            10,
            2,
            7,
            1,
            INT4_OID,
            INT4_SIGNED_SIZE,
            Box::new([RuntimeGenerationRebuildShard::from_role_sources(
                0,
                2,
                independent_role_sources(&runtime, &bytes, 2),
            )]),
        ));
        assert_eq!(
            actual.slots[abi::SLOT_TABLE_ROOT],
            expected.slots[abi::SLOT_TABLE_ROOT],
            "per-role source ownership cannot enter the table root",
        );
        assert_eq!(
            actual.slots[abi::SLOT_DATABASE_ROOT],
            expected.slots[abi::SLOT_DATABASE_ROOT],
            "per-role source ownership cannot enter the database root",
        );
    }

    #[test]
    fn former_vector_only_ceiling_is_rejected_before_cuda_enqueue() {
        let Some(runtime) = cuda_runtime() else {
            return;
        };
        let target = runtime
            .runtime_generation_rebuild_target(0)
            .expect("primary target");
        let former_vector_only_ceiling = u64::from(u32::MAX) / abi::DIGEST_BYTES;
        assert_eq!(
            former_vector_only_ceiling,
            abi::proof_vector_row_ceiling() + 2,
            "the sabotage must use a count accepted by the former payload-only limit"
        );
        let rejected = RuntimeGenerationRebuildInput::new(
            target,
            RuntimeGenerationRebuildAttempt::new(99).expect("nonzero attempt"),
            [0x5a; 16],
            41,
            9,
            10,
            former_vector_only_ceiling,
            7,
            1,
            INT4_OID,
            INT4_SIGNED_SIZE,
            Box::new([RuntimeGenerationRebuildShard::new(
                RuntimeGenerationRebuildSource::cold_ram(
                    Arc::<[u8]>::from(vec![0_u8]),
                    0,
                    u64::MAX,
                ),
                0,
                former_vector_only_ceiling,
                roles_for_rows(former_vector_only_ceiling),
            )]),
        );
        let failure = match PreparedRuntimeGenerationRebuild::prepare(rejected) {
            Ok(_) => panic!("former payload-only vector ceiling must reject before enqueue"),
            Err(failure) => failure,
        };
        assert_eq!(
            failure.error(),
            &RuntimeGenerationRebuildPrepareError::InvalidInput("rebuild proof-vector capacity")
        );
        assert_eq!(
            failure.into_input().logical_row_count,
            former_vector_only_ceiling,
            "the pre-enqueue failure must retain its exact source for recovery"
        );
    }

    fn digest(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update((domain.len() as u64).to_le_bytes());
        hasher.update(domain);
        for field in fields {
            hasher.update(field);
        }
        hasher.finalize().into()
    }

    fn row_map_reference(
        table_id: u64,
        rows: &[(u64, [u8; 32])],
        empty: &[[u8; 32]; 65],
        depth: u8,
    ) -> ([u8; 32], u64) {
        if rows.is_empty() {
            return (empty[depth as usize], 0);
        }
        if depth == 64 {
            let (row_id, leaf) = rows[0];
            return (
                digest(
                    b"gpu-db/runtime-generation/row-leaf/v1",
                    &[
                        &1_u16.to_le_bytes(),
                        &table_id.to_le_bytes(),
                        &row_id.to_le_bytes(),
                        &1_u64.to_le_bytes(),
                        &leaf,
                    ],
                ),
                1,
            );
        }
        let split = rows.partition_point(|(row_id, _)| ((row_id >> (63 - depth)) & 1) == 0);
        let (left, left_count) = row_map_reference(table_id, &rows[..split], empty, depth + 1);
        let (right, right_count) = row_map_reference(table_id, &rows[split..], empty, depth + 1);
        let count = left_count + right_count;
        (
            digest(
                b"gpu-db/runtime-generation/row-node/v1",
                &[
                    &1_u16.to_le_bytes(),
                    &table_id.to_le_bytes(),
                    &[depth],
                    &count.to_le_bytes(),
                    &left,
                    &right,
                ],
            ),
            count,
        )
    }

    fn database_root_reference(
        rows: &[(u64, Option<i32>)],
        actual: &OpaqueRuntimeGenerationRebuildProof,
    ) -> [u8; 32] {
        let database_id = [0x5a; 16];
        let table_id = 41_u64;
        let column_id = 7_u64;
        let shape = digest(
            b"gpu-db/runtime-generation/column-shape/v1",
            &[
                &table_id.to_le_bytes(),
                &column_id.to_le_bytes(),
                &1_i16.to_le_bytes(),
                b"int4",
                &INT4_OID.to_le_bytes(),
                &INT4_SIGNED_SIZE.to_le_bytes(),
            ],
        );
        assert_eq!(actual.slots[abi::SLOT_COLUMN_SHAPE], shape, "column shape");
        let leaf_rows = rows
            .iter()
            .map(|(row_id, value)| {
                let (null_flag, length, bytes) = match value {
                    Some(value) => (0_u8, 4_u32, value.to_le_bytes().to_vec()),
                    None => (1_u8, 0_u32, Vec::new()),
                };
                let typed = digest(
                    b"gpu-db/runtime-generation/typed-value/v1",
                    &[&shape, &[null_flag], &length.to_le_bytes(), &bytes],
                );
                let current = digest(
                    b"gpu-db/runtime-generation/current-row/v1",
                    &[
                        &table_id.to_le_bytes(),
                        &row_id.to_le_bytes(),
                        &3_u64.to_le_bytes(),
                        &1_u32.to_le_bytes(),
                        &column_id.to_le_bytes(),
                        &shape,
                        &typed,
                    ],
                );
                (*row_id, current)
            })
            .collect::<Vec<_>>();
        let mut empty = [[0_u8; 32]; 65];
        empty[64] = digest(
            b"gpu-db/runtime-generation/row-empty-leaf/v1",
            &[
                &1_u16.to_le_bytes(),
                &table_id.to_le_bytes(),
                &0_u64.to_le_bytes(),
            ],
        );
        for depth in (0..64).rev() {
            empty[depth] = digest(
                b"gpu-db/runtime-generation/row-empty-node/v1",
                &[
                    &1_u16.to_le_bytes(),
                    &table_id.to_le_bytes(),
                    &[depth as u8],
                    &0_u64.to_le_bytes(),
                    &empty[depth + 1],
                    &empty[depth + 1],
                ],
            );
        }
        for (depth, root) in empty.iter().enumerate() {
            assert_eq!(
                actual.slots[abi::SLOT_ROW_EMPTY + depth],
                *root,
                "row empty {depth}"
            );
        }
        let (row_map, row_count) = row_map_reference(table_id, &leaf_rows, &empty, 0);
        let table_root = digest(
            b"gpu-db/runtime-generation/table-root/v1",
            &[
                &table_id.to_le_bytes(),
                &9_u64.to_le_bytes(),
                &row_count.to_le_bytes(),
                &row_map,
                &0_u32.to_le_bytes(),
            ],
        );
        assert_eq!(actual.slots[abi::SLOT_TABLE_ROOT], table_root, "table root");
        let mut map_empty = [[0_u8; 32]; 65];
        map_empty[64] = digest(
            b"gpu-db/runtime-generation/map-empty-leaf/v1",
            &[&1_u16.to_le_bytes(), &database_id],
        );
        for depth in (0..64).rev() {
            map_empty[depth] = digest(
                b"gpu-db/runtime-generation/map-empty-node/v1",
                &[
                    &1_u16.to_le_bytes(),
                    &database_id,
                    &[depth as u8],
                    &map_empty[depth + 1],
                    &map_empty[depth + 1],
                ],
            );
        }
        for (depth, root) in map_empty.iter().enumerate() {
            assert_eq!(
                actual.slots[abi::SLOT_TABLE_MAP_EMPTY + depth],
                *root,
                "table map empty {depth}",
            );
        }
        let mut map = digest(
            b"gpu-db/runtime-generation/map-leaf/v1",
            &[
                &1_u16.to_le_bytes(),
                &database_id,
                &table_id.to_le_bytes(),
                &table_root,
            ],
        );
        assert_eq!(
            actual.slots[abi::SLOT_TABLE_MAP_LEAF],
            map,
            "table map leaf"
        );
        for depth in (0..64).rev() {
            map = if ((table_id >> (63 - depth)) & 1) == 0 {
                digest(
                    b"gpu-db/runtime-generation/map-node/v1",
                    &[
                        &1_u16.to_le_bytes(),
                        &database_id,
                        &[depth as u8],
                        &map,
                        &map_empty[depth + 1],
                    ],
                )
            } else {
                digest(
                    b"gpu-db/runtime-generation/map-node/v1",
                    &[
                        &1_u16.to_le_bytes(),
                        &database_id,
                        &[depth as u8],
                        &map_empty[depth + 1],
                        &map,
                    ],
                )
            };
            assert_eq!(
                actual.slots[abi::SLOT_TABLE_MAP_PATH + depth],
                map,
                "table map path {depth}",
            );
        }
        digest(
            b"gpu-db/runtime-generation/database-root/v1",
            &[&1_u16.to_le_bytes(), &database_id, &map],
        )
    }

    #[test]
    fn exact_output_abi_stays_opaque_and_closed() {
        assert_eq!(abi::OUTPUT_BYTES, 4 + 200 * 32);
        assert_eq!(abi::SLOT_COLUMN_SHAPE, 0);
        assert_eq!(abi::SLOT_TYPED_VECTOR, 1);
        assert_eq!(abi::SLOT_CURRENT_ROW_LEAVES, 2);
        assert_eq!(abi::SLOT_ROW_EMPTY + 64, abi::SLOT_TABLE_ROOT - 1);
        assert_eq!(abi::SLOT_TABLE_MAP_EMPTY + 64, abi::SLOT_TABLE_MAP_LEAF - 1);
        assert_eq!(abi::SLOT_TABLE_MAP_PATH + 63, abi::SLOT_DATABASE_ROOT - 1);
    }

    #[test]
    fn gpu_rebuild_is_logical_across_resident_cold_and_shard_layouts() {
        let Some(runtime) = cuda_runtime() else {
            return;
        };
        let target = runtime
            .runtime_generation_rebuild_target(0)
            .expect("primary target");
        let first = payload(&[(2, Some(11)), (5, None)], 3, 100, false);
        let second = payload(&[(9, Some(-7))], 3, 100, false);
        let resident = Arc::new(
            runtime
                .retain_device_memory_copy(0, &first)
                .expect("resident shard"),
        );
        let mixed = complete(input(
            target.clone(),
            1,
            vec![
                (
                    RuntimeGenerationRebuildSource::resident(resident, 0, first.len() as u64),
                    2,
                ),
                (
                    RuntimeGenerationRebuildSource::cold_ram(
                        Arc::<[u8]>::from(second.clone()),
                        0,
                        second.len() as u64,
                    ),
                    1,
                ),
            ],
        ));
        let merged = payload(&[(2, Some(11)), (5, None), (9, Some(-7))], 3, 100, false);
        let cold = complete(input(
            target,
            2,
            vec![(
                RuntimeGenerationRebuildSource::cold_ram(
                    Arc::<[u8]>::from(merged.clone()),
                    0,
                    merged.len() as u64,
                ),
                3,
            )],
        ));
        assert_eq!(mixed.slot_count(), abi::PROOF_DIGEST_SLOTS);
        assert_eq!(
            mixed.slots[abi::SLOT_DATABASE_ROOT],
            cold.slots[abi::SLOT_DATABASE_ROOT],
            "physical source tier and shard boundaries cannot enter the database root",
        );
        assert_eq!(
            mixed.slots[abi::SLOT_TABLE_ROOT],
            cold.slots[abi::SLOT_TABLE_ROOT],
            "the typed table root is independent of resident/cold ownership",
        );
        assert_eq!(
            mixed.slots[abi::SLOT_DATABASE_ROOT],
            database_root_reference(&[(2, Some(11)), (5, None), (9, Some(-7))], &mixed),
            "the device root must exactly follow the V1 canonical grammar",
        );
    }

    #[test]
    fn gpu_rebuild_rejects_nonzero_validity_tail_on_device() {
        let Some(runtime) = cuda_runtime() else {
            return;
        };
        let target = runtime
            .runtime_generation_rebuild_target(0)
            .expect("primary target");
        let source = payload(&[(2, Some(11))], 3, 100, true);
        match PreparedRuntimeGenerationRebuild::prepare(input(
            target,
            3,
            vec![(
                RuntimeGenerationRebuildSource::cold_ram(
                    Arc::<[u8]>::from(source.clone()),
                    0,
                    source.len() as u64,
                ),
                1,
            )],
        ))
        .expect("preparation")
        .enqueue()
        .complete()
        {
            RuntimeGenerationRebuildCompletion::Quiesced(Err(
                RuntimeGenerationRebuildError::DeviceRejected(3),
            )) => {}
            _ => panic!("validity-tail corruption must be rejected by the device"),
        }
    }

    #[test]
    fn gpu_rebuild_rejects_zero_or_nonascending_stable_row_ids_on_device() {
        let Some(runtime) = cuda_runtime() else {
            return;
        };
        let target = runtime
            .runtime_generation_rebuild_target(0)
            .expect("primary target");
        let source = payload(&[(5, Some(11)), (2, Some(12))], 3, 100, false);
        match PreparedRuntimeGenerationRebuild::prepare(input(
            target,
            33,
            vec![(
                RuntimeGenerationRebuildSource::cold_ram(
                    Arc::<[u8]>::from(source.clone()),
                    0,
                    source.len() as u64,
                ),
                2,
            )],
        ))
        .expect("preparation")
        .enqueue()
        .complete()
        {
            RuntimeGenerationRebuildCompletion::Quiesced(Err(
                RuntimeGenerationRebuildError::DeviceRejected(6),
            )) => {}
            _ => panic!(
                "stable row IDs must be validated by the device, never derived from row offsets"
            ),
        }
    }

    #[test]
    fn gpu_rebuild_unknown_fence_retries_without_relaunching() {
        let Some(runtime) = cuda_runtime() else {
            return;
        };
        let target = runtime
            .runtime_generation_rebuild_target(0)
            .expect("primary target");
        let source = payload(&[(2, Some(11))], 3, 100, false);
        let submission = PreparedRuntimeGenerationRebuild::prepare(input(
            target,
            4,
            vec![(
                RuntimeGenerationRebuildSource::cold_ram(
                    Arc::<[u8]>::from(source.clone()),
                    0,
                    source.len() as u64,
                ),
                1,
            )],
        ))
        .expect("preparation")
        .enqueue();
        crate::cuda_context::fail_owned_stream_syncs_for_test(1);
        let unknown = match submission.complete() {
            RuntimeGenerationRebuildCompletion::UnknownQuiescence(unknown) => unknown,
            _ => panic!("injected fence failure must retain unknown quiescence"),
        };
        match unknown.retry_complete() {
            RuntimeGenerationRebuildCompletion::Quiesced(Ok(proof)) => {
                assert_eq!(proof.attempt().get(), 4)
            }
            RuntimeGenerationRebuildCompletion::UnknownQuiescence(unknown) => {
                panic!("retry fence remained unknown: {:?}", unknown.error())
            }
            RuntimeGenerationRebuildCompletion::Quiesced(Err(error)) => {
                panic!("retry fence quiesced a failed submission: {error:?}")
            }
        }
    }

    #[test]
    fn hazard_persistent_rebuild_fence_loss_remains_bounded_and_retains_owners() {
        let Some(runtime) = cuda_runtime() else {
            return;
        };
        let target = runtime
            .runtime_generation_rebuild_target(0)
            .expect("primary target");
        let source = payload(&[(2, Some(11))], 3, 100, false);
        let submission = PreparedRuntimeGenerationRebuild::prepare(input(
            target,
            5,
            vec![(
                RuntimeGenerationRebuildSource::cold_ram(
                    Arc::<[u8]>::from(source.clone()),
                    0,
                    source.len() as u64,
                ),
                1,
            )],
        ))
        .expect("preparation")
        .enqueue();
        // The first failure makes quiescence unknown; the second proves that a fence retry can
        // remain unknown. Higher layers must surface this owner to their bounded fresh-context
        // retry instead of polling forever.
        crate::cuda_context::fail_owned_stream_syncs_for_test(2);
        let unknown = match submission.complete() {
            RuntimeGenerationRebuildCompletion::UnknownQuiescence(unknown) => unknown,
            _ => panic!("first persistent fence failure must retain unknown quiescence"),
        };
        match unknown.retry_complete() {
            RuntimeGenerationRebuildCompletion::UnknownQuiescence(unknown) => {
                assert!(matches!(
                    unknown.error(),
                    RuntimeGenerationRebuildError::Runtime(_)
                ));
                // Dropping the retained owner takes its bounded drain/parking path.
                drop(unknown);
            }
            RuntimeGenerationRebuildCompletion::Quiesced(Ok(_)) => {
                panic!("second injected fence failure unexpectedly proved quiescence")
            }
            RuntimeGenerationRebuildCompletion::Quiesced(Err(error)) => {
                panic!("persistent fence loss quiesced as a terminal error: {error:?}")
            }
        }
    }

    #[test]
    fn hazard_fence_retry_is_safe_for_three_serial_and_two_concurrent_rebuilds() {
        let Some(runtime) = cuda_runtime() else {
            return;
        };
        let target = runtime
            .runtime_generation_rebuild_target(0)
            .expect("primary target");
        for attempt in 40..43 {
            let source = payload(&[(attempt, Some(11))], 3, 100, false);
            let submission = PreparedRuntimeGenerationRebuild::prepare(input(
                target.clone(),
                attempt,
                vec![(
                    RuntimeGenerationRebuildSource::cold_ram(
                        Arc::<[u8]>::from(source.clone()),
                        0,
                        source.len() as u64,
                    ),
                    1,
                )],
            ))
            .expect("serial preparation")
            .enqueue();
            crate::cuda_context::fail_owned_stream_syncs_for_test(1);
            let unknown = match submission.complete() {
                RuntimeGenerationRebuildCompletion::UnknownQuiescence(unknown) => unknown,
                _ => panic!("serial injected fence failure must retain resources"),
            };
            assert!(matches!(
                unknown.retry_complete(),
                RuntimeGenerationRebuildCompletion::Quiesced(Ok(_))
            ));
        }
        let joins = (0..2)
            .map(|offset| {
                let target = target.clone();
                std::thread::spawn(move || {
                    let attempt = 50 + offset;
                    let source = payload(&[(70 + offset, Some(11))], 3, 100, false);
                    let submission = PreparedRuntimeGenerationRebuild::prepare(input(
                        target,
                        attempt,
                        vec![(
                            RuntimeGenerationRebuildSource::cold_ram(
                                Arc::<[u8]>::from(source.clone()),
                                0,
                                source.len() as u64,
                            ),
                            1,
                        )],
                    ))
                    .expect("concurrent preparation")
                    .enqueue();
                    crate::cuda_context::fail_owned_stream_syncs_for_test(1);
                    match submission.complete() {
                        RuntimeGenerationRebuildCompletion::UnknownQuiescence(unknown) => {
                            matches!(
                                unknown.retry_complete(),
                                RuntimeGenerationRebuildCompletion::Quiesced(Ok(_))
                            )
                        }
                        _ => false,
                    }
                })
            })
            .collect::<Vec<_>>();
        for join in joins {
            assert!(join.join().expect("concurrent rebuild thread"));
        }
    }
}
