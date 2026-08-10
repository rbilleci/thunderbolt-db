//! Plural, type-neutral GPU generation completion for WRITE-001 typed INSERT.
//!
//! Preparation reserves every host/device owner from count-only geometry.  The sole launch edge
//! then copies a sealed neutral input into that already-owned arena, enqueues one kernel, and
//! retains all resources until quiescence is proven.  No SQL type has a separate transport or
//! launch path; storage tags and exact logical bytes are data in the closed descriptor.

mod abi;

use std::{
    ffi::c_void,
    fmt,
    num::NonZeroU64,
    sync::{Arc, Mutex, OnceLock},
};

use crate::{
    cuda_context::{PinnedHostBufferOwned, PooledDeviceBufferOwned, PooledStreamOwned},
    sha256::CuLaunchKernel,
    CudaRuntimeProbeError, GpuPrimaryContext,
};

const PTX: &[u8] = include_bytes!("runtime_typed_insert_generation/kernel.ptx");
// `cached_function` only needs a NUL-terminated PTX image when it first loads an entry module.
// WRITE-000 resolves those already-cached entries for every committed statement, so rebuilding
// this multi-megabyte byte vector at every reservation was pure host control-plane overhead.
// Keep the immutable driver input process-lifetime just like the context's module cache.
static PTX_WITH_NUL: OnceLock<Box<[u8]>> = OnceLock::new();
const ROOT_FORMAT_V1: u16 = 1;
const RETAINED_FAILURES: usize = 8;
// A 1k-row WRITE-001 batch has only a few thousand independently hashed cells.  Thirty-two
// threads deliberately creates enough generic grid-stride blocks to distribute those opaque
// typed cells across the RTX PRO 6000's 188 SMs; this is a work-item mapping, not a type path.
const PARALLEL_PHASE_THREADS: u32 = 32;
const REDUCTION_PHASE_THREADS: u32 = 512;
const FINALIZE_PHASE_THREADS: u32 = 64;
pub const RUNTIME_TYPED_INSERT_GENERATION_COMMITMENT_BYTES: usize = 160;
pub const RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH: usize = abi::RADIX_DEPTH;

#[derive(Clone)]
pub struct RuntimeTypedInsertGenerationTarget {
    primary: Arc<GpuPrimaryContext>,
    device_ordinal: u16,
    context_identity: usize,
}

impl RuntimeTypedInsertGenerationTarget {
    pub fn device_ordinal(&self) -> u16 {
        self.device_ordinal
    }
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

impl fmt::Debug for RuntimeTypedInsertGenerationTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeTypedInsertGenerationTarget")
            .field("device_ordinal", &self.device_ordinal)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeTypedInsertGenerationAttempt(NonZeroU64);

impl RuntimeTypedInsertGenerationAttempt {
    pub fn new(value: u64) -> Result<Self, RuntimeTypedInsertGenerationPrepareError> {
        NonZeroU64::new(value).map(Self).ok_or(
            RuntimeTypedInsertGenerationPrepareError::InvalidGeometry("zero generation attempt"),
        )
    }
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeTypedInsertGenerationGeometry {
    pub rows: usize,
    pub cells: usize,
    pub value_bytes: usize,
    pub indexes: usize,
    pub index_keys: usize,
    pub index_effects: usize,
    pub index_effect_components: usize,
}

#[derive(Clone, Copy)]
pub struct RuntimeTypedInsertGenerationIdentity {
    pub database_id: [u8; 16],
    pub catalog_epoch: u64,
    pub catalog_digest: [u8; 32],
    pub stable_transaction_id: u64,
    pub commit_sequence: u64,
    /// Exact codec-5 S2 statement digest for device-produced S7 transition closure. Generic
    /// generation builders and empty CREATE use the zero sentinel and cannot consume that output.
    pub write001_typed_statement_digest: [u8; 32],
}

/// The one table-delta action presently admitted by this execution vertical.  This is a sealed
/// device input, rather than an inference from the row count: an empty CREATE and an empty (and
/// invalid here) row set must never share a root/publication meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeTypedInsertGenerationTableAction {
    CreateEmpty,
    RowSetInsert,
    /// GPU-only zero-row index enrollment for ordered CREATE INDEX.  It changes the immutable
    /// table/database identities but owns no terminal, WAL, or publication transition.
    EnrollIndex,
    /// Replace the predecessor row/index contents with this exact row set while retaining the
    /// same stable table identity and pinned table-map predecessor.
    ResetThenRowSetInsert,
    /// One GPU generation for transactional CREATE TABLE followed by its initial INSERT rows.
    CreateWithRowSet,
    /// One GPU generation for a transaction-created named index on an already-published table.
    /// The zero index predecessor is admitted only by the codec-5/S3 proof; the device binds the
    /// index successor to the authenticated table prefix and the exact typed suffix.  This owns
    /// no catalog, WAL, physical index, or publication path.
    CreateIndexThenRowSetInsert,
}

impl RuntimeTypedInsertGenerationTableAction {
    fn encode(self) -> u8 {
        match self {
            Self::CreateEmpty => 1,
            Self::RowSetInsert => 2,
            Self::EnrollIndex => 3,
            Self::ResetThenRowSetInsert => 4,
            Self::CreateWithRowSet => 5,
            Self::CreateIndexThenRowSetInsert => 6,
        }
    }
}

/// The pinned immutable predecessor needed for one COW table-map leaf substitution.
///
/// `UninitializedEmptyDatabase` is only the first-CREATE bootstrap case. Its zero input is an
/// uninitialized sentinel, not a root: the GPU derives the canonical empty map/database roots
/// and then applies the absent-to-present leaf. `Pinned` reconstructs the complete predecessor
/// on-device from its root and 64 siblings. `PinnedRetained` is the same pinned predecessor
/// authority, with immutable roots copied from its prior GPU-authenticated publication so the
/// device only computes the new successor path.
// This is the fixed, inline 64-level COW witness supplied to the CUDA descriptor.  Boxing it
// would add a host allocation/indirection to the write hot path and make the descriptor no
// longer `Copy`; the sentinel and pinned forms deliberately share this ABI carrier.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeTypedInsertGenerationTableMapPredecessor {
    UninitializedEmptyDatabase,
    Pinned {
        initial_database_root: [u8; 32],
        sibling_roots: [[u8; 32]; RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH],
    },
    PinnedRetained {
        initial_database_root: [u8; 32],
        sibling_roots: [[u8; 32]; RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH],
        empty_roots: [[u8; 32]; RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH + 1],
        initial_leaf_root: [u8; 32],
        initial_path_roots: [[u8; 32]; RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH],
    },
}

impl RuntimeTypedInsertGenerationTableMapPredecessor {
    fn encode_kind(self) -> u8 {
        match self {
            Self::UninitializedEmptyDatabase => 1,
            Self::Pinned { .. } => 2,
            Self::PinnedRetained { .. } => 3,
        }
    }
}

#[derive(Clone, Copy)]
pub struct RuntimeTypedInsertGenerationTable {
    pub action: RuntimeTypedInsertGenerationTableAction,
    pub table_map_predecessor: RuntimeTypedInsertGenerationTableMapPredecessor,
    pub stable_table_id: u64,
    /// S7 final-image directory reference. Single-table aggregates use zero; plural transaction
    /// generation passes the stable table-order image slot through the same device operator.
    pub write001_final_image_ref: u32,
    /// The exact predecessor generation. It is zero only for an absent table predecessor
    /// (`CreateEmpty` or `CreateWithRowSet`); the latter carries its commit generation here so
    /// the device derives the first nonempty root without a host-created table generation.
    pub base_data_generation: u64,
    /// The exact predecessor table root. It is zero only for an absent table predecessor
    /// (`CreateEmpty` or `CreateWithRowSet`).
    pub base_table_root: [u8; 32],
    pub row_allocator_before: u64,
    pub row_allocator_high_water: u64,
    pub initial_logical_row_count: u64,
    pub final_logical_row_count: u64,
    pub image_layout_digest: [u8; 32],
    pub image_content_digest: [u8; 32],
}

#[derive(Clone, Copy)]
pub struct RuntimeTypedInsertGenerationRow {
    pub stable_table_id: u64,
    pub stable_row_id: u64,
    pub source_statement_ordinal: u32,
    pub source_row_ordinal: u32,
    pub cell_count: u32,
}

pub struct RuntimeTypedInsertGenerationCell<'a> {
    pub catalog_column_ordinal: u32,
    pub stable_column_id: u32,
    pub attnum: i16,
    pub storage: [u8; 4],
    pub declared_type_oid: u32,
    pub signed_type_size: i16,
    pub is_null: bool,
    pub value: &'a [u8],
}

/// One exact catalog-bound key binding in the bounded named-index route. The key ordinal is
/// semantic: `(status, tenant_id)` is distinct from `(tenant_id, status)` even when the stable
/// IDs happen to sort the other way.
#[derive(Clone, Copy)]
pub struct RuntimeTypedInsertGenerationIndexKeyColumn {
    pub key_ordinal: u32,
    pub catalog_column_ordinal: u32,
    pub stable_column_id: u32,
    pub attnum: i16,
    pub storage: [u8; 4],
    pub declared_type_oid: u32,
    pub signed_type_size: i16,
    /// Existing canonical catalog witness digest; callers pass it through, never hash names in
    /// this host control-plane adapter.
    pub column_name_digest: [u8; 32],
}

/// Ordered predecessor identity and shape for one maintained index. The runtime currently admits
/// exactly one index with 1..32 type-neutral ordered keys. UNIQUE/PRIMARY flags describe that same
/// generation after their transaction verdict has closed; plural descriptors remain deferred.
#[derive(Clone, Copy)]
pub struct RuntimeTypedInsertGenerationIndex {
    pub stable_index_id: u64,
    pub raw_catalog_index_ordinal: u32,
    pub index_flags: u32,
    pub null_equality_policy: u8,
    pub base_generation: u64,
    pub base_root: [u8; 32],
    pub key_start: u32,
    pub key_count: u32,
    pub effect_start: u32,
    pub effect_count: u32,
}

/// One source binding for a typed index-effect component. The value and its type are resolved
/// from the already-sealed row cell on the GPU, so this descriptor cannot carry a second host
/// value vector or a caller-supplied digest.
#[derive(Clone, Copy)]
pub struct RuntimeTypedInsertGenerationIndexEffectComponent {
    pub catalog_column_ordinal: u32,
    pub stable_column_id: u32,
}

/// The one physical maintenance effect required for every `(index,row)` pair.  Components must
/// appear in the index's declared key order; the GPU rejects reordered, foreign-row, duplicate,
/// missing, or shape-mismatched effects.
#[derive(Clone, Copy)]
pub struct RuntimeTypedInsertGenerationIndexEffect {
    pub stable_table_id: u64,
    pub stable_index_id: u64,
    pub stable_row_id: u64,
    pub source_catalog_ordinal: u32,
    pub component_start: u32,
    pub component_count: u32,
}

/// One named index predecessor/successor tuple produced by a completed device generation. This
/// is the only public index-root handoff: there is deliberately no raw slot/digest accessor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeTypedInsertGenerationIndexRoot {
    pub stable_index_id: u64,
    pub initial_generation: u64,
    pub initial_root: [u8; 32],
    pub final_generation: u64,
    pub final_root: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeTypedInsertGenerationPrepareError {
    Runtime(CudaRuntimeProbeError),
    AsyncTransportUnavailable,
    PinnedHostStagingUnavailable,
    InvalidGeometry(&'static str),
}

impl From<CudaRuntimeProbeError> for RuntimeTypedInsertGenerationPrepareError {
    fn from(value: CudaRuntimeProbeError) -> Self {
        Self::Runtime(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeTypedInsertGenerationError {
    Runtime(CudaRuntimeProbeError),
    InvalidInput(&'static str),
    DeviceRejected(u32),
}

impl From<CudaRuntimeProbeError> for RuntimeTypedInsertGenerationError {
    fn from(value: CudaRuntimeProbeError) -> Self {
        Self::Runtime(value)
    }
}

impl fmt::Display for RuntimeTypedInsertGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(f),
            Self::InvalidInput(message) => write!(f, "invalid typed generation input: {message}"),
            Self::DeviceRejected(status) => {
                write!(
                    f,
                    "typed generation device validation rejected status {status}"
                )
            }
        }
    }
}

#[must_use = "a reserved typed generation must be launched or intentionally dropped"]
pub struct PreparedRuntimeTypedInsertGeneration {
    primary: Arc<GpuPrimaryContext>,
    resources: RuntimeTypedInsertGenerationResources,
    attempt: RuntimeTypedInsertGenerationAttempt,
    geometry: RuntimeTypedInsertGenerationGeometry,
    layout: abi::OutputLayout,
    kernels: RuntimeTypedInsertGenerationKernels,
    launch: CuLaunchKernel,
}

#[must_use = "an in-flight typed generation must be completed or safely dropped"]
pub struct RuntimeTypedInsertGenerationSubmission {
    primary: Arc<GpuPrimaryContext>,
    resources: Option<RuntimeTypedInsertGenerationResources>,
    attempt: RuntimeTypedInsertGenerationAttempt,
    geometry: RuntimeTypedInsertGenerationGeometry,
    layout: abi::OutputLayout,
    kernels: RuntimeTypedInsertGenerationKernels,
    launch: CuLaunchKernel,
    phase: Phase,
    first_error: Option<RuntimeTypedInsertGenerationError>,
    driver_error_observed: bool,
    /// Build-only CUDA-event timing covers the five generation kernels only.  It deliberately
    /// excludes descriptor construction and both DMA directions, which remain part of the
    /// engine's host-wall generation seam.
    #[cfg(feature = "probe-timing")]
    kernel_event_recorded: bool,
    #[cfg(feature = "probe-timing")]
    kernel_phase_events_recorded: bool,
}

#[must_use = "unknown typed generation quiescence must be retried or deliberately dropped to park/quarantine its resources"]
pub struct RuntimeTypedInsertGenerationUnknownQuiescence {
    error: RuntimeTypedInsertGenerationError,
    submission: Option<RuntimeTypedInsertGenerationSubmission>,
}

#[must_use = "typed generation completion must be handled"]
#[allow(
    clippy::large_enum_variant,
    reason = "boxing would allocate on the exceptional CUDA path"
)]
pub enum RuntimeTypedInsertGenerationCompletion {
    Quiesced(Result<OpaqueRuntimeTypedInsertGenerationProof, RuntimeTypedInsertGenerationError>),
    UnknownQuiescence(RuntimeTypedInsertGenerationUnknownQuiescence),
}

struct RuntimeTypedInsertGenerationResources {
    input_host: PinnedHostBufferOwned,
    input_device: PooledDeviceBufferOwned,
    _workspace: PooledDeviceBufferOwned,
    output_host: PinnedHostBufferOwned,
    output_device: PooledDeviceBufferOwned,
    stream: PooledStreamOwned,
    input_bytes: usize,
    output_bytes: usize,
    first_row_id: u64,
    table_id: u64,
    commit_sequence: u64,
    final_row_count: u64,
    write001_typed_statement_digest: [u8; 32],
    /// Ordered input identities retained beside the device buffers until completion.  This is
    /// host control-plane metadata only; every root in the paired tuple comes from the device
    /// output arena.
    indexes: Box<[Option<EncodedIndex>]>,
}

#[derive(Clone, Copy)]
struct RuntimeTypedInsertGenerationKernels {
    validate: *mut c_void,
    cells: *mut c_void,
    rows: *mut c_void,
    reduce: *mut c_void,
    finalize_indexed: *mut c_void,
    finalize_unindexed: *mut c_void,
}

enum Phase {
    Reserved,
    InFlight,
    ReadbackQueued,
    TerminalFailure { drain_required: bool },
    Quiesced,
}

impl Phase {
    fn drain_required(&self) -> bool {
        matches!(
            self,
            Self::InFlight
                | Self::ReadbackQueued
                | Self::TerminalFailure {
                    drain_required: true
                }
        )
    }
}

unsafe impl Send for PreparedRuntimeTypedInsertGeneration {}
unsafe impl Send for RuntimeTypedInsertGenerationSubmission {}
unsafe impl Send for RuntimeTypedInsertGenerationUnknownQuiescence {}
unsafe impl Send for RuntimeTypedInsertGenerationResources {}

/// Fixed domain-specific commitments.  There is no generic digest, slice, or byte accessor.
pub struct RuntimeTypedInsertGenerationCommitments {
    generation_input: [u8; 32],
    initial_table_root: [u8; 32],
    final_table_root: [u8; 32],
    initial_database_root: [u8; 32],
    final_database_root: [u8; 32],
}

impl RuntimeTypedInsertGenerationCommitments {
    pub fn encode_durable_into(&self, destination: &mut [u8]) -> Result<(), usize> {
        if destination.len() != RUNTIME_TYPED_INSERT_GENERATION_COMMITMENT_BYTES {
            return Err(destination.len());
        }
        destination[..32].copy_from_slice(&self.generation_input);
        destination[32..64].copy_from_slice(&self.initial_table_root);
        destination[64..96].copy_from_slice(&self.final_table_root);
        destination[96..128].copy_from_slice(&self.initial_database_root);
        destination[128..].copy_from_slice(&self.final_database_root);
        Ok(())
    }

    pub fn copy_generation_input_into(&self, destination: &mut [u8; 32]) {
        *destination = self.generation_input;
    }
    pub fn copy_initial_table_root_into(&self, destination: &mut [u8; 32]) {
        *destination = self.initial_table_root;
    }
    pub fn copy_final_table_root_into(&self, destination: &mut [u8; 32]) {
        *destination = self.final_table_root;
    }
    pub fn copy_initial_database_root_into(&self, destination: &mut [u8; 32]) {
        *destination = self.initial_database_root;
    }
    pub fn copy_final_database_root_into(&self, destination: &mut [u8; 32]) {
        *destination = self.final_database_root;
    }
}

impl fmt::Debug for RuntimeTypedInsertGenerationCommitments {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RuntimeTypedInsertGenerationCommitments(<opaque>)")
    }
}

/// The fixed table-map completion arena did not match the descriptor layout expected by the
/// narrow publication handoff.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeTypedInsertGenerationTableMapCompletionError;

pub struct RuntimeTypedInsertGenerationLogicalCompletion {
    /// One move-only readback owner. Named accessors project the few production consumers from
    /// this arena without copying every digest into a second family of boxed arrays.
    slots: Box<[[u8; 32]]>,
    layout: abi::OutputLayout,
    rows: usize,
    column_count: usize,
    write001_typed_statement_digest: [u8; 32],
    first_row_id: u64,
    commit_sequence: u64,
    indexes: Box<[Option<EncodedIndex>]>,
}

impl RuntimeTypedInsertGenerationLogicalCompletion {
    /// Copy the ordered, device-produced catalog column-shape roots into one exact caller-owned
    /// directory. This is a CREATE/publication handoff with fixed cardinality and order, not a
    /// generic digest accessor.
    pub fn copy_column_roots_into(
        &self,
        shape_destination: &mut [[u8; 32]],
        column_destination: &mut [[u8; 32]],
    ) -> Result<(), usize> {
        if shape_destination.len() != self.column_count
            || column_destination.len() != self.column_count
        {
            return Err(shape_destination.len());
        }
        shape_destination.copy_from_slice(
            &self.slots[self.layout.shape_roots..self.layout.shape_roots + self.column_count],
        );
        column_destination.copy_from_slice(
            &self.slots[self.layout.column_roots..self.layout.column_roots + self.column_count],
        );
        Ok(())
    }

    /// Copy the device-authenticated persistent table-map witnesses for the exact one-leaf
    /// predecessor/successor transition.  This is the narrow manifest handoff consumed by the
    /// publication owner; it is not a generic digest escape hatch.
    pub fn copy_table_map_transition_into(
        &self,
        empty_destination: &mut [[u8; 32]; RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH + 1],
        initial_leaf_destination: &mut [u8; 32],
        initial_path_destination: &mut [[u8; 32]; RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH],
        final_leaf_destination: &mut [u8; 32],
        final_path_destination: &mut [[u8; 32]; RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH],
    ) -> Result<(), RuntimeTypedInsertGenerationTableMapCompletionError> {
        let empty = self.layout.table_map_empty_slot();
        let initial_leaf = self.layout.initial_table_map_leaf_slot();
        let initial_path = self.layout.initial_table_map_path_slot();
        let final_leaf = self.layout.final_table_map_leaf_slot();
        let final_path = self.layout.final_table_map_path_slot();
        let end = self.layout.index_final_roots_slot() + self.indexes.len();
        if end != self.layout.digest_count || self.slots.len() != self.layout.digest_count {
            return Err(RuntimeTypedInsertGenerationTableMapCompletionError);
        }
        empty_destination.copy_from_slice(
            &self.slots[empty..empty + RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH + 1],
        );
        *initial_leaf_destination = self.slots[initial_leaf];
        initial_path_destination.copy_from_slice(
            &self.slots
                [initial_path..initial_path + RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH],
        );
        *final_leaf_destination = self.slots[final_leaf];
        final_path_destination.copy_from_slice(
            &self.slots[final_path..final_path + RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH],
        );
        Ok(())
    }

    /// Copy the exact ordered predecessor/successor index tuples produced by the device. The
    /// array is deliberately caller-owned and fixed to the descriptor's raw catalog order; this
    /// is not a generic root/digest reader.
    pub fn copy_index_generation_roots_into(
        &self,
        destination: &mut [RuntimeTypedInsertGenerationIndexRoot],
    ) -> Result<(), usize> {
        if destination.len() != self.indexes.len() {
            return Err(destination.len());
        }
        for (ordinal, (destination, index)) in
            destination.iter_mut().zip(self.indexes.iter()).enumerate()
        {
            let Some(index) = index else {
                return Err(ordinal);
            };
            *destination = RuntimeTypedInsertGenerationIndexRoot {
                stable_index_id: index.stable_index_id,
                initial_generation: index.base_generation,
                initial_root: self.slots[self.layout.index_initial_roots_slot() + ordinal],
                final_generation: self.commit_sequence,
                final_root: self.slots[self.layout.index_final_roots_slot() + ordinal],
            };
        }
        Ok(())
    }

    /// Copy one exact, device-produced S7 final-row digest into the live codec-5 writer's fixed
    /// transition slot. This is deliberately named for that sole grammar; it is not a generic
    /// digest accessor or a host-side rehash authority.
    pub fn copy_write001_s7_transition_digests_into(
        &self,
        row_ordinal: usize,
        stable_row_id: u64,
        typed_statement_digest: [u8; 32],
        final_row_destination: &mut [u8; 32],
        transition_destination: &mut [u8; 32],
    ) -> Result<(), usize> {
        let expected_row_id = self.first_row_id.checked_add(row_ordinal as u64);
        if row_ordinal >= self.rows {
            return Err(row_ordinal);
        }
        // The zero-index route owns the frozen complete S7 transition grammar. Indexed S7
        // descriptors/effects/components are canonical WAL control-plane bytes, so their
        // transition digest is deliberately not exposed from this root-generation runtime.
        // Returning an error prevents a caller from treating the device's final-row-only
        // intermediate as a complete indexed S7 transition.
        if !self.indexes.is_empty() {
            return Err(row_ordinal);
        }
        if expected_row_id != Some(stable_row_id)
            || typed_statement_digest != self.write001_typed_statement_digest
            || typed_statement_digest == [0; 32]
        {
            return Err(row_ordinal);
        }
        *final_row_destination = self.slots[self.layout.s7_final_row_digests + row_ordinal];
        *transition_destination = self.slots[self.layout.s7_transition_digests + row_ordinal];
        Ok(())
    }

    /// Copy only the device-produced S7 final-row digest.  Indexed codec-5 closure serializes
    /// its effect directory on the WAL control plane, so its transition digest has a different
    /// (and strictly richer) grammar than the unindexed device transition slot.  This remains a
    /// narrow final-row identity handoff, not a generic digest accessor.
    pub fn copy_write001_s7_final_row_digest_into(
        &self,
        row_ordinal: usize,
        stable_row_id: u64,
        typed_statement_digest: [u8; 32],
        final_row_destination: &mut [u8; 32],
    ) -> Result<(), usize> {
        let expected_row_id = self.first_row_id.checked_add(row_ordinal as u64);
        if row_ordinal >= self.rows
            || expected_row_id != Some(stable_row_id)
            || typed_statement_digest != self.write001_typed_statement_digest
            || typed_statement_digest == [0; 32]
        {
            return Err(row_ordinal);
        }
        *final_row_destination = self.slots[self.layout.s7_final_row_digests + row_ordinal];
        Ok(())
    }

    /// Copy the device-produced final-row commitment for a transition whose statement identity
    /// is serialized from the retained S1/S2/S4 directory. This is the multi-statement sibling
    /// of the legacy single-digest handoff above; it exposes no transition digest or root slot.
    pub fn copy_write001_s7_final_row_digest_for_serialized_transition_into(
        &self,
        row_ordinal: usize,
        stable_row_id: u64,
        final_row_destination: &mut [u8; 32],
    ) -> Result<(), usize> {
        let expected_row_id = self.first_row_id.checked_add(row_ordinal as u64);
        if row_ordinal >= self.rows || expected_row_id != Some(stable_row_id) {
            return Err(row_ordinal);
        }
        *final_row_destination = self.slots[self.layout.s7_final_row_digests + row_ordinal];
        Ok(())
    }
}

pub struct OpaqueRuntimeTypedInsertGenerationProof {
    attempt: RuntimeTypedInsertGenerationAttempt,
    slots: Box<[[u8; 32]]>,
    layout: abi::OutputLayout,
    first_row_id: u64,
    rows: usize,
    table_id: u64,
    commit_sequence: u64,
    final_row_count: u64,
    write001_typed_statement_digest: [u8; 32],
    indexes: Box<[Option<EncodedIndex>]>,
    #[cfg(feature = "probe-timing")]
    kernel_event_elapsed_nanos: Option<u64>,
    #[cfg(feature = "probe-timing")]
    kernel_phase_event_nanos: Option<[u64; 5]>,
}

impl OpaqueRuntimeTypedInsertGenerationProof {
    pub fn attempt(&self) -> RuntimeTypedInsertGenerationAttempt {
        self.attempt
    }
    pub fn row_count(&self) -> usize {
        self.rows
    }
    pub fn stable_table_id(&self) -> u64 {
        self.table_id
    }
    pub fn final_data_generation(&self) -> u64 {
        self.commit_sequence
    }
    pub fn final_logical_row_count(&self) -> u64 {
        self.final_row_count
    }

    /// CUDA-event duration for the kernel-only portion of this exact generation.  This exists
    /// solely in probe builds: the production completion contract carries no timing state.
    #[cfg(feature = "probe-timing")]
    pub fn kernel_event_elapsed_nanos(&self) -> Option<u64> {
        self.kernel_event_elapsed_nanos
    }

    /// Ordered CUDA-event spans for validation, cells, rows, reduction, and finalization.  This
    /// is probe-only diagnostic data; no write semantic or completion authority depends on it.
    #[cfg(feature = "probe-timing")]
    pub fn kernel_phase_event_nanos(&self) -> Option<[u64; 5]> {
        self.kernel_phase_event_nanos
    }

    pub fn consume<R>(
        self,
        consume: impl FnOnce(
            RuntimeTypedInsertGenerationAttempt,
            RuntimeTypedInsertGenerationCommitments,
            RuntimeTypedInsertGenerationLogicalCompletion,
        ) -> R,
    ) -> R {
        let commitments = RuntimeTypedInsertGenerationCommitments {
            generation_input: self.slots[abi::SLOT_GENERATION_INPUT],
            initial_table_root: self.slots[abi::SLOT_INITIAL_TABLE_ROOT],
            final_table_root: self.slots[abi::SLOT_FINAL_TABLE_ROOT],
            initial_database_root: self.slots[abi::SLOT_INITIAL_DATABASE_ROOT],
            final_database_root: self.slots[abi::SLOT_FINAL_DATABASE_ROOT],
        };
        let column_count = (self.layout.typed_roots - self.layout.shape_roots)
            .checked_div(self.rows)
            .unwrap_or(self.layout.typed_roots - self.layout.shape_roots);
        let logical = RuntimeTypedInsertGenerationLogicalCompletion {
            slots: self.slots,
            layout: self.layout,
            rows: self.rows,
            column_count,
            write001_typed_statement_digest: self.write001_typed_statement_digest,
            first_row_id: self.first_row_id,
            commit_sequence: self.commit_sequence,
            indexes: self.indexes,
        };
        consume(self.attempt, commitments, logical)
    }
}

impl fmt::Debug for OpaqueRuntimeTypedInsertGenerationProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpaqueRuntimeTypedInsertGenerationProof")
            .field("attempt", &self.attempt)
            .field("rows", &self.rows)
            .field("stable_table_id", &self.table_id)
            .finish_non_exhaustive()
    }
}

fn compact_active_level_counts(_first: u64, rows: usize) -> [usize; abi::RADIX_DEPTH] {
    let mut active = rows;
    std::array::from_fn(|_| {
        if active <= 1 {
            0
        } else {
            active = active.div_ceil(2);
            active
        }
    })
}

impl PreparedRuntimeTypedInsertGeneration {
    pub fn reserve(
        target: RuntimeTypedInsertGenerationTarget,
        attempt: RuntimeTypedInsertGenerationAttempt,
        geometry: RuntimeTypedInsertGenerationGeometry,
    ) -> Result<Self, RuntimeTypedInsertGenerationPrepareError> {
        validate_geometry(geometry)?;
        let primary = Arc::clone(&target.primary);
        if primary.cu_memcpy_htod_async.is_none() || primary.cu_memcpy_dtoh_async.is_none() {
            return Err(RuntimeTypedInsertGenerationPrepareError::AsyncTransportUnavailable);
        }
        primary.set_current()?;
        let layout = abi::OutputLayout::new(geometry.rows, geometry.cells, geometry.indexes)
            .ok_or(RuntimeTypedInsertGenerationPrepareError::InvalidGeometry(
                "output capacity",
            ))?;
        if layout.digest_count > u32::MAX as usize {
            return Err(RuntimeTypedInsertGenerationPrepareError::InvalidGeometry(
                "output slot addressability",
            ));
        }
        let input_bytes = abi::input_bytes(
            geometry.rows,
            geometry.cells,
            geometry.value_bytes,
            geometry.indexes,
            geometry.index_keys,
            geometry.index_effects,
            geometry.index_effect_components,
        )
        .ok_or(RuntimeTypedInsertGenerationPrepareError::InvalidGeometry(
            "input capacity",
        ))?;
        let workspace_bytes = abi::workspace_bytes(
            geometry.rows,
            geometry.cells,
            geometry.value_bytes,
            geometry.indexes,
            geometry.index_keys,
            geometry.index_effects,
            geometry.index_effect_components,
        )
        .ok_or(RuntimeTypedInsertGenerationPrepareError::InvalidGeometry(
            "workspace capacity",
        ))?;
        let output_bytes = layout.output_bytes().ok_or(
            RuntimeTypedInsertGenerationPrepareError::InvalidGeometry("output capacity"),
        )?;
        let input_device = primary.lease_device_buffer_owned(input_bytes)?;
        let workspace = primary.lease_device_buffer_owned(workspace_bytes)?;
        let output_device = primary.lease_device_buffer_owned(output_bytes)?;
        if input_device.ptr == workspace.ptr
            || input_device.ptr == output_device.ptr
            || workspace.ptr == output_device.ptr
        {
            return Err(RuntimeTypedInsertGenerationPrepareError::InvalidGeometry(
                "distinct device buffers",
            ));
        }
        let mut input_host = primary
            .lease_pinned_host_buffer_owned(input_bytes)
            .ok_or(RuntimeTypedInsertGenerationPrepareError::PinnedHostStagingUnavailable)?;
        input_host.as_mut_bytes(input_bytes)?.fill(0);
        let output_host = primary
            .lease_pinned_host_buffer_owned(output_bytes)
            .ok_or(RuntimeTypedInsertGenerationPrepareError::PinnedHostStagingUnavailable)?;
        let stream = PooledStreamOwned {
            primary: Arc::clone(&primary),
            pooled: Some(primary.acquire_pooled_stream()?),
        };
        let (kernels, launch) = resolve_launch(&primary)?;
        let resources = RuntimeTypedInsertGenerationResources {
            input_host,
            input_device,
            _workspace: workspace,
            output_host,
            output_device,
            stream,
            input_bytes,
            output_bytes,
            first_row_id: 0,
            table_id: 0,
            commit_sequence: 0,
            final_row_count: 0,
            write001_typed_statement_digest: [0; 32],
            indexes: vec![None; geometry.indexes].into_boxed_slice(),
        };
        Ok(Self {
            primary,
            resources,
            attempt,
            geometry,
            layout,
            kernels,
            launch,
        })
    }

    /// Total launch transition. Descriptor mistakes become a quiesced input error; they never
    /// create a fallback, allocate after reservation, or partly submit another route.
    pub fn launch(
        mut self,
        fill: impl FnOnce(&mut RuntimeTypedInsertGenerationEncoder<'_>),
    ) -> RuntimeTypedInsertGenerationSubmission {
        let workspace_pointer = self.resources._workspace.ptr;
        let encoding = {
            let resources = &mut self.resources;
            resources.indexes.fill(None);
            let bytes = resources
                .input_host
                .as_mut_bytes(resources.input_bytes)
                .expect("reserved pinned capacity remains exact");
            bytes.fill(0);
            abi::put_u16(bytes, abi::ROOT_FORMAT_OFFSET, ROOT_FORMAT_V1);
            abi::put_u32(bytes, abi::TABLE_COUNT_OFFSET, 1);
            abi::put_u32(bytes, abi::ROW_COUNT_OFFSET, self.geometry.rows as u32);
            abi::put_u32(bytes, abi::CELL_COUNT_OFFSET, self.geometry.cells as u32);
            abi::put_u32(
                bytes,
                abi::VALUE_BYTES_OFFSET,
                self.geometry.value_bytes as u32,
            );
            abi::put_u32(bytes, abi::INDEX_COUNT_OFFSET, self.geometry.indexes as u32);
            abi::put_u32(
                bytes,
                abi::HEADER_INDEX_KEY_COUNT_OFFSET,
                self.geometry.index_keys as u32,
            );
            abi::put_u32(
                bytes,
                abi::HEADER_INDEX_EFFECT_COUNT_OFFSET,
                self.geometry.index_effects as u32,
            );
            abi::put_u32(
                bytes,
                abi::HEADER_INDEX_EFFECT_COMPONENT_COUNT_OFFSET,
                self.geometry.index_effect_components as u32,
            );
            abi::put_u64(bytes, abi::TABLE_OFFSET_OFFSET, abi::HEADER_BYTES as u64);
            abi::put_u64(
                bytes,
                abi::ROW_OFFSET_OFFSET,
                (abi::HEADER_BYTES + abi::TABLE_BYTES) as u64,
            );
            abi::put_u64(
                bytes,
                abi::CELL_OFFSET_OFFSET,
                (abi::HEADER_BYTES + abi::TABLE_BYTES + self.geometry.rows * abi::ROW_BYTES) as u64,
            );
            abi::put_u64(
                bytes,
                abi::VALUE_OFFSET_OFFSET,
                (abi::HEADER_BYTES
                    + abi::TABLE_BYTES
                    + self.geometry.rows * abi::ROW_BYTES
                    + self.geometry.cells * abi::CELL_BYTES) as u64,
            );
            let index_offset = abi::HEADER_BYTES
                + abi::TABLE_BYTES
                + self.geometry.rows * abi::ROW_BYTES
                + self.geometry.cells * abi::CELL_BYTES
                + self.geometry.value_bytes;
            abi::put_u64(bytes, abi::INDEX_OFFSET_OFFSET, index_offset as u64);
            abi::put_u64(
                bytes,
                abi::INDEX_KEY_OFFSET_OFFSET,
                (index_offset + self.geometry.indexes * abi::INDEX_BYTES) as u64,
            );
            abi::put_u64(
                bytes,
                abi::INDEX_EFFECT_OFFSET_OFFSET,
                (index_offset
                    + self.geometry.indexes * abi::INDEX_BYTES
                    + self.geometry.index_keys * abi::INDEX_KEY_BYTES) as u64,
            );
            abi::put_u64(
                bytes,
                abi::INDEX_EFFECT_COMPONENT_OFFSET_OFFSET,
                (index_offset
                    + self.geometry.indexes * abi::INDEX_BYTES
                    + self.geometry.index_keys * abi::INDEX_KEY_BYTES
                    + self.geometry.index_effects * abi::INDEX_EFFECT_BYTES) as u64,
            );
            abi::put_u64(bytes, abi::WORKSPACE_POINTER_OFFSET, workspace_pointer);
            let mut encoder = RuntimeTypedInsertGenerationEncoder::new(
                bytes,
                self.geometry,
                &mut resources.indexes,
            );
            fill(&mut encoder);
            encoder.finish()
        };
        if let Ok(summary) = encoding {
            self.resources.first_row_id = summary.first_row_id;
            self.resources.table_id = summary.table_id;
            self.resources.commit_sequence = summary.commit_sequence;
            self.resources.final_row_count = summary.final_row_count;
            self.resources.write001_typed_statement_digest =
                summary.write001_typed_statement_digest;
        }
        let mut submission = RuntimeTypedInsertGenerationSubmission {
            primary: self.primary,
            resources: Some(self.resources),
            attempt: self.attempt,
            geometry: self.geometry,
            layout: self.layout,
            kernels: self.kernels,
            launch: self.launch,
            phase: Phase::Reserved,
            first_error: encoding
                .err()
                .map(RuntimeTypedInsertGenerationError::InvalidInput),
            driver_error_observed: false,
            #[cfg(feature = "probe-timing")]
            kernel_event_recorded: false,
            #[cfg(feature = "probe-timing")]
            kernel_phase_events_recorded: false,
        };
        submission.enqueue_inner();
        submission
    }
}

#[derive(Clone, Copy)]
struct EncodedIndex {
    stable_index_id: u64,
    base_generation: u64,
}

#[derive(Clone, Copy)]
struct EncodedSummary {
    first_row_id: u64,
    table_id: u64,
    commit_sequence: u64,
    final_row_count: u64,
    write001_typed_statement_digest: [u8; 32],
}

pub struct RuntimeTypedInsertGenerationEncoder<'a> {
    bytes: &'a mut [u8],
    geometry: RuntimeTypedInsertGenerationGeometry,
    identity_written: bool,
    table_written: bool,
    rows: usize,
    cells: usize,
    values: usize,
    expected_cells: usize,
    first_row_id: u64,
    table_id: u64,
    commit_sequence: u64,
    final_row_count: u64,
    action: Option<RuntimeTypedInsertGenerationTableAction>,
    indexes: usize,
    index_keys: usize,
    index_effects: usize,
    index_effect_components: usize,
    encoded_indexes: &'a mut [Option<EncodedIndex>],
    error: Option<&'static str>,
}

impl<'a> RuntimeTypedInsertGenerationEncoder<'a> {
    fn new(
        bytes: &'a mut [u8],
        geometry: RuntimeTypedInsertGenerationGeometry,
        encoded_indexes: &'a mut [Option<EncodedIndex>],
    ) -> Self {
        Self {
            bytes,
            geometry,
            identity_written: false,
            table_written: false,
            rows: 0,
            cells: 0,
            values: 0,
            expected_cells: 0,
            first_row_id: 0,
            table_id: 0,
            commit_sequence: 0,
            final_row_count: 0,
            action: None,
            indexes: 0,
            index_keys: 0,
            index_effects: 0,
            index_effect_components: 0,
            encoded_indexes,
            error: None,
        }
    }

    fn reject(&mut self, error: &'static str) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    pub fn write_identity(&mut self, value: RuntimeTypedInsertGenerationIdentity) {
        if self.identity_written {
            self.reject("duplicate generation identity");
            return;
        }
        self.identity_written = true;
        self.bytes[abi::DATABASE_ID_OFFSET..abi::DATABASE_ID_OFFSET + 16]
            .copy_from_slice(&value.database_id);
        abi::put_u64(self.bytes, abi::CATALOG_EPOCH_OFFSET, value.catalog_epoch);
        self.bytes[abi::CATALOG_DIGEST_OFFSET..abi::CATALOG_DIGEST_OFFSET + 32]
            .copy_from_slice(&value.catalog_digest);
        abi::put_u64(
            self.bytes,
            abi::STABLE_TRANSACTION_ID_OFFSET,
            value.stable_transaction_id,
        );
        abi::put_u64(
            self.bytes,
            abi::COMMIT_SEQUENCE_OFFSET,
            value.commit_sequence,
        );
        self.bytes[abi::WRITE001_TYPED_STATEMENT_DIGEST_OFFSET
            ..abi::WRITE001_TYPED_STATEMENT_DIGEST_OFFSET + 32]
            .copy_from_slice(&value.write001_typed_statement_digest);
        self.commit_sequence = value.commit_sequence;
    }

    pub fn write_table(&mut self, value: RuntimeTypedInsertGenerationTable) {
        if self.table_written {
            self.reject("duplicate generation table");
            return;
        }
        self.table_written = true;
        self.bytes[abi::TABLE_ACTION_OFFSET] = value.action.encode();
        self.bytes[abi::TABLE_MAP_PREDECESSOR_OFFSET] = value.table_map_predecessor.encode_kind();
        match value.table_map_predecessor {
            RuntimeTypedInsertGenerationTableMapPredecessor::UninitializedEmptyDatabase => {}
            RuntimeTypedInsertGenerationTableMapPredecessor::Pinned {
                initial_database_root,
                sibling_roots,
            } => {
                self.bytes
                    [abi::INITIAL_DATABASE_ROOT_OFFSET..abi::INITIAL_DATABASE_ROOT_OFFSET + 32]
                    .copy_from_slice(&initial_database_root);
                let sibling_start = abi::HEADER_BYTES + abi::TABLE_MAP_SIBLINGS_OFFSET;
                for (depth, sibling) in sibling_roots.iter().enumerate() {
                    let at = sibling_start + depth * abi::DIGEST_BYTES;
                    self.bytes[at..at + abi::DIGEST_BYTES].copy_from_slice(sibling);
                }
            }
            RuntimeTypedInsertGenerationTableMapPredecessor::PinnedRetained {
                initial_database_root,
                sibling_roots,
                empty_roots,
                initial_leaf_root,
                initial_path_roots,
            } => {
                self.bytes
                    [abi::INITIAL_DATABASE_ROOT_OFFSET..abi::INITIAL_DATABASE_ROOT_OFFSET + 32]
                    .copy_from_slice(&initial_database_root);
                let sibling_start = abi::HEADER_BYTES + abi::TABLE_MAP_SIBLINGS_OFFSET;
                for (depth, sibling) in sibling_roots.iter().enumerate() {
                    let at = sibling_start + depth * abi::DIGEST_BYTES;
                    self.bytes[at..at + abi::DIGEST_BYTES].copy_from_slice(sibling);
                }
                let empty_start = abi::HEADER_BYTES + abi::TABLE_MAP_RETAINED_EMPTY_ROOTS_OFFSET;
                for (depth, root) in empty_roots.iter().enumerate() {
                    let at = empty_start + depth * abi::DIGEST_BYTES;
                    self.bytes[at..at + abi::DIGEST_BYTES].copy_from_slice(root);
                }
                let leaf_start = abi::HEADER_BYTES + abi::TABLE_MAP_RETAINED_INITIAL_LEAF_OFFSET;
                self.bytes[leaf_start..leaf_start + abi::DIGEST_BYTES]
                    .copy_from_slice(&initial_leaf_root);
                let path_start = abi::HEADER_BYTES + abi::TABLE_MAP_RETAINED_INITIAL_PATH_OFFSET;
                for (depth, root) in initial_path_roots.iter().enumerate() {
                    let at = path_start + depth * abi::DIGEST_BYTES;
                    self.bytes[at..at + abi::DIGEST_BYTES].copy_from_slice(root);
                }
            }
        }
        let at = abi::HEADER_BYTES;
        abi::put_u64(self.bytes, at + abi::TABLE_ID_OFFSET, value.stable_table_id);
        abi::put_u32(
            self.bytes,
            abi::WRITE001_FINAL_IMAGE_REF_OFFSET,
            value.write001_final_image_ref,
        );
        abi::put_u64(
            self.bytes,
            at + abi::TABLE_BASE_GENERATION_OFFSET,
            value.base_data_generation,
        );
        self.bytes[at + abi::TABLE_BASE_ROOT_OFFSET..at + abi::TABLE_BASE_ROOT_OFFSET + 32]
            .copy_from_slice(&value.base_table_root);
        abi::put_u64(
            self.bytes,
            at + abi::TABLE_ALLOCATOR_BEFORE_OFFSET,
            value.row_allocator_before,
        );
        abi::put_u64(
            self.bytes,
            at + abi::TABLE_ALLOCATOR_HIGH_WATER_OFFSET,
            value.row_allocator_high_water,
        );
        abi::put_u64(
            self.bytes,
            at + abi::TABLE_INITIAL_ROW_COUNT_OFFSET,
            value.initial_logical_row_count,
        );
        abi::put_u64(
            self.bytes,
            at + abi::TABLE_FINAL_ROW_COUNT_OFFSET,
            value.final_logical_row_count,
        );
        self.bytes[at + abi::TABLE_IMAGE_LAYOUT_DIGEST_OFFSET
            ..at + abi::TABLE_IMAGE_LAYOUT_DIGEST_OFFSET + 32]
            .copy_from_slice(&value.image_layout_digest);
        self.bytes[at + abi::TABLE_IMAGE_CONTENT_DIGEST_OFFSET
            ..at + abi::TABLE_IMAGE_CONTENT_DIGEST_OFFSET + 32]
            .copy_from_slice(&value.image_content_digest);
        self.table_id = value.stable_table_id;
        self.final_row_count = value.final_logical_row_count;
        self.action = Some(value.action);
    }

    pub fn write_row(&mut self, value: RuntimeTypedInsertGenerationRow) {
        if self.rows >= self.geometry.rows {
            self.reject("extra generation row");
            return;
        }
        let at = abi::HEADER_BYTES + abi::TABLE_BYTES + self.rows * abi::ROW_BYTES;
        abi::put_u64(
            self.bytes,
            at + abi::ROW_TABLE_ID_OFFSET,
            value.stable_table_id,
        );
        abi::put_u64(self.bytes, at + abi::ROW_ID_OFFSET, value.stable_row_id);
        abi::put_u32(
            self.bytes,
            at + abi::ROW_STATEMENT_ORDINAL_OFFSET,
            value.source_statement_ordinal,
        );
        abi::put_u32(
            self.bytes,
            at + abi::ROW_SOURCE_ORDINAL_OFFSET,
            value.source_row_ordinal,
        );
        abi::put_u32(
            self.bytes,
            at + abi::ROW_CELL_START_OFFSET,
            self.expected_cells as u32,
        );
        abi::put_u32(
            self.bytes,
            at + abi::ROW_CELL_COUNT_OFFSET,
            value.cell_count,
        );
        if self.rows == 0 {
            self.first_row_id = value.stable_row_id;
        }
        self.expected_cells = self
            .expected_cells
            .saturating_add(value.cell_count as usize);
        self.rows += 1;
    }

    pub fn write_cell(&mut self, value: RuntimeTypedInsertGenerationCell<'_>) {
        if self.cells >= self.geometry.cells
            || self.values + value.value.len() > self.geometry.value_bytes
        {
            self.reject("extra generation cell or value bytes");
            return;
        }
        let at = abi::HEADER_BYTES
            + abi::TABLE_BYTES
            + self.geometry.rows * abi::ROW_BYTES
            + self.cells * abi::CELL_BYTES;
        abi::put_u32(
            self.bytes,
            at + abi::CELL_CATALOG_ORDINAL_OFFSET,
            value.catalog_column_ordinal,
        );
        abi::put_u32(
            self.bytes,
            at + abi::CELL_STABLE_COLUMN_ID_OFFSET,
            value.stable_column_id,
        );
        abi::put_i16(self.bytes, at + abi::CELL_ATTNUM_OFFSET, value.attnum);
        self.bytes[at + abi::CELL_STORAGE_OFFSET..at + abi::CELL_STORAGE_OFFSET + 4]
            .copy_from_slice(&value.storage);
        abi::put_u32(
            self.bytes,
            at + abi::CELL_DECLARED_OID_OFFSET,
            value.declared_type_oid,
        );
        abi::put_i16(
            self.bytes,
            at + abi::CELL_SIGNED_SIZE_OFFSET,
            value.signed_type_size,
        );
        self.bytes[at + abi::CELL_NULL_OFFSET] = u8::from(value.is_null);
        abi::put_u32(
            self.bytes,
            at + abi::CELL_VALUE_START_OFFSET,
            self.values as u32,
        );
        abi::put_u32(
            self.bytes,
            at + abi::CELL_VALUE_COUNT_OFFSET,
            value.value.len() as u32,
        );
        let value_at = abi::HEADER_BYTES
            + abi::TABLE_BYTES
            + self.geometry.rows * abi::ROW_BYTES
            + self.geometry.cells * abi::CELL_BYTES
            + self.values;
        self.bytes[value_at..value_at + value.value.len()].copy_from_slice(value.value);
        self.values += value.value.len();
        self.cells += 1;
    }

    /// Write the one ordered index predecessor admitted by the indexed route. The device owns
    /// descriptor validation and all root derivation; this method only seals neutral catalog
    /// facts into the pre-reserved input arena.
    pub fn write_index(&mut self, value: RuntimeTypedInsertGenerationIndex) {
        if self.indexes >= self.geometry.indexes {
            self.reject("extra generation index");
            return;
        }
        let at = abi::HEADER_BYTES
            + abi::TABLE_BYTES
            + self.geometry.rows * abi::ROW_BYTES
            + self.geometry.cells * abi::CELL_BYTES
            + self.geometry.value_bytes
            + self.indexes * abi::INDEX_BYTES;
        abi::put_u64(
            self.bytes,
            at + abi::INDEX_STABLE_ID_OFFSET,
            value.stable_index_id,
        );
        abi::put_u32(
            self.bytes,
            at + abi::INDEX_RAW_CATALOG_ORDINAL_OFFSET,
            value.raw_catalog_index_ordinal,
        );
        abi::put_u32(self.bytes, at + abi::INDEX_FLAGS_OFFSET, value.index_flags);
        self.bytes[at + abi::INDEX_NULL_EQUALITY_POLICY_OFFSET] = value.null_equality_policy;
        abi::put_u64(
            self.bytes,
            at + abi::INDEX_BASE_GENERATION_OFFSET,
            value.base_generation,
        );
        self.bytes[at + abi::INDEX_BASE_ROOT_OFFSET..at + abi::INDEX_BASE_ROOT_OFFSET + 32]
            .copy_from_slice(&value.base_root);
        abi::put_u32(
            self.bytes,
            at + abi::INDEX_KEY_START_OFFSET,
            value.key_start,
        );
        abi::put_u32(
            self.bytes,
            at + abi::INDEX_KEY_COUNT_OFFSET,
            value.key_count,
        );
        abi::put_u32(
            self.bytes,
            at + abi::INDEX_EFFECT_START_OFFSET,
            value.effect_start,
        );
        abi::put_u32(
            self.bytes,
            at + abi::INDEX_EFFECT_COUNT_OFFSET,
            value.effect_count,
        );
        self.encoded_indexes[self.indexes] = Some(EncodedIndex {
            stable_index_id: value.stable_index_id,
            base_generation: value.base_generation,
        });
        self.indexes += 1;
    }

    /// Write one type-neutral key descriptor in raw catalog-index order.
    pub fn write_index_key(&mut self, value: RuntimeTypedInsertGenerationIndexKeyColumn) {
        if self.index_keys >= self.geometry.index_keys {
            self.reject("extra generation index key");
            return;
        }
        let at = abi::HEADER_BYTES
            + abi::TABLE_BYTES
            + self.geometry.rows * abi::ROW_BYTES
            + self.geometry.cells * abi::CELL_BYTES
            + self.geometry.value_bytes
            + self.geometry.indexes * abi::INDEX_BYTES
            + self.index_keys * abi::INDEX_KEY_BYTES;
        abi::put_u32(
            self.bytes,
            at + abi::INDEX_KEY_ORDINAL_OFFSET,
            value.key_ordinal,
        );
        abi::put_u32(
            self.bytes,
            at + abi::INDEX_KEY_CATALOG_ORDINAL_OFFSET,
            value.catalog_column_ordinal,
        );
        abi::put_u32(
            self.bytes,
            at + abi::INDEX_KEY_STABLE_COLUMN_ID_OFFSET,
            value.stable_column_id,
        );
        abi::put_i16(self.bytes, at + abi::INDEX_KEY_ATTNUM_OFFSET, value.attnum);
        self.bytes[at + abi::INDEX_KEY_STORAGE_OFFSET..at + abi::INDEX_KEY_STORAGE_OFFSET + 4]
            .copy_from_slice(&value.storage);
        abi::put_u32(
            self.bytes,
            at + abi::INDEX_KEY_DECLARED_OID_OFFSET,
            value.declared_type_oid,
        );
        abi::put_i16(
            self.bytes,
            at + abi::INDEX_KEY_SIGNED_SIZE_OFFSET,
            value.signed_type_size,
        );
        self.bytes[at + abi::INDEX_KEY_COLUMN_NAME_DIGEST_OFFSET
            ..at + abi::INDEX_KEY_COLUMN_NAME_DIGEST_OFFSET + 32]
            .copy_from_slice(&value.column_name_digest);
        self.index_keys += 1;
    }

    /// Write one mandatory maintenance-effect source binding. Typed values remain the exact row
    /// cells already sealed above and are rechecked/committed only by the GPU.
    pub fn write_index_effect(&mut self, value: RuntimeTypedInsertGenerationIndexEffect) {
        if self.index_effects >= self.geometry.index_effects {
            self.reject("extra generation index effect");
            return;
        }
        let at = abi::HEADER_BYTES
            + abi::TABLE_BYTES
            + self.geometry.rows * abi::ROW_BYTES
            + self.geometry.cells * abi::CELL_BYTES
            + self.geometry.value_bytes
            + self.geometry.indexes * abi::INDEX_BYTES
            + self.geometry.index_keys * abi::INDEX_KEY_BYTES
            + self.index_effects * abi::INDEX_EFFECT_BYTES;
        abi::put_u64(
            self.bytes,
            at + abi::INDEX_EFFECT_STABLE_TABLE_ID_OFFSET,
            value.stable_table_id,
        );
        abi::put_u64(
            self.bytes,
            at + abi::INDEX_EFFECT_STABLE_INDEX_ID_OFFSET,
            value.stable_index_id,
        );
        abi::put_u64(
            self.bytes,
            at + abi::INDEX_EFFECT_STABLE_ROW_ID_OFFSET,
            value.stable_row_id,
        );
        abi::put_u32(
            self.bytes,
            at + abi::INDEX_EFFECT_SOURCE_CATALOG_ORDINAL_OFFSET,
            value.source_catalog_ordinal,
        );
        abi::put_u32(
            self.bytes,
            at + abi::INDEX_EFFECT_COMPONENT_START_OFFSET,
            value.component_start,
        );
        abi::put_u32(
            self.bytes,
            at + abi::INDEX_EFFECT_COMPONENT_COUNT_OFFSET,
            value.component_count,
        );
        self.index_effects += 1;
    }

    /// Write one ordered index-effect component. Its typed value is the matching sealed row
    /// cell, so the host never supplies a duplicate value arena or digest.
    pub fn write_index_effect_component(
        &mut self,
        value: RuntimeTypedInsertGenerationIndexEffectComponent,
    ) {
        if self.index_effect_components >= self.geometry.index_effect_components {
            self.reject("extra generation index effect component");
            return;
        }
        let at = abi::HEADER_BYTES
            + abi::TABLE_BYTES
            + self.geometry.rows * abi::ROW_BYTES
            + self.geometry.cells * abi::CELL_BYTES
            + self.geometry.value_bytes
            + self.geometry.indexes * abi::INDEX_BYTES
            + self.geometry.index_keys * abi::INDEX_KEY_BYTES
            + self.geometry.index_effects * abi::INDEX_EFFECT_BYTES
            + self.index_effect_components * abi::INDEX_EFFECT_COMPONENT_BYTES;
        abi::put_u32(
            self.bytes,
            at + abi::INDEX_EFFECT_COMPONENT_CATALOG_ORDINAL_OFFSET,
            value.catalog_column_ordinal,
        );
        abi::put_u32(
            self.bytes,
            at + abi::INDEX_EFFECT_COMPONENT_STABLE_COLUMN_ID_OFFSET,
            value.stable_column_id,
        );
        self.index_effect_components += 1;
    }

    fn finish(self) -> Result<EncodedSummary, &'static str> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if !self.identity_written || !self.table_written {
            return Err("missing generation identity or table");
        }
        let expected_cells = if self.geometry.rows == 0 {
            0
        } else {
            self.geometry.cells
        };
        if self.rows != self.geometry.rows
            || self.cells != self.geometry.cells
            || self.values != self.geometry.value_bytes
            || self.expected_cells != expected_cells
        {
            return Err("incomplete generation rows, cells, or values");
        }
        if self.indexes != self.geometry.indexes
            || self.index_keys != self.geometry.index_keys
            || self.index_effects != self.geometry.index_effects
            || self.index_effect_components != self.geometry.index_effect_components
        {
            return Err("incomplete generation indexes, keys, or effects");
        }
        if self.encoded_indexes.iter().any(Option::is_none) {
            return Err("missing generation index identity");
        }
        Ok(EncodedSummary {
            first_row_id: self.first_row_id,
            table_id: self.table_id,
            commit_sequence: self.commit_sequence,
            final_row_count: self.final_row_count,
            write001_typed_statement_digest: self.bytes[abi::WRITE001_TYPED_STATEMENT_DIGEST_OFFSET
                ..abi::WRITE001_TYPED_STATEMENT_DIGEST_OFFSET + 32]
                .try_into()
                .expect("fixed WRITE-001 typed digest slot"),
        })
    }
}

impl RuntimeTypedInsertGenerationSubmission {
    fn resources(&self) -> &RuntimeTypedInsertGenerationResources {
        self.resources
            .as_ref()
            .expect("generation resources retained")
    }
    fn resources_mut(&mut self) -> &mut RuntimeTypedInsertGenerationResources {
        self.resources
            .as_mut()
            .expect("generation resources retained")
    }
    fn stream(&self) -> *mut c_void {
        self.resources()
            .stream
            .pooled
            .as_ref()
            .expect("private stream retained")
            .stream
    }

    /// Record one of the pooled stream's timing events.  Timing is advisory instrumentation:
    /// inability to record an event must never change the exactly-once generation outcome.
    #[cfg(feature = "probe-timing")]
    fn try_record_kernel_event(&self, start: bool) -> bool {
        let Some(pooled) = self.resources().stream.pooled.as_ref() else {
            return false;
        };
        let event = if start {
            pooled.start_event
        } else {
            pooled.stop_event
        };
        !event.is_null() && unsafe { (self.primary.cu_event_record)(event, pooled.stream) } == 0
    }

    #[cfg(feature = "probe-timing")]
    fn try_record_kernel_phase_event(&self, phase: usize) -> bool {
        let Some(pooled) = self.resources().stream.pooled.as_ref() else {
            return false;
        };
        let Some(&event) = pooled.generation_phase_events.get(phase) else {
            return false;
        };
        !event.is_null() && unsafe { (self.primary.cu_event_record)(event, pooled.stream) } == 0
    }

    #[cfg(feature = "probe-timing")]
    fn cuda_event_elapsed_nanos(&self, start: *mut c_void, stop: *mut c_void) -> Option<u64> {
        if start.is_null() || stop.is_null() {
            return None;
        }
        let mut elapsed_ms = 0.0_f32;
        if unsafe { (self.primary.cu_event_elapsed_time)(&mut elapsed_ms, start, stop) } != 0
            || !elapsed_ms.is_finite()
            || elapsed_ms < 0.0
        {
            return None;
        }
        Some((f64::from(elapsed_ms) * 1_000_000.0).ceil() as u64)
    }

    /// The covering stream fence has already completed before this is called.  CUDA events are
    /// therefore a kernel-only elapsed-time source, not a substitute for the host-wall seam.
    #[cfg(feature = "probe-timing")]
    fn kernel_event_elapsed_nanos(&self) -> Option<u64> {
        if !self.kernel_event_recorded {
            return None;
        }
        let pooled = self.resources().stream.pooled.as_ref()?;
        self.cuda_event_elapsed_nanos(pooled.start_event, pooled.stop_event)
    }

    #[cfg(feature = "probe-timing")]
    fn kernel_phase_event_nanos(&self) -> Option<[u64; 5]> {
        if !self.kernel_phase_events_recorded {
            return None;
        }
        let pooled = self.resources().stream.pooled.as_ref()?;
        let boundaries = [
            pooled.start_event,
            pooled.generation_phase_events[0],
            pooled.generation_phase_events[1],
            pooled.generation_phase_events[2],
            pooled.generation_phase_events[3],
            pooled.stop_event,
        ];
        let mut phases = [0_u64; 5];
        for (index, destination) in phases.iter_mut().enumerate() {
            *destination =
                self.cuda_event_elapsed_nanos(boundaries[index], boundaries[index + 1])?;
        }
        Some(phases)
    }

    fn enqueue_inner(&mut self) {
        if self.first_error.is_some() {
            self.phase = Phase::TerminalFailure {
                drain_required: false,
            };
            return;
        }
        if let Err(error) = self.primary.bind_owned_stream_before_submission() {
            self.first_error = Some(error.into());
            self.phase = Phase::TerminalFailure {
                drain_required: false,
            };
            return;
        }
        let stream = self.stream();
        // Copy raw leased-buffer coordinates before queueing.  The leases remain owned by
        // `self`, but keeping no long-lived borrow lets probe-only event bookkeeping share the
        // same stream between the transport and kernel phases.
        let (input_device, input_host, input_bytes, output_device) = {
            let resources = self.resources();
            (
                resources.input_device.ptr,
                resources.input_host.ptr.cast_const(),
                resources.input_bytes,
                resources.output_device.ptr,
            )
        };
        if let Err(error) =
            self.primary
                .enqueue_owned_stream_htod(input_device, input_host, input_bytes, stream)
        {
            self.retain_driver_error(error);
            self.phase = Phase::TerminalFailure {
                drain_required: true,
            };
            return;
        }
        let columns = self
            .geometry
            .cells
            .checked_div(self.geometry.rows)
            .unwrap_or(self.geometry.cells);
        let rows_or_columns = self.geometry.rows.max(columns);
        #[cfg(feature = "probe-timing")]
        {
            self.kernel_event_recorded = self.try_record_kernel_event(true);
            self.kernel_phase_events_recorded = self.kernel_event_recorded;
        }
        let finalize = if self.geometry.indexes == 0 {
            self.kernels.finalize_unindexed
        } else {
            self.kernels.finalize_indexed
        };
        let phases = [
            (self.kernels.validate, 1, 1),
            (
                self.kernels.cells,
                grid_for(self.geometry.cells),
                PARALLEL_PHASE_THREADS,
            ),
            (
                self.kernels.rows,
                grid_for(rows_or_columns),
                PARALLEL_PHASE_THREADS,
            ),
            (self.kernels.reduce, 1, REDUCTION_PHASE_THREADS),
            (finalize, 1, FINALIZE_PHASE_THREADS),
        ];
        for (function, grid, block) in phases {
            let status = unsafe {
                launch_kernel(
                    self.launch,
                    function,
                    grid,
                    block,
                    input_device,
                    output_device,
                    stream,
                )
            };
            if let Err(error) = self.primary.check_owned_stream_launch_result(status) {
                self.retain_driver_error(error);
                self.phase = Phase::TerminalFailure {
                    drain_required: true,
                };
                return;
            }
            #[cfg(feature = "probe-timing")]
            if self.kernel_phase_events_recorded {
                let phase = match function {
                    _ if function == self.kernels.validate => Some(0),
                    _ if function == self.kernels.cells => Some(1),
                    _ if function == self.kernels.rows => Some(2),
                    _ if function == self.kernels.reduce => Some(3),
                    _ => None,
                };
                if let Some(phase) = phase {
                    self.kernel_phase_events_recorded = self.try_record_kernel_phase_event(phase);
                }
            }
        }
        #[cfg(feature = "probe-timing")]
        if self.kernel_event_recorded && !self.try_record_kernel_event(false) {
            self.kernel_event_recorded = false;
        }
        self.primary.after_owned_stream_enqueue();
        self.phase = Phase::InFlight;
    }

    fn retain_driver_error(&mut self, error: CudaRuntimeProbeError) {
        if self.first_error.is_none() {
            self.first_error = Some(error.into());
        }
        self.driver_error_observed = true;
    }

    fn queue_readback(&mut self) -> Result<(), CudaRuntimeProbeError> {
        if !matches!(self.phase, Phase::InFlight) {
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
        self.phase = Phase::ReadbackQueued;
        Ok(())
    }

    fn drain(&self) -> Result<(), crate::cuda_context::CompletionOwnedStreamFenceFailure> {
        self.primary
            .synchronize_owned_stream_for_completion(self.stream())
    }

    pub fn complete(mut self) -> RuntimeTypedInsertGenerationCompletion {
        if matches!(self.phase, Phase::InFlight) {
            if let Err(error) = self.queue_readback() {
                self.retain_driver_error(error);
                self.phase = Phase::TerminalFailure {
                    drain_required: true,
                };
            }
        }
        if self.phase.drain_required() {
            if let Err(failure) = self.drain() {
                let error = failure.error();
                if let Some(cuda) = failure.cuda_error() {
                    self.retain_driver_error(cuda.clone());
                }
                return RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(
                    RuntimeTypedInsertGenerationUnknownQuiescence {
                        error: error.into(),
                        submission: Some(self),
                    },
                );
            }
        }
        let result = if let Some(error) = self.first_error.clone() {
            Err(error)
        } else if matches!(self.phase, Phase::ReadbackQueued) {
            self.proof_after_quiescence()
        } else {
            Err(RuntimeTypedInsertGenerationError::Runtime(
                CudaRuntimeProbeError::KernelLaunchFailed(-1),
            ))
        };
        self.phase = Phase::Quiesced;
        self.quarantine_if_driver_error();
        RuntimeTypedInsertGenerationCompletion::Quiesced(result)
    }

    fn proof_after_quiescence(
        &mut self,
    ) -> Result<OpaqueRuntimeTypedInsertGenerationProof, RuntimeTypedInsertGenerationError> {
        let attempt = self.attempt;
        let layout = self.layout;
        let rows = self.geometry.rows;
        let cells = self.geometry.cells;
        #[cfg(feature = "probe-timing")]
        let kernel_event_elapsed_nanos = self.kernel_event_elapsed_nanos();
        #[cfg(feature = "probe-timing")]
        let kernel_phase_event_nanos = self.kernel_phase_event_nanos();
        let resources = self.resources_mut();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                resources.output_host.ptr.cast::<u8>(),
                resources.output_bytes,
            )
        };
        let status = abi::get_u32(bytes, abi::OUTPUT_STATUS_OFFSET);
        if status != 0 {
            return Err(RuntimeTypedInsertGenerationError::DeviceRejected(status));
        }
        let active_nodes: usize = compact_active_level_counts(resources.first_row_id, rows)
            .into_iter()
            .sum();
        if abi::get_u32(bytes, abi::OUTPUT_ACTIVE_ROW_NODES_OFFSET) as usize != active_nodes
            || active_nodes > layout.row_node_capacity
            || abi::get_u32(bytes, abi::OUTPUT_ROW_COUNT_OFFSET) as usize != rows
            || abi::get_u32(bytes, abi::OUTPUT_CELL_COUNT_OFFSET) as usize != cells
            || abi::get_u32(bytes, abi::OUTPUT_DIGEST_COUNT_OFFSET) as usize != layout.digest_count
        {
            return Err(RuntimeTypedInsertGenerationError::DeviceRejected(u32::MAX));
        }
        let mut slots = vec![[0_u8; 32]; layout.digest_count].into_boxed_slice();
        for (ordinal, slot) in slots.iter_mut().enumerate() {
            let at = abi::OUTPUT_HEADER_BYTES + ordinal * abi::DIGEST_BYTES;
            slot.copy_from_slice(&bytes[at..at + abi::DIGEST_BYTES]);
        }
        let indexes = std::mem::take(&mut resources.indexes);
        Ok(OpaqueRuntimeTypedInsertGenerationProof {
            attempt,
            slots,
            layout,
            first_row_id: resources.first_row_id,
            rows,
            table_id: resources.table_id,
            commit_sequence: resources.commit_sequence,
            final_row_count: resources.final_row_count,
            write001_typed_statement_digest: resources.write001_typed_statement_digest,
            indexes,
            #[cfg(feature = "probe-timing")]
            kernel_event_elapsed_nanos,
            #[cfg(feature = "probe-timing")]
            kernel_phase_event_nanos,
        })
    }

    fn quarantine_if_driver_error(&mut self) {
        if self.driver_error_observed {
            if let Some(resources) = self.resources.take() {
                quarantine(resources);
            }
        }
    }
}

impl Drop for RuntimeTypedInsertGenerationSubmission {
    fn drop(&mut self) {
        if !self.phase.drain_required() {
            self.quarantine_if_driver_error();
            return;
        }
        match self.drain() {
            Ok(()) => {
                self.phase = Phase::Quiesced;
                self.quarantine_if_driver_error();
            }
            Err(failure) => {
                if let Some(cuda) = failure.cuda_error() {
                    self.retain_driver_error(cuda.clone());
                }
                if let Some(resources) = self.resources.take() {
                    if self.driver_error_observed {
                        quarantine(resources);
                    } else {
                        park(resources);
                    }
                }
            }
        }
    }
}

impl RuntimeTypedInsertGenerationUnknownQuiescence {
    pub fn error(&self) -> &RuntimeTypedInsertGenerationError {
        &self.error
    }
    pub fn retry_complete(mut self) -> RuntimeTypedInsertGenerationCompletion {
        self.submission
            .take()
            .expect("unknown quiescence retains submission")
            .complete()
    }
}

static PARKED: Mutex<[Option<RuntimeTypedInsertGenerationResources>; RETAINED_FAILURES]> =
    Mutex::new([const { None }; RETAINED_FAILURES]);
static QUARANTINED: Mutex<[Option<RuntimeTypedInsertGenerationResources>; RETAINED_FAILURES]> =
    Mutex::new([const { None }; RETAINED_FAILURES]);

fn park(resources: RuntimeTypedInsertGenerationResources) {
    retain_or_forget(&PARKED, resources);
}
fn quarantine(resources: RuntimeTypedInsertGenerationResources) {
    retain_or_forget(&QUARANTINED, resources);
}
fn retain_or_forget(
    registry: &Mutex<[Option<RuntimeTypedInsertGenerationResources>; RETAINED_FAILURES]>,
    resources: RuntimeTypedInsertGenerationResources,
) {
    let Ok(mut registry) = registry.lock() else {
        std::mem::forget(resources);
        return;
    };
    if let Some(slot) = registry.iter_mut().find(|slot| slot.is_none()) {
        *slot = Some(resources);
    } else {
        std::mem::forget(resources);
    }
}

fn validate_geometry(
    geometry: RuntimeTypedInsertGenerationGeometry,
) -> Result<(), RuntimeTypedInsertGenerationPrepareError> {
    if geometry.cells == 0
        || geometry.rows > u32::MAX as usize
        || geometry.cells > u32::MAX as usize
        || geometry.value_bytes > u32::MAX as usize
        || geometry.indexes > u32::MAX as usize
        || geometry.index_keys > u32::MAX as usize
        || geometry.index_effects > u32::MAX as usize
        || geometry.index_effect_components > u32::MAX as usize
    {
        return Err(RuntimeTypedInsertGenerationPrepareError::InvalidGeometry(
            "typed generation geometry exceeds device addressability",
        ));
    }
    if geometry.rows == 0 && geometry.value_bytes != 0 {
        return Err(RuntimeTypedInsertGenerationPrepareError::InvalidGeometry(
            "empty CREATE descriptors cannot carry logical values",
        ));
    }
    abi::compact_row_node_capacity(geometry.rows).ok_or(
        RuntimeTypedInsertGenerationPrepareError::InvalidGeometry("row-node capacity"),
    )?;
    Ok(())
}

fn grid_for(items: usize) -> u32 {
    let block = PARALLEL_PHASE_THREADS as usize;
    u32::try_from(items.div_ceil(block))
        .expect("validated typed generation geometry has a representable launch grid")
}

fn resolve_launch(
    primary: &GpuPrimaryContext,
) -> Result<(RuntimeTypedInsertGenerationKernels, CuLaunchKernel), CudaRuntimeProbeError> {
    let ptx = PTX_WITH_NUL.get_or_init(|| {
        let mut bytes = Vec::with_capacity(PTX.len() + 1);
        bytes.extend_from_slice(PTX);
        bytes.push(0);
        bytes.into_boxed_slice()
    });
    let kernels = RuntimeTypedInsertGenerationKernels {
        validate: primary
            .cached_function(c"gpu_db_runtime_typed_insert_generation_v3_validate", &ptx)?,
        cells: primary.cached_function(c"gpu_db_runtime_typed_insert_generation_v3_cells", &ptx)?,
        rows: primary.cached_function(c"gpu_db_runtime_typed_insert_generation_v3_rows", &ptx)?,
        reduce: primary
            .cached_function(c"gpu_db_runtime_typed_insert_generation_v3_reduce", &ptx)?,
        finalize_indexed: primary
            .cached_function(c"gpu_db_runtime_typed_insert_generation_v3_finalize", &ptx)?,
        finalize_unindexed: primary.cached_function(
            c"gpu_db_runtime_typed_insert_generation_v3_finalize_unindexed",
            &ptx,
        )?,
    };
    let launch = unsafe {
        *primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    Ok((kernels, launch))
}

unsafe fn launch_kernel(
    launch: CuLaunchKernel,
    function: *mut c_void,
    grid_x: u32,
    block_x: u32,
    input_ptr: u64,
    output_ptr: u64,
    stream: *mut c_void,
) -> i32 {
    let mut input = input_ptr;
    let mut output = output_ptr;
    let mut args = [
        (&mut input as *mut u64).cast::<c_void>(),
        (&mut output as *mut u64).cast::<c_void>(),
    ];
    unsafe {
        launch(
            function,
            grid_x,
            1,
            1,
            block_x,
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
#[path = "runtime_typed_insert_generation/tests.rs"]
mod tests;
