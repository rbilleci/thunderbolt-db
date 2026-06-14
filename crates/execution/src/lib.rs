use std::collections::BTreeSet;
use std::fmt;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use libloading::Library;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceTarget {
    Cpu,
    Gpu(u16),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedOp {
    pub name: String,
    pub target: DeviceTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GpuFallbackReason {
    Unavailable,
    QueueSaturated,
    MemoryPressure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouteDecision {
    Cpu,
    Gpu(u16),
    CpuFallback {
        requested_gpu: u16,
        reason: GpuFallbackReason,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GpuRuntimeSnapshot {
    pub unavailable_gpu_ids: Vec<u16>,
    pub memory_pressured_gpu_ids: Vec<u16>,
    pub saturated: bool,
}

impl GpuRuntimeSnapshot {
    pub fn has_pressure(&self) -> bool {
        self.saturated
            || !self.unavailable_gpu_ids.is_empty()
            || !self.memory_pressured_gpu_ids.is_empty()
    }

    pub fn blocked_gpu_ids(&self) -> Vec<u16> {
        self.unavailable_gpu_ids
            .iter()
            .chain(self.memory_pressured_gpu_ids.iter())
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

pub trait GpuRuntime {
    fn can_run(&self, gpu_id: u16, op: &PlannedOp) -> Result<(), GpuFallbackReason>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaRuntimeSnapshot {
    pub driver_available: bool,
    pub driver_version: Option<i32>,
    pub device_count: u16,
    pub devices: Vec<CudaDeviceSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaDeviceSnapshot {
    pub id: u16,
    pub name: String,
    pub total_memory_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaDeviceMemoryProof {
    pub gpu_id: u16,
    pub device_name: String,
    pub allocated_bytes: u64,
    pub copied_bytes: u64,
    pub retained: bool,
}

pub struct CudaResidentDeviceMemory {
    metadata: CudaDeviceMemoryProof,
    device_ptr: u64,
    context: *mut c_void,
    cu_mem_free: unsafe extern "C" fn(u64) -> i32,
    cu_ctx_destroy: unsafe extern "C" fn(*mut c_void) -> i32,
    last_kernel_event_elapsed_us: Mutex<Option<u64>>,
    _lib: Arc<Library>,
}

#[derive(Clone)]
pub struct CudaResidentDeviceMemoryReadView {
    metadata: CudaDeviceMemoryProof,
    device_ptr: u64,
    context: *mut c_void,
    _lib: Arc<Library>,
}

#[derive(Debug, Clone, Copy)]
pub struct CudaDeviceMemoryChunk<'a> {
    pub byte_offset: u64,
    pub bytes: &'a [u8],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaOwnedDeviceMemoryChunk {
    pub byte_offset: u64,
    pub bytes: Vec<u8>,
}

impl fmt::Debug for CudaResidentDeviceMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaResidentDeviceMemory")
            .field("metadata", &self.metadata)
            .field("device_ptr", &self.device_ptr)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for CudaResidentDeviceMemoryReadView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaResidentDeviceMemoryReadView")
            .field("metadata", &self.metadata)
            .field("device_ptr", &self.device_ptr)
            .finish_non_exhaustive()
    }
}

// This view never frees or mutates the retained allocation. It is valid only
// while the owning resident allocation remains alive and the snapshot generation
// that published it has not been invalidated by the engine.
unsafe impl Send for CudaResidentDeviceMemoryReadView {}
unsafe impl Sync for CudaResidentDeviceMemoryReadView {}

// SAFETY: `CudaResidentDeviceMemory` is `!Send`/`!Sync` for exactly one reason — the
// raw `context: *mut c_void`. Every other field is already thread-safe: `device_ptr`
// is a `u64`, the two `cu_*` fields are `extern "C" fn` pointers, the telemetry slot
// is a `Mutex`, and `_lib` is `Arc<Library>` (libloading marks `Library: Send + Sync`).
// Sharing the owner across reader threads is sound under the engine's
// publish-don't-mutate snapshot discipline (P1-M3, doc 14):
//   * Immutable after publication. A published generation's owner is never mutated in
//     place; readers only *read* `device_ptr`/`context`/`metadata`. The sole interior-
//     mutable field, `last_kernel_event_elapsed_us`, is a `Mutex` (no data race);
//     under concurrent readers it is semantically last-writer-wins — a telemetry
//     caveat, not a safety one (per-read timing moves into the read result in step 4).
//   * Freed exactly once, after drain. `cu_mem_free`/`cu_ctx_destroy` run only in
//     `Drop` (see below), which under `SnapshotCell`'s `Arc<Generation<_>>` refcount
//     happens only after the last reader holding that generation has released its
//     handle. No reader can observe freed device memory — the use-after-free the
//     non-owning read view risks is exactly what holding the owner here prevents.
//   * CUDA calls are thread-agnostic. `cuMemFree`/`cuCtxDestroy` are valid from any
//     thread (the driver API is thread-safe), so dropping the owner on a non-creating
//     thread is sound. A reader issuing a kernel launch must first make the context
//     current on its thread. The probe below uses launch paths that `cuCtxSetCurrent`
//     themselves (`submit_*` + `complete_detached`); most other resident launches do
//     not, and assume an ambiently-current context. On a thread with *no* context
//     current those return `INVALID_CONTEXT` (a safe error); on a thread with a
//     *different* allocation's context current they would misdirect the launch into the
//     wrong address space (unsafe). That hazard is a property of the per-allocation
//     context API — present with or without this impl — and is *not* something
//     `Send`/`Sync` introduces; eliminating it by making every launch use one shared
//     device context is P1-M3 step 3 / the shared-context milestone (plan §9.3). (This
//     sub-claim is by inspection — the probe exercises only the context-setting path.)
// This is the load-bearing `unsafe` of P1-M3; its soundness is exercised by the
// `published_resident_generation_*` GPU probe below.
unsafe impl Send for CudaResidentDeviceMemory {}
unsafe impl Sync for CudaResidentDeviceMemory {}

impl CudaResidentDeviceMemoryReadView {
    pub fn metadata(&self) -> &CudaDeviceMemoryProof {
        &self.metadata
    }

    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    pub fn context(&self) -> *mut c_void {
        self.context
    }

    pub fn submit_match_project_i32_equal_any_from_payload(
        &self,
        filter_offset: u64,
        needles: &[i32],
        projection_offsets: &[u64],
        row_count: u64,
    ) -> Result<CudaI32EqualAnyProjectSubmission, CudaRuntimeProbeError> {
        submit_cuda_resident_i32_equal_any_project(
            self,
            filter_offset,
            needles,
            projection_offsets,
            row_count,
        )
    }

    pub fn match_project_i32_equal_any_text_from_payload(
        &self,
        filter_offset: u64,
        needles: &[i32],
        projection_offsets: &[u64],
        text_offsets_byte_offset: u64,
        text_bytes_byte_offset: u64,
        text_bytes_len: u64,
        row_count: u64,
    ) -> Result<Vec<CudaI32TextBatchProjectionRow>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_equal_any_project_text(
            self,
            filter_offset,
            needles,
            projection_offsets,
            text_offsets_byte_offset,
            text_bytes_byte_offset,
            text_bytes_len,
            row_count,
        )
    }
}

trait CudaResidentReadSource {
    fn metadata(&self) -> &CudaDeviceMemoryProof;
    fn device_ptr(&self) -> u64;
    fn context(&self) -> *mut c_void;
    fn lib(&self) -> &Library;
    fn record_kernel_event_elapsed_us(&self, _elapsed_us: Option<u64>) {}
}

impl CudaResidentReadSource for CudaResidentDeviceMemory {
    fn metadata(&self) -> &CudaDeviceMemoryProof {
        &self.metadata
    }

    fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    fn context(&self) -> *mut c_void {
        self.context
    }

    fn lib(&self) -> &Library {
        self._lib.as_ref()
    }

    fn record_kernel_event_elapsed_us(&self, elapsed_us: Option<u64>) {
        CudaResidentDeviceMemory::record_kernel_event_elapsed_us(self, elapsed_us);
    }
}

impl CudaResidentReadSource for CudaResidentDeviceMemoryReadView {
    fn metadata(&self) -> &CudaDeviceMemoryProof {
        &self.metadata
    }

    fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    fn context(&self) -> *mut c_void {
        self.context
    }

    fn lib(&self) -> &Library {
        self._lib.as_ref()
    }
}

impl CudaResidentDeviceMemory {
    pub fn metadata(&self) -> &CudaDeviceMemoryProof {
        &self.metadata
    }

    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    pub fn context(&self) -> *mut c_void {
        self.context
    }

    /// Bind this allocation's CUDA context to the calling thread. The driver keeps the
    /// current context per-thread, so a reader thread that did not create the context
    /// must make it current before launching a kernel — otherwise the launch fails with
    /// `CUDA_ERROR_INVALID_CONTEXT` (201). Safe to call concurrently from many reader
    /// threads: a context may be current on multiple threads at once (driver ≥ 4.0).
    /// This is what makes the `&self` concurrent read path work on the per-allocation
    /// context model; the shared-primary-context milestone (plan §9.3) will make it a
    /// once-per-thread bind instead of once-per-read.
    pub fn set_current_context(&self) -> Result<(), CudaRuntimeProbeError> {
        type CuCtxSetCurrent = unsafe extern "C" fn(*mut c_void) -> i32;
        let cu_ctx_set_current = unsafe {
            self.lib()
                .get::<CuCtxSetCurrent>(b"cuCtxSetCurrent\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        check_cuda(unsafe { cu_ctx_set_current(self.context) })
    }

    fn lib(&self) -> &Library {
        self._lib.as_ref()
    }

    pub fn read_view(&self) -> CudaResidentDeviceMemoryReadView {
        CudaResidentDeviceMemoryReadView {
            metadata: self.metadata.clone(),
            device_ptr: self.device_ptr,
            context: self.context,
            _lib: Arc::clone(&self._lib),
        }
    }

    pub fn last_kernel_event_elapsed_us(&self) -> Option<u64> {
        self.last_kernel_event_elapsed_us
            .lock()
            .ok()
            .and_then(|elapsed| *elapsed)
    }

    pub fn clear_last_kernel_event_elapsed_us(&self) {
        self.record_kernel_event_elapsed_us(None);
    }

    fn record_kernel_event_elapsed_us(&self, elapsed_us: Option<u64>) {
        if let Ok(mut last) = self.last_kernel_event_elapsed_us.lock() {
            *last = elapsed_us;
        }
    }

    pub fn count_rows_from_header(&self) -> Result<u64, CudaRuntimeProbeError> {
        launch_cuda_resident_row_count(self)
    }

    pub fn count_i32_equal_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needle: i32,
    ) -> Result<u64, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_equal_count(self, byte_offset, row_count, needle)
    }

    pub fn count_i32_in_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needles: &[i32],
    ) -> Result<u64, CudaRuntimeProbeError> {
        let mut total = 0_u64;
        for needle in needles {
            total = total
                .checked_add(launch_cuda_resident_i32_equal_count(
                    self,
                    byte_offset,
                    row_count,
                    *needle,
                )?)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        }
        Ok(total)
    }

    pub fn count_i32_compare_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needle: i32,
        comparison: CudaI32Comparison,
    ) -> Result<u64, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_compare_count(self, byte_offset, row_count, needle, comparison)
    }

    pub fn count_i32_between_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        lower_inclusive: i32,
        upper_inclusive: i32,
    ) -> Result<u64, CudaRuntimeProbeError> {
        if lower_inclusive > upper_inclusive {
            return Ok(0);
        }
        let greater_or_equal_lower_count = launch_cuda_resident_i32_compare_count(
            self,
            byte_offset,
            row_count,
            lower_inclusive,
            CudaI32Comparison::Gte,
        )?;
        let greater_than_upper_count = launch_cuda_resident_i32_compare_count(
            self,
            byte_offset,
            row_count,
            upper_inclusive,
            CudaI32Comparison::Gt,
        )?;
        Ok(greater_or_equal_lower_count.saturating_sub(greater_than_upper_count))
    }

    pub fn count_text_prefix_from_payload(
        &self,
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        row_count: u64,
        prefix: &[u8],
    ) -> Result<u64, CudaRuntimeProbeError> {
        launch_cuda_resident_text_prefix_count(
            self,
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len,
            row_count,
            prefix,
        )
    }

    pub fn sum_i32_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
    ) -> Result<i64, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_sum(self, byte_offset, row_count)
    }

    pub fn project_i32_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
    ) -> Result<Vec<i32>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_project(self, byte_offset, row_count)
    }

    pub fn project_i32_rows_from_payload(
        &self,
        byte_offset: u64,
        row_indices: &[u64],
    ) -> Result<Vec<i32>, CudaRuntimeProbeError> {
        copy_cuda_resident_i32_rows(self, byte_offset, row_indices)
    }

    pub fn match_project_i32_equal_from_payload(
        &self,
        filters: &[(u64, i32)],
        projection_offsets: &[u64],
        row_count: u64,
    ) -> Result<Vec<Vec<i32>>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_equal_project(self, filters, projection_offsets, row_count)
    }

    pub fn match_project_i32_equal_any_from_payload(
        &self,
        filter_offset: u64,
        needles: &[i32],
        projection_offsets: &[u64],
        row_count: u64,
    ) -> Result<Vec<CudaI32BatchProjectionRow>, CudaRuntimeProbeError> {
        if row_count == 0 {
            return Ok(Vec::new());
        }
        self.submit_match_project_i32_equal_any_from_payload(
            filter_offset,
            needles,
            projection_offsets,
            row_count,
        )?
        .complete(self)
    }

    pub fn submit_match_project_i32_equal_any_from_payload(
        &self,
        filter_offset: u64,
        needles: &[i32],
        projection_offsets: &[u64],
        row_count: u64,
    ) -> Result<CudaI32EqualAnyProjectSubmission, CudaRuntimeProbeError> {
        submit_cuda_resident_i32_equal_any_project(
            self,
            filter_offset,
            needles,
            projection_offsets,
            row_count,
        )
    }

    pub fn match_i32_equal_row_indices_from_payload(
        &self,
        filters: &[(u64, i32)],
        row_count: u64,
    ) -> Result<Vec<u64>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_equal_row_indices(self, filters, row_count)
    }

    pub fn match_i32_between_row_indices_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        lower_inclusive: i32,
        upper_inclusive: i32,
    ) -> Result<Vec<u64>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_between_row_indices(
            self,
            byte_offset,
            row_count,
            lower_inclusive,
            upper_inclusive,
        )
    }

    pub fn project_text_from_payload(
        &self,
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        row_count: u64,
    ) -> Result<Vec<String>, CudaRuntimeProbeError> {
        launch_cuda_resident_text_project(
            self,
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len,
            row_count,
        )
    }

    pub fn project_text_rows_from_payload(
        &self,
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        row_indices: &[u64],
    ) -> Result<Vec<String>, CudaRuntimeProbeError> {
        copy_cuda_resident_text_rows(
            self,
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len,
            row_indices,
        )
    }

    pub fn match_project_i32_equal_any_text_from_payload(
        &self,
        filter_offset: u64,
        needles: &[i32],
        projection_offsets: &[u64],
        text_offsets_byte_offset: u64,
        text_bytes_byte_offset: u64,
        text_bytes_len: u64,
        row_count: u64,
    ) -> Result<Vec<CudaI32TextBatchProjectionRow>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_equal_any_project_text(
            self,
            filter_offset,
            needles,
            projection_offsets,
            text_offsets_byte_offset,
            text_bytes_byte_offset,
            text_bytes_len,
            row_count,
        )
    }

    pub fn grouped_stats_i32_from_payload(
        &self,
        group_byte_offset: u64,
        value_byte_offset: u64,
        row_count: u64,
    ) -> Result<Vec<CudaI32GroupedStats>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_grouped_stats(
            self,
            group_byte_offset,
            value_byte_offset,
            None,
            row_count,
        )
    }

    pub fn filtered_grouped_stats_i32_compare_from_payload(
        &self,
        group_byte_offset: u64,
        value_byte_offset: u64,
        filter_byte_offset: u64,
        row_count: u64,
        needle: i32,
        comparison: CudaI32Comparison,
    ) -> Result<Vec<CudaI32GroupedStats>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_grouped_stats(
            self,
            group_byte_offset,
            value_byte_offset,
            Some((filter_byte_offset, needle, comparison)),
            row_count,
        )
    }

    pub fn filtered_stats_i32_compare_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needle: i32,
        comparison: CudaI32Comparison,
    ) -> Result<CudaI32Stats, CudaRuntimeProbeError> {
        let values = launch_cuda_resident_i32_compare_project(
            self,
            byte_offset,
            row_count,
            needle,
            comparison,
        )?;
        Ok(CudaI32Stats::from_values(&values))
    }

    pub fn stats_i32_between_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        lower_inclusive: i32,
        upper_inclusive: i32,
    ) -> Result<CudaI32Stats, CudaRuntimeProbeError> {
        if lower_inclusive > upper_inclusive {
            return Ok(CudaI32Stats::from_values(&[]));
        }
        launch_cuda_resident_i32_between_stats(
            self,
            byte_offset,
            row_count,
            lower_inclusive,
            upper_inclusive,
        )
    }

    pub fn project_i32_compare_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needle: i32,
        comparison: CudaI32Comparison,
    ) -> Result<Vec<i32>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_compare_project(self, byte_offset, row_count, needle, comparison)
    }

    pub fn project_i32_compare_ordered_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needle: i32,
        comparison: CudaI32Comparison,
        descending: bool,
        window: (u64, u64),
    ) -> Result<Vec<i32>, CudaRuntimeProbeError> {
        let mut values = launch_cuda_resident_i32_compare_project(
            self,
            byte_offset,
            row_count,
            needle,
            comparison,
        )?;
        if descending {
            values.sort_unstable_by(|left, right| right.cmp(left));
        } else {
            values.sort_unstable();
        }
        let (offset, limit) = window;
        let offset = usize::try_from(offset)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let limit = usize::try_from(limit)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        Ok(values.into_iter().skip(offset).take(limit).collect())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaI32Comparison {
    Lt,
    Lte,
    Gt,
    Gte,
}

impl CudaI32Comparison {
    fn code(self) -> u32 {
        match self {
            Self::Lt => 1,
            Self::Lte => 2,
            Self::Gt => 3,
            Self::Gte => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaI32GroupedStats {
    pub group: i32,
    pub count: u64,
    pub sum: i64,
    pub min: i32,
    pub max: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaI32BatchProjectionRow {
    pub needle_index: usize,
    pub row_index: u64,
    pub values: Vec<i32>,
}

pub struct CudaI32EqualAnyProjectSubmission {
    projection_count: usize,
    needles_len: usize,
    row_count: u64,
    context: *mut c_void,
    values_guard: CudaDeviceAllocationGuard,
    indices_guard: CudaDeviceAllocationGuard,
    row_indices_guard: CudaDeviceAllocationGuard,
    count_guard: CudaDeviceAllocationGuard,
    _needles_guard: CudaDeviceAllocationGuard,
    _module_guard: CudaModuleGuard,
    start_event_guard: CudaEventGuard,
    stop_event_guard: CudaEventGuard,
    cu_ctx_set_current: unsafe extern "C" fn(*mut c_void) -> i32,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, u64, usize) -> i32,
    cu_event_synchronize: unsafe extern "C" fn(*mut c_void) -> i32,
    cu_event_elapsed_time: unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> i32,
}

// Pending read submissions own their temporary CUDA allocations/events/module.
// The resident allocation itself remains owned elsewhere and must outlive
// completion.
unsafe impl Send for CudaI32EqualAnyProjectSubmission {}

impl CudaI32EqualAnyProjectSubmission {
    pub fn complete(
        self,
        resident: &CudaResidentDeviceMemory,
    ) -> Result<Vec<CudaI32BatchProjectionRow>, CudaRuntimeProbeError> {
        let (rows, elapsed_us) = self.complete_detached()?;
        resident.record_kernel_event_elapsed_us(elapsed_us);
        Ok(rows)
    }

    pub fn complete_detached(
        self,
    ) -> Result<(Vec<CudaI32BatchProjectionRow>, Option<u64>), CudaRuntimeProbeError> {
        check_cuda(unsafe { (self.cu_ctx_set_current)(self.context) })?;
        check_cuda(unsafe { (self.cu_event_synchronize)(self.stop_event_guard.event) })?;

        let mut elapsed_ms = 0.0_f32;
        check_cuda(unsafe {
            (self.cu_event_elapsed_time)(
                &mut elapsed_ms,
                self.start_event_guard.event,
                self.stop_event_guard.event,
            )
        })?;
        let elapsed_us = Some((f64::from(elapsed_ms) * 1_000.0).ceil() as u64);

        let mut match_count = 0_u32;
        check_cuda(unsafe {
            (self.cu_memcpy_dtoh)(
                (&mut match_count as *mut u32).cast::<c_void>(),
                self.count_guard.ptr,
                std::mem::size_of::<u32>(),
            )
        })?;
        let match_count = u64::from(match_count);
        if match_count > self.row_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        let match_count_usize = usize::try_from(match_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let mut values = vec![0_i32; match_count_usize.saturating_mul(self.projection_count)];
        if !values.is_empty() {
            check_cuda(unsafe {
                (self.cu_memcpy_dtoh)(
                    values.as_mut_ptr().cast::<c_void>(),
                    self.values_guard.ptr,
                    values.len() * std::mem::size_of::<i32>(),
                )
            })?;
        }
        let mut needle_indices = vec![0_u32; match_count_usize];
        if !needle_indices.is_empty() {
            check_cuda(unsafe {
                (self.cu_memcpy_dtoh)(
                    needle_indices.as_mut_ptr().cast::<c_void>(),
                    self.indices_guard.ptr,
                    needle_indices.len() * std::mem::size_of::<u32>(),
                )
            })?;
        }
        let mut row_indices = vec![0_u64; match_count_usize];
        if !row_indices.is_empty() {
            check_cuda(unsafe {
                (self.cu_memcpy_dtoh)(
                    row_indices.as_mut_ptr().cast::<c_void>(),
                    self.row_indices_guard.ptr,
                    row_indices.len() * std::mem::size_of::<u64>(),
                )
            })?;
        }

        let rows = values
            .chunks_exact(self.projection_count)
            .zip(needle_indices)
            .zip(row_indices)
            .map(|((row, needle_index), row_index)| {
                let needle_index = usize::try_from(needle_index)
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                if needle_index >= self.needles_len {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(needle_index));
                }
                if row_index >= self.row_count {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
                }
                Ok(CudaI32BatchProjectionRow {
                    needle_index,
                    row_index,
                    values: row.to_vec(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((rows, elapsed_us))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaI32TextBatchProjectionRow {
    pub needle_index: usize,
    pub row_index: u64,
    pub values: Vec<i32>,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaI32Stats {
    pub count: u64,
    pub sum: i64,
    pub min: Option<i32>,
    pub max: Option<i32>,
}

impl CudaI32Stats {
    fn from_values(values: &[i32]) -> Self {
        Self {
            count: values.len() as u64,
            sum: values.iter().map(|value| i64::from(*value)).sum(),
            min: values.iter().copied().min(),
            max: values.iter().copied().max(),
        }
    }
}

impl Drop for CudaResidentDeviceMemory {
    fn drop(&mut self) {
        unsafe {
            (self.cu_mem_free)(self.device_ptr);
            (self.cu_ctx_destroy)(self.context);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaMvccRowBatch {
    pub row_count: u32,
    pub key_offsets: Vec<u32>,
    pub key_bytes: Vec<u8>,
    pub value_offsets: Vec<u32>,
    pub value_bytes: Vec<u8>,
    pub begin_txn_ids: Vec<u64>,
    pub end_txn_ids: Vec<u64>,
    pub provenance_handles: Vec<u32>,
}

impl CudaMvccRowBatch {
    pub fn from_key_values<I, K, V>(rows: I) -> Result<Self, CudaRuntimeProbeError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        Self::from_key_values_with_metadata(
            rows.into_iter()
                .map(|(key, value)| (key, value, 0_u64, u64::MAX, None)),
        )
    }

    pub fn from_key_values_with_metadata<I, K, V>(rows: I) -> Result<Self, CudaRuntimeProbeError>
    where
        I: IntoIterator<Item = (K, V, u64, u64, Option<u32>)>,
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        let mut row_count = 0_u32;
        let mut key_offsets = vec![0_u32];
        let mut key_bytes = Vec::new();
        let mut value_offsets = vec![0_u32];
        let mut value_bytes = Vec::new();
        let mut begin_txn_ids = Vec::new();
        let mut end_txn_ids = Vec::new();
        let mut provenance_handles = Vec::new();

        for (key, value, begin_txn_id, end_txn_id, provenance_handle) in rows {
            let provenance_handle = provenance_handle.unwrap_or(row_count);
            row_count = row_count
                .checked_add(1)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

            key_bytes.extend_from_slice(key.as_ref());
            key_offsets.push(
                u32::try_from(key_bytes.len())
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(key_bytes.len()))?,
            );

            value_bytes.extend_from_slice(value.as_ref());
            value_offsets.push(
                u32::try_from(value_bytes.len())
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(value_bytes.len()))?,
            );
            begin_txn_ids.push(begin_txn_id);
            end_txn_ids.push(end_txn_id);
            provenance_handles.push(provenance_handle);
        }

        Ok(Self {
            row_count,
            key_offsets,
            key_bytes,
            value_offsets,
            value_bytes,
            begin_txn_ids,
            end_txn_ids,
            provenance_handles,
        })
    }

    pub fn transfer_bytes(&self) -> usize {
        (self.key_offsets.len() + self.value_offsets.len()) * std::mem::size_of::<u32>()
            + self.key_bytes.len()
            + self.value_bytes.len()
            + self.begin_txn_ids.len() * std::mem::size_of::<u64>()
            + self.end_txn_ids.len() * std::mem::size_of::<u64>()
            + self.provenance_handles.len() * std::mem::size_of::<u32>()
    }

    pub fn key_len(&self, row_index: usize) -> Option<u32> {
        row_segment_len(&self.key_offsets, row_index)
    }

    pub fn value_len(&self, row_index: usize) -> Option<u32> {
        row_segment_len(&self.value_offsets, row_index)
    }

    pub fn validate(&self) -> Result<(), CudaRuntimeProbeError> {
        let row_count = self.row_count as usize;
        if self.key_offsets.len() != row_count + 1 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                self.key_offsets.len(),
            ));
        }
        if self.value_offsets.len() != row_count + 1 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                self.value_offsets.len(),
            ));
        }
        if self.begin_txn_ids.len() != row_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                self.begin_txn_ids.len(),
            ));
        }
        if self.end_txn_ids.len() != row_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                self.end_txn_ids.len(),
            ));
        }
        if self.provenance_handles.len() != row_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                self.provenance_handles.len(),
            ));
        }
        validate_offsets(&self.key_offsets, self.key_bytes.len())?;
        validate_offsets(&self.value_offsets, self.value_bytes.len())?;
        Ok(())
    }
}

fn row_segment_len(offsets: &[u32], row_index: usize) -> Option<u32> {
    let start = *offsets.get(row_index)?;
    let end = *offsets.get(row_index + 1)?;
    end.checked_sub(start)
}

fn validate_offsets(offsets: &[u32], bytes_len: usize) -> Result<(), CudaRuntimeProbeError> {
    if offsets.first().copied() != Some(0) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes_len));
    }
    let mut previous = 0_u32;
    for offset in offsets.iter().copied().skip(1) {
        if offset < previous {
            return Err(CudaRuntimeProbeError::InvalidInputLength(offset as usize));
        }
        previous = offset;
    }
    if previous as usize != bytes_len {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes_len));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaRuntimeProbeError {
    DriverLibraryUnavailable,
    DriverInitFailed(i32),
    DeviceCountFailed(i32),
    InvalidDeviceCount(i32),
    InvalidInputLength(usize),
    KernelLaunchFailed(i32),
}

impl fmt::Display for CudaRuntimeProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DriverLibraryUnavailable => write!(f, "CUDA driver library is unavailable"),
            Self::DriverInitFailed(code) => write!(f, "CUDA driver initialization failed: {code}"),
            Self::DeviceCountFailed(code) => {
                write!(f, "CUDA device count query failed: {code}")
            }
            Self::InvalidDeviceCount(count) => write!(f, "invalid CUDA device count: {count}"),
            Self::InvalidInputLength(len) => write!(f, "invalid CUDA input length: {len}"),
            Self::KernelLaunchFailed(code) => write!(f, "CUDA kernel launch failed: {code}"),
        }
    }
}

impl std::error::Error for CudaRuntimeProbeError {}

#[derive(Debug, Clone)]
pub struct CudaDriverRuntime {
    snapshot: CudaRuntimeSnapshot,
}

impl CudaDriverRuntime {
    pub fn probe() -> Result<Self, CudaRuntimeProbeError> {
        let snapshot = probe_cuda_runtime_snapshot()?;
        Ok(Self { snapshot })
    }

    pub fn from_device_count(device_count: u16) -> Self {
        Self {
            snapshot: CudaRuntimeSnapshot {
                driver_available: true,
                driver_version: None,
                device_count,
                devices: (0..device_count)
                    .map(|id| CudaDeviceSnapshot {
                        id,
                        name: format!("cuda-device-{id}"),
                        total_memory_bytes: 0,
                    })
                    .collect(),
            },
        }
    }

    pub fn unavailable() -> Self {
        Self {
            snapshot: CudaRuntimeSnapshot {
                driver_available: false,
                driver_version: None,
                device_count: 0,
                devices: Vec::new(),
            },
        }
    }

    pub fn snapshot(&self) -> CudaRuntimeSnapshot {
        self.snapshot.clone()
    }

    pub fn launch_smoke_add_one(&self, input: u32) -> Result<u32, CudaRuntimeProbeError> {
        if !self.snapshot.driver_available || self.snapshot.device_count == 0 {
            return Err(CudaRuntimeProbeError::DriverLibraryUnavailable);
        }

        launch_cuda_smoke_add_one(input)
    }

    pub fn filter_equal_u32_mask(
        &self,
        input: &[u32],
        needle: u32,
    ) -> Result<Vec<bool>, CudaRuntimeProbeError> {
        if !self.snapshot.driver_available || self.snapshot.device_count == 0 {
            return Err(CudaRuntimeProbeError::DriverLibraryUnavailable);
        }

        launch_cuda_u32_equal_mask(input, needle)
    }

    pub fn filter_all_mask(&self, row_count: usize) -> Result<Vec<bool>, CudaRuntimeProbeError> {
        if !self.snapshot.driver_available || self.snapshot.device_count == 0 {
            return Err(CudaRuntimeProbeError::DriverLibraryUnavailable);
        }

        launch_cuda_all_mask(row_count)
    }

    pub fn filter_equal_bytes_mask(
        &self,
        input: &[&[u8]],
        needle: &[u8],
    ) -> Result<Vec<bool>, CudaRuntimeProbeError> {
        if !self.snapshot.driver_available || self.snapshot.device_count == 0 {
            return Err(CudaRuntimeProbeError::DriverLibraryUnavailable);
        }

        launch_cuda_bytes_equal_mask(input, needle)
    }

    pub fn filter_bytes_range_mask(
        &self,
        input: &[&[u8]],
        start_inclusive: &[u8],
        end_exclusive: &[u8],
    ) -> Result<Vec<bool>, CudaRuntimeProbeError> {
        if !self.snapshot.driver_available || self.snapshot.device_count == 0 {
            return Err(CudaRuntimeProbeError::DriverLibraryUnavailable);
        }

        launch_cuda_bytes_range_mask(input, start_inclusive, end_exclusive)
    }

    pub fn mvcc_row_batch_lengths(
        &self,
        batch: &CudaMvccRowBatch,
    ) -> Result<Vec<(u32, u32)>, CudaRuntimeProbeError> {
        if !self.snapshot.driver_available || self.snapshot.device_count == 0 {
            return Err(CudaRuntimeProbeError::DriverLibraryUnavailable);
        }

        launch_cuda_mvcc_row_batch_lengths(batch)
    }

    pub fn mvcc_visibility_mask(
        &self,
        batch: &CudaMvccRowBatch,
        read_txn_id: u64,
    ) -> Result<Vec<bool>, CudaRuntimeProbeError> {
        if !self.snapshot.driver_available || self.snapshot.device_count == 0 {
            return Err(CudaRuntimeProbeError::DriverLibraryUnavailable);
        }

        launch_cuda_mvcc_visibility_mask(batch, read_txn_id)
    }

    pub fn verify_device_memory_copy(
        &self,
        gpu_id: u16,
        payload: &[u8],
    ) -> Result<CudaDeviceMemoryProof, CudaRuntimeProbeError> {
        let resident = self.retain_device_memory_copy(gpu_id, payload)?;
        let mut metadata = resident.metadata().clone();
        metadata.retained = false;
        drop(resident);
        Ok(metadata)
    }

    pub fn retain_device_memory_copy(
        &self,
        gpu_id: u16,
        payload: &[u8],
    ) -> Result<CudaResidentDeviceMemory, CudaRuntimeProbeError> {
        if !self.snapshot.driver_available || gpu_id >= self.snapshot.device_count {
            return Err(CudaRuntimeProbeError::DriverLibraryUnavailable);
        }
        if payload.is_empty() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }

        let device = self
            .snapshot
            .devices
            .iter()
            .find(|device| device.id == gpu_id)
            .cloned()
            .ok_or(CudaRuntimeProbeError::InvalidDeviceCount(i32::from(gpu_id)))?;
        let resident = launch_cuda_resident_device_memory(gpu_id, payload)?;
        Ok(CudaResidentDeviceMemory {
            metadata: CudaDeviceMemoryProof {
                gpu_id,
                device_name: device.name,
                allocated_bytes: payload.len() as u64,
                copied_bytes: payload.len() as u64,
                retained: true,
            },
            device_ptr: resident.device_ptr,
            context: resident.context,
            cu_mem_free: resident.cu_mem_free,
            cu_ctx_destroy: resident.cu_ctx_destroy,
            last_kernel_event_elapsed_us: Mutex::new(None),
            _lib: Arc::new(resident._lib),
        })
    }

    pub fn retain_device_memory_chunks(
        &self,
        gpu_id: u16,
        allocated_bytes: u64,
        chunks: &[CudaDeviceMemoryChunk<'_>],
    ) -> Result<CudaResidentDeviceMemory, CudaRuntimeProbeError> {
        if !self.snapshot.driver_available || gpu_id >= self.snapshot.device_count {
            return Err(CudaRuntimeProbeError::DriverLibraryUnavailable);
        }
        if allocated_bytes == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
        let allocated_len = usize::try_from(allocated_bytes)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if chunks.is_empty() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(allocated_len));
        }
        let mut copied_bytes = 0_u64;
        for chunk in chunks {
            if chunk.bytes.is_empty() {
                continue;
            }
            let end = chunk
                .byte_offset
                .checked_add(chunk.bytes.len() as u64)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if end > allocated_bytes {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    usize::try_from(end).unwrap_or(usize::MAX),
                ));
            }
            copied_bytes = copied_bytes
                .checked_add(chunk.bytes.len() as u64)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        }
        if copied_bytes == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }

        let device = self
            .snapshot
            .devices
            .iter()
            .find(|device| device.id == gpu_id)
            .cloned()
            .ok_or(CudaRuntimeProbeError::InvalidDeviceCount(i32::from(gpu_id)))?;
        let resident = launch_cuda_resident_device_memory_chunks(gpu_id, allocated_len, chunks)?;
        Ok(CudaResidentDeviceMemory {
            metadata: CudaDeviceMemoryProof {
                gpu_id,
                device_name: device.name,
                allocated_bytes,
                copied_bytes,
                retained: true,
            },
            device_ptr: resident.device_ptr,
            context: resident.context,
            cu_mem_free: resident.cu_mem_free,
            cu_ctx_destroy: resident.cu_ctx_destroy,
            last_kernel_event_elapsed_us: Mutex::new(None),
            _lib: Arc::new(resident._lib),
        })
    }

    pub fn retain_device_memory_owned_chunks<I>(
        &self,
        gpu_id: u16,
        allocated_bytes: u64,
        chunks: I,
    ) -> Result<CudaResidentDeviceMemory, CudaRuntimeProbeError>
    where
        I: IntoIterator<Item = CudaOwnedDeviceMemoryChunk>,
    {
        if !self.snapshot.driver_available || gpu_id >= self.snapshot.device_count {
            return Err(CudaRuntimeProbeError::DriverLibraryUnavailable);
        }
        if allocated_bytes == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
        let allocated_len = usize::try_from(allocated_bytes)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

        let device = self
            .snapshot
            .devices
            .iter()
            .find(|device| device.id == gpu_id)
            .cloned()
            .ok_or(CudaRuntimeProbeError::InvalidDeviceCount(i32::from(gpu_id)))?;
        let resident =
            launch_cuda_resident_device_memory_owned_chunks(gpu_id, allocated_len, chunks)?;
        Ok(CudaResidentDeviceMemory {
            metadata: CudaDeviceMemoryProof {
                gpu_id,
                device_name: device.name,
                allocated_bytes,
                copied_bytes: resident.copied_bytes,
                retained: true,
            },
            device_ptr: resident.device_ptr,
            context: resident.context,
            cu_mem_free: resident.cu_mem_free,
            cu_ctx_destroy: resident.cu_ctx_destroy,
            last_kernel_event_elapsed_us: Mutex::new(None),
            _lib: Arc::new(resident._lib),
        })
    }
}

struct RawCudaResidentDeviceMemory {
    device_ptr: u64,
    context: *mut c_void,
    cu_mem_free: unsafe extern "C" fn(u64) -> i32,
    cu_ctx_destroy: unsafe extern "C" fn(*mut c_void) -> i32,
    copied_bytes: u64,
    _lib: Library,
}

fn launch_cuda_resident_device_memory(
    gpu_id: u16,
    payload: &[u8],
) -> Result<RawCudaResidentDeviceMemory, CudaRuntimeProbeError> {
    launch_cuda_resident_device_memory_chunks(
        gpu_id,
        payload.len(),
        &[CudaDeviceMemoryChunk {
            byte_offset: 0,
            bytes: payload,
        }],
    )
}

fn launch_cuda_resident_device_memory_chunks(
    gpu_id: u16,
    allocated_len: usize,
    chunks: &[CudaDeviceMemoryChunk<'_>],
) -> Result<RawCudaResidentDeviceMemory, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        *lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        *lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        *lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        *lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        *lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        *lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        *lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, i32::from(gpu_id)) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: cu_ctx_destroy,
    };

    let mut device_ptr = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_ptr, allocated_len) })?;
    let allocation_guard = CudaDeviceAllocationGuard {
        ptr: device_ptr,
        free: cu_mem_free,
    };

    let mut copied_bytes = 0_u64;
    for chunk in chunks {
        if chunk.bytes.is_empty() {
            continue;
        }
        copied_bytes = copied_bytes
            .checked_add(chunk.bytes.len() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let destination = allocation_guard
            .ptr
            .checked_add(chunk.byte_offset)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        check_cuda(unsafe {
            cu_memcpy_htod(
                destination,
                chunk.bytes.as_ptr().cast::<c_void>(),
                chunk.bytes.len(),
            )
        })?;
    }

    std::mem::forget(allocation_guard);
    std::mem::forget(context_guard);

    Ok(RawCudaResidentDeviceMemory {
        device_ptr,
        context,
        cu_mem_free,
        cu_ctx_destroy,
        copied_bytes,
        _lib: lib,
    })
}

fn launch_cuda_resident_device_memory_owned_chunks<I>(
    gpu_id: u16,
    allocated_len: usize,
    chunks: I,
) -> Result<RawCudaResidentDeviceMemory, CudaRuntimeProbeError>
where
    I: IntoIterator<Item = CudaOwnedDeviceMemoryChunk>,
{
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        *lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        *lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        *lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        *lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        *lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        *lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        *lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, i32::from(gpu_id)) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: cu_ctx_destroy,
    };

    let mut device_ptr = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_ptr, allocated_len) })?;
    let allocation_guard = CudaDeviceAllocationGuard {
        ptr: device_ptr,
        free: cu_mem_free,
    };

    let allocated_bytes = allocated_len as u64;
    let mut copied_bytes = 0_u64;
    for chunk in chunks {
        if chunk.bytes.is_empty() {
            continue;
        }
        let end = chunk
            .byte_offset
            .checked_add(chunk.bytes.len() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if end > allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(end).unwrap_or(usize::MAX),
            ));
        }
        copied_bytes = copied_bytes
            .checked_add(chunk.bytes.len() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let destination = allocation_guard
            .ptr
            .checked_add(chunk.byte_offset)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        check_cuda(unsafe {
            cu_memcpy_htod(
                destination,
                chunk.bytes.as_ptr().cast::<c_void>(),
                chunk.bytes.len(),
            )
        })?;
    }
    if copied_bytes == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }

    std::mem::forget(allocation_guard);
    std::mem::forget(context_guard);

    Ok(RawCudaResidentDeviceMemory {
        device_ptr,
        context,
        cu_mem_free,
        cu_ctx_destroy,
        copied_bytes,
        _lib: lib,
    })
}

fn launch_cuda_resident_row_count(
    resident: &CudaResidentDeviceMemory,
) -> Result<u64, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_row_count(
    .param .u64 resident_ptr,
    .param .u64 out_ptr
)
{
    .reg .u64 %resident;
    .reg .u64 %out;
    .reg .u64 %rows;
    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.global.u64 %rows, [%resident];
    st.global.u64 [%out], %rows;
    ret;
}
"#;

    if resident.metadata().allocated_bytes < std::mem::size_of::<u64>() as u64 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            resident.metadata().allocated_bytes as usize,
        ));
    }

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut device_output = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_output, std::mem::size_of::<u64>()) })?;
    let allocation_guard = CudaDeviceAllocationGuard {
        ptr: device_output,
        free: *cu_mem_free,
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(&mut function, module, c"gpu_db_resident_row_count".as_ptr())
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut output_arg = allocation_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
    ];
    launch_with_optional_cuda_event_timing(resident, *cu_ctx_synchronize, || unsafe {
        cu_launch_kernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;

    let mut output = 0_u64;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut output as *mut u64).cast::<c_void>(),
            allocation_guard.ptr,
            std::mem::size_of::<u64>(),
        )
    })?;

    drop(module_guard);
    drop(allocation_guard);

    Ok(output)
}

fn launch_cuda_resident_i32_equal_count(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
    needle: i32,
) -> Result<u64, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_equal_count(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .s32 needle,
    .param .u64 out_ptr
)
{
    .reg .pred %p_done;
    .reg .pred %p_match;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %addr;
    .reg .u64 %matches;
    .reg .s32 %needle;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.s32 %needle, [needle];
    ld.param.u64 %out, [out_ptr];

    add.u64 %base, %resident, %offset;
    mov.u64 %idx, 0;
    mov.u64 %matches, 0;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %addr, %idx, 4;
    add.u64 %addr, %base, %addr;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle;
    @!%p_match bra next;
    add.u64 %matches, %matches, 1;

next:
    add.u64 %idx, %idx, 1;
    bra loop;

done:
    st.global.u64 [%out], %matches;
    ret;
}
"#;

    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut device_output = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_output, std::mem::size_of::<u64>()) })?;
    let allocation_guard = CudaDeviceAllocationGuard {
        ptr: device_output,
        free: *cu_mem_free,
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_resident_i32_equal_count".as_ptr(),
        )
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut offset_arg = byte_offset;
    let mut rows_arg = row_count;
    let mut needle_arg = needle;
    let mut output_arg = allocation_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut offset_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut needle_arg as *mut i32).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
    ];
    launch_with_optional_cuda_event_timing(resident, *cu_ctx_synchronize, || unsafe {
        cu_launch_kernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;

    let mut output = 0_u64;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut output as *mut u64).cast::<c_void>(),
            allocation_guard.ptr,
            std::mem::size_of::<u64>(),
        )
    })?;
    drop(module_guard);
    drop(allocation_guard);
    Ok(output)
}

fn launch_cuda_resident_i32_compare_count(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
    needle: i32,
    comparison: CudaI32Comparison,
) -> Result<u64, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_compare_count(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .s32 needle,
    .param .u32 comparison,
    .param .u64 out_ptr
)
{
    .reg .pred %p_done;
    .reg .pred %p_lt;
    .reg .pred %p_lte;
    .reg .pred %p_gt;
    .reg .pred %p_gte;
    .reg .pred %p_code_lt;
    .reg .pred %p_code_lte;
    .reg .pred %p_code_gt;
    .reg .pred %p_code_gte;
    .reg .pred %p_match;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %addr;
    .reg .u64 %matches;
    .reg .u32 %comparison;
    .reg .s32 %needle;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.s32 %needle, [needle];
    ld.param.u32 %comparison, [comparison];
    ld.param.u64 %out, [out_ptr];

    add.u64 %base, %resident, %offset;
    mov.u64 %idx, 0;
    mov.u64 %matches, 0;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %addr, %idx, 4;
    add.u64 %addr, %base, %addr;
    ld.global.s32 %r_value, [%addr];
    setp.lt.s32 %p_lt, %r_value, %needle;
    setp.le.s32 %p_lte, %r_value, %needle;
    setp.gt.s32 %p_gt, %r_value, %needle;
    setp.ge.s32 %p_gte, %r_value, %needle;
    setp.eq.u32 %p_code_lt, %comparison, 1;
    setp.eq.u32 %p_code_lte, %comparison, 2;
    setp.eq.u32 %p_code_gt, %comparison, 3;
    setp.eq.u32 %p_code_gte, %comparison, 4;
    mov.pred %p_match, 0;
    and.pred %p_lt, %p_lt, %p_code_lt;
    or.pred %p_match, %p_match, %p_lt;
    and.pred %p_lte, %p_lte, %p_code_lte;
    or.pred %p_match, %p_match, %p_lte;
    and.pred %p_gt, %p_gt, %p_code_gt;
    or.pred %p_match, %p_match, %p_gt;
    and.pred %p_gte, %p_gte, %p_code_gte;
    or.pred %p_match, %p_match, %p_gte;
    @!%p_match bra next;
    add.u64 %matches, %matches, 1;

next:
    add.u64 %idx, %idx, 1;
    bra loop;

done:
    st.global.u64 [%out], %matches;
    ret;
}
"#;

    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut device_output = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_output, std::mem::size_of::<u64>()) })?;
    let allocation_guard = CudaDeviceAllocationGuard {
        ptr: device_output,
        free: *cu_mem_free,
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_resident_i32_compare_count".as_ptr(),
        )
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut offset_arg = byte_offset;
    let mut rows_arg = row_count;
    let mut needle_arg = needle;
    let mut comparison_arg = comparison.code();
    let mut output_arg = allocation_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut offset_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut needle_arg as *mut i32).cast::<c_void>(),
        (&mut comparison_arg as *mut u32).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
    ];
    launch_with_optional_cuda_event_timing(resident, *cu_ctx_synchronize, || unsafe {
        cu_launch_kernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;

    let mut output = 0_u64;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut output as *mut u64).cast::<c_void>(),
            allocation_guard.ptr,
            std::mem::size_of::<u64>(),
        )
    })?;
    drop(module_guard);
    drop(allocation_guard);
    Ok(output)
}

fn launch_cuda_resident_text_prefix_count(
    resident: &CudaResidentDeviceMemory,
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    bytes_len: u64,
    row_count: u64,
    prefix: &[u8],
) -> Result<u64, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;

    let offsets_len = row_count
        .checked_add(1)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u64>() as u64))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let offsets_end = offsets_byte_offset
        .checked_add(offsets_len)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes_end = bytes_byte_offset
        .checked_add(bytes_len)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if offsets_end > resident.metadata().allocated_bytes
        || bytes_end > resident.metadata().allocated_bytes
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            offsets_end.max(bytes_end) as usize,
        ));
    }
    let offsets_len_usize = usize::try_from(offsets_len)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes_len_usize = usize::try_from(bytes_len)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let row_count_usize = usize::try_from(row_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut raw_offsets = vec![0_u8; offsets_len_usize];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            raw_offsets.as_mut_ptr().cast::<c_void>(),
            resident.device_ptr() + offsets_byte_offset,
            offsets_len_usize,
        )
    })?;
    let mut bytes = vec![0_u8; bytes_len_usize];
    if bytes_len_usize > 0 {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                bytes.as_mut_ptr().cast::<c_void>(),
                resident.device_ptr() + bytes_byte_offset,
                bytes_len_usize,
            )
        })?;
    }

    let mut offsets = Vec::with_capacity(row_count_usize + 1);
    for chunk in raw_offsets.chunks_exact(std::mem::size_of::<u64>()) {
        offsets.push(u64::from_le_bytes(chunk.try_into().map_err(|_| {
            CudaRuntimeProbeError::InvalidInputLength(raw_offsets.len())
        })?));
    }
    if offsets.len() != row_count_usize + 1 || offsets.first().copied() != Some(0) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(offsets.len()));
    }
    let mut matches = 0_u64;
    for pair in offsets.windows(2) {
        let start = usize::try_from(pair[0])
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let end = usize::try_from(pair[1])
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if start > end || end > bytes.len() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(end));
        }
        if bytes[start..end].starts_with(prefix) {
            matches = matches.saturating_add(1);
        }
    }
    Ok(matches)
}

fn launch_cuda_resident_text_project(
    resident: &CudaResidentDeviceMemory,
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    bytes_len: u64,
    row_count: u64,
) -> Result<Vec<String>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;

    let offsets_len = row_count
        .checked_add(1)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u64>() as u64))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let offsets_end = offsets_byte_offset
        .checked_add(offsets_len)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes_end = bytes_byte_offset
        .checked_add(bytes_len)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if offsets_end > resident.metadata().allocated_bytes
        || bytes_end > resident.metadata().allocated_bytes
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            offsets_end.max(bytes_end) as usize,
        ));
    }
    let offsets_len_usize = usize::try_from(offsets_len)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes_len_usize = usize::try_from(bytes_len)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let row_count_usize = usize::try_from(row_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut raw_offsets = vec![0_u8; offsets_len_usize];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            raw_offsets.as_mut_ptr().cast::<c_void>(),
            resident.device_ptr() + offsets_byte_offset,
            offsets_len_usize,
        )
    })?;
    let mut bytes = vec![0_u8; bytes_len_usize];
    if bytes_len_usize > 0 {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                bytes.as_mut_ptr().cast::<c_void>(),
                resident.device_ptr() + bytes_byte_offset,
                bytes_len_usize,
            )
        })?;
    }

    let mut offsets = Vec::with_capacity(row_count_usize + 1);
    for chunk in raw_offsets.chunks_exact(std::mem::size_of::<u64>()) {
        offsets.push(u64::from_le_bytes(chunk.try_into().map_err(|_| {
            CudaRuntimeProbeError::InvalidInputLength(raw_offsets.len())
        })?));
    }
    if offsets.len() != row_count_usize + 1 || offsets.first().copied() != Some(0) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(offsets.len()));
    }
    let mut values = Vec::with_capacity(row_count_usize);
    for pair in offsets.windows(2) {
        let start = usize::try_from(pair[0])
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let end = usize::try_from(pair[1])
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if start > end || end > bytes.len() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(end));
        }
        let value = std::str::from_utf8(&bytes[start..end])
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(end))?;
        values.push(value.to_string());
    }
    Ok(values)
}

fn copy_cuda_resident_i32_rows(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_indices: &[u64],
) -> Result<Vec<i32>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;

    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut values = Vec::with_capacity(row_indices.len());
    for row_idx in row_indices {
        let value_offset = row_idx
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .and_then(|offset| byte_offset.checked_add(offset))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let value_end = value_offset
            .checked_add(std::mem::size_of::<i32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if value_end > resident.metadata().allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                value_end as usize,
            ));
        }
        let mut value = 0_i32;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                (&mut value as *mut i32).cast::<c_void>(),
                resident.device_ptr() + value_offset,
                std::mem::size_of::<i32>(),
            )
        })?;
        values.push(value);
    }
    Ok(values)
}

fn launch_cuda_resident_i32_equal_project(
    resident: &CudaResidentDeviceMemory,
    filters: &[(u64, i32)],
    projection_offsets: &[u64],
    row_count: u64,
) -> Result<Vec<Vec<i32>>, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const MAX_FILTERS: usize = 4;
    const MAX_PROJECTIONS: usize = 4;
    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_equal_project(
    .param .u64 resident_ptr,
    .param .u64 row_count,
    .param .u32 filter_count,
    .param .u32 projection_count,
    .param .u64 filter_offset0,
    .param .u64 filter_offset1,
    .param .u64 filter_offset2,
    .param .u64 filter_offset3,
    .param .u64 projection_offset0,
    .param .u64 projection_offset1,
    .param .u64 projection_offset2,
    .param .u64 projection_offset3,
    .param .s32 needle0,
    .param .s32 needle1,
    .param .s32 needle2,
    .param .s32 needle3,
    .param .u64 out_values_ptr,
    .param .u64 out_count_ptr
)
{
    .reg .pred %p_out;
    .reg .pred %p_done;
    .reg .pred %p_match;
    .reg .pred %p_check;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %idx32;
    .reg .u32 %filter_count;
    .reg .u32 %projection_count;
    .reg .u32 %slot;
    .reg .u32 %one;
    .reg .u64 %idx;
    .reg .u64 %rows;
    .reg .u64 %resident;
    .reg .u64 %filter_offset0;
    .reg .u64 %filter_offset1;
    .reg .u64 %filter_offset2;
    .reg .u64 %filter_offset3;
    .reg .u64 %projection_offset0;
    .reg .u64 %projection_offset1;
    .reg .u64 %projection_offset2;
    .reg .u64 %projection_offset3;
    .reg .u64 %out_values;
    .reg .u64 %out_count;
    .reg .u64 %row_byte;
    .reg .u64 %addr;
    .reg .u64 %slot64;
    .reg .u64 %projection_count64;
    .reg .u64 %base_slot;
    .reg .u64 %out_addr;
    .reg .s32 %needle0;
    .reg .s32 %needle1;
    .reg .s32 %needle2;
    .reg .s32 %needle3;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %rows, [row_count];
    ld.param.u32 %filter_count, [filter_count];
    ld.param.u32 %projection_count, [projection_count];
    ld.param.u64 %filter_offset0, [filter_offset0];
    ld.param.u64 %filter_offset1, [filter_offset1];
    ld.param.u64 %filter_offset2, [filter_offset2];
    ld.param.u64 %filter_offset3, [filter_offset3];
    ld.param.u64 %projection_offset0, [projection_offset0];
    ld.param.u64 %projection_offset1, [projection_offset1];
    ld.param.u64 %projection_offset2, [projection_offset2];
    ld.param.u64 %projection_offset3, [projection_offset3];
    ld.param.s32 %needle0, [needle0];
    ld.param.s32 %needle1, [needle1];
    ld.param.s32 %needle2, [needle2];
    ld.param.s32 %needle3, [needle3];
    ld.param.u64 %out_values, [out_values_ptr];
    ld.param.u64 %out_count, [out_count_ptr];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %idx32, %r_block, %r_block_dim, %r_tid;
    cvt.u64.u32 %idx, %idx32;

    setp.ge.u64 %p_out, %idx, %rows;
    @%p_out bra DONE;
    setp.eq.u32 %p_done, %filter_count, 0;
    @%p_done bra DONE;
    setp.eq.u32 %p_done, %projection_count, 0;
    @%p_done bra DONE;

    mul.lo.u64 %row_byte, %idx, 4;

    add.u64 %addr, %resident, %filter_offset0;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle0;
    @!%p_match bra DONE;

    setp.le.u32 %p_check, %filter_count, 1;
    @%p_check bra MATCHED;
    add.u64 %addr, %resident, %filter_offset1;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle1;
    @!%p_match bra DONE;

    setp.le.u32 %p_check, %filter_count, 2;
    @%p_check bra MATCHED;
    add.u64 %addr, %resident, %filter_offset2;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle2;
    @!%p_match bra DONE;

    setp.le.u32 %p_check, %filter_count, 3;
    @%p_check bra MATCHED;
    add.u64 %addr, %resident, %filter_offset3;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle3;
    @!%p_match bra DONE;

MATCHED:
    mov.u32 %one, 1;
    atom.global.add.u32 %slot, [%out_count], %one;
    cvt.u64.u32 %slot64, %slot;
    cvt.u64.u32 %projection_count64, %projection_count;
    mul.lo.u64 %base_slot, %slot64, %projection_count64;
    mul.lo.u64 %base_slot, %base_slot, 4;

    add.u64 %addr, %resident, %projection_offset0;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    st.global.s32 [%out_addr], %r_value;

    setp.le.u32 %p_check, %projection_count, 1;
    @%p_check bra DONE;
    add.u64 %addr, %resident, %projection_offset1;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 4;
    st.global.s32 [%out_addr], %r_value;

    setp.le.u32 %p_check, %projection_count, 2;
    @%p_check bra DONE;
    add.u64 %addr, %resident, %projection_offset2;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 8;
    st.global.s32 [%out_addr], %r_value;

    setp.le.u32 %p_check, %projection_count, 3;
    @%p_check bra DONE;
    add.u64 %addr, %resident, %projection_offset3;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 12;
    st.global.s32 [%out_addr], %r_value;

DONE:
    ret;
}
"#;

    if filters.is_empty() || filters.len() > MAX_FILTERS {
        return Err(CudaRuntimeProbeError::InvalidInputLength(filters.len()));
    }
    if projection_offsets.is_empty() || projection_offsets.len() > MAX_PROJECTIONS {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            projection_offsets.len(),
        ));
    }
    if row_count == 0 {
        return Ok(Vec::new());
    }
    for (byte_offset, _) in filters {
        let bytes = row_count
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .and_then(|bytes| byte_offset.checked_add(bytes))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if bytes > resident.metadata().allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
        }
    }
    for byte_offset in projection_offsets {
        let bytes = row_count
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .and_then(|bytes| byte_offset.checked_add(bytes))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if bytes > resident.metadata().allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
        }
    }
    let row_count_u32 = u32::try_from(row_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_cells = row_count
        .checked_mul(projection_offsets.len() as u64)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_bytes = usize::try_from(
        output_cells
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memset_d8 = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| resident.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut device_values = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_values, output_bytes) })?;
    let values_guard = CudaDeviceAllocationGuard {
        ptr: device_values,
        free: *cu_mem_free,
    };
    let mut device_count = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_count, std::mem::size_of::<u32>()) })?;
    let count_guard = CudaDeviceAllocationGuard {
        ptr: device_count,
        free: *cu_mem_free,
    };
    check_cuda(unsafe { cu_memset_d8(count_guard.ptr, 0, std::mem::size_of::<u32>()) })?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_resident_i32_equal_project".as_ptr(),
        )
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut rows_arg = row_count;
    let mut filter_count_arg = u32::try_from(filters.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(filters.len()))?;
    let mut projection_count_arg = u32::try_from(projection_offsets.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(projection_offsets.len()))?;
    let mut filter_offsets = [0_u64; MAX_FILTERS];
    let mut needles = [0_i32; MAX_FILTERS];
    for (idx, (offset, needle)) in filters.iter().enumerate() {
        filter_offsets[idx] = *offset;
        needles[idx] = *needle;
    }
    let mut projected_offsets = [0_u64; MAX_PROJECTIONS];
    for (idx, offset) in projection_offsets.iter().enumerate() {
        projected_offsets[idx] = *offset;
    }
    let mut output_arg = values_guard.ptr;
    let mut count_arg = count_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut filter_count_arg as *mut u32).cast::<c_void>(),
        (&mut projection_count_arg as *mut u32).cast::<c_void>(),
        (&mut filter_offsets[0] as *mut u64).cast::<c_void>(),
        (&mut filter_offsets[1] as *mut u64).cast::<c_void>(),
        (&mut filter_offsets[2] as *mut u64).cast::<c_void>(),
        (&mut filter_offsets[3] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[0] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[1] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[2] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[3] as *mut u64).cast::<c_void>(),
        (&mut needles[0] as *mut i32).cast::<c_void>(),
        (&mut needles[1] as *mut i32).cast::<c_void>(),
        (&mut needles[2] as *mut i32).cast::<c_void>(),
        (&mut needles[3] as *mut i32).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut count_arg as *mut u64).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count_u32.div_ceil(threads_per_block);
    launch_with_optional_cuda_event_timing(resident, *cu_ctx_synchronize, || unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;

    let mut match_count = 0_u32;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut match_count as *mut u32).cast::<c_void>(),
            count_guard.ptr,
            std::mem::size_of::<u32>(),
        )
    })?;
    let match_count = u64::from(match_count);
    if match_count > row_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    let match_count_usize = usize::try_from(match_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let projection_count = projection_offsets.len();
    let mut values = vec![0_i32; match_count_usize.saturating_mul(projection_count)];
    if !values.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                values.as_mut_ptr().cast::<c_void>(),
                values_guard.ptr,
                values.len() * std::mem::size_of::<i32>(),
            )
        })?;
    }

    drop(module_guard);
    drop(count_guard);
    drop(values_guard);
    Ok(values
        .chunks_exact(projection_count)
        .map(|row| row.to_vec())
        .collect())
}

fn submit_cuda_resident_i32_equal_any_project<R: CudaResidentReadSource>(
    resident: &R,
    filter_offset: u64,
    needles: &[i32],
    projection_offsets: &[u64],
    row_count: u64,
) -> Result<CudaI32EqualAnyProjectSubmission, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSetCurrent = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuEventCreate = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
    type CuEventDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuEventRecord = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
    type CuEventSynchronize = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuEventElapsedTime = unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> i32;

    const MAX_PROJECTIONS: usize = 4;
    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_equal_any_project(
    .param .u64 resident_ptr,
    .param .u64 row_count,
    .param .u32 needle_count,
    .param .u32 projection_count,
    .param .u64 filter_offset,
    .param .u64 projection_offset0,
    .param .u64 projection_offset1,
    .param .u64 projection_offset2,
    .param .u64 projection_offset3,
    .param .u64 needles_ptr,
    .param .u64 out_values_ptr,
    .param .u64 out_needle_indices_ptr,
    .param .u64 out_row_indices_ptr,
    .param .u64 out_count_ptr
)
{
    .reg .pred %p_out;
    .reg .pred %p_done;
    .reg .pred %p_match;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %idx32;
    .reg .u32 %needle_count;
    .reg .u32 %projection_count;
    .reg .u32 %needle_idx;
    .reg .u32 %slot;
    .reg .u32 %one;
    .reg .u64 %idx;
    .reg .u64 %rows;
    .reg .u64 %resident;
    .reg .u64 %filter_offset;
    .reg .u64 %projection_offset0;
    .reg .u64 %projection_offset1;
    .reg .u64 %projection_offset2;
    .reg .u64 %projection_offset3;
    .reg .u64 %needles;
    .reg .u64 %out_values;
    .reg .u64 %out_needle_indices;
    .reg .u64 %out_row_indices;
    .reg .u64 %out_count;
    .reg .u64 %row_byte;
    .reg .u64 %addr;
    .reg .u64 %needle_byte;
    .reg .u64 %slot64;
    .reg .u64 %projection_count64;
    .reg .u64 %base_slot;
    .reg .u64 %out_addr;
    .reg .s32 %row_value;
    .reg .s32 %needle_value;
    .reg .s32 %projection_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %rows, [row_count];
    ld.param.u32 %needle_count, [needle_count];
    ld.param.u32 %projection_count, [projection_count];
    ld.param.u64 %filter_offset, [filter_offset];
    ld.param.u64 %projection_offset0, [projection_offset0];
    ld.param.u64 %projection_offset1, [projection_offset1];
    ld.param.u64 %projection_offset2, [projection_offset2];
    ld.param.u64 %projection_offset3, [projection_offset3];
    ld.param.u64 %needles, [needles_ptr];
    ld.param.u64 %out_values, [out_values_ptr];
    ld.param.u64 %out_needle_indices, [out_needle_indices_ptr];
    ld.param.u64 %out_row_indices, [out_row_indices_ptr];
    ld.param.u64 %out_count, [out_count_ptr];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %idx32, %r_block, %r_block_dim, %r_tid;
    cvt.u64.u32 %idx, %idx32;

    setp.ge.u64 %p_out, %idx, %rows;
    @%p_out bra DONE;
    setp.eq.u32 %p_done, %needle_count, 0;
    @%p_done bra DONE;
    setp.eq.u32 %p_done, %projection_count, 0;
    @%p_done bra DONE;

    mul.lo.u64 %row_byte, %idx, 4;
    add.u64 %addr, %resident, %filter_offset;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %row_value, [%addr];

    mov.u32 %needle_idx, 0;
NEEDLE_LOOP:
    setp.ge.u32 %p_done, %needle_idx, %needle_count;
    @%p_done bra DONE;
    cvt.u64.u32 %needle_byte, %needle_idx;
    mul.lo.u64 %needle_byte, %needle_byte, 4;
    add.u64 %addr, %needles, %needle_byte;
    ld.global.s32 %needle_value, [%addr];
    setp.eq.s32 %p_match, %row_value, %needle_value;
    @%p_match bra MATCHED;
    add.u32 %needle_idx, %needle_idx, 1;
    bra NEEDLE_LOOP;

MATCHED:
    mov.u32 %one, 1;
    atom.global.add.u32 %slot, [%out_count], %one;
    cvt.u64.u32 %slot64, %slot;

    mul.lo.u64 %out_addr, %slot64, 4;
    add.u64 %out_addr, %out_needle_indices, %out_addr;
    st.global.u32 [%out_addr], %needle_idx;

    mul.lo.u64 %out_addr, %slot64, 8;
    add.u64 %out_addr, %out_row_indices, %out_addr;
    st.global.u64 [%out_addr], %idx;

    cvt.u64.u32 %projection_count64, %projection_count;
    mul.lo.u64 %base_slot, %slot64, %projection_count64;
    mul.lo.u64 %base_slot, %base_slot, 4;

    add.u64 %addr, %resident, %projection_offset0;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    st.global.s32 [%out_addr], %projection_value;

    setp.le.u32 %p_done, %projection_count, 1;
    @%p_done bra DONE;
    add.u64 %addr, %resident, %projection_offset1;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 4;
    st.global.s32 [%out_addr], %projection_value;

    setp.le.u32 %p_done, %projection_count, 2;
    @%p_done bra DONE;
    add.u64 %addr, %resident, %projection_offset2;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 8;
    st.global.s32 [%out_addr], %projection_value;

    setp.le.u32 %p_done, %projection_count, 3;
    @%p_done bra DONE;
    add.u64 %addr, %resident, %projection_offset3;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 12;
    st.global.s32 [%out_addr], %projection_value;

DONE:
    ret;
}
"#;

    if needles.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if projection_offsets.is_empty() || projection_offsets.len() > MAX_PROJECTIONS {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            projection_offsets.len(),
        ));
    }
    if row_count == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let filter_bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| filter_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if filter_bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            filter_bytes as usize,
        ));
    }
    for byte_offset in projection_offsets {
        let bytes = row_count
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .and_then(|bytes| byte_offset.checked_add(bytes))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if bytes > resident.metadata().allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
        }
    }
    let row_count_u32 = u32::try_from(row_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let needle_count_u32 = u32::try_from(needles.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(needles.len()))?;
    let output_cells = row_count
        .checked_mul(projection_offsets.len() as u64)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_bytes = usize::try_from(
        output_cells
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_indices_bytes = usize::try_from(
        row_count
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_row_indices_bytes = usize::try_from(
        row_count
            .checked_mul(std::mem::size_of::<u64>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let needle_bytes = needles
        .len()
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memset_d8 = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| resident.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_set_current = unsafe {
        resident
            .lib()
            .get::<CuCtxSetCurrent>(b"cuCtxSetCurrent\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_event_create = unsafe {
        resident
            .lib()
            .get::<CuEventCreate>(b"cuEventCreate\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_event_destroy = unsafe {
        resident
            .lib()
            .get::<CuEventDestroy>(b"cuEventDestroy_v2\0")
            .or_else(|_| resident.lib().get::<CuEventDestroy>(b"cuEventDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_event_record = unsafe {
        resident
            .lib()
            .get::<CuEventRecord>(b"cuEventRecord\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_event_synchronize = unsafe {
        resident
            .lib()
            .get::<CuEventSynchronize>(b"cuEventSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_event_elapsed_time = unsafe {
        resident
            .lib()
            .get::<CuEventElapsedTime>(b"cuEventElapsedTime\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { cu_ctx_set_current(resident.context()) })?;

    let mut device_needles = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_needles, needle_bytes) })?;
    let needles_guard = CudaDeviceAllocationGuard {
        ptr: device_needles,
        free: *cu_mem_free,
    };
    check_cuda(unsafe {
        cu_memcpy_htod(
            needles_guard.ptr,
            needles.as_ptr().cast::<c_void>(),
            needle_bytes,
        )
    })?;

    let mut device_values = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_values, output_bytes) })?;
    let values_guard = CudaDeviceAllocationGuard {
        ptr: device_values,
        free: *cu_mem_free,
    };
    let mut device_needle_indices = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_needle_indices, output_indices_bytes) })?;
    let indices_guard = CudaDeviceAllocationGuard {
        ptr: device_needle_indices,
        free: *cu_mem_free,
    };
    let mut device_row_indices = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_row_indices, output_row_indices_bytes) })?;
    let row_indices_guard = CudaDeviceAllocationGuard {
        ptr: device_row_indices,
        free: *cu_mem_free,
    };
    let mut device_count = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_count, std::mem::size_of::<u32>()) })?;
    let count_guard = CudaDeviceAllocationGuard {
        ptr: device_count,
        free: *cu_mem_free,
    };
    check_cuda(unsafe { cu_memset_d8(count_guard.ptr, 0, std::mem::size_of::<u32>()) })?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_resident_i32_equal_any_project".as_ptr(),
        )
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut rows_arg = row_count;
    let mut needle_count_arg = needle_count_u32;
    let mut projection_count_arg = u32::try_from(projection_offsets.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(projection_offsets.len()))?;
    let mut filter_offset_arg = filter_offset;
    let mut projected_offsets = [0_u64; MAX_PROJECTIONS];
    for (idx, offset) in projection_offsets.iter().enumerate() {
        projected_offsets[idx] = *offset;
    }
    let mut needles_arg = needles_guard.ptr;
    let mut output_arg = values_guard.ptr;
    let mut indices_arg = indices_guard.ptr;
    let mut row_indices_arg = row_indices_guard.ptr;
    let mut count_arg = count_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut needle_count_arg as *mut u32).cast::<c_void>(),
        (&mut projection_count_arg as *mut u32).cast::<c_void>(),
        (&mut filter_offset_arg as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[0] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[1] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[2] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[3] as *mut u64).cast::<c_void>(),
        (&mut needles_arg as *mut u64).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut indices_arg as *mut u64).cast::<c_void>(),
        (&mut row_indices_arg as *mut u64).cast::<c_void>(),
        (&mut count_arg as *mut u64).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count_u32.div_ceil(threads_per_block);
    let mut start = std::ptr::null_mut();
    check_cuda(unsafe { cu_event_create(&mut start, 0) })?;
    let start_event_guard = CudaEventGuard {
        event: start,
        destroy: *cu_event_destroy,
    };
    let mut stop = std::ptr::null_mut();
    check_cuda(unsafe { cu_event_create(&mut stop, 0) })?;
    let stop_event_guard = CudaEventGuard {
        event: stop,
        destroy: *cu_event_destroy,
    };

    check_cuda(unsafe { cu_event_record(start_event_guard.event, std::ptr::null_mut()) })?;
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_event_record(stop_event_guard.event, std::ptr::null_mut()) })?;

    Ok(CudaI32EqualAnyProjectSubmission {
        projection_count: projection_offsets.len(),
        needles_len: needles.len(),
        row_count,
        context: resident.context(),
        values_guard,
        indices_guard,
        row_indices_guard,
        count_guard,
        _needles_guard: needles_guard,
        _module_guard: module_guard,
        start_event_guard,
        stop_event_guard,
        cu_ctx_set_current: *cu_ctx_set_current,
        cu_memcpy_dtoh: *cu_memcpy_dtoh,
        cu_event_synchronize: *cu_event_synchronize,
        cu_event_elapsed_time: *cu_event_elapsed_time,
    })
}

fn launch_cuda_resident_i32_equal_any_project_text<R: CudaResidentReadSource>(
    resident: &R,
    filter_offset: u64,
    needles: &[i32],
    projection_offsets: &[u64],
    text_offsets_byte_offset: u64,
    text_bytes_byte_offset: u64,
    text_bytes_len: u64,
    row_count: u64,
) -> Result<Vec<CudaI32TextBatchProjectionRow>, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    type CuCtxSetCurrent = unsafe extern "C" fn(*mut c_void) -> i32;

    const MAX_PROJECTIONS: usize = 4;
    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_equal_any_project_text(
    .param .u64 resident_ptr,
    .param .u64 row_count,
    .param .u32 needle_count,
    .param .u32 projection_count,
    .param .u64 filter_offset,
    .param .u64 projection_offset0,
    .param .u64 projection_offset1,
    .param .u64 projection_offset2,
    .param .u64 projection_offset3,
    .param .u64 text_offsets_offset,
    .param .u64 text_bytes_offset,
    .param .u32 text_bytes_len,
    .param .u64 needles_ptr,
    .param .u64 out_values_ptr,
    .param .u64 out_needle_indices_ptr,
    .param .u64 out_row_indices_ptr,
    .param .u64 out_text_starts_ptr,
    .param .u64 out_text_lens_ptr,
    .param .u64 out_text_bytes_ptr,
    .param .u64 out_count_ptr,
    .param .u64 out_text_count_ptr
)
{
    .reg .pred %p_out;
    .reg .pred %p_done;
    .reg .pred %p_match;
    .reg .pred %p_copy_done;
    .reg .u16 %byte_value;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %idx32;
    .reg .u32 %needle_count;
    .reg .u32 %projection_count;
    .reg .u32 %needle_idx;
    .reg .u32 %slot;
    .reg .u32 %text_start32;
    .reg .u32 %text_end32;
    .reg .u32 %text_len;
    .reg .u32 %text_slot;
    .reg .u32 %copy_idx;
    .reg .u32 %one;
    .reg .u64 %idx;
    .reg .u64 %rows;
    .reg .u64 %resident;
    .reg .u64 %filter_offset;
    .reg .u64 %projection_offset0;
    .reg .u64 %projection_offset1;
    .reg .u64 %projection_offset2;
    .reg .u64 %projection_offset3;
    .reg .u64 %text_offsets_offset;
    .reg .u64 %text_bytes_offset;
    .reg .u64 %needles;
    .reg .u64 %out_values;
    .reg .u64 %out_needle_indices;
    .reg .u64 %out_row_indices;
    .reg .u64 %out_text_starts;
    .reg .u64 %out_text_lens;
    .reg .u64 %out_text_bytes;
    .reg .u64 %out_count;
    .reg .u64 %out_text_count;
    .reg .u64 %row_byte;
    .reg .u64 %addr;
    .reg .u64 %addr2;
    .reg .u64 %needle_byte;
    .reg .u64 %slot64;
    .reg .u64 %projection_count64;
    .reg .u64 %base_slot;
    .reg .u64 %out_addr;
    .reg .u64 %text_offset_addr;
    .reg .u64 %text_start64;
    .reg .u64 %text_end64;
    .reg .u64 %text_slot64;
    .reg .u64 %copy64;
    .reg .s32 %row_value;
    .reg .s32 %needle_value;
    .reg .s32 %projection_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %rows, [row_count];
    ld.param.u32 %needle_count, [needle_count];
    ld.param.u32 %projection_count, [projection_count];
    ld.param.u64 %filter_offset, [filter_offset];
    ld.param.u64 %projection_offset0, [projection_offset0];
    ld.param.u64 %projection_offset1, [projection_offset1];
    ld.param.u64 %projection_offset2, [projection_offset2];
    ld.param.u64 %projection_offset3, [projection_offset3];
    ld.param.u64 %text_offsets_offset, [text_offsets_offset];
    ld.param.u64 %text_bytes_offset, [text_bytes_offset];
    ld.param.u64 %needles, [needles_ptr];
    ld.param.u64 %out_values, [out_values_ptr];
    ld.param.u64 %out_needle_indices, [out_needle_indices_ptr];
    ld.param.u64 %out_row_indices, [out_row_indices_ptr];
    ld.param.u64 %out_text_starts, [out_text_starts_ptr];
    ld.param.u64 %out_text_lens, [out_text_lens_ptr];
    ld.param.u64 %out_text_bytes, [out_text_bytes_ptr];
    ld.param.u64 %out_count, [out_count_ptr];
    ld.param.u64 %out_text_count, [out_text_count_ptr];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %idx32, %r_block, %r_block_dim, %r_tid;
    cvt.u64.u32 %idx, %idx32;

    setp.ge.u64 %p_out, %idx, %rows;
    @%p_out bra DONE;
    setp.eq.u32 %p_done, %needle_count, 0;
    @%p_done bra DONE;

    mul.lo.u64 %row_byte, %idx, 4;
    add.u64 %addr, %resident, %filter_offset;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %row_value, [%addr];

    mov.u32 %needle_idx, 0;
NEEDLE_LOOP:
    setp.ge.u32 %p_done, %needle_idx, %needle_count;
    @%p_done bra DONE;
    cvt.u64.u32 %needle_byte, %needle_idx;
    mul.lo.u64 %needle_byte, %needle_byte, 4;
    add.u64 %addr, %needles, %needle_byte;
    ld.global.s32 %needle_value, [%addr];
    setp.eq.s32 %p_match, %row_value, %needle_value;
    @%p_match bra MATCHED;
    add.u32 %needle_idx, %needle_idx, 1;
    bra NEEDLE_LOOP;

MATCHED:
    mul.lo.u64 %text_offset_addr, %idx, 8;
    add.u64 %addr, %resident, %text_offsets_offset;
    add.u64 %addr, %addr, %text_offset_addr;
    ld.global.u32 %text_start32, [%addr];
    add.u64 %addr, %addr, 8;
    ld.global.u32 %text_end32, [%addr];
    cvt.u64.u32 %text_start64, %text_start32;
    cvt.u64.u32 %text_end64, %text_end32;
    sub.u32 %text_len, %text_end32, %text_start32;

    mov.u32 %one, 1;
    atom.global.add.u32 %slot, [%out_count], %one;
    cvt.u64.u32 %slot64, %slot;
    atom.global.add.u32 %text_slot, [%out_text_count], %text_len;
    cvt.u64.u32 %text_slot64, %text_slot;

    mul.lo.u64 %out_addr, %slot64, 4;
    add.u64 %out_addr, %out_needle_indices, %out_addr;
    st.global.u32 [%out_addr], %needle_idx;

    mul.lo.u64 %out_addr, %slot64, 8;
    add.u64 %out_addr, %out_row_indices, %out_addr;
    st.global.u64 [%out_addr], %idx;

    mul.lo.u64 %out_addr, %slot64, 4;
    add.u64 %addr, %out_text_starts, %out_addr;
    st.global.u32 [%addr], %text_slot;
    add.u64 %addr, %out_text_lens, %out_addr;
    st.global.u32 [%addr], %text_len;

    setp.eq.u32 %p_done, %projection_count, 0;
    @%p_done bra COPY_TEXT;
    cvt.u64.u32 %projection_count64, %projection_count;
    mul.lo.u64 %base_slot, %slot64, %projection_count64;
    mul.lo.u64 %base_slot, %base_slot, 4;

    add.u64 %addr, %resident, %projection_offset0;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    st.global.s32 [%out_addr], %projection_value;

    setp.le.u32 %p_done, %projection_count, 1;
    @%p_done bra COPY_TEXT;
    add.u64 %addr, %resident, %projection_offset1;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 4;
    st.global.s32 [%out_addr], %projection_value;

    setp.le.u32 %p_done, %projection_count, 2;
    @%p_done bra COPY_TEXT;
    add.u64 %addr, %resident, %projection_offset2;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 8;
    st.global.s32 [%out_addr], %projection_value;

    setp.le.u32 %p_done, %projection_count, 3;
    @%p_done bra COPY_TEXT;
    add.u64 %addr, %resident, %projection_offset3;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 12;
    st.global.s32 [%out_addr], %projection_value;

COPY_TEXT:
    mov.u32 %copy_idx, 0;
COPY_LOOP:
    setp.ge.u32 %p_copy_done, %copy_idx, %text_len;
    @%p_copy_done bra DONE;
    cvt.u64.u32 %copy64, %copy_idx;
    add.u64 %addr, %resident, %text_bytes_offset;
    add.u64 %addr, %addr, %text_start64;
    add.u64 %addr, %addr, %copy64;
    ld.global.u8 %byte_value, [%addr];
    add.u64 %addr2, %out_text_bytes, %text_slot64;
    add.u64 %addr2, %addr2, %copy64;
    st.global.u8 [%addr2], %byte_value;
    add.u32 %copy_idx, %copy_idx, 1;
    bra COPY_LOOP;

DONE:
    ret;
}
"#;

    if needles.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if projection_offsets.len() > MAX_PROJECTIONS {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            projection_offsets.len(),
        ));
    }
    if row_count == 0 {
        return Ok(Vec::new());
    }
    let _text_bytes_len_u32 = u32::try_from(text_bytes_len)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let filter_bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| filter_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let text_offsets_end = row_count
        .checked_add(1)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u64>() as u64))
        .and_then(|bytes| text_offsets_byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let text_bytes_end = text_bytes_byte_offset
        .checked_add(text_bytes_len)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if filter_bytes > resident.metadata().allocated_bytes
        || text_offsets_end > resident.metadata().allocated_bytes
        || text_bytes_end > resident.metadata().allocated_bytes
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            filter_bytes.max(text_offsets_end).max(text_bytes_end) as usize,
        ));
    }
    for byte_offset in projection_offsets {
        let bytes = row_count
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .and_then(|bytes| byte_offset.checked_add(bytes))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if bytes > resident.metadata().allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
        }
    }
    let row_count_u32 = u32::try_from(row_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let needle_count_u32 = u32::try_from(needles.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(needles.len()))?;
    let output_cells = row_count
        .checked_mul(projection_offsets.len() as u64)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_bytes = usize::try_from(
        output_cells
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?
    .max(1);
    let output_indices_bytes = usize::try_from(
        row_count
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_row_indices_bytes = usize::try_from(
        row_count
            .checked_mul(std::mem::size_of::<u64>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_text_bytes = usize::try_from(text_bytes_len)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?
        .max(1);
    let needle_bytes = needles
        .len()
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memset_d8 = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| resident.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_set_current = unsafe {
        resident
            .lib()
            .get::<CuCtxSetCurrent>(b"cuCtxSetCurrent\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { cu_ctx_set_current(resident.context()) })?;

    let mut device_needles = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_needles, needle_bytes) })?;
    let needles_guard = CudaDeviceAllocationGuard {
        ptr: device_needles,
        free: *cu_mem_free,
    };
    check_cuda(unsafe {
        cu_memcpy_htod(
            needles_guard.ptr,
            needles.as_ptr().cast::<c_void>(),
            needle_bytes,
        )
    })?;

    let mut device_values = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_values, output_bytes) })?;
    let values_guard = CudaDeviceAllocationGuard {
        ptr: device_values,
        free: *cu_mem_free,
    };
    let mut device_needle_indices = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_needle_indices, output_indices_bytes) })?;
    let indices_guard = CudaDeviceAllocationGuard {
        ptr: device_needle_indices,
        free: *cu_mem_free,
    };
    let mut device_row_indices = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_row_indices, output_row_indices_bytes) })?;
    let row_indices_guard = CudaDeviceAllocationGuard {
        ptr: device_row_indices,
        free: *cu_mem_free,
    };
    let mut device_text_starts = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_text_starts, output_indices_bytes) })?;
    let text_starts_guard = CudaDeviceAllocationGuard {
        ptr: device_text_starts,
        free: *cu_mem_free,
    };
    let mut device_text_lens = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_text_lens, output_indices_bytes) })?;
    let text_lens_guard = CudaDeviceAllocationGuard {
        ptr: device_text_lens,
        free: *cu_mem_free,
    };
    let mut device_text_bytes = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_text_bytes, output_text_bytes) })?;
    let text_bytes_guard = CudaDeviceAllocationGuard {
        ptr: device_text_bytes,
        free: *cu_mem_free,
    };
    let mut device_count = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_count, std::mem::size_of::<u32>()) })?;
    let count_guard = CudaDeviceAllocationGuard {
        ptr: device_count,
        free: *cu_mem_free,
    };
    let mut device_text_count = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_text_count, std::mem::size_of::<u32>()) })?;
    let text_count_guard = CudaDeviceAllocationGuard {
        ptr: device_text_count,
        free: *cu_mem_free,
    };
    check_cuda(unsafe { cu_memset_d8(count_guard.ptr, 0, std::mem::size_of::<u32>()) })?;
    check_cuda(unsafe { cu_memset_d8(text_count_guard.ptr, 0, std::mem::size_of::<u32>()) })?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_resident_i32_equal_any_project_text".as_ptr(),
        )
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut rows_arg = row_count;
    let mut needle_count_arg = needle_count_u32;
    let mut projection_count_arg = u32::try_from(projection_offsets.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(projection_offsets.len()))?;
    let mut filter_offset_arg = filter_offset;
    let mut projected_offsets = [0_u64; MAX_PROJECTIONS];
    for (idx, offset) in projection_offsets.iter().enumerate() {
        projected_offsets[idx] = *offset;
    }
    let mut text_offsets_arg = text_offsets_byte_offset;
    let mut text_bytes_arg = text_bytes_byte_offset;
    let mut text_bytes_len_arg = _text_bytes_len_u32;
    let mut needles_arg = needles_guard.ptr;
    let mut output_arg = values_guard.ptr;
    let mut indices_arg = indices_guard.ptr;
    let mut row_indices_arg = row_indices_guard.ptr;
    let mut text_starts_arg = text_starts_guard.ptr;
    let mut text_lens_arg = text_lens_guard.ptr;
    let mut text_output_arg = text_bytes_guard.ptr;
    let mut count_arg = count_guard.ptr;
    let mut text_count_arg = text_count_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut needle_count_arg as *mut u32).cast::<c_void>(),
        (&mut projection_count_arg as *mut u32).cast::<c_void>(),
        (&mut filter_offset_arg as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[0] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[1] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[2] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[3] as *mut u64).cast::<c_void>(),
        (&mut text_offsets_arg as *mut u64).cast::<c_void>(),
        (&mut text_bytes_arg as *mut u64).cast::<c_void>(),
        (&mut text_bytes_len_arg as *mut u32).cast::<c_void>(),
        (&mut needles_arg as *mut u64).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut indices_arg as *mut u64).cast::<c_void>(),
        (&mut row_indices_arg as *mut u64).cast::<c_void>(),
        (&mut text_starts_arg as *mut u64).cast::<c_void>(),
        (&mut text_lens_arg as *mut u64).cast::<c_void>(),
        (&mut text_output_arg as *mut u64).cast::<c_void>(),
        (&mut count_arg as *mut u64).cast::<c_void>(),
        (&mut text_count_arg as *mut u64).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count_u32.div_ceil(threads_per_block);
    launch_with_optional_cuda_event_timing(resident, *cu_ctx_synchronize, || unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;

    let mut match_count = 0_u32;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut match_count as *mut u32).cast::<c_void>(),
            count_guard.ptr,
            std::mem::size_of::<u32>(),
        )
    })?;
    let mut compact_text_len = 0_u32;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut compact_text_len as *mut u32).cast::<c_void>(),
            text_count_guard.ptr,
            std::mem::size_of::<u32>(),
        )
    })?;
    let match_count = u64::from(match_count);
    if match_count > row_count || u64::from(compact_text_len) > text_bytes_len {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    let match_count_usize = usize::try_from(match_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let projection_count = projection_offsets.len();
    let mut values = vec![0_i32; match_count_usize.saturating_mul(projection_count)];
    if !values.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                values.as_mut_ptr().cast::<c_void>(),
                values_guard.ptr,
                values.len() * std::mem::size_of::<i32>(),
            )
        })?;
    }
    let mut needle_indices = vec![0_u32; match_count_usize];
    if !needle_indices.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                needle_indices.as_mut_ptr().cast::<c_void>(),
                indices_guard.ptr,
                needle_indices.len() * std::mem::size_of::<u32>(),
            )
        })?;
    }
    let mut row_indices = vec![0_u64; match_count_usize];
    if !row_indices.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                row_indices.as_mut_ptr().cast::<c_void>(),
                row_indices_guard.ptr,
                row_indices.len() * std::mem::size_of::<u64>(),
            )
        })?;
    }
    let mut text_starts = vec![0_u32; match_count_usize];
    if !text_starts.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                text_starts.as_mut_ptr().cast::<c_void>(),
                text_starts_guard.ptr,
                text_starts.len() * std::mem::size_of::<u32>(),
            )
        })?;
    }
    let mut text_lens = vec![0_u32; match_count_usize];
    if !text_lens.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                text_lens.as_mut_ptr().cast::<c_void>(),
                text_lens_guard.ptr,
                text_lens.len() * std::mem::size_of::<u32>(),
            )
        })?;
    }
    let mut text_bytes = vec![
        0_u8;
        usize::try_from(compact_text_len).map_err(|_| {
            CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
        })?
    ];
    if !text_bytes.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                text_bytes.as_mut_ptr().cast::<c_void>(),
                text_bytes_guard.ptr,
                text_bytes.len(),
            )
        })?;
    }

    drop(module_guard);
    drop(text_count_guard);
    drop(count_guard);
    drop(text_bytes_guard);
    drop(text_lens_guard);
    drop(text_starts_guard);
    drop(row_indices_guard);
    drop(indices_guard);
    drop(values_guard);
    drop(needles_guard);
    values
        .chunks_exact(projection_count)
        .zip(needle_indices)
        .zip(row_indices)
        .zip(text_starts)
        .zip(text_lens)
        .map(
            |((((row, needle_index), row_index), text_start), text_len)| {
                let needle_index = usize::try_from(needle_index)
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                if needle_index >= needles.len() {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(needle_index));
                }
                if row_index >= row_count {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
                }
                let text_start = usize::try_from(text_start)
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                let text_len = usize::try_from(text_len)
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                let text_end = text_start
                    .checked_add(text_len)
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                if text_end > text_bytes.len() {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(text_end));
                }
                let text = std::str::from_utf8(&text_bytes[text_start..text_end])
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(text_len))?
                    .to_string();
                Ok(CudaI32TextBatchProjectionRow {
                    needle_index,
                    row_index,
                    values: row.to_vec(),
                    text,
                })
            },
        )
        .collect()
}

fn launch_cuda_resident_i32_equal_row_indices(
    resident: &CudaResidentDeviceMemory,
    filters: &[(u64, i32)],
    row_count: u64,
) -> Result<Vec<u64>, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const MAX_FILTERS: usize = 4;
    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_equal_row_indices(
    .param .u64 resident_ptr,
    .param .u64 row_count,
    .param .u32 filter_count,
    .param .u64 offset0,
    .param .u64 offset1,
    .param .u64 offset2,
    .param .u64 offset3,
    .param .s32 needle0,
    .param .s32 needle1,
    .param .s32 needle2,
    .param .s32 needle3,
    .param .u64 out_indices_ptr,
    .param .u64 out_count_ptr
)
{
    .reg .pred %p_out;
    .reg .pred %p_done;
    .reg .pred %p_match;
    .reg .pred %p_check;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %idx32;
    .reg .u64 %idx;
    .reg .u64 %rows;
    .reg .u32 %filters;
    .reg .u64 %resident;
    .reg .u64 %offset0;
    .reg .u64 %offset1;
    .reg .u64 %offset2;
    .reg .u64 %offset3;
    .reg .s32 %needle0;
    .reg .s32 %needle1;
    .reg .s32 %needle2;
    .reg .s32 %needle3;
    .reg .u64 %out_indices;
    .reg .u64 %out_count;
    .reg .u64 %row_byte;
    .reg .u64 %addr;
    .reg .u32 %slot;
    .reg .u64 %out_addr;
    .reg .u64 %slot64;
    .reg .u32 %one;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %rows, [row_count];
    ld.param.u32 %filters, [filter_count];
    ld.param.u64 %offset0, [offset0];
    ld.param.u64 %offset1, [offset1];
    ld.param.u64 %offset2, [offset2];
    ld.param.u64 %offset3, [offset3];
    ld.param.s32 %needle0, [needle0];
    ld.param.s32 %needle1, [needle1];
    ld.param.s32 %needle2, [needle2];
    ld.param.s32 %needle3, [needle3];
    ld.param.u64 %out_indices, [out_indices_ptr];
    ld.param.u64 %out_count, [out_count_ptr];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %idx32, %r_block, %r_block_dim, %r_tid;
    cvt.u64.u32 %idx, %idx32;

    setp.ge.u64 %p_out, %idx, %rows;
    @%p_out bra DONE;

    setp.eq.u32 %p_done, %filters, 0;
    @%p_done bra DONE;

    mul.lo.u64 %row_byte, %idx, 4;

    add.u64 %addr, %resident, %offset0;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle0;
    @!%p_match bra DONE;

    setp.le.u32 %p_check, %filters, 1;
    @%p_check bra MATCHED;
    add.u64 %addr, %resident, %offset1;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle1;
    @!%p_match bra DONE;

    setp.le.u32 %p_check, %filters, 2;
    @%p_check bra MATCHED;
    add.u64 %addr, %resident, %offset2;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle2;
    @!%p_match bra DONE;

    setp.le.u32 %p_check, %filters, 3;
    @%p_check bra MATCHED;
    add.u64 %addr, %resident, %offset3;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle3;
    @!%p_match bra DONE;

MATCHED:
    mov.u32 %one, 1;
    atom.global.add.u32 %slot, [%out_count], %one;
    cvt.u64.u32 %slot64, %slot;
    mul.lo.u64 %out_addr, %slot64, 8;
    add.u64 %out_addr, %out_indices, %out_addr;
    st.global.u64 [%out_addr], %idx;

DONE:
    ret;
}
"#;

    if filters.is_empty() || filters.len() > MAX_FILTERS {
        return Err(CudaRuntimeProbeError::InvalidInputLength(filters.len()));
    }
    if row_count == 0 {
        return Ok(Vec::new());
    }
    for (byte_offset, _) in filters {
        let bytes = row_count
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .and_then(|bytes| byte_offset.checked_add(bytes))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if bytes > resident.metadata().allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
        }
    }
    let row_count_u32 = u32::try_from(row_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_bytes = usize::try_from(
        row_count
            .checked_mul(std::mem::size_of::<u64>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memset_d8 = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| resident.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut device_indices = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_indices, output_bytes) })?;
    let indices_guard = CudaDeviceAllocationGuard {
        ptr: device_indices,
        free: *cu_mem_free,
    };
    let mut device_count = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_count, std::mem::size_of::<u32>()) })?;
    let count_guard = CudaDeviceAllocationGuard {
        ptr: device_count,
        free: *cu_mem_free,
    };
    check_cuda(unsafe { cu_memset_d8(count_guard.ptr, 0, std::mem::size_of::<u32>()) })?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_resident_i32_equal_row_indices".as_ptr(),
        )
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut rows_arg = row_count;
    let mut filter_count_arg = u32::try_from(filters.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(filters.len()))?;
    let mut offsets = [0_u64; MAX_FILTERS];
    let mut needles = [0_i32; MAX_FILTERS];
    for (idx, (offset, needle)) in filters.iter().enumerate() {
        offsets[idx] = *offset;
        needles[idx] = *needle;
    }
    let mut output_arg = indices_guard.ptr;
    let mut count_arg = count_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut filter_count_arg as *mut u32).cast::<c_void>(),
        (&mut offsets[0] as *mut u64).cast::<c_void>(),
        (&mut offsets[1] as *mut u64).cast::<c_void>(),
        (&mut offsets[2] as *mut u64).cast::<c_void>(),
        (&mut offsets[3] as *mut u64).cast::<c_void>(),
        (&mut needles[0] as *mut i32).cast::<c_void>(),
        (&mut needles[1] as *mut i32).cast::<c_void>(),
        (&mut needles[2] as *mut i32).cast::<c_void>(),
        (&mut needles[3] as *mut i32).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut count_arg as *mut u64).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count_u32.div_ceil(threads_per_block);
    launch_with_optional_cuda_event_timing(resident, *cu_ctx_synchronize, || unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;

    let mut match_count = 0_u32;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut match_count as *mut u32).cast::<c_void>(),
            count_guard.ptr,
            std::mem::size_of::<u32>(),
        )
    })?;
    let match_count = u64::from(match_count);
    if match_count > row_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    let match_count_usize = usize::try_from(match_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut indices = vec![0_u64; match_count_usize];
    if match_count_usize > 0 {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                indices.as_mut_ptr().cast::<c_void>(),
                indices_guard.ptr,
                match_count_usize * std::mem::size_of::<u64>(),
            )
        })?;
    }

    drop(module_guard);
    drop(count_guard);
    drop(indices_guard);
    Ok(indices)
}

fn launch_cuda_resident_i32_between_row_indices(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
    lower_inclusive: i32,
    upper_inclusive: i32,
) -> Result<Vec<u64>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;

    if lower_inclusive > upper_inclusive || row_count == 0 {
        return Ok(Vec::new());
    }
    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }
    let bytes_usize = usize::try_from(bytes - byte_offset)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut raw_values = vec![0_u8; bytes_usize];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            raw_values.as_mut_ptr().cast::<c_void>(),
            resident.device_ptr() + byte_offset,
            bytes_usize,
        )
    })?;

    let mut indices = Vec::new();
    for (idx, chunk) in raw_values
        .chunks_exact(std::mem::size_of::<i32>())
        .enumerate()
    {
        let value = i32::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(raw_values.len()))?,
        );
        if value >= lower_inclusive && value <= upper_inclusive {
            indices.push(
                u64::try_from(idx)
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
            );
        }
    }
    Ok(indices)
}

fn copy_cuda_resident_text_rows(
    resident: &CudaResidentDeviceMemory,
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    bytes_len: u64,
    row_indices: &[u64],
) -> Result<Vec<String>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;

    if row_indices.is_empty() {
        return Ok(Vec::new());
    }

    let min_row_idx = row_indices.iter().copied().min().unwrap_or(0);
    let max_row_idx = row_indices.iter().copied().max().unwrap_or(0);
    let offset_count = max_row_idx
        .checked_sub(min_row_idx)
        .and_then(|span| span.checked_add(2))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let offsets_offset = min_row_idx
        .checked_mul(std::mem::size_of::<u64>() as u64)
        .and_then(|offset| offsets_byte_offset.checked_add(offset))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let offsets_end = offsets_byte_offset
        .checked_add(
            max_row_idx
                .checked_add(2)
                .and_then(|count| count.checked_mul(std::mem::size_of::<u64>() as u64))
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        )
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes_end = bytes_byte_offset
        .checked_add(bytes_len)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if offsets_end > resident.metadata().allocated_bytes
        || bytes_end > resident.metadata().allocated_bytes
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            offsets_end.max(bytes_end) as usize,
        ));
    }

    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut offsets = vec![
        0_u64;
        usize::try_from(offset_count).map_err(|_| {
            CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
        })?
    ];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            offsets.as_mut_ptr().cast::<c_void>(),
            resident.device_ptr() + offsets_offset,
            offsets
                .len()
                .checked_mul(std::mem::size_of::<u64>())
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        )
    })?;

    let mut spans = Vec::with_capacity(row_indices.len());
    let mut min_text_start = u64::MAX;
    let mut max_text_end = 0_u64;
    for row_idx in row_indices {
        let offset_idx = row_idx
            .checked_sub(min_row_idx)
            .and_then(|idx| usize::try_from(idx).ok())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let start = offsets[offset_idx];
        let end = offsets[offset_idx + 1];
        if start > end || end > bytes_len {
            return Err(CudaRuntimeProbeError::InvalidInputLength(end as usize));
        }
        min_text_start = min_text_start.min(start);
        max_text_end = max_text_end.max(end);
        spans.push((start, end));
    }

    let text_span_len = max_text_end
        .checked_sub(min_text_start)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut text_bytes = vec![
        0_u8;
        usize::try_from(text_span_len).map_err(|_| {
            CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
        })?
    ];
    if text_span_len > 0 {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                text_bytes.as_mut_ptr().cast::<c_void>(),
                resident.device_ptr() + bytes_byte_offset + min_text_start,
                text_bytes.len(),
            )
        })?;
    }

    let mut values = Vec::with_capacity(row_indices.len());
    for (start, end) in spans {
        let value_len = usize::try_from(end - start)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let value_start = start
            .checked_sub(min_text_start)
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let value_end = value_start
            .checked_add(value_len)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let value = std::str::from_utf8(&text_bytes[value_start..value_end])
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(value_len))?;
        values.push(value.to_string());
    }
    Ok(values)
}

fn launch_cuda_resident_i32_sum(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
) -> Result<i64, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_sum(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .u64 out_ptr
)
{
    .reg .pred %p_done;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %stride;
    .reg .u64 %addr;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %thread;
    .reg .u32 %grid_dim;
    .reg .u64 %wide_block;
    .reg .u64 %wide_thread;
    .reg .u64 %wide_block_dim;
    .reg .u64 %wide_grid_dim;
    .reg .u64 %sum_bits;
    .reg .u64 %ignored;
    .reg .s64 %sum;
    .reg .s64 %wide;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.u64 %out, [out_ptr];

    add.u64 %base, %resident, %offset;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mov.u32 %thread, %tid.x;
    mov.u32 %grid_dim, %nctaid.x;
    cvt.u64.u32 %wide_block, %r_block;
    cvt.u64.u32 %wide_thread, %thread;
    cvt.u64.u32 %wide_block_dim, %r_block_dim;
    cvt.u64.u32 %wide_grid_dim, %grid_dim;
    mul.lo.u64 %idx, %wide_block, %wide_block_dim;
    add.u64 %idx, %idx, %wide_thread;
    mul.lo.u64 %stride, %wide_grid_dim, %wide_block_dim;
    mov.s64 %sum, 0;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %addr, %idx, 4;
    add.u64 %addr, %base, %addr;
    ld.global.s32 %r_value, [%addr];
    cvt.s64.s32 %wide, %r_value;
    add.s64 %sum, %sum, %wide;
    add.u64 %idx, %idx, %stride;
    bra loop;

done:
    cvt.u64.s64 %sum_bits, %sum;
    atom.global.add.u64 %ignored, [%out], %sum_bits;
    ret;
}
"#;

    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memset_d8 = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| resident.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut device_output = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_output, std::mem::size_of::<i64>()) })?;
    let allocation_guard = CudaDeviceAllocationGuard {
        ptr: device_output,
        free: *cu_mem_free,
    };
    check_cuda(unsafe { cu_memset_d8(allocation_guard.ptr, 0, std::mem::size_of::<i64>()) })?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(&mut function, module, c"gpu_db_resident_i32_sum".as_ptr())
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut offset_arg = byte_offset;
    let mut rows_arg = row_count;
    let mut output_arg = allocation_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut offset_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
    ];
    let block_dim = 256_u32;
    let grid_dim = if row_count == 0 {
        1
    } else {
        row_count.div_ceil(u64::from(block_dim)).min(1024) as u32
    };
    launch_with_optional_cuda_event_timing(resident, *cu_ctx_synchronize, || unsafe {
        cu_launch_kernel(
            function,
            grid_dim,
            1,
            1,
            block_dim,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;

    let mut output = 0_i64;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut output as *mut i64).cast::<c_void>(),
            allocation_guard.ptr,
            std::mem::size_of::<i64>(),
        )
    })?;
    drop(module_guard);
    drop(allocation_guard);
    Ok(output)
}

#[repr(C)]
#[derive(Default)]
struct CudaI32StatsRaw {
    count: u64,
    sum: i64,
    min: i32,
    max: i32,
}

fn launch_cuda_resident_i32_between_stats(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
    lower_inclusive: i32,
    upper_inclusive: i32,
) -> Result<CudaI32Stats, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_between_stats(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .s32 lower_inclusive,
    .param .s32 upper_inclusive,
    .param .u64 out_ptr
)
{
    .reg .pred %p_done;
    .reg .pred %p_ge_lower;
    .reg .pred %p_le_upper;
    .reg .pred %p_match;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %stride;
    .reg .u64 %addr;
    .reg .u64 %count_addr;
    .reg .u64 %sum_addr;
    .reg .u64 %min_addr;
    .reg .u64 %max_addr;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %thread;
    .reg .u32 %grid_dim;
    .reg .u64 %wide_block;
    .reg .u64 %wide_thread;
    .reg .u64 %wide_block_dim;
    .reg .u64 %wide_grid_dim;
    .reg .u64 %count;
    .reg .u64 %sum_bits;
    .reg .u64 %ignored64;
    .reg .s32 %ignored32;
    .reg .s64 %sum;
    .reg .s64 %wide;
    .reg .s32 %r_value;
    .reg .s32 %lower;
    .reg .s32 %upper;
    .reg .s32 %min;
    .reg .s32 %max;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.s32 %lower, [lower_inclusive];
    ld.param.s32 %upper, [upper_inclusive];
    ld.param.u64 %out, [out_ptr];

    add.u64 %base, %resident, %offset;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mov.u32 %thread, %tid.x;
    mov.u32 %grid_dim, %nctaid.x;
    cvt.u64.u32 %wide_block, %r_block;
    cvt.u64.u32 %wide_thread, %thread;
    cvt.u64.u32 %wide_block_dim, %r_block_dim;
    cvt.u64.u32 %wide_grid_dim, %grid_dim;
    mul.lo.u64 %idx, %wide_block, %wide_block_dim;
    add.u64 %idx, %idx, %wide_thread;
    mul.lo.u64 %stride, %wide_grid_dim, %wide_block_dim;
    mov.u64 %count, 0;
    mov.s64 %sum, 0;
    mov.s32 %min, 2147483647;
    mov.s32 %max, -2147483648;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %addr, %idx, 4;
    add.u64 %addr, %base, %addr;
    ld.global.s32 %r_value, [%addr];
    setp.ge.s32 %p_ge_lower, %r_value, %lower;
    setp.le.s32 %p_le_upper, %r_value, %upper;
    and.pred %p_match, %p_ge_lower, %p_le_upper;
    @!%p_match bra next;

    cvt.s64.s32 %wide, %r_value;
    add.s64 %sum, %sum, %wide;
    add.u64 %count, %count, 1;
    min.s32 %min, %min, %r_value;
    max.s32 %max, %max, %r_value;

next:
    add.u64 %idx, %idx, %stride;
    bra loop;

done:
    setp.eq.u64 %p_done, %count, 0;
    @%p_done bra ret_done;
    mov.u64 %count_addr, %out;
    atom.global.add.u64 %ignored64, [%count_addr], %count;
    add.u64 %sum_addr, %out, 8;
    cvt.u64.s64 %sum_bits, %sum;
    atom.global.add.u64 %ignored64, [%sum_addr], %sum_bits;
    add.u64 %min_addr, %out, 16;
    atom.global.min.s32 %ignored32, [%min_addr], %min;
    add.u64 %max_addr, %out, 20;
    atom.global.max.s32 %ignored32, [%max_addr], %max;

ret_done:
    ret;
}
"#;

    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut device_output = 0_u64;
    check_cuda(unsafe {
        cu_mem_alloc(&mut device_output, std::mem::size_of::<CudaI32StatsRaw>())
    })?;
    let output_guard = CudaDeviceAllocationGuard {
        ptr: device_output,
        free: *cu_mem_free,
    };
    let initial = CudaI32StatsRaw {
        count: 0,
        sum: 0,
        min: i32::MAX,
        max: i32::MIN,
    };
    check_cuda(unsafe {
        cu_memcpy_htod(
            output_guard.ptr,
            (&initial as *const CudaI32StatsRaw).cast::<c_void>(),
            std::mem::size_of::<CudaI32StatsRaw>(),
        )
    })?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_resident_i32_between_stats".as_ptr(),
        )
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut offset_arg = byte_offset;
    let mut rows_arg = row_count;
    let mut lower_arg = lower_inclusive;
    let mut upper_arg = upper_inclusive;
    let mut output_arg = output_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut offset_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut lower_arg as *mut i32).cast::<c_void>(),
        (&mut upper_arg as *mut i32).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
    ];
    let block_dim = 256_u32;
    let grid_dim = if row_count == 0 {
        1
    } else {
        row_count.div_ceil(u64::from(block_dim)).min(1024) as u32
    };
    launch_with_optional_cuda_event_timing(resident, *cu_ctx_synchronize, || unsafe {
        cu_launch_kernel(
            function,
            grid_dim,
            1,
            1,
            block_dim,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;

    let mut raw = CudaI32StatsRaw::default();
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut raw as *mut CudaI32StatsRaw).cast::<c_void>(),
            output_guard.ptr,
            std::mem::size_of::<CudaI32StatsRaw>(),
        )
    })?;
    drop(module_guard);
    drop(output_guard);

    Ok(CudaI32Stats {
        count: raw.count,
        sum: raw.sum,
        min: (raw.count > 0).then_some(raw.min),
        max: (raw.count > 0).then_some(raw.max),
    })
}

fn launch_cuda_resident_i32_project(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
) -> Result<Vec<i32>, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_project(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .u64 out_values_ptr,
    .param .u64 out_count_ptr
)
{
    .reg .pred %p_done;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out_values;
    .reg .u64 %out_count;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %input_addr;
    .reg .u64 %output_addr;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.u64 %out_values, [out_values_ptr];
    ld.param.u64 %out_count, [out_count_ptr];

    add.u64 %base, %resident, %offset;
    mov.u64 %idx, 0;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %input_addr, %idx, 4;
    add.u64 %input_addr, %base, %input_addr;
    ld.global.s32 %r_value, [%input_addr];
    mul.lo.u64 %output_addr, %idx, 4;
    add.u64 %output_addr, %out_values, %output_addr;
    st.global.s32 [%output_addr], %r_value;
    add.u64 %idx, %idx, 1;
    bra loop;

done:
    st.global.u64 [%out_count], %rows;
    ret;
}
"#;

    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }
    if row_count == 0 {
        return Ok(Vec::new());
    }

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let value_bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut device_values = 0_u64;
    let mut device_count = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_values, value_bytes) })?;
    let values_guard = CudaDeviceAllocationGuard {
        ptr: device_values,
        free: *cu_mem_free,
    };
    check_cuda(unsafe { cu_mem_alloc(&mut device_count, std::mem::size_of::<u64>()) })?;
    let count_guard = CudaDeviceAllocationGuard {
        ptr: device_count,
        free: *cu_mem_free,
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_resident_i32_project".as_ptr(),
        )
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut offset_arg = byte_offset;
    let mut rows_arg = row_count;
    let mut values_arg = values_guard.ptr;
    let mut count_arg = count_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut offset_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut values_arg as *mut u64).cast::<c_void>(),
        (&mut count_arg as *mut u64).cast::<c_void>(),
    ];
    launch_with_optional_cuda_event_timing(resident, *cu_ctx_synchronize, || unsafe {
        cu_launch_kernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;

    let mut copied_count = 0_u64;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut copied_count as *mut u64).cast::<c_void>(),
            count_guard.ptr,
            std::mem::size_of::<u64>(),
        )
    })?;
    let copied_len = usize::try_from(copied_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if copied_count > row_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(copied_len));
    }
    let mut values = vec![0_i32; copied_len];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            values.as_mut_ptr().cast::<c_void>(),
            values_guard.ptr,
            copied_len * std::mem::size_of::<i32>(),
        )
    })?;

    drop(module_guard);
    drop(count_guard);
    drop(values_guard);
    Ok(values)
}

fn launch_cuda_resident_i32_grouped_stats(
    resident: &CudaResidentDeviceMemory,
    group_byte_offset: u64,
    value_byte_offset: u64,
    filter: Option<(u64, i32, CudaI32Comparison)>,
    row_count: u64,
) -> Result<Vec<CudaI32GroupedStats>, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_grouped_stats(
    .param .u64 resident_ptr,
    .param .u64 group_byte_offset,
    .param .u64 value_byte_offset,
    .param .u64 filter_byte_offset,
    .param .u64 row_count,
    .param .s32 needle,
    .param .u32 comparison,
    .param .u64 out_groups_ptr,
    .param .u64 out_counts_ptr,
    .param .u64 out_sums_ptr,
    .param .u64 out_mins_ptr,
    .param .u64 out_maxs_ptr,
    .param .u64 out_count_ptr
)
{
    .reg .pred %p_done;
    .reg .pred %p_found;
    .reg .pred %p_scan_done;
    .reg .pred %p_same;
    .reg .u64 %resident;
    .reg .u64 %group_offset;
    .reg .u64 %r_value_offset;
    .reg .u64 %filter_offset;
    .reg .u64 %rows;
    .reg .u64 %out_groups;
    .reg .u64 %out_counts;
    .reg .u64 %out_sums;
    .reg .u64 %out_mins;
    .reg .u64 %out_maxs;
    .reg .u64 %out_count;
    .reg .u64 %group_base;
    .reg .u64 %r_value_base;
    .reg .u64 %filter_base;
    .reg .u64 %idx;
    .reg .u64 %scan;
    .reg .u64 %group_count;
    .reg .u64 %input_addr;
    .reg .u64 %output_addr;
    .reg .s32 %group_value;
    .reg .s32 %r_value;
    .reg .s32 %filter_value;
    .reg .s32 %needle;
    .reg .u32 %comparison;
    .reg .s32 %existing_group;
    .reg .s64 %r_value_wide;
    .reg .u64 %existing_count;
    .reg .u64 %new_count;
    .reg .s64 %existing_sum;
    .reg .s64 %new_sum;
    .reg .s32 %existing_min;
    .reg .s32 %existing_max;
    .reg .pred %p_less;
    .reg .pred %p_greater;
    .reg .pred %p_check;
    .reg .pred %p_match;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %group_offset, [group_byte_offset];
    ld.param.u64 %r_value_offset, [value_byte_offset];
    ld.param.u64 %filter_offset, [filter_byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.s32 %needle, [needle];
    ld.param.u32 %comparison, [comparison];
    ld.param.u64 %out_groups, [out_groups_ptr];
    ld.param.u64 %out_counts, [out_counts_ptr];
    ld.param.u64 %out_sums, [out_sums_ptr];
    ld.param.u64 %out_mins, [out_mins_ptr];
    ld.param.u64 %out_maxs, [out_maxs_ptr];
    ld.param.u64 %out_count, [out_count_ptr];

    add.u64 %group_base, %resident, %group_offset;
    add.u64 %r_value_base, %resident, %r_value_offset;
    add.u64 %filter_base, %resident, %filter_offset;
    mov.u64 %idx, 0;
    mov.u64 %group_count, 0;

row_loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;

    setp.eq.u32 %p_check, %comparison, 0;
    @%p_check bra predicate_pass;
    mul.lo.u64 %input_addr, %idx, 4;
    add.u64 %input_addr, %filter_base, %input_addr;
    ld.global.s32 %filter_value, [%input_addr];
    mov.pred %p_match, 0;
    setp.eq.u32 %p_check, %comparison, 1;
    @%p_check bra cmp_lt;
    setp.eq.u32 %p_check, %comparison, 2;
    @%p_check bra cmp_lte;
    setp.eq.u32 %p_check, %comparison, 3;
    @%p_check bra cmp_gt;
    setp.eq.u32 %p_check, %comparison, 4;
    @%p_check bra cmp_gte;
    bra next_row;
cmp_lt:
    setp.lt.s32 %p_match, %filter_value, %needle;
    bra predicate_checked;
cmp_lte:
    setp.le.s32 %p_match, %filter_value, %needle;
    bra predicate_checked;
cmp_gt:
    setp.gt.s32 %p_match, %filter_value, %needle;
    bra predicate_checked;
cmp_gte:
    setp.ge.s32 %p_match, %filter_value, %needle;
predicate_checked:
    @!%p_match bra next_row;

predicate_pass:
    mul.lo.u64 %input_addr, %idx, 4;
    add.u64 %input_addr, %group_base, %input_addr;
    ld.global.s32 %group_value, [%input_addr];
    mul.lo.u64 %input_addr, %idx, 4;
    add.u64 %input_addr, %r_value_base, %input_addr;
    ld.global.s32 %r_value, [%input_addr];
    cvt.s64.s32 %r_value_wide, %r_value;

    mov.u64 %scan, 0;
    mov.pred %p_found, 0;

scan_loop:
    setp.ge.u64 %p_scan_done, %scan, %group_count;
    @%p_scan_done bra insert_or_next;
    mul.lo.u64 %output_addr, %scan, 4;
    add.u64 %output_addr, %out_groups, %output_addr;
    ld.global.s32 %existing_group, [%output_addr];
    setp.eq.s32 %p_same, %existing_group, %group_value;
    @!%p_same bra scan_next;

    mul.lo.u64 %output_addr, %scan, 8;
    add.u64 %output_addr, %out_counts, %output_addr;
    ld.global.u64 %existing_count, [%output_addr];
    add.u64 %new_count, %existing_count, 1;
    st.global.u64 [%output_addr], %new_count;

    mul.lo.u64 %output_addr, %scan, 8;
    add.u64 %output_addr, %out_sums, %output_addr;
    ld.global.s64 %existing_sum, [%output_addr];
    add.s64 %new_sum, %existing_sum, %r_value_wide;
    st.global.s64 [%output_addr], %new_sum;

    mul.lo.u64 %output_addr, %scan, 4;
    add.u64 %output_addr, %out_mins, %output_addr;
    ld.global.s32 %existing_min, [%output_addr];
    setp.lt.s32 %p_less, %r_value, %existing_min;
    @!%p_less bra keep_min;
    st.global.s32 [%output_addr], %r_value;
keep_min:

    mul.lo.u64 %output_addr, %scan, 4;
    add.u64 %output_addr, %out_maxs, %output_addr;
    ld.global.s32 %existing_max, [%output_addr];
    setp.gt.s32 %p_greater, %r_value, %existing_max;
    @!%p_greater bra keep_max;
    st.global.s32 [%output_addr], %r_value;
keep_max:

    mov.pred %p_found, 1;
    bra next_row;

scan_next:
    add.u64 %scan, %scan, 1;
    bra scan_loop;

insert_or_next:
    @%p_found bra next_row;
    mul.lo.u64 %output_addr, %group_count, 4;
    add.u64 %output_addr, %out_groups, %output_addr;
    st.global.s32 [%output_addr], %group_value;
    mul.lo.u64 %output_addr, %group_count, 8;
    add.u64 %output_addr, %out_counts, %output_addr;
    mov.u64 %new_count, 1;
    st.global.u64 [%output_addr], %new_count;
    mul.lo.u64 %output_addr, %group_count, 8;
    add.u64 %output_addr, %out_sums, %output_addr;
    st.global.s64 [%output_addr], %r_value_wide;
    mul.lo.u64 %output_addr, %group_count, 4;
    add.u64 %output_addr, %out_mins, %output_addr;
    st.global.s32 [%output_addr], %r_value;
    mul.lo.u64 %output_addr, %group_count, 4;
    add.u64 %output_addr, %out_maxs, %output_addr;
    st.global.s32 [%output_addr], %r_value;
    add.u64 %group_count, %group_count, 1;

next_row:
    add.u64 %idx, %idx, 1;
    bra row_loop;

done:
    st.global.u64 [%out_count], %group_count;
    ret;
}
"#;

    let group_bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| group_byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let value_bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| value_byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if group_bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            group_bytes as usize,
        ));
    }
    if value_bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            value_bytes as usize,
        ));
    }
    let (filter_byte_offset, needle, comparison_code) =
        if let Some((filter_byte_offset, needle, comparison)) = filter {
            let filter_bytes = row_count
                .checked_mul(std::mem::size_of::<i32>() as u64)
                .and_then(|bytes| filter_byte_offset.checked_add(bytes))
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if filter_bytes > resident.metadata().allocated_bytes {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    filter_bytes as usize,
                ));
            }
            (filter_byte_offset, needle, comparison.code())
        } else {
            (group_byte_offset, 0, 0)
        };
    if row_count == 0 {
        return Ok(Vec::new());
    }

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let group_output_bytes = usize::try_from(
        row_count
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let sum_output_bytes = usize::try_from(
        row_count
            .checked_mul(std::mem::size_of::<i64>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut device_groups = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_groups, group_output_bytes) })?;
    let groups_guard = CudaDeviceAllocationGuard {
        ptr: device_groups,
        free: *cu_mem_free,
    };
    let counts_output_bytes = sum_output_bytes;
    let mut device_counts = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_counts, counts_output_bytes) })?;
    let counts_guard = CudaDeviceAllocationGuard {
        ptr: device_counts,
        free: *cu_mem_free,
    };
    let mut device_sums = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_sums, sum_output_bytes) })?;
    let sums_guard = CudaDeviceAllocationGuard {
        ptr: device_sums,
        free: *cu_mem_free,
    };
    let mut device_mins = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mins, group_output_bytes) })?;
    let mins_guard = CudaDeviceAllocationGuard {
        ptr: device_mins,
        free: *cu_mem_free,
    };
    let mut device_maxs = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_maxs, group_output_bytes) })?;
    let maxs_guard = CudaDeviceAllocationGuard {
        ptr: device_maxs,
        free: *cu_mem_free,
    };
    let mut device_count = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_count, std::mem::size_of::<u64>()) })?;
    let count_guard = CudaDeviceAllocationGuard {
        ptr: device_count,
        free: *cu_mem_free,
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_resident_i32_grouped_stats".as_ptr(),
        )
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut group_offset_arg = group_byte_offset;
    let mut value_offset_arg = value_byte_offset;
    let mut filter_offset_arg = filter_byte_offset;
    let mut rows_arg = row_count;
    let mut needle_arg = needle;
    let mut comparison_arg = comparison_code;
    let mut groups_arg = groups_guard.ptr;
    let mut counts_arg = counts_guard.ptr;
    let mut sums_arg = sums_guard.ptr;
    let mut mins_arg = mins_guard.ptr;
    let mut maxs_arg = maxs_guard.ptr;
    let mut count_arg = count_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut group_offset_arg as *mut u64).cast::<c_void>(),
        (&mut value_offset_arg as *mut u64).cast::<c_void>(),
        (&mut filter_offset_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut needle_arg as *mut i32).cast::<c_void>(),
        (&mut comparison_arg as *mut u32).cast::<c_void>(),
        (&mut groups_arg as *mut u64).cast::<c_void>(),
        (&mut counts_arg as *mut u64).cast::<c_void>(),
        (&mut sums_arg as *mut u64).cast::<c_void>(),
        (&mut mins_arg as *mut u64).cast::<c_void>(),
        (&mut maxs_arg as *mut u64).cast::<c_void>(),
        (&mut count_arg as *mut u64).cast::<c_void>(),
    ];
    launch_with_optional_cuda_event_timing(resident, *cu_ctx_synchronize, || unsafe {
        cu_launch_kernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;

    let mut output_count = 0_u64;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut output_count as *mut u64).cast::<c_void>(),
            count_guard.ptr,
            std::mem::size_of::<u64>(),
        )
    })?;
    if output_count > row_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(output_count).unwrap_or(usize::MAX),
        ));
    }

    let output_len = usize::try_from(output_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut groups = vec![0_i32; output_len];
    let mut counts = vec![0_u64; output_len];
    let mut sums = vec![0_i64; output_len];
    let mut mins = vec![0_i32; output_len];
    let mut maxs = vec![0_i32; output_len];
    if output_len > 0 {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                groups.as_mut_ptr().cast::<c_void>(),
                groups_guard.ptr,
                output_len * std::mem::size_of::<i32>(),
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                counts.as_mut_ptr().cast::<c_void>(),
                counts_guard.ptr,
                output_len * std::mem::size_of::<u64>(),
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                sums.as_mut_ptr().cast::<c_void>(),
                sums_guard.ptr,
                output_len * std::mem::size_of::<i64>(),
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                mins.as_mut_ptr().cast::<c_void>(),
                mins_guard.ptr,
                output_len * std::mem::size_of::<i32>(),
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                maxs.as_mut_ptr().cast::<c_void>(),
                maxs_guard.ptr,
                output_len * std::mem::size_of::<i32>(),
            )
        })?;
    }

    drop(module_guard);
    drop(count_guard);
    drop(maxs_guard);
    drop(mins_guard);
    drop(sums_guard);
    drop(counts_guard);
    drop(groups_guard);
    Ok(groups
        .into_iter()
        .zip(counts)
        .zip(sums)
        .zip(mins)
        .zip(maxs)
        .map(|((((group, count), sum), min), max)| CudaI32GroupedStats {
            group,
            count,
            sum,
            min,
            max,
        })
        .collect())
}

fn launch_cuda_resident_i32_compare_project(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
    needle: i32,
    comparison: CudaI32Comparison,
) -> Result<Vec<i32>, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_compare_project(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .s32 needle,
    .param .u32 comparison,
    .param .u64 out_values_ptr,
    .param .u64 out_count_ptr
)
{
    .reg .pred %p_done;
    .reg .pred %p_lt;
    .reg .pred %p_lte;
    .reg .pred %p_gt;
    .reg .pred %p_gte;
    .reg .pred %p_code_lt;
    .reg .pred %p_code_lte;
    .reg .pred %p_code_gt;
    .reg .pred %p_code_gte;
    .reg .pred %p_match;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out_values;
    .reg .u64 %out_count;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %input_addr;
    .reg .u64 %output_addr;
    .reg .u64 %matches;
    .reg .u32 %comparison;
    .reg .s32 %needle;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.s32 %needle, [needle];
    ld.param.u32 %comparison, [comparison];
    ld.param.u64 %out_values, [out_values_ptr];
    ld.param.u64 %out_count, [out_count_ptr];

    add.u64 %base, %resident, %offset;
    mov.u64 %idx, 0;
    mov.u64 %matches, 0;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %input_addr, %idx, 4;
    add.u64 %input_addr, %base, %input_addr;
    ld.global.s32 %r_value, [%input_addr];
    setp.lt.s32 %p_lt, %r_value, %needle;
    setp.le.s32 %p_lte, %r_value, %needle;
    setp.gt.s32 %p_gt, %r_value, %needle;
    setp.ge.s32 %p_gte, %r_value, %needle;
    setp.eq.u32 %p_code_lt, %comparison, 1;
    setp.eq.u32 %p_code_lte, %comparison, 2;
    setp.eq.u32 %p_code_gt, %comparison, 3;
    setp.eq.u32 %p_code_gte, %comparison, 4;
    mov.pred %p_match, 0;
    and.pred %p_lt, %p_lt, %p_code_lt;
    or.pred %p_match, %p_match, %p_lt;
    and.pred %p_lte, %p_lte, %p_code_lte;
    or.pred %p_match, %p_match, %p_lte;
    and.pred %p_gt, %p_gt, %p_code_gt;
    or.pred %p_match, %p_match, %p_gt;
    and.pred %p_gte, %p_gte, %p_code_gte;
    or.pred %p_match, %p_match, %p_gte;
    @!%p_match bra next;
    mul.lo.u64 %output_addr, %matches, 4;
    add.u64 %output_addr, %out_values, %output_addr;
    st.global.s32 [%output_addr], %r_value;
    add.u64 %matches, %matches, 1;

next:
    add.u64 %idx, %idx, 1;
    bra loop;

done:
    st.global.u64 [%out_count], %matches;
    ret;
}
"#;

    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }
    if row_count == 0 {
        return Ok(Vec::new());
    }

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let values_bytes = usize::try_from(
        row_count
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut device_values = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_values, values_bytes) })?;
    let values_guard = CudaDeviceAllocationGuard {
        ptr: device_values,
        free: *cu_mem_free,
    };
    let mut device_count = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_count, std::mem::size_of::<u64>()) })?;
    let count_guard = CudaDeviceAllocationGuard {
        ptr: device_count,
        free: *cu_mem_free,
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_resident_i32_compare_project".as_ptr(),
        )
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut offset_arg = byte_offset;
    let mut rows_arg = row_count;
    let mut needle_arg = needle;
    let mut comparison_arg = comparison.code();
    let mut values_arg = values_guard.ptr;
    let mut count_arg = count_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut offset_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut needle_arg as *mut i32).cast::<c_void>(),
        (&mut comparison_arg as *mut u32).cast::<c_void>(),
        (&mut values_arg as *mut u64).cast::<c_void>(),
        (&mut count_arg as *mut u64).cast::<c_void>(),
    ];
    launch_with_optional_cuda_event_timing(resident, *cu_ctx_synchronize, || unsafe {
        cu_launch_kernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;

    let mut output_count = 0_u64;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut output_count as *mut u64).cast::<c_void>(),
            count_guard.ptr,
            std::mem::size_of::<u64>(),
        )
    })?;
    if output_count > row_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(output_count).unwrap_or(usize::MAX),
        ));
    }

    let mut output = vec![
        0_i32;
        usize::try_from(output_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?
    ];
    if !output.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                output.as_mut_ptr().cast::<c_void>(),
                values_guard.ptr,
                output.len() * std::mem::size_of::<i32>(),
            )
        })?;
    }

    drop(module_guard);
    drop(count_guard);
    drop(values_guard);
    Ok(output)
}

impl GpuRuntime for CudaDriverRuntime {
    fn can_run(&self, gpu_id: u16, _op: &PlannedOp) -> Result<(), GpuFallbackReason> {
        if !self.snapshot.driver_available || gpu_id >= self.snapshot.device_count {
            return Err(GpuFallbackReason::Unavailable);
        }

        Ok(())
    }
}

fn probe_cuda_runtime_snapshot() -> Result<CudaRuntimeSnapshot, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDriverGetVersion = unsafe extern "C" fn(*mut i32) -> i32;
    type CuDeviceGetCount = unsafe extern "C" fn(*mut i32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuDeviceGetName = unsafe extern "C" fn(*mut i8, i32, i32) -> i32;
    type CuDeviceTotalMem = unsafe extern "C" fn(*mut usize, i32) -> i32;

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_driver_get_version = unsafe {
        lib.get::<CuDriverGetVersion>(b"cuDriverGetVersion\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get_count = unsafe {
        lib.get::<CuDeviceGetCount>(b"cuDeviceGetCount\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get_name = unsafe {
        lib.get::<CuDeviceGetName>(b"cuDeviceGetName\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_total_mem = unsafe {
        lib.get::<CuDeviceTotalMem>(b"cuDeviceTotalMem_v2\0")
            .or_else(|_| lib.get::<CuDeviceTotalMem>(b"cuDeviceTotalMem\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let init_code = unsafe { cu_init(0) };
    if init_code != 0 {
        return Err(CudaRuntimeProbeError::DriverInitFailed(init_code));
    }

    let mut driver_version = 0;
    let driver_version_code = unsafe { cu_driver_get_version(&mut driver_version) };
    let driver_version = if driver_version_code == 0 {
        Some(driver_version)
    } else {
        None
    };

    let mut count = 0;
    let count_code = unsafe { cu_device_get_count(&mut count) };
    if count_code != 0 {
        return Err(CudaRuntimeProbeError::DeviceCountFailed(count_code));
    }

    let device_count =
        u16::try_from(count).map_err(|_| CudaRuntimeProbeError::InvalidDeviceCount(count))?;
    let mut devices = Vec::with_capacity(device_count as usize);
    for id in 0..device_count {
        let mut device = 0;
        let device_code = unsafe { cu_device_get(&mut device, i32::from(id)) };
        if device_code != 0 {
            return Err(CudaRuntimeProbeError::DeviceCountFailed(device_code));
        }

        let mut name = [0_i8; 256];
        let name_code = unsafe { cu_device_get_name(name.as_mut_ptr(), name.len() as i32, device) };
        if name_code != 0 {
            return Err(CudaRuntimeProbeError::DeviceCountFailed(name_code));
        }

        let name = name
            .iter()
            .take_while(|byte| **byte != 0)
            .map(|byte| *byte as u8)
            .collect::<Vec<_>>();
        let name = String::from_utf8_lossy(&name).into_owned();

        let mut total_memory_bytes = 0_usize;
        let memory_code = unsafe { cu_device_total_mem(&mut total_memory_bytes, device) };
        if memory_code != 0 {
            return Err(CudaRuntimeProbeError::DeviceCountFailed(memory_code));
        }

        devices.push(CudaDeviceSnapshot {
            id,
            name,
            total_memory_bytes: total_memory_bytes as u64,
        });
    }

    Ok(CudaRuntimeSnapshot {
        driver_available: true,
        driver_version,
        device_count,
        devices,
    })
}

fn launch_cuda_smoke_add_one(input: u32) -> Result<u32, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_smoke_add_one(
    .param .u64 out_ptr,
    .param .u32 input
)
{
    .reg .u64 %out;
    .reg .u32 %r_value;
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %r_value, [input];
    add.u32 %r_value, %r_value, 1;
    st.global.u32 [%out], %r_value;
    ret;
}
"#;

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let mut device_output = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_output, std::mem::size_of::<u32>()) })?;
    let allocation_guard = CudaDeviceAllocationGuard {
        ptr: device_output,
        free: *cu_mem_free,
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(&mut function, module, c"gpu_db_cuda_smoke_add_one".as_ptr())
    })?;

    let mut output_arg = allocation_guard.ptr;
    let mut input_arg = input;
    let mut args = [
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut input_arg as *mut u32).cast::<c_void>(),
    ];
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut output = 0_u32;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut output as *mut u32).cast::<c_void>(),
            allocation_guard.ptr,
            std::mem::size_of::<u32>(),
        )
    })?;

    drop(module_guard);
    drop(allocation_guard);
    drop(context_guard);

    Ok(output)
}

fn launch_cuda_all_mask(row_count: usize) -> Result<Vec<bool>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_all_mask(
    .param .u64 mask_ptr,
    .param .u32 row_count
)
{
    .reg .pred %p_out;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_offset;
    .reg .u64 %rd_mask_addr;

    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u32 %r_row_count, [row_count];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset, %r_idx, 4;
    add.u64 %rd_mask_addr, %rd_mask, %rd_offset;
    st.global.u32 [%rd_mask_addr], 1;

DONE:
    ret;
}
"#;

    let row_count = u32::try_from(row_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(row_count))?;
    if row_count == 0 {
        return Ok(Vec::new());
    }

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let mask_len = row_count as usize;
    let mask_bytes = mask_len * std::mem::size_of::<u32>();
    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, mask_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
        free: *cu_mem_free,
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(&mut function, module, c"gpu_db_cuda_all_mask".as_ptr())
    })?;

    let mut mask_arg = mask_guard.ptr;
    let mut row_count_arg = row_count;
    let mut args = [
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count.div_ceil(threads_per_block);
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut mask = vec![0_u32; mask_len];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            mask.as_mut_ptr().cast::<c_void>(),
            mask_guard.ptr,
            mask_bytes,
        )
    })?;

    drop(module_guard);
    drop(mask_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
}

fn launch_cuda_u32_equal_mask(
    input: &[u32],
    needle: u32,
) -> Result<Vec<bool>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_u32_equal_mask(
    .param .u64 input_ptr,
    .param .u64 mask_ptr,
    .param .u32 len,
    .param .u32 needle
)
{
    .reg .pred %p_out;
    .reg .pred %p_match;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_len;
    .reg .u32 %r_needle;
    .reg .u32 %r_value;
    .reg .u32 %r_mask;
    .reg .u64 %rd_input;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_offset;
    .reg .u64 %rd_input_addr;
    .reg .u64 %rd_mask_addr;

    ld.param.u64 %rd_input, [input_ptr];
    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u32 %r_len, [len];
    ld.param.u32 %r_needle, [needle];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_len;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset, %r_idx, 4;
    add.u64 %rd_input_addr, %rd_input, %rd_offset;
    ld.global.u32 %r_value, [%rd_input_addr];
    setp.eq.u32 %p_match, %r_value, %r_needle;
    selp.u32 %r_mask, 1, 0, %p_match;
    add.u64 %rd_mask_addr, %rd_mask, %rd_offset;
    st.global.u32 [%rd_mask_addr], %r_mask;

DONE:
    ret;
}
"#;

    let len = u32::try_from(input.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(input.len()))?;
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let input_bytes = std::mem::size_of_val(input);
    let mut device_input = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_input, input_bytes) })?;
    let input_guard = CudaDeviceAllocationGuard {
        ptr: device_input,
        free: *cu_mem_free,
    };

    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, input_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
        free: *cu_mem_free,
    };

    check_cuda(unsafe {
        cu_memcpy_htod(
            input_guard.ptr,
            input.as_ptr().cast::<c_void>(),
            input_bytes,
        )
    })?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_cuda_u32_equal_mask".as_ptr(),
        )
    })?;

    let mut input_arg = input_guard.ptr;
    let mut mask_arg = mask_guard.ptr;
    let mut len_arg = len;
    let mut needle_arg = needle;
    let mut args = [
        (&mut input_arg as *mut u64).cast::<c_void>(),
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut len_arg as *mut u32).cast::<c_void>(),
        (&mut needle_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = len.div_ceil(threads_per_block);
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut mask = vec![0_u32; input.len()];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            mask.as_mut_ptr().cast::<c_void>(),
            mask_guard.ptr,
            input_bytes,
        )
    })?;

    drop(module_guard);
    drop(mask_guard);
    drop(input_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
}

fn launch_cuda_bytes_equal_mask(
    input: &[&[u8]],
    needle: &[u8],
) -> Result<Vec<bool>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_bytes_equal_mask(
    .param .u64 bytes_ptr,
    .param .u64 offsets_ptr,
    .param .u64 needle_ptr,
    .param .u64 mask_ptr,
    .param .u32 row_count,
    .param .u32 needle_len
)
{
    .reg .pred %p_out;
    .reg .pred %p_len_diff;
    .reg .pred %p_loop_done;
    .reg .pred %p_byte_diff;
    .reg .u16 %h_byte;
    .reg .u16 %n_byte;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u32 %r_needle_len;
    .reg .u32 %r_start;
    .reg .u32 %r_end;
    .reg .u32 %r_len;
    .reg .u32 %r_i;
    .reg .u32 %r_mask;
    .reg .u64 %rd_bytes;
    .reg .u64 %rd_offsets;
    .reg .u64 %rd_needle;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_offset_addr;
    .reg .u64 %rd_next_offset_addr;
    .reg .u64 %rd_byte_offset;
    .reg .u64 %rd_hay_addr;
    .reg .u64 %rd_needle_addr;
    .reg .u64 %rd_mask_offset;
    .reg .u64 %rd_mask_addr;

    ld.param.u64 %rd_bytes, [bytes_ptr];
    ld.param.u64 %rd_offsets, [offsets_ptr];
    ld.param.u64 %rd_needle, [needle_ptr];
    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u32 %r_row_count, [row_count];
    ld.param.u32 %r_needle_len, [needle_len];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset_addr, %r_idx, 4;
    add.u64 %rd_offset_addr, %rd_offsets, %rd_offset_addr;
    add.u64 %rd_next_offset_addr, %rd_offset_addr, 4;
    ld.global.u32 %r_start, [%rd_offset_addr];
    ld.global.u32 %r_end, [%rd_next_offset_addr];
    sub.u32 %r_len, %r_end, %r_start;
    setp.ne.u32 %p_len_diff, %r_len, %r_needle_len;
    @%p_len_diff bra NO_MATCH;

    mov.u32 %r_i, 0;
LOOP:
    setp.ge.u32 %p_loop_done, %r_i, %r_needle_len;
    @%p_loop_done bra MATCH;
    add.u32 %r_len, %r_start, %r_i;
    cvt.u64.u32 %rd_byte_offset, %r_len;
    add.u64 %rd_hay_addr, %rd_bytes, %rd_byte_offset;
    cvt.u64.u32 %rd_byte_offset, %r_i;
    add.u64 %rd_needle_addr, %rd_needle, %rd_byte_offset;
    ld.global.u8 %h_byte, [%rd_hay_addr];
    ld.global.u8 %n_byte, [%rd_needle_addr];
    setp.ne.u16 %p_byte_diff, %h_byte, %n_byte;
    @%p_byte_diff bra NO_MATCH;
    add.u32 %r_i, %r_i, 1;
    bra LOOP;

MATCH:
    mov.u32 %r_mask, 1;
    bra STORE;

NO_MATCH:
    mov.u32 %r_mask, 0;

STORE:
    mul.wide.u32 %rd_mask_offset, %r_idx, 4;
    add.u64 %rd_mask_addr, %rd_mask, %rd_mask_offset;
    st.global.u32 [%rd_mask_addr], %r_mask;

DONE:
    ret;
}
"#;

    let row_count = u32::try_from(input.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(input.len()))?;
    let needle_len = u32::try_from(needle.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(needle.len()))?;
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut offsets = Vec::with_capacity(input.len() + 1);
    let mut flattened = Vec::new();
    offsets.push(0_u32);
    for value in input {
        flattened.extend_from_slice(value);
        offsets.push(
            u32::try_from(flattened.len())
                .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(flattened.len()))?,
        );
    }

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let bytes_len = flattened.len().max(1);
    let mut device_bytes = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_bytes, bytes_len) })?;
    let bytes_guard = CudaDeviceAllocationGuard {
        ptr: device_bytes,
        free: *cu_mem_free,
    };

    let offsets_bytes = std::mem::size_of_val(offsets.as_slice());
    let mut device_offsets = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_offsets, offsets_bytes) })?;
    let offsets_guard = CudaDeviceAllocationGuard {
        ptr: device_offsets,
        free: *cu_mem_free,
    };

    let needle_bytes = needle.len().max(1);
    let mut device_needle = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_needle, needle_bytes) })?;
    let needle_guard = CudaDeviceAllocationGuard {
        ptr: device_needle,
        free: *cu_mem_free,
    };

    let mask_bytes = input.len() * std::mem::size_of::<u32>();
    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, mask_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
        free: *cu_mem_free,
    };

    if !flattened.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                bytes_guard.ptr,
                flattened.as_ptr().cast::<c_void>(),
                flattened.len(),
            )
        })?;
    }
    check_cuda(unsafe {
        cu_memcpy_htod(
            offsets_guard.ptr,
            offsets.as_ptr().cast::<c_void>(),
            offsets_bytes,
        )
    })?;
    if !needle.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                needle_guard.ptr,
                needle.as_ptr().cast::<c_void>(),
                needle.len(),
            )
        })?;
    }

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_cuda_bytes_equal_mask".as_ptr(),
        )
    })?;

    let mut bytes_arg = bytes_guard.ptr;
    let mut offsets_arg = offsets_guard.ptr;
    let mut needle_arg = needle_guard.ptr;
    let mut mask_arg = mask_guard.ptr;
    let mut row_count_arg = row_count;
    let mut needle_len_arg = needle_len;
    let mut args = [
        (&mut bytes_arg as *mut u64).cast::<c_void>(),
        (&mut offsets_arg as *mut u64).cast::<c_void>(),
        (&mut needle_arg as *mut u64).cast::<c_void>(),
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
        (&mut needle_len_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count.div_ceil(threads_per_block);
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut mask = vec![0_u32; input.len()];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            mask.as_mut_ptr().cast::<c_void>(),
            mask_guard.ptr,
            mask_bytes,
        )
    })?;

    drop(module_guard);
    drop(mask_guard);
    drop(needle_guard);
    drop(offsets_guard);
    drop(bytes_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
}

fn launch_cuda_bytes_range_mask(
    input: &[&[u8]],
    start_inclusive: &[u8],
    end_exclusive: &[u8],
) -> Result<Vec<bool>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_bytes_range_mask(
    .param .u64 bytes_ptr,
    .param .u64 offsets_ptr,
    .param .u64 start_ptr,
    .param .u64 end_ptr,
    .param .u64 mask_ptr,
    .param .u32 row_count,
    .param .u32 start_len,
    .param .u32 end_len
)
{
    .reg .pred %p_out;
    .reg .pred %p_loop_done;
    .reg .pred %p_row_done;
    .reg .pred %p_bound_done;
    .reg .pred %p_lt;
    .reg .pred %p_gt;
    .reg .pred %p_ge_start;
    .reg .pred %p_lt_end;
    .reg .u16 %h_byte;
    .reg .u16 %b_byte;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u32 %r_start_len;
    .reg .u32 %r_end_len;
    .reg .u32 %r_row_start;
    .reg .u32 %r_row_end;
    .reg .u32 %r_row_len;
    .reg .u32 %r_i;
    .reg .u32 %r_pos;
    .reg .u32 %r_mask;
    .reg .u64 %rd_bytes;
    .reg .u64 %rd_offsets;
    .reg .u64 %rd_start;
    .reg .u64 %rd_end;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_offset_addr;
    .reg .u64 %rd_next_offset_addr;
    .reg .u64 %rd_byte_offset;
    .reg .u64 %rd_hay_addr;
    .reg .u64 %rd_bound_addr;
    .reg .u64 %rd_mask_offset;
    .reg .u64 %rd_mask_addr;

    ld.param.u64 %rd_bytes, [bytes_ptr];
    ld.param.u64 %rd_offsets, [offsets_ptr];
    ld.param.u64 %rd_start, [start_ptr];
    ld.param.u64 %rd_end, [end_ptr];
    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u32 %r_row_count, [row_count];
    ld.param.u32 %r_start_len, [start_len];
    ld.param.u32 %r_end_len, [end_len];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset_addr, %r_idx, 4;
    add.u64 %rd_offset_addr, %rd_offsets, %rd_offset_addr;
    add.u64 %rd_next_offset_addr, %rd_offset_addr, 4;
    ld.global.u32 %r_row_start, [%rd_offset_addr];
    ld.global.u32 %r_row_end, [%rd_next_offset_addr];
    sub.u32 %r_row_len, %r_row_end, %r_row_start;

    mov.u32 %r_i, 0;
START_LOOP:
    setp.ge.u32 %p_row_done, %r_i, %r_row_len;
    setp.ge.u32 %p_bound_done, %r_i, %r_start_len;
    or.pred %p_loop_done, %p_row_done, %p_bound_done;
    @%p_loop_done bra START_PREFIX_DONE;

    add.u32 %r_pos, %r_row_start, %r_i;
    cvt.u64.u32 %rd_byte_offset, %r_pos;
    add.u64 %rd_hay_addr, %rd_bytes, %rd_byte_offset;
    cvt.u64.u32 %rd_byte_offset, %r_i;
    add.u64 %rd_bound_addr, %rd_start, %rd_byte_offset;
    ld.global.u8 %h_byte, [%rd_hay_addr];
    ld.global.u8 %b_byte, [%rd_bound_addr];
    setp.lt.u16 %p_lt, %h_byte, %b_byte;
    @%p_lt bra NO_MATCH;
    setp.gt.u16 %p_gt, %h_byte, %b_byte;
    @%p_gt bra START_MATCH;
    add.u32 %r_i, %r_i, 1;
    bra START_LOOP;

START_PREFIX_DONE:
    setp.ge.u32 %p_ge_start, %r_row_len, %r_start_len;
    @%p_ge_start bra START_MATCH;
    bra NO_MATCH;

START_MATCH:
    mov.u32 %r_i, 0;
END_LOOP:
    setp.ge.u32 %p_row_done, %r_i, %r_row_len;
    setp.ge.u32 %p_bound_done, %r_i, %r_end_len;
    or.pred %p_loop_done, %p_row_done, %p_bound_done;
    @%p_loop_done bra END_PREFIX_DONE;

    add.u32 %r_pos, %r_row_start, %r_i;
    cvt.u64.u32 %rd_byte_offset, %r_pos;
    add.u64 %rd_hay_addr, %rd_bytes, %rd_byte_offset;
    cvt.u64.u32 %rd_byte_offset, %r_i;
    add.u64 %rd_bound_addr, %rd_end, %rd_byte_offset;
    ld.global.u8 %h_byte, [%rd_hay_addr];
    ld.global.u8 %b_byte, [%rd_bound_addr];
    setp.lt.u16 %p_lt, %h_byte, %b_byte;
    @%p_lt bra MATCH;
    setp.gt.u16 %p_gt, %h_byte, %b_byte;
    @%p_gt bra NO_MATCH;
    add.u32 %r_i, %r_i, 1;
    bra END_LOOP;

END_PREFIX_DONE:
    setp.lt.u32 %p_lt_end, %r_row_len, %r_end_len;
    @%p_lt_end bra MATCH;
    bra NO_MATCH;

MATCH:
    mov.u32 %r_mask, 1;
    bra STORE;

NO_MATCH:
    mov.u32 %r_mask, 0;

STORE:
    mul.wide.u32 %rd_mask_offset, %r_idx, 4;
    add.u64 %rd_mask_addr, %rd_mask, %rd_mask_offset;
    st.global.u32 [%rd_mask_addr], %r_mask;

DONE:
    ret;
}
"#;

    let row_count = u32::try_from(input.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(input.len()))?;
    let start_len = u32::try_from(start_inclusive.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(start_inclusive.len()))?;
    let end_len = u32::try_from(end_exclusive.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(end_exclusive.len()))?;
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut offsets = Vec::with_capacity(input.len() + 1);
    let mut flattened = Vec::new();
    offsets.push(0_u32);
    for value in input {
        flattened.extend_from_slice(value);
        offsets.push(
            u32::try_from(flattened.len())
                .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(flattened.len()))?,
        );
    }

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let bytes_len = flattened.len().max(1);
    let mut device_bytes = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_bytes, bytes_len) })?;
    let bytes_guard = CudaDeviceAllocationGuard {
        ptr: device_bytes,
        free: *cu_mem_free,
    };

    let offsets_bytes = std::mem::size_of_val(offsets.as_slice());
    let mut device_offsets = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_offsets, offsets_bytes) })?;
    let offsets_guard = CudaDeviceAllocationGuard {
        ptr: device_offsets,
        free: *cu_mem_free,
    };

    let start_bytes = start_inclusive.len().max(1);
    let mut device_start = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_start, start_bytes) })?;
    let start_guard = CudaDeviceAllocationGuard {
        ptr: device_start,
        free: *cu_mem_free,
    };

    let end_bytes = end_exclusive.len().max(1);
    let mut device_end = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_end, end_bytes) })?;
    let end_guard = CudaDeviceAllocationGuard {
        ptr: device_end,
        free: *cu_mem_free,
    };

    let mask_bytes = input.len() * std::mem::size_of::<u32>();
    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, mask_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
        free: *cu_mem_free,
    };

    if !flattened.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                bytes_guard.ptr,
                flattened.as_ptr().cast::<c_void>(),
                flattened.len(),
            )
        })?;
    }
    check_cuda(unsafe {
        cu_memcpy_htod(
            offsets_guard.ptr,
            offsets.as_ptr().cast::<c_void>(),
            offsets_bytes,
        )
    })?;
    if !start_inclusive.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                start_guard.ptr,
                start_inclusive.as_ptr().cast::<c_void>(),
                start_inclusive.len(),
            )
        })?;
    }
    if !end_exclusive.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                end_guard.ptr,
                end_exclusive.as_ptr().cast::<c_void>(),
                end_exclusive.len(),
            )
        })?;
    }

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_cuda_bytes_range_mask".as_ptr(),
        )
    })?;

    let mut bytes_arg = bytes_guard.ptr;
    let mut offsets_arg = offsets_guard.ptr;
    let mut start_arg = start_guard.ptr;
    let mut end_arg = end_guard.ptr;
    let mut mask_arg = mask_guard.ptr;
    let mut row_count_arg = row_count;
    let mut start_len_arg = start_len;
    let mut end_len_arg = end_len;
    let mut args = [
        (&mut bytes_arg as *mut u64).cast::<c_void>(),
        (&mut offsets_arg as *mut u64).cast::<c_void>(),
        (&mut start_arg as *mut u64).cast::<c_void>(),
        (&mut end_arg as *mut u64).cast::<c_void>(),
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
        (&mut start_len_arg as *mut u32).cast::<c_void>(),
        (&mut end_len_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count.div_ceil(threads_per_block);
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut mask = vec![0_u32; input.len()];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            mask.as_mut_ptr().cast::<c_void>(),
            mask_guard.ptr,
            mask_bytes,
        )
    })?;

    drop(module_guard);
    drop(mask_guard);
    drop(end_guard);
    drop(start_guard);
    drop(offsets_guard);
    drop(bytes_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
}

fn launch_cuda_mvcc_row_batch_lengths(
    batch: &CudaMvccRowBatch,
) -> Result<Vec<(u32, u32)>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_mvcc_row_batch_lengths(
    .param .u64 key_offsets_ptr,
    .param .u64 value_offsets_ptr,
    .param .u64 output_ptr,
    .param .u32 row_count
)
{
    .reg .pred %p_out;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u32 %r_key_start;
    .reg .u32 %r_key_end;
    .reg .u32 %r_value_start;
    .reg .u32 %r_value_end;
    .reg .u32 %r_key_len;
    .reg .u32 %r_value_len;
    .reg .u64 %rd_key_offsets;
    .reg .u64 %rd_value_offsets;
    .reg .u64 %rd_output;
    .reg .u64 %rd_offset;
    .reg .u64 %rd_next_offset;
    .reg .u64 %rd_output_offset;
    .reg .u64 %addr;

    ld.param.u64 %rd_key_offsets, [key_offsets_ptr];
    ld.param.u64 %rd_value_offsets, [value_offsets_ptr];
    ld.param.u64 %rd_output, [output_ptr];
    ld.param.u32 %r_row_count, [row_count];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset, %r_idx, 4;
    add.u64 %addr, %rd_key_offsets, %rd_offset;
    add.u64 %rd_next_offset, %addr, 4;
    ld.global.u32 %r_key_start, [%addr];
    ld.global.u32 %r_key_end, [%rd_next_offset];
    sub.u32 %r_key_len, %r_key_end, %r_key_start;

    add.u64 %addr, %rd_value_offsets, %rd_offset;
    add.u64 %rd_next_offset, %addr, 4;
    ld.global.u32 %r_value_start, [%addr];
    ld.global.u32 %r_value_end, [%rd_next_offset];
    sub.u32 %r_value_len, %r_value_end, %r_value_start;

    mul.wide.u32 %rd_output_offset, %r_idx, 8;
    add.u64 %addr, %rd_output, %rd_output_offset;
    st.global.u32 [%addr], %r_key_len;
    add.u64 %addr, %addr, 4;
    st.global.u32 [%addr], %r_value_len;

DONE:
    ret;
}
"#;

    batch.validate()?;
    if batch.row_count == 0 {
        return Ok(Vec::new());
    }

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let key_offsets_bytes = std::mem::size_of_val(batch.key_offsets.as_slice());
    let mut device_key_offsets = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_key_offsets, key_offsets_bytes) })?;
    let key_offsets_guard = CudaDeviceAllocationGuard {
        ptr: device_key_offsets,
        free: *cu_mem_free,
    };

    let value_offsets_bytes = std::mem::size_of_val(batch.value_offsets.as_slice());
    let mut device_value_offsets = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_value_offsets, value_offsets_bytes) })?;
    let value_offsets_guard = CudaDeviceAllocationGuard {
        ptr: device_value_offsets,
        free: *cu_mem_free,
    };

    let output_words = batch.row_count as usize * 2;
    let output_bytes = output_words * std::mem::size_of::<u32>();
    let mut device_output = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_output, output_bytes) })?;
    let output_guard = CudaDeviceAllocationGuard {
        ptr: device_output,
        free: *cu_mem_free,
    };

    check_cuda(unsafe {
        cu_memcpy_htod(
            key_offsets_guard.ptr,
            batch.key_offsets.as_ptr().cast::<c_void>(),
            key_offsets_bytes,
        )
    })?;
    check_cuda(unsafe {
        cu_memcpy_htod(
            value_offsets_guard.ptr,
            batch.value_offsets.as_ptr().cast::<c_void>(),
            value_offsets_bytes,
        )
    })?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_cuda_mvcc_row_batch_lengths".as_ptr(),
        )
    })?;

    let mut key_offsets_arg = key_offsets_guard.ptr;
    let mut value_offsets_arg = value_offsets_guard.ptr;
    let mut output_arg = output_guard.ptr;
    let mut row_count_arg = batch.row_count;
    let mut args = [
        (&mut key_offsets_arg as *mut u64).cast::<c_void>(),
        (&mut value_offsets_arg as *mut u64).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = batch.row_count.div_ceil(threads_per_block);
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut output = vec![0_u32; output_words];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            output.as_mut_ptr().cast::<c_void>(),
            output_guard.ptr,
            output_bytes,
        )
    })?;

    drop(module_guard);
    drop(output_guard);
    drop(value_offsets_guard);
    drop(key_offsets_guard);
    drop(context_guard);

    Ok(output
        .chunks_exact(2)
        .map(|lengths| (lengths[0], lengths[1]))
        .collect())
}

fn launch_cuda_mvcc_visibility_mask(
    batch: &CudaMvccRowBatch,
    read_txn_id: u64,
) -> Result<Vec<bool>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_mvcc_visibility_mask(
    .param .u64 begin_txn_ids_ptr,
    .param .u64 end_txn_ids_ptr,
    .param .u64 mask_ptr,
    .param .u64 read_txn_id,
    .param .u32 row_count
)
{
    .reg .pred %p_out;
    .reg .pred %p_created;
    .reg .pred %p_not_deleted;
    .reg .pred %p_visible;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u32 %r_mask_value;
    .reg .u64 %rd_begin;
    .reg .u64 %rd_end;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_read_txn_id;
    .reg .u64 %rd_offset8;
    .reg .u64 %rd_offset4;
    .reg .u64 %addr;
    .reg .u64 %rd_created_by;
    .reg .u64 %rd_deleted_by;

    ld.param.u64 %rd_begin, [begin_txn_ids_ptr];
    ld.param.u64 %rd_end, [end_txn_ids_ptr];
    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u64 %rd_read_txn_id, [read_txn_id];
    ld.param.u32 %r_row_count, [row_count];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset8, %r_idx, 8;
    add.u64 %addr, %rd_begin, %rd_offset8;
    ld.global.u64 %rd_created_by, [%addr];
    add.u64 %addr, %rd_end, %rd_offset8;
    ld.global.u64 %rd_deleted_by, [%addr];

    setp.le.u64 %p_created, %rd_created_by, %rd_read_txn_id;
    setp.gt.u64 %p_not_deleted, %rd_deleted_by, %rd_read_txn_id;
    and.pred %p_visible, %p_created, %p_not_deleted;
    selp.u32 %r_mask_value, 1, 0, %p_visible;

    mul.wide.u32 %rd_offset4, %r_idx, 4;
    add.u64 %addr, %rd_mask, %rd_offset4;
    st.global.u32 [%addr], %r_mask_value;

DONE:
    ret;
}
"#;

    batch.validate()?;
    if batch.row_count == 0 {
        return Ok(Vec::new());
    }

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let begin_bytes = std::mem::size_of_val(batch.begin_txn_ids.as_slice());
    let mut device_begin = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_begin, begin_bytes) })?;
    let begin_guard = CudaDeviceAllocationGuard {
        ptr: device_begin,
        free: *cu_mem_free,
    };

    let end_bytes = std::mem::size_of_val(batch.end_txn_ids.as_slice());
    let mut device_end = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_end, end_bytes) })?;
    let end_guard = CudaDeviceAllocationGuard {
        ptr: device_end,
        free: *cu_mem_free,
    };

    let mask_bytes = batch.row_count as usize * std::mem::size_of::<u32>();
    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, mask_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
        free: *cu_mem_free,
    };

    check_cuda(unsafe {
        cu_memcpy_htod(
            begin_guard.ptr,
            batch.begin_txn_ids.as_ptr().cast::<c_void>(),
            begin_bytes,
        )
    })?;
    check_cuda(unsafe {
        cu_memcpy_htod(
            end_guard.ptr,
            batch.end_txn_ids.as_ptr().cast::<c_void>(),
            end_bytes,
        )
    })?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_cuda_mvcc_visibility_mask".as_ptr(),
        )
    })?;

    let mut begin_arg = begin_guard.ptr;
    let mut end_arg = end_guard.ptr;
    let mut mask_arg = mask_guard.ptr;
    let mut read_txn_id_arg = read_txn_id;
    let mut row_count_arg = batch.row_count;
    let mut args = [
        (&mut begin_arg as *mut u64).cast::<c_void>(),
        (&mut end_arg as *mut u64).cast::<c_void>(),
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut read_txn_id_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = batch.row_count.div_ceil(threads_per_block);
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut mask = vec![0_u32; batch.row_count as usize];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            mask.as_mut_ptr().cast::<c_void>(),
            mask_guard.ptr,
            mask_bytes,
        )
    })?;

    drop(module_guard);
    drop(mask_guard);
    drop(end_guard);
    drop(begin_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
}

fn check_cuda(code: i32) -> Result<(), CudaRuntimeProbeError> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaRuntimeProbeError::KernelLaunchFailed(code))
    }
}

struct CudaContextGuard {
    context: *mut c_void,
    destroy: unsafe extern "C" fn(*mut c_void) -> i32,
}

impl Drop for CudaContextGuard {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.context);
        }
    }
}

struct CudaDeviceAllocationGuard {
    ptr: u64,
    free: unsafe extern "C" fn(u64) -> i32,
}

impl Drop for CudaDeviceAllocationGuard {
    fn drop(&mut self) {
        unsafe {
            (self.free)(self.ptr);
        }
    }
}

struct CudaModuleGuard {
    module: *mut c_void,
    unload: unsafe extern "C" fn(*mut c_void) -> i32,
}

impl Drop for CudaModuleGuard {
    fn drop(&mut self) {
        unsafe {
            (self.unload)(self.module);
        }
    }
}

struct CudaEventGuard {
    event: *mut c_void,
    destroy: unsafe extern "C" fn(*mut c_void) -> i32,
}

impl Drop for CudaEventGuard {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.event);
        }
    }
}

fn launch_with_optional_cuda_event_timing<R, F>(
    resident: &R,
    cu_ctx_synchronize: unsafe extern "C" fn() -> i32,
    launch: F,
) -> Result<(), CudaRuntimeProbeError>
where
    R: CudaResidentReadSource,
    F: FnOnce() -> i32,
{
    type CuEventCreate = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
    type CuEventDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuEventRecord = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
    type CuEventSynchronize = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuEventElapsedTime = unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> i32;

    let event_symbols = unsafe {
        let create = resident.lib().get::<CuEventCreate>(b"cuEventCreate\0");
        let destroy = resident
            .lib()
            .get::<CuEventDestroy>(b"cuEventDestroy_v2\0")
            .or_else(|_| resident.lib().get::<CuEventDestroy>(b"cuEventDestroy\0"));
        let record = resident.lib().get::<CuEventRecord>(b"cuEventRecord\0");
        let synchronize = resident
            .lib()
            .get::<CuEventSynchronize>(b"cuEventSynchronize\0");
        let elapsed = resident
            .lib()
            .get::<CuEventElapsedTime>(b"cuEventElapsedTime\0");
        match (create, destroy, record, synchronize, elapsed) {
            (Ok(create), Ok(destroy), Ok(record), Ok(synchronize), Ok(elapsed)) => {
                Some((*create, *destroy, *record, *synchronize, *elapsed))
            }
            _ => None,
        }
    };

    if let Some((
        cu_event_create,
        cu_event_destroy,
        cu_event_record,
        cu_event_synchronize,
        cu_event_elapsed_time,
    )) = event_symbols
    {
        let mut start = std::ptr::null_mut();
        check_cuda(unsafe { cu_event_create(&mut start, 0) })?;
        let start_guard = CudaEventGuard {
            event: start,
            destroy: cu_event_destroy,
        };
        let mut stop = std::ptr::null_mut();
        check_cuda(unsafe { cu_event_create(&mut stop, 0) })?;
        let stop_guard = CudaEventGuard {
            event: stop,
            destroy: cu_event_destroy,
        };

        check_cuda(unsafe { cu_event_record(start_guard.event, std::ptr::null_mut()) })?;
        check_cuda(launch())?;
        check_cuda(unsafe { cu_event_record(stop_guard.event, std::ptr::null_mut()) })?;
        check_cuda(unsafe { cu_event_synchronize(stop_guard.event) })?;

        let mut elapsed_ms = 0.0_f32;
        check_cuda(unsafe {
            cu_event_elapsed_time(&mut elapsed_ms, start_guard.event, stop_guard.event)
        })?;
        resident
            .record_kernel_event_elapsed_us(Some((f64::from(elapsed_ms) * 1_000.0).ceil() as u64));
        return Ok(());
    }

    check_cuda(launch())?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    resident.record_kernel_event_elapsed_us(None);
    Ok(())
}

pub struct DeviceRouter<R> {
    runtime: R,
}

impl<R> DeviceRouter<R>
where
    R: GpuRuntime,
{
    pub fn new(runtime: R) -> Self {
        Self { runtime }
    }

    pub fn route(&self, op: &PlannedOp) -> RouteDecision {
        match op.target {
            DeviceTarget::Cpu => RouteDecision::Cpu,
            DeviceTarget::Gpu(gpu_id) => match self.runtime.can_run(gpu_id, op) {
                Ok(()) => RouteDecision::Gpu(gpu_id),
                Err(reason) => RouteDecision::CpuFallback {
                    requested_gpu: gpu_id,
                    reason,
                },
            },
        }
    }

    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    pub fn runtime_mut(&mut self) -> &mut R {
        &mut self.runtime
    }
}

#[derive(Debug, Default)]
pub struct MockGpuRuntime {
    unavailable: BTreeSet<u16>,
    memory_pressured: BTreeSet<u16>,
    saturated: bool,
}

impl MockGpuRuntime {
    pub fn snapshot(&self) -> GpuRuntimeSnapshot {
        GpuRuntimeSnapshot {
            unavailable_gpu_ids: self.unavailable.iter().copied().collect(),
            memory_pressured_gpu_ids: self.memory_pressured.iter().copied().collect(),
            saturated: self.saturated,
        }
    }

    pub fn mark_unavailable(&mut self, gpu_id: u16) {
        self.unavailable.insert(gpu_id);
    }

    pub fn clear_unavailable(&mut self, gpu_id: u16) {
        self.unavailable.remove(&gpu_id);
    }

    pub fn mark_memory_pressured(&mut self, gpu_id: u16) {
        self.memory_pressured.insert(gpu_id);
    }

    pub fn clear_memory_pressured(&mut self, gpu_id: u16) {
        self.memory_pressured.remove(&gpu_id);
    }

    pub fn set_saturated(&mut self, saturated: bool) {
        self.saturated = saturated;
    }
}

impl GpuRuntime for MockGpuRuntime {
    fn can_run(&self, gpu_id: u16, _op: &PlannedOp) -> Result<(), GpuFallbackReason> {
        if self.unavailable.contains(&gpu_id) {
            return Err(GpuFallbackReason::Unavailable);
        }
        if self.memory_pressured.contains(&gpu_id) {
            return Err(GpuFallbackReason::MemoryPressure);
        }
        if self.saturated {
            return Err(GpuFallbackReason::QueueSaturated);
        }

        Ok(())
    }
}

pub trait Operator<Row = Vec<u8>> {
    fn open(&mut self) {}
    fn next(&mut self) -> Option<Row>;
    fn close(&mut self) {}
}

#[derive(Debug, Default)]
pub struct CpuNoop;

impl Operator<Vec<u8>> for CpuNoop {
    fn next(&mut self) -> Option<Vec<u8>> {
        None
    }
}

#[derive(Debug, Clone, Default)]
pub struct VecOperator {
    rows: Vec<Vec<u8>>,
    next_index: usize,
}

impl VecOperator {
    pub fn new(rows: Vec<Vec<u8>>) -> Self {
        Self {
            rows,
            next_index: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

impl Operator<Vec<u8>> for VecOperator {
    fn open(&mut self) {
        self.next_index = 0;
    }

    fn next(&mut self) -> Option<Vec<u8>> {
        let row = self.rows.get(self.next_index)?.clone();
        self.next_index += 1;
        Some(row)
    }

    fn close(&mut self) {
        self.next_index = self.rows.len();
    }
}

#[derive(Debug, Clone, Default)]
pub struct ScanOperator<Row> {
    rows: Vec<Row>,
    next_index: usize,
}

impl<Row> ScanOperator<Row> {
    pub fn new(rows: Vec<Row>) -> Self {
        Self {
            rows,
            next_index: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

impl<Row: Clone> Operator<Row> for ScanOperator<Row> {
    fn open(&mut self) {
        self.next_index = 0;
    }

    fn next(&mut self) -> Option<Row> {
        let row = self.rows.get(self.next_index)?.clone();
        self.next_index += 1;
        Some(row)
    }

    fn close(&mut self) {
        self.next_index = self.rows.len();
    }
}

pub struct FilterOperator<Row, Child, Predicate> {
    child: Child,
    predicate: Predicate,
    _row: std::marker::PhantomData<Row>,
}

impl<Row, Child, Predicate> FilterOperator<Row, Child, Predicate> {
    pub fn new(child: Child, predicate: Predicate) -> Self {
        Self {
            child,
            predicate,
            _row: std::marker::PhantomData,
        }
    }
}

impl<Row, Child, Predicate> Operator<Row> for FilterOperator<Row, Child, Predicate>
where
    Child: Operator<Row>,
    Predicate: FnMut(&Row) -> bool,
{
    fn open(&mut self) {
        self.child.open();
    }

    fn next(&mut self) -> Option<Row> {
        while let Some(row) = self.child.next() {
            if (self.predicate)(&row) {
                return Some(row);
            }
        }
        None
    }

    fn close(&mut self) {
        self.child.close();
    }
}

pub struct ProjectOperator<Input, Output, Child, Projection> {
    child: Child,
    projection: Projection,
    _input: std::marker::PhantomData<Input>,
    _output: std::marker::PhantomData<Output>,
}

impl<Input, Output, Child, Projection> ProjectOperator<Input, Output, Child, Projection> {
    pub fn new(child: Child, projection: Projection) -> Self {
        Self {
            child,
            projection,
            _input: std::marker::PhantomData,
            _output: std::marker::PhantomData,
        }
    }
}

impl<Input, Output, Child, Projection> Operator<Output>
    for ProjectOperator<Input, Output, Child, Projection>
where
    Child: Operator<Input>,
    Projection: FnMut(Input) -> Output,
{
    fn open(&mut self) {
        self.child.open();
    }

    fn next(&mut self) -> Option<Output> {
        self.child.next().map(&mut self.projection)
    }

    fn close(&mut self) {
        self.child.close();
    }
}

pub struct LimitOperator<Row, Child> {
    child: Child,
    remaining: usize,
    initial_limit: usize,
    _row: std::marker::PhantomData<Row>,
}

impl<Row, Child> LimitOperator<Row, Child> {
    pub fn new(child: Child, limit: usize) -> Self {
        Self {
            child,
            remaining: limit,
            initial_limit: limit,
            _row: std::marker::PhantomData,
        }
    }
}

impl<Row, Child> Operator<Row> for LimitOperator<Row, Child>
where
    Child: Operator<Row>,
{
    fn open(&mut self) {
        self.remaining = self.initial_limit;
        self.child.open();
    }

    fn next(&mut self) -> Option<Row> {
        if self.remaining == 0 {
            return None;
        }

        let row = self.child.next()?;
        self.remaining -= 1;
        Some(row)
    }

    fn close(&mut self) {
        self.remaining = 0;
        self.child.close();
    }
}

pub struct SortOperator<Row, Child, Compare> {
    child: Child,
    compare: Compare,
    sorted_rows: Vec<Row>,
    next_index: usize,
}

impl<Row, Child, Compare> SortOperator<Row, Child, Compare> {
    pub fn new(child: Child, compare: Compare) -> Self {
        Self {
            child,
            compare,
            sorted_rows: Vec::new(),
            next_index: 0,
        }
    }
}

impl<Row, Child, Compare> Operator<Row> for SortOperator<Row, Child, Compare>
where
    Row: Clone,
    Child: Operator<Row>,
    Compare: FnMut(&Row, &Row) -> std::cmp::Ordering,
{
    fn open(&mut self) {
        self.child.open();
        self.sorted_rows.clear();
        self.next_index = 0;

        while let Some(row) = self.child.next() {
            self.sorted_rows.push(row);
        }

        self.sorted_rows.sort_by(&mut self.compare);
        self.child.close();
    }

    fn next(&mut self) -> Option<Row> {
        let row = self.sorted_rows.get(self.next_index)?.clone();
        self.next_index += 1;
        Some(row)
    }

    fn close(&mut self) {
        self.next_index = self.sorted_rows.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_op(id: u16) -> PlannedOp {
        PlannedOp {
            name: "scan".to_string(),
            target: DeviceTarget::Gpu(id),
        }
    }

    #[test]
    fn cpu_target_routes_to_cpu_without_runtime_check() {
        let router = DeviceRouter::new(MockGpuRuntime::default());
        let op = PlannedOp {
            name: "filter".to_string(),
            target: DeviceTarget::Cpu,
        };

        assert_eq!(router.route(&op), RouteDecision::Cpu);
    }

    #[test]
    fn gpu_target_routes_to_gpu_when_available() {
        let router = DeviceRouter::new(MockGpuRuntime::default());

        assert_eq!(router.route(&gpu_op(0)), RouteDecision::Gpu(0));
    }

    #[test]
    fn gpu_target_falls_back_when_unavailable() {
        let mut runtime = MockGpuRuntime::default();
        runtime.mark_unavailable(2);
        let router = DeviceRouter::new(runtime);

        assert_eq!(
            router.route(&gpu_op(2)),
            RouteDecision::CpuFallback {
                requested_gpu: 2,
                reason: GpuFallbackReason::Unavailable,
            }
        );
    }

    #[test]
    fn gpu_target_falls_back_when_memory_pressured() {
        let mut runtime = MockGpuRuntime::default();
        runtime.mark_memory_pressured(3);
        let router = DeviceRouter::new(runtime);

        assert_eq!(
            router.route(&gpu_op(3)),
            RouteDecision::CpuFallback {
                requested_gpu: 3,
                reason: GpuFallbackReason::MemoryPressure,
            }
        );
    }

    #[test]
    fn gpu_target_falls_back_when_queue_is_saturated() {
        let mut runtime = MockGpuRuntime::default();
        runtime.set_saturated(true);
        let router = DeviceRouter::new(runtime);

        assert_eq!(
            router.route(&gpu_op(1)),
            RouteDecision::CpuFallback {
                requested_gpu: 1,
                reason: GpuFallbackReason::QueueSaturated,
            }
        );
    }

    #[test]
    fn mock_gpu_runtime_snapshot_reports_blocked_ids_and_pressure() {
        let mut runtime = MockGpuRuntime::default();
        runtime.mark_unavailable(3);
        runtime.mark_unavailable(1);
        runtime.mark_memory_pressured(5);
        runtime.mark_memory_pressured(3);
        runtime.set_saturated(true);

        let snapshot = runtime.snapshot();

        assert_eq!(snapshot.unavailable_gpu_ids, vec![1, 3]);
        assert_eq!(snapshot.memory_pressured_gpu_ids, vec![3, 5]);
        assert!(snapshot.saturated);
        assert!(snapshot.has_pressure());
        assert_eq!(snapshot.blocked_gpu_ids(), vec![1, 3, 5]);
    }

    #[test]
    fn cuda_driver_runtime_routes_only_detected_devices() {
        let router = DeviceRouter::new(CudaDriverRuntime::from_device_count(1));

        assert_eq!(router.route(&gpu_op(0)), RouteDecision::Gpu(0));
        assert_eq!(
            router.route(&gpu_op(1)),
            RouteDecision::CpuFallback {
                requested_gpu: 1,
                reason: GpuFallbackReason::Unavailable,
            }
        );
    }

    #[test]
    fn cuda_driver_runtime_synthetic_snapshot_tracks_device_slots() {
        let runtime = CudaDriverRuntime::from_device_count(2);
        let snapshot = runtime.snapshot();

        assert!(snapshot.driver_available);
        assert_eq!(snapshot.driver_version, None);
        assert_eq!(snapshot.device_count, 2);
        assert_eq!(snapshot.devices.len(), 2);
        assert_eq!(snapshot.devices[0].id, 0);
        assert_eq!(snapshot.devices[0].name, "cuda-device-0");
        assert_eq!(snapshot.devices[0].total_memory_bytes, 0);
        assert_eq!(snapshot.devices[1].id, 1);
    }

    #[test]
    fn unavailable_cuda_driver_runtime_falls_back_cleanly() {
        let runtime = CudaDriverRuntime::unavailable();
        let snapshot = runtime.snapshot();

        assert!(!snapshot.driver_available);
        assert_eq!(snapshot.driver_version, None);
        assert_eq!(snapshot.device_count, 0);
        assert!(snapshot.devices.is_empty());
        assert_eq!(
            runtime.can_run(0, &gpu_op(0)),
            Err(GpuFallbackReason::Unavailable)
        );
    }

    #[test]
    fn unavailable_cuda_driver_runtime_rejects_smoke_launch() {
        let runtime = CudaDriverRuntime::unavailable();

        assert_eq!(
            runtime.launch_smoke_add_one(41),
            Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
        );
    }

    #[test]
    fn unavailable_cuda_driver_runtime_rejects_filter_launch() {
        let runtime = CudaDriverRuntime::unavailable();

        assert_eq!(
            runtime.filter_equal_u32_mask(&[7, 8, 7], 7),
            Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
        );
        assert_eq!(
            runtime.filter_equal_bytes_mask(&[b"open".as_slice()], b"open"),
            Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
        );
        assert_eq!(
            runtime.filter_bytes_range_mask(&[b"acct:1".as_slice()], b"acct:", b"acct:9"),
            Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
        );
        assert_eq!(
            runtime.filter_all_mask(3),
            Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
        );
        let batch =
            CudaMvccRowBatch::from_key_values([(b"k".as_slice(), b"v".as_slice())]).unwrap();
        assert_eq!(
            runtime.mvcc_visibility_mask(&batch, 1),
            Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
        );
        assert_eq!(
            runtime.mvcc_row_batch_lengths(&batch),
            Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
        );
        assert_eq!(
            runtime.verify_device_memory_copy(0, b"resident-snapshot"),
            Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
        );
        assert!(matches!(
            runtime.retain_device_memory_copy(0, b"resident-snapshot"),
            Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
        ));
        assert!(matches!(
            runtime.retain_device_memory_chunks(
                0,
                16,
                &[CudaDeviceMemoryChunk {
                    byte_offset: 8,
                    bytes: b"chunk",
                }]
            ),
            Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
        ));
        assert_eq!(
            runtime.verify_device_memory_copy(0, b""),
            Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
        );
    }

    #[test]
    fn cuda_mvcc_row_batch_encodes_offsets_metadata_and_transfer_size() {
        let batch = CudaMvccRowBatch::from_key_values_with_metadata([
            (b"acct:1".as_slice(), b"open".as_slice(), 3, 9, Some(11)),
            (b"acct:22".as_slice(), b"".as_slice(), 4, u64::MAX, Some(12)),
            (b"".as_slice(), b"closed".as_slice(), 5, 8, Some(13)),
        ])
        .unwrap();

        assert_eq!(batch.row_count, 3);
        assert_eq!(batch.key_offsets, vec![0, 6, 13, 13]);
        assert_eq!(batch.key_bytes, b"acct:1acct:22");
        assert_eq!(batch.value_offsets, vec![0, 4, 4, 10]);
        assert_eq!(batch.value_bytes, b"openclosed");
        assert_eq!(batch.begin_txn_ids, vec![3, 4, 5]);
        assert_eq!(batch.end_txn_ids, vec![9, u64::MAX, 8]);
        assert_eq!(batch.provenance_handles, vec![11, 12, 13]);
        assert_eq!(batch.key_len(0), Some(6));
        assert_eq!(batch.key_len(2), Some(0));
        assert_eq!(batch.value_len(1), Some(0));
        assert_eq!(batch.value_len(3), None);
        assert_eq!(
            batch.transfer_bytes(),
            (batch.key_offsets.len() + batch.value_offsets.len()) * std::mem::size_of::<u32>()
                + batch.key_bytes.len()
                + batch.value_bytes.len()
                + batch.begin_txn_ids.len() * std::mem::size_of::<u64>()
                + batch.end_txn_ids.len() * std::mem::size_of::<u64>()
                + batch.provenance_handles.len() * std::mem::size_of::<u32>()
        );
        assert_eq!(batch.validate(), Ok(()));
    }

    #[test]
    fn cuda_mvcc_row_batch_rejects_incoherent_offsets() {
        let mut batch =
            CudaMvccRowBatch::from_key_values([(b"acct:1".as_slice(), b"open".as_slice())])
                .unwrap();
        batch.key_offsets[1] = 99;

        assert_eq!(
            batch.validate(),
            Err(CudaRuntimeProbeError::InvalidInputLength(
                batch.key_bytes.len()
            ))
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_driver_runtime_probe_reports_local_devices() {
        let runtime = CudaDriverRuntime::probe().unwrap();
        let snapshot = runtime.snapshot();

        assert!(snapshot.driver_available);
        assert!(snapshot.device_count > 0);
        assert_eq!(snapshot.devices.len(), snapshot.device_count as usize);
        assert!(!snapshot.devices[0].name.is_empty());
        assert!(snapshot.devices[0].total_memory_bytes > 0);
        assert_eq!(runtime.can_run(0, &gpu_op(0)), Ok(()));
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_driver_runtime_launches_smoke_kernel() {
        let runtime = CudaDriverRuntime::probe().unwrap();

        assert_eq!(runtime.launch_smoke_add_one(41).unwrap(), 42);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_driver_runtime_filters_u32_equality_mask() {
        let runtime = CudaDriverRuntime::probe().unwrap();

        let input = [3, 8, 3, 0, 11, 3];
        let mask = runtime.filter_equal_u32_mask(&input, 3).unwrap();

        assert_eq!(mask, vec![true, false, true, false, false, true]);
        assert_eq!(
            runtime.filter_equal_u32_mask(&[], 3).unwrap(),
            Vec::<bool>::new()
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_driver_runtime_filters_all_rows_mask() {
        let runtime = CudaDriverRuntime::probe().unwrap();

        assert_eq!(runtime.filter_all_mask(4).unwrap(), vec![true; 4]);
        assert_eq!(runtime.filter_all_mask(0).unwrap(), Vec::<bool>::new());
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_driver_runtime_filters_byte_equality_mask() {
        let runtime = CudaDriverRuntime::probe().unwrap();

        let input = [
            b"open".as_slice(),
            b"closed".as_slice(),
            b"open".as_slice(),
            b"".as_slice(),
            b"opened".as_slice(),
        ];
        let mask = runtime.filter_equal_bytes_mask(&input, b"open").unwrap();

        assert_eq!(mask, vec![true, false, true, false, false]);
        assert_eq!(
            runtime
                .filter_equal_bytes_mask(&[b"".as_slice(), b"x".as_slice()], b"")
                .unwrap(),
            vec![true, false]
        );
        assert_eq!(
            runtime.filter_equal_bytes_mask(&[], b"open").unwrap(),
            Vec::<bool>::new()
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_driver_runtime_filters_byte_range_mask() {
        let runtime = CudaDriverRuntime::probe().unwrap();

        let input = [
            b"acct:0".as_slice(),
            b"acct:1".as_slice(),
            b"acct:7".as_slice(),
            b"acct:9".as_slice(),
            b"acct".as_slice(),
            b"user:1".as_slice(),
        ];
        let mask = runtime
            .filter_bytes_range_mask(&input, b"acct:1", b"acct:9")
            .unwrap();

        assert_eq!(mask, vec![false, true, true, false, false, false]);
        assert_eq!(
            runtime
                .filter_bytes_range_mask(&[b"".as_slice(), b"a".as_slice()], b"", b"a")
                .unwrap(),
            vec![true, false]
        );
        assert_eq!(
            runtime
                .filter_bytes_range_mask(&[], b"acct:1", b"acct:9")
                .unwrap(),
            Vec::<bool>::new()
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_driver_runtime_inspects_mvcc_row_batch_lengths() {
        let runtime = CudaDriverRuntime::probe().unwrap();
        let batch = CudaMvccRowBatch::from_key_values_with_metadata([
            (b"acct:1".as_slice(), b"open".as_slice(), 3, 9, Some(100)),
            (
                b"acct:22".as_slice(),
                b"".as_slice(),
                4,
                u64::MAX,
                Some(101),
            ),
            (b"".as_slice(), b"closed".as_slice(), 5, 8, Some(102)),
        ])
        .unwrap();

        assert_eq!(
            runtime.mvcc_row_batch_lengths(&batch).unwrap(),
            vec![(6, 4), (7, 0), (0, 6)]
        );
        let empty = CudaMvccRowBatch::from_key_values(Vec::<(&[u8], &[u8])>::new()).unwrap();
        assert_eq!(
            runtime.mvcc_row_batch_lengths(&empty).unwrap(),
            Vec::<(u32, u32)>::new()
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_driver_runtime_filters_mvcc_visibility_mask() {
        let runtime = CudaDriverRuntime::probe().unwrap();
        let batch = CudaMvccRowBatch::from_key_values_with_metadata([
            (b"pre".as_slice(), b"hidden".as_slice(), 4, u64::MAX, None),
            (b"live".as_slice(), b"open".as_slice(), 2, u64::MAX, None),
            (b"gone".as_slice(), b"closed".as_slice(), 1, 3, None),
            (b"edge".as_slice(), b"visible".as_slice(), 3, 5, None),
        ])
        .unwrap();

        assert_eq!(
            runtime.mvcc_visibility_mask(&batch, 3).unwrap(),
            vec![false, true, false, true]
        );
        assert_eq!(
            runtime.mvcc_visibility_mask(&batch, 5).unwrap(),
            vec![true, true, false, false]
        );
        let empty = CudaMvccRowBatch::from_key_values(Vec::<(&[u8], &[u8])>::new()).unwrap();
        assert_eq!(
            runtime.mvcc_visibility_mask(&empty, 3).unwrap(),
            Vec::<bool>::new()
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_resident_i32_equal_any_project_submit_complete_matches_sync() {
        let runtime = CudaDriverRuntime::probe().unwrap();
        let row_count = 5_u64;
        let filter_offset = std::mem::size_of::<u64>() as u64;
        let projection_offset = filter_offset + row_count * std::mem::size_of::<i32>() as u64;
        let mut header = Vec::new();
        header.extend_from_slice(&row_count.to_le_bytes());
        let mut filter = Vec::new();
        for value in [1_i32, 2, 3, 2, 4] {
            filter.extend_from_slice(&value.to_le_bytes());
        }
        let mut projection = Vec::new();
        for value in [10_i32, 20, 30, 21, 40] {
            projection.extend_from_slice(&value.to_le_bytes());
        }
        let allocated_len = projection_offset + projection.len() as u64;
        let resident = runtime
            .retain_device_memory_chunks(
                0,
                allocated_len,
                &[
                    CudaDeviceMemoryChunk {
                        byte_offset: 0,
                        bytes: &header,
                    },
                    CudaDeviceMemoryChunk {
                        byte_offset: filter_offset,
                        bytes: &filter,
                    },
                    CudaDeviceMemoryChunk {
                        byte_offset: projection_offset,
                        bytes: &projection,
                    },
                ],
            )
            .unwrap();

        let sync_rows = resident
            .match_project_i32_equal_any_from_payload(
                filter_offset,
                &[2, 4],
                &[projection_offset],
                row_count,
            )
            .unwrap();
        let async_rows = resident
            .submit_match_project_i32_equal_any_from_payload(
                filter_offset,
                &[2, 4],
                &[projection_offset],
                row_count,
            )
            .unwrap()
            .complete(&resident)
            .unwrap();
        let (read_view_rows, read_view_elapsed_us) = resident
            .read_view()
            .submit_match_project_i32_equal_any_from_payload(
                filter_offset,
                &[2, 4],
                &[projection_offset],
                row_count,
            )
            .unwrap()
            .complete_detached()
            .unwrap();

        assert_eq!(async_rows, sync_rows);
        assert_eq!(read_view_rows, sync_rows);
        assert!(read_view_elapsed_us.is_some());
        assert_eq!(
            async_rows,
            vec![
                CudaI32BatchProjectionRow {
                    needle_index: 0,
                    row_index: 1,
                    values: vec![20],
                },
                CudaI32BatchProjectionRow {
                    needle_index: 0,
                    row_index: 3,
                    values: vec![21],
                },
                CudaI32BatchProjectionRow {
                    needle_index: 1,
                    row_index: 4,
                    values: vec![40],
                },
            ]
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn published_resident_generation_survives_a_replacement_publish_and_is_freed_after_drain() {
        // P1-M3 step 1 — the real-GPU soundness probe (doc 14 acceptance gate 1).
        //
        // Retires the device-memory-lifetime risk the P1-M2 spike could only model with
        // a leaked-static buffer. With a REAL CudaResidentDeviceMemory whose Drop calls
        // the REAL cu_mem_free, it proves that under the SnapshotCell publish-on-commit
        // model a generation a reader still holds is:
        //   (a) NOT freed when the writer publishes a replacement, and still GPU-valid
        //       (a kernel read of it returns the correct rows), and
        //   (b) freed only after that last reader drains.
        // The probe cannot even compile unless `CudaResidentDeviceMemory: Send + Sync`
        // (the cell must cross the thread boundary), so it also witnesses that change.
        use gpu_db_snapshot::SnapshotCell;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Barrier};

        // Owner wrapper whose Drop sets `freed` as it is entered — immediately before
        // the inner CudaResidentDeviceMemory field drops (field-declaration order) and
        // calls the REAL cu_mem_free / cu_ctx_destroy on the same synchronous drop path.
        // So `freed == true` means that device free has entered and is about to run —
        // the Drop-observing wrapper doc 14 gate 1 sanctions. For "not freed while held"
        // this is conservative; for "freed after drain" the free is the next,
        // unconditional statements once Drop is entered.
        struct ObservableResident {
            resident: CudaResidentDeviceMemory,
            freed: Arc<AtomicBool>,
        }
        impl Drop for ObservableResident {
            fn drop(&mut self) {
                self.freed.store(true, Ordering::SeqCst);
            }
        }

        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        // Known payload: header(row_count) + i32 filter column + i32 projection column.
        let row_count = 5_u64;
        let filter_offset = std::mem::size_of::<u64>() as u64;
        let projection_offset = filter_offset + row_count * std::mem::size_of::<i32>() as u64;
        let mut header = Vec::new();
        header.extend_from_slice(&row_count.to_le_bytes());
        let mut filter = Vec::new();
        for value in [1_i32, 2, 3, 2, 4] {
            filter.extend_from_slice(&value.to_le_bytes());
        }
        let mut projection = Vec::new();
        for value in [10_i32, 20, 30, 21, 40] {
            projection.extend_from_slice(&value.to_le_bytes());
        }
        let allocated_len = projection_offset + projection.len() as u64;

        let build = |freed: &Arc<AtomicBool>| ObservableResident {
            resident: runtime
                .retain_device_memory_chunks(
                    0,
                    allocated_len,
                    &[
                        CudaDeviceMemoryChunk {
                            byte_offset: 0,
                            bytes: &header,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: filter_offset,
                            bytes: &filter,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: projection_offset,
                            bytes: &projection,
                        },
                    ],
                )
                .expect("retain resident device memory"),
            freed: Arc::clone(freed),
        };

        // needles [2,4] over filter [1,2,3,2,4] → rows 1,3 (=2) and 4 (=4),
        // projecting [20], [21], [40].
        let expected = vec![
            CudaI32BatchProjectionRow {
                needle_index: 0,
                row_index: 1,
                values: vec![20],
            },
            CudaI32BatchProjectionRow {
                needle_index: 0,
                row_index: 3,
                values: vec![21],
            },
            CudaI32BatchProjectionRow {
                needle_index: 1,
                row_index: 4,
                values: vec![40],
            },
        ];

        let freed_g1 = Arc::new(AtomicBool::new(false));
        // `SnapshotCell<Arc<owner>>` mirrors doc 14's prescribed engine residency shape
        // (the cell wraps each generation in its own `Arc`, so the inner `Arc<owner>` is
        // redundant *here* but is the type the engine application in step 2 adopts).
        let cell = Arc::new(SnapshotCell::new(Arc::new(build(&freed_g1))));

        let barrier = Arc::new(Barrier::new(2));
        let reader = {
            let cell = Arc::clone(&cell);
            let barrier = Arc::clone(&barrier);
            let expected = expected.clone();
            std::thread::spawn(move || {
                let handle = cell.load(); // pins g1 for the whole closure
                assert_eq!(handle.generation(), 1, "reader did not pin g1");
                barrier.wait(); // (1) signal: g1 is pinned
                barrier.wait(); // (2) resume only after the writer published g2

                // g2 is now current, but we still hold g1. A real GPU read of g1 must
                // still return the correct rows — proof its device memory was not freed
                // by the publish. read_view() is derived from the held owner, so the
                // owner remains the lifetime anchor; submit + complete_detached each set
                // the context current on this reader thread.
                let (rows, elapsed_us) = handle
                    .get()
                    .resident
                    .read_view()
                    .submit_match_project_i32_equal_any_from_payload(
                        filter_offset,
                        &[2, 4],
                        &[projection_offset],
                        row_count,
                    )
                    .expect("submit on pinned g1")
                    .complete_detached()
                    .expect("complete on pinned g1");
                assert_eq!(
                    rows, expected,
                    "pinned g1 returned wrong rows — freed early?"
                );
                assert!(elapsed_us.is_some(), "no CUDA-event timing from pinned g1");
                // handle drops here → releases the last reference to g1
            })
        };

        barrier.wait(); // (1) g1 is pinned by the reader
        let freed_g2 = Arc::new(AtomicBool::new(false));
        cell.publish(Arc::new(build(&freed_g2))); // writer publishes a replacement
        assert_eq!(cell.current_generation(), 2, "g2 was not published");
        assert!(
            !freed_g1.load(Ordering::SeqCst),
            "g1 was freed while a reader still held it (use-after-free risk)"
        );
        barrier.wait(); // (2) let the reader do its GPU read of g1

        reader.join().expect("reader thread panicked");
        // The reader drained → its handle (the last reference to g1) dropped, and the
        // cell holds g2, not g1. So g1 must now be reclaimed: the real cu_mem_free ran.
        assert!(
            freed_g1.load(Ordering::SeqCst),
            "g1 was not freed after its last reader drained (leak / reclamation broken)"
        );
        // g2 is still current (held by the cell), so it must still be alive.
        assert!(
            !freed_g2.load(Ordering::SeqCst),
            "current generation g2 was freed early"
        );
    }

    #[test]
    fn vec_operator_yields_rows_in_order() {
        let mut op = VecOperator::new(vec![b"row-1".to_vec(), b"row-2".to_vec()]);

        assert_eq!(op.len(), 2);
        assert!(!op.is_empty());
        assert_eq!(op.next(), Some(b"row-1".to_vec()));
        assert_eq!(op.next(), Some(b"row-2".to_vec()));
        assert_eq!(op.next(), None);
    }

    #[test]
    fn vec_operator_open_rewinds_and_close_exhausts() {
        let mut op = VecOperator::new(vec![b"row-1".to_vec()]);

        assert_eq!(op.next(), Some(b"row-1".to_vec()));
        assert_eq!(op.next(), None);

        op.open();
        assert_eq!(op.next(), Some(b"row-1".to_vec()));

        op.close();
        assert_eq!(op.next(), None);
    }

    #[test]
    fn scan_operator_yields_rows_in_order() {
        let mut op = ScanOperator::new(vec![1_u32, 2_u32, 3_u32]);

        assert_eq!(op.len(), 3);
        assert!(!op.is_empty());
        assert_eq!(op.next(), Some(1));
        assert_eq!(op.next(), Some(2));
        assert_eq!(op.next(), Some(3));
        assert_eq!(op.next(), None);

        op.open();
        assert_eq!(op.next(), Some(1));
        op.close();
        assert_eq!(op.next(), None);
    }

    #[test]
    fn filter_operator_skips_non_matching_rows() {
        let scan = ScanOperator::new(vec![1_i32, 2_i32, 3_i32, 4_i32]);
        let mut op = FilterOperator::new(scan, |row: &i32| row % 2 == 0);

        op.open();
        assert_eq!(op.next(), Some(2));
        assert_eq!(op.next(), Some(4));
        assert_eq!(op.next(), None);
        op.close();
    }

    #[test]
    fn project_operator_maps_child_rows() {
        let scan = ScanOperator::new(vec![1_i32, 2_i32, 3_i32]);
        let mut op = ProjectOperator::new(scan, |row| format!("row-{row}"));

        op.open();
        assert_eq!(op.next(), Some("row-1".to_string()));
        assert_eq!(op.next(), Some("row-2".to_string()));
        assert_eq!(op.next(), Some("row-3".to_string()));
        assert_eq!(op.next(), None);
        op.close();
    }

    #[test]
    fn limit_operator_caps_child_rows() {
        let scan = ScanOperator::new(vec![1_i32, 2_i32, 3_i32, 4_i32]);
        let mut op = LimitOperator::new(scan, 2);

        op.open();
        assert_eq!(op.next(), Some(1));
        assert_eq!(op.next(), Some(2));
        assert_eq!(op.next(), None);

        op.open();
        assert_eq!(op.next(), Some(1));
        op.close();
        assert_eq!(op.next(), None);
    }

    #[test]
    fn scan_filter_project_pipeline_composes() {
        let scan = ScanOperator::new(vec![1_i32, 2_i32, 3_i32, 4_i32]);
        let filter = FilterOperator::new(scan, |row: &i32| row % 2 == 1);
        let limit = LimitOperator::new(filter, 1);
        let mut op = ProjectOperator::new(limit, |row| row * 10);

        op.open();
        assert_eq!(op.next(), Some(10));
        assert_eq!(op.next(), None);
        op.close();
    }

    #[test]
    fn sort_operator_orders_child_rows() {
        let scan = ScanOperator::new(vec![3_i32, 1_i32, 4_i32, 2_i32]);
        let mut op = SortOperator::new(scan, |left: &i32, right: &i32| left.cmp(right));

        op.open();
        assert_eq!(op.next(), Some(1));
        assert_eq!(op.next(), Some(2));
        assert_eq!(op.next(), Some(3));
        assert_eq!(op.next(), Some(4));
        assert_eq!(op.next(), None);
        op.close();
    }

    #[test]
    fn scan_filter_sort_limit_project_pipeline_composes() {
        let scan = ScanOperator::new(vec![4_i32, 1_i32, 3_i32, 2_i32]);
        let filter = FilterOperator::new(scan, |row: &i32| row % 2 == 0);
        let sort = SortOperator::new(filter, |left: &i32, right: &i32| right.cmp(left));
        let limit = LimitOperator::new(sort, 1);
        let mut op = ProjectOperator::new(limit, |row| row * 10);

        op.open();
        assert_eq!(op.next(), Some(40));
        assert_eq!(op.next(), None);
        op.close();
    }
}
