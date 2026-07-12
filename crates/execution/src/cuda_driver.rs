use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use libloading::Library;

use crate::cuda_context::{
    check_cuda, gpu_primary_context, CudaDeviceAllocationGuard, GpuPrimaryContext,
    PendingCopyTransport,
};
use crate::{
    launch_cuda_all_mask, launch_cuda_bytes_equal_mask, launch_cuda_bytes_range_mask,
    launch_cuda_mvcc_row_batch_lengths, launch_cuda_mvcc_visibility_mask,
    launch_cuda_smoke_add_one, launch_cuda_u32_equal_mask, CudaDeviceMemoryChunk,
    CudaDeviceMemoryProof, CudaDeviceSnapshot, CudaMvccRowBatch, CudaOwnedDeviceMemoryChunk,
    CudaResidentDeviceMemory, CudaRuntimeProbeError, CudaRuntimeSnapshot, GpuFallbackReason,
    GpuRuntime, PendingCudaResidentDeviceCopy, PlannedOp, RecompactFill, RecompactSegment,
};

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
            primary: resident.primary,
            last_kernel_event_elapsed_us: Mutex::new(None),
        })
    }

    /// STRATA S-E.5 (streaming copy/compute overlap): retain a device allocation whose HtoD upload is
    /// enqueued ASYNCHRONOUSLY on a private pooled stream from a pinned staging buffer, so the DMA
    /// overlaps whatever the host (chunk staging) and the SMs (the previous chunk's kernels, on their
    /// own streams) are doing. The allocation MUST NOT be read until [`PendingCudaResidentDeviceCopy::
    /// wait`] returns. When the async symbols / pinned pool are unavailable the copy degrades to the
    /// proven synchronous path and `wait()` is a no-op — behavior-identical, just unoverlapped.
    pub fn retain_device_memory_copy_async(
        &self,
        gpu_id: u16,
        payload: &[u8],
    ) -> Result<PendingCudaResidentDeviceCopy, CudaRuntimeProbeError> {
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
        let primary = gpu_primary_context(gpu_id)?;
        primary.set_current()?;
        // Degrade to the synchronous copy when true-async staging is unavailable (missing
        // cuMemcpyHtoDAsync, or no pinned buffer — an async copy from PAGEABLE memory is not
        // asynchronous with respect to the host, so it would be overlap theater).
        let Some(htod_async) = primary.cu_memcpy_htod_async else {
            return Ok(PendingCudaResidentDeviceCopy {
                memory: Some(self.retain_device_memory_copy(gpu_id, payload)?),
                in_flight: None,
            });
        };
        let Some(pinned) = primary.lease_pinned_host_buffer(payload.len()) else {
            return Ok(PendingCudaResidentDeviceCopy {
                memory: Some(self.retain_device_memory_copy(gpu_id, payload)?),
                in_flight: None,
            });
        };
        // Stage the payload into the pinned buffer (a host memcpy — cheap next to the PCIe copy it
        // unblocks), then take OWNED custody of the lease fields (the pending handle outlives this
        // call; the transport releases them back to the pool after the sync).
        unsafe {
            std::ptr::copy_nonoverlapping(payload.as_ptr(), pinned.ptr.cast::<u8>(), payload.len());
        }
        let pinned_ptr = pinned.ptr;
        let pinned_capacity = pinned.capacity;
        std::mem::forget(pinned);

        let pooled = match primary.acquire_pooled_stream() {
            Ok(pooled) => pooled,
            Err(err) => {
                primary.release_pinned_host_buffer(pinned_ptr, pinned_capacity);
                return Err(err);
            }
        };
        let mut device_ptr = 0_u64;
        if let Err(err) =
            check_cuda(unsafe { (primary.cu_mem_alloc)(&mut device_ptr, payload.len()) })
        {
            primary.release_pooled_stream(pooled);
            primary.release_pinned_host_buffer(pinned_ptr, pinned_capacity);
            return Err(err);
        }
        // From here the allocation is RAII-owned by the memory handle (its Drop frees it).
        let memory = CudaResidentDeviceMemory {
            metadata: CudaDeviceMemoryProof {
                gpu_id,
                device_name: device.name,
                allocated_bytes: payload.len() as u64,
                copied_bytes: payload.len() as u64,
                retained: true,
            },
            device_ptr,
            primary: Arc::clone(&primary),
            last_kernel_event_elapsed_us: Mutex::new(None),
        };
        if let Err(err) = check_cuda(unsafe {
            htod_async(
                device_ptr,
                pinned_ptr.cast_const(),
                payload.len(),
                pooled.stream,
            )
        }) {
            primary.release_pooled_stream(pooled);
            primary.release_pinned_host_buffer(pinned_ptr, pinned_capacity);
            return Err(err);
        }
        Ok(PendingCudaResidentDeviceCopy {
            memory: Some(memory),
            in_flight: Some(PendingCopyTransport {
                primary,
                pooled: Some(pooled),
                pinned: Some((pinned_ptr, pinned_capacity)),
            }),
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
            primary: resident.primary,
            last_kernel_event_elapsed_us: Mutex::new(None),
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
            primary: resident.primary,
            last_kernel_event_elapsed_us: Mutex::new(None),
        })
    }

    /// S10c slice 2a: build ONE unified resident buffer fully ON-DEVICE by allocating
    /// `allocated_bytes` in the GPU's shared primary context, HtoD-copying the 8-byte row-count
    /// `header` to offset 0, then DEVICE-TO-DEVICE copying each [`RecompactSegment`] from its source
    /// resident allocation into the unified buffer. The host never sees the column bytes (only the
    /// tiny header crosses HtoD). Mirrors `retain_device_memory_owned_chunks`'s allocate/build/proof
    /// shape; the resulting proof's `copied_bytes`/`allocated_bytes` are both `allocated_bytes`
    /// (the unified buffer is fully populated) and `retained == true`.
    pub fn retain_device_memory_recompacted(
        &self,
        gpu_id: u16,
        allocated_bytes: u64,
        header: &[u8],
        fills: &[RecompactFill],
        segments: &[RecompactSegment],
    ) -> Result<CudaResidentDeviceMemory, CudaRuntimeProbeError> {
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
        let resident = launch_cuda_resident_device_memory_recompacted(
            gpu_id,
            allocated_len,
            header,
            fills,
            segments,
        )?;
        Ok(CudaResidentDeviceMemory {
            metadata: CudaDeviceMemoryProof {
                gpu_id,
                device_name: device.name,
                allocated_bytes,
                copied_bytes: allocated_bytes,
                retained: true,
            },
            device_ptr: resident.device_ptr,
            primary: resident.primary,
            last_kernel_event_elapsed_us: Mutex::new(None),
        })
    }
}

pub(super) struct RawCudaResidentDeviceMemory {
    pub(super) device_ptr: u64,
    pub(super) primary: Arc<GpuPrimaryContext>,
    copied_bytes: u64,
}

pub(super) fn launch_cuda_resident_device_memory(
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
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;

    // §9.3: allocate into the shared primary context for this GPU (created + cached on
    // first use), not a fresh per-allocation context. Make it current on this thread so
    // the allocation and host→device copies land in it.
    let primary = gpu_primary_context(gpu_id)?;
    primary.set_current()?;

    let cu_mem_alloc = unsafe {
        *primary
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| primary.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        *primary
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| primary.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        *primary
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
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

    Ok(RawCudaResidentDeviceMemory {
        device_ptr,
        primary,
        copied_bytes,
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
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;

    // §9.3: allocate into the shared primary context for this GPU (created + cached on
    // first use), not a fresh per-allocation context. Make it current on this thread so
    // the allocation and host→device copies land in it.
    let primary = gpu_primary_context(gpu_id)?;
    primary.set_current()?;

    let cu_mem_alloc = unsafe {
        *primary
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| primary.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        *primary
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| primary.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        *primary
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
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

    Ok(RawCudaResidentDeviceMemory {
        device_ptr,
        primary,
        copied_bytes,
    })
}

fn launch_cuda_resident_device_memory_recompacted(
    gpu_id: u16,
    allocated_len: usize,
    header: &[u8],
    fills: &[RecompactFill],
    segments: &[RecompactSegment],
) -> Result<RawCudaResidentDeviceMemory, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoD = unsafe extern "C" fn(u64, u64, usize) -> i32;
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;

    // §9.3: allocate into the shared primary context for this GPU (created + cached on
    // first use), not a fresh per-allocation context. Make it current on this thread so
    // the allocation, the header host→device copy, and the device→device copies land in it.
    let primary = gpu_primary_context(gpu_id)?;
    primary.set_current()?;

    let cu_mem_alloc = unsafe {
        *primary
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| primary.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        *primary
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| primary.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        *primary
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtod = unsafe {
        *primary
            .lib()
            .get::<CuMemcpyDtoD>(b"cuMemcpyDtoD_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyDtoD>(b"cuMemcpyDtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memset_d8 = if fills.is_empty() {
        None
    } else {
        Some(unsafe {
            *primary
                .lib()
                .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
                .or_else(|_| primary.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        })
    };

    let allocated_bytes = allocated_len as u64;
    let mut device_ptr = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_ptr, allocated_len) })?;
    let allocation_guard = CudaDeviceAllocationGuard {
        ptr: device_ptr,
        free: cu_mem_free,
    };

    // The 8-byte row-count header is the only host bytes that cross HtoD; it lands at offset 0.
    if !header.is_empty() {
        let end = header.len() as u64;
        if end > allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(end).unwrap_or(usize::MAX),
            ));
        }
        check_cuda(unsafe {
            cu_memcpy_htod(
                allocation_guard.ptr,
                header.as_ptr().cast::<c_void>(),
                header.len(),
            )
        })?;
    }

    // SV3: fills run BEFORE the segment copies — they initialize a section (e.g. a gathered `deleted_by`
    // born all-live) that segments then partially overwrite. `cuMemAlloc` leaves the buffer uninitialized,
    // so a fill is the only way an un-segment-covered section reads a defined value.
    if let Some(cu_memset_d8) = cu_memset_d8 {
        for fill in fills {
            if fill.len == 0 {
                continue;
            }
            let end = fill
                .byte_offset
                .checked_add(fill.len)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if end > allocated_bytes {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    usize::try_from(end).unwrap_or(usize::MAX),
                ));
            }
            let destination = allocation_guard
                .ptr
                .checked_add(fill.byte_offset)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            let len = usize::try_from(fill.len)
                .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            check_cuda(unsafe { cu_memset_d8(destination, fill.fill_byte, len) })?;
        }
    }

    // Each segment is a device→device copy from a source resident allocation into the unified
    // buffer; the host never touches the column bytes.
    for segment in segments {
        if segment.byte_len == 0 {
            continue;
        }
        let end = segment
            .dst_byte_offset
            .checked_add(segment.byte_len)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if end > allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(end).unwrap_or(usize::MAX),
            ));
        }
        let destination = allocation_guard
            .ptr
            .checked_add(segment.dst_byte_offset)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let source = segment
            .src_device_ptr
            .checked_add(segment.src_byte_offset)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let byte_len = usize::try_from(segment.byte_len)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        check_cuda(unsafe { cu_memcpy_dtod(destination, source, byte_len) })?;
    }

    std::mem::forget(allocation_guard);

    Ok(RawCudaResidentDeviceMemory {
        device_ptr,
        primary,
        copied_bytes: allocated_bytes,
    })
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
