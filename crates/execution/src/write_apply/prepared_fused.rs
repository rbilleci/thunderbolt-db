//! Move-only pre-WAL ownership for one fused i32 append launch.
//!
//! This is intentionally an execution-only capability: it proves the prepared launch boundary
//! without selecting it from the engine's live mutation route. Preparation owns every heap/GPU
//! resource that the post-WAL launch needs; consumption only writes stamps into its exact boxed
//! image and performs the ordered CUDA sequence.

use super::*;
#[cfg(any(test, feature = "probe-timing"))]
use std::cell::Cell;

type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
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

/// The pre-WAL inputs whose row contents and destinations are already fixed. Commit stamps are
/// intentionally absent: [`PreparedI32FusedApply::apply`] receives that one late scalar slice.
pub struct FusedApplyPreparation<'a> {
    pub columns: &'a [CudaWriteDestination],
    pub values: &'a [i32],
    pub created_by: CudaWriteDestination,
    pub row_ids: Option<(&'a [u64], CudaWriteDestination)>,
    pub index: Option<CudaWriteIndex>,
    pub base_row: u32,
    pub header: CudaWriteDestination,
}

/// Allocation-free resource accounting for one [`FusedApplyPreparation`].
///
/// This is intentionally a pre-materialization API: all byte counts are derived and validated
/// before the prepared token acquires either host backing or its pooled CUDA scratch lease.
/// `temporary_destination_array_bytes` is zero because preparation writes destination pointers
/// directly into the final staging image. Likewise, exact boxed owner construction has no spare
/// capacity. The only temporary host backing is the explicitly NUL-terminated PTX image needed
/// to prewarm the CUDA module cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FusedApplyPreparationFootprint {
    /// Number of distinct underlying CUDA allocations pinned by the token, not wrapper count.
    pub pinned_cuda_allocation_count: u64,
    /// Exact bytes and allocation slots retained for the boxed CUDA-owner array.
    pub owner_array_backing_bytes: u64,
    pub owner_array_allocation_slots: u64,
    /// Exact bytes and allocation slots retained for the boxed host staging image.
    pub staging_backing_bytes: u64,
    pub staging_allocation_slots: u64,
    /// One checked pool bucket retained on the device for the exact staging image.
    pub pooled_device_scratch_bytes: u64,
    pub pooled_device_scratch_slots: u64,
    /// Bounded DtoH decline word; it is stack-backed and has no host allocation slot.
    pub status_readback_bytes: u64,
    /// No destination-pointer Vec is materialized: pointers are written directly to staging.
    pub temporary_destination_array_bytes: u64,
    /// Exact temporary PTX byte image, including its trailing NUL.
    pub temporary_ptx_nul_staging_bytes: u64,
    /// Exact boxed owner construction leaves no capacity spare before becoming token retention.
    pub owner_construction_spare_bytes: u64,
    /// Largest concurrently live temporary host backing during preparation.
    pub maximum_temporary_host_scratch_bytes: u64,
}

impl FusedApplyPreparation<'_> {
    /// Validate this preparation and report all retained/preparation resource geometry without
    /// allocating host memory, leasing CUDA memory, or loading a CUDA module.
    pub fn footprint(
        &self,
        source: &CudaResidentDeviceMemory,
    ) -> Result<FusedApplyPreparationFootprint, CudaRuntimeProbeError> {
        Ok(validate_fused_apply_preparation(source, self)?.footprint)
    }
}

/// Allocation-free exact resource geometry for an i32 fused apply whose optional index tail is
/// absent.  Callers must prove the exact count of distinct CUDA allocations that the eventual
/// source/header allocation, column destinations, `created_by`, and row-id destination will
/// pin; the scalar API deliberately accepts no device pointer or allocation owner. The header
/// is required to alias the source allocation, so it does not consume a separate owner slot.
///
/// This is the admission counterpart to [`FusedApplyPreparation::footprint`].  The latter
/// validates concrete device spans and independently recomputes this same shape, so a later
/// prepared token can reject any owner/geometry drift rather than trusting a capacity estimate.
pub fn i32_fused_apply_footprint_for_shape(
    row_count: usize,
    column_count: usize,
    has_row_ids: bool,
    pinned_cuda_allocation_count: u64,
) -> Result<FusedApplyPreparationFootprint, CudaRuntimeProbeError> {
    let shape = fused_i32_apply_shape_geometry(row_count, column_count, has_row_ids)?;
    let max_pinned_allocations = column_count
        .checked_add(2)
        .and_then(|count| count.checked_add(usize::from(has_row_ids)))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let pinned_cuda_allocation_count = usize::try_from(pinned_cuda_allocation_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if pinned_cuda_allocation_count == 0 || pinned_cuda_allocation_count > max_pinned_allocations {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            pinned_cuda_allocation_count,
        ));
    }
    let owner_element_bytes =
        std::mem::size_of::<Arc<crate::resident_memory::CudaResidentDeviceAllocation>>();
    let owner_array_backing_bytes = pinned_cuda_allocation_count
        .checked_mul(owner_element_bytes)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let staging_backing_bytes = u64::try_from(shape.total_staging_bytes)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let pooled_device_scratch_bytes =
        crate::cuda_context::checked_output_buffer_bucket(shape.total_staging_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let temporary_ptx_nul_staging_bytes = FUSED_APPLY_PTX
        .len()
        .checked_add(1)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    Ok(FusedApplyPreparationFootprint {
        pinned_cuda_allocation_count: u64::try_from(pinned_cuda_allocation_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        owner_array_backing_bytes,
        owner_array_allocation_slots: u64::from(owner_array_backing_bytes != 0),
        staging_backing_bytes,
        staging_allocation_slots: u64::from(staging_backing_bytes != 0),
        pooled_device_scratch_bytes,
        pooled_device_scratch_slots: u64::from(pooled_device_scratch_bytes != 0),
        status_readback_bytes: std::mem::size_of::<u32>() as u64,
        temporary_destination_array_bytes: 0,
        temporary_ptx_nul_staging_bytes,
        owner_construction_spare_bytes: 0,
        maximum_temporary_host_scratch_bytes: temporary_ptx_nul_staging_bytes,
    })
}

#[derive(Debug, Clone, Copy)]
struct FusedI32ApplyShapeGeometry {
    k_u32: u32,
    cols_u32: u32,
    value_bytes_per_column: usize,
    header_bytes: usize,
    values_padded: usize,
    stamp_bytes: usize,
    total_staging_bytes: usize,
}

fn fused_i32_apply_shape_geometry(
    row_count: usize,
    column_count: usize,
    has_row_ids: bool,
) -> Result<FusedI32ApplyShapeGeometry, CudaRuntimeProbeError> {
    if row_count == 0 || column_count == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            row_count.max(column_count),
        ));
    }
    let k_u32 = u32::try_from(row_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(row_count))?;
    let cols_u32 = u32::try_from(column_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(column_count))?;
    let value_count = column_count
        .checked_mul(row_count)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if usize::try_from(
        cols_u32
            .checked_mul(k_u32)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(value_count))?,
    )
    .ok()
        != Some(value_count)
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(value_count));
    }
    let value_bytes_per_column = row_count
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let stamp_bytes = row_count
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let values_bytes = value_count
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let values_padded = values_bytes
        .checked_add(7)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?
        / 8
        * 8;
    let header_bytes = column_count
        .checked_mul(std::mem::size_of::<u64>())
        .and_then(|bytes| bytes.checked_add(80))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let total_staging_bytes = header_bytes
        .checked_add(values_padded)
        .and_then(|bytes| bytes.checked_add(stamp_bytes))
        .and_then(|bytes| bytes.checked_add(if has_row_ids { stamp_bytes } else { 0 }))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    Ok(FusedI32ApplyShapeGeometry {
        k_u32,
        cols_u32,
        value_bytes_per_column,
        header_bytes,
        values_padded,
        stamp_bytes,
        total_staging_bytes,
    })
}

#[derive(Debug, Clone, Copy)]
struct FusedApplyValidatedGeometry {
    k: usize,
    k_u32: u32,
    cols_u32: u32,
    new_row_count: u32,
    value_bytes_per_column: usize,
    header_bytes: usize,
    values_padded: usize,
    stamp_bytes: usize,
    total_staging_bytes: usize,
    pk_col: u32,
    index_ptr: u64,
    index_mask: u32,
    index_shift: u32,
    created_by_dest: u64,
    row_id_dest: u64,
    header_dest: u64,
    footprint: FusedApplyPreparationFootprint,
}

/// Exact host ownership retained by one prepared launch. The device lease is charged by the
/// execution pool, while this records the two host backings that survive the WAL boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PreparedI32FusedApplyHostRetention {
    pub owner_array_backing_identity: Option<usize>,
    pub owner_array_element_count: u64,
    pub owner_array_backing_bytes: u64,
    pub owner_array_allocation_slots: u64,
    pub staging_backing_identity: Option<usize>,
    pub staging_backing_bytes: u64,
    pub staging_allocation_slots: u64,
}

/// A single-use, pre-WAL fused apply. Its declaration order is load-bearing: a launched default
/// stream is drained before the pooled scratch lease can re-enter the shared pool.
#[must_use]
pub struct PreparedI32FusedApply {
    primary: Arc<crate::GpuPrimaryContext>,
    // Strong guards pin the source and every destination allocation across WAL/cache retirement.
    _owners: Box<[Arc<crate::resident_memory::CudaResidentDeviceAllocation>]>,
    // The exact prebuilt image owns static header/destinations/values/row IDs. `apply` mutates
    // only its stamp range and decline word, with no Vec/String/cache-key construction.
    staging: Box<[u8]>,
    // Must drop after the optional drain below.
    staging_guard: crate::PooledDeviceBufferOwned,
    function: *mut c_void,
    cu_memcpy_htod: CuMemcpyHtoD,
    cu_memcpy_dtoh: CuMemcpyDtoH,
    cu_launch_kernel: CuLaunchKernel,
    cu_stream_sync: CuStreamSync,
    header_dest: u64,
    header_value: [u8; std::mem::size_of::<u64>()],
    stamps_offset: usize,
    stamp_count: usize,
    blocks: u32,
}

/// The visibility half of one prepared fused append.
///
/// [`PreparedI32FusedApply::apply_before_header`] returns this token only after the payload,
/// identity sidecars, and bounded status fence have completed. Keeping the allocation owners here
/// pins the exact header destination while an engine submits its separately prepared index tail.
/// Publication performs one already-resolved HtoD and cannot be cloned or rebuilt.
#[must_use]
pub struct PreparedI32FusedHeader {
    primary: Arc<crate::GpuPrimaryContext>,
    _owners: Box<[Arc<crate::resident_memory::CudaResidentDeviceAllocation>]>,
    cu_memcpy_htod: CuMemcpyHtoD,
    header_dest: u64,
    header_value: [u8; std::mem::size_of::<u64>()],
}

// SAFETY: all device ownership is pinned, context operations rebind `primary`, and consuming
// apply is single-use. The token is intentionally not Sync.
unsafe impl Send for PreparedI32FusedApply {}
// SAFETY: the same pinned allocations and primary context are retained by the split header token.
unsafe impl Send for PreparedI32FusedHeader {}

impl PreparedI32FusedApply {
    /// Exact pre-acquired pooled scratch capacity retained across the durability boundary.
    pub fn preparation_bytes(&self) -> u64 {
        u64::try_from(self.staging_guard.capacity).unwrap_or(u64::MAX)
    }

    /// The host allocation accounting is deliberately observable without exposing owner Arcs or
    /// staging bytes for mutation. Empty/ZST backing has no allocation slot.
    pub fn host_retention_report(
        &self,
    ) -> Result<PreparedI32FusedApplyHostRetention, CudaRuntimeProbeError> {
        let owner_elements = self._owners.len();
        let owner_bytes = if owner_elements == 0
            || std::mem::size_of::<Arc<crate::resident_memory::CudaResidentDeviceAllocation>>() == 0
        {
            0
        } else {
            u64::try_from(owner_elements)
                .ok()
                .and_then(|count| {
                    u64::try_from(std::mem::size_of::<
                        Arc<crate::resident_memory::CudaResidentDeviceAllocation>,
                    >())
                    .ok()
                    .and_then(|width| count.checked_mul(width))
                })
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(owner_elements))?
        };
        let staging_bytes = u64::try_from(self.staging.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(self.staging.len()))?;
        Ok(PreparedI32FusedApplyHostRetention {
            owner_array_backing_identity: (owner_bytes != 0)
                .then_some(self._owners.as_ptr() as usize),
            owner_array_element_count: u64::try_from(owner_elements)
                .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(owner_elements))?,
            owner_array_backing_bytes: owner_bytes,
            owner_array_allocation_slots: u64::from(owner_bytes != 0),
            staging_backing_identity: (staging_bytes != 0)
                .then_some(self.staging.as_ptr() as usize),
            staging_backing_bytes: staging_bytes,
            staging_allocation_slots: u64::from(staging_bytes != 0),
        })
    }

    /// Consume the payload half after WAL, but deliberately leave the row-count header unpublished.
    ///
    /// This path has no heap allocation, PTX/module/cache work, source/destination rebuilding, or
    /// driver symbol lookup. A successful HtoD arms the drain before launch; every later error
    /// synchronizes the default stream before the leased scratch is returned to its pool. Success
    /// returns the only token capable of publishing the prebuilt count header.
    pub fn apply_before_header(
        mut self,
        stamps: &[u64],
    ) -> Result<(crate::CudaResidentIndexStatus, PreparedI32FusedHeader), CudaRuntimeProbeError>
    {
        if stamps.len() != self.stamp_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(stamps.len()));
        }
        for (position, stamp) in stamps.iter().copied().enumerate() {
            let offset = self
                .stamps_offset
                .checked_add(
                    position
                        .checked_mul(std::mem::size_of::<u64>())
                        .ok_or(CudaRuntimeProbeError::InvalidInputLength(position))?,
                )
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(position))?;
            let end = offset
                .checked_add(std::mem::size_of::<u64>())
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(position))?;
            let slot = self
                .staging
                .get_mut(offset..end)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(position))?;
            slot.copy_from_slice(&stamp.to_le_bytes());
        }
        self.staging[28..32].fill(0);
        self.primary.set_current()?;
        // A failed HtoD has not launched a kernel, so no poisoned invisible headroom exists and
        // no stream drain is needed. Once it succeeds, the launch/error tail owns the drain.
        check_cuda(unsafe {
            (self.cu_memcpy_htod)(
                self.staging_guard.ptr,
                self.staging.as_ptr().cast::<c_void>(),
                self.staging.len(),
            )
        })?;
        let mut stream_drain = DefaultStreamDrain {
            sync: self.cu_stream_sync,
            armed: true,
            #[cfg(any(test, feature = "probe-timing"))]
            drain_counter: Some(record_prepared_fused_drain),
        };
        let mut staging_arg = self.staging_guard.ptr;
        let mut args = [(&mut staging_arg as *mut u64).cast::<c_void>()];
        check_cuda(unsafe {
            (self.cu_launch_kernel)(
                self.function,
                self.blocks,
                1,
                1,
                128,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;
        #[cfg(test)]
        if take_fail_after_launch() {
            return Err(CudaRuntimeProbeError::KernelLaunchFailed(-1));
        }
        let mut decline = 0_u32;
        check_cuda(unsafe {
            (self.cu_memcpy_dtoh)(
                (&mut decline as *mut u32).cast::<c_void>(),
                self.staging_guard.ptr + 28,
                std::mem::size_of::<u32>(),
            )
        })?;
        #[cfg(test)]
        if take_fail_after_status() {
            return Err(CudaRuntimeProbeError::KernelLaunchFailed(-2));
        }
        #[cfg(test)]
        if take_fail_before_header() {
            return Err(CudaRuntimeProbeError::KernelLaunchFailed(-3));
        }
        stream_drain.armed = false;
        let header = PreparedI32FusedHeader {
            primary: Arc::clone(&self.primary),
            _owners: self._owners,
            cu_memcpy_htod: self.cu_memcpy_htod,
            header_dest: self.header_dest,
            header_value: self.header_value,
        };
        Ok((crate::CudaResidentIndexStatus::from_bits(decline), header))
    }

    /// Preserve the original one-call behavior for unindexed users while routing through the
    /// split, ordering-capable implementation.
    pub fn apply(
        self,
        stamps: &[u64],
    ) -> Result<crate::CudaResidentIndexStatus, CudaRuntimeProbeError> {
        let (status, header) = self.apply_before_header(stamps)?;
        header.publish()?;
        Ok(status)
    }
}

impl PreparedI32FusedHeader {
    /// Publish the already-prepared row-count header after every separately prepared index tail has
    /// succeeded. This is one resolved driver call over pinned memory and performs no allocation,
    /// module/cache lookup, descriptor construction, or fallback.
    pub fn publish(self) -> Result<(), CudaRuntimeProbeError> {
        self.primary.set_current()?;
        check_cuda(unsafe {
            (self.cu_memcpy_htod)(
                self.header_dest,
                self.header_value.as_ptr().cast::<c_void>(),
                self.header_value.len(),
            )
        })?;
        #[cfg(any(test, feature = "probe-timing"))]
        PREPARED_FUSED_SUBMITS.with(|value| value.set(value.get() + 1));
        Ok(())
    }
}

impl CudaResidentDeviceMemory {
    /// Build the move-only fused apply owner before WAL. This is deliberately not used by
    /// `submit_i32_fused_apply_status` or any engine strategy: live promotion requires a distinct
    /// WRITE-001 carrier that owns source values/destinations/currentness before the WAL boundary.
    pub fn prepare_i32_fused_apply(
        &self,
        preparation: &FusedApplyPreparation<'_>,
    ) -> Result<PreparedI32FusedApply, CudaRuntimeProbeError> {
        let geometry = validate_fused_apply_preparation(self, preparation)?;
        let primary = self.primary_arc();
        let owner_count = usize::try_from(geometry.footprint.pinned_cuda_allocation_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        // This boxed backing has exactly one slot per distinct allocation and becomes the token's
        // retained owner array directly: no Vec capacity spare can escape accounting.
        let mut owner_slots =
            Box::<[Arc<crate::resident_memory::CudaResidentDeviceAllocation>]>::new_uninit_slice(
                owner_count,
            );
        let mut owner_count_written = 0;
        visit_preparation_owner_memories(self, preparation, |position, memory| {
            if !owner_memory_seen_before(self, preparation, position, memory) {
                owner_slots[owner_count_written].write(memory.allocation_arc());
                owner_count_written += 1;
            }
        });
        assert_eq!(
            owner_count_written, owner_count,
            "validated owner cardinality must match exact boxed materialization"
        );
        // SAFETY: the allocation-free validation counted the same immutable owner sequence and
        // the loop above initializes exactly that many distinct slots.
        let owners = unsafe { owner_slots.assume_init() };

        // The staging backing is final token retention, not a temporary Vec. Zero initialization
        // preserves the prebuilt mutable stamp/decline ranges and the intentional header padding.
        let mut staging = unsafe {
            // SAFETY: `u8` accepts the all-zero bit pattern.
            Box::<[u8]>::new_zeroed_slice(geometry.total_staging_bytes).assume_init()
        };
        staging[0..4].copy_from_slice(&geometry.k_u32.to_le_bytes());
        staging[4..8].copy_from_slice(&geometry.cols_u32.to_le_bytes());
        staging[8..12].copy_from_slice(&geometry.pk_col.to_le_bytes());
        staging[12..16].copy_from_slice(&preparation.base_row.to_le_bytes());
        staging[16..20].copy_from_slice(&geometry.index_mask.to_le_bytes());
        staging[20..24].copy_from_slice(&geometry.index_shift.to_le_bytes());
        staging[24..28].copy_from_slice(&u32::from(preparation.row_ids.is_some()).to_le_bytes());
        staging[32..40].copy_from_slice(&geometry.index_ptr.to_le_bytes());
        staging[40..48].copy_from_slice(&geometry.created_by_dest.to_le_bytes());
        staging[48..56].copy_from_slice(&geometry.row_id_dest.to_le_bytes());
        for (column, destination) in preparation.columns.iter().enumerate() {
            let destination = checked_destination(
                &primary,
                destination,
                geometry.value_bytes_per_column as u64,
                std::mem::align_of::<i32>() as u64,
            )?;
            let offset = 80 + column * 8;
            staging[offset..offset + 8].copy_from_slice(&destination.to_le_bytes());
        }
        let values_offset = geometry.header_bytes;
        for (position, value) in preparation.values.iter().copied().enumerate() {
            let offset = values_offset + position * std::mem::size_of::<i32>();
            staging[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        let stamps_offset = values_offset + geometry.values_padded;
        if let Some((row_ids, _)) = preparation.row_ids.as_ref() {
            let row_ids_offset = stamps_offset + geometry.stamp_bytes;
            for (position, row_id) in row_ids.iter().copied().enumerate() {
                let offset = row_ids_offset + position * std::mem::size_of::<u64>();
                staging[offset..offset + 8].copy_from_slice(&row_id.to_le_bytes());
            }
        }
        primary.set_current()?;
        let cu_memcpy_htod = unsafe {
            *primary
                .lib()
                .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_memcpy_dtoh = unsafe {
            *primary
                .lib()
                .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_launch_kernel = unsafe {
            *primary
                .lib()
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_stream_sync = unsafe {
            *primary
                .lib()
                .get::<CuStreamSync>(b"cuStreamSynchronize\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let staging_guard = primary.lease_device_buffer_owned(geometry.total_staging_bytes)?;
        let ptx_len = usize::try_from(geometry.footprint.temporary_ptx_nul_staging_bytes)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        // The module loader requires a NUL-terminated image. This exact boxed temporary is the
        // entire reported preparation scratch peak; it drops immediately after cache prewarm.
        let mut ptx = Box::<[u8]>::new_uninit_slice(ptx_len);
        for (slot, byte) in ptx.iter_mut().zip(FUSED_APPLY_PTX.iter().copied()) {
            slot.write(byte);
        }
        ptx[FUSED_APPLY_PTX.len()].write(0);
        // SAFETY: every byte was initialized by the copy plus the trailing NUL above.
        let ptx = unsafe { ptx.assume_init() };
        let function = primary.cached_function(c"gpu_db_resident_i32_fused_apply", &ptx)?;
        #[cfg(any(test, feature = "probe-timing"))]
        PREPARED_FUSED_PREPARES.with(|value| value.set(value.get() + 1));
        Ok(PreparedI32FusedApply {
            primary,
            _owners: owners,
            staging,
            staging_guard,
            function,
            cu_memcpy_htod,
            cu_memcpy_dtoh,
            cu_launch_kernel,
            cu_stream_sync,
            header_dest: geometry.header_dest,
            header_value: u64::from(geometry.new_row_count).to_le_bytes(),
            stamps_offset,
            stamp_count: geometry.k,
            blocks: geometry.k_u32.div_ceil(128),
        })
    }
}

fn validate_fused_apply_preparation(
    source: &CudaResidentDeviceMemory,
    preparation: &FusedApplyPreparation<'_>,
) -> Result<FusedApplyValidatedGeometry, CudaRuntimeProbeError> {
    let num_cols = preparation.columns.len();
    if num_cols == 0 || preparation.values.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(num_cols));
    }
    let k = preparation.values.len() / num_cols;
    if !preparation.values.len().is_multiple_of(num_cols) || k == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            preparation.values.len(),
        ));
    }
    if let Some((row_ids, _)) = preparation.row_ids.as_ref() {
        if row_ids.len() != k {
            return Err(CudaRuntimeProbeError::InvalidInputLength(row_ids.len()));
        }
    }
    let shape = fused_i32_apply_shape_geometry(k, num_cols, preparation.row_ids.is_some())?;
    let k_u32 = shape.k_u32;
    let cols_u32 = shape.cols_u32;
    let new_row_count = preparation
        .base_row
        .checked_add(k_u32)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(k))?;
    let value_bytes_per_column = shape.value_bytes_per_column;
    let stamp_bytes = shape.stamp_bytes;
    let primary = source.primary_arc();
    for destination in preparation.columns {
        checked_destination(
            &primary,
            destination,
            value_bytes_per_column as u64,
            std::mem::align_of::<i32>() as u64,
        )?;
    }
    let created_by_dest = checked_destination(
        &primary,
        &preparation.created_by,
        stamp_bytes as u64,
        std::mem::align_of::<u64>() as u64,
    )?;
    let row_id_dest = if let Some((_, destination)) = preparation.row_ids.as_ref() {
        checked_destination(
            &primary,
            destination,
            stamp_bytes as u64,
            std::mem::align_of::<u64>() as u64,
        )?
    } else {
        0
    };
    let header_dest = checked_destination(
        &primary,
        &preparation.header,
        std::mem::size_of::<u64>() as u64,
        std::mem::align_of::<u64>() as u64,
    )?;
    if preparation.header.memory.allocation_identity() != source.allocation_identity() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    let (pk_col, index_ptr, index_mask, index_shift) =
        if let Some(index) = preparation.index.as_ref() {
            if !Arc::ptr_eq(&primary, &index.memory.primary_arc()) || index.key_column >= cols_u32 {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    index.key_column as usize,
                ));
            }
            validate_index_append(
                &index.memory,
                index.table_mask,
                index.hash_shift,
                preparation.base_row,
                k_u32,
            )?;
            (
                index.key_column,
                index.memory.device_ptr(),
                index.table_mask,
                index.hash_shift,
            )
        } else {
            (u32::MAX, 0, 0, 0)
        };
    let distinct_owner_count = distinct_preparation_owner_count(source, preparation)?;
    let footprint = i32_fused_apply_footprint_for_shape(
        k,
        num_cols,
        preparation.row_ids.is_some(),
        u64::try_from(distinct_owner_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )?;
    Ok(FusedApplyValidatedGeometry {
        k,
        k_u32,
        cols_u32,
        new_row_count,
        value_bytes_per_column,
        header_bytes: shape.header_bytes,
        values_padded: shape.values_padded,
        stamp_bytes: shape.stamp_bytes,
        total_staging_bytes: shape.total_staging_bytes,
        pk_col,
        index_ptr,
        index_mask,
        index_shift,
        created_by_dest,
        row_id_dest,
        header_dest,
        footprint,
    })
}

fn distinct_preparation_owner_count(
    source: &CudaResidentDeviceMemory,
    preparation: &FusedApplyPreparation<'_>,
) -> Result<usize, CudaRuntimeProbeError> {
    let candidate_upper_bound = preparation
        .columns
        .len()
        .checked_add(2)
        .and_then(|count| count.checked_add(usize::from(preparation.row_ids.is_some())))
        .and_then(|count| count.checked_add(usize::from(preparation.index.is_some())))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut distinct = 0_usize;
    visit_preparation_owner_memories(source, preparation, |position, memory| {
        if !owner_memory_seen_before(source, preparation, position, memory) {
            distinct = distinct
                .checked_add(1)
                .expect("owner count is bounded by the checked candidate count");
        }
    });
    debug_assert!(distinct <= candidate_upper_bound);
    Ok(distinct)
}

/// Visit owner candidates in the exact pinning order. The nested rescan in
/// [`owner_memory_seen_before`] is intentional: preparation cardinalities are bounded and this
/// avoids a map/set allocation before materialization while deduplicating allocation aliases.
fn visit_preparation_owner_memories(
    source: &CudaResidentDeviceMemory,
    preparation: &FusedApplyPreparation<'_>,
    mut visit: impl FnMut(usize, &CudaResidentDeviceMemory),
) {
    let mut position = 0;
    visit(position, source);
    position += 1;
    for destination in preparation.columns {
        visit(position, &destination.memory);
        position += 1;
    }
    visit(position, &preparation.created_by.memory);
    position += 1;
    if let Some((_, destination)) = preparation.row_ids.as_ref() {
        visit(position, &destination.memory);
        position += 1;
    }
    visit(position, &preparation.header.memory);
    position += 1;
    if let Some(index) = preparation.index.as_ref() {
        visit(position, &index.memory);
    }
}

fn owner_memory_seen_before(
    source: &CudaResidentDeviceMemory,
    preparation: &FusedApplyPreparation<'_>,
    target_position: usize,
    target: &CudaResidentDeviceMemory,
) -> bool {
    let target_identity = target.allocation_identity();
    let mut seen = false;
    visit_preparation_owner_memories(source, preparation, |position, candidate| {
        if position < target_position && candidate.allocation_identity() == target_identity {
            seen = true;
        }
    });
    seen
}

#[cfg(any(test, feature = "probe-timing"))]
thread_local! {
    static PREPARED_FUSED_PREPARES: Cell<u64> = const { Cell::new(0) };
    static PREPARED_FUSED_SUBMITS: Cell<u64> = const { Cell::new(0) };
    static PREPARED_FUSED_DRAINS: Cell<u64> = const { Cell::new(0) };
}

#[cfg(any(test, feature = "probe-timing"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedI32FusedApplyCounters {
    pub prepares: u64,
    pub submits: u64,
    pub drains: u64,
}

#[cfg(any(test, feature = "probe-timing"))]
pub fn prepared_i32_fused_apply_counters() -> PreparedI32FusedApplyCounters {
    PreparedI32FusedApplyCounters {
        prepares: PREPARED_FUSED_PREPARES.with(Cell::get),
        submits: PREPARED_FUSED_SUBMITS.with(Cell::get),
        drains: PREPARED_FUSED_DRAINS.with(Cell::get),
    }
}

#[cfg(test)]
thread_local! {
    static FAIL_AFTER_LAUNCH: Cell<bool> = const { Cell::new(false) };
    static FAIL_AFTER_STATUS: Cell<bool> = const { Cell::new(false) };
    static FAIL_BEFORE_HEADER: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn fail_next_prepared_i32_fused_apply_after_launch() {
    FAIL_AFTER_LAUNCH.with(|value| value.set(true));
}

#[cfg(test)]
pub(crate) fn fail_next_prepared_i32_fused_apply_after_status() {
    FAIL_AFTER_STATUS.with(|value| value.set(true));
}

#[cfg(test)]
pub(crate) fn fail_next_prepared_i32_fused_apply_before_header() {
    FAIL_BEFORE_HEADER.with(|value| value.set(true));
}

#[cfg(test)]
fn take_fail_after_launch() -> bool {
    FAIL_AFTER_LAUNCH.with(|value| value.replace(false))
}

#[cfg(test)]
fn take_fail_after_status() -> bool {
    FAIL_AFTER_STATUS.with(|value| value.replace(false))
}

#[cfg(test)]
fn take_fail_before_header() -> bool {
    FAIL_BEFORE_HEADER.with(|value| value.replace(false))
}

#[cfg(any(test, feature = "probe-timing"))]
fn record_prepared_fused_drain() {
    PREPARED_FUSED_DRAINS.with(|value| value.set(value.get() + 1));
}
