use std::collections::BTreeSet;
use std::fmt;
use std::os::raw::c_void;

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
        launch_cuda_device_memory_copy(gpu_id, payload)?;
        Ok(CudaDeviceMemoryProof {
            gpu_id,
            device_name: device.name,
            allocated_bytes: payload.len() as u64,
            copied_bytes: payload.len() as u64,
        })
    }
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
    .reg .u32 %value;
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %value, [input];
    add.u32 %value, %value, 1;
    st.global.u32 [%out], %value;
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

fn launch_cuda_device_memory_copy(
    gpu_id: u16,
    payload: &[u8],
) -> Result<(), CudaRuntimeProbeError> {
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

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, i32::from(gpu_id)) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let mut device_ptr = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_ptr, payload.len()) })?;
    let allocation_guard = CudaDeviceAllocationGuard {
        ptr: device_ptr,
        free: *cu_mem_free,
    };

    check_cuda(unsafe {
        cu_memcpy_htod(
            allocation_guard.ptr,
            payload.as_ptr().cast::<c_void>(),
            payload.len(),
        )
    })?;

    drop(allocation_guard);
    drop(context_guard);

    Ok(())
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
    .reg .u64 %rd_addr;

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
    add.u64 %rd_addr, %rd_key_offsets, %rd_offset;
    add.u64 %rd_next_offset, %rd_addr, 4;
    ld.global.u32 %r_key_start, [%rd_addr];
    ld.global.u32 %r_key_end, [%rd_next_offset];
    sub.u32 %r_key_len, %r_key_end, %r_key_start;

    add.u64 %rd_addr, %rd_value_offsets, %rd_offset;
    add.u64 %rd_next_offset, %rd_addr, 4;
    ld.global.u32 %r_value_start, [%rd_addr];
    ld.global.u32 %r_value_end, [%rd_next_offset];
    sub.u32 %r_value_len, %r_value_end, %r_value_start;

    mul.wide.u32 %rd_output_offset, %r_idx, 8;
    add.u64 %rd_addr, %rd_output, %rd_output_offset;
    st.global.u32 [%rd_addr], %r_key_len;
    add.u64 %rd_addr, %rd_addr, 4;
    st.global.u32 [%rd_addr], %r_value_len;

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
    .reg .u64 %rd_addr;
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
    add.u64 %rd_addr, %rd_begin, %rd_offset8;
    ld.global.u64 %rd_created_by, [%rd_addr];
    add.u64 %rd_addr, %rd_end, %rd_offset8;
    ld.global.u64 %rd_deleted_by, [%rd_addr];

    setp.le.u64 %p_created, %rd_created_by, %rd_read_txn_id;
    setp.gt.u64 %p_not_deleted, %rd_deleted_by, %rd_read_txn_id;
    and.pred %p_visible, %p_created, %p_not_deleted;
    selp.u32 %r_mask_value, 1, 0, %p_visible;

    mul.wide.u32 %rd_offset4, %r_idx, 4;
    add.u64 %rd_addr, %rd_mask, %rd_offset4;
    st.global.u32 [%rd_addr], %r_mask_value;

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
