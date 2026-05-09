use std::collections::BTreeSet;
use std::fmt;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaRuntimeProbeError {
    DriverLibraryUnavailable,
    DriverInitFailed(i32),
    DeviceCountFailed(i32),
    InvalidDeviceCount(i32),
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
