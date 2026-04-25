use std::collections::BTreeSet;

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
    fn scan_filter_project_pipeline_composes() {
        let scan = ScanOperator::new(vec![1_i32, 2_i32, 3_i32, 4_i32]);
        let filter = FilterOperator::new(scan, |row: &i32| row % 2 == 1);
        let mut op = ProjectOperator::new(filter, |row| row * 10);

        op.open();
        assert_eq!(op.next(), Some(10));
        assert_eq!(op.next(), Some(30));
        assert_eq!(op.next(), None);
        op.close();
    }
}
