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

pub trait Operator {
    fn open(&mut self) {}
    fn next(&mut self) -> Option<Vec<u8>>;
    fn close(&mut self) {}
}

#[derive(Debug, Default)]
pub struct CpuNoop;

impl Operator for CpuNoop {
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

impl Operator for VecOperator {
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
}
