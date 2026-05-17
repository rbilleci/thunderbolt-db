use gpu_db_execution::{DeviceTarget, PlannedOp};
use gpu_db_protocol::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanKind {
    Mutation,
    Read,
    TxnControl,
    Admin,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanNode {
    pub op: PlannedOp,
    pub kind: PlanKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPlan {
    nodes: Vec<PlanNode>,
}

impl ExecutionPlan {
    pub fn new(nodes: Vec<PlanNode>) -> Self {
        Self { nodes }
    }

    pub fn nodes(&self) -> &[PlanNode] {
        &self.nodes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlannerConfig {
    pub default_gpu_id: u16,
}

#[derive(Debug, Clone)]
pub struct Planner {
    cfg: PlannerConfig,
}

impl Planner {
    pub fn new(cfg: PlannerConfig) -> Self {
        Self { cfg }
    }

    pub fn default_gpu_id(&self) -> u16 {
        self.cfg.default_gpu_id
    }

    pub fn plan_command(&self, command: &Command) -> ExecutionPlan {
        let node = match command {
            Command::SetKv { .. } | Command::DeleteKv { .. } => PlanNode {
                op: PlannedOp {
                    name: "kv_write".to_string(),
                    target: DeviceTarget::Gpu(self.cfg.default_gpu_id),
                },
                kind: PlanKind::Mutation,
            },
            Command::CreateTable(_) | Command::Insert(_) => PlanNode {
                op: PlannedOp {
                    name: "relational_write".to_string(),
                    target: DeviceTarget::Gpu(self.cfg.default_gpu_id),
                },
                kind: PlanKind::Mutation,
            },
            Command::GetKv { .. } => PlanNode {
                op: PlannedOp {
                    name: "kv_point_get".to_string(),
                    target: DeviceTarget::Cpu,
                },
                kind: PlanKind::Read,
            },
            Command::Select(_) => PlanNode {
                op: PlannedOp {
                    name: "relational_select".to_string(),
                    target: DeviceTarget::Gpu(self.cfg.default_gpu_id),
                },
                kind: PlanKind::Read,
            },
            Command::Begin | Command::Commit { .. } | Command::Rollback { .. } => PlanNode {
                op: PlannedOp {
                    name: "txn_control".to_string(),
                    target: DeviceTarget::Cpu,
                },
                kind: PlanKind::TxnControl,
            },
            Command::Flush => PlanNode {
                op: PlannedOp {
                    name: "admin_flush".to_string(),
                    target: DeviceTarget::Cpu,
                },
                kind: PlanKind::Admin,
            },
            Command::ResetAll => PlanNode {
                op: PlannedOp {
                    name: "session_reset_all".to_string(),
                    target: DeviceTarget::Cpu,
                },
                kind: PlanKind::TxnControl,
            },
        };

        ExecutionPlan::new(vec![node])
    }
}

impl Default for Planner {
    fn default() -> Self {
        Self::new(PlannerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_marks_writes_as_gpu_targeted() {
        let planner = Planner::default();
        let plan = planner.plan_command(&Command::SetKv {
            key: "k".to_string(),
            value: "v".to_string(),
        });

        let node = &plan.nodes()[0];
        assert_eq!(node.kind, PlanKind::Mutation);
        assert_eq!(node.op.target, DeviceTarget::Gpu(0));
    }

    #[test]
    fn planner_marks_get_as_cpu_fallback_path() {
        let planner = Planner::default();
        let plan = planner.plan_command(&Command::GetKv {
            key: "k".to_string(),
        });

        let node = &plan.nodes()[0];
        assert_eq!(node.kind, PlanKind::Read);
        assert_eq!(node.op.target, DeviceTarget::Cpu);
    }

    #[test]
    fn planner_marks_relational_select_as_gpu_targeted() {
        let planner = Planner::default();
        let plan = planner.plan_command(&Command::Select(gpu_db_protocol::Select {
            table: "people".to_string(),
            projection: gpu_db_protocol::SelectProjection::All,
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
        }));

        let node = &plan.nodes()[0];
        assert_eq!(node.kind, PlanKind::Read);
        assert_eq!(node.op.target, DeviceTarget::Gpu(0));
    }

    #[test]
    fn planner_never_emits_device_agnostic_nodes() {
        let planner = Planner::new(PlannerConfig { default_gpu_id: 4 });
        let commands = [
            Command::SetKv {
                key: "a".to_string(),
                value: "1".to_string(),
            },
            Command::DeleteKv {
                key: "a".to_string(),
            },
            Command::GetKv {
                key: "a".to_string(),
            },
            Command::Select(gpu_db_protocol::Select {
                table: "t".to_string(),
                projection: gpu_db_protocol::SelectProjection::All,
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: None,
                limit: None,
                offset: None,
            }),
            Command::Begin,
            Command::Commit { chain: false },
            Command::Rollback { chain: false },
            Command::Flush,
            Command::ResetAll,
        ];

        for command in commands {
            let plan = planner.plan_command(&command);
            assert_eq!(plan.nodes().len(), 1);

            match plan.nodes()[0].op.target {
                DeviceTarget::Cpu | DeviceTarget::Gpu(_) => {}
            }
        }
    }
}
