use gpu_db_execution::{DeviceTarget, PlannedOp};
use gpu_db_sql::Command;

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
            Command::CreateTable(_)
            | Command::CreateSchema(_)
            | Command::DropSchema(_)
            | Command::CreateDatabase(_)
            | Command::DropDatabase(_)
            | Command::RenameDatabase(_)
            | Command::CreateTablespace(_)
            | Command::DropTablespace(_)
            | Command::RenameTablespace(_)
            | Command::AddPrimaryKey(_)
            | Command::AddUniqueConstraint(_)
            | Command::AddCheckConstraint(_)
            | Command::AddForeignKey(_)
            | Command::AddColumn(_)
            | Command::RenameTable(_)
            | Command::RenameColumn(_)
            | Command::RenameConstraint(_)
            | Command::DropColumn(_)
            | Command::DropConstraint(_)
            | Command::CreateIndex(_)
            | Command::RenameIndex(_)
            | Command::CreateView(_)
            | Command::RenameView(_)
            | Command::CreateMaterializedView(_)
            | Command::RefreshMaterializedView(_)
            | Command::RenameMaterializedView(_)
            | Command::CreateFunction(_)
            | Command::RenameFunction(_)
            | Command::DropFunction(_)
            | Command::CreateSequence(_)
            | Command::CreateDomain(_)
            | Command::SequenceNextVal(_)
            | Command::SequenceSetVal(_)
            | Command::RenameSequence(_)
            | Command::CreatePublication(_)
            | Command::DropPublication(_)
            | Command::CreateSubscription(_)
            | Command::DropSubscription(_)
            | Command::CreateRole(_)
            | Command::DropRole(_)
            | Command::RenameRole(_)
            | Command::DropTable(_)
            | Command::TruncateTable(_)
            | Command::DropIndex(_)
            | Command::DropMaterializedView(_)
            | Command::DropView(_)
            | Command::DropSequence(_)
            | Command::DropDomain(_)
            | Command::GrantTable(_)
            | Command::RevokeTable(_)
            | Command::GrantSchema(_)
            | Command::RevokeSchema(_)
            | Command::GrantDatabase(_)
            | Command::RevokeDatabase(_)
            | Command::GrantTablespace(_)
            | Command::RevokeTablespace(_)
            | Command::GrantFunction(_)
            | Command::RevokeFunction(_)
            | Command::GrantDefaultTablePrivileges(_)
            | Command::RevokeDefaultTablePrivileges(_)
            | Command::AlterColumnDefault(_)
            | Command::CommentOn(_)
            | Command::Insert(_)
            | Command::Delete(_)
            | Command::Update(_) => PlanNode {
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
            Command::Select(_) | Command::SequenceCurrVal(_) => PlanNode {
                op: PlannedOp {
                    name: "relational_select".to_string(),
                    target: DeviceTarget::Gpu(self.cfg.default_gpu_id),
                },
                kind: PlanKind::Read,
            },
            Command::SelectFunction(_) => PlanNode {
                op: PlannedOp {
                    name: "routine_select".to_string(),
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
            Command::CreateExtension(_) | Command::DropExtension(_) => PlanNode {
                op: PlannedOp {
                    name: "bootstrap_extension_admin".to_string(),
                    target: DeviceTarget::Cpu,
                },
                kind: PlanKind::Admin,
            },
            Command::ResetAll | Command::SetRole { .. } => PlanNode {
                op: PlannedOp {
                    name: "session_control".to_string(),
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
        let plan = planner.plan_command(&Command::Select(gpu_db_sql::Select {
            table: "people".to_string(),
            distinct: false,
            projection: gpu_db_sql::SelectProjection::All,
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        }));

        let node = &plan.nodes()[0];
        assert_eq!(node.kind, PlanKind::Read);
        assert_eq!(node.op.target, DeviceTarget::Gpu(0));
    }

    #[test]
    fn planner_marks_sql_function_select_as_gpu_targeted() {
        let planner = Planner::new(PlannerConfig { default_gpu_id: 3 });
        let plan = planner.plan_command(&Command::SelectFunction(gpu_db_sql::SelectFunction {
            name: "answer".to_string(),
        }));

        let node = &plan.nodes()[0];
        assert_eq!(node.kind, PlanKind::Read);
        assert_eq!(node.op.name, "routine_select");
        assert_eq!(node.op.target, DeviceTarget::Gpu(3));
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
            Command::Select(gpu_db_sql::Select {
                table: "t".to_string(),
                distinct: false,
                projection: gpu_db_sql::SelectProjection::All,
                group_by: None,
                having_groups: Vec::new(),
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: Vec::new(),
                limit: None,
                offset: None,
            }),
            Command::Delete(gpu_db_sql::Delete {
                table: "t".to_string(),
                filter: None,
                filters: vec![gpu_db_sql::SelectFilter {
                    column: "id".to_string(),
                    op: gpu_db_sql::SelectFilterOp::Eq,
                    value: gpu_db_sql::SqlValue::Int4(1),
                }],
                filter_groups: vec![vec![gpu_db_sql::SelectFilter {
                    column: "id".to_string(),
                    op: gpu_db_sql::SelectFilterOp::Eq,
                    value: gpu_db_sql::SqlValue::Int4(1),
                }]],
                returning: Vec::new(),
            }),
            Command::Update(gpu_db_sql::Update {
                table: "t".to_string(),
                assignments: vec![gpu_db_sql::UpdateAssignment {
                    column: "name".to_string(),
                    source_column: None,
                    value: gpu_db_sql::SqlValue::Text("updated".to_string()),
                }],
                filter: None,
                filters: vec![gpu_db_sql::SelectFilter {
                    column: "id".to_string(),
                    op: gpu_db_sql::SelectFilterOp::Eq,
                    value: gpu_db_sql::SqlValue::Int4(1),
                }],
                filter_groups: vec![vec![gpu_db_sql::SelectFilter {
                    column: "id".to_string(),
                    op: gpu_db_sql::SelectFilterOp::Eq,
                    value: gpu_db_sql::SqlValue::Int4(1),
                }]],
                returning: Vec::new(),
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
