//! Typed identity and dependency closure for transaction-owned CREATE VIEW operations.

use super::*;
use crate::engine_transaction_reset::table_schema_digest;

impl Engine {
    pub(crate) fn transaction_catalog_command_is_supported(command: &Command) -> bool {
        matches!(command, Command::CreateTable(_) | Command::CreateView(_))
    }

    pub(crate) fn apply_transaction_catalog_command(
        &self,
        working: &mut DdlCatalogState,
        command: Command,
    ) -> Result<(), EngineError> {
        match command {
            Command::CreateTable(create) => self.apply_create_table(working, create),
            Command::CreateView(create) => self.apply_create_view(working, create),
            _ => Err(EngineError::ApplyFailed(
                "transaction catalog command is outside the admitted family".to_string(),
            )),
        }
    }

    pub(crate) fn transaction_view_operation_identity(
        command_index: u32,
        ordinal: u32,
        before: &CatalogSnapshot,
        after: &CatalogSnapshot,
        create: &CreateView,
    ) -> Result<BinaryTransactionViewOperationIdentity, ExecuteError> {
        let target_before = before
            .relational_views
            .get(&create.name)
            .map(view_relation_identity)
            .transpose()?;
        if target_before.is_none() && relation_namespace_contains(before, &create.name) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "transactional CREATE VIEW target {:?} had an unsupported relation preimage",
                create.name
            ))));
        }
        let dependencies = view_dependency_closure(before, &create.query.table)?;
        let target_after = after
            .relational_views
            .get(&create.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "transactional CREATE VIEW target {:?} is absent from its private postimage",
                    create.name
                )))
            })
            .and_then(view_relation_identity)?;
        Ok(BinaryTransactionViewOperationIdentity {
            command_index,
            ordinal,
            target_before,
            dependencies,
            target_after,
        })
    }

    pub(crate) fn validate_transaction_view_before(
        catalog: &CatalogSnapshot,
        create: &CreateView,
        identity: &BinaryTransactionViewOperationIdentity,
    ) -> Result<(), EngineError> {
        let observed_before = catalog
            .relational_views
            .get(&create.name)
            .map(view_relation_identity)
            .transpose()
            .map_err(execute_to_engine)?;
        if observed_before != identity.target_before {
            return Err(EngineError::Durability(format!(
                "ordered CREATE VIEW target {:?} changed stable preimage",
                create.name
            )));
        }
        if observed_before.is_none() && relation_namespace_contains(catalog, &create.name) {
            return Err(EngineError::Durability(format!(
                "ordered CREATE VIEW target {:?} changed relation kind",
                create.name
            )));
        }
        let observed_dependencies =
            view_dependency_closure(catalog, &create.query.table).map_err(execute_to_engine)?;
        if observed_dependencies != identity.dependencies {
            return Err(EngineError::Durability(format!(
                "ordered CREATE VIEW {:?} changed its transitive source identity closure",
                create.name
            )));
        }
        Ok(())
    }

    pub(crate) fn validate_transaction_view_after(
        catalog: &CatalogSnapshot,
        create: &CreateView,
        identity: &BinaryTransactionViewOperationIdentity,
    ) -> Result<(), EngineError> {
        let observed = catalog
            .relational_views
            .get(&create.name)
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "ordered CREATE VIEW target {:?} is absent after apply",
                    create.name
                ))
            })
            .and_then(|view| view_relation_identity(view).map_err(execute_to_engine))?;
        if observed != identity.target_after {
            return Err(EngineError::Durability(format!(
                "ordered CREATE VIEW target {:?} changed stable postimage",
                create.name
            )));
        }
        Ok(())
    }

    pub(crate) fn apply_transaction_catalog_envelope(
        &self,
        working: &mut DdlCatalogState,
        commit_seq: Index,
        commands: &[BinaryTransactionCatalogCommand],
        view_operations: &[BinaryTransactionViewOperationIdentity],
    ) -> Result<(), EngineError> {
        let mut next_view = 0usize;
        for (command_index, operation) in commands.iter().enumerate() {
            let before = Self::catalog_snapshot_from_working(working, commit_seq);
            match &operation.command {
                Command::CreateTable(_) => {
                    if view_operations.get(next_view).is_some_and(|identity| {
                        usize::try_from(identity.command_index).ok() == Some(command_index)
                    }) {
                        return Err(EngineError::Durability(
                            "ordered CREATE TABLE carries a CREATE VIEW identity".to_string(),
                        ));
                    }
                }
                Command::CreateView(create) => {
                    let identity = view_operations.get(next_view).ok_or_else(|| {
                        EngineError::Durability(
                            "ordered CREATE VIEW lost its typed identity closure".to_string(),
                        )
                    })?;
                    if usize::try_from(identity.command_index).ok() != Some(command_index)
                        || identity.ordinal != operation.ordinal
                    {
                        return Err(EngineError::Durability(
                            "ordered CREATE VIEW identity changed command position".to_string(),
                        ));
                    }
                    Self::validate_transaction_view_before(&before, create, identity)?;
                }
                _ => {
                    return Err(EngineError::Durability(
                        "transaction WAL record contains an unsupported catalog operation"
                            .to_string(),
                    ))
                }
            }
            self.with_apply_catalog(Some(Arc::clone(&before)), || {
                self.apply_transaction_catalog_command(working, operation.command.clone())
            })?;
            if let Command::CreateView(create) = &operation.command {
                let identity = &view_operations[next_view];
                let after = Self::catalog_snapshot_from_working(working, commit_seq);
                Self::validate_transaction_view_after(&after, create, identity)?;
                next_view += 1;
            }
        }
        if next_view != view_operations.len() {
            return Err(EngineError::Durability(
                "ordered transaction carries an unbound CREATE VIEW identity".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_transaction_catalog_before_wal(
        &self,
        commands: &[BinaryTransactionCatalogCommand],
        view_operations: &[BinaryTransactionViewOperationIdentity],
        expected: &CatalogSnapshot,
    ) -> Result<(), ExecuteError> {
        let mut working = self.ddl_catalog().clone();
        self.apply_transaction_catalog_envelope(
            &mut working,
            expected.commit_seq,
            commands,
            view_operations,
        )
        .map_err(ExecuteError::Engine)?;
        let reconstructed = Self::catalog_snapshot_from_working(&working, expected.commit_seq);
        if !reconstructed.same_contents(expected) {
            return Err(ExecuteError::Serialization(
                "typed transactional catalog envelope no longer reconstructs its exact private post-state"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn view_semantic_digest(
    view: &RelationalView,
) -> Result<gpu_db_wal::CanonicalDigest, ExecuteError> {
    let query = serde_json::to_vec(&view.query).map_err(|error| {
        ExecuteError::Engine(EngineError::Durability(format!(
            "view query cannot be encoded for durable identity: {error}"
        )))
    })?;
    let mut body = Vec::new();
    body.extend_from_slice(b"GPUDBVIEWIDENTITY1");
    push_string(&mut body, &view.schema)?;
    push_string(&mut body, &view.name)?;
    body.extend_from_slice(&view.oid.to_le_bytes());
    push_bytes(&mut body, &query)?;
    push_string(&mut body, &view.definition)?;
    push_len(&mut body, view.acl.len())?;
    for (grantee, privileges) in &view.acl {
        push_string(&mut body, grantee)?;
        push_len(&mut body, privileges.len())?;
        for privilege in privileges {
            body.push(match privilege {
                TablePrivilege::Select => 1,
                TablePrivilege::Insert => 2,
                TablePrivilege::Update => 3,
                TablePrivilege::Delete => 4,
            });
        }
    }
    Ok(gpu_db_wal::canonical_request_digest(&body))
}

fn view_relation_identity(
    view: &RelationalView,
) -> Result<BinaryCatalogRelationIdentity, ExecuteError> {
    Ok(BinaryCatalogRelationIdentity {
        kind: BinaryCatalogRelationKind::View,
        oid: view.oid,
        digest: view_semantic_digest(view)?,
    })
}

fn table_relation_identity(
    table: &RelationalTable,
) -> Result<BinaryCatalogRelationIdentity, ExecuteError> {
    Ok(BinaryCatalogRelationIdentity {
        kind: BinaryCatalogRelationKind::Table,
        oid: table.oid,
        digest: table_schema_digest(table)?,
    })
}

fn view_dependency_closure(
    catalog: &CatalogSnapshot,
    source: &str,
) -> Result<BTreeMap<String, BinaryCatalogRelationIdentity>, ExecuteError> {
    let mut closure = BTreeMap::new();
    let mut visiting = BTreeSet::new();
    collect_view_dependency(catalog, source, &mut visiting, &mut closure)?;
    Ok(closure)
}

fn collect_view_dependency(
    catalog: &CatalogSnapshot,
    name: &str,
    visiting: &mut BTreeSet<String>,
    closure: &mut BTreeMap<String, BinaryCatalogRelationIdentity>,
) -> Result<(), ExecuteError> {
    if let Some(table) = catalog.relational_catalog.get(name) {
        closure.insert(name.to_string(), table_relation_identity(table)?);
        return Ok(());
    }
    let view = catalog.relational_views.get(name).ok_or_else(|| {
        ExecuteError::UndefinedRelation(format!("transactional CREATE VIEW dependency {name:?}"))
    })?;
    if !visiting.insert(name.to_string()) {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "view dependency cycle is unsupported".to_string(),
        )));
    }
    closure.insert(name.to_string(), view_relation_identity(view)?);
    collect_view_dependency(catalog, &view.query.table, visiting, closure)?;
    visiting.remove(name);
    Ok(())
}

fn relation_namespace_contains(catalog: &CatalogSnapshot, name: &str) -> bool {
    catalog.relational_catalog.contains_key(name)
        || catalog.relational_views.contains_key(name)
        || catalog.relational_materialized_views.contains_key(name)
        || catalog.relational_sequences.contains_key(name)
}

fn push_len(body: &mut Vec<u8>, len: usize) -> Result<(), ExecuteError> {
    let len = u64::try_from(len).map_err(|_| {
        ExecuteError::Unsupported("view identity count exceeds durable framing".to_string())
    })?;
    body.extend_from_slice(&len.to_le_bytes());
    Ok(())
}

fn push_bytes(body: &mut Vec<u8>, value: &[u8]) -> Result<(), ExecuteError> {
    push_len(body, value.len())?;
    body.extend_from_slice(value);
    Ok(())
}

fn push_string(body: &mut Vec<u8>, value: &str) -> Result<(), ExecuteError> {
    push_bytes(body, value.as_bytes())
}

fn execute_to_engine(error: ExecuteError) -> EngineError {
    match error {
        ExecuteError::Engine(error) => error,
        other => EngineError::Durability(other.to_string()),
    }
}
