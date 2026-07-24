//! Stable identity and column-default dependency closure for transactional sequence lifecycle and
//! owned-sequence resets.

use super::*;
use crate::engine_transaction_reset::table_schema_digest;

impl Engine {
    pub(crate) fn transaction_sequence_reset_overlay(
        ordinal: u32,
        table: &str,
        before: &Arc<CatalogSnapshot>,
    ) -> Result<
        (
            Arc<CatalogSnapshot>,
            BinaryTransactionSequenceResetOperationIdentity,
        ),
        ExecuteError,
    > {
        let names = owned_sequence_names(before, table)?;
        let mut after = before.as_ref().clone();
        for name in names {
            let sequence = after.relational_sequences.get_mut(&name).ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(format!(
                    "TRUNCATE-owned sequence binding {name:?} disappeared"
                )))
            })?;
            sequence.last_value = 1;
            sequence.is_called = false;
        }
        let after = Arc::new(after);
        let identity = Self::transaction_sequence_reset_identity(ordinal, table, before, &after)?;
        Ok((after, identity))
    }

    pub(crate) fn transaction_sequence_reset_identity(
        ordinal: u32,
        table: &str,
        before: &CatalogSnapshot,
        after: &CatalogSnapshot,
    ) -> Result<BinaryTransactionSequenceResetOperationIdentity, ExecuteError> {
        let names = owned_sequence_names(before, table)?;
        if owned_sequence_names(after, table)? != names {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "TRUNCATE sequence ownership changed while staging its private reset".to_string(),
            )));
        }
        let mut targets = Vec::with_capacity(names.len());
        for name in names {
            let sequence_before = required_sequence(before, &name)?;
            let sequence_after = required_sequence(after, &name)?;
            targets.push(BinaryTransactionSequenceLifecycleTargetIdentity {
                before_name: name.clone(),
                target_before: Some(sequence_identity(before, sequence_before)?),
                dependencies_before: sequence_dependency_closure(before, &name)?,
                after_name: Some(name.clone()),
                target_after: Some(sequence_identity(after, sequence_after)?),
                dependencies_after: sequence_dependency_closure(after, &name)?,
            });
        }
        let identity = BinaryTransactionSequenceResetOperationIdentity {
            ordinal,
            table: table.to_string(),
            targets,
        };
        if !valid_sequence_reset_operation_identity(&identity) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "transactional TRUNCATE sequence reset identity is not canonical".to_string(),
            )));
        }
        Ok(identity)
    }

    pub(crate) fn apply_transaction_sequence_reset_identity(
        working: &mut DdlCatalogState,
        commit_seq: Index,
        identity: &BinaryTransactionSequenceResetOperationIdentity,
    ) -> Result<(), EngineError> {
        if !valid_sequence_reset_operation_identity(identity) {
            return Err(EngineError::Durability(
                "ordered TRUNCATE carries a noncanonical sequence reset identity".to_string(),
            ));
        }
        let before = Self::catalog_snapshot_from_working(working, commit_seq);
        validate_sequence_reset_before(&before, identity)?;
        for target in &identity.targets {
            let expected_oid = target
                .target_before
                .as_ref()
                .map(|target| target.oid)
                .ok_or_else(|| {
                    EngineError::Durability(
                        "ordered TRUNCATE sequence reset lost its stable target".to_string(),
                    )
                })?;
            let sequence = working
                .relational_sequences
                .get_mut(&target.before_name)
                .filter(|sequence| sequence.oid == expected_oid)
                .ok_or_else(|| {
                    EngineError::Durability(format!(
                        "ordered TRUNCATE sequence target {:?} changed binding",
                        target.before_name
                    ))
                })?;
            sequence.last_value = 1;
            sequence.is_called = false;
        }
        let after = Self::catalog_snapshot_from_working(working, commit_seq);
        validate_sequence_reset_after(&after, identity)
    }

    pub(crate) fn transaction_sequence_operation_identity(
        command_index: u32,
        ordinal: u32,
        before: &CatalogSnapshot,
        after: &CatalogSnapshot,
        command: &Command,
    ) -> Result<BinaryTransactionSequenceLifecycleOperationIdentity, ExecuteError> {
        let targets = match command {
            Command::CreateSequence(create) => {
                ensure_sequence_absent(before, &create.name, "CREATE SEQUENCE target")?;
                let sequence = required_sequence(after, &create.name)?;
                let dependencies_after = sequence_dependency_closure(after, &create.name)?;
                vec![BinaryTransactionSequenceLifecycleTargetIdentity {
                    before_name: create.name.clone(),
                    target_before: None,
                    dependencies_before: BTreeMap::new(),
                    after_name: Some(create.name.clone()),
                    target_after: Some(sequence_identity(after, sequence)?),
                    dependencies_after,
                }]
            }
            Command::RenameSequence(rename) => {
                let sequence = required_sequence(before, &rename.old_name)?;
                ensure_sequence_absent(before, &rename.new_name, "ALTER SEQUENCE destination")?;
                let dependencies_before = sequence_dependency_closure(before, &rename.old_name)?;
                let renamed = required_sequence(after, &rename.new_name)?;
                let dependencies_after = sequence_dependency_closure(after, &rename.new_name)?;
                vec![BinaryTransactionSequenceLifecycleTargetIdentity {
                    before_name: rename.old_name.clone(),
                    target_before: Some(sequence_identity(before, sequence)?),
                    dependencies_before,
                    after_name: Some(rename.new_name.clone()),
                    target_after: Some(sequence_identity(after, renamed)?),
                    dependencies_after,
                }]
            }
            Command::SequenceRestart(restart) => {
                let sequence = required_sequence(before, &restart.name)?;
                let restarted = required_sequence(after, &restart.name)?;
                vec![BinaryTransactionSequenceLifecycleTargetIdentity {
                    before_name: restart.name.clone(),
                    target_before: Some(sequence_identity(before, sequence)?),
                    dependencies_before: sequence_dependency_closure(before, &restart.name)?,
                    after_name: Some(restart.name.clone()),
                    target_after: Some(sequence_identity(after, restarted)?),
                    dependencies_after: sequence_dependency_closure(after, &restart.name)?,
                }]
            }
            Command::DropSequence(drop) => {
                let mut targets = Vec::with_capacity(drop.names.len());
                for name in &drop.names {
                    let (target_before, dependencies_before) =
                        if let Some(sequence) = optional_sequence(before, name)? {
                            (
                                Some(sequence_identity(before, sequence)?),
                                sequence_dependency_closure(before, name)?,
                            )
                        } else {
                            ensure_sequence_absent(before, name, "DROP SEQUENCE target")?;
                            (None, BTreeMap::new())
                        };
                    ensure_sequence_absent(after, name, "DROP SEQUENCE postimage")?;
                    targets.push(BinaryTransactionSequenceLifecycleTargetIdentity {
                        before_name: name.clone(),
                        target_before,
                        dependencies_before,
                        after_name: None,
                        target_after: None,
                        dependencies_after: BTreeMap::new(),
                    });
                }
                targets
            }
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "transactional sequence identity names an unsupported command".to_string(),
                )))
            }
        };
        let identity = BinaryTransactionSequenceLifecycleOperationIdentity {
            command_index,
            ordinal,
            targets,
        };
        if !valid_sequence_lifecycle_operation_identity(command, &identity) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "transactional sequence identity is not canonical".to_string(),
            )));
        }
        Ok(identity)
    }

    pub(crate) fn validate_transaction_sequence_before(
        catalog: &CatalogSnapshot,
        command: &Command,
        identity: &BinaryTransactionSequenceLifecycleOperationIdentity,
    ) -> Result<(), EngineError> {
        if !valid_sequence_lifecycle_operation_identity(command, identity) {
            return Err(EngineError::Durability(
                "ordered sequence operation carries a noncanonical identity".to_string(),
            ));
        }
        match command {
            Command::CreateSequence(create) => {
                ensure_sequence_absent(catalog, &create.name, "CREATE SEQUENCE target")
                    .map_err(execute_to_engine)?;
                validate_preimage(&identity.targets[0], None, BTreeMap::new())
            }
            Command::RenameSequence(rename) => {
                let sequence =
                    required_sequence(catalog, &rename.old_name).map_err(execute_to_engine)?;
                ensure_sequence_absent(catalog, &rename.new_name, "ALTER SEQUENCE destination")
                    .map_err(execute_to_engine)?;
                validate_preimage(
                    &identity.targets[0],
                    Some(sequence_identity(catalog, sequence).map_err(execute_to_engine)?),
                    sequence_dependency_closure(catalog, &rename.old_name)
                        .map_err(execute_to_engine)?,
                )
            }
            Command::SequenceRestart(restart) => {
                let sequence =
                    required_sequence(catalog, &restart.name).map_err(execute_to_engine)?;
                validate_preimage(
                    &identity.targets[0],
                    Some(sequence_identity(catalog, sequence).map_err(execute_to_engine)?),
                    sequence_dependency_closure(catalog, &restart.name)
                        .map_err(execute_to_engine)?,
                )
            }
            Command::DropSequence(drop) => {
                for (name, target) in drop.names.iter().zip(&identity.targets) {
                    if let Some(sequence) =
                        optional_sequence(catalog, name).map_err(execute_to_engine)?
                    {
                        validate_preimage(
                            target,
                            Some(sequence_identity(catalog, sequence).map_err(execute_to_engine)?),
                            sequence_dependency_closure(catalog, name)
                                .map_err(execute_to_engine)?,
                        )?;
                    } else {
                        ensure_sequence_absent(catalog, name, "DROP SEQUENCE target")
                            .map_err(execute_to_engine)?;
                        validate_preimage(target, None, BTreeMap::new())?;
                    }
                }
                Ok(())
            }
            _ => Err(EngineError::Durability(
                "ordered sequence identity names an unsupported command".to_string(),
            )),
        }
    }

    pub(crate) fn validate_transaction_sequence_after(
        catalog: &CatalogSnapshot,
        command: &Command,
        identity: &BinaryTransactionSequenceLifecycleOperationIdentity,
    ) -> Result<(), EngineError> {
        match command {
            Command::CreateSequence(create) => {
                let sequence =
                    required_sequence(catalog, &create.name).map_err(execute_to_engine)?;
                validate_postimage(
                    &identity.targets[0],
                    &create.name,
                    sequence_identity(catalog, sequence).map_err(execute_to_engine)?,
                    sequence_dependency_closure(catalog, &create.name)
                        .map_err(execute_to_engine)?,
                )
            }
            Command::RenameSequence(rename) => {
                ensure_sequence_absent(catalog, &rename.old_name, "ALTER SEQUENCE old binding")
                    .map_err(execute_to_engine)?;
                let sequence =
                    required_sequence(catalog, &rename.new_name).map_err(execute_to_engine)?;
                validate_postimage(
                    &identity.targets[0],
                    &rename.new_name,
                    sequence_identity(catalog, sequence).map_err(execute_to_engine)?,
                    sequence_dependency_closure(catalog, &rename.new_name)
                        .map_err(execute_to_engine)?,
                )
            }
            Command::SequenceRestart(restart) => {
                let sequence =
                    required_sequence(catalog, &restart.name).map_err(execute_to_engine)?;
                validate_postimage(
                    &identity.targets[0],
                    &restart.name,
                    sequence_identity(catalog, sequence).map_err(execute_to_engine)?,
                    sequence_dependency_closure(catalog, &restart.name)
                        .map_err(execute_to_engine)?,
                )
            }
            Command::DropSequence(drop) => {
                for (name, target) in drop.names.iter().zip(&identity.targets) {
                    ensure_sequence_absent(catalog, name, "DROP SEQUENCE postimage")
                        .map_err(execute_to_engine)?;
                    if target.after_name.is_some()
                        || target.target_after.is_some()
                        || !target.dependencies_after.is_empty()
                    {
                        return Err(EngineError::Durability(format!(
                            "ordered DROP SEQUENCE target {name:?} carries a non-absent postimage"
                        )));
                    }
                }
                Ok(())
            }
            _ => Err(EngineError::Durability(
                "ordered sequence identity names an unsupported command".to_string(),
            )),
        }
    }
}

fn validate_sequence_reset_before(
    catalog: &CatalogSnapshot,
    identity: &BinaryTransactionSequenceResetOperationIdentity,
) -> Result<(), EngineError> {
    let names = owned_sequence_names(catalog, &identity.table).map_err(execute_to_engine)?;
    if names
        != identity
            .targets
            .iter()
            .map(|target| target.before_name.clone())
            .collect::<BTreeSet<_>>()
    {
        return Err(EngineError::Durability(format!(
            "ordered TRUNCATE target {:?} changed owned-sequence closure",
            identity.table
        )));
    }
    for target in &identity.targets {
        let sequence =
            required_sequence(catalog, &target.before_name).map_err(execute_to_engine)?;
        validate_preimage(
            target,
            Some(sequence_identity(catalog, sequence).map_err(execute_to_engine)?),
            sequence_dependency_closure(catalog, &target.before_name).map_err(execute_to_engine)?,
        )?;
    }
    Ok(())
}

fn validate_sequence_reset_after(
    catalog: &CatalogSnapshot,
    identity: &BinaryTransactionSequenceResetOperationIdentity,
) -> Result<(), EngineError> {
    if owned_sequence_names(catalog, &identity.table).map_err(execute_to_engine)?
        != identity
            .targets
            .iter()
            .map(|target| target.before_name.clone())
            .collect::<BTreeSet<_>>()
    {
        return Err(EngineError::Durability(format!(
            "ordered TRUNCATE target {:?} changed owned-sequence postimage closure",
            identity.table
        )));
    }
    for target in &identity.targets {
        let sequence =
            required_sequence(catalog, &target.before_name).map_err(execute_to_engine)?;
        validate_postimage(
            target,
            &target.before_name,
            sequence_identity(catalog, sequence).map_err(execute_to_engine)?,
            sequence_dependency_closure(catalog, &target.before_name).map_err(execute_to_engine)?,
        )?;
    }
    Ok(())
}

fn owned_sequence_names(
    catalog: &CatalogSnapshot,
    table: &str,
) -> Result<BTreeSet<String>, ExecuteError> {
    let relation = catalog.relational_catalog.get(table).ok_or_else(|| {
        ExecuteError::UndefinedRelation(format!("TRUNCATE sequence owner {table:?}"))
    })?;
    let names = relation
        .columns
        .iter()
        .filter_map(|column| match &column.default {
            Some(ColumnDefault::SequenceNextVal {
                sequence,
                create_if_missing: true,
            }) => Some(sequence.clone()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    for name in &names {
        required_sequence(catalog, name)?;
    }
    Ok(names)
}

fn validate_preimage(
    target: &BinaryTransactionSequenceLifecycleTargetIdentity,
    sequence: Option<BinaryCatalogRelationIdentity>,
    dependencies: BTreeMap<String, BinarySequenceColumnDependencyIdentity>,
) -> Result<(), EngineError> {
    if target.target_before != sequence || target.dependencies_before != dependencies {
        return Err(EngineError::Durability(format!(
            "ordered sequence target {:?} changed stable preimage or column-default dependency closure",
            target.before_name
        )));
    }
    Ok(())
}

fn validate_postimage(
    target: &BinaryTransactionSequenceLifecycleTargetIdentity,
    name: &str,
    sequence: BinaryCatalogRelationIdentity,
    dependencies: BTreeMap<String, BinarySequenceColumnDependencyIdentity>,
) -> Result<(), EngineError> {
    if target.after_name.as_deref() != Some(name)
        || target.target_after.as_ref() != Some(&sequence)
        || target.dependencies_after != dependencies
    {
        return Err(EngineError::Durability(format!(
            "ordered sequence target {name:?} changed stable postimage or column-default dependency closure"
        )));
    }
    Ok(())
}

fn optional_sequence<'a>(
    catalog: &'a CatalogSnapshot,
    name: &str,
) -> Result<Option<&'a RelationalSequence>, ExecuteError> {
    match catalog
        .pg_class_relation_kind(name)
        .map_err(ExecuteError::Engine)?
    {
        Some(PgClassRelationKind::Sequence) => catalog
            .relational_sequences
            .get(name)
            .map(Some)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(format!(
                    "catalog sequence binding {name:?} disappeared"
                )))
            }),
        Some(_) => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "relation {name:?} is not a sequence"
        )))),
        None => Ok(None),
    }
}

fn required_sequence<'a>(
    catalog: &'a CatalogSnapshot,
    name: &str,
) -> Result<&'a RelationalSequence, ExecuteError> {
    optional_sequence(catalog, name)?.ok_or_else(|| {
        ExecuteError::UndefinedRelation(format!("transactional sequence target {name:?}"))
    })
}

fn ensure_sequence_absent(
    catalog: &CatalogSnapshot,
    name: &str,
    context: &str,
) -> Result<(), ExecuteError> {
    if catalog
        .pg_class_relation_kind(name)
        .map_err(ExecuteError::Engine)?
        .is_some()
    {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "transactional {context} {name:?} is not absent"
        ))));
    }
    Ok(())
}

fn sequence_dependency_closure(
    catalog: &CatalogSnapshot,
    sequence: &str,
) -> Result<BTreeMap<String, BinarySequenceColumnDependencyIdentity>, ExecuteError> {
    let mut dependencies = BTreeMap::new();
    for table in catalog.relational_catalog.values() {
        for column in &table.columns {
            if !matches!(
                &column.default,
                Some(ColumnDefault::SequenceNextVal { sequence: name, .. }) if name == sequence
            ) {
                continue;
            }
            if column.id == 0 || column.table_oid != table.oid {
                return Err(ExecuteError::Engine(EngineError::Durability(format!(
                    "sequence dependency {}.{} has an invalid stable column identity",
                    table.name, column.name
                ))));
            }
            let key = format!("{}:{}", table.oid, column.id);
            let dependency = BinarySequenceColumnDependencyIdentity {
                table: BinaryCatalogRelationIdentity {
                    kind: BinaryCatalogRelationKind::Table,
                    oid: table.oid,
                    digest: table_schema_digest(table)?,
                },
                column_id: column.id,
            };
            if dependencies.insert(key, dependency).is_some() {
                return Err(ExecuteError::Engine(EngineError::Durability(
                    "sequence dependency closure contains a duplicate stable column identity"
                        .to_string(),
                )));
            }
        }
    }
    Ok(dependencies)
}

pub(crate) fn sequence_semantic_digest(
    catalog: &CatalogSnapshot,
    sequence: &RelationalSequence,
) -> Result<gpu_db_wal::CanonicalDigest, ExecuteError> {
    let mut body = Vec::new();
    body.extend_from_slice(b"GPUDBSEQUENCEIDENTITY1");
    push_string(&mut body, &sequence.schema)?;
    push_string(&mut body, &sequence.name)?;
    body.extend_from_slice(&sequence.oid.to_le_bytes());
    body.extend_from_slice(&sequence.last_value.to_le_bytes());
    body.push(u8::from(sequence.is_called));
    push_len(&mut body, sequence.acl.len())?;
    for (grantee, privileges) in &sequence.acl {
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
    match catalog
        .relational_comments
        .get(&RelationalCommentTarget::Sequence {
            sequence: sequence.name.clone(),
        }) {
        Some(comment) => {
            body.push(1);
            push_string(&mut body, comment)?;
        }
        None => body.push(0),
    }
    Ok(gpu_db_wal::canonical_request_digest(&body))
}

fn sequence_identity(
    catalog: &CatalogSnapshot,
    sequence: &RelationalSequence,
) -> Result<BinaryCatalogRelationIdentity, ExecuteError> {
    Ok(BinaryCatalogRelationIdentity {
        kind: BinaryCatalogRelationKind::Sequence,
        oid: sequence.oid,
        digest: sequence_semantic_digest(catalog, sequence)?,
    })
}

fn push_len(body: &mut Vec<u8>, len: usize) -> Result<(), ExecuteError> {
    body.extend_from_slice(
        &u64::try_from(len)
            .map_err(|_| {
                ExecuteError::Unsupported(
                    "sequence identity count exceeds durable framing".to_string(),
                )
            })?
            .to_le_bytes(),
    );
    Ok(())
}

fn push_string(body: &mut Vec<u8>, value: &str) -> Result<(), ExecuteError> {
    push_len(body, value.len())?;
    body.extend_from_slice(value.as_bytes());
    Ok(())
}

fn execute_to_engine(error: ExecuteError) -> EngineError {
    match error {
        ExecuteError::Engine(error) => error,
        other => EngineError::Durability(other.to_string()),
    }
}
