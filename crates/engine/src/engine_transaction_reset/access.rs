//! Stable relation and sequence access retained by live autocommit mutations.

use super::*;

impl Engine {
    /// Retain every stable relation identity a parsed live mutation can observe or change. Row
    /// mutations take their dependency closure; catalog mutations conservatively retain every
    /// published relation because a rename/drop/constraint/default/ACL operation can change the
    /// identity or schema proof of a concurrently staged table reset. DDL is not an OLTP hot path,
    /// and this deliberately simple boundary avoids a command-by-command dependency oracle.
    pub(crate) fn acquire_autocommit_command_table_access(
        &self,
        command: &Command,
    ) -> Result<Option<Arc<TableAccessLease>>, ExecuteError> {
        if command_is_sequence_lifecycle(command) {
            let catalog = self.catalog_snapshot();
            let lease =
                self.acquire_autocommit_table_accesses(catalog.relational_catalog.keys().cloned())?;
            let sequence_names = match command {
                Command::CreateSequence(_) => Vec::new(),
                Command::SequenceRestart(restart) => vec![restart.name.as_str()],
                Command::RenameSequence(rename) => vec![rename.old_name.as_str()],
                Command::DropSequence(drop) => {
                    drop.names.iter().map(String::as_str).collect::<Vec<_>>()
                }
                _ => unreachable!("sequence lifecycle classifier checked above"),
            };
            let sequence_oids = sequence_names
                .into_iter()
                .filter_map(|name| {
                    catalog
                        .relational_sequences
                        .get(name)
                        .map(|sequence| sequence.oid)
                })
                .collect::<BTreeSet<_>>();
            lease.acquire_exclusive(sequence_oids)?;
            return Ok(Some(lease));
        }
        let sequence_value_name = match command {
            Command::SequenceNextVal(nextval) => Some(nextval.name.as_str()),
            Command::SequenceSetVal(setval) => Some(setval.name.as_str()),
            _ => None,
        };
        if let Some(name) = sequence_value_name {
            let catalog = self.catalog_snapshot();
            let oid = match catalog
                .pg_class_relation_kind(name)
                .map_err(ExecuteError::Engine)?
            {
                Some(PgClassRelationKind::Sequence) => catalog
                    .relational_sequences
                    .get(name)
                    .map(|sequence| sequence.oid)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::Durability(format!(
                            "catalog sequence binding {name:?} disappeared"
                        )))
                    })?,
                Some(_) => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{name}\" is not a sequence"
                    ))))
                }
                None => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "sequence \"{name}\" does not exist"
                    ))))
                }
            };
            let lease = self.table_access.lease();
            lease.acquire_shared([oid])?;
            return Ok(Some(lease));
        }
        let tables = match command {
            Command::Insert(insert) => Some(vec![insert.table.clone()]),
            Command::Update(update) => Some(vec![update.table.clone()]),
            Command::Delete(delete) => Some(vec![delete.table.clone()]),
            Command::SessionControl {
                access_share_relations,
                ..
            } if !access_share_relations.is_empty() => Some(access_share_relations.clone()),
            command if Self::command_changes_catalog(command) => Some(
                self.read_state
                    .latest_catalog()
                    .relational_catalog
                    .keys()
                    .cloned()
                    .collect(),
            ),
            // Live TRUNCATE must acquire its exclusive dependency closure through the typed reset
            // owner. Acquiring a second shared owner here would reject that owner's own upgrade.
            Command::TruncateTable(_)
            | Command::Begin { .. }
            | Command::Commit { .. }
            | Command::Rollback { .. }
            | Command::Flush
            | Command::ResetAll
            | Command::SetRole { .. }
            | Command::SetKv { .. }
            | Command::DeleteKv { .. }
            | Command::GetKv { .. }
            | Command::SequenceNextVal(_)
            | Command::SequenceCurrVal(_)
            | Command::SequenceSetVal(_)
            | Command::Select(_)
            | Command::SelectFunction(_)
            | Command::SelectLiteral(_)
            | Command::ShowTransactionIsolation
            | Command::SessionControl { .. }
            | Command::PreparedCatalog(_) => None,
            _ => unreachable!("catalog-changing commands are classified above"),
        };
        let lease = tables
            .map(|tables| self.acquire_autocommit_table_accesses(tables))
            .transpose()?;
        if let (Some(lease), Command::Insert(insert)) = (&lease, command) {
            let catalog = self.catalog_snapshot();
            let sequence_oids = catalog
                .relational_catalog
                .get(&insert.table)
                .into_iter()
                .flat_map(|table| &table.columns)
                .filter_map(|column| match &column.default {
                    Some(ColumnDefault::SequenceNextVal { sequence, .. }) => catalog
                        .relational_sequences
                        .get(sequence)
                        .map(|sequence| sequence.oid),
                    _ => None,
                })
                .collect::<BTreeSet<_>>();
            lease.acquire_shared(sequence_oids)?;
        }
        Ok(lease)
    }
}
