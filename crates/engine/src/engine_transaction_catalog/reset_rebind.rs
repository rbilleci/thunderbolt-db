//! Rebind final table-reset output proofs after later ordered catalog operations.
//!
//! The reset statement's sequence identity remains fixed at its original ordinal. Metadata-only
//! catalog operations after that ordinal may, however, change the table descriptor installed with
//! the final empty root. Rebind that output under the reset's retained stable-OID guards rather
//! than comparing its statement-time descriptor with the transaction's final catalog.

use super::*;
use crate::engine_transaction_reset::{
    table_access_dependency_identities, table_reset_empty_digest, table_schema_digest,
};

impl Engine {
    pub(super) fn rebind_transaction_table_reset_outputs(
        &self,
        operations: &[TransactionOperation],
        before: &CatalogSnapshot,
        after: &CatalogSnapshot,
        command: &Command,
    ) -> Result<BTreeMap<u32, StagedTableReset>, ExecuteError> {
        let mut rebound = BTreeMap::new();
        for operation in operations {
            let TransactionOperation::TableReset(reset) = operation else {
                continue;
            };
            let before_table = before.relational_catalog.get(&reset.table).ok_or_else(|| {
                ExecuteError::Serialization(format!(
                    "table reset target \"{}\" left the catalog before output rebind",
                    reset.table
                ))
            })?;
            let after_table = after.relational_catalog.get(&reset.table).ok_or_else(|| {
                ExecuteError::Serialization(format!(
                    "table reset target \"{}\" left the catalog after output rebind",
                    reset.table
                ))
            })?;
            if before_table.oid != reset.table_oid || after_table.oid != reset.table_oid {
                return Err(ExecuteError::Serialization(format!(
                    "table reset target \"{}\" changed stable identity during output rebind",
                    reset.table
                )));
            }

            let before_identities =
                table_access_dependency_identities(&before.relational_catalog, before_table)?;
            if before_identities != reset.dependency_identities {
                return Err(ExecuteError::Serialization(format!(
                    "table reset dependency closure for \"{}\" changed before output rebind",
                    reset.table
                )));
            }
            let before_dependencies =
                reset_catalog_dependencies(before, &before_identities, &reset.table)?;
            if before_dependencies != reset.catalog_dependencies {
                return Err(ExecuteError::Serialization(format!(
                    "table reset catalog proof for \"{}\" changed before output rebind",
                    reset.table
                )));
            }

            let after_identities =
                table_access_dependency_identities(&after.relational_catalog, after_table)?;
            if after_identities != reset.dependency_identities {
                return Err(ExecuteError::Serialization(format!(
                    "catalog operation changes the guarded dependency closure of table reset \"{}\"",
                    reset.table
                )));
            }
            let after_dependencies =
                reset_catalog_dependencies(after, &after_identities, &reset.table)?;
            let table_changed = before_table != after_table;
            if !table_changed && before_dependencies == after_dependencies {
                continue;
            }
            if !command_is_index_lifecycle(command) && !command_is_sequence_lifecycle(command) {
                return Err(ExecuteError::Serialization(format!(
                    "unsupported catalog family changes the output of table reset \"{}\"",
                    reset.table
                )));
            }

            let mut next = reset.as_ref().clone();
            next.catalog_dependencies = after_dependencies;
            next.foreign_key_dependencies = after_identities
                .keys()
                .filter(|name| *name != &reset.table)
                .cloned()
                .collect();
            if table_changed {
                let before_schema = table_schema_digest(before_table)?;
                if before_schema != reset.schema_digest
                    || table_reset_empty_digest(reset.table_oid, before_schema)
                        != reset.after_empty_digest
                {
                    return Err(ExecuteError::Serialization(format!(
                        "table reset output proof for \"{}\" changed before rebind",
                        reset.table
                    )));
                }
                let (before_rows, before_digest) = self.table_reset_device_root_proof(
                    before_table,
                    reset.source_commit_seq,
                    self.committed_seq(),
                )?;
                if before_rows != reset.expected_rows || before_digest != reset.before_digest {
                    return Err(ExecuteError::Serialization(format!(
                        "table reset source proof for \"{}\" changed before output rebind",
                        reset.table
                    )));
                }

                let after_schema = table_schema_digest(after_table)?;
                let (after_rows, after_digest) = self.table_reset_device_root_proof(
                    after_table,
                    reset.source_commit_seq,
                    self.committed_seq(),
                )?;
                if after_rows != reset.expected_rows {
                    return Err(ExecuteError::Serialization(format!(
                        "table reset source cardinality for \"{}\" changed during output rebind",
                        reset.table
                    )));
                }
                next.schema_digest = after_schema;
                next.before_digest = after_digest;
                next.after_empty_digest = table_reset_empty_digest(reset.table_oid, after_schema);
            }
            rebound.insert(reset.ordinal, next);
        }
        Ok(rebound)
    }
}

fn reset_catalog_dependencies(
    catalog: &CatalogSnapshot,
    identities: &BTreeMap<String, u32>,
    reset_table: &str,
) -> Result<BTreeMap<String, RelationalTable>, ExecuteError> {
    identities
        .keys()
        .map(|name| {
            catalog
                .relational_catalog
                .get(name)
                .cloned()
                .map(|table| (name.clone(), table))
                .ok_or_else(|| {
                    ExecuteError::Serialization(format!(
                        "table reset \"{reset_table}\" dependency \"{name}\" left its catalog"
                    ))
                })
        })
        .collect()
}
