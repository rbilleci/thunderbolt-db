use std::collections::BTreeSet;

use gpu_db_sql::{ColumnDefault, Command, CommentTarget, SqlValue};
use gpu_db_storage::Visibility as StorageVisibility;
use gpu_db_types::{EngineError, TxnId};

use crate::{
    add_column_default_supported, coerce_column_default, sequence_defaults, DmlReadSnapshot,
    Engine, PUBLIC_SCHEMA_NAME,
};

impl Engine {
    pub(crate) fn preflight_unique_index_constraints(
        &self,
        cmd: &Command,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        self.ensure_dml_device_generation(cmd)?;
        let cat = self.catalog_snapshot();
        match cmd {
            Command::CreateSchema(create) => {
                if create.name != PUBLIC_SCHEMA_NAME {
                    return Err(EngineError::ApplyFailed(format!(
                        "schema \"{}\" is not supported",
                        create.name
                    )));
                }
                if cat.relational_public_schema_exists
                    && !create.if_not_exists
                    && !cat.relational_public_schema_implicit
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "schema \"{}\" already exists",
                        create.name
                    )));
                }
            }
            Command::DropSchema(drop) => {
                if drop.name != PUBLIC_SCHEMA_NAME {
                    if !drop.if_exists {
                        return Err(EngineError::ApplyFailed(format!(
                            "schema \"{}\" does not exist",
                            drop.name
                        )));
                    }
                    return Ok(());
                }
                if !cat.relational_public_schema_exists && !drop.if_exists {
                    return Err(EngineError::ApplyFailed(format!(
                        "schema \"{}\" does not exist",
                        drop.name
                    )));
                }
                if cat.relational_public_schema_exists
                    && (!cat.relational_catalog.is_empty()
                        || !cat.relational_views.is_empty()
                        || !cat.relational_materialized_views.is_empty()
                        || !cat.relational_functions.is_empty()
                        || !cat.relational_sequences.is_empty()
                        || !cat.relational_domains.is_empty()
                        || !cat.relational_publications.is_empty()
                        || !cat.relational_subscriptions.is_empty())
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot drop non-empty schema \"{}\"",
                        drop.name
                    )));
                }
            }
            Command::CreateDatabase(create) if self.database_exists(&create.name) => {
                return Err(EngineError::ApplyFailed(format!(
                    "database \"{}\" already exists",
                    create.name
                )));
            }
            Command::CreateDatabase(_) => {}
            Command::DropDatabase(drop) => {
                let mut seen = BTreeSet::new();
                for database in &drop.names {
                    if !seen.insert(database.clone()) {
                        return Err(EngineError::ApplyFailed(format!(
                            "database \"{}\" specified more than once",
                            database
                        )));
                    }
                    if database == "postgres" {
                        return Err(EngineError::ApplyFailed(
                            "cannot drop bootstrap database \"postgres\"".to_string(),
                        ));
                    }
                    if !drop.if_exists && !cat.relational_databases.contains_key(database) {
                        return Err(EngineError::ApplyFailed(format!(
                            "database \"{}\" does not exist",
                            database
                        )));
                    }
                }
            }
            Command::RenameDatabase(rename) => {
                if rename.old_name == "postgres" {
                    return Err(EngineError::ApplyFailed(
                        "cannot rename bootstrap database \"postgres\"".to_string(),
                    ));
                }
                if !cat.relational_databases.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "database \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if self.database_exists(&rename.new_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "database \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::CreateTablespace(create) if self.tablespace_exists(&create.name) => {
                return Err(EngineError::ApplyFailed(format!(
                    "tablespace \"{}\" already exists",
                    create.name
                )));
            }
            Command::CreateTablespace(_) => {}
            Command::DropTablespace(drop) => {
                let mut seen = BTreeSet::new();
                for tablespace in &drop.names {
                    if !seen.insert(tablespace.clone()) {
                        return Err(EngineError::ApplyFailed(format!(
                            "tablespace \"{}\" specified more than once",
                            tablespace
                        )));
                    }
                    if matches!(tablespace.as_str(), "pg_default" | "pg_global") {
                        return Err(EngineError::ApplyFailed(format!(
                            "cannot drop bootstrap tablespace \"{}\"",
                            tablespace
                        )));
                    }
                    if !drop.if_exists && !cat.relational_tablespaces.contains_key(tablespace) {
                        return Err(EngineError::ApplyFailed(format!(
                            "tablespace \"{}\" does not exist",
                            tablespace
                        )));
                    }
                }
            }
            Command::RenameTablespace(rename) => {
                if matches!(rename.old_name.as_str(), "pg_default" | "pg_global") {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot rename bootstrap tablespace \"{}\"",
                        rename.old_name
                    )));
                }
                if !cat.relational_tablespaces.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "tablespace \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if self.tablespace_exists(&rename.new_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "tablespace \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::CreateTable(create) => {
                // TYPE-COVERAGE #14 Track 3: compound PK/UNIQUE over i32-SECTION columns
                // (Int4/Date/Int2) is device-native (folded to a fingerprint surrogate — see
                // `compound_key_fingerprint`); a compound key touching any WIDER type stays REJECTED in
                // PREFLIGHT (before the WAL write — an apply-time reject would strand the command in the
                // WAL to poison every replay). Single-column keys pass. Keep this in lock-step with the
                // apply-layer guard in `apply_create_table`.
                let ct_compound_ok = |cols: &[String]| -> bool {
                    cols.iter().all(|name| {
                        create
                            .columns
                            .iter()
                            .find(|c| &c.name == name)
                            .is_some_and(|c| {
                                crate::engine_residency::compound_key_type_supported(c.ty)
                            })
                    })
                };
                if create
                    .primary_key
                    .as_ref()
                    .is_some_and(|pk| pk.columns.len() > 1 && !ct_compound_ok(&pk.columns))
                    || create
                        .unique_constraints
                        .iter()
                        .any(|u| u.columns.len() > 1 && !ct_compound_ok(&u.columns))
                {
                    return Err(EngineError::ApplyFailed(
                        "compound PRIMARY KEY / UNIQUE constraints are not yet supported \
                         (compound key columns must be int4, int2, date, int8, timestamp, numeric, uuid, or text)"
                            .to_string(),
                    ));
                }
                if !cat.relational_public_schema_exists {
                    return Err(EngineError::ApplyFailed(format!(
                        "schema \"{}\" does not exist",
                        PUBLIC_SCHEMA_NAME
                    )));
                }
                let mut implicit_sequences = BTreeSet::new();
                for column in &create.columns {
                    if let Some(domain_name) = column.domain.as_ref() {
                        if !cat.relational_domains.contains_key(domain_name) {
                            return Err(EngineError::ApplyFailed(format!(
                                "type \"{}\" does not exist",
                                domain_name
                            )));
                        }
                    }
                }
                for default in sequence_defaults(&create.columns) {
                    if let ColumnDefault::SequenceNextVal {
                        sequence,
                        create_if_missing: true,
                    } = default
                    {
                        if !implicit_sequences.insert(sequence.clone()) {
                            return Err(EngineError::ApplyFailed(format!(
                                "relation \"{sequence}\" already exists"
                            )));
                        }
                    }
                    self.preflight_column_default_target(default)?;
                }
            }
            Command::CreateIndex(create) if create.unique => {
                if cat
                    .relational_catalog
                    .values()
                    .any(|table| table.indexes.iter().any(|index| index.name == create.name))
                    || cat.relational_catalog.contains_key(&create.name)
                    || cat.relational_views.contains_key(&create.name)
                    || cat.relational_materialized_views.contains_key(&create.name)
                    || cat.relational_sequences.contains_key(&create.name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        create.name
                    )));
                }
                let table = cat.relational_catalog.get(&create.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        create.table
                    ))
                })?;
                // COMPOUND KEYS: resolve EVERY key column; a compound UNIQUE over i32-section columns
                // is device-native, any wider type stays rejected in preflight (lock-step with apply).
                let column_idxs = create
                    .columns
                    .iter()
                    .map(|name| {
                        table
                            .columns
                            .iter()
                            .position(|column| &column.name == name)
                            .ok_or_else(|| {
                                EngineError::ApplyFailed(format!(
                                    "column \"{name}\" does not exist"
                                ))
                            })
                    })
                    .collect::<Result<Vec<usize>, _>>()?;
                if column_idxs.len() > 1
                    && !column_idxs.iter().all(|&i| {
                        crate::engine_residency::compound_key_type_supported(table.columns[i].ty)
                    })
                {
                    return Err(EngineError::ApplyFailed(
                        "compound PRIMARY KEY / UNIQUE constraints are not yet supported \
                         (compound key columns must be int4, int2, date, int8, timestamp, numeric, uuid, or text)"
                            .to_string(),
                    ));
                }
                let rows = self.visible_relational_rows(
                    table,
                    StorageVisibility {
                        read_txn_id: self.committed_seq() as TxnId,
                    },
                )?;
                Self::validate_unique_values_tuple(&rows, &column_idxs, &create.name)?;
            }
            Command::AddPrimaryKey(add) => {
                if cat
                    .relational_catalog
                    .values()
                    .any(|table| table.indexes.iter().any(|index| index.name == add.name))
                    || cat.relational_catalog.contains_key(&add.name)
                    || cat.relational_views.contains_key(&add.name)
                    || cat.relational_materialized_views.contains_key(&add.name)
                    || cat.relational_sequences.contains_key(&add.name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        add.name
                    )));
                }
                let table = cat.relational_catalog.get(&add.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", add.table))
                })?;
                if table.indexes.iter().any(|index| index.primary_key) {
                    return Err(EngineError::ApplyFailed(format!(
                        "multiple primary keys for table \"{}\" are not allowed",
                        add.table
                    )));
                }
                // COMPOUND KEYS: resolve EVERY key column; compound PK over i32-section columns is
                // device-native, any wider type stays rejected in preflight (lock-step with apply).
                let column_idxs = add
                    .columns
                    .iter()
                    .map(|name| {
                        table
                            .columns
                            .iter()
                            .position(|column| &column.name == name)
                            .ok_or_else(|| {
                                EngineError::ApplyFailed(format!(
                                    "column \"{name}\" does not exist"
                                ))
                            })
                    })
                    .collect::<Result<Vec<usize>, _>>()?;
                if column_idxs.len() > 1
                    && !column_idxs.iter().all(|&i| {
                        crate::engine_residency::compound_key_type_supported(table.columns[i].ty)
                    })
                {
                    return Err(EngineError::ApplyFailed(
                        "compound PRIMARY KEY / UNIQUE constraints are not yet supported \
                         (compound key columns must be int4, int2, date, int8, timestamp, numeric, uuid, or text)"
                            .to_string(),
                    ));
                }
                let rows = self.visible_relational_rows(
                    table,
                    StorageVisibility {
                        read_txn_id: self.committed_seq() as TxnId,
                    },
                )?;
                // PG: ADD PRIMARY KEY over existing data requires EVERY key column non-null (23502) —
                // checked in PREFLIGHT (a failure inside apply would strand the entry in the commit
                // pipeline), byte-identical to the apply-layer guard in
                // `apply_create_index_with_constraint_flags`.
                if let Some(&null_col) = column_idxs
                    .iter()
                    .find(|&&i| rows.iter().any(|row| matches!(row[i], SqlValue::Null)))
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" of relation \"{}\" contains null values",
                        table.columns[null_col].name, add.table
                    )));
                }
                Self::validate_unique_values_tuple(&rows, &column_idxs, &add.name)?;
            }
            Command::AddUniqueConstraint(add) => {
                if cat
                    .relational_catalog
                    .values()
                    .any(|table| table.indexes.iter().any(|index| index.name == add.name))
                    || cat.relational_catalog.contains_key(&add.name)
                    || cat.relational_views.contains_key(&add.name)
                    || cat.relational_materialized_views.contains_key(&add.name)
                    || cat.relational_sequences.contains_key(&add.name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        add.name
                    )));
                }
                let table = cat.relational_catalog.get(&add.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", add.table))
                })?;
                // COMPOUND KEYS: resolve EVERY key column; compound UNIQUE over i32-section columns is
                // device-native, any wider type stays rejected in preflight (lock-step with apply).
                let column_idxs = add
                    .columns
                    .iter()
                    .map(|name| {
                        table
                            .columns
                            .iter()
                            .position(|column| &column.name == name)
                            .ok_or_else(|| {
                                EngineError::ApplyFailed(format!(
                                    "column \"{name}\" does not exist"
                                ))
                            })
                    })
                    .collect::<Result<Vec<usize>, _>>()?;
                if column_idxs.len() > 1
                    && !column_idxs.iter().all(|&i| {
                        crate::engine_residency::compound_key_type_supported(table.columns[i].ty)
                    })
                {
                    return Err(EngineError::ApplyFailed(
                        "compound PRIMARY KEY / UNIQUE constraints are not yet supported \
                         (compound key columns must be int4, int2, date, int8, timestamp, numeric, uuid, or text)"
                            .to_string(),
                    ));
                }
                let rows = self.visible_relational_rows(
                    table,
                    StorageVisibility {
                        read_txn_id: self.committed_seq() as TxnId,
                    },
                )?;
                Self::validate_unique_values_tuple(&rows, &column_idxs, &add.name)?;
            }
            Command::AddCheckConstraint(add) => self.preflight_add_check_constraint(add)?,
            Command::AddForeignKey(add) => self.preflight_add_foreign_key(add, txn_id)?,
            Command::AddColumn(add) => {
                if cat.relational_views.contains_key(&add.table)
                    || cat.relational_materialized_views.contains_key(&add.table)
                    || cat.relational_sequences.contains_key(&add.table)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        add.table
                    )));
                }
                let Some(default) = add.column.default.as_ref() else {
                    return Err(EngineError::ApplyFailed(
                        "ADD COLUMN requires a supported DEFAULT in the bootstrap relational subset"
                            .to_string(),
                    ));
                };
                if !add_column_default_supported(default) {
                    return Err(EngineError::ApplyFailed(
                        "ADD COLUMN SERIAL is unsupported in the bootstrap relational subset"
                            .to_string(),
                    ));
                }
                // Validate the default is coercible to the column type (parity with apply);
                // this concurrent-DDL preflight only checks — apply coerces and stores.
                coerce_column_default(default.clone(), add.column.ty, &add.column.name)?;
                let table = cat.relational_catalog.get(&add.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", add.table))
                })?;
                if table
                    .columns
                    .iter()
                    .any(|column| column.name == add.column.name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" of relation \"{}\" already exists",
                        add.column.name, add.table
                    )));
                }
                self.preflight_column_default_target(default)?;
            }
            Command::RenameTable(rename) => {
                if cat.relational_views.contains_key(&rename.old_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.old_name)
                    || cat.relational_sequences.contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        rename.old_name
                    )));
                }
                if !cat.relational_catalog.contains_key(&rename.old_name) {
                    if rename.if_exists {
                        return Ok(());
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
                if cat
                    .relational_views
                    .values()
                    .any(|view| view.query.table == rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot rename relation \"{}\" because a view depends on it",
                        rename.old_name
                    )));
                }
            }
            Command::RenameColumn(rename) => {
                if cat.relational_views.contains_key(&rename.table)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.table)
                    || cat.relational_sequences.contains_key(&rename.table)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        rename.table
                    )));
                }
                let table = cat.relational_catalog.get(&rename.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        rename.table
                    ))
                })?;
                if !table
                    .columns
                    .iter()
                    .any(|column| column.name == rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if table
                    .columns
                    .iter()
                    .any(|column| column.name == rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" of relation \"{}\" already exists",
                        rename.new_name, rename.table
                    )));
                }
            }
            Command::RenameConstraint(rename) => {
                if cat.relational_views.contains_key(&rename.table)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.table)
                    || cat.relational_sequences.contains_key(&rename.table)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        rename.table
                    )));
                }
                let Some(table) = cat.relational_catalog.get(&rename.table) else {
                    if rename.table_if_exists {
                        return Ok(());
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        rename.table
                    )));
                };
                if cat.relational_catalog.values().any(|candidate| {
                    candidate
                        .indexes
                        .iter()
                        .any(|index| index.name == rename.new_name)
                        || candidate
                            .check_constraints
                            .iter()
                            .any(|constraint| constraint.name == rename.new_name)
                        || candidate
                            .foreign_keys
                            .iter()
                            .any(|constraint| constraint.name == rename.new_name)
                }) || cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
                if !table.indexes.iter().any(|index| {
                    index.name == rename.old_name && (index.primary_key || index.unique_constraint)
                }) && !table
                    .check_constraints
                    .iter()
                    .any(|constraint| constraint.name == rename.old_name)
                    && !table
                        .foreign_keys
                        .iter()
                        .any(|constraint| constraint.name == rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "constraint \"{}\" does not exist",
                        rename.old_name
                    )));
                }
            }
            Command::RenameIndex(rename) => {
                if cat.relational_catalog.values().any(|table| {
                    table
                        .indexes
                        .iter()
                        .any(|index| index.name == rename.new_name)
                }) || cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
                let Some(index) = cat
                    .relational_catalog
                    .values()
                    .flat_map(|table| table.indexes.iter())
                    .find(|index| index.name == rename.old_name)
                else {
                    return Err(EngineError::ApplyFailed(format!(
                        "index \"{}\" does not exist",
                        rename.old_name
                    )));
                };
                if index.primary_key || index.unique_constraint {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot rename constraint-backed index \"{}\" with ALTER INDEX",
                        rename.old_name
                    )));
                }
            }
            Command::CreateView(create) => {
                if cat.relational_catalog.contains_key(&create.name)
                    || cat.relational_materialized_views.contains_key(&create.name)
                    || cat.relational_sequences.contains_key(&create.name)
                    || (!create.or_replace && cat.relational_views.contains_key(&create.name))
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        create.name
                    )));
                }
                if cat
                    .relational_materialized_views
                    .contains_key(&create.query.table)
                {
                    return Err(EngineError::ApplyFailed(
                        "views over materialized views are unsupported".to_string(),
                    ));
                }
                if create.or_replace && self.relational_view_has_dependents(&create.name) {
                    return Err(EngineError::ApplyFailed(
                        "cannot replace view because another view depends on it".to_string(),
                    ));
                }
                if cat.relational_views.contains_key(&create.query.table) {
                    if self.relational_view_depends_on(&create.query.table, &create.name) {
                        return Err(EngineError::ApplyFailed(
                            "view dependency cycle is unsupported".to_string(),
                        ));
                    }
                } else if !cat.relational_catalog.contains_key(&create.query.table) {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        create.query.table
                    )));
                }
            }
            Command::RenameView(rename) => {
                if cat.relational_catalog.contains_key(&rename.old_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.old_name)
                    || cat.relational_sequences.contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a view",
                        rename.old_name
                    )));
                }
                if !cat.relational_views.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "view \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
                if self.relational_view_has_dependents(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot rename view \"{}\" because another view depends on it",
                        rename.old_name
                    )));
                }
            }
            Command::CreateMaterializedView(create) => {
                self.preflight_create_materialized_view(create)?
            }
            Command::RefreshMaterializedView(refresh) => {
                self.preflight_refresh_materialized_view(refresh)?
            }
            Command::RenameMaterializedView(rename) => {
                if cat.relational_catalog.contains_key(&rename.old_name)
                    || cat.relational_views.contains_key(&rename.old_name)
                    || cat.relational_sequences.contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a materialized view",
                        rename.old_name
                    )));
                }
                if !cat
                    .relational_materialized_views
                    .contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "materialized view \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::CreateFunction(create)
                if cat.relational_functions.contains_key(&create.name) =>
            {
                return Err(EngineError::ApplyFailed(format!(
                    "function \"{}\" already exists",
                    create.name
                )));
            }
            Command::RenameFunction(rename) => {
                if !cat.relational_functions.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "function \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_functions.contains_key(&rename.new_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "function \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::DropFunction(drop)
                if !drop.if_exists && !cat.relational_functions.contains_key(&drop.name) =>
            {
                return Err(EngineError::ApplyFailed(format!(
                    "function \"{}\" does not exist",
                    drop.name
                )));
            }
            Command::CreateFunction(_) | Command::DropFunction(_) => {}
            Command::CommentOn(comment) => {
                if let CommentTarget::Function { function } = &comment.target {
                    if !cat.relational_functions.contains_key(function) {
                        return Err(EngineError::ApplyFailed(format!(
                            "function \"{}\" does not exist",
                            function
                        )));
                    }
                }
            }
            Command::CreateSequence(create) => self.preflight_create_sequence(create)?,
            Command::CreateDomain(create) => self.preflight_create_domain(create)?,
            Command::SequenceNextVal(nextval) => self.preflight_sequence_target(&nextval.name)?,
            Command::SequenceSetVal(setval) => self.preflight_sequence_target(&setval.name)?,
            Command::RenameSequence(rename) => {
                if cat.relational_catalog.contains_key(&rename.old_name)
                    || cat.relational_views.contains_key(&rename.old_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a sequence",
                        rename.old_name
                    )));
                }
                if !cat.relational_sequences.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "sequence \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::DropColumn(drop) => {
                if cat.relational_views.contains_key(&drop.table)
                    || cat.relational_materialized_views.contains_key(&drop.table)
                    || cat.relational_sequences.contains_key(&drop.table)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        drop.table
                    )));
                }
                let table = cat.relational_catalog.get(&drop.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", drop.table))
                })?;
                if !table
                    .columns
                    .iter()
                    .any(|column| column.name == drop.column)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" does not exist",
                        drop.column
                    )));
                }
                if table
                    .indexes
                    .iter()
                    // COMPOUND KEYS: block the drop when the column is ANY key column of a compound index.
                    .any(|index| index.key_columns.contains(&drop.column))
                    || table
                        .check_constraints
                        .iter()
                        .any(|constraint| constraint.column == drop.column)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot drop column \"{}\" because an index or constraint depends on it",
                        drop.column
                    )));
                }
            }
            Command::DropConstraint(drop) => {
                let Some(table) = cat.relational_catalog.get(&drop.table) else {
                    if drop.table_if_exists {
                        return Ok(());
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        drop.table
                    )));
                };
                if !table.indexes.iter().any(|index| {
                    index.name == drop.name && (index.primary_key || index.unique_constraint)
                }) && !table
                    .check_constraints
                    .iter()
                    .any(|constraint| constraint.name == drop.name)
                    && !table
                        .foreign_keys
                        .iter()
                        .any(|constraint| constraint.name == drop.name)
                    && !drop.if_exists
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "constraint \"{}\" does not exist",
                        drop.name
                    )));
                }
            }
            Command::DropTable(drop) => self.preflight_drop_table(drop)?,
            Command::DropIndex(drop) => self.preflight_drop_index(drop)?,
            Command::DropView(drop) => self.preflight_drop_view(drop)?,
            Command::DropMaterializedView(drop) => self.preflight_drop_materialized_view(drop)?,
            Command::DropSequence(drop) => self.preflight_drop_sequence(drop)?,
            Command::DropDomain(drop) => self.preflight_drop_domain(drop)?,
            Command::GrantTable(grant) => {
                self.preflight_acl_target(&grant.relation, grant.kind)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeTable(revoke) => {
                self.preflight_acl_target(&revoke.relation, revoke.kind)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::GrantSchema(grant) => {
                self.preflight_schema_acl_target(&grant.schema)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeSchema(revoke) => {
                self.preflight_schema_acl_target(&revoke.schema)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::GrantDatabase(grant) => {
                self.preflight_database_acl_target(&grant.database)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeDatabase(revoke) => {
                self.preflight_database_acl_target(&revoke.database)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::GrantTablespace(grant) => {
                self.preflight_tablespace_acl_target(&grant.tablespace)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeTablespace(revoke) => {
                self.preflight_tablespace_acl_target(&revoke.tablespace)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::GrantFunction(grant) => {
                self.preflight_function_acl_target(&grant.function)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeFunction(revoke) => {
                self.preflight_function_acl_target(&revoke.function)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::CreatePublication(create) => self.preflight_create_publication(create)?,
            Command::DropPublication(drop) => self.preflight_drop_publication(drop)?,
            Command::CreateSubscription(create) => self.preflight_create_subscription(create)?,
            Command::DropSubscription(drop) => self.preflight_drop_subscription(drop)?,
            Command::CreateRole(create) if self.role_exists(&create.name) => {
                return Err(EngineError::ApplyFailed(format!(
                    "role \"{}\" already exists",
                    create.name
                )));
            }
            Command::CreateRole(_) => {}
            Command::DropRole(drop) => {
                let mut seen = BTreeSet::new();
                for role in &drop.names {
                    if !seen.insert(role.clone()) {
                        return Err(EngineError::ApplyFailed(format!(
                            "role \"{}\" specified more than once",
                            role
                        )));
                    }
                    if role == "postgres" {
                        return Err(EngineError::ApplyFailed(
                            "cannot drop bootstrap role \"postgres\"".to_string(),
                        ));
                    }
                    if !drop.if_exists && !cat.relational_roles.contains_key(role) {
                        return Err(EngineError::ApplyFailed(format!(
                            "role \"{}\" does not exist",
                            role
                        )));
                    }
                    if cat.relational_roles.contains_key(role) && self.role_has_dependencies(role) {
                        return Err(EngineError::ApplyFailed(format!(
                            "role \"{}\" cannot be dropped because dependent metadata exists",
                            role
                        )));
                    }
                }
            }
            Command::RenameRole(rename) => {
                if rename.old_name == "postgres" {
                    return Err(EngineError::ApplyFailed(
                        "cannot rename bootstrap role \"postgres\"".to_string(),
                    ));
                }
                if !cat.relational_roles.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "role \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if self.role_exists(&rename.new_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "role \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::GrantDefaultTablePrivileges(grant) => {
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeDefaultTablePrivileges(revoke) => {
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::Insert(insert) => {
                let snapshot = DmlReadSnapshot {
                    // The facade transaction id is an idempotency identity, not an MVCC
                    // boundary. Serialized preflight reads the latest published generation;
                    // the under-lock apply re-resolves at its assigned commit sequence.
                    commit_seq: self.committed_seq(),
                    next_row_id: self.read_state.mvcc.current_row_id(),
                };
                self.prepare_insert(
                    insert,
                    snapshot,
                    None,
                    crate::engine_dml_prepare::InsertPrepareValidation::Full,
                )?;
            }
            Command::Update(update) => {
                let snapshot = DmlReadSnapshot {
                    commit_seq: self.committed_seq(),
                    next_row_id: self.read_state.mvcc.current_row_id(),
                };
                self.prepare_update(update, snapshot)?;
            }
            Command::Delete(delete) => {
                let snapshot = DmlReadSnapshot {
                    commit_seq: self.committed_seq(),
                    next_row_id: self.read_state.mvcc.current_row_id(),
                };
                self.prepare_delete(delete, snapshot)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Definitive serialized-DML constraint check. The ordinary preflight intentionally runs
    /// before `commit_mutex`, so it cannot close a child-insert/parent-delete race by itself.
    /// Re-resolve a class-authoritative parent DELETE at the current MVCC boundary while the
    /// caller holds that mutex, immediately before WAL append. Other DML keeps its existing
    /// sequencer/serialized path; binary recovery payloads are already durable records.
    pub(crate) fn preflight_serialized_dml_under_commit_lock(
        &self,
        payload: &[u8],
    ) -> Result<(), EngineError> {
        let Some(Command::Delete(delete)) = std::str::from_utf8(payload)
            .ok()
            .and_then(|text| gpu_db_sql::parse_command(text).ok())
        else {
            return Ok(());
        };
        let catalog = self.catalog_snapshot();
        let Some(table) = catalog.relational_catalog.get(&delete.table) else {
            return Ok(());
        };
        let has_inbound_fk = catalog.relational_catalog.values().any(|candidate| {
            candidate
                .foreign_keys
                .iter()
                .any(|foreign_key| foreign_key.referenced_table == table.name)
        });
        if !has_inbound_fk || self.table_chunk_authoritative(&table.name).is_none() {
            return Ok(());
        }
        let snapshot = DmlReadSnapshot {
            commit_seq: self.committed_seq(),
            next_row_id: self.read_state.mvcc.current_row_id(),
        };
        self.skip_leader_check_during_internal_read(|engine| {
            engine.prepare_delete(&delete, snapshot).map(|_| ())
        })
    }
}
