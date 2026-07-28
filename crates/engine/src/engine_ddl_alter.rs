//! ALTER COLUMN + COMMENT DDL (P0 §9.6 decomposition, behavior-preserving): a
//! focused `impl Engine` block for COMMENT ON every object kind (apply_comment_on)
//! and the ALTER TABLE column operations — apply_alter_column_default,
//! apply_add_column, apply_rename_column, apply_drop_column (rewriting resident
//! rows + catalog as needed).

use super::*;

impl Engine {
    pub(crate) fn apply_comment_on(
        &self,
        cat: &mut DdlCatalogState,
        comment: gpu_db_sql::CommentOn,
    ) -> Result<(), EngineError> {
        let target = match comment.target {
            CommentTarget::Database { database } => {
                if !self.database_exists(&database) {
                    return Err(EngineError::ApplyFailed(format!(
                        "database \"{}\" does not exist",
                        database
                    )));
                }
                RelationalCommentTarget::Database { database }
            }
            CommentTarget::Role { role } => {
                if !self.role_exists(&role) {
                    return Err(EngineError::ApplyFailed(format!(
                        "role \"{}\" does not exist",
                        role
                    )));
                }
                RelationalCommentTarget::Role { role }
            }
            CommentTarget::Schema { schema } => {
                if schema != "public" {
                    return Err(EngineError::ApplyFailed(format!(
                        "schema \"{}\" does not exist",
                        schema
                    )));
                }
                RelationalCommentTarget::Schema { schema }
            }
            CommentTarget::Tablespace { tablespace } => {
                if !self.tablespace_exists(&tablespace) {
                    return Err(EngineError::ApplyFailed(format!(
                        "tablespace \"{}\" does not exist",
                        tablespace
                    )));
                }
                RelationalCommentTarget::Tablespace { tablespace }
            }
            CommentTarget::Table { table } => {
                if !cat.relational_catalog.contains_key(&table) {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        table
                    )));
                }
                RelationalCommentTarget::Table { table }
            }
            CommentTarget::Column { table, column } => {
                let table_ref = cat.relational_catalog.get(&table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", table))
                })?;
                let column_ref = table_ref
                    .columns
                    .iter()
                    .find(|candidate| candidate.name == column)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!("column \"{}\" does not exist", column))
                    })?;
                RelationalCommentTarget::Column {
                    table,
                    attnum: column_ref.attnum,
                }
            }
            CommentTarget::Index { index } => {
                if !cat.relational_catalog.values().any(|table| {
                    table
                        .indexes
                        .iter()
                        .any(|candidate| candidate.name == index)
                }) {
                    return Err(EngineError::ApplyFailed(format!(
                        "index \"{}\" does not exist",
                        index
                    )));
                }
                RelationalCommentTarget::Index { index }
            }
            CommentTarget::View { view } => {
                if !cat.relational_views.contains_key(&view) {
                    if cat.relational_catalog.contains_key(&view)
                        || cat.relational_materialized_views.contains_key(&view)
                        || cat.relational_sequences.contains_key(&view)
                    {
                        return Err(EngineError::ApplyFailed(format!(
                            "relation \"{}\" is not a view",
                            view
                        )));
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "view \"{}\" does not exist",
                        view
                    )));
                }
                RelationalCommentTarget::View { view }
            }
            CommentTarget::MaterializedView { materialized_view } => {
                if !cat
                    .relational_materialized_views
                    .contains_key(&materialized_view)
                {
                    if cat.relational_catalog.contains_key(&materialized_view)
                        || cat.relational_views.contains_key(&materialized_view)
                        || cat.relational_sequences.contains_key(&materialized_view)
                    {
                        return Err(EngineError::ApplyFailed(format!(
                            "relation \"{}\" is not a materialized view",
                            materialized_view
                        )));
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "materialized view \"{}\" does not exist",
                        materialized_view
                    )));
                }
                RelationalCommentTarget::MaterializedView { materialized_view }
            }
            CommentTarget::Function { function } => {
                if !cat.relational_functions.contains_key(&function) {
                    return Err(EngineError::ApplyFailed(format!(
                        "function \"{}\" does not exist",
                        function
                    )));
                }
                RelationalCommentTarget::Function { function }
            }
            CommentTarget::Extension { extension } => {
                if extension != "plpgsql" {
                    return Err(EngineError::ApplyFailed(format!(
                        "extension \"{}\" does not exist",
                        extension
                    )));
                }
                RelationalCommentTarget::Extension { extension }
            }
            CommentTarget::Sequence { sequence } => {
                if !cat.relational_sequences.contains_key(&sequence) {
                    if cat.relational_catalog.contains_key(&sequence)
                        || cat.relational_views.contains_key(&sequence)
                        || cat.relational_materialized_views.contains_key(&sequence)
                    {
                        return Err(EngineError::ApplyFailed(format!(
                            "relation \"{}\" is not a sequence",
                            sequence
                        )));
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "sequence \"{}\" does not exist",
                        sequence
                    )));
                }
                RelationalCommentTarget::Sequence { sequence }
            }
            CommentTarget::Domain { domain } => {
                if !cat.relational_domains.contains_key(&domain) {
                    return Err(EngineError::ApplyFailed(format!(
                        "domain \"{}\" does not exist",
                        domain
                    )));
                }
                RelationalCommentTarget::Domain { domain }
            }
            CommentTarget::Publication { publication } => {
                if !cat.relational_publications.contains_key(&publication) {
                    return Err(EngineError::ApplyFailed(format!(
                        "publication \"{}\" does not exist",
                        publication
                    )));
                }
                RelationalCommentTarget::Publication { publication }
            }
            CommentTarget::Subscription { subscription } => {
                if !cat.relational_subscriptions.contains_key(&subscription) {
                    return Err(EngineError::ApplyFailed(format!(
                        "subscription \"{}\" does not exist",
                        subscription
                    )));
                }
                RelationalCommentTarget::Subscription { subscription }
            }
            CommentTarget::Constraint { table, constraint } => {
                let table_ref = cat.relational_catalog.get(&table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", table))
                })?;
                if !table_ref.indexes.iter().any(|candidate| {
                    candidate.name == constraint
                        && (candidate.primary_key || candidate.unique_constraint)
                }) && !table_ref
                    .check_constraints
                    .iter()
                    .any(|candidate| candidate.name == constraint)
                    && !table_ref
                        .foreign_keys
                        .iter()
                        .any(|candidate| candidate.name == constraint)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "constraint \"{}\" does not exist",
                        constraint
                    )));
                }
                RelationalCommentTarget::Constraint { table, constraint }
            }
        };
        if let Some(value) = comment.comment {
            cat.relational_comments.insert(target, value);
        } else {
            cat.relational_comments.remove(&target);
        }
        Ok(())
    }

    pub(crate) fn apply_alter_column_default(
        &self,
        cat: &mut DdlCatalogState,
        alter: gpu_db_sql::AlterColumnDefault,
    ) -> Result<(), EngineError> {
        // Resolve an explicit nextval regclass name before binding the target type, matching
        // CREATE and preflight.  A missing target therefore wins over a later assignment
        // mismatch, while full sequence-kind validation follows the successful bind.  The
        // coercion remains before the mutable borrow so
        // `ALTER ... SET DEFAULT 0` on a numeric column is stored at the column scale.
        let coerced_default = if let Some(default) = alter.default {
            let table = cat.relational_catalog.get(&alter.table).ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", alter.table))
            })?;
            let column = table
                .columns
                .iter()
                .find(|column| column.name == alter.column)
                .ok_or_else(|| {
                    EngineError::ApplyFailed(format!("column \"{}\" does not exist", alter.column))
                })?;
            self.preflight_column_default_target_before_binding(&default)?;
            let coerced = coerce_column_default(default, column.ty, &alter.column)?;
            self.preflight_column_default_target(&coerced)?;
            Some(coerced)
        } else {
            None
        };
        let table = cat
            .relational_catalog
            .get_mut(&alter.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", alter.table))
            })?;
        let column = table
            .columns
            .iter_mut()
            .find(|column| column.name == alter.column)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("column \"{}\" does not exist", alter.column))
            })?;
        column.default = coerced_default;
        Ok(())
    }

    pub(crate) fn apply_add_column(
        &self,
        cat: &mut DdlCatalogState,
        add: gpu_db_sql::AddColumn,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        let mut column_def = add.column;
        if cat.relational_views.contains_key(&add.table)
            || cat.relational_materialized_views.contains_key(&add.table)
            || cat.relational_sequences.contains_key(&add.table)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a table",
                add.table
            )));
        }
        // Resolve the structural target before touching DEFAULT semantics.  Keep this ordering
        // identical to preflight so apply cannot turn missing-table/duplicate-column diagnostics
        // into a DEFAULT failure.
        let table = cat
            .relational_catalog
            .get(&add.table)
            .ok_or_else(|| EngineError::UndefinedRelation(add.table.clone()))?
            .clone();
        if table
            .columns
            .iter()
            .any(|column| column.name == column_def.name)
        {
            return Err(EngineError::DuplicateColumn(column_def.name.clone()));
        }
        let (type_oid, type_size) = self.resolve_column_domain_type(&mut column_def)?;
        let Some(default) = column_def.default.clone() else {
            return Err(EngineError::ApplyFailed(
                "ADD COLUMN requires a supported DEFAULT in the bootstrap relational subset"
                    .to_string(),
            ));
        };
        if !add_column_default_supported(&default) {
            return Err(EngineError::ApplyFailed(
                "ADD COLUMN SERIAL is unsupported in the bootstrap relational subset".to_string(),
            ));
        }
        // Resolve explicit regclass existence before binding, then validate full sequence kind
        // only after a successful bind.  Store the coercion so existing-row backfill and
        // `from_def` share the same durable default expression.
        self.preflight_column_default_target_before_binding(&default)?;
        let default = coerce_column_default(default, column_def.ty, &column_def.name)?;
        self.preflight_column_default_target(&default)?;
        column_def.default = Some(default.clone());
        // ADD COLUMN evaluates scalar defaults once at the DDL boundary even for an
        // empty relation.  The durable expression remains on the new column for
        // future INSERTs; only the computed scalar is broadcast into rewritten rows.
        let eager_scalar_default = (!is_sequence(&default))
            .then(|| evaluate_scalar(&default, column_def.ty, &column_def.name))
            .transpose()?;
        let row_count = self
            .visible_relational_rows(
                &table,
                StorageVisibility {
                    read_txn_id: txn_id,
                },
            )?
            .len();
        let default_values = (0..row_count)
            .map(|_| match &eager_scalar_default {
                Some(value) => Ok(value.clone()),
                None => {
                    self.evaluate_column_default(cat, &default, column_def.ty, &column_def.name)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let next_attnum = i16::try_from(table.columns.len() + 1).map_err(|_| {
            EngineError::ApplyFailed("too many columns for bootstrap catalog".to_string())
        })?;
        let column_id = cat.relational_next_column_id;
        let next_column_id = cat
            .relational_next_column_id
            .checked_add(1)
            .ok_or_else(|| {
                EngineError::ApplyFailed("relational column id allocation exhausted".to_string())
            })?;
        let new_column = RelationalColumn::from_def(
            column_id,
            table.oid,
            next_attnum,
            column_def,
            type_oid,
            type_size,
        );

        let prefix = relational_key_prefix(&add.table);
        // Re-audit hardening: a row-REWRITING scan (see `ddl_rewrite_scan_visibility`) — an
        // elided stale prefix or a facade-below-committed boundary would silently leave rows
        // behind on the old layout.
        let visibility = self.ddl_rewrite_scan_visibility(&add.table, txn_id)?;
        let mut updates = Vec::new();
        let mut default_values = default_values.into_iter();
        {
            let table_rows = self.read_state.mvcc.table_rows(&add.table);
            let mut cursor = table_rows
                .store()
                .seq_scan_open(visibility)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            while let Some(tuple) = cursor.next() {
                if !tuple.key.starts_with(&prefix) {
                    continue;
                }
                let mut row = decode_relational_row(&tuple.value, &table.columns)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                let default_value = default_values.next().ok_or_else(|| {
                    EngineError::ApplyFailed(
                        "ADD COLUMN default rewrite row count drifted".to_string(),
                    )
                })?;
                row.push(default_value);
                updates.push((tuple.tuple_id, tuple.key.clone(), row));
            }
        }
        if default_values.next().is_some() {
            return Err(EngineError::ApplyFailed(
                "ADD COLUMN default rewrite row count drifted".to_string(),
            ));
        }

        let new_column_name = new_column.name.clone();
        self.read_state.mvcc.with_table_mut(&add.table, |data| {
            for (tuple_id, row_key, values) in &updates {
                let index_value = values.last().expect("new column default appended").clone();
                data.rows
                    .tuple_update(*tuple_id, encode_relational_row(values), txn_id)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                let index_key = ColumnValueKey {
                    column: new_column_name.clone(),
                    value: relational_index_value(&index_value),
                };
                let mut slot = data
                    .value_index
                    .get(&index_key)
                    .cloned()
                    .unwrap_or_default();
                slot.push_back(row_key.clone());
                data.value_index.insert(index_key, slot);
            }
            Ok::<(), EngineError>(())
        })?;
        let table_ref = cat
            .relational_catalog
            .get_mut(&add.table)
            .expect("table existence validated");
        table_ref.columns.push(new_column.clone());
        cat.relational_next_column_id = next_column_id;
        self.read_state
            .residency
            .with_snapshots_mut(|snapshots| snapshots.remove(&add.table));
        self.read_state.residency.device_memory.remove(&add.table);
        Ok(())
    }

    pub(crate) fn apply_rename_column(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameColumn,
    ) -> Result<(), EngineError> {
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
            EngineError::ApplyFailed(format!("relation \"{}\" does not exist", rename.table))
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

        // Rename the column within this table's per-table value-index (keys are `(column, value)`),
        // publishing one new generation. `imbl::OrdMap` has no in-place `retain`; rebuild the index,
        // re-keying the matching column's slots (carrying the same `Arc` row-key list, an O(1) move)
        // and keeping the rest by `Arc`-clone. Order is preserved (OrdMap is ordered).
        self.read_state.mvcc.with_table_mut(&rename.table, |data| {
            let mut rebuilt = imbl::OrdMap::new();
            for (key, row_keys) in data.value_index.iter() {
                let new_key = if key.column == rename.old_name {
                    ColumnValueKey {
                        column: rename.new_name.clone(),
                        value: key.value.clone(),
                    }
                } else {
                    key.clone()
                };
                rebuilt.insert(new_key, row_keys.clone());
            }
            data.value_index = rebuilt;
        });

        let table_ref = cat
            .relational_catalog
            .get_mut(&rename.table)
            .expect("table existence validated");
        let column = table_ref
            .columns
            .iter_mut()
            .find(|column| column.name == rename.old_name)
            .expect("column existence validated");
        column.name = rename.new_name.clone();
        for index in &mut table_ref.indexes {
            if index.column == rename.old_name {
                index.column = rename.new_name.clone();
            }
            // COMPOUND KEYS: rename the column everywhere it appears in a compound key too.
            for key_column in &mut index.key_columns {
                if *key_column == rename.old_name {
                    *key_column = rename.new_name.clone();
                }
            }
        }
        for constraint in &mut table_ref.check_constraints {
            if constraint.column == rename.old_name {
                constraint.column = rename.new_name.clone();
            }
        }
        for constraint in &mut table_ref.foreign_keys {
            if constraint.column == rename.old_name {
                constraint.column = rename.new_name.clone();
            }
            if constraint.referenced_table == rename.table
                && constraint.referenced_column == rename.old_name
            {
                constraint.referenced_column = rename.new_name.clone();
            }
        }
        for candidate in cat.relational_catalog.values_mut() {
            if candidate.name == rename.table {
                continue;
            }
            for constraint in &mut candidate.foreign_keys {
                if constraint.referenced_table == rename.table
                    && constraint.referenced_column == rename.old_name
                {
                    constraint.referenced_column = rename.new_name.clone();
                }
            }
        }
        self.read_state
            .residency
            .with_snapshots_mut(|snapshots| snapshots.remove(&rename.table));
        self.read_state
            .residency
            .device_memory
            .remove(&rename.table);
        Ok(())
    }

    pub(crate) fn apply_drop_column(
        &self,
        cat: &mut DdlCatalogState,
        drop_column: gpu_db_sql::DropColumn,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        if cat.relational_views.contains_key(&drop_column.table)
            || cat
                .relational_materialized_views
                .contains_key(&drop_column.table)
            || cat.relational_sequences.contains_key(&drop_column.table)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a table",
                drop_column.table
            )));
        }
        let table = cat
            .relational_catalog
            .get(&drop_column.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "relation \"{}\" does not exist",
                    drop_column.table
                ))
            })?
            .clone();
        let drop_idx = table
            .columns
            .iter()
            .position(|column| column.name == drop_column.column)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "column \"{}\" does not exist",
                    drop_column.column
                ))
            })?;
        if table
            .indexes
            .iter()
            // COMPOUND KEYS: a column that participates in ANY key column of a compound index also
            // blocks the drop (not just the single-column `index.column`).
            .any(|index| index.key_columns.contains(&drop_column.column))
            || table
                .check_constraints
                .iter()
                .any(|constraint| constraint.column == drop_column.column)
            || table
                .foreign_keys
                .iter()
                .any(|constraint| constraint.column == drop_column.column)
            || cat.relational_catalog.values().any(|candidate| {
                candidate.foreign_keys.iter().any(|constraint| {
                    constraint.referenced_table == drop_column.table
                        && constraint.referenced_column == drop_column.column
                })
            })
        {
            return Err(EngineError::ApplyFailed(format!(
                "cannot drop column \"{}\" because an index or constraint depends on it",
                drop_column.column
            )));
        }

        let prefix = relational_key_prefix(&drop_column.table);
        // Re-audit hardening: row-REWRITING scan (see `ddl_rewrite_scan_visibility`).
        let visibility = self.ddl_rewrite_scan_visibility(&drop_column.table, txn_id)?;
        let mut updates = Vec::new();
        {
            let table_rows = self.read_state.mvcc.table_rows(&drop_column.table);
            let mut cursor = table_rows
                .store()
                .seq_scan_open(visibility)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            while let Some(tuple) = cursor.next() {
                if !tuple.key.starts_with(&prefix) {
                    continue;
                }
                let mut row = decode_relational_row(&tuple.value, &table.columns)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                row.remove(drop_idx);
                updates.push((tuple.tuple_id, row));
            }
        }

        let dropped_column_name = drop_column.column.clone();
        self.read_state
            .mvcc
            .with_table_mut(&drop_column.table, |data| {
                for (tuple_id, values) in &updates {
                    data.rows
                        .tuple_update(*tuple_id, encode_relational_row(values), txn_id)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                }
                // `imbl::OrdMap` has no in-place `retain`; rebuild keeping every slot whose column is
                // not the dropped one (`Arc`-clone is O(1), and OrdMap preserves key order).
                data.value_index = data
                    .value_index
                    .iter()
                    .filter(|(key, _)| key.column != dropped_column_name)
                    .map(|(key, row_keys)| (key.clone(), row_keys.clone()))
                    .collect();
                Ok::<(), EngineError>(())
            })?;

        let dropped_attnum = table.columns[drop_idx].attnum;
        let table_ref = cat
            .relational_catalog
            .get_mut(&drop_column.table)
            .expect("table existence validated");
        table_ref.columns.remove(drop_idx);
        for (idx, column) in table_ref.columns.iter_mut().enumerate() {
            column.attnum = i16::try_from(idx + 1).map_err(|_| {
                EngineError::ApplyFailed("too many columns for bootstrap catalog".to_string())
            })?;
        }

        let shifted_comments = cat
            .relational_comments
            .iter()
            .filter_map(|(target, comment)| match target {
                RelationalCommentTarget::Column { table, attnum }
                    if table == &drop_column.table && *attnum > dropped_attnum =>
                {
                    Some((
                        target.clone(),
                        RelationalCommentTarget::Column {
                            table: table.clone(),
                            attnum: *attnum - 1,
                        },
                        comment.clone(),
                    ))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        cat.relational_comments.retain(|target, _| match target {
            RelationalCommentTarget::Column { table, attnum } => {
                !(table == &drop_column.table && *attnum >= dropped_attnum)
            }
            _ => true,
        });
        for (_, new_target, comment) in shifted_comments {
            cat.relational_comments.insert(new_target, comment);
        }
        self.read_state
            .residency
            .with_snapshots_mut(|snapshots| snapshots.remove(&drop_column.table));
        self.read_state
            .residency
            .device_memory
            .remove(&drop_column.table);
        Ok(())
    }
}
