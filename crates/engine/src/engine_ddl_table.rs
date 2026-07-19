//! Table / index / constraint DDL + row validation (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for CREATE TABLE, indexes
//! and constraints (primary key, unique, check, foreign key, drop/rename
//! constraint), the constraint-validation helpers over resident rows
//! (validate_unique_values/indexes, validate_check_constraints, validate_foreign_keys
//! and their preflights), DROP/RENAME INDEX, RENAME/DROP/TRUNCATE TABLE, and the
//! DROP appliers for view/materialized-view/sequence.

use super::*;

impl Engine {
    pub(crate) fn apply_create_table(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateTable,
    ) -> Result<(), EngineError> {
        if !cat.relational_public_schema_exists {
            return Err(EngineError::ApplyFailed(format!(
                "schema \"{}\" does not exist",
                PUBLIC_SCHEMA_NAME
            )));
        }
        if cat.relational_catalog.contains_key(&create.table)
            || cat.relational_views.contains_key(&create.table)
            || cat
                .relational_materialized_views
                .contains_key(&create.table)
            || cat.relational_sequences.contains_key(&create.table)
            || cat.relational_domains.contains_key(&create.table)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                create.table
            )));
        }
        let mut seen = BTreeSet::new();
        for column in &create.columns {
            if !seen.insert(column.name.clone()) {
                return Err(EngineError::ApplyFailed(format!(
                    "column \"{}\" specified more than once",
                    column.name
                )));
            }
        }
        let oid = cat.relational_next_oid;
        let next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational table OID allocation exhausted".to_string())
        })?;
        let implicit_sequences = create
            .columns
            .iter()
            .filter_map(|column| match &column.default {
                Some(ColumnDefault::SequenceNextVal {
                    sequence,
                    create_if_missing: true,
                }) => Some(sequence.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        for sequence in &implicit_sequences {
            self.preflight_implicit_sequence_name(sequence)?;
        }
        let mut columns = Vec::with_capacity(create.columns.len());
        let mut next_column_id = cat.relational_next_column_id;
        for (idx, mut column) in create.columns.into_iter().enumerate() {
            let (type_oid, type_size) = self.resolve_column_domain_type(&mut column)?;
            if let Some(default) = column.default.take() {
                // Coerce a cross-type default literal to the column type (parity with INSERT),
                // e.g. `bal NUMERIC DEFAULT 0` -> Numeric at the column scale.
                let default = coerce_column_default(default, column.ty, &column.name)?;
                self.preflight_column_default_target(&default)?;
                column.default = Some(default);
            }
            let attnum = i16::try_from(idx + 1).map_err(|_| {
                EngineError::ApplyFailed("too many columns for bootstrap catalog".to_string())
            })?;
            let id = next_column_id;
            next_column_id = next_column_id.checked_add(1).ok_or_else(|| {
                EngineError::ApplyFailed("relational column id allocation exhausted".to_string())
            })?;
            columns.push(RelationalColumn::from_def(
                id, oid, attnum, column, type_oid, type_size,
            ));
        }
        let primary_key = create.primary_key.clone();
        let unique_constraints = create.unique_constraints.clone();
        let check_constraints = create.check_constraints.clone();
        let name = create.table;
        let mut indexes = Vec::new();
        let mut checks = Vec::new();
        // TYPE-COVERAGE #14 Track 3: compound PK/UNIQUE over SUPPORTED key-column types (i32-section
        // Int4/Date/Int2 + i64-section Int8/Timestamp) is device-native (each column's i32-word
        // decomposition folds into a surrogate fingerprint — see `compound_key_fingerprint` /
        // `sql_value_key_words`). A compound key touching any UNSUPPORTED type stays REJECTED (honest
        // partial coverage; b128/text are follow-ups), so nothing silently-wrong ships.
        let compound_key_ok = |cols: &[String]| -> bool {
            cols.iter().all(|name| {
                columns
                    .iter()
                    .find(|c| &c.name == name)
                    .is_some_and(|c| crate::engine_residency::compound_key_type_supported(c.ty))
            })
        };
        if primary_key
            .as_ref()
            .is_some_and(|pk| pk.columns.len() > 1 && !compound_key_ok(&pk.columns))
            || unique_constraints
                .iter()
                .any(|u| u.columns.len() > 1 && !compound_key_ok(&u.columns))
        {
            return Err(EngineError::ApplyFailed(
                "compound PRIMARY KEY / UNIQUE constraints are not yet supported \
                 (compound key columns must be int4, int2, date, int8, timestamp, numeric, uuid, or text)"
                    .to_string(),
            ));
        }
        if let Some(primary_key) = primary_key {
            let constraint_name = primary_key.name.unwrap_or_else(|| format!("{}_pkey", name));
            indexes.push(RelationalIndex {
                name: constraint_name,
                table: name.clone(),
                column: primary_key.column,
                key_columns: primary_key.columns,
                unique: true,
                primary_key: true,
                unique_constraint: false,
            });
        }
        for unique in unique_constraints {
            let constraint_name = unique
                .name
                .unwrap_or_else(|| format!("{}_{}_key", name, unique.column));
            if indexes.iter().any(|index| index.name == constraint_name) {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" already exists",
                    constraint_name
                )));
            }
            indexes.push(RelationalIndex {
                name: constraint_name,
                table: name.clone(),
                column: unique.column,
                key_columns: unique.columns,
                unique: true,
                primary_key: false,
                unique_constraint: true,
            });
        }
        for check in check_constraints {
            let Some(column) = columns
                .iter()
                .find(|column| column.name == check.filter.column)
            else {
                return Err(EngineError::ApplyFailed(format!(
                    "column \"{}\" does not exist",
                    check.filter.column
                )));
            };
            if !sql_value_matches_type(&check.filter.value, column.ty) {
                return Err(EngineError::ApplyFailed(format!(
                    "invalid value for column \"{}\"",
                    check.filter.column
                )));
            }
            let constraint_name = check
                .name
                .unwrap_or_else(|| format!("{}_{}_check", name, check.filter.column));
            if indexes.iter().any(|index| index.name == constraint_name)
                || checks
                    .iter()
                    .any(|candidate: &RelationalCheckConstraint| candidate.name == constraint_name)
            {
                return Err(EngineError::ApplyFailed(format!(
                    "constraint \"{}\" already exists",
                    constraint_name
                )));
            }
            checks.push(RelationalCheckConstraint {
                name: constraint_name,
                column: check.filter.column,
                op: check.filter.op,
                value: check.filter.value,
            });
        }
        cat.relational_next_oid = next_oid;
        for sequence in &implicit_sequences {
            self.create_implicit_sequence(cat, sequence)?;
        }
        let next_oid = cat.relational_next_oid.max(next_oid);
        cat.relational_catalog.insert(
            name.clone(),
            RelationalTable {
                schema: PUBLIC_SCHEMA_NAME.to_string(),
                name,
                oid,
                columns,
                indexes,
                check_constraints: checks,
                foreign_keys: Vec::new(),
                acl: cat.relational_default_table_acl.clone(),
            },
        );
        cat.relational_next_oid = next_oid;
        cat.relational_next_column_id = next_column_id;
        Ok(())
    }

    pub(crate) fn apply_add_primary_key(
        &self,
        cat: &mut DdlCatalogState,
        add: gpu_db_sql::AddPrimaryKey,
    ) -> Result<(), EngineError> {
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
        if cat
            .relational_catalog
            .get(&add.table)
            .is_some_and(|table| table.indexes.iter().any(|index| index.primary_key))
        {
            return Err(EngineError::ApplyFailed(format!(
                "multiple primary keys for table \"{}\" are not allowed",
                add.table
            )));
        }
        let create = CreateIndex {
            name: add.name,
            table: add.table,
            column: add.column,
            columns: add.columns,
            unique: true,
        };
        self.apply_create_index_with_constraint_flags(cat, create, true, false)
    }

    pub(crate) fn apply_create_index(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateIndex,
    ) -> Result<(), EngineError> {
        self.apply_create_index_with_constraint_flags(cat, create, false, false)
    }

    fn apply_create_index_with_constraint_flags(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateIndex,
        primary_key: bool,
        unique_constraint: bool,
    ) -> Result<(), EngineError> {
        if create.columns.is_empty()
            || create.columns.len() > 32
            || create.columns.first() != Some(&create.column)
        {
            return Err(EngineError::ApplyFailed(
                "indexes require between 1 and 32 ordered key columns".to_string(),
            ));
        }
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
        let table = cat
            .relational_catalog
            .get(&create.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", create.table))
            })?
            .clone();
        // COMPOUND KEYS (TYPE-COVERAGE #14 Track 3): resolve EVERY key column (single-column keys
        // resolve `[column_idx]`). A compound UNIQUE/PK over i32-SECTION columns (Int4/Date/Int2) is
        // device-native (folded to a fingerprint surrogate); a compound key touching any wider type
        // stays REJECTED (honest partial coverage). `create.columns` is the ordered key-column list
        // the parser/catalog populate (== `[create.column]` for a single-column key).
        let Some(column_idxs) = create
            .columns
            .iter()
            .map(|name| table.columns.iter().position(|column| &column.name == name))
            .collect::<Option<Vec<usize>>>()
        else {
            let missing = create
                .columns
                .iter()
                .find(|name| !table.columns.iter().any(|column| &column.name == *name))
                .cloned()
                .unwrap_or_else(|| create.column.clone());
            return Err(EngineError::ApplyFailed(format!(
                "column \"{missing}\" does not exist"
            )));
        };
        if create.unique && column_idxs.len() > 1 {
            let all_supported = column_idxs.iter().all(|&i| {
                crate::engine_residency::compound_key_type_supported(table.columns[i].ty)
            });
            if !all_supported {
                return Err(EngineError::ApplyFailed(
                    "compound PRIMARY KEY / UNIQUE constraints are not yet supported \
                     (compound key columns must be int4, int2, date, int8, timestamp, numeric, uuid, or text)"
                        .to_string(),
                ));
            }
        }
        if create.unique {
            let visibility = StorageVisibility {
                read_txn_id: self.committed_seq() as TxnId,
            };
            let rows = self.visible_relational_rows(&table, visibility)?;
            // PG: ADD PRIMARY KEY over existing data requires EVERY key column non-null (23502) — the
            // creation-time half of the PK NOT NULL invariant the DML validators rely on.
            if primary_key {
                if let Some(&null_col) = column_idxs
                    .iter()
                    .find(|&&i| rows.iter().any(|row| matches!(row[i], SqlValue::Null)))
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" of relation \"{}\" contains null values",
                        table.columns[null_col].name, create.table
                    )));
                }
            }
            Self::validate_unique_values_tuple(&rows, &column_idxs, &create.name)?;
        }
        cat.relational_catalog
            .get_mut(&create.table)
            .expect("table existence validated")
            .indexes
            .push(RelationalIndex {
                name: create.name,
                table: create.table,
                column: create.column,
                key_columns: create.columns,
                unique: create.unique,
                primary_key,
                unique_constraint,
            });
        Ok(())
    }

    pub(crate) fn apply_add_unique_constraint(
        &self,
        cat: &mut DdlCatalogState,
        add: AddUniqueConstraint,
    ) -> Result<(), EngineError> {
        let create = CreateIndex {
            name: add.name,
            table: add.table,
            column: add.column,
            columns: add.columns,
            unique: true,
        };
        self.apply_create_index_with_constraint_flags(cat, create, false, true)
    }

    pub(crate) fn apply_add_check_constraint(
        &self,
        cat: &mut DdlCatalogState,
        add: AddCheckConstraint,
    ) -> Result<(), EngineError> {
        self.preflight_add_check_constraint(&add)?;
        let table = cat
            .relational_catalog
            .get_mut(&add.table)
            .expect("table existence preflighted");
        table.check_constraints.push(RelationalCheckConstraint {
            name: add.name,
            column: add.filter.column,
            op: add.filter.op,
            value: add.filter.value,
        });
        Ok(())
    }

    pub(crate) fn apply_add_foreign_key(
        &self,
        cat: &mut DdlCatalogState,
        add: AddForeignKey,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        self.preflight_add_foreign_key(&add, txn_id)?;
        let table = cat
            .relational_catalog
            .get_mut(&add.table)
            .expect("table existence preflighted");
        table.foreign_keys.push(RelationalForeignKey {
            name: add.name,
            column: add.column,
            referenced_table: add.referenced_table,
            referenced_column: add.referenced_column,
        });
        Ok(())
    }

    pub(crate) fn apply_drop_constraint(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropConstraint,
    ) -> Result<(), EngineError> {
        let Some(table) = cat.relational_catalog.get_mut(&drop.table) else {
            if drop.table_if_exists {
                return Ok(());
            }
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" does not exist",
                drop.table
            )));
        };
        let old_index_len = table.indexes.len();
        table.indexes.retain(|index| {
            !(index.name == drop.name && (index.primary_key || index.unique_constraint))
        });
        let old_check_len = table.check_constraints.len();
        table
            .check_constraints
            .retain(|constraint| constraint.name != drop.name);
        let old_foreign_key_len = table.foreign_keys.len();
        table
            .foreign_keys
            .retain(|constraint| constraint.name != drop.name);
        if table.indexes.len() == old_index_len
            && table.check_constraints.len() == old_check_len
            && table.foreign_keys.len() == old_foreign_key_len
        {
            if drop.if_exists {
                return Ok(());
            }
            return Err(EngineError::ApplyFailed(format!(
                "constraint \"{}\" does not exist",
                drop.name
            )));
        }
        cat.relational_comments
            .remove(&RelationalCommentTarget::Index {
                index: drop.name.clone(),
            });
        cat.relational_comments
            .remove(&RelationalCommentTarget::Constraint {
                table: drop.table,
                constraint: drop.name,
            });
        Ok(())
    }

    pub(crate) fn apply_rename_constraint(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameConstraint,
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
        if !cat.relational_catalog.contains_key(&rename.table) {
            if rename.table_if_exists {
                return Ok(());
            }
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" does not exist",
                rename.table
            )));
        }
        if cat.relational_catalog.values().any(|candidate| {
            candidate
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
        let table = cat
            .relational_catalog
            .get_mut(&rename.table)
            .expect("table existence validated");
        if let Some(index) = table.indexes.iter_mut().find(|index| {
            index.name == rename.old_name && (index.primary_key || index.unique_constraint)
        }) {
            index.name = rename.new_name.clone();
        } else if let Some(check) = table
            .check_constraints
            .iter_mut()
            .find(|constraint| constraint.name == rename.old_name)
        {
            check.name = rename.new_name.clone();
        } else if let Some(foreign_key) = table
            .foreign_keys
            .iter_mut()
            .find(|constraint| constraint.name == rename.old_name)
        {
            foreign_key.name = rename.new_name.clone();
        } else {
            return Err(EngineError::ApplyFailed(format!(
                "constraint \"{}\" does not exist",
                rename.old_name
            )));
        };

        let old_index_target = RelationalCommentTarget::Index {
            index: rename.old_name.clone(),
        };
        if let Some(comment) = cat.relational_comments.remove(&old_index_target) {
            cat.relational_comments.insert(
                RelationalCommentTarget::Index {
                    index: rename.new_name.clone(),
                },
                comment,
            );
        }
        let old_constraint_target = RelationalCommentTarget::Constraint {
            table: rename.table.clone(),
            constraint: rename.old_name,
        };
        if let Some(comment) = cat.relational_comments.remove(&old_constraint_target) {
            cat.relational_comments.insert(
                RelationalCommentTarget::Constraint {
                    table: rename.table,
                    constraint: rename.new_name,
                },
                comment,
            );
        }
        Ok(())
    }

    pub(crate) fn visible_relational_rows(
        &self,
        table: &RelationalTable,
        visibility: StorageVisibility,
    ) -> Result<Vec<Vec<SqlValue>>, EngineError> {
        // CONSTRAINED-ELISION SEAM (audit-B1 closure at the SOURCE): on an ELIDED table the host
        // store is a stale prefix — a seq_scan here would materialize MISSING elided-era rows
        // (DDL row-validators would silently pass over data that violates the new constraint;
        // the flag-off scan validators would bypass unique/FK checks). Rehydrate FIRST, whatever
        // the caller: this fn is elision-safe by construction, not by caller discipline.
        let mut visibility = visibility;
        if self.table_device_authoritative(&table.name) {
            if self.current_transaction_read_snapshot().is_some() {
                return Err(EngineError::ApplyFailed(format!(
                    "transaction-generation DML validation for \"{}\" declined its retained \
                     device source; refusing to rehydrate or read a newer host generation",
                    table.name
                )));
            }
            self.rehydrate_elided_serialized(&table.name)?;
            // Re-audit SHOULD-FIX (the FINDING-C class, scan-arm side): the reconcile stamps
            // every elided-era row at committed_seq; callers reading at a FACADE txn id below
            // it (the FK preflight, flag-off validators) would miss them all — the un-raised
            // sibling of the index arm's probe boundary. Raise identically.
            visibility.read_txn_id = visibility.read_txn_id.max(self.committed_seq());
        }
        let prefix = relational_key_prefix(&table.name);
        let mut rows = Vec::new();
        // Load this table's published MVCC generation; the cursor reads its immutable rows
        // lock-free (the prefix filter is redundant now each partition is single-table, but kept
        // so the read stays correct regardless of partition contents — write-half Stage 3).
        let table_rows = self.read_table_rows_at(&table.name, visibility.read_txn_id);
        let mut cursor = table_rows
            .store()
            .seq_scan_open(visibility)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        while let Some(tuple) = cursor.next() {
            if tuple.key.starts_with(&prefix) {
                rows.push(
                    decode_relational_row(&tuple.value, &table.columns)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?,
                );
            }
        }
        Ok(rows)
    }

    /// Re-audit hardening (2330965b item-2 NOTE): the visibility for a DDL applier that scans a
    /// table's store DIRECTLY (add-column rewrite, drop-column rewrite, rename-table move —
    /// the row-REWRITING appliers, where a missed row is silent data loss). Rehydrates an
    /// elided table first (stale prefix), then reads at `max(txn_id, committed_seq)`: DDL
    /// applies run under the commit lock with every commit <= committed_seq fully applied and
    /// must see ALL of them (a rename moves every row), while facade txn ids are DECOUPLED
    /// from commit seqs and can sit below — and an elision rehydration re-stamps rows at
    /// committed_seq. Row-DISCARDING appliers (DROP/TRUNCATE) deliberately skip this: a stale
    /// prefix only shrinks the set of tuples to delete, and the device invalidate + elided-flag
    /// purge make the end state correct regardless.
    pub(crate) fn ddl_rewrite_scan_visibility(
        &self,
        table: &str,
        txn_id: TxnId,
    ) -> Result<StorageVisibility, EngineError> {
        if self.table_device_authoritative(table) {
            self.rehydrate_elided_serialized(table)?;
        }
        Ok(StorageVisibility {
            read_txn_id: txn_id.max(self.committed_seq()),
        })
    }

    /// PG: PRIMARY KEY implies NOT NULL — a NULL may never enter a PK column (PG rejects it with a
    /// 23502 not-null violation BEFORE the unique check; previously this engine admitted exactly one
    /// NULL per PK because the PK was validated only as a unique index whose BTreeSet collides NULLs).
    /// `rows` = the NEW images only — existing rows are guaranteed by creation-time validation
    /// (`apply_create_index_with_constraint_flags` rejects a PK over null-bearing data), exactly PG's
    /// model. Checked in BOTH validator arms (scan + index-driven) with this one byte-identical
    /// message; a table can carry at most one PK (`apply_add_primary_key` enforces it), so this is a
    /// single pass over the new images.
    pub(crate) fn validate_primary_key_not_null<'a>(
        table: &RelationalTable,
        rows: impl IntoIterator<Item = &'a [SqlValue]>,
    ) -> Result<(), EngineError> {
        let Some(column_idxs) = table
            .indexes
            .iter()
            .find(|index| index.primary_key)
            .and_then(|index| crate::engine_residency::index_key_column_positions(table, index))
        else {
            return Ok(());
        };
        for row in rows {
            for &column_idx in &column_idxs {
                if matches!(row[column_idx], SqlValue::Null) {
                    return Err(EngineError::ApplyFailed(format!(
                        "null value in column \"{}\" of relation \"{}\" violates not-null constraint",
                        table.columns[column_idx].name, table.name
                    )));
                }
            }
        }
        Ok(())
    }

    /// COMPOUND KEYS (TYPE-COVERAGE #14 Track 3): a duplicate is a repeated ORDERED TUPLE of the key
    /// columns (`column_idxs.len() == 1` reproduces the single-column unique check exactly).
    /// PostgreSQL unique semantics: any tuple containing NULL is not comparable for uniqueness, so
    /// multiple such rows are accepted. PRIMARY KEY nullability is rejected separately above.
    pub(crate) fn validate_unique_values_tuple(
        rows: &[Vec<SqlValue>],
        column_idxs: &[usize],
        index_name: &str,
    ) -> Result<(), EngineError> {
        let mut seen: BTreeSet<Vec<SqlValue>> = BTreeSet::new();
        for row in rows {
            let key: Vec<SqlValue> = column_idxs.iter().map(|&i| row[i].clone()).collect();
            if key.iter().any(|value| matches!(value, SqlValue::Null)) {
                continue;
            }
            if !seen.insert(key) {
                return Err(EngineError::ApplyFailed(format!(
                    "duplicate key value violates unique index \"{}\"",
                    index_name
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn validate_check_constraints_for_rows(
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
    ) -> Result<(), EngineError> {
        for constraint in &table.check_constraints {
            let column_idx = relational_column_index(table, &constraint.column)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            for row in rows {
                // PG 3VL: a CHECK is violated only when the predicate evaluates to FALSE — a NULL
                // operand makes it UNKNOWN, which SATISFIES the constraint (PostgreSQL: "the check
                // expression should ... yield true or the null value"). `select_filter_matches`
                // returns false for a NULL operand (WHERE semantics: exclude), which here would
                // wrongly treat UNKNOWN as a violation — so NULL passes explicitly.
                if matches!(row[column_idx], SqlValue::Null) {
                    continue;
                }
                if !select_filter_matches(&row[column_idx], constraint.op, &constraint.value) {
                    return Err(EngineError::ApplyFailed(format!(
                        "new row for relation \"{}\" violates check constraint \"{}\"",
                        table.name, constraint.name
                    )));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn validate_foreign_keys_with_table_rows(
        &self,
        changed_table: &str,
        changed_rows: &[Vec<SqlValue>],
        visibility: StorageVisibility,
    ) -> Result<(), EngineError> {
        // Lock-free concurrent-DML path (Stage 2 — blocker #1): pin ONE catalog snapshot for the whole
        // cross-table FK scan so a concurrent DDL cannot change FK definitions mid-validation.
        let catalog = self.catalog_snapshot();
        for child_table in catalog.relational_catalog.values() {
            let child_rows = if child_table.name == changed_table {
                changed_rows.to_vec()
            } else {
                self.visible_relational_rows(child_table, visibility)?
            };
            for foreign_key in &child_table.foreign_keys {
                let Some(parent_table) = catalog
                    .relational_catalog
                    .get(&foreign_key.referenced_table)
                else {
                    continue;
                };
                let parent_rows = if parent_table.name == changed_table {
                    changed_rows.to_vec()
                } else {
                    self.visible_relational_rows(parent_table, visibility)?
                };
                Self::validate_foreign_key_rows(
                    child_table,
                    &child_rows,
                    parent_table,
                    &parent_rows,
                    foreign_key,
                )?;
            }
        }
        Ok(())
    }

    fn validate_foreign_key_rows(
        child_table: &RelationalTable,
        child_rows: &[Vec<SqlValue>],
        parent_table: &RelationalTable,
        parent_rows: &[Vec<SqlValue>],
        foreign_key: &RelationalForeignKey,
    ) -> Result<(), EngineError> {
        let child_column_idx = relational_column_index(child_table, &foreign_key.column)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let parent_column_idx =
            relational_column_index(parent_table, &foreign_key.referenced_column)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let parent_values = parent_rows
            .iter()
            .map(|row| row[parent_column_idx].clone())
            .collect::<BTreeSet<_>>();
        for row in child_rows {
            // PG 3VL (MATCH SIMPLE): a NULL foreign-key value SATISFIES the constraint — it
            // references nothing, so no provider is required (and a NULL in the parent's unique
            // column is never a provider). Same rule as the CHECK-on-NULL fix.
            if matches!(row[child_column_idx], SqlValue::Null) {
                continue;
            }
            if !parent_values.contains(&row[child_column_idx]) {
                return Err(EngineError::ApplyFailed(format!(
                    "insert or update on table \"{}\" violates foreign key constraint \"{}\"",
                    child_table.name, foreign_key.name
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn preflight_add_check_constraint(
        &self,
        add: &AddCheckConstraint,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let table = cat.relational_catalog.get(&add.table).ok_or_else(|| {
            EngineError::ApplyFailed(format!("relation \"{}\" does not exist", add.table))
        })?;
        if cat.relational_catalog.values().any(|candidate| {
            candidate.indexes.iter().any(|index| index.name == add.name)
                || candidate
                    .check_constraints
                    .iter()
                    .any(|constraint| constraint.name == add.name)
                || candidate
                    .foreign_keys
                    .iter()
                    .any(|constraint| constraint.name == add.name)
        }) || cat.relational_catalog.contains_key(&add.name)
            || cat.relational_views.contains_key(&add.name)
            || cat.relational_materialized_views.contains_key(&add.name)
            || cat.relational_sequences.contains_key(&add.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "constraint \"{}\" already exists",
                add.name
            )));
        }
        let column_idx = relational_column_index(table, &add.filter.column)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        if !sql_value_matches_type(&add.filter.value, table.columns[column_idx].ty) {
            return Err(EngineError::ApplyFailed(format!(
                "invalid value for column \"{}\"",
                add.filter.column
            )));
        }
        let rows = self.visible_relational_rows(
            table,
            StorageVisibility {
                read_txn_id: self.committed_seq() as TxnId,
            },
        )?;
        for row in rows {
            // PG 3VL (same rule as `validate_check_constraints_for_rows`): an existing NULL value
            // makes the check UNKNOWN, which SATISFIES it — ADD CHECK must not reject over NULLs.
            if matches!(row[column_idx], SqlValue::Null) {
                continue;
            }
            if !select_filter_matches(&row[column_idx], add.filter.op, &add.filter.value) {
                return Err(EngineError::ApplyFailed(format!(
                    "check constraint \"{}\" is violated by some row",
                    add.name
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn preflight_add_foreign_key(
        &self,
        add: &AddForeignKey,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if add.table == add.referenced_table {
            return Err(EngineError::ApplyFailed(
                "self-referential foreign keys are not supported".to_string(),
            ));
        }
        let table = cat.relational_catalog.get(&add.table).ok_or_else(|| {
            EngineError::ApplyFailed(format!("relation \"{}\" does not exist", add.table))
        })?;
        let referenced_table = cat
            .relational_catalog
            .get(&add.referenced_table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "relation \"{}\" does not exist",
                    add.referenced_table
                ))
            })?;
        if cat.relational_catalog.values().any(|candidate| {
            candidate.indexes.iter().any(|index| index.name == add.name)
                || candidate
                    .check_constraints
                    .iter()
                    .any(|constraint| constraint.name == add.name)
                || candidate
                    .foreign_keys
                    .iter()
                    .any(|constraint| constraint.name == add.name)
        }) || cat.relational_catalog.contains_key(&add.name)
            || cat.relational_views.contains_key(&add.name)
            || cat.relational_materialized_views.contains_key(&add.name)
            || cat.relational_sequences.contains_key(&add.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "constraint \"{}\" already exists",
                add.name
            )));
        }
        let column_idx = relational_column_index(table, &add.column)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let referenced_column_idx =
            relational_column_index(referenced_table, &add.referenced_column)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        if table.columns[column_idx].ty != referenced_table.columns[referenced_column_idx].ty {
            return Err(EngineError::ApplyFailed(
                "foreign key column type does not match referenced column type".to_string(),
            ));
        }
        // COMPOUND KEYS: a single-column FK matches only a SINGLE-COLUMN unique/primary key — the
        // first column of a compound key is NOT independently unique, so it must not satisfy the FK.
        let has_referenced_unique_key = referenced_table.indexes.iter().any(|index| {
            index.key_columns.len() == 1
                && index.column == add.referenced_column
                && (index.primary_key || index.unique_constraint)
        });
        if !has_referenced_unique_key {
            return Err(EngineError::ApplyFailed(format!(
                "there is no unique constraint matching given keys for referenced table \"{}\"",
                add.referenced_table
            )));
        }
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };
        let child_rows = self.visible_relational_rows(table, visibility)?;
        let parent_rows = self.visible_relational_rows(referenced_table, visibility)?;
        Self::validate_foreign_key_rows(
            table,
            &child_rows,
            referenced_table,
            &parent_rows,
            &RelationalForeignKey {
                name: add.name.clone(),
                column: add.column.clone(),
                referenced_table: add.referenced_table.clone(),
                referenced_column: add.referenced_column.clone(),
            },
        )?;
        Ok(())
    }

    pub(crate) fn apply_drop_index(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropIndex,
    ) -> Result<(), EngineError> {
        if !drop.if_exists {
            for name in &drop.names {
                if !cat
                    .relational_catalog
                    .values()
                    .any(|table| table.indexes.iter().any(|index| index.name == *name))
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "index \"{}\" does not exist",
                        name
                    )));
                }
            }
        }
        let drop_names = drop.names.iter().cloned().collect::<BTreeSet<_>>();
        let mut dropped_constraints = Vec::new();
        for table in cat.relational_catalog.values_mut() {
            dropped_constraints.extend(
                table
                    .indexes
                    .iter()
                    .filter(|index| {
                        drop_names.contains(&index.name)
                            && (index.primary_key || index.unique_constraint)
                    })
                    .map(|index| (index.table.clone(), index.name.clone())),
            );
            table
                .indexes
                .retain(|index| !drop_names.contains(&index.name));
        }
        for name in &drop.names {
            cat.relational_comments
                .remove(&RelationalCommentTarget::Index {
                    index: name.clone(),
                });
        }
        for (table, constraint) in dropped_constraints {
            cat.relational_comments
                .remove(&RelationalCommentTarget::Constraint { table, constraint });
        }
        Ok(())
    }

    pub(crate) fn preflight_drop_index(&self, drop: &DropIndex) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if drop.if_exists {
            return Ok(());
        }
        for name in &drop.names {
            if !cat
                .relational_catalog
                .values()
                .any(|table| table.indexes.iter().any(|index| index.name == *name))
            {
                return Err(EngineError::ApplyFailed(format!(
                    "index \"{}\" does not exist",
                    name
                )));
            }
        }
        let drop_names = drop.names.iter().cloned().collect::<BTreeSet<_>>();
        if cat.relational_catalog.values().any(|table| {
            table.foreign_keys.iter().any(|constraint| {
                drop_names.contains(&table.name)
                    || drop_names.contains(&constraint.referenced_table)
            })
        }) {
            return Err(EngineError::ApplyFailed(
                "cannot drop table because a foreign key constraint depends on it".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn apply_rename_index(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameIndex,
    ) -> Result<(), EngineError> {
        if cat.relational_catalog.values().any(|table| {
            table
                .indexes
                .iter()
                .any(|index| index.name == rename.new_name)
        }) {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                rename.new_name
            )));
        }

        for table in cat.relational_catalog.values_mut() {
            let Some(index) = table
                .indexes
                .iter_mut()
                .find(|index| index.name == rename.old_name)
            else {
                continue;
            };
            if index.primary_key || index.unique_constraint {
                return Err(EngineError::ApplyFailed(format!(
                    "cannot rename constraint-backed index \"{}\" with ALTER INDEX",
                    rename.old_name
                )));
            }
            index.name = rename.new_name.clone();
            let old_target = RelationalCommentTarget::Index {
                index: rename.old_name,
            };
            if let Some(comment) = cat.relational_comments.remove(&old_target) {
                cat.relational_comments.insert(
                    RelationalCommentTarget::Index {
                        index: rename.new_name,
                    },
                    comment,
                );
            }
            return Ok(());
        }

        Err(EngineError::ApplyFailed(format!(
            "index \"{}\" does not exist",
            rename.old_name
        )))
    }

    pub(crate) fn apply_rename_table(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameTable,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
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
        let Some(mut table) = cat.relational_catalog.remove(&rename.old_name) else {
            return Ok(());
        };

        let old_prefix = relational_key_prefix(&rename.old_name);
        // Re-audit hardening: the rename MOVES every row — a row-REWRITING scan (see
        // `ddl_rewrite_scan_visibility`): an elided stale prefix or a facade-below-committed
        // boundary would silently strand rows in the old partition.
        let visibility = self.ddl_rewrite_scan_visibility(&rename.old_name, txn_id)?;
        // The moved table starts NON-elided under its new name, and the OLD name's flag must
        // not linger (the SF4 drop-purge discipline): an orphaned entry would mislabel a
        // future same-name table as device-authoritative.
        self.set_table_device_authoritative(&rename.old_name, false);
        self.set_table_device_authoritative(&rename.new_name, false);
        // Read the rows to move out of the OLD partition's published generation.
        let mut moves = Vec::new();
        {
            let old_rows = self.read_state.mvcc.table_rows(&rename.old_name);
            let mut cursor = old_rows
                .store()
                .seq_scan_open(visibility)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            while let Some(tuple) = cursor.next() {
                if tuple.key.starts_with(&old_prefix) {
                    moves.push((tuple.tuple_id, tuple.value.clone()));
                }
            }
        }

        // Build the moved rows + their value-index for the NEW table (decoded against the renamed
        // table's columns), reserving fresh global tuple ids and relational row ids — identical to
        // the old in-line per-row bumps.
        table.name = rename.new_name.clone();
        for index in &mut table.indexes {
            index.table = rename.new_name.clone();
        }
        for foreign_key in &mut table.foreign_keys {
            if foreign_key.referenced_table == rename.old_name {
                foreign_key.referenced_table = rename.new_name.clone();
            }
        }
        let mut new_rows: Vec<(TupleId, String, String)> = Vec::with_capacity(moves.len());
        let mut new_value_index: BTreeMap<ColumnValueKey, Vec<String>> = BTreeMap::new();
        let old_tuple_ids: Vec<TupleId> = moves.iter().map(|(tuple_id, _)| *tuple_id).collect();
        for (_old_tuple_id, value) in &moves {
            let row_id = self.read_state.mvcc.current_row_id();
            self.read_state.mvcc.advance_row_id(1);
            let new_key = relational_row_key(&rename.new_name, row_id);
            let new_tuple_id = self.read_state.mvcc.reserve_tuple_id();
            let values = decode_relational_row(value, &table.columns)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            for (column, column_value) in table.columns.iter().zip(values.iter()) {
                new_value_index
                    .entry(ColumnValueKey {
                        column: column.name.clone(),
                        value: relational_index_value(column_value),
                    })
                    .or_default()
                    .push(new_key.clone());
            }
            new_rows.push((new_tuple_id, new_key, value.clone()));
        }

        // Publish the NEW table partition with the moved rows + rebuilt value-index.
        self.read_state
            .mvcc
            .with_table_mut(&rename.new_name, |data| {
                for (tuple_id, new_key, value) in &new_rows {
                    data.rows
                        .tuple_insert_with_id(
                            *tuple_id,
                            NewTuple {
                                key: new_key.clone(),
                                value: value.clone(),
                            },
                            txn_id,
                        )
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                }
                for (key, keys) in &new_value_index {
                    let mut slot = data.value_index.get(key).cloned().unwrap_or_default();
                    slot.extend(keys.iter().cloned());
                    data.value_index.insert(key.clone(), slot);
                }
                Ok::<(), EngineError>(())
            })?;

        // Tombstone the moved rows in the OLD partition (keeping their history, exactly as the
        // old in-line `tuple_delete` did) and clear the old partition's value-index.
        self.read_state
            .mvcc
            .with_table_mut(&rename.old_name, |data| {
                for old_tuple_id in &old_tuple_ids {
                    data.rows
                        .tuple_delete(*old_tuple_id, txn_id)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                }
                data.value_index = imbl::OrdMap::new();
                Ok::<(), EngineError>(())
            })?;

        cat.relational_catalog
            .insert(rename.new_name.clone(), table.clone());
        for candidate in cat.relational_catalog.values_mut() {
            for foreign_key in &mut candidate.foreign_keys {
                if foreign_key.referenced_table == rename.old_name {
                    foreign_key.referenced_table = rename.new_name.clone();
                }
            }
        }

        let mut retargeted_comments = Vec::new();
        cat.relational_comments
            .retain(|target, comment| match target {
                RelationalCommentTarget::Table { table } if table == &rename.old_name => {
                    retargeted_comments.push((
                        RelationalCommentTarget::Table {
                            table: rename.new_name.clone(),
                        },
                        comment.clone(),
                    ));
                    false
                }
                RelationalCommentTarget::Column { table, attnum } if table == &rename.old_name => {
                    retargeted_comments.push((
                        RelationalCommentTarget::Column {
                            table: rename.new_name.clone(),
                            attnum: *attnum,
                        },
                        comment.clone(),
                    ));
                    false
                }
                RelationalCommentTarget::Constraint { table, constraint }
                    if table == &rename.old_name =>
                {
                    retargeted_comments.push((
                        RelationalCommentTarget::Constraint {
                            table: rename.new_name.clone(),
                            constraint: constraint.clone(),
                        },
                        comment.clone(),
                    ));
                    false
                }
                _ => true,
            });
        for (target, comment) in retargeted_comments {
            cat.relational_comments.insert(target, comment);
        }
        self.read_state
            .residency
            .with_snapshots_mut(|snapshots| snapshots.remove(&rename.old_name));
        self.read_state
            .residency
            .device_memory
            .remove(&rename.old_name);
        Ok(())
    }

    pub(crate) fn apply_drop_table(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropTable,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        self.preflight_drop_table(&drop)?;

        for name in &drop.names {
            // RETIREMENT A4e (audit SF4): a dropped table must LEAVE the elided set — a later
            // CREATE reusing the name would otherwise skip its first installs against a
            // non-authoritative device (divergence). Mirrors the shard-region purge discipline.
            self.set_table_device_authoritative(name, false);
            let Some(table) = cat.relational_catalog.remove(name) else {
                continue;
            };
            let prefix = relational_key_prefix(&table.name);
            let visibility = StorageVisibility {
                read_txn_id: txn_id,
            };
            let mut tuple_ids = Vec::new();
            {
                let table_rows = self.read_state.mvcc.table_rows(name);
                let mut cursor = table_rows
                    .store()
                    .seq_scan_open(visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                while let Some(tuple) = cursor.next() {
                    if tuple.key.starts_with(&prefix) {
                        tuple_ids.push(tuple.tuple_id);
                    }
                }
            }
            // Tombstone the dropped table's rows in place (keeping their version history, exactly
            // as the old single-store `tuple_delete` did), publishing one new generation. The
            // partition cell + (now-stale) value-index are retained — `all_versions` still sees the
            // tombstoned versions, matching the pre-partition behavior.
            if !tuple_ids.is_empty() {
                self.read_state.mvcc.with_table_mut(name, |data| {
                    for tuple_id in &tuple_ids {
                        data.rows
                            .tuple_delete(*tuple_id, txn_id)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    }
                    Ok::<(), EngineError>(())
                })?;
            }

            let index_names = table
                .indexes
                .iter()
                .map(|index| index.name.clone())
                .collect::<BTreeSet<_>>();
            cat.relational_comments.retain(|target, _| match target {
                RelationalCommentTarget::Table { table }
                | RelationalCommentTarget::Column { table, .. }
                | RelationalCommentTarget::Constraint { table, .. } => table != name,
                RelationalCommentTarget::Index { index } => !index_names.contains(index),
                RelationalCommentTarget::Database { .. }
                | RelationalCommentTarget::Role { .. }
                | RelationalCommentTarget::Schema { .. }
                | RelationalCommentTarget::Tablespace { .. }
                | RelationalCommentTarget::View { .. }
                | RelationalCommentTarget::MaterializedView { .. }
                | RelationalCommentTarget::Extension { .. }
                | RelationalCommentTarget::Function { .. }
                | RelationalCommentTarget::Sequence { .. }
                | RelationalCommentTarget::Domain { .. }
                | RelationalCommentTarget::Publication { .. }
                | RelationalCommentTarget::Subscription { .. } => true,
            });
            self.read_state
                .residency
                .with_snapshots_mut(|snapshots| snapshots.remove(name));
            self.read_state.residency.device_memory.remove(name);
            // TYPE-COVERAGE #14 (text/shards): a SHARD-resident table (any elided table — int4/numeric/
            // bool/text) must also drop its SHARDS + shard device memory, else a re-created table of the
            // same name would bind stale shards (wrong results) and the buffers would leak. (Single-buffer
            // tables have no shards, so this is a no-op for them.)
            self.read_state
                .residency
                .with_shards_mut_for_table(name, |shards| shards.remove(name));
            self.read_state
                .residency
                .shard_device_memory
                .remove_table(name);
            // SV4 prereq #1 (lifecycle): ERASE the dropped table's on-demand `deleted_by` region cells
            // (the commit's `invalidate_table` only publishes `None`, freeing the device buffer but leaving
            // a dangling per-shard key). A DROPped table is gone for good, so fully remove its keys to avoid
            // an unbounded host-cell leak across distinct dropped tables. INERT until SV4 (no region today).
            self.read_state
                .residency
                .shard_deleted_by_memory
                .remove_table(name);
            // SV6: the dropped table's `created_by` region cells are erased the same way (same leak guard).
            self.read_state
                .residency
                .shard_created_by_memory
                .remove_table(name);
            self.read_state
                .residency
                .shard_row_id_memory
                .remove_table(name);
            // Sub-slice 3b: a DROPped table's cached per-shard PK indexes are gone for good -> purge them.
            self.read_state
                .residency
                .purge_shard_pk_index_for_table(name);
        }
        Ok(())
    }

    pub(crate) fn preflight_drop_table(&self, drop: &DropTable) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let mut seen = BTreeSet::new();
        for name in &drop.names {
            if !seen.insert(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "table \"{}\" specified more than once",
                    name
                )));
            }
            if cat.relational_views.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" is not a table",
                    name
                )));
            }
            if cat.relational_sequences.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" is not a table",
                    name
                )));
            }
            if !drop.if_exists && !cat.relational_catalog.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" does not exist",
                    name
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn apply_truncate_table(
        &self,
        cat: &mut DdlCatalogState,
        truncate: TruncateTable,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        if cat.relational_views.contains_key(&truncate.name)
            || cat
                .relational_materialized_views
                .contains_key(&truncate.name)
            || cat.relational_sequences.contains_key(&truncate.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a table",
                truncate.name
            )));
        }
        let table = cat
            .relational_catalog
            .get(&truncate.name)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", truncate.name))
            })?
            .clone();
        if cat.relational_catalog.values().any(|candidate| {
            candidate
                .foreign_keys
                .iter()
                .any(|foreign_key| foreign_key.referenced_table == table.name)
        }) {
            self.validate_foreign_keys_with_table_rows(
                &table.name,
                &[],
                StorageVisibility {
                    read_txn_id: txn_id,
                },
            )?;
        }
        let restart_sequences = if truncate.restart_identity {
            table
                .columns
                .iter()
                .filter_map(|column| match &column.default {
                    Some(ColumnDefault::SequenceNextVal { sequence, .. }) => Some(sequence.clone()),
                    _ => None,
                })
                .collect::<BTreeSet<_>>()
        } else {
            BTreeSet::new()
        };
        for sequence in &restart_sequences {
            if !cat.relational_sequences.contains_key(sequence) {
                return Err(EngineError::ApplyFailed(format!(
                    "sequence \"{sequence}\" does not exist"
                )));
            }
        }

        let prefix = relational_key_prefix(&table.name);
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };
        let mut tuple_ids = Vec::new();
        {
            let table_rows = self.read_state.mvcc.table_rows(&truncate.name);
            let mut cursor = table_rows
                .store()
                .seq_scan_open(visibility)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            while let Some(tuple) = cursor.next() {
                if tuple.key.starts_with(&prefix) {
                    tuple_ids.push(tuple.tuple_id);
                }
            }
        }
        // Tombstone all rows in place (keeping history), publishing one new generation. As before,
        // the value-index is left as-is; its now-stale entries point to tombstoned rows and are
        // filtered out by visibility + the predicate recheck.
        if !tuple_ids.is_empty() {
            self.read_state
                .mvcc
                .with_table_mut(&truncate.name, |data| {
                    for tuple_id in &tuple_ids {
                        data.rows
                            .tuple_delete(*tuple_id, txn_id)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    }
                    Ok::<(), EngineError>(())
                })?;
        }
        if truncate.restart_identity {
            for sequence in restart_sequences {
                let sequence_state = cat
                    .relational_sequences
                    .get_mut(&sequence)
                    .expect("restart identity sequence preflighted");
                sequence_state.last_value = 1;
                sequence_state.is_called = false;
            }
        }
        Ok(())
    }

    pub(crate) fn apply_drop_view(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropView,
    ) -> Result<(), EngineError> {
        self.preflight_drop_view(&drop)?;
        for name in &drop.names {
            if cat.relational_views.remove(name).is_none() {
                continue;
            }
            cat.relational_comments
                .remove(&RelationalCommentTarget::View { view: name.clone() });
        }
        Ok(())
    }

    pub(crate) fn apply_drop_materialized_view(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropMaterializedView,
    ) -> Result<(), EngineError> {
        self.preflight_drop_materialized_view(&drop)?;
        for name in &drop.names {
            if cat.relational_materialized_views.remove(name).is_none() {
                continue;
            }
            cat.relational_comments
                .remove(&RelationalCommentTarget::MaterializedView {
                    materialized_view: name.clone(),
                });
        }
        Ok(())
    }

    pub(crate) fn apply_drop_sequence(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropSequence,
    ) -> Result<(), EngineError> {
        self.preflight_drop_sequence(&drop)?;
        for name in &drop.names {
            if cat.relational_sequences.remove(name).is_none() {
                continue;
            }
            cat.relational_comments
                .remove(&RelationalCommentTarget::Sequence {
                    sequence: name.clone(),
                });
        }
        Ok(())
    }
}
