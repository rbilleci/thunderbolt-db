//! Catalog / ACL / comment introspection + COPY (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for the catalog-snapshot
//! accessors (relational_catalog_table/view/sequence/function/..., the
//! relational_*_acl readers, and the relational_*_comment readers across every
//! object kind) plus COPY-rows execution (execute_relational_copy_rows[_profiled]).

use super::*;

impl Engine {
    // --- Catalog introspection accessors (test/admin only; no production callers reach these). ---
    // They read the DDL working catalog under the catalog latch and return OWNED clones: a `MutexGuard`
    // cannot lend a borrow that outlives it, so the historical `Option<&T>` borrows became owned values
    // (lock-free read path, write-half MVCC). Behavior is otherwise identical.
    pub fn relational_catalog_table(&self, table: &str) -> Option<RelationalTable> {
        self.ddl_catalog().relational_catalog.get(table).cloned()
    }

    pub fn relational_copy_columns(&self, table: &str) -> Result<Vec<CopyColumn>, EngineError> {
        self.ensure_commit_path_available()?;
        let cat = self.ddl_catalog();
        let table = cat.relational_catalog.get(table).ok_or_else(|| {
            EngineError::ApplyFailed(format!("relation \"{}\" does not exist", table))
        })?;
        Ok(table
            .columns
            .iter()
            .map(|column| CopyColumn {
                name: column.name.clone(),
                ty: column.ty,
            })
            .collect())
    }

    pub fn execute_relational_copy_rows(
        &mut self,
        txn_id: u64,
        copy: &CopyFromStdin,
        rows: Vec<Vec<SqlValue>>,
    ) -> Result<usize, ExecuteError> {
        self.execute_relational_copy_rows_profiled(txn_id, copy, rows)
            .map(|(rows, _profile)| rows)
    }

    pub fn execute_relational_copy_rows_profiled(
        &mut self,
        txn_id: u64,
        copy: &CopyFromStdin,
        rows: Vec<Vec<SqlValue>>,
    ) -> Result<(usize, RelationalCopyAdmissionProfile), ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if rows.is_empty() {
            return Ok((0, RelationalCopyAdmissionProfile::default()));
        }
        let columns = copy.columns.clone().unwrap_or_else(|| {
            self.catalog_snapshot()
                .relational_catalog
                .get(&copy.table)
                .map(|table| {
                    table
                        .columns
                        .iter()
                        .map(|column| column.name.clone())
                        .collect()
                })
                .unwrap_or_default()
        });
        let row_count = rows.len();
        let insert = Insert {
            table: copy.table.clone(),
            columns,
            rows,
        };
        let mut profile = RelationalCopyAdmissionProfile {
            rows: row_count,
            ..RelationalCopyAdmissionProfile::default()
        };
        let unique_preflight_started = Instant::now();
        self.preflight_unique_index_constraints(&Command::Insert(insert.clone()), txn_id)
            .map_err(ExecuteError::Engine)?;
        profile.unique_preflight_micros += unique_preflight_started.elapsed().as_micros();
        let render_started = Instant::now();
        let sql = render_relational_insert(&insert).map_err(ExecuteError::Engine)?;
        profile.render_sql_wal_payload_micros = render_started.elapsed().as_micros();
        let timestamp_micros = self.next_commit_timestamp_micros();
        let mut apply_profile = RelationalCopyAdmissionProfile::default();
        let mut current_apply_total_micros = 0;
        let commit_started = Instant::now();
        let (_token, residency_invalidation_micros) = self
            .commit_mutation_at_with_current_apply(
                txn_id,
                sql.into_bytes().into(),
                timestamp_micros,
                |engine, cat, commit_seq| {
                    let apply_started = Instant::now();
                    // Stamp with the commit sequence (commit `Index`), NOT the façade txn_id, so the
                    // live COPY apply produces the same `created_by` a WAL replay would (Stage 0). The
                    // held catalog latch (`cat`) carries any working-map mutation (sequence advance).
                    // The COPY current-apply path re-admits residency (it does not use the open-shard
                    // append), so discard the applied-rows surfaced for the append path — keep this
                    // closure's type `Result<(), _>` (the generic `apply_current` bound is unchanged).
                    let result = engine
                        .apply_insert_with_profile(
                            cat,
                            insert.clone(),
                            commit_seq,
                            Some(&mut apply_profile),
                        )
                        .map(|_applied| ());
                    current_apply_total_micros += apply_started.elapsed().as_micros();
                    result
                },
            )
            .map_err(ExecuteError::Engine)?;
        profile.commit_total_micros = commit_started.elapsed().as_micros();
        profile.current_apply_total_micros = current_apply_total_micros;
        profile.row_prepare_micros = apply_profile.row_prepare_micros;
        profile.unique_preflight_micros += apply_profile.unique_preflight_micros;
        profile.check_preflight_micros = apply_profile.check_preflight_micros;
        profile.foreign_key_preflight_micros = apply_profile.foreign_key_preflight_micros;
        profile.mvcc_insert_micros = apply_profile.mvcc_insert_micros;
        profile.value_index_append_micros = apply_profile.value_index_append_micros;
        profile.residency_invalidation_micros = residency_invalidation_micros;
        profile.wal_commit_flush_boundary_micros = profile
            .commit_total_micros
            .saturating_sub(profile.current_apply_total_micros)
            .saturating_sub(profile.residency_invalidation_micros);
        Ok((row_count, profile))
    }

    pub fn relational_table_acl(
        &self,
        table: &str,
    ) -> Option<BTreeMap<String, BTreeSet<TablePrivilege>>> {
        self.ddl_catalog()
            .relational_catalog
            .get(table)
            .map(|table| table.acl.clone())
    }

    pub fn relational_relation_acl(
        &self,
        relation: &str,
    ) -> Option<BTreeMap<String, BTreeSet<TablePrivilege>>> {
        // Acquire the catalog latch ONCE: the `.or_else` chain must not re-call `ddl_catalog()` (the
        // latch is non-reentrant — a second acquisition while the first guard is alive self-deadlocks).
        let cat = self.ddl_catalog();
        cat.relational_catalog
            .get(relation)
            .map(|table| table.acl.clone())
            .or_else(|| {
                cat.relational_views
                    .get(relation)
                    .map(|view| view.acl.clone())
            })
            .or_else(|| {
                cat.relational_materialized_views
                    .get(relation)
                    .map(|view| view.acl.clone())
            })
            .or_else(|| {
                cat.relational_sequences
                    .get(relation)
                    .map(|sequence| sequence.acl.clone())
            })
    }

    pub fn relational_default_table_acl(&self) -> BTreeMap<String, BTreeSet<TablePrivilege>> {
        self.ddl_catalog().relational_default_table_acl.clone()
    }

    pub fn relational_schema_acl(&self) -> BTreeMap<String, BTreeSet<SchemaPrivilege>> {
        self.ddl_catalog().relational_schema_acl.clone()
    }

    pub fn relational_function_acl(
        &self,
        function: &str,
    ) -> Option<BTreeMap<String, BTreeSet<FunctionPrivilege>>> {
        self.ddl_catalog()
            .relational_functions
            .get(function)
            .map(|function| function.acl.clone())
    }

    pub fn relational_catalog_view(&self, view: &str) -> Option<RelationalView> {
        self.ddl_catalog().relational_views.get(view).cloned()
    }

    pub fn relational_catalog_materialized_view(
        &self,
        materialized_view: &str,
    ) -> Option<RelationalMaterializedView> {
        self.ddl_catalog()
            .relational_materialized_views
            .get(materialized_view)
            .cloned()
    }

    pub fn relational_catalog_function(&self, function: &str) -> Option<RelationalFunction> {
        self.ddl_catalog()
            .relational_functions
            .get(function)
            .cloned()
    }

    pub fn relational_catalog_sequence(&self, sequence: &str) -> Option<RelationalSequence> {
        self.ddl_catalog()
            .relational_sequences
            .get(sequence)
            .cloned()
    }

    pub fn relational_catalog_domain(&self, domain: &str) -> Option<RelationalDomain> {
        self.ddl_catalog().relational_domains.get(domain).cloned()
    }

    pub fn relational_catalog_publication(
        &self,
        publication: &str,
    ) -> Option<RelationalPublication> {
        self.ddl_catalog()
            .relational_publications
            .get(publication)
            .cloned()
    }

    pub fn relational_catalog_subscription(
        &self,
        subscription: &str,
    ) -> Option<RelationalSubscription> {
        self.ddl_catalog()
            .relational_subscriptions
            .get(subscription)
            .cloned()
    }

    pub fn relational_table_comment(&self, table: &str) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::Table {
                table: table.to_string(),
            })
            .cloned()
    }

    pub fn relational_database_comment(&self, database: &str) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::Database {
                database: database.to_string(),
            })
            .cloned()
    }

    pub fn relational_role_comment(&self, role: &str) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::Role {
                role: role.to_string(),
            })
            .cloned()
    }

    pub fn relational_role(&self, role: &str) -> Option<RelationalRole> {
        self.ddl_catalog().relational_roles.get(role).cloned()
    }

    pub fn relational_database(&self, database: &str) -> Option<RelationalDatabase> {
        self.ddl_catalog()
            .relational_databases
            .get(database)
            .cloned()
    }

    pub fn relational_database_acl(
        &self,
        database: &str,
    ) -> Option<BTreeMap<String, BTreeSet<DatabasePrivilege>>> {
        self.ddl_catalog()
            .relational_databases
            .get(database)
            .map(|database| database.acl.clone())
    }

    pub fn relational_tablespace(&self, tablespace: &str) -> Option<RelationalTablespace> {
        self.ddl_catalog()
            .relational_tablespaces
            .get(tablespace)
            .cloned()
    }

    pub fn relational_tablespace_acl(
        &self,
        tablespace: &str,
    ) -> Option<BTreeMap<String, BTreeSet<TablespacePrivilege>>> {
        self.ddl_catalog()
            .relational_tablespaces
            .get(tablespace)
            .map(|tablespace| tablespace.acl.clone())
    }

    pub fn relational_schema_comment(&self, schema: &str) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::Schema {
                schema: schema.to_string(),
            })
            .cloned()
    }

    pub fn relational_tablespace_comment(&self, tablespace: &str) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::Tablespace {
                tablespace: tablespace.to_string(),
            })
            .cloned()
    }

    pub fn relational_column_comment(&self, table: &str, attnum: i16) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::Column {
                table: table.to_string(),
                attnum,
            })
            .cloned()
    }

    pub fn relational_index_comment(&self, index: &str) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::Index {
                index: index.to_string(),
            })
            .cloned()
    }

    pub fn relational_view_comment(&self, view: &str) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::View {
                view: view.to_string(),
            })
            .cloned()
    }

    pub fn relational_sequence_comment(&self, sequence: &str) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::Sequence {
                sequence: sequence.to_string(),
            })
            .cloned()
    }

    pub fn relational_materialized_view_comment(&self, materialized_view: &str) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::MaterializedView {
                materialized_view: materialized_view.to_string(),
            })
            .cloned()
    }

    pub fn relational_function_comment(&self, function: &str) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::Function {
                function: function.to_string(),
            })
            .cloned()
    }

    pub fn relational_extension_comment(&self, extension: &str) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::Extension {
                extension: extension.to_string(),
            })
            .cloned()
    }

    pub fn relational_constraint_comment(&self, table: &str, constraint: &str) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::Constraint {
                table: table.to_string(),
                constraint: constraint.to_string(),
            })
            .cloned()
    }

    pub fn relational_publication_comment(&self, publication: &str) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::Publication {
                publication: publication.to_string(),
            })
            .cloned()
    }

    pub fn relational_subscription_comment(&self, subscription: &str) -> Option<String> {
        self.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::Subscription {
                subscription: subscription.to_string(),
            })
            .cloned()
    }
}
