//! Catalog / ACL / comment introspection + COPY (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for the catalog-snapshot
//! accessors (relational_catalog_table/view/sequence/function/..., the
//! relational_*_acl readers, and the relational_*_comment readers across every
//! object kind) plus COPY-rows execution (execute_relational_copy_rows[_profiled]).

use super::*;
use crate::engine_transaction_reset::table_access_dependency_identities;

/// Opaque identity of the exact relation definition accepted when COPY FROM begins.
///
/// The facade carries this value from COPY-in response through final row admission.  Its fields
/// intentionally remain private: protocol code may transport the proof, but only the engine may
/// compare it with a transaction or published catalog generation.
#[derive(Debug, Clone)]
pub struct CopyTargetProof {
    table: Arc<RelationalTable>,
    /// The source relation's shared stable-OID lease, owned from COPY-in description until the
    /// protocol target and any final admission clone are dropped.
    table_access: Option<Arc<TableAccessLease>>,
}

impl PartialEq for CopyTargetProof {
    fn eq(&self, other: &Self) -> bool {
        self.table == other.table
    }
}

impl Eq for CopyTargetProof {}

/// Structural origin of a COPY target resolved inside an explicit transaction.
///
/// A private CREATE and a concurrent published CREATE can receive value-identical table metadata,
/// so facade ownership must consume this provenance instead of inferring it from proof equality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionCopyTargetOrigin {
    SnapshotBase,
    TransactionOverlay,
}

impl Engine {
    // --- Catalog introspection accessors (test/admin only; no production callers reach these). ---
    // They read the DDL working catalog under the catalog latch and return OWNED clones: a `MutexGuard`
    // cannot lend a borrow that outlives it, so the historical `Option<&T>` borrows became owned values
    // (lock-free read path, write-half MVCC). Behavior is otherwise identical.
    pub fn relational_catalog_table(&self, table: &str) -> Option<RelationalTable> {
        self.ddl_catalog().relational_catalog.get(table).cloned()
    }

    /// Validate pg_dump's bounded access-share declaration against the transaction's immutable
    /// catalog generation. The retained catalog/data generation supplies the DDL-stability
    /// contract that PostgreSQL obtains from an ACCESS SHARE lock; this method does not sequence,
    /// write WAL, or publish state.
    pub fn validate_access_share_relations_in_transaction(
        &self,
        txn_id: TxnId,
        relations: &[String],
    ) -> Result<(), ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _statement = statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
        let snapshot = self.refresh_transaction_snapshot_for_statement(txn_id, &snapshot)?;
        let catalog = snapshot.transaction_catalog();
        for relation in relations {
            let exists = catalog.relational_catalog.contains_key(relation)
                || catalog.relational_views.contains_key(relation)
                || catalog.relational_materialized_views.contains_key(relation)
                || catalog.relational_sequences.contains_key(relation);
            if !exists {
                return Err(ExecuteError::UndefinedRelation(relation.clone()));
            }
        }
        self.acquire_transaction_table_access(&snapshot, relations.iter().cloned())?;
        Ok(())
    }

    pub fn relational_copy_columns(&self, table: &str) -> Result<Vec<CopyColumn>, ExecuteError> {
        self.relational_copy_target(table)
            .map(|(columns, _proof)| columns)
    }

    /// Resolve COPY input columns and retain the exact target relation definition.  Completion
    /// must carry the returned proof back through `submit_transaction`; a DROP/recreate or
    /// shape-changing DDL between COPY-in response and CopyDone then fails before WAL.
    pub fn relational_copy_target(
        &self,
        table: &str,
    ) -> Result<(Vec<CopyColumn>, CopyTargetProof), ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let table_access = self.acquire_autocommit_table_access(table)?;
        let (columns, mut proof) = copy_target_from_catalog(&self.catalog_snapshot(), table)?;
        proof.table_access = Some(table_access);
        Ok((columns, proof))
    }

    /// Resolve COPY input columns against an explicit transaction's exact catalog generation.
    /// This is description only: the caller must still submit the typed rows through
    /// `submit_transaction`, which revalidates and stages them under the transaction statement
    /// lock.
    pub fn relational_copy_columns_in_transaction(
        &self,
        txn_id: TxnId,
        table: &str,
    ) -> Result<Vec<CopyColumn>, ExecuteError> {
        self.relational_copy_target_in_transaction(txn_id, table)
            .map(|(columns, _proof)| columns)
    }

    /// Transaction-private counterpart of [`Self::relational_copy_target`].
    pub fn relational_copy_target_in_transaction(
        &self,
        txn_id: TxnId,
        table: &str,
    ) -> Result<(Vec<CopyColumn>, CopyTargetProof), ExecuteError> {
        self.relational_copy_target_in_transaction_with_origin(txn_id, table)
            .map(|(columns, proof, _origin)| (columns, proof))
    }

    /// Resolve a transaction target and report whether the selected relation came from its base
    /// snapshot or from a transaction-private catalog overlay.
    pub fn relational_copy_target_in_transaction_with_origin(
        &self,
        txn_id: TxnId,
        table: &str,
    ) -> Result<
        (
            Vec<CopyColumn>,
            CopyTargetProof,
            TransactionCopyTargetOrigin,
        ),
        ExecuteError,
    > {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _statement = statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
        let snapshot = self.refresh_transaction_snapshot_for_statement(txn_id, &snapshot)?;
        self.acquire_transaction_table_access(&snapshot, [table.to_string()])?;
        let transaction_catalog = snapshot.transaction_catalog();
        let origin = {
            let delta = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if delta.catalog_overlay.as_ref().is_some_and(|overlay| {
                overlay.relational_catalog.get(table)
                    != snapshot.catalog.relational_catalog.get(table)
            }) {
                TransactionCopyTargetOrigin::TransactionOverlay
            } else {
                TransactionCopyTargetOrigin::SnapshotBase
            }
        };
        let (columns, mut proof) = copy_target_from_catalog(&transaction_catalog, table)?;
        let target = transaction_catalog
            .relational_catalog
            .get(table)
            .ok_or_else(|| ExecuteError::UndefinedRelation(table.to_string()))?;
        let identities =
            table_access_dependency_identities(&transaction_catalog.relational_catalog, target)?;
        // Do not clone the transaction's accumulating owner into a named protocol object. A
        // separately droppable same-owner token retains only this COPY dependency closure; after
        // COMMIT/ROLLBACK it remains shared, while unrelated identities and any reset upgrade are
        // released with the transaction snapshot.
        proof.table_access = Some(
            snapshot
                .table_access
                .retain_shared(identities.values().copied())?,
        );
        Ok((columns, proof, origin))
    }

    pub(crate) fn copy_target_matches_catalog(
        &self,
        catalog: &CatalogSnapshot,
        copy: &CopyFromStdin,
        proof: &CopyTargetProof,
    ) -> bool {
        copy.table == proof.table.name
            && catalog
                .relational_catalog
                .get(&copy.table)
                .is_some_and(|table| table == proof.table.as_ref())
    }

    pub(crate) fn execute_copy_in_transaction_with_result(
        &self,
        txn_id: TxnId,
        insert: Insert,
        copy: &CopyFromStdin,
        target: &CopyTargetProof,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        let snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _statement = statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
        let snapshot = self.refresh_transaction_snapshot_for_statement(txn_id, &snapshot)?;
        let transaction_catalog = snapshot.transaction_catalog();
        if !self.copy_target_matches_catalog(&transaction_catalog, copy, target) {
            return Err(stale_copy_target(&copy.table));
        }
        if insert.rows.is_empty() {
            let _scope = self.enter_transaction_read(Arc::clone(&snapshot));
            let next_row_id = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .next_row_id;
            self.prepare_insert(
                &insert,
                DmlReadSnapshot {
                    commit_seq: snapshot.boundary,
                    next_row_id,
                },
                None,
            )?;
            return Ok(DmlExecutionResult {
                rows_affected: 0,
                returning: None,
            });
        }
        self.execute_parsed_dml_in_transaction_statement_locked(
            txn_id,
            Command::Insert(insert),
            &snapshot,
        )
    }

    pub fn execute_relational_copy_rows(
        &self,
        txn_id: u64,
        copy: &CopyFromStdin,
        rows: Vec<Vec<SqlValue>>,
    ) -> Result<usize, ExecuteError> {
        self.execute_relational_copy_rows_profiled(txn_id, copy, rows)
            .map(|(rows, _profile)| rows)
    }

    pub fn execute_relational_copy_rows_profiled(
        &self,
        txn_id: u64,
        copy: &CopyFromStdin,
        rows: Vec<Vec<SqlValue>>,
    ) -> Result<(usize, RelationalCopyAdmissionProfile), ExecuteError> {
        // Resolve the typed shape without claiming fresh table access. The complete rows below
        // determine canonical retry identity; only fresh work acquires and retains the guard.
        let (_columns, proof) = copy_target_from_catalog(&self.catalog_snapshot(), &copy.table)?;
        self.execute_relational_copy_rows_profiled_with_target(txn_id, copy, rows, &proof)
    }

    pub(crate) fn execute_relational_copy_rows_profiled_with_target(
        &self,
        txn_id: u64,
        copy: &CopyFromStdin,
        rows: Vec<Vec<SqlValue>>,
        proof: &CopyTargetProof,
    ) -> Result<(usize, RelationalCopyAdmissionProfile), ExecuteError> {
        self.execute_relational_copy_rows_profiled_with_target_and_hook(
            txn_id,
            copy,
            rows,
            proof,
            || {},
            || {},
        )
    }

    #[cfg(test)]
    pub(crate) fn execute_relational_copy_rows_instrumented(
        &self,
        txn_id: u64,
        copy: &CopyFromStdin,
        rows: Vec<Vec<SqlValue>>,
        on_precommit: impl FnOnce(),
    ) -> Result<(usize, RelationalCopyAdmissionProfile), ExecuteError> {
        let (_columns, proof) = copy_target_from_catalog(&self.catalog_snapshot(), &copy.table)?;
        self.execute_relational_copy_rows_profiled_with_target_and_hook(
            txn_id,
            copy,
            rows,
            &proof,
            on_precommit,
            || {},
        )
    }

    #[cfg(test)]
    pub(crate) fn execute_relational_empty_copy_instrumented(
        &self,
        txn_id: u64,
        copy: &CopyFromStdin,
        on_target_validated: impl FnOnce(),
    ) -> Result<(usize, RelationalCopyAdmissionProfile), ExecuteError> {
        let (_columns, proof) = copy_target_from_catalog(&self.catalog_snapshot(), &copy.table)?;
        self.execute_relational_copy_rows_profiled_with_target_and_hook(
            txn_id,
            copy,
            Vec::new(),
            &proof,
            || {},
            on_target_validated,
        )
    }

    fn execute_relational_copy_rows_profiled_with_target_and_hook<P, Z>(
        &self,
        txn_id: u64,
        copy: &CopyFromStdin,
        rows: Vec<Vec<SqlValue>>,
        proof: &CopyTargetProof,
        on_precommit: P,
        on_zero_target_validated: Z,
    ) -> Result<(usize, RelationalCopyAdmissionProfile), ExecuteError>
    where
        P: FnOnce(),
        Z: FnOnce(),
    {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let mut proof = proof.clone();
        let mut normalized_copy = copy.clone();
        if normalized_copy.columns.is_none() {
            normalized_copy.columns = Some(
                proof
                    .table
                    .columns
                    .iter()
                    .map(|column| column.name.clone())
                    .collect(),
            );
        }
        let insert = self.relational_copy_insert(&normalized_copy, rows);
        let row_count = insert.rows.len();
        let mut profile = RelationalCopyAdmissionProfile {
            rows: row_count,
            ..RelationalCopyAdmissionProfile::default()
        };
        if row_count == 0 {
            if proof.table_access.is_none() {
                proof.table_access = Some(self.acquire_autocommit_table_access(&copy.table)?);
            }
            let preflight_started = Instant::now();
            let command = Command::Insert(insert);
            let mut target_changed = false;
            let mut on_target_validated = Some(on_zero_target_validated);
            let result = self.validate_effect_free_at_current_commit_boundary(
                txn_id,
                |engine, _prospective_commit_seq| {
                    let catalog = engine.catalog_snapshot();
                    if !engine.copy_target_matches_catalog(&catalog, &normalized_copy, &proof) {
                        target_changed = true;
                        return Err(EngineError::ApplyFailed(format!(
                            "COPY target relation \"{}\" changed after COPY began",
                            normalized_copy.table
                        )));
                    }
                    on_target_validated
                        .take()
                        .expect("zero-row validation hook must run once")();
                    {
                        let mut catalog = engine.ddl_catalog();
                        engine.ensure_dml_device_generation_with_catalog(&command, &mut catalog)?;
                    }
                    engine.preflight_constraints_against_current_device_generation(&command, txn_id)
                },
            );
            profile.unique_preflight_micros = preflight_started.elapsed().as_micros();
            match result {
                Ok(()) => {}
                Err(_error) if target_changed => {
                    return Err(stale_copy_target(&normalized_copy.table));
                }
                Err(error) => return Err(ExecuteError::Engine(error)),
            }
            return Ok((0, profile));
        }
        // COPY is INSERT compatibility ingress, not a separate direct-current apply authority.
        // Establish the protocol target's exact catalog cut before staging, then carry that same
        // cut into the ordinary typed autocommit overlay.  The overlay rechecks it under its
        // statement lock, builds `TypedInsertBatch`, and owns the only WAL/apply/publication
        // terminal.  In particular, it cannot lose one of two overlapping COPY commits between
        // a direct CPU apply and the resident-generation append.
        let catalog = self.catalog_snapshot();
        if !self.copy_target_matches_catalog(&catalog, &normalized_copy, &proof) {
            return Err(stale_copy_target(&normalized_copy.table));
        }
        on_precommit();
        // Request identity is derived from the already-typed programmatic INSERT. Reconstructing
        // SQL here made text a second carrier below COPY decoding, performed a full row-matrix
        // render, and obscured that COPY values retain Programmatic provenance. The JSON shape is
        // the existing typed-command compatibility encoding; it is request identity only and is
        // never selected as WAL, replay, apply, or publication authority.
        let request_payload = serde_json::to_vec(&insert).map_err(|error| {
            ExecuteError::Engine(EngineError::Durability(format!(
                "typed COPY request identity encode failed: {error}"
            )))
        })?;
        let request_digest = gpu_db_wal::canonical_request_digest(&request_payload);
        let commit_started = Instant::now();
        let outcome = self.execute_autocommit_insert_as_one_statement_overlay(
            txn_id,
            Command::Insert(insert),
            request_digest,
            Some(
                crate::engine_mutation_admission::CatalogVersionExpectation::Prepared(
                    catalog.commit_seq,
                ),
            ),
            current_timestamp_micros(),
            || {},
        )?;
        profile.commit_total_micros = commit_started.elapsed().as_micros();
        let rows_affected = usize::try_from(outcome.rows_affected).map_err(|_| {
            ExecuteError::Engine(EngineError::Durability(
                "typed COPY affected-row count exceeds usize".to_string(),
            ))
        })?;
        Ok((rows_affected, profile))
    }

    pub(crate) fn relational_copy_insert(
        &self,
        copy: &CopyFromStdin,
        rows: Vec<Vec<SqlValue>>,
    ) -> Insert {
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
        Insert {
            table: copy.table.clone(),
            columns,
            rows: rows
                .into_iter()
                .map(|row| row.into_iter().map(InsertCell::programmatic).collect())
                .collect(),
            returning: Vec::new(),
        }
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

fn copy_target_from_catalog(
    catalog: &CatalogSnapshot,
    table_name: &str,
) -> Result<(Vec<CopyColumn>, CopyTargetProof), ExecuteError> {
    let table = catalog
        .relational_catalog
        .get(table_name)
        .ok_or_else(|| ExecuteError::UndefinedRelation(table_name.to_string()))?;
    let columns = table
        .columns
        .iter()
        .map(|column| CopyColumn {
            name: column.name.clone(),
            ty: column.ty,
        })
        .collect();
    Ok((
        columns,
        CopyTargetProof {
            table: Arc::new(table.clone()),
            table_access: None,
        },
    ))
}

pub(crate) fn stale_copy_target(table: &str) -> ExecuteError {
    ExecuteError::Serialization(format!(
        "COPY target relation \"{table}\" changed after COPY began; restart COPY"
    ))
}
