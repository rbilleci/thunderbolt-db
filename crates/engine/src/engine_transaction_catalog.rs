//! Transaction-private catalog staging.
//!
//! Database-local `CREATE TABLE` operations share the transaction's statement-ordered operation
//! envelope with DML and typed table resets. Each statement rebuilds the private catalog from the
//! exact published base plus its typed predecessors, then atomically replaces the private overlay.

use super::*;
use crate::engine_mutation_admission::validate_prepared_catalog_version;

mod view_identity;

impl Engine {
    pub(crate) fn execute_catalog_in_transaction(
        &self,
        txn_id: TxnId,
        command: Command,
        source: Arc<str>,
        expected_catalog_version: Option<Index>,
    ) -> Result<(), ExecuteError> {
        self.execute_catalog_in_transaction_with_hook(
            txn_id,
            command,
            source,
            expected_catalog_version,
            || {},
        )
    }

    #[cfg(test)]
    fn execute_catalog_in_transaction_instrumented(
        &self,
        txn_id: TxnId,
        command: Command,
        source: Arc<str>,
        expected_catalog_version: Option<Index>,
        on_catalog_latched: impl FnOnce(),
    ) -> Result<(), ExecuteError> {
        self.execute_catalog_in_transaction_with_hook(
            txn_id,
            command,
            source,
            expected_catalog_version,
            on_catalog_latched,
        )
    }

    fn execute_catalog_in_transaction_with_hook(
        &self,
        txn_id: TxnId,
        command: Command,
        source: Arc<str>,
        expected_catalog_version: Option<Index>,
        on_catalog_latched: impl FnOnce(),
    ) -> Result<(), ExecuteError> {
        if !Self::transaction_catalog_command_is_supported(&command) {
            return Err(ExecuteError::Unsupported(
                "transactional catalog staging currently supports CREATE TABLE and CREATE VIEW only"
                    .to_string(),
            ));
        }
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
        self.execute_catalog_in_transaction_statement_locked(
            txn_id,
            command,
            source,
            expected_catalog_version,
            &snapshot,
            on_catalog_latched,
        )
    }

    fn execute_catalog_in_transaction_statement_locked(
        &self,
        txn_id: TxnId,
        command: Command,
        _source: Arc<str>,
        expected_catalog_version: Option<Index>,
        snapshot: &Arc<TransactionSnapshot>,
        on_catalog_latched: impl FnOnce(),
    ) -> Result<(), ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        self.ensure_transaction_snapshot_current(txn_id, snapshot)?;
        if snapshot.characteristics.access == TransactionAccessMode::ReadOnly {
            return Err(ExecuteError::Unsupported(
                "cannot execute DDL in a READ ONLY transaction".to_string(),
            ));
        }
        if let Some(expected) = expected_catalog_version {
            validate_prepared_catalog_version(expected, snapshot.catalog.commit_seq)?;
        }

        let (generation, prior_commands, prior_overlay, prior_base) = {
            let delta = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                delta.generation,
                delta
                    .operations
                    .iter()
                    .filter_map(|operation| match operation {
                        TransactionOperation::Catalog(staged) => Some(staged.as_ref().clone()),
                        TransactionOperation::Row(_) | TransactionOperation::TableReset(_) => None,
                    })
                    .collect::<Vec<_>>(),
                delta.catalog_overlay.clone(),
                delta.catalog_base.clone(),
            )
        };
        if !prior_commands.is_empty() && (prior_overlay.is_none() || prior_base.is_none()) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "transactional catalog operation envelope lost its base or private overlay"
                    .to_string(),
            )));
        }
        if prior_base
            .as_deref()
            .is_some_and(|base| !base.same_contents(snapshot.catalog.as_ref()))
        {
            return Err(ExecuteError::Serialization(
                "transactional catalog base changed before the next catalog statement".to_string(),
            ));
        }

        // Validation works on a private clone. The exact published base must still match the
        // statement snapshot; otherwise retryable serialization wins before any private state is
        // installed. No allocator or global catalog field changes here, so rollback is complete.
        let catalog_guard = self.ddl_catalog();
        let current =
            Self::catalog_snapshot_from_working(&catalog_guard, snapshot.catalog.commit_seq);
        if current.as_ref() != snapshot.catalog.as_ref() {
            return Err(ExecuteError::Serialization(
                "catalog changed before transactional DDL staging".to_string(),
            ));
        }
        on_catalog_latched();
        let mut working = catalog_guard.clone();
        for staged in &prior_commands {
            let scoped = Self::catalog_snapshot_from_working(&working, snapshot.catalog.commit_seq);
            if let (Command::CreateView(create), Some(identity)) =
                (&staged.command, &staged.view_identity)
            {
                Self::validate_transaction_view_before(&scoped, create, identity)
                    .map_err(ExecuteError::Engine)?;
            }
            self.with_apply_catalog(Some(Arc::clone(&scoped)), || {
                self.apply_transaction_catalog_command(&mut working, staged.command.clone())
            })
            .map_err(ExecuteError::Engine)?;
            if let (Command::CreateView(create), Some(identity)) =
                (&staged.command, &staged.view_identity)
            {
                let applied =
                    Self::catalog_snapshot_from_working(&working, snapshot.catalog.commit_seq);
                Self::validate_transaction_view_after(&applied, create, identity)
                    .map_err(ExecuteError::Engine)?;
            } else if matches!(staged.command, Command::CreateView(_)) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "transactional CREATE VIEW lost its typed identity closure".to_string(),
                )));
            }
        }
        if let Some(expected) = prior_overlay.as_deref() {
            let reconstructed =
                Self::catalog_snapshot_from_working(&working, snapshot.catalog.commit_seq);
            if reconstructed.as_ref() != expected {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "typed transactional catalog operations no longer reconstruct their private overlay"
                        .to_string(),
                )));
            }
        }
        let scoped = Self::catalog_snapshot_from_working(&working, snapshot.catalog.commit_seq);
        self.with_apply_catalog(Some(Arc::clone(&scoped)), || {
            self.apply_transaction_catalog_command(&mut working, command.clone())
        })
        .map_err(ExecuteError::Engine)?;
        // CREATE validation helpers resolve domains and sequence/default dependencies through the
        // scoped working generation. Retain the catalog latch through reconstruction and the new
        // private apply so no global catalog generation can cross the checked base.
        drop(catalog_guard);
        let overlay = Self::catalog_snapshot_from_working(&working, snapshot.catalog.commit_seq);

        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if delta.generation != generation {
            return Err(ExecuteError::Serialization(
                "transaction catalog generation changed during statement staging".to_string(),
            ));
        }
        let ordinal = u32::try_from(delta.operations.len()).map_err(|_| {
            ExecuteError::Unsupported(
                "transaction operation ordinal exceeds typed catalog framing".to_string(),
            )
        })?;
        let command_index = u32::try_from(prior_commands.len()).map_err(|_| {
            ExecuteError::Unsupported(
                "transaction catalog operation count exceeds typed identity framing".to_string(),
            )
        })?;
        let statement_digest = transaction_statement_digest(&command)?;
        let view_identity = match &command {
            Command::CreateView(create) => Some(Self::transaction_view_operation_identity(
                command_index,
                ordinal,
                &scoped,
                &overlay,
                create,
            )?),
            Command::CreateTable(_) => None,
            _ => unreachable!("transactional catalog command was classified above"),
        };
        if let Command::CreateTable(create) = &command {
            Arc::make_mut(&mut delta.resident_shards)
                .entry(create.table.clone())
                .or_default();
        }
        if prior_commands.is_empty() {
            delta.catalog_base = Some(Arc::clone(&snapshot.catalog));
        }
        delta.catalog_overlay = Some(overlay);
        delta
            .operations
            .push(TransactionOperation::Catalog(Arc::new(
                StagedCatalogCommand {
                    ordinal,
                    statement_digest,
                    command,
                    view_identity,
                },
            )));
        delta.generation = delta.generation.saturating_add(1);
        Ok(())
    }
}

#[cfg(test)]
#[path = "engine_transaction_catalog/ordered_tests.rs"]
mod ordered_tests;

#[cfg(test)]
#[path = "engine_transaction_catalog/view_tests.rs"]
mod view_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_transaction_delta::TransactionGpuReservation;
    use std::sync::mpsc;
    use std::time::Duration;

    fn parsed(sql: &str) -> gpu_db_sql::ParsedCommand {
        gpu_db_sql::ParsedCommand::parse(sql).unwrap()
    }

    #[test]
    fn create_table_is_private_until_commit_and_replays_from_one_record() {
        let engine = Engine::new_local();
        engine.submit_transaction(10, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(10, parsed("CREATE TABLE private_ddl (id int4)"))
            .unwrap();
        assert!(!engine
            .catalog_snapshot()
            .relational_catalog
            .contains_key("private_ddl"));
        assert!(engine.durable_wal_records().is_empty());

        engine.submit_transaction(10, parsed("COMMIT")).unwrap();
        assert!(engine
            .catalog_snapshot()
            .relational_catalog
            .contains_key("private_ddl"));
        let records = engine.durable_wal_records();
        assert_eq!(records.len(), 1);
        let recovered = Engine::recover_from_durable_wal(&records).unwrap();
        assert!(recovered
            .catalog_snapshot()
            .relational_catalog
            .contains_key("private_ddl"));
    }

    #[test]
    fn rollback_discards_multiple_private_catalog_operations() {
        let engine = Engine::new_local();
        engine.submit_transaction(20, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(20, parsed("CREATE TABLE rolled_ddl (id int4)"))
            .unwrap();
        engine
            .submit_transaction(20, parsed("CREATE TABLE second_ddl (id int4)"))
            .unwrap();
        assert!(engine.durable_wal_records().is_empty());
        engine.submit_transaction(20, parsed("ROLLBACK")).unwrap();
        assert!(!engine
            .catalog_snapshot()
            .relational_catalog
            .contains_key("rolled_ddl"));
        assert!(!engine
            .catalog_snapshot()
            .relational_catalog
            .contains_key("second_ddl"));
    }

    #[test]
    fn staged_catalog_and_dml_reject_catalog_drift_before_wal() {
        let engine = Engine::new_local();
        engine.submit_transaction(30, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(30, parsed("CREATE TABLE guarded_ddl (id int4)"))
            .unwrap();
        engine
            .submit_transaction(30, parsed("INSERT INTO guarded_ddl VALUES (1)"))
            .unwrap();
        assert!(engine.durable_wal_records().is_empty());

        engine
            .submit_transaction(31, parsed("CREATE TABLE concurrent_ddl (id int4)"))
            .unwrap();
        let wal_after_concurrent = engine.durable_wal_records().len();
        let commit = engine.submit_transaction(30, parsed("COMMIT")).unwrap_err();
        assert!(matches!(commit, ExecuteError::Serialization(_)));
        assert_eq!(engine.durable_wal_records().len(), wal_after_concurrent);
        assert!(!engine
            .catalog_snapshot()
            .relational_catalog
            .contains_key("guarded_ddl"));
        engine.submit_transaction(30, parsed("ROLLBACK")).unwrap();
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn create_insert_read_commit_and_recovery_share_one_atomic_record() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine.submit_transaction(33, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(
                33,
                parsed("CREATE TABLE composite_ddl (id int4 PRIMARY KEY, value int4)"),
            )
            .unwrap();
        engine
            .submit_transaction(33, parsed("INSERT INTO composite_ddl VALUES (1, 9)"))
            .unwrap();

        let select = match parse_command("SELECT value FROM composite_ddl WHERE id = 1").unwrap() {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        };
        assert_eq!(
            engine
                .execute_relational_select_in_transaction(33, &select)
                .unwrap()
                .rows
                .row(0),
            &[SqlValue::Int4(9)]
        );
        assert!(engine.execute_relational_select(&select).is_err());
        assert!(engine.durable_wal_records().is_empty());

        assert!(matches!(
            engine.submit_transaction(33, parsed("COMMIT")).unwrap(),
            TransactionAdmissionResult::Transaction(None)
        ));
        assert_eq!(engine.visible_up_to(), 1);
        let marks = engine.replication_watermarks();
        assert_eq!(
            (marks.commit_index, marks.applied_index, marks.visible_index),
            (1, 1, 1)
        );
        assert!(engine
            .read_state
            .residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key("composite_ddl"));
        assert_eq!(
            engine
                .execute_relational_select(&select)
                .unwrap()
                .rows
                .row(0),
            &[SqlValue::Int4(9)]
        );
        let records = engine.durable_wal_records();
        assert_eq!(records.len(), 1);
        let envelope = gpu_db_wal::decode_canonical_record_payload(&records[0].payload)
            .unwrap()
            .unwrap();
        let operation_payload =
            Engine::decode_engine_operation(&envelope.fragments[0].body).unwrap();
        let digest = gpu_db_wal::canonical_request_digest(&operation_payload);
        let (_, affected_rows) = engine
            .commit_state()
            .resolve_transaction_retry_digest_outcome(33, digest)
            .unwrap()
            .expect("live composite terminal status");
        assert_eq!(affected_rows, 1);
        assert_eq!(envelope.header.operation_count, 2);
        assert_eq!(envelope.header.catalog_before_epoch, 0);
        assert_eq!(envelope.header.catalog_after_epoch, 1);
        assert_eq!(envelope.header.table_block_count, 1);
        assert_eq!(
            envelope.fragments[0].kind,
            gpu_db_wal::CanonicalFragmentKind::CatalogMutation
        );
        let decoded = decode_binary_record(&records[0].payload).unwrap();
        assert!(matches!(
            decoded,
            BinaryWalRecord::Transaction(record)
                if record.catalog_commands.len() == 1 && record.mutations.len() == 1
        ));

        let recovered = Engine::recover_from_durable_wal(&records).unwrap();
        let (_, recovered_rows) = recovered
            .commit_state()
            .resolve_transaction_retry_digest_outcome(33, digest)
            .unwrap()
            .expect("recovered composite terminal status");
        assert_eq!(recovered_rows, 1);
        let recovered_marks = recovered.replication_watermarks();
        assert_eq!(
            (
                recovered_marks.commit_index,
                recovered_marks.applied_index,
                recovered_marks.visible_index,
            ),
            (1, 1, 1)
        );
        assert!(recovered
            .read_state
            .residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key("composite_ddl"));
        assert_eq!(
            recovered
                .execute_relational_select(&select)
                .unwrap()
                .rows
                .row(0),
            &[SqlValue::Int4(9)]
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn ordered_catalog_create_reset_dml_rollback_discards_every_private_effect() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine.submit_transaction(34, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(34, parsed("CREATE TABLE rolled_composite (id int4)"))
            .unwrap();
        engine
            .submit_transaction(34, parsed("CREATE TABLE rolled_composite_peer (id int4)"))
            .unwrap();
        engine
            .submit_transaction(34, parsed("INSERT INTO rolled_composite VALUES (1)"))
            .unwrap();
        engine
            .submit_transaction(34, parsed("TRUNCATE rolled_composite"))
            .unwrap();
        engine
            .submit_transaction(34, parsed("INSERT INTO rolled_composite VALUES (2)"))
            .unwrap();
        engine
            .submit_transaction(34, parsed("INSERT INTO rolled_composite_peer VALUES (3)"))
            .unwrap();
        engine.submit_transaction(34, parsed("ROLLBACK")).unwrap();
        for table in ["rolled_composite", "rolled_composite_peer"] {
            assert!(!engine
                .catalog_snapshot()
                .relational_catalog
                .contains_key(table));
        }
        assert!(engine.durable_wal_records().is_empty());
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn transaction_created_compound_pk_and_unique_use_exact_device_verdicts() {
        for (txn_id, table, duplicate) in [
            (
                340,
                "private_compound_pk",
                "INSERT INTO private_compound_pk VALUES (1, 10, 8, 80)",
            ),
            (
                350,
                "private_compound_unique",
                "INSERT INTO private_compound_unique VALUES (3, 30, 7, 70)",
            ),
        ] {
            let engine = Engine::new_local();
            engine.set_shard_residency_enabled(true);
            engine.submit_transaction(txn_id, parsed("BEGIN")).unwrap();
            engine
                .submit_transaction(
                    txn_id,
                    parsed(&format!(
                        "CREATE TABLE {table} (tenant_id int4, id int4, code int4, region int4, \
                         PRIMARY KEY (tenant_id, id), UNIQUE (code, region))"
                    )),
                )
                .unwrap();
            engine
                .submit_transaction(
                    txn_id,
                    parsed(&format!("INSERT INTO {table} VALUES (1, 10, 7, 70)")),
                )
                .unwrap();
            engine
                .submit_transaction(
                    txn_id,
                    parsed(&format!("INSERT INTO {table} VALUES (2, 20, 8, 80)")),
                )
                .unwrap();
            let error = engine
                .submit_transaction(txn_id, parsed(duplicate))
                .expect_err("the duplicate compound tuple must reject before WAL");
            assert!(matches!(
                error,
                ExecuteError::Engine(EngineError::UniqueViolation(_))
                    | ExecuteError::Serialization(_)
            ));
            assert!(engine.durable_wal_records().is_empty());
            engine
                .submit_transaction(txn_id, parsed("ROLLBACK"))
                .unwrap();
        }

        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.submit_transaction(351, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(
                351,
                parsed(
                    "CREATE TABLE private_compound_batch (tenant_id int4, id int4, code int4, \
                     region int4, PRIMARY KEY (tenant_id, id), UNIQUE (code, region))",
                ),
            )
            .unwrap();
        let error = engine
            .submit_transaction(
                351,
                parsed(
                    "INSERT INTO private_compound_batch VALUES \
                     (1, 10, 7, 70), (2, 20, 7, 70)",
                ),
            )
            .expect_err(
                "the transient device relation must reject an in-statement tuple duplicate",
            );
        assert!(matches!(
            error,
            ExecuteError::Engine(EngineError::UniqueViolation(_)) | ExecuteError::Serialization(_)
        ));
        assert!(engine.durable_wal_records().is_empty());
        engine.submit_transaction(351, parsed("ROLLBACK")).unwrap();
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn transaction_created_check_is_a_device_verdict_with_pg_null_semantics() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.submit_transaction(360, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(
                360,
                parsed(
                    "CREATE TABLE private_check (id int4 PRIMARY KEY, value int4, \
                     CONSTRAINT value_positive CHECK (value > 0))",
                ),
            )
            .unwrap();
        engine
            .submit_transaction(
                360,
                parsed("INSERT INTO private_check VALUES (1, 9), (2, NULL)"),
            )
            .unwrap();
        let error = engine
            .submit_transaction(360, parsed("INSERT INTO private_check VALUES (3, 0)"))
            .expect_err("FALSE, unlike NULL/UNKNOWN, must violate CHECK");
        assert!(error.to_string().contains("value_positive"), "{error:?}");
        assert!(engine.durable_wal_records().is_empty());
        engine.submit_transaction(360, parsed("ROLLBACK")).unwrap();
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn exact_budget_credit_cannot_be_stolen_after_composite_wal_durability() {
        let mut engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.submit_transaction(35, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(
                35,
                parsed("CREATE TABLE credited_ddl (id int4 PRIMARY KEY, value int4)"),
            )
            .unwrap();
        engine
            .submit_transaction(35, parsed("INSERT INTO credited_ddl VALUES (1, 9)"))
            .unwrap();
        let table = engine
            .transaction_snapshot_handle(35)
            .unwrap()
            .transaction_catalog()
            .relational_catalog
            .get("credited_ddl")
            .cloned()
            .unwrap();
        let final_index_bytes =
            crate::engine_residency::estimated_named_index_bytes_for_shard(&table, 1, 1).unwrap();
        let exact_budget = engine
            .relational_resident_bytes_for_gpu(0)
            .saturating_add(final_index_bytes);
        engine.set_relational_residency_budget_bytes(0, exact_budget);
        let engine = Arc::new(engine);
        let (durable_tx, durable_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        engine.set_transaction_post_durable_hook(move || {
            durable_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        let committing = Arc::clone(&engine);
        let commit =
            std::thread::spawn(move || committing.submit_transaction(35, parsed("COMMIT")));
        durable_rx.recv_timeout(Duration::from_secs(2)).unwrap();

        let mut thief = TransactionGpuReservation::new(&engine);
        assert!(
            thief.reserve(0, 1).is_err(),
            "another allocator must see the retained publication credit"
        );
        release_tx.send(()).unwrap();
        commit.join().unwrap().unwrap();
        assert!(engine.relational_resident_bytes_for_gpu(0) <= exact_budget);
        assert!(engine
            .transaction_private_gpu_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_empty());
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn enrolled_index_purges_cannot_wedge_a_durable_composite_commit() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine
            .submit_transaction(
                358,
                parsed(
                    "CREATE TABLE indexed_existing (tenant_id int4, id int4, code int4, \
                     region int4, value int4, PRIMARY KEY (tenant_id, id), \
                     UNIQUE (code, region))",
                ),
            )
            .unwrap();
        engine
            .submit_transaction(
                359,
                parsed("INSERT INTO indexed_existing VALUES (1, 10, 7, 70, 1)"),
            )
            .unwrap();
        engine
            .publish_relational_resident_indexes("indexed_existing")
            .unwrap();
        engine
            .submit_transaction(
                361,
                parsed("CREATE TABLE unrelated_indexed (id int4 PRIMARY KEY)"),
            )
            .unwrap();
        engine
            .submit_transaction(362, parsed("INSERT INTO unrelated_indexed VALUES (1)"))
            .unwrap();
        engine
            .publish_relational_resident_indexes("unrelated_indexed")
            .unwrap();
        let table = engine.relational_catalog_table("indexed_existing").unwrap();
        assert!(engine.relational_named_index_publication_required(&table));

        engine.submit_transaction(360, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(
                360,
                parsed("INSERT INTO indexed_existing VALUES (2, 20, 8, 80, 2)"),
            )
            .unwrap();
        engine
            .submit_transaction(360, parsed("CREATE TABLE paired_index_lifecycle (id int4)"))
            .unwrap();

        // Model the auditor's preflight race exactly: mandatory enrollment survives while its
        // physical cache/coverage disappears before COMMIT performs exact publication sizing.
        engine
            .read_state
            .residency
            .purge_shard_pk_index_for_table("indexed_existing");
        assert!(!engine
            .read_state
            .residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key("indexed_existing"));
        assert!(engine.relational_named_index_publication_required(&table));

        let wal_before = engine.durable_wal_records().len();
        let engine = Arc::new(engine);
        let observing = Arc::clone(&engine);
        engine.set_transaction_post_durable_hook(move || {
            // The lifecycle is table-scoped: an unrelated retirement remains immediate and must
            // not be swallowed merely because another table is being canonically published.
            observing
                .read_state
                .residency
                .purge_shard_pk_index_for_table("unrelated_indexed");
            assert!(observing
                .read_state
                .residency
                .shard_pk_device_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .keys()
                .all(|(table, ..)| table != "unrelated_indexed"));
            // Restoration has completed and WAL is durable. A second destructive invalidation
            // must be deferred; otherwise canonical apply would lose mandatory append coverage.
            observing
                .read_state
                .residency
                .purge_shard_pk_index_for_table("indexed_existing");
            let allocations = observing
                .read_state
                .residency
                .shard_pk_device_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .filter(|((table, ..), entry)| {
                    table == "indexed_existing" && entry.device_index.is_some()
                })
                .count();
            assert_eq!(
                allocations, 2,
                "compound PK and UNIQUE coverage must stay pinned"
            );
            assert!(observing
                .read_state
                .residency
                .named_index_coverage_complete
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains_key("indexed_existing"));
        });
        engine.submit_transaction(360, parsed("COMMIT")).unwrap();
        assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
        assert!(engine
            .catalog_snapshot()
            .relational_catalog
            .contains_key("paired_index_lifecycle"));
        assert_eq!(
            engine
                .read_state
                .residency
                .shard_pk_device_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .filter(|((table, ..), entry)| {
                    table == "indexed_existing" && entry.device_index.is_some()
                })
                .count(),
            2,
            "the successful final publication supersedes old-generation purge requests"
        );
        assert!(engine
            .read_state
            .residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key("indexed_existing"));

        let select = match parse_command(
            "SELECT tenant_id, id, code, region, value FROM indexed_existing ORDER BY id",
        )
        .unwrap()
        {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        };
        let expected = vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Int4(10),
                SqlValue::Int4(7),
                SqlValue::Int4(70),
                SqlValue::Int4(1),
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Int4(20),
                SqlValue::Int4(8),
                SqlValue::Int4(80),
                SqlValue::Int4(2),
            ],
        ];
        assert_eq!(
            engine.execute_relational_select(&select).unwrap().rows,
            expected
        );

        let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
        assert_eq!(
            recovered.execute_relational_select(&select).unwrap().rows,
            expected
        );
        assert!(recovered
            .catalog_snapshot()
            .relational_catalog
            .contains_key("paired_index_lifecycle"));
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn dml_then_create_dense_text_reserves_rollover_before_wal_and_can_retry() {
        let mut engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine
            .submit_transaction(
                352,
                parsed("CREATE TABLE existing_dense_text (id int4, value text)"),
            )
            .unwrap();
        engine
            .submit_transaction(
                353,
                parsed("INSERT INTO existing_dense_text VALUES (1, 'seed')"),
            )
            .unwrap();
        engine.submit_transaction(354, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(
                354,
                parsed("INSERT INTO existing_dense_text VALUES (2, 'transaction')"),
            )
            .unwrap();
        engine
            .submit_transaction(354, parsed("CREATE TABLE paired_empty (id int4)"))
            .unwrap();

        let private_peak = engine.relational_resident_bytes_for_gpu(0);
        let wal_before = engine.durable_wal_records().len();
        engine.set_relational_residency_budget_bytes(0, private_peak);
        let rejected = engine
            .submit_transaction(354, parsed("COMMIT"))
            .unwrap_err();
        assert!(
            matches!(rejected, ExecuteError::Engine(EngineError::ApplyFailed(_))),
            "the missing canonical created-by allocation must reject before WAL: {rejected:?}"
        );
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert!(engine.transaction_snapshot_handle(354).is_some());

        // The private dense payload and row-id buffer are publication credit. Canonical rollover
        // additionally retains exactly one u64 created-by slot for this one-row text shard.
        let exact_peak = private_peak + 8;
        engine.set_relational_residency_budget_bytes(0, exact_peak);
        engine.submit_transaction(354, parsed("COMMIT")).unwrap();
        assert!(engine.relational_resident_bytes_for_gpu(0) <= exact_peak);
        assert!(engine
            .catalog_snapshot()
            .relational_catalog
            .contains_key("paired_empty"));
        let select =
            match parse_command("SELECT id, value FROM existing_dense_text ORDER BY id").unwrap() {
                Command::Select(select) => select,
                other => panic!("expected SELECT, got {other:?}"),
            };
        let expected = vec![
            vec![SqlValue::Int4(1), SqlValue::Text("seed".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Text("transaction".to_string())],
        ];
        assert_eq!(
            engine.execute_relational_select(&select).unwrap().rows,
            expected
        );
        let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
        assert_eq!(
            recovered.execute_relational_select(&select).unwrap().rows,
            expected
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn coalesced_created_compound_indexes_fit_exact_final_budget_commit_and_recovery() {
        let mut engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.submit_transaction(356, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(
                356,
                parsed(
                    "CREATE TABLE final_compound (tenant_id int4, id int4, code int4, \
                     region int4, value int4, PRIMARY KEY (tenant_id, id), \
                     UNIQUE (code, region))",
                ),
            )
            .unwrap();
        engine
            .submit_transaction(
                356,
                parsed("INSERT INTO final_compound VALUES (1, 10, 7, 70, 1)"),
            )
            .unwrap();
        engine
            .submit_transaction(
                356,
                parsed("UPDATE final_compound SET value = 2 WHERE tenant_id = 1 AND id = 10"),
            )
            .unwrap();

        let snapshot = engine.transaction_snapshot_handle(356).unwrap();
        let transaction_catalog = snapshot.transaction_catalog();
        let table = transaction_catalog
            .relational_catalog
            .get("final_compound")
            .unwrap();
        let names = table
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let types = table
            .columns
            .iter()
            .map(|column| column.ty)
            .collect::<Vec<_>>();
        let final_row = vec![
            SqlValue::Int4(1),
            SqlValue::Int4(10),
            SqlValue::Int4(7),
            SqlValue::Int4(70),
            SqlValue::Int4(2),
        ];
        let final_payload = crate::engine_residency::build_relational_device_payload(
            &names,
            &types,
            std::slice::from_ref(&final_row),
        )
        .unwrap()
        .0
        .len() as u64;
        let final_index_bytes =
            crate::engine_residency::estimated_named_index_bytes_for_shard(table, 1, 1).unwrap();
        let final_publication = final_payload + 8 + final_index_bytes;
        let private_peak = engine.relational_resident_bytes_for_gpu(0);
        assert!(snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .commit_gpu_bytes_by_gpu
            .is_empty());
        let exact_peak = private_peak.max(final_publication);
        engine.set_relational_residency_budget_bytes(0, exact_peak);
        engine.submit_transaction(356, parsed("COMMIT")).unwrap();
        assert!(engine.relational_resident_bytes_for_gpu(0) <= exact_peak);
        assert!(engine
            .read_state
            .residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key("final_compound"));
        let allocation_count = engine
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|((table, ..), entry)| {
                table == "final_compound" && entry.device_index.is_some()
            })
            .filter_map(|(_, entry)| {
                entry
                    .device_index
                    .as_ref()
                    .map(|memory| memory.device_ptr())
            })
            .collect::<BTreeSet<_>>()
            .len();
        assert_eq!(
            allocation_count, 2,
            "compound PK and compound UNIQUE require two mandatory device allocations"
        );

        for duplicate in [
            "INSERT INTO final_compound VALUES (1, 10, 8, 80, 3)",
            "INSERT INTO final_compound VALUES (2, 20, 7, 70, 3)",
        ] {
            let error = engine
                .submit_transaction(357, parsed(duplicate))
                .expect_err("exact duplicate compound tuple must raise unique_violation");
            assert!(
                error.is_unique_violation(),
                "the typed error maps to PostgreSQL SQLSTATE 23505: {error:?}"
            );
        }

        let records = engine.durable_wal_records();
        let recovered = Engine::recover_from_durable_wal(&records).unwrap();
        assert!(recovered
            .read_state
            .residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key("final_compound"));
        let select =
            match parse_command("SELECT value FROM final_compound WHERE tenant_id = 1 AND id = 10")
                .unwrap()
            {
                Command::Select(select) => select,
                other => panic!("expected SELECT, got {other:?}"),
            };
        assert_eq!(
            recovered
                .execute_relational_select(&select)
                .unwrap()
                .rows
                .row(0),
            &[SqlValue::Int4(2)]
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn ordered_catalog_durable_composite_remains_observer_invisible_until_one_publication() {
        let engine = Arc::new(Engine::new_local());
        engine.set_shard_residency_enabled(true);
        engine.submit_transaction(355, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(
                355,
                parsed("CREATE TABLE paused_composite (id int4 PRIMARY KEY, value int4)"),
            )
            .unwrap();
        engine
            .submit_transaction(
                355,
                parsed("CREATE TABLE paused_composite_peer (id int4 PRIMARY KEY)"),
            )
            .unwrap();
        engine
            .submit_transaction(355, parsed("INSERT INTO paused_composite VALUES (1, 9)"))
            .unwrap();
        engine
            .submit_transaction(355, parsed("INSERT INTO paused_composite_peer VALUES (2)"))
            .unwrap();
        let (durable_tx, durable_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let observing = Arc::clone(&engine);
        engine.set_transaction_post_durable_hook(move || {
            assert_eq!(observing.visible_up_to(), 0);
            assert!(!observing
                .catalog_snapshot()
                .relational_catalog
                .contains_key("paused_composite"));
            assert!(!observing
                .catalog_snapshot()
                .relational_catalog
                .contains_key("paused_composite_peer"));
            durable_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        let committing = Arc::clone(&engine);
        let commit =
            std::thread::spawn(move || committing.submit_transaction(355, parsed("COMMIT")));
        durable_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(engine.visible_up_to(), 0);
        assert!(!engine
            .catalog_snapshot()
            .relational_catalog
            .contains_key("paused_composite"));
        release_tx.send(()).unwrap();
        assert!(matches!(
            commit.join().unwrap().unwrap(),
            TransactionAdmissionResult::Transaction(None)
        ));
        assert_eq!(engine.visible_up_to(), 1);
        assert!(engine
            .catalog_snapshot()
            .relational_catalog
            .contains_key("paused_composite"));
        assert!(engine
            .catalog_snapshot()
            .relational_catalog
            .contains_key("paused_composite_peer"));
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn ordered_catalog_transaction_created_serial_post_state_matches_live_and_recovery() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.submit_transaction(365, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(
                365,
                parsed("CREATE TABLE private_serial (id serial PRIMARY KEY, value int4)"),
            )
            .unwrap();
        engine
            .submit_transaction(
                365,
                parsed("INSERT INTO private_serial (value) VALUES (9), (10)"),
            )
            .unwrap();
        engine.submit_transaction(365, parsed("COMMIT")).unwrap();
        let select =
            match parse_command("SELECT id, value FROM private_serial ORDER BY id").unwrap() {
                Command::Select(select) => select,
                other => panic!("expected SELECT, got {other:?}"),
            };
        let live_rows = engine.execute_relational_select(&select).unwrap().rows;
        assert_eq!(
            live_rows,
            vec![
                vec![SqlValue::Int4(1), SqlValue::Int4(9)],
                vec![SqlValue::Int4(2), SqlValue::Int4(10)],
            ]
        );
        let live_sequence = engine
            .catalog_snapshot()
            .relational_sequences
            .get("private_serial_id_seq")
            .cloned()
            .unwrap();
        assert_eq!(
            (live_sequence.last_value, live_sequence.is_called),
            (2, true)
        );
        let BinaryWalRecord::Transaction(record) =
            decode_binary_record(&engine.durable_wal_records()[0].payload).unwrap()
        else {
            panic!("serial catalog transaction did not use resolved binary WAL");
        };
        assert_eq!(
            record.sequence_input_oids,
            BTreeMap::from([
                ((0, "private_serial_id_seq".to_string()), live_sequence.oid),
                ((1, "private_serial_id_seq".to_string()), live_sequence.oid),
            ])
        );
        assert_eq!(
            record.sequence_advances,
            BTreeMap::from([("private_serial_id_seq".to_string(), (2, true))])
        );

        let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
        assert_eq!(
            recovered.execute_relational_select(&select).unwrap().rows,
            live_rows
        );
        let recovered_sequence = recovered
            .catalog_snapshot()
            .relational_sequences
            .get("private_serial_id_seq")
            .cloned()
            .unwrap();
        assert_eq!(recovered_sequence, live_sequence);
        assert_eq!(
            recovered.read_state.mvcc.current_row_id(),
            engine.read_state.mvcc.current_row_id()
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn existing_table_residency_loss_serializes_before_composite_wal() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine
            .submit_transaction(
                36,
                parsed("CREATE TABLE retained_existing (id int4, value int4)"),
            )
            .unwrap();
        engine
            .submit_transaction(37, parsed("INSERT INTO retained_existing VALUES (1, 10)"))
            .unwrap();
        engine.submit_transaction(38, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(38, parsed("INSERT INTO retained_existing VALUES (2, 20)"))
            .unwrap();
        engine
            .submit_transaction(38, parsed("CREATE TABLE paired_create (id int4)"))
            .unwrap();
        let wal_before = engine.durable_wal_records().len();
        engine.ddl_catalog().relational_resident_cache.remove_table(
            "retained_existing",
            &engine.read_state.residency,
            &engine.read_state.route_telemetry,
        );
        let error = engine.submit_transaction(38, parsed("COMMIT")).unwrap_err();
        assert!(matches!(error, ExecuteError::Serialization(_)));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        engine.submit_transaction(38, parsed("ROLLBACK")).unwrap();
    }

    #[test]
    fn unrelated_commit_rebases_staged_catalog_without_changing_private_identity() {
        let engine = Engine::new_local();
        engine.submit_transaction(41, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(41, parsed("CREATE TABLE conservative_ddl (id int4)"))
            .unwrap();
        let private_before = engine
            .transaction_snapshot_handle(41)
            .unwrap()
            .transaction_catalog()
            .relational_catalog["conservative_ddl"]
            .clone();

        engine
            .submit_transaction(42, parsed("SET unrelated = value"))
            .unwrap();
        let wal_after_unrelated = engine.durable_wal_records().len();
        // This metadata lookup is a new READ COMMITTED statement and therefore exercises private
        // overlay rebasing before COMMIT, not only the terminal revalidation shortcut.
        let columns = engine
            .relational_copy_columns_in_transaction(41, "conservative_ddl")
            .unwrap();
        assert_eq!(columns.len(), 1);
        let private_after = engine
            .transaction_snapshot_handle(41)
            .unwrap()
            .transaction_catalog()
            .relational_catalog["conservative_ddl"]
            .clone();
        assert_eq!(private_after, private_before);

        engine.submit_transaction(41, parsed("COMMIT")).unwrap();
        assert_eq!(engine.durable_wal_records().len(), wal_after_unrelated + 1);
        assert_eq!(
            engine
                .catalog_snapshot()
                .relational_catalog
                .get("conservative_ddl"),
            Some(&private_before)
        );
        let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
        assert_eq!(
            recovered
                .catalog_snapshot()
                .relational_catalog
                .get("conservative_ddl"),
            Some(&private_before)
        );
    }

    #[test]
    fn catalog_allocator_aba_still_serializes_staged_create_before_wal() {
        let engine = Engine::new_local();
        engine.submit_transaction(43, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(43, parsed("CREATE TABLE allocator_guarded (id int4)"))
            .unwrap();

        engine
            .submit_transaction(44, parsed("CREATE TABLE allocator_aba (id int4)"))
            .unwrap();
        engine
            .submit_transaction(45, parsed("DROP TABLE allocator_aba"))
            .unwrap();
        let wal_after_aba = engine.durable_wal_records().len();
        let commit = engine.submit_transaction(43, parsed("COMMIT")).unwrap_err();
        assert!(matches!(commit, ExecuteError::Serialization(_)));
        assert_eq!(engine.durable_wal_records().len(), wal_after_aba);
        assert!(!engine
            .catalog_snapshot()
            .relational_catalog
            .contains_key("allocator_guarded"));
        engine.submit_transaction(43, parsed("ROLLBACK")).unwrap();
    }

    #[test]
    fn repeatable_read_create_commits_after_unrelated_change_without_statement_rebase() {
        let engine = Engine::new_local();
        engine
            .submit_transaction(46, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
            .unwrap();
        engine
            .submit_transaction(46, parsed("CREATE TABLE rr_unrelated_ddl (id int4)"))
            .unwrap();
        let private = engine
            .transaction_snapshot_handle(46)
            .unwrap()
            .transaction_catalog()
            .relational_catalog["rr_unrelated_ddl"]
            .clone();

        engine
            .submit_transaction(47, parsed("SET rr_unrelated = value"))
            .unwrap();
        // No transaction statement follows the unrelated commit: this goes directly through the
        // terminal catalog-content proof rather than READ COMMITTED overlay rebasing.
        engine.submit_transaction(46, parsed("COMMIT")).unwrap();
        assert_eq!(
            engine
                .catalog_snapshot()
                .relational_catalog
                .get("rr_unrelated_ddl"),
            Some(&private)
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn unrelated_commit_rebases_private_create_dml_with_null_and_recovers() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine
            .submit_transaction(
                48,
                parsed("CREATE TABLE unrelated_row_ids (id int4 PRIMARY KEY)"),
            )
            .unwrap();
        engine.submit_transaction(49, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(
                49,
                parsed("CREATE TABLE rebased_private (id int4 PRIMARY KEY, value int4)"),
            )
            .unwrap();
        engine
            .submit_transaction(49, parsed("INSERT INTO rebased_private VALUES (1, NULL)"))
            .unwrap();

        // A real unrelated row commit republishes identical catalog contents, advances the global
        // row-id allocator, and forces rekey + replay of the private NULL-bearing insert.
        engine
            .submit_transaction(50, parsed("INSERT INTO unrelated_row_ids VALUES (100)"))
            .unwrap();
        engine
            .submit_transaction(49, parsed("INSERT INTO rebased_private VALUES (2, 9)"))
            .unwrap();
        engine.submit_transaction(49, parsed("COMMIT")).unwrap();

        let select =
            match parse_command("SELECT id, value FROM rebased_private ORDER BY id").unwrap() {
                Command::Select(select) => select,
                other => panic!("expected SELECT, got {other:?}"),
            };
        let live = engine.execute_relational_select(&select).unwrap().rows;
        assert_eq!(live.row(0), &[SqlValue::Int4(1), SqlValue::Null]);
        assert_eq!(live.row(1), &[SqlValue::Int4(2), SqlValue::Int4(9)]);

        let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
        recovered.set_shard_residency_enabled(true);
        recovered.set_auto_admit_on_commit(true);
        let replayed = recovered.execute_relational_select(&select).unwrap().rows;
        assert_eq!(replayed, live);
    }

    #[test]
    fn catalog_latch_keeps_domain_resolution_on_the_pinned_generation() {
        let engine = Arc::new(Engine::new_local());
        engine
            .submit_transaction(50, parsed("CREATE DOMAIN pinned_type AS int4"))
            .unwrap();
        engine
            .submit_transaction(51, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
            .unwrap();

        let (release_racer_tx, release_racer_rx) = mpsc::channel();
        let (racer_started_tx, racer_started_rx) = mpsc::channel();
        let (racer_done_tx, racer_done_rx) = mpsc::channel();
        let racer_engine = Arc::clone(&engine);
        let racer = std::thread::spawn(move || {
            release_racer_rx.recv().unwrap();
            racer_started_tx.send(()).unwrap();
            let result = racer_engine.submit_transaction(52, parsed("DROP DOMAIN pinned_type"));
            racer_done_tx.send(result).unwrap();
        });

        let statement = parsed("CREATE TABLE pinned_domain_table (value pinned_type)");
        let (command, source) = statement.into_parts();
        let hook_engine = Arc::clone(&engine);
        engine
            .execute_catalog_in_transaction_instrumented(51, command, source, None, || {
                assert!(matches!(
                    hook_engine.catalog_latch.try_lock(),
                    Err(std::sync::TryLockError::WouldBlock)
                ));
                release_racer_tx.send(()).unwrap();
                racer_started_rx.recv().unwrap();
                assert!(matches!(
                    racer_done_rx.recv_timeout(Duration::from_millis(100)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ));
            })
            .unwrap();
        racer_done_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        racer.join().unwrap();

        let private_catalog = engine
            .transaction_snapshot_handle(51)
            .unwrap()
            .transaction_catalog();
        let column = &private_catalog.relational_catalog["pinned_domain_table"].columns[0];
        assert_eq!(column.domain.as_deref(), Some("pinned_type"));
        assert_eq!(column.ty, SqlType::Int4);
        let wal_after_drop = engine.durable_wal_records().len();
        let commit = engine.submit_transaction(51, parsed("COMMIT")).unwrap_err();
        assert!(matches!(commit, ExecuteError::Serialization(_)));
        assert_eq!(engine.durable_wal_records().len(), wal_after_drop);
        engine.submit_transaction(51, parsed("ROLLBACK")).unwrap();
    }

    #[test]
    fn stale_prepared_create_rejects_without_private_or_global_effects() {
        let engine = Engine::new_local();
        let prepared_catalog_version = engine.catalog_snapshot().commit_seq;
        engine
            .submit_transaction(60, parsed("CREATE TABLE version_changer (id int4)"))
            .unwrap();
        engine.submit_transaction(61, parsed("BEGIN")).unwrap();
        let wal_before = engine.durable_wal_records().len();

        let error = engine
            .submit_transaction(
                61,
                MutationRequest::new(parsed("CREATE TABLE stale_prepared_ddl (id int4)"))
                    .with_expected_catalog_version(prepared_catalog_version),
            )
            .unwrap_err();
        assert!(matches!(error, ExecuteError::Unsupported(_)));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        let snapshot = engine.transaction_snapshot_handle(61).unwrap();
        assert!(snapshot.transaction_delta_is_empty());
        assert!(!snapshot
            .transaction_catalog()
            .relational_catalog
            .contains_key("stale_prepared_ddl"));
        engine.submit_transaction(61, parsed("ROLLBACK")).unwrap();
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn composite_create_dml_post_durable_failure_is_fail_stop_and_recovery_owned() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.submit_transaction(70, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(
                70,
                parsed("CREATE TABLE durable_private_ddl (id int4 PRIMARY KEY, value int4)"),
            )
            .unwrap();
        engine
            .submit_transaction(70, parsed("INSERT INTO durable_private_ddl VALUES (1, 9)"))
            .unwrap();
        engine.fail_next_transaction_post_durable_apply();

        let error = engine.submit_transaction(70, parsed("COMMIT")).unwrap_err();
        assert!(error.is_indeterminate());
        assert!(engine.is_commit_path_poisoned());
        assert!(engine.transaction_snapshot_handle(70).is_some());
        assert!(engine.submit_transaction(70, parsed("ROLLBACK")).is_err());
        let records = engine.durable_wal_records();
        assert_eq!(records.len(), 1);
        let recovered = Engine::recover_from_durable_wal(&records).unwrap();
        assert!(recovered
            .catalog_snapshot()
            .relational_catalog
            .contains_key("durable_private_ddl"));
        let select =
            match parse_command("SELECT value FROM durable_private_ddl WHERE id = 1").unwrap() {
                Command::Select(select) => select,
                other => panic!("expected SELECT, got {other:?}"),
            };
        assert_eq!(
            recovered
                .execute_relational_select(&select)
                .unwrap()
                .rows
                .row(0),
            &[SqlValue::Int4(9)]
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn staged_dml_composes_with_following_catalog_statement() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine
            .submit_transaction(80, parsed("CREATE TABLE dml_first (id int4 PRIMARY KEY)"))
            .unwrap();
        engine.submit_transaction(81, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(81, parsed("INSERT INTO dml_first VALUES (1)"))
            .unwrap();
        engine
            .submit_transaction(81, parsed("CREATE TABLE must_not_stage (id int4)"))
            .unwrap();
        assert!(engine
            .transaction_snapshot_handle(81)
            .unwrap()
            .transaction_catalog()
            .relational_catalog
            .contains_key("must_not_stage"));
        engine.submit_transaction(81, parsed("COMMIT")).unwrap();
        assert!(engine
            .catalog_snapshot()
            .relational_catalog
            .contains_key("must_not_stage"));
        let records = engine.durable_wal_records();
        assert_eq!(records.len(), 2);
        let recovered = Engine::recover_from_durable_wal(&records).unwrap();
        assert!(recovered
            .catalog_snapshot()
            .relational_catalog
            .contains_key("must_not_stage"));
        let select = match parse_command("SELECT id FROM dml_first WHERE id = 1").unwrap() {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        };
        assert_eq!(
            recovered
                .execute_relational_select(&select)
                .unwrap()
                .rows
                .row(0),
            &[SqlValue::Int4(1)]
        );
    }

    #[test]
    fn read_only_transaction_rejects_catalog_staging_pre_effect() {
        let engine = Engine::new_local();
        engine
            .submit_transaction(40, parsed("BEGIN READ ONLY"))
            .unwrap();
        let error = engine
            .submit_transaction(40, parsed("CREATE TABLE read_only_ddl (id int4)"))
            .unwrap_err();
        assert!(matches!(error, ExecuteError::Unsupported(_)));
        assert!(engine.durable_wal_records().is_empty());
        assert!(engine
            .transaction_snapshot_handle(40)
            .unwrap()
            .transaction_delta_is_empty());
        engine.submit_transaction(40, parsed("ROLLBACK")).unwrap();
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn transaction_truncate_continue_identity_composes_with_insert_and_rollback() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine
            .submit_transaction(
                90,
                parsed("CREATE TABLE restore_target (id int4 PRIMARY KEY)"),
            )
            .unwrap();
        engine
            .submit_transaction(91, parsed("INSERT INTO restore_target VALUES (1)"))
            .unwrap();

        engine.submit_transaction(92, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(92, parsed("TRUNCATE TABLE ONLY restore_target"))
            .unwrap();
        engine
            .submit_transaction(92, parsed("INSERT INTO restore_target VALUES (1)"))
            .unwrap();
        engine.submit_transaction(92, parsed("COMMIT")).unwrap();
        let rows = engine
            .execute_relational_select_text("SELECT id FROM restore_target ORDER BY id")
            .unwrap()
            .rows;
        assert_eq!(rows, vec![vec![SqlValue::Int4(1)]]);

        engine.submit_transaction(93, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(93, parsed("TRUNCATE TABLE ONLY restore_target"))
            .unwrap();
        engine.submit_transaction(93, parsed("ROLLBACK")).unwrap();
        let rows = engine
            .execute_relational_select_text("SELECT id FROM restore_target ORDER BY id")
            .unwrap()
            .rows;
        assert_eq!(rows, vec![vec![SqlValue::Int4(1)]]);
    }

    #[test]
    fn transaction_truncate_restart_identity_remains_pre_effect() {
        let engine = Engine::new_local();
        engine
            .submit_transaction(94, parsed("CREATE TABLE restart_target (id int4)"))
            .unwrap();
        engine.submit_transaction(95, parsed("BEGIN")).unwrap();
        let wal_before = engine.durable_wal_records().len();

        let error = engine
            .submit_transaction(95, parsed("TRUNCATE restart_target RESTART IDENTITY"))
            .unwrap_err();
        assert!(matches!(error, ExecuteError::Unsupported(_)));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert!(engine
            .transaction_snapshot_handle(95)
            .unwrap()
            .transaction_delta_is_empty());
        engine.submit_transaction(95, parsed("ROLLBACK")).unwrap();
    }
}
