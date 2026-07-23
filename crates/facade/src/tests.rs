use super::*;
use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

#[derive(Clone, Copy)]
struct HarnessSessionId(u64);

struct MultiSessionHarness {
    shared: SharedEngine,
    engine: Arc<Engine>,
    sessions: BTreeMap<u64, SharedSession>,
    next_session_id: u64,
}

impl MultiSessionHarness {
    fn new() -> Self {
        let engine = Arc::new(Engine::new_local());
        Self {
            shared: SharedEngine::from_engine_arc(Arc::clone(&engine)),
            engine,
            sessions: BTreeMap::new(),
            next_session_id: 1,
        }
    }

    fn open_session(&mut self) -> HarnessSessionId {
        let id = self.next_session_id;
        self.next_session_id += 1;
        self.sessions.insert(id, self.shared.open_session());
        HarnessSessionId(id)
    }

    fn close_session(&mut self, id: HarnessSessionId) {
        if let Some(mut session) = self.sessions.remove(&id.0) {
            let _ = self
                .shared
                .submit(&mut session, SubmissionRequest::CloseSession);
        }
    }

    fn session_in_transaction(&self, id: HarnessSessionId) -> bool {
        self.sessions
            .get(&id.0)
            .is_some_and(SharedSession::in_transaction)
    }

    fn execute(&mut self, id: HarnessSessionId, sql: &str) -> Result<QueryOutcome, DbError> {
        let session = self.sessions.get_mut(&id.0).ok_or_else(|| DbError {
            category: ErrorCategory::Internal,
            message: format!("unknown session {}", id.0),
        })?;
        self.shared
            .submit(session, SubmissionRequest::Text(sql))
            .into_immediate()
    }

    fn execute_parameterized(
        &mut self,
        id: HarnessSessionId,
        sql: &str,
        params: &[DbValue],
    ) -> Result<QueryOutcome, DbError> {
        if !self.sessions.contains_key(&id.0) {
            return Err(DbError {
                category: ErrorCategory::Internal,
                message: format!("unknown session {}", id.0),
            });
        }
        let bound = PreparedStatement::parse(sql).and_then(|prepared| prepared.bind_values(params));
        let session = self
            .sessions
            .get_mut(&id.0)
            .expect("session existence checked before binding");
        match bound {
            Ok(bound) => self
                .shared
                .submit(session, SubmissionRequest::Prepared(&bound))
                .into_immediate(),
            Err(error) => {
                session.mark_transaction_failed();
                Err(error)
            }
        }
    }

    fn next_txn_id(&self) -> u64 {
        self.shared.next_txn_id.load(Ordering::Relaxed)
    }
}

fn submit_ephemeral_text(shared: &SharedEngine, sql: &str) -> Result<QueryOutcome, DbError> {
    let mut session = shared.open_session();
    let result = shared
        .submit(&mut session, SubmissionRequest::Text(sql))
        .into_immediate();
    let _ = shared.submit(&mut session, SubmissionRequest::CloseSession);
    result
}

fn submit_session_text(
    shared: &SharedEngine,
    session: &mut SharedSession,
    sql: &str,
) -> Result<QueryOutcome, DbError> {
    shared
        .submit(session, SubmissionRequest::Text(sql))
        .into_immediate()
}

#[test]
fn public_execution_surface_has_one_submission_boundary() {
    let facade_source = include_str!("lib.rs");
    let prepared_source = include_str!("prepared.rs");
    let batcher_source = include_str!("point_lookup_batcher.rs");
    let public_surface = [facade_source, prepared_source, batcher_source].concat();
    assert_eq!(facade_source.matches("pub fn submit(").count(), 1);
    for forbidden in [
        "pub struct EngineFacade",
        "pub struct SessionId",
        "pub fn execute_on_engine",
        "pub fn execute_on_shared_engine",
        "pub fn execute_on_shared_engine_session",
        "pub fn execute_on_shared_engine_batched",
        "pub fn execute_on_shared_engine_session_batched",
        "pub fn execute_prepared",
        "pub fn execute_prepared_on_shared_engine_session",
        "pub fn execute_concurrent_dml_with_prepared_hook",
        "pub fn execute_select_with_pinned_hook",
        "pub fn close_session",
        "pub fn enqueue(",
    ] {
        assert!(
            !public_surface.contains(forbidden),
            "superseded public facade execution entry returned: {forbidden}"
        );
    }
}

#[test]
fn batched_submission_rejects_a_batcher_from_another_engine_pre_effect() {
    let first = Arc::new(SharedEngine::new());
    let second = Arc::new(SharedEngine::new());
    let batcher =
        PointLookupBatcher::with_triggers(Arc::clone(&second), 1, std::time::Duration::ZERO);
    let mut session = first.open_session();
    let txn_before = first.next_txn_id.load(Ordering::Relaxed);
    let result = first.submit(
        &mut session,
        SubmissionRequest::BatchedText {
            sql: "SELECT * FROM missing",
            batcher: &batcher,
        },
    );
    let SubmissionDispatch::Immediate(Err(error)) = result else {
        panic!("cross-engine batcher must reject immediately");
    };
    assert_eq!(error.category, ErrorCategory::InvalidRequest);
    assert!(error.message.contains("different SharedEngine"));
    assert_eq!(first.next_txn_id.load(Ordering::Relaxed), txn_before);

    let rich = first
        .submit(
            &mut session,
            SubmissionRequest::BatchedText {
                sql: "SELECT c.oid FROM pg_catalog.pg_class c \
                      JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace",
                batcher: &batcher,
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(rich.category, ErrorCategory::InvalidRequest);
    assert!(rich.message.contains("different SharedEngine"));
    assert_eq!(first.next_txn_id.load(Ordering::Relaxed), txn_before);

    submit_session_text(&first, &mut session, "BEGIN").unwrap();
    assert!(submit_session_text(&first, &mut session, "SELEC broken").is_err());
    assert_eq!(
        session.transaction_status(),
        SessionTransactionStatus::FailedTransaction
    );
    let foreign_rollback = first
        .submit(
            &mut session,
            SubmissionRequest::BatchedText {
                sql: "ROLLBACK",
                batcher: &batcher,
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(foreign_rollback.category, ErrorCategory::InvalidRequest);
    assert_eq!(
        session.transaction_status(),
        SessionTransactionStatus::FailedTransaction
    );
    let rollback = submit_session_text(&first, &mut session, "ROLLBACK").unwrap();
    assert_eq!(
        rollback,
        QueryOutcome::Command {
            tag: CommandTag::Rollback,
            rows_affected: None,
        }
    );
    assert_eq!(session.transaction_status(), SessionTransactionStatus::Idle);
}

#[test]
fn instrumented_dml_respects_session_ownership_and_validates_before_txn_allocation() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_session_text(&shared, &mut session, "BEGIN").unwrap();

    let called = Arc::new(AtomicBool::new(false));
    let hook_called = Arc::clone(&called);
    let txn_before = shared.next_txn_id.load(Ordering::Relaxed);
    let error = shared
        .submit(
            &mut session,
            SubmissionRequest::InstrumentedDml {
                sql: "UPDATE missing SET value = 1",
                on_prepared: Box::new(move || hook_called.store(true, Ordering::Relaxed)),
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::Unsupported);
    assert_eq!(shared.next_txn_id.load(Ordering::Relaxed), txn_before);
    assert!(!called.load(Ordering::Relaxed));
    assert_eq!(
        session.transaction_status(),
        SessionTransactionStatus::FailedTransaction
    );

    let txn_before = shared.next_txn_id.load(Ordering::Relaxed);
    let error = shared
        .submit(
            &mut session,
            SubmissionRequest::InstrumentedDml {
                sql: "DELETE FROM missing",
                on_prepared: Box::new(|| panic!("failed session must not invoke hook")),
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::InFailedTransaction);
    assert_eq!(shared.next_txn_id.load(Ordering::Relaxed), txn_before);

    submit_session_text(&shared, &mut session, "ROLLBACK").unwrap();
    let txn_before = shared.next_txn_id.load(Ordering::Relaxed);
    let error = shared
        .submit(
            &mut session,
            SubmissionRequest::InstrumentedDml {
                sql: "CREATE TABLE must_not_exist (id INT)",
                on_prepared: Box::new(|| panic!("non-DML must not invoke hook")),
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::InvalidRequest);
    assert_eq!(shared.next_txn_id.load(Ordering::Relaxed), txn_before);
}

#[test]
fn instrumented_select_rejects_active_and_failed_sessions_without_reading() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_session_text(&shared, &mut session, "BEGIN").unwrap();

    let called = Arc::new(AtomicBool::new(false));
    let hook_called = Arc::clone(&called);
    let error = shared
        .submit(
            &mut session,
            SubmissionRequest::InstrumentedSelect {
                sql: "SELECT * FROM missing",
                on_pinned: Box::new(move || hook_called.store(true, Ordering::Relaxed)),
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::Unsupported);
    assert!(!called.load(Ordering::Relaxed));
    assert_eq!(
        session.transaction_status(),
        SessionTransactionStatus::FailedTransaction
    );

    let error = shared
        .submit(
            &mut session,
            SubmissionRequest::InstrumentedSelect {
                sql: "SELECT * FROM missing",
                on_pinned: Box::new(|| panic!("failed session must not pin or execute")),
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::InFailedTransaction);
}

#[test]
fn parameterized_execution_rejects_shape_before_claiming_a_transaction() {
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    let missing = facade
        .execute_parameterized(session, "SELECT $1", &[])
        .unwrap_err();
    assert_eq!(missing.category, ErrorCategory::Syntax);
    assert_eq!(
        facade.next_txn_id(),
        1,
        "parameter failure claims no txn id"
    );

    let quoted_only = facade
        .execute_parameterized(session, "SELECT '$1'", &[DbValue::Int4(7)])
        .unwrap_err();
    assert_eq!(quoted_only.category, ErrorCategory::Syntax);
    assert_eq!(facade.next_txn_id(), 1);

    let unknown = facade
        .execute_parameterized(HarnessSessionId(u64::MAX), "SELECT $1", &[])
        .unwrap_err();
    assert_eq!(unknown.category, ErrorCategory::Internal);
    assert!(unknown.message.contains("unknown session"));
}

#[test]
fn unsupported_copy_is_feature_not_supported_without_reclassifying_unknown_sql() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    let copy = submit_session_text(
        &shared,
        &mut session,
        "COPY people FROM STDIN WITH CSV HEADER DELIMITER ','",
    )
    .unwrap_err();
    assert_eq!(copy.category, ErrorCategory::Unsupported);

    let syntax = submit_session_text(&shared, &mut session, "SELEC broken").unwrap_err();
    assert_eq!(syntax.category, ErrorCategory::Syntax);
}

#[test]
fn transaction_error_enters_failed_state_and_commit_rolls_back() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_session_text(&shared, &mut session, "BEGIN").unwrap();
    assert_eq!(
        session.transaction_status(),
        SessionTransactionStatus::InTransaction
    );

    assert_eq!(
        submit_session_text(&shared, &mut session, "SELEC broken")
            .unwrap_err()
            .category,
        ErrorCategory::Syntax
    );
    assert_eq!(
        session.transaction_status(),
        SessionTransactionStatus::FailedTransaction
    );
    assert_eq!(
        submit_session_text(&shared, &mut session, "BEGIN")
            .unwrap_err()
            .category,
        ErrorCategory::InFailedTransaction
    );

    assert_eq!(
        submit_session_text(&shared, &mut session, "COMMIT").unwrap(),
        QueryOutcome::Command {
            tag: CommandTag::Rollback,
            rows_affected: None,
        }
    );
    assert_eq!(session.transaction_status(), SessionTransactionStatus::Idle);
}

#[test]
fn session_cleanup_and_compatibility_reads_do_not_enter_mutation_admission() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    let next_before = shared.next_txn_id.load(Ordering::Relaxed);

    assert!(matches!(
        submit_session_text(&shared, &mut session, "RESET ALL").unwrap(),
        QueryOutcome::Command { .. }
    ));
    assert_eq!(shared.next_txn_id.load(Ordering::Relaxed), next_before);

    submit_session_text(&shared, &mut session, "BEGIN").unwrap();
    let next_after_begin = shared.next_txn_id.load(Ordering::Relaxed);
    assert!(matches!(
        submit_session_text(&shared, &mut session, "SELECT pg_advisory_unlock_all()").unwrap(),
        QueryOutcome::Command { .. }
    ));
    for cleanup in ["CLOSE ALL", "UNLISTEN *", "RESET ALL"] {
        assert!(matches!(
            submit_session_text(&shared, &mut session, cleanup).unwrap(),
            QueryOutcome::Command { .. }
        ));
    }
    assert_eq!(
        session.transaction_status(),
        SessionTransactionStatus::InTransaction
    );
    assert_eq!(
        shared.next_txn_id.load(Ordering::Relaxed),
        next_after_begin,
        "effect-free session cleanup must not allocate a second transaction identity"
    );
    assert!(shared.engine.durable_wal_records().is_empty());

    submit_session_text(&shared, &mut session, "ROLLBACK").unwrap();
    assert_eq!(session.transaction_status(), SessionTransactionStatus::Idle);
    assert!(shared.engine.durable_wal_records().is_empty());
}

#[test]
fn pg_dump_session_controls_and_access_share_lock_are_wal_neutral() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_session_text(
        &shared,
        &mut session,
        "CREATE TABLE dump_lock_target (id INT PRIMARY KEY)",
    )
    .unwrap();
    let wal_before = shared.engine.durable_wal_records().len();
    let visible_before = shared.engine.visible_up_to();

    for sql in [
        "SET statement_timeout = 0",
        "SET search_path = pg_catalog, public",
        "SET standard_conforming_strings = on",
    ] {
        assert!(matches!(
            submit_session_text(&shared, &mut session, sql).unwrap(),
            QueryOutcome::Command { .. }
        ));
        assert_eq!(
            shared.engine.durable_wal_records().len(),
            wal_before,
            "{sql} claimed WAL"
        );
    }
    let outside = submit_session_text(
        &shared,
        &mut session,
        "LOCK TABLE dump_lock_target IN ACCESS SHARE MODE",
    )
    .unwrap_err();
    assert_eq!(outside.category, ErrorCategory::InvalidRequest);
    assert_eq!(shared.engine.durable_wal_records().len(), wal_before);

    submit_session_text(&shared, &mut session, "BEGIN").unwrap();
    assert_eq!(shared.engine.durable_wal_records().len(), wal_before);
    let next_after_begin = shared.next_txn_id.load(Ordering::Relaxed);
    submit_session_text(
        &shared,
        &mut session,
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY",
    )
    .unwrap();
    assert_eq!(shared.next_txn_id.load(Ordering::Relaxed), next_after_begin);
    assert_eq!(shared.engine.durable_wal_records().len(), wal_before);
    assert!(matches!(
        submit_session_text(
            &shared,
            &mut session,
            "LOCK TABLE dump_lock_target IN ACCESS SHARE MODE"
        )
        .unwrap(),
        QueryOutcome::Command {
            tag: CommandTag::Other(ref tag),
            ..
        } if tag == "LOCK TABLE"
    ));
    assert_eq!(shared.engine.durable_wal_records().len(), wal_before);
    assert_eq!(
        submit_session_text(&shared, &mut session, "SHOW TRANSACTION ISOLATION LEVEL").unwrap(),
        QueryOutcome::Rows {
            columns: vec![ColumnMeta {
                name: "transaction_isolation".to_string(),
                logical_type: LogicalType::Text,
            }],
            rows: vec![vec![DbValue::Text("repeatable read".to_string())]],
        }
    );
    submit_session_text(&shared, &mut session, "ROLLBACK").unwrap();

    assert_eq!(shared.engine.durable_wal_records().len(), wal_before);
    assert_eq!(shared.engine.visible_up_to(), visible_before);
    assert_eq!(session.transaction_status(), SessionTransactionStatus::Idle);
}

#[test]
fn rejected_set_transaction_preserves_the_original_failed_block_and_never_autocommits() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_session_text(
        &shared,
        &mut session,
        "CREATE TABLE set_txn_atomicity (id INT PRIMARY KEY)",
    )
    .unwrap();
    let wal_before = shared.engine.durable_wal_records();

    submit_session_text(&shared, &mut session, "BEGIN").unwrap();
    let next_after_begin = shared.next_txn_id.load(Ordering::Relaxed);
    let unsupported = submit_session_text(
        &shared,
        &mut session,
        "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE",
    )
    .unwrap_err();
    assert_eq!(unsupported.category, ErrorCategory::Unsupported);
    assert_eq!(
        session.transaction_status(),
        SessionTransactionStatus::FailedTransaction
    );
    assert_eq!(shared.next_txn_id.load(Ordering::Relaxed), next_after_begin);
    assert_eq!(shared.engine.durable_wal_records(), wal_before);

    let blocked = submit_session_text(
        &shared,
        &mut session,
        "INSERT INTO set_txn_atomicity VALUES (1)",
    )
    .unwrap_err();
    assert_eq!(blocked.category, ErrorCategory::InFailedTransaction);
    assert_eq!(shared.engine.durable_wal_records(), wal_before);
    submit_session_text(&shared, &mut session, "ROLLBACK").unwrap();

    let QueryOutcome::Rows { rows, .. } =
        submit_ephemeral_text(&shared, "SELECT id FROM set_txn_atomicity WHERE id = 1").unwrap()
    else {
        panic!("verification SELECT must return rows");
    };
    assert!(
        rows.is_empty(),
        "rejected SET allowed a later write to autocommit"
    );

    submit_session_text(&shared, &mut session, "BEGIN").unwrap();
    let deferrable = submit_session_text(
        &shared,
        &mut session,
        "SET TRANSACTION READ ONLY, DEFERRABLE",
    )
    .unwrap_err();
    assert_eq!(deferrable.category, ErrorCategory::Unsupported);
    assert_eq!(
        session.transaction_status(),
        SessionTransactionStatus::FailedTransaction
    );
    assert_eq!(shared.engine.durable_wal_records(), wal_before);
    submit_session_text(&shared, &mut session, "ROLLBACK").unwrap();
}

#[test]
fn show_transaction_isolation_reports_session_state_without_claiming_work() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    let next_before = shared.next_txn_id.load(Ordering::Relaxed);

    let expected = |value: &str| QueryOutcome::Rows {
        columns: vec![ColumnMeta {
            name: "transaction_isolation".to_string(),
            logical_type: LogicalType::Text,
        }],
        rows: vec![vec![DbValue::Text(value.to_string())]],
    };
    assert_eq!(
        submit_session_text(&shared, &mut session, "SHOW TRANSACTION ISOLATION LEVEL").unwrap(),
        expected("read committed")
    );
    assert_eq!(shared.next_txn_id.load(Ordering::Relaxed), next_before);

    submit_session_text(
        &shared,
        &mut session,
        "BEGIN ISOLATION LEVEL REPEATABLE READ",
    )
    .unwrap();
    let next_after_begin = shared.next_txn_id.load(Ordering::Relaxed);
    assert_eq!(
        submit_session_text(&shared, &mut session, "SHOW transaction_isolation").unwrap(),
        expected("repeatable read")
    );
    assert_eq!(shared.next_txn_id.load(Ordering::Relaxed), next_after_begin);

    submit_session_text(&shared, &mut session, "COMMIT AND CHAIN").unwrap();
    assert_eq!(
        submit_session_text(&shared, &mut session, "SHOW TRANSACTION ISOLATION LEVEL").unwrap(),
        expected("repeatable read")
    );
    submit_session_text(&shared, &mut session, "ROLLBACK AND CHAIN").unwrap();
    assert_eq!(
        submit_session_text(&shared, &mut session, "SHOW TRANSACTION ISOLATION LEVEL").unwrap(),
        expected("repeatable read")
    );
    submit_session_text(&shared, &mut session, "COMMIT").unwrap();
    assert_eq!(
        submit_session_text(&shared, &mut session, "SHOW TRANSACTION ISOLATION LEVEL").unwrap(),
        expected("read committed")
    );

    submit_session_text(
        &shared,
        &mut session,
        "BEGIN ISOLATION LEVEL READ UNCOMMITTED",
    )
    .unwrap();
    assert_eq!(
        submit_session_text(&shared, &mut session, "SHOW TRANSACTION ISOLATION LEVEL").unwrap(),
        expected("read committed")
    );
    submit_session_text(&shared, &mut session, "ROLLBACK").unwrap();

    submit_session_text(
        &shared,
        &mut session,
        "BEGIN ISOLATION LEVEL REPEATABLE READ",
    )
    .unwrap();
    let _relation_error = submit_session_text(
        &shared,
        &mut session,
        "SELECT id FROM missing_isolation_relation",
    )
    .unwrap_err();
    let blocked =
        submit_session_text(&shared, &mut session, "SHOW TRANSACTION ISOLATION LEVEL").unwrap_err();
    assert_eq!(blocked.category, ErrorCategory::InFailedTransaction);
    // PostgreSQL treats COMMIT in an aborted block as rollback; AND CHAIN must retain the
    // characteristics of that failed block for the successor transaction.
    assert!(matches!(
        submit_session_text(&shared, &mut session, "COMMIT AND CHAIN").unwrap(),
        QueryOutcome::Command {
            tag: CommandTag::Rollback,
            ..
        }
    ));
    assert_eq!(
        submit_session_text(&shared, &mut session, "SHOW TRANSACTION ISOLATION LEVEL").unwrap(),
        expected("repeatable read")
    );
    submit_session_text(&shared, &mut session, "ROLLBACK").unwrap();
    assert_eq!(
        submit_session_text(&shared, &mut session, "SHOW TRANSACTION ISOLATION LEVEL").unwrap(),
        expected("read committed")
    );
    assert!(shared.engine.durable_wal_records().is_empty());
}

#[test]
fn batched_text_fallback_preserves_session_metadata_reads() {
    let shared = Arc::new(SharedEngine::new());
    let batcher = PointLookupBatcher::with_triggers(
        Arc::clone(&shared),
        8,
        std::time::Duration::from_secs(1),
    );
    let mut session = shared.open_session();
    let outcome = shared
        .submit(
            &mut session,
            SubmissionRequest::BatchedText {
                sql: "SHOW TRANSACTION ISOLATION LEVEL",
                batcher: &batcher,
            },
        )
        .into_immediate()
        .unwrap();
    assert!(matches!(
        outcome,
        QueryOutcome::Rows { rows, .. }
            if rows == vec![vec![DbValue::Text("read committed".to_string())]]
    ));
    assert!(shared.engine.durable_wal_records().is_empty());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn parameterized_w1_update_preserves_types_and_returning_rows() {
    let mut facade = MultiSessionHarness::new();
    facade.engine.set_auto_admit_on_commit(true);
    let session = facade.open_session();
    facade
        .execute(
            session,
            "CREATE TABLE accounts (tenant_id int4, account_id int8, balance_cents int8, \
                 version int8, status int2, PRIMARY KEY (tenant_id, account_id))",
        )
        .unwrap();
    facade
        .execute_parameterized(
            session,
            "INSERT INTO accounts VALUES ($1, $2, $3, $4, $5)",
            &[
                DbValue::Int4(7),
                DbValue::Int8(70_001),
                DbValue::Int8(1_000),
                DbValue::Int8(0),
                DbValue::Int2(1),
            ],
        )
        .unwrap();
    let outcome = facade
        .execute_parameterized(
            session,
            "UPDATE accounts SET balance_cents = balance_cents + $3, version = version + 1 \
                 WHERE tenant_id = $1 AND account_id = $2 RETURNING balance_cents, version",
            &[DbValue::Int4(7), DbValue::Int8(70_001), DbValue::Int8(-25)],
        )
        .unwrap();
    assert_eq!(
        outcome,
        QueryOutcome::Returning {
            tag: CommandTag::Update,
            columns: vec![
                ColumnMeta {
                    name: "balance_cents".to_string(),
                    logical_type: LogicalType::Int8,
                },
                ColumnMeta {
                    name: "version".to_string(),
                    logical_type: LogicalType::Int8,
                },
            ],
            rows: vec![vec![DbValue::Int8(975), DbValue::Int8(1)]],
            rows_affected: 1,
        }
    );
    assert_eq!(pg_adapter::command_complete_tag(&outcome), "UPDATE 1");
}

#[test]
fn durable_wal_segment_env_decode_is_unset_for_missing_or_blank() {
    // D4 config decode (pure — no process-global env mutation in parallel tests): unset and
    // blank values keep the in-memory default; anything else is the durable segment path.
    assert_eq!(durable_wal_segment_from_env(None), None);
    assert_eq!(
        durable_wal_segment_from_env(Some(std::ffi::OsStr::new(""))),
        None
    );
    assert_eq!(
        durable_wal_segment_from_env(Some(std::ffi::OsStr::new("   "))),
        None
    );
    assert_eq!(
        durable_wal_segment_from_env(Some(std::ffi::OsStr::new("/var/lib/gpu-db/wal.segment"))),
        Some(std::path::PathBuf::from("/var/lib/gpu-db/wal.segment"))
    );
}

#[test]
fn new_durable_shared_engine_fsyncs_commits_and_recovers_after_restart() {
    // D4: the served engine, when configured durable, survives a "crash" (drop + reopen at
    // the same segment path) with its committed writes intact.
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-facade-durable-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let segment_path = dir.join("served.wal");

    let engine = SharedEngine::new_durable(&segment_path).unwrap();
    assert!(engine.is_durable());
    submit_ephemeral_text(&engine, "CREATE TABLE t (id INT)").unwrap();
    submit_ephemeral_text(&engine, "INSERT INTO t (id) VALUES (7)").unwrap();
    drop(engine);

    let recovered = SharedEngine::new_durable(&segment_path).unwrap();
    let outcome = submit_ephemeral_text(&recovered, "SELECT id FROM t").unwrap();
    let QueryOutcome::Rows { rows, .. } = outcome else {
        panic!("expected rows after durable recovery");
    };
    assert_eq!(rows, vec![vec![DbValue::Int4(7)]]);
    // The in-memory default remains non-durable (the assessment's D4 gap, now a setting).
    assert!(!SharedEngine::new().is_durable());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn durable_typed_copy_recovers_through_the_canonical_facade_boundary() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-facade-copy-recovery-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let segment_path = dir.join("copy.wal");
    let copy = CopyFromStdin {
        table: "copy_recovery".to_string(),
        columns: Some(vec!["id".to_string(), "name".to_string()]),
        options: gpu_db_sql::CopyOptions::CSV,
    };

    let shared = SharedEngine::new_durable(&segment_path).unwrap();
    let mut session = shared.open_session();
    submit_session_text(
        &shared,
        &mut session,
        "CREATE TABLE copy_recovery (id INT, name TEXT)",
    )
    .unwrap();
    let QueryOutcome::CopyIn { target } = shared
        .submit(&mut session, SubmissionRequest::CopyFromStart(&copy))
        .into_immediate()
        .unwrap()
    else {
        panic!("COPY start must return a target proof")
    };
    let wal_before = shared.read_engine().unwrap().durable_wal_records().len();
    assert_eq!(
        shared
            .submit(
                &mut session,
                SubmissionRequest::CopyFrom {
                    target: &target,
                    rows: vec![
                        vec![DbValue::Int4(1), DbValue::Null],
                        vec![DbValue::Int4(2), DbValue::Text(String::new())],
                    ],
                },
            )
            .into_immediate()
            .unwrap(),
        QueryOutcome::Command {
            tag: CommandTag::Copy,
            rows_affected: Some(2),
        }
    );
    assert_eq!(
        shared.read_engine().unwrap().durable_wal_records().len(),
        wal_before + 1,
        "one COPY request appends one canonical WAL record"
    );
    drop(session);
    drop(shared);

    let recovered = SharedEngine::new_durable(&segment_path).unwrap();
    let outcome =
        submit_ephemeral_text(&recovered, "SELECT id, name FROM copy_recovery ORDER BY id")
            .unwrap();
    let QueryOutcome::Rows { rows, .. } = outcome else {
        panic!("expected recovered COPY rows")
    };
    assert_eq!(
        rows,
        vec![
            vec![DbValue::Int4(1), DbValue::Null],
            vec![DbValue::Int4(2), DbValue::Text(String::new())],
        ]
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn null_maps_through_the_value_model_to_a_wire_null() {
    // M3 slice 1: SqlValue::Null → DbValue::Null → the wire boundary returns `None`
    // (the protocol's `-1` DataRow field length), while every typed value still
    // renders to `Some(text)`. This closes the engine→wire NULL path.
    assert_eq!(map_value(SqlValue::Null), DbValue::Null);
    assert_eq!(pg_adapter::db_value_text_opt(&DbValue::Null), None);
    assert_eq!(
        pg_adapter::db_value_text_opt(&DbValue::Int4(42)),
        Some("42".to_string())
    );
    assert_eq!(
        pg_adapter::db_value_text_opt(&DbValue::Text("x".to_string())),
        Some("x".to_string())
    );
    // A non-null value's text encoding is unchanged by the new boundary.
    assert_eq!(pg_adapter::db_value_text(&DbValue::Bool(true)), "t");
}

#[test]
fn relational_lifecycle_round_trips_through_facade() {
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    facade
        .execute(session, "CREATE TABLE accounts (id INT, name TEXT)")
        .unwrap();
    facade
        .execute(
            session,
            "INSERT INTO accounts (id, name) VALUES (1, 'alice')",
        )
        .unwrap();
    facade
        .execute(session, "INSERT INTO accounts (id, name) VALUES (2, 'bob')")
        .unwrap();

    let outcome = facade
        .execute(session, "SELECT id, name FROM accounts WHERE id = 1")
        .unwrap();
    match outcome {
        QueryOutcome::Rows { columns, rows } => {
            let column_shape: Vec<(&str, LogicalType)> = columns
                .iter()
                .map(|column| (column.name.as_str(), column.logical_type))
                .collect();
            assert_eq!(
                column_shape,
                vec![("id", LogicalType::Int4), ("name", LogicalType::Text)]
            );
            assert_eq!(
                rows,
                vec![vec![DbValue::Int4(1), DbValue::Text("alice".to_string())]]
            );
        }
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn count_aggregate_round_trips_as_a_neutral_integer() {
    // Finding (recorded in the P0-M1 milestone report): this engine returns
    // `COUNT(*)` as `Int4`, not `Int8`. The façade faithfully surfaces what the
    // engine produces; it does not invent wider aggregate typing. Richer
    // aggregate result types (int8/numeric for COUNT/SUM) are part of the
    // Phase 3 type-system work, at which point this test's expectation widens.
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    facade.execute(session, "CREATE TABLE t (a INT)").unwrap();
    facade
        .execute(session, "INSERT INTO t (a) VALUES (1)")
        .unwrap();
    facade
        .execute(session, "INSERT INTO t (a) VALUES (2)")
        .unwrap();

    let outcome = facade.execute(session, "SELECT COUNT(*) FROM t").unwrap();
    match outcome {
        QueryOutcome::Rows { rows, .. } => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].len(), 1);
            let count = match &rows[0][0] {
                DbValue::Int4(value) => i64::from(*value),
                DbValue::Int8(value) => *value,
                other => panic!("expected an integer count, got {other:?}"),
            };
            assert_eq!(count, 2);
        }
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn select_from_unknown_table_returns_neutral_error() {
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    let error = facade
        .execute(session, "SELECT id FROM missing_table")
        .unwrap_err();
    assert!(!error.message.is_empty());
    // The neutral error maps to a SQLSTATE only at the adapter; no SQLSTATE
    // ever appears in the façade type itself.
    let sqlstate = pg_adapter::error_sqlstate(error.category);
    assert_eq!(sqlstate.len(), 5);
}

#[test]
fn command_tags_are_neutral_until_adapter_formats_them() {
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    let outcome = facade.execute(session, "CREATE TABLE t (a INT)").unwrap();
    assert_eq!(
        outcome,
        QueryOutcome::Command {
            tag: CommandTag::CreateTable,
            rows_affected: None
        }
    );
    assert_eq!(pg_adapter::command_complete_tag(&outcome), "CREATE TABLE");

    let parsed_truncate = parse_command("TRUNCATE TABLE t CONTINUE IDENTITY").unwrap();
    let truncate = command_tag(&parsed_truncate);
    assert_eq!(truncate, CommandTag::Truncate);
    assert_eq!(
        pg_adapter::command_complete_tag(&QueryOutcome::Command {
            tag: truncate,
            rows_affected: None,
        }),
        "TRUNCATE TABLE"
    );
}

#[test]
fn sessions_track_transaction_state_independent_of_connection() {
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    assert!(!facade.session_in_transaction(session));
    facade.execute(session, "BEGIN").unwrap();
    assert!(facade.session_in_transaction(session));
    assert_eq!(facade.engine.active_txn_count(), 1);
    facade.execute(session, "COMMIT").unwrap();
    assert!(!facade.session_in_transaction(session));
    assert_eq!(facade.engine.active_txn_count(), 0);
}

#[test]
fn session_close_rolls_back_engine_transaction_context() {
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    facade.execute(session, "BEGIN").unwrap();
    assert_eq!(facade.engine.active_txn_count(), 1);

    facade.close_session(session);

    assert!(!facade.session_in_transaction(session));
    assert_eq!(facade.engine.active_txn_count(), 0);
}

#[test]
fn session_and_chain_transfers_to_a_known_engine_transaction() {
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    facade.execute(session, "BEGIN").unwrap();

    facade.execute(session, "COMMIT AND CHAIN").unwrap();
    assert!(facade.session_in_transaction(session));
    assert_eq!(facade.engine.active_txn_count(), 1);

    facade.execute(session, "ROLLBACK").unwrap();
    assert!(!facade.session_in_transaction(session));
    assert_eq!(facade.engine.active_txn_count(), 0);
}

#[test]
fn active_transaction_stages_create_table_until_rollback() {
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    facade.execute(session, "BEGIN").unwrap();
    facade
        .execute(session, "CREATE TABLE must_not_autocommit (id INT)")
        .unwrap();
    assert!(facade.session_in_transaction(session));
    let QueryOutcome::Rows { rows, .. } = facade
        .execute(session, "SELECT id FROM must_not_autocommit")
        .unwrap()
    else {
        panic!("the creating transaction must read its private catalog");
    };
    assert!(rows.is_empty());
    let QueryOutcome::Rows { rows, .. } = facade
        .execute(session, "SELECT COUNT(*) FROM must_not_autocommit")
        .unwrap()
    else {
        panic!("the private empty relation must preserve scalar aggregate semantics");
    };
    assert_eq!(rows, vec![vec![DbValue::Int8(0)]]);

    let observer = facade.open_session();
    let hidden = facade
        .execute(observer, "SELECT id FROM must_not_autocommit")
        .unwrap_err();
    assert!(hidden.message.contains("must_not_autocommit"));

    facade.execute(session, "ROLLBACK").unwrap();
    facade
        .execute(session, "CREATE TABLE must_not_autocommit (id INT)")
        .unwrap();
}

#[test]
fn shared_active_transaction_publishes_create_table_only_at_commit() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_session_text(&shared, &mut session, "BEGIN").unwrap();
    submit_session_text(
        &shared,
        &mut session,
        "CREATE TABLE must_not_autocommit (id INT)",
    )
    .unwrap();
    assert!(session.in_transaction());

    let hidden = submit_ephemeral_text(&shared, "SELECT id FROM must_not_autocommit").unwrap_err();
    assert!(hidden.message.contains("must_not_autocommit"));

    submit_session_text(&shared, &mut session, "COMMIT").unwrap();
    let QueryOutcome::Rows { rows, .. } =
        submit_ephemeral_text(&shared, "SELECT id FROM must_not_autocommit").unwrap()
    else {
        panic!("committed relation must be queryable");
    };
    assert!(rows.is_empty());
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn general_gpu_catalog_select_retains_transaction_snapshot_and_failed_state() {
    let shared = SharedEngine::new();
    let mut creator = shared.open_session();
    submit_session_text(&shared, &mut creator, "BEGIN").unwrap();
    submit_session_text(
        &shared,
        &mut creator,
        "CREATE TABLE private_catalog_join_target (id INT)",
    )
    .unwrap();
    let lookup = "SELECT c.relname, n.nspname \
                  FROM pg_catalog.pg_class c \
                  JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
                  WHERE c.relname = 'private_catalog_join_target'";
    let QueryOutcome::Rows { rows, .. } =
        submit_session_text(&shared, &mut creator, lookup).unwrap()
    else {
        panic!("transactional catalog lookup must return rows");
    };
    assert_eq!(
        rows,
        vec![vec![
            DbValue::Text("private_catalog_join_target".to_string()),
            DbValue::Text("public".to_string()),
        ]]
    );

    let hidden = submit_ephemeral_text(&shared, lookup).unwrap();
    let QueryOutcome::Rows { rows, .. } = hidden else {
        panic!("observer catalog lookup must return a row set");
    };
    assert!(
        rows.is_empty(),
        "private catalog state escaped its transaction"
    );

    let error = submit_session_text(
        &shared,
        &mut creator,
        "SELECT pg_catalog.pg_get_userbyid(c.oid + 1) \
         FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n \
         ON n.oid = c.relnamespace",
    )
    .unwrap_err();
    assert_ne!(error.category, ErrorCategory::InFailedTransaction);
    assert_eq!(
        creator.transaction_status(),
        SessionTransactionStatus::FailedTransaction
    );
    let blocked = submit_session_text(&shared, &mut creator, lookup).unwrap_err();
    assert_eq!(blocked.category, ErrorCategory::InFailedTransaction);
    submit_session_text(&shared, &mut creator, "ROLLBACK").unwrap();

    submit_session_text(
        &shared,
        &mut creator,
        "CREATE TABLE pg_type (oid INT PRIMARY KEY)",
    )
    .unwrap();
    submit_session_text(&shared, &mut creator, "INSERT INTO pg_type VALUES (9001)").unwrap();
    let public_prepared = shared
        .prepare_statement(&creator, "SELECT oid FROM public.pg_type ORDER BY oid", &[])
        .unwrap();
    let public_bound = public_prepared.bind_values(&[]).unwrap();
    let QueryOutcome::Rows { rows, .. } = shared
        .submit(&mut creator, SubmissionRequest::Prepared(&public_bound))
        .into_immediate()
        .unwrap()
    else {
        panic!("explicit public prepared lookup must produce user rows");
    };
    assert_eq!(rows, vec![vec![DbValue::Int4(9001)]]);

    let prepared = shared
        .prepare_statement(
            &creator,
            "SELECT oid, * FROM pg_catalog.pg_type WHERE typname = $1",
            &[Some(LogicalType::Text)],
        )
        .unwrap();
    let bound = prepared
        .bind_values(&[DbValue::Text("int4".to_string())])
        .unwrap();
    let QueryOutcome::Rows { columns, rows } = shared
        .submit(&mut creator, SubmissionRequest::Prepared(&bound))
        .into_immediate()
        .unwrap()
    else {
        panic!("bound catalog AST must produce rows");
    };
    assert_eq!(
        columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        vec!["oid", "oid", "typname", "typlen", "typtype", "typnamespace"]
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], rows[0][1]);
    assert_eq!(rows[0][2], DbValue::Text("int4".to_string()));
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn successful_rich_select_prevents_late_set_transaction_snapshot_replacement() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_session_text(
        &shared,
        &mut session,
        "CREATE TABLE rich_statement_marker (id INT)",
    )
    .unwrap();
    submit_session_text(&shared, &mut session, "BEGIN").unwrap();
    let QueryOutcome::Rows { rows, .. } = submit_session_text(
        &shared,
        &mut session,
        "SELECT c.relname, n.nspname FROM pg_catalog.pg_class c \
         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
         WHERE c.relname = 'rich_statement_marker'",
    )
    .unwrap() else {
        panic!("rich catalog SELECT must return rows");
    };
    assert_eq!(rows.len(), 1);

    let late = submit_session_text(
        &shared,
        &mut session,
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ",
    )
    .unwrap_err();
    assert_eq!(late.category, ErrorCategory::InvalidRequest);
    assert!(late
        .message
        .contains("must precede the first transaction statement"));
    assert_eq!(
        session.transaction_status(),
        SessionTransactionStatus::FailedTransaction
    );
    submit_session_text(&shared, &mut session, "ROLLBACK").unwrap();
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn pg_dump_sequence_state_routing_uses_the_pinned_catalog_and_falls_through_for_tables() {
    let shared = SharedEngine::new();
    let mut setup = shared.open_session();
    submit_session_text(&shared, &mut setup, "CREATE SEQUENCE snapshot_sequence").unwrap();
    submit_session_text(
        &shared,
        &mut setup,
        "SELECT nextval('snapshot_sequence'::regclass)",
    )
    .unwrap();
    submit_session_text(
        &shared,
        &mut setup,
        "SELECT nextval('snapshot_sequence'::regclass)",
    )
    .unwrap();

    let mut repeatable = shared.open_session();
    submit_session_text(
        &shared,
        &mut repeatable,
        "BEGIN ISOLATION LEVEL REPEATABLE READ",
    )
    .unwrap();
    let old_state = submit_session_text(
        &shared,
        &mut repeatable,
        "SELECT last_value, is_called FROM public.snapshot_sequence",
    )
    .unwrap();
    assert!(matches!(
        old_state,
        QueryOutcome::Rows { ref rows, .. }
            if rows == &vec![vec![DbValue::Int8(2), DbValue::Bool(true)]]
    ));

    submit_session_text(&shared, &mut setup, "DROP SEQUENCE snapshot_sequence").unwrap();
    submit_session_text(&shared, &mut setup, "CREATE SEQUENCE snapshot_sequence").unwrap();
    submit_session_text(
        &shared,
        &mut setup,
        "SELECT nextval('snapshot_sequence'::regclass)",
    )
    .unwrap();
    assert_eq!(
        submit_session_text(
            &shared,
            &mut repeatable,
            "SELECT last_value, is_called FROM public.snapshot_sequence",
        )
        .unwrap(),
        old_state,
        "repeatable-read sequence routing consulted the live DROP/recreate catalog"
    );
    submit_session_text(&shared, &mut repeatable, "COMMIT").unwrap();

    submit_session_text(
        &shared,
        &mut setup,
        "CREATE TABLE sequence_shape_table (last_value BIGINT, is_called BOOL)",
    )
    .unwrap();
    submit_session_text(
        &shared,
        &mut setup,
        "INSERT INTO sequence_shape_table VALUES (41, TRUE)",
    )
    .unwrap();
    assert!(matches!(
        submit_session_text(
            &shared,
            &mut setup,
            "SELECT last_value, is_called FROM public.sequence_shape_table",
        )
        .unwrap(),
        QueryOutcome::Rows { rows, .. }
            if rows == vec![vec![DbValue::Int8(41), DbValue::Bool(true)]]
    ));
}

#[test]
fn prepared_public_catalog_lookalike_fails_closed_and_survives_drop_aba() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    let sql = "SELECT oid FROM public.pg_type ORDER BY oid";
    let missing = shared
        .prepare_statement(&session, sql, &[])
        .expect_err("missing explicit public relation must fail during Parse/Describe");
    assert!(missing.message.contains("public.pg_type"), "{missing:?}");

    submit_session_text(&shared, &mut session, "CREATE TABLE pg_type (oid INT)").unwrap();
    let prepared = shared.prepare_statement(&session, sql, &[]).unwrap();
    let bound = prepared.bind_values(&[]).unwrap();
    submit_session_text(&shared, &mut session, "DROP TABLE pg_type").unwrap();
    let dropped = shared
        .submit(&mut session, SubmissionRequest::Prepared(&bound))
        .into_immediate()
        .expect_err("drop must not rebind public.pg_type to pg_catalog.pg_type");
    assert!(dropped.message.contains("public.pg_type"), "{dropped:?}");
}

#[test]
fn catalog_lookup_fails_closed_for_user_aliases_and_non_table_public_shadows() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_session_text(
        &shared,
        &mut session,
        "CREATE TABLE catalog_shadow_source (oid INT)",
    )
    .unwrap();
    submit_session_text(
        &shared,
        &mut session,
        "INSERT INTO catalog_shadow_source VALUES (4242)",
    )
    .unwrap();
    submit_session_text(
        &shared,
        &mut session,
        "CREATE VIEW pg_class AS SELECT oid FROM catalog_shadow_source",
    )
    .unwrap();
    submit_session_text(
        &shared,
        &mut session,
        "CREATE VIEW pg_namespace AS SELECT oid FROM catalog_shadow_source",
    )
    .unwrap();
    submit_session_text(
        &shared,
        &mut session,
        "CREATE MATERIALIZED VIEW pg_type AS SELECT oid FROM catalog_shadow_source WITH NO DATA",
    )
    .unwrap();
    submit_session_text(&shared, &mut session, "CREATE SEQUENCE pg_policy").unwrap();
    for index in ["pg_attribute", "pg_trigger", "pg_am"] {
        submit_session_text(
            &shared,
            &mut session,
            &format!("CREATE INDEX {index} ON catalog_shadow_source (oid)"),
        )
        .unwrap();
    }

    let alias_error = submit_session_text(
        &shared,
        &mut session,
        "SELECT oid FROM catalog_shadow_source AS s(alias_oid) WHERE oid + 0 > 0",
    )
    .expect_err("resident user alias lists must reject before exposing hidden names");
    assert!(
        alias_error.message.contains("column-alias"),
        "{alias_error:?}"
    );

    for sql in [
        "SELECT oid FROM pg_class WHERE oid + 0 > 0",
        "SELECT attrelid FROM pg_attribute WHERE attrelid + 0 > 0",
        "SELECT oid FROM pg_policy",
        "SELECT oid FROM pg_trigger",
        "SELECT c.oid, n.oid FROM pg_catalog.pg_class c JOIN pg_namespace n \
         ON c.relnamespace = n.oid",
        "SELECT c.oid, a.oid FROM pg_catalog.pg_class c JOIN pg_am a ON c.relam = a.oid",
    ] {
        submit_session_text(&shared, &mut session, sql)
            .expect_err("a bare public relation or index owner must block catalog synthesis");
    }

    for sql in [
        "SELECT oid FROM pg_class",
        "SELECT oid FROM pg_type",
        "SELECT oid FROM pg_policy",
        "SELECT oid FROM pg_attribute",
        "SELECT oid FROM pg_trigger",
    ] {
        shared.prepare_statement(&session, sql, &[]).expect_err(
            "prepared Describe must not synthesize behind a public relation or index owner",
        );
    }

    let QueryOutcome::Rows { rows, .. } =
        submit_session_text(&shared, &mut session, "SELECT * FROM public.pg_class").unwrap()
    else {
        panic!("explicit public view lookup must return rows");
    };
    assert_eq!(rows, vec![vec![DbValue::Int4(4242)]]);

    let QueryOutcome::Rows { rows, .. } = submit_session_text(
        &shared,
        &mut session,
        "SELECT oid FROM pg_catalog.pg_type WHERE oid + 0 > 0 ORDER BY oid LIMIT 1",
    )
    .unwrap() else {
        panic!("explicit system lookup must remain catalog-bound");
    };
    assert_eq!(rows, vec![vec![DbValue::Int4(16)]]);
}

#[test]
fn typed_catalog_selects_cannot_bypass_general_aggregate_binding() {
    let shared = Arc::new(SharedEngine::new());
    let sql = "SELECT count(*) FROM pg_catalog.pg_policy ORDER BY oid";
    let error = submit_ephemeral_text(&shared, sql)
        .expect_err("invalid catalog aggregate must fail at the general binder");
    assert!(!error.message.contains("GPU execution is required"));

    let batcher = PointLookupBatcher::with_triggers(
        Arc::clone(&shared),
        8,
        std::time::Duration::from_secs(1),
    );
    let mut session = shared.open_session();
    let sql = "SELECT 'not-an-integer'::int4 FROM pg_catalog.pg_policy";
    let error = shared
        .submit(
            &mut session,
            SubmissionRequest::BatchedText {
                sql,
                batcher: &batcher,
            },
        )
        .into_immediate()
        .expect_err("canonical batched text must retain catalog binding");
    assert!(
        !error.message.contains("GPU execution is required"),
        "`{sql}` escaped binding and reached device execution: {error:?}"
    );
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn gpu_rich_batched_catalog_fallback_checks_engine_owner_before_dispatch() {
    let shared = Arc::new(SharedEngine::new());
    let mut session = shared.open_session();
    submit_session_text(
        &shared,
        &mut session,
        "CREATE TABLE rich_batch_catalog_control (id INT)",
    )
    .unwrap();
    let local = PointLookupBatcher::with_triggers(
        Arc::clone(&shared),
        8,
        std::time::Duration::from_secs(1),
    );
    let sql = "SELECT c.relname, n.nspname FROM pg_catalog.pg_class c \
               JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
               WHERE c.relname = 'rich_batch_catalog_control'";
    let QueryOutcome::Rows { rows, .. } = shared
        .submit(
            &mut session,
            SubmissionRequest::BatchedText {
                sql,
                batcher: &local,
            },
        )
        .into_immediate()
        .unwrap()
    else {
        panic!("owned rich catalog fallback must produce rows");
    };
    assert_eq!(
        rows,
        vec![vec![
            DbValue::Text("rich_batch_catalog_control".to_string()),
            DbValue::Text("public".to_string()),
        ]]
    );

    let other = Arc::new(SharedEngine::new());
    let foreign = PointLookupBatcher::with_triggers(other, 8, std::time::Duration::from_secs(1));
    let error = shared
        .submit(
            &mut session,
            SubmissionRequest::BatchedText {
                sql,
                batcher: &foreign,
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::InvalidRequest);
    assert!(error.message.contains("different SharedEngine"));
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn facade_gpu_catalog_aggregate_alias_and_grouping_share_general_binding() {
    let shared = SharedEngine::new();
    for sql in [
        "SELECT table_name FROM information_schema.tables ORDER BY table_name",
        "SELECT column_name FROM information_schema.columns ORDER BY column_name",
    ] {
        let QueryOutcome::Rows { rows, .. } = submit_ephemeral_text(&shared, sql).unwrap() else {
            panic!("empty information_schema query must return rows");
        };
        assert!(rows.is_empty(), "{sql}");
    }
    let QueryOutcome::Rows { columns, rows } = submit_ephemeral_text(
        &shared,
        "SELECT count(*) AS oid FROM pg_catalog.pg_policy ORDER BY oid",
    )
    .expect("a scalar aggregate may order by its output alias") else {
        panic!("catalog aggregate must return rows");
    };
    assert_eq!(columns[0].name, "oid");
    assert_eq!(rows, vec![vec![DbValue::Int8(0)]]);

    let QueryOutcome::Rows { rows, .. } = submit_ephemeral_text(
        &shared,
        "SELECT oid, count(*) FROM pg_catalog.pg_policy GROUP BY oid",
    )
    .expect("a valid grouped empty catalog query must not enter the typed aggregate path") else {
        panic!("grouped catalog query must return rows");
    };
    assert!(rows.is_empty());
}

#[test]
fn transactional_unsupported_catalog_family_enters_failed_state_and_rollback_discards_catalog() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_session_text(&shared, &mut session, "BEGIN").unwrap();
    submit_session_text(
        &shared,
        &mut session,
        "CREATE TABLE failed_private_ddl (id INT)",
    )
    .unwrap();

    let error = submit_session_text(
        &shared,
        &mut session,
        "CREATE MATERIALIZED VIEW unsupported_second_ddl AS SELECT id FROM failed_private_ddl WITH NO DATA",
    )
    .unwrap_err();
    assert_eq!(error.category, ErrorCategory::Unsupported);
    assert_eq!(
        session.transaction_status(),
        SessionTransactionStatus::FailedTransaction
    );
    let blocked = submit_session_text(&shared, &mut session, "SELECT id FROM failed_private_ddl")
        .unwrap_err();
    assert_eq!(blocked.category, ErrorCategory::InFailedTransaction);

    submit_session_text(&shared, &mut session, "ROLLBACK").unwrap();
    assert_eq!(session.transaction_status(), SessionTransactionStatus::Idle);
    submit_session_text(
        &shared,
        &mut session,
        "CREATE TABLE failed_private_ddl (id INT)",
    )
    .unwrap();
    submit_session_text(
        &shared,
        &mut session,
        "CREATE MATERIALIZED VIEW unsupported_second_ddl AS SELECT id FROM failed_private_ddl WITH NO DATA",
    )
    .unwrap();
}

#[test]
fn shared_session_owns_engine_transaction_and_aborts_on_close() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    assert!(!session.in_transaction());

    submit_session_text(&shared, &mut session, "BEGIN").unwrap();
    assert!(session.in_transaction());
    assert_eq!(shared.read_engine().unwrap().active_txn_count(), 1);

    submit_session_text(&shared, &mut session, "COMMIT AND CHAIN").unwrap();
    assert!(session.in_transaction());
    assert_eq!(shared.read_engine().unwrap().active_txn_count(), 1);

    let _ = shared.submit(&mut session, SubmissionRequest::CloseSession);
    assert!(!session.in_transaction());
    assert_eq!(shared.read_engine().unwrap().active_txn_count(), 0);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sequential_session_select_uses_its_retained_gpu_generation() {
    let mut facade = MultiSessionHarness::new();
    facade.engine.set_shard_residency_enabled(true);
    facade.engine.set_auto_admit_on_commit(true);
    let reader = facade.open_session();
    let writer = facade.open_session();
    facade
        .execute(reader, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    facade
        .execute(reader, "INSERT INTO accounts (id, balance) VALUES (1, 100)")
        .unwrap();
    facade
        .execute(reader, "BEGIN ISOLATION LEVEL REPEATABLE READ")
        .unwrap();
    let first = facade
        .execute(reader, "SELECT balance FROM accounts WHERE id = 1")
        .unwrap();
    let QueryOutcome::Rows { rows, .. } = first else {
        panic!("expected rows")
    };
    assert_eq!(rows, vec![vec![DbValue::Int4(100)]]);
    facade
        .execute(writer, "UPDATE accounts SET balance = 200 WHERE id = 1")
        .unwrap();

    let outcome = facade
        .execute(reader, "SELECT balance FROM accounts WHERE id = 1")
        .unwrap();
    let QueryOutcome::Rows { rows, .. } = outcome else {
        panic!("expected rows")
    };
    assert_eq!(rows, vec![vec![DbValue::Int4(100)]]);
    facade.execute(reader, "ROLLBACK").unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn shared_session_select_uses_its_retained_gpu_generation() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1, 100)")
        .unwrap();
    let shared = SharedEngine::from_engine(engine);
    let mut reader = shared.open_session();
    submit_session_text(
        &shared,
        &mut reader,
        "BEGIN ISOLATION LEVEL REPEATABLE READ",
    )
    .unwrap();
    let first = submit_session_text(
        &shared,
        &mut reader,
        "SELECT balance FROM accounts WHERE id = 1",
    )
    .unwrap();
    let QueryOutcome::Rows { rows, .. } = first else {
        panic!("expected rows")
    };
    assert_eq!(rows, vec![vec![DbValue::Int4(100)]]);
    submit_ephemeral_text(&shared, "UPDATE accounts SET balance = 200 WHERE id = 1").unwrap();

    let outcome = submit_session_text(
        &shared,
        &mut reader,
        "SELECT balance FROM accounts WHERE id = 1",
    )
    .unwrap();
    let QueryOutcome::Rows { rows, .. } = outcome else {
        panic!("expected rows")
    };
    assert_eq!(rows, vec![vec![DbValue::Int4(100)]]);
    submit_session_text(&shared, &mut reader, "ROLLBACK").unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sequential_session_dml_is_private_until_atomic_commit() {
    let mut facade = MultiSessionHarness::new();
    facade.engine.set_shard_residency_enabled(true);
    facade.engine.set_auto_admit_on_commit(true);
    let writer = facade.open_session();
    let observer = facade.open_session();
    facade
        .execute(
            writer,
            "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT)",
        )
        .unwrap();
    facade
        .execute(writer, "INSERT INTO accounts (id, balance) VALUES (1, 100)")
        .unwrap();
    facade.execute(writer, "BEGIN").unwrap();
    let wal_before = facade.engine.durable_wal_records().len();
    facade
        .execute(writer, "UPDATE accounts SET balance = 200 WHERE id = 1")
        .unwrap();

    let QueryOutcome::Rows { rows, .. } = facade
        .execute(writer, "SELECT balance FROM accounts WHERE id = 1")
        .unwrap()
    else {
        panic!("expected writer rows")
    };
    assert_eq!(rows, vec![vec![DbValue::Int4(200)]]);
    let QueryOutcome::Rows { rows, .. } = facade
        .execute(observer, "SELECT balance FROM accounts WHERE id = 1")
        .unwrap()
    else {
        panic!("expected observer rows")
    };
    assert_eq!(rows, vec![vec![DbValue::Int4(100)]]);
    assert_eq!(facade.engine.durable_wal_records().len(), wal_before);

    facade.execute(writer, "COMMIT").unwrap();
    assert_eq!(facade.engine.durable_wal_records().len(), wal_before + 1);
    let QueryOutcome::Rows { rows, .. } = facade
        .execute(observer, "SELECT balance FROM accounts WHERE id = 1")
        .unwrap()
    else {
        panic!("expected committed rows")
    };
    assert_eq!(rows, vec![vec![DbValue::Int4(200)]]);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn shared_session_dml_is_private_until_atomic_commit() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .execute_text(1, "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT)")
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1, 100)")
        .unwrap();
    let shared = SharedEngine::from_engine(engine);
    let mut writer = shared.open_session();
    submit_session_text(&shared, &mut writer, "BEGIN").unwrap();
    submit_session_text(
        &shared,
        &mut writer,
        "UPDATE accounts SET balance = 200 WHERE id = 1",
    )
    .unwrap();

    let QueryOutcome::Rows { rows, .. } = submit_session_text(
        &shared,
        &mut writer,
        "SELECT balance FROM accounts WHERE id = 1",
    )
    .unwrap() else {
        panic!("expected writer rows")
    };
    assert_eq!(rows, vec![vec![DbValue::Int4(200)]]);
    let QueryOutcome::Rows { rows, .. } =
        submit_ephemeral_text(&shared, "SELECT balance FROM accounts WHERE id = 1").unwrap()
    else {
        panic!("expected observer rows")
    };
    assert_eq!(rows, vec![vec![DbValue::Int4(100)]]);

    submit_session_text(&shared, &mut writer, "COMMIT").unwrap();
    let QueryOutcome::Rows { rows, .. } =
        submit_ephemeral_text(&shared, "SELECT balance FROM accounts WHERE id = 1").unwrap()
    else {
        panic!("expected committed rows")
    };
    assert_eq!(rows, vec![vec![DbValue::Int4(200)]]);
}

#[test]
fn empty_statement_is_neutral_empty_not_a_syntax_error() {
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    for sql in ["", "   ", ";", ";;;"] {
        assert_eq!(
            facade.execute(session, sql),
            Ok(QueryOutcome::Empty),
            "empty statement {sql:?} should be QueryOutcome::Empty"
        );
    }
}

#[test]
fn unknown_session_is_rejected() {
    let mut facade = MultiSessionHarness::new();
    let error = facade
        .execute(HarnessSessionId(999), "SELECT 1")
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::Internal);
}

fn shared_count(shared: &SharedEngine) -> i64 {
    match submit_ephemeral_text(shared, "SELECT COUNT(*) FROM t").unwrap() {
        QueryOutcome::Rows { rows, .. } => match &rows[0][0] {
            DbValue::Int4(value) => i64::from(*value),
            DbValue::Int8(value) => *value,
            other => panic!("expected integer count, got {other:?}"),
        },
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn shared_engine_round_trips_write_then_read() {
    // The read/write-lock split (P1-M4): writes go through the write lock, reads the
    // read lock, both via the single shared entry point.
    let shared = SharedEngine::new();
    submit_ephemeral_text(&shared, "CREATE TABLE t (a INT)").unwrap();
    submit_ephemeral_text(&shared, "INSERT INTO t (a) VALUES (1)").unwrap();
    submit_ephemeral_text(&shared, "INSERT INTO t (a) VALUES (2)").unwrap();
    assert_eq!(shared_count(&shared), 2);
}

#[test]
fn shared_engine_serves_concurrent_readers() {
    // Many threads read one shared engine concurrently (read lock) and all see the
    // correct committed state — the concurrent-dispatch property the server relies on.
    use std::sync::Arc;
    use std::thread;

    let shared = Arc::new(SharedEngine::new());
    submit_ephemeral_text(&shared, "CREATE TABLE t (a INT)").unwrap();
    for i in 0..50 {
        submit_ephemeral_text(&shared, &format!("INSERT INTO t (a) VALUES ({i})")).unwrap();
    }
    let mut handles = Vec::new();
    for _ in 0..8 {
        let shared = Arc::clone(&shared);
        handles.push(thread::spawn(move || {
            for _ in 0..25 {
                assert_eq!(shared_count(&shared), 50);
            }
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }
}

// ---- Phase-3 M1: typed storable columns (NUMERIC / BIGINT / BOOL) ----

/// Run a SELECT and return its rows, panicking on anything else.
fn select_rows(
    facade: &mut MultiSessionHarness,
    session: HarnessSessionId,
    sql: &str,
) -> Vec<Vec<DbValue>> {
    match facade.execute(session, sql).unwrap() {
        QueryOutcome::Rows { rows, .. } => rows,
        other => panic!("expected rows from {sql:?}, got {other:?}"),
    }
}

#[test]
fn typed_columns_round_trip_create_insert_select_and_text_wire() {
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    facade
        .execute(
            session,
            "CREATE TABLE acct (bal NUMERIC(12,2), n BIGINT, ok BOOL)",
        )
        .unwrap();
    facade
        .execute(
            session,
            "INSERT INTO acct (bal, n, ok) VALUES (1234.5, 9000000000, TRUE)",
        )
        .unwrap();

    let outcome = facade
        .execute(session, "SELECT bal, n, ok FROM acct")
        .unwrap();
    let QueryOutcome::Rows { columns, rows } = outcome else {
        panic!("expected rows");
    };
    // Column logical types map to numeric / int8 / bool.
    assert_eq!(
        columns.iter().map(|c| c.logical_type).collect::<Vec<_>>(),
        vec![LogicalType::Numeric, LogicalType::Int8, LogicalType::Bool]
    );
    // Stored values round-trip; the numeric rescales to the column scale (2).
    assert_eq!(
        rows,
        vec![vec![
            DbValue::Numeric(Decimal128::new(123450, 2)),
            DbValue::Int8(9_000_000_000),
            DbValue::Bool(true),
        ]]
    );
    // Text wire encoding: money with two decimals, plain bigint, `t` for true.
    let wire: Vec<String> = rows[0].iter().map(pg_adapter::db_value_text).collect();
    assert_eq!(wire, vec!["1234.50", "9000000000", "t"]);
}

#[test]
fn numeric_equality_is_scale_insensitive() {
    // A stored `1.0` (declared NUMERIC(12,2), so persisted as 1.00) must match a
    // `WHERE bal = 1.00` literal — canonical-by-construction value-index equality.
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    facade
        .execute(session, "CREATE TABLE m (bal NUMERIC(12,2))")
        .unwrap();
    facade
        .execute(session, "INSERT INTO m (bal) VALUES (1.0)")
        .unwrap();

    let matched = select_rows(&mut facade, session, "SELECT bal FROM m WHERE bal = 1.00");
    assert_eq!(
        matched,
        vec![vec![DbValue::Numeric(Decimal128::new(100, 2))]]
    );
    // And the equality fast-path renders the stored money form on the wire.
    assert_eq!(pg_adapter::db_value_text(&matched[0][0]), "1.00");

    // A different value does not match.
    let unmatched = select_rows(&mut facade, session, "SELECT bal FROM m WHERE bal = 2.00");
    assert!(unmatched.is_empty());
}

#[test]
fn numeric_foreign_key_equality_links_parent_and_child() {
    // A NUMERIC equality check across a declared FK exercises the value-index equality
    // path on Decimal128 keys end to end (42.0 stored under NUMERIC(12,2) == 42.00).
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    facade
        .execute(
            session,
            "CREATE TABLE parent (id NUMERIC(12,2) PRIMARY KEY)",
        )
        .unwrap();
    facade
        .execute(session, "INSERT INTO parent (id) VALUES (42.00)")
        .unwrap();
    facade
        .execute(session, "CREATE TABLE child (pid NUMERIC(12,2))")
        .unwrap();
    facade
            .execute(
                session,
                "ALTER TABLE ONLY public.child ADD CONSTRAINT child_pid_fk FOREIGN KEY (pid) REFERENCES public.parent(id)",
            )
            .unwrap();
    // The FK check resolves the parent row by NUMERIC equality (42.0 == 42.00).
    facade
        .execute(session, "INSERT INTO child (pid) VALUES (42.0)")
        .unwrap();
    // A missing parent key is rejected by the FK.
    let err = facade
        .execute(session, "INSERT INTO child (pid) VALUES (7.00)")
        .unwrap_err();
    assert!(!err.message.is_empty());
}

#[test]
fn numeric_overflow_beyond_precision_is_a_clean_error() {
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    facade
        .execute(session, "CREATE TABLE small (bal NUMERIC(4,2))")
        .unwrap();
    // 999.99 needs 5 significant digits but the column allows 4 -> numeric field overflow.
    let err = facade
        .execute(session, "INSERT INTO small (bal) VALUES (999.99)")
        .unwrap_err();
    assert!(
        err.message.contains("numeric field overflow"),
        "unexpected error: {}",
        err.message
    );
}

#[test]
fn count_star_returns_a_neutral_int8() {
    // Phase-3 widening: COUNT(*) is now Int8 (the expectation the P0-M1 test anticipated).
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    facade.execute(session, "CREATE TABLE t (a INT)").unwrap();
    facade
        .execute(session, "INSERT INTO t (a) VALUES (1)")
        .unwrap();
    facade
        .execute(session, "INSERT INTO t (a) VALUES (2)")
        .unwrap();
    let rows = select_rows(&mut facade, session, "SELECT COUNT(*) FROM t");
    assert_eq!(rows, vec![vec![DbValue::Int8(2)]]);
}

#[test]
fn bigint_and_bool_round_trip_on_gpu_native_facade() {
    let mut facade = MultiSessionHarness::new();
    let session = facade.open_session();
    facade
        .execute(session, "CREATE TABLE flags (n BIGINT, ok BOOL)")
        .unwrap();
    facade
        .execute(
            session,
            "INSERT INTO flags (n, ok) VALUES (10000000000, TRUE)",
        )
        .unwrap();
    facade
        .execute(
            session,
            "INSERT INTO flags (n, ok) VALUES (20000000000, FALSE)",
        )
        .unwrap();
    // The production facade requires a supported GPU route. Predicate-family coverage lives
    // in the engine's actual-GPU expression suite; this facade test owns neutral type mapping
    // and the false wire representation without depending on the removed CPU query fallback.
    let rows = select_rows(&mut facade, session, "SELECT n, ok FROM flags");
    assert_eq!(
        rows,
        vec![
            vec![DbValue::Int8(10_000_000_000), DbValue::Bool(true)],
            vec![DbValue::Int8(20_000_000_000), DbValue::Bool(false)],
        ]
    );
    assert_eq!(pg_adapter::db_value_text(&rows[1][1]), "f");
}
