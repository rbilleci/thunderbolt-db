use super::*;

fn seeded_prepared_engine() -> Engine {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine.set_shard_size_target(128);
    engine
        .execute_text(
            1,
            "CREATE TABLE prepared_accounts (id INT, balance INT, PRIMARY KEY (id))",
        )
        .unwrap();
    let values = (1..=64)
        .map(|id| format!("({id}, {})", id * 10))
        .collect::<Vec<_>>()
        .join(", ");
    engine
        .execute_text(2, &format!("INSERT INTO prepared_accounts VALUES {values}"))
        .unwrap();
    engine
}

fn prepare_int4_route(engine: &Engine, sql: &[String]) -> PreparedTransactionRoute {
    prepare_int4_route_with_characteristics(
        engine,
        sql,
        TransactionCharacteristics::READ_COMMITTED_READ_WRITE,
    )
}

fn prepare_int4_route_with_characteristics(
    engine: &Engine,
    sql: &[String],
    characteristics: TransactionCharacteristics,
) -> PreparedTransactionRoute {
    let operations = sql
        .iter()
        .map(|sql| PreparedCommand::parse(sql).unwrap())
        .collect::<Vec<_>>();
    let hints = operations
        .iter()
        .map(|operation| vec![Some(SqlType::Int4); operation.parameter_count()])
        .collect();
    engine
        .prepare_transaction_route(operations, hints, characteristics)
        .unwrap()
}

fn submit_route(
    engine: &Engine,
    txn_id: u64,
    route: &PreparedTransactionRoute,
    parameters: Vec<Vec<SqlValue>>,
) -> PredeclaredTransactionResult {
    let bound = route.bind(parameters).unwrap();
    match engine.submit_transaction(txn_id, bound).unwrap() {
        TransactionAdmissionResult::Predeclared(result) => result,
        other => panic!("prepared transaction returned the wrong admission result: {other:?}"),
    }
}

fn int4_parameters(rows: impl IntoIterator<Item = Vec<i32>>) -> Vec<Vec<SqlValue>> {
    rows.into_iter()
        .map(|row| row.into_iter().map(SqlValue::Int4).collect())
        .collect()
}

#[test]
fn prepared_insert_rejects_out_of_domain_typed_temporal_carriers_before_execution() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE prepared_temporal (d date, t timestamp)")
        .unwrap();
    for (sql, ty, value) in [
        (
            "INSERT INTO prepared_temporal (d) VALUES ($1)",
            SqlType::Date,
            SqlValue::Date(gpu_db_sql::datetime::PG_DATE_END_DAYS_EXCLUSIVE),
        ),
        (
            "INSERT INTO prepared_temporal (t) VALUES ($1)",
            SqlType::Timestamp,
            SqlValue::Timestamp(gpu_db_sql::datetime::PG_TIMESTAMP_MIN_MICROS - 1),
        ),
    ] {
        let route = engine
            .prepare_transaction_route(
                vec![PreparedCommand::parse(sql).unwrap()],
                vec![vec![Some(ty)]],
                TransactionCharacteristics::READ_COMMITTED_READ_WRITE,
            )
            .unwrap();
        assert!(matches!(
            route.bind(vec![vec![value]]),
            Err(ExecuteError::Engine(EngineError::DatetimeFieldOverflow(_)))
        ));
    }
}

#[test]
fn prepared_insert_executes_finite_temporal_lower_boundaries_with_bc_canonical_source() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE prepared_temporal_bc (d date, t timestamp)")
        .unwrap();
    let route = engine
        .prepare_transaction_route(
            vec![
                PreparedCommand::parse("INSERT INTO prepared_temporal_bc (d, t) VALUES ($1, $2)")
                    .unwrap(),
            ],
            vec![vec![Some(SqlType::Date), Some(SqlType::Timestamp)]],
            TransactionCharacteristics::READ_COMMITTED_READ_WRITE,
        )
        .unwrap();
    let bound = route
        .bind(vec![vec![
            SqlValue::Date(gpu_db_sql::datetime::PG_DATE_MIN_DAYS),
            SqlValue::Timestamp(gpu_db_sql::datetime::PG_TIMESTAMP_MIN_MICROS),
        ]])
        .expect("finite lower carriers bind through the BC-form canonical source");
    assert!(matches!(
        engine.submit_transaction(2, bound),
        Ok(TransactionAdmissionResult::Predeclared(_))
    ));
}

fn selected_int4(result: &PredeclaredOperationResult) -> i32 {
    let PredeclaredOperationResult::Read(result) = result else {
        panic!("expected prepared read result")
    };
    match &result.rows[0][0] {
        SqlValue::Int4(value) => *value,
        value => panic!("expected int4 result, got {value:?}"),
    }
}

fn assert_index_read(result: &PredeclaredOperationResult) {
    let PredeclaredOperationResult::Read(result) = result else {
        panic!("expected prepared read result")
    };
    assert!(matches!(
        result.access_path.as_ref(),
        RelationalAccessPath::EqualityIndex { .. }
    ));
}

fn reset_slot_scan_probes() {
    crate::engine_expr::RESIDENT_CONJUNCT_SLOT_SCAN_PROBES.store(0, AtomicOrdering::Relaxed);
    crate::engine_expr::RESIDENT_ALL_SLOT_SCAN_PROBES.store(0, AtomicOrdering::Relaxed);
}

fn assert_no_slot_scan_probes() {
    assert_eq!(
        crate::engine_expr::RESIDENT_CONJUNCT_SLOT_SCAN_PROBES.load(AtomicOrdering::Relaxed),
        0,
        "prepared execution entered the conjunct O(rows) locator"
    );
    assert_eq!(
        crate::engine_expr::RESIDENT_ALL_SLOT_SCAN_PROBES.load(AtomicOrdering::Relaxed),
        0,
        "prepared execution entered the all-slot O(rows) locator"
    );
}

fn device_row_id_for_prepared_key(engine: &Engine, key: i32) -> u64 {
    let table = engine
        .relational_catalog_table("prepared_accounts")
        .unwrap();
    let hits = engine
        .locate_resident_pk_via_shard_index_detailed(&table, 0, key)
        .expect("prepared key locate must answer from the device index");
    let hit = hits
        .first()
        .expect("prepared key must have a resident slot");
    let region = hit
        .row_id
        .as_ref()
        .expect("prepared resident slot must carry a row identity");
    let halves = region
        .read_resident_i32_column(u64::from(hit.slot) * 8, 2)
        .unwrap();
    (halves[0] as u32 as u64) | ((halves[1] as u32 as u64) << 32)
}

/// The three bounded classes are derived by the engine, each commits as one canonical WAL record,
/// and INSERT's historical uniqueness proof never enters the full-shard predicate scan.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_w1_t8_t32_classes_are_atomic_and_index_bounded() {
    use crate::engine_dml_prepare::RESIDENT_EXACT_KEY_FULL_SCAN_PROBES;
    use crate::engine_prepared_transaction::{
        PREPARED_PRIVATE_INDEX_REBUILD_MAX_ROWS, PREPARED_PRIVATE_INDEX_REBUILD_ROWS,
    };

    let empty = Engine::new_local();
    empty.set_shard_residency_enabled(true);
    empty.set_auto_admit_on_commit(true);
    empty
        .execute_text(
            3,
            "CREATE TABLE empty_prepared_accounts (id INT, balance INT, PRIMARY KEY (id))",
        )
        .unwrap();
    let first_insert = prepare_int4_route(
        &empty,
        &["INSERT INTO empty_prepared_accounts VALUES ($1, $2) RETURNING id".to_string()],
    );
    let first = submit_route(
        &empty,
        4,
        &first_insert,
        int4_parameters([[1, 10].to_vec()]),
    );
    assert_eq!(first.class, TransactionClass::W1);

    let engine = seeded_prepared_engine();

    let insert = prepare_int4_route(
        &engine,
        &["INSERT INTO prepared_accounts VALUES ($1, $2) RETURNING id".to_string()],
    );
    let scans_before = RESIDENT_EXACT_KEY_FULL_SCAN_PROBES.load(AtomicOrdering::Relaxed);
    let wal_before = engine.durable_wal_records().len();
    let inserted = submit_route(
        &engine,
        10,
        &insert,
        int4_parameters([[100, 1_000].to_vec()]),
    );
    assert_eq!(inserted.class, TransactionClass::W1);
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    assert_eq!(
        RESIDENT_EXACT_KEY_FULL_SCAN_PROBES.load(AtomicOrdering::Relaxed),
        scans_before,
        "prepared INSERT uniqueness history must use named-index candidates, not a shard scan"
    );

    let update = prepare_int4_route(
        &engine,
        &[
            "UPDATE prepared_accounts SET balance = balance + $2 WHERE id = $1 RETURNING balance"
                .to_string(),
        ],
    );
    let updated = submit_route(&engine, 11, &update, int4_parameters([[1, 5].to_vec()]));
    assert_eq!(updated.class, TransactionClass::W1);

    let delete = prepare_int4_route(
        &engine,
        &["DELETE FROM prepared_accounts WHERE id = $1 RETURNING id".to_string()],
    );
    let deleted = submit_route(&engine, 12, &delete, int4_parameters([[2].to_vec()]));
    assert_eq!(deleted.class, TransactionClass::W1);

    let mut t8_sql = (5..=8)
        .map(|_| "SELECT balance FROM prepared_accounts WHERE id = $1".to_string())
        .collect::<Vec<_>>();
    t8_sql.extend((1..=4).map(|_| {
        "UPDATE prepared_accounts SET balance = balance + $2 WHERE id = $1 RETURNING balance"
            .to_string()
    }));
    let t8 = prepare_int4_route(&engine, &t8_sql);
    let mut t8_params = (5..=8).map(|id| vec![id]).collect::<Vec<_>>();
    t8_params.extend((1..=4).map(|id| vec![id, 1]));
    let wal_before = engine.durable_wal_records().len();
    PREPARED_PRIVATE_INDEX_REBUILD_ROWS.store(0, AtomicOrdering::Relaxed);
    PREPARED_PRIVATE_INDEX_REBUILD_MAX_ROWS.store(0, AtomicOrdering::Relaxed);
    let t8_result = submit_route(&engine, 13, &t8, int4_parameters(t8_params));
    assert_eq!(t8_result.class, TransactionClass::T8);
    assert_eq!(t8_result.operations.len(), 8);
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    assert!(
        PREPARED_PRIVATE_INDEX_REBUILD_MAX_ROWS.load(AtomicOrdering::Relaxed) <= 4,
        "T8 may rebuild only its bounded transaction-private overlay, never the base relation"
    );
    assert!(
        PREPARED_PRIVATE_INDEX_REBUILD_ROWS.load(AtomicOrdering::Relaxed) <= 32,
        "T8 private-index work must remain within twice its squared four-mutation bound"
    );

    let mut t32_sql = (17..=32)
        .map(|_| "SELECT balance FROM prepared_accounts WHERE id = $1".to_string())
        .collect::<Vec<_>>();
    t32_sql.extend((1..=16).map(|_| {
        "UPDATE prepared_accounts SET balance = balance + $2 WHERE id = $1 RETURNING balance"
            .to_string()
    }));
    let t32 = prepare_int4_route(&engine, &t32_sql);
    let mut t32_params = (17..=32).map(|id| vec![id]).collect::<Vec<_>>();
    t32_params.extend((1..=16).map(|id| vec![id, 1]));
    let wal_before = engine.durable_wal_records().len();
    PREPARED_PRIVATE_INDEX_REBUILD_ROWS.store(0, AtomicOrdering::Relaxed);
    PREPARED_PRIVATE_INDEX_REBUILD_MAX_ROWS.store(0, AtomicOrdering::Relaxed);
    let t32_result = submit_route(&engine, 14, &t32, int4_parameters(t32_params));
    assert_eq!(t32_result.class, TransactionClass::T32);
    assert_eq!(t32_result.operations.len(), 32);
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    assert!(
        PREPARED_PRIVATE_INDEX_REBUILD_MAX_ROWS.load(AtomicOrdering::Relaxed) <= 16,
        "T32 may rebuild only its bounded transaction-private overlay, never the base relation"
    );
    let t32_private_rebuild_rows =
        PREPARED_PRIVATE_INDEX_REBUILD_ROWS.load(AtomicOrdering::Relaxed);
    assert!(
        t32_private_rebuild_rows <= 512,
        "T32 private-index work must remain within twice its squared sixteen-mutation bound; observed {t32_private_rebuild_rows} indexed rows"
    );

    let admissions = engine.prepared_transaction_class_admissions();
    assert_eq!(admissions.w1, 3);
    assert_eq!(admissions.t8, 1);
    assert_eq!(admissions.t32, 1);
    assert_eq!(admissions.general, 0);
}

/// More than 32 operations remain a supported General transaction, while READ COMMITTED programs
/// read their own private generation. Neither behavior borrows a T32 latency label.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_general_over_32_and_read_committed_read_your_writes() {
    use crate::engine_dml_prepare::RESIDENT_EXACT_KEY_FULL_SCAN_PROBES;

    let engine = seeded_prepared_engine();

    let ryw = prepare_int4_route(
        &engine,
        &[
            "INSERT INTO prepared_accounts VALUES ($1, $2) RETURNING id".to_string(),
            "SELECT balance FROM prepared_accounts WHERE id = $1".to_string(),
        ],
    );
    let result = submit_route(
        &engine,
        20,
        &ryw,
        int4_parameters([[101, 2_020].to_vec(), [101].to_vec()]),
    );
    assert_eq!(result.class, TransactionClass::T8);
    assert_eq!(selected_int4(&result.operations[1]), 2_020);
    assert_index_read(&result.operations[1]);

    let mut sql = (1..=33)
        .map(|_| "SELECT balance FROM prepared_accounts WHERE id = $1".to_string())
        .collect::<Vec<_>>();
    sql.push(
        "UPDATE prepared_accounts SET balance = balance + $2 WHERE id = $1 RETURNING balance"
            .to_string(),
    );
    let route = prepare_int4_route(&engine, &sql);
    let mut params = (1..=33).map(|id| vec![id]).collect::<Vec<_>>();
    params.push(vec![64, 7]);
    let wal_before = engine.durable_wal_records().len();
    let scans_before = RESIDENT_EXACT_KEY_FULL_SCAN_PROBES.load(AtomicOrdering::Relaxed);
    PREPARED_PRIVATE_INDEX_REBUILD_ROWS.store(0, AtomicOrdering::Relaxed);
    PREPARED_PRIVATE_INDEX_REBUILD_MAX_ROWS.store(0, AtomicOrdering::Relaxed);
    let result = submit_route(&engine, 21, &route, int4_parameters(params));
    assert_eq!(result.class, TransactionClass::General);
    assert_eq!(result.operations.len(), 34);
    for read in &result.operations[..33] {
        assert_index_read(read);
    }
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    assert_eq!(
        RESIDENT_EXACT_KEY_FULL_SCAN_PROBES.load(AtomicOrdering::Relaxed),
        scans_before,
        "indexed General execution must not enter the legacy unique-history scan"
    );
    assert!(
        PREPARED_PRIVATE_INDEX_REBUILD_MAX_ROWS.load(AtomicOrdering::Relaxed) <= 1,
        "General indexed execution may rebuild only its one-row private overlay"
    );
    let admissions = engine.prepared_transaction_class_admissions();
    assert_eq!(admissions.t8, 1);
    assert_eq!(admissions.general, 1);
}

/// Type, ownership, physical-index, catalog, and staged-semantic failures all remain pre-WAL and
/// leave no active program context. A staged prefix is never partially published.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_route_sabotage_fails_closed_without_partial_publication() {
    let engine = seeded_prepared_engine();
    let update = prepare_int4_route(
        &engine,
        &[
            "UPDATE prepared_accounts SET balance = balance + $2 WHERE id = $1 RETURNING balance"
                .to_string(),
        ],
    );
    let wal_before = engine.durable_wal_records().len();
    let type_error = update
        .bind(vec![vec![
            SqlValue::Text("1".to_string()),
            SqlValue::Int4(1),
        ]])
        .unwrap_err();
    assert!(matches!(type_error, ExecuteError::DatatypeMismatch(_)));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(engine.transaction_snapshot_handle(30).is_none());

    let foreign_bound = update.bind(int4_parameters([[1, 1].to_vec()])).unwrap();
    let other = seeded_prepared_engine();
    let other_wal_before = other.durable_wal_records().len();
    let owner_error = other.submit_transaction(31, foreign_bound).unwrap_err();
    assert!(owner_error.to_string().contains("another engine instance"));
    assert_eq!(other.durable_wal_records().len(), other_wal_before);
    assert!(other.transaction_snapshot_handle(31).is_none());

    let physical = prepare_int4_route(
        &engine,
        &["DELETE FROM prepared_accounts WHERE id = $1".to_string()],
    );
    engine
        .read_state
        .residency
        .shard_pk_device_index
        .lock()
        .unwrap()
        .clear();
    let wal_before = engine.durable_wal_records().len();
    let physical_error = engine
        .submit_transaction(32, physical.bind(int4_parameters([[3].to_vec()])).unwrap())
        .unwrap_err();
    assert!(
        physical_error
            .to_string()
            .contains("exact device index allocation"),
        "unexpected physical sabotage error: {physical_error}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(engine.transaction_snapshot_handle(32).is_none());

    let catalog_engine = seeded_prepared_engine();
    let stale = prepare_int4_route(
        &catalog_engine,
        &["DELETE FROM prepared_accounts WHERE id = $1".to_string()],
    );
    catalog_engine
        .execute_text(
            33,
            "CREATE INDEX prepared_accounts_balance_idx ON prepared_accounts (balance)",
        )
        .unwrap();
    let wal_before = catalog_engine.durable_wal_records().len();
    let stale_error = catalog_engine
        .submit_transaction(34, stale.bind(int4_parameters([[4].to_vec()])).unwrap())
        .unwrap_err();
    assert!(stale_error.to_string().contains("catalog"));
    assert_eq!(catalog_engine.durable_wal_records().len(), wal_before);
    assert!(catalog_engine.transaction_snapshot_handle(34).is_none());

    let atomic_engine = seeded_prepared_engine();
    let failing = prepare_int4_route(
        &atomic_engine,
        &[
            "UPDATE prepared_accounts SET balance = balance + $2 WHERE id = $1 RETURNING balance"
                .to_string(),
            "INSERT INTO prepared_accounts VALUES ($1, $2) RETURNING id".to_string(),
        ],
    );
    let wal_before = atomic_engine.durable_wal_records().len();
    let visible_before = atomic_engine.committed_seq();
    let admissions_before = atomic_engine.prepared_transaction_class_admissions();
    let error = atomic_engine
        .submit_transaction(
            35,
            failing
                .bind(int4_parameters([[1, 99].to_vec(), [2, 999].to_vec()]))
                .unwrap(),
        )
        .unwrap_err();
    assert!(error.is_unique_violation(), "unexpected failure: {error}");
    assert_eq!(atomic_engine.durable_wal_records().len(), wal_before);
    assert_eq!(atomic_engine.committed_seq(), visible_before);
    assert!(atomic_engine.transaction_snapshot_handle(35).is_none());
    assert_eq!(
        atomic_engine.prepared_transaction_class_admissions(),
        admissions_before,
        "a failed prepared program is never counted as admitted"
    );
    let Command::Select(select) =
        parse_command("SELECT balance FROM prepared_accounts WHERE id = 1").unwrap()
    else {
        unreachable!()
    };
    let unchanged = atomic_engine.execute_relational_select(&select).unwrap();
    assert_eq!(unchanged.rows[0], [SqlValue::Int4(10)]);
}

/// Active READ COMMITTED prepared DML refreshes first, then compares only its exact catalog/FK
/// dependency closure. An unrelated row commit must not spuriously invalidate it; a target-table
/// DDL change must fail before the statement publishes a private delta.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn active_prepared_dml_uses_refreshed_exact_catalog_dependencies() {
    let engine = seeded_prepared_engine();
    let template = PreparedCommand::parse(
        "UPDATE prepared_accounts SET balance = balance + $2 WHERE id = $1 RETURNING balance",
    )
    .unwrap();
    let description = engine
        .describe_prepared_command(&template, &[Some(SqlType::Int4), Some(SqlType::Int4)])
        .unwrap();
    engine
        .submit_transaction(
            40,
            gpu_db_sql::ParsedCommand::parse("BEGIN ISOLATION LEVEL READ COMMITTED READ WRITE")
                .unwrap(),
        )
        .unwrap();
    engine
        .execute_text(
            41,
            "UPDATE prepared_accounts SET balance = balance + 3 WHERE id = 2",
        )
        .unwrap();
    let request = MutationRequest::new(
        template
            .bind(&[SqlValue::Int4(1), SqlValue::Int4(4)])
            .unwrap(),
    )
    .with_expected_catalog_version(description.catalog_version);
    let result = engine.submit_transaction(40, request).unwrap();
    let TransactionAdmissionResult::Dml(result) = result else {
        panic!("active prepared DML returned the wrong admission outcome")
    };
    assert_eq!(result.rows_affected, 1);
    engine
        .submit_transaction(40, gpu_db_sql::ParsedCommand::parse("COMMIT").unwrap())
        .unwrap();

    let ddl_engine = seeded_prepared_engine();
    let description = ddl_engine
        .describe_prepared_command(&template, &[Some(SqlType::Int4), Some(SqlType::Int4)])
        .unwrap();
    ddl_engine
        .submit_transaction(
            42,
            gpu_db_sql::ParsedCommand::parse("BEGIN ISOLATION LEVEL READ COMMITTED READ WRITE")
                .unwrap(),
        )
        .unwrap();
    ddl_engine
        .execute_text(
            43,
            "CREATE INDEX prepared_accounts_balance_live_idx ON prepared_accounts (balance)",
        )
        .unwrap();
    let wal_before = ddl_engine.durable_wal_records().len();
    let request = MutationRequest::new(
        template
            .bind(&[SqlValue::Int4(1), SqlValue::Int4(4)])
            .unwrap(),
    )
    .with_expected_catalog_version(description.catalog_version);
    let error = ddl_engine.submit_transaction(42, request).unwrap_err();
    assert!(error.to_string().contains("catalog dependencies changed"));
    assert_eq!(ddl_engine.durable_wal_records().len(), wal_before);
    let snapshot = ddl_engine.transaction_snapshot_handle(42).unwrap();
    assert!(snapshot.transaction_delta_is_empty());
    ddl_engine
        .submit_transaction(42, gpu_db_sql::ParsedCommand::parse("ROLLBACK").unwrap())
        .unwrap();
}

/// READ COMMITTED refreshes between operations while REPEATABLE READ retains its first data
/// snapshot; both keep transaction-local generations available for read-your-writes.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_program_snapshot_refresh_matches_declared_isolation() {
    let sql = [
        "SELECT balance FROM prepared_accounts WHERE id = $1".to_string(),
        "SELECT balance FROM prepared_accounts WHERE id = $1".to_string(),
    ];

    let committed = seeded_prepared_engine();
    let route = prepare_int4_route_with_characteristics(
        &committed,
        &sql,
        TransactionCharacteristics::READ_COMMITTED_READ_WRITE,
    );
    let bound = route
        .bind(int4_parameters([[1].to_vec(), [1].to_vec()]))
        .unwrap();
    PREPARED_PINNED_INDEX_HITS.store(0, AtomicOrdering::Relaxed);
    PREPARED_BASE_INDEX_MISSES.store(0, AtomicOrdering::Relaxed);
    PREPARED_PRIVATE_INDEX_REBUILD_ROWS.store(0, AtomicOrdering::Relaxed);
    let committed_result = committed
        .submit_bound_prepared_transaction_instrumented(50, bound, |operation| {
            if operation == 0 {
                std::thread::scope(|scope| {
                    scope
                        .spawn(|| {
                            committed
                                .execute_text(
                                    51,
                                    "UPDATE prepared_accounts SET balance = balance + 100 WHERE id = 1",
                                )
                                .unwrap();
                        })
                        .join()
                        .unwrap();
                });
            }
        })
        .unwrap_or_else(|error| {
            panic!(
                "RC refresh failed: {error}; pinned_hits={} base_misses={} private_rows={}",
                PREPARED_PINNED_INDEX_HITS.load(AtomicOrdering::Relaxed),
                PREPARED_BASE_INDEX_MISSES.load(AtomicOrdering::Relaxed),
                PREPARED_PRIVATE_INDEX_REBUILD_ROWS.load(AtomicOrdering::Relaxed),
            )
        });
    assert_eq!(selected_int4(&committed_result.operations[0]), 10);
    assert_eq!(selected_int4(&committed_result.operations[1]), 110);

    let repeatable = seeded_prepared_engine();
    let route = prepare_int4_route_with_characteristics(
        &repeatable,
        &sql,
        TransactionCharacteristics {
            isolation: TransactionIsolation::RepeatableRead,
            ..TransactionCharacteristics::READ_COMMITTED_READ_WRITE
        },
    );
    let bound = route
        .bind(int4_parameters([[1].to_vec(), [1].to_vec()]))
        .unwrap();
    let repeatable_result = repeatable
        .submit_bound_prepared_transaction_instrumented(52, bound, |operation| {
            if operation == 0 {
                std::thread::scope(|scope| {
                    scope
                        .spawn(|| {
                            repeatable
                                .execute_text(
                                    53,
                                    "UPDATE prepared_accounts SET balance = balance + 100 WHERE id = 1",
                                )
                                .unwrap();
                        })
                        .join()
                        .unwrap();
                });
            }
        })
        .unwrap();
    assert_eq!(selected_int4(&repeatable_result.operations[0]), 10);
    assert_eq!(selected_int4(&repeatable_result.operations[1]), 10);

    let repeatable_ryw = seeded_prepared_engine();
    let route = prepare_int4_route_with_characteristics(
        &repeatable_ryw,
        &[
            "UPDATE prepared_accounts SET balance = balance + $2 WHERE id = $1 RETURNING balance"
                .to_string(),
            "SELECT balance FROM prepared_accounts WHERE id = $1".to_string(),
        ],
        TransactionCharacteristics {
            isolation: TransactionIsolation::RepeatableRead,
            ..TransactionCharacteristics::READ_COMMITTED_READ_WRITE
        },
    );
    PREPARED_PINNED_INDEX_HITS.store(0, AtomicOrdering::Relaxed);
    PREPARED_BASE_INDEX_MISSES.store(0, AtomicOrdering::Relaxed);
    PREPARED_PRIVATE_INDEX_REBUILD_ROWS.store(0, AtomicOrdering::Relaxed);
    let bound = route
        .bind(int4_parameters([[1, 5].to_vec(), [1].to_vec()]))
        .unwrap();
    let result = repeatable_ryw
        .submit_bound_prepared_transaction(54, bound)
        .unwrap_or_else(|error| {
            panic!(
                "RR RYW failed: {error}; pinned_hits={} base_misses={} private_rows={}",
                PREPARED_PINNED_INDEX_HITS.load(AtomicOrdering::Relaxed),
                PREPARED_BASE_INDEX_MISSES.load(AtomicOrdering::Relaxed),
                PREPARED_PRIVATE_INDEX_REBUILD_ROWS.load(AtomicOrdering::Relaxed),
            )
        });
    assert_eq!(result.class, TransactionClass::T8);
    assert_eq!(selected_int4(&result.operations[1]), 15);
    assert_index_read(&result.operations[1]);
}

/// A retained RR route consumes the index allocation it pinned, even when a concurrent append
/// rotates the published generation and cache retirement removes the global lookup entry.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_repeatable_read_uses_retained_index_across_generation_replacement() {
    let engine = seeded_prepared_engine();
    let route = prepare_int4_route_with_characteristics(
        &engine,
        &[
            "SELECT balance FROM prepared_accounts WHERE id = $1".to_string(),
            "SELECT balance FROM prepared_accounts WHERE id = $1".to_string(),
        ],
        TransactionCharacteristics {
            isolation: TransactionIsolation::RepeatableRead,
            ..TransactionCharacteristics::READ_COMMITTED_READ_WRITE
        },
    );
    let before_generation =
        Arc::clone(&engine.read_residency_shards()["prepared_accounts"][0].point_route_generation);
    let bound = route
        .bind(int4_parameters([[1].to_vec(), [1].to_vec()]))
        .unwrap();
    let result = engine
        .submit_bound_prepared_transaction_instrumented(55, bound, |operation| {
            if operation == 0 {
                std::thread::scope(|scope| {
                    scope
                        .spawn(|| {
                            for (batch, start) in (65..=320).step_by(32).enumerate() {
                                let values = (start..=(start + 31))
                                    .map(|id| format!("({id}, {})", id * 10))
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                engine
                                    .execute_text(
                                        56 + batch as u64,
                                        &format!("INSERT INTO prepared_accounts VALUES {values}"),
                                    )
                                    .unwrap();
                            }
                        })
                        .join()
                        .unwrap();
                });
                engine
                    .read_state
                    .residency
                    .purge_shard_pk_index_for_table("prepared_accounts");
            }
        })
        .unwrap();
    let current_generation =
        Arc::clone(&engine.read_residency_shards()["prepared_accounts"][0].point_route_generation);
    assert!(!Arc::ptr_eq(&before_generation, &current_generation));
    assert_eq!(selected_int4(&result.operations[0]), 10);
    assert_eq!(selected_int4(&result.operations[1]), 10);
    assert_index_read(&result.operations[1]);
}

/// PostgreSQL's `key = NULL` predicate is UNKNOWN and therefore matches no row. It remains an
/// indexed empty verdict, not a missing-needle error or a full scan.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_null_unique_key_is_an_empty_select_or_noop_mutation() {
    let engine = seeded_prepared_engine();
    let null = vec![vec![SqlValue::Null]];

    let select = prepare_int4_route(
        &engine,
        &["SELECT balance FROM prepared_accounts WHERE id = $1".to_string()],
    );
    let selected = submit_route(&engine, 60, &select, null.clone());
    let PredeclaredOperationResult::Read(read) = &selected.operations[0] else {
        panic!("expected read")
    };
    assert!(read.rows.is_empty());
    assert_index_read(&selected.operations[0]);

    let update = prepare_int4_route(
        &engine,
        &["UPDATE prepared_accounts SET balance = 99 WHERE id = $1 RETURNING balance".to_string()],
    );
    let updated = submit_route(&engine, 61, &update, null.clone());
    let PredeclaredOperationResult::Mutation(updated) = &updated.operations[0] else {
        panic!("expected mutation")
    };
    assert_eq!(updated.rows_affected, 0);

    let delete = prepare_int4_route(
        &engine,
        &["DELETE FROM prepared_accounts WHERE id = $1 RETURNING id".to_string()],
    );
    let deleted = submit_route(&engine, 62, &delete, null);
    let PredeclaredOperationResult::Mutation(deleted) = &deleted.operations[0] else {
        panic!("expected mutation")
    };
    assert_eq!(deleted.rows_affected, 0);
}

/// An unindexed inbound FK is PostgreSQL-valid but cannot borrow a bounded class: its child
/// existence check has no bounded exact probe. General execution remains supported and exact.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_unindexed_fk_auxiliary_probe_is_general_not_hidden_fast_work() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .execute_text(70, "CREATE TABLE prepared_parent (id INT PRIMARY KEY)")
        .unwrap();
    engine
        .execute_text(
            71,
            "CREATE TABLE prepared_child (id INT PRIMARY KEY, parent_id INT)",
        )
        .unwrap();
    engine
        .execute_text(
            72,
            "ALTER TABLE ONLY prepared_child ADD CONSTRAINT prepared_child_parent_fk FOREIGN KEY (parent_id) REFERENCES prepared_parent(id)",
        )
        .unwrap();
    engine
        .execute_text(73, "INSERT INTO prepared_parent VALUES (1), (2)")
        .unwrap();
    engine
        .execute_text(74, "INSERT INTO prepared_child VALUES (10, 1)")
        .unwrap();
    let route = prepare_int4_route(
        &engine,
        &["DELETE FROM prepared_parent WHERE id = $1 RETURNING id".to_string()],
    );
    assert!(!route.proof.indexed_execution_eligible);
    let deleted = submit_route(&engine, 75, &route, int4_parameters([[2].to_vec()]));
    assert_eq!(deleted.class, TransactionClass::General);
    assert_eq!(engine.prepared_transaction_class_admissions().general, 1);

    let wal_before = engine.durable_wal_records().len();
    let admissions_before = engine.prepared_transaction_class_admissions();
    let error = engine
        .submit_transaction(76, route.bind(int4_parameters([[1].to_vec()])).unwrap())
        .unwrap_err();
    assert!(error.to_string().contains("foreign key"));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(
        engine.prepared_transaction_class_admissions(),
        admissions_before
    );
}

/// Resource declaration uses value-independent maximum fixed-row widths, so a READ COMMITTED
/// refresh cannot enlarge the WAL image beyond its pre-BEGIN permit.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_read_committed_row_width_race_stays_within_every_declared_dimension() {
    let engine = seeded_prepared_engine();
    let route = prepare_int4_route(
        &engine,
        &[
            "UPDATE prepared_accounts SET balance = balance + $2 WHERE id = $1 RETURNING balance"
                .to_string(),
        ],
    );
    let bound = route.bind(int4_parameters([[1, 0].to_vec()])).unwrap();
    let estimate = engine
        .estimate_bound_prepared_resources(&bound.operations, &bound.proof)
        .unwrap()
        .resources;
    let result = engine
        .submit_bound_prepared_transaction_with_admission_hook(80, bound, || {
            engine
                .execute_text(
                    81,
                    "UPDATE prepared_accounts SET balance = -2147483648 WHERE id = 1",
                )
                .unwrap();
        })
        .unwrap();
    assert_eq!(result.class, TransactionClass::W1);
    let actual = result.actual_resources;
    assert!(actual.operations <= estimate.operations);
    assert!(actual.mutations <= estimate.mutations);
    assert!(actual.post_image_and_wal_bytes <= estimate.post_image_and_wal_bytes);
    assert!(actual.maintained_index_fanout <= estimate.maintained_index_fanout);
    assert!(actual.touched_tables <= estimate.touched_tables);
    assert!(actual.cold_accesses <= estimate.cold_accesses);
    assert!(actual.result_bytes <= estimate.result_bytes);
}

/// Internal history/tombstone probes use the declared named key, not the first physical i32
/// column. More than 32 operations remain indexed even when the primary key is not column zero.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_named_key_probe_is_column_order_independent_and_general_scan_free() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine.set_shard_size_target(128);
    engine
        .execute_text(
            90,
            "CREATE TABLE ordered_key (payload INT, id INT PRIMARY KEY)",
        )
        .unwrap();
    let values = (1..=64)
        .map(|id| format!("({}, {id})", id * 10))
        .collect::<Vec<_>>()
        .join(", ");
    engine
        .execute_text(91, &format!("INSERT INTO ordered_key VALUES {values}"))
        .unwrap();

    let mut sql =
        vec!["UPDATE ordered_key SET payload = $2 WHERE id = $1 RETURNING payload".to_string()];
    sql.extend((0..32).map(|_| "SELECT payload FROM ordered_key WHERE id = $1".to_string()));
    let route = prepare_int4_route(&engine, &sql);
    let mut parameters = vec![vec![SqlValue::Int4(1), SqlValue::Int4(111)]];
    parameters.extend((0..32).map(|_| vec![SqlValue::Int4(1)]));
    let bound = route.bind(parameters).unwrap();
    reset_slot_scan_probes();
    PREPARED_BASE_INDEX_MISSES.store(0, AtomicOrdering::Relaxed);
    let result = engine.submit_bound_prepared_transaction(92, bound).unwrap();

    assert_eq!(result.class, TransactionClass::General);
    let PredeclaredOperationResult::Mutation(updated) = &result.operations[0] else {
        panic!("expected UPDATE result")
    };
    assert_eq!(updated.rows_affected, 1);
    for operation in &result.operations[1..] {
        assert_eq!(selected_int4(operation), 111);
        assert_index_read(operation);
    }
    assert_no_slot_scan_probes();
    assert_eq!(PREPARED_BASE_INDEX_MISSES.load(AtomicOrdering::Relaxed), 0);
}

/// A prepared commit whose exact posting chain exceeds the bounded write-locate output fails
/// retryably. It never turns that physical decline into a conjunct or all-slot scan.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_version_churn_conflict_fails_without_commit_time_scan_or_allocator_leak() {
    let engine = seeded_prepared_engine();
    let route = prepare_int4_route(
        &engine,
        &[
            "UPDATE prepared_accounts SET balance = balance + $2 WHERE id = $1 RETURNING balance"
                .to_string(),
        ],
    );
    let bound = route.bind(int4_parameters([[1, 100].to_vec()])).unwrap();
    let mut post_competitor = None;
    PREPARED_BASE_INDEX_MISSES.store(0, AtomicOrdering::Relaxed);
    let error = engine
        .submit_bound_prepared_transaction_instrumented(100, bound, |operation| {
            if operation != 0 {
                return;
            }
            std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        for step in 0..5_u64 {
                            engine
                                .execute_text(
                                    101 + step,
                                    "UPDATE prepared_accounts SET balance = balance + 1 WHERE id = 1",
                                )
                                .unwrap();
                        }
                    })
                    .join()
                    .unwrap();
            });
            post_competitor = Some((
                engine.read_state.mvcc.current_row_id(),
                engine.durable_wal_records().len(),
            ));
            reset_slot_scan_probes();
        })
        .unwrap_err();
    assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
    let (row_id_after_competitor, wal_after_competitor) = post_competitor.unwrap();
    assert_eq!(
        engine.read_state.mvcc.current_row_id(),
        row_id_after_competitor
    );
    assert_eq!(engine.durable_wal_records().len(), wal_after_competitor);
    assert_no_slot_scan_probes();
    assert_eq!(PREPARED_BASE_INDEX_MISSES.load(AtomicOrdering::Relaxed), 0);
}

/// Unique and FK changes that happen after staging reject before WAL without consuming the
/// tentative inserted entity id.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_commit_time_constraint_rejection_does_not_claim_row_ids() {
    let unique = seeded_prepared_engine();
    let route = prepare_int4_route(
        &unique,
        &["INSERT INTO prepared_accounts VALUES ($1, $2)".to_string()],
    );
    let bound = route.bind(int4_parameters([[65, 650].to_vec()])).unwrap();
    let unique_admissions_before = unique.prepared_transaction_class_admissions();
    let mut unique_cut = None;
    let error = unique
        .submit_bound_prepared_transaction_instrumented(110, bound, |operation| {
            if operation != 0 {
                return;
            }
            std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        unique
                            .execute_text(111, "INSERT INTO prepared_accounts VALUES (65, 999)")
                            .unwrap();
                    })
                    .join()
                    .unwrap();
            });
            unique_cut = Some((
                unique.read_state.mvcc.current_row_id(),
                unique.durable_wal_records().len(),
            ));
        })
        .unwrap_err();
    assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
    let (unique_row_id, unique_wal) = unique_cut.unwrap();
    assert_eq!(unique.read_state.mvcc.current_row_id(), unique_row_id);
    assert_eq!(unique.durable_wal_records().len(), unique_wal);
    let unique_admissions_after = unique.prepared_transaction_class_admissions();
    assert_eq!(
        unique_admissions_after.w1,
        unique_admissions_before.w1 + 1,
        "the staged route was admitted exactly once even though commit serialized"
    );
    let unique_rows = unique
        .execute_relational_select_text("SELECT balance FROM prepared_accounts WHERE id = 65")
        .unwrap();
    assert_eq!(unique_rows.rows, vec![vec![SqlValue::Int4(999)]]);

    let foreign_key = Engine::new_local();
    foreign_key.set_shard_residency_enabled(true);
    foreign_key.set_auto_admit_on_commit(true);
    foreign_key
        .execute_text(120, "CREATE TABLE parents (id INT PRIMARY KEY)")
        .unwrap();
    foreign_key
        .execute_text(
            121,
            "CREATE TABLE children (id INT PRIMARY KEY, parent_id INT)",
        )
        .unwrap();
    foreign_key
        .execute_text(
            122,
            "ALTER TABLE ONLY children ADD CONSTRAINT children_parent_fk FOREIGN KEY (parent_id) REFERENCES parents(id)",
        )
        .unwrap();
    foreign_key
        .execute_text(123, "INSERT INTO parents VALUES (1)")
        .unwrap();
    let route = prepare_int4_route(
        &foreign_key,
        &["INSERT INTO children VALUES ($1, $2)".to_string()],
    );
    let bound = route.bind(int4_parameters([[1, 1].to_vec()])).unwrap();
    let fk_admissions_before = foreign_key.prepared_transaction_class_admissions();
    let mut fk_cut = None;
    let error = foreign_key
        .submit_bound_prepared_transaction_instrumented(124, bound, |operation| {
            if operation != 0 {
                return;
            }
            std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        foreign_key
                            .execute_text(125, "DELETE FROM parents WHERE id = 1")
                            .unwrap();
                    })
                    .join()
                    .unwrap();
            });
            fk_cut = Some((
                foreign_key.read_state.mvcc.current_row_id(),
                foreign_key.durable_wal_records().len(),
            ));
        })
        .unwrap_err();
    assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
    let (fk_row_id, fk_wal) = fk_cut.unwrap();
    assert_eq!(foreign_key.read_state.mvcc.current_row_id(), fk_row_id);
    assert_eq!(foreign_key.durable_wal_records().len(), fk_wal);
    let fk_admissions_after = foreign_key.prepared_transaction_class_admissions();
    assert_eq!(
        fk_admissions_after.w1
            + fk_admissions_after.t8
            + fk_admissions_after.t32
            + fk_admissions_after.general,
        fk_admissions_before.w1
            + fk_admissions_before.t8
            + fk_admissions_before.t32
            + fk_admissions_before.general
            + 1,
        "the FK route was admitted exactly once before commit serialization"
    );
    assert!(foreign_key
        .execute_relational_select_text("SELECT id FROM children")
        .unwrap()
        .rows
        .is_empty());
    assert!(foreign_key
        .execute_relational_select_text("SELECT id FROM parents")
        .unwrap()
        .rows
        .is_empty());
}

/// A serial classic wave that reaches the canonical cut before a prepared transaction commits
/// must reload the allocator only after acquiring that cut. Otherwise the wave can reuse the
/// prepared insert's device-native row identity when it resumes.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_commit_and_serial_classic_wave_share_fresh_row_id_basis() {
    let engine = seeded_prepared_engine();
    let prepared = prepare_int4_route(
        &engine,
        &["INSERT INTO prepared_accounts VALUES ($1, $2) RETURNING id".to_string()],
    );
    let bound = prepared
        .bind(int4_parameters([[100, 1_000].to_vec()]))
        .unwrap();
    let allocator_before = engine.read_state.mvcc.current_row_id();
    let reached = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    engine.set_serial_pre_commit_lock_hook(Arc::clone(&reached), Arc::clone(&resume));

    let (prepared_result, classic_result) = std::thread::scope(|scope| {
        let classic = scope.spawn(|| {
            engine.execute_dml_concurrent(141, "INSERT INTO prepared_accounts VALUES (101, 1010)")
        });
        reached.wait();
        let prepared_result = engine.submit_transaction(140, bound);
        resume.wait();
        (prepared_result, classic.join().unwrap())
    });
    let TransactionAdmissionResult::Predeclared(prepared_result) = prepared_result.unwrap() else {
        panic!("prepared insert returned the wrong admission result")
    };
    assert_eq!(prepared_result.class, TransactionClass::W1);
    classic_result.unwrap();

    let prepared_row_id = device_row_id_for_prepared_key(&engine, 100);
    let classic_row_id = device_row_id_for_prepared_key(&engine, 101);
    assert_eq!(prepared_row_id, allocator_before);
    assert_eq!(classic_row_id, allocator_before + 1);
    assert_ne!(prepared_row_id, classic_row_id);
    assert_eq!(
        engine.read_state.mvcc.current_row_id(),
        allocator_before + 2
    );
}

/// A recycled numeric CUDA address is not source identity. Prepared pins retain the original
/// source allocation and reject a synthetic ABA candidate whose numeric tag matches a different
/// live allocation but whose source Arc does not.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_pin_rejects_aba_numeric_match_without_source_arc_identity() {
    let original_engine = seeded_prepared_engine();
    let route = prepare_int4_route(
        &original_engine,
        &["SELECT balance FROM prepared_accounts WHERE id = $1".to_string()],
    );
    let original_pins = original_engine
        .validate_prepared_route_physical(&route.proof)
        .unwrap();
    let ((_, _, key_id), original_pin) = original_pins
        .indexes
        .iter()
        .next()
        .expect("prepared route must pin one named index");

    let replacement_engine = seeded_prepared_engine();
    let replacement_shards = replacement_engine.read_residency_shards();
    let replacement_shard = replacement_shards["prepared_accounts"]
        .iter()
        .find(|shard| shard.row_count != 0)
        .expect("replacement table must have a nonempty shard");
    let replacement_memory = Arc::clone(
        replacement_shard
            .device_memory
            .as_ref()
            .expect("replacement shard must be device resident"),
    );
    assert!(!Arc::ptr_eq(
        &original_pin.resident_guard,
        &replacement_memory
    ));

    let mut synthetic_aba = original_pin.clone();
    synthetic_aba.resident_device_ptr = replacement_memory.device_ptr();
    assert!(
        synthetic_aba
            .published_row_count
            .load(AtomicOrdering::Acquire)
            >= replacement_shard.row_count
    );
    let index_key = (
        "prepared_accounts".to_string(),
        replacement_shard.shard_id,
        *key_id,
    );
    let registry = Arc::new(Mutex::new(vec![PreparedPhysicalPins {
        table_generations: BTreeMap::new(),
        indexes: BTreeMap::from([(index_key, synthetic_aba)]),
    }]));
    let _required = PreparedIndexRequirementGuard::enter(registry);
    assert!(
        prepared_pinned_device_index(
            "prepared_accounts",
            replacement_shard.shard_id,
            *key_id,
            &replacement_memory,
            replacement_shard.row_count,
            replacement_engine.committed_seq(),
        )
        .is_none(),
        "numeric pointer equality must not authorize a pin from another source allocation"
    );
}

/// Purging the mutable cache after an in-place index kernel cannot strand a retained RC pin on
/// stale extent/posting metadata. The refreshed second statement requires the appended 65th slot,
/// so the basis-owned publication state is load-bearing and independent of the replaceable map.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_retained_pin_survives_post_index_kernel_cache_purge() {
    let engine = seeded_prepared_engine();
    let route = prepare_int4_route(
        &engine,
        &[
            "SELECT balance FROM prepared_accounts WHERE id = $1".to_string(),
            "SELECT balance FROM prepared_accounts WHERE id = $1".to_string(),
        ],
    );
    let bound = route
        .bind(int4_parameters([[1].to_vec(), [1].to_vec()]))
        .unwrap();
    PREPARED_PINNED_INDEX_HITS.store(0, AtomicOrdering::Relaxed);
    PREPARED_BASE_INDEX_MISSES.store(0, AtomicOrdering::Relaxed);
    PREPARED_PRIVATE_INDEX_REBUILD_ROWS.store(0, AtomicOrdering::Relaxed);
    let result = engine
        .submit_bound_prepared_transaction_instrumented(130, bound, |operation| {
            if operation != 0 {
                return;
            }
            let table = engine
                .relational_catalog_table("prepared_accounts")
                .unwrap();
            let key_id =
                crate::engine_residency::index_probe_key_id(&table, &table.indexes[0], 0).unwrap();
            let (shard_id, generation, published_rows, published_postings) = {
                let shards = engine.read_residency_shards();
                let shard = shards["prepared_accounts"]
                    .iter()
                    .find(|shard| shard.row_count != 0)
                    .expect("the seeded prepared table must have a nonempty shard");
                assert_eq!(shard.row_count, 64);
                let (published_rows, published_postings) = engine
                    .read_state
                    .residency
                    .shard_pk_device_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .get(&("prepared_accounts".to_string(), shard.shard_id, key_id))
                    .map(|entry| {
                        (
                            Arc::clone(&entry.published_row_count),
                            Arc::clone(&entry.published_has_postings),
                        )
                    })
                    .expect("the prepared pin basis must still be cache-accounted before purge");
                (
                    shard.shard_id,
                    Arc::clone(&shard.point_route_generation),
                    published_rows,
                    published_postings,
                )
            };
            // Turn the already-pinned allocation into an optional cache owner for this race only.
            // That lets the concurrent commit complete after cache retirement instead of invoking
            // mandatory named-index repair; the prepared RC proof continues to own the exact Arc.
            let table_oid = engine
                .relational_catalog_table("prepared_accounts")
                .unwrap()
                .oid;
            engine
                .read_state
                .residency
                .named_index_publications
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&table_oid);
            let reached = Arc::new(std::sync::Barrier::new(2));
            let resume = Arc::new(std::sync::Barrier::new(2));
            engine.set_shard_pk_index_append_post_launch_hook(
                Arc::clone(&reached),
                Arc::clone(&resume),
            );
            std::thread::scope(|scope| {
                let writer = scope.spawn(|| {
                    engine.execute_text(
                        131,
                        "UPDATE prepared_accounts SET balance = balance + 100 WHERE id = 1",
                    )
                });
                reached.wait();
                engine
                    .read_state
                    .residency
                    .purge_shard_pk_index_for_table("prepared_accounts");
                resume.wait();
                writer.join().unwrap().unwrap();
            });
            let shards = engine.read_residency_shards();
            let shard = shards["prepared_accounts"]
                .iter()
                .find(|shard| shard.shard_id == shard_id)
                .expect("the in-place append must retain its shard");
            assert_eq!(shard.row_count, 65);
            assert!(
                !Arc::ptr_eq(&generation, &shard.point_route_generation),
                "descriptor publication must rotate the mutable-cache route token"
            );
            assert_eq!(published_rows.load(AtomicOrdering::Acquire), 65);
            assert!(published_postings.load(AtomicOrdering::Relaxed));
            reset_slot_scan_probes();
        })
        .unwrap();
    assert_eq!(selected_int4(&result.operations[0]), 10);
    assert_eq!(selected_int4(&result.operations[1]), 110);
    assert_index_read(&result.operations[1]);
    assert_no_slot_scan_probes();
    assert!(PREPARED_PINNED_INDEX_HITS.load(AtomicOrdering::Relaxed) > 0);
    assert_eq!(PREPARED_BASE_INDEX_MISSES.load(AtomicOrdering::Relaxed), 0);
    assert_eq!(
        PREPARED_PRIVATE_INDEX_REBUILD_ROWS.load(AtomicOrdering::Relaxed),
        0
    );
}
