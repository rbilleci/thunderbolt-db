use super::*;
use crate::engine_transaction_delta::TransactionGpuReservation;
use std::sync::{mpsc, Arc};
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
    let operation_payload = Engine::decode_engine_operation(&envelope.fragments[0].body).unwrap();
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

/// WRITE-001 recovery vertical for the consumed typed transaction carrier.  This must not be
/// covered only by the generic transaction-record test: the live statement has already built a
/// private GPU shard from its columnar payload before the canonical record is bound at COMMIT.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_fixed_transaction_insert_reopens_from_its_canonical_wal_record() {
    let runtime = CudaDriverRuntime::probe()
        .expect("the explicitly requested typed INSERT recovery proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(
        runtime.driver_available && runtime.device_count > 0,
        "the explicitly requested typed INSERT recovery proof requires an NVIDIA device"
    );

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_001,
            parsed(
                "CREATE TABLE typed_reopen \
                 (id int4 DEFAULT 41, tally int8 DEFAULT 7000000000)",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_003,
            parsed("CREATE INDEX typed_reopen_by_tally ON typed_reopen (tally)"),
        )
        .unwrap();
    engine
        .submit_transaction(44_004, parsed("INSERT INTO typed_reopen VALUES (1, 1)"))
        .unwrap();
    engine
        .publish_relational_resident_indexes("typed_reopen")
        .unwrap();

    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_002, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            44_002,
            parsed("INSERT INTO typed_reopen (tally) VALUES (NULL), (9000000000)"),
        )
        .unwrap();
    assert!(
        engine.durable_wal_records().len() == 3,
        "the typed private generation must not become durable before COMMIT"
    );
    engine.submit_transaction(44_002, parsed("COMMIT")).unwrap();

    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 1, "{probe:?}");

    let records = engine.durable_wal_records();
    assert_eq!(
        records.len(),
        4,
        "one table CREATE, one index CREATE, one seed, plus one typed transaction terminal"
    );
    let envelope = gpu_db_wal::decode_canonical_record_payload(&records[3].payload)
        .expect("decode the typed transaction canonical envelope")
        .expect("typed transaction must use a canonical envelope");
    assert!(
        Engine::canonical_envelope_is_codec5(&envelope),
        "the typed fixed-width transaction must not select a resolved binary INSERT authority"
    );
    let fragments = envelope
        .fragments
        .iter()
        .map(|fragment| gpu_db_wal::CanonicalFragmentRef {
            kind: fragment.kind,
            body: &fragment.body,
        })
        .collect::<Vec<_>>();
    let replay = crate::typed_insert_aggregate::decode_closed_semantics_v2_replay(
        &envelope.header,
        &envelope.outcome,
        &fragments,
    )
    .expect("strictly close the typed fixed-width codec-5 transaction")
    .expect("typed fixed-width transaction must select semantics-v2 replay");
    assert_eq!(replay.metadata().affected_rows, 2);
    assert_eq!(
        engine.relational_named_index_covered_rows("typed_reopen"),
        Some(3),
        "the typed terminal must extend the named GPU index beyond its pre-transaction seed"
    );

    let select = match parse_command("SELECT id, tally FROM typed_reopen").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    let live_result = engine.execute_relational_select(&select).unwrap();
    assert!(matches!(live_result.executed_target, DeviceTarget::Gpu(_)));
    let live = live_result.rows;
    assert_eq!(
        live,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(1)],
            vec![SqlValue::Int4(41), SqlValue::Null],
            vec![SqlValue::Int4(41), SqlValue::Int8(9_000_000_000)],
        ]
    );

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert_eq!(
        recovered.relational_named_index_covered_rows("typed_reopen"),
        Some(3),
        "fresh replay must rebuild complete named GPU index coverage"
    );
    let replayed = recovered.execute_relational_select(&select).unwrap();
    assert!(matches!(replayed.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        replayed.rows, live,
        "fresh replay must install the typed transaction's exact final row images"
    );
    assert_eq!(
        recovered.read_state.mvcc.current_row_id(),
        engine.read_state.mvcc.current_row_id(),
        "fresh replay must restore the typed transaction allocator high-water"
    );
}

/// A fixed-width physical payload uses its pre-WAL GPU key proof for same-statement duplicates,
/// then extends that same private GPU proof to reject cross-statement duplicates at their second
/// statement while retaining the canonical history check for ABA conflicts. It remains a physical
/// strategy beneath the sole typed codec-5 lifecycle, not a separate INSERT carrier.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_fixed_transaction_insert_uses_device_unique_indexes() {
    let runtime =
        CudaDriverRuntime::probe().expect("the typed UNIQUE-index proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(
        runtime.driver_available && runtime.device_count > 0,
        "the typed UNIQUE-index proof requires an NVIDIA device"
    );

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_101,
            parsed(
                "CREATE TABLE typed_unique_decline \
                 (id int4 PRIMARY KEY, tally int8, \
                  CONSTRAINT typed_unique_positive CHECK (tally > 0))",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_103,
            parsed("INSERT INTO typed_unique_decline VALUES (0, 1)"),
        )
        .unwrap();
    engine
        .publish_relational_resident_indexes("typed_unique_decline")
        .unwrap();
    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_102, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            44_102,
            parsed("INSERT INTO typed_unique_decline VALUES (1, 10), (2, 20)"),
        )
        .unwrap();
    engine.submit_transaction(44_102, parsed("COMMIT")).unwrap();

    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 1, "{probe:?}");
    let result = engine
        .execute_relational_select_text("SELECT id, tally FROM typed_unique_decline ORDER BY id")
        .unwrap();
    assert!(matches!(result.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(0), SqlValue::Int8(1)],
            vec![SqlValue::Int4(1), SqlValue::Int8(10)],
            vec![SqlValue::Int4(2), SqlValue::Int8(20)],
        ]
    );

    // The typed private stage consumes the same device-local key proof before it can allocate a
    // private generation or write WAL. An intra-statement duplicate must therefore leave the
    // active overlay untouched rather than falling back to a generic row delta.
    let wal_before_duplicate = engine.durable_wal_records().len();
    engine.submit_transaction(44_104, parsed("BEGIN")).unwrap();
    let duplicate = engine
        .submit_transaction(
            44_104,
            parsed("INSERT INTO typed_unique_decline VALUES (3, 30), (3, 31)"),
        )
        .unwrap_err();
    assert!(
        duplicate.to_string().contains("duplicate key"),
        "{duplicate}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before_duplicate);
    assert!(engine
        .transaction_snapshot_handle(44_104)
        .unwrap()
        .transaction_delta_is_empty());
    engine
        .submit_transaction(44_104, parsed("ROLLBACK"))
        .unwrap();

    // Separate typed statements receive distinct immutable private shards. The second statement
    // temporarily extends the one proven private generation for the same GPU validator, then
    // restores its exact predecessor when the duplicate is rejected before WAL.
    let wal_before_cross_statement = engine.durable_wal_records().len();
    engine.submit_transaction(44_107, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            44_107,
            parsed("INSERT INTO typed_unique_decline VALUES (5, 50)"),
        )
        .unwrap();
    let cross_statement = engine
        .submit_transaction(
            44_107,
            parsed("INSERT INTO typed_unique_decline VALUES (5, 51)"),
        )
        .unwrap_err();
    assert!(
        cross_statement.to_string().contains("duplicate key"),
        "{cross_statement}"
    );
    assert_eq!(
        engine.durable_wal_records().len(),
        wal_before_cross_statement,
        "the second-statement private GPU unique proof must run before WAL"
    );
    engine
        .submit_transaction(44_107, parsed("ROLLBACK"))
        .unwrap();

    // TRUNCATE removes the base generation, but rows staged after it still survive into the
    // canonical record. Their separate typed shards therefore need the same final GPU duplicate
    // proof; reset cannot be a blanket exemption for the table.
    let wal_before_reset_duplicate = engine.durable_wal_records().len();
    engine.submit_transaction(44_108, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(44_108, parsed("TRUNCATE typed_unique_decline"))
        .unwrap();
    engine
        .submit_transaction(
            44_108,
            parsed("INSERT INTO typed_unique_decline VALUES (6, 60)"),
        )
        .unwrap();
    let reset_duplicate = engine
        .submit_transaction(
            44_108,
            parsed("INSERT INTO typed_unique_decline VALUES (6, 61)"),
        )
        .unwrap_err();
    assert!(
        reset_duplicate.to_string().contains("duplicate key"),
        "{reset_duplicate}"
    );
    assert_eq!(
        engine.durable_wal_records().len(),
        wal_before_reset_duplicate,
        "the post-reset second-statement private GPU proof must run before WAL"
    );
    engine
        .submit_transaction(44_108, parsed("ROLLBACK"))
        .unwrap();

    // A key already visible at BEGIN is rejected by the same statement-local GPU proof. A later
    // concurrent history change remains a serialization concern, but an already-visible key is
    // PostgreSQL's ordinary duplicate-key error.
    let wal_before_existing = engine.durable_wal_records().len();
    engine.submit_transaction(44_106, parsed("BEGIN")).unwrap();
    let existing = engine
        .submit_transaction(
            44_106,
            parsed("INSERT INTO typed_unique_decline VALUES (0, 99)"),
        )
        .unwrap_err();
    assert!(existing.to_string().contains("duplicate key"), "{existing}");
    assert_eq!(engine.durable_wal_records().len(), wal_before_existing);
    engine
        .submit_transaction(44_106, parsed("ROLLBACK"))
        .unwrap();
    assert_eq!(
        engine
            .execute_relational_select_text(
                "SELECT id, tally FROM typed_unique_decline ORDER BY id",
            )
            .unwrap()
            .rows,
        result.rows,
        "the failed private generation must not publish the conflicting key"
    );

    let wal_before_check = engine.durable_wal_records().len();
    engine.submit_transaction(44_105, parsed("BEGIN")).unwrap();
    let check = engine
        .submit_transaction(
            44_105,
            parsed("INSERT INTO typed_unique_decline VALUES (4, -1)"),
        )
        .unwrap_err();
    assert!(check.to_string().contains("check constraint"), "{check}");
    assert_eq!(engine.durable_wal_records().len(), wal_before_check);
    assert!(engine
        .transaction_snapshot_handle(44_105)
        .unwrap()
        .transaction_delta_is_empty());
    engine
        .submit_transaction(44_105, parsed("ROLLBACK"))
        .unwrap();
}

/// NULL-bearing unique keys are deliberately excluded by the device predicate. Two separately
/// staged typed shards therefore commit, and their exact canonical rows survive recovery.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_fixed_transaction_unique_nulls_reopen_from_canonical_wal() {
    let runtime = CudaDriverRuntime::probe()
        .expect("the typed UNIQUE NULL recovery proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(
        runtime.driver_available && runtime.device_count > 0,
        "the typed UNIQUE NULL recovery proof requires an NVIDIA device"
    );

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_108,
            parsed(
                "CREATE TABLE typed_unique_null_reopen \
                 (id int4 PRIMARY KEY, tally int8, \
                  CONSTRAINT typed_unique_null_reopen_tally UNIQUE (tally))",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_110,
            parsed("INSERT INTO typed_unique_null_reopen VALUES (0, 1)"),
        )
        .unwrap();
    engine
        .publish_relational_resident_indexes("typed_unique_null_reopen")
        .unwrap();
    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_109, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            44_109,
            parsed("INSERT INTO typed_unique_null_reopen VALUES (1, NULL)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_109,
            parsed("INSERT INTO typed_unique_null_reopen VALUES (2, NULL)"),
        )
        .unwrap();
    engine.submit_transaction(44_109, parsed("COMMIT")).unwrap();
    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 2, "{probe:?}");

    let records = engine.durable_wal_records();
    let envelope = gpu_db_wal::decode_canonical_record_payload(&records.last().unwrap().payload)
        .expect("decode the typed NULL transaction envelope")
        .expect("typed NULL transaction must use the canonical envelope");
    assert!(
        Engine::canonical_envelope_is_codec5(&envelope),
        "two private NULL-bearing typed INSERTs must not select the resolved transaction record"
    );
    let select = match parse_command("SELECT id, tally FROM typed_unique_null_reopen ORDER BY id")
        .unwrap()
    {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    let live = engine.execute_relational_select(&select).unwrap();
    assert!(matches!(live.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        live.rows,
        vec![
            vec![SqlValue::Int4(0), SqlValue::Int8(1)],
            vec![SqlValue::Int4(1), SqlValue::Null],
            vec![SqlValue::Int4(2), SqlValue::Null],
        ]
    );
    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    let replayed = recovered.execute_relational_select(&select).unwrap();
    assert!(matches!(replayed.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(replayed.rows, live.rows);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_fixed_transaction_insert_uses_device_check_validation() {
    let runtime = CudaDriverRuntime::probe().expect("the typed CHECK proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(
        runtime.driver_available && runtime.device_count > 0,
        "the typed CHECK proof requires an NVIDIA device"
    );

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_150,
            parsed(
                "CREATE TABLE typed_checked \
                 (id int4, tally int8, \
                  CONSTRAINT typed_checked_positive CHECK (tally > 0))",
            ),
        )
        .unwrap();
    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_151, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            44_151,
            parsed("INSERT INTO typed_checked VALUES (1, 10), (2, 20)"),
        )
        .unwrap();
    engine.submit_transaction(44_151, parsed("COMMIT")).unwrap();
    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 1, "{probe:?}");

    let wal_before_check = engine.durable_wal_records().len();
    engine.submit_transaction(44_152, parsed("BEGIN")).unwrap();
    let check = engine
        .submit_transaction(44_152, parsed("INSERT INTO typed_checked VALUES (3, -1)"))
        .unwrap_err();
    assert!(check.to_string().contains("check constraint"), "{check}");
    assert_eq!(engine.durable_wal_records().len(), wal_before_check);
    assert!(engine
        .transaction_snapshot_handle(44_152)
        .unwrap()
        .transaction_delta_is_empty());
    engine
        .submit_transaction(44_152, parsed("ROLLBACK"))
        .unwrap();

    let rows = engine
        .execute_relational_select_text("SELECT id, tally FROM typed_checked ORDER BY id")
        .unwrap();
    assert!(matches!(rows.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        rows.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(10)],
            vec![SqlValue::Int4(2), SqlValue::Int8(20)],
        ]
    );
}

/// RETURNING reads exactly the new private device shard: duplicate/out-of-order projection stays
/// on the GPU, the response precedes durability, and the same staged images commit normally.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_fixed_transaction_insert_projects_private_shard_returning_before_commit() {
    let runtime =
        CudaDriverRuntime::probe().expect("the typed RETURNING proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(
        runtime.driver_available && runtime.device_count > 0,
        "the typed RETURNING proof requires an NVIDIA device"
    );

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_154,
            parsed("CREATE TABLE typed_private_returning (id int4, tally int8)"),
        )
        .unwrap();
    let wal_before_insert = engine.durable_wal_records().len();
    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_155, parsed("BEGIN")).unwrap();
    let result = engine
        .execute_dml_in_transaction_with_result(
            44_155,
            "INSERT INTO typed_private_returning VALUES (1, 10), (2, 20) \
             RETURNING tally, id, tally",
        )
        .unwrap();
    assert_eq!(result.rows_affected, 2);
    let returning = result.returning.expect("typed INSERT RETURNING result");
    assert!(matches!(returning.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        returning.rows,
        vec![
            vec![SqlValue::Int8(10), SqlValue::Int4(1), SqlValue::Int8(10)],
            vec![SqlValue::Int8(20), SqlValue::Int4(2), SqlValue::Int8(20)],
        ]
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before_insert);
    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 0, "{probe:?}");

    engine.submit_transaction(44_155, parsed("COMMIT")).unwrap();
    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 1, "{probe:?}");
    let rows = engine
        .execute_relational_select_text("SELECT id, tally FROM typed_private_returning ORDER BY id")
        .unwrap();
    assert_eq!(
        rows.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(10)],
            vec![SqlValue::Int4(2), SqlValue::Int8(20)],
        ]
    );
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .execute_relational_select_text(
                "SELECT id, tally FROM typed_private_returning ORDER BY id",
            )
            .unwrap()
            .rows,
        rows.rows,
        "a committed statement that returned private-shard rows reopens from its canonical record"
    );
}

/// The private-shard projection preserves the nullable vector layout rather than interpreting a
/// zero payload as a host-side value while constructing the RETURNING response.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_fixed_transaction_private_shard_returning_preserves_nulls() {
    let runtime = CudaDriverRuntime::probe()
        .expect("the typed nullable RETURNING proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(runtime.driver_available && runtime.device_count > 0);

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_152,
            parsed("CREATE TABLE typed_private_returning_nulls (id int4, tally int8)"),
        )
        .unwrap();
    engine.submit_transaction(44_153, parsed("BEGIN")).unwrap();
    let result = engine
        .execute_dml_in_transaction_with_result(
            44_153,
            "INSERT INTO typed_private_returning_nulls VALUES (NULL, 10), (2, NULL) \
             RETURNING tally, id",
        )
        .unwrap();
    let returning = result.returning.expect("typed nullable RETURNING result");
    assert!(matches!(returning.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        returning.rows,
        vec![
            vec![SqlValue::Int8(10), SqlValue::Null],
            vec![SqlValue::Null, SqlValue::Int4(2)],
        ]
    );
    engine
        .submit_transaction(44_153, parsed("ROLLBACK"))
        .unwrap();
    assert!(engine
        .execute_relational_select_text("SELECT id FROM typed_private_returning_nulls")
        .unwrap()
        .rows
        .is_empty());
}

/// TEXT follows the same consumed typed batch into a dense private GPU shard. A transaction read
/// must read its offsets/blob on device before COMMIT, while the canonical row images remain the
/// sole durable representation used to reopen the committed transaction.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_text_transaction_private_shard_read_reopens_from_canonical_wal() {
    let runtime = CudaDriverRuntime::probe().expect("the typed TEXT proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(runtime.driver_available && runtime.device_count > 0);

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_156,
            parsed("CREATE TABLE typed_private_text (id int4, body text)"),
        )
        .unwrap();
    let wal_before_insert = engine.durable_wal_records().len();
    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_157, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            44_157,
            parsed("INSERT INTO typed_private_text VALUES (1, 'green'), (2, NULL), (3, 'violet')"),
        )
        .unwrap();
    assert_eq!(engine.durable_wal_records().len(), wal_before_insert);
    let _probe = engine.insert_probe_snapshot().delta_since(probe_before);

    let select = match parse_command("SELECT id, body FROM typed_private_text ORDER BY id").unwrap()
    {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    let private_rows = engine
        .execute_relational_select_in_transaction(44_157, &select)
        .unwrap();
    assert!(matches!(private_rows.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        private_rows.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("green".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Null],
            vec![SqlValue::Int4(3), SqlValue::Text("violet".to_string())],
        ]
    );

    engine.submit_transaction(44_157, parsed("COMMIT")).unwrap();
    let expected = vec![
        vec![SqlValue::Int4(1), SqlValue::Text("green".to_string())],
        vec![SqlValue::Int4(2), SqlValue::Null],
        vec![SqlValue::Int4(3), SqlValue::Text("violet".to_string())],
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

/// The private `RETURNING` projection itself must decode TEXT from the staged GPU descriptor;
/// returning a neighboring fixed-width column alone would not prove that the text offsets/blob
/// reached the general GPU result path.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_text_transaction_private_shard_projects_text_returning_before_commit() {
    let runtime =
        CudaDriverRuntime::probe().expect("the typed TEXT RETURNING proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(runtime.driver_available && runtime.device_count > 0);

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_158,
            parsed("CREATE TABLE typed_private_text_returning (id int4, body text)"),
        )
        .unwrap();
    let wal_before = engine.durable_wal_records().len();
    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_159, parsed("BEGIN")).unwrap();
    let command = parse_command(
        "INSERT INTO typed_private_text_returning VALUES (1, 'left'), (2, NULL), (3, 'right') RETURNING body, id, body",
    )
    .expect("TEXT RETURNING statement parses before private-shard projection");
    let result = engine
        .execute_parsed_dml_in_transaction_with_result(44_159, command)
        .unwrap();
    assert_eq!(result.rows_affected, 3);
    let returning = result.returning.expect("typed TEXT RETURNING result");
    assert!(matches!(returning.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        returning.rows,
        vec![
            vec![
                SqlValue::Text("left".to_string()),
                SqlValue::Int4(1),
                SqlValue::Text("left".to_string()),
            ],
            vec![SqlValue::Null, SqlValue::Int4(2), SqlValue::Null],
            vec![
                SqlValue::Text("right".to_string()),
                SqlValue::Int4(3),
                SqlValue::Text("right".to_string()),
            ],
        ]
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    let _probe = engine.insert_probe_snapshot().delta_since(probe_before);
    engine.submit_transaction(44_159, parsed("COMMIT")).unwrap();
    assert_eq!(
        engine
            .execute_relational_select_text(
                "SELECT id, body FROM typed_private_text_returning ORDER BY id",
            )
            .unwrap()
            .rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("left".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Null],
            vec![SqlValue::Int4(3), SqlValue::Text("right".to_string())],
        ]
    );
}

/// A result-frame fault after a TEXT-bearing private shard allocation must release the whole
/// generation. Retrying the exact statement keeps its dense text offsets/blob on the same typed
/// path rather than falling back to a row delta.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_text_transaction_private_shard_returning_frame_fault_drops_and_retries() {
    let runtime =
        CudaDriverRuntime::probe().expect("the typed TEXT retry proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(runtime.driver_available && runtime.device_count > 0);

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_158,
            parsed("CREATE TABLE typed_private_text_fault (id int4, body text)"),
        )
        .unwrap();
    let resident_before = engine.relational_resident_bytes_for_gpu(0);
    let wal_before = engine.durable_wal_records().len();
    engine.submit_transaction(44_159, parsed("BEGIN")).unwrap();

    crate::engine_expr::fail_next_device_result_frame_read_for_test();
    let error = engine
        .execute_dml_in_transaction_with_result(
            44_159,
            "INSERT INTO typed_private_text_fault VALUES (1, 'first'), (2, 'second') RETURNING id",
        )
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("test-only device result frame read fault"));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.relational_resident_bytes_for_gpu(0), resident_before);
    assert!(engine.transaction_snapshot_handle(44_159).is_some());
    assert!(engine
        .transaction_private_gpu_bytes
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .is_empty());

    let retried = engine
        .execute_dml_in_transaction_with_result(
            44_159,
            "INSERT INTO typed_private_text_fault VALUES (1, 'first'), (2, 'second') RETURNING id",
        )
        .unwrap();
    assert_eq!(retried.rows_affected, 2);
    assert_eq!(
        retried.returning.expect("TEXT retry result").rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(2)]]
    );
    engine.submit_transaction(44_159, parsed("COMMIT")).unwrap();
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    assert_eq!(
        engine
            .execute_relational_select_text(
                "SELECT id, body FROM typed_private_text_fault ORDER BY id",
            )
            .unwrap()
            .rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("first".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Text("second".to_string())],
        ]
    );
}

/// BOOL vectors are bit-packed in the dense private shard. This verifies the descriptor reaches
/// the general GPU RETURNING path before commit, retains NULL semantics, and remains bound to
/// the canonical transaction record used for recovery.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_bool_transaction_private_shard_projects_bool_returning_and_recovers() {
    let runtime = CudaDriverRuntime::probe().expect("the typed BOOL proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(runtime.driver_available && runtime.device_count > 0);

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_260,
            parsed("CREATE TABLE typed_private_bool (id int4, enabled bool)"),
        )
        .unwrap();
    let wal_before = engine.durable_wal_records().len();
    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_261, parsed("BEGIN")).unwrap();
    let result = engine
        .execute_dml_in_transaction_with_result(
            44_261,
            "INSERT INTO typed_private_bool VALUES (1, true), (2, NULL), (3, false) \
             RETURNING enabled, id, enabled",
        )
        .unwrap();
    assert_eq!(result.rows_affected, 3);
    let returning = result.returning.expect("typed BOOL RETURNING result");
    assert!(matches!(returning.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        returning.rows,
        vec![
            vec![
                SqlValue::Bool(true),
                SqlValue::Int4(1),
                SqlValue::Bool(true)
            ],
            vec![SqlValue::Null, SqlValue::Int4(2), SqlValue::Null],
            vec![
                SqlValue::Bool(false),
                SqlValue::Int4(3),
                SqlValue::Bool(false)
            ],
        ]
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    let _probe = engine.insert_probe_snapshot().delta_since(probe_before);

    engine.submit_transaction(44_261, parsed("COMMIT")).unwrap();
    let expected = vec![
        vec![SqlValue::Int4(1), SqlValue::Bool(true)],
        vec![SqlValue::Int4(2), SqlValue::Null],
        vec![SqlValue::Int4(3), SqlValue::Bool(false)],
    ];
    assert_eq!(
        engine
            .execute_relational_select_text(
                "SELECT id, enabled FROM typed_private_bool ORDER BY id"
            )
            .unwrap()
            .rows,
        expected
    );
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .execute_relational_select_text(
                "SELECT id, enabled FROM typed_private_bool ORDER BY id"
            )
            .unwrap()
            .rows,
        expected
    );
}

/// A result-frame failure after a BOOL private shard allocation must roll back the packed bitmap
/// generation completely, so retrying cannot silently use the legacy row-delta path.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_bool_transaction_private_shard_returning_frame_fault_drops_and_retries() {
    let runtime =
        CudaDriverRuntime::probe().expect("the typed BOOL retry proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(runtime.driver_available && runtime.device_count > 0);

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_262,
            parsed("CREATE TABLE typed_private_bool_fault (id int4, enabled bool)"),
        )
        .unwrap();
    let resident_before = engine.relational_resident_bytes_for_gpu(0);
    let wal_before = engine.durable_wal_records().len();
    engine.submit_transaction(44_263, parsed("BEGIN")).unwrap();
    let probe_before = engine.insert_probe_snapshot();

    crate::engine_expr::fail_next_device_result_frame_read_for_test();
    let error = engine
        .execute_dml_in_transaction_with_result(
            44_263,
            "INSERT INTO typed_private_bool_fault VALUES (1, true), (2, false) RETURNING enabled",
        )
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("test-only device result frame read fault"));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.relational_resident_bytes_for_gpu(0), resident_before);
    assert!(engine.transaction_snapshot_handle(44_263).is_some());
    assert!(engine
        .transaction_private_gpu_bytes
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .is_empty());

    let retried = engine
        .execute_dml_in_transaction_with_result(
            44_263,
            "INSERT INTO typed_private_bool_fault VALUES (1, true), (2, false) RETURNING enabled",
        )
        .unwrap();
    assert_eq!(retried.rows_affected, 2);
    assert_eq!(
        retried.returning.expect("BOOL retry result").rows,
        vec![vec![SqlValue::Bool(true)], vec![SqlValue::Bool(false)]]
    );
    engine.submit_transaction(44_263, parsed("COMMIT")).unwrap();
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    let _probe = engine.insert_probe_snapshot().delta_since(probe_before);
}

/// Autocommit result-bearing INSERT uses the same one-statement private overlay as an explicit
/// transaction. A terminal result-frame fault must therefore roll that implicit overlay back
/// completely before it can append WAL or publish any BOOL/TEXT row.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_autocommit_returning_frame_fault_rolls_back_private_overlay() {
    let runtime = CudaDriverRuntime::probe()
        .expect("the typed autocommit RETURNING fault proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(runtime.driver_available && runtime.device_count > 0);

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .execute_text(
            44_270,
            "CREATE TABLE typed_autocommit_returning_fault (id int4, enabled bool, body text)",
        )
        .unwrap();
    let resident_before = engine.relational_resident_bytes_for_gpu(0);
    let wal_before = engine.durable_wal_records().len();
    let probe_before = engine.insert_probe_snapshot();

    crate::engine_expr::fail_next_device_result_frame_read_for_test();
    let error = engine
        .execute_dml_concurrent_with_result(
            44_271,
            "INSERT INTO typed_autocommit_returning_fault VALUES (1, true, 'lost') \
             RETURNING body, enabled, id, body",
        )
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("test-only device result frame read fault"));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.relational_resident_bytes_for_gpu(0), resident_before);
    assert!(engine.transaction_snapshot_handle(44_271).is_none());
    assert!(engine
        .transaction_private_gpu_bytes
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .is_empty());
    assert!(engine
        .execute_relational_select_text("SELECT id FROM typed_autocommit_returning_fault")
        .unwrap()
        .rows
        .is_empty());

    let retried = engine
        .execute_dml_concurrent_with_result(
            44_271,
            "INSERT INTO typed_autocommit_returning_fault VALUES (1, true, 'kept'), (2, NULL, NULL) \
             RETURNING body, enabled, id, body",
        )
        .unwrap();
    assert_eq!(retried.rows_affected, 2);
    let returning = retried
        .returning
        .expect("typed autocommit RETURNING result");
    assert!(matches!(returning.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        returning.rows,
        vec![
            vec![
                SqlValue::Text("kept".to_string()),
                SqlValue::Bool(true),
                SqlValue::Int4(1),
                SqlValue::Text("kept".to_string()),
            ],
            vec![
                SqlValue::Null,
                SqlValue::Null,
                SqlValue::Int4(2),
                SqlValue::Null
            ],
        ]
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 1, "{probe:?}");
    let expected = vec![
        vec![
            SqlValue::Int4(1),
            SqlValue::Bool(true),
            SqlValue::Text("kept".to_string()),
        ],
        vec![SqlValue::Int4(2), SqlValue::Null, SqlValue::Null],
    ];
    assert_eq!(
        engine
            .execute_relational_select_text(
                "SELECT id, enabled, body FROM typed_autocommit_returning_fault ORDER BY id",
            )
            .unwrap()
            .rows,
        expected
    );
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .execute_relational_select_text(
                "SELECT id, enabled, body FROM typed_autocommit_returning_fault ORDER BY id",
            )
            .unwrap()
            .rows,
        expected
    );
}

/// A pre-WAL canonical COMMIT rejection happens after the implicit overlay has produced its
/// device RETURNING frame. It must cancel that private generation exactly like a statement fault,
/// leaving the same autocommit identity free to retry a nonconflicting typed INSERT.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_autocommit_returning_pre_wal_commit_rejection_cancels_private_overlay() {
    let runtime = CudaDriverRuntime::probe()
        .expect("the typed autocommit commit-rejection proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(runtime.driver_available && runtime.device_count > 0);

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .execute_text(
            44_273,
            "CREATE TABLE typed_autocommit_returning_unique \
             (id int4 PRIMARY KEY, enabled bool, body text)",
        )
        .unwrap();
    engine
        .execute_text(
            44_274,
            "INSERT INTO typed_autocommit_returning_unique VALUES (1, false, 'seed')",
        )
        .unwrap();
    engine
        .publish_relational_resident_indexes("typed_autocommit_returning_unique")
        .unwrap();
    let resident_before = engine.relational_resident_bytes_for_gpu(0);
    let wal_before = engine.durable_wal_records().len();
    let probe_before = engine.insert_probe_snapshot();

    let rejected = engine
        .execute_dml_concurrent_with_result(
            44_275,
            "INSERT INTO typed_autocommit_returning_unique VALUES (1, true, 'conflict') \
             RETURNING body, enabled, id",
        )
        .unwrap_err();
    assert!(
        rejected.to_string().contains("device unique conflict"),
        "{rejected}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(
        engine.relational_resident_bytes_for_gpu(0) <= resident_before,
        "the rejected private overlay must not add resident bytes: before={resident_before}, after={}",
        engine.relational_resident_bytes_for_gpu(0),
    );
    assert!(engine.transaction_snapshot_handle(44_275).is_none());
    assert!(engine
        .transaction_private_gpu_bytes
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .is_empty());

    let retried = engine
        .execute_dml_concurrent_with_result(
            44_275,
            "INSERT INTO typed_autocommit_returning_unique VALUES (2, true, 'kept') \
             RETURNING body, enabled, id",
        )
        .unwrap();
    assert_eq!(retried.rows_affected, 1);
    let returning = retried.returning.expect("typed retry RETURNING result");
    assert!(matches!(returning.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        returning.rows,
        vec![vec![
            SqlValue::Text("kept".to_string()),
            SqlValue::Bool(true),
            SqlValue::Int4(2),
        ]]
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 1, "{probe:?}");
    assert_eq!(
        engine
            .execute_relational_select_text(
                "SELECT id, enabled, body FROM typed_autocommit_returning_unique ORDER BY id",
            )
            .unwrap()
            .rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Bool(false),
                SqlValue::Text("seed".to_string()),
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Bool(true),
                SqlValue::Text("kept".to_string()),
            ],
        ]
    );
}

/// A terminal GPU result-frame failure happens after the typed shard is allocated but before its
/// generation is published. The exact statement is then retryable in the same transaction: no
/// private charge, WAL record, or provisional row identity may escape the failed projection.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_fixed_transaction_private_shard_returning_frame_fault_drops_and_retries() {
    let runtime = CudaDriverRuntime::probe()
        .expect("the typed RETURNING frame-fault proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(runtime.driver_available && runtime.device_count > 0);

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_159,
            parsed("CREATE TABLE typed_private_returning_fault (id int4, tally int8)"),
        )
        .unwrap();
    let resident_before = engine.relational_resident_bytes_for_gpu(0);
    let wal_before = engine.durable_wal_records().len();
    engine.submit_transaction(44_160, parsed("BEGIN")).unwrap();

    crate::engine_expr::fail_next_device_result_frame_read_for_test();
    let error = engine
        .execute_dml_in_transaction_with_result(
            44_160,
            "INSERT INTO typed_private_returning_fault VALUES (1, 10), (2, 20) \
             RETURNING tally, id",
        )
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("test-only device result frame read fault"));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.relational_resident_bytes_for_gpu(0), resident_before);
    assert!(engine.transaction_snapshot_handle(44_160).is_some());
    assert!(engine
        .transaction_private_gpu_bytes
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .is_empty());

    let retried = engine
        .execute_dml_in_transaction_with_result(
            44_160,
            "INSERT INTO typed_private_returning_fault VALUES (1, 10), (2, 20) \
             RETURNING tally, id",
        )
        .unwrap();
    assert_eq!(retried.rows_affected, 2);
    assert_eq!(
        retried.returning.expect("retry result").rows,
        vec![
            vec![SqlValue::Int8(10), SqlValue::Int4(1)],
            vec![SqlValue::Int8(20), SqlValue::Int4(2)],
        ]
    );
    engine.submit_transaction(44_160, parsed("COMMIT")).unwrap();
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    assert_eq!(
        engine
            .execute_relational_select_text(
                "SELECT id, tally FROM typed_private_returning_fault ORDER BY id",
            )
            .unwrap()
            .rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(10)],
            vec![SqlValue::Int4(2), SqlValue::Int8(20)],
        ]
    );
}

/// A sealed domain witness must survive typed private staging: its base-type vectors remain GPU
/// resident while the exact domain OID/name binding is checked before consumption and again by
/// the transaction's pinned catalog generation at COMMIT.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_fixed_transaction_insert_uses_pinned_domain_binding_through_recovery() {
    let runtime =
        CudaDriverRuntime::probe().expect("the typed domain proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(
        runtime.driver_available && runtime.device_count > 0,
        "the typed domain proof requires an NVIDIA device"
    );

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(44_157, parsed("CREATE DOMAIN typed_amount AS int4"))
        .unwrap();
    engine
        .submit_transaction(
            44_158,
            parsed("CREATE TABLE typed_domain_rows (id typed_amount, tally int8)"),
        )
        .unwrap();
    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_159, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            44_159,
            parsed("INSERT INTO typed_domain_rows VALUES (1, 10), (2, 20)"),
        )
        .unwrap();
    engine.submit_transaction(44_159, parsed("COMMIT")).unwrap();
    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 1, "{probe:?}");

    let select = match parse_command("SELECT id, tally FROM typed_domain_rows ORDER BY id").unwrap()
    {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    let live = engine.execute_relational_select(&select).unwrap();
    assert!(matches!(live.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        live.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(10)],
            vec![SqlValue::Int4(2), SqlValue::Int8(20)],
        ]
    );
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert_eq!(
        recovered.execute_relational_select(&select).unwrap().rows,
        live.rows
    );
}

/// An FK-bearing child must retain its provider/consumer dependency closure through the typed
/// private generation. COMMIT then validates the final canonical rows with the existing GPU FK
/// operator; the negative case proves that staging itself has not acknowledged or published it.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_fixed_transaction_insert_uses_device_foreign_key_validation_through_recovery() {
    let runtime = CudaDriverRuntime::probe().expect("the typed FK proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(
        runtime.driver_available && runtime.device_count > 0,
        "the typed FK proof requires an NVIDIA device"
    );

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_160,
            parsed("CREATE TABLE typed_fk_parent (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_161,
            parsed("CREATE TABLE typed_fk_child (id int4, parent_id int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_162,
            parsed(
                "ALTER TABLE ONLY typed_fk_child ADD CONSTRAINT typed_fk_child_parent \
                 FOREIGN KEY (parent_id) REFERENCES typed_fk_parent(id)",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(44_163, parsed("INSERT INTO typed_fk_parent VALUES (7)"))
        .unwrap();
    engine
        .publish_relational_resident_indexes("typed_fk_parent")
        .unwrap();

    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_164, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            44_164,
            parsed("INSERT INTO typed_fk_child VALUES (1, 7), (2, 7)"),
        )
        .unwrap();
    engine.submit_transaction(44_164, parsed("COMMIT")).unwrap();
    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 1, "{probe:?}");

    let select =
        match parse_command("SELECT id, parent_id FROM typed_fk_child ORDER BY id").unwrap() {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        };
    let live = engine.execute_relational_select(&select).unwrap();
    assert!(matches!(live.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        live.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(7)],
            vec![SqlValue::Int4(2), SqlValue::Int4(7)],
        ]
    );
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert_eq!(
        recovered.execute_relational_select(&select).unwrap().rows,
        live.rows
    );

    let wal_before_invalid = engine.durable_wal_records().len();
    engine.submit_transaction(44_165, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(44_165, parsed("INSERT INTO typed_fk_child VALUES (3, 99)"))
        .unwrap();
    let invalid = engine
        .submit_transaction(44_165, parsed("COMMIT"))
        .unwrap_err();
    assert!(
        matches!(invalid, ExecuteError::Serialization(ref message) if message.contains("device foreign-key conflict")),
        "{invalid}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before_invalid);
    engine
        .submit_transaction(44_165, parsed("ROLLBACK"))
        .unwrap();
}

/// The typed child retains the exact parent history floor. A delete/reinsert ABA after staging
/// must be rejected before WAL even though the parent key is visible again at COMMIT.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_fixed_transaction_insert_rejects_foreign_key_provider_aba() {
    let runtime =
        CudaDriverRuntime::probe().expect("the typed FK ABA proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(runtime.driver_available && runtime.device_count > 0);

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_170,
            parsed("CREATE TABLE typed_fk_aba_parent (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_171,
            parsed("CREATE TABLE typed_fk_aba_child (id int4, parent_id int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_172,
            parsed(
                "ALTER TABLE ONLY typed_fk_aba_child ADD CONSTRAINT typed_fk_aba_child_parent \
                 FOREIGN KEY (parent_id) REFERENCES typed_fk_aba_parent(id)",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(44_173, parsed("INSERT INTO typed_fk_aba_parent VALUES (7)"))
        .unwrap();
    engine
        .publish_relational_resident_indexes("typed_fk_aba_parent")
        .unwrap();

    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_174, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            44_174,
            parsed("INSERT INTO typed_fk_aba_child VALUES (1, 7)"),
        )
        .unwrap();
    let _probe = engine.insert_probe_snapshot().delta_since(probe_before);
    engine
        .execute_dml_concurrent(44_175, "DELETE FROM typed_fk_aba_parent WHERE id = 7")
        .unwrap();
    engine
        .execute_dml_concurrent(44_176, "INSERT INTO typed_fk_aba_parent VALUES (7)")
        .unwrap();
    let wal_before_commit = engine.durable_wal_records().len();
    let commit = engine
        .submit_transaction(44_174, parsed("COMMIT"))
        .unwrap_err();
    assert!(
        matches!(commit, ExecuteError::Serialization(ref message) if message.contains("foreign-key dependency relation \"typed_fk_aba_parent\" changed")),
        "{commit}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before_commit);
    engine
        .submit_transaction(44_174, parsed("ROLLBACK"))
        .unwrap();
}

/// FK membership is decided from the final private generation, so a generic parent staged later
/// in the same explicit transaction may provide a key for an earlier typed child.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_fixed_transaction_insert_accepts_later_private_foreign_key_provider() {
    let runtime = CudaDriverRuntime::probe()
        .expect("the typed private FK-provider proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(runtime.driver_available && runtime.device_count > 0);

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_180,
            parsed("CREATE TABLE typed_fk_private_parent (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_181,
            parsed("CREATE TABLE typed_fk_private_child (id int4, parent_id int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_182,
            parsed(
                "ALTER TABLE ONLY typed_fk_private_child ADD CONSTRAINT typed_fk_private_child_parent \
                 FOREIGN KEY (parent_id) REFERENCES typed_fk_private_parent(id)",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_184,
            parsed("INSERT INTO typed_fk_private_parent VALUES (7)"),
        )
        .unwrap();
    engine
        .publish_relational_resident_indexes("typed_fk_private_parent")
        .unwrap();

    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_183, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            44_183,
            parsed("INSERT INTO typed_fk_private_child VALUES (1, 8)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_183,
            parsed("INSERT INTO typed_fk_private_parent VALUES (8)"),
        )
        .unwrap();
    engine.submit_transaction(44_183, parsed("COMMIT")).unwrap();
    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 2, "{probe:?}");
    let rows = engine
        .execute_relational_select_text("SELECT id, parent_id FROM typed_fk_private_child")
        .unwrap();
    assert_eq!(rows.rows, vec![vec![SqlValue::Int4(1), SqlValue::Int4(8)]]);
}

/// An explicit typed child COMMIT cannot cross a classic-wave parent delete after the delete has
/// applied but before its durability tail has been handed off.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_fixed_transaction_fk_commit_waits_across_classic_wave_tail_handoff() {
    let runtime =
        CudaDriverRuntime::probe().expect("the typed FK wave-tail proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(runtime.driver_available && runtime.device_count > 0);

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            44_190,
            parsed("CREATE TABLE typed_fk_tail_parent (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_191,
            parsed("CREATE TABLE typed_fk_tail_child (id int4, parent_id int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_192,
            parsed(
                "ALTER TABLE ONLY typed_fk_tail_child ADD CONSTRAINT typed_fk_tail_child_parent \
                 FOREIGN KEY (parent_id) REFERENCES typed_fk_tail_parent(id)",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_193,
            parsed("INSERT INTO typed_fk_tail_parent VALUES (7)"),
        )
        .unwrap();
    engine
        .publish_relational_resident_indexes("typed_fk_tail_parent")
        .unwrap();
    let engine = Arc::new(engine);

    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_194, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            44_194,
            parsed("INSERT INTO typed_fk_tail_child VALUES (1, 7)"),
        )
        .unwrap();
    let _probe = engine.insert_probe_snapshot().delta_since(probe_before);

    let reached = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    engine.set_wave_tail_handoff_hook(Arc::clone(&reached), Arc::clone(&resume));
    let delete = {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            engine.execute_dml_concurrent(44_195, "DELETE FROM typed_fk_tail_parent WHERE id = 7")
        })
    };
    reached.wait();

    let (done_tx, done_rx) = mpsc::channel();
    let commit = {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            let result = engine.submit_transaction(44_194, parsed("COMMIT"));
            done_tx.send(()).unwrap();
            result
        })
    };
    assert!(
        done_rx.recv_timeout(Duration::from_millis(100)).is_err(),
        "typed FK COMMIT crossed an applied classic wave before its tail was handed off"
    );
    resume.wait();
    delete.join().unwrap().unwrap();
    let result = commit.join().unwrap();
    assert!(
        matches!(result, Err(ExecuteError::Serialization(ref message)) if message.contains("foreign-key dependency relation \"typed_fk_tail_parent\" changed")),
        "{result:?}"
    );
    engine
        .submit_transaction(44_194, parsed("ROLLBACK"))
        .unwrap();
}

/// Typed INSERT owns no legacy row delta, but it must still reject a key that another transaction
/// claimed and released after this transaction's snapshot. The exact history verdict remains
/// device-native; current visibility alone cannot detect this ABA shape.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn explicit_transaction_unique_claim_release_after_snapshot_uses_history_validation() {
    let runtime =
        CudaDriverRuntime::probe().expect("the typed UNIQUE history proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(
        runtime.driver_available && runtime.device_count > 0,
        "the typed UNIQUE history proof requires an NVIDIA device"
    );

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine
        .submit_transaction(
            44_201,
            parsed(
                "CREATE TABLE typed_unique_history \
                 (id int4 PRIMARY KEY, tenant_key int4, \
                  CONSTRAINT typed_unique_history_tenant UNIQUE (tenant_key))",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            44_202,
            parsed("INSERT INTO typed_unique_history VALUES (1, 10)"),
        )
        .unwrap();
    engine
        .publish_relational_resident_indexes("typed_unique_history")
        .unwrap();

    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_203, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            44_203,
            parsed("INSERT INTO typed_unique_history VALUES (3, 30)"),
        )
        .unwrap();
    let _probe = engine.insert_probe_snapshot().delta_since(probe_before);

    engine
        .execute_dml_concurrent(44_204, "INSERT INTO typed_unique_history VALUES (4, 30)")
        .unwrap();
    engine
        .execute_dml_concurrent(44_205, "DELETE FROM typed_unique_history WHERE id = 4")
        .unwrap();

    let commit = engine
        .submit_transaction(44_203, parsed("COMMIT"))
        .unwrap_err();
    assert!(
        matches!(commit, ExecuteError::Serialization(ref message) if message.contains("device unique conflict")),
        "{commit}"
    );
    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 0, "{probe:?}");
    engine
        .submit_transaction(44_203, parsed("ROLLBACK"))
        .unwrap();

    let rows = engine
        .execute_relational_select_text(
            "SELECT id, tenant_key FROM typed_unique_history ORDER BY id",
        )
        .unwrap();
    assert!(matches!(rows.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(rows.rows, vec![vec![SqlValue::Int4(1), SqlValue::Int4(10)]]);
}

/// Published sequence transitions stay independently durable, while their materialized default
/// values and exact row identities now ride the same typed private artifact as the INSERT.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_fixed_transaction_insert_binds_published_sequence_defaults_through_recovery() {
    let runtime = CudaDriverRuntime::probe()
        .expect("the typed sequence-default proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(
        runtime.driver_available && runtime.device_count > 0,
        "the typed sequence-default proof requires an NVIDIA device"
    );

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(44_201, parsed("CREATE SEQUENCE typed_sequence_spine"))
        .unwrap();
    engine
        .submit_transaction(
            44_202,
            parsed(
                "CREATE TABLE typed_sequence_owner \
                 (id int4 DEFAULT nextval('typed_sequence_spine'::regclass), value int4)",
            ),
        )
        .unwrap();

    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(44_203, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            44_203,
            parsed("INSERT INTO typed_sequence_owner (value) VALUES (10), (20)"),
        )
        .unwrap();
    engine.submit_transaction(44_203, parsed("COMMIT")).unwrap();
    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 1, "{probe:?}");

    let committed_records = engine.durable_wal_records();
    let decoded = decode_binary_record(&committed_records.last().unwrap().payload).unwrap();
    let BinaryWalRecord::Transaction(record) = decoded else {
        panic!("typed sequence-default terminal must use a transaction record");
    };
    assert_eq!(record.mutations.len(), 2);
    assert_eq!(record.sequence_value_references.len(), 2);
    assert!(record
        .sequence_value_references
        .iter()
        .all(|reference| reference.default_expression && reference.row_id != 0));

    // A sequence transition is deliberately not rolled back, but the typed private generation
    // is: the next committed default proves this statement neither leaked a row nor evaluated
    // its DEFAULT twice before its rollback discarded the parent artifact.
    engine.submit_transaction(45_000, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            45_000,
            parsed("INSERT INTO typed_sequence_owner (value) VALUES (30)"),
        )
        .unwrap();
    engine
        .submit_transaction(45_000, parsed("ROLLBACK"))
        .unwrap();
    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 1, "{probe:?}");

    let select =
        match parse_command("SELECT id, value FROM typed_sequence_owner ORDER BY id").unwrap() {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        };
    let live = engine.execute_relational_select(&select).unwrap();
    assert!(matches!(live.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        live.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(10)],
            vec![SqlValue::Int4(2), SqlValue::Int4(20)],
        ]
    );

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    let replayed = recovered.execute_relational_select(&select).unwrap();
    assert!(matches!(replayed.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(replayed.rows, live.rows);
    let sequence = recovered
        .relational_catalog_sequence("typed_sequence_spine")
        .unwrap();
    assert_eq!((sequence.last_value, sequence.is_called), (3, true));
}

/// Once the typed terminal is durable, a failed canonical publication is fail-stop: recovery
/// must replay both its materialized rows and its transaction-private sequence effects from the
/// same codec-5 authority.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
fn typed_sequence_defaults_post_durable_failure_is_recovery_owned() {
    let runtime = CudaDriverRuntime::probe()
        .expect("the typed sequence post-durable proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(
        runtime.driver_available && runtime.device_count > 0,
        "the typed sequence post-durable proof requires an NVIDIA device"
    );

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(46_201, parsed("CREATE SEQUENCE typed_sequence_spine"))
        .unwrap();
    engine
        .submit_transaction(
            46_202,
            parsed(
                "CREATE TABLE typed_sequence_owner \
                 (id int4 DEFAULT nextval('typed_sequence_spine'::regclass), value int4)",
            ),
        )
        .unwrap();

    let probe_before = engine.insert_probe_snapshot();
    engine.submit_transaction(46_203, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            46_203,
            parsed("ALTER SEQUENCE typed_sequence_spine RESTART WITH 50"),
        )
        .unwrap();
    engine
        .submit_transaction(
            46_203,
            parsed("INSERT INTO typed_sequence_owner (value) VALUES (50), (60)"),
        )
        .unwrap();
    engine.fail_next_transaction_post_durable_apply();

    let error = engine
        .submit_transaction(46_203, parsed("COMMIT"))
        .unwrap_err();
    assert!(error.is_indeterminate(), "{error}");
    assert!(engine.is_commit_path_poisoned());
    assert!(engine.transaction_snapshot_handle(46_203).is_some());
    assert!(engine
        .submit_transaction(46_203, parsed("ROLLBACK"))
        .is_err());
    let probe = engine.insert_probe_snapshot().delta_since(probe_before);
    assert_eq!(probe.successful_insert_statements, 0, "{probe:?}");

    let records = engine.durable_wal_records();
    let envelope = gpu_db_wal::decode_canonical_record_payload(&records.last().unwrap().payload)
        .expect("decode durable private-sequence transaction")
        .expect("private-sequence transaction uses the canonical envelope");
    assert!(
        Engine::canonical_envelope_is_codec5(&envelope),
        "private sequence effects must not select the resolved transaction authority"
    );
    let fragments = envelope
        .fragments
        .iter()
        .map(|fragment| gpu_db_wal::CanonicalFragmentRef {
            kind: fragment.kind,
            body: &fragment.body,
        })
        .collect::<Vec<_>>();
    let replay = crate::typed_insert_aggregate::decode_closed_semantics_v2_replay(
        &envelope.header,
        &envelope.outcome,
        &fragments,
    )
    .expect("strictly close private-sequence codec-5 transaction")
    .expect("private-sequence transaction selects semantics-v2 replay");
    assert_eq!(replay.metadata().stable_transaction_id, 46_203);
    assert_eq!(replay.metadata().affected_rows, 2);

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    let select =
        match parse_command("SELECT id, value FROM typed_sequence_owner ORDER BY id").unwrap() {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        };
    let replayed = recovered.execute_relational_select(&select).unwrap();
    assert!(matches!(replayed.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        replayed.rows,
        vec![
            vec![SqlValue::Int4(50), SqlValue::Int4(50)],
            vec![SqlValue::Int4(51), SqlValue::Int4(60)],
        ]
    );
    let sequence = recovered
        .relational_catalog_sequence("typed_sequence_spine")
        .unwrap();
    assert_eq!((sequence.last_value, sequence.is_called), (51, true));
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
            ExecuteError::Engine(EngineError::UniqueViolation(_)) | ExecuteError::Serialization(_)
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
        .expect_err("the transient device relation must reject an in-statement tuple duplicate");
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
    let snapshot = engine.transaction_snapshot_handle(35).unwrap();
    let table = snapshot
        .transaction_catalog()
        .relational_catalog
        .get("credited_ddl")
        .cloned()
        .unwrap();
    let private_payload = {
        let shards = snapshot.transaction_shards();
        let memory = shards["credited_ddl"]
            .iter()
            .find(|shard| shard.row_count != 0)
            .and_then(|shard| shard.device_memory.as_ref())
            .expect("private transaction shard owns its payload");
        Arc::downgrade(memory)
    };
    drop(snapshot);
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
        assert!(
            private_payload.upgrade().is_none(),
            "live and authority witnesses must both release the private allocation before canonical apply"
        );
        durable_tx.send(()).unwrap();
        release_rx.recv().unwrap();
    });
    let committing = Arc::clone(&engine);
    let commit = std::thread::spawn(move || committing.submit_transaction(35, parsed("COMMIT")));
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

    // The private dense payload is publication credit. Canonical rollover additionally
    // retains one u64 created-by slot and one u64 stable row-id slot for this one-row text
    // shard; both allocations are sized before WAL.
    let exact_peak = private_peak + 16;
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
        .filter(|((table, ..), entry)| table == "final_compound" && entry.device_index.is_some())
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
    let commit = std::thread::spawn(move || committing.submit_transaction(355, parsed("COMMIT")));
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
            parsed(
                "ALTER SEQUENCE private_serial_id_seq \
                 RENAME TO private_serial_id_seq_final",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            365,
            parsed("INSERT INTO private_serial (value) VALUES (9), (10)"),
        )
        .unwrap();
    engine.submit_transaction(365, parsed("COMMIT")).unwrap();
    let select = match parse_command("SELECT id, value FROM private_serial ORDER BY id").unwrap() {
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
        .get("private_serial_id_seq_final")
        .cloned()
        .unwrap();
    assert_eq!(
        (live_sequence.last_value, live_sequence.is_called),
        (2, true)
    );
    let records = engine.durable_wal_records();
    let envelope = gpu_db_wal::decode_canonical_record_payload(&records[0].payload)
        .expect("decode the private serial transaction envelope")
        .expect("private serial transaction uses the canonical envelope");
    assert!(
        Engine::canonical_envelope_is_codec5(&envelope),
        "implicit serial CREATE TABLE + RENAME + INSERT must not select resolved binary WAL"
    );
    let fragments = envelope
        .fragments
        .iter()
        .map(|fragment| gpu_db_wal::CanonicalFragmentRef {
            kind: fragment.kind,
            body: &fragment.body,
        })
        .collect::<Vec<_>>();
    let replay = crate::typed_insert_aggregate::decode_closed_semantics_v2_replay(
        &envelope.header,
        &envelope.outcome,
        &fragments,
    )
    .expect("strictly close the implicit serial codec-5 transaction")
    .expect("implicit serial transaction selects semantics-v2 replay");
    assert_eq!(replay.metadata().stable_transaction_id, 365);
    assert_eq!(replay.metadata().affected_rows, 2);

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert_eq!(
        recovered.execute_relational_select(&select).unwrap().rows,
        live_rows
    );
    let recovered_sequence = recovered
        .catalog_snapshot()
        .relational_sequences
        .get("private_serial_id_seq_final")
        .cloned()
        .unwrap();
    assert_eq!(recovered_sequence, live_sequence);
    assert!(!recovered
        .catalog_snapshot()
        .relational_sequences
        .contains_key("private_serial_id_seq"));
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

    let select = match parse_command("SELECT id, value FROM rebased_private ORDER BY id").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    let live = engine.execute_relational_select(&select).unwrap().rows;
    assert_eq!(live.row(0), &[SqlValue::Int4(1), SqlValue::Null]);
    assert_eq!(live.row(1), &[SqlValue::Int4(2), SqlValue::Int4(9)]);
    #[cfg(feature = "probe-timing")]
    assert_eq!(
        engine.relational_named_index_covered_rows("rebased_private"),
        Some(2),
        "the transaction-created primary key is physically published with its first rowset"
    );

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    recovered.set_shard_residency_enabled(true);
    recovered.set_auto_admit_on_commit(true);
    let replayed = recovered.execute_relational_select(&select).unwrap().rows;
    assert_eq!(replayed, live);
    #[cfg(feature = "probe-timing")]
    assert_eq!(
        recovered.relational_named_index_covered_rows("rebased_private"),
        Some(2),
        "fresh replay republishes the transaction-created primary-key coverage"
    );
}

/// The private DDL/DML composition used by the mandatory pgwire route must retain an inline
/// primary-key descriptor in every statement's sealed S2 record.  This is deliberately checked
/// before COMMIT so a later catalog-composition defect cannot be misdiagnosed as a statement-time
/// semantic omission.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn private_created_indexed_sequence_table_seals_primary_key_into_every_s2_statement() {
    const TXN: TxnId = 49_010;
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(49_000, parsed("CREATE DOMAIN private_s2_tally AS int4"))
        .unwrap();
    engine
        .submit_transaction(
            49_001,
            parsed("CREATE TABLE private_s2_parent (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(49_002, parsed("INSERT INTO private_s2_parent VALUES (7)"))
        .unwrap();

    engine.submit_transaction(TXN, parsed("BEGIN")).unwrap();
    for statement in [
        "CREATE SEQUENCE private_s2_sequence",
        "CREATE TABLE private_s2_owner \
         (id int4 DEFAULT nextval('private_s2_sequence'::regclass), \
          row_key int4 PRIMARY KEY, parent_id int4, tally private_s2_tally, note text, \
          CONSTRAINT private_s2_tally_positive CHECK (tally > 0))",
        "ALTER TABLE ONLY private_s2_owner \
         ADD CONSTRAINT private_s2_owner_parent \
         FOREIGN KEY (parent_id) REFERENCES private_s2_parent(id)",
        "INSERT INTO private_s2_owner (row_key, parent_id, tally, note) \
         VALUES (1, 7, 11, 'before-rename')",
        "ALTER SEQUENCE private_s2_sequence RENAME TO private_s2_sequence_final",
        "INSERT INTO private_s2_owner (row_key, parent_id, tally, note) \
         VALUES (2, 7, 22, 'after-rename')",
    ] {
        engine.submit_transaction(TXN, parsed(statement)).unwrap();
    }

    let snapshot = engine.transaction_snapshot_handle(TXN).unwrap();
    let catalog = snapshot.transaction_catalog();
    assert_eq!(
        catalog.relational_catalog["private_s2_owner"].indexes.len(),
        1
    );
    let delta = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let typed = delta
        .operations
        .iter()
        .filter_map(|operation| match operation {
            TransactionOperation::TypedInsert(staged) => Some(staged),
            TransactionOperation::Catalog(_)
            | TransactionOperation::Row(_)
            | TransactionOperation::TableReset(_) => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(typed.len(), 2);
    for staged in typed {
        let record = crate::typed_insert_batch::decode_canonical_typed_insert_record(
            staged.codec5_sources.record(),
        )
        .expect("private typed S2 remains canonical");
        assert_eq!(record.indexes().count(), 1);
    }
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
    let select = match parse_command("SELECT value FROM durable_private_ddl WHERE id = 1").unwrap()
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
fn transaction_truncate_restart_identity_without_owned_sequence_is_private() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(94, parsed("CREATE TABLE restart_target (id int4)"))
        .unwrap();
    engine.submit_transaction(95, parsed("BEGIN")).unwrap();
    let wal_before = engine.durable_wal_records().len();

    engine
        .submit_transaction(95, parsed("TRUNCATE restart_target RESTART IDENTITY"))
        .unwrap();
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(!engine
        .transaction_snapshot_handle(95)
        .unwrap()
        .transaction_delta_is_empty());
    engine.submit_transaction(95, parsed("ROLLBACK")).unwrap();
    assert!(engine.relational_catalog_table("restart_target").is_some());
    assert_eq!(engine.durable_wal_records().len(), wal_before);
}
