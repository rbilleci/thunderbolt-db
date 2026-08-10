use super::*;

fn mixed_rows(engine: &Engine, table: &str) -> Vec<Vec<SqlValue>> {
    let Command::Select(select) =
        parse_command(&format!("SELECT id, value FROM {table} ORDER BY id"))
            .expect("parse mixed-DML result query")
    else {
        panic!("expected SELECT")
    };
    engine
        .execute_relational_select(&select)
        .expect("execute mixed-DML result query")
        .rows
        .iter()
        .map(|row| row.to_vec())
        .collect()
}

fn stage_mixed_insert_update_delete(engine: &Engine, txn_id: u64, table: &str) {
    for sql in [
        "BEGIN".to_string(),
        format!("INSERT INTO {table} VALUES (1, 10)"),
        format!("UPDATE {table} SET value = 11 WHERE id = 1"),
        format!("INSERT INTO {table} VALUES (2, 20)"),
        format!("DELETE FROM {table} WHERE id = 2"),
    ] {
        engine
            .submit_transaction(txn_id, parsed(&sql))
            .expect("stage mixed INSERT/UPDATE/DELETE transaction");
    }
}

#[test]
fn product_001_mixed_insert_update_delete_crash_prefixes_retry_once() {
    use crate::engine_transaction_delta::TransactionTerminalPreDurableFault;

    let boundaries = [
        (
            "before-replication-proposal",
            TransactionTerminalPreDurableFault::ReplicationProposal,
        ),
        (
            "before-canonical-wal-append",
            TransactionTerminalPreDurableFault::CanonicalWalAppend,
        ),
        (
            "before-wal-flush",
            TransactionTerminalPreDurableFault::WalFlush,
        ),
    ];

    for (ordinal, (label, boundary)) in boundaries.into_iter().enumerate() {
        let fixture = DurableFixture::new(&format!("mixed-dml-{label}"));
        let engine = Engine::with_durable_wal_segment(&fixture.wal);
        let create_txn = 71_500 + u64::try_from(ordinal).unwrap() * 10;
        let txn_id = create_txn + 1;
        let table = "product_codec5_mixed_dml_prefix";
        engine
            .execute_text(
                create_txn,
                &format!("CREATE TABLE {table} (id INT PRIMARY KEY, value INT)"),
            )
            .expect("durable mixed-DML crash-prefix fixture DDL");
        let durable_before = engine.durable_wal_records();
        let visible_before = engine.visible_up_to();
        let flushed_before = engine.wal_flushed_count();

        stage_mixed_insert_update_delete(&engine, txn_id, table);
        engine.fail_next_transaction_terminal_pre_durable_at(boundary);
        let failure = engine.submit_transaction(txn_id, parsed("COMMIT"));
        assert!(
            matches!(
                failure,
                Err(ExecuteError::Engine(EngineError::Durability(_)))
            ),
            "mixed INSERT/UPDATE/DELETE {label} must fail before acknowledgement: {failure:?}"
        );
        assert_eq!(engine.visible_up_to(), visible_before, "mixed DML {label}");
        assert_eq!(
            engine.wal_flushed_count(),
            flushed_before,
            "mixed DML {label}"
        );
        assert_eq!(
            engine.durable_wal_records(),
            durable_before,
            "mixed DML {label}"
        );
        assert!(durable_ids(&engine, table).is_empty());
        assert!(mixed_rows(&engine, table).is_empty());

        // The pre-durable terminal has no publication or replay authority.  A new engine must
        // therefore see the same empty prefix, after which the same request identity takes the
        // sole typed overlay -> DeviceInsertPlan -> codec-5 terminal exactly once.
        drop(engine);
        let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("recover mixed-DML crash prefix");
        assert_eq!(
            recovered.durable_wal_records(),
            durable_before,
            "mixed DML {label}"
        );
        assert!(mixed_rows(&recovered, table).is_empty());
        stage_mixed_insert_update_delete(&recovered, txn_id, table);
        recovered
            .submit_transaction(txn_id, parsed("COMMIT"))
            .expect("the unacknowledged mixed DML request must retry");
        let retry_record = recovered
            .durable_wal_records()
            .into_iter()
            .find(|record| record.txn_id == txn_id)
            .expect("the acknowledged mixed DML retry remains in canonical WAL");
        // S6 authenticates the original typed INSERT input count (two), while S7 separately
        // authenticates the one surviving final-image transition below.
        assert_eq!(codec5_replay_metadata(&retry_record).affected_rows, 2);

        drop(recovered);
        let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("recover acknowledged mixed-DML retry");
        assert_eq!(
            mixed_rows(&reopened, table),
            vec![vec![SqlValue::Int4(1), SqlValue::Int4(11)]],
            "mixed DML {label} must apply its UPDATE but not its deleted INSERT"
        );
        assert_eq!(durable_ids(&reopened, table), vec![1]);
    }
}

#[test]
fn product_001_mixed_insert_update_delete_post_durable_recovery_owns_once() {
    let fixture = DurableFixture::new("mixed-dml-post-durable");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    let txn_id = 71_531;
    let table = "product_codec5_mixed_dml_indeterminate";
    engine
        .execute_text(
            txn_id - 1,
            &format!("CREATE TABLE {table} (id INT PRIMARY KEY, value INT)"),
        )
        .expect("durable mixed-DML post-durable fixture DDL");
    let records_before = engine.durable_wal_records().len();
    stage_mixed_insert_update_delete(&engine, txn_id, table);

    engine.fail_next_transaction_post_durable_apply();
    let failure = engine
        .submit_transaction(txn_id, parsed("COMMIT"))
        .expect_err("post-durable mixed-DML apply fault must be indeterminate");
    assert!(
        failure.is_indeterminate(),
        "mixed DML must report the generic terminal's indeterminate outcome: {failure}"
    );
    assert!(engine.is_commit_path_poisoned());
    assert_eq!(engine.durable_wal_records().len(), records_before + 1);

    // The one durable codec-5 status record is the replay authority.  Fresh recovery performs
    // the mixed transaction once, and a second fresh reopen proves it cannot duplicate either
    // the updated survivor or the deleted second INSERT.
    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("recover indeterminate mixed-DML transaction");
    assert_eq!(recovered.durable_wal_records().len(), records_before + 1);
    assert_eq!(
        mixed_rows(&recovered, table),
        vec![vec![SqlValue::Int4(1), SqlValue::Int4(11)]]
    );
    let record = recovered
        .durable_wal_records()
        .into_iter()
        .find(|record| record.txn_id == txn_id)
        .expect("durable mixed-DML transaction remains present");
    assert_eq!(codec5_replay_metadata(&record).affected_rows, 2);

    let records_before_second_reopen = recovered.durable_wal_records().len();
    drop(recovered);
    let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("repeat mixed-DML recovery is exact");
    assert_eq!(
        reopened.durable_wal_records().len(),
        records_before_second_reopen,
        "repeat recovery must not append or duplicate the mixed transaction"
    );
    assert_eq!(
        mixed_rows(&reopened, table),
        vec![vec![SqlValue::Int4(1), SqlValue::Int4(11)]]
    );
    assert_eq!(durable_ids(&reopened, table), vec![1]);
}
