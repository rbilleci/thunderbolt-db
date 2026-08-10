use super::*;

#[test]
fn product_001_copy_compatibility_crash_prefixes_retry_once() {
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
    let copy = gpu_db_sql::parse_copy_from_stdin(
        "COPY product_codec5_copy_prefix (id, note) FROM STDIN WITH (FORMAT csv)",
    )
    .expect("parse COPY compatibility fixture");
    let rows = vec![
        vec![SqlValue::Int4(1), SqlValue::Text("first".to_string())],
        vec![SqlValue::Int4(2), SqlValue::Text("second".to_string())],
    ];

    for (ordinal, (label, boundary)) in boundaries.into_iter().enumerate() {
        let fixture = DurableFixture::new(&format!("copy-{label}"));
        let engine = Engine::with_durable_wal_segment(&fixture.wal);
        let create_txn = 71_125 + u64::try_from(ordinal).unwrap() * 10;
        let copy_txn = create_txn + 1;
        engine
            .execute_text(
                create_txn,
                "CREATE TABLE product_codec5_copy_prefix (id INT PRIMARY KEY, note TEXT)",
            )
            .expect("durable COPY crash-prefix fixture DDL");
        let durable_before = engine.durable_wal_records();
        let visible_before = engine.visible_up_to();
        let flushed_before = engine.wal_flushed_count();

        engine.fail_next_transaction_terminal_pre_durable_at(boundary);
        let failure = engine.execute_relational_copy_rows(copy_txn, &copy, rows.clone());
        assert!(
            matches!(
                failure,
                Err(ExecuteError::Engine(EngineError::Durability(_)))
            ),
            "COPY {label} must fail before acknowledgement: {failure:?}"
        );
        assert_eq!(engine.visible_up_to(), visible_before, "COPY {label}");
        assert_eq!(engine.wal_flushed_count(), flushed_before, "COPY {label}");
        assert_eq!(engine.durable_wal_records(), durable_before, "COPY {label}");
        assert!(durable_ids(&engine, "product_codec5_copy_prefix").is_empty());

        // COPY has no durable authority of its own. Its failed prefix must therefore recover as
        // the same empty prefix as a textual INSERT, and the same request identity must retry
        // through the ordinary typed transaction overlay exactly once.
        drop(engine);
        let recovered =
            Engine::open_durable_wal_segment_auto(&fixture.wal).expect("recover COPY crash prefix");
        assert_eq!(
            recovered.durable_wal_records(),
            durable_before,
            "COPY {label}"
        );
        assert!(durable_ids(&recovered, "product_codec5_copy_prefix").is_empty());
        assert_eq!(
            recovered
                .execute_relational_copy_rows(copy_txn, &copy, rows.clone())
                .expect("the unacknowledged COPY request must retry"),
            2
        );
        drop(recovered);
        let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("recover acknowledged COPY retry");
        assert_eq!(
            durable_ids(&reopened, "product_codec5_copy_prefix"),
            vec![1, 2]
        );
        let retry_record = reopened
            .durable_wal_records()
            .into_iter()
            .find(|record| record.txn_id == copy_txn)
            .expect("the acknowledged COPY retry remains in canonical WAL");
        let retry_envelope = gpu_db_wal::decode_canonical_record_payload(&retry_record.payload)
            .expect("decode COPY crash-prefix retry record")
            .expect("COPY crash-prefix retry uses a canonical envelope");
        assert!(
            Engine::canonical_envelope_is_codec5(&retry_envelope),
            "COPY {label} retry must not enter a displaced INSERT authority"
        );
    }
}

#[test]
fn product_001_copy_compatibility_post_durable_recovery_owns_once() {
    let fixture = DurableFixture::new("copy-post-durable");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    let copy = gpu_db_sql::parse_copy_from_stdin(
        "COPY product_codec5_copy_indeterminate (id, note) FROM STDIN WITH (FORMAT csv)",
    )
    .expect("parse COPY compatibility fixture");
    let rows = vec![
        vec![SqlValue::Int4(1), SqlValue::Text("first".to_string())],
        vec![SqlValue::Int4(2), SqlValue::Text("second".to_string())],
    ];
    engine
        .execute_text(
            71_160,
            "CREATE TABLE product_codec5_copy_indeterminate (id INT PRIMARY KEY, note TEXT)",
        )
        .expect("durable COPY post-durable fixture DDL");
    let records_before = engine.durable_wal_records().len();

    engine.fail_next_transaction_post_durable_apply();
    let failure = engine
        .execute_relational_copy_rows(71_161, &copy, rows.clone())
        .expect_err("post-durable COPY apply fault must be indeterminate");
    assert!(
        failure.is_indeterminate(),
        "COPY must report the generic terminal's indeterminate outcome: {failure}"
    );
    assert!(engine.is_commit_path_poisoned());
    assert_eq!(engine.durable_wal_records().len(), records_before + 1);

    // The status/replay authority is the same codec-5 record written for every INSERT ingress;
    // a fresh engine owns the one successful apply and a second fresh reopen cannot replay it.
    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("recover indeterminate COPY transaction");
    assert_eq!(
        durable_ids(&recovered, "product_codec5_copy_indeterminate"),
        vec![1, 2]
    );
    let record = recovered
        .durable_wal_records()
        .into_iter()
        .find(|record| record.txn_id == 71_161)
        .expect("durable COPY transaction remains present");
    let envelope = gpu_db_wal::decode_canonical_record_payload(&record.payload)
        .expect("decode durable COPY record")
        .expect("durable COPY uses a canonical envelope");
    assert!(
        Engine::canonical_envelope_is_codec5(&envelope),
        "post-durable COPY recovery must retain the one codec-5 INSERT authority"
    );
    assert_eq!(
        recovered
            .execute_relational_copy_rows(71_161, &copy, rows)
            .expect("matching COPY retry returns the recovered durable outcome"),
        2
    );
    let records_before_second_reopen = recovered.durable_wal_records().len();
    drop(recovered);
    let reopened =
        Engine::open_durable_wal_segment_auto(&fixture.wal).expect("repeat COPY recovery is exact");
    assert_eq!(
        reopened.durable_wal_records().len(),
        records_before_second_reopen,
        "repeat recovery must not append or duplicate the COPY transaction"
    );
    assert_eq!(
        durable_ids(&reopened, "product_codec5_copy_indeterminate"),
        vec![1, 2]
    );
}

fn stage_private_sequence_copy(engine: &Engine, txn_id: u64, table: &str, sequence: &str) {
    let renamed_sequence = format!("{sequence}_final");
    for sql in [
        "BEGIN".to_string(),
        format!("CREATE SEQUENCE {sequence}"),
        format!(
            "CREATE TABLE {table} \
             (id INT DEFAULT nextval('{sequence}'::regclass), row_key INT PRIMARY KEY, note TEXT)"
        ),
        format!("INSERT INTO {table} (row_key, note) VALUES (1, 'text-insert')"),
    ] {
        engine
            .submit_transaction(txn_id, parsed(&sql))
            .expect("stage the private sequence/table typed-INSERT prefix");
    }
    let copy = gpu_db_sql::parse_copy_from_stdin(&format!(
        "COPY {table} (row_key, note) FROM STDIN WITH (FORMAT csv)"
    ))
    .expect("parse transaction-private COPY");
    let (_columns, target) = engine
        .relational_copy_target_in_transaction(txn_id, table)
        .expect("resolve the COPY target from the private catalog overlay");
    engine
        .submit_transaction(
            txn_id,
            crate::CopyMutationRequest::new(
                copy,
                vec![vec![
                    SqlValue::Int4(2),
                    SqlValue::Text("copy-row".to_string()),
                ]],
                target,
            ),
        )
        .expect("stage COPY in the same private typed overlay");
    engine
        .submit_transaction(
            txn_id,
            parsed(&format!(
                "ALTER SEQUENCE {sequence} RENAME TO {renamed_sequence}"
            )),
        )
        .expect("stage the stable-OID sequence rename after private COPY");
}

#[test]
fn product_001_private_sequence_copy_crash_prefixes_retry_once() {
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
        let fixture = DurableFixture::new(&format!("private-copy-{label}"));
        let engine = Engine::with_durable_wal_segment(&fixture.wal);
        let txn_id = 71_175 + u64::try_from(ordinal).unwrap();
        let table = "product_private_sequence_copy_prefix";
        let sequence = "product_private_sequence_copy_prefix_sequence";
        let durable_before = engine.durable_wal_records();
        let visible_before = engine.visible_up_to();
        let oid_before = engine.catalog_snapshot().relational_next_oid;
        stage_private_sequence_copy(&engine, txn_id, table, sequence);
        engine.fail_next_transaction_terminal_pre_durable_at(boundary);
        let failure = engine.submit_transaction(txn_id, parsed("COMMIT"));
        assert!(
            matches!(
                failure,
                Err(ExecuteError::Engine(EngineError::Durability(_)))
            ),
            "private COPY {label} must fail before acknowledgement: {failure:?}"
        );
        assert_eq!(
            engine.durable_wal_records(),
            durable_before,
            "private COPY {label}"
        );
        assert_eq!(
            engine.visible_up_to(),
            visible_before,
            "private COPY {label}"
        );
        assert_eq!(engine.catalog_snapshot().relational_next_oid, oid_before);
        assert!(engine.relational_catalog_table(table).is_none());
        assert!(engine.relational_catalog_sequence(sequence).is_none());
        assert!(engine
            .relational_catalog_sequence(&format!("{sequence}_final"))
            .is_none());

        drop(engine);
        let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("recover the private COPY crash prefix");
        assert_eq!(
            recovered.durable_wal_records(),
            durable_before,
            "private COPY {label}"
        );
        assert!(recovered.relational_catalog_table(table).is_none());
        assert!(recovered.relational_catalog_sequence(sequence).is_none());

        stage_private_sequence_copy(&recovered, txn_id, table, sequence);
        recovered
            .submit_transaction(txn_id, parsed("COMMIT"))
            .expect("the unacknowledged private COPY transaction retries through codec-5");
        assert_eq!(durable_ids(&recovered, table), vec![1, 2]);
        let final_sequence = recovered
            .relational_catalog_sequence(&format!("{sequence}_final"))
            .expect("retry publishes the renamed private sequence exactly once");
        assert_eq!(
            (final_sequence.last_value, final_sequence.is_called),
            (2, true)
        );
        let record = recovered
            .durable_wal_records()
            .into_iter()
            .find(|record| record.txn_id == txn_id)
            .expect("the acknowledged private COPY retry remains in canonical WAL");
        assert!(codec5_replay_metadata(&record).initial_table_absent);

        drop(recovered);
        let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("fresh reopen recovers the acknowledged private COPY retry");
        assert_eq!(durable_ids(&reopened, table), vec![1, 2]);
        let final_sequence = reopened
            .relational_catalog_sequence(&format!("{sequence}_final"))
            .expect("fresh replay retains the renamed private sequence");
        assert_eq!(
            (final_sequence.last_value, final_sequence.is_called),
            (2, true)
        );
    }
}

#[test]
fn product_001_private_sequence_copy_post_durable_recovery_owns_once() {
    let fixture = DurableFixture::new("private-copy-post-durable");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    let txn_id = 71_190;
    let table = "product_private_sequence_copy_indeterminate";
    let sequence = "product_private_sequence_copy_indeterminate_sequence";
    let records_before = engine.durable_wal_records().len();
    stage_private_sequence_copy(&engine, txn_id, table, sequence);

    engine.fail_next_transaction_post_durable_apply();
    let failure = engine
        .submit_transaction(txn_id, parsed("COMMIT"))
        .expect_err("post-durable private COPY apply fault must be indeterminate");
    assert!(
        failure.is_indeterminate(),
        "private COPY must report the generic terminal's indeterminate outcome: {failure}"
    );
    assert!(engine.is_commit_path_poisoned());
    assert_eq!(engine.durable_wal_records().len(), records_before + 1);

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("recovery owns the durable private COPY transaction");
    assert_eq!(durable_ids(&recovered, table), vec![1, 2]);
    let final_sequence = recovered
        .relational_catalog_sequence(&format!("{sequence}_final"))
        .expect("recovery publishes the renamed private sequence");
    assert_eq!(
        (final_sequence.last_value, final_sequence.is_called),
        (2, true)
    );
    let record = recovered
        .durable_wal_records()
        .into_iter()
        .find(|record| record.txn_id == txn_id)
        .expect("durable private COPY transaction remains present");
    assert!(codec5_replay_metadata(&record).initial_table_absent);

    let records_before_second_reopen = recovered.durable_wal_records().len();
    drop(recovered);
    let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("repeat private COPY recovery is exact");
    assert_eq!(
        reopened.durable_wal_records().len(),
        records_before_second_reopen,
        "repeat recovery must not append or duplicate the private COPY transaction"
    );
    assert_eq!(durable_ids(&reopened, table), vec![1, 2]);
    let final_sequence = reopened
        .relational_catalog_sequence(&format!("{sequence}_final"))
        .expect("repeat recovery retains the renamed private sequence");
    assert_eq!(
        (final_sequence.last_value, final_sequence.is_called),
        (2, true)
    );
}
