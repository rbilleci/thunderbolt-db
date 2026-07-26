use super::{
    transaction_private_create_is_dml_target, validate_transaction_characteristics, Engine,
    MutationRequest, PredeclaredOperationResult, PredeclaredTransaction,
    TransactionAdmissionResult, TransactionCharacteristics, TransactionClass, TransactionResources,
};
use gpu_db_sql::{ParsedCommand, SqlValue};
use std::sync::{mpsc, Arc, Barrier};
use std::time::Duration;

fn parsed(sql: &str) -> ParsedCommand {
    ParsedCommand::parse(sql).unwrap()
}

fn resources(operations: u32, mutations: u32) -> TransactionResources {
    TransactionResources {
        operations,
        mutations,
        post_image_and_wal_bytes: u64::MAX,
        maintained_index_fanout: u32::MAX,
        touched_tables: u32::MAX,
        cold_accesses: u32::MAX,
        result_bytes: u64::MAX,
    }
}

#[test]
fn only_engine_prepared_routes_derive_fast_classes_and_general_has_no_t32_cap() {
    assert_eq!(
        TransactionClass::derive(TransactionResources::W1_MAX, false),
        TransactionClass::General
    );
    assert_eq!(
        TransactionClass::derive(TransactionResources::W1_MAX, true),
        TransactionClass::W1
    );
    assert_eq!(
        TransactionClass::derive(TransactionResources::T8_MAX, true),
        TransactionClass::T8
    );
    assert_eq!(
        TransactionClass::derive(TransactionResources::T32_MAX, true),
        TransactionClass::T32
    );
    assert_eq!(
        TransactionClass::derive(resources(33, 17), true),
        TransactionClass::General
    );
    assert_eq!(
        TransactionClass::derive(resources(10_000, 5_000), true),
        TransactionClass::General
    );
}

#[test]
fn resource_envelope_checks_every_manifest_dimension() {
    let declared = TransactionResources {
        operations: 1,
        mutations: 1,
        post_image_and_wal_bytes: 1,
        maintained_index_fanout: 1,
        touched_tables: 1,
        cold_accesses: 1,
        result_bytes: 1,
    };
    let cases = [
        (
            TransactionResources {
                operations: 2,
                ..declared
            },
            "operations",
        ),
        (
            TransactionResources {
                mutations: 2,
                ..declared
            },
            "mutations",
        ),
        (
            TransactionResources {
                post_image_and_wal_bytes: 2,
                ..declared
            },
            "post-image plus logical-WAL bytes",
        ),
        (
            TransactionResources {
                maintained_index_fanout: 2,
                ..declared
            },
            "maintained-index fanout",
        ),
        (
            TransactionResources {
                touched_tables: 2,
                ..declared
            },
            "touched tables",
        ),
        (
            TransactionResources {
                cold_accesses: 2,
                ..declared
            },
            "cold accesses",
        ),
        (
            TransactionResources {
                result_bytes: 2,
                ..declared
            },
            "result bytes",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual.first_excess(declared), Some(expected));
    }
}

#[test]
fn invalid_predeclared_shape_fails_before_transaction_sequence_or_wal_effects() {
    let engine = Engine::new_local();
    let visible_before = engine.committed_seq();
    let wal_before = engine.durable_wal_records().len();
    let row_id_before = engine.read_state.mvcc.current_row_id();
    let transaction = PredeclaredTransaction::new(
        vec![parsed("CREATE TABLE forbidden (id INT)")],
        resources(1, 0),
        TransactionCharacteristics::REPEATABLE_READ_WRITE,
    );

    let error = engine.submit_transaction(71, transaction).unwrap_err();

    assert!(matches!(error, crate::ExecuteError::Unsupported(_)));
    assert_eq!(engine.committed_seq(), visible_before);
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
    assert!(engine.transaction_snapshot_handle(71).is_none());
}

#[test]
fn transaction_characteristics_support_rc_and_reject_unsupported_modes_before_begin() {
    for isolation in [
        super::TransactionIsolation::ReadCommitted,
        super::TransactionIsolation::ReadUncommitted,
    ] {
        let normalized = validate_transaction_characteristics(super::TransactionCharacteristics {
            isolation,
            access: super::TransactionAccessMode::ReadWrite,
            deferrable: false,
        })
        .unwrap();
        assert_eq!(
            normalized.isolation,
            super::TransactionIsolation::ReadCommitted
        );
    }

    for (txn_id, characteristics, expected) in [
        (
            72,
            super::TransactionCharacteristics {
                isolation: super::TransactionIsolation::Serializable,
                access: super::TransactionAccessMode::ReadWrite,
                deferrable: false,
            },
            "SERIALIZABLE",
        ),
        (
            73,
            super::TransactionCharacteristics {
                isolation: super::TransactionIsolation::ReadCommitted,
                access: super::TransactionAccessMode::ReadWrite,
                deferrable: true,
            },
            "DEFERRABLE",
        ),
    ] {
        let engine = Engine::new_local();
        let visible_before = engine.committed_seq();
        let wal_before = engine.durable_wal_records().len();
        let transaction = PredeclaredTransaction::new(
            vec![parsed("SELECT id FROM t WHERE id = 1")],
            resources(1, 0),
            characteristics,
        );
        let error = engine.submit_transaction(txn_id, transaction).unwrap_err();
        assert!(matches!(error, crate::ExecuteError::Unsupported(_)));
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(engine.committed_seq(), visible_before);
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert!(engine.transaction_snapshot_handle(txn_id).is_none());
    }
}

#[test]
fn dependency_closure_is_preclaimed_before_begin() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(1, parsed("CREATE TABLE p (id INT PRIMARY KEY)"))
        .unwrap();
    engine
        .submit_transaction(2, parsed("CREATE TABLE c (id INT PRIMARY KEY, pid INT)"))
        .unwrap();
    engine
        .submit_transaction(
            3,
            parsed("ALTER TABLE ONLY c ADD CONSTRAINT c_fk FOREIGN KEY (pid) REFERENCES p(id)"),
        )
        .unwrap();
    let visible_before = engine.committed_seq();
    let wal_before = engine.durable_wal_records().len();
    let transaction = PredeclaredTransaction::new(
        vec![parsed("INSERT INTO c VALUES (1, 1)")],
        TransactionResources {
            operations: 1,
            mutations: 1,
            post_image_and_wal_bytes: u64::MAX,
            maintained_index_fanout: u32::MAX,
            touched_tables: 1,
            cold_accesses: u32::MAX,
            result_bytes: u64::MAX,
        },
        TransactionCharacteristics::REPEATABLE_READ_WRITE,
    );

    let error = engine.submit_transaction(4, transaction).unwrap_err();

    assert!(error.to_string().contains("touched tables"));
    assert_eq!(engine.committed_seq(), visible_before);
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(engine.transaction_snapshot_handle(4).is_none());
}

#[test]
fn parsed_mutation_admission_routes_generic_and_concurrent_dml() {
    let engine = Engine::new_local();
    assert!(matches!(
        engine
            .submit_transaction(
                1,
                MutationRequest::new(parsed("CREATE TABLE t (id INT, v INT)")),
            )
            .unwrap(),
        TransactionAdmissionResult::Command
    ));
    let TransactionAdmissionResult::Dml(result) = engine
        .submit_transaction(
            2,
            MutationRequest::new(parsed("INSERT INTO t VALUES (7, 9)")),
        )
        .unwrap()
    else {
        panic!("concurrent DML must retain its result through typed admission");
    };
    assert_eq!(result.rows_affected, 1);
    assert_eq!(engine.durable_wal_records().len(), 2);
}

#[test]
fn private_catalog_expected_proof_is_scoped_to_the_exact_created_table() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(10, parsed("CREATE TABLE published_target (id int4)"))
        .unwrap();
    engine.submit_transaction(11, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            11,
            parsed("CREATE TABLE private_target (id int4, value int4)"),
        )
        .unwrap();
    let snapshot = engine.transaction_snapshot_handle(11).unwrap();

    assert!(transaction_private_create_is_dml_target(
        &snapshot,
        parsed("INSERT INTO private_target VALUES (1, 2)").command()
    ));
    assert!(transaction_private_create_is_dml_target(
        &snapshot,
        parsed("UPDATE private_target SET value = 3 WHERE id = 1").command()
    ));
    assert!(transaction_private_create_is_dml_target(
        &snapshot,
        parsed("DELETE FROM private_target WHERE id = 1").command()
    ));
    assert!(!transaction_private_create_is_dml_target(
        &snapshot,
        parsed("INSERT INTO published_target VALUES (1)").command()
    ));
    assert!(!transaction_private_create_is_dml_target(
        &snapshot,
        parsed("SELECT id FROM private_target").command()
    ));

    engine.submit_transaction(11, parsed("ROLLBACK")).unwrap();
}

#[test]
fn read_only_command_is_rejected_without_sequence_or_wal_claim() {
    let engine = Engine::new_local();
    let visible_before = engine.committed_seq();
    let wal_before = engine.durable_wal_records().len();
    let error = engine
        .submit_transaction(1, MutationRequest::new(parsed("SELECT id FROM t")))
        .unwrap_err();
    assert!(error.to_string().contains("read-only commands"));
    assert_eq!(engine.committed_seq(), visible_before);
    assert_eq!(engine.durable_wal_records().len(), wal_before);
}

#[test]
fn compatibility_reads_are_unsequenced_and_do_not_enter_representation_repair() {
    let engine = Engine::new_local();
    engine.execute_text(1, "SET answer=forty-two").unwrap();
    let visible_before = engine.committed_seq();
    let wal_before = engine.durable_wal_records().len();
    let fallback_before = engine.metrics().snapshot().fallback_total;

    for sql in ["GET answer", "SELECT currval('seq')"] {
        engine
            .execute_parsed_compatibility_read(parsed(sql))
            .unwrap();
    }

    assert_eq!(engine.committed_seq(), visible_before);
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(
        engine.metrics().snapshot().fallback_total,
        fallback_before + 2
    );
}

#[test]
fn nonconcurrent_returning_rejects_before_sequence_or_wal_claim() {
    let engine = Engine::new_local();
    let visible_before = engine.committed_seq();
    let wal_before = engine.durable_wal_records().len();
    let error = engine
        .submit_transaction(
            1,
            MutationRequest::new(parsed("INSERT INTO missing VALUES (1) RETURNING id")),
        )
        .unwrap_err();
    assert!(matches!(error, crate::ExecuteError::Unsupported(_)));
    assert_eq!(engine.committed_seq(), visible_before);
    assert_eq!(engine.durable_wal_records().len(), wal_before);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn general_atomic_program_reads_its_writes_commits_once_and_recovers() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(
            1,
            parsed("CREATE TABLE t (id INT PRIMARY KEY, v INT UNIQUE)"),
        )
        .unwrap();
    let visible_before = engine.committed_seq();
    let wal_before = engine.durable_wal_records().len();
    let transaction = PredeclaredTransaction::new(
        [
            "INSERT INTO t VALUES (1, 10) RETURNING id",
            "SELECT id, v FROM t WHERE id = 1",
            "UPDATE t SET v = 11 WHERE id = 1 RETURNING v",
            "SELECT id, v FROM t WHERE id = 1",
            "INSERT INTO t VALUES (2, 20) RETURNING id",
            "SELECT id, v FROM t WHERE id = 2",
            "DELETE FROM t WHERE id = 2 RETURNING id",
            "SELECT id, v FROM t WHERE id = 2",
        ]
        .into_iter()
        .map(parsed)
        .collect(),
        TransactionResources::T8_MAX,
        TransactionCharacteristics::REPEATABLE_READ_WRITE,
    );

    let TransactionAdmissionResult::Predeclared(result) =
        engine.submit_transaction(2, transaction).unwrap()
    else {
        panic!("predeclared submission must retain its ordered results");
    };
    assert_eq!(result.class, TransactionClass::General);
    assert_eq!(result.actual_resources.operations, 8);
    assert_eq!(result.actual_resources.mutations, 4);
    let PredeclaredOperationResult::Read(first_read) = &result.operations[1] else {
        panic!("operation 2 must be the first read-own-write result");
    };
    assert_eq!(
        first_read.rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Int4(10)]]
    );
    let PredeclaredOperationResult::Read(updated_read) = &result.operations[3] else {
        panic!("operation 4 must be the updated read-own-write result");
    };
    assert_eq!(
        updated_read.rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Int4(11)]]
    );
    let PredeclaredOperationResult::Read(deleted_read) = &result.operations[7] else {
        panic!("operation 8 must read after delete");
    };
    assert!(deleted_read.rows.is_empty());
    assert_eq!(engine.committed_seq(), visible_before + 1);
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    let records = engine.durable_wal_records();
    let envelope = gpu_db_wal::decode_canonical_record_payload(
        &records.last().expect("transaction WAL").payload,
    )
    .unwrap()
    .expect("canonical transaction WAL");
    assert_eq!(
        envelope.header.isolation,
        gpu_db_wal::CanonicalIsolation::RepeatableRead
    );

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    let recovered_rows = recovered
        .execute_relational_select_text("SELECT id, v FROM t")
        .unwrap();
    assert_eq!(
        recovered_rows.rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Int4(11)]]
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn same_id_commit_and_dml_cannot_interleave_a_predeclared_prefix() {
    let engine = Arc::new(Engine::new_local());
    engine
        .submit_transaction(1, parsed("CREATE TABLE t (id INT PRIMARY KEY)"))
        .unwrap();
    let visible_before = engine.committed_seq();
    let wal_before = engine.durable_wal_records().len();
    let registered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let program_engine = Arc::clone(&engine);
    let program_registered = Arc::clone(&registered);
    let program_release = Arc::clone(&release);
    let program = std::thread::spawn(move || {
        program_engine.submit_predeclared_transaction_instrumented(
            90,
            PredeclaredTransaction::new(
                vec![
                    parsed("INSERT INTO t VALUES (1)"),
                    parsed("INSERT INTO t VALUES (2)"),
                ],
                resources(2, 2),
                TransactionCharacteristics::REPEATABLE_READ_WRITE,
            ),
            move || {
                program_registered.wait();
                program_release.wait();
            },
            |_| {},
        )
    });
    registered.wait();

    let (commit_tx, commit_rx) = mpsc::channel();
    let commit_engine = Arc::clone(&engine);
    let commit = std::thread::spawn(move || {
        let result = commit_engine.submit_transaction(90, parsed("COMMIT"));
        commit_tx.send(result).unwrap();
    });
    let (dml_tx, dml_rx) = mpsc::channel();
    let dml_engine = Arc::clone(&engine);
    let dml = std::thread::spawn(move || {
        let result = dml_engine.submit_transaction(90, parsed("INSERT INTO t VALUES (3)"));
        dml_tx.send(result).unwrap();
    });
    assert!(commit_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .is_err());
    assert!(dml_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .is_err());

    release.wait();
    let result = program.join().unwrap().unwrap();
    assert_eq!(result.operations.len(), 2);
    commit.join().unwrap();
    dml.join().unwrap();
    assert_eq!(engine.committed_seq(), visible_before + 1);
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id FROM t")
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(2)]]
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn post_durable_failure_is_indeterminate_and_recovery_owns_the_commit() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(1, parsed("CREATE TABLE t (id INT PRIMARY KEY)"))
        .unwrap();
    engine.fail_next_transaction_post_durable_apply();
    let transaction = PredeclaredTransaction::new(
        vec![parsed("INSERT INTO t VALUES (1)")],
        resources(1, 1),
        TransactionCharacteristics::REPEATABLE_READ_WRITE,
    );

    let error = engine.submit_transaction(2, transaction).unwrap_err();

    assert!(error.is_indeterminate());
    assert!(engine.is_commit_path_poisoned());
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .execute_relational_select_text("SELECT id FROM t")
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(1)]]
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn actual_resource_excess_rolls_back_before_global_claims() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(1, parsed("CREATE TABLE t (id INT PRIMARY KEY, v TEXT)"))
        .unwrap();
    let visible_before = engine.committed_seq();
    let wal_before = engine.durable_wal_records().len();
    let row_id_before = engine.read_state.mvcc.current_row_id();
    let transaction = PredeclaredTransaction::new(
        vec![parsed("INSERT INTO t VALUES (1, 'too large')")],
        TransactionResources {
            operations: 1,
            mutations: 1,
            post_image_and_wal_bytes: 0,
            maintained_index_fanout: 1,
            touched_tables: 1,
            cold_accesses: 0,
            result_bytes: 0,
        },
        TransactionCharacteristics::REPEATABLE_READ_WRITE,
    );

    let error = engine.submit_transaction(2, transaction).unwrap_err();

    assert!(error
        .to_string()
        .contains("post-image plus logical-WAL bytes"));
    assert_eq!(engine.committed_seq(), visible_before);
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
    assert!(engine.transaction_snapshot_handle(2).is_none());
    assert!(engine
        .execute_relational_select_text("SELECT id FROM t")
        .unwrap()
        .rows
        .is_empty());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn general_predeclared_and_interactive_transactions_exceed_thirty_two_operations() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(1, parsed("CREATE TABLE t (id INT PRIMARY KEY)"))
        .unwrap();
    let reads = (0..40)
        .map(|_| parsed("SELECT id FROM t WHERE id = -1"))
        .collect();
    let TransactionAdmissionResult::Predeclared(result) = engine
        .submit_transaction(
            2,
            PredeclaredTransaction::new(
                reads,
                TransactionResources {
                    operations: 40,
                    mutations: 0,
                    post_image_and_wal_bytes: 0,
                    maintained_index_fanout: 0,
                    touched_tables: 1,
                    cold_accesses: 0,
                    result_bytes: 0,
                },
                TransactionCharacteristics::REPEATABLE_READ_WRITE,
            ),
        )
        .unwrap()
    else {
        panic!("forty-operation predeclared transaction must be supported");
    };
    assert_eq!(result.class, TransactionClass::General);
    assert_eq!(result.operations.len(), 40);

    engine.submit_transaction(3, parsed("BEGIN")).unwrap();
    for id in 0..40 {
        engine
            .submit_transaction(3, parsed(&format!("INSERT INTO t VALUES ({id})")))
            .unwrap();
    }
    engine.submit_transaction(3, parsed("COMMIT")).unwrap();
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id FROM t")
            .unwrap()
            .rows
            .len(),
        40
    );
}
