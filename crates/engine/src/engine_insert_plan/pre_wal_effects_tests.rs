use super::*;
use crate::{
    parse_command, PreparedMutation, StagedRowOperation, TransactionDeltaState,
    TransactionOperation, WriteDelta, WriteSet,
};
use gpu_db_sql::ParsedCommand;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

fn parsed_insert(sql: &str) -> Insert {
    let crate::Command::Insert(insert) = parse_command(sql).expect("INSERT parses") else {
        panic!("fixture SQL must parse as INSERT");
    };
    insert
}

fn explicit_fixture(txn_id: TxnId) -> (Engine, Arc<TransactionSnapshot>, Insert) {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE effect_baseline (id serial, payload int4)")
        .expect("fixture table creates");
    engine
        .execute_text(txn_id, "BEGIN")
        .expect("BEGIN succeeds");
    engine
        .submit_transaction(
            txn_id,
            ParsedCommand::parse("CREATE SEQUENCE effect_baseline_predecessor").unwrap(),
        )
        .expect("catalog-only predecessor stages");
    let snapshot = engine
        .transaction_snapshot_handle(txn_id)
        .expect("active transaction snapshot");
    (
        engine,
        snapshot,
        parsed_insert("INSERT INTO effect_baseline (id, payload) VALUES (DEFAULT, 2)"),
    )
}

fn private_sequence_fixture(txn_id: TxnId) -> (Engine, Arc<TransactionSnapshot>, Insert) {
    let engine = Engine::new_local();
    engine
        .execute_text(txn_id, "BEGIN")
        .expect("BEGIN succeeds");
    engine
        .submit_transaction(
            txn_id,
            ParsedCommand::parse("CREATE TABLE effect_private_seed (id serial, payload int4)")
                .unwrap(),
        )
        .expect("private catalog-only table creation stages");
    engine
        .submit_transaction(
            txn_id,
            ParsedCommand::parse("ALTER SEQUENCE effect_private_seed_id_seq RESTART WITH 41")
                .unwrap(),
        )
        .expect("private sequence restart stages");
    let snapshot = engine
        .transaction_snapshot_handle(txn_id)
        .expect("active transaction snapshot");
    (
        engine,
        snapshot,
        parsed_insert("INSERT INTO effect_private_seed (id, payload) VALUES (DEFAULT, 2)"),
    )
}

fn truncate_fixture(
    txn_id: TxnId,
    truncate_statement: &str,
) -> (Engine, Arc<TransactionSnapshot>, Insert) {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE effect_truncate (id serial, payload int4)")
        .expect("fixture table creates");
    engine
        .execute_text(txn_id, "BEGIN")
        .expect("BEGIN succeeds");
    engine
        .submit_transaction(txn_id, ParsedCommand::parse(truncate_statement).unwrap())
        .expect("transactional truncate stages");
    let snapshot = engine
        .transaction_snapshot_handle(txn_id)
        .expect("active transaction snapshot");
    (
        engine,
        snapshot,
        parsed_insert("INSERT INTO effect_truncate (id, payload) VALUES (DEFAULT, 2)"),
    )
}

#[derive(Debug, PartialEq, Eq)]
struct DeltaCaptureFingerprint {
    generation: u64,
    operations: gpu_db_wal::CanonicalDigest,
    references: gpu_db_wal::CanonicalDigest,
    catalog_base: Option<usize>,
    catalog_overlay: Option<usize>,
    resident_shards: usize,
    resident_authority: usize,
    cold_chunks: usize,
    cold_authority: usize,
    resident_strong_count: usize,
    cold_strong_count: usize,
    sequence_state_by_oid: BTreeMap<u32, (i64, bool)>,
    sequence_state: BTreeMap<String, (i64, bool)>,
    private_gpu_bytes_by_gpu: BTreeMap<u16, u64>,
    commit_gpu_bytes_by_gpu: BTreeMap<u16, u64>,
}

fn delta_capture_fingerprint(snapshot: &TransactionSnapshot) -> DeltaCaptureFingerprint {
    let statement_lock = Arc::clone(&snapshot.statement_lock);
    let _statement_guard = statement_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let delta = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    DeltaCaptureFingerprint {
        generation: delta.generation,
        operations: operation_identity_fingerprint(&delta.operations).unwrap(),
        references: sequence_reference_fingerprint(&delta.sequence_value_references).unwrap(),
        catalog_base: delta
            .catalog_base
            .as_ref()
            .map(|catalog| Arc::as_ptr(catalog) as usize),
        catalog_overlay: delta
            .catalog_overlay
            .as_ref()
            .map(|catalog| Arc::as_ptr(catalog) as usize),
        resident_shards: Arc::as_ptr(&delta.resident_shards) as usize,
        resident_authority: Arc::as_ptr(&delta.resident_shards_authority) as usize,
        cold_chunks: Arc::as_ptr(&delta.streaming_cold_chunks) as usize,
        cold_authority: Arc::as_ptr(&delta.streaming_cold_chunks_authority) as usize,
        resident_strong_count: Arc::strong_count(&delta.resident_shards),
        cold_strong_count: Arc::strong_count(&delta.streaming_cold_chunks),
        sequence_state_by_oid: delta.sequence_state_by_oid.clone(),
        sequence_state: delta.sequence_state.clone(),
        private_gpu_bytes_by_gpu: delta.private_gpu_bytes_by_gpu.clone(),
        commit_gpu_bytes_by_gpu: delta.commit_gpu_bytes_by_gpu.clone(),
    }
}

fn synthetic_sequence_reference(
    statement_ordinal: u32,
    expression_ordinal: u32,
) -> BinarySequenceValueReference {
    BinarySequenceValueReference {
        transition_txn_id: 1,
        parent_txn_id: 1,
        statement_ordinal,
        expression_ordinal,
        sequence_oid: 1,
        returned_value: 1,
        input_digest: [7; 32],
        table_oid: 1,
        column_id: 1,
        staging_row_ordinal: 0,
        row_id: 0,
        final_value_overwritten: false,
        default_expression: true,
    }
}

fn synthetic_row_operation(
    sequence_oid: u32,
    sequence_advance: (i64, bool),
) -> TransactionOperation {
    TransactionOperation::Row(Arc::new(StagedRowOperation {
        statement_digest: [9; 32],
        sequence_input_oids: BTreeMap::from([("effect_row_seq".to_string(), sequence_oid)]),
        delta: WriteDelta {
            write_set: WriteSet::default(),
            read_snapshot: 0,
            catalog_dependencies: BTreeMap::new(),
            foreign_key_dependencies: BTreeSet::new(),
            rows_consumed: 0,
            mutation: PreparedMutation::Insert {
                table: "effect_row".to_string(),
                inserted_rows: Vec::new(),
                seq_advances: BTreeMap::from([("effect_row_seq".to_string(), sequence_advance)]),
            },
        },
    }))
}

fn capture_explicit(engine: &Engine, txn_id: TxnId, insert: &Insert) -> PreparedInsertEffectPlan {
    PreparedInsertEffectPlan::prepare_explicit_for_test(engine, txn_id, insert)
        .expect("effect baseline captures")
}

#[test]
fn effect_baseline_parent_identity_is_typed_for_autocommit_and_explicit() {
    const EXPLICIT_TXN: TxnId = 91_001;
    const AUTOCOMMIT_TXN: TxnId = 91_002;
    let (explicit_engine, _snapshot, insert) = explicit_fixture(EXPLICIT_TXN);
    let explicit = capture_explicit(&explicit_engine, EXPLICIT_TXN, &insert);

    let autocommit_engine = Engine::new_local();
    autocommit_engine
        .execute_text(1, "CREATE TABLE effect_baseline (id serial, payload int4)")
        .unwrap();
    let autocommit = PreparedInsertEffectPlan::prepare_autocommit_for_test(
        &autocommit_engine,
        AUTOCOMMIT_TXN,
        &insert,
    )
    .unwrap();
    let legacy_sql_digest =
        transaction_statement_digest(&crate::Command::Insert(insert.clone())).unwrap();
    let explicit_parent = explicit.parent_for_test();
    let autocommit_parent = autocommit.parent_for_test();
    assert_eq!(explicit_parent.0, EXPLICIT_TXN);
    assert!(!explicit_parent.1);
    assert_ne!(
        explicit_parent.2, legacy_sql_digest,
        "typed INSERT effects must be parented by the prepared canonical intent, not SQL/JSON"
    );
    assert_eq!(
        explicit_parent.3.as_u32(),
        1,
        "the staged predecessor is the exact operation ordinal"
    );
    assert_eq!(explicit_parent.4, 0);
    assert_eq!(autocommit_parent.0, AUTOCOMMIT_TXN);
    assert!(autocommit_parent.1);
    assert_ne!(
        autocommit_parent.2, legacy_sql_digest,
        "autocommit uses the same typed-intent identity boundary"
    );
    assert_ne!(
        explicit_parent.2, autocommit_parent.2,
        "the statement ordinal is intentional typed-statement identity"
    );
    assert_eq!(
        autocommit_parent.3,
        crate::insert_semantic_ir::InsertStatementOrdinal::FIRST
    );
    assert_eq!(autocommit_parent.4, 0);
    assert!(
        PreparedInsertEffectPlan::prepare_autocommit_for_test(&autocommit_engine, 0, &insert,)
            .is_err()
    );
}

#[test]
fn effect_baseline_validation_is_read_only_and_autocommit_rejects_catalog_drift() {
    const TXN: TxnId = 91_003;
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE effect_autocommit (id serial, payload int4)",
        )
        .unwrap();
    let insert = parsed_insert("INSERT INTO effect_autocommit (id, payload) VALUES (DEFAULT, 3)");
    let wal_before = engine.durable_wal_records().len();
    let committed_before = engine.committed_seq();
    let catalog_before = engine.catalog_snapshot();
    let sequences_before = catalog_before.relational_sequences.clone();
    let currval_before = engine.sequence_currval_effects(TXN);
    let plan = PreparedInsertEffectPlan::prepare_autocommit_for_test(&engine, TXN, &insert)
        .expect("autocommit baseline captures");
    plan.validate_current(&engine).unwrap();
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.committed_seq(), committed_before);
    assert!(Arc::ptr_eq(&engine.catalog_snapshot(), &catalog_before));
    assert_eq!(
        engine.catalog_snapshot().relational_sequences,
        sequences_before
    );
    assert_eq!(engine.sequence_currval_effects(TXN), currval_before);

    engine
        .execute_text(2, "CREATE TABLE effect_autocommit_drift (id int4)")
        .unwrap();
    assert!(plan.validate_current(&engine).is_err());
}

#[test]
fn effect_baseline_explicit_validation_is_read_only_and_owner_is_movable() {
    const TXN: TxnId = 91_004;
    fn assert_send<T: Send>() {}
    assert_send::<PreparedInsertEffectPlan>();

    let (engine, snapshot, insert) = explicit_fixture(TXN);
    let wal_before = engine.durable_wal_records().len();
    let committed_before = engine.committed_seq();
    let catalog_before = engine.catalog_snapshot();
    let sequence_values_before = catalog_before.relational_sequences.clone();
    let currval_before = engine.sequence_currval_effects(TXN);
    let (next_row_id_before, sequence_state_before) = {
        let delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (delta.next_row_id, delta.sequence_state_by_oid.clone())
    };
    let fingerprint_before = delta_capture_fingerprint(&snapshot);
    let plan = capture_explicit(&engine, TXN, &insert);
    assert_eq!(
        delta_capture_fingerprint(&snapshot),
        fingerprint_before,
        "baseline capture must not retain a GPU map generation or mutate transaction state"
    );
    plan.validate_current(&engine).unwrap();
    assert!(plan
        .explicit_baseline_for_test()
        .expect("explicit baseline")
        .touched_sequence_seeds
        .iter()
        .all(|seed| seed.oid_state.is_none() && seed.name_state.is_none()));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.committed_seq(), committed_before);
    assert!(Arc::ptr_eq(&engine.catalog_snapshot(), &catalog_before));
    assert_eq!(
        engine.catalog_snapshot().relational_sequences,
        sequence_values_before
    );
    assert_eq!(engine.sequence_currval_effects(TXN), currval_before);
    let delta = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(delta.next_row_id, next_row_id_before);
    assert_eq!(delta.sequence_state_by_oid, sequence_state_before);
}

#[test]
fn effect_baseline_captures_private_sequence_restart_in_both_state_maps() {
    const TXN: TxnId = 91_004_001;
    let (engine, snapshot, insert) = private_sequence_fixture(TXN);
    let fingerprint_before = delta_capture_fingerprint(&snapshot);
    let plan = capture_explicit(&engine, TXN, &insert);
    assert_eq!(delta_capture_fingerprint(&snapshot), fingerprint_before);
    plan.validate_current(&engine).unwrap();

    let baseline = plan
        .explicit_baseline_for_test()
        .expect("explicit baseline");
    let seed = baseline
        .touched_sequence_seeds
        .iter()
        .find(|seed| seed.effective_name == "effect_private_seed_id_seq")
        .expect("private serial default contributes its effective sequence name");
    assert_ne!(seed.oid, 0);
    assert_eq!(seed.oid_state, Some((41, false)));
    assert_eq!(seed.name_state, Some((41, false)));
}

#[test]
fn effect_baseline_distinguishes_plain_and_restart_identity_truncates() {
    const PLAIN_TXN: TxnId = 91_004_010;
    let (engine, snapshot, insert) = truncate_fixture(PLAIN_TXN, "TRUNCATE effect_truncate");
    {
        let delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(matches!(
            delta.operations.as_slice(),
            [TransactionOperation::TableReset(reset)] if reset.sequence_reset_identity.is_none()
        ));
        assert!(
            delta.catalog_base.is_none(),
            "plain TRUNCATE is row-only and must not retain a catalog base"
        );
    }
    let plain = capture_explicit(&engine, PLAIN_TXN, &insert);
    let plain_baseline = plain
        .explicit_baseline_for_test()
        .expect("explicit baseline");
    assert!(plain_baseline.catalog_base.is_none());
    plain.validate_current(&engine).unwrap();

    const RESTART_TXN: TxnId = 91_004_011;
    let (engine, snapshot, insert) =
        truncate_fixture(RESTART_TXN, "TRUNCATE effect_truncate RESTART IDENTITY");
    {
        let delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(matches!(
            delta.operations.as_slice(),
            [TransactionOperation::TableReset(reset)] if reset.sequence_reset_identity.is_some()
        ));
        assert!(
            delta
                .catalog_base
                .as_ref()
                .is_some_and(|base| Arc::ptr_eq(base, &snapshot.catalog)),
            "RESTART IDENTITY retains the exact catalog base for its sequence reset"
        );
    }
    let restart = capture_explicit(&engine, RESTART_TXN, &insert);
    let restart_baseline = restart
        .explicit_baseline_for_test()
        .expect("explicit baseline");
    assert!(restart_baseline
        .catalog_base
        .as_ref()
        .is_some_and(|base| Arc::ptr_eq(base, &snapshot.catalog)));
    restart.validate_current(&engine).unwrap();
}

#[test]
fn effect_baseline_counts_contiguous_existing_statement_references() {
    const TXN: TxnId = 91_004_002;
    let (engine, snapshot, insert) = explicit_fixture(TXN);
    {
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _statement_guard = statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let statement_ordinal = u32::try_from(delta.operations.len()).unwrap();
        delta
            .sequence_value_references
            .push(synthetic_sequence_reference(statement_ordinal, 0));
        delta
            .sequence_value_references
            .push(synthetic_sequence_reference(statement_ordinal, 1));
    }
    let plan = capture_explicit(&engine, TXN, &insert);
    assert_eq!(plan.parent_for_test().4, 2);
    plan.validate_current(&engine).unwrap();
}

#[test]
fn effect_baseline_operation_witness_binds_row_sequence_inputs_and_advances() {
    let original = synthetic_row_operation(41, (41, false));
    let changed_oid = synthetic_row_operation(42, (41, false));
    let changed_advance = synthetic_row_operation(41, (42, true));
    let original_fingerprint = operation_identity_fingerprint(&[original]).unwrap();
    assert_ne!(
        operation_identity_fingerprint(&[changed_oid]).unwrap(),
        original_fingerprint,
        "stable sequence input OIDs are classifier-relevant operation identity"
    );
    assert_ne!(
        operation_identity_fingerprint(&[changed_advance]).unwrap(),
        original_fingerprint,
        "private sequence advances are classifier-relevant operation identity"
    );
}

#[test]
fn effect_baseline_rejects_removed_or_replaced_explicit_snapshot() {
    const REMOVED_TXN: TxnId = 91_005;
    let (engine, _snapshot, insert) = explicit_fixture(REMOVED_TXN);
    let plan = capture_explicit(&engine, REMOVED_TXN, &insert);
    engine.execute_text(REMOVED_TXN, "ROLLBACK").unwrap();
    assert!(plan.validate_current(&engine).is_err());

    const REPLACED_TXN: TxnId = 91_006;
    let (engine, snapshot, insert) = explicit_fixture(REPLACED_TXN);
    let plan = capture_explicit(&engine, REPLACED_TXN, &insert);
    let replacement =
        engine.capture_transaction_snapshot(snapshot.boundary, snapshot.characteristics);
    engine
        .active_snapshots
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .replace_transaction_snapshot(REPLACED_TXN, &snapshot, replacement, false)
        .unwrap();
    assert!(plan.validate_current(&engine).is_err());
    engine.execute_text(REPLACED_TXN, "ROLLBACK").unwrap();
}

fn assert_explicit_delta_drift(
    mutate: impl FnOnce(&mut TransactionDeltaState, &ExplicitOverlayBaseline),
) {
    const TXN: TxnId = 91_007;
    let (engine, snapshot, insert) = explicit_fixture(TXN);
    let plan = capture_explicit(&engine, TXN, &insert);
    let baseline = plan
        .explicit_baseline_for_test()
        .expect("explicit plan exposes its test baseline");
    let statement_lock = Arc::clone(&snapshot.statement_lock);
    let _statement_guard = statement_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut delta = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    mutate(&mut delta, baseline);
    drop(delta);
    drop(_statement_guard);
    assert!(plan.validate_current(&engine).is_err());
}

#[test]
fn effect_baseline_rejects_each_explicit_delta_witness_drift() {
    assert_explicit_delta_drift(|delta, _| delta.generation = delta.generation.saturating_add(1));
    assert_explicit_delta_drift(|delta, _| {
        let operation = delta
            .operations
            .first()
            .expect("fixture retains one staged operation")
            .clone();
        delta.operations.push(operation);
    });
    assert_explicit_delta_drift(|delta, _| {
        let TransactionOperation::Catalog(staged) = delta
            .operations
            .first_mut()
            .expect("fixture retains one staged catalog operation")
        else {
            panic!("fixture predecessor must remain catalog-only");
        };
        Arc::make_mut(staged).statement_digest[0] ^= 0x5a;
    });
    assert_explicit_delta_drift(|delta, _| delta.next_row_id = delta.next_row_id.saturating_add(1));
    assert_explicit_delta_drift(|delta, _| {
        delta
            .sequence_value_references
            .push(synthetic_sequence_reference(0, 0));
    });
    assert_explicit_delta_drift(|delta, _| {
        let replacement = Arc::new(delta.resident_shards.as_ref().clone());
        delta.resident_shards = Arc::clone(&replacement);
        delta.resident_shards_authority = replacement;
    });
    assert_explicit_delta_drift(|delta, _| {
        let replacement = Arc::new(delta.streaming_cold_chunks.as_ref().clone());
        delta.streaming_cold_chunks = Arc::clone(&replacement);
        delta.streaming_cold_chunks_authority = replacement;
    });
    assert_explicit_delta_drift(|delta, baseline| {
        assert!(baseline.catalog_overlay.is_some());
        delta.catalog_overlay = None;
    });
    assert_explicit_delta_drift(|delta, baseline| {
        let seed = baseline
            .touched_sequence_seeds
            .first()
            .expect("serial DEFAULT contributes one touched OID");
        let replacement = match seed.oid_state {
            Some((value, is_called)) => (value, !is_called),
            None => (1, true),
        };
        delta.sequence_state_by_oid.insert(seed.oid, replacement);
    });
    assert_explicit_delta_drift(|delta, baseline| {
        let seed = baseline
            .touched_sequence_seeds
            .first()
            .expect("serial DEFAULT contributes one touched OID");
        delta
            .sequence_state
            .insert(seed.effective_name.clone(), (1, true));
    });
}

#[test]
fn effect_baseline_rejects_equal_content_catalog_overlay_arc_replacement() {
    const TXN: TxnId = 91_009;
    let (engine, snapshot, insert) = explicit_fixture(TXN);
    let plan = capture_explicit(&engine, TXN, &insert);
    let baseline = plan
        .explicit_baseline_for_test()
        .expect("explicit plan exposes its test baseline");
    let original = baseline
        .catalog_overlay
        .as_ref()
        .expect("fixture retained a catalog overlay");
    let statement_lock = Arc::clone(&snapshot.statement_lock);
    let statement_guard = statement_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut delta = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let current = delta
        .catalog_overlay
        .as_ref()
        .expect("current overlay remains present");
    assert!(Arc::ptr_eq(current, original));
    delta.catalog_overlay = Some(Arc::new(current.as_ref().clone()));
    drop(delta);
    drop(statement_guard);
    assert!(plan.validate_current(&engine).is_err());
}

#[test]
fn effect_baseline_rejects_equal_content_catalog_base_arc_replacement() {
    const TXN: TxnId = 91_009_001;
    let (engine, snapshot, insert) = explicit_fixture(TXN);
    let plan = capture_explicit(&engine, TXN, &insert);
    let baseline = plan
        .explicit_baseline_for_test()
        .expect("explicit plan exposes its test baseline");
    let original = baseline
        .catalog_base
        .as_ref()
        .expect("catalog-bearing fixture retained a base catalog");
    let statement_lock = Arc::clone(&snapshot.statement_lock);
    let statement_guard = statement_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut delta = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let current = delta
        .catalog_base
        .as_ref()
        .expect("current catalog base remains present");
    assert!(Arc::ptr_eq(current, original));
    delta.catalog_base = Some(Arc::new(current.as_ref().clone()));
    drop(delta);
    drop(statement_guard);
    assert!(plan.validate_current(&engine).is_err());
}

#[test]
fn effect_baseline_rejects_noncontiguous_or_duplicate_existing_statement_references() {
    const TXN: TxnId = 91_008;
    for (witness, expression_ordinals) in
        [("gap", &[1_u32][..]), ("duplicate", &[0_u32, 0_u32][..])]
    {
        let (engine, snapshot, insert) = explicit_fixture(TXN);
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let statement_guard = statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        {
            let mut delta = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let statement_ordinal = u32::try_from(delta.operations.len()).unwrap();
            for expression_ordinal in expression_ordinals {
                delta
                    .sequence_value_references
                    .push(synthetic_sequence_reference(
                        statement_ordinal,
                        *expression_ordinal,
                    ));
            }
        }
        drop(statement_guard);
        assert!(
            PreparedInsertEffectPlan::prepare_explicit_for_test(&engine, TXN, &insert).is_err(),
            "{witness} expression ordinals must reject the capture"
        );
    }
}

#[test]
fn effect_baseline_leaf_has_no_physical_or_state_mutation_authority() {
    let source = include_str!("pre_wal_effects.rs");
    for forbidden in [
        "WalBuffer",
        "PreparedBinaryInsertTemplate",
        "BoundBinaryInsert",
        "append_canonical",
        "commit_sequence_value_transition",
        "materialize_published_sequence_defaults",
        "bind_sequence_default_insert_rows",
        "allocate_transaction_id",
        "DeviceInsertPlan",
        "compile_typed_insert_device_plan",
        "into_resident_append_source",
        "DmlExecutionResult",
        "RelationalSelectResult",
        "sequence_value_outcomes",
        "sequence_currval_effects",
        "reserve_gpu",
        "reserve_capacity",
        "apply_and_publish",
    ] {
        assert!(
            !source.contains(forbidden),
            "baseline leaf must not own {forbidden}"
        );
    }
    let owner_fields = source
        .split("struct PreparedInsertEffectPlan")
        .nth(1)
        .expect("owner struct exists")
        .split("impl PreparedInsertEffectPlan")
        .next()
        .expect("owner impl follows fields");
    assert!(
        !owner_fields.contains("MutexGuard"),
        "the move-only owner retains no statement guard"
    );
    for line in source.lines().filter(|line| line.contains("MutexGuard")) {
        assert!(
            line.contains("statement_guard"),
            "MutexGuard may occur only as a borrowed constructor parameter: {line}"
        );
    }
}
