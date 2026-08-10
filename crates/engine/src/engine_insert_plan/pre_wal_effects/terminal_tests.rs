use super::super::classification::TerminalPlanSabotage;
use super::super::receipt::{PublishedReceiptInput, SequenceReceiptBundle};
use crate::{
    parse_command, transaction_statement_digest, Command, Engine, Insert, PreparedMutation,
    SqlValue, StagedRowOperation, TransactionOperation, TransactionSnapshot, TxnId, WriteDelta,
    WriteSet, PROJECTION_WILDCARD_SENTINEL,
};
use gpu_db_sql::ParsedCommand;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::Ordering;
use std::sync::Arc;

fn parsed_insert(sql: &str) -> Insert {
    let Command::Insert(insert) = parse_command(sql).expect("INSERT parses") else {
        panic!("fixture SQL must parse as INSERT");
    };
    insert
}

fn stage(engine: &Engine, txn_id: TxnId, sql: &str) {
    engine
        .submit_transaction(
            txn_id,
            ParsedCommand::parse(sql).expect("fixture DDL parses"),
        )
        .expect("fixture DDL stages");
}

fn create_sequence_and_table(engine: &Engine, txn_id: TxnId, sequence: &str, table: &str) {
    stage(engine, txn_id, &format!("CREATE SEQUENCE {sequence}"));
    stage(
        engine,
        txn_id,
        &format!(
            "CREATE TABLE {table} (id int4 DEFAULT nextval('{sequence}'::regclass), payload int4)"
        ),
    );
}

fn synthetic_private_row_advance(
    engine: &Engine,
    txn_id: TxnId,
    sequence: &str,
    table: &str,
    state: (i64, bool),
) {
    let snapshot = engine.transaction_snapshot_handle(txn_id).unwrap();
    let statement_lock = Arc::clone(&snapshot.statement_lock);
    let _statement_guard = statement_lock.lock().unwrap();
    let oid = snapshot.transaction_catalog().relational_sequences[sequence].oid;
    let command = parse_command(&format!("INSERT INTO {table} (id) VALUES (DEFAULT)")).unwrap();
    let mut delta = snapshot.delta.lock().unwrap();
    delta.sequence_state.insert(sequence.to_string(), state);
    delta.sequence_state_by_oid.insert(oid, state);
    delta
        .operations
        .push(TransactionOperation::Row(Arc::new(StagedRowOperation {
            statement_digest: transaction_statement_digest(&command).unwrap(),
            sequence_input_oids: BTreeMap::from([(sequence.to_string(), oid)]),
            delta: WriteDelta {
                write_set: WriteSet::default(),
                read_snapshot: snapshot.boundary,
                catalog_dependencies: BTreeMap::new(),
                foreign_key_dependencies: BTreeSet::new(),
                rows_consumed: 0,
                mutation: PreparedMutation::Insert {
                    table: table.to_string(),
                    inserted_rows: Vec::new(),
                    seq_advances: BTreeMap::from([(sequence.to_string(), state)]),
                },
            },
        })));
    delta.generation = delta.generation.saturating_add(1);
}

fn synthetic_sequence_reference(
    statement_ordinal: u32,
    expression_ordinal: u32,
) -> crate::BinarySequenceValueReference {
    crate::BinarySequenceValueReference {
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

fn published_receipts(
    plan: &super::super::PreparedInsertEffectPlan,
    values: &[i64],
) -> SequenceReceiptBundle {
    let digests = plan.published_receipt_input_digests_for_test();
    assert_eq!(
        digests.len(),
        values.len(),
        "fixture supplies every published receipt"
    );
    SequenceReceiptBundle::for_test(
        digests
            .into_iter()
            .zip(values)
            .enumerate()
            .map(|(ordinal, (input_digest, value))| {
                PublishedReceiptInput::for_test(
                    990_000 + u64::try_from(ordinal).unwrap(),
                    *value,
                    input_digest,
                )
            })
            .collect(),
    )
}

#[derive(Debug, PartialEq, Eq)]
struct TerminalStateFingerprint {
    wal_len: usize,
    committed: u64,
    catalog: usize,
    currval: Vec<(u32, i64)>,
    transaction_id_allocator: u64,
    pending_claims: usize,
    sequence_outcomes: usize,
    transaction_private_gpu_bytes: BTreeMap<u16, u64>,
    transaction_retained_gpu: String,
    commit_status: String,
    publication: String,
    delta: Option<ExplicitDeltaFingerprint>,
}

#[derive(Debug, PartialEq, Eq)]
struct ExplicitDeltaFingerprint {
    generation: u64,
    operations: String,
    references: String,
    sequence_state_by_oid: BTreeMap<u32, (i64, bool)>,
    sequence_state: BTreeMap<String, (i64, bool)>,
    catalog_base: Option<usize>,
    catalog_overlay: Option<usize>,
    private_gpu_bytes_by_gpu: BTreeMap<u16, u64>,
    commit_gpu_bytes_by_gpu: BTreeMap<u16, u64>,
}

fn terminal_fingerprint(
    engine: &Engine,
    txn_id: TxnId,
    snapshot: Option<&TransactionSnapshot>,
) -> TerminalStateFingerprint {
    let delta = snapshot.map(|snapshot| {
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _statement_guard = statement_lock.lock().unwrap();
        let delta = snapshot.delta.lock().unwrap();
        ExplicitDeltaFingerprint {
            generation: delta.generation,
            operations: format!("{:?}", delta.operations),
            references: format!("{:?}", delta.sequence_value_references),
            sequence_state_by_oid: delta.sequence_state_by_oid.clone(),
            sequence_state: delta.sequence_state.clone(),
            catalog_base: delta
                .catalog_base
                .as_ref()
                .map(|catalog| Arc::as_ptr(catalog) as usize),
            catalog_overlay: delta
                .catalog_overlay
                .as_ref()
                .map(|catalog| Arc::as_ptr(catalog) as usize),
            private_gpu_bytes_by_gpu: delta.private_gpu_bytes_by_gpu.clone(),
            commit_gpu_bytes_by_gpu: delta.commit_gpu_bytes_by_gpu.clone(),
        }
    });
    TerminalStateFingerprint {
        wal_len: engine.durable_wal_records().len(),
        committed: engine.committed_seq(),
        catalog: Arc::as_ptr(&engine.catalog_snapshot()) as usize,
        currval: engine.sequence_currval_effects(txn_id),
        transaction_id_allocator: engine.transaction_id_allocator.load(Ordering::Acquire),
        pending_claims: engine.pending_transaction_claims.lock().unwrap().len(),
        sequence_outcomes: engine.sequence_value_outcomes.lock().unwrap().len(),
        transaction_private_gpu_bytes: engine.transaction_private_gpu_bytes.lock().unwrap().clone(),
        transaction_retained_gpu: format!(
            "{:?}",
            engine.transaction_retained_gpu_allocations.lock().unwrap()
        ),
        commit_status: {
            let commit = engine.commit_state();
            format!("{:?}", commit.transaction_status)
        },
        publication: format!("{:?}", engine.commit_publication),
        delta,
    }
}

#[test]
fn inert_terminal_autocommit_published_receipt_seals_without_side_effects() {
    const TXN: TxnId = 99_001;
    let engine = Engine::new_local();
    engine
        .execute_text(99_000, "CREATE SEQUENCE terminal_published")
        .unwrap();
    engine
        .execute_text(
            99_000_001,
            "CREATE TABLE terminal_published_table (id int4 DEFAULT nextval('terminal_published'::regclass), payload int4)",
        )
        .unwrap();
    let plan = super::super::PreparedInsertEffectPlan::prepare_autocommit_for_test(
        &engine,
        TXN,
        &parsed_insert("INSERT INTO terminal_published_table (id, payload) VALUES (DEFAULT, 7)"),
    )
    .unwrap();
    let receipts = published_receipts(&plan, &[37]);
    let before = terminal_fingerprint(&engine, TXN, None);
    let evidence = plan.seal_for_test(&engine, receipts).unwrap();
    assert_eq!(terminal_fingerprint(&engine, TXN, None), before);
    assert!(evidence.autocommit);
    assert_eq!(evidence.sequence_outputs.len(), 1);
    assert_eq!(evidence.sequence_outputs[0].returned_value, 37);
    assert!(matches!(
        evidence.sequence_outputs[0].transition,
        super::super::receipt::SequenceTransitionEvidence::Published {
            transition_txn_id: 990_000
        }
    ));
    assert_eq!(evidence.returning.cell_count, 0);
}

#[test]
fn inert_terminal_explicit_private_restart_chain_is_derived_and_side_effect_free() {
    const TXN: TxnId = 99_002;
    let engine = Engine::new_local();
    engine.execute_text(TXN, "BEGIN").unwrap();
    create_sequence_and_table(&engine, TXN, "terminal_private", "terminal_private_table");
    stage(
        &engine,
        TXN,
        "ALTER SEQUENCE terminal_private RESTART WITH 40",
    );
    synthetic_private_row_advance(
        &engine,
        TXN,
        "terminal_private",
        "terminal_private_table",
        (41, true),
    );
    let snapshot = engine.transaction_snapshot_handle(TXN).unwrap();
    let plan = super::super::PreparedInsertEffectPlan::prepare_explicit_for_test(
        &engine,
        TXN,
        &parsed_insert(
            "INSERT INTO terminal_private_table (id, payload) VALUES (DEFAULT, 1), (DEFAULT, 2)",
        ),
    )
    .unwrap();
    let before = terminal_fingerprint(&engine, TXN, Some(&snapshot));
    let evidence = plan
        .seal_for_test(&engine, SequenceReceiptBundle::for_test(Vec::new()))
        .unwrap();
    assert_eq!(terminal_fingerprint(&engine, TXN, Some(&snapshot)), before);
    assert_eq!(
        evidence
            .sequence_outputs
            .iter()
            .map(|output| output.returned_value)
            .collect::<Vec<_>>(),
        [42, 43]
    );
    let [first, second] = evidence.sequence_outputs.as_ref() else {
        panic!("two defaults must retain ordered private evidence");
    };
    let super::super::receipt::SequenceTransitionEvidence::Private {
        outcome_digest: first_outcome,
        ..
    } = first.transition
    else {
        panic!("first default is private");
    };
    assert!(matches!(
        second.transition,
        super::super::receipt::SequenceTransitionEvidence::Private {
            predecessor: super::super::receipt::PrivatePredecessorEvidence::PlannedOutcome(digest),
            ..
        } if digest == first_outcome
    ));
}

/// The live seal uses the same stable-OID classifier as the inert terminal, but retains the
/// private advance with the consumed typed artifact so COMMIT can encode its final sequence
/// state without reopening a legacy `WriteDelta` path.
#[test]
fn live_explicit_private_restart_seal_retains_ordered_typed_advances() {
    const TXN: TxnId = 99_002_001;
    let engine = Engine::new_local();
    engine.execute_text(TXN, "BEGIN").unwrap();
    create_sequence_and_table(
        &engine,
        TXN,
        "live_private_restart",
        "live_private_restart_table",
    );
    stage(
        &engine,
        TXN,
        "ALTER SEQUENCE live_private_restart RESTART WITH 40",
    );
    let plan = super::super::PreparedInsertEffectPlan::prepare_explicit_for_test(
        &engine,
        TXN,
        &parsed_insert(
            "INSERT INTO live_private_restart_table (id, payload) \
             VALUES (DEFAULT, 1), (DEFAULT, 2)",
        ),
    )
    .unwrap();
    let sealed = plan.seal_live_explicit(&[]).unwrap();
    assert_eq!(sealed.private_sequence_advances.len(), 2);
    assert!(sealed
        .private_sequence_advances
        .iter()
        .all(|advance| advance.next_state.1));
    assert_eq!(
        sealed
            .private_sequence_advances
            .iter()
            .map(|advance| advance.next_state.0)
            .collect::<Vec<_>>(),
        [40, 41]
    );
    assert_eq!(
        sealed.private_sequence_advances[1].predecessor_tag, 2,
        "the second DEFAULT must retain the first typed private outcome"
    );
    drop(sealed.batch);
}

#[test]
fn inert_terminal_mixes_caller_published_receipts_with_private_plan_values() {
    const TXN: TxnId = 99_003;
    let engine = Engine::new_local();
    engine
        .execute_text(99_003_000, "CREATE SEQUENCE terminal_mixed_published")
        .unwrap();
    engine.execute_text(TXN, "BEGIN").unwrap();
    stage(&engine, TXN, "CREATE SEQUENCE terminal_mixed_private");
    stage(
        &engine,
        TXN,
        "CREATE TABLE terminal_mixed_table (published_id int4 DEFAULT nextval('terminal_mixed_published'::regclass), private_id int4 DEFAULT nextval('terminal_mixed_private'::regclass))",
    );
    let snapshot = engine.transaction_snapshot_handle(TXN).unwrap();
    let plan = super::super::PreparedInsertEffectPlan::prepare_explicit_for_test(
        &engine,
        TXN,
        &parsed_insert(
            "INSERT INTO terminal_mixed_table (published_id, private_id) VALUES (DEFAULT, DEFAULT)",
        ),
    )
    .unwrap();
    let receipts = published_receipts(&plan, &[71]);
    let before = terminal_fingerprint(&engine, TXN, Some(&snapshot));
    let evidence = plan.seal_for_test(&engine, receipts).unwrap();
    assert_eq!(terminal_fingerprint(&engine, TXN, Some(&snapshot)), before);
    assert_eq!(
        evidence
            .sequence_outputs
            .iter()
            .map(|output| output.returned_value)
            .collect::<Vec<_>>(),
        [71, 1]
    );
    assert!(matches!(
        evidence.sequence_outputs[0].transition,
        super::super::receipt::SequenceTransitionEvidence::Published { .. }
    ));
    assert!(matches!(
        evidence.sequence_outputs[1].transition,
        super::super::receipt::SequenceTransitionEvidence::Private { .. }
    ));
}

#[test]
fn inert_terminal_preserves_nonzero_expression_base_and_sparse_local_slots() {
    const TXN: TxnId = 99_004;
    let engine = Engine::new_local();
    engine.execute_text(TXN, "BEGIN").unwrap();
    stage(&engine, TXN, "CREATE SEQUENCE terminal_hole_a");
    stage(&engine, TXN, "CREATE SEQUENCE terminal_hole_b");
    stage(
        &engine,
        TXN,
        "CREATE TABLE terminal_hole_table (a int4 DEFAULT nextval('terminal_hole_a'::regclass), b int4 DEFAULT nextval('terminal_hole_b'::regclass), payload int4)",
    );
    let snapshot = engine.transaction_snapshot_handle(TXN).unwrap();
    {
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _statement_guard = statement_lock.lock().unwrap();
        let mut delta = snapshot.delta.lock().unwrap();
        let statement_ordinal = u32::try_from(
            delta
                .operations
                .iter()
                .filter(|operation| matches!(operation, TransactionOperation::TypedInsert(_)))
                .count(),
        )
        .unwrap();
        delta
            .sequence_value_references
            .push(synthetic_sequence_reference(statement_ordinal, 0));
        delta
            .sequence_value_references
            .push(synthetic_sequence_reference(statement_ordinal, 1));
    }
    let plan = super::super::PreparedInsertEffectPlan::prepare_explicit_for_test(
        &engine,
        TXN,
        &parsed_insert(
            "INSERT INTO terminal_hole_table (a, b, payload) VALUES (DEFAULT, 7, 1), (8, DEFAULT, 2)",
        ),
    )
    .unwrap();
    let evidence = plan
        .seal_for_test(&engine, SequenceReceiptBundle::for_test(Vec::new()))
        .unwrap();
    assert_eq!(evidence.expression_ordinal_base, 2);
    assert_eq!(
        evidence
            .sequence_outputs
            .iter()
            .map(|output| output.local_expression_ordinal)
            .collect::<Vec<_>>(),
        [0, 3]
    );
    assert_eq!(
        evidence
            .sequence_outputs
            .iter()
            .map(|output| output.absolute_expression_ordinal)
            .collect::<Vec<_>>(),
        [2, 5]
    );
}

#[test]
fn inert_terminal_retains_duplicate_and_wildcard_returning_projection_order() {
    const TXN: TxnId = 99_005;
    let engine = Engine::new_local();
    engine
        .execute_text(
            99_005_000,
            "CREATE TABLE terminal_returning_table (id int4, payload int4, extra int8)",
        )
        .unwrap();
    let insert = Insert {
        table: "terminal_returning_table".to_string(),
        columns: vec!["id".to_string(), "payload".to_string(), "extra".to_string()],
        rows: Insert::programmatic_rows(vec![vec![
            SqlValue::Int4(1),
            SqlValue::Int4(7),
            SqlValue::Int8(8),
        ]]),
        returning: vec![
            "payload".to_string(),
            PROJECTION_WILDCARD_SENTINEL.to_string(),
            "payload".to_string(),
        ],
    };
    let plan =
        super::super::PreparedInsertEffectPlan::prepare_autocommit_for_test(&engine, TXN, &insert)
            .unwrap();
    let evidence = plan
        .seal_for_test(&engine, SequenceReceiptBundle::for_test(Vec::new()))
        .unwrap();
    assert_eq!(evidence.returning.row_count, 1);
    assert_eq!(evidence.returning.column_count, 5);
    assert_eq!(evidence.returning.cell_count, 5);
    assert_eq!(
        evidence
            .returning
            .projections
            .iter()
            .map(|projection| projection.name.as_ref())
            .collect::<Vec<_>>(),
        ["payload", "id", "payload", "extra", "payload"]
    );
    assert_ne!(evidence.returning.projection_digest, [0; 32]);
    assert_eq!(evidence.returning.projections[3].ty, crate::SqlType::Int8);
}

fn two_published_plan() -> (Engine, super::super::PreparedInsertEffectPlan) {
    const TXN: TxnId = 99_006;
    let engine = Engine::new_local();
    engine
        .execute_text(99_006_000, "CREATE SEQUENCE terminal_two_a")
        .unwrap();
    engine
        .execute_text(99_006_001, "CREATE SEQUENCE terminal_two_b")
        .unwrap();
    engine
        .execute_text(
            99_006_002,
            "CREATE TABLE terminal_two_table (a int4 DEFAULT nextval('terminal_two_a'::regclass), b int4 DEFAULT nextval('terminal_two_b'::regclass))",
        )
        .unwrap();
    let plan = super::super::PreparedInsertEffectPlan::prepare_autocommit_for_test(
        &engine,
        TXN,
        &parsed_insert("INSERT INTO terminal_two_table (a, b) VALUES (DEFAULT, DEFAULT)"),
    )
    .unwrap();
    (engine, plan)
}

fn private_plan(
    txn_id: TxnId,
) -> (
    Engine,
    Arc<TransactionSnapshot>,
    super::super::PreparedInsertEffectPlan,
) {
    let engine = Engine::new_local();
    engine.execute_text(txn_id, "BEGIN").unwrap();
    create_sequence_and_table(
        &engine,
        txn_id,
        "terminal_sabotage",
        "terminal_sabotage_table",
    );
    let snapshot = engine.transaction_snapshot_handle(txn_id).unwrap();
    let plan = super::super::PreparedInsertEffectPlan::prepare_explicit_for_test(
        &engine,
        txn_id,
        &parsed_insert("INSERT INTO terminal_sabotage_table (id) VALUES (DEFAULT)"),
    )
    .unwrap();
    (engine, snapshot, plan)
}

#[test]
fn inert_terminal_rejects_parent_count_order_and_target_sabotage_before_seal() {
    for sabotage in [
        TerminalPlanSabotage::EffectCount,
        TerminalPlanSabotage::EffectOrder,
        TerminalPlanSabotage::Target,
    ] {
        let (engine, mut plan) = two_published_plan();
        let receipts = published_receipts(&plan, &[11, 12]);
        plan.sabotage_classification_for_terminal_test(sabotage);
        let before = terminal_fingerprint(&engine, 99_006, None);
        assert!(plan.seal_for_test(&engine, receipts).is_err());
        assert_eq!(terminal_fingerprint(&engine, 99_006, None), before);
    }

    let (engine, mut plan) = two_published_plan();
    let receipts = published_receipts(&plan, &[11, 12]);
    plan.sabotage_parent_for_terminal_test();
    let before = terminal_fingerprint(&engine, 99_006, None);
    assert!(plan.seal_for_test(&engine, receipts).is_err());
    assert_eq!(terminal_fingerprint(&engine, 99_006, None), before);
}

#[test]
fn inert_terminal_rejects_published_receipt_value_identity_input_and_order_sabotage() {
    for sabotage in [
        "missing",
        "trailing",
        "value",
        "transition",
        "input",
        "order",
    ] {
        let (engine, plan) = two_published_plan();
        let digests = plan.published_receipt_input_digests_for_test();
        let mut inputs = vec![
            PublishedReceiptInput::for_test(77, 11, digests[0]),
            PublishedReceiptInput::for_test(78, 12, digests[1]),
        ];
        match sabotage {
            "missing" => {
                inputs.pop();
            }
            "trailing" => inputs.push(PublishedReceiptInput::for_test(79, 13, [8; 32])),
            "value" => inputs[0].returned_value = i64::from(i32::MAX) + 1,
            "transition" => inputs[1].transition_txn_id = inputs[0].transition_txn_id,
            "input" => inputs[0].input_digest = [0; 32],
            "order" => inputs.swap(0, 1),
            _ => unreachable!(),
        }
        let before = terminal_fingerprint(&engine, 99_006, None);
        assert!(
            plan.seal_for_test(&engine, SequenceReceiptBundle::for_test(inputs))
                .is_err(),
            "{sabotage} receipt must reject before materialization"
        );
        assert_eq!(terminal_fingerprint(&engine, 99_006, None), before);
    }
}

#[test]
fn inert_terminal_rejects_private_descriptor_child_outcome_owner_and_predecessor_sabotage() {
    for sabotage in [
        TerminalPlanSabotage::Descriptor,
        TerminalPlanSabotage::Child,
        TerminalPlanSabotage::Outcome,
        TerminalPlanSabotage::Owner,
        TerminalPlanSabotage::Predecessor,
        TerminalPlanSabotage::PrivateValue,
        TerminalPlanSabotage::LifetimeOrigin,
        TerminalPlanSabotage::PrivateInput,
    ] {
        const TXN: TxnId = 99_007;
        let (engine, snapshot, mut plan) = private_plan(TXN);
        plan.sabotage_classification_for_terminal_test(sabotage);
        let before = terminal_fingerprint(&engine, TXN, Some(&snapshot));
        assert!(plan
            .seal_for_test(&engine, SequenceReceiptBundle::for_test(Vec::new()))
            .is_err());
        assert_eq!(terminal_fingerprint(&engine, TXN, Some(&snapshot)), before);
    }
}

#[test]
fn inert_terminal_rejects_rehashed_impossible_lifetime_owner_provenance() {
    for sabotage in [
        TerminalPlanSabotage::RehashedPublishedCreateOwner,
        TerminalPlanSabotage::RehashedNonCreateOwnerOrdinal,
    ] {
        const TXN: TxnId = 99_017;
        let (engine, snapshot, mut plan) = private_plan(TXN);
        plan.sabotage_classification_for_terminal_test(sabotage);
        let before = terminal_fingerprint(&engine, TXN, Some(&snapshot));
        assert!(
            plan.seal_for_test(&engine, SequenceReceiptBundle::for_test(Vec::new()))
                .is_err(),
            "{sabotage:?} must reject even after its child and outcome witnesses are rehashed"
        );
        assert_eq!(terminal_fingerprint(&engine, TXN, Some(&snapshot)), before);
    }
}

#[test]
fn inert_terminal_revalidates_catalog_and_explicit_snapshot_generations() {
    let (engine, plan) = two_published_plan();
    let receipts = published_receipts(&plan, &[11, 12]);
    engine
        .execute_text(99_006_003, "CREATE TABLE terminal_drift (id int4)")
        .unwrap();
    let before = terminal_fingerprint(&engine, 99_006, None);
    assert!(plan.seal_for_test(&engine, receipts).is_err());
    assert_eq!(terminal_fingerprint(&engine, 99_006, None), before);

    const TXN: TxnId = 99_008;
    let (engine, snapshot, plan) = private_plan(TXN);
    {
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _statement_guard = statement_lock.lock().unwrap();
        snapshot.delta.lock().unwrap().generation += 1;
    }
    let before = terminal_fingerprint(&engine, TXN, Some(&snapshot));
    assert!(plan
        .seal_for_test(&engine, SequenceReceiptBundle::for_test(Vec::new()))
        .is_err());
    assert_eq!(terminal_fingerprint(&engine, TXN, Some(&snapshot)), before);

    const OVERLAY_TXN: TxnId = 99_009;
    let (engine, snapshot, plan) = private_plan(OVERLAY_TXN);
    {
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _statement_guard = statement_lock.lock().unwrap();
        let mut delta = snapshot.delta.lock().unwrap();
        let overlay = delta
            .catalog_overlay
            .as_ref()
            .expect("private catalog fixture has an overlay");
        delta.catalog_overlay = Some(Arc::new(overlay.as_ref().clone()));
    }
    let before = terminal_fingerprint(&engine, OVERLAY_TXN, Some(&snapshot));
    assert!(plan
        .seal_for_test(&engine, SequenceReceiptBundle::for_test(Vec::new()))
        .is_err());
    assert_eq!(
        terminal_fingerprint(&engine, OVERLAY_TXN, Some(&snapshot)),
        before
    );

    const REPLACED_TXN: TxnId = 99_010;
    let (engine, snapshot, plan) = private_plan(REPLACED_TXN);
    let replacement =
        engine.capture_transaction_snapshot(snapshot.boundary, snapshot.characteristics);
    engine
        .active_snapshots
        .lock()
        .unwrap()
        .replace_transaction_snapshot(REPLACED_TXN, &snapshot, replacement, false)
        .unwrap();
    let before = terminal_fingerprint(&engine, REPLACED_TXN, None);
    assert!(plan
        .seal_for_test(&engine, SequenceReceiptBundle::for_test(Vec::new()))
        .is_err());
    assert_eq!(terminal_fingerprint(&engine, REPLACED_TXN, None), before);
}

#[test]
fn inert_terminal_rejects_an_autocommit_plan_inside_transaction_read_tls() {
    let (engine, plan) = two_published_plan();
    let receipts = published_receipts(&plan, &[11, 12]);
    let scoped_snapshot = engine.capture_statement_snapshot(engine.committed_seq());
    let _scope = engine.enter_transaction_read(scoped_snapshot);
    let before = terminal_fingerprint(&engine, 99_006, None);
    assert!(plan.seal_for_test(&engine, receipts).is_err());
    assert_eq!(terminal_fingerprint(&engine, 99_006, None), before);
}

#[test]
fn inert_terminal_is_move_only_and_has_no_live_authority() {
    let terminal_source = include_str!("terminal.rs");
    let receipt_source = include_str!("receipt.rs");
    assert!(terminal_source.contains("plan: PreparedInsertEffectPlan"));
    assert!(terminal_source.contains("let PreparedInsertEffectPlan { prepared, .. } = plan"));
    assert!(terminal_source.contains("commit_state_after_wave_quiescence"));
    assert!(receipt_source.contains("only structural binding"));
    for forbidden in [
        "PreparedDeviceInsertPlan",
        "from_typed_batch",
        "PreparedBinaryInsertTemplate",
        "DeviceInsertPlan",
        "append_canonical",
        "WalBuffer",
        "commit_sequence_value_transition",
        "resolve_sequence_value_transition_retry",
        "allocate_transaction_id",
        "sequence_currval_effects",
        "apply_and_publish",
        "bind_sequence_default_insert_rows",
        "materialize_published_sequence_defaults",
    ] {
        assert!(
            !terminal_source.contains(forbidden) && !receipt_source.contains(forbidden),
            "inert receipt terminal must not own {forbidden}"
        );
    }
}
