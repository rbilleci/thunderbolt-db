use super::*;
use crate::{
    parse_command, transaction_statement_digest, BinarySequenceValueReference, Command, Engine,
    Insert, PreparedMutation, StagedRowOperation, TransactionOperation, TransactionSnapshot, TxnId,
    WriteDelta, WriteSet,
};
use gpu_db_sql::ParsedCommand;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

fn parsed_insert(sql: &str) -> Insert {
    let Command::Insert(insert) = parse_command(sql).expect("INSERT parses") else {
        panic!("fixture SQL must parse as INSERT");
    };
    insert
}

fn explicit_plan(
    engine: &Engine,
    txn_id: TxnId,
    insert: &Insert,
) -> super::super::PreparedInsertEffectPlan {
    super::super::PreparedInsertEffectPlan::prepare_explicit_for_test(engine, txn_id, insert)
        .expect("effect plan classifies")
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

fn stage_synthetic_private_row_advance(
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

#[derive(Debug, PartialEq, Eq)]
struct FailedClassificationFingerprint {
    generation: u64,
    operations: String,
    references: String,
    sequence_state_by_oid: BTreeMap<u32, (i64, bool)>,
    sequence_state: BTreeMap<String, (i64, bool)>,
    catalog_base: Option<usize>,
    catalog_overlay: Option<usize>,
    resident_strong_count: usize,
    cold_strong_count: usize,
    private_gpu_bytes_by_gpu: BTreeMap<u16, u64>,
    commit_gpu_bytes_by_gpu: BTreeMap<u16, u64>,
}

fn failed_classification_fingerprint(
    snapshot: &TransactionSnapshot,
) -> FailedClassificationFingerprint {
    let statement_lock = Arc::clone(&snapshot.statement_lock);
    let _statement_guard = statement_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let delta = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    FailedClassificationFingerprint {
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
        resident_strong_count: Arc::strong_count(&delta.resident_shards),
        cold_strong_count: Arc::strong_count(&delta.streaming_cold_chunks),
        private_gpu_bytes_by_gpu: delta.private_gpu_bytes_by_gpu.clone(),
        commit_gpu_bytes_by_gpu: delta.commit_gpu_bytes_by_gpu.clone(),
    }
}

fn assert_failed_prepare_preserves(
    engine: &Engine,
    txn_id: TxnId,
    snapshot: &TransactionSnapshot,
    insert: &Insert,
) {
    let before = failed_classification_fingerprint(snapshot);
    assert!(
        super::super::PreparedInsertEffectPlan::prepare_explicit_for_test(engine, txn_id, insert)
            .is_err(),
        "sabotaged classifier fixture must reject before a live owner can consume it"
    );
    assert_eq!(failed_classification_fingerprint(snapshot), before);
}

#[test]
fn classifier_autocommit_published_effect_has_no_private_value() {
    let engine = Engine::new_local();
    engine
        .execute_text(98_000, "CREATE SEQUENCE classifier_published")
        .unwrap();
    engine
        .execute_text(
            98_000_001,
            "CREATE TABLE classifier_published_table (id int4 DEFAULT nextval('classifier_published'::regclass))",
        )
        .unwrap();
    let insert = parsed_insert("INSERT INTO classifier_published_table (id) VALUES (DEFAULT)");
    let plan = super::super::PreparedInsertEffectPlan::prepare_autocommit_for_test(
        &engine, 98_001, &insert,
    )
    .unwrap();
    let effects = plan.sequence_effects_for_test();
    assert!(effects.parent.autocommit);
    assert!(matches!(
        effects.effects.as_ref(),
        [PlannedSequenceEffect::Published { .. }]
    ));
}

#[test]
fn classifier_explicit_create_sequence_plans_private_i32_value() {
    const TXN: TxnId = 98_002;
    let engine = Engine::new_local();
    engine.execute_text(TXN, "BEGIN").unwrap();
    engine
        .submit_transaction(
            TXN,
            ParsedCommand::parse("CREATE SEQUENCE classifier_private").unwrap(),
        )
        .unwrap();
    engine
        .submit_transaction(
            TXN,
            ParsedCommand::parse(
                "CREATE TABLE classifier_private_table (id int4 DEFAULT nextval('classifier_private'::regclass))",
            )
            .unwrap(),
        )
        .unwrap();
    let insert = parsed_insert("INSERT INTO classifier_private_table (id) VALUES (DEFAULT)");
    let plan = explicit_plan(&engine, TXN, &insert);
    let effects = plan.sequence_effects_for_test();
    let [PlannedSequenceEffect::Private(effect)] = effects.effects.as_ref() else {
        panic!("private CREATE SEQUENCE must plan one private effect");
    };
    assert_eq!(effect.prior_state, (1, false));
    assert_eq!(effect.next_state, (1, true));
    assert_eq!(effect.output_i32, 1);
    assert_eq!(effect.prior_owner.kind, PrivateValueOwner::Create);
    assert_ne!(effect.planned_child_digest, [0; 32]);
    assert_ne!(effect.planned_outcome_digest, [0; 32]);
}

#[test]
fn classifier_implicit_serial_owner_keeps_creator_column_ordinal() {
    const TXN: TxnId = 98_003;
    let engine = Engine::new_local();
    engine.execute_text(TXN, "BEGIN").unwrap();
    stage(
        &engine,
        TXN,
        "CREATE TABLE classifier_implicit (payload int4, id serial)",
    );
    let plan = explicit_plan(
        &engine,
        TXN,
        &parsed_insert("INSERT INTO classifier_implicit (id, payload) VALUES (DEFAULT, 7)"),
    );
    let [PlannedSequenceEffect::Private(effect)] =
        plan.sequence_effects_for_test().effects.as_ref()
    else {
        panic!("implicit serial must become a private planned effect");
    };
    assert_eq!(effect.prior_owner.kind, PrivateValueOwner::Create);
    assert_eq!(effect.prior_owner.creator_catalog_column_ordinal, Some(1));
}

#[test]
fn classifier_create_table_existing_sequence_is_published_even_with_duplicate_defaults() {
    const TXN: TxnId = 98_004;
    let engine = Engine::new_local();
    engine
        .execute_text(98_004_000, "CREATE SEQUENCE classifier_existing")
        .unwrap();
    engine.execute_text(TXN, "BEGIN").unwrap();
    stage(
        &engine,
        TXN,
        "CREATE TABLE classifier_existing_table (a int4 DEFAULT nextval('classifier_existing'::regclass), b int4 DEFAULT nextval('classifier_existing'::regclass))",
    );
    let plan = explicit_plan(
        &engine,
        TXN,
        &parsed_insert("INSERT INTO classifier_existing_table (a, b) VALUES (DEFAULT, DEFAULT)"),
    );
    assert!(plan
        .sequence_effects_for_test()
        .effects
        .iter()
        .all(|effect| matches!(effect, PlannedSequenceEffect::Published { .. })));
}

#[test]
fn classifier_restart_then_row_advance_preserves_restart_owner_and_chains_outcomes() {
    const TXN: TxnId = 98_005;
    let engine = Engine::new_local();
    engine.execute_text(TXN, "BEGIN").unwrap();
    create_sequence_and_table(
        &engine,
        TXN,
        "classifier_restart",
        "classifier_restart_table",
    );
    stage(
        &engine,
        TXN,
        "ALTER SEQUENCE classifier_restart RESTART WITH 40",
    );
    stage_synthetic_private_row_advance(
        &engine,
        TXN,
        "classifier_restart",
        "classifier_restart_table",
        (41, true),
    );
    let plan = explicit_plan(
        &engine,
        TXN,
        &parsed_insert(
            "INSERT INTO classifier_restart_table (id, payload) VALUES (DEFAULT, 3), (DEFAULT, 4)",
        ),
    );
    let effects = plan.sequence_effects_for_test();
    let [PlannedSequenceEffect::Private(first), PlannedSequenceEffect::Private(second)] =
        effects.effects.as_ref()
    else {
        panic!("two defaults must produce a private chain");
    };
    assert_eq!(first.prior_state, (41, true));
    assert_eq!(first.output_i32, 42);
    assert_eq!(first.prior_owner.kind, PrivateValueOwner::Restart);
    assert_eq!(first.prior_owner.statement_ordinal, 2);
    assert!(matches!(
        first.predecessor,
        PlannedPrivatePredecessor::Lifecycle(owner) if owner == first.prior_owner
    ));
    assert_eq!(second.prior_state, first.next_state);
    assert_eq!(second.output_i32, 43);
    assert_eq!(second.prior_owner, first.prior_owner);
    assert!(matches!(
        second.predecessor,
        PlannedPrivatePredecessor::PlannedOutcome(digest) if digest == first.planned_outcome_digest
    ));
}

#[test]
fn classifier_preserves_published_and_private_lifetime_across_rename() {
    const PUBLISHED_TXN: TxnId = 98_006;
    let engine = Engine::new_local();
    engine
        .execute_text(98_006_000, "CREATE SEQUENCE classifier_rename_published")
        .unwrap();
    engine
        .execute_text(
            98_006_001,
            "CREATE TABLE classifier_rename_published_table (id int4 DEFAULT nextval('classifier_rename_published'::regclass))",
        )
        .unwrap();
    engine.execute_text(PUBLISHED_TXN, "BEGIN").unwrap();
    stage(
        &engine,
        PUBLISHED_TXN,
        "ALTER SEQUENCE classifier_rename_published RENAME TO classifier_rename_published_new",
    );
    let published = explicit_plan(
        &engine,
        PUBLISHED_TXN,
        &parsed_insert("INSERT INTO classifier_rename_published_table (id) VALUES (DEFAULT)"),
    );
    let [PlannedSequenceEffect::Published { target }] =
        published.sequence_effects_for_test().effects.as_ref()
    else {
        panic!("renamed published sequence remains published");
    };
    assert_eq!(
        target.effective_name.as_ref(),
        "classifier_rename_published_new"
    );

    const PRIVATE_TXN: TxnId = 98_007;
    let engine = Engine::new_local();
    engine.execute_text(PRIVATE_TXN, "BEGIN").unwrap();
    create_sequence_and_table(
        &engine,
        PRIVATE_TXN,
        "classifier_rename_private",
        "classifier_rename_private_table",
    );
    stage(
        &engine,
        PRIVATE_TXN,
        "ALTER SEQUENCE classifier_rename_private RENAME TO classifier_rename_private_new",
    );
    let private = explicit_plan(
        &engine,
        PRIVATE_TXN,
        &parsed_insert("INSERT INTO classifier_rename_private_table (id) VALUES (DEFAULT)"),
    );
    let [PlannedSequenceEffect::Private(effect)] =
        private.sequence_effects_for_test().effects.as_ref()
    else {
        panic!("renamed private sequence remains private");
    };
    assert_eq!(effect.lifetime_origin, SequenceLifetimeOrigin::Private);
    assert_eq!(
        effect.target.effective_name.as_ref(),
        "classifier_rename_private_new"
    );
    assert_eq!(effect.prior_owner.kind, PrivateValueOwner::Create);
}

#[test]
fn classifier_truncate_restart_and_drop_recreate_use_stable_oid_history() {
    const RESET_TXN: TxnId = 98_008;
    let engine = Engine::new_local();
    engine
        .execute_text(
            98_008_000,
            "CREATE TABLE classifier_reset_table (id serial)",
        )
        .unwrap();
    engine.execute_text(RESET_TXN, "BEGIN").unwrap();
    stage(
        &engine,
        RESET_TXN,
        "TRUNCATE classifier_reset_table RESTART IDENTITY",
    );
    let reset = explicit_plan(
        &engine,
        RESET_TXN,
        &parsed_insert("INSERT INTO classifier_reset_table (id) VALUES (DEFAULT)"),
    );
    let [PlannedSequenceEffect::Private(effect)] =
        reset.sequence_effects_for_test().effects.as_ref()
    else {
        panic!("TRUNCATE RESTART IDENTITY establishes private state");
    };
    assert_eq!(effect.lifetime_origin, SequenceLifetimeOrigin::Published);
    assert_eq!(effect.prior_owner.kind, PrivateValueOwner::TruncateRestart);

    const ABA_TXN: TxnId = 98_009;
    let engine = Engine::new_local();
    engine
        .execute_text(98_009_000, "CREATE SEQUENCE classifier_aba")
        .unwrap();
    let old_oid = engine.catalog_snapshot().relational_sequences["classifier_aba"].oid;
    engine.execute_text(ABA_TXN, "BEGIN").unwrap();
    stage(&engine, ABA_TXN, "DROP SEQUENCE classifier_aba");
    stage(&engine, ABA_TXN, "CREATE SEQUENCE classifier_aba");
    stage(
        &engine,
        ABA_TXN,
        "CREATE TABLE classifier_aba_table (id int4 DEFAULT nextval('classifier_aba'::regclass))",
    );
    let aba = explicit_plan(
        &engine,
        ABA_TXN,
        &parsed_insert("INSERT INTO classifier_aba_table (id) VALUES (DEFAULT)"),
    );
    let [PlannedSequenceEffect::Private(effect)] = aba.sequence_effects_for_test().effects.as_ref()
    else {
        panic!("drop/recreate must use the new private OID");
    };
    assert_ne!(effect.target.sequence_oid, old_oid);
}

#[test]
fn classifier_allows_absent_drop_if_exists_without_changing_the_fold() {
    const TXN: TxnId = 98_010;
    let engine = Engine::new_local();
    engine
        .execute_text(98_010_000, "CREATE TABLE classifier_drop_noop (id int4)")
        .unwrap();
    engine.execute_text(TXN, "BEGIN").unwrap();
    stage(&engine, TXN, "DROP SEQUENCE IF EXISTS classifier_absent");
    let plan = explicit_plan(
        &engine,
        TXN,
        &parsed_insert("INSERT INTO classifier_drop_noop (id) VALUES (1)"),
    );
    assert!(plan.sequence_effects_for_test().effects.is_empty());
}

#[test]
fn classifier_rejects_i32_and_reachability_drift_without_mutating_state() {
    const I32_TXN: TxnId = 98_011;
    let engine = Engine::new_local();
    engine.execute_text(I32_TXN, "BEGIN").unwrap();
    create_sequence_and_table(
        &engine,
        I32_TXN,
        "classifier_i32_overflow",
        "classifier_i32_overflow_table",
    );
    stage(
        &engine,
        I32_TXN,
        "ALTER SEQUENCE classifier_i32_overflow RESTART WITH 2147483648",
    );
    assert!(
        super::super::PreparedInsertEffectPlan::prepare_explicit_for_test(
            &engine,
            I32_TXN,
            &parsed_insert("INSERT INTO classifier_i32_overflow_table (id) VALUES (DEFAULT)"),
        )
        .is_err()
    );

    const REACH_TXN: TxnId = 98_012;
    let engine = Engine::new_local();
    engine.execute_text(REACH_TXN, "BEGIN").unwrap();
    create_sequence_and_table(
        &engine,
        REACH_TXN,
        "classifier_unreachable",
        "classifier_unreachable_table",
    );
    stage_synthetic_private_row_advance(
        &engine,
        REACH_TXN,
        "classifier_unreachable",
        "classifier_unreachable_table",
        (1, false),
    );
    assert!(
        super::super::PreparedInsertEffectPlan::prepare_explicit_for_test(
            &engine,
            REACH_TXN,
            &parsed_insert("INSERT INTO classifier_unreachable_table (id) VALUES (DEFAULT)"),
        )
        .is_err()
    );
}

#[test]
fn classifier_uses_nonzero_expression_base_with_sparse_local_ordinals() {
    const TXN: TxnId = 98_013;
    let engine = Engine::new_local();
    engine.execute_text(TXN, "BEGIN").unwrap();
    stage(&engine, TXN, "CREATE SEQUENCE classifier_hole_a");
    stage(&engine, TXN, "CREATE SEQUENCE classifier_hole_b");
    stage(
        &engine,
        TXN,
        "CREATE TABLE classifier_hole_table (a int4 DEFAULT nextval('classifier_hole_a'::regclass), b int4 DEFAULT nextval('classifier_hole_b'::regclass), payload int4)",
    );
    let snapshot = engine.transaction_snapshot_handle(TXN).unwrap();
    {
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _statement_guard = statement_lock.lock().unwrap();
        let mut delta = snapshot.delta.lock().unwrap();
        let statement_ordinal = u32::try_from(delta.operations.len()).unwrap();
        delta
            .sequence_value_references
            .push(synthetic_sequence_reference(statement_ordinal, 0));
        delta
            .sequence_value_references
            .push(synthetic_sequence_reference(statement_ordinal, 1));
    }
    let plan = explicit_plan(
        &engine,
        TXN,
        &parsed_insert(
            "INSERT INTO classifier_hole_table (a, b, payload) VALUES (DEFAULT, 7, 1), (8, DEFAULT, 2)",
        ),
    );
    let effects = plan.sequence_effects_for_test();
    assert_eq!(effects.parent.expression_ordinal_base, 2);
    let local_and_absolute = effects
        .effects
        .iter()
        .map(|effect| match effect {
            PlannedSequenceEffect::Private(effect) => (
                effect.target.local_expression_ordinal,
                effect.target.absolute_expression_ordinal,
                effect.target.input_digest,
            ),
            PlannedSequenceEffect::Published { .. } => {
                panic!("transaction-created sequences must retain private planned state")
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(
        local_and_absolute
            .iter()
            .map(|(local, _, _)| *local)
            .collect::<Vec<_>>(),
        [0, 3],
        "only active/defaulted slots become requests; the intervening local slots stay holes"
    );
    assert_eq!(
        local_and_absolute
            .iter()
            .map(|(_, absolute, _)| *absolute)
            .collect::<Vec<_>>(),
        [2, 5]
    );
    assert_ne!(local_and_absolute[0].2, [0; 32]);
    assert_ne!(local_and_absolute[0].2, local_and_absolute[1].2);
}

#[test]
fn classifier_rejects_i64_checked_add_overflow_without_consuming_the_delta() {
    const TXN: TxnId = 98_014;
    let engine = Engine::new_local();
    engine.execute_text(TXN, "BEGIN").unwrap();
    create_sequence_and_table(
        &engine,
        TXN,
        "classifier_i64_overflow",
        "classifier_i64_overflow_table",
    );
    stage(
        &engine,
        TXN,
        "ALTER SEQUENCE classifier_i64_overflow RESTART WITH 9223372036854775806",
    );
    stage_synthetic_private_row_advance(
        &engine,
        TXN,
        "classifier_i64_overflow",
        "classifier_i64_overflow_table",
        (i64::MAX, true),
    );
    let snapshot = engine.transaction_snapshot_handle(TXN).unwrap();
    assert_failed_prepare_preserves(
        &engine,
        TXN,
        &snapshot,
        &parsed_insert("INSERT INTO classifier_i64_overflow_table (id) VALUES (DEFAULT)"),
    );
}

#[test]
fn classifier_rejects_private_oid_name_and_dual_map_mismatches_without_mutating_them() {
    for mismatch in ["oid", "name", "dual"] {
        const TXN: TxnId = 98_015;
        let engine = Engine::new_local();
        engine.execute_text(TXN, "BEGIN").unwrap();
        create_sequence_and_table(
            &engine,
            TXN,
            "classifier_map_state",
            "classifier_map_state_table",
        );
        let snapshot = engine.transaction_snapshot_handle(TXN).unwrap();
        {
            let statement_lock = Arc::clone(&snapshot.statement_lock);
            let _statement_guard = statement_lock.lock().unwrap();
            let oid =
                snapshot.transaction_catalog().relational_sequences["classifier_map_state"].oid;
            let mut delta = snapshot.delta.lock().unwrap();
            match mismatch {
                "oid" => {
                    delta.sequence_state_by_oid.insert(oid, (2, true));
                }
                "name" => {
                    delta
                        .sequence_state
                        .insert("classifier_map_state".to_string(), (2, true));
                }
                "dual" => {
                    delta.sequence_state_by_oid.insert(oid, (2, true));
                    delta
                        .sequence_state
                        .insert("classifier_map_state".to_string(), (2, true));
                }
                _ => unreachable!(),
            }
        }
        assert_failed_prepare_preserves(
            &engine,
            TXN,
            &snapshot,
            &parsed_insert("INSERT INTO classifier_map_state_table (id) VALUES (DEFAULT)"),
        );
    }
}

#[test]
fn classifier_rejects_catalog_lifecycle_ordinal_and_command_index_sabotage() {
    for sabotage in ["ordinal", "command-index"] {
        const TXN: TxnId = 98_016;
        let engine = Engine::new_local();
        engine.execute_text(TXN, "BEGIN").unwrap();
        create_sequence_and_table(
            &engine,
            TXN,
            "classifier_lifecycle",
            "classifier_lifecycle_table",
        );
        let snapshot = engine.transaction_snapshot_handle(TXN).unwrap();
        {
            let statement_lock = Arc::clone(&snapshot.statement_lock);
            let _statement_guard = statement_lock.lock().unwrap();
            let mut delta = snapshot.delta.lock().unwrap();
            let TransactionOperation::Catalog(staged) = &mut delta.operations[0] else {
                panic!("fixture begins with CREATE SEQUENCE catalog operation");
            };
            let staged = Arc::make_mut(staged);
            match sabotage {
                "ordinal" => staged.ordinal = staged.ordinal.saturating_add(1),
                "command-index" => {
                    staged
                        .sequence_identity
                        .as_mut()
                        .expect("CREATE SEQUENCE has lifecycle identity")
                        .command_index = 1;
                }
                _ => unreachable!(),
            }
        }
        assert_failed_prepare_preserves(
            &engine,
            TXN,
            &snapshot,
            &parsed_insert("INSERT INTO classifier_lifecycle_table (id) VALUES (DEFAULT)"),
        );
    }
}

#[test]
fn classifier_rejects_reset_table_and_ordinal_sabotage() {
    for sabotage in ["table", "ordinal"] {
        const TXN: TxnId = 98_017;
        let engine = Engine::new_local();
        engine
            .execute_text(
                98_017_000,
                "CREATE TABLE classifier_reset_sabotage (id serial)",
            )
            .unwrap();
        engine.execute_text(TXN, "BEGIN").unwrap();
        stage(
            &engine,
            TXN,
            "TRUNCATE classifier_reset_sabotage RESTART IDENTITY",
        );
        let snapshot = engine.transaction_snapshot_handle(TXN).unwrap();
        {
            let statement_lock = Arc::clone(&snapshot.statement_lock);
            let _statement_guard = statement_lock.lock().unwrap();
            let mut delta = snapshot.delta.lock().unwrap();
            let TransactionOperation::TableReset(reset) = &mut delta.operations[0] else {
                panic!("fixture contains one table reset operation");
            };
            let reset = Arc::make_mut(reset);
            let identity = reset
                .sequence_reset_identity
                .as_mut()
                .expect("RESTART IDENTITY retains reset identity");
            match sabotage {
                "table" => identity.table = "classifier_reset_other".to_string(),
                "ordinal" => identity.ordinal = identity.ordinal.saturating_add(1),
                _ => unreachable!(),
            }
        }
        assert_failed_prepare_preserves(
            &engine,
            TXN,
            &snapshot,
            &parsed_insert("INSERT INTO classifier_reset_sabotage (id) VALUES (DEFAULT)"),
        );
    }
}

#[test]
fn classifier_rejects_a_complete_fold_that_disagrees_with_the_captured_overlay() {
    const TXN: TxnId = 98_018;
    let engine = Engine::new_local();
    engine.execute_text(TXN, "BEGIN").unwrap();
    stage(&engine, TXN, "CREATE SEQUENCE classifier_overlay_unused");
    create_sequence_and_table(
        &engine,
        TXN,
        "classifier_overlay_used",
        "classifier_overlay_used_table",
    );
    let snapshot = engine.transaction_snapshot_handle(TXN).unwrap();
    {
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _statement_guard = statement_lock.lock().unwrap();
        let mut delta = snapshot.delta.lock().unwrap();
        let mut overlay = delta
            .catalog_overlay
            .as_ref()
            .expect("catalog operations retain a transaction overlay")
            .as_ref()
            .clone();
        overlay
            .relational_sequences
            .remove("classifier_overlay_unused");
        delta.catalog_overlay = Some(Arc::new(overlay));
    }
    assert_failed_prepare_preserves(
        &engine,
        TXN,
        &snapshot,
        &parsed_insert("INSERT INTO classifier_overlay_used_table (id) VALUES (DEFAULT)"),
    );
}

#[test]
fn classifier_private_child_digest_binds_sequence_lifetime_origin() {
    let parent = PlannedSequenceParent {
        txn_id: 98_019,
        autocommit: false,
        request_digest: [3; 32],
        statement_ordinal: crate::insert_semantic_ir::InsertStatementOrdinal::FIRST,
        expression_ordinal_base: 0,
    };
    let target = PlannedSequenceTarget {
        target_table_oid: 11,
        row_ordinal: 0,
        catalog_column_ordinal: 0,
        column_id: 12,
        sequence_oid: 13,
        source_name: "classifier_origin".into(),
        effective_name: "classifier_origin".into(),
        statement_ordinal: crate::insert_semantic_ir::InsertStatementOrdinal::FIRST,
        local_expression_ordinal: 0,
        absolute_expression_ordinal: 0,
        input_digest: [4; 32],
        descriptor_digest: [5; 32],
    };
    let owner = PrivateValueOwnerIdentity {
        kind: PrivateValueOwner::Create,
        statement_ordinal: 0,
        statement_digest: [6; 32],
        creator_catalog_column_ordinal: None,
    };
    let predecessor = PlannedPrivatePredecessor::Lifecycle(owner);
    let published = planned_private_child_digest(
        &parent,
        &target,
        SequenceLifetimeOrigin::Published,
        (1, false),
        predecessor,
    );
    let private = planned_private_child_digest(
        &parent,
        &target,
        SequenceLifetimeOrigin::Private,
        (1, false),
        predecessor,
    );
    assert_ne!(
        published, private,
        "same parent, target, state, and predecessor must not collide across sequence lifetimes"
    );
}

#[test]
fn classifier_leaf_has_no_execution_or_durability_authority() {
    let source = include_str!("classification.rs");
    for forbidden in [
        "WalBuffer",
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
        "sequence_currval_effects",
        "reserve_gpu",
        "reserve_capacity",
        "apply_and_publish",
    ] {
        assert!(
            !source.contains(forbidden),
            "classifier must not own {forbidden}"
        );
    }
    assert!(source.contains("GPUDBPRIVATESEQCHILD1"));
    assert!(source.contains("GPUDBPRIVATESEQOUTCOME1"));
}
