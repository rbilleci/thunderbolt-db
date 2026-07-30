use super::ForeignKeyConstraintProof;

#[test]
fn foreign_key_proof_keeps_the_gpu_generation_and_resource_contract_production_compiled() {
    let source = include_str!("foreign_key_constraints.rs");
    let production = source
        .split("\n#[cfg(test)]\n#[path = \"foreign_key_constraints_tests.rs\"]")
        .next()
        .expect("FK production leaf precedes its separate tests");

    assert!(production.contains("pub(super) fn validate_current_generation"));
    assert!(!production.contains("#[cfg(test)]\n#[allow(clippy::too_many_arguments)]\npub(super) fn validate_current_generation"));
    assert!(production.contains("_proof: ForeignKeyConstraintProof"));
    assert!(production.contains("_parents: Box<[ForeignKeyParentGenerationEvidence]>"));
    assert!(production.contains("payload: Arc<gpu_db_execution::CudaResidentDeviceMemory>"));
    assert!(
        production.contains("created_by: Option<Arc<gpu_db_execution::CudaResidentDeviceMemory>>")
    );
    assert!(
        production.contains("deleted_by: Option<Arc<gpu_db_execution::CudaResidentDeviceMemory>>")
    );
    assert!(production.contains("generation: Arc<()>"));
    assert!(production.contains("let chunk_authoritative = engine"));
    assert!(production.contains("let cold_chunks = engine.read_streaming_cold_chunks()"));
    assert_eq!(
        production
            .matches("chunk_authoritative_tables\n        .load()")
            .count(),
        1
    );
    assert_eq!(
        production.matches("read_streaming_cold_chunks()").count(),
        1
    );
    assert_eq!(
        production
            .matches("row_local_constraint_device_source(")
            .count(),
        1
    );
    assert!(production.contains("insert_foreign_key_verdict_scratch_bytes("));
    assert!(production.contains("let allocation_peak_bytes = allocation_scope.peak_bytes()"));
    assert!(production.contains("if allocation_peak_bytes != peak"));
    assert!(production.contains("ConstraintCandidate::foreign_key("));
    assert!(production.contains("if fk.parent_generation.history_floor_requires_retry()"));
    assert!(production.contains("let mut history_floor_requires_retry = false"));
    assert!(production.contains("if let Some(row) = first_history"));
    assert!(production.contains("if history_floor_requires_retry"));
}

#[test]
fn foreign_key_binding_requires_exact_single_column_constraint_index_identity() {
    let source = include_str!("foreign_key_constraints.rs");
    assert!(source.contains("index.table == parent.name"));
    assert!(source.contains("index.column == foreign_key.referenced_column"));
    assert!(source.contains(
        "index.unique\n                && (index.primary_key || index.unique_constraint)"
    ));
    assert!(
        source.contains("index.key_columns.as_slice() == [foreign_key.referenced_column.as_str()]")
    );
    assert!(source.contains("table.indexes.get(binding.raw_ordinal)"));
    assert!(source.contains("table_schema_digest(table)"));
    assert!(source.contains("column.id == binding.id"));
}

#[test]
fn raw_ordinal_backing_index_drift_is_rejected_after_table_digest_is_rebound() {
    let engine = crate::Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE fk_catalog_parent (id int4 PRIMARY KEY)")
        .unwrap();
    engine
        .execute_text(
            2,
            "CREATE TABLE fk_catalog_child (id int4 PRIMARY KEY, pid int4)",
        )
        .unwrap();
    engine
        .execute_text(
            3,
            "ALTER TABLE ONLY fk_catalog_child ADD CONSTRAINT fk_catalog_child_pid_fkey FOREIGN KEY (pid) REFERENCES fk_catalog_parent(id)",
        )
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let command = gpu_db_sql::parse_command("INSERT INTO fk_catalog_child VALUES (1, 7)").unwrap();
    let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch_foreign_key_proof_only(
        &command,
        &catalog,
        catalog.commit_seq,
    )
    .unwrap()
    .expect("test-only FK builder accepts the supported shape");
    let mut proof = ForeignKeyConstraintProof::compile(&engine, &batch, &catalog).unwrap();
    let mut drifted = (*catalog).clone();
    let binding = proof.bindings.first_mut().expect("one FK binding");
    let parent = drifted
        .relational_catalog
        .get_mut(&binding.parent.name)
        .expect("parent remains catalogued");
    let index = parent
        .indexes
        .get_mut(binding.supporting_index.raw_ordinal)
        .expect("bound raw index ordinal remains in range");
    index.oid = index.oid.wrapping_add(1);
    // Deliberately rebind the parent table digest to the mutated catalog so this assertion reaches
    // the supporting-index identity check rather than short-circuiting on the table digest.
    binding.parent.schema_digest = crate::engine_transaction_reset::table_schema_digest(parent)
        .expect("mutated parent remains digestible");
    assert!(matches!(
        proof.validate_current_catalog(&drifted, drifted.commit_seq),
        Err(crate::EngineError::ApplyFailed(message))
            if message == "resident INSERT foreign-key proof parent binding drifted"
    ));
}

#[test]
fn zero_row_direct_operator_is_an_explicit_execution_tested_contract() {
    let operator = include_str!("../../../execution/src/insert_foreign_key_verdict.rs");
    let operator_tests = include_str!("../../../execution/src/tests/insert_foreign_key_verdict.rs");
    assert!(operator.contains("if child_row_count == 0"));
    assert!(operator.contains("readback_bytes: 0"));
    assert!(operator_tests.contains(
        "cuda_insert_foreign_key_verdict_handles_empty_duplicate_null_and_self_provider"
    ));
    assert!(operator_tests.contains("insert_foreign_key_verdict_against_shards(&columns, 0"));
    assert!(operator_tests
        .contains("cuda_insert_foreign_key_verdict_preserves_current_original_truth_table"));
}

fn gpu_foreign_key_engine(prefix: &str) -> Option<(crate::Engine, String, String)> {
    let mut engine = crate::Engine::new_local();
    let hardware = engine.cuda_driver_probe_runtime().snapshot();
    if !hardware.driver_available || hardware.device_count == 0 {
        return None;
    }
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(1);
    let parent = format!("{prefix}_parent");
    let child = format!("{prefix}_child");
    engine
        .execute_text(1, &format!("CREATE TABLE {parent} (id int4 PRIMARY KEY)"))
        .expect("parent DDL succeeds");
    engine
        .execute_text(
            2,
            &format!("CREATE TABLE {child} (id int4 PRIMARY KEY, pid int4)"),
        )
        .expect("child DDL succeeds");
    engine
        .execute_text(
            3,
            &format!(
                "ALTER TABLE ONLY {child} ADD CONSTRAINT {child}_pid_fkey FOREIGN KEY (pid) REFERENCES {parent}(id)"
            ),
        )
        .expect("FK DDL succeeds");
    engine
        .execute_text(4, &format!("INSERT INTO {parent} VALUES (7)"))
        .expect("parent seed succeeds");
    engine
        .execute_text(5, &format!("INSERT INTO {child} VALUES (1, 7)"))
        .expect("child seed succeeds");
    engine
        .populate_relational_residency_snapshot(&parent)
        .expect("parent becomes resident");
    engine
        .populate_relational_residency_snapshot(&child)
        .expect("child becomes resident");
    Some((engine, parent, child))
}

fn foreign_key_proof_plan(
    engine: &crate::Engine,
    sql: &str,
) -> crate::engine_insert_plan::PreparedDeviceInsertPlan {
    let catalog = engine.catalog_snapshot();
    let command = gpu_db_sql::parse_command(sql).expect("test INSERT parses");
    let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch_foreign_key_proof_only(
        &command,
        &catalog,
        catalog.commit_seq,
    )
    .expect("foreign-key proof builder succeeds")
    .expect("foreign-key proof shape remains eligible");
    crate::engine_insert_plan::PreparedDeviceInsertPlan::from_typed_batch(batch, engine, &catalog)
        .expect("foreign-key pre-WAL preparation succeeds")
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn actual_gpu_foreign_key_proof_succeeds_for_match_and_null_with_exact_peak() {
    let Some((engine, _parent, child)) = gpu_foreign_key_engine("fk_proof_match") else {
        return;
    };
    let plan = foreign_key_proof_plan(&engine, &format!("INSERT INTO {child} VALUES (2, 7)"));
    crate::typed_insert_batch::reset_constraint_source_upload_count();
    let report = plan
        .inspect_current_resident_foreign_key_constraints(&engine, |report| report)
        .expect("both-world parent match succeeds");
    assert_eq!(report.child_table, child);
    assert!(report.parent_shard_count > 0);
    assert_eq!(
        report.allocation_peak_bytes,
        report.expected_allocation_peak_bytes
    );
    assert_eq!(
        crate::typed_insert_batch::constraint_source_upload_count(),
        1
    );

    let plan = foreign_key_proof_plan(&engine, &format!("INSERT INTO {child} VALUES (3, NULL)"));
    plan.inspect_current_resident_foreign_key_constraints(&engine, |_| ())
        .expect("MATCH SIMPLE NULL is satisfied without a parent value");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn actual_gpu_foreign_key_proof_handles_multiple_parent_shards() {
    let Some((mut engine, parent, child)) = gpu_foreign_key_engine("fk_proof_shards") else {
        return;
    };
    engine
        .execute_text(6, &format!("INSERT INTO {parent} VALUES (8)"))
        .expect("second parent seed succeeds");
    engine
        .populate_relational_residency_snapshot(&parent)
        .expect("parent republish succeeds");
    let plan = foreign_key_proof_plan(&engine, &format!("INSERT INTO {child} VALUES (2, 8)"));
    let report = plan
        .inspect_current_resident_foreign_key_constraints(&engine, |report| report)
        .expect("match in a later parent shard succeeds");
    assert!(report.parent_shard_count >= 2);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn actual_gpu_foreign_key_proof_arbitrates_violation_and_history_floor_without_side_effects() {
    let Some((engine, parent, child)) = gpu_foreign_key_engine("fk_proof_arbitration") else {
        return;
    };
    let plan = foreign_key_proof_plan(&engine, &format!("INSERT INTO {child} VALUES (2, 99)"));
    let error = plan
        .inspect_current_resident_foreign_key_constraints(&engine, |_| ())
        .expect_err("missing parent must be the FK terminal");
    assert!(matches!(
        error,
        crate::ExecuteError::Engine(crate::EngineError::ForeignKeyViolation(message))
            if message == format!(
                "insert or update on table \"{child}\" violates foreign key constraint \"{child}_pid_fkey\""
            )
    ));

    let original = engine.committed_seq();
    engine
        .read_state
        .residency
        .with_shards_mut_for_table(&parent, |tables| {
            for shard in tables
                .get_mut(&parent)
                .expect("parent shards remain resident")
            {
                shard.history_floor_index = original.saturating_add(1);
            }
        });
    let matched = foreign_key_proof_plan(&engine, &format!("INSERT INTO {child} VALUES (3, 7)"));
    matched
        .inspect_current_resident_foreign_key_constraints(&engine, |_| ())
        .expect("positive both-world match remains valid above a parent history floor");
    let missing = foreign_key_proof_plan(&engine, &format!("INSERT INTO {child} VALUES (4, 99)"));
    let wal_before = engine.durable_wal_records().len();
    let row_before = engine.read_state.mvcc.current_row_id();
    let boundary_before = engine.committed_seq();
    assert!(matches!(
        missing.inspect_current_resident_foreign_key_constraints(&engine, |_| ()),
        Err(crate::ExecuteError::Serialization(_))
    ));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.read_state.mvcc.current_row_id(), row_before);
    assert_eq!(engine.committed_seq(), boundary_before);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn actual_gpu_foreign_key_proof_rejects_backing_index_identity_drift() {
    let Some((engine, parent, child)) = gpu_foreign_key_engine("fk_proof_index_drift") else {
        return;
    };
    let plan = foreign_key_proof_plan(&engine, &format!("INSERT INTO {child} VALUES (2, 7)"));
    let index_name = engine
        .catalog_snapshot()
        .relational_catalog
        .get(&parent)
        .and_then(|table| table.indexes.iter().find(|index| index.primary_key))
        .map(|index| index.name.clone())
        .expect("parent primary-key index remains catalogued");
    engine
        .execute_text(
            6,
            &format!(
                "ALTER TABLE ONLY {parent} RENAME CONSTRAINT {index_name} TO {parent}_renamed_pkey"
            ),
        )
        .expect("backing index rename succeeds");
    let wal_before = engine.durable_wal_records().len();
    let row_before = engine.read_state.mvcc.current_row_id();
    let boundary_before = engine.committed_seq();
    let error = plan
        .inspect_current_resident_foreign_key_constraints(&engine, |_| ())
        .expect_err("changed backing index must invalidate the prepared FK proof");
    assert!(matches!(
        error,
        crate::ExecuteError::Serialization(_)
            | crate::ExecuteError::Engine(crate::EngineError::ApplyFailed(_))
    ));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.read_state.mvcc.current_row_id(), row_before);
    assert_eq!(engine.committed_seq(), boundary_before);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn actual_gpu_foreign_key_proof_retries_current_only_parent_xor_without_side_effects() {
    let Some((mut engine, parent, child)) = gpu_foreign_key_engine("fk_proof_current_only") else {
        return;
    };
    let original_read_snapshot = engine.committed_seq();
    let plan = foreign_key_proof_plan(&engine, &format!("INSERT INTO {child} VALUES (2, 8)"));
    engine
        .execute_text(6, &format!("INSERT INTO {parent} VALUES (8)"))
        .expect("current-only parent version commits after the proof read snapshot");
    engine
        .populate_relational_residency_snapshot(&parent)
        .expect("parent current generation republish succeeds");
    let wal_before = engine.durable_wal_records().len();
    let row_before = engine.read_state.mvcc.current_row_id();
    let boundary_before = engine.committed_seq();
    super::reset_current_generation_validation_entries();
    let error = plan
        .inspect_current_resident_foreign_key_constraints(&engine, |_| ())
        .expect_err("current-only parent truth must force the post-verdict history retry");
    assert!(matches!(
        error,
        crate::ExecuteError::Serialization(message)
            if message == format!(
                "resident foreign-key history changed after read snapshot {original_read_snapshot} at incoming row 0"
            )
    ));
    assert_eq!(
        super::current_generation_validation_entries(),
        1,
        "current-only XOR must reach the GPU FK evaluator, not fail a stale generic catalog gate"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.read_state.mvcc.current_row_id(), row_before);
    assert_eq!(engine.committed_seq(), boundary_before);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn actual_gpu_foreign_key_proof_rejects_target_semantic_drift_before_the_evaluator() {
    let Some((engine, _parent, child)) = gpu_foreign_key_engine("fk_proof_target_drift") else {
        return;
    };
    let plan = foreign_key_proof_plan(&engine, &format!("INSERT INTO {child} VALUES (2, 7)"));
    engine
        .execute_text(
            6,
            &format!("ALTER TABLE ONLY {child} ADD CONSTRAINT {child}_id_positive CHECK (id > 0)"),
        )
        .expect("target CHECK mutation succeeds");
    let wal_before = engine.durable_wal_records().len();
    let row_before = engine.read_state.mvcc.current_row_id();
    let boundary_before = engine.committed_seq();
    super::reset_current_generation_validation_entries();
    assert!(matches!(
        plan.inspect_current_resident_foreign_key_constraints(&engine, |_| ()),
        Err(crate::ExecuteError::Serialization(_))
    ));
    assert_eq!(
        super::current_generation_validation_entries(),
        0,
        "target semantic drift must be rejected by the exact local proof before GPU evaluation"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.read_state.mvcc.current_row_id(), row_before);
    assert_eq!(engine.committed_seq(), boundary_before);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn actual_gpu_foreign_key_proof_declines_pressure_and_torn_parent_generation() {
    let Some((mut engine, _parent, child)) = gpu_foreign_key_engine("fk_proof_sabotage") else {
        return;
    };
    let pressure = foreign_key_proof_plan(&engine, &format!("INSERT INTO {child} VALUES (2, 7)"));
    engine.mark_gpu_memory_pressured(0);
    let wal_before = engine.durable_wal_records().len();
    let row_before = engine.read_state.mvcc.current_row_id();
    let boundary_before = engine.committed_seq();
    assert!(matches!(
        pressure.inspect_current_resident_foreign_key_constraints(&engine, |_| ()),
        Err(crate::ExecuteError::Serialization(_))
    ));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.read_state.mvcc.current_row_id(), row_before);
    assert_eq!(engine.committed_seq(), boundary_before);

    let Some((mut engine, parent, child)) = gpu_foreign_key_engine("fk_proof_torn") else {
        return;
    };
    engine
        .execute_text(6, &format!("INSERT INTO {parent} VALUES (8)"))
        .expect("second parent seed succeeds");
    engine
        .populate_relational_residency_snapshot(&parent)
        .expect("two-shard parent generation publishes");
    let plan = foreign_key_proof_plan(&engine, &format!("INSERT INTO {child} VALUES (2, 7)"));
    // Bypass the regular descriptor publisher deliberately: it retokens every shard as one
    // generation, whereas this sabotage must retain one valid shard and replace only another
    // shard's token to model a torn published map.
    let current = engine.read_state.residency.shards.load_full();
    let mut torn = (*current).clone();
    let shards = torn
        .get_mut(&parent)
        .expect("parent shards remain resident");
    assert!(
        shards.len() >= 2,
        "torn-generation sabotage needs two parent shards"
    );
    shards[1].point_route_generation = std::sync::Arc::new(());
    engine
        .read_state
        .residency
        .shards
        .store(std::sync::Arc::new(torn));
    let wal_before = engine.durable_wal_records().len();
    let row_before = engine.read_state.mvcc.current_row_id();
    let boundary_before = engine.committed_seq();
    assert!(matches!(
        plan.inspect_current_resident_foreign_key_constraints(&engine, |_| ()),
        Err(crate::ExecuteError::Serialization(_))
    ));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.read_state.mvcc.current_row_id(), row_before);
    assert_eq!(engine.committed_seq(), boundary_before);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn actual_gpu_foreign_key_proof_declines_cross_gpu_parent_shard() {
    let Some((engine, parent, child)) = gpu_foreign_key_engine("fk_proof_cross_gpu") else {
        return;
    };
    let plan = foreign_key_proof_plan(&engine, &format!("INSERT INTO {child} VALUES (2, 7)"));
    engine
        .read_state
        .residency
        .with_shards_mut_for_table(&parent, |tables| {
            let shard = tables
                .get_mut(&parent)
                .and_then(|shards| shards.first_mut())
                .expect("parent shard remains resident");
            shard.gpu_id = shard.gpu_id.saturating_add(1);
        });
    let wal_before = engine.durable_wal_records().len();
    let row_before = engine.read_state.mvcc.current_row_id();
    let boundary_before = engine.committed_seq();
    assert!(matches!(
        plan.inspect_current_resident_foreign_key_constraints(&engine, |_| ()),
        Err(crate::ExecuteError::Serialization(_))
    ));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.read_state.mvcc.current_row_id(), row_before);
    assert_eq!(engine.committed_seq(), boundary_before);
}
