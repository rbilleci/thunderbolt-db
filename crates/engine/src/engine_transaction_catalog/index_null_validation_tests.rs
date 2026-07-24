use super::*;

/// Exact-cap NULL validation is its own invariant cohort: NULL-bearing keys never enter the GPU
/// duplicate group, but their source/validity staging remains fully charged.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cold_all_null_unique_uses_only_staging_bytes_under_an_exact_cap() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            4_260,
            "CREATE TABLE cold_all_null_unique (id int4 PRIMARY KEY, code int4)",
        )
        .unwrap();
    let values = (0..600)
        .map(|id| format!("({id}, NULL)"))
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(
            4_261,
            &format!("INSERT INTO cold_all_null_unique VALUES {values}"),
        )
        .unwrap();
    engine.set_relational_residency_budget_bytes(0, 8192);
    engine
        .execute_relational_select(&select("SELECT COUNT(*) FROM cold_all_null_unique"))
        .unwrap();
    engine
        .execute_text(4_262, "INSERT INTO cold_all_null_unique VALUES (600, NULL)")
        .unwrap();
    assert!(engine
        .table_chunk_authoritative("cold_all_null_unique")
        .is_some());

    engine
        .read_state
        .residency
        .cold_index_validation_peak_device_bytes
        .store(0, std::sync::atomic::Ordering::Relaxed);
    engine.submit_transaction(4_263, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            4_263,
            parsed(
                "CREATE UNIQUE INDEX cold_all_null_unique_code \
                 ON cold_all_null_unique (code)",
            ),
        )
        .expect("an all-NULL key set has no GROUP input");
    let exact_staging_peak = engine
        .read_state
        .residency
        .cold_index_validation_peak_device_bytes
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(exact_staging_peak > 0 && exact_staging_peak <= 8192);
    engine
        .submit_transaction(4_263, parsed("ROLLBACK"))
        .unwrap();

    engine.set_relational_residency_budget_bytes(0, exact_staging_peak);
    engine
        .read_state
        .residency
        .cold_index_validation_peak_device_bytes
        .store(0, std::sync::atomic::Ordering::Relaxed);
    engine.submit_transaction(4_264, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            4_264,
            parsed(
                "CREATE UNIQUE INDEX cold_all_null_unique_code \
                 ON cold_all_null_unique (code)",
            ),
        )
        .expect("the exact staged-source cap must not reserve nonexistent GROUP scratch");
    let retried_peak = engine
        .read_state
        .residency
        .cold_index_validation_peak_device_bytes
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(retried_peak > 0 && retried_peak <= exact_staging_peak);
    engine.submit_transaction(4_264, parsed("COMMIT")).unwrap();
}

/// Historical repeated nullable keys exercise shared validity descriptors across publication and
/// restart, keeping the compatibility path on the same GPU index implementation.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn legacy_repeated_nullable_index_republishes_on_gpu_and_restart() {
    const LEGACY_REPEATED: &[u8] = br#"{"CreateTable":{"table":"legacy_repeated_nullable","columns":[{"name":"a","ty":"Int4","domain":null,"default":null}],"primary_key":null,"unique_constraints":[{"name":"legacy_repeated_nullable_key","column":"a","columns":["a","a"]}],"check_constraints":[]}}"#;
    let mut typed = Vec::with_capacity(10 + LEGACY_REPEATED.len());
    typed.push(0xfe);
    typed.push(1);
    typed.extend_from_slice(&(LEGACY_REPEATED.len() as u64).to_le_bytes());
    typed.extend_from_slice(LEGACY_REPEATED);
    let engine = Engine::recover_from_durable_wal(&[gpu_db_wal::WalRecord {
        txn_id: 4_250,
        payload: Arc::from(typed),
    }])
    .expect("literal legacy repeated-key fixture must recover");
    let table = engine
        .relational_catalog_table("legacy_repeated_nullable")
        .unwrap();
    assert_eq!(table.indexes[0].key_columns, ["a", "a"]);
    engine
        .execute_text(
            4_251,
            "INSERT INTO legacy_repeated_nullable VALUES (NULL), (NULL), (7)",
        )
        .expect("shared validity descriptors preserve NULLS DISTINCT");
    let publication = engine
        .publish_relational_resident_indexes("legacy_repeated_nullable")
        .expect("repeated nullable descriptors publish one GPU index");
    assert_eq!(publication.indexes.len(), 1);
    let rows = engine
        .execute_relational_select(&select("SELECT a FROM legacy_repeated_nullable ORDER BY a"))
        .unwrap();
    assert!(matches!(rows.executed_target, DeviceTarget::Gpu(_)));

    let restarted = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    restarted
        .publish_relational_resident_indexes("legacy_repeated_nullable")
        .expect("restart republishes the historical repeated nullable index");
    restarted
        .execute_text(4_252, "INSERT INTO legacy_repeated_nullable VALUES (NULL)")
        .expect("NULL-bearing repeated key remains distinct after restart");
    let duplicate = restarted
        .execute_text(4_253, "INSERT INTO legacy_repeated_nullable VALUES (7)")
        .expect_err("present repeated key must remain unique after restart");
    assert!(
        matches!(
            duplicate,
            ExecuteError::Engine(EngineError::UniqueViolation(_))
        ),
        "{duplicate}"
    );
}
