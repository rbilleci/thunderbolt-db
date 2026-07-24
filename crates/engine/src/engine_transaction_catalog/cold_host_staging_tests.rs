use super::*;
use crate::engine_streaming_exec::STREAMING_COLD_SPILL_THRESHOLD_TEST;
use std::sync::atomic::Ordering;

struct SpillThresholdGuard(u64);

impl Drop for SpillThresholdGuard {
    fn drop(&mut self) {
        STREAMING_COLD_SPILL_THRESHOLD_TEST.store(self.0, Ordering::Relaxed);
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn spilled_wide_text_unique_validation_is_host_bounded_effect_free_and_retryable() {
    let prior_threshold = STREAMING_COLD_SPILL_THRESHOLD_TEST.swap(1024, Ordering::Relaxed);
    let _threshold_guard = SpillThresholdGuard(prior_threshold);
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            9_200,
            "CREATE TABLE cold_spilled_text_host_bound \
             (id int4 PRIMARY KEY, payload text)",
        )
        .unwrap();
    let wide = "x".repeat(160 * 1024);
    let values = (0..12)
        .map(|id| format!("({id}, '{wide}-{id:02}')"))
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(
            9_201,
            &format!("INSERT INTO cold_spilled_text_host_bound VALUES {values}"),
        )
        .unwrap();

    // The relation exceeds this configured GPU working-set budget, forcing a complete cold
    // capture. The tiny test-only spill threshold makes every retained payload file-backed.
    engine.set_relational_residency_budget_bytes(0, 1024 * 1024);
    let count = engine
        .execute_relational_select(&select("SELECT COUNT(*) FROM cold_spilled_text_host_bound"))
        .unwrap();
    assert!(matches!(count.executed_target, DeviceTarget::Gpu(_)));
    assert!(engine.streaming_cold_spills() > 0);
    engine
        .execute_text(
            9_202,
            "INSERT INTO cold_spilled_text_host_bound VALUES (12, 'short-tail')",
        )
        .unwrap();
    assert!(engine
        .table_chunk_authoritative("cold_spilled_text_host_bound")
        .is_some());

    engine.submit_transaction(9_203, parsed("BEGIN")).unwrap();
    let wal_before = engine.durable_wal_records().len();
    let low_budget = 64 * 1024u64;
    engine.set_relational_residency_budget_bytes(0, low_budget);
    engine
        .read_state
        .residency
        .cold_index_validation_peak_host_staging_bytes
        .store(0, Ordering::Relaxed);
    let exhausted = engine
        .submit_transaction(
            9_203,
            parsed(
                "CREATE UNIQUE INDEX cold_spilled_text_payload_idx \
                 ON cold_spilled_text_host_bound (payload)",
            ),
        )
        .expect_err("even a one-row/pair proof exceeds the deliberately tiny host lease");
    assert!(
        matches!(exhausted, ExecuteError::ResourceExhausted(_))
            && exhausted.to_string().contains("cold host staging"),
        "{exhausted}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    let failed_snapshot = engine.transaction_snapshot_handle(9_203).unwrap();
    assert!(failed_snapshot.transaction_delta_is_empty());
    assert!(failed_snapshot.transaction_catalog().relational_catalog
        ["cold_spilled_text_host_bound"]
        .indexes
        .iter()
        .all(|index| index.name != "cold_spilled_text_payload_idx"));
    let failed_peak = engine
        .read_state
        .residency
        .cold_index_validation_peak_host_staging_bytes
        .load(Ordering::Relaxed);
    assert!(
        failed_peak <= low_budget,
        "failed staging crossed its host lease: peak={failed_peak}, limit={low_budget}"
    );
    drop(failed_snapshot);

    let retry_budget = 32 * 1024 * 1024u64;
    engine.set_relational_residency_budget_bytes(0, retry_budget);
    engine
        .read_state
        .residency
        .cold_index_validation_peak_host_staging_bytes
        .store(0, Ordering::Relaxed);
    engine
        .submit_transaction(
            9_203,
            parsed(
                "CREATE UNIQUE INDEX cold_spilled_text_payload_idx \
                 ON cold_spilled_text_host_bound (payload)",
            ),
        )
        .expect("the same transaction can retry from immutable spilled windows");
    let retry_peak = engine
        .read_state
        .residency
        .cold_index_validation_peak_host_staging_bytes
        .load(Ordering::Relaxed);
    assert!(
        retry_peak > wide.len() as u64 && retry_peak <= retry_budget,
        "wide-text retry must be non-vacuous and bounded: peak={retry_peak}, limit={retry_budget}"
    );

    // Canonical publication may rebuild all named device indexes; the temporary proof lease above
    // is the bounded behavior under test, so restore unconstrained admission for that owner step.
    engine.set_relational_residency_budget_bytes(0, u64::MAX);
    engine.submit_transaction(9_203, parsed("COMMIT")).unwrap();
    assert!(
        index_named(
            &engine
                .relational_catalog_table("cold_spilled_text_host_bound")
                .unwrap(),
            "cold_spilled_text_payload_idx"
        )
        .unique
    );
    assert!(engine
        .table_chunk_authoritative("cold_spilled_text_host_bound")
        .is_some());
    let duplicate = engine
        .execute_text(
            9_204,
            &format!(
                "INSERT INTO cold_spilled_text_host_bound VALUES (13, '{}-00')",
                wide
            ),
        )
        .expect_err("the published cold unique index must reject the wide duplicate");
    assert!(
        matches!(
            duplicate,
            ExecuteError::Engine(EngineError::UniqueViolation(_))
        ),
        "{duplicate}"
    );
}
