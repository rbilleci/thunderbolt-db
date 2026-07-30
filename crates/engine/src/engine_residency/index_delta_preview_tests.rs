//! Focused zero-CUDA/currentness checks for `index_delta_preview`.

use crate::engine_insert_plan::PreparedDeviceInsertPlan;
use crate::Engine;

fn indexed_preview_engine() -> Option<Engine> {
    let mut engine = Engine::new_local();
    let hardware = engine.cuda_driver_probe_runtime().snapshot();
    if !hardware.driver_available || hardware.device_count == 0 {
        return None;
    }
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(64);
    engine
        .execute_text(
            1,
            "CREATE TABLE inert_index_preview (id int4 PRIMARY KEY, code int4 UNIQUE)",
        )
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO inert_index_preview VALUES (1, 10)")
        .unwrap();
    engine
        .populate_relational_residency_snapshot("inert_index_preview")
        .unwrap();
    engine
        .publish_relational_resident_indexes("inert_index_preview")
        .unwrap();
    Some(engine)
}

fn indexed_preview_plan(engine: &Engine) -> PreparedDeviceInsertPlan {
    let catalog = engine.catalog_snapshot();
    let command =
        gpu_db_sql::parse_command("INSERT INTO inert_index_preview VALUES (2, 20)").unwrap();
    let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch_proof_only(
        &command,
        &catalog,
        catalog.commit_seq,
    )
    .unwrap()
    .expect("indexed proof-only builder remains eligible");
    PreparedDeviceInsertPlan::from_typed_batch(batch, engine, &catalog).unwrap()
}

#[test]
fn indexed_in_place_preview_pins_exact_arcs_without_typed_index_cuda_work() {
    let Some(engine) = indexed_preview_engine() else {
        return;
    };
    let plan = indexed_preview_plan(&engine);
    let proposal = plan
        .prepare_row_id_proposal(engine.read_state.mvcc.current_row_id())
        .unwrap();
    #[cfg(feature = "probe-timing")]
    let typed_before = gpu_db_execution::prepared_resident_typed_indexes_insert_counters();
    let wal_before = engine.durable_wal_records().len();
    let boundary_before = engine.committed_seq();
    let allocator_before = engine.read_state.mvcc.current_row_id();
    let report = plan
        .inspect_current_resident_index_delta_preview(&engine, proposal, |report| report)
        .unwrap();
    #[cfg(feature = "probe-timing")]
    {
        let typed_after = gpu_db_execution::prepared_resident_typed_indexes_insert_counters();
        assert_eq!(typed_after.prepares, typed_before.prepares);
        assert_eq!(typed_after.submits, typed_before.submits);
        assert_eq!(typed_after.drains, typed_before.drains);
    }
    assert_eq!(report.raw_index_count, 2);
    assert_eq!(report.physical_index_count, 2);
    assert_eq!(report.base_row, 1);
    assert_eq!(report.incoming_rows, 1);
    assert_eq!(report.end_row, 2);
    assert_eq!(report.bounded_readback_bytes, 4);
    assert!(report.fused_preparation_bytes > 0);
    assert!(report.fused_pooled_allocation_slots > 0);
    assert!(report.fused_owner_array_bytes > 0);
    assert!(report.fused_staging_bytes > 0);
    assert_eq!(report.fused_stamp_bytes, 8);
    assert!(report.fused_materialization_scratch_bytes > 0);
    assert_eq!(
        report.combined_preparation_bytes,
        report
            .preparation_bytes
            .checked_add(report.fused_preparation_bytes)
            .unwrap()
    );
    assert_eq!(
        report.combined_allocation_slots,
        u64::try_from(report.transient_allocation_slot_count).unwrap()
            + report.fused_pooled_allocation_slots
    );
    assert!(report.retained_allocation_pin_count >= 3);
    assert!(report.source_payload_bytes > 0);
    assert!(report.pinned_persistent_index_bytes > 0);
    assert!(report.host_retained_bytes > 0);
    assert!(report.host_allocation_slots > 0);
    assert_eq!(report.host_generation_pin_slots, 0);
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.committed_seq(), boundary_before);
    assert_eq!(engine.read_state.mvcc.current_row_id(), allocator_before);
}

#[test]
fn indexed_preview_pre_wal_forecast_includes_the_fused_tail_before_materialization() {
    let source = include_str!("index_delta_preview.rs");
    let forecast = source
        .rsplit("let fused_footprint =")
        .next()
        .and_then(|section| section.split("Ok(preview)").next())
        .expect("allocation-free fused forecast section");
    let compact_forecast: String = forecast
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    for required in [
        "indexed_in_place_fused_footprint_forecast",
        "indexed_in_place_fused_host_retention_forecast",
        "indexed_in_place_fused_materialization_scratch_forecast",
        "checked_add(preview.fused_footprint.pooled_device_scratch_bytes)",
        "pooled_device_scratch_slots",
        "max(preview.fused_footprint.status_readback_bytes)",
    ] {
        assert!(
            compact_forecast.contains(required),
            "pre-WAL indexed forecast must include {required}"
        );
    }
    let binding = include_str!("indexed_forecast.rs");
    assert!(binding
        .contains("retained_device_transient_bytes: resource.retained_device_transient_bytes"));
    assert!(binding.contains("incremental_allocation_slots: resource.incremental_allocation_slots"));
}

#[test]
fn indexed_in_place_preview_source_has_no_cuda_or_terminal_write_capability() {
    let source = include_str!("index_delta_preview.rs")
        .split("pub(super) fn into_append_and_parts")
        .next()
        .expect("zero-CUDA preview precedes allocating consumer");
    for forbidden in [
        "CudaAllocationScope",
        ".prepare_resident_typed_indexes_insert",
        ".submit",
        "WalBuffer",
        "canonical_operation",
        "apply_resident",
        "begin_point_index_mutation",
    ] {
        assert!(
            !source.contains(forbidden),
            "zero-CUDA preview must not expose {forbidden}"
        );
    }
    assert!(source.contains("Arc::ptr_eq"));
    assert!(include_str!("index_delta_preview.rs").contains("sharded_point_route_publish_lock"));
    assert!(include_str!("index_delta_preview.rs").contains("point_index_mutation_epoch"));
}

#[test]
fn indexed_preview_prelease_consumer_never_constructs_an_identity_report() {
    let source = include_str!("index_delta_preview.rs");
    let consumer = source
        .split("pub(super) fn into_append_and_parts")
        .nth(1)
        .and_then(|tail| tail.split("/// Fields transferred only").next())
        .expect("preview consumer is bounded by its transferred-parts declaration");
    for forbidden in [
        "host_retention_report",
        "HostRetentionReport",
        "BTreeMap",
        "BTreeSet",
    ] {
        assert!(
            !consumer.contains(forbidden),
            "pre-lease preview consumer must not construct {forbidden}"
        );
    }
    assert!(consumer.contains("host_retention_geometry"));
}

#[cfg(feature = "probe-timing")]
#[test]
fn indexed_in_place_preview_deduplicates_distinct_wrapper_aliases_by_allocation() {
    let Some(engine) = indexed_preview_engine() else {
        return;
    };
    let source = engine
        .read_state
        .residency
        .shards
        .load_full()
        .get("inert_index_preview")
        .and_then(|shards| shards.last())
        .and_then(|open| open.device_memory.as_ref())
        .cloned()
        .unwrap();
    assert!(
        super::index_delta_preview::distinct_wrapper_alias_deduplicates_for_test(&source).unwrap()
    );
}
