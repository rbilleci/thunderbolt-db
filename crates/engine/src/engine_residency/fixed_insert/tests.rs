//! Ownership and source-contract regressions for the fixed physical DeviceInsertPlan strategy.
//!
//! These tests retain the parent module name so private implementation contracts remain covered
//! without extending the production source root.

#[test]
fn live_device_plan_apply_requires_the_post_wal_typed_claim_permit() {
    let source = include_str!("../fixed_insert.rs")
        .split("\n#[cfg(test)]\n#[path = \"fixed_insert/tests.rs\"]\nmod ownership_tests")
        .next()
        .expect("implementation precedes tests");
    assert!(source.contains("fn apply_after_typed_wal_claim"));
    assert!(source.contains("TypedInsertPostWalApplyPermit"));
    assert!(source.contains("permit.into_append_created_by()"));
    assert!(
        !source.contains("pub(crate) fn apply(\n"),
        "DeviceInsertPlan must not expose an unclaimed apply entry"
    );
}

#[test]
fn exact_row_id_box_is_slot_only_and_synthetic_or_consumed_forms_fail_closed() {
    let row_ids = super::DeviceInsertRowIds::exact(vec![7_u64, 8].into());
    let mut report = crate::engine_insert_plan::host_retention::HostRetentionReport::default();
    row_ids.append_host_allocation_slot(&mut report).unwrap();
    assert_eq!(report.retained_bytes(), 0);
    assert_eq!(report.allocation_slots().unwrap(), 1);

    let synthetic = super::DeviceInsertRowIds::synthetic_no_identity();
    assert!(synthetic.append_host_allocation_slot(&mut report).is_err());
    let mut consumed = super::DeviceInsertRowIds::exact(vec![9_u64].into());
    assert!(consumed.take_exact().is_some());
    assert!(consumed.append_host_allocation_slot(&mut report).is_err());
}

#[test]
fn indexed_reservation_mode_reuses_the_single_append_preparation_core() {
    let source = include_str!("../fixed_insert.rs")
        .split("\n#[cfg(test)]\n#[path = \"fixed_insert/tests.rs\"]\nmod ownership_tests")
        .next()
        .expect("implementation precedes tests");
    let core = source
        .split("fn prepare_resident_open_shard_append_core")
        .nth(1)
        .and_then(|section| {
            section
                .split("\n    /// Apply exactly one pre-WAL plan")
                .next()
        })
        .expect("one append preparation core");
    assert!(core.contains("TransactionTerminalUnindexed"));
    assert!(!core.contains("LiveUnindexed"));
    assert!(core.contains("IndexedInPlaceReservation"));
    assert!(core.contains("source_matches_table(&source, table, true)"));
    assert!(core.contains("source_matches_indexed_in_place_reservation(&source, index_table)"));
    let reservation_decline = core
        .find("if indexed_in_place_proof\n            && (resets_existing_rows")
        .expect("in-place proof mode rejects reset/dense/bootstrap/headroom shapes");
    let rollover_decline = core
        .find("if indexed_fixed_rollover_proof\n            && ((!")
        .expect("rollover proof mode rejects an in-place shape");
    for later in [
        ".checked_dense_payload(index_table)",
        "fixed_width_desired_capacity",
        "PendingInPlaceCreatedBy::reserve_pre_wal",
        "budget_allocation_lock",
    ] {
        let later = core.find(later).expect("ordinary later preparation branch");
        assert!(
            reservation_decline < later && rollover_decline < later,
            "reservation mode must decline before {later}"
        );
    }
    assert!(
            [
                "dense_rollover",
                "bootstrap_sentinel",
                ".checked_add(k)",
                "end > identity.capacity"
            ]
            .into_iter()
            .all(|decline| core[reservation_decline..rollover_decline].contains(decline)),
            "reservation mode must explicitly decline every fixed/bootstrap/dense shape before allocation"
        );
    let adapter = source
        .split("fn prepare_resident_open_shard_append_indexed_in_place_reservation")
        .nth(1)
        .and_then(|section| section.split("\n}\n\nfn source_matches_table").next())
        .expect("proof adapter");
    assert!(adapter.contains("prepare_resident_open_shard_append_core"));
    assert!(!adapter.contains("let catalog ="));
    assert!(!adapter.contains("reserve_pre_wal"));

    let transaction_selector = source
        .split("pub(crate) fn compile_transaction_terminal_typed_insert_device_plan<'a>")
        .nth(1)
        .and_then(|section| {
            section
                .split(
                    "pub(crate) fn compile_transaction_terminal_typed_insert_device_plan_with_gate",
                )
                .next()
        })
        .expect("transaction-terminal unindexed device-plan selector");
    assert!(transaction_selector.contains("ResidentOpenShardAppend"));
    assert!(
        !transaction_selector.contains("IndexedInPlaceReservation")
            && !transaction_selector.contains("IndexedFixedRolloverReservation"),
        "the unindexed selector must not infer an indexed reservation"
    );
    let indexed_selector = source
        .split("fn compile_transaction_terminal_indexed_typed_insert_device_plan_core<'a>")
        .nth(1)
        .and_then(|section| {
            section
                .split("fn prepare_resident_open_shard_append_core")
                .next()
        })
        .expect("transaction-terminal indexed selector");
    assert!(indexed_selector.contains("index_delta_preview::prepare_from_resident_source"));
    assert!(indexed_selector.contains("index_rollover::prepare_fixed_rollover_preview"));
    assert!(indexed_selector.contains("index_rollover::materialize"));
    assert!(indexed_selector.contains("DeviceInsertPlanKind::IndexedInPlace"));
}

#[test]
fn indexed_in_place_fused_owner_is_all_i32_only_and_has_no_logical_publication_surface() {
    let source = include_str!("indexed_fused.rs");
    let fused = source
        .split("pub(in super::super) struct IndexedInPlaceFusedApplyInputs")
        .nth(1)
        .and_then(|section| {
            section
                .split("pub(in super::super) fn prepare_inputs")
                .next()
        })
        .expect("private fused input and owner section");
    for required in [
        "prepare_i32_fused_apply",
        "index: None",
        "created_by_stamps",
        "stamps_match_expected_commit",
        "owner_array_backing_identity",
        "staging_backing_identity",
    ] {
        assert!(
            fused.contains(required),
            "indexed fused owner must retain {required}"
        );
    }
    assert!(
            fused.contains("apply_payload_before_header")
                && fused.contains("apply_before_header(&self.created_by_stamps)"),
            "the sealed physical token must expose its post-WAL payload write while withholding visibility"
        );
    for forbidden in [
        "submit_resident",
        ".publish(",
        "shard_pk_device_index",
        "wal",
    ] {
        assert!(
            !fused.contains(forbidden),
            "indexed fused owner must not expose {forbidden}"
        );
    }
    let inputs = source
        .split("pub(in super::super) fn prepare_inputs")
        .nth(1)
        .and_then(|section| {
            section
                .split("pub(in super::super) fn source_is_all_i32_fixed")
                .next()
        })
        .expect("sealed fused input derivation");
    let compact_inputs: String = inputs
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    assert!(compact_inputs.contains("source_is_all_i32_fixed(&plan.source)"));
    assert!(compact_inputs.contains("plan.identity.device_memory"));
    assert!(compact_inputs.contains("plan.row_ids"));
    assert!(compact_inputs.contains("expected_commit_seq"));
}
#[test]
fn rollover_coordinates_are_checked_before_allocation() {
    assert_eq!(
        super::checked_rollover_coordinates(7, 11, 13),
        Some((8, 24))
    );
    assert_eq!(super::checked_rollover_coordinates(u32::MAX, 0, 0), None);
    assert_eq!(super::checked_rollover_coordinates(7, usize::MAX, 1), None);
}
#[test]
fn typed_post_wal_paths_cannot_allocate_a_replacement_generation() {
    let mutation = include_str!("../mutation.rs");
    let fixed_apply = mutation
        .split("} else if preallocated_fixed_plan {")
        .nth(1)
        .and_then(|section| {
            section
                .split("} else if !has_text && !batch_has_null {")
                .next()
        })
        .expect("typed fixed post-WAL apply section");
    assert!(fixed_apply.contains("publish_uniform_post_wal"));
    for forbidden in [
        "relational_residency_device_memory",
        "retain_device_memory_",
        "PendingResidentShard::build",
        "fixed_width_desired_capacity",
    ] {
        assert!(
            !fixed_apply.contains(forbidden),
            "typed fixed post-WAL apply must not {forbidden}"
        );
    }

    let fixed_plan = include_str!("../fixed_insert.rs");
    assert!(
        fixed_plan.contains("prepare_fixed_rollover_uniform_commit(expected_commit_seq)"),
        "the unindexed compiler must pre-resolve the fixed rollover publication before WAL"
    );
    let rollover = include_str!("../rollover.rs");
    assert!(
        !rollover.contains("fn finish_post_wal"),
        "the fixed rollover must not retain a separate post-WAL mutable finalizer"
    );

    let typed_in_place = mutation
        .split("let typed_created_by_region = match &mut source {")
        .nth(1)
        .and_then(|section| section.split("let fused = if").next())
        .expect("typed in-place sidecar handoff");
    assert!(
        typed_in_place.contains("pending.into_region")
            && typed_in_place.contains("get_or_alloc_created_by_region"),
        "mutation must consume and publish the sealed in-place Arc"
    );
    assert!(!typed_in_place.contains("install_preallocated_created_by_region"));
    let pending_in_place = rollover
        .split("impl PendingInPlaceCreatedBy")
        .nth(1)
        .and_then(|section| section.split("impl PendingResidentShard").next())
        .expect("in-place allocation lifecycle");
    for forbidden in [
        "with_shards_mut",
        ".insert_shard(",
        "shard_created_by_memory",
    ] {
        assert!(
            !pending_in_place.contains(forbidden),
            "rollover lifecycle leaf must not publish {forbidden}"
        );
    }
}

#[test]
fn adapter_has_no_second_residency_or_durability_publisher() {
    let source = include_str!("../fixed_insert.rs");
    let implementation = source
        .split("#[cfg(test)]\n#[path = \"fixed_insert/tests.rs\"]\nmod ownership_tests")
        .next()
        .expect("source has an implementation prefix");
    for (prefix, suffix) in [
        ("RelationalResident", "Shard {"),
        (".insert_", "shard("),
        ("retain_device_", "memory_"),
        (".append_owned_", "chunks("),
        ("write_", "wal"),
        ("append_", "wal"),
    ] {
        let forbidden = format!("{prefix}{suffix}");
        assert!(
            !implementation.contains(&forbidden),
            "fixed INSERT adapter must not own {forbidden}"
        );
    }
    assert_eq!(
        implementation.matches("with_shards_mut_for_table(").count(),
        1,
        "the opaque post-WAL plan owns exactly one bounded open-shard metadata visibility cut"
    );
    for forbidden in [
        "relational_residency_device_memory",
        "retain_device_memory_",
        "shard_created_by_memory.insert",
        "shard_row_id_memory.insert",
    ] {
        assert!(
            !implementation.contains(forbidden),
            "fixed INSERT adapter must delegate {forbidden} to the allocation/publisher owner"
        );
    }
    assert!(implementation.contains("ResidentAppendSource::DevicePlan"));
    assert_eq!(
        implementation
            .matches("try_append_to_resident_open_shard")
            .count(),
        1,
        "the typed plan must enter mutation through exactly one publisher"
    );
    assert!(implementation.contains("_device_apply"));
    assert!(implementation.contains("budget_allocation"));
    assert!(
        !implementation.contains("pub(crate) fn new"),
        "the move-only plan must have no raw public constructor"
    );
    assert!(
        implementation.contains("row_ids: DeviceInsertRowIds"),
        "the plan must own its row-id input before WAL"
    );
    assert!(
        !implementation.contains("row_ids_present: bool"),
        "a caller-provided row-id presence bit must not cross the WAL/apply boundary"
    );
    let apply = implementation
        .split("fn apply_resident_open_shard_append")
        .nth(1)
        .expect("sealed apply exists")
        .split("\n    /// Internal reservation adapter")
        .next()
        .expect("sealed apply precedes reservation adapters");
    assert!(
        !apply.contains("row_ids:"),
        "apply must consume only row IDs sealed into the opaque plan"
    );
    assert!(
        source.contains("#[cfg(test)]\n    pub(crate) fn synthetic_no_identity"),
        "synthetic no-identity construction must remain test-only"
    );
}
