use super::*;

#[test]
fn prepared_u64_htod_publication_is_move_only_and_post_wal_build_free() {
    fn assert_send<T: Send>() {}
    assert_send::<PreparedU64HtoDPublication>();

    let source = include_str!("../resident_memory/prepared_u64_publish.rs");
    let publish = source
        .split("pub fn publish(self)")
        .nth(1)
        .and_then(|body| body.split("impl CudaResidentDeviceMemory").next())
        .expect("prepared publication is bounded before preparation");
    let prepare = source
        .split("pub fn prepare_u64_htod_publication")
        .nth(1)
        .and_then(|body| body.split("fn checked_u64_publication_destination").next())
        .expect("preparation is bounded before destination validation");

    assert!(source.contains("pub struct PreparedU64HtoDPublication"));
    assert!(source.contains("_allocation: Arc<CudaResidentDeviceAllocation>"));
    assert!(source.contains("value: [u8; U64_BYTES]"));
    assert!(source.contains("unsafe impl Send for PreparedU64HtoDPublication"));
    assert!(source.contains("byte_offset.is_multiple_of"));
    assert!(source.contains("validate_u64_publication_span(memory.metadata.allocated_bytes"));
    assert!(prepare.contains("Arc::clone(&allocation.primary)"));
    assert!(prepare.contains("get::<CuMemcpyHtoD>"));
    assert!(!source.contains("impl Clone for PreparedU64HtoDPublication"));

    for forbidden in [
        "Vec",
        "cached_function",
        ".get::<",
        "prepare_u64_htod_publication",
        "to_string",
        "collect",
    ] {
        assert!(
            !publish.contains(forbidden),
            "post-WAL u64 publication must not build {forbidden}"
        );
    }
    assert!(publish.contains("self.primary.set_current()?"));
    assert!(publish.contains("(self.cu_memcpy_htod)"));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_u64_htod_publication_pins_exact_allocation_and_writes_exact_value() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let owner = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &[0_u8; 16])
            .expect("resident publication destination"),
    );
    assert!(
        owner.prepare_u64_htod_publication(4, 1).is_err(),
        "unaligned u64 destination declines before publication"
    );
    assert!(
        owner.prepare_u64_htod_publication(16, 1).is_err(),
        "out-of-bounds u64 destination declines before publication"
    );
    let token = owner
        .prepare_u64_htod_publication(8, 0x1122_3344_5566_7788)
        .expect("pre-WAL u64 publication token");
    std::thread::spawn(move || token.publish())
        .join()
        .expect("publishing thread must not panic")
        .expect("already-resolved HtoD publication");
    assert_eq!(
        owner
            .read_resident_u64_column(8, 1)
            .expect("read exact published u64"),
        vec![0x1122_3344_5566_7788],
    );

    let token = owner
        .prepare_u64_htod_publication(0, 7)
        .expect("sabotaged token preparation");
    crate::fail_next_prepared_u64_htod_publication();
    assert!(
        token.publish().is_err(),
        "failed publication consumes its token"
    );
    assert_eq!(
        owner.read_resident_u64_column(0, 1).unwrap(),
        vec![0],
        "the injected synchronous failure writes no partial publication"
    );
    owner
        .prepare_u64_htod_publication(0, 9)
        .expect("fresh token after synchronous failure")
        .publish()
        .expect("retry needs a distinct pre-WAL token");
    assert_eq!(owner.read_resident_u64_column(0, 1).unwrap(), vec![9]);

    let wrong_context = owner.clone_with_primary_for_test(
        crate::cuda_context::distinct_gpu_primary_context_for_test(0)
            .expect("distinct wrapper context"),
    );
    wrong_context
        .prepare_u64_htod_publication(0, 11)
        .expect("token must use the allocation's primary context")
        .publish()
        .expect("exact allocation primary remains publishable");
    assert_eq!(owner.read_resident_u64_column(0, 1).unwrap(), vec![11]);

    let lifetime_owner = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &[0_u8; 8])
            .expect("lifetime-pinned publication destination"),
    );
    let allocation = lifetime_owner.allocation_weak_for_test();
    let token = lifetime_owner
        .prepare_u64_htod_publication(0, 13)
        .expect("lifetime-pinned token");
    drop(lifetime_owner);
    assert!(
        allocation.is_alive(),
        "the prepared publication retains the exact resident allocation"
    );
    token
        .publish()
        .expect("publication remains valid after outer owner retirement");
    assert!(
        !allocation.is_alive(),
        "consuming the final token releases its exact allocation pin"
    );
}
