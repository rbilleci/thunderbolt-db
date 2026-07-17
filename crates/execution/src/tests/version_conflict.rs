#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_exact_mask_version_conflict_is_fixed_size_and_accepts_four_mod_eight_sidecars() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let values = [10_i32, 20, 30, 40];
    let mut payload = Vec::with_capacity(values.len() * 4);
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let resident = runtime
        .retain_device_memory_copy(0, &payload)
        .expect("resident predicate source");

    // Both u64 sidecars begin at 4 mod 8. The terminal must use two u32 loads; an ld.u64 here is
    // the CUDA-716 regression control. Row 1 conflicts by creation and row 2 by deletion. Every
    // other delete stamp is the live sentinel, which must not conflict despite being very large.
    const OFFSET: u64 = 4;
    const LIVE: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    let encode = |stamps: [u64; 4]| {
        let mut bytes = vec![0xA5; OFFSET as usize];
        for stamp in stamps {
            bytes.extend_from_slice(&stamp.to_le_bytes());
        }
        bytes
    };
    let created = runtime
        .retain_device_memory_copy(0, &encode([0, 9, 0, 0]))
        .expect("created sidecar");
    let deleted = runtime
        .retain_device_memory_copy(0, &encode([LIVE, LIVE, 11, LIVE]))
        .expect("deleted sidecar");
    let mask_for = |needle| {
        resident
            .run_expr_predicate_mask_with_text(
                &[
                    ExprStep::LoadColumn { byte_offset: 0 },
                    ExprStep::CompareScalar {
                        cmp: 0,
                        scalar: needle,
                        scalar_on_left: false,
                    },
                ],
                &[],
                4,
                ResidentElemType::I32,
            )
            .expect("exact mask")
    };

    for (needle, expected) in [(20, true), (30, true), (40, false)] {
        let verdict = resident
            .predicate_mask_version_conflict(
                &mask_for(needle),
                Some((&created, OFFSET)),
                0,
                Some((&deleted, OFFSET)),
                LIVE,
                LIVE,
                8,
            )
            .expect("device history verdict");
        assert_eq!(verdict.conflict, expected, "needle {needle}");
        assert_eq!(
            verdict.readback_bytes,
            std::mem::size_of::<u32>(),
            "the D2H contract is one fixed scalar"
        );
    }
}
