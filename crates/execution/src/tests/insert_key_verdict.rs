use super::*;

fn fixed_i32_payload(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn gpu_runtime() -> Option<CudaDriverRuntime> {
    let runtime = CudaDriverRuntime::probe().ok()?;
    runtime.snapshot().driver_available.then_some(runtime)
}

#[test]
fn insert_key_verdict_scratch_is_exact_bucketed_and_overflow_safe() {
    assert_eq!(insert_batch_key_verdict_scratch_bytes(0, 1), Some(1_024));
    assert_eq!(insert_batch_key_verdict_scratch_bytes(1, 1), Some(1_024));
    assert_eq!(insert_batch_key_verdict_scratch_bytes(17, 3), Some(1_280));
    assert_eq!(insert_batch_key_verdict_scratch_bytes(17, 9), Some(1_536));
    assert!(insert_batch_key_verdict_scratch_bytes(usize::MAX, 1).is_none());
    assert!(insert_batch_key_verdict_scratch_bytes(1, usize::MAX).is_none());
}

#[test]
fn insert_key_verdict_descriptor_contract_rejects_bad_order_width_and_extents_before_cuda() {
    use crate::insert_key_verdict::validate_insert_batch_key_descriptors;

    let fixed = CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    };
    let validity = CudaCompoundFoldColumn::Validity {
        bitmap_byte_offset: 16,
    };
    assert!(validate_insert_batch_key_descriptors(32, 1, &[fixed, validity], 4).is_ok());
    assert!(validate_insert_batch_key_descriptors(32, 1, &[], 4).is_err());
    assert!(validate_insert_batch_key_descriptors(
        32,
        1,
        &[
            validity,
            CudaCompoundFoldColumn::Bool {
                bitmap_byte_offset: 20,
            },
        ],
        4,
    )
    .is_err());
    assert!(validate_insert_batch_key_descriptors(
        32,
        1,
        &[CudaCompoundFoldColumn::Fixed {
            byte_offset: 0,
            width_words: 3,
        }],
        4,
    )
    .is_err());
    assert!(validate_insert_batch_key_descriptors(
        32,
        1,
        &[CudaCompoundFoldColumn::Fixed {
            byte_offset: 1,
            width_words: 1,
        }],
        4,
    )
    .is_err());
    assert!(validate_insert_batch_key_descriptors(
        32,
        1,
        &[CudaCompoundFoldColumn::Text {
            offsets_byte_offset: 4,
            bytes_byte_offset: 24,
            bytes_len: 8,
        }],
        2,
    )
    .is_err());
    assert!(validate_insert_batch_key_descriptors(
        16,
        1,
        &[CudaCompoundFoldColumn::Fixed {
            byte_offset: 4,
            width_words: 4,
        }],
        2,
    )
    .is_err());
    assert!(validate_insert_batch_key_descriptors(
        16,
        1,
        &[CudaCompoundFoldColumn::Bool {
            bitmap_byte_offset: 16,
        }],
        1,
    )
    .is_err());
    assert!(validate_insert_batch_key_descriptors(
        24,
        1,
        &[CudaCompoundFoldColumn::Text {
            offsets_byte_offset: 0,
            bytes_byte_offset: 16,
            bytes_len: 9,
        }],
        1,
    )
    .is_err());
    assert!(validate_insert_batch_key_descriptors(64, 1, &[fixed, validity, validity], 4).is_err());
    assert!(validate_insert_batch_key_descriptors(
        64,
        1,
        &[
            fixed,
            CudaCompoundFoldColumn::Validity {
                bitmap_byte_offset: 16,
            },
            CudaCompoundFoldColumn::Validity {
                bitmap_byte_offset: 17,
            },
        ],
        4,
    )
    .is_err());
}

#[test]
fn insert_key_verdict_static_contract_keeps_the_terminal_bounded_and_two_pass() {
    let source = include_str!("../insert_key_verdict.rs");
    let ptx = include_str!("../insert_key_verdict.ptx");
    assert_eq!(source.matches("memcpy_dtoh(").count(), 1);
    assert!(source.contains("INSERT_BATCH_KEY_VERDICT_READBACK_BYTES"));
    assert!(source.contains("launch_verdict(0)?"));
    assert!(source.contains("launch_verdict(1)?"));
    assert!(source.contains("NullStreamDrain"));
    let drain = source
        .find("let mut stream_drain = NullStreamDrain")
        .expect("null-stream drain is constructed");
    let first_queued_initialization = source
        .find("check_cuda(unsafe { memset(directory.ptr")
        .expect("directory initialization is queued");
    let failpoint = source
        .rfind("if take_fail_after_initialization()")
        .expect("post-initialization failpoint exists");
    let module_lookup = source
        .find("primary.cached_function(c\"gpu_db_insert_batch_key_verdict\"")
        .expect("PTX lookup exists");
    assert!(
        drain < first_queued_initialization,
        "pooled-buffer drain must arm before the first queued initialization"
    );
    assert!(
        first_queued_initialization < failpoint && failpoint < module_lookup,
        "failpoint must exercise the initialized-but-not-launched lifetime gap"
    );
    for forbidden in [
        "Vec<i32>",
        "submit_compound_fold_fingerprints",
        "predicate_mask_indices",
        "resident_index",
        "launch_cuda_group",
        "join_",
    ] {
        assert!(
            !source.contains(forbidden),
            "verdict path reused {forbidden}"
        );
    }
    for required in [
        "EXACT_COLUMN_LOOP",
        "PROBE_EXHAUSTED",
        "MISSING_REPRESENTATIVE",
        "NULL_CAS_MIN_LOOP",
        "atom.global.cas.b64",
        "atom.global.min.u32",
        "atom.global.or.b32",
        "mul.wide.u32 %rd7, %r6, %r7",
        "mul.wide.u32 %rd10, %r8, %r7",
    ] {
        assert!(ptx.contains(required), "PTX lacks {required}");
    }
    for forbidden in ["mad.lo.u32", "setp.ge.u32 %p1, %r", "add.u32 %r6, %r6, %r7"] {
        assert!(
            !ptx.contains(forbidden),
            "PTX keeps wrapping grid arithmetic: {forbidden}"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_batch_key_verdict_preserves_nulls_distinct_and_validity_tails() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let mut payload = fixed_i32_payload(&[0, 0, 7, 7]);
    let validity_offset = payload.len() as u64;
    payload.push(0b1100);
    let resident = runtime
        .retain_device_memory_copy(0, &payload)
        .expect("resident nulls-distinct payload");
    let columns = [
        CudaCompoundFoldColumn::Fixed {
            byte_offset: 0,
            width_words: 1,
        },
        CudaCompoundFoldColumn::Validity {
            bitmap_byte_offset: validity_offset,
        },
    ];
    let verdict = resident
        .insert_batch_key_verdict_from_payload(&columns, 4)
        .expect("nulls distinct verdict");
    assert_eq!(
        verdict,
        CudaInsertBatchKeyVerdict {
            first_null: Some(CudaInsertBatchNull {
                row: 0,
                validity_ordinal: 0,
            }),
            first_duplicate_row: Some(3),
            readback_bytes: INSERT_BATCH_KEY_VERDICT_READBACK_BYTES,
        }
    );

    let mut rank_payload = fixed_i32_payload(&[1, 2]);
    let second_key_offset = rank_payload.len() as u64;
    rank_payload.extend_from_slice(&fixed_i32_payload(&[10, 20]));
    let first_validity = rank_payload.len() as u64;
    rank_payload.push(0b0001); // row 1 is NULL through suffix ordinal 0.
    let second_validity = rank_payload.len() as u64;
    rank_payload.push(0b0000); // row 0 is NULL through suffix ordinal 1 and wins row-major.
    let rank = runtime
        .retain_device_memory_copy(0, &rank_payload)
        .expect("null-rank payload");
    assert_eq!(
        rank.insert_batch_key_verdict_from_payload(
            &[
                CudaCompoundFoldColumn::Fixed {
                    byte_offset: 0,
                    width_words: 1,
                },
                CudaCompoundFoldColumn::Fixed {
                    byte_offset: second_key_offset,
                    width_words: 1,
                },
                CudaCompoundFoldColumn::Validity {
                    bitmap_byte_offset: first_validity,
                },
                CudaCompoundFoldColumn::Validity {
                    bitmap_byte_offset: second_validity,
                },
            ],
            2,
        )
        .expect("first null rank"),
        CudaInsertBatchKeyVerdict {
            first_null: Some(CudaInsertBatchNull {
                row: 0,
                validity_ordinal: 1,
            }),
            first_duplicate_row: None,
            readback_bytes: 16,
        }
    );

    for row_count in [31_u32, 32, 33] {
        let mut tail_payload = fixed_i32_payload(&(0..row_count as i32).collect::<Vec<_>>());
        let tail_offset = tail_payload.len() as u64;
        let mut bitmap = vec![u8::MAX; row_count.div_ceil(8) as usize];
        let null_row = row_count - 1;
        bitmap[null_row as usize / 8] &= !(1 << (null_row % 8));
        tail_payload.extend_from_slice(&bitmap);
        let tail = runtime
            .retain_device_memory_copy(0, &tail_payload)
            .expect("validity-tail payload");
        let result = tail
            .insert_batch_key_verdict_from_payload(
                &[
                    CudaCompoundFoldColumn::Fixed {
                        byte_offset: 0,
                        width_words: 1,
                    },
                    CudaCompoundFoldColumn::Validity {
                        bitmap_byte_offset: tail_offset,
                    },
                ],
                row_count,
            )
            .expect("tail verdict");
        assert_eq!(
            result.first_null,
            Some(CudaInsertBatchNull {
                row: null_row,
                validity_ordinal: 0,
            })
        );
        assert_eq!(result.first_duplicate_row, None);
        assert_eq!(result.readback_bytes, 16);
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_batch_key_verdict_handles_mixed_types_repeated_descriptors_and_collisions() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let mut payload = fixed_i32_payload(&[0, 0, 0, 0]);
    let wide_offset = payload.len() as u64;
    for value in [5_i64; 4] {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let words4_offset = payload.len() as u64;
    for _ in 0..4 {
        for word in [11_u32, 12, 13, 14] {
            payload.extend_from_slice(&word.to_le_bytes());
        }
    }
    let bool_offset = payload.len() as u64;
    payload.push(0b0100); // row 2 differs; rows 0/1/3 are false.
    payload.resize(120, 0);
    let text_offsets = payload.len() as u64;
    for _ in 0..5 {
        payload.extend_from_slice(&0_u64.to_le_bytes());
    }
    let text_bytes = payload.len() as u64;
    let validity_offset = payload.len() as u64;
    payload.push(0b1101); // row 1 is NULL; valid zero/false/empty rows remain complete keys.
    let resident = runtime
        .retain_device_memory_copy(0, &payload)
        .expect("mixed resident payload");
    let mixed = [
        CudaCompoundFoldColumn::Fixed {
            byte_offset: 0,
            width_words: 1,
        },
        CudaCompoundFoldColumn::Fixed {
            byte_offset: wide_offset,
            width_words: 2,
        },
        CudaCompoundFoldColumn::Fixed {
            byte_offset: words4_offset,
            width_words: 4,
        },
        CudaCompoundFoldColumn::Bool {
            bitmap_byte_offset: bool_offset,
        },
        CudaCompoundFoldColumn::Text {
            offsets_byte_offset: text_offsets,
            bytes_byte_offset: text_bytes,
            bytes_len: 0,
        },
        CudaCompoundFoldColumn::Validity {
            bitmap_byte_offset: validity_offset,
        },
    ];
    assert_eq!(
        resident
            .insert_batch_key_verdict_from_payload(&mixed, 4)
            .expect("mixed verdict"),
        CudaInsertBatchKeyVerdict {
            first_null: Some(CudaInsertBatchNull {
                row: 1,
                validity_ordinal: 0,
            }),
            first_duplicate_row: Some(3),
            readback_bytes: 16,
        }
    );

    let mut repeated_payload = fixed_i32_payload(&[5, 5, 9]);
    let repeated_validity_offset = repeated_payload.len() as u64;
    repeated_payload.push(0b0111);
    let repeated = runtime
        .retain_device_memory_copy(0, &repeated_payload)
        .expect("repeated descriptor payload");
    let repeated_columns = [
        CudaCompoundFoldColumn::Fixed {
            byte_offset: 0,
            width_words: 1,
        },
        CudaCompoundFoldColumn::Fixed {
            byte_offset: 0,
            width_words: 1,
        },
        CudaCompoundFoldColumn::Validity {
            bitmap_byte_offset: repeated_validity_offset,
        },
    ];
    for _ in 0..8 {
        assert_eq!(
            repeated
                .insert_batch_key_verdict_from_payload(&repeated_columns, 3)
                .expect("deterministic second occurrence")
                .first_duplicate_row,
            Some(1)
        );
    }

    let (first, second) = colliding_bigints();
    let collision_payload = [first, second, first]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let collision = runtime
        .retain_device_memory_copy(0, &collision_payload)
        .expect("collision payload");
    let collision_columns = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 2,
    }];
    assert_eq!(
        collision
            .insert_batch_key_verdict_from_payload(&collision_columns, 3)
            .expect("repeat-first exact collision recheck")
            .first_duplicate_row,
        Some(2)
    );
    let no_duplicate = runtime
        .retain_device_memory_copy(
            0,
            &[first, second]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .expect("distinct collision pair");
    assert_eq!(
        no_duplicate
            .insert_batch_key_verdict_from_payload(&collision_columns, 2)
            .expect("distinct collision pair recheck")
            .first_duplicate_row,
        None
    );
    let repeat_second = runtime
        .retain_device_memory_copy(
            0,
            &[first, second, second]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .expect("repeat-second collision pair");
    assert_eq!(
        repeat_second
            .insert_batch_key_verdict_from_payload(&collision_columns, 3)
            .expect("repeat-second exact collision recheck")
            .first_duplicate_row,
        Some(2)
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_batch_key_verdict_fails_closed_on_bad_text_and_honors_exact_scope() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let mut malformed = Vec::new();
    malformed.extend_from_slice(&0_u64.to_le_bytes());
    malformed.extend_from_slice(&2_u64.to_le_bytes());
    malformed.push(b'x');
    let malformed = runtime
        .retain_device_memory_copy(0, &malformed)
        .expect("malformed text payload");
    assert!(malformed
        .insert_batch_key_verdict_from_payload(
            &[CudaCompoundFoldColumn::Text {
                offsets_byte_offset: 0,
                bytes_byte_offset: 16,
                bytes_len: 1,
            }],
            1,
        )
        .is_err());

    let normal = runtime
        .retain_device_memory_copy(0, &fixed_i32_payload(&[1, 2]))
        .expect("context reuse payload");
    let columns = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let required = insert_batch_key_verdict_scratch_bytes(2, columns.len()).unwrap();
    let exact_scope = CudaAllocationScope::with_budget(required);
    assert_eq!(
        normal
            .insert_batch_key_verdict_from_payload(&columns, 2)
            .expect("context reuse and exact scope"),
        CudaInsertBatchKeyVerdict {
            first_null: None,
            first_duplicate_row: None,
            readback_bytes: 16,
        }
    );
    assert_eq!(exact_scope.peak_bytes(), required);
    drop(exact_scope);

    let too_small = CudaAllocationScope::with_budget(required - 1);
    assert!(matches!(
        normal.insert_batch_key_verdict_from_payload(&columns, 2),
        Err(CudaRuntimeProbeError::AllocationBudgetExceeded { .. })
    ));
    assert_eq!(too_small.peak_bytes(), 0);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_batch_key_verdict_failpoint_drains_initialized_pool_leases_before_reuse() {
    use crate::insert_key_verdict::fail_next_insert_key_verdict_after_initialization;

    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let resident = runtime
        .retain_device_memory_copy(0, &fixed_i32_payload(&[4, 4, 9]))
        .expect("resident failpoint payload");
    let columns = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    fail_next_insert_key_verdict_after_initialization();
    assert!(matches!(
        resident.insert_batch_key_verdict_from_payload(&columns, 3),
        Err(CudaRuntimeProbeError::KernelLaunchFailed(-1))
    ));
    // The injected return happens after all HtoD/memset setup but before module lookup/launch.
    // A successful immediate repeat proves the default-stream drain preserved both pool ownership
    // and context usability; it must not surface a CUDA 700/716/717 fault.
    assert_eq!(
        resident
            .insert_batch_key_verdict_from_payload(&columns, 3)
            .expect("initialized pooled buffers and context are reusable"),
        CudaInsertBatchKeyVerdict {
            first_null: None,
            first_duplicate_row: Some(1),
            readback_bytes: 16,
        }
    );
}

fn compound_fold(words: &[i32]) -> u32 {
    words.iter().fold(0x811C_9DC5_u32, |hash, word| {
        (hash ^ *word as u32)
            .wrapping_mul(0x0100_0193)
            .rotate_left(13)
            .wrapping_add(0x9E37_79B1)
    })
}

fn colliding_bigints() -> (i64, i64) {
    let step = |hash: u32, word: u32| {
        (hash ^ word)
            .wrapping_mul(0x0100_0193)
            .rotate_left(13)
            .wrapping_add(0x9E37_79B1)
    };
    let mut buckets = std::collections::HashMap::<u32, (u32, u32)>::new();
    let pair = (10_000_u32..2_000_000).find_map(|low| {
        let state = step(0x811C_9DC5, low);
        buckets
            .insert(state >> 20, (low, state))
            .map(|(prior_low, prior_state)| {
                let high_a = 1_u32 << 20;
                let high_b = high_a ^ (prior_state ^ state);
                let first = ((u64::from(high_a) << 32) | u64::from(prior_low)) as i64;
                let second = ((u64::from(high_b) << 32) | u64::from(low)) as i64;
                (first, second)
            })
    });
    let (first, second) = pair.expect("construct BIGINT fold collision");
    assert_ne!(first, second);
    assert_eq!(
        compound_fold(&[first as i32, (first >> 32) as i32]),
        compound_fold(&[second as i32, (second >> 32) as i32])
    );
    (first, second)
}
