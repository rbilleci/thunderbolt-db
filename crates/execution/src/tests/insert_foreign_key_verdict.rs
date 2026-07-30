use super::*;

fn i32_payload(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u64_payload(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn push_u64s(payload: &mut Vec<u8>, values: &[u64]) {
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
}

fn fold_word(hash: u32, word: u32) -> u32 {
    (hash ^ word)
        .wrapping_mul(16_777_619)
        .rotate_left(13)
        .wrapping_add(2_654_435_761)
}

fn inverse_fold_word(output: u32, prior: u32) -> u32 {
    // 0x359c449b is 16777619^-1 modulo 2^32.
    prior
        ^ output
            .wrapping_sub(2_654_435_761)
            .rotate_right(13)
            .wrapping_mul(0x359c_449b)
}

fn gpu_runtime() -> Option<CudaDriverRuntime> {
    let runtime = CudaDriverRuntime::probe().ok()?;
    runtime.snapshot().driver_available.then_some(runtime)
}

fn plain_parent<'a>(
    payload: &'a CudaResidentDeviceMemory,
    columns: &'a [CudaCompoundFoldColumn],
    row_count: u32,
) -> CudaInsertForeignKeyParentShard<'a> {
    let deleted_live = u64::from_le_bytes([0x7f; 8]);
    CudaInsertForeignKeyParentShard {
        payload,
        columns,
        row_count,
        created_by: None,
        created_default: 0,
        deleted_by: None,
        deleted_default: deleted_live,
        deleted_live,
    }
}

#[test]
fn insert_foreign_key_verdict_scratch_is_exact_bucketed_and_overflow_safe() {
    assert_eq!(
        insert_foreign_key_verdict_scratch_bytes(0, 1, 1, None),
        Some(0)
    );
    assert_eq!(
        insert_foreign_key_verdict_scratch_bytes(0, usize::MAX, usize::MAX, Some(usize::MAX)),
        Some(0)
    );
    assert_eq!(
        insert_foreign_key_verdict_scratch_bytes(1, 1, 1, None),
        Some(1_536)
    );
    assert_eq!(
        insert_foreign_key_verdict_scratch_bytes(17, 2, 3, Some(2)),
        Some(2_048)
    );
    assert_eq!(
        insert_foreign_key_verdict_scratch_bytes(17, 2, 0, Some(2)),
        Some(1_792)
    );
    assert!(insert_foreign_key_verdict_scratch_bytes(usize::MAX, 1, 1, None).is_none());
    assert!(insert_foreign_key_verdict_scratch_bytes(1, usize::MAX, 1, None).is_none());
    assert!(insert_foreign_key_verdict_scratch_bytes(1, 1, usize::MAX, None).is_none());
    assert!(insert_foreign_key_verdict_scratch_bytes(1, 1, 1, Some(usize::MAX)).is_none());
    assert_eq!(
        insert_foreign_key_verdict_scratch_bytes(257, 1, 1, None),
        Some(17_152),
        "both u32 match buckets must charge P(4S), not byte flags"
    );
}

#[test]
fn insert_foreign_key_verdict_rejects_u32_grid_stride_wrap_before_cuda() {
    use crate::insert_foreign_key_verdict::{
        foreign_key_grid_stride_is_safe, FOREIGN_KEY_MAX_SAFE_ROWS,
    };

    assert!(foreign_key_grid_stride_is_safe(0));
    assert!(foreign_key_grid_stride_is_safe(FOREIGN_KEY_MAX_SAFE_ROWS));
    assert!(!foreign_key_grid_stride_is_safe(
        FOREIGN_KEY_MAX_SAFE_ROWS + 1
    ));
    assert!(!foreign_key_grid_stride_is_safe(u32::MAX));
}

#[test]
fn insert_foreign_key_verdict_static_contract_is_execution_only_and_terminal_bounded() {
    let source = include_str!("../insert_foreign_key_verdict.rs");
    let ptx = include_str!("../insert_foreign_key_verdict.ptx");
    assert_eq!(source.matches("memcpy_dtoh(").count(), 1);
    assert!(source.contains("validate_insert_batch_key_descriptors"));
    assert!(source.contains("same_key_data_layout"));
    assert!(source.contains("sidecar_ptr"));
    assert!(source.contains("CudaAllocationScope::ensure_available"));
    assert!(source.contains("NullStreamDrain"));
    assert!(source.contains("fail_next_insert_foreign_key_verdict_after_initialization"));
    assert!(source.contains("fail_next_insert_foreign_key_verdict_after_first_parent_launch"));
    assert!(source.contains("parent_shards"));
    assert!(source.contains("self_provider"));
    assert!(source.contains("FOREIGN_KEY_MAX_SAFE_ROWS"));
    assert!(source.contains("require_grid_stride_safe(child_row_count)"));
    assert!(!source.contains("provider.payload"));
    assert!(!source.contains("provider.row_count"));
    for forbidden in [
        "crate::engine",
        "wal_",
        "publication",
        "apply_",
        "execute_text",
        "host_rows",
    ] {
        assert!(
            !source.contains(forbidden),
            "execution leaf contains {forbidden}"
        );
    }
    for required in [
        "BUILD_PROBE",
        "PARENT_PROBE",
        "FINAL_PROBE",
        "SELF_MARK",
        "atom.global.cas.b64",
        "atom.global.or.b32",
        "atom.global.min.u32",
        "BAD_VERSION",
        "BAD_TEXT",
    ] {
        assert!(ptx.contains(required), "PTX lacks {required}");
    }
    assert!(
        !ptx.contains("st.global.u8"),
        "the FK scratch contains no byte-addressed match writes"
    );
    assert!(ptx.contains("ld.global.u32 %r26, [%rd40]"));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_foreign_key_verdict_handles_empty_duplicate_null_and_self_provider() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let columns = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let empty = runtime
        .retain_device_memory_copy(0, &[0; 4])
        .expect("empty child backing");
    assert_eq!(
        empty
            .insert_foreign_key_verdict_against_shards(&columns, 0, &[], None, 1, 0)
            .expect("empty verdict"),
        CudaInsertForeignKeyVerdict {
            first_missing_row: None,
            first_history_row: None,
            readback_bytes: 0,
        }
    );

    let child = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7, 7, 8]))
        .expect("duplicate child payload");
    let parent = runtime
        .retain_device_memory_copy(0, &i32_payload(&[8, 7]))
        .expect("parent payload");
    let parents = [plain_parent(&parent, &columns, 2)];
    assert_eq!(
        child
            .insert_foreign_key_verdict_against_shards(&columns, 3, &parents, None, 1, 0)
            .expect("duplicate children share their stable representative"),
        CudaInsertForeignKeyVerdict {
            first_missing_row: None,
            first_history_row: None,
            readback_bytes: INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES,
        }
    );
    let duplicate_self = CudaInsertForeignKeySelfProvider { columns: &columns };
    assert_eq!(
        child
            .insert_foreign_key_verdict_against_shards(
                &columns,
                3,
                &[],
                Some(duplicate_self),
                1,
                0,
            )
            .expect("self provider marks both worlds"),
        CudaInsertForeignKeyVerdict {
            first_missing_row: None,
            first_history_row: None,
            readback_bytes: INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES,
        }
    );

    let mut self_payload = i32_payload(&[7, 8]);
    let self_provider_offset = self_payload.len() as u64;
    self_payload.extend_from_slice(&i32_payload(&[8, 7]));
    let self_child = runtime
        .retain_device_memory_copy(0, &self_payload)
        .expect("self provider shares child allocation");
    let self_provider_columns = [CudaCompoundFoldColumn::Fixed {
        byte_offset: self_provider_offset,
        width_words: 1,
    }];
    assert_eq!(
        self_child
            .insert_foreign_key_verdict_against_shards(
                &columns,
                2,
                &[],
                Some(CudaInsertForeignKeySelfProvider {
                    columns: &self_provider_columns,
                }),
                1,
                0,
            )
            .expect("self provider may select a distinct referenced child column"),
        CudaInsertForeignKeyVerdict {
            first_missing_row: None,
            first_history_row: None,
            readback_bytes: INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES,
        }
    );

    let mut null_payload = i32_payload(&[7, 9]);
    let validity_offset = null_payload.len() as u64;
    null_payload.push(0b01); // row 1 is a SQL NULL child and therefore satisfied.
    let null_child = runtime
        .retain_device_memory_copy(0, &null_payload)
        .expect("nullable child payload");
    let null_columns = [
        CudaCompoundFoldColumn::Fixed {
            byte_offset: 0,
            width_words: 1,
        },
        CudaCompoundFoldColumn::Validity {
            bitmap_byte_offset: validity_offset,
        },
    ];
    let null_parent = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7]))
        .expect("nullable parent payload");
    let null_parents = [plain_parent(&null_parent, &columns, 1)];
    assert_eq!(
        null_child
            .insert_foreign_key_verdict_against_shards(&null_columns, 2, &null_parents, None, 1, 0,)
            .expect("MATCH SIMPLE NULL child"),
        CudaInsertForeignKeyVerdict {
            first_missing_row: None,
            first_history_row: None,
            readback_bytes: INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES,
        }
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_foreign_key_verdict_stabilizes_duplicate_misses_and_exact_rechecks_collisions() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let i32_columns = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let duplicate_missing = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7, 9, 9, 9]))
        .expect("duplicate missing child keys");
    let provider = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7]))
        .expect("only non-missing provider");
    assert_eq!(
        duplicate_missing
            .insert_foreign_key_verdict_against_shards(
                &i32_columns,
                4,
                &[plain_parent(&provider, &i32_columns, 1)],
                None,
                1,
                0,
            )
            .expect("stable duplicate miss representative"),
        CudaInsertForeignKeyVerdict {
            first_missing_row: Some(1),
            first_history_row: None,
            readback_bytes: INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES,
        }
    );

    let duplicate_provider_child = runtime
        .retain_device_memory_copy(0, &i32_payload(&[11]))
        .expect("child with contended providers");
    let provider_a = runtime
        .retain_device_memory_copy(0, &i32_payload(&[11, 11, 11]))
        .expect("same-shard duplicate providers");
    let provider_b = runtime
        .retain_device_memory_copy(0, &i32_payload(&[11, 11]))
        .expect("cross-shard duplicate providers");
    let providers = [
        plain_parent(&provider_a, &i32_columns, 3),
        plain_parent(&provider_b, &i32_columns, 2),
    ];
    assert_eq!(
        duplicate_provider_child
            .insert_foreign_key_verdict_against_shards(&i32_columns, 1, &providers, None, 1, 0)
            .expect("duplicate providers all mark the same match bucket"),
        CudaInsertForeignKeyVerdict {
            first_missing_row: None,
            first_history_row: None,
            readback_bytes: INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES,
        }
    );

    let first = (11_u32, 29_u32);
    let second_first = 41_u32;
    let target = fold_word(fold_word(2_166_136_261, first.0), first.1);
    let second_second = inverse_fold_word(target, fold_word(2_166_136_261, second_first));
    let second = (second_first, second_second);
    assert_ne!(first, second);
    assert_eq!(
        target,
        fold_word(fold_word(2_166_136_261, second.0), second.1),
        "the test pair must collide in the exact directory fingerprint"
    );
    let collision_child = runtime
        .retain_device_memory_copy(
            0,
            &[first, second]
                .into_iter()
                .flat_map(|(left, right)| [left, right])
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .expect("two-word collision child");
    let collision_parent = runtime
        .retain_device_memory_copy(
            0,
            &[second]
                .into_iter()
                .flat_map(|(left, right)| [left, right])
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .expect("true collision provider only");
    let wide = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 2,
    }];
    assert_eq!(
        collision_child
            .insert_foreign_key_verdict_against_shards(
                &wide,
                2,
                &[plain_parent(&collision_parent, &wide, 1)],
                None,
                1,
                0,
            )
            .expect("fingerprint collision exact recheck"),
        CudaInsertForeignKeyVerdict {
            first_missing_row: Some(0),
            first_history_row: None,
            readback_bytes: INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES,
        },
        "only the exactly equal two-word child value may be provided"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_foreign_key_verdict_handles_fixed_bool_text_and_multiple_parent_shards() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let fixed = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 2,
    }];
    let fixed_child = runtime
        .retain_device_memory_copy(0, &u64_payload(&[11, 29]))
        .expect("int8 child");
    let fixed_a = runtime
        .retain_device_memory_copy(0, &u64_payload(&[29]))
        .expect("first int8 parent shard");
    let fixed_b = runtime
        .retain_device_memory_copy(0, &u64_payload(&[11]))
        .expect("second int8 parent shard");
    let fixed_parents = [
        plain_parent(&fixed_a, &fixed, 1),
        plain_parent(&fixed_b, &fixed, 1),
    ];
    assert!(fixed_child
        .insert_foreign_key_verdict_against_shards(&fixed, 2, &fixed_parents, None, 1, 0)
        .expect("fixed multi-shard FK")
        .first_missing_row
        .is_none());

    let bool_columns = [CudaCompoundFoldColumn::Bool {
        bitmap_byte_offset: 0,
    }];
    let bool_child = runtime
        .retain_device_memory_copy(0, &[0b01])
        .expect("bool child");
    let bool_parent = runtime
        .retain_device_memory_copy(0, &[0b01])
        .expect("bool parent");
    assert!(bool_child
        .insert_foreign_key_verdict_against_shards(
            &bool_columns,
            1,
            &[plain_parent(&bool_parent, &bool_columns, 1)],
            None,
            1,
            0,
        )
        .expect("bool FK")
        .first_missing_row
        .is_none());

    let mut text_payload = Vec::new();
    push_u64s(&mut text_payload, &[0, 1, 2]);
    text_payload.extend_from_slice(b"ab");
    let text_columns = [CudaCompoundFoldColumn::Text {
        offsets_byte_offset: 0,
        bytes_byte_offset: 24,
        bytes_len: 2,
    }];
    let text_child = runtime
        .retain_device_memory_copy(0, &text_payload)
        .expect("text child");
    let text_parent = runtime
        .retain_device_memory_copy(0, &text_payload)
        .expect("text parent");
    assert!(text_child
        .insert_foreign_key_verdict_against_shards(
            &text_columns,
            2,
            &[plain_parent(&text_parent, &text_columns, 2)],
            None,
            1,
            0,
        )
        .expect("text FK")
        .first_missing_row
        .is_none());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_foreign_key_verdict_preserves_current_original_truth_table() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let columns = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let child = runtime
        .retain_device_memory_copy(0, &i32_payload(&[1, 2, 3, 4]))
        .expect("child");
    let parent = runtime
        .retain_device_memory_copy(0, &i32_payload(&[1, 2, 3]))
        .expect("parent");
    let created = runtime
        .retain_device_memory_copy(0, &u64_payload(&[0, 9, 0]))
        .expect("created sidecar");
    let deleted_live = u64::from_le_bytes([0x7f; 8]);
    let deleted = runtime
        .retain_device_memory_copy(0, &u64_payload(&[deleted_live, deleted_live, 8]))
        .expect("deleted sidecar");
    let parents = [CudaInsertForeignKeyParentShard {
        payload: &parent,
        columns: &columns,
        row_count: 3,
        created_by: Some(CudaInsertResidentKeySidecar {
            memory: &created,
            byte_offset: 0,
        }),
        created_default: 0,
        deleted_by: Some(CudaInsertResidentKeySidecar {
            memory: &deleted,
            byte_offset: 0,
        }),
        deleted_default: deleted_live,
        deleted_live,
    }];
    assert_eq!(
        child
            .insert_foreign_key_verdict_against_shards(&columns, 4, &parents, None, 10, 5)
            .expect("current/original FK truth table"),
        CudaInsertForeignKeyVerdict {
            first_missing_row: Some(3),
            first_history_row: Some(1),
            readback_bytes: INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES,
        }
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_foreign_key_verdict_fails_closed_on_malformed_contracts_and_budget() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let fixed = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let child = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7]))
        .expect("child");
    let bool_parent = runtime
        .retain_device_memory_copy(0, &[1])
        .expect("bool parent");
    let bool_columns = [CudaCompoundFoldColumn::Bool {
        bitmap_byte_offset: 0,
    }];
    assert!(child
        .insert_foreign_key_verdict_against_shards(
            &fixed,
            1,
            &[plain_parent(&bool_parent, &bool_columns, 1)],
            None,
            1,
            0,
        )
        .is_err());
    assert!(child
        .insert_foreign_key_verdict_against_shards(
            &fixed,
            u32::MAX,
            &[plain_parent(&bool_parent, &fixed, 1)],
            None,
            1,
            0,
        )
        .is_err());
    let unsafe_rows = crate::insert_foreign_key_verdict::FOREIGN_KEY_MAX_SAFE_ROWS + 1;
    let boundary_scope = CudaAllocationScope::with_budget(0);
    assert!(child
        .insert_foreign_key_verdict_against_shards(
            &fixed,
            unsafe_rows,
            &[plain_parent(&bool_parent, &fixed, 1)],
            Some(CudaInsertForeignKeySelfProvider { columns: &fixed }),
            1,
            0,
        )
        .is_err());
    assert!(child
        .insert_foreign_key_verdict_against_shards(
            &fixed,
            1,
            &[plain_parent(&bool_parent, &fixed, unsafe_rows)],
            None,
            1,
            0,
        )
        .is_err());
    assert_eq!(boundary_scope.peak_bytes(), 0);
    drop(boundary_scope);

    let mut malformed_text_payload = Vec::new();
    push_u64s(&mut malformed_text_payload, &[0, 2]);
    malformed_text_payload.push(b'x');
    let malformed_text = runtime
        .retain_device_memory_copy(0, &malformed_text_payload)
        .expect("malformed text payload");
    let text = [CudaCompoundFoldColumn::Text {
        offsets_byte_offset: 0,
        bytes_byte_offset: 16,
        bytes_len: 1,
    }];
    assert!(malformed_text
        .insert_foreign_key_verdict_against_shards(&text, 1, &[], None, 1, 0)
        .is_err());

    let parent = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7]))
        .expect("parent");
    let sidecar = runtime
        .retain_device_memory_copy(0, &u64_payload(&[0, 0]))
        .expect("sidecar");
    let deleted_live = u64::from_le_bytes([0x7f; 8]);
    let malformed_sidecar = CudaInsertForeignKeyParentShard {
        payload: &parent,
        columns: &fixed,
        row_count: 1,
        created_by: Some(CudaInsertResidentKeySidecar {
            memory: &sidecar,
            byte_offset: 4,
        }),
        created_default: 0,
        deleted_by: None,
        deleted_default: deleted_live,
        deleted_live,
    };
    assert!(child
        .insert_foreign_key_verdict_against_shards(&fixed, 1, &[malformed_sidecar], None, 1, 0)
        .is_err());
    let malformed_version = CudaInsertForeignKeyParentShard {
        payload: &parent,
        columns: &fixed,
        row_count: 1,
        created_by: None,
        created_default: 10,
        deleted_by: None,
        deleted_default: 9,
        deleted_live,
    };
    assert!(child
        .insert_foreign_key_verdict_against_shards(&fixed, 1, &[malformed_version], None, 20, 0)
        .is_err());

    let required = insert_foreign_key_verdict_scratch_bytes(1, fixed.len(), fixed.len(), None)
        .expect("checked exact scratch");
    let exact = CudaAllocationScope::with_budget(required);
    assert!(child
        .insert_foreign_key_verdict_against_shards(
            &fixed,
            1,
            &[plain_parent(&parent, &fixed, 1)],
            None,
            1,
            0,
        )
        .is_ok());
    assert_eq!(exact.peak_bytes(), required);
    drop(exact);
    let too_small = CudaAllocationScope::with_budget(required - 1);
    assert!(matches!(
        child.insert_foreign_key_verdict_against_shards(
            &fixed,
            1,
            &[plain_parent(&parent, &fixed, 1)],
            None,
            1,
            0,
        ),
        Err(CudaRuntimeProbeError::AllocationBudgetExceeded { .. })
    ));
    assert_eq!(too_small.peak_bytes(), 0);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_foreign_key_verdict_failpoints_drain_before_drop_and_immediate_reuse() {
    use crate::insert_foreign_key_verdict::{
        fail_next_insert_foreign_key_verdict_after_first_parent_launch,
        fail_next_insert_foreign_key_verdict_after_initialization,
    };

    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let columns = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    {
        let child = runtime
            .retain_device_memory_copy(0, &i32_payload(&[7]))
            .expect("child");
        let parent = runtime
            .retain_device_memory_copy(0, &i32_payload(&[7]))
            .expect("parent");
        let parents = [plain_parent(&parent, &columns, 1)];
        fail_next_insert_foreign_key_verdict_after_initialization();
        assert!(matches!(
            child.insert_foreign_key_verdict_against_shards(&columns, 1, &parents, None, 1, 0),
            Err(CudaRuntimeProbeError::KernelLaunchFailed(-1))
        ));
        fail_next_insert_foreign_key_verdict_after_first_parent_launch();
        assert!(matches!(
            child.insert_foreign_key_verdict_against_shards(&columns, 1, &parents, None, 1, 0),
            Err(CudaRuntimeProbeError::KernelLaunchFailed(-1))
        ));
        // The scope ends immediately after the armed-drain errors: releasing these caller-owned
        // payloads must not race queued setup or first-parent work.
    }
    assert_eq!(
        runtime
            .launch_smoke_add_one(41)
            .expect("default stream survived error-path drain"),
        42
    );
}
