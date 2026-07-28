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

#[test]
fn insert_resident_key_verdict_scratch_is_bucketed_and_total() {
    assert_eq!(
        insert_resident_key_verdict_scratch_bytes(1, 1, 1),
        Some(1_024)
    );
    assert_eq!(
        insert_resident_key_verdict_scratch_bytes(17, 3, 3),
        Some(1_280)
    );
    assert_eq!(
        insert_resident_key_verdict_scratch_bytes(17, 9, 9),
        Some(1_792)
    );
    assert!(insert_resident_key_verdict_scratch_bytes(usize::MAX, 1, 1).is_none());
    assert!(insert_resident_key_verdict_scratch_bytes(1, usize::MAX, 1).is_none());
    assert!(insert_resident_key_verdict_scratch_bytes(1, 1, usize::MAX).is_none());
}

#[test]
fn insert_resident_key_verdict_static_contract_keeps_exact_gpu_sidecars_and_terminal_only() {
    let source = include_str!("../insert_resident_key_verdict.rs");
    let ptx = include_str!("../insert_resident_key_verdict.ptx");
    assert_eq!(source.matches("memcpy_dtoh(").count(), 1);
    assert!(source.contains("CudaInsertResidentKeyShard"));
    assert!(source.contains("published_constraint_boundary"));
    assert!(source.contains("original_read_snapshot"));
    assert!(source.contains("NullStreamDrain"));
    assert!(source.contains("sidecar_ptr"));
    assert!(source.contains("fail_next_insert_resident_key_verdict_after_initialization"));
    assert!(source.contains("fail_next_insert_resident_key_verdict_after_first_launch"));
    assert!(source.contains("incoming_row_count == u32::MAX"));
    assert!(source.contains("published_constraint_boundary < original_read_snapshot"));
    assert!(!source.contains("submit_compound_fold_fingerprints"));
    assert!(!source.contains("read_resident_"));
    for required in [
        "BUILD_PROBE",
        "SCAN_PROBE",
        "EXACT_COLUMN",
        "atom.global.cas.b64",
        "atom.global.min.u32",
        "created_default",
        "deleted_live",
    ] {
        assert!(ptx.contains(required), "PTX lacks {required}");
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_resident_key_verdict_failpoint_drains_the_default_stream() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let incoming = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7]))
        .expect("incoming payload");
    let resident = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7]))
        .expect("resident payload");
    let columns = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let deleted_live = u64::from_le_bytes([0x7f; 8]);
    let shard = CudaInsertResidentKeyShard {
        payload: &resident,
        columns: &columns,
        row_count: 1,
        created_by: None,
        created_default: 0,
        deleted_by: None,
        deleted_default: deleted_live,
        deleted_live,
    };
    crate::insert_resident_key_verdict::fail_next_insert_resident_key_verdict_after_initialization(
    );
    assert!(matches!(
        incoming.insert_resident_key_verdict_against_shard(&columns, 1, &shard, 1, 0),
        Err(CudaRuntimeProbeError::KernelLaunchFailed(-1))
    ));
    assert_eq!(
        runtime
            .launch_smoke_add_one(41)
            .expect("context survives drain"),
        42
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_resident_key_verdict_after_launch_failpoint_drains_and_reuses_context() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let incoming = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7]))
        .expect("incoming payload");
    let resident = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7]))
        .expect("resident payload");
    let columns = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let deleted_live = u64::from_le_bytes([0x7f; 8]);
    let shard = CudaInsertResidentKeyShard {
        payload: &resident,
        columns: &columns,
        row_count: 1,
        created_by: None,
        created_default: 0,
        deleted_by: None,
        deleted_default: deleted_live,
        deleted_live,
    };
    crate::insert_resident_key_verdict::fail_next_insert_resident_key_verdict_after_first_launch();
    assert!(matches!(
        incoming.insert_resident_key_verdict_against_shard(&columns, 1, &shard, 1, 0),
        Err(CudaRuntimeProbeError::KernelLaunchFailed(-1))
    ));
    assert_eq!(
        runtime
            .launch_smoke_add_one(7)
            .expect("context survives drain"),
        8
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_resident_key_verdict_rejects_layout_sidecar_and_boundary_sabotage() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let incoming = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7]))
        .expect("incoming payload");
    let resident = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7]))
        .expect("resident payload");
    let i32_columns = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let bool_columns = [CudaCompoundFoldColumn::Bool {
        bitmap_byte_offset: 0,
    }];
    let deleted_live = u64::from_le_bytes([0x7f; 8]);
    let bad_layout = CudaInsertResidentKeyShard {
        payload: &resident,
        columns: &bool_columns,
        row_count: 1,
        created_by: None,
        created_default: 0,
        deleted_by: None,
        deleted_default: deleted_live,
        deleted_live,
    };
    assert!(incoming
        .insert_resident_key_verdict_against_shard(&i32_columns, 1, &bad_layout, 1, 0)
        .is_err());
    let bad_extent = CudaInsertResidentKeyShard {
        payload: &resident,
        columns: &i32_columns,
        row_count: 1,
        created_by: Some(CudaInsertResidentKeySidecar {
            memory: &resident,
            byte_offset: 4,
        }),
        created_default: 0,
        deleted_by: None,
        deleted_default: deleted_live,
        deleted_live,
    };
    assert!(incoming
        .insert_resident_key_verdict_against_shard(&i32_columns, 1, &bad_extent, 1, 0)
        .is_err());
    let version_sidecar = runtime
        .retain_device_memory_copy(0, &u64_payload(&[0, 0]))
        .expect("in-bounds sidecar allocation");
    let bad_alignment = CudaInsertResidentKeyShard {
        payload: &resident,
        columns: &i32_columns,
        row_count: 1,
        created_by: Some(CudaInsertResidentKeySidecar {
            memory: &version_sidecar,
            byte_offset: 4,
        }),
        created_default: 0,
        deleted_by: None,
        deleted_default: deleted_live,
        deleted_live,
    };
    assert!(incoming
        .insert_resident_key_verdict_against_shard(&i32_columns, 1, &bad_alignment, 1, 0)
        .is_err());
    let good = CudaInsertResidentKeyShard {
        payload: &resident,
        columns: &i32_columns,
        row_count: 1,
        created_by: None,
        created_default: 0,
        deleted_by: None,
        deleted_default: deleted_live,
        deleted_live,
    };
    assert!(incoming
        .insert_resident_key_verdict_against_shard(&i32_columns, u32::MAX, &good, 1, 0)
        .is_err());
    assert!(incoming
        .insert_resident_key_verdict_against_shard(&i32_columns, 1, &good, 0, 1)
        .is_err());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_resident_key_verdict_rejects_malformed_text_and_version_sidecars() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let mut bad_text = Vec::new();
    push_u64s(&mut bad_text, &[0, 3]); // end exceeds the declared two-byte blob.
    bad_text.extend_from_slice(b"ok");
    let text = runtime
        .retain_device_memory_copy(0, &bad_text)
        .expect("malformed text payload");
    let text_columns = [CudaCompoundFoldColumn::Text {
        offsets_byte_offset: 0,
        bytes_byte_offset: 16,
        bytes_len: 2,
    }];
    let deleted_live = u64::from_le_bytes([0x7f; 8]);
    let text_shard = CudaInsertResidentKeyShard {
        payload: &text,
        columns: &text_columns,
        row_count: 1,
        created_by: None,
        created_default: 0,
        deleted_by: None,
        deleted_default: deleted_live,
        deleted_live,
    };
    assert!(text
        .insert_resident_key_verdict_against_shard(&text_columns, 1, &text_shard, 1, 0)
        .is_err());

    let incoming = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7]))
        .expect("incoming payload");
    let resident = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7]))
        .expect("resident payload");
    let created = runtime
        .retain_device_memory_copy(0, &u64_payload(&[10]))
        .expect("created sidecar");
    let deleted = runtime
        .retain_device_memory_copy(0, &u64_payload(&[9]))
        .expect("deleted sidecar");
    let fixed = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let malformed_version = CudaInsertResidentKeyShard {
        payload: &resident,
        columns: &fixed,
        row_count: 1,
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
    };
    assert!(incoming
        .insert_resident_key_verdict_against_shard(&fixed, 1, &malformed_version, 20, 0)
        .is_err());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_resident_key_verdict_separates_visible_and_history_minima() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let incoming = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7, 8, 9]))
        .expect("incoming payload");
    let resident = runtime
        .retain_device_memory_copy(0, &i32_payload(&[8, 7, 9]))
        .expect("resident payload");
    let created = runtime
        .retain_device_memory_copy(0, &u64_payload(&[0, 0, 12]))
        .expect("created sidecar");
    let deleted_live = u64::from_le_bytes([0x7f; 8]);
    let deleted = runtime
        .retain_device_memory_copy(0, &u64_payload(&[deleted_live; 3]))
        .expect("deleted sidecar");
    let columns = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let shard = CudaInsertResidentKeyShard {
        payload: &resident,
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
    };
    assert_eq!(
        incoming
            .insert_resident_key_verdict_against_shard(&columns, 3, &shard, 10, 5)
            .expect("device exact resident verdict"),
        CudaInsertResidentKeyVerdict {
            first_visible_conflict_row: Some(0),
            first_history_conflict_row: Some(2),
            readback_bytes: INSERT_RESIDENT_KEY_VERDICT_READBACK_BYTES,
        }
    );

    let released = runtime
        .retain_device_memory_copy(0, &u64_payload(&[12]))
        .expect("post-snapshot delete sidecar");
    let released_resident = runtime
        .retain_device_memory_copy(0, &i32_payload(&[7]))
        .expect("released resident key");
    let released_shard = CudaInsertResidentKeyShard {
        payload: &released_resident,
        columns: &columns,
        row_count: 1,
        created_by: None,
        created_default: 0,
        deleted_by: Some(CudaInsertResidentKeySidecar {
            memory: &released,
            byte_offset: 0,
        }),
        deleted_default: deleted_live,
        deleted_live,
    };
    assert_eq!(
        incoming
            .insert_resident_key_verdict_against_shard(&columns, 1, &released_shard, 15, 5)
            .expect("device claim-release verdict"),
        CudaInsertResidentKeyVerdict {
            first_visible_conflict_row: None,
            first_history_conflict_row: Some(0),
            readback_bytes: INSERT_RESIDENT_KEY_VERDICT_READBACK_BYTES,
        }
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_resident_key_verdict_skips_null_unique_keys() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let mut incoming_payload = i32_payload(&[7, 7]);
    let incoming_validity = incoming_payload.len() as u64;
    incoming_payload.push(0b10); // row 0 is NULL, row 1 is present.
    let mut resident_payload = i32_payload(&[7]);
    let resident_validity = resident_payload.len() as u64;
    resident_payload.push(0b1);
    let incoming = runtime
        .retain_device_memory_copy(0, &incoming_payload)
        .expect("incoming payload");
    let resident = runtime
        .retain_device_memory_copy(0, &resident_payload)
        .expect("resident payload");
    let incoming_columns = [
        CudaCompoundFoldColumn::Fixed {
            byte_offset: 0,
            width_words: 1,
        },
        CudaCompoundFoldColumn::Validity {
            bitmap_byte_offset: incoming_validity,
        },
    ];
    let resident_columns = [
        CudaCompoundFoldColumn::Fixed {
            byte_offset: 0,
            width_words: 1,
        },
        CudaCompoundFoldColumn::Validity {
            bitmap_byte_offset: resident_validity,
        },
    ];
    let deleted_live = u64::from_le_bytes([0x7f; 8]);
    let shard = CudaInsertResidentKeyShard {
        payload: &resident,
        columns: &resident_columns,
        row_count: 1,
        created_by: None,
        created_default: 0,
        deleted_by: None,
        deleted_default: deleted_live,
        deleted_live,
    };
    assert_eq!(
        incoming
            .insert_resident_key_verdict_against_shard(&incoming_columns, 2, &shard, 1, 0)
            .expect("nulls distinct resident verdict")
            .first_visible_conflict_row,
        Some(1)
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_resident_key_verdict_rechecks_a_forced_compound_fingerprint_collision() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    let first = (11_u32, 29_u32);
    let second_first = 41_u32;
    let target = fold_word(fold_word(2_166_136_261, first.0), first.1);
    let second_second = inverse_fold_word(target, fold_word(2_166_136_261, second_first));
    assert_ne!(first, (second_first, second_second));
    assert_eq!(
        target,
        fold_word(fold_word(2_166_136_261, second_first), second_second)
    );

    let mut incoming_payload = i32_payload(&[first.0 as i32, second_first as i32]);
    incoming_payload.extend_from_slice(&i32_payload(&[first.1 as i32, second_second as i32]));
    let mut resident_payload = i32_payload(&[second_first as i32]);
    resident_payload.extend_from_slice(&i32_payload(&[second_second as i32]));
    let incoming = runtime
        .retain_device_memory_copy(0, &incoming_payload)
        .expect("incoming collision payload");
    let resident = runtime
        .retain_device_memory_copy(0, &resident_payload)
        .expect("resident collision payload");
    let incoming_columns = [
        CudaCompoundFoldColumn::Fixed {
            byte_offset: 0,
            width_words: 1,
        },
        CudaCompoundFoldColumn::Fixed {
            byte_offset: 8,
            width_words: 1,
        },
    ];
    let resident_columns = [
        CudaCompoundFoldColumn::Fixed {
            byte_offset: 0,
            width_words: 1,
        },
        CudaCompoundFoldColumn::Fixed {
            byte_offset: 4,
            width_words: 1,
        },
    ];
    let deleted_live = u64::from_le_bytes([0x7f; 8]);
    let shard = CudaInsertResidentKeyShard {
        payload: &resident,
        columns: &resident_columns,
        row_count: 1,
        created_by: None,
        created_default: 0,
        deleted_by: None,
        deleted_default: deleted_live,
        deleted_live,
    };
    assert_eq!(
        incoming
            .insert_resident_key_verdict_against_shard(&incoming_columns, 2, &shard, 1, 0)
            .expect("collision exact verdict")
            .first_visible_conflict_row,
        Some(1),
        "fingerprint equality must only select the exact GPU recheck candidate"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_insert_resident_key_verdict_handles_every_typed_key_layout_and_repetition() {
    let Some(runtime) = gpu_runtime() else {
        return;
    };
    fn payload() -> (Vec<u8>, [CudaCompoundFoldColumn; 7]) {
        let mut payload = i32_payload(&[10, 20]);
        let i64_offset = payload.len() as u64;
        push_u64s(&mut payload, &[100, 200]);
        let i128_offset = payload.len() as u64;
        payload.extend_from_slice(&[1; 16]);
        payload.extend_from_slice(&[2; 16]);
        let bool_offset = payload.len() as u64;
        payload.push(0b10);
        payload.resize(payload.len().next_multiple_of(8), 0);
        let text_offsets = payload.len() as u64;
        push_u64s(&mut payload, &[0, 1, 2]);
        let text_bytes = payload.len() as u64;
        payload.extend_from_slice(b"ab");
        let validity = payload.len() as u64;
        payload.push(0b11);
        (
            payload,
            [
                CudaCompoundFoldColumn::Fixed {
                    byte_offset: 0,
                    width_words: 1,
                },
                CudaCompoundFoldColumn::Fixed {
                    byte_offset: i64_offset,
                    width_words: 2,
                },
                CudaCompoundFoldColumn::Fixed {
                    byte_offset: i128_offset,
                    width_words: 4,
                },
                CudaCompoundFoldColumn::Bool {
                    bitmap_byte_offset: bool_offset,
                },
                CudaCompoundFoldColumn::Text {
                    offsets_byte_offset: text_offsets,
                    bytes_byte_offset: text_bytes,
                    bytes_len: 2,
                },
                CudaCompoundFoldColumn::Fixed {
                    byte_offset: 0,
                    width_words: 1,
                },
                CudaCompoundFoldColumn::Validity {
                    bitmap_byte_offset: validity,
                },
            ],
        )
    }
    let (incoming_payload, incoming_columns) = payload();
    let (resident_payload, resident_columns) = payload();
    let incoming = runtime
        .retain_device_memory_copy(0, &incoming_payload)
        .expect("all-types incoming payload");
    let resident = runtime
        .retain_device_memory_copy(0, &resident_payload)
        .expect("all-types resident payload");
    let deleted_live = u64::from_le_bytes([0x7f; 8]);
    let shard = CudaInsertResidentKeyShard {
        payload: &resident,
        columns: &resident_columns,
        row_count: 2,
        created_by: None,
        created_default: 0,
        deleted_by: None,
        deleted_default: deleted_live,
        deleted_live,
    };
    assert_eq!(
        incoming
            .insert_resident_key_verdict_against_shard(&incoming_columns, 2, &shard, 1, 0)
            .expect("all-types exact verdict")
            .first_visible_conflict_row,
        Some(0)
    );
}
