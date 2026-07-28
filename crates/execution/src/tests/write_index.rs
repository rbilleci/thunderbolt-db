fn posting_index_bytes(words: &[u64], row_capacity: usize) -> Vec<u8> {
    let mut bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    bytes.resize(bytes.len() + row_capacity * std::mem::size_of::<u32>(), 0);
    bytes
}

#[test]
fn prepared_multi_index_token_is_move_only_and_owns_exact_scratch_geometry() {
    assert_eq!(
        resident_typed_indexes_insert_preparation_bytes(2, 4),
        Some(512),
        "two 256-byte pool buckets cover the descriptor image and terminal"
    );
    assert_eq!(
        resident_typed_indexes_insert_preparation_bytes(usize::MAX, 1),
        None,
        "overflow must decline before any device lease"
    );
    if let Ok(auditor_column_count) = usize::try_from(1_u64 << 58) {
        assert_eq!(
            resident_typed_indexes_insert_preparation_bytes(1, auditor_column_count),
            None,
            "descriptor bucket rounding must fail closed instead of panicking"
        );
    }
    fn assert_send<T: Send>() {}
    assert_send::<PreparedResidentTypedIndexesInsert>();
    let source = include_str!("../resident_index_build.rs");
    let drain = source
        .split("struct NullStreamDrain")
        .nth(1)
        .expect("prepared launch drain definition");
    assert!(source.contains("pub struct PreparedResidentTypedIndexesInsert"));
    assert!(drain.contains("primary: Arc<crate::GpuPrimaryContext>"));
    assert!(drain.contains("let _ = self.primary.set_current();"));
    assert!(source.contains("_source_owner:"));
    assert!(source.contains("_index_owners:"));
    assert!(
        source.find("preparation_drain:") < source.find("descriptor_guard:"),
        "the drain must drop before pooled descriptor ownership"
    );
    assert!(source.contains("descriptor_guard:"));
    assert!(source.contains("decline_guard:"));
    assert!(source.contains("pub fn submit(mut self)"));
    assert!(!source.contains("impl Clone for PreparedResidentTypedIndexesInsert"));
}

/// M1 (ledger #24): the INCREMENTAL index-insert kernel == a full rebuild. Build an index for
/// a prefix of keys, INSERT the appended tail via the kernel, and verify the extended index
/// probes IDENTICALLY to a from-scratch build over all keys (via the write-locate kernel).
/// Also verifies same-key MVCC twins prepend one posting chain without declining the index.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn index_insert_kernel_extends_like_a_rebuild() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    if !runtime.snapshot().driver_available {
        return;
    }
    // Build for 6 keys, then INSERT 4 more into the SAME (over-sized) table.
    let all: Vec<i32> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    let prefix = &all[..6];
    // Size the table for ALL 10 (so the tail fits without a resize).
    let table_size = (all.len() as u64 * 2).next_power_of_two().max(2);
    let table_mask = (table_size - 1) as u32;
    let hash_shift = 32 - table_size.trailing_zeros();
    // Host-build the prefix into a table of the full size.
    let mut words = vec![0u64; table_size as usize];
    for (row, &key) in prefix.iter().enumerate() {
        let kb = key as u32;
        let mut slot = (kb.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
        loop {
            if words[slot as usize] == 0 {
                words[slot as usize] = ((kb as u64) << 32) | (row as u64 + 1);
                break;
            }
            slot = (slot + 1) & table_mask;
        }
    }
    let bytes = posting_index_bytes(&words, 12);
    let Ok(mem) = runtime.retain_device_memory_copy(0, &bytes) else {
        return;
    };
    let index = std::sync::Arc::new(mem);
    // INSERT the tail (rows 6..10) via the kernel.
    let tail = &all[6..];
    let insert_status = index
        .submit_i32_index_insert_status(table_mask, hash_shift, tail, 6)
        .expect("index insert");
    assert!(!insert_status.declined, "fresh keys do not decline");
    assert!(
        !insert_status.created_posting,
        "fresh keys remain singleton heads"
    );
    // Probe ALL keys via the write-locate kernel: each must resolve to its row.
    let shards = [WriteLocateShard {
        index: std::sync::Arc::clone(&index),
        table_mask,
        hash_shift,
        row_count: all.len() as u32,
    }];
    let result = index
        .submit_multi_shard_i32_write_locate(&shards, &all, 2)
        .expect("locate");
    for (i, &key) in all.iter().enumerate() {
        assert_eq!(
            result.count[i], 1,
            "key {key}: exactly one hit after extend"
        );
        let slot = result.slot[i * result.max_hits as usize];
        assert_eq!(slot as usize, i, "key {key}: row {i} preserved");
    }
    // Same-key MVCC twin: insert a newer physical version in the next logical row. It must
    // prepend the old row to one posting chain and expose both candidate coordinates.
    let version_status = index
        .submit_i32_index_insert_status(table_mask, hash_shift, &[30], 10)
        .expect("version-twin insert");
    assert!(
        !version_status.declined,
        "same-key version twin must not decline"
    );
    assert!(
        version_status.created_posting,
        "same-key insert reports its posting"
    );
    let twin_shards = [WriteLocateShard {
        index: std::sync::Arc::clone(&index),
        table_mask,
        hash_shift,
        row_count: 11,
    }];
    let twin = index
        .submit_multi_shard_i32_write_locate(&twin_shards, &[30], 2)
        .expect("locate version twins");
    assert_eq!(twin.count, vec![2]);
    assert_eq!(twin.slot, vec![10, 2]);
    let fresh_after_posting = index
        .submit_i32_index_insert_status(table_mask, hash_shift, &[110], 11)
        .expect("fresh key after an existing posting");
    assert!(
        !fresh_after_posting.created_posting,
        "status is operation-scoped; retained owners monotonically OR it"
    );
}

/// R3-002: the initial hash build consumes resident keys/stamps directly. Prove raw-key placement,
/// duplicate decline, and the oldest-active-boundary skip without a host-built table.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn resident_typed_index_build_handles_duplicates_and_gc_boundary() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    if !runtime.snapshot().driver_available {
        return;
    }
    let locate = |index: std::sync::Arc<CudaResidentDeviceMemory>, rows, needles: &[i32]| {
        index
            .submit_multi_shard_i32_write_locate(
                &[WriteLocateShard {
                    index: std::sync::Arc::clone(&index),
                    table_mask: 7,
                    hash_shift: 29,
                    row_count: rows,
                }],
                needles,
                2,
            )
            .expect("resident-built index locate")
    };

    let keys = [10_i32, 20, 30];
    let key_bytes = keys
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let source = runtime
        .retain_device_memory_copy(0, &key_bytes)
        .expect("resident keys");
    let index = std::sync::Arc::new(
        runtime
            .retain_device_memory_zeroed(0, resident_index_allocated_bytes(7, 3).unwrap())
            .expect("zeroed index"),
    );
    let unique_status = source
        .submit_resident_typed_index_build_status(
            &index,
            7,
            29,
            &[CudaCompoundFoldColumn::Fixed {
                byte_offset: 0,
                width_words: 1,
            }],
            keys.len(),
            None,
            0,
            false,
        )
        .expect("raw resident build");
    assert_eq!(
        unique_status,
        CudaResidentIndexStatus {
            declined: false,
            created_posting: false,
        }
    );
    let built = locate(index, keys.len() as u32, &keys);
    assert_eq!(built.count, vec![1, 1, 1]);
    assert_eq!([built.slot[0], built.slot[2], built.slot[4]], [0, 1, 2]);

    let twins = [30_i32, 30];
    let twin_bytes = twins
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let twin_source = runtime
        .retain_device_memory_copy(0, &twin_bytes)
        .expect("duplicate resident keys");
    let duplicate_index = runtime
        .retain_device_memory_zeroed(0, resident_index_allocated_bytes(7, 2).unwrap())
        .expect("duplicate index");
    let strict_duplicate = twin_source
        .submit_resident_typed_index_build_status(
            &duplicate_index,
            7,
            29,
            &[CudaCompoundFoldColumn::Fixed {
                byte_offset: 0,
                width_words: 1,
            }],
            twins.len(),
            None,
            0,
            false,
        )
        .expect("non-tolerant duplicate build");
    assert!(strict_duplicate.declined);
    assert!(!strict_duplicate.created_posting);

    let live_stamp = 0x7f7f_7f7f_7f7f_7f7f_u64;
    let stamps = [5_u64, live_stamp];
    let stamp_bytes = stamps
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let deleted = runtime
        .retain_device_memory_copy(0, &stamp_bytes)
        .expect("resident deleted stamps");
    let gc_index = std::sync::Arc::new(
        runtime
            .retain_device_memory_zeroed(0, resident_index_allocated_bytes(7, 2).unwrap())
            .expect("gc index"),
    );
    assert!(
        !twin_source
            .submit_resident_typed_index_build(
                &gc_index,
                7,
                29,
                &[CudaCompoundFoldColumn::Fixed {
                    byte_offset: 0,
                    width_words: 1,
                }],
                twins.len(),
                Some(&deleted),
                5,
                false,
            )
            .expect("GC-bound resident build")
    );
    let after_gc = locate(gc_index, twins.len() as u32, &[30]);
    assert_eq!(after_gc.count, vec![1]);
    assert_eq!(after_gc.slot[0], 1);
}

/// PRODUCT-001 NULLS DISTINCT: validity descriptors are device predicates, not fingerprint
/// material. Physical zero placeholders for NULL rows must never enter the resident directory,
/// while equal present values still produce the strict unique verdict.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn resident_typed_index_validity_predicate_omits_null_keys() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let build = |keys: &[i32], validity: u32, duplicate_tolerant: bool| {
        let mut payload = keys
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let validity_offset = payload.len() as u64;
        payload.extend_from_slice(&validity.to_le_bytes());
        let source = runtime
            .retain_device_memory_copy(0, &payload)
            .expect("resident keys and validity");
        let index = std::sync::Arc::new(
            runtime
                .retain_device_memory_zeroed(0, resident_index_allocated_bytes(15, 4).unwrap())
                .expect("zeroed validity index"),
        );
        let status = source
            .submit_resident_typed_index_build_status(
                &index,
                15,
                28,
                &[
                    CudaCompoundFoldColumn::Fixed {
                        byte_offset: 0,
                        width_words: 1,
                    },
                    CudaCompoundFoldColumn::Validity {
                        bitmap_byte_offset: validity_offset,
                    },
                ],
                keys.len(),
                None,
                0,
                duplicate_tolerant,
            )
            .expect("validity-filtered resident build");
        (index, status)
    };

    let (distinct, status) = build(&[0, 0, 7, 8], 0b1100, false);
    assert_eq!(
        status,
        CudaResidentIndexStatus {
            declined: false,
            created_posting: false,
        }
    );
    let hits = distinct
        .submit_multi_shard_i32_write_locate(
            &[WriteLocateShard {
                index: std::sync::Arc::clone(&distinct),
                table_mask: 15,
                hash_shift: 28,
                row_count: 4,
            }],
            &[0, 7, 8],
            2,
        )
        .expect("probe validity-filtered index");
    assert_eq!(hits.count, vec![0, 1, 1]);

    let (_, duplicate) = build(&[0, 0, 7, 7], 0b1100, false);
    assert!(
        duplicate.declined,
        "equal present keys still violate UNIQUE"
    );
}

/// PRODUCT-002 adversarial boundary: a hot non-unique key and its MVCC versions are posting-chain
/// entries, not open-addressing collisions. Counts well beyond the 256 distinct-key probe bound
/// must build, incrementally extend, and enumerate without decline.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn resident_index_posting_chain_exceeds_256_versions() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    const ROWS: usize = 600;
    const PREFIX: usize = 300;
    let keys = vec![77_i32; ROWS];
    let key_bytes = keys
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let source = runtime
        .retain_device_memory_copy(0, &key_bytes)
        .expect("resident duplicate keys");
    let table_size = (ROWS as u64 * 2).next_power_of_two();
    let table_mask = (table_size - 1) as u32;
    let hash_shift = 32 - table_size.trailing_zeros();
    let columns = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];

    let rebuilt = std::sync::Arc::new(
        runtime
            .retain_device_memory_zeroed(
                0,
                resident_index_allocated_bytes(table_mask, ROWS as u64).unwrap(),
            )
            .expect("posting index"),
    );
    let rebuilt_status = source
        .submit_resident_typed_index_build_status(
            &rebuilt, table_mask, hash_shift, &columns, ROWS, None, 0, true,
        )
        .expect("duplicate-tolerant build");
    assert!(!rebuilt_status.declined);
    assert!(rebuilt_status.created_posting);
    let rebuilt_hits = rebuilt
        .submit_multi_shard_i32_write_locate(
            &[WriteLocateShard {
                index: std::sync::Arc::clone(&rebuilt),
                table_mask,
                hash_shift,
                row_count: ROWS as u32,
            }],
            &[77],
            ROWS as u32,
        )
        .expect("enumerate rebuilt posting chain");
    assert_eq!(rebuilt_hits.count, vec![ROWS as u32]);

    let extended = std::sync::Arc::new(
        runtime
            .retain_device_memory_zeroed(
                0,
                resident_index_allocated_bytes(table_mask, ROWS as u64).unwrap(),
            )
            .expect("incremental posting index"),
    );
    let extended_twin = std::sync::Arc::new(
        runtime
            .retain_device_memory_zeroed(
                0,
                resident_index_allocated_bytes(table_mask, ROWS as u64).unwrap(),
            )
            .expect("second incremental posting index"),
    );
    for index in [&extended, &extended_twin] {
        assert!(
            !source
                .submit_resident_typed_index_build(
                    index, table_mask, hash_shift, &columns, PREFIX, None, 0, true,
                )
                .expect("posting prefix build")
        );
    }
    let extended_status = source
        .submit_resident_typed_indexes_insert_status(
            &[
                CudaResidentTypedIndexInsert {
                    index: std::sync::Arc::clone(&extended),
                    table_mask,
                    hash_shift,
                    columns: columns.to_vec(),
                },
                CudaResidentTypedIndexInsert {
                    index: std::sync::Arc::clone(&extended_twin),
                    table_mask,
                    hash_shift,
                    columns: columns.to_vec(),
                },
            ],
            PREFIX,
            ROWS - PREFIX,
        )
        .expect("resident multi-index posting tail insert");
    assert!(!extended_status.declined);
    assert!(extended_status.created_posting);
    for index in [extended, extended_twin] {
        let extended_hits = index
            .submit_multi_shard_i32_write_locate(
                &[WriteLocateShard {
                    index: std::sync::Arc::clone(&index),
                    table_mask,
                    hash_shift,
                    row_count: ROWS as u32,
                }],
                &[77],
                ROWS as u32,
            )
            .expect("enumerate extended posting chain");
        assert_eq!(extended_hits.count, vec![ROWS as u32]);
        let mut slots = extended_hits.slot;
        slots.sort_unstable();
        assert_eq!(slots, (0..ROWS as u32).collect::<Vec<_>>());
    }
}

/// The pre-WAL multi-index token must own every launch resource.  This fixture deliberately uses
/// a raw single-i32 key plus a compound i32/TEXT/validity key, drops the source and one index's
/// public owner after preparation, then proves submit neither leases nor resolves anything new.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_multi_index_insert_pins_resources_and_submit_allocates_nothing() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let table_mask = 7;
    let hash_shift = 29;
    let mut payload = Vec::new();
    payload.extend(
        [11_i32, 12, 13]
            .iter()
            .flat_map(|value| value.to_le_bytes()),
    );
    payload.resize(16, 0);
    payload.extend(
        [0_u64, 1, 2, 3]
            .iter()
            .flat_map(|value| value.to_le_bytes()),
    );
    payload.extend_from_slice(b"abc");
    payload.resize(52, 0);
    payload.extend_from_slice(&0b111_u32.to_le_bytes());

    let source = runtime
        .retain_device_memory_copy(0, &payload)
        .expect("resident compound source");
    let raw_index = std::sync::Arc::new(
        runtime
            .retain_device_memory_zeroed(0, resident_index_allocated_bytes(table_mask, 3).unwrap())
            .expect("raw destination index"),
    );
    let compound_index = std::sync::Arc::new(
        runtime
            .retain_device_memory_zeroed(0, resident_index_allocated_bytes(table_mask, 3).unwrap())
            .expect("compound destination index"),
    );
    let raw_columns = vec![CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let compound_columns = vec![
        CudaCompoundFoldColumn::Fixed {
            byte_offset: 0,
            width_words: 1,
        },
        CudaCompoundFoldColumn::Text {
            offsets_byte_offset: 16,
            bytes_byte_offset: 48,
            bytes_len: 3,
        },
        CudaCompoundFoldColumn::Validity {
            bitmap_byte_offset: 52,
        },
    ];
    for (index, columns) in [
        (&raw_index, raw_columns.as_slice()),
        (&compound_index, compound_columns.as_slice()),
    ] {
        let initial = source
            .submit_resident_typed_index_build_status(
                index, table_mask, hash_shift, columns, 1, None, 0, true,
            )
            .expect("prefix build");
        assert!(!initial.declined);
    }
    let requests = vec![
        CudaResidentTypedIndexInsert {
            index: std::sync::Arc::clone(&raw_index),
            table_mask,
            hash_shift,
            columns: raw_columns,
        },
        CudaResidentTypedIndexInsert {
            index: std::sync::Arc::clone(&compound_index),
            table_mask,
            hash_shift,
            columns: compound_columns,
        },
    ];

    let exact_preparation =
        resident_typed_indexes_insert_preparation_bytes(2, 4).expect("bounded descriptor geometry");
    assert_eq!(exact_preparation, 512);
    let too_small_scope = CudaAllocationScope::with_budget(exact_preparation - 1);
    assert!(matches!(
        source.prepare_resident_typed_indexes_insert(&requests, 1, 2),
        Err(CudaRuntimeProbeError::AllocationBudgetExceeded { .. })
    ));
    assert_eq!(
        too_small_scope.peak_bytes(),
        0,
        "complete launch admission must refuse before either pooled lease"
    );
    drop(too_small_scope);

    let counters_before = prepared_resident_typed_indexes_insert_counters();
    let scope = CudaAllocationScope::with_budget(exact_preparation);
    fail_next_prepared_resident_typed_indexes_insert_after_leases();
    assert!(matches!(
        source.prepare_resident_typed_indexes_insert(&requests, 1, 2),
        Err(CudaRuntimeProbeError::InvalidInputLength(0))
    ));
    assert_eq!(
        prepared_resident_typed_indexes_insert_counters(),
        counters_before,
        "a post-lease failure must not create a submit-capable token"
    );

    let prepared = source
        .prepare_resident_typed_indexes_insert(&requests, 1, 2)
        .expect("pre-WAL launch preparation");
    let after_prepare = prepared_resident_typed_indexes_insert_counters();
    assert_eq!(after_prepare.prepares, counters_before.prepares + 1);
    assert_eq!(after_prepare.submits, counters_before.submits);
    let peak_before_submit = scope.peak_bytes();
    assert_eq!(peak_before_submit, exact_preparation);
    assert_eq!(prepared.preparation_bytes(), exact_preparation);

    // The token retains allocation guards, so cache/public owners are not required after the
    // pre-WAL boundary. Keep only the raw index needed for the final probe.
    drop(source);
    drop(requests);
    drop(compound_index);
    let (worker_before_submit, status, worker_after_submit, worker_peak) = std::thread::spawn(move || {
        let before = prepared_resident_typed_indexes_insert_counters();
        let worker_scope = CudaAllocationScope::with_budget(0);
        let status = prepared.submit();
        let worker_peak = worker_scope.peak_bytes();
        assert_eq!(
            worker_peak, 0,
            "the consuming submit must not acquire any worker-local GPU lease"
        );
        drop(worker_scope);
        let after = prepared_resident_typed_indexes_insert_counters();
        (before, status, after, worker_peak)
    })
        .join()
        .expect("prepared worker must not panic");
    let status = status.expect("consuming prepared launch on a worker thread");
    assert!(!status.declined);
    assert_eq!(worker_peak, 0);
    assert_eq!(worker_after_submit.prepares, worker_before_submit.prepares);
    assert_eq!(
        worker_after_submit.submits,
        worker_before_submit.submits + 1,
        "the worker consumes the cross-thread token exactly once"
    );
    assert_eq!(
        scope.peak_bytes(),
        peak_before_submit,
        "cross-thread token destruction preserves the preparation-owned tracker's exact peak"
    );
    let after_submit = prepared_resident_typed_indexes_insert_counters();
    assert_eq!(after_submit.prepares, after_prepare.prepares);
    assert_eq!(after_submit.submits, after_prepare.submits);

    // This scope proves only retained preparation accounting; the worker's zero-budget scope is
    // the no-new-lease proof. The later probe has separate scratch ownership.
    drop(scope);

    let hits = raw_index
        .submit_multi_shard_i32_write_locate(
            &[WriteLocateShard {
                index: std::sync::Arc::clone(&raw_index),
                table_mask,
                hash_shift,
                row_count: 3,
            }],
            &[11, 12, 13],
            2,
        )
        .expect("raw key probe after prepared launch");
    assert_eq!(hits.count, vec![1, 1, 1]);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_multi_index_insert_rejects_physical_destination_aliases() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let source = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &[7_i32.to_le_bytes(), 8_i32.to_le_bytes()].concat())
            .expect("resident source"),
    );
    let columns = vec![CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let source_as_index = CudaResidentTypedIndexInsert {
        index: std::sync::Arc::clone(&source),
        table_mask: 3,
        hash_shift: 30,
        columns: columns.clone(),
    };
    assert!(matches!(
        source.prepare_resident_typed_indexes_insert(&[source_as_index], 0, 1),
        Err(CudaRuntimeProbeError::InvalidInputLength(0))
    ));

    let index = std::sync::Arc::new(
        runtime
            .retain_device_memory_zeroed(0, resident_index_allocated_bytes(3, 2).unwrap())
            .expect("index destination"),
    );
    let duplicate = CudaResidentTypedIndexInsert {
        index: std::sync::Arc::clone(&index),
        table_mask: 3,
        hash_shift: 30,
        columns,
    };
    assert!(matches!(
        source.prepare_resident_typed_indexes_insert(&[duplicate.clone(), duplicate], 0, 1,),
        Err(CudaRuntimeProbeError::InvalidInputLength(0))
    ));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_multi_index_insert_after_launch_failure_drains_before_pool_reuse() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let source = runtime
        .retain_device_memory_copy(0, &[21_i32.to_le_bytes(), 22_i32.to_le_bytes()].concat())
        .expect("resident source");
    let index = std::sync::Arc::new(
        runtime
            .retain_device_memory_zeroed(0, resident_index_allocated_bytes(3, 2).unwrap())
            .expect("index destination"),
    );
    let columns = vec![CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    assert!(
        !source
            .submit_resident_typed_index_build(&index, 3, 30, &columns, 1, None, 0, true,)
            .expect("prefix build")
    );
    let request = CudaResidentTypedIndexInsert {
        index: std::sync::Arc::clone(&index),
        table_mask: 3,
        hash_shift: 30,
        columns,
    };
    fail_next_prepared_resident_typed_indexes_insert_after_launch();
    assert!(matches!(
        source
            .prepare_resident_typed_indexes_insert(std::slice::from_ref(&request), 1, 1)
            .expect("prepared launch")
            .submit(),
        Err(CudaRuntimeProbeError::KernelLaunchFailed(-1))
    ));
    assert_eq!(
        runtime
            .launch_smoke_add_one(41)
            .expect("post-launch drain preserves context"),
        42
    );
    // A launch accepted by the driver may already have mutated its original directory, so product
    // code must never retry it. Reuse the pooled descriptor/verdict buffers only with a fresh
    // physical destination and verify the resulting index has the exact one-row-per-key shape.
    let fresh_index = std::sync::Arc::new(
        runtime
            .retain_device_memory_zeroed(0, resident_index_allocated_bytes(3, 2).unwrap())
            .expect("fresh index destination"),
    );
    assert!(
        !source
            .submit_resident_typed_index_build(
                &fresh_index,
                3,
                30,
                &request.columns,
                1,
                None,
                0,
                true,
            )
            .expect("fresh prefix build")
    );
    let fresh_request = CudaResidentTypedIndexInsert {
        index: std::sync::Arc::clone(&fresh_index),
        ..request
    };
    assert!(
        !source
            .prepare_resident_typed_indexes_insert(&[fresh_request], 1, 1)
            .expect("pooled descriptor/verdict reuse")
            .submit()
            .expect("reused prepared launch")
            .declined
    );
    let hits = fresh_index
        .submit_multi_shard_i32_write_locate(
            &[WriteLocateShard {
                index: std::sync::Arc::clone(&fresh_index),
                table_mask: 3,
                hash_shift: 30,
                row_count: 2,
            }],
            &[21, 22],
            2,
        )
        .expect("fresh index probe");
    assert_eq!(hits.count, vec![1, 1]);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_multi_index_insert_drop_without_submit_drains_queued_setup_before_reuse() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let source = runtime
        .retain_device_memory_copy(0, &[31_i32.to_le_bytes(), 32_i32.to_le_bytes()].concat())
        .expect("resident source");
    let columns = vec![CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let abandoned_index = std::sync::Arc::new(
        runtime
            .retain_device_memory_zeroed(0, resident_index_allocated_bytes(3, 2).unwrap())
            .expect("abandoned destination"),
    );
    assert!(!source
        .submit_resident_typed_index_build(
            &abandoned_index, 3, 30, &columns, 1, None, 0, true,
        )
        .expect("abandoned prefix build"));
    let counters_before = prepared_resident_typed_indexes_insert_counters();
    let abandoned = source
        .prepare_resident_typed_indexes_insert(
            &[CudaResidentTypedIndexInsert {
                index: std::sync::Arc::clone(&abandoned_index),
                table_mask: 3,
                hash_shift: 30,
                columns: columns.clone(),
            }],
            1,
            1,
        )
        .expect("queued setup token");
    let after_prepare = prepared_resident_typed_indexes_insert_counters();
    assert_eq!(after_prepare.prepares, counters_before.prepares + 1);
    assert_eq!(after_prepare.submits, counters_before.submits);
    assert_eq!(after_prepare.drains, counters_before.drains);
    let has_gpu1 = runtime.snapshot().device_count > 1;
    let (worker_before_drop, worker_after_drop) = std::thread::spawn(move || {
        let before = prepared_resident_typed_indexes_insert_counters();
        if has_gpu1 {
            // The fresh worker has no current context; when available, make GPU1 current so the
            // token must actively rebind GPU0 before it drains its queued default-stream setup.
            let foreign_primary = gpu_primary_context(1).expect("GPU1 primary context");
            foreign_primary.set_current().expect("bind GPU1");
            drop(
                foreign_primary
                    .lease_device_buffer_owned(1)
                    .expect("use GPU1 pooled buffer"),
            );
        }
        drop(abandoned);
        let after = prepared_resident_typed_indexes_insert_counters();
        (before, after)
    })
    .join()
    .expect("drop worker must not panic");
    let after_drop = prepared_resident_typed_indexes_insert_counters();
    assert_eq!(after_drop.prepares, after_prepare.prepares);
    assert_eq!(after_drop.submits, after_prepare.submits);
    assert_eq!(
        after_drop.drains,
        after_prepare.drains,
        "worker-local drain instrumentation must not leak into the preparation thread"
    );
    assert_eq!(worker_after_drop.prepares, worker_before_drop.prepares);
    assert_eq!(worker_after_drop.submits, worker_before_drop.submits);
    assert_eq!(
        worker_after_drop.drains,
        worker_before_drop.drains + 1,
        "an abandoned queued token must rebind and drain before pooled buffers recycle"
    );
    assert_eq!(
        runtime
            .launch_smoke_add_one(51)
            .expect("abandoned token drained before context reuse"),
        52
    );

    let fresh_index = std::sync::Arc::new(
        runtime
            .retain_device_memory_zeroed(0, resident_index_allocated_bytes(3, 2).unwrap())
            .expect("fresh destination"),
    );
    assert!(!source
        .submit_resident_typed_index_build(
            &fresh_index, 3, 30, &columns, 1, None, 0, true,
        )
        .expect("fresh prefix build"));
    assert!(
        !source
            .prepare_resident_typed_indexes_insert(
                &[CudaResidentTypedIndexInsert {
                    index: std::sync::Arc::clone(&fresh_index),
                    table_mask: 3,
                    hash_shift: 30,
                    columns,
                }],
                1,
                1,
            )
            .expect("pooled setup after abandoned token")
            .submit()
            .expect("fresh prepared submit")
            .declined
    );
    let hits = fresh_index
        .submit_multi_shard_i32_write_locate(
            &[WriteLocateShard {
                index: std::sync::Arc::clone(&fresh_index),
                table_mask: 3,
                hash_shift: 30,
                row_count: 2,
            }],
            &[31, 32],
            2,
        )
        .expect("fresh index probe");
    assert_eq!(hits.count, vec![1, 1]);
}

/// A posting retry for an immutable plan must keep its captured row ceiling authoritative while
/// the physical index capacity permits following a future head back to the old visible version.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_plan_posting_retry_follows_future_head_for_old_snapshot() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let keys = [77_i32, 77];
    let payload = [700_i32, 701];
    let mut resident_bytes = Vec::new();
    resident_bytes.extend(keys.iter().flat_map(|value| value.to_le_bytes()));
    resident_bytes.extend(payload.iter().flat_map(|value| value.to_le_bytes()));
    let resident = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &resident_bytes)
            .expect("two-row resident capacity"),
    );
    let table_mask = 3;
    let hash_shift = 30;
    let index = std::sync::Arc::new(
        runtime
            .retain_device_memory_zeroed(0, resident_index_allocated_bytes(table_mask, 2).unwrap())
            .expect("two-row posting index"),
    );
    let columns = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let initial = resident
        .submit_resident_typed_index_build_status(
            &index, table_mask, hash_shift, &columns, 1, None, 0, true,
        )
        .expect("build singleton prefix");
    assert_eq!(
        initial,
        CudaResidentIndexStatus {
            declined: false,
            created_posting: false,
        }
    );
    let plan = resident
        .prepare_multi_shard_i32_index_probe_dense(&[MultiShardProbeShard {
            resident: std::sync::Arc::clone(&resident),
            index: std::sync::Arc::clone(&index),
            table_mask,
            hash_shift,
            projection_offsets: vec![0, 8],
            row_count: 1,
            row_capacity: 2,
            created_by: None,
            deleted_by: None,
            has_postings: false,
            min: 77,
            max: 77,
        }])
        .expect("prepare singleton plan");

    let appended = resident
        .submit_resident_typed_index_insert_status(
            &index, table_mask, hash_shift, &columns, 1, 1, true,
        )
        .expect("publish future same-key posting");
    assert!(appended.created_posting);

    let (retried, _) = resident
        .submit_prepared_multi_shard_i32_index_probe_dense_posting_retry(&plan, &[77], 1)
        .expect("submit posting retry")
        .complete_detached_columnar()
        .expect("complete posting retry");
    assert_eq!(retried.status, vec![1]);
    assert_eq!(retried.values, vec![77, 700]);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn dense_point_probe_finds_visible_version_beyond_256_postings() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    const ROWS: usize = 300;
    let keys = vec![77_i32; ROWS];
    let payload = (0..ROWS as i32).collect::<Vec<_>>();
    let mut source_bytes = Vec::with_capacity(ROWS * 8);
    source_bytes.extend(keys.iter().flat_map(|value| value.to_le_bytes()));
    source_bytes.extend(payload.iter().flat_map(|value| value.to_le_bytes()));
    let resident = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &source_bytes)
            .expect("resident version payload"),
    );
    let table_size = (ROWS as u64 * 2).next_power_of_two();
    let table_mask = (table_size - 1) as u32;
    let hash_shift = 32 - table_size.trailing_zeros();
    let index = std::sync::Arc::new(
        runtime
            .retain_device_memory_zeroed(
                0,
                resident_index_allocated_bytes(table_mask, ROWS as u64).unwrap(),
            )
            .expect("posting index"),
    );
    assert!(
        !resident
            .submit_resident_typed_index_build(
                &index,
                table_mask,
                hash_shift,
                &[CudaCompoundFoldColumn::Fixed {
                    byte_offset: 0,
                    width_words: 1,
                }],
                ROWS,
                None,
                0,
                true,
            )
            .expect("build version postings")
    );
    let created = (1..=ROWS as u64).collect::<Vec<_>>();
    let mut deleted = (2..=ROWS as u64 + 1).collect::<Vec<_>>();
    *deleted.last_mut().unwrap() = u64::MAX;
    let created_bytes = created
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let deleted_bytes = deleted
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let shard = MultiShardProbeShard {
        resident: std::sync::Arc::clone(&resident),
        index,
        table_mask,
        hash_shift,
        projection_offsets: vec![0, (ROWS * 4) as u64],
        row_count: ROWS as u64,
        row_capacity: ROWS as u64,
        created_by: Some(std::sync::Arc::new(
            runtime
                .retain_device_memory_copy(0, &created_bytes)
                .expect("created versions"),
        )),
        deleted_by: Some(std::sync::Arc::new(
            runtime
                .retain_device_memory_copy(0, &deleted_bytes)
                .expect("deleted versions"),
        )),
        has_postings: true,
        min: 77,
        max: 77,
    };
    let (columns, _) = resident
        .submit_multi_shard_i32_index_probe_dense(&[shard], &[77], 257)
        .expect("submit posting-chain point read")
        .complete_detached_columnar()
        .expect("complete posting-chain point read");
    assert_eq!(columns.status, vec![1]);
    assert_eq!(columns.values, vec![77, 256]);
}

/// M1 BAKEOFF micro-bench: the write-locate kernel's AMORTIZATION curve — us/needle at batch
/// sizes 1..256 across a realistic shard set. Decides A-vs-B viability: if launch-dominated
/// (us/needle falls ~linearly with batch size), a wave of ~17 needles amortizes ~17x and both
/// batching designs reach the host-oracle class; if compute-dominated, batching gains less.
/// Prints a table; asserts only that batching HELPS (256-batch us/needle < single-needle).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn write_locate_kernel_amortization_curve() {
    use std::time::Instant;
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    if !runtime.snapshot().driver_available {
        return;
    }
    // 8 shards of 4096 keys each (a mid-size resident table, ~32k rows — the SLO regime).
    const SHARDS: usize = 8;
    const PER_SHARD: usize = 4096;
    let mut all_keys: Vec<i32> = Vec::new();
    let mut shards = Vec::new();
    for s in 0..SHARDS {
        let keys: Vec<i32> = (0..PER_SHARD).map(|i| (s * PER_SHARD + i) as i32).collect();
        all_keys.extend_from_slice(&keys);
        let (index_words, table_mask, hash_shift) = build_pk_hash(&keys);
        let bytes = posting_index_bytes(&index_words, keys.len());
        let Ok(mem) = runtime.retain_device_memory_copy(0, &bytes) else {
            return;
        };
        shards.push(WriteLocateShard {
            index: std::sync::Arc::new(mem),
            table_mask,
            hash_shift,
            row_count: PER_SHARD as u32,
        });
    }
    let ctx = std::sync::Arc::clone(&shards[0].index);
    // Warm up (JIT the kernel).
    let _ = ctx.submit_multi_shard_i32_write_locate(&shards, &all_keys[..1], 2);
    eprintln!("  batch |   total us | us/needle");
    let mut single_us = 0.0;
    let mut big_us = 0.0;
    for &batch in &[1usize, 4, 16, 32, 64, 256] {
        // Spread needles across shards (real workload hits many shards).
        let needles: Vec<i32> = (0..batch)
            .map(|i| all_keys[(i * 997) % all_keys.len()])
            .collect();
        const REPS: u32 = 200;
        let t0 = Instant::now();
        for _ in 0..REPS {
            let r = ctx
                .submit_multi_shard_i32_write_locate(&shards, &needles, 2)
                .expect("locate");
            std::hint::black_box(&r.count);
        }
        let per_call_us = t0.elapsed().as_micros() as f64 / REPS as f64;
        let per_needle = per_call_us / batch as f64;
        eprintln!("  {batch:>5} | {per_call_us:>9.1} | {per_needle:>8.3}");
        if batch == 1 {
            single_us = per_needle;
        }
        if batch == 256 {
            big_us = per_needle;
        }
    }
    assert!(
        big_us < single_us,
        "batching must amortize: 256-batch {big_us:.3} us/needle vs single {single_us:.3}"
    );
}

/// M1 device WRITE-LOCATE kernel: probe multi-shard device indexes for a batch of needles and
/// verify EVERY (shard, slot) hit matches a naive host scan of the same key sets — including a
/// CROSS-SHARD DUPLICATE (a key in two shards, the SV5 old+new case) which must emit BOTH hits.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn write_locate_kernel_matches_host_scan() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    if !runtime.snapshot().driver_available {
        return;
    }
    // Two shards. Key 130 is in BOTH (the cross-shard version case).
    let shard_keys: [Vec<i32>; 2] = [vec![10, 20, 130, 40], vec![130, 200, 300]];
    let mut shards = Vec::new();
    for keys in &shard_keys {
        let (index_words, table_mask, hash_shift) = build_pk_hash(keys);
        let bytes = posting_index_bytes(&index_words, keys.len());
        let Ok(mem) = runtime.retain_device_memory_copy(0, &bytes) else {
            return; // no device -> skip
        };
        shards.push(WriteLocateShard {
            index: std::sync::Arc::new(mem),
            table_mask,
            hash_shift,
            row_count: keys.len() as u32,
        });
    }
    let needles = [130, 20, 999, 300, 10];
    // The launch context is any device buffer on the GPU (the method mirrors the v2 probe).
    let ctx = std::sync::Arc::clone(&shards[0].index);
    let result = ctx
        .submit_multi_shard_i32_write_locate(&shards, &needles, 2)
        .expect("write-locate kernel");
    for (ni, &needle) in needles.iter().enumerate() {
        // Naive host truth: every (shard, local slot) whose key == needle.
        let mut expected: Vec<(u32, u32)> = Vec::new();
        for (si, keys) in shard_keys.iter().enumerate() {
            for (row, &k) in keys.iter().enumerate() {
                if k == needle {
                    expected.push((si as u32, row as u32));
                }
            }
        }
        let count = result.count[ni];
        assert_ne!(
            count,
            u32::MAX,
            "needle {needle}: no overflow at max_hits=2"
        );
        assert_eq!(count as usize, expected.len(), "needle {needle}: hit count");
        let mut got: Vec<(u32, u32)> = (0..count as usize)
            .map(|h| {
                let idx = ni * result.max_hits as usize + h;
                (result.shard_idx[idx], result.slot[idx])
            })
            .collect();
        got.sort_unstable();
        expected.sort_unstable();
        assert_eq!(got, expected, "needle {needle}: exact (shard, slot) hits");
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn write_locate_inputs_fail_closed_and_leave_context_reusable() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let (valid_words, table_mask, hash_shift) = build_pk_hash(&[10]);
    let valid_bytes = posting_index_bytes(&valid_words, 1);
    let valid_index = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &valid_bytes)
            .expect("valid device index"),
    );
    let ctx = std::sync::Arc::clone(&valid_index);

    let bad_geometry = [WriteLocateShard {
        index: std::sync::Arc::clone(&valid_index),
        table_mask: table_mask + 2,
        hash_shift,
        row_count: 1,
    }];
    assert!(
        ctx.submit_multi_shard_i32_write_locate(&bad_geometry, &[10], 1)
            .is_err()
    );

    let mut corrupt_words = valid_words.clone();
    let packed = corrupt_words.iter_mut().find(|word| **word != 0).unwrap();
    *packed = (*packed & 0xffff_ffff_0000_0000) | 100;
    let corrupt_bytes = posting_index_bytes(&corrupt_words, 1);
    let corrupt_index = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &corrupt_bytes)
            .expect("corrupt-slot device index"),
    );
    let corrupt = [WriteLocateShard {
        index: std::sync::Arc::clone(&corrupt_index),
        table_mask,
        hash_shift,
        row_count: 1,
    }];
    assert!(
        ctx.submit_multi_shard_i32_write_locate(&corrupt, &[10], 1)
            .is_err(),
        "packed slot beyond logical rows must fail closed"
    );

    let short_region = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &0u64.to_le_bytes())
            .expect("short version region"),
    );
    let short_visible = [VisibleLocateShard {
        index: std::sync::Arc::clone(&valid_index),
        table_mask,
        hash_shift,
        row_count: 2,
        created_by: Some(short_region),
        deleted_by: None,
        row_id: None,
    }];
    assert!(
        ctx.submit_multi_shard_i32_visible_locate(&short_visible, &[10], &[1])
            .is_err()
    );

    let corrupt_visible = [VisibleLocateShard {
        index: corrupt_index,
        table_mask,
        hash_shift,
        row_count: 1,
        created_by: None,
        deleted_by: None,
        row_id: None,
    }];
    assert!(
        ctx.submit_multi_shard_i32_visible_locate(&corrupt_visible, &[10], &[1])
            .is_err()
    );

    let mut cyclic_bytes = valid_bytes.clone();
    let next_offset = valid_words.len() * std::mem::size_of::<u64>();
    cyclic_bytes[next_offset..next_offset + 4].copy_from_slice(&1_u32.to_le_bytes());
    let cyclic_index = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &cyclic_bytes)
            .expect("self-linked posting index"),
    );
    let cyclic = [WriteLocateShard {
        index: std::sync::Arc::clone(&cyclic_index),
        table_mask,
        hash_shift,
        row_count: 1,
    }];
    assert!(
        ctx.submit_multi_shard_i32_write_locate(&cyclic, &[10], 1)
            .is_err(),
        "a self-linked posting must terminate and fail closed in write-locate"
    );
    let cyclic_visible = [VisibleLocateShard {
        index: cyclic_index,
        table_mask,
        hash_shift,
        row_count: 1,
        created_by: None,
        deleted_by: None,
        row_id: None,
    }];
    assert!(
        ctx.submit_multi_shard_i32_visible_locate(&cyclic_visible, &[10], &[1])
            .is_err(),
        "a self-linked posting must terminate and fail closed in visible-locate"
    );

    if runtime.snapshot().device_count > 1 {
        let foreign = std::sync::Arc::new(
            runtime
                .retain_device_memory_copy(1, &valid_bytes)
                .expect("foreign-context index"),
        );
        let foreign_shard = [WriteLocateShard {
            index: foreign,
            table_mask,
            hash_shift,
            row_count: 1,
        }];
        assert!(
            ctx.submit_multi_shard_i32_write_locate(&foreign_shard, &[10], 1)
                .is_err()
        );
    }

    let valid = [WriteLocateShard {
        index: std::sync::Arc::clone(&valid_index),
        table_mask,
        hash_shift,
        row_count: 1,
    }];
    let result = ctx
        .submit_multi_shard_i32_write_locate(&valid, &[10], 1)
        .expect("valid write locate after rejected inputs");
    assert_eq!(result.count, vec![1]);
    assert_eq!(result.slot, vec![0]);

    let identity = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &42u64.to_le_bytes())
            .expect("visible-locate identity region"),
    );
    let created_by = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &7u64.to_le_bytes())
            .expect("visible-locate created-by region"),
    );
    let deleted_by = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &9u64.to_le_bytes())
            .expect("visible-locate deleted-by region"),
    );
    let valid_visible = [VisibleLocateShard {
        index: valid_index,
        table_mask,
        hash_shift,
        row_count: 1,
        created_by: Some(created_by),
        deleted_by: Some(deleted_by),
        row_id: Some(identity),
    }];
    let result = ctx
        .submit_multi_shard_i32_visible_locate(&valid_visible, &[10], &[8])
        .expect("valid visible locate after rejected inputs");
    assert_eq!(result.count, vec![1]);
    assert_eq!(result.slot, vec![0]);
    assert_eq!(result.row_id, vec![42]);
    assert_eq!(result.latest_write, vec![9]);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn write_apply_inputs_fail_closed_and_leave_context_reusable() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let owner = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &[0_u8; 64])
            .expect("write owner"),
    );

    assert!(owner.submit_i32_index_insert(2, 30, &[7], 0).is_err());
    assert!(
        owner
            .submit_i32_index_insert(7, 29, &[7], u32::MAX)
            .is_err()
    );
    assert!(owner.submit_i32_index_insert(7, 29, &[7], 4).is_err());

    let invalid_column = [CudaWriteDestination {
        memory: std::sync::Arc::clone(&owner),
        byte_offset: 63,
    }];
    let invalid_request = FusedApplyRequest {
        columns: &invalid_column,
        values: &[7],
        stamps: &[1],
        created_by: CudaWriteDestination {
            memory: std::sync::Arc::clone(&owner),
            byte_offset: 8,
        },
        row_ids: None,
        index: None,
        base_row: 0,
        header: CudaWriteDestination {
            memory: std::sync::Arc::clone(&owner),
            byte_offset: 0,
        },
    };
    assert!(owner.submit_i32_fused_apply(&invalid_request).is_err());
    let misaligned_column = [CudaWriteDestination {
        memory: std::sync::Arc::clone(&owner),
        byte_offset: 1,
    }];
    let misaligned_request = FusedApplyRequest {
        columns: &misaligned_column,
        values: &[7],
        stamps: &[1],
        created_by: CudaWriteDestination {
            memory: std::sync::Arc::clone(&owner),
            byte_offset: 24,
        },
        row_ids: None,
        index: None,
        base_row: 0,
        header: CudaWriteDestination {
            memory: std::sync::Arc::clone(&owner),
            byte_offset: 0,
        },
    };
    assert!(owner.submit_i32_fused_apply(&misaligned_request).is_err());

    if runtime.snapshot().device_count > 1 {
        let foreign = std::sync::Arc::new(
            runtime
                .retain_device_memory_copy(1, &[0_u8; 64])
                .expect("foreign write owner"),
        );
        let foreign_column = [CudaWriteDestination {
            memory: foreign,
            byte_offset: 0,
        }];
        let foreign_request = FusedApplyRequest {
            columns: &foreign_column,
            values: &[7],
            stamps: &[1],
            created_by: CudaWriteDestination {
                memory: std::sync::Arc::clone(&owner),
                byte_offset: 8,
            },
            row_ids: None,
            index: None,
            base_row: 0,
            header: CudaWriteDestination {
                memory: std::sync::Arc::clone(&owner),
                byte_offset: 0,
            },
        };
        assert!(owner.submit_i32_fused_apply(&foreign_request).is_err());
    }

    let mut malformed_text = Vec::new();
    malformed_text.extend_from_slice(&0_u64.to_le_bytes());
    malformed_text.extend_from_slice(&2_u64.to_le_bytes());
    malformed_text.push(b'x');
    let text_owner = runtime
        .retain_device_memory_copy(0, &malformed_text)
        .expect("malformed text owner");
    let malformed = [CudaCompoundFoldColumn::Text {
        offsets_byte_offset: 0,
        bytes_byte_offset: 16,
        bytes_len: 1,
    }];
    assert!(
        text_owner
            .submit_compound_fold_fingerprints(&malformed, 1)
            .is_err()
    );

    let out_of_bounds = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 63,
        width_words: 1,
    }];
    assert!(
        owner
            .submit_compound_fold_fingerprints(&out_of_bounds, 1)
            .is_err()
    );
    let misaligned = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 1,
        width_words: 1,
    }];
    assert!(
        owner
            .submit_compound_fold_fingerprints(&misaligned, 1)
            .is_err()
    );
    let misaligned_text = [CudaCompoundFoldColumn::Text {
        offsets_byte_offset: 1,
        bytes_byte_offset: 32,
        bytes_len: 1,
    }];
    assert!(
        owner
            .submit_compound_fold_fingerprints(&misaligned_text, 1)
            .is_err()
    );

    let valid_column = [CudaWriteDestination {
        memory: std::sync::Arc::clone(&owner),
        byte_offset: 16,
    }];
    let valid_request = FusedApplyRequest {
        columns: &valid_column,
        values: &[7],
        stamps: &[1],
        created_by: CudaWriteDestination {
            memory: std::sync::Arc::clone(&owner),
            byte_offset: 24,
        },
        row_ids: None,
        index: None,
        base_row: 0,
        header: CudaWriteDestination {
            memory: std::sync::Arc::clone(&owner),
            byte_offset: 0,
        },
    };
    let unindexed_status = owner
        .submit_i32_fused_apply_status(&valid_request)
        .expect("valid fused apply after rejected inputs");
    assert_eq!(
        unindexed_status,
        CudaResidentIndexStatus {
            declined: false,
            created_posting: false,
        }
    );

    let (index_words, table_mask, hash_shift) = build_pk_hash(&[7]);
    let version_index = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &posting_index_bytes(&index_words, 2))
            .expect("version posting index"),
    );
    let indexed_request = FusedApplyRequest {
        columns: &valid_column,
        values: &[7],
        stamps: &[2],
        created_by: CudaWriteDestination {
            memory: std::sync::Arc::clone(&owner),
            byte_offset: 32,
        },
        row_ids: None,
        index: Some(CudaWriteIndex {
            memory: std::sync::Arc::clone(&version_index),
            table_mask,
            hash_shift,
            key_column: 0,
        }),
        base_row: 1,
        header: CudaWriteDestination {
            memory: std::sync::Arc::clone(&owner),
            byte_offset: 0,
        },
    };
    let fused_status = owner
        .submit_i32_fused_apply_status(&indexed_request)
        .expect("same-key fused append prepends a posting");
    assert!(!fused_status.declined);
    assert!(fused_status.created_posting);
    let versions = version_index
        .submit_multi_shard_i32_write_locate(
            &[WriteLocateShard {
                index: std::sync::Arc::clone(&version_index),
                table_mask,
                hash_shift,
                row_count: 2,
            }],
            &[7],
            2,
        )
        .expect("locate fused version postings");
    assert_eq!(versions.count, vec![2]);
    assert_eq!(versions.slot, vec![1, 0]);

    let valid = [CudaCompoundFoldColumn::Fixed {
        byte_offset: 0,
        width_words: 1,
    }];
    let fingerprints = owner
        .submit_compound_fold_fingerprints(&valid, 1)
        .expect("context reusable after rejected and device-reported inputs");
    assert_eq!(fingerprints.len(), 1);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn resident_sidecar_inputs_fail_closed_and_leave_context_reusable() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    let mut owner_bytes = vec![0_u8; 64];
    owner_bytes[32..36].fill(0xff);
    let owner = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &owner_bytes)
            .expect("sidecar destination"),
    );
    owner
        .scatter_u64_slots(&[1, 3], &[9, 11])
        .expect("bounded scatter");
    assert_eq!(
        owner
            .read_resident_u64_column(0, 4)
            .expect("scatter readback"),
        vec![0, 9, 0, 11]
    );
    assert!(owner.scatter_u64_slots(&[8], &[1]).is_err());
    assert!(owner.scatter_u64_slots(&[0, 1], &[1]).is_err());
    assert!(owner.scatter_u64_slots(&[], &[1]).is_err());

    owner
        .set_bool_bitmap_range(32, 0, &[1, 0, 1])
        .expect("bounded bool set");
    assert_eq!(
        owner.read_resident_bytes(32, 4).expect("bool readback"),
        0xffff_fffd_u32.to_le_bytes()
    );
    assert!(owner.set_bool_bitmap_range(32, 0, &[2]).is_err());
    assert!(owner.set_bool_bitmap_range(63, 0, &[1]).is_err());

    let bool_source = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &5_u32.to_le_bytes())
            .expect("bool source"),
    );
    let bool_source = CudaSidecarSource {
        memory: bool_source,
        byte_offset: 0,
    };
    let bool_dst = runtime
        .retain_device_memory_copy(0, &[0xff_u8; 8])
        .expect("bool gather destination");
    bool_dst
        .gather_bool_bitmap_from_shard(0, 3, &bool_source, 3)
        .expect("bool gather");
    assert_eq!(
        bool_dst
            .read_resident_bytes(0, 4)
            .expect("bool gather readback"),
        0xffff_ffef_u32.to_le_bytes()
    );
    assert!(
        bool_dst
            .gather_bool_bitmap_from_shard(5, 0, &bool_source, 3)
            .is_err()
    );

    let null_dst = runtime
        .retain_device_memory_copy(0, &[0_u8; 8])
        .expect("null gather destination");
    null_dst
        .gather_null_bitmap_from_shard(0, 3, &bool_source, 3)
        .expect("null gather");
    assert_eq!(
        null_dst
            .read_resident_bytes(0, 4)
            .expect("null gather readback"),
        40_u32.to_le_bytes()
    );
    let alias_source = CudaSidecarSource {
        memory: std::sync::Arc::clone(&owner),
        byte_offset: 32,
    };
    assert!(
        owner
            .gather_bool_bitmap_from_shard(32, 1, &alias_source, 3)
            .is_err()
    );

    let text_bytes = [0_u64, 2, 3]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect::<Vec<_>>();
    let mut text_payload = text_bytes;
    text_payload.extend_from_slice(b"abc");
    let text_source = CudaTextOffsetSource {
        memory: std::sync::Arc::new(
            runtime
                .retain_device_memory_copy(0, &text_payload)
                .expect("text offsets"),
        ),
        offsets_byte_offset: 0,
        bytes_byte_offset: 24,
        bytes_len: 3,
    };
    let text_dst = std::sync::Arc::new(
        runtime
            .retain_device_memory_copy(0, &[0_u8; 32])
            .expect("text rebase destination"),
    );
    text_dst
        .rebase_text_offsets_from_shard(0, 0, 5, &text_source, 3)
        .expect("valid text rebase");
    assert_eq!(
        text_dst
            .read_resident_u64_column(0, 3)
            .expect("text readback"),
        vec![5, 7, 8]
    );

    for malformed in [[1_u64, 2, 3], [0, 3, 2], [0, 2, 4], [0, 1, 2]] {
        let bytes = malformed
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>();
        let mut payload = bytes;
        payload.extend_from_slice(b"abc");
        let source = CudaTextOffsetSource {
            memory: std::sync::Arc::new(
                runtime
                    .retain_device_memory_copy(0, &payload)
                    .expect("malformed text offsets"),
            ),
            offsets_byte_offset: 0,
            bytes_byte_offset: 24,
            bytes_len: 3,
        };
        assert!(
            text_dst
                .rebase_text_offsets_from_shard(0, 0, 0, &source, 3)
                .is_err()
        );
    }
    assert!(
        text_dst
            .rebase_text_offsets_from_shard(0, 0, u64::MAX, &text_source, 3)
            .is_err()
    );
    let missing_blob = CudaTextOffsetSource {
        memory: std::sync::Arc::new(
            runtime
                .retain_device_memory_copy(0, &[0_u8; 24])
                .expect("offsets-only allocation"),
        ),
        offsets_byte_offset: 0,
        bytes_byte_offset: 24,
        bytes_len: 3,
    };
    assert!(
        text_dst
            .rebase_text_offsets_from_shard(0, 0, 0, &missing_blob, 3)
            .is_err()
    );
    let aliased_text = CudaTextOffsetSource {
        memory: std::sync::Arc::clone(&text_dst),
        offsets_byte_offset: 0,
        bytes_byte_offset: 24,
        bytes_len: 0,
    };
    assert!(
        text_dst
            .rebase_text_offsets_from_shard(0, 0, 0, &aliased_text, 3)
            .is_err()
    );

    if runtime.snapshot().device_count > 1 {
        let foreign = CudaSidecarSource {
            memory: std::sync::Arc::new(
                runtime
                    .retain_device_memory_copy(1, &5_u32.to_le_bytes())
                    .expect("foreign sidecar source"),
            ),
            byte_offset: 0,
        };
        assert!(
            bool_dst
                .gather_bool_bitmap_from_shard(0, 0, &foreign, 3)
                .is_err()
        );
    }

    text_dst
        .rebase_text_offsets_from_shard(0, 0, 0, &text_source, 3)
        .expect("context reusable after rejected sidecar inputs");
}
