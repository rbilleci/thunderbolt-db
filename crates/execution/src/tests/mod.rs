use super::*;

#[test]
fn radix_input_validation_is_total_before_cuda() {
    let aligned = 0x1000_u64;
    assert_eq!(validate_i64_argsort_host_len(2), Ok((2, 16)));
    if let Ok(oversized) = usize::try_from(u64::from(u32::MAX) + 1) {
        assert!(validate_i64_argsort_host_len(oversized).is_err());
    }
    assert_eq!(validate_i64_argsort_input(7, 7, aligned, 16, 2), Ok(2));
    assert!(validate_i64_argsort_input(7, 8, aligned, 16, 2).is_err());
    assert!(validate_i64_argsort_input(7, 7, aligned + 1, 16, 2).is_err());
    assert!(validate_i64_argsort_input(7, 7, aligned, 8, 2).is_err());
    assert!(validate_i64_argsort_input(7, 7, u64::MAX - 3, 16, 2).is_err());
    assert!(
        validate_i64_argsort_input(7, 7, aligned, usize::MAX, u64::from(u32::MAX) + 1).is_err()
    );
    // Empty work still validates ownership and alignment rather than bypassing the contract.
    assert!(validate_i64_argsort_input(7, 8, aligned, 0, 0).is_err());
    assert!(validate_i64_argsort_input(7, 7, aligned + 1, 0, 0).is_err());
}

fn gpu_op(id: u16) -> PlannedOp {
    PlannedOp {
        name: "scan".to_string(),
        target: DeviceTarget::Gpu(id),
    }
}

#[test]
fn ordered_index_result_retyping_preserves_allocation_and_bits() {
    let mut signed = Vec::with_capacity(8);
    signed.extend_from_slice(&[0, 1, i32::MAX, i32::MIN, -1]);
    let ptr = signed.as_ptr().cast::<u32>();
    let capacity = signed.capacity();

    let unsigned = i32_bits_into_u32(signed);

    assert_eq!(unsigned.as_ptr(), ptr);
    assert_eq!(unsigned.capacity(), capacity);
    assert_eq!(unsigned, [0, 1, i32::MAX as u32, 1 << 31, u32::MAX]);
}

#[test]
fn ordered_i32_compaction_preflight_is_total() {
    assert_eq!(validate_ordered_i32_comparison(5, 5), Ok(()));
    assert_eq!(
        validate_ordered_i32_comparison(6, 5),
        Err(CudaRuntimeProbeError::UnsupportedComparison(6))
    );

    assert_eq!(
        validate_ordered_i32_index_domain(u64::from(u32::MAX)),
        Ok(())
    );
    assert!(matches!(
        validate_ordered_i32_index_domain(u64::from(u32::MAX) + 1),
        Err(CudaRuntimeProbeError::InvalidInputLength(_))
    ));
    assert_eq!(validate_ordered_i32_context_identity(7, 7, 64), Ok(()));
    assert_eq!(
        validate_ordered_i32_context_identity(7, 8, 64),
        Err(CudaRuntimeProbeError::InvalidInputLength(64))
    );

    assert_eq!(validate_ordered_i32_input_window(12, 4, 2), Ok(()));
    assert_eq!(
        validate_ordered_i32_input_window(11, 4, 2),
        Err(CudaRuntimeProbeError::InvalidInputLength(12))
    );
    assert_eq!(
        validate_ordered_i32_input_window(12, 1, 2),
        Err(CudaRuntimeProbeError::InvalidInputLength(1))
    );
    assert_eq!(validate_ordered_i32_input_window(8, 8, 0), Ok(()));
    assert_eq!(
        validate_ordered_i32_input_window(8, 12, 0),
        Err(CudaRuntimeProbeError::InvalidInputLength(12))
    );
    assert!(matches!(
        validate_ordered_i32_input_window(u64::MAX, u64::MAX - 3, 1),
        Err(CudaRuntimeProbeError::InvalidInputLength(_))
    ));
}

/// Build an int4 PK hash table in the kernel's format: open-addressing `(key<<32)|(row+1)`,
/// 0 = empty, size = next_pow2(2*n), fib hash `(key*0x9E3779B1) >> shift`, linear probe.
/// Returns `(index_words, table_mask, hash_shift)` — mirrors the engine's host builder byte-for-byte.
#[cfg(test)]
fn build_pk_hash(keys: &[i32]) -> (Vec<u64>, u32, u32) {
    let table_size = (keys.len() as u64 * 2).next_power_of_two().max(2);
    let table_mask = (table_size - 1) as u32;
    let hash_shift = 32 - table_size.trailing_zeros();
    let mut index = vec![0u64; table_size as usize];
    for (row, &key) in keys.iter().enumerate() {
        let kb = key as u32;
        let mut slot = (kb.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
        loop {
            if index[slot as usize] == 0 {
                index[slot as usize] = ((kb as u64) << 32) | (row as u64 + 1);
                break;
            }
            slot = (slot + 1) & table_mask;
        }
    }
    (index, table_mask, hash_shift)
}

include!("write_index.rs");
include!("version_conflict.rs");

#[test]
fn cuda_driver_runtime_routes_only_detected_devices() {
    let router = DeviceRouter::new(CudaDriverRuntime::from_device_count(1));

    assert_eq!(router.route(&gpu_op(0)), RouteDecision::Gpu(0));
    assert_eq!(
        router.route(&gpu_op(1)),
        RouteDecision::CpuFallback {
            requested_gpu: 1,
            reason: GpuFallbackReason::Unavailable,
        }
    );
}

#[test]
fn cuda_driver_runtime_synthetic_snapshot_tracks_device_slots() {
    let runtime = CudaDriverRuntime::from_device_count(2);
    let snapshot = runtime.snapshot();

    assert!(snapshot.driver_available);
    assert_eq!(snapshot.driver_version, None);
    assert_eq!(snapshot.device_count, 2);
    assert_eq!(snapshot.devices.len(), 2);
    assert_eq!(snapshot.devices[0].id, 0);
    assert_eq!(snapshot.devices[0].name, "cuda-device-0");
    assert_eq!(snapshot.devices[0].total_memory_bytes, 0);
    assert_eq!(snapshot.devices[1].id, 1);
}

#[test]
fn unavailable_cuda_driver_runtime_falls_back_cleanly() {
    let runtime = CudaDriverRuntime::unavailable();
    let snapshot = runtime.snapshot();

    assert!(!snapshot.driver_available);
    assert_eq!(snapshot.driver_version, None);
    assert_eq!(snapshot.device_count, 0);
    assert!(snapshot.devices.is_empty());
    assert_eq!(
        runtime.can_run(0, &gpu_op(0)),
        Err(GpuFallbackReason::Unavailable)
    );
}

#[test]
fn unavailable_cuda_driver_runtime_rejects_smoke_launch() {
    let runtime = CudaDriverRuntime::unavailable();

    assert_eq!(
        runtime.launch_smoke_add_one(41),
        Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
    );
}

#[test]
fn unavailable_cuda_driver_runtime_rejects_filter_launch() {
    let runtime = CudaDriverRuntime::unavailable();

    assert_eq!(
        runtime.filter_equal_u32_mask(&[7, 8, 7], 7),
        Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
    );
    assert_eq!(
        runtime.filter_equal_bytes_mask(&[b"open".as_slice()], b"open"),
        Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
    );
    assert_eq!(
        runtime.filter_bytes_range_mask(&[b"acct:1".as_slice()], b"acct:", b"acct:9"),
        Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
    );
    assert_eq!(
        runtime.filter_all_mask(3),
        Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
    );
    let batch = CudaMvccRowBatch::from_key_values([(b"k".as_slice(), b"v".as_slice())]).unwrap();
    assert_eq!(
        runtime.mvcc_visibility_mask(&batch, 1),
        Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
    );
    assert_eq!(
        runtime.mvcc_row_batch_lengths(&batch),
        Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
    );
    assert_eq!(
        runtime.verify_device_memory_copy(0, b"resident-snapshot"),
        Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
    );
    assert!(matches!(
        runtime.retain_device_memory_copy(0, b"resident-snapshot"),
        Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
    ));
    assert!(matches!(
        runtime.retain_device_memory_chunks(
            0,
            16,
            &[CudaDeviceMemoryChunk {
                byte_offset: 8,
                bytes: b"chunk",
            }]
        ),
        Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
    ));
    assert_eq!(
        runtime.verify_device_memory_copy(0, b""),
        Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_driver_runtime_probe_reports_local_devices() {
    let runtime = CudaDriverRuntime::probe().unwrap();
    let snapshot = runtime.snapshot();

    assert!(snapshot.driver_available);
    assert!(snapshot.device_count > 0);
    assert_eq!(snapshot.devices.len(), snapshot.device_count as usize);
    assert!(!snapshot.devices[0].name.is_empty());
    assert!(snapshot.devices[0].total_memory_bytes > 0);
    assert_eq!(runtime.can_run(0, &gpu_op(0)), Ok(()));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_driver_runtime_launches_smoke_kernel() {
    let runtime = CudaDriverRuntime::probe().unwrap();

    assert_eq!(runtime.launch_smoke_add_one(41).unwrap(), 42);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn append_owned_chunks_into_headroom_equals_a_fresh_upload() {
    // Slice 1a: appending rows IN PLACE into an open shard's headroom must produce byte-identical
    // device contents to a single fresh upload of the whole column — that equality is what lets us
    // replace the full re-admit (the dual-store tax) with an O(rows-appended) write.
    let runtime = CudaDriverRuntime::probe().expect("probe");
    // Allocate with HEADROOM: room for 16 i32 (64 bytes); seed the first 8.
    let allocated_bytes = 64_u64;
    let seed: Vec<i32> = (0..8).collect();
    let seed_bytes: Vec<u8> = seed.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mem = runtime
        .retain_device_memory_owned_chunks(
            0,
            allocated_bytes,
            std::iter::once(CudaOwnedDeviceMemoryChunk {
                byte_offset: 0,
                bytes: seed_bytes,
            }),
        )
        .expect("retain with headroom");
    // Append the next 4 i32 in place at the tail (offset 8*4 = 32) — no realloc.
    let tail: Vec<i32> = (8..12).collect();
    let tail_bytes: Vec<u8> = tail.iter().flat_map(|v| v.to_le_bytes()).collect();
    let appended = mem
        .append_owned_chunks(std::iter::once(CudaOwnedDeviceMemoryChunk {
            byte_offset: 32,
            bytes: tail_bytes.clone(),
        }))
        .expect("append into headroom");
    assert_eq!(appended, tail_bytes.len() as u64);
    // The 12 i32 read back must equal a fresh full upload of 0..12 (non-vacuous: seed=0..8,
    // appended=8..12, so a wrong offset/contents would corrupt the join).
    let read = mem.read_resident_i32_column(0, 12).expect("read back");
    assert_eq!(
        read,
        (0..12).collect::<Vec<i32>>(),
        "append must equal a fresh upload"
    );
    // A chunk that would overrun the allocation is REJECTED (caller must roll over to a new shard).
    let overrun = mem.append_owned_chunks(std::iter::once(CudaOwnedDeviceMemoryChunk {
        byte_offset: 60,
        bytes: vec![0_u8; 8],
    }));
    assert!(
        overrun.is_err(),
        "a chunk overrunning allocated_bytes must be rejected"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn open_shard_append_two_int4_columns_equals_full_build() {
    // Slice 1a END-TO-END: build a 2-column i32 OPEN shard (capacity C, seeded with R rows in the
    // capacity-padded layout: header + col0[C] + col1[C]), append K rows into the HEADROOM of both
    // sections + bump the header, then read each column back at its capacity-based offset — the
    // result must equal a fresh build of all R+K rows (incremental append == full rebuild).
    let runtime = CudaDriverRuntime::probe().expect("probe");
    let capacity = 8_usize;
    let r = 3_usize; // seeded rows
    let k = 2_usize; // appended rows
    let header = 8_usize;
    let col = capacity * 4; // bytes per capacity-padded i32 section
    let allocated = (header + 2 * col) as u64;
    // Padded initial buffer: header = R; col0 = ids 0..R then zero headroom; col1 = 0,10,20 then pad.
    let mut init = vec![0_u8; header + 2 * col];
    init[0..8].copy_from_slice(&(r as u64).to_le_bytes());
    for row in 0..r {
        init[header + row * 4..header + row * 4 + 4].copy_from_slice(&(row as i32).to_le_bytes());
        init[header + col + row * 4..header + col + row * 4 + 4]
            .copy_from_slice(&((row as i32) * 10).to_le_bytes());
    }
    let mem = runtime
        .retain_device_memory_owned_chunks(
            0,
            allocated,
            std::iter::once(CudaOwnedDeviceMemoryChunk {
                byte_offset: 0,
                bytes: init,
            }),
        )
        .expect("retain padded open shard");
    // Append K rows (ids R..R+K, balances *10) into both section tails + bump the live-row header.
    let new_ids: Vec<u8> = (r..r + k)
        .flat_map(|row| (row as i32).to_le_bytes())
        .collect();
    let new_bals: Vec<u8> = (r..r + k)
        .flat_map(|row| ((row as i32) * 10).to_le_bytes())
        .collect();
    mem.append_owned_chunks(vec![
        CudaOwnedDeviceMemoryChunk {
            byte_offset: 0,
            bytes: ((r + k) as u64).to_le_bytes().to_vec(),
        },
        CudaOwnedDeviceMemoryChunk {
            byte_offset: (header + r * 4) as u64,
            bytes: new_ids,
        },
        CudaOwnedDeviceMemoryChunk {
            byte_offset: (header + col + r * 4) as u64,
            bytes: new_bals,
        },
    ])
    .expect("append rows into open shard");
    // Read each column back at its capacity-based offset — must equal a fresh full build of R+K rows.
    let col0 = mem
        .read_resident_i32_column(header as u64, r + k)
        .expect("read col0");
    let col1 = mem
        .read_resident_i32_column((header + col) as u64, r + k)
        .expect("read col1");
    assert_eq!(col0, (0..(r + k) as i32).collect::<Vec<i32>>(), "id column");
    assert_eq!(
        col1,
        (0..(r + k) as i32).map(|i| i * 10).collect::<Vec<i32>>(),
        "balance column"
    );
    // The 8-byte header now records R+K live rows.
    let header_i32 = mem.read_resident_i32_column(0, 2).expect("read header");
    assert_eq!(
        header_i32[0],
        (r + k) as i32,
        "header tracks the live row count"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_index_probe_matches_equal_any_scan() {
    // R1a: the GPU index-probe point lookup must return EXACTLY the equal_any scan's rows.
    let runtime = CudaDriverRuntime::probe().expect("probe");
    let rows: u64 = 1000;
    // Distinct key values (so the index is a real map) + distinct payload (so the gather is load-bearing).
    let keys: Vec<i32> = (0..rows as i32).map(|r| r * 3 + 1).collect();
    let payload: Vec<i32> = (0..rows as i32).map(|r| r * 1000 + 7).collect();
    let mut buf: Vec<u8> = Vec::with_capacity(rows as usize * 8);
    for &k in &keys {
        buf.extend_from_slice(&k.to_le_bytes());
    }
    for &v in &payload {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let resident = runtime
        .retain_device_memory_copy(0, &buf)
        .expect("resident device memory");
    let key_offset = 0_u64;
    let payload_offset = rows * 4;
    let projections = [key_offset, payload_offset];

    // Open-addressing index over the key column (same format/hash as the wave probes).
    let table_size = ((rows * 2) as u32).next_power_of_two();
    let table_mask = table_size - 1;
    let hash_shift = 32 - table_size.trailing_zeros();
    let mut index = vec![0_u64; table_size as usize];
    for (r, &k) in keys.iter().enumerate() {
        let key = k as u32;
        let mut h = (key.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
        while index[h as usize] != 0 {
            h = (h + 1) & table_mask;
        }
        index[h as usize] = ((key as u64) << 32) | (r as u64 + 1);
    }
    let index_bytes: Vec<u8> = index.iter().flat_map(|e| e.to_le_bytes()).collect();
    let index_resident = Arc::new(
        runtime
            .retain_device_memory_copy(0, &index_bytes)
            .expect("index device memory"),
    );

    // Present needles (rows 5, 100, 999, 0) + one absent value.
    let needles = vec![keys[5], keys[100], keys[999], -12345, keys[0]];

    let scan = resident
        .submit_match_project_i32_equal_any_from_payload(key_offset, &needles, &projections, rows)
        .expect("scan submit");
    let mut scan_rows = scan.complete(&resident).expect("scan complete");

    let probe = resident
        .submit_match_project_i32_index_probe_from_payload(
            &index_resident,
            table_mask,
            hash_shift,
            &needles,
            &projections,
            rows,
        )
        .expect("index submit");
    let mut probe_rows = probe.complete(&resident).expect("index complete");

    // Match modulo append order (both append via the atomic counter in schedule order).
    scan_rows.sort_by_key(|r| (r.needle_index, r.row_index));
    probe_rows.sort_by_key(|r| (r.needle_index, r.row_index));
    assert_eq!(
        probe_rows, scan_rows,
        "index probe must return exactly the equal_any scan's rows"
    );
    assert_eq!(probe_rows.len(), 4, "4 present needles, 1 absent");
    let first = &probe_rows[0];
    assert_eq!(
        first.needle_index, 0,
        "first sorted row is needle 0 (keys[5])"
    );
    assert_eq!(first.row_index, 5);
    assert_eq!(
        first.values,
        vec![keys[5], payload[5]],
        "gather: (key, payload) at row 5"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_multi_shard_dense_probe_honors_captured_rows_and_version_twins() {
    let runtime = CudaDriverRuntime::probe().expect("probe");
    let build = |keys: &[i32], payload: &[i32]| {
        let mut bytes = Vec::new();
        for value in keys.iter().chain(payload) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let resident = Arc::new(runtime.retain_device_memory_copy(0, &bytes).unwrap());
        let size = ((keys.len() * 2) as u32).next_power_of_two();
        let mask = size - 1;
        let shift = 32 - size.trailing_zeros();
        let mut entries = vec![0_u64; size as usize];
        for (slot, &key) in keys.iter().enumerate() {
            let mut at = (key as u32).wrapping_mul(0x9E37_79B1) >> shift;
            at &= mask;
            while entries[at as usize] != 0 {
                at = (at + 1) & mask;
            }
            entries[at as usize] = ((key as u32 as u64) << 32) | (slot as u64 + 1);
        }
        let index_bytes: Vec<u8> = entries.iter().flat_map(|v| v.to_le_bytes()).collect();
        let index = Arc::new(runtime.retain_device_memory_copy(0, &index_bytes).unwrap());
        (resident, index, mask, shift)
    };

    // An append-mutated index may be ahead of a reader's captured descriptor. With no birth region in that
    // older descriptor, row_count is the only sound upper bound: slot 2 must not leak through row_count 2.
    let (resident, index, mask, shift) = build(&[1, 2, 3], &[10, 20, 30]);
    let malformed_zone = MultiShardProbeShard {
        resident: Arc::clone(&resident),
        index: Arc::clone(&index),
        table_mask: mask,
        hash_shift: shift,
        projection_offsets: vec![0, 12],
        row_count: 2,
        created_by: None,
        deleted_by: None,
        min: 4,
        max: 3,
    };
    assert!(
        resident
            .prepare_multi_shard_i32_index_probe_dense(&[malformed_zone])
            .is_err(),
        "safe plan API rejects min > max before binary routing"
    );
    let malformed_projection = MultiShardProbeShard {
        resident: Arc::clone(&resident),
        index: Arc::clone(&index),
        table_mask: mask,
        hash_shift: shift,
        projection_offsets: vec![resident.metadata().allocated_bytes],
        row_count: 2,
        created_by: None,
        deleted_by: None,
        min: 1,
        max: 3,
    };
    assert!(
        resident
            .prepare_multi_shard_i32_index_probe_dense(&[malformed_projection])
            .is_err(),
        "safe plan API rejects an out-of-bounds projection span"
    );
    let short_created = Arc::new(
        runtime
            .retain_device_memory_copy(0, &1_u64.to_le_bytes())
            .unwrap(),
    );
    let malformed_visibility = MultiShardProbeShard {
        resident: Arc::clone(&resident),
        index: Arc::clone(&index),
        table_mask: mask,
        hash_shift: shift,
        projection_offsets: vec![0],
        row_count: 2,
        created_by: Some(short_created),
        deleted_by: None,
        min: 1,
        max: 3,
    };
    assert!(
        resident
            .prepare_multi_shard_i32_index_probe_dense(&[malformed_visibility])
            .is_err(),
        "safe plan API rejects a short MVCC visibility region"
    );
    let ahead = MultiShardProbeShard {
        resident: Arc::clone(&resident),
        index,
        table_mask: mask,
        hash_shift: shift,
        projection_offsets: vec![0, 12],
        row_count: 2,
        created_by: None,
        deleted_by: None,
        min: 1,
        max: 3,
    };
    let ahead_plan = resident
        .prepare_multi_shard_i32_index_probe_dense(&[ahead])
        .expect("prepare ahead-index plan");

    // Panic after async H2D/memset but before launch instrumentation. The submission owner must already exist
    // and drain before its pooled device buffers/stream or the borrowed needle bytes can be released.
    crate::point_read_dense::force_next_dense_panic(1);
    let submit_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = resident.submit_prepared_multi_shard_i32_index_probe_dense(&ahead_plan, &[3], 100);
    }));
    assert!(submit_panic.is_err(), "the submit panic hook fired");

    // Panic after asynchronous D2H is queued. The host-copy guard must drain before local pinned leases/Vecs
    // unwind, and the surrounding submission must then leave all shared pools reusable.
    let completion = resident
        .submit_prepared_multi_shard_i32_index_probe_dense(&ahead_plan, &[3], 100)
        .expect("submit before completion panic");
    crate::point_read_dense::force_next_dense_panic(2);
    let completion_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = completion.complete_detached_columnar();
    }));
    assert!(completion_panic.is_err(), "the completion panic hook fired");

    let (cols, _) = resident
        .submit_prepared_multi_shard_i32_index_probe_dense(&ahead_plan, &[3], 100)
        .unwrap()
        .complete_detached_columnar()
        .unwrap();
    assert_eq!(
        cols.status,
        vec![2],
        "ahead-index slot is outside captured row_count"
    );

    // A dead and live version of the same key can occupy adjacent probe slots in one dup-tolerant index.
    // The kernel must advance past the invisible twin and gather the visible one.
    let (resident, index, mask, shift) = build(&[3, 3], &[30, 31]);
    let created_bytes: Vec<u8> = [1_u64, 5].iter().flat_map(|v| v.to_le_bytes()).collect();
    let deleted_bytes: Vec<u8> = [5_u64, u64::MAX]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let twin = MultiShardProbeShard {
        resident: Arc::clone(&resident),
        index,
        table_mask: mask,
        hash_shift: shift,
        projection_offsets: vec![0, 8],
        row_count: 2,
        created_by: Some(Arc::new(
            runtime
                .retain_device_memory_copy(0, &created_bytes)
                .unwrap(),
        )),
        deleted_by: Some(Arc::new(
            runtime
                .retain_device_memory_copy(0, &deleted_bytes)
                .unwrap(),
        )),
        min: 3,
        max: 3,
    };
    let (cols, _) = resident
        .submit_multi_shard_i32_index_probe_dense(&[twin], &[3], 5)
        .unwrap()
        .complete_detached_columnar()
        .unwrap();
    assert_eq!(cols.status, vec![1]);
    assert_eq!(
        cols.values,
        vec![3, 31],
        "visible twin gathered after dead twin"
    );

    // Two separately valid shards may transiently expose the same visible key. Exercise the real multi-shard
    // kernel's status=3 contract directly: compact callers see the decline status, compatibility callers get a
    // typed duplicate error, and the same pooled resources remain reusable for a subsequent exact normal read.
    let (duplicate_resident_0, duplicate_index_0, duplicate_mask_0, duplicate_shift_0) =
        build(&[7], &[70]);
    let (duplicate_resident_1, duplicate_index_1, duplicate_mask_1, duplicate_shift_1) =
        build(&[7], &[71]);
    let duplicate_shards = [
        MultiShardProbeShard {
            resident: Arc::clone(&duplicate_resident_0),
            index: duplicate_index_0,
            table_mask: duplicate_mask_0,
            hash_shift: duplicate_shift_0,
            projection_offsets: vec![0, 4],
            row_count: 1,
            created_by: None,
            deleted_by: None,
            min: 7,
            max: 7,
        },
        MultiShardProbeShard {
            resident: duplicate_resident_1,
            index: duplicate_index_1,
            table_mask: duplicate_mask_1,
            hash_shift: duplicate_shift_1,
            projection_offsets: vec![0, 4],
            row_count: 1,
            created_by: None,
            deleted_by: None,
            min: 7,
            max: 7,
        },
    ];
    let duplicate_plan = duplicate_resident_0
        .prepare_multi_shard_i32_index_probe_dense(&duplicate_shards)
        .expect("prepare overlapping duplicate-key shards");
    let (duplicate_compact, _) = duplicate_resident_0
        .submit_prepared_multi_shard_i32_index_probe_dense(&duplicate_plan, &[7], 100)
        .expect("submit duplicate compact probe")
        .complete_detached_columnar_compact()
        .expect("complete duplicate compact probe");
    assert_eq!(
        duplicate_compact.status(),
        &[3],
        "the real GPU kernel declines a duplicate visible match"
    );
    assert_eq!(
        duplicate_resident_0
            .submit_prepared_multi_shard_i32_index_probe_dense(&duplicate_plan, &[7], 100)
            .expect("submit duplicate compatibility probe")
            .complete_detached_columnar(),
        Err(CudaRuntimeProbeError::DuplicatePointReadMatch(0)),
        "compatibility completion surfaces duplicate status as a typed error"
    );

    let normal_shard = [MultiShardProbeShard {
        resident: Arc::clone(&duplicate_resident_0),
        index: Arc::clone(&duplicate_shards[0].index),
        table_mask: duplicate_mask_0,
        hash_shift: duplicate_shift_0,
        projection_offsets: vec![0, 4],
        row_count: 1,
        created_by: None,
        deleted_by: None,
        min: 7,
        max: 7,
    }];
    let (normal, _) = duplicate_resident_0
        .submit_multi_shard_i32_index_probe_dense(&normal_shard, &[7], 100)
        .expect("submit normal probe after duplicate decline")
        .complete_detached_columnar()
        .expect("pooled resources remain reusable after duplicate decline");
    assert_eq!(normal.status, vec![1]);
    assert_eq!(normal.values, vec![7, 70]);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_chunk_bloom_probe_has_no_false_negatives() {
    let runtime = CudaDriverRuntime::probe().expect("probe");
    let build = |keys: &[i32]| {
        let bit_count = (keys.len() as u64 * 8).max(256).next_power_of_two();
        let bit_mask = (bit_count - 1) as u32;
        let mut words = vec![0_u32; (bit_count / 32) as usize];
        for &key in keys {
            let key = key as u32;
            let h1 = key.wrapping_mul(2_654_435_761);
            let h2 = (key ^ (key >> 16)).wrapping_mul(2_246_822_519) | 1;
            for i in 0..3_u32 {
                let bit = h1.wrapping_add(i.wrapping_mul(h2)) & bit_mask;
                words[(bit >> 5) as usize] |= 1_u32 << (bit & 31);
            }
        }
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        ChunkBloomProbeShard {
            bloom: Arc::new(runtime.retain_device_memory_copy(0, &bytes).unwrap()),
            bit_mask,
        }
    };
    let chunks = [vec![1, 2, 3], vec![10, 20], vec![-5, 77]];
    let blooms: Vec<_> = chunks.iter().map(|keys| build(keys)).collect();
    let needles = [1, 20, -5, 404];
    let candidates = blooms[0]
        .bloom
        .probe_chunk_blooms(&blooms, &needles)
        .expect("Bloom candidate probe");
    assert!(candidates[0].contains(&0), "key 1 must retain chunk 0");
    assert!(candidates[1].contains(&1), "key 20 must retain chunk 1");
    assert!(candidates[2].contains(&2), "key -5 must retain chunk 2");
    assert!(
        candidates[3].len() <= chunks.len(),
        "an absent key may false-positive but cannot produce an invalid chunk"
    );
}

include!("sort_join.rs");

mod cuda_generation_lifetime;
include!("cuda_paths.rs");

include!("resident_text.rs");

include!("context_aggregate.rs");

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_i64_compare_filters_and_projects_with_large_values() {
    // int8 (s64) comparison + projection on the GPU (the type matrix, doc 19). All values are
    // ABOVE i32::MAX, so a passing assert proves genuine 64-bit comparison (an int4 truncation
    // would wrap and mis-answer). Closed-form oracle: big[i] = BASE + i, big2[i] = BASE + (n-1-i).
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    const ROW_COUNT: u64 = 5000;
    const BASE: i64 = 4_000_000_000; // > i32::MAX (2_147_483_647)
    let big_off = std::mem::size_of::<u64>() as u64;
    let big2_off = big_off + ROW_COUNT * std::mem::size_of::<i64>() as u64;

    let mut header = Vec::new();
    header.extend_from_slice(&ROW_COUNT.to_le_bytes());
    let mut big_bytes = Vec::new();
    let mut big2_bytes = Vec::new();
    for row in 0..ROW_COUNT as i64 {
        big_bytes.extend_from_slice(&(BASE + row).to_le_bytes());
        big2_bytes.extend_from_slice(&(BASE + (ROW_COUNT as i64 - 1 - row)).to_le_bytes());
    }
    let allocated_len = big2_off + big2_bytes.len() as u64;
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            allocated_len,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: big_off,
                    bytes: &big_bytes,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: big2_off,
                    bytes: &big2_bytes,
                },
            ],
        )
        .expect("retain resident device memory");

    // big > BASE+4000 (gt=3): BASE+i > BASE+4000 <=> i > 4000 => [4001, 5000).
    let needle = BASE + 4000;
    let gt = resident
        .expr_i64_compare_scalar_filter(big_off, needle, false, 3, ROW_COUNT)
        .expect("big > needle");
    let gt_expected: Vec<u32> = (4001..ROW_COUNT).map(|i| i as u32).collect();
    assert_eq!(
        gt, gt_expected,
        "big > BASE+4000 => [4001, 5000) (64-bit needle)"
    );

    // scalar_on_left: needle < big (lt=1) is the SAME set, exercising the left/right flag.
    let flipped = resident
        .expr_i64_compare_scalar_filter(big_off, needle, true, 1, ROW_COUNT)
        .expect("needle < big");
    assert_eq!(flipped, gt_expected, "needle < big == big > needle");

    // col-vs-col big < big2 (lt=1): BASE+i < BASE+(4999-i) <=> 2i < 4999 => [0, 2500).
    let lt_cols = resident
        .expr_i64_compare_columns_filter(big_off, big2_off, 1, ROW_COUNT)
        .expect("big < big2");
    let lt_cols_expected: Vec<u32> = (0..2500).map(|i| i as u32).collect();
    assert_eq!(lt_cols, lt_cols_expected, "big < big2 => [0, 2500)");

    // Project big at the gt survivors: big[i] = BASE + i for i in [4001, 5000) — i64 values that
    // do not fit i32, proving the projection is genuinely 8-byte.
    let indices_u64: Vec<u64> = gt.iter().map(|&i| u64::from(i)).collect();
    let projected = resident
        .project_i64_rows_from_payload(big_off, &indices_u64)
        .expect("project big at survivors");
    let projected_expected: Vec<i64> = (4001..ROW_COUNT as i64).map(|i| BASE + i).collect();
    assert_eq!(
        projected, projected_expected,
        "projected big == BASE + i for the survivors (i64, above i32::MAX)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_i128_compare_filters_and_projects_beyond_i64_range() {
    // numeric (i128) comparison + projection on the GPU (the type matrix, doc 19). Mantissas
    // exceed the i64 range and exercise BOTH limbs + signed values, so a pass proves genuine
    // 128-bit SIGNED compare (an i64 truncation would wrap). Closed-form oracles.
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    const ROW_COUNT: u64 = 5000;
    const N: i64 = ROW_COUNT as i64;
    let base: i128 = 1i128 << 70; // high limb nonzero, low limb 0
    let elem = std::mem::size_of::<i128>() as u64; // 16
    let big_off = std::mem::size_of::<u64>() as u64;
    let big2_off = big_off + ROW_COUNT * elem;
    let hi_off = big2_off + ROW_COUNT * elem;
    let neg_off = hi_off + ROW_COUNT * elem;

    let mut header = Vec::new();
    header.extend_from_slice(&ROW_COUNT.to_le_bytes());
    let mut big_bytes = Vec::new();
    let mut big2_bytes = Vec::new();
    let mut hi_bytes = Vec::new();
    let mut neg_bytes = Vec::new();
    for row in 0..N {
        big_bytes.extend_from_slice(&(base + row as i128).to_le_bytes());
        big2_bytes.extend_from_slice(&(base + (N - 1 - row) as i128).to_le_bytes());
        hi_bytes.extend_from_slice(&((row as i128) << 64).to_le_bytes());
        neg_bytes.extend_from_slice(&(-base - row as i128).to_le_bytes());
    }
    let allocated_len = neg_off + neg_bytes.len() as u64;
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            allocated_len,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: big_off,
                    bytes: &big_bytes,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: big2_off,
                    bytes: &big2_bytes,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: hi_off,
                    bytes: &hi_bytes,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: neg_off,
                    bytes: &neg_bytes,
                },
            ],
        )
        .expect("retain resident device memory");

    // LOW-limb compare (high limbs equal): big > base+4000 (gt=3) => i > 4000 => [4001, 5000).
    let needle = base + 4000;
    let gt = resident
        .expr_i128_compare_scalar_filter(big_off, needle, false, 3, ROW_COUNT)
        .expect("big > needle");
    let gt_expected: Vec<u32> = (4001..ROW_COUNT).map(|i| i as u32).collect();
    assert_eq!(gt, gt_expected, "big > base+4000 => [4001, 5000)");

    // scalar_on_left: needle < big (lt=1) is the same set.
    let flipped = resident
        .expr_i128_compare_scalar_filter(big_off, needle, true, 1, ROW_COUNT)
        .expect("needle < big");
    assert_eq!(flipped, gt_expected, "needle < big == big > needle");

    // HIGH-limb compare (low limbs equal=0): hi >= 2500<<64 (ge=4) => high limb i >= 2500.
    let hi_needle = 2500i128 << 64;
    let ge = resident
        .expr_i128_compare_scalar_filter(hi_off, hi_needle, false, 4, ROW_COUNT)
        .expect("hi >= 2500<<64");
    let ge_expected: Vec<u32> = (2500..ROW_COUNT).map(|i| i as u32).collect();
    assert_eq!(
        ge, ge_expected,
        "hi >= 2500<<64 => [2500, 5000) (high-limb compare)"
    );

    // col-vs-col: big < big2 (lt=1) => 2i < n-1 => [0, 2500).
    let lt_cols = resident
        .expr_i128_compare_columns_filter(big_off, big2_off, 1, ROW_COUNT)
        .expect("big < big2");
    let lt_cols_expected: Vec<u32> = (0..2500).map(|i| i as u32).collect();
    assert_eq!(lt_cols, lt_cols_expected, "big < big2 => [0, 2500)");

    // SIGNED: neg < 0 (lt=1, scalar 0) => all rows (a negative high limb is < 0).
    let neg_lt = resident
        .expr_i128_compare_scalar_filter(neg_off, 0, false, 1, ROW_COUNT)
        .expect("neg < 0");
    let all_expected: Vec<u32> = (0..ROW_COUNT).map(|i| i as u32).collect();
    assert_eq!(neg_lt, all_expected, "neg < 0 => all rows (signed compare)");

    // Projection (16-byte): big at the gt survivors => base + i (128-bit, beyond i64).
    let gt_indices: Vec<u64> = gt.iter().map(|&i| u64::from(i)).collect();
    let projected = resident
        .project_i128_rows_from_payload(big_off, &gt_indices)
        .expect("project big");
    let projected_expected: Vec<i128> = (4001..N).map(|i| base + i as i128).collect();
    assert_eq!(
        projected, projected_expected,
        "projected big == base + i (128-bit)"
    );

    // Projection of NEGATIVE i128 round-trips: neg at [0, 3) => -base - i.
    let neg_proj = resident
        .project_i128_rows_from_payload(neg_off, &[0, 1, 2])
        .expect("project neg");
    assert_eq!(
        neg_proj,
        vec![-base, -base - 1, -base - 2],
        "negative i128 projection round-trips"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_compare_tolerates_4byte_aligned_int8_and_i128_sections() {
    // REGRESSION (the numeric audit's P0): the resident int8 / i128 compare kernels read each
    // 64-bit limb as two 4-byte loads, so a column section that is only 4-byte aligned (which
    // happens when an int4 section of ODD length — n_int4*rows odd — precedes it) does NOT fault
    // with cudaErrorMisalignedAddress (a sticky error that poisons the CUDA context). Every prior
    // test used an even row count and missed it. Layout: header(8) + one int4 col (rows*4) + one
    // int8 col + one i128 col, ROWS odd, so the int8 and i128 sections land at 4-mod-8 offsets.
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    const ROWS: u64 = 5; // odd -> int4 section = 20 bytes
    let i4_off = std::mem::size_of::<u64>() as u64; // 8
    let i8_off = i4_off + ROWS * 4; // 28 == 4 (mod 8): 4-byte-aligned int8 section
    let i128_off = i8_off + ROWS * 8; // 68 == 4 (mod 8): 4-byte-aligned i128 section
    assert_eq!(
        i8_off % 8,
        4,
        "int8 section must be 4-byte aligned to exercise the fix"
    );
    assert_eq!(
        i128_off % 8,
        4,
        "i128 section must be 4-byte aligned to exercise the fix"
    );

    let mut header = Vec::new();
    header.extend_from_slice(&ROWS.to_le_bytes());
    let mut i4 = Vec::new();
    let mut i8 = Vec::new();
    let mut i128v = Vec::new();
    let i8_base: i64 = 4_000_000_000; // > i32::MAX
    let i128_base: i128 = 1i128 << 70; // high limb set
    for r in 0..ROWS as i64 {
        i4.extend_from_slice(&(r as i32).to_le_bytes());
        i8.extend_from_slice(&(i8_base + r).to_le_bytes());
        i128v.extend_from_slice(&(i128_base + r as i128).to_le_bytes());
    }
    let allocated_len = i128_off + i128v.len() as u64;
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            allocated_len,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: i4_off,
                    bytes: &i4,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: i8_off,
                    bytes: &i8,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: i128_off,
                    bytes: &i128v,
                },
            ],
        )
        .expect("retain resident device memory");

    // int8 at a 4-byte-aligned offset: big > base+1 (gt=3) => r > 1 => [2, 5). Must NOT fault.
    let i8_gt = resident
        .expr_i64_compare_scalar_filter(i8_off, i8_base + 1, false, 3, ROWS)
        .expect("int8 compare at a 4-byte-aligned offset must not fault (misalignment regression)");
    assert_eq!(i8_gt, vec![2u32, 3, 4], "int8 big > base+1 => [2, 5)");

    // i128 at a 4-byte-aligned offset: num > base+1 (gt=3) => r > 1 => [2, 5). Must NOT fault.
    let i128_gt = resident
        .expr_i128_compare_scalar_filter(i128_off, i128_base + 1, false, 3, ROWS)
        .expect("i128 compare at a 4-byte-aligned offset must not fault (misalignment regression)");
    assert_eq!(i128_gt, vec![2u32, 3, 4], "i128 num > base+1 => [2, 5)");
}

#[test]
fn expression_family_ptx_is_pure_ascii() {
    // The runtime JIT's ptxas rejects a non-ASCII byte ("Unexpected non-ASCII character") even
    // though the LOCAL ptxas tolerates it — so a stray em-dash / smart-quote in a comment fails
    // every GPU launch with INVALID_PTX (218). This non-GPU test keeps every decomposed leaf
    // pure ASCII; adding a leaf requires adding it here.
    const PTX_FILES: &[(&str, &[u8])] = &[
        (
            "expression_i32.ptx",
            include_bytes!("../expression_i32.ptx"),
        ),
        (
            "expression_i64.ptx",
            include_bytes!("../expression_i64.ptx"),
        ),
        (
            "expression_i128.ptx",
            include_bytes!("../expression_i128.ptx"),
        ),
        (
            "expression_varlen.ptx",
            include_bytes!("../expression_varlen.ptx"),
        ),
        (
            "resident_gather.ptx",
            include_bytes!("../resident_gather.ptx"),
        ),
        (
            "derived_column.ptx",
            include_bytes!("../derived_column.ptx"),
        ),
        (
            "staged_hash_join.ptx",
            include_bytes!("../staged_hash_join.ptx"),
        ),
        (
            "resident_aggregate.ptx",
            include_bytes!("../resident_aggregate.ptx"),
        ),
        ("device_fill.ptx", include_bytes!("../device_fill.ptx")),
        (
            "resident_group_compact.ptx",
            include_bytes!("../resident_group_compact.ptx"),
        ),
        ("resident_sort.ptx", include_bytes!("../resident_sort.ptx")),
        (
            "resident_argsort.ptx",
            include_bytes!("../resident_argsort.ptx"),
        ),
        (
            "resident_group.ptx",
            include_bytes!("../resident_group.ptx"),
        ),
        (
            "resident_group_extra.ptx",
            include_bytes!("../resident_group_extra.ptx"),
        ),
    ];
    for &(name, ptx) in PTX_FILES {
        if let Some(pos) = ptx.iter().position(|&byte| !byte.is_ascii()) {
            let line = ptx[..pos].iter().filter(|&&byte| byte == b'\n').count() + 1;
            panic!("{name} has a non-ASCII byte at offset {pos} (line {line})");
        }
    }
}

#[test]
fn compare_ordered_ptx_is_pure_ascii() {
    // Same runtime-JIT ASCII gate for the ordered compare-compaction module (`COMPARE_ORDERED_PTX`,
    // shared by the value-emit and index-emit launches): a stray non-ASCII byte in a comment fails
    // every GPU launch with INVALID_PTX (218) under the runtime ptxas. Keep it pure ASCII.
    if let Some(pos) = COMPARE_ORDERED_PTX
        .iter()
        .position(|&byte| !byte.is_ascii())
    {
        let line = COMPARE_ORDERED_PTX[..pos]
            .iter()
            .filter(|&&byte| byte == b'\n')
            .count()
            + 1;
        panic!("COMPARE_ORDERED_PTX has a non-ASCII byte at offset {pos} (line {line})");
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_buffer_i128_arith_adds_subtracts_with_numeric_overflow() {
    // numeric (i128) CHECKED add/sub via the buffer VM with ElemType::I128 (the type matrix, doc
    // 19). Values beyond i64 range; closed-form oracles + an i128 overflow boundary -> PG `numeric
    // field overflow`. Hand-built programs over the run_expr_predicate_filter (mask) terminal.
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    const N: u64 = 600;
    let base: i128 = 1i128 << 70; // 2*base ~ 2^71, well within i128
    let elem = std::mem::size_of::<i128>() as u64;
    let a_off = std::mem::size_of::<u64>() as u64;
    let b_off = a_off + N * elem;
    let c_off = b_off + N * elem;
    let big_off = c_off + N * elem;

    let mut header = Vec::new();
    header.extend_from_slice(&N.to_le_bytes());
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    let mut big = Vec::new();
    for i in 0..N as i128 {
        a.extend_from_slice(&(base + i).to_le_bytes());
        b.extend_from_slice(&base.to_le_bytes());
        c.extend_from_slice(&(2 * base + 300).to_le_bytes());
        big.extend_from_slice(&i128::MAX.to_le_bytes());
    }
    let allocated_len = big_off + big.len() as u64;
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            allocated_len,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: a_off,
                    bytes: &a,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: b_off,
                    bytes: &b,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: c_off,
                    bytes: &c,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: big_off,
                    bytes: &big,
                },
            ],
        )
        .expect("retain resident device memory");

    // (a + b) > c : (2*base+i) > (2*base+300) => i > 300 => [301, 600). col-vs-col compare.
    let add_prog = [
        ExprStep::LoadColumn { byte_offset: a_off },
        ExprStep::LoadColumn { byte_offset: b_off },
        ExprStep::BufferBinary { op: 0 },
        ExprStep::LoadColumn { byte_offset: c_off },
        ExprStep::CompareBuffers { cmp: 3 },
    ];
    let added = resident
        .run_expr_predicate_filter(&add_prog, N, ResidentElemType::I128)
        .expect("(a+b)>c on GPU");
    let add_expected: Vec<u32> = (301..N as u32).collect();
    assert_eq!(added, add_expected, "(a+b) > c => [301, 600)");

    // (a - b) > 200 : i > 200 => [201, 600). scalar compare (200 fits the i32 ExprStep literal).
    let sub_prog = [
        ExprStep::LoadColumn { byte_offset: a_off },
        ExprStep::LoadColumn { byte_offset: b_off },
        ExprStep::BufferBinary { op: 1 },
        ExprStep::CompareScalar {
            cmp: 3,
            scalar: 200,
            scalar_on_left: false,
        },
    ];
    let subbed = resident
        .run_expr_predicate_filter(&sub_prog, N, ResidentElemType::I128)
        .expect("(a-b)>200 on GPU");
    let sub_expected: Vec<u32> = (201..N as u32).collect();
    assert_eq!(subbed, sub_expected, "(a-b) > 200 => [201, 600)");

    // big + big with big = i128::MAX overflows i128 -> numeric field overflow (never wraps).
    let ovf_prog = [
        ExprStep::LoadColumn {
            byte_offset: big_off,
        },
        ExprStep::LoadColumn {
            byte_offset: big_off,
        },
        ExprStep::BufferBinary { op: 0 },
        ExprStep::CompareScalar {
            cmp: 3,
            scalar: 0,
            scalar_on_left: false,
        },
    ];
    let err = resident
        .run_expr_predicate_filter(&ovf_prog, N, ResidentElemType::I128)
        .expect_err("i128::MAX + i128::MAX must overflow");
    assert!(
        err.to_string().contains("numeric field overflow"),
        "i128 overflow must raise `numeric field overflow`, got: {err}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_buffer_i128_mul_scalar_checks_overflow() {
    // numeric (i128) CHECKED scalar multiply via the signed 128x128->256 mul kernel + the I128 VM
    // (the type matrix, doc 19). Values beyond i64 range, a NEGATIVE multiplier (exact product),
    // and an i128 overflow boundary -> PG `numeric field overflow`.
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    const N: u64 = 600;
    let base: i128 = 1i128 << 70;
    let elem = std::mem::size_of::<i128>() as u64;
    let a_off = std::mem::size_of::<u64>() as u64;
    let c_off = a_off + N * elem;
    let negc_off = c_off + N * elem;
    let big_off = negc_off + N * elem;

    let mut header = Vec::new();
    header.extend_from_slice(&N.to_le_bytes());
    let mut a = Vec::new();
    let mut c = Vec::new();
    let mut negc = Vec::new();
    let mut big = Vec::new();
    for i in 0..N as i128 {
        a.extend_from_slice(&(base + i).to_le_bytes());
        c.extend_from_slice(&(3 * base + i).to_le_bytes());
        negc.extend_from_slice(&(-3 * (base + i)).to_le_bytes());
        big.extend_from_slice(&i128::MAX.to_le_bytes());
    }
    let allocated_len = big_off + big.len() as u64;
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            allocated_len,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: a_off,
                    bytes: &a,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: c_off,
                    bytes: &c,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: negc_off,
                    bytes: &negc,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: big_off,
                    bytes: &big,
                },
            ],
        )
        .expect("retain resident device memory");

    // a * 3 > c : (3*base+3i) > (3*base+i) => i > 0 => [1, 600). Positive scalar, beyond i64.
    let mul_prog = [
        ExprStep::LoadColumn { byte_offset: a_off },
        ExprStep::ScalarBinary {
            op: 2,
            scalar: 3,
            scalar_on_left: false,
        },
        ExprStep::LoadColumn { byte_offset: c_off },
        ExprStep::CompareBuffers { cmp: 3 },
    ];
    let gt = resident
        .run_expr_predicate_filter(&mul_prog, N, ResidentElemType::I128)
        .expect("a*3>c on GPU");
    let gt_expected: Vec<u32> = (1..N as u32).collect();
    assert_eq!(gt, gt_expected, "a*3 > c => [1, 600)");

    // a * (-3) == negc : exact NEGATIVE-scalar product => all rows.
    let neg_prog = [
        ExprStep::LoadColumn { byte_offset: a_off },
        ExprStep::ScalarBinary {
            op: 2,
            scalar: -3,
            scalar_on_left: false,
        },
        ExprStep::LoadColumn {
            byte_offset: negc_off,
        },
        ExprStep::CompareBuffers { cmp: 0 },
    ];
    let eq = resident
        .run_expr_predicate_filter(&neg_prog, N, ResidentElemType::I128)
        .expect("a*(-3)==negc on GPU");
    let all_expected: Vec<u32> = (0..N as u32).collect();
    assert_eq!(
        eq, all_expected,
        "a*(-3) == negc => all rows (exact negative product)"
    );

    // big * 2 with big = i128::MAX overflows i128 -> numeric field overflow (never wraps).
    let ovf_prog = [
        ExprStep::LoadColumn {
            byte_offset: big_off,
        },
        ExprStep::ScalarBinary {
            op: 2,
            scalar: 2,
            scalar_on_left: false,
        },
        ExprStep::CompareScalar {
            cmp: 3,
            scalar: 0,
            scalar_on_left: false,
        },
    ];
    let err = resident
        .run_expr_predicate_filter(&ovf_prog, N, ResidentElemType::I128)
        .expect_err("i128::MAX * 2 must overflow");
    assert!(
        err.to_string().contains("numeric field overflow"),
        "i128 multiply overflow must raise `numeric field overflow`, got: {err}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_buffer_i128_mul_columns_checks_overflow() {
    // numeric (i128) CHECKED column*column multiply via gpu_db_buffer_i128_mul + the I128 VM
    // (BufferBinary op 2). Exact product (vs a precomputed column) beyond i64 range + an overflow.
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    const N: u64 = 600;
    let base: i128 = 1i128 << 70;
    let elem = std::mem::size_of::<i128>() as u64;
    let a_off = std::mem::size_of::<u64>() as u64;
    let b_off = a_off + N * elem;
    let prod_off = b_off + N * elem;
    let big_off = prod_off + N * elem;
    let two_off = big_off + N * elem;

    let mut header = Vec::new();
    header.extend_from_slice(&N.to_le_bytes());
    let (mut a, mut b, mut prod, mut big, mut two) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..N as i128 {
        a.extend_from_slice(&(base + i).to_le_bytes());
        b.extend_from_slice(&(i + 2).to_le_bytes());
        prod.extend_from_slice(&((base + i) * (i + 2)).to_le_bytes());
        big.extend_from_slice(&i128::MAX.to_le_bytes());
        two.extend_from_slice(&2i128.to_le_bytes());
    }
    let allocated_len = two_off + two.len() as u64;
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            allocated_len,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: a_off,
                    bytes: &a,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: b_off,
                    bytes: &b,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: prod_off,
                    bytes: &prod,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: big_off,
                    bytes: &big,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: two_off,
                    bytes: &two,
                },
            ],
        )
        .expect("retain resident device memory");

    // a * b == prod : exact column*column product (beyond i64) => all rows.
    let eq_prog = [
        ExprStep::LoadColumn { byte_offset: a_off },
        ExprStep::LoadColumn { byte_offset: b_off },
        ExprStep::BufferBinary { op: 2 },
        ExprStep::LoadColumn {
            byte_offset: prod_off,
        },
        ExprStep::CompareBuffers { cmp: 0 },
    ];
    let eq = resident
        .run_expr_predicate_filter(&eq_prog, N, ResidentElemType::I128)
        .expect("a*b==prod on GPU");
    let all_expected: Vec<u32> = (0..N as u32).collect();
    assert_eq!(eq, all_expected, "a*b == prod => all rows (exact col*col)");

    // big * two with big = i128::MAX overflows i128 -> numeric field overflow.
    let ovf_prog = [
        ExprStep::LoadColumn {
            byte_offset: big_off,
        },
        ExprStep::LoadColumn {
            byte_offset: two_off,
        },
        ExprStep::BufferBinary { op: 2 },
        ExprStep::CompareScalar {
            cmp: 3,
            scalar: 0,
            scalar_on_left: false,
        },
    ];
    let err = resident
        .run_expr_predicate_filter(&ovf_prog, N, ResidentElemType::I128)
        .expect_err("i128::MAX * 2 (col*col) must overflow");
    assert!(
        err.to_string().contains("numeric field overflow"),
        "col*col multiply overflow must raise `numeric field overflow`, got: {err}"
    );
}
