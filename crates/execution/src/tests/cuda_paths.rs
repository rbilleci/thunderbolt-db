#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_i32_equal_any_project_submit_complete_matches_sync() {
    let runtime = CudaDriverRuntime::probe().unwrap();
    let row_count = 5_u64;
    let filter_offset = std::mem::size_of::<u64>() as u64;
    let projection_offset = filter_offset + row_count * std::mem::size_of::<i32>() as u64;
    let mut header = Vec::new();
    header.extend_from_slice(&row_count.to_le_bytes());
    let mut filter = Vec::new();
    for value in [1_i32, 2, 3, 2, 4] {
        filter.extend_from_slice(&value.to_le_bytes());
    }
    let mut projection = Vec::new();
    for value in [10_i32, 20, 30, 21, 40] {
        projection.extend_from_slice(&value.to_le_bytes());
    }
    let allocated_len = projection_offset + projection.len() as u64;
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
                    byte_offset: filter_offset,
                    bytes: &filter,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: projection_offset,
                    bytes: &projection,
                },
            ],
        )
        .unwrap();

    let sync_rows = resident
        .match_project_i32_equal_any_from_payload(
            filter_offset,
            &[2, 4],
            &[projection_offset],
            row_count,
        )
        .unwrap();
    let async_rows = resident
        .submit_match_project_i32_equal_any_from_payload(
            filter_offset,
            &[2, 4],
            &[projection_offset],
            row_count,
        )
        .unwrap()
        .complete(&resident)
        .unwrap();
    let (read_view_rows, read_view_elapsed_us) = resident
        .read_view()
        .submit_match_project_i32_equal_any_from_payload(
            filter_offset,
            &[2, 4],
            &[projection_offset],
            row_count,
        )
        .unwrap()
        .complete_detached()
        .unwrap();

    assert_eq!(async_rows, sync_rows);
    assert_eq!(read_view_rows, sync_rows);
    assert!(read_view_elapsed_us.is_some());
    assert_eq!(
        async_rows,
        vec![
            CudaI32BatchProjectionRow {
                needle_index: 0,
                row_index: 1,
                values: vec![20],
            },
            CudaI32BatchProjectionRow {
                needle_index: 0,
                row_index: 3,
                values: vec![21],
            },
            CudaI32BatchProjectionRow {
                needle_index: 1,
                row_index: 4,
                values: vec![40],
            },
        ]
    );

    // The public read view and split submission are independently owned safe handles. Prove both layers:
    // the view remains usable after the allocation facade drops, then the returned submission pins the exact
    // source allocation after the view drops. The Weak witness makes this independent of `cuMemFree`'s
    // synchronizing behavior (which could otherwise let a broken submit-before-drop test pass vacuously).
    let read_view = resident.read_view();
    let allocation = resident.allocation_weak_for_test();
    drop(resident);
    assert!(
        allocation.is_alive(),
        "the independently owned read view retains the source allocation"
    );
    let detached = read_view
        .submit_match_project_i32_equal_any_from_payload(
            filter_offset,
            &[2, 4],
            &[projection_offset],
            row_count,
        )
        .expect("submit through surviving independently owned read view");
    drop(read_view);
    assert!(
        allocation.is_alive(),
        "the deferred submission retains the source allocation after the view drops"
    );
    assert_eq!(
        detached
            .complete_detached()
            .expect("submission retains source allocation after owner/view drop")
            .0,
        sync_rows
    );
    assert!(
        !allocation.is_alive(),
        "completion releases the submission's final allocation guard"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_i32_equal_any_project_drop_without_complete_drains_before_pool_reuse() {
    // P2-M2 BLOCKER regression: `submit` enqueues HtoD(needles) + memset(count) + the kernel on
    // the held pooled private stream WITHOUT syncing (the covering sync lives only in
    // `complete`). If the submission is DROPPED before `complete` runs (the engine's early `Err`
    // return, or any `?`/cancel/panic in the submit→complete window), its `Drop` MUST drain the
    // stream before its field guards return the pooled device buffers + stream to the SHARED
    // pools — otherwise a concurrent `lease_device_buffer` on another thread re-leases memory the
    // in-flight kernel is still writing (cross-thread use-after-free → silent corruption).
    //
    // This test repeatedly submits-then-drops-without-complete on N threads while OTHER threads
    // hammer the full submit→complete path on the SAME resident allocation (same pools). The
    // dropped submissions return their buffers/stream to the pools mid-flight; the concurrent
    // completers immediately re-lease them. Without the `Drop` drain, a completer would read a
    // buffer still under a dropped submission's kernel and observe wrong rows (or the run would
    // crash). The exact-match assertion on every completion makes a missing drain surface
    // deterministically. With the drain, every completion is exactly the known result.
    use std::sync::Arc;

    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    // Known payload: header(row_count) + i32 filter column + i32 projection column.
    let row_count = 5_u64;
    let filter_offset = std::mem::size_of::<u64>() as u64;
    let projection_offset = filter_offset + row_count * std::mem::size_of::<i32>() as u64;
    let mut header = Vec::new();
    header.extend_from_slice(&row_count.to_le_bytes());
    let mut filter = Vec::new();
    for value in [1_i32, 2, 3, 2, 4] {
        filter.extend_from_slice(&value.to_le_bytes());
    }
    let mut projection = Vec::new();
    for value in [10_i32, 20, 30, 21, 40] {
        projection.extend_from_slice(&value.to_le_bytes());
    }
    let allocated_len = projection_offset + projection.len() as u64;

    // Shared resident allocation (one device buffer, one shared primary context with its shared
    // buffer/stream pools) so the droppers and completers contend on the very same pools.
    let resident = Arc::new(
        runtime
            .retain_device_memory_chunks(
                0,
                allocated_len,
                &[
                    CudaDeviceMemoryChunk {
                        byte_offset: 0,
                        bytes: &header,
                    },
                    CudaDeviceMemoryChunk {
                        byte_offset: filter_offset,
                        bytes: &filter,
                    },
                    CudaDeviceMemoryChunk {
                        byte_offset: projection_offset,
                        bytes: &projection,
                    },
                ],
            )
            .expect("retain resident device memory"),
    );

    // needles [2,4] over filter [1,2,3,2,4] → rows 1,3 (=2) and 4 (=4), projecting [20],[21],[40].
    let expected = vec![
        CudaI32BatchProjectionRow {
            needle_index: 0,
            row_index: 1,
            values: vec![20],
        },
        CudaI32BatchProjectionRow {
            needle_index: 0,
            row_index: 3,
            values: vec![21],
        },
        CudaI32BatchProjectionRow {
            needle_index: 1,
            row_index: 4,
            values: vec![40],
        },
    ];

    const ITERS: usize = 200;
    const DROPPER_THREADS: usize = 3;
    const COMPLETER_THREADS: usize = 3;

    let mut handles = Vec::new();

    // Droppers: submit then DROP without `complete`, repeatedly. Each drop runs the `Drop` impl's
    // safety drain, then returns the buffers/stream to the shared pools.
    for _ in 0..DROPPER_THREADS {
        let resident = Arc::clone(&resident);
        handles.push(std::thread::spawn(move || {
            for _ in 0..ITERS {
                let submission = resident
                    .submit_match_project_i32_equal_any_from_payload(
                        filter_offset,
                        &[2, 4],
                        &[projection_offset],
                        row_count,
                    )
                    .expect("submit (to be dropped without complete)");
                // Explicit drop = the BLOCKER's drop-without-complete path. If `Drop` did not
                // drain, the just-returned pooled buffers are re-leasable while this kernel is
                // still in flight.
                drop(submission);
            }
        }));
    }

    // Completers: full submit→complete on the SAME pools, repeatedly. Re-leases buffers the
    // droppers just returned; a missing drain would corrupt these reads.
    for _ in 0..COMPLETER_THREADS {
        let resident = Arc::clone(&resident);
        let expected = expected.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..ITERS {
                let rows = resident
                    .submit_match_project_i32_equal_any_from_payload(
                        filter_offset,
                        &[2, 4],
                        &[projection_offset],
                        row_count,
                    )
                    .expect("submit on completer")
                    .complete(&resident)
                    .expect("complete on completer");
                assert_eq!(
                    rows, expected,
                    "completer observed wrong rows — a dropped submission's buffer/stream was \
                         re-leased while its kernel was still in flight (Drop drain missing?)"
                );
            }
        }));
    }

    for handle in handles {
        handle
            .join()
            .expect("worker thread panicked (crash under pool reuse?)");
    }

    // After the contention storm, a final completion on the (heavily reused) pools must still be
    // exactly correct — the pools are left in a sound state.
    let final_rows = resident
        .submit_match_project_i32_equal_any_from_payload(
            filter_offset,
            &[2, 4],
            &[projection_offset],
            row_count,
        )
        .expect("final submit")
        .complete(&resident)
        .expect("final complete");
    assert_eq!(
        final_rows, expected,
        "pools left unsound after drop-without-complete reuse"
    );

    let mut caller_needles = vec![2, 4];
    let staged = resident
        .submit_match_project_i32_equal_any_from_payload(
            filter_offset,
            &caller_needles,
            &[projection_offset],
            row_count,
        )
        .expect("owned-source submit");
    assert_ne!(
        staged.staged_needles_ptr_for_test(),
        caller_needles.as_ptr().cast(),
        "safe deferred submission must DMA from an owned copy, not the caller slice"
    );
    caller_needles.fill(-1);
    assert_eq!(
        staged.complete(&resident).expect("owned-source complete"),
        expected,
        "caller mutation after submit cannot alter the staged H2D source"
    );

    // Panic after the first result D2H has been queued. The local host-copy drain must synchronize before
    // any pinned lease/Vec unwinds; submission Drop then returns all shared resources in a reusable state.
    let completion = resident
        .submit_match_project_i32_equal_any_from_payload(
            filter_offset,
            &[2, 4],
            &[projection_offset],
            row_count,
        )
        .expect("submit before atomic completion panic");
    crate::point_read_submission::force_next_atomic_completion_panic(1);
    let completion_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = completion.complete_detached_columnar();
    }));
    assert!(
        completion_panic.is_err(),
        "the atomic completion panic hook fired"
    );
    assert_eq!(
        resident
            .submit_match_project_i32_equal_any_from_payload(
                filter_offset,
                &[2, 4],
                &[projection_offset],
                row_count,
            )
            .expect("submit after atomic completion panic")
            .complete(&resident)
            .expect("shared pools remain reusable after atomic completion panic"),
        expected
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_dense_index_probe_drop_without_complete_drains_before_pool_reuse() {
    // The dense submission owns the same asynchronous HtoD/kernel work and pooled buffers/stream as its
    // atomic sibling. Dropping it before completion must synchronize before any guard returns to a shared
    // pool. Droppers and completers intentionally contend on one primary context so an omitted drain turns
    // into corrupt status/value slots or a CUDA safety failure.
    use std::sync::Arc;

    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let row_count = 4_096_u64;
    let keys: Vec<i32> = (1..=row_count as i32).collect();
    let payload: Vec<i32> = keys.iter().map(|key| key * 10).collect();
    let mut bytes = Vec::with_capacity(row_count as usize * 8);
    for value in keys.iter().chain(&payload) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let resident = Arc::new(
        runtime
            .retain_device_memory_copy(0, &bytes)
            .expect("resident device memory"),
    );
    let (index, table_mask, hash_shift) = build_pk_hash(&keys);
    let index_bytes: Vec<u8> = index.iter().flat_map(|entry| entry.to_le_bytes()).collect();
    let index = Arc::new(
        runtime
            .retain_device_memory_copy(0, &index_bytes)
            .expect("index device memory"),
    );
    let projection_offsets = [0, row_count * std::mem::size_of::<i32>() as u64];
    let needles = [1, 2_048, 4_096, -1];

    let bounded_resident = runtime
        .retain_device_memory_copy(0, &7_i32.to_le_bytes())
        .expect("one-row resident memory");
    let bounded_mask = 1_u32;
    let bounded_shift = 31_u32;
    let mut corrupt_index = [0_u64; 2];
    let corrupt_slot = (7_u32.wrapping_mul(0x9E37_79B1) >> bounded_shift) & bounded_mask;
    corrupt_index[corrupt_slot as usize] = (7_u64 << 32) | 2;
    let corrupt_bytes: Vec<u8> = corrupt_index
        .iter()
        .flat_map(|entry| entry.to_le_bytes())
        .collect();
    let corrupt_index = Arc::new(
        runtime
            .retain_device_memory_copy(0, &corrupt_bytes)
            .expect("corrupt-row index memory"),
    );
    let corrupt_dense = bounded_resident
        .submit_match_project_i32_index_probe_dense_from_payload(
            &corrupt_index,
            bounded_mask,
            bounded_shift,
            &[7],
            &[0],
            1,
        )
        .expect("structurally valid corrupt-row dense submit")
        .complete_detached_columnar()
        .expect("corrupt row fails closed without a device fault")
        .0;
    assert_eq!(corrupt_dense.status, [2]);
    let corrupt_atomic = bounded_resident
        .submit_match_project_i32_index_probe_from_payload(
            &corrupt_index,
            bounded_mask,
            bounded_shift,
            &[7],
            &[0],
            1,
        )
        .expect("structurally valid corrupt-row atomic submit")
        .complete(&bounded_resident)
        .expect("atomic corrupt row fails closed without a device fault");
    assert!(corrupt_atomic.is_empty());

    let short_index = Arc::new(
        runtime
            .retain_device_memory_zeroed(0, std::mem::size_of::<u64>() as u64)
            .expect("short index allocation"),
    );
    assert!(
        resident
            .submit_match_project_i32_index_probe_dense_from_payload(
                &short_index,
                table_mask,
                hash_shift,
                &needles,
                &projection_offsets,
                row_count,
            )
            .is_err(),
        "safe dense API rejects an index allocation shorter than mask geometry"
    );
    assert!(
        resident
            .submit_match_project_i32_index_probe_dense_from_payload(
                &index,
                table_mask,
                hash_shift.wrapping_add(1),
                &needles,
                &projection_offsets,
                row_count,
            )
            .is_err(),
        "safe dense API rejects incoherent mask/hash-shift geometry"
    );
    assert!(
        resident
            .submit_match_project_i32_index_probe_dense_from_payload(
                &index,
                table_mask,
                hash_shift,
                &needles,
                &[1],
                row_count,
            )
            .is_err(),
        "safe dense API rejects a misaligned projection before launch"
    );

    let assert_complete = |columns: CudaI32BatchProjectionColumns| {
        assert_eq!(columns.status, [1, 1, 1, 2]);
        assert_eq!(&columns.values[0..2], &[1, 10]);
        assert_eq!(&columns.values[2..4], &[2_048, 20_480]);
        assert_eq!(&columns.values[4..6], &[4_096, 40_960]);
    };

    const ITERS: usize = 100;
    let mut handles = Vec::new();
    for _ in 0..2 {
        let resident = Arc::clone(&resident);
        let index = Arc::clone(&index);
        handles.push(std::thread::spawn(move || {
            for _ in 0..ITERS {
                let submission = resident
                    .submit_match_project_i32_index_probe_dense_from_payload(
                        &index,
                        table_mask,
                        hash_shift,
                        &needles,
                        &projection_offsets,
                        row_count,
                    )
                    .expect("dense submit to drop");
                drop(submission);
            }
        }));
    }
    for _ in 0..2 {
        let resident = Arc::clone(&resident);
        let index = Arc::clone(&index);
        handles.push(std::thread::spawn(move || {
            for _ in 0..ITERS {
                let (columns, _) = resident
                    .submit_match_project_i32_index_probe_dense_from_payload(
                        &index,
                        table_mask,
                        hash_shift,
                        &needles,
                        &projection_offsets,
                        row_count,
                    )
                    .expect("dense submit to complete")
                    .complete_detached_columnar()
                    .expect("dense complete");
                assert_complete(columns);
            }
        }));
    }
    for handle in handles {
        handle.join().expect("dense pool-reuse worker panicked");
    }

    let (columns, _) = resident
        .submit_match_project_i32_index_probe_dense_from_payload(
            &index,
            table_mask,
            hash_shift,
            &needles,
            &projection_offsets,
            row_count,
        )
        .expect("final dense submit")
        .complete_detached_columnar()
        .expect("final dense complete");
    assert_complete(columns);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_i32_compare_project_matches_expected_under_concurrent_pool_reuse() {
    // P2-M2 regression for the `compare_project` migration to the pooled-async substrate.
    // `compare_project` is a SINGLE-FRAME route (no submit→complete split), so the
    // drop-without-complete hazard does not apply — the device/pinned buffers + private
    // stream are released only after the route's own covering syncs. What the migration DID
    // introduce is concurrent contention on the SHARED device-buffer pool, pinned-host pool,
    // private-stream pool, and module cache: many readers leasing/returning the same buckets
    // while each route is mid-flight. This test (a) pins down byte-exact parity against a
    // known result for every comparison op, and (b) hammers the route on N threads over ONE
    // shared resident allocation (hence the same shared pools) asserting the exact result on
    // every call — a wrong readback (e.g. a buffer re-leased before its async D2H drained, or
    // values read past [0, count)) would surface deterministically as a mismatch or a crash.
    use std::sync::{Arc, Barrier};

    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    // Known payload: header(row_count) + one i32 column [10, 20, 30, 20, 40].
    let row_count = 5_u64;
    let byte_offset = std::mem::size_of::<u64>() as u64;
    let column_values = [10_i32, 20, 30, 20, 40];
    let mut header = Vec::new();
    header.extend_from_slice(&row_count.to_le_bytes());
    let mut column = Vec::new();
    for value in column_values {
        column.extend_from_slice(&value.to_le_bytes());
    }
    let allocated_len = byte_offset + column.len() as u64;
    let resident = Arc::new(
        runtime
            .retain_device_memory_chunks(
                0,
                allocated_len,
                &[
                    CudaDeviceMemoryChunk {
                        byte_offset: 0,
                        bytes: &header,
                    },
                    CudaDeviceMemoryChunk {
                        byte_offset,
                        bytes: &column,
                    },
                ],
            )
            .expect("retain resident device memory"),
    );

    // The kernel appends matches in ascending row order, so the expected vectors are the
    // column values filtered in order. needle = 20:
    //   Lt  → [10]; Lte → [10, 20, 20]; Gt → [30, 40]; Gte → [20, 30, 20, 40].
    let needle = 20_i32;
    let cases: &[(CudaI32Comparison, Vec<i32>)] = &[
        (CudaI32Comparison::Lt, vec![10]),
        (CudaI32Comparison::Lte, vec![10, 20, 20]),
        (CudaI32Comparison::Gt, vec![30, 40]),
        (CudaI32Comparison::Gte, vec![20, 30, 20, 40]),
    ];

    // (a) Single-threaded parity: each comparison returns exactly the expected values.
    for (comparison, expected) in cases {
        let got = resident
            .project_i32_compare_from_payload(byte_offset, row_count, needle, *comparison)
            .expect("project_i32_compare");
        assert_eq!(
            &got, expected,
            "compare_project parity failed for {comparison:?}"
        );
    }

    // Empty input short-circuits to an empty vector (no device work).
    assert!(resident
        .project_i32_compare_from_payload(byte_offset, 0, needle, CudaI32Comparison::Gte)
        .expect("empty project_i32_compare")
        .is_empty());

    // (b) Concurrent pool-reuse storm: every thread runs all four comparisons in a loop on the
    // shared allocation (shared pools), asserting the exact expected vector each time.
    const THREADS: usize = 8;
    const ITERS: usize = 300;
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();
    for _ in 0..THREADS {
        let resident = Arc::clone(&resident);
        let barrier = Arc::clone(&barrier);
        let cases = cases.to_vec();
        handles.push(std::thread::spawn(move || {
            resident
                .set_current_context()
                .expect("bind primary context on reader thread");
            barrier.wait();
            for _ in 0..ITERS {
                for (comparison, expected) in &cases {
                    let got = resident
                        .project_i32_compare_from_payload(
                            byte_offset,
                            row_count,
                            needle,
                            *comparison,
                        )
                        .expect("concurrent project_i32_compare");
                    assert_eq!(
                        &got, expected,
                        "concurrent compare_project returned wrong values for {comparison:?} \
                             — a pooled buffer/stream was reused before its async D2H drained?"
                    );
                }
            }
        }));
    }
    for handle in handles {
        handle
            .join()
            .expect("reader thread panicked (crash under pool reuse?)");
    }

    // After the storm the pools are left sound: a final call is still exactly correct.
    let final_rows = resident
        .project_i32_compare_from_payload(byte_offset, row_count, needle, CudaI32Comparison::Gte)
        .expect("final project_i32_compare");
    assert_eq!(final_rows, vec![20, 30, 20, 40]);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_i32_compare_project_multi_block_returns_ascending_row_order() {
    // Coverage gap closer for the PARALLEL ordered-compaction kernel (analogue of the
    // `row_indices` multi-warp ascending test). `compare_project` must return the matching i32
    // VALUES in ASCENDING ROW ORDER, byte-identical to the legacy single-thread scan, for ANY
    // row_count. The new kernel parallelizes this as an ordered compaction over a CONTIGUOUS
    // block partition (chunk = 256 rows/block): a parallel per-block count, a host exclusive
    // scan into per-block base offsets, then a per-block ascending scatter at base+local. This
    // test drives a payload that is FAR past one block AND one warp (~700 matches across ~20
    // blocks) and asserts the EXACT expected vector — proving the compaction preserves row order
    // ACROSS BLOCKS, not just within one.
    //
    // Why it is NON-VACUOUS (would fail under an unordered impl): the matching VALUES are a
    // by-row SCRAMBLED sequence (a hash of the row index), so the correct ascending-row-order
    // output is deliberately NOT sorted-by-value and NOT contiguous. An atomic-append
    // implementation would emit the matches in atomic-SCHEDULE order — non-deterministic across
    // the ~20 racing blocks — which is essentially never this exact by-row sequence; a "sort the
    // values" shortcut would emit them sorted, which this sequence is not. Only a compaction that
    // is stable by row across blocks reproduces the asserted vector. We also assert the expected
    // sequence is not already sorted (so the sort shortcut provably diverges) and run the route
    // repeatedly (a schedule-ordered impl would flake across iterations).
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    // 5000 rows => grid = ceil(5000/256) = 20 blocks (multi-block AND multi-warp). Every 7th row
    // matches (Gt 0): its value is a positive by-row hash (scrambled); other rows hold -1 (a Gt 0
    // non-match). 5000/7 ≈ 715 matches spread across all 20 blocks.
    const ROW_COUNT: u64 = 5000;
    const NEEDLE: i32 = 0;
    let col_offset = std::mem::size_of::<u64>() as u64;
    // Positive by-row hash in [1, 1_000_000], deterministic and scrambled relative to row order.
    let row_value = |row: u64| -> i32 {
        let h = row.wrapping_mul(2_654_435_761) ^ (row << 13) ^ 0x9E37_79B9;
        (1 + (h % 1_000_000)) as i32
    };
    let column: Vec<i32> = (0..ROW_COUNT)
        .map(|row| if row % 7 == 0 { row_value(row) } else { -1 })
        .collect();
    // Reference = matches in ASCENDING ROW ORDER (exactly what the serial kernel emits).
    let expected: Vec<i32> = (0..ROW_COUNT)
        .filter(|row| row % 7 == 0)
        .map(row_value)
        .collect();

    assert!(
        expected.len() > 32,
        "test must use a multi-warp/multi-block match count to exercise cross-block order"
    );
    // The expected by-row order must NOT already be sorted, or a "sort the values" impl would
    // pass vacuously. (The hash scrambles values relative to row order, so this holds.)
    let mut sorted = expected.clone();
    sorted.sort_unstable();
    assert_ne!(
        expected, sorted,
        "expected by-row sequence is accidentally sorted — pick a payload that isn't, else the \
             test cannot distinguish ordered compaction from a value sort"
    );

    let mut header = Vec::new();
    header.extend_from_slice(&ROW_COUNT.to_le_bytes());
    let mut column_bytes = Vec::new();
    for value in &column {
        column_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let allocated_len = col_offset + column_bytes.len() as u64;
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
                    byte_offset: col_offset,
                    bytes: &column_bytes,
                },
            ],
        )
        .expect("retain resident device memory");

    // Run repeatedly: an atomic-schedule-ordered impl would surface a non-ascending-by-row
    // permutation on at least one iteration (the cross-block schedule varies run to run); the
    // ordered compaction is exactly the by-row sequence every time.
    for iter in 0..50 {
        let got = resident
            .project_i32_compare_from_payload(col_offset, ROW_COUNT, NEEDLE, CudaI32Comparison::Gt)
            .expect("multi-block project_i32_compare");
        assert_eq!(
            got, expected,
            "multi-block compare_project values were not in ascending ROW order on iteration \
                 {iter} — the parallel compaction is not stable across blocks (atomic-append \
                 schedule order?) or the per-block base offsets are wrong"
        );
    }

    // Boundary coverage in the same payload shape: 0 matches, all matches, and a count that
    // straddles a block boundary, each byte-identical to the by-row reference.
    // (a) 0 matches: Gt a value larger than every row value.
    let none = resident
        .project_i32_compare_from_payload(col_offset, ROW_COUNT, 2_000_000, CudaI32Comparison::Gt)
        .expect("zero-match project_i32_compare");
    assert!(none.is_empty(), "Gt 2_000_000 must match nothing");
    // (b) all matches: Gte i32::MIN matches every row, ascending by row == the raw column.
    let all = resident
        .project_i32_compare_from_payload(col_offset, ROW_COUNT, i32::MIN, CudaI32Comparison::Gte)
        .expect("all-match project_i32_compare");
    assert_eq!(
        all, column,
        "Gte i32::MIN must return the whole column in row order"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_expr_two_col_filter_evaluates_arithmetic_predicate_on_gpu() {
    // PROTOTYPE for the general GPU executor (docs/architecture/17-general-gpu-executor.md §5):
    // prove a predicate the enumerated shape-path CANNOT express — an ARITHMETIC expression
    // `(a <op> b) <cmp> k` — runs fully on the GPU through typed VM column loads and buffer
    // arithmetic followed by ordered compare-compaction. This is the vectorized-interpreter
    // model that replaces hand-coded per-shape kernels.
    //
    // GPU-NATIVE oracle = CLOSED FORM, never a CPU re-implementation of the operator (project
    // rule). With a[i]=i and b[i]=i the intermediate is a+b = 2*i, MONOTONE in the row index, so
    // {i : 2*i > k} is exactly the contiguous range [k/2 + 1, n). That closed form is distinct
    // from "only column a" ({i : i > k}) and from "a*b" ({i : i*i > k}), so matching it proves the
    // kernel actually evaluated the Add-then-Gt tree over the intermediate, not a shortcut.
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    const ROW_COUNT: u64 = 5000;
    let a_off = std::mem::size_of::<u64>() as u64;
    let b_off = a_off + ROW_COUNT * std::mem::size_of::<i32>() as u64;

    let mut header = Vec::new();
    header.extend_from_slice(&ROW_COUNT.to_le_bytes());
    let mut a_bytes = Vec::new();
    let mut b_bytes = Vec::new();
    for row in 0..ROW_COUNT as i32 {
        a_bytes.extend_from_slice(&row.to_le_bytes());
        b_bytes.extend_from_slice(&row.to_le_bytes());
    }
    let allocated_len = b_off + b_bytes.len() as u64;
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
                    bytes: &a_bytes,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: b_off,
                    bytes: &b_bytes,
                },
            ],
        )
        .expect("retain resident device memory");

    let assert_invalid = |result: Result<Vec<u32>, CudaRuntimeProbeError>| {
        assert!(
            matches!(
                result,
                Err(CudaRuntimeProbeError::InvalidInputLength(_))
                    | Err(CudaRuntimeProbeError::UnsupportedComparison(_))
            ),
            "invalid two-column expression input must fail closed, got {result:?}"
        );
    };
    // Codes and the u32 output-index domain are checked even when no kernel work is needed.
    assert_invalid(resident.expr_filter_two_col_compare_from_payload(a_off, b_off, 3, 0, 0, 0));
    assert_invalid(resident.expr_filter_two_col_compare_from_payload(a_off, b_off, 0, 0, 0, 5));
    assert_invalid(resident.expr_filter_two_col_compare_from_payload(
        a_off,
        b_off,
        0,
        u64::from(u32::MAX) + 1,
        0,
        0,
    ));
    // Both raw resident offsets must be naturally aligned and their complete read windows must
    // fit the allocation before the VM allocates or launches anything.
    assert_invalid(resident.expr_filter_two_col_compare_from_payload(
        a_off + 1,
        b_off,
        0,
        ROW_COUNT,
        0,
        0,
    ));
    assert_invalid(resident.expr_filter_two_col_compare_from_payload(
        a_off,
        b_off + 1,
        0,
        ROW_COUNT,
        0,
        0,
    ));
    assert_invalid(resident.expr_filter_two_col_compare_from_payload(
        a_off,
        b_off + std::mem::size_of::<i32>() as u64,
        0,
        ROW_COUNT,
        0,
        0,
    ));

    // op_code 0=add, comparison 3=gt. a+b = 2*i > K  <=>  i >= K/2 + 1  (K even).
    const K: i32 = 4000;
    let got = resident
        .expr_filter_two_col_compare_from_payload(a_off, b_off, 0, ROW_COUNT, K, 3)
        .expect("expr_filter add+gt");
    let add_start = (K as u64 / 2) + 1; // 2001
    let expected_add: Vec<u32> = (add_start..ROW_COUNT).map(|i| i as u32).collect();
    assert_eq!(
        got, expected_add,
        "GPU a+b>{K} must be the closed-form monotone range [{add_start}, {ROW_COUNT}) — proves \
             the interpreter evaluated Add then Gt over the intermediate buffer"
    );
    // Guard vacuity: 'only column a' (a>K) would be [K+1, n) = 999 matches, not 2999.
    let only_a_count = (ROW_COUNT - (K as u64 + 1)) as usize;
    assert_ne!(
        got.len(),
        only_a_count,
        "result must differ from a>K — else the kernel ignored column b (read the intermediate?)"
    );

    // op_code 2=mul: a*b = i*i > M  <=>  i >= isqrt(M)+1. M = 3969 = 63*63 => i >= 64.
    const M: i32 = 3969;
    const MUL_START: u64 = 64;
    let got_mul = resident
        .expr_filter_two_col_compare_from_payload(a_off, b_off, 2, ROW_COUNT, M, 3)
        .expect("expr_filter mul+gt");
    let expected_mul: Vec<u32> = (MUL_START..ROW_COUNT).map(|i| i as u32).collect();
    assert_eq!(
        got_mul, expected_mul,
        "GPU a*b>{M} must be [{MUL_START}, {ROW_COUNT}) — proves op_code routing (mul != add)"
    );

    // Boundaries: 0 matches (a+b can never exceed i32::MAX here) and all matches.
    let none = resident
        .expr_filter_two_col_compare_from_payload(a_off, b_off, 0, ROW_COUNT, i32::MAX, 3)
        .expect("expr_filter zero-match");
    assert!(none.is_empty(), "a+b > i32::MAX matches nothing");
    let all = resident
        .expr_filter_two_col_compare_from_payload(a_off, b_off, 0, ROW_COUNT, i32::MIN, 3)
        .expect("expr_filter all-match");
    let expected_all: Vec<u32> = (0..ROW_COUNT).map(|i| i as u32).collect();
    assert_eq!(
        all, expected_all,
        "a+b > i32::MIN matches every row, in ascending row order"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_expr_arith_vm_evaluates_deep_trees_on_gpu() {
    // Recursive interpreter (device bytecode VM, docs/architecture/17 section 2.3): an ARBITRARY
    // int4 arithmetic tree evaluated on the GPU by chaining buffer->buffer primitives over a stack
    // of intermediates (LoadColumn / BufferBinary / ScalarBinary), then compared to a literal. GPU
    // -NATIVE CLOSED-FORM oracle (a[i]=b[i]=i so every tree below is monotone in i => matches are a
    // contiguous range derived in closed form, never a CPU re-implementation of the operator).
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    const ROW_COUNT: u64 = 600;
    let a_off = std::mem::size_of::<u64>() as u64;
    let b_off = a_off + ROW_COUNT * std::mem::size_of::<i32>() as u64;
    let mut header = Vec::new();
    header.extend_from_slice(&ROW_COUNT.to_le_bytes());
    let mut a_bytes = Vec::new();
    let mut b_bytes = Vec::new();
    for row in 0..ROW_COUNT as i32 {
        a_bytes.extend_from_slice(&row.to_le_bytes());
        b_bytes.extend_from_slice(&row.to_le_bytes());
    }
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            b_off + b_bytes.len() as u64,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: a_off,
                    bytes: &a_bytes,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: b_off,
                    bytes: &b_bytes,
                },
            ],
        )
        .expect("retain resident device memory");

    // Tree 1: (a + b) * 2 - 5 = 4*i - 5. Predicate > 395 <=> 4i > 400 <=> i >= 101.
    // Exercises BufferBinary(add) + ScalarBinary(mul, right) + ScalarBinary(sub, right).
    let p1 = [
        ExprStep::LoadColumn { byte_offset: a_off },
        ExprStep::LoadColumn { byte_offset: b_off },
        ExprStep::BufferBinary { op: 0 },
        ExprStep::ScalarBinary {
            op: 2,
            scalar: 2,
            scalar_on_left: false,
        },
        ExprStep::ScalarBinary {
            op: 1,
            scalar: 5,
            scalar_on_left: false,
        },
    ];
    let got1 = resident
        .run_expr_arith_filter(&p1, ROW_COUNT, 3, 395)
        .expect("vm (a+b)*2-5 > 395");
    let expected1: Vec<u32> = (101..ROW_COUNT).map(|i| i as u32).collect();
    assert_eq!(got1, expected1, "(a+b)*2-5 > 395 <=> i >= 101");

    // Tree 2: (a + b) * a = 2*i*i. Predicate > 200 <=> i*i > 100 <=> i >= 11. Buffer x buffer mul.
    let p2 = [
        ExprStep::LoadColumn { byte_offset: a_off },
        ExprStep::LoadColumn { byte_offset: b_off },
        ExprStep::BufferBinary { op: 0 },
        ExprStep::LoadColumn { byte_offset: a_off },
        ExprStep::BufferBinary { op: 2 },
    ];
    let got2 = resident
        .run_expr_arith_filter(&p2, ROW_COUNT, 3, 200)
        .expect("vm (a+b)*a > 200");
    let expected2: Vec<u32> = (11..ROW_COUNT).map(|i| i as u32).collect();
    assert_eq!(got2, expected2, "(a+b)*a = 2i^2 > 200 <=> i >= 11");

    // Tree 3: 10 - a. Predicate > 0 <=> i < 10. Exercises scalar_on_left (non-commutative sub).
    let p3 = [
        ExprStep::LoadColumn { byte_offset: a_off },
        ExprStep::ScalarBinary {
            op: 1,
            scalar: 10,
            scalar_on_left: true,
        },
    ];
    let got3 = resident
        .run_expr_arith_filter(&p3, ROW_COUNT, 3, 0)
        .expect("vm 10-a > 0");
    let expected3: Vec<u32> = (0..10).collect();
    assert_eq!(got3, expected3, "10 - a > 0 <=> i < 10 (scalar_on_left)");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_text_eq_scalar_filters_rows() {
    // Text equality over a resident TEXT column (the type matrix, doc 19): offsets[n+1] (u64 LE) +
    // byte blob, where row i = blob[offsets[i]..offsets[i+1]]. The public resident contract keeps
    // the u64 offsets section 8-byte aligned; misaligned descriptors must fail before launch.
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    let rows: [&str; 6] = ["apple", "banana", "apple", "cherry", "banana", "apple"];
    const N: u64 = 6;
    let offsets_off: u64 = 16;
    let bytes_off: u64 = offsets_off + (N + 1) * 8;

    let mut header = Vec::new();
    header.extend_from_slice(&N.to_le_bytes()); // [0..8)
    let mut offsets = Vec::new();
    let mut blob = Vec::new();
    offsets.extend_from_slice(&0u64.to_le_bytes());
    for r in rows {
        blob.extend_from_slice(r.as_bytes());
        offsets.extend_from_slice(&(blob.len() as u64).to_le_bytes());
    }

    let resident = runtime
        .retain_device_memory_chunks(
            0,
            bytes_off + blob.len() as u64,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: offsets_off,
                    bytes: &offsets,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: bytes_off,
                    bytes: &blob,
                },
            ],
        )
        .expect("retain resident device memory");

    let eq = resident
        .expr_text_eq_scalar_filter(
            offsets_off,
            bytes_off,
            blob.len() as u64,
            b"apple",
            false,
            N,
        )
        .expect("text = apple");
    assert_eq!(eq, vec![0, 2, 5], "text = 'apple' => rows 0,2,5");

    let ne = resident
        .expr_text_eq_scalar_filter(offsets_off, bytes_off, blob.len() as u64, b"apple", true, N)
        .expect("text <> apple");
    assert_eq!(ne, vec![1, 3, 4], "text <> 'apple' => rows 1,3,4");

    let banana = resident
        .expr_text_eq_scalar_filter(
            offsets_off,
            bytes_off,
            blob.len() as u64,
            b"banana",
            false,
            N,
        )
        .expect("text = banana");
    assert_eq!(banana, vec![1, 4], "text = 'banana' => rows 1,4");

    let none = resident
        .expr_text_eq_scalar_filter(
            offsets_off,
            bytes_off,
            blob.len() as u64,
            b"grape",
            false,
            N,
        )
        .expect("text = grape");
    assert!(none.is_empty(), "text = 'grape' matches nothing");

    // length mismatches are NOT equal (equality is full-string, not prefix/contains)
    let prefix = resident
        .expr_text_eq_scalar_filter(offsets_off, bytes_off, blob.len() as u64, b"app", false, N)
        .expect("text = app");
    assert!(prefix.is_empty(), "text = 'app' (shorter) matches nothing");
    let longer = resident
        .expr_text_eq_scalar_filter(
            offsets_off,
            bytes_off,
            blob.len() as u64,
            b"apples",
            false,
            N,
        )
        .expect("text = apples");
    assert!(
        longer.is_empty(),
        "text = 'apples' (longer) matches nothing"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_typed_filters_reject_out_of_bounds_descriptors_before_launch() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let bytes = [0_u8; 64];
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            bytes.len() as u64,
            &[CudaDeviceMemoryChunk {
                byte_offset: 0,
                bytes: &bytes,
            }],
        )
        .expect("retain resident device memory");

    assert!(resident
        .expr_i64_compare_scalar_filter(60, 0, false, 0, 1)
        .is_err());
    assert!(resident
        .expr_i64_compare_columns_filter(0, 60, 0, 1)
        .is_err());
    assert!(resident
        .expr_i128_compare_scalar_filter(56, 0, false, 0, 1)
        .is_err());
    assert!(resident
        .expr_i128_compare_columns_filter(0, 56, 0, 1)
        .is_err());
    assert!(resident
        .expr_uuid_compare_scalar_filter(56, &[0; 16], false, 0, 1, &[])
        .is_err());
    assert!(resident
        .expr_uuid_compare_columns_filter(0, 56, 0, 1, &[])
        .is_err());
    assert!(resident.expr_bool_to_mask_filter(64, false, 1).is_err());
    assert!(resident
        .expr_uuid_compare_scalar_filter(0, &[0; 16], false, 0, 1, &[64])
        .is_err());
    assert!(resident
        .expr_text_eq_scalar_filter(60, 0, 0, b"", false, 1)
        .is_err());

    assert_eq!(
        resident
            .expr_i64_compare_scalar_filter(0, 0, false, 0, 1)
            .expect("valid filter after rejected descriptors"),
        vec![0],
        "rejected safe-API descriptors must not poison the CUDA context"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_filtered_aggregates_reject_invalid_windows_before_launch() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let bytes = [0_u8; 64];
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            bytes.len() as u64,
            &[CudaDeviceMemoryChunk {
                byte_offset: 0,
                bytes: &bytes,
            }],
        )
        .expect("retain resident device memory");

    assert!(resident.sum_i32_at_indices_from_payload(60, &[1]).is_err());
    assert!(resident
        .sum_i64_at_indices_i128_from_payload(56, &[1])
        .is_err());
    assert!(resident.min_i32_at_indices_from_payload(60, &[1]).is_err());
    assert!(resident.max_i32_at_indices_from_payload(60, &[1]).is_err());
    assert!(resident.min_i64_at_indices_from_payload(56, &[1]).is_err());
    assert!(resident.max_i64_at_indices_from_payload(56, &[1]).is_err());
    assert!(resident.min_i128_at_indices_from_payload(48, &[1]).is_err());
    assert!(resident.max_i128_at_indices_from_payload(48, &[1]).is_err());
    assert!(resident.sum_i128_at_indices_from_payload(48, &[1]).is_err());
    assert!(resident.sum_i32_at_indices_from_payload(0, &[]).is_err());
    assert!(resident
        .sum_i64_at_indices_i128_from_payload(0, &[])
        .is_err());

    assert_eq!(
        resident
            .sum_i32_at_indices_from_payload(0, &[0, 1])
            .expect("valid aggregate after rejected windows"),
        0,
        "rejected safe-API windows must not poison the CUDA context"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_text_filters_fail_closed_on_malformed_resident_spans() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    const N: u64 = 2;
    let offsets_off = 8_u64;
    let bytes_off = offsets_off + (N + 1) * 8;
    let mut offsets = Vec::new();
    for offset in [0_u64, u64::MAX, 0] {
        offsets.extend_from_slice(&offset.to_le_bytes());
    }
    let blob = *b"x";
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            bytes_off + blob.len() as u64,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: offsets_off,
                    bytes: &offsets,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: bytes_off,
                    bytes: &blob,
                },
            ],
        )
        .expect("retain malformed text descriptor safely");

    assert!(resident
        .expr_text_eq_scalar_filter(offsets_off, bytes_off, blob.len() as u64, b"x", false, N,)
        .expect("malformed equality spans fail closed")
        .is_empty());
    assert!(resident
        .expr_text_eq_scalar_filter(offsets_off, bytes_off, blob.len() as u64, b"x", true, N,)
        .expect("malformed inequality spans fail closed")
        .is_empty());
    assert!(resident
        .expr_text_compare_scalar_filter(
            offsets_off,
            bytes_off,
            blob.len() as u64,
            b"x",
            false,
            0,
            N,
            &[],
        )
        .expect("malformed ordered-comparison spans fail closed")
        .is_empty());
    assert!(resident
        .expr_text_like_scalar_filter(
            offsets_off,
            bytes_off,
            blob.len() as u64,
            &[u32::from(b'x')],
            N,
        )
        .expect("malformed LIKE spans fail closed")
        .is_empty());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_text_like_scalar_matches_rows() {
    // LIKE over a resident TEXT column (the type matrix, doc 19): the host compiles a pattern to
    // u32 tokens (op<<8 | byte; 0=literal, 1=`_`, 2=`%`); the kernel backtracks. Differential vs a
    // Rust byte-wise oracle over many patterns. ASCII data, so byte-`_` == char-`_`. Offsets at a
    // 4-mod-8 offset.
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    // test helper: pattern -> kernel tokens (no escapes here; the engine handles `\`)
    fn like_tokens(pattern: &str) -> Vec<u32> {
        pattern
            .bytes()
            .map(|b| match b {
                b'%' => 2u32 << 8,
                b'_' => 1u32 << 8,
                other => u32::from(other),
            })
            .collect()
    }
    // Rust oracle: iterative byte-wise `%`/`_` backtracking match.
    fn like_match(text: &[u8], pat: &[u8]) -> bool {
        let (n, m) = (text.len(), pat.len());
        let (mut s, mut p) = (0usize, 0usize);
        let (mut star, mut sstar) = (None::<usize>, 0usize);
        while s < n {
            if p < m && (pat[p] == b'_' || (pat[p] != b'%' && pat[p] == text[s])) {
                s += 1;
                p += 1;
            } else if p < m && pat[p] == b'%' {
                star = Some(p);
                sstar = s;
                p += 1;
            } else if let Some(sp) = star {
                p = sp + 1;
                sstar += 1;
                s = sstar;
            } else {
                return false;
            }
        }
        while p < m && pat[p] == b'%' {
            p += 1;
        }
        p == m
    }

    let rows = ["apple", "apply", "banana", "grape", "applet", "ape", ""];
    let n = rows.len() as u64;
    let offsets_off: u64 = 16; // safe resident text descriptors require u64 alignment
    let bytes_off: u64 = offsets_off + (n + 1) * 8;
    let mut header = Vec::new();
    header.extend_from_slice(&n.to_le_bytes());
    let mut offsets = Vec::new();
    let mut blob = Vec::new();
    offsets.extend_from_slice(&0u64.to_le_bytes());
    for r in rows {
        blob.extend_from_slice(r.as_bytes());
        offsets.extend_from_slice(&(blob.len() as u64).to_le_bytes());
    }
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            bytes_off + blob.len() as u64,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: offsets_off,
                    bytes: &offsets,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: bytes_off,
                    bytes: &blob,
                },
            ],
        )
        .expect("retain resident device memory");

    for pattern in [
        "app%", "%e", "a_p%", "%an%", "_____", "%", "", "apple", "xyz%", "ap_le", "%a%a%", "_",
        "%%", "apple%", "%apple",
    ] {
        let tokens = like_tokens(pattern);
        let got = resident
            .expr_text_like_scalar_filter(offsets_off, bytes_off, blob.len() as u64, &tokens, n)
            .unwrap_or_else(|e| panic!("LIKE '{pattern}': {e:?}"));
        let expected: Vec<u32> = rows
            .iter()
            .enumerate()
            .filter(|(_, r)| like_match(r.as_bytes(), pattern.as_bytes()))
            .map(|(i, _)| i as u32)
            .collect();
        assert_eq!(got, expected, "LIKE '{pattern}' mismatch vs oracle");

        // The general predicate VM must use the exact same eight-argument matcher ABI, including
        // the checked text-byte limit. This is the nullable/compound-LIKE path; omitting the limit
        // shifts the token/count/output arguments and deterministically faults the CUDA context.
        let token_bytes = tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect::<Vec<_>>();
        let vm_got = resident
            .run_expr_predicate_filter_with_text(
                &[ExprStep::TextLikeMask {
                    offsets_byte_offset: offsets_off,
                    bytes_byte_offset: bytes_off,
                    bytes_len: blob.len() as u64,
                    pattern_idx: 0,
                }],
                &[token_bytes],
                n,
                ResidentElemType::I32,
            )
            .unwrap_or_else(|e| panic!("VM LIKE '{pattern}': {e:?}"));
        assert_eq!(vm_got, expected, "VM LIKE '{pattern}' mismatch vs oracle");
    }

    assert!(
        resident
            .run_expr_predicate_filter_with_text(
                &[ExprStep::TextLikeMask {
                    offsets_byte_offset: offsets_off,
                    bytes_byte_offset: bytes_off,
                    bytes_len: blob.len() as u64 + 1,
                    pattern_idx: 0,
                }],
                &[Vec::new()],
                n,
                ResidentElemType::I32,
            )
            .is_err(),
        "the VM LIKE path must reject an out-of-bounds text blob before launch"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_text_predicate_vm_scalar_abi_matches_rows() {
    // The compound-predicate VM must pass the exact bounded-text ABI used by the standalone
    // equality and ordering launchers. Keep the offsets contract-valid and the payload tiny:
    // transient catalog joins exercise precisely this shape.
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let rows = ["r", "v", "r", ""];
    let n = rows.len() as u64;
    let offsets_off = 16_u64;
    let bytes_off = offsets_off + (n + 1) * std::mem::size_of::<u64>() as u64;
    let mut header = Vec::new();
    header.extend_from_slice(&n.to_le_bytes());
    let mut offsets = Vec::new();
    let mut blob = Vec::new();
    offsets.extend_from_slice(&0_u64.to_le_bytes());
    for row in rows {
        blob.extend_from_slice(row.as_bytes());
        offsets.extend_from_slice(&(blob.len() as u64).to_le_bytes());
    }
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            bytes_off + blob.len() as u64,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: offsets_off,
                    bytes: &offsets,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: bytes_off,
                    bytes: &blob,
                },
            ],
        )
        .expect("retain resident device memory");

    let text = |needle: &[u8], step| {
        resident
            .run_expr_predicate_filter_with_text(
                &[step],
                &[needle.to_vec()],
                n,
                ResidentElemType::I32,
            )
            .expect("bounded text predicate VM")
    };
    assert_eq!(
        text(
            b"r",
            ExprStep::TextEqMask {
                offsets_byte_offset: offsets_off,
                bytes_byte_offset: bytes_off,
                bytes_len: blob.len() as u64,
                needle_idx: 0,
                negate: false,
            },
        ),
        vec![0, 2]
    );
    assert_eq!(
        text(
            b"r",
            ExprStep::TextCmpMask {
                offsets_byte_offset: offsets_off,
                bytes_byte_offset: bytes_off,
                bytes_len: blob.len() as u64,
                needle_idx: 0,
                scalar_on_left: false,
                cmp: 3,
            },
        ),
        vec![1]
    );
    assert!(
        resident
            .run_expr_predicate_filter_with_text(
                &[ExprStep::TextEqMask {
                    offsets_byte_offset: offsets_off,
                    bytes_byte_offset: bytes_off,
                    bytes_len: blob.len() as u64 + 1,
                    needle_idx: 0,
                    negate: false,
                }],
                &[b"r".to_vec()],
                n,
                ResidentElemType::I32,
            )
            .is_err(),
        "the VM equality path must reject an out-of-bounds text blob before launch"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_expr_compare_buffers_filter_evaluates_col_vs_col_on_gpu() {
    // Col-vs-col / expr-vs-expr comparisons: the comparison RHS is an arbitrary expression buffer,
    // not just a literal. GPU-NATIVE closed-form oracle: a[i]=i, b[i]=N-1-i (strictly decreasing),
    // so `a < b` <=> i < N-1-i <=> 2i < N-1 (a contiguous range), and a/b never tie.
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    const N: u64 = 600;
    let a_off = std::mem::size_of::<u64>() as u64;
    let b_off = a_off + N * std::mem::size_of::<i32>() as u64;
    let mut header = Vec::new();
    header.extend_from_slice(&N.to_le_bytes());
    let mut a_bytes = Vec::new();
    let mut b_bytes = Vec::new();
    for i in 0..N as i32 {
        a_bytes.extend_from_slice(&i.to_le_bytes());
        b_bytes.extend_from_slice(&(N as i32 - 1 - i).to_le_bytes());
    }
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            b_off + b_bytes.len() as u64,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: a_off,
                    bytes: &a_bytes,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: b_off,
                    bytes: &b_bytes,
                },
            ],
        )
        .expect("retain resident device memory");

    // a < b  (cmp=1): program leaves [a, b]; 2i < 599 <=> i <= 299.
    let p_ab = [
        ExprStep::LoadColumn { byte_offset: a_off },
        ExprStep::LoadColumn { byte_offset: b_off },
    ];
    let lt = resident
        .run_expr_compare_buffers_filter(&p_ab, N, 1)
        .expect("a < b");
    let lt_expected: Vec<u32> = (0..300).collect();
    assert_eq!(lt, lt_expected, "a < b <=> i in [0, 300)");

    // a > b  (cmp=3): the complementary range (a and b never tie).
    let gt = resident
        .run_expr_compare_buffers_filter(&p_ab, N, 3)
        .expect("a > b");
    let gt_expected: Vec<u32> = (300..N as u32).collect();
    assert_eq!(gt, gt_expected, "a > b <=> i in [300, 600)");

    // expr-vs-expr: a < a*2  (a=i, a*2=2i): i < 2i <=> i >= 1.
    let p_expr = [
        ExprStep::LoadColumn { byte_offset: a_off },
        ExprStep::LoadColumn { byte_offset: a_off },
        ExprStep::ScalarBinary {
            op: 2,
            scalar: 2,
            scalar_on_left: false,
        },
    ];
    let expr_lt = resident
        .run_expr_compare_buffers_filter(&p_expr, N, 1)
        .expect("a < a*2");
    let expr_expected: Vec<u32> = (1..N as u32).collect();
    assert_eq!(expr_lt, expr_expected, "a < a*2 <=> i >= 1");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_expr_predicate_filter_evaluates_boolean_predicates_on_gpu() {
    // AND/OR/Ne over masks (boolean combinators): each comparison produces a 0/1 mask, MaskBinary
    // combines them, the terminal compacts the mask. GPU-NATIVE closed-form oracle: a[i]=i, so the
    // matching sets are explicit index ranges (no CPU re-implementation of the operator).
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    const N: u64 = 600;
    let a_off = std::mem::size_of::<u64>() as u64;
    let mut header = Vec::new();
    header.extend_from_slice(&N.to_le_bytes());
    let mut a_bytes = Vec::new();
    for i in 0..N as i32 {
        a_bytes.extend_from_slice(&i.to_le_bytes());
    }
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            a_off + a_bytes.len() as u64,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: a_off,
                    bytes: &a_bytes,
                },
            ],
        )
        .expect("retain resident device memory");

    let cmp_scalar = |cmp: u32, scalar: i32| ExprStep::CompareScalar {
        cmp,
        scalar,
        scalar_on_left: false,
    };
    let load = ExprStep::LoadColumn { byte_offset: a_off };

    // a > 200 AND a < 400  ->  i in (200, 400) = [201, 400).
    let p_and = [
        load,
        cmp_scalar(3, 200),
        load,
        cmp_scalar(1, 400),
        ExprStep::MaskBinary { op: 0 },
    ];
    let got_and = resident
        .run_expr_predicate_filter(&p_and, N, ResidentElemType::I32)
        .expect("a>200 AND a<400");
    let and_expected: Vec<u32> = (201..400).collect();
    assert_eq!(got_and, and_expected, "a>200 AND a<400 <=> i in [201, 400)");

    // a < 100 OR a > 500  ->  [0, 100) U [501, 600).
    let p_or = [
        load,
        cmp_scalar(1, 100),
        load,
        cmp_scalar(3, 500),
        ExprStep::MaskBinary { op: 1 },
    ];
    let got_or = resident
        .run_expr_predicate_filter(&p_or, N, ResidentElemType::I32)
        .expect("a<100 OR a>500");
    let mut or_expected: Vec<u32> = (0..100).collect();
    or_expected.extend(501..N as u32);
    assert_eq!(
        got_or, or_expected,
        "a<100 OR a>500 <=> [0,100) U [501,600)"
    );

    // a != 300  ->  everything except index 300.
    let p_ne = [load, cmp_scalar(5, 300)];
    let got_ne = resident
        .run_expr_predicate_filter(&p_ne, N, ResidentElemType::I32)
        .expect("a != 300");
    let mut ne_expected: Vec<u32> = (0..300).collect();
    ne_expected.extend(301..N as u32);
    assert_eq!(got_ne, ne_expected, "a != 300 <=> all rows but index 300");
}

/// SV3b (mixed-width VM): ONE predicate program run at `elem = I32` that combines an i32 value predicate
/// (`id == 5`) with an i64 `deleted_by > read_txn_id` visibility compare, via `LoadColumnI64` +
/// `CompareScalarI64` — the capability that unblocks point-lookup visibility. Hand-built columns are the
/// GPU-native oracle. NON-VACUITY: the i32-only WHERE keeps the deleted row; the i64 visibility compare
/// is what removes it — so a broken mixed-width path (i64 step no-op or mis-read) fails.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_expr_mixed_width_i32_where_and_i64_visibility() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    const N: u64 = 4;
    let header = N.to_le_bytes().to_vec();
    let id_off = std::mem::size_of::<u64>() as u64; // i32 id column after the 8-byte header
    let db_off = id_off + N * std::mem::size_of::<i32>() as u64; // i64 deleted_by after the i32 column
    let ids: [i32; 4] = [5, 5, 7, 5];
    // row1 deleted @10, row3 deleted @3; rows 0,2 LIVE. The live sentinel is 0x7F7F... — a LARGE POSITIVE
    // signed i64 (the compare kernel is signed s64), NOT u64::MAX (which is -1 as signed and would fail
    // `> read_txn_id`). This mirrors the engine's `DELETED_BY_LIVE`.
    const LIVE: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    let deleted_by: [u64; 4] = [LIVE, 10, LIVE, 3];
    let mut id_bytes = Vec::new();
    for v in ids {
        id_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut db_bytes = Vec::new();
    for v in deleted_by {
        db_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            db_off + db_bytes.len() as u64,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: id_off,
                    bytes: &id_bytes,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: db_off,
                    bytes: &db_bytes,
                },
            ],
        )
        .expect("retain resident device memory");

    // MIXED program at elem=I32: `id == 5` (i32) AND `deleted_by > 6` (i64).
    let mixed = [
        ExprStep::LoadColumn {
            byte_offset: id_off,
        },
        ExprStep::CompareScalar {
            cmp: 0,
            scalar: 5,
            scalar_on_left: false,
        }, // 0 = `=`
        ExprStep::LoadColumnI64 {
            byte_offset: db_off,
        },
        ExprStep::CompareScalarI64 {
            cmp: 3,
            scalar: 6,
            scalar_on_left: false,
        }, // 3 = `>`
        ExprStep::MaskBinary { op: 0 }, // 0 = AND
    ];
    // row0 (id5, MAX>6) yes; row1 (id5, 10>6) yes; row2 (id7) no; row3 (id5, 3>6) no.
    assert_eq!(
        resident
            .run_expr_predicate_filter(&mixed, N, ResidentElemType::I32)
            .expect("mixed i32 WHERE + i64 visibility"),
        vec![0, 1],
        "id==5 AND deleted_by>6"
    );

    // NON-VACUITY: id==5 ALONE (no visibility) keeps the deleted row3 -> [0,1,3]; the i64 visibility
    // compare is what removes it. A no-op mixed-width path would leave row3 in.
    let where_only = [
        ExprStep::LoadColumn {
            byte_offset: id_off,
        },
        ExprStep::CompareScalar {
            cmp: 0,
            scalar: 5,
            scalar_on_left: false,
        },
    ];
    assert_eq!(
        resident
            .run_expr_predicate_filter(&where_only, N, ResidentElemType::I32)
            .unwrap(),
        vec![0, 1, 3],
        "id==5 alone keeps the deleted row"
    );

    // The i64 visibility compare ALONE (the no-WHERE scan/COUNT case) at elem=I64: deleted_by>6 -> [0,1,2].
    let vis_only = [
        ExprStep::LoadColumnI64 {
            byte_offset: db_off,
        },
        ExprStep::CompareScalarI64 {
            cmp: 3,
            scalar: 6,
            scalar_on_left: false,
        },
    ];
    assert_eq!(
        resident
            .run_expr_predicate_filter(&vis_only, N, ResidentElemType::I64)
            .unwrap(),
        vec![0, 1, 2],
        "deleted_by>6 alone (scan visibility) hides rows deleted at <= the snapshot"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_expr_comparison_codes_and_operand_order_coverage() {
    // Closes adversarial-audit coverage gaps (all verified correct by the audit, now permanent):
    // every comparison code on BOTH the fused compact path (run_expr_arith_filter, codes 0..4) and
    // the mask path (run_expr_predicate_filter, codes 0..5), the mask CompareScalar
    // {scalar_on_left:true} branch (reachable from `WHERE 5 < a`), buffer-vs-buffer le/lt/ne, and
    // the new cmp>4 guard on the fused compact fns. a[i]=i; closed-form ranges.
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    const N: u64 = 600;
    let a_off = std::mem::size_of::<u64>() as u64;
    let mut header = Vec::new();
    header.extend_from_slice(&N.to_le_bytes());
    let mut a_bytes = Vec::new();
    for i in 0..N as i32 {
        a_bytes.extend_from_slice(&i.to_le_bytes());
    }
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            a_off + a_bytes.len() as u64,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: a_off,
                    bytes: &a_bytes,
                },
            ],
        )
        .expect("retain resident device memory");
    let load = ExprStep::LoadColumn { byte_offset: a_off };
    let range = |lo: u32, hi: u32| -> Vec<u32> { (lo..hi).collect() };

    // Fused compact path (run_expr_arith_filter, value buffer vs scalar): eq / le / ge.
    assert_eq!(
        resident.run_expr_arith_filter(&[load], N, 0, 300).unwrap(),
        vec![300u32],
        "a == 300"
    );
    assert_eq!(
        resident.run_expr_arith_filter(&[load], N, 2, 100).unwrap(),
        range(0, 101),
        "a <= 100"
    );
    assert_eq!(
        resident.run_expr_arith_filter(&[load], N, 4, 500).unwrap(),
        range(500, 600),
        "a >= 500"
    );
    // cmp 5 (ne) is mask-path only — the fused path now rejects it instead of returning empty.
    assert!(
        resident.run_expr_arith_filter(&[load], N, 5, 0).is_err(),
        "cmp=ne must be rejected by the fused compact path, not silently empty"
    );

    // Mask path (run_expr_predicate_filter): eq / le / ge + scalar_on_left:true.
    let csm = |cmp: u32, scalar: i32, left: bool| ExprStep::CompareScalar {
        cmp,
        scalar,
        scalar_on_left: left,
    };
    assert_eq!(
        resident
            .run_expr_predicate_filter(&[load, csm(0, 300, false)], N, ResidentElemType::I32)
            .unwrap(),
        vec![300u32],
        "mask a == 300"
    );
    assert_eq!(
        resident
            .run_expr_predicate_filter(&[load, csm(2, 100, false)], N, ResidentElemType::I32)
            .unwrap(),
        range(0, 101),
        "mask a <= 100"
    );
    assert_eq!(
        resident
            .run_expr_predicate_filter(&[load, csm(4, 500, false)], N, ResidentElemType::I32)
            .unwrap(),
        range(500, 600),
        "mask a >= 500"
    );
    assert_eq!(
        resident
            .run_expr_predicate_filter(&[load, csm(3, -100, false)], N, ResidentElemType::I32)
            .unwrap(),
        range(0, 600),
        "mask a > -100 preserves the signed scalar ABI"
    );
    // scalar_on_left: `5 < a` <=> a > 5 <=> [6, 600).
    assert_eq!(
        resident
            .run_expr_predicate_filter(&[load, csm(1, 5, true)], N, ResidentElemType::I32)
            .unwrap(),
        range(6, 600),
        "5 < a (scalar_on_left) <=> a > 5"
    );
    // `5 >= a` <=> a <= 5 <=> [0, 6).
    assert_eq!(
        resident
            .run_expr_predicate_filter(&[load, csm(4, 5, true)], N, ResidentElemType::I32)
            .unwrap(),
        range(0, 6),
        "5 >= a (scalar_on_left) <=> a <= 5"
    );

    // Buffer-vs-buffer mask, self-compare: a<=a all, a<a none, a!=a none.
    let cb = |cmp: u32| [load, load, ExprStep::CompareBuffers { cmp }];
    assert_eq!(
        resident
            .run_expr_predicate_filter(&cb(2), N, ResidentElemType::I32)
            .unwrap(),
        range(0, N as u32),
        "a <= a is all rows"
    );
    assert!(
        resident
            .run_expr_predicate_filter(&cb(1), N, ResidentElemType::I32)
            .unwrap()
            .is_empty(),
        "a < a is empty"
    );
    assert!(
        resident
            .run_expr_predicate_filter(&cb(5), N, ResidentElemType::I32)
            .unwrap()
            .is_empty(),
        "a != a is empty"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_ordered_i32_compaction_preflight_fails_closed_then_reuses_context() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let values = [10_i32, 20];
    let bytes: Vec<u8> = values.into_iter().flat_map(i32::to_le_bytes).collect();
    let resident = runtime
        .retain_device_memory_copy(0, &bytes)
        .expect("resident device memory");
    resident
        .set_current_context()
        .expect("bind primary context");

    assert_eq!(
        resident.compare_indices_ordered_from_payload(0, 2, 10, 6),
        Err(CudaRuntimeProbeError::UnsupportedComparison(6)),
        "unknown comparison codes must fail before launch"
    );
    assert!(matches!(
        resident.compare_indices_ordered_from_payload(0, u64::from(u32::MAX) + 1, 10, 0,),
        Err(CudaRuntimeProbeError::InvalidInputLength(_))
    ));
    assert!(matches!(
        resident.compare_indices_ordered_from_payload(4, 2, 10, 0),
        Err(CudaRuntimeProbeError::InvalidInputLength(_))
    ));
    assert_eq!(
        resident.compare_indices_ordered_from_payload(1, 1, 10, 0),
        Err(CudaRuntimeProbeError::InvalidInputLength(1)),
        "misaligned resident offsets must fail before device access"
    );
    assert!(resident
        .compare_indices_ordered_from_payload(bytes.len() as u64, 0, 10, 0)
        .expect("empty window at allocation end")
        .is_empty());
    assert!(matches!(
        resident.compare_indices_ordered_from_payload(bytes.len() as u64 + 4, 0, 10, 0),
        Err(CudaRuntimeProbeError::InvalidInputLength(_))
    ));

    {
        let short = resident
            .primary()
            .lease_device_buffer(1)
            .expect("single input lease");
        let too_many = short.capacity as u64 / std::mem::size_of::<i32>() as u64 + 1;
        assert!(matches!(
            launch_cuda_buffer_i32_compare_indices_ordered(&resident, &short, too_many, 0, 0,),
            Err(CudaRuntimeProbeError::InvalidInputLength(_))
        ));

        let rhs = resident
            .primary()
            .lease_device_buffer(1)
            .expect("second input lease");
        let too_many =
            short.capacity.max(rhs.capacity) as u64 / std::mem::size_of::<i32>() as u64 + 1;
        assert!(matches!(
            launch_cuda_resident_i32_compare_buffers_indices_ordered(
                &resident, &short, &rhs, too_many, 0,
            ),
            Err(CudaRuntimeProbeError::InvalidInputLength(_))
        ));
        assert_eq!(
            launch_cuda_resident_i32_compare_buffers_indices_ordered(&resident, &short, &rhs, 1, 5,),
            Err(CudaRuntimeProbeError::UnsupportedComparison(5))
        );
    }

    assert_eq!(
        resident
            .compare_indices_ordered_from_payload(0, 2, 15, 3)
            .expect("valid launch after preflight failures"),
        vec![1],
        "preflight failures must leave the CUDA context reusable"
    );
    assert_eq!(
        resident
            .compare_indices_ordered_from_payload(0, 2, -100, 3)
            .expect("signed negative needle"),
        vec![0, 1],
        "ordered i32 comparison must preserve the signed scalar ABI"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_grouped_int4_full_result_retains_complete_compaction_key_scratch() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let row_count = 65_536usize;
    let value_off = row_count * std::mem::size_of::<i32>();
    let mut payload = Vec::with_capacity(value_off * 2);
    for key in 0..row_count as i32 {
        payload.extend_from_slice(&key.to_le_bytes());
    }
    for _ in 0..row_count {
        payload.extend_from_slice(&1_i32.to_le_bytes());
    }
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            payload.len() as u64,
            &[CudaDeviceMemoryChunk {
                byte_offset: 0,
                bytes: &payload,
            }],
        )
        .expect("resident full-result grouped payload");
    let indices = (0..row_count as u32).collect::<Vec<_>>();

    let mut groups = resident
        .group_by_i32_count_sum_from_payload(
            CudaGroupByInput::resident_i32(0, value_off as u64, row_count as u64),
            &indices,
            grouped_agg_mask::ALL,
        )
        .expect("ordinary int4 full-result compaction must stay allocation-bounded");
    groups.sort_unstable_by_key(|group| group.key);
    assert_eq!(groups.len(), row_count);
    assert_eq!((groups[0].key, groups[0].count, groups[0].sum), (0, 1, 1));
    assert_eq!(
        (
            groups[row_count - 1].key,
            groups[row_count - 1].count,
            groups[row_count - 1].sum,
        ),
        (row_count as i64 - 1, 1, 1)
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_grouped_inputs_reject_or_fail_closed_then_reuse_context() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let row_count = 3u64;
    let key_off = 0u64;
    let value_off = 12u64;
    let offsets_off = 24u64;
    let bytes_off = offsets_off + 4 * 8;
    let numeric_off = (bytes_off + 3).next_multiple_of(4);
    let mut payload = Vec::new();
    for value in [1i32, 1, 2, 10, 20, 30] {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    for offset in [0u64, 100, 100, 100] {
        payload.extend_from_slice(&offset.to_le_bytes());
    }
    payload.extend_from_slice(b"abc");
    payload.resize(numeric_off as usize, 0);
    for value in [1i128, 2, 3] {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            payload.len() as u64,
            &[CudaDeviceMemoryChunk {
                byte_offset: 0,
                bytes: &payload,
            }],
        )
        .expect("resident grouped safety payload");
    let indices = [0u32, 1, 2];

    let fixed_oob = CudaGroupByInput {
        key: CudaGroupKeySource::Fixed(CudaGroupFixedSource::Resident {
            byte_offset: payload.len() as u64 - 1,
            width: 4,
            row_count,
        }),
        value: CudaGroupValueSource::Unused { row_count },
        key_validity_bitmap_offset: None,
        value_validity_bitmap_offset: None,
    };
    assert!(resident
        .group_by_i32_count_sum_minmax_from_payload(fixed_oob, &indices, grouped_agg_mask::COUNT,)
        .is_err());

    let fixed_misaligned = CudaGroupByInput {
        key: CudaGroupKeySource::Fixed(CudaGroupFixedSource::Resident {
            byte_offset: 1,
            width: 4,
            row_count,
        }),
        value: CudaGroupValueSource::Unused { row_count },
        key_validity_bitmap_offset: None,
        value_validity_bitmap_offset: None,
    };
    assert!(resident
        .group_by_i32_count_sum_minmax_from_payload(
            fixed_misaligned,
            &indices,
            grouped_agg_mask::COUNT,
        )
        .is_err());

    let text_misaligned = CudaGroupByInput {
        key: CudaGroupKeySource::Text {
            text: CudaGroupTextSource {
                offsets_byte_offset: 4,
                bytes_byte_offset: bytes_off,
                bytes_len: 3,
                row_count,
            },
            fixed_component: None,
        },
        value: CudaGroupValueSource::Unused { row_count },
        key_validity_bitmap_offset: None,
        value_validity_bitmap_offset: None,
    };
    assert!(resident
        .group_by_i32_count_sum_minmax_from_payload(
            text_misaligned,
            &indices,
            grouped_agg_mask::COUNT,
        )
        .is_err());

    let bitmap_misaligned = CudaGroupByInput {
        key: CudaGroupKeySource::Fixed(CudaGroupFixedSource::Resident {
            byte_offset: key_off,
            width: 4,
            row_count,
        }),
        value: CudaGroupValueSource::Unused { row_count },
        key_validity_bitmap_offset: Some(1),
        value_validity_bitmap_offset: None,
    };
    assert!(resident
        .group_by_i32_count_sum_minmax_from_payload(
            bitmap_misaligned,
            &indices,
            grouped_agg_mask::COUNT,
        )
        .is_err());

    let bad_text = CudaGroupTextSource {
        offsets_byte_offset: offsets_off,
        bytes_byte_offset: bytes_off,
        bytes_len: 3,
        row_count,
    };
    let numeric_value = CudaGroupValueSource::Numeric(CudaGroupFixedSource::Resident {
        byte_offset: numeric_off,
        width: 16,
        row_count,
    });
    let bad_text_key = CudaGroupByInput {
        key: CudaGroupKeySource::Text {
            text: bad_text,
            fixed_component: None,
        },
        value: numeric_value,
        key_validity_bitmap_offset: None,
        value_validity_bitmap_offset: None,
    };
    assert!(
        resident
            .group_by_i32_count_sum_minmax_from_payload(
                bad_text_key,
                &indices,
                grouped_agg_mask::ALL,
            )
            .is_err(),
        "malformed text key must suppress numeric pass two and fail closed"
    );

    let bad_text_value = CudaGroupByInput {
        key: CudaGroupKeySource::Fixed(CudaGroupFixedSource::Resident {
            byte_offset: key_off,
            width: 4,
            row_count,
        }),
        value: CudaGroupValueSource::Text(bad_text),
        key_validity_bitmap_offset: None,
        value_validity_bitmap_offset: None,
    };
    assert!(
        resident
            .group_by_i32_count_sum_minmax_from_payload(
                bad_text_value,
                &indices,
                grouped_agg_mask::ALL,
            )
            .is_err()
    );

    let descriptors = resident
        .upload_group_text_descriptors(&[bad_text])
        .expect("descriptor owner");
    let bad_composite = CudaGroupByInput {
        key: CudaGroupKeySource::Composite {
            fixed: None,
            text: Some(descriptors.descriptors()),
            row_count,
        },
        value: CudaGroupValueSource::Unused { row_count },
        key_validity_bitmap_offset: None,
        value_validity_bitmap_offset: None,
    };
    assert!(resident
        .group_by_i32_count_sum_minmax_from_payload(
            bad_composite,
            &indices,
            grouped_agg_mask::COUNT,
        )
        .is_err());
    assert!(resident
        .group_by_i32_count_sum_kernel_timed(
            CudaGroupByInput::resident_i32(key_off, value_off, row_count),
            &indices,
            true,
            0,
            grouped_agg_mask::ALL,
        )
        .is_err());
    assert_eq!(
        resident
            .group_by_i32_count_sum_kernel_timed(
                CudaGroupByInput::resident_i32(key_off, value_off, row_count),
                &[],
                true,
                1,
                grouped_agg_mask::ALL,
            )
            .expect("empty timed input preserves zero-work semantics"),
        (Vec::new(), 0.0),
    );
    assert!(resident
        .group_by_i32_count_sum_kernel_timed(
            CudaGroupByInput {
                key: CudaGroupKeySource::Fixed(CudaGroupFixedSource::Resident {
                    byte_offset: key_off,
                    width: 4,
                    row_count,
                }),
                value: CudaGroupValueSource::Unused { row_count },
                key_validity_bitmap_offset: None,
                value_validity_bitmap_offset: None,
            },
            &indices,
            true,
            1,
            grouped_agg_mask::ALL,
        )
        .is_err());

    let mut groups = resident
        .group_by_i32_count_sum_from_payload(
            CudaGroupByInput::resident_i32(key_off, value_off, row_count),
            &indices,
            grouped_agg_mask::ALL,
        )
        .expect("valid grouped launch after rejected inputs");
    groups.sort_unstable_by_key(|group| group.key);
    assert_eq!((groups[0].key, groups[0].count, groups[0].sum), (1, 2, 30));
    assert_eq!((groups[1].key, groups[1].count, groups[1].sum), (2, 1, 30));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_derived_column_inputs_fail_closed_then_reuse_context() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let n = 3u64;
    let fixed_off = 0u64;
    let offsets_off = 16u64;
    let bytes_off = offsets_off + 4 * 8;
    let mut payload = Vec::new();
    for value in [1i32, 2, 3] {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    payload.resize(offsets_off as usize, 0);
    // Rows 1 and 2 are malformed despite the offsets table itself being allocation-bounded:
    // row 1 ends beyond bytes_len and row 2 is reversed. Device validation must reject these
    // before the marker's byte loop can dereference either interval.
    for offset in [0u64, 2, 9, 6] {
        payload.extend_from_slice(&offset.to_le_bytes());
    }
    payload.extend_from_slice(b"abcdef");
    let resident = std::sync::Arc::new(
        runtime
            .retain_device_memory_chunks(
                0,
                payload.len() as u64,
                &[CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &payload,
                }],
            )
            .expect("derived safety payload"),
    );

    assert_eq!(
        resident
            .run_expr_predicate_filter_with_text(
                &[ExprStep::TextCmpColumnsMask {
                    a_offsets_byte_offset: offsets_off,
                    a_bytes_byte_offset: bytes_off,
                    a_bytes_len: 6,
                    b_offsets_byte_offset: offsets_off,
                    b_bytes_byte_offset: bytes_off,
                    b_bytes_len: 6,
                    cmp: 0,
                }],
                &[],
                n,
                ResidentElemType::I32,
            )
            .expect("malformed text intervals must return false without poisoning CUDA"),
        vec![0],
    );

    assert!(resident
        .arith_value_column_device(
            &[ExprStep::LoadColumn {
                byte_offset: payload.len() as u64 - 1,
            }],
            n,
            ResidentElemType::I32,
        )
        .is_err());
    assert!(resident
        .arith_value_column_device(
            &[ExprStep::LoadColumn { byte_offset: 1 }],
            n,
            ResidentElemType::I32,
        )
        .is_err());
    assert!(resident.bool_to_int4_column_device(1, n).is_err());
    assert!(resident
        .bool_to_int4_column_device(payload.len() as u64, n)
        .is_err());
    assert!(resident.pack_two_int4_cols_device(1, fixed_off, n).is_err());
    assert!(resident
        .pack_two_int4_cols_device(fixed_off, payload.len() as u64 - 1, n)
        .is_err());
    assert!(resident
        .pack_two_cols_i128_device(1, 4, fixed_off, 4, n)
        .is_err());
    assert!(resident.widen_col_to_i64_device(1, 4, n).is_err());

    let wide = [CudaWideKeyDescriptor {
        source: CudaWideKeySource::ResidentI32 {
            byte_offset: fixed_off,
        },
        destination_byte_offset: 0,
    }];
    assert!(resident
        .build_wide_key_device(
            &wide,
            16,
            n,
            &[
                CudaWideKeyValidity::NonNullable,
                CudaWideKeyValidity::NonNullable
            ],
        )
        .is_err());
    let overlap = [wide[0], wide[0]];
    assert!(resident
        .build_wide_key_device(&overlap, 16, n, &[])
        .is_err());
    let fixed_keys = [1i64, 10, 1, 10, 2, 20];
    assert!(resident
        .mark_new_distinct_device(&fixed_keys, &[0, 0, 2], n, 2)
        .is_err());

    let text = CudaGroupTextSource {
        offsets_byte_offset: offsets_off,
        bytes_byte_offset: bytes_off,
        bytes_len: 6,
        row_count: n,
    };
    let perm = [0u32, 1, 2];
    let indices = [0u64, 1, 2];
    let groups = [1i64, 1, 2];
    assert!(resident
        .mark_new_distinct_text_device(&[0, 0, 2], &indices, &groups, text, n)
        .is_err());
    assert!(resident
        .mark_new_distinct_text_device(&perm, &[0, 1, 3], &groups, text, n)
        .is_err());
    assert!(
        resident
            .mark_new_distinct_text_device(&perm, &indices, &groups, text, n)
            .is_err(),
        "device offsets outside the declared blob must fail closed"
    );

    const THREADS: usize = 4;
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
    let mut handles = Vec::new();
    for _ in 0..THREADS {
        let resident = std::sync::Arc::clone(&resident);
        let barrier = std::sync::Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            resident
                .set_current_context()
                .expect("bind primary context on validator thread");
            barrier.wait();
            for _ in 0..8 {
                assert!(resident
                    .mark_new_distinct_text_device(&perm, &indices, &groups, text, n)
                    .is_err());
            }
        }));
    }
    for handle in handles {
        handle.join().expect("concurrent validator thread");
    }

    let (_group, _new_distinct) = resident
        .mark_new_distinct_device(&fixed_keys, &perm, n, 2)
        .expect("valid derived launch after host/device rejects");
    assert_eq!(
        resident
            .count_i32_equal_from_payload(fixed_off, n, 2, None)
            .expect("context remains reusable"),
        1
    );
}
