    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_driver_runtime_filters_u32_equality_mask() {
        let runtime = CudaDriverRuntime::probe().unwrap();

        let input = [3, 8, 3, 0, 11, 3];
        let mask = runtime.filter_equal_u32_mask(&input, 3).unwrap();

        assert_eq!(mask, vec![true, false, true, false, false, true]);
        assert_eq!(
            runtime.filter_equal_u32_mask(&[], 3).unwrap(),
            Vec::<bool>::new()
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_driver_runtime_filters_all_rows_mask() {
        let runtime = CudaDriverRuntime::probe().unwrap();

        assert_eq!(runtime.filter_all_mask(4).unwrap(), vec![true; 4]);
        assert_eq!(runtime.filter_all_mask(0).unwrap(), Vec::<bool>::new());
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_driver_runtime_filters_byte_equality_mask() {
        let runtime = CudaDriverRuntime::probe().unwrap();

        let input = [
            b"open".as_slice(),
            b"closed".as_slice(),
            b"open".as_slice(),
            b"".as_slice(),
            b"opened".as_slice(),
        ];
        let mask = runtime.filter_equal_bytes_mask(&input, b"open").unwrap();

        assert_eq!(mask, vec![true, false, true, false, false]);
        assert_eq!(
            runtime
                .filter_equal_bytes_mask(&[b"".as_slice(), b"x".as_slice()], b"")
                .unwrap(),
            vec![true, false]
        );
        assert_eq!(
            runtime.filter_equal_bytes_mask(&[], b"open").unwrap(),
            Vec::<bool>::new()
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_driver_runtime_filters_byte_range_mask() {
        let runtime = CudaDriverRuntime::probe().unwrap();

        let input = [
            b"acct:0".as_slice(),
            b"acct:1".as_slice(),
            b"acct:7".as_slice(),
            b"acct:9".as_slice(),
            b"acct".as_slice(),
            b"user:1".as_slice(),
        ];
        let mask = runtime
            .filter_bytes_range_mask(&input, b"acct:1", b"acct:9")
            .unwrap();

        assert_eq!(mask, vec![false, true, true, false, false, false]);
        assert_eq!(
            runtime
                .filter_bytes_range_mask(&[b"".as_slice(), b"a".as_slice()], b"", b"a")
                .unwrap(),
            vec![true, false]
        );
        assert_eq!(
            runtime
                .filter_bytes_range_mask(&[], b"acct:1", b"acct:9")
                .unwrap(),
            Vec::<bool>::new()
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_driver_runtime_inspects_mvcc_row_batch_lengths() {
        let runtime = CudaDriverRuntime::probe().unwrap();
        let batch = CudaMvccRowBatch::from_key_values_with_metadata([
            (b"acct:1".as_slice(), b"open".as_slice(), 3, 9, Some(100)),
            (
                b"acct:22".as_slice(),
                b"".as_slice(),
                4,
                u64::MAX,
                Some(101),
            ),
            (b"".as_slice(), b"closed".as_slice(), 5, 8, Some(102)),
        ])
        .unwrap();

        assert_eq!(
            runtime.mvcc_row_batch_lengths(&batch).unwrap(),
            vec![(6, 4), (7, 0), (0, 6)]
        );
        let empty = CudaMvccRowBatch::from_key_values(Vec::<(&[u8], &[u8])>::new()).unwrap();
        assert_eq!(
            runtime.mvcc_row_batch_lengths(&empty).unwrap(),
            Vec::<(u32, u32)>::new()
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_driver_runtime_filters_mvcc_visibility_mask() {
        let runtime = CudaDriverRuntime::probe().unwrap();
        let batch = CudaMvccRowBatch::from_key_values_with_metadata([
            (b"pre".as_slice(), b"hidden".as_slice(), 4, u64::MAX, None),
            (b"live".as_slice(), b"open".as_slice(), 2, u64::MAX, None),
            (b"gone".as_slice(), b"closed".as_slice(), 1, 3, None),
            (b"edge".as_slice(), b"visible".as_slice(), 3, 5, None),
        ])
        .unwrap();

        assert_eq!(
            runtime.mvcc_visibility_mask(&batch, 3).unwrap(),
            vec![false, true, false, true]
        );
        assert_eq!(
            runtime.mvcc_visibility_mask(&batch, 5).unwrap(),
            vec![true, true, false, false]
        );
        let empty = CudaMvccRowBatch::from_key_values(Vec::<(&[u8], &[u8])>::new()).unwrap();
        assert_eq!(
            runtime.mvcc_visibility_mask(&empty, 3).unwrap(),
            Vec::<bool>::new()
        );
    }

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
            .project_i32_compare_from_payload(
                byte_offset,
                row_count,
                needle,
                CudaI32Comparison::Gte,
            )
            .expect("final project_i32_compare");
        assert_eq!(final_rows, vec![20, 30, 20, 40]);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_resident_i32_equal_project_matches_expected_under_concurrent_pool_reuse() {
        // P2-M2 follow-up 3 regression for centralizing the error-path stream drain into
        // `launch_on_pooled_stream`. `match_project_i32_equal_from_payload` (→
        // `launch_cuda_resident_i32_equal_project`) is a PRIMARY-path DIRECT caller of that helper
        // with NO per-op `drain_err` of its own: it leases `values_guard`/`count_guard` from the
        // SHARED device-buffer pool, hands their pointers to the kernel, and relies entirely on the
        // helper to drain the stream before any error unwinds those leases back to the pool. The
        // helper's success path now routes the in-flight steps through `map_err(drain_err)`, which
        // skips the closure on `Ok`, so the single covering sync stays the only success-path sync —
        // this test pins that the success path is byte-exact AND unchanged under contention. It
        // (a) checks single-threaded parity for the multi-column projection shape, then (b) hammers
        // the route on N threads over ONE shared resident allocation (same pools) asserting the
        // exact result each call. A buffer re-leased before its kernel drained, or a regressed sync,
        // would surface as a mismatch or a crash. Output rows arrive in atomic-append (warp) order,
        // so every comparison sorts both sides — order-independent, like the row-indices route.
        use std::sync::{Arc, Barrier};

        fn sorted(mut rows: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
            rows.sort();
            rows
        }

        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        // Known payload: header(row_count) + filter column + two projection columns (so the
        // `Vec<Vec<i32>>` multi-column row shape is exercised, not just a single value per row).
        let row_count = 6_u64;
        let i32_bytes = std::mem::size_of::<i32>() as u64;
        let filter_offset = std::mem::size_of::<u64>() as u64;
        let proj_a_offset = filter_offset + row_count * i32_bytes;
        let proj_b_offset = proj_a_offset + row_count * i32_bytes;

        let filter_values = [5_i32, 9, 5, 7, 5, 9];
        let proj_a_values = [100_i32, 200, 300, 400, 500, 600];
        let proj_b_values = [101_i32, 201, 301, 401, 501, 601];

        let mut header = Vec::new();
        header.extend_from_slice(&row_count.to_le_bytes());
        let mut filter = Vec::new();
        for value in filter_values {
            filter.extend_from_slice(&value.to_le_bytes());
        }
        let mut proj_a = Vec::new();
        for value in proj_a_values {
            proj_a.extend_from_slice(&value.to_le_bytes());
        }
        let mut proj_b = Vec::new();
        for value in proj_b_values {
            proj_b.extend_from_slice(&value.to_le_bytes());
        }
        let allocated_len = proj_b_offset + proj_b.len() as u64;

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
                            byte_offset: proj_a_offset,
                            bytes: &proj_a,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: proj_b_offset,
                            bytes: &proj_b,
                        },
                    ],
                )
                .expect("retain resident device memory"),
        );

        // filter == 5 → rows 0,2,4 → projecting [A,B] = [100,101],[300,301],[500,501].
        let filters: &[(u64, i32)] = &[(filter_offset, 5)];
        let projections: &[u64] = &[proj_a_offset, proj_b_offset];
        let expected = sorted(vec![vec![100, 101], vec![300, 301], vec![500, 501]]);

        // (a) Single-threaded parity (order-independent).
        let got = resident
            .match_project_i32_equal_from_payload(filters, projections, row_count)
            .expect("match_project_i32_equal");
        assert_eq!(sorted(got), expected, "equal_project single-thread parity");

        // A needle that matches nothing returns an empty result (kernel ran, count stayed 0).
        assert!(resident
            .match_project_i32_equal_from_payload(&[(filter_offset, -1)], projections, row_count)
            .expect("no-match equal_project")
            .is_empty());

        // Empty input short-circuits to an empty vector (no device work).
        assert!(resident
            .match_project_i32_equal_from_payload(filters, projections, 0)
            .expect("empty equal_project")
            .is_empty());

        // (b) Concurrent pool-reuse storm: every thread runs the route in a loop on the shared
        // allocation (shared pools), asserting the exact expected rows each time.
        const THREADS: usize = 8;
        const ITERS: usize = 300;
        let barrier = Arc::new(Barrier::new(THREADS));
        let mut handles = Vec::new();
        for _ in 0..THREADS {
            let resident = Arc::clone(&resident);
            let barrier = Arc::clone(&barrier);
            let expected = expected.clone();
            let filters = filters.to_vec();
            let projections = projections.to_vec();
            handles.push(std::thread::spawn(move || {
                resident
                    .set_current_context()
                    .expect("bind primary context on reader thread");
                barrier.wait();
                for _ in 0..ITERS {
                    let got = resident
                        .match_project_i32_equal_from_payload(&filters, &projections, row_count)
                        .expect("concurrent match_project_i32_equal");
                    assert_eq!(
                        sorted(got),
                        expected,
                        "concurrent equal_project returned wrong rows — a pooled buffer/stream was \
                         reused before its kernel drained, or the helper's success sync regressed?"
                    );
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
            .match_project_i32_equal_from_payload(filters, projections, row_count)
            .expect("final match_project_i32_equal");
        assert_eq!(
            sorted(final_rows),
            expected,
            "pools left unsound after reuse"
        );
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
                .project_i32_compare_from_payload(
                    col_offset,
                    ROW_COUNT,
                    NEEDLE,
                    CudaI32Comparison::Gt,
                )
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
            .project_i32_compare_from_payload(
                col_offset,
                ROW_COUNT,
                2_000_000,
                CudaI32Comparison::Gt,
            )
            .expect("zero-match project_i32_compare");
        assert!(none.is_empty(), "Gt 2_000_000 must match nothing");
        // (b) all matches: Gte i32::MIN matches every row, ascending by row == the raw column.
        let all = resident
            .project_i32_compare_from_payload(
                col_offset,
                ROW_COUNT,
                i32::MIN,
                CudaI32Comparison::Gte,
            )
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
        // `(a <op> b) <cmp> k` — runs fully on the GPU by COMPOSING two buffer->buffer primitives
        // (elementwise into an intermediate device buffer, then compare->matching-row-indices). This
        // is the vectorized-interpreter model that replaces hand-coded per-shape kernels.
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
        // byte blob, where row i = blob[offsets[i]..offsets[i+1]]. The offsets are placed at a 4-mod-8
        // byte offset to exercise the 2x 4-byte-load path (an 8-byte load there faults 716, sticky).
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        let rows: [&str; 6] = ["apple", "banana", "apple", "cherry", "banana", "apple"];
        const N: u64 = 6;
        let offsets_off: u64 = 12; // 4-mod-8 alignment: stress the 4-byte offset loads
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
            .expr_text_eq_scalar_filter(offsets_off, bytes_off, b"apple", false, N)
            .expect("text = apple");
        assert_eq!(eq, vec![0, 2, 5], "text = 'apple' => rows 0,2,5");

        let ne = resident
            .expr_text_eq_scalar_filter(offsets_off, bytes_off, b"apple", true, N)
            .expect("text <> apple");
        assert_eq!(ne, vec![1, 3, 4], "text <> 'apple' => rows 1,3,4");

        let banana = resident
            .expr_text_eq_scalar_filter(offsets_off, bytes_off, b"banana", false, N)
            .expect("text = banana");
        assert_eq!(banana, vec![1, 4], "text = 'banana' => rows 1,4");

        let none = resident
            .expr_text_eq_scalar_filter(offsets_off, bytes_off, b"grape", false, N)
            .expect("text = grape");
        assert!(none.is_empty(), "text = 'grape' matches nothing");

        // length mismatches are NOT equal (equality is full-string, not prefix/contains)
        let prefix = resident
            .expr_text_eq_scalar_filter(offsets_off, bytes_off, b"app", false, N)
            .expect("text = app");
        assert!(prefix.is_empty(), "text = 'app' (shorter) matches nothing");
        let longer = resident
            .expr_text_eq_scalar_filter(offsets_off, bytes_off, b"apples", false, N)
            .expect("text = apples");
        assert!(
            longer.is_empty(),
            "text = 'apples' (longer) matches nothing"
        );
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
        let offsets_off: u64 = 12; // 4-mod-8
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
                .expr_text_like_scalar_filter(offsets_off, bytes_off, &tokens, n)
                .unwrap_or_else(|e| panic!("LIKE '{pattern}': {e:?}"));
            let expected: Vec<u32> = rows
                .iter()
                .enumerate()
                .filter(|(_, r)| like_match(r.as_bytes(), pattern.as_bytes()))
                .map(|(i, _)| i as u32)
                .collect();
            assert_eq!(got, expected, "LIKE '{pattern}' mismatch vs oracle");
        }
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
    fn cuda_resident_i32_equal_row_indices_matches_expected_under_concurrent_pool_reuse() {
        // P2-M2 regression for the `equal_row_indices` migration to the pooled-async substrate.
        // This gather route is MORE exposed than its `compare_project` twin: it drives TWO device
        // buffers (the indices buffer + the atomic-append count buffer) with a count→indices data
        // dependency (sync #1 reads the device-computed count to size the indices D2H; sync #2
        // covers the indices D2H), and — unlike the serial single-thread compare/range kernels —
        // its kernel is PARALLEL (one thread per row + an `atom.global.add` ordered append). The
        // migration leases the indices/count buffers + the private stream + the cached module from
        // the SHARED pools while mid-flight, so the new hazard is concurrent contention on those
        // shared buckets: a buffer (or stream) re-leased before its async D2H drained would surface
        // as a wrong count, garbage/out-of-range indices, or a duplicated/missing index. Single-
        // threaded parity cannot exercise that shared-pool reuse, so this test (a) pins byte-exact
        // ordered parity against known results for the 0-row / all-rows / partial / multi-filter
        // cases, then (b) hammers the route on N threads over ONE shared resident allocation (hence
        // the same shared pools) asserting the EXACT ordered result on every call — a reuse bug
        // surfaces deterministically as a mismatch or a crash.
        //
        // Determinism of the EXACT order: the kernel appends via `atom.global.add`, so the physical
        // order in the output buffer is the atomic SCHEDULE order, which is ascending-by-row only
        // while every matching row lives in a single warp. With `threads_per_block = 128` a payload
        // of <= 32 rows launches exactly one warp, so the append order is deterministically
        // ascending row order (empirically 200/200 stable at <= 32 rows; it stops being stable the
        // moment a second warp races the atomic). We therefore keep the payload tiny (5 rows, like
        // the twin), which lets us assert the exact ordered vector without flakiness.
        use std::sync::{Arc, Barrier};

        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        // Known payload: header(row_count) + two i32 columns (5 rows => single warp).
        //   col A (constant 5)   @ a_offset : [5, 5, 5, 5, 5]
        //   col B                @ b_offset : [10, 20, 10, 20, 30]
        let row_count = 5_u64;
        let a_offset = std::mem::size_of::<u64>() as u64;
        let b_offset = a_offset + row_count * std::mem::size_of::<i32>() as u64;
        let column_a = [5_i32, 5, 5, 5, 5];
        let column_b = [10_i32, 20, 10, 20, 30];
        let mut header = Vec::new();
        header.extend_from_slice(&row_count.to_le_bytes());
        let mut bytes_a = Vec::new();
        for value in column_a {
            bytes_a.extend_from_slice(&value.to_le_bytes());
        }
        let mut bytes_b = Vec::new();
        for value in column_b {
            bytes_b.extend_from_slice(&value.to_le_bytes());
        }
        let allocated_len = b_offset + bytes_b.len() as u64;
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
                            byte_offset: a_offset,
                            bytes: &bytes_a,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: b_offset,
                            bytes: &bytes_b,
                        },
                    ],
                )
                .expect("retain resident device memory"),
        );

        // Each case is (filters, expected ordered matching row indices). The matching rows in every
        // case fit in the single warp, so the atomic-append order is ascending row order.
        //   all rows : A == 5            -> [0, 1, 2, 3, 4]   (edge: matches ALL rows)
        //   no rows  : A == 9            -> []                (edge: matches 0 rows)
        //   partial  : B == 10           -> [0, 2]            (single filter, normal partial)
        //   AND      : A == 5 AND B == 20-> [1, 3]            (two device-read columns + count dep)
        type Case = (Vec<(u64, i32)>, Vec<u64>);
        let cases: &[Case] = &[
            (vec![(a_offset, 5)], vec![0, 1, 2, 3, 4]),
            (vec![(a_offset, 9)], vec![]),
            (vec![(b_offset, 10)], vec![0, 2]),
            (vec![(a_offset, 5), (b_offset, 20)], vec![1, 3]),
        ];

        // (a) Single-threaded parity: each filter set returns exactly the expected indices.
        for (filters, expected) in cases {
            let got = resident
                .match_i32_equal_row_indices_from_payload(filters, row_count)
                .expect("match_i32_equal_row_indices");
            assert_eq!(
                &got, expected,
                "row_indices parity failed for filters {filters:?}"
            );
        }

        // Empty input short-circuits to an empty vector (no device work).
        assert!(resident
            .match_i32_equal_row_indices_from_payload(&[(a_offset, 5)], 0)
            .expect("empty match_i32_equal_row_indices")
            .is_empty());

        // (b) Concurrent pool-reuse storm: every thread runs all cases in a loop on the shared
        // allocation (shared device/pinned/stream pools + shared module cache), asserting the exact
        // ordered vector each time. The 0-row and all-rows edges exercise the count→indices
        // dependency at both extremes (a zero-length and a full-length indices D2H).
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
                    for (filters, expected) in &cases {
                        let got = resident
                            .match_i32_equal_row_indices_from_payload(filters, row_count)
                            .expect("concurrent match_i32_equal_row_indices");
                        assert_eq!(
                            &got, expected,
                            "concurrent row_indices returned wrong indices for filters {filters:?} \
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
            .match_i32_equal_row_indices_from_payload(&[(a_offset, 5), (b_offset, 20)], row_count)
            .expect("final match_i32_equal_row_indices");
        assert_eq!(final_rows, vec![1, 3]);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_resident_i32_equal_row_indices_multi_warp_returns_ascending_indices() {
        // Coverage gap closer for the host-sort fix. The kernel appends matching row indices via
        // `atom.global.add`, so the physical order in the output buffer is the atomic SCHEDULE
        // order — ascending-by-row ONLY while every matching row lives in a single warp (<= 32
        // matches; with threads_per_block=128 that is one warp). The moment a SECOND warp races the
        // counter, the append order across warps is non-deterministic and is NOT ascending row
        // order. This route's gather is strictly positional, so without a host-side sort the
        // returned Vec<u64> would (a) come back in a non-deterministic, non-ascending order and (b)
        // diverge from the CPU/non-resident reference (always ascending) and break the engine's
        // partitioned ascending-merge.
        //
        // This test deliberately uses a MULTI-WARP payload: ~150 matching rows (>> 32, spanning at
        // least 5 warps within a 128-thread block and several blocks overall) that are INTERLEAVED
        // with non-matching rows across the whole row range, so the matches are spread over many
        // warps that race the atomic in parallel. We then assert the EXACT ascending index vector.
        //
        // Why this is non-vacuous (would fail/flake WITHOUT the sort): with >32 interleaved matches
        // the raw atomic-append order is non-deterministic across warps and is essentially never the
        // ascending order we assert here; the equality below would fail (often flakily). It passes
        // only because the fix sorts the [0, count) prefix host-side before returning. (Empirically
        // the single-warp twin above is stable at <= 32; this payload is far past that boundary.)
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        // One i32 column. Even rows hold the needle (7), odd rows hold a non-match (-1), so the
        // matching rows are EXACTLY the even indices [0, 2, 4, ...] — interleaved, not contiguous,
        // and spread across the whole range so many warps contribute matches. 300 rows => 150
        // matches (well past the 32-per-warp single-warp boundary, across multiple blocks).
        const ROW_COUNT: u64 = 300;
        const NEEDLE: i32 = 7;
        let col_offset = std::mem::size_of::<u64>() as u64;
        let column: Vec<i32> = (0..ROW_COUNT)
            .map(|row| if row % 2 == 0 { NEEDLE } else { -1 })
            .collect();
        let expected: Vec<u64> = (0..ROW_COUNT).filter(|row| row % 2 == 0).collect();
        assert!(
            expected.len() > 32,
            "test must use a multi-warp match count to exercise the cross-warp append order"
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

        // Run it repeatedly: a sort-less route would surface a non-ascending permutation on at least
        // one of these iterations (the cross-warp schedule varies run to run); the sorted route is
        // exactly ascending every time.
        for iter in 0..50 {
            let got = resident
                .match_i32_equal_row_indices_from_payload(&[(col_offset, NEEDLE)], ROW_COUNT)
                .expect("multi-warp match_i32_equal_row_indices");
            assert_eq!(
                got, expected,
                "multi-warp row_indices were not ascending on iteration {iter} \
                 — the host sort over [0, count) is missing or ineffective?"
            );
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_resident_i32_equal_project_multi_warp_returns_ascending_row_order() {
        // Coverage gap closer for the Thread-3 Stage-4 stable-order fix on the FUSED single-needle
        // multi-column projection kernel (`gpu_db_resident_i32_equal_project`). Like the
        // `row_indices` kernel, it appends each match via `atom.global.add`, so the physical order
        // in the output buffer is the atomic SCHEDULE order — ascending-by-row ONLY within a single
        // warp (<= 32 matches with threads_per_block=128). Past that the cross-warp append order is
        // non-deterministic. The fix tags each match with its `row_index` and sorts the rows
        // ascending host-side; this asserts the EXACT ascending-by-row projection for a multi-warp
        // payload, byte-identical to the CPU/non-resident reference and to the batched `equal_any`
        // path.
        //
        // Why NON-VACUOUS (would fail/flake WITHOUT the sort): the projected VALUES are a by-row
        // SCRAMBLED sequence (a hash of the row index), so the correct ascending-by-row output is
        // deliberately NOT sorted-by-value. An atomic-append (sort-less) impl emits matches in
        // schedule order — non-deterministic across the racing warps, essentially never this exact
        // by-row sequence; a "sort the values" shortcut would emit them value-sorted, which this
        // sequence is not. Only an ascending-by-`row_index` sort reproduces the asserted vector. We
        // assert the expected sequence is not already value-sorted and run the route 50× (a
        // schedule-ordered impl flakes across iterations).
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        // Two i32 columns: a FILTER column (needle 7 on even rows, -1 on odd) and a PROJECTION
        // column holding a positive by-row hash (scrambled vs row order). 300 rows => 150 matches
        // (>> 32, across multiple warps/blocks).
        const ROW_COUNT: u64 = 300;
        const NEEDLE: i32 = 7;
        let filter_offset = std::mem::size_of::<u64>() as u64;
        let proj_offset = filter_offset + ROW_COUNT * std::mem::size_of::<i32>() as u64;
        let row_value = |row: u64| -> i32 {
            let h = row.wrapping_mul(2_654_435_761) ^ (row << 13) ^ 0x9E37_79B9;
            (1 + (h % 1_000_000)) as i32
        };
        let filter_col: Vec<i32> = (0..ROW_COUNT)
            .map(|row| if row % 2 == 0 { NEEDLE } else { -1 })
            .collect();
        let proj_col: Vec<i32> = (0..ROW_COUNT).map(row_value).collect();
        // Reference: matching rows (even) in ASCENDING ROW ORDER, projecting [filter, proj].
        let expected: Vec<Vec<i32>> = (0..ROW_COUNT)
            .filter(|row| row % 2 == 0)
            .map(|row| vec![NEEDLE, row_value(row)])
            .collect();
        assert!(
            expected.len() > 32,
            "test must use a multi-warp match count to exercise the cross-warp append order"
        );
        // The projected (second-column) by-row sequence must NOT already be value-sorted, or a
        // "sort the values" impl would pass vacuously.
        let proj_by_row: Vec<i32> = expected.iter().map(|row| row[1]).collect();
        let mut proj_sorted = proj_by_row.clone();
        proj_sorted.sort_unstable();
        assert_ne!(
            proj_by_row, proj_sorted,
            "projected by-row sequence is accidentally value-sorted — pick a payload that isn't"
        );

        let mut header = Vec::new();
        header.extend_from_slice(&ROW_COUNT.to_le_bytes());
        let mut filter_bytes = Vec::new();
        for value in &filter_col {
            filter_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut proj_bytes = Vec::new();
        for value in &proj_col {
            proj_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let allocated_len = proj_offset + proj_bytes.len() as u64;
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
                        bytes: &filter_bytes,
                    },
                    CudaDeviceMemoryChunk {
                        byte_offset: proj_offset,
                        bytes: &proj_bytes,
                    },
                ],
            )
            .expect("retain resident device memory");

        for iter in 0..50 {
            let got = resident
                .match_project_i32_equal_from_payload(
                    &[(filter_offset, NEEDLE)],
                    &[filter_offset, proj_offset],
                    ROW_COUNT,
                )
                .expect("multi-warp match_project_i32_equal");
            assert_eq!(
                got, expected,
                "multi-warp equal_project rows were not in ascending ROW order on iteration {iter} \
                 — the host sort by tagged row_index is missing or ineffective?"
            );
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn published_resident_generation_survives_a_replacement_publish_and_is_freed_after_drain() {
        // P1-M3 step 1 — the real-GPU soundness probe (doc 14 acceptance gate 1).
        //
        // Retires the device-memory-lifetime risk the P1-M2 spike could only model with
        // a leaked-static buffer. With a REAL CudaResidentDeviceMemory whose Drop calls
        // the REAL cu_mem_free, it proves that under the SnapshotCell publish-on-commit
        // model a generation a reader still holds is:
        //   (a) NOT freed when the writer publishes a replacement, and still GPU-valid
        //       (a kernel read of it returns the correct rows), and
        //   (b) freed only after that last reader drains.
        // The probe cannot even compile unless `CudaResidentDeviceMemory: Send + Sync`
        // (the cell must cross the thread boundary), so it also witnesses that change.
        use gpu_db_snapshot::SnapshotCell;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Barrier};

        // Owner wrapper whose Drop sets `freed` as it is entered — immediately before
        // the inner CudaResidentDeviceMemory field drops (field-declaration order) and
        // calls the REAL cu_mem_free / cu_ctx_destroy on the same synchronous drop path.
        // So `freed == true` means that device free has entered and is about to run —
        // the Drop-observing wrapper doc 14 gate 1 sanctions. For "not freed while held"
        // this is conservative; for "freed after drain" the free is the next,
        // unconditional statements once Drop is entered.
        struct ObservableResident {
            resident: CudaResidentDeviceMemory,
            freed: Arc<AtomicBool>,
        }
        impl Drop for ObservableResident {
            fn drop(&mut self) {
                self.freed.store(true, Ordering::SeqCst);
            }
        }

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

        let build = |freed: &Arc<AtomicBool>| ObservableResident {
            resident: runtime
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
            freed: Arc::clone(freed),
        };

        // needles [2,4] over filter [1,2,3,2,4] → rows 1,3 (=2) and 4 (=4),
        // projecting [20], [21], [40].
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

        let freed_g1 = Arc::new(AtomicBool::new(false));
        // `SnapshotCell<Arc<owner>>` mirrors doc 14's prescribed engine residency shape
        // (the cell wraps each generation in its own `Arc`, so the inner `Arc<owner>` is
        // redundant *here* but is the type the engine application in step 2 adopts).
        let cell = Arc::new(SnapshotCell::new(Arc::new(build(&freed_g1))));

        let barrier = Arc::new(Barrier::new(2));
        let reader = {
            let cell = Arc::clone(&cell);
            let barrier = Arc::clone(&barrier);
            let expected = expected.clone();
            std::thread::spawn(move || {
                let handle = cell.load(); // pins g1 for the whole closure
                assert_eq!(handle.generation(), 1, "reader did not pin g1");
                barrier.wait(); // (1) signal: g1 is pinned
                barrier.wait(); // (2) resume only after the writer published g2

                // g2 is now current, but we still hold g1. A real GPU read of g1 must
                // still return the correct rows — proof its device memory was not freed
                // by the publish. read_view() is derived from the held owner, so the
                // owner remains the lifetime anchor; submit + complete_detached each set
                // the context current on this reader thread.
                let (rows, elapsed_us) = handle
                    .get()
                    .resident
                    .read_view()
                    .submit_match_project_i32_equal_any_from_payload(
                        filter_offset,
                        &[2, 4],
                        &[projection_offset],
                        row_count,
                    )
                    .expect("submit on pinned g1")
                    .complete_detached()
                    .expect("complete on pinned g1");
                assert_eq!(
                    rows, expected,
                    "pinned g1 returned wrong rows — freed early?"
                );
                assert!(elapsed_us.is_some(), "no CUDA-event timing from pinned g1");
                // handle drops here → releases the last reference to g1
            })
        };

        barrier.wait(); // (1) g1 is pinned by the reader
        let freed_g2 = Arc::new(AtomicBool::new(false));
        cell.publish(Arc::new(build(&freed_g2))); // writer publishes a replacement
        assert_eq!(cell.current_generation(), 2, "g2 was not published");
        assert!(
            !freed_g1.load(Ordering::SeqCst),
            "g1 was freed while a reader still held it (use-after-free risk)"
        );
        barrier.wait(); // (2) let the reader do its GPU read of g1

        reader.join().expect("reader thread panicked");
        // The reader drained → its handle (the last reference to g1) dropped, and the
        // cell holds g2, not g1. So g1 must now be reclaimed: the real cu_mem_free ran.
        assert!(
            freed_g1.load(Ordering::SeqCst),
            "g1 was not freed after its last reader drained (leak / reclamation broken)"
        );
        // g2 is still current (held by the cell), so it must still be alive.
        assert!(
            !freed_g2.load(Ordering::SeqCst),
            "current generation g2 was freed early"
        );
    }
