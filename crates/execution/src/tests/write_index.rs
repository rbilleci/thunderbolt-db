    /// M1 (ledger #24): the INCREMENTAL index-insert kernel == a full rebuild. Build an index for
    /// a prefix of keys, INSERT the appended tail via the kernel, and verify the extended index
    /// probes IDENTICALLY to a from-scratch build over all keys (via the write-locate kernel).
    /// Also verifies DUP detection (re-inserting an existing key sets the decline flag).
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
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let Ok(mem) = runtime.retain_device_memory_copy(0, &bytes) else {
            return;
        };
        let index = std::sync::Arc::new(mem);
        // INSERT the tail (rows 6..10) via the kernel.
        let tail = &all[6..];
        let dup = index
            .submit_i32_index_insert(&index, table_mask, hash_shift, tail, 6)
            .expect("index insert");
        assert!(!dup, "no dup inserting fresh keys");
        // Probe ALL keys via the write-locate kernel: each must resolve to its row.
        let shards = [WriteLocateShard {
            index: std::sync::Arc::clone(&index),
            table_mask,
            hash_shift,
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
        // DUP: re-inserting an existing key sets the decline flag.
        let dup2 = index
            .submit_i32_index_insert(&index, table_mask, hash_shift, &[30], 99)
            .expect("dup insert");
        assert!(dup2, "re-inserting key 30 must flag a dup");
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
            let bytes: Vec<u8> = index_words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let Ok(mem) = runtime.retain_device_memory_copy(0, &bytes) else {
                return;
            };
            shards.push(WriteLocateShard {
                index: std::sync::Arc::new(mem),
                table_mask,
                hash_shift,
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
            let bytes: Vec<u8> = index_words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let Ok(mem) = runtime.retain_device_memory_copy(0, &bytes) else {
                return; // no device -> skip
            };
            shards.push(WriteLocateShard {
                index: std::sync::Arc::new(mem),
                table_mask,
                hash_shift,
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

