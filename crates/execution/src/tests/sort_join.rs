    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_bitonic_sort_i64_orders_and_permutes() {
        let runtime = CudaDriverRuntime::probe().expect("probe");
        // The sort never reads the residency; a tiny dummy payload just supplies the shared context.
        let resident = runtime
            .retain_device_memory_copy(0, &0_u64.to_le_bytes())
            .expect("resident device memory");

        // Verify the GPU permutation is a bijection of 0..n AND orders the keys monotonically (a sort
        // is correct iff it produces such a permutation -- bitonic is unstable, so we do NOT pin the
        // exact perm on ties).
        let check = |keys: &[i64], descending: bool| {
            let perm = resident.bitonic_sort_i64(keys, descending).expect("sort");
            assert_eq!(perm.len(), keys.len(), "one position per row");
            let mut seen: Vec<u32> = perm.clone();
            seen.sort_unstable();
            assert_eq!(
                seen,
                (0..keys.len() as u32).collect::<Vec<_>>(),
                "perm must be a permutation of 0..n"
            );
            for w in perm.windows(2) {
                let (a, b) = (keys[w[0] as usize], keys[w[1] as usize]);
                if descending {
                    assert!(a >= b, "descending order violated: {a} before {b}");
                } else {
                    assert!(a <= b, "ascending order violated: {a} before {b}");
                }
            }
        };

        // (a) padding (n=6 -> npot=8).
        check(&[5, 2, 8, 1, 9, 3], false);
        check(&[5, 2, 8, 1, 9, 3], true);
        // (b) negatives + i64::MIN/MAX, non-power-of-2 n=9, with duplicates of the extremes.
        let extremes = [0, -1, i64::MAX, i64::MIN, 7, -7, 100, i64::MIN, i64::MAX];
        check(&extremes, false);
        check(&extremes, true);
        // (c) all-equal keys -- any permutation is valid (the monotonic check is vacuously satisfied;
        // the bijection check still bites).
        check(&[42_i64; 5], false);
        // (d) 1000 keys via a deterministic xorshift (no rand dep), ASC + DESC.
        let mut x = 0x2545_F491_4F6C_DD1D_u64;
        let big: Vec<i64> = (0..1000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as i64
            })
            .collect();
        check(&big, false);
        check(&big, true);
        // n<=1 is the already-sorted fast path.
        assert_eq!(
            resident.bitonic_sort_i64(&[], false).unwrap(),
            Vec::<u32>::new()
        );
        assert_eq!(resident.bitonic_sort_i64(&[7], true).unwrap(), vec![0]);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_order_by_sort_i64_dispatches_radix_and_matches_bitonic() {
        // order_by_sort_i64 dispatches BITONIC below ADAPTIVE_SORT_CROSSOVER_ROWS (10_000) and RADIX
        // at/above it. Across the crossover (and at the boundary +/-1) it must produce a CORRECT sort:
        // a valid permutation of 0..n with monotonic keys. For UNIQUE keys we additionally cross-check
        // byte-equality vs the proven bitonic_sort_i64 (identical perm); for duplicates both arms are
        // valid sorts so we assert monotonic keys, not the exact perm. Covers signs, i64::MIN/MAX, dups.
        let runtime = CudaDriverRuntime::probe().expect("probe");
        let resident = runtime
            .retain_device_memory_copy(0, &0_u64.to_le_bytes())
            .expect("resident device memory");
        let assert_sorted = |keys: &[i64], perm: &[u32], descending: bool, label: &str| {
            let n = keys.len();
            assert_eq!(perm.len(), n, "{label}: perm length");
            let mut bij = perm.to_vec();
            bij.sort_unstable();
            assert!(
                bij.iter().copied().eq(0..n as u32),
                "{label}: valid permutation 0..n"
            );
            for w in perm.windows(2) {
                let (a, b) = (keys[w[0] as usize], keys[w[1] as usize]);
                if descending {
                    assert!(a >= b, "{label}: descending monotonic ({a} >= {b})");
                } else {
                    assert!(a <= b, "{label}: ascending monotonic ({a} <= {b})");
                }
            }
        };
        for &n in &[100usize, 9_999, 10_000, 10_001, 50_000] {
            // Dense duplicates (value range << n at large n) + planted extremes stress the signed
            // transform + tie handling.
            let mut dup: Vec<i64> = (0..n)
                .map(|i| {
                    ((((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) % 8_000) as i64) - 4_000
                })
                .collect();
            if n >= 4 {
                dup[0] = i64::MIN;
                dup[1] = i64::MAX;
                dup[2] = -1;
                dup[3] = 0;
            }
            for &desc in &[false, true] {
                let perm = resident
                    .order_by_sort_i64(&dup, desc)
                    .expect("order_by_sort dup");
                assert_sorted(&dup, &perm, desc, &format!("dup n={n} desc={desc}"));
            }
            // UNIQUE keys (distinct, signed, shuffled): the radix/bitonic dispatch must equal bitonic.
            let mut uniq: Vec<i64> = (0..n).map(|i| i as i64 - (n as i64) / 2).collect();
            for i in (1..n).rev() {
                let j = (((i as u64).wrapping_mul(2_654_435_761)) % (i as u64 + 1)) as usize;
                uniq.swap(i, j);
            }
            for &desc in &[false, true] {
                let dispatched = resident
                    .order_by_sort_i64(&uniq, desc)
                    .expect("order_by_sort uniq");
                let bitonic = resident
                    .bitonic_sort_i64(&uniq, desc)
                    .expect("bitonic uniq");
                assert_eq!(
                    dispatched, bitonic,
                    "unique-key n={n} desc={desc}: order_by_sort_i64 (radix>=10k) must equal bitonic"
                );
                assert_sorted(&uniq, &dispatched, desc, &format!("uniq n={n} desc={desc}"));
            }
        }
    }

    #[test]
    #[ignore = "benchmark; requires a local NVIDIA driver and GPU"]
    fn bench_order_by_sort_i64_radix_vs_bitonic() {
        // Confirms the ORDER BY single-i64-key radix fast-path beats bitonic above the 10k crossover.
        // order_by_sort_i64 = bitonic <10k / radix >=10k; bitonic_sort_i64 = always bitonic. Both upload
        // the keys, so the delta is the algorithm. Run with --nocapture to see the numbers.
        let runtime = CudaDriverRuntime::probe().expect("probe");
        let resident = runtime
            .retain_device_memory_copy(0, &0_u64.to_le_bytes())
            .expect("resident device memory");
        for &n in &[1_000usize, 10_000, 100_000, 1_000_000] {
            let keys: Vec<i64> = (0..n)
                .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) as i64)
                .collect();
            let _ = resident.order_by_sort_i64(&keys, false).unwrap(); // warm
            let _ = resident.bitonic_sort_i64(&keys, false).unwrap();
            let runs = 5u32;
            let t0 = std::time::Instant::now();
            for _ in 0..runs {
                let _ = resident.order_by_sort_i64(&keys, false).unwrap();
            }
            let adaptive_us = t0.elapsed().as_micros() / u128::from(runs);
            let t1 = std::time::Instant::now();
            for _ in 0..runs {
                let _ = resident.bitonic_sort_i64(&keys, false).unwrap();
            }
            let bitonic_us = t1.elapsed().as_micros() / u128::from(runs);
            let arm = if n >= 10_000 { "radix" } else { "bitonic" };
            println!(
                "n={n:>9}  order_by({arm})={adaptive_us:>7}us  bitonic={bitonic_us:>7}us  \
                 speedup={:.2}x",
                bitonic_us as f64 / adaptive_us.max(1) as f64
            );
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_bitonic_sort_multikey_orders_and_permutes() {
        let runtime = CudaDriverRuntime::probe().expect("probe");
        let resident = runtime
            .retain_device_memory_copy(0, &0_u64.to_le_bytes())
            .expect("resident device memory");

        // A sort is correct iff its output permutation is (1) a bijection of 0..n and (2) orders the
        // rows monotonically under the multi-key comparator (key 0 most significant, per-key direction
        // from desc_mask). We check that INVARIANT on adjacent pairs -- we do NOT recompute the expected
        // permutation on the CPU (bitonic is unstable, so ties are not pinned). `cmp` is the order
        // DEFINITION, not a re-implementation of the sort.
        let cmp = |keys: &[i64], k: usize, mask: u64, a: usize, b: usize| -> std::cmp::Ordering {
            for kk in 0..k {
                let ka = keys[a * k + kk];
                let kb = keys[b * k + kk];
                if ka != kb {
                    let asc = ka.cmp(&kb);
                    return if (mask >> kk) & 1 == 1 {
                        asc.reverse()
                    } else {
                        asc
                    };
                }
            }
            std::cmp::Ordering::Equal
        };
        let check = |keys: &[i64], n: usize, k: usize, mask: u64| {
            let perm = resident
                .bitonic_sort_multikey(keys, n, k, mask)
                .expect("sort");
            assert_eq!(perm.len(), n, "one position per row");
            let mut seen = perm.clone();
            seen.sort_unstable();
            assert_eq!(
                seen,
                (0..n as u32).collect::<Vec<_>>(),
                "perm must be a permutation of 0..n"
            );
            for w in perm.windows(2) {
                assert_ne!(
                    cmp(keys, k, mask, w[0] as usize, w[1] as usize),
                    std::cmp::Ordering::Greater,
                    "multi-key order violated: row {} placed before row {}",
                    w[0],
                    w[1]
                );
            }
        };

        // (a) 2 keys, a ASC + b DESC, n=6 (-> npot=8 padding); a-ties broken by b. Duplicate (2,50) row
        // exercises full key equality.
        let ab: Vec<i64> = [(1, 10), (1, 30), (1, 20), (2, 50), (2, 40), (2, 50)]
            .iter()
            .flat_map(|&(a, b)| [a as i64, b as i64])
            .collect();
        check(&ab, 6, 2, 0b10);
        // (b) 3 keys a ASC, b DESC, c ASC -- the two (1,5,*) rows tie on a AND b, ordered only by c.
        let abc: Vec<i64> = [(1, 5, 100), (1, 5, 50), (1, 8, 10), (2, 3, 7), (2, 3, 9)]
            .iter()
            .flat_map(|&(a, b, c)| [a as i64, b as i64, c as i64])
            .collect();
        check(&abc, 5, 3, 0b010);
        // (c) all-equal first key -> the SECOND key fully determines order (asc). Negatives included.
        let eqfirst: Vec<i64> = [(7, 3), (7, -1), (7, 9), (7, i64::MIN), (7, 5)]
            .iter()
            .flat_map(|&(a, b)| [a as i64, b])
            .collect();
        check(&eqfirst, 5, 2, 0b00);
        // (d) 300 rows, 2 keys via xorshift: a low-cardinality first key (0..5, lots of ties) and a
        // full-range second key (the tie-breaker), non-power-of-2 -> padding. All 4 direction masks.
        let mut x = 0x2545_F491_4F6C_DD1D_u64;
        let mut big = Vec::with_capacity(300 * 2);
        for _ in 0..300 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            big.push((x % 5) as i64);
            big.push(x as i64);
        }
        check(&big, 300, 2, 0b00);
        check(&big, 300, 2, 0b10);
        check(&big, 300, 2, 0b01);
        check(&big, 300, 2, 0b11);
        // n<=1 fast path + the keys.len() == n*k validation.
        assert_eq!(
            resident.bitonic_sort_multikey(&[], 0, 3, 0).unwrap(),
            Vec::<u32>::new()
        );
        assert_eq!(
            resident.bitonic_sort_multikey(&[1, 2, 3], 1, 3, 0).unwrap(),
            vec![0]
        );
        assert!(
            resident.bitonic_sort_multikey(&[1, 2, 3], 2, 2, 0).is_err(),
            "keys.len() must equal n*k"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_hash_join_inner_i64_matches_unique_build_key() {
        let runtime = CudaDriverRuntime::probe().expect("probe");
        let resident = runtime
            .retain_device_memory_copy(0, &0_u64.to_le_bytes())
            .expect("resident device memory");
        // Sort the (build_idx, probe_idx) pairs for a stable comparison (the atomic append order is
        // race-dependent; the SET of pairs is the oracle).
        let sorted = |o: HashJoinOutcome| -> Vec<(u32, u32)> {
            match o {
                HashJoinOutcome::Pairs {
                    build_idxs,
                    probe_idxs,
                } => {
                    let mut v: Vec<(u32, u32)> = build_idxs.into_iter().zip(probe_idxs).collect();
                    v.sort_unstable();
                    v
                }
                HashJoinOutcome::DuplicateBuildKey => panic!("unexpected DuplicateBuildKey"),
            }
        };
        // (a) unique build keys, a mix of matches + a non-matching probe (40).
        //   probe 20->build1, probe 10->build0, probe 20->build1, probe 40->none.
        assert_eq!(
            sorted(
                resident
                    .hash_join_inner_i64(&[10, 20, 30], &[20, 10, 20, 40], None, None)
                    .unwrap()
            ),
            vec![(0, 1), (1, 0), (1, 2)],
            "unique build key, 1:1 + a repeated probe key + a miss"
        );
        // (b) 1:N fan-out: one build row, three matching probe rows.
        assert_eq!(
            sorted(
                resident
                    .hash_join_inner_i64(&[5], &[5, 5, 5], None, None)
                    .unwrap()
            ),
            vec![(0, 0), (0, 1), (0, 2)],
            "1:N fan-out (unique build, repeated probe)"
        );
        // (c) no matches at all.
        assert_eq!(
            sorted(
                resident
                    .hash_join_inner_i64(&[1, 2], &[3, 4], None, None)
                    .unwrap()
            ),
            Vec::<(u32, u32)>::new(),
            "disjoint key sets -> no pairs"
        );
        // (d) negatives + a build value beyond a small range; unique build, mixed probe.
        //   probe 100->build1, probe -5->build0, probe 7->none.
        assert_eq!(
            sorted(
                resident
                    .hash_join_inner_i64(&[-5, 100], &[100, -5, 7], None, None)
                    .unwrap()
            ),
            vec![(0, 1), (1, 0)],
            "negative + larger keys round-trip through the hash"
        );
        // (e) duplicate build key -> reject (N:N is a follow-up).
        assert_eq!(
            resident
                .hash_join_inner_i64(&[10, 10], &[10], None, None)
                .unwrap(),
            HashJoinOutcome::DuplicateBuildKey,
            "a repeated build key is rejected"
        );
        // (f) empty sides -> no matches.
        assert_eq!(
            sorted(
                resident
                    .hash_join_inner_i64(&[], &[1, 2], None, None)
                    .unwrap()
            ),
            Vec::<(u32, u32)>::new(),
            "empty build -> no pairs"
        );
        assert_eq!(
            sorted(
                resident
                    .hash_join_inner_i64(&[1, 2], &[], None, None)
                    .unwrap()
            ),
            Vec::<(u32, u32)>::new(),
            "empty probe -> no pairs"
        );
        // (g) a larger case stressing the PROBE-side collision walk: 200 unique build keys with heavy
        // hash collisions (many land far from their home bucket), probed by ALL 200 keys + some misses.
        // Probing every key forces walk>0 lookups, so a broken probe linear-probe step is caught.
        let build: Vec<i64> = (0..200).collect();
        let mut probe: Vec<i64> = (0..200).collect();
        probe.extend_from_slice(&[1000, -7]); // two misses
        let mut expected: Vec<(u32, u32)> = Vec::new();
        for (p, &k) in probe.iter().enumerate() {
            if (0..200).contains(&k) {
                expected.push((k as u32, p as u32));
            }
        }
        expected.sort_unstable();
        assert_eq!(
            sorted(
                resident
                    .hash_join_inner_i64(&build, &probe, None, None)
                    .unwrap()
            ),
            expected,
            "200 unique build keys probed by all 200 (forces collision walks) + 2 misses"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_payload_join_retains_nn_outer_coordinates_on_device() {
        let runtime = CudaDriverRuntime::probe().expect("probe");
        let encode = |keys: &[i32], validity: u32| {
            let mut bytes = keys
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            bytes.extend_from_slice(&validity.to_le_bytes());
            bytes
        };
        let left = runtime
            .retain_device_memory_copy(0, &encode(&[1, 2, 2, 0], 0b0111))
            .expect("left payload");
        let right = runtime
            .retain_device_memory_copy(0, &encode(&[2, 2, 3, 0], 0b0111))
            .expect("right payload");
        let left_key = CudaJoinPayloadKey {
            payload: &left,
            byte_offset: 0,
            validity_bitmap_offset: Some(16),
            width: 4,
            text_bytes_byte_offset: None,
            text_bytes_len: 0,
        };
        let right_key = CudaJoinPayloadKey {
            payload: &right,
            byte_offset: 0,
            validity_bitmap_offset: Some(16),
            width: 4,
            text_bytes_byte_offset: None,
            text_bytes_len: 0,
        };
        let coordinates = left
            .join_fixed_payload_coordinates(
                None,
                4,
                &[0],
                &[left_key],
                4,
                &[right_key],
                None,
                None,
                true,
                true,
            )
            .expect("full outer payload join");
        assert_eq!(coordinates.row_count(), 8);
        assert_eq!(coordinates.relation_count(), 2);
        let mut actual = coordinates.readback_for_test().expect("coordinate readback");
        actual.sort_unstable();
        let x = u32::MAX;
        let mut expected = vec![
            vec![1, 0],
            vec![1, 1],
            vec![2, 0],
            vec![2, 1],
            vec![0, x],
            vec![3, x],
            vec![x, 2],
            vec![x, 3],
        ];
        expected.sort_unstable();
        assert_eq!(actual, expected);

        let right_is_two = right
            .run_expr_predicate_mask_with_text(
                &[
                    ExprStep::LoadColumn { byte_offset: 0 },
                    ExprStep::CompareScalar {
                        cmp: 0,
                        scalar: 2,
                        scalar_on_left: false,
                    },
                ],
                &[],
                4,
                ResidentElemType::I32,
            )
            .expect("device predicate mask");
        let pad_false = left
            .row_range_mask_u32(1, 0, 0)
            .expect("false NULL-pad mask");
        let filtered = left
            .filter_join_coordinates(
                &coordinates,
                &[None, Some(&right_is_two)],
                &[Some(&pad_false), Some(&pad_false)],
            )
            .expect("device post-join filter");
        assert_eq!(filtered.row_count(), 4);
        let bitmap = left.create_match_bitmap_u32(4).expect("match bitmap");
        bitmap
            .mark_coordinates(&filtered, 1)
            .expect("device coordinate marking");
        assert_eq!(
            bitmap
                .unmatched_coordinates(2, 1)
                .expect("device complement")
                .readback_for_test()
                .expect("test-only coordinate readback"),
            vec![vec![u32::MAX, 2], vec![u32::MAX, 3]]
        );
        let (projected, valid) = left
            .project_fixed_from_join_coordinates(&coordinates, 0, &left, 0, Some(16), 4)
            .expect("final coordinate projection");
        let values: Vec<i32> = projected
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 8);
        assert_eq!(valid.iter().filter(|&&v| v).count(), 5);
        assert_eq!(
            values
                .iter()
                .zip(&valid)
                .filter_map(|(&value, &is_valid)| is_valid.then_some(value))
                .filter(|&value| value == 2)
                .count(),
            4
        );
        let sorted = left
            .sort_join_coordinates(
                &coordinates,
                &[CudaJoinOrderKey {
                    relation: 0,
                    key: left_key,
                    descending: false,
                    nulls_first: false,
                    lexicographic_16: false,
                }],
            )
            .expect("device coordinate sort");
        let (sorted_bytes, sorted_valid) = left
            .project_fixed_from_join_coordinates(&sorted, 0, &left, 0, Some(16), 4)
            .expect("sorted coordinate projection");
        let sorted_values: Vec<Option<i32>> = sorted_bytes
            .chunks_exact(4)
            .zip(sorted_valid)
            .map(|(bytes, valid)| {
                valid.then(|| i32::from_le_bytes(bytes.try_into().unwrap()))
            })
            .collect();
        assert_eq!(
            sorted_values,
            vec![Some(1), Some(2), Some(2), Some(2), Some(2), None, None, None]
        );
        let window = left
            .window_join_coordinates(&sorted, 1, Some(2))
            .expect("device coordinate window");
        assert_eq!(window.row_count(), 2);

        let rank_payload_bytes: Vec<u8> = [1_i32, 1, 1, 2, 10, 10, 20, 5]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let rank_payload = runtime
            .retain_device_memory_copy(0, &rank_payload_bytes)
            .expect("rank payload");
        let identity = rank_payload
            .identity_join_coordinates(4, None)
            .expect("identity coordinates");
        let part_key = CudaJoinOrderKey {
            relation: 0,
            key: CudaJoinPayloadKey {
                payload: &rank_payload,
                byte_offset: 0,
                validity_bitmap_offset: None,
                width: 4,
                text_bytes_byte_offset: None,
                text_bytes_len: 0,
            },
            descending: false,
            nulls_first: false,
            lexicographic_16: false,
        };
        let order_key = CudaJoinOrderKey {
            relation: 0,
            key: CudaJoinPayloadKey {
                payload: &rank_payload,
                byte_offset: 16,
                validity_bitmap_offset: None,
                width: 4,
                text_bytes_byte_offset: None,
                text_bytes_len: 0,
            },
            descending: false,
            nulls_first: false,
            lexicographic_16: false,
        };
        let rank_sorted = rank_payload
            .sort_join_coordinates(&identity, &[part_key, order_key])
            .expect("rank physical ordering");
        let ranks = rank_payload
            .window_ranks_from_join_coordinates(&rank_sorted, &[part_key], &[order_key])
            .expect("coordinate ranks");
        assert_eq!(
            ranks
                .readback(CudaWindowRankKind::RowNumber, 0, None)
                .expect("row number"),
            vec![1, 2, 3, 1]
        );
        assert_eq!(
            ranks
                .readback(CudaWindowRankKind::Rank, 0, None)
                .expect("rank"),
            vec![1, 1, 3, 1]
        );
        assert_eq!(
            ranks
                .readback(CudaWindowRankKind::DenseRank, 0, None)
                .expect("dense rank"),
            vec![1, 1, 2, 1]
        );

        // A later left-deep step consumes relation-1 coordinates in place; no pair vector is exposed.
        let third = runtime
            .retain_device_memory_copy(0, &encode(&[2, 9], 0b11))
            .expect("third payload");
        let third_key = CudaJoinPayloadKey {
            payload: &third,
            byte_offset: 0,
            validity_bitmap_offset: Some(8),
            width: 4,
            text_bytes_byte_offset: None,
            text_bytes_len: 0,
        };
        let extended = left
            .join_fixed_payload_coordinates(
                Some(&coordinates),
                0,
                &[1],
                &[right_key],
                2,
                &[third_key],
                None,
                None,
                false,
                false,
            )
            .expect("three-way coordinate extension");
        assert_eq!(extended.row_count(), 4);
        assert!(extended
            .readback_for_test()
            .expect("extended coordinate readback")
            .iter()
            .all(|row| row.len() == 3 && row[1] < 2 && row[2] == 0));

        let encode_text = |values: &[Option<&str>]| {
            let mut offsets = vec![0_u64];
            let mut data = Vec::new();
            let mut validity = 0_u32;
            for (row, value) in values.iter().enumerate() {
                if let Some(value) = value {
                    validity |= 1 << row;
                    data.extend_from_slice(value.as_bytes());
                }
                offsets.push(data.len() as u64);
            }
            let bytes_offset = offsets.len() as u64 * 8;
            let validity_offset = (bytes_offset + data.len() as u64).next_multiple_of(4);
            let mut payload = offsets
                .iter()
                .flat_map(|offset| offset.to_le_bytes())
                .collect::<Vec<_>>();
            payload.extend_from_slice(&data);
            payload.resize(validity_offset as usize, 0);
            payload.extend_from_slice(&validity.to_le_bytes());
            (payload, bytes_offset, data.len() as u64, validity_offset)
        };
        let (left_text_bytes, left_text_off, left_text_len, left_text_valid) =
            encode_text(&[Some("a"), Some("b"), Some("b"), None]);
        let (right_text_bytes, right_text_off, right_text_len, right_text_valid) =
            encode_text(&[Some("b"), Some("b"), Some("c"), None]);
        let left_text = runtime
            .retain_device_memory_copy(0, &left_text_bytes)
            .expect("left text payload");
        let right_text = runtime
            .retain_device_memory_copy(0, &right_text_bytes)
            .expect("right text payload");
        let text_coordinates = left_text
            .join_fixed_payload_coordinates(
                None,
                4,
                &[0],
                &[CudaJoinPayloadKey {
                    payload: &left_text,
                    byte_offset: 0,
                    validity_bitmap_offset: Some(left_text_valid),
                    width: 255,
                    text_bytes_byte_offset: Some(left_text_off),
                    text_bytes_len: left_text_len,
                }],
                4,
                &[CudaJoinPayloadKey {
                    payload: &right_text,
                    byte_offset: 0,
                    validity_bitmap_offset: Some(right_text_valid),
                    width: 255,
                    text_bytes_byte_offset: Some(right_text_off),
                    text_bytes_len: right_text_len,
                }],
                None,
                None,
                false,
                false,
            )
            .expect("text payload join");
        assert_eq!(text_coordinates.row_count(), 4);
        assert_eq!(
            left_text
                .project_text_from_join_coordinates(
                    &text_coordinates,
                    1,
                    &right_text,
                    0,
                    right_text_off,
                    right_text_len,
                    Some(right_text_valid),
                )
                .expect("text coordinate projection"),
            vec![
                Some("b".to_string()),
                Some("b".to_string()),
                Some("b".to_string()),
                Some("b".to_string()),
            ]
        );
        let materialized = left
            .materialize_join_coordinates(
                &text_coordinates,
                &[
                    CudaMaterializeJoinColumn::Text {
                        relation: 0,
                        payload: &left_text,
                        offsets_byte_offset: 0,
                        bytes_byte_offset: left_text_off,
                        bytes_len: left_text_len,
                        validity_bitmap_offset: Some(left_text_valid),
                    },
                    CudaMaterializeJoinColumn::Text {
                        relation: 1,
                        payload: &right_text,
                        offsets_byte_offset: 0,
                        bytes_byte_offset: right_text_off,
                        bytes_len: right_text_len,
                        validity_bitmap_offset: Some(right_text_valid),
                    },
                ],
            )
            .expect("D2D materialized relation");
        drop(text_coordinates);
        drop(left_text);
        drop(right_text);
        let run_identity = materialized
            .memory()
            .identity_join_coordinates(materialized.row_count(), None)
            .expect("materialized identity");
        let run_key = CudaJoinOrderKey {
            relation: 0,
            key: materialized.payload_key(1).expect("run text key"),
            descending: false,
            nulls_first: false,
            lexicographic_16: false,
        };
        let run_sorted = materialized
            .memory()
            .sort_join_coordinates(&run_identity, &[run_key])
            .expect("materialized sort");
        let run_layout = materialized.columns()[0];
        assert_eq!(
            materialized
                .memory()
                .project_text_from_join_coordinates(
                    &run_sorted,
                    0,
                    materialized.memory(),
                    run_layout.value_byte_offset,
                    run_layout.text_bytes_byte_offset.unwrap(),
                    run_layout.text_bytes_len,
                    Some(run_layout.validity_bitmap_offset),
                )
                .expect("materialized final projection"),
            vec![
                Some("b".to_string()),
                Some("b".to_string()),
                Some("b".to_string()),
                Some("b".to_string()),
            ]
        );
        let concatenated = materialized
            .memory()
            .concat_materialized_relations(&materialized, &materialized)
            .expect("D2D concatenated runs");
        let concat_identity = concatenated
            .memory()
            .identity_join_coordinates(concatenated.row_count(), None)
            .expect("concat identity");
        let concat_layout = concatenated.columns()[1];
        assert_eq!(
            concatenated
                .memory()
                .project_text_from_join_coordinates(
                    &concat_identity,
                    0,
                    concatenated.memory(),
                    concat_layout.value_byte_offset,
                    concat_layout.text_bytes_byte_offset.unwrap(),
                    concat_layout.text_bytes_len,
                    Some(concat_layout.validity_bitmap_offset),
                )
                .expect("concat projection"),
            vec![Some("b".to_string()); 8]
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_payload_composite_join_reads_each_accumulated_relation() {
        let runtime = CudaDriverRuntime::probe().expect("probe");
        let retain = |values: &[i32]| {
            runtime
                .retain_device_memory_copy(
                    0,
                    &values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .expect("key payload")
        };
        let ak = retain(&[1, 2]);
        let bk = retain(&[1, 2]);
        let ax = retain(&[10, 20]);
        let by = retain(&[100, 200]);
        let cx = retain(&[10, 20, 10]);
        let cy = retain(&[100, 200, 200]);
        fn key(payload: &CudaResidentDeviceMemory) -> CudaJoinPayloadKey<'_> {
            CudaJoinPayloadKey {
                payload,
                byte_offset: 0,
                validity_bitmap_offset: None,
                width: 4,
                text_bytes_byte_offset: None,
                text_bytes_len: 0,
            }
        }
        let direct = ax
            .join_fixed_payload_coordinates(
                None,
                2,
                &[0, 0],
                &[key(&ax), key(&by)],
                3,
                &[key(&cx), key(&cy)],
                None,
                None,
                false,
                false,
            )
            .expect("direct composite join");
        let mut direct_rows = direct.readback_for_test().unwrap();
        direct_rows.sort_unstable();
        assert_eq!(direct_rows, vec![vec![0, 0], vec![1, 1]]);
        let ab = ak
            .join_fixed_payload_coordinates(
                None,
                2,
                &[0],
                &[key(&ak)],
                2,
                &[key(&bk)],
                None,
                None,
                false,
                false,
            )
            .expect("first join");
        assert_eq!(ab.readback_for_test().unwrap(), vec![vec![0, 0], vec![1, 1]]);
        let abc = ak
            .join_fixed_payload_coordinates(
                Some(&ab),
                0,
                &[0, 1],
                &[key(&ax), key(&by)],
                3,
                &[key(&cx), key(&cy)],
                None,
                None,
                false,
                false,
            )
            .expect("cross-relation composite join");
        let mut rows = abc.readback_for_test().unwrap();
        rows.sort_unstable();
        assert_eq!(rows, vec![vec![0, 0, 0], vec![1, 1, 1]]);

        let mixed_left = retain(&[-1, 2]);
        let mixed_right = runtime
            .retain_device_memory_copy(
                0,
                &[-1_i64, 2, i64::from(i32::MAX) + 1]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .expect("int8 key payload");
        let mixed = mixed_left
            .join_fixed_payload_coordinates(
                None,
                2,
                &[0],
                &[key(&mixed_left)],
                3,
                &[CudaJoinPayloadKey {
                    payload: &mixed_right,
                    byte_offset: 0,
                    validity_bitmap_offset: None,
                    width: 8,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                }],
                None,
                None,
                false,
                false,
            )
            .expect("mixed int4/int8 join");
        let mut mixed_rows = mixed.readback_for_test().unwrap();
        mixed_rows.sort_unstable();
        assert_eq!(mixed_rows, vec![vec![0, 0], vec![1, 1]]);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_hash_join_inner_text_matches_unique_build_key() {
        let runtime = CudaDriverRuntime::probe().expect("probe");
        let resident = runtime
            .retain_device_memory_copy(0, &0_u64.to_le_bytes())
            .expect("resident device memory");
        let sorted = |o: HashJoinOutcome| -> Vec<(u32, u32)> {
            match o {
                HashJoinOutcome::Pairs {
                    build_idxs,
                    probe_idxs,
                } => {
                    let mut v: Vec<(u32, u32)> = build_idxs.into_iter().zip(probe_idxs).collect();
                    v.sort_unstable();
                    v
                }
                HashJoinOutcome::DuplicateBuildKey => panic!("unexpected DuplicateBuildKey"),
            }
        };
        let bytes = |texts: &[&str]| -> Vec<Vec<u8>> {
            texts.iter().map(|t| t.as_bytes().to_vec()).collect()
        };
        fn refs(v: &[Vec<u8>]) -> Vec<&[u8]> {
            v.iter().map(|b| b.as_slice()).collect()
        }
        let join = |build: &[&str], probe: &[&str]| -> Vec<(u32, u32)> {
            let (b, p) = (bytes(build), bytes(probe));
            sorted(
                resident
                    .hash_join_inner_text(&refs(&b), &refs(&p), None, None)
                    .unwrap(),
            )
        };
        // (a) unique build texts, matches + a non-matching probe.
        assert_eq!(
            join(
                &["apple", "banana", "cherry"],
                &["banana", "apple", "banana", "date"]
            ),
            vec![(0, 1), (1, 0), (1, 2)],
            "unique build text, 1:1 + a repeated probe key + a miss"
        );
        // (b) 1:N fan-out.
        assert_eq!(
            join(&["x"], &["x", "x", "x"]),
            vec![(0, 0), (0, 1), (0, 2)],
            "1:N text fan-out"
        );
        // (c) no matches.
        assert_eq!(
            join(&["a", "b"], &["c", "d"]),
            Vec::<(u32, u32)>::new(),
            "disjoint text sets"
        );
        // (d) prefix / length sensitivity: "ab" must NOT match "abc"/"abcd" (the byte-verify checks
        // length + bytes, not just the hash). build0=ab, build1=abc; probe abc->1, ab->0, abcd->none.
        assert_eq!(
            join(&["ab", "abc"], &["abc", "ab", "abcd"]),
            vec![(0, 1), (1, 0)],
            "a prefix text does not match a longer one (verify is exact)"
        );
        // (e) empty-string key (len 0 -> the FNV basis hash; a valid, matchable key).
        assert_eq!(
            join(&[""], &["", ""]),
            vec![(0, 0), (0, 1)],
            "empty-string text key matches"
        );
        // (f) duplicate build text -> reject (N:N is a follow-up).
        assert_eq!(
            {
                let (b, p) = (bytes(&["k", "k"]), bytes(&["k"]));
                resident
                    .hash_join_inner_text(&refs(&b), &refs(&p), None, None)
                    .unwrap()
            },
            HashJoinOutcome::DuplicateBuildKey,
            "a repeated build text is rejected"
        );
        // (g) empty sides.
        assert_eq!(
            join(&[], &["a"]),
            Vec::<(u32, u32)>::new(),
            "empty build -> no pairs"
        );
        assert_eq!(
            join(&["a"], &[]),
            Vec::<(u32, u32)>::new(),
            "empty probe -> no pairs"
        );
        // (h) 200 unique build texts probed by all 200 (forces collision walks) + 2 misses -- a broken
        // probe linear-probe step or a verify that mishandles a collision walk is caught here.
        let build_strs: Vec<String> = (0..200).map(|i| format!("key-{i}")).collect();
        let mut probe_strs: Vec<String> = build_strs.clone();
        probe_strs.push("absent-1".to_string());
        probe_strs.push("absent-2".to_string());
        let build_b: Vec<Vec<u8>> = build_strs.iter().map(|s| s.as_bytes().to_vec()).collect();
        let probe_b: Vec<Vec<u8>> = probe_strs.iter().map(|s| s.as_bytes().to_vec()).collect();
        let mut expected: Vec<(u32, u32)> = (0..200).map(|i| (i, i)).collect();
        expected.sort_unstable();
        assert_eq!(
            sorted(
                resident
                    .hash_join_inner_text(&refs(&build_b), &refs(&probe_b), None, None)
                    .unwrap()
            ),
            expected,
            "200 unique build texts probed by all 200 + 2 misses"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_hash_join_inner_i64_nn_emits_full_cross_product() {
        let runtime = CudaDriverRuntime::probe().expect("probe");
        let resident = runtime
            .retain_device_memory_copy(0, &0_u64.to_le_bytes())
            .expect("resident device memory");
        let join = |build: &[i64], probe: &[i64]| -> Vec<(u32, u32)> {
            let (b, p) = resident
                .hash_join_inner_i64_nn(build, probe, None, None)
                .unwrap();
            let mut v: Vec<(u32, u32)> = b.into_iter().zip(p).collect();
            v.sort_unstable();
            v
        };
        // (a) N:N -- key 1 has 2 build x 2 probe = 4 pairs; key 2 = 1x1; key 3 (probe-only) = 0.
        assert_eq!(
            join(&[1, 1, 2], &[1, 1, 2, 3]),
            vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)],
            "N:N cross product per key + a probe-only key dropped"
        );
        // (b) 1:1 (no dups) behaves like the unique join.
        assert_eq!(join(&[10, 20], &[20, 10]), vec![(0, 1), (1, 0)], "1:1");
        // (c) 1:N (unique build) and (d) N:1 (unique probe).
        assert_eq!(join(&[5], &[5, 5, 5]), vec![(0, 0), (0, 1), (0, 2)], "1:N");
        assert_eq!(
            join(&[5, 5], &[5]),
            vec![(0, 0), (1, 0)],
            "N:1 (both build rows match)"
        );
        // (e) a heavier 3x2 cross product on one key.
        assert_eq!(
            join(&[7, 7, 7], &[7, 7]),
            vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
            "3 build x 2 probe = 6 pairs"
        );
        // (f) no matches / empty sides.
        assert_eq!(join(&[1], &[2]), Vec::<(u32, u32)>::new(), "disjoint");
        assert_eq!(join(&[], &[1, 2]), Vec::<(u32, u32)>::new(), "empty build");
        assert_eq!(join(&[1, 2], &[]), Vec::<(u32, u32)>::new(), "empty probe");
        // (g) negatives + a larger spread of duplicated keys (forces collision walks in both passes).
        //   build: keys 0..50 each appearing 3x; probe: keys 0..50 each 2x + 5 misses. Each of the 50
        //   keys -> 3 build x 2 probe = 6 pairs -> 300 total.
        let mut build: Vec<i64> = Vec::new();
        for k in 0..50 {
            build.extend_from_slice(&[k - 25, k - 25, k - 25]);
        }
        let mut probe: Vec<i64> = Vec::new();
        for k in 0..50 {
            probe.extend_from_slice(&[k - 25, k - 25]);
        }
        probe.extend_from_slice(&[1000, 1001, 1002, 1003, 1004]);
        let got = join(&build, &probe);
        assert_eq!(
            got.len(),
            300,
            "50 keys x (3 build x 2 probe) = 300 pairs (misses dropped)"
        );
        // every emitted pair must share the same key on both sides.
        for &(b, p) in &got {
            assert_eq!(
                build[b as usize], probe[p as usize],
                "a pair must match on the key"
            );
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_hash_join_inner_text_nn_emits_full_cross_product() {
        let runtime = CudaDriverRuntime::probe().expect("probe");
        let resident = runtime
            .retain_device_memory_copy(0, &0_u64.to_le_bytes())
            .expect("resident device memory");
        fn refs(v: &[Vec<u8>]) -> Vec<&[u8]> {
            v.iter().map(|b| b.as_slice()).collect()
        }
        let bytes =
            |t: &[&str]| -> Vec<Vec<u8>> { t.iter().map(|s| s.as_bytes().to_vec()).collect() };
        let join = |build: &[&str], probe: &[&str]| -> Vec<(u32, u32)> {
            let (b, p) = (bytes(build), bytes(probe));
            let (bi, pi) = resident
                .hash_join_inner_text_nn(&refs(&b), &refs(&p), None, None)
                .unwrap();
            let mut v: Vec<(u32, u32)> = bi.into_iter().zip(pi).collect();
            v.sort_unstable();
            v
        };
        // (a) N:N text -- "a" 2 build x 2 probe = 4; "b" 1x1; "c" (probe-only) 0.
        assert_eq!(
            join(&["a", "a", "b"], &["a", "a", "b", "c"]),
            vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)],
            "text N:N cross product per key + a probe-only key dropped"
        );
        // (b) 1:1 / (c) 1:N / (d) N:1.
        assert_eq!(join(&["x", "y"], &["y", "x"]), vec![(0, 1), (1, 0)], "1:1");
        assert_eq!(
            join(&["z"], &["z", "z", "z"]),
            vec![(0, 0), (0, 1), (0, 2)],
            "1:N"
        );
        assert_eq!(join(&["z", "z"], &["z"]), vec![(0, 0), (1, 0)], "N:1");
        // (e) prefix/length sensitivity (the byte-verify): "ab" must NOT match "abc". build "ab","ab","abc"
        // probe "ab","abc" -> "ab":2x1=2, "abc":1x1=1.
        assert_eq!(
            join(&["ab", "ab", "abc"], &["ab", "abc"]),
            vec![(0, 0), (1, 0), (2, 1)],
            "prefix text does not cross-match (verify is exact)"
        );
        // (f) no matches / empty sides.
        assert_eq!(join(&["a"], &["b"]), Vec::<(u32, u32)>::new(), "disjoint");
        assert_eq!(join(&[], &["a"]), Vec::<(u32, u32)>::new(), "empty build");
        assert_eq!(join(&["a"], &[]), Vec::<(u32, u32)>::new(), "empty probe");
        // (g) 50 distinct keys, each 3 build x 2 probe = 6 -> 300; + misses; verify each pair shares the key.
        let build_strs: Vec<String> = (0..50)
            .flat_map(|k| (0..3).map(move |_| format!("k-{k}")))
            .collect();
        let mut probe_strs: Vec<String> = (0..50)
            .flat_map(|k| (0..2).map(move |_| format!("k-{k}")))
            .collect();
        probe_strs.extend(["miss-1".to_string(), "miss-2".to_string()]);
        let bb: Vec<Vec<u8>> = build_strs.iter().map(|s| s.as_bytes().to_vec()).collect();
        let pb: Vec<Vec<u8>> = probe_strs.iter().map(|s| s.as_bytes().to_vec()).collect();
        let (bi, pi) = resident
            .hash_join_inner_text_nn(&refs(&bb), &refs(&pb), None, None)
            .unwrap();
        assert_eq!(bi.len(), 300, "50 keys x (3 build x 2 probe) = 300 pairs");
        for (b, p) in bi.iter().zip(&pi) {
            assert_eq!(
                build_strs[*b as usize], probe_strs[*p as usize],
                "a pair must match on the text"
            );
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cuda_staged_hash_joins_validate_and_apply_validity_bitmaps() {
        let runtime = CudaDriverRuntime::probe().expect("probe");
        let resident = runtime
            .retain_device_memory_copy(0, &0_u64.to_le_bytes())
            .expect("resident device memory");
        macro_rules! assert_invalid {
            ($result:expr) => {
                assert!(
                    matches!(
                        $result,
                        Err(CudaRuntimeProbeError::InvalidInputLength(_))
                    ),
                    "non-exact validity length must fail before a device launch"
                )
            };
        }

        assert_invalid!(resident.hash_join_inner_i64(&[1], &[1], Some(&[]), None));
        assert_invalid!(resident.hash_join_inner_text(&[b"x"], &[b"x"], None, Some(&[])));
        assert_invalid!(resident.hash_join_inner_i64_nn(
            &[1, 1],
            &[1],
            Some(&[u32::MAX, u32::MAX]),
            None,
        ));
        let thirty_three_texts = vec![&b"x"[..]; 33];
        assert_invalid!(resident.hash_join_inner_text_nn(
            &thirty_three_texts,
            &[b"x"],
            Some(&[u32::MAX]),
            None,
        ));

        assert_eq!(
            resident
                .hash_join_inner_i64(&[], &[1], Some(&[]), None)
                .expect("zero-row side accepts its exact zero-word bitmap"),
            HashJoinOutcome::Pairs {
                build_idxs: Vec::new(),
                probe_idxs: Vec::new(),
            }
        );

        let sorted_unique = |outcome: HashJoinOutcome| {
            let HashJoinOutcome::Pairs {
                build_idxs,
                probe_idxs,
            } = outcome
            else {
                panic!("exact validity must not produce a duplicate verdict");
            };
            let mut pairs: Vec<(u32, u32)> =
                build_idxs.into_iter().zip(probe_idxs).collect();
            pairs.sort_unstable();
            pairs
        };
        assert_eq!(
            sorted_unique(
                resident
                    .hash_join_inner_i64(&[1, 2], &[1, 2], Some(&[0b01]), Some(&[0b11]))
                    .expect("exact i64 validity")
            ),
            vec![(0, 0)]
        );
        assert_eq!(
            sorted_unique(
                resident
                    .hash_join_inner_text(
                        &[b"x", b"y"],
                        &[b"x", b"y"],
                        Some(&[0b01]),
                        Some(&[0b11]),
                    )
                    .expect("exact text validity")
            ),
            vec![(0, 0)]
        );

        let (build, probe) = resident
            .hash_join_inner_i64_nn(&[7, 7], &[7, 7], Some(&[0b01]), Some(&[0b10]))
            .expect("exact N:N i64 validity");
        assert_eq!(build.into_iter().zip(probe).collect::<Vec<_>>(), vec![(0, 1)]);
        let (build, probe) = resident
            .hash_join_inner_text_nn(
                &[b"z", b"z"],
                &[b"z", b"z"],
                Some(&[0b01]),
                Some(&[0b10]),
            )
            .expect("exact N:N text validity");
        assert_eq!(build.into_iter().zip(probe).collect::<Vec<_>>(), vec![(0, 1)]);
    }
