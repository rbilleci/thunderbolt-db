use crate::tests::invalidate_test_relational_residency;
use crate::{
    BenchmarkRelationalResidencyOwnedShard, BenchmarkRelationalResidencyOwnedShardInstall, Engine,
};
use gpu_db_execution::{CudaOwnedDeviceMemoryChunk, DeviceTarget};
use gpu_db_sql::{parse_command, Command, SqlValue};

#[test]
fn p8_sharded_resident_count_reduces_valid_shards_and_rejects_invalidated() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let shards = (0..4_u32)
        .map(|shard_id| {
            let row_count = 256_usize;
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            BenchmarkRelationalResidencyOwnedShard {
                shard_id,
                row_start: shard_id as usize * row_count + 1,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: Vec::new(),
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();

    let installed = e.install_benchmark_relational_residency_owned_shards(
        BenchmarkRelationalResidencyOwnedShardInstall {
            table: "order_line",
            gpu_id: 0,
            shards,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected shard install error: {err}"
        );
        return;
    }

    let Command::Select(select) = parse_command("SELECT COUNT(*) FROM order_line").unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "sharded_count_all");
    assert_eq!(route.shard_count, 4);
    assert_eq!(route.estimated_rows, 1024);
    assert_eq!(route.resident_bytes, 32);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.h2d_bytes_if_cold, 32);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(result.rows, vec![vec![SqlValue::Int8(1024)]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "sharded_count_all");
    assert_eq!(decision.shard_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(
        decision.last_execution_kernel_samples,
        Some(
            after
                .kernel_exec_samples
                .saturating_sub(before.kernel_exec_samples)
        )
    );
    // S10c: this shape now executes via the per-shard `&Select`->general bridge, which records the
    // generic execution observation (rows/h2d/d2h/kernel_samples) but NOT the probe-only per-shard
    // `record_route_device_lookup_micros`. So `last_execution_matched_rows` (set only by that recorder)
    // is no longer populated; assert the generic `last_execution_rows == Some(1)` (one COUNT(*) row).
    assert_eq!(decision.last_execution_rows, Some(1));

    invalidate_test_relational_residency(&e, "order_line");
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "sharded_count_all");
    assert_eq!(invalidated.shard_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident shard set is Invalidated");
}

#[test]
fn p8_sharded_resident_key_lookup_merges_matches_and_rejects_invalidated() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let shard_values: [Vec<i32>; 4] = [
        vec![42, 1, 42, 2],
        vec![3, 4, 5, 6],
        vec![42, 7, 8, 42],
        vec![9, 10, 11, 12],
    ];
    let shards = shard_values
        .iter()
        .enumerate()
        .map(|(shard_id, values)| {
            let row_count = values.len();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for value in values {
                bytes.extend_from_slice(&(*value).to_le_bytes());
            }
            BenchmarkRelationalResidencyOwnedShard {
                shard_id: shard_id as u32,
                row_start: shard_id * row_count + 1,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: vec!["ol_o_id".to_string()],
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();

    let installed = e.install_benchmark_relational_residency_owned_shards(
        BenchmarkRelationalResidencyOwnedShardInstall {
            table: "order_line",
            gpu_id: 0,
            shards,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected shard install error: {err}"
        );
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT ol_o_id FROM order_line WHERE ol_o_id = 42").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "sharded_int4_equality_projection");
    assert_eq!(route.shard_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(42)],
            vec![SqlValue::Int4(42)],
            vec![SqlValue::Int4(42)],
            vec![SqlValue::Int4(42)],
        ]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "sharded_int4_equality_projection");
    assert_eq!(decision.shard_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    // S10c: executes via the per-shard `&Select`->general bridge. The probe-only
    // `record_route_device_lookup_micros` (which set `last_execution_matched_rows`) is no longer called,
    // so assert the generic `last_execution_rows == Some(4)` (the 4 concatenated projected rows) instead.
    assert_eq!(decision.last_execution_rows, Some(4));

    invalidate_test_relational_residency(&e, "order_line");
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "sharded_int4_equality_projection");
    assert_eq!(invalidated.shard_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident shard set is Invalidated");
}

#[test]
fn p8_sharded_resident_multi_column_lookup_merges_projected_rows_and_rejects_missing_layout() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let shard_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![42, 1, 42, 2],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![500, 501, 502, 503],
        ],
        [
            vec![3, 4, 5, 6],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![600, 601, 602, 603],
        ],
        [
            vec![42, 7, 8, 42],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![700, 701, 702, 703],
        ],
        [
            vec![9, 10, 11, 12],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![800, 801, 802, 803],
        ],
    ];
    let shards = shard_values
        .iter()
        .enumerate()
        .map(|(shard_id, columns)| {
            let row_count = columns[0].len();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for column in columns {
                for value in column {
                    bytes.extend_from_slice(&(*value).to_le_bytes());
                }
            }
            BenchmarkRelationalResidencyOwnedShard {
                shard_id: shard_id as u32,
                row_start: shard_id * row_count + 1,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: vec![
                    "ol_o_id".to_string(),
                    "ol_i_id".to_string(),
                    "ol_quantity".to_string(),
                    "ol_amount".to_string(),
                ],
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();

    let installed = e.install_benchmark_relational_residency_owned_shards(
        BenchmarkRelationalResidencyOwnedShardInstall {
            table: "order_line",
            gpu_id: 0,
            shards,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected shard install error: {err}"
        );
        return;
    }

    let Command::Select(select) = parse_command(
        "SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = 42",
    )
    .unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(
        route.query_shape,
        "sharded_int4_equality_multi_column_projection"
    );
    assert_eq!(route.shard_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert!(route.d2h_bytes_estimate > 0);

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![
            vec![
                SqlValue::Int4(42),
                SqlValue::Int4(100),
                SqlValue::Int4(5),
                SqlValue::Int4(500)
            ],
            vec![
                SqlValue::Int4(42),
                SqlValue::Int4(102),
                SqlValue::Int4(7),
                SqlValue::Int4(502)
            ],
            vec![
                SqlValue::Int4(42),
                SqlValue::Int4(300),
                SqlValue::Int4(13),
                SqlValue::Int4(700)
            ],
            vec![
                SqlValue::Int4(42),
                SqlValue::Int4(303),
                SqlValue::Int4(16),
                SqlValue::Int4(703)
            ],
        ]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "sharded_int4_equality_multi_column_projection"
    );
    assert_eq!(decision.shard_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    // S10c: executes via the per-shard `&Select`->general bridge, which records the generic execution
    // observation but NOT the probe-only `record_route_{device_lookup,selected_projection}_micros`. So
    // `last_execution_matched_rows` / `last_execution_match_index_micros` /
    // `last_execution_selected_projection_micros` are no longer populated; assert the generic
    // `last_execution_rows == Some(4)` (the 4 concatenated projected rows) instead.
    assert_eq!(decision.last_execution_rows, Some(4));

    invalidate_test_relational_residency(&e, "order_line");
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(
        invalidated.query_shape,
        "sharded_int4_equality_multi_column_projection"
    );
    assert_eq!(invalidated.shard_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident shard set is Invalidated");

    let mut missing_layout_engine = Engine::new_local_test_engine();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_shards = shard_values
        .iter()
        .enumerate()
        .map(|(shard_id, columns)| {
            let row_count = columns[0].len();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for column in columns {
                for value in column {
                    bytes.extend_from_slice(&(*value).to_le_bytes());
                }
            }
            BenchmarkRelationalResidencyOwnedShard {
                shard_id: shard_id as u32,
                row_start: shard_id * row_count + 1,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: vec![
                    "ol_o_id".to_string(),
                    "ol_i_id".to_string(),
                    "ol_quantity".to_string(),
                    "ol_amount".to_string(),
                ],
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();
    missing_layout_shards[2].resident_device_int4_columns.pop();
    let installed = missing_layout_engine.install_benchmark_relational_residency_owned_shards(
        BenchmarkRelationalResidencyOwnedShardInstall {
            table: "order_line",
            gpu_id: 0,
            shards: missing_layout_shards,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected shard install error: {err}"
        );
        return;
    }
    let missing_layout = missing_layout_engine.plan_relational_resident_route(&select);
    assert!(!missing_layout.accepted);
    assert_eq!(
        missing_layout.query_shape,
        "sharded_int4_equality_multi_column_projection"
    );
    assert_eq!(
        missing_layout.reason,
        "resident shard 2 lacks required int4 projection layout"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn p8_sharded_resident_multi_column_lookup_orders_more_than_one_warp_of_matches_per_shard() {
    // Engine-side coverage for the RETIRE-003 device-ordered row-index replacement. The
    // sharded multi-column route iterates shards in order and, within each shard,
    // materializes one output row per device-compacted match in
    // that vector's order via a strictly positional gather. The CPU/non-resident reference emits
    // a shard's matching rows in ASCENDING row order. The resident kernel, however, appends
    // matches in `atom.global.add` SCHEDULE order, which is ascending only while all matches in
    // a shard fit in ONE warp (<= 32). Every existing sharded parity test stays under
    // that boundary (<= 2 matches per shard), so this is the first test that puts MORE THAN
    // ONE WARP of matches in a SINGLE shard.
    //
    // More than 32 interleaved matches make this non-vacuous: a regression to the retired atomic
    // append plus host-sort seam would no longer exercise the predicate VM's ordered compactor,
    // while an unordered append without repair would diverge from the exact rows below.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    const NEEDLE: i32 = 42;
    // Two shards. Shard 0 carries a MULTI-WARP block of matches: 200 rows where the
    // even rows match the needle (100 matches >> 32, interleaved across many warps and several
    // 128-thread blocks); the projected columns are distinct per row so the asserted order is
    // load-bearing. Shard 1 is a small non-matching tail (exercises the cross-shard
    // merge after the multi-warp shard).
    let p0_rows: usize = 200;
    let p0_ol_o_id: Vec<i32> = (0..p0_rows as i32)
        .map(|row| if row % 2 == 0 { NEEDLE } else { row + 1000 })
        .collect();
    // Make the other three columns unique, monotonic functions of the row so a mis-ordered
    // gather is caught by the exact row comparison (not just by the key column).
    let p0_ol_i_id: Vec<i32> = (0..p0_rows as i32).map(|row| 10_000 + row).collect();
    let p0_ol_quantity: Vec<i32> = (0..p0_rows as i32).map(|row| 20_000 + row).collect();
    let p0_ol_amount: Vec<i32> = (0..p0_rows as i32).map(|row| 30_000 + row).collect();

    let p1_ol_o_id = vec![1, 2, 3, 4];
    let p1_ol_i_id = vec![401, 402, 403, 404];
    let p1_ol_quantity = vec![17, 18, 19, 20];
    let p1_ol_amount = vec![801, 802, 803, 804];

    let shard_columns: Vec<[Vec<i32>; 4]> = vec![
        [p0_ol_o_id, p0_ol_i_id, p0_ol_quantity, p0_ol_amount],
        [p1_ol_o_id, p1_ol_i_id, p1_ol_quantity, p1_ol_amount],
    ];

    // CPU reference: rows from each shard in ASCENDING row order, shards in order.
    let mut expected_rows: Vec<Vec<SqlValue>> = Vec::new();
    for columns in &shard_columns {
        for (row, key) in columns[0].iter().enumerate() {
            if *key == NEEDLE {
                expected_rows.push(vec![
                    SqlValue::Int4(*key),
                    SqlValue::Int4(columns[1][row]),
                    SqlValue::Int4(columns[2][row]),
                    SqlValue::Int4(columns[3][row]),
                ]);
            }
        }
    }
    assert!(
        expected_rows.len() > 32,
        "test must match more than one warp of rows in a single shard"
    );

    let mut row_cursor = 1usize;
    let shards = shard_columns
        .iter()
        .enumerate()
        .map(|(shard_id, columns)| {
            let row_count = columns[0].len();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for column in columns {
                for value in column {
                    bytes.extend_from_slice(&(*value).to_le_bytes());
                }
            }
            let row_start = row_cursor;
            row_cursor += row_count;
            BenchmarkRelationalResidencyOwnedShard {
                shard_id: shard_id as u32,
                row_start,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: vec![
                    "ol_o_id".to_string(),
                    "ol_i_id".to_string(),
                    "ol_quantity".to_string(),
                    "ol_amount".to_string(),
                ],
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();

    let installed = e.install_benchmark_relational_residency_owned_shards(
        BenchmarkRelationalResidencyOwnedShardInstall {
            table: "order_line",
            gpu_id: 0,
            shards,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected shard install error: {err}"
        );
        return;
    }

    let Command::Select(select) = parse_command(
        "SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = 42",
    )
    .unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(
        route.query_shape,
        "sharded_int4_equality_multi_column_projection"
    );
    assert_eq!(route.shard_count, 2);

    // Re-run so a non-deterministic (sort-less) cross-warp order is caught on some iteration.
    for iter in 0..25 {
        let result = e.execute_relational_select(&select).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(result.fallback_reason, None);
        assert_eq!(
            result.rows, expected_rows,
            "sharded multi-warp resident rows were not in ascending reference order on \
                 iteration {iter} — ordered device compaction regressed"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn p8_batched_multi_column_projection_matches_per_query_for_more_than_one_warp_of_matches() {
    // Thread-3 Stage-4 ordered-parity gate (multi-column all-int4). The batched submit/complete
    // path scatters rows per needle in the `equal_any` kernel's `atom.global.add` SCHEDULE
    // order; the per-query path (`execute_relational_select` -> the multi-column probe) uses the
    // general predicate VM's ordered device compactor. Both must return the SAME rows in
    // the SAME (ascending) order for a MULTI-WARP match count (>32, where the atomic-append
    // order is non-deterministic), so the batched output is byte-identical to the per-query path.
    //
    // Non-vacuous: the projected `seq` column is a by-row SCRAMBLED hash, so the ascending-by-row
    // reference is NOT value-sorted and NOT the atomic-append order. WITHOUT the stable-order
    // sort the batched scatter would emit a non-deterministic permutation (caught by the exact
    // comparison and the 25× loop), and it would differ from the per-query path.
    let mut e = Engine::new_local_test_engine();
    // R3-004: the retained API must preserve stable row order over the authoritative shard set.
    e.execute_text(1, "CREATE TABLE t (k INT, seq INT)")
        .unwrap();
    const NEEDLE: i32 = 7;
    let rows: usize = 200; // even rows match => 100 matches >> 32 (multi-warp/multi-block).
    let row_value = |row: usize| -> i32 {
        let r = row as u64;
        let h = r.wrapping_mul(2_654_435_761) ^ (r << 13) ^ 0x9E37_79B9;
        (1 + (h % 1_000_000)) as i32
    };
    let mut values = String::new();
    for row in 0..rows {
        if row > 0 {
            values.push_str(", ");
        }
        let k = if row % 2 == 0 {
            NEEDLE
        } else {
            row as i32 + 1000
        };
        values.push_str(&format!("({k}, {})", row_value(row)));
    }
    e.execute_text(2, &format!("INSERT INTO t (k, seq) VALUES {values}"))
        .unwrap();
    e.populate_relational_residency_snapshot("t").unwrap();

    let Command::Select(select) = parse_command("SELECT k, seq FROM t WHERE k = 7").unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    assert_eq!(
        route.query_shape,
        "sharded_int4_equality_multi_column_projection"
    );

    // Ascending-by-row reference (independent of either GPU path).
    let expected: Vec<Vec<SqlValue>> = (0..rows)
        .filter(|row| row % 2 == 0)
        .map(|row| vec![SqlValue::Int4(NEEDLE), SqlValue::Int4(row_value(row))])
        .collect();
    assert!(
        expected.len() > 32,
        "test must match more than one warp of rows"
    );
    // The projected by-row sequence must NOT already be value-sorted, else an atomic-append or
    // value-sort impl would pass vacuously.
    let proj_by_row: Vec<i32> = expected
        .iter()
        .map(|row| match row[1] {
            SqlValue::Int4(v) => v,
            _ => unreachable!(),
        })
        .collect();
    let mut proj_sorted = proj_by_row.clone();
    proj_sorted.sort_unstable();
    assert_ne!(
        proj_by_row, proj_sorted,
        "projected by-row sequence is accidentally value-sorted — pick a payload that isn't"
    );

    for iter in 0..25 {
        let per_query = e.execute_relational_select(&select).unwrap();
        assert_eq!(
            per_query.rows, expected,
            "per-query multi-column rows not ascending-by-row on iteration {iter}"
        );

        let job = e.prepare_relational_retained_read_job(&select).unwrap();
        let submission = e
            .submit_relational_retained_read_jobs_with_resident_device_memory_probe(
                std::slice::from_ref(&job),
            )
            .unwrap();
        let batched = e
            .complete_relational_retained_read_submission(submission)
            .unwrap();
        assert_eq!(batched.len(), 1);
        assert_eq!(
            batched[0].rows, expected,
            "batched multi-column rows not ascending-by-row on iteration {iter} \
                 — the stable-order sort in the result assembly is missing or ineffective?"
        );
        assert_eq!(
            batched[0].rows, per_query.rows,
            "batched multi-column output diverged from the per-query path on iteration {iter}"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn p8_batched_mixed_column_projection_matches_per_query_for_more_than_one_warp_of_matches() {
    // Thread-3 Stage-4 ordered-parity gate (mixed int4/text). The batched path for the mixed
    // shape falls through to `..._batch_inner` (the int4 `equal_any` fast path is int4-only), and
    // the per-query mixed path delegates to the SAME `batch_inner`; both now sort each needle's
    // slice ascending-by-row_index. They must return the SAME rows in the SAME order for a
    // MULTI-WARP match count.
    //
    // Non-vacuous: the projected `label` text is a by-row SCRAMBLED value, so the ascending-by-row
    // reference is neither value-sorted nor the atomic-append order. WITHOUT the sort the scatter
    // is a non-deterministic permutation (caught by the exact comparison + the 25× loop).
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (k INT, label TEXT)")
        .unwrap();
    const NEEDLE: i32 = 7;
    let rows: usize = 200; // even rows match => 100 matches >> 32.
                           // Scrambled-by-row label so the ascending-by-row text order is not lexicographically sorted.
    let label_value = |row: usize| -> String {
        let r = row as u64;
        let h = r.wrapping_mul(2_654_435_761) ^ (r << 13) ^ 0x9E37_79B9;
        format!("L{:08}", h % 100_000_000)
    };
    let mut values = String::new();
    for row in 0..rows {
        if row > 0 {
            values.push_str(", ");
        }
        let k = if row % 2 == 0 {
            NEEDLE
        } else {
            row as i32 + 1000
        };
        values.push_str(&format!("({k}, '{}')", label_value(row)));
    }
    e.execute_text(2, &format!("INSERT INTO t (k, label) VALUES {values}"))
        .unwrap();
    e.populate_relational_residency_snapshot("t").unwrap();

    let Command::Select(select) = parse_command("SELECT k, label FROM t WHERE k = 7").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    assert_eq!(
        route.query_shape,
        "sharded_int4_equality_mixed_column_projection"
    );

    let expected: Vec<Vec<SqlValue>> = (0..rows)
        .filter(|row| row % 2 == 0)
        .map(|row| vec![SqlValue::Int4(NEEDLE), SqlValue::Text(label_value(row))])
        .collect();
    assert!(
        expected.len() > 32,
        "test must match more than one warp of rows"
    );
    let label_by_row: Vec<String> = expected
        .iter()
        .map(|row| match &row[1] {
            SqlValue::Text(v) => v.clone(),
            _ => unreachable!(),
        })
        .collect();
    let mut label_sorted = label_by_row.clone();
    label_sorted.sort();
    assert_ne!(
        label_by_row, label_sorted,
        "projected by-row label sequence is accidentally sorted — pick a payload that isn't"
    );

    for iter in 0..25 {
        let per_query = e.execute_relational_select(&select).unwrap();
        assert_eq!(
            per_query.rows, expected,
            "per-query mixed rows not ascending-by-row on iteration {iter}"
        );

        let job = e.prepare_relational_retained_read_job(&select).unwrap();
        let submission = e
            .submit_relational_retained_read_jobs_with_resident_device_memory_probe(
                std::slice::from_ref(&job),
            )
            .unwrap();
        let batched = e
            .complete_relational_retained_read_submission(submission)
            .unwrap();
        assert_eq!(batched.len(), 1);
        assert_eq!(
            batched[0].rows, expected,
            "batched mixed rows not ascending-by-row on iteration {iter} \
                 — the stable-order sort in the result assembly is missing or ineffective?"
        );
        assert_eq!(
            batched[0].rows, per_query.rows,
            "batched mixed output diverged from the per-query path on iteration {iter}"
        );
    }
}
