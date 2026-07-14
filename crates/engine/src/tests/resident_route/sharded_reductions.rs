use crate::{
    BenchmarkRelationalResidencyOwnedShard, BenchmarkRelationalResidencyOwnedShardInstall, Engine,
};
use gpu_db_execution::{CudaOwnedDeviceMemoryChunk, DeviceTarget};
use gpu_db_sql::{parse_command, Command, Decimal128, SqlValue};

#[test]
fn p8_sharded_resident_sum_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local_cpu_oracle();
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
    let build_shards = || {
        shard_values
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
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_shards(
        BenchmarkRelationalResidencyOwnedShardInstall {
            table: "order_line",
            gpu_id: 0,
            shards: build_shards(),
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
        parse_command("SELECT SUM(ol_amount) FROM order_line WHERE ol_o_id = 42").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "sharded_int4_equality_sum");
    assert_eq!(route.shard_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(result.rows, vec![vec![SqlValue::Int8(2405)]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "sharded_int4_equality_sum");
    assert_eq!(decision.shard_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    // S10c: executes via the per-shard `&Select`->general bridge, which records the generic execution
    // observation (`last_execution_rows == Some(1)`, one SUM row) but NOT the probe-only
    // `record_route_selected_projection_micros`. So `last_execution_matched_rows` /
    // `last_execution_match_index_micros` / `last_execution_selected_projection_micros` /
    // `last_execution_result_materialization_micros` are no longer populated; the generic rows check covers it.
    assert_eq!(decision.last_execution_rows, Some(1));

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (42, 1, 1, 1, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "sharded_int4_equality_sum");
    assert_eq!(invalidated.shard_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident shard set is Invalidated");

    let mut missing_layout_engine = Engine::new_local_cpu_oracle();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_shards = build_shards();
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
    assert_eq!(missing_layout.query_shape, "sharded_int4_equality_sum");
    assert_eq!(
        missing_layout.reason,
        "resident shard 2 lacks required int4 projection layout"
    );
}

#[test]
fn p8_sharded_resident_between_avg_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let shard_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![10, 20, 30, 40],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![10, 20, 30, 40],
        ],
        [
            vec![15, 25, 35, 45],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![15, 25, 35, 45],
        ],
        [
            vec![50, 60, 70, 80],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![50, 60, 70, 80],
        ],
        [
            vec![20, 21, 22, 23],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![20, 21, 22, 23],
        ],
    ];
    let build_shards = || {
        shard_values
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
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_shards(
        BenchmarkRelationalResidencyOwnedShardInstall {
            table: "order_line",
            gpu_id: 0,
            shards: build_shards(),
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
        parse_command("SELECT AVG(ol_amount) FROM order_line WHERE ol_o_id BETWEEN 20 AND 35")
            .unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "sharded_int4_between_avg");
    assert_eq!(route.shard_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Numeric(
            Decimal128::parse("24.5000000000000000").unwrap()
        )]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "sharded_int4_between_avg");
    assert_eq!(decision.shard_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    // S10c: executes via the per-shard `&Select`->general bridge, which records the generic execution
    // observation (`last_execution_rows == Some(1)`, one AVG row) but NOT the probe-only
    // `record_route_selected_projection_micros`. So `last_execution_matched_rows` /
    // `last_execution_match_index_micros` / `last_execution_selected_projection_micros` /
    // `last_execution_result_materialization_micros` are no longer populated; the generic rows check covers it.
    assert_eq!(decision.last_execution_rows, Some(1));

    let Command::Select(no_match_select) =
        parse_command("SELECT AVG(ol_amount) FROM order_line WHERE ol_o_id BETWEEN 90 AND 99")
            .unwrap()
    else {
        unreachable!()
    };
    let no_match = e.execute_relational_select(&no_match_select).unwrap();
    // AVG over zero matched rows is SQL NULL (PG), not the legacy canonical-zero numeric sentinel.
    assert_eq!(no_match.rows, vec![vec![SqlValue::Null]]);

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (25, 1, 1, 1, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "sharded_int4_between_avg");
    assert_eq!(invalidated.shard_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident shard set is Invalidated");

    let mut missing_layout_engine = Engine::new_local_cpu_oracle();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_shards = build_shards();
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
    assert_eq!(missing_layout.query_shape, "sharded_int4_between_avg");
    assert_eq!(
        missing_layout.reason,
        "resident shard 2 lacks required int4 projection layout"
    );
}

#[test]
fn p8_sharded_resident_filtered_max_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let shard_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![10, 20, 30, 40],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![10, 25, 70, 15],
        ],
        [
            vec![50, 60, 70, 80],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![20, 30, 40, 10],
        ],
        [
            vec![90, 91, 92, 93],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![75, 55, 65, 80],
        ],
        [
            vec![94, 95, 96, 97],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![1, 2, 3, 4],
        ],
    ];
    let build_shards = || {
        shard_values
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
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_shards(
        BenchmarkRelationalResidencyOwnedShardInstall {
            table: "order_line",
            gpu_id: 0,
            shards: build_shards(),
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
        parse_command("SELECT MAX(ol_amount) FROM order_line WHERE ol_amount >= 50").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "sharded_int4_filtered_max");
    assert_eq!(route.shard_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(80)]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "sharded_int4_filtered_max");
    assert_eq!(decision.shard_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    // S10c: executes via the per-shard `&Select`->general bridge, which records the generic execution
    // observation (`last_execution_rows == Some(1)`, one MAX row) but NOT the probe-only
    // `record_route_selected_projection_micros`. So `last_execution_matched_rows` /
    // `last_execution_match_index_micros` / `last_execution_result_materialization_micros` are no longer
    // populated; the generic rows check covers it.
    assert_eq!(decision.last_execution_rows, Some(1));

    let Command::Select(no_match_select) =
        parse_command("SELECT MAX(ol_amount) FROM order_line WHERE ol_amount >= 100").unwrap()
    else {
        unreachable!()
    };
    let no_match = e.execute_relational_select(&no_match_select).unwrap();
    // MAX over zero matched rows is SQL NULL (PG), not the legacy empty-text sentinel.
    assert_eq!(no_match.rows, vec![vec![SqlValue::Null]]);

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (25, 1, 1, 99, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "sharded_int4_filtered_max");
    assert_eq!(invalidated.shard_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident shard set is Invalidated");

    let mut missing_layout_engine = Engine::new_local_cpu_oracle();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_shards = build_shards();
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
    assert_eq!(missing_layout.query_shape, "sharded_int4_filtered_max");
    assert_eq!(
        missing_layout.reason,
        "resident shard 2 lacks required int4 projection layout"
    );
}

#[test]
fn p8_sharded_resident_filtered_min_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let shard_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![10, 20, 30, 40],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![90, 25, 70, 85],
        ],
        [
            vec![50, 60, 70, 80],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![20, 30, 40, 10],
        ],
        [
            vec![90, 91, 92, 93],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![75, 55, 65, 80],
        ],
        [
            vec![94, 95, 96, 97],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![91, 92, 93, 94],
        ],
    ];
    let build_shards = || {
        shard_values
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
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_shards(
        BenchmarkRelationalResidencyOwnedShardInstall {
            table: "order_line",
            gpu_id: 0,
            shards: build_shards(),
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
        parse_command("SELECT MIN(ol_amount) FROM order_line WHERE ol_amount <= 60").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "sharded_int4_filtered_min");
    assert_eq!(route.shard_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(10)]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "sharded_int4_filtered_min");
    assert_eq!(decision.shard_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    // S10c: executes via the per-shard `&Select`->general bridge, which records the generic execution
    // observation (`last_execution_rows == Some(1)`, one MIN row) but NOT the probe-only
    // `record_route_selected_projection_micros`. So `last_execution_matched_rows` /
    // `last_execution_match_index_micros` / `last_execution_result_materialization_micros` are no longer
    // populated; the generic rows check covers it.
    assert_eq!(decision.last_execution_rows, Some(1));

    let Command::Select(no_match_select) =
        parse_command("SELECT MIN(ol_amount) FROM order_line WHERE ol_amount <= 0").unwrap()
    else {
        unreachable!()
    };
    let no_match = e.execute_relational_select(&no_match_select).unwrap();
    // MIN over zero matched rows is SQL NULL (PG), not the legacy empty-text sentinel.
    assert_eq!(no_match.rows, vec![vec![SqlValue::Null]]);

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (25, 1, 1, 5, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "sharded_int4_filtered_min");
    assert_eq!(invalidated.shard_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident shard set is Invalidated");

    let mut missing_layout_engine = Engine::new_local_cpu_oracle();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_shards = build_shards();
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
    assert_eq!(missing_layout.query_shape, "sharded_int4_filtered_min");
    assert_eq!(
        missing_layout.reason,
        "resident shard 2 lacks required int4 projection layout"
    );
}

#[test]
fn p8_sharded_resident_filtered_avg_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let shard_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![10, 20, 30, 40],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![10, 25, 70, 15],
        ],
        [
            vec![50, 60, 70, 80],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![20, 30, 40, 10],
        ],
        [
            vec![90, 91, 92, 93],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![75, 55, 65, 80],
        ],
        [
            vec![94, 95, 96, 97],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![91, 92, 93, 94],
        ],
    ];
    let build_shards = || {
        shard_values
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
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_shards(
        BenchmarkRelationalResidencyOwnedShardInstall {
            table: "order_line",
            gpu_id: 0,
            shards: build_shards(),
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
        parse_command("SELECT AVG(ol_amount) FROM order_line WHERE ol_amount <= 60").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "sharded_int4_filtered_avg");
    assert_eq!(route.shard_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Numeric(
            Decimal128::parse("25.6250000000000000").unwrap()
        )]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "sharded_int4_filtered_avg");
    assert_eq!(decision.shard_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    // S10c: executes via the per-shard `&Select`->general bridge, which records the generic execution
    // observation (`last_execution_rows == Some(1)`, one AVG row) but NOT the probe-only
    // `record_route_selected_projection_micros`. So `last_execution_matched_rows` /
    // `last_execution_match_index_micros` / `last_execution_result_materialization_micros` are no longer
    // populated; the generic rows check covers it.
    assert_eq!(decision.last_execution_rows, Some(1));

    let Command::Select(no_match_select) =
        parse_command("SELECT AVG(ol_amount) FROM order_line WHERE ol_amount <= 0").unwrap()
    else {
        unreachable!()
    };
    let no_match = e.execute_relational_select(&no_match_select).unwrap();
    // AVG over zero matched rows is SQL NULL (PG), not the legacy canonical-zero numeric sentinel.
    assert_eq!(no_match.rows, vec![vec![SqlValue::Null]]);

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (25, 1, 1, 5, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "sharded_int4_filtered_avg");
    assert_eq!(invalidated.shard_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident shard set is Invalidated");

    let mut missing_layout_engine = Engine::new_local_cpu_oracle();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_shards = build_shards();
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
    assert_eq!(missing_layout.query_shape, "sharded_int4_filtered_avg");
    assert_eq!(
        missing_layout.reason,
        "resident shard 2 lacks required int4 projection layout"
    );
}
