# P7 GPU Residency Baseline

- git_sha: 51fdeba2f2c81f346ac307b493a953a1b38cf4a9
- stream: benchmark
- dataset_rows: 1000
- lookup_count: 16
- device_info: NVIDIA GeForce RTX 3090, 595.58.03, 24576 MiB
- reproduction: GPU_DB_BENCH_ROWS=1000 GPU_DB_BENCH_LOOKUPS=16 scripts/run_p7_gpu_residency_baseline.sh

### P7 GPU Residency Baseline

- dataset_rows: 1000
- lookup_count: 16
- concurrency: 1
- device_info: NVIDIA GeForce RTX 3090, 595.58.03, 24576 MiB
- current_data_residency_model: accounted_invalidated_refresh_cost_plus_per_query_h2d_probe
- warm_resident_execution_supported: false
- resident_snapshot_valid_before_mutation: true
- resident_snapshot_valid_after_mutation: false
- resident_snapshot_valid_under_memory_pressure: false
- resident_bytes_current: 48341
- resident_rows_current: 1000
- resident_valid_through_index: 1001
- resident_invalidated_by_txn_id: 1002
- resident_invalidated_at_index: 1002
- resident_refreshed_rows_current: 1001
- resident_refreshed_bytes_current: 48399
- resident_refresh_supported: manual_snapshot_refresh_with_cost_accounting
- resident_refresh_cost_recorded: true
- resident_refresh_previous_rows: 1000
- resident_refresh_refreshed_rows: 1001
- resident_refresh_row_delta: 1
- resident_refresh_previous_bytes: 48341
- resident_refresh_refreshed_bytes: 48399
- resident_refresh_byte_delta: 58
- resident_refresh_from_index: 1001
- resident_refresh_through_index: 1003
- resident_refresh_invalidated_by_txn_id: 1002
- resident_refresh_invalidated_at_index: 1002
- resident_refresh_invalidated_by_memory_pressure: true
- resident_refresh_elapsed_ms: 1.982
- memory_pressure_fallback_supported: true
- memory_pressure_invalidates_resident_snapshot: true
- memory_pressure_active_on_snapshot: true
- memory_pressure_fallback_count: 1
- correctness_oracle: CPU relational engine

### cold_per_query_h2d_probe
- elapsed_us: 498613
- result_rows: 16
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: OrderedKeyBatch { table: "events", predicate_column: Some("id"), predicate_op: Some(Eq), order_column: "id", descending: false, matched_keys: 16 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 1365
- d2h_bytes: 909
- kernel_exec_samples: 1
- kernel_exec_total_ms: 460
- correctness_validated: true

### warm_runtime_per_query_h2d_probe
- elapsed_us: 418373
- result_rows: 16
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: OrderedKeyBatch { table: "events", predicate_column: Some("id"), predicate_op: Some(Eq), order_column: "id", descending: false, matched_keys: 16 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 1365
- d2h_bytes: 909
- kernel_exec_samples: 1
- kernel_exec_total_ms: 412
- correctness_validated: true

### post_mutation_per_query_h2d_probe
- elapsed_us: 407683
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: EqualityIndex { table: "events", column: "id", matched_keys: 1 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 105
- d2h_bytes: 69
- kernel_exec_samples: 1
- kernel_exec_total_ms: 407
- correctness_validated: true

decision: current P7 evidence includes resident-byte accounting, WAL-safe invalidation metadata, manual mutation refresh-cost accounting, and memory-pressure fallback metadata, but query execution still uses per-query H2D probe transfer. Do not claim warm-resident performance until the engine executes from resident table data.
