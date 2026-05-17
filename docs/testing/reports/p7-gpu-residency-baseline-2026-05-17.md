# P7 GPU Residency Baseline

- git_sha: d2bce6640744271c290137a8a6a72588e42d304d
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
- current_data_residency_model: accounted_invalidated_snapshot_plus_per_query_h2d_probe
- warm_resident_execution_supported: false
- resident_snapshot_valid_before_mutation: true
- resident_snapshot_valid_after_mutation: false
- resident_bytes_current: 48341
- resident_rows_current: 1000
- resident_valid_through_index: 1001
- resident_invalidated_by_txn_id: 1002
- resident_invalidated_at_index: 1002
- resident_refresh_supported: manual_snapshot_refresh_only
- memory_pressure_fallback_supported: false
- correctness_oracle: CPU relational engine

### cold_per_query_h2d_probe
- elapsed_us: 506744
- result_rows: 16
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: OrderedKeyBatch { table: "events", predicate_column: Some("id"), predicate_op: Some(Eq), order_column: "id", descending: false, matched_keys: 16 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 1365
- d2h_bytes: 909
- kernel_exec_samples: 1
- kernel_exec_total_ms: 468
- correctness_validated: true

### warm_runtime_per_query_h2d_probe
- elapsed_us: 432962
- result_rows: 16
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: OrderedKeyBatch { table: "events", predicate_column: Some("id"), predicate_op: Some(Eq), order_column: "id", descending: false, matched_keys: 16 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 1365
- d2h_bytes: 909
- kernel_exec_samples: 1
- kernel_exec_total_ms: 427
- correctness_validated: true

### post_mutation_per_query_h2d_probe
- elapsed_us: 416461
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: EqualityIndex { table: "events", column: "id", matched_keys: 1 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 105
- d2h_bytes: 69
- kernel_exec_samples: 1
- kernel_exec_total_ms: 415
- correctness_validated: true

decision: current P7 evidence includes resident-byte accounting plus WAL-safe invalidation metadata, but query execution still uses per-query H2D probe transfer. Do not claim warm-resident performance until the engine executes from resident table data and adds memory-pressure fallback plus mutation refresh-cost evidence.
