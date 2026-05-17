# P7 GPU Residency Baseline

- git_sha: 59030c1a86ef1ab83de07d30729e16cb5f168269
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
- current_data_residency_model: per_query_h2d_probe
- warm_resident_execution_supported: false
- resident_bytes_current: 0
- resident_refresh_supported: false
- memory_pressure_fallback_supported: false
- correctness_oracle: CPU relational engine

### cold_per_query_h2d_probe
- elapsed_us: 498582
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
- elapsed_us: 424955
- result_rows: 16
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: OrderedKeyBatch { table: "events", predicate_column: Some("id"), predicate_op: Some(Eq), order_column: "id", descending: false, matched_keys: 16 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 1365
- d2h_bytes: 909
- kernel_exec_samples: 1
- kernel_exec_total_ms: 419
- correctness_validated: true

### post_mutation_per_query_h2d_probe
- elapsed_us: 417786
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: EqualityIndex { table: "events", column: "id", matched_keys: 1 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 105
- d2h_bytes: 69
- kernel_exec_samples: 1
- kernel_exec_total_ms: 417
- correctness_validated: true

decision: current P7 evidence measures cached CUDA runtime plus per-query H2D transfer, not GPU-resident table data. Do not claim warm-resident performance until the engine implements MVCC/WAL-safe resident invalidation or refresh, resident-byte accounting, memory-pressure fallback, and a mutation refresh-cost benchmark.
