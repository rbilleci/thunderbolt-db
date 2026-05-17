# P7 GPU Residency Baseline

- git_sha: a9b4c8f7038e5f5f074b939a0b4b0622bec6d8ae
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
- current_data_residency_model: bounded_resident_snapshot_probe_plus_per_query_h2d_fallback
- warm_resident_snapshot_execution_supported: true
- production_device_cache_supported: false
- resident_snapshot_valid_before_mutation: true
- resident_snapshot_valid_after_mutation: false
- resident_snapshot_valid_under_memory_pressure: false
- resident_bytes_current: 48341
- resident_rows_current: 1000
- resident_valid_through_index: 1003
- resident_invalidated_by_txn_id: 1004
- resident_invalidated_at_index: 1004
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
- resident_refresh_from_index: 1003
- resident_refresh_through_index: 1005
- resident_refresh_invalidated_by_txn_id: 1004
- resident_refresh_invalidated_at_index: 1004
- resident_refresh_invalidated_by_memory_pressure: true
- resident_refresh_elapsed_ms: 2.503
- resident_budget_admission_supported: true
- resident_budget_bytes: 48399
- resident_budget_bytes_after_admission: 48399
- resident_budget_evicted_tables: resident_aux
- resident_budget_evicted_aux_snapshot: true
- resident_budget_aux_bytes_before_eviction: 44
- resident_budget_oversize_rejected: true
- memory_pressure_fallback_supported: true
- memory_pressure_invalidates_resident_snapshot: true
- memory_pressure_active_on_snapshot: true
- memory_pressure_fallback_count: 1
- correctness_oracle: CPU relational engine

### cold_per_query_h2d_probe
- elapsed_us: 496901
- result_rows: 16
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: OrderedKeyBatch { table: "events", predicate_column: Some("id"), predicate_op: Some(Eq), order_column: "id", descending: false, matched_keys: 16 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 1365
- d2h_bytes: 909
- kernel_exec_samples: 1
- kernel_exec_total_ms: 458
- correctness_validated: true

### warm_resident_snapshot_probe
- elapsed_us: 8740
- result_rows: 16
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: OrderedKeyBatch { table: "events", predicate_column: Some("id"), predicate_op: Some(Eq), order_column: "id", descending: false, matched_keys: 16 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 0
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

### warm_resident_aggregate_distinct_probe
- elapsed_us: 32210
- result_rows: 9
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: EqualityIndex { table: "events", column: "category", matched_keys: 500 }; FilteredKeyBatch { table: "events", predicate_column: "amount", predicate_op: Gte, matched_keys: 101 }; OrderedKeyBatch { table: "events", predicate_column: None, predicate_op: None, order_column: "category", descending: false, matched_keys: 1000 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 0
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

### post_mutation_per_query_h2d_probe
- elapsed_us: 428536
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: EqualityIndex { table: "events", column: "id", matched_keys: 1 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 105
- d2h_bytes: 69
- kernel_exec_samples: 1
- kernel_exec_total_ms: 428
- correctness_validated: true

decision: current P7 evidence includes bounded resident table-data snapshot SELECT probes with zero per-query H2D transfer for the app lookup workload and supported aggregate/distinct SQL shapes, plus resident-byte accounting, WAL-safe invalidation, manual refresh-cost accounting, memory-pressure fallback metadata, and deterministic resident-snapshot budget admission/eviction. Keep production CUDA allocator claims out of scope until resident snapshots own real device memory.
