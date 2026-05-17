# P7 GPU Residency Baseline

- git_sha: 8bf0f394144b1a8e61847f88e4da167426461e70
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
- current_data_residency_model: bounded_resident_snapshot_probe_with_retained_cuda_allocation_plus_per_query_h2d_fallback
- warm_resident_snapshot_execution_supported: true
- production_device_cache_supported: bounded_retained_snapshot_handle
- resident_device_memory_query_kernel_supported: bounded_count_all_int4_equality_count_int4_range_count_int4_sum_avg_min_max_int4_projection_int4_ordered_projection_and_int4_grouped_count_sum_avg_min_max
- resident_device_memory_proof_supported: true
- resident_device_memory_allocated_bytes: 73135
- resident_device_memory_copied_bytes: 73135
- resident_device_memory_gpu_id: 0
- resident_device_memory_retained: true
- resident_snapshot_valid_before_mutation: true
- resident_snapshot_valid_after_mutation: false
- resident_snapshot_valid_under_memory_pressure: false
- resident_bytes_current: 52341
- resident_rows_current: 1000
- resident_valid_through_index: 1003
- resident_invalidated_by_txn_id: 1004
- resident_invalidated_at_index: 1004
- resident_refreshed_rows_current: 1001
- resident_refreshed_bytes_current: 52403
- resident_refresh_supported: manual_snapshot_refresh_with_cost_accounting
- resident_refresh_cost_recorded: true
- resident_refresh_previous_rows: 1000
- resident_refresh_refreshed_rows: 1001
- resident_refresh_row_delta: 1
- resident_refresh_previous_bytes: 52341
- resident_refresh_refreshed_bytes: 52403
- resident_refresh_byte_delta: 62
- resident_refresh_from_index: 1003
- resident_refresh_through_index: 1005
- resident_refresh_invalidated_by_txn_id: 1004
- resident_refresh_invalidated_at_index: 1004
- resident_refresh_invalidated_by_memory_pressure: true
- resident_refresh_elapsed_ms: 80.590
- resident_budget_admission_supported: true
- resident_budget_bytes: 52403
- resident_budget_bytes_after_admission: 52403
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
- elapsed_us: 497115
- result_rows: 16
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: OrderedKeyBatch { table: "events", predicate_column: Some("id"), predicate_op: Some(Eq), order_column: "id", descending: false, matched_keys: 16 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 1429
- d2h_bytes: 973
- kernel_exec_samples: 1
- kernel_exec_total_ms: 457
- correctness_validated: true

### warm_resident_snapshot_probe
- elapsed_us: 9654
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
- elapsed_us: 37560
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

### resident_device_memory_count_kernel_probe
- elapsed_us: 425
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FullTableScan
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 0
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

### resident_device_memory_filtered_count_kernel_probe
- elapsed_us: 399
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: EqualityIndex { table: "events", column: "id", matched_keys: 1 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 0
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

### resident_device_memory_range_count_kernel_probe
- elapsed_us: 2731
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FilteredKeyBatch { table: "events", predicate_column: "id", predicate_op: Gte, matched_keys: 251 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 0
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

### resident_device_memory_sum_kernel_probe
- elapsed_us: 385
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FullTableScan
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 0
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

### resident_device_memory_avg_kernel_probe
- elapsed_us: 27465
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FullTableScan
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 28008
- kernel_exec_samples: 1
- kernel_exec_total_ms: 27
- correctness_validated: true

### resident_device_memory_min_kernel_probe
- elapsed_us: 27462
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FullTableScan
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 28008
- kernel_exec_samples: 1
- kernel_exec_total_ms: 27
- correctness_validated: true

### resident_device_memory_max_kernel_probe
- elapsed_us: 27458
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FullTableScan
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 28008
- kernel_exec_samples: 1
- kernel_exec_total_ms: 27
- correctness_validated: true

### resident_device_memory_projection_kernel_probe
- elapsed_us: 2781
- result_rows: 201
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FilteredKeyBatch { table: "events", predicate_column: "amount", predicate_op: Gte, matched_keys: 201 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 812
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_ordered_projection_kernel_probe
- elapsed_us: 62824
- result_rows: 8
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: OrderedKeyBatch { table: "events", predicate_column: Some("amount"), predicate_op: Some(Gte), order_column: "amount", descending: true, matched_keys: 201 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 40
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_grouped_sum_kernel_probe
- elapsed_us: 843
- result_rows: 8
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FullTableScan
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 232
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_grouped_count_kernel_probe
- elapsed_us: 816
- result_rows: 8
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FullTableScan
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 232
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_grouped_avg_kernel_probe
- elapsed_us: 831
- result_rows: 8
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FullTableScan
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 232
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_grouped_min_kernel_probe
- elapsed_us: 818
- result_rows: 8
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FullTableScan
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 232
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_grouped_max_kernel_probe
- elapsed_us: 817
- result_rows: 8
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FullTableScan
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 232
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### post_mutation_per_query_h2d_probe
- elapsed_us: 417216
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: EqualityIndex { table: "events", column: "id", matched_keys: 1 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 111
- d2h_bytes: 75
- kernel_exec_samples: 1
- kernel_exec_total_ms: 416
- correctness_validated: true

decision: current P7 evidence includes bounded resident table-data snapshot SELECT probes with zero per-query H2D transfer for the app lookup workload and supported aggregate/distinct SQL shapes, retained-device-memory COUNT(*), int4 equality-predicate COUNT(*), int4 range-predicate COUNT(*), int4 scalar SUM/AVG/MIN/MAX, int4 predicate-projection, bounded int4 ordered-projection, and int4 grouped COUNT/SUM/AVG/MIN/MAX proofs over the resident allocation, resident-byte accounting, WAL-safe invalidation, manual refresh-cost accounting, memory-pressure fallback metadata, deterministic resident-snapshot budget admission/eviction, and a retained real CUDA allocation/copy handle for encoded snapshot bytes when local driver hardware is available. Keep broad production CUDA cache claims out of scope until expression kernels read directly from retained device-memory handles.
