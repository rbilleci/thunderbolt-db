# P8 Resident Cache Manager Proof

- git_sha: a5d9330d7bcdffa2f32da1b912d11b39f44d2019
- stream: benchmark
- milestone: P8 resident-cache-manager first code slice
- dataset_rows: 1000
- lookup_count: 16
- device_info: NVIDIA GeForce RTX 3090, 595.58.03, 24576 MiB
- reproduction: GPU_DB_BENCH_ROWS=1000 GPU_DB_BENCH_LOOKUPS=16 scripts/run_p7_gpu_residency_baseline.sh

### P8 Resident Cache Manager Proof

- dataset_rows: 1000
- lookup_count: 16
- concurrency: 1
- device_info: NVIDIA GeForce RTX 3090, 595.58.03, 24576 MiB
- current_data_residency_model: bounded_resident_snapshot_probe_with_retained_cuda_allocation_plus_per_query_h2d_fallback
- p8_cache_manager_component_supported: explicit_relational_resident_cache
- warm_resident_snapshot_execution_supported: true
- production_device_cache_supported: bounded_retained_snapshot_handle
- resident_device_memory_query_kernel_supported: bounded_count_all_int4_equality_count_int4_membership_count_int4_range_count_int4_between_count_int4_filter_group_count_text_prefix_like_count_int4_sum_avg_min_max_filtered_sum_avg_min_max_between_sum_avg_min_max_int4_projection_int4_paginated_distinct_projection_int4_paginated_filtered_distinct_projection_int4_paginated_filtered_ordered_projection_int4_grouped_count_sum_avg_min_max_grouped_having_and_filtered_grouped_count_sum_avg_min_max_filtered_grouped_having
- resident_device_memory_proof_supported: true
- resident_device_memory_allocated_bytes: 98492
- resident_device_memory_copied_bytes: 98492
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
- resident_refresh_elapsed_ms: 81.781
- resident_budget_admission_supported: true
- resident_budget_cache_state: Valid
- resident_budget_decision_accepted: true
- resident_budget_decision_reason: admitted after deterministic eviction
- resident_budget_current_bytes_before: 44
- resident_budget_current_bytes_after: 52403
- resident_budget_bytes: 52403
- resident_budget_bytes_after_admission: 52403
- resident_budget_evicted_tables: resident_aux
- resident_budget_evicted_aux_snapshot: true
- resident_budget_aux_bytes_before_eviction: 44
- resident_budget_oversize_rejected: true
- resident_budget_oversize_decision_accepted: false
- resident_budget_oversize_decision_reason: resident snapshot exceeds GPU budget
- memory_pressure_fallback_supported: true
- memory_pressure_invalidates_resident_snapshot: true
- memory_pressure_active_on_snapshot: true
- memory_pressure_fallback_count: 1
- correctness_oracle: CPU relational engine

### Retained Query-Kernel Capability Matrix

- supported_count_filters: unfiltered, int4 equality, int4 IN membership, int4 range comparison, int4 BETWEEN, retained int4 AND/OR filter groups, retained text prefix LIKE
- supported_retained_aggregates: int4 scalar SUM/AVG/MIN/MAX, int4 filtered scalar SUM/AVG/MIN/MAX, int4 BETWEEN scalar SUM/AVG/MIN/MAX, int4 grouped and filtered-grouped COUNT/SUM/AVG/MIN/MAX with grouped HAVING
- supported_retained_projections: int4 predicate projection, int4 paginated distinct projection, int4 paginated filtered distinct projection, bounded int4 paginated filtered ordered projection
- unsupported_retained_filters: non-prefix LIKE, subqueries, text filter groups beyond the single-prefix count proof, non-int4 filter-group payload columns, arbitrary expression trees
- retained_filter_family_closeout: supported retained int4 filter groups and text prefix LIKE count are closed for the current SQL subset
- unsupported_production_cache_claims: broad workload-level GPU advantage, normal planner routing to retained handles, production allocator beyond deterministic budget admission evidence

### cold_per_query_h2d_probe
- elapsed_us: 505334
- result_rows: 16
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: OrderedKeyBatch { table: "events", predicate_column: Some("id"), predicate_op: Some(Eq), order_column: "id", descending: false, matched_keys: 16 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 1429
- d2h_bytes: 973
- kernel_exec_samples: 1
- kernel_exec_total_ms: 465
- correctness_validated: true

### warm_resident_snapshot_probe
- elapsed_us: 9681
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
- elapsed_us: 38370
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
- elapsed_us: 494
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
- elapsed_us: 406
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

### resident_device_memory_membership_count_kernel_probe
- elapsed_us: 641
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: EqualityIndex { table: "events", column: "id", matched_keys: 3 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 0
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

### resident_device_memory_range_count_kernel_probe
- elapsed_us: 2780
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

### resident_device_memory_between_count_kernel_probe
- elapsed_us: 2946
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: ConjunctiveFilteredKeyBatch { table: "events", predicate_count: 2, matched_keys: 501 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 16
- kernel_exec_samples: 2
- kernel_exec_total_ms: 2
- correctness_validated: true

### resident_device_memory_filter_group_count_kernel_probe
- elapsed_us: 3793
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: DisjunctiveFilteredKeyBatch { table: "events", predicate_group_count: 2, matched_keys: 607 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 12000
- kernel_exec_samples: 3
- kernel_exec_total_ms: 3
- correctness_validated: true

### resident_device_memory_text_prefix_count_probe
- elapsed_us: 2696
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FilteredKeyBatch { table: "events", predicate_column: "category", predicate_op: LikePrefix, matched_keys: 500 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 11508
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

### resident_device_memory_sum_kernel_probe
- elapsed_us: 391
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
- elapsed_us: 26660
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FullTableScan
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 28008
- kernel_exec_samples: 1
- kernel_exec_total_ms: 26
- correctness_validated: true

### resident_device_memory_min_kernel_probe
- elapsed_us: 26682
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FullTableScan
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 28008
- kernel_exec_samples: 1
- kernel_exec_total_ms: 26
- correctness_validated: true

### resident_device_memory_max_kernel_probe
- elapsed_us: 26685
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FullTableScan
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 28008
- kernel_exec_samples: 1
- kernel_exec_total_ms: 26
- correctness_validated: true

### resident_device_memory_filtered_sum_kernel_probe
- elapsed_us: 2797
- result_rows: 1
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

### resident_device_memory_filtered_avg_kernel_probe
- elapsed_us: 2757
- result_rows: 1
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

### resident_device_memory_filtered_min_kernel_probe
- elapsed_us: 2764
- result_rows: 1
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

### resident_device_memory_filtered_max_kernel_probe
- elapsed_us: 2769
- result_rows: 1
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

### resident_device_memory_between_sum_kernel_probe
- elapsed_us: 2792
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: ConjunctiveFilteredKeyBatch { table: "events", predicate_count: 2, matched_keys: 101 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 812
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_between_avg_kernel_probe
- elapsed_us: 2780
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: ConjunctiveFilteredKeyBatch { table: "events", predicate_count: 2, matched_keys: 101 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 812
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_between_min_kernel_probe
- elapsed_us: 2796
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: ConjunctiveFilteredKeyBatch { table: "events", predicate_count: 2, matched_keys: 101 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 812
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_between_max_kernel_probe
- elapsed_us: 2775
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: ConjunctiveFilteredKeyBatch { table: "events", predicate_count: 2, matched_keys: 101 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 812
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_projection_kernel_probe
- elapsed_us: 2789
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

### resident_device_memory_distinct_projection_kernel_probe
- elapsed_us: 3319
- result_rows: 6
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: OrderedKeyBatch { table: "events", predicate_column: None, predicate_op: None, order_column: "bucket", descending: true, matched_keys: 1000 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 4008
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_filtered_distinct_projection_kernel_probe
- elapsed_us: 156346
- result_rows: 3
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: OrderedKeyBatch { table: "events", predicate_column: Some("bucket"), predicate_op: Some(Gte), order_column: "bucket", descending: true, matched_keys: 500 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 2008
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_filtered_ordered_projection_kernel_probe
- elapsed_us: 62891
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
- elapsed_us: 894
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
- elapsed_us: 884
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
- elapsed_us: 895
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
- elapsed_us: 884
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
- elapsed_us: 885
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

### resident_device_memory_filtered_grouped_sum_kernel_probe
- elapsed_us: 2977
- result_rows: 8
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FilteredKeyBatch { table: "events", predicate_column: "amount", predicate_op: Gte, matched_keys: 201 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 232
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_filtered_grouped_count_kernel_probe
- elapsed_us: 2978
- result_rows: 8
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FilteredKeyBatch { table: "events", predicate_column: "amount", predicate_op: Gte, matched_keys: 201 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 232
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_filtered_grouped_avg_kernel_probe
- elapsed_us: 2987
- result_rows: 8
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FilteredKeyBatch { table: "events", predicate_column: "amount", predicate_op: Gte, matched_keys: 201 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 232
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_filtered_grouped_min_kernel_probe
- elapsed_us: 2954
- result_rows: 8
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FilteredKeyBatch { table: "events", predicate_column: "amount", predicate_op: Gte, matched_keys: 201 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 232
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### resident_device_memory_filtered_grouped_max_kernel_probe
- elapsed_us: 2972
- result_rows: 8
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: FilteredKeyBatch { table: "events", predicate_column: "amount", predicate_op: Gte, matched_keys: 201 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 0
- d2h_bytes: 232
- kernel_exec_samples: 1
- kernel_exec_total_ms: 1
- correctness_validated: true

### post_mutation_per_query_h2d_probe
- elapsed_us: 419224
- result_rows: 1
- planned_target: Gpu(0)
- executed_target: Gpu(0)
- access_path: EqualityIndex { table: "events", column: "id", matched_keys: 1 }
- sql_fallback: false
- fallback_reason: None
- h2d_bytes: 111
- d2h_bytes: 75
- kernel_exec_samples: 1
- kernel_exec_total_ms: 418
- correctness_validated: true

decision: current P7 evidence includes bounded resident table-data snapshot SELECT probes with zero per-query H2D transfer for the app lookup workload and supported aggregate/distinct SQL shapes, retained-device-memory COUNT(*), int4 equality-predicate COUNT(*), int4 membership-predicate COUNT(*), int4 range-predicate COUNT(*), int4 BETWEEN-predicate COUNT(*), retained int4 AND/OR filter-group COUNT(*), retained text prefix LIKE COUNT(*), int4 scalar SUM/AVG/MIN/MAX, int4 filtered scalar SUM/AVG/MIN/MAX, int4 BETWEEN scalar SUM/AVG/MIN/MAX, int4 predicate-projection, int4 paginated distinct projection, int4 paginated filtered distinct projection, bounded int4 paginated filtered ordered-projection, int4 grouped COUNT/SUM/AVG/MIN/MAX with grouped HAVING, and int4 filtered grouped COUNT/SUM/AVG/MIN/MAX with filtered grouped HAVING proofs over the resident allocation, resident-byte accounting, WAL-safe invalidation, manual refresh-cost accounting, memory-pressure fallback metadata, deterministic resident-snapshot budget admission/eviction, and a retained real CUDA allocation/copy handle for encoded snapshot bytes when local driver hardware is available. Supported retained string/filter expression-kernel families are closed for the current SQL subset; the remaining production CUDA-cache boundary is normal planner routing to retained handles, a cache manager with allocator/eviction policy, richer expressions beyond the current literal comparison/range/membership/prefix/count/aggregate/projection subset, and driver-level timing refinement.
