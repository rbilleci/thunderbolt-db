# P7 Batching Sweep Benchmark

- git_sha: 928917e24af47c45c1a8610fce166e7e365741b2
- stream: benchmark
- dataset_rows: 1000
- lookup_sizes: 1 4 16 64
- device_info: NVIDIA GeForce RTX 3090, 595.58.03
- reproduction: GPU_DB_BENCH_LOOKUP_SIZES="1 4 16 64" GPU_DB_BENCH_ROWS=1000 scripts/run_p7_batching_sweep.sh

## lookup_count=1

### P7 Relational Workload Benchmark

- dataset_rows: 1000
- concurrency: 1
- device_info: NVIDIA GeForce RTX 3090, 595.58.03
- cuda_driver_probe_runtime_cache: per_engine

#### app_indexed_point_lookup
- queries: 1
- result_rows: 1
- cpu_total_us: 465
- gpu_probe_total_us: 488332
- cpu_qps: 2147.47
- gpu_probe_qps: 2.05
- cpu_latency_p50_us: 464
- cpu_latency_p95_us: 464
- cpu_latency_max_us: 464
- gpu_probe_latency_p50_us: 488331
- gpu_probe_latency_p95_us: 488331
- gpu_probe_latency_max_us: 488331
- gpu_probe_vs_cpu_total_ratio: 1048.680
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - EqualityIndex { table: "events", column: "id", matched_keys: 1 }
- h2d_bytes_total: 88
- h2d_bytes_per_query: 88.00
- d2h_bytes_total: 52
- d2h_bytes_per_result_row: 52.00
- kernel_exec_samples: 1
- kernel_exec_total_ms: 454
- correctness_validated: true

#### app_batched_or_lookup
- queries: 1
- result_rows: 1
- cpu_total_us: 634
- gpu_probe_total_us: 419604
- cpu_qps: 1575.72
- gpu_probe_qps: 2.38
- cpu_latency_p50_us: 634
- cpu_latency_p95_us: 634
- cpu_latency_max_us: 634
- gpu_probe_latency_p50_us: 419603
- gpu_probe_latency_p95_us: 419603
- gpu_probe_latency_max_us: 419603
- gpu_probe_vs_cpu_total_ratio: 661.181
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("id"), predicate_op: Some(Eq), order_column: "id", descending: false, matched_keys: 1 }
- h2d_bytes_total: 88
- h2d_bytes_per_query: 88.00
- d2h_bytes_total: 52
- d2h_bytes_per_result_row: 52.00
- kernel_exec_samples: 1
- kernel_exec_total_ms: 418
- correctness_validated: true

#### analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2375
- gpu_probe_total_us: 421570
- cpu_qps: 420.99
- gpu_probe_qps: 2.37
- cpu_latency_p50_us: 2374
- cpu_latency_p95_us: 2374
- cpu_latency_max_us: 2374
- gpu_probe_latency_p50_us: 421569
- gpu_probe_latency_p95_us: 421569
- gpu_probe_latency_max_us: 421569
- gpu_probe_vs_cpu_total_ratio: 177.477
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - FullTableScan
- h2d_bytes_total: 85135
- h2d_bytes_per_query: 85135.00
- d2h_bytes_total: 57127
- d2h_bytes_per_result_row: 57.13
- kernel_exec_samples: 1
- kernel_exec_total_ms: 419
- correctness_validated: true

#### analytic_range_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 60008
- gpu_probe_total_us: 452639
- cpu_qps: 16.66
- gpu_probe_qps: 2.21
- cpu_latency_p50_us: 60008
- cpu_latency_p95_us: 60008
- cpu_latency_max_us: 60008
- gpu_probe_latency_p50_us: 452637
- gpu_probe_latency_p95_us: 452637
- gpu_probe_latency_max_us: 452637
- gpu_probe_vs_cpu_total_ratio: 7.543
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("amount"), predicate_op: Some(Gte), order_column: "amount", descending: true, matched_keys: 101 }
- h2d_bytes_total: 8630
- h2d_bytes_per_query: 8630.00
- d2h_bytes_total: 1440
- d2h_bytes_per_result_row: 57.60
- kernel_exec_samples: 1
- kernel_exec_total_ms: 421
- correctness_validated: true

#### analytic_conjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 31537
- gpu_probe_total_us: 436719
- cpu_qps: 31.71
- gpu_probe_qps: 2.29
- cpu_latency_p50_us: 31536
- cpu_latency_p95_us: 31536
- cpu_latency_max_us: 31536
- gpu_probe_latency_p50_us: 436717
- gpu_probe_latency_p95_us: 436717
- gpu_probe_latency_max_us: 436717
- gpu_probe_vs_cpu_total_ratio: 13.848
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("<conjunction>"), predicate_op: None, order_column: "amount", descending: true, matched_keys: 51 }
- h2d_bytes_total: 4388
- h2d_bytes_per_query: 4388.00
- d2h_bytes_total: 1447
- d2h_bytes_per_result_row: 57.88
- kernel_exec_samples: 1
- kernel_exec_total_ms: 419
- correctness_validated: true

#### analytic_disjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 302379
- gpu_probe_total_us: 587631
- cpu_qps: 3.31
- gpu_probe_qps: 1.70
- cpu_latency_p50_us: 302378
- cpu_latency_p95_us: 302378
- cpu_latency_max_us: 302378
- gpu_probe_latency_p50_us: 587629
- gpu_probe_latency_p95_us: 587629
- gpu_probe_latency_max_us: 587629
- gpu_probe_vs_cpu_total_ratio: 1.943
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("<disjunction>"), predicate_op: None, order_column: "amount", descending: true, matched_keys: 506 }
- h2d_bytes_total: 42836
- h2d_bytes_per_query: 42836.00
- d2h_bytes_total: 1429
- d2h_bytes_per_result_row: 57.16
- kernel_exec_samples: 1
- kernel_exec_total_ms: 435
- correctness_validated: true

decision: batching lookup predicates into one supported OR query reduces GPU probe latency versus repeated point lookups, but analytical scans still do not beat CPU; prioritize batching plus transfer layout before making broad performance claims.

## lookup_count=4

### P7 Relational Workload Benchmark

- dataset_rows: 1000
- concurrency: 1
- device_info: NVIDIA GeForce RTX 3090, 595.58.03
- cuda_driver_probe_runtime_cache: per_engine

#### app_indexed_point_lookup
- queries: 4
- result_rows: 4
- cpu_total_us: 1433
- gpu_probe_total_us: 1695176
- cpu_qps: 2789.69
- gpu_probe_qps: 2.36
- cpu_latency_p50_us: 325
- cpu_latency_p95_us: 452
- cpu_latency_max_us: 452
- gpu_probe_latency_p50_us: 412928
- gpu_probe_latency_p95_us: 452515
- gpu_probe_latency_max_us: 452515
- gpu_probe_vs_cpu_total_ratio: 1182.253
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - EqualityIndex { table: "events", column: "id", matched_keys: 1 }
- h2d_bytes_total: 365
- h2d_bytes_per_query: 91.25
- d2h_bytes_total: 221
- d2h_bytes_per_result_row: 55.25
- kernel_exec_samples: 4
- kernel_exec_total_ms: 1677
- correctness_validated: true

#### app_batched_or_lookup
- queries: 1
- result_rows: 4
- cpu_total_us: 2382
- gpu_probe_total_us: 420547
- cpu_qps: 419.67
- gpu_probe_qps: 2.38
- cpu_latency_p50_us: 2382
- cpu_latency_p95_us: 2382
- cpu_latency_max_us: 2382
- gpu_probe_latency_p50_us: 420546
- gpu_probe_latency_p95_us: 420546
- gpu_probe_latency_max_us: 420546
- gpu_probe_vs_cpu_total_ratio: 176.492
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("id"), predicate_op: Some(Eq), order_column: "id", descending: false, matched_keys: 4 }
- h2d_bytes_total: 341
- h2d_bytes_per_query: 341.00
- d2h_bytes_total: 221
- d2h_bytes_per_result_row: 55.25
- kernel_exec_samples: 1
- kernel_exec_total_ms: 418
- correctness_validated: true

#### analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2397
- gpu_probe_total_us: 424997
- cpu_qps: 417.09
- gpu_probe_qps: 2.35
- cpu_latency_p50_us: 2396
- cpu_latency_p95_us: 2396
- cpu_latency_max_us: 2396
- gpu_probe_latency_p50_us: 424996
- gpu_probe_latency_p95_us: 424996
- gpu_probe_latency_max_us: 424996
- gpu_probe_vs_cpu_total_ratio: 177.264
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - FullTableScan
- h2d_bytes_total: 85135
- h2d_bytes_per_query: 85135.00
- d2h_bytes_total: 57127
- d2h_bytes_per_result_row: 57.13
- kernel_exec_samples: 1
- kernel_exec_total_ms: 422
- correctness_validated: true

#### analytic_range_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 62039
- gpu_probe_total_us: 463419
- cpu_qps: 16.12
- gpu_probe_qps: 2.16
- cpu_latency_p50_us: 62037
- cpu_latency_p95_us: 62037
- cpu_latency_max_us: 62037
- gpu_probe_latency_p50_us: 463417
- gpu_probe_latency_p95_us: 463417
- gpu_probe_latency_max_us: 463417
- gpu_probe_vs_cpu_total_ratio: 7.470
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("amount"), predicate_op: Some(Gte), order_column: "amount", descending: true, matched_keys: 101 }
- h2d_bytes_total: 8630
- h2d_bytes_per_query: 8630.00
- d2h_bytes_total: 1440
- d2h_bytes_per_result_row: 57.60
- kernel_exec_samples: 1
- kernel_exec_total_ms: 431
- correctness_validated: true

#### analytic_conjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 31702
- gpu_probe_total_us: 437474
- cpu_qps: 31.54
- gpu_probe_qps: 2.29
- cpu_latency_p50_us: 31701
- cpu_latency_p95_us: 31701
- cpu_latency_max_us: 31701
- gpu_probe_latency_p50_us: 437473
- gpu_probe_latency_p95_us: 437473
- gpu_probe_latency_max_us: 437473
- gpu_probe_vs_cpu_total_ratio: 13.799
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("<conjunction>"), predicate_op: None, order_column: "amount", descending: true, matched_keys: 51 }
- h2d_bytes_total: 4388
- h2d_bytes_per_query: 4388.00
- d2h_bytes_total: 1447
- d2h_bytes_per_result_row: 57.88
- kernel_exec_samples: 1
- kernel_exec_total_ms: 420
- correctness_validated: true

#### analytic_disjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 303637
- gpu_probe_total_us: 581915
- cpu_qps: 3.29
- gpu_probe_qps: 1.72
- cpu_latency_p50_us: 303637
- cpu_latency_p95_us: 303637
- cpu_latency_max_us: 303637
- gpu_probe_latency_p50_us: 581913
- gpu_probe_latency_p95_us: 581913
- gpu_probe_latency_max_us: 581913
- gpu_probe_vs_cpu_total_ratio: 1.916
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("<disjunction>"), predicate_op: None, order_column: "amount", descending: true, matched_keys: 506 }
- h2d_bytes_total: 42836
- h2d_bytes_per_query: 42836.00
- d2h_bytes_total: 1429
- d2h_bytes_per_result_row: 57.16
- kernel_exec_samples: 1
- kernel_exec_total_ms: 427
- correctness_validated: true

decision: batching lookup predicates into one supported OR query reduces GPU probe latency versus repeated point lookups, but analytical scans still do not beat CPU; prioritize batching plus transfer layout before making broad performance claims.

## lookup_count=16

### P7 Relational Workload Benchmark

- dataset_rows: 1000
- concurrency: 1
- device_info: NVIDIA GeForce RTX 3090, 595.58.03
- cuda_driver_probe_runtime_cache: per_engine

#### app_indexed_point_lookup
- queries: 16
- result_rows: 16
- cpu_total_us: 5082
- gpu_probe_total_us: 6639723
- cpu_qps: 3148.23
- gpu_probe_qps: 2.41
- cpu_latency_p50_us: 307
- cpu_latency_p95_us: 436
- cpu_latency_max_us: 436
- gpu_probe_latency_p50_us: 412582
- gpu_probe_latency_p95_us: 450095
- gpu_probe_latency_max_us: 450095
- gpu_probe_vs_cpu_total_ratio: 1306.461
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - EqualityIndex { table: "events", column: "id", matched_keys: 1 }
- h2d_bytes_total: 1485
- h2d_bytes_per_query: 92.81
- d2h_bytes_total: 909
- d2h_bytes_per_result_row: 56.81
- kernel_exec_samples: 16
- kernel_exec_total_ms: 6607
- correctness_validated: true

#### app_batched_or_lookup
- queries: 1
- result_rows: 16
- cpu_total_us: 9276
- gpu_probe_total_us: 429345
- cpu_qps: 107.80
- gpu_probe_qps: 2.33
- cpu_latency_p50_us: 9275
- cpu_latency_p95_us: 9275
- cpu_latency_max_us: 9275
- gpu_probe_latency_p50_us: 429343
- gpu_probe_latency_p95_us: 429343
- gpu_probe_latency_max_us: 429343
- gpu_probe_vs_cpu_total_ratio: 46.284
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("id"), predicate_op: Some(Eq), order_column: "id", descending: false, matched_keys: 16 }
- h2d_bytes_total: 1365
- h2d_bytes_per_query: 1365.00
- d2h_bytes_total: 909
- d2h_bytes_per_result_row: 56.81
- kernel_exec_samples: 1
- kernel_exec_total_ms: 424
- correctness_validated: true

#### analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2387
- gpu_probe_total_us: 423963
- cpu_qps: 418.76
- gpu_probe_qps: 2.36
- cpu_latency_p50_us: 2387
- cpu_latency_p95_us: 2387
- cpu_latency_max_us: 2387
- gpu_probe_latency_p50_us: 423962
- gpu_probe_latency_p95_us: 423962
- gpu_probe_latency_max_us: 423962
- gpu_probe_vs_cpu_total_ratio: 177.541
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - FullTableScan
- h2d_bytes_total: 85135
- h2d_bytes_per_query: 85135.00
- d2h_bytes_total: 57127
- d2h_bytes_per_result_row: 57.13
- kernel_exec_samples: 1
- kernel_exec_total_ms: 421
- correctness_validated: true

#### analytic_range_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 59396
- gpu_probe_total_us: 452352
- cpu_qps: 16.84
- gpu_probe_qps: 2.21
- cpu_latency_p50_us: 59395
- cpu_latency_p95_us: 59395
- cpu_latency_max_us: 59395
- gpu_probe_latency_p50_us: 452350
- gpu_probe_latency_p95_us: 452350
- gpu_probe_latency_max_us: 452350
- gpu_probe_vs_cpu_total_ratio: 7.616
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("amount"), predicate_op: Some(Gte), order_column: "amount", descending: true, matched_keys: 101 }
- h2d_bytes_total: 8630
- h2d_bytes_per_query: 8630.00
- d2h_bytes_total: 1440
- d2h_bytes_per_result_row: 57.60
- kernel_exec_samples: 1
- kernel_exec_total_ms: 421
- correctness_validated: true

#### analytic_conjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 31224
- gpu_probe_total_us: 438392
- cpu_qps: 32.03
- gpu_probe_qps: 2.28
- cpu_latency_p50_us: 31223
- cpu_latency_p95_us: 31223
- cpu_latency_max_us: 31223
- gpu_probe_latency_p50_us: 438391
- gpu_probe_latency_p95_us: 438391
- gpu_probe_latency_max_us: 438391
- gpu_probe_vs_cpu_total_ratio: 14.040
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("<conjunction>"), predicate_op: None, order_column: "amount", descending: true, matched_keys: 51 }
- h2d_bytes_total: 4388
- h2d_bytes_per_query: 4388.00
- d2h_bytes_total: 1447
- d2h_bytes_per_result_row: 57.88
- kernel_exec_samples: 1
- kernel_exec_total_ms: 421
- correctness_validated: true

#### analytic_disjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 299536
- gpu_probe_total_us: 582949
- cpu_qps: 3.34
- gpu_probe_qps: 1.72
- cpu_latency_p50_us: 299535
- cpu_latency_p95_us: 299535
- cpu_latency_max_us: 299535
- gpu_probe_latency_p50_us: 582948
- gpu_probe_latency_p95_us: 582948
- gpu_probe_latency_max_us: 582948
- gpu_probe_vs_cpu_total_ratio: 1.946
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("<disjunction>"), predicate_op: None, order_column: "amount", descending: true, matched_keys: 506 }
- h2d_bytes_total: 42836
- h2d_bytes_per_query: 42836.00
- d2h_bytes_total: 1429
- d2h_bytes_per_result_row: 57.16
- kernel_exec_samples: 1
- kernel_exec_total_ms: 431
- correctness_validated: true

decision: batching lookup predicates into one supported OR query reduces GPU probe latency versus repeated point lookups, but analytical scans still do not beat CPU; prioritize batching plus transfer layout before making broad performance claims.

## lookup_count=64

### P7 Relational Workload Benchmark

- dataset_rows: 1000
- concurrency: 1
- device_info: NVIDIA GeForce RTX 3090, 595.58.03
- cuda_driver_probe_runtime_cache: per_engine

#### app_indexed_point_lookup
- queries: 64
- result_rows: 64
- cpu_total_us: 19171
- gpu_probe_total_us: 26480421
- cpu_qps: 3338.23
- gpu_probe_qps: 2.42
- cpu_latency_p50_us: 292
- cpu_latency_p95_us: 334
- cpu_latency_max_us: 431
- gpu_probe_latency_p50_us: 411887
- gpu_probe_latency_p95_us: 422326
- gpu_probe_latency_max_us: 453446
- gpu_probe_vs_cpu_total_ratio: 1381.216
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - EqualityIndex { table: "events", column: "id", matched_keys: 1 }
- h2d_bytes_total: 5959
- h2d_bytes_per_query: 93.11
- d2h_bytes_total: 3655
- d2h_bytes_per_result_row: 57.11
- kernel_exec_samples: 64
- kernel_exec_total_ms: 26388
- correctness_validated: true

#### app_batched_or_lookup
- queries: 1
- result_rows: 64
- cpu_total_us: 37222
- gpu_probe_total_us: 441465
- cpu_qps: 26.87
- gpu_probe_qps: 2.27
- cpu_latency_p50_us: 37222
- cpu_latency_p95_us: 37222
- cpu_latency_max_us: 37222
- gpu_probe_latency_p50_us: 441464
- gpu_probe_latency_p95_us: 441464
- gpu_probe_latency_max_us: 441464
- gpu_probe_vs_cpu_total_ratio: 11.860
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("id"), predicate_op: Some(Eq), order_column: "id", descending: false, matched_keys: 64 }
- h2d_bytes_total: 5455
- h2d_bytes_per_query: 5455.00
- d2h_bytes_total: 3655
- d2h_bytes_per_result_row: 57.11
- kernel_exec_samples: 1
- kernel_exec_total_ms: 422
- correctness_validated: true

#### analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2889
- gpu_probe_total_us: 430977
- cpu_qps: 346.05
- gpu_probe_qps: 2.32
- cpu_latency_p50_us: 2889
- cpu_latency_p95_us: 2889
- cpu_latency_max_us: 2889
- gpu_probe_latency_p50_us: 430976
- gpu_probe_latency_p95_us: 430976
- gpu_probe_latency_max_us: 430976
- gpu_probe_vs_cpu_total_ratio: 149.142
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - FullTableScan
- h2d_bytes_total: 85135
- h2d_bytes_per_query: 85135.00
- d2h_bytes_total: 57127
- d2h_bytes_per_result_row: 57.13
- kernel_exec_samples: 1
- kernel_exec_total_ms: 428
- correctness_validated: true

#### analytic_range_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 59973
- gpu_probe_total_us: 449548
- cpu_qps: 16.67
- gpu_probe_qps: 2.22
- cpu_latency_p50_us: 59972
- cpu_latency_p95_us: 59972
- cpu_latency_max_us: 59972
- gpu_probe_latency_p50_us: 449547
- gpu_probe_latency_p95_us: 449547
- gpu_probe_latency_max_us: 449547
- gpu_probe_vs_cpu_total_ratio: 7.496
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("amount"), predicate_op: Some(Gte), order_column: "amount", descending: true, matched_keys: 101 }
- h2d_bytes_total: 8630
- h2d_bytes_per_query: 8630.00
- d2h_bytes_total: 1440
- d2h_bytes_per_result_row: 57.60
- kernel_exec_samples: 1
- kernel_exec_total_ms: 418
- correctness_validated: true

#### analytic_conjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 31558
- gpu_probe_total_us: 439764
- cpu_qps: 31.69
- gpu_probe_qps: 2.27
- cpu_latency_p50_us: 31557
- cpu_latency_p95_us: 31557
- cpu_latency_max_us: 31557
- gpu_probe_latency_p50_us: 439763
- gpu_probe_latency_p95_us: 439763
- gpu_probe_latency_max_us: 439763
- gpu_probe_vs_cpu_total_ratio: 13.935
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("<conjunction>"), predicate_op: None, order_column: "amount", descending: true, matched_keys: 51 }
- h2d_bytes_total: 4388
- h2d_bytes_per_query: 4388.00
- d2h_bytes_total: 1447
- d2h_bytes_per_result_row: 57.88
- kernel_exec_samples: 1
- kernel_exec_total_ms: 422
- correctness_validated: true

#### analytic_disjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 300102
- gpu_probe_total_us: 575440
- cpu_qps: 3.33
- gpu_probe_qps: 1.74
- cpu_latency_p50_us: 300101
- cpu_latency_p95_us: 300101
- cpu_latency_max_us: 300101
- gpu_probe_latency_p50_us: 575439
- gpu_probe_latency_p95_us: 575439
- gpu_probe_latency_max_us: 575439
- gpu_probe_vs_cpu_total_ratio: 1.917
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- access_paths:
  - OrderedKeyBatch { table: "events", predicate_column: Some("<disjunction>"), predicate_op: None, order_column: "amount", descending: true, matched_keys: 506 }
- h2d_bytes_total: 42836
- h2d_bytes_per_query: 42836.00
- d2h_bytes_total: 1429
- d2h_bytes_per_result_row: 57.16
- kernel_exec_samples: 1
- kernel_exec_total_ms: 423
- correctness_validated: true

decision: batching lookup predicates into one supported OR query reduces GPU probe latency versus repeated point lookups, but analytical scans still do not beat CPU; prioritize batching plus transfer layout before making broad performance claims.

