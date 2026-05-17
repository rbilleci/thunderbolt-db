# P7 Batching Sweep Benchmark

- git_sha: 1ee02afd3f8c00aa0114b58bdc379f76e53a0a58
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
- cpu_total_us: 436
- gpu_probe_total_us: 492889
- cpu_qps: 2290.46
- gpu_probe_qps: 2.03
- cpu_latency_p50_us: 435
- cpu_latency_p95_us: 435
- cpu_latency_max_us: 435
- gpu_probe_latency_p50_us: 492888
- gpu_probe_latency_p95_us: 492888
- gpu_probe_latency_max_us: 492888
- gpu_probe_vs_cpu_total_ratio: 1128.944
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 88
- h2d_bytes_per_query: 88.00
- d2h_bytes_total: 52
- d2h_bytes_per_result_row: 52.00
- kernel_exec_samples: 1
- kernel_exec_total_ms: 459
- correctness_validated: true

#### app_batched_or_lookup
- queries: 1
- result_rows: 1
- cpu_total_us: 617
- gpu_probe_total_us: 420555
- cpu_qps: 1619.57
- gpu_probe_qps: 2.38
- cpu_latency_p50_us: 616
- cpu_latency_p95_us: 616
- cpu_latency_max_us: 616
- gpu_probe_latency_p50_us: 420554
- gpu_probe_latency_p95_us: 420554
- gpu_probe_latency_max_us: 420554
- gpu_probe_vs_cpu_total_ratio: 681.118
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 88
- h2d_bytes_per_query: 88.00
- d2h_bytes_total: 52
- d2h_bytes_per_result_row: 52.00
- kernel_exec_samples: 1
- kernel_exec_total_ms: 419
- correctness_validated: true

#### analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2443
- gpu_probe_total_us: 424073
- cpu_qps: 409.18
- gpu_probe_qps: 2.36
- cpu_latency_p50_us: 2443
- cpu_latency_p95_us: 2443
- cpu_latency_max_us: 2443
- gpu_probe_latency_p50_us: 424072
- gpu_probe_latency_p95_us: 424072
- gpu_probe_latency_max_us: 424072
- gpu_probe_vs_cpu_total_ratio: 173.521
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
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
- cpu_total_us: 59086
- gpu_probe_total_us: 460647
- cpu_qps: 16.92
- gpu_probe_qps: 2.17
- cpu_latency_p50_us: 59085
- cpu_latency_p95_us: 59085
- cpu_latency_max_us: 59085
- gpu_probe_latency_p50_us: 460645
- gpu_probe_latency_p95_us: 460645
- gpu_probe_latency_max_us: 460645
- gpu_probe_vs_cpu_total_ratio: 7.796
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 8630
- h2d_bytes_per_query: 8630.00
- d2h_bytes_total: 1440
- d2h_bytes_per_result_row: 57.60
- kernel_exec_samples: 1
- kernel_exec_total_ms: 429
- correctness_validated: true

#### analytic_conjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 30781
- gpu_probe_total_us: 438918
- cpu_qps: 32.49
- gpu_probe_qps: 2.28
- cpu_latency_p50_us: 30780
- cpu_latency_p95_us: 30780
- cpu_latency_max_us: 30780
- gpu_probe_latency_p50_us: 438917
- gpu_probe_latency_p95_us: 438917
- gpu_probe_latency_max_us: 438917
- gpu_probe_vs_cpu_total_ratio: 14.259
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
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
- cpu_total_us: 291799
- gpu_probe_total_us: 573666
- cpu_qps: 3.43
- gpu_probe_qps: 1.74
- cpu_latency_p50_us: 291798
- cpu_latency_p95_us: 291798
- cpu_latency_max_us: 291798
- gpu_probe_latency_p50_us: 573665
- gpu_probe_latency_p95_us: 573665
- gpu_probe_latency_max_us: 573665
- gpu_probe_vs_cpu_total_ratio: 1.966
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 42836
- h2d_bytes_per_query: 42836.00
- d2h_bytes_total: 1429
- d2h_bytes_per_result_row: 57.16
- kernel_exec_samples: 1
- kernel_exec_total_ms: 425
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
- cpu_total_us: 1414
- gpu_probe_total_us: 1683854
- cpu_qps: 2827.55
- gpu_probe_qps: 2.38
- cpu_latency_p50_us: 321
- cpu_latency_p95_us: 444
- cpu_latency_max_us: 444
- gpu_probe_latency_p50_us: 407847
- gpu_probe_latency_p95_us: 459257
- gpu_probe_latency_max_us: 459257
- gpu_probe_vs_cpu_total_ratio: 1190.296
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 365
- h2d_bytes_per_query: 91.25
- d2h_bytes_total: 221
- d2h_bytes_per_result_row: 55.25
- kernel_exec_samples: 4
- kernel_exec_total_ms: 1666
- correctness_validated: true

#### app_batched_or_lookup
- queries: 1
- result_rows: 4
- cpu_total_us: 4064
- gpu_probe_total_us: 428238
- cpu_qps: 246.00
- gpu_probe_qps: 2.34
- cpu_latency_p50_us: 4064
- cpu_latency_p95_us: 4064
- cpu_latency_max_us: 4064
- gpu_probe_latency_p50_us: 428237
- gpu_probe_latency_p95_us: 428237
- gpu_probe_latency_max_us: 428237
- gpu_probe_vs_cpu_total_ratio: 105.348
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 341
- h2d_bytes_per_query: 341.00
- d2h_bytes_total: 221
- d2h_bytes_per_result_row: 55.25
- kernel_exec_samples: 1
- kernel_exec_total_ms: 424
- correctness_validated: true

#### analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2361
- gpu_probe_total_us: 415821
- cpu_qps: 423.44
- gpu_probe_qps: 2.40
- cpu_latency_p50_us: 2361
- cpu_latency_p95_us: 2361
- cpu_latency_max_us: 2361
- gpu_probe_latency_p50_us: 415820
- gpu_probe_latency_p95_us: 415820
- gpu_probe_latency_max_us: 415820
- gpu_probe_vs_cpu_total_ratio: 176.075
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- h2d_bytes_per_query: 85135.00
- d2h_bytes_total: 57127
- d2h_bytes_per_result_row: 57.13
- kernel_exec_samples: 1
- kernel_exec_total_ms: 413
- correctness_validated: true

#### analytic_range_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 58831
- gpu_probe_total_us: 449449
- cpu_qps: 17.00
- gpu_probe_qps: 2.22
- cpu_latency_p50_us: 58831
- cpu_latency_p95_us: 58831
- cpu_latency_max_us: 58831
- gpu_probe_latency_p50_us: 449448
- gpu_probe_latency_p95_us: 449448
- gpu_probe_latency_max_us: 449448
- gpu_probe_vs_cpu_total_ratio: 7.640
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
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
- cpu_total_us: 30791
- gpu_probe_total_us: 434748
- cpu_qps: 32.48
- gpu_probe_qps: 2.30
- cpu_latency_p50_us: 30790
- cpu_latency_p95_us: 30790
- cpu_latency_max_us: 30790
- gpu_probe_latency_p50_us: 434747
- gpu_probe_latency_p95_us: 434747
- gpu_probe_latency_max_us: 434747
- gpu_probe_vs_cpu_total_ratio: 14.119
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 4388
- h2d_bytes_per_query: 4388.00
- d2h_bytes_total: 1447
- d2h_bytes_per_result_row: 57.88
- kernel_exec_samples: 1
- kernel_exec_total_ms: 418
- correctness_validated: true

#### analytic_disjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 294419
- gpu_probe_total_us: 572745
- cpu_qps: 3.40
- gpu_probe_qps: 1.75
- cpu_latency_p50_us: 294418
- cpu_latency_p95_us: 294418
- cpu_latency_max_us: 294418
- gpu_probe_latency_p50_us: 572744
- gpu_probe_latency_p95_us: 572744
- gpu_probe_latency_max_us: 572744
- gpu_probe_vs_cpu_total_ratio: 1.945
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 42836
- h2d_bytes_per_query: 42836.00
- d2h_bytes_total: 1429
- d2h_bytes_per_result_row: 57.16
- kernel_exec_samples: 1
- kernel_exec_total_ms: 423
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
- cpu_total_us: 5026
- gpu_probe_total_us: 6664985
- cpu_qps: 3183.29
- gpu_probe_qps: 2.40
- cpu_latency_p50_us: 304
- cpu_latency_p95_us: 426
- cpu_latency_max_us: 426
- gpu_probe_latency_p50_us: 413779
- gpu_probe_latency_p95_us: 451303
- gpu_probe_latency_max_us: 451303
- gpu_probe_vs_cpu_total_ratio: 1326.036
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 1485
- h2d_bytes_per_query: 92.81
- d2h_bytes_total: 909
- d2h_bytes_per_result_row: 56.81
- kernel_exec_samples: 16
- kernel_exec_total_ms: 6631
- correctness_validated: true

#### app_batched_or_lookup
- queries: 1
- result_rows: 16
- cpu_total_us: 11238
- gpu_probe_total_us: 428143
- cpu_qps: 88.98
- gpu_probe_qps: 2.34
- cpu_latency_p50_us: 11238
- cpu_latency_p95_us: 11238
- cpu_latency_max_us: 11238
- gpu_probe_latency_p50_us: 428142
- gpu_probe_latency_p95_us: 428142
- gpu_probe_latency_max_us: 428142
- gpu_probe_vs_cpu_total_ratio: 38.095
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 1365
- h2d_bytes_per_query: 1365.00
- d2h_bytes_total: 909
- d2h_bytes_per_result_row: 56.81
- kernel_exec_samples: 1
- kernel_exec_total_ms: 420
- correctness_validated: true

#### analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2382
- gpu_probe_total_us: 421345
- cpu_qps: 419.74
- gpu_probe_qps: 2.37
- cpu_latency_p50_us: 2381
- cpu_latency_p95_us: 2381
- cpu_latency_max_us: 2381
- gpu_probe_latency_p50_us: 421344
- gpu_probe_latency_p95_us: 421344
- gpu_probe_latency_max_us: 421344
- gpu_probe_vs_cpu_total_ratio: 176.857
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
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
- cpu_total_us: 58970
- gpu_probe_total_us: 449324
- cpu_qps: 16.96
- gpu_probe_qps: 2.23
- cpu_latency_p50_us: 58969
- cpu_latency_p95_us: 58969
- cpu_latency_max_us: 58969
- gpu_probe_latency_p50_us: 449323
- gpu_probe_latency_p95_us: 449323
- gpu_probe_latency_max_us: 449323
- gpu_probe_vs_cpu_total_ratio: 7.619
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
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
- cpu_total_us: 30995
- gpu_probe_total_us: 431959
- cpu_qps: 32.26
- gpu_probe_qps: 2.32
- cpu_latency_p50_us: 30994
- cpu_latency_p95_us: 30994
- cpu_latency_max_us: 30994
- gpu_probe_latency_p50_us: 431957
- gpu_probe_latency_p95_us: 431957
- gpu_probe_latency_max_us: 431957
- gpu_probe_vs_cpu_total_ratio: 13.936
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 4388
- h2d_bytes_per_query: 4388.00
- d2h_bytes_total: 1447
- d2h_bytes_per_result_row: 57.88
- kernel_exec_samples: 1
- kernel_exec_total_ms: 415
- correctness_validated: true

#### analytic_disjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 293521
- gpu_probe_total_us: 571950
- cpu_qps: 3.41
- gpu_probe_qps: 1.75
- cpu_latency_p50_us: 293520
- cpu_latency_p95_us: 293520
- cpu_latency_max_us: 293520
- gpu_probe_latency_p50_us: 571948
- gpu_probe_latency_p95_us: 571948
- gpu_probe_latency_max_us: 571948
- gpu_probe_vs_cpu_total_ratio: 1.949
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 42836
- h2d_bytes_per_query: 42836.00
- d2h_bytes_total: 1429
- d2h_bytes_per_result_row: 57.16
- kernel_exec_samples: 1
- kernel_exec_total_ms: 424
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
- cpu_total_us: 19710
- gpu_probe_total_us: 26320058
- cpu_qps: 3247.04
- gpu_probe_qps: 2.43
- cpu_latency_p50_us: 291
- cpu_latency_p95_us: 351
- cpu_latency_max_us: 751
- gpu_probe_latency_p50_us: 409706
- gpu_probe_latency_p95_us: 422380
- gpu_probe_latency_max_us: 460392
- gpu_probe_vs_cpu_total_ratio: 1335.346
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 5959
- h2d_bytes_per_query: 93.11
- d2h_bytes_total: 3655
- d2h_bytes_per_result_row: 57.11
- kernel_exec_samples: 64
- kernel_exec_total_ms: 26232
- correctness_validated: true

#### app_batched_or_lookup
- queries: 1
- result_rows: 64
- cpu_total_us: 40016
- gpu_probe_total_us: 434252
- cpu_qps: 24.99
- gpu_probe_qps: 2.30
- cpu_latency_p50_us: 40015
- cpu_latency_p95_us: 40015
- cpu_latency_max_us: 40015
- gpu_probe_latency_p50_us: 434250
- gpu_probe_latency_p95_us: 434250
- gpu_probe_latency_max_us: 434250
- gpu_probe_vs_cpu_total_ratio: 10.852
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 5455
- h2d_bytes_per_query: 5455.00
- d2h_bytes_total: 3655
- d2h_bytes_per_result_row: 57.11
- kernel_exec_samples: 1
- kernel_exec_total_ms: 412
- correctness_validated: true

#### analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2399
- gpu_probe_total_us: 426110
- cpu_qps: 416.70
- gpu_probe_qps: 2.35
- cpu_latency_p50_us: 2399
- cpu_latency_p95_us: 2399
- cpu_latency_max_us: 2399
- gpu_probe_latency_p50_us: 426108
- gpu_probe_latency_p95_us: 426108
- gpu_probe_latency_max_us: 426108
- gpu_probe_vs_cpu_total_ratio: 177.558
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- h2d_bytes_per_query: 85135.00
- d2h_bytes_total: 57127
- d2h_bytes_per_result_row: 57.13
- kernel_exec_samples: 1
- kernel_exec_total_ms: 424
- correctness_validated: true

#### analytic_range_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 58783
- gpu_probe_total_us: 448907
- cpu_qps: 17.01
- gpu_probe_qps: 2.23
- cpu_latency_p50_us: 58782
- cpu_latency_p95_us: 58782
- cpu_latency_max_us: 58782
- gpu_probe_latency_p50_us: 448906
- gpu_probe_latency_p95_us: 448906
- gpu_probe_latency_max_us: 448906
- gpu_probe_vs_cpu_total_ratio: 7.637
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
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
- cpu_total_us: 30867
- gpu_probe_total_us: 436656
- cpu_qps: 32.40
- gpu_probe_qps: 2.29
- cpu_latency_p50_us: 30866
- cpu_latency_p95_us: 30866
- cpu_latency_max_us: 30866
- gpu_probe_latency_p50_us: 436655
- gpu_probe_latency_p95_us: 436655
- gpu_probe_latency_max_us: 436655
- gpu_probe_vs_cpu_total_ratio: 14.146
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
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
- cpu_total_us: 293083
- gpu_probe_total_us: 567803
- cpu_qps: 3.41
- gpu_probe_qps: 1.76
- cpu_latency_p50_us: 293083
- cpu_latency_p95_us: 293083
- cpu_latency_max_us: 293083
- gpu_probe_latency_p50_us: 567802
- gpu_probe_latency_p95_us: 567802
- gpu_probe_latency_max_us: 567802
- gpu_probe_vs_cpu_total_ratio: 1.937
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 42836
- h2d_bytes_per_query: 42836.00
- d2h_bytes_total: 1429
- d2h_bytes_per_result_row: 57.16
- kernel_exec_samples: 1
- kernel_exec_total_ms: 418
- correctness_validated: true

decision: batching lookup predicates into one supported OR query reduces GPU probe latency versus repeated point lookups, but analytical scans still do not beat CPU; prioritize batching plus transfer layout before making broad performance claims.

