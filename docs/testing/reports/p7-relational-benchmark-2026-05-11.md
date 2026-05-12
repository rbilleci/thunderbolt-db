# P7 Relational Workload Benchmark

- dataset_rows: 1000
- concurrency: 1
- device_info: NVIDIA GeForce RTX 3090, 595.58.03
- cuda_driver_probe_runtime_cache: per_engine

## app_indexed_point_lookup
- queries: 16
- result_rows: 16
- cpu_total_us: 4968
- gpu_probe_total_us: 6671874
- cpu_qps: 3219.98
- gpu_probe_qps: 2.40
- cpu_latency_p50_us: 300
- cpu_latency_p95_us: 430
- cpu_latency_max_us: 430
- gpu_probe_latency_p50_us: 411150
- gpu_probe_latency_p95_us: 492198
- gpu_probe_latency_max_us: 492198
- gpu_probe_vs_cpu_total_ratio: 1342.708
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 1485
- h2d_bytes_per_query: 92.81
- d2h_bytes_total: 909
- d2h_bytes_per_result_row: 56.81
- kernel_exec_samples: 16
- kernel_exec_total_ms: 6620
- correctness_validated: true

## app_batched_or_lookup
- queries: 1
- result_rows: 16
- cpu_total_us: 11200
- gpu_probe_total_us: 430670
- cpu_qps: 89.28
- gpu_probe_qps: 2.32
- cpu_latency_p50_us: 11199
- cpu_latency_p95_us: 11199
- cpu_latency_max_us: 11199
- gpu_probe_latency_p50_us: 430668
- gpu_probe_latency_p95_us: 430668
- gpu_probe_latency_max_us: 430668
- gpu_probe_vs_cpu_total_ratio: 38.452
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 1365
- h2d_bytes_per_query: 1365.00
- d2h_bytes_total: 909
- d2h_bytes_per_result_row: 56.81
- kernel_exec_samples: 1
- kernel_exec_total_ms: 423
- correctness_validated: true

## analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2444
- gpu_probe_total_us: 424267
- cpu_qps: 409.00
- gpu_probe_qps: 2.36
- cpu_latency_p50_us: 2444
- cpu_latency_p95_us: 2444
- cpu_latency_max_us: 2444
- gpu_probe_latency_p50_us: 424265
- gpu_probe_latency_p95_us: 424265
- gpu_probe_latency_max_us: 424265
- gpu_probe_vs_cpu_total_ratio: 173.525
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- h2d_bytes_per_query: 85135.00
- d2h_bytes_total: 57127
- d2h_bytes_per_result_row: 57.13
- kernel_exec_samples: 1
- kernel_exec_total_ms: 422
- correctness_validated: true

## analytic_range_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 59076
- gpu_probe_total_us: 449203
- cpu_qps: 16.93
- gpu_probe_qps: 2.23
- cpu_latency_p50_us: 59076
- cpu_latency_p95_us: 59076
- cpu_latency_max_us: 59076
- gpu_probe_latency_p50_us: 449201
- gpu_probe_latency_p95_us: 449201
- gpu_probe_latency_max_us: 449201
- gpu_probe_vs_cpu_total_ratio: 7.604
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 8630
- h2d_bytes_per_query: 8630.00
- d2h_bytes_total: 1440
- d2h_bytes_per_result_row: 57.60
- kernel_exec_samples: 1
- kernel_exec_total_ms: 418
- correctness_validated: true

## analytic_conjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 30789
- gpu_probe_total_us: 431079
- cpu_qps: 32.48
- gpu_probe_qps: 2.32
- cpu_latency_p50_us: 30789
- cpu_latency_p95_us: 30789
- cpu_latency_max_us: 30789
- gpu_probe_latency_p50_us: 431077
- gpu_probe_latency_p95_us: 431077
- gpu_probe_latency_max_us: 431077
- gpu_probe_vs_cpu_total_ratio: 14.001
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 4388
- h2d_bytes_per_query: 4388.00
- d2h_bytes_total: 1447
- d2h_bytes_per_result_row: 57.88
- kernel_exec_samples: 1
- kernel_exec_total_ms: 414
- correctness_validated: true

## analytic_disjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 296664
- gpu_probe_total_us: 579004
- cpu_qps: 3.37
- gpu_probe_qps: 1.73
- cpu_latency_p50_us: 296663
- cpu_latency_p95_us: 296663
- cpu_latency_max_us: 296663
- gpu_probe_latency_p50_us: 579003
- gpu_probe_latency_p95_us: 579003
- gpu_probe_latency_max_us: 579003
- gpu_probe_vs_cpu_total_ratio: 1.952
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 42836
- h2d_bytes_per_query: 42836.00
- d2h_bytes_total: 1429
- d2h_bytes_per_result_row: 57.16
- kernel_exec_samples: 1
- kernel_exec_total_ms: 430
- correctness_validated: true

decision: batching lookup predicates into one supported OR query reduces GPU probe latency versus repeated point lookups, but analytical scans still do not beat CPU; prioritize batching plus transfer layout before making broad performance claims.
