# P7 Relational Workload Benchmark

- dataset_rows: 1000
- concurrency: 1
- device_info: NVIDIA GeForce RTX 3090, 595.58.03
- cuda_driver_probe_runtime_cache: per_engine

## app_indexed_point_lookup
- queries: 16
- result_rows: 16
- cpu_total_us: 5084
- gpu_probe_total_us: 6666015
- cpu_qps: 3147.08
- gpu_probe_qps: 2.40
- cpu_latency_p50_us: 306
- cpu_latency_p95_us: 445
- cpu_latency_max_us: 445
- gpu_probe_latency_p50_us: 410023
- gpu_probe_latency_p95_us: 502637
- gpu_probe_latency_max_us: 502637
- gpu_probe_vs_cpu_total_ratio: 1311.154
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 1362160
- h2d_bytes_per_query: 85135.00
- d2h_bytes_total: 909
- d2h_bytes_per_result_row: 56.81
- kernel_exec_samples: 16
- kernel_exec_total_ms: 6615
- correctness_validated: true

## app_batched_or_lookup
- queries: 1
- result_rows: 16
- cpu_total_us: 11226
- gpu_probe_total_us: 6560757
- cpu_qps: 89.07
- gpu_probe_qps: 0.15
- cpu_latency_p50_us: 11225
- cpu_latency_p95_us: 11225
- cpu_latency_max_us: 11225
- gpu_probe_latency_p50_us: 6560756
- gpu_probe_latency_p95_us: 6560756
- gpu_probe_latency_max_us: 6560756
- gpu_probe_vs_cpu_total_ratio: 584.399
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- h2d_bytes_per_query: 85135.00
- d2h_bytes_total: 909
- d2h_bytes_per_result_row: 56.81
- kernel_exec_samples: 1
- kernel_exec_total_ms: 6553
- correctness_validated: true

## analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2365
- gpu_probe_total_us: 437278
- cpu_qps: 422.74
- gpu_probe_qps: 2.29
- cpu_latency_p50_us: 2364
- cpu_latency_p95_us: 2364
- cpu_latency_max_us: 2364
- gpu_probe_latency_p50_us: 437277
- gpu_probe_latency_p95_us: 437277
- gpu_probe_latency_max_us: 437277
- gpu_probe_vs_cpu_total_ratio: 184.856
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- h2d_bytes_per_query: 85135.00
- d2h_bytes_total: 57127
- d2h_bytes_per_result_row: 57.13
- kernel_exec_samples: 1
- kernel_exec_total_ms: 435
- correctness_validated: true

## analytic_range_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 59252
- gpu_probe_total_us: 41505415
- cpu_qps: 16.88
- gpu_probe_qps: 0.02
- cpu_latency_p50_us: 59251
- cpu_latency_p95_us: 59251
- cpu_latency_max_us: 59251
- gpu_probe_latency_p50_us: 41505414
- gpu_probe_latency_p95_us: 41505414
- gpu_probe_latency_max_us: 41505414
- gpu_probe_vs_cpu_total_ratio: 700.484
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- h2d_bytes_per_query: 85135.00
- d2h_bytes_total: 1440
- d2h_bytes_per_result_row: 57.60
- kernel_exec_samples: 1
- kernel_exec_total_ms: 41474
- correctness_validated: true

## analytic_conjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 31123
- gpu_probe_total_us: 20956485
- cpu_qps: 32.13
- gpu_probe_qps: 0.05
- cpu_latency_p50_us: 31122
- cpu_latency_p95_us: 31122
- cpu_latency_max_us: 31122
- gpu_probe_latency_p50_us: 20956484
- gpu_probe_latency_p95_us: 20956484
- gpu_probe_latency_max_us: 20956484
- gpu_probe_vs_cpu_total_ratio: 673.337
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- h2d_bytes_per_query: 85135.00
- d2h_bytes_total: 1447
- d2h_bytes_per_result_row: 57.88
- kernel_exec_samples: 1
- kernel_exec_total_ms: 20939
- correctness_validated: true

## analytic_disjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 296707
- gpu_probe_total_us: 208007128
- cpu_qps: 3.37
- gpu_probe_qps: 0.00
- cpu_latency_p50_us: 296706
- cpu_latency_p95_us: 296706
- cpu_latency_max_us: 296706
- gpu_probe_latency_p50_us: 208007126
- gpu_probe_latency_p95_us: 208007126
- gpu_probe_latency_max_us: 208007126
- gpu_probe_vs_cpu_total_ratio: 701.051
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- h2d_bytes_per_query: 85135.00
- d2h_bytes_total: 1429
- d2h_bytes_per_result_row: 57.16
- kernel_exec_samples: 1
- kernel_exec_total_ms: 207857
- correctness_validated: true

decision: batching lookup predicates into one supported OR query reduces GPU probe latency versus repeated point lookups, but analytical scans still do not beat CPU; prioritize batching plus transfer layout before making broad performance claims.
