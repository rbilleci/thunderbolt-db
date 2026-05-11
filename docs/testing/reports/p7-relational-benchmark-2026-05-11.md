# P7 Relational Workload Benchmark

- dataset_rows: 1000
- concurrency: 1
- device_info: NVIDIA GeForce RTX 3090, 595.58.03

## app_indexed_point_lookup
- queries: 16
- result_rows: 16
- cpu_total_us: 5997
- gpu_probe_total_us: 6862369
- cpu_qps: 2667.59
- gpu_probe_qps: 2.33
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 1362160
- d2h_bytes_total: 909
- kernel_exec_samples: 16
- kernel_exec_total_ms: 6806
- correctness_validated: true

## app_batched_or_lookup
- queries: 1
- result_rows: 16
- cpu_total_us: 11447
- gpu_probe_total_us: 6657467
- cpu_qps: 87.36
- gpu_probe_qps: 0.15
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 909
- kernel_exec_samples: 1
- kernel_exec_total_ms: 6650
- correctness_validated: true

## analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2389
- gpu_probe_total_us: 422704
- cpu_qps: 418.45
- gpu_probe_qps: 2.37
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 57127
- kernel_exec_samples: 1
- kernel_exec_total_ms: 420
- correctness_validated: true

## analytic_range_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 60552
- gpu_probe_total_us: 41453909
- cpu_qps: 16.51
- gpu_probe_qps: 0.02
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 1440
- kernel_exec_samples: 1
- kernel_exec_total_ms: 41422
- correctness_validated: true

## analytic_conjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 31506
- gpu_probe_total_us: 21012097
- cpu_qps: 31.74
- gpu_probe_qps: 0.05
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 1447
- kernel_exec_samples: 1
- kernel_exec_total_ms: 20994
- correctness_validated: true

## analytic_disjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 301707
- gpu_probe_total_us: 207824511
- cpu_qps: 3.31
- gpu_probe_qps: 0.00
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 1429
- kernel_exec_samples: 1
- kernel_exec_total_ms: 207671
- correctness_validated: true

decision: batching lookup predicates into one supported OR query reduces GPU probe latency versus repeated point lookups, but analytical scans still do not beat CPU; prioritize batching plus transfer layout before making broad performance claims.
