# P7 Relational Workload Benchmark

- dataset_rows: 1000
- concurrency: 1
- device_info: NVIDIA GeForce RTX 3090, 595.58.03

## app_indexed_point_lookup
- queries: 16
- result_rows: 16
- cpu_total_us: 4956
- gpu_probe_total_us: 6708673
- cpu_qps: 3227.77
- gpu_probe_qps: 2.38
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 1362160
- d2h_bytes_total: 909
- kernel_exec_samples: 16
- kernel_exec_total_ms: 6658
- correctness_validated: true

## analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2418
- gpu_probe_total_us: 425743
- cpu_qps: 413.45
- gpu_probe_qps: 2.35
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 57127
- kernel_exec_samples: 1
- kernel_exec_total_ms: 423
- correctness_validated: true

## analytic_range_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 59641
- gpu_probe_total_us: 41599880
- cpu_qps: 16.77
- gpu_probe_qps: 0.02
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 1440
- kernel_exec_samples: 1
- kernel_exec_total_ms: 41568
- correctness_validated: true

## analytic_conjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 31331
- gpu_probe_total_us: 20942888
- cpu_qps: 31.92
- gpu_probe_qps: 0.05
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 1447
- kernel_exec_samples: 1
- kernel_exec_total_ms: 20925
- correctness_validated: true

## analytic_disjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 299288
- gpu_probe_total_us: 208370259
- cpu_qps: 3.34
- gpu_probe_qps: 0.00
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 1429
- kernel_exec_samples: 1
- kernel_exec_total_ms: 208217
- correctness_validated: true

decision: analytical scans reach GPU execution with SQL-level transfer and timing telemetry but do not yet beat the CPU baseline in this run; prioritize transfer layout, batching, and driver-level timing refinement before making broad performance claims.
