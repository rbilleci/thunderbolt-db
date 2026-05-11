# P7 Relational Workload Benchmark

- dataset_rows: 1000
- concurrency: 1
- device_info: NVIDIA GeForce RTX 3090, 595.58.03

## app_indexed_point_lookup
- queries: 16
- result_rows: 16
- cpu_total_us: 5023
- gpu_probe_total_us: 6695393
- cpu_qps: 3185.08
- gpu_probe_qps: 2.39
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 0
- d2h_bytes_total: 909
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

## analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2370
- gpu_probe_total_us: 425174
- cpu_qps: 421.77
- gpu_probe_qps: 2.35
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 0
- d2h_bytes_total: 57127
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

## analytic_range_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 60494
- gpu_probe_total_us: 41430289
- cpu_qps: 16.53
- gpu_probe_qps: 0.02
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 0
- d2h_bytes_total: 1440
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

## analytic_conjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 31476
- gpu_probe_total_us: 20960351
- cpu_qps: 31.77
- gpu_probe_qps: 0.05
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 0
- d2h_bytes_total: 1447
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

## analytic_disjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 299460
- gpu_probe_total_us: 208689656
- cpu_qps: 3.34
- gpu_probe_qps: 0.00
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 0
- d2h_bytes_total: 1429
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

decision: analytical scans reach GPU execution but do not yet beat the CPU baseline in this run; prioritize transfer layout, batching, and CUDA timing publication before making broad performance claims.
