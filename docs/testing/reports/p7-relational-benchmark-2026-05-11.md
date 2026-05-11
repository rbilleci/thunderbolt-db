# P7 Relational Workload Benchmark

- dataset_rows: 1000
- concurrency: 1
- device_info: NVIDIA GeForce RTX 3090, 595.58.03

## app_indexed_point_lookup
- queries: 16
- result_rows: 16
- cpu_total_us: 5022
- gpu_probe_total_us: 6715464
- cpu_qps: 3185.64
- gpu_probe_qps: 2.38
- cpu_latency_p50_us: 297
- cpu_latency_p95_us: 446
- cpu_latency_max_us: 446
- gpu_probe_latency_p50_us: 415986
- gpu_probe_latency_p95_us: 479542
- gpu_probe_latency_max_us: 479542
- gpu_probe_vs_cpu_total_ratio: 1337.067
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 1362160
- d2h_bytes_total: 909
- kernel_exec_samples: 16
- kernel_exec_total_ms: 6666
- correctness_validated: true

## app_batched_or_lookup
- queries: 1
- result_rows: 16
- cpu_total_us: 11220
- gpu_probe_total_us: 6582943
- cpu_qps: 89.12
- gpu_probe_qps: 0.15
- cpu_latency_p50_us: 11220
- cpu_latency_p95_us: 11220
- cpu_latency_max_us: 11220
- gpu_probe_latency_p50_us: 6582942
- gpu_probe_latency_p95_us: 6582942
- gpu_probe_latency_max_us: 6582942
- gpu_probe_vs_cpu_total_ratio: 586.669
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 909
- kernel_exec_samples: 1
- kernel_exec_total_ms: 6575
- correctness_validated: true

## analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2345
- gpu_probe_total_us: 417386
- cpu_qps: 426.39
- gpu_probe_qps: 2.40
- cpu_latency_p50_us: 2344
- cpu_latency_p95_us: 2344
- cpu_latency_max_us: 2344
- gpu_probe_latency_p50_us: 417385
- gpu_probe_latency_p95_us: 417385
- gpu_probe_latency_max_us: 417385
- gpu_probe_vs_cpu_total_ratio: 177.971
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 57127
- kernel_exec_samples: 1
- kernel_exec_total_ms: 415
- correctness_validated: true

## analytic_range_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 58887
- gpu_probe_total_us: 41340492
- cpu_qps: 16.98
- gpu_probe_qps: 0.02
- cpu_latency_p50_us: 58886
- cpu_latency_p95_us: 58886
- cpu_latency_max_us: 58886
- gpu_probe_latency_p50_us: 41340491
- gpu_probe_latency_p95_us: 41340491
- gpu_probe_latency_max_us: 41340491
- gpu_probe_vs_cpu_total_ratio: 702.027
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 1440
- kernel_exec_samples: 1
- kernel_exec_total_ms: 41309
- correctness_validated: true

## analytic_conjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 30760
- gpu_probe_total_us: 20969701
- cpu_qps: 32.51
- gpu_probe_qps: 0.05
- cpu_latency_p50_us: 30759
- cpu_latency_p95_us: 30759
- cpu_latency_max_us: 30759
- gpu_probe_latency_p50_us: 20969700
- gpu_probe_latency_p95_us: 20969700
- gpu_probe_latency_max_us: 20969700
- gpu_probe_vs_cpu_total_ratio: 681.709
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 1447
- kernel_exec_samples: 1
- kernel_exec_total_ms: 20953
- correctness_validated: true

## analytic_disjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 292865
- gpu_probe_total_us: 207879009
- cpu_qps: 3.41
- gpu_probe_qps: 0.00
- cpu_latency_p50_us: 292864
- cpu_latency_p95_us: 292864
- cpu_latency_max_us: 292864
- gpu_probe_latency_p50_us: 207879008
- gpu_probe_latency_p95_us: 207879008
- gpu_probe_latency_max_us: 207879008
- gpu_probe_vs_cpu_total_ratio: 709.811
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 1429
- kernel_exec_samples: 1
- kernel_exec_total_ms: 207732
- correctness_validated: true

decision: batching lookup predicates into one supported OR query reduces GPU probe latency versus repeated point lookups, but analytical scans still do not beat CPU; prioritize batching plus transfer layout before making broad performance claims.
