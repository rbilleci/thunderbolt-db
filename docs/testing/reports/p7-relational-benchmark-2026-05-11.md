# P7 Relational Workload Benchmark

- dataset_rows: 1000
- concurrency: 1
- device_info: NVIDIA GeForce RTX 3090, 595.58.03

## app_indexed_point_lookup
- queries: 16
- result_rows: 16
- cpu_total_us: 5032
- gpu_probe_total_us: 6771488
- cpu_qps: 3179.27
- gpu_probe_qps: 2.36
- cpu_latency_p50_us: 299
- cpu_latency_p95_us: 433
- cpu_latency_max_us: 433
- gpu_probe_latency_p50_us: 417622
- gpu_probe_latency_p95_us: 504246
- gpu_probe_latency_max_us: 504246
- gpu_probe_vs_cpu_total_ratio: 1345.523
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 1362160
- d2h_bytes_total: 909
- kernel_exec_samples: 16
- kernel_exec_total_ms: 6718
- correctness_validated: true

## app_batched_or_lookup
- queries: 1
- result_rows: 16
- cpu_total_us: 13172
- gpu_probe_total_us: 6626837
- cpu_qps: 75.92
- gpu_probe_qps: 0.15
- cpu_latency_p50_us: 13171
- cpu_latency_p95_us: 13171
- cpu_latency_max_us: 13171
- gpu_probe_latency_p50_us: 6626836
- gpu_probe_latency_p95_us: 6626836
- gpu_probe_latency_max_us: 6626836
- gpu_probe_vs_cpu_total_ratio: 503.088
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 909
- kernel_exec_samples: 1
- kernel_exec_total_ms: 6619
- correctness_validated: true

## analytic_full_table_scan
- queries: 1
- result_rows: 1000
- cpu_total_us: 2380
- gpu_probe_total_us: 414249
- cpu_qps: 420.05
- gpu_probe_qps: 2.41
- cpu_latency_p50_us: 2380
- cpu_latency_p95_us: 2380
- cpu_latency_max_us: 2380
- gpu_probe_latency_p50_us: 414248
- gpu_probe_latency_p95_us: 414248
- gpu_probe_latency_max_us: 414248
- gpu_probe_vs_cpu_total_ratio: 174.004
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 57127
- kernel_exec_samples: 1
- kernel_exec_total_ms: 412
- correctness_validated: true

## analytic_range_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 59049
- gpu_probe_total_us: 41605141
- cpu_qps: 16.93
- gpu_probe_qps: 0.02
- cpu_latency_p50_us: 59049
- cpu_latency_p95_us: 59049
- cpu_latency_max_us: 59049
- gpu_probe_latency_p50_us: 41605140
- gpu_probe_latency_p95_us: 41605140
- gpu_probe_latency_max_us: 41605140
- gpu_probe_vs_cpu_total_ratio: 704.578
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 1440
- kernel_exec_samples: 1
- kernel_exec_total_ms: 41574
- correctness_validated: true

## analytic_conjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 30864
- gpu_probe_total_us: 20965263
- cpu_qps: 32.40
- gpu_probe_qps: 0.05
- cpu_latency_p50_us: 30863
- cpu_latency_p95_us: 30863
- cpu_latency_max_us: 30863
- gpu_probe_latency_p50_us: 20965262
- gpu_probe_latency_p95_us: 20965262
- gpu_probe_latency_max_us: 20965262
- gpu_probe_vs_cpu_total_ratio: 679.270
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 1447
- kernel_exec_samples: 1
- kernel_exec_total_ms: 20948
- correctness_validated: true

## analytic_disjunctive_filter
- queries: 1
- result_rows: 25
- cpu_total_us: 296637
- gpu_probe_total_us: 208553407
- cpu_qps: 3.37
- gpu_probe_qps: 0.00
- cpu_latency_p50_us: 296636
- cpu_latency_p95_us: 296636
- cpu_latency_max_us: 296636
- gpu_probe_latency_p50_us: 208553406
- gpu_probe_latency_p95_us: 208553406
- gpu_probe_latency_max_us: 208553406
- gpu_probe_vs_cpu_total_ratio: 703.058
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 85135
- d2h_bytes_total: 1429
- kernel_exec_samples: 1
- kernel_exec_total_ms: 208400
- correctness_validated: true

decision: batching lookup predicates into one supported OR query reduces GPU probe latency versus repeated point lookups, but analytical scans still do not beat CPU; prioritize batching plus transfer layout before making broad performance claims.
