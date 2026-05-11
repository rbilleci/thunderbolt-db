# P7 Relational Workload Benchmark - 2026-05-11

Command:

```bash
scripts/run_p7_relational_benchmark.sh
```

Environment:

- dataset_rows: 1000
- concurrency: 1
- device_info: NVIDIA GeForce RTX 3090, 595.58.03

## app_indexed_point_lookup

- queries: 16
- result_rows: 16
- cpu_total_us: 4894
- gpu_probe_total_us: 6656925
- cpu_qps: 3268.81
- gpu_probe_qps: 2.40
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 10000
- h2d_bytes_total: 0
- d2h_bytes_total: 909
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

## analytic_full_table_scan

- queries: 1
- result_rows: 1000
- cpu_total_us: 2371
- gpu_probe_total_us: 411756
- cpu_qps: 421.67
- gpu_probe_qps: 2.43
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 0
- d2h_bytes_total: 57127
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

## Decision

The analytical scan reaches GPU execution and validates correctness, but it does not beat the CPU baseline in this run. The app-shaped indexed lookup workload still records host SQL-finalization fallback even when the underlying MVCC fetch reports GPU execution.

Supported performance claim: current P7 evidence proves reproducible routing and correctness measurement for two relational workload shapes, not workload-level GPU advantage.

Named follow-up: prioritize transfer layout, batching, SQL predicate/order/projection pushdown, and publishing CUDA driver timing into engine metrics before making broad performance claims.
