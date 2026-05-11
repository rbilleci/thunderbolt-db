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
- cpu_total_us: 5969
- gpu_probe_total_us: 6716333
- cpu_qps: 2680.51
- gpu_probe_qps: 2.38
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
- cpu_total_us: 2380
- gpu_probe_total_us: 434839
- cpu_qps: 419.99
- gpu_probe_qps: 2.30
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 0
- d2h_bytes_total: 57127
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

## Decision

The indexed lookup and analytical scan workloads both reach GPU execution with no SQL fallback for the current benchmark mix after equality predicate bridge pushdown, but neither workload beats the CPU baseline in this run.

Supported performance claim: current P7 evidence proves reproducible routing and correctness measurement for two relational workload shapes, not workload-level GPU advantage.

Named follow-up: prioritize transfer layout, batching, remaining SQL order/projection pushdown, and publishing CUDA driver timing into engine metrics before making broad performance claims.
