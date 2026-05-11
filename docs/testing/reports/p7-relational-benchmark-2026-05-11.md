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
- cpu_total_us: 5914
- gpu_probe_total_us: 6780205
- cpu_qps: 2705.24
- gpu_probe_qps: 2.36
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
- cpu_total_us: 2341
- gpu_probe_total_us: 417580
- cpu_qps: 427.12
- gpu_probe_qps: 2.39
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 0
- d2h_bytes_total: 57127
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

## Decision

The indexed lookup and analytical scan workloads both reach GPU execution with no SQL fallback for the current benchmark mix after equality predicate bridge pushdown and bridge-level decoded-column ordering pushdown, but neither workload beats the CPU baseline in this run.

Supported performance claim: current P7 evidence proves reproducible routing and correctness measurement for two relational workload shapes, not workload-level GPU advantage.

Named follow-up: prioritize transfer layout, batching, and publishing CUDA driver timing into engine metrics before making broad performance claims. Projection-only SQL result shaping and supported decoded-column ordering are now treated as bridge-level result shaping rather than GPU fallback when the relational row fetch already executed on GPU.
