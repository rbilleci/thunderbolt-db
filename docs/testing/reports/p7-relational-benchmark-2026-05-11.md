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
- cpu_total_us: 4895
- gpu_probe_total_us: 6666146
- cpu_qps: 3268.51
- gpu_probe_qps: 2.40
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
- cpu_total_us: 2368
- gpu_probe_total_us: 416343
- cpu_qps: 422.24
- gpu_probe_qps: 2.40
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
- cpu_total_us: 58714
- gpu_probe_total_us: 41392057
- cpu_qps: 17.03
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
- cpu_total_us: 30850
- gpu_probe_total_us: 20893546
- cpu_qps: 32.41
- gpu_probe_qps: 0.05
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 0
- d2h_bytes_total: 1447
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

## Decision

The indexed lookup, analytical scan, analytical range-filter, and analytical conjunctive-filter workloads all reach GPU execution with no SQL fallback for the current benchmark mix after equality predicate bridge pushdown, bridge-level decoded-column ordering pushdown, range predicate key-batch bridging, and conjunction key-batch bridging, but none of the workloads beats the CPU baseline in this run.

Supported performance claim: current P7 evidence proves reproducible routing and correctness measurement for four relational workload shapes, not workload-level GPU advantage.

Named follow-up: prioritize transfer layout, batching, richer expression/boolean pushdown, and publishing CUDA driver timing into engine metrics before making broad performance claims. Projection-only SQL result shaping, supported decoded-column ordering, narrow range predicates, and narrow `AND` predicate conjunctions are now treated as bridge-supported result/source shaping rather than GPU fallback when the relational row fetch executes on GPU.
