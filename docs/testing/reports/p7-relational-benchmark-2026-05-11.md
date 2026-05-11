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
- cpu_total_us: 5255
- gpu_probe_total_us: 6668727
- cpu_qps: 3044.37
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
- cpu_total_us: 2354
- gpu_probe_total_us: 398459
- cpu_qps: 424.64
- gpu_probe_qps: 2.51
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
- cpu_total_us: 58904
- gpu_probe_total_us: 41313704
- cpu_qps: 16.98
- gpu_probe_qps: 0.02
- gpu_executed_rate_permyriad: 10000
- cpu_fallback_rate_permyriad: 0
- h2d_bytes_total: 0
- d2h_bytes_total: 1440
- kernel_exec_samples: 0
- kernel_exec_total_ms: 0
- correctness_validated: true

## Decision

The indexed lookup, analytical scan, and analytical range-filter workloads all reach GPU execution with no SQL fallback for the current benchmark mix after equality predicate bridge pushdown, bridge-level decoded-column ordering pushdown, and range predicate key-batch bridging, but none of the workloads beats the CPU baseline in this run.

Supported performance claim: current P7 evidence proves reproducible routing and correctness measurement for three relational workload shapes, not workload-level GPU advantage.

Named follow-up: prioritize transfer layout, batching, multi-predicate/expression pushdown, and publishing CUDA driver timing into engine metrics before making broad performance claims. Projection-only SQL result shaping, supported decoded-column ordering, and narrow range predicates are now treated as bridge-supported result/source shaping rather than GPU fallback when the relational row fetch executes on GPU.
