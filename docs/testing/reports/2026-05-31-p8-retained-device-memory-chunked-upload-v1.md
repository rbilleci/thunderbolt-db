# P8 Retained Device-Memory Chunked Upload Probe

- date: 2026-05-31
- stream: benchmark
- milestone: P8 CH-benCHmark-derived retained CUDA device-memory chunked upload/admission prerequisite
- validation_gate: `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`; `scripts/run_p8_ch_benchmark_residency_probe.sh --self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --streaming-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --chunked-install-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --chunked-upload-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-preflight`; `scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct`; `cargo clippy --all-features --example p8_ch_benchmark_residency_probe -- -D warnings`; `git diff --check`
- report_artifacts: `target/p8-ch-benchmark-residency/chunked-upload-self-check/self-check.md`; `target/p8-ch-benchmark-residency/chunked-install-self-check/self-check.md`; `target/p8-ch-benchmark-residency/25pct-preflight.md`; `target/p8-ch-benchmark-residency/pgsql-baseline/preflight.md`

## Work Order

- active_lane: P8 CH-benCHmark-derived retained CUDA device-memory chunked upload/admission prerequisite for the 25% / 6 GiB retained-residency tier
- classification: blocked
- falsifiable_claim: the execution layer can either allocate one retained CUDA resident layout and copy generated header, int4-column, text-offset, and text-byte chunks into explicit offsets without one contiguous host payload, or name the narrower missing retained-kernel/runtime/cache interface
- evidence_required: focused CUDA/runtime chunked-upload self-check, streaming artifact self-check, chunked-install self-check, PostgreSQL baseline gate, 25% preflight decision, cleanup verification, and focused Rust checks
- non_goals: normal SQL durability for generated artifacts, full CH-benCHmark/BenchBase compatibility, joins, transaction mix, production cache daemon behavior, durable GPU pages, and 50/100/200% tiers
- minimum_meaningful_chunk: prove the retained device-memory upload boundary or replace `missing_chunked_retained_device_memory_upload_api` with a narrower blocker
- stop_rule: stop after the 25% tier can start safely or after a narrower cache/admission blocker replaces the device-memory upload blocker

## Result

The retained CUDA device-memory upload API boundary is now closed for the
benchmark layout. `CudaDriverRuntime::retain_device_memory_chunks(...)` allocates
the final resident byte length once and copies caller-provided chunks into
explicit device offsets. The self-check copies the row-count header, four int4
`order_line` columns, the text-offset vector, and text bytes without first
assembling one full host payload, then verifies the existing retained kernels
can read the uploaded layout.

The chunked upload self-check passed on the local RTX 3090 with:

- rows: 64
- copied_chunks: 24
- allocated_bytes: 1936
- copied_bytes: 1936
- row_count_kernel: 64
- amount_17_count_kernel: 1
- alpha_prefix_count_kernel: 32

The remaining blocker is now narrower:

```text
missing_benchmark_only_relational_resident_cache_chunked_admission_api
```

`RelationalResidencySnapshot` still carries `resident_rows:
Vec<Vec<SqlValue>>` for snapshot-backed correctness and route admission. A safe
25% / 6 GiB path still needs a benchmark-only `RelationalResidentCache`
admission boundary that consumes deterministic generated artifacts under
`target/`, preserves the existing retained-kernel layout, and avoids whole-tier
row materialization without claiming normal SQL insertion/MVCC durability.

## 25% Preflight Decision

`scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct` remains blocked
before heavy generation starts. Disk preflight passed on this host with
377554292736 available bytes versus 23194920608 required bytes, but run
preflight now reports
`missing_benchmark_only_relational_resident_cache_chunked_admission_api` instead
of the prior device-memory upload blocker.

## PostgreSQL Baseline Boundary

The PostgreSQL comparator requirement remains intact. The disposable Docker
baseline path passed with PostgreSQL 16.14 through
`scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-preflight`
and wrote the same deterministic `order_line` workload SQL/output under
`target/p8-ch-benchmark-residency/pgsql-baseline/`. No GPU DB performance result
should be interpreted unless that same dataset/query set has a passed PostgreSQL
baseline artifact or a precise PostgreSQL-baseline blocker.

## Next Defensible Slice

Add a benchmark-only resident-cache admission API that installs generated
`order_line` chunks into a `RelationalResidentCache` entry and retained
device-memory handle without constructing whole-tier `resident_rows`, while
keeping small snapshot-backed correctness self-checks and preserving the
PostgreSQL baseline and cleanup gates.
