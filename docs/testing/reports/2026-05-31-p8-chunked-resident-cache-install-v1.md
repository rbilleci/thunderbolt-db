# P8 Chunked Resident-Cache Install Probe

- date: 2026-05-31
- stream: benchmark
- milestone: P8 CH-benCHmark-derived chunked resident-cache install/admission prerequisite
- validation_gate: `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`; `scripts/run_p8_ch_benchmark_residency_probe.sh --self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --streaming-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --chunked-install-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-preflight`; `scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct`; `scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup`; `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`; `cargo clippy --all-features --example p8_ch_benchmark_residency_probe -- -D warnings`; `git diff --check`
- report_artifacts: `target/p8-ch-benchmark-residency/chunked-install-self-check/self-check.md`; `target/p8-ch-benchmark-residency/25pct-preflight.md`; `target/p8-ch-benchmark-residency/pgsql-baseline/preflight.md`

## Work Order

- active_lane: P8 CH-benCHmark-derived benchmark-only chunked resident cache install/admission prerequisite for the 25% / 6 GiB retained-residency tier
- classification: blocked
- falsifiable_claim: generated `order_line` chunks can either be installed into `RelationalResidentCache` without whole-tier `resident_rows` and one whole retained host payload, or the exact remaining cache/device-memory interface blocker can be named
- evidence_required: streaming artifact self-check, focused chunked-install self-check, PostgreSQL baseline gate, 25% preflight decision, cleanup verification, and focused Rust checks
- non_goals: normal SQL durability for generated artifacts, BenchBase compatibility, joins, transaction mix, production cache daemon behavior, durable GPU pages, and 50/100/200% tiers
- minimum_meaningful_chunk: prove a checked chunked-install/admission slice or replace the previous broad blocker with a narrower interface blocker
- stop_rule: stop after the 25% tier can start safely or after the missing interface is narrowed

## Result

The prior broad blocker is narrowed. The generator can produce deterministic
benchmark-only `order_line` chunks under `target/`, and PostgreSQL baseline
evidence remains a required gate. The 25% / 6 GiB tier still must not start,
because the resident device-memory boundary cannot yet admit chunks directly.

The narrower blocker is:

```text
missing_chunked_retained_device_memory_upload_api
```

`RelationalResidencySnapshot` still carries `resident_rows:
Vec<Vec<SqlValue>>` for snapshot-backed reads, and the retained device-memory
path is `CudaDriverRuntime::retain_device_memory_copy(gpu_id, payload: &[u8])`.
That API allocates and copies one contiguous payload. For the 25% tier, a safe
benchmark-only install path needs to allocate the resident layout once and copy
generated column chunks into known device offsets without constructing the whole
host payload first.

## 25% Preflight Decision

`scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct` remains blocked
before heavy generation starts. Disk preflight is expected to pass on this host,
but run preflight now reports the narrower device-memory upload blocker instead
of the older generic chunked cache-install blocker.

## PostgreSQL Baseline Boundary

The PostgreSQL comparator requirement remains intact. The local disposable
Docker path loads and runs the same deterministic small `order_line` workload
through `scripts/run_p8_ch_benchmark_residency_probe.sh
--pgsql-baseline-docker-preflight`. No GPU DB performance result should be
interpreted unless the same dataset/query set has a passed PostgreSQL baseline
artifact or a precise PostgreSQL-baseline blocker.

## Next Defensible Slice

Add a retained device-memory API that can allocate the resident payload layout
once, copy row-count/header, int4 column chunks, text offsets, and text bytes
into explicit offsets, and return a `CudaResidentDeviceMemory` proof compatible
with the existing retained query kernels. After that lands, the benchmark-only
cache install path can avoid whole-tier `resident_rows` and whole-payload host
materialization for the 25% tier.
