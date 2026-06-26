# P8 Streaming Full 25% Boundary Report

- date: 2026-05-31
- stream: benchmark
- milestone: P8 CH-benCHmark-derived 25% / 6 GiB retained-residency streaming full-tier boundary
- status: blocked
- implementation_result: streaming upload/admission boundary landed
- narrowed_blocker: `docker_cleanup_permission_denied_for_new_pgsql_baseline_container`
- next_full_tier_blocker: `full_25pct_requires_operator_long_run_after_streaming_boundary`

## Work Order

- active_lane: P8 25% / 6 GiB retained-residency full-tier execution boundary
- lane_classification: open
- falsifiable_claim: the harness can stream generated resident chunks through upload/admission without a full caller-owned chunk vector/text layout, or identify the exact remaining blocker
- evidence_required: scaled streaming execution, formula validation, PostgreSQL comparator lifecycle, 25% preflight, cleanup verification, focused Rust checks, and source-truth reconciliation
- non_goals: 50/100/200% tiers, the removed 400% tier, full CH-benCHmark/BenchBase compatibility, joins, transaction mix, normal SQL durability for generated artifacts, durable GPU pages, production cache-daemon scheduling, external orchestration, and kernel optimization
- minimum_meaningful_chunk: one checked streaming upload/admission boundary plus either a full 25% report or a narrower blocker
- stop_rule: stop after a checked full 25% report or one precise blocker

## Result

The implementation removes the prior host-memory boundary. The execution layer
now accepts `CudaOwnedDeviceMemoryChunk` iterators, and the engine exposes
`install_benchmark_relational_residency_owned_chunks(...)` for benchmark-only
resident-cache admission. The P8 example now derives resident layout metadata
from formulas and streams header, int4, text-offset, and text-byte chunks
incrementally. It no longer builds a full `Vec<ResidentUploadChunk>`, full text
offset vector, or full text byte layout before admission.

Scaled validation passed with resident execution and formula-backed answers.
The emitted execution metadata includes `peak_caller_owned_chunk_bytes`, so the
full 25% command can report bounded caller-owned chunk memory.

## Operator Summary

| item | status | evidence |
|---|---:|---|
| streaming upload/admission boundary | pass | `CudaDriverRuntime::retain_device_memory_owned_chunks(...)`; `Engine::install_benchmark_relational_residency_owned_chunks(...)` |
| scaled retained CUDA upload | pass | `scripts/run_p8_ch_benchmark_residency_probe.sh --chunked-upload-self-check` |
| scaled benchmark cache admission | pass | `scripts/run_p8_ch_benchmark_residency_probe.sh --chunked-install-self-check` |
| formula-backed execution | pass, scaled | `--run-25pct-execute` with 128 rows before cleanup regression surfaced |
| 25% full tier | blocked | `full_25pct_requires_operator_long_run_after_streaming_boundary` |
| PostgreSQL Docker comparator lifecycle | blocked | `existing_container_cleanup_failed` for new disposable `gpu-db-p8-pgsql-baseline-disposable` |
| target artifact cleanup | pass | `GPU_DB_CH_BENCH_OUT_DIR=target/p8-streaming-boundary-check scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup` |

## Docker Cleanup Blocker

The previous fixed-name container `gpu-db-p8-pgsql-baseline` was already
unremovable on this host. This round switched the default comparator lifecycle
to a fresh disposable name and port:

- container: `gpu-db-p8-pgsql-baseline-disposable`
- port: `55434`
- observed_state: running
- observed_pid: `1216779`
- started_at: `2026-05-31T11:22:37.383916086Z`

Docker also refused to stop/remove that new disposable container:

```text
Error response from daemon: cannot stop container: gpu-db-p8-pgsql-baseline-disposable: permission denied
Error response from daemon: cannot remove container "gpu-db-p8-pgsql-baseline-disposable": could not kill container: permission denied
```

The script now refuses to reuse an existing comparator container when cleanup
fails, so future Docker comparator preflights fail early with:

```text
p8_ch_benchmark_pgsql_docker=blocked reason=existing_container_cleanup_failed name=gpu-db-p8-pgsql-baseline-disposable
```

## Validation

- `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --streaming-self-check`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --chunked-upload-self-check`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --chunked-install-self-check`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-preflight`: blocked as expected by cleanup failure
- `scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct`: blocked as expected because no clean PostgreSQL baseline artifact is available after cleanup failure
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-streaming-boundary-final ... --run-25pct-execute`: blocked as expected by Docker cleanup preflight
- `cargo check -p gpu_db_engine --example p8_ch_benchmark_residency_probe`: passed
- `cargo fmt --all`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-streaming-boundary-check scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup`: passed

## Exact Full-Run Command

After the Docker cleanup policy is fixed and an operator approves a long run:

```bash
GPU_DB_CH_BENCH_ALLOW_FULL_25PCT=1 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=1048576 scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute
```
