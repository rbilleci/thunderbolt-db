# P8 Retained Read Runtime View Rejection

## Summary

This probe wired the existing retained device read-view scaffold into the
pgwire endpoint as an opt-in cache-off runtime path:

- `GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RUNTIME_VIEW=1`
- all-INT4 retained point reads can execute from client IO workers
- the owner still publishes the retained snapshot generation after COPY/warmup
- COPY and non-SELECT paths invalidate the runtime and wait for in-flight reads
- default remains off

The goal was to bypass the owner queue for immutable retained reads without
using the response cache.

## Change

- Added a shared retained read runtime to the benchmark endpoint.
- Published a narrow runtime route after retained warmup for one table with
  INT4 filter/projection columns.
- Used `CudaResidentDeviceMemoryReadView` from client IO workers to submit and
  complete `i32_equal_any_project` work.
- Added runtime facts for published routes, invalidations, attempts, hits,
  misses, unsupported requests, and failures.
- Added the benchmark script passthrough/report field
  `GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RUNTIME_VIEW`.
- Fixed the async retained INT4 submit primitive to set the CUDA context on the
  calling thread before allocation/launch. Without that, cross-thread read-view
  launches failed with CUDA error `201`.

## Validation

Passed:

```text
cargo fmt --all -- --check
cargo check -q -p gpu_db_execution
cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint
bash -n scripts/run_p8_ch_benchmark_residency_probe.sh
git diff --check
```

## Results

Common settings:

- rows: `64`
- requests/client: `8`
- warmup/client: `1`
- cache: off
- select facts: `none`
- pipeline depth: `1`
- route lane scan: fixed `32`

c8 runtime-view-on artifact:
`target/2026-06-13-read-runtime-view-c8-v3/engine-backed-pgwire-concurrency-smoke/`

- count: `9365 qps / 657us p50`
- exact multi-column: `2786 qps / 2050us p50`
- multi-column literal: `2814 qps / 2155us p50`
- projection literal: `2720 qps / 2260us p50`
- mixed int4/text: `5771 qps / 1038us p50`
- heterogeneous: `3254 qps / 1945us p50`
- runtime attempts/hits/unsupported/failures: `360/264/96/0`

c64 runtime-view-off guard artifact:
`target/2026-06-13-read-runtime-view-default-off-c64/engine-backed-pgwire-concurrency-smoke/`

- count: `51047 qps / 913us p50`
- exact multi-column: `37664 qps / 1276us p50`
- multi-column literal: `23032 qps / 2243us p50`
- projection literal: `20220 qps / 2833us p50`
- mixed int4/text: `24008 qps / 2200us p50`
- heterogeneous: `19723 qps / 2582us p50`

c64 runtime-view-on artifact:
`target/2026-06-13-read-runtime-view-c64/engine-backed-pgwire-concurrency-smoke/`

- count: `40356 qps / 1023us p50`
- exact multi-column: `1780 qps / 25902us p50`
- multi-column literal: `1898 qps / 24672us p50`
- projection literal: `1859 qps / 25425us p50`
- mixed int4/text: `23464 qps / 2193us p50`
- heterogeneous: `3042 qps / 14764us p50`
- runtime attempts/hits/unsupported/failures: `2880/2112/768/0`

## Decision

Reject the runtime view as a default optimization.

The read view successfully bypasses the owner and executes correctly from client
IO workers, but it destroys retained batch fill. All-INT4 runtime hits become
per-client singleton GPU launches, which is much worse than the owner-side
route-lane batches at c64. Mixed/text rows mostly fall back to the owner path,
which is why mixed remains near the default guard.

Keep the code default-off as an architectural probe and safety harness, not as
the production path.

## Next Direction

The next useful M4 slice should not route single requests directly to CUDA. It
needs independent work while preserving batch shape:

- a small per-route read runtime that batches typed params before submit, or
- a true stream-pool probe where each stream receives filled retained jobs, not
  singleton launches.

The durable lesson is the same as the pipeline-depth rejection: more in-flight
work only helps when it preserves GPU batch fill and avoids head-of-line or
singleton-launch behavior.
