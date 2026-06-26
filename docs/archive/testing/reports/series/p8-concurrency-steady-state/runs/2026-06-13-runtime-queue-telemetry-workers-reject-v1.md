# P8 Runtime Queue Telemetry And Worker-Shard Rejection

## Summary

This slice added runtime-side wait telemetry after the mixed int4/text runtime
path removed the owner fallback. It also tested route-key worker sharding as an
opt-in probe.

Decision: keep one runtime worker as the default. Route-key sharding reduced
some queue totals, but increased batch execution wall time and did not improve
the heterogeneous schedule reliably.

## Change

- Added per-request runtime queue wait facts:
  `retained_read_runtime_request_queue_wait_micros_total` and
  `retained_read_runtime_request_queue_wait_micros_max`.
- Added per-route runtime stats emitted as
  `retained_read_runtime_route_stats_json`.
- Added opt-in route-key sharding with
  `GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RUNTIME_WORKERS`.
- Kept `GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RUNTIME_WORKERS=1` as the
  default.
- Updated the benchmark report metadata now that one-text retained point reads
  are on the runtime path by default.

## Validation

Passed:

```text
cargo fmt --all -- --check
cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint
bash -n scripts/run_p8_ch_benchmark_residency_probe.sh
git diff --check
```

Benchmark probes passed:

- default one-worker c64:
  `target/2026-06-13-runtime-queue-telemetry-default-c64/engine-backed-pgwire-concurrency-smoke/`
- workers=2 c64:
  `target/2026-06-13-runtime-workers2-c64/engine-backed-pgwire-concurrency-smoke/`
- workers=3 c64:
  `target/2026-06-13-runtime-workers3-c64/engine-backed-pgwire-concurrency-smoke/`

## Default Worker Read

Default one-worker c64:
`target/2026-06-13-runtime-queue-telemetry-default-c64/engine-backed-pgwire-concurrency-smoke/`

- count: `52713 qps / 888us p50`
- exact multi-column: `36548 qps / 1298us p50`
- multi-column literal: `35942 qps / 1290us p50`
- projection literal: `40917 qps / 1230us p50`
- mixed int4/text: `38464 qps / 1286us p50`
- heterogeneous: `28044 qps / 1811us p50`
- runtime attempts/hits/unsupported/failures: `2880/2880/0/0`
- runtime batches/batched requests/max batch: `110/2880/50`
- runtime request queue wait total/max: `1242365us/2007us`
- runtime batch execute wall total/max: `45816us/677us`

Average request queue wait was about `431us/request`. Average batch execution
wall was about `417us/batch`.

Route detail:

- `ol_o_id`: queue total/max `350571us/1527us`, execute total/max
  `11626us/486us`
- `ol_o_id,ol_dist_info`: queue total/max `367354us/2007us`, execute total/max
  `13340us/577us`
- `ol_o_id,ol_i_id,ol_quantity,ol_amount`: queue total/max
  `524440us/1225us`, execute total/max `20850us/677us`

## Worker-Shard Probe

Workers=2 c64:
`target/2026-06-13-runtime-workers2-c64/engine-backed-pgwire-concurrency-smoke/`

- mixed int4/text: `37578 qps / 1313us p50`
- heterogeneous: `25508 qps / 2036us p50`
- runtime queue total/max: `1207842us/2070us`
- runtime execute wall total/max: `62283us/1046us`

Workers=3 c64:
`target/2026-06-13-runtime-workers3-c64/engine-backed-pgwire-concurrency-smoke/`

- mixed int4/text: `34344 qps / 1486us p50`
- heterogeneous: `25957 qps / 1980us p50`
- runtime queue total/max: `1252967us/1783us`
- runtime execute wall total/max: `68535us/879us`

The worker-shard probe lowered queue wait for some routes, especially the mixed
text route in the workers=3 run, but it raised total batch execution wall and
lost enough batch quality/concurrency locality to regress the overall target.

## Decision

Reject worker sharding as a default. Keep the worker-count knob for further
experiments, but leave the default at one worker.

The data points to two better next steps:

1. make compact text projection asynchronous so text batches do not synchronize
   a whole runtime worker, and
2. introduce stream ownership deliberately instead of simply letting multiple
   host workers contend for the same retained context/default stream.
