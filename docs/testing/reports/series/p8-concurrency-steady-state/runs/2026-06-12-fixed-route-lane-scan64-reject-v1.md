# P8 Fixed Route Lane Scan64 Rejection

- stream: benchmark
- round_id: 2026-06-12-fixed-route-lane-scan64-reject-v1
- status: closed
- focus: test whether fixed route-lane scan limit should move from 32 to 64
- decision: keep fixed scan limit at 32
- next_target: owner-loop pending completion overlap

## Result

Fixed scan limit `64` was not a clean win over fixed scan limit `32`.

Scan64 c64:

- count: `52175 qps / 914us p50`
- exact multi-column lookup: `34846 qps / 1419us p50`
- multi-column literal batch: `13166 qps / 4057us p50`
- projection literal batch: `12344 qps / 4489us p50`
- mixed int4/text literal batch: `14122 qps / 3796us p50`
- heterogeneous literal batch: `12707 qps / 4211us p50`

Fixed32 c64 from `2026-06-12-fixed-route-lane-default-after-async-int4-v1`:

- count: `45193 qps / 1022us p50`
- exact multi-column lookup: `31503 qps / 1576us p50`
- multi-column literal batch: `13634 qps / 3940us p50`
- projection literal batch: `12739 qps / 4351us p50`
- mixed int4/text literal batch: `13716 qps / 3875us p50`
- heterogeneous literal batch: `12805 qps / 4117us p50`

## Interpretation

Scan64 helps count, exact lookup, and mixed slightly. It hurts the key
route-lane rows: multi-column literal batch, projection literal batch, and
heterogeneous. That is not enough to move the default from the already simpler
fixed32 policy.

## Decision

Keep:

- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=fixed`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32`

Avoid more scan-limit tuning unless a later owner-overlap change shifts the
batch-fill distribution enough to justify retesting.
