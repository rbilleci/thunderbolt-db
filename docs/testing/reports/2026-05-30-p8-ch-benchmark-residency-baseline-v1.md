# P8 CH-benCHmark-Derived Residency Baseline

- date: 2026-05-30
- stream: benchmark
- milestone: P8 CH-benCHmark-derived retained-residency benchmark harness and first safe baseline probe
- validation_gate: `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`; `scripts/run_p8_ch_benchmark_residency_probe.sh --run-baseline`
- raw_metrics: `target/p8-ch-benchmark-residency/metrics.jsonl`
- estimate_metrics: `target/p8-ch-benchmark-residency/estimate.jsonl`
- cleanup: `scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup`

## Workload Contract

The checked harness defines a supported CH-benCHmark-derived analytical subset
over operational table names `order_line`, `stock`, `customer`, and `orders`.
The first executable slice intentionally uses accepted single-table resident
routes on `order_line`: `COUNT(*)`, `SUM(int4)`, `AVG(int4)`, and `MAX(int4)`
with supported int4 comparison and `BETWEEN` filters.

This is not a full CH-benCHmark, BenchBase, join, transaction-mix, or external
load-generator claim.

## Dry-Run Estimate

The harness preserves the requested target plan:

- 25% VRAM: 6 GiB retained target, estimated `161061274` `order_line` rows,
  about `15461882304` generated table bytes plus `7730941152` WAL/log bytes.
- 50% VRAM: 12 GiB retained target, estimated `322122548` rows.
- 100% VRAM: 24 GiB retained target, estimated `644245095` rows.
- 200% VRAM: 48 GiB retained target, estimated `1288490189` rows.
- Logical request targets: `1`, `10`, `100`, `1000`, and `10000`.

The scheduled-worker guardrail is `10000` generated rows by default, so the
25% VRAM target is not defensible for one cron slice. The first run therefore
uses a calibration tier while keeping the full tier plan machine-readable.

## Baseline Evidence

Calibration parameters:

- rows: `512`
- retained table: `order_line`
- resident bytes: `29184`
- attempted logical request targets: `1`, `10`

Result summary:

| Logical requests | Query | Status | p95 us | p99 us | Throughput qps | H2D bytes | CUDA event samples | Resident zero-H2D |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `order_line_count_all` | pass | 306 | 306 | 3238.97 | 0 | 1 | true |
| 1 | `order_line_sum_amount` | pass | 275 | 275 | 3606.61 | 0 | 1 | true |
| 1 | `order_line_avg_quantity_between` | pass | 2172 | 2172 | 459.91 | 0 | 1 | true |
| 1 | `order_line_max_amount_filter` | pass | 1941 | 1941 | 514.52 | 0 | 1 | true |
| 10 | `order_line_count_all` | pass | 253 | 253 | 4184.33 | 0 | 10 | true |
| 10 | `order_line_sum_amount` | pass | 272 | 272 | 3828.28 | 0 | 10 | true |
| 10 | `order_line_avg_quantity_between` | pass | 1862 | 1862 | 557.33 | 0 | 10 | true |
| 10 | `order_line_max_amount_filter` | pass | 1813 | 1813 | 562.74 | 0 | 10 | true |

The over-residency probe marks GPU 0 memory pressured after the resident
snapshot is admitted. The planned resident route is rejected with
`InvalidatedByMemoryPressure`, proving the harness can capture fallback
behavior before larger tiers are attempted.

## Cleanup Evidence

Generated reports and raw metrics are bounded under
`target/p8-ch-benchmark-residency/`. Cleanup is:

```bash
scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup
```

The cleanup command removes that directory and reports
`p8_ch_benchmark_cleanup=passed`.

## Decision

The benchmark harness is checked and reproducible, and the first safe baseline
changes the P8 production-readiness decision: the repo now has a tier-aware
CH-benCHmark-derived residency probe with p95/p99, throughput, raw JSONL,
zero-H2D resident-route evidence, CUDA event timing where the current resident
kernel exposes it, and memory-pressure fallback evidence.

The next bottleneck is not another local retained-route permutation. Attempting
the 6 GiB 25% VRAM tier needs a streaming or on-disk workload generator plus an
operator-approved longer run window and cleanup budget. Larger 50/100/200%
tiers should remain dry-run guarded until that first 6 GiB tier completes.
