# gpu-database-engine

Bootstrap implementation workspace for the GPU-first PostgreSQL-compatible
database engine.

The current tree is a local product-readiness proof, not a production database
claim. The durable source of truth is CPU/WAL/checkpoint/archive state; GPU
resident state is explicit acceleration cache that can be rebuilt, refreshed,
invalidated, or evicted without weakening WAL-before-visibility.

## Current Envelope

- PostgreSQL-facing compatibility covers a bounded `public` `int4`/`text`
  relational subset with real `psql` and checked application-driver smokes for
  `tokio-postgres`, `sqlx`, `node-postgres`, `asyncpg`, `psycopg`, `pgx`, JDBC,
  and R2DBC.
- DDL/catalog support is intentionally narrow but tested: supported public
  tables, bounded indexes/constraints/defaults, views, materialized views,
  sequences, domains, zero-argument literal SQL functions, selected comments,
  bounded ACL/default-privilege metadata, and dump/restore surfaces are covered
  where documented in the compatibility matrix.
- Local release-candidate evidence is script-driven. The top-level preflight
  aggregates validation, PostgreSQL product compatibility, local GPU residency,
  and connection-security posture gates.
- P8 retained GPU residency is bounded to supported public `int4`/`text` table
  shapes. Current retained routes include aggregate/count families, supported
  int4 projection and lookup families, selected-row composite/text lookup
  materialization, warmup, maintenance, admission/eviction, invalidation, and
  route telemetry.
- The engine-backed benchmark pgwire endpoint is a bounded proof endpoint for
  retained-route benchmarking through real `psql`/libpq traffic. It is not a
  broad replacement for the full compatibility server.

See [docs/compatibility/matrix.md](docs/compatibility/matrix.md) and
[docs/compatibility/scorecard.latest.md](docs/compatibility/scorecard.latest.md)
for the detailed compatibility source of truth.

## Quick Validation

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
scripts/run_psql_golden.sh
scripts/run_application_driver_smokes.sh
scripts/run_local_release_candidate_preflight.sh
```

GPU/P8-specific local gates:

```bash
scripts/run_local_gpu_residency_preflight.sh
scripts/run_p8_resident_warmup_preflight_smoke.sh
scripts/run_p8_resident_maintenance_smoke.sh
scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run
```

P8 long-run benchmark commands are guarded. Do not run the full 25% or 125%
tiers from unattended automation; use the benchmark doc and reports below to
choose an operator-approved window and artifact budget.

## P8 Benchmark Status

P8 benchmark work is currently focused on CH-benCHmark-derived `order_line`
retained-residency evidence under identical PostgreSQL-compatible client
boundaries.

Current accepted smoke evidence:

- The same scaled `psql`/libpq driver, query text, concurrency schedule, and
  metric schema now runs against default PostgreSQL, tuned PostgreSQL, and the
  GPU DB retained endpoint.
- The latest scaled identical target includes the composite/text point lookup:
  `SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info FROM order_line WHERE ol_o_id = <literal> AND ol_i_id = <literal>`.
- The scaled smoke is graph-ready for concurrency `1,2`; it is not the full
  161,061,274-row 25% curve.

Current blockers and non-claims:

- `full_25pct_identical_curves_require_operator_long_run`
- `missing_partitioned_over_resident_execution`
- `retained_match_index_compaction_required_for_fully_device_side_filtering`
- no completed full 25% default/tuned PostgreSQL/GPU DB retained curve
- no completed 125% over-resident PostgreSQL-vs-GPU retained tier
- no full CH-benCHmark, BenchBase, join, transaction-mix, external load
  generation, production cache-daemon, durable GPU page, or external
  orchestration claim

Details live in [docs/testing/benchmarks/README.md](docs/testing/benchmarks/README.md).
The latest report is
[docs/testing/reports/2026-06-01-p8-identical-composite-text-pgwire-curves-v1.md](docs/testing/reports/2026-06-01-p8-identical-composite-text-pgwire-curves-v1.md).

## Security And Operations

The default compatibility endpoint remains a local/dev no-TLS, trust-style
profile. The opt-in production security profile v1 requires complete TLS and
SCRAM-SHA-256 verifier material and is covered by the local security posture
preflight.

Still out of scope: mTLS, certificate lifecycle automation, enterprise identity,
external secret-manager/KMS/HSM integration, audit hash-chain, row-level
security, masking, broad authorization, live systemd/Kubernetes rollout policy,
physical page-image backup, production object storage, PITR/DR production
runbooks beyond the documented local gates, migrations/upgrades, and broad
prepared-statement/portal/cursor parity beyond the current supported subset.

## Docs Map

- [docs/roadmap/v0-v1.md](docs/roadmap/v0-v1.md): phase roadmap and P8 design
  track.
- [docs/compatibility/matrix.md](docs/compatibility/matrix.md): detailed
  compatibility claims, partials, and non-goals.
- [docs/compatibility/scorecard.latest.md](docs/compatibility/scorecard.latest.md):
  latest checked compatibility scorecard.
- [docs/GPU_GUARDRAILS.md](docs/GPU_GUARDRAILS.md): GPU-first engineering
  guardrails.
- [docs/architecture/10-p8-gpu-optimized-storage-engine.md](docs/architecture/10-p8-gpu-optimized-storage-engine.md):
  P8 storage/cache architecture.
- [docs/testing/benchmarks/README.md](docs/testing/benchmarks/README.md): P8
  benchmark methodology, commands, artifacts, accepted evidence, and blockers.
- [docs/testing/reports/](docs/testing/reports/): durable report artifacts for
  compatibility, resilience, security, GPU residency, and P8 benchmark slices.

## Safety Invariant

The engine preserves **WAL-before-visibility**: a state transition is never
visible to readers before its corresponding WAL record is durably flushed.
