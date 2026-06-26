# P0-M2 — First Serving Path Through the Façade

Status: closed
Date: 2026-06-13
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 0, §5.7
Branch: `phase0-m1-engine-facade`
Builds on: P0-M1 (`2026-06-13-p0-m1-engine-facade-v1.md`), M0 baseline
(`2026-06-13-phase0-m0-baseline-v1.md`)

## Goal

Route a real serving path through the protocol-neutral façade **without**
perturbing the optimized retained-read path that M0 measures, and prove no
regression against M0.

## What changed

- Façade: extracted the stateless execution core into
  `gpu_db_facade::execute_on_engine(&mut Engine, txn_id, sql) -> QueryOutcome`,
  so a serving path that already owns its `Engine` can route a statement through
  the neutral boundary without giving up ownership. `EngineFacade::execute` now
  calls it and keeps session transaction tracking. (All 9 façade tests still pass.)
- Benchmark endpoint: the `CREATE TABLE` simple-query branch now executes through
  `execute_on_engine`, and the PostgreSQL completion tag is formatted by
  `pg_adapter::command_complete_tag`. One serving site changed; the hot
  retained-read/microbatch paths and their telemetry are untouched.
- `gpu_db_facade` added as an engine **dev-dependency** (a dev-only back-edge;
  the library build graph stays acyclic).

## Why only this path (recorded finding)

The owner-thread **SELECT** branch is entangled with phase-fact telemetry that
reads engine-internal results (route decision, residency h2d/d2h deltas, type
OIDs). Routing it through the neutral façade would strip that telemetry, which the
benchmark emits by default (`SELECT_FACT_DETAIL=phase_only`). Fully neutralizing
the read path therefore requires the façade to expose **neutral** telemetry first
— a Phase 0/1 design item. `CREATE TABLE` has no such entanglement, so it is the
clean first migration. The measured int4 lookups run on the separate retained-read
runtime and were not touched at all.

## Benchmark gate (re-run of the M0 command)

`--engine-backed-pgwire-concurrency-smoke`, c1–c64, cache-off, identical to M0.
Façade fact `create_table_through_facade=true` confirmed present — the serving
path executed through the neutral boundary. **All families `correctness=pass`,
`error_count=0` at every concurrency.**

**Read the verdict first, then the table.** The verdict rests on *mechanistic
isolation*, **not** on the c64 deltas below — and those deltas are below this
harness's noise floor (±15–40% per cell, see the Observation section), so they are
reported only for completeness. CREATE TABLE is dispatched once during table setup
(`p8_engine_pgwire_benchmark_endpoint.rs:935`); the measured loop is the retained
SELECT path. The changed code therefore never runs inside the measured loop and
*cannot* move SELECT latency — the −29% (faster) and +18% (slower) cells are noise
in opposite directions, which is itself consistent with that.

c64, p50 µs / qps, M0 → M2 (all below the noise floor):

| query | p50 M0 | p50 M2 | Δ | qps M0 | qps M2 | err |
|---|--:|--:|--:|--:|--:|--:|
| count | 1579 | 1584 | 0% | 25427 | 25662 | 0 |
| exact-multicol | 2083 | 2058 | −1% | 20705 | 21843 | 0 |
| multicol-literal | 2880 | 2043 | −29% | 16044 | 20513 | 0 |
| proj-literal | 1459 | 1451 | −1% | 26048 | 26027 | 0 |
| mixed | 1723 | 2025 | +18% | 22084 | 18846 | 0 |
| heterogeneous | 1958 | 2085 | +6% | 17988 | 16706 | 0 |

## Verdict: pass (on mechanistic isolation + correctness)

No regression is possible from this change: it is confined to table creation and
does not execute inside the measured SELECT loop. Correctness is intact end to end
(`create_table_through_facade=true`, all families `correctness=pass`/`err=0`). The
c64 deltas are bidirectional noise and are **not** evidence either way — per §5.7,
with no measured-path change the gate is mechanistic isolation, not the numbers.

Secondary observation for the benchmark discipline: at 64-row tables / 8-request
bursts this harness shows ±15–40% per-cell variance at c8, so it cannot resolve
sub-10% regressions. The steady-state, open-loop, longer-duration harness (Phase
5) is needed before latency claims at that resolution are trustworthy.

## Next

- **P0-M3**: unify a real query path in the production `gpu-db-server` through the
  façade behind an off-by-default toggle, keeping the 352 golden scenarios green;
  this is the strategically central unification step and reuses `execute_on_engine`.
- Expose **neutral telemetry** through the façade so the read serving path can be
  migrated without losing phase facts.
- Then begin Phase 1 (P1-M2 reader/writer split), where the read path becomes a
  shared-snapshot read and the queue-wait term that dominates M0 starts to drop.
