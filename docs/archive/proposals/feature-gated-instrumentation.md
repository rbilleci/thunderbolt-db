# ARCHIVED — Proposal: feature-gated instrumentation

> Historical implementation record. It is not an executable plan. Any surviving work is tracked only in
> `docs/PLAN.md`.

> **Status: ACCEPTED + IMPLEMENTED (`53f9c133`).** The `probe-timing` Cargo feature + `crates/execution/
> src/probe.rs` (`Probe` / `ProbeScope`) exist; `gpu_db_engine` forwards the feature. First permanent
> residents: the VM-lever probes (`predicate_total` in `engine_expr.rs`, `compact` in the compaction
> functions). Default build is byte-identical — the gated code is not compiled. Enable with
> `--features probe-timing`. **Still TODO from this doc:** the CI `cargo check --features probe-timing` step
> + the HANDOVER one-liner (run green manually for now).
> _(Recreated as a tracked file: the original was untracked and went missing on disk.)_

## Problem
Agents repeatedly **write instrumentation for a measurement, take the reading, then delete it** — e.g. the
drain sub-phase timers and the result-path / predicate phase-split timers, recorded as "instrumented ...
since removed." The cost of that churn:
- **Lost institutional knowledge** — the *measurement points* (where the phase-split timers go) are the
  hard-won part, and they're thrown away each time.
- **Re-invention** — the next agent needing the same number rewrites the same probes from scratch.
- **Risk** — every add/revert touches hot, safety-critical code for a throwaway change.

The reverts happen because the instrumentation can't be *left in*: a runtime check (or stray `eprintln!`) in
a hot loop perturbs the very thing being measured and shouldn't ship. The fix is to keep the code
**permanently, compiled out by default**.

## Goal
Instrumentation that lives in the source forever, is **zero-cost when off** (not compiled at all), and is
**one flag away** when needed — so probes *accrete* instead of churning.

## Convention (implemented)

### 1. One Cargo feature for analysis instrumentation
```toml
# crates/execution/Cargo.toml ; engine forwards it: probe-timing = ["gpu_db_execution/probe-timing"]
[features]
probe-timing = []
```
Off by default → gated code is absent from the binary (no branch, no symbol, no perturbation). Measure with
`cargo run --features probe-timing ...`. A feature is the right axis because we measure in `--release`;
`#[cfg(debug_assertions)]` is **wrong** here — release strips it.

### 2. A small `probe` module with zero-cost timers (clean call sites)
`crates/execution/src/probe.rs` — `Probe` is a ZST whose methods are empty when the feature is off, so the
compiler elides everything: `Probe::start()` + `.mark(label)` / `.lap(label)`, and `Probe::scope(label)`
for a drop-scoped timer (prints on guard drop — convenient for whole-function/block timing). With the
feature off, all of it is elided.

### 3. Pick the gate by granularity
- **Hot-path timing (per-row / per-drain): compile-time only** (the feature). A runtime env-var branch in a
  hot loop skews the measurement and blocks inlining.
- **Coarse / operational (per-batch): a runtime env var is acceptable** — but still behind the feature so it
  doesn't ship by default.
- **GPU / PTX instrumentation: select the instrumented vs. clean PTX at build time** via the feature. Do
  **not** add a runtime "instrument" kernel param — a device-side branch skews kernel timing.

## The discipline that makes it pay off (non-optional)
Feature-gated code that nothing compiles **bit-rots**, then gets rewritten anyway — defeating the point.
1. **CI compiles the gated code:** `cargo check --workspace --features probe-timing`. Keeps every probe
   valid without running it. _(TODO: wire into CI; run manually green for now.)_
2. **Document the feature** in `HANDOVER`: *"perf instrumentation lives behind the `probe-timing` feature —
   enable it, do not rewrite it."* This is the line that changes agent behavior. _(TODO.)_

## Non-goals
- Not a logging/observability framework redesign (operational telemetry is separate; this is *analysis*
  instrumentation). `tracing` may later subsume the structured cases.
- Not mandating instrumentation everywhere — just a home for it when written, instead of revert.
