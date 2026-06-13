# P1-M3 — Noise-Controlled Baseline (median-of-N + CI)

Status: closed (noise-control harness landed + independently audited; baseline captured)
Date: 2026-06-13
Plan: `docs/roadmap/prototype-to-production-plan.md` §5.7 (benchmark discipline),
Phase 5 (harness noise reduction — minimal slice pulled forward)
Branch: `phase0-m1-engine-facade`

## Why this exists (the decision)

P1-M3 step 4 is the first milestone allowed to claim a real latency improvement
(collapsing M0's queue-wait term via concurrent `&self` reads). Per §5.7 that claim
is only credible **with** noise controls — and today's harness "cannot resolve
sub-10% deltas (±15–40% per-cell variance)." Rather than make the step-4 claim on a
noisy single-shot harness, this inter-milestone slice builds the minimal
median-of-N + confidence-interval layer and captures a noise-controlled baseline
**now**, so step 4 has a trustworthy before/after on the same harness.

This is a deliberate, scoped pull-forward of Phase-5 harness work — **not** the full
Phase-5 harness (see *Scope boundaries*).

## What was built (additive scripts; the measured path is untouched)

- `scripts/run_p8_engine_pgwire_median_of_n.sh` — runs the **existing**
  `--engine-backed-pgwire-concurrency-smoke` N times (plus one **discarded** warm-up
  run — warmup separated from measurement), each in its own output dir and on a
  fresh port, pinned to the M0 conditions (cache-off, batched read runtime on,
  concurrency targets `1,2,4,8,16,32,64`). It then calls the aggregator. A failed or
  partial run (non-zero rc **or** empty `metrics.jsonl`) is **excluded**; transient
  bind/startup failures retry on a fresh port (a monotonic, per-batch-jittered port
  allocator avoids TIME_WAIT reuse; retry count is recorded in `host-facts.txt`).
- `scripts/aggregate_concurrency_runs.py` — reads the N runs' `metrics.jsonl`, groups
  by `(query, concurrency)` cell, and reports per metric: median, mean, sample
  stddev, **coefficient of variation** (CV = stddev/mean — the noise measure), a
  **95% CI** for the mean (Student-t) and its half-width / median (the **precision**
  of this estimate), and the **A/B minimum detectable effect** — the smallest true
  before/after difference detectable with two N-sample groups at α=0.05 (two-sided),
  power=0.80 (`(t.975+t.80)·sd·√(2/n)/median`). Records with `error_count>0` or
  `correctness_status≠pass` are **excluded** from the statistics.

It changes nothing on the engine/server serving path; it only repeats and
post-processes the existing harness artifacts.

## Baseline run

- Host: RTX PRO 6000 Blackwell Max-Q (97,887 MiB), driver 595.71.05; cargo/rustc
  1.94.0; commit `82d69ffb`.
- Command:
  `GPU_DB_MEDIAN_RUNS=10 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64 GPU_DB_MEDIAN_OUT_DIR=target/2026-06-13-p1-m3-median-baseline scripts/run_p8_engine_pgwire_median_of_n.sh`
- 10/10 measured runs usable, 0 errors, 0 excluded-bad, **all cells `correctness=pass`**,
  even sample counts (n=10/cell). 4 transient port-bind collisions occurred and were
  all recovered by retry on a fresh port (`retries_total=4` in host-facts) — the
  retry/exclusion paths are exercised and validated.
- Artifacts: `target/2026-06-13-p1-m3-median-baseline/aggregate/{summary.md,aggregate.csv,aggregate.json}`
  plus per-run `run-*/` and `host-facts.txt`.

### Harness noise and the step-4 gate

| measure (p50, across cells) | median | max |
|---|--:|--:|
| coefficient of variation (CV) | 9.7% | 17.6% |
| estimate precision (95% CI half-width / median) | 7.0% | 13.1% |
| **A/B minimum detectable effect** (two-sample, α=0.05, power=0.80, N=10/side) | **12.9%** | **24.2%** |

(qps CV: median 7.9%, max 16.6%.)

**The step-4 gate is the A/B minimum detectable effect (~13% median cell, up to ~24%
noisiest), not the CI half-width.** A before/after comparison must clear this to count
as real. The CI half-width (~7%) is the *precision of this single baseline estimate*
and is ~2× smaller — using it as the gate would overstate the harness's resolving
power (an error caught in the audit below). At N=10 the harness resolves ~13% median
before/after deltas (vs. the prior "cannot resolve sub-10%"); larger N tightens this
(MDE ∝ (t.975+t.80)/√n).

### c64 cells (median [95% CI])

| query | p50 µs median [CI] | qps median [CI] | queue-wait µs med | CUDA µs med | p50 CV% | p50 A/B MDE% |
|---|--:|--:|--:|--:|--:|--:|
| count_all | 1492 [1386–1677] | 28224 [25733–28869] | 213 | 13 | 13.3 | 18.1 |
| lookup multi_column | 1744 [1663–2084] | 22085 [20693–23271] | 213 | 13 | 15.7 | 22.4 |
| lookup multi_column_literal_batch | 1950 [1814–2230] | 19723 [19031–21422] | 213 | 13 | 14.4 | 19.8 |
| lookup projection_literal_batch | 1599 [1527–1771] | 23386 [22254–24810] | 213 | 13 | 10.3 | 14.1 |
| lookup mixed_projection_literal_batch | 1880 [1786–2208] | 21355 [19896–21730] | 213 | 13 | 14.8 | 20.8 |
| lookup heterogeneous_literal_batch | 2266 [2132–2465] | 16945 [16009–18212] | 213 | 13 | 10.1 | 13.6 |

All cells `correctness=pass`, 0 errors.

### Sanity check vs. committed M0

M0 (single-shot) `count` c64 was p50 **1579µs**, qps **25427**. The median-of-10 is
p50 **1492µs** [CI 1386–1677] (min–max 1264–1854), qps **28224** [CI 25733–28869].
M0's p50 sits **inside** the median-of-10 CI; its qps sits just below the qps CI but
inside the qps min–max — i.e. M0 was a representative single draw. The noise-controlled
baseline **reproduces M0 within run-to-run variance** (consistent with §5.7's
mechanistic-isolation argument: step 1 changed nothing on the measured path).

## Independent adversarial audit

A reviewer was tasked to **refute** (A) the statistics, (B) wrapper integrity,
(C) measured-path isolation, (D) honesty. Result: **no blocker** — all 336
cell-metrics recomputed with 0 mismatches, the Student-t table validated, the
headline numbers reproduced byte-identically, and the layer confirmed to add zero
lines to any serving-path source. Findings, all fixed in this commit:

- **MAJOR — "minimum detectable effect" was mis-named (~2× optimistic).** The original
  figure was the CI half-width / median (a *precision* measure). A real before/after
  A/B at 80% power needs ~2×. Fixed: report the CI half-width as *estimate precision*
  **and** add the power-based two-sample A/B MDE as the actual gate (the headline
  above now uses the A/B MDE).
- **MAJOR — per-cell error/non-pass records were surfaced but not excluded** from the
  statistics (latent — this baseline had 0). Fixed: the aggregator drops any record
  with `error_count>0`/`correctness≠pass` and reports `excluded_bad` + uneven-n; an
  added synthetic test confirms a contaminated value never reaches a median.
- **MINOR — t-table fallback picked the least-conservative value** for untabulated
  df. Fixed: round *down* to the nearest tabulated df (conservative); table extended
  to df 40/60/120 and filled 21–29.
- **MINOR — no retry audit trail.** Fixed: `retries_total` recorded in host-facts.
- **NIT — n=1 reported 0% CV** (read as "zero noise"). Fixed: n<2 → `null` CV/CI/MDE.

## What this establishes — and what it does NOT

**Establishes:** a reusable, audited median-of-N + CI + CV + A/B-MDE harness for the
engine-pgwire concurrency path, and a fresh noise-controlled baseline (the regression
anchor for steps 2–4) with a quantified per-cell significance gate.

**Does NOT establish any performance change.** Step 1 is runtime-inert on the measured
path; these numbers are a *noise characterization*, not an improvement. The
improvement claim is step 4, after the `&self` read-path flip.

## Scope boundaries (still Phase 5, not done here)

- Still a **closed-loop, single-shot-per-cell** microbench on a **64-row** dataset —
  no open-loop/offered-rate load, no steady-state duration, no **p99.9** capture, no
  three-way PostgreSQL comparison curves.
- CV/CI are computed across whole-run repetitions; warmup is separated only at the
  run level (one discarded warm-up), not within a run.
- The point estimate is the median while the CI is for the mean (a common small-N
  simplification, labeled as such); both are in `aggregate.csv`/`.json`.

These remain the full Phase-5 harness work; this slice exists solely to make the
step-4 latency claim credible.

## Next

- Steps 2–3 (residency `SnapshotCell<Arc<owner>>`, publish-on-commit, `&self` read
  flip) — unchanged by this slice.
- Step 4: re-run this exact median-of-N command after the read-path flip and report
  the c64 queue-wait / p50 deltas **against this baseline**, accepting only deltas
  above each cell's **A/B minimum detectable effect**.
