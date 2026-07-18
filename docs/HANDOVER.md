# HANDOVER — Resume Baton

This file records only the current boundary and where the next agent resumes. `PLAN.md` owns all open work;
`STATUS.md` owns completed evidence. Do not turn this file into another backlog.

## Current boundary

- **R3-001 through R3-005, DUR-002, STRUCT-001, and RETIRE-001 are complete.** Production relational reads and
  writes are GPU-required; unsupported execution fails loudly without host relational fallback or fabricated
  fallback telemetry.
- RETIRE-001 deleted the test CPU MVCC backend, fake parity backend, fallback adapter, host reference operators,
  CPU-oracle constructor, and merged host SQL finalization fixtures. Host-neutral semantics now use rows-only
  specification objects; actual CUDA metadata is separately typed execution evidence.
- Explicit DDL/recovery/import/VACUUM reverse-gather and the bounded hot-to-cold representation transition remain
  isolated under **RETIRE-002**. Generic CUDA-MVCC host result post-processing remains **RETIRE-003**.
- **PERF-001 is the sole NOW task.** Recover the R3-004 descriptor/shard-count-scaled point-read loss without
  restoring host authority, late conversion, or weaker fail-stop publication. **RETIRE-003** is NEXT after
  PERF-001 acceptance.
- Physical multi-GPU work remains user-parked under **MULTI-001/002/003**.

## Integrated baseline — preserve it

- RETIRE-001's closed-form specifications cannot claim execution targets or enter backend dispatch. Relational
  actual-CUDA fixtures call the engine's cached CUDA-driver dispatcher directly and assert execution evidence
  separately from semantic rows/schema/access paths.
- Shard-resident point-batch declines use the per-query GPU route directly; do not feed nullable `Ready` results
  into the non-null flat-int4 batch ABI.
- The production decline seams named for CPU compatibility execute no relational work and emit no fallback metric.
  Do not restore host execution behind those names for a benchmark win or while later removing RETIRE-003
  post-processing.
- The pre-R3-004 late-converted unified snapshot reached **264.2M/s in-L2 and 275.7M/s out-of-L2** at batch 65,536.
  Insert-published authoritative shards reached **89.6M/s and 3.29M/s**; RETIRE-001 reproduced **87.3M/s and
  3.20M/s**. The latter is a stable attribution baseline, not an accepted performance target. Preserve the new
  device-authoritative representation while removing its descriptor/shard-count-scaled submission cost.
- `PLAN.md` is reconciled to one active path and `STATUS.md` owns completed evidence. Preserve that ownership split.

## Resume here

1. Read the required project documents, then begin **PERF-001** with permanent `probe-timing` attribution. Hold row
   count and query shape fixed while sweeping shard/descriptor count; separate descriptor enumeration, route
   preparation, submission, launch/synchronization, kernel, and final readback time.
2. Make one narrow recovery change only after the causal scaling term is demonstrated. Preserve insert-published
   device authority, byte-identical GPU results, fail-stop publication, and a nonzero production-route counter; do
   not restore the retired late-conversion or any host relational cache/index/probe.
3. Run focused correctness/performance gates and commission an independent slice audit after every change. Apply
   the canonical report card to both layers and cache regimes and account for every residual gap against the
   pre-R3-004 evidence.
4. Accept and publish PERF-001 before starting **RETIRE-003**. Keep **RETIRE-002** repair outside the slice unless
   PLAN explicitly promotes it.

## Last green evidence — 2026-07-18

- Engine library: **486/486** ordinary and **1,017/1,017** including ignored actual-GPU tests.
- Execution library: **47/47** ordinary and **126/126** including ignored GPU tests. Facade library: **47/47**;
  serialized concurrency integration: **14/14**.
- The final relational bridge family passed three sequential and two concurrent **23/23** HAZARD waves with no
  CUDA 700/716/717/719 or context-loss signature; all six RETIRE-001 slice audits/re-audits are clean.
- The canonical report card completed both layers/cache regimes: raw rooflines were **1,478.8/1,450.6 GB/s** and
  production batched point reads reached **87.3M/s at p50 622us** in-L2 and **3.20M/s at p50 20.324ms** out-of-L2,
  within roughly 3% of the accepted R3-004 card. This proves RETIRE-001 added no further loss; PERF-001 owns the
  roughly 3x in-L2 and 84–86x out-of-L2 gap from the pre-R3-004 late-converted evidence.
- Workspace all-target/all-feature check, strict Clippy, formatting, diff whitespace, source-size, reference, and
  documentation ownership gates pass.

## Required operations

- Read `AGENTS.md`, `docs/CHARTER.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/PLAN.md`,
  `docs/STATUS.md`, and `docs/CODE_SIZE.md` before changing runtime, storage, or scheduling.
- Never use `--gpu-reset`. Serialize ordinary GPU sweeps, use timeouts, and use workspace-local `target/tmp` rather
  than `/tmp` for large generated artifacts.
- Run `scripts/benchmark_report_card.sh` after any read-kernel, residency-layout, or result-path change; compare
  ratios rather than absolute bandwidth.
