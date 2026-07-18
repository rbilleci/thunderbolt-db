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
- **RETIRE-003 is the sole NOW task.** Keep selection compaction, ordering, projection, and result assembly on the
  device until one bounded final readback; unsupported shapes fail loudly.
- Physical multi-GPU work remains user-parked under **MULTI-001/002/003**.

## Integrated baseline — preserve it

- RETIRE-001's closed-form specifications cannot claim execution targets or enter backend dispatch. Relational
  actual-CUDA fixtures call the engine's cached CUDA-driver dispatcher directly and assert execution evidence
  separately from semantic rows/schema/access paths.
- Shard-resident point-batch declines use the per-query GPU route directly; do not feed nullable `Ready` results
  into the non-null flat-int4 batch ABI.
- The production decline seams named for CPU compatibility execute no relational work and emit no fallback metric.
  Do not restore host execution behind those names while removing RETIRE-003 post-processing.
- `PLAN.md` is reconciled to one active path and `STATUS.md` owns completed evidence. Preserve that ownership split.

## Resume here

1. Read the required project documents, then start **RETIRE-003** with one generic CUDA-MVCC result shape.
2. Preserve its semantics with a non-vacuous GPU differential, keep intermediates device-resident, perform one
   bounded final readback, and make unsupported work fail loudly before deleting the host helper.
3. Keep **RETIRE-002** repair outside the slice unless PLAN explicitly promotes it. Do not restore any host
   write/index/probe/test-oracle authority removed by R3-004 or RETIRE-001.
4. Run focused gates and an independent slice audit. Apply the canonical report card to every affected read/result
   path and compare both layers and cache regimes.

## Last green evidence — 2026-07-18

- Engine library: **486/486** ordinary and **1,017/1,017** including ignored actual-GPU tests.
- Execution library: **47/47** ordinary and **126/126** including ignored GPU tests. Facade library: **47/47**;
  serialized concurrency integration: **14/14**.
- The final relational bridge family passed three sequential and two concurrent **23/23** HAZARD waves with no
  CUDA 700/716/717/719 or context-loss signature; all six RETIRE-001 slice audits/re-audits are clean.
- The canonical report card completed both layers/cache regimes: raw rooflines were **1,478.8/1,450.6 GB/s** and
  production batched point reads reached **87.3M/s at p50 622us** in-L2 and **3.20M/s at p50 20.324ms** out-of-L2,
  within roughly 3% of the accepted R3-004 card.
- Workspace all-target/all-feature check, strict Clippy, formatting, diff whitespace, source-size, reference, and
  documentation ownership gates pass.

## Required operations

- Read `AGENTS.md`, `docs/CHARTER.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/PLAN.md`,
  `docs/STATUS.md`, and `docs/CODE_SIZE.md` before changing runtime, storage, or scheduling.
- Never use `--gpu-reset`. Serialize ordinary GPU sweeps, use timeouts, and use workspace-local `target/tmp` rather
  than `/tmp` for large generated artifacts.
- Run `scripts/benchmark_report_card.sh` after any read-kernel, residency-layout, or result-path change; compare
  ratios rather than absolute bandwidth.
