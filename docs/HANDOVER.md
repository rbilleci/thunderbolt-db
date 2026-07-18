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
- **PERF-001 is complete.** The accepted tree removes shard-count-scaled descriptor preparation with exact-generation
  prepared device plans and lock-free route reuse, while preserving fail-stop publication, hard-budget accounting,
  immutable snapshot semantics, async CUDA ownership, and public result contracts. Three independent final lanes
  accepted publication/accounting, CUDA ownership/contracts, and benchmark/evidence with no severity finding. The
  canonical card completed both layers and cache regimes: production compact is **232.641M/s at p50 156us** in-L2
  and **198.933M/s at p50 203us** out-of-L2. The archive owns the full remediation/audit chronology.
- **RETIRE-003 is the sole NOW task.** Remove generic CUDA-MVCC host compaction, ordering, projection, and result
  assembly while preserving the accepted PERF-001 prepared-route/result-path baseline.
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
  3.20M/s**. Accepted PERF-001 production-compact evidence is **232.641M/s at p50 156us** in-L2 and **198.933M/s
  at p50 203us** out-of-L2. Required transfers are common to both representations and do not explain the remaining
  **11.9%/27.8%** gaps; fragmentation is an inference, not proof.
- `PLAN.md` is reconciled to one active path and `STATUS.md` owns completed evidence. Preserve that ownership split.

## Resume here

1. Pick up **RETIRE-003** from PLAN. Inventory the generic CUDA-MVCC host compaction/order/projection/result-assembly
   seams and define the first bounded device-resident deletion slice.
2. Preserve PERF-001's exact-generation prepared plans, allocation/stream ownership, duplicate-decline contract,
   route accounting, and accepted report-card baseline. Run the standard card after any result-path change.
3. Keep **RETIRE-002** repair outside the slice unless PLAN explicitly promotes it.

## Last green evidence — 2026-07-18

- Engine library: **1,026/1,026** including ignored GPU tests on the exact candidate (397.69s).
- Execution library: **129/129** including ignored GPU tests on the exact candidate (18.30s). Facade ordinary
  all-target tests pass **39 with 8
  GPU-ignored**; serialized concurrency passes **13 with 1 GPU-ignored**.
- The PERF-001 nine-test point-read family passed three sequential and two simultaneous HAZARD invocations with no
  device fault. It includes deterministic old-DELETE-boundary, invalidation-republish, NULL-generation, and
  budget-publication interleavings in addition to byte/visibility and failure-phase coverage. The DELETE test runs
  the failing current-route-first/old-reader-second order; the execution gate injects panics after H2D and D2H and
  proves exact pool reuse. It also pauses a losing preparer and proves route retirement releases the replaced index
  and the stale plan cannot displace the accounted semantic-superset route. A paused index build cannot republish
  after DROP, and a separate fused/unfused append gate replaces the index after launch and proves Arc-identity
  publication plus exact rebuild.
- The final canonical card completed with exit 0. Production compact reached **232.641M/s at p50 156us** in-L2 and
  **198.933M/s at p50 203us** out-of-L2; the 48M-row fixture built in **1,633.7s** with zero late residency work.
  Layer-1 in/out-of-L2 rooflines were **1,313.7/1,433.8 GB/s**, isolated gather **349.5/155.3 GB/s**, and GROUP BY
  **1,673.5M elements/s at p50 5.012ms**. All ratio/algorithmic regression signals are green.
- Workspace all-target/all-feature check, strict Clippy, formatting, diff whitespace, and changed-source-size gates
  pass. The latest proof-focused exact execution sweep is **129/129** in 18.20s; all three final audit lanes accept.

## Required operations

- Read `AGENTS.md`, `docs/CHARTER.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/PLAN.md`,
  `docs/STATUS.md`, and `docs/CODE_SIZE.md` before changing runtime, storage, or scheduling.
- Never use `--gpu-reset`. Serialize ordinary GPU sweeps, use timeouts, and use workspace-local `target/tmp` rather
  than `/tmp` for large generated artifacts.
- Run `scripts/benchmark_report_card.sh` after any read-kernel, residency-layout, or result-path change; compare
  ratios rather than absolute bandwidth.
