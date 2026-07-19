# HANDOVER — Resume Baton

This file records only the current boundary and where the next agent resumes. `PLAN.md` owns all open work;
`STATUS.md` owns completed evidence. Do not turn this file into another backlog.

## Current boundary

- **R3-001 through R3-005, DUR-002, STRUCT-001, RETIRE-001, RETIRE-003, and PERF-001 are complete.** Production
  relational reads and writes are GPU-required; unsupported work fails loudly without host relational execution or
  fabricated fallback telemetry.
- RETIRE-003 deleted generic KV/MVCC fake-CUDA result execution and its host compaction/order/projection seams.
  Ordinary SELECT and direct/streaming joins now retain coordinates, order/window, and result values on-device until
  one strict terminal frame readback. Unsupported shapes fail independently of cardinality.
- PERF-001's exact-generation prepared point routes, hard-budget accounting, duplicate-decline contract, async CUDA
  ownership, and public result contracts remain intact. Explicit reverse-gather/deauthorization and DDL/recovery/
  import repair remain isolated under blocked **RETIRE-002**.
- Physical multi-GPU work remains user-parked under **MULTI-001/002/003**.

## Integrated baseline — preserve it

- The generic KV/MVCC entry preserves commit-wedge and leader-precedence errors, then fails before source resolution
  or GPU/fallback telemetry. Do not restore a host or fake-device implementation behind that compatibility name.
- Ordinary SELECT survivors are device coordinates with same-context provenance. Predicate compaction, checked
  arithmetic, ORDER/LIMIT/OFFSET, fixed/text/validity materialization, and terminal framing remain one device pipeline;
  the host may decode the single bounded frame only for final protocol values.
- Empty results do not bypass type, width, ORDER, timestamp, bool, or aggregate-shape validation. Launched errors
  drain before pool reuse, and nullable or filtered-out rows cannot manufacture overflow.
- The canonical report card is the result-path regression gate. The accepted RETIRE-003 card reached
  **232.466M/s at p50 156us** in-L2 and **197.040M/s at p50 206us** out-of-L2 at batch 65,536. Layer-1 rooflines were
  **1,478.8/1,441.1 GB/s**; the 48M-row fixture built in **1,683.3s** with zero late residency work.

## Resume here

1. Pick up **BENCH-001** from PLAN: reconcile the stale mutation-boundary wrapper with the live resident post-INSERT
   fact, then run the immutable sustained and `B01`–`B10` PostgreSQL comparison in a quiet reproducible window.
2. Take **DUR-001** next: add automatic intent-lane checkpoint cadence and timestamped lane records for archive/PITR,
   retaining explicit operator checkpoint/refusal behavior until crash gates pass.
3. Keep **RETIRE-002** outside either slice unless its device-native repair prerequisites are satisfied and PLAN
   explicitly promotes it.

## Last green evidence — 2026-07-19

- Engine library: **948/948** including ignored GPU tests on the exact candidate (**250.74s**). Execution library:
  **120/120** (**17.04s**). Normal and all-feature workspace suites pass, as do workspace all-target/all-feature
  check, strict Clippy, PTX assembly, scoped rustfmt, and diff whitespace.
- Three affected GPU families each passed three sequential plus two simultaneous HAZARD runs with zero CUDA
  700/716/717. Independent adversarial audits and re-audits are clean; every concrete finding was adopted.
- The canonical two-layer/two-cache report card completed with exit 0. Production compact reached
  **232.466M/s at p50 156us** in-L2 and **197.040M/s at p50 206us** out-of-L2; raw rooflines were
  **1,478.8/1,441.1 GB/s**, isolated gather **349.5/155.3 GB/s**, and GROUP BY **1,674.9M elements/s**.
- Tokio-postgres, SQLx, and node-postgres smokes pass. Full psql/application-driver harnesses were locally
  prerequisite-limited by absent libpq connection variables and missing Python 3.14 `pip`, not by product failures.

## Required operations

- Read `AGENTS.md`, `docs/CHARTER.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/PLAN.md`,
  `docs/STATUS.md`, and `docs/CODE_SIZE.md` before changing runtime, storage, or scheduling.
- Never use `--gpu-reset`. Serialize ordinary GPU sweeps, use timeouts, and clear stale generated `target/tmp`
  artifacts before a long full-GPU run if disk headroom is low.
- Run `scripts/benchmark_report_card.sh` after any read-kernel, residency-layout, or result-path change; compare
  ratios rather than absolute bandwidth.
