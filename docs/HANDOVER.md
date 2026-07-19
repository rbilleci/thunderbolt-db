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
- READ-002's BENCH prerequisite milestone is complete: a typed, exact-generation `(int4, int8)` GPU equality
  directory provides O(1) lookup, on-device exact-key/MVCC checks, fixed-width gathering, nonzero index/cache
  evidence, and zero cold access. Its broader type breadth remains deferred under READ-002 after BENCH evidence.
- PRODUCT-002's canonical SQL/type milestone is complete: the immutable schema and W1 DML parse unchanged, required
  typed parameters/codecs are available, named resident indexes mutate incrementally, and checked UPDATE plus DML
  `RETURNING` stay on GPU execution/result paths. Broader catalog/type breadth remains blocked under PRODUCT-002
  until BENCH evidence.
- The active BENCH prerequisite is now **PRODUCT-001** prepared engine-backed serving.
- Physical multi-GPU work remains user-parked under **MULTI-001/002/003**.

## Integrated baseline — preserve it

- The generic KV/MVCC entry preserves commit-wedge and leader-precedence errors, then fails before source resolution
  or GPU/fallback telemetry. Do not restore a host or fake-device implementation behind that compatibility name.
- Ordinary SELECT survivors are device coordinates with same-context provenance. Predicate compaction, checked
  arithmetic, ORDER/LIMIT/OFFSET, fixed/text/validity materialization, and terminal framing remain one device pipeline;
  the host may decode the single bounded frame only for final protocol values.
- Empty results do not bypass type, width, ORDER, timestamp, bool, or aggregate-shape validation. Launched errors
  drain before pool reuse, and nullable or filtered-out rows cannot manufacture overflow.
- The canonical report card is the result-path regression gate. PRODUCT-002's final card reached
  **231.634M/s at p50 156us** in-L2 and **200.515M/s at p50 203us** out-of-L2 at batch 65,536. Layer-1 rooflines were
  **1,443.1/1,449.9 GB/s**; the 48M-row fixture built in **1,707.7s** with zero late residency work.

## Resume here

1. Start the **PRODUCT-001** BENCH milestone: engine-backed
   Parse/Bind/Execute, prepared R1/W1, atomic T8/T32, durable synchronous acknowledgement, and pre-WAL enforcement
   of every manifest numeric route envelope. Do not add workload behavior to the legacy host-relational endpoint.
2. Resume **BENCH-001** for its remaining seed/open-loop driver, tuned PostgreSQL profile, artifacts, quiet-window
   qualification, and sustained plus `B01`–`B10` execution. The mutation wrapper is already repaired and audited;
   do not redo it or substitute P8 microbenchmarks.
3. Take **DUR-001** only after preserving the accepted pre-DUR BENCH artifact, then rerun the affected cohorts for
   checkpoint-policy overhead. **ROUTE-001** and **SCALE-001** remain downstream of BENCH evidence.
4. Keep **RETIRE-002** outside these slices unless its device-native repair prerequisites are satisfied and PLAN
   explicitly promotes it.

## Last green evidence — 2026-07-19

- Workspace tests pass, including engine **466 passed/501 GPU-ignored**, execution **52/75**, SQL **33/0**, facade
  **41/9**, protocol **71 plus 127 binary tests**, integration suites, and doc tests. Workspace all-target check,
  strict affected Clippy, scoped rustfmt, source-size, and diff-whitespace gates are clean.
- PRODUCT-002's final candidate passed three sequential plus two simultaneous HAZARD runs at 2/2 each with zero
  CUDA 700/716/717. Independent per-slice and final adversarial audits are clean after every concrete finding was
  adopted, including API result-discard, NULL arithmetic, and old typed-WAL compatibility seams.
- The canonical two-layer/two-cache report card completed with exit 0. Production compact reached the best
  PRODUCT-002 result of **231.634M/s at p50 156us** in-L2 and **200.515M/s at p50 203us** out-of-L2; raw rooflines were
  **1,443.1/1,449.9 GB/s**, isolated gather **349.5/155.3 GB/s**, and GROUP BY **1,674.2M elements/s**. The out-of-L2
  point path is 0.8% above the immediately preceding slice and within 0.7% of the historical best. The 48M-row
  fixture built in **1,707.7s** with zero late residency work.
- Tokio-postgres, SQLx, and node-postgres smokes pass. Full psql/application-driver harnesses were locally
  prerequisite-limited by absent libpq connection variables and missing Python 3.14 `pip`, not by product failures.
- BENCH-001's stale mutation boundary is repaired and sabotage-gated, but no sustained/peak result exists yet.
  READ-002 and PRODUCT-002 canonical milestones are complete; PLAN now promotes PRODUCT-001 before the remaining
  runner and campaign work. DUR-001 remains blocked until the pre-DUR result is preserved.

## Required operations

- Read `AGENTS.md`, `docs/CHARTER.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/PLAN.md`,
  `docs/STATUS.md`, and `docs/CODE_SIZE.md` before changing runtime, storage, or scheduling.
- Never use `--gpu-reset`. Serialize ordinary GPU sweeps, use timeouts, and clear stale generated `target/tmp`
  artifacts before a long full-GPU run if disk headroom is low.
- Run `scripts/benchmark_report_card.sh` after any read-kernel, residency-layout, or result-path change; compare
  ratios rather than absolute bandwidth.
