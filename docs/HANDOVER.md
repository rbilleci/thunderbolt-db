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
- The active BENCH prerequisite sequence now starts at **PRODUCT-002** canonical SQL/type support, then
  **PRODUCT-001** prepared engine-backed serving. No PRODUCT-002 implementation has been entered yet.
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

1. Start the **PRODUCT-002** BENCH milestone: exact schema, SQL-placeholder and wire/OID type handling,
   compound secondary indexes, resident publication plus live mutation maintenance of every named index,
   `RETURNING`, and expression UPDATE. READ-002's completed directory is generation-scoped and O(rows) to build;
   PRODUCT-002 must maintain or replace it without rebuilding the full 10M-row accounts directory per mutation.
   Do not treat the current reprepare-on-INSERT/UPDATE behavior as BENCH-ready mutation evidence.
2. Follow with the **PRODUCT-001** BENCH milestone: engine-backed
   Parse/Bind/Execute, prepared R1/W1, atomic T8/T32, durable synchronous acknowledgement, and pre-WAL enforcement
   of every manifest numeric route envelope. Do not add workload behavior to the legacy host-relational endpoint.
3. Resume **BENCH-001** for its remaining seed/open-loop driver, tuned PostgreSQL profile, artifacts, quiet-window
   qualification, and sustained plus `B01`–`B10` execution. The mutation wrapper is already repaired and audited;
   do not redo it or substitute P8 microbenchmarks.
4. Take **DUR-001** only after preserving the accepted pre-DUR BENCH artifact, then rerun the affected cohorts for
   checkpoint-policy overhead. **ROUTE-001** and **SCALE-001** remain downstream of BENCH evidence.
5. Keep **RETIRE-002** outside these slices unless its device-native repair prerequisites are satisfied and PLAN
   explicitly promotes it.

## Last green evidence — 2026-07-19

- Engine library: **952/952** including ignored GPU tests on the exact candidate (**265.33s**). Execution library:
  **121/121** (**16.50s**). Ordinary and release suites pass **461/461** engine plus **50/50** execution; workspace
  all-target/all-feature check, strict execution/engine Clippy, rustfmt, and diff whitespace are clean.
- The canonical compound READ route passed three sequential plus two simultaneous HAZARD runs with zero CUDA
  700/716/717. Independent engine, kernel/accounting, and evidence audits are clean; every concrete finding was
  adopted.
- The canonical two-layer/two-cache report card completed with exit 0. Production compact reached
  **230.621M/s at p50 158us** in-L2 and **201.845M/s at p50 199us** out-of-L2; raw rooflines were
  **1,481.4/1,440.6 GB/s**, isolated gather **349.5/155.3 GB/s**, and GROUP BY **1,674.9M elements/s**. The 48M-row
  fixture built in **1,623.9s** with zero late residency work. An initial legacy-cache regression was independently
  reproduced against clean HEAD and removed by separating the compound route cache; the final branch shape and card
  are back inside the accepted RETIRE-003 envelope.
- Tokio-postgres, SQLx, and node-postgres smokes pass. Full psql/application-driver harnesses were locally
  prerequisite-limited by absent libpq connection variables and missing Python 3.14 `pip`, not by product failures.
- BENCH-001's stale mutation boundary is repaired and sabotage-gated, but no sustained/peak result exists yet.
  READ-002's canonical milestone is complete; PLAN now promotes PRODUCT-002 → PRODUCT-001 before the remaining
  runner and campaign work. DUR-001 remains blocked until the pre-DUR result is preserved.

## Required operations

- Read `AGENTS.md`, `docs/CHARTER.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/PLAN.md`,
  `docs/STATUS.md`, and `docs/CODE_SIZE.md` before changing runtime, storage, or scheduling.
- Never use `--gpu-reset`. Serialize ordinary GPU sweeps, use timeouts, and clear stale generated `target/tmp`
  artifacts before a long full-GPU run if disk headroom is low.
- Run `scripts/benchmark_report_card.sh` after any read-kernel, residency-layout, or result-path change; compare
  ratios rather than absolute bandwidth.
