# PLAN — Unified Work Ledger

This is the **only document allowed to own open, deferred, blocked, or sequenced project work**.
`STATUS.md` owns current facts, `HANDOVER.md` owns the short resume baton, `ARCHITECTURE.md` and
`docs/design/` own design, and `DECISIONS.md` owns rationale. Action language elsewhere must reference a
task ID here or be explicitly historical.

## How to use this ledger

- States: **NOW** (active focus), **NEXT** (ready after NOW), **BLOCKED** (named prerequisite), **PARKED**
  (deliberately outside the current horizon), and **VERIFY** (old finding must be checked against the tree).
- A deferred task must name its trigger. Completed work is removed from this file and summarized in
  `STATUS.md` or the implementation archive.
- Future agents update one row here rather than creating a new plan, checklist, proposal sequence, or open
  board. Design documents may be linked as evidence but never override this ledger.
- Every implementation slice follows `CHARTER.md`, includes non-vacuous GPU execution evidence, and runs the
  relevant correctness/performance gates in `AGENTS.md`.

## Current focus

1. **STRUCT-001Q — isolate scalar-count kernel ownership.** Move int4 equality/comparison count reduction,
   their serial parity kernels, and shared nullable-bitmap validation behind a focused private module while
   preserving the crate-root API and later aggregate users of the validation contract.
2. **STRUCT-001 — analyze and disposition every oversized source file.** Establish safe module boundaries and
   reduce the highest context risks before broad implementation work expands them further.
3. **R3-001 — reconcile the live write path with the target GPU-native write design.** This remains the next
   architecture decision needed before wider write work or host-store deletion; structural extraction may
   precede it, but must not make the decision implicitly.
4. **BENCH-001 — complete the open-loop OLTP comparison.** Run in parallel when benchmark capacity is
   available; it remains the evidence gate for ordering performance work.

## STRUCT-001 — oversized-file remediation method

The source-size standard is [`CODE_SIZE.md`](CODE_SIZE.md). The 2026-07-12 baseline has **29 files outside its
analysis envelopes**: 17 production files over 2,000 lines, eight test files over 3,000 lines, and four examples
or tools over 3,000 lines. This inventory is a review queue, not a predetermined request to split every file.

### Analysis packet required for each file

Before editing an outlier, put a compact packet in its implementation commit/PR and distill the result into
that file's disposition cell in the inventory below:

1. current line count, class, and generated/archive status;
2. responsibility map by major types/functions/tests/kernels and the invariant owned by each cluster;
3. dependency map covering callers, imports/re-exports, shared state, feature gates, unsafe/device boundaries,
   build integration, and external API consumers;
4. change-history/co-change evidence sufficient to distinguish real cohesion from accidental accumulation;
5. one disposition: decompose, retain by registered exception, regenerate, archive, or delete;
6. for decomposition, the target module map, dependency direction, stable facade/re-exports, test moves, risks,
   and exact compile/test/GPU/performance gates.

Execute one behavior-preserving ownership extraction at a time. Keep semantic rewrites separate, update every
code/build/test/doc reference in the same slice, search for old paths and symbols, and do not create numbered
shards, catch-all modules, cycles, or a compensating `pub(crate)` sprawl. Target new modules below 1,500 lines.
Critical outliers over 5,000 lines receive priority; a pure structural move uses targeted correctness gates,
while any runtime/kernel/residency/result-path change also uses the full applicable `AGENTS.md` gates.

### Ordered inventory

**Wave 1 — critical production context risks (over 10,000 lines).** Analyze facades and embedded implementation
first so extraction preserves crate APIs and GPU/kernel ownership.

| Lines | File | Disposition / evidence |
|---:|---|---|
| 40,308 | `crates/execution/src/lib.rs` | **DECOMPOSE — STRUCT-001Q structural child active; root now 24,946 lines.** Handwritten production facade with distinct routing policy, CUDA runtime/context/driver layers, resident-memory ownership, MVCC batch encoding, point-read submission/result ownership, write locate/apply, and scan/aggregate/join/sort/expression kernel families. Public root symbols are consumed by engine, planner, metrics, observability, examples, and server tests; stable crate-root re-exports are the facade. **Completed modules:** `routing.rs` (241), RETIRE-001 `reference_operators.rs` (391), `mvcc_batch.rs` (208), `runtime_contract.rs`, sub-3,000-line test seams, `cuda_context.rs` (948), `cuda_driver.rs` (883), `resident_memory.rs` (405), `resident_sort.rs` (846), `point_read_submission.rs` (442), `point_read_submit.rs` (1,308), `point_read_dense.rs` (1,086), `point_read_bloom.rs` (206), `point_read_text.rs` (1,205 after STRUCT-001M/N correctness hardening), and `point_read_rows.rs` (475). STRUCT-001P moved all five GPU ORDER BY launchers byte-for-byte behind the private sort module; PTX paths/symbols/args/grids, public wrappers, buffers, uploaded-payload guard, stream synchronization, NULL/direction/text/multikey semantics, and padding filtering are unchanged. All-target/downstream checks, exact unchanged 101-test inventory, 31 non-ignored tests, and six i64/radix/multikey/heterogeneous/dedicated-text HAZARD gates passed 3× sequential + 2× concurrent with zero CUDA safety errors. Independent audit's dedicated-text evidence finding was adopted. The canonical card remained green: bitonic sort 346.6 Melem/s, in-L2/out-of-L2 `count_i32_compare` 0.91x/1.00x roofline, and 65,536-batch point reads 247.7M/251.5M lookups/s at p50 137/132us. **Next:** isolate the int4 scalar-count reduction family and its serial parity kernels. |
| 31,858 | `crates/protocol/src/bin/gpu-db-server.rs` | QUEUED |
| 20,164 | `crates/replication/src/lib.rs` | QUEUED |
| 12,958 | `crates/engine/src/engine_residency.rs` | QUEUED |
| 11,173 | `crates/engine/src/engine_expr.rs` | QUEUED |
| 10,271 | `crates/protocol/src/lib.rs` | QUEUED |

**Wave 2 — other critical production files (5,001–10,000 lines).** Start after each affected crate has a stable
module map; work may run independently across crates but GPU validation remains serialized.

| Lines | File | Disposition / evidence |
|---:|---|---|
| 9,410 | `crates/engine/src/engine_streaming_exec.rs` | QUEUED |
| 7,749 | `crates/wal/src/lib.rs` | QUEUED |
| 7,445 | `crates/sql/src/lib.rs` | QUEUED |

**Wave 3 — production review outliers (2,001–5,000 lines).** Analyze after Waves 1–2 establish the relevant
facades, unless one is a safe leaf extraction that directly reduces an earlier wave.

| Lines | File | Disposition / evidence |
|---:|---|---|
| 4,727 | `crates/engine/src/engine_dml_concurrent.rs` | QUEUED |
| 4,636 | `crates/engine/src/mvcc_read_exec.rs` | QUEUED |
| 3,869 | `crates/engine/src/engine_retained_read.rs` | QUEUED |
| 3,437 | `crates/engine/src/engine_sql_pg.rs` | QUEUED |
| 2,696 | `crates/write_conveyor/src/wal_segment.rs` | QUEUED |
| 2,255 | `crates/engine/src/engine_write_apply.rs` | QUEUED |
| 2,195 | `crates/engine/src/engine_dml_prepare.rs` | QUEUED |
| 2,135 | `crates/engine/src/rel_exec_helpers.rs` | QUEUED |

**Wave 4 — test-suite outliers (over 3,000 lines).** Split by behavioral seam and fixture ownership after, or
alongside, the production module they cover; do not fragment tests merely to reduce a count.

| Lines | File | Disposition / evidence |
|---:|---|---|
| 10,369 | `crates/engine/src/tests/resident_expr.rs` | QUEUED |
| 6,460 | `crates/engine/src/tests/streaming_exec.rs` | QUEUED |
| 5,827 | `crates/engine/src/tests/sql_pg.rs` | QUEUED |
| 5,374 | `crates/engine/src/tests/mvcc_bundles.rs` | QUEUED |
| 5,206 | `crates/engine/src/tests/intent_fast_path.rs` | QUEUED |
| 4,649 | `crates/engine/src/tests/resident_route.rs` | QUEUED |
| 3,958 | `crates/engine/src/tests/sql_catalog.rs` | QUEUED |
| 3,502 | `crates/engine/src/tests/mvcc_query.rs` | QUEUED |

**Wave 5 — example and tool outliers (over 3,000 lines).** Determine whether each is a cohesive executable,
handwritten tool, reproducible generated artifact, or obsolete evidence before choosing modules or an exception.

| Lines | File | Disposition / evidence |
|---:|---|---|
| 14,889 | `scripts/generate_research_paper_mechanism_links.py` | QUEUED |
| 4,738 | `crates/write_conveyor/examples/write_conveyor_bench.rs` | QUEUED |
| 3,634 | `crates/server/examples/p8_engine_pgwire_benchmark_endpoint.rs` | QUEUED |
| 3,137 | `scripts/run_p8_ch_benchmark_residency_probe.sh` | QUEUED |

STRUCT-001 closes only when every row has an audited disposition; every accepted retention appears in the
`CODE_SIZE.md` exception registry; no non-excepted production file exceeds 2,000 lines and no non-excepted
test/example/tool exceeds 3,000; all references resolve; targeted gates pass after every extraction; and a fresh
inventory finds no unowned outlier. Line-count drift is expected, so the fresh inventory—not this snapshot—is
the final acceptance source.

## Work ledger

| ID | State | Priority | Outcome and acceptance gate | Dependencies / trigger | Design or evidence |
|---|---|---:|---|---|---|
| **STRUCT-001Q** | NOW | P0 | Extract nullable int4 equality and scalar-comparison count reductions plus their serial parity kernels into one private module below 1,500 lines. Move `validity_bitmap_kernel_arg` with them, re-import it parent-privately for later scalar-stats users, and preserve public count methods plus test access through the root. Prove byte-level move equivalence across the non-contiguous serial/parallel functions, exact inventory, NULL/count parity and concurrent-pool HAZARD gates, independent audit, and the canonical report card. | STRUCT-001P; count-family ownership packet | `CODE_SIZE.md`; scalar COUNT roofline contract |
| **STRUCT-001** | NOW | P0 | Analyze and disposition every source-size outlier through the method and ordered inventory above. Decompose by ownership, register a bounded exception, or prove generated/archive/delete status; update all references and pass targeted gates. Close only when a fresh inventory has no unowned outlier. | None | `docs/CODE_SIZE.md` |
| **R3-001** | NOW | P0 | Audit the current lane, chunk-authoritative, MVCC-sidecar, and recovery implementations against the target write model; choose the surviving version-storage/index/CC design in an ADR. Explicitly disposition the retired mega-fuse idea rather than reviving it from archived handovers. No implementation begins from an unaccepted proposal. | None | `docs/design/write-path-design-inputs.md` |
| **BENCH-001** | NOW | P0 | Open-loop offered-rate harness reports p50/p99/p99.9/p99.99 and saturation TPS against tuned PostgreSQL on the same host, split by deterministic-fast and interactive-slow transaction classes. Exclude warm-up from sustained metrics and publish the exact Postgres/host configuration. Results identify whether the residual is GPU-architectural or host-serial. | Quiet benchmark window and reproducible Postgres config | ADR-008; ARCHITECTURE §9 |
| **R3-002** | NEXT | P0 | Extend the GPU-native write/read fast path beyond int4-PK: numeric/UUID, bool, wider fixed-width types, then variable-width text and compound keys. For every graduated shape, DML locate and constraint validation use device indexes/predicates without `CachedShardPkIndex` or host-probe fallback; GPU-fired differentials, recovery parity, and mixed read/write coverage are required. | R3-001 | Type-coverage evidence in archived handovers; `docs/design/non-int4-index-design-inputs.md` |
| **R3-003** | NEXT | P0 | Complete deterministic concurrency control and transaction-held snapshot semantics, including write-write conflicts and the chosen sparse/version metadata model. Add bounded VACUUM/GC with active-reader fencing and update-heavy capacity gates. | R3-001 | ADR-009; ARCHITECTURE §10 |
| **R3-004** | BLOCKED | P0 | Delete the host write/commit/MVCC tuple-store relational data path, including `CachedShardPkIndex`, host DML/constraint probes, and their fallback dispatch. Recovery reconstructs device-native state without acknowledged-commit loss; production and tests contain no host relational execution. | R3-002, R3-003, DUR-002 | ADR-006/007 |
| **R3-005** | VERIFY | P1 | Reproduce or close the lane DELETE residuals recorded at Tier-1 closeout: duplicate same-key deletes must not double-count, and a zero-row delete must not poison a retryable same-key insert through a stale ledger slot. | Current lane path | Archived pre-unification handover |
| **RETIRE-001** | NEXT | P1 | Replace `new_local_cpu_oracle`, `CpuMvccExecutionBackend`, `FirstCudaSliceParityBackend`, and host SQL finalization fixtures with GPU-native or closed-form specification oracles, preserving semantic coverage before deletion. | Per-module GPU oracle coverage | ADR-007 |
| **RETIRE-002** | BLOCKED | P1 | Replace chunk reverse-gather, deauthorization, and scan-build repair with device-native DDL/recovery/import repair; then delete those host relational repair operators. Acked commits remain recoverable after every injected repair failure. | Device-native DDL validation and recovery repair | ADR-006; STRATA repair boundary |
| **RETIRE-003** | NEXT | P1 | Remove host relational post-processing from the generic CUDA-MVCC source path: selection compaction, ordering, projection, and result assembly stay device-resident until the one bounded final readback. Delete the host compaction/sort/project helpers and make unsupported shapes fail loud rather than return a CPU-computed result. | Device-resident generic MVCC result representation and per-shape GPU differentials | ADR-006/007 |
| **DUR-001** | NEXT | P1 | Add an automatic intent-lane checkpoint policy and timestamped lane records sufficient for archive/PITR. Keep explicit operator checkpointing and refusal behavior until both are crash-gated. | Cadence and timestamp format decision | Archived durable-path handover and write-conveyor record |
| **DUR-002** | NEXT | P0 | Crash/power-fail campaign covers FUA lanes, checkpoint sidecar, WAL truncation, cold artifacts, recovery, and post-durable apply failure. Before multi-entry apply, bind DDL existence/dependency helpers to the working catalog and add a multi-entry regression. No acknowledged commit is lost and rejected commits never become visible. | Test harness and bounded artifact budget | ARCHITECTURE §15 |
| **HA-001** | BLOCKED | P1 | Wire engine sequencing to replicated log indices; lane claims reserve Raft log-index ranges and client acknowledgement waits for quorum commit. Add follower rejection, catch-up, promotion/fencing, and snapshot-install gates. | R3 sequencing contract and multi-node runtime | ADR-001/004/005 |
| **READ-001** | VERIFY | P1 | Reproduce or close the remaining correctness debt against the current tree: filtered expression overflow ordering and route-gate case handling. Every live defect receives a focused GPU/spec regression; closed findings leave no task behind. | None | STATUS known-debt facts |
| **READ-002** | NEXT | P2 | Provide O(1) point lookup for bigint/text/UUID/numeric and composite keys where measurement justifies it. Each route is byte-identical to the GPU scan and proves a nonzero index-hit counter. | BENCH-001 may reorder | `docs/design/non-int4-index-design-inputs.md` |
| **READ-003** | VERIFY | P2 | Confirm an empty filtered SUM/AVG/MIN/MAX reaches a real pgwire client as a typed SQL NULL. Add one end-to-end GPU test if the existing engine and wire-unit coverage do not cross that seam. | Current aggregate and pgwire paths | Archived M3 proposal acceptance criteria |
| **PERF-001** | VERIFY | P2 | Re-profile general row-producing result paths and scan kernels before adopting archived optimization hypotheses. Promote only measured bottlenecks; compare both report-card layers and cache regimes. | BENCH-001 or a demonstrated regression | Archived optimization analyses |
| **MULTI-001** | BLOCKED | P1 | Run and pass the existing non-vacuous STRATA scheduler/budget test on at least two physical GPUs; record per-device completed work and failure isolation. | Access to a >=2-GPU host | STATUS multi-GPU gap |
| **CFG-001** | NOW | P2 | Reckon remaining product/runtime flags and setters: winner becomes unconditional, losing arm and knob are deleted together. Remove superseded sequencer/classic-path and dead benchmark knobs when their code is touched. | Replacement path complete and gated | `CONFIG.md` |
| **TOOL-001** | NEXT | P2 | CI compiles the permanent `probe-timing` instrumentation feature so probes cannot bit-rot. Agent guidance already requires reuse. | CI edit window | Archived instrumentation proposal |
| **PRODUCT-001** | NEXT | P1 | Consolidate the three pgwire servers into one protocol-neutral serving path and invert the engine-to-protocol dependency. Preserve all driver and pgwire golden gates. | Stable execution interfaces | ARCHITECTURE §2–4 |
| **PRODUCT-002** | NEXT | P1 | Close PostgreSQL type/protocol/catalog breadth: text+binary codecs, OID/typmod, typed NULL parameters, persistent GPU system relations, catalog/function execution, large NUMERIC, and checked aggregate overflow. Every type graduates to a GPU route. Host catalog work may retain DDL bookkeeping and row encoding/upload only; filtering, joining, sorting, validation, and result-value decisions execute on-device. | R3-002 for write-capable types | ARCHITECTURE §3 |
| **ROUTE-001** | NEXT | P2 | Productize the OLTP route classes beyond PK microbenchmarks: tenant/security-filtered page reads, bounded two-table joins with a fanout contract, and computed-detail routes with resident summaries/invalidation. | BENCH-001 workload evidence | CHARTER transaction model; ARCHITECTURE §9 |
| **PRODUCT-003** | PARKED | P2 | Production hardening: packaging, SBOM/audit/deny, GPU CI fatbins, panic/unsafe review, Prometheus/OTLP/audit logging, mTLS/channel binding, credential management, and deployment runbooks. | Core correctness and recovery gates | Trigger: v1 release candidate |
| **SCALE-001** | NEXT | P2 | Replace unbounded connection/thread/queue behavior with explicit admission and bounded ownership domains; validate 100k+ logical connections and bounded result streaming. | BENCH-001 workload model | ARCHITECTURE §4/§9 |
| **SCALE-002** | VERIFY | P2 | Disposition the legacy scalability ledger against the current tree: uncapped per-row locate loops, populate-vs-commit admission race, first-transition elision TOCTOU, eviction rollback, and the recorded 32-writer PK inversion. Keep only reproducible issues. | None | Archived pre-unification handover/reviews |
| **MEDIA-001** | BLOCKED | P2 | Re-run low-client latency and 30-second attribution on PLP-class NVMe to separate software cost from consumer-drive FUA stalls. | PLP-class storage hardware | Archived durable-path handover |
| **SIDE-001** | PARKED | P3 | GPU-resident Redis/KV product exploration. It is not part of the database v1 plan. | Trigger: explicit post-v1 product decision | `docs/archive/future/redis/` |

## Legacy host-engine deletion coverage

| Remaining host surface | Owning task | Deletion boundary |
|---|---|---|
| Test-only SELECT/MVCC semantic oracle and host SQL finalization | **RETIRE-001** | GPU/specification oracles preserve every semantic fixture before source deletion |
| Generic CUDA-MVCC host compaction, sorting, projection, and result assembly | **RETIRE-003** | Device-resident result pipeline reaches the single final readback; unsupported work fails loud |
| Host tuple store, write/apply engine, `CachedShardPkIndex`, and DML/constraint host probes | **R3-002**, **R3-003**, **R3-004** | All supported writes/constraints are device-native; recovery and CC gates pass; host sources and dispatch are deleted |
| Reverse gather, deauthorization, scan-build, and DDL/recovery/import repair | **RETIRE-002** | Device-native repair preserves RPO under injected failures before host repair deletion |
| Catalog construction boundary | **PRODUCT-002** | Host retains bookkeeping and deterministic encoding/upload only; all relational catalog decisions execute on-device |

## Standing acceptance gates

- WAL-before-visibility and `commit >= applied >= visible` monotonicity.
- GPU execution must be non-vacuous; production relational decline/fault is fail-loud, never host fallback.
- Relevant GPU tests use timeouts, run serially for sweeps, and never use `--gpu-reset`.
- Read-kernel/residency/result changes run `scripts/benchmark_report_card.sh`; compare ratios, both layers,
  and both cache regimes.
- Durability/HA changes include restart, torn/failing I/O, and acknowledged-commit recovery tests.
- Documentation changes pass the single-plan audit described in `docs/README.md`.

## Explicit non-goals until post-v1

Automated multi-region reconfiguration, broad extension compatibility, and the Redis/KV side product are not
active work unless the user promotes their task IDs.
