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

1. **STRUCT-001AU — isolate legacy DDL syntax ownership.** Move the shared simple-identifier grammar and
   TRUNCATE, DROP TABLE, ALTER TABLE DROP CONSTRAINT, and unsupported-FK-option parsers into one private syntax
   leaf. Preserve exact accepted forms, names, options, and rejection behavior without moving DDL execution.
2. **MULTI-003 — prove resident-sidecar context isolation on physical multi-GPU.** This newly promoted safety
   gate is blocked on a physical two-GPU host; every typed sidecar owner must reject a foreign primary context
   before CUDA work and leave both contexts reusable.
3. **MULTI-002 — partition device-locate work by CUDA context.** This GPU-native boundary is
   blocked on a physical two-GPU host for its non-vacuous acceptance gate; descriptors must never cross a
   primary context and bounded result metadata must merge deterministically without host relational probing.
4. **STRUCT-001 — analyze and disposition every oversized source file.** Establish safe module boundaries and
   reduce the highest context risks before broad implementation work expands them further.
5. **R3-001 — reconcile the live write path with the target GPU-native write design.** This remains the next
   architecture decision needed before wider write work or host-store deletion; structural extraction may
   precede it, but must not make the decision implicitly.
6. **BENCH-001 — complete the open-loop OLTP comparison.** Run in parallel when benchmark capacity is
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
| 40,308 | `crates/execution/src/lib.rs` | **DECOMPOSE — resident header isolated; root now 17,940 lines.** Handwritten production facade with distinct routing policy, CUDA runtime/context/driver layers, resident-memory ownership, MVCC batch encoding, point-read submission/result ownership, write locate/apply, and scan/aggregate/join/sort/expression kernel families. Public root symbols are consumed by engine, planner, metrics, observability, examples, and server tests; stable crate-root re-exports are the facade. **Completed modules:** `routing.rs` (241), RETIRE-001 `reference_operators.rs` (391), `mvcc_batch.rs` (208), `runtime_contract.rs`, sub-3,000-line test seams, `cuda_context.rs` (953), `cuda_driver.rs` (883), `resident_memory.rs` (405), `resident_header.rs` (92), `resident_sort.rs` (846), `resident_count.rs` (1,066), `resident_gather.rs` (258), `resident_filter.rs` (1,221 after STRUCT-001T safety hardening), `resident_aggregate.rs` (866 after STRUCT-001V safety hardening), `resident_group.rs` (1,282), `group_input.rs` (639), `write_locate.rs` (865 after STRUCT-001Z safety hardening), `write_apply.rs` (1,256 after STRUCT-001AB safety hardening), `resident_sidecar.rs` (1,102 after STRUCT-001AD safety hardening), `unique_coordinate.rs` (195), `point_read_submission.rs` (442), `point_read_submit.rs` (1,308), `point_read_dense.rs` (1,086), `point_read_bloom.rs` (206), `point_read_text.rs` (1,205 after STRUCT-001M/N correctness hardening), and `point_read_rows.rs` (475). STRUCT-001Z made the extracted write/visible-locate boundary total: typed same-context owners, exact index/version extents, on-device packed-slot bounds, reserved fail-closed count sentinels, and guaranteed launched-error draining. The exact execution inventory is 118 tests with 43 active and 75 GPU-ignored; five affected GPU gates passed 15 sequential and 10 concurrent invocations, and independent audit is clean. The remaining cross-context orchestration boundary is promoted as **MULTI-002** rather than hidden behind host probing. STRUCT-001AA isolated fused append publication, incremental int4 index insertion, and device compound-key folding with stable facade. STRUCT-001AB replaced raw addresses with typed owners/exact spans, fixed cross-block header publication ordering, added device text bounds and launched-error drains, and closed cleanly under the 120-test suite plus 24/16 HAZARD matrix. STRUCT-001AC isolated the five resident sidecar layout launchers with stable inherent APIs. STRUCT-001AD made those safe APIs total: owned same-context sources, exact source/destination spans, explicit bool/validity state writes, actual text-blob ownership and device validation, alias rejection, checked engine geometry, and failure drains. Its 124-test suite, 18/12 final HAZARD matrix, and independent re-audit are clean; the physical cross-context branch is **MULTI-003**. STRUCT-001AE isolated the bounded device uniqueness verdict exactly; STRUCT-001AF removed its exposed optimizer-sensitive ordered-index recast allocation and restored the report-card ratio. STRUCT-001AG isolated the exact row-count header PTX/launcher behind the unchanged facade; its direct context/cache gate passed 3 sequential and 2 concurrent invocations, independent re-audit is clean, and the canonical two-layer/two-cache card remained green. **Next:** analyze and disposition the protocol server outlier before selecting its first behavior-preserving ownership slice. |
| 31,858 | `crates/protocol/src/bin/gpu-db-server.rs` | **DECOMPOSE — backend, bootstrap/security, transport, test, bind/describe, dispatch, and extended-query owners isolated; root now 16,541 lines.** Handwritten legacy compatibility binary: 21,203 production lines plus a 10,654-line inline 123-test module, accumulated through 312 file-touching commits. Responsibilities are independently separable: host-backed compatibility session/catalog/DML and constraint model (lines 1–4,455), prepared/portal/cursor and security state (4,456–4,568), listener/startup/SCRAM/ready-loop and extended protocol (4,569–8,184), monolithic statement and catalog/`pg_dump` compatibility dispatch (8,185–19,756), bind/describe parameter rewriting (19,757–21,040), thin `BackendWriter` adapter (21,041–21,202), and inline tests (21,205–31,858). The binary consumes only `gpu_db_protocol` plus wire/security libraries and is directly exercised by protocol unit tests, tokio-postgres/sqlx and external driver smokes, pg_dump/pg_dumpall/psql goldens, benchmark scripts, and the security-posture source guard. It is explicitly the legacy host-relational endpoint pending **PRODUCT-001**; it is not the product GPU execution path and must not gain new relational behavior during decomposition. Target ownership is `backend_adapter`, `security`, `session_state`, `extended_query`, `bind_describe`, catalog metadata/query-family modules, statement dispatch, and invariant-grouped test modules, all behind the unchanged binary boundary with dependencies directed toward protocol library primitives. The private 194-line `backend_adapter` now owns the 28 wrappers/conversions that already delegate framing to `gpu_db_protocol::backend`; normalized definitions, all checks/smokes, the repaired security preflight, and independent audit are clean. The private 798-line `server_bootstrap` now owns configuration, listener/TLS/SCRAM startup, startup framing, and its two focused tests behind a root `main` delegate; exactness, full protocol/driver/security gates, and independent audit are clean, with no relational/session type crossing the seam. The private 127-line `frontend_transport` owns the shared connection trait and length-validated tagged reader. STRUCT-001AK now caps startup/tagged declarations at 64 KiB/64 MiB before allocation, constructs one exact buffer, and closes under 127 binary tests, full driver/security gates, and clean independent audit; aggregate slow-client memory remains **SCALE-001**. STRUCT-001AL externalized the canonically exact 121-test body behind the unchanged `tests` module name; the 127-name inventory and all gates/audit remain clean. STRUCT-001AM split that intermediate owner into a 120-line support parent plus seven coherent 360–2,125-line modules; canonical bodies, the exact 127-item inventory/name map, sequential+concurrent runs, full gates, and audit are clean. STRUCT-001AN isolated the exact 1,152-line bind/describe body in a 1,179-line private module with 15 production entry points, 32 private helpers, three test-only delegates, and clean full gates/audit. STRUCT-001AO isolated the exact ready-loop/frontend state machine in a 261-line private module with one production entry point, two test-only delegates, repaired security guards, and clean full gates/audit. STRUCT-001AP isolated the exact 590-line extended-query body in a 622-line private module with five production entry points and one test-only delegate; statement/portal replacement, bind formats, skip-until-Sync, suspension/resume, COPY delegation, exact errors, full gates, and independent audit are clean. STRUCT-001AQ isolated the canonical-exact SQL compatibility body behind an acyclic 913-line syntax leaf and 301-line PREPARE/DEALLOCATE execution bridge; seven proven exports per module, full gates, and independent audit are clean. STRUCT-001AR isolated the exact cursor parser/executor blocks in a 407-line private module with ten proven entry points, acyclic dependencies, full gates, and independent audit. STRUCT-001AS isolated the exact 197-line legacy extended-DML body in a 205-line private leaf with three one-consumer exports, full gates, and independent audit; inherited integrity/partial-mutation limitations remain compatibility debt rather than claimed guarantees. STRUCT-001AT isolated the exact 273-line COPY state/execution body in a 283-line private module with four proven entry points, full gates, and independent audit. **Next:** isolate legacy DDL syntax under STRUCT-001AU without creating a host product API. Preserve all source guards and driver/golden consumers; run protocol all-target checks, the current exact binary-test inventory, tokio-postgres/sqlx smokes, connection-security preflight, and relevant compatibility goldens for each affected boundary. No extraction may migrate catalog filtering/joining into a new host product API or conflict with **PRODUCT-001/PRODUCT-002** GPU-native system relations. |
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
| **STRUCT-001AU** | NOW | P0 | Analyze and extract the cohesive legacy DDL syntax cluster: simple qualified-identifier validation, `TRUNCATE [TABLE] [ONLY]`, `DROP TABLE [IF EXISTS] ... [CASCADE|RESTRICT]`, `ALTER TABLE ... DROP CONSTRAINT [IF EXISTS] ... [CASCADE|RESTRICT]`, and unsupported foreign-key option recognition. Preserve exact case/comment/semicolon handling, multi-target ordering, option precedence, returned names/flags, and narrow rejection behavior. Expose only proven parent-private parser entry points; DDL mutation, catalog, COPY, and relational execution remain root dependencies, with no host product execution API. Run focused COPY/DDL/catalog tests plus full protocol/driver/security gates and independent audit. | STRUCT-001AT | PRODUCT-001 containment; legacy DDL syntax boundary |
| **MULTI-003** | BLOCKED | P0 | Run a non-vacuous physical multi-GPU isolation matrix for the typed resident-sidecar APIs. Allocate valid bool/validity/text sources and destinations on at least two primary contexts; prove valid work fires on each device, every crossed owner is rejected before launch, and both contexts remain reusable after each rejection. Recompaction stays device-local and GPU-only; do not introduce peer copies or host bitmap/text interpretation merely to satisfy the gate. | Access to a >=2-GPU host | ADR-013; STRUCT-001AD audit |
| **MULTI-002** | BLOCKED | P0 | Partition write/visible-locate descriptor sets by owning CUDA primary context, issue one launch per GPU while retaining the exact index/version generation, and deterministically merge only bounded coordinate/count/error metadata in the host control plane. Key lookup, MVCC visibility, duplicate decisions, and mutation targeting remain device decisions; any device failure fails the whole operation and leaves every context reusable, with no host relational probe/fallback. A non-vacuous gate on at least two physical GPUs must prove both devices perform work, no descriptor crosses context, input order and duplicate/visibility semantics survive the merge, and one-device injected failure cannot yield a partial result. | Access to a >=2-GPU host | ARCHITECTURE §5–6; ADR-013; STRUCT-001Z audit |
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
| **SCALE-001** | NEXT | P2 | Replace unbounded connection/thread/queue behavior with explicit admission and bounded ownership domains; add pre-authentication read deadlines and an aggregate connection-memory budget so slow clients cannot concurrently pin the bounded 64 MiB tagged-frame allowance; validate 100k+ logical connections and bounded result streaming. | BENCH-001 workload model | ARCHITECTURE §4/§9; STRUCT-001AK residual |
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
