# GPU DB End-to-End Architecture Design Space

This document defines the architecture-search model used to compose and score
end-to-end GPU DB architecture candidates from the completed research dataset.
It does not claim a globally optimal design. It defines a constrained search
space, hard correctness filters, objective axes, workload classes, evidence
levels, and proof gates so candidate families can be compared under explicit
assumptions.

Source inputs:

- mechanism catalog:
  `docs/research/architecture-compatibility/mechanisms.json`
- compatibility edges:
  `docs/research/architecture-compatibility/compatibility-edges.json`
- generated compatibility view:
  `docs/research/architecture-compatibility.md`
- benchmark/proof backlog:
  `docs/research/architecture-compatibility/benchmark-backlog.md`
- system invariants:
  `docs/architecture/01-system-invariants.md`
- CPU/GPU execution contract:
  `docs/architecture/04-execution-model-cpu-gpu.md`
- session/admission contract:
  `docs/architecture/09-session-management-and-admission.md`
- P8 retained storage design:
  `docs/architecture/10-p8-gpu-optimized-storage-engine.md`
- production runtime target:
  `docs/architecture/11-high-throughput-query-runtime.md`
- ACID/GPU memory envelope:
  `docs/architecture/12-acid-isolation-and-gpu-memory.md`
- current benchmark status:
  `docs/testing/benchmarks/README.md`

## Search Thesis

The search is constrained multi-objective architecture selection. Correctness
and operability constraints are hard filters. Performance, latency, scale,
memory efficiency, implementation risk, operational clarity, and proof quality
are scored only after a candidate passes those filters.

The current engine evidence favors an architecture that keeps WAL and CPU/MVCC
state as the durable truth, publishes immutable read generations for retained
GPU execution, routes by explicit freshness and fallback contracts, and admits
work through bounded resource budgets. More aggressive GPU-resident write or
all-in-GPU designs remain candidates only if they preserve the same visibility,
recovery, fallback, and admission contracts.

## Hard Constraint Checklist

Any candidate that fails one hard constraint is rejected before scoring.

| Constraint | Required property | Source mechanisms | Source artifacts or gates |
| --- | --- | --- | --- |
| WAL-before-visibility | No SQL-visible state, route eligibility, resident snapshot, or metadata generation advances before durable commit evidence exists. | `wal_before_visibility`, `immutable_route_roots`, `dependency_witnesses` | `01-system-invariants.md`; `12-acid-isolation-and-gpu-memory.md`; `benchmark-wal_before_visibility`; `benchmark-immutable_route_roots` |
| Durable source of truth | GPU memory, encoded responses, and resident indexes are acceleration state only; recovery rebuilds from WAL/checkpoint/archive plus CPU-visible state. | `wal_before_visibility`, `semantic_crash_oracle`, `retained_gpu_snapshots`, `multi_tier_placement` | `10-p8-gpu-optimized-storage-engine.md`; `12-acid-isolation-and-gpu-memory.md`; `benchmark-semantic_crash_oracle` |
| SQL-visible semantic parity | CPU, GPU, retained, cold-transfer, and fallback routes return the same SQL-visible results for supported semantics. | `cpu_fallback_policy`, `isolation_trace_oracle`, `cost_based_route_optimizer`, `retained_gpu_snapshots` | `01-system-invariants.md`; `04-execution-model-cpu-gpu.md`; `benchmark-cpu_fallback_policy`; `benchmark-isolation_trace_oracle` |
| Isolation traceability | Every returned read can be tied to an allowed visibility boundary, snapshot frontier, fallback route, or wait/reject decision. | `snapshot_frontier_vectors`, `isolation_trace_oracle`, `mvcc_gc_frontiers`, `htap_freshness_router` | `12-acid-isolation-and-gpu-memory.md`; `benchmark-snapshot_frontier_vectors`; `benchmark-mvcc_gc_frontiers`; `benchmark-htap_freshness_router` |
| Crash/recovery proof | Crash states recover to a valid durable prefix and never make speculative GPU, route, catalog, or residency state externally eligible. | `semantic_crash_oracle`, `wal_before_visibility`, `dependency_witnesses`, `immutable_route_roots` | `12-acid-isolation-and-gpu-memory.md`; `benchmark-semantic_crash_oracle`; `benchmark-wal_before_visibility` |
| Explicit route and fallback semantics | Every unsupported, stale, nonresident, over-budget, low-benefit, or saturated accelerated path has a named fallback, wait, reject, or retry decision. | `cpu_fallback_policy`, `htap_freshness_router`, `cost_based_route_optimizer`, `vector_credit_admission` | `04-execution-model-cpu-gpu.md`; `09-session-management-and-admission.md`; `11-high-throughput-query-runtime.md`; `benchmark-cpu_fallback_policy` |
| Bounded admission and queues | Sessions, mutation work, read work, GPU work, residency work, WAL slots, buffers, and responses have explicit budgets and overload outcomes. | `vector_credit_admission`, `effective_session_counting`, `owner_ring_bundling`, `resource_dag_scheduling`, `deficit_fairness` | `09-session-management-and-admission.md`; `11-high-throughput-query-runtime.md`; `benchmark-vector_credit_admission`; `benchmark-effective_session_counting` |
| Bounded memory and snapshot lifetime | Resident buffers, descriptor generations, old MVCC versions, pinned buffers, and retired route/catalog handles are reclaimed at named frontiers. | `bounded_descriptor_reclamation`, `mvcc_gc_frontiers`, `stable_handle_indirection`, `snapshot_frontier_vectors`, `retained_gpu_snapshots` | `10-p8-gpu-optimized-storage-engine.md`; `11-high-throughput-query-runtime.md`; `benchmark-bounded_descriptor_reclamation`; `benchmark-mvcc_gc_frontiers` |
| Operator-observable failures | Saturation, fallback, stale residency, invalidation, recovery, replay, and benchmark non-coverage are observable as classed facts, not hidden slow paths. | `cpu_fallback_policy`, `vector_credit_admission`, `semantic_crash_oracle`, `cost_based_route_optimizer` | `01-system-invariants.md`; `09-session-management-and-admission.md`; `docs/testing/benchmarks/README.md`; `benchmark-cpu_fallback_policy` |

## Objective Axes

Scores use a 1 to 5 ordinal scale after hard constraints pass. A score of 5 is
best for the named objective under the workload class being evaluated. The
weights below are the default search weights for the first candidate pass; P4
may include sensitivity notes if a candidate wins only under a narrow workload
weighting.

| Axis | Weight | Score direction | Primary evidence |
| --- | ---: | --- | --- |
| Write throughput | 12 | Higher sustained COPY/INSERT/UPDATE throughput with durable visibility proof. | `wal_before_visibility`, current 10% COPY benchmark evidence, `benchmark-wal_before_visibility` |
| Retained-read p50/p99 | 14 | Lower retained lookup/aggregate latency without stale or unsupported route leakage. | `retained_gpu_snapshots`, `same_shape_microbatching`, current retained-route reports |
| Mixed workload stability | 12 | Stable read latency and write progress under refresh, invalidation, and fallback pressure. | `htap_freshness_router`, `snapshot_frontier_vectors`, `mvcc_gc_frontiers` |
| 1M logical-session viability | 10 | Low idle-session cost, bounded active-flow cost, and explicit response backpressure. | `effective_session_counting`, `vector_credit_admission`, `11-high-throughput-query-runtime.md` |
| Recovery time and proof quality | 10 | Fast, deterministic recovery from WAL/checkpoint/archive with crash-oracle coverage. | `semantic_crash_oracle`, `wal_before_visibility`, `12-acid-isolation-and-gpu-memory.md` |
| HBM/DRAM/NVMe efficiency | 10 | Good useful-work-per-byte across resident, warm, cold, and over-resident paths. | `multi_tier_placement`, `log_structured_warm_tier`, `db_owned_cold_objects` |
| Implementation complexity | 8 | Smaller, staged implementation surface with local proof slices. | current architecture docs, benchmark backlog decision status |
| Operational clarity | 8 | Clear owner domains, telemetry, fallback reasons, and incident controls. | `cpu_fallback_policy`, `vector_credit_admission`, `owner_ring_bundling` |
| Benchmarkability | 8 | Architecture assumptions map to bounded correctness gates and performance probes. | `benchmark-backlog.md`, current P8 benchmark methodology |
| Extensibility | 8 | Supports later partitioning, tiering, optimizer, and session-scale extensions without violating the spine. | compatibility edges, `multi_tier_placement`, `cost_based_route_optimizer` |

## Target Workload Classes

Candidates must state which workload classes they primarily optimize and which
classes they deliberately leave to fallback, prototype, or future work.

| Workload class | Search question | Dominant mechanisms | Initial proof gate |
| --- | --- | --- | --- |
| COPY/INSERT-heavy ingest | Can write admission clear durable throughput targets while preserving WAL-before-visibility and invalidation order? | `wal_before_visibility`, `vector_credit_admission`, `owner_ring_bundling`, `deterministic_hot_write_templates` | `benchmark-wal_before_visibility`; current COPY admission reports |
| Hot point lookups | Can repeated key lookups execute from retained generations with low p50/p99 and result-sized D2H? | `retained_gpu_snapshots`, `same_shape_microbatching`, `stable_handle_indirection`, `cpu_fallback_policy` | `benchmark-retained_gpu_snapshots`; current retained lookup reports |
| Retained aggregates | Can resident scans and reductions amortize GPU work without hiding refresh or stale-route cost? | `retained_gpu_snapshots`, `same_shape_microbatching`, `cost_based_route_optimizer` | `benchmark-same_shape_microbatching`; current retained aggregate reports |
| Mixed HTAP freshness reads | Can freshness-sensitive reads choose wait, retained, CPU, or reject paths without violating isolation? | `htap_freshness_router`, `snapshot_frontier_vectors`, `mvcc_gc_frontiers`, `isolation_trace_oracle` | `benchmark-htap_freshness_router`; `benchmark-isolation_trace_oracle` |
| Over-resident partitioned reads | Can data larger than HBM be partitioned, placed, routed, and reduced with explicit movement costs? | `multi_tier_placement`, `log_structured_warm_tier`, `db_owned_cold_objects`, `resource_dag_scheduling` | `benchmark-multi_tier_placement`; current 125% readiness blocker |
| High-concurrency idle plus bursty active sessions | Can 1M logical sessions exist while only active flows consume hot-path resources? | `effective_session_counting`, `vector_credit_admission`, `deficit_fairness`, `resource_dag_scheduling` | `benchmark-effective_session_counting`; `benchmark-vector_credit_admission` |
| Crash/recovery/replay paths | Can every publication boundary recover to a valid durable prefix and reject speculative state? | `semantic_crash_oracle`, `wal_before_visibility`, `dependency_witnesses`, `immutable_route_roots` | `benchmark-semantic_crash_oracle`; `benchmark-wal_before_visibility` |

## Scoring Semantics

Candidate scoring has four layers:

1. Hard constraints: pass or fail. A failure rejects the candidate.
2. Weighted objective score: each axis receives 1 to 5 points, multiplied by
   the axis weight.
3. Confidence: evidence confidence is scored independently so a speculative
   candidate cannot beat a proven candidate solely by optimistic point values.
4. Risk penalties: implementation, compatibility, operability, and benchmark
   uncertainty subtract from the weighted score.

Score definitions:

- 5: Strong expected fit, direct source evidence, and a bounded proof gate.
- 4: Good fit with clear mechanism support and only moderate unknowns.
- 3: Plausible fit, but important assumptions are still unproven.
- 2: Weak fit or high operational/implementation friction.
- 1: Poor fit for the objective, even if the hard constraints pass.

Confidence levels:

- `implemented`: supported by current code and accepted benchmark or test
  evidence.
- `adopted-invariant`: required by architecture docs and mechanism decision
  status, with correctness gates defined.
- `prototype-backed`: mechanism has enough reviewed evidence for a first
  prototype but needs local measurement before adoption.
- `benchmark-only`: mechanism should influence experiments, not the baseline
  architecture.
- `deferred`: mechanism should not shape the current preferred architecture.

Risk penalty levels:

- `0`: already implemented or purely documentation-level.
- `-1`: local prototype needed but ownership boundaries are clear.
- `-2`: cross-component change with unresolved performance or memory behavior.
- `-3`: large semantic, recovery, scheduling, or operator-risk surface.
- `reject`: violates a hard constraint or a compatibility rule.

Evidence strength:

- `direct`: current architecture doc, implementation map, accepted benchmark
  report, or `adopt_now` mechanism.
- `graph-supported`: mechanism card plus compatibility edges support the claim.
- `paper-traced`: paper-mechanism traceability supports the claim, but local
  proof remains required.
- `assumption`: explicitly named assumption pending a proof gate.

## Baseline Search Assumptions

- WAL/checkpoint/archive plus CPU-replayed MVCC state remain the durable source
  of truth.
- GPU resident snapshots, indexes, and encoded responses are performance tiers,
  not durability tiers.
- Immutable route and residency generations are the default publication shape.
- CPU fallback remains a correctness-preserving route, not a performance claim.
- Retained GPU reads are valuable only when residency, freshness, queue, and
  response costs are explicitly measured.
- 1M logical-session viability depends on active-flow admission and response
  backpressure, not one thread or hot allocation per logical session.
- Over-resident data requires partition/tier routing before full 125% benchmark
  claims can be made.
- Learned optimizer advice is deferred until deterministic route telemetry and
  guarded cost baselines exist.

## Research-Derived Design Dimensions

P2 groups the mechanism graph into architecture dimensions. A candidate is a
vector of choices across these dimensions, not a bag of isolated papers. The
state column says how the mechanism may shape the first candidate set:

- `baseline invariant`: required for every valid candidate.
- `candidate default`: expected in the preferred or fallback family, pending
  the named proof gate.
- `optional extension`: may strengthen a family but cannot be required for the
  first preferred architecture.
- `benchmark-only experiment`: useful for measurements, but not part of the
  baseline claim until the gate decides it.
- `deferred feature`: excluded from current candidate construction.

| Dimension | Mechanism | State | Candidate role | Required coupling or rule | First proof gate |
| --- | --- | --- | --- | --- | --- |
| Durability and visibility | `wal_before_visibility` | baseline invariant | Durable source of truth and visibility publication gate. | Every route, resident snapshot, catalog root, and metadata generation must lag durable commit evidence. | `benchmark-wal_before_visibility` |
| Metadata publication | `immutable_route_roots` | baseline invariant | Common publication shape for route, catalog, layout, residency, and visibility generations. | Requires WAL-before-visibility and reclaimable old roots before external eligibility advances. | `benchmark-immutable_route_roots` |
| Metadata publication | `dependency_witnesses` | candidate default | Explains async body persistence and delayed route eligibility. | Strengthens immutable roots and crash proof; required before log-structured warm-tier claims. | `benchmark-dependency_witnesses` |
| Validation | `semantic_crash_oracle` | baseline invariant | Recovery and unsafe-publication proof harness. | Strengthens WAL and dependency witnesses; required for DB-owned cold-object recovery claims. | `benchmark-semantic_crash_oracle` |
| Validation | `isolation_trace_oracle` | baseline invariant | External proof that returned reads match declared isolation and route choices. | Required by CPU fallback; strengthens retained snapshots and freshness routing. | `benchmark-isolation_trace_oracle` |
| MVCC and snapshot frontiers | `snapshot_frontier_vectors` | candidate default | Cross-owner snapshot proof with scalar hot-path generations where possible. | Required by retained snapshots and HTAP freshness routing. | `benchmark-snapshot_frontier_vectors` |
| MVCC and snapshot frontiers | `mvcc_gc_frontiers` | candidate default | Bounded version memory and long-reader accounting. | Strengthens snapshot vectors; has tension with retained snapshots because residency extends version lifetime. | `benchmark-mvcc_gc_frontiers` |
| Memory reclamation | `bounded_descriptor_reclamation` | candidate default | Bounded lifetime for route, catalog, plan, residency, and old-root descriptors. | Strengthens immutable roots; compatible with MVCC GC and stable handles. | `benchmark-bounded_descriptor_reclamation` |
| Storage layout | `stable_handle_indirection` | candidate default | Logical identity survives movement, compaction, and residency changes. | Strengthens retained snapshots; required by multi-tier placement. | `benchmark-stable_handle_indirection` |
| Execution | `retained_gpu_snapshots` | candidate default | Main low-latency retained-read acceleration path. | Requires snapshot frontiers and immutable roots; must be guarded by stale-route rejection, CPU fallback, and MVCC retention bounds. | `benchmark-retained_gpu_snapshots` |
| Execution | `same_shape_microbatching` | candidate default | Kernel-launch amortization for repeated prepared retained routes. | Requires vector-credit admission; strengthens retained snapshots; tension with fairness when batches grow. | `benchmark-same_shape_microbatching` |
| Execution and fallback | `cpu_fallback_policy` | baseline invariant | Correctness-preserving route for stale, unsupported, saturated, low-benefit, or over-budget work. | Requires isolation trace oracle; strengthens retained snapshots by making rejection explicit. | `benchmark-cpu_fallback_policy` |
| Routing | `htap_freshness_router` | candidate default | Chooses retained, wait, refresh, CPU, reject, or retry paths by freshness and benefit. | Requires snapshot frontier vectors; strengthened by CPU fallback, retained snapshots, and isolation traces. | `benchmark-htap_freshness_router` |
| Query optimization | `cost_based_route_optimizer` | candidate default | Deterministic route choice over CPU, GPU, retained, refresh, transfer, and cold paths. | Requires vector credits and strengthens freshness routing; learned advice may only sit behind this guard. | `benchmark-cost_based_route_optimizer` |
| Query optimization | `learned_optimizer_advisor` | deferred feature | Future hinting layer for route, knob, or placement suggestions. | Requires fallback baseline and isolation trace oracle; has tension with WAL/visibility guardrails if allowed to affect correctness. | `benchmark-learned_optimizer_advisor` |
| Runtime admission | `vector_credit_admission` | baseline invariant | Bounded admission across requests, bytes, pinned buffers, GPU streams, WAL slots, and responses. | Strengthens effective session counting and resource DAG scheduling; required by batching and costed routes. | `benchmark-vector_credit_admission` |
| Runtime admission | `effective_session_counting` | candidate default | 1M logical-session model where only ready work and blocked responses consume hot capacity. | Requires vector credits; cannot imply one hot allocation, thread, or queue slot per logical session. | `benchmark-effective_session_counting` |
| Runtime scheduling | `owner_ring_bundling` | candidate default | Low-allocation owner-local drain loop and natural micro-batch formation. | Requires vector credits; strengthened by deficit fairness and deterministic hot-write templates. | `benchmark-owner_ring_bundling` |
| Runtime scheduling | `resource_dag_scheduling` | optional extension | Dependency-aware packing for refresh, transfer, query, write, kernel, and response work. | Strengthened by vector credits; compatible with dependency witnesses and batching; tension with fairness complexity. | `benchmark-resource_dag_scheduling` |
| Runtime scheduling | `deficit_fairness` | candidate default | Bounds starvation caused by batching or throughput-favorable owner scheduling. | Strengthens owner rings; must cap same-shape batching delay. | `benchmark-deficit_fairness` |
| Transaction execution | `gpu_oltp_conflict_ordering` | benchmark-only experiment | Explore GPU write-path conflict ordering for known-access batches. | Requires WAL-before-visibility; alternative-to deterministic templates; tension with same-shape batching on dynamic conflicts. | `benchmark-gpu_oltp_conflict_ordering` |
| Transaction execution | `deterministic_hot_write_templates` | benchmark-only experiment | Hot-key tail-control experiment using template or queue-positioned write lanes. | Strengthens owner-ring scheduling; alternative to GPU conflict ordering for first write-path experiments. | `benchmark-deterministic_hot_write_templates` |
| Storage placement | `multi_tier_placement` | candidate default | HBM/DRAM/NVMe/object placement by reuse, movement cost, and freshness. | Requires stable handles; strengthens retained snapshots; compatible with costed route optimization. | `benchmark-multi_tier_placement` |
| Storage placement | `log_structured_warm_tier` | benchmark-only experiment | Warm-tier write coalescing and rebuildable metadata experiment. | Requires dependency witnesses and crash oracle; compatible with multi-tier placement but not required for the first preferred family. | `benchmark-log_structured_warm_tier` |
| Storage placement | `db_owned_cold_objects` | optional extension | Cold object lifecycle, scan/index control, backup, and PITR alignment. | Requires immutable roots; strengthened by multi-tier placement; semantic crash oracle is required for recovery claims. | `benchmark-db_owned_cold_objects` |

## Dimension Coupling Notes

- The correctness spine is
  `wal_before_visibility` -> `immutable_route_roots` -> route eligibility,
  with `semantic_crash_oracle` and `isolation_trace_oracle` proving externally
  visible outcomes. Candidate families may vary their acceleration strategy,
  but not this spine.
- Retained GPU reads require a four-way coupling:
  `retained_gpu_snapshots`, `snapshot_frontier_vectors`,
  `mvcc_gc_frontiers`, and `cpu_fallback_policy`. A candidate that keeps
  resident read generations but omits explicit snapshot proof, reclamation
  frontiers, or fallback semantics is rejected.
- Runtime scale requires `vector_credit_admission` before any claim about
  micro-batching, 1M logical sessions, costed routing, owner rings, or GPU
  stream scheduling. Queue growth without a typed overload outcome fails the
  hard admission constraint.
- Tiering requires `stable_handle_indirection` before it can become more than
  an experiment. Without stable handles, movement across HBM, DRAM, NVMe, and
  object tiers can invalidate residency descriptors or snapshot proofs.
- Benchmark-only write acceleration mechanisms are not allowed to displace the
  CPU/WAL authority path in the preferred architecture. They may appear as
  explicit labs for bounded write-shape families.

## Candidate Construction Rules

P3 candidates must be constructed as whole architecture families. A family is
valid only when every selected mechanism has a role in the owner topology,
publication path, routing path, admission path, storage path, recovery path, or
validation path. Isolated mechanism adoption is not a candidate.

### Mandatory Spine

Every candidate starts with this non-negotiable spine:

1. `wal_before_visibility` gates SQL visibility, route eligibility, resident
   snapshot publication, catalog/root publication, and tier metadata
   publication.
2. `immutable_route_roots` publishes route, catalog, layout, residency, and
   visibility generations as immutable roots with reclaimable old bodies.
3. `vector_credit_admission` bounds request, byte, pinned-buffer, GPU-stream,
   WAL-slot, response, and residency-refresh pressure before work enters a hot
   path.
4. `cpu_fallback_policy` gives every unsupported, stale, saturated,
   nonresident, over-budget, low-benefit, or unproven route an explicit CPU,
   wait, reject, retry, or refresh outcome.
5. `semantic_crash_oracle` and `isolation_trace_oracle` are validation
   requirements, not optional tests, because they prove the durable prefix and
   SQL-visible route semantics.

A candidate may be conservative or aggressive about retained GPU work, tiering,
or runtime scheduling, but it cannot weaken this spine.

### Compatibility Rules

The compatibility graph is interpreted as construction logic:

- `requires`: the target mechanism must be present and assigned a concrete
  owner/proof role before the source mechanism may appear in a candidate.
- `strengthens`: the source may appear without the target only if the candidate
  states why the weaker form is deliberately used; otherwise use the pair
  together in the natural direction.
- `compatible`: the pair can co-exist, but P3 must still name the shared
  resource boundary or descriptor that keeps the interaction bounded.
- `tension`: the candidate must name the conflict, the cap or fallback that
  contains it, and the proof gate that decides whether the combination remains
  viable.
- `alternative_to`: the candidate may choose one as its write-path experiment,
  but the preferred baseline cannot depend on both until benchmark gates show a
  reason to merge them.

### Construction Order

Build each candidate in this order so correctness and operability remain ahead
of performance claims:

1. Choose the durable authority and publication model. For this search, the
   default is CPU/WAL/MVCC authority with immutable route roots.
2. Choose the read-visibility model. Retained GPU candidates must bind
   `retained_gpu_snapshots`, `snapshot_frontier_vectors`,
   `mvcc_gc_frontiers`, `bounded_descriptor_reclamation`, and
   `cpu_fallback_policy`.
3. Choose the runtime admission and scheduling model. Any claim about 1M
   sessions, micro-batching, owner rings, GPU streams, or response backpressure
   must flow through `vector_credit_admission`.
4. Choose the routing and optimizer model. Freshness-sensitive routes require
   `htap_freshness_router`; deterministic route selection over CPU, GPU,
   retained, refresh, transfer, and cold paths requires
   `cost_based_route_optimizer`.
5. Choose the storage placement model. Tiered or over-resident candidates must
   include `stable_handle_indirection`; multi-tier candidates then add
   `multi_tier_placement` and may optionally add `db_owned_cold_objects`.
6. Choose optional experiments. `gpu_oltp_conflict_ordering`,
   `deterministic_hot_write_templates`, and `log_structured_warm_tier` may
   appear as bounded labs, not as unproven baseline dependencies.
7. Attach validation gates. Every publication, route, isolation, recovery,
   admission, reclamation, and tiering claim must map to a benchmark/proof gate
   from `benchmark-backlog.md`.

### Candidate Archetype Seeds

P3 should produce 3 to 5 families from these seeds. The seeds are intentionally
architecture-shaped rather than mechanism-shaped.

| Seed | Required mechanisms beyond mandatory spine | Optional mechanisms | Primary workload fit | First rejection risk |
| --- | --- | --- | --- | --- |
| Conservative retained-read evolution | `retained_gpu_snapshots`, `snapshot_frontier_vectors`, `mvcc_gc_frontiers`, `bounded_descriptor_reclamation`, `stable_handle_indirection`, `same_shape_microbatching`, `htap_freshness_router`, `cost_based_route_optimizer`, `effective_session_counting`, `owner_ring_bundling`, `deficit_fairness` | `resource_dag_scheduling`, `multi_tier_placement` | Hot point lookups, retained aggregates, mixed freshness reads within current P8 boundaries | Rejected if retained snapshot routes cannot bound HBM/version lifetime or fallback reasons under write churn. |
| Partition-owner retained snapshots | Conservative seed plus partition-scoped owner topology, `multi_tier_placement`, stronger `snapshot_frontier_vectors` usage, and partition-level route roots | `db_owned_cold_objects`, `resource_dag_scheduling` | Over-resident partitioned reads, mixed HTAP, tenant or partition locality | Rejected if cross-owner snapshot vectors make route preflight too expensive or partition movement breaks stable-handle proofs. |
| Aggressive GPU-resident hot path | Conservative seed plus one benchmark-only write lab: either `gpu_oltp_conflict_ordering` or `deterministic_hot_write_templates` | `resource_dag_scheduling`, selected `log_structured_warm_tier` lab | Known-shape hot writes, repeated prepared reads, narrow high-throughput lanes | Rejected if write acceleration tries to bypass WAL-before-visibility, CPU/MVCC authority, or isolation traceability. |
| Multi-tier freshness router | Partition-owner seed plus `multi_tier_placement`, `db_owned_cold_objects`, and costed transfer/refresh routing | `log_structured_warm_tier` after dependency witnesses and crash oracle pass | HBM/DRAM/NVMe/object placement, cold and warm reads, over-resident scans | Rejected if cold/warm metadata cannot recover from WAL/checkpoint/archive plus immutable roots. |
| Runtime-first 1M-session architecture | Mandatory spine plus `effective_session_counting`, `owner_ring_bundling`, `deficit_fairness`, `same_shape_microbatching`, and optional `resource_dag_scheduling`; retained execution can be narrower than the conservative seed | `retained_gpu_snapshots`, `cost_based_route_optimizer`, `multi_tier_placement` as staged additions | High-concurrency idle plus bursty active sessions, response backpressure, admission stability | Rejected if active-flow accounting hides standing queues or p99 response pressure behind logical-session counts. |

### Named Assumptions For Candidate Scoring

P3 and P4 must use these assumption ids instead of loose prose when a claim is
not yet implemented:

| Assumption id | Statement | Deciding gate |
| --- | --- | --- |
| `A1-retained-refresh-latency` | Retained GPU generations can refresh or invalidate fast enough that common hot reads beat CPU fallback at p50 and p99 under write churn. | `benchmark-retained_gpu_snapshots`; `benchmark-htap_freshness_router` |
| `A2-frontier-preflight-cost` | Snapshot frontier vectors can prove cross-owner visibility without making route preflight dominate short read latency. | `benchmark-snapshot_frontier_vectors`; `benchmark-isolation_trace_oracle` |
| `A3-version-retention-bound` | MVCC and descriptor reclamation frontiers keep old versions, resident generations, and retired route bodies bounded under long readers. | `benchmark-mvcc_gc_frontiers`; `benchmark-bounded_descriptor_reclamation` |
| `A4-vector-credit-sufficiency` | Vector credits can prevent hidden queues across network, WAL, CPU, GPU, pinned memory, residency refresh, and response paths. | `benchmark-vector_credit_admission`; `benchmark-effective_session_counting` |
| `A5-microbatch-latency-cap` | Same-shape micro-batching improves GPU occupancy without violating p99, fairness, or response backpressure caps. | `benchmark-same_shape_microbatching`; `benchmark-deficit_fairness` |
| `A6-tier-route-benefit` | Stable handles plus multi-tier placement can route over-resident data with useful work per transferred byte above CPU fallback. | `benchmark-stable_handle_indirection`; `benchmark-multi_tier_placement`; `benchmark-cost_based_route_optimizer` |
| `A7-cold-object-recovery` | DB-owned cold objects can be recovered, invalidated, and backed up through immutable roots and crash-oracle proofs. | `benchmark-db_owned_cold_objects`; `benchmark-semantic_crash_oracle` |
| `A8-owner-ring-tail-control` | Owner-ring bundling plus deficit fairness can raise throughput without starving interactive or freshness-sensitive classes. | `benchmark-owner_ring_bundling`; `benchmark-deficit_fairness` |
| `A9-write-lab-containment` | GPU conflict ordering or deterministic hot-write templates can be evaluated without changing the durable CPU/WAL authority path. | `benchmark-gpu_oltp_conflict_ordering`; `benchmark-deterministic_hot_write_templates`; `benchmark-wal_before_visibility` |
| `A10-dependency-witness-cost` | Dependency witnesses can explain async body persistence and delayed route eligibility without adding unacceptable publish latency. | `benchmark-dependency_witnesses`; `benchmark-immutable_route_roots` |

### Rejection Rules

Reject a candidate before scoring when any of these conditions hold:

- It publishes a SQL-visible generation, route root, resident snapshot, catalog
  root, tier manifest, or cold-object descriptor without durable commit or
  dependency-witness evidence.
- It treats GPU memory, resident indexes, encoded responses, warm-tier logs, or
  cold objects as durability authority rather than rebuildable acceleration
  state.
- It includes retained GPU reads without snapshot frontiers, MVCC/version
  retention bounds, stale-route rejection, and CPU/fallback semantics.
- It claims 1M logical-session viability without separating idle logical
  sessions from active flows and response backpressure.
- It uses batching, owner rings, or resource-DAG scheduling without vector
  credits and fairness or latency caps.
- It uses multi-tier placement without stable handle indirection and explicit
  movement/freshness costs.
- It adopts learned optimizer advice as a correctness-affecting decision path
  before deterministic route telemetry and fallback guardrails exist.
- It promotes benchmark-only write mechanisms into the preferred baseline
  before WAL, isolation, conflict, and tail-latency gates pass.
- It has no crash/recovery, isolation, fallback, and operator-observability
  validation plan.

## P2 Output Boundary

P2 is complete. The design-space file now contains the objective model,
hard-constraint checklist, mechanism-to-dimension matrix, compatibility
interpretation rules, candidate construction order, candidate archetype seeds,
named assumptions, and rejection rules. P3 can now generate architecture
families without rereading the full literature journal.
