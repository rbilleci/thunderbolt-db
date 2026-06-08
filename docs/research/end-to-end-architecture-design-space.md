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

## P1 Output Boundary

This P1 baseline intentionally stops before assigning mechanism states to every
design dimension. P2 will extend this document with a mechanism-to-dimension
matrix, candidate construction rules, and named assumptions derived from the
compatibility edge graph.
