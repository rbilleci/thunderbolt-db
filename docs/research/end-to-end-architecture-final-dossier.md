# GPU DB End-to-End Architecture Final Dossier

This dossier is the P5 output of the end-to-end architecture search. It can be
read without reopening the raw literature journal: architectural claims point
back to the design-space model, candidate scorecard, mechanism ids,
compatibility view, benchmark backlog, paper traceability ids, current
architecture docs, or accepted benchmark reports.

This is not a global mathematical optimum. It is the preferred defensible
architecture under the current constraints: preserve WAL/MVCC correctness,
build on the accepted P8 retained-read direction, keep one-million logical
session viability visible, and defer broader tiering or GPU write authority
until focused gates pass.

Source map:

- search plan: `docs/research/end-to-end-architecture-search-plan.md`
- design-space model: `docs/research/end-to-end-architecture-design-space.md`
- scored candidates: `docs/research/end-to-end-architecture-candidates.md`
- mechanism graph: `docs/research/architecture-compatibility.md`
- mechanism and edge sources:
  `docs/research/architecture-compatibility/mechanisms.json`,
  `docs/research/architecture-compatibility/compatibility-edges.json`
- paper traceability:
  `docs/research/architecture-compatibility/paper-mechanism-links.json`,
  `docs/research/architecture-compatibility/paper-mechanism-coverage.md`
- proof gates: `docs/research/architecture-compatibility/benchmark-backlog.md`
- current architecture fit:
  `docs/architecture/01-system-invariants.md`,
  `docs/architecture/04-execution-model-cpu-gpu.md`,
  `docs/architecture/09-session-management-and-admission.md`,
  `docs/architecture/10-p8-gpu-optimized-storage-engine.md`,
  `docs/architecture/11-high-throughput-query-runtime.md`,
  `docs/architecture/12-acid-isolation-and-gpu-memory.md`
- accepted benchmark evidence: `docs/testing/benchmarks/README.md`

## Executive Thesis

The preferred architecture is Candidate A, conservative retained-read
evolution, with Candidate E's runtime discipline embedded as a mandatory
implementation posture:

Keep CPU/WAL/MVCC as durable authority. Publish immutable retained GPU read
generations for measured hot shapes. Route every read through explicit
freshness, cost, fallback, and isolation decisions. Admit read, write, refresh,
GPU, pinned-buffer, WAL, and response work through vector credits with
owner-ring and fairness telemetry.

The preferred mechanism set is:

- correctness spine: `wal_before_visibility`, `immutable_route_roots`,
  `cpu_fallback_policy`, `semantic_crash_oracle`,
  `isolation_trace_oracle`, `vector_credit_admission`
- retained-read core: `retained_gpu_snapshots`,
  `snapshot_frontier_vectors`, `mvcc_gc_frontiers`,
  `bounded_descriptor_reclamation`, `stable_handle_indirection`
- routing and execution: `htap_freshness_router`,
  `cost_based_route_optimizer`, `same_shape_microbatching`
- runtime discipline: `effective_session_counting`, `owner_ring_bundling`,
  `deficit_fairness`

The excluded baseline mechanisms are `gpu_oltp_conflict_ordering`,
`deterministic_hot_write_templates`, `log_structured_warm_tier`,
`db_owned_cold_objects`, and `learned_optimizer_advisor`. They remain
experiment, long-range, or deferred mechanisms until their proof gates pass.

The architecture is preferred because the current accepted P8 evidence already
shows SQL-visible retained execution through the identical pgwire benchmark
lane, including the 10% run with GPU DB COPY above the 30k rows/sec gate and
retained query correctness through concurrency 64. The same evidence also keeps
the 125% over-resident tier blocked on partitioned resident route execution,
which argues against promoting partition/tier complexity before the retained
single-resident path and runtime gates are closed.

## Component Design

### Durable Authority

The mutation authority remains CPU/WAL/MVCC. Durable WAL evidence gates SQL
visibility, route eligibility, resident snapshot publication, catalog/root
publication, and tier metadata publication. GPU resident buffers, encoded
responses, resident indexes, and warmup state are acceleration state only.

References:

- mechanisms: `wal_before_visibility`, `semantic_crash_oracle`
- architecture docs: `01-system-invariants.md`,
  `10-p8-gpu-optimized-storage-engine.md`,
  `12-acid-isolation-and-gpu-memory.md`
- proof gates: `benchmark-wal_before_visibility`,
  `benchmark-semantic_crash_oracle`
- paper traceability examples:
  `2026-06-03-memory-optimized-mvcc-for-disk-backed-storage`,
  `2026-06-03-modern-nvme-storage-engine-exploitation`,
  `2026-06-07-b3-turns-crash-consistency-into-bounded-witness-generation`

### Immutable Publication

Route, catalog, layout, residency, visibility, and future tier descriptors
publish as immutable bodies with compact root generations. Readers consume
root generations; owners retire old bodies only after active readers and
descriptor protection permit it. The first architecture does not require
async warm-tier publication, but it reserves `dependency_witnesses` for later
metadata that cannot be made externally eligible at body-write time.

References:

- mechanisms: `immutable_route_roots`, `bounded_descriptor_reclamation`,
  `dependency_witnesses`
- architecture docs: `10-p8-gpu-optimized-storage-engine.md`,
  `11-high-throughput-query-runtime.md`
- proof gates: `benchmark-immutable_route_roots`,
  `benchmark-bounded_descriptor_reclamation`,
  `benchmark-dependency_witnesses`
- paper traceability examples:
  `2026-06-03-ankerdb-fine-granular-virtual-snapshotting`,
  `2026-06-04-cross-paper-synthesis-hot-routes-need-separate-execution-and-publication-frontiers`,
  `2026-06-06-orcgc-makes-reclamation-bounds-part-of-the-hot-path-contract`

### Retained GPU Read Tier

The GPU tier holds versioned resident snapshots for admitted hot tables and
query shapes. A retained snapshot names relation identity, schema generation,
source WAL or transaction boundary, visibility boundary, resident layout
identity, supported route families, invalidation generation, and optional
partition identity. New readers enter a retained route only when freshness,
residency, predicate support, route cost, and queue pressure prove the route is
eligible.

References:

- mechanisms: `retained_gpu_snapshots`, `snapshot_frontier_vectors`,
  `mvcc_gc_frontiers`, `stable_handle_indirection`
- architecture docs: `10-p8-gpu-optimized-storage-engine.md`,
  `11-high-throughput-query-runtime.md`,
  `12-acid-isolation-and-gpu-memory.md`
- proof gates: `benchmark-retained_gpu_snapshots`,
  `benchmark-snapshot_frontier_vectors`,
  `benchmark-mvcc_gc_frontiers`, `benchmark-stable_handle_indirection`
- paper traceability examples:
  `2026-06-03-aocc-adaptive-validation-for-heterogeneous-occ`,
  `2026-06-03-mordred-semantic-cpu-gpu-placement`,
  `2026-06-04-snapshot-reconstruction-as-an-optimizable-route`

### Freshness And Fallback Router

Every read is routed by a declared freshness contract, snapshot availability,
estimated benefit, and explicit fallback policy. The initial routes are:
retained GPU, wait-for-refresh, CPU fallback, reject, retry, and cold-transfer
GPU where the current engine already supports a correct path. The architecture
requires classed fallback reasons: unsupported predicate, stale generation,
missing residency, GPU saturation, pinned-buffer pressure, low expected
benefit, response backpressure, and unproven route.

References:

- mechanisms: `htap_freshness_router`, `cost_based_route_optimizer`,
  `cpu_fallback_policy`, `isolation_trace_oracle`
- architecture docs: `04-execution-model-cpu-gpu.md`,
  `09-session-management-and-admission.md`,
  `11-high-throughput-query-runtime.md`
- proof gates: `benchmark-htap_freshness_router`,
  `benchmark-cost_based_route_optimizer`,
  `benchmark-cpu_fallback_policy`, `benchmark-isolation_trace_oracle`
- paper traceability examples:
  `2026-06-02-parqo-penalty-aware-robust-plan-selection`,
  `2026-06-03-hint-qpt-hints-for-robust-query-performance-tuning`,
  `2026-06-04-cardood-treats-route-estimator-drift-as-a-first-class-optimizer-risk`

### Runtime Admission And Scheduling

Production readiness depends on Candidate E's runtime model even while the
architecture selection remains Candidate A. One million logical sessions are
allowed only if idle sessions stay compact and only ready work or
response-blocked sessions consume hot-path resources. Admission is by vector
credits for requests, bytes, pinned buffers, GPU streams, WAL slots, residency
refresh, and response capacity. Owner rings may create natural micro-batches,
but deficit fairness must bound starvation and p99 damage.

References:

- mechanisms: `vector_credit_admission`, `effective_session_counting`,
  `owner_ring_bundling`, `same_shape_microbatching`, `deficit_fairness`
- architecture docs: `09-session-management-and-admission.md`,
  `11-high-throughput-query-runtime.md`
- proof gates: `benchmark-vector_credit_admission`,
  `benchmark-effective_session_counting`,
  `benchmark-owner_ring_bundling`,
  `benchmark-same_shape_microbatching`,
  `benchmark-deficit_fairness`
- paper traceability examples:
  `2026-06-05-backpressure-flow-control-makes-admission-local-selective-and-bounded`,
  `2026-06-05-ndp-re-architecting-datacenter-networks-and-stacks-for-low-latency`,
  `2026-06-04-gpu-learned-indexes-need-batch-shaped-residency-contracts`

### Storage And Placement

The first architecture keeps the P8 four-tier model: WAL/checkpoint/archive as
durable truth, CPU canonical MVCC state, CPU derived indexes/statistics, and
GPU resident snapshots as a performance tier. Stable handles should be used in
route and residency descriptors so future movement or compaction does not
change logical route identity. `multi_tier_placement` is the next-stage
Candidate B/D bridge, not the first baseline claim.

A future `direct_wal_gpu_ingest` path may be explored under Candidate B/D, but
only as fenced WAL replay/refresh acceleration. Chronicle-style mmap WAL
segments can reduce CPU append overhead, and GPUDirect Storage or equivalent
NVMe-to-GPU DMA can be compared against pinned host H2D copies for sealed WAL
ranges. The GPU side may build retained snapshots, visibility directories,
resident indexes, or refresh artifacts; it must not read the live mutable WAL
tail or decide commit visibility.

References:

- mechanisms: `stable_handle_indirection`, `multi_tier_placement`,
  `db_owned_cold_objects`
- architecture docs: `10-p8-gpu-optimized-storage-engine.md`
- benchmark status: 125% over-resident readiness remains blocked on
  `partitioned_resident_route_execution_primitive_required`
- proof gates: `benchmark-stable_handle_indirection`,
  `benchmark-multi_tier_placement`, `benchmark-db_owned_cold_objects`

## End-to-End Flows

### Write And Invalidation Flow

1. Network/runtime admission checks request, byte, WAL, mutation-owner,
   response, and memory credits.
2. Mutation owner validates the mutation.
3. WAL append and flush establish durable commit evidence.
4. Affected resident route roots and table generations are invalidated before
   post-mutation visibility is published.
5. CPU-visible MVCC/catalog state is applied.
6. New visibility boundary is published.
7. Residency refresh is admitted as explicit work; it may publish a new
   retained generation only after it can name the durable source boundary.
8. Readers that cannot prove compatibility with an old immutable generation
   fall back, wait, reject, or retry.

Invalidation gates:

- `benchmark-wal_before_visibility`
- `benchmark-immutable_route_roots`
- `benchmark-retained_gpu_snapshots`
- `benchmark-isolation_trace_oracle`

### Retained Read Flow

1. IO worker parses the query and submits bounded work.
2. Route preflight checks relation identity, schema generation, resident
   validity, supported predicate/output shape, requested freshness, queue
   pressure, response capacity, and cost.
3. If accepted, the read holds an immutable retained snapshot reference.
4. GPU execution owner either launches immediately or joins a compatible
   same-shape micro-batch under the latency cap.
5. Result-sized D2H transfer and response encoding proceed through response
   credits.
6. The response includes enough route telemetry for later isolation and
   fallback audit.
7. The snapshot reference is released; old generations become reclaimable only
   after active readers and MVCC/descriptor frontiers allow it.

Invalidation gates:

- `benchmark-retained_gpu_snapshots`
- `benchmark-htap_freshness_router`
- `benchmark-same_shape_microbatching`
- `benchmark-mvcc_gc_frontiers`
- `benchmark-bounded_descriptor_reclamation`

### Fallback Or Reject Flow

1. Route preflight fails because the shape is unsupported, freshness cannot be
   proven, residency is missing/stale, GPU or pinned-buffer credits are
   saturated, CPU is expected to be cheaper, response pressure is excessive, or
   the route lacks proof coverage.
2. The router chooses CPU fallback when semantics and latency budget permit.
3. The router chooses bounded wait, retry, or reject when CPU fallback would
   violate the caller contract or hide overload.
4. The fallback reason and selected route are recorded for cost calibration and
   isolation trace validation.

Invalidation gates:

- `benchmark-cpu_fallback_policy`
- `benchmark-cost_based_route_optimizer`
- `benchmark-vector_credit_admission`
- `benchmark-isolation_trace_oracle`

### Crash And Recovery Flow

1. Startup validates checkpoint/archive/control metadata.
2. WAL replay reconstructs CPU canonical MVCC/catalog state.
3. CPU indexes/statistics are rebuilt as needed.
4. GPU resident entries initialize as absent, stale, or rebuildable
   acceleration state; no retained route is eligible solely because device
   memory or a prior descriptor exists.
5. Immutable route roots are rebuilt only from durable facts.
6. Warmup may build retained generations; correctness traffic can proceed
   through CPU fallback while GPU warmup is incomplete.

Invalidation gates:

- `benchmark-semantic_crash_oracle`
- `benchmark-wal_before_visibility`
- `benchmark-immutable_route_roots`

## Fallback Architecture Rules

If `benchmark-retained_gpu_snapshots` or `benchmark-htap_freshness_router`
fails on freshness, benefit, or p99 stability, demote Candidate A and use
Candidate E as the baseline: finish vector credits, effective session
counting, response backpressure, owner-ring telemetry, and deficit fairness
before widening retained GPU scope.

If retained reads pass but partition locality or larger-than-HBM data becomes
dominant, promote Candidate B only after `benchmark-snapshot_frontier_vectors`,
`benchmark-stable_handle_indirection`, and `benchmark-multi_tier_placement`
show partition route preflight and movement cost do not erase retained-read
benefit.

If warm/cold placement becomes the central product constraint, promote
Candidate D only after `benchmark-db_owned_cold_objects`,
`benchmark-dependency_witnesses`, and `benchmark-semantic_crash_oracle` prove
that warm/cold metadata can recover and route eligibility remains fenced.

Candidate C remains a lab unless `benchmark-deterministic_hot_write_templates`
or `benchmark-gpu_oltp_conflict_ordering` proves containment under
`benchmark-wal_before_visibility`, `benchmark-isolation_trace_oracle`, and
retained-read p99 interference gates.

## Prioritized Proof Plan

| Priority | Gate packet | Decides | Architecture change if it fails |
| ---: | --- | --- | --- |
| 1 | WAL/root/crash spine: `benchmark-wal_before_visibility`, `benchmark-immutable_route_roots`, `benchmark-semantic_crash_oracle` | Durable publication safety for all candidates. | Block production-facing retained/tier routes until publication and recovery proof is fixed. |
| 2 | Runtime admission spine: `benchmark-vector_credit_admission`, `benchmark-effective_session_counting` | Whether Candidate E's runtime discipline is real. | Stop widening GPU scope; finish response credits and hidden-queue controls first. |
| 3 | Retained read gate: `benchmark-retained_gpu_snapshots`, `benchmark-htap_freshness_router`, `benchmark-cpu_fallback_policy` | Whether Candidate A remains the preferred baseline. | Demote to Candidate E and keep retained routes narrow or disabled. |
| 4 | Lifetime gate: `benchmark-mvcc_gc_frontiers`, `benchmark-bounded_descriptor_reclamation` | Whether retained generations have bounded memory/version cost. | Restrict retained reads to short-lived generations; defer long-reader and partition promotion. |
| 5 | Micro-batch/fairness gate: `benchmark-same_shape_microbatching`, `benchmark-deficit_fairness`, `benchmark-owner_ring_bundling` | Whether batching improves throughput without p99 harm. | Disable or sharply cap batching; keep immediate retained execution only where beneficial. |
| 6 | Partition/tier gate: `benchmark-snapshot_frontier_vectors`, `benchmark-stable_handle_indirection`, `benchmark-multi_tier_placement`, `benchmark-cost_based_route_optimizer` | Whether Candidate B can become next stage. | Keep B/D deferred and do not claim 125% over-resident retained execution. |
| 7 | Warm/cold recovery gate: `benchmark-db_owned_cold_objects`, `benchmark-dependency_witnesses`, `benchmark-log_structured_warm_tier` | Whether Candidate D can own warm/cold route metadata. | Treat warm/cold bodies as rebuildable experiments with CPU fallback. |
| 8 | Write lab gate: `benchmark-deterministic_hot_write_templates` or `benchmark-gpu_oltp_conflict_ordering` | Whether Candidate C is worth retaining as a lab. | Remove GPU write work from scheduling and route-cost assumptions. |
| 9 | Direct WAL GPU ingest gate: fenced mmap segment ranges, pinned H2D versus GPUDirect Storage or equivalent NVMe-to-GPU DMA, PCIe/NVMe/GPU credits, and crash injection | Whether Candidate B/D should add direct WAL-to-GPU replay or refresh acceleration. | Keep WAL ingest CPU/pinned-buffer based; forbid GPU consumption of live mutable WAL tails or unfenced batches. |

## Worker-Ready Next Packets

### Packet 1 - Publication Spine Harness

Target docs/code/tests:

- `docs/architecture/12-acid-isolation-and-gpu-memory.md`
- `docs/architecture/10-p8-gpu-optimized-storage-engine.md`
- `scripts/run_local_gpu_residency_preflight.sh`
- new or existing crash/recovery tests around WAL, route roots, residency
  manifests, and invalidation generations

Deliverable: a crash-injected matrix proving no SQL-visible route generation or
resident snapshot can survive without durable WAL and fenced root evidence.

### Packet 2 - Vector-Credit Runtime Simulator

Target docs/code/tests:

- `docs/architecture/09-session-management-and-admission.md`
- `docs/architecture/11-high-throughput-query-runtime.md`
- a bounded runtime simulator or focused benchmark for logical sessions,
  active flows, response-blocked sessions, GPU slots, WAL slots, pinned
  buffers, and standing queue depth

Deliverable: one-million logical-session evidence that hot resource use is
driven by active work and response pressure, not logical session count.

### Packet 3 - Retained Freshness Router Prototype

Target docs/code/tests:

- `docs/architecture/10-p8-gpu-optimized-storage-engine.md`
- `docs/architecture/04-execution-model-cpu-gpu.md`
- retained route decision and execution tests
- `scripts/run_p8_ch_benchmark_residency_probe.sh`

Deliverable: a route preflight that classifies retained, wait, CPU fallback,
reject, and retry decisions with freshness, residency, queue, response, and
benefit facts.

### Packet 4 - MVCC And Descriptor Lifetime Bounds

Target docs/code/tests:

- MVCC active-reader registry and resident generation lifetime tests
- descriptor/root protection tests for long readers and stalled sessions
- telemetry for active protected handles, retired bytes, maximum retirement
  age, and cleanup backlog

Deliverable: bounded version and descriptor memory under long retained readers
and hot mutation pressure.

### Packet 5 - Same-Shape Micro-Batching Curves

Target docs/code/tests:

- `docs/architecture/11-high-throughput-query-runtime.md`
- retained lookup and aggregate micro-batch benchmarks
- fairness/deficit tests for mixed interactive and bulk classes

Deliverable: p50/p95/p99, throughput, kernel launches, CUDA time, D2H bytes,
queue wait, batch size, and fairness bounds across immediate execution and
latency-capped batching.

### Packet 6 - Partitioned Retained Route Primitive

Target docs/code/tests:

- partition metadata, per-partition route acceptance, per-partition retained
  execution, and deterministic result reduction
- `docs/testing/benchmarks/README.md` blocker
  `partitioned_resident_route_execution_primitive_required`
- 125% over-resident readiness harness

Deliverable: the minimum Candidate B primitive needed before another full
125% tier run is defensible.

### Packet 7 - Direct WAL GPU Ingest Prototype

Target docs/code/tests:

- `docs/architecture/05-storage-and-recovery.md`
- `docs/architecture/10-p8-gpu-optimized-storage-engine.md`
- `docs/architecture/12-acid-isolation-and-gpu-memory.md`
- mmap-backed WAL segment append and sealed-range cursor tests
- pinned host H2D versus GPUDirect Storage or equivalent NVMe-to-GPU transfer
  benchmark when hardware support is available
- crash/recovery tests for partially written, partially transferred, and
  unfenced WAL batches

Deliverable: a fenced WAL ingest path that lets GPU workers consume only
durable LSN ranges to build retained snapshots, visibility directories,
resident indexes, or refresh artifacts. The packet must report transfer bytes,
PCIe/NVMe/GPU credit pressure, HBM staging cost, replay/refresh speedup, and
CPU fallback comparison.

## Non-Claims And Assumptions

This dossier does not claim:

- GPU memory is durable.
- GPU writes are authoritative.
- GPU ingest of a live mutable WAL mmap tail is safe.
- one million sessions are active at once.
- the 125% over-resident benchmark is solved.
- learned optimizer advice is safe to use for correctness-sensitive route
  decisions.
- full PostgreSQL isolation semantics beyond the current engine envelope.
- all workloads should choose GPU over CPU fallback.

Named assumptions inherited from the P4 candidate scorecard:

- `A1-retained-refresh-latency`: retained generations survive write churn long
  enough to beat CPU/cold-transfer routes.
- `A3-version-retention-bound`: retained readers do not cause unbounded MVCC
  or descriptor retention.
- `A4-vector-credit-sufficiency`: all hidden queue boundaries are captured by
  vector credits, including responses.
- `A5-microbatch-latency-cap`: launch amortization can be gained without p99
  damage or starvation.
- `A6-tier-route-benefit`: partition/tier movement costs do not erase GPU
  gains when over-resident routing is promoted.
- `A9-write-lab-containment`: any GPU write lab remains unable to bypass WAL,
  isolation, invalidation, and CPU authority.

## Final Selection

Proceed with Candidate A plus Candidate E runtime discipline as the next
implementation architecture. Treat Candidate E as the fallback if retained-read
gates fail, Candidate B as the next-stage partition/over-resident path if
retained reads succeed and partition pressure dominates, Candidate D as the
long-range warm/cold frontier, and Candidate C as a contained benchmark-only
write lab.
