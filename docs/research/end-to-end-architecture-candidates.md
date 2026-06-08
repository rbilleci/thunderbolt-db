# GPU DB End-to-End Architecture Candidates

This document contains the P3 candidate-family draft for the end-to-end
architecture search. It composes whole-system architectures from the completed
research dataset rather than adopting individual papers or mechanisms in
isolation.

Sources:

- design-space model:
  `docs/research/end-to-end-architecture-design-space.md`
- mechanism catalog:
  `docs/research/architecture-compatibility/mechanisms.json`
- compatibility graph:
  `docs/research/architecture-compatibility/compatibility-edges.json`
- generated compatibility view:
  `docs/research/architecture-compatibility.md`
- benchmark/proof backlog:
  `docs/research/architecture-compatibility/benchmark-backlog.md`
- current architecture docs:
  `docs/architecture/01-system-invariants.md`,
  `docs/architecture/04-execution-model-cpu-gpu.md`,
  `docs/architecture/09-session-management-and-admission.md`,
  `docs/architecture/10-p8-gpu-optimized-storage-engine.md`,
  `docs/architecture/11-high-throughput-query-runtime.md`, and
  `docs/architecture/12-acid-isolation-and-gpu-memory.md`

This draft does not claim a global optimum. It defines five valid architecture
families that pass the P1/P2 hard filters under named assumptions. P4 will score
these families, identify the Pareto frontier, and name the preferred
defensible architecture under explicit constraints.

## Shared Validity Spine

Every candidate in this document includes the mandatory correctness and
operability spine from the design-space model:

- `wal_before_visibility`: durable WAL evidence gates SQL visibility, route
  eligibility, resident snapshot publication, catalog/root publication, and
  tier metadata publication.
- `immutable_route_roots`: route, catalog, layout, residency, and visibility
  generations publish as immutable roots whose old bodies can be reclaimed.
- `vector_credit_admission`: work is admitted by typed credits for requests,
  bytes, pinned buffers, GPU streams, WAL slots, residency refresh, and
  responses.
- `cpu_fallback_policy`: unsupported, stale, saturated, nonresident,
  over-budget, low-benefit, and unproven routes return an explicit CPU, wait,
  reject, retry, or refresh outcome.
- `semantic_crash_oracle` and `isolation_trace_oracle`: recovery and
  SQL-visible route semantics are validation requirements for all families.

Common non-claims:

- GPU memory is never the durable source of truth.
- Encoded retained responses are not treated as queryable read snapshots.
- Learned optimizer advice is excluded from the current baseline candidates.
- Benchmark-only write acceleration does not replace CPU/WAL/MVCC authority.

## Candidate A - Conservative Retained-Read Evolution

### Thesis

Candidate A is the smallest end-to-end evolution from the current P8 direction:
keep CPU/WAL/MVCC authority, publish immutable retained GPU read generations
for hot tables, route through explicit freshness and fallback decisions, and
bound active work with vector credits. It is conservative because it optimizes
repeated reads without changing write authority, storage tiering, or owner
partitioning beyond what the current architecture already anticipates.

### Mechanism Set

- Mandatory spine: `wal_before_visibility`, `immutable_route_roots`,
  `vector_credit_admission`, `cpu_fallback_policy`,
  `semantic_crash_oracle`, `isolation_trace_oracle`
- Retained-read core: `retained_gpu_snapshots`,
  `snapshot_frontier_vectors`, `mvcc_gc_frontiers`,
  `bounded_descriptor_reclamation`, `stable_handle_indirection`
- Runtime/read shaping: `same_shape_microbatching`,
  `effective_session_counting`, `owner_ring_bundling`, `deficit_fairness`
- Routing and planning: `htap_freshness_router`,
  `cost_based_route_optimizer`
- Explicitly excluded from the baseline: `gpu_oltp_conflict_ordering`,
  `deterministic_hot_write_templates`, `log_structured_warm_tier`,
  `db_owned_cold_objects`, `learned_optimizer_advisor`

### Owner Topology

The first implementation can keep mutation, catalog, residency, and GPU
execution co-located while preserving the target owner model from
`11-high-throughput-query-runtime.md`. The logical topology is:

- mutation owner: WAL admission, transaction visibility, MVCC apply, and
  invalidation sequencing
- catalog/residency owner: immutable route roots, resident snapshot
  descriptors, refresh, eviction, and root publication
- GPU execution owners: CUDA streams, pinned buffers, retained handles, and
  same-shape retained-read batches
- network/response workers: protocol parsing, response rings, and response
  backpressure

Splitting these owners is an optimization, not a semantic dependency. The
critical boundary is that mutable state has one authority and readers consume
immutable generations.

### Data Movement Model

Writes enter CPU/WAL/MVCC state first. A refresh builds GPU column-group or
resident index snapshots from CPU-visible state after the durable boundary.
Read results return as result-sized D2H transfers or CPU fallback responses.
No query waits for a speculative GPU refresh to become visible unless the
freshness router has admitted a bounded wait.

### Visibility Model

Each retained snapshot names relation identity, schema generation, source WAL
or transaction boundary, visibility boundary, resident layout identity,
supported route families, and invalidation generation. New readers are admitted
only if the route root is current enough for the requested freshness contract.
Old generations retire through `mvcc_gc_frontiers` and
`bounded_descriptor_reclamation` after active readers release them.

### Route And Fallback Model

`htap_freshness_router` chooses retained, wait-for-refresh, CPU fallback,
reject, or retry. `cost_based_route_optimizer` compares CPU, retained GPU,
cold-transfer GPU, and refresh costs using measured telemetry. Fallback reasons
must be classed as unsupported predicate, stale generation, missing residency,
GPU saturation, pinned-buffer pressure, low expected benefit, response
backpressure, or unproven route.

### Scheduling And Admission Model

`vector_credit_admission` is the entry point for read, write, refresh, GPU, and
response work. Same-shape retained reads may micro-batch only under latency and
fairness caps. `owner_ring_bundling` forms low-allocation bundles; `deficit_fairness`
prevents long batches from starving interactive or freshness-sensitive work.
`effective_session_counting` keeps idle logical sessions out of hot resource
counts.

### Storage And Tier Model

Candidate A uses WAL/checkpoint/archive and CPU MVCC state as truth, plus GPU
resident snapshots as a performance tier. `stable_handle_indirection` is used
inside route and residency descriptors so compaction or movement does not
change logical route identity. Multi-tier placement is a future extension, not
part of the first claim.

### Recovery Model

Recovery replays WAL/checkpoint/archive into CPU-visible state, marks GPU
resident generations empty or stale, rebuilds route roots from durable metadata,
and only then allows residency warmup. The `semantic_crash_oracle` must prove
that no speculative route, resident snapshot, or invalidation state survives as
externally eligible without durable evidence.

### Validation Model

First gates:

- `benchmark-wal_before_visibility`
- `benchmark-immutable_route_roots`
- `benchmark-retained_gpu_snapshots`
- `benchmark-snapshot_frontier_vectors`
- `benchmark-mvcc_gc_frontiers`
- `benchmark-bounded_descriptor_reclamation`
- `benchmark-cpu_fallback_policy`
- `benchmark-isolation_trace_oracle`
- `benchmark-vector_credit_admission`
- `benchmark-same_shape_microbatching`
- `benchmark-deficit_fairness`

### Expected Strengths

- Best fit with the current P8 and runtime architecture direction.
- Lowest implementation risk among GPU-accelerated candidates.
- Strong hot lookup and retained aggregate path when residency remains valid.
- Clear failure modes because every accelerated route has a fallback reason.

### Risks

- Depends on `A1-retained-refresh-latency`: write churn may invalidate resident
  generations faster than reads can benefit.
- Depends on `A3-version-retention-bound`: long retained readers can extend
  MVCC and descriptor lifetime.
- Depends on `A5-microbatch-latency-cap`: launch amortization may harm p99 if
  batching waits too long.

### Non-Claims

- Does not solve data larger than HBM.
- Does not accelerate writes on GPU.
- Does not claim full PostgreSQL transaction isolation semantics beyond the
  current engine envelope.

## Candidate B - Partition-Owner Retained Snapshots

### Thesis

Candidate B extends Candidate A by making partition ownership explicit. It
keeps the same correctness spine but publishes retained GPU snapshots and route
roots per table partition or owner domain. The goal is to scale retained reads,
freshness proofs, and over-resident placement without forcing every route
preflight through one global visibility frontier.

### Mechanism Set

- Candidate A mechanisms
- Partition/tiering defaults: `multi_tier_placement`
- Stronger use of `snapshot_frontier_vectors` for cross-owner proof
- Optional: `resource_dag_scheduling`, `db_owned_cold_objects`
- Explicitly excluded from the baseline: `gpu_oltp_conflict_ordering`,
  `deterministic_hot_write_templates`, `learned_optimizer_advisor`

### Owner Topology

The system has a mutation authority that preserves WAL order plus partition
owners that own disjoint resident partition state. A catalog/residency
publisher emits partition-level immutable route roots. GPU execution owners are
bound to device, stream, and partition-local handles. Cross-partition reads
either use a frontier-vector proof or fall back to CPU/coordinator execution
when the preflight cost or freshness uncertainty is too high.

### Data Movement Model

Partitions refresh independently. Hot partitions can remain in HBM while warm
or cold partitions stay in DRAM/NVMe-backed tiers. Stable handles preserve
logical row, partition, and route identity through movement, compaction, and
eviction. Cross-partition aggregation may execute as partition-local GPU work
followed by CPU or coordinator reduction when the result shape is bounded.

### Visibility Model

Each route root carries partition identity, local publication generation,
source WAL frontier, catalog generation, and invalidation generation.
`snapshot_frontier_vectors` prove when a query spanning multiple owners can
read a coherent snapshot. Scalar generation checks remain the hot path for
single-partition reads.

### Route And Fallback Model

The freshness router adds partition state to its decision: retained local GPU,
multi-partition retained GPU with vector proof, wait-for-partition-refresh,
CPU/coordinator fallback, reject, or retry. The cost optimizer must account for
partition residency, movement bytes, reduction bytes, and cross-owner preflight
cost.

### Scheduling And Admission Model

Vector credits are scoped by global boundary and partition owner. Owner rings
drain partition-local retained reads and refresh tasks. `deficit_fairness`
tracks tenant, freshness class, and partition deficits so hot partitions do not
starve cold or interactive work. `resource_dag_scheduling` is useful but
optional until benchmarks show simple partition rings cannot pack refresh,
transfer, kernel, and response work well enough.

### Storage And Tier Model

Candidate B makes `multi_tier_placement` a default. HBM holds hot resident
partition snapshots; DRAM/NVMe tiers hold warm segments or transfer-ready
layouts; object storage remains optional unless `db_owned_cold_objects` is
adopted. The tiering claim is only valid with `stable_handle_indirection`.

### Recovery Model

Recovery replays durable state, reconstructs partition ownership metadata, and
marks every resident partition empty or stale until route roots are rebuilt
from durable facts. Partition-level roots cannot become eligible until their
dependency witnesses and WAL frontiers are fenced.

### Validation Model

First gates:

- Candidate A validation gates
- `benchmark-multi_tier_placement`
- `benchmark-stable_handle_indirection`
- `benchmark-snapshot_frontier_vectors`
- `benchmark-htap_freshness_router`
- `benchmark-resource_dag_scheduling` if DAG packing becomes part of the claim

### Expected Strengths

- Better path toward over-resident and partition-local workloads.
- Reduces global retained-read invalidation when writes touch only one
  partition.
- Gives the route optimizer richer placement and freshness choices.

### Risks

- Depends on `A2-frontier-preflight-cost`: cross-owner vectors may dominate
  short read latency.
- Depends on `A6-tier-route-benefit`: movement costs may erase partitioned GPU
  gains.
- Partition movement can make route descriptor reclamation and stable-handle
  proof more complex than Candidate A.

### Non-Claims

- Does not make every query partition-local.
- Does not require object storage ownership.
- Does not claim cross-partition GPU execution beats CPU fallback before
  measurement.

## Candidate C - Aggressive GPU-Resident Hot Path

### Thesis

Candidate C keeps the mandatory CPU/WAL/MVCC authority path but adds a bounded
GPU write-path lab for known-shape hot writes. It is a candidate family because
it defines how an aggressive hot lane can be evaluated without weakening
durability, visibility, or isolation. It should not become the preferred
baseline unless P4 and later proof gates show the write lab is contained and
valuable.

### Mechanism Set

- Candidate A mechanisms
- One benchmark-only write lab:
  `deterministic_hot_write_templates` by default, or
  `gpu_oltp_conflict_ordering` as the alternative experiment
- Optional: `resource_dag_scheduling`, selected `log_structured_warm_tier`
  experiment after `dependency_witnesses` pass
- Excluded from correctness authority: GPU-resident write state,
  `learned_optimizer_advisor`

### Owner Topology

The mutation owner remains the durable authority. The hot write lab is an
admission and preprocessing lane for known templates, hot keys, or deterministic
conflict classes. It may prepare write batches, classify conflicts, or stage
candidate GPU work, but the mutation owner still owns WAL reservation,
commit-order publication, MVCC apply, invalidation, and final visibility.

### Data Movement Model

Known-shape writes can be classified and optionally preprocessed for GPU
execution, but outputs become SQL-visible only after WAL durability and CPU/MVCC
apply. Retained read snapshots are invalidated or refreshed from the durable
publication path, not from device-local side effects.

### Visibility Model

The write lab has no external visibility boundary. Its outputs are speculative
until the mutation owner binds them to a durable commit generation. Reads use
the same retained snapshot and fallback rules as Candidate A.

### Route And Fallback Model

Write admission can choose ordinary CPU/WAL/MVCC, deterministic hot template
lane, GPU conflict-ordering lab, reject, or retry. A lab route falls back to the
ordinary mutation owner when the write shape, conflict class, WAL budget,
isolation proof, or tail-latency cap is not satisfied.

### Scheduling And Admission Model

Vector credits include write-template slots, GPU preprocessing slots, WAL
slots, mutation-owner drain budget, and invalidation/refresh budget. Owner-ring
bundling can coalesce hot writes, but deficit fairness must cap the damage to
ordinary reads and writes.

### Storage And Tier Model

Candidate C keeps Candidate A's simple tier model unless a
`log_structured_warm_tier` experiment is explicitly attached. Any warm-tier lab
requires `dependency_witnesses` and the crash oracle before route eligibility
or recovery claims are made.

### Recovery Model

Crash recovery ignores speculative GPU write state. WAL/checkpoint/archive
replay reconstructs CPU/MVCC truth, then invalidates or rebuilds resident GPU
generations. The crash oracle must include cases where GPU preprocessing
completed but WAL commit, dependency witnesses, or visibility publication did
not.

### Validation Model

First gates:

- Candidate A validation gates
- `benchmark-deterministic_hot_write_templates` or
  `benchmark-gpu_oltp_conflict_ordering`
- `benchmark-wal_before_visibility`
- `benchmark-isolation_trace_oracle`
- `benchmark-owner_ring_bundling`
- `benchmark-deficit_fairness`
- `benchmark-log_structured_warm_tier` only if the warm-tier lab is included

### Expected Strengths

- Potentially improves known-shape hot-write tail behavior.
- Creates a safe experiment lane for GPU write research without making GPU
  write state authoritative.
- Can expose whether deterministic preprocessing is more valuable than
  optimistic GPU conflict ordering for the engine's workload.

### Risks

- Depends on `A9-write-lab-containment`: the lab must not bypass WAL,
  isolation, invalidation, or CPU authority.
- High implementation and validation cost relative to the current read-heavy
  product thesis.
- Can make scheduling harder by adding GPU write work that competes with
  retained reads.

### Non-Claims

- Does not promote GPU OLTP conflict ordering into the baseline.
- Does not remove CPU/MVCC apply from the commit path.
- Does not claim write throughput gains until local benchmarks prove them.

## Candidate D - Multi-Tier Freshness Router

### Thesis

Candidate D is the over-resident architecture: route reads and refresh work
across HBM, DRAM, NVMe, and optional DB-owned cold objects using stable handles,
immutable roots, freshness contracts, and measured movement cost. It is the
strongest long-range architecture for data larger than GPU memory, but it has
more recovery and operability surface than Candidates A and B.

### Mechanism Set

- Candidate B mechanisms
- Tiering defaults: `multi_tier_placement`, `db_owned_cold_objects`
- Publication support: `dependency_witnesses`
- Optional experiment: `log_structured_warm_tier`
- Excluded from the first baseline: `gpu_oltp_conflict_ordering`,
  `deterministic_hot_write_templates`, `learned_optimizer_advisor`

### Owner Topology

The topology adds tier/placement ownership to the partition-owner model.
Residency owners publish HBM/DRAM/NVMe/object placement descriptors as immutable
roots. A placement owner tracks reuse, movement cost, cold-object lifecycle,
backup/PITR alignment, and readiness. GPU execution owners consume only fenced,
route-eligible placement descriptors.

### Data Movement Model

Reads can execute from retained HBM snapshots, warm host/NVMe layouts with
explicit transfer, cold object scans with DB-owned descriptors, or CPU fallback.
Movement is a first-class route cost. Warm and cold bodies can persist
asynchronously only when `dependency_witnesses` explain delayed eligibility and
`immutable_route_roots` keep readers away from unfenced bodies.

### Visibility Model

Visibility combines WAL frontier, catalog generation, partition generation,
placement descriptor generation, and freshness requirement. Cold and warm
metadata cannot become route-eligible until durable object/body checksums,
dependency witnesses, and immutable roots are fenced.

### Route And Fallback Model

The freshness router chooses retained HBM, warm transfer, cold scan,
wait-for-refresh, CPU fallback, reject, or retry. The cost optimizer compares
transfer bytes, decompression cost where applicable, kernel work, response
bytes, refresh cost, route staleness, and queue pressure.

### Scheduling And Admission Model

Vector credits extend to NVMe/object transfer slots, host memory budgets, pinned
buffers, dependency-witness flush work, and cold-object response pressure.
`resource_dag_scheduling` is more likely to be needed here because refresh,
H2D, kernel, D2H, object, and response work have real dependency structure.

### Storage And Tier Model

Candidate D uses stable handles for all movable fragments. HBM is the hot tier,
DRAM/NVMe is the warm route tier, and DB-owned object storage is the cold tier
only if backup, PITR, invalidation, and recovery are owned by the database
metadata model. `log_structured_warm_tier` remains a benchmark-only experiment
until its recovery and write-amortization gates pass.

### Recovery Model

Recovery starts from WAL/checkpoint/archive and CPU truth. Warm and cold route
descriptors are replayed only if their immutable roots, checksums, and
dependency witnesses are durable and fenced. Otherwise they are treated as
rebuildable acceleration state. The crash oracle must cover object descriptors,
warm-tier bodies, partial writes, retired roots, and stale placement facts.

### Validation Model

First gates:

- Candidate B validation gates
- `benchmark-db_owned_cold_objects`
- `benchmark-dependency_witnesses`
- `benchmark-semantic_crash_oracle`
- `benchmark-cost_based_route_optimizer`
- `benchmark-resource_dag_scheduling`
- `benchmark-log_structured_warm_tier` only for the warm-tier experiment

### Expected Strengths

- Best fit for data sets larger than HBM.
- Makes movement cost, freshness, and fallback observable route facts.
- Aligns with the P8 storage direction that treats GPU residency as one tier,
  not the whole storage engine.

### Risks

- Depends on `A6-tier-route-benefit`: transfer and route overhead can erase
  GPU gains.
- Depends on `A7-cold-object-recovery`: DB-owned cold objects add recovery,
  backup, and invalidation obligations.
- Depends on `A10-dependency-witness-cost`: witness checks may slow
  publication or route preflight.

### Non-Claims

- Does not make cold object storage authoritative without WAL/checkpoint/archive
  recovery proof.
- Does not claim all over-resident scans should use GPU.
- Does not require learned placement or learned route advice.

## Candidate E - Runtime-First 1M-Session Architecture

### Thesis

Candidate E prioritizes session scale, admission stability, and response
backpressure before broad retained-read or tiering scope. It narrows the GPU
execution claim if needed so the runtime can prove that 1M logical sessions can
exist while only active flows consume hot-path resources.

### Mechanism Set

- Mandatory spine
- Runtime core: `effective_session_counting`, `owner_ring_bundling`,
  `deficit_fairness`, `same_shape_microbatching`
- Optional staged additions: `retained_gpu_snapshots`,
  `snapshot_frontier_vectors`, `mvcc_gc_frontiers`,
  `bounded_descriptor_reclamation`, `cost_based_route_optimizer`,
  `multi_tier_placement`
- Excluded from the first runtime claim: `gpu_oltp_conflict_ordering`,
  `deterministic_hot_write_templates`, `log_structured_warm_tier`,
  `learned_optimizer_advisor`

### Owner Topology

Network IO workers own socket readiness and protocol parsing. Mutation and
catalog state stay behind owners. Read snapshot and GPU execution workers can
be narrower than in Candidate A, but all ingress, execution, residency, and
response paths use bounded rings with explicit credits. Logical sessions are
compact state records, not one hot thread, queue slot, or large allocation each.

### Data Movement Model

Candidate E moves data only for admitted active flows. Retained GPU snapshots
may be introduced for a small set of hot shapes, but the architecture is valid
even if many reads choose CPU fallback while runtime admission and response
backpressure are proven.

### Visibility Model

The visibility spine is unchanged. If retained snapshots are staged in, they
must use Candidate A's retained visibility model. Otherwise CPU/MVCC visibility
is the reference route while the runtime proof focuses on bounded admission.

### Route And Fallback Model

Routes are selected with simpler deterministic rules at first: CPU execution,
retained read if already valid and beneficial, reject, wait, or retry. A full
cost optimizer can be staged after vector-credit and active-flow telemetry are
stable.

### Scheduling And Admission Model

This is Candidate E's core. Admission separates logical-session count from
active-flow count, response-blocked sessions, WAL slots, GPU slots, pinned
buffers, request bytes, and response bytes. Owner rings batch naturally but
must expose selected/skipped reasons and queue wait. Deficit fairness protects
interactive and tenant classes when batching improves throughput.

### Storage And Tier Model

The baseline storage model can remain CPU/WAL/MVCC plus a narrow retained GPU
cache. Tiering and partition ownership are deferred until the runtime can prove
standing queues, response pressure, and active-flow accounting.

### Recovery Model

Recovery is simple relative to the other candidates: rebuild CPU truth from
WAL/checkpoint/archive, mark acceleration state stale, and rehydrate compact
session/runtime state only where protocol semantics require it. The crash
oracle still covers route roots, response backpressure, and visibility
publication.

### Validation Model

First gates:

- `benchmark-vector_credit_admission`
- `benchmark-effective_session_counting`
- `benchmark-owner_ring_bundling`
- `benchmark-deficit_fairness`
- `benchmark-same_shape_microbatching`
- `benchmark-cpu_fallback_policy`
- `benchmark-isolation_trace_oracle`
- retained-read gates only if retained snapshots are included in the first
  implementation slice

### Expected Strengths

- Best direct fit for the 1M logical-session target.
- Reduces the risk of hiding unbounded queues behind GPU speedups.
- Produces useful runtime infrastructure for every other candidate.

### Risks

- Depends on `A4-vector-credit-sufficiency`: credits must cover every hidden
  queue boundary, including responses.
- Depends on `A8-owner-ring-tail-control`: owner bundles must not starve
  latency-sensitive work.
- May underdeliver on GPU read performance if retained snapshots are delayed
  too long.

### Non-Claims

- Does not claim the most aggressive retained-read latency.
- Does not solve over-resident placement.
- Does not make one million sessions active at once; it targets one million
  logical sessions with bounded active-flow cost.

## Candidate Interaction Summary

| Candidate | Primary optimization | Best current fit | Main assumptions | First decisive failure |
| --- | --- | --- | --- | --- |
| A - Conservative retained-read evolution | Hot retained reads with low implementation risk | Current P8 and runtime docs | `A1`, `A3`, `A5` | Retained routes cannot bound freshness, version lifetime, or p99 batching. |
| B - Partition-owner retained snapshots | Partition-local retained reads and over-resident path | P8 plus future owner split | `A2`, `A6`, `A8` | Cross-owner preflight or movement breaks latency and stable-handle proof. |
| C - Aggressive GPU-resident hot path | Known-shape write acceleration lab | Future benchmark lane only | `A9`, `A8` | GPU write lab leaks authority or harms retained-read/runtime p99. |
| D - Multi-tier freshness router | HBM/DRAM/NVMe/object routing | Long-range over-resident architecture | `A6`, `A7`, `A10` | Warm/cold metadata cannot recover or route benefit loses to CPU fallback. |
| E - Runtime-first 1M-session architecture | Active-flow accounting and response backpressure | Production runtime foundation | `A4`, `A8`, `A5` | Standing queues or response pressure remain hidden under logical-session counts. |

## P4 Inputs

P4 should score these families without adding new broad mechanisms. The likely
comparison questions are:

- Does Candidate A already dominate near-term implementation because it
  advances the GPU value proposition while preserving the current correctness
  spine?
- Does Candidate E need to be treated as a prerequisite layer for A rather than
  a competing architecture?
- Is Candidate B the natural fallback or next-stage architecture once
  partition and tier measurements exist?
- Should Candidate D be ranked as a long-range architecture rather than the
  preferred near-term path because recovery and tiering proof gates are large?
- Should Candidate C remain explicitly benchmark-only unless write-lab
  containment and scheduling gates pass?

P4 must name the preferred and fallback architecture under explicit workload
weights. It should not collapse all axes into a false global optimum.
