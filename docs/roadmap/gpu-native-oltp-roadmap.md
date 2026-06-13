# GPU-Native OLTP Roadmap

This roadmap narrows the GPU-native target to OLTP workloads: high-concurrency
entity reads, tenant/security-filtered page reads, bounded joins, and computed
detail routes for banking and e-commerce style systems.

The goal is not to make every arbitrary SQL query sub-millisecond immediately.
The goal is to build a GPU-resident prepared-route engine whose hot OLTP routes
execute from immutable snapshot generations with clear latency and throughput
evidence.

## Target State

The v1 GPU-native OLTP target is:

- GPU-resident hot data, lookup structures, and route-ready column layouts
- prepared OLTP route ids with typed parameters
- immutable GPU-resident snapshot generations
- serialized COPY/write/DDL generation publication
- concurrent read-only GPU execution over one published generation
- separate latency and throughput admission policies per prepared route
- benchmark evidence for p50/p95 latency, queue wait, GPU time, and throughput

## Route Classes

1. **Entity route**
   - Fetch one logical entity by key or unique business key.
   - May project many columns.
   - May include tenant/security predicates.

2. **Tenant-filtered page route**
   - Fetch 20-50 rows from a bounded access path.
   - Tenant/account/security filters are part of the route, not post-filter
     decoration.

3. **Bounded join route**
   - Join one or two tables where fanout is bounded by key, tenant, date range,
     or page limit.
   - Produce stable projections suitable for response-buffer reuse.

4. **Computed detail route**
   - Fetch one entity plus computed values such as balance, count, status, or
     derived flags.
   - Prefer resident auxiliary summaries or bounded correlated lookup plans over
     arbitrary per-row subquery execution.

5. **Throughput batch route**
   - Same-shape retained reads grouped for GPU occupancy and qps.
   - Optimizes throughput rather than p50 floor.

## Executable Milestones

### M1: Prepared Entity Route Skeleton (closed)

Implement an internal prepared retained route descriptor for the current
`order_line WHERE ol_o_id = $1` benchmark shape.

Deliverables:

- route id and typed int4 parameter representation
- route metadata for table, projection, filter column, expected cardinality,
  and tenant/security predicate placeholder
- benchmark path that can execute the route without reparsing SQL on every
  request
- telemetry separating SQL-text path from prepared-route path

Evidence:

- c8 and c64 cache-off p50/p95 for prepared route vs SQL-text route
- queue wait, engine execute, retained wall, and materialization facts
- closed by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-12-prepared-retained-route-cache-off-v1.md`

### M2: Immutable Retained Snapshot Handle (closed)

Extract read-only resident state into an explicit snapshot handle that retained
routes can reference without borrowing the whole mutable engine owner.

Deliverables:

- snapshot generation id
- resident table/layout identity
- immutable device handle references for supported columns
- invalidation/publication metadata
- tests showing old snapshots are not mutated in place

Evidence:

- retained route execution reports snapshot generation
- writes/COPY invalidate or publish generations deterministically
- closed by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-12-retained-snapshot-handle-cache-off-v1.md`

### M3: Async Read Job Lifecycle

Split retained read execution into submit and completion phases.

Deliverables:

- read job descriptor for prepared entity route
- submit API that launches or schedules GPU work
- completion API that collects events/copies/materialized output
- owner loop no longer blocks on full request lifecycle for read-only jobs

Evidence:

- multiple in-flight read jobs visible in telemetry
- c64 queue wait reduction without CPU fallback
- descriptor slice closed by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-12-retained-read-job-cache-off-v1.md`
- synchronous submit/complete slice closed by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-12-retained-read-submit-complete-cache-off-v1.md`
- preplanned execution slice closed by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-12-preplanned-read-job-cache-off-v1.md`
- execution-layer nonblocking all-int4 submit primitive closed by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-12-execution-async-retained-int4-submit-v1.md`
- engine-level pending all-int4 retained read-job submission closed by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-12-engine-pending-retained-int4-submit-v1.md`
- adjacent submit/complete A/B closed by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-12-async-retained-int4-c64-ab-v1.md`
- owner-loop pending completion queue implementation and rejection closed by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-12-owner-loop-pending-completion-impl-v1.md`
- select phase fact hot-path removal closed by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-12-select-phase-facts-off-c64-v1.md`
- no-phase-facts full c1-c64 fast-path guard closed by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-12-select-facts-none-full-guard-v1.md`
- direct client-thread retained read-view runtime probe rejected by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-13-read-runtime-view-reject-v1.md`
- batched retained read runtime promoted for all-INT4 point reads by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-13-batched-read-runtime-default-v1.md`

Current state:

- all-int4 retained read jobs can submit CUDA work without immediately
  synchronizing in the execution layer
- mixed/text retained read jobs still use the synchronous path
- endpoint pending completion can overlap all-int4 prepared microbatch work when
  `GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_PENDING_COMPLETION_CAP` is enabled,
  but the c64 A/B regressed the main literal rows, so the cap defaults to `0`
- prepared retained microbatches remain opt-in; do not keep tuning pending caps
  as the default path unless a new design removes a whole owner-loop boundary
- per-retained-SELECT phase JSON emission is no longer on the endpoint default
  hot path; benchmark runs request `phase_only` explicitly when they need phase
  aggregates
- client-thread read-view execution can bypass the owner but regresses badly
  because it converts filled all-INT4 retained batches into singleton GPU
  launches
- batched read-runtime execution preserves route-shape batch fill off-owner and
  is the default cache-off path for all-INT4 retained point reads
- mixed int4/text retained point reads now use the same batched runtime for one
  compact text projection per route, closing the owner route-lane fallback for
  the current heterogeneous benchmark shape; closed by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-13-mixed-text-read-runtime-default-v1.md`
- remaining runtime wait is in the single runtime worker and the synchronous
  compact text projection/copy path; add per-route runtime queue telemetry
  before tuning stream-pool width
- per-route runtime queue telemetry was added and naive route-key worker
  sharding was rejected as a default by
  `docs/testing/reports/series/p8-concurrency-steady-state/runs/2026-06-13-runtime-queue-telemetry-workers-reject-v1.md`;
  keep one runtime worker until stream ownership or async text completion is
  designed explicitly

### M4: Small GPU Stream Pool

Run read-only retained jobs concurrently over one snapshot generation.

Deliverables:

- fixed-size stream pool for read-only retained routes
- stream ownership and event lifecycle rules
- per-stream telemetry
- mutation barrier that waits for or fences readers before publishing a new
  generation

Evidence:

- c64 and c128 curves, if hardware allows
- p50/p95 queue wait and throughput versus single-stream owner baseline

### M5: Tenant-Filtered Page Route

Add a bounded page route returning roughly 20-50 rows with tenant/security
predicate included in the device access path.

Deliverables:

- synthetic OLTP table shape with tenant id, entity id, sort/filter column, and
  projected payload
- route metadata for tenant predicate and page limit
- GPU-resident lookup/range/filter structure
- stable response shape telemetry

Evidence:

- c8/c64 p50/p95 for 20-row and 50-row pages
- queue wait and D2H bytes per page
- correctness with tenant isolation cases

### M6: Bounded Two-Table Join Route

Add a route for one bounded join where one side is key/page bounded.

Deliverables:

- route metadata for join keys and fanout bound
- resident join-side layout and lookup structure
- projection plan spanning two tables
- fallback reason when fanout is unbounded

Evidence:

- p50/p95 and throughput for bounded join pages
- correctness against CPU reference for tenant/security-filtered joins

### M7: Computed Detail Route

Add a single-entity detail route with derived columns.

Deliverables:

- resident auxiliary summary or bounded correlated lookup structure
- computed-column route plan
- invalidation/publication rules for summaries

Evidence:

- p50/p95 for detail route with computed columns
- correctness across updates or refresh generation changes

## Benchmark Policy

Every milestone should report at least:

- p50/p95/p99 latency
- throughput qps
- scheduler queue wait
- engine execution time
- retained GPU wall/CUDA event time
- H2D/D2H bytes
- materialization/write time
- fallback rate and fallback reason
- snapshot generation or route id where applicable

Use separate curves for:

- latency-oriented prepared routes
- throughput-oriented batch routes
- mixed/fairness workloads

Do not treat a CPU response cache or CPU index win as satisfying the
GPU-native route target unless the report explicitly labels it as a fallback or
comparison baseline.

## Near-Term Next Slice

Continue M3/M4 with async/concurrent execution. Do not spend more default-path
work on increasingly complex single owner-queue heuristics unless they are
short probes that protect an existing win. The next implementation should aim
for a 5x-class boundary reduction, not a 1.2x row improvement:

1. Move response materialization/write work out of the owner critical path, or
   advance to a small read-only stream pool with enough independent work to
   hide completion.
   Preserve filled retained job batches; direct singleton read-view launches
   are a rejected default path.
   The batched read runtime now satisfies this for all-INT4 point reads.
2. Keep the owner-loop pending completion queue as an opt-in diagnostic with
   cap `0` by default; cap `2` proved real overlap but regressed c64 p50.
3. Preserve generation mismatch rejection and mutation publication barriers.
4. Keep diagnostic fact logging off the default endpoint hot path; use
   `phase_only` only for evidence runs that need per-request aggregates.
5. Report c8/c64 queue wait and p50 impact before moving beyond M4.
6. Treat route-lane depth, payload, and diversity heuristics as opt-in probes
   unless they collapse queue wait by multiple times without hurting homogeneous
   route batches.
7. Use fixed route-lane scan as the default baseline:
   `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=fixed` and
   scan limit `32`.
