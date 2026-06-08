# GPU DB Research Architecture Compatibility

This document is generated from mechanism cards and typed compatibility
edges. It turns the literature journal into a design-composition map:
papers provide evidence, mechanisms provide contracts, and edges show
which mechanisms can be combined in an end-to-end architecture.

Regenerate with:

```sh
python3 scripts/generate_research_architecture_compatibility.py
```

## Source Layers

- Layer 1: `docs/research/architecture-compatibility/mechanisms.json`
- Layer 2: `docs/research/architecture-compatibility/compatibility-edges.json`
- Layer 3: this generated compatibility view
- Paper traceability: `docs/research/architecture-compatibility/paper-mechanism-links.json`
- Paper coverage report: `docs/research/architecture-compatibility/paper-mechanism-coverage.md`

## Summary

- mechanisms: 25
- edges: 51
- alternative_to: 1
- compatible: 7
- requires: 16
- strengthens: 22
- tension: 5

Decision status counts:
- adopt_now: 6 (make this a baseline architecture invariant)
- benchmark_only: 4 (keep as an experiment until measurement decides adoption)
- defer: 1 (postpone until prerequisite mechanisms or product pressure exist)
- prototype: 14 (build the first implementation behind an explicit proof gate)
- reject: 0 (do not include in the architecture)
- unknown: 0 (not enough reviewed evidence to decide yet)

## Recommended End-to-End Spine

- `wal_before_visibility` (durability): Durable commit evidence must exist before SQL-visible state or route eligibility advances.
- `immutable_route_roots` (metadata publication): Publish route, catalog, layout, residency, and visibility metadata as immutable bodies plus compact root generations.
- `dependency_witnesses` (metadata publication): Represent delayed internal publication as explicit dependency chains that must fence before external eligibility.
- `semantic_crash_oracle` (validation): Test recovery by semantic states and externally visible outcomes rather than by exhaustively enumerating all crash points.
- `snapshot_frontier_vectors` (MVCC): Use compact scalar generations on hot paths and vector/frontier state only where cross-owner snapshot proof is required.
- `mvcc_gc_frontiers` (MVCC): Reclaim old versions by exact active snapshot generations and bounded multiversion collection rather than periodic blind cleanup.
- `bounded_descriptor_reclamation` (memory reclamation): Retire route, catalog, and plan descriptors through bounded hazard/era/epoch-style protection.
- `stable_handle_indirection` (storage layout): Separate logical row/route identity from movable physical placement through handles, forwarding records, or page tables.
- `retained_gpu_snapshots` (execution): Keep hot read generations resident on GPU and route matching reads directly when freshness and residency prove valid.
- `vector_credit_admission` (runtime admission): Admit work using per-boundary credits for requests, bytes, pinned buffers, GPU streams, WAL slots, and response capacity.
- `effective_session_counting` (runtime admission): Count only sessions with ready work or blocked responses as active flows; idle logical sessions consume minimal hot-path capacity.
- `owner_ring_bundling` (runtime scheduling): Drain bounded bundles from owner rings and record why selected or skipped work moved forward.
- `resource_dag_scheduling` (runtime scheduling): Represent query, refresh, write, transfer, kernel, and response work as small dependency DAGs over scarce resources.
- `deficit_fairness` (runtime scheduling): Allow short-term throughput-favorable scheduling while tracking deficits so interactive or tenant classes do not starve.
- `same_shape_microbatching` (execution): Batch repeated prepared route shapes by snapshot generation, partition, and result shape before GPU execution.
- `htap_freshness_router` (routing): Route reads by freshness requirement, snapshot availability, wait budget, and acceleration benefit.
- `cost_based_route_optimizer` (query optimization): Choose CPU, GPU, retained snapshot, refresh, transfer, or fallback routes from measured cost and cardinality signals.
- `cpu_fallback_policy` (execution): Route stale, unsupported, over-budget, or low-cardinality work to CPU without violating visibility or response guarantees.
- `multi_tier_placement` (storage placement): Place fragments across HBM, DRAM, compressed host memory, NVMe, object storage, and future CXL/NVM by measured reuse and movement cost.
- `isolation_trace_oracle` (validation): Validate every observed read against allowed snapshot intervals, fallback routes, and write-publication claims.

## Mechanism Cards By Layer

### Mvcc

#### `mvcc_gc_frontiers` - MVCC garbage-collection frontiers

Reclaim old versions by exact active snapshot generations and bounded multiversion collection rather than periodic blind cleanup.

- evidence: Scalable Garbage Collection for In-Memory MVCC, Practically and Theoretically Efficient GC for Multiversioning, BOHM
- provides: bounded version memory; long-reader accounting; hot-chain pruning
- requires: active snapshot registry; safe reclamation frontier; index/version unlink protocol
- decision: prototype - Exact active-reader frontiers are needed to bound old versions, but retention behavior under GPU snapshots must be measured.
- benchmark: Long-reader and hot-write MVCC retention benchmark with exact frontier pruning.

#### `snapshot_frontier_vectors` - Snapshot frontier vectors

Use compact scalar generations on hot paths and vector/frontier state only where cross-owner snapshot proof is required.

- evidence: Taurus MM, Chardonnay, Read-Safe Snapshots, Scalable Snapshot Isolation
- provides: multi-owner visibility proof; long-reader admission basis; refresh safety
- requires: owner generation map; active reader horizon; frontier compaction
- decision: prototype - Cross-owner visibility proof is required for retained snapshots, while the scalar/vector split needs implementation evidence.
- benchmark: Vector-scalar visibility simulator across owners with retained snapshot admission and fallback.

### Durability

#### `wal_before_visibility` - WAL-before-visibility boundary

Durable commit evidence must exist before SQL-visible state or route eligibility advances.

- evidence: ARIES lineage, FineLine, Siberia, fsync failures, Chipmunk, CCFS
- provides: crash-safe commit boundary; old-or-new recovery basis; replication handoff point
- requires: commit generation; durable log frontier; visibility publication gate
- decision: adopt_now - Durability before SQL-visible publication is a non-negotiable correctness invariant, and many downstream route/root/fallback mechanisms require it.
- benchmark: Crash-injected commit and route-publication matrix proving no visible generation lacks durable WAL evidence.

### Execution

#### `cpu_fallback_policy` - CPU fallback policy

Route stale, unsupported, over-budget, or low-cardinality work to CPU without violating visibility or response guarantees.

- evidence: ADR-003 local decision, GPU-Accelerated OLTP, veDB-HTAP, TetriSched
- provides: correctness-preserving escape hatch; latency cap; unsupported-route handling
- requires: fallback reason; same SQL semantics; snapshot proof
- decision: adopt_now - Every accelerated route needs an explicit safe fallback reason so correctness and latency budgets survive stale residency or resource pressure.
- benchmark: Split CPU/GPU route benchmark with fallback reasons and isolation trace validation.

#### `retained_gpu_snapshots` - Retained GPU snapshots

Keep hot read generations resident on GPU and route matching reads directly when freshness and residency prove valid.

- evidence: vDriver, vWeaver, Diva, veDB-HTAP, GPU-Accelerated OLTP
- provides: low-latency repeated reads; reduced CPU owner pressure; GPU amortization
- requires: snapshot generation; residency descriptor; stale-route rejection; refresh policy
- decision: prototype - Resident read generations are central to the GPU value proposition, but freshness, HBM pressure, and version retention need prototype evidence.
- benchmark: Same-shape retained lookup benchmark under refresh lag, write churn, and fallback pressure.

#### `same_shape_microbatching` - Same-shape micro-batching

Batch repeated prepared route shapes by snapshot generation, partition, and result shape before GPU execution.

- evidence: GaccO, LTPG, GPU-Accelerated OLTP, Pipelined Query Processing, push/pull query papers
- provides: kernel launch amortization; coalesced reads; response scatter efficiency
- requires: route shape id; snapshot generation; batch size/latency cap
- decision: prototype - Repeated route shapes are a core GPU execution opportunity, while batch limits need latency and occupancy curves before adoption.
- benchmark: Prepared point lookup and aggregate micro-batch curves over p50, p99, and GPU occupancy.

### Memory Reclamation

#### `bounded_descriptor_reclamation` - Bounded descriptor reclamation

Retire route, catalog, and plan descriptors through bounded hazard/era/epoch-style protection.

- evidence: WFE, Crystalline, Hyaline, Publish on Ping, NBR, Hazard Eras
- provides: safe lock-free descriptor lifetime; bounded retired metadata; reader-friendly publication
- requires: reader protection token; retire queues; stalled-reader policy
- decision: prototype - Immutable publication requires safe descriptor lifetime, while the specific hazard/era/epoch strategy should follow churn measurements.
- benchmark: Route-descriptor churn benchmark with long readers, stalled sessions, and bounded retired bytes.

### Metadata Publication

#### `dependency_witnesses` - Dependency witnesses

Represent delayed internal publication as explicit dependency chains that must fence before external eligibility.

- evidence: ASAP, MOD, Graphene, CCFS
- provides: safe async metadata persistence; explainable route readiness; ordered recovery proof
- requires: dependency ids; fence state; root eligibility check
- decision: prototype - Async publication needs explicit dependency proof, but the witness representation should be prototyped before fixing hot-path shape.
- benchmark: Async body-persist simulator that proves readers see only fenced roots while measuring ordered flush reduction.

#### `immutable_route_roots` - Immutable route roots

Publish route, catalog, layout, residency, and visibility metadata as immutable bodies plus compact root generations.

- evidence: MOD, Zen, REWIND, Falcon, ASAP
- provides: atomic route eligibility; rebuildable metadata; compact invalidation handle
- requires: body checksum; root generation; reclaimable old roots
- decision: adopt_now - Compact immutable generations give the architecture a shared publication contract for routes, catalogs, residency, and visibility.
- benchmark: Route-root publish benchmark comparing full rewrite, in-place mutation, and immutable-body/root-swap under crash injection.

### Query Optimization

#### `cost_based_route_optimizer` - Cost-based route optimizer

Choose CPU, GPU, retained snapshot, refresh, transfer, or fallback routes from measured cost and cardinality signals.

- evidence: Holon, Cardinality Estimation surveys, Free Join, Learned Query Optimizer, NoisePage/DBOS line
- provides: route choice; join/scan planning; transfer avoidance
- requires: cardinality signals; route telemetry; guarded planner contract
- decision: prototype - The system needs deterministic route choice over CPU, GPU, retained, refresh, and cold paths before learning or advanced placement can matter.
- benchmark: Route-choice benchmark over point, range, join, retained, and cold-tier queries.

#### `learned_optimizer_advisor` - Learned optimizer advisor

Use ML or adaptive learning to suggest route, knob, or placement decisions while keeping correctness outside the model.

- evidence: Learned Query Optimizer, Bao, Holon, Polyjuice
- provides: adaptive tuning; large design-space pruning; workload-sensitive route hints
- requires: safe action envelope; fallback baseline; online regret telemetry
- decision: defer - Learned route advice should wait until deterministic route telemetry, guardrails, and baseline costs are stable.
- benchmark: Advisor-vs-baseline route selection with guardrail rejections and p99/regret tracking.

### Routing

#### `htap_freshness_router` - HTAP freshness router

Route reads by freshness requirement, snapshot availability, wait budget, and acceleration benefit.

- evidence: veDB-HTAP, vWeaver, Diva, F1 Lightning
- provides: freshness-aware acceleration; bounded stale-route rejection; wait-or-fallback decision
- requires: freshness contract; snapshot frontier; route cost model
- decision: prototype - Freshness-aware routing is fundamental to retained reads, but exact wait, refresh, and fallback thresholds need prototype feedback.
- benchmark: Freshness router benchmark over read-committed, bounded-staleness, and exact-snapshot routes.

### Runtime Admission

#### `effective_session_counting` - Effective session counting

Count only sessions with ready work or blocked responses as active flows; idle logical sessions consume minimal hot-path capacity.

- evidence: TFC, Demikernel, DBOS, HFT/runtime papers
- provides: 1M logical-session scalability; idle-session cheapness; fair active-flow accounting
- requires: ready-work flags; socket writeability state; session memory tiers
- decision: prototype - Large logical-session counts are a target workload, but active-flow accounting needs a simulator before adoption as a fixed runtime rule.
- benchmark: Fan-in activation simulator comparing logical sessions, active flows, and hot-path memory.

#### `vector_credit_admission` - Vector-credit admission

Admit work using per-boundary credits for requests, bytes, pinned buffers, GPU streams, WAL slots, and response capacity.

- evidence: TFC, PIFO/SP-PIFO, HostCC, Justitia, Silo
- provides: zero-standing-queue target; explicit overload reason; bounded p99 pressure
- requires: resource counters; admission interval; fallback/reject policy
- decision: adopt_now - Route choice must expose bounded resource budgets up front so GPU, CPU, WAL, buffer, and response queues cannot hide overload.
- benchmark: 1M logical-session simulator with vector credits and standing-queue telemetry.

### Runtime Scheduling

#### `deficit_fairness` - Deficit fairness

Allow short-term throughput-favorable scheduling while tracking deficits so interactive or tenant classes do not starve.

- evidence: Graphene, PIFO/SP-PIFO, Justitia, Silo
- provides: bounded unfairness; tenant/session SLO guard; batching without starvation
- requires: class counters; queue-wait telemetry; overload policy
- decision: prototype - Fairness counters are needed to bound batching and DAG scheduling bias, but policy constants should come from mixed-class measurements.
- benchmark: Mixed class retained-read benchmark with throughput, p99, and deficit bound gates.

#### `owner_ring_bundling` - Owner-ring bundling

Drain bounded bundles from owner rings and record why selected or skipped work moved forward.

- evidence: Graphene, TetriSched, Syrup, libpreemptible, mechanical sympathy queue papers
- provides: low-allocation scheduling; observable queue decisions; micro-batch formation
- requires: bounded ring window; selection policy; skip reason telemetry
- decision: prototype - Owner-local bounded drains are promising for low-allocation scheduling, with selection policy and skip telemetry still needing proof.
- benchmark: Owner-drain benchmark comparing FIFO, bundle packing, and troublesome-first selection.

#### `resource_dag_scheduling` - Resource-DAG scheduling

Represent query, refresh, write, transfer, kernel, and response work as small dependency DAGs over scarce resources.

- evidence: Graphene, TetriSched, Hawk, Firmament
- provides: dependency-aware GPU scheduling; refresh/read co-scheduling; scarce-resource packing
- requires: fragment estimates; dependency edges; bounded online planner
- decision: benchmark_only - DAG scheduling could improve scarce-resource packing, but estimation errors can hurt p99 and must be benchmarked against simpler queues.
- benchmark: Route-DAG simulator over refresh, H2D, kernel, D2H, and response fragments.

### Storage Layout

#### `stable_handle_indirection` - Stable handle indirection

Separate logical row/route identity from movable physical placement through handles, forwarding records, or page tables.

- evidence: AIFM, RUMA, Log-Structured NVM, GC research lane, Mosaic
- provides: safe movement across tiers; compaction support; resident descriptor stability
- requires: handle table; generation checks; movement publication protocol
- decision: prototype - Tier movement and compaction need stable identity, but lookup overhead must be measured before the handle shape is fixed.
- benchmark: Handle-table lookup and movement benchmark across HBM, DRAM, and NVMe-resident fragments.

### Storage Placement

#### `db_owned_cold_objects` - DB-owned cold objects

Manage cold blobs, segments, and object-store files with database-owned layout, aging, and metadata rather than generic filesystem assumptions.

- evidence: BLOB papers, Vortex, RocksDB workload papers, storage-device mismatch papers
- provides: cold-tier scan/index control; object lifecycle ownership; backup/PITR alignment
- requires: object manifest; compaction/aging policy; backup boundary
- decision: prototype - Cold object manifests should be DB-owned to preserve recovery and routing semantics, with compaction and backup boundaries still to prove.
- benchmark: Cold-object read/write/compaction benchmark with object manifest crash recovery.

#### `log_structured_warm_tier` - Log-structured warm tier

Treat future DRAM/NVM/CXL warm metadata and fragments as append/log-structured homes with rebuildable volatile indexes.

- evidence: Log-Structured NVM, NOVA, REWIND, Falcon, DudeTM
- provides: write coalescing; rebuildable metadata; reduced in-place persist ordering
- requires: segment log; mapping rebuild; GC/compaction policy
- decision: benchmark_only - A rebuildable warm tier may simplify recovery, but append-map, root-publication, and in-place metadata variants need direct comparison.
- benchmark: Warm-tier manifest benchmark comparing append-map rebuild, in-place metadata, and root publication.

#### `multi_tier_placement` - Multi-tier placement policy

Place fragments across HBM, DRAM, compressed host memory, NVMe, object storage, and future CXL/NVM by measured reuse and movement cost.

- evidence: AIFM, Mosaic, vDriver/vWeaver, BtrBlocks, Five-minute rule cloud papers, SAP HANA CXL work
- provides: bounded HBM use; warm/cold separation; placement explainability
- requires: fragment heat; movement cost; tier capacity; freshness policy
- decision: prototype - HBM/DRAM/NVMe placement is necessary for capacity, but movement costs and hit-rate targets need simulator and prototype data.
- benchmark: Tier-placement simulator measuring HBM hit rate, movement latency, p99, and write amplification.

### Transaction Execution

#### `deterministic_hot_write_templates` - Deterministic hot-write templates

Use predictable per-object queues or deterministic templates for highly contended writes to avoid abort/retry amplification.

- evidence: DecentSched, PLOR, batching/reordering OCC, Mostly-Optimistic CC
- provides: hot-key tail control; lower abort rate; owner-local scheduling
- requires: hot-key detection; template or queue position; fallback for dynamic transactions
- decision: benchmark_only - Hot-write templates may beat abort/retry loops for narrow key shapes, but they should remain benchmark-gated until workload fit is proven.
- benchmark: Zipfian hot-key simulator comparing retry, ordered locking, owner serialization, and deterministic queue positions.

#### `gpu_oltp_conflict_ordering` - GPU OLTP conflict ordering

Preprocess large transaction batches to detect conflicts and execute compatible work on GPU.

- evidence: GaccO, LTPG, GPU-Accelerated OLTP, PLOR
- provides: GPU write-path throughput; high-contention batch ordering; conflict visibility
- requires: known access sets or conservative conflict model; batch boundary; CPU WAL publication
- decision: benchmark_only - GPU conflict preprocessing is useful only for specific hot-key batch regimes and must compete with CPU OCC and owner serialization.
- benchmark: YCSB/TPC-C style hot-key batch benchmark comparing CPU OCC, owner serialization, and GPU conflict ordering.

### Validation

#### `isolation_trace_oracle` - Isolation trace oracle

Validate every observed read against allowed snapshot intervals, fallback routes, and write-publication claims.

- evidence: Leopard, PolySI, Elle/Jepsen lineage
- provides: black-box isolation validation; fallback correctness gate; snapshot route audit
- requires: operation trace; version intervals; declared isolation contract
- decision: adopt_now - Fallback and retained-snapshot routes need continuous external validation that returned versions match the declared isolation contract.
- benchmark: CPU-only, GPU-retained, split CPU/GPU, and fallback MVCC trace oracle.

#### `semantic_crash_oracle` - Semantic crash oracle

Test recovery by semantic states and externally visible outcomes rather than by exhaustively enumerating all crash points.

- evidence: B3, Pathfinder, Chipmunk, fsync failures
- provides: recovery proof gate; failure-state coverage; unsafe publication detector
- requires: observable invariants; crash-state generator; old-or-new expectation model
- decision: adopt_now - Crash-state validation is required to prove WAL, route-root, catalog, and residency publication boundaries before optimized paths are trusted.
- benchmark: Crash-state generator for WAL, route roots, catalog roots, and residency manifests.

## Compatibility Matrices

## End To End Spine

| mechanism | wal_before_visibility | immutable_route_roots | dependency_witnesses | semantic_crash_oracle | snapshot_frontier_vectors | retained_gpu_snapshots | vector_credit_admission | owner_ring_bundling | cost_based_route_optimizer | cpu_fallback_policy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wal_before_visibility | -- | R< | . | S< | . | . | . | . | . | . |
| immutable_route_roots | R | -- | S< | . | . | R< | . | . | . | . |
| dependency_witnesses | . | S | -- | S< | . | . | . | . | . | . |
| semantic_crash_oracle | S | . | S | -- | . | . | . | . | . | . |
| snapshot_frontier_vectors | . | . | . | . | -- | R< | . | . | . | . |
| retained_gpu_snapshots | . | R | . | . | R | -- | . | . | . | S< |
| vector_credit_admission | . | . | . | . | . | . | -- | R< | R< | . |
| owner_ring_bundling | . | . | . | . | . | . | R | -- | . | . |
| cost_based_route_optimizer | . | . | . | . | . | . | R | . | -- | . |
| cpu_fallback_policy | . | . | . | . | . | S | . | . | . | -- |

Legend: `R` requires, `S` strengthens, `C` compatible, `T` tension, `X` conflicts, `A` alternative, `?` unknown. A trailing `<` means the strongest edge points from the column mechanism back to the row mechanism.

Mechanisms in this lens:
- `wal_before_visibility`: Durable commit evidence must exist before SQL-visible state or route eligibility advances.
- `immutable_route_roots`: Publish route, catalog, layout, residency, and visibility metadata as immutable bodies plus compact root generations.
- `dependency_witnesses`: Represent delayed internal publication as explicit dependency chains that must fence before external eligibility.
- `semantic_crash_oracle`: Test recovery by semantic states and externally visible outcomes rather than by exhaustively enumerating all crash points.
- `snapshot_frontier_vectors`: Use compact scalar generations on hot paths and vector/frontier state only where cross-owner snapshot proof is required.
- `retained_gpu_snapshots`: Keep hot read generations resident on GPU and route matching reads directly when freshness and residency prove valid.
- `vector_credit_admission`: Admit work using per-boundary credits for requests, bytes, pinned buffers, GPU streams, WAL slots, and response capacity.
- `owner_ring_bundling`: Drain bounded bundles from owner rings and record why selected or skipped work moved forward.
- `cost_based_route_optimizer`: Choose CPU, GPU, retained snapshot, refresh, transfer, or fallback routes from measured cost and cardinality signals.
- `cpu_fallback_policy`: Route stale, unsupported, over-budget, or low-cardinality work to CPU without violating visibility or response guarantees.

## Mvcc And Freshness

| mechanism | retained_gpu_snapshots | snapshot_frontier_vectors | mvcc_gc_frontiers | htap_freshness_router | isolation_trace_oracle | cpu_fallback_policy | bounded_descriptor_reclamation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| retained_gpu_snapshots | -- | R | T | S< | S< | S< | . |
| snapshot_frontier_vectors | R< | -- | S< | R< | . | . | . |
| mvcc_gc_frontiers | T< | S | -- | . | . | . | C< |
| htap_freshness_router | S | R | . | -- | S< | S | . |
| isolation_trace_oracle | S | . | . | S | -- | R< | . |
| cpu_fallback_policy | S | . | . | S< | R | -- | . |
| bounded_descriptor_reclamation | . | . | C | . | . | . | -- |

Legend: `R` requires, `S` strengthens, `C` compatible, `T` tension, `X` conflicts, `A` alternative, `?` unknown. A trailing `<` means the strongest edge points from the column mechanism back to the row mechanism.

Mechanisms in this lens:
- `retained_gpu_snapshots`: Keep hot read generations resident on GPU and route matching reads directly when freshness and residency prove valid.
- `snapshot_frontier_vectors`: Use compact scalar generations on hot paths and vector/frontier state only where cross-owner snapshot proof is required.
- `mvcc_gc_frontiers`: Reclaim old versions by exact active snapshot generations and bounded multiversion collection rather than periodic blind cleanup.
- `htap_freshness_router`: Route reads by freshness requirement, snapshot availability, wait budget, and acceleration benefit.
- `isolation_trace_oracle`: Validate every observed read against allowed snapshot intervals, fallback routes, and write-publication claims.
- `cpu_fallback_policy`: Route stale, unsupported, over-budget, or low-cardinality work to CPU without violating visibility or response guarantees.
- `bounded_descriptor_reclamation`: Retire route, catalog, and plan descriptors through bounded hazard/era/epoch-style protection.

## Runtime Admission

| mechanism | vector_credit_admission | effective_session_counting | owner_ring_bundling | resource_dag_scheduling | deficit_fairness | same_shape_microbatching | cpu_fallback_policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vector_credit_admission | -- | S/R | R< | S | . | R< | . |
| effective_session_counting | R/S | -- | . | . | . | . | . |
| owner_ring_bundling | R | . | -- | S< | S< | . | . |
| resource_dag_scheduling | S< | . | S | -- | T< | C | . |
| deficit_fairness | . | . | S | T | -- | T< | . |
| same_shape_microbatching | R | . | . | C< | T | -- | . |
| cpu_fallback_policy | . | . | . | . | . | . | -- |

Legend: `R` requires, `S` strengthens, `C` compatible, `T` tension, `X` conflicts, `A` alternative, `?` unknown. A trailing `<` means the strongest edge points from the column mechanism back to the row mechanism.

Mechanisms in this lens:
- `vector_credit_admission`: Admit work using per-boundary credits for requests, bytes, pinned buffers, GPU streams, WAL slots, and response capacity.
- `effective_session_counting`: Count only sessions with ready work or blocked responses as active flows; idle logical sessions consume minimal hot-path capacity.
- `owner_ring_bundling`: Drain bounded bundles from owner rings and record why selected or skipped work moved forward.
- `resource_dag_scheduling`: Represent query, refresh, write, transfer, kernel, and response work as small dependency DAGs over scarce resources.
- `deficit_fairness`: Allow short-term throughput-favorable scheduling while tracking deficits so interactive or tenant classes do not starve.
- `same_shape_microbatching`: Batch repeated prepared route shapes by snapshot generation, partition, and result shape before GPU execution.
- `cpu_fallback_policy`: Route stale, unsupported, over-budget, or low-cardinality work to CPU without violating visibility or response guarantees.

## Storage And Recovery

| mechanism | wal_before_visibility | immutable_route_roots | dependency_witnesses | semantic_crash_oracle | stable_handle_indirection | multi_tier_placement | log_structured_warm_tier | db_owned_cold_objects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wal_before_visibility | -- | R< | . | S< | . | . | . | . |
| immutable_route_roots | R | -- | S< | . | . | . | . | R< |
| dependency_witnesses | . | S | -- | S< | . | . | R< | . |
| semantic_crash_oracle | S | . | S | -- | . | . | R< | R |
| stable_handle_indirection | . | . | . | . | -- | S/R | . | . |
| multi_tier_placement | . | . | . | . | R/S | -- | C< | S< |
| log_structured_warm_tier | . | . | R | R | . | C | -- | . |
| db_owned_cold_objects | . | R | . | R< | . | S | . | -- |

Legend: `R` requires, `S` strengthens, `C` compatible, `T` tension, `X` conflicts, `A` alternative, `?` unknown. A trailing `<` means the strongest edge points from the column mechanism back to the row mechanism.

Mechanisms in this lens:
- `wal_before_visibility`: Durable commit evidence must exist before SQL-visible state or route eligibility advances.
- `immutable_route_roots`: Publish route, catalog, layout, residency, and visibility metadata as immutable bodies plus compact root generations.
- `dependency_witnesses`: Represent delayed internal publication as explicit dependency chains that must fence before external eligibility.
- `semantic_crash_oracle`: Test recovery by semantic states and externally visible outcomes rather than by exhaustively enumerating all crash points.
- `stable_handle_indirection`: Separate logical row/route identity from movable physical placement through handles, forwarding records, or page tables.
- `multi_tier_placement`: Place fragments across HBM, DRAM, compressed host memory, NVMe, object storage, and future CXL/NVM by measured reuse and movement cost.
- `log_structured_warm_tier`: Treat future DRAM/NVM/CXL warm metadata and fragments as append/log-structured homes with rebuildable volatile indexes.
- `db_owned_cold_objects`: Manage cold blobs, segments, and object-store files with database-owned layout, aging, and metadata rather than generic filesystem assumptions.

## Optimizer And Execution

| mechanism | cost_based_route_optimizer | learned_optimizer_advisor | htap_freshness_router | retained_gpu_snapshots | same_shape_microbatching | gpu_oltp_conflict_ordering | deterministic_hot_write_templates | resource_dag_scheduling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cost_based_route_optimizer | -- | S< | S | . | . | . | . | . |
| learned_optimizer_advisor | S | -- | . | . | . | . | . | . |
| htap_freshness_router | S< | . | -- | S | . | . | . | . |
| retained_gpu_snapshots | . | . | S< | -- | S< | . | . | . |
| same_shape_microbatching | . | . | . | S | -- | T< | . | C< |
| gpu_oltp_conflict_ordering | . | . | . | . | T | -- | A | . |
| deterministic_hot_write_templates | . | . | . | . | . | A< | -- | . |
| resource_dag_scheduling | . | . | . | . | C | . | . | -- |

Legend: `R` requires, `S` strengthens, `C` compatible, `T` tension, `X` conflicts, `A` alternative, `?` unknown. A trailing `<` means the strongest edge points from the column mechanism back to the row mechanism.

Mechanisms in this lens:
- `cost_based_route_optimizer`: Choose CPU, GPU, retained snapshot, refresh, transfer, or fallback routes from measured cost and cardinality signals.
- `learned_optimizer_advisor`: Use ML or adaptive learning to suggest route, knob, or placement decisions while keeping correctness outside the model.
- `htap_freshness_router`: Route reads by freshness requirement, snapshot availability, wait budget, and acceleration benefit.
- `retained_gpu_snapshots`: Keep hot read generations resident on GPU and route matching reads directly when freshness and residency prove valid.
- `same_shape_microbatching`: Batch repeated prepared route shapes by snapshot generation, partition, and result shape before GPU execution.
- `gpu_oltp_conflict_ordering`: Preprocess large transaction batches to detect conflicts and execute compatible work on GPU.
- `deterministic_hot_write_templates`: Use predictable per-object queues or deterministic templates for highly contended writes to avoid abort/retry amplification.
- `resource_dag_scheduling`: Represent query, refresh, write, transfer, kernel, and response work as small dependency DAGs over scarce resources.

## Tension And Alternative Zones

- `tension`: `retained_gpu_snapshots` (Retained GPU snapshots) -> `mvcc_gc_frontiers` (MVCC garbage-collection frontiers): Retained snapshots improve reads but can pin old versions unless GC accounts for reader horizons.
- `tension`: `deficit_fairness` (Deficit fairness) -> `resource_dag_scheduling` (Resource-DAG scheduling): Resource-efficient schedules can starve small classes unless deficits bound schedule preference.
- `tension`: `same_shape_microbatching` (Same-shape micro-batching) -> `deficit_fairness` (Deficit fairness): Batch formation can delay singleton queries unless fairness and latency caps are explicit.
- `alternative_to`: `gpu_oltp_conflict_ordering` (GPU OLTP conflict ordering) -> `deterministic_hot_write_templates` (Deterministic hot-write templates): Both address hot write contention; choose GPU batch ordering for large batches and templates for low-latency hot keys.
- `tension`: `learned_optimizer_advisor` (Learned optimizer advisor) -> `wal_before_visibility` (WAL-before-visibility boundary): Learned decisions must never choose actions that weaken durability or visibility invariants.
- `tension`: `gpu_oltp_conflict_ordering` (GPU OLTP conflict ordering) -> `same_shape_microbatching` (Same-shape micro-batching): Read micro-batches and write conflict batches can compete for GPU streams; admission must separate budgets.

## Full Edge List

- `gpu_oltp_conflict_ordering` --alternative_to--> `deterministic_hot_write_templates`: Both address hot write contention; choose GPU batch ordering for large batches and templates for low-latency hot keys.
- `bounded_descriptor_reclamation` --compatible--> `mvcc_gc_frontiers`: Descriptor retirement and MVCC version reclamation can share reader horizon concepts while reclaiming different objects.
- `bounded_descriptor_reclamation` --compatible--> `stable_handle_indirection`: Handles stabilize logical identity while reclamation retires obsolete physical descriptors.
- `gpu_oltp_conflict_ordering` --compatible--> `same_shape_microbatching`: Both use batch boundaries, but one targets write conflicts while the other targets repeated read/route shapes.
- `log_structured_warm_tier` --compatible--> `multi_tier_placement`: A warm log-structured home can act as the middle tier between HBM/DRAM and cold objects.
- `multi_tier_placement` --compatible--> `cost_based_route_optimizer`: Placement supplies physical choices and the optimizer selects among them based on cost and freshness.
- `resource_dag_scheduling` --compatible--> `dependency_witnesses`: Runtime dependencies and durable publication dependencies are distinct but share a witness vocabulary.
- `resource_dag_scheduling` --compatible--> `same_shape_microbatching`: Repeated route-shape batches can be represented as DAG fragments with shared transfer and kernel nodes.
- `cost_based_route_optimizer` --requires--> `vector_credit_admission`: The optimizer needs live resource budgets; otherwise cheap-looking routes can create hidden queues.
- `cpu_fallback_policy` --requires--> `isolation_trace_oracle`: Fallback must be validated as the same SQL-visible snapshot contract, not a weaker path.
- `db_owned_cold_objects` --requires--> `immutable_route_roots`: Cold-object manifests must be published through generation roots to avoid dangling route metadata.
- `effective_session_counting` --requires--> `vector_credit_admission`: The active-flow count becomes useful only when translated into bounded per-resource budgets.
- `gpu_oltp_conflict_ordering` --requires--> `wal_before_visibility`: GPU conflict-ordered batches still need CPU-owned durable publication before visibility.
- `htap_freshness_router` --requires--> `snapshot_frontier_vectors`: Freshness routing depends on knowing which generation satisfies the request.
- `immutable_route_roots` --requires--> `wal_before_visibility`: Route roots become externally eligible only after their referenced visibility/WAL frontier is durable.
- `learned_optimizer_advisor` --requires--> `isolation_trace_oracle`: Adaptive route decisions need continuous validation that observed reads remain within the declared isolation contract.
- `log_structured_warm_tier` --requires--> `dependency_witnesses`: Append-style warm metadata can persist asynchronously only if publication dependencies are explicit.
- `log_structured_warm_tier` --requires--> `semantic_crash_oracle`: Rebuildable mappings need crash tests that verify replay, root, and compaction states.
- `multi_tier_placement` --requires--> `stable_handle_indirection`: Promotion, demotion, and compaction need handles or generation-checked indirection.
- `owner_ring_bundling` --requires--> `vector_credit_admission`: Bundle selection must respect available tokens before it pushes work into downstream queues.
- `retained_gpu_snapshots` --requires--> `immutable_route_roots`: Readers need compact route metadata tying residency, layout, visibility, and freshness together.
- `retained_gpu_snapshots` --requires--> `snapshot_frontier_vectors`: Resident snapshots need a proof that their generation is valid across owners or partitions.
- `same_shape_microbatching` --requires--> `vector_credit_admission`: Micro-batches need request, byte, buffer, GPU, and latency budgets before admission.
- `semantic_crash_oracle` --requires--> `db_owned_cold_objects`: Cold object manifests, aging, and backup boundaries must survive crash-state checks.
- `bounded_descriptor_reclamation` --strengthens--> `immutable_route_roots`: Old immutable roots and descriptors need bounded retirement under long readers.
- `cost_based_route_optimizer` --strengthens--> `htap_freshness_router`: Cost estimates help choose wait, refresh, retained GPU, CPU, or cold-tier routes inside the freshness envelope.
- `cpu_fallback_policy` --strengthens--> `retained_gpu_snapshots`: Fallback preserves correctness when a resident snapshot is stale, missing, or over budget.
- `db_owned_cold_objects` --strengthens--> `multi_tier_placement`: Cold objects become an explicit tier in placement and route decisions.
- `deficit_fairness` --strengthens--> `owner_ring_bundling`: Deficit counters cap the unfairness introduced by packing-friendly bundle selection.
- `dependency_witnesses` --strengthens--> `immutable_route_roots`: Witnesses explain which route body, generation, and checksum dependencies must fence before root eligibility.
- `deterministic_hot_write_templates` --strengthens--> `owner_ring_bundling`: Owner rings can schedule hot-key queues by deterministic position instead of retrying optimistic conflicts.
- `htap_freshness_router` --strengthens--> `cpu_fallback_policy`: Fallback is one of the router's safe outcomes when freshness or residency does not fit.
- `htap_freshness_router` --strengthens--> `retained_gpu_snapshots`: Freshness contracts determine when retained GPU snapshots are eligible.
- `isolation_trace_oracle` --strengthens--> `htap_freshness_router`: Trace validation catches stale-route mistakes made by freshness routing.
- `isolation_trace_oracle` --strengthens--> `retained_gpu_snapshots`: Resident reads need black-box proof that their returned versions fit the requested isolation contract.
- `learned_optimizer_advisor` --strengthens--> `cost_based_route_optimizer`: Learning should suggest route or knob choices inside a deterministic optimizer guardrail.
- `multi_tier_placement` --strengthens--> `retained_gpu_snapshots`: Placement policy decides which generations deserve scarce HBM residency.
- `mvcc_gc_frontiers` --strengthens--> `snapshot_frontier_vectors`: Exact active frontiers make old-version reclamation compatible with cross-owner snapshots.
- `resource_dag_scheduling` --strengthens--> `owner_ring_bundling`: DAG annotations guide which bounded bundle fragments unlock scarce resources.
- `same_shape_microbatching` --strengthens--> `retained_gpu_snapshots`: Resident generations provide a natural grouping key for same-shape read batches.
- `semantic_crash_oracle` --strengthens--> `dependency_witnesses`: The oracle checks whether dependency chains recover as old-or-new instead of partial mixed states.
- `semantic_crash_oracle` --strengthens--> `wal_before_visibility`: Crash-state validation proves the WAL boundary is not bypassed by optimized publication paths.
- `stable_handle_indirection` --strengthens--> `multi_tier_placement`: Tier movement is safer when logical identity survives physical relocation.
- `stable_handle_indirection` --strengthens--> `retained_gpu_snapshots`: Handles let resident snapshots survive movement or compaction of their backing fragments.
- `vector_credit_admission` --strengthens--> `effective_session_counting`: Credits should be allocated to effective active work, not total logical connection count.
- `vector_credit_admission` --strengthens--> `resource_dag_scheduling`: Credits prevent resource-DAG scheduling from pushing too much work into a scarce downstream boundary.
- `deficit_fairness` --tension--> `resource_dag_scheduling`: Resource-efficient schedules can starve small classes unless deficits bound schedule preference.
- `gpu_oltp_conflict_ordering` --tension--> `same_shape_microbatching`: Read micro-batches and write conflict batches can compete for GPU streams; admission must separate budgets.
- `learned_optimizer_advisor` --tension--> `wal_before_visibility`: Learned decisions must never choose actions that weaken durability or visibility invariants.
- `retained_gpu_snapshots` --tension--> `mvcc_gc_frontiers`: Retained snapshots improve reads but can pin old versions unless GC accounts for reader horizons.
- `same_shape_microbatching` --tension--> `deficit_fairness`: Batch formation can delay singleton queries unless fairness and latency caps are explicit.
