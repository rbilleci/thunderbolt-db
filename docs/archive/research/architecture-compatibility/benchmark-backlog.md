# GPU DB Research Benchmark Backlog

This report is generated from mechanism decisions and paper-mechanism
relations. It answers: what experiments decide the architecture?

Regenerate with:

```sh
python3 scripts/generate_research_paper_mechanism_links.py
```

## Summary

- backlog items: 25
- mechanisms with benchmark questions: 25
- paper links requiring benchmark/proof gates: 560

Gate type counts:
- correctness_gate: 6 (must continuously validate a baseline architecture invariant)
- decision_gate: 4 (must decide whether the mechanism graduates or stays experimental)
- deferred_gate: 1 (recorded for later once prerequisite architecture telemetry exists)
- prototype_gate: 14 (must produce prototype evidence before the mechanism is adopted)

Decision status counts:
- adopt_now: 6
- prototype: 14
- benchmark_only: 4
- defer: 1

Paper relation counts feeding the backlog:
- alternative_to: 170 (paper evidence describes an alternative to this mechanism)
- benchmark_required: 560 (paper evidence is inconclusive without a benchmark or proof gate)
- contradicts: 3 (paper evidence conflicts with this mechanism)
- only_valid_if: 259 (paper evidence supports this mechanism only under named conditions)
- supports: 3558 (paper evidence supports or motivates this mechanism)
- warns_against: 121 (paper evidence warns against adopting this mechanism without constraints)

## Experiment Backlog

### `benchmark-wal_before_visibility`

- mechanism: `wal_before_visibility` (WAL-before-visibility boundary)
- layer: durability
- decision: adopt_now - Durability before SQL-visible publication is a non-negotiable correctness invariant, and many downstream route/root/fallback mechanisms require it.
- gate: correctness_gate - must continuously validate a baseline architecture invariant
- experiment: Crash-injected commit and route-publication matrix proving no visible generation lacks durable WAL evidence.
- evidence links: 410
- benchmark-required links: 48
- relation counts: alternative_to=9, benchmark_required=48, only_valid_if=25, supports=309, warns_against=19

Evidence examples:
- `2026-06-03-hybridgc-production-mvcc-garbage-collection-in-sap-hana` benchmark_required/high: Required metrics: commit-path overhead, GC scan work, reclaimed versions per pass, and WAL replay correctness. - Add per-table or per-partition snapshot trackers to the MVCC test harness.
- `2026-06-03-memory-optimized-mvcc-for-disk-backed-storage` benchmark_required/high: WAL replay rebuilds the durable latest page state, and the MVCC subsystem restarts with empty version chains and fresh timestamp state. - Rollback is coordinated with ARIES-style logging by scanning log records, writing compensation log records, restoring before-images on pages, and unlinking irrelevant versions. - Bulk operations take exclusive write acc...
- `2026-06-03-modern-nvme-storage-engine-exploitation` benchmark_required/high: GPU DB cannot copy those numbers into a WAL/MVCC benchmark without measuring durable writes, visibility publication, and replay.
- `2026-06-03-pasha-partitioned-shared-cxl-pod-architecture` benchmark_required/high: The partitioner should minimize shared-region operations, not merely minimize multi-host transactions. - High core counts in a future pod motivate scheduling transactions before execution, rather than resolving all conflicts reactively at runtime. - Durability and atomicity still require logging and checkpoints; the authors call out parallel logging and p...
- `2026-06-03-polaris-priority-aware-optimistic-concurrency-control` benchmark_required/high: The paper does not evaluate durable WAL flush cost, checkpointing, recovery replay, GPU execution, PostgreSQL protocol state, distributed clocks, or million-session admission.

### `benchmark-cpu_fallback_policy`

- mechanism: `cpu_fallback_policy` (CPU fallback policy)
- layer: execution
- decision: adopt_now - Every accelerated route needs an explicit safe fallback reason so correctness and latency budgets survive stale residency or resource pressure.
- gate: correctness_gate - must continuously validate a baseline architecture invariant
- experiment: Split CPU/GPU route benchmark with fallback reasons and isolation trace validation.
- evidence links: 166
- benchmark-required links: 35
- relation counts: alternative_to=5, benchmark_required=35, only_valid_if=10, supports=112, warns_against=4

Evidence examples:
- `2026-06-02-parqo-penalty-aware-robust-plan-selection` benchmark_required/high: A resident GPU route may be fastest when cardinality, residency, queue delay, and transfer estimates are right, but it can be a bad choice when a predicate is less selective than expected, a partition is not resident, the GPU queue is saturated, a refresh is pending, or the CPU fallback path would avoid transfer and launch overhead.
- `2026-06-03-accelerating-gpu-data-processing-with-fastlanes-compression` benchmark_required/high: If a cold partition is stored compressed on NVMe or host memory, the engine should benchmark whether moving compressed vectors to GPU and decoding in-kernel beats moving dense columns or relying on CPU fallback.
- `2026-06-03-hint-qpt-hints-for-robust-query-performance-tuning` benchmark_required/high: Proof gate: fewer bad route choices without increasing p50 latency for easy decisions. - Extend route telemetry to distinguish "unsupported", "not resident", "over budget", "queue saturated", "fragile selectivity", and "fragile response size" fallback reasons.
- `2026-06-03-learned-cost-models-need-optimizer-task-proof` benchmark_required/high: Keep hand-written cost components for H2D/D2H bytes, kernel launch count, expected rows, CPU fallback cost, resident validity, refresh cost, and queue delay; then learn residuals or rank candidates using measured endpoint telemetry.
- `2026-06-04-cardood-treats-route-estimator-drift-as-a-first-class-optimizer-risk` benchmark_required/high: Instead of one global GPU cost model, train and evaluate across explicit groups: resident versus nonresident, fresh versus stale snapshot, small lookup versus scan, low versus high queue pressure, tenant A versus tenant B, and CPU fallback versus GPU execution.

### `benchmark-immutable_route_roots`

- mechanism: `immutable_route_roots` (Immutable route roots)
- layer: metadata publication
- decision: adopt_now - Compact immutable generations give the architecture a shared publication contract for routes, catalogs, residency, and visibility.
- gate: correctness_gate - must continuously validate a baseline architecture invariant
- experiment: Route-root publish benchmark comparing full rewrite, in-place mutation, and immutable-body/root-swap under crash injection.
- evidence links: 340
- benchmark-required links: 30
- relation counts: alternative_to=10, benchmark_required=30, only_valid_if=18, supports=280, warns_against=2

Evidence examples:
- `2026-06-03-ankerdb-fine-granular-virtual-snapshotting` benchmark_required/high: - Add a CPU-side column-generation snapshot prototype for one admitted table: record a visibility boundary, lazily materialize only requested column families, and publish an immutable generation root.
- `2026-06-03-empirical-in-memory-mvcc-design-tradeoffs` benchmark_required/high: Measure CPU build time, cache misses if available, generated GPU bytes, and retained query p50/p99 after publication. - Prototype generation-level resident retirement.
- `2026-06-03-mmap-is-not-a-buffer-pool-substitute` benchmark_required/high: Gate: routes that may fault are not eligible for retained GPU low-latency execution. - Test WAL ordering with mapped immutable files only: publish a segment after WAL and checksum completion, then prove that subsequent mutation invalidates the segment generation before any stale mapped bytes can be routed.
- `2026-06-03-tiga-synchronized-clock-transaction-ordering` benchmark_required/high: A network worker can admit a batch with a target publication generation based on measured ring delay, WAL queue delay, and mutation-owner drain time.
- `2026-06-04-cross-paper-synthesis-hot-routes-need-separate-execution-and-publication-frontiers` benchmark_required/high: - Track execution frontier versus publication frontier for writes, compressed resident generations, and cold-tier offload routes in one telemetry schema. - Add crash/recovery tests that leave speculative GPU or offload work complete but unpublished, then verify no stale route becomes selectable after replay. - Measure when speculative refresh or compresse...

### `benchmark-vector_credit_admission`

- mechanism: `vector_credit_admission` (Vector-credit admission)
- layer: runtime admission
- decision: adopt_now - Route choice must expose bounded resource budgets up front so GPU, CPU, WAL, buffer, and response queues cannot hide overload.
- gate: correctness_gate - must continuously validate a baseline architecture invariant
- experiment: 1M logical-session simulator with vector credits and standing-queue telemetry.
- evidence links: 80
- benchmark-required links: 13
- relation counts: alternative_to=2, benchmark_required=13, only_valid_if=5, supports=60

Evidence examples:
- `2026-06-03-programmable-packet-scheduling-with-a-single-queue` benchmark_required/high: In the reported resource table, AIFO uses less SRAM than SP-PIFO but more stateful ALU and logical-table resources. - Simulations evaluate web-search and data-mining workloads, SRPT-like pFabric behavior, and fair queueing.
- `2026-06-03-timely-rtt-based-congestion-control-for-the-datacenter` benchmark_required/high: - TIMELY measures segment RTT using NIC hardware timestamps and prompt hardware-generated acknowledgements, then subtracts serialization time so the remaining variable component tracks propagation plus queueing delay. - The design treats NIC queueing as part of the congestion signal rather than noise, because host/NIC buffering can still inflate end-to-en...
- `2026-06-04-x-ssd-moves-wal-propagation-into-the-storage-device` benchmark_required/high: Proof gate: visibility can be published only after the configured durability counter advances. - Build a WAL credit-admission microbenchmark for COPY/INSERT batches.
- `2026-06-05-backpressure-flow-control-makes-admission-local-selective-and-bounded` benchmark_required/high: - Build an active-session admission benchmark: allocate compact idle session state for a large logical population, but bind only active requests to a limited set of read lanes, mutation lanes, response lanes, and buffer credits.
- `2026-06-05-ndp-re-architecting-datacenter-networks-and-stacks-for-low-latency` benchmark_required/high: - Add a response-ring incast benchmark: complete one retained micro-batch for 1K, 10K, and 100K logical sessions and compare unbounded response enqueueing with writer-owned response credits.

### `benchmark-isolation_trace_oracle`

- mechanism: `isolation_trace_oracle` (Isolation trace oracle)
- layer: validation
- decision: adopt_now - Fallback and retained-snapshot routes need continuous external validation that returned versions match the declared isolation contract.
- gate: correctness_gate - must continuously validate a baseline architecture invariant
- experiment: CPU-only, GPU-retained, split CPU/GPU, and fallback MVCC trace oracle.
- evidence links: 34
- benchmark-required links: 4
- relation counts: benchmark_required=4, only_valid_if=1, supports=27, warns_against=2

Evidence examples:
- `2026-06-03-oltp-through-the-looking-glass-16-years-later` benchmark_required/high: The authors benchmark whole-stack OLTP performance using VoltDB as the main modern single-partition OLTP engine, with PostgreSQL as a reference point, and compare client-side transaction logic with stored procedures under different user-code isolation mechanisms.
- `2026-06-03-tictoc-data-driven-timestamp-occ` benchmark_required/high: ...unnecessary aborts, but the paper reports no measurable performance gain for its evaluated workloads. - TicToc sketches snapshot isolation by splitting one serializable timestamp into `commit_rts` for reads and `commit_wts` for writes, while checking that updated tuples were not modified after the read timestamp. - For durability, the paper says TicToc...
- `2026-06-03-cross-paper-synthesis-gpu-writes-need-classed-conflict-lanes` benchmark_required/low: It is a classed write-admission lab: trace batches, identify hot-key and known-template shapes, measure optimistic versus deterministic preprocessing costs, and prove isolation/WAL publication with event traces.
- `2026-06-05-gpu-multitasking-needs-explicit-compute-memory-and-fault-isolation-contracts` benchmark_required/medium: Its main claim is not a measured database speedup; it is a requirements and mechanism map for moving from single-task GPU ownership to practical multitasking: high utilization, performance guarantees, fault isolation, and large-scale deployment.
- `2026-06-04-snapshot-reconstruction-as-an-optimizable-route` only_valid_if/high: - The paper assumes full Snapshot Isolation and MVCC, but deliberately focuses on read-path snapshot reconstruction.

### `benchmark-semantic_crash_oracle`

- mechanism: `semantic_crash_oracle` (Semantic crash oracle)
- layer: validation
- decision: adopt_now - Crash-state validation is required to prove WAL, route-root, catalog, and residency publication boundaries before optimized paths are trusted.
- gate: correctness_gate - must continuously validate a baseline architecture invariant
- experiment: Crash-state generator for WAL, route roots, catalog roots, and residency manifests.
- evidence links: 15
- benchmark-required links: 1
- relation counts: benchmark_required=1, only_valid_if=1, supports=13

Evidence examples:
- `2026-06-07-b3-turns-crash-consistency-into-bounded-witness-generation` benchmark_required/high: If a persisted boundary claims to survive a crash, the test harness should be able to generate short operation sequences, crash after each persistence boundary, recover, and compare the recovered database state against an oracle.
- `2026-06-07-ccfs-makes-durability-ordering-a-per-stream-contract` only_valid_if/high: Failure condition: a crash or long reader can observe a reused id without a durable free-and-retire boundary. - Measure order-preserving allocation for ingest: reserve logical segment or WAL positions at admission, delay physical placement until flush, then commit placement facts in owner order.
- `2026-06-05-easycommit-makes-non-blocking-commit-a-message-redundancy-tradeoff` supports/high: **Core idea:** EasyCommit tries to keep the two communication phases of 2PC while avoiding 2PC's blocking behavior under node failures.
- `2026-06-06-palf-replicated-wal-should-return-file-like-commit-facts-not-just-consensus-progress` supports/high: It should enter a draining classification state where every reserved commit record becomes success, failure, or unknown-crash-replay, and only then can route metadata and sessions be advanced.
- `2026-06-06-txbug-says-transaction-correctness-tests-should-be-small-semantic-and-route-aware` supports/high: The paper reports that only 23.6% of bugs cause explicit failures such as crashes or errors; 76.4% produce silent failures such as incorrect database states, incorrect query results, incorrect DBMS state, wrong blocking behavior, or performance degradation. - Existing approaches miss much of the corpus.

### `benchmark-mvcc_gc_frontiers`

- mechanism: `mvcc_gc_frontiers` (MVCC garbage-collection frontiers)
- layer: MVCC
- decision: prototype - Exact active-reader frontiers are needed to bound old versions, but retention behavior under GPU snapshots must be measured.
- gate: prototype_gate - must produce prototype evidence before the mechanism is adopted
- experiment: Long-reader and hot-write MVCC retention benchmark with exact frontier pruning.
- evidence links: 62
- benchmark-required links: 8
- relation counts: alternative_to=2, benchmark_required=8, only_valid_if=2, supports=48, warns_against=2

Evidence examples:
- `2026-06-03-ankerdb-fine-granular-virtual-snapshotting` benchmark_required/high: Their 200 MB column microbenchmark finds `vm_snapshot` stable under VMA fragmentation, 68x faster than rewiring after all 51,200 pages have been touched, and up to 6x faster on writes to snapshotted pages because ordinary kernel COW handles the write path. - Old version garbage collection is simplified for analytical history: when no transaction can acces...
- `2026-06-04-kvell-propagates-old-scan-versions-instead-of-retaining-snapshots` benchmark_required/high: Compare ordinary MVCC retention, eager propagation, and delayed propagation on version bytes, owner queue wait, mutation p99, and cleanup time. - When GPU hardware is available, benchmark a long resident aggregate while concurrent updates hit unscanned rows.
- `2026-06-06-orcgc-makes-reclamation-bounds-part-of-the-hot-path-contract` benchmark_required/high: Gate: retired bytes must remain bounded under long readers and high update rates without blocking WAL visibility publication. - Add telemetry fields for future runtime slices: protected handles by owner, retired handles by generation, maximum retirement age, handoff count, cleanup owner, and cleanup backlog bytes. - Prototype an automatic descriptor API o...
- `2026-06-03-sixth-modern-batch-synthesis` benchmark_required/medium: The latest batch spans durable write-path authority, MVCC cleanup, and robust route planning: LeanStore logging/recovery, Steam MVCC garbage collection, and PAR2QO.
- `2026-06-05-cross-paper-synthesis-frontiers-must-preflight-both-ownership-and-tiers` benchmark_required/low: - Measure route preflight as an admission boundary before mutation-owner lock hold time begins. - Add stale-certificate tests for owner movement, resident refresh, eviction, and GC of old versions. - Track separate counters for local frontier reads, global frontier reads, cross-owner ordering descriptors, and tier-pin failures. - Compare adaptive prefligh...

### `benchmark-snapshot_frontier_vectors`

- mechanism: `snapshot_frontier_vectors` (Snapshot frontier vectors)
- layer: MVCC
- decision: prototype - Cross-owner visibility proof is required for retained snapshots, while the scalar/vector split needs implementation evidence.
- gate: prototype_gate - must produce prototype evidence before the mechanism is adopted
- experiment: Vector-scalar visibility simulator across owners with retained snapshot admission and fallback.
- evidence links: 215
- benchmark-required links: 25
- relation counts: alternative_to=7, benchmark_required=25, only_valid_if=10, supports=171, warns_against=2

Evidence examples:
- `2026-06-03-ankerdb-fine-granular-virtual-snapshotting` benchmark_required/high: Proof gate: old generations retire as soon as the last statement/cursor holder releases, without relying on global epoch lag from idle sessions. - Build a negative-control virtual-memory snapshot experiment using ordinary OS mechanisms available without a patched kernel, such as fork or mmap/COW where feasible.
- `2026-06-03-caracal-deterministic-contention-management` benchmark_required/high: GPU DB should separate short read-snapshot batches from mutation epochs and measure mutation batch ceilings in microseconds or low milliseconds before considering deterministic placeholders for production.
- `2026-06-03-mainlining-databases-supporting-fast-transactional-workloads-on-universal-columnar-data-file-for` benchmark_required/high: An uncommitted transaction uses a commit timestamp with the sign bit flipped; unsigned timestamp comparison keeps uncommitted versions invisible. - Readers reconstruct a snapshot by copying the latest tuple image and applying before-images until they reach a version older than their start timestamp. - Abort handling restores the old image but does not imm...
- `2026-06-03-tesseract-online-schema-evolution` benchmark_required/high: Tesseract can overlap CDC with the scan phase, and uses multiple CDC threads. - After the scan phase, the DDL transaction obtains a pre-commit timestamp, makes the new schema visible in a pending state, and directs newly started transactions toward the new schema so no new CDC work is added. - Relaxed snapshots let the DDL scan migrate the latest committe...
- `2026-06-03-tictoc-data-driven-timestamp-occ` benchmark_required/high: ...cannot be acquired, avoiding lock convoying in the commit phase. - The preemptive-abort optimization uses an approximate commit timestamp and latest read-tuple `wts` checks to identify transactions that will fail read-set validation before they lock the write set. - The timestamp-history optimization keeps a bounded history of recent `wts` values per t...

### `benchmark-retained_gpu_snapshots`

- mechanism: `retained_gpu_snapshots` (Retained GPU snapshots)
- layer: execution
- decision: prototype - Resident read generations are central to the GPU value proposition, but freshness, HBM pressure, and version retention need prototype evidence.
- gate: prototype_gate - must produce prototype evidence before the mechanism is adopted
- experiment: Same-shape retained lookup benchmark under refresh lag, write churn, and fallback pressure.
- evidence links: 488
- benchmark-required links: 57
- relation counts: alternative_to=11, benchmark_required=57, only_valid_if=35, supports=361, warns_against=24

Evidence examples:
- `2026-06-03-aocc-adaptive-validation-for-heterogeneous-occ` benchmark_required/high: Gate: no unrelated partition write can force a retained read to scan global evidence. - Test an HTAP transaction shape: read a retained aggregate over a resident partition, then apply a write.
- `2026-06-03-autonomous-commit-for-low-latency-nvme-durability` benchmark_required/high: Gate: WAL replay produces identical CPU table state and resident invalidation generations. - Build a COPY admission benchmark with bursty clients: many small commits, idle gaps, and one long retained read snapshot.
- `2026-06-03-ermia-snapshot-friendly-mixed-workload-oltp` benchmark_required/high: Expected result: indirection helps update/index maintenance while resident GPU snapshots should flatten the extra hop before kernel execution. - Evaluate a bounded SSN-like dependency tracker only as a serializability experiment for CPU transactions first.
- `2026-06-03-gcctb-gpu-oltp-concurrency-control-study` benchmark_required/high: Device code is generated and compiled at runtime with NVRTC so table format, benchmark, index choice, and CC scheme can be changed by configuration. - The testbed keeps table and index data resident in GPU memory before the experiment and leaves updates/results on device.
- `2026-06-03-mordred-semantic-cpu-gpu-placement` benchmark_required/high: Required measurements: HBM bytes, PCIe/D2H bytes, CPU materialization time, response bytes, and null/empty correctness. - Add correlated-component telemetry to residency: required components, missing companions, partial-route hit, full-route hit, and reason why a resident component could not shorten the request.

### `benchmark-same_shape_microbatching`

- mechanism: `same_shape_microbatching` (Same-shape micro-batching)
- layer: execution
- decision: prototype - Repeated route shapes are a core GPU execution opportunity, while batch limits need latency and occupancy curves before adoption.
- gate: prototype_gate - must produce prototype evidence before the mechanism is adopted
- experiment: Prepared point lookup and aggregate micro-batch curves over p50, p99, and GPU occupancy.
- evidence links: 65
- benchmark-required links: 12
- relation counts: alternative_to=4, benchmark_required=12, only_valid_if=2, supports=44, warns_against=3

Evidence examples:
- `2026-06-03-themis-gpu-relational-pipeline-load-balancing` benchmark_required/high: Required measurements: HBM bytes read, D2H bytes, kernel time, response encoding time, and correctness under null/empty result cases. - For same-shape lookup micro-batches, test whether fixed batch size is enough or whether batches need work-aware splitting by expected fanout.
- `2026-06-04-compound-gpu-pipelines-trade-materialization-for-explicit-reduction-pressure` benchmark_required/high: For same-shape retained lookups and micro-batched aggregates, compound kernels suggest a first benchmark shape: fuse predicate evaluation, visibility-vector check, row-id/key lookup, projection, and compact result scattering into one kernel when the snapshot generation and output schema match.
- `2026-06-04-gpu-learned-indexes-need-batch-shaped-residency-contracts` benchmark_required/high: Measure throughput, p50/p95/p99 latency, kernel launches, CPU last-mile time, D2H result bytes, and queue wait. - Sweep micro-batch sizes for same-shape retained lookups.
- `2026-06-05-adaptive-execution-makes-compilation-a-runtime-route-not-a-startup-tax` benchmark_required/high: For short retained reads, catalog queries, tiny point lookups, or first-run prepared statements, the fastest route may be an interpreted/fused CPU path or an existing generic kernel, even if a compiled or specialized GPU route would win on a larger batch.
- `2026-06-05-relaxed-operator-fusion-makes-materialization-a-route-shape-decision` benchmark_required/high: Proof gate: planner avoids prefetch/GPU staging when overhead dominates. - Compare micro-batch sizes for same-shape retained lookups as ROF-style stage vector sizes: 64, 256, 1k, 4k, 16k, and latency-capped dynamic fill.

### `benchmark-bounded_descriptor_reclamation`

- mechanism: `bounded_descriptor_reclamation` (Bounded descriptor reclamation)
- layer: memory reclamation
- decision: prototype - Immutable publication requires safe descriptor lifetime, while the specific hazard/era/epoch strategy should follow churn measurements.
- gate: prototype_gate - must produce prototype evidence before the mechanism is adopted
- experiment: Route-descriptor churn benchmark with long readers, stalled sessions, and bounded retired bytes.
- evidence links: 489
- benchmark-required links: 43
- relation counts: alternative_to=23, benchmark_required=43, only_valid_if=28, supports=384, warns_against=11

Evidence examples:
- `2026-06-03-activepointers-software-address-translation-on-gpus` benchmark_required/high: The paper uses a hash table for all files in the GPU page cache, fine-grained bucket locking for inserts, lock-free reads, and batched host- to-GPU transfers for 4KB pages. - Fault-free microbenchmarks show the cost shape: 8-byte memory copies reached 97.6% of the measured `cudaMemcpyDeviceToDevice` bandwidth, while 4-byte copies reached 65.4%; compute-in...
- `2026-06-03-learned-cost-models-need-optimizer-task-proof` benchmark_required/high: It evaluates seven learned cost models against PostgreSQL cost models on three optimizer tasks: join ordering, access path selection, and physical operator selection.
- `2026-06-04-bght-makes-gpu-hash-indexes-a-probe-budgeted-route-not-just-a-lookup-primitive` benchmark_required/high: Proof gate: post-mutation reads never miss visible delta rows and stale resident hits are filtered by snapshot/generation boundary. - Add a route descriptor field for expected probes per key and observed probes per batch.
- `2026-06-04-cross-paper-synthesis-route-descriptors-should-carry-value-visibility-and-scheduling-intent` benchmark_required/high: Compare per-owner FIFO, JSQ, adaptive `JBSQ(n)`, priority classes, and deadline-aware rejection. - Couple resident-object directory metadata with the route descriptor and test range/generation invalidation against table-wide invalidation. - Add a resource-value ledger that can be queried by the scheduler: saved microseconds, rebuild cost, bytes, copy coun...
- `2026-06-04-delilah-exposes-the-real-cost-of-programmable-storage-offload` benchmark_required/high: It does not measure SQL operators, concurrency, throughput, MVCC semantics, recovery, GPU handoff, or multi-tenant admission.

### `benchmark-dependency_witnesses`

- mechanism: `dependency_witnesses` (Dependency witnesses)
- layer: metadata publication
- decision: prototype - Async publication needs explicit dependency proof, but the witness representation should be prototyped before fixing hot-path shape.
- gate: prototype_gate - must produce prototype evidence before the mechanism is adopted
- experiment: Async body-persist simulator that proves readers see only fenced roots while measuring ordered flush reduction.
- evidence links: 122
- benchmark-required links: 11
- relation counts: alternative_to=15, benchmark_required=11, only_valid_if=12, supports=80, warns_against=4

Evidence examples:
- `2026-06-03-taurus-lightweight-parallel-logging` benchmark_required/high: Measure rows/sec, commit wait, fsync bytes, recovery time, and dependency-vector bytes per transaction. - Build a recovery topological-replay proof over small synthetic log streams.
- `2026-06-04-star-phase-switches-ownership-instead-of-paying-distributed-commit-on-every-transaction` benchmark_required/high: Measure throughput, p50/p99 latency, aborted/redeferred work, and fence overhead. - Add recovery proof for phase-fenced metadata: replay WAL/checkpoint state, ignore uncommitted current-fence cache updates, and reconstruct the same route metadata generation before any GPU cache is trusted.
- `2026-06-06-poplar-relaxes-wal-order-to-the-dependencies-recovery-actually-needs` benchmark_required/high: The proof gate is crash replay plus route invalidation reproducing the same visible snapshot. - Add recovery replay benchmarks with independent WAL streams and WAW-conflicting records to measure whether parallel loading and dependency-ordered apply shorten restart time without changing final state. - For OCC write paths, compare centralized timestamp allo...
- `2026-06-07-ccfs-makes-durability-ordering-a-per-stream-contract` benchmark_required/high: In the paper's false-dependency microbenchmarks, a single globally ordered stream can turn a small `fsync` into a 100 MB flush, while separate streams behave like ext4.
- `2026-06-07-decentsched-makes-deterministic-hot-writes-self-schedule` benchmark_required/high: Measure false-positive waits, cache misses, dependency-search time, and metadata footprint. - Test GPU batch pre-ordering for known key updates: CPU generates schedule witnesses and version slots, GPU computes updates, CPU publishes WAL/MVCC.

### `benchmark-cost_based_route_optimizer`

- mechanism: `cost_based_route_optimizer` (Cost-based route optimizer)
- layer: query optimization
- decision: prototype - The system needs deterministic route choice over CPU, GPU, retained, refresh, and cold paths before learning or advanced placement can matter.
- gate: prototype_gate - must produce prototype evidence before the mechanism is adopted
- experiment: Route-choice benchmark over point, range, join, retained, and cold-tier queries.
- evidence links: 254
- benchmark-required links: 23
- relation counts: alternative_to=10, benchmark_required=23, only_valid_if=14, supports=206, warns_against=1

Evidence examples:
- `2026-06-02-data-path-fusion-in-gpu-for-analytical-query-processing` benchmark_required/high: Minimum gate: stable pruning metadata tied to source WAL and schema generation. - For future `text` retained routes, test an FSST/RID-index page layout against the current offsets-plus-bytes resident representation for prefix predicates.
- `2026-06-02-parqo-penalty-aware-robust-plan-selection` benchmark_required/high: A resident GPU route may be fastest when cardinality, residency, queue delay, and transfer estimates are right, but it can be a bad choice when a predicate is less selective than expected, a partition is not resident, the GPU queue is saturated, a refresh is pending, or the CPU fallback path would avoid transfer and launch overhead.
- `2026-06-03-learned-cost-models-need-optimizer-task-proof` benchmark_required/high: It evaluates seven learned cost models against PostgreSQL cost models on three optimizer tasks: join ordering, access path selection, and physical operator selection.
- `2026-06-03-lero-learning-to-rank-query-optimization` benchmark_required/high: This is intended to uncover join-order and physical-operator choices hidden by cardinality error. - Candidate generation is prioritized near the native optimizer's choice and has bounded growth, reported as at most `O(q * log_alpha Delta)` candidates for `q` tables under the paper's heuristic. - The authors evaluate on PostgreSQL 13.1 with JOB/IMDB, STATS...
- `2026-06-03-rtcudb-ray-tracing-core-query-execution` benchmark_required/high: Gate: same SQL answers and better bytes-read per result for selective predicates. - Add a low-cardinality atomic-contention benchmark: one group, 8 groups, 1k groups, and high cardinality.

### `benchmark-htap_freshness_router`

- mechanism: `htap_freshness_router` (HTAP freshness router)
- layer: routing
- decision: prototype - Freshness-aware routing is fundamental to retained reads, but exact wait, refresh, and fallback thresholds need prototype feedback.
- gate: prototype_gate - must produce prototype evidence before the mechanism is adopted
- experiment: Freshness router benchmark over read-committed, bounded-staleness, and exact-snapshot routes.
- evidence links: 87
- benchmark-required links: 15
- relation counts: alternative_to=7, benchmark_required=15, only_valid_if=4, supports=59, warns_against=2

Evidence examples:
- `2026-06-03-ankerdb-fine-granular-virtual-snapshotting` benchmark_required/high: This fits P8's first slice of `int4` and `text` column groups and gives a concrete benchmark for refresh cost versus route freshness.
- `2026-06-03-natto-distributed-transaction-prioritization` benchmark_required/high: Natto has only two priority levels in the evaluated prototype, while GPU DB likely needs classes such as commit acknowledgement, short retained read, COPY admission, refresh, cold scan, analytical read, and maintenance.
- `2026-06-04-cockroachdb-makes-transaction-routing-an-ownership-problem` benchmark_required/high: Expected result: older snapshots reduce owner contention and invalidation races but can raise staleness or fallback rates for freshness-sensitive queries. - Benchmark manual/policy placement versus adaptive movement for resident table ownership.
- `2026-06-04-cross-paper-synthesis-freshness-safe-routes-and-short-commit-windows` benchmark_required/high: HyBench makes freshness a benchmark dimension, F1 Lightning turns freshness into a safe timestamp window for analytical routing, and D2PC splits transaction ordering from final durable completion to shorten the time scarce concurrency-control resources are held.
- `2026-06-04-hybench-frames-htap-as-freshness-bound-mixed-pressure-not-olap-plus-oltp-in-isolation` benchmark_required/high: The paper builds a benchmark around an online finance workload where transactional writes and analytical reads share one database, then scores the system with freshness-aware metrics.

### `benchmark-effective_session_counting`

- mechanism: `effective_session_counting` (Effective session counting)
- layer: runtime admission
- decision: prototype - Large logical-session counts are a target workload, but active-flow accounting needs a simulator before adoption as a fixed runtime rule.
- gate: prototype_gate - must produce prototype evidence before the mechanism is adopted
- experiment: Fan-in activation simulator comparing logical sessions, active flows, and hot-path memory.
- evidence links: 173
- benchmark-required links: 27
- relation counts: alternative_to=2, benchmark_required=27, only_valid_if=9, supports=130, warns_against=5

Evidence examples:
- `2026-06-03-polaris-priority-aware-optimistic-concurrency-control` benchmark_required/high: The paper does not evaluate durable WAL flush cost, checkpointing, recovery replay, GPU execution, PostgreSQL protocol state, distributed clocks, or million-session admission.
- `2026-06-03-powertcp-power-based-congestion-control` benchmark_required/high: Gate: overload reports name the boundary that dominated the decision. - Test a synthetic session-incast: many logical sessions become active in a short window and issue same-shape retained reads.
- `2026-06-03-ringleader-offloads-intra-server-orchestration-to-nics` benchmark_required/high: Gate: lower p99 lookup latency without starving scans or writes. - Extend pgwire retained-read benchmarks with explicit logical session count versus runnable request count.
- `2026-06-03-scalerpc-reliable-connection-resource-sharing` benchmark_required/high: Measure p50/p99 latency, throughput, fairness, queue wait, and starvation under mixed hot/idle clients. - Add telemetry that separates logical sessions, active admitted sessions, active request slots, response slots, pinned staging slots, and owner-queue occupancy.
- `2026-06-03-tas-tcp-acceleration-as-an-os-service` benchmark_required/high: The evaluated 64K connection scale is useful but still far below the 1M logical-session target, and sockets compatibility still requires application relinking in the prototype.

### `benchmark-deficit_fairness`

- mechanism: `deficit_fairness` (Deficit fairness)
- layer: runtime scheduling
- decision: prototype - Fairness counters are needed to bound batching and DAG scheduling bias, but policy constants should come from mixed-class measurements.
- gate: prototype_gate - must produce prototype evidence before the mechanism is adopted
- experiment: Mixed class retained-read benchmark with throughput, p99, and deficit bound gates.
- evidence links: 65
- benchmark-required links: 6
- relation counts: alternative_to=2, benchmark_required=6, only_valid_if=5, supports=51, warns_against=1

Evidence examples:
- `2026-06-03-carousel-time-indexed-shaping-for-bounded-session-admission` benchmark_required/high: GPU DB must test whether time-slot granularity and horizon choices harm p50 query latency or create unfairness between tiny point lookups and large result sets.
- `2026-06-03-scalerpc-reliable-connection-resource-sharing` benchmark_required/high: Measure p50/p99 latency, throughput, fairness, queue wait, and starvation under mixed hot/idle clients. - Add telemetry that separates logical sessions, active admitted sessions, active request slots, response slots, pinned staging slots, and owner-queue occupancy.
- `2026-06-04-homa-makes-receiver-admission-a-latency-control-surface` benchmark_required/high: The evaluated implementation did not measure incoming message-size distributions online; benchmark priority thresholds were precomputed. - In the RAMCloud implementation on 10 Gbps Ethernet, Homa reports 99th-percentile round-trip latency below 15 microseconds for short messages at 80% network load. - In the same setup, the paper reports that Homa's 99th-...
- `2026-06-03-cross-paper-synthesis-logical-scale-needs-active-resource-budgets` benchmark_required/medium: - Track logical versus active counts for sessions, snapshots, descriptors, buffers, and queue entries in every runtime report. - Add mixed read/write/cold-miss workloads where read-priority lanes and virtualized request slots are both stressed, proving that p99 read latency improves without unbounded dirty backlog. - Treat buffer ownership as a correctnes...
- `2026-06-04-cross-paper-synthesis-warm-state-should-be-bounded-semantic-and-visible` benchmark_required/low: Benchmark priority should shift toward cross-route accounting: measure one mixed workload where a hot retained read, a cold point lookup, a repeated miss, a prefix scan, and a slow client all compete for owner pools.

### `benchmark-owner_ring_bundling`

- mechanism: `owner_ring_bundling` (Owner-ring bundling)
- layer: runtime scheduling
- decision: prototype - Owner-local bounded drains are promising for low-allocation scheduling, with selection policy and skip telemetry still needing proof.
- gate: prototype_gate - must produce prototype evidence before the mechanism is adopted
- experiment: Owner-drain benchmark comparing FIFO, bundle packing, and troublesome-first selection.
- evidence links: 452
- benchmark-required links: 62
- relation counts: alternative_to=13, benchmark_required=62, only_valid_if=22, supports=348, warns_against=7

Evidence examples:
- `2026-06-03-aria-deterministic-oltp-batches` benchmark_required/high: When conflict abort rate crosses a threshold, route the hot partition/key range to deterministic owner ordering and compare p50/p99, abort count, queue wait, and throughput against optimistic retry. - Prototype resident-refresh reservation facts: source WAL boundary, partition ids, selected column families, companion columns, and invalidation generation.
- `2026-06-03-chiller-contention-centric-transaction-partitioning` benchmark_required/high: Minimum gate: no behavior change and bounded cardinality for hot telemetry. - Build a TPC-C-style hot-record benchmark comparing ordinary partition-owner mutation order against "hot reservation last." Measure committed rows/sec, p50/p99 latency, owner queue wait, WAL wait, invalidation delay, and retry/abort rate. - Prototype a small hot-key routing table...
- `2026-06-03-cross-paper-synthesis-fast-devices-require-explicit-service-ownership` benchmark_required/high: Treat any unmeasured segment as unknown, not free. - Build service-owned buffer pools for request, response, pinned host staging, and cold-tier cache lines with explicit saturation counters. - Prove IO-worker multiplexing and response-ring backpressure under thousands of logical sessions before attempting DPDK/RDMA. - For P8 over-resident work, test async...
- `2026-06-03-gmt-gpu-orchestrated-memory-tiering-for-the-big-data-era` benchmark_required/high: Measure H2D bytes, NVMe bytes, host-memory occupancy, p50/p99 latency, and write-path interference. - Evaluate whether GPU execution owners should request host-tier promotions directly or ask a residency owner through a bounded ring.
- `2026-06-03-morty-transaction-re-execution` benchmark_required/high: Re-run only the fragment whose read missed a newer write, preserve WAL-before-visibility, and compare against full abort/retry at concurrency `1,2,4,8,16,32,64`. - Add a hot-key policy switch benchmark: optimistic retry, partial re-execution, and deterministic owner-queue ordering.

### `benchmark-stable_handle_indirection`

- mechanism: `stable_handle_indirection` (Stable handle indirection)
- layer: storage layout
- decision: prototype - Tier movement and compaction need stable identity, but lookup overhead must be measured before the handle shape is fixed.
- gate: prototype_gate - must produce prototype evidence before the mechanism is adopted
- experiment: Handle-table lookup and movement benchmark across HBM, DRAM, and NVMe-resident fragments.
- evidence links: 39
- benchmark-required links: 4
- relation counts: alternative_to=3, benchmark_required=4, only_valid_if=3, supports=28, warns_against=1

Evidence examples:
- `2026-06-02-virtual-memory-assisted-buffer-management-in-tiered-memory` benchmark_required/high: Proof gate: readers only use fully valid generations, while maintenance can continue moving other segments without blocking unrelated valid partitions. - Create a future-tier capability matrix for GPU DB route planning: GPU HBM, pinned host DRAM, ordinary DRAM, System-DRAM CXL, DAX/device memory, NVMe, and remote memory.
- `2026-06-03-towards-buffer-management-with-tiered-main-memory` benchmark_required/high: The accessible abstract frames remote memory as much lower latency than SSD while potentially cheaper or more elastic than local DRAM, then evaluates five indexing designs that place or buffer data in remote memory in different ways.
- `2026-06-06-orcgc-makes-reclamation-bounds-part-of-the-hot-path-contract` benchmark_required/high: GPU DB should benchmark pointer-protection fences, reference-count updates, epoch loads, and owner-local handles on the actual CPU path before putting them in per-row or per-request hot loops.
- `2026-06-06-fptree-persistent-leaves-volatile-routing-and-crash-bounded-index-repair` benchmark_required/medium: Measure reader correctness under forced crashes or injected interruption points. - Compare persistent-pointer-style route identities against raw in-process handles for cached route metadata.
- `2026-06-04-cxl-pooling-is-a-costed-route-not-transparent-memory` only_valid_if/high: **GPU DB mapping:** GPU DB should keep CXL and future remote memory out of correctness-critical hot paths unless measurements prove otherwise.

### `benchmark-db_owned_cold_objects`

- mechanism: `db_owned_cold_objects` (DB-owned cold objects)
- layer: storage placement
- decision: prototype - Cold object manifests should be DB-owned to preserve recovery and routing semantics, with compaction and backup boundaries still to prove.
- gate: prototype_gate - must produce prototype evidence before the mechanism is adopted
- experiment: Cold-object read/write/compaction benchmark with object manifest crash recovery.
- evidence links: 52
- benchmark-required links: 3
- relation counts: alternative_to=2, benchmark_required=3, supports=46, warns_against=1

Evidence examples:
- `2026-06-06-geminifs-makes-gpu-storage-metadata-explicit-enough-for-device-side-io` benchmark_required/high: That is a reasonable benchmark mechanism for DB-owned segment files, but it can become stale if the host file system moves blocks, if files are resized dynamically, or if compaction rewrites cold segments without publishing a new descriptor.
- `2026-06-03-cross-paper-synthesis-placement-and-scheduling-need-request-shaped-metrics` benchmark_required/low: DeToX says cache value should be measured by whether a whole request or transaction critical path is shortened, not whether a single object hit in cache.
- `2026-06-04-cross-paper-synthesis-active-window-certificates-should-choose-the-write-lane` benchmark_required/medium: QueCC makes operation queues and priority lanes explicit; ORTHRUS separates concurrency-control ownership from transaction execution; CXL placement says object families need measured tier classes; Strife says the current batch's conflict graph should decide which writes can run without per-record control.
- `2026-06-05-hybridtier-tracks-both-long-term-heat-and-short-term-momentum-for-cxl-tiering` warns_against/medium: **Risks and mismatches:** HybridTier is application-transparent page tiering, not DBMS-owned object placement.
- `2026-06-03-dbms-owned-large-objects-instead-of-files` supports/high: **Core idea:** The paper argues that large binary objects are often kept outside databases mostly because current DBMS BLOB paths are inefficient and external programs expect file APIs.

### `benchmark-multi_tier_placement`

- mechanism: `multi_tier_placement` (Multi-tier placement policy)
- layer: storage placement
- decision: prototype - HBM/DRAM/NVMe placement is necessary for capacity, but movement costs and hit-rate targets need simulator and prototype data.
- gate: prototype_gate - must produce prototype evidence before the mechanism is adopted
- experiment: Tier-placement simulator measuring HBM hit rate, movement latency, p99, and write amplification.
- evidence links: 505
- benchmark-required links: 72
- relation counts: alternative_to=12, benchmark_required=72, only_valid_if=27, supports=378, warns_against=16

Evidence examples:
- `2026-06-02-virtual-memory-assisted-buffer-management-in-tiered-memory` benchmark_required/high: Proof gate: readers only use fully valid generations, while maintenance can continue moving other segments without blocking unrelated valid partitions. - Create a future-tier capability matrix for GPU DB route planning: GPU HBM, pinned host DRAM, ordinary DRAM, System-DRAM CXL, DAX/device memory, NVMe, and remote memory.
- `2026-06-03-adaptive-multi-tier-buffer-management-for-nvm` benchmark_required/high: **GPU DB mapping:** GPU DB should treat GPU HBM, host DRAM, future CXL/NVM-like memory, and NVMe as a measured tier hierarchy rather than a simple cache ladder.
- `2026-06-03-bonspiel-low-tail-geo-distributed-transactions` benchmark_required/high: It does not evaluate GPU execution, PostgreSQL protocol serving, NVMe tiering, MVCC version storage inside a production SQL engine, or durable WAL on the local storage path GPU DB currently cares about.
- `2026-06-03-cross-paper-synthesis-fast-devices-require-explicit-service-ownership` benchmark_required/high: Treat any unmeasured segment as unknown, not free. - Build service-owned buffer pools for request, response, pinned host staging, and cold-tier cache lines with explicit saturation counters. - Prove IO-worker multiplexing and response-ring backpressure under thousands of logical sessions before attempting DPDK/RDMA. - For P8 over-resident work, test async...
- `2026-06-03-cross-paper-synthesis-fast-routes-need-measurable-boundaries` benchmark_required/high: Minimum fields: visibility generation, placement generation, queue wait, execution time, bytes by tier, fallback reason, and response-release delay. - Build a mixed-frontier benchmark where one axis varies write or refresh pressure and the other varies read shape: resident lookup, hot-overlay lookup, compressed cold scan, and CPU fallback. - Add correctne...

### `benchmark-resource_dag_scheduling`

- mechanism: `resource_dag_scheduling` (Resource-DAG scheduling)
- layer: runtime scheduling
- decision: benchmark_only - DAG scheduling could improve scarce-resource packing, but estimation errors can hurt p99 and must be benchmarked against simpler queues.
- gate: decision_gate - must decide whether the mechanism graduates or stays experimental
- experiment: Route-DAG simulator over refresh, H2D, kernel, D2H, and response fragments.
- evidence links: 100
- benchmark-required links: 8
- relation counts: alternative_to=9, benchmark_required=8, only_valid_if=2, supports=80, warns_against=1

Evidence examples:
- `2026-06-03-ringleader-offloads-intra-server-orchestration-to-nics` benchmark_required/high: Hardware microbenchmarks report request scheduling/dispatch within about 150 ns, end-to-end host ping-pong latency around 6 microseconds, and modest FPGA resource use.
- `2026-06-04-schedule-first-concurrency-turns-hot-key-contention-into-an-admission-problem` benchmark_required/high: - The paper frames transaction scheduling as minimizing makespan: for a finite batch, lower makespan corresponds to higher throughput. - Shortest Makespan First (SMF) greedily appends the transaction whose placement causes the least incremental execution-time increase, using conflict cost rather than only per-transaction priority. - SMF avoids needing ful...
- `2026-06-03-cross-paper-synthesis-logical-scale-needs-active-resource-budgets` benchmark_required/low: Recent reviews have good ingredients from storage resource ownership, scheduling, learned robust routing, and MVCC, but the loop should still look for modern papers that evaluate these signals together under SQL or HTAP workloads.
- `2026-06-04-cross-paper-synthesis-route-certificates-need-live-control-loops` benchmark_required/low: - Route-certificate schema and validator tests before any GPU scheduling policy becomes trusted. - Synthetic co-run scheduler harness with measured compatibility, overload reasons, and latency-budget gates. - Delta-overlay snapshot benchmarks that vary delta cardinality, tombstone count, and resident base size. - Buffer-fragmentation benchmarks for pinned...
- `2026-06-04-cross-paper-synthesis-route-certificates-should-combine-freshness-estimates-and-measured-resourc` benchmark_required/medium: CD-search says the scheduler should also know the measured resource class before co-running GPU work.

### `benchmark-log_structured_warm_tier`

- mechanism: `log_structured_warm_tier` (Log-structured warm tier)
- layer: storage placement
- decision: benchmark_only - A rebuildable warm tier may simplify recovery, but append-map, root-publication, and in-place metadata variants need direct comparison.
- gate: decision_gate - must decide whether the mechanism graduates or stays experimental
- experiment: Warm-tier manifest benchmark comparing append-map rebuild, in-place metadata, and root publication.
- evidence links: 48
- benchmark-required links: 0
- relation counts: alternative_to=3, supports=43, warns_against=2

Evidence examples:
- `2026-06-06-rewind-makes-byte-addressable-durability-a-log-structure-problem` warns_against/high: **Risks and mismatches:** REWIND targets byte-addressable NVM and programmer-managed persistent data structures, not a conventional SQL DBMS, a GPU database, or a multi-tier HBM/DRAM/NVMe engine.
- `2026-06-06-leanstore-recovery-makes-wal-a-sharded-tiered-and-checkpoint-bounded-pipeline` warns_against/medium: **Risks and mismatches:** The design is tied to LeanStore's buffer manager, pointer swizzling, page ids, B+-tree pages, and persistent-memory/NVMe hardware assumptions.
- `2026-06-03-adaptive-multi-tier-buffer-management-for-nvm` supports/high: **Core idea:** The paper argues that once a middle tier such as byte-addressable NVM is close enough to DRAM, a DBMS should stop treating every page miss as "copy to DRAM before doing work." NVM creates more legal data paths: operate on NVM-resident data directly, persist some writes directly to NVM, sometimes skip NVM on SSD reads, and sometimes skip NVM...
- `2026-06-03-agile-lightweight-and-efficient-asynchronous-gpu-ssd-integration` supports/high: GPU threads can issue NVMe requests and continue useful work while a lightweight GPU service handles completion queue polling, resource release, and request progress.
- `2026-06-03-cam-asynchronous-gpu-initiated-cpu-managed-ssd-access` supports/high: Fully GPU-managed paths such as BaM avoid the CPU staging copy and let GPU thread blocks submit NVMe work directly, but their synchronous API can consume many GPU SMs waiting on SSD latency and can serialize storage access with useful GPU computation.

### `benchmark-deterministic_hot_write_templates`

- mechanism: `deterministic_hot_write_templates` (Deterministic hot-write templates)
- layer: transaction execution
- decision: benchmark_only - Hot-write templates may beat abort/retry loops for narrow key shapes, but they should remain benchmark-gated until workload fit is proven.
- gate: decision_gate - must decide whether the mechanism graduates or stays experimental
- experiment: Zipfian hot-key simulator comparing retry, ordered locking, owner serialization, and deterministic queue positions.
- evidence links: 209
- benchmark-required links: 29
- relation counts: alternative_to=11, benchmark_required=29, contradicts=1, only_valid_if=16, supports=147, warns_against=5

Evidence examples:
- `2026-06-03-btrim-hybrid-in-memory-row-store-for-extreme-oltp` benchmark_required/high: For session concurrency, the hot-row lesson is that contention should be measured at the resource where it occurs.
- `2026-06-03-caerus-partial-order-transaction-sequencing` benchmark_required/high: Measure graph size, SCC size, queue delay, abort or throttle rate, and p99 latency under skew. - Compare one global mutation queue against owner-local partial sequences plus deterministic merge for COPY-like batches.
- `2026-06-03-gcctb-gpu-oltp-concurrency-control-study` benchmark_required/high: For high-contention hot rows, the GaccO result suggests a benchmarkable alternative: build a per-batch access table for known write templates and execute a deterministic conflict order on GPU, or at least use the access table to route hot keys to partition owners with bounded admission.
- `2026-06-03-pwv-early-write-visibility` benchmark_required/high: PWV suggests a narrower benchmark track: inside a deterministic owner-local batch, classify statements into abortable validation pieces and non-abortable apply/index/refresh pieces, then see whether later pieces can unblock dependent reads or writes sooner without exposing unlogged or rollbackable state.
- `2026-06-03-rebirth-retire-adaptive-contention-control` benchmark_required/high: Compare first-writer-wins, abort/retry, deterministic batch order, and a bounded rebirth-style priority-demotion prototype.

### `benchmark-gpu_oltp_conflict_ordering`

- mechanism: `gpu_oltp_conflict_ordering` (GPU OLTP conflict ordering)
- layer: transaction execution
- decision: benchmark_only - GPU conflict preprocessing is useful only for specific hot-key batch regimes and must compete with CPU OCC and owner serialization.
- gate: decision_gate - must decide whether the mechanism graduates or stays experimental
- experiment: YCSB/TPC-C style hot-key batch benchmark comparing CPU OCC, owner serialization, and GPU conflict ordering.
- evidence links: 129
- benchmark-required links: 18
- relation counts: alternative_to=1, benchmark_required=18, contradicts=2, only_valid_if=4, supports=101, warns_against=3

Evidence examples:
- `2026-06-03-polaris-priority-aware-optimistic-concurrency-control` benchmark_required/high: Promote priority after configurable thresholds, then measure throughput, p50, p99, p999, abort count distribution, and starvation under YCSB-like hot keys and TPC-C-like new-order/payment mixes. - Prototype a reservation sidecar for CPU canonical row ids or hot index keys: version/generation remains the correctness guard, while priority and reservation ge...
- `2026-06-03-polyjuice-learned-concurrency-control-policies` benchmark_required/high: The paper reports that this was enough for TPC-C, TPC-E, and a synthetic 10-transaction workload; adding data contention level helped only contrived microbenchmarks. - Wait actions are not absolute sleeps.
- `2026-06-04-acc-chooses-concurrency-control-per-cluster-instead-of-globally` benchmark_required/high: The paper reports preliminary prototype results on YCSB, Smallbank, and TPC-C where ACC tracks the best protocol as workloads shift between low-conflict non-partitionable, well-partitionable, and highly conflicted non-partitionable phases.
- `2026-06-04-epic-deterministic-mvcc-removes-version-search-from-gpu-oltp-batches` benchmark_required/high: Execution may run on the GPU when the dataset fits device memory, or on the CPU while the GPU still accelerates indexing and initialization for larger datasets. - The paper evaluates with TPC-C and YCSB against CPU and deterministic baselines.
- `2026-06-04-gpu-tps-maps-oltp-writes-onto-simt-with-grouping-locks-and-gpu-indexes` benchmark_required/high: The DOI page reports that GPU-TPS evaluates SmallBank and TPC-C, outperforming a hardware-transactional-memory CPU OLTP baseline by 3.8x on SmallBank and 1.9x on TPC-C, and outperforming the older GPUTx GPU OLTP baseline by 1.6x and 1.8x respectively.

### `benchmark-learned_optimizer_advisor`

- mechanism: `learned_optimizer_advisor` (Learned optimizer advisor)
- layer: query optimization
- decision: defer - Learned route advice should wait until deterministic route telemetry, guardrails, and baseline costs are stable.
- gate: deferred_gate - recorded for later once prerequisite architecture telemetry exists
- experiment: Advisor-vs-baseline route selection with guardrail rejections and p99/regret tracking.
- evidence links: 72
- benchmark-required links: 6
- relation counts: alternative_to=7, benchmark_required=6, only_valid_if=4, supports=52, warns_against=3

Evidence examples:
- `2026-06-03-kepler-robust-parametric-query-optimization` benchmark_required/high: Instead of trying to replace a whole optimizer, it generates a bounded candidate plan set per template, executes those candidates offline or on isolated training instances, and trains a small model to choose the fastest plan for new parameter bindings.
- `2026-06-03-lero-learning-to-rank-query-optimization` benchmark_required/high: They report stable-model execution-time reductions versus PostgreSQL of 70%, 44%, 21%, and 13% on those benchmark families, respectively, with lower regression frequency than Bao/Bao+ on STATS. - The evaluation includes dynamic data insertion on STATS and finds the relative ordering labels easier to adapt than exact latency labels. - The paper explicitly...
- `2026-06-06-lsched-makes-query-scheduling-a-physical-plan-and-pressure-problem` benchmark_required/high: Expected value: learned or adaptive scheduling can be evaluated without first trusting it in the serving path. - Extend micro-batch compatibility checks with pipeline-breaking facts, such as decompression format, join build/probe boundary, aggregate state size, and response encoding.
- `2026-06-04-revisiting-gpu-db-query-performance-and-resource-allocation` benchmark_required/medium: The evaluated systems are mostly GPU analytical engines; only PG-Strom represents CPU/GPU co-execution, and it is not optimized for this engine's retained snapshot model.
- `2026-06-07-cross-paper-synthesis-learned-route-control-needs-deterministic-envelopes` benchmark_required/medium: - First, build deterministic route-DAG telemetry and policy-snapshot publication before training any scheduler. - Second, measure fixed heuristic, learned ranking, and learned batch-limit variants under continuous mixed arrivals. - Third, require advisor timeout, validation restore, and native fallback in every learned-route benchmark. - Fourth, report ro...
