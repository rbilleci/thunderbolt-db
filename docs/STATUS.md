# STATUS — Current Implementation Facts

This document records what exists and what has been verified. It does not own tasks or sequencing. Every live
gap points to a stable ID in [`PLAN.md`](PLAN.md).

## Product direction and execution boundary

- The engine requires an NVIDIA GPU. The host is control plane: wire I/O, SQL parse/plan, transaction
  sequencing, WAL/durability, replication, staging upload, and final result readback.
- Production relational reads execute through resident, streaming, transient-relation, or CUDA-native MVCC
  GPU paths. A decline or device fault fails loudly; it never executes relational work on the host.
- Test-only CPU semantic infrastructure remains under `cfg(test)` pending **RETIRE-001**.
- The host write/commit/MVCC tuple-store path, host DML indexes/probes, generic CUDA-MVCC host result
  post-processing, and recovery repair operators remain pending **R3-001**, **R3-002**, **R3-003**,
  **R3-004**, **RETIRE-002**, and **RETIRE-003**.

## Read path and STRATA

- Commit-triggered GPU admission is production-default. Recovery suppresses per-record admission, replays the
  durable state, then bulk-admits at the settled boundary.
- Purely-int4 relations default to immutable/versioned GPU shards. Fixed-width, text, bool, NULL, identity, and
  visibility regions have generation-consistent publication.
- Working sets above the resident budget use byte-bounded GPU streaming over host/NVMe cold storage. Supported
  folds include scalar, projection, grouped/distinct, ordered/top-N, joins/outer joins, and rank/window routes.
- Chunk-authoritative and keyed chunk-authoritative relations support delta maintenance, spill/LRU, exact/Bloom
  skipping, device predicate recheck, recovery artifacts, and bounded index/Bloom accounting.
- Published residency contains device descriptors/resources only. Decoded admission rows are discarded after
  upload; no relational host-row shadow or append mirror remains.
- Catalog, information-schema, materialized-view, and bounded SQL-function results use transient GPU relations.

## SQL and execution surface

- General GPU execution covers int2/int4/int8, numeric, text, date, timestamp, UUID, bool, NULL/3VL, arithmetic,
  comparison/boolean predicates, five scalar aggregates, GROUP BY, HAVING, DISTINCT, ORDER BY, LIMIT/OFFSET,
  joins, and supported windows.
- PostgreSQL-facing support includes bounded tables, indexes/constraints/defaults, views/materialized views,
  sequences, domains, roles/ACL metadata, literal SQL functions, extended-protocol operations, COPY, and
  dump/restore surfaces. Broader type/protocol breadth is **PRODUCT-002**.
- The facade and pgwire server expose the engine, including the production point-lookup batcher. Server
  consolidation remains **PRODUCT-001**.

## Write path, durability, and recovery

- WAL-before-visibility is enforced. The durable path includes append-only/checkpointed WAL, FUA intent lanes,
  contiguous durable cuts, lane recovery/recycle, group commit, and fused device apply for eligible shapes.
- Covered int4-PK INSERT/UPDATE/DELETE intent paths and mixed GPU read/write execution are live. Wider write
  shapes and the final GPU-native write/MVCC model remain **R3-001**, **R3-002**, and **R3-003**.
- Crash-durable replay exists; the broader fault campaign, automatic lane checkpointing, PITR timestamps, and
  multi-node quorum integration are **DUR-001**, **DUR-002**, and **HA-001**.

## Verification snapshot — 2026-07-12

- Engine library: **505 passed, 0 failed, 485 GPU-ignored**.
- Production transient GPU integrations: catalog and bounded-function routes pass.
- Pgwire: ordinary suite **3 passed** plus the ignored non-vacuous sharded/NULL GPU golden passes.
- Production mixed gate: **116.2k reads/s**, p50 **246us**, p99 **501us**, p99.9 **671us**; zero host gathers,
  zero fallback groups, and 160/160 host-install-elided writes.
- Read roofline: in-L2 `count_i32_compare` approximately **0.91x** the same-run `sum_i32` roofline; grouped
  kernel approximately **1,678 M elements/s**.
- Canonical report card: 48M-row out-of-L2 batched route **252.4M lookups/s at batch 65,536, p50 131us**;
  indexed single-flight route **3.23x** the scan.
- Production release check, engine/facade examples, static host-row-removal guard, and diff whitespace check pass.

## Structural decomposition

- **STRUCT-001** began with the execution facade. Device-routing policy now lives in `execution::routing`, and
  the RETIRE-001-owned host-reference iterator operators live in `execution::reference_operators`; stable
  crate-root re-exports preserve downstream APIs. The MVCC device-transfer layout and validation contract now
  lives in `execution::mvcc_batch`, with its encoding tests. These pure-move slices passed all 24 non-ignored
  execution tests and downstream engine/planner/metrics/observability checks. CUDA runtime/device snapshots,
  device-memory proof, and the typed CUDA error taxonomy now live in `execution::runtime_contract`; exact
  client-visible error text has a focused regression. The resulting 25 non-ignored execution tests pass.
  The former 7,177-line inline execution tests now live under `execution/src/tests/`: full test names and the
  94-test inventory are byte-for-byte stable, and every resulting test file is below 3,000 lines. The execution
  shared CUDA primary context, module/stream caches, pooled buffers, allocation budgets, pending-copy transport,
  and CUDA RAII guards now live in the 948-line `execution::cuda_context` module with parent-private visibility.
  Real-GPU context/pool/count and NULL-validity gates passed 3× sequential and 2× concurrent with zero CUDA
  safety errors; an independent audit found no ownership/drop-order regression and its visibility findings were
  adopted. The 883-line `execution::cuda_driver` module now owns runtime probing/snapshots, primitive dispatch,
  resident allocation construction, async-copy fallback, raw HtoD/DtoD builders, and `GpuRuntime` routing. Its
  real-GPU driver/smoke/allocation/NULL matrix passed 3× sequential and 2× concurrent; independent audit found
  proof, fencing, guard, and recompaction behavior unchanged. The 405-line `execution::resident_memory` module
  now owns resident allocation/read-view/chunk contracts, bounded copies, append, telemetry, the read-source
  trait, Debug, and context-bound Drop. Its append/generation/submit-complete/NULL GPU matrix passed 3×
  sequential and 2× concurrent; independent audit found bounds, Arc lifetime, and field-drop behavior unchanged.
  The 441-line `execution::point_read_submission` module now owns batched projection result contracts, owned
  pooled guards, detached completion, pinned final readback, error drains, and drop-without-complete fencing. Its
  submit/drop/pool/NULL matrix passed 3× sequential and 2× concurrent; independent audit found the single-sync
  and field-drop behavior unchanged and its documentation-placement finding was adopted. The 1,308-line
  `execution::point_read_submit` module now owns equal-project launch and equal-any scan/unique-index atomic
  submission/PTX. Its all-target/downstream checks, exact 94-test inventory, 25 non-ignored tests, and NULL,
  scan, index, and drop HAZARD matrix passed 3× sequential and 2× concurrent with zero CUDA 700/716/717 errors;
  independent byte-level audit found PTX, launch parameters, bounds, fallbacks, guards, fences, and index pinning
  unchanged. The canonical report card stayed green in both layers and cache regimes: in-L2/out-of-L2 `sum_i32`
  measured 1,471/1,453 GB/s, `count_i32_compare` measured 0.89x/1.00x roofline, and 65,536-batch point reads
  measured 246.1M/253.3M lookups/s at p50 138/132us. The 1,086-line `execution::point_read_dense` module now
  owns the dense unique-index result/lifecycle contract, single- and multi-shard submit/PTX, and shard descriptor.
  Device hash probing, MVCC birth/death and row bounds, zone pruning, binary/linear routing, root APIs, guards,
  events, fallback/drain behavior, and drop fencing remain unchanged. The exact inventory is now 95 tests due to
  one intentional dense drop-without-complete regression; 25 non-ignored tests pass. Five dense/MVCC/binary/NULL/
  drop GPU gates passed 3× sequential and 2× concurrent with zero CUDA 700/716/717 errors, and independent audit
  found no issue at any severity. Its canonical report card stayed green: in-L2/out-of-L2 `sum_i32` measured
  1,483/1,443 GB/s, `count_i32_compare` measured 0.86x/1.00x roofline, and 65,536-batch point reads measured
  247.1M/252.3M lookups/s at p50 139/132us. The 206-line `execution::point_read_bloom` module now owns chunk-
  Bloom device pruning. Two-hash/three-probe bit addressing, descriptor/output layout, bounds, pooled allocations,
  resident guards, synchronous launch/readback, and the root API remain unchanged. The exact 95-test inventory,
  25 non-ignored tests, and Bloom false-negative/NULL GPU gates passed 3× sequential and 2× concurrent with zero
  CUDA 700/716/717 errors; independent audit found no remaining issue after its formatting finding was adopted.
  Its canonical report card stayed green: in-L2/out-of-L2 `sum_i32` measured 1,479/1,444 GB/s,
  `count_i32_compare` measured 0.89x/1.00x roofline, and 65,536-batch point reads measured 246.3M/253.5M
  lookups/s at p50 139/131us. The 901-line `execution::point_read_text` module now owns equal-any int4-filter/
  text projection and bounded row assembly. PTX, compaction, bounds, pooled buffers/streams, async pinned and
  blocking readback, event timing, ordering, UTF-8 validation, and root APIs remain unchanged. The exact 95-test
  inventory, 25 non-ignored tests, and text multi-warp/NULL/pinned-pool GPU gates passed 3× sequential and 2×
  concurrent with zero CUDA 700/716/717 errors; independent audit found no extraction issue. Its canonical card
  stayed green: in-L2/out-of-L2 `sum_i32` measured 1,485/1,443 GB/s, `count_i32_compare` measured 0.88x/1.01x
  roofline, and 65,536-batch point reads measured 248.2M/252.7M lookups/s at p50 138/132us. The audit confirmed a
  pre-existing TEXT-only zero-int4-projection framing panic; the fact is owned here and its immediate fix is
  now complete: match-count-indexed framing supports zero numeric projections and validates every result vector;
  nullable int4 filter validity is passed through both wrappers and benchmark handles and checked in PTX before
  comparing the raw placeholder. A retained GPU differential proves present/absent, ordering, `''`, real `0`, and
  NULL-key exclusion with no CPU fallback. The exact inventory is 99 tests (four intentional framing additions),
  29 non-ignored tests pass, and TEXT-only/filter-NULL/pinned-pool gates pass 3× sequential and 2× concurrent with
  zero CUDA 700/716/717 errors. Independent audit passed after its correctness, exact-count, and test non-vacuity
  findings were adopted. The final card stayed green: `count_i32_compare` measured 0.96x/1.01x same-run roofline
  in-L2/out-of-L2, while 65,536-batch point reads measured 247.3M/252.2M lookups/s at p50 139/132us. The focused
  GPU regression is isolated in a focused test module. Projected nullable text validity now travels from the
  resident bitmap through PTX compaction and the execution result contract; NULL text receives a distinct status
  byte and both engine/example consumers map it to `SqlValue::Null`, while `''` remains a valid empty string. The
  retained GPU differential covers NULL, empty, and nonempty text on both ordinary and retained-job routes with
  no host fallback. The exact execution inventory is 101 tests, 31 non-ignored tests pass, and the engine
  inventory is 992 tests. Nullable-text/TEXT-only/pinned-pool HAZARD gates passed 3× sequential and 2× concurrent
  with zero CUDA 700/716/717 errors; independent audit found no remaining issue after two stale comment counts
  were removed. The final card stayed green: `count_i32_compare` measured 0.91x/1.01x same-run roofline in-L2/
  out-of-L2, while 65,536-batch point reads measured 245.4M/253.9M lookups/s at p50 139/131us. The focused GPU
  regression module is 122 lines. The resident int4 equality row selector now lives in the private 475-line
  `execution::point_read_rows` module. The move is byte-identical apart from parent-private visibility: PTX,
  validation, 128-thread grid, device-buffer/stream pooling, async pinned and blocking readback, timing, error
  draining, final deterministic ordering, and the public method remain unchanged. The adjacent CPU range selector
  remains in the root and the equality selector's host ordering remains **RETIRE-003** debt. The exact 101-test
  inventory and 31 passing non-ignored tests are unchanged; equality/multi-warp/pinned-pool gates passed 3×
  sequential and 2× concurrent with no CUDA safety errors. Independent audit and all target/feature checks passed.
  The final card stayed green: `count_i32_compare` measured 0.97x/1.02x roofline in-L2/out-of-L2 and 65,536-batch
  point reads measured 247.5M/252.9M lookups/s at p50 140/132us. The five GPU ORDER BY launchers now live in
  the private 846-line `execution::resident_sort` module. Their normalized source is byte-identical: public
  wrappers, PTX paths/symbols/arguments/grids, device and uploaded-payload lifetimes, stream synchronization,
  NULL/direction/text/multikey semantics, and padding filtering are unchanged. The exact 101-test inventory and
  31 passing non-ignored tests remain stable. Six i64/radix/multikey/heterogeneous/dedicated-text gates passed 3×
  sequential and 2× concurrent (18/24 invocations) without CUDA safety errors; independent audit's missing
  dedicated-text evidence finding was adopted. The final card kept bitonic sort at 346.6 Melem/s, count at
  0.91x/1.00x same-run roofline, and 65,536-batch point reads at 247.7M/251.5M lookups/s with p50 137/132us.
  The nullable int4 equality/comparison reductions and both test-only GPU serial parity kernels now live in the
  private 1,066-line `execution::resident_count` module. Their normalized source is byte-identical: bitmap bounds
  and NULL semantics, PTX/symbols/arguments/grids, serial allocation/module guards, parallel cached-module and
  pooled-stream reduction, public methods, test access, and parent-private bitmap validation for later scalar
  stats are unchanged. The exact 101-test inventory and 31 passing non-ignored tests remain stable. Five equal/
  compare/NULL/concurrency/later-stats gates passed 3× sequential and 2× concurrent (15/20 invocations) without
  CUDA safety errors. Independent audit's two low-severity extraction-seam blank findings were adopted. The final
  card kept count at 0.86x/1.02x same-run roofline and 65,536-batch point reads at 247.3M/253.1M lookups/s with
  p50 140/131us. The shared fixed-width gather lifecycle and int4/bool/int8/i128 projection wrappers now live
  in the private 258-line `execution::resident_gather` module. Their normalized source is byte-identical: bounds
  precede every device access; PTX entries/arguments/grids, row-index H2D, pooled stream/scratch, synchronization,
  one bounded result D2H, bool bitmap addressing, i128 endian layout, and public methods are unchanged. The exact
  101-test inventory and 31 passing non-ignored tests remain stable. Five width/bool/pool gates passed 3×
  sequential and 2× concurrent (15/20 invocations) without CUDA safety errors. Independent audit's low-severity
  seam and stale public bool-documentation findings were adopted; the public docs now describe the actual one-
  kernel/one-readback path. The final card kept gather at 349.5/155.3 GB/s in-L2/out-of-L2, count at 0.85x/1.01x
  roofline, and 65,536-batch point reads at 246.1M/251.1M lookups/s with p50 140/131us. The int8/i128/text/UUID/
  bool predicate launchers and
  nullable-mask compactor now live in the private 1,064-line `execution::resident_filter` module. Their normalized
  source is byte-identical: PTX, scalar limbs/signedness, text/UUID ordering, LIKE token lifetime, bool bitmap
  expansion, validity AND, device compaction, bounded index readback, and public methods are unchanged. The exact
  101-test inventory and 31 passing non-ignored tests remain stable. Seven scalar/column/text/LIKE/nullable-UUID/
  bool gates passed 3× sequential and 2× concurrent (21/28 invocations) without CUDA safety errors. The final card
  kept i64/i128 1%-selectivity filters at 165.6/286.0 GB/s in-L2 and 363.7/594.6 GB/s out-of-L2, count at
  0.87x/1.00x roofline, and point reads at 248.0M/250.9M lookups/s with p50 138/131us. Independent audit found a
  pre-existing safe-API device-window validation defect. It is now fixed: every fixed-width/bitmap/text descriptor
  is checked with overflow-safe arithmetic before CUDA mutation; text APIs carry the exact resident byte extent,
  and equality/order/LIKE PTX fails malformed spans closed before any byte load. Engine callers pass the descriptor
  extent. Three pure validation tests and two GPU safety regressions raise the exact inventory to 106 tests, with
  34 passing and 72 ignored. Nine safety/type/NULL gates passed 3× sequential and 2× concurrent (27/36 final-PTX
  invocations) without CUDA errors. Independent audit's malformed-inequality HIGH and indentation LOW findings
  were adopted. The final card kept i64/i128 filters at 165.6/285.6 GB/s in-L2 and 362.4/591.9 GB/s out-of-L2,
  count at 0.86x/1.00x roofline, and point reads at 247.8M/251.9M lookups/s with p50 138/132us. The execution root
  was 22,608 lines and `resident_filter` is 1,221 lines. The six filtered fixed-width SUM/MIN/MAX and i128
  partial-reduction launchers now live in the private 803-line `execution::resident_aggregate` module. Their
  normalized bodies are byte-identical: the nine public root methods, PTX symbols and arguments, bounded grids,
  index H2D lifetime, pooled synchronization, output initialization and readback, signed i128 low/high partials,
  atomic carry/min/max behavior, checked host combine, and overflow result are unchanged. The exact 106-test
  inventory remains 34 active and 72 ignored. Five aggregate/type/overflow gates passed 15 sequential and 20
  concurrent invocations without CUDA 700/716/717. Independent audit found no extraction regression and exposed
  a pre-existing unchecked resident-window boundary now owned by **STRUCT-001V**. The canonical card measured
  count at 0.88x/1.01x same-run roofline, gather at 349.5/155.3 GB/s, grouped aggregation at 1,678.3 M elements/s,
  and 65,536-batch point reads at 246.2M/232.4M lookups/s with p50 140/141us in-L2/out-of-L2. All nine safe
  filtered aggregate APIs now reject empty inputs and check the maximum host index using overflow-safe 4/8/16-
  byte extent arithmetic before context selection, module lookup, buffer leasing, or device access. The engine
  continues to map empty SQL SUM/AVG/MIN/MAX to typed NULL before this layer; no host relational fallback or
  device-value scan was added. Two pure boundary/overflow tests and one real-GPU malformed-window/reuse test
  raise the exact inventory to 109 tests, with 36 passing and 73 ignored. Six safety/aggregate/NULL/overflow
  gates passed 18 sequential and 24 concurrent invocations without CUDA 700/716/717; independent audit found
  no issue. The final card measured count at 0.93x/1.00x same-run roofline, gather at 349.5/155.3 GB/s, grouped
  aggregation at 1,678.0 M elements/s, and 65,536-batch point reads at 249.4M/255.4M lookups/s with p50 136/127us
  in-L2/out-of-L2. The grouped public result contract, production launcher, and benchmark-only event-timed
  launcher now live in the private 1,207-line `execution::resident_group` module. Their normalized source is
  byte-identical: the four public root methods and `GroupByI32Row` re-export, PTX ABI/grids, all int4/int8/i128/
  UUID/text/composite modes, aggregate masks, NULL slots, one/two-pass behavior, numeric overflow, pooled
  lifetimes, dense result order/readback, and event brackets are unchanged. The exact 109-test inventory and
  36/73 suite remain stable. Nine grouped type/NULL/overflow/two-level gates passed 27 sequential and 36
  concurrent invocations without CUDA 700/716/717. The final card measured grouped aggregation at 1,677.7 M
  elements/s, count at 0.87x/1.00x roofline, and 65,536-batch point reads at 251.1M/257.5M lookups/s with p50
  136/127us in-L2/out-of-L2. Independent audit found no extraction regression and exposed pre-existing unchecked
  grouped fixed/text/derived/composite windows plus timed-launcher lifecycle gaps. Those gaps are now closed:
  every safe grouped entry accepts typed resident/derived/text/composite descriptors with exact logical extents,
  owner lifetimes, and originating CUDA-context identity; raw kernel pointers and flags remain private. Checked
  host preflight precedes context binding and covers indexed fixed windows, text sections, bitmaps, descriptor
  triples, row agreement, initialized derived bytes, mode coherence, and timed-run arithmetic. Direct and
  representative varlen spans are bounded on-device and report through a fail-closed error flag; numeric pass two
  is suppressed after malformed input. Timed events are RAII-owned and error exits drain the default stream before
  event/buffer destruction while preserving empty zero-work and kernel-only timing semantics. Six pure grouped
  boundary/mode tests plus a retained-GPU malformed/reuse regression raise the exact execution inventory to 116,
  with 42 passing and 74 ignored. Ten grouped safety/type/NULL/overflow/two-level gates passed 30 sequential and
  20 concurrent invocations without CUDA 700/716/717; the final focused safety gate additionally passed three
  sequential invocations. Independent audit is clean. The final canonical card measured grouped aggregation at
  1,678.0 M elements/s, count at 0.88x/1.01x same-run roofline, gather at 349.5/155.3 GB/s, and 65,536-batch point
  reads at 247.4M/251.4M lookups/s with p50 138/132us in-L2/out-of-L2. The execution root is 20,589 lines;
  `group_input.rs` is 639 lines and `resident_group.rs` is 1,282 lines. Further extraction remains **STRUCT-001**.
  The four device write/visible-locate contracts, both ASCII PTX programs, and both launchers now live in the
  private 721-line `execution::write_locate` module with stable crate-root re-exports. The PTX byte arrays and
  moved method/documentation bodies are exact matches to their prior root definitions; descriptor packing,
  duplicate advancement, overflow sentinel, unsigned MVCC visibility, output ordering, and GPU-only addressing
  are unchanged. The exact 116-test inventory and 42/74 suite remain stable. Write locate, visible DELETE/UPDATE,
  and sharded duplicate gates passed 12 sequential and 8 concurrent invocations without CUDA 700/716/717.
  Independent audit found no extraction defect and identified a pre-existing safe-boundary gap: cross-context or
  geometrically incoherent indexes, unowned/unbounded version pointers, packed slots beyond version extents, and
  post-launch DtoH errors without a best-effort drain. Those gaps are now owned by **STRUCT-001Z** ahead of further
  decomposition. The execution root is 19,881 lines.
  That safe-boundary hardening is complete. `WriteLocateShard` and `VisibleLocateShard` now carry exact logical
  row extents and owned device allocations; every index/version owner is checked against the submission primary
  context, and table masks, hash shifts, allocated index bytes, version-region capacity, descriptor/output
  arithmetic, and reserved count ranges are validated before descriptor allocation or CUDA work. Cached indexes
  return their exact row extent, and visible locate rebinds a newer in-place index to the matching currently
  published payload/version generation. The 24/40-byte PTX descriptors bound every decoded packed slot before
  output or MVCC loads and propagate per-needle reserved count sentinels without adding an allocation, memset,
  ABI argument, or extra count-only readback. A default-stream RAII drain fences launch and readback failures
  before pooled buffers or owners drop. Duplicate advancement, ordinary overflow, unsigned visibility, output
  order, and GPU-only key addressing remain intact. One pure geometry test and one retained-GPU malformed/reuse
  differential raise the execution inventory to 118 tests, with 43 active and 75 GPU-ignored. Five write-locate,
  safety, visible DELETE/UPDATE, and sharded-duplicate gates passed 15 sequential and 10 concurrent invocations
  without CUDA 700/716/717; workspace all-target/all-feature check, execution clippy, and independent audit are
  clean. `write_locate.rs` is 865 lines and the execution root remains 19,881 lines. The single-GPU host cannot
  make the cross-context test non-vacuous, so per-context partition/merge is explicitly owned by **MULTI-002**.
  The final canonical report card is green in both layers and cache regimes: in-L2/out-of-L2 `sum_i32` measured
  1,475.5/1,453.3 GB/s, `count_i32_compare` measured 0.89x/1.00x same-run roofline, grouped aggregation measured
  1,678.3 M elements/s, and 65,536-batch point reads measured 245.6M/253.1M lookups/s at p50 138/132us.

## Known boundaries

| Boundary | Work ID |
|---|---|
| 29 source files exceed the production/test/tool analysis envelopes in `CODE_SIZE.md` | **STRUCT-001** |
| Open-loop OLTP comparison against tuned PostgreSQL remains incomplete | **BENCH-001** |
| Current write implementation and target MVCC/write design need one accepted reconciliation | **R3-001** |
| Wider-type/compound-key write and read fast-path coverage | **R3-002**, **READ-002** |
| Deterministic CC, transaction-held snapshots, VACUUM/GC | **R3-003** |
| Host write/store deletion | **R3-004** |
| Test-only CPU semantic oracle | **RETIRE-001** |
| Reverse-gather/deauthorization/scan-build DDL and recovery repair | **RETIRE-002** |
| Generic CUDA-MVCC host compaction, ordering, projection, and result assembly | **RETIRE-003** |
| Host `CachedShardPkIndex` and DML/constraint probe fallback | **R3-002**, **R3-004** |
| Persistent GPU catalog plus strict metadata-staging boundary | **PRODUCT-002** |
| Two physical GPUs have not executed the existing scheduler or device-locate context gates | **MULTI-001**, **MULTI-002** |
| Filtered expression-overflow ordering and route-case behavior require current-tree disposition | **READ-001** |
| Lane DELETE residuals and empty-aggregate pgwire NULL seam require focused disposition | **R3-005**, **READ-003** |
| Lanes auto-checkpoint/PITR and full crash campaign | **DUR-001**, **DUR-002** |
| Multi-node Raft/quorum serving is not integrated | **HA-001** |
| Connection/runtime scale and bounded result streaming | **SCALE-001** |
| Historical scalability-ledger findings require current-tree disposition | **SCALE-002** |

Do not add work here. Add or update one row in `PLAN.md`, then reference its ID from this table if the boundary
is an important current fact.
