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
- The legacy compatibility endpoint's opt-in production security profile requires TLS plus a SCRAM-SHA-256
  verifier; plaintext password input is restricted to its explicit local/test credential bootstrap. The
  connection-security preflight exercises valid, invalid-password, recovery, and non-TLS rejection paths.

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
  Fused append publication, incremental int4 index insertion, compound fixed/wide/UUID/text fingerprint folding,
  `FusedApplyRequest`, and their three ASCII PTX programs now live in the private 873-line
  `execution::write_apply` module. The root type re-export and inherent method APIs remain stable; all three PTX
  byte images hash exactly to their prior definitions, and moved launcher bodies differ only by module imports
  and one rustfmt trailing comma. Staging layout, cached symbols, arguments/grids, default-stream blocking
  readbacks, pool/index ownership, partial-failure and 256-probe decline behavior, and compound hash parity are
  unchanged. An obsolete index-insert test/doc expectation was corrected to the established F3/U4 contract:
  same-key MVCC versions advance to distinct slots rather than declining. The exact 118-test inventory and
  43/75 suite remain stable. Seven fused/index/fixed/wide/UUID/text/streaming GPU gates passed 21 sequential and
  14 concurrent invocations without CUDA 700/716/717. Workspace all-target/all-feature check and execution
  clippy are green. The execution root is 19,020 lines. Independent extraction audit was clean and exposed
  pre-existing raw address/context/extent/geometry and launched-error-drain gaps in the safe APIs, closed by
  the following **STRUCT-001AB** hardening without changing the broader **R3-001** design.
  That write-apply hardening is complete. `CudaWriteDestination`, `CudaWriteIndex`, and
  `CudaCompoundFoldColumn` replace arbitrary device addresses; same-context identity, 4/8-byte alignment,
  allocation extents, staging/u32 arithmetic, exact power-of-two index geometry, append load capacity, and
  fixed/text source spans are checked before CUDA mutation. Text offsets are bounded in PTX before byte loads;
  fingerprints plus one status word use a single `4*rows+4` readback. Fused row-count publication was removed
  from the multi-block kernel: its blocking decline read now fences every value/stamp/index write before one
  ordered 8-byte header HtoD, closing the prior cross-block publication race. Every launcher arms an RAII
  default-stream drain before launch, including DtoH failure paths. The under-lock device-index cache recheck
  now also verifies resident generation pointer identity. Same-key MVCC twin advancement, 256-probe decline,
  compound hash parity, and GPU-only derivation remain unchanged. One pure boundary test and one retained-GPU
  malformed/misaligned/load/cross-owner/fail-closed/reuse test raise the execution inventory to 120 tests, with
  44 active and 76 GPU-ignored. Eight safety/index/fused/fixed/wide/UUID/text/streaming gates passed 24
  sequential and 16 concurrent invocations without CUDA 700/716/717; workspace check, execution clippy, and
  independent audit are clean. `write_apply.rs` is 1,256 lines and the execution root is 19,022 lines. The
  physical cross-context branch remains necessarily vacuous on this one-GPU host; **MULTI-002** owns the
  non-vacuous multi-device gate.
  The five resident sidecar launchers and their byte-identical ASCII PTX now live in the private 819-line
  `execution::resident_sidecar` module: u64 slot scatter, bool bitmap range set, bool/null bitmap shard gather,
  and text-offset rebase. Method bodies are normalized-exact apart from blank lines; symbols, arguments/grids,
  source/destination addressing, bit/NULL/text semantics, synchronization, inherent APIs, and all engine callers
  are unchanged. The exact 120-test execution inventory and 44/76 suite remain stable. Five resident DELETE,
  bool, NULL, multi-shard text-rebase, and compound-text gates passed 15 sequential and 10 concurrent invocations
  without CUDA 700/716/717; workspace check and execution clippy are green. Independent extraction audit is
  clean and exposed pre-existing raw source ownership/context/extent, unbounded scatter/bitmap destinations,
  and implicit semantic preconditions, promoted as **STRUCT-001AD** ahead of further decomposition. The execution
  root is 18,207 lines.
  That sidecar hardening is complete. `CudaSidecarSource` and `CudaTextOffsetSource` retain the exact source
  allocation and originating primary context; checked 4/8-byte windows bound every source and destination before
  context selection or CUDA work, same-allocation gather/rebase aliasing fails closed, and engine row/blob
  geometry uses checked conversions and accumulation. Scatter bounds its maximum slot and rejects mismatched
  empty inputs. Bool set/gather and NULL gather atomically write both logical states, independent of destination
  prefill. Text rebase validates a real owned blob span and checks zero start, monotonicity, exact terminal blob
  length, and rebase overflow on-device before publication. Every launched failure drains before owners or pooled
  leases drop. Three pure arithmetic tests and one direct retained-GPU malformed/prefill/alias/reuse test raise
  the exact execution inventory to 124 tests, with 47 active and 77 GPU-ignored. Six sidecar/DELETE/bool/NULL/text
  gates passed 18 sequential and 12 concurrent final-PTX invocations without CUDA 700/716/717; workspace check,
  execution clippy, and independent re-audit are clean. `resident_sidecar.rs` is 1,102 lines and the root is
  18,208 lines. The one-GPU host leaves the physical cross-context branch vacuous, explicitly owned by
  **MULTI-003**. The final canonical card is green in both layers and cache regimes: in-L2/out-of-L2
  `sum_i32` measured 1,430.2/1,439.1 GB/s, `count_i32_compare` measured 0.92x/1.02x same-run roofline,
  grouped aggregation measured 1,676.6 M elements/s, and 65,536-batch point reads measured 243.4M/251.4M
  lookups/s at p50 139/132us.
  The bounded device-final uniqueness authorization kernel now lives in the private 195-line
  `execution::unique_coordinate` module. Its normalized source is byte-identical apart from required
  `pub(super)` visibility (independent hashes match); PTX/symbol/ABI, candidate and exclusion H2D marshaling,
  threshold semantics, cached function, pooled lifetimes/error draining, one-u32 readback, public inherent
  facade, and engine callers are unchanged. The exact 124-test execution inventory and 47/77 suite remain
  stable. Five keyed INSERT/self-excluding UPDATE/collision/NULL/compound-NULL gates passed 15 sequential and
  10 concurrent invocations; workspace check, execution clippy, and independent audit are clean. The root is
  18,029 lines. The post-move card exposed a source-layout-sensitive host result conversion: identical PTX
  measured ~34.7 GB/s in the parent image but ~23.7 GB/s after extraction at 50% selectivity because a
  4M-element `Vec<i32>` to `Vec<u32>` allocation/copy was only optimizer-elided in some code layouts.
  STRUCT-001AF made that ownership transfer explicit: identical size/alignment and all-bit-valid scalar types
  permit one documented allocation-preserving reinterpretation, with pointer/length/capacity and boundary-bit
  unit coverage. The execution inventory is now 125 tests, with 48 active and 77 GPU-ignored. Three ordered
  multi-block/pool-reuse/engine-route gates passed 9 sequential and 6 concurrent
  invocations, and independent unsafe-code audit is clean. The final card restores ordered compaction to
  34.4 GB/s (0.0234x same-run roofline) in-L2 and 3.7 GB/s out-of-L2; count is 0.87x/1.02x roofline, grouped is
  1,678.3 M elements/s, and 65,536-batch point reads are 246.2M/253.2M lookups/s at p50 139/132us. Both AE and
  AF are closed.
  The exact resident u64 row-count PTX and launcher now live in the private 92-line
  `execution::resident_header` module behind the unchanged `CudaResidentDeviceMemory` facade. Normalized
  implementation comparison is exact apart from parent-private visibility; the >=8-byte allocation guard,
  cached symbol, 2xu64 ABI, 1x1x1 grid, pooled stream/scratch/events, launched-error drains, one bounded 8-byte
  readback, and little-endian decode are unchanged. The direct primary-context/module-cache gate passed three
  sequential and two concurrent invocations. Workspace all-target/all-feature check, execution clippy, and
  independent re-audit are clean; the audit's sole stale test-owner comment was corrected. The exact 125-test
  inventory and 48 active/77 GPU-ignored suite remain stable, and the execution root is 17,940 lines. The
  canonical card remained green: in-L2/out-of-L2 `sum_i32` measured 1,480.1/1,448.3 GB/s,
  `count_i32_compare` measured 0.873x/1.000x same-run roofline, ordered compaction measured 34.8 GB/s
  (0.0235x roofline)/3.7 GB/s, grouped aggregation measured 1,678.0 M elements/s, and 65,536-batch point reads
  measured 246.6M/252.6M lookups/s at p50 139/131us. STRUCT-001AG is closed.
  The legacy compatibility server now has an explicit private 194-line `backend_adapter` leaf. Its 28 moved
  definitions normalize exactly to the parent after visibility/format normalization; `BackendWriter` delegation,
  authentication and command framing, row tags, text/binary format codes, and `Column`/`ErrorField` conversion
  remain unchanged. The root is 31,698 lines. Protocol all-target check and clippy with warnings denied, the
  exact 123-test binary suite, tokio-postgres and SQLx end-to-end smokes, and the complete TLS/SCRAM connection-
  security preflight pass. The preflight's stale owner and removed-document checks now follow the adapter,
  protocol-library framing owner, and canonical STATUS fact. Independent audit is clean. STRUCT-001AH is closed;
  this is containment of the legacy host-backed compatibility endpoint pending **PRODUCT-001**, not a product
  relational path.
  Legacy listener bootstrap and connection security now live in the private 754-line `server_bootstrap` module.
  Root `main` delegates to its single `pub(super) run`; every config, TLS, and SCRAM type/function remains module-
  private. The module calls only the existing ready loop, shared tagged-frame reader, and backend adapter, with
  no session, catalog, DML, prepared, portal, or cursor type crossing the seam. Normalized production ranges and
  the two moved argument tests hash exactly against the parent after the `main`-to-`run` rename. CLI/environment
  precedence, thread-per-connection listener behavior, TLS upgrade/nested-SSL rejection, startup packet handling,
  SCRAM crypto/error/wire order, and startup frame bounds are unchanged. The root is 30,966 lines and the exact
  123-test inventory remains stable. Protocol full tests, all-target check, clippy with warnings denied,
  tokio-postgres/SQLx smokes, full TLS/SCRAM preflight, formatting, diff, and independent audit are clean.
  STRUCT-001AI is closed under the same **PRODUCT-001** containment boundary.
  The shared legacy connection trait and tagged frontend-frame reader now live in the private 33-line
  `frontend_transport` leaf. Normalized trait/blanket-implementation/reader tokens exactly match the parent
  apart from required parent-private visibility. Bootstrap, the root ready loop, and the backend adapter now
  depend on this lower transport owner; no parser or session behavior moved and no cycle or public API was
  introduced. The 71 library + 123 binary tests, tokio-postgres/SQLx smokes, all-target check, warning-denied
  clippy, TLS/SCRAM preflight, touched-module formatting, and independent audit are clean. STRUCT-001AJ is
  closed. Its audit exposed inherited unbounded attacker-declared frame allocation, now owned by
  **STRUCT-001AK**.
  Frontend frame allocation is now explicitly bounded before allocation: startup declarations include their
  four-byte length and are capped at 64 KiB; tagged declarations include their four-byte length and are capped
  at 64 MiB. Accepted frames are read directly into one exact buffer, removing the former tagged payload
  allocation and copy. Four focused tests cover both readers' clean EOF/partial reads, minimum, ordinary
  reconstruction, allocation-free exact-limit validation, and one-byte-over rejection. The binary inventory is
  now 127 tests; all pass alongside the 71 library tests, driver smokes, all-target/clippy gates, and full
  TLS/SCRAM preflight. Independent security audit is clean for this per-frame contract. Legal single messages
  above 64 MiB are intentionally rejected; ordinary driver and COPY chunking pass. Aggregate pre-authentication
  DoS remains under **SCALE-001** because the legacy thread-per-connection endpoint lacks admission limits and
  read deadlines, so many slow clients can still pin bounded buffers concurrently. STRUCT-001AK is closed.
  The legacy server's 121 root tests and five shared helpers now live behind the private external `tests` module.
  After deindent and canonical formatting, the complete parent/current bodies are byte-identical; all 126
  module-level functions occur once, all 121 attributes remain, and the exact 127-test binary inventory retains
  every legacy `tests::...` name. The production root change is only a cfg/path declaration, with no visibility,
  API, or non-test build expansion. Protocol full tests, all-target/all-feature check, warning-denied clippy,
  driver smokes, security preflight, formatting, and independent audit are clean. The root is 20,413 lines.
  STRUCT-001AL is closed; the intentionally intermediate 10,480-line test owner remained explicitly owned by
  **STRUCT-001AM** and was not treated as a size exception.
  That intermediate owner is now split into a 120-line support parent and seven real invariant child modules:
  `shared_catalog` (1,324 lines/18 tests), `copy_dml` (1,700/28), `catalog_introspection` (1,981/5),
  `extended_lifecycle` (2,125/30), `extended_bind` (1,629/25), `cursor_prepare` (1,263/8), and
  `catalog_metadata` (360/6). All are below the 3,000-line test envelope. Parent support is byte-exact and every
  child canonically reconstructs the parent source. All 121 tests occur exactly once: the support asyncpg name
  is unchanged, while each other `tests::<leaf>` is exactly `tests::<group>::<leaf>`. No production visibility,
  include fragment, cycle, catch-all, or numbered shard was introduced. Both sequential and default-concurrent
  127-test runs, full package/driver/security gates, formatting, and independent audit are clean. STRUCT-001AM
  is closed.
  Legacy bind/describe ownership now lives in the private 1,179-line `bind_describe` module. The normalized
  1,152-line/47-function production body is exact. Exactly 15 functions with proven external production callers
  are parent-private; all 32 remaining helpers stay module-private. Three additional `cfg(test)` delegates keep
  existing direct-helper tests without production-build visibility. Parameter error precedence/text and OIDs,
  quoted/comment placeholder handling, SQL EXECUTE mapping, negative LIMIT/OFFSET describe behavior, and result
  columns are unchanged. Sequential and concurrent 127-test runs, full workspace/protocol checks, warning-denied
  clippy, driver smokes, TLS/SCRAM preflight, affected psql scenarios, formatting, and independent audit are
  clean. The root is 19,278 lines and STRUCT-001AN is closed.
  The legacy ready loop and frontend-message state machine now live in the private 261-line
  `frontend_dispatch` module. The production body is normalized-exact; only `handle_ready_client` is
  parent-private, while direct tests use two `cfg(test)` delegates for the private message handler and
  unsupported mapping. `ReadyLoopState`, transaction status, skip-until-Sync recovery, COPY data/done/fail,
  Flush/Terminate, and exact errors are unchanged. Session/catalog/DML ownership stays in the root and the
  module calls existing handlers rather than exposing host execution. Sequential/concurrent 127-test runs,
  full protocol/driver/security gates, formatting, and independent audit are clean. Security source guards now
  follow the moved unsupported-message owner. The root is 19,052 lines and STRUCT-001AO is closed.
  Legacy Parse/Bind/Describe/Execute/Close handling now lives in the private 622-line `extended_query` module.
  Its 590-line production body is normalized-exact; exactly five handler entry points are parent-private, all
  portal batching and format/decode helpers remain private, and one `cfg(test)` delegate preserves direct batch
  testing. Statement/portal replacement, skip-until-Sync signaling, parameter/result format arity, binary
  int4/text errors, describe timing, suspension/resume positions and completion tags, COPY/cursor/DML delegation,
  and exact errors are unchanged. Sequential/concurrent 127-test runs, all-target/all-feature check,
  warning-denied clippy, the full 71-library-test protocol package and driver smokes, security preflight,
  formatting, and independent audit are clean. The root is 18,466 lines and STRUCT-001AP is closed.
  SQL PREPARE/EXECUTE/DEALLOCATE compatibility is now split along an acyclic ownership boundary. The private
  913-line `sql_execute_syntax` leaf owns comment stripping, keyword/name/list parsing, SQL EXECUTE argument and
  literal/cast decoding, and exposes seven proven sibling/root helpers; all remaining helpers are private. The
  private 301-line `sql_prepare` owner contains PREPARE/DEALLOCATE parsing plus describe and prepared-result
  execution behind seven proven entry points. Both `bind_describe` and `sql_prepare` depend downward on the
  syntax leaf; only `sql_prepare` depends on bind semantics, so no module cycle is hidden by the root facade.
  The original 1,083-line root body and the 105-line comment-strip move are canonical-exact apart from required
  visibility/formatting. Quote/comment/dollar-quote handling, literal/NULL/cast normalization, type validation,
  replacement/deallocation behavior, result/error tags, and legacy host containment are unchanged. Sequential
  and concurrent 127-test runs, the full 71-library-test protocol package and driver smokes, all-target checks,
  warning-denied clippy, security preflight, formatting, and independent audit are clean. The root is 17,394
  lines and STRUCT-001AQ is closed.
  Cursor compatibility now lives in the private 407-line `cursor` module. The two original parser/execution
  blocks are normalized-exact and expose exactly ten proven entry points; name/direction/count parsing and all
  other helpers remain private. The result-column-count helper moved byte-exactly to its sole `extended_query`
  owner, removing a potential reverse dependency: `cursor` depends on the lower SQL syntax/PREPARE owners, while
  neither depends on cursor. DECLARE duplicate and parameter errors, SQL EXECUTE/SELECT result ownership,
  transaction-local lifetime, FETCH/MOVE forward-only position and ALL/count behavior, missing-cursor errors,
  rows, tags, and exact messages are unchanged. Sequential/concurrent 127-test runs, the full protocol package
  and driver smokes, all-target checks, warning-denied clippy, security preflight, formatting, and independent
  audit are clean. The root is 17,003 lines and STRUCT-001AR is closed.
  Legacy extended INSERT/DELETE/UPDATE application now lives in the private 205-line `extended_dml` leaf. The
  original 197-line body is normalized-exact and exposes exactly three entry points, each consumed once by the
  extended-query owner. Relation-before-permission precedence, column/type/duplicate validation, precomputed
  DELETE/UPDATE masks, mutation order, dirty-table publication, exact error fields, and completion tags are
  unchanged. This is preserved compatibility behavior, not a new integrity guarantee: these handlers do not
  themselves call not-null/unique/foreign-key/check validators, and multi-row INSERT retains its inherited
  possibility of partial mutation when a later row fails validation. The extraction adds no host product API or
  relational capability. Sequential/concurrent 127-test runs, the full protocol package and driver smokes,
  all-target checks, warning-denied clippy, security preflight, formatting, and independent audit are clean. The
  root is 16,809 lines and STRUCT-001AS is closed.
  Legacy COPY state and execution now live in the private 283-line `copy_execution` module. The original
  273-line body is normalized-exact and exposes exactly four proven entry points: COPY TO/FROM setup serves the
  simple and extended paths, while data buffering and row application serve frontend dispatch. Text/CSV
  delimiter, quote, escape, header and NULL encoding, wire framing, buffering/terminator handling, validation,
  rollback/no-mutation behavior, dirty-table publication, snapshot persistence, CopyFail recovery, error
  precedence, and completion tags are unchanged. Dependencies remain private and acyclic, and no host product
  API or relational behavior was added. Sequential/concurrent 127-test runs, the full protocol package and
  driver smokes, all-target checks, warning-denied clippy, security preflight, formatting, and independent audit
  are clean. No read kernel, residency, or result path changed, so the GPU report card was not applicable. The
  root is 16,541 lines and STRUCT-001AT is closed.
  Legacy TRUNCATE/DROP syntax now lives in the private 154-line `ddl_syntax` leaf. The original 150-line body is
  canonical-exact and exposes four proven parser/predicate entry points plus the three returned record types and
  their required fields; shared identifier helpers remain private. TRUNCATE options, multi-table DROP ordering,
  IF EXISTS and constraint flags, qualified-name restrictions, comment/case/semicolon handling, narrow rejection,
  and unsupported foreign-key option recognition are unchanged. The leaf depends only on pure SQL normalizers;
  DDL mutation and catalog execution remain in the root, with no sibling cycle or host product API. Sequential/
  concurrent 127-test runs, the full protocol package and driver smokes, all-target checks, warning-denied clippy,
  security preflight, formatting, and independent audit are clean. The root is 16,398 lines and STRUCT-001AU is
  closed.
  Simple-query COPY routing now enters the existing private `copy_execution` owner through one tri-state
  `try_execute_copy_statement` delegate. Parsed COPY TO, parsed COPY FROM with `simple_query=true`, and recognized-
  but-unsupported COPY retain their exact precedence, arguments, wire/state effects, `0A000` error, and return
  behavior; empty-query/canonicalization and all non-COPY routing remain in `execute_statement`. Protocol parse
  helpers are imported directly by the owner and remain test-only at the root facade. Sequential/concurrent
  127-test runs, the full protocol package and driver smokes, all-target checks, warning-denied clippy, security
  preflight, touched formatting, diff checks, and independent audit are clean. The root is 16,380 lines,
  `copy_execution` is 322 lines, and STRUCT-001AW is closed.
  Simple-query cursor routing now enters the existing private `cursor` owner through one tri-state delegate.
  DECLARE, unsupported DECLARE, FETCH, unsupported FETCH, MOVE, unsupported MOVE, CLOSE, then unhandled fallthrough
  retain exact order, errors, tags, count/ALL and position behavior, and state mutation. DECLARE still discards the
  extended-path boolean while propagating I/O failure. `CloseCursorTarget` moved intact from the root to its owner;
  direct parse/FETCH/MOVE/CLOSE helpers and the enum now cross the root only under `cfg(test)`, while the three
  production cursor facades remain solely for extended-query consumers. Sequential/concurrent 127-test runs,
  the full protocol package and driver smokes, all-target checks, warning-denied clippy, security preflight,
  touched formatting, diff checks, and independent audit are clean. The root is 16,322 lines, `cursor` is 479
  lines, and STRUCT-001AX is closed.
  Simple-query SQL prepared routing now enters the existing private `sql_prepare` owner through one tri-state
  delegate. Duplicate-name precheck, pg_dump domain/function PREPARE and EXECUTE cases, relational SQL PREPARE
  validation/type resolution/install, AddTen/function/SQL EXECUTE results, and DEALLOCATE ALL/named mutations
  retain exact precedence, map effects, row-description timing, errors, and tags. The already-computed canonical
  statement is passed unchanged; unhandled statements fall through. `SqlDeallocateTarget` moved into its owner,
  with parse helpers and the enum crossing the root only under `cfg(test)`. Sequential/concurrent 127-test runs,
  the full protocol package and driver smokes, all-target checks, warning-denied clippy, security preflight,
  touched formatting, diff checks, and independent audit are clean. The root is 16,139 lines, `sql_prepare` is
  502 lines, and STRUCT-001AY is closed.
  Legacy DDL mutation routing now enters the private 291-line `ddl_execution` owner through one tri-state
  delegate. TRUNCATE (including restart-identity state), DROP TABLE dependency/index/comment cleanup, and ALTER
  TABLE DROP CONSTRAINT retain exact branch order, relation-kind/existence precedence, foreign-key preflight and
  rollback, dirty publication, catalog persistence, errors, and command tags. The three private bodies are
  normalized byte-equivalent to their prior root bodies; `ddl_syntax` remains a one-way dependency and only the
  parent-private delegate is exposed for production. Sequential/concurrent 127-test runs, the full protocol
  package and driver smokes, all-target checks, warning-denied clippy, security preflight, touched formatting,
  diff checks, and independent audit are clean. No read kernel, residency, or result path changed, so the GPU
  report card was not applicable. The root is 15,898 lines and STRUCT-001AZ is closed.
  Legacy session compatibility now lives in the private 101-line `session_compat` owner. One tri-state delegate
  preserves pg_dump SET normalization plus RESET search_path and ACCESS SHARE LOCK acknowledgement before DDL;
  a second preserves the fixed search-path/restrict-kind set_config, advisory-unlock, recovery-status, and
  current-schemas queries after DDL. Exact predicates, columns, rows/nulls, row-description behavior, tags, and
  fallthrough are unchanged. A later 37-line subset was deleted after independent audit proved every predicate
  byte-identical to and dominated by these earlier unconditional-return branches; no unique public-path or
  current-schemas branch was removed. Sequential/concurrent 127-test runs, the full protocol package and driver
  smokes, all-target checks, warning-denied clippy, security preflight, touched formatting, diff checks, and
  independent audit are clean. The root is 15,797 lines and STRUCT-001BA is closed.
  The contiguous pg_dump/pg_dumpall catalog compatibility prelude now lives in the private 568-line
  `pg_dump_compat` owner behind one tri-state delegate immediately before parsed-command execution. All 48
  handled branches preserve exact order, predicates, metadata/row/null construction, comments/ACL/dependency
  behavior, errors, framing, and sequence setval/last-value state, dirty marking, and persistence. After
  normalizing only `Some` wrapping and direct `&str` argument spelling, the complete moved body hashes identically
  to its source. Sequential/concurrent 127-test runs, the full protocol package and driver smokes, all-target
  checks, warning-denied clippy, security preflight, touched formatting, diff checks, and independent audit are
  clean. The host's PostgreSQL 18.4 `pg_dump` class-metadata query and final `psql \db+` verification shape exceed
  the repository's PostgreSQL 16 compatibility baseline: pg_dumpall generation/restore succeeds, while those
  version-18 introspection probes still fail through the unchanged predicates; broader catalog-version coverage
  remains PRODUCT-002. The root is 15,274 lines and STRUCT-001BB is closed.
  Parsed session and transaction commands now enter the private 58-line `session_commands` owner through one
  borrowed-command tri-state delegate in a preceding successful-parse arm. SET/RESET ROLE validation and state,
  BEGIN, COMMIT/ROLLBACK chain state, transaction-end cursor cleanup, exact errors, and tags are unchanged;
  non-role ResetAll falls through as before. The one role-name clone is ownership-only. Sequential/concurrent
  127-test runs, the full protocol package and driver smokes, all-target checks, warning-denied clippy, security
  preflight, touched formatting, diff checks, and independent audit are clean. The root is 15,255 lines and
  STRUCT-001BC is closed.
  Parsed bootstrap catalog DDL now enters the private 165-line `bootstrap_ddl` owner through one successful-
  parse variant delegate. The bounded plpgsql extension and public-schema CREATE/DROP arms preserve exact name
  and schema restrictions, IF EXISTS/IF NOT EXISTS precedence, public-schema state, non-empty dependency checks,
  ACL/comment cleanup, dirty flags, persistence, SQLSTATEs/messages, and tags. The complete moved body hashes
  identically after normalizing only tri-state wrapping and clippy-required tail expressions. Sequential/
  concurrent 127-test runs, the full protocol package and driver smokes, all-target checks, warning-denied
  clippy, security preflight, touched formatting, diff checks, and independent audit are clean. The root is
  15,125 lines and STRUCT-001BD is closed.
  Parsed cluster-object DDL now enters the private 303-line `cluster_ddl` owner through one exact successful-
  parse variant gate that consumes the owned command. Database and tablespace CREATE/DROP/RENAME preserve
  duplicate/protected/missing/target precedence, checked OID publication, maps, ACL/comment cleanup or retargeting,
  dirty old/new keys, snapshot persistence, errors, and tags. The complete six-arm body is normalized-exact;
  only clippy-required tail expressions differ. Sequential/concurrent 127-test runs, the full protocol package
  and driver smokes, all-target checks, warning-denied clippy, security preflight, touched formatting, diff
  checks, and independent audit are clean. The root is 14,859 lines and STRUCT-001BE is closed.
  Parsed table-definition DDL now enters the existing private `ddl_execution` owner through one exact
  successful-parse variant gate. CREATE TABLE, ADD PRIMARY KEY/UNIQUE/CHECK/FOREIGN KEY, DROP/RENAME CONSTRAINT,
  and RENAME TABLE preserve schema permissions, type/domain/default and implicit-sequence/OID preflight,
  validation/rollback, indexes/constraints/comments/ACLs, dirty flags, persistence, errors, and tags. The full
  eight-arm body is normalized-exact; only clippy-required tail expressions differ. Sequential/concurrent
  127-test runs, the full protocol package and driver smokes, all-target checks, warning-denied clippy, security
  preflight, touched formatting, diff checks, and independent audit are clean. The root is 14,573 lines,
  `ddl_execution` is 609 lines, and STRUCT-001BF is closed.
  Parsed index DDL now enters the private 161-line `index_ddl` owner through one exact successful-parse variant
  gate. CREATE/RENAME/DROP INDEX preserve permission and name/kind/existence precedence, column and unique-data
  validation, constraint-backed guards, duplicate-list semantics, comment retarget/removal, dirty publication,
  persistence, errors, and tags. The complete three-arm body is normalized-exact; CREATE retains its original
  absence of relation-OID mutation. Sequential/concurrent 127-test runs, the full protocol package and driver
  smokes, all-target checks, warning-denied clippy, security preflight, touched formatting, diff checks, and
  independent audit are clean. The root is 14,440 lines and STRUCT-001BG is closed.
  Parsed view/materialized-view lifecycle commands now enter the private 514-line `view_ddl` owner through one
  exact successful-parse variant gate. CREATE/REFRESH/RENAME/DROP preserve schema permissions, name/kind/
  existence ordering, SELECT and dependency validation, materialized rows and checked OIDs, ACL/comment movement
  or removal, dirty keys, persistence, errors, and tags. The seven-arm body is normalized-exact; only clippy-
  required tail expressions differ. Sequential/concurrent 127-test runs, the full protocol package and driver
  smokes, all-target checks, warning-denied clippy, security preflight, touched formatting, diff checks, and
  independent audit are clean. The root is 13,965 lines and STRUCT-001BH is closed.
  Parsed bounded-function lifecycle and invocation commands now enter the private 113-line `function_execution`
  owner through one exact successful-parse variant gate. CREATE/RENAME/DROP preserve permission, signature/name/
  existence and supported-body validation, checked OIDs, ACL/comment state, dirty publication, persistence,
  errors, and tags; invocation preserves result propagation and the row-description flag. The four-arm body is
  normalized-exact; only clippy-required tail expressions differ. Sequential/concurrent 127-test runs, the full
  protocol package and driver smokes, all-target checks, warning-denied clippy, security preflight, touched
  formatting, diff checks, and independent audit are clean. The root is 13,888 lines and STRUCT-001BI is closed.
  Parsed sequence lifecycle and value-function commands now enter the private 220-line `sequence_execution`
  owner through one exact successful-parse variant gate. CREATE/RENAME/DROP plus nextval/currval/setval preserve
  permission, name/kind/existence, option/range/OID and overflow ordering, last-value/is-called/currval state,
  returned rows, comments/ACLs, dirty publication, persistence, errors, and tags. The six-arm body is normalized-
  exact; only clippy-required tail expressions differ. This does not strengthen inherited dependency semantics:
  DROP does not scan table defaults and rename does not retarget stored `SequenceNextVal` defaults; broader
  PostgreSQL catalog/dependency compatibility remains PRODUCT-002. Sequential/concurrent 127-test runs, the full
  protocol package and driver smokes, all-target checks, warning-denied clippy, security preflight, touched
  formatting, diff checks, and independent audit are clean. The root is 13,709 lines and STRUCT-001BJ is closed.
  Parsed domain lifecycle commands now enter the private 127-line `domain_ddl` owner through one exact
  successful-parse variant gate. CREATE/DROP DOMAIN preserve schema permission, name/kind/existence and supported
  base-type/default validation, checked OIDs, duplicate-list/IF EXISTS behavior, whole-list table-column dependency
  rejection before mutation, comment cleanup, dirty publication, persistence, errors, and tags. The two-arm body
  is normalized-exact; only clippy-required tail expressions differ. Sequential/concurrent 127-test runs, the
  full protocol package and driver smokes, all-target checks, warning-denied clippy, security preflight, touched
  formatting, diff checks, and independent audit are clean. The root is 13,608 lines and STRUCT-001BK is closed.
  Parsed publication/subscription lifecycle commands now enter the private 56-line `replication_catalog` owner
  through one exact successful-parse variant gate. CREATE/DROP PUBLICATION and SUBSCRIPTION preserve schema
  checks, helper arguments/ownership/order, propagated errors, snapshot-persistence placement, and tags. The
  four-arm body is token-equivalent after normalizing formatting and clippy-required tail expressions.
  Sequential/concurrent 127-test runs, the full protocol package and driver smokes, all-target checks, warning-
  denied clippy, security preflight, touched formatting, diff checks, and independent audit are clean. The root
  is 13,580 lines and STRUCT-001BL is closed.
  Parsed role lifecycle commands now enter the private 203-line `role_ddl` owner through one exact successful-
  parse variant gate. CREATE/DROP/RENAME ROLE preserve reserved/bootstrap, duplicate/existence/dependency and
  checked-OID behavior, all implemented ACL/comment cleanup or retargeting, dirty keys, persistence, errors, and
  tags. The three-arm body is normalized-exact; only clippy-required tail expressions differ. This does not
  strengthen inherited identity semantics: rename does not update `session.current_role`, active-role identity is
  not a DROP dependency, and this endpoint stores no membership graph; the stale identity normally fails closed
  and broader role/catalog compatibility remains PRODUCT-002. Sequential/concurrent 127-test runs, the full
  protocol package and driver smokes, all-target checks, warning-denied clippy, security preflight, touched
  formatting, diff checks, and independent audit are clean. The root is 13,404 lines and STRUCT-001BM is closed.
  The remaining parsed table-definition mutation commands now enter the existing private `ddl_execution` owner
  through the same exact successful-parse gate. DROP TABLE, ALTER COLUMN SET/DROP DEFAULT, ADD COLUMN, RENAME
  COLUMN, and DROP COLUMN preserve relation kind/existence and permission precedence, default/domain/type
  preflight, checked attnums and implicit sequences, candidate validation/rollback, indexes, constraints, ACLs,
  comments, dirty publication, persistence, errors, and tags. The five-arm body is token-equivalent after only
  clippy-required tail expressions and rustfmt normalization. Sequential/concurrent 127-test runs, the full
  protocol package and driver smokes, all-target checks, warning-denied clippy, security preflight, touched
  formatting, diff checks, and independent audit are clean. The root is 13,104 lines, `ddl_execution` is 914
  lines, and STRUCT-001BN is closed.
  Parsed catalog comments now enter the private 336-line `catalog_comments` owner through one exact successful-
  parse variant gate. All 16 supported targets preserve their kind/existence and shared-catalog precedence;
  columns retain stored attnums and constraints retain primary/unique/check/foreign-key lookup. Comment insert/
  removal, dirty publication, persistence, errors, and tag are unchanged. The complete arm is normalized-exact;
  only its clippy-required tail expression differs. Sequential/concurrent 127-test runs, the full protocol
  package and driver smokes, all-target checks, warning-denied clippy, security preflight, touched formatting,
  diff checks, and independent audit are clean. The root is 12,795 lines and STRUCT-001BO is closed.
  Parsed ACL mutations now enter the private 146-line `acl_execution` owner through one exact successful-parse
  family gate. Relation/table, schema, database, tablespace, function, and default-table-privilege GRANT/REVOKE
  preserve helper selection and argument order, propagated errors, helper-owned dirty state, success-only
  persistence, and command tags. The twelve-arm body is token-equivalent after only clippy-required tail
  expressions and rustfmt normalization. Sequential/concurrent 127-test runs, the full protocol package and
  driver smokes, all-target checks, warning-denied clippy, security preflight, focused ACL/shared-catalog tests,
  touched formatting, diff checks, and independent audit are clean. The root is 12,698 lines and STRUCT-001BP
  is closed.
  Parsed simple-query relational DML now enters the private 274-line `simple_dml` owner through one exact
  successful-parse family gate. INSERT/DELETE/UPDATE preserve existence and ACL precedence, mapping/type/default
  evaluation, filters/assignments, candidate construction, unique/check/foreign-key validation and rollback,
  counts, dirty publication, persistence, errors, and tags. The three-arm body is normalized-exact; only clippy-
  required tail expressions differ. It intentionally remains separate from behaviorally different
  `extended_dml`. Inherited compatibility debt remains: duplicate INSERT target columns are not rejected and
  the last supplied value wins; broader PostgreSQL parser/catalog compatibility remains PRODUCT-002.
  Sequential/concurrent 127-test runs, the full protocol package and driver smokes, all-target checks, warning-
  denied clippy, security preflight, focused DML tests, touched formatting, diff checks, and independent audit
  are clean. The root is 12,451 lines and STRUCT-001BQ is closed.
  Legacy host-backed SELECT execution now lives in the private 1,432-line `select_execution` owner, explicitly
  labeled parity/bootstrap debt rather than product direction. It owns comparison and SELECT/DELETE predicates,
  view/materialized-view recursion and ACLs, projection/distinct/order/limit/offset, scalar/grouped aggregates
  and HAVING, formatting/parsing/materialization, and simple-query SELECT response. Seven narrow parent-private
  exports serve proven DML, COPY, extended-query, cursor/prepared, view, root-check, and test consumers. Both
  moved ranges and the response arm are normalized-exact; only visibility, one clippy-required tail expression,
  and rustfmt normalization differ. This also removed the root's two inherited formatting exceptions, so both
  files are rustfmt-clean. Sequential/concurrent 127-test runs, the full protocol package and driver smokes,
  all-target checks, warning-denied clippy, security preflight, focused query/view/DML/ACL tests, diff checks,
  and independent audit are clean. The root is 11,053 lines and STRUCT-001BR is closed.

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
| Two physical GPUs have not executed the scheduler, device-locate, or typed sidecar context gates | **MULTI-001**, **MULTI-002**, **MULTI-003** |
| Filtered expression-overflow ordering and route-case behavior require current-tree disposition | **READ-001** |
| Lane DELETE residuals and empty-aggregate pgwire NULL seam require focused disposition | **R3-005**, **READ-003** |
| Lanes auto-checkpoint/PITR and full crash campaign | **DUR-001**, **DUR-002** |
| Multi-node Raft/quorum serving is not integrated | **HA-001** |
| Connection/runtime scale and bounded result streaming | **SCALE-001** |
| Historical scalability-ledger findings require current-tree disposition | **SCALE-002** |

Do not add work here. Add or update one row in `PLAN.md`, then reference its ID from this table if the boundary
is an important current fact.
