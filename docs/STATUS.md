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

## Verification snapshot — 2026-07-14

- Engine library: ordinary mode **505 passed, 0 failed, 487 GPU-ignored**; complete serial mode
  **992 passed, 0 failed, 0 ignored**.
- Production transient GPU integrations: catalog and bounded-function routes pass.
- Pgwire: ordinary suite **3 passed** plus the ignored non-vacuous sharded/NULL GPU golden passes.
- Production mixed gate: **116.2k reads/s**, p50 **246us**, p99 **501us**, p99.9 **671us**; zero host gathers,
  zero fallback groups, and 160/160 host-install-elided writes.
- Read roofline: in-L2 `count_i32_compare` approximately **0.87x** the same-run `sum_i32` roofline; grouped
  kernel approximately **1,678 M elements/s**.
- Canonical report card: 48M-row out-of-L2 batched route **250.2M lookups/s at batch 65,536, p50 132us**;
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
  Bounded SQL-function result evaluation and its body parser now live inside the existing private 282-line
  `function_execution` owner. Schema USAGE, lookup, function EXECUTE ACL order, all supported return-type literal
  branches, numeric scale, quote and keyword boundaries, unsupported-body errors, column/value construction,
  and result rows are exact moves. Only `execute_function_result` gained parent visibility for three existing
  direct tests, with a test-only root import; the parser helpers remain private and the command calls the result
  helper internally. Full protocol and driver gates, all-target checks, warning-denied clippy, security preflight,
  focused function/error/recovery tests, formatting/diff checks, and independent audit are clean. The root is
  10,887 lines and STRUCT-001BS is closed.
  Recursive session-view dependency detection now lives inside the existing private 543-line `view_ddl` owner.
  Direct target detection, cycle termination through the visited set, self exclusion in dependent scans, and all
  CREATE OR REPLACE/rename/drop call sites and errors are exact moves. All three helpers remain module-private;
  no new export or dependency edge was added. Full protocol and driver gates, all-target checks, warning-denied
  clippy, security preflight, focused view/shared-catalog tests, formatting/diff checks, and independent audit are
  clean. The root is 10,857 lines and STRUCT-001BT is closed.
  Table, index, sequence, and function rename mutations now live in their sole existing private command owners;
  each helper is visible to the parent only for unchanged direct tests, and root aliases are `cfg(test)` only.
  Kind/existence/dependency precedence; table, foreign-key, index, ACL, currval and comment retargeting; dirty
  state; persistence; errors; and no-op paths are exact moves. Inherited PRODUCT-002 debt is not strengthened:
  sequence rename still does not retarget stored `SequenceNextVal` defaults, and table rename neither blocks nor
  retargets dependent materialized-view source names, which can make later refresh fail. Full protocol and driver
  gates, all-target checks, warning-denied clippy, security preflight, four focused rename persistence tests,
  formatting/diff checks, and independent audit are clean. The root is 10,563 lines; `ddl_execution`,
  `index_ddl`, `sequence_execution`, and `function_execution` are 1,045, 216, 288, and 326 lines respectively;
  STRUCT-001BU is closed.
  Legacy unique-index, check-constraint, and foreign-key validation now live in the private 109-line
  `integrity_validation` owner, labeled host parity/bootstrap debt. Only the three validators are parent-private;
  violation constructors remain internal, and all root, table/index DDL, simple-DML, and COPY consumers use the
  same narrow aliases. Key comparison, predicate dispatch, missing-metadata skip behavior, FK parent sets,
  validation order, and generic errors are exact moves. Inherited PRODUCT-002 compatibility debt remains: NULL
  is an ordinary uniqueness/FK key rather than PostgreSQL NULL semantics; missing referenced metadata is skipped;
  and violation messages omit object names. Full protocol and driver gates, all-target checks, warning-denied
  clippy, security preflight, focused COPY/DML/index tests, formatting/diff checks, and independent audit are
  clean. The root is 10,466 lines and STRUCT-001BV is closed.
  Foreign-key addition and column/constraint drop/rename mutations now live inside the existing private
  1,467-line `ddl_execution` owner, leaving 33 lines of headroom under the preferred limit. Self-FK, duplicate,
  kind/existence, column/type and unique-parent precedence; `CatalogForeignKey` construction; insert/validate/pop
  rollback; checked attnums and row mutation; index/check/FK/comment changes; dirty state; persistence; and errors
  are token-exact moves. Inherited PRODUCT-002 name/dependency debt remains: referenced-column rename does not
  retarget inbound FKs stored on other tables; column rename/drop neither blocks nor retargets view or
  materialized-view column references; and dropping a serial/default-owned column leaves its implicit sequence
  orphaned. Full protocol and driver gates, all-target checks, warning-denied clippy, security preflight, focused
  table/comment persistence tests, formatting/diff checks, and independent audit are clean. The root is 10,043
  lines and STRUCT-001BW is closed.
  Primary-key, unique, and check-constraint construction now live in the private 213-line `table_constraints`
  owner with exactly three parent-private helpers consumed by `ddl_execution`. Relation/duplicate precedence,
  table/column/type checks, candidate validation, constructed index/check fields, dirty state, arguments, errors,
  and publication order are exact moves. Inherited PRODUCT-002 rollback artifacts remain: failed later CREATE
  TABLE constraints do not rewind allocated relation OIDs, and earlier helpers can leave transient dirty flags
  even though rollback removes the catalog objects and no failing snapshot is persisted. Full protocol and driver
  gates, all-target checks, warning-denied clippy, security preflight, focused table/index/shared-catalog tests,
  formatting/diff checks, and independent audit are clean. The root is 9,841 lines and STRUCT-001BX is closed.
  Sequence target validation, next-value state transition, and implicit creation now live inside the existing
  private 357-line `sequence_execution` owner. Kind-before-missing precedence, overflow-before-mutation,
  `last_value`/`is_called`, collision-before-OID behavior, checked allocation, inserted fields, dirty keys, and
  errors are exact moves; command calls are internal and three parent-private aliases serve only default and DDL
  consumers. Inherited PRODUCT-002 debt remains: sequence/default ownership is name-based across rename/drop and
  column deletion, failed later table creation can consume OIDs/transient dirty state, and target/collision checks
  omit indexes even though PostgreSQL puts indexes in the relation namespace. Full protocol and driver gates,
  all-target checks, warning-denied clippy, security preflight, focused sequence/shared-catalog tests, formatting/
  diff checks, and independent audit are clean. The root is 9,776 lines and STRUCT-001BY is closed.
  Column-default validation, formatting, sequence/domain preflight, ADD support policy, and runtime evaluation now
  live in the private 130-line `column_defaults` owner. Six narrow exports serve DDL, simple-DML, COPY, and root
  catalog formatting; literal formatting remains internal. All value types and escaping, domain OID/type mutation,
  sequence target/value/currval/dirty behavior, int4 conversion, and errors are exact moves. Inherited PRODUCT-002
  debt remains: literal compatibility uses the narrow legacy type policy, and sequence-default evaluation advances
  and records `currval` before int4 conversion, so an int4 overflow returns `22003` after consuming the value.
  Full protocol and driver gates, all-target checks, warning-denied clippy, security preflight, focused default/
  domain/sequence/COPY/DML/catalog tests, formatting/diff checks, and independent audit are clean. The root is
  9,661 lines and STRUCT-001BZ is closed.
  ACL policy, permission checks, role dependency scans, target validation, and all implemented GRANT/REVOKE
  mutations now live in the private 740-line `acl_execution` owner. Eight production policy exports serve proven
  sibling consumers; five mutation helpers are parent-visible only through test-only aliases, and command calls
  are internal. PUBLIC/grantee/current-role precedence, composed schema/object checks, target errors, set cleanup,
  dirty keys, arguments, persistence, and tags are normalized-exact. Inherited PRODUCT-002 debt remains: no role
  membership graph or active-role dependency, no owner/grantor/grant-option authorization, an empty schema ACL
  stands in for implicit PUBLIC USAGE and can change when unrelated entries appear, and function PUBLIC EXECUTE
  defaults are not modeled. Full protocol and driver gates, all-target checks, warning-denied clippy, security
  preflight, focused ACL/role/shared-catalog tests, formatting/diff checks, and independent audit are clean. The
  root is 9,078 lines and STRUCT-001CA is closed.
  Publication target validation and publication/subscription create/drop mutations now live inside the existing
  private 245-line `replication_catalog` owner; all five helpers are internal. Kind/existence and duplicate
  precedence, dependency checks, all-name validation, `IF EXISTS`, checked OIDs, constructed fields, comment
  cleanup, dirty keys, errors, command permission checks, persistence, and tags are exact moves. Inherited
  PRODUCT-002 debt remains: dropping a publication does not reject or rewrite subscriptions that name it, and
  missing `IF EXISTS` names still mark object/comment keys dirty before the success snapshot. Full protocol and
  driver gates, all-target checks, warning-denied clippy, security preflight, comment/shared-catalog tests,
  formatting/diff checks, and independent audit are clean. The root is 8,889 lines and STRUCT-001CB is closed.
  The five shared-catalog comment-target lookups now live as private helpers inside the existing 386-line
  `catalog_comments` owner. Their mutex acquisition and poison handling, table/view/sequence membership,
  live-index ownership, primary/unique/check constraint predicates, callers, evaluation order, and errors are
  exact moves. Inherited PRODUCT-002 debt remains: shared-catalog foreign keys are omitted from the constraint
  predicate, multi-kind checks can observe different catalog generations across separate lock acquisitions, and
  mutex poison panics instead of producing a protocol error. Full protocol and driver gates, all-target checks,
  warning-denied clippy, security preflight, four executing focused catalog tests, formatting/diff checks, and
  independent audit are clean. The root is 8,837 lines and STRUCT-001CC is closed.
  The final canonical session/golden compatibility match now lives behind one parent-visible delegate in the
  existing private 205-line `session_compat` owner. Its branch order, transaction flags, extended-statement
  retention on `DEALLOCATE ALL`, golden prepared-state mutations, rows, columns, tags, errors, positions, terminal
  unsupported response, and final placement after every parsing/catalog route are exact. Inherited PRODUCT-002
  debt remains: transaction state is shallow, `DEALLOCATE ALL` intentionally preserves extended statements,
  golden execution checks only name existence, golden deallocation succeeds unconditionally, and the fallback is
  an exact-string compatibility stub. All protocol/driver/security/formatting gates and independent audit are
  clean; real-psql task scenarios 01/02 and 137-198 pass, while the complete suite still fails on the separately
  tracked PostgreSQL 18 client query/render drift. The root is 8,744 lines and STRUCT-001CD is closed.
  All eleven publication/subscription catalog read shapes now execute from the existing private 672-line
  `replication_catalog` owner: eight through the original post-parse catalog stage and three through the exact
  pre-parse pg-dump position. Four query literals, one strict relation-publication OID parser, three column
  builders, and eleven row builders are token-identical private moves; the only additional exposure is five
  explicit `cfg(test)` aliases. Predicates, columns, order, fixed/synthetic OIDs, table-liveness filtering,
  all-table rows, subscription fields, and describe output are unchanged, while `pg_dump_compat` is 542 lines.
  Inherited PRODUCT-002 debt remains: hard-coded owner/feature/OID values, concatenated synthetic relation OIDs,
  public-only all-table namespace representation, disabled subscriptions, stale table names hidden by liveness
  filtering, and broad pg-dump prefix predicates. Full protocol/driver/security/formatting gates and independent
  audit are clean. Fresh PostgreSQL 18 scenarios 65, 66, and 337 pass; scenario 335's publication rows pass and
  only its known `\d` query drift fails. The root is 8,373 lines and STRUCT-001CE is closed.
  The normal/verbose psql domain listings and direct `pg_type` domain query now execute from the existing private
  245-line `domain_ddl` owner behind one tri-state delegate. Both query literals, both row builders, and all three
  route bodies are exact moves; helpers remain private and only two `cfg(test)` query aliases were added. The
  original post-language/pre-role dispatcher position, columns, name order, base-type display/OIDs, comments,
  null placeholders, and `typtype` are unchanged. Inherited PRODUCT-002 debt remains: exact client-version query
  literals, public-only domains, null collation/nullability/default/check/ACL metadata, base-type-only domain
  representation, and no visibility filtering. Full protocol/driver/security/formatting gates and independent
  audit are clean. Fresh PostgreSQL 18 scenarios 59/336 fail only on known `\dD` query drift while scenario 336's
  direct domain OID/type rows pass. The root is 8,282 lines and STRUCT-001CF is closed.
  The normal and verbose psql sequence listings now execute from the existing private 459-line
  `sequence_execution` owner behind one tri-state delegate. Both query literals, both sorted row builders, and
  both route bodies are exact moves; helpers remain private and the only new test exposure is two query aliases
  plus the verbose row alias required by shared-catalog persistence coverage. Dispatcher position, columns,
  public/type/owner/persistence/size constants, comments, and order are unchanged. Inherited PRODUCT-002 debt
  remains: exact client-version literals, hard-coded public/postgres/permanent/zero-size metadata, no visibility
  filtering, and comments only in the verbose projection. Full protocol/driver/security/formatting gates and
  independent audit are clean. Fresh PostgreSQL 18 scenarios 50/330 retain exact sequence rows/comments/state
  and differ only in known client wording/query drift. The root is 8,214 lines and STRUCT-001CG is closed.
  The normal/verbose psql stored-view and materialized-view listings now execute from the existing private
  722-line `view_ddl` owner behind one tri-state delegate. All four query literals, all four sorted row builders,
  and all four route bodies are exact moves; helpers remain private and the only new test exposure is four query
  aliases. Dispatcher position, equality, columns, public/type/owner/persistence/access-method/size constants,
  null stored-view size, comment targets, and name order are unchanged. Inherited PRODUCT-002 debt remains:
  exact client-version literals, hard-coded public/postgres/permanent/heap/static-size metadata, no application
  of the encoded visibility predicates to the returned session maps, and legacy host-catalog execution. Full
  protocol/driver/security/formatting gates, focused view/comment/shared-catalog tests, fresh real-psql scenarios
  49/50/309/331, and independent audit are clean. The root is 8,068 lines and STRUCT-001CH is closed.
  The normal/verbose psql function listings plus direct public `pg_proc` metadata and description reads now
  execute from the existing private 508-line `function_execution` owner behind one tri-state delegate. The psql
  query literal, four sorted row builders, ACL formatter, and all four routes are exact moves; helpers remain
  private and the only new test exposure is three proven aliases. Dispatcher position, equality and broad verbose
  structural predicates, columns, SQL types/OIDs/body, comments, ACL formatting, constants, nulls, and name order
  are unchanged. Inherited PRODUCT-002 debt remains: exact client-version literals, broad verbose matching,
  hard-coded metadata, omitted argument/internal-name values, no overload-aware identity or applied visibility
  filtering, and legacy host-catalog execution. Full protocol/driver/security/formatting gates, focused function/
  ACL/shared-catalog tests, fresh real-psql scenarios 51/348, and independent audit are clean. The root is 7,916
  lines and STRUCT-001CI is closed.
  The normal/verbose psql index listings now execute from the existing private 346-line `index_ddl` owner behind
  one tri-state delegate. Both query literals, the public-schema filter parser, both sorted row builders, and both
  routes are exact moves; all five helpers remain private and the only new test exposure is four proven aliases.
  Dispatcher position, verbose-first predicate order, exact-or-filter normal match, live-table filtering, columns,
  public/index/postgres/table/permanent/btree constants, null size, comments, and name order are unchanged. The
  separately positioned direct `pg_indexes` stage is untouched. Inherited PRODUCT-002 debt remains: exact client-
  version literals, unusual fixed relkind lists, public-only filtering, unapplied visibility predicates, hard-coded
  metadata, hidden orphan indexes, null size, and legacy host-catalog execution. Full protocol/driver/security/
  formatting gates, focused index/comment/shared-catalog tests, fresh real-psql scenarios 21/305/306/323/327, and
  independent audit are clean. The root is 7,824 lines and STRUCT-001CJ is closed.
  Both later direct `pg_indexes` reads now execute from the existing private 427-line `index_ddl` owner behind a
  second, separately positioned tri-state delegate. Both exact query literals, the schema-bearing sorted row
  builder, and the schema-dropping projection are byte-identical moves; helpers remain private and the only new
  test exposure is two proven aliases. The post-`pg_tables`/pre-`pg_class` stage, route order, columns, live-table
  filtering, public schema, table/index sorting, primary/unique definition formatting, and positional projection
  are unchanged. Inherited PRODUCT-002 debt remains: exact-query routing, host/session-backed public-only data,
  hidden orphan indexes, unquoted simple single-column btree definitions, and positional `skip(1)` projection.
  Full protocol/driver/security/formatting gates, focused index/shared-catalog tests, fresh real-psql scenarios
  306/323/327/329, and independent audit are clean. The root is 7,767 lines and STRUCT-001CK is closed.
  The later direct sequence `pg_class` read now executes from the existing private 506-line
  `sequence_execution` owner behind a second, separately positioned tri-state delegate. Its exact query literal,
  name-sorted row builder, and route are byte-identical moves; both helpers remain private and the only new test
  exposure is one proven row alias. The post-plain-table/pre-materialized-view class stage, equality, five columns,
  sequence OIDs, public schema, `s` relation kind, `p` persistence, and name order are unchanged. Inherited
  PRODUCT-002 debt remains: exact-query routing, host/session-backed catalog data and OIDs, public-only metadata,
  hard-coded relation-kind/persistence values, and absent GPU-native catalog execution. Full protocol/driver/
  security/formatting gates, focused sequence/shared-catalog tests, fresh real-psql scenario 330, and independent
  audit are clean. The root is 7,737 lines and STRUCT-001CL is closed.
  The later direct materialized-view `pg_class` read now executes from the existing private 764-line `view_ddl`
  owner behind a second, separately positioned tri-state delegate. Its exact query literal, name-sorted row
  builder, and route are byte-identical moves; both helpers remain private, the sole new export is the production
  delegate, and no unconsumed test alias was added. The post-sequence/pre-filtered-plain-table class stage,
  equality, five columns, materialized-view OIDs, public schema, `m` relation kind, `p` persistence, and name order
  are unchanged. Inherited PRODUCT-002 debt remains: exact-query routing, host/session-backed catalog data and
  OIDs, public-only metadata, hard-coded relation-kind/persistence fields, and absent GPU-native catalog execution.
  Full protocol/driver/security/formatting gates, focused view/materialized-view/shared-catalog tests, fresh real-
  psql scenario 331, and independent audit are clean. The root is 7,711 lines and STRUCT-001CM is closed.
  The later adjacent `information_schema.views` and `pg_catalog.pg_views` reads now execute from the existing
  private 869-line `view_ddl` owner behind a third, separately positioned tri-state delegate. Both exact query
  literals, both sorted row builders, and both routes are byte-identical moves; all helpers remain private and the
  only new test exposure is four proven aliases. The post-key-column/pre-constraint stage, route order, 10-/4-
  column shapes, definitions, database/schema/owner values, `NONE` and five `NO` capability fields, and respective
  name sorts are unchanged. Inherited PRODUCT-002 debt remains: exact-query routing, host/session-backed metadata,
  hard-coded database/schema/owner/capability fields, limited visibility semantics, and absent GPU-native catalog
  execution. Full protocol/driver/security/formatting gates, focused view/catalog tests, fresh real-psql scenario
  309, and independent audit are clean. The root is 7,640 lines and STRUCT-001CN is closed.
  The six later relation-description routes now execute from the existing private 888-line `catalog_comments`
  owner behind one tri-state delegate. Nine exact query literals, one structural psql predicate, and six row
  builders are byte-identical moves; all implementation helpers remain private and the only new exposure is the
  production delegate plus six proven `cfg(test)` aliases. The exact post-attrdef/pre-type stage, branch order,
  columns, row widths, null positions, relation/object labels, table/column/index/view/materialized-view/sequence/
  publication/subscription/constraint comment coverage, family sorting, and live-index filtering are unchanged.
  Inherited PRODUCT-002 debt remains: brittle exact/structural SQL matching, host/session-backed comments and CPU
  sorting, public-only metadata, query variants sharing broader row builders, incomplete psql object-class
  emulation, hidden orphan-index comments, and absent GPU-native catalog execution. Full protocol/driver/security/
  formatting gates, focused comment/constraint/shared-catalog tests, PostgreSQL 16 scenarios 43/305/311/330/331,
  and independent audit are clean. The root is 7,186 lines and STRUCT-001CO is closed.
  The direct public-schema description read now executes from the existing private 929-line `catalog_comments`
  owner behind a second, separately positioned tri-state delegate. Its exact query literal and row builder are
  byte-identical moves; both implementation helpers remain private and the only new exposure is the production
  delegate plus one proven `cfg(test)` row alias. The post-psql-schema/pre-namespace stage, equality, two columns,
  public-schema existence behavior, optional comment/null output, and writer are unchanged; adjacent psql schema,
  namespace/ACL, later relation-description, and pg-dump handling remain untouched. Inherited PRODUCT-002 debt
  remains exact-query matching, hard-coded public-schema metadata, host/session-backed optional comments, CPU
  result construction, and absent GPU-native catalog execution. Full protocol/driver/security/formatting gates,
  focused schema/comment/catalog tests, PostgreSQL 16 scenarios 312/340, and independent audit are clean. The root
  is 7,168 lines and STRUCT-001CP is closed.
  The remaining pg-dump description row builder now lives in the existing private 1,192-line `catalog_comments`
  owner and is consumed directly by the 543-line sibling `pg_dump_compat`; the latter's predicate, columns, writer,
  route order, and dispatcher stage are byte-identical. The builder itself is byte-identical apart from required
  private visibility; one `cfg(test)` alias serves all three existing test consumers. Every existing comment
  family, class/object/subobject value, per-family order, final numeric tuple sort, null, and index/constraint
  liveness check is unchanged. Inherited PRODUCT-002 debt remains brittle exact routing, host/synthetic OIDs, CPU
  construction, orphan and non-index-backed constraint suppression, the legacy function-comment tuple shape, and
  absent GPU-native catalog execution. Full protocol/driver/security/formatting gates, focused catalog/comment/
  shared tests, all 18 PostgreSQL 16 pg-dump restore/metadata/privilege gates, and independent audit are clean. The
  root is 6,913 lines and STRUCT-001CQ is closed.
  All effective psql/direct/pg_dumpall role catalog reads now execute from the existing private 420-line
  `role_ddl` owner behind two stage-preserving delegates; the 531-line `pg_dump_compat` consumes only the role
  delegate. Seven query/column/row helpers are byte-identical private moves and three `cfg(test)` aliases have
  proven consumers. Exact predicates, direct-before-pg_dumpall order, dynamic 10-/11-column psql output, bootstrap
  and application OIDs/flags/login/comments/nulls, database-ACL slot, writers, and existing iteration behavior are
  unchanged. The later root direct-OID branch was deleted as an exact duplicate proven dominated by unconditional
  earlier pg-dump dispatch; the security source guard now follows `rolpassword` to its owner. Inherited PRODUCT-002
  debt remains exact routing, host/synthetic OIDs, hard-coded capabilities, absent membership/grant-option
  modeling, unsorted map iteration despite query ordering, the bootstrap ACL occupying nominal `rolvaliduntil`,
  null password/expiry metadata, and absent GPU-native catalog execution. Full gates, PostgreSQL 16 scenarios
  52/315/343/346, the pg-dumpall globals restore, and independent audit are clean. The root is 6,747 lines and
  STRUCT-001CR is closed.
  All effective psql/direct/pg-dump database catalog reads now execute from the existing private 580-line
  `cluster_ddl` owner behind two stage-preserving delegates; the 527-line `pg_dump_compat` consumes only the
  database delegate. Ten query/column/row helpers are byte-identical private moves and eight `cfg(test)` aliases
  have proven consumers. Exact normal/verbose/OID/ACL order, 9-/12-/2-/2-/19-column shapes, bootstrap/application
  names and OIDs, owner/encoding/locale/size/tablespace fields, ACLs, comments, nulls, name sorting, and writers are
  unchanged. The later root pg-dump metadata block was deleted as an exact duplicate proven dominated by earlier
  unconditional dispatch. Inherited PRODUCT-002 debt remains exact routing, host/synthetic OIDs, name/OID-only
  application database modeling, hard-coded metadata, simplified ACLs, null pg-dump ACL/default/ICU fields,
  bootstrap-only pg-dump metadata, and absent GPU-native catalog execution. Full gates, PostgreSQL 16 scenarios
  53/313/344/346, all 18 pg-dump restore/metadata/privilege gates, and independent audit are clean. The root is
  6,544 lines and STRUCT-001CS is closed.
  All effective psql/direct/pg_dumpall tablespace catalog reads now execute from the existing private 820-line
  `cluster_ddl` owner behind two stage-preserving delegates; the 524-line `pg_dump_compat` consumes only the
  tablespace delegate. Eight query/column/row helpers are byte-identical private moves and seven `cfg(test)`
  aliases have proven consumers. Exact normal/verbose/OID-location/ACL order, 3-/7-/3-/2-/8-column shapes,
  bootstrap and application OIDs/names/locations, owner and fixed-size fields, ACL display/array/defaults,
  comments, options/nulls, name sorting, OID sorting, and writers are unchanged. The later root pg_dumpall route
  and its still-later empty-catalog classifier branch were deleted only after both were proven dominated by
  unconditional earlier pg-dump compatibility. Inherited PRODUCT-002 debt remains structural/exact routing,
  host/synthetic OIDs, hard-coded ownership and zero-byte sizing, metadata-only locations, simplified ACLs and
  defaults, bootstrap tablespaces omitted from pg_dumpall metadata, and absent GPU-native catalog execution. Full
  gates, PostgreSQL 16 scenarios 55/315/345/346, the pg-dumpall globals restore, and independent audit are clean.
  The root is 6,375 lines and STRUCT-001CT is closed.
  All effective psql and pg-dump extension catalog reads now execute from the existing private 257-line
  `bootstrap_ddl` owner behind two stage-preserving delegates; the 510-line `pg_dump_compat` consumes only the
  extension delegate. Four helpers are byte-identical private moves and two `cfg(test)` wrapper aliases have
  proven consumers. Exact equality predicates, post-default-ACL/pre-language and post-role/pre-language stages,
  4-/8-column shapes, `plpgsql` name/version/schema, class/object OIDs, relocatable flag, stored-comment or
  fallback description, null configuration fields, and writers are unchanged. The later root pg-dump discovery
  route was deleted only after it was proven dominated by unconditional earlier pg-dump compatibility. Inherited
  PRODUCT-002 debt remains exact routing, a single synthesized bootstrap extension, fixed OIDs/version/schema/
  relocatability, null configuration, fallback description masking absent stored metadata, CPU-built host rows,
  and absent GPU-native catalog execution. Full gates, PostgreSQL 16 scenario 54, all 18 pg-dump restore/metadata/
  privilege gates, and independent audit are clean. The root is 6,317 lines and STRUCT-001CU is closed.
  All effective psql, direct, pg-dump, and bounded information-schema schema/namespace catalog reads now execute
  from the existing private 504-line `bootstrap_ddl` owner behind four stage-preserving delegates; the 473-line
  `pg_dump_compat` consumes only the schema delegate. Eleven helpers are byte-identical private moves and nine
  `cfg(test)` wrappers have proven consumers; shared ACL rendering remains at its unchanged owner boundary. Exact
  stage and branch order, predicates, 2-/4-/2-/2-/6-/1-/2-column shapes, public-existence filtering, namespace
  table/OIDs/names/owners, ACL display/array/defaults, optional comments, nulls, and writers are unchanged. The
  two later root pg-dump routes were deleted only after both were proven dominated by unconditional earlier
  pg-dump compatibility. Inherited PRODUCT-002 debt remains exact routing, public-only schema modeling, fixed
  namespace OIDs/owners, simplified ACLs, pg-dump exposing public metadata after public-schema deletion, CPU host
  row construction, and absent GPU-native catalog execution. Full gates, PostgreSQL 16 scenarios 09/46/47/312/
  340/342, all 18 pg-dump restore/metadata/privilege gates, and independent audit are clean. The root is 6,177
  lines and STRUCT-001CV is closed.
  All effective psql and pg-dump procedural-language catalog reads now execute from the existing private 588-line
  `bootstrap_ddl` owner behind two stage-preserving delegates; the 468-line `pg_dump_compat` consumes only the
  language delegate. Four helpers are byte-identical private moves and one `cfg(test)` wrapper has a proven
  consumer. Exact equality, post-extension/pre-domain and post-extension/pre-schema stages, 4-/10-column shapes,
  `plpgsql` name, postgres owner, trust flag, fixed description, class/language/handler OIDs, ACL/default, nulls,
  and writers are unchanged. The later root discovery and still-later empty-catalog classifier branches were
  deleted only after both were proven dominated by unconditional earlier pg-dump compatibility. Inherited
  PRODUCT-002 debt remains exact routing, one synthesized language, fixed metadata/OIDs, simplified ACL/default,
  absent lifecycle-sensitive state, CPU row construction, and absent GPU-native catalog execution. Full gates,
  PostgreSQL 16 scenario 56, all 18 pg-dump restore/metadata/privilege gates, and independent audit are clean. The
  root is 6,112 lines and STRUCT-001CW is closed.
  All effective psql and pg-dump access-method catalog reads now execute from the existing private 642-line
  `bootstrap_ddl` owner behind two stage-preserving delegates; the 471-line `pg_dump_compat` consumes the empty
  `pg_am` delegate immediately before its remaining empty-catalog classifier. Two helpers are byte-identical
  private moves and two `cfg(test)` wrappers have proven consumers. Exact equality, post-tablespace/pre-schema and
  post-tablespace/pre-classifier stages, 2-/5-column shapes, the single `heap`/`Table` psql row, empty pg-dump rows,
  column names/types, and writers are unchanged. The old classifier branch was deleted only after the new exact
  delegate dominated it. Inherited PRODUCT-002 debt remains exact routing, hard-coded psql heap metadata paired
  with empty pg-dump access methods, CPU row construction, and absent GPU-native catalog execution. Full gates,
  PostgreSQL 16 scenario 57, all 18 pg-dump restore/metadata/privilege gates, and independent audit are clean. The
  root is 6,092 lines and STRUCT-001CX is closed.
  All effective psql and pg-dump default-ACL catalog reads now execute from the existing private 843-line
  `acl_execution` owner behind two stage-preserving delegates; the 467-line `pg_dump_compat` consumes only the
  pg-dump delegate. Six helpers are byte-identical private moves and two `cfg(test)` wrappers have proven
  consumers; shared generic ACL renderers remain root-owned and unchanged. Exact predicates, post-replication/
  pre-extension and post-empty-classifier/pre-dependency stages, 4-/7-column shapes, conditional empty/nonempty
  behavior, owner/schema/object type, ACL display/array/default wrapping, fixed OIDs/namespace, non-null fields,
  and writers are unchanged. The later root route was deleted only after unconditional earlier pg-dump dispatch
  proved it dominated. Inherited PRODUCT-002 debt remains exact/broad routing, CPU rows, hard-coded postgres/
  public ownership and OIDs, table-only default ACLs, and absent GPU-native catalog execution. Full gates,
  PostgreSQL 16 scenarios 68/334, all 18 pg-dump restore/metadata/privilege gates, and independent audit are clean.
  The root is 6,023 lines and STRUCT-001CY is closed.
  All effective psql and pg-dump aggregate catalog reads now execute from the existing private 563-line
  `function_execution` owner behind two stage-preserving delegates; the 471-line `pg_dump_compat` consumes the
  aggregate delegate immediately before its remaining empty classifier. The exact psql query helper is a private
  byte-identical move and one `cfg(test)` wrapper has a proven consumer. Exact equality/structural predicates,
  post-function/pre-conversion and pre-classifier stages, 5-/9-column names/types, empty rows, and writers are
  unchanged. The old aggregate classifier branch was deleted only after the new delegate dominated it. Inherited
  PRODUCT-002 debt remains exact/broad routing, always-empty aggregate catalogs, CPU compatibility rows, absent
  native aggregate metadata, and absent GPU-native catalog execution. Full gates, PostgreSQL 16 scenario 60, all
  18 pg-dump restore/metadata/privilege gates, and independent audit are clean. The root is 5,998 lines and
  STRUCT-001CZ is closed.
  The four adjacent empty psql conversion/operator/collation/cast catalog reads and their four pg-dump metadata
  responses now execute from a bounded private 174-line `type_system_catalog` owner behind two stage-preserving
  delegates; the 475-line `pg_dump_compat` consumes the pg-dump delegate immediately before its remaining empty
  classifier. Four psql query helpers are byte-identical private moves and four `cfg(test)` wrappers have proven
  consumers. Exact psql and pg-dump branch order, equality/structural predicates, 5-/6-/8-/4- and 9-/6-/5-/7-
  column names/types, empty rows, and writers are unchanged. Only those four old classifier branches were deleted.
  Inherited PRODUCT-002 debt remains exact/broad routing, eight always-empty catalogs, CPU compatibility metadata,
  and absent GPU-resident type-system relations/operators. Full gates, PostgreSQL 16 scenarios 61/62/63/64, all
  18 pg-dump restore/metadata/privilege gates, and independent audit are clean. The root is 5,895 lines and
  STRUCT-001DA is closed.
  Supported built-in psql type listings and effective pg-dump type metadata now execute from the existing private
  412-line `type_system_catalog` owner behind two additional stage-preserving delegates; the 472-line
  `pg_dump_compat` consumes only the type-metadata delegate. Twelve parser/query/registry/size/column/row helpers
  are byte-identical private moves and five `cfg(test)` aliases have proven consumers. Exact post-namespace/pre-
  table-OID and post-attribute/pre-database stages, parser and branch order, 3-/3-/8-/13-column shapes, supported-
  type display ordering and sizes, built-in/domain OIDs/names/namespaces/kinds/flags, domain OID sorting, nulls,
  and writers are unchanged. The later root pg-dump type route was deleted only after unconditional earlier
  pg-dump dispatch proved it dominated. PostgreSQL 16 scenario 58's obsolete original two-type golden was
  reconciled to the pre-existing nine-entry `SUPPORTED_SQL_TYPES` behavior; this slice made no runtime type
  behavior change. Inherited PRODUCT-002 debt remains exact routing, host/synthetic type and domain metadata,
  incomplete PostgreSQL size/ACL/array semantics, CPU row construction, and absent GPU-resident system relations.
  Full protocol/driver/security/formatting gates, PostgreSQL 16 scenarios 10/58, and all 18 pg-dump restore/
  metadata/privilege gates are clean. The root is 5,713 lines and STRUCT-001DB is closed.
  The three later direct `pg_catalog.pg_type` reads now execute from the existing private 538-line
  `type_system_catalog` owner behind one additional stage-preserving delegate. Four full/filtered OID/name
  registry row helpers are byte-identical private moves; the two full helpers remain test-only fixtures through
  two proven `cfg(test)` aliases. Exact post-relation-description/pre-attribute stage, branch order and equality,
  literal `(23, 25)` / `('int4', 'text')` filtering, OID/name sorting, 3-/3-column names and types, the one-column
  empty `hstore`/`geometry`/`vector` extension probe, and writers are unchanged. Inherited PRODUCT-002 debt
  remains exact routing, hard-coded discovery literals, host registry filtering/sorting, the narrow empty
  extension probe, and absent GPU-resident system relations. Full protocol/driver/security/formatting gates,
  PostgreSQL 16 scenario 04, and independent audit are clean. The root is 5,610 lines and STRUCT-001DC is closed.
  The early psql `\gdesc` result-type formatter now executes from the existing private 590-line
  `type_system_catalog` owner behind one additional stage-preserving delegate. The exact OID lookup and values-
  parser/row helper are byte-identical private moves, and one `cfg(test)` alias retains its proven consumer.
  Exact post-parsed-command/pre-table-discovery stage, prefix/suffix and three-field tuple parsing, whitespace
  trimming, comma/input ordering, name/OID decoding, two text columns, SQL-type display mapping, malformed or
  unsupported-OID fall-through, and writer are unchanged. Inherited PRODUCT-002 debt remains brittle generated-
  SQL parsing, host registry lookup and display-name decisions, typmod ignored after shape validation, and absent
  GPU-resident type metadata. Full protocol/driver/security/formatting gates, the focused helper test, PostgreSQL
  16 scenario 69, and independent audit are clean. The root is 5,578 lines and STRUCT-001DD is closed.
  The complete adjacent early table catalog family now executes from a new private 542-line `table_catalog`
  owner behind one stage-preserving delegate: two direct public-table discovery reads, normal and verbose exact/
  filtered psql table listings, and filtered relation-privilege listings. Five exact query constants, three
  parsers, table-name/OID/normal/verbose/privilege row builders, four compatibility size helpers, and the table ACL
  display helper are byte-identical moves. Only the filter record/fields gain parent-private visibility; they and
  fifteen `cfg(test)` wrappers serve proven existing consumers. Exact post-`\gdesc`/pre-index stage and internal
  branch order, equality/parser semantics, public-schema and relname filtering, table/view/materialized-view/
  sequence privilege coverage, name/OID sorting, comment/ACL/null behavior, size estimates, 1-/2-/4-/8-/6-column
  shapes, and writers are unchanged. Shared pattern matching and generic ACL rendering remain root-owned.
  Inherited PRODUCT-002 debt remains exact generated SQL matching, host catalog iteration/filtering/sorting and
  size estimation, public-only metadata, simplified ACL/policy fields, and absent GPU-resident system relations.
  Full protocol/driver/security/formatting gates, focused catalog/ACL tests, all 17 PostgreSQL 16 scenarios 04/06/
  13/14/17/18/19/34/35/37/75/79/80/326/333/334/341, and independent audit are clean. The root is 5,167 lines and
  STRUCT-001DE is closed.
  Three later direct table/class catalog reads now execute from the existing private 715-line `table_catalog`
  owner behind three additional stage-preserving delegates. Seven exact query/parser/row helpers are byte-
  identical private moves and six `cfg(test)` wrappers retain proven consumers. Exact post-description/pre-index,
  post-index/pre-sequence, and post-sequence/materialized-view stages preserve the interleaved route order;
  equality and `IN` parsing, requested-name deduplication, public-only filtering, name sorting with OID-bearing rows, fixed owner/
  schema/kind/persistence fields, 3-/5-/5-column shapes, and writers are unchanged. Inherited PRODUCT-002 debt
  remains exact routing, host table maps and CPU filtering/sorting, hard-coded public/postgres/kind/persistence
  metadata, and absent GPU-resident system relations. Full protocol/driver/security/formatting gates, focused
  catalog tests, PostgreSQL 16 scenarios 22/23/29, and independent audit are clean. The root is 5,074 lines and
  STRUCT-001DF is closed.
  The complete contiguous `information_schema.tables` family now executes from the existing private 1,016-line
  `table_catalog` owner behind one additional stage-preserving delegate. Six exact public-table, base-table,
  table-name `IN`, rich-table, exact-rich-table, and catalog-qualified-rich-table routes plus twelve query/parser/
  row helpers are byte-identical moves. Ten `cfg(test)` wrappers retain only proven existing consumers. Exact
  post-filtered-class/pre-column stage and internal branch order, equality/`IN`/catalog parser semantics,
  requested-name deduplication, public-only tables, name sorting, fixed postgres/public/base-table/insertable/
  typed fields and nulls, 3-/2-/3-/12-column shapes, writers, and fall-through are unchanged. Inherited
  PRODUCT-002 debt remains exact generated-SQL routing, host table maps and CPU filtering/sorting, hard-coded
  catalog/schema/type fields, and absent GPU-resident system relations. Full protocol/driver/security/formatting
  gates, focused catalog tests, PostgreSQL 16 scenarios 08/20/27/38/45/73/324, and independent audit are clean.
  The root is 4,861 lines and STRUCT-001DG is closed.
  The complete contiguous `information_schema.columns` family now executes from a new private 648-line
  `column_catalog` owner behind one stage-preserving delegate. Eleven exact table/all/discovery/`IN`/detail/UDT/
  rich/extended routes plus twenty-three query/parser/metadata/row helpers, including the separately located UDT
  helper, are byte-identical moves. Nineteen `cfg(test)` wrappers retain only proven existing consumers, and the
  shared column-display/default-format dependencies remain narrow and one-way. Exact post-table/pre-schemata
  stage and internal branch order, equality/`IN`/catalog parsers, requested-name deduplication, public tables,
  table-name and column ordering, missing-table empty behavior, default/type/UDT/numeric metadata and nulls,
  5-/4-/9-/14-column shapes, writers, and fall-through are unchanged. Inherited PRODUCT-002 debt remains exact
  generated-SQL routing, host table/column maps and CPU filtering/sorting, simplified nullability/length metadata,
  hard-coded catalog/schema fields, and absent GPU-resident system relations. Full protocol/driver/security/
  formatting gates, focused catalog/default tests, PostgreSQL 16 scenarios 08/15/16/25/26/30/31/39/44/74/304/
  319/320/321/324/332/336, and independent audit are clean. The root is 4,387 lines and STRUCT-001DH is closed.
  The four constraint/default catalog routes now execute from a new private 248-line `constraint_catalog` owner
  behind two stage-preserving delegates around the interleaved view-relation route: information-schema table/
  key constraints before the view route, and `pg_constraint`/`pg_attrdef` after it. Eight query/row helpers are
  byte-identical private moves and eight `cfg(test)` wrappers retain only proven existing consumers. Shared
  constraint-entry/type/contype and default-format dependencies remain narrow and one-way. Exact stage/internal
  order, public/index/check/foreign-key coverage, table/name/ordinal sorting, schema/type/contype/default fields,
  4-/5-/4-/4-column shapes, writers, and fall-through are unchanged. Inherited PRODUCT-002 debt remains exact
  generated-SQL routing, host constraint/index/table maps and CPU construction/sorting, simplified ordinal and
  constraint metadata, and absent GPU-resident system relations. Full protocol/driver/security/formatting gates,
  focused catalog/default tests, PostgreSQL 16 scenarios 40/41/42/304/307/308/319/320/321/322/324/332/338/339,
  and independent audit are clean. The root is 4,231 lines and STRUCT-001DI is closed.
  The three final direct `pg_catalog.pg_attribute` routes now execute from the existing private 843-line
  `column_catalog` owner behind one additional stage-preserving delegate. Six query-parser/row helpers are byte-
  identical private moves and six `cfg(test)` wrappers retain only proven existing consumers. Shared type-OID/
  type-size/type-display dependencies remain narrow and one-way. Exact post-direct-type/pre-session-fallback
  stage and internal branch order, regclass/public-table parsers, missing-relation `42P01` errors/messages,
  session-domain OIDs, declared type sizes/display, column iteration/ordinal order, fixed false not-null field,
  2-/4-/4-column shapes, writers, and fall-through are unchanged. Inherited PRODUCT-002 debt remains exact
  generated-SQL routing, host table/column/domain maps and CPU construction, simplified not-null metadata, and
  absent GPU-resident system relations. Full protocol/driver/security/formatting gates, focused attribute/catalog
  tests, PostgreSQL 16 scenarios 04/24/42/43/304/305/311/319/320/321/323/324/325/328/330/331/332/336, and
  independent audit are clean. The root is 4,095 lines and STRUCT-001DJ is closed.
  Eight obsolete post-parse pg-dump catalog branches are deleted after a line-by-line dominance proof: table OID,
  class metadata, index metadata, foreign-key metadata, view definition, attribute-default metadata, function
  metadata, and empty catalog queries. The earlier unconditional `try_execute_pg_dump_compat_statement` stage
  invokes the identical predicate/parser, column helper, row helper, writer, stream/session inputs and returns
  `Some` for every match before parsing, so none of the later copies was reachable. The change is exactly 52 root
  deletions; every canonical helper and existing consumer remains, and adjacent relation lookup/description code
  is untouched. Full protocol/driver/security/formatting and focused pg-dump/catalog gates, all eighteen
  PostgreSQL 16 pg-dump restore variants, the pg-dumpall globals restore, scenario 295, source/diff checks, and
  independent dominance audit are clean. The root is 4,043 lines and STRUCT-001DK is closed.
  The complete remaining psql `\d` relation-introspection family now executes from a new private 800-line
  `relation_introspection` owner behind one stage-preserving delegate. Fourteen branch arms and twenty-six query-
  parser/row helpers are byte-identical moves; fifteen `cfg(test)` wrappers retain only proven existing consumers.
  Shared relation-pattern, type display/storage, index/constraint/default/value dependencies remain narrow and
  one-way. Exact post-built-in-type/pre-direct-table stage and internal order, lookup regex/public filtering and
  per-kind sorting, relation flags, normal/verbose attribute default/comment/storage fields, index/check/foreign-
  key definitions and sorting, trigger/policy/statistics/inheritance empty rows, all fixed/null fields, column
  names/types/row shapes, writers, and fall-through are unchanged. Inherited PRODUCT-002 debt remains brittle
  generated-SQL parsing, host catalog maps and CPU filtering/sorting, synthetic/hard-coded metadata, incomplete
  psql object semantics, and absent GPU-resident system relations. Full protocol/driver/security/formatting gates,
  three focused catalog tests, PostgreSQL 16 scenarios 07/11/12/33/36/76/305/307/308/311/319/320/321/322/323/
  324/325/330/331/338/339/341, and independent audit are clean. The root is 3,376 lines and STRUCT-001DL is closed.
  Pg-dump relation metadata now lives in a new private 619-line `pg_dump_relation_metadata` owner: table OID,
  class, attribute, index, foreign-key, view-definition, dependency, and attribute-default metadata. Twenty exact
  helpers are parent-private for the sole production sibling, three helpers and the class-metadata record remain
  module-private, and one `cfg(test)` index-row wrapper retains its sole proven consumer. The `pg_dump_compat`
  executable body is byte-identical and only its imports are retargeted. Exact predicates/parsers, liveness/orphan
  filtering, OID synthesis, ACL/default/type/domain fields, definitions, sorting, column names/types/row shapes,
  and output are unchanged. Shared ACL/default/type, index/constraint/OID, foreign-key-definition and catalog-state
  dependencies remain narrow and one-way. Inherited PRODUCT-002 debt remains exact generated-SQL matching,
  host/synthetic metadata and OIDs, CPU filtering/sorting/construction, simplified catalog semantics, and absent
  GPU-resident system relations. Full protocol/driver/security/formatting and focused pg-dump/index gates, all
  eighteen PostgreSQL 16 pg-dump restore variants, pg-dumpall globals restore, scenario 295, and independent audit
  are clean. The root is 2,780 lines and STRUCT-001DM is closed.
  ACL display ownership is now complete in the existing private 1,084-line `acl_execution` owner. Fifteen exact
  relation/schema/database/tablespace/function/table rendering and privilege-letter helpers moved from the root;
  nine remain parent-private for proven sibling consumers and six are owner-private, with no new test alias.
  The original 240-line helper block is byte-identical after intended visibility normalization. Grantee ordering,
  PUBLIC empty-name rendering, privilege-letter order/case, bootstrap/default ACL strings, sequence-vs-relation
  defaults, array braces/newlines, empty/null behavior, sibling call sites, and output are unchanged. Inherited
  PRODUCT-002 debt remains simplified host ACL/membership/grant-option semantics, CPU metadata construction, and
  absent GPU-resident system relations. Formatting/diff checks, all-target check and clippy, four focused ACL
  tests, 71 library tests, 127 binary tests serial and 16-thread, tokio-postgres/SQLx smokes, the connection-
  security preflight, PostgreSQL 16 scenarios 18/19/68/333/334/341/342/347/349/350/351/352, all eighteen pg-dump
  restore variants, pg-dumpall globals restore, and independent audit are clean. The root is 2,543 lines and
  STRUCT-001DN is closed.
  Pg-dump compatibility ownership is now complete in the existing private 1,031-line `pg_dump_compat` owner.
  Nineteen exact helpers moved from the root: five sequence, six function, one bounded empty-catalog classifier,
  and seven domain helpers. Eight remain owner-private, ten retain parent-private names for `sql_prepare`, and the
  classifier retains its existing name through one `cfg(test)` parent import without a wrapper or new test alias.
  The compatibility prelude executable body is byte-identical. Exact predicates/parsers, synthetic metadata/OIDs,
  column names/types/row shapes, function/domain/type/ACL fields, sequence state/setval behavior, classification,
  sorting, empty/null behavior, route order, sibling call sites, and output are unchanged. Inherited PRODUCT-002
  debt remains exact host routing, CPU metadata/state construction, simplified catalog semantics, and absent GPU-
  resident system relations. Formatting/diff checks, all-target check and clippy, six focused pg-dump/function/
  sequence tests, 71 library tests, 127 binary tests serial and 16-thread, tokio-postgres/SQLx smokes, the connection-
  security preflight, PostgreSQL 16 scenarios 295/330/332/336/341/348/352, all eighteen pg-dump restore variants,
  pg-dumpall globals restore on a fresh dedicated port range, and independent audit are clean. The root is 1,993
  lines and STRUCT-001DO is closed.
  The legacy protocol root now satisfies the 2,000-line production envelope without an exception. Its remaining
  400-line `execute_statement` is the ordered private orchestration facade over the extracted family owners; state
  and shared type/helper ownership remain at the binary boundary. The protocol inventory disposition and the
  sequencing parent STRUCT-001AV are closed; further work belongs to PRODUCT-001/PRODUCT-002 rather than source-
  size remediation.
  STRUCT-001DP completed the required replication outlier packet. `crates/replication/src/lib.rs` is 20,164
  handwritten lines: 2,021 production plus an 18,143-line inline module with exactly 187 tests; it has no generated,
  feature-gated, or unsafe code. Production ownership separates into operational reports, RPC data/codec, TCP/TLS
  transport, progress/status/recovery invariants, stable replication/state-machine traits, and local/Raft owners.
  Engine consumers use only the local owner and traits; operational examples/scripts consume the public Raft/RPC/
  transport/report facade. I/O/rustls/certificate/timeouts are confined to transport, while the large April 2026
  co-change tail is a Raft snapshot-repair test family rather than production coupling. The accepted module/test
  dependency map and bounded test-family ranges are recorded in PLAN’s replication inventory; STRUCT-001DQ is the
  first exact leaf and multi-GPU remains deferred.
  STRUCT-001DQ isolated the exact TCP/TLS/mTLS AppendEntries transport body in a private 259-line `transport`
  owner. Seven public send/serve/configuration functions remain at the unchanged crate-root facade; seven private
  helpers plus the private stream trait now own TCP connection/accept, timeout/shutdown ordering, frame I/O,
  certificate/root/key parsing, client-auth configuration, and error construction. AppendEntries data, codec
  primitives, request application, and all Raft state remain in the root. The four exact transport tests live in
  a bounded 131-line included file and retain their original `tests::append_entries_transport_*` harness names.
  Both 187-test serial/16-thread runs, workspace all-target check, replication all-target strict clippy, cluster/
  multiprocess/service/channel-security smokes, targeted formatting/diff/reference checks, and independent
  re-audit are clean. The root is 19,785 lines, with 1,773 production lines before the test facade; STRUCT-001DR
  is the next behavior-preserving ownership leaf.
  STRUCT-001DR isolated the five operational cluster/transport/election/package/preflight report contracts in a
  private 208-line `operational` owner with only `Index`, `Role`, and `Term` dependencies. All derives,
  public fields/types, readiness predicates, operator strings/order, and five crate-root APIs are byte-identical.
  The exact three-node operational smoke test now lives in a bounded 208-line included file and retains its
  original fully qualified harness name. The focused test, both 187-test serial/16-thread runs, replication all-
  target check and strict clippy, cluster output smoke, targeted formatting/diff/reference checks, and independent
  audit are clean. The root is 19,374 lines with 1,571 production lines before the test facade; STRUCT-001DS is
  the next acyclic RPC ownership leaf.
  STRUCT-001DS isolated the four public RequestVote/AppendEntries records, both exact AppendEntries frame codecs,
  and all private framing primitives in a private 202-line `rpc` owner that depends only on `gpu_db_types`.
  The byte-identical `AppendEntriesRequest::apply_to` adapter remains at the root beside Raft orchestration, so
  RPC has no reverse consensus/transport dependency; all four public root paths and transport consumers remain
  unchanged. The two exact frame tests now live in a 55-line included file while retaining their original harness
  names; request-application and TCP loopback tests remain in the transport test owner. Focused frame/election
  tests, both 187-test serial/16-thread runs, replication all-target check and strict clippy, cluster/multiprocess/
  service/channel-security smokes, targeted formatting/diff/reference checks, and independent audit are clean.
  The root is 19,180 lines with 1,376 production lines before the test facade; STRUCT-001DT is next.
  STRUCT-001DT isolated four progress/status/recovery records, three invariant error enums, and their exact
  implementations in a private 410-line `progress` owner with only `Index`, `LogEntry`, `Role`,
  `SnapshotMeta`, and `Term` dependencies. One helper is narrowly `pub(super)` for its sole pre-existing root
  Raft-status consumer; the module remains private and all seven public root APIs are unchanged. Validation order/
  errors, saturating gap arithmetic, snapshot-identity checks, recovery projection, and methods are otherwise
  byte-identical. The exact contiguous 16-test invariant/projection block now lives in a bounded 666-line included
  file with all original harness names. Focused progress/status/recovery tests, both 187-test serial/16-thread
  runs, replication all-target check and strict clippy, cluster smoke, targeted formatting/diff/reference checks,
  and independent audit are clean. The root is 18,111 lines with 972 production lines; STRUCT-001DU is next.
  STRUCT-001DU isolated the exact `LocalReplicator` record, inherent behavior, and `LogReplicator`
  implementation in a private 294-line `local` owner with the unchanged crate-root re-export. Role/term
  transitions, indexing/compaction, commit/apply watermarks, snapshot identity/install, recovery/status
  projection, public methods, errors, and trait behavior are exact; only `entry_at` is narrowly `pub(super)`
  for its two pre-existing owner/parent consumers. Raft production is byte-identical. The selected exact 20-test
  local family now lives in a bounded 354-line included file with all original harness names. Focused local,
  snapshot, and progress tests, both 187-test serial/16-thread runs, replication all-target check and strict
  clippy, downstream engine all-target check, targeted formatting/diff/reference checks, and independent audit
  are clean. The root is 17,469 lines with 683 production lines; STRUCT-001DV is next.
  STRUCT-001DV externalized the exact contiguous 11-test baseline Raft/progress family into a bounded 272-line
  `tests/raft_baseline.rs` owner through an `include!` at the same parent position. Every body is byte-identical
  after removing only parent indentation, all compiled harness names remain `tests::*`, and ownership is unique;
  production, visibility, public APIs, and consensus behavior are untouched. Focused proposal, quorum, progress,
  lagging-follower, recovery, and snapshot-boundary tests, both 187-test serial/16-thread runs, replication
  all-target check and strict clippy, scoped formatting/diff/reference checks, and independent audit are clean.
  The root is 17,198 lines with 683 production lines; STRUCT-001DW is next.
  STRUCT-001DW externalized the exact contiguous 27-test replicator lifecycle/progress family into a bounded
  551-line `tests/replicator_lifecycle.rs` owner through an `include!` at the same parent position. Every body
  is byte-identical after parent deindent, all compiled harness names remain `tests::*`, and the Local/Raft
  wait tests coherently cover the root `LogReplicator::wait_committed` contract. Production, visibility, APIs,
  and semantics are untouched. Focused recovery, single-node, acknowledgement, role/tail, wait, snapshot, and
  progress tests, both 187-test serial/16-thread runs, replication all-target check and strict clippy, scoped
  formatting/diff/reference checks, and independent audit are clean. The root is 16,648 lines with 683
  production lines; STRUCT-001DX is next.
  STRUCT-001DX externalized the exact contiguous 28-test AppendEntries family into a bounded 1,029-line
  `tests/append_entries.rs` owner through an `include!` at the same parent position. Every body is byte-identical
  after parent deindent, all compiled harness names remain `tests::*`, and the owner coherently covers heartbeat
  commit, term/role and previous-entry validation, batch/conflict handling, newer-leader repair, and snapshot-
  boundary append invariants. Production, visibility, APIs, and semantics are untouched. Focused term, conflict,
  heartbeat, repair, and snapshot-boundary tests, both 187-test serial/16-thread runs, replication all-target
  check and strict clippy, scoped formatting/diff/reference checks, and independent audit are clean. The root is
  15,620 lines with 683 production lines; STRUCT-001DY is next.
  STRUCT-001DY externalized the exact contiguous 24-test snapshot-identity family into a bounded 2,133-line
  `tests/snapshot_identity.rs` owner through an `include!` at the same parent position. Formatting-normalized
  source compares exactly; only two long declarations rewrapped after parent deindent, while bodies, attributes,
  names, and semantics are unchanged. The owner coherently covers stale/incompatible installs, same/advanced-
  frontier identity, compatible suffixes, recovery bundle/gap projection, stale refresh, and role/newer-leader
  transitions. Focused snapshot/refresh/suffix/recovery/role tests, both 187-test serial/16-thread runs,
  replication all-target check and strict clippy, scoped formatting/diff/reference checks, and independent audit
  are clean. The root is 13,486 lines with 683 production lines; STRUCT-001DZ is next.
  STRUCT-001DZ externalized the exact contiguous seven-test advanced-snapshot repair family into a bounded
  913-line `tests/repair_advanced_snapshot.rs` owner through an `include!` at the same parent position.
  Formatting-normalized source compares exactly; only two long declarations rewrapped, while bodies, attributes,
  names, and semantics are unchanged. The owner coherently covers durable identity replacement, compatible fresh
  suffixes, same-frontier/second refresh, role discard, newer-leader rejection collapse, and commit/apply
  retirement. All seven exact tests, the broad repair filter, both 187-test serial/16-thread runs, replication
  all-target check and strict clippy, scoped formatting/diff/reference checks, and independent audit are clean.
  The root is 12,573 lines with 683 production lines; STRUCT-001EA is next.
  STRUCT-001EA externalized the exact contiguous seven-test second-refresh advanced-replacement repair family
  into a bounded 1,107-line `tests/repair_second_refresh.rs` owner through an `include!` at the same parent
  position. Formatting-normalized source compares exactly; only one long declaration rewrapped, while bodies,
  attributes, names, and semantics are unchanged. The owner covers later advanced replacement, replacement/
  refresh identity, role discard, commit/apply retirement, rejection collapse, and subsequent repair. All seven
  exact tests, both 187-test serial/16-thread runs, replication all-target check and strict clippy, scoped
  formatting/diff/reference checks, and independent audit are clean. The root is 11,466 lines with 683
  production lines; STRUCT-001EB is next.
  STRUCT-001EB externalized the exact contiguous ten-test post-rejection repair family into a cohesive 2,256-line
  `tests/repair_post_rejection.rs` owner through an `include!` at the same parent position. Parent-deindented
  source is byte-identical and rustfmt-clean; bodies, attributes, names, and semantics are unchanged. The owner
  covers identity refresh, role handoff, repeated rejection/later repair, commit/apply retirement, stale-install
  inertness, gap retirement, and clean repeated collapse. All ten exact tests, both 187-test serial/16-thread
  runs, replication all-target check and strict clippy, scoped formatting/diff/reference checks, and independent
  audit are clean. The file is below the 3,000-line required-analysis threshold. The root is 9,211 lines with
  683 production lines; STRUCT-001EC is next.
  STRUCT-001EC externalized the exact contiguous nine-test refresh-rejection repair cycle into a cohesive
  2,830-line `tests/repair_refresh_rejection.rs` owner through an `include!` at the same parent position.
  Parent-deindented source is byte-identical and rustfmt-clean; bodies, attributes, names, and semantics are
  unchanged. The owner covers later repair, role handoff, commit/apply retirement, subsequent repair/refresh
  identity, repeated collapse, and stale-install inertness. All nine exact tests, both 187-test serial/16-thread
  runs, replication all-target check and strict clippy, scoped formatting/diff/reference checks, and independent
  audit are clean. The file is 170 lines below the 3,000-line required-analysis threshold. The root is 6,382
  lines with 683 production lines; STRUCT-001ED is next.
  STRUCT-001ED externalized the exact contiguous eight-test deep refresh/rejection cycle into a cohesive
  2,838-line `tests/repair_deep_refresh.rs` owner through an `include!` at the same parent position. Parent-
  deindented source is byte-identical and rustfmt-clean; bodies, attributes, names, and semantics are unchanged.
  The owner covers the deepest later-repair cycle, role handoff, commit/apply retirement, subsequent refresh
  identity/gap retirement, and stale-install inertness. All eight exact tests, both 187-test serial/16-thread
  runs, replication all-target check and strict clippy, scoped formatting/diff/reference checks, and independent
  audit are clean. The file is 162 lines below the 3,000-line required-analysis threshold. The root is 3,545
  lines with 683 production lines; STRUCT-001EE is next.
  STRUCT-001EE externalized the exact contiguous six-test stale-install unwind family into a cohesive
  1,645-line `tests/repair_stale_unwind.rs` owner through an `include!` at the same parent position. Parent-
  deindented source is byte-identical and rustfmt-clean; bodies, attributes, names, and semantics are unchanged.
  The owner covers stale-install inertness at progressively unwound repair depths plus paired later-repair
  commit/apply retirement. All six exact tests, both 187-test serial/16-thread runs, replication all-target
  check and strict clippy, scoped formatting/diff/reference checks, and independent audit are clean. The root
  is 1,901 lines with 683 production lines, below the production-file analysis threshold; STRUCT-001EF is next.
  STRUCT-001EF externalized the exact final six-test advanced repair cleanup family into a cohesive 1,023-line
  `tests/repair_advanced_cleanup.rs` owner through the last `include!` in the unchanged parent test facade.
  Formatting-normalized source compares exactly; only two long function declarations rewrapped, while bodies,
  attributes, names, and semantics are unchanged. All six exact tests, both 187-test serial/16-thread runs,
  replication all-target check and strict clippy, scoped formatting/diff/reference checks, and independent
  audit are clean. The root is now 877 lines with 683 production lines. Its stable traits, public re-exports,
  Raft orchestration, and cross-type request-application seam are cohesive; forcing the previously sketched
  `raft` leaf would add artificial visibility/cycle risk. The replication outlier disposition is complete and
  STRUCT-001EG is next.
  STRUCT-001EG completed the required analysis packet for the 12,964-line handwritten
  `engine_residency.rs` outlier without changing runtime code. It contains 6,145 production/support lines and a
  6,819-line inline 76-test residency matrix; 132 file-touching commits co-change distinct expression, state,
  retained-read, lifecycle, commit, DML, execution, and resident-storage owners. PLAN now records the complete
  responsibility/caller/state/CUDA/history map, native-GPU publication and layout invariants, an acyclic seven-
  leaf production target plus seven bounded test families, visibility discipline, and exact structural versus
  device-changing gates. The disposition is decompose behind the stable private facade and public inherent
  `Engine` APIs. STRUCT-001EH is the first bounded pure test-ownership slice.
  STRUCT-001EH externalized the exact contiguous four-test payload layout/capacity/append family into a
  rustfmt-clean 386-line `tests/residency_payload.rs` owner through an `include!` at the same parent position.
  Parent-deindented source is byte-identical, all attributes and `capacity_payload_tests::*` names are unchanged,
  and expanding the include reconstructs the prior 12,964-line source exactly. The three ordinary exact tests,
  992-name/76-family inventory, moved ignored GPU gate 3× sequential and 2× concurrent with zero CUDA
  700/716/717, all-target check, scoped formatting/diff/reference checks, and independent audit are clean. The
  root is 12,579 lines; production, visibility, payload bytes, CUDA calls, and routes are untouched, so the report
  card was not applicable. The broad ordinary suites exposed a pre-existing CUDA 700 in nullable-text `LIKE`
  count, reproduced on detached pre-slice HEAD; it poisoned later tests and caused the observed serial/16-thread
  cascades. READ-004 subsequently closed that fault as described below.
  READ-004 eliminated that pre-existing nullable-text `LIKE` fault in the resident general predicate VM. The
  VM launched `gpu_db_resident_text_like_scalar_to_mask` with seven arguments even though the PTX ABI requires
  eight: omitting `text_bytes_limit` shifted the token pointer/count/row-count fields and left the output-mask
  pointer absent, deterministically raising CUDA 700. `ExprStep::TextLikeMask` now carries the resident text
  blob length, applies the same overflow-safe offsets/blob window validation as the standalone LIKE launcher,
  and passes the exact eight-argument ABI; matching and nullable validity-mask 3VL remain entirely on-device.
  The existing CUDA LIKE oracle now also executes the VM route for every pattern and proves an oversized blob
  fails before launch. Nullable/non-null bridge/general/VM gates passed three sequential and two concurrent
  rounds with zero CUDA 700/716/717; the full 992-test engine inventory passed serial and 16-thread (505 passed,
  487 ignored); execution and engine all-target checks passed; execution strict clippy passed, and the changed
  engine line introduced no new lint finding. The direct roofline measured
  1,467/1,440 GB/s in-/out-of-L2 `sum_i32`, 1,288/1,463 GB/s COUNT, and 347/252/1,678 M-elem/s sort/join/grouped.
  The canonical card was likewise clean: in-/out-of-L2 rooflines 1,429/1,444 GB/s and the production batched
  point-read route reached 251.3M/256.6M lookups/s at batch 65,536 with 135/128 us p50. Independent audit found
  no host fallback, ABI/lifetime/bounds defect, semantic drift, or regression.
  QUALITY-001 restored the strict package lint baseline in two bounded commits. The WAL slice removed its two
  transitive findings; the engine slice resolved all 54 package-local findings with mechanical rewrites, named
  parameter/tuple shapes, and narrow documented exceptions at only allocation-sensitive or benchmark/test
  boundaries. No host relational fallback or product-path allow was added. One baseline shard test was stale
  after the production auto-admission flip; the same failure reproduced on pre-QUALITY commit `b2e0fc45`, and
  its setup now disables auto-admit so the test continues to exercise explicit post-load admission. Engine
  strict clippy with dependencies and all targets, all-target check, WAL tests, and both serial/16-thread
  992-test engine runs are green (505 passed, 487 ignored). Five residency/text/binary/deletion/streaming GPU
  routes passed three sequential and two concurrent rounds with zero CUDA 700/716/717. The direct roofline
  measured 1,468/1,448 GB/s in-/out-of-L2 `sum_i32`, 1,292/1,459 GB/s COUNT, and 349/258/1,678 M-elem/s
  sort/join/grouped. The canonical card measured 1,476/1,451 GB/s rooflines and 65,536-batch production point
  reads at 246.5M/255.3M lookups/s with p50 136/130us. Independent audit found no semantic, ownership, API,
  scheduler, or GPU-path drift. This unblocked STRUCT-001EI.
  STRUCT-001EI externalized the exact contiguous five-test baseline shard family into the rustfmt-clean 429-line
  `tests/residency_shard_baseline.rs` owner through an `include!` at the same parent position. Normalizing the
  prior parent-indented source leaves no difference beyond one rustfmt line wrap; all bodies, attributes, order,
  and `capacity_payload_tests::*` names are unchanged. The engine root is 12,154 lines, and the 992-test/76-
  family inventories remain exact. All five GPU gates passed 15 sequential and 10 concurrent invocations with
  zero CUDA 700/716/717; both ordinary suite modes passed 505/487, and all-target check, restored strict clippy,
  scoped formatting/diff/reference checks, and independent audit are clean. Production code, visibility,
  residency, routing, and architecture are untouched, so the report card was not applicable.
  STRUCT-001EJ externalized the exact eight-test sparse `deleted_by` visibility family into the rustfmt-clean
  524-line `tests/residency_sparse_visibility.rs` owner through an `include!` at the same parent position. The
  parent-deindented source comparison is byte-exact, all bodies/attributes/names/order are unchanged, and the
  five shared sparse-region helpers remain at the parent facade for later slices. The engine root is 11,631
  lines, and the 992-test/76-family inventories remain exact. The eight GPU gates passed 24 sequential and 16
  concurrent invocations with zero CUDA 700/716/717; both ordinary modes passed 505/487, and all-target check,
  strict clippy, scoped formatting/diff/reference checks, and independent audit are clean. No production or
  architectural behavior changed, so the report card was not applicable.
  STRUCT-001EK externalized the exact seven-test SQL DELETE/UPDATE and `created_by` visibility family plus its
  sole `device_row_id_for` helper into the rustfmt-clean 748-line `tests/residency_update_visibility.rs` owner at
  the same parent position. Normalized comparison differs only by one rustfmt wrap; all bodies, attributes,
  names, and order remain unchanged, while shared sparse-region helpers stay at the parent facade. The engine
  root is 10,886 lines, and the 992-test/76-family inventories remain exact. Seven GPU gates passed 21 sequential
  and 14 concurrent invocations with zero CUDA 700/716/717; both ordinary modes passed 505/487, and all-target
  check, strict clippy, scoped formatting/diff/reference checks, and independent audit are clean. No production
  or architectural behavior changed, so the report card was not applicable.
  STRUCT-001EL externalized the exact seven-test cross-shard primary-key index/cache/route family into the
  rustfmt-clean 568-line `tests/residency_pk_index.rs` owner at the same parent position. Normalized comparison
  differs only by two rustfmt closure layouts; all bodies, attributes, names, and order remain unchanged, no
  helpers moved, and the adjacent NULL/3VL test remains at the parent boundary. The engine root is 10,317 lines,
  and the 992-test/76-family inventories remain exact. Seven GPU gates passed 21 sequential and 14 concurrent
  invocations with zero CUDA 700/716/717; both ordinary modes passed 505/487, and all-target check, strict clippy,
  scoped formatting/diff/reference checks, and independent audit are clean. No production cache, route, or
  architectural behavior changed, so the report card was not applicable.
  STRUCT-001EM externalized the exact three-test sharded NULL/filter/join route-parity family into the rustfmt-
  clean 159-line `tests/residency_route_parity.rs` owner at the same parent position. Normalized comparison
  differs only by one rustfmt wrap; all bodies, attributes, names, and order remain unchanged, and the adjacent
  A2 device-resolve test remains at the parent boundary. The engine root is 10,159 lines, and the 992-test/76-
  family inventories remain exact. Three GPU gates passed nine sequential and six concurrent invocations with
  zero CUDA 700/716/717 and explicitly reported GPU execution for the filtered shapes; both ordinary modes
  passed 505/487, and all-target check, strict clippy, scoped formatting/diff/reference checks, and independent
  audit are clean. No production predicate, join, route, or architectural behavior changed, so the report card
  was not applicable.
  STRUCT-001EN externalized the exact six-test device-resolve and core elision lifecycle/concurrency family into
  the rustfmt-clean 611-line `tests/residency_elision_core.rs` owner at the same parent position. Normalized
  comparison differs only in two rustfmt-aligned array comments; all bodies, attributes, names, and order remain
  unchanged, no helper moved, and the adjacent DATE/INT2 test remains at the parent boundary. The engine root is
  9,549 lines, and the 992-test/76-family inventories remain exact. Six GPU gates passed 18 sequential and 12
  concurrent invocations with zero CUDA 700/716/717; both ordinary modes passed 505/487, and all-target check,
  strict clippy, scoped formatting/diff/reference checks, and independent audit are clean. No production
  resolver, elision, concurrency, or architectural behavior changed, so the report card was not applicable.
  STRUCT-001EO externalized the exact four-test DATE/INT2 and INT8 shard/elision family into the byte-exact,
  rustfmt-clean 426-line `tests/residency_type_coverage.rs` owner at the same parent position. All bodies,
  attributes, names, and order remain unchanged, and the adjacent device-write locate test remains at the parent
  boundary. The engine root is 9,124 lines, and the 992-test/76-family inventories remain exact. Four GPU gates
  passed 12 sequential and eight concurrent invocations with zero CUDA 700/716/717; both ordinary modes passed
  505/487, and all-target check, strict clippy, scoped formatting/diff/reference checks, and independent audit
  are clean. No production encoding, append, elision, or architectural behavior changed, so the report card was
  not applicable.
  STRUCT-001EP externalized the exact three-test device locate/reinsert/versioned-scan family into the byte-
  exact, rustfmt-clean 214-line `tests/residency_device_locate.rs` owner at the same parent position. All bodies,
  attributes, names, and order remain unchanged, and the adjacent text-column test remains at the parent
  boundary. The engine root is 8,911 lines, and the 992-test/76-family inventories remain exact. Three GPU gates
  passed nine sequential and six concurrent invocations with zero CUDA 700/716/717; both ordinary modes passed
  505/487, and all-target check, strict clippy, scoped formatting/diff/reference checks, and independent audit
  are clean. No production locate, reinsert, scan, or architectural behavior changed, so the report card was not
  applicable.
  STRUCT-001EQ externalized the exact five-test text/numeric/bool/b128 and grouped/versioned residency-read
  family into the rustfmt-clean 448-line `tests/residency_wide_type_reads.rs` owner at the same parent position.
  Normalized comparison differs only by one rustfmt brace-line collapse; all bodies, attributes, names, and
  order remain unchanged, and the adjacent wave-batch validation test remains at the parent boundary. The
  engine root is 8,463 lines, and the 992-test/76-family inventories remain exact. Five GPU gates passed 15
  sequential and 10 concurrent invocations with zero CUDA 700/716/717; both ordinary modes passed 505/487, and
  all-target check, strict clippy, scoped formatting/diff/reference checks, and independent audit are clean. No
  production type, rehydration, grouping, read-route, or architectural behavior changed, so the report card was
  not applicable.
  STRUCT-001ER externalized the exact five-test wave validation/duplicate-race/multi-writer family into the
  byte-exact, rustfmt-clean 407-line `tests/residency_elision_waves.rs` owner at the same parent position. All
  bodies, attributes, names, and order remain unchanged, and the adjacent vacuum test remains at the parent
  boundary. The engine root is 8,057 lines, and the 992-test/76-family inventories remain exact. Five GPU gates
  passed 15 sequential and 10 concurrent invocations with zero CUDA 700/716/717; both ordinary modes passed
  505/487, and all-target check, strict clippy, scoped formatting/diff/reference checks, and independent audit
  are clean. No production validation, elision, write-wave, concurrency, or architectural behavior changed, so
  the report card was not applicable.
  STRUCT-001ES externalized the exact six-test vacuum/gather/multi-row DML/concurrency/materialization family
  into the rustfmt-clean 695-line `tests/residency_maintenance_materialization.rs` owner at the same parent
  position. Normalized comparison differs only by one rustfmt assertion collapse; all bodies, attributes, names,
  and order remain unchanged, and the adjacent A3 validation test remains at the parent boundary. The engine
  root is 7,360 lines, and the 992-test/76-family inventories remain exact. Six GPU gates passed 18 sequential
  and 12 concurrent invocations with zero CUDA 700/716/717; both ordinary modes passed 505/487, and all-target
  check, strict clippy, scoped formatting/diff/reference checks, and independent audit are clean. No production
  vacuum, gather, DML, materialization, or architectural behavior changed, so the report card was not applicable.
  STRUCT-001ET externalized the exact three-test A3/A2/A1 validation/update-chain/row-identity family into the
  rustfmt-clean 377-line `tests/residency_identity_validation.rs` owner at the same parent position. Normalized
  comparison differs only in rustfmt comment and panic/closure layouts; all bodies, attributes, names, and order
  remain unchanged, and the adjacent mixed-type shard test remains at the parent boundary. The engine root is
  6,985 lines, and the 992-test/76-family inventories remain exact. Three GPU gates passed nine sequential and
  six concurrent invocations with zero CUDA 700/716/717; both ordinary modes passed 505/487, and all-target
  check, strict clippy, scoped formatting/diff/reference checks, and independent audit are clean. No production
  validation, update-chain, row-identity, or architectural behavior changed, so the report card was not
  applicable.
  STRUCT-001EU externalized the exact seven-test mixed-type/NULL/3VL/batched-point/binary/deletion-gate family
  into the rustfmt-clean 625-line `tests/residency_sharded_point_reads.rs` owner at the same parent position.
  Normalized comparison differs only in three rustfmt gather-call layouts; all bodies, attributes, names, and
  order remain unchanged, and the adjacent ordinary capacity-padding test remains at the parent boundary. The
  engine root is 6,355 lines, and the 992-test/76-family inventories remain exact. Seven GPU gates passed 21
  sequential and 14 concurrent invocations with zero CUDA 700/716/717; both ordinary modes passed 505/487, and
  all-target check, strict clippy, scoped formatting/diff/reference checks, and independent audit are clean. The
  inherited CPU-pinned mixed-type oracle remains test-only; no production route or architectural behavior
  changed, so the report card was not applicable.
  STRUCT-001EV externalized the final three-test capacity/open-payload/residency-budget family into the rustfmt-
  clean 87-line `tests/residency_capacity_budget.rs` owner immediately before the parent module close. Normalized
  comparison differs only in two rustfmt binding layouts; all bodies, attributes, names, and order remain
  unchanged. Both focused ordinary tests passed, and the budget GPU gate passed three sequential invocations and
  two concurrent rounds with zero CUDA 700/716/717. The engine root is 6,267 lines, both ordinary modes pass
  505/487, and the exact 992-test/76-family inventories, all-target check, strict clippy, scoped formatting/diff/
  reference checks, and independent audit are clean. No production capacity, payload, budget, or architectural
  behavior changed, so the report card was not applicable. All 76 residency tests now live in bounded owners.
  STRUCT-001EW then moved the normalized-exact typed payload/key/open-append production owner into the
  rustfmt-clean 806-line `engine_residency/payload.rs` child, reducing the facade root to 5,478 lines. Comparing
  the former root range with the child finds exactly one required change: `AppendCreatedBy::stamps_for` is now
  the narrow `pub(super)` needed by its existing parent consumer. Explicit crate-visible re-exports preserve
  every proven facade path, while the unconsumed `RelationalDevicePayload` alias and `fnv1a_bytes` helper remain
  private to the leaf. Five focused ordinary tests passed; four representative GPU routes passed 12 sequential
  and eight concurrent invocations with zero CUDA 700/716/717. Both ordinary modes pass 505/487, the exact
  992-test/76-family inventories are unchanged, and all-target check, strict clippy, scoped formatting/diff/
  reference checks, and independent audit are clean. Payload bytes/offsets/NULL layouts, key codecs,
  fingerprints, MVCC fills, append ordering, and device behavior are source-identical, so the report card was
  not applicable.
  STRUCT-001EX then moved the six snapshot construction/admission/publication methods into the rustfmt-clean
  691-line `engine_residency/admission.rs` child, reducing the facade root to 4,793 lines. The moved methods are
  normalized-exact: rustfmt changed four layouts, and only the three methods with proven existing parent
  consumers gained narrow `pub(super)` visibility; all public/crate APIs and call sites are unchanged. Six
  focused ordinary tests passed, and four admission-sensitive GPU routes passed 12 sequential plus eight
  concurrent invocations with zero CUDA 700/716/717. Both ordinary modes pass 505/487, exact 992-test/76-family
  inventories are unchanged, and all-target check, strict clippy, scoped formatting/diff/reference checks, and
  independent audit are clean after correcting the two child-module ownership comments. Allocation-before-
  eviction, deterministic admission, payload/layout/NULL/MVCC semantics, publication order, errors, and device
  behavior are source-equivalent, so the report card was not applicable.
  STRUCT-001EY then moved the byte-exact 60-method feature-policy, elision-eligibility, shard-sizing, and
  telemetry owner into the rustfmt-clean 624-line `engine_residency/policy.rs` child, reducing the facade root
  to 4,176 lines. Four focused ordinary policy/telemetry tests passed, and four policy-sensitive GPU routes
  passed 12 sequential plus eight concurrent invocations with zero CUDA 700/716/717. Both ordinary modes pass
  505/487, exact 992-test/76-family inventories are unchanged, and all-target check, strict clippy, scoped
  formatting/diff/reference checks, and independent audit are clean. Every method name/visibility, default,
  atomic ordering, predicate, cache/state mutation, and counter source is source-identical; there is no new
  dependency cycle or CPU-first path, so the report card was not applicable.
  STRUCT-001EZ then moved the normalized-exact eight-method append/rollover, sparse-version stamping,
  row-identity, tombstone, and fused-apply owner into the rustfmt-clean 1,230-line
  `engine_residency/mutation.rs` child, reducing the facade root to 2,952 lines. Rustfmt joined one offset
  expression, and only `stamp_created_by_resident_shard_slots` gained the narrow `pub(super)` required by its
  existing parent differential; public/crate APIs and production callers are unchanged. Four focused ordinary
  gates passed, and five mutation-sensitive GPU routes passed 15 sequential plus ten concurrent invocations
  with zero CUDA 700/716/717. Both ordinary modes pass 505/487, exact 992-test/76-family inventories are
  unchanged, and all-target check, strict clippy, scoped formatting/diff/reference checks, and independent audit
  are clean. Values→birth/identity stamps→row-count/HWM/zone-map publication, sparse MVCC regions, fused apply,
  tombstone ordering, offsets, launches, counters, and fail-safe paths are source-equivalent, so the report card
  was not applicable. STRUCT-001FA is the next production ownership slice.
  During STRUCT-001FA validation, `gpu_inner_join_catalog_relations_transient_payload` deterministically failed
  with CUDA 700 while a known-safe append route remained healthy. A detached worktree at pre-FA HEAD
  `43134442` reproduced the same CUDA 700, proving the ownership move did not introduce the fault; READ-005 is
  promoted ahead of FA closure to repair the native GPU catalog-join path without a host fallback. READ-005 is
  now closed: the compound predicate VM's `TextEqMask` and `TextCmpMask` launchers had retained the old CUDA ABI
  after the kernels gained a bounded `text_bytes_limit`, shifting the needle and every later parameter and
  causing the illegal access. Both bytecodes now carry descriptor `bytes_len`, validate offsets/blob windows
  before launch, and pass the exact bounded-text ABI; no catalog, join, or host relational fallback was added.
  A focused 4-mod-8 tiny-payload GPU regression proves equality, ordering, and pre-launch OOB rejection. The
  catalog target passed three sequential and two concurrent runs with zero CUDA 700/716/717; adjacent per-side
  WHERE, NULL-key 3VL, text-key, and catalog-metadata join gates passed. Both 992-test engine modes passed
  505/487, workspace all-target check and strict execution/engine clippy are clean, and independent audit found
  no issue. The direct roofline remained at baseline ratios (in-L2 `equal_any` 0.43, count compare 0.90, between
  0.45, ordered project 0.12; sort/join/grouped 349/257/1678 M-elem/s). The canonical two-layer/two-regime card
  also completed: raw out-of-L2 roofline 1,441 GB/s; point-read lpb-index p50/throughput at batch 32 was
  20us/1.52M lookups/s in both regimes, and batch 65,536 was 1,625us/37.18M in-L2 versus 1,570us/38.60M
  out-of-L2. STRUCT-001FA then closed the source-equivalent nine-method auto-admit/transient/benchmark-install
  owner in the 765-line `engine_residency/transient.rs` child, reducing the facade root to 2,188 lines. The only
  visibility change is the required narrow `pub(super)` bridge for `relational_residency_device_memory`; two
  final shard-install calls were rustfmt-wrapped. Five focused ordinary admission/proof gates passed. The sync
  transient catalog route passed its three sequential/two concurrent READ-005 matrix; five additional auto-admit,
  async transient, installer fail-safe, and sharded installer/read routes passed 15 sequential plus ten concurrent
  invocations with zero CUDA 700/716/717. Both ordinary engine modes passed 505/487 with the exact 992-test
  inventory; all-target check, strict clippy, dependency-boundary/scoped source/reference/format checks, and
  independent audit are clean. Runtime, layout, admission, residency, transient, and result-path behavior are
  unchanged, so the report card was not applicable. STRUCT-001FB then isolated the normalized-exact 14-method
  vacuum/churn, serialized rehydration, int4 identity resolution, host-store reconciliation, and typed device
  gather owner in the 607-line `engine_residency/maintenance.rs` child, reducing the facade root to 1,590 lines.
  Audit found and the slice corrected one ownership cycle by retaining the byte-exact shared
  `reset_tombstone_churn` helper at the parent facade; all other moved source reconstructs HEAD exactly and no
  visibility changed. Six vacuum/gather/rehydration routes passed 18 sequential plus 12 concurrent GPU
  invocations; after the placement correction, the minimal four-route matrix passed another 12 sequential plus
  eight concurrent invocations, all with zero CUDA 700/716/717. Both ordinary modes passed 505/487 with the
  exact 992-test inventory; all-target check, strict clippy, dependency-boundary/scoped source/reference/format
  checks, and independent re-audit are clean. Lock/catalog/snapshot, NULL/MVCC/identity, repair, and fail-loud
  behavior are source-identical, so the report card was not applicable. STRUCT-001FC then isolated the byte-
  exact nine-method warmup/maintenance-policy, cache-state readiness, single/sharded route-planning, rejection-
  evidence, and status owner in the rustfmt-clean 988-line `engine_residency/routes.rs` child, reducing the
  facade root to 609 lines and completing its disposition. Eight focused ordinary gates passed. Four
  representative single/sharded GPU routes passed 12 sequential plus eight concurrent invocations with zero
  CUDA 700/716/717; both ordinary modes passed 505/487 with the exact 992-test inventory. All-target check,
  strict clippy, dependency-boundary/scoped source/reference/format checks, and independent audit are clean.
  Route decisions, reasons, estimates, counters, status, fallback, runtime, layout, and result behavior are byte-
  identical, so the report card was not applicable.
  STRUCT-001FD then isolated the normalized-exact join-coordinate window/rank/shift owner in the rustfmt-clean
  1,076-line `join_window.rs` child, reducing the execution root to 17,032 lines. The only visibility change is
  the narrow `pub(super)` window-launcher bridge required by the existing root sort owner; crate-root public
  types and inherent APIs are unchanged. The three core GPU gates passed nine sequential plus six concurrent
  invocations with zero CUDA 700/716/717. Both execution modes passed 48/78, both engine modes passed 505/487,
  and workspace all-target check, strict execution/engine clippy, dependency/scoped source/reference/format
  checks, and independent audit are clean. PTX, ABI, geometry, partition/peer/NULL, shift, window, rank, error,
  synchronization, allocation, and bounded-readback behavior are source-equivalent, so the report card was not
  applicable.
  STRUCT-001FE then isolated the normalized-exact device materialization/concatenation owner in the 904-line
  `join_materialize.rs` child and bounded final fixed/bool/text projection in the 782-line
  `join_projection.rs` child, reducing the execution root to 15,921 lines. A pre-edit size simulation rejected
  one 1,668-line leaf; the final leaves use exact explicit imports with no visibility change or cycle. Four
  direct/engine/streaming GPU gates passed 12 sequential plus eight concurrent invocations with zero CUDA
  700/716/717. Both execution modes passed 48/78, both engine modes passed 505/487, and workspace all-target
  check, strict execution/engine clippy, dependency/scoped source/reference/format checks, and independent re-
  audit are clean. All five PTX blobs, APIs, layouts, validation, geometry, NULL/OUTER/type/empty/error/
  allocation/ownership/readback behavior are source-equivalent, so the report card was not applicable.
  STRUCT-001FF then isolated the normalized-exact stable device-coordinate sort method and merge launcher/PTX in
  the 417-line `join_sort.rs` child, reducing the execution root to 15,542 lines. The shared
  `CudaJoinOrderKey` remains uniquely at the root facade to avoid a cycle; the former root window-launcher
  bridge is now the explicit one-way `join_sort -> join_window` dependency, with no new visibility. Three
  direct/engine/streaming GPU sort gates passed nine sequential plus six concurrent invocations with zero CUDA
  700/716/717. Both execution modes passed 48/78, both engine modes passed 505/487, and workspace all-target
  check, strict execution/engine clippy, dependency/scoped source/reference/format checks, and independent audit
  are clean. Rust/PTX, comparator, stable-tie, NULL/OUTER, direction, descriptor, context, merge, geometry,
  synchronization, allocation, error, and small/window behavior are source-equivalent, so the report card was
  not applicable.
  STRUCT-001FG then isolated the normalized-exact GPU OUTER-join match bitmap and unmatched-coordinate
  completion owner in the rustfmt-clean 507-line `join_outer.rs` child, reducing the execution root to 15,144
  lines. The bitmap fields use narrow `pub(super)` visibility so the unchanged parent fixed-join launcher can
  retain its prior effective access; the crate-root public type and inherent APIs are unchanged, with no sibling
  dependency or cycle. Five direct/LEFT/RIGHT/FULL/type-wide/streaming GPU gates passed 15 sequential plus ten
  concurrent invocations with zero CUDA 700/716/717. Both execution modes passed 48/78, both engine modes passed
  505/487, and workspace all-target check, strict execution/engine clippy, dependency/scoped source/reference/
  format checks, and independent audit are clean. Match allocation/zeroing, atomic N:N collapse, two-pass
  complement cardinality, bounded scalar D2H, accumulated coordinate D2D copy, OUTER `u32::MAX` pads,
  validation, geometry, synchronization, errors, and allocation accounting are source-equivalent, so the report
  card was not applicable.
  STRUCT-001FH then isolated the normalized-exact device-resident join-coordinate identity and post-join filter
  owner in the rustfmt-clean 428-line `join_filter.rs` child, reducing the execution root to 14,761 lines.
  Shared `CudaJoinCoordinatesU32` and `CudaPredicateMaskI32` contracts remain uniquely at the root because
  fixed-join, expression, and other coordinate owners consume them; the child uses explicit imports with no
  visibility bridge, sibling dependency, or cycle. Five direct/OUTER-WHERE/Kleene/real-NULL-versus-pad/
  streaming GPU gates passed 15 sequential plus ten concurrent invocations with zero CUDA 700/716/717. Both
  execution modes passed 48/78, both engine modes passed 505/487, and workspace all-target check, strict
  execution/engine clippy, dependency/scoped source/reference/format checks, and independent audit are clean.
  Identity eligibility, real/pad mask descriptors, descriptor H2D, two-pass coordinate compaction, bounded
  scalar D2H, coordinate D2D, validation, empty paths, ordering, geometry, synchronization, errors, and
  allocation accounting are source-equivalent, so the report card was not applicable.
  STRUCT-001FI then isolated the normalized-exact accumulated fixed/text/composite coordinate join in the
  rustfmt-clean 1,309-line `join_fixed.rs` child, reducing the execution root to 13,489 lines. Shared key,
  coordinate, and predicate contracts remain uniquely root-owned, and the existing root re-export supplies the
  match-bitmap contract without a sibling path; the private child adds no visibility bridge or cycle. Five
  direct/typed-NULL/N:N-text-numeric/streaming GPU gates passed 15 sequential plus ten concurrent invocations
  with zero CUDA 700/716/717. Both execution modes passed 48/78, both engine modes passed 505/487, and workspace
  all-target check, strict execution/engine clippy, dependency/scoped source/reference/format checks, and
  independent audit are clean. Rust/PTX, same-context and descriptor validation, 4/8/16/text/composite keys,
  eligibility and NULL exclusion, hash build/probe, owned/persistent match marking, INNER/OUTER count-and-emit,
  padding, bounded scalar readback, emitted-count verification, geometry, ordering, errors, synchronization,
  and allocation lifetime are source-equivalent, so the report card was not applicable.
  STRUCT-001FJ then isolated the normalized-exact host-staged unique/N:N int/text hash-join benchmark/reference
  owner in the rustfmt-clean 1,390-line `staged_hash_join.rs` child, reducing the execution root to 12,112
  lines. `HashJoinOutcome` remains root-re-exported; the only non-test consumer is the standard Layer-1
  roofline example, and a stale transient-residency comment now names the actual production resident-coordinate
  API. Four direct GPU families passed 12 sequential plus eight concurrent invocations with zero CUDA
  700/716/717. Both execution modes passed 48/78, both engine modes passed 505/487, and workspace all-target
  check, strict execution/engine clippy, dependency/scoped source/reference/format checks are clean. Extraction
  audit found no moved-source/PTX/API regression and promoted the inherited undersized-validity device-read
  hazard as **STRUCT-001FK**; source-equivalent FJ itself did not require the report card.
  STRUCT-001FK then enforced one exact staged validity contract before every early return, context access,
  allocation, or launch across the unique/N:N int/text families: `None` means all valid and every present
  bitmap contains exactly `row_count.div_ceil(32)` words, including exactly zero words for zero rows. The new
  retained GPU gate rejects undersized, empty-for-nonempty, and oversized inputs and proves positive NULL
  exclusion plus the zero-row boundary across all four APIs. It and the existing four-family matrix passed 15
  sequential plus ten concurrent invocations without device faults. Both execution modes passed 48/79, both
  engine modes passed 505/487, workspace all-target check, strict execution/engine clippy, dependency/scoped
  format/diff checks, and independent re-audit are clean. The 1,403-line child remains within policy. Layer-1
  was stable: staged `hash_join_inner_i64` moved from 256.0 to 257.8 M-element/s (+0.70%) and the resident
  payload-join control from 6359.4 to 6313.5 M-element/s (-0.72%).
  STRUCT-001FL then isolated the shared payload-key/order-key descriptors and opaque device-coordinate owner
  in the rustfmt-clean 85-line `join_contract.rs` child, reducing the execution root to 12,035 lines. The
  private module re-exports all three unchanged crate-root public names; narrow `pub(super)` coordinate fields
  and test readback preserve their former root-private effective scope. `CudaPredicateMaskI32` remains with
  expression execution. Direct composite-coordinate, production three-way SQL, and over-budget streaming GPU
  joins passed nine sequential plus six concurrent invocations with zero device faults. Both execution modes
  passed 48/79, both engine modes passed 505/487, workspace all-target check, strict execution/engine clippy,
  dependency/scoped source/reference/format checks, and independent audit are clean. Type layout, derives,
  accessors, unsafe test readback, external consumers, runtime, kernels, synchronization, allocations, and
  result behavior are source-equivalent, so the report card was not applicable.
  STRUCT-001FM then isolated predicate-mask ownership and execution in the rustfmt-clean 385-line
  `predicate_mask.rs` child, reducing the execution root to 11,682 lines. The stable crate-root type and five
  resident APIs remain unchanged and opaque externally; narrow `pub(super)` fields preserve fixed/filter join
  consumers, while one root-private compact import preserves exactly nine typed-filter call sites without a
  cycle. Direct predicate VM, OUTER-WHERE 3VL, and over-budget streaming-filter GPU routes passed nine
  sequential plus six concurrent invocations with zero device faults. Both execution modes passed 48/79, both
  engine modes passed 505/487, workspace all-target check, strict execution/engine clippy, dependency/scoped
  source/reference/format checks, and independent audit are clean. Both PTX blobs, symbols/arguments,
  arithmetic-VM calls, probe scope, retained-buffer ownership transfer, context/allocation/launch/
  synchronization, empty/error behavior, and ordered results are source-equivalent, so the report card was not
  applicable.
  STRUCT-001FN then isolated the normalized-exact resident postfix expression VM in the rustfmt-clean 1,069-line
  `expression_vm.rs` child and moved its shared checked text-window validation into the 41-line neutral
  `resident_window.rs` leaf, reducing the execution root to 10,624 lines. The neutral leaf removes the otherwise
  hidden `expression_vm -> resident_filter -> predicate_mask -> expression_vm` cycle while preserving the stable
  crate-root `ExprStep`/`ResidentElemType` re-exports and one root-private runner import. The i32/i64/i128/varlen/
  production GPU matrix passed 15 sequential plus ten concurrent invocations without device faults. Both
  execution modes passed 48/79, both engine modes passed 505/487, workspace all-target check, strict execution/
  engine clippy, dependency/scoped source/reference/format checks, and independent audit are clean. The PTX bytes,
  25 referenced symbols, typed/scalar ABIs, dispatch, stack/lease/overflow, uploads/readback, synchronization,
  error, and result behavior are source-equivalent, so the report card was not applicable. Audit found inherited
  `ConstMask` GPU-native debt—its docs claim device fill while it builds and uploads an O(n) host vector—now owned
  by STRUCT-001FO.
  STRUCT-001FO then removed that host materialization: constant masks now use `cuMemsetD32Async` on a leased,
  individually synchronized pooled stream, filling exactly `n` i32 words with FALSE=0 or TRUE=1 before the output
  lease can be reused. The symbol is loaded only for programs containing `ConstMask`; every other VM branch keeps
  its former driver contract. A permanent two-cache Layer-1 line now measures the synchronized on-device fill
  without compaction or result D2H. Direct before/after moved IN-L2 from 10,106us/3.3 GB/s to 37us/905.1 GB/s
  (~274x) and OUT-OF-L2 from 80,271us/3.3 GB/s to 188us/1428.0 GB/s (~427x), while the read roofline remained
  stable. The canonical card confirmed 36us/944.9 GB/s and 187us/1431.8 GB/s and completed both production
  point-read cache regimes. The production TRUE/FALSE SQL route passed three sequential plus two concurrent GPU
  invocations; both full suite modes, all static/boundary gates, and independent runtime/benchmark audit are clean.
  STRUCT-001FP then isolated `CudaGroupDeviceView`, `DeviceArithBuffer`, and the nine arith/bool/pack/widen/upload/
  wide-key/distinct device-derived launchers in the rustfmt-clean 920-line `derived_column.rs` child, reducing the
  execution root to 9,738 lines and `group_input.rs` to 618. Both public type paths and inherent APIs remain stable;
  `group_input -> derived_column` is the sole child edge and there is no reverse dependency. Six focused validation
  tests and five arith/bool/composite/text-distinct/streaming GPU routes passed 15 sequential plus ten concurrent
  invocations. Both execution modes passed 48/79, both engine modes passed 505/487, workspace all-target check,
  strict execution/engine clippy, dependency/scoped source/reference/format checks, and independent audit are clean.
  PTX, seven symbols/arguments, layouts, extents, context identity, transfers, synchronization, errors, results, and
  leases are source-equivalent, so the report card was not applicable. Audit promoted inherited safe-API device OOB/
  raw-pointer risk across eight launcher families as STRUCT-001FQ; `upload_u64_device` alone is already total.
  STRUCT-001FQ then closed that safe-API boundary. Expression preflight now checks aligned resident windows,
  opcodes, needle contracts, typed stack transitions, and exact Value/Mask/TwoValues terminals before CUDA setup.
  Two-column text comparisons carry both blob lengths through the engine and PTX; malformed device intervals write
  false before byte loads. Derived and grouped inputs use checked 4-byte fixed/bitmap alignment and 8-byte text-
  offset alignment, typed same-context views, exact initialized/source/destination/validity geometry, exact fixed/
  text DISTINCT permutations, bounded row domains, and a drained device validator for actual text offsets. The
  public derived buffer exposes only its typed group view; its raw pointer is crate-internal for the gather owner.
  The rustfmt-clean leaves are `expression_vm.rs` (1,361 lines), `expression_vm/tests.rs` (222),
  `derived_column.rs` (1,269), `resident_window.rs` (64), `group_input.rs` (636), and `predicate_mask.rs` (393);
  the execution root is 9,750 lines. The final eight-route safety/arith/bool/composite/distinct/streaming GPU
  matrix passed 24 sequential plus 16 concurrent invocations without CUDA 700/716/717. Both execution modes pass
  53/80 and both engine modes pass 505/487; workspace all-target/all-feature check, strict execution/engine clippy,
  dependency/read-policy/scoped source/format/diff gates, and independent audit are clean. The canonical report
  card remained stable: IN-L2/OUT-OF-L2 rooflines were 1,482.7/1,446.7 GB/s, count was 0.91x/1.01x roofline,
  grouped aggregation was 1,678.0 M elements/s, and batch-65,536 point reads were 236.8M at p50 140us IN-L2 and
  254.8M at p50 128us OUT-OF-L2; the OUT-OF-L2 index route was 3.22x scan. Audit also exposed the omitted,
  hand-maintained 9,745-line expression PTX dependency hub, which was promoted as **STRUCT-001FR**.
  STRUCT-001FR then dispositioned and deleted that ownership hub. Its 67 live entry points now reside in 13
  operator/type-owned PTX leaves ranging from 171 to 1,398 lines; no leaf crosses the preferred production
  envelope. Normalized declaration-through-body comparison against the former file proves every surviving symbol,
  ABI, and body unchanged. The only removals are two unreferenced legacy compactors whose ordered replacements were
  already live. Every cache/include consumer now loads the one leaf defining its requested entry, and a repository-
  wide audit found 72 unique PTX definitions with no duplicates. The complete responsibility, history, consumer,
  and ABI evidence is in `design/expression-ptx-disposition.md`. All 13 leaves independently assemble for `sm_90`,
  pass the permanent ASCII guard, and contain no include dependency. Fifteen expression/gather/wide/varlen/derived/
  join/aggregate/group/sort production routes passed 45 sequential plus 30 concurrent invocations without CUDA
  700/716/717. Execution passes 53/80 in both ordinary modes and 133/133 with ignored tests included; engine passes
  505/487 in both ordinary modes. Workspace all-target/all-feature check, strict execution/engine clippy, dependency
  and read-policy boundaries, scoped format/reference/diff checks, and independent audit are clean. Three obsolete
  direct text fixtures were corrected to the exact eight-byte alignment already required by the safe API, and stale
  engine comments now describe that contract. The canonical card remains stable: IN-L2/OUT-OF-L2 rooflines are
  1,486.0/1,451.0 GB/s, count is 0.86x/1.00x roofline, grouped aggregation is 1,677.3 M elements/s, and batch-65,536
  point reads are 227.1M at p50 139us IN-L2 and 253.2M at p50 131us OUT-OF-L2; the OUT-OF-L2 index route is 3.25x
  scan. A broader charter sweep also found seven independently reproducible ignored engine failures unrelated to
  the unchanged PTX bodies; their disposition is promoted first as **QUALITY-002**.
  QUALITY-002 then restored the complete ignored engine gate. The bridge materialized-view failure exposed a real
  descriptor invariant defect: `SUM(int2/int4)` materialized `Int8` values while scalar and grouped binding still
  declared `SqlType::Int4`; both forms now consistently use `Int8`, bigint OID 20, and size 8, with ordinary
  catalog-binding plus persisted materialized-view regressions. The join failure was an obsolete expectation:
  PostgreSQL permits a hidden `ORDER BY` key for a non-DISTINCT projection, and the existing route already kept
  the key on-device through sorting before final projection; the test now asserts that GPU result. Recovery now
  eagerly publishes a resident snapshot, so five cold-checkpoint fixtures no longer reached their intended
  over-budget streaming tier. Their shared setup now sets the tiny budget and explicitly evicts only that eager
  resident-cache entry; the independently owned cold artifact remains intact and the restore, forward-patch,
  checksum, boundary, and lane-frontier assertions execute again. The seven routes passed 21 sequential plus 14
  concurrent final-tree invocations. The complete engine gate passes 992/992, ordinary engine passes 505/487,
  ordinary execution passes 53/80, workspace all-target/all-feature check, strict execution/engine clippy,
  dependency boundary, scoped format/diff checks, and independent audit are clean. The canonical report card is
  stable in both layers and cache regimes: IN-L2/OUT-OF-L2 rooflines are 1,464.6/1,450.3 GB/s, count is
  0.92x/1.00x roofline, grouped aggregation is 1,678.3 M elements/s, and batch-65,536 point reads are 247.4M at
  p50 139us IN-L2 and 249.7M at p50 132us OUT-OF-L2; the OUT-OF-L2 index route is 3.20x scan.
  STRUCT-001FS then removed the public resident TEXT-prefix route's full offsets/blob D2H and host predicate.
  The 181-line ASCII PTX leaf compares prefix bytes and block-reduces the count on-device; the host receives only
  a 16-byte count/error result. Checked host windows precede CUDA work, while the kernel validates the canonical
  zero offset and every actual start/end pair before text loads. The shared primary context is bound before module
  caching or buffer leasing; deleting that bind makes the first fresh-reader-thread regression fail with CUDA 201,
  proving non-vacuity. Positive, empty-prefix, empty-relation, no-match, malformed-window, malformed-device-offset,
  reuse, and fresh-thread coverage passed three sequential plus two concurrent final-tree invocations. The retained
  512-row chunked-upload consumer passes with 256 prefix matches. Complete execution passes 135/135, ordinary
  engine passes 505/487 and complete engine passes 992/992; workspace all-target/all-feature check, strict
  execution/engine clippy, dependency boundary, `sm_90` assembly, ASCII/scoped-format/diff checks, and independent
  audit are clean. The canonical card is stable: IN-L2/OUT-OF-L2 rooflines are 1,489.9/1,448.9 GB/s, count is
  0.91x/1.00x roofline, grouped aggregation is 1,677.7 M elements/s, and batch-65,536 point reads are 252.6M at
  p50 136us IN-L2 and 257.1M at p50 128us OUT-OF-L2; the OUT-OF-L2 index route is 3.22x scan.
  STRUCT-001FT then deleted the unconsumed `match_i32_between_row_indices_from_payload` facade and its private
  full-column D2H plus host BETWEEN filter (73 production lines). They were introduced for the old partitioned
  BETWEEN-AVG probe in `8381e7f7`; its last engine caller was retired in `8da01cb6` when the route moved to the
  general GPU expression/aggregate bridge. That bridge rebuilds the BETWEEN predicate, evaluates it through the
  device expression path, and runs AVG on-device; the specialized `stats_i32_between[_nullable]_from_payload`
  reductions remain live for a separate single-resident probe. Repository source, build, example, and test audit
  finds no surviving reference. Execution passes 54/81, engine passes 505/487, workspace all-target/all-feature
  check, strict execution/engine clippy, dependency boundary, source/diff checks, and independent audit are clean.
  No runtime path changed, so GPU HAZARD and report-card gates were not applicable.
  STRUCT-001FU then deleted the unconsumed public `project_text_from_payload` facade and private
  `launch_cuda_resident_text_project`, which unconditionally copied the full resident offsets array and text blob
  D2H before assembling every string on the host (101 production lines, plus three inherited surplus blank lines).
  Commit `4aeaa14f` introduced the API and sole engine caller; `5c21cad8` moved that caller to
  `project_text_rows_from_payload`, leaving no repository consumer. The surviving generic projector reads the
  contiguous offset and text spans bounded by the requested row extrema and performs permitted final host result
  assembly; sparse extrema can still approach a full-column readback, so existing RETIRE-003 owns that residual.
  Specialized point-read and join routes retain their fused GPU projectors. Execution passes 54/81, engine passes
  505/487 with the GPU sweep serial, and workspace all-target/all-feature check, strict execution/engine clippy,
  dependency/source/diff checks, and independent audit are clean. The execution root is 9,609 lines. No live
  runtime path changed, so GPU HAZARD and report-card gates were not applicable.
  STRUCT-001FV then made the surviving generic retained TEXT projector total. Its public API now accepts the exact
  logical `row_count`; all five engine consumers pass the count paired with their resident snapshot/source. It
  validates an aligned offset window, a bounded blob window, and every selected index before transfer, binds the
  owning primary CUDA context before every D2H, verifies canonical `offset[0] == 0` even for empty/zero-row
  results, rejects malformed selected spans, and preserves requested order, duplicates, empty strings, and final
  host string assembly. The
  permanent GPU contract covers a fresh nonempty and zero-row reader thread, a physically present row beyond a
  smaller declared logical extent, misalignment, malformed/noncanonical offsets, empty selection, and post-error
  reuse. Before the fix that fresh-thread assertion failed with CUDA 201; the final direct matrix passes three
  sequential plus two concurrent runs, while mixed retained, grouped TEXT, and ordered TEXT consumers pass nine
  sequential plus six concurrent runs without CUDA 700/716/717/201. Execution passes 54/81 and engine passes
  505/487 with the GPU sweep serial; workspace all-target/all-feature check, strict execution/engine clippy,
  source/diff/single-plan gates, and independent audit are clean. The canonical report card is stable: IN-L2/
  OUT-OF-L2 rooflines are 1,480.7/1,441.0 GB/s, count is 0.88x/1.01x roofline, grouped aggregation is 1,677.3M
  elements/s, and batch-65,536 point reads are 247.2M at p50 140us IN-L2 and 252.1M at p50 132us OUT-OF-L2;
  index/scan is 3.22x/3.19x. The execution root is 9,636 lines.
  STRUCT-001FW then isolated both stable resident TEXT inherent APIs, the prefix-count launcher/PTX owner, and
  bounded selected-row projection/readback into the rustfmt-clean 310-line `resident_text.rs` child, reducing the
  execution root to 9,332 lines. All four moved segments are byte-exact against the prior root. The private child
  adds no visibility bridge or cycle: its only dependencies are the neutral text-window validator and existing
  parent context/stream services, while the stable crate-root method paths and every caller remain unchanged. The
  direct TEXT contract passed three sequential plus two concurrent invocations; mixed retained, grouped-TEXT,
  and ordered-TEXT consumer commands passed nine sequential plus six concurrent invocations without device
  faults. Execution passes 54/81 and engine passes 505/487 with the GPU sweep serial; workspace
  all-target/all-feature check, strict execution/engine clippy, source/reference/scoped-format/diff gates, and
  independent audit are clean. The 181-line PTX file, symbol/ABI/body, checked geometry, primary-context binding,
  pooled-stream and lease lifetime, bounded D2H, requested ordering/duplicates, UTF-8 assembly, and error behavior
  are source-equivalent, so the report card was not applicable.
  STRUCT-001FX then deleted the unconsumed public `filtered_stats_i32_compare_from_payload` facade (17 production
  lines) and removed its three stale live-source symbol references. The deleted wrapper launched compare-project,
  copied every matching int4 to a host `Vec`, and reduced it on the CPU. Whole-tree and history audit proves its
  sole engine caller moved in `6bc6b188` to `filtered_scalar_stats_i32_from_payload`, which performs filtering and
  reduction on-device and reads back only count/sum/min/max; the direct route, row-project APIs, underlying
  compare-project launcher, remaining `CudaI32Stats::from_values` uses, and archived history are unchanged.
  Execution passes 54/81 and engine passes 505/487 with the GPU sweep serial; workspace all-target/all-feature
  check, strict execution/engine clippy, source/diff/docs gates, and independent audit are clean. The execution
  root is 9,314 lines. No live call path, kernel, layout, or result behavior changed, so HAZARD and report-card
  gates were not applicable.
  STRUCT-001FY then isolated the six resident int4 SUM/scalar-statistics APIs, `CudaI32Stats` and its raw result
  layout, and the three reduction launcher/PTX owners in the rustfmt-clean 1,083-line `resident_scalar.rs` child,
  reducing the execution root to 8,244 lines. All four source spans are normalized-exact against the prior root.
  The private child preserves the crate-root `CudaI32Stats` re-export and inherent method paths; shared
  `CudaI32Comparison` remains root-owned for both scalar and count consumers, and scalar depends one-way on the
  existing resident-count bitmap validator with no reverse edge, cycle, or visibility expansion. The five direct
  SUM/unfiltered/filtered/nullable/BETWEEN GPU families passed 15 sequential plus ten concurrent invocations
  without device faults. Execution passes 54/81 and engine passes 505/487 with the GPU sweep serial; workspace
  all-target/all-feature check, strict execution/engine clippy, source/reference/scoped-format/diff gates, and
  independent audit are clean. PTX symbols/ABIs/bodies, validation, NULL/empty/sentinel/overflow behavior,
  geometry, initialization, stream/lease/error lifetime, bounded scalar readback, and callers are
  source-equivalent, so the report card was not applicable.
  STRUCT-001FZ then deleted the unconsumed public whole-column int4 D2H facade and private kernel-less transfer,
  plus their sole remaining consumer: a test-only serial identity-copy launcher/embedded PTX and ignored
  migration A/B gate. The deletion removes 312 lines from the execution root (8,244 to 7,932) and 94 lines from
  `context_aggregate.rs` (2,602 to 2,508), leaving execution at 54 active and 80 GPU-ignored tests. Whole-tree and
  history audit proves the last product consumers moved by 2026-06-25 to bounded selected-row gather or the
  on-device GROUP BY/general-expression bridges; no engine, example, benchmark, roofline, script, or nonarchive
  documentation consumer remained. The live selected-row gather and ordered compare/project families are
  unchanged. Both execution modes pass 54/80 and both engine modes pass 505/487 with GPU sweeps serial; workspace
  all-target/all-feature check, strict execution/engine clippy, source/history/diff/docs gates, and independent
  audit are clean. No live path or surviving kernel/result behavior changed, so HAZARD and report-card gates were
  not applicable.
  STRUCT-001GA then deleted the unconsumed public `project_i32_compare_ordered_from_payload` facade (29 root
  lines), which materialized every matching int4 value to a host vector, CPU-sorted it, and host-windowed
  OFFSET/LIMIT. History proves `36470be8` moved `int4_ordered_projection` to the general GPU expression/sort route
  and `24ebe8cf` deleted the sole engine probe; only archived implementation-log facts remain. The standard
  Layer-1 compare/project benchmark, production device-ordered index compaction, and selected-row gathers are
  unchanged. The execution root is 7,903 lines. Both execution modes pass 54/80 and both engine modes pass
  505/487 with GPU sweeps serial; workspace all-target/all-feature check, strict execution/engine clippy,
  source/history/diff/docs gates, and independent audit are clean. No live caller, kernel, or surviving result
  behavior changed, so HAZARD and report-card gates were not applicable.
  STRUCT-001GB then made the surviving ordered int4 compare-compaction family total before extraction. Public raw
  predicate codes now accept only 0..=5; scalar/two-input routes retain their narrower 0..=4 contract; every index
  emitter rejects row domains above `u32::MAX` at runtime. `OrderedI32InputWindow` couples each device base with its
  exact extent and byte offset, resident inputs reuse the shared aligned-window validator, and every single/two-
  input intermediate is a typed same-context `PooledBufferLease` whose exact capacity is checked before CUDA setup.
  All mask and arithmetic callers retain those borrows across both count/scatter passes. Permanent host and GPU
  coverage exercises invalid code/domain/context, short and misaligned windows, checked arithmetic overflow,
  zero-at-end/zero-beyond behavior, two-input limits, and valid post-error context reuse. The final five-family
  matrix passes 15 sequential plus ten concurrent invocations with no CUDA 700/716/717; execution passes 55/81 and
  engine passes 505/487 with GPU sweeps serial. Workspace all-target/all-feature check, strict execution/engine
  clippy, source/diff/docs gates, and independent re-audit are clean. The final canonical card is stable: IN-L2/
  OUT-OF-L2 rooflines are 1,471.6/1,447.4 GB/s, compare count is 0.86x/1.01x roofline, grouped aggregation is
  1,677.3M elements/s, and batch-65,536 point reads are 247.0M at p50 140us IN-L2 and 253.0M at p50 132us OUT-OF-L2.
  The 8M-row ordered-compaction control remains at 0.0248x roofline; the reproducible 64M-row/50%-selectivity line
  returns roughly 134MB of host indices and remains result-path dominated, an existing boundary already owned by
  RETIRE-003. PTX, ABI, launch geometry, ordering, and valid results are unchanged. The root is 7,951 lines.
  STRUCT-001GC then isolated that total ordered-compaction boundary in the rustfmt-clean 1,088-line
  `resident_compare_ordered.rs` child and byte-exact 803-line `resident_compare_ordered.ptx`, reducing the execution
  root to 6,077 lines. The inherent public methods and `CudaI32Comparison` path remain at the crate root; nine narrow
  production/test `pub(super)` bridges preserve the former effective scope with no public child surface, reverse
  sibling dependency, or cycle. Independent reconstruction proves the four PTX entry symbols, parameter order, and
  complete 29,242-byte payload exact; normalized Rust differs only by those bridges and rustfmt wrapping. The final
  five-family matrix passes 15 sequential plus ten concurrent invocations without CUDA 700/716/717. Execution passes
  55/81 and engine passes 505/487 with GPU sweeps serial; workspace all-target/all-feature check, strict execution/
  engine clippy, source/reference/scoped-format/diff gates, and independent audit are clean. The canonical card remains
  stable: IN-L2/OUT-OF-L2 rooflines are 1,484.6/1,425.4 GB/s, ordered compaction is 0.0236x roofline IN-L2, grouped
  aggregation is 1,678.3M elements/s, and batch-65,536 point reads are 247.5M at p50 139us IN-L2 and 253.6M at p50
  132us OUT-OF-L2. Valid-path behavior, safety validation, stream draining, and lease lifetimes are unchanged.
  STRUCT-001GD then isolated the host-staged CUDA compatibility family in the rustfmt-clean 1,447-line
  `staged_filter.rs` and 592-line `staged_mvcc.rs` leaves, reducing the execution root to 4,067 lines.
  `CudaDriverRuntime` remains the stable public facade and now imports seven narrow `pub(super)` launchers directly;
  dependency flow is one-way from that facade into the leaves, with no reverse edge, cycle, public child surface, or
  old root ownership. Normalized reconstruction is exact apart from required visibility and import formatting. The
  fixed device-0 selection, fresh per-call context/PTX load and allocation, host upload, synchronization, readback,
  empty-input, and error behavior are intentionally unchanged. Both leaves explicitly identify this host-materialized
  generic path as RETIRE-003 bootstrap debt rather than resident product execution. All seven direct GPU families pass
  21 sequential plus 14 concurrent invocations; the full ignored GPU suites also pass. Execution passes 55/81 and
  engine passes 505/487; workspace all-target/all-feature check, strict execution/engine clippy, source/reference/
  Rust-2021-format/diff gates, and independent audit are clean. No runtime behavior changed, so the report card was
  not applicable.
  STRUCT-001GE then made the product-live two-column int4 expression facade total and isolated its surviving owner.
  The unchecked raw-offset/opcode elementwise kernel plus atomic index append and host sort are gone; the facade now
  lowers through the typed postfix VM and ordered compaction, validating operation, comparison, `u32` row domain,
  alignment, and exact allocation windows before CUDA. Permanent negative coverage proves malformed empty and
  nonempty calls fail closed, misaligned/short inputs do not launch, and the context remains reusable; valid add,
  multiply, zero/all-result, ordering, and checked-overflow semantics remain intact. The two obsolete PTX entries were
  deleted, leaving `expression_i32.ptx` at 520 lines and six live entries. Expression filter and value-materialization
  orchestration now lives in the rustfmt-clean 393-line private `expression_filter.rs` leaf behind five stable inherent
  APIs and one-way dependencies on `expression_vm` and `resident_compare_ordered`; the execution root is 3,502 lines.
  Full-column expression D2H plus host selected gather remains explicitly owned by RETIRE-003. Five GPU families pass
  15 sequential plus ten concurrent invocations; execution passes 55/81 and engine passes 505/487. Workspace/static/
  source/reference/scoped-format/diff gates, `ptxas`, ASCII and exact-entry audits, full suites, and independent audit
  are clean. The canonical card remains healthy: IN-L2/OUT-OF-L2 rooflines are 1,405.1/1,449.9 GB/s, scalar compare
  count is 1,275.3/1,457.6 GB/s, grouped aggregation is 1,679.0M elements/s, and batch-65,536 point reads are 237.0M/s
  at p50 140us IN-L2 and 251.8M/s at p50 134us OUT-OF-L2.
  STRUCT-001GF then completed the execution-root disposition. The sole product-live parallel LSD-radix argsort now
  lives in the 1,206-line `resident_sort.rs` owner with byte-identical 490-line/four-entry
  `resident_argsort.ptx`; independent bitonic and serial-radix parity oracles live only in the 768-line
  `#[cfg(test)]` support owner. The typed radix boundary borrows a same-context pooled lease and validates the u32
  permutation domain, checked `n*8` extent, alignment, capacity, and address range before CUDA; the host-slice facade
  also rejects an oversized domain before context selection or allocation. Product-dead adaptive, ORDER-BY-LIMIT,
  and raw-HAVING implementations, PTX, and self-only tests are deleted; production HAVING remains on the predicate
  VM. The stable ORDER BY facade, crossover, direction, ties, stability, and results are unchanged, while host-key
  H2D and permutation D2H remain explicit RETIRE-003 debt. Five GPU families passed 15 sequential plus ten concurrent
  invocations; execution passed 56 ordinary/77 ignored-GPU tests and engine passed 505/487. The canonical card stayed
  healthy at 1,478.8/1,444.3 GB/s roofline, 1,679.0M grouped elements/s, and 247.5M/252.8M batch-65,536 point reads at
  p50 138/132us. Strict checks, clippy, exact PTX (`d940c475...`), `ptxas`, ASCII, deletion/reference, formatting,
  diff, full-suite, and independent re-audit gates are clean. The execution root is now 1,660 lines and no longer an
  outlier.
  STRUCT-001GG completed the full analysis packet for the handwritten `engine_expr.rs` hub and isolated its
  standalone grouped-result GPU permutation in the 186-line `engine_result_sort.rs` leaf. The executable function
  is source-exact, remains available at `crate::engine_expr::gpu_sort_permutation` through one narrow crate-private
  re-export, and has exactly two callers: multi-pass group alignment and grouped final ORDER BY. Dependencies are
  one-way to residency payload construction, typed execution sorting, and SQL/error types, with no unsafe, PTX,
  cycle, or visibility expansion. Inherited comments that still attributed this owner to join sorting were removed
  or corrected: joins now sort and window device coordinates directly. Host result-key/payload construction, H2D,
  and permutation D2H remain explicit RETIRE-003 debt. Six affected GPU families passed 18 sequential plus 12
  concurrent HAZARD invocations; engine passed 505 ordinary/487 ignored-GPU tests. All-target check, strict engine/
  execution clippy, exact-source/consumer/scoped-format/diff/docs gates, and independent audit are clean. No runtime,
  kernel, residency, or result behavior changed, so the canonical report card was not applicable. The expression
  root is now 11,231 lines.
  STRUCT-001GH then isolated the state-free scalar expression contract in the 63-line `engine_expr_ir.rs` leaf.
  `ResidentBinaryOp` and `ResidentExpr` retain their exact derives, variants, fields, ordering, and variant-specific
  docs; only inherited prose that still called production parser/operator coverage future work was corrected. Their
  stable `crate::engine_expr` paths remain narrow crate-private re-exports. Dependencies now flow one-way from SQL
  binding, DML predicate construction, streaming, tests, and the GPU executor toward the neutral IR and its sole
  explicit `Decimal128` dependency; the leaf has no state, reverse engine dependency, feature gate, unsafe, PTX,
  cycle, or visibility expansion. Focused SQL boolean, numeric lowering, mixed-width DML, and streaming type-matrix
  GPU routes pass, as do both 505/487 engine modes, all-target check, strict clippy, exact-definition/consumer/
  scoped-format/diff/docs gates, and independent re-audit. Runtime behavior did not change, so HAZARD and report-card
  gates were not applicable. The expression root is now 11,168 lines.
  STRUCT-001GI then isolated the five state-free join-plan contract types in the 76-line `engine_join_ir.rs` leaf.
  `JoinColRef`, `JoinProjItem`, `JoinRelationRef`, `JoinStep`, and `JoinPlan` retain their exact derives, variants,
  fields, visibility, and order behind unchanged `crate::engine_expr` crate-private paths. Dependencies remain one-way
  from SQL binding, resident/streaming execution, and tests to the prelude-only contract; `JOIN_NULL_ROW`, device
  ownership, parsing, projection, device-coordinate sorting/windowing, and execution remain at their established
  owners. Stale transient-intermediate and retired result-sort comments were corrected to the live coordinate
  pipeline. Focused parsed, two-way, three-way user, three-way catalog, outer, and streaming GPU join routes pass, as
  do both 505/487 engine modes, all-target check, strict clippy, exact-definition/consumer/scoped-format/diff/docs
  gates, and independent re-audit. Runtime behavior did not change, so HAZARD and report-card gates were not
  applicable. The expression root is now 11,097 lines.
  STRUCT-001GJ then isolated the five state-free select/predicate normalization helpers in the 238-line private
  `engine_expr/normalization.rs` leaf. All bodies and their semantic docs are source-equivalent; the two parent-only
  helpers gained only `pub(super)`, while `grouped_projection_to_aggregates`,
  `like_pattern_for_literal_prefix`, and `resident_predicate_from_bound_filters` retain their existing
  `crate::engine_expr` crate-private facade paths. Dependencies flow one-way to scalar IR, bound-select, error, and
  SQL value/projection contracts; pruning, predicate compilation/lowering, grouped materialization, every `Engine`
  method, and all runtime/device state remain at their established owners. Six focused legacy-group, LIKE, HAVING,
  DML, and streaming GPU/active routes pass, as do both 505/487 engine modes, all-target check, strict clippy,
  exact-source/consumer/visibility/scoped-format/diff/docs gates, and independent re-audit. Runtime behavior did not
  change, so HAZARD and report-card gates were not applicable. The expression root is now 10,872 lines.
  STRUCT-001GK then isolated the pure shard-pruning and cross-shard point-shape owner in the 213-line private
  `engine_expr/shard_pruning.rs` leaf. `mandatory_int4_equalities`, `shard_point_lookup_int4_eq`,
  `shard_zone_map_excludes`, and both colocated unit tests retain exact bodies, docs, names, and attributes; only the
  two parent-used helpers gained `pub(super)`, while the existing `crate::engine_expr::shard_point_lookup_int4_eq`
  crate-private facade remains one narrow re-export. Dependencies flow one-way to scalar IR, bound-select,
  relational table/stat, and SQL filter/type/value contracts; unified-source construction, lookup execution,
  device/runtime state, and MULTI work remain in place. Both unit gates, three focused GPU zone-map/index/locate
  parity routes, final 505/487 engine modes, all-target check, strict clippy, exact-source/test/consumer/visibility/
  scoped-format/diff/docs gates, and independent re-audit pass. The first ignored sweep exposed 9,062 obsolete
  generated `target/tmp/gpu-db-*` files and 342 directories consuming 547.4 GiB; scoped cleanup preserved audit and
  benchmark logs, restored 566 GiB free, all 11 ENOSPC failures passed alone, the complete 487-test rerun passed, and
  fresh generated residue was removed. Runtime behavior did not change, so HAZARD and report-card gates were not
  applicable. The expression root is now 10,668 lines.
  STRUCT-001GL then isolated grouped-value reconstruction and representative-row grouping in the 174-line private
  `engine_expr/grouped_values.rs` leaf. `narrow_ordered_value` and `composite_group_count_reps` retain exact bodies
  and gained only `pub(super)` for parent use; dependencies explicitly name the relational offset/layout helpers,
  snapshot/table, SQL/error, and typed execution contracts. The select orchestrator, predicate compiler/lowerers,
  every `Engine` method, residency construction, runtime/device owners, and MULTI work remain in place. Inherited
  prose was corrected to state that the representative route appends the DISTINCT value member, which may be int8
  or bool, rather than an always-non-int member. Nine focused GPU routes cover int2/date/timestamp, bool, numeric,
  UUID, int8, text, and composite COUNT(DISTINCT); both 505/487 engine modes, all-target check, strict clippy,
  exact-source/consumer/visibility/scoped-format/diff/docs gates, and independent re-audit pass. Fresh generated
  `gpu-db-*` test residue was removed after the suite. Runtime behavior did not change, so HAZARD and report-card
  gates were not applicable. The expression root is now 10,510 lines.
  STRUCT-001GM then isolated resident device-source handles and on-device MVCC visibility-program contracts in the
  76-line private `engine_expr/execution_source.rs` leaf. `ResidentExecSource`, `ShardedUnifiedExecSource`,
  `ResidentVisibility`, and `ResidentVisibility::push_conjuncts` retain source-equivalent derives, field order,
  visibility, instruction order, and comparison/mask opcodes; only explicit import/path normalization and durable
  ownership prose differ. Their stable `crate::engine_expr` paths remain narrow crate-private re-exports.
  Dependencies remain one-way to the relational snapshot and typed execution contracts; unified-source construction,
  routing, join/source orchestration, every `Engine` method, runtime/device action, and MULTI work remain in place.
  Three focused MVCC visibility GPU gates plus streaming injected-source and resident-select routes pass, as do both
  505/487 engine modes, all-target check, strict clippy, dependency/source/consumer/visibility/scoped-format/diff/docs
  gates, and independent re-audit. Another 513 MiB of fresh generated `gpu-db-*` test residue was removed. Runtime
  behavior did not change, so HAZARD and report-card gates were not applicable. The expression root is now 10,441
  lines.
  STRUCT-001GN then isolated join device-source ownership contracts in the 44-line private
  `engine_expr/join_source.rs` leaf. `JOIN_NULL_ROW`, `JoinDeviceMemory`, `JoinNullPadMask`, `JoinExecSide`, and
  `JoinDeviceMemory::mem` retain exact values, variant/field/tuple order, types, docs, borrow behavior, ownership,
  and drop order; only explicit imports and the narrow `pub(super)` exposure matching the former parent scope differ.
  Existing `crate::engine_expr::{JoinDeviceMemory, JoinExecSide}` paths remain narrow crate-private re-exports.
  Dependencies remain one-way to the resident-visibility, relational-residency, and typed CUDA ownership contracts;
  join construction, routing, execution, filtering, projection, streaming orchestration, every `Engine` method, and
  MULTI remain in place. Focused resident, transient-catalog, NULL-padded OUTER, and over-budget streaming GPU joins
  pass, as do both 505/487 engine modes, the complete 992-test GPU suite, all-target check, strict clippy,
  source/consumer/visibility/scoped-format/diff/docs gates, and independent audit. The suite's 15 GiB of fresh
  generated `gpu-db-*` residue was removed. Runtime behavior did not change, so HAZARD and report-card gates were
  not applicable. The expression root is now 10,410 lines.
  STRUCT-001GO then isolated the 25 state-free resident-predicate operand/type-recognition, literal-normalization,
  and LIKE-tokenization helpers in the rustfmt-clean 400-line private `engine_expr/predicate_operands.rs` leaf.
  Definitions, exhaustive `ResidentExpr` matches, docs, errors, and behavior are exact apart from narrow
  parent-equivalent `pub(super)` visibility and one rustfmt-wrapped timestamp signature. DATE still rejects raw
  integers, TIMESTAMP still accepts bound int8 microseconds, numeric literals retain canonical scale behavior, and
  LIKE retains bytewise escape/`%`/`_` tokenization. All 25 consumers remain in the parent behind explicit imports;
  dependencies are one-way to neutral expression, relational table, SQL parser/value, and error contracts, with no
  CUDA/runtime, `Engine`, unsafe, cfg, cycle, or API expansion. Compiler/lowerer bodies, every `Engine` method,
  orchestration, routing, mutation, joins, R3 decisions, and MULTI remain in place. Fourteen focused typed GPU
  predicate and DML routes pass, as do both 505/487 engine modes, the complete 992-test GPU suite, all-target check,
  strict clippy, exact-source/consumer/visibility/scoped-format/diff/docs gates, and independent audit. The suite's
  15 GiB of fresh generated residue was removed. Runtime behavior did not change, so HAZARD and report-card gates
  were not applicable. The expression root is now 10,030 lines.

  STRUCT-001GP then isolated the complete 25-function resident predicate compiler in the rustfmt-clean 1,252-line
  private `engine_expr/predicate_compiler.rs` leaf, reducing the expression root to 8,812 lines. The exact executable
  block now depends one-way on predicate operands through explicit imports; seven helpers remain leaf-private and only
  the 18 proven parent-used functions gained `pub(super)`. Every `ExprStep` opcode/order, postfix-stack transition,
  numeric rescale/canonicalization/overflow error, typed offset, 3VL validity-AND order, element-width diversion,
  LIKE byte token, bool constant-fold result, text/UUID orientation, and error string remains source-equivalent.
  Every `Engine` method, lowerer/execution orchestrator, runtime/device action, routing, mutation, join, R3 decision,
  and MULTI concern remains in place. Twenty-three focused typed GPU predicate/DML tests pass, as do both 505/487
  engine modes, the complete 992-test GPU suite, all-target check, strict clippy, exact-source/consumer/visibility/
  dependency/scoped-format/diff/docs gates, and independent audit. Fresh generated test residue was removed. Runtime
  behavior did not change, so HAZARD and report-card gates were not applicable.

  STRUCT-001GQ then isolated the current resident DML transition orchestration in the rustfmt-clean 486-line private
  `engine_expr/resident_dml.rs` leaf, reducing the expression root to 8,341 lines. The eight method docs and bodies are
  byte-exact: five stable inherent methods remain `pub(crate)` and three helpers remain leaf-private. Shard snapshot
  and generation capture, zone-map/local-slot correctness, W0 liveness, the already-dead filter, exact-one fingerprint
  tuple verification, zero-row handling, partial-failure re-admit, tombstone-before-append ordering, multi-row identity,
  the SV6 `created_by` stamp, and churn accounting are unchanged. Dependencies flow one-way to shard pruning,
  predicate lowering, and existing resident mutation/read primitives; R3-001 remains the sole future-design owner.
  Twelve focused GPU transition tests pass, as do both 505/487 engine modes, the complete 992-test GPU suite,
  all-target check, strict clippy, exact-source/consumer/visibility/dependency/scoped-format/diff/docs gates, and
  independent audit. Fifteen GiB of generated residue was removed. Runtime behavior did not change, so HAZARD and
  report-card gates were not applicable.

  STRUCT-001GR then isolated private `TextRebaseOp` and the exact sharded unified-source layout/recompaction method in
  the rustfmt-clean 727-line private `engine_expr/sharded_source.rs` leaf, reducing the expression root to 7,620 lines.
  The executable body, errors, segment/fill/copy order, checked arithmetic, offsets, descriptor field order, counters,
  and device proof/source-owner/sidecar/`Arc` lifetimes remain equivalent; only two rustfmt line wraps differ. The
  stable inherent method and four callers are unchanged, dependencies are one-way, and obsolete parent imports were
  removed without visibility expansion. Seventeen focused GPU sharded-layout routes pass, as do both 505/487 engine
  modes, the complete 992-test GPU suite, all-target check, strict clippy, exact-source/import/consumer/visibility/
  lifetime/scoped-format/diff/docs gates, and independent audit. The required roofline comparison showed no material
  regression: IN/OUT `sum_i32` was 1488/1445 GB/s before and 1481/1448 after; representative normalized scan ratios
  and 349–350/260/1678 Melem/s sort/join/grouped rates remained stable. Fifteen GiB of generated residue was removed.
  Runtime behavior did not change, so HAZARD and the full report card were not applicable.

  STRUCT-001GS then isolated the exact sharded general bridge and private point-index route helper in the rustfmt-clean
  421-line private `engine_expr/sharded_route.rs` leaf, reducing the expression root to 7,219 lines. The normalized old
  and new source has an identical SHA-256; docs, errors, ordering, visibility, fallbacks, dispatch, result framing,
  counters, ownership, and stable caller/helper graph are exact. Dependencies flow one-way to source construction,
  normalization/pruning, index locate, and downstream executors with no visibility expansion or reverse cycle.
  Twenty-four focused GPU route tests pass, as do both 505/487 engine modes, the complete 992-test GPU suite,
  all-target check, strict clippy, exact-source/import/consumer/visibility/scoped-format/diff/docs gates, and independent
  audit. Fifteen GiB of generated residue was removed. Runtime behavior did not change, so HAZARD and report-card gates
  were not applicable.

  STRUCT-001GT then isolated the exact production DISTINCT/select bridges in the rustfmt-clean 91-line private
  `engine_expr/select_bridge.rs` leaf, reducing the expression root to 7,141 lines. The normalized old/new source hash
  is identical; docs, errors, one-column validation, GROUP BY plus COUNT synthesis, source/visibility forwarding,
  schema `Arc` truncation, flat `RowBlock` reframing, distinct-first dispatch, stable visibility, and consumers remain
  exact. Dependencies flow one-way to grouped execution and neutral source/result contracts with no cycle or expansion.
  Thirteen focused GPU/route tests pass, as do both 505/487 engine modes, the complete 992-test GPU suite, all-target
  check, strict clippy, exact-source/import/consumer/visibility/scoped-format/diff/docs gates, and independent audit.
  Fifteen GiB of generated residue was removed. Runtime behavior did not change, so HAZARD and report-card gates were
  not applicable.

  STRUCT-001GU then isolated the exact two test-only GROUP BY benchmark helpers in the rustfmt-clean 115-line private
  `engine_expr/group_bench.rs` leaf, reducing the expression root to 7,031 lines. The module and both methods retain
  explicit test gates; docs, inputs, catalog/snapshot/device failures, column resolution, capacity-strided offsets, u32
  bounds, index domains, `rows_limit`, `two_level`, `runs`, `agg_mask`, timing projection, returned rows, and error
  mapping are exact apart from explicit execution-type import normalization and consequent rustfmt wrapping. Its sole
  live consumer remains the ignored two-level-vs-single-level GPU benchmark, which passed non-vacuously. Both 505/487
  engine modes, the complete 992-test GPU suite, all-target check, strict test-target clippy, exact-source/import/
  consumer/cfg/visibility/scoped-format/diff/docs gates, and independent audit pass. Fifteen GiB of generated residue
  was removed. Runtime behavior did not change, so HAZARD and report-card gates were not applicable.

  STRUCT-001GV then isolated exact join-side resident/sharded/transient source resolution in the rustfmt-clean 81-line
  private `engine_expr/join_side.rs` leaf, reducing the expression root to 6,962 lines. Single-store precedence,
  nonempty-shard detection, statement `copin_s`, unified descriptor/device/visibility ownership, row counts,
  validity/memory failures, transient probe/upload/`Arc` lifetime, GPU-only errors, stable visibility, and consumers
  remain equivalent; only rustfmt wrapping differs. The five rustdoc lines accidentally orphaned by the historical JOIN
  insertion were moved byte-for-byte back immediately above `execute_resident_expr_select_with_binding`, their original
  owner. Seven focused GPU resident, mixed-key, transient-catalog, sharded-flip, and streaming joins pass, as do both
  505/487 engine modes, the complete 992-test GPU suite, all-target check, strict clippy, exact-source/import/consumer/
  visibility/history/scoped-format/diff/docs gates, and independent audit. Fifteen GiB of generated residue was
  removed. Runtime behavior did not change, so HAZARD and report-card gates were not applicable.

  STRUCT-001GW then isolated the exact four-method join entry/adapter/plan owner in the rustfmt-clean 454-line private
  `engine_expr/join_plan.rs` leaf, reducing the expression root to 6,515 lines. Three inherent methods remain
  `pub(crate)` and the override remains private. Docs, attributes, bodies, errors, adapter argument/option ordering,
  arity checks, statement `copin_s`, OUTER decisions, ambiguity handling, type/key classification, NATURAL/USING
  coalescing, projection/star order, sentinel guard, ORDER alias/NULL order resolution, GPU selection, and final
  coordinate/materialized dispatch are equivalent; only rustfmt compaction differs. Twenty distinct focused GPU
  resident, unsupported, multi-way, comma/star/composite/text/numeric/UUID, USING/NATURAL, OUTER/WHERE, order/window,
  and streaming join-plan routes pass, as do both 505/487 engine modes, the complete 992-test GPU suite, all-target
  check, strict clippy, exact-source/import/consumer/visibility/scoped-format/diff/docs gates, and independent audit.
  Fifteen GiB of generated residue was removed. Runtime behavior did not change, so HAZARD and report-card gates were
  not applicable.

  STRUCT-001GX then isolated exact NULL-pad and resident predicate-device mask construction in the rustfmt-clean
  107-line private `engine_expr/predicate_mask.rs` leaf, reducing the expression root to 6,430 lines. The resident
  builder remains `pub(crate)` and the pad helper gained only narrow `pub(super)` visibility for its two parent callers.
  Compiler/type selection, zero/no-op gates, visibility conjunct order and predicate flag, `probe-timing` output,
  text-aware launch, errors, allocation-before-build, the all-NULL source, and retained mask/source/allocation lifetimes
  are exact apart from explicit execution-type import normalization. Twelve focused GPU pushed-filter, OUTER-pad 3VL,
  typed NULL/N:N, rank, and streaming consumers pass, as do both 505/487 engine modes, the complete 992-test GPU suite,
  all-target/probe-feature checks, strict clippy, exact-source/import/consumer/visibility/feature/scoped-format/diff/docs
  gates, and independent audit. Fifteen GiB of generated residue was removed. Runtime behavior did not change, so
  HAZARD and report-card gates were not applicable.

  STRUCT-001GY then isolated the exact two streaming/incremental join-coordinate builders in the rustfmt-clean 248-line
  private `engine_expr/join_incremental.rs` leaf, reducing the expression root to 6,196 lines. Both methods remain
  `pub(crate)`. Resident predicate mask before row-range intersection, device-only mask AND, identity order, arity and
  ambiguity errors, accumulated-versus-new orientation, relation/key order, text/numeric/UUID/int8/int4 payload widths,
  bool rejection, key-count/NATURAL gates, right visibility/range intersection, bitmap argument order, final coordinate
  arguments, and source lifetimes are exact apart from typed execution-path import normalization. Seven focused GPU
  streaming, mixed-width, text, numeric/UUID, N:N, NULL-key, and composite controls pass, as do both 505/487 engine
  modes, the complete 992-test GPU suite, all-target check, strict clippy, exact-source/import/consumer/visibility/
  layout/argument-order/scoped-format/diff/docs gates, and independent audit. Fifteen GiB of generated residue was
  removed. Runtime behavior did not change, so HAZARD and report-card gates were not applicable.

  STRUCT-001GZ then isolated the exact complete join projection resolution/materialization owner in the
  rustfmt-clean 478-line private `engine_expr/join_projection.rs` leaf, reducing the expression root to 5,727
  lines. All four methods remain `pub(crate)`. Missing/ambiguous/FROM/alias-misalignment errors, relation-0
  coalescing, bare and qualified star order, alias expansion and attnum, device materialization specs and typed
  layouts, side-0 ownership, validity zip, endian conversion, numeric scale, UUID bytes, SQL NULL mapping,
  consumers, and dependency direction are source-equivalent. Eleven focused GPU star/USING/NATURAL/catalog/
  transient/three-way/typed-NULL/OUTER-pad/text/UUID/numeric/streaming routes pass, as do both 505/487 engine
  modes, the complete 992-test serial GPU suite, all-target check, strict clippy, static/scoped gates, and
  independent audit. Generated test residue was removed. The raw roofline remained stable (out-of-L2 roofline
  1440.2 to 1444.1 GB/s, gather kernel 152.5 to 155.3 GB/s, join 258.8 to 259.8 M-element/s, grouped 1677.7 to
  1678.1 M-element/s). The canonical report card likewise remained stable: out-of-L2 batched point reads moved
  from 251.6M to 253.1M lookups/s at p50 132us, and indexed single-flight moved from 3.20x to 3.25x scan. Runtime
  behavior did not change, so HAZARD was not applicable.

  STRUCT-001HA then isolated the exact two streaming-only join-coordinate post-filters in the rustfmt-clean 84-line
  private `engine_expr/join_coordinate_filter.rs` leaf, reducing the expression root to 5,657 lines. Both methods
  remain `pub(crate)`. Arity rejection before work, relation-order predicate and visibility masks, retained synthetic
  NULL-pad sources/allocations, temporary `Option::as_ref` vector order, side-0 execution context, real-mask before
  pad-mask arguments, the visibility path's all-`None` pad vector, errors, consumers, and dependency direction are
  source-exact. Five focused GPU OUTER-WHERE, pad 3VL, typed/FULL-pad, real-NULL-versus-pad, and over-budget streaming
  controls pass, as do both 505/487 engine modes, the complete 992-test serial GPU suite, all-target check, strict
  clippy, static/scoped gates, and independent audit. Generated residue was removed. Runtime behavior did not change,
  so HAZARD and report-card gates were not applicable.

  STRUCT-001HB then isolated the exact main non-streaming/direct device-coordinate join executor in the rustfmt-clean
  525-line private `engine_expr/join_coordinate_exec.rs` leaf, reducing the expression root to 5,143 lines. Its sole
  sibling `join_plan` caller retains the former root-private visibility envelope through the authorized `pub(super)`
  token; every body invariant and error remains source-exact, imports are explicit, the obsolete parent-only
  `JoinNullPadMask` import is gone, and dependencies remain one-way. U32 bounds, input/post/range/pad masks and retained
  lifetimes, typed key descriptors and widths, coordinate orientation and OUTER flags, pre-post-filter coordinate copy,
  sort/null/UUID and window semantics, aliases/attnum, optional materialization, typed readback/NULL/endian/numeric/
  UUID conversion, row-major results, and metadata are unchanged. Eleven focused GPU direct/n-way/composite/text/
  numeric/UUID/typed-NULL/OUTER/order/window/streaming routes pass, as do both 505/487 engine modes, the complete
  992-test serial GPU suite, all-target check, strict clippy, static/scoped gates, and independent audit. Generated
  residue was removed. The raw roofline remained stable (out-of-L2 roofline 1448.1 to 1444.7 GB/s, gather kernel
  155.3 to 155.3 GB/s, join 257.9 to 259.1 M-element/s, grouped 1677.0 to 1678.0 M-element/s). The canonical report
  card likewise remained stable: out-of-L2 batched point reads moved from 253.2M at p50 131us to 250.2M at p50 132us,
  and indexed single-flight moved from 3.24x to 3.23x scan. Runtime behavior did not change, so HAZARD was not
  applicable.

  STRUCT-001HC then consolidated the exact three general-select/grouped entry bridges in the rustfmt-clean 226-line
  existing private `engine_expr/select_bridge.rs` leaf, reducing the expression root to 5,014 lines. All three
  methods remain `pub(crate)` and the programmatic wrapper retains `#[allow(dead_code)]`. Every rustdoc/comment/body,
  the single binding and argument order, legacy grouped normalization before binding, injected source/visibility,
  predicate reconstruction before clearing all bound filters, group-key derivation, order metadata vector lengths/
  values, consumers, imports, and dependency direction are source-equivalent; the module ownership prose now names
  the complete bridge owner. Ten focused GPU programmatic-predicate, grouped/plain/ordered/DISTINCT dispatch,
  sharded-source, and versioned-visibility controls pass, as do both 505/487 engine modes, the complete 992-test
  serial GPU suite, all-target check, strict clippy, static/scoped gates, and independent audit. Generated residue
  was removed. Runtime behavior did not change, so HAZARD and report-card gates were not applicable.

  STRUCT-001HD then re-homed the history-proven general dispatcher rustdocs and isolated exact predicate dispatch
  plus bool/standalone-NULL fast paths in the rustfmt-clean 639-line private
  `engine_expr/predicate_dispatch.rs` leaf, reducing the expression root to 4,400 lines and below the mandatory
  5,000-line threshold. Commit history proves the eight general lines originally documented
  `lower_resident_predicate`; they and the bool-specific lines are byte-identical at their corrected owners. Three
  helpers remain private and the dispatcher remains `pub(crate)`. MVCC visibility composition, bare-bool/IS NULL,
  nullable and typed dispatch order, AND/OR VM, arithmetic/int4 peepholes, bounds, offsets/validity, errors, retained
  device references, GPU-only fail-loud behavior, six consumers, and one-way dependencies are source-exact; four
  now-leaf-only compiler imports were removed from the root. Fifteen focused GPU visibility/bool/NULL/3VL/mixed-width/
  typed/arithmetic/DML/streaming controls pass, as do both 505/487 engine modes, the complete 992-test serial GPU
  suite, all-target check, strict clippy, static/scoped gates, and independent audit. Generated residue was removed.
  Runtime behavior did not change, so HAZARD and report-card gates were not applicable.

  STRUCT-001HE then isolated the exact complete typed predicate-lowering owner in the rustfmt-clean 1,147-line
  private `engine_expr/predicate_typed_lowering.rs` leaf, reducing the expression root to 3,262 lines. The nine
  dispatcher-called lowerers gained only `pub(super)` to recreate their former ancestor-private visibility;
  `numeric_cross_scale_scalar` remains private and leaf-local. After normalizing those nine authorized visibility
  tokens, the old and new docs/attributes/bodies have identical SHA-256 hashes. Every `Ok(None)` versus hard-error
  branch, int8/int2 width and orientation, temporal parse/op/validity program, numeric i128 scale/rescale/overflow,
  UUID unsigned ordering, text equality/lex/LIKE/layout, row-count conversion, element type, and CUDA error remains
  source-exact. Twenty-four focused GPU controls and 33 independent-audit GPU controls pass, as do both 505/487
  engine modes, the complete 992-test serial GPU suite, all-target check, strict clippy, static/scoped gates, and
  independent audit. Generated test residue was removed. Runtime behavior did not change, so HAZARD and report-card
  gates were not applicable.

  STRUCT-001HF then isolated the shared GPU COUNT(DISTINCT) sort/mark/group owner in the rustfmt-clean 161-line
  private `engine_expr/grouped_count_distinct.rs` leaf, reducing the expression root to 3,127 lines. A
  rustfmt-normalized reconstruction from the old closure matches the new function exactly after explicit captures
  and four redundant snapshot-borrow removals. The two grouped/scalar callers preserve argument order; fixed/text
  descriptors, numeric/UUID high-low layout, multikey/heterogeneous sort, new-distinct marking, grouped SUM mask,
  constant scalar group, empty/NULL/errors, and synchronized derived-buffer lifetimes remain unchanged. Twenty-seven
  focused GPU controls and 25 independent-audit GPU controls pass, as do both 505/487 engine modes, the complete
  992-test serial GPU suite, all-target check, strict clippy, static/scoped gates, and independent audit. Generated
  test residue was removed. Runtime behavior did not change, so HAZARD and report-card gates were not applicable.

  STRUCT-001HG then isolated the complete scalar aggregate phase in the rustfmt-clean 231-line private
  `engine_expr/scalar_aggregate.rs` leaf, reducing the expression root to 2,925 lines. A normalized reconstruction
  is token-identical after explicit capture-borrow adaptations and the required helper tail-result syntax. Selected
  columns and access path transfer by value and are Arc-wrapped exactly once without clones. Zero-row typed NULL
  versus COUNT zero, int4/int8/numeric SUM/MIN/MAX/AVG result types and scales, checked overflow/error strings,
  COUNT(DISTINCT) constant-group delegation, GPU metadata, and resident/device/index lifetimes are unchanged. Ten
  focused GPU/spec controls and 28 independent-audit GPU/spec controls pass, as do both 505/487 engine modes, the
  complete 992-test serial GPU suite, all-target check, strict clippy, static/scoped gates, and independent audit.
  Generated test residue was removed. Runtime behavior did not change, so HAZARD and report-card gates were not
  applicable.

  STRUCT-001HH then isolated terminal typed/nullable projected-row materialization in the rustfmt-clean 216-line
  private `engine_expr/projected_rows.rs` leaf, reducing the expression root to 2,746 lines. The normalized helper
  is token-identical after explicit snapshot borrowing. Bound selection, access path, and the already-windowed
  survivor vector move by value without clones; GPU ORDER and LIMIT/OFFSET remain before every gather. Exact typed
  dispatch/default behavior, numeric scale, UUID little-endian reconstruction, int2 narrowing, bool/date/timestamp
  tagging, text layout/row count, validity-bitmap SQL NULL override, row-major flat output, targets, metadata, and
  errors are unchanged. Twenty-seven focused GPU controls and 35 independent-audit GPU controls pass, as do both
  505/487 engine modes, the complete 992-test serial GPU suite, all-target check, strict clippy, static/scoped gates,
  and independent audit. Generated test residue was removed. Runtime behavior did not change, so HAZARD and
  report-card gates were not applicable.

  STRUCT-001HI then isolated exact non-grouped GPU ORDER and post-sort LIMIT/OFFSET in the rustfmt-clean 354-line
  private `engine_expr/non_grouped_order.rs` leaf, reducing the expression root to 2,433 lines. The normalized phase
  is token-identical after explicit snapshot borrowing and the helper result wrapper; survivor indices move by value
  through ORDER, windowing, and projected-row materialization. Expression I32/I64 execution, nullable sentinel/errors,
  typed/text/heterogeneous/numeric/UUID key layouts, NULL/DESC masks, matrix slots, 64-key guards, bitonic/radix choice,
  permutation remapping, identity, and the sole conditional window copy are unchanged. The obsolete crate-root
  `ExprStep` import was removed. Twenty focused and 20 independent-audit GPU controls pass, as do both 505/487 engine
  modes, the complete 992-test serial GPU suite, all-target check, strict clippy, static/scoped gates, and independent
  audit. Generated test residue was removed. Runtime behavior did not change, so HAZARD and report-card gates were
  not applicable.

  The 2,433-line expression root now contains exactly one function: the cohesive resident SELECT/grouped GPU
  orchestrator. Every stable IR, predicate, join, source, route, scalar, DISTINCT, ORDER, and projection owner is
  already isolated. Splitting the remaining grouped pipeline would detach derived-buffer lifetime guards from their
  consumers or require a catch-all context bag, so the root is now an accepted `CODE_SIZE.md` exception with
  concrete growth, responsibility, contract, public-boundary, and 5,000-line re-review triggers.

  STRUCT-001HJ then moved the exact private `with_length_prefix` helper and all 26 startup-packet tests into the
  rustfmt-clean 641-line `crates/protocol/src/tests/startup.rs` child, reducing the protocol root from 10,271 to
  9,635 lines. The child is path-declared inside the existing inline test module; every name, attribute, body,
  strict length/minimum/mismatch check, SSL/GSS shape, cancel-key cap, v3 minor rule, parameter ordering/duplicate/
  empty/UTF-8/null/pairing case, and error variant is source-equivalent. The exact 71-test name-tail inventory is
  unchanged. The focused 26-test family and 71-test library pass, as does the full protocol package (71 library,
  127 binary, and both one-test driver integrations), protocol all-target check/strict clippy, server all-target
  check, connection-security preflight, scoped format/diff/source/reference gates, and independent audit. No
  production/API/visibility or runtime behavior changed, so GPU, HAZARD, and report-card gates were not applicable.

  STRUCT-001HK then moved the exact eight-test SET/session-control parser-facade family into the rustfmt-clean
  447-line `crates/protocol/src/tests/session_commands.rs` child, reducing the protocol root from 9,635 to 9,191
  lines. Direct source inventory corrected the initially promoted count from nine to eight before implementation.
  Normalized reconstruction is exact; quoted identifiers, whitespace separators, role/auth and transaction-
  characteristic aliases, CHECKPOINT/FLUSH, RESET/DISCARD/DEALLOCATE/CLOSE/LISTEN/NOTIFY, payload/dollar-quote
  cases, names, bodies, and errors are unchanged. The exact 71-test name-tail inventory is preserved. The focused
  eight tests and full protocol package (71 library, 127 binary, and both one-test driver integrations), protocol
  all-target check/strict clippy, server all-target check, security preflight, scoped gates, and independent audit
  are clean. Production bytes/API/visibility and runtime are unchanged, so GPU, HAZARD, and report-card gates were
  not applicable.

  STRUCT-001HL then moved the exact four-test transaction-command family into the rustfmt-clean 275-line
  `crates/protocol/src/tests/transaction_commands.rs` child, reducing the protocol root from 9,191 to 8,920 lines.
  Normalized reconstruction is exact; BEGIN/START, COMMIT/END, ROLLBACK/ABORT, WORK/TRANSACTION, AND [NO] CHAIN,
  isolation/read-write/deferrability modes, mixed ordering/comma/whitespace, duplicate-kind rejection, names,
  bodies, and errors are unchanged. The mixed 441-line control-command rejection matrix remains byte-identical in
  the parent. The exact 71-test name-tail inventory is preserved. Four focused tests and the full protocol package
  (71 library, 127 binary, and both one-test driver integrations), protocol all-target check/strict clippy, server
  all-target check, security preflight, scoped gates, and independent audit are clean. Production bytes/API/
  visibility and runtime are unchanged, so GPU, HAZARD, and report-card gates were not applicable.

  STRUCT-001HM then moved the exact single 441-line negative control-command parser matrix into the rustfmt-clean
  443-line `crates/protocol/src/tests/control_command_rejections.rs` child, reducing the protocol root from 8,920
  to 8,481 lines. Its broad historical function name is unchanged while the module explicitly owns malformed
  transaction modes/chains, FLUSH/CHECKPOINT extras, RESET/DISCARD/DEALLOCATE/CLOSE/LISTEN/UNLISTEN/NOTIFY,
  SET ROLE/AUTH/TRANSACTION/session-characteristic errors, exact error classifications, quoted identifiers,
  commas, typed/dollar strings, and unterminated inputs. Normalized reconstruction and the exact 71-test name-tail
  inventory are unchanged. The focused test and full protocol package (71 library, 127 binary, and both one-test
  driver integrations), protocol all-target check/strict clippy, server check, security preflight, scoped gates,
  and independent audit are clean. Production bytes/API/visibility and runtime are unchanged, so GPU, HAZARD, and
  report-card gates were not applicable.

  STRUCT-001HN then moved the exact three-test command-terminator family into the rustfmt-clean 140-line
  `crates/protocol/src/tests/command_terminators.rs` child, reducing the protocol root from 8,481 to 8,347 lines.
  Normalized reconstruction is exact; optional trailing semicolon/newline behavior across all covered command
  families, repeated terminators with intervening whitespace/newlines, terminator-only `ParseError::Empty`, parsed
  keys/values/chain bits/classifications, names, bodies, and errors are unchanged. The exact 71-test name-tail
  inventory is preserved. Three focused tests and the full protocol package (71 library, 127 binary, and both
  one-test driver integrations), protocol all-target check/strict clippy, server check, security preflight, scoped
  gates, and independent audit are clean. Production bytes/API/visibility and runtime are unchanged, so GPU,
  HAZARD, and report-card gates were not applicable.

  STRUCT-001HO then moved the exact six-test legacy GET/DEL parser-facade family into the rustfmt-clean 77-line
  `crates/protocol/src/tests/kv_commands.rs` child, reducing the protocol root from 8,347 to 8,276 lines. Normalized
  reconstruction is exact; DEL/DELETE/DELETE FROM aliases, exact keys, `DeleteKv`/`GetKv`, missing/extra arity,
  `InvalidDel`/`InvalidGet`, unsupported `DELETE TABLE`, names, bodies, and errors are unchanged. This remains
  test-only coverage of the existing bootstrap facade and does not broaden into relational DELETE or product
  direction. The exact 71-test name-tail inventory is preserved. Six focused tests and the full protocol package
  (71 library, 127 binary, and both one-test driver integrations), protocol all-target check/strict clippy, server
  check, security preflight, scoped gates, and independent audit are clean. Production bytes/API/visibility and
  runtime are unchanged, so GPU, HAZARD, and report-card gates were not applicable.

## Known boundaries

| Boundary | Work ID |
|---|---|
| 26 source files exceed the production/test/tool analysis envelopes in `CODE_SIZE.md` | **STRUCT-001** |
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
