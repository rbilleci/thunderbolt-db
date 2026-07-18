# PERF-001 remediation evidence — 2026-07-18

This is a historical test artifact. `docs/PLAN.md` owns the acceptance state and all remaining work.

## First audit findings and disposition

The first three-agent adversarial audit rejected the locally qualified implementation. All three reviews were
read-only and independent. Their material findings and the remediation in this candidate are:

| Finding | Remediation |
|---|---|
| A captured shard payload could be paired with a newer live device index during concurrent publication. | Route preparation validates the captured live cell before and after index preparation, rechecks the table generation under the publication lock, and cannot publish a stale route after replacement. |
| Prepared-route retention was unbounded, unaccounted, and vulnerable to purge-before-publish and explicit-transaction stale republish. | The cache retains at most one shape per table and 64 routes globally; descriptor allocations are charged to residency and the hard budget; store-before-purge publication closes republish; explicit/private transactions decline the shared route. |
| A database-global outer shard-map identity invalidated cached routes on unrelated-table writes. | Every table publication carries a stable per-table generation token. Known-table mutations rotate only that token; rare generic multi-table maintenance derives the exact changed-table set. |
| CUDA prepare, submit, and complete failures could collapse into an eligibility decline and retry/fallback. | Preparation/index-build and asynchronous submission/completion return typed execution errors. The facade fans an error to every waiter without retry. One-shot tests cover each of the three phases. |
| Public result contracts had been changed to an internal one-byte status and empty-range dense sentinel. | Public `CudaI32BatchProjectionColumns.status` remains `Vec<u32>` and public retained results contain one range per needle. Opaque internal dense result types carry the one-byte status and all-present identity through the production hot path. |
| Safe-plan validation did not fully reject malformed descriptor geometry. | Validation now checks context/pointer ownership, min/max ordering, projection and visibility spans, index geometry, checked capacities/sizes, and impossible row counts before launch. |
| The benchmark did not prove that the measured path and results were non-vacuous. | The production benchmark arm calls the compact production entry directly. Compatibility arms assert one real row per needle, and compact arms assert exact values. The earlier empty-public-range compatibility measurements are explicitly invalidated below. |

## Second audit findings and disposition

The first remediation was also rejected by three independent read-only reviews. Every material finding was
remediated before the follow-up audit:

| Finding | Remediation |
|---|---|
| A lock-free point read at boundary S could rebuild its index after a DELETE at S+1 using the newer GC boundary, permanently omitting the row still visible at S. | Cached device indexes carry their build GC boundary. Point reads build at their exact read boundary, and a cache hit is reusable only when its boundary is at least as conservative. A deterministic test purges the index after DELETE and queries the pre-DELETE boundary. |
| Descriptor invalidation cloned invalid shards with the old table token, allowing a paused G0 preparer to republish after purge. | Invalidation rotates one new per-table generation across every affected shard before storing the invalid publication, then purges routes while the descriptor publication lock remains held. A paused-prepublication test proves G0 cannot republish. |
| Route descriptor preflight/publication did not participate in the hard-budget allocation transaction. | Route publication takes the global budget allocation lock before the route publication lock and retains it through accounting preflight and cache store. A deterministic test holds the budget lock and proves publication waits. |
| A panic after CUDA launch or during pinned staging could unwind before a draining owner covered every in-flight resource. | Submission constructs its draining owner immediately after launch and before any fallible probe. Completion retains the stream in that owner through host allocation, asynchronous copy, probes, synchronization, and event readback; only then may it release the stream. Drop-without-complete pool-reuse tests pass. |
| NULL eligibility checked one shard generation and the GPU helper reloaded another. | The caller loads one shard map, performs the referenced-column NULL gate on that exact table-shard slice, and passes the same slice into route preparation. A deterministic generation-replacement test covers the interleaving. |
| The R3 helper counted only explicit ranges, so the private all-present dense identity reported zero; release used only `debug_assert!`. | `BatchedShardProjection::matched_needle_count()` understands dense identity, the helper uses it, and the release example uses `assert_eq!` against the exact expected count. |
| R2 advertised scan/index and atomic/dense A/B modes that all selected the same authoritative sharded dense route. | The example removes those controls and labels. It reports only the genuinely distinct public compatibility and production-compact result contracts over the same production sharded kernel, with exact-value assertions. |

## First follow-up findings and disposition

The contract/evidence reviewer accepted the second remediation. The publication and CUDA reviewers rejected three
remaining cases, now remediated:

| Finding | Remediation |
|---|---|
| A current-boundary route cached after DELETE could be reused by an older paused reader even though its index had omitted the older-visible row. Publication also favored the newer, narrower boundary. | Route cache hits now require `route.read_boundary <= reader_boundary`. Within one generation/shape, a newly prepared route publishes only when its boundary is older than the cached boundary; the older semantic superset remains reusable by every newer reader. The DELETE regression builds the current route first, then proves the old reader rebuilds and finds the row. |
| Multi-shard submit enqueued async H2D/memset, then ran a potentially panicking probe before constructing the draining submission. | The complete submission owner is constructed before the first async enqueue. Its Drop therefore synchronizes before owned staged needle bytes, pooled device buffers, the plan, or the stream can unwind. |
| Completion queued D2H into local pinned leases or Vecs that unwind before the by-value submission parameter. | A dedicated host-copy panic guard is declared after every D2H destination. On unwind it synchronizes first; only then may local pinned leases return to the shared pool or pageable Vecs free. A thread-local test hook injects panics immediately after H2D and D2H, followed by exact-result pool reuse. |

## Second follow-up finding and disposition

The CUDA and contract/evidence reviewers accepted the third remediation. The publication reviewer found one
remaining accounting interleaving:

| Finding | Remediation |
|---|---|
| Replacing `I-new` with an older-boundary `I-old` removed `I-new` from the accounted index map while a cached route still pinned it; another durable allocation could preflight before route publication later reacquired the budget lock. A losing prepared plan could similarly outlive index replacement before it reached publication. | Index replacement now holds budget -> route -> index locks, retires every cached route for the table while the old index is still map-accounted, then swaps the index entry before releasing the budget transaction. Route publication revalidates every prepared index Arc under that same lock order and declines caching a superseded plan. The deterministic DELETE test pauses before publication, proves the replaced index Weak is released, then races a different-shape older route against a paused newer plan and proves the accounted semantic-superset route remains cached. |

## Third follow-up finding and disposition

The contract/accounting reviewer generalized the replacement finding to another live removal family:

| Finding | Remediation |
|---|---|
| Incremental append removed an index directly when its load rule was crossed, while mutation publication retired routes only later. Fused append had equivalent direct-removal paths for load-rule rollover, bounded-probe overflow, and launch error. A cached route could therefore pin the removed allocation outside index-map accounting during another durable allocation's preflight. | Every append-side index exit now calls the central ordered route -> index purge. Incremental and fused append retire cached routes while the old index is still map-accounted before removal for load-rule rollover, bounded-probe overflow, or launch failure. The normal success path mutates the same accounted allocation in place and advances only its row-count basis. |

## Fourth follow-up findings and disposition

All three fresh final-tree reviewers rejected one further high-severity interleaving each:

| Finding | Remediation |
|---|---|
| Normal append could insert into `I1`, then update a concurrently installed same-pointer/count `I2` even though `I2` never received the key. Fused apply had the same success-publication gap. | Both success paths retain the launched index Arc and advance the cache basis only when the current Arc is pointer-identical. A deterministic fused/unfused gate installs a different allocation after launch, proves it retains the old basis, and proves the next lookup rebuilds with the committed key. |
| An index build could finish after DROP removed the current generation and completed its purge, then unconditionally insert a cache entry whose resident guard durably pinned the dropped payload outside payload accounting. | Under the route/purge lock and immediately before index-map publication, the builder revalidates the globally current shard payload Arc. A retired build may serve only its captured attempt transiently. A paused-build/DROP gate proves the durable cache remains empty after completion. |
| The safe submit API returned an independent asynchronous submission while H2D still read the caller's borrowed needle slice. A pinned caller could free or mutate that source before DMA completed. | Single- and multi-shard dense submissions copy the exact needles into pooled pinned staging when available, otherwise an owned pageable Vec, and retain that source guard through the covering stream synchronization. The draining submission exists before the first async enqueue. |

## Fifth follow-up findings and disposition

The next independent review rejected three remaining enforcement/evidence gaps:

| Finding | Remediation |
|---|---|
| The public single-shard dense safe entry accepted an index from another CUDA context, an allocation too short for `mask + 1` slots, an incoherent hash shift, or a misaligned projection offset even though the multi-shard entry rejected them. | Single-shard submission now requires a nonempty representable row count, pointer-identical resident/index primary context, checked table-mask/hash-shift geometry within the retained index allocation, and aligned projection offsets before launch. The actual-GPU drop/pool gate covers short index memory, incoherent shift, and misalignment. |
| Repeated benchmark shard installation replaced side-map payload `P0` with `P1` without purging a cached PK index whose resident guard still owned `P0`, hiding the old payload from hard-budget accounting. | `RelationalResidentCache::install_shards` now publishes the new side-map cells, centrally performs ordered route/index retirement, and only then publishes the replacement descriptors. A deterministic actual-GPU gate builds the old index, reinstalls the table, proves both old index and payload Weak owners retire, and proves accounting contains only `P1`. |
| The fused append Arc-identity race test accepted any earlier setup hit, so its target insert could fall back to the unfused path and still pass. | The gate snapshots `fused_apply_hits` immediately before the target append and requires an exact delta of one for its fused half and zero for its unfused half. |

## Sixth follow-up findings and disposition

The next three independent reviewers rejected four additional safe-boundary, generation, and evidence gaps:

| Finding | Remediation |
|---|---|
| A structurally valid single-shard index could encode `row + 1` beyond the resident row count; dense and atomic kernels decoded that device value and gathered without checking it. | Both kernels now receive the validated row count and reject a decoded row before any gather/output append. The actual-GPU gate installs a valid two-slot index whose matching entry names row 1 for a one-row payload; dense reports not-found and atomic reports no row without a device fault. |
| The older public atomic scan/index deferred submissions still enqueued H2D from `needles.as_ptr()`, returned independently, and owned no source bytes. Their complete submission owner was also assembled only after enqueue/launch. | The shared `I32NeedlesHostGuard` copies every atomic/dense input to pooled pinned memory or an owned Vec. Atomic scan/index submission owners are constructed before the first enqueue and drain before releasing that guard. The pool gate proves the staged pointer differs from the caller, mutates the caller after submit, and obtains the original exact rows. The atomic index entry also receives the same context, mask/shift/allocation, alignment, and extent preflight as dense. |
| Synthetic benchmark replacement could leave prior DELETE/CREATE/row-id sidecars published. A new payload index reloaded stale global DELETE state, causing false misses or extent failure, while the old sidecar bytes escaped current-descriptor accounting. | Common shard installation republishes the exact sidecar maps carried by replacement descriptors and tombstones absent prior regions before descriptor publication. Index construction consumes the DELETE Arc from the same captured descriptor generation as its payload instead of reloading a global cell. A normal insert/delete-all generation is replaced by synthetic all-live rows; the gate proves the old sidecar retires, accounting contains only the replacement before index build, and the new key is found. |
| The R3 attribution example labelled its default 1M-row/4MB column out-of-L2 on the documented 128MB-L2 GPU, described retired per-shard D2H build/gather behavior, and printed only cumulative binary-route hits. | R3 now describes the production prepared-plan/single-kernel path, labels only row/shard geometry, explicitly defers cache-regime evidence to the canonical report card, snapshots counters after warmup/single-flight, and requires a positive measured-loop binary-hit delta for the labelled many-shard case. |

## Seventh follow-up findings and disposition

The publication/accounting reviewer accepted the sixth remediation. The CUDA and contract/evidence reviewers
found three remaining safe-boundary cases and four evidence-record inconsistencies:

| Finding | Remediation |
|---|---|
| A public read view copied a raw device pointer and primary context but did not retain the allocation; dropping the owner freed memory still addressable by the view. A deferred submission could likewise outlive its owner/view. | Every raw allocation transfers into one shared allocation-lifetime guard. The owner, every cloned read view, and every atomic/dense deferred submission retain that exact guard. An actual-GPU regression drops both facade handles before completion and obtains the exact expected rows. |
| Atomic completion removed its stream from the submission before initial synchronization and queued three result D2H operations without a panic guard for pinned/pageable destinations. | The submission retains the stream through the successful completion tail. Separate local guards cover count and result D2H and synchronize before any destination/lease unwinds. An injected panic after the first result copy is followed by exact pool reuse. |
| Multi-shard status 3 was copied into the public dense compatibility result, whose row conversion silently kept only status 1 and therefore translated a duplicate decline into not-found. | Opaque compact production completion retains status 3 for engine re-resolution. Public compatibility completion returns `DuplicatePointReadMatch`, and public row conversion fails loud on any invalid dense status. |
| Timed R3 SQL 3b results were discarded without proving row/value, GPU execution, or measured route hits; stale closing prose described the former result path. | Every timed lookup now asserts the exact requested row/value and GPU target, the loop requires a positive shard-index hit delta, and the closing text names the current canonical production-compact point card. |
| R2 one-caller summaries and concurrent performance rows reported throughput without p50 despite the canonical card's every-line contract. | Concurrent workers retain per-batch samples while aggregate wall time still supplies throughput. Every detailed, comparison, summary, and concurrent performance row now carries p50 latency plus throughput. |
| PLAN claimed unsupported corrected-card values and causal attribution, while STATUS/archive recorded only corrected in-L2 evidence and a valid pre-correction out-of-L2 compact arm. | PLAN now records 231.565M/s corrected in-L2, 199.671M/s valid pre-correction out-of-L2, 12.3%/27.6% gaps, and explicitly treats fragmentation as an inference pending the corrected complete card. |
| The ledger's 1,024 engine count predated two added regressions. | The exact candidate passes 1,026/1,026 engine tests and 129/129 execution tests including ignored actual-GPU coverage. |

## Eighth follow-up findings and disposition

The publication/accounting reviewer accepted the seventh remediation. The CUDA and contract/evidence reviewers
found four remaining proof and measurement gaps:

| Finding | Remediation |
|---|---|
| The allocation-lifetime regression submitted work before dropping the owner. Because `cuMemFree` synchronizes, the old broken implementation could finish the kernel during owner drop and still complete from temporary outputs. | The regression now creates a read view, takes a test-only non-owning Weak witness for the exact allocation, drops the owner, and only then submits through the surviving view. It separately proves the view retains the allocation, the deferred submission retains it after view drop, exact GPU rows complete, and the final guard releases at completion. |
| Status 3 was covered only by a fabricated host compact result; no device execution proved the multi-shard kernel emits it or that pooled state survives both completion contracts. | The actual-GPU multi-shard gate builds two valid shards with the same visible key. Compact completion observes status 3, a fresh compatibility completion returns `DuplicatePointReadMatch(0)`, and a subsequent one-shard completion reuses the pools and returns exact status/value bytes. Public compact-result docs define status 1 found, 2 not found, and 3 duplicate/decline plus the compatibility conversion. |
| The raw GROUP BY report-card line claimed p50, but `group_by_i32_count_sum_kernel_timed` retained only the minimum CUDA-event sample across ten runs. | The benchmark-only helper now retains and sorts all event samples and returns nearest-rank p50. Its API docs, raw-card label/comments, and radix probe all name p50 rather than a minimum. |
| R3 proved only positive route-counter deltas, so a partially attributed measured loop could retain the SQL 3b, GPU-probe, or binary labels. | The SQL loop requires its shard-index delta to equal its call count. Batched and GPU-probe deltas must each equal `batch_sizes * batches`; the many-shard binary delta must equal that same count, while one-shard regimes require zero binary hits. |

Both focused actual-GPU regressions pass. The exact execution tree passes **50 active tests** and **129/129**
including ignored actual-GPU tests in 18.20s. Workspace all-target/all-feature check, strict
execution/engine/facade Clippy, formatting, diff whitespace, and changed-source size gates pass. No command used
`--gpu-reset`.

## Ninth follow-up findings and disposition

The restarted publication/accounting and CUDA ownership reviewers accepted the exact eighth-remediation tree. The
contract/evidence reviewer found two remaining raw-card presentation defects:

| Finding | Remediation |
|---|---|
| The isolated gather-kernel row performed one event-timed launch and labelled that single value p50. The closing roofline summary printed throughput/ratio without the associated latency despite the every-performance-row contract. | Isolated gather now records one CUDA-event sample per configured timed iteration and reports their p50. `run_scan_pass` retains the `sum_i32` p50 with its GB/s, and the closing in-L2/out-of-L2 comparison prints p50 latency plus throughput for both regimes. |
| The archive retained the old **1,677.7M elements/s** minimum-based GROUP BY value in a list introduced as valid raw evidence, even after documenting the statistic defect. | The historical number is explicitly invalidated as p50 evidence. A corrected standalone raw run reports GROUP BY at **5.018ms p50 / 1,671.8M elements/s**; the complete corrected card remains pending. |

The corrected standalone raw benchmark completed both cache regimes. `sum_i32` measured **1,476.9 GB/s at p50
23us** in-L2 and **1,450.6 GB/s at p50 185us** out-of-L2; isolated gather measured **349.5 GB/s at p50 6us**
and **155.3 GB/s at p50 108us** respectively. Formatting, strict example Clippy, example compilation, and diff
whitespace pass. No command used `--gpu-reset`.

## Tenth follow-up findings and disposition

The next full read-only pass found no runtime or measurement defect. It found two low-severity labels:

| Finding | Remediation |
|---|---|
| Layer-1 module/header prose said all algorithmic rows were `SORT_N`-sized, while GROUP BY actually runs a full `ROWS` pass. | Module docs, source comment, and emitted section header now distinguish sort/join at `SORT_N` from GROUP BY at full `ROWS`. |
| Public `CudaI32IndexProbeDenseSubmission` rustdoc said every thread writes only status 1/2 although the same owner represents multi-shard status 3. | The public rustdoc now distinguishes single-shard 1/2 from multi-shard 1/2/3 duplicate decline and documents rejection of zero/unknown status. |

Formatting, workspace all-target/all-feature check, strict roofline-example Clippy, the 50-test active execution
suite, and diff whitespace pass. No command used `--gpu-reset`.

## Eleventh follow-up findings and disposition

Publication/accounting and contract/evidence accepted the tenth tree. CUDA ownership found no runtime defect and
two low prose-scope contradictions: public submission docs attributed zero/unknown rejection to compact completion
instead of compatibility/engine consumers, and an internal PTX comment said probing stopped at the first match
although the kernel deliberately continues to detect a duplicate. Both comments now describe the implemented
opaque compact-status and duplicate-detection contracts exactly. Formatting, workspace all-target/all-feature
check, active execution tests, and diff whitespace pass. No command used `--gpu-reset`.

## Twelfth follow-up findings and disposition

The next freeze pass found no runtime defect and identified the remaining stale dense-path prose as low severity:
prepared-route docs ignored binary candidate selection and duplicate continuation, single/multi status-clear comments
attributed validation to completion, and compatibility completion claimed it compacted into atomic-style indices.
A scoped module comment audit now describes linear versus binary candidates, second-match status 3, single-shard
1/2 initialization, compact preservation, compatibility/engine validation, and the compatibility layout's empty
needle/row-index vectors exactly. Formatting, workspace all-target/all-feature check, active execution tests, and
diff whitespace pass. No command used `--gpu-reset`.

## Thirteenth follow-up findings and disposition

Publication/accounting and contract/evidence accepted the twelfth tree. CUDA ownership found no runtime defect and
two last low stale phrases outside the prior comment search: the public shard descriptor said one launch probed all
shards despite binary candidate routing, and a PTX preamble's “only” list omitted range routing, zone pruning, and
MVCC visibility. The descriptor now says linear candidates versus one binary-range candidate; the preamble names
the shared hash/gather contract and every added responsibility without an exclusivity claim. A broader module
keyword audit found no remaining version of the stale first-match/all-shards/debug-assert/compat-compaction claims.
No command used `--gpu-reset`.

## Final acceptance — canonical report card

All three independent exact-tree lanes finally returned **ACCEPT** with no severity finding: publication/accounting,
CUDA ownership/public contracts, and benchmark/contract/evidence. The canonical
`scripts/benchmark_report_card.sh` then completed with exit 0 on the RTX PRO 6000 Blackwell Max-Q, driver
595.71.05, covering both layers and both cache regimes.

Layer 1 raw evidence:

| family | in-L2 p50 / throughput | out-of-L2 p50 / throughput | disposition |
|---|---:|---:|---|
| `sum_i32` roofline | 26us / 1,313.7 GB/s | 187us / 1,433.8 GB/s | run-local roofline |
| `equal_any` | 54us / 624.4 GB/s | 1,922us / 139.6 GB/s | stable versus prior card |
| compare count | 27us / 1,223.7 GB/s | 185us / 1,449.9 GB/s | 0.93x / 1.01x roofline |
| between count | 55us / 608.4 GB/s | 369us / 727.8 GB/s | 0.46x / 0.51x roofline |
| constant-mask output | 31us / 1,097.9 GB/s | 184us / 1,458.4 GB/s | 0.84x / 1.02x roofline; no host staging |
| gather i32 kernel | 6us / 349.5 GB/s | 108us / 155.3 GB/s | multi-sample event p50 |
| sort | 3,013us / 348.1M elem/s | not in cache sweep | stable single algorithmic regime |
| hash join | 4,058us / 258.4M elem/s | not in cache sweep | stable single algorithmic regime |
| GROUP BY kernel | 5.012ms / 1,673.5M elem/s | not in cache sweep | corrected full-`ROWS` event p50 |

Layer 2 production result-path evidence at batch 65,536:

| cache regime | compat-public | production-compact | compact/public |
|---|---:|---:|---:|
| in-L2, 1,048,576 rows / 526 shards | 8,118us / 7.931M lookups/s | **156us / 232.641M lookups/s** | 29.33x |
| out-of-L2, 48,000,000 rows / 24,001 shards | 2,834us / 22.020M lookups/s | **203us / 198.933M lookups/s** | 9.03x |

The out-of-L2 production result is within 0.4% of the valid pre-correction 199.671M/s arm. The large fixture built
in **1,633.7s insert + 0.0s residency**, proving the accepted result uses device-authoritative insert publication
rather than late conversion. Relative to the retired comparison-only unified evidence, the remaining gaps are
**11.9% in-L2 / 27.8% out-of-L2**. Required needle/result transfers are shared by both representations;
fragmented payload/descriptor traversal remains a bounded inference, not proven causal attribution.

No read-kernel ratio or algorithmic-rate regression signal fired. PERF-001 is accepted and removed from the open
PLAN ledger; RETIRE-003 is promoted to NOW. No command used `--gpu-reset`.

## Permanent attribution

The build-only `probe-timing` feature records descriptor bytes/shards, needle H2D, values/status D2H, CUDA-event
kernel time, submission, completion, and assembly. On the 1M-row, 526-shard, 65,536-needle case:

- first descriptor encode/upload: about **27us / 31us**, **46,200 bytes**, **525 non-empty shards**;
- cached descriptor enumeration: **3–7us**;
- per batch: **262,144 H2D bytes** and **589,824 D2H bytes**;
- kernel event: **11–13us**;
- cached submission: roughly **54–68us**, completion usually **85–95us**, assembly **42–78us**.

The earlier fixed-row 2,049-shard case improved to **97.5M lookups/s at p50 547us** with cached preparation at
**8–16us** and kernel time at **12–13us**. This removes the measured per-batch descriptor/shard-count host slope.

The H2D/D2H traffic is required by both this implementation and the retired unified comparison, so it is not a
causal explanation for the residual gap. The remaining fragmentation cost is bounded but not causally isolated
without restoring the forbidden retired representation: the current evidence is consistent with fragmented device
payload/descriptor traversal, but does not prove that attribution.

## Performance evidence

The complete card captured before the public-range correction retained valid raw scan/sort/join and
production-compact arms. Its old minimum-based GROUP BY line is historical only and invalid as p50 evidence:

- raw in/out-of-L2 `sum_i32`: **1,468.4 / 1,450.6 GB/s**;
- raw in/out-of-L2 `equal_any`: **639.6 / 140.9 GB/s**;
- raw in/out-of-L2 compare count: **1,284.1 / 1,451.5 GB/s**;
- raw in/out-of-L2 between count: **654.8 / 728.5 GB/s**;
- constant-mask output: **1,172.0 / 1,490.8 GB/s**;
- sort / hash join: **348.7M / 257.1M elements/s**;
- grouped kernel historical minimum: **1,677.7M elements/s — invalid as p50 evidence**; final canonical evidence is
  **1,673.5M elements/s at p50 5.012ms**;
- production compact route: **232.594M/s at p50 154us** in-L2 and **199.671M/s at p50 201us** out-of-L2.

The earlier card's `scan` and compatibility `lpb` rows used the internal empty-range sentinel through a public
wrapper and were vacuous; those two rows are invalid evidence. The separately labelled `dense` row invoked the
same compact production route, so it is redundant rather than distinct A/B evidence. After restoring one public
range per needle and wiring the benchmark explicitly:

- compatibility `scan` and `lpb` materialize real per-needle rows at about **8.04ms p50 / 7.95–7.97M/s**;
- the non-vacuous production compact route reaches **231.565M/s at p50 154us** in-L2;
- a 300-batch confirmation reaches **232.87M/s at p50 152us**.

R2 now exposes only `compat-public` and `prod-compact`: distinct result-materialization contracts over the same
production authoritative-shard kernel. It no longer claims nonexistent scan/index or atomic/dense route controls.

Relative to the retired late-converted unified evidence (**264.2M/s** in-L2, **275.7M/s** out-of-L2), the corrected
final-card production measurements leave about **11.9%** and **27.8%** throughput gaps. The retired representation
remains comparison evidence only, not an executable option.

## Correctness and safety gates

- Engine ordinary debug and release modes: **487 passed** in each.
- Engine including ignored actual-GPU tests: **1,026 passed**.
- Execution including ignored actual-GPU tests: **129 passed**.
- Facade all-target ordinary suite: **39 passed, 8 GPU-ignored**; serialized concurrency integration:
  **13 passed, 1 GPU-ignored** in the ordinary invocation.
- The nine-test sharded point-read family passed three sequential and two simultaneous HAZARD invocations.
- The multi-shard execution gate injects and catches panics after H2D and D2H, then proves the same pooled plan
  returns exact results; the complete execution suite remains **129/129**.
- Workspace all-target/all-feature check, strict execution/engine/facade Clippy, formatting, diff whitespace, and
  changed-source size gates pass. All changed production modules remain below the 2,000-line audit threshold and
  all changed tests/examples remain below 3,000 lines.

No command used `--gpu-reset`.
