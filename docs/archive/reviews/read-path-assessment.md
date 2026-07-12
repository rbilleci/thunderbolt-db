# ARCHIVED — Read-path assessment

> Historical point-in-time analysis. It is not an executable plan. Any surviving obligation is tracked only
> in `docs/PLAN.md`.

**Date:** 2026-07-02
**Method:** Static code analysis only (no runtime/benchmarks). Every finding is anchored to `file:line`.
**Target state assumed:** a GPU-native relational engine (data-plane execution on the GPU; CPU is control
plane only) whose system of record is a Write-Ahead Log, per `docs/CHARTER.md` / `docs/ARCHITECTURE.md`. The
CPU relational engine is interim parity/safety debt scheduled for deletion (ADR-006 / S10d).

This report is the counterpart to `docs/reviews/write-path-assessment.md`. It covers the *read/query* path
end-to-end: statement ingress → parse/route → resident-GPU vs host dispatch → MVCC read visibility →
residency/shard read → result materialization. Findings are rated against the **GPU-native target**, not the
current default. Where a finding is already tracked in `STATUS.md` / `HANDOVER.md` / the `scalability-ledger`,
that is noted so novel issues stand out. **Focus is the GPU path**; the host path is assessed only for what
blocks deleting it.

> **Baton-drift warning up front (see §7):** several statements in `HANDOVER.md` / `STATUS.md` about the read
> path are **stale**. Chiefly: the persistent **wave read engine was retired and archived** (`docs/future/redis/
> wave.rs`, `-6180 LOC`, DECISIONS 2026-06-29) — it is *not* wired, and the `wave_*_enabled` flags the baton
> lists do not exist in compiled code. The production read data plane is the **launch-per-batch dense index
> probe + scan fallback**. This report describes the code as it actually compiles at HEAD `52e1b6be`.

---

## 1. Read path as actually built (traced from code)

Two live ingress paths, both in `crates/facade/src/lib.rs`:

| Server mode | SELECT entry | Read dispatch |
|---|---|---|
| Shared/concurrent (`serve()` → `SharedEngine`) | `execute_on_shared_engine` (`:375`) | `engine.execute_relational_select(&select)` (lock-free, poison-checked) |
| Async server + batcher (`serve_async`) | `execute_on_shared_engine_batched` (`:522`) | int4-equality point lookups → `PointLookupBatcher`; everything else → `execute_on_shared_engine` |

The production binary (`crates/server/src/main.rs`) runs the **sync** `serve()` over a fresh
`SharedEngine::new()` → `Engine::new_local()`; the point-lookup batcher exists only on the async path.

**Engine dispatch** (`crates/engine/src/engine_select_exec.rs::execute_relational_select_instrumented:122`):
1. Pin one read boundary `s = committed_seq()` (`:154`); resolve the catalog **as-of** `s` (`:155`).
2. Views / materialized views / synthesized `pg_catalog`+`information_schema` relations are served **on the
   host** (`:156-214`, `DeviceTarget::Cpu`).
3. `plan_relational_resident_route(select)` (`:215`). If **accepted** →
   `execute_relational_select_with_resident_route` (GPU); on a residency-invalidated probe error it transparently
   falls back to `execute_relational_select_cpu_pinned` (`:230-232`).
4. Otherwise → `execute_relational_select_cpu_pinned_instrumented` (`:236`) — the **host CPU relational engine**
   (`engine_mvcc_dispatch.rs` routes it through `CpuMvccExecutionBackend`).
5. Results reach the facade as an owned `RelationalSelectResult` and are re-materialized to `QueryOutcome::Rows`
   (`lib.rs:273-279` / `392-398`); the server encodes that to pgwire `DataRow`s.

**GPU resident route** (`plan_relational_resident_route` → `resident_route.rs` shape classifiers): the route is
accepted only for an enumerated set of **int4** shapes on a **GPU-resident** table. Point lookups take the O(1)
launch-per-batch dense hash-index probe (`engine_retained_read.rs::submit_resident_int4_equal_any_payload:487`,
`index_probe_enabled`/`dense_index_probe_enabled` **default-ON**, `engine_lifecycle.rs:172/176`); scans /
scalar-aggregates / grouped / ordered / distinct run the general `ResidentExpr` executor
(`engine_expr.rs::execute_resident_expr_select_with_binding`); multi-shard tables recompact-then-execute
(`engine_expr.rs::execute_resident_sharded_via_general:2368`), all behind **default-OFF** shard flags.

**The production default is still host-side reads.** `auto_admit_on_commit` is **false**
(`engine_lifecycle.rs:168`), so a freshly committed table is **not** GPU-resident, so its reads take the host
`execute_relational_select_cpu_pinned` path. The GPU resident route only fires for a table warmed to residency
explicitly. Deleting the host path (S10d) is gated on flipping auto-admit (S-F) **and** closing the shape gap in
§4 (STATUS "blocking gap").

**What is sound (credit where due).** The kernel/gather audit found **no live wrong-results defect** in the
probe/gather path: the multi-shard duplicate-key guard (`status=3` → host declines,
`execution/src/lib.rs:10900-10963` / `engine_retained_read.rs:1463`), the O(1) **binary shard-routing**
self-validation (`windows(2).all(max<min)`, cannot mis-route), 8-aligned 64-bit loads (no 716 hazard),
host/kernel hash agreement, index-cache ABA protection (`(ptr,row_count)` + `Arc`-pin), and lease/UAF discipline
(all derived buffers pinned; `Drop` stream-syncs before pooling) all check out. Residency **retire-path hygiene**
is clean — `device_memory`, `shard_deleted_by/created_by_memory`, and host+device PK-index caches are purged at
all invalidate/evict/drop/re-admit sites. The **`created_by` read gate is implemented** on every GPU read path
(SV6: `created_by <= read_txn` ANDed with `deleted_by > read_txn`, `engine_expr.rs:1985-2009`,
`engine_retained_read.rs:1160-1182`). Empty filtered `SUM/MIN/MAX/AVG` correctly returns SQL **NULL**
(`engine_expr.rs:6382-6404`), and single-buffer `LIMIT/OFFSET` without `ORDER BY` is deterministic
(ascending-by-construction compaction). The lock-free one-`committed_seq`-pin read design (catalog↔data
co-pinned) is a genuinely good shape.

---

## 2. Snapshot-consistency: the read-side correctness cluster (highest priority for flipping the GPU data plane ON)

These four findings share one root cause: **the read pins `committed_seq` once, but resolves device state
(shards, buffers, MVCC version regions) through *separate* lock-free `ArcSwap`/`get()` loads that are not
co-pinned to that boundary.** The point-lookup routes were hardened against this (3b captures
`(descriptor, device_memory, deleted_by, created_by)` from one `shards.load()` and `Arc`-pins the buffer,
`engine_retained_read.rs:746-757`); the **scan/recompaction path was not**. All are **inert on the default
deployment** (non-resident reads go host-side; incremental resident writes are default-OFF) but are precisely the
gates to clear before flipping `auto_admit_on_commit` / `resident_update_tombstone_enabled` / `shard_residency_enabled` on.

### C1 — Resident scan reads the *published* device generation, not the reader's `committed_seq` 🟠 High · Partly tracked (HANDOVER A, write side)
The resident route resolves rows against `residency.shards.load()` / `device_memory.get()` as they are *now*, not
as-of the bound `s`. The design comment concedes it: `relational_select_mvcc_query_pinned` "does not pin the
device generation… consistency is enforced by the residency generation's `valid_through_index`/
`invalidated_at_index`, not this pin" (`engine_select_bind.rs:41-47`). Consequences once auto-admit + concurrent
writes are on: an `s = C-1` reader can observe `C`'s rows. UPDATE/DELETE are covered by the device
`created_by/deleted_by` gates *when regions exist*, but a **born-visible in-place INSERT append** bumps
`descriptor.row_count` **before** `publish_committed_seq` (`engine_residency.rs:3775-3791`), so a `C-1` reader
sees the `C` INSERT (premature visibility). **Fix:** stamp each published shard generation with its `commit_seq`
and validate against the bound `s` (decline/fallback on skew), or stamp INSERT appends with `created_by` too.

### C2 — Scan recompaction pairs a gen-N shard descriptor with a gen-N+1 device buffer (TOCTOU → wrong rows / OOB device read) 🟠 High · Novel
`execute_resident_sharded_via_general` clones the shard descriptor list (`shards.load()...cloned()`, guard
dropped, `engine_expr.rs:2386`), then **later**, per shard, independently `shard_device_memory.get()`s the buffer
(`:2493`). These are two separate `ArcSwap`s, and `install_shards` republishes the **buffer first, descriptors
second** (`resident_storage.rs:832-835`). So a concurrent serialized re-admit/rollover can hand the reader the
**gen-N+1 buffer** while it computes byte offsets from the **gen-N** `row_count`/`capacity`
(`src_byte_offset = 8 + ordinal*capacity*4`, `:2601`). A grown buffer → silent wrong rows/columns; a
reshaped/smaller buffer → **out-of-bounds source read** (the DtoD copy bounds-checks only the destination,
`:2560`). Distinctly, the recompaction fetches `shard_deleted_by_memory`/`shard_created_by_memory` with their own
`.get()`s (`engine_expr.rs:2633/2638/2659`) also not pinned to the buffer generation → misaligned slot
visibility. **Fix:** co-publish descriptor+buffer atomically (embed the buffer `Arc` in `RelationalResidentShard`
so one `shards.load()` yields both), or cross-check the buffer's `device_ptr`/size against the descriptor's
`allocated_bytes` in `source_for` and decline on mismatch. (Note `CudaDeviceMemoryProof` carries no `device_ptr`
today — the exact check needs that field.)

### C3 — Resident `COUNT(*)` is visibility-blind (no `deleted_by`/`created_by` gate, no is-versioned guard) 🟠 High · Partly tracked (mvcc-visibility review §4)
`run_resident_count` counts **all physical rows** — unfiltered returns `snapshot.row_count`; filtered runs
`count_i32_{equal,compare}_from_payload` over `row_count` with **no** visibility conjunct
(`engine_resident_probe.rs:411-470`). On a versioned buffer it would count tombstoned + not-yet-visible rows as
live, and it inherits the C1 premature-INSERT count. It is inert only because the single-store buffer *declines*
versioned appends — but the scan path applies visibility while COUNT does not, so the two **diverge** the moment
COUNT routes over a versioned buffer. **Fix:** gate resident COUNT on the version synopsis; fall back to a
filtered reduction with the `deleted_by/created_by` conjuncts when versioned.

### C4 — The fully-GPU dense probe kernel applies no visibility gate; correctness rests on an unenforced decline 🟠 High · Novel (fragility)
The device-native multi-shard dense probe applies **neither** the `created_by` nor `deleted_by` gate; it stays
correct only by the host **declining** any shard that carries a version region
(`engine_retained_read.rs:1364-1386`). The code labels this "DEFENSIVE… UNREACHABLE-to-violate today" but notes
it "becomes **LOAD-BEARING** the moment `deleted_by` regions can be reclaimed independently (VACUUM/GC, ledger
#5)." So the read-side flip-gate is met by an emergent ordering property, not an enforced invariant. **Fix:** tie
the dense-kernel version-free precondition to GC as an asserted invariant, and land the concurrent-reader
double-read differential HANDOVER option A calls for **before** flipping the resident-write flags.

---

## 3. Other correctness / PG-divergence findings

### C5 — `GROUP BY <expr>` evaluates the key over ALL rows before WHERE (overflow-before-filter) 🟠 High · Tracked (STATUS charter-debt a) — with a correction
`engine_expr.rs:4980-5006`: the group-key expression is materialized over the **full** `row_count`
(`arith_value_column_device`, `:5001`) and only read at post-filter survivor `indices` by the grouping kernel. A
row WHERE would drop still has its key expr checked-evaluated, so a filtered-out overflow **errors** where PG
(which evaluates the key only for survivors) succeeds — stricter than PG (ADR-011). **Correction to the tracked
note:** STATUS says "ORDER BY/GROUP BY `<expr>`", but **ORDER BY was already fixed** — it evaluates at survivor
indices (`arith_value_column_at_indices`, `engine_expr.rs:6646/6892`). **Only GROUP BY `<expr>` remains.**
**Fix:** mirror ORDER BY — gather survivors, evaluate the key at those indices, group by compact position.

### C6 — Batched point-lookup returns a spurious error (not correct rows) when single-buffer residency is lost mid-flight 🟠 High · Novel
`classify_batchable_point_lookup` accepts a lookup only if the table is resident *at classify time*
(`lib.rs:554-585`). The coalescer later prepares the retained template; if a concurrent writer/DDL invalidated
that single-buffer residency in the gap, `prepare_relational_retained_read_template` fails and — because
`resident_shard_count == 0` — the code calls `fail_group` (`point_lookup_batcher.rs:499-513`, error at `:509`),
sending a hard engine error to **every** waiter. The unbatched path answers the identical query correctly (it
falls back to the host CPU-pinned read on residency invalidation, `engine_select_exec.rs:230`). This violates the
documented invariant "misclassification only ever costs a slow path, never a wrong result" (`lib.rs:514-517`).
**Fix:** on any single-buffer prepare/submit failure that is a residency/generation loss, fall back to
`run_group_per_query` (already done for the shard-decline case at `:507`) instead of `fail_group`.

### C7 — Batcher `prepare`→`submit` window is not generation-pinned 🟡 Medium · Novel
The header claims "one pinned generation per batch," but `run_group` calls `prepare_…template` (`:499`) and
`submit_…point_lookups` (`:530`) as two separate `&self` calls; the engine only pins across submit→complete
(`engine_retained_read.rs:410-425` re-validates `handle.generation == template.snapshot_generation`). A commit in
that microsecond window makes `submit` return a generation-mismatch `Err`, again `fail_group`'d to all waiters.
**Fix:** pin one `committed_seq` across the whole prepare→submit→complete of a group, or retry-prepare once on
mismatch.

### C8 — `WHERE col IS NULL` / `IS NOT NULL` on a sharded-only table ERRORS 🟠 High (clean-error, no wrong rows) · Tracked (HANDOVER B)
The sharded dispatch is a fixed shape allow-list with **no IS NULL shape** (`engine_select_exec.rs:410-427`);
`resident_route_query_shape` can't classify an IS-NULL predicate, so it never reaches the sharded executor and
falls to the single-buffer path, which errors `"relation … has no resident snapshot"` (a sharded table has no
unified snapshot). **Fix:** make IS NULL/IS NOT NULL sharded-router-eligible and route it to
`execute_resident_sharded_via_general` — the unified descriptor already carries the null bitmaps
(`engine_expr.rs:2683-2755`) and the executor already lowers `IsNull` against them (`:1734-1745`).

### C9 — Single-buffer batched path has no NULL decline (unlike the sharded gather) — latent NULL-as-0 blind spot 🟡 Medium · Partly tracked (M3)
`distribute_results_batched` maps every value as `DbValue::Int4(v)` with no validity channel
(`point_lookup_batcher.rs:605`; the batch result is raw `i32`, "NULL already encoded as 0"). The sharded gather
*declines* null-bearing tables so the NULL-aware scan serves them (`engine_retained_read.rs:1124-1141`), but the
single-buffer `submit_resident_int4_equal_any_payload` path has **no** equivalent decline, and the route gate for
`int4_equality_projection` never inspects nulls. Correctness rests on an unenforced, one-sided cross-crate
invariant; a projected SQL NULL becomes `Int4(0)`. (No proven live divergence today because the per-query path is
equally null-blind on this shape — so both are uniformly wrong, not divergent — but it is the batcher's blind
spot.) **Fix:** carry a validity channel end-to-end, or have the single-buffer submit decline null-bearing tables
the way the sharded gather does; add a null-bearing single-buffer int4 test.

### C10 — `col = x` NULL-exclusion (3VL) untested on the sharded path 🟡 Medium · Tracked (HANDOVER B2)
Structurally correct (the sharded scan re-lowers the predicate through the VM that ANDs value-operand validity,
`engine_expr.rs:1618-1631`), but there is no sharded-vs-single-buffer differential proving `NULL = x` →
UNKNOWN → excluded. Test-coverage gap, not a confirmed defect. **Fix:** add the differential.

### C11 — `COUNT(DISTINCT nullable)` / `GROUP BY nullable-key + COUNT(DISTINCT)` hard-error 🟡 Medium · Partly tracked (M3 3VL follow-up)
`engine_expr.rs:6571-6577` / `5282-5294` / `5263-5274` raise `ApplyFailed("… not yet supported (M3 3VL
follow-up)")` instead of computing the PG answer (count distinct **non-NULL**). Clean-error, but a PG-feature gap
on the GPU read path not in the charter's two-item list. **Fix:** AND value-validity into the distinct pass.

### C12 — Route-gate case-insensitive vs exact-match exec 🟡 Medium · Tracked (STATUS, rated "never wrong rows")
`resident_route_grouped_aggregate_shape` (`resident_route.rs:388`) and `select_is_gpu_sortable_projection`
(`engine_select_exec.rs:39`) match columns with `eq_ignore_ascii_case`, whereas the shape matchers /
`relational_column_index` use exact `==`. Because the parser normalizes identifiers and the path is bind-first, a
case mismatch is normally a clean decline. **Residual:** a table with two columns differing only by case (quoted
`"COL"` + `col`) plus `SELECT "COL" … GROUP BY col` passes the case-insensitive gate but resolves the wrong
`group_idx` (`:394`) → wrong grouping — so STATUS's "never wrong rows" is slightly optimistic. **Fix:** exact-match
on normalized identifiers; add the two-same-case-columns test.

### C13 — `ReadyForQuery` transaction-status byte hardcoded to idle 🟡 Medium · Novel
Every `ready_for_query(false)` site (`server/src/lib.rs:195/281/510/…`) passes the constant `false`, so the
backend always reports `'I'` (idle) — never `'T'` (in-transaction) or `'E'` (aborted). A psql/driver relying on
the status byte to detect an open or failed transaction gets wrong information. **Fix:** thread real session
transaction state into `encode_outcome`.

### C14 — CPU-fallback keyed on an error *substring* 🟡 Medium · Novel
The residency-invalidated→CPU fallback is detected by `is_residency_invalidated`, which does
`msg.contains("has no retained resident device memory")` (`lib.rs:248-260`). A string discriminant is fragile: a
future probe error whose message happens to contain that phrase would be silently re-served on CPU, masking a
real device failure. **Fix:** carry a typed `EngineError::ResidencyInvalidated` variant; add a fallback metric.

---

## 4. GPU-native gap — read shapes only the host serves today (the S10d blocker list)

Every shape below declines to `execute_relational_select_cpu_pinned` (or errors); each needs a GPU-native route
before the host relational engine can be deleted. This is the read-side counterpart of the write path's O(table)
residuals.

| # | Shape only host serves today | Anchor | Tracked? |
|---|---|---|---|
| G1 | **All non-resident (freshly committed) tables** — the production default (`auto_admit_on_commit=false`) | `engine_lifecycle.rs:168` | Yes (S-F, STATUS blocking gap) |
| G2 | **Non-int4 types** — enumerated shapes require `Int4`; Int8/Date/Timestamp/Numeric/Uuid/Bool projections + non-int4 scalar aggregates fall to host | `resident_route.rs:106/154/262/399` | Partly |
| G3 | **Text shards / any sharded non-int4** — sharded router + unified recompaction are int4-only | `engine_residency.rs:80-85/4794`; `engine_expr.rs:2562` | Yes (STATUS, PLAN S-D) |
| G4 | **`WHERE col IS NULL/IS NOT NULL` on a sharded-only table** — errors (C8) | `engine_select_exec.rs:410-427` | Yes (HANDOVER B1) |
| G5 | **OFFSET without ORDER BY+LIMIT; GROUP BY/HAVING outside the enumerated grouped shape; multi-column non-equality projection** | `resident_route.rs:89/92/192` | Partly |
| G6 | **Views, materialized views, `pg_catalog`/`information_schema`** — hard-wired to CPU `finalize_relational_select`; matview rows are host-resident | `engine_select_exec.rs:156-214` | No (charter §14 says these should be GPU-resident system relations) |
| G7 | **Multi-key `ORDER BY` on a non-resident table** — errors (host sort is first-key-only, so routing there = wrong rows) | `engine_select_exec.rs:133-139` | Partly |
| G8 | **Scalar `COUNT(DISTINCT v)` without GROUP BY** — host path errors though the GPU path supports it (parity inversion) | `engine_select_bind.rs:643` | No |
| G9 | **Over-VRAM working sets** — the streaming/out-of-core executor (ADR-012) is UNBUILT; recompaction needs the entire surviving set resident on `shards[0].gpu_id`, so an over-budget relation can't be read on GPU at all | `engine_expr.rs:2473/2761` | Yes (ADR-012, hard S10d precondition) |

The **keystone gaps** are G1 (flip auto-admit — the admission producer exists but is default-off) and G9 (the
streaming executor + real cross-shard combine to replace recompact-all-to-one-buffer). G6 (catalog/views on the
GPU substrate) is the least-tracked structural gap.

---

## 5. Performance / scalability findings

### P1 — Every sharded scan re-allocates a unified buffer and DtoD-recompacts all surviving rows 🟠 High · Tracked (ledger #4)
`execute_resident_sharded_via_general` allocates a fresh unified buffer and DtoD-copies each int4 column slice of
every kept shard (+ deleted_by/created_by/null regions), runs the executor once, then frees it — O(kept rows ×
cols) DtoD **per query**, with no caching across identical repeated reads (`engine_expr.rs:2554-2782`;
`retain_device_memory_recompacted:2760`). Zone-map pruning shrinks the kept set but the recompaction of kept rows
is unconditional. This is why "more shards = slower" and sharding never beats single-buffer when it fits. **Fix
(tracked direction):** push-down-to-shard + combine (ARCHITECTURE §13, ADR-012) instead of recompact-to-unified;
or version-cache the unified buffer per residency generation.

### P2 — No cross-shard/multi-GPU combine; recompaction forces the whole surviving table onto one GPU 🟠 High · Tracked (STATUS, ADR-012/S-E)
The unified buffer is allocated on `shards[0].gpu_id` (`engine_expr.rs:2473/2761`), so sharding does **not**
relieve memory pressure — the surviving set must fit one GPU. No per-shard push-down + merge, no cross-GPU partial
combine. **Fix:** the shared combine primitive (scalar reduce / group-merge / k-way ORDER BY merge) that both the
streaming executor and multi-GPU spill require.

### P3 — Single-buffer residency caps ~536M rows 🟠 High · Tracked (ledger #9)
`engine_residency.rs:3141/3233`; the segmented shard layout is the intended relief but is default-OFF, int4-only,
single-GPU. **Fix:** productionize shards (gated on the cross-shard index + the combine of P2).

### P4 — Resident index maintenance is a full O(table) host round-trip on every generation bump 🟠 High · Tracked (ledger #3/#6)
`ensure_shard_pk_device_index` / `build_wave_resident_int4_index` do **DtoH the whole key column → CPU hash build →
HtoD** on any ptr/generation change (`engine_retained_read.rs:701-726/1274-1290`). At billions of rows every
write that bumps a generation pays an O(table) index rebuild — the opposite of the incremental
(insert=add-key/delete=tombstone) maintenance the target requires. **Fix:** incremental per-shard index
maintenance (ledger #6, a hard design gate of the index slice) + a GPU-native build (ledger #8).

### P5 — Point-lookup coalescer is one synchronous thread: host-serial cap + head-of-line blocking + unbounded queue 🟠 High · Tracked (cap) + Novel (queue/HOL)
One OS thread drains one `mpsc` and runs prepare→submit→complete synchronously
(`point_lookup_batcher.rs:184/254-378`). Three coupled issues: (a) all per-item host work is single-threaded and
does not scale with cores/GPU (the documented ~15µs/item, ~156k ops/s cap); (b) `complete` blocks the thread, so
one slow/hung GPU op stalls the whole read path (no submit/complete pipelining); (c) the `mpsc` is **unbounded**
(`:184`) — offered-rate > drain-rate grows the queue and parked-`oneshot` set without backpressure. **Fix (charter
direction):** move coalescing/gather onto the GPU; interim, shard coalescers by table/shape + a bounded ring +
pipelined submit.

### P6 — General SELECT result is materialized on the host three times 🟠 High (charter) · Partly tracked (batched path only)
The engine returns an owned `RelationalSelectResult`; the facade does `result.rows.iter().cloned().map(map_value)`
— a full deep clone of every row (heap-cloning every `Text`) purely to convert `SqlValue`→`DbValue`, two
structurally identical enums (`lib.rs:273-278/392-397`) — and the server then walks the whole set again into
`Vec<Option<String>>` (`server/src/lib.rs:264-270`). Each cell is touched ≥3× on the host. The `.iter().cloned()`
is gratuitous (`into_iter()` would consume with zero clone). The batched path already avoids the `SqlValue`
intermediate (`distribute_results_batched`); the general path does not. **Fix:** consume with `into_iter`; make
`QueryOutcome::Rows.columns` an `Arc` (the engine already `Arc`-shares schema); consider a columnar wire outcome.

### P7 — Route is planned + bound two-to-three times per accepted query 🟡 Medium · Novel
For an accepted resident route, `plan_relational_resident_route` runs in `_instrumented` (`engine_select_exec.rs:215`)
**and again** inside `execute_relational_select_with_resident_route` (`:371`) — each does a catalog snapshot +
table clone + `bind_relational_select` **and records a route decision into telemetry** (so every query is counted
twice). The general-bridge executor then binds a **third** time (`engine_expr.rs:2089`). Pure host per-query
redundancy in the hot path. **Fix:** thread the computed decision + bound tuple from `_instrumented` into the
executor; record telemetry once.

### P8 — No plan cache 🟡 Medium · Novel
`ARCHITECTURE.md §5` promises an "AST tier + physical tier" plan cache invalidated on residency/DDL change; grep
finds **no plan cache** in `engine`/`facade`/`server`. Every query re-parses (`libpg_query`) and re-plans on every
execution. The batcher's per-shape retained template amortizes this for int4 point lookups only; all other SELECTs
pay full parse+plan per call — host-serial work the charter marks "in scope to fix." **Fix:** the promised
AST/physical plan cache keyed on normalized text + residency generation.

### P9 — Async default-on batching parses every query twice and plans every SELECT twice 🟡 Medium · Novel
On the async path, `classify_batchable_point_lookup` parses (`lib.rs:555`) and runs a full route-plan probe
(`:565`) for **every** statement; anything non-batchable (the majority) then re-parses at `:376` and re-plans
inside `execute_relational_select`. **Fix:** return the parsed `Command` + plan decision from classify and thread
them into execute.

### P10 — Per-request coalescer allocations: shape-key `format!` + schema deep-clone per waiter 🟡 Medium · Partly tracked
`point_lookup_shape_key` heap-`format!`s per request and `run_batch` builds a `HashMap<String,_>` with a key clone
(`point_lookup_batcher.rs:428-436/465-476`); `distribute_results_batched` deep-clones the whole `Vec<DbColumn>`
per waiter (`:609`) and re-collects `Vec<Vec<DbValue>>` per needle. This is the residual "result materialization +
oneshot distribution" cost STATUS names. The `PointLookupRequest` doc still references a `route_id` field the
struct no longer has (`:142-151` — stale). **Fix:** thread a small shape enum/hash (no `String`); `Arc` the
schema; fix the stale doc.

### P11 — Redundant `COUNT(*)` precheck on every sharded scalar aggregate (stale premise) 🟡 Medium · Novel
The sharded path runs the executor **twice** for a plain scalar aggregate — a `CountAll` precheck to detect the
empty case, then the real run (`engine_expr.rs:2815-2854`). The precheck's premise ("general SUM/MIN/MAX/AVG
hard-error on empty") is **no longer true** — the general path returns SQL NULL for empty
(`engine_expr.rs:6382-6404`), so the single real run already produces the PG-correct empty result. The precheck is
a removable extra full pass over the unified buffer. **Fix:** delete the precheck.

### P12 — Host CPU read path decodes + re-filters every row twice 🟡 Medium · Tracked (ledger #2, dies with host path)
The CPU path builds the matching key set by full `seq_scan` + `decode_relational_row` per row
(`engine_select_bind.rs:358-462`), then `finalize_relational_select` **re-decodes** each fetched row and
**re-applies the filter** (`:494-516`); disjunction/range/FullScan reads walk `visible_versions` per chain
(`storage/src/lib.rs:246-259`). O(table) host CPU per query — the charter violation the host path embodies.
Equality reads correctly use the pinned value-index. Retires with the host engine; interim, thread decoded rows so
`finalize` doesn't re-decode.

### P13 — Text-entry double-parse on the general GPU path 🟢 Low · Novel
`execute_relational_select_text` parses via the hand-rolled parser, then on the gpu-sortable/Err arms calls
`execute_resident_expr_select_sql(text)` which **re-parses via libpg_query** (`engine_select_exec.rs:98/109`).
**Fix:** pass the already-parsed AST.

### P14 — Read kernels use plain `ld.global` for immutable index/columns (no read-only cache) 🟢 Low · Novel
The immutable per-shard index and resident columns are read with plain `ld.global`
(`execution/src/lib.rs:10423/10883/10920-10949`), not `ld.global.nc` (read-only data cache). For read-only
immutable data this is free throughput left on the table. **Fix:** `ld.global.nc` on the immutable read path.

---

## 6. Robustness / resource findings

### R1 — No GC/VACUUM for GPU-resident `deleted_by`/`created_by` regions → unbounded VRAM growth 🟠 High · Tracked (ledger #5, HANDOVER C)
The host store has version GC (`prune_versions_deleted_at_or_before`, `storage/src/lib.rs:148-187`); the
**GPU-resident** version regions have **none** — they are released only on invalidate/re-admit/drop
(`engine_commit.rs:333-393`). Once resident UPDATE/DELETE is on, tombstones + dead versions accumulate in VRAM
until a full re-admit, and a single long-lived reader pins them (OOM). This is also the event that makes C4's
dense-kernel decline load-bearing. **Fix:** re-clustering compaction that reclaims below the oldest-active
snapshot without an O(table) rewrite + a device-memory-pressure backstop for stuck snapshots.

### R2 — Point-lookup coalescer is unsupervised and outside the engine's poison detection 🟡 Medium · Novel
All batched lookups funnel through one coalescer thread not covered by `is_commit_path_poisoned` /
`is_catalog_latch_poisoned` (those guard only the commit_mutex/catalog_latch). If it panics (an engine-side
`unwrap`/GPU fault inside submit), it is never restarted; every subsequent `enqueue` resolves to "coalescer
unavailable" (`server/src/lib.rs:626-630`) — batching is **permanently, silently dead** process-wide with no
health signal. Degrades to clean errors (not hangs), so not Critical, but a silent permanent feature-kill.
**Fix:** supervise/restart the coalescer; expose a liveness metric.

### R3 — Sync `serve()` (the shipped binary) spawns an unbounded OS thread per connection 🟡 Medium · Partly tracked
`serve_with_engine` does `thread::spawn` per accepted connection with no cap/pool/backpressure
(`server/src/lib.rs:73-86`) — N connections = N threads, and threads are spawned before any auth/workload (an
unauthenticated resource-exhaustion vector). The async `serve_async` bounds *in-flight engine work* via a
semaphore but not connection/thread count, and is not the wired default. **Fix:** bounded connection admission +
worker pool (the aspirational runtime topology, STATUS).

### R4 — No result streaming or row cap — a large SELECT buffers the whole result twice in host RAM 🟡 Medium · Novel
The engine returns all rows, the facade materializes them all, and `encode_outcome` serializes the entire result
into one buffer before a single `write_all` (`server/src/lib.rs:240-284`). No incremental flush, no row cap, no
memory ceiling — `SELECT * FROM big` has unbounded host memory (materialized set + encoded bytes simultaneously)
and pays full latency before the first byte. Latency + OOM/DoS surface. **Fix:** chunked `DataRow` streaming with
a bounded buffer.

### R5 — `deleted_by` sentinel signedness is a latent split-constant trap 🟢 Low · Novel
The "live" fill is `0x7F…` (a large **positive** i64) because the compare is signed s64; `0xFF`/`u64::MAX` would
read as `-1` and wrongly hide live rows (`execution/src/lib.rs:808-816`). No live bug, but the fill constants
(`engine_residency.rs:308/315`) live far from the compare kernel and from `push_conjuncts` — a future switch to
unsigned compare or a `0xFF` fill silently breaks visibility. The sibling KV mask kernel uses a **different**
convention (u64 compare, `u64::MAX` sentinel, `execution/src/lib.rs:23124`), which is exactly the divergence that
makes the trap real. **Fix:** co-locate the sentinel with the compare + a compile-time/test assertion binding
fill-byte ↔ signedness ↔ compare op; unify the KV kernel's convention.

### R6 — Per-GPU byte budget + deterministic eviction are bypassed for shard-resident tables 🟠 High · Partly tracked
`admit_relational_residency_snapshot_inner` (`engine_residency.rs:3332`) builds eviction candidates **only from
`residency.snapshots` (single-buffer)**, but the byte sum `relational_resident_bytes_for_gpu_excluding`
(`:4914`) includes **both** `snapshots` and `shards`, and there is **no post-loop budget re-check**. When
residency is sharded the
eviction loop can free nothing and admission silently succeeds over budget → GPU OOM. **Fix:** include shard
tables in the candidate set, evict by oldest `valid_through` across both maps, hard-error if still over budget.
(Listed under §6 because it is a resource-exhaustion hazard, though it is equally a correctness gate for turning
shard residency on.)

---

## 7. Documentation / baton drift (fix regardless of any code change)

Several read-path claims in the canonical docs are stale and actively mislead (this report's own briefing
inherited three of them). These are cheap to fix and prevent wrong decisions.

- **S1 — The persistent wave read engine is retired, but docs present it as wired.** `WaveReadEngine` and the
  flags `wave_persistent_engine_enabled` / `wave_route_hits` exist in **no** compiled crate (only in
  `docs/future/redis/wave.rs`, an archived reference; DECISIONS 2026-06-29 "LPB OVER THE WAVE ENGINE",
  `-6180 LOC`). Yet `HANDOVER.md:121` lists both as live default-OFF flags, and `engine_commit.rs:394-404` carries
  a stale comment explaining why it "does NOT evict the persistent wave read engine." The real production route is
  the launch-per-batch dense index probe; `wave_engine_enabled` was **renamed `index_probe_enabled`** and is
  **default-ON**. **Fix:** strike the two flags from HANDOVER; rewrite `engine_commit.rs:394-404`; mark the
  ARCHITECTURE/PLAN wave narration as historical.
- **S2 — `index_probe_enabled` is documented default-OFF but is default-ON.** `engine_state.rs:524` comment says
  "behind the default-OFF `index_probe_enabled` flag"; `engine_lifecycle.rs:172` initializes it `true`.
- **S3 — Charter "COUNT(\*) returns Int4" is stale.** The code declares COUNT as `(Int8, oid 20, size 8)`
  (`rel_exec_helpers.rs:1193-1195`) and returns `Int8` on the resident, general, and CPU paths — PG-correct
  (bigint). The charter operational-gotcha should be removed so tests don't expect Int4.
- **S4 — Charter/STATUS "empty aggregate hard-errors (M3-gated)" is stale.** The general path returns PG-`NULL`
  for empty `SUM/MIN/MAX/AVG` (`engine_expr.rs:6382-6404`); the enumerated + sharded paths too.
- **S5 — Stale "NULL-blind == scan" comment.** `engine_expr.rs:2955-2961` asserts the raw-i32 slot gather is
  "byte-identical to the scan on NULLs" — false since the M3 fix made the scan NULL-aware; correctness is preserved
  only by the caller's `if !any_shard_has_nulls` gate at `:2418`, which the comment never mentions. A maintainer
  trusting it could remove the gate and reintroduce the NULL-as-0 bug. **Fix:** rewrite to "route is null-blind →
  caller gates nulls at 2418." (Related: the NULL-as-0 invariant is enforced by convention across three
  duplicated gates — `engine_expr.rs:2418`, `engine_retained_read.rs:1129-1142`, and the null-free-multi-shard
  construction invariant — with no central enforcement; funnel it into one helper.)

The `SqlType`/OID divergence for `SUM(int4)` (declared `SqlType::Int4` but oid 20 / size 8, value `Int8` —
`rel_exec_helpers.rs:1196-1220`) is intentional (PG's `SUM(int4)` is bigint) but fragile: any code that binary-
encodes by `SqlType` rather than OID would emit 4 bytes for an 8-byte value. Worth a typed guard if binary-mode
results are wired.

---

## 8. GPU-native gap summary (current vs. target)

| Concern | Today | Target (charter) |
|---|---|---|
| Read data plane (default deployment) | Host CPU engine (auto-admit OFF, §4 G1) | GPU resident route on every committed table (flip S-F) |
| Resident read snapshot | Pins `committed_seq`, but reads the *published* device generation via un-co-pinned loads (C1/C2) | Device generation stamped with `commit_seq`, co-pinned to the reader's boundary |
| Resident COUNT under MVCC | Visibility-blind full-row count (C3) | Filtered reduction gated on the version synopsis |
| Multi-shard scan | Recompact ALL kept shards into one buffer on one GPU, O(rows)/query (P1/P2) | Push-down-to-shard + cross-shard combine (single-GPU streaming + multi-GPU spill) |
| Over-VRAM reads | Unservable on GPU (host fallback), streaming executor unbuilt (G9) | ADR-012 streaming fold over shards |
| Non-int4 / text / catalog / views | Host only (§4 G2/G3/G6) | GPU-resident columnar + GPU catalog relations |
| Index maintenance | O(table) host rebuild on generation bump (P4) | Incremental per-shard maintenance + GPU-native build |
| Read ingress throughput | Single synchronous coalescer + per-query re-parse/re-plan (P5/P8/P9) | GPU-side coalescing (wave-style) or sharded rings + plan cache |
| Result readback | 3× host re-materialization (P6) | Single device→wire readback (columnar) |

The read **control** plane (parse, route, launch, the single final readback) is legitimately host per the charter.
The distance to the target is (a) the **data-plane coverage gap** (§4 — everything but resident int4 shapes is
still host), (b) the **snapshot-consistency debt** (§2 — the resident route isn't atomically co-pinned to
`committed_seq`), and (c) **host-serial ingress overhead** (§5 P5/P8/P9 — a fix target, not a bet-invalidator).

---

## 9. Prioritized recommendations

| # | Action | Severity | Effort | Status |
|---|---|---|---|---|
| 1 | Co-pin the resident read to `committed_seq`: stamp shard generations with `commit_seq`, capture buffers+regions from one `shards.load()` on the scan path (mirror the point-lookup capture) — fixes C1/C2 and the scan-path region skew | 🔴 Critical | M | Novel/partly-tracked |
| 2 | Gate resident COUNT on the version synopsis (filtered reduction with visibility conjuncts) — fixes C3 | 🟠 High | S | Partly tracked |
| 3 | Land the concurrent-reader double-read differential + tie the dense-kernel version-free precondition to GC — clears C4 before flipping resident-write flags | 🟠 High | M | Tracked (HANDOVER A) |
| 4 | Fall back to per-query (not `fail_group`) on any batcher residency/generation loss; pin one boundary across prepare→submit→complete — fixes C6/C7 | 🟠 High | S | Novel |
| 5 | Fix `GROUP BY <expr>` to evaluate the key at survivor indices (gather-then-evaluate) — fixes C5 | 🟠 High | M | Tracked |
| 6 | Fix shard-residency budget/eviction accounting (candidates from both maps + post-loop re-check) — fixes R6 | 🟠 High | S | Partly tracked |
| 7 | Push-down-to-shard + cross-shard combine to replace recompact-to-unified; version-cache the unified buffer — fixes P1/P2 (and unblocks G9 streaming) | 🟠 High | L | Tracked (ledger #4, ADR-012) |
| 8 | Incremental per-shard index maintenance + GPU-native build — fixes P4 | 🟠 High | L | Tracked (ledger #3/#6/#8) |
| 9 | GC/VACUUM for resident version regions (re-clustering compaction) — fixes R1 | 🟠 High | L | Tracked (ledger #5) |
| 10 | Route IS NULL to the sharded executor; add sharded NULL-exclusion + COUNT(DISTINCT nullable) — fixes C8/C10/C11 | 🟡 Med | M | Tracked (HANDOVER B) |
| 11 | Consume results with `into_iter`; `Arc` the outcome schema; chunked streaming with a row/byte cap — fixes P6/R4 | 🟡 Med | M | Partly tracked |
| 12 | Thread parsed AST + plan decision through classify/execute; add the promised plan cache — fixes P7/P8/P9/P13 | 🟡 Med | M | Novel |
| 13 | Correct the txn-status byte; typed `ResidencyInvalidated` error; supervise the coalescer; bounded connection admission — fixes C13/C14/R2/R3 | 🟡 Med | M | Novel |
| 14 | Documentation sweep — retire the wave-engine flags/comments, fix the stale default-OFF/COUNT-Int4/empty-agg/NULL-blind notes — §7 | 🟡 Med | S | Novel |

**Suggested sequence:** #1–#4 are the correctness gates that must land before flipping the GPU read data plane on
by default (they pair with the write path's `created_by` gate); #5/#6/#10 are cheap correctness wins; #7/#8/#9 are
the scale investments (cross-shard combine, incremental index, GC) that make the shard path viable at billions of
rows and let the host path be deleted for over-VRAM relations; #11/#12/#13 attack the host-serial ingress
overhead the charter marks in-scope-to-fix; #14 stops the docs from mis-steering the next iteration.

---

## Appendix — key file map (read path)

- **Ingress / routing / result materialization:** `crates/facade/src/lib.rs` — `execute_on_shared_engine:375`,
  `execute_on_shared_engine_batched:522`, `classify_batchable_point_lookup:554`, `map_value:672`;
  `crates/server/src/lib.rs` — `serve_with_engine:73`, `encode_outcome:240`.
- **Point-lookup batcher:** `crates/facade/src/point_lookup_batcher.rs` — `coalescer_loop:254`, `run_group:484`
  (fail_group `:509`), `distribute_results_batched:588`.
- **SELECT dispatch:** `crates/engine/src/engine_select_exec.rs` — `execute_relational_select_instrumented:122`,
  `execute_relational_select_cpu_pinned:245`; host bind/finalize `crates/engine/src/engine_select_bind.rs`.
- **Route gate / shape classifiers:** `crates/engine/src/resident_route.rs` (`resident_route_query_shape:21`,
  `sharded_resident_route_query_shape:237`).
- **General GPU executor + sharded recompaction:** `crates/engine/src/engine_expr.rs` —
  `execute_resident_sharded_via_general:2368`, GROUP BY expr `:4980`, empty-agg NULL `:6382`,
  visibility conjuncts `:1985`.
- **Point-lookup / index probe / batched gather:** `crates/engine/src/engine_retained_read.rs` —
  `submit_resident_int4_equal_any_payload:487`, `ensure_shard_pk_device_index:1274`, gather `:1090`+.
- **Resident COUNT / scalar aggregate:** `crates/engine/src/engine_resident_probe.rs` (`run_resident_count:390`).
- **MVCC read visibility (system of record):** `crates/storage/src/lib.rs` (`is_visible:196`);
  `crates/engine/src/mvcc_read_exec.rs` (KV/provenance), `engine_mvcc_dispatch.rs` (host backend).
- **Residency / shards:** `crates/engine/src/engine_residency.rs` (admit `:3332`, byte budget `:4914`),
  `crates/engine/src/resident_storage.rs` (`install_shards:825`), `crates/engine/src/engine_state.rs`.
- **GPU read kernels (hand-written PTX):** `crates/execution/src/lib.rs` — dense/multi-shard index probe
  `:10768`+, binary route `:10793`, deleted_by compare `:808`.
- **Retired (not compiled, historical):** `docs/future/redis/wave.rs`.
