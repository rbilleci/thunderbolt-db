# ARCHIVED — Read/query-path assessment

> Historical point-in-time analysis. It is not an executable plan. Any surviving obligation is tracked only
> in `docs/PLAN.md`.

> **Superseded in part:** the tree moved 19 commits the same day (THE FLIP shipped with audit fixes, WAL
> D1–D4, phase C, retirement A1/A2). See `read-path-reassessment-post-flip.md` for the finding-by-finding
> delta (F1 and §11.2 closed; D1 now LIVE on main; most else unchanged).

**Date:** 2026-07-02
**Method:** static code analysis only (no builds, no benchmarks); every finding anchored to `file:line` in the
**current working tree** — i.e. HEAD `47978626` **plus the uncommitted "THE FLIP" changes** (default-ON
`shard_residency_enabled` / `resident_delete_tombstone_enabled` / `resident_update_tombstone_enabled` /
`shard_index_probe_enabled` / `shard_batched_point_read_enabled`, the zero-copy single-shard path, the
metadata COUNT(*) path, the sharded JOIN bridge, the `sharded_int4_scalar_aggregate` shape, and the moved
sharded-source resolution). Six independent subsystem analyses (ingress/egress, routing, general executor,
point-lookup/index, residency/MVCC/WAL, GPU kernels) were cross-checked against each other and against the
earlier pre-FLIP draft `docs/reviews/read-path-assessment.md`; a disposition table for that draft's findings
is in Appendix B.

**Target state assumed:** a GPU-native relational engine — data plane on the GPU, CPU as control plane only —
whose system of record is the WAL. The host CPU relational engine is scheduled for deletion; the host path is
assessed only for what blocks deleting it.

---

## 1. Executive summary

The GPU read plane itself is in strong shape at the kernel level: **no live wrong-results defect was found in
any read-path PTX kernel** (host/device hash agreement, 716-alignment discipline, duplicate/status protocols,
drop-drain pool safety all verified — §10, §14). The correctness problems live one layer up, in how the host
control plane binds device state to a snapshot, and in what the FLIP changes route where.

**The uncommitted FLIP is not shippable as-is.** It introduces or activates:

1. **One new wrong-results Critical** — NULL-blind scalar aggregates now route to the sharded GPU bridge for
   every nullable pure-int4 table, and the diff pins the two NULL-handling gate tests to single-buffer,
   removing the tests that would catch it (D1).
2. **A wrong-results Critical that was latent** — the batched single-buffer point-read path is NULL-divergent
   from the per-query path (D2).
3. **A snapshot-consistency cluster now live by default** — unstamped INSERT appends visible pre-publish (D3)
   and un-co-pinned descriptor/buffer/region loads that can resurrect tombstoned rows or read wrong offsets
   (D4).
4. **A large availability regression** — the first tombstoned DELETE/UPDATE on a table permanently breaks
   DISTINCT / GROUP BY / ORDER BY / JOIN on it (no VACUUM, no CPU fallback) (E1), and one UPDATE poisons the
   PK index into per-read O(table) scans (F2).
5. **A silent performance regression** — at least 8 previously-GPU shapes (including `COUNT(*) WHERE id = k`)
   now fall to the host CPU scan on default-sharded tables, a class the FLIP's own comment measures at ~1000×
   (F1).

Independently of the FLIP, **the GPU data plane is still unreachable from the shipped product**: the
production binary runs the sync thread-per-connection server with no batcher, `auto_admit_on_commit` defaults
`false` so no table ever becomes resident, writes invalidate residency without re-admitting when the flag is
off, recovery never rebuilds residency, and the facade's strict parser cannot even express the shapes
(JOIN, IS NULL, multi-key ORDER BY) that only the GPU general executor serves (§6). Flipping five engine flags
flipped the engine's internals, not the product.

The distance to the GPU-native target is therefore: (a) close the FLIP correctness/coverage gates (§3–§5),
(b) pull the three real levers — admission, server mode, wire-path parser routing (§6), (c) land the
structural scale work already ledgered — visibility-threaded reshaping paths, VACUUM/re-clustering,
incremental index maintenance, budget-aware shard eviction, streaming/combine (§7–§9), and (d) retire the
host store's three remaining structural roles: recovery source, residency rebuild source, and correctness
backstop (§11).

---

## 2. The read path as actually built (current tree)

Two ingress universes exist and they do not overlap:

| Entry | Who uses it | Parser | Reaches |
|---|---|---|---|
| `execute_on_shared_engine` (`facade/src/lib.rs:375-445`) | **shipped binary** (`server/src/main.rs:20` → sync `serve`, `server/src/lib.rs:66-88`); also the async batched loop's fallback | strict hand-rolled `parse_command` | `engine.execute_relational_select` only; anything the grammar can't express (IS NULL, JOIN, COUNT(DISTINCT), CTEs, expressions) is a **syntax error** |
| `execute_relational_select_text` (`engine_select_exec.rs:84-140`) | examples/tests only | hand-rolled, with Err→libpg_query fallback | the GPU-sortable interception and the **general Expr executor** (GPU hash join, IS NULL, multi-key ORDER BY, expression predicates) |

Engine dispatch (`execute_relational_select_instrumented`, `engine_select_exec.rs`): pin `s = committed_seq()`
→ views/matviews/`pg_catalog` served on host → `plan_relational_resident_route` → accepted: GPU resident
route (single-buffer enumerated probes, or the sharded dispatch list `engine_select_exec.rs:432-449` →
`execute_resident_sharded_via_general`); rejected: `execute_relational_select_cpu_pinned` (host engine).
Mid-statement residency loss falls back to CPU only when the error message contains
`RESIDENT_DEVICE_MEMORY_MISSING` (`engine/src/lib.rs:248-260`).

Under FLIP defaults, admission (`engine_residency.rs:3431`) sends every **purely-int4** table (int4/int2/date
sections, `< 2^29` rows, `engine_residency.rs:3315-3320`) to the **sharded** layout; mixed-type tables keep
the single buffer. Point lookups take the per-shard PK-index routes (3b single-flight, batched gather, or the
multi-shard dense kernel with binary routing); scans take the unified-recompaction bridge, now with a
zero-copy single-shard fast path and a metadata COUNT(*) fast path. The persistent wave read engine is
**retired** (archived under `docs/future/`; no `WaveReadEngine` symbol compiles in any crate) — the
production read plane is launch-per-batch.

But: `auto_admit_on_commit` is **`false`** (`engine_lifecycle.rs:168`), and every admission/incremental-write
site is gated on it (`engine_commit.rs:133,165,261`; `engine_dml_concurrent.rs:326,345`). Out of the box
nothing is ever resident, and even an explicitly warmed table is invalidated by its first write and never
re-admitted. Every measured GPU number in the repo is produced with this flag (or explicit warms) on.

---

## 3. Wrong-results defects

### D1 — FLIP routes NULL-blind scalar aggregates to the GPU on default-sharded tables 🔴 Critical · NEW (uncommitted FLIP)
The new remap `int4_scalar_aggregate` → `sharded_int4_scalar_aggregate` (`engine_residency.rs:5703-5709`;
dispatch `engine_select_exec.rs:433`) sends bare `SUM/AVG/MIN/MAX(col)` through
`execute_resident_sharded_via_general` into the general executor's scalar-aggregate arms — which reduce raw
i32/i64 at survivor indices with **no validity-bitmap consult** (`engine_expr.rs:6593-6741`; only
`CountDistinct` guards nullability at `:6753-6758`). Nullable int4 tables DO shard-admit (`purely_int4` is
type-only, `engine_residency.rs:3315-3320`; bitmaps carried at `:3479`). Failure: `amount = {10, NULL, 30}` →
`MIN(amount)` returns **0** (NULL stored as payload 0), `AVG` divides by 3 not 2, all-NULL `SUM` returns 0
instead of NULL. The single-buffer path is correct because it routes to the null-aware
`scalar_stats` probe kernels — and **the FLIP diff pins the two gate tests for exactly this to single-buffer**
(`tests/resident_probe.rs:1230-1235`, `:1306-1311` now call `set_shard_residency_enabled(false)`), hiding the
regression instead of gating it. The same hole applies to the pre-existing filtered sharded shapes when the
aggregate column is nullable (`sharded_int4_equality_sum`, `sharded_int4_between_avg`,
`sharded_int4_filtered_{avg,min,max}`), now default-reachable.
**Fix:** decline null-bearing aggregate columns in the bridge/route (mirroring the point-route decline at
`engine_expr.rs:2788-2806`) or thread `value_null_off` into the scalar reduce as the grouped path already
does (`engine_expr.rs:5582`); re-point the two pinned tests at the sharded default so they gate it.

### D2 — Batched single-buffer point reads are NULL-divergent from the per-query path 🔴 Critical · confirmed-prior, upgraded
`distribute_results_batched` maps every cell `DbValue::Int4(v)` with no validity channel
(`point_lookup_batcher.rs:605`); the single-buffer submit core (`engine_retained_read.rs:487-537`) never
reads `resident_device_null_columns`, and neither the route shape (`resident_route.rs:144-191`) nor template
prepare declines null-bearing tables — unlike the **sharded** gather, which does
(`engine_retained_read.rs:1129-1142`). Meanwhile the per-query resident path for the same shape is now
NULL-correct (routes through the NULL-aware general bridge, `engine_select_exec.rs:481-487`). Consequence: on
a null-bearing resident table, the **same SQL returns different results depending on whether it coalesced** —
batched `WHERE id = 0` matches NULL rows and projects NULL as `0`. (Exposure today requires the async batched
server; see S2.) **Fix:** decline in `classify_batchable_point_lookup` or template prepare when filter or
projected columns carry a null bitmap — the exact mirror of the sharded decline.

### D3 — Plain INSERT append is published unstamped, before `committed_seq` 🟠 High (SI violation, live flip-gate) · confirmed-prior, now live
Both append paths pass `created_by: None` for plain INSERTs (`engine_commit.rs:138`,
`engine_dml_concurrent.rs:329` → `try_append_resident_int4_open_shard`, contract at
`engine_residency.rs:3885-3890` "unstamped, born-visible"); the shard append bumps `row_count`
(`engine_residency.rs:4103-4118`) **before** `publish_committed_seq` (`engine_commit.rs:161`). A reader
pinned at `s = C-1` that loads `shards` after the bump sees commit C's rows — on the scan, on the FLIP
metadata COUNT (`engine_expr.rs:2814-2838` sums `row_count`; version-free tables are exactly the ones with
unstamped appends), and on the zero-copy path. Repeat reads within one pinned statement can see a phantom
appear. SV6 fixed precisely this class for UPDATE (stamp at `:4085-4091`); INSERT was consciously exempted
(the code documents it as a "milder as-if-later read", `engine_expr.rs:2324-2325`) — but the project's own
SV5-P2 standard treated "C-1 reader sees C's write" as a do-not-flip gate, and this contradicts it.
**Fix:** stamp `created_by = commit_seq` on ALL appends — the region machinery and the read gates
(`created_by <= s`) already exist on every gated route; the cost is the on-demand region alloc on first
insert-append. (Alternative — publish `row_count` after `committed_seq` — requires read-side ordering too.)

### D4 — Un-co-pinned (descriptor, buffer, version-region) loads: tombstone resurrection / wrong offsets / OOB 🟠 High · confirmed-prior + extended, now default-exposed
Three independent analyses converged on this cluster. The scan and its FLIP fast paths capture device state
in **separate lock-free loads**: descriptors from one `shards.load()` (`engine_expr.rs:2375-2387`), buffers
from later per-shard `shard_device_memory.get()`s (`:2460-2470`), regions from separate
`shard_deleted_by/created_by_memory.get()`s (zero-copy check `:2486-2508`, metadata COUNT `:2815-2825`,
recompaction gather `:2588-2628`). Writers publish buffers **before** descriptors
(`resident_storage.rs:832-835`), and a re-admit purges the region maps then reinstalls **the same
`shard_id 0`** (`engine_residency.rs:3444-3452`). Two concrete wrong-results interleavings:
- **Tombstone resurrection:** reader loads the OLD (tombstone-bearing) shard list; a re-admit purges the
  regions; the reader's version-free check then passes → zero-copy serves the old buffer raw / metadata
  COUNT sums stale `row_count` → a row deleted at `D <= s` resurfaces.
- **Generation mismatch:** OLD cloned descriptor (its `is_valid()` flag predates the invalidate) paired with
  the NEW buffer → recompaction segments computed from stale `row_count`/`capacity` read the wrong
  generation's bytes (silent wrong rows if the new buffer is larger; CUDA illegal-address if smaller — the
  DtoD primitive bounds-checks the destination only, `execution/src/lib.rs:3891-3899`).
The same two-load residual exists inside the point-lookup locate/gather (`engine_retained_read.rs:792-799`,
`1043-1050`, `1387-1394`) — the audited 3b `ShardPkHit` fix made locate→materialize consistent
(`engine_expr.rs:3046-3056`) but the descriptor→buffer pairing inside locate is still two snapshots, and
`CudaDeviceMemoryProof` carries no `device_ptr` to cross-check (`execution/src/lib.rs:87-93`).
**Fix (one pattern, applied everywhere):** capture (descriptor, pinned buffer, regions) in ONE
generation-consistent snapshot — embed the buffer/region `Arc`s (or a `device_ptr` + generation stamp) in
`RelationalResidentShard` so a single `shards.load()` yields the whole tuple, and verify
`descriptor.device_ptr == buffer.device_ptr()` in `source_for`, declining on mismatch. The 3b hit-capture is
the proven in-repo template.

### D5 — `int4_range_count` is NULL-blind (single-buffer / mixed-type tables) 🟠 High · NEW
`ResidentPredicate::Int4Equal` passes `null_bitmap_offset` but `Int4Compare` calls
`count_i32_compare_from_payload` with **no null channel** (`engine_resident_probe.rs:440-464`; wrapper
`execution/src/lib.rs:1571-1579`). NULL int4 is stored as payload 0, so `COUNT(*) WHERE col < 5` counts NULL
rows. Reachable in the default config: `resident_route_count_shape` emits the shape with no nullability gate
(`resident_route.rs:447-451`), and mixed-type tables stay single-buffer-resident under the FLIP
(`engine_residency.rs:3425-3431`). (Purely-int4 tables escape only because the sharded planner rejects the
shape — which is itself regression F1.) **Fix:** pass the bitmap like the equality count, or decline nullable
columns to the general 3VL path. The test/example-only BETWEEN/membership count probes
(`engine_resident_probe.rs:818-1084`) share the blindness — deletion-sweep candidates.

### D6 — `GROUP BY <expr>` evaluates the key over ALL rows before WHERE 🟡 Medium · confirmed-prior
`arith_value_column_device(&program, row_count, …)` materializes the key expression checked over the full
table (`engine_expr.rs:5162-5188`); a WHERE-excluded row's overflow errors where PG succeeds
(error-strictness, not wrong values — the grouping kernel reads only survivors, `:5561-5594`). ORDER BY is
already fixed (evaluates at survivor indices, `:6795-6802`). **Fix:** mirror ORDER BY — gather survivors,
evaluate at indices.

### D7 — Small confirmed corners 🟡 Low
- `SELECT COUNT(*) FROM t LIMIT 0` returns 1 row — `resident_route_count_shape` never checks `select.limit`
  (`resident_route.rs:423-439`); the metadata fast path inherits it (pre-existing; device path equally wrong).
- `COUNT(DISTINCT nullable)` / grouped nullable-key/value + COUNT(DISTINCT) clean-error `ApplyFailed`
  ("M3 3VL follow-up") instead of computing the PG answer (`engine_expr.rs:6753-6758`, `:5445-5456`,
  `:5464-5475`). Never wrong counts.
- Prior-draft C12 (case-insensitive route gate → wrong `group_idx`) is **refuted**: bind runs first in the
  planner and enforces exact `relational_column_index` equality for every grouped arm
  (`rel_exec_helpers.rs:1390-1526`) — a case-mismatch fails bind → clean decline. Hygiene only.

---

## 4. Availability / spurious-error defects

### E1 — First tombstoned DELETE/UPDATE permanently breaks DISTINCT / GROUP BY / ORDER BY / JOIN on the table 🟠 High · NEW (FLIP-exposed)
With the tombstone flags default-ON, one single-row DELETE/UPDATE installs a per-shard version region that
**nothing ever removes** (regions release only on invalidate/re-admit/drop, `engine_commit.rs:333-351`;
VACUUM is unbuilt). Every subsequent reshaping read then hard-errors: the sharded bridge guard
(`engine_expr.rs:2902-2910`), the new general-path guard (`:4433-4447`), and the new JOIN arm
(`:3310-3315`) all return `ApplyFailed("resident visibility filter (SV3b/SV6) is not yet wired…")`. The
planner accepts these shapes without checking version regions (`engine_residency.rs:5807-5830`), and the
errors do not match the CPU-fallback substring (E2), so they surface to the client. Net: under default
flags, `DELETE FROM t WHERE id=1` breaks `SELECT x FROM t ORDER BY x` **indefinitely** (until an unrelated
invalidating commit). Pre-FLIP, DELETE/UPDATE re-admitted all-live shards and these shapes worked.
**Fix (ordered):** (1) immediate — planner-level decline of versioned tables for un-wired shapes → CPU
pinned path (correct, slower); (2) thread the visibility conjuncts through the sort/group/distinct/join
paths (the VM conjuncts already exist for scans, `engine_expr.rs:1992-2003`); (3) VACUUM/re-clustering so
regions are not permanent (§9/R-2).

### E2 — Sharded error strings bypass the CPU fallback → client errors on routine concurrent invalidation 🟠 High · NEW
The transparent mid-statement fallback matches only `"has no retained resident device memory"`
(`engine/src/lib.rs:248-260`; used at `engine_select_exec.rs:252-254`). The sharded paths emit
`"resident shard {} has no retained device memory"` (`engine_expr.rs:2466-2469`) and
`"resident shard {} is invalid"` (`:2455-2458`) — neither matches. A concurrent commit that invalidates a
sharded table between plan-accept and `source_for` (the invalidate+re-admit fallback fires on every
multi-row DML, NULL insert, or ambiguous locate — routine) produces a **client-visible hard error** where a
single-buffer table degrades transparently. Related fragility: the discriminant is a substring re-`format!`ed
at ~12 sites rather than a typed variant (`engine_retained_read.rs:531,1574`;
`engine_resident_probe.rs:406,…`; `engine_expr.rs:3345,4500`). **Fix:** a typed
`EngineError::ResidencyInvalidated` (the typed `ExecuteError::Serialization` precedent exists,
`facade/src/lib.rs:634-649`); match shard strings in the interim.

### E3 — Batcher turns benign residency/generation races into hard errors for every waiter 🟠 High · confirmed-prior
Two arms of the same defect (`point_lookup_batcher.rs`):
- Residency lost between classify and prepare: the else-arm calls `fail_group` when
  `resident_shard_count == 0` (`:499-512`) — every coalesced waiter gets `ApplyFailed`; the per-query path
  would have served from CPU. Violates the documented "misclassification only ever costs a slow path"
  invariant (`facade/src/lib.rs:514-517`).
- Commit in the prepare→submit window: submit hard-errors on `snapshot_generation` mismatch
  (`engine_retained_read.rs:420-425`) and `run_group` routes any submit error to `fail_group` (`:530-535`).
  The mismatch message also doesn't match the fallback discriminant, so nothing downstream saves it.
**Fix:** on ANY prepare/submit/complete failure, fall back to `run_group_per_query` (the shard-decline arm
already does, `:507`); optionally retry-prepare once on generation mismatch.

### E4 — Coalescer panic permanently kills all batchable reads; unsupervised 🟠 High · confirmed-prior
One thread, no `catch_unwind`, no respawn (`point_lookup_batcher.rs:184-188`, join-on-Drop `:238-248`). After
a panic, `enqueue`'s `tx.send` fails silently (`let _ =`, `:230-231`) and every batchable lookup resolves to
"coalescer unavailable" (`server/src/lib.rs:622-631`) **forever** — classify keeps routing to the dead
batcher; there is no fallback-on-RecvError to the per-query path. **Fix:** re-execute via
`execute_on_shared_engine` on oneshot `RecvError`; supervise/respawn; export a liveness metric.

### E5 — Multi-key ORDER BY is a hard error on the production wire path 🟡 Medium · updated-prior
`engine_select_exec.rs:155-161` rejects >1 sort key for every select reaching `_instrumented` (guarding the
first-key-only host sort — correct, never wrong rows). Via the shipped facade entry the GPU-sortable
interception never runs (it lives only on the text entry, `:116-121`), so `ORDER BY a, b` errors even on a
fully resident table whose general executor could sort it. Subsumed by S3.

---

## 5. FLIP regressions (coverage & performance)

### F1 — ≥8 previously-GPU shapes silently fall to the host CPU scan on default-sharded tables 🟠 High · NEW
The sharded remap (`engine_residency.rs:5688-5734`) + accept-list (`:5807-5830`) omit:
`int4_equality_count` (`COUNT(*) WHERE id=5` — the canonical OLTP shape), `int4_range_count`,
`int4_filter_group_count` (AND/OR/IN/BETWEEN counts), `int4_filtered_scalar_aggregate` **for SUM** (only
Avg/Min/Max are mapped, `:5711-5722`), `int4_between_scalar_aggregate` (all four aggregates),
`int4_projection` (range projections), `int4_composite_equality_multi_column_projection`,
`int4_equality_mixed_column_projection`, `text_prefix_like_count`. Each falls through
`else { query_shape }` (`:5732-5733`), fails the accept-list, → `execute_relational_select_cpu_pinned`
(`engine_select_exec.rs:258`) — the O(table) host scan. The FLIP's own comment quantifies the class:
**496,554µs vs 460µs p50 (~1000×) at 524k rows** (`engine_residency.rs:5703-5708`). The general executor
already serves all of these over the unified source; they need remap entries + `required_int4_columns`
branches. **Fix:** add the missing remaps (mostly one-liners) + a differential test that walks EVERY
single-buffer shape against a sharded table and asserts route parity.

### F2 — One UPDATE poisons the PK index: within-shard dup key → cached decline → per-read O(table) scans 🟠 High · NEW
`try_update_resident_commit` appends the new version into the open shard where the tombstoned old version
usually also lives → duplicate key in the key column → `build_int4_pk_hash_table_host` declines
(`engine_retained_read.rs:1978-1980`) and the decline is **cached** (`index: None`, `:900`, `:1293`) →
`locate_sharded_pk_batch` returns `None` for whole batches (`:1062-1063`), the 3b route declines (`:837`),
the GPU dense route already declined (versioned shard) → every point read pays the versioned recompaction
scan; the zero-copy path is also disqualified (versioned). Net: the first UPDATE demotes the table from
~22µs index reads to per-read O(table) DtoD until a fallback re-admit happens to fire. **Fix:** build the
index over live slots only (skip slots whose `deleted_by` is stamped — needs the region at build time), or
map key→newest-slot with a probe-time `deleted_by` gate, or trigger re-clustering/re-admit on dup-decline.

### F3 — Version-region permanence degrades the FLIP's own fast paths permanently 🟠 High · confirmed-prior, upgraded
Both new fast paths gate on `version_free` (`engine_expr.rs:2486-2508`, `:2815-2825`). With tombstones
default-ON and no VACUUM, the FIRST delete/update permanently drops a table from the ~19µs metadata COUNT /
zero-copy scan to the ~430-530µs full recompaction — a silent, persistent perf cliff the FLIP itself created.
Churn workloads also grow shards O(total updates) (each update appends a version → fills headroom → rolls a
new shard) with no reclamation. **Fix:** the ledgered VACUUM/re-clustering compaction, now urgent (see R-2).

### F4 — Index maintenance is a full O(shard) rebuild after every in-place append — twice 🟠 High · confirmed-prior, upgraded
`ensure_shard_pk_device_index` rebuilds on any `(ptr,row_count)` change: full key-column DtoH
(`engine_retained_read.rs:1274`) → host hash build (`:1279`) → HtoD (`:1287`); the **host** cache rebuilds
separately with a second full DtoH (`:878-916`, `:961-1006`). Every single-row INSERT/UPDATE commit bumps the
open shard → the next point-read batch pays ~16MB DtoH + O(n) host build + ~64MB HtoD at the 4M shard target,
on the hot path. Sealed shards never rebuild (the immutable design works); the OPEN shard is the problem.
With writes default-incremental this is now the dominant unscalable read-path cost (ledger #3/#6).
**Fix:** incremental maintenance for the open shard (append keys into a device-side index; the sealed/open
split makes this tractable), GPU-native build later (ledger #8).

### F5 — JOIN bridge: snapshot re-read, un-pruned recompaction, vacuous assert 🟡 Medium · NEW (uncommitted FLIP)
`resolve_join_side` builds the unified source at a fresh `self.committed_seq()` (`engine_expr.rs:3306-3309`)
instead of the statement's pinned `s` — inert today (versioned relations clean-error) but it seeds
cross-relation snapshot skew the day the join threads visibility; thread the caller's `copin_s` now. Each
sharded join side pays a full **all-shards** recompaction (predicate `None` passed even when the per-side
predicate could zone-map-prune). `from_dense_host_rows(descriptor, Vec::new())` is safe (join gather is fully
device-side) but makes the `JOIN_NULL_ROW` distinguishability `debug_assert` vacuous for sharded sides
(`:3665-3670` checks `host_row_count()==0`; should check device `row_count`).

### F6 — Versioned+reshaping clean-error pays the full recompaction first 🟢 Low · NEW
The resolution block builds the unified source (full DtoD + region gather) at `engine_expr.rs:4432`, then
throws it away at `:4433-4447`. The clause checks are pure metadata — hoist them before the build.

### F7 — Tables ≥2^29 rows cannot shard at admission — the cap sharding exists to remove 🟡 Medium · NEW
`purely_int4` requires `row_count < (1 << 29)` (`engine_residency.rs:3315-3320`) and the sharded branch
requires `purely_int4` (`:3431`). A ≥536M-row table admits via the single dense buffer (or fails allocation →
host path); multi-shard layout is reachable only by incremental rollover growth. After restart + warm of a
large table this recreates the exact single-buffer cap (and per-shard indexes cap at 2^29 rows/shard by the
hash-slot packing — safe, but a shard above it silently loses its index). **Fix:** chunked admission that
lays a large table down as N sealed shards.

### F8 — Metadata COUNT specifics 🟡 Medium · NEW
Race-safety of the version-free gate is sound (region insert precedes `publish_committed_seq`), and the
check+sum use one `shards_guard`. But it inherits D3's premature INSERT visibility (sums `row_count`
as-of-now against a `copin_s` reader) — fix together with D3 — and it hard-codes a second copy of that
anomaly that must not be forgotten when D3 lands.

---

## 6. Structural: the GPU plane is not reachable from the shipped product

### S1 — `auto_admit_on_commit=false` keeps the whole data plane dormant; writes kill residency when it's off 🟠 High (keystone) · confirmed
`engine_lifecycle.rs:168`. Admission never fires (`engine_commit.rs:133,165,261`;
`engine_dml_concurrent.rs:326,345`); worse, a **warmed** table is invalidated by its first write
(`invalidate_relational_residency_for_commit`) and never re-admitted because re-admission is also behind the
flag. The five FLIP flags are subordinate to this master switch. This is the actual unpulled lever (S-F).

### S2 — The shipped binary never engages the batcher or the async server 🟠 High · NEW
`server/src/main.rs:20` → sync `serve` → thread-per-connection (`server/src/lib.rs:66-88`), per-query
dispatch only. `PointLookupBatcher` exists only on `serve_async_with_engine_batching` (`:432-457`), which no
shipped binary constructs. The entire batched read arc (the 121.6M lookups/s lever, and D2's exposure) is
examples/tests-only. Also: the sync path spawns unbounded pre-auth threads, `complete_startup` grants
`authentication_ok` unconditionally (`:180-197`), and each pre-auth connection can pin up to the 64MiB frame
cap (`:58`, `:341-344`) — resource exhaustion surface in the default mode.

### S3 — The general GPU executor is unreachable from the wire 🟠 High (CPU-deletion blocker) · NEW
The facade parses with the strict hand-rolled grammar and errors on anything it can't express
(`facade/src/lib.rs:376-381`); there is no route to `execute_resident_expr_select_sql`. Everything ONLY the
general executor serves — IS NULL/IS NOT NULL (no `IsNull` in the grammar), the **GPU hash JOIN**
(`engine_sql_pg.rs:41-118`), multi-key/expression ORDER BY, arithmetic predicates, scalar COUNT(DISTINCT),
NULLS FIRST/LAST — is wire-unreachable: clients get syntax errors while the GPU capability sits tested but
unused. Conversely the prior draft's C8 (sharded IS NULL errors) is **FIXED on the text entry** by the
SLICE B/FLIP resolution (`engine_expr.rs:4422-4451`), which makes the wire gap the only remaining blocker for
those shapes. **Fix:** route the facade's parse-Err arm through the text-entry logic (parse-Err →
libpg_query → general executor), exactly as `execute_relational_select_text:131` does.

### S4 — Recovery never rebuilds residency; first post-restart reads are 100% host 🟡 Medium · NEW
All `recover_from_*` entries (`engine_lifecycle.rs:16-60`) replay through `commit_mutation` with admission
gated on the default-false flag — post-restart the engine has ZERO GPU residency until an operator warms.
(With auto-admit ON during replay, every replayed commit would pay residency work — also wrong; replay wants
admit-once-at-end.) No read can see pre-replay state (recovery is constructor-scoped). **Fix:** bulk
admission pass at recovery end, gated on the same budget machinery.

---

## 7. Other snapshot/MVCC notes

- **Read-boundary design concession confirmed** (`engine_select_bind.rs:41-47`): the resident route
  deliberately does not pin the device generation to `s`. With SV6/SV3b stamps the *stamped* mutation classes
  reconstruct the pinned snapshot correctly on every gated route (verified: 3b per-hit, batched gather region
  gather + signed compares, scan VM conjuncts, dense-route decline). The residual anomalies are exactly D3
  (unstamped INSERT) and re-admit (rebuilds all-live at the newest boundary → as-if-later read). D4 is the
  mechanism that makes even the stamped classes breakable.
- **Batched gather + JOIN bind their own `committed_seq()`** instead of the statement pin
  (`engine_retained_read.rs:1153` — the passed `_copin_s` is unused; `engine_expr.rs:3306-3309`): internally
  consistent per batch, but breaks the catalog↔data co-pinning discipline. Thread `copin_s` through both.
- **`run_resident_count` visibility-blindness downgraded to Low** (prior C3): a versioned single-buffer table
  cannot exist today — version regions are allocated only via shard tombstone paths
  (`engine_residency.rs:4270-4306`), and non-shard DML declines to invalidate+re-admit. Sharded COUNT routes
  to the (version-gated) metadata path or bridge. The invariant is implicit and cross-file: add a debug
  assert (no shard regions for the table) in `run_resident_count` (`engine_resident_probe.rs:412-464`).
- **The dense multi-shard kernel's version-free decline is now load-bearing and HOLDS** (prior C4, rerated
  OK): the kernel applies no gate, but the host decline (`engine_retained_read.rs:1364-1386`) is enforced on
  the kernel's only call path and is race-safe (region insert before publish). Keep the "do NOT remove under
  independent region reclaim" comment — VACUUM (R-2) must not break this.
- **Null-decline TOCTOU** 🟢 Low: the batched/3b null-bearing declines run on their own `shards.load()`
  (`engine_retained_read.rs:1129-1141`, `engine_expr.rs:2788-2806`); a re-admit introducing NULL bitmaps
  between loads is served NULL-blind. The comment at `engine_expr.rs:2786-2787` claims internal re-validation
  that does not exist. Cheap fix: decline inside locate/gather on the captured descriptor's
  `resident_device_null_columns`.

---

## 8. Performance findings (host path)

| # | Finding | Anchor | Severity |
|---|---|---|---|
| P-1 | Per-query recompaction remains for every multi-shard survivor set, every versioned table, and **every JOIN side**; zero-copy covers only the 1-shard version-free case; nothing is cached across identical reads (ledger #4) | `engine_expr.rs:2717-2724`, `:2486-2509` | 🟠 High |
| P-2 | Unified buffer allocated on `shards[0].gpu_id` — the surviving set must fit ONE GPU; no cross-shard/cross-GPU combine (ADR-012 unbuilt) | `engine_expr.rs:2440,2718` | 🟠 High |
| P-3 | Redundant COUNT(*) precheck runs 2 full executor passes per bare/filtered sharded aggregate — premise ("general path errors on empty") is stale, the general path returns NULL-on-empty itself; the new scalar shape routes every bare aggregate through it; each pass also builds `(0..n)` u32 + u64 index Vecs on host (~12B/row ×2) | `engine_expr.rs:2921-2956`, `:6564-6586`, `:4550-4561` | 🟡 Medium |
| P-4 | Route planned+bound 2× per accepted query (plan at `engine_select_exec.rs:237` AND `:393`), telemetry double-counted (`engine_residency.rs:5524-5528`), executor re-binds (3rd), sharded scalar path binds a synthesized COUNT select (4th) | see anchors | 🟡 Medium |
| P-5 | No plan cache anywhere; every statement re-parses + re-plans; batcher templates rebuilt per group per flush | `facade/src/lib.rs:376,555` | 🟡 Medium |
| P-6 | Async classify parses+plans every statement, then the fallback re-parses and re-plans (double); batchable arm binds job + template (triple) | `facade/src/lib.rs:555,565,376`; `engine_retained_read.rs:38,355` | 🟡 Medium |
| P-7 | Result egress materializes the full set 3×: engine `RowBlock` → facade re-nests `Vec<Vec<DbValue>>` with per-cell `.cloned()` (gratuitous — result is owned) → server re-walks into `Vec<Option<String>>` → single wire buffer; no streaming, no row cap → large SELECT = host OOM + full-latency-before-first-byte | `facade/src/lib.rs:392-397`; `server/src/lib.rs:240-289` | 🟠 High (charter) |
| P-8 | Coalescer: one synchronous thread, unbounded mpsc, strict-serial groups, and the dup-decline fallback runs N full per-query selects ON the coalescer thread (head-of-line stall) | `point_lookup_batcher.rs:184-188,438-441,545-561` | 🟠 High (when batcher is wired) |
| P-9 | Per-request `format!` shape keys + per-waiter schema deep-clone; stale `route_id` doc | `point_lookup_batcher.rs:454-477,609-610,143-146` | 🟢 Low |
| P-10 | Warm/maintenance treats healthy shard-resident tables as non-resident → full O(table) re-admit every cycle | `engine_residency.rs:5309-5314` | 🟢 Low |
| P-11 | Sharded plane loses kernel telemetry: `kernel_event_elapsed_us` read only from the single-buffer cell (absent for shards) → always `None` | `engine_select_exec.rs:507-512` | 🟢 Low |
| P-12 | Host locate hot-path allocations: `(table.name.clone(), shard_id)` String keys per shard per batch across 3-4 maps | `engine_retained_read.rs` locate paths | 🟢 Low |
| P-13 | Text-entry double parse (hand-rolled then libpg_query) | `engine_select_exec.rs:84,120,131` | 🟢 Low |

---

## 9. Robustness / resource findings

### R-1 — Budget & eviction are blind to the now-default shard layout 🟠 High · confirmed-prior, upgraded
Eviction candidates come only from single-buffer `snapshots` (`engine_residency.rs:3593-3608`) while byte
sums include shard bytes (`:5106-5128`); there is **no post-loop re-check** — when candidates run out the
decision still records `accepted: true`, "admitted within budget" (`:3622-3639`). With every purely-int4
table now shard-admitted, steady state = nothing evictable, admissions always "succeed", VRAM silently blown
with a misleading audit trail. Compounding: rollover allocates whole new shard buffers with **no budget check
at all** (`:4126-4211`); version regions, device PK indexes, and per-read unified recompaction buffers are
unaccounted; `resident_bytes` is a host-value estimate that ignores ~2× capacity padding. **Fix:** shard
tables in the candidate set, evict oldest `valid_through` across both maps, hard-fail (or decline to host) if
still over budget, account regions/indexes/recompaction leases, budget-check rollover.

### R-2 — No GC/VACUUM for device version regions 🟠 High · confirmed-prior, upgraded
Regions and dead versions persist until invalidate/re-admit/drop (`engine_commit.rs:333-351,380-392`).
Default-ON incremental UPDATE turns churn into unbounded shard growth + permanent fast-path loss (F3) +
permanent reshaping breakage (E1) + eventual VRAM exhaustion, with a long-lived pinned reader blocking any
reclaim. The dense-route decline (§7) must be preserved by whatever reclaim design lands. This is now the
single most consequential missing subsystem for the read path's steady state.

### R-3 — `wave_index` cache is never purged at retire sites — pinned-VRAM leak 🟡 Medium · NEW
Entries pin the resident buffer + device index (`resident_storage.rs:24-32`) but only the in-place append
(`engine_residency.rs:3994-3999`) and probe-time rebuild remove them. `invalidate_relational_residency_table`
(`engine_commit.rs:297-352`), budget eviction (`resident_storage.rs:778-801`), and `apply_drop_table`
(`engine_ddl_table.rs:1099-1188`) purge the shard PK caches but NOT `wave_index` — a dropped/evicted
single-buffer table leaks its pinned buffer + index until a same-name table's probe replaces it. Pure leak,
never wrong results. **Fix:** add `wave_index.remove(table)` at the three sites (mirror
`purge_shard_pk_index_for_table`, which IS complete at all 7 sites — verified).

### R-4 — Worst-case scan output lease is O(rows × projections) — above the pool cap it cuMemAllocs per query 🟡 Medium · NEW
`submit_cuda_resident_i32_equal_any_project` leases `row_count × projection_count × 4` bytes because the
atomic emit is unbounded (`execution/src/lib.rs:9708-9716`). At the 536M-row cap ×4 projections = 8.6GB —
over the 1GiB pool cap (`:143`), so every call pays driver alloc/free (or OOMs). Masked today by the index
routes. **Fix:** two-pass count→exact-size (the ordered-compaction pattern already in-crate) or capped
emit + retry.

### R-5 — Flag/lifecycle notes 🟢 Low
Runtime `shard_residency_enabled` ON→OFF is inert until the next admission (routing keys off the shards map);
both flip directions clear stale state at admission (verified) — document that the flag is admission-time
policy. Module cache is keyed by entry name only (`execution/src/lib.rs:252-276`) — key by (name, ptx-hash).

---

## 10. Kernel-level findings (crates/execution)

**No live wrong-results defect found.** Verified sound: host/device hash agreement exact (constant
0x9E3779B1, shift-then-mask order, 256-probe cap made unreachable-as-miss by builder declines); all resident
8/16-byte data loaded as 4-byte words (no misaligned-716 exposure); binary-routing prologue traced correct
including BKEEP0/gap fallbacks; dup status=3 protocol (linear) + host disjointness proof (binary); memset-0
status gap-guard + covering sync + Drop-that-drains-before-pooling (the cross-thread UAF window is closed);
divergence-safe `bar.sync` trees; grid-stride math safe for u64 counts.

**Correctness-adjacent (all bounded by the current 536M/2^29 caps — none safe to forget if caps lift):**

| # | Finding | Anchor | Severity |
|---|---|---|---|
| K-1 | Ordered-compaction INDEX mode truncates row index to u32 guarded only by a `debug_assert` — compiled out in release; >2^32-row buffer would silently emit wrapped indices | `execution/src/lib.rs:19799-19802`, guard `:20775-20778` | 🟡 Medium (make it a hard `Err`) |
| K-2 | ORDER BY COUNT sort key truncates u64 count to u32 (`cvt.u32.u64`) — ≥2^32-row groups mis-sort | `pack.ptx:67` | 🟢 Low |
| K-3 | Scalar SUM reduction has no i64-overflow detection (PG raises `bigint out of range`); the VM's checked-overflow machinery exists but the reductions don't use it | `execution/src/lib.rs:13361-13363,13439-13441` | 🟢 Low→Med at billions |
| K-4 | Recompaction DtoD validates destination bounds, never source | `execution/src/lib.rs:3891-3907` | 🟢 Low (also the D4 backstop) |
| K-5 | deleted_by sentinel conventions split: resident = signed s64 / 0x7F-fill vs KV kernel = unsigned u64 / u64::MAX — each internally consistent, cross-feeding hides all live rows; plus one wrong doc (`engine_state.rs:508` says "u64::MAX" for the region fill — it's the 0x7F fill) | `expr_proto.ptx:1237-1242`; `execution/src/lib.rs:23124-23125,808-816` | 🟢 Low (shared constant + build-time assert) |
| K-6 | KV mask-kernel family does dlopen+cuInit+cuCtxCreate+JIT **per call** — on the host tuple-store path slated for deletion; do not optimize, put on the deletion sweep | `execution/src/lib.rs:23143-23219,21347-22783` | 🟡 (deletion item) |

**Kernel-side visibility truth** (host routing is the only line of defense for the probe routes):

| Kernel family | Visibility gate in-kernel? | Protected by |
|---|---|---|
| expr VM (scan route) | **YES** — i64 conjuncts `created_by <= s AND deleted_by > s`, signed | — |
| dense probe / multi-shard probe / binary route | NO | host version-free decline (verified enforced, §7) |
| equal/compare counts, scalar stats, gathers, group-by, sorts, DISTINCT, joins | NO (3VL bitmap only, where wired) | host routing declines / versioned-error guards |
| KV batch mask (host-store path) | YES — unsigned u64 convention | — |

**Optimization headroom (honest ratings for the current launch-per-batch design):**

| # | Opportunity | Value |
|---|---|---|
| O-1 | Sharded recompaction = S×C serialized blocking `cuMemcpyDtoD` + blocking memset fills (`execution/src/lib.rs:3858-3911`); replace with one descriptor-driven device copy kernel (the multi-shard-probe pattern applied to copies) or batched async DtoD + one covering sync. **Directly attacks the 1-shard recompaction tax gating the flip** | High |
| O-2 | Mid-query host round trip in ordered compaction (kernel → DtoH counts → host scan → HtoD bases → kernel, `:20927-20993`); device scan / decoupled-lookback removes a full sync per general predicate | Medium |
| O-3 | Radix ORDER BY ≈ 49 launches (16 passes × 3, 4-bit digits, `:18001-18763`); 8-bit digits halve it; onesweep better | Medium |
| O-4 | Dense-probe completion does two sync **pageable** DtoHs (`:11276-11295`) while the atomic path already stages via pinned+async; pinned staging both directions; relevant to the known ~4-thread concurrency crossover | Medium |
| O-5 | No `ld.global.nc` anywhere (0 hits, both PTX corpora) despite all resident data being immutable per generation; blocked in the 21 modules still targeting `.target sm_30` — uplift to sm_90 (the product already requires sm_90 elsewhere) in one sweep | Medium |
| O-6 | Multi-shard kernel re-loads the 64B shard descriptor from global per (needle × shard) (`:10855-10863`); stage per block in shared / `.nc`; warp-per-needle split is the next lever if shard counts grow | Medium |
| O-7 | Shared-memory over-allocation: `scalar_stats` declares 24KB static smem for 256-thread blocks (arrays sized 1024, `:13207-13210`) — 4× oversized, caps SM residency | Low-Med |
| O-8 | No CUDA graphs (0 `cuGraph` hits); the fixed memset→HtoD→kernel→DtoH sequence is graph-shaped; visible at small-batch single-flight latencies | Low-Med |
| O-9 | No vectorized loads; only material for the recompaction-replacement kernel and radix passes, not the probes | Low |

---

## 11. WAL ↔ read path (target: WAL as system of record)

**What's sound:** durability→visibility ordering is correct — fsync strictly precedes the data publish and
the `committed_seq` Release-store (`engine_commit.rs:61-88,161`; `engine_introspection.rs:86-101`), readers
Acquire-load, so read-your-writes after ack holds on host and resident paths. Replay stamp determinism:
`created_by`/`deleted_by` re-derived from the log position (`engine_commit.rs:551-561`), so post-replay host
visibility is byte-identical, and the GPU can never diverge from WAL-replayed state **because the GPU is
never the recovery source**.

**Gaps against the target:**
1. **The host store is structurally load-bearing three ways** — recovery applies WAL SQL through the host
   `InMemoryTupleStore` (`engine_commit.rs:535-763`), residency admission scans that store
   (`engine_residency.rs:3256-3277`), and every incremental GPU path's failure fallback is
   invalidate+re-admit-from-host. Deleting the CPU engine requires a WAL-or-GPU-native rebuild+locate path
   first. This is *the* structural blocker, above any shape gap.
2. **`flush_all` rewrites the entire segment per commit** (temp-write + rename + fsync of the full prefix,
   `wal/src/lib.rs:262-298`) with effective group size 1 — commit latency, and therefore read-visibility
   latency, grows O(total WAL length). Ledger #7 (group commit / append-only segment), still open.
3. **Recovery leaves the GPU cold** (S4).

---

## 12. GPU-native gap map (current vs target)

| Concern | Today (current tree, defaults) | Target |
|---|---|---|
| Reads from the wire | Host CPU engine (S1/S2); GPU plane benchmark-only | GPU resident route on every committed table |
| Shapes on sharded tables | 14-shape accept list; ≥8 shapes regress to CPU (F1); reshaping shapes break on versioned tables (E1) | Full parity: every single-buffer shape + visibility-threaded sort/group/distinct/join |
| Wire-expressible SQL | Hand-rolled grammar; JOIN/IS NULL/COUNT(DISTINCT)/multi-key ORDER BY = syntax errors (S3) | libpg_query-backed wire path into the general executor |
| Snapshot correctness | Stamped classes correct; unstamped INSERT + un-co-pinned loads (D3/D4) | Every append stamped; one-snapshot (descriptor,buffer,regions) capture |
| Steady state under writes | First UPDATE/DELETE: index poisoned (F2), fast paths lost (F3), reshaping broken (E1), forever (no VACUUM) | Incremental open-shard index + VACUUM/re-clustering |
| Multi-shard scan | Recompact-all-to-one-buffer per query, one GPU (P-1/P-2) | Push-down-to-shard + combine; streaming over-VRAM (ADR-012) |
| Non-int4 / text / catalog / views | Single-buffer only (mixed tables excluded from sharding); views/matviews/pg_catalog host-only | Sharded type coverage; GPU catalog relations |
| Recovery | WAL→host store; GPU cold (S4); host store = rebuild source | WAL→GPU admission; host store retired |
| Result egress | 3× host materialization, no streaming (P-7) | Single device→wire columnar readback, chunked |

---

## 13. Prioritized recommendations

**Gate the FLIP on (do before committing/pushing the working tree):**
| # | Action | Fixes | Effort |
|---|---|---|---|
| 1 | Decline null-bearing aggregate columns on the sharded bridge (or thread `value_null_off` into the scalar reduce); un-pin the two NULL gate tests to sharded | D1 | S |
| 2 | Add the missing shape remaps + a full single-buffer↔sharded route-parity differential | F1 | S-M |
| 3 | Planner-level decline of versioned tables for un-wired reshaping shapes → CPU fallback (then thread visibility through sort/group/distinct/join) | E1 | S (then L) |
| 4 | Typed `ResidencyInvalidated` error (or match shard strings) so sharded invalidation falls back like single-buffer | E2 | S |
| 5 | Stamp `created_by` on ALL appends (INSERT too) | D3, F8 | M |
| 6 | One-snapshot capture: embed buffer+region Arcs (or ptr+generation) in the shard descriptor; verify ptr in `source_for` | D4 | M |
| 7 | Build PK index over live slots / probe-time deleted_by gate (or re-admit on dup-decline) so UPDATE doesn't poison the index | F2 | M |
| 8 | Shard-aware budget/eviction + rollover budget check | R-1 | M |

**Then pull the real levers:**
| # | Action | Fixes |
|---|---|---|
| 9 | Flip `auto_admit_on_commit` (with 8's budget guard); bulk-admit at recovery end | S1, S4 |
| 10 | Ship the async batched server as the default binary; batcher fallback-to-per-query on ANY failure; supervise the coalescer; NULL decline on the single-buffer template | S2, E3, E4, D2 |
| 11 | Route the facade parse-Err arm to the general executor (wire-expressible JOIN/IS NULL/multi-key ORDER BY) | S3, E5 |

**Structural (ledgered, sequenced by impact):**
| # | Action | Fixes |
|---|---|---|
| 12 | VACUUM / re-clustering compaction for version regions (preserving the dense-route decline) | R-2, F3, E1-permanence |
| 13 | Incremental open-shard index maintenance (then GPU-native build) | F4 |
| 14 | Descriptor-driven device recompaction copy kernel (O-1) → then per-shard push-down + combine (kills recompaction entirely) | P-1, P-2 |
| 15 | Flat result block + chunked DataRow streaming + row/byte cap | P-7 |
| 16 | WAL group commit / append-only segments; WAL-driven (not host-store-driven) residency rebuild — the host-store retirement prerequisite | §11 |
| 17 | Plan/template cache keyed on normalized text + residency generation; thread parsed AST classify→execute | P-4/5/6 |
| 18 | Kernel sweep: sm_90 uplift + `ld.global.nc`, hard-Err the K-1 guard, two-pass emit sizing, pinned DtoH staging, smem sizing | K-1..5, R-4, O-4/5/7 |
| 19 | Deletion sweep: KV mask family (K-6) with host-store retire; test/example-only BETWEEN/membership probes; stale comments (default-OFF claims at `engine_state.rs:525,531`, `engine_expr.rs:2776`, `engine_commit.rs:131,404`; the relocated NULL-blind comment `engine_expr.rs:3061-3067`; `engine_state.rs:508` sentinel doc; batcher `route_id` doc) | hygiene |

---

## 14. Verified sound (credit where due)

- **Kernel layer:** hash agreement, alignment discipline, dup/status protocols, binary-route
  self-validation, pool drop-drain safety, reduction correctness, group-by slot sizing, bitonic sentinels —
  all traced clean (§10).
- **Stamped-visibility machinery:** SV6 stamp ordering (stamp while slots are invisible headroom, region
  install before publish), signed-compare agreement host↔device, region-layout agreement across all three
  consumers, `publish_committed_seq` Release/Acquire pairing.
- **Retire-path hygiene:** `shard_deleted_by/created_by_memory` + host & device shard PK caches purged at
  all 7 retire sites (the one gap is `wave_index`, R-3). Index caches `(ptr,row_count)`-validated +
  Arc-pinned (ABA-safe); no UAF anywhere (buffers Arc-pinned; eviction publishes tombstones, never frees
  under a reader).
- **The FLIP's zero-copy layout contract holds:** capacity-strided single-shard descriptors flow only
  through capacity-aware offset helpers (`relational_model.rs:891-913`); predicate VM, gathers, sorts,
  aggregates, and the join gather all resolve via the descriptor — no dense-stride assumption found.
- **Metadata COUNT internal consistency** (one guard for check+sum) and the version-free gates'
  race-safety-by-publish-ordering.
- **Batcher plumbing:** every exit answers every waiter; needle dedup honors the engine's distinct-needles
  contract; adaptive wait bounded; frame-cap DoS guard on all four read paths; serialization conflicts
  typed (40001).
- **Bind-first discipline** forecloses the case-sensitivity wrong-grouping class (C12 refuted).
- **3VL predicate routing** on the general path: versioned → forced VM+conjuncts; nullable → validity-AND
  or clean error; IS NULL lowers against bitmaps; grouped keys/values validity-aware.

---

## Appendix A — key file map (read path)

- Ingress/egress: `crates/server/src/{main,lib}.rs`; `crates/facade/src/lib.rs`;
  `crates/facade/src/point_lookup_batcher.rs`; `crates/protocol/src/lib.rs:990` (ReadyForQuery).
- Dispatch/routing: `crates/engine/src/engine_select_exec.rs`; `resident_route.rs`;
  `engine_residency.rs:5657-6093` (sharded plan/remap); `engine_select_bind.rs` (bind + boundary pin).
- General executor + sharded bridge + FLIP: `crates/engine/src/engine_expr.rs` — unified source `:2367+`,
  zero-copy `:2486`, metadata COUNT `:2814`, join bridge `:3293`, general resolution `:4422`, visibility
  conjuncts `:1992`, scalar aggregates `:6593`.
- Point-lookup/index: `crates/engine/src/engine_retained_read.rs` (template `:354`, submit `:487`, locate
  `:792+`, batched gather `:1148`, dense route `:1364`, hash/bloom build `:1948+`);
  `engine_resident_probe.rs` (counts `:412`).
- Residency/MVCC: `engine_residency.rs` (admission `:3256+`, purely_int4 `:3315`, append `:3881+`, rollover
  `:4126+`, regions `:4270+`); `resident_storage.rs` (install `:825`, caches `:24-74`); `engine_commit.rs`
  (ordering `:133-161`, invalidation `:297+`); `engine_lifecycle.rs` (flags `:160-190`).
- Kernels: `crates/execution/src/lib.rs` (probes `:9972+`, multi-shard `:10756`, recompaction `:3858`,
  reductions `:13189+`, compaction `:19408+`, sorts `:18001+`); `expr_proto.ptx` (VM); `pack.ptx`,
  `having.ptx`, `gather.ptx`.
- WAL: `crates/wal/src/lib.rs` (`flush_all:262`); recovery `engine_lifecycle.rs:16-60`.

## Appendix B — disposition of the pre-FLIP draft's findings

| Prior | Disposition |
|---|---|
| C1 | Confirmed, now live → D3 (+scope sharpened: stamped classes are correct; residual = INSERT + re-admit) |
| C2 | Confirmed, extended (regions too, re-admit shard_id reuse) → D4 |
| C3 | Downgraded (versioned single-buffer unreachable) → §7; add debug assert |
| C4 | Rerated OK — decline now load-bearing, enforced, race-safe → §7 |
| C5 | Confirmed → D6 |
| C6/C7 | Confirmed → E3 |
| C8 | FIXED on text entry (SLICE B/FLIP); wire path can't express IS NULL → S3 |
| C9 | Confirmed, upgraded to Critical divergence → D2 |
| C10 | Superseded: sharded 3VL structurally verified (§14); differential still worth adding with F1's parity test |
| C11 | Confirmed → D7 |
| C12 | **Refuted** (bind-first forecloses it) → D7 |
| C13/C14 | Confirmed → E2 note + rec #19; ReadyForQuery still hardcoded 'I' (`server/src/lib.rs:124-577`) |
| P1-P14 | Confirmed/updated → §8, §10 (P11 worse on the new shape; P6 partly fixed engine-side, facade discards it) |
| R1/R6 | Confirmed, upgraded (defaults) → R-2, R-1 |
| R2/R3/R4 | Confirmed → E4, S2, P-7 |
| R5 | Confirmed + one wrong doc found (`engine_state.rs:508`) → K-5 |
| S1-S5 (doc drift) | Wave-engine retirement confirmed in code (no compiled symbol); stale default-OFF comments EXPANDED by the FLIP (5 new sites, rec #19) |
