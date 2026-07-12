# ARCHIVED — Read-path post-FLIP reassessment

> Historical point-in-time analysis. It is not an executable plan. Any surviving obligation is tracked only
> in `docs/PLAN.md`.

**Date:** 2026-07-02 (later the same day as the baseline report)
**Baseline:** `docs/reviews/read-query-path-assessment.md`, written against `47978626` + the then-uncommitted
FLIP working tree.
**Now:** HEAD `e5632bc2` (19 commits later: THE FLIP `689ab73f` shipped with its own opus audit, WAL D1–D4,
phase-C write slices, det-CC waves, retirement A1) **plus** the uncommitted retirement-A2 working tree
(`dml_device_resolve_enabled` default-ON, dup-decline caching, `ShardPkHit.row_id`). Method: static code
analysis, all claims re-verified against this tree.

---

## 1. What changed on the read path

### Closed since the baseline report

| Baseline finding | Status | Evidence |
|---|---|---|
| **F1 — sharded remap gap (~1000× CPU cliffs)** | ✅ **CLOSED** (FLIP audit adopted it) | Remap now covers `int4_equality_count`, `int4_range_count`, `int4_between_scalar_aggregate`, `int4_projection`, `int4_composite_equality_multi_column_projection` (prefix arm), `int4_filter_group_count` (grouped-bridge arm), and a `sharded_int4_filtered_scalar_aggregate` residue arm for filtered SUM (`engine_residency.rs`, remap block). Gate test asserts rows == single-buffer oracle AND no new CPU demotion via `executed_target`. Residue: **Eq-filtered SUM is CPU on BOTH layouts** (pre-existing parity, ledgered) — still a GPU-native gap, no longer a *regression*. |
| **§11.2 — WAL `flush_all` rewrites the whole segment per commit (O(history) visibility latency)** | ✅ **CLOSED** | D1+D2 `a0f69732`: append-only writer O(record)/commit + checkpoint truncation; D3a `7cf4a9c9` one fsync per apply_batch; D3b `8976c483` designated-flusher group commit (concurrent DML fsync lock-free, shared per group); D4 `d1ec63ae` deployment config. Ledger #7 done. |
| **Ledger #1 — host O(table) `prepare_delete/update` seq-scan** (read-adjacent: it dominated DELETE/UPDATE latency) | ✅ **CLOSED** | Phase C slice 1 `9876d297` value-index resolve O(matches), default-ON kill-switch; `8d7e812e` fixed the low-cardinality O(rows-sharing-value) commit cliff (persistent `imbl::Vector` slots); slice 1b `7b48fda8` extends the flat path to constrained tables. |
| **D4 (one sub-hazard) — heterogeneous shard layout → wrong-column recompaction slice** | ✅ closed by the FLIP audit | Per-shard layout-uniformity guard: every shard's `resident_device_int4_columns` must equal shard 0's, clean error on mismatch (`engine_expr.rs:~2588-2610`). |
| *(not in baseline — found by the FLIP burn-in)* matview CREATE/REFRESH deadlock: the sharded planner arm read budgets via the `ddl_catalog()` latch the matview's internal read already held | ✅ fixed in `689ab73f` | Lock-free ArcSwap budget mirror on `read_state`, republished by the `&mut` budget setters — also removes a latch acquisition from every planned read (small perf win). |
| *(baseline §2 dedup)* SLICE-B `engine_sql_pg` sharded special-case | ✅ deleted as redundant — `execute_resident_expr_select_with_binding` is now the ONE `src:None` resolution chokepoint (as the baseline recommended structurally). |

### Partially addressed

| Baseline finding | Status | Evidence |
|---|---|---|
| **F2 — one UPDATE poisons the PK index (dup key → cached decline → per-read O(table) scans)** | 🟡 **cost capped, demotion remains** (uncommitted A2) | The A2 working tree confirms the finding empirically — "single-row UPDATE p50 went linear, 358→887µs at 64k→262k, rebuild-to-decline each statement" — and caches the decline (dup-ness is monotone under appends; only a ptr change revalidates) (`engine_retained_read.rs:~886-897`). This removes the *rebuild* cliff but the table **still declines every index route** → recompaction scans until re-admit/VACUUM. The real fix (build over live slots / newest-slot map with probe-time `deleted_by` gate) is still open. |
| **F4 — O(shard) index rebuild per write→read cycle** | 🟡 dup case only | A2's cached decline skips rebuilds that would *re-discover a dup*; a normal (non-dup) append still triggers the full DtoH → host-build → HtoD, twice (host + device caches). Incremental open-shard maintenance still open. |

### New read-path surface added by the retirement program (A1/A2)

- **A third per-shard device region**: `shard_row_id_memory` (row-identity, A1 `1875441c`) — used by the DML
  device resolve; now captured into `ShardPkHit` (A2, uncommitted). It is correctly **not** part of the
  read-side `version_free` checks (it carries no visibility semantics). Its retire-site purges were covered
  by the A1 audit; any future retire/admit site must now mirror **three** regions + two index caches.
- The read-path locate machinery is becoming the DML resolve engine (`dml_device_resolve_enabled`
  default-ON, uncommitted): the point-lookup index probes now serve reads AND writes — F2/F4's index-health
  findings therefore now gate **write** latency too, raising their priority.

---

## 2. Still open — defects (unchanged unless noted)

**Live wrong-results, now shipped on origin/main (was working-tree-only at baseline):**

1. **D1 — NULL-blind sharded scalar aggregates** 🔴 Critical → ✅ **FIXED (`cdb44395`, opus audit
   SHIP-WITH-FIXES, both findings adopted as ledger rows):** the aggregate validity conjunct
   (`col IS NOT NULL` ANDed into the predicate on-device) fixes SUM/AVG/MIN/MAX + unlocks scalar
   COUNT(DISTINCT nullable); the HOST finalizer's own NULL bugs (MIN→NULL-as-smallest, SUM/AVG
   hard-error) fixed in the same slice; PK NOT NULL (PG 23502) enforced in preflight + both validator
   arms + ADD PRIMARY KEY. Sharded-default GPU gates + sabotage-verified. Original finding for the
   record: `purely_int4` is
   type-only (nullable int4 tables shard-admit, `engine_residency.rs:4223-4228`); the remapped
   bare/filtered aggregate shapes reach the general executor's Sum/Min/Max/Avg arms which reduce raw
   payloads with **no validity consult** (`engine_expr.rs:6772-6920`; only the empty-set→NULL guard at
   `:6744-6765` and the `CountDistinct` nullable clean-error at `:6925+` exist); no nullable decline exists
   in the bridge or the route; the three NULL gate tests are pinned `set_shard_residency_enabled(false)`
   (`tests/resident_probe.rs:1227,1331,1410`). `MIN(amount)` over `{10, NULL, 30}` on a default-sharded
   table returns **0**. The FLIP's F1 oracle gate compares layouts on non-null data only, so it cannot
   catch this. **This is the top open item.** Fix: decline null-bearing aggregate columns on the sharded
   bridge (mirror the point-route decline) or thread validity into the scalar reduce (the grouped path
   already does); un-pin the gate tests.
2. **D2 — batched single-buffer NULL divergence** 🔴 (facade untouched since baseline; exposure still
   gated on the async server, S2).
3. **D3 — INSERT appends unstamped/born-visible before `publish_committed_seq`** 🟠 — still explicit policy
   (`engine_residency.rs:4869`, `:5078-5088`); the metadata COUNT inherits it.
4. **D4 — generation TOCTOU** 🟠 — the layout guard landed (above) but descriptors, buffers, and regions
   are still resolved in separate lock-free loads (`shards.load()` at `engine_expr.rs:~2440`; per-shard
   `source_for` `.get()`s at `:~2596`; region `.get()`s at `:~2660+`; metadata-COUNT region checks
   `:2890-2907`); re-admit still purges regions and reinstalls `shard_id 0`. Tombstone-resurrection and
   stale-descriptor pairings remain possible in the re-admit window.
5. **D5 — `int4_range_count` NULL-blind** 🟡 (downgraded from High): sharded tables now route range counts
   to the 3VL-aware bridge (F1 remap), shrinking exposure to **mixed-type single-buffer** tables
   (`engine_resident_probe.rs:440-464` unchanged).
6. **E1 — versioned reshaping/JOIN hard-errors, permanent (no VACUUM)** 🟠 — clean-errors unchanged
   (`engine_expr.rs:2992`, `:4591`, join arm); no planner decline, no CPU fallback.
7. **E2 — sharded error strings bypass the CPU fallback** 🟠 — matcher still substring-only
   (`lib.rs:245-257`); shard strings still non-matching (`engine_expr.rs:2526,2537`).
8. **E3/E4 — batcher `fail_group` + unsupervised coalescer** 🟠 — byte-identical
   (`point_lookup_batcher.rs:509,533,623`).
9. **R-1 — eviction blind to shards / rollover unbudgeted** 🟠 — the code's own comment now concedes it:
   "the eviction loop draws its candidates only from single-buffer snapshots"
   (`engine_residency.rs:1721-1724`). The FLIP's budget-mirror fix made budget *reads* lock-free but did
   not add shard-aware eviction.
10. **R-2 — no VACUUM** 🟠 — unchanged; A2's dup-decline comment explicitly names "VACUUM re-clustering"
    as the only event that clears a dup, adding a fourth consumer waiting on it (E1 permanence, F2
    demotion, F3 fast-path loss, dup clearing).
11. **R-3 `wave_index` purge gap, R-4 O(rows×proj) lease, K-1..K-6, D6, D7, F5 (join `committed_seq()`
    re-read — re-verified at `engine_expr.rs:3396`), F6, F7, F8** — all unchanged.

**Unpulled product levers (unchanged):** S1 `auto_admit_on_commit=false` (`engine_lifecycle.rs:172`);
S2 shipped binary = sync `serve`, no batcher (`server/src/main.rs:20`; server changes since baseline were
WAL-config only); S3 facade strict parser — GPU JOIN / IS NULL / multi-key ORDER BY still wire-unreachable
(`facade/src/lib.rs:261,419,529`); S4 recovery leaves the GPU cold.

---

## 3. Optimizations that still exist (ranked)

The execution crate has **zero commits** since the baseline — every kernel-level item stands. Ranked by
expected impact on the current default (sharded) read plane:

1. **Device recompaction copy kernel (O-1)** — the multi-shard scan tax is still S×C serialized blocking
   `cuMemcpyDtoD` + blocking fills (`execution/src/lib.rs:3858-3911`). Zero-copy covers only the
   1-surviving-shard version-free case; every multi-shard survivor set, every versioned table, and **every
   sharded JOIN side** (built un-pruned, predicate `None`, `engine_expr.rs:3396`) still pays it per query.
   One descriptor-driven copy kernel (the multi-shard-probe pattern applied to copies) or batched async
   DtoD is the single biggest read-path lever, and the ledgered #4 endgame (per-shard push-down + combine)
   subsumes it.
2. **Unified-source caching (P-1)** — nothing caches the recompacted buffer across identical reads within a
   generation; a version-cache keyed on the shard generation would erase the tax for read-heavy workloads
   without waiting for #1.
3. **Incremental open-shard index maintenance (F4)** + **live-slot index build (F2)** — now doubly valuable
   because A2 makes the same index drive DML resolution; every single-row write currently invalidates the
   open shard's index and the first UPDATE disables it entirely.
4. **Result egress (P-7)** — unchanged: 3× host materialization (`facade/src/lib.rs:392-397` per-cell
   `.cloned()`; `server/src/lib.rs:240-289`), no streaming, no row cap. Flat result block + chunked
   DataRow flush.
5. **COUNT precheck removal (P-3)** — the sharded bridge still runs the CountAll precheck + real pass
   (2 executor passes + 2× host index Vecs) for every scalar aggregate, now including all the newly-remapped
   F1 shapes — the fix got *more* valuable post-FLIP. The premise (empty-set hard-error) is dead code:
   the general path returns typed NULL itself (`engine_expr.rs:6744-6765`).
6. **Plan/decision threading + plan cache (P-4/P-5/P-6)** — still plan×2 + telemetry×2 + bind×3-4 per
   accepted query; no cache.
7. **Pinned-DtoH staging on the dense-probe completion (O-4)** and **needle HtoD** — two sync pageable
   copies per batch on the hottest route; the atomic path already has the pinned+async pattern to copy.
8. **sm_90 uplift + `ld.global.nc` sweep (O-5)** — 21 hot probe/scan modules still target sm_30; `.nc` is
   free bandwidth on immutable resident data.
9. **Device scan for ordered compaction (O-2)** — removes the mid-query DtoH→host-scan→HtoD round trip on
   every general predicate; **radix 8-bit digits (O-3)** halves ~49 launches per ORDER BY.
10. **Two-pass emit sizing (R-4)**, **smem sizing to blockDim (O-7)**, **CUDA graphs for the fixed
    probe sequence (O-8)**, **descriptor staging in the multi-shard kernel (O-6)**, **shape-key/String
    allocs in the batcher and locate paths (P-9/P-12)** — all still available, all unchanged.
11. **Cheap correctness-adjacent hardening from the baseline still pending:** hard-`Err` the K-1 u32 index
    emit guard, `wave_index` purge at the three retire sites (R-3), typed `ResidencyInvalidated` (E2),
    metadata-COUNT `LIMIT` corner (D7).

---

## 4. Bottom line

The FLIP shipped in materially better shape than the working tree I assessed — its audit independently
closed F1 (the biggest coverage regression), added the layout guard, and caught a deadlock the baseline
missed; the WAL and phase-C commits closed two of the three structural §11 blockers' worst costs (O(history)
fsync, O(table) DML resolve). What did **not** get fixed is the baseline's #1 gate: **D1 ships NULL-wrong
aggregates on the now-default sharded layout**, with its regression tests pinned to the non-default layout —
that, plus the D3/D4 snapshot cluster and the E1/E2 versioned-table availability cluster, are the remaining
correctness debt of the flip. The optimization frontier is unchanged at the kernel layer (no execution-crate
commits) and has consolidated host-side into one theme: **stop paying the per-read recompaction** (device
copy kernel → generation cache → per-shard push-down/combine), with the open-shard index lifecycle now
gating both reads and writes.
