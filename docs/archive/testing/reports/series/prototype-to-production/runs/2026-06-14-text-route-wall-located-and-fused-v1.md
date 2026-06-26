# Text-route wall LOCATED (per-section breakdown) + fixed: `mixed_int_text` c64 1.95×

Status: closed (wall located by measurement; targeted generic fix landed; `mixed_int_text` c64
25.1 ms → 12.8 ms p50 / 2.29k → 4.36k qps; 14 execution + 39 engine GPU tests green; fmt clean;
no new clippy)
Date: 2026-06-14
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 2 ("Parallel kernels"), Open thread (1)
Branch: `phase0-m1-engine-facade`
Follows / resolves: `.../runs/2026-06-14-text-route-wall-investigation-handoff.md`

## TL;DR

The handoff demanded the wall be **located by a per-section wall-clock breakdown before any fix**
(four prior GPU-orchestration guesses had failed). It was located, and the breakdown showed the
prior guesses had all been aimed at **a function the live `mixed_int_text` path never calls**.

- **Located:** ~99.8 % of the c64 wall is the route's **GPU compute path**, but not a single kernel
  — it is a **cascade of 4 separate synchronous GPU launches** the live path issues per query
  (`match_i32_equal_row_indices` + 2×`project_i32_rows` + 1×`project_text_rows`), all on the
  **un-migrated** default/NULL-stream + per-call-`cuModuleLoadData` + per-call-`cuMemAlloc`
  substrate. The single row-indices launch alone is ~20.7 ms/call @c64 (73 % of the wall).
- **Why the 4 prior fixes did nothing:** they all modified
  `launch_cuda_resident_i32_equal_any_project_text` (the *fused* text kernel). The live
  single-query `mixed_int_text` path does **not** call it — it takes the per-column **cascade**
  branch of `execute_relational_equality_multi_column_projection_with_resident_device_memory_probe`.
  The "kernel is only 7 µs" measurement was one kernel of the fused route that wasn't on the path.
- **Fix (generic):** route the single-query mixed int+text projection through the **already-migrated,
  already-audited** single-statement batch path, which fuses the common int4+single-text shape into
  **one** pooled-stream launch (`match_project_i32_equal_any_text_from_payload`) and keeps a cascade
  fallback only for the rare multi-text shape. 4 GPU round-trips → 1.

## 1. Reproduced baseline (concurrent, 50k rows, RTX PRO 6000 Blackwell, HEAD `fd3a291c`)

| query | c1 p50 | c64 p50 | c64 qps |
|---|---:|---:|---:|
| count_all | 59 µs | 3,771 µs | 17,397 |
| equality_count | 351 µs | 3,136 µs | 20,277 |
| multi_col_projection | 108 µs | 5,546 µs | 9,527 |
| **mixed_int_text** | **211 µs** | **25,151 µs** | **2,290** |

Matches the handoff numbers (text c64 ~27 ms / ~2.2k qps). The environment's incremental-build
landmine was checked — a `cargo clean -p …` + clean rebuild reproduced exactly, and a
`strings | grep <literal>` confirmed the instrumented binary.

## 2. THE measurement — per-section wall-clock breakdown (this is the deliverable)

A throwaway probe (env-gated `GPU_DB_WALL_PROBE`, **process-global atomics** so it survives the c64
served-thread fan-out, dumped at process exit) was wrapped around every section of the live
`mixed_int_text` engine path. Run once at **c1** (intrinsic) and once at **c64** (contended).

### µs **per call**, `mixed_int_text`

| section | c1 µs/call | c64 µs/call | c64/c1 | share of c64 |
|---|---:|---:|---:|---:|
| engine_prep (bind/parse/snapshot/offsets) | 2.4 | 8.1 | — | <0.1 % |
| **gpu_row_indices** (`match_i32_equal_row_indices`, launch #1) | **121.1** | **20,658** | **171×** | **73 %** |
| gpu_project_int (`project_i32_rows`, ×2 int columns) | 12.4 | 4,587 | 370× | 16 % |
| gpu_project_text (`project_text_rows`, ×1 text column) | 12.5 | 3,031 | 242× | 11 % |
| materialize (SqlValue row build) | 0.5 | 1.4 | — | <0.1 % |
| metrics_record (bookkeeping) | 0.3 | 1.2 | — | <0.1 % |
| **TOTAL attributed** | **149.2** | **28,287** | | |

(The ~28.3 ms attributed/call @c64 tracks the route's c64 mean; p50 25.1 ms is the median. The
c1 wall is 211 µs vs 149 µs attributed — the ~60 µs remainder is the wire/parse/RwLock-read outside
this method, also negligible.)

### What this proves

1. **The wall is the GPU compute path — but a 4-launch cascade, not one kernel.** GPU sections =
   **99.8 %** of the c64 wall. Everything non-GPU (prep + materialize + metrics + wire) is <30 µs.
   This kills the "wall is engine-side / pgwire-encode / lock-contention" hypotheses by measurement.
2. **All four GPU sub-calls serialize ~170–370× from c1→c64** — the fingerprint of
   default/NULL-stream + whole-context `cuCtxSynchronize` + per-call `cuMemAlloc` contending on the
   one shared driver context. These three launchers
   (`launch_cuda_resident_i32_equal_row_indices`, `launch_cuda_resident_i32_project`,
   `copy_cuda_resident_text_rows`) were **never migrated** to the P2-M1/P2-M2 pooled-stream +
   pooled-buffer + cached-module substrate that fixed `multi_col` (they still call
   `cu_module_load_data` per launch and `launch_with_optional_cuda_event_timing`, which uses the
   NULL stream).
3. **The prior four fixes missed because they targeted the wrong function.** They tuned
   `launch_cuda_resident_i32_equal_any_project_text` (the *fused* kernel). The single-query
   `mixed_int_text` path takes the **cascade** branch and never calls that function — so a
   correct change to it produces exactly the observed **zero** route-throughput delta. The 7 µs
   "kernel" was a real measurement of a kernel that isn't on this path.

## 3. The fix (targeted, generic, reuses audited code)

`execute_relational_equality_multi_column_projection_with_resident_device_memory_probe` had two
branches: all-int4 (already a **single fused** `match_project_i32_equal_from_payload` launch — fast,
left untouched) and **not-all-int4** (the slow per-column cascade). The not-all-int4 branch now
**gates on the predicate count**:

- **single int4 equality predicate** (the `mixed_int_text` route, and the overwhelmingly common
  shape): delegate to the **single-statement batch path**
  (`execute_relational_equality_multi_column_projection_batch_inner`, 1-element slice), which fuses
  **int4 + single-text** into **one** pooled-stream launch
  (`match_project_i32_equal_any_text_from_payload` — the P2-M1/P2-M2 substrate: cached module, no
  per-call JIT; private pooled stream synced individually, no whole-context sync; pooled output
  buffers, no per-call `cuMemAlloc`), and itself falls back to a cascade only for the rare
  multi-text shape.
- **multiple predicates** with a text projection (e.g. `WHERE a = 1 AND b = 2 SELECT …, text`):
  **retain the legacy cascade** unchanged. The batch path does not (yet) accept multi-predicate
  equality, so delegating it would *reject* a query the cascade served correctly — caught as a real
  regression by the existing non-GPU test
  `p8_resident_route_executes_same_column_equality_projection` during this work and fixed by this
  gate. This shape is not on any benchmarked hot path; it stays correct, just unmigrated.

This is generic (the general int4+single-text projection, not a query-shape hack) and inherits a path
already validated by the engine GPU tests and a prior independent audit. Net diff on the method:
**−9 lines** (the gate + delegation added; the cascade retained for the multi-predicate case).

Why this is the right generalization (and ties to Open thread 2): the same un-migrated launchers
(`row_indices`, the per-column `project`s) are exactly what Open thread (2) lists as still on the
pre-pool orchestration. Routing the cascade through the fused/pooled path removes them from the hot
single-query route in one move rather than re-plumbing each launcher.

## 4. Result (concurrent, 50k rows, same machine)

| `mixed_int_text` | before | after | gain |
|---|---:|---:|---:|
| c1 p50 | 211 µs | **132–158 µs** | ~1.5× |
| c8 p50 | (≈2.9 ms) | **1,450–1,551 µs** | ~1.9× |
| c64 p50 | 25,151 µs | **12,808–12,947 µs** | **~1.95×** |
| c64 qps | 2,290 | **4,316–4,360** | **~1.90×** |

Stable across ≥4 re-runs (c64 p50 12.8–12.9 ms; qps 4.31–4.36k), including after the multi-predicate
gate fix below. The other routes are unchanged within noise (count_all 17.3k, equality_count 20.0k,
multi_col 9.5k qps @c64) — confirming the change is isolated to the single-predicate text path.
Results stay correct (1 row returned; correctness re-verified by the 39 engine GPU tests, which
exercise the resident mixed int+text route end-to-end). Artifact:
`target/2026-06-14-text-route-wall/concurrent.txt`.

### After-fix probe (c64) — the cascade is gone

Re-running the probe after the fix attributes the **entire** route cost to one fused call
(14.7 ms/call @c64, vs the old 28.3 ms cascade); the per-column-projection sections are now **0**.
The residual ~13 ms is the fused text kernel's own 11 synchronous round-trips (1 HtoD + 2 memset +
8 D2H), the wall the prior `output-buffer-pooling` report already identified for the fused route —
that is the next lever (below), not this one.

## 5. Validation

- **14** execution GPU tests + **39** engine GPU tests + **373** engine non-GPU lib tests green on a
  clean build. The engine GPU suite validates the resident mixed int+text projection route
  end-to-end (the parity-sensitive path); the non-GPU suite includes
  `p8_resident_route_executes_same_column_equality_projection`, which **caught a regression** during
  this work: a first, un-gated version delegated *all* not-all-int4 mixed queries to the batch path,
  which rejected the multi-predicate-with-text shape (`WHERE id = 3 AND amount = 40 SELECT id,
  amount, label`) that the old cascade served. Fixed by the predicate-count gate (§3); the test (and
  the multi-predicate-mixed query) is green again.
- `cargo fmt --check` clean. `cargo clippy -p gpu_db_engine` introduces **no** new warning (the one
  pre-existing `large_size_difference` on `RelationalRetainedReadSubmissionInner` at lib.rs:6638 is
  unchanged from HEAD — verified by `git stash` A/B).
- Throwaway instrumentation fully removed (`grep wall_probe` → none).

## 5a. Independent adversarial audit — no live blocker

A reviewer was charged to **refute** (A) correctness/parity, (B) that the win is real and not a
stale binary, (C) the test results, (D) the localization methodology, and (E) any metrics/behavioral
regression — reading the diff and **re-running on the GPU**. Verdict: **no LIVE BLOCKER**; A–E upheld.

- (A) Independently reproduced the multi-predicate+text regression and confirmed the
  `filter_offsets.len() == 1` gate fixes it. Sharpened the framing: `execute_relational_select` does
  **no CPU fallback on a resident-route error**, so the un-gated delegation would have made the
  multi-predicate-mixed query **error outright** (not silently fall back) — i.e. the gate is
  load-bearing for correctness, not just perf. Single-predicate parity (exact `[Int4, Text]` rows,
  column order) asserted by tests and passing.
- (B) Clean rebuild (`Compiling gpu_db_engine` confirmed) → c64 25,151→13,116 µs / 2,290→4,249 qps
  (~1.9×), other routes within noise — matches §4.
- (C) execution GPU 14/14, engine GPU 39/39, engine default suite 373/0 — all on a forced-clean
  build. fmt clean; the lone `large_size_difference` clippy warning is **identical on HEAD**
  (pre-existing).
- (D) Methodology sound (per-call means reconcile with p50); throwaway probe confirmed removed.
- (E) The §7 metrics double-count is results-neutral (route telemetry only); correctly disclosed.
- **Process note (carried forward):** the build-env landmine is **active** — the auditor's first
  build no-op'd ("Finished in 0.08s") and produced a spurious FAILED from a stale binary. Every
  number here was derived after `cargo clean -p … --release`. Re-verifiers MUST force-clean.

## 6. Is `mixed_int_text` resolved? — partially; one clear next step

The **located wall (the 4-launch cascade) is eliminated** and the route is ~1.95× faster at c64.
It is **not yet at `multi_col` parity** (12.8 ms vs 5.5 ms @c64) because the now-single fused text
launch still does **11 synchronous round-trips** vs `multi_col`'s 3 — exactly the residual the
`2026-06-14-p2-m2-output-buffer-pooling-v1.md` report flagged for the fused route. That is a
**different, generic** lever and the recommended next step:

> **Cut the fused text route's synchronous round-trips:** issue the 2 memsets + the HtoD on the
> pooled stream inside the launch closure (stream-ordered → kernel still sees zeroed counters +
> uploaded needles), read `count`/`text_count` in one D2H, then the result D2H async-on-stream
> behind a single `cuStreamSynchronize`. Target 11 → ~2–3 sync points, which should bring
> `mixed_int_text` toward `multi_col` (~9.5k qps). This generalizes to the other projection routes
> (Open thread 2) verbatim.

A second, architectural lever (Open thread 3) remains the per-op floor: batched multi-query GPU
submission.

## 7. Note for the next session / audit

This change makes the single-query mixed path call the **batch** path's metrics bookkeeping, which
additionally calls `record_route_execution_observation` (the cascade did not; the dispatcher also
records one). That is a **metrics double-count** of the route observation on the text path — it does
**not** affect query results or the benchmark p50/qps (which time the wire round-trip), but a
follow-up could thread a "don't self-record observation" flag into `batch_inner` for the
single-query caller. Flagged, not fixed, to keep this slice minimal and the hot path correct.
