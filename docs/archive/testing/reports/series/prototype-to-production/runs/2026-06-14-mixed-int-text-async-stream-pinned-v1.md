# `mixed_int_text` async-on-pooled-stream + pinned D2H — c64 23× / 11.7× qps; new wall = host API submit

Status: **fix APPLIED, UNCOMMITTED** (user reviews before commit). The `mixed_int_text` c64 wall
(12.7 ms p50 / 4.4k qps) located by the prior cost-breakdown is eliminated: c64 **548 µs p50 /
51.6k qps** (re-runs 526–610 µs / 51.2–55.9k). The serialized-resource fingerprint is gone — qps
now *rises* with concurrency and plateaus, and the route is the **fastest** GPU projection route
(beats `multi_col` 9.7k and `equality_count` 20k). 36/36 execution tests (serial) + 412/412 engine
tests (incl. 39 GPU end-to-end) green; fmt clean; no new clippy.
Date: 2026-06-14
Branch: `phase0-m1-engine-facade` (HEAD `ee13fa32`, real GPU: RTX PRO 6000 Blackwell)
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 2, Open thread (1)/(2)
Follows / resolves: `.../runs/2026-06-14-mixed-int-text-c64-cost-breakdown-analysis.md` (which proved
the wall is the 11 synchronous default/NULL-stream memory ops; this applies the lever it picked) and
`.../runs/2026-06-14-text-route-wall-located-and-fused-v1.md` (the fuse this builds on).

## TL;DR

The cost-breakdown proved 99.7 % of the c64 wall was the route's **11 synchronous, blocking
default/NULL-stream memory ops** (1 `cuMemcpyHtoD` + 2 `cuMemsetD8` + 8 `cuMemcpyDtoH`) — only the
kernel had been migrated to the pooled private stream, so those 11 ops serialized all 64 threads at
the legacy default-stream context-wide barrier. This change moves **every one of them onto the
route's already-pooled private stream** via the `*Async` variants behind exactly **two**
`cuStreamSynchronize`, and adds two reductions the breakdown pointed at:

1. **Async-on-the-pooled-stream** (the lever the data picked): HtoD(needles) + memset(counters) +
   kernel + all result D2H issued stream-ordered on the **pooled** private stream (reused, no
   per-call `cuStreamCreate`/`Destroy` — that per-call churn is exactly why the breakdown's throwaway
   knob only reached 1.64×). Sync #1 after the kernel (so the device-computed counts are readable to
   size the result reads); sync #2 after the result D2H.
2. **Cut copy count + bytes:** the two atomic-append counters are **fused into one 8-byte device
   buffer read back in ONE D2H** (was the single largest post-fix section because it forces the
   mid-pipeline sync), and every device→host copy stages through **pooled pinned (page-locked) host
   buffers** so the async D2H is truly async + DMA-fast.

Result @c64: **12.7 ms → 548 µs p50 (23× lower latency), 4.4k → 51.6k qps (11.7×)** — far past the
breakdown's conservative 1.64× lower bound, because pooling the stream + fusing the counter read +
pinned memory removed the residuals the knob experiment still paid.

## 1. Reproduced baseline (committed `ee13fa32`, concurrent, 50k rows) — apples-to-apples

Forced-clean build (`cargo clean -p gpu_db_execution -p gpu_db_engine -p gpu_db_server --release`
first — the build-env landmine is still active). Matches the prior reports (text c64 ~12.8 ms / 4.3k).

| `mixed_int_text` | c1 | c2 | c4 | c8 | c16 | c32 | c64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| p50 µs | 134 | 167 | 388 | 1,519 | 3,069 | 6,208 | 12,679 |
| qps | 7,285 | 11,701 | 9,643 | 4,990 | 4,739 | 4,546 | 4,409 |

Controls (c64): `multi_col_projection` 5,486 µs / 9,674 qps; `equality_count` 3,134 µs / 20,275 qps;
`count_all` 3,773 µs / 17,312 qps.

## 2. After the fix — full scaling curve (concurrent, 50k rows, same machine, clean build)

| `mixed_int_text` | c1 | c2 | c4 | c8 | c16 | c32 | c64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **p50 µs** | **96** | **112** | **118** | **168** | **287** | **508** | **548** |
| **qps** | **10,113** | **17,756** | **32,764** | **46,812** | **53,574** | **52,347** | **51,597** |
| qps gain | 1.39× | 1.52× | 3.40× | **9.38×** | **11.3×** | **11.5×** | **11.7×** |

**The SHAPE changed completely.** Baseline qps collapsed and flatlined at ~4.3–5.0k from c8 (the
fingerprint of a fully serialized resource). After the fix qps **rises** through c16 and plateaus at
~52k — the serializer is gone; concurrency now adds throughput. p50 grows only 96 → 548 µs across
c1→c64 (was 134 → 12,679 µs). Stable across ≥4 re-runs (c64 p50 526–610 µs, qps 51.2–55.9k).
Controls unchanged within noise (multi_col 9.6–9.7k, equality_count 20.0–20.7k, count_all 17.5–17.9k
qps @c64) — confirming the change is isolated to the single-predicate text path. Results stay correct
(1 row returned for the point lookup). Artifacts: `target/2026-06-14-mixed-int-text-async/`.

## 3. New per-section breakdown @c64 — the wall is now host-side API submit

Same throwaway, env-gated (`GPU_DB_WALL_PROBE`), process-global-atomics probe methodology as the
prior reports, wrapped around the new sections; run at c64; **fully removed before this report**
(`strings | grep -c WALLPROBE` → 0 on the rebuilt binary). µs **per call**:

| section | c64 µs | share | note |
|---|---:|---:|---|
| A_stream_acquire (pool pop) | 0.3 | 0.1 % | pooled — free |
| **B_htod+memset+kernel SUBMIT** | **177.8** | **33 %** | host driver-API submit cost (3 calls) |
| C_sync1 (kernel + counter D2H complete) | 52.3 | 10 % | actual GPU + tiny PCIe |
| **D_result_d2h SUBMIT (6 async + 6 pinned leases)** | **299.5** | **56 %** | host driver-API submit cost |
| E_sync2 (result D2H complete) | 2.1 | 0.4 % | actual copy-engine time |
| F_pinned_copyout (pinned→Vec) | 0.5 | 0.1 % | host memcpy |
| G_assemble (rows + `String`) | 1.1 | 0.2 % | |
| **TOTAL attributed** | **~534** | | tracks the c64 p50 (~595 µs in this run) |

### What this proves

1. **The default-stream serializer is eliminated.** The old breakdown's four ballooning sections
   (C_htod 3,128 / D_memset 1,409 / G_counts 5,687 / H_arrays 3,845 µs = 14,069 µs/call) are gone.
   The *device/PCIe* work is now trivial: **C_sync1 = 52 µs and E_sync2 = 2 µs** (the kernel + all
   transfers complete in ~54 µs combined). The two memsets + counter read collapsed into the 8-byte
   fused-counter D2H inside C.
2. **The NEW bottleneck is the per-call host-side CUDA driver-API submit cost — `B` + `D` = 89 %.**
   Not PCIe bandwidth (E_sync2 = 2 µs), not the kernel (inside C), not a lock (A = 0.3 µs), not
   materialize (F+G = 1.6 µs). It is the raw cost of *issuing* the driver calls (HtoD/memset/kernel
   submit = B; six async D2H + six pinned-buffer leases = D), which still partially serialize on the
   driver's internal submit path at 64 concurrent threads — but at ~30–50 µs/op instead of the old
   barrier's thousands.
3. **This is precisely Open thread (3)'s "architectural per-op floor."** The prior report predicted
   it: *"A second, architectural lever (Open thread 3) remains the per-op floor: batched multi-query
   GPU submission."* The remaining wall is API **call count × driver-lock contention** — the lever
   that addresses it is amortizing submits across queries (batched submission), not anything on this
   route's single-query path.

### How far from OLTP-fast

At **~52k qps / ~550 µs p50 @c64** for a point-lookup mixed int+text projection, the route is now the
**fastest GPU route in the mix** (vs equality_count 20k, multi_col 9.7k, count_all 17.5k) and is
solidly in OLTP territory. The next ~5–10× would come from cutting the host API submit count
(Open thread 3, batched multi-query submission) — e.g. fusing the six result D2H into ONE via an
AoS packed-output kernel would cut D, and a batched submit path would amortize B across queries.
Those are larger, generic changes; this slice already removed the located wall.

## 4. What changed (all in `crates/execution/src/lib.rs`; engine path unchanged)

`launch_cuda_resident_i32_equal_any_project_text` (the single fused launch the `mixed_int_text` route
already takes since `ee13fa32`) now has two paths:

- **async path (driver supports the symbols — the live path here):** lease the **pooled** private
  stream; issue `cuMemcpyHtoDAsync`(needles) + `cuMemsetD8Async`(fused counters) + kernel
  stream-ordered; **sync #1**; read the fused 8-byte counter pair (one `cuMemcpyDtoHAsync` into a
  pooled pinned host buffer); issue the six result `cuMemcpyDtoHAsync` (each only over the populated
  `[0, count)` prefix) into pooled pinned host buffers; **sync #2**; copy pinned→owned `Vec` by typed
  pointer; assemble rows. The kernel-event timing (`last_kernel_event`) is preserved via the pooled
  stream's events, exactly as `launch_on_pooled_stream` did.
- **blocking fallback (old driver lacking any of the async/pinned symbols):** the original blocking
  default-stream path, kept verbatim, but with the same counter fusion (one memset + one D2H of the
  8-byte pair). Correctness is unconditional; only the acceleration is best-effort.

New substrate added (mirrors the existing pools):

- Five **optional** driver symbols loaded once at context construction (`cuMemcpyHtoDAsync`,
  `cuMemcpyDtoHAsync`, `cuMemsetD8Async`, `cuMemHostAlloc`, `cuMemFreeHost`) via a new `opt_sym!`
  macro (returns `Option`; absent → blocking fallback).
- A **`PinnedHostBufferPool`** on `GpuPrimaryContext` with `lease_pinned_host_buffer` /
  `release_pinned_host_buffer` + a `PinnedHostLease` RAII guard — a power-of-two-bucketed free list
  of page-locked host staging buffers, mirroring the existing `OutputBufferPool` (same bucketing,
  same 1 GiB idle cap, freed via `cuMemFreeHost` on drop / context drop). Pooled because
  `cuMemHostAlloc`/`cuMemFreeHost` are themselves driver-serialized — a per-call alloc would
  re-introduce the contention the fix removes.
- Three small shared free functions: `stage_result_dtoh_async` (queue one async D2H into a pinned
  staging buffer, or straight to the dst if no pinned buffer), `copy_pinned_into` (drain a completed
  pinned buffer into its `Vec` by typed pointer — no `bytemuck`), and
  `assemble_i32_text_batch_projection_rows` (the row stitcher, factored out so the async and blocking
  paths share **identical** assembly — same field semantics, same column order → byte-identical rows).

Net: `crates/execution/src/lib.rs` +521 / −66. No other file changed.

### Reuse / recovered prior work

- **Reused** the P2-M1/P2-M2 substrate verbatim: the pooled private stream
  (`acquire_pooled_stream`/`release_pooled_stream` + its timing events), `lease_device_buffer` /
  `OutputBufferPool` (`8c476939`), `cached_function` (module cache), `check_cuda`. The new pinned
  pool is a structural copy of `OutputBufferPool`. The `equality_count` route's existing
  `cuMemsetD8Async`-on-pooled-stream pattern was the template for the async sequencing.
- **Recovery attempt:** searched `git log --all`, `git reflog`, `git stash list`, and dangling
  commits (`git fsck --dangling`) for the two reverted attempts the handoff named ("fused AoS
  packed-output", "pinned batched D2H"). The dangling commits are all autostash WIP snapshots of the
  *output-buffer-pool* work (later committed as `8c476939`) and the *fuse* (`ee13fa32`) — the two
  named attempts were **not** recoverable (reverted via working-tree discard, no stash/commit). So
  the async path was implemented fresh following the breakdown's §4 proven recipe + the report's
  fusion description; the counter-fusion (2→1 D2H) and pinned-host pool are the recovered *ideas*,
  re-validated against the current fused path.

## 5. Correctness / parity

- **Parity preserved by construction.** The result memory layout is unchanged: the same six
  device buffers (values / needle_indices / row_indices / text_starts / text_lens / text_bytes) are
  read back into the same `Vec`s and stitched by the **same** assembler both paths call. The only
  layout change is the two counters sharing one 8-byte buffer (`count` at +0, `text_count` at +4) —
  the kernel still receives two distinct pointers and `atom.add`s into each independently, so no PTX
  change and no observable behavior change. `row_index` (consumed only by tests, not the engine's
  compact-text materialization) is still read and carried correctly.
- **412/412 engine tests** green in parallel (373 non-GPU + 39 GPU), including the GPU suite that
  exercises the resident mixed int+text projection **end-to-end** (the parity-sensitive path) — this
  is the load-bearing correctness gate and it passed on every run.
- **36/36 execution tests** green run serially (`--test-threads=1`).
- `cargo fmt --check` clean. `cargo clippy -p gpu_db_execution -p gpu_db_engine` introduces **no new
  warning** (the lone pre-existing `large_size_difference` on `RelationalRetainedReadSubmissionInner`
  at `engine/src/lib.rs:6638` is unchanged from HEAD — that file is untouched).

### Process note — a PRE-EXISTING flaky GPU test (not this change)

Run **in parallel**, the execution suite intermittently fails `gpu_primary_context_is_cached_and_
loads_each_module_once` (its `cached_module_count() after <= before + 1` assertion races *sibling*
GPU tests loading their kernels into the shared module cache between the two snapshots). Verified
**pre-existing**: `git stash` of this change → unmodified `ee13fa32` failed the **same** test in 2 of
3 parallel runs (identical 35/1 split). It passes 100 % in isolation and 100 % run serially
(`--test-threads=1`), with and without this change. Re-verifiers should run the execution GPU suite
serially (as the prior reports' "14 GPU tests" runs effectively did). Not fixed here — it is an
unrelated test-isolation hazard, not a correctness regression.

## 6. Reproduction

- Baseline / after: `GPU_DB_BENCH_MODE=concurrent GPU_DB_BENCH_ROWS=50000
  GPU_DB_BENCH_CONNECTIONS=1,2,4,8,16,32,64 GPU_DB_BENCH_DURATION_SECS=3 cargo run -p gpu_db_server
  --example gpu_retained_query_mix --release` (force `cargo clean -p gpu_db_execution
  -p gpu_db_engine -p gpu_db_server --release` first — landmine still active; a clean rebuild + the
  baked-in `cuMemcpyHtoDAsync`/`cuMemHostAlloc` strings confirm the binary).
- Tests: `cargo test -p gpu_db_engine --release -- --include-ignored` (412/412);
  `cargo test -p gpu_db_execution --release -- --include-ignored --test-threads=1` (36/36).
- The wall probe + the c64 breakdown were **throwaway** (`GPU_DB_WALL_PROBE`); fully reverted
  (`strings | grep -c WALLPROBE` → 0). **Fix left applied, UNCOMMITTED** — `git status` shows the one
  `crates/execution/src/lib.rs` change + this report. The user decides on commit.
