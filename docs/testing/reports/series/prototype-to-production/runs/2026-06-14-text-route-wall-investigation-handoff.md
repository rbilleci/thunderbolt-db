# HANDOFF — locate the `mixed_int_text` GPU-retained throughput wall (per-section timing breakdown)

Status: OPEN investigation, handoff to a fresh session
Date: 2026-06-14
Branch: `phase0-m1-engine-facade` (HEAD `8c476939` at handoff time — clean working tree)
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 2 ("Parallel kernels")
Related memory: `gpu-projection-routes-perf` (in the session memory dir)

---

## 0. The ask (read this first)

The GPU-retained projection route `mixed_int_text` is stuck at **~27 ms p50 / ~2.2k qps at c64**,
while the comparable `multi_col_projection` is at **~5.7 ms / 9.3k qps** and the scalar routes
(`count_all`, `equality_count`) are at **~3–4 ms / 17–20k qps**. **Four** GPU-orchestration
interventions failed to move it, and the GPU **kernel itself is only 7 µs** (measured). The wall
is therefore **somewhere other than the GPU kernel/orchestration**, and it has NOT been located.

**Your task: instrument a per-section wall-clock breakdown of the text route, run it, and report
where the time actually goes — at c1 AND c64. Do NOT attempt another fix before you have that
breakdown.** The previous session guessed the wall four times (allocation → D2H sync count →
kernel occupancy → per-call op count) and was wrong every time. Measure, don't guess.

---

## 1. Background: what is already done (do not redo)

- **Shipped + audited win (committed `8c476939`):** a device output-buffer pool
  (`OutputBufferPool` / `lease_device_buffer` / `PooledBufferLease` on `GpuPrimaryContext` in
  `crates/execution/src/lib.rs`). It removed per-call `cuMemAlloc`/`cuMemFree` and gave
  `multi_col_projection` **14.9 ms → 5.7 ms p50 @c64 (2.6×)** because that route was
  allocation-bound. See `.../runs/2026-06-14-p2-m2-output-buffer-pooling-v1.md`.
- **Four interventions on the text route — all CORRECT but ZERO throughput change** (each left
  `mixed_int_text` at ~26–27 ms / ~2.2k qps @c64; all either reverted or were no-ops for it):
  1. Buffer pooling its 9 buffers (the shipped pool; helped multi_col, not text).
  2. Async-on-stream pre-kernel `cuMemsetD8Async` + `cuMemcpyHtoDAsync` (reverted).
  3. Pinned-host-memory batched async D2H, 8 D2H → 2 syncs (built, 39 engine GPU tests green on a
     clean build, then reverted — no win).
  4. Fused AoS packed-output: 5 result arrays → 1 records buffer, 8 D2H → 3, 2 memset → 1, 9
     buffers → 4 (built, 39 engine GPU tests green on a clean build, then reverted — no win).
- **The one solid measurement:** GPU kernel event time (via `last_kernel_event_elapsed_us`) at c1
  is `count_all` 3 µs, `equality_count` 29 µs, `multi_col` 6 µs, **`mixed_int_text` 7 µs**. The
  text kernel is ~1.5 % of the route's c64 serialized cost. `ptxas -v` registers: eq_count 15,
  multi_col 16, text 29 — but 29 does not limit occupancy on this arch.

**Conclusion carried into this handoff:** the wall is NOT the GPU kernel, NOT allocation, NOT D2H
sync count, NOT per-call op count. It is unlocated. The cross-route correlation "qps vs op-count"
is real but did NOT hold under intervention — so it is not causal for this route.

---

## 2. Baseline numbers (concurrent, 50k rows, RTX PRO 6000 Blackwell, HEAD `8c476939`)

| query | kernel µs (c1) | c1 p50 | c64 p50 | c64 qps | serialized µs/op (1/qps) |
|---|---:|---:|---:|---:|---:|
| count_all | 3 | 58 | ~3,840 | ~17,100 | 57 |
| equality_count | 29 | 386 | ~3,190 | ~20,100 | 50 |
| multi_col_projection | 6 | 85–108 | ~5,650 | ~9,400 | 106 |
| **mixed_int_text** | **7** | **216–242** | **~26,500** | **~2,200** | **455** |

The "serialized µs/op" column = the effective per-call cost when 64 connections contend. Note
text's c1→c64 gap vs multi_col GROWS (134 µs at c1 → ~350 µs at c64): something in the text path
both costs ~130 µs extra per call AND contends/serializes under concurrency.

---

## 3. THE TASK — per-section wall-clock breakdown

Instrument the text route end-to-end and attribute the ~455 µs serialized (and the ~240 µs c1)
to sections. Measure at **c1** (intrinsic) and **c64** (contended) and compare to `multi_col`.

### 3a. Where to instrument (two layers — do both)

**Layer A — the GPU route** in `crates/execution/src/lib.rs`,
`fn launch_cuda_resident_i32_equal_any_project_text` (currently ~line 3719). Wrap
`std::time::Instant::now()` / `.elapsed()` around each section and accumulate into a
thread-local or a `Mutex<HashMap<&str,Duration>>`, or just `eprintln!` the per-section µs on
the first N calls:
  1. symbol lookups (`resident.lib().get::<...>`)
  2. each `lease_device_buffer` (9 of them)
  3. the `cu_memcpy_htod` needles upload  ← *text-specific; multi_col has none (needles are kernel params)*
  4. the two `cu_memset_d8`
  5. `cached_function`
  6. `launch_on_pooled_stream` (this includes the kernel + `cuStreamSynchronize`)
  7. each of the 8 `cu_memcpy_dtoh`
  8. the materialization loop (`.chunks_exact(...).map(...).collect()`, incl. `String` alloc per row)

**Layer B — the engine method** in `crates/engine/src/lib.rs`,
`fn execute_relational_equality_multi_column_projection_with_resident_device_memory_probe`
(~line 19443). The text branch calls `match_project_i32_equal_any_text_from_payload` at ~line
19966; there is engine-side work before/after (text-column layout lookup
`resident_device_text_column_layout`, metrics snapshot diffing, route-observation recording at
~20199, result-row construction). Time: (i) the GPU call, (ii) everything else in this method,
(iii) the result encode/return. Compare against the int-only branch (which `multi_col` takes).

Also consider timing in `crates/server/src/lib.rs` (pgwire encode/send) if Layers A+B don't
account for the c1 latency — but start with A+B.

### 3b. How to read it out cheaply (no over-the-wire needed for c1)

The benchmark's **acceptance gate** runs each query once **in-process** (no wire). Use it: in
`crates/server/examples/gpu_retained_query_mix.rs` (gate at ~line 178–192), loop each accepted
query ~50× and print the per-section accumulators. This isolates engine+GPU from the wire.
Pattern already proven last session — that is how kernel time (7 µs) was obtained, via
`engine.metrics().snapshot().last_kernel_event_elapsed_us` (`engine.metrics()` is public at
`engine/src/lib.rs:24193`; the getter is `execution` `last_kernel_event_elapsed_us()` at ~731).

For **c64**, you need the per-section timing aggregated across the served threads (e.g. atomic
counters summed, printed on shutdown), since the gate is single-threaded. The c64 contention
behavior is the real target — design the instrumentation to survive concurrency (atomics, not a
shared `Instant`).

### 3c. What a good result looks like

A table: section → µs at c1 → µs at c64 (text), beside the same for multi_col. The wall is the
section whose c64 time dominates and that text has but multi_col doesn't (or has much more of).
THEN propose a targeted fix for that specific section and measure it.

---

## 4. Ranked hypotheses to test with the breakdown (do not pre-commit to any)

1. **The `cu_memcpy_htod` needles upload.** Text uploads needles to a device buffer every call
   (synchronous, default/NULL stream → a context-wide barrier at c64). `multi_col` passes needles
   as **kernel params** (no HtoD at all). This is the single clearest text-vs-multi_col
   structural difference. If the breakdown fingers it: the generic fix is to pass small needle
   sets as kernel params (like `equal_project`), falling back to HtoD only for large needle sets.
2. **Default/NULL-stream serialization.** The route's synchronous `cuMemcpyHtoD`/`cuMemsetD8`/
   `cuMemcpyDtoH` run on the default stream; at c64 each is a global sync point. (Pinned-D2H moved
   the D2H to a private stream and it did NOT help — so weigh this against that negative result.)
3. **Engine-side text handling** (Layer B): text-column layout, result `String` construction, or
   a shared lock touched only on the text path.
4. **pgwire TEXT encoding** of the result column (server layer) — only if A+B don't explain c1.

After the wall is found, the architectural lever the previous session flagged is **batched
multi-query GPU submission** (one submission serving many concurrent point-lookups — recovers the
old M0 owner-thread batching advantage; per-op floor today is ~50 µs serialized / ~20k qps). Keep
any fix GENERIC across scan/projection shapes — the user explicitly does NOT want query-shape
fast paths beyond the params-vs-HtoD distinction that `multi_col` already embodies.

---

## 5. Key file/symbol anchors (line numbers drift — grep by symbol)

- `crates/execution/src/lib.rs`
  - text route: `launch_cuda_resident_i32_equal_any_project_text` (~3719); its PTX entry
    `gpu_db_resident_i32_equal_any_project_text` (~3752); host body after the PTX (~3990–4302).
  - multi_col route (the responsive one to compare): `launch_cuda_resident_i32_equal_project`
    (~2850).
  - equality_count (the well-scaling grid-stride scan): `launch_cuda_resident_i32_equal_count`
    (~2225), entry `gpu_db_resident_i32_equal_count_parallel`.
  - substrate: `launch_on_pooled_stream` (~8715), `GpuPrimaryContext` + `lease_device_buffer`
    (~290), `last_kernel_event_elapsed_us()` (~731).
- `crates/engine/src/lib.rs`
  - mixed/multi projection method: `execute_relational_equality_multi_column_projection_with_resident_device_memory_probe`
    (~19443); text branch call (~19966); kernel-time read (~20199).
  - resident-route dispatch + context bind: `set_current_context()` (~15298); resident-route
    kernel-time read (~15401); `pub fn metrics()` (~24193).
- `crates/server/examples/gpu_retained_query_mix.rs` — the benchmark; gate (~178–192);
  `bench_cell` (~48); env knobs (~100–108).

---

## 6. How to build, test, benchmark (and the BUILD-ENV LANDMINE)

**LANDMINE:** this environment's cargo **incremental build is unreliable** (a mid-session system
clock change corrupted fingerprinting; `cargo build` would report "Finished" without recompiling,
and recompiled rlibs were not relinked into dependent crates/examples). Symptoms: your source
edit appears to have no effect.

**Mitigations (use every time you need a trustworthy result):**
- Before a trustworthy run: `cargo clean -p gpu_db_execution -p gpu_db_engine -p gpu_db_server --release`
  (note `--release`; a profile-mismatched `clean -p` silently no-ops). A full `cargo clean` (≈15
  min rebuild) is the nuclear-but-certain option.
- VERIFY the binary actually contains your change: `strings target/release/examples/gpu_retained_query_mix | grep <your-new-literal>` and confirm `Compiling gpu_db_<crate>` lines appear in build output.

**Commands:**
- Execution GPU tests (14 expected): `cargo test -p gpu_db_execution --release -- --ignored`
- Engine GPU tests (39 expected — these validate text-route correctness end-to-end; ~82 s):
  `cargo test -p gpu_db_engine --release -- --ignored`
- Benchmark (the source of all numbers above):
  `GPU_DB_BENCH_MODE=concurrent GPU_DB_BENCH_ROWS=50000 GPU_DB_BENCH_CONNECTIONS=1,8,64 GPU_DB_BENCH_DURATION_SECS=3 cargo run -p gpu_db_server --example gpu_retained_query_mix --release`
- GPU tests are `#[ignore]`-gated (need the local GPU); run with `-- --ignored`.
- Profilers: `ncu` is present but **blocked** (`ERR_NVGPUCTRPERM`, needs admin); `nsys` runs but
  its importer binary is missing so `--stats` can't post-process. Don't rely on either — use
  in-code `Instant` timing.

---

## 7. Working norms for this project (apply them)

- Spike/soundness-first; additive verifiable slices; checkpoint rather than rush a deep refactor.
- Every milestone: `cargo test` (incl. `--ignored` GPU) + `cargo fmt` + `cargo clippy` clean, a
  dated run report under `docs/testing/reports/series/prototype-to-production/runs/`, and a commit.
- Run an independent adversarial audit of substantial work before declaring done; fix findings.
- Report outcomes honestly — no overclaiming (this whole investigation is a model of that).
- No backwards-compat constraints (delete/rewrite freely). Commit/push only when the user asks.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- The text route must stay GENERIC — no per-query-shape fast paths.

---

## 8. Definition of done for the next session

1. A per-section timing breakdown (c1 + c64, text vs multi_col) that **names the dominant
   section**. Write it up as a dated run report.
2. A targeted, GENERIC fix for that section, with before/after benchmark numbers, 14 execution +
   39 engine GPU tests green (on a clean build), fmt + clippy clean, an adversarial audit, and a
   commit. Target: move `mixed_int_text` c64 meaningfully toward `multi_col` (~9k qps) or beyond.
3. Update the `gpu-projection-routes-perf` memory with the located wall (replacing the
   "UNLOCATED" status).
