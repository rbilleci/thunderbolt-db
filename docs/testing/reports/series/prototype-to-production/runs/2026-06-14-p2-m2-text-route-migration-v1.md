# P2-M2 cont. — text route migrated; the projection wall is allocation churn, not orchestration

Status: closed (text route landed on the substrate; **no throughput change** — and that negative
result is the finding: the projection wall is per-call device allocation, not module JIT/sync)
Date: 2026-06-14
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 2 ("Parallel kernels")
Branch: `phase0-m1-engine-facade`
Follows: `.../runs/2026-06-14-p2-m2-projection-migration-v1.md`

## What changed

Migrated the last benchmarked projection route — `launch_cuda_resident_i32_equal_any_project_text`
(`mixed_int_text`: `SELECT int, int, text WHERE int = k`) — onto the P2-M1 substrate, exactly as
`equal_project` was:

- per-launch `cuModuleLoadData`/`Unload` (re-JIT every call) → `primary().cached_function(...)`;
- `launch_with_optional_cuda_event_timing(resident, cuCtxSynchronize, …)` (a **whole-context**
  sync on the default stream) → `launch_on_pooled_stream(resident, None, |stream, _| …)` (a
  private pooled stream synced individually);
- dropped the route's own `cuCtxSetCurrent`/module symbol lookups (the engine dispatcher already
  binds the context at `engine/src/lib.rs:15298`, and the helper re-binds — idempotent).

The route is generic over `R: CudaResidentReadSource` (reader/owner both reach `primary()`).

## Result: the migration did NOT move the text route (GPU-retained mix, concurrent, 50k rows)

| `mixed_int_text` | before (v1, unmigrated) | after (this slice) |
|---|---:|---:|
| c1 p50 | 217 µs | 240 µs |
| c8 p50 | 2,906 µs | 3,041 µs |
| c64 p50 | 26,669 µs | **26,668 µs** |
| c64 qps | 2,199 | **2,167** |

Essentially **identical** — c64 p50 is unchanged within run-to-run noise (re-runs land at
26.1–27.3 ms; pre-migration was 26.7 ms). Contrast `equal_project`
(`multi_col_projection`), which the prior slice moved 24.7 ms → 14.9 ms and which **holds** that
14.9 ms here (3,727 qps) — proving the substrate works and the binary under test is the migrated
one. So the same migration helped one route and not the other. Why?

## The finding: projection routes are allocation-bound

`equal_project` allocates **2** device buffers per call (values + count). The text route allocates
**9** — and they are sized to the **worst case** (`row_count`), not to the match count, because a
single-pass append kernel cannot know the match count before it runs. For this benchmark (50,000
rows, 3 projected columns):

| buffer | size | bytes |
|---|---|---:|
| values (`row_count × proj × 4`) | 50k × 3 × 4 | 600 KB |
| row_indices (`row_count × 8`) | 50k × 8 | 400 KB |
| needle_indices (`row_count × 4`) | 50k × 4 | 200 KB |
| text_starts (`row_count × 4`) | 50k × 4 | 200 KB |
| text_lens (`row_count × 4`) | 50k × 4 | 200 KB |
| text_bytes (`text_bytes_len` = whole column) | ~50k × 6 (`dist-N`) | ~300 KB |
| needles / count / text_count | — | ~tiny |
| **total per call** | | **≈ 1.9 MB** |

That is **9 `cuMemAlloc` + 9 `cuMemFree` per query**, ~2 MB churned, for a point lookup that
returns **one** row. `cuMemAlloc`/`cuMemFree` are heavyweight, context-serializing driver calls;
at c64 they contend on the driver's allocator and dominate the wall. The per-launch JIT and the
whole-context sync the migration removed are real waste, but they are a *small* fraction of this
route's cost, so removing them is invisible against ~2 MB of alloc/free churn. (`equal_project`'s
residual 14.9 ms wall is the same phenomenon with 2 buffers instead of 9 — it too is now
allocation-bound, just less severely.)

This reframes the remaining projection work: **the wall is allocation/transfer, not the kernel and
not the orchestration.** Both are already parallel and already on the cached-module + pooled-stream
substrate. The lever is to stop allocating per call.

## Why land the migration anyway (no perf win)

1. It removes per-launch `cuModuleLoadData` (re-JIT of the text PTX on **every** call) — objectively
   wasteful even when masked, and it will matter once the alloc wall is removed.
2. It removes the **whole-context `cuCtxSynchronize`** — which under concurrency stalls on unrelated
   streams' work; a per-stream sync is the correct concurrency hygiene regardless of this
   benchmark's numbers.
3. It puts the text route on the **same substrate** as `equal_project`, so the upcoming
   output-buffer pooling can be applied uniformly rather than per-route.

It is a correct, tested prerequisite — not a throughput win for this route, and this report does not
claim one.

## Validation

- 13 execution GPU tests + 39 engine GPU tests green (incl. the resident mixed int+text projection
  route end-to-end); `cargo fmt --check` clean; `cargo clippy` clean (the 3 pre-existing
  `too_many_arguments` warnings on the 8-arg text-route family — present at HEAD before this slice —
  are now silenced with `#[allow]`, since one of them is the function migrated here).
- Benchmark artifact: `target/2026-06-14-p2-m2-text-migration/concurrent.txt`.

## Independent adversarial audit

A reviewer was charged to **refute** (A) the migration's correctness (cached-fn reuse, pooled-stream
sync-before-D2H, cross-stream ordering of the synchronous HtoD + memsets, module/buffer
leak/aliasing, context-current on a fresh reader thread), (B) that the "no improvement" result is
real and not a stale-binary artifact, (C) the allocation-churn diagnosis, and (D) test/lint
regressions — reading the diff and **re-running on the GPU**.

**Result: no LIVE BLOCKER.** A–D upheld. The decisive checks: (A) the pre-launch `cuMemcpyHtoD` +
two `cuMemsetD8` resolve to the **synchronous** symbols (host-blocking → complete before the kernel
is enqueued on the pooled stream, despite the different stream), and the helper
`cuStreamSynchronize`s before the route's D2H — no stale read; the removed `module_guard` is correct
(module owned by the context cache, unloaded on Drop). (B) Source/binary mtimes confirm the binary
is freshly built from the migrated source; 3 re-runs reproduced `mixed_int_text` c64 ≈ 26.1–27.3 ms
(unchanged) while `multi_col_projection` held ≈ 15.1 ms. (C) The reviewer counted exactly **9
`cuMemAlloc`** of the reported sizes and wrote a **standalone driver micro-benchmark** on this GPU:
at c64, 9-buffer alloc+free = 17.3 ms vs 2-buffer = 6.9 ms — a **10.4 ms delta that accounts for
~88% of the observed 11.8 ms route-to-route c64 gap** (multi_col ~15 ms vs mixed_int_text ~26.8 ms),
the remainder consistent with the extra D2H/HtoD/memset round-trips. The route ladder
(equality_count 0 allocs → 3.2 ms; equal_project 2 → 15 ms; mixed_int_text 9 → 26.8 ms) is monotonic
in per-call allocs — supporting "allocation-bound," not "orchestration-bound." (D) 13 execution + 39
engine GPU tests pass; clippy + fmt clean.

Acted on its findings (all report refinements; no code defect): softened the "identical to within
1 µs" framing to "unchanged within noise"; corrected `text_bytes` to ~300 KB (`dist-N` is 6 chars),
total ≈ 1.9 MB; recorded the micro-benchmark confirmation above.

## Next (now precisely targeted by the finding)

1. **Output-buffer pooling** — reuse right-sized device output buffers across calls (a per-context
   pool keyed by size, or buffers tied to the resident snapshot's `row_count`), eliminating the
   per-call `cuMemAlloc`/`cuMemFree` churn; pair with **async-on-stream `cuMemsetD8Async`** to drop
   the synchronous memsets. Apply to `equal_project` (2 buffers) and the text route (9 buffers).
   This is the slice that should actually move both projection rows toward the scalar routes.
2. Re-run the GPU-retained mix; expect `multi_col_projection` and `mixed_int_text` to approach
   `equality_count`/`count_all` at c64.
3. The pre-existing claim-order fragility (atomic-append order vs asserted row order) remains tracked
   for wide multi-match projections — today's queries are single-match, so still latent.
