# P2-M2 cont. — output-buffer pooling: multi_col 2.6× faster; text route's wall moves to memcpy

Status: closed (pooling lands; `multi_col_projection` 14.9 ms → 5.7 ms @c64; `mixed_int_text`
**unchanged** — its wall is now the synchronous memcpy round-trips, not allocation)
Date: 2026-06-14
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 2 ("Parallel kernels")
Branch: `phase0-m1-engine-facade`
Follows: `.../runs/2026-06-14-p2-m2-text-route-migration-v1.md`

## What changed

The text-route report identified the projection wall as per-call device **allocation churn**
(`cuMemAlloc`/`cuMemFree`, driver-serialized, sized to worst-case `row_count`). This slice adds a
**device output-buffer pool** on the shared `GpuPrimaryContext` and routes both projection paths
through it:

- `OutputBufferPool` — a power-of-two-bucketed free list (`BTreeMap<bucket_bytes, Vec<ptr>>`) with
  an idle-bytes cap (1 GiB; releases beyond it free instead of pooling). `lease_device_buffer(n)`
  reuses a pooled buffer of the matching bucket or allocates one; `PooledBufferLease` returns it on
  drop (success/error/panic). `ptr` is a plain field, so routes read `lease.ptr` exactly like the
  old `CudaDeviceAllocationGuard.ptr` — only the construction site changed.
- **Reused buffers are not zeroed.** Correctness rests on: only the atomic-append **counters** are
  memset each call; the **needles** input buffer is fully overwritten by its HtoD upload; and every
  output is read back only over `[0, count)` (the region the kernel wrote). Stale bytes beyond the
  written region are never observed.
- Applied to `equal_project` (2 buffers) and `equal_any_project_text` (9 buffers). Each route
  dropped its `cuMemAlloc`/`cuMemFree` symbol lookups + the unused type aliases.
- New GPU test `gpu_output_buffer_pool_reuses_buffers_and_isolates_concurrent_leases`: a released
  buffer is reused (same device ptr); two simultaneously-held leases of one bucket are distinct (no
  aliasing); released leases return to the idle pool; different sizes use different buckets.

## Result (GPU-retained query mix, concurrent, 50k rows)

| route | metric | pre-orchestration | post-orchestration | **post-pooling** |
|---|---|---:|---:|---:|
| `multi_col_projection` | c64 p50 | 24,690 µs | 14,912 µs | **5,689 µs** |
| `multi_col_projection` | c64 qps | 2,142 | 3,793 | **9,312** |
| `multi_col_projection` | c8 p50 | 3,202 µs | — | **694 µs** |
| `mixed_int_text` | c64 p50 | 26,669 µs | 26,668 µs | **27,249 µs** |
| `mixed_int_text` | c64 qps | 2,199 | 2,167 | **2,141** |

For context, the scalar routes this run: `count_all` c64 ≈ 3.8 ms / 17.1k qps; `equality_count` c64
≈ 3.2 ms / 20.0k qps.

- **`multi_col_projection`: a real 2.6× at c64** (14.9 ms → 5.7 ms, 3,793 → 9,312 qps) — pooling
  removed the per-call `cuMemAlloc`/`cuMemFree` that was its binding constraint. It now lands within
  ~1.7× of the scalar routes (was ~5×). c1 is unchanged (~107 µs — at low concurrency the alloc was
  cheap; the win is purely at contention).
- **`mixed_int_text`: unchanged** (~27 ms @c64), even though its 9 buffers are now pooled too. The
  same fix that moved `multi_col` did nothing here — so allocation was **not** this route's binding
  constraint at c64.

## The finding: the text route is bound by synchronous memcpy round-trips, not allocation

Pooling is genuinely active in the text route (no `cuMemAlloc` remains in it). What it has that
`multi_col` does not is **synchronous host-blocking memcpy/memset round-trips** — each a separate
context round-trip that serializes on the driver at c64:

| route | HtoD | D2H | memset | **total sync round-trips** | c64 p50 |
|---|---:|---:|---:|---:|---:|
| `multi_col_projection` | 0 | 2 | 1 | **3** | 5.7 ms |
| `mixed_int_text` | 1 | 8 | 2 | **11** | 27.2 ms |

The latency ratio (27.2 / 5.7 ≈ **4.7×**) tracks the round-trip ratio (11 / 3 ≈ **3.7×**), the kernel
and fixed overheads diluting it. The 8 D2H are the bulk: `match_count`, `compact_text_len`, then
`values`, `needle_indices`, `row_indices`, `text_starts`, `text_lens`, `text_bytes` — each a
separate synchronous `cuMemcpyDtoH`. **This refines the prior report's diagnosis:** the earlier
driver micro-benchmark measured alloc/free churn *in isolation* and found it large, but removing it
from the live route freed ~0 wall-clock at c64 — so for the text route the alloc churn overlapped
with (was hidden behind) the longer memcpy pole. Allocation was `multi_col`'s binding constraint;
**memcpy round-trips are the text route's.**

This is why the slice is still worth landing for both routes: pooling is correct and removes real
driver-serialized work from each, and it is a prerequisite for the next lever — but only `multi_col`
was alloc-bound, so only `multi_col` shows the win here. No claim is made that the text route
improved.

## Validation

- New pool GPU test + 13 prior execution GPU tests = **14** green; **39** engine GPU tests green
  (both pooled routes end-to-end, incl. the resident mixed int+text projection); `cargo fmt --check`
  + `cargo clippy` clean.
- Artifact: `target/2026-06-14-p2-m2-buffer-pooling/concurrent.txt`. The `equality_projection`
  candidate remains CPU-fallback (excluded) as before.

## Next (precisely targeted)

1. **Cut the text route's synchronous round-trips** (the now-identified wall): async the 2 memsets
   + the HtoD on the pooled stream (in the launch closure — stream-ordered, so the kernel still sees
   zeroed counters + uploaded needles); read `count`/`text_count` in one D2H, then issue the result
   D2H async-on-stream behind a single `cuStreamSynchronize`. Target: 11 → ~2–3 sync points, which
   should bring `mixed_int_text` toward `multi_col`.
2. Optionally async `multi_col`'s remaining 3 round-trips for a further drop toward the scalar
   routes.
3. The pre-existing atomic-claim-order fragility (multi-match projections) remains tracked — today's
   queries are single-match, so latent.
