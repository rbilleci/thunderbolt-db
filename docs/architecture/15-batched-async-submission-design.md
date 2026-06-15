# 15. Batched / async GPU submission (Thread 3) — design

**Status:** design approved, staged implementation in progress (2026-06-15).
**Branch:** `phase0-m1-engine-facade`. **Supersedes the lost M0 owner-thread batching.**

## Problem

After the P2-M2 pooled-async migration, every GPU read route rises-then-plateaus at the
**per-call host-side CUDA driver-submit floor** (~14–19 µs/op serialized ⇒ ~70–95k qps).
Each concurrent query is still its OWN submit→sync cycle, so the host cannot submit fast
enough past that point. Thread 3 amortizes that floor by **batching many concurrent
point-lookups into one GPU submission** with non-blocking completion — recovering the
owner-thread batching M0 had.

## Key finding (this is mostly wiring, not CUDA)

The machinery already exists and is tested; only the *production wiring* is missing:

- **Batched kernel** — `gpu_db_resident_i32_equal_any_project` (`crates/execution/src/lib.rs:3673`)
  already scans each row against ALL N needles and tags every match with `needle_index`.
- **Submit→complete split** — `submit_cuda_resident_i32_equal_any_project` (`:3641`) +
  `CudaI32EqualAnyProjectSubmission` (`Send`, with a Drop-drain) / `complete_detached` (`:1425`).
- **Engine batch API** — `submit_relational_retained_read_jobs_with_resident_device_memory_probe`
  (`crates/engine/src/lib.rs:15550`) coalesces N jobs into ONE kernel launch;
  `complete_relational_retained_read_submission` (`:15616`) returns `Vec<RelationalSelectResult>`
  **already sliced one-per-job** by `needle_index` (`:15880`). Per-query slicing + the
  stale-generation guard are proven by engine tests (`:29044`, `:29129`). **Callers today: only
  the benchmark example + tests — never `crates/server`/`crates/facade`.**
- **Dual-trigger batcher** — `gpu_db_batching::DualTriggerBatcher` (`crates/batching/src/lib.rs`):
  count + time triggers, requeue, deadline. Generic, unit-tested.

M0's owner loop lived in `crates/engine/examples/p8_engine_pgwire_benchmark_endpoint.rs`
(`run_worker:407`, `execute_retained_read_runtime_batch:500`) and was never carried into the
new façade server (`crates/facade` `execute_on_shared_engine`, which is per-query by construction).

## Architecture

An **engine-side coalescing batcher owned by `SharedEngine`** (in `crates/facade`), fronted by
per-route-class request queues and a small pool of **owner/coalescer threads**. Each coalescer:
drain a queue into a batch (size-or-time triggered) → take ONE engine **read lock** → run
submit→complete for the whole batch → scatter sliced results back to waiting callers via
`oneshot` channels. The async ingress, instead of `spawn_blocking(execute_on_shared_engine)` per
query, hands batch-eligible `SELECT`s to the batcher and `await`s a oneshot (holding no semaphore
permit while parked).

### Design decisions (with trade-offs)

1. **Batcher in `crates/facade` (on `SharedEngine`), engine stays single-threaded mechanism-only.**
   Concurrency stays out of the engine (plan §2.1); the façade already owns the `RwLock` and so
   owns the "run the batch under one read lock" guarantee. (Reject: batcher-in-engine.)
2. **Dedicated coalescer thread(s); ingress parks on a oneshot, holds no permit.** M0-proven shape;
   decouples batch formation from tokio. (Defer: leader-election among `spawn_blocking` tasks.)
3. **Model 1 (synchronous `complete` on the coalescer thread) first; Model 2 (detached pipeline)
   only if a gate demands it.** Model 1 alone amortizes the host-submit floor (one submit per
   batch); the per-batch `cuStreamSynchronize` blocks only the coalescer thread, not tokio or the
   connection tasks. Model 2's prior art (the 2026-06-12 detached-worker / pending-completion
   reports) was **neutral-to-worse** in M0 — earn it with evidence.
4. **Reuse `gpu_db_batching::DualTriggerBatcher`** for count+time triggering — don't re-derive M0's
   ad-hoc backlog loop.
5. **Reuse the existing engine batch API + kernel unchanged; widen only route-class acceptance.**
   Minimal CUDA risk; slicing / async-complete / drain are already proven.
6. **Strictly additive routing:** only batchable, resident, valid-generation equality point-lookups
   are batched; everything else (COUNT, aggregates, ranges, writes, CPU fallback) keeps today's
   per-query `execute_on_shared_engine` path. Bounded blast radius.
7. **One read-lock per batch + per-job generation guard** for snapshot consistency — leans on the
   existing writer/reader mutual exclusion on `RwLock<Engine>` and the `SnapshotCell` `Arc`
   reclamation rather than reintroducing M0's `Condvar` quiesce barrier.

### Completion model

- **Model 1 (baseline):** coalescer thread calls `complete_*` itself; a small pool (2–4 threads
  per route class) overlaps submit of batch N+1 with completion of batch N across threads.
- **Model 2 (only if gated in):** split into a submitter (submit_* only, pushes the `Send`
  submission to a bounded pending queue) + a completion worker. Defer — M0 found it neutral-to-worse.

## Snapshot consistency

`SnapshotCell` (`crates/snapshot`) + `ResidentDeviceMemoryMap` (`engine:6041`, `get()` returns an
owned `Arc`): a held generation's device memory survives concurrent `invalidate`/`remove`.
Invalidation happens only under the **write lock** (`invalidate_relational_residency_for_commit`,
`engine:9295`), mutually exclusive with the batch's read lock — so no writer interleaves a batch.
The engine's per-job `snapshot_generation` check (`:15563`) rejects any stale item (test `:29129`).
**Rule:** run a batch's submit+complete inside one read-lock acquisition.

## Correctness risks

- **Per-query slicing** — kernel tags `needle_index`; engine scatters into `rows_by_select`
  (`:15880`); proven (`:29044`). Façade must preserve request↔job order + dedup-map.
- **Multi-match ordering (the most likely surprise)** — equal_any does NOT sort matched rows by
  `row_index` (unlike the `row_indices` gather, which needed `sort_unstable()` — §8 thread-2(b),
  `4b750a94`). For a unique-key point-lookup (≤1 match) this is moot; for a non-unique filter
  returning >1 row, multi-row order is GPU-atomic-append. **Confirm before promoting whether any
  batched route can return >1 row/needle and whether parity requires stable order.**
- **Error fanout** — a `complete_*` failure MUST reach EVERY waiter in the batch (no hung
  connection). Cancel/disconnect-while-parked: dropped receiver → coalescer `send` fails
  harmlessly; the batch still completes.
- **Mixed shapes in one batch** — classifier routes only identical-key items together; engine
  `submit_*` bails to per-item if keys differ (`:15695`) — misclassification degrades to a slow
  path, never wrong results.

## Latency / throughput

The knob is `DualTriggerBatcher`'s `max_wait`. Flush on `max_items` (high rate never waits for the
timer) AND `max_wait` (low rate never starves). At c1 batches are size-1 → no regression. Tune
`max_items ∈ {8,16,32,64}`, `max_wait ∈ {10,25,50,100} µs` at the knee of the qps/p99.9 curve.
The harness (`crates/server/examples/p1_m4_concurrent_dispatch_load.rs`) reports p50..p99.9 per
connection count — the tail cost is directly measurable at each gate.

## Routes

- **Batch first:** `int4_equality_projection` (single-column point-lookup), then
  `int4_equality_multi_column_projection`. **Then (Stage 4):** `int4_equality_mixed_column_projection`
  (text batch kernel `:4115` exists).
- **Stays unbatched:** `COUNT(*)`/`row_count` (no needle), aggregates, grouped/distinct/ordered,
  ranges, partitioned routes, writes, DDL, CPU fallback — all keep today's path.

## Staged plan (each: implement → adversarial audit → benchmark A/B vs HEAD → commit)

- **Stage 0** — make the batch entry points `&self`-clean and façade-reachable (no behavior
  change). Gate: compile + unit.
- **Stage 1** — façade batcher core: single coalescer, Model 1, `int4_equality_projection`;
  `enqueue(select, needle) -> oneshot`; classify-or-fallback entry; minimal server-ingress wiring
  behind an A/B flag. Audit: read-lock-per-batch, error-fanout completeness, generation-stale
  handling, dropped-receiver, shutdown drain. **Gate 1 (the important one):** GPU parity tests +
  an end-to-end concurrency A/B (batching ON vs OFF) — **re-measure the host-submit floor** and
  confirm c-high qps rises above the per-call plateau without c1 regressing and with p99.9 bounded.
- **Stage 2** — productionize + tune the ingress wiring; sweep `max_items`/`max_wait`.
- **Stage 3** — coalescer pool + per-route-class queues (overlap submit/complete across threads).
- **Stage 4** — add the mixed int4/text route class.
- **Stage 5 (optional, gated)** — Model 2 (detached pipeline) and/or adaptive `max_wait`, only if
  a gate shows the per-batch sync is the residual ceiling.

## Biggest unknowns

- **The achievable speedup is unmeasured at this branch** — Stage 1's gate must re-measure the
  floor and confirm batching moves the c-high plateau BEFORE investing in Stages 3–5.
- **Tail-latency vs throughput** — `max_wait` tuning; measured via the harness's p99.9.
- **Multi-match ordering** (above) — the one likely correctness surprise.

## Critical files

- `crates/facade/src/lib.rs` — `SharedEngine` + `execute_on_shared_engine` (`:282`, `:321`):
  where the batcher + classify-or-fallback entry are added.
- `crates/server/src/lib.rs` — `run_async_query_loop` (`:457`/`:464`): the per-query
  `spawn_blocking` to replace with batch-enqueue + oneshot-await for batchable selects.
- `crates/batching/src/lib.rs` — `DualTriggerBatcher`: the batcher core to reuse.
- `crates/engine/src/lib.rs` — batch API (`:15550`/`:15616`/`:15446`) + residency/invalidation
  (`:6041`/`:9295`).
- `crates/execution/src/lib.rs` — batched kernel + async submit/complete substrate
  (`:3641`/`:1425`/`:1623`/`:9943`) — no changes expected early; read for the lifetime/drain contract.
- Reference (not edited): `crates/engine/examples/p8_engine_pgwire_benchmark_endpoint.rs` (M0
  owner loop) and `crates/server/examples/p1_m4_concurrent_dispatch_load.rs` (the p50..p99.9 A/B harness).
