# P1-M3 step 4 — Concurrent-Read Scaling (the gate-2 payoff, measured)

Status: closed (first real concurrency benchmark; redirects the GPU latency win)
Date: 2026-06-14
Plan: `docs/roadmap/prototype-to-production-plan.md` §5.7, Phase 1, §7
Branch: `phase0-m1-engine-facade`

## What was measured

A controlled engine-level A/B of the gate-2 (`&self`) read path: the **same**
`Arc<Engine>`, the **same** resident query (`SELECT COUNT(*) FROM order_line`, 512
rows), executed two ways across a concurrency sweep —

- **serialized:** every call holds a shared lock for its whole duration (the
  owner-thread / serialized-access model M0 runs on), vs.
- **concurrent:** every call runs `execute_relational_select(&self)` with no lock (the
  new reader path).

The only difference is the lock, so the gap is exactly the serialization (queue-wait)
the P1-M3 substrate removes. Run on RTX PRO 6000 Blackwell, median of 5 reps × 100
ops/thread. Example: `crates/engine/examples/p1_m3_concurrent_read_scaling.rs`.
Artifacts: `target/2026-06-14-p1-m3-step4-concurrent-read-scaling/{gpu-resident,cpu}.txt`.

## Result — two very different stories

### CPU read path (residency off) — the substrate works: ~40× concurrent throughput

| conc | serial qps | concurrent qps | qps speedup | concurrent p50 µs |
|---:|---:|---:|---:|---:|
| 1 | 4368 | 4333 | 0.99× | 214 |
| 8 | 2470 | 27136 | 11.0× | 264 |
| 16 | 2399 | 49273 | 20.5× | 267 |
| 32 | 2710 | 79308 | 29.3× | 274 |
| 64 | 2443 | **97651** | **40.0×** | 449 |

Concurrent `&self` reads scale to **~40× the serialized throughput at c64** with p50
essentially flat (214→449µs), while serialized throughput plateaus (~2.4k qps, one at
a time). **This is the gate-2 payoff:** removing the `&mut self` bottleneck lets reads
parallelize on the CPU path. The P1-M3 substrate (steps 1–3) does exactly what it was
built to do.

### GPU resident path (residency on) — concurrency *regresses*

| conc | serial p50 µs | concurrent p50 µs | concurrent qps | qps speedup |
|---:|---:|---:|---:|---:|
| 1 | 92 | 92 | 10372 | 0.98× |
| 8 | 196 | 1519 | 4272 | 0.74× |
| 32 | 214 | 5292 | 3908 | 0.83× |
| 64 | 215 | **11466** | 3478 | **0.76×** |

On the GPU resident route, concurrent reads are **slower** than serialized (p50 blows
up 92µs→11.5ms at c64; throughput *drops*). A single GPU read is faster than CPU
(92µs vs 214µs), but the resident kernel path does **not** tolerate concurrency.

## Why the GPU path regresses (the load-bearing finding)

The resident COUNT route, per call, does `cuModuleLoadData` + `cuModuleUnload` (loads
the PTX **every launch**), on a **per-allocation CUDA context** with the **default
stream**. Under N concurrent readers that means N concurrent module loads + launches
contending on one context/stream — the driver serializes and thrashes. The CPU-side
serialization (the M0 owner-thread queue-wait) is gone, but the **GPU execution's
shallowness is now the dominant bottleneck**. This is precisely the "shallow GPU" the
plan calls out (§1.2: single-thread kernels, no streams, module re-loaded per launch,
per-table context) and is the work of **Phase 2** (real parallel GPU execution: cache
modules/functions, stream pool, async copies) and **§9.3** (one shared primary context
instead of per-allocation contexts).

## What this means for the plan

- **Confirmed (§1.2/§7):** removing CPU-side serialization is a real, large unlock —
  ~40× on the CPU read path. The concurrency substrate is necessary and it works.
- **Refined:** the plan's expectation that the gate-2 flip would, by itself, collapse
  the *GPU* read latency under load is **not borne out** — once CPU serialization is
  removed, the GPU resident path is bottlenecked by per-launch module loads + a single
  context/stream and *regresses* under concurrency. The GPU latency-under-load win is
  **gated on Phase 2 / §9.3**, not on the concurrency substrate alone.
- **The pgwire median-of-N re-run is therefore not the right next benchmark yet.** The
  engine-backed server still funnels reads through its owner thread *and* uses the GPU
  resident route; wiring it for concurrent dispatch would expose the same GPU thrash,
  not a latency win. Server reader/writer dispatch should land **after** the GPU path
  is made concurrency-friendly (cached modules + stream pool + shared context). This
  engine-level A/B is the honest measurement until then.

## Honest scope

- Engine-level microbenchmark (no pgwire), reads-only over one frozen resident
  generation — the same safe scenario gate 2 proved; no concurrent writer.
- The "serialized" arm holds the lock for the whole call (a strict stand-in for
  owner-thread serialization), and `std::sync::Mutex` is unfair, so serial p50 is
  noisy and understated at high c — but the *throughput* contrast (plateau vs 40×) and
  the GPU p50 blow-up are unambiguous and reproducible across reps.
- No claim of meeting the `DESIGN.md §1.1` latency/throughput targets — this isolates
  the substrate's effect and locates the next bottleneck.

## Next

- **Phase 2 / §9.3 (now the gating work for the GPU win):** cache the resident kernel
  module/function (stop per-launch `cuModuleLoadData`), add a CUDA stream pool, and
  move to one shared primary context — then re-run this A/B and expect the GPU resident
  path to scale like the CPU path does.
- The two latent hazards from the 3c audit (`partition_device_memory` → SnapshotCell;
  dispatcher bind/launch TOCTOU) remain owed before concurrent *writes*.
