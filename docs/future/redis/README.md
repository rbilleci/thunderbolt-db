# GPU-resident Redis — a high-throughput KV tier on the persistent wave kernel

> **Status: FUTURE / REFERENCE ONLY.** This directory holds an **archived snapshot** of the persistent
> "wave" kernel + engine (`wave.rs`, recovered verbatim from git `c45907d4^`, the commit just before the wave
> read path was retired) plus the pitch for a Redis-protocol KV tier built on it. **Nothing here is wired
> into the live engine.** `wave.rs` is reference code, **not** a workspace member — do not add it to `crates/`.
> The wave read path was retired *for SQL* ("lpb over wave", DECISIONS) because SQL's production path is
> single-coalescer and mixed-workload; this captures the engine for a future, better-fit use case.

## TL;DR
A GPU-resident, **Redis-protocol (RESP) high-throughput key-value tier**, powered by the persistent wave
kernel: one always-resident GPU kernel draining a lock-free **multi-producer** ring, serving pipelined point
GET/MGET (and, with the write path, SET/INCR/DEL) at **tens of millions of ops/s** under massive client
concurrency. The play is **consolidation + throughput** — one GPU standing in for a throughput-bound Redis
cluster — **not** beating CPU Redis on a single isolated-op latency.

## Why the wave is a good fit for Redis (when it was a bad fit for SQL)
The wave was retired for SQL for two reasons, **both of which dissolve for Redis**, and ADR-009 already named
the wave the "*homogeneous-wave throughput engine* — not sub-µs for arbitrary transactions." Redis is a
homogeneous point-op workload — its actual design target.

1. **SQL production is single-coalescer; Redis is inherently multi-producer.** The wave's *only* genuine edge
   is the multi-producer continuous-fill regime (N producers enqueue into a lock-free ring, one resident
   kernel drains, **no per-request launch**). SQL point reads funnel through one coalescer, so lpb matched it.
   Redis is the opposite by nature — thousands of connections + pipelining = exactly that continuous fill.
2. **SQL is mixed; Redis is homogeneous on a dedicated GPU.** The wave's permanent SM-coexistence tax (~60%
   loss to concurrent scans at 8 reserved SMs) hurt because SQL runs scans/aggregations/joins beside point
   reads. A Redis cache is all point ops — the resident kernel can **own all the SMs**, nothing to starve.
3. **Multi-shape coexistence (the wave's biggest unbuilt blocker) is irrelevant.** A Redis keyspace is *one*
   logical index = one shape, so the at-most-one-kernel limit is fine — no multi-shape thrash, no need for the
   shared-multi-ring redesign (for the in-VRAM case).
4. **Per-request latency favors the wave over lpb** (no launch: ~10µs vs lpb's ~25µs launch-per-batch; the
   measured P2 result was batch-1 2.5× lpb, lower latency at every batch).

## How it works (architecture)
```
RESP clients (pipelined) ─┐
RESP clients (pipelined) ─┤  N producers enqueue                ┌── one always-resident GPU kernel
RESP clients (pipelined) ─┼─► lock-free multi-producer ring ───►│   (gpu_db_wave_read_dataplane):
        ...               ─┘   (host-pinned / device-mapped)    │   grid-stride drain, probe keyspace
                                                                 │   index, gather value, per-slot status
   RESP-encode  ◄── needle-ordered harvest ◄── device result ring + bulk DtoH  ◄── depth-K pipelined
```
- **RESP front-end** (trivial protocol) atop the engine's facade (already multi-protocol-shaped — see the
  `phase0-m1-engine-facade` work).
- **Client pipelines → waves**: a pipeline of GETs is a coalesced wave enqueued into the ring — the engine's
  batched-wave model consumes exactly this.
- **The persistent kernel** = the recovered `wave.rs`: grid-stride claim (no CAS), per-slot status ring for
  depth-K pipelining (harvest any order), device result ring + bulk `cuMemcpyDtoHAsync`, a thread-0 coordinator
  (mirrors host doorbell/head to device so workers don't PCIe-poll), and a crash-safe watchdog/backstop.
- **GET** = point lookup (dense single-pass compaction, byte-identical to the lpb index probe). **SET/INCR/DEL**
  = writes (need the write path; INCR maps cleanly to on-device checked atomics, ADR-011).

## Why it can be competitive
- **CPU Redis is single-threaded per shard** (~100k–1M ops/s/core); scaling = sharding across many cores/nodes.
  The wave does point lookups at **tens of millions/s on one GPU** — measured at the GPU level: ~30M+ drain,
  depth-K pipelining 1.4–2× over single-flight, bare probes 31–45M. So one GPU ≈ a sizable Redis cluster on
  throughput, with far less hardware to operate.
- **Latency:** per-op is ~a host↔GPU round-trip — it **loses to CPU Redis on a single isolated GET** (~µs from
  cache) but is competitive-to-better under pipelining. Sell it as **aggregate throughput + consolidation**,
  not single-op latency.

## What must be done (honest gap list)
1. **Build the multi-producer lock-free enqueue (the core new work).** The *wired* wave was single-flight
   (per-engine `Mutex`); the multi-producer ring — the whole point for Redis — **was never lifted into the
   engine** (it was R2.2c "gate-2", deferred because SQL didn't need it). `wave.rs` has `submit_columnar` /
   `harvest_columnar` + the per-slot status gate; the MPSC ring + concurrent harvest is the piece to add.
2. **Resurrect + de-rot `wave.rs`.** It compiled and passed two independent audits at `c45907d4^`, but the
   execution-crate result-path APIs moved after it (columnar drain, dense kernel, the `index_probe_enabled`
   rename). Re-integrate against current APIs (as reference, then a real crate module).
3. **String/bytes keyspace index on the GPU.** The optimized index is **int4-only** (`(key<<32)|(row+1)`).
   Redis keys are byte strings → hash to 64-bit + store the full key for collision compare. New index type.
4. **The write path (R3).** SET/INCR/DEL/EXPIRE need durable writes + concurrency control. Lock-free index
   inserts are proven fast (write-probe-1, tens of billions/s); commit/durability/CC are unbuilt. INCR →
   on-device checked atomics (ADR-011).
5. **RESP front-end + pipelining→wave-batch mapping** in the facade.
6. **TTL / per-key expiry + eviction** (LRU/LFU): new machinery (TTL column + lazy-expire-on-read + background
   sweep). STRATA eviction is shard-level, not per-key.
7. **Over-VRAM keyspace.** In-VRAM = one persistent kernel (fine). A sharded/over-VRAM **random-access**
   keyspace needs a **shared multi-ring kernel** (one kernel draining K rings — the option that survived the
   R2.2c gate-1 analysis) **or a hot-key residency cache.** NOTE: the streaming/out-of-core executor (ADR-012)
   folds over shards in plan order and does **not** help random point lookups — wrong tool here.
8. **Scope the value model.** Strings (KV) only. Hashes / lists / sets / sorted sets / streams are **explicit
   non-goals** — rich per-key structures don't map to the columnar batched GPU model and would land host-side
   against the GPU-native charter (ADR-006/007).
9. **Re-validate coexistence assumptions for the dedicated-GPU homogeneous case** (they should be favorable —
   one persistent kernel owning all SMs — but measure).

## Non-goals (be explicit)
- Not a low-single-op-latency store (CPU Redis wins there). It's a **throughput KV tier**.
- Not full Redis. KV subset only; data structures, pub/sub, MULTI/EXEC, Lua scripting, streams are out of
  scope (host-side / no GPU mapping).

## Hard-won lessons baked into `wave.rs` (a future implementer MUST keep these)
These were discovered the hard way during the wave campaign; re-learning them costs days and can zombie the GPU:
- **`cuMemAlloc` while the persistent kernel is live can freeze the context** (the original R2.2 freeze). Use
  pre-allocated rings + the device-buffer pool; **never** fresh driver-serialized allocs during the kernel's
  life.
- **At-most-one persistent kernel** — two never-exiting kernels **deadlock at teardown** (the doorbell-exit
  fails when a second wave kernel is resident; R2.2c gate-1, vindicated even for 1-SM kernels). Fine for one
  keyspace; a sharded keyspace needs the **shared multi-ring** kernel, not multiple persistent kernels.
- **The watchdog/backstop is load-bearing.** A hung persistent kernel zombies the context, and `--gpu-reset`
  is forbidden on a shared box. The host petter heartbeat + thread-0 stale detection (armed on first pet,
  fenced petter) lets the kernel self-terminate on host death.
- **Memory ordering is real:** per-slot status uses `st.release.sys`; cross-engine (DtoH copy engine reading
  device body) relies on `.sys`-release + `cuStreamSynchronize`. ASCII-only PTX (driver JIT error 218 on
  non-ASCII; `ptxas -arch=sm_70` check before launch).
- **Validate non-vacuously.** An independent audit once caught a harvest-gate underflow that FAKED 348M
  ops/s (a no-op false-pass). Any throughput claim needs a byte-identical stress gate, not a counter that can
  silently no-op.

## Recovered artifacts + how to get the rest
- **`wave.rs`** (in this folder) — the persistent kernel (`gpu_db_wave_read_dataplane` PTX) + the
  `WaveReadEngine` (drain, per-slot status ring, device result ring, thread-0 coordinator, watchdog) + its
  in-file oracle / depth-K / throughput / offered-rate tests. Recovered from `c45907d4^`.
- **Supporting probes/benchmarks** (the evidence) live at the same commit; recover with
  `git show c45907d4^:<path> > <dest>`:
  - `crates/execution/examples/wave_dataplane_probe.rs` — the data-plane throughput probe.
  - `crates/execution/examples/wave_coexist_probe.rs`, `wave_multikernel_probe.rs` — the coexistence /
    at-most-one findings.
  - `crates/execution/examples/wave_freeze_probe.rs` — the `cuMemAlloc`-freeze isolation.
  - `crates/engine/examples/r2_wave_engine_ab.rs` — the engine-level offered-rate A/B (the ship-decision
    instrument).
  - `crates/engine/examples/p8_persistent_pgwire_concurrency_runner.rs` — persistent-client concurrency runner.
- **Decision history:** DECISIONS ADR-009 (wave engine framing), the R2.2 / R2.2b / R2.2c entries (the build +
  the lpb-over-wave retirement), and ADR-008 (SM-coexistence: ~1-SM sidecar OR full replacement).

## Bottom line
The wave retirement wasn't "the wave is bad" — it was "the wave is wrong for *mixed, single-coalescer SQL*."
A homogeneous, massively-concurrent **Redis-style KV is the strongest case for resurrecting it**: build the
multi-producer ring it always implied, add a string-key index and the write path, and position it as a
high-throughput KV tier. This folder is the starting point.
