# DECISIONS — Decision Ledger (ADRs)

Append-only record of decisions that are expensive to reverse. Newest first. Each entry: status, context,
decision, consequences. Supersession is recorded, never silently rewritten. The resulting *rules* live in
[CHARTER.md](CHARTER.md); the *design* in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## ADR-011 — Checked on-device integer arithmetic (no wrap, no fallback)
- **Status:** Accepted (decision 2026-06-18; recorded in the ledger 2026-06-26)
- **Context:** Integer overflow on the GPU must match PostgreSQL semantics (error, not silent wrap) and must never
  escape to a CPU path.
- **Decision:** int4/int8 `+ - *` are **range-checked on-device** and raise PG `integer out of range` /
  `numeric field overflow` via a shared device overflow flag — **never silently wrap, never CPU-fallback**. The
  `ResidentExpr` interpreter evaluates every arithmetic sub-expr over **all** rows before combining masks, so a
  query errors if any row overflows in any conjunct (**stricter than PG**).
- **Consequences:** A correctness contract for the executor (ARCHITECTURE §8). The over-all-rows property surfaces
  the known **gather-then-evaluate** divergence (a WHERE-filtered overflow row still errors) — tracked as charter
  debt in STATUS.

## ADR-010 — STRATA: GPU-resident shards + auto-admission on commit
- **Status:** Accepted (2026-06-26)
- **Context:** No production producer of GPU residency exists (residency is operator-triggered; a committed table
  is non-resident by default and reads run host-side). The host read path therefore stays live, blocking the
  "host out of the data path" goal.
- **Decision:** A table is laid down as **1..N GPU shards** (the physical residency unit, L2 — distinct from a
  future SQL `PARTITION BY`, L1; and from a shard's columnar **sections**, L3). A **commit-triggered admission
  producer** (post-durability, best-effort, via the `&self`+held-catalog-guard seam) makes committed tables
  resident. Reads = push-down to shards + cross-shard combine. Explicit residency/admission owns placement.
- **Consequences:** Renames `RelationalResidentPartition → RelationalResidentShard` (**landed: S-A, 2026-06-27,
  suite 729/0**). The commit-triggered admission producer **landed: S-B, 2026-06-27** (default-off
  `auto_admit_on_commit`; 730/0). Unblocks host-read-path deletion once admission is the default (S-F). Over-VRAM
  tables spill across shards (needs cross-shard combine — not yet built).

## ADR-009 — Deterministic batched OLTP execution model
- **Status:** Accepted (2026-06-26); incorporates the `feedback.md` review corrections.
- **Context:** OLTP parallelism on a GPU comes from running many transactions at once, not from inside one
  transaction. Prior GPU-OLTP attempts died on per-op overhead and lock-based concurrency control.
- **Decision:** (1) **Persistent-kernel wave engine** draining a host-pinned lock-free ring — framed as the
  *homogeneous-wave throughput engine*, not sub-µs for arbitrary transactions. (2) **Concurrency control =
  deterministic spine (Calvin-style) + MV dependency-graph execution (BOHM/PWV)**, *not* OCC (OCC under a total
  order re-introduces aborts). The order *is* the replication log; non-deterministic inputs are host-materialized
  into the ordered intent. (3) **Coherent memory is a fast-path target, not a requirement**; explicit placement
  (STRATA) owns the tail, never hardware demand-paging. (4) **Group-commit + host-written WAL** (GDS reserved for
  checkpoints, not WAL). (5) GPU indexes; resident **layout decided by measurement** (leaning PAX), not assumed.
- **Consequences:** Re-prioritizes STRATA toward the wave engine + deterministic CC + index/point path ahead of
  cross-shard analytical combine. Detail: ARCHITECTURE §OLTP execution.

## ADR-008 — Workload bet = high-throughput OLTP on GPU
- **Status:** Accepted (user, 2026-06-26)
- **Context:** The architecture (sharded resident columns, push-down + combine, agg/sort kernels) is analytical-
  shaped; GPU economics classically favor analytics; OLTP point lookups stress PCIe/launch overhead.
- **Decision:** The target workload **is OLTP**, betting AI-driven GPU advances make GPU OLTP outpace CPU engines.
  Obstacles are in scope to fix. Optimized target = predeclarable transaction waves (ADR-009); interactive
  multi-statement is a supported slow class.
- **Consequences:** Benchmark mandate (open-loop p99 vs tuned Postgres) becomes the gate that proves/kills the bet.
- **Refinement (2026-06-27):** Success = **same ballpark on today's hardware**, NOT beating the CPU today. Win
  condition: (a) within an order of magnitude of a tuned CPU engine now, and (b) the residual gap is GPU-architectural
  (amortizable launch / parallelism / bandwidth) so it narrows as hardware advances. Host-side serial overhead does
  not count against the bet — it is a fix (see CHARTER "Success bar").
- **First measurement (2026-06-27, RTX PRO 6000; `engine/examples/oltp_auto_admit_ab` +
  `facade/examples/oltp_batched_read_scaling`), point-lookup `WHERE id=?`:** (1) one GPU read is a **fixed ~72µs**
  (launch + context-set + stream-sync + host round-trip — flat across a 100× table-size sweep, so overhead not compute)
  vs ~3µs host. (2) Batching (`PointLookupBatcher`) amortizes that **13.7×** but **plateaus at ~68k ops/s, 11× under
  the CPU's ~770k**, and does NOT scale with concurrency (flat throughput, p50 grows linearly). Root cause (queueing
  signature + Little's law + code at `point_lookup_batcher.rs:395`): **~15µs/item of serial host-side work in the single
  coalescer thread** (per-item `route_id` String build + `HashMap` grouping + per-item oneshot) — **not GPU compute.**
  The gap is host-serial → in scope to fix, consistent with the bet. Path to ballpark: remove the per-item host cost →
  the persistent-kernel wave engine (ADR-009) with on-GPU result slots. **S-F (flip auto-admit default) stays OFF** —
  resident GPU point reads lose to host today (DECISIONS ADR-010 / PLAN §2).
- **Tier-1 result (2026-06-27):** removed the per-request plan/bind — a needle-invariant resident-read **template**
  prepared once per shape, reused across all needles (`prepare_relational_retained_read_template` /
  `submit_relational_retained_template_point_lookups`; batcher groups by a cheap shape key). **Batched point-read
  throughput 68k → 156k ops/s (2.3× @ 1024 threads), now scales with concurrency (was flat); CPU gap 11× → ~4.5×.**
  Behavior-preserving (731/0), HAZARD clean. Confirms the diagnosis (the wall WAS host-serial per-item work). Residual
  ~6µs/item = result materialization + oneshot distribution (still single-coalescer). Next: Tier 2 (parallelize the
  coalescer) and/or the wave engine (ADR-009) — the template is the ingress the wave engine reuses.
  - **Audit finding adopted:** the independent adversarial audit caught that the batcher admitted the **mixed int4+text**
    shape (`int4_equality_mixed_column_projection`) but the int4 `equal_any` kernel cannot project text — and the
    text-capable general executor errors **CUDA 201 (invalid context)** on the coalescer thread (a *pre-existing latent*
    bug — that path was never exercised; the prior engine-level mixed test runs on the main thread). Fix: `classify`
    now routes mixed point lookups to the **per-query path** (correct; 18 engine mixed tests + a new facade regression
    test with NULL data lock it in). Batched mixed-column lookups are a follow-up (need the coalescer-thread context
    fix, or the wave engine). The batcher is now all-int4-only by construction.
- **Wave-engine data-plane proof (2026-06-27) — the bet validated for point reads.** Tier-1 left the read bottleneck
  HOST-serial (single coalescer ~167k cap, does NOT scale with HW). The wave engine (ADR-009) moves it host→GPU. Built
  + measured as isolated standalone probes (`crates/execution/examples/wave_{lifecycle,dataplane,index}_probe.rs`;
  own libcuda + ctx, can't touch the engine; persistent kernel always terminates via doorbell + `%globaltimer`
  backstop). **1a:** clean ~3.5µs persistent-kernel doorbell exit (the `--gpu-reset`-box risk de-risked; 9/9 across 3
  processes). **1b:** parallel data plane — lock-free claim + on-GPU scan+gather + packed atomic result; host reads
  slots (no per-request host materialization) → **the host-serial cap is GONE.** Independently audited (REAL across
  20 runs; correctness SOUND; `membar.sys` ordering fixed). Full-scan is O(rows): 4k:10M → 1M-row table:485k. **1c:**
  a GPU hash index removes the scan → **~10.5M point lookups/s, FLAT across 1M/4M/16M-row tables (O(1)) — ~13.6× the
  CPU's 770k, ~67× the batcher, independent of table size.** Now atomic-ceiling-bound (host-mapped claim/`completed`
  atomics → device memory raises it). **Conclusion: the GPU does millions of point lookups/s at realistic scale; the
  residual bottleneck is GPU-architectural (scales with hardware) — the bet holds, and not just "ballpark", >10× the
  CPU at the data-plane level.** *Not yet integrated into the engine; the batcher remains the production default.*
  Honest caveats (1c independently audited — no P0, number REAL + index NON-VACUOUS by sabotage tests): bare data
  plane (no slot→wire mapping / facade — the ~13.6× is data-plane-only, not end-to-end); static host-built index (real
  OLTP needs **concurrent index maintenance on writes** — ADR-009 lock-free CAS inserts at epoch boundaries — the key
  write-path gap); **synthetic keys are BEST-CASE** (sequential + Fibonacci hash → equidistribution caps clusters at
  length 2, avg ~1.18 probes; arbitrary OLTP key/insertion orders → longer chains, still O(1) but no depth-2 ceiling);
  the ~10M ceiling is the **host-mapped PCIe atomic** (256 threads already saturate it → device memory next); single-GPU.
  Read ceiling later pushed to ~30M (device-mem atomics, 1d-i) then ~45–53M (batched claiming, 1d-ii); slot→wire mapping
  measured CHEAP (~200M–1.1B rows/s, parallel). **The read half of the bet is settled (~tens of M/s, O(1), ≫ CPU).**
- **Write-path probe (2026-06-27) — index maintenance is not the obstacle.** The novel, historically-hard piece of GPU
  writes is concurrent **lock-free index maintenance**. Measured (`wave_index_insert_probe.rs`, all inserts verified):
  many threads `atom.cas.b64`-insert (key,row) into a shared open-addressing table at **~tens of BILLIONS of inserts/s**
  (1M ~29G/s wall-clock / ~77G/s minus launch floor; 16M L2-spill 4.2G/s). **So GPU concurrent index maintenance is NOT
  a bottleneck.** The remaining write constraints — **durability (WAL fsync, group commit)** and **deterministic CC** —
  are host-I/O + coordination problems CPU OLTP engines face too, so the GPU isn't disadvantaged. Caveats: low contention
  (sequential keys + Fibonacci spread); raw insert only (no commit/durability/MVCC/CC); synthetic keys. Next write
  probes: contended inserts; the commit/durability floor; deterministic CC.
- **R1 end-to-end measurement (2026-06-27, RTX PRO 6000; `engine/examples/r1_wave_index_ab`) — the index win, through
  the engine.** R1 wired the GPU hash-index probe into the resident int4 unique-key point-lookup route behind default-OFF
  `wave_engine_enabled` (`submit_resident_int4_equal_any_payload`); this measures the SAME swap on the production
  retained-read **template** path (results materialized to `Vec<SqlValue>`), flag OFF (full-scan `equal_any`, O(rows)/batch)
  vs ON (index probe, O(1)/needle) — identical distinct needles, **ON==OFF byte-identical asserted** each size. Batched
  lookups/s (batch=256, single coalescer): **1M 1.69M→2.34M (1.39×), 4M 733k→1.90M (2.59×), 16M 242k→1.69M (6.99×)**.
  Table grew 16× → **scan fell 7.0× (≈O(rows)), index fell only 1.38× (≈flat ⇒ O(1))**; per-batch scan latency
  150→346→**1057µs** vs index flat **107–149µs**. **Crossover ≈1M rows:** below it the fixed ~110µs host+launch floor
  dominates and the index is a wash (~1.0× at 64k–256k). So the per-batch O(rows)→O(1) win is REAL end-to-end and GROWS
  with table size — but is table-size-dependent, so the batcher stays the production default and any flip should be
  size-aware (or land with R2). The index's own ~1.7M/s end-to-end ≪ the 1c data-plane ~10.5M: the residual gap is the
  same per-batch host+launch floor R2 (persistent kernel + ring) removes — consistent with the bet.
  **Verified (independent re-run, 2026-06-27):** ON==OFF byte-identity held at every size; the scan curve reproduced
  exactly (1M→4M→16M scan = 1.68M→731k→247k lookups/s, the load-bearing O(rows) falloff). The index is FLAT at ~2.3M
  across all sizes — the recorded 16M index 1.69M was a slow sample (re-run 2.31M ⇒ 16M speedup 9.34×, index falloff
  ≈1.0× not 1.38×), so the O(1) claim holds even more cleanly than first measured. Conclusion (real, size-dependent win;
  batcher stays default; flip is size-aware or lands with R2) unchanged.
- **R2 SM-coexistence gate (2026-06-27, RTX PRO 6000 = 188 SMs; `execution/examples/wave_coexist_probe`) — the recon's #1
  unknown, resolved.** R2 wants ONE always-resident persistent wave kernel on the SAME context as the engine's
  launch-per-batch kernels. The probe runs the proven 1a persistent kernel (reserving `pblocks` SMs) concurrently with a
  storm of full-grid int4 scan launches on a second non-blocking stream, sweeping `pblocks ∈ {1,8,32}` in two poll modes
  (busy-spin vs ~1µs `%globaltimer` backoff). **VIABLE: zero deadlock/starvation — the persistent heartbeat keeps
  advancing through every scan storm and the kernel exits cleanly on the doorbell (no backstop), at all footprints**, so
  a wave kernel CAN share the engine's context. **But the SM-reservation cost is steeply non-linear (reproduced 2×):
  1 SM = ~2% (97–99% of baseline), 8 SMs (4.3% of chip) = ~60% LOST (≈40% of baseline), 32 SMs (17%) = ~87% lost (~13%).**
  Busy-spin ≈ gentle (the persistent kernel's host-mapped doorbell poll is itself ~µs-throttled over PCIe, so neither is
  truly high-frequency) ⇒ the cost is **SM co-residency/scheduling, not the kernel's polling memory traffic** (exact
  mechanism — scheduler vs L2 pollution — not isolated; the design rule holds either way). Baseline scan ≈1.04 Trows/s
  (128MB column, L2-resident). **Design rule for ADR-009 integration:** the wave kernel must be a **~1-SM-minimal sidecar
  OR fully REPLACE the per-batch launch path** (its actual intent) — NOT a wide always-resident data-plane co-resident
  with heavy engine kernels. (Caveat: the scan storm is latency-bound serial launches, faithful to today's launch-per-
  batch engine; `cuMemHostGetDevicePointer` already in the probe FFI; the `all_done` ordering audit is still owed before
  any engine lift.)
- **R2 `all_done` ordering audit (2026-06-27) — owed gate, now done; DONT lift the `all_done`-only pattern.** Two
  independent adversarial GPU-memory-model auditors examined the 1d-i/1d-ii completion mechanism (worker: `st.volatile`
  result slot → `membar.sys` → `atom.add(completed)` on DEVICE mem; the LAST completer — the one whose RMW makes
  `completed==head` — does `membar.sys` → set host-mapped `all_done`; host polls `all_done`). **They SPLIT:** (A) **GAP /
  sound-only-by-luck** — the last completer learns "done" from its OWN RMW return value and never acquires the other
  workers' stores; single-location coherence on `completed` carries the COUNT, not a happens-before for other threads'
  result stores, so `all_done` alone has no synchronizes-with edge to those slots. (B) **SOUND (~90%)** — `membar.sys`
  is CUMULATIVE, so each worker's fenced slot store propagates with its counter bump through coherence and the last
  completer's 2nd `membar.sys` re-flushes before `all_done`; the HOST (not the last completer) reads the slots so no
  acquire-by-L is needed — BUT B's confidence explicitly hinges on cumulativity surviving a **bare device-scope RMW**
  ("the one assumption worth pinning empirically"). **CONVERGENCE (decision):** both agree (1) `all_done` ALONE is not a
  robust correctness gate, and (2) the host's synchronous **`cuMemcpyDtoH(completed)==requests`, performed BEFORE reading
  any slot, IS the sound acquire — the proven-sound 1b pattern** (host acquires the counter each worker released into via
  `membar.sys`+bump). The 1d probes read slots only after that DtoH ⇒ sound *for the DtoH reason, not the `all_done`
  reason* (`all_done` is only a wake hint). **DESIGN RULE for the engine lift:** completion MUST be the host acquiring
  the `completed` counter (read-and-check `==expected`, host-mapped read like 1b OR via DtoH), with each worker's slot
  store released ahead of its bump (`membar.sys`, ideally explicit `.release.sys`/`.acquire.sys` under an sm_70 target to
  drop the cumulativity assumption) — **never lift an `all_done`-only gate, never drop the counter-acquire**. If the
  cumulativity question ever needs settling empirically: remove the DtoH + the per-thread `membar.sys`, stress at max
  threads / small claim-K, poison slots with a sentinel, and watch for `done==0` stale-slot reads after `all_done`.
- **R2.2a wave data plane in-crate + the INTERLEAVED-LAUNCH FREEZE (2026-06-27, `crates/execution/src/wave.rs`).**
  `WaveReadEngine` now does multi-column projection (the R1 index probe's 4-way gather) over a circular lock-free ring,
  returning `CudaI32BatchProjectionRow`s **byte-identical to the R1 index probe** (GPU oracle test, 2 waves + spot,
  green). Claim uses `atom.cas` (bounded, no overshoot) so cumulative `head` works across waves. **BUT a critical
  coexistence discovery gates R2.2 wiring:** when the GPU index-probe oracle is launched BETWEEN two waves, the idle
  persistent kernel FREEZES — wave 2's `claim` stays frozen (no CUDA error, no fault, no backstop; just stops claiming)
  and the wave times out. Reordering so all `submit`s precede any oracle launch makes it pass. So **an interleaved
  kernel launch stalls the idle persistent wave kernel.** This SHARPENS the SM-coexistence verdict: that probe showed a
  persistent kernel coexists with concurrent scans launched on a **non-blocking** stream (heartbeat advanced), but the
  index-probe oracle launches on a **flag-0 (blocking) pooled stream + synchronous NULL-stream memcpy + stream sync** —
  the legacy-default-stream path is the leading suspect. **Implication:** the engine's existing launches use flag-0
  pooled streams, so a co-resident wave kernel would freeze under normal engine traffic → R2.2 must FIRST resolve this
  (candidate fixes: engine launches / wave stream interaction via non-blocking streams or avoiding NULL-stream sync; OR
  the wave kernel REPLACES the launch-per-batch path so they never interleave — the ADR-008 SM-coexistence "replace"
  option). NEXT R2 step = a focused freeze-mechanism probe (blocking vs non-blocking vs NULL-stream-sync interleave),
  NOT blind wiring. Multi-projection correctness itself is settled.
- **R2.2 freeze ROOT CAUSE PINNED (2026-06-27, `execution/examples/wave_freeze_probe`) — it is `cuMemAlloc`/`cuMemFree`,
  NOT the stream type.** The probe runs each candidate interleave op against a fresh non-blocking persistent kernel
  (heartbeat liveness) and times it. Result (persistent grid 4x256, 5s backstop): **ALIVE** for non-blocking launch,
  flag-0/blocking launch, NULL-stream HtoD, NULL-stream DtoH, the blocking+HtoD+sync combo, AND CUDA events — every op
  in microseconds. **FROZEN only for `cuMemAlloc + cuMemFree`, and the op itself blocked 4.96s ~= the backstop** (vs us
  for all others) before the kernel died. So **`cuMemAlloc`/`cuMemFree` are device-synchronizing: they block until ALL
  GPU work drains, including the never-ending persistent kernel — which only ends when its `%globaltimer` backstop fires,
  killing it.** This explains the original symptom end-to-end: the failing data-plane test ran ~50s = ~30s (the first R1
  oracle's COLD `lease_device_buffer` -> `cuMemAlloc` blocking to the 30s backstop, killing the wave kernel) + 20s
  (wave 2 then finding a dead kernel and timing out). The stream type / NULL-stream sync were red herrings. **R2.2 design
  consequence:** a co-resident wave kernel dies the instant the engine does a synchronizing `cuMemAlloc`/`cuMemFree` on
  the shared context. The engine's device-buffer **pool amortizes** this — steady-state leases REUSE pooled buffers (no
  `cuMemAlloc`); it only syncs on COLD pool growth (new bucket / empty pool) or pool-overflow `cuMemFree`. So the
  pragmatic R2.2 path: **pre-warm the device-buffer pool + suppress pool shrink (`cuMemFree`) while a wave kernel is
  resident** (the wave path is itself alloc-free — pre-allocated device-mapped ring), so no synchronizing alloc occurs
  during wave operation. The robust long-term fix is migrating the engine's device allocation to **`cuMemAllocAsync`/
  `cuMemFreeAsync`** (stream-ordered, non-synchronizing). NOTE: "replace the per-batch path" ALONE is insufficient —
  other concurrent engine activity (residency admission, other routes) still `cuMemAlloc`s on the shared context.
- **R2.2 EVIDENCE GATE (2026-06-27, `wave::tests::wave_vs_launch_per_batch_throughput`, release, 1M rows) — the
  persistent wave engine LOSES to launch-per-batch in request-response; wave-in-engine is PARKED, async-alloc fix NOT
  built.** Before investing in the async-alloc refactor or wiring, measured the persistent `WaveReadEngine` vs the
  launch-per-batch R1 index probe directly (same table/index/needles/projection, single-thread serialized batches).
  **Result (lookups/s, wave vs lpb): batch1 8.6k vs 39k (0.22x); batch8 35k vs 309k (0.11x); batch64 55k vs 2.29M
  (0.02x); batch256 60k vs 7.08M (0.01x).** The wave per-submit latency is ~116us (batch1) growing to ~4.3ms (batch256);
  the launch-per-batch call is ~25-36us. So the wave engine is **4.5x-118x SLOWER**, and plateaus at ~60k (below even
  the 156k batcher). **Mechanism:** in a request-response (submit a batch, wait for it) pattern, each wave pays a host<->
  device round-trip (publish `head` -> kernel notices/drains -> host DtoH the `completed` counter) of ~116us, which is
  WORSE than a kernel launch+stream-sync (~25us); the 1024 persistent threads also continuously poll host-mapped memory
  over PCIe, congesting the bus and inflating every wave. A kernel launch that runs-to-completion has a clean, fast
  stream-sync completion and no idle polling. **The probes' ~45M req/s was ONE GIANT continuous-fill wave** (round-trip
  amortized over 200k requests); request-response does not amortize it. **Decision:** do NOT wire the wave engine into
  the read path and do NOT do the async-alloc fix — the launch-per-batch GPU index probe (R1, already shipped: 2.3M
  end-to-end via the template path, ~7M at the execution layer, O(1)) is the point-read winner. R2's premise ("the
  persistent kernel breaks the host+launch floor") does NOT hold for the engine's request-response point-read path: the
  launch path is already GPU-bound, and the persistent kernel's signaling round-trip is a worse floor. The wave engine
  would only win under sustained **continuous-fill high concurrency with async result delivery** (many producers keeping
  the ring full, host reading slots without per-batch waits) — a much larger architectural change, unproven to be
  needed, so PARKED. "Measure first" saved the async-alloc refactor + the wiring. (R2.1/R2.2a code retained as proven,
  audited building blocks; R3 writes + deterministic CC are independent of this.)
- **R2.2 evidence gate CORRECTION (2026-06-27) — the prior verdict's COMPARISON WAS IMPROPER; "wave loses / PARK" is
  RETRACTED, the wave engine was NOT fairly tested.** (Self-correction after review.) Three flaws in the entry above:
  (1) **Wrong regime** — it ran the wave engine in SYNCHRONOUS single-flight (submit one batch, block on a DtoH
  round-trip, repeat), the *worst* mode for a persistent kernel; the wave's thesis (ADR-009) is continuous fill under
  CONCURRENCY (many in flight, no per-batch wait). (2) **Wrong baseline** — it compared against the raw single-threaded
  execution-layer index probe (7-24M/s, which has NO coalescer because it's one thread submitting directly), not the
  thing the wave engine is designed to beat: the **batcher's 156k single-coalescer concurrent cap** (the host-serial
  bottleneck). (3) **A naive, unoptimized wave port** — a large-batch / thread-count sweep
  (`wave_vs_launch_per_batch_throughput`, 128 threads, 1M rows) shows the wave throughput PLATEAUS at **~520k/s FLAT
  across batch 256/4096/65536** — i.e. NOT a per-wave round-trip artifact (that would amortize with batch) but a
  per-needle DRAIN ceiling of ~2us/needle. The proven probe (1d) hit **~45M/s (~22ns/needle)** — so this in-engine port
  is **~85x slower than the data plane it was meant to be**, because it dropped the probe's optimizations (1d-ii batched
  claiming, 1d-i device-memory atomics) and ADDED per-needle cost (a 32-byte multi-word host-mapped result record +
  `membar.sys` per needle + per-needle CAS). It is CONGESTION-bound: MORE threads make it WORSE (256+ time out on large
  batches; 1024 timed out), the opposite of the probe (scaled to 8192). **So the gate measured a crippled implementation
  in the wrong regime against the wrong baseline — it says nothing about whether an OPTIMIZED wave engine beats the
  batcher.** The conceptual framing stands (the wave engine IS R1's index probe driven by a persistent kernel — same
  index, same probe/gather), and the OPEN question is unchanged: can GPU-side coalescing (lock-free ring + persistent
  drain) beat the host-serial single-coalescer (156k) under concurrency? **UN-PARKED.** A FAIR test needs: (a) an
  OPTIMIZED drain (port 1d-ii batched claiming + 1d-i device atomics + leaner/packed result records -> target
  tens-of-M/s), (b) a CONCURRENT lock-free enqueue host model (N threads claim ring slots + spin on their own
  result-slot done flag; no central coalescer, no per-wave DtoH gate), measured (c) vs the BATCHER's 156k concurrent cap
  under concurrency. What DOES stand from the gate: synchronous single-flight wave is genuinely bad (round-trip > launch),
  and the freeze root cause (`cuMemAlloc`/`cuMemFree` device-sync) is real.
- **R2.2 PROPER PORT — the verdict FLIPS: the wave WINS at small (OLTP) batches (2026-06-27, commits `5e6b2302` +
  `bcc12af5`, `wave_vs_launch_per_batch_throughput`).** Did the proper port the review demanded: (1) optimized drain —
  clamped-batched CAS claim `[c, min(c+K,head))` + ONE `membar.sys`/`atom.add(completed)` per batch (520k -> ~2.27M/s,
  4.7x); (2) async `submit_async`/`harvest` API; (3) completion via a HOST-MAPPED `completed` MIRROR on its OWN cacheline
  (harvest reads local RAM, no DtoH; the cacheline split fixed a false-sharing per-wave-latency pathology). All
  byte-identical to the R1 index probe. **CORRECTED single-flight measurement (release, 1M rows, wave vs lpb lookups/s,
  wave/lpb):** batch1 124k/39k = **3.20x**; batch8 448k/309k = **1.45x**; batch32 1.21M/1.19M = **1.02x**; batch256
  2.06M/6.94M = 0.30x; batch65536 2.27M/23.4M = 0.10x. **The persistent kernel WINS at small batches** (the OLTP
  point-lookup regime) because it has NO per-batch launch: 8us/submit at batch 1 vs lpb's fixed ~25us launch+sync. lpb
  wins only at LARGE batches (launch amortized; the GPU index probe is bandwidth-bound at 23M while the wave hits its
  ~2.27M drain ceiling — CAS contention on the single `claim` counter; sharded counters are the lever to lift it). At
  batch 8-32 the wave (0.45-1.2M) is already 3-8x the batcher's 156k host-coalescer cap. **So the earlier naive-port
  "wave loses 118x" was the wrong regime + a crippled impl; the proper port is competitive-to-winning exactly where OLTP
  lives.** STILL DEFERRED: the CONCURRENT/pipelined depth-K test (many producers -> the wave's full advantage over the
  host-serial coalescer, the regime that would show the biggest win) — its harness had a bug (the engine itself is fine,
  ~8-26us/submit); rebuilding it is the remaining R2.2 step before any engine wiring (R2.2b).

- **R2.2 DEVICE-RESULT REWRITE — the wave now EXCEEDS lpb at EVERY batch size (2026-06-27, commits `7dd041e2`,
  `e1b2072f`, `aeaee74d`; `wave_vs_launch_per_batch_throughput`).** An independent adversarial audit of the proper port
  caught a **harvest-gate underflow** (a REAL correctness bug): `completed.wrapping_sub(base) < len` wraps to ~u32::MAX
  when the kernel lags the host (`completed < base`) -> gate FALSE-fires -> `read_records` returns UNWRITTEN slots
  (empty). It had inflated the benchmark to a fake **348M/s** (30/30 waves actually empty). Fixed with the
  `completed < base` behind-guard, and the timed loop now verifies every wave (row count + `black_box`) so a no-op
  can't inflate it again. Then three levers, each measured: (1) **grid-stride claim** (each thread statically owns
  `idx = tid, tid+T, ...`; no claim counter, no CAS) replaced clamped-CAS, which was claim-contention-capped ~3.2M at
  128 threads and DEGRADED past it; (2) a **thread-0 coordinator** mirrors host doorbell+head into DEVICE memory so the
  other 1000s of workers don't poll host-mapped ctrl over PCIe (that congestion starved the host's head write ->
  timeouts past ~512 threads); (3) the decisive one — **records to a DEVICE result ring + bulk DtoH harvest** on a
  separate stream. Sweeps proved the ~7.5M plateau was the per-needle **host-mapped record WRITE** (flat across K=8..2048,
  flat across 128..32768 threads, unchanged by bulk host I/O); moving records to device lifted it **4.2x**. **VERIFIED
  (T=8192, 1M rows, wave vs lpb, byte-identical in a reused-slot stress gate at K=1/max-threads):** batch1 68k/40k =
  **1.72x**; batch8 530k/313k = **1.69x**; batch32 2.17M/1.20M = **1.81x**; batch256 12.1M/7.06M = **1.72x**; batch65536
  **31.6M/23.4M = 1.35x** (peak ~31.8M @ 16k threads). The wave now MATCHES/EXCEEDS the launch-per-batch index probe
  across the whole sweep and reaches the bare probe's ballpark (~31.8M vs 45M / 1d-i's 30M). Two audits confirmed the
  number is REAL (not a no-op) and that the cross-stream DtoH ordering (membar.sys system-scope cumulativity reaching the
  copy engine) is sound — empirically validated by the stress gate. Small batches pay ~6us DtoH latency vs the prior
  host-mapped read but still beat lpb ~1.7x. **Caveats:** the cumulative-counter completion gate is **in-order /
  single-flight ONLY** (out-of-order depth-K pipelining needs a per-slot status gate); thread 0's block must stay
  co-resident (modest grids; backstop catches eviction). needles-to-device (toward the 45M bare ceiling) is a further
  lever. **The read-ceiling bet is now demonstrated IN-CRATE on the shared context, not just in standalone probes.**

- **R2.2 FOLLOW-UP REVIEW correctness gates C1/C2/C3 (2026-06-28, commit `6b28832e`; `docs/reviews/r2.2-wave-port-followup-review.md`).**
  A second independent review accepted the device-result milestone and required correctness fixes BEFORE engine wiring
  (R2.2b). Done; two independent adversarial audits of the fixes both returned SHIP/SOUND. **C1 (persistent-grid
  occupancy):** a grid-stride persistent kernel needs EVERY launched block co-resident (index i owned by thread i mod T);
  an un-resident block's indices never drain -> completed stalls -> 30s hang + empty rows. `new()` now clamps `threads`
  to `cuOccupancyMaxActiveBlocksPerMultiprocessor * SM_count` (test: 50M threads -> clamped -> correct rows in 0.12s, no
  hang). NOTE: occupancy is SOLO-device, so this closes the isolated-launch trap, NOT the shared-context coexistence trap
  (a co-resident fat grid could still evict wave blocks — the open R2 "sidecar-or-replace" question). **C2 (u64
  counters):** head/completed/base/idx were cumulative u32 and wrap at ~2^32 lookups (~135s at 31M/s), after which the
  `completed < base` gate guard false-fires -> permanent hang. Widened to u64 across PTX (8-byte-aligned layout: ctrl
  [doorbell@0, head@8, mirror@64], counters [completed@0, dev_head@8, dev_doorbell@16]; `atom.add.u64`) + host (gate via
  new `wave_ready()`). Oracle byte-identical, stress gate passes, throughput unchanged (31.1M @ 65536 = 1.32x lpb); unit
  tests cover the old-u32-boundary/kernel-behind cases + the host<->PTX offset contract. **C3:** `debug_assert!(status
  != 0)` in `read_records` makes a gate violation fail loudly. **`WaveReadEngine` is now wireable (C1/C2/C3 landed);**
  the concurrent-regime win remains unproven until P2 (per-slot status gate -> depth-K pipelining) + an offered-rate
  harness — keep R1's launch-per-batch index probe as the shipped default until then.

- **R2.2 P1 needles-to-device = TRIED + REJECTED (negative result, 2026-06-28; reverted, not committed).** The
  follow-up review's P1 hypothesis was that the host-mapped needle ring (kernel PCIe-reads each needle) caps the drain,
  and moving needles to a DEVICE ring via bulk HtoD would push ~31.8M toward the bare-probe ~45M. Implemented it (device
  `req_dev` ring + pinned-staged `cuMemcpyHtoDAsync` + `cu_stream_synchronize` before publishing head) and measured at
  T=8192 vs the host-mapped baseline (wave/lpb): **the needle read is NOT the bottleneck.** Large batch was UNCHANGED
  (65536: 30.7M vs 31.3M = noise; the ~5us HtoD+sync is negligible against the ~2.1ms drain), while EVERY small/mid batch
  REGRESSED from the added per-submit HtoD+sync latency (batch1 14.6us->17.6us = 1.72x->1.46x; batch8 1.69x->1.38x;
  batch32 1.81x->1.32x; batch256 1.72x->1.38x). So device-records already saturated the per-needle path; the residual
  ~31M cap is the GATHER/RECORD work (multi-col index probe + 32B record), not the needle read. The bare probe's 45M is a
  SIMPLER kernel (single u64 result, no multi-col row materialization) -> not a reachable target for the real
  row-materializing workload. **Conclusion: ~31.8M (1.35x lpb at 65536, 1.7-1.85x at smaller) is at/near the realistic
  in-crate ceiling for this workload; do NOT pursue needles-to-device.** The remaining real lever is **P2** (per-slot
  status gate -> depth-K pipelining + offered-rate harness) — the CONCURRENT regime the wave exists for, still
  unbenchmarked. (Oracle + stale-DtoH stress gate stayed green during the P1 experiment.)

- **R2.2 P2 = per-slot status gate + depth-K pipelining: the CONCURRENT premise VALIDATED (2026-06-28, commits
  `65c2b3de`, `494e3c9d`, `38e0f874`; two independent audits = no new runtime bug).** The cumulative-counter gate was
  single-flight-only (a later wave's indices push it past an earlier wave's range while a slot there is unwritten). **P2a:**
  replaced it with a PER-SLOT status ring — the kernel (now `.target sm_70`) `st.release.sys`-writes each slot's status
  (1=found/2=not-found) into a host-mapped ring after the device record body; `harvest` is ready iff EVERY slot of the wave
  is non-zero (host-local reads, ANY order); `submit_async` clears the wave's slots before publishing head. Removed the
  completed counter + thread-0 completed-mirror. BONUS: removing the thread-0 mirror hop CUT single-flight small-batch
  latency (batch1 14.6us->10.2us = 1.72x->2.51x lpb; batch8 2.46x, batch32 2.40x, batch256 1.90x; batch65536 ~1.25x).
  **P2b:** a depth-K test keeps 8 waves in flight and harvests them in REVERSE order, byte-identical across 64 reused-slot
  rounds — proves out-of-order pipelining soundness. **P2c (the premise gate):** depth-K pipelining lifts sustained
  throughput **1.4-2.1x over single-flight** (batch1 96k->190k=1.97x, batch8 2.08x, batch32 1.74x, batch256 1.42x),
  saturating ~depth-4 (the single host thread's submit+harvest loop then caps it; multi-producer offered-rate would push
  further). The PIPELINED wave is **~2.6-4.6x lpb and ~9-32x the 156k batcher** in the concurrent regime. **AUDITS:** two
  independent adversarial audits found NO new runtime bug (gate completeness, status/body alignment, clear-vs-write
  ordering, submit timeout all SOUND; the prior false-ready class is closed). Open items, documented + deferred to R2.2b
  wiring: the in-flight ring bound + per-ticket-harvest are caller contracts (UNENFORCED — assert when wired); depth-K
  `harvest` spin loops need their own deadline (the kernel writes no terminal status for a backstop-skipped wave). The
  cross-engine body visibility (DtoH copy engine reads the device body, ordered by the kernel's `.sys`-release of the
  status + harvest's `cuStreamSynchronize`) is the SAME property the device-result design relies on, empirically validated
  by the stale-DtoH stress gate + the depth-K reused-slot test. **The wave engine's premise (concurrent point reads beating
  the host-serial coalescer) is now demonstrated IN-CRATE.** NEXT = R2.2b (wire into a query path behind the default-OFF
  flag, with the contracts enforced) OR R3 writes.

- **R2.2b STARTED — integration map + blocker#3 crash-safe watchdog (2026-06-28, commits `3207f457`,`7210d484`).**
  Mapped the wiring (Explore): the `wave_engine_enabled` flag (lib.rs:316, default OFF), the swap point
  `submit_resident_int4_equal_any_payload` (engine_retained_read.rs ~479), the residency lifecycle (build in
  `populate_relational_residency_snapshot*`, drop in `invalidate_relational_residency_*`, engine_commit.rs), and where
  per-table read state lives (`ResidencyReadState`, engine_state.rs:502; R1's `WaveResidentIndex` cache,
  resident_storage.rs:25). DESIGN DECISIONS surfaced: (i) `WaveReadEngine` is `pub(crate)` in execution -> must export
  `pub`; (ii) the wave kernel BAKES projection offsets at launch but the engine only knows them at probe/template-prepare
  -> build the wave engine LAZILY per (filter_col, projection set), like the R1 index, NOT at admission; (iii) `submit`
  takes `&mut self` but residency state is a shared `Arc` -> `Arc<Mutex<WaveReadEngine>>` (serializes submits =
  single-flight first; the multi-producer headroom comes later from real connection concurrency).
  **Blocker#3 DONE (crash-safe watchdog):** a long-lived engine-owned kernel can't use a fixed backstop (it would die
  mid-operation) and must not zombie ~30s on SIGKILL (shared --gpu-reset-denied box). Added a host petter thread +
  heartbeat (ctrl+16); thread 0 rings the doorbell if the heartbeat goes stale > `watchdog_ns` -> kernel self-terminates
  in ~watchdog_ns. Independent audit = SHIP; it found a latent bug (petter write wasn't fenced -> GPU didn't see pets;
  old test passed only via a spurious fire-from-launch) + a startup race -> fixed: ARM the watchdog only after the first
  real pet, fence the petter, petter skips 0 (sentinel). Test: petted kernel stays alive through a healthy window
  (no spurious fire), self-terminates ~0.5s after simulated host-death. **Blockers #1 (in-flight ring bound) + #2
  (host harvest deadline) + audit RISK (pass a near-infinite `backstop_ns` with the watchdog) are deferred to R2.2b-1/2
  (the engine wiring), where they're enforced.** NEXT R2.2b increments: (1) engine owns the WaveReadEngine lifecycle;
  (2) route point reads through it (single-flight, fallback lpb); (3) end-to-end offered-rate A/B = the ship decision.

- **R2.2b-2 DONE + MERGED — wave engine WIRED + ROUTED into the read path (2026-06-28, commits `8feb4131`,`04f20bfc`,
  `b0ff7bb3`,`50bfdec5`,`f6fd436e`).** The persistent `WaveReadEngine` now serves resident int4 unique-key point
  lookups behind a new default-OFF `wave_persistent_engine_enabled` flag NESTED under `wave_engine_enabled` (which stays
  the lpb default) — flipping just the inner flag is the R2.2b-3 A/B lever. Built in audited slices:
  - **IMPEDANCE (the wiring blocker) resolved at the source.** The read path returns a DEFERRED
    `CudaI32EqualAnyProjectSubmission` (drained at completion); the wave returns rows SYNCHRONOUSLY. Replaced the
    submission field with a payload enum `RelationalRetainedInt4ProjectionPayload::{Deferred(submission) |
    Materialized(Vec<CudaI32BatchProjectionRow>)}`; the completion branches, and BOTH arms feed the identical downstream
    materialization (stable sort by row_index, SqlValue::Int4 map, per-needle grouping) -> byte-identical by construction.
  - **LIFECYCLE.** Per-table `Arc<Mutex<WaveReadEngine>>` cache (`WaveResidentReadEngine`), keyed + validated by
    `(column_idx, resident_device_ptr, projection_offsets)` — proj-set joins the key because the kernel BAKES projection
    offsets at launch. Lazy accessor `wave_read_engine_for` reuses the R1 index build (same NULL-as-0 + gather semantics).
    Serial-commit (catalog-latch) path EVICTS (drops the kernel) on DDL/drop/memory-pressure; the concurrent commit path
    does NOT (the `device_memory.get -> Err` gate makes the route unreachable after a tombstone, and a re-admission's
    ptr-keyed rebuild reclaims the stale engine — eviction in the commit critical section is too costly: Drop joins the
    petter ~watchdog window). `WaveReadEngine` is `Send`, so `shutdown`/`read_records` set_current the primary context for
    cross-thread Drop/harvest (the completion of `unsafe impl Send`; harmless on this auto-binding driver, load-bearing
    for portability/multi-GPU/multi-producer).
  - **LIVENESS — at-most-one resident wave kernel (LOAD-BEARING).** Two full-occupancy persistent spin-kernels in one
    context MUTUALLY STARVE (neither yields its SMs -> the descheduled one can't observe doorbell/watchdog), so tearing
    one down (`cuStreamSynchronize`, infinite backstop) HANGS FOREVER. A non-vacuity test (a proj-rebuild) caught this
    as a REAL hang. Fix: the accessor DRAINS every existing wave engine BEFORE launching a new one, under a
    `wave_build_latch` (serializes builds) + a double-checked lookup. KNOWN LIMITATION (documented): one wave engine
    TOTAL across tables (alternating shapes rebuild); and a concurrent reader holding a to-be-drained engine across a
    rebuild is out of scope for single-flight — both are the central concern for multi-producer R2.2b-3 (-> sub-occupancy
    sizing or a shared multi-table kernel = R2.2c).
  - **ROUTE.** `submit_resident_int4_equal_any_payload` picks wave (both flags) -> lpb index -> scan; the wave arm
    enforces blocker#1 (batch <= ring 65536) + blocker#2 (`submit`'s DRAIN_TIMEOUT) and FALLS BACK to lpb on any wave
    error/timeout/oversize (never a wrong result). `wave_route_hits` (AtomicU64) telemetry counts wave-served batches
    (vs fallback) — also the test signal that the ROUTE produced the rows (all routes are byte-identical, so output
    equality alone can't prove it).
  - **AUDITS (independent, never self-audit).** Slice 1 (impedance) = SHIP (5 sabotages caught). Combined (Slice 2a +
    lifecycle + routing) = SHIP-WITH-FIXES: confirmed liveness airtight for single-flight (route confined to the facade's
    single coalescer thread) + the deadlock fix load-bearing (proven by a hang-on-revert); found two VACUOUS test
    assertions (a silent always-fallback AND a removed eviction both still passed) -> ADOPTED: added `wave_route_hits`
    asserts + reordered the differential so eviction runs against a populated cache; plus a wrong-mechanism comment + a
    missing cross-thread set_current, both fixed.
  - **TESTS.** `r2_wave_engine_matches_lpb_differential` (GPU): wave == lpb == scan byte-identical across NULL projection,
    NULL-as-0 key (needle 0), absent needle, non-unique fallback, generation rebuild + eviction, proj-set rebuild + a
    direct-submit non-vacuity. `r2_wave_engine_concurrent_same_shape_single_flight` (GPU): 8 readers x 50, exactly one
    engine, all wave-served, no hang (the A/B workload). HAZARD: 3x sequential + 2x concurrent, zero CUDA 700/716/717.
    engine 438/0/297-ignored; execution 25/0/69-ignored; workspace clean. **R1 lpb stays default until the A/B (R2.2b-3).**

- **R2.2b-3 DONE — wave-engine offered-rate A/B; VERDICT = validated WIN for its regime, default-flip GATED on R2.2c
  (2026-06-28, `57640e0f`, `engine/examples/r2_wave_engine_ab`).** A 3-mode (scan / lpb-index / wave) A/B through the
  production retained-template path, RTX PRO 6000, 1M-row resident table, 2000 batches/mode. NON-VACUITY built in: every
  route asserted byte-identical before timing + `Engine::wave_route_hits()` confirms the wave SERVED every batch (aborts
  on a silent lpb fallback). Methodology cross-checks (the R2.2 "loses 118x" retraction lesson): regime = single-flight
  is the PRODUCTION single-coalescer path (point reads flow through the facade's one coalescer thread); baseline = lpb
  (the R1 shipped default) AND scan, not the raw 1-thread probe; impl = the wired device-result + per-slot-gate wave;
  batch=1 wave/lpb 2.45x ~= the documented P2 single-flight 2.51x (corroborates).
  - **SINGLE-FLIGHT (production regime) — wave WINS at every batch, biggest small.** wave/lpb lookups/s: b1 2.45x, b8
    2.08x, b32 1.61x, b256 1.17x, b4096 1.02x (wave/scan 2.0-2.7x). Wave LATENCY is also lower at every batch (b1 p50
    10us vs lpb 27us; p99 15 vs 35; p99.9 23 vs 47). The win shrinks as batch grows (per-batch launch amortizes) — the
    wave's edge is exactly the small-batch OLTP point-lookup regime.
  - **CONCURRENT (per-engine-Mutex ceiling — NOT production) — wave crosses UNDER lpb ~4 threads.** wave/lpb @batch=32:
    1t 1.69x, 2t 1.26x, 4t 0.93x, 8t 0.66x. The wired wave is single-flight (submits serialized by its `Arc<Mutex<_>>`),
    so it plateaus ~1.4M while lpb's pipelined per-batch launches scale with threads. (This is the multi-PRODUCER-direct
    regime, which the production single-coalescer path does NOT use today.)
  - **VERDICT: the wave is a STRICT WIN (throughput + latency) for the single-coalescer point-read path it is wired for,
    but DO NOT flip the default yet — two architectural gates remain, both R2.2c:** (1) AT-MOST-ONE-KERNEL -> a
    multi-shape point-read workload would THRASH (the coalescer processes shape-groups sequentially; each shape switch
    rebuilds the one wave engine = teardown+launch per batch). Needs per-shape coexistence (sub-occupancy sizing or a
    shared multi-table kernel). (2) The per-engine-Mutex concurrency ceiling -> a multi-PRODUCER lock-free-ring
    replacement of the coalescer (the wave's original premise; depth-K pipelining, no central Mutex) is what wins the
    concurrent regime. **Keep R1 lpb the default; the wave is validated + shipped behind the flag, ready for R2.2c to
    remove the gates and become the point-read fast-path default.** This is the charter trajectory bet: the wave is the
    better engine for the regime, gated on the concurrency architecture, not on raw GPU speed.

- **R2.2c gate-1 PROBE — K-coexisting-minimal-kernel approach REJECTED (negative result, 2026-06-28,
  `execution/examples/wave_multikernel_probe`).** To lift the at-most-one-resident invariant (so multiple point-read
  SHAPES each own a coexisting wave engine = no thrash), the obvious fix was minimal-grid (1-SM) kernels so K leave SMs
  free. The probe measured two unknowns for minimal (threads=256, 1 block) vs fat (threads=8192): (1) BUILD a 2nd engine
  while the 1st runs, (2) TEAR one down while the other runs, finite 4s backstop, timed. RESULT: (1) build-while-running
  = 0ms for BOTH -> `cuMemAlloc` did NOT device-sync (contra the earlier freeze worry; two kernels DO run concurrently —
  e1 served correctly while e0 spun); (2) teardown-while-running = 4000ms (=backstop) for BOTH minimal AND fat, and the
  survivor then failed. So **the torn-down kernel's DOORBELL EXIT fails whenever a second wave kernel is resident — it
  dies only via the backstop — and that cascades to break the survivor.** Single-engine teardown is fast (proven by the
  Send/foreign-drop tests), so it is coexistence-specific, NOT SM starvation (minimal kernels have ~all 188 SMs free) and
  NOT the alloc device-sync. **Conclusion: lifting at-most-one is NOT a grid-size change; the at-most-one invariant is
  VINDICATED.** Gate-1 options now: (a) diagnose+fix the coexistence doorbell/teardown (uncertain; deep GPU-scheduling/
  zero-copy-visibility question), (b) a single SHARED multi-shape kernel (one persistent kernel draining K rings — big
  redesign), or (c) a no-rebuild "one sticky/hottest shape gets the wave, all others use lpb" policy (small accessor
  change: on a DIFFERENT-shape miss return None->lpb instead of drain+rebuild; keeps at-most-one, captures the
  dominant-shape win without coexistence). The wave remains a validated single-shape single-coalescer win behind the flag.

- **R2.2c graphs spike — CUDA graphs do NOT close the lpb->wave gap; the gap is HOST MACHINERY, not launch
  (negative result, 2026-06-28, `execution/examples/lpb_cudagraph_probe`).** Hypothesis (user): the wave's
  single-coalescer win is launch-overhead avoidance (lpb 27us vs wave 10us), so a CUDA-graph-captured lpb launch
  (~2-5us) would close it with none of the persistent-kernel complexity. Spike: a gather op (same shape as the lpb
  point read — HtoD needles -> kernel gathers table[needle&mask] -> DtoH), per-batch p50 DIRECT (HtoDAsync+launch+
  DtoHAsync+sync) vs GRAPH (cuGraphLaunch+sync), byte-identical. RESULT: DIRECT == GRAPH == ~7us at batch 1/8/32
  (graph speedup 1.00x; 1.14x at 256). So **the raw GPU op floor is ~7us and it is the GPU ROUND-TRIP (sync-bound),
  not CPU-side launch submission** — graphs only collapse submission (~2 async ops, <1us for a single-launch batch),
  dwarfed by the round-trip. The premise is FALSIFIED: raw launch is 7us (not 27, not 2-5), graphs can't reduce it.
  **KEY BYPRODUCT: lpb's A/B 27us vs the raw 7us => ~20us is per-batch ENGINE HOST MACHINERY** (per-batch
  `cuMemAlloc` for result buffers + pooled-stream lease + event timing + deferred-complete), which the wave avoids via
  its pre-allocated ring (-> ~10us). So the genuine "cheaper alternative" to the wave for the single-coalescer regime
  is NOT graphs — it is **pooling/optimizing lpb's per-batch host machinery** (no persistent kernel, no at-most-one, no
  coexistence, no SM tax). The wave's unique remaining edge is the multi-PRODUCER concurrent regime (gate-2). Open
  before any wave default-flip regardless: measure the wave's SM-coexistence tax on a MIXED workload (~60% loss at 8
  reserved SMs per the coexistence gate) — size the kernel down / spin up only under point-read load, don't flip blind.

- **R2.2c host-machinery spike (phase split) — the lpb->wave gap is ALL in lpb's COMPLETE phase; recoverable
  without the wave (2026-06-28, `engine/examples/lpb_phase_split_probe`).** Timed SUBMIT vs COMPLETE separately on the
  REAL retained-template path (1M rows, 3000 batches), lpb vs wave, non-vacuity via `wave_route_hits`. RESULT (p50,
  batch=1): lpb = 9us submit + 16us complete = 25us; wave = 10us submit + 0us complete = 10us. So lpb SUBMIT ~= wave
  SUBMIT (~9-10us host enqueue); **the ENTIRE gap is lpb's 16us COMPLETE** = the TWO sync round-trips (DtoH match-count
  -> sync to size arrays, THEN DtoH the 3 result arrays -> sync) + event-elapsed + materialize. The wave's complete is
  ~0 because it harvests via its host-mapped per-slot STATUS ring (no count DtoH) and folds its ONE bulk DtoH into
  submit. (batch=32: lpb 13+20=34 vs wave 16+5=21 — same story.) **So the wave's single-flight win is RECOVERABLE on the
  lpb path with NO persistent kernel:** (a) collapse the two round-trips into ONE covering sync — over-fetch result
  arrays to `needles.len()` (valid: the index probe has <=1 match/needle, unlike the scan, so this needs a
  submission-type flag to not break the equal_any scan path) + DtoH count+results together + trim by count; (b) trim
  SUBMIT's 5 per-batch device-buffer leases (-> a single arena) + sample event timing instead of per-batch. Estimated
  lpb-optimized ~12-13us vs wave 10us -> MOST of the gap closes, with no at-most-one / coexistence / SM tax. The wave's
  only IRREDUCIBLE edge is the multi-PRODUCER regime (gate-2; the wave avoids the per-batch launch entirely). VERDICT:
  the host-machinery optimization is the genuine "cheaper alternative" — but it is a multi-part change to the SHIPPED R1
  lpb path (round-trip collapse + buffer arena + event sampling), so weigh it vs R3 (writes) before committing the surgery.

- **R2.2c throughput at scale/threads -> the read path is ALREADY over the OLTP SLO; read-side opt is polish, R3 is the
  bottleneck (2026-06-28, `engine/examples/r2_wave_engine_ab` swept over rows x batch x threads).** Data (RTX PRO 6000):
  - SINGLE-FLIGHT, wave/lpb ratio is TABLE-SIZE-INDEPENDENT (both O(1)): batch1 ~2.4x, batch32 ~1.6x, batch256 ~1.17x,
    batch4096 ~1.05x across 256k/1M/4M rows. Absolute SATURATES ~3.1-3.3M lookups/s at batch4096 (GPU-bound, scale-
    independent). lpb absolute by batch (~scale-independent): b1 ~36k, b32 ~0.9M, b256 ~2.35M, b4096 ~3.15M. (Only the
    SCAN baseline degrades with scale: wave/scan grows 2.4x->4x as rows 256k->4M; but lpb is the shipped baseline.)
  - CONCURRENT (batch=32, O(1) -> scale-independent), lookups/s by threads: lpb 840k(1t) -> 1.02M(2t) -> 1.45M(4t) ->
    1.67M(8t) -> 1.83M(16t) = SCALES with threads (pipelined launches); the wired single-flight wave PLATEAUS ~1.35M
    (per-engine Mutex) and crosses UNDER lpb at ~4t (wave/lpb 1.58x@1t -> 0.95x@4t -> 0.67x@16t).
  - vs the SLO (100k sustained / 400k peak TPS; a point-read txn ~= 1-several lookups -> ~100k-2M lookups/s): the read
    path delivers ~0.9M-3.3M lookups/s at batch>=32 across ALL scales AND scales with threads to ~1.8M+ -> comfortably
    2-30x the SLO. The ONLY sub-SLO regime is un-batched batch=1 single-flight (~36k lpb / ~88k wave) -- exactly what the
    coalescer batches away under real load. So NEITHER the wave NOR an lpb host-machinery opt addresses an SLO
    bottleneck; the read half is settled AND over-provisioned. **DECISION: the write half (R3) -- entirely unmeasured,
    gating the SLO and the CPU-engine deletion -- is the priority. Pivot to R3; keep lpb the read default + the wave
    validated behind the flag (its only niche, the multi-producer regime, is itself below where the read SLO bites).**

- **R2.2c CORRECTION — the read path is NOT settled: the GPU drains tens of millions; HOST result materialization caps
  end-to-end at ~3M (2026-06-28).** The prior "read over-provisioned, pivot to R3" note measured only the
  materialization-throttled END-TO-END rate (~3M), NOT the GPU drain. The standalone raw-drain benchmark
  (`wave_vs_launch_per_batch_throughput`, no SqlValue materialization, 1M rows, 8192 threads) shows the real ceiling:
  wave vs lpb lookups/s = batch8 707k/307k (2.30x), batch32 2.81M/1.18M (2.38x), batch256 12.86M/6.87M (1.87x),
  batch65536 30.0M/23.7M (1.26x). So **the wave really does tens of millions and is 1.26-2.4x lpb at the GPU level.**
  BUT through the engine (with `CudaI32BatchProjectionRow` -> `RelationalSelectResult`/`Vec<SqlValue>` materialization,
  ~270ns/row) BOTH cap at ~3M (batch256 wave 2.76M/lpb 2.35M; batch65536 wave 3.28M/lpb 3.03M). **The bottleneck for
  end-to-end read throughput is the HOST result materialization, not the GPU** — a ~10x headroom (3M delivered vs 30M
  GPU-capable), and it is a HOST-SERIAL bottleneck the charter says to FIX (it does not scale with GPU hardware, unlike
  the GPU drain). The small-batch cap is the per-batch host machinery (~16us round-trips, the earlier phase-split); the
  large-batch cap is per-row materialization (~270ns/row). NET: read-side decision is RE-OPENED — the wave is the better
  GPU engine (1.26-2.4x lpb), and the lever to deliver its tens-of-millions end-to-end is GPU-native / streamlined
  result materialization (likely GPU-side result formatting toward the wire, the charter's "final device->wire
  readback"), NOT the wave-vs-lpb choice alone. Supersedes "R2.2c throughput at scale/threads" + "host-machinery spike".

- **Result-path optimization — the ~3M end-to-end cap is the per-needle SCHEMA DEEP-CLONE; Arc-sharing it recovers 15x
  (>GPU drain) (2026-06-28, `engine/examples/{lpb_phase_split_probe,result_assembly_probe}`).** Phase-split across batch
  sizes localized the host overhead per row at batch=65536: lpb SUBMIT ~107ns/row (the per-needle
  `template.result_columns.clone()` + `access_path.clone()` in member-building — cloning identical template data once
  per needle) + COMPLETE ~226ns/row (`Vec<SqlValue>`-per-row + per-needle `RelationalSelectResult` assembly). CPU
  prototype (N=65536 point-read results): CURRENT (deep-clone columns/access_path per needle + Vec<SqlValue>/row) =
  399ns/row -> 2.5M/s (matches the engine's ~3M end-to-end); LEAN-ARC (Arc-share columns + access_path, same
  Vec<SqlValue>) = 25.8ns/row -> **38.8M/s (15.5x)**; FLAT (+drop per-row Vec) = 1.8ns/row. **So the entire end-to-end
  read cap is the per-needle SCHEMA DEEP-CLONE (each RelationalColumn has String fields -> heap allocs, x N needles);
  the per-row Vec<SqlValue> is NOT the bottleneck.** Arc-sharing the result schema (`RelationalSelectResult.columns` ->
  `Arc<Vec<RelationalColumn>>`, `access_path` -> Arc) is a CONTAINED change (~21 construction sites; reads deref
  transparently) that lifts the host path to 38.8M/s -- ABOVE the GPU drain (30M). **CONSEQUENCE (answers the sequencing
  question): materialization-first is right + cheap, and solving it makes end-to-end GPU-DRAIN-bound -> the persistent-
  kernel decision UNMASKS to wave 30M vs lpb 23M (1.26x large batch, up to 2.4x small batch), measurable for the first
  time. NEXT: implement the Arc-share + re-measure end-to-end, then decide the persistent kernel on the real number.**
  **IMPLEMENTED (`a59add53`): `RelationalSelectResult.{columns,access_path}` -> `Arc`, retained-read batched path shares
  ONE Arc/batch (refcount-clone per needle). Measured end-to-end (1M rows): batch256 lpb 2.35M->4.06M / wave
  2.76M->5.42M; batch4096 lpb 3.17M->7.15M / wave 3.32M->7.84M; batch65536 similar. ~2.4x; the wave edge is now PARTLY
  unmasked (wave/lpb 1.10x large -> 1.83x batch32, trending toward the raw-drain 1.26-2.4x). Behavior-preserving (engine
  438/0, GPU differential byte-identical 3x). Residual ~127ns/row cap (7.8M, still < the 30M drain) = the per-row
  `Vec<SqlValue>` + completion grouping/sort/round-trips (the "flat"/columnar layer the prototype showed at 1.8ns/row) --
  a FURTHER, bigger change (rows representation) if the full GPU drain end-to-end is wanted. Kernel decision: still
  trending wave-favorable; full unmask needs the per-row layer.**
  **STEP-1 BREAKDOWN (`result_materialization_probe`, CPU, N=65536): current `Vec<Vec<SqlValue>>` + group/sort =
  153ns/row (6.5M); FLAT-SQLVALUE (one row-major `Vec<SqlValue>` + per-needle ranges) = 2.3ns/row (433M, 66.5x);
  FLAT-I32 (raw, no enum-wrap) = 0.8ns/row (186x). So the residual is ENTIRELY the per-row `Vec<SqlValue>` boxing +
  per-needle grouping/sort -- NOT the SqlValue conversion (the enum-wrap is only a further 3x). FLATTENING THE `rows`
  CONTAINER to a row-major `Vec<SqlValue>` (KEEPING SqlValue -> wire value-type + encoding unchanged) recovers 66.5x ->
  2.3ns/row, FAR below the GPU drain (33ns/row) -> the engine materialization becomes negligible and end-to-end goes
  GPU-DRAIN-bound (wave 30M / lpb 23M fully unmasked). flat-i32/GPU->wire is a further 3x, NOT needed to reach the
  drain. So the columnar layer = a `rows`-CONTAINER flatten (Vec<Vec<SqlValue>> -> flat + shape), value semantics
  unchanged -- wide (rows is read everywhere) but more tractable than a value-type/GPU->wire rewrite. NEXT: prototype
  flat `rows` on the hot retained path + facade, re-measure end-to-end vs the drain, then the kernel call is on the
  real GPU-bound number.**
  **ROWBLOCK IMPLEMENTED + MERGED (`c4b0bddb`+`1df5dceb`): `RelationalSelectResult.rows: Vec<Vec<SqlValue>>` ->
  `RowBlock { values: Vec<SqlValue>, ncols }` (flat row-major; transparent traits iter/Index/PartialEq/From/IntoIterator;
  manual row-based PartialEq; hot completion builds flat). ~50 consumer sites updated; pgwire byte-identical (protocol
  71/0). Audit = SHIP (3 sabotages caught; 2 unreachable P3s, one adopted = uniform-width debug_assert in From).
  MEASURED end-to-end (1M rows): 7.8M -> 10.8M (+37%); **wave/lpb UNMASKED to 1.93x@b32 .. 1.13x@b65536 (was ~1.0-1.08x),
  tracking the raw-drain 1.26-2.4x -- the wave's edge is now VISIBLE end-to-end.** Did NOT reach the full ~28M drain: the
  residual (10.8M) is the per-needle `RelationalSelectResult` struct + by-needle grouping/sort, INHERENT to one-result-
  per-point-read (the batcher dispatches a result per coalesced needle) and EQUAL for lpb+wave, so it caps absolute
  throughput but no longer HIDES the wave ratio. Going past it needs a per-needle-result-MODEL change (a batcher-contract
  change = a separate effort). **NET: the wave-vs-lpb kernel decision is now on real, mostly-unmasked numbers
  (1.13-1.93x). Result-path optimization is DONE for the masking purpose.**
  **PER-NEEDLE RESULT MODEL -> BATCHED, IMPLEMENTED + MERGED (`d49aed6f`+`98f677c8`+`68b8e455`): replaced the
  N-per-needle `RelationalSelectResult` model with `RelationalRetainedBatchResult { columns: Arc, access_path: Arc,
  gpu_id, rows: RowBlock (ALL needles flat, needle-ordered), needle_ranges: Vec<(start_row,count)> }` built by ONE
  sort by (needle_index, row_index) + one flat buffer + per-needle ranges (vs N grouping Vecs + N RowBlocks + N
  structs + 2N Arc clones). `complete_relational_retained_read_submission_batched` returns it; the point-lookup
  batcher (`distribute_results_batched`) slices the flat block per needle + maps the SHARED schema to wire ONCE
  (was per-needle). MEASURED (1M rows, engine A/B): wave per-needle 10.93M -> wave-BATCHED 19.47M (+78%, approaching
  the 30M raw drain); BATCHED wave/lpb 1.62x@b256 .. 1.76x@b65536 (was 1.13-1.39x per-needle) -- the wave's edge is
  now CLEARLY unmasked AND substantial. Byte-identity: r2_batched_completion_matches_per_needle{,_multirow} (GPU,
  the multirow one closes an audit-found vacuity on intra-needle order: proven by a reversed-sort sabotage that
  FAILS it) + facade 32/0 (batcher wire byte-identity) + protocol 71/0. Audit = SHIP (no P1; P2 test-vacuity +
  P3 ncols-from-nonempty both adopted). CAVEAT on the 1.62-1.76x: lpb-batched (11M) is still bounded by lpb's
  2-round-trip complete, NOT its 23M raw drain -> the TRUE GPU ratio is the raw 1.26x(b65536)..2.4x(small); the
  remaining lever to hit the full 30M end-to-end on BOTH routes is collapsing lpb's two round-trips (host-machinery,
  separate). **NET: read-path absolute now ~19.5M (6.5x the session-start 3M); the wave-vs-lpb kernel decision is
  on strong, mostly-unmasked numbers.**
  **LAST READ LEVER — O(n) SCATTER, IMPLEMENTED + MERGED (`3f719e3a`): instrumented (GPU_DB_PROBE_TIMING, since
  removed) the engine batched completion at b65536 -> the dominant host cost was NOT lpb's round-trips: it was the
  assembly's O(n log n) global `sort_by_key((needle_index,row_index))` = ~1800us (~80% of the ~2200us assembly;
  flatten ~240us, GPU drain ~1775us, members ~360us, payload ~30us). lpb's projected_rows arrive in atomic-add emit
  order (UNSORTED) so the global sort was its assembly bottleneck; the wave harvests needle-ordered so it was never
  sort-bound. FIX: replaced the global sort with an O(n) counting-sort scatter (cursor walks each needle's prefix-
  summed range; a within-needle row_index sort runs ONLY for needles matching >1 row = a non-unique predicate).
  Byte-identical (row_index unique within a needle => total order; differentials incl the multirow one that
  exercises the within-needle sort). MEASURED (1M rows): lpb-batched 11.09M -> 14.62M @b65536 (+32%), 10.81M ->
  15.05M @b4096 (+39%); wave-batched unchanged (already needle-ordered). **BATCHED wave/lpb 1.76x -> 1.33x @b65536,
  1.21x @b4096 -- CONVERGING ON THE TRUE RAW GPU RATIO (1.26x): both routes are now at their drains, so the ratio
  reflects the GPU mechanism (wave's ring-harvest vs lpb's 3-sync D2H), not host assembly overhead. THE READ-PATH
  RATIO IS NOW HONEST.** (Caveat: lpb b65536 p99 spiked to ~30ms vs p50 3.8ms — a blocking-sync tail on the shared
  box, lpb-path only, not the wave; worth a look if lpb ever becomes default. Remaining marginal levers: members
  ~360us + flatten/SqlValue ~240us, help both equally, do NOT change the now-honest ratio.)**

## Tail latency (SHOWSTOPPER, user 2026-06-28) -> COLUMNAR RESULT, IMPLEMENTED + MERGED (`10c724e6`)
- **Context:** user flagged the lpb b65536 p99 tail (~30ms vs ~3.8ms p50; the wave was clean) as a SHOWSTOPPER to fix
  BEFORE the other levers, and to treat as a real defect not shared-box jitter. ([[working-agreement-sequencing]].)
- **Root-cause (MEASURED, `lpb_phase_split_probe` percentiles + drain sub-phase instrumentation, since removed):** NOT
  the GPU (kernel-drain `cuStreamSynchronize` = 1-6us even on tail events) and NOT shared-box jitter. The drain
  (`complete_detached`) built **65536 owned `CudaI32BatchProjectionRow` structs (one heap `Vec<i32>` each)** from the 3
  flat D2H arrays it ALREADY had -- which the engine then immediately re-flattened. That per-row allocation storm was
  **~1585us of EVERY batch's drain (the single biggest cost)** and ballooned to ~3850us under allocator pressure = the
  tail. (Steady-state sub-phase split @b65536: rows-assembly ~1585us, result-D2H ~130us, count ~6us, sync1 ~0us.)
- **Decision/FIX:** carry matched rows COLUMNAR end-to-end (the 3 flat arrays), never per-row.
  `CudaI32BatchProjectionColumns {values, needle_indices, row_indices, projection_count}` (+ from_rows/into_rows
  bridges); `complete_detached_columnar` drains to it (index-array validation in a tight alloc-free loop, same
  invariants); the wave's `read_records`/`submit_columnar`/`harvest_columnar` build it directly; the engine's
  Materialized payload + batched completion + `assemble_batched_rows` consume columnar. `complete_detached`/`submit`/
  `harvest` kept as thin `into_rows` wrappers for the cold per-needle path + tests/probes (no churn).
- **MEASURED (1M rows, b65536):** lpb COMPLETE **max 9073us -> 1785us**, p99.9 4292 -> 1245 (TAIL FIXED). BONUS (the
  per-row Vec was also the steady-state cap + forced a cache-hostile re-read of 65536 scattered Vecs): lpb-batched
  **14.6M -> 44.1M (+3x)**, wave-batched 19.5M -> 37.4M (+~2x). **REVERSAL: with the boxing gone, lpb now BEATS the wave
  (batched single-flight wave/lpb 0.85x@b65536, 0.94x@b4096)** -- the per-row Vec had been MASKING lpb's true speed (it
  sat in lpb's critical-path COMPLETE; the wave's was partly hidden in submit). Re-opens the wave-vs-lpb question:
  for the batched single-flight regime lpb is now the faster route. Byte-identical (GPU differentials wave==lpb==scan +
  batched==per-needle + multirow 3/0; execution wave 8/0; facade 32/0; protocol 71/0). Audit = SHIP (P3 into_rows/
  from_rows zero-width footgun, structurally unreachable, hardened with fail-loud debug_asserts).
- **The other read levers (user "attempt optimization", after the tail) -> MERGED (`07186f1b`+`c1a27f3a`+`ac1a6901`):**
  (A) **SLIM THE SUBMISSION (big win):** the template submit built N member tuples (2N Arc clones + a 1.5MB Vec)
  whose only used content was `members.len()` + the shared schema (the needle value was unused). Replaced
  `members: Vec<(Arc,Arc,i32)>` with `needle_count` + `shared_columns: Arc` + `shared_access_path: Arc`; cold
  per-needle completion refcount-clones the schema per needle at completion. MEASURED: lpb submit p50 388us -> 28us;
  **lpb-batched 44.1M -> 78.5M @b65536 (+78%)**, 35.7M -> 53.4M @b4096; wave 37.4M -> 57.5M. (B) **i32 BATCHED RESULT
  (byte-identical, memory/batcher win, NOT engine throughput):** the int4 route is always i32 (NULL pre-encoded 0)
  and the batcher mapped SqlValue::Int4 right back to DbValue::Int4, so `RelationalRetainedBatchResult.rows: RowBlock`
  -> `values: Vec<i32>`; batcher maps i32->DbValue::Int4 directly. HONEST: the engine A/B is UNCHANGED (78.5M->77.9M,
  noise) -- post-columnar+members the SqlValue flatten was NOT the engine bottleneck (the ~240us estimate was a stale
  pre-columnar number). The win is off-A/B: ~6x smaller result buffer (~3MB->512KB/batch, eases allocator pressure +
  tail) + the batcher drops the SqlValue->DbValue re-map. Audit (both) = SHIP, no P1/P2; the i32-vs-NULL question
  adjudicated NO (route gated all-Int4, NULL-as-0, old path only ever wrapped SqlValue::Int4, mixed-type routes away
  from the batcher; 5 sabotages caught). P3: jobs-path (test-only, non-wire) now shares the first job's access_path --
  valid (each job is single-key so matched_keys==1 for all), documented.
  **READ-PATH ARC THIS SESSION (lpb-batched @b65536): 11M (pre-result-path) -> 14.6M (tail-fix start) -> 44.1M
  (columnar) -> 78.5M (slim). The wave-vs-lpb question is OPEN (lpb now wins large batches, wave wins small ~b256);
  do NOT push it. Other open points remain per the user.**
- **lpb timing breakdown (user-requested, b65536 steady, instrumented then reverted): submit ~23us; complete ~688us =
  drain ~215us (kernel-sync ~18 + count ~5 + result-D2H ~125 [1.3MB, PCIe-bound] + validate ~53) + assemble ~452us
  (scatter+flatten). The per-row-Vec storm is GONE; the host ASSEMBLE is now the dominant cost.** Two follow-on levers:
  - **(1) ASSEMBLE unique-key fast-path -> DONE + MERGED + audit SHIP (`92d17f65`):** for `!any_multi` (<=1 row/needle,
    the dominant point read) write each row's i32 DIRECTLY at its needle's prefix-summed offset in ONE pass — no
    `slot`/`cursor` (-512KB allocs), no separate flatten. Byte-identical (count==1 => same offset; CPU test
    r2_batched_assembly_unique_fastpath + GPU differentials, 3 sabotages caught). lpb-batched 77.9M -> 84.6M @b65536
    (+9%). Smaller than the ~250us estimate — the residual scatter-copy (cache-unfriendly random write by needle
    offset) + the values buffer remain (inherent: GPU emits in atomic-add order, host must reorder).
  - **(2) ROW_INDICES elision (the 512KB u64 array, ~48us, used only to sort MULTI-row needles -> waste for unique) ->
    ANALYZED, RECOMMEND SKIP:** runtime-detection version nets only ~18us (the unique case pays a ~30us count pass the
    engine then repeats) + adds a conditional 2nd round-trip to the delicate async-D2H drain; metadata version (plumb
    catalog unique-index flag -> submission -> drain skip) gets the full ~48us but is multi-layer + needs a uniqueness
    guard (wrong flag -> empty row_indices -> panic). BOTH are ~3-7.5% LARGE-BATCH-ONLY (at small batches row_indices is
    a few rows = ~0) and touch the safety-critical drain. Deferred in favor of small-batch work. Revisit only if the
    large-batch case specifically matters.

## ADR-007 — Full GPU-native, zero deferrals (scope = everything, incl. the oracle)
- **Status:** Accepted (user, 2026-06-23)
- **Context:** A cross-session pattern of deferring the hard GPU kernel and shipping a host-side stub.
- **Decision:** Go full GPU-native with **zero charter violations, zero deferrals**. Retire the CPU parity oracle
  AND the GPU-absent bootstrap fallback entirely — zero host relational code anywhere, even tests/CI. Parity uses
  GPU-native oracles. The engine **requires** a GPU.
- **Consequences:** The GPU-native read-path campaign (S1–S10c) executed this for the read path; S10d (delete the host path) is
  gated on STRATA auto-admission (ADR-010).

## ADR-006 — GPU required; no CPU steady-state fallback (supersedes ADR-003)
- **Status:** Accepted (2026-06-26). **Supersedes ADR-003.**
- **Context:** ADR-003 made permanent CPU fallback "mandatory" on the premise GPU availability varies. The mandate
  changed: the engine requires a GPU; CPU relational execution is interim WIP being deleted.
- **Decision:** No CPU-only / GPU-absent / hybrid steady-state mode. Any CPU relational execution is interim
  GPU-parity debt, tracked and scheduled for deletion. Parity verified against a GPU-native oracle, never a CPU
  re-implementation.
- **Consequences:** The CPU relational read/execute path (`finalize_relational_select`, MVCC `cpu_fallback`,
  `FirstCudaSliceParityBackend`) is retired once STRATA admission makes the GPU path the default.
  **Durability/replication remain host control-plane responsibilities (unchanged; ADR-001/ADR-004)** — "no CPU
  steady-state" scopes the *relational data path*, not the host's control-plane role.
- **Alternative rejected:** keep ADR-003's permanent CPU fallback — contradicts the GPU-required charter.

## ADR-005 — Snapshot / install-snapshot strategy
- **Status:** Accepted. **Decision:** Replication snapshot / install-snapshot hooks required from early phases
  (before full distributed rollout) — for both log **compaction** AND **fast follower catch-up** (install-snapshot).
  **Consequences:** lowers future integration risk; slight early overhead.

## ADR-004 — Replicator interface contract
- **Status:** Accepted. **Decision:** Define `LogReplicator` + `ReplicatedStateMachine` interfaces before deep
  implementation. **Consequences:** prevents transport leakage into storage/executor; forces early API rigor.

## ADR-003 — CPU fallback policy ❌ SUPERSEDED
- **Status:** SUPERSEDED by ADR-006 (2026-06-26). Originally "CPU fallback is a mandatory, permanent safety path."
  Reversed — the engine now requires a GPU and treats CPU relational execution as interim debt. Retained as the
  record of the reversal.

## ADR-002 — Deterministic batch ordering
- **Status:** Accepted. **Decision:** Replicate ordered transactional intent; apply in deterministic log order.
  **Consequences:** simplifies follower convergence; requires strict ordering metadata + replay discipline. (The
  foundation ADR-009's deterministic CC extends.) **Alternative rejected:** replicate post-execution effects only —
  divergence/debug complexity.

## ADR-001 — Log boundary is the WAL / replicated log
- **Status:** Accepted. **Decision:** All commits pass through `LogReplicator` and are **durable before
  visibility**. **Consequences:** one interface for local + raft modes; requires strict durability/visibility
  sequencing. **Alternative rejected:** direct local writes + later raft overlay — high refactor risk.
