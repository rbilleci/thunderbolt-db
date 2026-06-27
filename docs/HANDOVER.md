# HANDOVER — Resume Baton

> **This is a SINGLE ROLLING file. Overwrite it each session — never date it, never accrete.** Keep it short:
> where we are, the one next action, and the open decisions. Everything else lives in the other five docs.

**Updated:** 2026-06-27.

## Where we are
- Docs are **6 canonical files**: [CHARTER](CHARTER.md), [ARCHITECTURE](ARCHITECTURE.md), [DECISIONS](DECISIONS.md),
  [PLAN](PLAN.md), [STATUS](STATUS.md), this. Suite **731/0**; `main` == branch == origin.
- **Workload = high-throughput OLTP** (ADR-008). **Success bar (clarified): same ballpark on TODAY's hardware + a
  GPU-architectural gap that CLOSES with hardware — NOT beat-the-CPU-today** (host-serial gaps are fixes, not losses).
- GPU-native **resident read path complete for int4** (S1–S10c). **STRATA S-A + S-B landed** (auto-admission producer,
  default-OFF). Execution = waves (ADR-009); residency = STRATA (ADR-010).
- **OLTP benchmark v1 built** (PLAN §1, `engine/examples/oltp_auto_admit_ab` + `facade/examples/oltp_batched_read_scaling`).
  Findings (DECISIONS ADR-008 "First measurement"): a GPU point read is a fixed ~72µs (launch overhead); even *batched*
  it was 11× under the CPU — the gap was **host-serial**, not GPU. → **STRATA S-F (flip auto-admit ON) decided OFF**
  (net-negative today; resident point reads lose).
- **Batcher Tier-1 landed** (per-shape resident-read template): batched point reads **68k → 156k ops/s (2.3×)**, CPU
  gap **11× → ~4.5×**. Residual bottleneck = the **single-coalescer host-serial cap (~167k)**, which does NOT scale
  with GPU hardware. (Audit caught + fixed a pre-existing mixed-int4+text CUDA-201 bug → mixed routes to per-query.)
- **Wave engine (ADR-009) 1a + 1b + 1c PROVEN** (isolated standalone probes `crates/execution/examples/wave_*_probe.rs`;
  each independently audited):
  - **1a** = persistent kernel clean **~3.5µs doorbell exit** + `%globaltimer` backstop (the `--gpu-reset`-box risk is
    de-risked).
  - **1b** = parallel data plane (lock-free claim, on-GPU scan+gather, packed atomic result) = host-serial cap broken;
    full-scan O(rows) → 1M-row table = 485k req/s.
  - **1c** = a GPU hash index removes the scan → **~10.5M point lookups/s FLAT across 1M/4M/16M-row tables (O(1)),
    ~13.6× the CPU's 770k, table-size-independent. THE OLTP POINT-READ BET VALIDATED at the data-plane level** — the GPU
    does millions of lookups/s at realistic scale; the residual bottleneck (host-mapped atomics) is GPU-architectural
    (scales with hardware). NOT integrated into the engine; the batcher remains the production default.

- **Wave 1d-i + 1d-ii** (push the read ceiling) ✅ DONE: claim/`completed` atomics → **device memory** (1d-i, ~30M),
  then **batched claiming** K=8 (1d-ii) = read ceiling **10.5M → ~45–53M req/s (~5×, ~60–69× the CPU)**, 3× stable.
  slot→wire mapping quantified MINOR (~200M–1.1B rows/s, ≪ GPU drain). **Read half of the bet is SETTLED** — further
  read gains are diminishing/tuning-sensitive. (`wave_devatomic_probe.rs`, `wave_batchclaim_probe.rs`.)

- **Write probe 1** (concurrent lock-free index INSERT) ✅ DONE: many threads `atom.cas.b64`-insert into a shared
  open-addressing table at **~tens of BILLIONS of inserts/s** (all verified) → **GPU index maintenance is NOT a
  bottleneck**; the remaining write constraints (durability/WAL fsync, deterministic CC) are host-I/O + coordination
  problems CPU engines face too. (`wave_index_insert_probe.rs`; caveats: low contention, raw insert only.)

## R1 ✅ DONE + MERGED (index in the engine read path, default-OFF flag) — SHIP; next = measure, then R2
**R1a (`53b5fc97`) + R1b (`3f7e5b08`) + audit-adoption (`ffeccfc4`) + re-audit fix (`5b84bb9e`)** are merged to main.
Two independent adversarial audit cycles; **re-audit verdict = SHIP the flag-on path** (both P1 divergences + all P2s
closed at the root). The GPU hash-index probe serves resident int4 unique-key point lookups when `wave_engine_enabled`
is on, returning **byte-identical** results to the scan; default OFF leaves the scan path untouched.
- **Key design (audit-driven):** the index is built from the **same device bytes the scan reads** — `CudaResidentDeviceMemory::
  read_resident_i32_column` DtoH-reads the key column, `build_wave_resident_int4_index` hashes it (NO host_rows). This makes
  NULL-as-0 keys match the scan AND makes the index inherently consistent with the buffer it gathers from (cache keyed by
  `(column_idx, resident_device_ptr)`, buffer pinned so the ptr can't be reused). Index buffer pinned in the submission.
- **Swap:** `submit_resident_int4_equal_any_payload` (`engine_retained_read.rs`), flag-gated index-vs-scan, same submission
  type → byte-identical completion. Falls back to scan on: non-unique keys (dup detect), >256-probe, un-buildable. Distinct
  needles required (batcher `dedup_needles`, debug-asserted).
- **Gate:** `r1_wave_index_probe_matches_scan_differential` (flag OFF vs ON) — NULL projection + NULL-as-0 key (needle 0) +
  absent + dup-fallback + generation rebuild, non-vacuous. Suites green: engine 437 + 295 GPU, execution 84, facade 34.
  Independent adversarial audit done (2 P1 + 2 P2 all adopted via `ffeccfc4`); **re-audit of the fix in flight.**
- **NEXT:** (1) ✅ DONE + VERIFIED — end-to-end O(rows)→O(1) win MEASURED (`engine/examples/r1_wave_index_ab`, DECISIONS
  ADR-008 "R1 end-to-end measurement"): flag ON vs OFF on the production template path, ON==OFF byte-identical each size.
  Verified re-run: scan falls ≈O(rows) (1M→4M→16M = 1.68M→731k→247k lookups/s), index is **FLAT ~2.3M across all sizes
  (true O(1))** → **16M rows = 9.34×**. **Crossover ≈1M rows**; below it the fixed ~110µs host+launch floor dominates
  (~1.0× at ≤256k) → batcher stays default; any flip is size-aware or lands with R2. (2) **R2** — persistent kernel + ring
  (breaks the host-serial coalescer cap = the ~110µs floor this measurement is now bound by → proven 10–30M).
  **SM-coexistence gate ✅ MEASURED (`execution/examples/wave_coexist_probe`, DECISIONS ADR-008 "R2 SM-coexistence
  gate"): VIABLE** — a persistent kernel + concurrent engine scans on ONE shared context never deadlock/starve and exit
  cleanly (188 SMs), BUT the cost is steeply non-linear: 1 reserved SM ~2%, **8 SMs ~60%, 32 SMs ~87%** of concurrent
  scan throughput (≈ same busy-spin vs gentle ⇒ SM co-residency, not poll traffic). **Design rule: the wave kernel must
  be ~1-SM minimal as a sidecar, or REPLACE the per-batch path (ADR-009 intent) — never a fat always-resident
  co-resident.** **`all_done` ordering audit ✅ DONE** (two independent auditors, DECISIONS ADR-008 "R2 `all_done`
  ordering audit"): they split (cumulativity GAP vs SOUND) but CONVERGED — `all_done` alone is NOT a robust gate; the
  sound completion gate is the host **acquiring the `completed` counter** (DtoH `==requests` before reading slots = the
  proven 1b pattern; `all_done` is only a wake hint). **Engine-lift rule: gate on the counter-acquire, never on
  `all_done`-only** (ideally `.release.sys`/`.acquire.sys` under sm_70). `cuMemHostGetDevicePointer` already in the FFI
  (probes). **R2.1 BUILD ✅ DONE** (`crates/execution/src/wave.rs`, new child module): `WaveReadEngine` lifts the proven
  1d data-plane into the crate, running the persistent kernel on the engine's **shared primary context** (not a throwaway
  one). Multi-wave `submit(needles)->Vec<Option<i32>>` over a circular lock-free ring; completion GATED ON the DtoH
  `completed` counter-acquire (audit rule), `all_done` = wake hint; doorbell + `%globaltimer` backstop + clean Drop.
  Kernel = the audited probe kernel with ONE change: `atom.cas` claim (bounded, no overshoot) instead of `atom.add`, so
  cumulative `head` works across waves (the result/completion ordering path is unchanged → audit still holds). GPU test
  `wave_engine_point_lookups_match_oracle_on_shared_context` (#[ignore]) passes: 2 waves, results == CPU oracle, clean
  exit; ASCII-PTX guard + exec suite 24/0/62-ignored green. **R2.2a ✅ DONE** — multi-column projection (R1's 4-way
  gather) over the ring, `submit` returns `CudaI32BatchProjectionRow`s **byte-identical to the R1 index probe** (GPU
  oracle test green); `atom.cas` bounded claim makes cumulative multi-wave work. **BUT a CRITICAL BLOCKER for R2.2
  wiring surfaced (DECISIONS ADR-008 "R2.2a ... INTERLEAVED-LAUNCH FREEZE"):** launching the GPU index-probe oracle
  BETWEEN waves FREEZES the idle persistent kernel (claim frozen, no error/fault/backstop); reordering so all submits
  precede any launch passes. The SM-coexistence probe only tested NON-blocking concurrent launches (fine); the
  index-probe uses a flag-0 (blocking) pooled stream + NULL-stream memcpy — the legacy-default-stream path is the prime
  suspect. **ROOT CAUSE PINNED (`execution/examples/wave_freeze_probe`, DECISIONS ADR-008 "R2.2 freeze ROOT CAUSE"): it
  is `cuMemAlloc`/`cuMemFree`, NOT the stream type.** The probe shows every interleave op (non-blocking/blocking launch,
  NULL-stream HtoD/DtoH, events, the combo) keeps the kernel ALIVE in µs; only `cuMemAlloc+cuMemFree` FREEZES it, and
  that op blocks ~the backstop (4.96s of a 5s backstop) — i.e. `cuMemAlloc/Free` **device-synchronize**, blocking until
  the never-ending wave kernel hits its backstop and dies. (Explains the original 50s test: ~30s cold `cuMemAlloc` to
  the 30s backstop + 20s wave-2 timeout.) **R2.2 path:** the engine's device-buffer pool amortizes `cuMemAlloc` (steady
  state reuses pooled buffers; syncs only on cold growth / overflow free), and the wave path is itself alloc-free
  (pre-allocated device-mapped ring) — so **pre-warm the pool + suppress pool shrink while a wave kernel is resident**
  (pragmatic), or migrate engine device alloc to `cuMemAllocAsync` (robust). "Replace per-batch" ALONE is insufficient
  (other engine activity still `cuMemAlloc`s). **NEXT = R2.2b** wiring under that constraint → R2.3 (audit + measure vs
  batcher). (3) **R3** — writes + deterministic CC.
- **Discovered pre-existing bug (out of R1 scope, follow-up):** the jobs-batch path
  (`submit_relational_retained_int4_projection_batch`) does NOT dedup needles; the scan kernel emits a matched row under
  only the FIRST matching needle_index, so a duplicate job (`WHERE id=1` twice) gets an empty result for the 2nd. The
  facade-template path is unaffected (`dedup_needles`). Fix = dedup in the jobs-batch path (or map results by value).
*Parked (verify before starting):* open-loop offered-rate + tuned-Postgres baseline (the real end-to-end OLTP-fitness
instrument, incl. the WRITE path which is still unmeasured); batched-mixed int4+text; STRATA S-C/S-D/S-E.

## Top open decisions (unresolved)
- **Data-size envelope:** OLTP working set vs aggregate VRAM, and the over-VRAM spill/tiering model (cross-shard
  combine doesn't exist; STRATA placement, not hardware paging, owns the tail).
- **Coherent-memory dependency:** the strongest latency wins assume GH200/GB200 (untestable on the dev box) — the
  PCIe baseline must be competitive or the bet is confined to premium hardware.
- **Wave-engine integration shape (1c+):** coexistence is now MEASURED (above) — viable but a fat co-resident wave
  kernel is too costly, so the shape is **either ~1-SM sidecar or full replacement of the per-batch path**, not a wide
  always-resident data-plane beside launch-per-batch. Still open: the slow-class (dependent-read, data-dependent-
  predicate) path; and the deterministic spine + MV-dependency-graph CC the writes will need (ADR-009).
- **S-B v1 tradeoffs (now lower priority — S-F is OFF):** admission holds the catalog latch during the upload + re-admits
  the whole table per commit. Only matters if auto-admit is ever turned on for an analytical/read-heavy resident case.

## Discipline reminders
GPU-native-or-it-doesn't-land; differential WITH NULL data; HAZARD on device-touching slices; **independent
adversarial audit, never self-audit**; engine crate is fmt-dirty (never crate-wide `cargo fmt`); `--gpu-reset`
DENIED, run GPU tests under `timeout`. Full list: CHARTER "Operational gotchas".
