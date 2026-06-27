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

## The one next action — **engine integration R1** (design done; recon cited below)
Wire the GPU index into the engine the LOW-RISK way (defers the persistent-kernel SM-coexistence risk to R2):
- **R1 — index in the engine, behind a default-OFF `wave_engine_enabled` flag, NO persistent kernel:**
  1. Add the flag (copy `auto_admit_on_commit`: `engine/src/lib.rs:311`).
  2. Extend `RelationalResidencySnapshot` (`relational_model.rs:197`) with `wave_gpu_index_ptr: Option<u64>` + `wave_hash_shift`.
  3. Build the GPU hash index on admission (host-build + HtoD in `populate_relational_residency_snapshot`, reuse the probe).
  4. **Crux:** an index-probe kernel that emits the EXISTING `CudaI32EqualAnyProjectSubmission` format (result path unchanged).
  5. Guard the swap in `submit_resident_int4_equal_any_payload` (`engine_residency.rs:479`): flag-on + index present → probe; else scan.
  - Gates: differential vs scan (byte-identical, **WITH NULL**), HAZARD, **independent audit**. Win: per-batch GPU cost O(rows)→O(1).
- **R2 — persistent kernel + ring** (breaks the host-serial coalescer cap → proven 10–30M), **gated by an SM-coexistence
  measurement** (wave kernel + concurrent engine kernels on one shared context = the recon's #1 unknown). FFI
  `cuMemHostGetDevicePointer` + the `all_done` ordering audit land here.
- **R3 — writes** (concurrent index maintenance proven fast) + deterministic CC.
Integration recon cites: facade dispatch `facade/src/lib.rs:522`; snapshot `relational_model.rs:197`; scan-vs-index
`engine_residency.rs:479`; retained-read `engine_retained_read.rs:16`; GpuPrimaryContext+FFI `execution/src/lib.rs:182,545`;
threading `server/src/lib.rs:66`. The batcher stays the default until the integrated wave path wins end-to-end.
*Parked (verify before starting):* open-loop offered-rate + tuned-Postgres baseline (the real end-to-end OLTP-fitness
instrument, incl. the WRITE path which is still unmeasured); batched-mixed int4+text; STRATA S-C/S-D/S-E.

## Top open decisions (unresolved)
- **Data-size envelope:** OLTP working set vs aggregate VRAM, and the over-VRAM spill/tiering model (cross-shard
  combine doesn't exist; STRATA placement, not hardware paging, owns the tail).
- **Coherent-memory dependency:** the strongest latency wins assume GH200/GB200 (untestable on the dev box) — the
  PCIe baseline must be competitive or the bet is confined to premium hardware.
- **Wave-engine integration shape (1c+):** how the persistent kernel coexists with the current launch-per-batch
  engine (one always-resident wave kernel? SM reservation on the shared box?); the slow-class (dependent-read,
  data-dependent-predicate) path; and the deterministic spine + MV-dependency-graph CC the writes will need (ADR-009).
- **S-B v1 tradeoffs (now lower priority — S-F is OFF):** admission holds the catalog latch during the upload + re-admits
  the whole table per commit. Only matters if auto-admit is ever turned on for an analytical/read-heavy resident case.

## Discipline reminders
GPU-native-or-it-doesn't-land; differential WITH NULL data; HAZARD on device-touching slices; **independent
adversarial audit, never self-audit**; engine crate is fmt-dirty (never crate-wide `cargo fmt`); `--gpu-reset`
DENIED, run GPU tests under `timeout`. Full list: CHARTER "Operational gotchas".
