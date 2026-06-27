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
- **Wave engine (ADR-009) 1a + 1b PROVEN** (isolated standalone probes `crates/execution/examples/wave_*_probe.rs`):
  **1a** = persistent kernel clean **~3.5µs doorbell exit** + `%globaltimer` backstop (the `--gpu-reset`-box risk is
  de-risked). **1b** = parallel data plane (lock-free claim, on-GPU scan+gather, packed atomic result) = **~9.8M point
  lookups/s, ~60× the cap, all gathers verified — the host-serial bottleneck moved host→GPU (= the bet).** NOT yet
  integrated into the engine; the batcher remains the production default.

## The one next action — **wave 1c** (PLAN §3)
Move the hot claim/`completed` atomics to **device memory** (host-mapped PCIe atomics are the current bound) + build the
**slot → neutral-result mapping** (parallelizable), then **integrate the wave path into the engine behind a default-OFF
flag** — request descriptor = Tier-1's `RelationalRetainedReadTemplate`. Gates: differential vs the batcher **WITH NULL**,
HAZARD, **independent adversarial audit**. The batcher stays the default until the integrated wave path wins end-to-end.
*Parked (verify before starting):* open-loop offered-rate + tuned-Postgres baseline (the real OLTP-fitness instrument);
batched-mixed int4+text (needs coalescer-thread CUDA-context fix or the wave engine); STRATA S-C/S-D/S-E.

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
