# HANDOVER — Resume Baton

> **This is a SINGLE ROLLING file. Overwrite it each session — never date it, never accrete.** (11 dated
> handovers became sprawl; that is what this discipline prevents.) Keep it short: where we are, the one next
> action, and the open decisions. Everything else lives in the other five docs.

**Updated:** 2026-06-26.

## Where we are
- Docs consolidated to **6 canonical files**: [CHARTER](CHARTER.md), [ARCHITECTURE](ARCHITECTURE.md),
  [DECISIONS](DECISIONS.md), [PLAN](PLAN.md), [STATUS](STATUS.md), this.
- **Workload decided: high-throughput OLTP** (DECISIONS ADR-008); execution model = deterministic batched waves
  (ADR-009); residency = STRATA shards + auto-admission on commit (ADR-010).
- The GPU-native **resident read path is complete for int4** (S1–S10c, audited). Suite **729/0**.
- **The blocker:** nothing auto-admits tables to GPU residency, so production reads still run host-side — the
  entire GPU read path is dormant in production until STRATA auto-admission lands (STATUS "blocking gap").

## The one next action (pick with the user)
The two highest-leverage moves, both pointing at the OLTP bet:
1. **Build the benchmark first** — an open-loop / offered-rate p99 harness vs **tuned Postgres** on the same box,
   on a real OLTP workload (TPC-C / sysbench-oltp). This is the instrument that proves or kills the bet; nothing
   today can validate it. *(Recommended first — it tells you which architecture work actually matters.)*
2. **STRATA S-A → S-B** — the vocabulary rename, then the auto-admission producer (N=1, behind a default-off flag),
   which makes the GPU path reachable end-to-end and unblocks S10d. PLAN §3.

## Top open decisions (big-picture, unresolved)
- **Data-size envelope:** OLTP working set vs aggregate VRAM — and the over-VRAM spill/tiering model (cross-shard
  combine doesn't exist; STRATA placement, not hardware paging, must own it).
- **Coherent-memory dependency:** the strongest latency wins assume GH200/GB200 (untestable on the dev box) — the
  PCIe baseline must be competitive or the bet is confined to premium hardware.
- **STRATA §11 tactical:** per-shard byte budget; N=1 unified vs "just one shard"; combine-GPU selection;
  admission synchrony; egress format.

## Discipline reminders
GPU-native-or-it-doesn't-land; differential WITH NULL data; HAZARD on device-touching slices; **independent
adversarial audit, never self-audit**; engine crate is fmt-dirty (never crate-wide `cargo fmt`); `--gpu-reset`
DENIED, run GPU tests under `timeout`. Full list: CHARTER "Operational gotchas".
