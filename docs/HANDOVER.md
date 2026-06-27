# HANDOVER — Resume Baton

> **This is a SINGLE ROLLING file. Overwrite it each session — never date it, never accrete.** Keep it short:
> where we are, the one next action, and the open decisions. Everything else lives in the other five docs.

**Updated:** 2026-06-27.

## Where we are
- Docs are **6 canonical files**: [CHARTER](CHARTER.md), [ARCHITECTURE](ARCHITECTURE.md), [DECISIONS](DECISIONS.md),
  [PLAN](PLAN.md), [STATUS](STATUS.md), this.
- **Workload = high-throughput OLTP** (DECISIONS ADR-008); execution = deterministic batched waves (ADR-009);
  residency = STRATA shards + auto-admission on commit (ADR-010).
- The GPU-native **resident read path is complete for int4** (S1–S10c, audited). Suite **729/0**.
- **STRATA S-A landed (2026-06-27):** the L2 vocabulary rename `partition → shard`
  (`RelationalResidentShard`, `residency.shards`, `shard_device_memory`, `sharded_*` route shapes; engine +
  observability; the MVCC tuple-store "partition" namespace was deliberately left intact). Behavior-preserving,
  729/0. PLAN §2.
- **The blocker still stands:** nothing auto-admits tables to GPU residency, so production reads run host-side —
  the GPU read path is dormant in production until **S-B** (auto-admission) lands (STATUS "blocking gap").

## The one next action (pick with the user)
1. **STRATA S-B** — the auto-admission producer v1 (N=1 unified, behind a default-off `auto_admit_on_commit` flag),
   commit-triggered, post-`publish_committed_seq`, via the `&self`+held-catalog-guard seam. Makes the GPU path
   reachable end-to-end via the wire and unblocks S10d. PLAN §2. *(The natural continuation of S-A.)*
2. **Build the benchmark first** — open-loop p99 vs tuned Postgres on a real OLTP workload, to prove/kill the
   bet before deeper OLTP-engine investment. PLAN §1.

## Top open decisions (unresolved)
- **Data-size envelope:** OLTP working set vs aggregate VRAM, and the over-VRAM spill/tiering model (cross-shard
  combine doesn't exist; STRATA placement, not hardware paging, must own it).
- **Coherent-memory dependency:** the strongest latency wins assume GH200/GB200 (untestable on the dev box) — the
  PCIe baseline must be competitive or the bet is confined to premium hardware.
- **STRATA §11 tactical:** per-shard byte budget; N=1 unified vs "just one shard"; combine-GPU selection;
  admission synchrony; egress format.

## Discipline reminders
GPU-native-or-it-doesn't-land; differential WITH NULL data; HAZARD on device-touching slices; **independent
adversarial audit, never self-audit**; engine crate is fmt-dirty (never crate-wide `cargo fmt`); `--gpu-reset`
DENIED, run GPU tests under `timeout`. Full list: CHARTER "Operational gotchas".
