# HANDOVER — Resume Baton

> **This is a SINGLE ROLLING file. Overwrite it each session — never date it, never accrete.** Keep it short:
> where we are, the one next action, and the open decisions. Everything else lives in the other five docs.

**Updated:** 2026-06-27.

## Where we are
- Docs are **6 canonical files**: [CHARTER](CHARTER.md), [ARCHITECTURE](ARCHITECTURE.md), [DECISIONS](DECISIONS.md),
  [PLAN](PLAN.md), [STATUS](STATUS.md), this.
- **Workload = high-throughput OLTP** (ADR-008); execution = deterministic batched waves (ADR-009); residency =
  STRATA shards + auto-admission (ADR-010).
- GPU-native **resident read path complete for int4** (S1–S10c, audited). Suite **730/0**.
- **STRATA S-A landed** — L2 rename `partition → shard` (`RelationalResidentShard`, `residency.shards`).
- **STRATA S-B landed (2026-06-27)** — the **auto-admission producer v1** (N=1 unified, behind default-off
  `auto_admit_on_commit`): a committed table becomes GPU-resident **on commit, with no explicit warm**, via the
  `&self`+held-catalog-guard seam on all 3 commit paths. Verified non-vacuously on the GPU box; HAZARD clean. PLAN §2.
- **The blocker is now smaller:** auto-admission **exists** but is **default-OFF**, so the production default is
  still host-side reads until **S-F** flips it (gated on perf + S-C/S-D/S-E). The host read path stays live; S10d
  is gated on S-F (STATUS "blocking gap").

## The one next action (pick with the user)
1. **Build the OLTP benchmark** (PLAN §1) — open-loop p99 vs tuned Postgres; the instrument that proves/kills the
   OLTP bet and tells us which architecture work matters next. *(Recommended — nothing today can validate the bet.)*
2. **STRATA S-C** — same-GPU N>1 shards + partial-combine (int4): the producer emits multiple shards; reads prefer
   partial-combine. PLAN §2.
3. **The full pgwire-socket golden test** for S-B (drive auto-admit through the real wire) — the remaining S-B
   acceptance piece (today's test is engine-level).

## Top open decisions (unresolved)
- **Data-size envelope:** OLTP working set vs aggregate VRAM, and the over-VRAM spill/tiering model (cross-shard
  combine doesn't exist; STRATA placement, not hardware paging, owns the tail).
- **Coherent-memory dependency:** the strongest latency wins assume GH200/GB200 (untestable on the dev box) — the
  PCIe baseline must be competitive or the bet is confined to premium hardware.
- **S-B v1 tradeoffs to revisit:** admission holds the catalog latch during the upload (serializes commits behind
  it) and re-admits the whole table per commit — fine for v1/default-off, but incremental + off-lock admission are
  the perf follow-ups before flipping the default (S-F). Plus STRATA §11 tactical (per-shard budget, etc.).

## Discipline reminders
GPU-native-or-it-doesn't-land; differential WITH NULL data; HAZARD on device-touching slices; **independent
adversarial audit, never self-audit**; engine crate is fmt-dirty (never crate-wide `cargo fmt`); `--gpu-reset`
DENIED, run GPU tests under `timeout`. Full list: CHARTER "Operational gotchas".
