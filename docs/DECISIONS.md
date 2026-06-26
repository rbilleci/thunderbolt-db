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
- **Consequences:** Renames `RelationalResidentPartition → ResidentShard`. Unblocks host-read-path deletion once
  admission is the default. Over-VRAM tables spill across shards (needs cross-shard combine — not yet built).

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
