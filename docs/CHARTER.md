# CHARTER — GPU-Native OLTP Database

The mandate, the bet, the invariants, and the rules. The **why** is in [DECISIONS.md](DECISIONS.md); the
**how** in [ARCHITECTURE.md](ARCHITECTURE.md); the **next** in [PLAN.md](PLAN.md); the **now** in
[STATUS.md](STATUS.md); the **resume baton** in [HANDOVER.md](HANDOVER.md).

## Mission
A PostgreSQL-compatible (PG16 wire + SQL) **GPU-native** relational database for **high-throughput OLTP** —
core-banking-class workloads (ledgers, postings, transfers, balance inquiries). The GPU is the execution
substrate for the **entire** relational data path, including the system catalog; the host is **control plane only**.

## The bet (DECISIONS ADR-008, 2026-06-26)
GPU advances driven by AI demand — coherent CPU–GPU memory (NVLink-C2C), HBM scale, faster atomics — let GPU OLTP
**outpace CPU engines**. Architectural or benchmark obstacles to OLTP-on-GPU are **in scope to fix**, not a reason
to retreat to analytics. The bet is **unproven until measured** against a tuned CPU baseline (PLAN benchmark mandate).

## Success criteria (SLO targets, mid-size core-banking deployment)
| Metric | Target |
|---|---|
| Sustained OLTP throughput | > 100,000 TPS |
| Peak burst | ≥ 400,000 TPS |
| p50 / p99 / p99.9 latency (simple OLTP) | < 0.5 ms / < 1 ms / < 5 ms |
| Concurrent connections | > 100,000 (up to 1,000,000) |
| RPO / RTO | 0 (no committed loss) / < 30 s failover, < 5 min full GPU recovery |

## The invariant — host is control plane ONLY
- Every relational decision and every result value is computed on, and read back from, the **device**.
  `host_rows` is ingest-staging only.
- **Host MAY:** wire I/O; SQL parse + plan; kernel orchestration/launch; txn coordination + **sequencing**;
  WAL/durability I/O; replication; the COPY/DDL **staging upload** (build + upload the next device generation);
  the single **final device→wire result readback**.
- **Host MUST NOT:** scans, filters, joins, aggregates, sorts, grouping, DISTINCT, HAVING, LIMIT/OFFSET on data,
  expression eval, NULL/3VL — and must not materialize results from `host_rows`.
- **The engine REQUIRES a GPU** (sm_120 floor). No CPU-only / hybrid steady-state mode. CPU relational execution
  exists ONLY as (a) the **parity oracle** and (b) **operational-safety on GPU fault** — both interim GPU-parity
  **debt to be deleted**, never product direction (DECISIONS ADR-006, supersedes ADR-003).

## Transaction model (the OLTP shape — DECISIONS ADR-009)
- **Fast path = deterministic, predeclarable transaction _waves_**: PK / unique-key equality point operations +
  pre-declarable stored procedures (access set **statically derivable** from the statement; a large share of OLTP
  traffic by volume).
- **Slow class (supported, not optimized)** = anything whose access set is *not* known up front: interactive
  multi-statement with dependent reads, **and** single-statement writes over data-dependent predicates
  (`UPDATE … WHERE non_indexed_col=?`, ranges). The full pgwire surface is kept.

## Target hardware
Coherent CPU–GPU memory (GH200/GB200) is the strategic target and a first-class **fast path — NOT a hot-path
requirement**; PCIe (dev box: RTX PRO 6000) is the **testable baseline**. **Explicit residency/admission (STRATA),
not hardware demand-paging, owns placement and the p99 tail** (page faults wreck OLTP tail latency).

## Execution discipline (non-negotiable)
- **GPU-native or it does not land.** No host stub committed as "done"; no "deferred/follow-up/clean-error-for-now"
  escape past a hard kernel. Interim host code a later slice will replace is **WIP, not a deferral.**
- **Every device-touching slice:** behavior-preserving where possible; **differential test WITH NULL data** (the
  S10a/S10b lesson — the deleted probes were NULL-blind); **HAZARD** protocol (`--ignored` 3× sequential + 2×
  concurrent, **zero CUDA 700/716/717**); a **separate independent adversarial audit** (never self-audit; prove
  non-vacuity by sabotage); **adopt findings, don't defer.**
- **Parity tests use GPU-native oracles** (on-device serial-vs-parallel / construction / closed-form), never a CPU
  re-implementation as the source of truth.

## Operational gotchas (carry into every slice)
- **fmt:** the engine crate is **fmt-DIRTY at HEAD** (~632 `cargo fmt --check` diffs). **NEVER run crate-wide
  `cargo fmt` in a slice** (it reflows ~all files; ~17 files / +4000 lines; has caused false starts). Hand-format.
  Cleaning the fmt debt is a deliberate standalone commit — confirm with the user first.
- **GPU safety:** spin-locks **deadlock** (zombie context survives SIGKILL) → lock-free atomics only
  (`atom.cas` advance-on-failure / `atom.add`). **`--gpu-reset` is DENIED** (shared box). Run GPU tests under
  `timeout`. **716 misaligned load:** read a 64-bit device value as 2× `ld.global.u32` when a section can be
  4-mod-8; varlen text-offset sections must be **8-aligned**. Build ptxas tops at sm_90; runtime JITs to sm_120
  and **rejects non-ASCII PTX** (INVALID_PTX/218).
- **Tests:** `cargo test -p gpu_db_engine --lib -- --include-ignored --test-threads=1` (GPU tests are `#[ignore]`
  or guarded by a `device_memory_proof.is_none()` early-return). Box: RTX PRO 6000 (Blackwell).
- **NULL/3VL goes IN the kernel** — never a host pre/post filter/partition/overwrite.
- **Checked arithmetic (correctness contract):** int4/int8 `+ - *` are range-checked **on-device** and raise PG
  `integer out of range` / `numeric field overflow` via a shared device overflow flag — **never silently wrap,
  never CPU-fallback** (DECISIONS ADR-011). The interpreter evaluates every arithmetic sub-expr over **all** rows
  before combining masks, so a query errors if any row overflows in any conjunct — *stricter than PG* (see the
  gather-then-evaluate debt in STATUS).
- **Empty/edge PG-correctness:** the general executor **hard-errors** on an empty filtered SUM/MAX/AVG rather than
  returning the legacy probe's `Int8(0)` / empty-text sentinel; PG-correct **NULL** for empty aggregates is
  M3-gated (not yet wired on that path). Routing a shape to the bridge is a correctness fix, not byte-identical on empty.
- **Type derivation:** catalog-declared type ≠ materialized value type (e.g. `SUM(int4)` declared Int4, valued
  Int8; `SUM(int8)`→numeric) — derive transient-relation types from VALUES. **`COUNT(*)` returns `Int4`** in this
  engine, not PG's `Int8` — a real divergence drivers/tests must expect.
- **Lease lifetime:** a derived device buffer's lease must outlive ALL passes / the kernel call (early free + pool
  reuse = UAF).
- **Merge workflow:** commit on the feature branch → push → checkout main → `merge --ff-only` → push → back to
  branch. Commit footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
