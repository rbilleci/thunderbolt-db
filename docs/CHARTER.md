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

**Success bar (clarified 2026-06-27): same ballpark on *today's* hardware, not beat-the-CPU-today.** This is a
*trajectory* bet. We win when (a) GPU OLTP is within the **same order of magnitude** as a tuned CPU engine on current
hardware, AND (b) the residual gap is **GPU-architectural** (launch amortization, parallelism, memory bandwidth/
coherence) so it **closes as GPU hardware advances** while the CPU path sits near its ceiling. A measured gap that is
**host-side serial overhead** (not GPU-bound) is in scope to fix — it is not evidence against the bet.

## Success criteria (SLO targets, mid-size core-banking deployment)
| Metric | Target |
|---|---|
| Sustained OLTP throughput | > 100,000 TPS |
| Peak burst | ≥ 400,000 TPS |
| R1 prepared bounded read p50 / p99 / p99.9 | < 0.5 ms / < 1 ms / < 5 ms |
| W1 single keyed synchronous INSERT, UPDATE, or DELETE p50 / p99 / p99.9 | < 0.8 ms / < 1.5 ms / < 5 ms |
| T8 predeclared atomic transaction p50 / p99 / p99.9 | < 1.5 ms / < 3 ms / < 10 ms |
| T32 predeclared atomic transaction p50 / p99 / p99.9 | < 3 ms / < 6 ms / < 20 ms |
| Concurrent connections | > 100,000 (up to 1,000,000) |
| RPO / RTO | 0 (no committed loss) / < 30 s failover, < 5 min full GPU recovery |

The throughput targets are aggregate **committed transactions per second** for the immutable
[`oltp-benchmark-workload-v1.md`](design/oltp-benchmark-workload-v1.md) BENCH-001 core-banking workload, not a per-
class TPS promise and not a logical-operations/s target. Its repeating 200-transaction schedule is binding:

| Class | Transactions / 200 | Canonical work |
|---|---:|---|
| R1 | 120 (60%) | one prepared bounded point/page read |
| W1 | 50 (25%) | 35 INSERT, 10 UPDATE, 5 DELETE (70/20/10 within W1) |
| T8 | 20 (10%) | the frozen eight-operation route, with exactly four mutations |
| T32 | 10 (5%) | the frozen 32-operation route, with exactly 16 mutations |

The schedule executes 650 logical operations per 200 transactions: 3.25 operations/transaction. Consequently the
same run must report more than 325,000 logical operations/s at the strict sustained target and at least 1,300,000
logical operations/s at peak; those figures are consequences of the TPS gate, not substitute acceptance units.
Read-only R1 requests count as committed read-only transactions. Workload v1 freezes the schema/cardinality, seed
and 80/20 hot/cold account selection, exact SQL and operation order, and numeric post-image/WAL-byte, index-fanout,
touched-table, cold-access, and result bounds. A different or missing manifest invalidates the comparison.

The sustained gate is the manifest's fixed, evenly paced 3,300,000-transaction warm-up followed by 66,000,000
measurement arrivals over 30+600 contiguous seconds at 110,000 scheduled TPS. Measurement-scheduled transactions
whose terminal committed completions are timestamped inside the measurement window, divided by 600, must exceed
100,000 TPS; warm-up completions are excluded. Class latency is scheduled-arrival through terminal completion, and
stage populations must finish at/below their starting values and drain to idle within one second. Peak bursts are the fixed sequence
`B01`–`B10`; each schedules exactly 400,000 arrivals over one second. Peak **cohort TPS** is eventual terminal
committed cohort count divided by that fixed arrival second, not completions timestamped inside the same second.
Every named cohort must commit all 400,000 transactions, preserve the mix, pass every class latency envelope, and
return stage populations to/below their pre-burst values within one second of the last arrival. Wall-clock completion
throughput is an additional diagnostic. Best-window extrapolation, an omitted/failed cohort, a different workload,
or standalone class saturation cannot satisfy either system throughput gate. Standalone R1/W1/T8/T32 sweeps remain
mandatory diagnostics in both TPS and logical operations/s, but have no separate charter throughput threshold.

Latency classes are explicit acceptance envelopes, not percentiles pooled across unlike work:

- **R1** is one prepared bounded point/page read through the final client-visible result.
- **W1** is one keyed autocommit INSERT, UPDATE, or DELETE through publication-covered synchronous acknowledgement.
  INSERT, UPDATE, DELETE, and the declared I/U/D mix each pass independently.
- **T8** is 2–8 predeclared relational operations with at most four mutations; **T32** is 9–32 predeclared
  operations with at most 16 mutations. Both remain within route-declared post-image/WAL bytes, maintained-index
  fanout, touched-table, cold-access, and result bounds. Work outside those bounds is not admitted under the class
  merely because its operation count fits.
- Data-dependent or client-interactive transactions remain the supported slow class. Arbitrary client think time is
  excluded from database service latency; statement latency, terminal commit/rollback latency, database-active time,
  and wall time are reported separately. There is no generic whole-transaction latency promise for this class.

Every advertised class measures open-loop latency from scheduled arrival, includes producer slip and queueing, and
passes independently in the canonical sustained and peak runs. Mixed read/write tests report R1, each W1 operation,
T8/T32, the actual achieved mix, TPS, and logical operations/s separately; their pooled latency distribution is
supplementary only. `p99.99` is reported by BENCH-001 but has no binding threshold yet. A synchronous write/
transaction profile qualifies only when its measured p50, p99, and p99.9 durability values plus percentile-matched
bounded downstream margins each fit the corresponding strict class target; otherwise the profile is explicit
non-SLO service or is refused, never silently acknowledged asynchronously. A downstream margin is a hard bound or a
joint residual distribution from the same correlated end-to-end traces; adding independently sampled stage
percentiles cannot qualify a profile. Direct open-loop end-to-end class latency is the final authority.

## The invariant — host is control plane ONLY
- Every relational decision and every result value is computed on, and read back from, the **device**.
  Decoded host rows are ingest/upload staging only and are discarded after upload; a published residency
  generation does not retain a host relational shadow.
- **Host MAY:** wire I/O; SQL parse + plan; kernel orchestration/launch; txn coordination + **sequencing**;
  WAL/durability I/O; replication; the COPY/DDL **staging upload** (build + upload the next device generation);
  the single **final device→wire result readback**.
- **Host MUST NOT:** scans, filters, joins, aggregates, sorts, grouping, DISTINCT, HAVING, LIMIT/OFFSET on data,
  expression eval, NULL/3VL — and must not materialize results from `host_rows`.
- **The engine REQUIRES a GPU** (sm_120 floor). No CPU-only / hybrid steady-state mode. Production SELECT/MVCC
  declines and GPU faults fail loud; they never execute relational work on the host. CPU relational execution is
  compiled only under `cfg(test)` as an interim parity oracle, never product direction (ADR-006/007).

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
**Data volumes larger than GPU memory are a required capability** — served by the STRATA streaming executor
(out-of-core: shards admitted on demand, host/NVMe as cold STORAGE, the GPU the sole execution tier; DECISIONS
ADR-012), never by CPU execution or hardware demand-paging.

## Execution discipline (non-negotiable)
- **GPU-native or it does not land.** No host stub committed as "done"; no "deferred/follow-up/clean-error-for-now"
  escape past a hard kernel. Interim host code a later slice will replace is **WIP, not a deferral.**
- **Every device-touching milestone candidate:** behavior-preserving where possible; **differential test WITH NULL
  data** (the S10a/S10b lesson — the deleted probes were NULL-blind); **HAZARD** protocol (`--ignored` 3×
  sequential + 2× concurrent, **zero CUDA 700/716/717**); a **separate independent adversarial audit** (never
  self-audit; prove non-vacuity by sabotage); **adopt findings, don't defer.** Private helpers and proof phases are
  WIP inside that candidate, never separately accepted slices. They receive focused tests while the production
  vertical route stays executable; the complete milestone receives the HAZARD/audit/performance seal once.
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
- **Type derivation:** the bound catalog descriptor, materialized `SqlValue`, GPU relation layout, and wire
  metadata must agree. PostgreSQL integer aggregate rules are the contract: `SUM(int2/int4)` and `COUNT(*)`
  produce `Int8`/bigint (OID 20), while `SUM(int8)` produces numeric. Never preserve a mismatched descriptor and
  attempt to repair it by inferring a transient-relation type from values.
- **Lease lifetime:** a derived device buffer's lease must outlive ALL passes / the kernel call (early free + pool
  reuse = UAF).
- **Merge workflow:** commit on the feature branch → push → checkout main → `merge --ff-only` → push → back to
  branch. Commit footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
