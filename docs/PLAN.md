# PLAN — Ordered Forward Work

What to build next, in order. The **rules** are in [CHARTER.md](CHARTER.md); the **why** in
[DECISIONS.md](DECISIONS.md); the **design** in [ARCHITECTURE.md](ARCHITECTURE.md); what's **done** in
[STATUS.md](STATUS.md); the **resume baton** in [HANDOVER.md](HANDOVER.md). This doc owns **sequencing**.

## 0. The fork that orders everything
The OLTP bet (DECISIONS ADR-008) is **unproven until measured**. Two near-term tracks compete for "first":
- **Prove the bet** — build the open-loop benchmark vs tuned Postgres (§1). *Recommended first: it tells you which
  architecture work actually matters.*
- **Unblock production** — STRATA auto-admission (§2), without which the entire GPU read path is dormant.

## 1. Benchmark mandate (prove or kill the OLTP bet)
Before deep OLTP-engine investment:
- **Open-loop / offered-rate** harness measuring **p99 / p99.9 / p99.99 at a target TPS** (today's numbers are
  closed-loop / self-throttling — they cannot validate a latency bet).
- A **tuned CPU OLTP baseline on the same box** (Postgres; ideally an in-memory engine too), on a real OLTP
  workload (**TPC-C / sysbench-oltp / YCSB-A**).
- Report **split by transaction class** (CHARTER): the deterministic fast path vs the interactive slow class.
- A win = GPU beats the tuned baseline on p99 at TPS. The result tells us where GPU loses today (launch? PCIe?
  concurrency control? index?) and which of §3 to build first.

## 2. STRATA — make the GPU path the production default (DECISIONS ADR-010)
Spec: ARCHITECTURE §7 + §13.
- **S-A — vocabulary rename** (`RelationalResidentPartition → ResidentShard`, `partitions → shards`, etc.).
  Mechanical, behavior-preserving; lands the L1/L2/L3 disambiguation first.
- **S-B — admission producer v1 (N=1 unified) behind `auto_admit_on_commit` (default OFF).** Commit-triggered,
  post-`publish_committed_seq`, best-effort, via the `&self`+held-catalog-guard seam. Makes the GPU path reachable
  end-to-end via the wire for fits-one-GPU tables (int4 + text). Golden wire tests (below) go green.
- **S-C — same-GPU N>1 shards + partial-combine (int4);** incremental shard append.
- **S-D — text shard recompaction/combine** (per-shard offset rebasing + 8-alignment); removes the int4-only guard.
- **S-E — cross-shard combine (peer-copy / NCCL) → true spill / multi-GPU;** the missing scale mechanism + the
  production shard producer for over-VRAM tables.
- **S-F — flip `auto_admit_on_commit` ON** + migrate the non-resident test contracts. This is the real precondition
  for **S10d** (delete the host read path; the `FirstCudaSliceParityBackend` tests retire *with* it).

**Golden wire tests** (acceptance spec): drive SQL over the real pgwire socket
(`crates/server/tests/pgwire_roundtrip.rs` pattern), assert exact rows + that the GPU sharded route served them
(non-vacuous), differential vs 1/N shards/host, **with NULL data**, GPU-guarded.

## 3. OLTP execution engine (ARCHITECTURE §9 — gated on the benchmark telling us it's needed)
- Persistent-kernel wave engine + lock-free submission ring (replace launch-per-batch).
- Deterministic spine + MV dependency-graph concurrency control (BOHM/PWV); host sequencing materializes
  non-deterministic inputs; the order is the replication log.
- GPU index + point-access path; resident **layout decided by measurement** (PAX vs columnar).
- GPU-side WAL-record generation + tighter group-commit batching (durability is already crash-safe — STATUS).
  **Group-commit prerequisite (latent bug):** before enabling multi-entry apply, fix the 4 DDL apply helpers
  (`database_exists`, `tablespace_exists`, `relational_view_depends_on_inner`/`has_dependents` in
  `engine_ddl_objects.rs`) that read the *published* catalog while the apply loop mutates the *working* copy (safe
  today only because `to_apply ≤ 1`); add a multi-entry-batch regression test.

## 4. Correctness gates (build alongside the relevant work)
- **Durability + recovery:** WAL-flush-failure injection; rejected commits never visible; restart/recover;
  `commit ≥ applied ≥ visible` monotonicity.
- **Replication consistency:** follower write-rejection, leader promotion/demotion, catch-up, snapshot install; no
  committed entry lost, no divergence after convergence.
- **Jepsen-style fault campaign (v1 gate):** kill / partition / reorder / flush-fail; linearizability; minimized
  reproducer discipline (seed/topology/fault-schedule/WAL+watermark).
- **Oracle hygiene (CHARTER):** GPU parity uses GPU-native oracles, never a CPU re-implementation. (The old
  CPU-reference parity stream is **retired/forbidden**, not to be resurrected.)

## 5. Product backlog (verify status before starting)
Grouped, terse. Detail lives in ARCHITECTURE.
- **Unification:** consolidate the **three pgwire servers → one**; invert the `engine → protocol` dependency; flip
  the store so the engine is the single source of truth.
- **Engine core:** real MVCC (SI→**SSI**, write-write conflict detection); GPU write path (real device work on
  commit); de-monolith oversized source files (tests-first extraction).
- **GPU-resident catalog + function engine** (ARCHITECTURE §14).
- **Type-system breadth (Phase 8 review):** per-type **text AND binary wire codecs + OID/typmod** with no silent
  PG-compat gaps; typed param binding (binary + NULL); real queryable `pg_catalog`/`information_schema`; bignum
  NUMERIC >38-digit handling; SUM/AVG accumulator-overflow checked-arithmetic fix. **Exit gate: every type
  graduates to a GPU-resident route.**
- **OLTP route classes:** tenant/security-filtered page route; bounded two-table join route (package the join
  engine + fanout bound); computed-detail route with resident summaries + invalidation.
- **Charter debt:** GPU-ify the remaining CPU-execution resident routes (between/range→row-ids; text-prefix-count;
  text-project filter); adaptive sort dispatch + HAVING/LIMIT operator to replace any residual host `.sort_by`.
- **Durability / HA:** live streaming replication (network AppendEntries/RequestVote, heartbeats, leases/fencing,
  auto-failover); synchronous commit on quorum; crash-safety integration tests + object-store archival.
- **Scale:** connection scale toward 100k–1M (session admission); working-set > VRAM (STRATA spill / S-E);
  pinned-host D2H evaluation.
- **Hardening:** packaging (deb/rpm, Dockerfile, k8s/systemd); supply-chain (SBOM, cargo-audit/deny/geiger, GPU CI
  fatbins); `// SAFETY:` on every unsafe + panic reduction; observability (Prometheus/OTLP/tracing + audit log);
  mTLS + SCRAM channel binding + credential store.
- **Hot-path efficiency audit (Phase 7):** a deliberate **asymptotic** sweep of hot-path data structures —
  correctness review does NOT catch asymptotics (motivated by the live O(n²)-write `with_table_mut` deep-clone that
  passed two correctness audits yet nearly sank a milestone). A standing methodology gate, run under the Phase-5 harness.
- **Productization acceptance gate:** driver smokes (`tokio-postgres`, `sqlx`, `asyncpg`, `node-postgres`,
  `psycopg`, `pgx`, JDBC/R2DBC); `pg_dump`/`pg_restore`, `pg_dumpall --globals-only`; release-candidate preflight.

## 6. Parked / verify-before-starting
- **Two charter-debt correctness items** (STATUS "known gaps"): expression-overflow PG-divergence
  (gather-then-evaluate, `engine_expr.rs:~5757`); routing-gate case-sensitivity (`resident_route.rs:388`).
- **Modernization #21:** raise `.ptx` arch targets toward an sm_90 floor; numeric/uuid MIN/MAX two-pass →
  `atom.cas.b128`. *Verify against the tree — may be partly done.*
- **Perf residual:** async the fused `mixed_int_text` route's ~11 sync round-trips → ~2–3. Gated on the Phase-5
  open-loop harness.
- **Single-GPU multi-partition read-side decision:** the 8 partitioned probes are GPU-native; retiring them needs
  the general executor to iterate multi-partition resident tables, OR an explicit decision to keep them (the
  campaign's S10a was BLOCKED here). Decide alongside S-C/S-D.

## 7. Non-goals (until post-v1)
Cluster-reconfiguration automation; multi-region failover automation; broad extension surface; broad async-driver /
binary / COPY-streaming parity beyond the acceptance gate.
