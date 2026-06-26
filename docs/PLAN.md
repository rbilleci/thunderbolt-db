# THE PLAN — GPU-Native Database (Unified Forward Plan)

**This is the single source of truth for what to build next and in what order.** Start here each session.
It is charter-based and begins with the active design (STRATA, doc 23). It supersedes all session handovers
(removed) and the *sequencing* in the older roadmap docs (retained for detail, marked superseded). Last
consolidated: **2026-06-26**.

> Authoritative for **order**. Detailed specs are linked inline; this doc owns sequencing, those own detail.

---

## 1. Charter (non-negotiable) — from doc 22 §1-3 and doc 00

- **Invariant:** every relational decision and every result value is computed on, and read back from, the
  **device**. `host_rows` is ingest-staging only.
- **Host MAY (control plane):** wire I/O; SQL parse + plan; kernel orchestration/launch; txn/WAL/replication;
  the COPY/write/DDL **staging upload** (build + upload the next device generation); the single **final
  device→wire result readback**.
- **Host MUST NOT:** scans, filters, joins, aggregates, sorts, grouping, DISTINCT, HAVING, LIMIT/OFFSET on
  data, expression eval, NULL/3VL — and must not materialize results from `host_rows`.
- **The engine REQUIRES a GPU** (sm_120 floor). No CPU-bootstrap / no GPU-absent path. **Zero deferrals:** a
  slice is **GPU-native or it does not land** — no host stub shipped as "done," no escape hatch past a hard
  kernel. Interim host code a later slice will replace is WIP, not a deferral.
- **Every device-touching slice:** behavior-preserving where possible; differential test **with NULL data**;
  **HAZARD** protocol; **separate independent adversarial audit** (never self-audit); adopt findings, don't defer.

---

## 2. Current state (2026-06-26)

- **Resident read path is GPU-native for int4** — doc 22 campaign S1–S10c landed + audited (TEXT projection,
  GROUP BY/ORDER BY/DISTINCT/HAVING/LIMIT, joins V1–V3, on-device oracles, partitioned recompaction). Suite **729/0**
  (`cargo test -p gpu_db_engine --lib -- --include-ignored --test-threads=1`).
- **THE GAP that gates everything next (verified 2026-06-26):** there is **no production producer of GPU
  residency.** `residency.partitions` is written only by a test helper; the operator warm path makes only a
  unified single buffer and is operator-triggered. So a committed table is **non-resident by default and reads
  run host-side** (`execute_relational_select_cpu_pinned` → `finalize_relational_select`). The host read path is
  therefore **live**, and doc 22 **S10d (delete the host path) cannot run** until admission exists.
- Also: the current sharded read path requires all shards on one GPU (`cuMemcpyDtoD` recompaction), so it does
  not yet relieve memory pressure; true spill needs cross-shard combine (absent).

---

## 3. IMMEDIATE — STRATA (latest design → start here). Spec: `architecture/23-strata-resident-shard-data-plane.md`

STRATA makes the charter *reachable in production*: a commit-triggered **admission producer** that lays a table
down as 1..N GPU **shards** (L2; distinct from a future SQL `PARTITION BY` = L1), read via **push-down + cross-shard
combine**, host as control plane only. Sequence:

- **S-A — vocabulary rename** (`RelationalResidentPartition`→`ResidentShard`, `partitions`→`shards`, etc.).
  Mechanical, behavior-preserving. Lands the L1/L2/L3 disambiguation before anything builds on it.
- **S-B — admission producer v1 (N=1 unified) behind `auto_admit_on_commit` (default OFF).** Commit-triggered,
  post-`publish_committed_seq`, best-effort, via the `&self`+held-catalog-guard seam. Makes the GPU path reachable
  end-to-end via the wire for fits-one-GPU tables (int4 + text). **Golden wire tests** (doc 23 §9) go green here.
- **S-C — same-GPU N>1 shards + partial-combine (int4).** Producer emits multiple same-GPU shards; reads prefer
  partial-combine for aggregates, recompaction for projection. Incremental shard append.
- **S-D — text shard recompaction/combine** — per-shard offset rebasing + bytes-blob concat + 8-alignment (not
  pure DtoD → host DtoH→add→HtoD on the small offsets section, or a tiny rebase kernel → HAZARD). Removes the
  int4-only guard. Files: `execution/src/lib.rs`, `engine_residency.rs::resident_snapshot_for_unified`,
  `engine_expr.rs::execute_resident_sharded_via_general`.
- **S-E — cross-shard combine (peer-copy/NCCL) → true spill / multi-GPU.** Per-shard partials cross GPUs, never
  full data. Unblocks over-budget admission. Pairs with the **production partition producer** (splits an
  over-VRAM table into shards — does not exist; was deferred as roadmap-v2).
- **S-F — flip `auto_admit_on_commit` ON** + migrate the non-resident test contracts. This is the real
  precondition for **doc 22 S10d** (host read path becomes disused → deletable). Note the
  `FirstCudaSliceParityBackend` tests guard the **live** MVCC CPU-fallback dispatch and retire **with** the host
  path, not before.

**Open decisions to settle first (doc 23 §11):** per-shard byte budget; whether N=1 stays a unified fast-path or
"just one shard"; combine-GPU selection; admission synchrony (inside vs just-after the commit critical section);
how far to push the compact device→wire egress path now.

**Golden wire tests** = drive SQL over the real pgwire socket (`crates/server/tests/pgwire_roundtrip.rs` pattern),
assert exact rows + that the GPU sharded route served them (non-vacuous), differential vs 1/N shards/host, with
NULL data; GPU-guarded (`#[ignore]`).

---

## 4. Correctness gates (merge of `testing/parity-and-jepsen-plan.md`)

Required before claiming v1; build alongside the relevant engine phases:
- **Durability + recovery:** WAL-flush-failure injection; rejected commits never become visible; restart/recover
  from the persisted WAL; `commit ≥ applied ≥ visible` monotonicity. Gates §5 "real WAL durability."
- **Replication consistency:** follower write-rejection, leader promotion/demotion, lagging-follower catch-up,
  snapshot install + watermark advance; no committed entry lost, no divergence after convergence. Gates §5 HA.
- **Jepsen-style fault campaign (v1 gate):** process kill, partition, reorder/drop, flush-failure; linearizability
  checks; minimized reproducer discipline (seed/topology/fault-schedule/WAL+watermark snapshots).
- **Oracle hygiene (charter):** GPU parity tests use a **GPU-native oracle** (on-device serial reference or
  closed-form expected), **never a CPU re-implementation**. (The plan's old "CPU reference mode" stream is the
  anti-pattern — rewrite it to this form.)

---

## 5. Product milestones backlog (merged; verify status before starting)

Sequencing is here; milestone DETAIL stays in `roadmap/prototype-to-production-plan.md` (P0–P8, §9 deferred-work)
and `roadmap/gpu-native-oltp-roadmap.md` (route classes) — both marked "sequencing superseded by this doc."

- **Unification (p2p §9.1/9.2):** consolidate the **three pgwire servers → one**; invert the `engine → protocol`
  dependency; flip the store so the engine is the single source of truth; engine-result→wire encoding + txn/error
  mapping + golden coverage; retire the legacy server in stages.
- **Engine core:** real **MVCC** (per-txn snapshots, commit-ts oracle, write-write conflict detection, SI→SSI);
  real **commit durability** (fsync `flush_all` + LSN + CRC + parent-dir fsync); **GPU write path** (real device
  work on commit); de-monolith `engine/lib.rs` and oversized source files (p2p §9.6, tests-first extraction).
- **GPU-resident catalog + function engine** (doc 20 — catalog-as-relations Part A; intrinsics/inlining/PTX-JIT
  function execution Part B).
- **Type-system breadth (p2p Phase 3):** NUMERIC(p,s), int8/int2/bool, timestamp[tz]/date, uuid, bytea, float8,
  varchar, json/jsonb — each with text+binary codecs + OID; server-side typed param binding (binary + NULL);
  real queryable `pg_catalog`/`information_schema`.
- **OLTP route classes (gpu-native-oltp M5–M7):** tenant/security-filtered **page route** (M5); bounded two-table
  **join route** as the route-level packaging of the existing join engine + fanout bound (M6); **computed-detail
  route** with resident summaries + invalidation rules (M7). Carry the 6-route taxonomy + per-route benchmark
  policy (p50/p95/p99, queue/GPU-wall/H2D/D2H/fallback; never count a CPU cache/index win as GPU-native).
- **Charter debt (p2p §9.4/§9.5):** GPU-ify the 3 remaining CPU-execution resident routes (between/range→row
  indices; text-prefix-count; text-project filter). Sort operator S4/S5 (adaptive dispatch + HAVING/LIMIT operator)
  then wire it to replace any residual host `.sort_by`/HAVING/LIMIT.
- **Expression-overflow PG-divergence (charter correctness):** `ORDER BY`/`GROUP BY <expr>` evaluates the
  expression over **all** rows *before* WHERE drops survivors, so a query whose only overflowing rows are
  filtered out **errors where PG (survivors-only) succeeds**. Fix = gather-then-evaluate
  (`engine_expr.rs:~5757` `arith_value_column_at_indices`, shared with the WHERE-arith VM). Distinct from the
  Phase 8 SUM/AVG accumulator-overflow review.
- **Durability / HA (p2p Phase 4):** live streaming replication (network AppendEntries/RequestVote, heartbeats,
  election timers, leases/fencing, auto-failover); synchronous commit on fsync'd quorum; **group commit + WAL-fsync**
  (incl. the S1 prerequisite: fix the 4 DDL apply helpers reading published-vs-working catalog before multi-entry
  apply); crash-safety integration tests + object-store archival.
- **Scale (p2p Phase 5):** the **open-loop / offered-rate p99/p99.9/p99.99 steady-state harness** (does not exist
  — gates trusting any latency number); connection scale toward 100k–1M (session admission); **working-set > VRAM**
  (ties directly to STRATA spill / S-E); pinned-host D2H evaluation.
- **Hardening (p2p Phase 6):** packaging (deb/rpm, Dockerfile, k8s/systemd); supply-chain (SBOM, cargo-audit/deny/
  geiger, GPU CI fatbins); `// SAFETY:` on every unsafe + unwrap/panic reduction; Prometheus/OTLP/tracing + audit
  log; mTLS + SCRAM channel binding + credential store (remove static-salt bootstrap).
- **Audits (p2p Phase 7/8):** hot-path data-structure efficiency audit under the Phase-5 harness; type-system
  PG-fidelity × GPU-representation review (bignum NUMERIC >38-digit handling; SUM/AVG overflow checking).
- **Productization acceptance gate (from v0-v1):** driver smokes — `tokio-postgres`, `sqlx`, `asyncpg`,
  `node-postgres`, `psycopg`, `pgx`, JDBC/R2DBC; `pg_dump`/`pg_restore`, `pg_dumpall --globals-only`; the
  release-candidate preflight / evidence-bundle scripts.

---

## 6. Parked / verify-before-starting

- **Modernization #21** — raise `.ptx` arch targets toward an sm_90 floor; check whether current ptxas targets
  sm_120 natively; migrate numeric/uuid MIN/MAX two-pass → single `atom.cas.b128` CAS loop. *May be partly done —
  verify against the tree.*
- **NULL 3VL breadth tails** — nullable composite/expression GROUP BY key; nullable ORDER BY expression + explicit
  `NULLS FIRST/LAST`; OUTER+WHERE per-side pushdown. *May have landed in the doc-22 campaign — verify.*
- **Perf residual** — async the fused `mixed_int_text` route's ~11 sync round-trips → ~2–3. Gated on the
  (not-yet-built) Phase-5 perf harness; parked.
- **Routing-gate case-sensitivity** — ORDER BY/GROUP BY column matching is case-**in**sensitive in the route
  gate (`resident_route.rs:388`, `engine_select_exec.rs:39`) but case-sensitive in the executor, so a
  case-mismatch clean-errors instead of resolving. Align eventually (low priority; never wrong rows).

---

## 7. Non-goals (until post-v1, from v0-v1)

Cluster-reconfiguration automation; multi-region failover; broad extension surface; broad async-driver / binary /
COPY-streaming parity beyond the acceptance gate above.

---

## 8. Working discipline & gotchas (non-negotiable)

- **fmt:** the engine crate is **fmt-DIRTY at HEAD** (~632 `cargo fmt --check` diffs). **NEVER run crate-wide
  `cargo fmt` in a slice** — it reflows ~all files (~17 files / +4000 lines; has caused false starts). Hand-format
  additions. Cleaning the fmt debt is a deliberate standalone commit — confirm with the user first.
- **HAZARD protocol** (kernel/PTX-touching): `--ignored` 3× sequential + 2× concurrent (engine‖execution),
  **ZERO CUDA 700/716/717**. `atom.cas.b128` paths are historically hazard-prone.
- **GPU safety:** GPU spin-locks **deadlock** (zombie context survives SIGKILL) — lock-free atomics only
  (`atom.cas` advance-on-failure / `atom.add`). **`--gpu-reset` is DENIED** (shared box). Run GPU tests under
  `timeout`. **716 misaligned load:** read a 64-bit device value as 2× `ld.global.u32` when a section can be
  4-mod-8 (varlen text offsets after a data-dependent blob); varlen offset sections must be **8-aligned**.
- **Tests:** `cargo test -p gpu_db_engine --lib -- --include-ignored --test-threads=1` (GPU tests are `#[ignore]`
  or guarded by a `device_memory_proof.is_none()` early-return). Box: RTX PRO 6000 (Blackwell). Build ptxas tops
  at sm_90; runtime JITs to sm_120 and **rejects non-ASCII PTX** (INVALID_PTX/218).
- **Independent adversarial audit** — separate `general-purpose` subagent, never self-audit; give it the parent
  SHA to diff + a relentless mandate; use a `git worktree` (leave the main tree clean); prove non-vacuity by
  **sabotage**. Caught real silent-wrong-answer P0s every campaign. Adopt findings, don't defer.
- **Differential WITH NULL data** — the deleted probes were NULL-blind; always include NULL/tie/boundary/empty.
- **NULL/3VL goes IN the kernel** — never a host pre/post filter/partition/overwrite.
- **Empty/edge PG-correctness:** legacy probes returned `Int8(0)` for empty SUM / empty-text sentinel for empty
  MAX; the general executor returns PG-correct **NULL**. Routing a shape to the bridge is a correctness fix, not
  byte-identical on empty/NULL.
- **Type derivation:** catalog-declared type ≠ materialized value type (e.g. `SUM(int4)` declared Int4, valued
  Int8) — derive transient-relation types from VALUES.
- **Lease lifetime:** a derived device buffer's lease must outlive ALL passes / the kernel call (early free +
  pool reuse = UAF).
- **Merge workflow:** commit on `phase0-m1-engine-facade` → push → checkout main → `merge --ff-only` → push →
  back to branch. Commit footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## 9. Document map

- **Forward plan (this doc):** `docs/PLAN.md` — order of work, charter, gotchas. Start here.
- **Active design spec:** `architecture/23-strata-resident-shard-data-plane.md` (STRATA).
- **Campaign record:** `architecture/22-full-gpu-native-read-path.md` (resident read path S1–S10c; forward work
  now lives here in PLAN).
- **Other design specs (reference):** doc 17 general executor, doc 18 SQL→Expr, doc 19 type matrix, doc 20
  catalog/function engine, doc 21 NULL/3VL, doc 00 principles, doc 01–16 system design.
- **Detail roadmaps (sequencing superseded by this doc; kept for detail):** `roadmap/prototype-to-production-plan.md`,
  `roadmap/gpu-native-oltp-roadmap.md`, `roadmap/v0-v1.md` (early baseline), `roadmap/implementation-log.md`
  (history), `testing/parity-and-jepsen-plan.md`.
- **Archived (dead-premise, no forward content):** `docs/archive/` — `no-nvidia-bootstrap-plan`,
  `no-gpu-bootstrap-closeout-review` (the project now requires a GPU; their CPU-bootstrap premise is void).
- **Reference (not plans):** `research/` (concluded architecture search + literature), `architecture/00–21`,
  `adr/`, `interfaces/`, `operations/`, `compatibility/`, `testing/reports/` (dated run evidence).
