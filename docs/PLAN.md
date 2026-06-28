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
**v1 landed (2026-06-27):** `engine/examples/oltp_auto_admit_ab` (per-op A/B) + `facade/examples/oltp_batched_read_scaling`
(concurrent batched-vs-host). First findings in DECISIONS ADR-008 — they already redirected priority: the point-read
gap is **host-side serial coalescer cost (~15µs/item)**, so §3's wave engine / coalescer fix is the measured critical
path, ahead of more resident-route breadth (S-C/S-D/S-E). Still owed below:
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
- **S-A — vocabulary rename ✅ DONE (2026-06-27).** `RelationalResidentPartition → RelationalResidentShard`,
  `residency.partitions → shards`, `partition_device_memory → shard_device_memory`, `partition_id/count → shard_*`,
  route shapes `partitioned_* → sharded_*` (engine + observability; MVCC tuple-store "partition" left intact).
  Behavior-preserving, suite 729/0.
- **S-B — admission producer v1 (N=1 unified) behind `auto_admit_on_commit` ✅ DONE (2026-06-27).**
  Commit-triggered (all 3 commit paths), post-`publish_committed_seq`, best-effort (never fails the durable
  commit), via the `&self`+held-catalog-guard seam (`populate_relational_residency_snapshot_inner` /
  `admit_..._inner`). Default OFF → behavior-preserving (suite 730/0). Acceptance test proves a CREATE+INSERT(with
  NULL) table is GPU-resident with **no explicit warm**, reads on the GPU route, results match the host path;
  HAZARD clean. Remaining: the full pgwire-socket golden test; flipping the default is **S-F**.
- **S-C — same-GPU N>1 shards + partial-combine (int4);** incremental shard append.
- **S-D — text shard recompaction/combine** (per-shard offset rebasing + 8-alignment); removes the int4-only guard.
- **S-E — streaming executor + cross-shard combine → out-of-core (single-GPU) and multi-GPU spill (DECISIONS
  ADR-012).** The committed mechanism for working sets **> GPU memory**: a fold over shards (admit → push-down →
  combine partial → evict → next, prefetching ahead), host/NVMe as the cold STORAGE tier, the GPU the sole execution
  tier. Replaces the current recompact-ALL-shards-to-unified read path with push-down-to-shard + combine. **Hard
  precondition for S10d** (alongside S-F): deleting the host *execution* path is unsafe for over-VRAM relations until
  this exists. Cross-GPU (peer-copy / NCCL) moves only partials, never full data.
- **S-F — flip `auto_admit_on_commit` ON** + migrate the non-resident test contracts. This is the real precondition
  for **S10d** (delete the host read path; the `FirstCudaSliceParityBackend` tests retire *with* it).
  **HELD OFF (2026-06-27), evidence-gated:** the OLTP benchmark (DECISIONS ADR-008 "First measurement") shows resident
  GPU point reads lose to host (even batched, 11×) and per-commit re-admit costs O(rows) — flipping is net-negative
  today. Flip only once the read path is in the CPU ballpark (kill the coalescer per-item cost → wave engine, §3).
  Empirically flipping breaks 19/731 tests; that contract migration rides with the flip.

**Golden wire tests** (acceptance spec): drive SQL over the real pgwire socket
(`crates/server/tests/pgwire_roundtrip.rs` pattern), assert exact rows + that the GPU sharded route served them
(non-vacuous), differential vs 1/N shards/host, **with NULL data**, GPU-guarded.

## 3. OLTP execution engine (ARCHITECTURE §9 — the benchmark says this is the critical path for the bet)
- **Point-read coalescer (the measured bottleneck, DECISIONS ADR-008):**
  - **Tier 1 ✅ DONE (2026-06-27):** per-shape resident-read template (prepare once, reuse across needles) — removed the
    per-request plan/bind. 68k → 156k ops/s (2.3×), now scales; CPU gap 11× → ~4.5×. The template is the ingress the
    wave engine reuses. (Tier 2 = parallelize the single coalescer thread is **dropped** — the wave engine replaces it.)
  - **Residual:** ~6µs/item = result materialization + oneshot distribution, still single-coalescer.
- **Persistent-kernel wave engine + lock-free submission ring** (replace launch-per-batch); on-GPU result slots remove
  the remaining host per-item orchestration — the real path to the CPU ballpark (moves the bottleneck host→GPU, where
  it scales with hardware). **Started (2026-06-27): design + infra recon done.** Reuses Tier-1's template as the request
  descriptor + the existing CUDA FFI / context / module-cache / stream-pool / pinned-host pool / int4 `equal_any` kernel.
  Greenfield = (1) host-pinned **device-mapped** ring (`cuMemHostAlloc`+`DEVICEMAP`, add `cuMemHostGetDevicePointer`),
  (2) persistent kernel loop, (3) **clean-exit doorbell**, (4) result-slot layout. **Biggest risk = the exit** on this
  `--gpu-reset`-denied shared box (a hung kernel zombies the context). Mitigation: doorbell **plus a hard iteration-cap
  backstop** so the kernel ALWAYS self-terminates; single block (1 SM); lock-free atomics only. **Increment staging:**
  **1a ✅ DONE (2026-06-27)** = bare lifecycle proven: `crates/execution/examples/wave_lifecycle_probe.rs` — a persistent
  kernel polls a device-mapped doorbell, advances a heartbeat, and EXITS on the doorbell in **~3.5µs** (with a
  `%globaltimer` 30s wall-clock backstop as the zombie-prevention net). **9/9 clean lifecycles across 3 processes, no
  zombie context.** The biggest risk (clean exit on the `--gpu-reset`-denied box) is de-risked. Add
  `cuMemHostGetDevicePointer` to the engine FFI for 1b. **1b ✅ DONE (2026-06-27)** =
  `crates/execution/examples/wave_dataplane_probe.rs` — persistent-kernel threads lock-free-claim requests
  (`atom.add`, no barriers), scan a resident key column, gather a payload, and write `(value<<32)|done` as one atomic
  8-byte store; host enqueues needles + reads packed slots (no per-request host materialization). **The host-serial
  bottleneck is GONE — the bottleneck moved host→GPU (scales with hardware = the bet).** Independently audited: the
  number is REAL (reproduced ~9.5–10.1M across 20 runs), correctness SOUND (proven non-vacuous via sabotage variants);
  fixed a `membar.sys` ordering gap (held the number) + broadened not-found sampling. **Honest scan-knee curve (8192
  threads):** small tables are atomic-ceiling-bound (4k:10.0M, 20k:8.3M req/s), but the full-scan is O(rows) so
  100k:2.7M, 500k:755k, **1M-row table: 485k req/s** (~3× the 156k cap, ~CPU-ballpark, scan-bound). *Caveats:* bare
  data plane (no slot→wire mapping yet — parallelizable, not serial); vs the 156k batcher is apples-to-oranges
  (omits the full facade + neutral mapping).
- **1c ✅ DONE (2026-06-27)** = `crates/execution/examples/wave_index_probe.rs` — a GPU hash index (host-built
  open-addressing, Fibonacci hash, in-kernel probe + gather, bounded probe count) **removes the O(rows) scan →
  ~10.5M point lookups/s FLAT across 1M/4M/16M-row tables (O(1)), ~13.6× the CPU's 770k, independent of table size.**
  Atomic-ceiling-bound now. **This validates the OLTP point-read bet at the data-plane level: the GPU does millions
  of lookups/s at realistic scale, residual bottleneck is GPU-architectural (scales with HW).**
- **1d-i ✅ DONE (2026-06-27)** = `crates/execution/examples/wave_devatomic_probe.rs` — moved the claim/`completed`
  atomics to **device memory** (last completer sets a host-mapped `all_done` flag; device counter DtoH-verified):
  **read ceiling 10.5M → ~30M req/s (~2.9×), 3× stable, all correct.** New limiter = device-atomic contention on the
  single counter (peaks at LOW thread count, 512–1024; → batched/striped claiming next). Also quantified the
  **slot→wire mapping**: 200k packed slots → neutral rows in ~180µs–1ms single-threaded (~200M–1.1B rows/s), far below
  the ~7ms GPU drain + embarrassingly parallel → the bare-data-plane caveat is MINOR. (NB the `all_done` cross-thread
  ordering wants an independent audit before it's lifted into the engine.)
- **1d-ii ✅ DONE (2026-06-27)** = `crates/execution/examples/wave_batchclaim_probe.rs` — **batched claiming**: each
  thread reserves K requests per `atom.add(claim, K)` + bumps `completed` once/batch under one `membar.sys`. Sweet spot
  **K=8 → ~45–53M req/s** (~1.5–1.75× over 1d-i's 30M, **~5× the original 10.5M, ~60–69× the CPU**), 3× stable. K is a
  balance (K≥32 collapses: fewer batches than threads → under-parallel + serial host-mapped writes). **Read ceiling is
  now firmly tens-of-millions; further gains need a structural lever (sharded per-block counters / cheaper result
  writes) — diminishing, tuning-sensitive. The read half of the bet is SETTLED.**

### Write path (the frontier) — probing
- **Write probe 1 ✅ DONE (2026-06-27)** = `crates/execution/examples/wave_index_insert_probe.rs` — **concurrent
  lock-free index INSERT** (the novel, historically-hard piece: many threads `atom.cas.b64`-install (key,row) into a
  shared open-addressing table, no locks). All keys verified inserted exactly once (no lost/dup/torn). **~tens of
  BILLIONS of inserts/s** (1M: ~29G/s wall-clock, ~77G/s after removing the ~21µs launch floor; 16M/256MB L2-spill:
  4.2G/s). **Conclusion: GPU concurrent index maintenance is NOT a bottleneck.** The remaining write constraints
  (durability/WAL fsync, deterministic CC) are host-I/O + coordination problems CPU OLTP engines face too — the GPU
  isn't disadvantaged there. *Caveats:* low contention (sequential keys + Fibonacci spread); raw insert only (no
  commit/durability/MVCC/CC); synthetic keys. → **NEXT write probes:** contended inserts; the **commit/durability**
  path (group commit — the likely real write floor); deterministic CC for conflicts.
- **SM-coexistence gate ✅ DONE (2026-06-27)** = `crates/execution/examples/wave_coexist_probe.rs` — the R2 prerequisite
  (the recon's #1 unknown). A persistent kernel + concurrent engine scans on one shared context COEXIST cleanly (no
  deadlock/starvation/zombie, 188 SMs), but the SM-reservation cost is steeply non-linear (1 SM ~2%, 8 SMs ~60%, 32 SMs
  ~87% of concurrent scan throughput; busy-spin ≈ gentle ⇒ co-residency cost, not poll traffic). **⇒ the wave kernel is a
  ~1-SM sidecar OR replaces the per-batch path; never a fat co-resident** (DECISIONS ADR-008 "R2 SM-coexistence gate").
- **`all_done` ordering audit ✅ DONE (2026-06-27):** two independent adversarial auditors (split cumulativity GAP vs
  SOUND, converged) → **the sound completion gate is the host ACQUIRING the `completed` counter (DtoH `==requests` before
  reading slots = the proven 1b pattern); `all_done` is only a wake hint, never the correctness gate** (DECISIONS
  ADR-008 "R2 `all_done` ordering audit"). Carry this rule into (iv).
- **R2.1 wave read engine ✅ DONE (2026-06-27)** = `crates/execution/src/wave.rs` (new child module) — the proven 1d
  data-plane lifted in-crate as `WaveReadEngine`, running the persistent kernel on the engine's SHARED primary context.
  Multi-wave `submit` over a circular lock-free ring; completion GATED on the DtoH `completed` counter-acquire (audit
  rule), `all_done` = wake hint; doorbell + `%globaltimer` backstop + clean Drop. Kernel = the audited probe kernel with
  one change — `atom.cas` claim (bounded, no overshoot) vs `atom.add` — so cumulative `head` works across waves (the
  audited result/completion ordering path is unchanged). GPU test vs a CPU oracle (2 waves, clean exit) + ASCII guard
  pass; NOT yet wired into any query path.
- **R2.2a wave multi-projection ✅ DONE (2026-06-27):** `WaveReadEngine::submit` now gathers up to 4 int4 columns
  (R1's unrolled gather) over the ring and returns `CudaI32BatchProjectionRow`s byte-identical to the R1 index probe
  (GPU oracle test green); `atom.cas` bounded claim makes cumulative multi-wave work.
- **R2.2 freeze ROOT CAUSE PINNED ✅ (2026-06-27, `execution/examples/wave_freeze_probe`, DECISIONS ADR-008):** the
  freezer is **`cuMemAlloc`/`cuMemFree` (device-synchronizing), NOT the stream type.** The probe shows every interleave
  op keeps the persistent kernel ALIVE in µs except `cuMemAlloc+cuMemFree`, which blocks ~the backstop and kills it
  (it device-syncs, waiting for the never-ending kernel). Consequence: a co-resident wave kernel dies the instant the
  engine does a synchronizing alloc. The device-buffer **pool amortizes** alloc (steady-state reuses; syncs only on cold
  growth / overflow free), and the wave path is alloc-free → **R2.2 path: pre-warm the pool + suppress pool shrink while
  a wave kernel is resident** (pragmatic), or move engine device alloc to `cuMemAllocAsync` (robust). "Replace per-batch"
  alone is insufficient (other engine activity still allocs).
- **R2.2 PROPER PORT — verdict FLIPS: the wave WINS at small (OLTP) batches (2026-06-27, `5e6b2302`+`bcc12af5`,
  DECISIONS ADR-008 "R2.2 PROPER PORT").** The first "wave loses 118x, park" was a naive per-needle port measured in the
  wrong regime vs the wrong baseline (retracted). The proper port = optimized drain (clamped-batched CAS claim + amortized
  membar, 520k->2.27M/s) + async `submit_async`/`harvest` + host-mapped `completed` mirror on its own cacheline
  (fixed a false-sharing latency pathology). Single-flight, 1M rows, wave vs lpb: **batch1 124k/39k=3.20x, batch8
  448k/309k=1.45x, batch32 1.21M/1.19M=1.02x**, batch256 0.30x, batch65536 0.10x. The persistent kernel WINS at small
  batches (no per-batch launch: 8us/submit vs lpb ~25us) — exactly the OLTP point-lookup regime; lpb wins only at large
  batches (GPU-bound 23M vs the wave's ~2.27M drain ceiling, CAS-contention-limited). At batch 8-32 the wave is already
  3-8x the batcher's 156k. (Single-flight improved further with the device result ring + per-slot gate below.)
- **R2.2 device result ring + P2 per-slot gate / depth-K pipelining ✅ DONE (2026-06-28) — CONCURRENT PREMISE VALIDATED.**
  Device result ring + separate-stream DtoH harvest lifted single-flight drain to ~31.8M (exceeds lpb across the sweep);
  the follow-up-review correctness gates landed (C1 occupancy clamp, C2 u64 counters, C3 status-0 assert); **P1
  needles-to-device TRIED + REJECTED** (needle read is 4B/coalesced/L2-cacheable, not the cap — ~31.8M is the realistic
  in-crate ceiling for the row-materializing workload; 45M was a simpler single-u64 probe). **P2 per-slot status gate
  (sm_70 `st.release.sys`) replaces the single-flight cumulative gate → depth-K pipelining (harvest any order):** the
  offered-rate benchmark shows pipelining 1.15–2.0× over single-flight, **2.6–4.6× lpb / 9–32× the 156k batcher in the
  concurrent regime** — the regime the wave exists for, now directly benchmarked. Independent re-review concurs (qualitative
  result robust; reviews `docs/reviews/r2.2-wave-port-followup-review.md` + the P2 follow-up).
- **THROUGHPUT HEADROOM (≈5–10×) — realize IN R2.2b, do NOT chase in an isolated probe.** The offered rate **saturates at
  depth-4** because a SINGLE host thread's submit+harvest loop is the cap (3–6M mid/large batch), far below the GPU's ~30M
  single-flight drain ceiling. The lever is **multi-producer host submission** (concurrent ring with atomic head
  reservation, many producer threads) — which real OLTP connections provide for free. So realize it as part of the wiring
  and measure end-to-end, not as another synthetic multi-thread probe.
- **R2.2b = wire `WaveReadEngine` into the engine read path** (behind the existing default-OFF flag, with R1's lpb index
  probe as the fallback) and run the END-TO-END offered-rate A/B vs R1, real connection concurrency = the multi-producer
  load — the ship/no-ship decision. **Wiring BLOCKERS (must land in R2.2b):** (1) enforce the in-flight bound
  (un-harvested needles ≤ ring_capacity, else circular slots clobber → wrong rows); (2) a host-side harvest DEADLINE
  (a stalled wave must not spin forever); (3) crash/`SIGKILL`-safe shutdown (a killed process leaves the persistent
  kernel zombied until the 30s backstop, perturbing other GPU tenants — the `--gpu-reset`-denied risk in practice).
  **NEXT = R2.2b (above), or R3 writes (independent).**
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

### 4b. CPU relational-engine retirement (ADR-006) — sequenced deletion plan (scoped 2026-06-28)
ADR-006 deletes CPU relational execution; it survives ONLY as parity oracle + GPU-fault fallback. Survey finding:
the big deletion can't start now because **the GPU read path is the FALLBACK, not the default** (residency
auto-admit / STRATA S-F is default-OFF, so the host SELECT executor is the live serving path), and the **entire
write/commit path is host-only** (no GPU-native commit exists). The gates are perf + two unbuilt programs, not
idle deletable code. Named ADR-006 targets: `finalize_relational_select` (host SELECT executor), the MVCC
`cpu_fallback` (`CpuMvccExecutionBackend`), `FirstCudaSliceParityBackend` (test oracle). Durability/WAL/replication
stay (control plane). Three tiers:
- **Tier 1 — NOW (no perf gate):** migrate the ~57 CPU-as-oracle tests to GPU-native oracles (~46 are mechanical —
  they already carry a hardcoded expected literal, so delete the `let cpu = …` + `assert_eq!(.., cpu.rows)`; ~3 are
  pure CPU-diff needing a fresh closed-form oracle). Files: `tests/{mvcc_query,mvcc_provenance,sql_catalog,sql_dml}.rs`;
  choke point `tests/common.rs FirstCudaSliceParityBackend`. PREP only — the assertions guard the LIVE fallback, so
  they can't fully drop until the path under test is gone (Tier 3a). Effort **M**.
- **Tier 2 — gated on R3 (GPU-native writes + deterministic CC):** delete the host write/commit/MVCC-store/CC
  (`engine_write_apply.rs`, `engine_dml_*`, `engine_commit.rs` apply paths, `storage` mutate/visibility,
  `write_path.rs` ledger). R3 not started. Effort **L** (largest tier).
- **Tier 3a — gated on S-F (GPU read path = default; itself gated on read perf = the wave/R2.2c work):** delete the
  host SELECT executor (`engine_select_bind.rs finalize_relational_select` + the cpu_pinned path) + `CpuMvccExecutionBackend`
  + the backend-chain CPU fallback. BLOCKED FIRST by building GPU-native impls for the read ops still CPU-only:
  HAVING, ORDER-BY-over-aggregates, cross-type compare, PG-exact AVG, `FollowValueKeyRef*`, non-int4
  aggregates/DISTINCT/sort. Then drop the orphaned parity assertions + `FirstCudaSliceParityBackend`. Effort **L**.
- **Tier 3b — gated on the GPU-fault-recovery ADR:** remove the `GpuUnavailable/QueueSaturated/MemoryPressure` → CPU
  fallback. Needs either the §15 GPU-health→failover path built (large, target-only today) OR a policy decision that
  a GPU fault returns a client error. Effort **M**.
Order: Tier 1 now → Tier 3a-prep (GPU impls for the CPU-only read ops, in parallel with wave/R2.2c) → flip S-F →
Tier 3a delete → Tier 2 after R3 → Tier 3b after the recovery ADR. **The single biggest gate is S-F (read perf),
which R2.2c feeds.**

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
