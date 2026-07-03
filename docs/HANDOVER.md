# HANDOVER — Resume Baton

> **This is a SINGLE ROLLING file. Overwrite it each session — never date it, never accrete.** Where we are,
> the open decision, and the rules. The **why** is in DECISIONS.md; the **how** in ARCHITECTURE.md; the
> **mandate** in CHARTER.md; the **plan** in PLAN.md.

**Updated:** 2026-07-03.

---

## ⛳ THE CHARTER IS THE CONSTRAINT — READ THIS FIRST

**The GPU is the execution substrate for the ENTIRE relational data path. The host is CONTROL PLANE ONLY.**
The end goal is to **REMOVE the CPU relational engine** — it exists today only as a parity oracle + a GPU-fault
safety net, both **interim debt to be deleted**, never product direction (ADR-006).

- **Host MAY:** wire I/O; SQL parse + plan; kernel orchestration/launch; txn coordination + sequencing;
  WAL/durability I/O; the staging upload (build + upload the next device generation); the single final
  device→wire result readback.
- **Host MUST NOT:** scans, filters, joins, aggregates, sorts, grouping, DISTINCT, HAVING, LIMIT/OFFSET,
  expression eval, NULL/3VL — and MUST NOT materialize results from `host_rows`.

**THE LOAD-BEARING DIRECTIVE (user, verbatim):** *"We need to stay GPU native with the solution. Follow the
charter."* When you MEASURE a host-side cost in a data-plane hot path, the fix is to **MOVE THE WORK ONTO THE
GPU**, not to optimize the host. The ONLY authorized host work is CHARTER.md's "Host MAY" list — wire I/O,
parse+plan, kernel orchestration/launch, txn coordination+sequencing, WAL/replication I/O, the staging
upload, the final readback. **GOVERNANCE (user, 2026-07-03): a prior baton added an "index BUILD once per
generation" host-work exception THE USER NEVER AUTHORIZED — agent-authored docs must never widen the
charter; exceptions exist only if written into CHARTER.md by the user.** The host-side addressing
structures built under that unauthorized gloss (shard_pk_index et al.) are PENDING THE USER'S RULING —
see the open decision below. Memory: `stay-gpu-native-charter`.

**Success bar (trajectory bet):** same ORDER OF MAGNITUDE as a tuned CPU engine on today's hardware, with the
residual gap being GPU-ARCHITECTURAL (launch amortization, bandwidth/coherence) so it closes as hardware
advances. A gap that is host-side serial overhead is IN SCOPE TO FIX. SLO: >100k TPS sustained, ≥400k burst,
p50/p99/p99.9 < 0.5/1/5 ms.

---

## >>> THE ONE NEXT ACTION: THE CHARTER RULING (user, 2026-07-03) GOVERNS — host addressing structures were built under an UNAUTHORIZED agent-invented exception; the user ruled MIGRATE-TO-DEVICE. Program: M1 wave-batched DEVICE locate (replace host shard_pk_index probes; reuse the v2/v3 probe kernels + shard_pk_device_index) → M2 bigint PKs via a DEVICE i64 index (no host cache ever; the non-int4 proposal's layout) → M3 deletion sweep (shard_pk_index, wave_index, host zone-map decisions; ledger #20-22). SLO dips during migration are ACCEPTED — performance arguments do not create charter exceptions. In flight: the i64-section flip commit awaits its blast-radius audit verdict, then pushes. Memory: `charter-governance-ruling` (BINDING). <<<

**THE RETIREMENT PROGRAM A1→A5 IS COMPLETE; THE FLIP IS LIVE** (`fd154409`/`f0c3101e`:
`host_install_elision_enabled` + `auto_vacuum_enabled` default ON; SLO 104-124k sustained on PK-less
tables). **The post-A5 frontier is LEDGER #14 TYPE COVERAGE — the true CPU-engine-deletion gate — decomposed
into THREE user-ratified tracks (2026-07-03, memory `type-coverage-14`):**

1. **TRACK 1 — CONSTRAINED-TABLE ELISION (ACTIVE).** Any declared PK made a table elision-INELIGIBLE
   (`table_elision_eligible`'s unique-index clause), so the flagship SLO applied to ZERO realistic
   core-banking tables. MEASURED baseline (`GPU_DB_BENCH_PK=1`): **923 TPS @16w, p50 19.5ms** (the O(table)
   `prepare_insert` candidate scan, paid TWICE per concurrent commit — off-lock prepare + under-lock
   re-resolve). Slice 1 (local commit, in audit): index-driven INSERT validation (the 1b
   `validate_dml_constraints_via_index` wired into `prepare_insert`), the probe ladder SELF-PINS its views,
   `visible_relational_rows` rehydrates elided tables itself (B1 closed at the source), default-OFF
   **`constrained_elision_enabled`** extends eligibility to unique-indexed strictly-Int4 tables, and
   **incremental PK-index cache maintenance** (writer-side extension at the append chokepoint PRE-publish +
   prober-side tail-DtoH + ahead-entry slot-bound probing; `shard_pk_index` Mutex→RwLock).
   **RESULT: 923 → 77.4k TPS @16w (p50 0.19ms, p99 0.28ms) = 84×.** Found+fixed the **FACADE-SEQ POISON**
   (preflight probes at the facade txn id; rehydration seams stamped store versions with it → "tuple not
   found" for later readers; every seam now stamps at `committed_seq()`). Two GPU differentials, both
   sabotage-verified; full sweep 828/0 @87s; CPU 472/0; facade 34+13/0; clippy HEAD-parity.
   **OPEN (ledger #18):** 32w contention INVERSION (~52k stable, bistable to ~84k; maintenance counters
   healthy — suspects: probe RwLock reads, the String-keyed unique-slot conflict ledger; perf is locked
   down on this box, needs in-process instrumentation) and the >100k PK'd sustained target. Also open:
   CHECK-constraint + FK eligibility (CHECKs are row-local post-slice — likely near-free; FKs are
   cross-table elision interplay), the flag's default-flip decision.

2. **TRACK 2 — PER-TYPE COVERAGE, SECTIONS + KEYS TOGETHER** (a bigint-PK table goes end-to-end fast in one
   arc): Date/Int2 typing (near-free: they ride the i32 section; only the A4a/A4c/A3 `SqlValue` typing +
   eligibility gate them) → int8/Timestamp (i64 section + i64 key path) → numeric/uuid (i128) → bool
   (bitmap append) → text LAST (variable-length forces a new append/rollover design; the open-shard builder
   rejects it). The single-buffer payload builder + descriptor offset helpers are the type-complete
   template; the shard struct lacks int8/numeric/bool fields; the recompaction gather is int4-ordinal-only;
   the append chunk builder is i32-stride-only. Key-side: the device index hashes i32 with inline-packed
   slots that can't widen — `docs/proposals/non-int4-point-lookup-index.md` (UNACCEPTED) is the design to
   fold in (general hash-bucket layout, bigint → text → uuid/numeric phasing).

3. **TRACK 3 — COMPOUND PKs (LAST; product-scope).** A multi-column PRIMARY KEY is REJECTED AT PARSE TIME
   (`sql/lib.rs:3646` single-column destructure; PrimaryKey/Unique/FK/Index types are single-column by
   construction). Net-new surface: parser → AST → catalog → validators → wide-key index (pack primitives
   `build_wide_key`/`pack_two_int4_cols` exist). Compound PREDICATES on int4 tables already execute as
   device AND-scans with single-column opportunistic resolve + full recheck (correct, not O(1)).

**Also open on the board:** burst ≥400k (wave-level WAL/propose batching + pinned-staging scatter append —
the 4th-round profile mapped the wave budget), multi-GPU, the read-lane residuals E1/E2/R-1, ledger #15
(uncapped per-row locate loop), the concurrent-elision first-transition TOCTOU, VACUUM V2s (key-clustered
rebuild, background thread, park starvation).

**MULTI-AGENT:** the second (read-path) lane is idle since D3/D4 shipped; STILL: always
`git fetch && git rebase origin/main` before pushing; no workspace-wide cargo fmt (scope `-p gpu_db_engine`).

---

## Where we are (DONE + on origin/main; HEAD `f0c3101e` + the local track-1 slice in audit)

- **A1→A5 COMPLETE — THE FLIP IS LIVE** (elision + auto-vacuum default ON; the elided-churn SI bug
  root-caused re-pin-the-fallback-view, audited, SV6-hammer-gated). SLO on pure defaults (PK-less):
  104.1k@16w / 124.2k@32w sustained, target >100k MET; burst 138-142k (≥400k OPEN).
- **D3 stamp-all-appends + D4 generation-atomic publication** (ADR-013) shipped by the read-path lane;
  `docs/SHARD_STORAGE.md` is the as-built shard reference.
- **VACUUM #5 V1** (`ffc2eabd`): churn-triggered dense rebuild, deferred-tail auto-trigger, re-elision.
- **Perf arc (PK-less):** 13.3k dual-store → 34.2k elision → 120k+ wave-batched appends.
  **Perf arc (PK'd, track 1):** 923 → 16.5k (index-driven validation) → 77k (constrained elision) @16w.
- Read path SETTLED at its architectural ceiling (banked; do not re-litigate). Sharding = a SCALE play.

---

## The per-iteration protocol (NON-NEGOTIABLE)

1. **MEASURE first** — reproduce/confirm before changing code.
2. **Smallest correct slice behind a default-OFF flag** — byte-identical to HEAD until the flip.
3. **GPU-native solution (the charter)** — data-plane hot path on the device, not the host.
4. **Differentials**: GPU == CPU == the SQL SPEC (memory `sql-spec-over-cpu-parity`). Prove the new path
   FIRED (a counter), not a silent fallback.
5. **NON-VACUOUS sabotage-verified asserts** — break the mechanism, watch the test FAIL, revert.
6. **INDEPENDENT ADVERSARIAL OPUS AUDIT before EVERY push** — never self-audit, never push unaudited;
   adopt findings → re-verify → push. Memory `audit-with-opus-subagents`.
7. **Pair LATENCY (p50/p99) with THROUGHPUT** on every benchmark line (memory `benchmark-report-card`).
8. **Update memory** after each slice; re-read the scalability ledger each loop (no new unscalable
   hot path without a ledger row).

**GPU discipline (hard rules):** GPU tests under `timeout`; **NEVER `--gpu-reset`**; **sweeps with
`--test-threads=1`** (parallel oversubscribes → spurious failures; re-run failures in isolation);
never a GPU test right after a timeout-killed one; **ASCII-only PTX**. Push to origin/main per standing
authorization. User prefs: **BIGGER SLICES, FEWER CHECK-INS**; the USER sets the sequence at track
boundaries (memory `working-agreement-sequencing`).

---

## Pointers

- **Memory (read first):** `MEMORY.md` index at
  `~/.claude/projects/-home-richard-projects-gpu-database-engine/memory/`. Key files: `type-coverage-14`
  (THE ACTIVE PHASE), `scalability-ledger` (#17 done, #18 open — re-read each loop),
  `retirement-program-option-a` (the completed A1-A5 arc), `stay-gpu-native-charter`,
  `billions-rows-scale`, `gpu-test-threads-serial`, `benchmark-report-card`.
- **Code:** elision/eligibility + rehydration + VACUUM = `engine_residency.rs`; DML prepare + the probe
  ladder (`visible_row_with_value`, self-pinning) = `engine_dml_prepare.rs`; serialized preflight arms =
  `engine_write_apply.rs`; the PK-index cache + incremental maintenance = `engine_retained_read.rs`
  (`extend_shard_pk_index_cache_on_append`, `try_extend_cached_shard_pk_index`); waves + flush =
  `engine_dml_concurrent.rs`; unique-slot conflict ledger = `write_path.rs`; shard storage reference =
  `docs/SHARD_STORAGE.md`.
- **Benchmarks:** `oltp_commit_slo_benchmark` (knobs: `GPU_DB_BENCH_PK/ELIDE/CELIDE/ADMIT/DURABLE/WRITERS/
  SECONDS`; prints elision + pk-index maintenance telemetry), `r3_shard_index_route_ab`,
  `r2_wave_engine_ab`, `r3_insert_profile`.
- **Green baselines to preserve:** engine CPU 472/0; full GPU sweep 828/0 (`--test-threads=1`, ~87s);
  facade 34/0 + 13/0; clippy 22 warnings (HEAD parity).
