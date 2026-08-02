# Project Agent Guidance

This repository is pursuing a GPU-native database engine. Future agents should
optimize for that thesis unless the user explicitly changes direction.

## GPU-Native North Star

- Treat GPU-resident execution as the product direction, not as an optional
  accelerator around a CPU-first database.
- Optimize for GPU-native OLTP: entity fetches, tenant/security-filtered page
  reads, bounded joins, and computed detail routes, not only analytical scans or
  primary-key microbenchmarks.
- Prefer designs where hot data, lookup structures, encoded columns, and read
  snapshots live in GPU memory.
- The CPU is the host/control plane ONLY: wire protocol, SQL parse/plan,
  transaction coordination, WAL/durability I/O, and GPU orchestration. CPU
  relational execution exists solely as parity-reference plus temporary
  bootstrap scaffold, tracked as debt with a GPU milestone — never product
  direction and never the hot-path design.
- The catalog is GPU-native and joins are GPU operators: `pg_catalog` and
  `information_schema` are GPU-resident system relations, and catalog joins
  (`psql \d`, ORM introspection) run on the GPU join path. Do not build a CPU
  catalog or CPU nested-loop/hash join as the target answer.
- For hot reads, prefer prepared route ids, typed parameters, resident snapshot
  handles, and device-ready projection plans over repeated SQL-text parsing.
- Favor immutable/versioned GPU-resident snapshots for read concurrency.
  Serialize mutation and generation publication until a stronger MVCC model is
  intentionally designed.
- Optimize batching for throughput, but do not make batching the only latency
  answer. GPU-native low latency likely requires concurrent read execution over
  resident snapshots.

## Architecture Bias

When choosing between implementation approaches:

1. Keep the GPU hot path explicit and measurable.
2. Preserve CPU/GPU semantic parity, but treat CPU relational execution as
   parity-reference/bootstrap debt with a milestone — never product direction.
3. Avoid adding CPU caches or CPU indexes as the primary answer for benchmark
   wins unless the change is clearly documented as a non-GPU-native escape
   hatch.
4. Prefer principled concurrency boundaries: immutable read snapshots,
   serialized writers, CUDA stream ownership, epoch/generation retirement.
5. Be cautious with heuristic scheduler complexity. If a policy becomes hard to
   explain, consider a simpler split between latency-oriented prepared reads and
   throughput-oriented batch routes.

## Model and Agent Routing

- The primary agent inherits the model and reasoning selection from the active Codex session or host. Every
  unspecified subagent uses GPT-5.6 Terra with `xhigh` reasoning.
- Use the `worker` agent for bounded implementation, fixes, tests, and verification. It uses Terra with `xhigh`
  reasoning and may inherit the parent write permissions.
- Use the `architect` agent for ambiguous, cross-subsystem architecture, transaction, WAL/recovery, concurrency,
  GPU-residency, and ownership decisions. It uses Sol with `max` reasoning and is read-only.
- Use the `acceptance_auditor` agent for the independent acceptance gate. It uses Sol with `max` reasoning and is
  read-only.
- Use the `explorer` agent only for read-only search, inventory, evidence extraction, and log triage. It uses Luna
  with `medium` reasoning and must return any coding work to the parent.
- Any task that creates or modifies source, tests, build scripts, benchmark harnesses, migrations, generated code,
  or patches is coding. Never perform or delegate coding below `xhigh` reasoning. `max` also satisfies this
  requirement. If the required model or reasoning level is unavailable, stop before coding and report the blocker
  instead of silently substituting a lower setting.

## Documentation Expectations

`docs/PLAN.md` is the **only** document that owns open, deferred, blocked, or sequenced work. `STATUS.md`
records facts; `HANDOVER.md` is a short pointer to active PLAN IDs; architecture, ADR, and `docs/design/`
documents do not own tasks. Everything under `docs/archive/` is historical and non-actionable even when it
contains words such as `NEXT`, `TODO`, or `OPEN`.

Permanent analysis instrumentation lives behind the build-only `probe-timing` Cargo feature. Reuse and extend
those probes instead of writing and reverting one-off hot-path timers.

Before major runtime, storage, or scheduler changes, read:

- `docs/CHARTER.md` — mandate, invariants, the OLTP bet, execution discipline + gotchas
- `docs/ARCHITECTURE.md` — the full system design (execution model, residency/STRATA, deterministic OLTP, MVCC,
  durability, multi-GPU)
- `docs/DECISIONS.md` — the decision ledger (ADRs)
- `docs/PLAN.md` (ordered work) · `docs/STATUS.md` (current state) · `docs/HANDOVER.md` (resume baton)
- `docs/CODE_SIZE.md` — source-size thresholds, decomposition method, reference updates, and exceptions

When a change intentionally favors CPU-first behavior, document why it is a
fallback, bootstrap step, or product-scope exception.

## Source File Size and Module Boundaries

Follow `docs/CODE_SIZE.md`. Production source over 3,000 lines and test/example/tool source over 4,500 lines
requires an audited disposition; any file over 7,500 lines must remain owned by a PLAN task until it is split or
accepted in the exception registry. New modules should normally remain below 2,250 lines.

Split by invariant and ownership, not by line range. Preserve stable facades, move the closest tests, update
module/import/re-export/build/test/doc references in the same slice, and verify old paths and symbols are gone.
Do not create `part1`/`part2` shards, catch-all modules, dependency cycles, or broad visibility solely to make a
split compile. Keep behavior changes separate from structural extraction and run the gates prescribed by the
standard and the affected subsystem.

## Development gate order

### End-to-end delivery discipline

- The unit of acceptance is a user-visible `PLAN.md` milestone, not a private helper, proof type, codec phase,
  source extraction, or production-unreachable sub-boundary. Intermediate work is integrated WIP: keep it tested,
  but do not call it accepted, add a `STATUS.md` acceptance entry, advance `HANDOVER.md`, or run a standalone
  acceptance audit merely because it can be reviewed in isolation.
- Start every feature milestone with one production-reachable vertical route and keep that route executable while
  breadth is added. A private or `cfg(test)` prerequisite may exist only when a failing end-to-end test proves it is
  required. It must be connected to the production call graph in the same milestone candidate; source guards whose
  purpose is to preserve production unreachability are prohibited.
- Measure progress by closed end-to-end acceptance rows, production call-graph reachability, and deletion of the
  superseded path. Commit count, slice count, test count, lines added, proof types, and audit rounds are not progress
  measures.
- Use one integrated candidate and one milestone acceptance cycle. Focused tests, sabotage, static checks, and
  optional advisory reviews run during implementation. Freeze once after the complete functional matrix passes,
  then run the required independent adversarial audit and applicable full card. A valid audit finding reopens that
  same candidate; it does not create a new accepted sub-milestone.
- For an explicitly time-boxed milestone, `PLAN.md` must carry elapsed-time checkpoints and a production-reachability
  stop-loss. Ninety minutes without a newly passing production end-to-end assertion or removal of a superseded live
  branch requires stopping helper expansion and returning to the shortest failing vertical route.
- Parallel agents are encouraged only as lanes inside the same milestone candidate. Give them disjoint ownership
  such as live cutover, durability/recovery, and acceptance evidence; integrate at least every 90 minutes. Do not
  assign separate agents to invent or accept the next micro-boundary.
- New prerequisite milestones, wire-format versions, or generalized authorities may be added only when an existing
  milestone acceptance test fails for their absence and the current design cannot satisfy it. Record that evidence
  in the existing PLAN item; do not grow a prerequisite chain from architecture preference alone.

Quality attaches to the exact accepted milestone candidate, not to repeated full-card runs on intermediate repairs.
Use this order once per complete PLAN milestone:

1. Use the preceding accepted comparable card as the before-baseline. Rerun the base revision only when the device,
   driver/runtime, Rust/C toolchain, release profile, lockfile/native inputs, report-card harness, or calibrated
   workload changed enough to make the accepted artifact non-comparable.
2. Iterate with focused correctness/static gates, required NULL differential and HAZARD coverage, and
   `scripts/benchmark_report_card.sh --quick` when read performance may move. Quick mode is a clean-build A+B screen;
   it is never acceptance evidence.
3. Freeze the candidate, then delegate the independent read-only adversarial audit to the project-scoped
   `acceptance_auditor` custom agent defined in `.codex/agents/acceptance-auditor.toml`. That agent is pinned to
   GPT-5.6 Sol with max reasoning; do not silently substitute another model or reasoning level. If that configuration
   is unavailable, stop before acceptance and report the blocker. Audit implementation, focused gates, and sabotage
   evidence, and repair/re-audit findings before paying for the full card. A canonical full run permits staged
   changes but rejects unstaged or untracked files, builds an exported snapshot of the exact staged tree, and
   invalidates itself if the candidate changes.
4. Run `scripts/benchmark_report_card.sh --full` exactly once for the provisionally accepted candidate whenever the
   full card is applicable. The auditor then verifies completeness, provenance, and baseline comparison before final
   ACCEPT. A card is complete evidence, not an automatic performance verdict.
5. Any code repair after the full card creates a new candidate. Rerun affected gates and audit; rerun the card when
   the repair touches a read kernel, residency/layout, successful point-read route, result path, allocator/runtime
   dependency, release/link/code-placement setting, or benchmark harness. Otherwise record why the existing card
   remains applicable, as with a documentation-only or fail-path-only repair.

Do not weaken the full card to accelerate development: keep its fresh target, both layers, both cache regimes,
calibrated row/batch counts, exact artifact identities, GPU cool-downs, and exclusive GPU run. The speedup comes from
screening early, auditing before Section C, reusing a comparable accepted baseline, and running the full seal once.

## Read-path performance regression benchmark (standard)

There is ONE standard read-kernel benchmark; run it before/after any change that touches a
read kernel, the residency layout, or the result path, and compare to the baseline below.

```
timeout 300 cargo run --release --example read_kernel_roofline -p gpu_db_execution
#   (no GPU? it prints a skip line. Never use --gpu-reset. ROWS=/SORT_N=/ITERS= override.)
```

It drives each read-kernel FAMILY directly on a resident column and reports effective GB/s
(bandwidth-bound scans / gather) or M-elem/s (sort / join / grouped). The **roofline is `sum_i32`**
(a pure 1-pass read+reduce = the HBM streaming peak). NOTE: `equal_any` is also measured but is NOT
the roofline -- its 8-needle per-element compare is compute-bound, ~2x slower than a pure read.

WHAT IS KERNEL-CLEAN: only section (1) (resident-input scans) is wall ~= kernel. gather/sort/join
take a HOST slice and upload it to the device EVERY call (per-call input H2D the engine does NOT pay
-- its inputs are device-resident), so their wall OVERSTATES the kernel; each line is H2D-labeled.
The single-launch gather kernel is additionally isolated via the CUDA event (~6us vs ~885us wall).
The GROUP BY line shows the aggregate KERNEL (event-timed, ~5ms) AND its full result path (~467ms;
the ~99% non-kernel tail = the 8M-row index H2D + ~2*row_count slot-table setup + host Vec build).

**Compare by RATIO, not absolute GB/s.** Absolute bandwidth varies by GPU/driver, so the portable
signal is `kernel_GB_s / sum_i32_roofline_in_the_same_run`. A material drop in a kernel's ratio
(or in the algorithmic M-elem/s) vs the baseline is a regression to investigate.

Baseline (8M i32 rows, captured 2026-06-30; roofline `sum_i32` was ~1486 GB/s on that box):

| family | kernel | ratio-to-roofline (or Melem/s) | note |
|---|---|---|---|
| scalar reduce (ROOFLINE) | `sum_i32` | 1.00 (~1486 GB/s) | pure 1-pass read+reduce = HBM peak |
| 8-needle scan (NOT roof) | `equal_any` | ~0.43 | compute-bound, ~2x a pure read |
| scan-project ordered (1% sel) | `project_compare` | ~0.12 (2-pass) | near roof/pass — SATURATED |
| ordered compaction (50% sel) | `compare_indices_ordered` | ~0.024 (2-pass) | output-bound — expected |
| arith VM | `arith_filter` | ~0.023 (2-pass) | ok |
| constant mask output | `const_mask_false` | ~0.62 IN-L2 / ~0.98 OUT-OF-L2 | 1-pass i32 device fill; no host vector/H2D or result D2H (~37us / ~188us) |
| **scalar COUNT (reduce)** | `count_i32_compare` | **~0.85** | block-reduced (was KNOWN HEADROOM) |
| | `count_i32_between` | **~0.44** | calls count x2 |
| filter -> indices | `expr_i64_compare_scalar` (1% sel) | ~0.11 | ok (8B col) |
| | `expr_i128_compare_scalar` (1% sel) | ~0.19 | ok (16B col) |
| gather (scattered) | `gather_i32` kernel-only / wall | ~350 GB/s / ~2.4 GB/s | wall = +~4MB idx H2D |
| algorithmic | sort / join / grouped-KERNEL | ~321 / ~259 / ~1679 Melem/s | sort/join wall = +key H2D; grouped = event-timed kernel (full path ~18) |

The scalar COUNT reductions (`count_i32_compare`, `count_i32_between`, `equal_count`) used to be the
KNOWN headroom target (a per-thread `red.global.add` on one counter = ~N serialized atomics); they
are now block-reduced and near roof in this baseline (`count_i32_compare` ~0.85; `count_i32_between`
~0.44 since it calls count twice). (The i64/i128 FILTERS earlier looked "slow" only as a
100%-selectivity output artifact; at ~1% sel they are fine.) Treat a ratio FALLING below the
baseline as the regression signal.

The constant-mask line was added on 2026-07-13 after removing an inherited O(rows) host `Vec<i32>` plus H2D
upload. Its throughput denominator is the four output bytes written per row, not input bytes read; it retains
the synchronized VM mask on device without compaction or result D2H. Compare both cache regimes and treat a
return toward the former ~3.3 GB/s / 10,106us IN-L2 / 80,271us OUT-OF-L2 behavior as host-staging regression.

### Standard benchmark report card (BOTH layers x BOTH cache regimes)

The roofline above is Layer 1 only. The canonical, recurring artifact is the **full report card**, which
ALWAYS reports BOTH layers x BOTH cache regimes, p50 latency + throughput on every line:

```
scripts/benchmark_report_card.sh --full # no arguments is a backward-compatible alias
```

The card deliberately ignores an ambient `CARGO_TARGET_DIR`: Cargo native build-script fingerprints do not
necessarily invalidate when the compiler executable changes version, and a stale native archive can change the
final executable's link layout enough to distort this latency-sensitive benchmark. By default the script builds both
examples once in a fresh `target/benchmark-report-card.*`, records host/toolchain/git identity plus SHA-256 and size
for both exact binaries and the AWS-LC native archive, invokes those binaries directly for every section, and removes
the isolated target at exit. `BENCH_TARGET_DIR` is allowed only when the caller supplies an empty, caller-owned
directory; `BENCH_KEEP_TARGET=1` retains an automatically created target for diagnosis. Do not accept a card built
from a shared or pre-populated target. The runner serializes report-card invocations with a process-wide GPU lock,
forces the calibrated full-card workload regardless of ambient benchmark variables, requires configuration-bound
machine-readable completion from each example, skips expensive Section C when A or B is incomplete or the candidate
drifts, and returns nonzero for incomplete or candidate-drifted runs. Acceptance requires the final exact
`report_card_execution_status=complete mode=full sections=A,B,C canonical=true` record; example and section
`complete` markers alone are not acceptance evidence.

- **Layer 1 -- RAW READ KERNELS** (`read_kernel_roofline`, crates/execution): emits IN-L2 (32MB/col,
  8M rows) AND OUT-OF-L2 (256MB/col, 64M rows) in ONE invocation.
- **Layer 2 -- lpb/wave POINT-READ PATH** (`r2_wave_engine_ab`, crates/engine): batched equal-any
  point reads = the production default route, run IN-L2 (Section B, 1M rows) and OUT-OF-L2 (Section C).

**This card's L2 = 128 MB** (cudaDevAttrL2CacheSize, RTX PRO 6000 Blackwell Max-Q). OUT-OF-L2 needs the
gathered i32 column (4B/row) to exceed 128MB => > 32M rows. Section C uses **48M rows = 192MB/col
(1.5x L2)** by default.

**BUILD-TIME NOTE (why Section C has its own timeout):** `r2_wave_engine_ab` builds its table via a SQL
INSERT loop (CPU-bound SQL parse + txn/MVCC apply; in-memory WAL, no fsync). The final R3-004 tree retains
each device-authoritative insert publication instead of late-converting the finished fixture: a 2026-07-18 run
measured 1620.4s insert + 0.0s final residency, and the former 1200s limit expired during the build. Section C
therefore gets `SECTION_C_TIMEOUT=2400` (Sections A/B keep 280). The INSERT-chunk size was measured
non-helpful (250/1000/10000 all ~11 us/row -- the cost is the engine's per-row apply, not per-statement
overhead), so the lever is the timeout, not the chunk. Full mode fixes 48M rows, 300 measured batches, and
12-second cool-downs; operational timeout overrides can only make a section fail, not weaken its workload.
`GPU_GAP` is a quick-screen-only diagnostic override. Never `--gpu-reset`.
