# HANDOVER — Resume Baton

> **This is a SINGLE ROLLING file. Overwrite it each session — never date it, never accrete.** Where we are,
> the open decision, and the rules. The **why** is in DECISIONS.md; the **how** in ARCHITECTURE.md; the
> **mandate** in CHARTER.md; the **plan** in PLAN.md.

**Updated:** 2026-07-05.

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

## CURRENT BATON — Durable Write Conveyor / Future WAL Basis

**Branch:** `codex/durable-write-throughput` (not merged). **User instruction:** stop optimizing for now, wrap up,
and prepare handover. Do not merge without explicit permission.

**Scope of this baton:** the `gpu_db_write_conveyor` WAL/conveyor prototype and benchmark harness, not the engine
SQL commit path. It is being shaped as the future WAL basis: single-owner append, sequential recoverable blocks,
Chronicle-style visible/durable split, scan recovery, manager segment rolling, background durability probes, and raw
durability isolation.

**Latest default OLTP durable lane (validated 2026-07-05):**

```text
CONVEYOR_MODE=file-wal-manager-coalesced-durable CONVEYOR_EVENTS=500000 CONVEYOR_CLIENTS=128
file-wal-manager-coalesced-durable        0.143 M/s  7002.48 ns/write  elapsed=3.501s clients=128 workers=1 wal-backend=manager wal-block=64 durable-sync-mode=write-and-file-data durable-syncs=4017 blocks/sync=2.0 sync-avg=0.821ms recover-final-durable=0.048s
client->logged        p50=0.022ms p90=0.029ms p99=0.047ms max=9.661ms
client->data-fenced-wal+volatile-store-applied p50=0.878ms p90=0.916ms p99=2.493ms max=0.015s
```

Clean same-default comparison with the older mmap-range fence:

```text
CONVEYOR_MODE=file-wal-manager-coalesced-durable CONVEYOR_EVENTS=500000 CONVEYOR_CLIENTS=128 CONVEYOR_DURABLE_SYNC_MODE=range-and-file-data
file-wal-manager-coalesced-durable        0.128 M/s  7829.14 ns/write  elapsed=3.915s clients=128 workers=1 wal-backend=manager wal-block=64 durable-sync-mode=range-and-file-data durable-syncs=4012 blocks/sync=2.0 sync-avg=0.922ms recover-final-durable=0.049s
client->data-fenced-wal+volatile-store-applied p50=0.958ms p90=1.006ms p99=2.914ms max=0.021s
```

Current defaults that produced this:
- client-latency modes default `CONVEYOR_WORKERS` to `1` unless explicitly overridden; worker/bulk modes still default
  workers to producer count.
- durable client modes default `CONVEYOR_DURABLE_SYNC_MODE=write-and-file-data` (descriptor write of the WAL byte range
  plus `sync_data`) because it beat the older mmap `range-and-file-data` path in the same-default filesystem-backed
  OLTP lane comparison above.
- coalesced manager durable mode still uses `CONVEYOR_CLIENT_WAL_BLOCK=64`, `CONVEYOR_CLIENT_APPEND_GROUP_US=25`,
  `CONVEYOR_DURABLE_GROUP_US=25`, and adaptive `CONVEYOR_DURABLE_MIN_BLOCKS=0`.

**Positive levers from the last round:**
- Reducing apply-worker fanout was real. 127 apply workers added scheduler pressure. In 500k/128-client runs:
  default-old 127 workers was around `0.130 M/s`, p90 ~`1.120ms`; one worker reached `0.141-0.143 M/s`, p90
  `0.916-1.113ms` depending on sync mode and storage variance.
- Switching the default durable fence from mmap range flush to `write-and-file-data` was real in client-shaped runs:
  all-sample 500k/128/default-workers run reached `0.143 M/s`, p90 `0.916ms`; the same-default `range-and-file-data`
  comparison reached `0.128 M/s`, p90 `1.006ms`; the comparable `sync-write-data` probe was about `0.134-0.140 M/s`,
  p90 `1.109-1.116ms`.
- Reduced latency sampling (`CONVEYOR_CLIENT_LATENCY_SAMPLES=4096`) can improve throughput a little, but all-sample
  remains the default and still shows p90 < 1ms under the new durable default.

**Negative/closed levers from the last round:**
- Removing the bounce-buffer copy and writing directly from the mmap range regressed both raw one-block fence and
  client durable benchmarks. It was rolled back.
- `CONVEYOR_DURABLE_MIN_BLOCKS=1` and `CONVEYOR_DURABLE_GROUP_US=0` were worse; they created too many small fences and
  pushed client p90 near `3ms`.
- Larger durable window (`75us`) also lost p90/throughput.
- Multiplexed async clients (`CONVEYOR_CLIENT_DRIVER_THREADS=16`) did not improve p90; durable fence still dominated,
  and client scheduler delay became visible.
- Same-device durable striping remains negative; keep it as a multi-device/global-cut hypothesis, not a production
  default.
- Synchronous one-op io_uring raw probe was negative for low-latency durable commits. It is only competitive in
  large-group Chronicle throughput regimes and should not be promoted without a deeper queued design beating the raw
  page-cache baseline.

**Validation already green for the current slice:**
- `cargo test -p gpu_db_write_conveyor -- --nocapture`
- `cargo check -p gpu_db_write_conveyor --example write_conveyor_bench`
- `cargo clippy -p gpu_db_write_conveyor --all-targets -- -D warnings`
- `git diff --check -- crates/write_conveyor/examples/write_conveyor_bench.rs docs/WRITE_CONVEYOR.md`
- Latest no-override benchmark command shown above.

**Audits:**
- Laplace audited the worker-default slice: no high severity issues; medium doc ambiguity fixed by making
  `CONVEYOR_WORKERS` an optional override, adding a current-default benchmark, and labeling historical workers=8 /
  `range-and-file-data` examples.
- Euler audited the durable-sync-default/docs slice: no code-level durability regression found in `write-and-file-data`.
  Medium benchmark-evidence issue fixed by adding the same-default `range-and-file-data` comparison and aligning the
  interpretation numbers. Low follow-ups remain: `CONVEYOR_DURABLE_SYNC_MODE` parsing is not mode-gated, so invalid
  durability env can fail unrelated modes; `CONVEYOR_BACKGROUND_DURABLE`/`CONVEYOR_DURABLE_LANES` can be silently
  ignored in direct client modes and should become warnings or validation errors if those knobs remain benchmark-facing.

**Files most relevant to the current baton:**
- `crates/write_conveyor/examples/write_conveyor_bench.rs`
- `crates/write_conveyor/src/wal_segment.rs`
- `crates/write_conveyor/src/lib.rs`
- `crates/write_conveyor/examples/raw_wal_durability_bench.rs`
- `crates/write_conveyor/examples/raw_direct_wal_bench.rs`
- `crates/write_conveyor/examples/raw_uring_wal_bench.rs`
- `docs/WRITE_CONVEYOR.md`

**Recommended next steps:**
1. Wait for Euler audit and fix any medium/high findings.
2. Run one final `cargo test`, `cargo clippy`, `git diff --check`, and the no-override durable benchmark if anything
   changes.
3. Do not keep chasing group-window/client-driver tweaks unless fresh instrumentation contradicts the current evidence:
   the dominant p90 cost is the storage data fence (`sync-avg` around `0.8ms`).
4. Next big bets are not edge tweaks: design the production WAL integration around the manager-backed coalesced lane,
   add explicit crash/power-fail validation for accepted sync modes, and only then wire this into the engine commit path.
   A deeper queued io_uring/SQPOLL/linked-write+fsync backend is worthwhile only if its raw probe beats
   `raw_wal_durability_bench` first.

---

## >>> THE ONE NEXT ACTION: WRITE-PATH REIMPLEMENTATION (user-mandated 2026-07-04, single lane — this agent owns the write path end-to-end). W0 SHIPPED (`d828dea6`): the concurrent-invalidation stale-descriptor hazard was REAL (dup-key false-pass + stale sharded reads at shipped defaults) — fixed (descriptor flagging shared with the serialized path + publisher lock + cell-liveness gates), fable-audited, 473+367 tests green, zero bench regression. PROGRAM (from the verified decomposition — incremental ceiling ~150-230k, 10x needs structure): P0 = W1 payload-clones+timestamp-map -> W2 OVERLAP wave fsync with next wave (the WAL already supports it) -> W3 move wave-batch locate OFF the commit critical section -> W4 open-loop ingest (`;`-split + open-loop bench arm, waves of 1000s). P1 = W5 binary row-op WAL records (kill re-parse/re-resolve; replay=decode) + W6 fused wave-write kernel (validate+append+index-insert, ONE launch + ONE DtoH verdict array) + W7 wave-batched UPDATE/DELETE locates w/ in-kernel identity gather. P2 = W8 DEVICE CAS conflict ledger (claim_seq beside the device PK hash index; deletes host shard_pk_index + host String ledger = the M3 mandate). Ledger #26 (populate-vs-commit re-admission race, pre-existing) deferred to the A5 generation-atomic gate. Memory: `write-path-throughput-decomposition` (the full verified model + design study). <<<

**LOCAL W1a (branch `codex/durable-write-throughput`, NOT merged):** fixed the PK-index rebuild stampede that
was hiding in durable constrained-elision writes. New builds size host PK hash+bloom structures with
capacity-capped geometric headroom (extra slack capped at the default 4M-row shard target), and host cache rebuilds
are same-entry singleflight outside the global cache write lock after fast-path/extension rechecks. The device index
uses the same build basis but keeps exact row-count validation for now. Durable PK'd benchmark card after the patch:
32/128/512 writers = 18.1k/68.2k/71.4k sustained TPS, pk-index rebuilds 3/4/4; 512w off-lock prepare 38.1us,
p99 15.74ms, p99.9 99.64ms. Same diagnostic 512w phase before the patch was 48.7k TPS, off-lock prepare 1.33ms,
rebuilds 1971. Device wave-batch smoke: 128w durable = 62.4k TPS, `devlocate=4793`, host rebuilds 0; ignored GPU
parity tests `wave_batch_validation_matches_host_oracle` and `device_write_locate_matches_host_probe_twin` pass.
NEXT bottleneck: tail spikes from first rollover / group-commit scheduling, then W2 fsync overlap and W3 moving
wave-batch locate off the commit critical section.

**LOCAL W1b (same branch, NOT merged):** sharded admission now gives appendable open shards a byte-bounded first
capacity floor (4MiB allocation budget including row-id sidecar; two-int4 OLTP shape gets 262k rows, small
`shard_size_target` tests still cap at the target, wide rows get a smaller row floor). This avoids the immediate
capacity-2 first rollover without reserving a full 4M-row shard for wide tables. The residency budget accounting now
uses allocated shard bytes, including sidecars, and post-admission sidecar growth (`created_by`, `deleted_by`, rollover
shards) declines back to invalidate/re-admit instead of silently exceeding a configured budget; constrained-budget,
no-CUDA, and current-pressure admissions now fall back to dense snapshots rather than overcharging, transiently uploading
open-shard capacity, or evicting another resident table for speculative headroom. Shards now carry immutable admission
metadata plus a real `valid_through_index`; sharded table
eviction/status use table-level freshness, invalid resident state evicts before valid state, and residency status
separates snapshot vs sharded representations. Post-Hegel audit fix 512w durable samples: 69.2k / 70.2k TPS,
off-lock prepare 51.1us / 55.3us, p99 15.93ms / 15.53ms, p99.9 193.27ms / 201.72ms, rebuilds 3.
Known residual (separate STRATA admission-transaction work, not W1b write-throughput blocker): dense/single-buffer
admission that must evict still cannot fully roll back evictions if the post-eviction CUDA retain fails.

**LOCAL W3 (same branch, NOT merged):** wave-batch unique validation now runs before `commit_mutex` and outside the
internal-read leader-skip thread-local; the mutex-held sequencing body receives the precomputed verdicts. The verdict map
is stamped with the catalog generation and discarded under the mutex if DDL moved the catalog, so drifted items fall back
to non-deferred full validation instead of consuming stale 23505s. Nash audit found one more non-catalog drift hazard:
if `prepare_insert` deferred unique validation while a table was elided, then the table de-elided before wave validation
without a catalog bump, re-derived eligibility could say "not batchable" and falsely assume off-lock validation already
ran. Fix: `WriteDelta::unique_validation_deferred_to_wave` is the exact prepare-time obligation, `CommitWaveItem` carries
it, `WaveUniqueValidation` records the positions actually discharged, and the under-lock re-resolve grants
`ReResolveLedgerCovered` only when the catalog stamps match AND any deferred obligation was discharged; otherwise it
full-validates. Feynman audit then found that fallback `Full` could re-defer on a still-eligible table; fixed with
`InsertPrepareValidation::FullNoWaveDefer` for wave fallback + under-lock fallback. McClintock follow-up found two more
holes: serialized apply still used deferrable `Full`, and fallback rehydration could de-elide from an old published
frontier while W2 tails were applied-but-unpublished. Fix: serialized `apply_insert_with_profile` now uses
`FullNoWaveDefer`; rare wave fallback drains pending tails before non-deferred validation, and the commit-locked loop
retry-aborts an undischarged deferred item if an unpublished tail is still outstanding rather than rehydrating from an
unsafe frontier. Post-fix tests green:
`cargo check -p gpu_db_engine`, `phase0_unique...`, `only_offlock_full_may_defer...`, `wave_insert_prepared...`,
ignored GPU `wave_deferred_unique_revalidates_after_deelision_without_catalog_bump`,
`wave_batch_validation_matches_host_oracle`, `wave_batch_concurrent_dup_race_single_winner`,
`constrained_elision_same_snapshot_dup_insert_single_winner`, and
`constrained_elision_concurrent_dup_race_single_winner_per_key`. Franklin follow-up found that the rare `count > 0`
phase-0 detailed recheck still called `visible_row_with_value` before the W2 drain and fail-opened errors with
`unwrap_or(false)`. Fix: `count > 0` no longer rechecks in phase 0; it routes to the drained `FullNoWaveDefer` fallback,
and a row position is marked phase-0 validated only after all unique-index groups are count-zero or after full fallback
validates the whole row. Diagnostic 128w durable wavebatch+host/dev phase samples:
pre-Nash-fix 64.8k / 66.9k TPS, p99 4.35ms / 3.84ms, p99.9 11.58ms / 12.09ms; post-Nash-fix 68.8k TPS, p99 3.34ms,
p99.9 9.50ms, `validate_offlock` 1.81us/item, device locate 32.1us/wave, rebuilds 0; post-Feynman-fix 69.0k TPS,
p99 3.48ms, p99.9 9.00ms, `validate_offlock` 1.84us/item, device locate 32.1us/wave, rebuilds 0; post-McClintock-fix
69.7k TPS, p99 3.15ms, p99.9 9.01ms, `validate_offlock` 1.81us/item, device locate 32.2us/wave, rebuilds 0. Regular
post-Franklin-fix 69.4k TPS, p99 3.15ms, p99.9 11.36ms, `validate_offlock` 1.95us/item, device locate 31.9us/wave,
rebuilds 0. Pauli found serialized DDL/non-DML rehydration had the same W2 applied-but-unpublished tail race; first fix
made off-lock `rehydrate_elided_serialized` drain pending wave tails before gathering/de-eliding (internal commit-lock
rehydrates keep the direct branch), but Copernicus found the handoff gap between commit-lock release and pending-tail
enqueue. Final fix adds `tails_applied`, incremented before releasing `commit_mutex`, and all de-elision drains now wait
for `tails_applied == tails_finished`; the post-lock recheck retries if a wave applied while waiting for the mutex.
Post-Pauli-drain 128w sample: 67.4k TPS, p99 3.89ms, p99.9 11.11ms, `validate_offlock` 1.97us/item, device locate
32.4us/wave, rebuilds 0. Post-Copernicus-counter 128w sample: 66.3k TPS, p99 4.20ms, p99.9 12.18ms,
`validate_offlock` 2.09us/item, device locate 31.7us/wave, rebuilds 0. Lorentz then found `vacuum_table`'s
off-lock direct de-elision bypassed the same applied-tail drain; fixed by routing external vacuum through the
drain/commit-lock/recheck loop before `vacuum_table_locked` while keeping the internal commit-lock branch direct.
Post-Lorentz-vacuum 128w sample: 68.5k TPS, p50 1.63ms, p99 3.25ms, p99.9 11.82ms, `validate_offlock`
2.05us/item, device locate 31.8us/wave, rebuilds 0. Regular 512w durable sample pre-Nash-fix: 70.9k TPS, p99
15.25ms, p99.9 206.13ms, rebuilds 3.
Tail remains high, so the next real target is group-commit / fsync scheduling rather than just open-shard first
capacity.

**LOCAL W4a (same branch, NOT merged):** `docs/PLAN.md` now records the Chronicle/Disruptor/SEDA-style validation
model for writes: staged bounded conveyor, private/provisional work before the durability barrier, WAL-before-visibility
unchanged, applied-but-unpublished state bounded/drainable, and open-loop/offered-rate report cards that include queue
depth, wave size, fsync groups, and p50/p99/p99.9. First implementation slice adds optional commit-wave coalescing
(`Engine::set_commit_wave_coalescing`, benchmark knobs `GPU_DB_BENCH_WAVE_MIN` + `GPU_DB_BENCH_WAVE_WAIT_US`) so the
sequencer can wait a bounded few microseconds for a fatter wave; defaults remain `(1, 0us)` and preserve immediate
drain/latency behavior. Helmholtz audit found the first wakeup used `notify_one` on the shared condvar, so a normal
waiter could consume the threshold wake and leave the sequencer sleeping until timeout. Fix: threshold-crossing arrivals
use `notify_all`, the unusable `min=1/wait>0` notify path is gated off, and `WAVE_COALESCE_STATS` lets the regression
prove the wait branch, target-sized drain, and threshold wake fired. Correctness regression
`commit_wave_coalescing_preserves_durable_visibility_and_recovery` is green. Performance read: passive coalescing is
NOT the throughput breakthrough. Post-fix 128w default sample:
66.8k TPS, p50 1.65ms, p99 3.63ms, p99.9 9.70ms, mean wave 24.3, mean fsync group 61.9, rebuilds 0. Post-fix 128w
`min=128/wait=100us`: 65.6k TPS, p50 1.67ms, p99 3.53ms, p99.9 12.44ms, mean wave 46.0, mean fsync group 61.4,
rebuilds 0. Earlier 512w check: default 67.5k TPS, p99 15.67ms, p99.9 202.10ms, mean wave 134.0, mean fsync group
190.7; `min=256/wait=100us` 67.3k TPS, p99 14.23ms, p99.9 215.98ms, mean wave 222.5, mean fsync group 231.4.
Binary WAL short sample also did not move this path (66.7k TPS; sequence time rose), so the next real throughput lever
is a true async/open-loop submit lane or fused write work, not just waiting longer at the sequencer boundary.

**LOCAL W4b (same branch, NOT merged):** added opt-in hidden-stage write instrumentation behind
`GPU_DB_BENCH_PIPEPHASE=1` (`WAVE_PIPE_STATS`, exported for `oltp_commit_slo_benchmark`). It measures queue residence
before wave drain, coalescing waits, pipelined tail finish, durability wait, publish/ack, group-flush follower waits,
flusher elections, `begin_group_flush` under `commit_mutex` (including WAL record serialization), and real write+fsync
I/O. Default/off path stays immediate-drain and does not timestamp per item. Corrected one diagnostic bug during the
slice: "fast_covered" now counts only tails already durable on entry, not the post-loop return after a thread did work.
Avicenna audit then found low-severity measurement issues: follower-wait timing held `group_flush.coord` for relaxed
stat updates, failure-path tail denominators could skew, and the benchmark's nonzero check used an overflowing sum.
Fixes landed: drop the coord guard before stats, count claimed tails before durability wait, and use `any()` for nonzero.
Latest benchmark card:
128w durable clean (PIPEPHASE off) = 68.5k TPS, p50 1.63ms, p99 3.29ms, p99.9 9.91ms, mean wave 24.6, fsync group
62.3, rebuilds 0. 128w durable diagnostic = 68.5k TPS, p50 1.63ms, p99 3.29ms, p99.9 9.91ms; pipe means:
queue 371us/item, tail 1.140ms/tail, durability 1.137ms/tail, publish 2.7us/tail, follower wait 560us/wait,
`begin_group_flush` 44.4us/election, I/O 785us/job, 3304 real I/O jobs, mean fsync group 62.3. 512w durable diagnostic
= 67.4k TPS, p50 6.62ms, p99 16.03ms, p99.9 200.06ms, mean wave 135.2, fsync group 189.3; pipe means:
queue 2.85ms/item, tail 2.24ms/tail, durability 2.22ms/tail, follower wait 1.71ms/wait, `begin_group_flush`
118us/election, I/O 1.27ms/job. In-memory diagnostics: 128w = 59.3k TPS (tail 9us, queue 935us/item);
512w = 90.1k TPS (tail 17us, queue 2.42ms/item, mean wave 199). Read: durable is capped by the ~0.8-1.3ms disk fsync
job plus follower/tail scheduling, and extra writers mostly add queueing/tail latency rather than TPS. The in-memory
512w ceiling shows the engine can approach 90k once durability is removed, but the closed-loop submit/scheduler shape
and serial wave body still cap sustained throughput before the 100k+ target. Next best move: build the open-loop
write-ingest benchmark/lane to keep the conveyor full without client threads blocking per commit, then target
WAL-group serialization/IO (`begin_group_flush` + fsync) and fused wave write work.

**LOCAL W4c (same branch, NOT merged; Nietzsche final focused audit pending):** implemented the first write-conveyor submit lane.
Default path is unchanged (~67.8k TPS at 128w durable after the refactor). New default-off controls:
`Engine::set_group_flush_coalescing` + `GPU_DB_BENCH_FLUSH_MIN/WAIT_US` and
`Engine::execute_dml_concurrent_pipelined` + `GPU_DB_BENCH_PIPELINE=N`. Gibbs audit found a real safety gap in the
first version: the pipelined public API lacked the `is_concurrent_dml` sequence-default gate and returned only one batch
`Result`, hiding partial success. Fixes landed: a shared `concurrent_dml_command_eligible` predicate rejects SERIAL /
`nextval` INSERTs before enqueue, and the API now returns `Result<Vec<(txn_id, Result<(), ExecuteError>)>, ExecuteError>`
after waiting all outcomes. Raman re-audit found a TOCTOU in the first fix: a catalog could gain a sequence default
between eligibility and `prepare_dml`, producing a prepared delta with non-empty `seq_advances`. Sartre then found the
same invariant had to hold after the sequencer's commit-time re-resolve (`FullNoWaveDefer` after DDL drift), not just
off-lock prepare. Final fix validates every actual prepared `WriteDelta` is sequence-free immediately after off-lock
prepare in BOTH single and pipelined concurrent paths AND again after the final under-lock re-resolve, before
WAL/propose/apply. New regressions prove direct concurrent sequence advances are rejected, commit-time re-resolve drift
to `nextval` is rejected before visibility, pre-enqueue rejection commits no earlier prepared member, and same-batch
duplicates return per-statement success/error. The naive flusher-side
sleep was tested and rejected as a main optimization:
128w durable `FLUSH_MIN=128/WAIT=100us` fell to 59.6k TPS and mean fsync group barely moved (62 -> 64;
target-after-wait only 25 times); 25us also fell to 65.0k and never hit target. Pipelined submit is the win:
prepare/enqueue N independent autocommit statements, then wait for all outcomes at the final durability barrier. Clean
128w durable `PIPELINE=3` after fixes = 135.6k TPS, burst 160.4k, p50 2.47ms, p99 4.89ms, p99.9 87.06ms, mean wave 132.3,
mean fsync group 163.8, rebuilds 0. Earlier diagnostic 128w durable `PIPELINE=3` = 131.4k TPS, p50 2.58ms, p99 4.46ms,
p99.9 87.24ms, pipe queue 855us/item, tail 1.03ms/tail, durability 1.02ms/tail, I/O 787us/job. Depth sweep:
`PIPELINE=2` = 75.7k TPS; `PIPELINE=4` = 108.1k TPS; `PIPELINE=8` = 134.0k TPS but p50 7.22ms. 96w/depth4 =
102.0k TPS, p50 3.32ms. In-memory depth3 = 129.7k clean / 122.4k diagnostic, still latency-heavy because it measures
the final batch wait. Correctness coverage: `direct_concurrent_dml_rejects_prepared_sequence_advances`,
`concurrent_dml_rejects_sequence_advances_from_commit_time_reresolve`, and
`pipelined_concurrent_dml_{rejects_sequence_defaults_before_enqueue,returns_per_statement_outcomes,
preserves_durable_visibility_and_recovery}` green (durable WAL, multithreaded pipelined inserts, group-flush coalescing
branch, visibility, flush accounting, restart). Read: pipelining proves the staged
conveyor can exceed the sustained TPS target by increasing wave/fsync group size,
but latency SLO still needs a lower-latency async completion model and/or fused wave-write work rather than client-side
batch waits.

**LOCAL W4d (same branch, NOT merged):** added an explicit cooperative submit validation surface:
`Engine::submit_dml_concurrent_pipelined` returns `ConcurrentDmlOutcome` handles whose `wait()` returns the per-txn
result; this is NOT a true background executor because progress is still driven by commit-wave sequencers and wait/drop
probes. Dropping a handle waits as a safety fallback so the active snapshot guard cannot vanish while its queued item may
still need the conflict ledger; during panic unwind it releases the guard if the outcome is already done, otherwise leaks
the guard instead of causing a double-panic or deregistering early. Benchmark knob `GPU_DB_BENCH_ASYNC_WINDOW=N` keeps N
individual commits in flight per writer and records per-handle elapsed time (not batch-barrier time); reports label it a
"cooperative submit window" (current benchmark builds also print the submit-call elapsed metric). Correctness smoke
`submitted_concurrent_dml_handles_wait_or_drop_to_completion` is green. Performance verdict: async window did NOT move
the frontier on this path: 128w durable window2 = 76.2k TPS, p50 3.05ms; window3 = 128.5k TPS, p50 2.61ms, p99 5.86ms;
window4 = 108.5k TPS, p50 4.12ms. Compared with batch `PIPELINE=3` (~135.6k, p50 2.47ms), this is a useful API/validation
primitive but not the throughput/latency lever. Next performance work should reduce actual wave/flush work: fused
wave-write kernel / host sequence body, or a real durability I/O improvement.

**LOCAL W4e (same branch, NOT merged; Einstein audit resolved):** moved group-flush WAL record encoding out of the engine
`commit_mutex`. `WalBuffer::begin_group_flush` now snapshots the unflushed `WalRecord` handles (`Arc<[u8]>` payload refs)
under the mutex and returns a `WalGroupFlushJob`; the job builds the on-disk bytes/checksums in `commit()` alongside the
existing lock-free write+fdatasync. Durability ordering is unchanged: `io_in_flight` is still set before the job is handed
out, the durable watermark advances only after job success, and failure still poisons/wedges before visibility. This
targets the diagnostic gap where `WAVE_STATS` time greatly exceeded charged host phases because the sequencer waited
behind flusher `begin_group_flush` serialization. Focused checks green: `cargo fmt --package gpu_db_wal`, `cargo check -p
gpu_db_wal`, `cargo test -p gpu_db_wal --lib`, `cargo check -p gpu_db_engine`,
`group_fsync_failure_wedges_the_concurrent_commit_path_without_exposing_the_delta`, and
`pipelined_concurrent_dml_preserves_durable_visibility_and_recovery`. Diagnostic 128w durable `PIPELINE=3
HOSTPHASE+PIPEPHASE`: before this local slice on the same noisy host was ~52.5k TPS with `begin/election` ~104us and
wave time ~17.2us/item; after = ~73.8k TPS, `begin/election` ~35us, wave time ~12.2us/item. Clean 128w samples on the
same host were unstable (50-58k TPS after, and earlier saved W4c/W4d clean runs were ~133-136k), so compare by the phase
counter first, not today's absolute TPS. Remaining visible bottlenecks: commit-mutex acquisition still dominates
unattributed wave time, tail/durable wait remains ~0.9ms/tail, and true open-loop/background sequencing is still not built.
Einstein audit found one medium unwind-safety issue: the job was disarmed before out-of-lock allocation/encoding/IO, which
could leave `io_in_flight` stuck if a panic occurred. Fixed by disarming only after `io_in_flight` is cleared and waiters
are notified; re-audit confirmed resolved.

**LOCAL W4f (same branch, NOT merged; Maxwell audit resolved):** added hidden host-phase timing for the full
commit-wave body (`GPU_DB_BENCH_HOSTPHASE=1` now reports validate, commit-lock wait, conflict check, re-resolve,
sequence/WAL, ledger, apply, invalidate, fast-run flush, resident append flush, ledger prune, and commit setup). The
measurement found the remaining host-side bottleneck after W4e: `fast_run_flush` dominated at ~13.8us/item before this
slice, with under-lock re-resolve next at ~3.4us/item. The first optimization aggregates value-index publication across
the whole table batch instead of cloning/reinserting the same low-cardinality slot once per delta. Correctness regression
`pipelined_fast_run_preserves_repeated_value_index_slots` proves repeated `v=7` inserts remain deletable by value index.
Maxwell audit found one medium replay/parity issue in the initial optimization: grouping deltas by table before tuple-id
reservation could assign live tuple ids in table-sort order while WAL replay assigned them in log order. Fixed by
reserving tuple ids in original wave order inside `flush_fast_run`, then grouping only for publication; new regression
`pipelined_fast_run_preserves_tuple_id_order_across_tables` compares live MVCC versions to durable-WAL recovery. The
audit's low measurement issue was also fixed by charging commit-mutex setup before the per-item loop. Focused checks
green: `cargo check -p gpu_db_engine`, both W4f regressions,
`pipelined_concurrent_dml_preserves_durable_visibility_and_recovery`,
`stage0_wal_replay_reproduces_byte_identical_version_stamps`,
`execute_dml_concurrent_matches_the_serialized_path_single_threaded`, and `git diff --check`. Latest 128w durable
`PIPELINE=3 HOSTPHASE+PIPEPHASE` diagnostic on this noisy host: 64.6k TPS, p50 6.55ms, p99 8.56ms, p99.9 15.55ms,
mean wave 185, mean fsync group 189, `fast_run_flush` ~9.83us/item, re-resolve ~2.74us/item, sequence ~0.60us/item,
`begin/election` ~28us, I/O ~860us/job. Clean sample: 65.2k TPS, p50 6.38ms, p99 9.80ms, p99.9 15.33ms. Earlier W4c
samples on this branch reached ~135k TPS, so do not overfit this host's absolute TPS; the phase counters say the next
local lever is host-store publication inside `fast_run_flush`, with true GPU-native fused write apply still the product
direction.

**LOCAL W4g (same branch, NOT merged; Jason audit resolved):** added a storage-layer reserved-key batch insert API
(`InMemoryTupleStore::tuple_insert_reserved_key_batch_with_ids`) with upfront validation/no partial install and direct
fresh-key slots, then wired fast-run table publication through it. `GPU_DB_BENCH_HOSTPHASE=1` now also prints a
fast-run breakdown (`reserve_group`, preplan, row encode, row store, value index) with its own fast-run item/row
denominators. Broadened off-lock INSERT delta carry for sequence-free inserts, but under-lock reuse is still gated by
catalog generation plus either the old elided predicate or the live constraint-free fast-run table predicate; re-keying
now rewrites value-index row keys, with a single-row fast path for the OLTP benchmark shape. Jason audit found a real
high-severity ordering bug: a later slow same-wave item could re-resolve before prior buffered fast-run inserts were
published, causing live/WAL replay divergence for `INSERT` then `DELETE`. Fixed by flushing `fast_run` before any
non-fast item re-resolves; consecutive constraint-free fast INSERTs still chain. New regression
`pipelined_fast_run_flushes_before_later_delete_reresolve` proves live count and WAL recovery both delete the row. Audit
low finding (fast-run timing divided by all wave items) fixed with fast-run item/row counters; re-audit found no remaining
issues. Focused checks green:
`cargo test -p gpu_db_storage --lib`, `cargo check -p gpu_db_engine`, W4f/W4g regressions,
sequence/default rejection regressions, durable pipelined recovery, and `git diff --check`. Benchmarks remain noisy:
best post-W4g 128w durable `PIPELINE=3 HOSTPHASE+PIPEPHASE` diagnostic = 86.5k TPS, burst 103k, p50 3.95ms, p99 7.81ms,
p99.9 21.20ms, wave sequencing 10.2us/item, `reresolve` 1.85us/item, `fast_run_flush` 7.48us/item (`row_store`
1.15us/item, `value_index` 3.81us/item), I/O ~828us/job. Low-cluster clean/diagnostic runs still appear around
55-63k TPS with ~16us/item wave time, so compare phase counters and repeat samples rather than one absolute TPS.
Current local bottleneck is value-index publication plus the durable tail (~0.85-0.9ms/job); the product-level next
move is still fused GPU write apply/private device state before durability publish, not deeper CPU data-plane work.

**LOCAL W4h (same branch, NOT merged; Curie audit resolved):** replaced per-table value-index values with
`ValueIndexSlot::{Empty, One, Many}` and changed the map shape from flat `(column,value)` to nested
`column -> value -> slot`. This lets a fast-run batch clone/publish the column submap once and then update many values,
instead of touching the whole table-wide map for every high-cardinality `id` value. Singleton slots avoid allocating an
`imbl::Vector` until a second row key arrives; low-cardinality slots still promote to the persistent vector representation
that protects snapshot/COW append cost. Updated `index_keys`, `value_index_snapshot`, fast-run publication,
add/drop/rename column, rename table, rehydration rebuild, and tests through helper APIs. Hume/Curie audits found no
correctness issues across append-only semantics, snapshot/COW, row-key ordering, DML resolution, DDL add/drop/rename/rebuild,
rehydration rebuild, and W4g re-keying. Low performance caveat: inline `One(String)` may trade vector allocation savings
for more String cloning in persistent-map node copies, so keep benchmarks for high-cardinality multi-column inserts.
Focused checks green: `cargo check -p gpu_db_engine`, value-index oracle/fallback tests, stage3 value-index snapshot tests,
null key test, write-set snapshot tests, W4g/W4f regressions, durable pipelined recovery,
`relational_{rename_column,drop_column,rename_table}_..._replays`, `cargo test -p gpu_db_storage --lib`, and
`git diff --check`. Benchmark result: 128w durable `PIPELINE=3 HOSTPHASE+PIPEPHASE` = 103.7k TPS, burst 119.1k, p50
3.41ms, p99 6.82ms, p99.9 12.99ms, wave sequencing 8.2us/item, `fast_run_flush` 5.70us/item, `value_index` 2.57us/item,
I/O ~808us/job. Clean sample = 104.9k TPS, burst 121.2k, p50 3.35ms, p99 6.36ms, p99.9 13.31ms. Sustained throughput
target is now met on this host for the pipelined durable insert workload; latency SLO and burst >=400k remain open.
Follow-up benchmark instrumentation now reports pipelined/async submit-call elapsed time (not pure enqueue: the cooperative
submit may run sequencer/tail work when it becomes the leader). A post-W4h diagnostic sample showed the submit call around
51us/commit in that run, queue residence ~1.26ms/item, and durable tail ~0.86ms/tail. The remaining latency gap is therefore
not explained by SQL parse alone; it is dominated by closed-loop queueing plus the final durability/visibility barrier.

**E2.5b-2 RECORD + FULL-ARC AUDIT ADOPTED (2026-07-06, `9073c816`): 1,662,843 sustained / 2,683,310 burst
durable TPS** (closed-loop per-request durable acks, 15s, p50 20.3ms p90 44.9ms p99 191ms), measured AFTER
adopting all five findings from the adversarial opus audit of the whole lanes arc (b3e4922c..d3c77701):
F1 CRITICAL — oracle activation double-seed (two lanes racing the cold-start check-then-act could seed
duplicate global seq spaces → acked-commit loss at recovery); fixed with double-checked seeding under the
commit lock, activation latch stored LAST. F2 — apply-leader panic stranded waiters (livelock); fixed with
catch_unwind + ApplySlot.failed + resume_unwind. F3 — async WAL poison never reached settle (clients hung
instead of erroring); fixed with poison_reason() drain in settle_intent_lane. F4 — commit_mutation_batch
missing the lanes guard. F5 — archive/PITR would silently drop lane commits; now refuses in lanes mode +
dead ts_side/reserve_timestamps removed. Gates: 486/486 default AND fua modes, GPU intent suites green
(serial + lanes=2). Champion config: 10 lanes / 10 pumps / 10 bench drivers, WINDOW=6144, MIN_WAVE=1024,
GROUP_US=4000, FLOOR/SHARD_TARGET=48000000, GPU_DB_WAL_DURABILITY=fua. Driver sweep at 10L: 8→1.58M,
10→1.66M, 12→1.51M. 30s STABILITY (floor/target 64M): **1,552,410 sustained / 2,625,190 burst — 46.6M rows
durable, durable cut == applied cut, pk-rebuilds 1, clean exit** (93% of the 15s record held for 2x the
duration; per-wave: validate 400us, publish 158us, device-apply 1101us wait — apply is the next wall).
Bench fix that run surfaced: the old fixed 4M-ids/writer stride overflowed into the neighbor's key range
at 1.66M TPS x 30s and VALIDATION CORRECTLY REJECTED the duplicates — stride now 2.1e9/writers. Deferred to E2.5c (documented, unblocked): lanes reopen/replay merge, lane-log
truncation/archive, Raft-compatible seq oracle, default flips, engine-side FuaWalSegment recycle.

**E2.5b-2 FIRST MILESTONE (2026-07-06): SUSTAINED SEVEN FIGURES — 1,323,628 durable TPS / 2,547,760 burst**
(closed-loop per-request durable acks, 15s, p50 18.5ms p90 31ms p99 84ms, recovery cut clean, pk-rebuilds 1).
Config: GPU_DB_WAL_DURABILITY=fua GPU_DB_INTENT_LANES=6 GPU_DB_INTENT_LANE_MIN_WAVE=1024
GPU_DB_INTENT_LANE_GROUP_US=4000 GPU_DB_OPEN_SHARD_FLOOR_ROWS=48000000 + bench: ARM=driver WRITERS=8
PUMPS=6 WINDOW=6144 SHARD_TARGET=48000000. Driver knee: 8 drivers optimal (16→1.16M, 12→1.20M, 8→1.32M,
6→1.25M, 20→0.97M; the record was unlocked by NOT letting bench client threads strangle the pumps).
8L/8P variant: 1.29M/2.55M burst. Engine trajectory this program: 32k → 73k → 414k → 597k → 894k → 1.32M
sustained. The architecture that did it (all committed, gates green): N-lane pumps + lock-free seq oracle +
cross-lane validate/apply coalescers + lean LaneIntent + capacity floor/capacity-sized PK index + FUA lane
WAL + cut-gated settlement. KNOWN ISSUE: 4-driver x >=12288-window bench runs exit silently (uninvestigated;
not the record path). NEXT toward 2M+: whole-system profile of the flat residual, wider lanes with freed
cores, and the E2.5c program (lanes reopen/replay, truncation, Raft-compatible seq oracle, default flips).

**E2 PROGRAM (2026-07-06, commits 907d6b88..a88cf690, ALL PUSHED; user mandate: multi-million engine TPS,
disruptor staging, main agent implements / opus audits only):** classic 32k -> intent fast path (E2.1) ->
disruptor submit/poll + integer ledger (E2.2, 414k) -> fast lane (E2.3, 597k peak) -> sharded-dispatch
negative (E2.4a, documented) -> FuaWalLaneSet in crates/wal (E2.5a: N ordered lanes, global seqs, cross-lane
cut, merge recovery, property-tested) -> batched thin cut + HashMap timestamps (E2.5b-1) -> audit clean, two
LOW fixes (unique wave timestamps for PITR, self-releasing IntentTicket) -> single-copy WAL payload
(E2.5b-3): **511k sustained / 666k burst durable, sequencer 1.70us/item, recovery parity to 1.94M commits.**
Bench: GPU_DB_BENCH_ARM=driver WRITERS=32 PUMPS=4 WINDOW=256 + GPU_DB_WAL_DURABILITY=fua.
**NEXT (E2.5b-2, the structural 2x):** wire FuaWalLaneSet into the intent path — N lane-sequencers claiming
global-seq blocks (exact tiling contract), single writer per lane, visibility = cross-lane contiguous cut ∧
applied; then GPU wave scaling (device ~1us/item, FUA budget 2.75M/s both idle). The ordered serial cut
(~1.5-1.7us/item ~= 625k ceiling) is the LAST wall before the multi-million band.

**LOCAL FUA-WAL (same branch, NOT merged; new lane owner 2026-07-05, opus audit in flight):** the write-conveyor
durable wall was DIAGNOSED AS A MEASUREMENT ARTIFACT and fixed. All prior durable fences ran over
`posix_fallocate` UNWRITTEN extents (per-fence XFS extent-conversion journal force, ~2.45ms) and were issued
SERIALLY through `sync_in_progress` into `fdatasync` (full NVMe cache FLUSH — unpipelineable; ceiling ~1.3k
fences/s). FUA write-through (`O_DIRECT|O_DSYNC`) over PRE-WRITTEN extents pipelines: this drive does ~28.6k
durable 4KiB fences/s at qd=16 (p50 0.68ms — latency DROPS with depth). Shipped in `crates/write_conveyor`:
`FuaWalSegment` create/RECYCLE (epoch-stamped headers, `WAL_SEGMENT_FLAG_EPOCH_STAMPED` format addition so a
recycled file's previous-life frames are scan-rejected; legacy files unaffected), single appender into anonymous
aligned staging, `spawn_fence_pool` + contiguous durable cut, `free_fence_slots` pacing gate; 6 unit tests, all
54 crate tests green, clippy clean. Client-contract benchmark (`fua_wal_client_bench`, on the library types,
scan-recovery + store validated): **1.99M durable writes/s @4096 clients (p50 1.88ms p99 2.46ms), 2.66M @8192
(p99 3.55ms)** vs the prior lane's best-ever 0.357M — the old serial-flush design is superseded. Two load-bearing
scheduling laws (violating either re-creates the <150k collapse): ADAPTIVE FRAME SIZING (frame ≈ pending/qd) and
FENCE-POOL PACING (publish only into a free lane). Full write-up + corrected negative results:
docs/WRITE_CONVEYOR.md "FUA-Pipelined Durable Lane". Next: manager-level segment rolling (recycle pool +
background prep), event-driven acks, then gating engine SQL commit visibility on `durable_record_seq` — the
engine host-side per-commit cost is now unambiguously the remaining wall on the 500k-durable mission.

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
