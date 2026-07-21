# HANDOVER — Resume Baton

This file records only the current boundary and where the next agent resumes. `PLAN.md` owns all open work;
`STATUS.md` owns completed evidence. Do not turn this file into another backlog.

## Current boundary

- **R3-001 through R3-005, DUR-002, STRUCT-001, RETIRE-001, RETIRE-003, and PERF-001 are complete.** Production
  relational reads and writes are GPU-required; unsupported work fails loudly without host relational execution or
  fabricated fallback telemetry.
- RETIRE-003 deleted generic KV/MVCC fake-CUDA result execution and its host compaction/order/projection seams.
  Ordinary SELECT and direct/streaming joins now retain coordinates, order/window, and result values on-device until
  one strict terminal frame readback. Unsupported shapes fail independently of cardinality.
- PERF-001's exact-generation prepared point routes, hard-budget accounting, duplicate-decline contract, async CUDA
  ownership, and public result contracts remain intact. Explicit reverse-gather/deauthorization and DDL/recovery/
  import repair remain isolated under blocked **RETIRE-002**.
- READ-002's BENCH prerequisite milestone is complete: a typed, exact-generation `(int4, int8)` GPU equality
  directory provides O(1) lookup, on-device exact-key/MVCC checks, fixed-width gathering, nonzero index/cache
  evidence, and zero cold access. Its broader type breadth remains deferred under READ-002 after BENCH evidence.
- PRODUCT-002's canonical SQL/type milestone is complete: the immutable schema and W1 DML parse unchanged, required
  typed parameters/codecs are available, named resident indexes mutate incrementally, and checked UPDATE plus DML
  `RETURNING` stay on GPU execution/result paths. Broader catalog/type breadth remains blocked under PRODUCT-002
  until BENCH evidence.
- The active priority is **PRODUCT-001** canonical serving and write-path unification. It remains open through
  mixed-traffic/recovery, compatibility migration, legacy deletion, and final source-ownership proof; BENCH-001 does
  not unlock on a serving-only subset.
- PRODUCT-001's inventory and first runtime slice are accepted. Canonical facade/server mutation dispatch now carries
  one exact parsed owner through `Engine::submit_transaction`; reads and stateless transaction-control errors are
  separate and pre-effect. Existing lower commit strategies remain plural and public, so this is an admission
  foundation rather than the claimed final coordinator.
- The General atomic resource/user-transaction foundation is accepted after an independent rejection and re-audit.
  A program owns its transaction identity before registration and holds one statement guard through terminal
  publication; stale waiters revalidate, post-WAL uncertainty is indeterminate and recovery-owned, logical intent
  bytes are distinct from physical WAL, and dependency/cold-access closure is preclaimed. Parsed programs remain
  unbounded `General` work. The later accepted engine-owned prepared-route proof now supplies exact identity,
  READ COMMITTED/REPEATABLE READ semantics, and bounded W1/T8/T32 derivation without imposing a 32-operation cap.
- The typed prepared-command foundation is accepted after independent rejection and re-audit. Parse owns typed AST
  `$n` slots; Bind replaces them directly for sequential, borrowed, and shared-session facade execution without
  reparsing reconstructed SQL. Raw parameters in ordinary or unsupported positions fail pre-effect, explicit casts
  retain replayable canonical identity, and NUMERIC typmods round/rescale/enforce precision. The remaining lowering
  is identity-only for transitional SQL-text WAL/retry records.
- The canonical engine-backed server now has one connection-local extended-query owner and one dispatcher shared by
  blocking shared-engine, sequential-baseline, and async ingresses. Parse asks the engine/facade to resolve neutral
  parameter types and result columns against one committed catalog snapshot without reparsing the syntax-validated
  AST; typed unused trailing Parse parameters remain part of Bind arity. Bind decodes without executing; Describe
  emits statement/portal metadata; Execute caches one outcome per portal and supports `max_rows` suspension. The
  shared owner now also decides PostgreSQL failed-transaction precedence, duplicate/unnamed replacement timing,
  implicit-cycle commit/rollback, Close, skip-until-Sync, and ReadyForQuery. Host raw-pgwire and tokio-postgres tests
  pass. The first independent audit rejected precedence, codec SQLSTATE, rollback-proof, and documentation holes;
  all were fixed and the independent re-audit accepted the slice.
- The live commit-authority consolidation is accepted after independent adversarial audit. One commit mutex,
  replicator, canonical WAL/status index, synchronous apply order, checkpoint entrance, and contiguous publication
  join serve serialized, batch, COPY, explicit, classic-wave, and optimized-lane work. Optimized lanes are
  preparation only: their separate sequence/WAL/apply queue/publication bridge/checkpoint writer are deleted, fresh
  traffic creates no `.lane-*` files, and historical physical-lane input is closed after startup replay and remains
  read-only until offline migration.
- The prepared transaction route/class and isolation slice is accepted. W1/T8/T32 are engine-derived scheduling
  classes only; indexed programs above 32 operations remain `General`. READ COMMITTED refresh, REPEATABLE READ,
  read-your-writes, NULL, exact dependency/resource closure, one-record atomic publication, and pre-WAL allocator
  safety are proven. Retained prepared pins own exact resident-source and index Arcs, publish extent/posting state
  across cache purge, reject source ABA/replacement/rollover, and never rebuild or scan the base relation.
- The public facade unification slice is accepted after independent rejection and re-audit. `SharedEngine::submit`
  is the sole public execution boundary for text, bound prepared ASTs, optional point-read batching, session-close
  rollback, and deterministic instrumentation. The production `EngineFacade`/numeric session registry, borrowed and
  stateless execution functions, separate shared/session/batched/prepared free functions, public hook functions, and
  old sequential server loop are deleted. Batchers are bound to one engine instance with crate-private enqueue;
  instrumented variants cannot bypass active/failed transaction state or caller-independent transaction allocation.
  Sync, sequential-acceptance, and async ingresses now share the same `SharedEngine` handler. The next boundary is
  PostgreSQL compatibility migration, not another facade.
- Describe-time catalog revalidation is accepted after independent rejection and re-audit. Statement and portal
  metadata is revalidated through `SharedEngine` before blocking or async ingress can encode it; stale inferred
  parameter OIDs fail with `42804`, stale result shapes fail with `0A000`, and `26000`/`34000`/`25P02` precedence is
  preserved without execution, transaction allocation, or publication.
- Transaction-private Parse/Describe is accepted after two substantive rejection/repair rounds and a final fresh
  ACCEPT. The connection's engine-bound `SharedSession` now owns effect-free metadata resolution through the same
  READ COMMITTED/REPEATABLE READ statement snapshot as execution. A private CREATE is describable, bindable,
  executable, and readable only by its creator until COMMIT; rollback invalidates it. A foreign engine session
  rejects before effects even when numeric transaction IDs collide. Failed rowless pgwire Describe emits only cached
  private metadata, while new facade description work returns failed-transaction precedence. The exact private-table
  prepared-DML proof does not weaken published-table catalog validation. Mechanical test extraction leaves the two
  server production roots at **1,176/1,043** lines and their test owners at **944/1,259**.
- The transaction-private composite envelope is accepted after four rejection/repair rounds and a fifth independent
  ACCEPT. Exactly one `CREATE TABLE` composes with DML in either supported order, retains private read-your-writes,
  and commits as one typed/coalesced WAL operation plus one table-batched catalog/data publication and recovery path.
  Exact final-record GPU reservations, retained-generation accounting, and a table-scoped named-index lifecycle close
  every known post-WAL allocation/index wedge; unrelated purges remain immediate. The transaction coordinator is
  **1,758** lines after extracting exact accounting. Its final card records **217.840M/s at p50 175us** in-L2 and
  **185.692M/s at p50 226us** out-of-L2; Layer 1 records **1,485.3/1,449.8 GB/s** rooflines and **1,674.2M
  elements/s** GROUP BY, with a **2,052.8s** 48M-row build and zero final residency work. Multiple DDL, broader
  compatibility, and conservative any-intervening-commit serialization remain open under PRODUCT-001.
- The PRODUCT-001 SQLx/simple-query compatibility slice is **accepted**. SQLx now runs its unchanged supported
  subset against `gpu-db-engine-server`; bounded `SELECT 1 AS one` uses a typed transient GPU relation; ordinary
  multi-statement Query messages use one implicit transaction per idle segment; and the application-driver smoke
  script names the relocated canonical test.
  The first independent audit rejected escape-string, identifier-adjacent dollar-token, and CR-comment splitting;
  those findings are fixed and gated. A fresh re-audit then rejected PostgreSQL comment trivia being submitted as
  executable statements, which could roll back valid trailing-comment DDL or hide commented transaction control.
  That second finding is now repaired by one `gpu_db_sql::split_simple_query` lexical pass that returns executable
  spans, omits valid boundary/comment-only trivia, and preserves unterminated block comments for syntax failure.
  A third fresh audit rejected explicit transaction-control segmentation, comment-only Query cleanup of a pending
  extended implicit transaction, interior comment-as-whitespace handling, missing permanent typed-NULL/sabotage
  coverage, and the then-current card's Layer-2 regression signal. The repaired candidate dynamically opens a fresh
  implicit segment after explicit COMMIT/ROLLBACK when multiple suffix statements remain, completes pending extended
  cycles even for an empty/comment-only Query, normalizes interior comments through the same quote/dollar-aware SQL
  scanner while retaining exact original command text for WAL identity, and adds permanent NULL-vs-zero/empty raw-
  wire plus invalid-GPU fail-loud proofs. A fourth fresh audit then found that the default implicit BEGIN opened
  before an explicit multi-statement `BEGIN` silently discarded its requested isolation/access/deferrable modes.
  The segment owner now looks ahead only to its next transaction boundary and opens the implicit segment with the
  exact client BEGIN when present; that statement therefore promotes the same segment without losing its modes.
  Blocking and async/batching-on raw-wire regressions prove unsupported SERIALIZABLE remains `0A000` and supported
  READ ONLY prevents DDL publication rather than being replaced by defaults. A fifth fresh audit found that parsing
  and executing each split statement in turn allowed an explicit pre-error COMMIT to publish before a later syntax
  error, unlike PostgreSQL's whole-message parse precedence. Multi-statement Query now performs a side-effect-free
  parse plus zero-arity Bind of every executable span before any synthetic BEGIN or facade submission. Syntax and
  unbound-parameter failures therefore emit no prefix completion, roll back a pending extended implicit cycle, or
  leave an existing explicit transaction failed until standalone ROLLBACK; catalog/constraint errors remain ordered
  execution-time outcomes. Blocking and async/batching-on raw-wire tests cover errors after COMMIT and ROLLBACK,
  staged-DDL non-publication, and both cleanup states. SQL
  **49**, canonical server **29/4 ignored + 3/1 pgwire + 1 SQLx**, legacy protocol **71 + 127 + 1 tokio-postgres**,
  workspace check, strict affected all-target clippy, scoped formatting, diff, source-size, three sequential plus two
  overlapping typed-NULL GPU runs, and invalid-device sabotage pass with no CUDA 700/716/717. The replacement report
  card passes at **230.786M/s, p50 157us** in-L2 and **199.406M/s, p50 202us** out-of-L2, with
  **1,477.5/1,438.3 GB/s** rooflines, **1,674.9M elements/s** GROUP BY, and a **2,049.8s + 0.0s residency**
  48M-row build; this is above the preceding accepted Layer-2 card in throughput and below it in latency in both
  cache regimes. Five independent rejection rounds drove every repair above; the sixth fresh read-only re-audit
  confirmed syntax/parameter/COPY precedence, semantic-error ordering, exact transaction modes, GPU evidence, and
  sole-owner boundaries, and returned **ACCEPT**.
- The PRODUCT-001 COPY compatibility slice is **accepted** after one independent rejection and a fresh adversarial
  re-audit. Simple and extended text/CSV COPY FROM/TO now run on both canonical ingresses. COPY FROM retains an
  opaque exact engine/transaction/relation proof through Parse, Describe, Execute, zero-row completion, and CopyDone;
  definitive constraints repeat under the commit mutex before WAL, and every nonempty mutation crosses
  `SharedEngine::submit` into the existing canonical claimant. Simple COPY TO synthesizes its exact parsed SELECT;
  extended COPY TO executes its exact retained bound SELECT AST. Raw phase/ABA/failed-state/confused-deputy/whole-
  message-precedence tests, same-key WAL sabotage, recovery, and three sequential plus two overlapping live-GPU
  lifetime cases pass. The relocated tokio-postgres smoke now targets `gpu-db-engine-server`. Final Layer-2 results
  are **231.468M/s at p50 156us** in-L2 and **198.067M/s at p50 205us** out-of-L2; Layer-1 rooflines are
  **1,418.1/1,440.2 GB/s**, GROUP BY is **1,674.9M elements/s**, and the 48M-row build is **2,061.6s + 0.0s
  residency**. The frozen-tree re-audit verdict is **ACCEPT**.
- The PRODUCT-001 canonical TLS/SCRAM slice is **accepted** after two independent rejection/repair rounds and a
  third frozen-tree audit. The product binary now owns an explicit local-development trust profile and a fail-closed
  production rustls/SCRAM-SHA-256 profile around the existing dispatcher. PostgreSQL SASLprep/raw fallback, uniform
  wrong-user proof work, strict transcript/attribute validation, bounded framing, and permanent blocking/async raw-
  wire recovery are proven. The live security preflight now boots `gpu-db-engine-server`; no execution, WAL,
  sequence, or publication authority moved. CancelRequest still closes without resolving its key or cancelling work,
  so real cancellation is the next PRODUCT-001 boundary rather than a claim of this slice.
- Physical multi-GPU work remains user-parked under **MULTI-001/002/003**.

## Integrated baseline — preserve it

- The generic KV/MVCC entry preserves commit-wedge and leader-precedence errors, then fails before source resolution
  or GPU/fallback telemetry. Do not restore a host or fake-device implementation behind that compatibility name.
- Ordinary SELECT survivors are device coordinates with same-context provenance. Predicate compaction, checked
  arithmetic, ORDER/LIMIT/OFFSET, fixed/text/validity materialization, and terminal framing remain one device pipeline;
  the host may decode the single bounded frame only for final protocol values.
- Empty results do not bypass type, width, ORDER, timestamp, bool, or aggregate-shape validation. Launched errors
  drain before pool reuse, and nullable or filtered-out rows cannot manufacture overflow.
- The canonical report card is the result-path regression gate. PRODUCT-002's final card reached
  **231.634M/s at p50 156us** in-L2 and **200.515M/s at p50 203us** out-of-L2 at batch 65,536. Layer-1 rooflines were
  **1,443.1/1,449.9 GB/s**; the 48M-row fixture built in **1,707.7s** with zero late residency work.
- The accepted prepared-route card reached **229.397M/s at p50 158us** in-L2 and **200.834M/s at p50 200us**
  out-of-L2 at batch 65,536. Layer-1 rooflines were **1,480.8/1,439.7 GB/s**; the 48M-row fixture built in
  **1,722.8s** with zero late residency work.

## Resume here

1. Continue **PRODUCT-001** with real keyed CancelRequest ownership on `gpu-db-engine-server`, including
   BackendKeyData, wrong/stale-key no-op behavior, active-request cancellation/recovery, and the canonical node-
   postgres smoke. Obtain a fresh independent ACCEPT before taking the prepared/portal/transaction-state,
   psql/catalog, or pg_dump/restore slices. Delete the legacy server and P8 endpoint only after every named
   compatibility gate is accepted. Broader transactional DDL and the conservative
   full-catalog conflict remain PRODUCT-001-owned gaps. Reduce the PLAN-owned 2,070-line
   `engine_dml_concurrent.rs` below 2,000 lines before PRODUCT-001 closes. Every independently reviewable slice needs
   a fresh read-only adversarial audit and re-audit to ACCEPT; the final whole-tree audit must prove one product
   server, one public facade mutation boundary, one WAL claimant, and one publication owner. Do not add workload
   behavior to the legacy endpoint.
2. Resume **BENCH-001** for its remaining seed/open-loop driver, tuned PostgreSQL profile, artifacts, quiet-window
   qualification, and sustained plus `B01`–`B10` execution. The mutation wrapper is already repaired and audited;
   do not redo it or substitute P8 microbenchmarks.
3. Take **DUR-001** only after preserving the accepted pre-DUR BENCH artifact, then rerun the affected cohorts for
   checkpoint-policy overhead. **ROUTE-001** and **SCALE-001** remain downstream of BENCH evidence.
4. Keep **RETIRE-002** outside these slices unless its device-native repair prerequisites are satisfied and PLAN
   explicitly promotes it.

## Last green evidence — 2026-07-21

- The accepted canonical TLS/SCRAM slice passes server **46/4 ignored**, pgwire **4/2 ignored**, SQLx **1**,
  tokio-postgres **1**, protocol **71 + 127**, workspace check, strict server Clippy, formatting, diff, shell, and
  source-size gates. The live canonical preflight passes fail-closed configuration, verifier and plaintext-source
  conflict checks, real TLS/SCRAM engine-backed DDL/DML/read, wrong password/user, recovery, non-TLS rejection, and
  Unicode SASLprep. Permanent raw startup, transcript, and both-ingress late-auth recovery tests pass. Two rejection
  rounds drove the repairs; the third independent frozen-tree audit returned **ACCEPT**. No report card applies to
  this transport/authentication-only change.
- The accepted canonical COPY slice passes engine **502/536 ignored**, SQL **49**, protocol **71 + 127**, facade
  **66/10 ignored** plus concurrency **13/1 ignored**, canonical server **31/4 ignored**, pgwire **4/2 ignored**,
  SQLx **1**, and canonical tokio-postgres **1**, plus strict affected Clippy, formatting, diff, and source-size
  gates. Raw blocking/async syntax precedence, phase timing, target ABA, zero-row, same-key concurrency, recovery,
  and confused-deputy sabotage pass. The live COPY lifetime gate passes three sequential plus two overlapping GPU
  cases. The final card records **231.468M/s at p50 156us** in-L2 and **198.067M/s at p50 205us** out-of-L2,
  **1,418.1/1,440.2 GB/s** rooflines, **1,674.9M elements/s** GROUP BY, and a **2,061.6s + 0.0s residency** 48M-row
  build. One independent rejection plus the fresh frozen-tree re-audit closed every finding and returned **ACCEPT**.
- The accepted PRODUCT-001 parsed-admission plus General atomic resource-envelope foundations pass engine
  **475/506**, facade **44/9**, server pgwire **3/1**, and concurrency **13/1** CPU/ignored gates plus strict affected
  Clippy, scoped rustfmt, source-size, and diff-whitespace checks. Three sequential and two overlapping fresh GPU
  HAZARD runs pass without CUDA 700/716/717. Both slices' independent rejection findings were fixed and their
  re-audits are ACCEPT.
- The accepted contiguous-publication slice passes engine **479/506**, four deterministic host coordinator/race
  tests, all 41 recovery tests, strict engine Clippy, and diff checks. Three sequential lane recovery GPU runs plus
  two overlapping lane/General/explicit GPU pairs pass with no CUDA 700/716/717. Its independent audit rejected
  lane-activation over-credit and non-wave terminal-gating holes; both were fixed and the re-audit is ACCEPT.
- The accepted live commit-authority consolidation passes workspace all-target check, strict engine Clippy,
  **479 passed / 505 GPU-ignored** ordinary engine tests, **44** recovery tests, focused coordinator/admission/retry/
  checkpoint tests, and `git diff --check`. The owner HAZARD matrix passed the complete 10-test lane suite three
  sequential and two overlapping times; the independent auditor also ran seven focused GPU lane/mixed-traffic/
  recovery tests. No run reported CUDA 700/716/717. The audit found one stale central ownership comment; it was
  corrected and re-audited to ACCEPT.
- The accepted typed prepared-command slice passes SQL **43/0**, facade **44/10**, facade concurrency **13/1**,
  workspace all-target check, strict SQL/facade Clippy, diff/source-size checks, and a two-test GPU W1 matrix three
  sequential plus two overlapping times with no CUDA 700/716/719. The independent auditor rejected double-cast
  identity, nested slots, zero-arity reparse, raw-parameter leakage/DDL panic, and missing NUMERIC typmod coercion;
  all were fixed, independently probed, and re-audited to **ACCEPT** with an explicit-cast GPU W1 pass.
- The accepted canonical extended-query lifecycle passes server **21/2**, SQL **43/0**, facade **46/10**, engine **483/505**,
  raw pgwire and tokio-postgres ingress tests, workspace check, strict affected engine/facade/protocol/server
  lib-bin-test Clippy, and diff/source-size gates. The final GPU matrix covers exact prepared W1 plus raw atomic
  rollback-after-later-Execute-error; it passed three sequential invocations (six cases) and two overlapping
  invocations (four cases) with no CUDA 700/716/719 output. The canonical report card exits 0:
  raw rooflines are **1,482.1/1,440.1 GB/s**, grouped aggregation is **1,675.6M elements/s**, and batch-65,536 compact
  point reads are **217.864M/s at p50 175us** in-L2 and **188.584M/s at p50 221us** out-of-L2; the 48M-row fixture
  built in **1,628.8s** with zero late residency work. The first audit returned REJECT; failed-state and duplicate
  precedence, implicit rollback cleanup, Bind SQLSTATEs, raw atomic-rollback proof, malformed Query/Sync recovery,
  and stale documentation were corrected; the final independent adversarial re-audit verdict is **ACCEPT**. At that
  lifecycle boundary, transactional DDL, broader codecs, and bounded response streaming remained sequenced under
  PRODUCT-001/SCALE-001 rather than being claimed by the lifecycle slice; the later single-CREATE and composite
  transaction boundaries are recorded above.
- The accepted Describe-revalidation slice passes facade **50/10** and server **24/2**, blocking and async raw-wire
  stale-metadata regressions, server all-target check, strict facade/server Clippy, rustfmt, and diff checks. The
  first independent audit rejected missing stale-parameter and new-path precedence coverage; both were added and the
  re-audit verdict is **ACCEPT**.
- The accepted prepared route/class and isolation slice passes its 15-test actual-GPU matrix sequentially and in two
  simultaneous processes; ordinary engine **487/520 ignored**, facade **46/10**, execution **52/75**, and facade
  concurrency **13/1** gates; workspace check; strict five-crate all-target Clippy; rustfmt/diff/source-size checks;
  and the final two-layer/two-cache card. The final card records **229.397M/s at p50 158us** in-L2 and
  **200.834M/s at p50 200us** out-of-L2, raw rooflines **1,480.8/1,439.7 GB/s**, GROUP BY **1,674.9M elements/s**,
  and a **1,722.8s** 48M-row build with zero late residency. Independent audits exposed and drove fixes for key
  choice, scan fallback, allocator leaks/races, purge metadata, vacuous coverage, and CUDA-pointer ABA; the final
  frozen-tree verdict is **ACCEPT**.
- PRODUCT-002's final candidate passed three sequential plus two simultaneous HAZARD runs at 2/2 each with zero
  CUDA 700/716/717. Independent per-slice and final adversarial audits are clean after every concrete finding was
  adopted, including API result-discard, NULL arithmetic, and old typed-WAL compatibility seams.
- The canonical two-layer/two-cache report card completed with exit 0. Production compact reached the best
  PRODUCT-002 result of **231.634M/s at p50 156us** in-L2 and **200.515M/s at p50 203us** out-of-L2; raw rooflines were
  **1,443.1/1,449.9 GB/s**, isolated gather **349.5/155.3 GB/s**, and GROUP BY **1,674.2M elements/s**. The out-of-L2
  point path is 0.8% above the immediately preceding slice and within 0.7% of the historical best. The 48M-row
  fixture built in **1,707.7s** with zero late residency work.
- Tokio-postgres, SQLx, and node-postgres smokes pass. Full psql/application-driver harnesses were locally
  prerequisite-limited by absent libpq connection variables and missing Python 3.14 `pip`, not by product failures.
- BENCH-001's stale mutation boundary is repaired and sabotage-gated, but no sustained/peak result exists yet.
  READ-002 and PRODUCT-002 canonical milestones are complete; PLAN now promotes PRODUCT-001 before the remaining
  runner and campaign work. DUR-001 remains blocked until the pre-DUR result is preserved.

## Required operations

- Read `AGENTS.md`, `docs/CHARTER.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/PLAN.md`,
  `docs/STATUS.md`, and `docs/CODE_SIZE.md` before changing runtime, storage, or scheduling.
- Never use `--gpu-reset`. Serialize ordinary GPU sweeps, use timeouts, and clear stale generated `target/tmp`
  artifacts before a long full-GPU run if disk headroom is low.
- Run `scripts/benchmark_report_card.sh` after any read-kernel, residency-layout, or result-path change; compare
  ratios rather than absolute bandwidth.
