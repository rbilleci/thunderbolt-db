# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts.

## Current boundary

- **INSERT-001 completed and received final independent acceptance on 2026-07-27.** Its frozen seal is HEAD
  `fa477aa86052bbef914c5ad2ef2dbc60ae284d93`, staged tree `ed0febc2e8eabc10852baa765e7b7dcaf9c61f48`, cached diff
  `189ad3c728357c7704aac2b3c660ca7b60aae5c6b263e415e4f1f8666191f1f8`, and 88 staged paths without drift.
- The exact 48M qualification artifact is
  `target/insert001-exact-seal-qualification-v3-48m-all-20260727`; the 17-case byte-equal/SQLSTATE differential is
  `target/insert001-postgresql-differential-exact-seal-v1-20260727`; and exact HAZARD is
  `target/insert001-exact-seal-hazard-v3-20260727`. The final 48M development/durable GPU rates are
  **853,624.942 / 459,141.726 rows/s**; 12 qualification trials, W1/recovery, route/probe/FUA/geometry, and durable
  reopen/retry evidence are accepted.
- The exact full A/B/C card is `target/insert001-canonical-full-ed0febc2-20260727/runner.log` (SHA-256
  `54fdb682900d8280124d0d6f183d4b76e4ba1c326014c5bcbaa7072787521d74`): Section B is
  **272,880,443 lookups/s at p50 113us** and Section C is **235,370,206 at p50 126us** after a **438.4s** build.
  Its final audit accepted the Section C whole-loop variance; PERF-002 remains the point-read baseline and floor.
- **PERF-002 completed and received final independent acceptance on 2026-07-26.** The canonical 1M-row,
  one-caller, production-compact batch-65,536 route is now permanently gated at **260M whole-run lookups/s** in
  both quick and full report cards. The fixed three-sample rule requires a 2/3 median pass; 1/3 is
  environment-invalid and 0/3 is candidate-performance failure. Malformed, reused, candidate/config/binary-drifted,
  or structurally incomplete evidence fails execution-invalid before Section C; GPU/context identity drift is
  environment-invalid.
- The accepted clean quick measured **270.893M/p50 116us**. The canonical full card passed the floor at
  **267.086M/p50 117us**, measured **250.007M/p50 139us** out-of-L2 after a **2,303.5s** fixed-fixture build,
  completed A/B/C, and removed its isolated target. Raw out-of-L2 roofline was **1,427.0 GB/s** and grouped
  execution **1,672.9 M-elem/s**; the post-card auditor found no material regression.
- **WRITE-001 is the active PLAN convergence/migration owner.** Its non-owning contract is
  [`design/write-001-general-insert-pipeline.md`](design/write-001-general-insert-pipeline.md): all INSERT inputs
  converge on move-only `TypedInsertBatch` and composable `DeviceInsertPlan`, with the accepted i32 append retained
  only as a physical strategy beneath the one canonical allocator/WAL/status/apply/poison/publication path.
- WRITE-001's 2026-07-28 checkpoint accepted device pre-WAL CHECK/primary-key NULL/dense-batch key/current-resident
  key proof, exact diagnostic ordering, default/sequence error precedence, and an inert test-only indexed in-place
  append/index-delta ownership proof. The accepted runtime/card seal is tree `e1a7713f633403a9552b9f202fc8c87932fe3305`,
  diff `eef525f70d0807f1f9ac882a9c675ed849d2fb37849fa7550c2f8e35e4b5ffde`, with 99 staged paths and no drift.
  HAZARD passed three serial plus two simultaneous cohorts; the canonical card is
  `target/write001-full-e1a7713f-20260728/runner.log` (SHA-256
  `1183ab7c8140e778640fa6df38b1973ee79ecb642cfec5b26fd6c3caee56c330`) and received post-card **FINAL ACCEPT**.
  Its B samples were **247.196M / 266.784M / 273.832M** (median **266.784M**, p50 **113us**); C was
  **257.313M**, p50 **129us**, after a **460.6s** 48M build.
- The accepted WRITE-001 checkpoint extends that inert proof to indexed fixed rollover without WAL, a live
  operation, or publication. It privately allocates the complete successor payload and all four distinct index
  generations before the first build, GPU-probes every destination against its sealed raw/fingerprinted key,
  reserves exact persistent bytes plus maximum sequential build/probe scratch, and hardens full-build default-stream
  draining. The first
  pre-card audit rejected launch/status-only non-vacuity; swapped-directory and reversed-compound sabotage now close
  that finding. Focused engine/execution CUDA gates pass **7/4**, ordinary suites pass **797/63**, three serial plus
  two simultaneous repaired HAZARD cohorts are clean, and the preserved repaired quick point-read median is
  **272.273M** at p50 **114–115us**. Independent pre-card re-audit accepted exact staged runtime tree
  `66222cc3bf4ef3493c2128250fa31b4a837a3a7a` and diff
  `e3f89bc4e473c9a145be48a0cc4b690d27b87cd12337eb09548fb78c8bdbf7fd` with no finding.
- Its single canonical card is `target/write001-index-rollover-full-66222cc3-20260728.log` (SHA-256
  `4f2127ddf2e82bacf840c5c466d8377827f7639fb956bea3e631ace42a6ef664`, **48,723 bytes**) and completed exact
  A/B/C evidence. Section A recorded **1,433.3 GB/s** out-of-L2 roofline and **1,674.2M elements/s** grouped.
  Section B passed all three samples (**273.334M / 272.526M / 263.766M**, median **272.526M**, p50 **112–114us**);
  Section C measured **246.077M** at p50 **130us** after a **463.1s** build. Documentation-only ledger updates
  follow the sealed runtime tree; independent post-card audit returned **FINAL ACCEPT** with no finding.
- WRITE-001's accepted 2026-07-29 inert effect/codec checkpoint gives the move-only typed batch one pre-effect
  statement digest shared by semantic preparation and the test-only effect terminal. Stable-OID sequence
  classification, autocommit/explicit currentness locks, private predecessor chains, and ordered typed
  `RETURNING` evidence remain side-effect-free. The private v1 codec adds an exact 100-byte header, eight ordered
  logical sections, a 16 MiB inclusive ceiling, complete captured catalog/index/FK/domain/sequence closure, and a
  hostile decoder that recomputes digests and byte-reencodes the parsed model. It has no live encoder/decoder
  caller, WAL opcode, row identity, device plan, apply, recovery, status, or publication authority.
- The repaired physical adapter declines unsupported constraint/`RETURNING`/requested-sequence shapes from the
  shared pre-semantic carrier only after catalog/generation/target/currentness checks, so fallback evaluates each
  scalar default once. The compatibility path reuses the typed `RETURNING` binder; supplied sequence columns remain
  eligible and malformed FK closure still fails closed in direct semantic preparation.
- Exact checkpoint gates pass canonical codec **27/27**, typed-batch **67 passed / 11 GPU-only ignored**,
  pre-WAL effects **42/42**, and the serial all-feature engine library sweep **893 passed / 661 GPU-marked
  ignored**, plus workspace all-target/all-feature check, strict Clippy, scoped formatting,
  whitespace, source-boundary, and sub-1,500-line production-leaf checks. Rehashed sabotage covers sequence
  geometry/mode/ownership, global catalog identities, temporal bounds, and catalog closure; raw structural
  sabotage covers the unrepresentable UTF-8 split-offset form. Independent architecture review returned
  **ACCEPT** with no blocker. GPU/HAZARD and report-card evidence are inapplicable because no device, residency,
  read-kernel, result, WAL, or apply behavior moved.
- The accepted codec-5 S1--S6 source checkpoint adds exact one-to-four-chunk aggregate, fixed status, borrowed
  outer-envelope, allocation-free S1/S4 traversal, strict S2/S3 decoding, and compact sequence/outcome closure.
  Its incomplete move-only draft has no replay conversion and deliberately stops before S7/S8. Independent repair
  audit accepted exact tree `a17b68e4a994a3339c5e4b56ce8bd620b7edabab` and diff
  `a3291bcfdcaec6af0ee4585797e8a1ac5fb9d8a5ee22623a404e2c0bd1efa159` across 172 staged paths with no
  finding.
- The accepted shared-image checkpoint adds one exact `GPUDBTYPEDIMAGE2` final-table/retained-response codec over
  the same v1 typed-vector grammar. It covers all nine types, NULL placeholders, zero-row forms, exact persistent
  and scratch allocation terms, raw pre-allocation sabotage, and one move-only decoded owner without a live
  consumer. Independent re-audit accepted exact tree `bd313d349e5258af33d4f9ac9f9662736b10dfa4` and diff
  `676b6bcb04864f2a36f0ce0e5a056cb6876af1a924c0c2b5d5cb0c23733968aa` across 177 staged paths with no
  finding; GPU/HAZARD/recovery/card gates are inapplicable to the inert host codec.
- The exact typed-INSERT-only semantics-v2 S7 contract is now frozen at
  [`design/write-001-codec5-semantics-v2.md`](design/write-001-codec5-semantics-v2.md), SHA-256
  `b673127af53148fb26a5e26e32ad00bacb3206aadea0440a155d1a8d05624453`. Independent architecture audit
  returned **ACCEPT** after closing logical `RETURNING`/retry formats, the full-success/final-abort matrix,
  constraint error identity, dependency/index owner and generation equality, ADR-014 lease authority, exact
  catalog/allocator/generation witnesses, final index roots, an acyclic generation graph, recovery witness phasing,
  terminal guard keys, synthesized NOT NULL identities, and the resolution/S1 request-digest echo. This checkpoint
  changes design only and remains inert.
- The accepted Q0 structural quarantine/reservation prerequisite dispatches semantics before v1 validation,
  preserves the sole v1 writer and literal v1 golden, performs allocation-free semantics-v2 S7 pass zero, reserves
  exact persistent plus maximum scratch ownership, and strictly fills source-backed S2/images into a private
  move-only quarantine owner. The initial audit rejected incomplete canonical-v1 global registries; repaired
  cross-kind OID, class-name, column-ID, external-domain, and `attnum` coverage plus an unmasked coherently rehashed
  matrix received final **ACCEPT**. Exact staged tree
  `166a1a1aba92e5854836d63a6946d2777ef61ba5`, diff
  `896c2aa60d8dc7d5d182996f1b6e444b70b2635d530506fb76f93628506648f4`, 41 source paths; focused suites and the
  **1,103 passed / 667 GPU-required ignored / 0 failed** serial engine sweep are clean. Q0 has no constructible
  fully validated state, reencoder, or live consumer and is not the complete normative S4/S7 checkpoint.
- Regenerable benchmark/build state was reduced from about **143 GB to 4.9 GB**. The runner now recreates the
  source-relative temporary directory required by clean quick and exported-full builds.
- **PRODUCT-001 completed and received final independent acceptance on 2026-07-26.** The superseded listener,
  host-relational module tree, callable P8 adapters, P8-only launchers, and direct-Engine benchmark endpoint are
  deleted. Ownership guards prove one product server, one public `SharedEngine::submit` boundary, and the sole
  commit/WAL/recovery/publication authority.
- The final PostgreSQL 16.14 migration gate passes **352/352** scenarios; all dump/restore, eight application-driver,
  TLS/SCRAM/mTLS, COPY, cancellation, prepared/portal, transaction, recovery, catalog, security, and product gates
  pass. The physical aggregate passes **697/697** GPU tests with `executed_target=Gpu(0)`, protected-path
  zero-fallback evidence, and fail-closed sabotage. The compatibility scorecard passes **1,866/1,866**.
- The former transaction-catalog mixed owner is a 594-line production root plus a 1,398-line test leaf.
  `engine_dml_concurrent.rs`, `engine_commit.rs`, and `engine_mutation_admission.rs` are below the
  comment-excluded production threshold; no source-size exception remains.
- Three pre-card audits drove the final host-authority, authorization, role-scope, catalog-join, sabotage,
  fail-open matcher, and point-read code-placement repairs. The repaired official quick screen restored the
  production batch-65,536 route to **269.642M lookups/s at p50 117us**.
- The accepted candidate seal is HEAD `573e5142…`, staged tree `1f721b31…`, and cached-diff
  `7684ee88…` across 178 paths without drift. After the default Section C timeout produced only incomplete
  evidence, the auditor authorized one workload-identical retry with `SECTION_C_TIMEOUT=2700`. It completed
  A/B/C with the canonical terminal record, removed its isolated target, and recorded **1,431.7 GB/s** out-of-L2
  roofline, **1,673.2 M-elem/s** grouped, **236.658M/p50 117us** in-L2 point reads, and
  **246.203M/p50 139us** out-of-L2 point reads after a **2,318.3s** fixture build. The post-card auditor accepted
  provenance and performance with no unresolved finding.

## Resume here

**WRITE-001** is the sole active **NOW** task. Q2 witness closure received architecture and final independent
**ACCEPT** on 2026-07-31 with no finding. Its exact source commit is
`f288df1c3f4ddd8ef4ff1212c74f6c0a305fe434`, tree `540275d1a581010a2f3d1196c2dc019407eefaec`,
cached-diff SHA-256 `d427b19ddee9b77dfd1147236301008c1fe1d369701fec3a3e3c7ed8db09f3e0`, and 11 source paths
without drift. Semantics-v2 passes **68/68** and the serial all-feature engine sweep passes
**1,168 / 669 GPU-required ignored / 0 failed**; workspace all-target/all-feature check, strict Clippy, scoped
formatting, source-boundary, diff, and size gates are clean. GPU/HAZARD, recovery execution, and cards were
inapplicable to the test-only host slice.

The exact S8 retained-response and replay design is accepted at
[design/write-001-codec5-s8-replay.md](design/write-001-codec5-s8-replay.md), file SHA-256
`acef17d5bf1a53bd12b9635add7b84648d09edf9effd5a383360692efb6fbd1c`, commit
`e15304618afc160e4d8bba0914b1021bb6572ad6`, tree
`fb768ad38b3df7cbf77b0842ee45916074a9cab9`; architecture and the independent acceptance audit both returned
**ACCEPT**. It fixes the 256-byte S8 header and adjacent 288-byte artifact, 32-byte row-selection, and image-arena
directories; distinguishes the inherited length-prefixed v1 response root from local v2 S8 digests; and closes
RetentionIntent/RetentionAuthorityPending, sealed live and historical no-retention proof, durable sequence proof,
source-guarded GPU replay, launch/drain/quarantine, and pre-WAL capacity ownership.

Resume with PLAN.md's next inert final-abort RETURNING repair plus nonempty S8
pass-zero/reservation/fill/local codec closure. It may introduce only the minimal
`AggregateReplayTxn<CodecQuarantined> -> RetentionAuthorityPending` shell. It must not add claim, catalog, sequence,
GPU, live, WAL, recovery, apply, publication, or other eligibility authority. Q2's accepted source facts above and
the CARD-001/COPY-001 blocks remain unchanged.

**CARD-001** is blocked on accepted WRITE-001, and **COPY-001** is blocked on both.
Preserve PRODUCT-001's sole server/facade/admission/WAL/recovery/publication authority, PERF-002's 260M point-read
floor, and INSERT-001's sealed qualification, differential, HAZARD, and canonical-card evidence in `STATUS.md`.
