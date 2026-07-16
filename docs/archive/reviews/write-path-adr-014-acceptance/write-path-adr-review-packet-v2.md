# Frozen R3-001 independent-review packet — remediation 2

> Archived frozen review manifest. Non-actionable; current work lives only in `docs/PLAN.md`.

**Frozen:** 2026-07-15\
**Baseline commit:** `f701d8b6e9e0a9a1904bc990f23162632f382045`\
**Decision state:** proposed, explicitly not accepted\
**Prior review:** [`write-path-adr-final-independent-review.md`](write-path-adr-final-independent-review.md),
**REJECT**\
**Review question:** Does the remediated packet contain a coherent, source-grounded GPU-native write design that may
be accepted, or does any design-acceptance blocker remain?

This manifest freezes the replacement packet submitted after the first final independent review. The worktree
contains unrelated user changes outside this list; they are not part of R3-001 evidence. `docs/DECISIONS.md` and
`docs/ARCHITECTURE.md` have no diff from the baseline and remain authoritative until a separate explicit acceptance
decision.

## Review inputs and SHA-256

| SHA-256 | File |
|---|---|
| `85458d4bf5cea1f91f86eb3852462e31302c27b0420b3012ae6c9b59edd35215` | `docs/HANDOVER.md` |
| `51b7a556cb094f473c627a510b1c8661bd6d56c0cc036538de35c0b389294ee8` | `docs/PLAN.md` |
| `02332d2ae890605e1f650bc6c84379e1df8d0d31f25ee511d17bb497ab5da838` | `docs/STATUS.md` |
| `6bc2e816e932d9aceb4bbcb870afcd721a46c8048fbffb29bf6309ca2cccca67` | `docs/design/write-path-design-inputs.md` |
| `ec74bdeb79b9de628cb39935dc2e01187e20827bd581f70c5b7eb327897cdfab` | `docs/design/write-path-adr-acid-audit.md` |
| `40cd93dbc21a5d0f8cdf5fba5a4830caec06e703ddede0de2ad26e1cd3286684` | `docs/design/write-path-adr-consistency-audit.md` |
| `6059445650c1b8b777ec35b0c0eeff5eb8597ed7671fab4770a18dbcf111ec1a` | `docs/design/write-path-adr-durability-audit.md` |
| `747735f305333c2b59642a50a68466a3689d7309c4d4a1d24ceadbd238b96d4a` | `docs/design/write-path-adr-evidence.md` |
| `1e484518b16b562d53432cfe224555b2cc8d8ee53df323489e01f2b6f760326a` | `docs/design/write-path-adr-final-independent-review.md` |
| `bf7b9e313d07d631cc8e3b9020c6a9025949bb7382757c15c8a254171ff500b9` | `docs/design/write-path-adr-performance-audit.md` |
| `86fccef91a303b576b1853118b7725120b56650a98c15c16095b4e266b582fda` | `docs/design/write-path-adr-physical-selection.md` |
| `93ee0affcb10cf44f37064a2b37a397a15aa5b3567295a1ec7db89a7f6776c20` | `docs/design/write-path-adr-proposal.md` |
| `75d77625f2822d1eff13de44a04db35d9c7fb5ac499fc7557dc96f6a4c18a0ac` | `docs/design/write-path-adr-review-matrix.md` |
| `73005df6b65244e6e46bfe65f8541ebc441168781321e5cfd7a6b2b673b604c5` | `docs/design/write-path-adr-review-packet.md` |
| `681581f5a11d28720862d7b2be9ef3c05bfb6437c731ca4a99df4d2edd386b4d` | `docs/design/write-path-adr-rto-capacity.md` |
| `390ec047b111d8affcdb3eb8e539b13e5ae8c6271aff4ffe86503675239c9c5a` | `docs/design/write-path-adr-slo-footprint.md` |
| `a50089b8bdc8938be11b40fa73c15af42d847ab2e3f95df4b689ccfaf331b141` | `docs/design/write-path-adr-traces.md` |
| `690ada24c3276a8af531357fcbab64cc55316fd7974545626a4221e127b0354f` | `crates/engine/examples/intent_fast_path_bench.rs` |
| `41cc357d1b562ad096e49a3994c4986d95e68266417cd4f63114980c67436962` | `crates/engine/examples/wal_recovery_time_probe.rs` |
| `6ea6055e769b714e05c500a112fee000be398a26f201b6fb91ed24b1d7bfc2f4` | `crates/engine/src/engine_dml_concurrent/lane.rs` |
| `ce4079153ee04c7432f3ebce1776b60c97e4f646360ceb56139b0eb1d75543bb` | `crates/engine/src/engine_dml_concurrent/lane_apply.rs` |
| `784f9e5fff551f4fbfcc2ba216653e9d4c38606128d5ecf7e97483b61b247419` | `crates/engine/src/engine_intent_lanes.rs` |
| `d874a88afb048b2d2e098e7f2ac9b6333cd392576f3d6da77e96a70b4bbf54a8` | `crates/engine/src/engine_residency.rs` |
| `ff0264581c46f2436ab404ff85395181786f41d22104848a2948d11ed272d677` | `crates/engine/src/engine_streaming_exec.rs` |
| `8347aaccfe95b5e51d59f5971d52117001fb67dd6e8e7e58540dd7c23c4f40b1` | `crates/engine/src/engine_wal_archive.rs` |
| `7a075e640fcd5f5720f3c2c9c92669c4a83ba84cce7b5fe93074f5f5d2d754e3` | `crates/engine/src/tests/streaming_exec/cold_checkpoint.rs` |
| `d791e287cc5c483c169c11d28bee2e03802bc2762d5ee9a49e09b6de41a62bb0` | `crates/engine/src/tests/streaming_exec/sidecars.rs` |
| `5034b96c308eea2f9df8e7fdb813bab11072b3581c4c38c75b4804cc4f159d54` | `crates/execution/examples/write_path_candidate_ab.rs` |
| `dafbd8f29e56537b6008daa101130d883971093c854b53b72a46c2e234ae1dac` | `crates/write_conveyor/examples/fua_frame_log_bench.rs` |
| `5249c5a36a37c877e0eec52f47484c5537abe729d0ae4bbb43bfca4a402ba845` | `crates/write_conveyor/examples/fua_wal_client_bench.rs` |
| `38eca0dc2a5e0f9728664bc6829310d4aac759727b568fa17ea7a2de515a249e` | `crates/write_conveyor/src/fua_frame_log.rs` |
| `fbd222d400f93a9a0d2a53e2a89dcf94f4c51938c055756bb6846059159e4eda` | `crates/write_conveyor/src/fua_wal.rs` |
| `54ec7f3a587b4ca65e336a01711daa1bd969c4ed7215aab689f6eb3f1bdf630a` | `crates/wal/src/fua_lanes.rs` |

## First-review blocker remediation

| First-review blocker | Remediation submitted |
|---|---|
| Proposal selected append/tombstone normatively but alternatives said no representation was selected. | Proposal, alternatives, evidence, PLAN/STATUS/HANDOVER, and N-02/N-22 consistently select compact append/tombstone. |
| N-14/N-15/N-22 had failed or absent physical/controller/footprint evidence. | The actual engine-facing `FuaFrameLog` rate plus a clearly labeled same-physics fixed-record percentile harness isolate the common synchronous-durability envelope without calling the harness the production lane. A build-only resident GPU A/B covers 8/32/128-byte rows, 1/3/6 indexes, batches 1/256/4,096, exact bounded-format history/index bytes, and snapshot-age growth. It selects A. PERF/GC traces plus fail-loud durability-profile qualification close the design-level controller/pressure rules; canonical end-to-end and sabotage results remain explicit post-acceptance production graduation. |
| Frozen provenance incorrectly described a documentation-only packet. | Evidence now lists the exact review-only Rust, benchmark/probe, build-only PTX/example, and document scope; this manifest hashes each in-scope file. |
| Completed R3-006 was described as future work. | Proposal, audits, and design inputs assign final publication implementation to R3-003/DUR-002; R3-006 remains only historical/current-path evidence. |

## Evidence and verification submitted

- Actual-code crosswalk covers the facade/session boundary, production engine intent lanes, FUA conveyor, resident
  shards, chunk-authoritative STRATA, indexes, WAL/checkpoint, and recovery. Prototype conveyor types are explicitly
  excluded as relational authority.
- Normative rules N-01 through N-23 contain no X. N-24 alone remains X until this frozen packet receives a fresh
  independent review.
- The same-physics fixed-record queue-depth-one harness measured 4,000 acknowledgements at 1.662-ms p50 and
  1.723-ms p99; a queue-depth-16 run measured raw FUA writes at 2.433-ms p50 and 2.951-ms p99. The actual
  engine-facing variable-payload `FuaFrameLog` completed 4,000 queue-depth-one fences in 6.169 seconds, or
  1.542 ms/fence on average. The packet does not mislabel the fixed-record harness as `FuaWalLaneSet`.
- Three consecutive full physical A/B runs found append/tombstone faster at p50 in all 27 cells per run. Depending
  on width/fanout/batch, dense-latest/undo was 1.11–1.94x slower. The semantically complete bounded current,
  history, and index formats are byte-tied; an earlier B undercount was corrected before this freeze.
- `cargo check` and strict Clippy pass for the 642-line build-only example; `git diff --check` passes. The earlier
  509/487 engine and focused 4/4 GPU R3-006 results remain unchanged because this remediation adds no production
  runtime path.
- The current end-to-end matrix remains FAIL and the current approximately 816-B narrow allocation is not accepted
  as canonical. Synchronous acknowledgement/RPO and charter targets are unchanged. A durability floor that consumes
  the residual budget must make the advertised low-latency profile unqualified; it cannot trigger async mode or a
  representation switch.
- The 292.18-second RTO equation remains a conditional design bound whose restore/replay floors are mandatory
  fail-loud production qualification.

## Review protocol

The reviewer is independent of the integration work, may inspect the frozen files and actual source, and must not
edit the packet. It must:

1. verify every hash and the absence of a `DECISIONS.md`/`ARCHITECTURE.md` diff;
2. compare the decision with the actual production code, STRATA placement, and FUA conveyor rather than accepting
   document assertions or prototype-conveyor behavior;
3. adversarially test whether separating the common durability floor from physical selection is legitimate and
   does not weaken latency, synchronous acknowledgement, RPO, or the pre-acceptance standard;
4. determine whether N-14/N-15/N-22 are genuinely closed at design-selection level or whether the deferred
   canonical implementation evidence hides a remaining design choice;
5. state one verdict—**ACCEPT**, **REVISE**, or **REJECT**—and list any acceptance blockers in severity order with
   exact file/section or code citations.

No accepted ledger or architecture edit is authorized by this review alone. Even an **ACCEPT** review is an input
to the user's explicit acceptance decision, not that decision itself.
