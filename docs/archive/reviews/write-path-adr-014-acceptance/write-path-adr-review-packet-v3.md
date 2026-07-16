# Frozen R3-001 independent-review packet — remediation 3

> Archived frozen review manifest. Non-actionable; current work lives only in `docs/PLAN.md`.

**Frozen:** 2026-07-15\
**Baseline commit:** `f701d8b6e9e0a9a1904bc990f23162632f382045`\
**Decision state:** proposed, explicitly not accepted\
**Prior reviews:**
[`write-path-adr-final-independent-review.md`](write-path-adr-final-independent-review.md) and
[`write-path-adr-final-independent-review-v2.md`](write-path-adr-final-independent-review-v2.md), both **REJECT**\
**Review question:** Does the remediated packet contain a coherent, source-grounded GPU-native write design that may
be accepted, or does any design-acceptance blocker remain?

This manifest freezes the packet submitted after review v2. The worktree contains unrelated user changes outside
this list; they are not part of R3-001 evidence. `docs/DECISIONS.md` and `docs/ARCHITECTURE.md` have no diff from the
baseline and remain authoritative until a separate explicit acceptance decision.

## Review inputs and SHA-256

| SHA-256 | File |
|---|---|
| `699e9d829e04a7bcd2bc78eaca7dd42ae9c8b19d1d81d49192d3ef91eff1cc03` | `docs/HANDOVER.md` |
| `0a7519ff886e5a536cb8bf44bdef52303f3a5972aad98eab52082264614c4290` | `docs/PLAN.md` |
| `ef8127859647c4b727f40f9444151c3e1a8b548df56b9ec3ac815db68083dc26` | `docs/STATUS.md` |
| `bf909f826b146279840210fe6da70edfb8e84b525d8bfae0c1414c0aab6e17b1` | `docs/design/write-path-design-inputs.md` |
| `ec74bdeb79b9de628cb39935dc2e01187e20827bd581f70c5b7eb327897cdfab` | `docs/design/write-path-adr-acid-audit.md` |
| `6f97d9938456f06549892354a12b8e54142d23fe765f21fb6926bc5c166c5e3b` | `docs/design/write-path-adr-consistency-audit.md` |
| `3f55c043cedefe2e08fc86b6c01110b2fd9b7a2a64d89ba00ef5f04c1432307c` | `docs/design/write-path-adr-durability-audit.md` |
| `f06a12a316b4141bd65b4974b1550d07e3c89a021284846d65732a56bd3d998a` | `docs/design/write-path-adr-evidence.md` |
| `1e484518b16b562d53432cfe224555b2cc8d8ee53df323489e01f2b6f760326a` | `docs/design/write-path-adr-final-independent-review.md` |
| `93f3b3bb386f6f21f6b26a9a395a7386671321aa8198138f94e2c8ed3c057fcb` | `docs/design/write-path-adr-final-independent-review-v2.md` |
| `23db5343b33ef7c80bf8f99e19bfa26d6a15fbb8090fd75862a1a762c43490b4` | `docs/design/write-path-adr-performance-audit.md` |
| `8410df2219d6229329e800c97e8dba472e6e0d49b61870e03f228165ac0f8717` | `docs/design/write-path-adr-physical-selection.md` |
| `2e3eafce71611e8ef2959e31252079a65f9ffaa2d0eeac45556bab49a5381368` | `docs/design/write-path-adr-controller-injections.md` |
| `b0905957402d031c85aeb32585dc01d47e3c7d10042164468756fb6941cf8fcb` | `docs/design/write-path-adr-proposal.md` |
| `501d643ecefe18644f7d1ebe157c4613ad70da7a9ee4e5ef2441a441b6345e14` | `docs/design/write-path-adr-review-matrix.md` |
| `73005df6b65244e6e46bfe65f8541ebc441168781321e5cfd7a6b2b673b604c5` | `docs/design/write-path-adr-review-packet.md` |
| `75dc935843a700a3dd5c101f89b01fcb6af044b8be35460fb4495c3a17fad265` | `docs/design/write-path-adr-review-packet-v2.md` |
| `681581f5a11d28720862d7b2be9ef3c05bfb6437c731ca4a99df4d2edd386b4d` | `docs/design/write-path-adr-rto-capacity.md` |
| `390ec047b111d8affcdb3eb8e539b13e5ae8c6271aff4ffe86503675239c9c5a` | `docs/design/write-path-adr-slo-footprint.md` |
| `2a4fb0f5a158a0606d44a1d17ad782d1479a824ef6c1d40f47b7d399856e8329` | `docs/design/write-path-adr-traces.md` |
| `690ada24c3276a8af531357fcbab64cc55316fd7974545626a4221e127b0354f` | `crates/engine/examples/intent_fast_path_bench.rs` |
| `41cc357d1b562ad096e49a3994c4986d95e68266417cd4f63114980c67436962` | `crates/engine/examples/wal_recovery_time_probe.rs` |
| `d4b9db283da3d4cbabd4f54f05e8512fca2aaf119568998b1e15ea7348be405d` | `crates/engine/examples/write_path_adaptation_injections.rs` |
| `6ea6055e769b714e05c500a112fee000be398a26f201b6fb91ed24b1d7bfc2f4` | `crates/engine/src/engine_dml_concurrent/lane.rs` |
| `ce4079153ee04c7432f3ebce1776b60c97e4f646360ceb56139b0eb1d75543bb` | `crates/engine/src/engine_dml_concurrent/lane_apply.rs` |
| `784f9e5fff551f4fbfcc2ba216653e9d4c38606128d5ecf7e97483b61b247419` | `crates/engine/src/engine_intent_lanes.rs` |
| `d874a88afb048b2d2e098e7f2ac9b6333cd392576f3d6da77e96a70b4bbf54a8` | `crates/engine/src/engine_residency.rs` |
| `ff0264581c46f2436ab404ff85395181786f41d22104848a2948d11ed272d677` | `crates/engine/src/engine_streaming_exec.rs` |
| `8347aaccfe95b5e51d59f5971d52117001fb67dd6e8e7e58540dd7c23c4f40b1` | `crates/engine/src/engine_wal_archive.rs` |
| `7a075e640fcd5f5720f3c2c9c92669c4a83ba84cce7b5fe93074f5f5d2d754e3` | `crates/engine/src/tests/streaming_exec/cold_checkpoint.rs` |
| `d791e287cc5c483c169c11d28bee2e03802bc2762d5ee9a49e09b6de41a62bb0` | `crates/engine/src/tests/streaming_exec/sidecars.rs` |
| `21f221da0bc19d9497886ae15256f3bd2fb64d07cbddba7a6478d9bb4dda51c6` | `crates/execution/examples/write_path_candidate_ab.rs` |
| `dafbd8f29e56537b6008daa101130d883971093c854b53b72a46c2e234ae1dac` | `crates/write_conveyor/examples/fua_frame_log_bench.rs` |
| `5249c5a36a37c877e0eec52f47484c5537abe729d0ae4bbb43bfca4a402ba845` | `crates/write_conveyor/examples/fua_wal_client_bench.rs` |
| `38eca0dc2a5e0f9728664bc6829310d4aac759727b568fa17ea7a2de515a249e` | `crates/write_conveyor/src/fua_frame_log.rs` |
| `fbd222d400f93a9a0d2a53e2a89dcf94f4c51938c055756bb6846059159e4eda` | `crates/write_conveyor/src/fua_wal.rs` |
| `54ec7f3a587b4ca65e336a01711daa1bd969c4ed7215aab689f6eb3f1bdf630a` | `crates/wal/src/fua_lanes.rs` |

## Review-v2 blocker remediation

| Review-v2 blocker | Remediation submitted |
|---|---|
| Candidate B copied the old live-death sentinel into undo instead of closing undo at the replacement commit, and did not fence the odd seqlock marker before data mutation. | The B kernel now fences immediately after publishing odd, stamps `undo_deleted = commit_seq`, fences before publishing even, and asserts the final even epoch and exact version intervals. Old-snapshot assertions require undo visible/current hidden; replacement-snapshot assertions require undo hidden/current visible. The corrected comparison was rerun. |
| N-14/N-15 claimed closure while the required cold/index/lag/skew/pressure sabotage remained measurement-pending. | The build-only controller model injects 12 bounded policy families: sparse/global skew, byte/service caps, cold/index preclaim, both first-gap lag directions, fence/profile qualification, held-snapshot demotion, pressure hysteresis, cold quota/disabled maintenance, hard intent/byte credits, overlap/yield, starvation override, and drain-resize refusal. Every family passes. Traces and the review matrix distinguish this pre-acceptance decision evidence from the mandatory production controller/fault campaign. |
| PLAN/STATUS/HANDOVER and packet provenance would need reconciliation before another freeze. | The active ledger, factual status, resume baton, proposal, evidence ledger, audit follow-ups, traces, and review matrix consistently describe both rejected reviews, both remediation sets, the failed current implementation gate, and the remaining independent-review/explicit-acceptance boundary. |

## Evidence and verification submitted

- The actual-code crosswalk covers the facade/session boundary, production engine intent lanes, FUA conveyor,
  resident shards, chunk-authoritative STRATA, indexes, WAL/checkpoint, and recovery. Prototype conveyor types remain
  explicitly excluded as relational authority.
- N-14/N-15/N-22 have bounded pre-acceptance measurement or executable evidence. N-24 alone remains X until this
  packet receives an independent review with no blocker and the user makes an explicit acceptance decision.
- The current end-to-end Candidate-A matrix remains **FAIL**, including strict latency/mixed tails, approximately
  816 B per narrow appended version, and narrow-only route coverage. The packet does not relabel that result.
- The common durability envelope is measured separately from representation-specific device mechanics. The actual
  engine-facing `FuaFrameLog` measured 1.542 ms/fence on average at queue depth one; the labeled same-physics
  fixed-record harness measured 1.662-ms p50/1.723-ms p99. This cannot cause an async switch or weaken RPO/SLO.
- The corrected resident-input GPU A/B covers 8/32/128-byte rows, 1/3/6 indexes, and batches 1/256/4,096. A is
  faster in every p50 cell; the semantically complete current/history/index formats are byte-tied. The final run
  completed after strict Clippy and explicit visibility/interval assertions.
- The adaptation executable prints all 12 named family passes plus its aggregate pass after a strict Clippy build.
  It never serves or mutates engine durable state and does not substitute for post-acceptance implementation tests.
- The 292.18-second RTO equation remains a conditional design bound whose canonical artifact/replay floors and byte/
  record caps are mandatory fail-loud production qualification.
- Direct `rustfmt` and strict Clippy passed for both build-only examples; `git diff --check` passed. No read kernel,
  residency layout, or production result path was changed, so the standard read report card was not triggered.

## Review protocol

The reviewer is independent of the integration work, may inspect the frozen files and actual source, and must not
edit the packet. It must:

1. verify every hash and the absence of a `DECISIONS.md`/`ARCHITECTURE.md` diff;
2. compare the decision with actual production code, STRATA placement, and the FUA conveyor rather than accepting
   document assertions or prototype-conveyor behavior;
3. inspect Candidate B's seqlock ordering and undo visibility, including both old and replacement snapshots;
4. adversarially determine whether the 12-family executable genuinely closes the pre-acceptance controller design
   choices or whether deferred production evidence still hides a design decision;
5. recheck ACID, durability/resilience, consistency/accuracy, throughput, latency, and automated-adaptation
   consequences for any new contradiction introduced by the remediations;
6. state one verdict—**ACCEPT**, **REVISE**, or **REJECT**—and list any acceptance blockers in severity order with
   exact file/section or code citations.

No accepted ledger or architecture edit is authorized by this review alone. Even an **ACCEPT** review is an input
to the user's explicit acceptance decision, not that decision itself.
