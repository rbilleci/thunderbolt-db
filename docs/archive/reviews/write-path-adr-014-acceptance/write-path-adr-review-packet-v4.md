# Frozen R3-001 independent-review packet — remediation 4

> Archived frozen review manifest. Non-actionable; current work lives only in `docs/PLAN.md`.

**Frozen:** 2026-07-15\
**Baseline commit:** `f701d8b6e9e0a9a1904bc990f23162632f382045`\
**Decision state:** proposed, explicitly not accepted\
**Prior reviews:** v1, v2, and v3, all **REJECT**\
**Review question:** Does the remediated packet contain a coherent, source-grounded GPU-native write design that may
be accepted, or does any design-acceptance blocker remain?

This manifest freezes the packet submitted after review v3. The worktree contains unrelated user changes outside
this list; they are not part of R3-001 evidence. `docs/DECISIONS.md` and `docs/ARCHITECTURE.md` have no diff from the
baseline and remain authoritative until a separate explicit acceptance decision.

**Historical scope note:** these hashes and the resulting v4 review apply to the tree at commit `e1861025` and the
then-uniform simple-OLTP latency target. The later accepted R1/W1/T8/T32 target refinement does not rewrite this
frozen evidence and is not covered by the v4 verdict; R3-001 owns its focused post-v4 target-consistency review.

## Review inputs and SHA-256

| SHA-256 | File |
|---|---|
| `1f5a7cb18727b465fc25735bb11ed5346a87b5290bb23e988fb338a51566c29e` | `docs/HANDOVER.md` |
| `052a92fce70f2165e72a8c28d2390b5314a827550ada0eb61632e75811f1a3eb` | `docs/PLAN.md` |
| `312ab2119c07fc8f1b8753738f962e73afe520e789d9d06a7302334e1233ec5b` | `docs/STATUS.md` |
| `40cd0d1624e5b6d77a08a1c66121d65ecac183c882f973dd4c758dd52284498d` | `docs/design/write-path-design-inputs.md` |
| `ec74bdeb79b9de628cb39935dc2e01187e20827bd581f70c5b7eb327897cdfab` | `docs/design/write-path-adr-acid-audit.md` |
| `cf522f5d7b101e71bebc213182524fdbbced24b0f109c99660e18d71d1abc3fb` | `docs/design/write-path-adr-consistency-audit.md` |
| `8bf1abd63f18243b5ef7bb47ee7b55baef65426119688ff3e4737ba0488cdd8e` | `docs/design/write-path-adr-durability-audit.md` |
| `716110a6ae6ac38ab58a7c3019c2b2cee85e8b1a6be8c63948dfb7fbb2dad566` | `docs/design/write-path-adr-evidence.md` |
| `1e484518b16b562d53432cfe224555b2cc8d8ee53df323489e01f2b6f760326a` | `docs/design/write-path-adr-final-independent-review.md` |
| `93f3b3bb386f6f21f6b26a9a395a7386671321aa8198138f94e2c8ed3c057fcb` | `docs/design/write-path-adr-final-independent-review-v2.md` |
| `8008333229f8217741b86fb025f8b0ba663cdd4c0c219066dba2cbf0cd51f5c7` | `docs/design/write-path-adr-final-independent-review-v3.md` |
| `a46accca5ec34418fa42cedca60326723711df7f7d2494b56c723d6a70f60e31` | `docs/design/write-path-adr-performance-audit.md` |
| `8410df2219d6229329e800c97e8dba472e6e0d49b61870e03f228165ac0f8717` | `docs/design/write-path-adr-physical-selection.md` |
| `c22e4de816ad50a93538c06aa0570f9f61e283f8296993ce03d5cbc7a58db314` | `docs/design/write-path-adr-controller-injections.md` |
| `749655b1f627f195298397919b73953180efb3da4ea37ab820524759c4bdb24c` | `docs/design/write-path-adr-proposal.md` |
| `e7ade0378474c95386d4184345c309acc49dc648695f63358fb8aa834ecf0a20` | `docs/design/write-path-adr-review-matrix.md` |
| `73005df6b65244e6e46bfe65f8541ebc441168781321e5cfd7a6b2b673b604c5` | `docs/design/write-path-adr-review-packet.md` |
| `75dc935843a700a3dd5c101f89b01fcb6af044b8be35460fb4495c3a17fad265` | `docs/design/write-path-adr-review-packet-v2.md` |
| `eefb11048de38e8882919cd8f5fc1b2a47313a88fa3ee36f4b8f4ed4b046e77c` | `docs/design/write-path-adr-review-packet-v3.md` |
| `681581f5a11d28720862d7b2be9ef3c05bfb6437c731ca4a99df4d2edd386b4d` | `docs/design/write-path-adr-rto-capacity.md` |
| `390ec047b111d8affcdb3eb8e539b13e5ae8c6271aff4ffe86503675239c9c5a` | `docs/design/write-path-adr-slo-footprint.md` |
| `724714369fd18a9846e3e95a144ab0cae1fa92a742a749ffef26a3213d330575` | `docs/design/write-path-adr-traces.md` |
| `690ada24c3276a8af531357fcbab64cc55316fd7974545626a4221e127b0354f` | `crates/engine/examples/intent_fast_path_bench.rs` |
| `41cc357d1b562ad096e49a3994c4986d95e68266417cd4f63114980c67436962` | `crates/engine/examples/wal_recovery_time_probe.rs` |
| `17f832c66e9a1af804bd72e93da61e609548f753bf61e775ed22a10f36dc0739` | `crates/engine/examples/write_path_adaptation_injections.rs` |
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

## Review-v3 blocker remediation

| Review-v3 blocker | Remediation submitted |
|---|---|
| The pressure model omitted soft and allowed a prior `Rejecting` state to return to `Normal` above lower. | Resident and cold budgets now each carry lower/soft/high/hard thresholds. Soft enters `Maintaining` with ordinary admission, high enters `Throttling`, hard enters `Rejecting`, and prior `Maintaining`/`Throttling`/`Rejecting` states cannot restore `Normal` until both budgets are at/below lower. Tests exercise resident and cold transitions, including `Rejecting` recovery. |
| The byte/service-cap test shipped only because age was already expired, and an individually oversized item waited indefinitely. | Independent byte-only and predicted-service-only queues now ship a one-item partial wave before the age deadline when the next item would cross the respective cap. A single item above either cap returns an explicit `RejectOversizedBeforeClaim` result. |
| The report, traces, matrix, and active ledger overstated closure of the flawed v3 model. | The v3 REJECT is retained as review evidence. Controller report, PERF-07, GC-03, N-14/N-15/N-24, PLAN, STATUS, HANDOVER, proposal, evidence ledger, and audit follow-ups describe the exact corrections and retain the implementation/graduation boundary. |

## Evidence and verification submitted

- All 12 named controller families and their aggregate pass after direct formatting and warnings-as-errors Clippy.
- Candidate B's corrected seqlock/undo evidence is unchanged; strict Clippy passes and the latest GPU A/B still
  selects Candidate A in all 27 p50 cells.
- The actual production-code/STRATA/conveyor crosswalk, common-FUA-floor separation, decision-level ACID/failure
  traces, and conditional 292.18-second RTO argument are unchanged from the areas review v3 found sound.
- The current implementation gate remains **FAIL**. The model is build-only and does not substitute for real
  queues, device indexes, STRATA maintenance, canonical SLO, destructive recovery, or restore-rate qualification.
- `git diff --check` passes; the accepted decision and architecture documents remain unchanged from the baseline.
  Both examples remain below the repository's 3,000-line example threshold. No read kernel, residency layout, or
  production result path changed, so the canonical read report card was not triggered.

## Review protocol

The reviewer is independent of the integration work, may inspect the frozen files and actual source, and must not
edit the packet. It must:

1. verify every hash and the absence of a `DECISIONS.md`/`ARCHITECTURE.md` baseline diff;
2. rerun both build-only executables and compare the design with actual production code, STRATA, and the FUA
   conveyor;
3. inspect byte-only, service-only, and individually oversized wave behavior before the age deadline;
4. inspect resident and cold lower/soft/high/hard transitions, all prior-state recovery paths, held-snapshot
   demotion/reclaim rules, and disabled-maintenance behavior;
5. recheck whether any deferred production evidence still hides a pre-acceptance design choice, while keeping
   implementation/graduation work out of this acceptance gate;
6. re-audit ACID, durability/resilience, consistency/accuracy, throughput, latency, and automated adaptation for
   contradictions introduced by the corrections;
7. state one verdict—**ACCEPT**, **REVISE**, or **REJECT**—and list any acceptance blockers in severity order with
   exact file/section or code citations.

No accepted ledger or architecture edit is authorized by this review alone. Even an **ACCEPT** review is an input
to the user's explicit acceptance decision, not that decision itself.
