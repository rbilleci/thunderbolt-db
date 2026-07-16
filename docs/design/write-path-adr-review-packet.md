# Frozen R3-001 independent-review packet

**Frozen:** 2026-07-15\
**Baseline commit:** `f701d8b6e9e0a9a1904bc990f23162632f382045`\
**Decision state:** proposed, explicitly not accepted\
**Review question:** Does this packet contain a coherent, source-grounded GPU-native write design that may be
accepted now, or does any design-acceptance blocker remain?

This manifest freezes the exact files submitted to the final independent adversarial review. The worktree contains
unrelated user changes outside this list; they are not part of the decision packet. `docs/DECISIONS.md` and
`docs/ARCHITECTURE.md` have no diff from the baseline and remain authoritative until an explicit later acceptance.

## Review inputs and SHA-256

| SHA-256 | File |
|---|---|
| `42897e26c6dff3d2aa1a72ef2755d308f9caf9fe4c189fd9b63ac524e48cdb9f` | `docs/HANDOVER.md` |
| `560872d15fd4f938e854bf9364f06388f5d90a215a271cf467410cdb489fd083` | `docs/PLAN.md` |
| `c79d5a765a1be280c9a72aaab227dd346c3f60d7bb10600aeed08762e4d70fbf` | `docs/STATUS.md` |
| `e4adbdf61e9e4146c551a53710791fe988fc2f39f1d35291fb41db42f686db3c` | `docs/design/write-path-design-inputs.md` |
| `261ca22c53c33cfae36c062b9edb9f53db61ce56f79fa685971bc06d50fcafdf` | `docs/design/write-path-adr-acid-audit.md` |
| `6f759d626a2caa82fb5a090d697b68550a6bd50cded899052b2031d4a1a11372` | `docs/design/write-path-adr-consistency-audit.md` |
| `dd47f998c580fe971910a3e1c1011da96068194c1357f02cf3585cd6c4b4963a` | `docs/design/write-path-adr-durability-audit.md` |
| `c6b5ee178097173b31af6f404ea0690be3c283b4402a6cac036ea61115d1b0fa` | `docs/design/write-path-adr-evidence.md` |
| `bf69fa7ea7af78af4367e17457282f19d93fff3b8d68c7724210f6d2edbfb152` | `docs/design/write-path-adr-performance-audit.md` |
| `bf451635b60e2f0bec2356904a40a24a89227b26e3124eea69593d0b0818254c` | `docs/design/write-path-adr-proposal.md` |
| `cf9dff6dfe81e18d403c942e8faac3a6e1f72f64b4e600e4068a2468ef9df460` | `docs/design/write-path-adr-review-matrix.md` |
| `681581f5a11d28720862d7b2be9ef3c05bfb6437c731ca4a99df4d2edd386b4d` | `docs/design/write-path-adr-rto-capacity.md` |
| `43973945c344463bb351b4b2652d3bebae6629fa3cf13eff85b14c72cc3778ef` | `docs/design/write-path-adr-slo-footprint.md` |
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

## Evidence submitted

- Source/code crosswalk, compatibility deviations CD-01 through CD-11, and normative rules N-01 through N-24.
- 115 compact reviewed decision traces covering FT/TX/ISO/CON/SEQ/WAV/PUB/ACK/WAL/CKP/ACT/REC/MIG/PERF/GC.
- R3-006 reproduction and correction: exclusive local `[0,1)` at base 41 formerly published inclusive 42 rather
  than 41; empty/first/normal/exhausted conversion and both lag directions are now checked.
- Ordinary engine library: 509 passed, 487 ignored, zero failed. Four focused real-GPU lane/recovery/checkpoint
  gates passed. The intent benchmark and recovery probe compile, including build-only `probe-timing` footprint
  instrumentation.
- Candidate-A decision report: strict low-load and target/mixed SLO failure; actual narrow retained allocation and
  physical FUA bytes; explicit non-INT4/index-fanout refusal; physical choice reopened.
- Recovery report: current serial and FUA replay near 38–40k outcomes/s; current rotation remains O(full history);
  parameterized 292.18-second, two-attempt, fail-loud capacity profile.

## Known decision state supplied to the reviewer

The packet does not ask the reviewer to overlook an expected future implementation. It explicitly distinguishes
design-selection evidence from R3-002/003, DUR-001/002, RETIRE-002, and conditional HA-001 graduation. It also
contains three current **X** rows:

- N-14: the implemented controller fails the binding latency/overload evidence;
- N-15: the rejected candidate lacks canonical width/fanout and held-snapshot footprint evidence; and
- N-22: Candidate A failed, while no bounded competing or materially revised candidate comparison exists.

N-24 consequently remains X. The independent reviewer should reject acceptance if those rows are genuinely
required by the proposal, identify any additional contradiction or under-evidenced assumption, and distinguish a
design defect from a deliberately deferred post-acceptance implementation proof.

## Review protocol

The reviewer is independent of the integration work, may inspect the pinned files and actual source, and must not
edit the packet. The response must state one verdict—**ACCEPT**, **REVISE**, or **REJECT**—then list acceptance
blockers in severity order, cite exact files/sections or code where possible, and say whether the RTO capacity
argument is a valid design bound despite its future qualification floors. No accepted ledger or architecture edit
is authorized by this review alone.
