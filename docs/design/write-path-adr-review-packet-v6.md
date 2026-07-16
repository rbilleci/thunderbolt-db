# Focused R3-001 independent-review packet — target refinement 6

- **Frozen:** 2026-07-16
- **Pre-change commit:** `aa7ea543`
- **Decision state:** proposed, explicitly not accepted
- **Prior focused review:** packet v5 returned **REVISE** with four blockers

**Review question:** Does this delta close v5's class-derivation, full-profile-qualification, throughput-acceptance,
and stale-benchmark blockers without weakening the proposed GPU-native write design or relabeling current evidence?

This is a focused correction review, not a product benchmark or a rewrite of packet v4's accepted physical/ACID/
durability design verdict. Open review state and sequencing remain exclusively in `PLAN.md` under R3-001.

## Binding target contract

| Class | Envelope | p50 | p99 | p99.9 |
|---|---|---:|---:|---:|
| R1 | one prepared bounded point/page read | <0.5 ms | <1 ms | <5 ms |
| W1 | one keyed synchronous INSERT, UPDATE, or DELETE | <0.8 ms | <1.5 ms | <5 ms |
| T8 | 2–8 predeclared operations, at most four mutations, within declared resource bounds | <1.5 ms | <3 ms | <10 ms |
| T32 | 9–32 predeclared operations, at most 16 mutations, within declared resource bounds | <3 ms | <6 ms | <20 ms |

Every class and every W1 operation passes independently; equality fails and pooled distributions are supplementary
only. Write/transaction profile qualification checks p50, p99, and p99.9 durability plus percentile-matched bounded
downstream margins. The scheduler uses the admitted class's residual p99 budget, but that calculation cannot qualify
the complete profile.

## Binding throughput decision

The >100,000 sustained and ≥400,000 peak figures mean aggregate **committed transactions/s** for this deterministic
repeating schedule:

| Class | Transactions / 200 | Canonical work |
|---|---:|---|
| R1 | 120 | one-operation read |
| W1 | 50 | 35 INSERT, 10 UPDATE, 5 DELETE |
| T8 | 20 | exactly eight operations and four mutations |
| T32 | 10 | exactly 32 operations and 16 mutations |

The schedule contains 650 logical operations, or 3.25 operations/transaction. Therefore the same passing runs must
also report >325,000 logical operations/s sustained and ≥1,300,000 logical operations/s at peak. Those rates are
derived report values, not alternative gates. The sustained gate is ten contiguous post-warm-up minutes. The peak
gate is ten individually passing one-second 400,000-scheduled-TPS bursts; each preserves the mix and latency targets,
achieves at least 400,000 TPS, and restores pre-burst queue bounds within one second. Best-window extrapolation,
another mix, or standalone class capacity cannot pass the system gate. Standalone class sweeps remain required
diagnostics with no separate charter throughput threshold.

BENCH-001 must freeze exact SQL/route manifests and numeric post-image/WAL-byte, index-fanout, touched-table, cold-
access, and result bounds before either engine runs. Missing or changing that manifest invalidates the comparison.

## v5 findings and submitted corrections

1. **Class escalation:** `admit_sync_request` now derives W1/T8/T32 from the predeclared operation/mutation shape and
   all five resource dimensions. Only `AdmittedSyncRequest` reaches wave/deadline or profile qualification. The
   executable admits exact lower/upper class boundaries and rejects W1→T8/T32, every out-of-range operation/mutation
   count, undeclared work, and each resource bound plus one.
2. **p99-only qualification:** `LatencyPercentiles` validates monotonic p50/p99/p99.9 profiles and strict
   percentile-matched margins. All nine class/percentile equality cases fail; one microsecond below every boundary
   passes. The observed 1.662/1.723-ms p50/p99 floor remains unqualified for W1 and is no longer called T8-qualified.
3. **Throughput ambiguity:** the charter, ADR-008, architecture, proposal, and BENCH-001 now use the single aggregate-
   TPS/reference-mix decision above. The Candidate-A I/U/D-only report is explicitly diagnostic for throughput and
   remains a W1 latency/coverage/footprint failure.
4. **Stale benchmark:** `oltp_commit_slo_benchmark` now labels its isolated CPU-only INSERT TPS diagnostic, prints
   W1's 0.8/1.5/5-ms latency reference, and points aggregate throughput acceptance to BENCH-001.

The correction does not change compact append/tombstone selection, ACID semantics, WAL/marker format, publication,
RPO/RTO, recovery capacity, STRATA placement, or the host-control-plane invariant.

## Review inputs and SHA-256

| SHA-256 | File |
|---|---|
| `5720634a885b113da3436af4e03926ae04eeaf9c352b37087a05098c2c40471a` | `docs/CHARTER.md` |
| `8ec57cf2746ef9485ada28731f7fcbdac27a3ee9622d86137a54dce4abae67bd` | `docs/DECISIONS.md` |
| `d51f697079ad8a97a0c2ced9a525cd028fc9fde18f3de7a41b32cf94c265bda6` | `docs/ARCHITECTURE.md` |
| `f574965b0945c6b26f4244e988a84bbcdd6cc4dd957f47d795ae2f4d43c99d71` | `docs/PLAN.md` |
| `283775f3ed858e04e1b00f96561fd48754f9edb3759df0bf74f785692c5e2678` | `docs/STATUS.md` |
| `bf3cc3bc518e1dbc69719431eb60524b922c5bf49331798a3e70e332389b4d5c` | `docs/HANDOVER.md` |
| `2d593d0bb58f7d17c0e0d77863a77cf49134fe74c63498679e56d422bacecbfb` | `docs/design/write-path-adr-proposal.md` |
| `216df3ee87caf6030e5ee3451218db92098586bb28fed65a3aa98be212423642` | `docs/design/write-path-adr-slo-footprint.md` |
| `11a7e2608e381cd8fc2b4adb29624ff0e3fdd6b78963b9511853f0e525d07231` | `docs/design/write-path-adr-performance-audit.md` |
| `c61e84410adc181b4c4e0b69c7840041a243e17bf1d5b195aebef35377358b90` | `docs/design/write-path-adr-controller-injections.md` |
| `e0b3ab2d410b12fa6c5db01c83814497e35f1550812d19b98c0ff1a43a0b693d` | `crates/engine/examples/write_path_adaptation_injections.rs` |
| `fabb09e26599122921eb3f873a277fd6b2107a52e556992b4dbb5350c94dfc39` | `crates/engine/examples/oltp_commit_slo_benchmark.rs` |
| `2bb2573a3e453fb5220a24bb804a62a7029fd4c821141b96e93ff72ff3d7d9bc` | `docs/design/write-path-adr-review-matrix.md` |
| `92db2d43fcd2b1a70d8b55e3fe4204bd5306207190a461b5b99d5a6efb009e5c` | `docs/design/write-path-adr-rto-capacity.md` |
| `3bd69577b2a264eb8007bef0d258b1d979e3b3e148d33e7b23a3915a14e1755d` | `docs/design/write-path-adr-traces.md` |
| `c6c2a627247503e372d828d7c2e0d43df5f5d4b8612a77e33b744b4e4c53883c` | `docs/design/write-path-adr-evidence.md` |
| `fc1f5af9ebde4126f89e8ea26fe6d10574dba01343c3e9f441c4d847c4aeb7a4` | `docs/design/write-path-adr-review-packet-v5.md` |
| `348125bb3aab360b2577f8a553e8afac983bab0cff7244abe03f5a31020a9a1a` | `docs/design/write-path-adr-final-independent-review-v5.md` |

## Verification submitted

```text
rustfmt --edition 2021 --check \
  crates/engine/examples/write_path_adaptation_injections.rs \
  crates/engine/examples/oltp_commit_slo_benchmark.rs
cargo clippy -p gpu_db_engine \
  --example write_path_adaptation_injections \
  --example oltp_commit_slo_benchmark -- -D warnings
cargo run --quiet -p gpu_db_engine --example write_path_adaptation_injections
git diff --check
```

Strict Clippy passes and all 13 named controller families plus the aggregate pass. This changes build-only examples
and policy documents, not a production read kernel, residency layout, result path, or runtime controller; GPU HAZARD
and report-card gates are not triggered.

## Independent review protocol

The reviewer must not edit the packet or rely on the authoring verdict. It must:

1. verify all 18 hashes and confirm the delta from `aa7ea543` is limited to v5 remediation and review provenance;
2. prove the 200-transaction mix sums to 650 operations and that 100,000/400,000 TPS map only to the stated aggregate
   system gates and 325,000/1,300,000 derived operations/s, with no per-class, standalone, or best-window escape;
3. inspect class admission and sabotage every operation/mutation boundary, undeclared request, resource dimension,
   W1→T8/T32 escalation, and derived wave-budget boundary;
4. inspect strict p50/p99/p99.9 qualification, monotonicity, all nine equality failures, and the separation between
   p99 scheduler compatibility and complete profile qualification;
5. confirm all authoritative/proposal/plan consumers and the active isolated benchmark use the same interpretation;
6. confirm Candidate-A raw measurements are unchanged, remain a W1 failure, and are not claimed to test the new
   system throughput gate;
7. confirm no ACID, durability, acknowledgement, publication, recovery, STRATA, RPO/RTO, or physical-selection
   semantic changed; and
8. return exactly one verdict—**ACCEPT**, **REVISE**, or **REJECT**—with blockers cited by file and section.

An **ACCEPT** closes only the post-v4 target-consistency gap. The proposed write-path ADR still requires the user's
explicit acceptance decision.
