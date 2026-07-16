# Focused R3-001 independent review v5 — target refinement

**Date:** 2026-07-16

**Packet:** [`write-path-adr-review-packet-v5.md`](write-path-adr-review-packet-v5.md)

**Reviewed commit:** `aa7ea543`

**Verdict:** **REVISE**

This was a read-only adversarial review of the post-v4 R1/W1/T8/T32 target delta. All 14 packet hashes matched,
the worktree was clean, the submitted formatting/Clippy/controller checks passed, the headline latency values were
consistent, pooling could not satisfy the written class gates, Candidate-A remained a W1 failure, and the delta did
not change ACID, acknowledgement, publication, recovery, STRATA, RPO/RTO, or physical-selection semantics.

## Blocking findings

1. **Class escalation was asserted but not modeled.** `choose_wave` accepted an arbitrary oldest-age budget and
   `SyncLatencyClass` was supplied directly. The executable had no operation, mutation, post-image/WAL-byte,
   maintained-index-fanout, touched-table, cold-access, or result envelope, so it could label W1-shaped work T8/T32.
2. **Synchronous-profile qualification checked p99 only.** The charter binds p50, p99, and p99.9 independently, but
   the model could return `QualifiedSync` while p50 or p99.9 failed. A p99 scheduler-compatibility check was
   overstated as complete profile qualification.
3. **The 100,000/400,000 throughput acceptance unit was undefined for T8/T32.** The documents did not decide whether
   those figures meant per-class transactions/s, logical operations/s, aggregate transactions/s for a canonical
   mix, or measurement-only capacity. With up to 32 operations per transaction, the ambiguity was material.
4. **An active benchmark retained the retired write target.** `oltp_commit_slo_benchmark.rs` still printed
   0.5/1/5-ms single-row INSERT targets instead of W1's 0.8/1.5/5 ms and did not distinguish its isolated capacity
   from the system throughput gate.

## Required correction boundary

The next packet must derive W1/T8/T32 from the complete admitted envelope before either wave or durability budget
selection; sabotage W1→T8/T32 escalation plus every count/resource boundary; qualify p50, p99, and p99.9 with strict
percentile-matched margins and equality failures; choose one binding throughput unit/reference workload; and update
or explicitly demote the stale benchmark. This verdict does not reject the underlying v4 physical/ACID/durability
design and does not accept the write-path ADR.

## Verification run

```text
sha256sum <all 14 packet inputs>                 # exact
git diff --check e1861025..HEAD                  # pass
rustfmt --check --edition 2021 crates/engine/examples/write_path_adaptation_injections.rs
cargo clippy -p gpu_db_engine --example write_path_adaptation_injections -- -D warnings
cargo run --quiet -p gpu_db_engine --example write_path_adaptation_injections
git status --short                               # clean
```

The executable printed all 12 then-modeled families plus its aggregate. Those passing assertions did not close the
class-derivation or full-profile blockers above.
