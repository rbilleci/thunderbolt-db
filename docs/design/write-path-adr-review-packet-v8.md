# Focused R3-001 independent-review packet — target refinement 8

- **Frozen:** 2026-07-16
- **Pre-change commit:** `aa7ea543`
- **Decision state:** proposed, explicitly not accepted
- **Prior focused reviews:** packets v5, v6, and v7 returned **REVISE**

**Review question:** Does this replacement close every v5/v6/v7 target-integration and workload-determinism blocker
with one executable, non-gameable acceptance contract while leaving the proposed GPU-native write design unchanged?

This is a focused target/workload review, not execution of BENCH-001 and not a rewrite of packet v4's accepted
physical/ACID/durability design verdict. Open work and sequencing remain exclusively in `PLAN.md`.

## Binding class contract

| Class | Envelope | p50 | p99 | p99.9 |
|---|---|---:|---:|---:|
| R1 | one prepared bounded point/page read | <0.5 ms | <1 ms | <5 ms |
| W1 | one keyed synchronous INSERT, UPDATE, or DELETE | <0.8 ms | <1.5 ms | <5 ms |
| T8 | 2–8 predeclared operations, at most four mutations, within declared resource bounds | <1.5 ms | <3 ms | <10 ms |
| T32 | 9–32 predeclared operations, at most 16 mutations, within declared resource bounds | <3 ms | <6 ms | <20 ms |

Every class and W1 operation passes independently with strict boundaries and no pooled-latency escape. Complete
profile qualification checks p50/p99/p99.9 with a hard or same-trace joint downstream margin; independently sampled
stage percentiles cannot be added to manufacture a pass. Direct open-loop end-to-end class latency is authoritative.

## Immutable workload and throughput decision

[`oltp-benchmark-workload-v1.md`](oltp-benchmark-workload-v1.md) freezes before implementation:

- exact PostgreSQL schema/indexes and deterministic 10M account, 10M limit, 10M ledger, and 4M pending seed rows;
- complete SplitMix64 algorithm/seed, cycle permutation, 80% one-percent-hot/20% cold account selection, collision
  resolution, IDs, amounts, and no measured retry/replacement;
- exact zero-based ledger-insert and W1 DELETE ordinals, parameter-stream consumption, T8/T32 account pairing,
  amount assignment, signed deltas, directions, transfer IDs, and created sequences;
- exact prepared R1 and W1 SQL, T8's ordered four reads/four mutations, and T32's ordered 16 reads/16 mutations;
- numeric operation/mutation, post-image+logical-WAL-byte, index-fanout, touched-table, zero-cold-access, and result
  caps for every route; and
- exact warm-up, sustained measurement, named peak-cohort, queue-drain, reporting, and failure rules.

The repeating 200 transactions are 120 R1, 35 W1 INSERT, 10 W1 UPDATE, 5 W1 DELETE, 20 T8, and 10 T32. They contain
exactly 650 logical operations. Thus >100,000 sustained aggregate committed TPS implies >325,000 operations/s and a
passing 400,000-transaction peak cohort contains 1,300,000 operations/s. These are system-mix gates; standalone
class capacity and logical operations/s cannot substitute.

Sustained load is fixed rather than distribution-selectable. Warm-up schedules 3,300,000 transactions at
`warmup_start + floor(i*1e9/110000)` ns. Measurement immediately schedules 66,000,000 transactions at the same
formula relative to `measurement_start = warmup_start + 30s`. Sustained TPS counts only measurement-scheduled
terminal committed completions timestamped inside the fixed 600-second measurement window and divides by 600;
warm-up completions cannot inflate it. Every class latency passes from scheduled arrival, stage populations finish
at/below measurement-start values, and all stages drain to the idle bound within one second.

Peak uses fixed cohorts `B01`–`B10`. Each schedules exactly 400,000 transactions over one second at the manifest's
nanosecond formula. Cohort TPS is eventual terminal committed cohort count divided by that fixed arrival second, not
completions timestamped inside it. Every named cohort must commit all 400,000 requests, pass every class latency
envelope, and restore stage populations to/below pre-burst values within one second of the last arrival. Wall-clock
completion throughput and last-completion time are separate diagnostics; failed or missing cohorts cannot be
omitted or replaced.

## v5/v6/v7 findings and corrections

1. Full-envelope admission produces the only value consumed by wave and durability logic; exact lower/upper class
   shapes pass, while class escalation, undeclared work, every count overflow, and every resource cap+1 fail.
2. All nine class/percentile equality boundaries fail and one microsecond below each complete profile passes. The
   observed 1.662/1.723-ms floor remains W1-unqualified and is not relabeled T8-qualified.
3. System throughput has one unit, immutable route/data/access manifest, sustained completion-window definition,
   named peak-cohort definition, and operations/s consequence. BENCH-001 may execute but not choose these inputs.
4. `oltp_commit_slo_benchmark` labels isolated W1 TPS diagnostic and uses 0.8/1.5/5-ms latency.
5. `gpu_mixed_read_write_gate` labels >100,000 read QPS as a gate-local non-vacuity floor, not charter system TPS;
   `STATUS.md` carries the same qualification.
6. Candidate-A raw measurements are unchanged and remain a W1 latency/coverage/footprint failure, not a system-mix
   throughput result.
7. Sustained arrivals now have one exact evenly paced timestamp sequence, and throughput excludes warm-up
   completions. Poisson, clumped, closed-loop, or alternative timestamp generation is invalid.
8. Generated ledger/DELETE ordinals are explicitly zero-based and scoped. T8/T32 account pairing, debit/credit
   roles, amount-output consumption and assignment, entry-ID ranges, directions, and transfer IDs are executable
   from the manifest without a BENCH-001 choice.

No correction changes compact append/tombstone selection, ACID, WAL/marker format, acknowledgement, publication,
RPO/RTO, recovery, STRATA placement, or the host-control-plane invariant.

## Review inputs and SHA-256

| SHA-256 | File |
|---|---|
| `93a9a279297e2dffc5fa4829d0d97a09f4a7c63760d8a77e76ded2ec04612bc9` | `docs/CHARTER.md` |
| `6551dd2be10818348d1f98662fcfd463ed146326eb553dc53d318d80d2a16f6b` | `docs/DECISIONS.md` |
| `6bc94f9be26cbb54a41077bb2f99556f5c1f948bc75959148a68b0dd16a415f2` | `docs/ARCHITECTURE.md` |
| `7920f8a35a14c54e70714d8267fc74486eac2fde804d706df2c890060fb5f41e` | `docs/PLAN.md` |
| `f17c37c9c794dfc0d4e0c809353bc32184e4588ba1bda636bc375d8f32e07ec6` | `docs/STATUS.md` |
| `f842fe43a0b0f30333cbceff1b986833a511991bd9e4b6fa1d79986cdfe32f46` | `docs/HANDOVER.md` |
| `c69910565864c1f348284ac9c326c19952efae2e1518608b6d0e2564b7851f7e` | `docs/design/oltp-benchmark-workload-v1.md` |
| `57f217b55982cce9c146be98ae1fe4784a3ee50fb3ec0e44ca39d2eb57b8c188` | `docs/design/write-path-adr-proposal.md` |
| `39c74d541ca9353b9b85d50bbfee1d5045ccbd2e9abc9730b8a54ee73cf74e72` | `docs/design/write-path-adr-slo-footprint.md` |
| `b393432b3e602001584d165ebb99504a78c2d024656c7eb193ee7e2f469c114f` | `docs/design/write-path-adr-performance-audit.md` |
| `360a860d409addafd5d073f95fa452fc8c2c07c1c01518f55523f6b49b04b086` | `docs/design/write-path-adr-controller-injections.md` |
| `8fd37df34b1bb2d6ec17ebb334356a2e1fd0fd6127c3c6c476046cd1dcb9dc99` | `crates/engine/examples/write_path_adaptation_injections.rs` |
| `fabb09e26599122921eb3f873a277fd6b2107a52e556992b4dbb5350c94dfc39` | `crates/engine/examples/oltp_commit_slo_benchmark.rs` |
| `12426a7f69b15099969af860c2d88c8315fc8d5950988d9b181b56c3679dd328` | `crates/facade/examples/gpu_mixed_read_write_gate.rs` |
| `e0c4b094b87354015f2fe1957f39367c20b469a8e2291fb4c357ab4648c179c9` | `docs/design/write-path-adr-review-matrix.md` |
| `92db2d43fcd2b1a70d8b55e3fe4204bd5306207190a461b5b99d5a6efb009e5c` | `docs/design/write-path-adr-rto-capacity.md` |
| `3bd69577b2a264eb8007bef0d258b1d979e3b3e148d33e7b23a3915a14e1755d` | `docs/design/write-path-adr-traces.md` |
| `973ecda6a50ad3215f0d754680f40d143496882dc00cab8e37c75935b5e19a7e` | `docs/design/write-path-adr-evidence.md` |
| `fc1f5af9ebde4126f89e8ea26fe6d10574dba01343c3e9f441c4d847c4aeb7a4` | `docs/design/write-path-adr-review-packet-v5.md` |
| `348125bb3aab360b2577f8a553e8afac983bab0cff7244abe03f5a31020a9a1a` | `docs/design/write-path-adr-final-independent-review-v5.md` |
| `e584d88fac07580efab6e2b7fab915f611d65ee827018ad2dd44d1a85f842943` | `docs/design/write-path-adr-review-packet-v6.md` |
| `45ebf0ba6109f8998bdc2eec9fc8ae36cdcc0ef38728e3b816e648f1a06c919a` | `docs/design/write-path-adr-final-independent-review-v6.md` |
| `642779c975d186975b8789620ccdcd360f2eb07b9296b9a60ee2837a5ffb70e1` | `docs/design/write-path-adr-review-packet-v7.md` |
| `8a33d655b4ab82136735cca46fe5f510c5babb0767e3220c422bb67d5a4a7bdc` | `docs/design/write-path-adr-final-independent-review-v7.md` |

## Verification submitted

```text
rustfmt --edition 2021 --check \
  crates/engine/examples/write_path_adaptation_injections.rs \
  crates/engine/examples/oltp_commit_slo_benchmark.rs \
  crates/facade/examples/gpu_mixed_read_write_gate.rs
cargo clippy -p gpu_db_engine \
  --example write_path_adaptation_injections \
  --example oltp_commit_slo_benchmark -- -D warnings
cargo clippy -p gpu_db_facade --example gpu_mixed_read_write_gate -- -D warnings
cargo run --quiet -p gpu_db_engine --example write_path_adaptation_injections
git diff --check
```

Both strict Clippy commands pass and the model prints 13 named families plus the aggregate. The facade change is
comment/error attribution only. No production runtime, read kernel, residency layout, or result path changes, so GPU
HAZARD and report-card gates are not triggered.

## Independent review protocol

The reviewer must not edit the packet or rely on an authoring verdict. It must:

1. verify all 24 hashes and the exact delta from `aa7ea543` plus v5/v6/v7 review provenance;
2. independently reconstruct the seed rows, SplitMix64/permutation/selection rules, exact generator call order,
   zero-based ID domains, T8/T32 pairs/amounts, SQL/order, and route counts/resources, proving no BENCH-001 choice;
3. reconstruct every warm-up/measurement timestamp, confirm warm-up completions cannot enter sustained TPS, and
   verify the 200/650 arithmetic, all-population drain rule, peak denominator, fixed `B01`–`B10`, no replacement,
   and derived operations/s;
4. verify the 4M pending seed cannot exhaust under 3.3M warm-up + 66M measurement + ten 400k peak cohorts;
5. rerun and inspect admission/class-escalation/resource-bound, derived wave-budget, monotonicity, and all nine strict
   percentile-boundary scenarios, including direct end-to-end authority over component arithmetic;
6. search every active source/document consumer for stale write latency, standalone system-throughput, pooled,
   best-window, deferred-manifest, distribution-selectable arrival, or ambiguous cohort wording;
7. confirm Candidate-A measurements and all physical-selection semantics are unchanged; and
8. confirm no ACID, durability, acknowledgement, publication, recovery, STRATA, RPO/RTO, or host-control-plane
   invariant changed, then return exactly **ACCEPT** or **REVISE** with every material blocker cited.

## Review focus

The fresh review must explicitly try to falsify:

- that v7's sustained-arrival ambiguity is closed by one exact timestamp sequence and cohort-restricted counting;
- that every transaction parameter, generated ID, debit/credit role, amount, and row value is executable from v1;
- that >100,000/400,000 mean only canonical aggregate committed TPS and cannot be replaced by class QPS or ops/s;
- that direct open-loop end-to-end class latency, not component-percentile addition, is binding; and
- that this target-only correction did not silently accept or change the proposed write-path design.

An **ACCEPT** verdict means no remaining material blocker to the user's explicit ADR review decision. It does not
accept the ADR, claim the current implementation passes, or waive post-acceptance implementation/fault graduation.
