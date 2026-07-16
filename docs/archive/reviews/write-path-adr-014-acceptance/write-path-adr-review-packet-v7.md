# Focused R3-001 independent-review packet — target refinement 7

> Archived frozen review manifest. Non-actionable; current work lives only in `docs/PLAN.md`.

- **Frozen:** 2026-07-16
- **Pre-change commit:** `aa7ea543`
- **Decision state:** proposed, explicitly not accepted
- **Prior focused reviews:** packets v5 and v6 returned **REVISE**

**Review question:** Does this replacement close every v5/v6 target-integration blocker with one deterministic,
non-gameable workload and acceptance contract while leaving the proposed GPU-native write design unchanged?

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

## Immutable workload decision

[`oltp-benchmark-workload-v1.md`](oltp-benchmark-workload-v1.md) now freezes before implementation:

- exact PostgreSQL schema/indexes and deterministic 10M account, 10M limit, 10M ledger, and 4M pending seed rows;
- complete SplitMix64 algorithm/seed, cycle permutation, 80% one-percent-hot/20% cold account selection, collision
  resolution, IDs, amounts, and no measured retry/replacement;
- exact prepared R1 and W1 SQL, T8's ordered four reads/four mutations, and T32's ordered 16 reads/16 mutations;
- numeric operation/mutation, post-image+logical-WAL-byte, index-fanout, touched-table, zero-cold-access, and result
  caps for every route; and
- exact warm-up, sustained measurement, named peak-cohort, queue-drain, reporting, and failure rules.

The repeating 200 transactions are 120 R1, 35 W1 INSERT, 10 W1 UPDATE, 5 W1 DELETE, 20 T8, and 10 T32. They contain
exactly 650 logical operations. Thus >100,000 sustained aggregate committed TPS implies >325,000 operations/s and a
passing 400,000-TPS peak cohort contains 1,300,000 operations/s. These are system-mix TPS gates; standalone class
capacity and logical operations/s cannot substitute.

Sustained throughput is wall-clock terminal completions inside the fixed 600-second window after 30 seconds at
110,000 scheduled TPS, divided by 600. Stage populations must finish at/below their measurement-start values and
drain to idle within one second. Peak uses fixed cohorts `B01`–`B10`. Each schedules exactly 400,000 transactions in
one second; cohort TPS is the eventual terminal committed cohort count divided by that fixed arrival second, not
completions timestamped inside it. Every named cohort must commit all 400,000 requests, pass every class latency
envelope, and restore stage populations to/below pre-burst values within one second of the last arrival. Wall-clock
completion throughput and last-completion time are separate diagnostics; a failed or missing cohort cannot be
omitted.

## v5/v6 findings and corrections

1. Full-envelope admission now produces the only value consumed by wave and durability logic; exact lower/upper
   class shapes pass, while W1→T8/T32, T8→T32, undeclared work, every count overflow, and every resource cap+1 fail.
2. All nine class/percentile equality boundaries fail and one microsecond below each complete profile passes. The
   observed 1.662/1.723-ms floor remains W1-unqualified and is not mislabeled T8-qualified.
3. System throughput has one unit, immutable route/data/access manifest, sustained wall-clock definition, named
   peak-cohort definition, and operations/s consequence. BENCH-001 may execute but not choose these inputs.
4. `oltp_commit_slo_benchmark` labels isolated W1 TPS diagnostic and uses 0.8/1.5/5-ms latency.
5. `gpu_mixed_read_write_gate` labels >100,000 read QPS as a gate-local non-vacuity floor, not charter system TPS;
   `STATUS.md` carries the same qualification.
6. Candidate-A raw measurements are unchanged and remain a W1 latency/coverage/footprint failure, not a system-mix
   throughput result.

No correction changes compact append/tombstone selection, ACID, WAL/marker format, acknowledgement, publication,
RPO/RTO, recovery, STRATA placement, or the host-control-plane invariant.

## Review inputs and SHA-256

| SHA-256 | File |
|---|---|
| `1afcbe352136fa14fe39265896d3c2141d76f87ec069f2bf3c82dbeaf8c8bba0` | `docs/CHARTER.md` |
| `e96d6e98a9c9712ea3e11d649c0f5a637b1814dec06280b503ee469c156084dd` | `docs/DECISIONS.md` |
| `42d6677eeaeea11fd83d474ba5c8d5276a42c11e00480d04aaa86531a84ba83c` | `docs/ARCHITECTURE.md` |
| `19eaf19e077b130bbd75fd3f1f97bb4e8617362f720d789a9cc8b2f0f4a46b98` | `docs/PLAN.md` |
| `be646618162ee2cb6e27a6dcee16bff75d9e5de9d14e36defc191401833c1d9f` | `docs/STATUS.md` |
| `d86404720c84af704aabf148d18f1f87dc49cee5c963d554274ebd45a073504a` | `docs/HANDOVER.md` |
| `7654ae79840abcd63d35d12d7c4b3f7d5b40934d905bfd79a96b906830c3f394` | `docs/design/oltp-benchmark-workload-v1.md` |
| `77eb28180ddb8a3b90f98cd7afda66bd146d645a6f35595f01810f7d9f67c6e5` | `docs/design/write-path-adr-proposal.md` |
| `04aa1fcb8427076494d328c355401108706f446e81cb29b5ccc3573c1b7f8fa8` | `docs/design/write-path-adr-slo-footprint.md` |
| `b393432b3e602001584d165ebb99504a78c2d024656c7eb193ee7e2f469c114f` | `docs/design/write-path-adr-performance-audit.md` |
| `360a860d409addafd5d073f95fa452fc8c2c07c1c01518f55523f6b49b04b086` | `docs/design/write-path-adr-controller-injections.md` |
| `8fd37df34b1bb2d6ec17ebb334356a2e1fd0fd6127c3c6c476046cd1dcb9dc99` | `crates/engine/examples/write_path_adaptation_injections.rs` |
| `fabb09e26599122921eb3f873a277fd6b2107a52e556992b4dbb5350c94dfc39` | `crates/engine/examples/oltp_commit_slo_benchmark.rs` |
| `12426a7f69b15099969af860c2d88c8315fc8d5950988d9b181b56c3679dd328` | `crates/facade/examples/gpu_mixed_read_write_gate.rs` |
| `56cf86ae233a4181dc98de1a4f21bac986ec1c4d1f41c9679573e3579c865a35` | `docs/design/write-path-adr-review-matrix.md` |
| `92db2d43fcd2b1a70d8b55e3fe4204bd5306207190a461b5b99d5a6efb009e5c` | `docs/design/write-path-adr-rto-capacity.md` |
| `3bd69577b2a264eb8007bef0d258b1d979e3b3e148d33e7b23a3915a14e1755d` | `docs/design/write-path-adr-traces.md` |
| `c2f2127bdeb518372bb4306a49a9ff1e592f718a75d0da87cc96b63ce5e3a2e8` | `docs/design/write-path-adr-evidence.md` |
| `fc1f5af9ebde4126f89e8ea26fe6d10574dba01343c3e9f441c4d847c4aeb7a4` | `docs/design/write-path-adr-review-packet-v5.md` |
| `348125bb3aab360b2577f8a553e8afac983bab0cff7244abe03f5a31020a9a1a` | `docs/design/write-path-adr-final-independent-review-v5.md` |
| `e584d88fac07580efab6e2b7fab915f611d65ee827018ad2dd44d1a85f842943` | `docs/design/write-path-adr-review-packet-v6.md` |
| `45ebf0ba6109f8998bdc2eec9fc8ae36cdcc0ef38728e3b816e648f1a06c919a` | `docs/design/write-path-adr-final-independent-review-v6.md` |

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

1. verify all 22 hashes and the exact delta from `aa7ea543` plus review provenance;
2. independently reconstruct the seed rows, SplitMix64/permutation/selection rules, ID domains, exact SQL/order,
   route counts/resources, and prove no BENCH-001 workload choice remains;
3. verify the 200/650 arithmetic, sustained wall-clock completion definition, all-population drain rule, peak cohort
   denominator, fixed `B01`–`B10` sequence, no omission/replacement, and derived operations/s;
4. verify the 4M pending seed cannot exhaust under 30+600 seconds at 110,000 TPS plus all ten peak cohorts;
5. rerun and inspect admission/class-escalation/resource-bound, derived wave-budget, monotonicity, and all nine strict
   percentile-boundary scenarios, including the direct end-to-end authority over component arithmetic;
6. search every active source/document consumer for stale 0.5/1/5-ms write, standalone 100k/400k system-SLO, pooled,
   best-window, deferred-manifest, or ambiguous peak wording;
7. confirm Candidate-A evidence is not relabeled and no ACID, durability, acknowledgement, publication, recovery,
   STRATA, RPO/RTO, or physical-selection semantic changed; and
8. return exactly one verdict—**ACCEPT**, **REVISE**, or **REJECT**—with blockers cited by exact file and section.

An **ACCEPT** closes only the post-v4 target/workload consistency gap. The write-path ADR still requires the user's
explicit acceptance decision.
