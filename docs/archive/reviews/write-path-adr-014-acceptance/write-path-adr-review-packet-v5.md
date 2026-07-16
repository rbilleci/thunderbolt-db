# Focused R3-001 independent-review packet — target refinement 5

> Archived frozen review manifest. Non-actionable; current work lives only in `docs/PLAN.md`.

- **Frozen:** 2026-07-15
- **Pre-change commit:** `e1861025`
- **Decision state:** proposed, explicitly not accepted
- **Prior review:** packet v4 returned **ACCEPT** under the prior uniform latency target
**Review question:** Does the accepted R1/W1/T8/T32 target refinement remain internally consistent with the proposed
GPU-native write design, controller qualification, evidence interpretation, and production-graduation gates, or did
the refinement introduce a new design-acceptance blocker?

This is a focused post-v4 delta review, not a rewrite of the independent v4 verdict and not an implementation or
product benchmark. The user accepted the target policy, not the proposed write-path ADR. Open review state and
sequencing remain exclusively in `PLAN.md` under R3-001.

## Target contract under review

| Class | Envelope | p50 | p99 | p99.9 |
|---|---|---:|---:|---:|
| R1 | one prepared bounded point/page read | <0.5 ms | <1 ms | <5 ms |
| W1 | one keyed synchronous INSERT, UPDATE, or DELETE | <0.8 ms | <1.5 ms | <5 ms |
| T8 | 2–8 predeclared operations, at most four mutations, within declared resource bounds | <1.5 ms | <3 ms | <10 ms |
| T32 | 9–32 predeclared operations, at most 16 mutations, within declared resource bounds | <3 ms | <6 ms | <20 ms |

Each class and each W1 operation passes independently. A mixed distribution is supplementary and cannot hide a
failing class. Interactive/data-dependent work remains a separately reported slow class without a generic client-
wall-time promise. Targets are strict: equality fails qualification.

## Delta submitted

1. `CHARTER.md`, ADR-008, and `ARCHITECTURE.md` define one matching class taxonomy and measurement boundary.
2. BENCH-001 requires independent R1, INSERT, UPDATE, DELETE, I/U/D-mix, T8, T32, mixed read/write, and interactive
   reporting, including TPS plus logical operations/s and no pooled-distribution acceptance.
3. The proposed controller derives its residual oldest-age budget from the admitted class after measured downstream
   margins. T8/T32 budgets require operation, mutation, byte, index-fanout, touched-table, cold-access, and result
   bounds; work cannot borrow a larger class budget opportunistically.
4. The build-only controller model uses strict class-specific W1/T8/T32 p99 qualification. The measured 1,723-us
   fence plus 200-us margin remains unqualified for W1, qualifies for T8, and the T32 equality boundary fails.
5. The existing Candidate-A rows remain a W1 **FAIL**. Their raw evidence is unchanged, but the report now makes
   explicit that its “mixed” rows are I/U/D-only and that pooled DML percentiles cannot graduate the canonical path.
6. Packet/review v4 are retained as historical evidence at `e1861025`; neither is relabeled as reviewing this delta.

The target refinement does not change the compact append/tombstone physical selection, ACID semantics, WAL/marker
format, publication law, RPO/RTO contract, recovery-capacity profile, STRATA placement, or host-control-plane rule.

## Review inputs and SHA-256

| SHA-256 | File |
|---|---|
| `080bc48ae7f215a630153bf5d43bb9ac5491b033df6e0dc61b4ee6cd689a558e` | `docs/CHARTER.md` |
| `7c9d396a5ec32b4f3c2a9eefd95725fbe6da2fe514ca9c76d74b201e462ccc96` | `docs/DECISIONS.md` |
| `2e2e9474848a1e76b2c3f42ec9ca7112de1223fbd8ebc75ea674c1baef72e551` | `docs/ARCHITECTURE.md` |
| `6e004bcc85fa198ae6785221bfa762645ffeb105a9f4d837c323f7e56ce07347` | `docs/PLAN.md` |
| `76c39ffbed3fec595fd4036c3f360b48bb8a9cb351cd9494803a6b3b17e74de0` | `docs/STATUS.md` |
| `38cc76bd0ab0bdfdc0ca9b9e69307a42ea297438b2c2de52de38f248528043f6` | `docs/HANDOVER.md` |
| `f6b188e7efa8b8cc3a188ab36b0bfec13546b1890f777fcad815fb2cff4a019d` | `docs/design/write-path-adr-proposal.md` |
| `3c3f76edb7a103e90d408e15b0d6433860e8772c33ce9977f3ab3ce95b70a823` | `docs/design/write-path-adr-slo-footprint.md` |
| `0d9b3005cc2b531dc397ff3170c1007b6e3e7a42d9260b2983c60d97b6315b4c` | `docs/design/write-path-adr-performance-audit.md` |
| `a12d8442c08dae44e94d103a6d8aab5e47258c6cd35d2024f42a76a60ff19396` | `docs/design/write-path-adr-controller-injections.md` |
| `f023097113edec6358bcec8319273cf0894217ec6bc52a43972b9129839f2d17` | `crates/engine/examples/write_path_adaptation_injections.rs` |
| `eb0e2eef9af2f47bc9d9dcad30a06a194a37c056e5c9e36fa3416d78fb5ab7af` | `docs/design/write-path-adr-review-matrix.md` |
| `6f7b21db9783075a7540c8f433f2c7f1a28cbeebf39d39e58bf7bca056bc5ecc` | `docs/design/write-path-adr-review-packet-v4.md` |
| `fab827b733bd0ecf8fd56a2e9c5efacecebf4ba18f75be3727168e07fe6e1427` | `docs/design/write-path-adr-final-independent-review-v4.md` |

## Verification submitted

```text
rustfmt --edition 2021 crates/engine/examples/write_path_adaptation_injections.rs
cargo clippy -p gpu_db_engine --example write_path_adaptation_injections -- -D warnings
cargo run --quiet -p gpu_db_engine --example write_path_adaptation_injections
git diff --check
```

Strict Clippy passes and all 12 named controller families plus the aggregate pass. This target-only change does not
touch a production read kernel, residency layout, or result path, so the GPU HAZARD and report-card gates are not
triggered.

## Independent review protocol

The reviewer must not edit the packet. It must:

1. verify every hash and confirm the delta is limited to the target policy, its consumers, and review provenance;
2. verify all authoritative and proposal documents use the same R1/W1/T8/T32 values and strict inequality;
3. verify W1 INSERT/UPDATE/DELETE and mixed read/write acceptance cannot pass through pooled percentiles;
4. verify T8/T32 operation counts cannot bypass declared mutation/resource bounds or let work borrow a larger
   residual budget;
5. rerun the controller model and inspect W1-unqualified, T8-qualified, and strict T32-boundary behavior;
6. confirm the revised Candidate-A evidence remains a W1 failure and no historical measurement was relabeled;
7. confirm the target revision does not alter ACID, durability, publication, recovery, STRATA, or physical-selection
   semantics; and
8. return one verdict—**ACCEPT**, **REVISE**, or **REJECT**—with any blocker cited by exact file and section.

An **ACCEPT** closes only this post-v4 target-consistency gap. The proposed write-path ADR still requires the user's
explicit acceptance decision.
