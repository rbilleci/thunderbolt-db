# Design Alignment Check

Date: 2026-03-15
Baseline compared against: `DESIGN.md`
Compared docs:
- `docs/GPU_GUARDRAILS.md`
- `docs/architecture/*`
- `docs/interfaces/*`
- `docs/adr/*`
- `docs/roadmap/v0-v1.md`

## Summary

Overall status: **Aligned (with intentional scope compression)**

- The recent docs preserve core correctness and architecture intent from `DESIGN.md`.
- They intentionally compress implementation scope for early phases to reduce delivery risk.
- No direct contradictions found on core invariants.

## Alignment Matrix

### 1) Postgres compatibility intent
- `DESIGN.md`: PG protocol + SQL/catalag compatibility targets
- New docs: compatibility discipline and explicit supported/partial/unsupported stance
- Status: **Aligned**

### 2) WAL-before-visibility and durability model
- `DESIGN.md`: strict WAL-before-visibility invariant
- New docs: invariant explicitly elevated and enforced in commit path docs
- Status: **Aligned**

### 3) GPU-native execution (GPU is the relational substrate; CPU is host/control plane)
- `DESIGN.md`: GPU-native charter — the GPU executes the entire relational data path including the catalog; CPU is the host/control plane only (no "hybrid CPU-GPU" co-execution principle, no permanent CPU fallback for hot relational work)
- New docs: GPU-native principles (`docs/architecture/00-gpu-native-principles.md`) + GPU-First guardrails; CPU relational execution is parity-reference / bootstrap **debt** with a milestone, not a product pillar
- Status: **Aligned**

### 4) Deterministic batching / replay semantics
- `DESIGN.md`: deterministic ordering central to GPU OLTP model
- New docs: ADR + architecture docs define deterministic apply and replay constraints
- Status: **Aligned**

### 5) Multi-node replication / Raft readiness
- `DESIGN.md`: strong replication and failover requirements
- New docs: “Raft-aware now, Raft-enabled later” with interfaces and role gates
- Status: **Aligned (phased)**

### 6) Observability and SLO orientation
- `DESIGN.md`: extensive observability requirements
- New docs: minimum required metric set and SLO tracking structure
- Status: **Aligned (condensed)**

### 7) Security/compliance depth
- `DESIGN.md`: extensive banking-grade security/compliance controls
- New docs: currently referenced indirectly, not yet deeply expanded in dedicated docs
- Status: **Partially aligned (needs expansion)**

### 8) Backup/PITR/DR operational detail
- `DESIGN.md`: detailed PITR, backup, multi-region DR
- New docs: boundaries included (storage/recovery, snapshots), but not full runbook-level detail
- Status: **Partially aligned (expected at later phase)**

## Gaps to Close Next

1. Add `docs/architecture/07-security-and-compliance.md` mapping controls to implementation milestones.
2. Add `docs/architecture/08-backup-pitr-dr.md` with operational runbooks and test gates.
3. Add `docs/compatibility/matrix.md` for explicit feature support by phase and postgres version.
4. Add `docs/testing/parity-and-jepsen-plan.md` for deterministic replay + consistency validation gates.

## Conclusion

Recent docs are directionally and technically consistent with the original design while making implementation sequencing realistic. The main deltas are in depth and timing (security/compliance and DR detail), not architectural intent.
