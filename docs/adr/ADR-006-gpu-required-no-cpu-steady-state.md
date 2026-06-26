# ADR-006: GPU is required — no CPU steady-state fallback (supersedes ADR-003)

- **Status:** Accepted (2026-06-26)
- **Supersedes:** [ADR-003](ADR-003-cpu-fallback-policy.md) (CPU fallback policy)

## Context
ADR-003 (Accepted, 2026-06-12) declared CPU fallback a *mandatory, permanent* safety path, on the premise that
"GPU availability and eligibility vary." The mandate has since changed: the engine **requires a GPU** (charter —
doc 22 §1 / `docs/PLAN.md` §1; sm_120 floor), the host is the **control plane only**, and CPU relational
execution is **interim WIP being deleted**, not a steady-state tier. (Verified 2026-06-26: the host read path is
live today only because GPU residency is operator-triggered; the STRATA design — doc 23 — adds automatic
residency admission on commit so it can be retired.)

## Decision
- The engine requires a GPU. There is **no CPU-only / GPU-absent / hybrid steady-state mode.**
- Any CPU relational execution is **interim GPU-parity debt**, tracked and scheduled for deletion
  (PLAN §3 S-F / doc 22 S10d). It is never a permanent product path.
- Parity is verified against a **GPU-native oracle** (on-device serial reference or closed-form), never a CPU
  re-implementation used as the source of truth.

## Consequences
- The CPU relational read/execute path (`finalize_relational_select`, the MVCC `cpu_fallback`, the
  `FirstCudaSliceParityBackend` oracle) is retired once STRATA admission makes the GPU path the production default.
- Durability/replication remain host control-plane responsibilities (unchanged; see ADR-001 / ADR-004).

## Alternatives considered
- Keep ADR-003 (permanent CPU fallback): rejected — contradicts the GPU-required charter and perpetuates the
  host data-plane the campaign is deleting.

## Links
- Charter: `docs/architecture/22-full-gpu-native-read-path.md` §1-3; `docs/PLAN.md` §1.
- Admission producer that unblocks deletion: `docs/architecture/23-strata-resident-shard-data-plane.md`.
