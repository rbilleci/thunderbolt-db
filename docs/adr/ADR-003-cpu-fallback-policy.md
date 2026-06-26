# ADR-003: CPU fallback policy

- **Status:** SUPERSEDED by [ADR-006](ADR-006-gpu-required-no-cpu-steady-state.md) (2026-06-26)

> The engine now **requires a GPU** (charter — doc 22 §1 / `docs/PLAN.md` §1). CPU relational execution is
> **interim parity-oracle / WIP debt to be deleted** (PLAN §3 S-F / doc 22 S10d), **not** a mandatory steady-state
> safety path. The decision below is reversed — see ADR-006. Retained as the historical record of the reversal.

## Context
GPU availability and eligibility vary; correctness must be preserved.

## Decision
CPU fallback is mandatory safety path; every fallback path requires parity tracking.

## Consequences
- High resilience
- Requires disciplined parity work to avoid permanent CPU drift
