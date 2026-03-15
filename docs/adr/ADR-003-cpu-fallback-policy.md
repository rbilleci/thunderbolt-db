# ADR-003: CPU fallback policy

- **Status:** Accepted

## Context
GPU availability and eligibility vary; correctness must be preserved.

## Decision
CPU fallback is mandatory safety path; every fallback path requires parity tracking.

## Consequences
- High resilience
- Requires disciplined parity work to avoid permanent CPU drift
