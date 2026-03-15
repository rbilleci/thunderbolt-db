# ADR-005: Snapshot/install-snapshot strategy

- **Status:** Accepted

## Context
Long-running systems need compaction and fast follower catch-up.

## Decision
Snapshot hooks are required from early phases, even before full distributed rollout.

## Consequences
- Lowers future integration risk
- Slight early implementation overhead
