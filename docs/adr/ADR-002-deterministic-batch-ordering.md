# ADR-002: Deterministic batch ordering model

- **Status:** Accepted

## Context
GPU batching and replication need convergent outcomes across nodes.

## Decision
Replicate ordered transactional intent; apply in deterministic log order.

## Consequences
- Simplifies follower convergence
- Requires strict ordering metadata and replay discipline

## Alternatives considered
- Replicate post-execution effects only (rejected: divergence/debug complexity)
