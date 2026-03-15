# Execution Model (CPU + GPU)

GPU is a first-class execution target; CPU is the reference semantics implementation.

## Operator contract

Each physical operator must declare:
- CPU implementation
- GPU implementation or explicit fallback rule
- Semantics parity notes

## Batching model

- Dual-trigger batch close (count/time)
- Deterministic ordering metadata attached to batch
- Per-transaction status mapping for partial failures

## Fallback rules

- Fallback preserves transaction semantics and visibility/durability boundaries.
- Every fallback path requires parity tracking issue with owner + milestone.

## Deterministic replay constraints

- Replication replays ordered intent, not device-local side effects.
- Followers apply in log order with convergent logical outcomes.
