# Execution Model (CPU + GPU)

GPU is a first-class execution target; CPU is the reference semantics implementation.

This document defines the execution contract. The runtime topology that serves
many client sessions, owns mutable state, routes work through bounded queues,
and publishes read snapshots is specified in
`docs/architecture/11-high-throughput-query-runtime.md`.

## Operator contract

Each physical operator must declare:
- CPU implementation
- GPU implementation or explicit fallback rule
- Semantics parity notes

## Batching model

- Dual-trigger batch close (count/time)
- Deterministic ordering metadata attached to batch
- Per-transaction status mapping for partial failures

Batching is allowed only when it preserves the SQL-visible ordering and
visibility contract. The batcher may group compatible work for amortized GPU
execution, but it must still report per-command success, fallback, or rejection.

Supported batch forms:

- **mutation batches**: ordered WAL/MVCC admission for commands that must remain
  serialized by the mutation owner
- **read micro-batches**: compatible retained read requests grouped by query
  shape, table, partition, and snapshot generation
- **GPU kernel batches**: multiple predicates, keys, or aggregate requests
  executed by one launch or one coordinated set of launches
- **response batches**: prebuilt row descriptions, command-complete messages,
  and reusable protocol buffers where the response shape is stable

The execution layer must expose whether a result came from CPU execution, cold
GPU transfer, retained resident execution, retained read snapshot execution, or
an encoded response fast path.

## Read snapshot execution

Production retained reads should execute from immutable, versioned snapshots
rather than directly touching mutable engine state. A read snapshot records the
catalog/schema generation, source WAL or transaction boundary, resident layout
identity, visibility boundary, and invalidation generation.

Readers may hold a snapshot by reference while executing. Writers and DDL do not
mutate a published snapshot in place; they publish a newer generation and retire
old snapshots only after readers release them. This gives read workers a safe
path to execute concurrently while the mutation owner preserves WAL-before-
visibility ordering.

The benchmark-only retained response cache is not a substitute for this model.
It proves that avoiding the owner queue for hot read responses improves scale,
but it caches encoded bytes rather than executing from a queryable snapshot.

## GPU execution model

GPU execution workers own the CUDA streams, events, pinned host buffers, and
resident device handles they use. They may execute:

- one retained read at a time when latency is the priority
- a drained micro-batch of same-shape retained lookups
- partition-local aggregate batches that later reduce on CPU or a coordinator
- COPY or refresh staging work where the input already has deterministic order

Kernel launches should be amortized across compatible work when queue depth
exists, but the scheduler must preserve explicit latency ceilings so low
concurrency requests do not wait indefinitely for a larger batch.

## Fallback rules

- Fallback preserves transaction semantics and visibility/durability boundaries.
- Every fallback path requires parity tracking issue with owner + milestone.

## Deterministic replay constraints

- Replication replays ordered intent, not device-local side effects.
- Followers apply in log order with convergent logical outcomes.
