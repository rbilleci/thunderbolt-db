# 11 — High-Throughput Query Runtime

## Purpose

Define the production serving architecture for high-throughput SQL over the GPU
engine. This document complements the P8 storage design:

- P8 defines resident data, invalidation, refresh, and planner route validity.
- This runtime document defines how sessions, queues, owners, read snapshots,
  and GPU execution workers move query work at scale.

The benchmark endpoint currently proves bounded ideas with a simpler topology.
It should not be treated as the final serving architecture.

## Current Benchmark Topology

The engine-backed pgwire benchmark endpoint currently uses:

- one TCP accept loop
- one OS thread per accepted client connection
- one `mpsc` command queue into the engine owner
- one owner thread that owns `EndpointState`, `Engine`, WAL/MVCC state,
  catalog state, residency metadata, and retained CUDA state
- an optional shared encoded retained read response cache at the protocol edge

That topology is useful for measurement and correctness isolation. It is not
the intended production scale model. A thread per client will eventually cost
too much memory, scheduler overhead, context switching, and tail latency.

The encoded response cache is also only a demonstrator. It proves that repeated
read-only retained responses scale better when they avoid the owner-thread
queue, but it caches exact pgwire bytes rather than executing from immutable
queryable snapshots.

## Target Runtime Topology

The production target is:

```text
client sockets
  -> network IO workers
  -> bounded command rings
  -> owner domains / read snapshot workers / GPU execution workers
  -> response rings
  -> network IO workers
  -> client sockets
```

The runtime should use a small pool of network IO workers instead of one OS
thread per client. IO workers own socket readiness, protocol parsing, request
framing, and response writes. They do not directly mutate engine state.

Hot execution paths should use bounded queues or ring buffers with explicit
capacity, backpressure, and predictable allocation behavior. Generic channels
remain acceptable for tests and early proofs, but the long-term hot path should
prefer fixed-capacity, cache-friendly queues where ownership and queue depth are
observable.

## Owner Definition

An owner is the one thread, task, or component allowed to directly mutate or use
a specific piece of state.

Ownership does not mean one owner for the whole system forever. It means a
mutable state domain has exactly one authority at a time. Other workers interact
with that domain through messages, immutable snapshots, or published handles.

The rule is:

- mutable state has one owner
- immutable snapshots may be shared by many readers
- multiple owners must own disjoint mutable state, or coordinate through an
  explicit publication protocol

## Owner Domains

The first production split should keep correctness simple:

- **Mutation owner**: WAL admission, transaction visibility, COPY/INSERT/UPDATE
  mutation, and WAL-before-visibility sequencing.
- **Catalog/DDL owner**: schema metadata, relation identity, route-relevant
  generation counters, and DDL invalidation.
- **Residency owner**: GPU residency lifecycle, refresh, eviction,
  invalidation, and resident snapshot publication.
- **GPU execution owners**: CUDA streams, events, pinned host buffers, device
  handles, and partition-local retained execution.
- **Partition owners**: optional future shard owners for disjoint table or
  resident partition state.

These domains may initially be co-located in one thread while the implementation
is small. They should become separately owned only when measurements show the
shared owner is the bottleneck and the split preserves ordering.

## Read Snapshot Publication

Read concurrency should come from immutable versioned snapshots.

The mutation/catalog/residency path publishes a retained read snapshot after a
valid resident refresh or route-ready generation. A snapshot includes:

- relation identity and schema generation
- source WAL or transaction boundary
- visibility boundary
- resident layout identity and device handles
- supported columns, predicates, and route families
- invalidation generation
- partition or shard identity where applicable

Readers hold a snapshot by reference while executing. Writers do not mutate the
snapshot in place. DDL, COPY, mutation, refresh, eviction, and memory pressure
publish a newer generation or mark the old generation invalid for new readers.
Old snapshots retire after the last reader releases them.

This is the production replacement for the benchmark encoded response cache.

## Command And Response Rings

The runtime should separate queues by purpose:

- **network ingress rings**: parsed frontend messages from IO workers
- **mutation command rings**: serialized writes into the mutation owner
- **read snapshot rings**: read-only work eligible for immutable snapshot
  execution
- **residency rings**: refresh, eviction, warmup, and invalidation work
- **GPU execution rings**: resident kernels grouped by device, stream,
  partition, and query shape
- **response rings**: encoded or partially encoded responses back to IO workers

Each ring needs:

- bounded capacity
- explicit saturation metrics
- deterministic rejection or fallback policy
- queue wait telemetry
- batch-drain telemetry
- ownership of reusable buffers where possible

## Micro-Batching

Micro-batching should happen by natural queue drain under latency ceilings.

The scheduler may collect compatible work until either:

- a count threshold is reached
- a microsecond threshold is reached
- an explicit flush event occurs

Compatible retained read work must match:

- snapshot generation
- relation or partition identity
- query shape
- selected column family
- predicate family
- output response shape, when response metadata is reused

Expected micro-batches:

- **lookup batches**: many `WHERE key = ?` predicates executed as a vector of
  keys with result scattering by request id
- **aggregate batches**: compatible `COUNT`, `SUM`, `MIN`, `MAX`, or `AVG`
  requests grouped by resident partition and reduced deterministically
- **COPY admission batches**: WAL/MVCC/index work flushed at deterministic
  chunk or time boundaries
- **refresh batches**: residency rebuild and publication grouped by table or
  partition generation
- **response batches**: stable row descriptions, command-complete messages, and
  reusable output buffers for same-shape responses

Micro-batching must not hide correctness. Each request still receives its own
success, fallback, rejection, or error result.

## GPU Execution Workers

GPU execution workers own the CUDA resources they use:

- CUDA stream
- event objects
- retained device handles
- pinned host staging buffers
- scratch buffers
- partition-local execution metadata

They may execute one request immediately when latency is the priority, or drain
a compatible micro-batch when queue depth exists. Batching is most valuable when
it amortizes kernel launch, transfer setup, and result scattering.

GPU workers must report:

- queued requests
- drained batch size
- queue wait time
- kernel launch count
- CUDA event elapsed time
- H2D/D2H bytes
- result rows
- fallback or rejection reason

## Backpressure And Admission

Backpressure is part of correctness. The runtime must never preserve throughput
by silently violating ordering, visibility, or residency validity.

Admission should reject or delay work at the narrowest saturated boundary:

- network session cap
- network IO queue cap
- mutation queue cap
- read snapshot queue cap
- residency queue cap
- GPU execution queue cap
- memory or pinned-buffer budget

When a GPU queue is saturated, the runtime may choose CPU fallback only if the
query semantics and latency goal allow it. Otherwise it should return an
explicit overload reason.

## Ordering And Invalidation

The mutation owner preserves WAL-before-visibility:

1. validate mutation
2. append and flush WAL
3. invalidate affected resident generations
4. apply CPU-visible MVCC/catalog state
5. publish the new visibility boundary

Read workers may execute only against snapshots compatible with their read
boundary. A read that cannot prove compatibility must fall back to the owner or
CPU path.

DDL and residency refreshes publish new generations. New readers choose the
newest compatible generation. Existing readers may finish on an older immutable
snapshot unless the operation requires a stronger barrier.

## Mechanical Sympathy Guidelines

Hot paths should prefer:

- bounded queues or rings over unbounded allocation
- preallocated command and response buffers
- reusable row descriptions and response metadata
- pinned host buffers for GPU staging
- cache-line-aware counters and queue state
- low allocation in per-request paths
- explicit queue wait, batch size, and saturation telemetry

These are borrowed from high-throughput and HFT-style systems, but they remain
subordinate to database invariants: WAL-before-visibility, MVCC correctness,
replay, DDL safety, and deterministic fallback.

## Benchmark Gates

Each runtime slice should prove one bottleneck moved:

- replace thread-per-client harness with IO worker pool without correctness
  regression
- replace generic command channel with bounded rings and queue telemetry
- publish immutable retained read snapshots and retire them safely
- execute same-shape lookup micro-batches from one snapshot
- execute partition-local aggregate micro-batches and reduce deterministically
- preserve explicit overload/fallback reasons under queue saturation
- compare p50/p95/p99, throughput, queue wait, batch size, kernel launch count,
  CUDA event time, H2D/D2H bytes, and correctness status

Do not claim a production runtime slice from response-cache evidence alone. The
encoded response fast path is a useful proof of the owner-queue boundary; the
production path must execute from retained read snapshots.
