# Project Agent Guidance

This repository is pursuing a GPU-native database engine. Future agents should
optimize for that thesis unless the user explicitly changes direction.

## GPU-Native North Star

- Treat GPU-resident execution as the product direction, not as an optional
  accelerator around a CPU-first database.
- Optimize for GPU-native OLTP: entity fetches, tenant/security-filtered page
  reads, bounded joins, and computed detail routes, not only analytical scans or
  primary-key microbenchmarks.
- Prefer designs where hot data, lookup structures, encoded columns, and read
  snapshots live in GPU memory.
- The CPU is the host/control plane ONLY: wire protocol, SQL parse/plan,
  transaction coordination, WAL/durability I/O, and GPU orchestration. CPU
  relational execution exists solely as parity-reference plus temporary
  bootstrap scaffold, tracked as debt with a GPU milestone — never product
  direction and never the hot-path design.
- The catalog is GPU-native and joins are GPU operators: `pg_catalog` and
  `information_schema` are GPU-resident system relations, and catalog joins
  (`psql \d`, ORM introspection) run on the GPU join path. Do not build a CPU
  catalog or CPU nested-loop/hash join as the target answer.
- For hot reads, prefer prepared route ids, typed parameters, resident snapshot
  handles, and device-ready projection plans over repeated SQL-text parsing.
- Favor immutable/versioned GPU-resident snapshots for read concurrency.
  Serialize mutation and generation publication until a stronger MVCC model is
  intentionally designed.
- Optimize batching for throughput, but do not make batching the only latency
  answer. GPU-native low latency likely requires concurrent read execution over
  resident snapshots.

## Architecture Bias

When choosing between implementation approaches:

1. Keep the GPU hot path explicit and measurable.
2. Preserve CPU/GPU semantic parity, but treat CPU relational execution as
   parity-reference/bootstrap debt with a milestone — never product direction.
3. Avoid adding CPU caches or CPU indexes as the primary answer for benchmark
   wins unless the change is clearly documented as a non-GPU-native escape
   hatch.
4. Prefer principled concurrency boundaries: immutable read snapshots,
   serialized writers, CUDA stream ownership, epoch/generation retirement.
5. Be cautious with heuristic scheduler complexity. If a policy becomes hard to
   explain, consider a simpler split between latency-oriented prepared reads and
   throughput-oriented batch routes.

## Documentation Expectations

Before major runtime, storage, or scheduler changes, read:

- `docs/architecture/00-gpu-native-principles.md`
- `docs/roadmap/gpu-native-oltp-roadmap.md`
- `docs/GPU_GUARDRAILS.md`
- `docs/architecture/04-execution-model-cpu-gpu.md`
- `docs/architecture/11-high-throughput-query-runtime.md`
- `docs/architecture/12-acid-isolation-and-gpu-memory.md`

When a change intentionally favors CPU-first behavior, document why it is a
fallback, bootstrap step, or product-scope exception.
