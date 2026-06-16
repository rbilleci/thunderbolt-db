# GPU-Native Principles

This project is **GPU-native**: the GPU is the engine's execution substrate for
the entire relational data path, not a side accelerator on a CPU engine. GPU
technology, memory capacity, interconnects, and programming models are expected
to keep improving, and the architecture commits to that bet — every relational
capability is designed for the GPU first.

> **Charter (the one rule future work must not violate).** The GPU executes the
> whole relational data path — scans, filters, projections, aggregates, **joins**,
> sorts, grouping — over GPU-resident columnar snapshots, **including the system
> catalog**. The CPU is the host/control plane ONLY. There is no "hybrid
> CPU–GPU" co-execution design and no permanent CPU fallback for hot relational
> work. CPU relational execution exists solely as (a) reference semantics for
> CPU↔GPU parity tests and (b) a temporary bootstrap scaffold for GPU-absent
> dev/CI — both are tracked GPU-parity **debt** with a milestone, never product
> direction and never the optimized hot path. If a design choice routes hot
> relational work to the CPU as its answer, it is wrong by definition here.

## What GPU-Native Means Here

- **GPU memory is the hot data tier.** Resident columns, lookup structures,
  dictionaries, indexes, and read snapshots are designed for device execution
  first. Host memory and disk are staging/spill tiers, not an execution tier.
- **CPU is the host/control plane only.** The CPU owns wire protocol, SQL
  parse/plan, transaction coordination, WAL/durability I/O, and GPU
  orchestration — never the relational execution path. Any CPU relational
  execution is parity-reference or temporary bootstrap scaffold (see the Charter
  above), tracked as debt with a GPU milestone — never the optimized hot path.
- **The catalog is GPU-native.** `pg_catalog` and `information_schema` are
  GPU-resident system relations, executed by the SAME GPU operators as user
  tables — not a CPU-side metadata carve-out. Catalog introspection (including
  the multi-relation joins `psql \d`/ORMs issue) runs on the GPU join path.
- **Joins are GPU operators.** Relational joins are first-class GPU execution
  (partitioned/hash join over GPU-resident relations), never a CPU nested-loop.
- **Hot reads should become prepared routes.** SQL text can exist at the
  protocol boundary, but latency-sensitive retained reads should compile into
  route ids, typed parameters, device-ready projection plans, and snapshot
  handles.
- **Concurrent reads should run over immutable snapshots.** Read concurrency
  should come from GPU-resident snapshot generations that workers can execute
  against safely. Writers publish new generations through a serialized commit
  path until a stronger MVCC design is explicit.
- **Batching is a throughput tool, not the only latency tool.** GPU batching is
  valuable, but consistent low p50 requires avoiding unnecessary owner-queue
  waits and eventually allowing multiple in-flight read-only GPU jobs.

## Target Workload: GPU-Native OLTP

The target workload is OLTP for domains such as banking and e-commerce, not only
primary-key microbenchmarks and not only analytical scans. The engine should
prioritize hot, bounded, high-concurrency routes such as:

- **Entity fetches:** single-row or single-object reads that may be wide.
- **Tenant/security-filtered reads:** every access path may include tenant,
  account, ACL, or visibility predicates.
- **Page reads:** filtered/indexed result pages of roughly 20-50 rows.
- **Bounded joins:** one or two table joins where at least one side is keyed,
  tenant-bounded, or otherwise limited.
- **Computed detail routes:** entity/detail views with derived values,
  summaries, balances, or correlated lookup-like fields.

The core optimization unit should become a prepared OLTP route, not an arbitrary
SQL string. A prepared route records:

- route id and SQL/protocol source
- snapshot generation
- tenant/security predicates
- typed key/range/filter parameters
- join shape and bounded fanout assumptions
- projection and computed-column plan
- expected cardinality: one row, bounded page, or bounded detail fanout
- latency/throughput admission policy

Point lookup support is the first proof, but it is not the final target.

## Preferred Long-Term Shape

```text
SQL/protocol ingress
  -> route compiler / prepared route cache
  -> scheduler over GPU-resident snapshot generations
  -> read-only GPU stream pool
  -> async completion and response materialization

serialized COPY/write/DDL path
  -> build or mutate next generation
  -> publish snapshot generation
  -> retire old generations after readers drain
```

This is a concurrent execution state machine, not unconstrained shared mutable
state. The catalog itself is a GPU-resident set of system relations (queried by
the same GPU operators as user tables); its mutation, residency, and generation
publication remain controlled and serialized until the transaction model says
otherwise.

## Design Rules

1. **Prefer resident device structures over CPU hot-path shortcuts.**
   A CPU cache can be useful as a benchmark probe or compatibility fallback, but
   it should not become the primary answer for GPU-native retained reads.

2. **Keep fallback visible.**
   CPU fallback must report why the GPU path did not run, and the fallback
   should link to a parity or roadmap item when it is not intended as permanent
   product behavior.

3. **Separate latency and throughput goals.**
   Latency-oriented prepared point reads and throughput-oriented batch routes
   may need different admission rules. Avoid burying both goals in opaque queue
   heuristics.

4. **Use snapshots and epochs for safe concurrency.**
   Read-only GPU jobs should reference immutable generations. Writers should not
   mutate published generations in place.

5. **Make GPU costs first-class telemetry.**
   Track queue wait, admission wait, H2D/D2H bytes, kernel time, CUDA event
   time, materialization time, fallback rate, batch size, route shape, and
   snapshot generation where relevant.

6. **Do not overfit to simple pgwire SQL text benchmarks.**
   Pgwire compatibility matters, but the GPU-native hot path should also expose
   prepared/typed route measurements that show the engine floor without repeated
   SQL parse and text protocol overhead.

7. **Prefer 5x boundary collapses over 1.2x tuning.**
   Small scheduler or materialization tweaks are useful only when they are
   simple, validate a larger design, or protect an existing win. Default roadmap
   work should target whole-boundary reductions such as bypassing owner-queue
   waits, overlapping read-only GPU work, or replacing repeated SQL/protocol
   work with prepared route execution. If a slice cannot plausibly change a
   measured row by multiple times, keep it as a short probe or skip it.

## Architecture Choices This Rules Out As Defaults

- CPU indexes as the main answer for hot retained point reads.
- CPU response caches as the primary product latency path.
- GPU only as a batch analytics accelerator while OLTP reads stay CPU-first.
- CPU execution of catalog / `pg_catalog` / `information_schema` introspection as
  the answer — the catalog is GPU-resident and its joins run on the GPU join path.
- CPU nested-loop or CPU hash joins as the join implementation — joins are GPU
  operators.
- Ad hoc locks around mutable CUDA/device state to create accidental
  multi-threaded execution.
- Scheduler policies that improve one benchmark by hiding GPU fallback or
  moving hot work back to CPU.

These techniques may still be valid as fallback, bootstrap, or comparative
baselines, but they should be labeled as such.

## Near-Term Implementation Bias

For the current P8 retained-read work, prefer steps that move toward:

- prepared retained route ids and typed parameters
- immutable retained snapshot handles
- bounded OLTP route classes: entity, page, tenant-filtered, bounded join, and
  computed detail
- route-family scheduling that is simple to explain
- multiple in-flight read-only GPU jobs over a stream pool
- serialized mutation and snapshot publication

Avoid spending too much effort on increasingly complex single-queue heuristics
unless the measurement clearly shows they are a stepping stone toward this
model. Favor changes that can plausibly deliver a 5x class improvement on a
measured bottleneck over changes that merely smooth one benchmark row.
