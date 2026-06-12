# GPU-Native Principles

This project is betting on GPU-native database execution: GPU technology,
memory capacity, interconnects, and programming models are expected to improve,
so the architecture should make the GPU the primary hot execution target rather
than a side accelerator.

## What GPU-Native Means Here

- **GPU memory is the hot data tier.** Resident columns, lookup structures,
  dictionaries, indexes, and read snapshots should be designed for device
  execution first.
- **CPU is the control plane and reference path.** CPU execution remains
  necessary for correctness, fallback, compatibility, and bootstrap, but it
  should not quietly become the optimized hot path.
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
state. Mutable catalog, residency, and generation publication remain controlled
and serialized until the transaction model says otherwise.

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

## Architecture Choices This Rules Out As Defaults

- CPU indexes as the main answer for hot retained point reads.
- CPU response caches as the primary product latency path.
- GPU only as a batch analytics accelerator while OLTP reads stay CPU-first.
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
- route-family scheduling that is simple to explain
- multiple in-flight read-only GPU jobs over a stream pool
- serialized mutation and snapshot publication

Avoid spending too much effort on increasingly complex single-queue heuristics
unless the measurement clearly shows they are a stepping stone toward this
model.
