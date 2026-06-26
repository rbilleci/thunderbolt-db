# P8 Local GPU Production-Readiness Probe

- timestamp_utc: 2026-05-24T04:26:00Z
- git_sha: a2c5d49c4eeceeab9c5fa70adaed1cda7acebef4
- stream: benchmark
- milestone: P8 GPU runtime safety and performance-predictability closeout probe
- device_info: NVIDIA GeForce RTX 3090, driver 595.58.03, 24576 MiB
- validation_gate: `scripts/run_local_gpu_residency_preflight.sh`
- result: pass

## Context

This report closes the bounded local P8 evidence probe for the current
resident-cache, default resident-route, warmup, and maintenance envelope. It
reconciles the local NVIDIA evidence with the source-truth boundary in
`README.md`, `docs/roadmap/v0-v1.md`,
`docs/architecture/10-p8-gpu-optimized-storage-engine.md`, and
`docs/compatibility/matrix.md`.

The supported proof remains a local operator/runtime envelope, not a broad
production CUDA cache claim. WAL/checkpoint/archive replay and CPU-visible
state remain the durable source of truth; GPU-resident state is acceleration
state that can be admitted, invalidated, refreshed, evicted, or rebuilt.

## Observed Outcome

- local_gpu_residency_preflight: passed
- local_gpu_residency_preflight_scope: residency_baseline_warmup_maintenance
- retained CUDA allocation: supported
- resident device memory retained: true
- resident_device_memory_allocated_bytes: 98492
- resident_device_memory_copied_bytes: 98492
- resident_device_memory_gpu_id: 0
- first accepted-route CUDA event timing: supported
- resident_device_memory_cuda_event_timing_samples: 1
- resident_device_memory_cuda_event_elapsed_total_us: 9
- correctness oracle: CPU relational engine

The gate proves a retained CUDA allocation for encoded resident snapshot bytes
on the local RTX 3090. It also proves accepted resident routes with zero
per-query H2D transfer for bounded supported kernel shapes, cache
admission/eviction/invalidation/refresh evidence, operator-triggered warmup,
and a scheduler-friendly maintenance tick.

## Supported Local Envelope

- Retained CUDA allocation and copy handle for encoded resident snapshot bytes.
- Zero-H2D resident routes for bounded supported shapes.
- First accepted resident route reports CUDA event timing samples separately
  from broader metrics-derived route execution deltas.
- Resident cache manager evidence covers deterministic budget admission,
  deterministic eviction, oversized-budget rejection, mutation invalidation,
  manual refresh-cost accounting, and memory-pressure fallback metadata.
- Operator-triggered warmup supports dry-run/apply parity, named/all-table
  selection, invalidated-entry refresh, budget application, memory-pressure
  skips, and route-readiness reporting.
- Scheduler-friendly maintenance tick summarizes warmed, refreshed,
  already-resident, skipped, error, route-ready, and route-blocked outcomes
  without claiming autonomous scheduling.

## Supported Retained Query Families

- Count filters: unfiltered, int4 equality, int4 `IN` membership, int4 range,
  int4 `BETWEEN`, retained int4 `AND`/`OR` filter groups, and retained text
  prefix `LIKE`.
- Aggregates: int4 scalar `SUM`/`AVG`/`MIN`/`MAX`, int4 filtered scalar
  aggregates, int4 `BETWEEN` scalar aggregates, and int4 grouped and
  filtered-grouped `COUNT`/`SUM`/`AVG`/`MIN`/`MAX` with grouped `HAVING`.
- Projections: int4 predicate projection, int4 paginated distinct projection,
  int4 paginated filtered distinct projection, and bounded int4 paginated
  filtered ordered projection.

## Remaining Non-Claims

- durable GPU pages: missing
- autonomous cache-daemon scheduling: missing
- external orchestration: missing
- broad retained expressions: missing
- broad CUDA event timing: missing

The current evidence does not claim broad workload-level GPU advantage,
durable GPU cache behavior, production allocator behavior beyond deterministic
budget admission evidence, autonomous background cache scheduling, live
systemd/Kubernetes orchestration, richer expressions beyond the current
literal comparison/range/membership/prefix/count/aggregate/projection subset,
or CUDA event timing coverage beyond first accepted-route resident-kernel
samples.

## Missing Slice Finding

No new narrowly scoped implementation or observability slice was found inside
the bounded local envelope. The remaining gaps are the same production-facing
boundaries already named by source truth: durable GPU pages, autonomous
cache-daemon scheduling, external orchestration, broader retained expressions,
and broad CUDA event timing. Those should remain blocked until a new product
decision, target environment, or named workload makes one of them a bounded
worker contract.

## Reproduction

```bash
scripts/run_local_gpu_residency_preflight.sh
```

The command ran successfully on 2026-05-24 against commit
`a2c5d49c4eeceeab9c5fa70adaed1cda7acebef4`.
