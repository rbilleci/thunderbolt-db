# 09 — Session Management and Admission Control (Bootstrap)

> **Disambiguation:** "admission" here = **session / connection + mutation-queue** admission (request
> backpressure) — a *different control loop* from STRATA's **GPU-residency on-commit admission** (doc 23 §5 /
> PLAN §3). Several "Runtime Admission State" snapshot fields below are forward (v1), not yet built.

## Purpose

Define deterministic session and admission-control behavior so connection pressure cannot violate core safety invariants:

- WAL-before-visibility
- role-aware write gating
- predictable degradation under load

## Session Model (v0/v0.5)

- One client session maps to one logical command stream.
- Session state tracks:
  - transaction context (active txn id/state)
  - role constraints (leader-only mutation eligibility)
  - last-activity timestamp
- Sessions are isolated from each other except through replicated state.

Logical sessions do not imply one permanent OS thread per client. The benchmark
endpoint may use a thread-per-connection harness for bounded measurement, but
the production runtime should multiplex many sessions across a small pool of
network IO workers. IO workers own socket/protocol progress; execution owners
own mutable engine, WAL/MVCC, residency, or GPU state. The high-throughput
runtime topology is specified in
`docs/architecture/11-high-throughput-query-runtime.md`.

## Admission-Control Principles

1. **Safety over throughput**: reject new work before violating durability or ordering guarantees.
2. **Bounded resource usage**: enforce caps on active sessions and queued batched mutations.
3. **Deterministic rejection**: return explicit, classed errors for overload or role mismatch.

## Bootstrap Limits (recommended defaults)

- `max_active_sessions`: 1,000
- `max_idle_session_ttl`: 15 minutes
- `max_pending_batch_items_per_node`: 10,000 (hard cap)
- `max_inflight_read_commands_per_session`: 64

These values are intentionally conservative until real GPU runtime and multi-node soak data are available.

## Runtime Admission State Contract

Each node should expose a minimal admission snapshot that operators can reason about without reconstructing hidden state:

- `active_sessions`
- `idle_sessions`
- `network_io_worker_count`
- `network_io_queue_depth`
- `pending_batch_len`
- `pending_batch_cap`
- `pending_batch_remaining_capacity`
- `pending_batch_remaining_capacity_permyriad`
- `mutation_admission_saturated`
- `read_snapshot_queue_depth`
- `gpu_execution_queue_depth`
- `residency_queue_depth`
- `active_txn_count`
- `role`

This aligns session pressure and mutation-queue pressure with existing replication watermarks so failover-readiness checks and admission checks are not divergent control loops.

## Admission Actions by Signal

- `active_sessions >= max_active_sessions`:
  - reject new session establishment (do not drop existing active transaction sessions).
- `network_io_queue_depth` above its configured cap:
  - stop accepting new sockets or apply listener-level backpressure before
    starving existing sessions.
- `pending_batch_len == pending_batch_cap` OR `mutation_admission_saturated`:
  - reject new mutation enqueue with overload error.
- `read_snapshot_queue_depth` above its configured cap:
  - reject or delay read work with an explicit read-admission overload reason
    rather than routing it through the mutation owner as an accidental fallback.
- `gpu_execution_queue_depth` above its configured cap:
  - either fall back to CPU when semantics allow, or reject with a named GPU
    execution overload reason.
- `role != Leader` for mutation command:
  - reject with `NotLeader`; never enqueue for deferred replay on followers/candidates.
- `active_txn_count` non-zero during failover-prep:
  - hold promotion readiness until transactions drain/resolve.

## Rejection/Backpressure Semantics

- If node is not leader for mutation requests: reject with `NotLeader`.
- If pending mutation queue reaches cap while retry backlog is pending: reject new mutation enqueue with explicit `MutationQueueOverloaded { pending, cap }` error.
- If session budget exhausted: reject new session with admission error; do not evict active transaction sessions abruptly.
- If read-snapshot or GPU-execution queues are saturated: prefer explicit read
  or GPU overload errors over silently entering the single-writer mutation path.
- If residency refresh or invalidation work is saturated: keep correctness on
  CPU/fallback paths and expose the residency blocker instead of serving stale
  resident state.

## Operational Checks

Before enabling traffic after deploy:

1. Confirm role-gating tests pass.
2. Confirm queue-depth metrics and peaks are observable.
3. Confirm fallback reason telemetry is healthy and not masking overload.

During incident response:

- Prefer temporary admission tightening over disabling durability checks.
- Never bypass WAL flush requirements to recover throughput.

## Forward Path (v1)

- Replace static caps with adaptive admission (CPU/GPU memory and queue pressure aware).
- Replace thread-per-client serving with bounded network IO worker pools.
- Add bounded command rings for mutation, read snapshot, residency, and GPU
  execution queues.
- Add per-tenant/session fairness and rate shaping.
- Add explicit session lease/heartbeat semantics for multi-node failover boundaries.
