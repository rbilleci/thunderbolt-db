# 09 — Session Management and Admission Control (Bootstrap)

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

## Rejection/Backpressure Semantics

- If node is not leader for mutation requests: reject with `NotLeader`.
- If pending mutation queue reaches cap while retry backlog is pending: reject new mutation enqueue with explicit `MutationQueueOverloaded { pending, cap }` error.
- If session budget exhausted: reject new session with admission error; do not evict active transaction sessions abruptly.

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
- Add per-tenant/session fairness and rate shaping.
- Add explicit session lease/heartbeat semantics for multi-node failover boundaries.
