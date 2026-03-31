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

## Runtime Admission State Contract

Each node should expose a minimal admission snapshot that operators can reason about without reconstructing hidden state:

- `active_sessions`
- `idle_sessions`
- `pending_batch_len`
- `pending_batch_cap`
- `pending_batch_remaining_capacity`
- `mutation_admission_saturated`
- `active_txn_count`
- `role`

This aligns session pressure and mutation-queue pressure with existing replication watermarks so failover-readiness checks and admission checks are not divergent control loops.

## Admission Actions by Signal

- `active_sessions >= max_active_sessions`:
  - reject new session establishment (do not drop existing active transaction sessions).
- `pending_batch_len == pending_batch_cap` OR `mutation_admission_saturated`:
  - reject new mutation enqueue with overload error.
- `role != Leader` for mutation command:
  - reject with `NotLeader`; never enqueue for deferred replay on followers/candidates.
- `active_txn_count` non-zero during failover-prep:
  - hold promotion readiness until transactions drain/resolve.

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
