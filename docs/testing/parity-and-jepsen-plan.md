# Parity and Jepsen Validation Plan

This plan defines how we prove correctness while moving from local deterministic execution to replicated, GPU-native operation.

## Objectives

1. Preserve **WAL-before-visibility** and role gating invariants under all tested failures.
2. Prove deterministic replay for GPU-eligible workloads against CPU reference behavior.
3. Add consistency/failure-injection validation before declaring v1 readiness.

## Test Streams

## 1) Deterministic Parity Stream (CPU vs GPU)

Scope:
- GPU-eligible mutation subset (starting with key/value `SET`/`DEL` equivalent plan nodes).
- Mixed workloads including explicit non-GPU-eligible commands for fallback coverage.

Method:
- Generate deterministic operation traces with fixed seeds.
- Execute traces in:
  - CPU reference mode
  - GPU-target mode (with fallback allowed)
- Capture canonical outputs:
  - final state hash
  - committed log index range
  - fallback reason counters
- Assert parity where required:
  - identical committed mutation order
  - identical final visible state
  - fallback telemetry explains any routed-to-CPU segments

Exit criteria (v1):
- Zero parity mismatches across agreed seed corpus.
- Any mismatch yields reproducible trace artifact + minimization case.

## 2) Durability + Recovery Stream

Scope:
- WAL flush failures, process interruption windows, replay from durable boundary.

Method:
- Inject one-shot and burst flush failures.
- Assert rejected commits never become visible.
- Restart/recover from persisted WAL snapshots (as available per phase).
- Validate monotonicity:
  - commit index >= applied index >= visible index (with expected equalities in steady state)

Exit criteria (v0.5+):
- No visibility leak for failed durability events.
- Recovery replay yields deterministic state from same WAL prefix.

## 3) Replication Consistency Stream

Scope:
- Leader/follower role transitions, log catch-up, snapshot install, failover basics.

Method:
- Multi-node simulation first; 3-node deployment in v1.
- Exercise:
  - follower write rejection
  - leader promotion/demotion
  - lagging follower catch-up
  - snapshot install and watermark advancement
- Validate safety/liveness signals:
  - no committed entry lost
  - no divergent visible state after convergence

Exit criteria (v1):
- Stable failover/recovery runs over repeated randomized schedules.

## 4) Jepsen-Style Fault Stream (v1 gate)

Fault classes:
- process kill/restart
- network partition (leader isolated, minority/majority splits)
- delayed/dropped/reordered replication messages
- disk flush failure injection at durability boundary

Workload model:
- deterministic write/read mix with monotonic counters and set/delete keys.
- linearizability-friendly operation recording.

Checks:
- safety first: no stale or phantom visibility beyond durable committed boundary.
- convergence after healing: all nodes reach same visible state.
- optional linearizability checks for declared operation subset.

Exit criteria (v1 readiness):
- No invariant violations in agreed run budget.
- All failures produce actionable repro artifacts.

## Artifacts and Reporting

For each failing run, capture:
- seed / workload profile
- node topology and fault schedule
- command/error excerpts
- WAL and replication watermark snapshots
- minimized reproducer status

Store reports under `docs/testing/reports/` (phase-dependent as infra appears).

## Phase Mapping

- v0: deterministic parity harness scaffold + failure-injection unit tests
- v0.5: replication simulation parity + durability/recovery expansion
- v1: 3-node faulted runs + Jepsen-style campaign baseline

## Non-Goals

- Proving full SQL-level linearizability before protocol/SQL surface maturity
- Broad extension/plugin correctness guarantees pre-v1

## Ownership

- Engine + execution owners: deterministic parity stream
- Replication owners: consistency + fault stream
- Release owner: v1 gate sign-off based on this plan
