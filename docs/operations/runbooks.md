# Operations Runbooks (Bootstrap / Pre-NVIDIA)

These runbooks define deterministic operator procedures that preserve the project invariants:

- **WAL-before-visibility**: never make data visible before WAL durability boundary is crossed.
- **GPU-first architecture**: mutation paths remain GPU-targeted in planning, with explicit CPU fallback reasons when needed.

## 1) Pre-deploy Readiness Gate

Run from repository root:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

Pass criteria:

- All three commands succeed.
- No skipped invariant tests (`wal_before_visibility_holds`, replication watermark regressions, snapshot regressions).

If failed:

1. Stop deploy.
2. Capture exact failing command + output.
3. Open/append incident note with root-cause hypothesis.
4. Re-run full gate only after fix is committed.

## 2) WAL Durability Incident (Flush Failure)

Symptoms:

- Error contains `EngineError::Durability(..)`
- Commit rejected and visibility does not advance.

Immediate actions:

1. Freeze mutation traffic to the affected node.
2. Confirm no visibility advance beyond durable boundary.
3. Preserve logs/artifacts for forensic analysis.

Verification checklist:

- `visible_index <= commit_index`
- `wal_unflushed_count == 0` after rollback/recovery handling
- Failed payload is not partially visible in KV state.

Recovery:

1. Repair/restore underlying WAL sink.
2. Re-run deterministic test gate.
3. Replay pending writes from durable source only.
4. Re-enable mutation traffic.

## 3) Role Transition Guardrail (Leader/Follower/Candidate)

Objective:

- Prevent follower/candidate from accepting writes.

Procedure:

1. Confirm current role via replication watermarks.
2. In follower/candidate mode, assert mutations are rejected with `NotLeader`.
3. Ensure pending batch queue is not dropped on rejected flush/tick paths.
4. Only resume flush/commit after valid leader promotion.

Rollback criteria:

- If any rejected write advanced visibility or commit counters unexpectedly, halt rollout and investigate before proceeding.

## 4) Snapshot Install Safety

Before install:

- Record current `commit_index`, `applied_index`, `visible_index`, `snapshot_id`.

During install:

- Apply snapshot metadata atomically.
- Do not rewind indices when receiving older snapshot metadata.

After install:

- `commit_index`, `applied_index`, and `visible_index` are monotonic.
- Next commit index continues from installed/applied frontier.

## 5) CPU Fallback Monitoring (No-GPU bootstrap)

Expected behavior in bootstrap phase:

- Read/control/admin commands may register `NotGpuEligible` fallbacks.
- Mutation planning remains GPU-targeted even when runtime execution is simulated.

Operator checks:

- Track fallback reason counters for unusual spikes.
- Use `Engine::status_snapshot()` as the truth surface:
  - `status.snapshot.snapshot_id` + `status.served_snapshot_frontier()` answer what snapshot/frontier served a read.
  - `status.latest_fallback_reason()` plus `status.why_routed_to_fallback_labels()` answer why work routed to CPU fallback.
  - `status.replication_lag` / `status.replication_distance()` answer how far replication is behind.
- Distinguish expected `NotGpuEligible` from infrastructure-driven reasons.
- Treat unexplained fallback pattern changes as release blockers.

## 6) Release Evidence Bundle

For each release candidate, attach:

1. Git commit SHA.
2. Output logs from fmt/clippy/test gates.
3. Any incident notes since previous candidate.
4. Short statement confirming WAL-before-visibility and role-gating invariants were revalidated.

This keeps DR/security posture auditable and repeatable while implementation iterates toward full GPU and multi-node production readiness.
