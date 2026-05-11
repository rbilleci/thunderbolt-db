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
- Reject incoherent snapshot frontier jumps where `last_included_index` advances but `last_included_term` regresses.

After install:

- `commit_index`, `applied_index`, and `visible_index` are monotonic.
- Next commit index continues from installed/applied frontier.

## 4b) Resume / Restart Validation

After interruption or restart:

1. Restore follower replication state from snapshot metadata plus contiguous committed tail only.
2. Confirm resumed node does **not** recover speculative/uncommitted entries.
3. Confirm `commit_index` is preserved and `applied_index` may legitimately lag until replay/apply catches up.
4. Confirm next append resumes from the durable tail (`next_index = durable_tail + 1`).

For the current single-node relational WAL segment proof:

1. Persist only the flushed prefix with `Engine::persist_durable_wal_to_file(...)`.
2. Recover with `Engine::recover_from_durable_wal_file(...)`.
3. Verify relational catalog metadata, table rows, and equality-index-backed read paths are present after replay.
4. For packaged checkpoint proof, persist with `Engine::persist_durable_wal_checkpoint(...)` and recover with `Engine::recover_from_durable_wal_checkpoint(...)`; the control file validates durable record count and last transaction id before replay.
5. Treat PITR selection, multi-segment archive discovery, and automatic retention cleanup as not yet implemented.

Failure criteria:

- resumed state invents gaps past the snapshot boundary
- resumed state advances `applied_index` beyond durable committed boundary
- resumed state recovers uncommitted tail as if durable
- recovered relational state includes rows not present in the flushed WAL segment

## 4c) Local Operational Replication Smoke

Current implemented scope:

- `scripts/run_replication_cluster_smoke.sh` runs an in-process 3-node Raft smoke scenario.
- The scenario demonstrates leader write admission, follower catch-up, read-after-apply, old-leader `NotLeader` rejection after failover, and continued writes on the promoted leader.
- The same command emits an `operational_deployment_preflight=passed` line only when the smoke proof passes and the current deployment scope/gaps are explicitly reported.

Current simulated/not-yet-implemented scope:

- No network transport between node processes.
- No automatic election or membership reconfiguration.
- No packaged container/Kubernetes deployment harness.

Run from repository root:

```bash
scripts/run_replication_cluster_smoke.sh
```

Pass criteria:

- Output includes `operational_replication_smoke=passed`.
- Output includes `operational_deployment_preflight=passed`.
- Output includes `deployment_scope=in_process_three_node_raft_smoke`.
- Follower output reports `follower_caught_up=true`.
- `follower_read_after_apply` includes the post-failover write.
- `promoted_leader_commit`, `follower_commit`, and `follower_applied` are equal.
- `failover_admission_gate` reports old-leader write rejection and `promoted_node_role=Leader`.
- `deployment_gap_network_transport`, `deployment_gap_automatic_election`, and `deployment_gap_packaged_deployment` are present and currently report `missing`.

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

## 5b) MVCC Retention Boundary

Current bootstrap storage can vacuum old MVCC tuple versions only through `Engine::checkpoint_vacuum_mvcc_versions(safe_txn_id)`. Choose a non-zero safe transaction id that is at or below the flushed WAL boundary and older than every active transaction. The call refuses unsafe boundaries, reports removed tuple/version counts, and keeps the durable WAL prefix as the replay source of truth.

Do not manually prune tuple versions, relational row keys, or equality-index entries. Relational indexes remain volatile and are rebuilt from the durable WAL prefix during recovery; the current control file covers one selected durable segment, while PITR selection and multi-segment archive cleanup are still future storage work.

## 6) Release Evidence Bundle

For each release candidate, attach:

1. Git commit SHA.
2. Output logs from fmt/clippy/test gates.
3. Any incident notes since previous candidate.
4. Short statement confirming WAL-before-visibility and role-gating invariants were revalidated.

This keeps DR/security posture auditable and repeatable while implementation iterates toward full GPU and multi-node production readiness.
