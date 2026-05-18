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

### 1b) Bounded Replication Deployment Preflight

Run before treating the current P6 replication packaging envelope as deployable in a local/operator test environment:

```bash
scripts/run_replication_deployment_preflight.sh
```

Pass criteria:

- `operational_replication_deployment_preflight=passed`
- Packaged follower service smoke passes.
- Systemd unit/environment contract verification passes.
- Kubernetes two-follower manifest contract verification passes.
- Docker Compose restart smoke passes with one follower restarted and replayed from the durable prefix while the other follower stays live.

Boundaries:

- This is not a live systemd installation.
- This is not a live Kubernetes rollout.
- Missing or unusable local Docker/Compose is an environment blocker for this preflight, not a product pass.

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
5. For multi-segment archive proof, persist with `Engine::persist_durable_wal_archive(...)` and recover with `Engine::recover_from_durable_wal_archive(...)`; the manifest validates segment paths, per-segment record counts, per-segment transaction ranges, overall durable record count, last durable transaction id, and increasing transaction order before replay.
6. For transaction-bound PITR proof, recover with `Engine::recover_from_durable_wal_archive_to_txn(...)`; the archive reader validates the whole archive, then replays only the exact requested transaction-bound prefix and rejects before-first, beyond-durable, or missing-boundary targets.
7. For timestamp-bound PITR proof, recover engine-written archives with `Engine::recover_from_durable_wal_archive_to_timestamp_micros(...)`; the archive reader validates the whole archive, requires per-transaction timestamp metadata, then replays only an exact timestamp-bound prefix while rejecting missing metadata, before-first, beyond-durable, between-boundary, and ambiguous timestamp targets.
8. For checkpoint-backed base-plus-archive restore, recover with `Engine::recover_from_durable_wal_checkpoint_and_archive_to_txn(...)` or `Engine::recover_from_durable_wal_checkpoint_and_archive_to_timestamp_micros(...)`; the recovery path validates the base checkpoint, validates the archive target, requires the archive prefix to overlap and match the base boundary, then replays the base plus archive suffix only.
9. For local PITR-branch archive cleanup, inspect `Engine::plan_durable_wal_archive_retention_to_txn(...)` or `Engine::plan_durable_wal_archive_retention_to_timestamp_micros(...)`, then apply `Engine::apply_durable_wal_archive_retention_to_txn(...)` or `Engine::apply_durable_wal_archive_retention_to_timestamp_micros(...)`; cleanup validates the whole archive, rewrites the manifest and segment files to the exact retained transaction/timestamp prefix, preserves retained timestamp metadata, and removes only obsolete post-target segments after the replacement manifest is installed.
10. For checkpoint-backed base-window archive cleanup, inspect `Engine::plan_durable_wal_archive_retention_from_checkpoint(...)`, then apply `Engine::apply_durable_wal_archive_retention_from_checkpoint(...)`; cleanup validates the base checkpoint against the archive before rewriting, retains the base-boundary record plus durable suffix, and preserves later base-plus-archive transaction/timestamp restore.
11. For checkpoint-backed PITR-window cleanup selection, inspect `Engine::plan_durable_wal_archive_retention_from_checkpoint_window(...)`, then apply `Engine::apply_durable_wal_archive_retention_from_checkpoint_window(...)`; cleanup validates archive timestamps, computes the operator-supplied PITR cutoff, applies only when the base checkpoint is old enough to preserve that window, and rejects unsafe recent-base windows before mutation.
12. For local streaming-style archive ingestion, write the incoming WAL segment with the checksummed segment format, then register it with `Engine::ingest_durable_wal_archive_segment(...)`; ingestion validates the existing archive, validates the incoming segment transaction order, preserves required timestamp metadata, and installs the new manifest only after validation.
13. For local PITR timeline-branch proof, fork a transaction or timestamp target with `Engine::fork_durable_wal_archive_timeline_to_txn(...)` or `Engine::fork_durable_wal_archive_timeline_to_timestamp_micros(...)`; the helper validates the source archive and target, writes a branch archive manifest containing only the selected durable prefix, and installs sidecar timeline identity plus parent ancestry metadata. Register sidecars with `Engine::register_durable_wal_archive_timeline(...)` so local timeline ids stay unique, child timelines require an already registered parent, and the branch manifest validates before the registry is updated. Select a registered local failover target with `Engine::select_durable_wal_archive_timeline(...)`, or recover directly with `Engine::recover_from_registered_durable_wal_archive_timeline(...)`; both paths reject missing registry entries, stale sidecars, and corrupted branch archives before recovery.
14. For local object-bundle backup proof, export a validated archive with `Engine::export_durable_wal_archive_object_backup(...)` and restore it with `Engine::restore_durable_wal_archive_object_backup(...)`; restore verifies every manifest/segment object length and checksum, cross-checks the manifest object bytes against the backup manifest metadata, stages restored segment files until every object verifies, and then installs the restored archive manifest.
15. Treat physical page-image base backups, production object-storage APIs, production timeline failover/GC orchestration beyond local registered-target selection, and live background cleanup scheduling as not yet implemented.

Failure criteria:

- resumed state invents gaps past the snapshot boundary
- resumed state advances `applied_index` beyond durable committed boundary
- resumed state recovers uncommitted tail as if durable
- recovered relational state includes rows not present in the flushed WAL segment

## 4c) Local Operational Replication Smoke

Current implemented scope:

- `scripts/run_replication_cluster_smoke.sh` runs a packaged-local 3-node Raft smoke scenario through the checked-in Rust example entrypoint.
- `scripts/run_replication_packaged_smoke.sh` builds `gpu_db_replication`'s `operational_cluster_smoke` example and executes the resulting local binary from `target/debug/examples/operational_cluster_smoke`.
- `scripts/run_replication_multiprocess_smoke.sh` builds `gpu_db_replication`'s `operational_multiprocess_smoke` example and executes a bounded parent-plus-child-process deployment proof: the parent leader sends TCP AppendEntries frames to two independent follower child processes, then verifies follower catch-up and read-after-apply output.
- `scripts/run_replication_service_smoke.sh` builds `gpu_db_replication`'s `operational_service_smoke` example and executes a bounded follower-service deployment proof: the parent leader sends multiple sequential TCP AppendEntries requests to two long-running follower service processes, verifies committed apply/read-after-apply output, and observes controlled service shutdown.
- `scripts/run_replication_supervised_restart_smoke.sh` reuses the packaged service example to execute a bounded restart-supervision proof: the parent leader restarts one follower service, replays the full durable prefix after restart, keeps a second follower service live, and verifies both followers catch up through real TCP AppendEntries traffic before controlled shutdown.
- `scripts/run_replication_container_smoke.sh` builds a local Docker image from the same `operational_service_smoke` example, starts two follower service containers on a user-defined local network, drives real TCP AppendEntries from the host leader through published container ports, verifies committed apply/read-after-apply output, and observes controlled shutdown.
- `scripts/run_replication_container_restart_smoke.sh` reuses that local Docker image path to start two follower containers, restart one follower container, replay the full durable prefix into the restarted container while a second container remains live, and verify both containers catch up before controlled shutdown.
- `scripts/run_replication_compose_restart_smoke.sh` builds the same local Docker image, starts two follower services from `docker/replication-service/compose-smoke.yml`, restarts one service through `docker compose restart`, re-discovers its published port, replays the full durable prefix into the restarted service while a second service remains live, and verifies both services catch up before Compose teardown.
- `scripts/run_replication_systemd_verify.sh` builds the same follower service binary, validates `systemd/replication-follower/gpu-db-replication-follower@.service` with `systemd-analyze`, and checks the instance environment examples preserve the follower id / expected request count / listen address command contract.
- `scripts/run_replication_kubernetes_verify.sh` builds the same follower service binary and validates `k8s/replication-service/follower-services.yml` as two follower Deployments plus two Services that preserve the follower id / expected request count / listen address / append port command contract. If `kubectl` exists locally, the script also runs a client-side dry-run; otherwise it reports manifest-contract-only evidence.
- `scripts/run_replication_deployment_preflight.sh` aggregates the packaged follower-service smoke, systemd contract verification, Kubernetes manifest verification, and Docker Compose restart smoke into one bounded local deployment preflight.
- The scenario demonstrates leader write admission, typed append-entries request/response handling with a tested binary frame codec and single-request TCP send/serve helper, follower catch-up, read-after-apply, deterministic request-vote election before failover, old-leader `NotLeader` rejection after failover, and continued writes on the elected leader.
- The smoke commands emit an `operational_deployment_preflight=passed` line only when the smoke proof passes and the current TCP append-entries transport evidence, deterministic election evidence, packaged-local entrypoint evidence, deployment scope, and gap status are explicitly reported.

Current simulated/not-yet-implemented scope:

- No live production service manager or daemon supervision. The implemented packaging proofs are reproducible local binary, Docker, Docker Compose, systemd-artifact, Kubernetes-manifest validation, and aggregate deployment-preflight harnesses, including bounded multi-process parent/follower, long-running follower-service, service restart-plus-replay, container deployment, container restart-plus-replay, Compose-managed restart-plus-replay, checked systemd unit/environment command-contract proofs, checked Kubernetes Deployment/Service command-contract proofs, and one combined local preflight gate.
- No membership reconfiguration. The current election proof is deterministic request-vote voting inside the local smoke harness, not a timer-driven production election loop.
- No live Kubernetes deployment harness. The checked Kubernetes manifests are a local artifact contract for the existing follower service, not a production rollout, readiness/liveness probe, storage, or service-discovery proof.

Run from repository root:

```bash
scripts/run_replication_cluster_smoke.sh
scripts/run_replication_packaged_smoke.sh
scripts/run_replication_multiprocess_smoke.sh
scripts/run_replication_service_smoke.sh
scripts/run_replication_supervised_restart_smoke.sh
scripts/run_replication_container_smoke.sh
scripts/run_replication_container_restart_smoke.sh
scripts/run_replication_compose_restart_smoke.sh
scripts/run_replication_systemd_verify.sh
scripts/run_replication_kubernetes_verify.sh
scripts/run_replication_deployment_preflight.sh
```

Pass criteria:

- Output includes `operational_replication_smoke=passed`.
- Output includes `operational_deployment_preflight=passed`.
- Output includes `deployment_scope=packaged_local_three_node_raft_smoke`.
- Output includes `deployment_transport=single_request_tcp_append_entries` with append batch, heartbeat batch, and follower-ack counts.
- Output includes `deployment_election=deterministic_request_vote` with candidate id, elected term, vote count, quorum, and elected status.
- Output includes `deployment_package=local_cargo_example_binary` with the Rust example entrypoint and both smoke script paths.
- Aggregate preflight output includes `operational_replication_deployment_preflight=passed`.
- Follower output reports `follower_caught_up=true`.
- `follower_read_after_apply` includes the post-failover write.
- `promoted_leader_commit`, `follower_commit`, and `follower_applied` are equal.
- `failover_admission_gate` reports old-leader write rejection and `promoted_node_role=Leader`.
- `deployment_gap_network_transport`, `deployment_gap_automatic_election`, and `deployment_gap_packaged_deployment` report `implemented`.
- Multi-process smoke output includes `operational_replication_multiprocess_smoke=passed`, `multiprocess_transport=tcp_append_entries`, one `multiprocess_follower ... caught_up=true` line per child process, and explicit `deployment_gap_packaged_multiprocess_smoke=implemented`.
- Service smoke output includes `operational_replication_service_smoke=passed`, `service_transport=tcp_append_entries`, one `service_follower ... caught_up=true` line per follower service, `service_shutdown=controlled`, `deployment_gap_long_running_service=implemented`, and `deployment_gap_container_deployment=missing`.
- Supervised-restart smoke output includes `operational_replication_supervised_restart_smoke=passed`, `supervised_restart_transport=tcp_append_entries`, the restarted follower reporting catch-up before and after restart, `supervised_restart_replay=full_durable_prefix_after_restart`, `deployment_gap_service_restart_supervision=implemented_bounded_local_smoke`, and explicit missing lines for production supervision and Kubernetes deployment.
- Container smoke output includes `operational_replication_container_smoke=host_parent_passed`, `container_deployment_scope=host_leader_two_follower_service_containers`, one `service_follower ... caught_up=true` line per follower container, `service_shutdown=controlled`, and `deployment_gap_container_deployment=implemented`.
- Container-restart smoke output includes `operational_replication_container_restart_smoke=host_parent_passed`, `container_restart_transport=tcp_append_entries`, the restarted follower container reporting catch-up before and after restart, `container_restart_replay=full_durable_prefix_after_restart`, `deployment_gap_container_restart_supervision=implemented_bounded_local_smoke`, and explicit missing lines for production supervision and Kubernetes deployment.
- Compose-restart smoke output includes `operational_replication_compose_restart_smoke=host_parent_passed`, `compose_restart_transport=tcp_append_entries`, the restarted Compose service reporting catch-up before and after restart, `compose_restart_replay=full_durable_prefix_after_restart`, `deployment_gap_compose_restart_supervision=implemented_bounded_local_smoke`, and explicit missing lines for production supervision and Kubernetes deployment.
- Systemd verification output includes `operational_replication_systemd_verify=passed`, the checked unit and environment file paths, `systemd_service_contract=follower_service id_expected_requests_listen`, `deployment_gap_production_service_manager=implemented_unit_syntax_and_command_contract`, and explicit missing lines for live systemd supervision and Kubernetes deployment.
- Kubernetes verification output includes `operational_replication_kubernetes_verify=passed`, the checked manifest path, `kubernetes_service_contract=follower_service id_expected_requests_listen_append_port`, `deployment_gap_kubernetes_deployment=implemented_manifest_contract`, and `deployment_gap_live_kubernetes_rollout=missing`.

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

Do not manually prune tuple versions, relational row keys, or equality-index entries. Relational indexes remain volatile and are rebuilt from the durable WAL prefix during recovery; the current control file covers one selected durable segment, and the archive manifest covers ordered multi-segment replay, local segment ingestion, exact transaction-bound prefix restore, exact timestamp-bound prefix restore for engine-written archives, checkpoint-backed base-plus-archive restore, transaction/timestamp-target suffix cleanup, checkpoint-backed base-window archive cleanup, checkpoint-backed PITR-window cleanup selection, local PITR timeline-branch forks with registry-checked ancestry metadata and named target recovery, and checked local object-bundle backup/restore. Live background cleanup scheduling is still future storage work.

## 6) Release Evidence Bundle

For each release candidate, attach:

1. Git commit SHA.
2. Output logs from fmt/clippy/test gates.
3. Any incident notes since previous candidate.
4. Short statement confirming WAL-before-visibility and role-gating invariants were revalidated.

This keeps DR/security posture auditable and repeatable while implementation iterates toward full GPU and multi-node production readiness.
