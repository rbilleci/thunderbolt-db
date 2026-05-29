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

### 1c) Bounded Backup/PITR/DR Drill

Run as the recurring local backup/PITR/DR verification gate:

```bash
scripts/run_backup_pitr_dr_drill.sh
```

Pass criteria:

- `backup_pitr_dr_drill=passed`
- Focused base-plus-archive transaction and timestamp restore tests pass.
- Checkpoint-backed PITR-window archive retention tests pass.
- Scheduler-safe maintenance cleanup preflight passes, including stale-sidecar and unsafe-window rejection-before-mutation evidence.
- Object-bundle backup preflight passes, including export/restore/recover evidence and corrupt-object rejection before final restored archive state.
- MVCC retention boundary tests pass.

Boundaries:

- This is a local checkpoint-control, WAL archive, timeline-registry, and file-backed object-bundle drill.
- This is not a physical page-image base-backup restore.
- This is not a production object-storage integration.
- This is not live background cleanup scheduling or production timeline failover orchestration.

### 1d) Local Resilience Drill

Run as the combined local game-day gate for the currently supported backup/PITR/DR and operational replication envelopes:

```bash
scripts/run_local_resilience_drill.sh
```

Pass criteria:

- `local_resilience_drill=passed`
- Backup/PITR/DR drill evidence includes `backup_pitr_dr_drill=passed`.
- Replication deployment preflight evidence includes `operational_replication_deployment_preflight=passed`.
- The output names the remaining physical backup, production object-storage, live scheduling, live systemd/Kubernetes rollout, and production timeline-failover gaps.

Boundaries:

- This is an aggregate local verification command over existing checked gates.
- This is not a physical page-image base-backup restore.
- This is not a production object-storage integration.
- This is not live systemd or Kubernetes rollout.
- This is not live background scheduling or production timeline failover orchestration.

### 1e) Local PostgreSQL-Compatible Product Preflight

Run before treating the current supported PostgreSQL-facing envelope as a local release candidate:

```bash
scripts/run_local_product_preflight.sh
```

Pass criteria:

- `local_product_preflight=passed`
- Application-driver evidence includes the checked `tokio-postgres`, `sqlx`, `node-postgres`, `asyncpg`, `psycopg`, `pgx`, JDBC, and R2DBC smoke gates.
- Dump/restore evidence includes plain, custom, directory, tar, parallel directory, clean, insert-style, and split schema/data restore modes for the supported public object subset.
- Privilege restore evidence includes bounded schema `USAGE`/`CREATE`, relation, sequence, zero-argument function `EXECUTE`, and default table ACLs for a pre-created bounded role.
- Local resilience evidence includes backup/PITR/DR plus replication deployment preflight.
- The output names the remaining physical backup, production object-storage, live scheduling, live systemd/Kubernetes rollout, and production timeline-failover gaps.

Boundaries:

- This is an aggregate local product-readiness command over existing checked gates.
- This does not add new SQL/protocol/catalog support.
- This does not claim R2DBC pooling or broad driver parity.
- This does not claim production orchestration or broad PostgreSQL parity beyond the supported subset.

### 1f) Local GPU Residency Product Preflight

Run before treating the current GPU-resident execution envelope as locally checked:

```bash
scripts/run_local_gpu_residency_preflight.sh
```

Pass criteria:

- `local_gpu_residency_preflight=passed`
- Residency baseline evidence includes retained CUDA device-memory support and zero-H2D resident route proofs.
- Warmup evidence includes dry-run/apply parity, invalidated-entry refresh, memory-pressure skips, oversized-budget rejection, and route-readiness checks.
- Maintenance evidence includes scheduler-friendly basic, invalidated, memory-pressure, and oversized-budget ticks.
- The output names the remaining durable GPU page, autonomous cache-daemon, external orchestration, broad retained-expression, and broad CUDA-event-timing gaps.

Boundaries:

- This is an aggregate local verification command over existing checked P8/P7 gates.
- This does not create durable GPU pages or a production cache daemon.
- This does not claim broad retained-expression or broad CUDA-event-timing coverage.
- Missing local NVIDIA/CUDA runtime evidence is an environment blocker for this preflight, not a product pass.

### 1g) Local Release-Candidate Preflight

Run before treating the combined validation, PostgreSQL-facing, GPU-residency, and connection-security posture envelope as locally checked for a release candidate:

```bash
scripts/run_local_release_candidate_preflight.sh
```

Pass criteria:

- `local_release_candidate_preflight=passed`
- Validation evidence includes fmt, clippy, all-features tests, real psql golden coverage, regenerated compatibility scorecard freshness, and checked-in scorecard parity.
- PostgreSQL product evidence includes the local product preflight over application drivers, pg_dump/restore including bounded schema/relation/sequence/function/default-table privilege restore, and local resilience.
- GPU residency evidence includes the local residency preflight over retained CUDA allocation, zero-H2D resident routes, warmup, and maintenance.
- Connection-security posture evidence includes the local/dev trust-auth, no-TLS boundary, the opt-in production security profile v1 TLS+SCRAM/password gate, and explicit remaining security non-claims.
- The output names the remaining physical backup, production object-storage, live scheduling, live systemd/Kubernetes rollout, production timeline-failover, durable GPU page, autonomous cache-daemon, external orchestration, broad retained-expression, and broad CUDA-event-timing gaps.
- The output names the remaining replication mTLS, certificate lifecycle automation, enterprise identity, KMS/HSM or external secret-manager integration, audit hash-chain, row-level security, masking, and broad authorization gaps.

Boundaries:

- This is a top-level aggregate over existing checked gates.
- This does not add SQL/protocol/catalog support or CUDA kernel/runtime behavior.
- This does not claim production orchestration, durable GPU pages, or broad PostgreSQL/CUDA parity beyond the supported local envelope.
- This claims only the checked production security profile v1 local gate for client TLS plus SCRAM/password authentication. It does not claim production deployment policy, mTLS, certificate rotation automation, enterprise identity, KMS/HSM, external secret-manager integration, audit hash-chain, row-level security, masking, or broad authorization.

### 1h) Local Connection-Security Posture Preflight

Run before treating the current client-connection security boundary as documented:

```bash
scripts/run_connection_security_posture_preflight.sh
```

Pass criteria:

- `connection_security_posture_preflight=passed`
- The default local/dev profile remains documented as trust-auth with no TLS.
- `connection_security_posture_preflight_production_profile_v1=passed`
- `connection_security_posture_preflight_production_config_validation=passed`
- `connection_security_posture_preflight_production_tls_required=passed`
- `connection_security_posture_preflight_production_scram_sha_256_valid_password=passed`
- `connection_security_posture_preflight_production_scram_sha_256_invalid_password=passed`
- `connection_security_posture_preflight_production_recovery_after_invalid_password=passed`
- The output names the remaining mTLS, enterprise identity, KMS/HSM or secret-manager, certificate rotation, audit hash-chain, row-level security, masking, and broad authorization non-claims.

Boundaries:

- This is a source-truth/code-reality reconciliation gate plus local production-profile smoke.
- Production profile v1 is explicitly opt-in through `--security-profile production` or `GPU_DB_SECURITY_PROFILE=production`.
- Production profile v1 requires explicit TLS certificate/key and auth user/password inputs.
- This is not mTLS, certificate rotation automation, enterprise identity, KMS/HSM, external secret-manager integration, audit hash-chain, row-level security, masking, broad authorization, replication-channel security, or live deployment policy.

### 1i) Local Release-Candidate Evidence Bundle

Run when a release review needs attachable evidence instead of transient console output:

```bash
scripts/run_local_release_candidate_evidence_bundle.sh
```

Pass criteria:

- `local_release_candidate_evidence_bundle=passed`
- The script reports `local_release_candidate_evidence_tarball=<path>` and `local_release_candidate_evidence_sha256=<hash>`.
- The evidence directory contains `manifest.env`, `local-release-candidate-preflight.log`, and `remaining-gaps.env`.
- The embedded top-level preflight evidence includes `local_release_candidate_preflight=passed` plus the required blocked/open gap lines.

Fast wrapper self-check:

```bash
scripts/run_local_release_candidate_evidence_bundle_smoke.sh
```

The smoke uses a deterministic fake preflight with
`LOCAL_RELEASE_CANDIDATE_EVIDENCE_ALLOW_DIRTY=1` and verifies only packaging
mechanics: manifest, preflight log, remaining-gap file, tarball contents,
checksum, and dirty-worktree override behavior.

Boundaries:

- This is a release-review packaging wrapper around the existing local release-candidate preflight.
- By default the wrapper requires a clean git worktree so the evidence maps to a reproducible HEAD.
- Set `LOCAL_RELEASE_CANDIDATE_EVIDENCE_ALLOW_DIRTY=1` only for wrapper development smoke tests.
- The smoke does not prove the full local release-candidate preflight; it proves wrapper packaging behavior only.
- This does not add SQL/protocol/catalog behavior, CUDA runtime coverage, driver support, or production orchestration.

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
13. For local PITR timeline-branch proof, fork a transaction or timestamp target with `Engine::fork_durable_wal_archive_timeline_to_txn(...)` or `Engine::fork_durable_wal_archive_timeline_to_timestamp_micros(...)`; the helper validates the source archive and target, writes a branch archive manifest containing only the selected durable prefix, and installs sidecar timeline identity plus parent ancestry metadata. Register sidecars with `Engine::register_durable_wal_archive_timeline(...)` so local timeline ids stay unique, child timelines require an already registered parent, and the branch manifest validates before the registry is updated. Select a registered local failover target with `Engine::select_durable_wal_archive_timeline(...)`, or recover directly with `Engine::recover_from_registered_durable_wal_archive_timeline(...)`; both paths reject missing registry entries, stale sidecars, and corrupted branch archives before recovery. To prune local timeline artifacts, inspect `Engine::plan_durable_wal_archive_timeline_prune(...)`, then apply `Engine::apply_durable_wal_archive_timeline_prune(...)`; pruning validates registered sidecars and branch archives before mutation, retains the selected target plus ancestors, installs the pruned registry, and removes only unreferenced sidecars plus branch archive manifests/segments after that registry update.
14. For scheduler-safe local maintenance cleanup, inspect `Engine::plan_durable_wal_archive_maintenance_cleanup(...)`, then apply `Engine::apply_durable_wal_archive_maintenance_cleanup(...)`; the dry-run validates checkpoint-backed PITR-window archive retention and registered-timeline pruning together before mutation, so stale timeline sidecars or corrupt branch archives reject the whole cleanup before archive retention changes are installed. Operators can run the checked preflight with `cargo run -p gpu_db_engine --example wal_archive_maintenance_preflight -- --control <CONTROL> --archive-manifest <MANIFEST> --timeline-registry <TIMELINE_REGISTRY> --retain-timeline <timeline> --current-timestamp-micros <now> --pitr-window-micros <window> --recover-retained`, add `--apply` only after reviewing the dry-run evidence, and use `scripts/run_wal_archive_maintenance_preflight_smoke.sh` as the local regression gate. A successful apply performs the validated archive retention, prunes the local timeline registry to the retained target plus ancestors, and preserves `Engine::recover_from_registered_durable_wal_archive_timeline(...)` for the retained target.
15. For local object-bundle backup proof, export a validated archive with `Engine::export_durable_wal_archive_object_backup(...)` and restore it with `Engine::restore_durable_wal_archive_object_backup(...)`; restore verifies every manifest/segment object length and checksum, cross-checks the manifest object bytes against the backup manifest metadata, stages restored segment files until every object verifies, and then installs the restored archive manifest. Operators can run the checked preflight with `cargo run -p gpu_db_engine --example wal_archive_object_backup_preflight -- --archive-manifest <MANIFEST> --backup-manifest <BACKUP> --object-dir <OBJECT_DIR> --restored-manifest <RESTORED_MANIFEST> --restored-segment-dir <RESTORED_SEGMENTS> --recover-timestamp-micros <target>`, use `--restore-only` to verify an existing backup manifest/object directory, and use `scripts/run_wal_archive_object_backup_preflight_smoke.sh` as the local regression gate for export/restore/recover evidence plus corrupt-object rejection-before-install.
16. For the recurring local DR drill, run `scripts/run_backup_pitr_dr_drill.sh`; it aggregates the focused base-plus-archive restore tests, checkpoint PITR-window retention tests, scheduler-safe maintenance preflight, object-bundle backup preflight, and MVCC retention boundary tests into one operator gate.
17. For a combined local resilience game-day gate, run `scripts/run_local_resilience_drill.sh`; it runs the local backup/PITR/DR drill and the local replication deployment preflight, verifies both evidence contracts, and reports the combined supported scope.
18. For a local PostgreSQL-compatible product preflight, run `scripts/run_local_product_preflight.sh`; it runs the checked application-driver gate, pg_dump/pg_restore gate including bounded public ACL/default-privilege restore, and local resilience drill, verifies stable evidence from each, and reports the current supported envelope plus blocked/open gaps.
19. For a top-level local release-candidate preflight, run `scripts/run_local_release_candidate_preflight.sh`; it runs the local validation preflight, PostgreSQL-compatible product preflight, GPU residency preflight, and connection-security posture preflight, verifies all four evidence contracts, and reports the combined local supported envelope plus blocked/open gaps.
20. For connection-security posture reconciliation, run `scripts/run_connection_security_posture_preflight.sh`; it verifies the default local/dev trust-auth no-TLS endpoint boundary, the opt-in production security profile v1 client TLS plus SCRAM/password path, and the remaining security non-claims outside this local proof.
21. For an attachable local release-candidate evidence bundle, run `scripts/run_local_release_candidate_evidence_bundle.sh`; it captures git/tooling facts, the full top-level preflight log, remaining-gap lines, and a tarball checksum under `target/release-candidate-evidence/`. For fast wrapper-only checks, run `scripts/run_local_release_candidate_evidence_bundle_smoke.sh`.
22. Treat physical page-image base backups, production object-storage APIs, automated production timeline failover orchestration beyond local registered-target selection/pruning, live systemd/Kubernetes rollout, and live background cleanup scheduling as not yet implemented.

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

Do not manually prune tuple versions, relational row keys, or equality-index entries. Relational indexes remain volatile and are rebuilt from the durable WAL prefix during recovery; the current control file covers one selected durable segment, and the archive manifest covers ordered multi-segment replay, local segment ingestion, exact transaction-bound prefix restore, exact timestamp-bound prefix restore for engine-written archives, checkpoint-backed base-plus-archive restore, transaction/timestamp-target suffix cleanup, checkpoint-backed base-window archive cleanup, checkpoint-backed PITR-window cleanup selection, local PITR timeline-branch forks with registry-checked ancestry metadata and named target recovery, scheduler-safe local maintenance cleanup dry-run/apply, and checked local object-bundle backup/restore. Live background cleanup scheduling is still future storage work.

## 6) Release Evidence Bundle

For each release candidate, attach:

1. Git commit SHA.
2. Output logs from fmt/clippy/test gates.
3. Any incident notes since previous candidate.
4. Output from `scripts/run_local_release_candidate_preflight.sh` when the release candidate includes the current local PostgreSQL-compatible and GPU-residency envelope, or the tarball/checksum emitted by `scripts/run_local_release_candidate_evidence_bundle.sh` when a durable evidence attachment is needed.
5. Short statement confirming WAL-before-visibility and role-gating invariants were revalidated.

This keeps DR/security posture auditable and repeatable while implementation iterates toward full GPU and multi-node production readiness.
