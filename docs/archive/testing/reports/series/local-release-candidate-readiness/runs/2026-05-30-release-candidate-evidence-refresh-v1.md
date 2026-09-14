# Release-Candidate Evidence Refresh V1

- timestamp_utc: 2026-05-30T14:25:20Z
- git_sha: aa78c3e588af04ca89a004291a7b9d81ad3c1cb4
- stream: release-candidate
- milestone: post-security-replication-P8 release-candidate evidence refresh
- validation_gate: `scripts/run_local_release_candidate_evidence_bundle.sh`
- result: pass

## Work Order

- active_lane: local release-candidate evidence refresh after production-security, verifier, replication mTLS, and P8 CH benchmark changes
- lane_classification: open
- falsifiable_claim: the repo can produce a current attachable local release-candidate evidence bundle whose manifest, log, and gap file capture the post-2026-05-29 envelope: all application-driver evidence, dump/restore/DR evidence, opt-in production TLS plus SCRAM verifier profile, replication generated-CA mTLS AppendEntries proof, P8 GPU residency preflight, and the P8 CH-benCHmark-derived residency harness/baseline with explicit 6 GiB tier and production-orchestration gaps
- evidence_required: checked wrapper assertions, deterministic smoke/self-checks, a clean full evidence-bundle run, artifact paths, checksum, and preserved non-claim lines
- non_goals: no new SQL/protocol/catalog expansion, no production object storage, no physical page-image backup, no live systemd/Kubernetes rollout, no certificate lifecycle automation, no enterprise identity/KMS/HSM, no durable GPU pages, no autonomous cache daemon, no external orchestration, and no 6 GiB+ benchmark tier execution
- minimum_meaningful_chunk: one checked evidence refresh report plus wrapper assertions that prevent stale release-candidate claims
- validation_gate: evidence-bundle smoke, P8 CH self-check, focused clippy fix, full evidence bundle, and `git diff --check`
- stop_rule: stop after the current release-candidate evidence refresh is checked, documented, and validated with artifact paths/checksum and explicit remaining gaps
- escalation_trigger: full evidence bundle failure, contradictory required gap lines, unbounded P8 CH evidence, dirty-worktree constraint, or only external production target decisions remaining

## Wrapper Refresh

The release-candidate preflight and bundle assertions now require current
post-bundle evidence lines for:

- production SCRAM verifier-file configuration plus plaintext/verifier conflict rejection
- replication generated-CA mTLS AppendEntries proof and production certificate lifecycle/trust-distribution gap
- P8 CH-benCHmark-derived residency harness self-check and baseline report reference
- explicit 6 GiB retained-tier blocker: streaming generator, longer run window, and cleanup budget

The first full-bundle attempt exposed a validation regression in the P8 example:
`cargo clippy --all-features -- -D warnings` rejected two static strings wrapped
in `format!`. Commit `aa78c3e5` fixed that before the clean bundle was produced.

## Command Result

```bash
scripts/run_local_release_candidate_evidence_bundle.sh
```

The command passed and produced a clean release-review bundle under `target/`.

- evidence_dir: `target/release-candidate-evidence/20260530T142520Z`
- manifest_path: `target/release-candidate-evidence/20260530T142520Z/manifest.env`
- preflight_log_path: `target/release-candidate-evidence/20260530T142520Z/local-release-candidate-preflight.log`
- remaining_gap_file: `target/release-candidate-evidence/20260530T142520Z/remaining-gaps.env`
- tarball_path: `target/release-candidate-evidence/20260530T142520Z.tar.gz`
- sha256: `ab922dd66537101f58eedc068d895ba252f3e9cffc6baae29335d74ed46e1f82`

## Manifest Facts

- git_head: aa78c3e588af04ca89a004291a7b9d81ad3c1cb4
- git_branch: main
- git_upstream: origin/main
- git_dirty: 0
- cargo_version: cargo 1.94.0
- rustc_version: rustc 1.94.0
- psql_version: PostgreSQL 16.14
- go_version: go1.26.3
- javac_version: javac 21.0.11
- maven_version: Apache Maven 3.9.16 on Java 21.0.11
- gradle_version: Gradle 9.5.1 on Java 21.0.11
- nvidia_smi_version: NVIDIA GeForce RTX 3090, driver 595.58.03

## Supported Local Envelope Proven

The refreshed bundle proves the current local release-candidate envelope across:

- fmt, clippy, all-features tests, psql golden coverage, and scorecard freshness
- checked application-driver smokes for `tokio-postgres`, `sqlx`, `node-postgres`, `asyncpg`, `psycopg`, `pgx`, JDBC, and R2DBC
- PostgreSQL 16 `pg_dump`/`pg_restore`, `pg_dumpall --globals-only --no-role-passwords`, bounded privilege restore, and local backup/PITR/DR
- packaged replication service/systemd/Kubernetes/compose evidence plus local generated-CA mTLS AppendEntries channel security
- retained CUDA allocation, zero-H2D resident routes, first accepted-route CUDA event timing samples, operator warmup, and scheduler-friendly maintenance
- opt-in production security profile v1 with required TLS, SCRAM-SHA-256 valid/invalid password paths, verifier-file hygiene, and recovery after failed auth
- P8 CH-benCHmark-derived residency harness self-check plus the checked baseline report at `docs/testing/reports/series/p8-ch-residency-setup/runs/2026-05-30-p8-ch-benchmark-residency-baseline-v1.md`

## Explicit Non-Claims

The generated remaining-gap file preserves the current production boundaries:

- production replication certificate lifecycle and trust distribution
- 6 GiB+ P8 CH benchmark tiers until streaming/on-disk generation, a longer run window, and cleanup budget are approved
- certificate lifecycle automation, enterprise identity, KMS/HSM/secret-manager integration, audit hash-chain, row-level security, masking, and broad authorization
- pg_dumpall bootstrap-role restore and database ACL restore
- physical page-image backup and production object storage
- live background scheduling, live systemd supervision, live Kubernetes rollout, and production timeline failover
- durable GPU pages, autonomous cache daemon, external orchestration, broad retained expressions, and broad CUDA event timing

## Validation

- `scripts/run_local_release_candidate_evidence_bundle_smoke.sh`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --self-check`: passed
- `cargo clippy --all-features --example p8_ch_benchmark_residency_probe -- -D warnings`: passed
- `scripts/run_local_release_candidate_evidence_bundle.sh`: passed
- `git diff --check`: passed before report creation

## Follow-Up Slice

No additional bounded release-candidate wrapper slice remains in this local
envelope. The next credible work requires an external trigger: an approved
longer 6 GiB P8 benchmark window with generation/cleanup budget, a production
certificate lifecycle/trust-distribution decision, a production object-store or
physical-backup format decision, or a named live deployment target.
