# All-Driver Local Release-Candidate Evidence Bundle

- timestamp_utc: 2026-05-29T18:22:51Z
- git_sha: 51ae8e87d4afd4dacc3dd5156700982f19765678
- stream: release-candidate
- milestone: all-driver local release-candidate evidence-bundle rehearsal
- validation_gate: `scripts/run_local_release_candidate_evidence_bundle.sh`
- result: pass

## Work Order

- active_lane: all-driver local release-candidate evidence bundle for the current supported local envelope
- lane_classification: open
- falsifiable_claim: after R2DBC became a checked application-driver gate, the repository can produce a fresh attachable local release-candidate evidence bundle proving the supported local envelope with all eight application-driver smokes, PostgreSQL product preflight, GPU residency preflight, connection-security posture preflight, git/tooling facts, required remaining-gap lines, tarball, and checksum
- evidence_required: full evidence-bundle command with manifest, preflight log, remaining-gap file, tarball, and SHA-256 checksum
- non_goals: no SQL/protocol/catalog expansion, no additional driver parity, no auth/TLS implementation, no live deployment, no object-storage/physical-backup implementation, and no P8 scope broadening
- minimum_meaningful_chunk: one durable release-review report grounded in the generated all-driver bundle
- validation_gate: `scripts/run_local_release_candidate_evidence_bundle.sh` and `git diff --check`
- stop_rule: stop after the report records the bundle artifacts/checksum plus remaining production gaps, or after a precise blocker/mismatch is reported
- escalation_trigger: missing local prerequisite, contradictory source-truth claim, non-reproducible evidence, or only broad production-feature follow-up remaining

## Command Result

```bash
scripts/run_local_release_candidate_evidence_bundle.sh
```

The command passed and produced a release-review bundle under `target/`.

- evidence_dir: `target/release-candidate-evidence/20260529T182251Z`
- manifest_path: `target/release-candidate-evidence/20260529T182251Z/manifest.env`
- preflight_log_path: `target/release-candidate-evidence/20260529T182251Z/local-release-candidate-preflight.log`
- remaining_gap_file: `target/release-candidate-evidence/20260529T182251Z/remaining-gaps.env`
- tarball_path: `target/release-candidate-evidence/20260529T182251Z.tar.gz`
- sha256: `e3f8e0b6b4145ba41a6b8696edbe3430610377d99fa60dfc19e3a0c35fc70663`

## Manifest Facts

- git_head: 51ae8e87d4afd4dacc3dd5156700982f19765678
- git_branch: main
- git_upstream: origin/main
- git_dirty: 0
- cargo_version: cargo 1.94.0 (85eff7c80 2026-01-15)
- rustc_version: rustc 1.94.0 (4a4ef493e 2026-03-02)
- psql_version: psql (PostgreSQL) 16.14 (Ubuntu 16.14-0ubuntu0.24.04.1)
- python3_version: Python 3.12.3
- node_version: v22.22.2
- npm_version: 10.9.7
- go_version: go version go1.26.3 linux/amd64
- javac_version: javac 21.0.11
- maven_version: Apache Maven 3.9.16 with Java 21.0.11
- gradle_version: Gradle 9.5.1 with Java 21.0.11
- nvidia_smi_version: NVIDIA GeForce RTX 3090, 595.58.03
- nvcc_version: CUDA compilation tools release 12.0, V12.0.140

## Supported Local Envelope Proven

The bundle proves the current local release-candidate envelope across:

- fmt, clippy, all-features tests, psql golden coverage, and scorecard freshness
- checked application-driver smokes for `tokio-postgres`, `sqlx`, `node-postgres`, `asyncpg`, `psycopg`, `pgx`, JDBC, and R2DBC
- real PostgreSQL 16 `pg_dump`/`pg_restore` paths, including bounded privilege restore
- bounded `pg_dumpall --globals-only --no-role-passwords` global metadata and tablespace-ACL restore
- local backup/PITR/DR, file-backed object-bundle backup, and packaged replication deployment preflights
- retained CUDA allocation, zero-H2D resident routes, first accepted-route CUDA event timing samples, operator warmup, and scheduler-friendly maintenance
- local/dev PostgreSQL connection-security posture: trust-style startup via `AuthenticationOk`, SSL/GSS declined with `N`, unsupported password/SASL frames after startup, and no role password verifier storage

## Explicit Non-Claims

- pg_dumpall bootstrap-role restore: existing bootstrap role is filtered.
- pg_dumpall database ACL restore: not emitted by `--globals-only`.
- Physical page-image backup: missing.
- Production object storage: missing.
- Live background scheduling: missing.
- Live systemd supervision: missing.
- Live Kubernetes rollout: missing.
- Production timeline failover: missing.
- Durable GPU pages: missing.
- Autonomous cache daemon: missing.
- External orchestration: missing.
- Broad retained expressions: missing.
- Broad CUDA event timing: missing.
- SCRAM-SHA-256: missing.
- Password authentication/storage: missing.
- TLS client connections: missing.
- Replication mTLS: missing.
- Certificate lifecycle: missing.
- Audit hash-chain: missing.
- Row-level security: missing.
- Masking: missing.

## Source-Truth Reconciliation

The generated bundle, `README.md`, `docs/operations/runbooks.md`,
`docs/roadmap/v0-v1.md`, `docs/compatibility/matrix.md`,
`scripts/run_application_driver_smokes.sh`, `scripts/run_local_product_preflight.sh`,
and `scripts/run_local_release_candidate_preflight.sh` agree that the checked
local application-driver envelope now includes all eight suites:
`tokio-postgres`, `sqlx`, `node-postgres`, `asyncpg`, `psycopg`, `pgx`, JDBC,
and R2DBC.

No stale source-truth wording was found that needed a report-time code or docs
reconciliation.

## Follow-Up Slice

No bounded follow-up slice was found inside the current supported local
release-candidate envelope. The credible next work requires one of the existing
external triggers: a named live deployment/storage environment, a named
production security profile, or a workload/performance target for broader P8
behavior.
