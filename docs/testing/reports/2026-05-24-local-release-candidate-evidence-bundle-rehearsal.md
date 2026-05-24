# Local Release-Candidate Evidence Bundle Rehearsal

- timestamp_utc: 2026-05-24T07:38:51Z
- git_sha: 3cc040bee433f6f346bb2dbba0b18aa5062110a5
- stream: release-candidate
- milestone: local release-candidate evidence-bundle rehearsal
- validation_gate: `scripts/run_local_release_candidate_evidence_bundle.sh`
- result: pass

## Work Order

- active_lane: local release-candidate evidence bundle for the integrated current local envelope
- lane_classification: open
- falsifiable_claim: the repository can produce a reproducible, attachable local release-candidate evidence bundle after connection-security posture integration
- evidence_required: full evidence-bundle command with manifest, preflight log, remaining-gap file, tarball, and SHA-256 checksum
- non_goals: no SQL/protocol/catalog expansion, no new driver implementation, no auth/TLS implementation, no production orchestration, no object-storage/physical-backup implementation, and no P8 scope broadening
- minimum_meaningful_chunk: one durable release-review rehearsal report grounded in the generated bundle
- validation_gate: `scripts/run_local_release_candidate_evidence_bundle.sh` and `git diff --check`
- stop_rule: stop after the report is committed and pushed, or after a precise blocker/mismatch is reported
- escalation_trigger: missing local prerequisite, contradictory source-truth claim, non-reproducible evidence, or only broad production-feature follow-up remaining

## Command Result

```bash
scripts/run_local_release_candidate_evidence_bundle.sh
```

The command passed and produced a release-review bundle under `target/`.

- evidence_dir: `target/release-candidate-evidence/20260524T073851Z`
- manifest_path: `target/release-candidate-evidence/20260524T073851Z/manifest.env`
- preflight_log_path: `target/release-candidate-evidence/20260524T073851Z/local-release-candidate-preflight.log`
- remaining_gap_file: `target/release-candidate-evidence/20260524T073851Z/remaining-gaps.env`
- tarball_path: `target/release-candidate-evidence/20260524T073851Z.tar.gz`
- sha256: `ea419a2111890f52130fb42787a26e7dc5ad52be9ea9e14712377a79675cea7b`

## Manifest Facts

- git_head: 3cc040bee433f6f346bb2dbba0b18aa5062110a5
- git_branch: main
- git_upstream: origin/main
- git_dirty: 0
- cargo_version: cargo 1.95.0 (f2d3ce0bd 2026-03-21)
- rustc_version: rustc 1.95.0 (59807616e 2026-04-14)
- psql_version: psql (PostgreSQL) 16.14 (Ubuntu 16.14-0ubuntu0.24.04.1)
- python3_version: Python 3.12.3
- node_version: v22.22.2
- npm_version: 10.9.7
- nvidia_smi_version: NVIDIA GeForce RTX 3090, 595.58.03

Local prerequisite check also confirmed the expected missing future-driver
tooling: `go`, `javac`, Maven, and Gradle are absent.

## Supported Local Envelope Proven

The bundle proves the current local release-candidate envelope across:

- fmt, clippy, all-features tests, psql golden coverage, and scorecard freshness
- checked application-driver smokes for `tokio-postgres`, `sqlx`, `node-postgres`, `asyncpg`, and `psycopg`
- real PostgreSQL 16 `pg_dump`/`pg_restore` paths, including bounded privilege restore
- bounded `pg_dumpall --globals-only --no-role-passwords` global metadata and tablespace-ACL restore
- local backup/PITR/DR, file-backed object-bundle backup, and packaged replication deployment preflights
- retained CUDA allocation, zero-H2D resident routes, first accepted-route CUDA event timing samples, operator warmup, and scheduler-friendly maintenance
- local/dev PostgreSQL connection-security posture: trust-style startup via `AuthenticationOk`, SSL/GSS declined with `N`, unsupported password/SASL frames after startup, and no role password verifier storage

## Explicit Non-Claims

- `pgx`: blocked by missing Go tooling.
- JDBC/R2DBC: blocked by missing Java build tooling.
- pg_dumpall bootstrap-role restore: existing bootstrap role is filtered.
- pg_dumpall database ACL restore: not emitted by `--globals-only`.
- Physical page-image backup: missing.
- Production object storage: missing.
- Live background scheduling: missing.
- Live systemd/Kubernetes rollout: missing.
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
- Production security profile: missing.

## Source-Truth Reconciliation

The generated bundle, `docs/operations/runbooks.md`, `docs/roadmap/v0-v1.md`,
and `docs/compatibility/matrix.md` agree on the four-leg top-level preflight
and the explicit remaining gaps.

One stale README paragraph still described
`scripts/run_local_release_candidate_preflight.sh` as a three-leg gate without
the connection-security posture leg. This report updates that paragraph so the
README matches the actual bundle evidence.

## Follow-Up Slice

No bounded follow-up slice was found inside the current local envelope. The
credible next work requires one of the existing external triggers: Go for
`pgx`, Java build tooling for JDBC/R2DBC, a named live deployment/storage
environment, a named production security profile, or a workload/performance
target for broader P8 behavior.
