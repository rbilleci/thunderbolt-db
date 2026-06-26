# Production Credential Verifier Hygiene v1

Date: 2026-05-30

## Result

Closed.

The opt-in production security profile now accepts PostgreSQL-style
SCRAM-SHA-256 verifier material through `--auth-scram-verifier` /
`GPU_DB_AUTH_SCRAM_VERIFIER` or `--auth-scram-verifier-file` /
`GPU_DB_AUTH_SCRAM_VERIFIER_FILE`. Production startup rejects missing credential
material, malformed verifier material, and conflicting plaintext/verifier
inputs before serving clients.

The existing `--auth-password` / `GPU_DB_AUTH_PASSWORD` path remains available
only as a local/test bootstrap input. It is rejected when combined with verifier
material and is not the production-oriented credential path.

## Evidence

- `scripts/run_connection_security_posture_preflight.sh` authenticates real
  `psql` TLS+SCRAM traffic through a verifier file, rejects invalid passwords,
  and demonstrates same-server recovery for a later valid client.
- The same gate rejects incomplete production config, malformed verifier
  material, conflicting plaintext/verifier inputs, and non-TLS production
  clients.
- The default local/dev profile remains trust-auth/no-TLS and unchanged.
- README, security architecture, operations runbook, compatibility matrix, and
  release-candidate preflight evidence lines now distinguish verifier handling
  from the local/test plaintext bootstrap path.

## Validation

- `cargo test -p gpu_db_protocol --all-features args_production_security_profile_requires_explicit_material -- --nocapture`
- `scripts/run_connection_security_posture_preflight.sh`
- `cargo test -p gpu_db_protocol --all-features`
- `scripts/run_local_product_preflight.sh`
- `scripts/run_local_release_candidate_preflight.sh`

One earlier top-level release-candidate preflight attempt failed while waiting
for a product-smoke server readiness probe. The changed security gate passed
before and after that failure, `scripts/run_local_product_preflight.sh` passed
separately, and the repeated top-level release-candidate preflight passed.

## Remaining Non-Claims

This slice does not claim mTLS, certificate lifecycle automation, enterprise
identity, KMS/HSM, external secret-manager integration, rotation automation,
audit hash-chain, row-level security, masking, broad authorization, replication
channel security, or live deployment policy.
