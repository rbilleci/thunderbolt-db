# Production Security Profile v1

- round: `2026-05-29-production-security-profile-v1`
- date: 2026-05-30
- scope: PostgreSQL compatibility endpoint connection-security profile

## Result

Production security profile v1 is checked as an opt-in local profile. The
default server mode remains the local/dev trust-auth no-TLS profile.

Checked production-profile claims:

- incomplete production config is rejected before serving clients
- production TCP clients must negotiate TLS before normal startup
- valid TLS plus SCRAM-SHA-256 password authentication succeeds through real
  `psql`
- invalid password authentication fails
- the same server recovers for a later valid TLS+SCRAM client
- non-TLS production clients are rejected

Explicit non-claims:

- no mTLS
- no enterprise identity provider integration
- no KMS/HSM or external secret-manager integration
- no certificate rotation automation
- no replication-channel security claim
- no audit hash chain
- no row-level security
- no masking
- no broad authorization model
- no live production deployment policy

## Validation

- `scripts/run_connection_security_posture_preflight.sh`: passed
- `cargo test -p gpu_db_protocol --all-features`: passed
- `scripts/run_local_product_preflight.sh`: passed
- `scripts/run_local_release_candidate_preflight.sh`: passed
- `scripts/run_local_release_candidate_evidence_bundle_smoke.sh`: passed
