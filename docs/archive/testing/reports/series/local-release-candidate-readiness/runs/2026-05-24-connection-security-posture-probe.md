# Connection Security Posture Probe

- timestamp_utc: 2026-05-24T05:38:15Z
- git_sha: a898af276e78e810094f3ea3517e5fb39cd95fdb
- stream: security
- milestone: connection-authentication and TLS production-readiness probe
- validation_gate: `scripts/run_connection_security_posture_preflight.sh`
- result: pass

## Context

This report closes the bounded security posture probe for the current
PostgreSQL compatibility endpoint. It reconciles the production security phase
gate intent in `docs/architecture/07-security-and-compliance.md` with the
implemented local/dev endpoint behavior in `crates/protocol/src/lib.rs` and
`crates/protocol/src/bin/gpu-db-server.rs`.

The architecture document still defines SCRAM-SHA-256, TLS, mTLS, certificate
lifecycle, audit hash-chain, row-level security, masking, and compliance
evidence as production controls. Those are target controls, not current support
claims for the compatibility endpoint.

## Observed Code Reality

- `crates/protocol/src/lib.rs` parses PostgreSQL SSLRequest and GSSENCRequest
  startup packet codes.
- `crates/protocol/src/bin/gpu-db-server.rs` responds to SSLRequest and
  GSSENCRequest with `N`.
- Normal startup writes PostgreSQL `AuthenticationOk`.
- PasswordMessage frames are parsed but rejected after startup with
  `password messages are not supported after startup`.
- SASL initial/response frames are parsed but rejected with
  `SASL authentication is not supported`.
- The role metadata catalog exposes `rolpassword` as a nullable column but
  stores no password verifier for bootstrap or created roles.

## Supported Local Envelope

- Local/dev PostgreSQL compatibility endpoint.
- Trust-style startup authentication via `AuthenticationOk`.
- No TLS negotiation for client connections.
- No GSS encryption negotiation.
- Bounded role metadata and ACL enforcement for supported catalog/object
  operations after a connection exists.
- Protocol parser coverage for password/SASL frontend frames so unsupported
  authentication traffic fails explicitly after startup.

## Current Non-Claims

- SCRAM-SHA-256: missing.
- Password authentication/storage: missing.
- TLS client connections: missing.
- Replication mTLS: missing.
- Certificate lifecycle: missing.
- Audit hash-chain: missing.
- Row-level security: missing.
- Masking: missing.
- Production security profile: missing.

The local release-candidate evidence must not be interpreted as production
database connection-security readiness.

## Actionable Trigger

Full auth/TLS implementation becomes actionable only after a named production
security profile defines:

- password/verifier storage and secret-handling policy,
- SCRAM mechanism requirements and migration behavior,
- TLS/mTLS certificate issuance, rotation, validation, and deployment policy,
- replication-channel security requirements,
- audit/compliance evidence expectations,
- acceptance tests for real clients and operator deployment.

## Reproduction

```bash
scripts/run_connection_security_posture_preflight.sh
```
