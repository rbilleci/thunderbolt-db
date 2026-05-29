# Security and Compliance Architecture

This document operationalizes the security/compliance intent from `DESIGN.md` into implementation-ready controls.

## Scope

- Authentication and authorization model
- Data protection (in transit, at rest, in memory)
- Auditability and tamper evidence
- Regulatory control mapping
- Validation and evidence requirements by phase

## Current implementation status

The default local/dev PostgreSQL compatibility endpoint uses trust-style
startup: it declines SSLRequest and GSSENCRequest negotiation with `N`, accepts
normal startup with PostgreSQL `AuthenticationOk`, and has no password verifier
catalog. Password and SASL frontend frames are parsed for protocol hygiene but
rejected after startup in this profile.

Production security profile v1 is an explicit opt-in for the PostgreSQL
compatibility endpoint. It requires configured certificate/key material plus a
configured auth user/password credential, accepts PostgreSQL SSLRequest, requires
TLS before normal startup, runs SCRAM-SHA-256 password authentication, rejects
invalid passwords, and rejects non-TLS production clients. The checked
`scripts/run_connection_security_posture_preflight.sh` gate exercises incomplete
config rejection, a valid TLS+SCRAM `psql` path, invalid-password failure,
same-server recovery, and non-TLS rejection.

The profile is intentionally narrow. It does not claim mTLS, enterprise
identity, KMS/HSM, external secret-manager integration, certificate rotation
automation, broad authorization policy, replication-channel security, audit hash
chain, row-level security, masking, or a production deployment policy.

## Security principles

1. **Default deny** for privileged operations.
2. **Least privilege** for users, services, and internal components.
3. **Separation of duties** between operational admin and audit roles.
4. **Cryptographic integrity** for logs and critical metadata.
5. **No hidden bypass paths** between CPU and GPU execution modes.

## Identity, authn, authz

### Authentication
- SCRAM-SHA-256 minimum baseline for database auth.
- TLS required for all client and replication channels.
- Support enterprise identity providers in phased rollout.

### Authorization
- Role-based access control for schema/data access.
- Privileged actions (replication reconfig, snapshot install, failover ops) restricted to admin roles.
- Security-sensitive SQL functions are explicitly allowlisted.

### Session security
- Session parameters that weaken safety (if any) are disabled in production profile.
- Administrative sessions are audited with stronger retention and tamper checks.

## Data protection model

### In transit
- TLS 1.2+ required (TLS 1.3 preferred).
- Strong cipher policy; no legacy/weak suites.
- Certificate lifecycle management documented and testable.

### At rest
- Disk encryption for WAL/data/snapshots/backups.
- Key management externalized to KMS/HSM where available.
- Backup encryption with per-backup key metadata and rotation policy.

### In memory (CPU/GPU)
- GPU memory treated as sensitive in production.
- H100+ confidential compute modes preferred for regulated workloads.
- Device memory must be zeroized on deallocation, device drain, and process shutdown where supported.
- CPU pinned memory pools cleared before reuse.

## Audit and tamper evidence

- Compliance audit stream is separate from operational logs.
- Audit records include actor, action, object, timestamp, outcome, and correlation IDs.
- Hash chain links each record to previous digest.
- Periodic anchor hash export to external immutable store.
- Audit log mutation/deletion prohibited by policy and controls.

## Row-level security and masking

- RLS semantics are uniform across CPU and GPU paths.
- Any operator path that cannot enforce RLS must fallback to CPU path that can.
- Data masking functions must have deterministic behavior across execution targets.

## Compliance control mapping (initial)

### PCI DSS (high-level mapping)
- Data protection at rest/in transit -> encryption controls
- Access control -> RBAC + MFA/IdP integration (where deployed)
- Logging/monitoring -> tamper-evident audit + metrics + alerting
- Vulnerability management -> dependency SBOM + patch workflow

### SOC 2 (high-level mapping)
- Security: authz/authn + change controls + audit logs
- Availability: failover, backup, recovery testing
- Confidentiality: encryption and access boundaries
- Processing integrity: deterministic replay + parity checks

### GDPR (high-level mapping)
- Data minimization and access control at query/runtime level
- Deletion workflows include primary, replicas, caches, and backup policy boundaries
- Access and erasure events auditable

## Threat model checkpoints

- Protocol abuse and parser fuzzing
- Privilege escalation through SQL/function paths
- GPU memory leakage between sessions/tenants
- Replication channel interception/tampering
- Backup theft and key compromise scenarios

## Phase gates and evidence

### v0
- TLS enforced
- SCRAM baseline
- Audit stream and hash chain scaffold
- Security test harness (fuzzing + authz tests)

### v0.5
- Replication channel mTLS
- Role-gated replication/failover operations
- External anchor for audit digests

### v1
- Expanded control mapping package
- Runbooks for key rotation and incident response
- Compliance evidence bundle generation pipeline

## Required tests

- Authn/authz negative tests
- RLS parity tests CPU vs GPU
- Audit completeness tests
- Crypto configuration validation tests
- Secret handling and redaction tests in logs/metrics
