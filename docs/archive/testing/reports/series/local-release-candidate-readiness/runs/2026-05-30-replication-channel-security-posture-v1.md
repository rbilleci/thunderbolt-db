# Replication Channel Security Posture - 2026-05-30

## Work Order

- Active lane / milestone: replication-channel security posture and bounded mTLS readiness for the local operational replication envelope.
- Lane classification: open -> closed by this slice.
- Falsifiable claim: the repo can prove a local authenticated/encrypted AppendEntries channel while keeping existing plain TCP smokes as dev/test-only transport and naming production certificate lifecycle/trust-distribution gaps.
- Evidence required: local mTLS AppendEntries success, missing certificate/key rejection before serving, missing client certificate rejection, deployment/resilience/release-candidate source-truth reconciliation, and targeted replication validation.
- Non-goals: live systemd/Kubernetes rollout, certificate rotation automation, production trust distribution, external secret-manager/KMS/HSM, enterprise identity, broad authorization, backup/PITR/DR, GPU/P8, or client endpoint security changes.
- Minimum meaningful chunk: one bounded replication-channel security posture slice with local transport/preflight evidence and operator-facing source-truth reconciliation.
- Validation gate: targeted channel-security preflight, `cargo test -p gpu_db_replication --all-features`, replication deployment preflight, local resilience drill, local release-candidate preflight, and `git diff --check`.
- Stop rule: stop when source truth distinguishes the checked local mTLS channel, the plain dev/test transport profile, and remaining production non-claims.

## Result

Closed for the bounded local contract.

Implemented a rustls-backed mTLS AppendEntries helper in `gpu_db_replication`, plus `scripts/run_replication_channel_security_preflight.sh`. The preflight generates a local CA, server certificate, and client certificate, then proves:

- `operational_replication_channel_security_smoke=passed`
- `replication_channel_security_transport=mtls_append_entries`
- `replication_channel_security_missing_material_rejection=passed`
- `replication_channel_security_missing_client_cert_rejection=passed`
- `replication_channel_security_plain_transport_profile=dev_test_only`

The existing service/container/Compose plain TCP smokes remain supported as the explicitly named dev/test profile. Production certificate lifecycle automation and production trust distribution remain open non-claims.

## Validation

- `scripts/run_replication_channel_security_preflight.sh` passed.
- `cargo test -p gpu_db_replication --all-features` passed: 186 tests.
- `scripts/run_replication_deployment_preflight.sh` passed, now including `deployment_preflight_channel_security=local_mtls_append_entries`.
- `scripts/run_local_resilience_drill.sh` passed, now reporting `local_resilience_replication_scope=packaged_service_systemd_contract_kubernetes_manifest_compose_restart_channel_mtls`.
- `scripts/run_local_release_candidate_preflight.sh` passed, now reporting `local_release_candidate_preflight_replication_mtls=local_generated_ca_append_entries_smoke` and `local_release_candidate_preflight_gap_replication_mtls=production_certificate_lifecycle_and_trust_distribution_missing`.
- `git diff --check` passed.

## Remaining Bounded Follow-Up

No further local implementation lane is defensible inside this same contract. The next replication-security work needs a product/security decision for production certificate lifecycle, trust distribution, naming/identity policy, or live rollout target.
