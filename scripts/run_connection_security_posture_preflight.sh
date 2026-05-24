#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

require_line() {
  local file="$1"
  local pattern="$2"
  if ! grep -Fq "$pattern" "$file"; then
    printf 'connection security posture preflight missing evidence in %s: %s\n' "$file" "$pattern" >&2
    exit 1
  fi
}

require_line crates/protocol/src/lib.rs "PG_SSL_REQUEST_CODE"
require_line crates/protocol/src/lib.rs "PG_GSSENC_REQUEST_CODE"
require_line crates/protocol/src/bin/gpu-db-server.rs "StartupPacket::SslRequest | StartupPacket::GssEncRequest => stream.write_all(b\"N\")?"
require_line crates/protocol/src/bin/gpu-db-server.rs "write_authentication_ok(stream)?"
require_line crates/protocol/src/bin/gpu-db-server.rs "FrontendMessage::PasswordMessage(_) => \"password messages are not supported after startup\""
require_line crates/protocol/src/bin/gpu-db-server.rs "\"SASL authentication is not supported\""
require_line crates/protocol/src/bin/gpu-db-server.rs "fn write_authentication_ok(stream: &mut TcpStream) -> io::Result<()>"
require_line crates/protocol/src/bin/gpu-db-server.rs "write_message(stream, b'R', &0_i32.to_be_bytes())"
require_line crates/protocol/src/bin/gpu-db-server.rs "text_column(\"rolpassword\")"
require_line crates/protocol/src/bin/gpu-db-server.rs "None,"
require_line README.md "passwords/authentication"
require_line docs/compatibility/matrix.md "Passwords, memberships, authentication"
require_line docs/architecture/07-security-and-compliance.md "Current implementation status"
require_line docs/architecture/07-security-and-compliance.md "local/dev PostgreSQL compatibility endpoint currently uses trust-style"

printf 'connection_security_posture_preflight=passed\n'
printf 'connection_security_posture_preflight_scope=local_dev_trust_auth_no_tls_boundary\n'
printf 'connection_security_posture_preflight_ssl_request=declined_N\n'
printf 'connection_security_posture_preflight_gssenc_request=declined_N\n'
printf 'connection_security_posture_preflight_startup_authentication=authentication_ok_trust_style\n'
printf 'connection_security_posture_preflight_password_messages=unsupported_after_startup\n'
printf 'connection_security_posture_preflight_sasl=unsupported\n'
printf 'connection_security_posture_preflight_role_password_catalog=no_password_storage\n'
printf 'connection_security_posture_preflight_production_security_profile=not_supported\n'
printf 'connection_security_posture_preflight_gap_scram_sha_256=missing\n'
printf 'connection_security_posture_preflight_gap_password_authentication_storage=missing\n'
printf 'connection_security_posture_preflight_gap_tls_client_connections=missing\n'
printf 'connection_security_posture_preflight_gap_replication_mtls=missing\n'
printf 'connection_security_posture_preflight_gap_certificate_lifecycle=missing\n'
printf 'connection_security_posture_preflight_gap_audit_hash_chain=missing\n'
printf 'connection_security_posture_preflight_gap_row_level_security=missing\n'
printf 'connection_security_posture_preflight_gap_masking=missing\n'
printf 'connection_security_posture_preflight_trigger_full_auth_tls=named_production_security_profile_secret_storage_certificate_lifecycle_deployment_policy\n'
