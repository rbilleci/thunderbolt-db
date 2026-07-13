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
require_line crates/protocol/src/bin/gpu-db-server.rs "SecurityConfig::LocalDev"
require_line crates/protocol/src/bin/gpu-db-server.rs "SecurityConfig::Production"
require_line crates/protocol/src/bin/gpu-db-server.rs "production security profile requires TLS"
require_line crates/protocol/src/bin/gpu-db-server.rs "ScramCredential::Verifier"
require_line crates/protocol/src/bin/gpu-db-server.rs "production SCRAM verifier must start with SCRAM-SHA-256$"
require_line crates/protocol/src/bin/gpu-db-server.rs "local/test --auth-password"
require_line crates/protocol/src/bin/gpu-db-server.rs "write_authentication_sasl(stream, &[\"SCRAM-SHA-256\"])?"
require_line crates/protocol/src/bin/gpu-db-server.rs "write_authentication_ok(stream)"
require_line crates/protocol/src/bin/gpu-db-server.rs "FrontendMessage::PasswordMessage(_) => \"password messages are not supported after startup\""
require_line crates/protocol/src/bin/gpu-db-server.rs "\"SASL authentication is not supported\""
require_line crates/protocol/src/bin/gpu-db-server/backend_adapter.rs "fn write_authentication_ok(stream: &mut dyn ReadWrite) -> io::Result<()>"
require_line crates/protocol/src/lib.rs "self.message(b'R', &0_i32.to_be_bytes())"
require_line crates/protocol/src/bin/gpu-db-server.rs "text_column(\"rolpassword\")"
require_line docs/STATUS.md "opt-in production security profile requires TLS plus a SCRAM-SHA-256"

cargo build -p gpu_db_protocol --bin gpu-db-server >/dev/null

if ./target/debug/gpu-db-server --security-profile production >/tmp/gpu-db-security-missing.out 2>/tmp/gpu-db-security-missing.err; then
  printf 'production profile accepted incomplete config\n' >&2
  exit 1
fi
if ! grep -Fq "production security profile requires --tls-cert" /tmp/gpu-db-security-missing.err; then
  printf 'production profile did not report missing TLS material\n' >&2
  cat /tmp/gpu-db-security-missing.err >&2
  exit 1
fi

if ./target/debug/gpu-db-server \
  --security-profile production \
  --tls-cert /tmp/missing.crt \
  --tls-key /tmp/missing.key \
  --auth-user gpudb \
  --auth-scram-verifier not-a-verifier \
  >/tmp/gpu-db-security-malformed.out 2>/tmp/gpu-db-security-malformed.err; then
  printf 'production profile accepted malformed SCRAM verifier\n' >&2
  exit 1
fi
if ! grep -Fq "production SCRAM verifier must start with SCRAM-SHA-256$" /tmp/gpu-db-security-malformed.err; then
  printf 'production profile did not report malformed SCRAM verifier\n' >&2
  cat /tmp/gpu-db-security-malformed.err >&2
  exit 1
fi

tmp="$(mktemp -d)"
server_pid=""
cleanup() {
  if [ -n "${server_pid:-}" ]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT

openssl req -new -x509 -nodes -subj '/CN=localhost' -days 1 \
  -keyout "$tmp/server.key" -out "$tmp/server.crt" >/dev/null 2>&1
chmod 600 "$tmp/server.key"

python3 - <<'PY' >"$tmp/scram.verifier"
import base64
import hashlib
import hmac

password = b"secret"
salt = b"gpu-db-production-verifier-preflight-v1"
iterations = 4096
salted = hashlib.pbkdf2_hmac("sha256", password, salt, iterations)
client_key = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
stored_key = hashlib.sha256(client_key).digest()
server_key = hmac.new(salted, b"Server Key", hashlib.sha256).digest()
print(
    "SCRAM-SHA-256${}:{}${}:{}".format(
        iterations,
        base64.b64encode(salt).decode(),
        base64.b64encode(stored_key).decode(),
        base64.b64encode(server_key).decode(),
    )
)
PY

if ./target/debug/gpu-db-server \
  --security-profile production \
  --tls-cert "$tmp/server.crt" \
  --tls-key "$tmp/server.key" \
  --auth-user gpudb \
  --auth-password secret \
  --auth-scram-verifier-file "$tmp/scram.verifier" \
  >"$tmp/conflict.out" 2>"$tmp/conflict.err"; then
  printf 'production profile accepted conflicting plaintext/verifier inputs\n' >&2
  exit 1
fi
if ! grep -Fq "only one credential source" "$tmp/conflict.err"; then
  printf 'production profile did not report conflicting credential inputs\n' >&2
  cat "$tmp/conflict.err" >&2
  exit 1
fi

port="$(
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"

./target/debug/gpu-db-server \
  --listen "127.0.0.1:${port}" \
  --shared-catalog \
  --security-profile production \
  --tls-cert "$tmp/server.crt" \
  --tls-key "$tmp/server.key" \
  --auth-user gpudb \
  --auth-scram-verifier-file "$tmp/scram.verifier" \
  >"$tmp/server.out" 2>"$tmp/server.err" &
server_pid="$!"

for _ in $(seq 1 100); do
  if (echo >"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1; then
    break
  fi
  sleep 0.05
done

PGPASSWORD=secret psql "host=127.0.0.1 port=${port} user=gpudb dbname=postgres sslmode=require" \
  -Atc "create table prod_auth (id int4); insert into prod_auth (id) values (7); select id from prod_auth" \
  >"$tmp/valid.out" 2>"$tmp/valid.err"
if ! grep -Fxq "7" "$tmp/valid.out"; then
  printf 'production TLS+SCRAM valid path did not return expected row\n' >&2
  cat "$tmp/valid.out" >&2
  cat "$tmp/valid.err" >&2
  exit 1
fi

if PGPASSWORD=wrong psql "host=127.0.0.1 port=${port} user=gpudb dbname=postgres sslmode=require" \
  -Atc "select id from prod_auth" >"$tmp/invalid.out" 2>"$tmp/invalid.err"; then
  printf 'production SCRAM accepted invalid password\n' >&2
  exit 1
fi
if ! grep -Fq "password authentication failed" "$tmp/invalid.err"; then
  printf 'production SCRAM invalid password did not fail with expected error\n' >&2
  cat "$tmp/invalid.err" >&2
  exit 1
fi

PGPASSWORD=secret psql "host=127.0.0.1 port=${port} user=gpudb dbname=postgres sslmode=require" \
  -Atc "select id from prod_auth" >"$tmp/recovery.out" 2>"$tmp/recovery.err"
if ! grep -Fxq "7" "$tmp/recovery.out"; then
  printf 'production server did not recover after invalid password attempt\n' >&2
  cat "$tmp/recovery.out" >&2
  cat "$tmp/recovery.err" >&2
  exit 1
fi

if PGPASSWORD=secret psql "host=127.0.0.1 port=${port} user=gpudb dbname=postgres sslmode=disable" \
  -Atc "select id from prod_auth" >"$tmp/notls.out" 2>"$tmp/notls.err"; then
  printf 'production profile accepted non-TLS client\n' >&2
  exit 1
fi
if ! grep -Fq "production security profile requires TLS" "$tmp/notls.err"; then
  printf 'production non-TLS client did not fail with expected error\n' >&2
  cat "$tmp/notls.err" >&2
  exit 1
fi

printf 'connection_security_posture_preflight=passed\n'
printf 'connection_security_posture_preflight_scope=local_dev_trust_auth_no_tls_plus_opt_in_production_tls_scram\n'
printf 'connection_security_posture_preflight_local_dev_profile=trust_auth_no_tls_supported\n'
printf 'connection_security_posture_preflight_production_profile_v1=passed\n'
printf 'connection_security_posture_preflight_production_config_validation=passed\n'
printf 'connection_security_posture_preflight_production_scram_verifier_config=passed\n'
printf 'connection_security_posture_preflight_production_plaintext_password_conflict_rejection=passed\n'
printf 'connection_security_posture_preflight_production_tls_required=passed\n'
printf 'connection_security_posture_preflight_production_scram_sha_256_valid_password=passed\n'
printf 'connection_security_posture_preflight_production_scram_sha_256_invalid_password=passed\n'
printf 'connection_security_posture_preflight_production_recovery_after_invalid_password=passed\n'
printf 'connection_security_posture_preflight_non_claim_mtls=not_supported\n'
printf 'connection_security_posture_preflight_non_claim_enterprise_identity=not_supported\n'
printf 'connection_security_posture_preflight_non_claim_kms_hsm_secret_manager=not_supported\n'
printf 'connection_security_posture_preflight_non_claim_certificate_rotation=not_supported\n'
printf 'connection_security_posture_preflight_non_claim_audit_hash_chain=not_supported\n'
printf 'connection_security_posture_preflight_non_claim_row_level_security=not_supported\n'
printf 'connection_security_posture_preflight_non_claim_masking=not_supported\n'
