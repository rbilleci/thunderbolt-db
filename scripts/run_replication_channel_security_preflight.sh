#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

workdir="$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-replication-channel-security.XXXXXX")"

cleanup() {
  rm -rf "$workdir"
}
trap cleanup EXIT

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'replication channel security preflight missing required command: %s\n' "$command_name" >&2
    return 1
  fi
}

require_command cargo
require_command openssl

cat >"$workdir/server.ext" <<'EOF'
basicConstraints=CA:FALSE
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:follower.local
EOF

cat >"$workdir/client.ext" <<'EOF'
basicConstraints=CA:FALSE
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=clientAuth
EOF

openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "$workdir/ca.key" \
  -out "$workdir/ca.crt" \
  -subj "/CN=gpu-db-replication-local-ca" \
  -days 1 >/dev/null 2>&1

openssl req -newkey rsa:2048 -nodes \
  -keyout "$workdir/server.key" \
  -out "$workdir/server.csr" \
  -subj "/CN=follower.local" >/dev/null 2>&1
openssl x509 -req \
  -in "$workdir/server.csr" \
  -CA "$workdir/ca.crt" \
  -CAkey "$workdir/ca.key" \
  -CAcreateserial \
  -out "$workdir/server.crt" \
  -days 1 \
  -extfile "$workdir/server.ext" >/dev/null 2>&1

openssl req -newkey rsa:2048 -nodes \
  -keyout "$workdir/client.key" \
  -out "$workdir/client.csr" \
  -subj "/CN=gpu-db-replication-leader" >/dev/null 2>&1
openssl x509 -req \
  -in "$workdir/client.csr" \
  -CA "$workdir/ca.crt" \
  -CAkey "$workdir/ca.key" \
  -CAcreateserial \
  -out "$workdir/client.crt" \
  -days 1 \
  -extfile "$workdir/client.ext" >/dev/null 2>&1

cargo run -q -p gpu_db_replication --example operational_channel_security_smoke -- \
  --ca-cert "$workdir/ca.crt" \
  --server-cert "$workdir/server.crt" \
  --server-key "$workdir/server.key" \
  --client-cert "$workdir/client.crt" \
  --client-key "$workdir/client.key" \
  --server-name follower.local
