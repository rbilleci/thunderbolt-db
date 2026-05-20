#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
created_node_modules=0

if [[ ! -d "$SCRIPT_DIR/node_modules" ]]; then
  created_node_modules=1
fi

cleanup() {
  if [[ "$created_node_modules" == "1" ]]; then
    rm -rf "$SCRIPT_DIR/node_modules"
  fi
}
trap cleanup EXIT

npm --prefix "$SCRIPT_DIR" ci --silent
npm --prefix "$SCRIPT_DIR" test
