#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

mvn -q -f "$SCRIPT_DIR/pom.xml" compile exec:java -Dexec.args="$REPO_ROOT"
