#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/configs/metadata.env"
source "$REPO_ROOT/scripts/require_experiment_user.sh"
mkdir -p "$SMOKE_OUTPUT_DIR"
"${EXPERIMENT_LAUNCHER[@]}" "$METADATA_VENV/bin/python" "$REPO_ROOT/scripts/verify_nuscenes.py" --root "$NUSCENES_ROOT" --output "$SMOKE_OUTPUT_DIR" "$@" 2>&1 | tee "$SMOKE_OUTPUT_DIR/smoke_test.log"
