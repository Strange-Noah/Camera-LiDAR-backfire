#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/configs/metadata.env"
source "$REPO_ROOT/scripts/require_experiment_user.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
exec "${EXPERIMENT_LAUNCHER[@]}" "$METADATA_VENV/bin/python" "$REPO_ROOT/scripts/run_metadata_audit.py" "$@"
