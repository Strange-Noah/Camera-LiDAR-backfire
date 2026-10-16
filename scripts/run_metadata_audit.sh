#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/configs/metadata.env"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
exec "$METADATA_VENV/bin/python" "$REPO_ROOT/scripts/run_metadata_audit.py" "$@"
