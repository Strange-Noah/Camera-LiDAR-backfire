#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/configs/metadata.env"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 required"'
mkdir -p "$PIP_CACHE_DIR" "$MPLCONFIGDIR"
if [[ ! -x "$METADATA_VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$METADATA_VENV"
fi
"$METADATA_VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,12)'
"$METADATA_VENV/bin/python" -m pip install pip==25.1.1
SPEC="$REPO_ROOT/environment/metadata/requirements.lock.txt"
[[ -f "$SPEC" ]] || SPEC="$REPO_ROOT/environment/metadata/requirements.in"
"$METADATA_VENV/bin/python" -m pip install -r "$SPEC"
"$METADATA_VENV/bin/python" -m pip check
mkdir -p "$PERSISTENT_ROOT/results/environment"
"$METADATA_VENV/bin/python" --version | tee "$PERSISTENT_ROOT/results/environment/python-version.txt"
"$METADATA_VENV/bin/python" -m pip freeze > "$PERSISTENT_ROOT/results/environment/pip-freeze.txt"
"$METADATA_VENV/bin/python" -c 'from nuscenes.nuscenes import NuScenes; print("Official devkit import OK")'
