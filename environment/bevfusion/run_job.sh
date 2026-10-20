#!/usr/bin/env bash
# Compatibility entrypoint: preparation belongs in prepare_cpu.sh.
set -euo pipefail
exec bash "$(dirname "$0")/infer_gpu.sh"
