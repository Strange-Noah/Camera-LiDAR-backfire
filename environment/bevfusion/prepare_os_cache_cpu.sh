#!/usr/bin/env bash
set -euo pipefail
python3 -u /opt/deployment/bevfusion/cache_os_runtime.py
bash /opt/deployment/bevfusion/validate_trainval_cpu.sh
export PATH=/opt/venvs/bevfusion/bin:$PATH PYTHONPATH=/opt/bevfusion
runuser -u bfbaseline -- env PROJECT_OUTPUT_MODE=local MPLBACKEND=Agg PATH="$PATH" PYTHONPATH=/opt/bevfusion \
 /usr/bin/python3 /usr/local/lib/camera-lidar/run_readonly_experiment.py -- \
 /opt/venvs/bevfusion/bin/python /opt/deployment/bevfusion/render_trainval_cpu.py
mkdir -p "/workspace/results/deployment/trainval-inspection-${RUNPOD_POD_ID:?}"
cp /home/bfbaseline/trainval-figures/*.png "/workspace/results/deployment/trainval-inspection-$RUNPOD_POD_ID/"
