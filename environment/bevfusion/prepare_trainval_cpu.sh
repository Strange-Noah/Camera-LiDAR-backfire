#!/usr/bin/env bash
set -euo pipefail
job_dir="/workspace/results/deployment/trainval-cpu-${RUNPOD_POD_ID:?}"
mkdir -p "$job_dir"
exec > >(tee -a "$job_dir/preparation.log") 2>&1
python3 -u /opt/deployment/bevfusion/cache_os_runtime.py
sha256sum -c /workspace/cache/bevfusion-runtime/runtime.sha256
tar -C / -xzf /workspace/cache/bevfusion-runtime/runtime.tar.gz
export PATH=/opt/venvs/bevfusion/bin:$PATH PYTHONPATH=/opt/bevfusion
python -m pip install --disable-pip-version-check 'ijson==3.3.0'
python -u /opt/deployment/bevfusion/prepare_trainval_subset.py select
python -u /opt/deployment/bevfusion/prepare_trainval_subset.py download
bash /opt/deployment/bevfusion/setup_gpu_user.sh
runuser -u bfbaseline -- env PROJECT_OUTPUT_MODE=local PATH="$PATH" PYTHONPATH=/opt/bevfusion \
 /usr/bin/python3 /usr/local/lib/camera-lidar/run_readonly_experiment.py -- \
 /opt/venvs/bevfusion/bin/python /opt/deployment/bevfusion/prepare_trainval_infos.py
cp /home/bfbaseline/trainval-infos/nuscenes_infos_val.pkl /workspace/datasets/nuscenes-trainval-smoke/
cp /home/bfbaseline/trainval-infos/input-check.json "$job_dir/"
runuser -u bfbaseline -- env PROJECT_OUTPUT_MODE=local PATH="$PATH" PYTHONPATH=/opt/bevfusion \
 /usr/bin/python3 /usr/local/lib/camera-lidar/run_readonly_experiment.py -- \
 /opt/venvs/bevfusion/bin/python /opt/deployment/bevfusion/smoke.py --model fusion --limit 1 --preflight-only \
 --data-view /workspace/datasets/nuscenes-trainval-smoke --output /home/bfbaseline/trainval-preflight
cp /home/bfbaseline/trainval-preflight/acceptance.json "$job_dir/preflight.json"
cp /workspace/datasets/nuscenes-trainval-smoke/subset-manifest.json "$job_dir/"
echo TRAINVAL_CPU_PREPARATION_PASSED
