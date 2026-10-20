#!/usr/bin/env bash
# Cold-image acceptance: same optional OS cache and runner as the GPU deployment.
set -euo pipefail
job_dir="/workspace/results/deployment/trainval-cold-cpu-${RUNPOD_POD_ID:?}"
mkdir -p "$job_dir"
exec > >(tee -a "$job_dir/validation.log") 2>&1
sha256sum -c /workspace/cache/bevfusion-runtime/runtime.sha256
tar -C / -xzf /workspace/cache/bevfusion-runtime/runtime.tar.gz
export PATH=/opt/venvs/bevfusion/bin:$PATH PYTHONPATH=/opt/bevfusion
python -c 'import torch,mmcv,mmdet3d.ops,lib2to3; assert not torch.cuda.is_available(); print("Cold cached OS/runtime imports passed")'
bash /opt/deployment/bevfusion/setup_gpu_user.sh
runuser -u bfbaseline -- env PROJECT_OUTPUT_MODE=local PATH="$PATH" PYTHONPATH=/opt/bevfusion \
 /usr/bin/python3 /usr/local/lib/camera-lidar/run_readonly_experiment.py -- \
 /opt/venvs/bevfusion/bin/python /opt/deployment/bevfusion/smoke.py --model fusion --limit 1 --preflight-only \
 --data-view /workspace/datasets/nuscenes-trainval-smoke --output /home/bfbaseline/trainval-cold-preflight
cp /home/bfbaseline/trainval-cold-preflight/acceptance.json "$job_dir/preflight.json"
echo TRAINVAL_COLD_CPU_ACCEPTANCE_PASSED
