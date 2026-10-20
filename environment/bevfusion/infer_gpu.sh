#!/usr/bin/env bash
# No downloads, package builds or infos generation in the GPU phase.
set -euo pipefail
job_dir="/workspace/results/deployment/bevfusion-${RUNPOD_POD_ID:?}"
mkdir -p "$job_dir"
exec > >(tee -a "$job_dir/job.log") 2>&1
sha256sum -c /workspace/cache/bevfusion-runtime/runtime.sha256
tar -C / -xzf /workspace/cache/bevfusion-runtime/runtime.tar.gz
bash /opt/management/scripts/setup_experiment_user.sh bfbaseline 20002
export PATH=/opt/venvs/bevfusion/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/opt/bevfusion
nvidia-smi > "$job_dir/nvidia-smi.txt"
python -c 'import torch,mmcv,mmdet3d.ops; assert torch.cuda.is_available(); print(torch.__version__,mmcv.__version__,torch.cuda.get_device_name(0))'
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2 > "$job_dir/gpu-memory.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
runuser -u bfbaseline -- /usr/local/bin/project-experiment-shell -c '
 export PATH=/opt/venvs/bevfusion/bin:$PATH PYTHONPATH=/opt/bevfusion
 for count in 8; do
  for model in fusion lidar; do
   python /opt/deployment/bevfusion/smoke.py --model "$model" --limit "$count" --raw-only \
    --data-view /workspace/cache/bevfusion-runtime/mini-view \
    --output /workspace/results/bfbaseline/bevfusion-'"$RUNPOD_POD_ID"'-"$model"-"$count" || exit $?
  done
 done
'
echo 'GPU_SMOKE_TESTS_PASSED'
