#!/usr/bin/env bash
# No downloads, package builds or infos generation in the GPU phase.
set -euo pipefail
job_dir="/workspace/results/deployment/bevfusion-${RUNPOD_POD_ID:?}"
mkdir -p "$job_dir"
exec > >(tee -a "$job_dir/job.log") 2>&1
sha256sum -c /workspace/cache/bevfusion-runtime/runtime.sha256
tar -C / -xzf /workspace/cache/bevfusion-runtime/runtime.tar.gz
bash /opt/deployment/bevfusion/setup_gpu_user.sh
export PATH=/opt/venvs/bevfusion/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/opt/bevfusion
nvidia-smi > "$job_dir/nvidia-smi.txt"
python -c 'import torch,mmcv,mmdet3d.ops; assert torch.cuda.is_available(); print(torch.__version__,mmcv.__version__,torch.cuda.get_device_name(0))'
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2 > "$job_dir/gpu-memory.csv" &
monitor_pid=$!
persist_results() {
  mkdir -p "$job_dir/predictions"
  cp -r --no-preserve=ownership /home/bfbaseline/results/. "$job_dir/predictions/"
}
finalize() {
  status=$?
  kill "$monitor_pid" 2>/dev/null || true
  persist_results || { echo 'Result persistence failed' >&2; status=1; }
  trap - EXIT
  exit "$status"
}
trap finalize EXIT
models=${BEVFUSION_MODELS:-"fusion lidar"}
limit=${BEVFUSION_LIMIT:-8}
data_view=${BEVFUSION_DATA_VIEW:-/workspace/cache/bevfusion-runtime/mini-view}
for model in $models; do
  [[ "$model" = fusion || "$model" = lidar ]] || { echo "Invalid model selection" >&2; exit 1; }
  runuser -u bfbaseline -- env PROJECT_OUTPUT_MODE=local PROJECT_CUDA_SANDBOX=1 \
    BEVFUSION_PROGRESS_FILE=/home/bfbaseline/results/progress.json \
    PATH="$PATH" PYTHONPATH=/opt/bevfusion \
    /usr/bin/python3 /usr/local/lib/camera-lidar/run_readonly_experiment.py -- \
    /opt/venvs/bevfusion/bin/python /opt/deployment/bevfusion/smoke.py --model "$model" --limit "$limit" --raw-only \
    --data-view "$data_view" \
    --output /home/bfbaseline/results/"$model"-"$limit"
done
echo 'GPU_SMOKE_TESTS_PASSED'
