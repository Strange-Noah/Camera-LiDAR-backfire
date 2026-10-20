#!/usr/bin/env bash
set -euo pipefail
job_dir="/workspace/results/deployment/bevfusion-${RUNPOD_POD_ID:?}"
mkdir -p "$job_dir"
exec > >(tee -a "$job_dir/validation.log") 2>&1
sha256sum -c /workspace/cache/bevfusion-runtime/runtime.sha256
tar -C / -xzf /workspace/cache/bevfusion-runtime/runtime.tar.gz
export PATH=/opt/venvs/bevfusion/bin:$PATH PYTHONPATH=/opt/bevfusion
python -c 'import torch,mmcv,mmdet3d.ops,lib2to3; assert not torch.cuda.is_available(); print("Minimal runtime imports passed")'
python /opt/deployment/bevfusion/check_models_cpu.py
bash /opt/deployment/bevfusion/setup_gpu_user.sh
runuser -u bfbaseline -- /usr/local/bin/project-experiment-shell -c '
 export PATH=/opt/venvs/bevfusion/bin:$PATH PYTHONPATH=/opt/bevfusion
 python /opt/deployment/bevfusion/validate_inputs_cpu.py
'
cp /home/bfbaseline/input-validation.json /workspace/cache/bevfusion-runtime/input-validation.json
for model in fusion lidar; do
  runuser -u bfbaseline -- env PROJECT_OUTPUT_MODE=local PROJECT_CUDA_SANDBOX=1 \
    PATH="$PATH" PYTHONPATH=/opt/bevfusion \
    /usr/bin/python3 /usr/local/lib/camera-lidar/run_readonly_experiment.py -- \
    /opt/venvs/bevfusion/bin/python /opt/deployment/bevfusion/smoke.py \
      --model "$model" --limit 8 --preflight-only \
      --data-view /workspace/cache/bevfusion-runtime/mini-view \
      --output "/home/bfbaseline/preflight-$model"
  cp "/home/bfbaseline/preflight-$model/acceptance.json" "/workspace/cache/bevfusion-runtime/runner-preflight-$model.json"
done
echo 'MINIMAL_CPU_RUNTIME_VALIDATION_PASSED'
