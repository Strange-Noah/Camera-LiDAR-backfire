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
echo 'MINIMAL_CPU_RUNTIME_VALIDATION_PASSED'
