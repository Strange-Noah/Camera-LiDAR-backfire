#!/usr/bin/env bash
set -euo pipefail
out=/workspace/results/validation-baseline-preparation
exec > >(tee -a "$out/continue-cpu.log") 2>&1
export VALIDATION_PROGRESS_FILE="$out/prepare-progress.json"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
sha256sum -c /workspace/cache/bevfusion-runtime/runtime.sha256
tar -C / -xzf /workspace/cache/bevfusion-runtime/runtime.tar.gz
export PATH=/opt/venvs/bevfusion/bin:$PATH PYTHONPATH=/opt/bevfusion MPLBACKEND=Agg
python -c 'import torch,mmcv,mmdet3d.ops; assert not torch.cuda.is_available(); print("CPU_RUNTIME_IMPORTS_PASSED",flush=True)'
python -m pip install --disable-pip-version-check ijson==3.3.0
python -u /opt/deployment/bevfusion/validation/acquire_clean.py
python -u /opt/deployment/bevfusion/validation/verify_payload_cpu.py \
 --manifest "$out/validation-file-manifest.json" --output "$out/validation-acquisition-report.json"
python -u /opt/deployment/bevfusion/validation/prepare_cpu.py
python -u /opt/deployment/bevfusion/validation/test_formatter_cpu.py
python -u /opt/deployment/bevfusion/validation/test_real_evaluator_cpu.py
bash /opt/deployment/bevfusion/setup_gpu_user.sh
python3 /opt/deployment/bevfusion/validation/progress_relay.py "$$" &
relay=$!
trap 'kill "$relay" 2>/dev/null || true' EXIT
for model in fusion lidar; do
 runuser -u bfbaseline -- env PROJECT_OUTPUT_MODE=local PROJECT_CUDA_SANDBOX=1 \
  PATH="$PATH" PYTHONPATH=/opt/bevfusion \
  /usr/bin/python3 /usr/local/lib/camera-lidar/run_readonly_experiment.py -- \
  /opt/venvs/bevfusion/bin/python /opt/deployment/bevfusion/validation/preflight_official.py \
  --model "$model" --output "/home/bfbaseline/validation-full-preflight-$model"
 cp -r "/home/bfbaseline/validation-full-preflight-$model" "$out/"
done
python /opt/deployment/bevfusion/validation/test_acceptance.py >"$out/acceptance-tests.log" 2>&1
du -sx --block-size=1 /workspace > "$out/post-acquisition-volume-usage.txt"
echo FULL_CPU_PREPARATION_COMPLETE
