#!/usr/bin/env bash
set -euo pipefail
out=/workspace/results/validation-baseline-preparation
mkdir -p "$out"
exec > >(tee -a "$out/legacy-cpu-preparation.log") 2>&1
python3 /opt/deployment/bevfusion/validation/progress_relay.py "$$" &
relay_pid=$!
trap 'kill "$relay_pid" 2>/dev/null || true' EXIT
sha256sum -c /workspace/cache/bevfusion-runtime/runtime.sha256
tar -C / -xzf /workspace/cache/bevfusion-runtime/runtime.tar.gz
export PATH=/opt/venvs/bevfusion/bin:$PATH PYTHONPATH=/opt/bevfusion MPLBACKEND=Agg
python -c 'import torch, mmcv, mmdet3d.ops; assert not torch.cuda.is_available(); print("CPU_RUNTIME_IMPORTS_PASSED")'
python -m pip install --disable-pip-version-check ijson==3.3.0
python /opt/deployment/bevfusion/check_models_cpu.py
cp /workspace/cache/bevfusion-runtime/model-loads-cpu.json "$out/model-loads-cpu.json"
python /opt/deployment/bevfusion/validation/prepare_maps.py
python -u /opt/deployment/bevfusion/validation/prepare_cpu.py
python -u /opt/deployment/bevfusion/validation/test_formatter_cpu.py
python -u /opt/deployment/bevfusion/validation/verify_payload_cpu.py \
 --manifest "$out/validation-file-manifest.json" --output "$out/validation-acquisition-report.json"
bash /opt/deployment/bevfusion/setup_gpu_user.sh
for model in fusion lidar; do
 runuser -u bfbaseline -- env PROJECT_OUTPUT_MODE=local PROJECT_CUDA_SANDBOX=1 \
  PATH="$PATH" PYTHONPATH=/opt/bevfusion \
  /usr/bin/python3 /usr/local/lib/camera-lidar/run_readonly_experiment.py -- \
  /opt/venvs/bevfusion/bin/python /opt/deployment/bevfusion/validation/preflight_official.py \
  --model "$model" \
  --output "/home/bfbaseline/validation-preflight-$model"
 cp "/home/bfbaseline/validation-preflight-$model/preflight.json" "$out/official-pipeline-partial-preflight-$model.json"
done
python /opt/deployment/bevfusion/validation/test_acceptance.py >"$out/acceptance-tests.log" 2>&1
python /opt/deployment/bevfusion/validation/baseline-acceptance.py \
 --manifest "$out/validation-file-manifest.json" --output "$out/baseline-results.json"
echo VALIDATION_CPU_PREPARATION_COMPLETE
