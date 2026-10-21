#!/usr/bin/env bash
# Restore the CPU-built runtime, infer, and persist. No installation or evaluation.
set -euo pipefail
out=/workspace/results/validation-baseline-preparation
job="$out/gpu-${RUNPOD_POD_ID:?}"
mkdir -p "$job"
exec > >(tee -a "$job/job.log") 2>&1
sha256sum -c /workspace/cache/bevfusion-runtime/runtime.sha256
tar -C / -xzf /workspace/cache/bevfusion-runtime/runtime.tar.gz
bash /opt/deployment/bevfusion/setup_gpu_user.sh
export PATH=/opt/venvs/bevfusion/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/opt/bevfusion
nvidia-smi > "$job/nvidia-smi.txt"
python -c 'import torch,mmcv,mmdet3d.ops; assert torch.cuda.is_available(); print(torch.__version__,mmcv.__version__,torch.cuda.get_device_name(0))'
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > "$job/gpu-memory.csv" &
monitor=$!
python3 -u /opt/deployment/bevfusion/validation/persist_gpu.py > "$job/persistence.log" 2>&1 &
mirror=$!
finish() {
 status=$?
 touch /root/gpu-inference-done
 if ! wait "$mirror"; then
  status=1
  # One bounded salvage pass after a failed mirror; preserve any completed
  # local records before the guard deletes this Pod. Never loop on GPU time.
  python3 -u /opt/deployment/bevfusion/validation/persist_gpu.py >> "$job/persistence.log" 2>&1 || true
 fi
 kill "$monitor" 2>/dev/null || true
 sync
 trap - EXIT
 exit "$status"
}
trap finish EXIT
read -r -a models <<< "${VALIDATION_MODELS:-fusion lidar}"
test "${#models[@]}" -gt 0
for model in "${models[@]}"; do
 case "$model" in fusion|lidar) ;; *) echo "Invalid model selection" >&2; exit 2;; esac
done
for model in "${models[@]}"; do
 kill -0 "$mirror"
 runuser -u bfbaseline -- env -u RUNPOD_API_KEY PROJECT_OUTPUT_MODE=local PROJECT_CUDA_SANDBOX=1 \
  PATH="$PATH" PYTHONPATH=/opt/bevfusion \
  /usr/bin/python3 /usr/local/lib/camera-lidar/run_readonly_experiment.py -- \
  /opt/venvs/bevfusion/bin/python /opt/deployment/bevfusion/validation/infer_validation.py \
  --model "$model" --phase "${VALIDATION_PHASE:?}" \
  --output "/home/bfbaseline/results/$model-${VALIDATION_PHASE}-$RUNPOD_POD_ID" &
 inference=$!
 while kill -0 "$inference" 2>/dev/null; do
  if ! kill -0 "$mirror" 2>/dev/null; then
   kill -TERM "$inference" 2>/dev/null || true
   wait "$inference" || true
   echo 'Persistence worker failed; inference stopped' >&2
   exit 1
  fi
  sleep 3
 done
 wait "$inference"
done
echo VALIDATION_GPU_INFERENCE_COMPLETE
