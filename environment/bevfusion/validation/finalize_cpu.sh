#!/usr/bin/env bash
# CPU evaluation only; all inference-contributing GPUs must already be deleted.
set -euo pipefail
out=/workspace/results/validation-baseline-preparation
job="$out/evaluation-${RUNPOD_POD_ID:?}"
mkdir -p "$job"
exec > >(tee -a "$job/job.log") 2>&1
finish() {
 status=$?
 python3 -u /opt/deployment/bevfusion/validation/package_final_cpu.py --job-exit-code "$status" || true
 sync
 trap - EXIT
 exit "$status"
}
trap finish EXIT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
printf '{"stage":"restore_cpu_evaluation_runtime"}' > "$out/prepare-progress.json"
sha256sum -c /workspace/cache/bevfusion-runtime/runtime.sha256
tar -C / -xzf /workspace/cache/bevfusion-runtime/runtime.tar.gz
export PATH=/opt/venvs/bevfusion/bin:$PATH PYTHONPATH=/opt/bevfusion
python -c 'import torch,mmcv,mmdet3d.ops; assert not torch.cuda.is_available(); print("CPU_EVALUATION_RUNTIME_READY",flush=True)'
python -u /opt/deployment/bevfusion/validation/assemble_cpu.py \
 --cleanup-proof "$out/gpu-cleanup-proof.json"
sync
echo FULL_CPU_EVALUATION_AND_PACKAGE_COMPLETE
