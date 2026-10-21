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
python3 - <<'PY'
from pathlib import Path
import json
root=Path('/workspace/results/validation-baseline-preparation')
tokens={json.loads(line)['token'] for line in (root/'validation-full-preflight-fusion/input-selection.jsonl').read_text().splitlines()}
assert len(tokens)==6019
coverage={}
for model in ['fusion','lidar']:
    actual={p.parent.name for p in (root/'clean-predictions'/model).glob('*/complete.json')}
    coverage[model]=dict(expected=6019,complete=len(actual),missing=sorted(tokens-actual),extra=sorted(actual-tokens))
(root/'prediction-coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
print('DURABLE_RAW_COVERAGE',json.dumps(coverage),flush=True)
PY
python -u /opt/deployment/bevfusion/validation/assemble_cpu.py \
 --cleanup-proof "$out/gpu-cleanup-proof.json"
sync
echo FULL_CPU_EVALUATION_AND_PACKAGE_COMPLETE
