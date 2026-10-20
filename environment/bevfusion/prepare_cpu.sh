#!/usr/bin/env bash
# Build and prepare without renting a GPU. Called by job_guard.py.
set -euo pipefail
job_dir="/workspace/results/deployment/bevfusion-${RUNPOD_POD_ID:?}"
mkdir -p "$job_dir"
exec > >(tee -a "$job_dir/job.log") 2>&1
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
mountpoint -q /workspace
du -sx --block-size=1 /workspace
sha256sum /workspace/checkpoints/bevfusion/*-det.pth
bash /opt/management/scripts/setup_experiment_user.sh bfprepare 20003
apt-get update
apt-get install -y --no-install-recommends python3.8-dev python3.8-venv build-essential \
  libgl1 libglib2.0-0 libopenmpi-dev openmpi-bin
python3.8 -m venv /opt/venvs/bevfusion
export PATH=/opt/venvs/bevfusion/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda MAX_JOBS=2
bash /opt/deployment/bevfusion/setup.sh
export PYTHONPATH=/opt/bevfusion
python -c 'import torch,mmcv,mmdet3d.ops; assert not torch.cuda.is_available(); print("CPU build and CUDA-extension imports passed", torch.__version__,mmcv.__version__)'
python /opt/deployment/bevfusion/check_models_cpu.py
runuser -u bfprepare -- /usr/local/bin/project-experiment-shell -c '
 export PATH=/opt/venvs/bevfusion/bin:$PATH PYTHONPATH=/opt/bevfusion
 python /opt/deployment/bevfusion/prepare_mini.py --output /workspace/staging/bfprepare/bevfusion-mini
'
mkdir -p /workspace/cache/bevfusion-runtime
mkdir -p /workspace/cache/bevfusion-runtime/mini-view
for name in samples sweeps maps v1.0-mini; do
  ln -sfn "/workspace/datasets/nuscenes/$name" "/workspace/cache/bevfusion-runtime/mini-view/$name"
done
cp /workspace/staging/bfprepare/bevfusion-mini/nuscenes_infos_*.pkl /workspace/cache/bevfusion-runtime/mini-view/
chmod 0755 /workspace/cache/bevfusion-runtime/mini-view
chmod 0644 /workspace/cache/bevfusion-runtime/mini-view/*.pkl
cp /opt/deployment/bevfusion/installed.freeze.txt /workspace/cache/bevfusion-runtime/installed.freeze.txt
cp /opt/deployment/bevfusion/upstream.json /workspace/cache/bevfusion-runtime/upstream.json
dpkg-query -W > /workspace/cache/bevfusion-runtime/os-packages.txt
tar -C / -czf /workspace/cache/bevfusion-runtime/runtime.tar.gz.part opt/venvs/bevfusion opt/bevfusion
mv /workspace/cache/bevfusion-runtime/runtime.tar.gz.part /workspace/cache/bevfusion-runtime/runtime.tar.gz
sha256sum /workspace/cache/bevfusion-runtime/runtime.tar.gz > /workspace/cache/bevfusion-runtime/runtime.sha256
echo 'CPU_PREPARATION_PASSED'
