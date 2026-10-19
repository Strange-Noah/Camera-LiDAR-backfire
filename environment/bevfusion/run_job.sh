#!/usr/bin/env bash
set -euo pipefail
job_dir="/workspace/results/deployment/bevfusion-${RUNPOD_POD_ID:?}"
mkdir -p "$job_dir"
exec > >(tee -a "$job_dir/job.log") 2>&1
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
mountpoint -q /workspace
used=$(du -sx --block-size=1 /workspace | awk '{print $1}')
[[ "$used" -lt 35000000000 ]] || { echo 'Insufficient 50GB quota headroom'; exit 1; }
printf 'Volume used bytes: %s\n' "$used"
nvidia-smi > "$job_dir/nvidia-smi.txt"
df -h /dev/shm
python3 /opt/deployment/bevfusion/download_checkpoints.py
cp /workspace/checkpoints/bevfusion/manifest.json "$job_dir/checkpoint-manifest.json"
apt-get update
apt-get install -y --no-install-recommends python3.8-dev python3.8-venv build-essential \
  libgl1 libglib2.0-0 libopenmpi-dev openmpi-bin
python3.8 -m venv /opt/venvs/bevfusion
export PATH=/opt/venvs/bevfusion/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
bash /opt/deployment/bevfusion/setup.sh
cp /opt/deployment/bevfusion/installed.freeze.txt "$job_dir/installed.freeze.txt"
nvcc --version > "$job_dir/nvcc.txt"
dpkg-query -W > "$job_dir/os-packages.txt"
git clone --branch feat/bevfusion-mini-deployment https://github.com/Strange-Noah/Camera-LiDAR-backfire.git /root/Camera-LiDAR-backfire
bash /root/Camera-LiDAR-backfire/scripts/setup_experiment_user.sh researcher 20001
chmod -R a+rX /opt/deployment/bevfusion
export PYTHONPATH=/opt/bevfusion
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2 > "$job_dir/gpu-memory.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
runuser -u researcher -- /usr/local/bin/project-experiment-shell -c '
  export PATH=/opt/venvs/bevfusion/bin:$PATH PYTHONPATH=/opt/bevfusion
  python /opt/deployment/bevfusion/prepare_mini.py --output /workspace/staging/researcher/bevfusion-mini
  for count in 1 8; do
    for model in fusion lidar; do
      python /opt/deployment/bevfusion/smoke.py --model "$model" --limit "$count" \
        --data-view /workspace/staging/researcher/bevfusion-mini \
        --output /workspace/results/researcher/bevfusion-'"$RUNPOD_POD_ID"'-"$model"-"$count" || exit $?
    done
  done
'
echo 'Both mini smoke tests completed'
