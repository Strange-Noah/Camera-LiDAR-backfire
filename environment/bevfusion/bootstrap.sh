#!/usr/bin/env bash
set -euo pipefail
# Run as the image entrypoint, before any interactive connection. The startup
# environment includes the Runpod-provided scoped key; SSH sessions do not.
export DEBIAN_FRONTEND=noninteractive PIP_CACHE_DIR=/workspace/cache/bevfusion-build
cleanup() {
  sleep 30
  while true; do
    curl --silent --show-error --max-time 20 --request DELETE \
      --header "Authorization: Bearer $RUNPOD_API_KEY" \
      "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" >/dev/null || true
    sleep 15
  done
}
trap cleanup EXIT
apt-get update
apt-get install -y --no-install-recommends openssh-server ca-certificates curl python3 git \
  libgl1 libglib2.0-0 libopenmpi3
mkdir -p /run/sshd /root/.ssh /opt/deployment
chmod 700 /root/.ssh
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
/usr/sbin/sshd
umask 077
printf '%s' "$RUNPOD_API_KEY" > /root/.bevfusion-pod-key
umask 022
git -c http.lowSpeedLimit=1 -c http.lowSpeedTime=90 clone --depth 1 \
  --branch feat/bevfusion-mini-deployment \
  https://github.com/Strange-Noah/Camera-LiDAR-backfire.git /opt/management
cp -r /opt/management/environment/bevfusion /opt/deployment/
job_dir="/workspace/results/deployment/bevfusion-$RUNPOD_POD_ID"
mkdir -p "$job_dir"
python3 -u /opt/deployment/bevfusion/job_guard.py --pod-id "$RUNPOD_POD_ID" \
  --key-file /root/.bevfusion-pod-key --state "$job_dir/job-state.json" \
  -- bash "/opt/deployment/bevfusion/${BEVFUSION_PHASE:-infer_gpu}.sh" > "$job_dir/guard.log" 2>&1
