#!/usr/bin/env bash
# CPU-only wrapper around the tested runtime restore, guard and handoff stages.
# Image must be the existing pinned CUDA 11.3.1 Ubuntu 20.04 image, CPU only.
set -euo pipefail
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
sha256sum -c /workspace/cache/bevfusion-runtime/os-runtime.sha256
tar -C / -xzf /workspace/cache/bevfusion-runtime/os-runtime.tar.gz
mkdir -p /run/sshd /root/.ssh /opt/deployment
chmod 700 /root/.ssh
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
if ! id -u sshd >/dev/null 2>&1; then
 useradd --system --home-dir /run/sshd --shell /usr/sbin/nologin sshd
fi
ssh-keygen -A
/usr/sbin/sshd
umask 077
printf '%s' "$RUNPOD_API_KEY" > /root/.bevfusion-pod-key
umask 022
cp -a /workspace/cache/validation-baseline-preparation/management /opt/management
cp -r /opt/management/environment/bevfusion /opt/deployment/
out=/workspace/results/validation-baseline-preparation
mkdir -p "$out"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python3 -u /opt/deployment/bevfusion/job_guard.py \
 --pod-id "$RUNPOD_POD_ID" --key-file /root/.bevfusion-pod-key \
 --state "$out/prepare-${RUNPOD_POD_ID}-job-state.json" --progress-file "$out/prepare-progress.json" \
 --stall-seconds 900 --grace-seconds 30 -- \
 python3 -u /opt/deployment/bevfusion/validation/cpu_handoff.py >"$out/prepare-${RUNPOD_POD_ID}-guard.log" 2>&1
