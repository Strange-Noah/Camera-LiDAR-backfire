#!/usr/bin/env bash
set -euo pipefail
cleanup() {
 while true; do
  curl -sS --max-time 20 -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
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
id -u sshd >/dev/null 2>&1 || useradd --system --home-dir /run/sshd --shell /usr/sbin/nologin sshd
ssh-keygen -A
/usr/sbin/sshd
umask 077
printf '%s' "$RUNPOD_API_KEY" > /root/.bevfusion-pod-key
umask 022
cp -a /workspace/cache/validation-baseline-preparation/management /opt/management
cp -r /opt/management/environment/bevfusion /opt/deployment/
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
out=/workspace/results/validation-baseline-preparation
job="$out/gpu-$RUNPOD_POD_ID"
mkdir -p "$job"
printf '{"stage":"runtime_restore"}' > /root/gpu-progress.json
# Data loading writes real progress locally; root relays only changed counters.
python3 -u /opt/deployment/bevfusion/validation/gpu_progress_relay.py > "$job/progress-relay.log" 2>&1 &
relay=$!
python3 -u /opt/deployment/bevfusion/job_guard.py --pod-id "$RUNPOD_POD_ID" \
 --key-file /root/.bevfusion-pod-key --state "$job/job-state.json" \
 --progress-file /root/gpu-progress.json --stall-seconds 900 --grace-seconds 0 -- \
 bash /opt/deployment/bevfusion/validation/infer_gpu.sh > "$job/guard.log" 2>&1
kill "$relay" 2>/dev/null || true
