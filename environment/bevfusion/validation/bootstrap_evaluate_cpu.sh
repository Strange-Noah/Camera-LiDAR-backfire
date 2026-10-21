#!/usr/bin/env bash
# Submit this as the CPU container command after account-level GPU deletion.
set -euo pipefail
cleanup() {
 sleep 30
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
staged=/workspace/cache/validation-baseline-preparation/cpu-evaluation-source
sha256sum -c "$staged/files.sha256"
cp "$staged/finalize_cpu.sh" "$staged/package_final_cpu.py" /opt/deployment/bevfusion/validation/
out=/workspace/results/validation-baseline-preparation
job="$out/evaluation-$RUNPOD_POD_ID"
mkdir -p "$job"
printf '%s\n' "${GPU_CLEANUP_PROOF_JSON:?Account-level cleanup proof required}" > "$out/gpu-cleanup-proof.json"
printf '{"stage":"cpu_evaluation_start"}' > "$out/prepare-progress.json"
python3 -u /opt/deployment/bevfusion/job_guard.py --pod-id "$RUNPOD_POD_ID" \
 --key-file /root/.bevfusion-pod-key --state "$job/job-state.json" \
 --progress-file "$out/prepare-progress.json" --stall-seconds 900 --grace-seconds 30 -- \
 bash /opt/deployment/bevfusion/validation/finalize_cpu.sh > "$job/guard.log" 2>&1
