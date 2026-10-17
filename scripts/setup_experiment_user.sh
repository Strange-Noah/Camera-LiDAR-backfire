#!/usr/bin/env bash
# Root bootstrap on EACH Pod. Usage: script LOGIN UID [authorized_keys_file]
set -euo pipefail
[[ "$EUID" -eq 0 && "$#" -ge 2 && "$#" -le 3 ]] || { echo "Usage (root): $0 LOGIN UID [authorized_keys_file]" >&2; exit 2; }
login="$1"; member_uid="$2"
[[ "$login" =~ ^[a-z][a-z0-9_-]{0,30}$ && "$member_uid" =~ ^[0-9]+$ ]] || exit 2
[[ "$member_uid" -ge 20001 && "$member_uid" -le 29999 ]] || exit 2
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$script_dir/run_readonly_experiment.py" ]]
mountpoint -q /workspace
[[ -f /workspace/results/storage_permissions/protection_report.json ]]
if getent passwd "$login" >/dev/null; then
  [[ "$(id -u "$login")" = "$member_uid" ]]
else
  if getent passwd "$member_uid" >/dev/null; then echo 'UID already allocated' >&2; exit 2; fi
  useradd --create-home --uid "$member_uid" --user-group --shell /bin/bash "$login"
fi
member_gid="$(id -g "$login")"
[[ "$(id -G "$login")" = "$member_gid" ]] || { echo 'Unexpected supplementary groups; review before use' >&2; exit 2; }
# Refuse takeover of another member's existing directory.
for top in results cache staging checkpoints; do
  target="/workspace/$top/$login"
  [[ ! -L "$target" ]]
  if [[ -e "$target" ]]; then
    [[ "$(stat -c %u "$target")" = "$member_uid" ]]
  else
    install -d -o "$member_uid" -g "$member_gid" -m 0750 "$target"
  fi
done
if [[ "$#" -eq 3 ]]; then
  [[ -f "$3" ]]
  install -d -o "$member_uid" -g "$member_gid" -m 0700 "/home/$login/.ssh"
  install -o "$member_uid" -g "$member_gid" -m 0600 "$3" "/home/$login/.ssh/authorized_keys"
fi
install -d -m 0755 /usr/local/lib/camera-lidar
install -m 0755 "$script_dir/run_readonly_experiment.py" /usr/local/lib/camera-lidar/run_readonly_experiment.py
cat > /usr/local/bin/project-experiment-shell <<'SHELL'
#!/bin/bash
exec /usr/bin/python3 /usr/local/lib/camera-lidar/run_readonly_experiment.py -- /bin/bash "$@"
SHELL
chmod 0755 /usr/local/bin/project-experiment-shell
usermod --shell /usr/local/bin/project-experiment-shell "$login"
runuser -u "$login" -- test -r /workspace/datasets/nuscenes/v1.0-mini/scene.json
if runuser -u "$login" -- test -w /workspace/datasets/nuscenes; then
  echo 'Dataset still writable; refusing success' >&2; exit 1
fi
printf 'Use direct SSH as %s (not root). UID=%s; no supplementary groups.\n' "$login" "$member_uid"
