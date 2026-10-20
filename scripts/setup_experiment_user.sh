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
  if getent group "$login" >/dev/null; then
    [[ "$(getent group "$login" | cut -d: -f3)" = "$member_uid" ]]
  else
    groupadd --gid "$member_uid" "$login"
  fi
  useradd --create-home --uid "$member_uid" --gid "$login" --shell /bin/bash "$login"
fi
member_gid="$(id -g "$login")"
mapped_uid=$(awk -v id="$member_uid" '$1<=id && id<$1+$3 {print $2+id-$1}' /proc/self/uid_map)
mapped_gid=$(awk -v id="$member_gid" '$1<=id && id<$1+$3 {print $2+id-$1}' /proc/self/gid_map)
[[ "$(id -G "$login")" = "$member_gid" ]] || { echo 'Unexpected supplementary groups; review before use' >&2; exit 2; }
# Refuse takeover of another member's existing directory.
for top in results cache staging checkpoints; do
  target="/workspace/$top/$login"
  [[ ! -L "$target" ]]
  if [[ -e "$target" ]]; then
    owner=$(stat -c %u "$target")
    [[ "$owner" = "$member_uid" || "$owner" = "$mapped_uid" ]] || { echo "Unexpected owner $owner for $target (member $member_uid, mapped $mapped_uid)" >&2; exit 1; }
  else
    # Some GPU-host FUSE mounts reject chown even for root. Create as the
    # intended member instead, then verify ownership rather than weakening it.
    runuser -u "$login" -- mkdir -m 0750 -- "$target"
  fi
  owner=$(stat -c %u "$target"); group=$(stat -c %g "$target")
  [[ "$owner" = "$member_uid" || "$owner" = "$mapped_uid" ]] || { echo "Owner mapping failed: $owner for $target" >&2; exit 1; }
  [[ "$group" = "$member_gid" || "$group" = "$mapped_gid" ]] || { echo "Group mapping failed: $group for $target (expected $member_gid or $mapped_gid)" >&2; exit 1; }
  [[ "$(stat -c %a "$target")" = 750 ]]
  runuser -u "$login" -- test -w "$target"
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
