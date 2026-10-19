clea#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/configs/metadata.env"
mountpoint -q "$PERSISTENT_ROOT" || { echo "Refusing: persistent root is not a mount point" >&2; exit 1; }
FSTYPE="$(findmnt -n -o FSTYPE --target "$PERSISTENT_ROOT")"
case "$FSTYPE" in overlay|tmpfs|rootfs) echo "Refusing ephemeral filesystem: $FSTYPE" >&2; exit 1;; esac
mkdir -p "$NUSCENES_ROOT" "$PERSISTENT_ROOT/checkpoints" "$PERSISTENT_ROOT/results" "$PERSISTENT_ROOT/cache" "$PERSISTENT_ROOT/staging"
mkdir -p "$PERSISTENT_ROOT/results/infrastructure"
{ date -u +%FT%TZ; df -h "$PERSISTENT_ROOT"; findmnt --target "$PERSISTENT_ROOT"; mount; } > "$PERSISTENT_ROOT/results/infrastructure/mount-verification.txt"
cat "$PERSISTENT_ROOT/results/infrastructure/mount-verification.txt"
