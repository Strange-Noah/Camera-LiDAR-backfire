#!/usr/bin/env bash
# GPU FUSE can report every owner as root. Keep experiment writes on local POSIX
# storage; a root supervisor copies results to the volume before releasing it.
set -euo pipefail
groupadd --gid 20002 bfbaseline
useradd --create-home --uid 20002 --gid bfbaseline --shell /bin/bash bfbaseline
install -d -m 0755 /usr/local/lib/camera-lidar
install -m 0755 /opt/management/scripts/run_readonly_experiment.py /usr/local/lib/camera-lidar/run_readonly_experiment.py
cat > /usr/local/bin/project-experiment-shell <<'SHELL'
#!/bin/bash
export PROJECT_OUTPUT_MODE=local
exec /usr/bin/python3 /usr/local/lib/camera-lidar/run_readonly_experiment.py -- /bin/bash "$@"
SHELL
chmod 0755 /usr/local/bin/project-experiment-shell
usermod --shell /usr/local/bin/project-experiment-shell bfbaseline
runuser -u bfbaseline -- mkdir -p /home/bfbaseline/results
runuser -u bfbaseline -- /usr/local/bin/project-experiment-shell -c 'test -r /workspace/datasets/nuscenes/v1.0-mini/scene.json'
