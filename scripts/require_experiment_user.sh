#!/usr/bin/env bash
# Source AFTER configs/metadata.env. Protect standard experiment entrypoints from root mistakes.
if [[ "${EUID:-$(id -u)}" -eq 0 && -f "$PERSISTENT_ROOT/results/storage_permissions/protection_report.json" ]]; then
  echo 'Dataset protection is enabled. Run experiments from your non-root member login, not root.' >&2
  exit 1
fi

EXPERIMENT_LAUNCHER=()
if [[ -f "$PERSISTENT_ROOT/results/storage_permissions/protection_report.json" ]]; then
  EXPERIMENT_LAUNCHER=(python3 "$REPO_ROOT/scripts/run_readonly_experiment.py" --)
fi
