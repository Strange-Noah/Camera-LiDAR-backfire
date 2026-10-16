# Metadata environment

Target: Linux x86_64, Ubuntu 24.04, CPython 3.12 (observed 3.12.3).
Base image: `runpod/base:1.0.2-ubuntu2404`, read from Runpod's official CPU catalog.

`requirements.in` documents deliberately chosen top-level versions.
`requirements.lock.txt` records every installed application dependency from the Pod.
`setup_metadata_env.sh` pins pip to 25.1.1 and installs the full lock when present.
The venv does not inherit system site-packages. No PyTorch, MMCV, BEVFusion or
MultiCorrupt is installed by these specifications.

From the repository root on a prepared Pod:

```bash
./scripts/setup_metadata_env.sh
./scripts/verify_nuscenes.sh
```

For a fresh second environment:

```bash
METADATA_VENV=/opt/venvs/nuscenes-metadata-rebuild ./scripts/setup_metadata_env.sh
METADATA_VENV=/opt/venvs/nuscenes-metadata-rebuild \
  SMOKE_OUTPUT_DIR=/workspace/results/smoke_test_rebuild ./scripts/verify_nuscenes.sh
```

OS prerequisites are installed separately as documented in the root README.
The lock fixes Python packages, not a complete OS snapshot; future changes to
Ubuntu packages or registry tag contents remain an external reproducibility limit.
Keep these files in GitHub; the venv is disposable Pod state. Never reuse a future
BEVFusion CUDA/PyTorch environment for this check.
