# Camera–LiDAR backfire

Research on camera–LiDAR fusion robustness under sensor misalignment.
This milestone prepares Runpod and official nuScenes mini for a later metadata audit.
It does not deploy BEVFusion/MultiCorrupt or run an audit or inference.

## Storage and compute

- GitHub is the source of truth for scripts, configuration, environment locks, and reports.
- A Runpod Network Volume is mounted at `/workspace`; dataset root is
  `/workspace/datasets/nuscenes` (not its `v1.0-mini` subdirectory).
- The Pod, `/root/camera-lidar-backfire` checkout, and `/opt/venvs/nuscenes-metadata`
  environment are disposable and recreated from this repository.
- `/workspace/{datasets,checkpoints,results,cache,staging}` survives Pod replacement.
  Keep the volume in the same datacenter as replacement compute.

## Prepare a new Ubuntu 24.04 CPU Pod

Attach the project Network Volume at `/workspace` at creation. Use at least
4 vCPU and 16 GB RAM. Register your SSH public key before boot; expose only 22/tcp.
From a root shell inside the Pod:

```bash
apt-get update
apt-get install -y python3.12-venv git curl ca-certificates
git clone https://github.com/Strange-Noah/Camera-LiDAR-backfire.git /root/camera-lidar-backfire
cd /root/camera-lidar-backfire
./scripts/prepare_storage.sh
./scripts/setup_metadata_env.sh
```

Configuration is in `configs/metadata.env`; environment variables can override paths.
The setup script uses the committed full package lock when present. See
`environment/metadata/` for the environment definition and captured Python version.

## Deliberate dataset acquisition (once per volume)

Log in to the official nuScenes website and accept its terms yourself first.
The URL in the acquisition script was supplied by the user from that official page.
Run this **on the Pod**, never on the Mac:

```bash
python3.12 scripts/acquire_nuscenes_mini.py --persistent-root /workspace
```

This downloads only `v1.0-mini.tgz` to `/workspace/staging`, tests gzip integrity,
validates archive paths, extracts mini, verifies every extracted file size against
its archive entry, and records SHA-256 and provenance in `dataset_manifest.json`.
The archive is retained. A local SHA-256 is a fingerprint, not an official checksum.
Acquisition refuses to overwrite an existing dataset. Do not rerun it on a populated
volume; verification never downloads data.

## Pre-audit verification

```bash
./scripts/verify_nuscenes.sh --render-once
```

The bounded check loads the official devkit, resolves one camera and LiDAR sample,
calibration, ego pose and timestamps, and a short annotation/instance chain.
`--render-once` optionally creates one camera/LiDAR projection. Results are saved to
`/workspace/results/smoke_test`. A `PASS` marker is written only after all checks.
This is schema and file-access validation, not a statistical metadata audit.

No datasets, model weights, credentials, or large predictions belong in Git.
Small verification reports and environment locks may be committed.

See [storage and recovery](docs/storage_and_recovery.md) for volume expansion and
Pod replacement. `configs/runpod-create-pod.json` is an MCP `create-pod` request
shape for a future replacement, not an instruction to create another Pod now.
Re-check inventory and pricing before reusing it.

## Verified checkpoint

See [pre-audit deployment report](docs/pre_audit_deployment.md) for the completed
acceptance checklist, live resource IDs, reproducibility proof, and known limits.
Both isolated environments loaded 10 scenes / 404 samples / 18,538 annotations.
No formal metadata audit or model deployment has been performed.

## Full metadata audit checkpoint

The official 850-scene trainval metadata-only audit is complete: 34,149 samples, 64,386 trajectories and 1,166,187 annotations. The reviewed recommendation is **CONDITIONAL GO**, with class/perturbation-specific restrictions; this is not a model-performance result.

- [Research report and supported/unsupported claims](results/metadata_audit/README.md)
- [Machine-readable gates](results/metadata_audit/gating_report.json)
- [Methods, validity rules and reproduction commands](docs/metadata_audit_methods.md)

Row-level Parquet and large trajectory tables remain on the persistent volume at `/workspace/results/metadata_audit/`. Only small derived summaries and figures are versioned here. No trainval sensor blobs, GPU deployment, BEVFusion or corruption/inference were needed.
