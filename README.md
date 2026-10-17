# Camera–LiDAR backfire

Research on camera–LiDAR fusion robustness under sensor misalignment.
The 850-scene nuScenes metadata audit is complete (CONDITIONAL GO).
BEVFusion/MultiCorrupt deployment and inference are separate, future milestones.

## Storage and compute

- GitHub is the source of truth for scripts, configuration, environment locks, and reports.
- A Runpod Network Volume is mounted at `/workspace`; dataset root is
  `/workspace/datasets/nuscenes` (not its `v1.0-mini` subdirectory).
- The Pod, `/root/camera-lidar-backfire` checkout, and `/opt/venvs/nuscenes-metadata`
  environment are disposable and recreated from this repository.
- `/workspace/{datasets,checkpoints,results,cache,staging}` survives Pod replacement.
  Keep the volume in the same datacenter as replacement compute.

## Team collaboration

We use a Runpod **Team with Dev members**: GitHub `main` holds reviewed code,
environment definitions and project conventions; each member develops on a personal
branch and a separate disposable Pod. All project Pods use the same Network Volume
in EU-RO-1, mounted at `/workspace`. Clone the repository into your Pod-local home;
never share a Git working tree or writable Python environment across Pods.

Dev members create their own Pods **in the Team context**. Charges are deducted from
the Team account, with member Cost Centers for internal settlement; this does not
charge each member's personal Runpod balance.

The dataset is canonical input, not an output directory. Use non-root experiment
users and administrator-managed filesystem permissions; Dev membership alone does
not enforce read-only data and still permits deleting the entire volume.

**Rollout status:** the owner confirmed Team conversion. Dataset modes and a
Landlock-protected non-root experiment login have been applied and tested on the
maintenance Pod. Every new/restarted Pod must provision that protected login before
experiments; default root sessions are not protected. Member invitations and Cost
Centers still require separate verification. The existing volume is reused in place.
See [dataset protection and per-Pod setup](docs/dataset_readonly.md).

- [Team workflow, billing, safety and rollout checklist](docs/team_collaboration.md)
- [English message for teammates](docs/team_invitation.md)
- [Personal environment example](configs/member.env.example)

## Administrator bootstrap: new Ubuntu 24.04 CPU Pod

Attach the project Network Volume at `/workspace` at creation. Use at least
4 vCPU and 16 GB RAM. Register your SSH public key before boot; expose only 22/tcp.
These are administrator bootstrap commands, not the member experiment login.
For member checkouts and non-root environments, follow the team workflow above.
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
This describes the earlier pre-audit checkpoint; the subsequent full metadata audit
is documented below. No model deployment has been performed.

## Full metadata audit checkpoint

The official 850-scene trainval metadata-only audit is complete: 34,149 samples, 64,386 trajectories and 1,166,187 annotations. The reviewed recommendation is **CONDITIONAL GO**, with class/perturbation-specific restrictions; this is not a model-performance result.

- [Research report and supported/unsupported claims](results/metadata_audit/README.md)
- [Machine-readable gates](results/metadata_audit/gating_report.json)
- [Methods, validity rules and reproduction commands](docs/metadata_audit_methods.md)

Row-level Parquet and large trajectory tables remain on the persistent volume at `/workspace/results/metadata_audit/`. Only small derived summaries and figures are versioned here. No trainval sensor blobs, GPU deployment, BEVFusion or corruption/inference were needed.
