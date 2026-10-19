# Camera–LiDAR Backfire

Research on camera–LiDAR fusion robustness under sensor misalignment, with a focus on identifying conditions under which sensor fusion may perform worse than an individual sensor.

**Current status: CONDITIONAL GO.** The official 850-scene nuScenes trainval metadata-only audit is complete. The recommendation includes class- and perturbation-specific restrictions and does not constitute a model-performance result.

BEVFusion deployment, MultiCorrupt integration, and model inference are separate future milestones.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Storage, Compute, and Repository Conventions](#storage-compute-and-repository-conventions)
- [Team Collaboration and Access Control](#team-collaboration-and-access-control)
- [Setup A: Linux on RunPod](#setup-a-linux-on-runpod)
  - [Administrator Bootstrap](#administrator-bootstrap)
  - [RunPod Dataset Acquisition](#runpod-dataset-acquisition)
  - [RunPod Pre-Audit Verification](#runpod-pre-audit-verification)
  - [Pod Replacement and Recovery](#pod-replacement-and-recovery)
- [Setup B: Windows with Anaconda and Conda](#setup-b-windows-with-anaconda-and-conda)
  - [Windows Environment Setup](#windows-environment-setup)
  - [Windows Dataset Acquisition](#windows-dataset-acquisition)
  - [Windows Pre-Audit Verification](#windows-pre-audit-verification)
- [Verified Checkpoints](#verified-checkpoints)
- [Full Metadata Audit Results](#full-metadata-audit-results)
- [Data Management and GitHub Policy](#data-management-and-github-policy)
- [Documentation and Reproduction](#documentation-and-reproduction)

---

## Project Overview

The project investigates the robustness of camera–LiDAR sensor fusion under spatial and temporal sensor misalignment.

The completed metadata audit provides an initial assessment of whether the dataset and its annotations support the proposed investigation. It does not establish that camera–LiDAR fusion outperforms either sensor individually or demonstrate that fusion backfires under any particular perturbation.

The next research milestones are separate from the completed audit:

1. Deploy and configure BEVFusion.
2. Integrate or configure MultiCorrupt for controlled corruption experiments.
3. Run inference under appropriate baseline and sensor-misalignment conditions.
4. Compare fused-model performance with camera-only and LiDAR-only baselines.
5. Evaluate class- and perturbation-specific failure conditions using the documented research gates.

## Storage, Compute, and Repository Conventions

GitHub is the source of truth for reviewed scripts, configuration, environment definitions, reproduction commands, and reports.

The project supports two separate development environments.

| Component | Linux on RunPod | Windows workstation |
|---|---|---|
| Compute | Disposable RunPod Pod | Local NVIDIA GPU workstation |
| Project checkout | Pod-local directory | Local project directory |
| Environment | Configured Linux Python environment | Conda environment `VisualComputing` |
| Dataset root | `/workspace/datasets/nuscenes` | `./datasets/nuscenes` |
| Persistent storage | RunPod Network Volume mounted at `/workspace` | Local workstation storage |
| Experiment results | `/workspace/results/` | `./results/` |
| Git remote | [Camera-LiDAR-backfire](https://github.com/Strange-Noah/Camera-LiDAR-backfire) | Same repository |

### RunPod Storage Model

The RunPod Network Volume is mounted at `/workspace`. The following directories are intended to survive Pod replacement:

- `/workspace/datasets/`
- `/workspace/checkpoints/`
- `/workspace/results/`
- `/workspace/cache/`
- `/workspace/staging/`

The canonical dataset root is:

```text
/workspace/datasets/nuscenes
```

It is not the nested `v1.0-mini` directory.

The Pod, its Pod-local repository checkout, and its local Python environment are disposable. Recreate them from the repository and environment definitions when necessary.

Keep replacement compute in the same datacenter as the existing Network Volume. All project Pods currently use the Network Volume in `EU-RO-1`.

### Windows Storage Model

The Windows workstation uses a local project directory as its working root. The canonical local dataset root is:

```text
./datasets/nuscenes
```

The Windows dataset and the RunPod Network Volume are separate storage locations. Files generated in one environment should not be assumed to exist in the other unless they are explicitly transferred or retrieved through an appropriate storage mechanism.

---

## Team Collaboration and Access Control

We use a RunPod **Team with Dev members**.

### Git and Development Workflow

- GitHub `main` contains reviewed code, environment definitions, and project conventions.
- Each member develops on a personal branch.
- Each member uses a separate disposable Pod and a Pod-local repository checkout.
- All project Pods mount the same Network Volume at `/workspace`.
- Never share a writable Git working tree or writable Python environment across Pods.

The Network Volume is shared storage, not a shared development environment.

### Billing

Dev members create their Pods in the **Team context**. Charges are deducted from the Team account, with member Cost Centers available for internal settlement. These charges do not automatically deduct from each member's personal RunPod balance.

Member invitations and Cost Center configuration still require separate verification.

### Dataset Protection

The dataset is canonical input, not an experiment output directory.

Dataset permissions are administrator-managed. Dev membership alone does not enforce read-only access to the dataset or protect the Network Volume from deletion.

The owner has confirmed Team conversion. Dataset permission modes and a Landlock-protected, non-root experiment login have been applied and tested on the maintenance Pod.

**Every new or restarted Pod must have the protected experiment login provisioned and tested before experiments begin.** Default root sessions are not protected by this arrangement.

Refer to [Dataset protection and per-Pod setup](docs/dataset_readonly.md) and the [Team workflow, billing, safety, and rollout checklist](docs/team_collaboration.md).

---

# Setup A: Linux on RunPod

This setup describes the Linux workflow for RunPod, including disposable compute, persistent dataset storage, and separate administrator and experiment-user responsibilities.

## Administrator Bootstrap

### Pod Requirements

Create a new Ubuntu 24.04 CPU Pod with the following configuration:

- Attach the existing project Network Volume at `/workspace` when creating the Pod.
- Use at least 4 vCPU and 16 GB RAM.
- Register your SSH public key before boot.
- Expose only `22/tcp` for SSH.
- Use the same datacenter, `EU-RO-1`, as the existing Network Volume.

These are administrator bootstrap instructions, not the member experiment login.

### Install System Packages and Initialize the Project

From a root shell inside the Pod:

```bash
apt-get update
apt-get install -y python3.12-venv git curl ca-certificates

git clone https://github.com/Strange-Noah/Camera-LiDAR-backfire.git /root/camera-lidar-backfire

cd /root/camera-lidar-backfire

./scripts/prepare_storage.sh
./scripts/setup_metadata_env.sh
```

Configuration is maintained in:

```text
configs/metadata.env
environment/metadata/
```

Environment variables can override configured paths. The setup script uses the committed full package lock when present. Refer to `environment/metadata/` for the environment definition and captured Python version.

The administrator checkout above is for bootstrap and maintenance. Members should use their own Pod-local checkout rather than sharing this working tree.

### Provision the Protected Experiment Login

Before running experiments:

1. Follow [Dataset protection and per-Pod setup](docs/dataset_readonly.md).
2. Provision the non-root experiment login and its Landlock restrictions.
3. Check the intended dataset access and writable output locations.
4. Verify that the protected login works as expected.

Do not run experiments from a default root shell and assume that the protections are active.

## RunPod Dataset Acquisition

Dataset acquisition is a one-time operation for a storage location that does not already contain the dataset.

Before acquisition, log in to the official nuScenes website and accept its terms yourself. The URL in the acquisition script was supplied by the user from that official page.

Run the acquisition script **on the Pod, not on the Mac**. Ensure that the executing account has the necessary write access to the staging and dataset directories.

From the project checkout:

```bash
python3.12 scripts/acquire_nuscenes_mini.py --persistent-root /workspace
```

The script:

1. Downloads only `v1.0-mini.tgz` to `/workspace/staging`.
2. Tests gzip integrity.
3. Validates archive paths.
4. Extracts the mini dataset.
5. Verifies each extracted file size against its archive entry.
6. Records SHA-256 and provenance in `dataset_manifest.json`.

The downloaded archive is retained. A locally calculated SHA-256 is a fingerprint of the downloaded file, not an official checksum.

**Important safeguards:**

- Acquisition refuses to overwrite an existing dataset.
- Do not rerun acquisition against a populated directory or volume.
- Verification commands do not download data.
- The mini acquisition is for bounded validation and does not replace the full trainval metadata used by the completed audit.

## RunPod Pre-Audit Verification

Run verification from the **protected member login**, after sourcing the configured `.local/member.env` file. Do not run this command from the administrator root bootstrap shell.

From the member's project checkout:

```bash
source .local/member.env

./scripts/verify_nuscenes.sh --render-once
```

The bounded verification checks that the official nuScenes devkit can load and resolve:

- One camera sample and one LiDAR sample.
- Sensor calibration.
- Ego pose and timestamps.
- A short annotation and instance chain.

The `--render-once` option optionally produces a camera/LiDAR projection.

Results are saved to:

```text
/workspace/results/smoke_test/
```

A `PASS` marker is written only after all checks succeed.

This is schema and file-access validation, not a statistical metadata audit or model-performance evaluation.

## Pod Replacement and Recovery

When replacing a Pod:

1. Recreate the disposable compute environment from the documented configuration.
2. Attach the existing Network Volume at `/workspace` in `EU-RO-1`.
3. Clone the repository into the new Pod's own working directory.
4. Recreate the required Python environment using the repository's setup scripts.
5. Reapply and verify dataset protection and the non-root experiment login.
6. Confirm the configured storage paths before resuming experiments.

Do not download a second copy of the dataset simply because the Pod has been replaced.

See [Storage and recovery](docs/storage_and_recovery.md) for volume expansion, Pod replacement, and recovery procedures.

`configs/runpod-create-pod.json` describes an MCP `create-pod` request shape for a possible future replacement. It is **not** an instruction to create another Pod now. Recheck current inventory, configuration, and pricing before using it.

---

# Setup B: Windows with Anaconda and Conda

This setup is intended for local development and bounded dataset verification on the Windows workstation using an NVIDIA GPU, including the RTX 5060.

Use Anaconda Prompt or Git Bash. Commands below assume the working directory is the repository root.

## Windows Environment Setup

### Clone the Repository

```bash
git clone https://github.com/Strange-Noah/Camera-LiDAR-backfire.git
cd Camera-LiDAR-backfire
```

### Create and Activate the Conda Environment

```bash
conda create -y -n VisualComputing python=3.12
conda activate VisualComputing
```

### Install PyTorch with GPU Support

The project's documented installation command uses the CUDA 12.8 (`cu128`) PyTorch package index:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then install the project dependencies:

```bash
pip install -r environment/metadata/requirements.lock.txt
```

Keep the environment aligned with the repository's dependency definitions. If changing Python, PyTorch, or CUDA-related packages becomes necessary, document the change rather than silently diverging from the environment lock.

The RunPod metadata environment and the Windows Conda environment are independent. Do not try to share an installed Python environment across Windows and Linux.

## Windows Dataset Acquisition

The local Windows dataset is separate from the RunPod Network Volume.

If you intentionally need a local copy of the nuScenes mini dataset, first log in to the official nuScenes website and accept its terms yourself. Then run the acquisition script from the active Conda environment and the repository root:

```bash
python scripts/acquire_nuscenes_mini.py --persistent-root .
```

This downloads the mini archive to the local staging directory and places the resulting dataset under the configured local dataset path.

The acquisition script validates the archive, verifies extracted file sizes, and records a local SHA-256 fingerprint and provenance in `dataset_manifest.json`.

**Do not rerun acquisition if the target dataset already exists.** The script refuses to overwrite an existing dataset.

The mini dataset is intended for bounded verification. It should not be treated as a substitute for the official 850-scene trainval metadata used in the completed research audit.

## Windows Pre-Audit Verification

Run from the active `VisualComputing` Conda environment and the repository root:

```bash
python scripts/verify_nuscenes.py --root ./datasets/nuscenes --output ./results/smoke_test --render-once
```

The bounded check uses the official nuScenes devkit to resolve a camera and LiDAR sample, calibration, ego pose, timestamps, and a short annotation/instance chain.

The `--render-once` option optionally creates one camera/LiDAR projection.

Results are saved to:

```text
./results/smoke_test/
```

A `PASS` marker is written only after all checks succeed.

This verification confirms basic schema compatibility and file access. It does not reproduce the full metadata audit or establish model-performance results.

---

## Verified Checkpoints

### Earlier Pre-Audit Checkpoint

Both isolated environments successfully loaded:

| Metric | Result |
|---|---:|
| Scenes | 10 |
| Samples | 404 |
| Annotations | 18,538 |

This describes the earlier bounded, pre-audit checkpoint. It is not the result of the subsequent full metadata audit.

No model deployment was performed as part of this checkpoint.

## Full Metadata Audit Results

The official 850-scene trainval metadata-only audit is complete.

| Metric | Result |
|---|---:|
| Scenes | 850 |
| Samples | 34,149 |
| Trajectories | 64,386 |
| Annotations | 1,166,187 |
| Reviewed recommendation | **CONDITIONAL GO** |

The recommendation includes class- and perturbation-specific restrictions. It indicates that the proposed research can proceed conditionally according to the documented gates; it does not demonstrate that camera–LiDAR fusion backfires or establish the performance of any deployed model.

The audit required neither trainval sensor blobs nor GPU deployment, BEVFusion, MultiCorrupt, or model inference.

---

## Data Management and GitHub Policy

GitHub is the source of truth for reviewed project code and reproducibility information.

### Files That May Be Committed

Examples include:

- Scripts and configuration.
- Environment definitions and lockfiles.
- Documentation and reproduction commands.
- Small verification reports.
- Small derived metadata-audit summaries and figures.
- Research methods, validity rules, and gating reports.

### Files That Must Remain Outside Git

Do not commit:

- Downloaded datasets or trainval sensor blobs.
- Model weights and checkpoints.
- Credentials, tokens, or other secrets.
- Large prediction outputs.
- Large row-level Parquet files and trajectory tables.

The row-level Parquet files and large trajectory tables from the full metadata audit remain on the persistent RunPod volume at:

```text
/workspace/results/metadata_audit/
```

Only small derived summaries and figures are versioned in the repository.

For future experiments, keep the raw dataset as canonical input and store generated outputs under the appropriate results or checkpoint directory. Preserve the existing dataset rather than using it as a destination for experiment outputs.

---

## Documentation and Reproduction

The following repository files document the workflow and completed audit.

| Document | Purpose |
|---|---|
| [Dataset protection and per-Pod setup](docs/dataset_readonly.md) | Dataset permissions, Landlock protection, and protected experiment login setup |
| [Team workflow, billing, safety, and rollout checklist](docs/team_collaboration.md) | Shared Network Volume conventions, Pod isolation, billing, and rollout verification |
| [English message for teammates](docs/team_invitation.md) | Team onboarding message |
| [Personal environment example](configs/member.env.example) | Example member-specific environment configuration |
| [Storage and recovery](docs/storage_and_recovery.md) | Persistent storage, volume expansion, and Pod replacement |
| [Pre-audit deployment report](docs/pre_audit_deployment.md) | Earlier acceptance checklist, resource IDs, reproducibility proof, and known limitations |
| [Research report and supported/unsupported claims](results/metadata_audit/README.md) | Audit findings, interpretation, and claim limitations |
| [Machine-readable gates](results/metadata_audit/gating_report.json) | Machine-readable research gates |
| [Methods, validity rules, and reproduction commands](docs/metadata_audit_methods.md) | Audit methodology and reproduction instructions |

The pre-audit deployment report describes the earlier checkpoint and should not be mistaken for evidence of model deployment. The metadata-audit report and its gates document the completed 850-scene audit.

**Current research milestone:** The metadata audit is complete. BEVFusion/MultiCorrupt setup, deployment, corruption experiments, inference, and model-performance comparisons remain future work.